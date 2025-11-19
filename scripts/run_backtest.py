#!/usr/bin/env python3
"""
CLI script to run backtests on basket trading strategies.

Usage:
    python scripts/run_backtest.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2023-12-31
    python scripts/run_backtest.py --config config/config.yaml --output results/backtest_results.json

"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtester import Backtester
from src.cointegration.engine import CointegrationEngine
from src.cointegration.spread import SpreadCalculator
from src.data.market_data import MarketDataAdapter
from src.strategy.filters import apply_combined_filters
from src.strategy.signals import SignalGenerator
from src.utils.config import ConfigManager
from src.utils.io import save_json
from src.utils.logger import StructuredLogger


def run_backtest(
    tickers: List[str],
    start_date: str,
    end_date: str,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 4.0,
    position_size: float = 0.20,
    min_holding_period: int = 5,
    output_path: str = None,
) -> None:
    """
    Run complete backtest pipeline.

    Parameters
    ----------
    tickers : List[str]
        Asset tickers
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    entry_threshold : float
        Signal entry threshold
    exit_threshold : float
        Signal exit threshold
    stop_loss : float
        Stop loss threshold
    position_size : float
        Position size as fraction of portfolio
    min_holding_period : int
        Minimum holding period (days)
    output_path : str
        Path to save results (JSON)
    """
    logger = StructuredLogger(__name__)
    logger.info(
        "Starting backtest pipeline",
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # Step 1: Fetch market data
    logger.info("Fetching market data")
    adapter = MarketDataAdapter()
    prices = adapter.fetch_data(tickers, start_date, end_date)

    if prices.empty:
        logger.error("No data fetched. Exiting.")
        return

    logger.info("Data fetched", shape=prices.shape)

    # Step 2: Test cointegration
    logger.info("Testing cointegration")
    coint_engine = CointegrationEngine()
    coint_result = coint_engine.test_cointegration(prices)

    if not coint_result.is_cointegrated:
        logger.warning(
            "Assets are NOT cointegrated. Backtest may not be meaningful.",
            test_statistic=float(coint_result.test_statistic),
            critical_value=float(coint_result.critical_value),
        )
    else:
        logger.info("Assets are cointegrated", rank=coint_result.cointegration_rank)

    # Use first eigenvector as basket weights
    weights = coint_result.eigenvectors[:, 0]
    logger.info("Basket weights", weights=weights.tolist())

    # Step 3: Calculate spread and z-score
    logger.info("Calculating spread")
    spread_calc = SpreadCalculator()
    spread = spread_calc.create_spread(prices, weights)
    zscore = spread_calc.calculate_zscore(spread, lookback=252)

    # Calculate spread statistics
    half_life = spread_calc.calculate_half_life(spread)
    logger.info("Spread statistics", half_life=round(half_life, 2))

    # Step 4: Generate signals
    logger.info("Generating signals")
    signal_gen = SignalGenerator(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
    )
    signals = signal_gen.generate_signals(zscore)

    # Apply filters
    filtered_signals = apply_combined_filters(
        signals, min_holding_period=min_holding_period
    )

    # Step 5: Run backtest
    logger.info("Running backtest")
    backtester = Backtester(
        initial_capital=100000.0,
        commission=0.001,  # 10 bps
        slippage=0.0005,  # 5 bps
        position_size=position_size,
    )

    result = backtester.run_backtest(prices, filtered_signals, weights)

    # Step 6: Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital:      ${result.portfolio_value.iloc[0]:,.2f}")
    print(f"Final Value:          ${result.portfolio_value.iloc[-1]:,.2f}")
    print(f"Total Return:         {result.metrics.total_return*100:.2f}%")
    print(f"Annualized Return:    {result.metrics.annualized_return*100:.2f}%")
    print(f"Sharpe Ratio:         {result.metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:        {result.metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown:         {result.metrics.max_drawdown*100:.2f}%")
    print(f"Calmar Ratio:         {result.metrics.calmar_ratio:.2f}")
    print(f"Win Rate:             {result.metrics.win_rate*100:.2f}%")
    print(f"Profit Factor:        {result.metrics.profit_factor:.2f}")
    print(f"Number of Trades:     {result.metrics.num_trades}")
    print(f"Transaction Costs:    ${result.transaction_costs.sum():,.2f}")
    print("=" * 60 + "\n")

    # Step 7: Save results
    if output_path:
        logger.info("Saving results", output_path=output_path)

        results_dict = {
            "backtest_parameters": {
                "tickers": tickers,
                "start_date": start_date,
                "end_date": end_date,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "stop_loss": stop_loss,
                "position_size": position_size,
                "min_holding_period": min_holding_period,
            },
            "cointegration": {
                "is_cointegrated": bool(coint_result.is_cointegrated),
                "rank": int(coint_result.cointegration_rank),
                "test_statistic": float(coint_result.test_statistic),
                "weights": weights.tolist(),
            },
            "spread_statistics": {"half_life": float(half_life)},
            "performance_metrics": {
                "total_return": float(result.metrics.total_return),
                "annualized_return": float(result.metrics.annualized_return),
                "sharpe_ratio": float(result.metrics.sharpe_ratio),
                "sortino_ratio": float(result.metrics.sortino_ratio),
                "max_drawdown": float(result.metrics.max_drawdown),
                "calmar_ratio": float(result.metrics.calmar_ratio),
                "win_rate": float(result.metrics.win_rate),
                "profit_factor": float(result.metrics.profit_factor),
                "num_trades": int(result.metrics.num_trades),
            },
            "transaction_costs": float(result.transaction_costs.sum()),
        }

        save_json(results_dict, output_path)
        logger.info("Results saved successfully")

    logger.info("Backtest pipeline completed")


def main():
    """Parse arguments and run backtest."""
    parser = argparse.ArgumentParser(
        description="Run backtest on basket trading strategy"
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Asset tickers (e.g., AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--start", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=2.0,
        help="Signal entry threshold (default: 2.0)",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.5,
        help="Signal exit threshold (default: 0.5)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=4.0,
        help="Stop loss threshold (default: 4.0)",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.20,
        help="Position size as fraction of portfolio (default: 0.20)",
    )
    parser.add_argument(
        "--min-holding-period",
        type=int,
        default=5,
        help="Minimum holding period in days (default: 5)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save results JSON (optional)"
    )

    args = parser.parse_args()

    run_backtest(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_loss=args.stop_loss,
        position_size=args.position_size,
        min_holding_period=args.min_holding_period,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
