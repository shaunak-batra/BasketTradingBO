#!/usr/bin/env python3
"""
Main pipeline script for complete basket trading workflow.

Integrates all phases:
1. Data acquisition
2. Cointegration testing
3. Signal generation
4. Backtesting
5. Bayesian optimization
6. Risk analysis
7. Report generation

Usage:
    python scripts/run_pipeline.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2023-12-31
    python scripts/run_pipeline.py --config config/config.yaml --optimize --generate-report


"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtester import Backtester
from src.cointegration.engine import CointegrationEngine
from src.cointegration.spread import SpreadCalculator
from src.data.market_data import MarketDataAdapter
from src.optimization.optimizer import BayesianOptimizer
from src.risk.manager import RiskManager
from src.strategy.filters import apply_combined_filters
from src.strategy.signals import SignalGenerator
from src.utils.io import save_json
from src.utils.logger import StructuredLogger
from src.visualization.plots import (
    plot_correlation_matrix,
    plot_optimization_convergence,
    plot_var_distribution,
)
from src.visualization.reports import generate_backtest_report


def run_complete_pipeline(
    tickers: List[str],
    start_date: str,
    end_date: str,
    optimize: bool = False,
    n_optimization_iterations: int = 50,
    generate_report: bool = True,
    output_dir: str = "results",
) -> None:
    """
    Run complete basket trading pipeline.

    Parameters
    ----------
    tickers : List[str]
        Asset tickers
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    optimize : bool
        Run Bayesian optimization
    n_optimization_iterations : int
        Number of optimization iterations
    generate_report : bool
        Generate HTML report
    output_dir : str
        Output directory for results
    """
    logger = StructuredLogger(__name__)
    logger.info(
        "=" * 60 + "\n" + "BASKET TRADING PIPELINE - STARTED\n" + "=" * 60,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # Create organized output directory structure
    # Format: results/{TICKER1_TICKER2_TICKER3}/{YYYYMMDD_HHMMSS}/
    ticker_str = "_".join(tickers)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(output_dir) / ticker_str / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # Create plots subdirectory
    plots_path = output_path / "plots"
    plots_path.mkdir(exist_ok=True)

    logger.info(f"Output directory created: {output_path}")

    # ============================================================
    # PHASE 1: DATA ACQUISITION
    # ============================================================
    logger.info("\n" + "=" * 60 + "\nPHASE 1: DATA ACQUISITION\n" + "=" * 60)

    adapter = MarketDataAdapter()
    prices = adapter.fetch_data(tickers, start_date, end_date)

    if prices.empty:
        logger.error("No data fetched. Pipeline aborted.")
        return

    logger.info(
        "Data fetched successfully", shape=prices.shape, tickers=list(prices.columns)
    )

    # Save correlation matrix
    plot_correlation_matrix(
        prices, save_path=str(output_path / "correlation_matrix.png")
    )

    # ============================================================
    # PHASE 2: COINTEGRATION ANALYSIS
    # ============================================================
    logger.info("\n" + "=" * 60 + "\nPHASE 2: COINTEGRATION ANALYSIS\n" + "=" * 60)

    coint_engine = CointegrationEngine()
    coint_result = None  # Initialize to None

    try:
        coint_result = coint_engine.test_cointegration(prices)

        if not coint_result.is_cointegrated:
            logger.warning(
                "Assets are NOT cointegrated!",
                test_statistic=float(coint_result.test_statistic),
                critical_value=float(coint_result.critical_values.get("5%", 0)),
            )
            logger.warning(
                "Continuing pipeline for demonstration, but results may not be meaningful."
            )
        else:
            logger.info(
                "Assets are cointegrated",
                rank=coint_result.cointegrating_rank,
                test_statistic=float(coint_result.test_statistic),
            )

        # Use first eigenvector
        weights = coint_result.eigenvectors[:, 0]

    except Exception as e:
        logger.warning(f"Cointegration test failed: {e}", error_type=type(e).__name__)
        logger.warning("Using equal weights instead for demonstration purposes.")
        # Use equal weights as fallback
        weights = np.ones(len(tickers)) / len(tickers)
        weights[0] = 1.0  # Make first asset the dependent variable
        for i in range(1, len(weights)):
            weights[i] = -1.0 / (len(weights) - 1)  # Spread other assets equally

        # Set coint_result to None if test failed
        coint_result = None

    logger.info(
        "Basket weights (cointegrating vector)",
        weights=dict(zip(tickers, weights.tolist())),
    )

    # ============================================================
    # PHASE 3: SPREAD ANALYSIS
    # ============================================================
    logger.info("\n" + "=" * 60 + "\nPHASE 3: SPREAD ANALYSIS\n" + "=" * 60)

    spread_calc = SpreadCalculator()
    spread = spread_calc.calculate_spread(prices, weights)
    zscore = spread_calc.calculate_zscore(spread, lookback=252)

    try:
        half_life = spread_calc.calculate_half_life(spread)
        logger.info("Spread half-life", half_life_days=round(half_life, 2))
    except ValueError as e:
        logger.warning(f"Could not calculate half-life: {e}")
        half_life = None

    # Calculate Hurst exponent as proxy for stationarity
    try:
        hurst = spread_calc.calculate_hurst_exponent(spread)
        is_stationary = hurst < 0.5  # Mean reverting if H < 0.5
        logger.info(
            "Spread Hurst exponent",
            hurst=round(hurst, 4),
            is_mean_reverting=is_stationary,
        )
    except Exception as e:
        logger.warning(f"Could not calculate Hurst exponent: {e}")
        is_stationary = False

    # ============================================================
    # PHASE 4: PARAMETER OPTIMIZATION (OPTIONAL)
    # ============================================================
    if optimize:
        logger.info("\n" + "=" * 60 + "\nPHASE 4: BAYESIAN OPTIMIZATION\n" + "=" * 60)

        def objective(params):
            """Objective function for optimization."""
            entry_threshold = params.get("entry_threshold", 2.0)
            exit_threshold = params.get("exit_threshold", 0.5)
            stop_loss = params.get("stop_loss", 4.0)
            lookback_window = int(params.get("lookback_window", 252))
            position_size = params.get("position_size", 0.20)
            min_holding_period = int(params.get("min_holding_period", 5))

            try:
                # Recalculate z-score with new lookback
                zscore_opt = spread_calc.calculate_zscore(
                    spread, lookback=lookback_window
                )

                # Generate signals
                signal_gen = SignalGenerator(
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    stop_loss=stop_loss,
                )
                signals = signal_gen.generate_signals(zscore_opt)
                filtered_signals = apply_combined_filters(
                    signals, min_holding_period=min_holding_period
                )

                # Backtest
                backtester = Backtester(
                    initial_capital=100000.0,
                    commission=0.001,
                    slippage=0.0005,
                    position_size=position_size,
                )
                result = backtester.run_backtest(prices, filtered_signals, weights)

                # Return negative Sharpe ratio (to minimize)
                return -float(result.metrics.sharpe_ratio)

            except Exception as e:
                logger.error("Optimization objective error", error=str(e))
                return 1e6

        parameter_space = {
            "entry_threshold": (1.5, 3.0),
            "exit_threshold": (0.2, 1.0),
            "stop_loss": (3.0, 5.0),
            "lookback_window": (60.0, 504.0),
            "position_size": (0.10, 0.30),
            "min_holding_period": (3.0, 10.0),
        }

        optimizer = BayesianOptimizer(
            objective_function=objective,
            parameter_space=parameter_space,
            n_initial_points=10,
            n_iterations=n_optimization_iterations,
            random_state=42,
            verbose=True,
        )

        opt_result = optimizer.optimize()

        logger.info(
            "Optimization completed",
            best_sharpe=round(-opt_result.best_score, 4),
            best_params=opt_result.best_params,
        )

        # Use optimized parameters
        entry_threshold = opt_result.best_params["entry_threshold"]
        exit_threshold = opt_result.best_params["exit_threshold"]
        stop_loss = opt_result.best_params["stop_loss"]
        lookback_window = int(opt_result.best_params["lookback_window"])
        position_size = opt_result.best_params["position_size"]
        min_holding_period = int(opt_result.best_params["min_holding_period"])

        # Plot convergence
        plot_optimization_convergence(
            [-s for s in opt_result.convergence_history],
            parameter_name="Sharpe Ratio",
            save_path=str(output_path / "optimization_convergence.png"),
        )

        # Recalculate z-score with optimized lookback
        zscore = spread_calc.calculate_zscore(spread, lookback=lookback_window)

    else:
        # Use default parameters
        logger.info(
            "\n" + "=" * 60 + "\nPHASE 4: USING DEFAULT PARAMETERS\n" + "=" * 60
        )
        entry_threshold = 2.0
        exit_threshold = 0.5
        stop_loss = 4.0
        lookback_window = 252
        position_size = 0.20
        min_holding_period = 5

    # ============================================================
    # PHASE 5: SIGNAL GENERATION
    # ============================================================
    logger.info("\n" + "=" * 60 + "\nPHASE 5: SIGNAL GENERATION\n" + "=" * 60)

    signal_gen = SignalGenerator(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
    )
    signals = signal_gen.generate_signals(zscore)
    filtered_signals = apply_combined_filters(
        signals, min_holding_period=min_holding_period
    )

    logger.info(
        "Signals generated",
        total_signals=len(signals),
        long_signals=(signals == 1).sum(),
        short_signals=(signals == -1).sum(),
        filtered_signals=(filtered_signals != signals).sum(),
    )

    # ============================================================
    # PHASE 6: BACKTESTING
    # ============================================================
    logger.info("\n" + "=" * 60 + "\nPHASE 6: BACKTESTING\n" + "=" * 60)

    backtester = Backtester(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005,
        position_size=position_size,
    )

    result = backtester.run_backtest(prices, filtered_signals, weights)

    logger.info(
        "Backtest completed",
        final_value=round(float(result.portfolio_value.iloc[-1]), 2),
        total_return_pct=round(result.metrics.total_return * 100, 2),
        sharpe_ratio=round(result.metrics.sharpe_ratio, 2),
        max_drawdown_pct=round(result.metrics.max_drawdown * 100, 2),
        num_trades=result.metrics.num_trades,
    )

    # ============================================================
    # PHASE 7: RISK ANALYSIS
    # ============================================================
    logger.info("\n" + "=" * 60 + "\nPHASE 7: RISK ANALYSIS\n" + "=" * 60)

    risk_manager = RiskManager(
        max_position_size=0.30, max_portfolio_var=0.02, var_confidence_level=0.95
    )

    var_95 = risk_manager.calculate_var(
        result.returns, method="historical", confidence_level=0.95
    )
    var_99 = risk_manager.calculate_var(
        result.returns, method="historical", confidence_level=0.99
    )

    logger.info(
        "VaR Analysis",
        var_95=round(var_95.var, 4),
        var_95_es=(
            round(var_95.expected_shortfall, 4) if var_95.expected_shortfall else None
        ),
        var_99=round(var_99.var, 4),
        var_99_es=(
            round(var_99.expected_shortfall, 4) if var_99.expected_shortfall else None
        ),
    )

    # Plot VaR distribution
    plot_var_distribution(
        result.returns,
        var_95=var_95.var,
        var_99=var_99.var,
        save_path=str(output_path / "var_distribution.png"),
    )

    # ============================================================
    # PHASE 8: SAVE RESULTS
    # ============================================================
    logger.info("\n" + "=" * 60 + "\nPHASE 8: SAVING RESULTS\n" + "=" * 60)

    results_dict = {
        "pipeline_parameters": {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "optimized": optimize,
        },
        "strategy_parameters": {
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "stop_loss": stop_loss,
            "lookback_window": lookback_window,
            "position_size": position_size,
            "min_holding_period": min_holding_period,
        },
        "cointegration": {
            "is_cointegrated": (
                bool(coint_result.is_cointegrated) if coint_result else False
            ),
            "rank": int(coint_result.cointegrating_rank) if coint_result else 0,
            "weights": weights.tolist(),
            "half_life_days": float(half_life) if half_life else None,
            "is_stationary": bool(is_stationary),
        },
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
        "risk_metrics": {
            "var_95": float(var_95.var),
            "var_95_es": (
                float(var_95.expected_shortfall) if var_95.expected_shortfall else None
            ),
            "var_99": float(var_99.var),
            "var_99_es": (
                float(var_99.expected_shortfall) if var_99.expected_shortfall else None
            ),
        },
        "transaction_costs": float(result.transaction_costs.sum()),
    }

    save_json(results_dict, str(output_path / "pipeline_results.json"))
    logger.info("Results saved", path=str(output_path / "pipeline_results.json"))

    # ============================================================
    # PHASE 9: GENERATE REPORT
    # ============================================================
    if generate_report:
        logger.info("\n" + "=" * 60 + "\nPHASE 9: GENERATING REPORT\n" + "=" * 60)

        report_path = generate_backtest_report(
            result=result,
            spread=spread,
            zscore=zscore,
            parameters={
                "tickers": tickers,
                "start_date": start_date,
                "end_date": end_date,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "stop_loss": stop_loss,
                "lookback_window": lookback_window,
                "position_size": position_size,
                "min_holding_period": min_holding_period,
            },
            output_dir=str(plots_path),  # Save plots in the plots/ subfolder
            use_timestamp=False,  # Don't create timestamped folders, use organized structure
        )

        logger.info("HTML report generated", path=report_path)

    # ============================================================
    # PIPELINE COMPLETE
    # ============================================================
    logger.info(
        "\n"
        + "=" * 60
        + "\n"
        + "BASKET TRADING PIPELINE - COMPLETED\n"
        + "=" * 60
        + "\n"
        + f"Results saved to: {output_path}\n"
        + f"\nQuick Access:\n"
        + f"  - View Report:  start {output_path / 'backtest_report.html'}\n"
        + f"  - View Metrics: cat {output_path / 'pipeline_results.json'}\n"
        + f"  - View Plots:   explorer {plots_path}\n"
        + "=" * 60
    )


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(description="Run complete basket trading pipeline")

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
        "--optimize",
        action="store_true",
        help="Run Bayesian optimization for parameters",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=50,
        help="Number of optimization iterations (default: 50)",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip HTML report generation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.tickers or len(args.tickers) < 2:
        parser.error("Must provide at least 2 tickers for cointegration analysis")

    # Validate date format
    try:
        start_date = pd.to_datetime(args.start)
        end_date = pd.to_datetime(args.end)
    except:
        parser.error("Invalid date format. Use YYYY-MM-DD (e.g., 2023-01-01)")

    # Validate date logic
    if start_date >= end_date:
        parser.error("start_date must be before end_date")

    # Check date range is reasonable
    date_diff = (end_date - start_date).days
    if date_diff < 30:
        parser.error(
            f"Date range too short ({date_diff} days). Need at least 30 days of data."
        )

    # Validate optimization iterations
    if args.optimize and args.n_iterations < 1:
        parser.error("n_iterations must be at least 1")

    # Convert output_dir to absolute path
    output_dir = Path(args.output_dir).resolve()

    run_complete_pipeline(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        optimize=args.optimize,
        n_optimization_iterations=args.n_iterations,
        generate_report=not args.no_report,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
