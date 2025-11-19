#!/usr/bin/env python3
"""
CLI script to run Bayesian optimization for strategy parameters.

Usage:
    python scripts/run_optimization.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2023-12-31
    python scripts/run_optimization.py --config config/config.yaml --output results/optimization_results.json

"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtester import Backtester
from src.cointegration.engine import CointegrationEngine
from src.cointegration.spread import SpreadCalculator
from src.data.market_data import MarketDataAdapter
from src.optimization.optimizer import BayesianOptimizer
from src.strategy.filters import apply_combined_filters
from src.strategy.signals import SignalGenerator
from src.utils.io import save_json
from src.utils.logger import StructuredLogger


def create_objective_function(
    prices: pd.DataFrame, weights: List[float], metric: str = "sharpe_ratio"
):
    """
    Create objective function for Bayesian optimization.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    weights : List[float]
        Basket weights
    metric : str
        Metric to optimize ("sharpe_ratio", "sortino_ratio", "calmar_ratio")

    Returns
    -------
    Callable
        Objective function that takes parameters and returns score to minimize
    """
    logger = StructuredLogger(__name__)

    def objective(params: Dict[str, float]) -> float:
        """
        Objective function to minimize (negative metric).

        Parameters
        ----------
        params : Dict[str, float]
            Strategy parameters to evaluate

        Returns
        -------
        float
            Negative metric value (to minimize)
        """
        try:
            # Extract parameters
            entry_threshold = params.get("entry_threshold", 2.0)
            exit_threshold = params.get("exit_threshold", 0.5)
            stop_loss = params.get("stop_loss", 4.0)
            lookback_window = int(params.get("lookback_window", 252))
            position_size = params.get("position_size", 0.20)
            min_holding_period = int(params.get("min_holding_period", 5))

            # Calculate spread and z-score
            spread_calc = SpreadCalculator()
            spread = spread_calc.create_spread(prices, weights)
            zscore = spread_calc.calculate_zscore(spread, lookback=lookback_window)

            # Generate signals
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

            # Run backtest
            backtester = Backtester(
                initial_capital=100000.0,
                commission=0.001,
                slippage=0.0005,
                position_size=position_size,
            )

            result = backtester.run_backtest(prices, filtered_signals, weights)

            # Extract metric
            metric_value = getattr(result.metrics, metric)

            # Return negative (we minimize)
            score = -float(metric_value)

            logger.info(
                "Objective evaluated",
                params=params,
                metric=metric,
                value=round(metric_value, 4),
                score=round(score, 4),
            )

            return score

        except Exception as e:
            logger.error("Objective function error", error=str(e), params=params)
            # Return large penalty on error
            return 1e6

    return objective


def run_optimization(
    tickers: List[str],
    start_date: str,
    end_date: str,
    metric: str = "sharpe_ratio",
    n_iterations: int = 50,
    n_initial_points: int = 10,
    output_path: str = None,
) -> None:
    """
    Run Bayesian optimization for strategy parameters.

    Parameters
    ----------
    tickers : List[str]
        Asset tickers
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    metric : str
        Metric to optimize
    n_iterations : int
        Number of optimization iterations
    n_initial_points : int
        Number of random initialization points
    output_path : str
        Path to save results (JSON)
    """
    logger = StructuredLogger(__name__)
    logger.info(
        "Starting optimization pipeline",
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
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
            "Assets are NOT cointegrated. Optimization may not be meaningful.",
            test_statistic=float(coint_result.test_statistic),
        )

    # Use first eigenvector as basket weights
    weights = coint_result.eigenvectors[:, 0]
    logger.info("Basket weights", weights=weights.tolist())

    # Step 3: Define parameter space
    parameter_space = {
        "entry_threshold": (1.5, 3.0),
        "exit_threshold": (0.2, 1.0),
        "stop_loss": (3.0, 5.0),
        "lookback_window": (60.0, 504.0),  # ~3 months to 2 years
        "position_size": (0.10, 0.30),
        "min_holding_period": (3.0, 10.0),
    }

    logger.info("Parameter space defined", parameter_space=parameter_space)

    # Step 4: Create objective function
    logger.info("Creating objective function", metric=metric)
    objective = create_objective_function(prices, weights, metric=metric)

    # Step 5: Run optimization
    logger.info("Starting Bayesian optimization")
    optimizer = BayesianOptimizer(
        objective_function=objective,
        parameter_space=parameter_space,
        n_initial_points=n_initial_points,
        n_iterations=n_iterations,
        random_state=42,
        verbose=True,
    )

    result = optimizer.optimize()

    # Step 6: Display results
    print("\n" + "=" * 60)
    print("BAYESIAN OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Metric Optimized:     {metric}")
    print(f"Best Score:           {-result.best_score:.4f}")
    print(f"Total Iterations:     {result.n_iterations}")
    print(f"\nBest Parameters:")
    for param, value in result.best_params.items():
        if param in ["lookback_window", "min_holding_period"]:
            print(f"  {param:20s} = {int(value)}")
        else:
            print(f"  {param:20s} = {value:.4f}")
    print("=" * 60 + "\n")

    # Step 7: Save results
    if output_path:
        logger.info("Saving results", output_path=output_path)

        results_dict = {
            "optimization_parameters": {
                "tickers": tickers,
                "start_date": start_date,
                "end_date": end_date,
                "metric": metric,
                "n_iterations": n_iterations,
                "n_initial_points": n_initial_points,
            },
            "cointegration": {
                "is_cointegrated": bool(coint_result.is_cointegrated),
                "weights": weights.tolist(),
            },
            "best_parameters": result.best_params,
            "best_score": float(-result.best_score),  # Convert back to positive
            "convergence_history": [float(-s) for s in result.convergence_history],
        }

        save_json(results_dict, output_path)
        logger.info("Results saved successfully")

        # Save optimizer state
        state_path = str(Path(output_path).parent / "optimizer_state.pkl")
        optimizer.save_state(state_path)
        logger.info("Optimizer state saved", path=state_path)

    logger.info("Optimization pipeline completed")


def main():
    """Parse arguments and run optimization."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for strategy parameters"
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
        "--metric",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
        help="Metric to optimize (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=50,
        help="Number of optimization iterations (default: 50)",
    )
    parser.add_argument(
        "--n-initial-points",
        type=int,
        default=10,
        help="Number of random initialization points (default: 10)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save results JSON (optional)"
    )

    args = parser.parse_args()

    run_optimization(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        metric=args.metric,
        n_iterations=args.n_iterations,
        n_initial_points=args.n_initial_points,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
