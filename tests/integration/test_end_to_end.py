"""
Integration tests for end-to-end workflow.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtester import Backtester
from src.cointegration.engine import CointegrationEngine
from src.cointegration.spread import SpreadCalculator
from src.strategy.filters import apply_whipsaw_filter
from src.strategy.signals import SignalGenerator
from tests.fixtures.sample_data import generate_cointegrated_prices


class TestEndToEndWorkflow:
    """Integration tests for complete trading workflow."""

    def test_complete_backtest_pipeline(self):
        """Test complete pipeline from data to backtest results."""
        # Step 1: Generate cointegrated price data
        prices = generate_cointegrated_prices(n_assets=3, n_periods=500, seed=42)

        # Step 2: Test cointegration
        coint_engine = CointegrationEngine()
        coint_result = coint_engine.test_cointegration(prices)

        assert coint_result.is_cointegrated
        assert len(coint_result.eigenvectors) > 0

        # Step 3: Calculate spread and z-score
        weights = coint_result.eigenvectors[:, 0]
        spread_calc = SpreadCalculator()
        spread = spread_calc.calculate_spread(prices, weights)
        zscore = spread_calc.calculate_zscore(spread, lookback=60)

        assert len(spread) == len(prices)
        assert len(zscore) == len(prices)

        # Step 4: Generate signals
        signal_gen = SignalGenerator(
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_loss=4.0
        )
        signals = signal_gen.generate_signals(zscore)

        # Apply filters
        filtered_signals = apply_whipsaw_filter(signals, min_holding_period=5)

        assert len(filtered_signals) == len(signals)

        # Step 5: Run backtest
        backtester = Backtester(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
            position_size=0.20
        )

        result = backtester.run_backtest(prices, filtered_signals, weights)

        # Verify results
        assert result.portfolio_value.iloc[0] == backtester.initial_capital
        assert len(result.returns) == len(prices)
        assert result.metrics.num_trades >= 0
        assert -1 <= result.metrics.max_drawdown <= 0

    def test_spread_stationarity_workflow(self):
        """Test spread calculation and stationarity check."""
        # Generate cointegrated data
        prices = generate_cointegrated_prices(n_assets=2, n_periods=300, seed=123)

        # Find cointegration
        coint_engine = CointegrationEngine()
        coint_result = coint_engine.test_cointegration(prices)

        # Calculate spread
        weights = coint_result.eigenvectors[:, 0]
        spread_calc = SpreadCalculator()
        spread = spread_calc.calculate_spread(prices, weights)

        # Calculate half-life (if it's calculated without error, spread is mean-reverting)
        half_life = spread_calc.calculate_half_life(spread)

        # Verify mean reversion properties
        assert half_life > 0
        assert half_life < len(spread)  # Should revert within data period

    def test_signal_generation_workflow(self):
        """Test signal generation from z-score."""
        # Create z-score that crosses thresholds
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        zscore = pd.Series([
            0, 0, 0,  # Start neutral
            -2.5, -2.2, -1.8, -1.0, -0.3,  # Cross entry threshold, then exit
            0, 0, 0,
            2.5, 2.2, 1.8, 1.0, 0.3,  # Cross short threshold, then exit
            0, 0, 0
        ] + [0] * 82, index=dates)

        # Generate signals
        signal_gen = SignalGenerator(entry_threshold=2.0, exit_threshold=0.5)
        signals = signal_gen.generate_signals(zscore)

        # Verify signal transitions
        # Should enter long around index 3
        assert signals.iloc[3] == 1 or signals.iloc[4] == 1

        # Should enter short around index 11
        assert signals.iloc[11] == -1 or signals.iloc[12] == -1

    def test_portfolio_allocation_workflow(self):
        """Test complete portfolio allocation workflow."""
        # Generate test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'AAPL': np.linspace(100, 110, 100),
            'MSFT': np.linspace(200, 210, 100),
            'GOOGL': np.linspace(1000, 1100, 100)
        }, index=dates)

        # Simple alternating signals
        signals = pd.Series([1 if i % 20 < 10 else 0 for i in range(100)], index=dates)

        # Weights
        weights = np.array([1.0, -0.5, -0.5])

        # Run backtest
        backtester = Backtester(initial_capital=100000.0, position_size=0.20)
        result = backtester.run_backtest(prices, signals, weights)

        # Verify positions were created
        assert result.positions.shape == prices.shape
        assert (result.positions != 0).any().any()  # Some non-zero positions

        # Verify transaction costs calculated
        assert result.transaction_costs.sum() > 0

        # Verify trades logged
        assert len(result.trades) > 0

    def test_optimization_objective_evaluation(self):
        """Test that optimization objective can be evaluated."""
        # Generate data
        prices = generate_cointegrated_prices(n_assets=2, n_periods=200, seed=42)

        # Get cointegration weights
        coint_engine = CointegrationEngine()
        coint_result = coint_engine.test_cointegration(prices)
        weights = coint_result.eigenvectors[:, 0]

        # Define objective function
        def objective(params):
            entry_threshold = params.get('entry_threshold', 2.0)
            exit_threshold = params.get('exit_threshold', 0.5)
            lookback_window = int(params.get('lookback_window', 60))

            # Calculate spread
            spread_calc = SpreadCalculator()
            spread = spread_calc.create_spread(prices, weights)
            zscore = spread_calc.calculate_zscore(spread, lookback=lookback_window)

            # Generate signals
            signal_gen = SignalGenerator(
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold
            )
            signals = signal_gen.generate_signals(zscore)

            # Backtest
            backtester = Backtester(initial_capital=100000.0)
            result = backtester.run_backtest(prices, signals, weights)

            # Return negative Sharpe (to minimize)
            return -result.metrics.sharpe_ratio

        # Test evaluation with sample parameters
        test_params = {
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'lookback_window': 60.0
        }

        score = objective(test_params)

        # Should return a finite number
        assert np.isfinite(score)
        assert isinstance(score, (int, float))


class TestRiskManagementIntegration:
    """Integration tests for risk management."""

    def test_var_calculation_on_backtest_returns(self):
        """Test VaR calculation on backtest results."""
        from src.risk.manager import RiskManager

        # Generate data and run backtest
        prices = generate_cointegrated_prices(n_assets=2, n_periods=300, seed=42)

        coint_engine = CointegrationEngine()
        coint_result = coint_engine.test_cointegration(prices)
        weights = coint_result.eigenvectors[:, 0]

        spread_calc = SpreadCalculator()
        spread = spread_calc.calculate_spread(prices, weights)
        zscore = spread_calc.calculate_zscore(spread, lookback=60)

        signal_gen = SignalGenerator()
        signals = signal_gen.generate_signals(zscore)

        backtester = Backtester()
        result = backtester.run_backtest(prices, signals, weights)

        # Calculate VaR on returns
        risk_manager = RiskManager()
        var_result = risk_manager.calculate_var(result.returns, method="historical")

        assert var_result.var > 0
        assert var_result.expected_shortfall >= var_result.var

    def test_position_sizing_integration(self):
        """Test position sizing based on VaR."""
        from src.risk.manager import RiskManager

        # Generate returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(300) * 0.02)

        risk_manager = RiskManager(max_position_size=0.30)

        # Calculate position size for 2% VaR target
        position_size = risk_manager.calculate_position_size(
            returns,
            target_var=0.02,
            method="parametric"
        )

        # Position size should be reasonable
        assert 0 < position_size <= 0.30
        assert isinstance(position_size, float)


class TestDataPersistence:
    """Integration tests for data saving/loading."""

    def test_save_and_load_backtest_results(self, tmp_path):
        """Test saving and loading backtest results."""
        from src.utils.io import save_json, load_json

        # Create sample results
        results_dict = {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "num_trades": 25
        }

        # Save
        file_path = str(tmp_path / "results.json")
        save_json(results_dict, file_path)

        # Load
        loaded_results = load_json(file_path)

        assert loaded_results["total_return"] == 0.15
        assert loaded_results["sharpe_ratio"] == 1.5
        assert loaded_results["num_trades"] == 25
