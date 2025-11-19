"""
Unit tests for backtesting engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtester import Backtester, BacktestResult, calculate_returns
from src.backtesting.metrics import PerformanceMetrics


class TestBacktester:
    """Tests for Backtester class."""

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        backtester = Backtester(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
            position_size=0.20
        )

        assert backtester.initial_capital == 100000.0
        assert backtester.commission == 0.001
        assert backtester.slippage == 0.0005
        assert backtester.position_size == 0.20

    def test_run_backtest_simple(self):
        """Test simple backtest execution."""
        # Create simple test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'AAPL': np.linspace(100, 110, 100),
            'MSFT': np.linspace(200, 210, 100)
        }, index=dates)

        # Simple signals: long for first half, flat for second half
        signals = pd.Series([1] * 50 + [0] * 50, index=dates)

        # Equal weights
        weights = np.array([0.5, 0.5])

        backtester = Backtester(initial_capital=100000.0)
        result = backtester.run_backtest(prices, signals, weights)

        # Verify result structure
        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, PerformanceMetrics)
        assert len(result.portfolio_value) == len(prices)
        assert len(result.returns) == len(prices)
        assert result.positions.shape == prices.shape

    def test_transaction_costs_calculation(self):
        """Test transaction cost calculation."""
        backtester = Backtester(commission=0.001, slippage=0.0005)

        # Create positions with a trade
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        positions = pd.DataFrame({
            'AAPL': [0, 100, 100, 100, 0],
            'MSFT': [0, 50, 50, 50, 0]
        }, index=dates)

        prices = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104],
            'MSFT': [200, 201, 202, 203, 204]
        }, index=dates)

        costs = backtester.calculate_transaction_costs(positions, prices)

        # First row: open position
        # Cost = (100*100 + 50*200) * 0.0015 = 20000 * 0.0015 = 30
        assert costs.iloc[0] == pytest.approx(30.0, rel=1e-2)

        # Rows 1-3: no position change
        assert costs.iloc[1] == 0
        assert costs.iloc[2] == 0
        assert costs.iloc[3] == 0

        # Last row: close position
        # Cost = (100*104 + 50*204) * 0.0015 = 20600 * 0.0015 = 30.9
        assert costs.iloc[4] == pytest.approx(30.9, rel=1e-2)

    def test_trade_log_generation(self):
        """Test trade log generation."""
        backtester = Backtester()

        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        # Positions: open, hold, close
        positions = pd.DataFrame({
            'AAPL': [0, 100, 100, 100, 100, 0, 0, 0, 0, 0]
        }, index=dates)

        prices = pd.DataFrame({
            'AAPL': np.linspace(100, 110, 10)
        }, index=dates)

        signals = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], index=dates)

        trades = backtester.generate_trade_log(positions, prices, signals)

        # Should have 2 trades: open and close
        assert len(trades) == 2
        assert trades.iloc[0]['action'] == 'OPEN'
        assert trades.iloc[0]['signal'] == 1
        assert trades.iloc[1]['action'] == 'CLOSE'
        assert trades.iloc[1]['signal'] == 0

    def test_backtest_with_no_trades(self):
        """Test backtest with no signals."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = pd.DataFrame({
            'AAPL': np.random.randn(50).cumsum() + 100,
            'MSFT': np.random.randn(50).cumsum() + 200
        }, index=dates)

        # No signals
        signals = pd.Series([0] * 50, index=dates)
        weights = np.array([0.5, 0.5])

        backtester = Backtester()
        result = backtester.run_backtest(prices, signals, weights)

        # Portfolio value should remain constant at initial capital
        assert result.portfolio_value.iloc[0] == backtester.initial_capital
        assert result.portfolio_value.iloc[-1] == backtester.initial_capital
        assert len(result.trades) == 0


class TestCalculateReturns:
    """Tests for calculate_returns function."""

    def test_simple_returns(self):
        """Test return calculation."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')

        positions = pd.DataFrame({
            'AAPL': [100, 100, 100, 100, 100]
        }, index=dates)

        prices = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104]
        }, index=dates)

        returns = calculate_returns(positions, prices, initial_capital=100000)

        # Returns should reflect price changes
        # Portfolio = 100000 + 100 * price
        # Day 0: 100000 + 10000 = 110000
        # Day 1: 100000 + 10100 = 110100
        # Return = (110100 - 110000) / 110000 ≈ 0.0909%

        assert len(returns) == 5
        assert returns.iloc[0] == 0  # First return is 0
        assert returns.iloc[1] > 0  # Price went up

    def test_returns_with_zero_positions(self):
        """Test returns when no positions."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')

        positions = pd.DataFrame({
            'AAPL': [0, 0, 0, 0, 0]
        }, index=dates)

        prices = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104]
        }, index=dates)

        returns = calculate_returns(positions, prices, initial_capital=100000)

        # All returns should be zero (no positions)
        assert (returns == 0).all()


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_backtest_result_creation(self):
        """Test BacktestResult instantiation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        result = BacktestResult(
            portfolio_value=pd.Series(np.linspace(100000, 110000, 10), index=dates),
            returns=pd.Series(np.random.randn(10) * 0.01, index=dates),
            positions=pd.DataFrame({'AAPL': [100] * 10}, index=dates),
            signals=pd.Series([1] * 10, index=dates),
            metrics=PerformanceMetrics(
                total_return=0.10,
                annualized_return=0.15,
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                max_drawdown=-0.05,
                calmar_ratio=3.0,
                win_rate=0.6,
                profit_factor=1.8,
                num_trades=10,
                avg_trade_duration=5.0
            ),
            transaction_costs=pd.Series([10] * 10, index=dates),
            trades=pd.DataFrame({
                'timestamp': dates[:2],
                'action': ['OPEN', 'CLOSE'],
                'signal': [1, 0],
                'notional': [10000, 10000]
            })
        )

        assert isinstance(result.portfolio_value, pd.Series)
        assert isinstance(result.metrics, PerformanceMetrics)
        assert result.metrics.sharpe_ratio == 1.5
