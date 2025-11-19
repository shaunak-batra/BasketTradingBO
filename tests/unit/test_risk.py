"""
Unit tests for risk management.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.manager import (
    RiskManager,
    VaRResult,
    calculate_var_cornish_fisher,
    calculate_var_historical,
    calculate_var_parametric,
)


class TestRiskManager:
    """Tests for RiskManager class."""

    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        risk_manager = RiskManager(
            max_position_size=0.25,
            max_portfolio_var=0.02,
            var_confidence_level=0.95
        )

        assert risk_manager.max_position_size == 0.25
        assert risk_manager.max_portfolio_var == 0.02
        assert risk_manager.var_confidence_level == 0.95

    def test_calculate_var_historical(self):
        """Test historical VaR calculation."""
        # Generate returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(300) * 0.01)

        risk_manager = RiskManager()
        var_result = risk_manager.calculate_var(returns, method="historical")

        assert isinstance(var_result, VaRResult)
        assert var_result.var > 0
        assert var_result.confidence_level == 0.95
        assert var_result.method == "historical"
        assert var_result.expected_shortfall is not None

    def test_calculate_var_parametric(self):
        """Test parametric VaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(300) * 0.01)

        risk_manager = RiskManager()
        var_result = risk_manager.calculate_var(returns, method="parametric")

        assert isinstance(var_result, VaRResult)
        assert var_result.var > 0
        assert var_result.method == "parametric"
        assert var_result.expected_shortfall is not None

    def test_calculate_var_cornish_fisher(self):
        """Test Cornish-Fisher VaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(300) * 0.01)

        risk_manager = RiskManager()
        var_result = risk_manager.calculate_var(returns, method="cornish_fisher")

        assert isinstance(var_result, VaRResult)
        assert var_result.var > 0
        assert var_result.method == "cornish_fisher"

    def test_calculate_position_size(self):
        """Test position sizing based on VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(300) * 0.02)  # 2% volatility

        risk_manager = RiskManager(max_position_size=0.30)
        position_size = risk_manager.calculate_position_size(
            returns,
            target_var=0.02,
            method="historical"
        )

        assert 0 < position_size <= 0.30
        assert isinstance(position_size, float)

    def test_check_risk_limits_pass(self):
        """Test risk limits check when within limits."""
        risk_manager = RiskManager(max_position_size=0.30)

        positions = pd.Series({'AAPL': 100, 'MSFT': 50})
        prices = pd.Series({'AAPL': 150.0, 'MSFT': 300.0})
        portfolio_value = 100000.0

        # Position value = 100*150 + 50*300 = 30000
        # Position size = 30000 / 100000 = 0.30 (exactly at limit)

        is_safe = risk_manager.check_risk_limits(positions, portfolio_value, prices)
        assert is_safe is True

    def test_check_risk_limits_fail(self):
        """Test risk limits check when limit violated."""
        risk_manager = RiskManager(max_position_size=0.20)

        positions = pd.Series({'AAPL': 200})
        prices = pd.Series({'AAPL': 150.0})
        portfolio_value = 100000.0

        # Position value = 200*150 = 30000
        # Position size = 30000 / 100000 = 0.30 > 0.20 (limit)

        is_safe = risk_manager.check_risk_limits(positions, portfolio_value, prices)
        assert is_safe is False

    def test_insufficient_data_error(self):
        """Test error when insufficient data for VaR."""
        risk_manager = RiskManager()

        # Too few data points
        returns = pd.Series([0.01, 0.02, -0.01])

        with pytest.raises(Exception):  # Should raise RiskManagementException
            risk_manager.calculate_var(returns, method="historical")


class TestVaRFunctions:
    """Tests for standalone VaR functions."""

    def test_historical_var_calculation(self):
        """Test historical VaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)

        var_result = calculate_var_historical(returns, confidence_level=0.95)

        assert isinstance(var_result, VaRResult)
        assert var_result.var > 0
        assert var_result.confidence_level == 0.95
        assert var_result.expected_shortfall >= var_result.var

    def test_parametric_var_calculation(self):
        """Test parametric VaR calculation."""
        # Create normally distributed returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)

        var_result = calculate_var_parametric(returns, confidence_level=0.95)

        assert isinstance(var_result, VaRResult)
        assert var_result.var > 0
        assert var_result.method == "parametric"

    def test_cornish_fisher_var_calculation(self):
        """Test Cornish-Fisher VaR with skewed distribution."""
        # Create skewed returns
        np.random.seed(42)
        returns = pd.Series(np.random.exponential(0.01, 500) - 0.01)

        var_result = calculate_var_cornish_fisher(returns, confidence_level=0.95)

        assert isinstance(var_result, VaRResult)
        assert var_result.var > 0
        assert var_result.method == "cornish_fisher"

    def test_var_comparison_methods(self):
        """Compare different VaR methods on same data."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)

        var_hist = calculate_var_historical(returns, 0.95)
        var_param = calculate_var_parametric(returns, 0.95)
        var_cf = calculate_var_cornish_fisher(returns, 0.95)

        # All should be positive
        assert var_hist.var > 0
        assert var_param.var > 0
        assert var_cf.var > 0

        # For normal data, parametric and CF should be similar
        assert abs(var_param.var - var_cf.var) / var_param.var < 0.2  # Within 20%

    def test_var_99_vs_95(self):
        """Test that 99% VaR > 95% VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)

        var_95 = calculate_var_historical(returns, 0.95)
        var_99 = calculate_var_historical(returns, 0.99)

        # 99% VaR should be higher (more conservative)
        assert var_99.var > var_95.var


class TestVaRResult:
    """Tests for VaRResult dataclass."""

    def test_var_result_creation(self):
        """Test VaRResult instantiation."""
        result = VaRResult(
            var=0.025,
            confidence_level=0.95,
            method="historical",
            expected_shortfall=0.030
        )

        assert result.var == 0.025
        assert result.confidence_level == 0.95
        assert result.method == "historical"
        assert result.expected_shortfall == 0.030

    def test_var_result_without_es(self):
        """Test VaRResult without expected shortfall."""
        result = VaRResult(
            var=0.025,
            confidence_level=0.95,
            method="cornish_fisher"
        )

        assert result.var == 0.025
        assert result.expected_shortfall is None
