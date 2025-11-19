"""
Unit tests for cointegration analysis.

Tests cover Johansen test, ADF test, spread calculation, z-score, and half-life.
"""

import numpy as np
import pandas as pd
import pytest

from src.cointegration.engine import CointegrationEngine
from src.cointegration.spread import SpreadCalculator, calculate_half_life, calculate_zscore
from src.utils.exceptions import NoCointegrationException, StationarityException
from tests.fixtures.sample_data import generate_cointegrated_prices, generate_mean_reverting_spread


class TestCointegrationEngine:
    """Tests for CointegrationEngine class."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CointegrationEngine()

    @pytest.fixture
    def cointegrated_data(self):
        """Create cointegrated price series."""
        return generate_cointegrated_prices(n_days=500, cointegration_strength=0.9, seed=42)

    @pytest.fixture
    def non_cointegrated_data(self):
        """Create non-cointegrated price series."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'X': 100 + np.cumsum(np.random.randn(100)),
            'Y': 200 + np.cumsum(np.random.randn(100))
        }, index=dates)
        return data

    def test_cointegration_detected(self, engine, cointegrated_data):
        """Test that cointegration is detected in cointegrated series."""
        result = engine.test_cointegration(cointegrated_data)

        assert result.is_cointegrated is True
        assert result.cointegrating_rank >= 1
        assert result.eigenvectors is not None
        assert result.eigenvalues is not None

    def test_cointegration_not_detected_raises(self, engine, non_cointegrated_data):
        """Test that NoCointegrationException is raised for non-cointegrated series."""
        with pytest.raises(NoCointegrationException):
            engine.test_cointegration(non_cointegrated_data)

    def test_calculate_spread(self, engine, cointegrated_data):
        """Test spread calculation."""
        weights = np.array([1.0, -0.5])
        spread = engine.calculate_spread(cointegrated_data, weights)

        assert isinstance(spread, pd.Series)
        assert len(spread) == len(cointegrated_data)
        assert not spread.isnull().any()

    def test_stationarity_test_stationary(self, engine):
        """Test stationarity test on stationary series."""
        # Generate stationary series
        mean_reverting = generate_mean_reverting_spread(n_days=500, half_life=20, seed=42)

        result = engine.test_stationarity(mean_reverting)

        assert result.is_stationary is True
        assert result.p_value < 0.05
        assert result.test_statistic < result.critical_values["5%"]

    def test_stationarity_test_non_stationary_raises(self, engine):
        """Test that StationarityException is raised for non-stationary series."""
        # Generate random walk (non-stationary)
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        random_walk = pd.Series(np.cumsum(np.random.randn(500)), index=dates)

        with pytest.raises(StationarityException):
            engine.test_stationarity(random_walk)


class TestSpreadCalculator:
    """Tests for SpreadCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SpreadCalculator()

    @pytest.fixture
    def mean_reverting_spread(self):
        """Create mean reverting spread."""
        return generate_mean_reverting_spread(n_days=500, half_life=20, seed=42)

    def test_calculate_zscore(self, calculator, mean_reverting_spread):
        """Test z-score calculation."""
        zscore = calculator.calculate_zscore(mean_reverting_spread, lookback=252)

        assert isinstance(zscore, pd.Series)
        assert len(zscore) == len(mean_reverting_spread)
        # Z-score should have approximately mean=0 and std=1
        assert abs(zscore.mean()) < 0.5
        assert abs(zscore.std() - 1.0) < 0.5

    def test_calculate_half_life(self, calculator, mean_reverting_spread):
        """Test half-life calculation."""
        half_life = calculator.calculate_half_life(mean_reverting_spread)

        # Should be a positive number
        assert half_life > 0
        # Should be in reasonable range (generated with half_life=20)
        assert 5 < half_life < 60

    def test_calculate_half_life_no_mean_reversion(self, calculator):
        """Test half-life calculation with non-mean-reverting series."""
        # Generate random walk
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        random_walk = pd.Series(np.cumsum(np.random.randn(500)), index=dates)

        with pytest.raises(ValueError, match="No mean reversion detected"):
            calculator.calculate_half_life(random_walk)

    def test_calculate_hurst_exponent(self, calculator, mean_reverting_spread):
        """Test Hurst exponent calculation."""
        hurst = calculator.calculate_hurst_exponent(mean_reverting_spread)

        # Should be between 0 and 1
        assert 0 < hurst < 1
        # Mean reverting should have H < 0.5
        assert hurst < 0.5


class TestStandaloneFunctions:
    """Tests for standalone utility functions."""

    def test_calculate_zscore_function(self):
        """Test standalone calculate_zscore function."""
        spread = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        zscore = calculate_zscore(spread, lookback=5)

        assert isinstance(zscore, pd.Series)
        assert len(zscore) == len(spread)

        # Last value should be normalized
        expected = (10 - 8) / np.std([6, 7, 8, 9, 10], ddof=1)
        assert abs(zscore.iloc[-1] - expected) < 0.01

    def test_calculate_half_life_function(self):
        """Test standalone calculate_half_life function."""
        spread = generate_mean_reverting_spread(n_days=500, half_life=25, seed=42)
        half_life = calculate_half_life(spread)

        assert half_life > 0
        assert 5 < half_life < 60
