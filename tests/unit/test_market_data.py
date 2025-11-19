"""
Unit tests for market data adapter.

Tests cover data fetching, validation, and saving functionality.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import DataValidator, MarketDataAdapter, ValidationReport
from src.utils.exceptions import DataQualityException


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return DataValidator()

    @pytest.fixture
    def clean_data(self):
        """Create clean test data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        data = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'MSFT': 200 + np.cumsum(np.random.randn(100) * 0.5)
        }, index=dates)
        return data

    @pytest.fixture
    def data_with_missing(self):
        """Create data with missing values."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        data = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'MSFT': 200 + np.cumsum(np.random.randn(100) * 0.5)
        }, index=dates)
        # Add missing values
        data.iloc[10:15, 0] = np.nan
        return data

    def test_check_missing_values_clean(self, validator, clean_data):
        """Test missing value check with clean data."""
        is_valid, ratio = validator.check_missing_values(clean_data)
        assert is_valid is True
        assert ratio == 0.0

    def test_check_missing_values_with_missing(self, validator, data_with_missing):
        """Test missing value check with missing data."""
        is_valid, ratio = validator.check_missing_values(data_with_missing)
        # 5 missing out of 200 total = 2.5%
        assert ratio == 0.025
        assert is_valid is True  # Under 5% threshold

    def test_detect_outliers(self, validator, clean_data):
        """Test outlier detection."""
        outliers, count = validator.detect_outliers(clean_data)
        assert isinstance(outliers, pd.DataFrame)
        assert count >= 0

    def test_check_price_continuity(self, validator, clean_data):
        """Test price continuity check."""
        discontinuities = validator.check_price_continuity(clean_data)
        assert isinstance(discontinuities, list)
        # Clean data should have no major discontinuities
        assert len(discontinuities) == 0

    def test_validate_clean_data(self, validator, clean_data):
        """Test full validation on clean data."""
        report = validator.validate(clean_data)
        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        assert report.missing_ratio == 0.0
        assert len(report.errors) == 0

    def test_validate_data_with_issues(self, validator):
        """Test validation with data quality issues."""
        # Create data with excessive missing values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': [np.nan] * 50 + list(range(50)),
            'MSFT': list(range(100))
        }, index=dates)

        report = validator.validate(data)
        assert report.is_valid is False
        assert report.missing_ratio > 0.05
        assert len(report.errors) > 0


class TestFeatures:
    """Tests for feature engineering."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        data = pd.DataFrame({
            'AAPL': 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01)),
            'MSFT': 200 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        }, index=dates)
        return data

    def test_calculate_log_prices(self, sample_prices):
        """Test log price calculation."""
        from src.data.features import calculate_log_prices

        log_prices = calculate_log_prices(sample_prices)
        assert log_prices.shape == sample_prices.shape
        assert not log_prices.isnull().any().any()
        # Log prices should be roughly log(100) ≈ 4.6 for AAPL
        assert 4.0 < log_prices['AAPL'].mean() < 5.5

    def test_calculate_returns_log(self, sample_prices):
        """Test log return calculation."""
        from src.data.features import calculate_returns

        returns = calculate_returns(sample_prices, method='log')
        assert returns.shape == sample_prices.shape
        # First value should be NaN
        assert returns.iloc[0].isnull().all()
        # Returns should be small (around 1%)
        assert abs(returns['AAPL'].mean()) < 0.05

    def test_calculate_returns_simple(self, sample_prices):
        """Test simple return calculation."""
        from src.data.features import calculate_returns

        returns = calculate_returns(sample_prices, method='simple')
        assert returns.shape == sample_prices.shape
        # First value should be NaN
        assert returns.iloc[0].isnull().all()

    def test_create_spread(self, sample_prices):
        """Test spread creation."""
        from src.data.features import create_spread

        weights = np.array([1.0, -0.5])
        spread = create_spread(sample_prices, weights)

        assert isinstance(spread, pd.Series)
        assert len(spread) == len(sample_prices)
        assert not spread.isnull().any()

    def test_create_spread_wrong_weights(self, sample_prices):
        """Test spread creation with wrong number of weights."""
        from src.data.features import create_spread

        weights = np.array([1.0])  # Only 1 weight for 2 columns

        with pytest.raises(ValueError):
            create_spread(sample_prices, weights)


class TestCacheManager:
    """Tests for CacheManager class."""

    @pytest.fixture
    def cache_manager(self, tmp_path):
        """Create cache manager with temporary directory."""
        from src.data.cache import CacheManager
        from src.utils.config import ConfigManager

        # Create temporary config
        config = ConfigManager({
            'data': {
                'storage': {
                    'raw_data_path': str(tmp_path / 'cache'),
                    'cache_ttl': 3600
                }
            }
        })

        return CacheManager(config)

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'AAPL': range(10),
            'MSFT': range(10, 20)
        }, index=dates)

    def test_set_and_get(self, cache_manager, sample_df):
        """Test setting and getting cache values."""
        key = "test_key"
        cache_manager.set(key, sample_df)

        retrieved = cache_manager.get(key)
        assert retrieved is not None
        pd.testing.assert_frame_equal(retrieved, sample_df)

    def test_get_nonexistent(self, cache_manager):
        """Test getting nonexistent key."""
        result = cache_manager.get("nonexistent_key")
        assert result is None

    def test_invalidate(self, cache_manager, sample_df):
        """Test cache invalidation."""
        key = "test_key_123"
        cache_manager.set(key, sample_df)

        # Verify it's cached
        assert cache_manager.get(key) is not None

        # Invalidate
        count = cache_manager.invalidate(key)
        assert count == 1

        # Verify it's gone
        assert cache_manager.get(key) is None

    def test_get_stats(self, cache_manager, sample_df):
        """Test cache statistics."""
        cache_manager.set("key1", sample_df)
        cache_manager.set("key2", sample_df)

        cache_manager.get("key1")  # Hit
        cache_manager.get("nonexistent")  # Miss

        stats = cache_manager.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert 0 < stats.hit_rate < 1
        assert stats.total_entries >= 2
