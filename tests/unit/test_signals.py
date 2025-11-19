"""
Unit tests for signal generation and filtering.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.filters import apply_whipsaw_filter
from src.strategy.signals import SignalGenerator, generate_signals


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    def test_signal_generation_logic(self):
        """Test basic signal state machine."""
        # Create z-score series that should generate specific signals
        zscore = pd.Series([0, -2.5, -2.0, -0.5, 0.5, 2.5, 2.0, 0.5, 0])

        signals = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)

        # Expected: 0, 1 (enter long), 1, 1, 0 (exit), -1 (enter short), -1, 0 (exit), 0
        expected = pd.Series([0, 1, 1, 1, 0, -1, -1, 0, 0])
        pd.testing.assert_series_equal(signals, expected)

    def test_signal_generator_class(self):
        """Test SignalGenerator class."""
        generator = SignalGenerator(entry_threshold=2.0, exit_threshold=0.5)

        zscore = pd.Series([0, -2.5, -1.0, 0.0, 2.5, 1.0, 0.0])
        signals = generator.generate_signals(zscore)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(zscore)


class TestFilters:
    """Tests for signal filters."""

    def test_whipsaw_filter(self):
        """Test whipsaw filter enforces minimum holding period."""
        # Signal that exits too quickly
        signals = pd.Series([0, 1, 1, 0, 0, -1, -1, -1, 0])

        filtered = apply_whipsaw_filter(signals, min_holding_period=5)

        # Should extend holding period
        assert filtered.iloc[3] == 1  # Should still be long
        assert filtered.iloc[4] == 1  # Should still be long
