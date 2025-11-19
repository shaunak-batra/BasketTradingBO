"""
Module: Signal Generator

Generate trading signals based on mean reversion logic using z-score thresholds.
Implements state machine for position management.

Classes
-------
SignalGenerator
    Main interface for signal generation

Functions
---------
generate_signals
    Generate mean reversion trading signals

Notes
-----
Signal Logic (State Machine):
    IDLE → LONG:   zscore < -entry_threshold (spread oversold)
    IDLE → SHORT:  zscore > entry_threshold (spread overbought)
    LONG → IDLE:   zscore > -exit_threshold OR zscore < -stop_loss
    SHORT → IDLE:  zscore < exit_threshold OR zscore > stop_loss

Author: Quantitative Research Team
Created: 2025-01-18
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.config import ConfigManager
from src.utils.exceptions import SignalGenerationException
from src.utils.logger import StructuredLogger, timed_execution


class SignalGenerator:
    """
    Generate trading signals based on mean reversion logic.

    Attributes
    ----------
    entry_threshold : float
        Z-score threshold for trade entry
    exit_threshold : float
        Z-score threshold for trade exit
    stop_loss : float
        Z-score threshold for stop loss
    position_limits : Dict[str, float]
        Position size limits

    Methods
    -------
    generate_signals(zscore)
        Generate trading signals from z-score
    get_current_position()
        Get current position state

    Examples
    --------
    >>> generator = SignalGenerator(entry_threshold=2.0, exit_threshold=0.5)
    >>> signals = generator.generate_signals(zscore)
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss: float = 4.0,
        position_limits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize signal generator.

        Parameters
        ----------
        entry_threshold : float
            Z-score threshold for entry (absolute value)
        exit_threshold : float
            Z-score threshold for exit (absolute value)
        stop_loss : float
            Z-score threshold for stop loss (absolute value)
        position_limits : Optional[Dict[str, float]]
            Position size limits
        """
        self.logger = StructuredLogger(__name__)

        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss

        if position_limits is None:
            position_limits = {"max_long": 1.0, "max_short": 1.0}

        self.position_limits = position_limits
        self.current_position = 0

        # Validate thresholds
        if entry_threshold <= 0:
            raise SignalGenerationException(
                "Entry threshold must be positive",
                context={"entry_threshold": entry_threshold}
            )

        if exit_threshold < 0:
            raise SignalGenerationException(
                "Exit threshold must be non-negative",
                context={"exit_threshold": exit_threshold}
            )

        if exit_threshold >= entry_threshold:
            raise SignalGenerationException(
                "Exit threshold must be less than entry threshold",
                context={"exit_threshold": exit_threshold, "entry_threshold": entry_threshold}
            )

        if stop_loss <= entry_threshold:
            raise SignalGenerationException(
                "Stop loss must be greater than entry threshold",
                context={"stop_loss": stop_loss, "entry_threshold": entry_threshold}
            )

        self.logger.info(
            "SignalGenerator initialized",
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss=stop_loss
        )

    @timed_execution
    def generate_signals(self, zscore: pd.Series) -> pd.Series:
        """
        Generate mean reversion trading signals.

        Parameters
        ----------
        zscore : pd.Series
            Spread z-score time series

        Returns
        -------
        pd.Series
            Signal time series: 1 (long), -1 (short), 0 (no position)

        Examples
        --------
        >>> signals = generator.generate_signals(zscore)
        >>> print(signals.value_counts())

        Notes
        -----
        State Machine:
            IDLE (0) → LONG (1):   zscore < -entry_threshold
            IDLE (0) → SHORT (-1): zscore > entry_threshold
            LONG (1) → IDLE (0):   zscore > -exit_threshold OR zscore < -stop_loss
            SHORT (-1) → IDLE (0): zscore < exit_threshold OR zscore > stop_loss
        """
        signals = pd.Series(0, index=zscore.index, name='signal')
        position = 0  # Current position state

        for i in range(len(zscore)):
            z = zscore.iloc[i]

            # Skip if NaN
            if pd.isna(z):
                signals.iloc[i] = position
                continue

            if position == 0:  # No position (IDLE)
                if z < -self.entry_threshold:
                    position = 1  # Enter long
                elif z > self.entry_threshold:
                    position = -1  # Enter short

            elif position == 1:  # Long position
                # Exit if crosses exit threshold or hits stop loss
                if z > -self.exit_threshold or z < -self.stop_loss:
                    position = 0  # Exit long

            elif position == -1:  # Short position
                # Exit if crosses exit threshold or hits stop loss
                if z < self.exit_threshold or z > self.stop_loss:
                    position = 0  # Exit short

            signals.iloc[i] = position

        # Count signals
        long_count = (signals == 1).sum()
        short_count = (signals == -1).sum()
        idle_count = (signals == 0).sum()

        self.logger.info(
            "Signals generated",
            total_points=len(signals),
            long_signals=long_count,
            short_signals=short_count,
            idle=idle_count
        )

        self.current_position = position

        return signals

    def get_current_position(self) -> int:
        """
        Get current position state.

        Returns
        -------
        int
            Current position (1: long, -1: short, 0: no position)
        """
        return self.current_position


# Standalone utility function

def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 4.0
) -> pd.Series:
    """
    Generate mean reversion trading signals.

    Parameters
    ----------
    zscore : pd.Series
        Spread z-score time series
    entry_threshold : float
        Z-score threshold for trade entry
    exit_threshold : float
        Z-score threshold for trade exit
    stop_loss : float
        Z-score threshold for stop loss

    Returns
    -------
    pd.Series
        Signal time series: 1 (long), -1 (short), 0 (no position)

    Examples
    --------
    >>> signals = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)

    Notes
    -----
    State Machine:
        IDLE → LONG:   zscore < -entry_threshold (spread oversold)
        IDLE → SHORT:  zscore > entry_threshold (spread overbought)
        LONG → IDLE:   zscore > -exit_threshold OR zscore < -stop_loss
        SHORT → IDLE:  zscore < exit_threshold OR zscore > stop_loss
    """
    signals = pd.Series(0, index=zscore.index)
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if pd.isna(z):
            signals.iloc[i] = position
            continue

        if position == 0:  # No position
            if z < -entry_threshold:
                position = 1  # Enter long
            elif z > entry_threshold:
                position = -1  # Enter short

        elif position == 1:  # Long position
            if z > -exit_threshold or z < -stop_loss:
                position = 0  # Exit long

        elif position == -1:  # Short position
            if z < exit_threshold or z > stop_loss:
                position = 0  # Exit short

        signals.iloc[i] = position

    return signals
