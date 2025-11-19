"""
Module: Signal Filters

Filter trading signals to avoid whipsaws and enforce minimum holding periods.

Functions
---------
apply_whipsaw_filter
    Enforce minimum holding period
apply_max_trades_filter
    Limit maximum number of trades
apply_time_filter
    Filter signals based on time of day/week

Author: Quantitative Research Team
Created: 2025-01-18
"""

from typing import Optional

import pandas as pd

from src.utils.logger import StructuredLogger, timed_execution


@timed_execution
def apply_whipsaw_filter(
    signals: pd.Series,
    min_holding_period: int = 5
) -> pd.Series:
    """
    Enforce minimum holding period to avoid excessive trading.

    Parameters
    ----------
    signals : pd.Series
        Raw trading signals
    min_holding_period : int
        Minimum days to hold a position

    Returns
    -------
    pd.Series
        Filtered signals

    Examples
    --------
    >>> filtered = apply_whipsaw_filter(signals, min_holding_period=5)

    Notes
    -----
    If a position is exited before the minimum holding period,
    the exit signal is ignored and the position is maintained.
    """
    logger = StructuredLogger(__name__)

    filtered_signals = signals.copy()
    position_entry_idx = None
    position_entry_signal = None

    for i in range(len(signals)):
        current_signal = signals.iloc[i]

        if current_signal != 0 and position_entry_idx is None:
            # New position entered
            position_entry_idx = i
            position_entry_signal = current_signal

        elif current_signal == 0 and position_entry_idx is not None:
            # Position exit signal
            holding_period = i - position_entry_idx

            if holding_period < min_holding_period:
                # Force hold position (ignore exit signal)
                filtered_signals.iloc[i] = position_entry_signal
            else:
                # Allow exit
                position_entry_idx = None
                position_entry_signal = None

        elif current_signal != 0 and position_entry_idx is not None:
            # Position reversal
            holding_period = i - position_entry_idx

            if holding_period < min_holding_period:
                # Keep old position
                filtered_signals.iloc[i] = position_entry_signal
            else:
                # Allow reversal
                position_entry_idx = i
                position_entry_signal = current_signal

    # Count filtered signals
    original_changes = (signals.diff() != 0).sum()
    filtered_changes = (filtered_signals.diff() != 0).sum()
    filtered_count = original_changes - filtered_changes

    logger.info(
        "Whipsaw filter applied",
        min_holding_period=min_holding_period,
        original_signal_changes=original_changes,
        filtered_signal_changes=filtered_changes,
        signals_filtered=filtered_count
    )

    return filtered_signals


@timed_execution
def apply_max_trades_filter(
    signals: pd.Series,
    max_trades: int = 100
) -> pd.Series:
    """
    Limit maximum number of trades.

    Parameters
    ----------
    signals : pd.Series
        Trading signals
    max_trades : int
        Maximum number of trades allowed

    Returns
    -------
    pd.Series
        Filtered signals (stops after max_trades reached)

    Examples
    --------
    >>> filtered = apply_max_trades_filter(signals, max_trades=50)
    """
    logger = StructuredLogger(__name__)

    filtered_signals = signals.copy()
    trade_count = 0

    for i in range(1, len(signals)):
        # Count trades (position changes)
        if signals.iloc[i] != signals.iloc[i-1]:
            trade_count += 1

            if trade_count >= max_trades:
                # Stop trading, close all positions
                filtered_signals.iloc[i:] = 0
                break

    logger.info(
        "Max trades filter applied",
        max_trades=max_trades,
        actual_trades=trade_count
    )

    return filtered_signals


def apply_combined_filters(
    signals: pd.Series,
    min_holding_period: int = 5,
    max_trades: Optional[int] = None
) -> pd.Series:
    """
    Apply multiple filters in sequence.

    Parameters
    ----------
    signals : pd.Series
        Raw trading signals
    min_holding_period : int
        Minimum days to hold position
    max_trades : Optional[int]
        Maximum number of trades (None for no limit)

    Returns
    -------
    pd.Series
        Filtered signals

    Examples
    --------
    >>> filtered = apply_combined_filters(
    ...     signals, min_holding_period=5, max_trades=100
    ... )
    """
    # Apply whipsaw filter first
    filtered = apply_whipsaw_filter(signals, min_holding_period)

    # Apply max trades filter if specified
    if max_trades is not None:
        filtered = apply_max_trades_filter(filtered, max_trades)

    return filtered
