"""Spread construction and mean-reversion diagnostics.

Every rolling statistic at bar t uses bars t-L+1 .. t only, so nothing here can
leak future prices into a signal.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# A window counts as having no variance when its standard deviation is below
# this fraction of max(1, |window mean|). Round-off in a constant window is ~1e-16.
_RELATIVE_ZERO_STD = 1e-12


def compute_spread(prices: pd.DataFrame, weights: npt.ArrayLike) -> pd.Series:
    """Basket spread ``S_t = sum_i w_i * log(P_{i,t})``."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (prices.shape[1],) or not np.all(np.isfinite(w)):
        raise ValueError(f"weights must be a finite vector of length {prices.shape[1]}")
    values = prices.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return pd.Series(np.log(values) @ w, index=prices.index, name="spread")


def rolling_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Trailing z-score ``(S_t - mean_L) / std_L`` over the last ``lookback`` bars, including bar t.

    NaN for the first ``lookback - 1`` bars (no partial windows) and for windows with
    no variance. The window statistics are computed exactly with numpy rather than
    with an online rolling algorithm.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, (int, np.integer)) or lookback < 2:
        raise ValueError(f"lookback must be an integer >= 2, got {lookback!r}")
    values = spread.to_numpy(dtype=float)
    z = np.full(len(values), np.nan)
    if len(values) >= lookback:
        windows = sliding_window_view(values, int(lookback))
        mean = windows.mean(axis=1)
        std = windows.std(axis=1, ddof=1)
        valid = std > _RELATIVE_ZERO_STD * np.maximum(1.0, np.abs(mean))
        current = values[lookback - 1 :]
        z[lookback - 1 :] = np.where(valid, (current - mean) / np.where(valid, std, 1.0), np.nan)
    return pd.Series(z, index=spread.index, name="zscore")


def half_life(spread: pd.Series | npt.ArrayLike) -> float:
    """Mean-reversion half-life in bars, from the AR(1) fit ``dS_t = a + b * S_{t-1} + e_t``.

    With ``phi = 1 + b``, an expected deviation decays as ``phi**h``, so the half-life
    is ``ln(0.5) / ln(phi)``. Returns ``inf`` when ``phi >= 1`` (no mean reversion) and
    ``nan`` when ``phi <= 0`` (sign-flipping, not a smooth reversion).
    """
    s = np.asarray(spread, dtype=float)
    if s.ndim != 1 or len(s) < 20 or not np.all(np.isfinite(s)):
        raise ValueError("half_life needs at least 20 finite observations")
    lagged = s[:-1]
    design = np.column_stack([np.ones_like(lagged), lagged])
    coefficients, *_ = np.linalg.lstsq(design, np.diff(s), rcond=None)
    phi = 1.0 + float(coefficients[1])
    if phi >= 1.0:
        return math.inf
    if phi <= 0.0:
        return math.nan
    return math.log(0.5) / math.log(phi)
