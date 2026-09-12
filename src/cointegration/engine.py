"""Cointegration tests on log prices.

The Johansen trace test decides whether a basket is traded, and its leading
eigenvector supplies the basket weights. The Engle-Granger residual test is
reported alongside it as a cross-check only.

References
----------
Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in
    Gaussian vector autoregressive models. Econometrica 59(6), 1551-1580.
Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction.
    Econometrica 55(2), 251-276.
MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics
    Department Working Paper 1227.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.utils.exceptions import ConfigError, DataError

CRITICAL_VALUE_COLUMN = {0.10: 0, 0.05: 1, 0.01: 2}
MAX_JOHANSEN_SERIES = 12  # statsmodels tabulates critical values for up to 12 series


@dataclass(frozen=True)
class JohansenResult:
    """Johansen test output. Index r of each array tests H0: rank <= r."""

    trace_stats: npt.NDArray[np.float64]
    trace_critical: npt.NDArray[np.float64]
    max_eig_stats: npt.NDArray[np.float64]
    max_eig_critical: npt.NDArray[np.float64]
    eigenvalues: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]  # leading eigenvector, sum(|w|) = 1, first non-zero element > 0
    rank_trace: int
    rank_max_eig: int
    significance: float
    n_obs: int

    @property
    def is_cointegrated(self) -> bool:
        """True when the trace test rejects rank 0 at ``significance``."""
        return self.rank_trace > 0

    def summary(self) -> dict:
        return {
            "trace_stat_r0": float(self.trace_stats[0]),
            "trace_critical_r0": float(self.trace_critical[0]),
            "max_eig_stat_r0": float(self.max_eig_stats[0]),
            "max_eig_critical_r0": float(self.max_eig_critical[0]),
            "rank_trace": self.rank_trace,
            "rank_max_eig": self.rank_max_eig,
            "significance": self.significance,
            "n_obs": self.n_obs,
            "weights": [float(w) for w in self.weights],
        }


@dataclass(frozen=True)
class EngleGrangerResult:
    statistic: float
    p_value: float
    critical_5pct: float

    @property
    def rejects_at_5pct(self) -> bool:
        return self.statistic < self.critical_5pct


def normalize_weights(vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Scale a cointegrating vector to ``sum(|w|) = 1`` with its first non-zero element positive.

    Any non-zero multiple of a cointegrating vector is also cointegrating. This fixes
    one representative, so weights are comparable across folds and can be used
    directly as dollar allocations per unit of gross exposure.
    """
    w = np.asarray(vector, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.abs(w).sum() == 0:
        raise ValueError(f"cannot normalise weights {vector!r}")
    w = w / np.abs(w).sum()
    first = w[np.flatnonzero(w)[0]]
    return w if first > 0 else -w


def net_exposure(weights: npt.ArrayLike) -> float:
    """Net exposure as a share of gross: ``|sum(w)| / sum(|w|)``.

    Zero means a perfectly dollar-hedged basket; one means every leg points the same
    way, so the "spread" is really a directional position. A cointegrating vector is
    not required to be hedged, which is why this is worth checking before trading.
    """
    w = np.asarray(weights, dtype=float)
    gross = np.abs(w).sum()
    if w.ndim != 1 or not np.all(np.isfinite(w)) or gross == 0:
        raise ValueError(f"cannot measure net exposure of {weights!r}")
    return float(abs(w.sum()) / gross)


def _sequential_rank(stats: npt.NDArray[np.float64], critical: npt.NDArray[np.float64]) -> int:
    rank = 0
    for statistic, critical_value in zip(stats, critical):
        if statistic > critical_value:
            rank += 1
        else:
            break
    return rank


def _validated_panel(log_prices: pd.DataFrame | npt.ArrayLike) -> npt.NDArray[np.float64]:
    data = np.asarray(log_prices, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise DataError("cointegration tests need a 2-D panel with at least two series")
    if not np.all(np.isfinite(data)):
        raise DataError("log prices contain NaN or infinite values")
    return data


def johansen_test(
    log_prices: pd.DataFrame | npt.ArrayLike,
    significance: float = 0.05,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Johansen trace and maximum-eigenvalue tests on a panel of log prices.

    Parameters
    ----------
    log_prices
        T x k panel of log prices (the same scale the spread is built on).
    significance
        0.10, 0.05 or 0.01 (the levels statsmodels tabulates).
    det_order
        -1 no deterministic term, 0 constant, 1 linear trend.
    k_ar_diff
        Number of lagged differences in the VECM.
    """
    if significance not in CRITICAL_VALUE_COLUMN:
        raise ConfigError(f"significance must be one of {sorted(CRITICAL_VALUE_COLUMN)}, got {significance}")
    if det_order not in (-1, 0, 1):
        raise ConfigError(f"det_order must be -1, 0 or 1, got {det_order}")
    if not isinstance(k_ar_diff, (int, np.integer)) or isinstance(k_ar_diff, bool) or k_ar_diff < 0:
        raise ConfigError(f"k_ar_diff must be an integer >= 0, got {k_ar_diff!r}")

    data = _validated_panel(log_prices)
    n_obs, n_series = data.shape
    if n_series > MAX_JOHANSEN_SERIES:
        raise DataError(f"Johansen critical values are available for at most {MAX_JOHANSEN_SERIES} series")
    min_obs = max(50, 10 * n_series)
    if n_obs < min_obs:
        raise DataError(f"Johansen test needs at least {min_obs} observations, got {n_obs}")

    result = coint_johansen(data, det_order, int(k_ar_diff))
    column = CRITICAL_VALUE_COLUMN[significance]
    trace_stats = np.asarray(result.lr1, dtype=float)
    trace_critical = np.asarray(result.cvt[:, column], dtype=float)
    max_eig_stats = np.asarray(result.lr2, dtype=float)
    max_eig_critical = np.asarray(result.cvm[:, column], dtype=float)
    eigenvalues = np.asarray(result.eig, dtype=float)
    leading = int(np.argmax(eigenvalues))

    return JohansenResult(
        trace_stats=trace_stats,
        trace_critical=trace_critical,
        max_eig_stats=max_eig_stats,
        max_eig_critical=max_eig_critical,
        eigenvalues=eigenvalues,
        weights=normalize_weights(result.evec[:, leading]),
        rank_trace=_sequential_rank(trace_stats, trace_critical),
        rank_max_eig=_sequential_rank(max_eig_stats, max_eig_critical),
        significance=significance,
        n_obs=n_obs,
    )


def engle_granger_test(log_prices: pd.DataFrame) -> EngleGrangerResult:
    """Engle-Granger residual ADF test, regressing the first series on the others.

    Uses MacKinnon (2010) critical values for the number of series, which are more
    negative than plain ADF values because the residual comes from an estimated
    regression. The result depends on which series is the regressand, so it is a
    cross-check on the Johansen decision rather than a substitute for it.
    """
    data = _validated_panel(log_prices)
    statistic, p_value, critical = coint(data[:, 0], data[:, 1:], trend="c", autolag="aic")
    return EngleGrangerResult(float(statistic), float(p_value), float(critical[1]))
