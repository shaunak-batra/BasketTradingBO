"""Value-at-Risk and Expected Shortfall of daily returns.

Both are reported as positive fractions of equity (a loss of 1% is 0.01). These are
risk *analytics* computed on realised strategy returns. They do not drive position
sizing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MIN_OBSERVATIONS = 30


@dataclass(frozen=True)
class RiskEstimate:
    method: str
    confidence: float
    var: float
    expected_shortfall: float | None


def _prepare(returns: pd.Series | np.ndarray, confidence: float) -> np.ndarray:
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence}")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < MIN_OBSERVATIONS:
        raise ValueError(f"need at least {MIN_OBSERVATIONS} finite returns, got {len(r)}")
    return r


def historical_var(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> RiskEstimate:
    """Empirical quantile VaR; ES is the mean return at or below that quantile."""
    r = _prepare(returns, confidence)
    quantile = float(np.quantile(r, 1.0 - confidence))
    tail = r[r <= quantile]
    return RiskEstimate("historical", confidence, -quantile, float(-tail.mean()))


def parametric_var(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> RiskEstimate:
    """Gaussian VaR and ES: VaR = -(mu + sigma z), ES = -mu + sigma * phi(z) / (1 - c), z = Phi^-1(1 - c)."""
    r = _prepare(returns, confidence)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    z = stats.norm.ppf(1.0 - confidence)
    var = -(mu + sigma * z)
    expected_shortfall = -mu + sigma * stats.norm.pdf(z) / (1.0 - confidence)
    return RiskEstimate("parametric", confidence, float(var), float(expected_shortfall))


def cornish_fisher_quantile(z: float, skewness: float, excess_kurtosis: float) -> float:
    """Cornish-Fisher adjusted standard-normal quantile."""
    return (
        z
        + (z**2 - 1) * skewness / 6
        + (z**3 - 3 * z) * excess_kurtosis / 24
        - (2 * z**3 - 5 * z) * skewness**2 / 36
    )


def cornish_fisher_var(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> RiskEstimate:
    """VaR with a quantile adjusted for sample skewness and excess kurtosis (no closed-form ES)."""
    r = _prepare(returns, confidence)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    skewness = float(stats.skew(r, bias=False))
    excess_kurtosis = float(stats.kurtosis(r, fisher=True, bias=False))
    z = cornish_fisher_quantile(stats.norm.ppf(1.0 - confidence), skewness, excess_kurtosis)
    return RiskEstimate("cornish_fisher", confidence, float(-(mu + sigma * z)), None)


def risk_table(returns: pd.Series | np.ndarray, confidences: tuple[float, ...] = (0.95, 0.99)) -> list[RiskEstimate]:
    """VaR/ES by every method at each confidence level."""
    estimates: list[RiskEstimate] = []
    for confidence in confidences:
        estimates.append(historical_var(returns, confidence))
        estimates.append(parametric_var(returns, confidence))
        estimates.append(cornish_fisher_var(returns, confidence))
    return estimates
