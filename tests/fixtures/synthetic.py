"""Synthetic price data with known statistical properties."""

from __future__ import annotations

import numpy as np
import pandas as pd


def business_days(n: int, start: str = "2015-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, name="date")


def ar1_process(n: int, half_life: float, sigma: float, seed: int, mean: float = 0.0) -> np.ndarray:
    """AR(1) ``x_t = mean + phi (x_{t-1} - mean) + sigma e_t`` with ``phi = 0.5 ** (1 / half_life)``.

    ``phi`` is chosen so the expected deviation halves after exactly ``half_life`` steps.
    """
    rng = np.random.default_rng(seed)
    phi = 0.5 ** (1.0 / half_life)
    shocks = rng.standard_normal(n) * sigma
    x = np.empty(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = mean + phi * (x[t - 1] - mean) + shocks[t]
    return x


def cointegrated_prices(
    n: int = 1500,
    weights: tuple[float, ...] = (1.0, -0.6, -0.4),
    half_life: float = 10.0,
    spread_sigma: float = 0.01,
    trend_sigma: float = 0.015,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Prices whose log basket ``sum_i w_i log P_i`` follows an AR(1) with the given half-life.

    Assets 1..k-1 are independent log random walks; asset 0 is solved so the spread is
    exactly the AR(1) process (plus a constant). Returns prices and the true weights.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=float)
    k = len(w)
    logs = np.empty((n, k))
    logs[:, 1:] = np.log(100.0) + np.cumsum(rng.standard_normal((n, k - 1)) * trend_sigma, axis=0)
    spread = ar1_process(n, half_life, spread_sigma, seed=seed + 1)
    logs[:, 0] = (spread - logs[:, 1:] @ w[1:]) / w[0]
    logs[:, 0] += np.log(100.0) - logs[0, 0]
    prices = pd.DataFrame(np.exp(logs), index=business_days(n), columns=[f"A{i}" for i in range(k)])
    return prices, w


def random_walk_prices(
    n: int = 1500, k: int = 3, sigma: float = 0.015, seed: int = 0, drift: float = 0.0
) -> pd.DataFrame:
    """Independent geometric random walks (not cointegrated), with an optional daily log drift."""
    rng = np.random.default_rng(seed)
    logs = np.log(100.0) + np.cumsum(drift + rng.standard_normal((n, k)) * sigma, axis=0)
    return pd.DataFrame(np.exp(logs), index=business_days(n), columns=[f"R{i}" for i in range(k)])
