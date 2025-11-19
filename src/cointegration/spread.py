"""
Module: Spread Calculator

Calculate and normalize mean-reverting spreads from cointegrated baskets.
Includes z-score normalization, half-life estimation, and Hurst exponent.

Classes
-------
SpreadCalculator
    Main interface for spread calculations

Functions
---------
calculate_zscore
    Calculate rolling z-score
calculate_half_life
    Estimate mean reversion half-life
calculate_hurst_exponent
    Calculate Hurst exponent

Notes
-----
All operations are vectorized for performance.

References
----------
.. [1] Ornstein, L. S., & Uhlenbeck, G. E. (1930). "On the theory of Brownian motion."
       Physical Review, 36(5), 823.

Author: Quantitative Research Team
Created: 2025-01-18
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.utils.config import ConfigManager
from src.utils.logger import StructuredLogger, timed_execution


class SpreadCalculator:
    """
    Calculate and normalize trading spreads.

    Methods
    -------
    calculate_spread(prices, weights)
        Calculate spread from prices and weights
    calculate_zscore(spread, lookback)
        Calculate rolling z-score
    calculate_half_life(spread)
        Estimate mean reversion half-life
    calculate_hurst_exponent(spread)
        Calculate Hurst exponent

    Examples
    --------
    >>> calculator = SpreadCalculator()
    >>> spread = calculator.calculate_spread(prices, weights)
    >>> zscore = calculator.calculate_zscore(spread, lookback=252)
    >>> half_life = calculator.calculate_half_life(spread)
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize spread calculator."""
        self.logger = StructuredLogger(__name__)

        if config is None:
            config = ConfigManager.load_config()

        self.config = config
        self.lookback_window = config.get("cointegration.spread.lookback_window", 252)
        self.half_life_min = config.get("cointegration.spread.half_life_min", 5)
        self.half_life_max = config.get("cointegration.spread.half_life_max", 60)

    @timed_execution
    def calculate_spread(
        self,
        prices: pd.DataFrame,
        weights: npt.NDArray[np.float64]
    ) -> pd.Series:
        """
        Calculate spread from prices and cointegrating vector.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        weights : np.ndarray
            Cointegrating vector weights

        Returns
        -------
        pd.Series
            Spread time series
        """
        if len(weights) != len(prices.columns):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of columns ({len(prices.columns)})"
            )

        # Use log prices
        log_prices = np.log(prices)

        # Calculate weighted sum
        spread = (log_prices * weights).sum(axis=1)

        self.logger.info(
            "Spread calculated",
            components=len(weights),
            length=len(spread)
        )

        return spread

    @timed_execution
    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling z-score of spread.

        Parameters
        ----------
        spread : pd.Series
            Spread time series
        lookback : Optional[int]
            Rolling window size (uses config default if None)

        Returns
        -------
        pd.Series
            Z-score time series

        Examples
        --------
        >>> zscore = calculator.calculate_zscore(spread, lookback=252)

        Notes
        -----
        Formula: z_t = (spread_t - μ_lookback) / σ_lookback
        Uses expanding window for first 'lookback' points to avoid NaNs
        """
        if lookback is None:
            lookback = self.lookback_window

        # Calculate rolling statistics
        rolling_mean = spread.rolling(lookback).mean()
        rolling_std = spread.rolling(lookback).std()

        # Fill initial NaNs with expanding window
        rolling_mean = rolling_mean.fillna(spread.expanding().mean())
        rolling_std = rolling_std.fillna(spread.expanding().std())

        # Calculate z-score
        zscore = (spread - rolling_mean) / rolling_std

        self.logger.info(
            "Z-score calculated",
            lookback=lookback,
            mean=round(zscore.mean(), 4),
            std=round(zscore.std(), 4)
        )

        return zscore

    @timed_execution
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Estimate mean reversion half-life using OU process.

        Parameters
        ----------
        spread : pd.Series
            Spread time series

        Returns
        -------
        float
            Half-life in units of spread frequency (days if daily data)

        Raises
        ------
        ValueError
            If θ ≥ 0 (no mean reversion detected)

        Examples
        --------
        >>> half_life = calculator.calculate_half_life(spread)
        >>> print(f"Half-life: {half_life:.1f} days")

        Notes
        -----
        Model: Δspread_t = θ(μ - spread_{t-1}) + ε_t
        Half-life: τ = -ln(2) / θ
        Reasonable range: 5-60 trading days
        """
        # Calculate lagged spread and differences
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()

        # Drop NaN values
        df = pd.DataFrame({
            'spread_lag': spread_lag,
            'spread_diff': spread_diff
        }).dropna()

        # OLS regression: Δspread = α + θ*spread_lag
        model = LinearRegression()
        model.fit(df[['spread_lag']], df['spread_diff'])

        theta = model.coef_[0]

        if theta >= 0:
            self.logger.warning(
                "No mean reversion detected",
                theta=round(theta, 6)
            )
            raise ValueError(f"No mean reversion detected: θ = {theta:.4f} ≥ 0")

        half_life = -np.log(2) / theta

        # Check if in reasonable range
        if half_life < self.half_life_min:
            self.logger.warning(
                "Half-life too short (mean reverts too quickly)",
                half_life=round(half_life, 2),
                min_threshold=self.half_life_min
            )
        elif half_life > self.half_life_max:
            self.logger.warning(
                "Half-life too long (mean reverts too slowly)",
                half_life=round(half_life, 2),
                max_threshold=self.half_life_max
            )

        self.logger.info(
            "Half-life calculated",
            half_life=round(half_life, 2),
            theta=round(theta, 6)
        )

        return half_life

    @timed_execution
    def calculate_hurst_exponent(self, spread: pd.Series, max_lag: int = 100) -> float:
        """
        Calculate Hurst exponent to measure mean reversion.

        Parameters
        ----------
        spread : pd.Series
            Spread time series
        max_lag : int
            Maximum lag for calculation

        Returns
        -------
        float
            Hurst exponent (H < 0.5: mean reverting, H = 0.5: random walk, H > 0.5: trending)

        Examples
        --------
        >>> hurst = calculator.calculate_hurst_exponent(spread)
        >>> if hurst < 0.5:
        ...     print("Spread is mean reverting")

        Notes
        -----
        H < 0.5: Mean reverting (anti-persistent)
        H = 0.5: Random walk (Brownian motion)
        H > 0.5: Trending (persistent)
        """
        lags = range(2, min(max_lag, len(spread) // 2))
        tau = []

        for lag in lags:
            # Calculate standard deviation of differences
            std = np.std([spread[i] - spread[i - lag] for i in range(lag, len(spread))])
            tau.append(std)

        # Linear regression of log(tau) on log(lags)
        model = LinearRegression()
        model.fit(np.log(list(lags)).reshape(-1, 1), np.log(tau))

        hurst = model.coef_[0]

        self.logger.info(
            "Hurst exponent calculated",
            hurst=round(hurst, 4),
            interpretation="mean reverting" if hurst < 0.5 else "trending" if hurst > 0.5 else "random walk"
        )

        return hurst


# Standalone utility functions

def calculate_zscore(spread: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Calculate rolling z-score of spread.

    Parameters
    ----------
    spread : pd.Series
        Spread time series
    lookback : int
        Rolling window size

    Returns
    -------
    pd.Series
        Z-score time series

    Examples
    --------
    >>> zscore = calculate_zscore(spread, lookback=252)
    """
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()

    # Fill initial NaNs with expanding window
    rolling_mean = rolling_mean.fillna(spread.expanding().mean())
    rolling_std = rolling_std.fillna(spread.expanding().std())

    zscore = (spread - rolling_mean) / rolling_std

    return zscore


def calculate_half_life(spread: pd.Series) -> float:
    """
    Estimate mean reversion half-life using OU process.

    Parameters
    ----------
    spread : pd.Series
        Spread time series

    Returns
    -------
    float
        Half-life in units of spread frequency

    Raises
    ------
    ValueError
        If θ ≥ 0 (no mean reversion detected)

    Examples
    --------
    >>> half_life = calculate_half_life(spread)
    """
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()

    df = pd.DataFrame({
        'spread_lag': spread_lag,
        'spread_diff': spread_diff
    }).dropna()

    model = LinearRegression()
    model.fit(df[['spread_lag']], df['spread_diff'])

    theta = model.coef_[0]

    if theta >= 0:
        raise ValueError(f"No mean reversion detected: θ = {theta:.4f} ≥ 0")

    half_life = -np.log(2) / theta

    return half_life


def calculate_hurst_exponent(spread: pd.Series, max_lag: int = 100) -> float:
    """
    Calculate Hurst exponent.

    Parameters
    ----------
    spread : pd.Series
        Spread time series
    max_lag : int
        Maximum lag

    Returns
    -------
    float
        Hurst exponent

    Examples
    --------
    >>> hurst = calculate_hurst_exponent(spread)
    """
    lags = range(2, min(max_lag, len(spread) // 2))
    tau = []

    for lag in lags:
        std = np.std([spread[i] - spread[i - lag] for i in range(lag, len(spread))])
        tau.append(std)

    model = LinearRegression()
    model.fit(np.log(list(lags)).reshape(-1, 1), np.log(tau))

    return model.coef_[0]
