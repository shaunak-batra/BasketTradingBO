"""
Module: Feature Engineer

Transform raw prices into trading features including log prices, returns,
rolling statistics, and spread construction.

Classes
-------
FeatureEngineer
    Main interface for feature engineering

Functions
---------
calculate_log_prices
    Convert prices to log scale
calculate_returns
    Calculate returns (simple or log)
calculate_rolling_statistics
    Calculate rolling mean, std, etc.
create_spread
    Construct spread from prices and weights

Notes
-----
All operations are vectorized using NumPy/Pandas for performance.

Author: Quantitative Research Team
Created: 2025-01-18
"""

from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.utils.logger import StructuredLogger, timed_execution


class FeatureEngineer:
    """
    Transform raw prices into trading features.

    Methods
    -------
    calculate_log_prices(df)
        Convert prices to log scale
    calculate_returns(df, method)
        Calculate returns
    calculate_rolling_statistics(df, windows)
        Calculate rolling statistics
    create_spread(prices, weights)
        Construct weighted spread

    Examples
    --------
    >>> engineer = FeatureEngineer()
    >>> log_prices = engineer.calculate_log_prices(prices)
    >>> returns = engineer.calculate_returns(prices, method='log')
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.logger = StructuredLogger(__name__)

    @timed_execution
    def calculate_log_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert prices to log scale.

        Parameters
        ----------
        df : pd.DataFrame
            Price data

        Returns
        -------
        pd.DataFrame
            Log-transformed prices

        Examples
        --------
        >>> prices = pd.DataFrame({'AAPL': [100, 101, 102]})
        >>> log_prices = engineer.calculate_log_prices(prices)
        """
        log_prices = np.log(df)

        self.logger.info(
            "Log prices calculated",
            rows=len(log_prices),
            columns=len(log_prices.columns)
        )

        return log_prices

    @timed_execution
    def calculate_returns(
        self,
        df: pd.DataFrame,
        method: str = "log",
        periods: int = 1
    ) -> pd.DataFrame:
        """
        Calculate returns.

        Parameters
        ----------
        df : pd.DataFrame
            Price data
        method : str
            Return calculation method ('log' or 'simple')
        periods : int
            Number of periods for return calculation

        Returns
        -------
        pd.DataFrame
            Returns

        Examples
        --------
        >>> returns = engineer.calculate_returns(prices, method='log')
        >>> daily_returns = engineer.calculate_returns(prices, periods=1)
        """
        if method == "log":
            returns = np.log(df / df.shift(periods))
        elif method == "simple":
            returns = df.pct_change(periods)
        else:
            raise ValueError(f"Unknown return method: {method}")

        self.logger.info(
            "Returns calculated",
            method=method,
            periods=periods,
            rows=len(returns),
            columns=len(returns.columns)
        )

        return returns

    @timed_execution
    def calculate_rolling_statistics(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate rolling statistics for multiple windows.

        Parameters
        ----------
        df : pd.DataFrame
            Price or return data
        windows : List[int]
            List of window sizes

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with keys like 'mean_20', 'std_20', etc.

        Examples
        --------
        >>> stats = engineer.calculate_rolling_statistics(prices, windows=[20, 50])
        >>> ma_20 = stats['mean_20']
        """
        results = {}

        for window in windows:
            # Rolling mean
            results[f'mean_{window}'] = df.rolling(window).mean()

            # Rolling standard deviation
            results[f'std_{window}'] = df.rolling(window).std()

            # Rolling min/max
            results[f'min_{window}'] = df.rolling(window).min()
            results[f'max_{window}'] = df.rolling(window).max()

        self.logger.info(
            "Rolling statistics calculated",
            windows=windows,
            features=len(results)
        )

        return results

    @timed_execution
    def create_spread(
        self,
        prices: pd.DataFrame,
        weights: npt.NDArray[np.float64]
    ) -> pd.Series:
        """
        Construct weighted spread from prices.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data for basket components
        weights : npt.NDArray[np.float64]
            Weights for each component (cointegrating vector)

        Returns
        -------
        pd.Series
            Spread time series

        Examples
        --------
        >>> weights = np.array([1.0, -0.5, -0.3])
        >>> spread = engineer.create_spread(prices, weights)

        Notes
        -----
        Spread is calculated as: spread = Σ(w_i * log(price_i))
        """
        if len(weights) != len(prices.columns):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of price columns ({len(prices.columns)})"
            )

        # Use log prices for spread construction
        log_prices = np.log(prices)

        # Calculate weighted sum
        spread = (log_prices * weights).sum(axis=1)

        self.logger.info(
            "Spread created",
            components=len(weights),
            rows=len(spread)
        )

        return spread

    def calculate_technical_indicators(
        self,
        prices: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate common technical indicators.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        indicators : Optional[List[str]]
            List of indicators to calculate
            Options: ['rsi', 'macd', 'bollinger', 'atr']

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of indicator DataFrames

        Examples
        --------
        >>> indicators = engineer.calculate_technical_indicators(
        ...     prices, indicators=['rsi', 'bollinger']
        ... )
        """
        if indicators is None:
            indicators = ['rsi', 'bollinger']

        results = {}

        for indicator in indicators:
            if indicator == 'rsi':
                results['rsi'] = self._calculate_rsi(prices)
            elif indicator == 'bollinger':
                results['bollinger_upper'], results['bollinger_lower'] = \
                    self._calculate_bollinger_bands(prices)
            elif indicator == 'macd':
                results['macd'], results['macd_signal'] = self._calculate_macd(prices)
            elif indicator == 'atr':
                # Requires high/low/close data
                pass
            else:
                self.logger.warning(f"Unknown indicator: {indicator}")

        return results

    def _calculate_rsi(self, prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        period : int
            RSI period

        Returns
        -------
        pd.DataFrame
            RSI values (0-100)
        """
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bollinger_bands(
        self,
        prices: pd.DataFrame,
        period: int = 20,
        num_std: float = 2.0
    ) -> tuple:
        """
        Calculate Bollinger Bands.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        period : int
            Moving average period
        num_std : float
            Number of standard deviations

        Returns
        -------
        tuple
            (upper_band, lower_band)
        """
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)

        return upper_band, lower_band

    def _calculate_macd(
        self,
        prices: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        fast : int
            Fast EMA period
        slow : int
            Slow EMA period
        signal : int
            Signal line period

        Returns
        -------
        tuple
            (macd_line, signal_line)
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        return macd_line, signal_line


# Standalone utility functions

def calculate_log_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices to log scale.

    Parameters
    ----------
    df : pd.DataFrame
        Price data

    Returns
    -------
    pd.DataFrame
        Log-transformed prices
    """
    return np.log(df)


def calculate_returns(
    df: pd.DataFrame,
    method: str = "log",
    periods: int = 1
) -> pd.DataFrame:
    """
    Calculate returns.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    method : str
        Return calculation method ('log' or 'simple')
    periods : int
        Number of periods

    Returns
    -------
    pd.DataFrame
        Returns
    """
    if method == "log":
        return np.log(df / df.shift(periods))
    elif method == "simple":
        return df.pct_change(periods)
    else:
        raise ValueError(f"Unknown return method: {method}")


def create_spread(
    prices: pd.DataFrame,
    weights: npt.NDArray[np.float64]
) -> pd.Series:
    """
    Construct weighted spread from prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data for basket components
    weights : npt.NDArray[np.float64]
        Weights for each component

    Returns
    -------
    pd.Series
        Spread time series
    """
    log_prices = np.log(prices)
    spread = (log_prices * weights).sum(axis=1)
    return spread
