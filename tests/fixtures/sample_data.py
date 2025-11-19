"""
Sample data fixtures for testing.

Provides synthetic and real-world-like data for unit and integration tests.
"""

import numpy as np
import pandas as pd


def generate_synthetic_prices(
    n_days: int = 252,
    n_assets: int = 3,
    start_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic price data.

    Parameters
    ----------
    n_days : int
        Number of trading days
    n_assets : int
        Number of assets
    start_price : float
        Starting price for all assets
    volatility : float
        Daily volatility
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Synthetic price data
    """
    np.random.seed(seed)

    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    tickers = [f'ASSET_{i}' for i in range(n_assets)]

    # Generate random returns
    returns = np.random.randn(n_days, n_assets) * volatility

    # Calculate cumulative prices
    prices = start_price * np.exp(np.cumsum(returns, axis=0))

    df = pd.DataFrame(prices, index=dates, columns=tickers)

    return df


def generate_cointegrated_prices(
    n_assets: int = 3,
    n_periods: int = 252,
    cointegration_strength: float = 0.95,
    noise_level: float = 0.01,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate cointegrated price series for multiple assets.

    Creates multiple price series that are cointegrated through a common factor.

    Parameters
    ----------
    n_assets : int
        Number of assets to generate (minimum 2)
    n_periods : int
        Number of trading days
    cointegration_strength : float
        Strength of cointegration relationship (0-1)
    noise_level : float
        Amount of noise to add (keep small for strong cointegration)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Cointegrated price series

    Examples
    --------
    >>> prices = generate_cointegrated_prices(n_assets=3, n_periods=500, seed=42)
    >>> print(prices.shape)
    (500, 3)
    """
    if n_assets < 2:
        raise ValueError("n_assets must be at least 2")

    np.random.seed(seed)

    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')

    # Generate common factor (the cointegrating relationship)
    # Use stronger random walk
    common_factor = np.cumsum(np.random.randn(n_periods) * 2.0)

    # Generate price series
    prices_dict = {}

    for i in range(n_assets):
        # Each asset follows the common factor with different scaling
        # and very small individual noise component
        asset_name = f'ASSET_{i}'

        # Different starting prices and betas to common factor
        start_price = 100 * (i + 1)
        beta = 1.0 + (i * 0.3)  # Different sensitivities to common factor

        # Price = start + beta * common_factor + minimal individual noise
        # Keep noise very small relative to common factor for strong cointegration
        individual_noise = noise_level * np.cumsum(np.random.randn(n_periods))

        prices_dict[asset_name] = start_price + \
            cointegration_strength * beta * common_factor + \
            individual_noise

    df = pd.DataFrame(prices_dict, index=dates)

    return df


def generate_mean_reverting_spread(
    n_days: int = 252,
    mean: float = 0.0,
    half_life: float = 20.0,
    volatility: float = 1.0,
    seed: int = 42
) -> pd.Series:
    """
    Generate mean-reverting spread using Ornstein-Uhlenbeck process.

    Parameters
    ----------
    n_days : int
        Number of trading days
    mean : float
        Long-term mean
    half_life : float
        Mean reversion half-life in days
    volatility : float
        Volatility of the process
    seed : int
        Random seed

    Returns
    -------
    pd.Series
        Mean-reverting spread
    """
    np.random.seed(seed)

    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

    # Calculate mean reversion speed
    theta = -np.log(2) / half_life

    # Generate OU process
    spread = np.zeros(n_days)
    spread[0] = mean

    for t in range(1, n_days):
        spread[t] = spread[t-1] + theta * (mean - spread[t-1]) + \
                    volatility * np.random.randn()

    series = pd.Series(spread, index=dates, name='spread')

    return series


def load_sample_financial_basket() -> pd.DataFrame:
    """
    Load sample financial basket prices.

    Returns
    -------
    pd.DataFrame
        Sample price data for financial basket
    """
    # Generate realistic financial sector prices
    n_days = 500
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

    np.random.seed(42)

    # Base factor (market/sector movement)
    market_factor = np.cumsum(np.random.randn(n_days) * 0.01)

    # Individual stocks with beta to market
    prices = pd.DataFrame({
        'JPM': 100 * np.exp(market_factor * 1.2 + np.cumsum(np.random.randn(n_days) * 0.015)),
        'BAC': 25 * np.exp(market_factor * 1.3 + np.cumsum(np.random.randn(n_days) * 0.018)),
        'GS': 200 * np.exp(market_factor * 1.1 + np.cumsum(np.random.randn(n_days) * 0.016)),
        'MS': 50 * np.exp(market_factor * 1.15 + np.cumsum(np.random.randn(n_days) * 0.017)),
        'C': 50 * np.exp(market_factor * 1.25 + np.cumsum(np.random.randn(n_days) * 0.019)),
        'XLF': 30 * np.exp(market_factor * 1.0 + np.cumsum(np.random.randn(n_days) * 0.012))
    }, index=dates)

    return prices


def generate_price_with_outliers(
    n_days: int = 252,
    n_outliers: int = 5,
    outlier_magnitude: float = 0.20,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate price data with outliers.

    Parameters
    ----------
    n_days : int
        Number of trading days
    n_outliers : int
        Number of outliers to inject
    outlier_magnitude : float
        Magnitude of outlier jumps (as fraction)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Price data with outliers
    """
    np.random.seed(seed)

    # Generate base prices
    df = generate_synthetic_prices(n_days=n_days, n_assets=2, seed=seed)

    # Inject outliers
    outlier_indices = np.random.choice(n_days, n_outliers, replace=False)

    for idx in outlier_indices:
        # Random jump up or down
        jump = (1 + outlier_magnitude) if np.random.rand() > 0.5 else (1 - outlier_magnitude)
        df.iloc[idx:] *= jump

    return df


def generate_price_with_missing_values(
    n_days: int = 252,
    missing_ratio: float = 0.10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate price data with missing values.

    Parameters
    ----------
    n_days : int
        Number of trading days
    missing_ratio : float
        Ratio of missing values (0-1)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Price data with missing values
    """
    np.random.seed(seed)

    # Generate base prices
    df = generate_synthetic_prices(n_days=n_days, n_assets=2, seed=seed)

    # Inject missing values
    n_missing = int(df.size * missing_ratio)
    missing_indices = np.random.choice(df.size, n_missing, replace=False)

    # Convert to flat array, set missing, reshape
    flat_values = df.values.flatten()
    flat_values[missing_indices] = np.nan

    df.values[:] = flat_values.reshape(df.shape)

    return df
