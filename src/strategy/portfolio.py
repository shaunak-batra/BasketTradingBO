"""
Module: Portfolio Manager

Manage portfolio positions, capital allocation, and risk management.

Classes
-------
PortfolioManager
    Main interface for portfolio management

Functions
---------
allocate_capital
    Allocate capital to basket components

Author: Quantitative Research Team
Created: 2025-01-18
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.utils.config import ConfigManager
from src.utils.exceptions import PositionSizingException
from src.utils.logger import StructuredLogger, timed_execution


class PortfolioManager:
    """
    Manage portfolio positions, capital allocation, and risk.

    Attributes
    ----------
    initial_capital : float
        Initial portfolio capital
    max_positions : int
        Maximum number of concurrent positions
    max_position_size : float
        Maximum position size as fraction of portfolio
    rebalance_frequency : str
        How often to rebalance positions

    Methods
    -------
    allocate_capital(signals, weights, prices)
        Allocate capital based on signals
    calculate_portfolio_value(positions, prices)
        Calculate current portfolio value
    rebalance_positions(target, current)
        Calculate rebalancing trades

    Examples
    --------
    >>> manager = PortfolioManager(initial_capital=100000)
    >>> positions = manager.allocate_capital(signals, weights, prices)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_positions: int = 5,
        max_position_size: float = 0.20,
        rebalance_frequency: str = "daily"
    ):
        """
        Initialize portfolio manager.

        Parameters
        ----------
        initial_capital : float
            Initial capital
        max_positions : int
            Maximum concurrent positions
        max_position_size : float
            Maximum position size (fraction of portfolio)
        rebalance_frequency : str
            Rebalancing frequency
        """
        self.logger = StructuredLogger(__name__)

        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.rebalance_frequency = rebalance_frequency

        self.current_capital = initial_capital

        self.logger.info(
            "PortfolioManager initialized",
            initial_capital=initial_capital,
            max_position_size=max_position_size
        )

    @timed_execution
    def allocate_capital(
        self,
        signals: pd.Series,
        weights: npt.NDArray[np.float64],
        prices: pd.DataFrame,
        position_size: float = 0.20
    ) -> pd.DataFrame:
        """
        Allocate capital to basket components based on signals.

        Parameters
        ----------
        signals : pd.Series
            Trading signals
        weights : np.ndarray
            Basket weights (cointegrating vector)
        prices : pd.DataFrame
            Price data for basket components
        position_size : float
            Position size as fraction of portfolio

        Returns
        -------
        pd.DataFrame
            Position sizes (shares) for each component over time

        Examples
        --------
        >>> positions = manager.allocate_capital(signals, weights, prices)
        """
        if position_size > self.max_position_size:
            raise PositionSizingException(
                "Position size exceeds maximum",
                context={
                    "requested": position_size,
                    "max_allowed": self.max_position_size
                }
            )

        # Initialize positions DataFrame
        positions = pd.DataFrame(
            0.0,
            index=prices.index,
            columns=prices.columns
        )

        portfolio_value = self.initial_capital

        for t in range(len(prices)):
            signal = signals.iloc[t]

            if signal != 0:
                # Calculate shares for each component
                shares = allocate_capital(
                    signal=signal,
                    basket_weights=weights,
                    prices=prices.iloc[t],
                    portfolio_value=portfolio_value,
                    position_size=position_size
                )

                positions.iloc[t] = shares
            else:
                # No position
                positions.iloc[t] = 0

            # Forward fill positions (hold until next signal)
            if t < len(prices) - 1:
                positions.iloc[t+1] = positions.iloc[t]

        self.logger.info(
            "Capital allocated",
            total_rows=len(positions),
            position_size=position_size
        )

        return positions

    def calculate_portfolio_value(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate portfolio value over time.

        Parameters
        ----------
        positions : pd.DataFrame
            Position sizes (shares)
        prices : pd.DataFrame
            Price data

        Returns
        -------
        pd.Series
            Portfolio value time series

        Examples
        --------
        >>> portfolio_value = manager.calculate_portfolio_value(positions, prices)
        """
        # Calculate position values
        position_values = positions * prices

        # Sum across components
        total_value = position_values.sum(axis=1)

        # Add initial capital
        portfolio_value = self.initial_capital + total_value

        self.logger.info(
            "Portfolio value calculated",
            initial=self.initial_capital,
            final=float(portfolio_value.iloc[-1]),
            return_pct=round((portfolio_value.iloc[-1] / self.initial_capital - 1) * 100, 2)
        )

        return portfolio_value


# Standalone utility function

def allocate_capital(
    signal: int,
    basket_weights: npt.NDArray[np.float64],
    prices: pd.Series,
    portfolio_value: float,
    position_size: float = 0.20
) -> npt.NDArray[np.float64]:
    """
    Allocate capital to basket components.

    Parameters
    ----------
    signal : int
        Trading signal (1, -1, or 0)
    basket_weights : np.ndarray
        Cointegrating vector weights
    prices : pd.Series
        Current prices of basket components
    portfolio_value : float
        Total portfolio value
    position_size : float
        Fraction of portfolio to allocate

    Returns
    -------
    np.ndarray
        Number of shares for each component

    Examples
    --------
    >>> shares = allocate_capital(1, weights, prices, 100000, position_size=0.2)

    Notes
    -----
    Logic:
        1. Calculate target dollar allocation: portfolio_value * position_size
        2. Distribute allocation according to basket_weights
        3. Convert dollars to shares: shares = dollars / price
        4. Round to whole shares (long: floor, short: ceil)
    """
    if signal == 0:
        return np.zeros(len(prices))

    # Calculate target allocation
    target_allocation = portfolio_value * position_size * signal

    # Distribute according to weights (normalize weights)
    weight_sum = np.abs(basket_weights).sum()
    component_allocations = target_allocation * (basket_weights / weight_sum)

    # Convert to shares
    shares = component_allocations / prices.values

    # Round shares
    if signal > 0:
        shares = np.floor(shares)
    else:
        shares = np.ceil(shares)

    return shares
