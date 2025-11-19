"""
Module: Backtesting Engine

Vectorized backtesting for basket trading strategies with transaction costs.

Classes
-------
Backtester
    Main backtesting engine with vectorized operations

BacktestResult
    Container for backtest results

Functions
---------
calculate_returns
    Calculate portfolio returns from positions and prices

Author: Quantitative Research Team
Created: 2025-01-18
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.backtesting.metrics import PerformanceMetrics, calculate_performance_metrics
from src.strategy.portfolio import PortfolioManager
from src.strategy.signals import SignalGenerator
from src.utils.logger import StructuredLogger, timed_execution


@dataclass
class BacktestResult:
    """
    Container for backtest results.

    Attributes
    ----------
    portfolio_value : pd.Series
        Portfolio value over time
    returns : pd.Series
        Daily returns
    positions : pd.DataFrame
        Position sizes (shares) over time
    signals : pd.Series
        Trading signals
    metrics : PerformanceMetrics
        Performance metrics
    transaction_costs : pd.Series
        Transaction costs over time
    trades : pd.DataFrame
        Trade log
    """
    portfolio_value: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    signals: pd.Series
    metrics: PerformanceMetrics
    transaction_costs: pd.Series
    trades: pd.DataFrame


class Backtester:
    """
    Vectorized backtesting engine for basket trading strategies.

    Attributes
    ----------
    initial_capital : float
        Initial portfolio capital
    commission : float
        Commission rate per trade (fraction)
    slippage : float
        Slippage per trade (fraction)
    position_size : float
        Position size as fraction of portfolio

    Methods
    -------
    run_backtest(prices, signals, weights)
        Run vectorized backtest
    calculate_transaction_costs(positions, prices)
        Calculate transaction costs
    generate_trade_log(positions, prices, signals)
        Generate trade execution log

    Examples
    --------
    >>> backtester = Backtester(initial_capital=100000)
    >>> result = backtester.run_backtest(prices, signals, weights)
    >>> print(result.metrics.sharpe_ratio)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 10 bps
        slippage: float = 0.0005,  # 5 bps
        position_size: float = 0.20
    ):
        """
        Initialize backtester.

        Parameters
        ----------
        initial_capital : float
            Initial capital
        commission : float
            Commission rate (fraction, e.g., 0.001 = 10 bps)
        slippage : float
            Slippage rate (fraction, e.g., 0.0005 = 5 bps)
        position_size : float
            Position size as fraction of portfolio
        """
        self.logger = StructuredLogger(__name__)

        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size

        self.logger.info(
            "Backtester initialized",
            initial_capital=initial_capital,
            commission_bps=commission * 10000,
            slippage_bps=slippage * 10000,
            position_size=position_size
        )

    @timed_execution
    def run_backtest(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        weights: npt.NDArray[np.float64]
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data for basket components
        signals : pd.Series
            Trading signals (1, -1, 0)
        weights : np.ndarray
            Basket weights (cointegrating vector)

        Returns
        -------
        BacktestResult
            Complete backtest results

        Examples
        --------
        >>> result = backtester.run_backtest(prices, signals, weights)

        Notes
        -----
        Vectorized implementation avoids Python loops for performance.
        Transaction costs calculated from position changes.
        """
        self.logger.info(
            "Starting backtest",
            num_periods=len(prices),
            num_assets=len(prices.columns)
        )

        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(
            initial_capital=self.initial_capital,
            max_position_size=self.position_size
        )

        # Allocate capital to get positions
        positions = portfolio_manager.allocate_capital(
            signals=signals,
            weights=weights,
            prices=prices,
            position_size=self.position_size
        )

        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs(positions, prices)

        # Calculate portfolio value
        portfolio_value = self._calculate_portfolio_value(
            positions, prices, transaction_costs
        )

        # Calculate returns
        returns = portfolio_value.pct_change().fillna(0)

        # Calculate performance metrics
        metrics = calculate_performance_metrics(returns)

        # Generate trade log
        trades = self.generate_trade_log(positions, prices, signals)

        self.logger.info(
            "Backtest completed",
            final_value=float(portfolio_value.iloc[-1]),
            total_return_pct=round((portfolio_value.iloc[-1] / self.initial_capital - 1) * 100, 2),
            num_trades=len(trades),
            total_transaction_costs=round(float(transaction_costs.sum()), 2)
        )

        return BacktestResult(
            portfolio_value=portfolio_value,
            returns=returns,
            positions=positions,
            signals=signals,
            metrics=metrics,
            transaction_costs=transaction_costs,
            trades=trades
        )

    def calculate_transaction_costs(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate transaction costs from position changes.

        Parameters
        ----------
        positions : pd.DataFrame
            Position sizes (shares) over time
        prices : pd.DataFrame
            Price data

        Returns
        -------
        pd.Series
            Transaction costs per period

        Examples
        --------
        >>> costs = backtester.calculate_transaction_costs(positions, prices)

        Notes
        -----
        Cost = (|Δposition| * price) * (commission + slippage)
        """
        # Calculate position changes
        position_changes = positions.diff().fillna(positions)

        # Calculate notional value of trades
        trade_notional = (position_changes.abs() * prices).sum(axis=1)

        # Apply commission and slippage
        total_cost_rate = self.commission + self.slippage
        transaction_costs = trade_notional * total_cost_rate

        return transaction_costs

    def generate_trade_log(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        signals: pd.Series
    ) -> pd.DataFrame:
        """
        Generate detailed trade execution log.

        Parameters
        ----------
        positions : pd.DataFrame
            Position sizes (shares)
        prices : pd.DataFrame
            Price data
        signals : pd.Series
            Trading signals

        Returns
        -------
        pd.DataFrame
            Trade log with columns: timestamp, action, signal, notional

        Examples
        --------
        >>> trades = backtester.generate_trade_log(positions, prices, signals)
        """
        # Detect position changes
        position_changes = positions.diff().fillna(positions)

        # Find rows where trades occurred
        trade_mask = (position_changes != 0).any(axis=1)
        trade_indices = trade_mask[trade_mask].index

        trades = []
        for idx in trade_indices:
            notional = (position_changes.loc[idx].abs() * prices.loc[idx]).sum()

            if notional > 0:  # Only record actual trades
                action = "OPEN" if signals.loc[idx] != 0 else "CLOSE"
                trades.append({
                    "timestamp": idx,
                    "action": action,
                    "signal": int(signals.loc[idx]),
                    "notional": float(notional)
                })

        trade_log = pd.DataFrame(trades)

        if len(trade_log) > 0:
            self.logger.info(
                "Trade log generated",
                num_trades=len(trade_log),
                total_notional=round(float(trade_log['notional'].sum()), 2)
            )
        else:
            self.logger.warning("No trades executed in backtest")

        return trade_log

    def _calculate_portfolio_value(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        transaction_costs: pd.Series
    ) -> pd.Series:
        """
        Calculate portfolio value accounting for transaction costs.

        Parameters
        ----------
        positions : pd.DataFrame
            Position sizes (shares)
        prices : pd.DataFrame
            Price data
        transaction_costs : pd.Series
            Transaction costs per period

        Returns
        -------
        pd.Series
            Portfolio value over time
        """
        # Calculate position values
        position_values = (positions * prices).sum(axis=1)

        # Calculate cumulative transaction costs
        cumulative_costs = transaction_costs.cumsum()

        # Portfolio value = initial capital + position P&L - transaction costs
        portfolio_value = self.initial_capital + position_values - cumulative_costs

        return portfolio_value


# Standalone utility function

def calculate_returns(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 100000.0
) -> pd.Series:
    """
    Calculate portfolio returns from positions and prices.

    Parameters
    ----------
    positions : pd.DataFrame
        Position sizes (shares) over time
    prices : pd.DataFrame
        Price data
    initial_capital : float
        Initial capital

    Returns
    -------
    pd.Series
        Daily returns

    Examples
    --------
    >>> returns = calculate_returns(positions, prices, initial_capital=100000)

    Notes
    -----
    Returns calculated as percentage change in portfolio value.
    """
    # Calculate position values
    position_values = (positions * prices).sum(axis=1)

    # Portfolio value
    portfolio_value = initial_capital + position_values

    # Calculate returns
    returns = portfolio_value.pct_change().fillna(0)

    return returns
