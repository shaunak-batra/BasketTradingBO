"""
Module: Performance Metrics

Calculate comprehensive performance metrics for trading strategies.

Classes
-------
PerformanceMetrics
    Container for strategy performance metrics

Functions
---------
calculate_performance_metrics
    Calculate all performance metrics from returns

Author: Quantitative Research Team
Created: 2025-01-18
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """
    Strategy performance metrics container.

    Attributes
    ----------
    total_return : float
        Total return over period
    annualized_return : float
        Annualized return
    sharpe_ratio : float
        Sharpe ratio (annualized, assuming Rf=0)
    sortino_ratio : float
        Sortino ratio (downside deviation)
    max_drawdown : float
        Maximum drawdown
    calmar_ratio : float
        Calmar ratio (return / max DD)
    win_rate : float
        Fraction of profitable days
    profit_factor : float
        Ratio of total wins to total losses
    num_trades : int
        Number of trades
    avg_trade_duration : float
        Average holding period
    """
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_duration: float


def calculate_performance_metrics(returns: pd.Series) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily returns

    Returns
    -------
    PerformanceMetrics
        Performance metrics

    Notes
    -----
    Assumes 252 trading days per year for annualization.

    Metrics:
    - Sharpe Ratio: (E[R] - R_f) / σ(R), annualized, assume R_f = 0
    - Sortino Ratio: (E[R] - R_f) / σ_downside(R)
    - Max Drawdown: max(peak - trough) / peak
    - Calmar Ratio: Annualized Return / |Max Drawdown|
    - Win Rate: # winning days / # trading days
    - Profit Factor: Σ(positive returns) / |Σ(negative returns)|
    """
    ann_factor = np.sqrt(252)

    # Total and annualized return
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + returns.mean()) ** 252 - 1

    # Sharpe Ratio
    sharpe = (returns.mean() * 252) / (returns.std() * ann_factor) if returns.std() > 0 else 0.0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino = (returns.mean() * 252) / (downside_std * ann_factor) if downside_std > 0 else 0.0

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Win Rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

    # Profit Factor
    positive_sum = returns[returns > 0].sum()
    negative_sum = abs(returns[returns < 0].sum())
    profit_factor = positive_sum / negative_sum if negative_sum > 0 else 0.0

    # Number of trades (approximate from return changes)
    num_trades = (returns != 0).sum()

    # Average trade duration (placeholder)
    avg_trade_duration = 1.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=num_trades,
        avg_trade_duration=avg_trade_duration
    )
