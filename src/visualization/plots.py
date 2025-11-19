"""
Module: Visualization Plots

Plotting functions for backtesting results, optimization convergence, and risk metrics.

Functions
---------
plot_portfolio_performance
    Plot portfolio value and returns over time
plot_spread_zscore
    Plot spread and z-score with entry/exit thresholds
plot_optimization_convergence
    Plot Bayesian optimization convergence
plot_var_distribution
    Plot VaR distribution and thresholds
plot_correlation_matrix
    Plot asset correlation heatmap

Author: Quantitative Research Team
Created: 2025-01-18
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from src.utils.logger import StructuredLogger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_portfolio_performance(
    portfolio_value: pd.Series,
    returns: pd.Series,
    signals: pd.Series,
    save_path: Optional[str] = None
) -> None:
    """
    Plot portfolio performance over time.

    Parameters
    ----------
    portfolio_value : pd.Series
        Portfolio value time series
    returns : pd.Series
        Daily returns
    signals : pd.Series
        Trading signals
    save_path : Optional[str]
        Path to save figure (if None, displays plot)

    Examples
    --------
    >>> plot_portfolio_performance(result.portfolio_value, result.returns, result.signals)
    """
    logger = StructuredLogger(__name__)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Portfolio value
    axes[0].plot(portfolio_value.index, portfolio_value.values, linewidth=2, color='#2E86AB')
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    axes[1].plot(cumulative_returns.index, cumulative_returns.values * 100, linewidth=2, color='#06A77D')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Cumulative Return (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Trading signals
    axes[2].plot(signals.index, signals.values, linewidth=1.5, color='#D62828', alpha=0.7)
    axes[2].fill_between(signals.index, 0, signals.values, where=(signals > 0), color='green', alpha=0.3, label='Long')
    axes[2].fill_between(signals.index, 0, signals.values, where=(signals < 0), color='red', alpha=0.3, label='Short')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2].set_title('Trading Signals', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Signal', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Portfolio performance plot saved", path=save_path)
        plt.close()
    else:
        plt.show()


def plot_spread_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    signals: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot spread and z-score with trading thresholds.

    Parameters
    ----------
    spread : pd.Series
        Spread time series
    zscore : pd.Series
        Z-score time series
    entry_threshold : float
        Entry threshold
    exit_threshold : float
        Exit threshold
    signals : Optional[pd.Series]
        Trading signals (optional)
    save_path : Optional[str]
        Path to save figure

    Examples
    --------
    >>> plot_spread_zscore(spread, zscore, entry_threshold=2.0, exit_threshold=0.5)
    """
    logger = StructuredLogger(__name__)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Spread
    axes[0].plot(spread.index, spread.values, linewidth=1.5, color='#2E86AB')
    axes[0].set_title('Basket Spread', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Spread', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Z-score
    axes[1].plot(zscore.index, zscore.values, linewidth=1.5, color='#06A77D')

    # Thresholds
    axes[1].axhline(y=entry_threshold, color='red', linestyle='--', linewidth=2, label=f'Entry (+{entry_threshold}σ)')
    axes[1].axhline(y=-entry_threshold, color='red', linestyle='--', linewidth=2, label=f'Entry (-{entry_threshold}σ)')
    axes[1].axhline(y=exit_threshold, color='orange', linestyle='--', linewidth=1.5, label=f'Exit (+{exit_threshold}σ)')
    axes[1].axhline(y=-exit_threshold, color='orange', linestyle='--', linewidth=1.5, label=f'Exit (-{exit_threshold}σ)')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Highlight signals if provided
    if signals is not None:
        long_entry = (signals == 1) & (signals.shift(1) != 1)
        short_entry = (signals == -1) & (signals.shift(1) != -1)

        axes[1].scatter(zscore[long_entry].index, zscore[long_entry].values,
                       color='green', marker='^', s=100, zorder=5, label='Long Entry')
        axes[1].scatter(zscore[short_entry].index, zscore[short_entry].values,
                       color='red', marker='v', s=100, zorder=5, label='Short Entry')

    axes[1].set_title('Z-Score with Trading Thresholds', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Z-Score', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Spread z-score plot saved", path=save_path)
        plt.close()
    else:
        plt.show()


def plot_optimization_convergence(
    convergence_history: List[float],
    parameter_name: str = "Sharpe Ratio",
    save_path: Optional[str] = None
) -> None:
    """
    Plot Bayesian optimization convergence.

    Parameters
    ----------
    convergence_history : List[float]
        Best score at each iteration
    parameter_name : str
        Name of optimized parameter
    save_path : Optional[str]
        Path to save figure

    Examples
    --------
    >>> plot_optimization_convergence(result.convergence_history, "Sharpe Ratio")
    """
    logger = StructuredLogger(__name__)

    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = np.arange(1, len(convergence_history) + 1)

    ax.plot(iterations, convergence_history, linewidth=2, color='#2E86AB', marker='o', markersize=4)
    ax.set_title(f'Bayesian Optimization Convergence - {parameter_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(f'Best {parameter_name}', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add final value annotation
    final_value = convergence_history[-1]
    ax.annotate(f'Final: {final_value:.4f}',
               xy=(len(convergence_history), final_value),
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Optimization convergence plot saved", path=save_path)
        plt.close()
    else:
        plt.show()


def plot_var_distribution(
    returns: pd.Series,
    var_95: float,
    var_99: float,
    save_path: Optional[str] = None
) -> None:
    """
    Plot return distribution with VaR thresholds.

    Parameters
    ----------
    returns : pd.Series
        Return time series
    var_95 : float
        95% VaR threshold
    var_99 : float
        99% VaR threshold
    save_path : Optional[str]
        Path to save figure

    Examples
    --------
    >>> plot_var_distribution(returns, var_95=0.02, var_99=0.03)
    """
    logger = StructuredLogger(__name__)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    ax.hist(returns, bins=50, density=True, alpha=0.7, color='#2E86AB', edgecolor='black')

    # Fit normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)),
           linewidth=2, color='red', label='Normal Fit')

    # VaR lines
    ax.axvline(x=-var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.2%}')
    ax.axvline(x=-var_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR: {var_99:.2%}')

    ax.set_title('Return Distribution with VaR Thresholds', fontsize=14, fontweight='bold')
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("VaR distribution plot saved", path=save_path)
        plt.close()
    else:
        plt.show()


def plot_correlation_matrix(
    prices: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot asset correlation heatmap.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data for assets
    save_path : Optional[str]
        Path to save figure

    Examples
    --------
    >>> plot_correlation_matrix(prices)
    """
    logger = StructuredLogger(__name__)

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Correlation matrix
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Correlation matrix plot saved", path=save_path)
        plt.close()
    else:
        plt.show()


def plot_drawdown(
    portfolio_value: pd.Series,
    save_path: Optional[str] = None
) -> None:
    """
    Plot drawdown over time.

    Parameters
    ----------
    portfolio_value : pd.Series
        Portfolio value time series
    save_path : Optional[str]
        Path to save figure

    Examples
    --------
    >>> plot_drawdown(result.portfolio_value)
    """
    logger = StructuredLogger(__name__)

    # Calculate drawdown
    running_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value - running_max) / running_max

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(drawdown.index, 0, drawdown.values * 100, color='#D62828', alpha=0.5)
    ax.plot(drawdown.index, drawdown.values * 100, linewidth=1.5, color='#D62828')

    ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Annotate max drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.annotate(f'Max DD: {max_dd*100:.2f}%',
               xy=(max_dd_date, max_dd*100),
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Drawdown plot saved", path=save_path)
        plt.close()
    else:
        plt.show()
