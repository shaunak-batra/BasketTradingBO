"""
Module: Risk Manager

Risk management including VaR calculation, position sizing, and exposure limits.

Classes
-------
RiskManager
    Main risk management interface

VaRResult
    Container for VaR calculation results

Functions
---------
calculate_var_historical
    Calculate historical VaR
calculate_var_parametric
    Calculate parametric VaR (Gaussian)
calculate_var_cornish_fisher
    Calculate VaR with Cornish-Fisher expansion

Author: Quantitative Research Team
Created: 2025-01-18
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from src.utils.exceptions import RiskManagementException
from src.utils.logger import StructuredLogger, timed_execution


@dataclass
class VaRResult:
    """
    Container for Value at Risk calculation results.

    Attributes
    ----------
    var : float
        Value at Risk (positive value representing potential loss)
    confidence_level : float
        Confidence level (e.g., 0.95)
    method : str
        Method used ("historical", "parametric", "cornish_fisher")
    expected_shortfall : Optional[float]
        Conditional VaR (CVaR) / Expected Shortfall
    """
    var: float
    confidence_level: float
    method: str
    expected_shortfall: Optional[float] = None


class RiskManager:
    """
    Risk management for portfolio positions and exposure.

    Attributes
    ----------
    max_position_size : float
        Maximum position size as fraction of portfolio
    max_portfolio_var : float
        Maximum portfolio VaR (fraction of portfolio)
    var_confidence_level : float
        Confidence level for VaR calculations
    lookback_window : int
        Lookback window for risk calculations (days)

    Methods
    -------
    calculate_var(returns, method)
        Calculate Value at Risk
    calculate_position_size(returns, target_var)
        Calculate position size for target VaR
    check_risk_limits(positions, portfolio_value)
        Check if positions violate risk limits

    Examples
    --------
    >>> risk_manager = RiskManager(max_position_size=0.25, max_portfolio_var=0.02)
    >>> var_result = risk_manager.calculate_var(returns, method="historical")
    >>> print(f"95% VaR: {var_result.var:.2%}")
    """

    def __init__(
        self,
        max_position_size: float = 0.25,
        max_portfolio_var: float = 0.02,
        var_confidence_level: float = 0.95,
        lookback_window: int = 252
    ):
        """
        Initialize risk manager.

        Parameters
        ----------
        max_position_size : float
            Maximum position size (fraction, e.g., 0.25 = 25%)
        max_portfolio_var : float
            Maximum portfolio VaR (fraction, e.g., 0.02 = 2%)
        var_confidence_level : float
            Confidence level for VaR (e.g., 0.95 = 95%)
        lookback_window : int
            Lookback period for risk calculations (days)
        """
        self.logger = StructuredLogger(__name__)

        self.max_position_size = max_position_size
        self.max_portfolio_var = max_portfolio_var
        self.var_confidence_level = var_confidence_level
        self.lookback_window = lookback_window

        self.logger.info(
            "RiskManager initialized",
            max_position_size=max_position_size,
            max_portfolio_var=max_portfolio_var,
            var_confidence_level=var_confidence_level
        )

    @timed_execution
    def calculate_var(
        self,
        returns: pd.Series,
        method: str = "historical",
        confidence_level: Optional[float] = None
    ) -> VaRResult:
        """
        Calculate Value at Risk.

        Parameters
        ----------
        returns : pd.Series
            Return time series
        method : str
            Method: "historical", "parametric", "cornish_fisher"
        confidence_level : Optional[float]
            Confidence level (uses default if None)

        Returns
        -------
        VaRResult
            VaR calculation result

        Examples
        --------
        >>> var_result = risk_manager.calculate_var(returns, method="historical")

        Notes
        -----
        - Historical: Empirical quantile from historical data
        - Parametric: Assumes normal distribution
        - Cornish-Fisher: Adjusts for skewness and kurtosis
        """
        if confidence_level is None:
            confidence_level = self.var_confidence_level

        # Use lookback window
        returns_window = returns.tail(self.lookback_window)

        if len(returns_window) < 30:
            raise RiskManagementException(
                "Insufficient data for VaR calculation",
                context={"data_points": len(returns_window), "required": 30}
            )

        if method == "historical":
            var_result = calculate_var_historical(returns_window, confidence_level)
        elif method == "parametric":
            var_result = calculate_var_parametric(returns_window, confidence_level)
        elif method == "cornish_fisher":
            var_result = calculate_var_cornish_fisher(returns_window, confidence_level)
        else:
            raise RiskManagementException(
                f"Unknown VaR method: {method}",
                context={"method": method, "valid_methods": ["historical", "parametric", "cornish_fisher"]}
            )

        self.logger.info(
            "VaR calculated",
            method=method,
            var=round(float(var_result.var), 4),
            confidence_level=confidence_level
        )

        return var_result

    def calculate_position_size(
        self,
        returns: pd.Series,
        target_var: float,
        method: str = "historical"
    ) -> float:
        """
        Calculate position size for target VaR.

        Parameters
        ----------
        returns : pd.Series
            Return time series
        target_var : float
            Target VaR (fraction of portfolio)
        method : str
            VaR calculation method

        Returns
        -------
        float
            Recommended position size (fraction)

        Examples
        --------
        >>> position_size = risk_manager.calculate_position_size(returns, target_var=0.02)

        Notes
        -----
        Position size = target_var / asset_var
        Capped at max_position_size
        """
        var_result = self.calculate_var(returns, method=method)

        if var_result.var <= 0:
            self.logger.warning("VaR is zero or negative, using max position size")
            return self.max_position_size

        # Position size to achieve target VaR
        position_size = target_var / var_result.var

        # Cap at maximum
        position_size = min(position_size, self.max_position_size)

        self.logger.info(
            "Position size calculated",
            asset_var=round(float(var_result.var), 4),
            target_var=target_var,
            position_size=round(position_size, 4)
        )

        return position_size

    def check_risk_limits(
        self,
        positions: pd.Series,
        portfolio_value: float,
        prices: pd.Series
    ) -> bool:
        """
        Check if positions violate risk limits.

        Parameters
        ----------
        positions : pd.Series
            Current positions (shares)
        portfolio_value : float
            Total portfolio value
        prices : pd.Series
            Current prices

        Returns
        -------
        bool
            True if within limits, False if violated

        Examples
        --------
        >>> is_safe = risk_manager.check_risk_limits(positions, portfolio_value, prices)
        """
        # Calculate position values
        position_values = positions * prices

        # Calculate position sizes as fraction of portfolio
        position_sizes = position_values.abs() / portfolio_value

        # Check individual position limits
        max_position = position_sizes.max()
        if max_position > self.max_position_size:
            self.logger.warning(
                "Position size limit violated",
                max_position=round(float(max_position), 4),
                limit=self.max_position_size
            )
            return False

        self.logger.info("Risk limits check passed")
        return True


# Standalone utility functions

def calculate_var_historical(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> VaRResult:
    """
    Calculate historical Value at Risk.

    Parameters
    ----------
    returns : pd.Series
        Return time series
    confidence_level : float
        Confidence level (e.g., 0.95)

    Returns
    -------
    VaRResult
        VaR result

    Examples
    --------
    >>> var_result = calculate_var_historical(returns, confidence_level=0.95)

    Notes
    -----
    Historical VaR = empirical (1 - α) quantile of returns
    Expected Shortfall = mean of returns below VaR threshold
    """
    alpha = 1 - confidence_level

    # Calculate VaR as quantile
    var = -returns.quantile(alpha)

    # Calculate Expected Shortfall (CVaR)
    # ES = E[return | return < -VaR]
    returns_below_var = returns[returns < -var]
    if len(returns_below_var) > 0:
        expected_shortfall = -returns_below_var.mean()
    else:
        expected_shortfall = var

    return VaRResult(
        var=float(var),
        confidence_level=confidence_level,
        method="historical",
        expected_shortfall=float(expected_shortfall)
    )


def calculate_var_parametric(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> VaRResult:
    """
    Calculate parametric VaR assuming normal distribution.

    Parameters
    ----------
    returns : pd.Series
        Return time series
    confidence_level : float
        Confidence level

    Returns
    -------
    VaRResult
        VaR result

    Examples
    --------
    >>> var_result = calculate_var_parametric(returns, confidence_level=0.95)

    Notes
    -----
    Parametric VaR = -μ + σ * z_α
    where z_α is the α-quantile of standard normal distribution
    """
    alpha = 1 - confidence_level

    # Calculate mean and std
    mu = returns.mean()
    sigma = returns.std()

    # Get z-score for confidence level
    z_score = stats.norm.ppf(alpha)

    # VaR = -μ + σ * z_α
    var = -(mu + sigma * z_score)

    # Expected Shortfall for normal distribution
    # ES = μ + σ * φ(z_α) / α
    phi_z = stats.norm.pdf(z_score)
    expected_shortfall = -(mu + sigma * phi_z / alpha)

    return VaRResult(
        var=float(var),
        confidence_level=confidence_level,
        method="parametric",
        expected_shortfall=float(expected_shortfall)
    )


def calculate_var_cornish_fisher(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> VaRResult:
    """
    Calculate VaR using Cornish-Fisher expansion.

    Parameters
    ----------
    returns : pd.Series
        Return time series
    confidence_level : float
        Confidence level

    Returns
    -------
    VaRResult
        VaR result

    Examples
    --------
    >>> var_result = calculate_var_cornish_fisher(returns, confidence_level=0.95)

    Notes
    -----
    Adjusts normal VaR for skewness and excess kurtosis:

    z_CF = z_α + (z_α² - 1) * S / 6 + (z_α³ - 3z_α) * K / 24 - (2z_α³ - 5z_α) * S² / 36

    where:
        S = skewness
        K = excess kurtosis
    """
    alpha = 1 - confidence_level

    # Calculate moments
    mu = returns.mean()
    sigma = returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()  # Excess kurtosis in pandas

    # Get base z-score
    z_alpha = stats.norm.ppf(alpha)

    # Cornish-Fisher adjustment
    z_cf = (
        z_alpha
        + (z_alpha**2 - 1) * skew / 6
        + (z_alpha**3 - 3 * z_alpha) * kurt / 24
        - (2 * z_alpha**3 - 5 * z_alpha) * skew**2 / 36
    )

    # VaR with adjusted z-score
    var = -(mu + sigma * z_cf)

    return VaRResult(
        var=float(var),
        confidence_level=confidence_level,
        method="cornish_fisher",
        expected_shortfall=None  # ES not trivial for Cornish-Fisher
    )
