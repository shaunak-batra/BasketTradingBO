"""Domain-specific exception hierarchy for basket trading system."""

from typing import Any, Dict, Optional


class BasketTradingException(Exception):
    """Base exception for basket trading system."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Initialize exception.

        Parameters
        ----------
        message : str
            Error message
        context : Optional[Dict]
            Additional context
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} | Context: {context_str}"
        return self.message


class CointegrationException(BasketTradingException):
    """Exception raised during cointegration testing."""
    pass


class NoCointegrationException(CointegrationException):
    """Exception raised when no cointegration is detected."""
    pass


class StationarityException(CointegrationException):
    """Exception raised when stationarity tests fail."""
    pass


class DataFetchException(BasketTradingException):
    """Exception raised when fetching market data fails."""
    pass


class DataValidationException(BasketTradingException):
    """Exception raised when data validation fails."""
    pass


class DataQualityException(DataValidationException):
    """Exception raised when data quality checks fail."""
    pass


class SignalGenerationException(BasketTradingException):
    """Exception raised during signal generation."""
    pass


class PositionSizingException(BasketTradingException):
    """Exception raised during position sizing."""
    pass


class RiskManagementException(BasketTradingException):
    """Exception raised during risk management calculations."""
    pass


class BacktestException(BasketTradingException):
    """Exception raised during backtesting."""
    pass


class OptimizationException(BasketTradingException):
    """Exception raised during Bayesian optimization."""
    pass


class ConfigurationException(BasketTradingException):
    """Exception raised for configuration errors."""
    pass


class CacheException(BasketTradingException):
    """Exception raised for caching errors."""
    pass


class VisualizationException(BasketTradingException):
    """Exception raised during visualization."""
    pass
