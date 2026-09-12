"""Exception hierarchy.

Errors are raised and propagated. Nothing in the research pipeline converts an
exception into a sentinel score or a silent fallback, because a swallowed error
in a backtest becomes a wrong number in a report.
"""


class BasketTradingError(Exception):
    """Base class for all project errors."""


class DataError(BasketTradingError):
    """Market data could not be fetched, aligned or validated."""


class ConfigError(BasketTradingError):
    """Invalid configuration value or parameter combination."""


class BacktestError(BasketTradingError):
    """Backtest inputs are inconsistent or the simulation became invalid."""
