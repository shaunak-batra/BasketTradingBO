"""Structured logging with context, timing, and color output."""

import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'


class StructuredLogger:
    """Structured logger with context and color output."""

    def __init__(self, name: str):
        """Initialize structured logger.

        Parameters
        ----------
        name : str
            Logger name (typically __name__)
        """
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _format_message(self, message: str, **context: Any) -> str:
        """Format message with structured context.

        Parameters
        ----------
        message : str
            Log message
        **context : Any
            Additional context fields

        Returns
        -------
        str
            Formatted message
        """
        if context:
            context_str = json.dumps(context, default=str)
            return f"{message} | {context_str}"
        return message

    def info(self, message: str, **context: Any) -> None:
        """Log info message with context.

        Parameters
        ----------
        message : str
            Log message
        **context : Any
            Additional context fields
        """
        formatted = self._format_message(message, **context)
        self.logger.info(f"{Colors.GREEN}{formatted}{Colors.RESET}")

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message with context.

        Parameters
        ----------
        message : str
            Log message
        **context : Any
            Additional context fields
        """
        formatted = self._format_message(message, **context)
        self.logger.warning(f"{Colors.YELLOW}{formatted}{Colors.RESET}")

    def error(self, message: str, **context: Any) -> None:
        """Log error message with context.

        Parameters
        ----------
        message : str
            Log message
        **context : Any
            Additional context fields
        """
        formatted = self._format_message(message, **context)
        self.logger.error(f"{Colors.RED}{formatted}{Colors.RESET}")

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message with context.

        Parameters
        ----------
        message : str
            Log message
        **context : Any
            Additional context fields
        """
        formatted = self._format_message(message, **context)
        self.logger.debug(f"{Colors.CYAN}{formatted}{Colors.RESET}")


def timed_execution(func: Callable) -> Callable:
    """Decorator to log function execution time.

    Parameters
    ----------
    func : Callable
        Function to time

    Returns
    -------
    Callable
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = StructuredLogger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(
                f"{func.__name__} completed",
                execution_time_seconds=round(elapsed, 3)
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} failed",
                execution_time_seconds=round(elapsed, 3),
                error=str(e)
            )
            raise

    return wrapper
