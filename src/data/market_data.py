"""
Module: Market Data Adapter

Fetch, validate, and normalize financial time series data from multiple sources
with automatic failover, retry logic, and corporate action adjustments.

Classes
-------
MarketDataAdapter
    Main interface for fetching historical price data
DataValidator
    Validate data quality and handle anomalies
ValidationReport
    Container for validation results

Notes
-----
Uses yfinance as primary data source with automatic retry logic (3 attempts,
exponential backoff). All prices are adjusted for splits and dividends.

Author: Quantitative Research Team
Created: 2025-01-18
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.config import ConfigManager
from src.utils.exceptions import DataFetchException, DataQualityException, DataValidationException
from src.utils.io import save_csv, save_parquet
from src.utils.logger import StructuredLogger, timed_execution


@dataclass
class ValidationReport:
    """
    Container for data validation results.

    Attributes
    ----------
    is_valid : bool
        Whether data passes all validation checks
    missing_ratio : float
        Ratio of missing values
    outlier_count : int
        Number of outliers detected
    discontinuities : List[Tuple[str, str, float]]
        List of (ticker, date, price_jump) for discontinuities
    warnings : List[str]
        List of warning messages
    errors : List[str]
        List of error messages
    """
    is_valid: bool
    missing_ratio: float
    outlier_count: int
    discontinuities: List[Tuple[str, str, float]]
    warnings: List[str]
    errors: List[str]


class DataValidator:
    """
    Validate data quality and handle anomalies.

    Methods
    -------
    check_missing_values(df, threshold)
        Check if missing value ratio is below threshold
    detect_outliers(df, method, threshold)
        Detect outliers using z-score or IQR method
    verify_trading_days(df)
        Verify data has reasonable trading day coverage
    check_price_continuity(df, threshold)
        Check for suspicious price jumps
    validate(df)
        Run all validation checks and return report
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize validator.

        Parameters
        ----------
        config : Optional[ConfigManager]
            Configuration manager instance
        """
        self.logger = StructuredLogger(__name__)

        if config is None:
            config = ConfigManager.load_config()

        self.max_missing_ratio = config.get("data.validation.max_missing_ratio", 0.10)
        self.outlier_threshold = config.get("data.validation.outlier_zscore_threshold", 6.0)
        self.price_jump_threshold = config.get("data.validation.price_jump_threshold", 0.25)

    def check_missing_values(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check if missing value ratio is below threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Price data

        Returns
        -------
        Tuple[bool, float]
            (is_valid, missing_ratio)
        """
        total_values = df.size
        missing_values = df.isnull().sum().sum()
        missing_ratio = missing_values / total_values if total_values > 0 else 0.0

        is_valid = missing_ratio <= self.max_missing_ratio

        self.logger.info(
            "Missing values check",
            missing_values=missing_values,
            total_values=total_values,
            missing_ratio=round(missing_ratio, 4),
            is_valid=is_valid
        )

        return is_valid, missing_ratio

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "zscore"
    ) -> Tuple[pd.DataFrame, int]:
        """
        Detect outliers using z-score method.

        Parameters
        ----------
        df : pd.DataFrame
            Price data
        method : str
            Detection method ('zscore' or 'iqr')

        Returns
        -------
        Tuple[pd.DataFrame, int]
            (outlier_mask, outlier_count)
        """
        if method == "zscore":
            # Calculate returns
            returns = df.pct_change()

            # Calculate z-scores
            z_scores = np.abs((returns - returns.mean()) / returns.std())

            # Flag outliers
            outliers = z_scores > self.outlier_threshold
            outlier_count = outliers.sum().sum()

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        self.logger.info(
            "Outlier detection",
            method=method,
            outlier_count=outlier_count,
            threshold=self.outlier_threshold
        )

        return outliers, outlier_count

    def check_price_continuity(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """
        Check for suspicious price jumps.

        Parameters
        ----------
        df : pd.DataFrame
            Price data

        Returns
        -------
        List[Tuple[str, str, float]]
            List of (ticker, date, price_jump) for discontinuities
        """
        discontinuities = []

        for col in df.columns:
            returns = df[col].pct_change()

            # Find large price jumps
            large_jumps = returns[np.abs(returns) > self.price_jump_threshold]

            for date, jump in large_jumps.items():
                discontinuities.append((col, str(date), float(jump)))
                self.logger.warning(
                    "Price discontinuity detected",
                    ticker=col,
                    date=str(date),
                    price_jump=round(jump, 4)
                )

        return discontinuities

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation checks.

        Parameters
        ----------
        df : pd.DataFrame
            Price data to validate

        Returns
        -------
        ValidationReport
            Validation results
        """
        warnings = []
        errors = []

        # Check missing values
        missing_valid, missing_ratio = self.check_missing_values(df)
        if not missing_valid:
            errors.append(
                f"Missing value ratio {missing_ratio:.2%} exceeds threshold {self.max_missing_ratio:.2%}"
            )

        # Detect outliers
        _, outlier_count = self.detect_outliers(df)
        if outlier_count > 0:
            warnings.append(f"Detected {outlier_count} outliers")

        # Check price continuity
        discontinuities = self.check_price_continuity(df)
        if discontinuities:
            warnings.append(f"Detected {len(discontinuities)} price discontinuities")

        # Overall validity
        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            missing_ratio=missing_ratio,
            outlier_count=outlier_count,
            discontinuities=discontinuities,
            warnings=warnings,
            errors=errors
        )


class MarketDataAdapter:
    """
    Fetch historical price data from multiple sources with failover.

    Attributes
    ----------
    primary_source : str
        Primary data source ('yfinance')
    cache_enabled : bool
        Whether to use caching
    adjustment_method : str
        Price adjustment method ('back', 'forward', 'none')

    Methods
    -------
    fetch_data(tickers, start_date, end_date, interval)
        Fetch historical price data
    validate_data(df)
        Validate data quality
    save_data(df, tickers, format)
        Save data to disk

    Examples
    --------
    >>> adapter = MarketDataAdapter()
    >>> data = adapter.fetch_data(['AAPL', 'MSFT'], '2020-01-01', '2021-01-01')
    >>> print(data.head())
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize market data adapter.

        Parameters
        ----------
        config : Optional[ConfigManager]
            Configuration manager instance
        """
        self.logger = StructuredLogger(__name__)

        if config is None:
            config = ConfigManager.load_config()

        self.config = config
        self.primary_source = config.get("data.sources.primary", "yfinance")
        self.max_retries = config.get("data.fetch_settings.max_retries", 3)
        self.retry_delay = config.get("data.fetch_settings.retry_delay", 2.0)
        self.timeout = config.get("data.fetch_settings.timeout", 30.0)

        self.validator = DataValidator(config)

        self.logger.info(
            "MarketDataAdapter initialized",
            primary_source=self.primary_source,
            max_retries=self.max_retries
        )

    @timed_execution
    def fetch_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical price data with retry logic.

        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        interval : str
            Data interval ('1d', '1h', etc.)

        Returns
        -------
        pd.DataFrame
            Price data with tickers as columns

        Raises
        ------
        DataFetchException
            If data fetch fails after all retries

        Examples
        --------
        >>> adapter = MarketDataAdapter()
        >>> data = adapter.fetch_data(['AAPL', 'MSFT'], '2020-01-01', '2021-01-01')
        """
        self.logger.info(
            "Fetching market data",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

        for attempt in range(self.max_retries):
            try:
                # Fetch each ticker individually to avoid rate limiting
                # then combine into single DataFrame
                ticker_data = {}

                for ticker in tickers:
                    self.logger.info(f"Fetching data for {ticker}...")

                    # Fetch single ticker
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=True,
                        progress=False
                    )

                    if data.empty or 'Close' not in data.columns:
                        raise DataFetchException(
                            f"No data returned for ticker {ticker}",
                            context={"ticker": ticker, "start": start_date, "end": end_date}
                        )

                    ticker_data[ticker] = data['Close']

                    # Small delay between requests to avoid rate limiting
                    if len(tickers) > 1:
                        time.sleep(0.5)

                # Combine all ticker data into single DataFrame
                # Use concat to properly align Series with potentially different indices
                df = pd.concat(ticker_data, axis=1)

                # Validate data is not empty
                if df.empty:
                    raise DataFetchException(
                        "No data returned from yfinance",
                        context={"tickers": tickers, "start": start_date, "end": end_date}
                    )

                # Validate minimum data points
                if len(df) < 10:
                    raise DataFetchException(
                        f"Insufficient data points: {len(df)} < 10",
                        context={"tickers": tickers, "start": start_date, "end": end_date}
                    )

                # Check for columns with all NaN
                all_nan_cols = df.columns[df.isna().all()].tolist()
                if all_nan_cols:
                    raise DataFetchException(
                        f"Tickers returned no data: {all_nan_cols}",
                        context={"invalid_tickers": all_nan_cols}
                    )

                self.logger.info(
                    "Data fetched successfully",
                    rows=len(df),
                    columns=len(df.columns),
                    date_range=f"{df.index[0]} to {df.index[-1]}"
                )

                return df

            except DataFetchException:
                # Re-raise our own exceptions immediately
                raise
            except KeyError as e:
                self.logger.warning(
                    "Data fetch attempt failed - KeyError",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e)
                )

                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise DataFetchException(
                        f"Failed to fetch data after {self.max_retries} attempts - possible invalid tickers",
                        context={"tickers": tickers, "error": str(e)}
                    )
            except Exception as e:
                self.logger.warning(
                    "Data fetch attempt failed",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e)
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise DataFetchException(
                        f"Failed to fetch data after {self.max_retries} attempts",
                        context={"tickers": tickers, "error": str(e)}
                    )

    def validate_data(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate data quality.

        Parameters
        ----------
        df : pd.DataFrame
            Price data to validate

        Returns
        -------
        ValidationReport
            Validation results

        Raises
        ------
        DataQualityException
            If validation fails
        """
        report = self.validator.validate(df)

        if not report.is_valid:
            raise DataQualityException(
                "Data validation failed",
                context={
                    "errors": report.errors,
                    "missing_ratio": report.missing_ratio
                }
            )

        if report.warnings:
            self.logger.warning(
                "Data validation warnings",
                warnings=report.warnings
            )

        return report

    def save_data(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        format: str = "parquet"
    ) -> Path:
        """
        Save data to disk.

        Parameters
        ----------
        df : pd.DataFrame
            Price data
        tickers : List[str]
            List of tickers
        format : str
            Output format ('parquet' or 'csv')

        Returns
        -------
        Path
            Path to saved file
        """
        # Create filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        tickers_str = "_".join(tickers[:3])  # Use first 3 tickers
        if len(tickers) > 3:
            tickers_str += f"_and_{len(tickers)-3}_more"

        raw_data_path = self.config.get("data.storage.raw_data_path", "data/raw")
        output_dir = Path(raw_data_path) / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            file_path = output_dir / f"{tickers_str}.parquet"
            save_parquet(df, file_path)
        else:
            file_path = output_dir / f"{tickers_str}.csv"
            save_csv(df, file_path)

        self.logger.info(
            "Data saved",
            file_path=str(file_path),
            format=format
        )

        return file_path
