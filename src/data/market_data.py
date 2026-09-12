"""Daily adjusted close prices: download, align, validate and snapshot.

Every run records the SHA-256 of the exact price file it used. Re-running from the
snapshot reproduces results bit-for-bit. Re-downloading may not, because Yahoo
revises dividend-adjusted history.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.utils.exceptions import DataError
from src.utils.io import load_json, save_json, sha256_file

logger = logging.getLogger(__name__)

Downloader = Callable[[str, str, str], pd.Series]


@dataclass(frozen=True)
class PriceData:
    prices: pd.DataFrame
    sha256: str
    path: Path
    source: str  # "download" or "snapshot"
    metadata: dict


def download_adjusted_close(ticker: str, start: str, end: str, attempts: int = 3) -> pd.Series:
    """Split/dividend-adjusted daily closes from Yahoo Finance. ``end`` is exclusive."""
    import yfinance as yf  # imported lazily: tests and snapshot runs never need the network

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False
            )
            if raw is not None and not raw.empty:
                return _extract_close(raw, ticker)
            last_error = DataError(f"Yahoo returned no rows for {ticker} between {start} and {end}")
        except Exception as exc:  # network / parsing errors from yfinance
            last_error = exc
        if attempt < attempts:
            time.sleep(2.0 * attempt)
    raise DataError(f"Failed to download {ticker} after {attempts} attempts: {last_error}")


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if "Close" not in raw.columns.get_level_values(0):
        raise DataError(f"No 'Close' column in data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns (field, ticker) MultiIndex columns
        if close.shape[1] != 1:
            raise DataError(f"Expected one Close column for {ticker}, got {close.shape[1]}")
        close = close.iloc[:, 0]
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)  # keep the exchange-local trading date
    return pd.Series(close.to_numpy(dtype=float), index=index.normalize(), name=ticker)


def align_prices(series: list[pd.Series], max_missing_fraction: float) -> tuple[pd.DataFrame, int]:
    """Inner-join price series on dates.

    Dates where any ticker is missing are dropped. If that removes more than
    ``max_missing_fraction`` of all dates, a ticker probably does not cover the
    requested window (e.g. a later IPO), and the caller must choose a valid window.
    Missing prices are never forward-filled.
    """
    if len(series) < 2:
        raise DataError("A basket needs at least two tickers")
    names = [s.name for s in series]
    if len(set(names)) != len(names):
        raise DataError(f"Duplicate tickers: {names}")
    for s in series:
        if s.index.has_duplicates:
            raise DataError(f"Duplicate dates in {s.name}")
    combined = pd.concat(series, axis=1, join="outer").sort_index()
    complete = combined.dropna(how="any")
    dropped = len(combined) - len(complete)
    fraction = dropped / len(combined) if len(combined) else 1.0
    if fraction > max_missing_fraction:
        first_dates = {s.name: str(s.dropna().index.min().date()) for s in series if not s.dropna().empty}
        raise DataError(
            f"Aligning tickers drops {dropped} of {len(combined)} dates ({fraction:.1%}), above the "
            f"{max_missing_fraction:.1%} limit. First available date per ticker: {first_dates}"
        )
    complete.index.name = "date"
    return complete, dropped


def validate_prices(prices: pd.DataFrame, min_rows: int = 2) -> pd.DataFrame:
    """Raise ``DataError`` unless prices are a clean, strictly positive, date-indexed panel."""
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2:
        raise DataError("prices must be a DataFrame with at least two columns")
    if len(prices) < min_rows:
        raise DataError(f"need at least {min_rows} rows, got {len(prices)}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataError("prices must be indexed by date")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise DataError("price dates must be unique and increasing")
    values = prices.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise DataError("prices contain NaN or infinite values")
    if np.any(values <= 0):
        raise DataError("prices must be strictly positive")
    return prices


def suspicious_moves(prices: pd.DataFrame, threshold: float = 0.4) -> list[dict]:
    """Daily log moves larger than ``threshold`` in absolute value (possible bad ticks)."""
    log_returns = np.log(prices).diff().iloc[1:]
    flagged = log_returns.abs().stack()
    flagged = flagged[flagged > threshold]
    return [
        {"date": str(date.date()), "ticker": ticker, "log_return": float(log_returns.loc[date, ticker])}
        for (date, ticker) in flagged.index
    ]


def snapshot_path(snapshot_dir: str | Path, tickers: list[str], start: str, end: str) -> Path:
    return Path(snapshot_dir) / f"{'_'.join(tickers)}_{start}_{end}.csv"


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    snapshot_dir: str | Path,
    max_missing_fraction: float = 0.02,
    refresh: bool = False,
    downloader: Downloader = download_adjusted_close,
) -> PriceData:
    """Load prices from a local snapshot, or download and snapshot them.

    The snapshot is a plain CSV written with round-trip float precision, so the
    SHA-256 recorded in results identifies the exact input data.
    """
    if len(tickers) < 2 or len(set(tickers)) != len(tickers):
        raise DataError(f"need at least two distinct tickers, got {tickers}")
    path = snapshot_path(snapshot_dir, tickers, start, end)
    meta_path = path.with_suffix(".meta.json")

    if path.exists() and not refresh:
        prices = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
        if list(prices.columns) != list(tickers):
            raise DataError(f"Snapshot {path} has columns {list(prices.columns)}, expected {tickers}")
        metadata = load_json(meta_path) if meta_path.exists() else {}
        source = "snapshot"
    else:
        series = [downloader(ticker, start, end) for ticker in tickers]
        prices, dropped = align_prices(series, max_missing_fraction)
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path, lineterminator="\n")  # identical bytes (and hash) on every OS
        metadata = {
            "tickers": tickers,
            "start": start,
            "end_exclusive": end,
            "rows": len(prices),
            "dropped_dates": dropped,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": "Yahoo Finance via yfinance (auto_adjust=True)",
        }
        save_json(metadata, meta_path)
        source = "download"
        # Re-read so downloaded and snapshot runs see identical floats.
        prices = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")

    validate_prices(prices)
    moves = suspicious_moves(prices)
    for move in moves:
        logger.warning("Large daily move: %s %s log return %.3f", move["ticker"], move["date"], move["log_return"])
    metadata = {**metadata, "suspicious_moves": moves}
    return PriceData(prices=prices, sha256=sha256_file(path), path=path, source=source, metadata=metadata)
