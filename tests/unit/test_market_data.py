"""Price alignment, validation and snapshot tests (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import _extract_close, align_prices, load_prices, suspicious_moves, validate_prices
from src.utils.exceptions import DataError


def series(values, name: str, start: str = "2020-01-01") -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(start, periods=len(values)), name=name)


def random_series(ticker: str, seed: int, n: int = 300) -> pd.Series:
    rng = np.random.default_rng(seed)
    return series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))), ticker)


class TestAlign:
    def test_dates_missing_in_any_ticker_are_dropped_not_filled(self):
        a = series(range(1, 101), "A")
        b = series(range(1, 101), "B").drop(pd.bdate_range("2020-01-01", periods=100)[10])
        prices, dropped = align_prices([a, b], max_missing_fraction=0.02)
        assert dropped == 1
        assert len(prices) == 99
        assert not prices.isna().any().any()

    def test_a_ticker_that_starts_late_is_rejected_with_its_first_date(self):
        a = series(range(1, 101), "A")
        b = series(range(1, 51), "B", start=str(a.index[50].date()))
        with pytest.raises(DataError, match="First available date"):
            align_prices([a, b], max_missing_fraction=0.02)

    def test_duplicate_tickers_are_rejected(self):
        with pytest.raises(DataError):
            align_prices([series([1, 2], "A"), series([1, 2], "A")], max_missing_fraction=0.02)


class TestValidate:
    def setup_method(self):
        self.prices = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]}, index=pd.bdate_range("2020-01-01", periods=3))

    def test_clean_prices_pass(self):
        assert validate_prices(self.prices) is self.prices

    @pytest.mark.parametrize("value", [np.nan, 0.0, -1.0, np.inf])
    def test_bad_values_are_rejected(self, value):
        prices = self.prices.copy()
        prices.iloc[1, 0] = value
        with pytest.raises(DataError):
            validate_prices(prices)

    def test_unsorted_dates_and_single_columns_are_rejected(self):
        with pytest.raises(DataError):
            validate_prices(self.prices.iloc[::-1])
        with pytest.raises(DataError):
            validate_prices(self.prices[["A"]])


class TestExtractClose:
    def test_multiindex_columns_and_timezones_keep_the_trading_date(self):
        index = pd.DatetimeIndex(["2020-01-02", "2020-01-03"], tz="Asia/Tokyo")
        columns = pd.MultiIndex.from_tuples([("Close", "AAA"), ("Open", "AAA")], names=["Price", "Ticker"])
        raw = pd.DataFrame([[1.0, 2.0], [1.5, 2.5]], index=index, columns=columns)
        close = _extract_close(raw, "AAA")
        assert close.name == "AAA"
        assert close.index.tz is None
        assert [str(date.date()) for date in close.index] == ["2020-01-02", "2020-01-03"]
        assert close.tolist() == [1.0, 1.5]

    def test_flat_columns(self):
        raw = pd.DataFrame({"Open": [1.0], "Close": [2.0]}, index=pd.DatetimeIndex(["2020-01-02"]))
        assert _extract_close(raw, "AAA").tolist() == [2.0]


class TestSnapshots:
    def test_second_load_uses_the_snapshot_and_reproduces_identical_data(self, tmp_path):
        calls = []

        def downloader(ticker, start, end):
            calls.append(ticker)
            return random_series(ticker, seed=len(calls))

        first = load_prices(["AAA", "BBB"], "2020-01-01", "2021-03-01", tmp_path, downloader=downloader)

        def offline(*_):
            raise AssertionError("the snapshot should have been used")

        second = load_prices(["AAA", "BBB"], "2020-01-01", "2021-03-01", tmp_path, downloader=offline)
        assert (first.source, second.source) == ("download", "snapshot")
        pd.testing.assert_frame_equal(first.prices, second.prices, check_exact=True)
        assert first.sha256 == second.sha256
        assert calls == ["AAA", "BBB"]
        assert b"\r\n" not in first.path.read_bytes()

    def test_refresh_downloads_again(self, tmp_path):
        calls = []

        def downloader(ticker, start, end):
            calls.append(ticker)
            return random_series(ticker, seed=1)

        load_prices(["AAA", "BBB"], "2020-01-01", "2021-03-01", tmp_path, downloader=downloader)
        load_prices(["AAA", "BBB"], "2020-01-01", "2021-03-01", tmp_path, refresh=True, downloader=downloader)
        assert len(calls) == 4

    def test_snapshot_with_different_columns_is_rejected(self, tmp_path):
        load_prices(["AAA", "BBB"], "2020-01-01", "2021-03-01", tmp_path, downloader=lambda t, s, e: random_series(t, 1))
        path = tmp_path / "AAA_BBB_2020-01-01_2021-03-01.csv"
        frame = pd.read_csv(path, index_col=0)
        frame.columns = ["AAA", "ZZZ"]
        frame.to_csv(path)
        with pytest.raises(DataError):
            load_prices(["AAA", "BBB"], "2020-01-01", "2021-03-01", tmp_path, downloader=lambda t, s, e: random_series(t, 1))

    def test_fewer_than_two_distinct_tickers_are_rejected(self, tmp_path):
        with pytest.raises(DataError):
            load_prices(["AAA", "AAA"], "2020-01-01", "2021-01-01", tmp_path)


def test_suspicious_moves_flags_large_daily_jumps():
    prices = pd.DataFrame({"A": [100.0, 101.0, 180.0], "B": [50.0, 50.5, 51.0]}, index=pd.bdate_range("2020-01-01", periods=3))
    moves = suspicious_moves(prices)
    assert len(moves) == 1
    assert moves[0]["ticker"] == "A"
