"""Universe screening: pair enumeration, overlap handling and multiple-testing control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.universe import benjamini_hochberg, load_universe_prices, pair_prices, universe_pairs
from src.utils.exceptions import BasketTradingError, DataError


class TestPairEnumeration:
    def test_every_pair_inside_a_family_and_none_across_families(self):
        pairs = universe_pairs({"metals": ["GLD", "IAU", "GDX"], "bonds": ["TLT", "IEF"]})
        assert [(f, a, b) for f, a, b in pairs] == [
            ("metals", "GDX", "GLD"),
            ("metals", "GDX", "IAU"),
            ("metals", "GLD", "IAU"),
            ("bonds", "IEF", "TLT"),
        ]

    def test_a_family_of_n_gives_n_choose_2_pairs(self):
        assert len(universe_pairs({"big": [f"T{i}" for i in range(6)]})) == 15

    @pytest.mark.parametrize("families", [{"x": ["A", "A"]}, {"x": ["A"]}])
    def test_malformed_families_are_rejected(self, families):
        with pytest.raises(BasketTradingError):
            universe_pairs(families)


class TestBenjaminiHochberg:
    def test_step_up_rejects_up_to_the_largest_rank_that_passes(self):
        # m=4, q=0.05: thresholds 0.0125, 0.025, 0.0375, 0.05. The first three pass.
        discoveries = benjamini_hochberg(np.array([0.01, 0.02, 0.03, 0.9]), 0.05)
        assert discoveries.tolist() == [True, True, True, False]

    def test_a_gap_does_not_stop_the_step_up_but_the_cutoff_is_the_largest_passing_p(self):
        # m=3, q=0.05: thresholds 0.0167, 0.0333, 0.05. Only the smallest passes.
        discoveries = benjamini_hochberg(np.array([0.001, 0.9, 0.04]), 0.05)
        assert discoveries.tolist() == [True, False, False]

    def test_no_discoveries_when_nothing_is_small_enough(self):
        assert not benjamini_hochberg(np.array([0.2, 0.4, 0.6]), 0.05).any()

    def test_is_more_permissive_than_bonferroni(self):
        p = np.array([0.001, 0.012, 0.020, 0.30, 0.40])
        assert benjamini_hochberg(p, 0.05).sum() >= (p <= 0.05 / len(p)).sum()

    def test_nan_p_values_are_never_discoveries(self):
        discoveries = benjamini_hochberg(np.array([0.001, np.nan, 0.9]), 0.05)
        assert discoveries.tolist() == [True, False, False]
        assert not benjamini_hochberg(np.array([np.nan, np.nan]), 0.05).any()

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1])
    def test_invalid_levels_are_rejected(self, level):
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([0.01]), level)


class TestPairPrices:
    def frame(self) -> pd.DataFrame:
        index = pd.bdate_range("2020-01-01", periods=300)
        old = pd.Series(100.0 + np.arange(300) * 0.1, index=index, name="OLD")
        young = old.copy().rename("YOUNG")
        young.iloc[:100] = np.nan  # listed later
        return pd.concat([old, young], axis=1)

    def test_prices_are_truncated_to_the_overlap(self):
        prices = pair_prices(self.frame(), "OLD", "YOUNG", min_bars=100, max_missing_fraction=0.02)
        assert len(prices) == 200
        assert list(prices.columns) == ["OLD", "YOUNG"]
        assert not prices.isna().any().any()

    def test_too_little_overlap_is_rejected(self):
        with pytest.raises(BasketTradingError, match="overlapping bars"):
            pair_prices(self.frame(), "OLD", "YOUNG", min_bars=250, max_missing_fraction=0.02)

    def test_gaps_inside_the_overlap_still_fail_validation(self):
        frame = self.frame()
        frame.iloc[150:170, 0] = np.nan  # a hole in the middle of the overlap
        with pytest.raises(DataError):
            pair_prices(frame, "OLD", "YOUNG", min_bars=100, max_missing_fraction=0.02)


class TestUniverseSnapshot:
    def test_download_once_then_reuse_the_snapshot(self, tmp_path):
        calls = []

        def downloader(ticker, start, end):
            calls.append(ticker)
            index = pd.bdate_range(start, periods=50)
            return pd.Series(100.0 + np.arange(50), index=index, name=ticker)

        snapshot = tmp_path / "universe.csv"
        frame, digest = load_universe_prices(["AAA", "BBB"], "2020-01-01", "2020-06-01", snapshot, downloader=downloader)
        assert calls == ["AAA", "BBB"]
        assert list(frame.columns) == ["AAA", "BBB"]

        def offline(*_):
            raise AssertionError("the snapshot should have been reused")

        again, same_digest = load_universe_prices(
            ["AAA", "BBB"], "2020-01-01", "2020-06-01", snapshot, downloader=offline
        )
        pd.testing.assert_frame_equal(frame, again, check_exact=True)
        assert digest == same_digest

    def test_a_snapshot_missing_a_ticker_is_rebuilt(self, tmp_path):
        def downloader(ticker, start, end):
            index = pd.bdate_range(start, periods=50)
            return pd.Series(100.0 + np.arange(50), index=index, name=ticker)

        snapshot = tmp_path / "universe.csv"
        load_universe_prices(["AAA"], "2020-01-01", "2020-06-01", snapshot, downloader=downloader)
        frame, _ = load_universe_prices(["AAA", "BBB"], "2020-01-01", "2020-06-01", snapshot, downloader=downloader)
        assert list(frame.columns) == ["AAA", "BBB"]
