"""Cointegration tests (Johansen, Engle-Granger) and spread diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.cointegration.engine import _sequential_rank, engle_granger_test, johansen_test, normalize_weights
from src.cointegration.spread import compute_spread, half_life, rolling_zscore
from src.utils.exceptions import ConfigError, DataError
from tests.fixtures.synthetic import ar1_process, business_days, cointegrated_prices, random_walk_prices


class TestJohansen:
    def test_detects_cointegration_and_recovers_the_true_weights(self):
        prices, true_weights = cointegrated_prices(n=1500, seed=3)
        result = johansen_test(np.log(prices))
        assert result.is_cointegrated
        np.testing.assert_allclose(result.weights, normalize_weights(true_weights), atol=0.02)

    def test_false_positive_rate_on_driftless_random_walks_is_bounded(self):
        # det_order=0 uses critical values that assume drifting prices. Without drift the nominal 5%
        # test over-rejects (about 10-13% in simulation, see README Limitations). This pins the rate
        # so a change in the library or the code cannot make it silently worse.
        rejections = [
            johansen_test(np.log(random_walk_prices(n=400, k=3, seed=seed))).is_cointegrated for seed in range(100)
        ]
        assert np.mean(rejections) <= 0.15

    def test_false_positive_rate_is_near_nominal_when_prices_drift(self):
        rejections = [
            johansen_test(np.log(random_walk_prices(n=504, k=2, drift=0.002, seed=seed))).is_cointegrated
            for seed in range(100)
        ]
        assert np.mean(rejections) <= 0.10

    def test_engle_granger_agrees_on_a_strongly_cointegrated_basket(self):
        prices, _ = cointegrated_prices(n=1500, seed=3)
        result = engle_granger_test(np.log(prices))
        assert result.rejects_at_5pct
        assert result.p_value < 0.05

    def test_rank_counts_consecutive_rejections_only(self):
        assert _sequential_rank(np.array([30.0, 10.0, 20.0]), np.array([29.0, 15.0, 3.0])) == 1
        assert _sequential_rank(np.array([10.0, 30.0]), np.array([15.0, 3.0])) == 0

    def test_weight_normalisation_convention(self):
        np.testing.assert_allclose(normalize_weights([-2.0, 1.0]), [2 / 3, -1 / 3])
        np.testing.assert_allclose(normalize_weights([0.0, -3.0, 1.0]), [0.0, 0.75, -0.25])
        with pytest.raises(ValueError):
            normalize_weights([0.0, 0.0])

    def test_invalid_inputs_are_rejected(self):
        prices, _ = cointegrated_prices(n=200, seed=1)
        logs = np.log(prices)
        with pytest.raises(ConfigError):
            johansen_test(logs, significance=0.02)
        with pytest.raises(ConfigError):
            johansen_test(logs, det_order=2)
        with pytest.raises(DataError):
            johansen_test(logs.iloc[:, :1])
        with pytest.raises(DataError):
            johansen_test(logs.iloc[:30])
        broken = logs.copy()
        broken.iloc[5, 0] = np.nan
        with pytest.raises(DataError):
            johansen_test(broken)


class TestSpread:
    def test_spread_is_the_weighted_sum_of_log_prices(self):
        prices = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 40.0]}, index=business_days(2))
        spread = compute_spread(prices, [0.5, -0.5])
        np.testing.assert_allclose(spread.to_numpy(), 0.5 * np.log(prices["A"]) - 0.5 * np.log(prices["B"]))

    def test_zscore_has_no_partial_windows_and_matches_a_manual_calculation(self):
        spread = pd.Series(ar1_process(50, half_life=5, sigma=1.0, seed=2), index=business_days(50))
        z = rolling_zscore(spread, 10)
        assert z.iloc[:9].isna().all()
        window = spread.iloc[20:30]
        assert z.iloc[29] == pytest.approx((spread.iloc[29] - window.mean()) / window.std(ddof=1), rel=1e-12)

    def test_zscore_is_nan_for_windows_without_variance(self):
        assert rolling_zscore(pd.Series(np.full(30, 3.0), index=business_days(30)), 10).isna().all()

    @settings(max_examples=100, deadline=None)
    @given(st.lists(st.floats(-5, 5), min_size=12, max_size=80), st.data())
    def test_zscore_is_causal(self, values, data):
        cut = data.draw(st.integers(0, len(values) - 2))
        tail = data.draw(st.lists(st.floats(-5, 5), min_size=len(values) - cut - 1, max_size=len(values) - cut - 1))
        index = business_days(len(values))
        base = rolling_zscore(pd.Series(values, index=index), 10).to_numpy()
        altered = rolling_zscore(pd.Series(values[: cut + 1] + tail, index=index), 10).to_numpy()
        np.testing.assert_allclose(altered[: cut + 1], base[: cut + 1], rtol=1e-12, atol=1e-12)


class TestHalfLife:
    def test_recovers_the_half_life_of_an_ar1_process(self):
        assert half_life(ar1_process(20_000, half_life=10.0, sigma=1.0, seed=1)) == pytest.approx(10.0, rel=0.1)

    def test_is_infinite_for_an_explosive_series(self):
        rng = np.random.default_rng(0)
        values = np.empty(300)
        values[0] = 1.0
        for t in range(1, 300):
            values[t] = 1.01 * values[t - 1] + 0.1 * rng.standard_normal()
        assert half_life(values) == math.inf

    def test_needs_twenty_finite_observations(self):
        with pytest.raises(ValueError):
            half_life(np.arange(10.0))
        with pytest.raises(ValueError):
            half_life(np.r_[np.arange(30.0), np.nan])
