"""Walk-forward protocol tests on synthetic data with known structure."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtester import BacktestConfig
from src.backtesting.metrics import returns_from_equity, sharpe_ratio
from src.backtesting.walk_forward import (
    OptimizationConfig,
    WalkForwardConfig,
    fold_windows,
    replay_with_costs,
    run_walk_forward,
)
from src.strategy.signals import SignalParams
from src.utils.exceptions import ConfigError, DataError
from tests.fixtures.synthetic import cointegrated_prices, random_walk_prices

WALK = WalkForwardConfig(formation_days=250, trading_days=60, max_half_life_days=60.0)
FREE = BacktestConfig(cost_bps=0.0, borrow_bps_annual=0.0)
PARAMS = SignalParams(entry_z=1.5, exit_z=0.25, stop_z=4.0, lookback=40)
SMALL_SEARCH = OptimizationConfig(
    space={"entry_z": (1.0, 3.0), "exit_fraction": (0.0, 0.8), "stop_offset": (0.5, 3.0), "lookback": (20, 120)},
    n_calls=8,
    n_initial_points=4,
    min_trades=2,
    random_state=1,
)


@pytest.fixture(scope="module")
def mean_reverting() -> pd.DataFrame:
    prices, _ = cointegrated_prices(n=1000, half_life=5.0, spread_sigma=0.01, seed=11)
    return prices


class TestFoldWindows:
    def test_trading_windows_follow_their_formation_windows_and_tile_the_sample(self):
        windows = fold_windows(1000, 250, 60)
        assert windows[0] == (0, 250, 250, 310)
        for (formation_start, formation_end, trading_start, trading_end), following in zip(windows, windows[1:]):
            assert formation_end - formation_start == 250
            assert trading_start == formation_end
            assert following[2] == trading_end
        assert windows[-1][3] == 1000

    def test_too_little_data_is_an_error(self):
        with pytest.raises(DataError):
            fold_windows(251, 250, 60)


class TestFixedMode:
    def test_makes_money_on_a_truly_mean_reverting_basket_before_costs(self, mean_reverting):
        result = run_walk_forward(mean_reverting, WALK, FREE, params=PARAMS)
        traded = [fold for fold in result.folds if fold.traded]
        # The 5% trace test on 250 bars has less than perfect power, so a few truly
        # cointegrated windows are (correctly, by the rule) left untraded.
        assert len(traded) >= 0.7 * len(result.folds)
        assert all(fold.backtest.equity.iloc[-1] > fold.backtest.equity.iloc[0] for fold in traded)
        assert sharpe_ratio(returns_from_equity(result.equity)) > 1.0

    def test_equity_carries_across_folds_and_every_fold_ends_flat(self, mean_reverting):
        result = run_walk_forward(mean_reverting, WALK, BacktestConfig(), params=PARAMS)
        for previous, current in zip(result.folds, result.folds[1:]):
            assert current.backtest.equity.iloc[0] == previous.backtest.equity.iloc[-1]
            assert previous.trading_end < current.trading_start
        for fold in result.folds:
            assert fold.formation_end < fold.trading_start
            assert fold.backtest.positions.iloc[-1].abs().sum() == 0.0
        assert result.equity.index.is_monotonic_increasing
        assert not result.equity.index.has_duplicates

    def test_independent_random_walks_are_rarely_traded(self):
        result = run_walk_forward(random_walk_prices(n=1000, k=3, seed=4), WALK, BacktestConfig(), params=PARAMS)
        assert sum(fold.traded for fold in result.folds) <= 2
        for fold in result.folds:
            if not fold.traded:
                assert fold.backtest.trades.empty
                assert fold.backtest.equity.nunique() == 1

    def test_future_prices_cannot_change_past_results(self, mean_reverting):
        cut = WALK.formation_days + 2 * WALK.trading_days + 17
        shocked = mean_reverting.copy()
        shocked.iloc[cut + 1 :] = shocked.iloc[cut + 1 :].to_numpy() * np.array([1.3, 0.8, 1.1])
        base = run_walk_forward(mean_reverting, WALK, BacktestConfig(), params=PARAMS)
        altered = run_walk_forward(shocked, WALK, BacktestConfig(), params=PARAMS)
        cut_date = mean_reverting.index[cut]
        np.testing.assert_allclose(
            altered.equity.loc[:cut_date].to_numpy(), base.equity.loc[:cut_date].to_numpy(), rtol=1e-12, atol=1e-9
        )
        for before, after in zip(base.folds, altered.folds):
            if before.bars[1] <= cut + 1:
                np.testing.assert_allclose(after.johansen.weights, before.johansen.weights, rtol=1e-12)
                assert after.skip_reason == before.skip_reason

    def test_replay_reproduces_the_run_and_higher_costs_reduce_equity(self, mean_reverting):
        config = BacktestConfig(cost_bps=5.0)
        result = run_walk_forward(mean_reverting, WALK, config, params=PARAMS)
        assert len(result.trades) > 0
        same = replay_with_costs(mean_reverting, result, config)
        pd.testing.assert_series_equal(same.equity, result.equity)
        dearer = replay_with_costs(mean_reverting, result, replace(config, cost_bps=25.0))
        assert dearer.equity.iloc[-1] < result.equity.iloc[-1]


class TestOptimizedMode:
    def test_parameters_are_tuned_within_the_search_box_on_each_formation_window(self, mean_reverting):
        result = run_walk_forward(mean_reverting.iloc[:500], WALK, FREE, optimization=SMALL_SEARCH)
        traded = [fold for fold in result.folds if fold.traded]
        assert traded
        for fold in traded:
            assert fold.in_sample["n_trials"] == SMALL_SEARCH.n_calls
            assert 1.0 <= fold.params.entry_z <= 3.0
            assert 20 <= fold.params.lookback <= 120
            deflated = fold.in_sample["deflated_sharpe"]
            if fold.in_sample["n_valid_trials"] >= 2:
                assert 0.0 <= deflated <= 1.0
            else:
                assert math.isnan(deflated)

    @pytest.mark.slow
    def test_future_prices_cannot_change_past_results_with_optimisation(self, mean_reverting):
        prices = mean_reverting.iloc[:500]
        cut = WALK.formation_days + WALK.trading_days + 10
        shocked = prices.copy()
        shocked.iloc[cut + 1 :] = shocked.iloc[cut + 1 :].to_numpy() * np.array([1.2, 0.9, 1.1])
        base = run_walk_forward(prices, WALK, FREE, optimization=SMALL_SEARCH)
        altered = run_walk_forward(shocked, WALK, FREE, optimization=SMALL_SEARCH)
        cut_date = prices.index[cut]
        np.testing.assert_allclose(
            altered.equity.loc[:cut_date].to_numpy(), base.equity.loc[:cut_date].to_numpy(), rtol=1e-12, atol=1e-9
        )


class TestValidation:
    def test_exactly_one_of_params_or_optimisation(self, mean_reverting):
        with pytest.raises(ConfigError):
            run_walk_forward(mean_reverting, WALK, FREE)
        with pytest.raises(ConfigError):
            run_walk_forward(mean_reverting, WALK, FREE, params=PARAMS, optimization=SMALL_SEARCH)

    def test_lookback_must_leave_room_inside_the_formation_window(self, mean_reverting):
        with pytest.raises(ConfigError):
            run_walk_forward(mean_reverting, WALK, FREE, params=replace(PARAMS, lookback=240))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"formation_days": 10},
            {"trading_days": 1},
            {"significance": 0.02},
            {"max_half_life_days": 0},
            {"require_cointegration": "yes"},
        ],
    )
    def test_invalid_walk_forward_config(self, kwargs):
        with pytest.raises(ConfigError):
            WalkForwardConfig(**kwargs)

    @pytest.mark.parametrize(
        "space",
        [
            {"entry_z": (1.0, 3.0)},
            {"entry_z": (1.0, 3.0), "exit_fraction": (0.0, 1.2), "stop_offset": (0.5, 3.0), "lookback": (20, 120)},
            {"entry_z": (1.0, 3.0), "exit_fraction": (0.0, 0.8), "stop_offset": (0.5, 3.0), "lookback": (20.0, 120.0)},
        ],
    )
    def test_invalid_search_spaces_are_rejected(self, space):
        with pytest.raises(ConfigError):
            OptimizationConfig(space=space)
