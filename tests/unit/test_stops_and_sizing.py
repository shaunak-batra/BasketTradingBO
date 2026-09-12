"""Tests for the v3 execution rules: loss stop, time stop and volatility targeting.

All three default to off, so a config that does not set them reproduces v2 exactly.

Note on direction: with weights [1, -1] a target state of +1 is long A and short B, so a
position loses when A *falls*. The price paths below are built that way.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtester import STOP_LOSS, TIME_STOP, BacktestConfig, resolve_gross, run_backtest
from src.cointegration.engine import net_exposure
from src.utils.exceptions import BacktestError, ConfigError
from tests.fixtures.synthetic import business_days

FREE = BacktestConfig(cost_bps=0.0, borrow_bps_annual=0.0)


def two_assets(a: list[float], b: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"A": np.asarray(a, dtype=float), "B": np.asarray(b, dtype=float)}, index=business_days(len(a)))


def states_for(prices: pd.DataFrame, values: list[int]) -> pd.Series:
    return pd.Series(values, index=prices.index, dtype=np.int64)


def with_stop(fraction: float, **kwargs) -> BacktestConfig:
    return BacktestConfig(cost_bps=0.0, borrow_bps_annual=0.0, stop_loss_fraction=fraction, **kwargs)


class TestLossStop:
    # Entry at bar 1 puts $50,000 (500 shares) long A. A then falls 20%, costing 10% of
    # equity by bar 4, before recovering completely by bar 7.
    PRICES = two_assets([100, 100, 100, 90, 80, 85, 95, 100, 100], [50] * 9)
    ALWAYS_LONG = [1] * 9

    def test_position_is_closed_on_the_bar_the_loss_threshold_is_breached(self):
        prices = self.PRICES
        result = run_backtest(prices, states_for(prices, self.ALWAYS_LONG), [1.0, -1.0], with_stop(0.08))
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == STOP_LOSS
        assert trade["exit_date"] == prices.index[4]  # -5% at bar 3, -10% at bar 4
        assert trade["return_on_equity"] == pytest.approx(-0.10)
        assert result.positions.loc[trade["exit_date"]].abs().sum() == 0.0

    def test_without_the_stop_the_same_path_rides_the_loss_out(self):
        # This is the v2 behaviour that lost 52% on XOM/CVX without a -4 sigma stop ever firing.
        prices = self.PRICES
        result = run_backtest(prices, states_for(prices, self.ALWAYS_LONG), [1.0, -1.0], FREE)
        assert result.equity.min() / 100_000 - 1 == pytest.approx(-0.10)
        assert result.trades.iloc[0]["exit_reason"] == "end_of_window"
        assert result.equity.iloc[-1] == pytest.approx(100_000)  # fully recovered

    def test_the_stopped_direction_is_not_re_entered_while_the_signal_persists(self):
        prices = two_assets([100, 100, 100, 70, 100, 100, 100, 100], [50] * 8)
        result = run_backtest(prices, states_for(prices, [1] * 8), [1.0, -1.0], with_stop(0.10))
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["exit_reason"] == STOP_LOSS
        assert result.positions.iloc[-1].abs().sum() == 0.0

    def test_a_fresh_signal_after_going_flat_allows_re_entry(self):
        prices = two_assets([100, 100, 100, 70, 100, 100, 100, 100, 100, 100], [50] * 10)
        states = states_for(prices, [1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        result = run_backtest(prices, states, [1.0, -1.0], with_stop(0.10))
        assert result.trades["exit_reason"].tolist() == [STOP_LOSS, "end_of_window"]
        # The second entry is sized on the equity left after the stop.
        assert result.trades.iloc[1]["entry_equity"] == pytest.approx(85_000)

    def test_no_effect_when_the_threshold_is_never_breached(self):
        prices = two_assets([100, 100, 99, 98, 99], [50] * 5)
        states = states_for(prices, [1] * 5)
        pd.testing.assert_series_equal(
            run_backtest(prices, states, [1.0, -1.0], with_stop(0.10)).equity,
            run_backtest(prices, states, [1.0, -1.0], FREE).equity,
        )


class TestTimeStop:
    def test_position_is_closed_after_the_maximum_holding_period(self):
        prices = two_assets([100] * 10, [50] * 10)
        result = run_backtest(
            prices,
            states_for(prices, [1] * 10),
            [1.0, -1.0],
            BacktestConfig(cost_bps=0.0, borrow_bps_annual=0.0, max_holding_bars=4),
        )
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == TIME_STOP
        assert trade["holding_bars"] == 4

    def test_loss_stop_takes_precedence_over_the_time_stop(self):
        prices = two_assets([100, 100, 100, 70, 70, 70], [50] * 6)
        result = run_backtest(
            prices, states_for(prices, [1] * 6), [1.0, -1.0], with_stop(0.10, max_holding_bars=2)
        )
        assert result.trades.iloc[0]["exit_reason"] == STOP_LOSS


class TestVolatilityTargeting:
    @pytest.mark.parametrize(
        "target, volatility, max_gross, expected",
        [(0.10, 0.20, 1.0, 0.5), (0.10, 0.05, 1.0, 1.0), (0.10, 0.05, 3.0, 2.0), (0.10, 0.0, 1.0, 1.0)],
    )
    def test_gross_is_the_ratio_of_target_to_estimated_volatility_capped(self, target, volatility, max_gross, expected):
        config = BacktestConfig(target_volatility=target, max_gross=max_gross)
        assert resolve_gross(config, volatility) == pytest.approx(expected)

    def test_fixed_gross_is_used_when_no_target_is_set(self):
        assert resolve_gross(BacktestConfig(gross_exposure=0.7), sizing_volatility=0.2) == 0.7

    def test_entry_notional_follows_the_resolved_gross(self):
        prices = two_assets([100.0] * 6, [50.0] * 6)
        states = states_for(prices, [0, 1, 1, 0, 0, 0])
        result = run_backtest(
            prices,
            states,
            [1.0, -1.0],
            BacktestConfig(cost_bps=0.0, borrow_bps_annual=0.0, target_volatility=0.10),
            sizing_volatility=0.25,
        )
        # gross = 0.10 / 0.25 = 0.4 of equity
        assert result.target_gross == pytest.approx(0.4)
        assert result.trades.iloc[0]["entry_gross_notional"] == pytest.approx(40_000.0)

    def test_a_missing_volatility_estimate_is_an_error(self):
        prices = two_assets([100.0] * 4, [50.0] * 4)
        with pytest.raises(BacktestError, match="sizing_volatility"):
            run_backtest(prices, states_for(prices, [0, 1, 1, 0]), [1.0, -1.0], BacktestConfig(target_volatility=0.1))


class TestNetExposure:
    @pytest.mark.parametrize(
        "weights, expected",
        [([0.5, -0.5], 0.0), ([0.5, 0.5], 1.0), ([0.6, -0.4], 0.2), ([0.5, -0.3, -0.2], 0.0)],
    )
    def test_net_exposure_as_a_share_of_gross(self, weights, expected):
        assert net_exposure(weights) == pytest.approx(expected)

    def test_rejects_a_zero_vector(self):
        with pytest.raises(ValueError):
            net_exposure([0.0, 0.0])


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stop_loss_fraction": 0.0},
            {"stop_loss_fraction": 1.0},
            {"stop_loss_fraction": -0.1},
            {"max_holding_bars": 0},
            {"max_holding_bars": 2.5},
            {"target_volatility": 0.0},
            {"target_volatility": math.inf},
            {"max_gross": 0.0},
        ],
    )
    def test_invalid_risk_settings_are_rejected(self, kwargs):
        with pytest.raises(ConfigError):
            BacktestConfig(**kwargs)
