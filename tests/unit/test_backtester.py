"""Backtester tests.

``TestHandComputedCases`` pins exact numbers worked out by hand. ``TestInvariants``
checks accounting and timing properties on random inputs with Hypothesis.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.backtesting.backtester import BacktestConfig, run_backtest
from src.strategy.signals import STOP
from src.utils.exceptions import BacktestError, ConfigError
from tests.fixtures.synthetic import business_days

FREE = BacktestConfig(cost_bps=0.0, borrow_bps_annual=0.0, execution_lag=1)


def two_assets(a: list[float], b: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"A": np.asarray(a, dtype=float), "B": np.asarray(b, dtype=float)}, index=business_days(len(a)))


def states_for(prices: pd.DataFrame, values: list[int]) -> pd.Series:
    return pd.Series(values, index=prices.index, dtype=np.int64)


class TestHandComputedCases:
    def test_flat_prices_lose_exactly_the_round_trip_cost(self):
        prices = two_assets([100.0] * 8, [50.0] * 8)
        config = BacktestConfig(cost_bps=10.0, borrow_bps_annual=0.0)
        result = run_backtest(prices, states_for(prices, [0, 1, 1, 1, 0, 0, 0, 0]), [1.0, -0.5], config)
        # Entry fills at bar 2 on 100,000 gross (cost 100); exit fills at bar 5 on the same notional (cost 100).
        expected = [100_000, 100_000, 99_900, 99_900, 99_900, 99_800, 99_800, 99_800]
        np.testing.assert_allclose(result.equity.to_numpy(), expected, rtol=0, atol=1e-9)
        assert result.pnl.abs().sum() == 0.0

    def test_long_spread_earns_the_move_on_the_shares_held(self):
        prices = two_assets([100, 100, 100, 110, 110, 110], [50] * 6)
        result = run_backtest(prices, states_for(prices, [0, 1, 1, 0, 0, 0]), [1.0, -1.0], FREE)
        # Entry at bar 2: $50,000 long A (500 shares) and $50,000 short B (1,000 shares).
        np.testing.assert_allclose(result.positions.iloc[2].to_numpy(), [500.0, -1000.0])
        # A gains 10 on bar 3: +5,000. The exit decided at bar 3 fills at bar 4.
        np.testing.assert_allclose(result.equity.to_numpy(), [100_000, 100_000, 100_000, 105_000, 105_000, 105_000])
        trade = result.trades.iloc[0]
        assert trade["net_pnl"] == pytest.approx(5_000.0)
        assert trade["holding_bars"] == 2

    def test_short_spread_loses_when_the_spread_rises(self):
        prices = two_assets([100, 100, 100, 110, 110, 110], [50] * 6)
        result = run_backtest(prices, states_for(prices, [0, -1, -1, 0, 0, 0]), [1.0, -1.0], FREE)
        assert result.equity.iloc[-1] == pytest.approx(95_000.0)

    def test_signal_is_filled_after_the_close_that_produced_it(self):
        # A jumps at bar 3. The entry decided at bar 2 fills at bar 3's close with a one-bar lag and
        # misses the jump; filled at bar 2's own close (lag 0) it would capture it.
        prices = two_assets([100, 100, 100, 110, 110, 110], [50] * 6)
        states = states_for(prices, [0, 0, 1, 1, 0, 0])
        lagged = run_backtest(prices, states, [1.0, -1.0], FREE)
        same_bar = run_backtest(
            prices, states, [1.0, -1.0], BacktestConfig(cost_bps=0.0, borrow_bps_annual=0.0, execution_lag=0)
        )
        assert lagged.equity.iloc[-1] == pytest.approx(100_000.0)
        assert same_bar.equity.iloc[-1] == pytest.approx(105_000.0)

    def test_no_signal_means_no_trades_and_constant_equity(self):
        prices = two_assets([100, 90, 120, 80], [50, 60, 40, 55])
        result = run_backtest(prices, states_for(prices, [0, 0, 0, 0]), [1.0, -1.0], BacktestConfig())
        assert (result.equity == 100_000.0).all()
        assert result.trades.empty

    def test_borrow_fee_accrues_on_short_notional_held_overnight(self):
        prices = two_assets([100.0] * 8, [50.0] * 8)
        config = BacktestConfig(cost_bps=0.0, borrow_bps_annual=252.0)  # 1 bp per day
        result = run_backtest(prices, states_for(prices, [0, 1, 1, 1, 0, 0, 0, 0]), [1.0, -1.0], config)
        # $50,000 short carried into bars 3, 4 and 5 at 1 bp per night.
        assert result.borrow_costs.sum() == pytest.approx(15.0)
        assert result.equity.iloc[-1] == pytest.approx(100_000.0 - 15.0)

    def test_trading_cost_is_linear_in_cost_bps(self):
        prices = two_assets([100.0] * 6, [50.0] * 6)
        states = states_for(prices, [0, 1, 1, 0, 0, 0])
        losses = [
            100_000.0 - run_backtest(prices, states, [1.0, -1.0], BacktestConfig(cost_bps=c, borrow_bps_annual=0.0)).equity.iloc[-1]
            for c in (5.0, 10.0)
        ]
        assert losses[0] == pytest.approx(100.0)
        assert losses[1] == pytest.approx(2 * losses[0])

    def test_next_trade_is_sized_on_current_equity(self):
        prices = two_assets([100, 100, 100, 110, 110, 110, 110, 110, 110], [50] * 9)
        result = run_backtest(prices, states_for(prices, [0, 1, 1, 0, 0, 1, 1, 0, 0]), [1.0, -1.0], FREE)
        np.testing.assert_allclose(result.trades["entry_gross_notional"].to_numpy(dtype=float), [100_000.0, 105_000.0])

    def test_open_position_is_closed_at_the_end_of_the_window(self):
        prices = two_assets([100, 101, 102, 103], [50, 50, 50, 50])
        result = run_backtest(prices, states_for(prices, [1, 1, 1, 1]), [1.0, -1.0], FREE)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["exit_reason"] == "end_of_window"
        assert result.positions.iloc[-1].abs().sum() == 0.0

    def test_open_position_is_reported_when_not_forced_flat(self):
        prices = two_assets([100, 101, 102, 103], [50, 50, 50, 50])
        result = run_backtest(prices, states_for(prices, [1, 1, 1, 1]), [1.0, -1.0], FREE, force_flat_at_end=False)
        trade = result.trades.iloc[0]
        assert bool(trade["is_open"]) and trade["exit_reason"] == "open"
        assert trade["net_pnl"] == pytest.approx(result.equity.iloc[-1] - 100_000.0)

    def test_exit_reason_is_taken_from_the_signal_event(self):
        prices = two_assets([100.0] * 6, [50.0] * 6)
        states = states_for(prices, [0, 1, 1, 0, 0, 0])
        events = pd.Series(["", "entry", "", STOP, "", ""], index=prices.index, dtype=object)
        result = run_backtest(prices, states, [1.0, -1.0], FREE, events=events)
        assert result.trades.iloc[0]["exit_reason"] == STOP

    def test_one_ledger_row_per_round_trip(self):
        prices = two_assets([100.0] * 10, [50.0] * 10)
        result = run_backtest(prices, states_for(prices, [0, 1, 1, 0, -1, -1, 0, 1, 0, 0]), [1.0, -1.0], FREE)
        assert result.trades["direction"].tolist() == [1, -1, 1]


class TestValidation:
    def setup_method(self):
        self.prices = two_assets([100.0, 101.0, 102.0], [50.0, 51.0, 52.0])
        self.states = states_for(self.prices, [0, 1, 0])

    @pytest.mark.parametrize("column, value", [("A", np.nan), ("B", 0.0), ("A", -1.0)])
    def test_rejects_missing_or_non_positive_prices(self, column, value):
        prices = self.prices.copy()
        prices.loc[prices.index[1], column] = value
        with pytest.raises(BacktestError):
            run_backtest(prices, self.states, [1.0, -1.0], FREE)

    def test_rejects_unknown_states(self):
        with pytest.raises(BacktestError):
            run_backtest(self.prices, states_for(self.prices, [0, 2, 0]), [1.0, -1.0], FREE)

    def test_rejects_states_on_a_different_index(self):
        shifted = pd.Series([0, 1, 0], index=self.prices.index + pd.Timedelta(days=1))
        with pytest.raises(BacktestError):
            run_backtest(self.prices, shifted, [1.0, -1.0], FREE)

    @pytest.mark.parametrize("weights", [[0.0, 0.0], [1.0], [1.0, np.nan]])
    def test_rejects_bad_weights(self, weights):
        with pytest.raises(BacktestError):
            run_backtest(self.prices, self.states, weights, FREE)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cost_bps": -1.0},
            {"gross_exposure": 0.0},
            {"initial_capital": math.nan},
            {"borrow_bps_annual": math.inf},
            {"execution_lag": -1},
            {"execution_lag": True},
        ],
    )
    def test_rejects_invalid_config(self, kwargs):
        with pytest.raises(ConfigError):
            BacktestConfig(**kwargs)


@st.composite
def backtest_inputs(draw):
    """Random prices, target states, weights and costs.

    Daily moves are capped at 1% and gross exposure at 1x, so no draw can bankrupt the
    account: each trade can lose at most ~86% of the equity it was sized on.
    """
    n = draw(st.integers(min_value=3, max_value=40))
    k = draw(st.integers(min_value=2, max_value=4))
    step = st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False)
    steps = np.array(draw(st.lists(st.lists(step, min_size=k, max_size=k), min_size=n, max_size=n)))
    index = business_days(n)
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=[f"S{i}" for i in range(k)])
    states = pd.Series(draw(st.lists(st.sampled_from([-1, 0, 1]), min_size=n, max_size=n)), index=index, dtype=np.int64)
    magnitudes = np.array(draw(st.lists(st.floats(0.05, 3.0), min_size=k, max_size=k)))
    signs = np.array(draw(st.lists(st.sampled_from([-1.0, 1.0]), min_size=k, max_size=k)))
    config = BacktestConfig(
        gross_exposure=draw(st.floats(0.1, 1.0)),
        cost_bps=draw(st.floats(0.0, 50.0)),
        borrow_bps_annual=draw(st.floats(0.0, 500.0)),
        execution_lag=draw(st.integers(0, 3)),
    )
    return prices, states, magnitudes * signs, config


class TestInvariants:
    @settings(max_examples=200, deadline=None)
    @given(backtest_inputs())
    def test_equity_change_is_fully_explained_by_pnl_and_costs(self, inputs):
        prices, states, weights, config = inputs
        result = run_backtest(prices, states, weights, config)
        change = result.equity.iloc[-1] - config.initial_capital
        explained = result.pnl.sum() - result.borrow_costs.sum() - result.trading_costs.sum()
        assert change == pytest.approx(explained, rel=1e-9, abs=1e-6)
        # Flat at the end, every dollar of P&L and cost belongs to exactly one closed trade.
        assert float(result.trades["net_pnl"].sum()) == pytest.approx(change, rel=1e-9, abs=1e-6)
        assert result.positions.iloc[-1].abs().sum() == 0.0

    @settings(max_examples=200, deadline=None)
    @given(backtest_inputs(), st.data())
    def test_future_prices_cannot_change_past_equity(self, inputs, data):
        prices, states, weights, config = inputs
        cut = data.draw(st.integers(min_value=0, max_value=len(prices) - 2))
        factors = np.array(data.draw(st.lists(st.floats(0.8, 1.25), min_size=prices.shape[1], max_size=prices.shape[1])))
        shocked = prices.copy()
        shocked.iloc[cut + 1 :] = shocked.iloc[cut + 1 :].to_numpy() * factors
        base = run_backtest(prices, states, weights, config, force_flat_at_end=False)
        altered = run_backtest(shocked, states, weights, config, force_flat_at_end=False)
        np.testing.assert_allclose(
            altered.equity.to_numpy()[: cut + 1], base.equity.to_numpy()[: cut + 1], rtol=1e-12, atol=1e-9
        )

    @settings(max_examples=100, deadline=None)
    @given(backtest_inputs())
    def test_entries_are_sized_to_the_configured_gross_exposure(self, inputs):
        prices, states, weights, config = inputs
        trades = run_backtest(prices, states, weights, config).trades
        np.testing.assert_allclose(
            trades["entry_gross_notional"].to_numpy(dtype=float),
            trades["entry_equity"].to_numpy(dtype=float) * config.gross_exposure,
            rtol=1e-9,
        )
