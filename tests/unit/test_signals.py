"""State-machine tests for signal generation."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.strategy.signals import ENTRY, EXIT, STOP, SignalParams, generate_signals
from src.utils.exceptions import ConfigError
from tests.fixtures.synthetic import business_days

PARAMS = SignalParams(entry_z=2.0, exit_z=0.5, stop_z=4.0, lookback=20)


def run(values: list[float]) -> tuple[list[int], list[str]]:
    zscore = pd.Series(values, index=business_days(len(values)), dtype=float)
    signals = generate_signals(zscore, PARAMS)
    return signals["state"].tolist(), signals["event"].tolist()


class TestTransitions:
    def test_long_cycle_with_inclusive_thresholds(self):
        states, events = run([0.0, -2.0, -1.0, -0.5, 0.0])
        assert states == [0, 1, 1, 0, 0]
        assert events == ["", ENTRY, "", EXIT, ""]

    def test_short_cycle_with_inclusive_thresholds(self):
        states, events = run([0.0, 2.0, 1.0, 0.5, 0.0])
        assert states == [0, -1, -1, 0, 0]
        assert events == ["", ENTRY, "", EXIT, ""]

    def test_stop_loss_exits_and_blocks_re_entry_until_back_inside_the_entry_band(self):
        states, events = run([-2.5, -3.0, -4.0, -3.5, -2.5, -1.9, -2.1])
        assert states == [1, 1, 0, 0, 0, 0, 1]
        assert events == [ENTRY, "", STOP, "", "", "", ENTRY]

    def test_no_entry_when_the_spread_is_already_beyond_the_stop(self):
        states, _ = run([-4.5, -3.0, -1.0, -2.2])
        assert states == [0, 0, 0, 1]

    def test_nan_never_opens_a_position_and_holds_an_open_one(self):
        states, _ = run([math.nan, -2.5, math.nan, -0.2])
        assert states == [0, 1, 1, 0]

    def test_long_exits_when_the_spread_overshoots_to_the_other_side(self):
        states, events = run([-2.5, 3.0])
        assert states == [1, 0]
        assert events == [ENTRY, EXIT]


class TestCausality:
    @settings(max_examples=200, deadline=None)
    @given(st.lists(st.one_of(st.floats(-6, 6), st.just(math.nan)), min_size=2, max_size=60), st.data())
    def test_states_depend_only_on_current_and_past_z(self, values, data):
        cut = data.draw(st.integers(0, len(values) - 2))
        tail = data.draw(st.lists(st.floats(-6, 6), min_size=len(values) - cut - 1, max_size=len(values) - cut - 1))
        base_states, base_events = run(values)
        altered_states, altered_events = run(values[: cut + 1] + tail)
        assert base_states[: cut + 1] == altered_states[: cut + 1]
        assert base_events[: cut + 1] == altered_events[: cut + 1]


class TestParams:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"exit_z": 2.0},
            {"exit_z": -0.1},
            {"stop_z": 2.0},
            {"entry_z": 0.0},
            {"lookback": 1},
            {"lookback": True},
            {"lookback": 20.5},
            {"entry_z": math.nan},
        ],
    )
    def test_invalid_parameters_are_rejected(self, kwargs):
        with pytest.raises(ConfigError):
            SignalParams(**kwargs)

    @settings(max_examples=200, deadline=None)
    @given(st.floats(1.0, 3.0), st.floats(0.0, 0.8), st.floats(0.5, 3.0), st.integers(20, 252))
    def test_every_point_in_the_search_box_is_a_valid_parameter_set(self, entry, exit_fraction, stop_offset, lookback):
        params = SignalParams.from_search_space(entry, exit_fraction, stop_offset, lookback)
        assert 0.0 <= params.exit_z < params.entry_z < params.stop_z
        assert isinstance(params.lookback, int)
