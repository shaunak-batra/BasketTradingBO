"""Bayesian optimiser wrapper tests."""

from __future__ import annotations

import math

import pytest

from src.optimization.optimizer import maximize
from src.utils.exceptions import ConfigError

SPACE = {"x": (1.0, 3.0), "n": (20, 252)}


def smooth(params: dict) -> float:
    return -((params["x"] - 2.2) ** 2) - ((params["n"] - 30) / 20) ** 2


def test_finds_the_optimum_of_a_smooth_function_with_an_integer_dimension():
    result = maximize(smooth, SPACE, n_calls=30, n_initial_points=10, random_state=0)
    assert abs(result.best_params["x"] - 2.2) < 0.2
    assert abs(result.best_params["n"] - 30) <= 10


def test_integer_bounds_give_integers_and_real_bounds_give_floats():
    result = maximize(smooth, SPACE, n_calls=12, n_initial_points=6, random_state=1)
    for trial in result.trials:
        assert isinstance(trial.params["n"], int)
        assert isinstance(trial.params["x"], float)
        assert 20 <= trial.params["n"] <= 252 and 1.0 <= trial.params["x"] <= 3.0


def test_every_trial_is_recorded_and_the_best_is_the_maximum():
    result = maximize(smooth, SPACE, n_calls=12, n_initial_points=6, random_state=2)
    assert result.n_trials == 12
    assert result.best_score == max(trial.score for trial in result.trials)


def test_the_same_seed_reproduces_the_same_trials():
    first = maximize(smooth, SPACE, n_calls=10, n_initial_points=5, random_state=3)
    second = maximize(smooth, SPACE, n_calls=10, n_initial_points=5, random_state=3)
    assert [trial.params for trial in first.trials] == [trial.params for trial in second.trials]


def test_objective_errors_propagate_instead_of_becoming_penalties():
    def broken(params: dict) -> float:
        raise RuntimeError("bug in objective")

    with pytest.raises(RuntimeError, match="bug in objective"):
        maximize(broken, SPACE, n_calls=5, n_initial_points=5)


def test_non_finite_scores_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        maximize(lambda params: math.nan, SPACE, n_calls=5, n_initial_points=5)


@pytest.mark.parametrize(
    "space, n_calls, n_initial_points",
    [(SPACE, 5, 10), (SPACE, 5, 0), ({"x": (3.0, 1.0)}, 5, 5), ({}, 5, 5)],
)
def test_invalid_settings_are_rejected(space, n_calls, n_initial_points):
    with pytest.raises(ConfigError):
        maximize(smooth, space, n_calls=n_calls, n_initial_points=n_initial_points)
