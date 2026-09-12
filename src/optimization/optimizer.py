"""Gaussian-process Bayesian optimisation with every trial recorded.

Thin wrapper around ``skopt.gp_minimize``: GP surrogate (Matern kernel with a learned
noise term), Expected Improvement acquisition and Latin-hypercube initial points.
Recording each trial matters because the number of trials is needed to deflate the
best in-sample Sharpe ratio for selection bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real

from src.utils.exceptions import ConfigError


@dataclass(frozen=True)
class Trial:
    params: dict[str, float]
    score: float


@dataclass(frozen=True)
class OptimizationResult:
    best_params: dict[str, float]
    best_score: float
    trials: list[Trial]

    @property
    def n_trials(self) -> int:
        return len(self.trials)


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _dimension(name: str, bounds: tuple) -> Integer | Real:
    if len(bounds) != 2:
        raise ConfigError(f"bounds for {name} must be (low, high), got {bounds!r}")
    low, high = bounds
    if not low < high:
        raise ConfigError(f"bounds for {name} must satisfy low < high, got {bounds!r}")
    if _is_int(low) and _is_int(high):
        return Integer(int(low), int(high), name=name)
    return Real(float(low), float(high), name=name)


def maximize(
    objective: Callable[[dict[str, float]], float],
    space: dict[str, tuple],
    n_calls: int = 40,
    n_initial_points: int = 10,
    random_state: int = 0,
) -> OptimizationResult:
    """Maximise ``objective(params)`` over a box.

    Integer bounds give integer dimensions; anything else is continuous. The
    objective must return a finite float. Exceptions propagate unchanged rather
    than being replaced by a penalty score.
    """
    if not (_is_int(n_calls) and _is_int(n_initial_points)) or not 1 <= n_initial_points <= n_calls:
        raise ConfigError(f"need 1 <= n_initial_points <= n_calls, got {n_initial_points} and {n_calls}")
    if not space:
        raise ConfigError("search space is empty")

    names = list(space)
    dimensions = [_dimension(name, tuple(space[name])) for name in names]
    trials: list[Trial] = []

    def negated(point: list) -> float:
        params = {
            name: int(value) if isinstance(dimension, Integer) else float(value)
            for name, value, dimension in zip(names, point, dimensions)
        }
        score = float(objective(params))
        if not math.isfinite(score):
            raise ValueError(f"objective returned a non-finite score {score} for {params}")
        trials.append(Trial(params=params, score=score))
        return -score

    gp_minimize(
        negated,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        initial_point_generator="lhs",
        acq_func="EI",
        random_state=random_state,
    )

    best = max(trials, key=lambda trial: trial.score)  # first maximum wins ties
    return OptimizationResult(best_params=dict(best.params), best_score=best.score, trials=trials)
