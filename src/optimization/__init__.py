"""Bayesian optimization module."""

from src.optimization.optimizer import (
    BayesianOptimizer,
    OptimizationResult,
    expected_improvement,
)

__all__ = [
    "BayesianOptimizer",
    "OptimizationResult",
    "expected_improvement",
]
