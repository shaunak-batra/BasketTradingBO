"""
Unit tests for Bayesian optimization.
"""

import numpy as np
import pytest

from src.optimization.optimizer import (
    BayesianOptimizer,
    OptimizationResult,
    expected_improvement,
)


class TestBayesianOptimizer:
    """Tests for BayesianOptimizer class."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        def dummy_objective(params):
            return params['x']**2

        parameter_space = {'x': (-5.0, 5.0)}

        optimizer = BayesianOptimizer(
            objective_function=dummy_objective,
            parameter_space=parameter_space,
            n_initial_points=5,
            n_iterations=20
        )

        assert optimizer.n_initial_points == 5
        assert optimizer.n_iterations == 20
        assert len(optimizer.dimensions) == 1

    def test_optimize_simple_quadratic(self):
        """Test optimization of simple quadratic function."""
        def quadratic(params):
            x = params['x']
            y = params['y']
            return (x - 2)**2 + (y + 1)**2

        parameter_space = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }

        optimizer = BayesianOptimizer(
            objective_function=quadratic,
            parameter_space=parameter_space,
            n_initial_points=5,
            n_iterations=30,
            random_state=42,
            verbose=False
        )

        result = optimizer.optimize()

        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert 'x' in result.best_params
        assert 'y' in result.best_params
        assert result.n_iterations == 30

        # Check convergence (should find minimum near x=2, y=-1)
        assert abs(result.best_params['x'] - 2.0) < 1.0
        assert abs(result.best_params['y'] - (-1.0)) < 1.0
        assert result.best_score < 1.0  # Near minimum

    def test_convergence_history(self):
        """Test that convergence history is tracked."""
        def simple_function(params):
            return params['x']**2

        parameter_space = {'x': (-10.0, 10.0)}

        optimizer = BayesianOptimizer(
            objective_function=simple_function,
            parameter_space=parameter_space,
            n_iterations=15,
            verbose=False
        )

        result = optimizer.optimize()

        # Convergence history should be non-increasing
        assert len(result.convergence_history) == 15
        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] <= result.convergence_history[i-1]

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading optimizer state."""
        def objective(params):
            return params['x']**2

        parameter_space = {'x': (-5.0, 5.0)}

        optimizer = BayesianOptimizer(
            objective_function=objective,
            parameter_space=parameter_space,
            n_iterations=10,
            verbose=False
        )

        result = optimizer.optimize()

        # Save state
        state_path = str(tmp_path / "optimizer_state.pkl")
        optimizer.save_state(state_path)

        # Load state
        loaded_optimizer = BayesianOptimizer.load_state(state_path, objective)

        assert len(loaded_optimizer.all_params) == len(optimizer.all_params)
        assert len(loaded_optimizer.all_scores) == len(optimizer.all_scores)
        assert loaded_optimizer.parameter_space == optimizer.parameter_space


class TestExpectedImprovement:
    """Tests for expected_improvement function."""

    def test_expected_improvement_calculation(self):
        """Test EI calculation with mock GP model."""
        # Create mock GP model
        class MockGP:
            def predict(self, X, return_std=False):
                # Return constant predictions for testing
                mu = np.array([0.5, 0.3, 0.7])
                sigma = np.array([0.1, 0.2, 0.15])
                if return_std:
                    return mu, sigma
                return mu

        X = np.array([[1.0], [2.0], [3.0]])
        model = MockGP()
        y_best = 0.4

        ei = expected_improvement(X, model, y_best, xi=0.01)

        # EI should be positive (we can improve on y_best)
        assert len(ei) == 3
        assert all(ei >= 0)

    def test_expected_improvement_zero_sigma(self):
        """Test EI when sigma is zero (deterministic prediction)."""
        class MockGP:
            def predict(self, X, return_std=False):
                mu = np.array([0.5])
                sigma = np.array([0.0])  # Zero uncertainty
                if return_std:
                    return mu, sigma
                return mu

        X = np.array([[1.0]])
        model = MockGP()
        y_best = 0.3

        ei = expected_improvement(X, model, y_best, xi=0.01)

        # Should handle zero sigma gracefully
        assert ei[0] >= 0


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test OptimizationResult instantiation."""
        result = OptimizationResult(
            best_params={'x': 1.5, 'y': 2.3},
            best_score=0.25,
            all_params=[{'x': 1.0, 'y': 2.0}, {'x': 1.5, 'y': 2.3}],
            all_scores=[0.30, 0.25],
            n_iterations=2,
            convergence_history=[0.30, 0.25]
        )

        assert result.best_params['x'] == 1.5
        assert result.best_score == 0.25
        assert result.n_iterations == 2
        assert len(result.convergence_history) == 2
