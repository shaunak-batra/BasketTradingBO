"""
Module: Bayesian Optimizer

Bayesian optimization for hyperparameter tuning using Gaussian Process surrogate models.

Classes
-------
BayesianOptimizer
    Main optimization engine with GP surrogate and Expected Improvement acquisition

OptimizationResult
    Container for optimization results

Functions
---------
expected_improvement
    Calculate Expected Improvement acquisition function

Author: Quantitative Research Team
Created: 2025-01-18
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from src.utils.logger import StructuredLogger, timed_execution


@dataclass
class OptimizationResult:
    """
    Container for Bayesian optimization results.

    Attributes
    ----------
    best_params : Dict[str, float]
        Best parameter values found
    best_score : float
        Best objective function value
    all_params : List[Dict[str, float]]
        All evaluated parameter sets
    all_scores : List[float]
        All objective function values
    n_iterations : int
        Number of iterations run
    convergence_history : List[float]
        Best score at each iteration
    """
    best_params: Dict[str, float]
    best_score: float
    all_params: List[Dict[str, float]]
    all_scores: List[float]
    n_iterations: int
    convergence_history: List[float]


class BayesianOptimizer:
    """
    Bayesian optimization engine using Gaussian Process surrogate models.

    Attributes
    ----------
    objective_function : Callable
        Function to optimize (minimize)
    parameter_space : Dict[str, Tuple[float, float]]
        Parameter bounds: {name: (low, high)}
    n_initial_points : int
        Number of random initialization points
    n_iterations : int
        Total number of iterations
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print optimization progress

    Methods
    -------
    optimize()
        Run Bayesian optimization
    save_state(path)
        Save optimizer state to disk
    load_state(path)
        Load optimizer state from disk

    Examples
    --------
    >>> def objective(params):
    ...     return -params['sharpe_ratio']  # Minimize negative Sharpe
    >>> optimizer = BayesianOptimizer(objective, parameter_space)
    >>> result = optimizer.optimize()
    >>> print(result.best_params)

    Notes
    -----
    Uses scikit-optimize with:
    - Gaussian Process surrogate (Matern 5/2 kernel)
    - Expected Improvement acquisition function
    - Latin Hypercube Sampling for initialization
    """

    def __init__(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameter_space: Dict[str, Tuple[float, float]],
        n_initial_points: int = 10,
        n_iterations: int = 50,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Bayesian optimizer.

        Parameters
        ----------
        objective_function : Callable
            Function to minimize: f(params) -> score
        parameter_space : Dict[str, Tuple[float, float]]
            Parameter bounds: {name: (low, high)}
        n_initial_points : int
            Number of random points for initialization
        n_iterations : int
            Total number of optimization iterations
        random_state : int
            Random seed
        verbose : bool
            Print progress
        """
        self.logger = StructuredLogger(__name__)

        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.verbose = verbose

        # Convert parameter space to skopt format
        self.dimensions = [
            Real(low, high, name=name)
            for name, (low, high) in parameter_space.items()
        ]

        self.param_names = list(parameter_space.keys())

        # State tracking
        self.all_params: List[Dict[str, float]] = []
        self.all_scores: List[float] = []
        self.convergence_history: List[float] = []

        self.logger.info(
            "BayesianOptimizer initialized",
            n_params=len(parameter_space),
            param_names=self.param_names,
            n_iterations=n_iterations,
            n_initial_points=n_initial_points
        )

    @timed_execution
    def optimize(self) -> OptimizationResult:
        """
        Run Bayesian optimization.

        Returns
        -------
        OptimizationResult
            Optimization results with best parameters

        Examples
        --------
        >>> result = optimizer.optimize()
        >>> print(f"Best params: {result.best_params}")
        >>> print(f"Best score: {result.best_score:.4f}")

        Notes
        -----
        Uses scikit-optimize's gp_minimize with:
        - Gaussian Process surrogate (Matern kernel)
        - Expected Improvement (EI) acquisition
        - Latin Hypercube Sampling for initialization
        """
        self.logger.info("Starting Bayesian optimization")

        # Define objective wrapper
        @use_named_args(self.dimensions)
        def objective_wrapper(**params):
            """Wrapper to track evaluations."""
            score = self.objective_function(params)

            # Track evaluation
            self.all_params.append(params.copy())
            self.all_scores.append(float(score))

            # Update convergence history (best score so far)
            if len(self.convergence_history) == 0:
                self.convergence_history.append(float(score))
            else:
                best_so_far = min(self.convergence_history[-1], float(score))
                self.convergence_history.append(best_so_far)

            if self.verbose:
                self.logger.info(
                    "Evaluation",
                    iteration=len(self.all_scores),
                    params=params,
                    score=round(float(score), 4),
                    best_so_far=round(self.convergence_history[-1], 4)
                )

            return score

        # Run optimization
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.dimensions,
            n_calls=self.n_iterations,
            n_initial_points=self.n_initial_points,
            acq_func="EI",  # Expected Improvement
            random_state=self.random_state,
            verbose=False
        )

        # Extract best parameters
        best_params = {
            name: float(value)
            for name, value in zip(self.param_names, result.x)
        }

        best_score = float(result.fun)

        self.logger.info(
            "Optimization completed",
            best_score=round(best_score, 4),
            best_params=best_params,
            total_evaluations=len(self.all_scores)
        )

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_params=self.all_params,
            all_scores=self.all_scores,
            n_iterations=len(self.all_scores),
            convergence_history=self.convergence_history
        )

    def save_state(self, path: str) -> None:
        """
        Save optimizer state to disk.

        Parameters
        ----------
        path : str
            Path to save state (pickle file)

        Examples
        --------
        >>> optimizer.save_state("results/optimizer_state.pkl")
        """
        state = {
            "parameter_space": self.parameter_space,
            "all_params": self.all_params,
            "all_scores": self.all_scores,
            "convergence_history": self.convergence_history,
            "n_iterations": self.n_iterations,
            "n_initial_points": self.n_initial_points,
            "random_state": self.random_state
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        self.logger.info("Optimizer state saved", path=path)

    @classmethod
    def load_state(
        cls,
        path: str,
        objective_function: Callable[[Dict[str, float]], float]
    ) -> "BayesianOptimizer":
        """
        Load optimizer state from disk.

        Parameters
        ----------
        path : str
            Path to saved state (pickle file)
        objective_function : Callable
            Objective function to use

        Returns
        -------
        BayesianOptimizer
            Restored optimizer instance

        Examples
        --------
        >>> optimizer = BayesianOptimizer.load_state(
        ...     "results/optimizer_state.pkl",
        ...     objective_function
        ... )
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        optimizer = cls(
            objective_function=objective_function,
            parameter_space=state["parameter_space"],
            n_initial_points=state["n_initial_points"],
            n_iterations=state["n_iterations"],
            random_state=state["random_state"]
        )

        optimizer.all_params = state["all_params"]
        optimizer.all_scores = state["all_scores"]
        optimizer.convergence_history = state["convergence_history"]

        optimizer.logger.info("Optimizer state loaded", path=path)

        return optimizer


# Standalone utility function

def expected_improvement(
    X: npt.NDArray[np.float64],
    model: object,
    y_best: float,
    xi: float = 0.01
) -> npt.NDArray[np.float64]:
    """
    Calculate Expected Improvement acquisition function.

    Parameters
    ----------
    X : np.ndarray
        Candidate points to evaluate (n_points, n_features)
    model : object
        Gaussian Process model with predict method
    y_best : float
        Best observed value so far
    xi : float
        Exploration-exploitation trade-off parameter

    Returns
    -------
    np.ndarray
        Expected improvement values (n_points,)

    Examples
    --------
    >>> ei = expected_improvement(X_candidates, gp_model, y_best=0.5)

    Notes
    -----
    EI(x) = E[max(y_best - f(x) - ξ, 0)]
          = (y_best - μ(x) - ξ) * Φ(Z) + σ(x) * φ(Z)

    where:
        Z = (y_best - μ(x) - ξ) / σ(x)
        Φ = CDF of standard normal
        φ = PDF of standard normal
    """
    from scipy.stats import norm

    # Get predictions
    mu, sigma = model.predict(X, return_std=True)

    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)

    # Calculate Z score
    Z = (y_best - mu - xi) / sigma

    # Expected Improvement
    ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

    return ei
