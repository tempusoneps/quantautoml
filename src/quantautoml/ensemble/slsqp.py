"""Smooth SLSQP Convex Optimization on Rank-Transformed Data (C2 Differentiable Objective)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import rankdata

from quantautoml.common.exceptions import OptimizationConvergenceError


class SmoothSLSQPOptimizer:
    """SLSQP convex weight optimizer with C2 differentiable objective.

    Solves for non-negative weights summing to 1 that maximize the Pearson correlation
    between the linear combination of rank-transformed predictions and the
    rank-transformed target (smooth surrogate of Spearman Rank IC).

    Parameters
    ----------
    min_weight : float, default=0.05
        Lower bound for each model weight.
    max_weight : float, default=0.60
        Upper bound for each model weight.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        min_weight: float = 0.05,
        max_weight: float = 0.60,
        random_state: int = 42,
    ) -> None:
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.random_state = int(random_state)

    def optimize(
        self,
        predictions: NDArray[np.float32],
        y_true: NDArray[np.float32 | np.int32],
    ) -> NDArray[np.float32]:
        """Optimize ensemble weights on Cal-A predictions (chronologically before Calibration).

        Parameters
        ----------
        predictions : NDArray[np.float32]
            2D array of shape (N, K) with model predictions.
        y_true : NDArray[np.float32 | np.int32]
            1D array of shape (N,) with ground truth labels.

        Returns
        -------
        NDArray[np.float32]
            1D array of shape (K,) with optimal convex weights summing to 1.
        """
        preds = np.asarray(predictions, dtype=np.float32)
        target = np.asarray(y_true, dtype=np.float32)

        if preds.ndim != 2:
            raise ValueError(f"predictions must be a 2D array of shape (N, K), got ndim={preds.ndim}.")
        if preds.shape[0] == 0 or preds.shape[1] == 0 or target.shape[0] == 0:
            raise ValueError("Input predictions or target cannot be empty.")
        if preds.shape[0] != target.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions has {preds.shape[0]} rows but target has {target.shape[0]} rows."
            )

        n_samples, n_models = preds.shape

        if n_models == 1:
            return np.array([1.0], dtype=np.float32)

        # Equal weights fallback vector
        equal_w = np.full(n_models, 1.0 / n_models, dtype=np.float32)

        # Step 1: Rank-transformation to [0, 1]
        try:
            rank_target = rankdata(target, method="average") / n_samples
            rank_preds = np.column_stack(
                [rankdata(preds[:, j], method="average") / n_samples for j in range(n_models)]
            )
        except Exception:
            return equal_w

        # Step 2: Center variables for Pearson correlation
        y_c = rank_target - np.mean(rank_target)
        sigma_y = float(np.std(y_c))
        if sigma_y < 1e-9:
            return equal_w

        m_centered = rank_preds - np.mean(rank_preds, axis=0)
        col_stds = np.std(m_centered, axis=0)
        if np.all(col_stds < 1e-9):
            return equal_w

        # Covariance vector c and Covariance matrix Sigma
        c = (m_centered.T @ y_c) / n_samples  # shape (K,)
        sigma = (m_centered.T @ m_centered) / n_samples  # shape (K, K)

        eps = 1e-12

        def objective(w: NDArray[np.float64]) -> float:
            cov = float(w @ c)
            var = float(w @ sigma @ w) + eps
            std_comb = np.sqrt(var)
            corr = cov / (sigma_y * std_comb)
            return -corr

        def jacobian(w: NDArray[np.float64]) -> NDArray[np.float64]:
            cov = float(w @ c)
            var = float(w @ sigma @ w) + eps
            std_comb = np.sqrt(var)
            grad = (c / std_comb) - (cov * (sigma @ w)) / (std_comb**3)
            return -(grad / sigma_y)

        # Feasibility check for bounds
        lb = min(self.min_weight, 0.999 / n_models)
        ub = max(self.max_weight, 1.001 / n_models)
        if lb > ub:
            lb, ub = 0.0, 1.0

        bounds = [(lb, ub) for _ in range(n_models)]
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones_like(w),
            }
        ]

        w0 = np.full(n_models, 1.0 / n_models, dtype=np.float64)

        try:
            res = minimize(
                fun=objective,
                x0=w0,
                jac=jacobian,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            if not res.success or np.any(np.isnan(res.x)):
                return equal_w

            w_opt = np.clip(res.x, lb, ub)
            sum_w = np.sum(w_opt)
            if sum_w <= 0.0 or np.isnan(sum_w):
                return equal_w
            w_opt = w_opt / sum_w
            return w_opt.astype(np.float32)
        except Exception:
            return equal_w
