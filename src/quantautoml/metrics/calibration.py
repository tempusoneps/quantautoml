"""Metrics for evaluating probability calibration error (ECE)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def calc_ece(
    y_true: NDArray[np.int32 | np.float32],
    y_prob: NDArray[np.float32],
    *,
    n_bins: int = 10,
) -> float:
    """Calculate the Expected Calibration Error (ECE) for binary or multiclass probabilities.

    Parameters
    ----------
    y_true : NDArray[np.int32 | np.float32]
        1D array of ground-truth class labels (integers or binary flags).
    y_prob : NDArray[np.float32]
        1D array of class-1 probabilities for binary tasks, or
        2D array of shape (N, C) containing class probabilities for multiclass tasks.
    n_bins : int, default=10
        Number of equal-width bins in [0, 1].

    Returns
    -------
    float
        Expected Calibration Error in [0, 1].
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob, dtype=np.float32)

    if y_t.size == 0 or y_p.size == 0:
        raise ValueError("Empty input provided to calc_ece.")

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    if np.isnan(y_t).any() or np.isnan(y_p).any() or np.isinf(y_p).any():
        raise ValueError("Input contains NaN or Inf values.")

    if (y_p < -1e-6).any() or (y_p > 1.0 + 1e-6).any():
        raise ValueError("Probabilities must be in range [0, 1].")

    y_p = np.clip(y_p, 0.0, 1.0)

    if y_p.ndim == 1:
        if len(y_t) != len(y_p):
            raise ValueError(f"Shape mismatch: y_true has len {len(y_t)} but y_prob has len {len(y_p)}.")
        preds = (y_p >= 0.5).astype(np.int32)
        conf = np.maximum(y_p, 1.0 - y_p)
    elif y_p.ndim == 2:
        if len(y_t) != y_p.shape[0]:
            raise ValueError(f"Shape mismatch: y_true has len {len(y_t)} but y_prob has shape {y_p.shape}.")
        preds = np.argmax(y_p, axis=1).astype(np.int32)
        conf = np.max(y_p, axis=1)
    else:
        raise ValueError(f"y_prob must be 1D or 2D, got ndim={y_p.ndim}.")

    n_samples = len(y_t)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == 0:
            in_bin = (conf >= bin_lower) & (conf <= bin_upper)
        else:
            in_bin = (conf > bin_lower) & (conf <= bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            acc = np.mean(y_t[in_bin] == preds[in_bin])
            avg_conf = np.mean(conf[in_bin])
            ece += (bin_count / n_samples) * np.abs(acc - avg_conf)

    return float(ece)
