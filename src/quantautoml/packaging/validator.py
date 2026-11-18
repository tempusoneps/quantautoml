"""Numerical Precision Contract (RULE-06): max|y_py - y_onnx| < tol on float32 outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.exceptions import PrecisionValidationError


def validate_numerical_precision(
    y_py: NDArray[np.float32],
    y_onnx: NDArray[np.float32],
    *,
    tol: float = 1e-4,
) -> float:
    """Return ``max|y_py - y_onnx|``; raise ``PrecisionValidationError`` unless it is ``< tol``."""
    if y_py.dtype != np.float32 or y_onnx.dtype != np.float32:
        raise PrecisionValidationError(
            f"outputs must be float32, got {y_py.dtype} and {y_onnx.dtype}"
        )
    if y_py.shape != y_onnx.shape:
        raise PrecisionValidationError(f"shape mismatch: {y_py.shape} vs {y_onnx.shape}")
    if y_py.size == 0:
        return 0.0
    if not (np.isfinite(y_py).all() and np.isfinite(y_onnx).all()):
        raise PrecisionValidationError("outputs contain NaN/inf; cannot certify precision")
    diff = float(np.max(np.abs(y_py.astype(np.float64) - y_onnx.astype(np.float64))))
    if not diff < tol:
        raise PrecisionValidationError(f"max|y_py - y_onnx| = {diff:.3e} >= tol {tol:.1e}")
    return diff
