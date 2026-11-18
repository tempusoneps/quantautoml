"""validate_numerical_precision enforces RULE-06: max|y_py - y_onnx| < tol."""

import numpy as np
import pytest

from quantautoml.common.exceptions import PrecisionValidationError
from quantautoml.packaging.validator import validate_numerical_precision


def _f32(*v: float) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def test_pass_returns_max_abs_diff() -> None:
    d = validate_numerical_precision(_f32(0.1, 0.5), _f32(0.1, 0.50005))
    assert d == pytest.approx(5e-5, rel=1e-2)


def test_at_tolerance_fails() -> None:
    with pytest.raises(PrecisionValidationError):
        validate_numerical_precision(_f32(0.0), _f32(1e-4), tol=float(np.float32(1e-4)))


def test_above_tolerance_fails() -> None:
    with pytest.raises(PrecisionValidationError):
        validate_numerical_precision(_f32(0.0, 1.0), _f32(0.0, 1.001))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(PrecisionValidationError):
        validate_numerical_precision(_f32(0.0, 1.0), _f32(0.0))


def test_float64_input_raises() -> None:
    with pytest.raises(PrecisionValidationError):
        validate_numerical_precision(np.zeros(2), np.zeros(2, dtype=np.float32))


def test_nan_on_one_side_raises() -> None:
    with pytest.raises(PrecisionValidationError):
        validate_numerical_precision(_f32(0.1, np.nan), _f32(0.1, 0.2))


def test_nan_on_both_sides_raises() -> None:
    with pytest.raises(PrecisionValidationError):
        validate_numerical_precision(_f32(np.nan), _f32(np.nan))


def test_empty_arrays_pass_with_zero() -> None:
    assert validate_numerical_precision(_f32(), _f32()) == 0.0
