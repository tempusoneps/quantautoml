"""Unit tests for Expected Calibration Error (ECE) metric."""

import numpy as np
import pytest

from quantautoml.metrics import calc_ece


def test_calc_ece_perfect_calibration_binary() -> None:
    # 1000 samples: half y=1 with prob=1.0, half y=0 with prob=0.0
    y_true = np.array([1] * 500 + [0] * 500, dtype=np.int32)
    y_prob = np.array([1.0] * 500 + [0.0] * 500, dtype=np.float32)

    ece = calc_ece(y_true, y_prob, n_bins=10)
    assert pytest.approx(ece, abs=1e-5) == 0.0


def test_calc_ece_overconfident_uncalibrated_binary() -> None:
    # Model is extremely confident (prob=0.99 for all), but ground truth is all 0
    y_true = np.zeros(500, dtype=np.int32)
    y_prob = np.full(500, 0.99, dtype=np.float32)

    ece = calc_ece(y_true, y_prob, n_bins=10)
    # Predicted label is 1 (confidence 0.99), actual label is 0 (accuracy 0.0)
    # ECE should be |0.0 - 0.99| = 0.99
    assert pytest.approx(ece, abs=1e-3) == 0.99


def test_calc_ece_multiclass_2d() -> None:
    # 3 classes, 300 samples
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=300, dtype=np.int32)
    # One-hot like perfect probabilities
    y_prob = np.zeros((300, 3), dtype=np.float32)
    for i, c in enumerate(y_true):
        y_prob[i, c] = 1.0

    ece = calc_ece(y_true, y_prob, n_bins=10)
    assert pytest.approx(ece, abs=1e-5) == 0.0


def test_calc_ece_range_and_bins() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=1000, dtype=np.int32)
    y_prob = rng.uniform(0.0, 1.0, size=1000).astype(np.float32)

    ece = calc_ece(y_true, y_prob, n_bins=15)
    assert 0.0 <= ece <= 1.0


def test_calc_ece_empty_or_invalid() -> None:
    with pytest.raises(ValueError, match="Empty"):
        calc_ece(np.array([], dtype=np.int32), np.array([], dtype=np.float32))

    with pytest.raises(ValueError, match="Shape mismatch"):
        calc_ece(np.array([1, 0]), np.array([0.5, 0.5, 0.5]))

    with pytest.raises(ValueError, match="NaN"):
        calc_ece(np.array([1, 0]), np.array([np.nan, 0.5]))

    with pytest.raises(ValueError, match="Probabilities must be in"):
        calc_ece(np.array([1, 0]), np.array([1.5, -0.2]))
