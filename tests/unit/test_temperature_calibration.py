"""Unit tests for TemperatureCalibrator and strict Rank Invariance (RULE-03)."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from quantautoml.common.types import TaskType
from quantautoml.ensemble.calibration import CalibrationResult, TemperatureCalibrator
from quantautoml.metrics.calibration import calc_ece


def test_temperature_calibration_binary_overconfident() -> None:
    # Model is overconfident: logits are large, producing extreme probabilities
    rng = np.random.default_rng(42)
    N = 1000
    # True probability of y=1 is roughly 0.6
    z = rng.normal(loc=1.5, scale=2.0, size=N).astype(np.float32)
    # Binary labels generated with sigmoid(z * 0.3) so true T ~ 3.3
    true_p = 1.0 / (1.0 + np.exp(-0.3 * z))
    y_true = rng.binomial(1, true_p).astype(np.int32)

    raw_p = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    raw_ece = calc_ece(y_true, raw_p, n_bins=10)

    calibrator = TemperatureCalibrator(task_type=TaskType.BINARY_CLASSIFICATION, random_state=42)
    calibrator.fit(raw_p, y_true)

    res = calibrator.get_result()
    assert isinstance(res, CalibrationResult)
    assert res.temperature > 1.0  # Temperature scaling cools down overconfidence
    assert res.rank_ic_preserved is True

    calibrated_p = calibrator.calibrate(raw_p)
    calibrated_ece = calc_ece(y_true, calibrated_p, n_bins=10)

    # Calibrated ECE should improve substantially over raw ECE
    assert calibrated_ece < raw_ece
    assert calibrated_ece < 0.05  # Meets the < 0.05 ECE spec requirement


def test_strict_rank_invariance_contract_rule_03() -> None:
    """RULE-03: Temperature Scaling must preserve 100% of Rank IC with 0 ties created."""
    rng = np.random.default_rng(123)
    N = 2000
    raw_p = rng.uniform(0.01, 0.99, size=N).astype(np.float32)
    y_true = rng.binomial(1, raw_p).astype(np.int32)

    calibrator = TemperatureCalibrator(task_type=TaskType.BINARY_CLASSIFICATION)
    calibrator.fit(raw_p, y_true)

    calibrated_p = calibrator.calibrate(raw_p)

    # 1. Verification method inside calibrator
    assert calibrator.verify_rank_invariance(raw_p, calibrated_p) is True

    # 2. Strict rank correlation test
    rank_corr, _ = spearmanr(raw_p, calibrated_p)
    assert pytest.approx(rank_corr, abs=1e-6) == 1.0

    # 3. Exact order invariance: for any pair i, j: raw_p[i] > raw_p[j] <=> cal_p[i] > cal_p[j]
    sample_i = rng.integers(0, N, size=100)
    sample_j = rng.integers(0, N, size=100)
    for i, j in zip(sample_i, sample_j):
        if raw_p[i] > raw_p[j]:
            assert calibrated_p[i] > calibrated_p[j]
        elif raw_p[i] < raw_p[j]:
            assert calibrated_p[i] < calibrated_p[j]


def test_temperature_calibration_multiclass() -> None:
    rng = np.random.default_rng(42)
    N = 600
    C = 3
    # Overconfident uncalibrated multiclass logits
    logits = rng.normal(0, 3.0, size=(N, C)).astype(np.float32)
    # Convert to probs with softmax
    exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    raw_probs = (exp_z / np.sum(exp_z, axis=1, keepdims=True)).astype(np.float32)

    y_true = rng.integers(0, C, size=N, dtype=np.int32)

    calibrator = TemperatureCalibrator(task_type=TaskType.MULTICLASS_CLASSIFICATION)
    calibrator.fit(raw_probs, y_true)

    cal_probs = calibrator.calibrate(raw_probs)
    assert cal_probs.shape == (N, C)
    assert cal_probs.dtype == np.float32
    assert np.allclose(np.sum(cal_probs, axis=1), 1.0, atol=1e-5)

    # Argmax prediction must remain 100% identical
    assert np.array_equal(np.argmax(raw_probs, axis=1), np.argmax(cal_probs, axis=1))


def test_temperature_calibration_validation() -> None:
    calibrator = TemperatureCalibrator()

    with pytest.raises(ValueError, match="empty"):
        calibrator.fit(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32))

    with pytest.raises(ValueError, match="Shape mismatch"):
        calibrator.fit(np.zeros(10, dtype=np.float32), np.zeros(15, dtype=np.int32))
