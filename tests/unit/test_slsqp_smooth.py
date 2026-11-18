"""Unit tests for SmoothSLSQPOptimizer (C2 Objective)."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from quantautoml.common.exceptions import OptimizationConvergenceError
from quantautoml.ensemble.slsqp import SmoothSLSQPOptimizer


def test_slsqp_single_model() -> None:
    opt = SmoothSLSQPOptimizer(min_weight=0.05, max_weight=0.60)
    preds = np.random.randn(100, 1).astype(np.float32)
    y_true = np.random.randn(100).astype(np.float32)

    w = opt.optimize(preds, y_true)
    assert w.shape == (1,)
    assert w.dtype == np.float32
    assert w[0] == 1.0


def test_slsqp_three_models_bounds_and_sum() -> None:
    rng = np.random.default_rng(42)
    N = 1000
    K = 3
    # Model 1 has high signal with y
    # Model 2 has moderate signal
    # Model 3 has pure noise
    y_true = rng.standard_normal(N).astype(np.float32)
    m1 = (y_true + 0.5 * rng.standard_normal(N)).astype(np.float32)
    m2 = (y_true + 1.5 * rng.standard_normal(N)).astype(np.float32)
    m3 = rng.standard_normal(N).astype(np.float32)

    preds = np.column_stack([m1, m2, m3])

    opt = SmoothSLSQPOptimizer(min_weight=0.05, max_weight=0.60, random_state=42)
    w = opt.optimize(preds, y_true)

    assert w.shape == (3,)
    assert w.dtype == np.float32
    assert pytest.approx(float(np.sum(w)), abs=1e-5) == 1.0

    # Bounds check: all weights must be in [0.05, 0.60]
    for weight in w:
        assert 0.05 - 1e-5 <= weight <= 0.60 + 1e-5

    # Model 1 is best, so it should receive highest weight (capped near max_weight=0.60)
    assert w[0] >= w[1]
    assert w[1] >= w[2]
    assert pytest.approx(w[0], abs=0.02) == 0.60


def test_slsqp_convergence_speed() -> None:
    import time

    rng = np.random.default_rng(123)
    N = 2000
    K = 3
    preds = rng.standard_normal((N, K)).astype(np.float32)
    y_true = rng.standard_normal(N).astype(np.float32)

    opt = SmoothSLSQPOptimizer(min_weight=0.05, max_weight=0.60)

    start = time.perf_counter()
    w = opt.optimize(preds, y_true)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Smooth C2 objective with analytical gradient must converge well under 50ms
    assert elapsed_ms < 50.0
    assert pytest.approx(float(np.sum(w)), abs=1e-5) == 1.0


def test_slsqp_fallback_on_constant_data() -> None:
    opt = SmoothSLSQPOptimizer(min_weight=0.05, max_weight=0.60)
    # Target is constant (variance == 0)
    preds = np.ones((100, 3), dtype=np.float32)
    y_true = np.zeros(100, dtype=np.float32)

    # Should safely fallback to equal weights [1/K, 1/K, 1/K] without crashing
    w = opt.optimize(preds, y_true)
    assert w.shape == (3,)
    assert pytest.approx(w[0], abs=1e-5) == 1.0 / 3.0
    assert pytest.approx(w[1], abs=1e-5) == 1.0 / 3.0
    assert pytest.approx(w[2], abs=1e-5) == 1.0 / 3.0


def test_slsqp_input_validation() -> None:
    opt = SmoothSLSQPOptimizer()

    with pytest.raises(ValueError, match="empty"):
        opt.optimize(np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32))

    with pytest.raises(ValueError, match="Shape mismatch"):
        opt.optimize(np.zeros((10, 2), dtype=np.float32), np.zeros(15, dtype=np.float32))

    with pytest.raises(ValueError, match="2D"):
        opt.optimize(np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32))
