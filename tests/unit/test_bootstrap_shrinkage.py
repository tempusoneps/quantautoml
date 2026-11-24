"""Unit tests for BootstrapShrinkageEngine and Hurdle Fallback (RULE-05)."""

import numpy as np
import pytest

from quantautoml.common.config import EnsembleConfig
from quantautoml.ensemble.bootstrap import BootstrapShrinkageEngine, EnsembleWeightsResult


def test_bootstrap_shrinkage_single_model() -> None:
    cfg = EnsembleConfig()
    engine = BootstrapShrinkageEngine(cfg)

    preds = np.random.randn(200, 1).astype(np.float32)
    y_true = np.random.randn(200).astype(np.float32)

    res = engine.fit(preds, y_true)
    assert isinstance(res, EnsembleWeightsResult)
    assert res.weights == [1.0]
    assert res.raw_weights == [1.0]
    assert res.shrunk_weights == [1.0]
    assert res.is_ensemble_accepted is True


def test_bootstrap_shrinkage_calculation() -> None:
    # 3 models, 500 samples
    rng = np.random.default_rng(42)
    N = 500
    y_true = rng.standard_normal(N).astype(np.float32)
    m1 = (y_true + 0.3 * rng.standard_normal(N)).astype(np.float32)
    m2 = (y_true + 0.6 * rng.standard_normal(N)).astype(np.float32)
    m3 = (y_true + 0.9 * rng.standard_normal(N)).astype(np.float32)
    preds = np.column_stack([m1, m2, m3])

    # n_bootstrap=20 in test for fast execution
    cfg = EnsembleConfig(n_bootstrap=20, shrinkage_factor=0.20, random_state=42)
    engine = BootstrapShrinkageEngine(cfg)

    res = engine.fit(preds, y_true)
    assert len(res.weights) == 3
    assert len(res.raw_weights) == 3
    assert len(res.shrunk_weights) == 3
    assert len(res.bootstrap_std) == 3
    assert pytest.approx(sum(res.shrunk_weights), abs=1e-5) == 1.0

    # Shrinkage pulls weights towards 1/K = 0.333
    equal_w = 1.0 / 3.0
    for i in range(3):
        # Shrunk weights are strictly between raw median and equal weights
        if res.raw_weights[i] > equal_w:
            assert res.raw_weights[i] >= res.shrunk_weights[i] >= equal_w
        elif res.raw_weights[i] < equal_w:
            assert res.raw_weights[i] <= res.shrunk_weights[i] <= equal_w


def test_hurdle_rule_acceptance_and_fallback() -> None:
    rng = np.random.default_rng(42)
    N = 1000

    # Target is signal
    target = rng.standard_normal(N).astype(np.float32)

    # Model 1 & Model 2: strong independent complementary components of target
    # Model 3: high noise
    e1 = rng.standard_normal(N).astype(np.float32)
    e2 = rng.standard_normal(N).astype(np.float32)
    e3 = rng.standard_normal(N).astype(np.float32)

    m1 = (target + 0.8 * e1).astype(np.float32)
    m2 = (target + 0.8 * e2).astype(np.float32)
    m3 = (0.1 * target + 2.0 * e3).astype(np.float32)
    preds = np.column_stack([m1, m2, m3])

    # Case A: Ensemble gives tiny weight to m3 and balances m1 & m2, beating equal-weight by > 5%
    cfg_accept = EnsembleConfig(
        n_bootstrap=20,
        hurdle_ratio=0.03,
        shrinkage_factor=0.05,
        random_state=42,
    )
    engine_accept = BootstrapShrinkageEngine(cfg_accept)
    res_accept = engine_accept.fit(preds, target)
    assert res_accept.is_ensemble_accepted is True
    assert res_accept.fallback_reason is None
    assert pytest.approx(sum(res_accept.weights), abs=1e-5) == 1.0

    # Case B: Require extreme hurdle (e.g. +500% gain) -> must trigger fallback
    cfg_reject = EnsembleConfig(n_bootstrap=20, hurdle_ratio=5.0, random_state=42)
    engine_reject = BootstrapShrinkageEngine(cfg_reject)
    res_reject = engine_reject.fit(preds, target)
    assert res_reject.is_ensemble_accepted is False
    assert res_reject.fallback_reason is not None
    # Weights should be fallback weights summing to 1.0
    assert pytest.approx(sum(res_reject.weights), abs=1e-5) == 1.0


def test_bootstrap_shrinkage_validation() -> None:
    cfg = EnsembleConfig()
    engine = BootstrapShrinkageEngine(cfg)

    with pytest.raises(ValueError, match="empty"):
        engine.fit(np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32))

    with pytest.raises(ValueError, match="Shape mismatch"):
        engine.fit(np.zeros((10, 2), dtype=np.float32), np.zeros(20, dtype=np.float32))


def test_block_bootstrap_preserves_contiguous_blocks() -> None:
    """Audit #7: Moving Block Bootstrap must sample contiguous time-series blocks."""
    from quantautoml.ensemble.bootstrap import _sample_block_indices

    rng = np.random.default_rng(7)
    idx = _sample_block_indices(rng, n_samples=500, target_size=400, block_size=20)
    assert idx.shape == (400,)
    assert np.all((idx >= 0) & (idx < 500))
    # Every block of 20 elements must be strictly consecutive (diff == 1)
    for start in range(0, 400, 20):
        block = idx[start : start + 20]
        assert np.all(np.diff(block) == 1)


def test_ensemble_config_bootstrap_block_size() -> None:
    cfg = EnsembleConfig(bootstrap_block_size=30)
    assert cfg.bootstrap_block_size == 30
