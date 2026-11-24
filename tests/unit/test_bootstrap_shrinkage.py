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


def test_resolve_ic_window_clamps_to_at_least_four_windows_when_possible() -> None:
    from quantautoml.ensemble.bootstrap import _resolve_ic_window

    assert _resolve_ic_window(380, 500) == 95
    assert _resolve_ic_window(7585, None) == 500
    assert _resolve_ic_window(7585, 500) == 500
    assert _resolve_ic_window(100, None) == 10
    with pytest.raises(ValueError, match="ic_window"):
        _resolve_ic_window(380, 2)


def test_calc_series_ic_sharpe_does_not_mix_rank_ic_when_mean_is_zero() -> None:
    from quantautoml.ensemble.bootstrap import _calc_series_ic_sharpe
    from quantautoml.metrics.ic import calc_rank_ic

    # 4 windows of 25: +1, -1, +1, -1 within-window IC (mean=0, std>0),
    # while shifting levels across windows makes overall Rank IC strongly positive.
    w = 25
    base = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y_parts = [base + 10.0 * i for i in range(4)]
    p_parts = [
        base + 10.0 * 0,
        (1.0 - base) + 10.0 * 1,
        base + 10.0 * 2,
        (1.0 - base) + 10.0 * 3,
    ]
    y = np.concatenate(y_parts)
    pred = np.concatenate(p_parts)
    assert calc_rank_ic(pred, y)[0] > 0.5
    score = _calc_series_ic_sharpe(pred, y, ic_window=w)
    assert score == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_adaptive_window_scales_on_long_intraday_series() -> None:
    rng = np.random.default_rng(77)
    n = 7500
    y = rng.standard_normal(n).astype(np.float32)
    preds = np.column_stack(
        [
            (y + 0.6 * rng.standard_normal(n)).astype(np.float32),
            (y + 0.8 * rng.standard_normal(n)).astype(np.float32),
            (y + 1.0 * rng.standard_normal(n)).astype(np.float32),
        ]
    )
    cfg = EnsembleConfig(n_bootstrap=10, random_state=42)
    res = BootstrapShrinkageEngine(cfg).fit(preds, y)
    assert res.ic_window_used == 500
    assert len(res.single_model_scores) == 3
    assert np.isfinite(res.overall_rank_ic) and res.overall_rank_ic > 0.0


def test_bootstrap_explicit_ic_window_and_single_model_branch() -> None:
    rng = np.random.default_rng(88)
    n = 1500
    y = rng.standard_normal(n).astype(np.float32)
    m1 = (y + 0.5 * rng.standard_normal(n)).astype(np.float32)
    m2 = (y + 0.7 * rng.standard_normal(n)).astype(np.float32)
    m3 = (y + 0.9 * rng.standard_normal(n)).astype(np.float32)

    cfg = EnsembleConfig(n_bootstrap=10, ic_window=300, random_state=42)
    engine = BootstrapShrinkageEngine(cfg)

    res_single = engine.fit(m1[:, None], y)
    assert res_single.ic_window_used == 300
    assert len(res_single.single_model_scores) == 1
    assert np.isfinite(res_single.overall_rank_ic) and res_single.overall_rank_ic > 0.0

    res_multi = engine.fit(np.column_stack([m1, m2, m3]), y)
    assert res_multi.ic_window_used == 300
    assert len(res_multi.single_model_scores) == 3
    assert np.isfinite(res_multi.overall_rank_ic) and res_multi.overall_rank_ic > 0.0
