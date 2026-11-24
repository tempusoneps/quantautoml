"""Unit tests for F2b: StationarityFilter (lag-1 autocorrelation + temporal mean drift)."""

from __future__ import annotations

import numpy as np
import pytest

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.features.base import FitContext
from quantautoml.features.stationarity import StationarityFilter


def _cfg(**overrides: object) -> FeaturesConfig:
    base: dict[str, object] = {
        "ic_horizons": {1: "fwd_ret_1", 5: "fwd_ret_5"},
        "target_horizon": 1,
    }
    base.update(overrides)
    return FeaturesConfig(**base)  # type: ignore[arg-type]


def _ctx(names: tuple[str, ...]) -> FitContext:
    return FitContext(
        feature_names=names,
        labels_aux={},
        task_type=TaskType.BINARY_CLASSIFICATION,
    )


def test_stage_name_is_f2b_stationarity() -> None:
    flt = StationarityFilter(_cfg())
    assert flt.name == "f2b_stationarity"


def test_drops_random_walk_and_linear_trend_keeps_stationary_and_slow_oscillator() -> None:
    rng = np.random.default_rng(42)
    n = 4000
    t = np.arange(n, dtype=np.float64)

    # 1. Pure iid stationary noise
    stationary = rng.normal(size=n)
    # 2. Slow stationary oscillator (high rho1 ~ 0.998, but bounded -> low temporal drift)
    slow_osc = np.sin(2.0 * np.pi * t / 100.0) + 0.05 * rng.normal(size=n)
    # 3. Linear price-like trend (rho1 ~ 1.0 AND high temporal drift > 2.0)
    price_trend = 1000.0 + 0.25 * t + rng.normal(scale=2.0, size=n)
    # 4. Random walk with drift (cumulative index like PVI/OBV)
    cum_index = np.cumsum(0.05 + rng.normal(scale=0.5, size=n))

    names = ("stationary", "slow_osc", "price_trend", "cum_index")
    X = np.column_stack([stationary, slow_osc, price_trend, cum_index]).astype(np.float32)
    y = rng.integers(0, 2, size=n, dtype=np.int32)

    flt = StationarityFilter(_cfg()).fit(X, y, _ctx(names))

    assert flt.output_names_ == ("stationary", "slow_osc")
    assert flt.report_ is not None
    assert flt.report_.stage == "f2b_stationarity"
    assert flt.report_.n_in == 4
    assert flt.report_.n_out == 2
    assert set(flt.report_.dropped.keys()) == {"price_trend", "cum_index"}
    for reason in flt.report_.dropped.values():
        assert "non-stationary" in reason
        assert "lag1_autocorr" in reason
        assert "temporal_drift" in reason

    out = flt.transform(X)
    assert out.dtype == np.float32
    assert out.shape == (n, 2)
    assert np.array_equal(out, X[:, [0, 1]])


def test_disabled_when_max_temporal_drift_is_none() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    t = np.arange(n, dtype=np.float64)
    X = np.column_stack([rng.normal(size=n), 100.0 + t]).astype(np.float32)
    y = rng.integers(0, 2, size=n, dtype=np.int32)

    flt = StationarityFilter(_cfg(max_temporal_drift=None)).fit(X, y, _ctx(("a", "trend")))

    assert flt.output_names_ == ("a", "trend")
    assert flt.report_ is not None
    assert flt.report_.dropped == {}
    assert any("disabled" in note for note in flt.report_.notes)
    assert np.array_equal(flt.transform(X), X)


def test_handles_nans_and_constant_columns_without_error() -> None:
    rng = np.random.default_rng(7)
    n = 800
    col_nan = rng.normal(size=n)
    col_nan[::10] = np.nan
    col_const = np.full(n, 3.0)
    col_all_nan = np.full(n, np.nan)
    X = np.column_stack([col_nan, col_const, col_all_nan]).astype(np.float32)
    y = rng.integers(0, 2, size=n, dtype=np.int32)

    flt = StationarityFilter(_cfg()).fit(X, y, _ctx(("with_nan", "const", "all_nan")))
    assert flt.output_names_ == ("with_nan", "const", "all_nan")
    out = flt.transform(X)
    assert out.shape == (n, 3)
    assert out.dtype == np.float32


def test_and_gate_does_not_drop_high_drift_with_low_autocorr() -> None:
    """A noisy regime-mean shift with low lag-1 autocorrelation must survive the AND gate."""
    rng = np.random.default_rng(12)
    n = 2000
    shift = np.concatenate([rng.normal(-5.0, 1.0, n // 2), rng.normal(5.0, 1.0, n // 2)])
    # Interleave high-frequency noise so lag-1 autocorr stays well below 0.995
    noisy_shift = shift + rng.normal(0.0, 3.0, n)
    X = noisy_shift.reshape(-1, 1).astype(np.float32)
    y = rng.integers(0, 2, size=n, dtype=np.int32)

    flt = StationarityFilter(_cfg()).fit(X, y, _ctx(("noisy_shift",)))
    rho1, drift = flt.metrics_["noisy_shift"]
    assert rho1 < 0.995
    assert drift > 1.5
    assert flt.output_names_ == ("noisy_shift",)


def test_transform_before_fit_raises() -> None:
    flt = StationarityFilter(_cfg())
    with pytest.raises(FeatureSelectionError):
        flt.transform(np.zeros((10, 2), dtype=np.float32))


def test_unsupervised_raises() -> None:
    flt = StationarityFilter(_cfg())
    ctx = FitContext(feature_names=("a",), labels_aux={}, task_type=TaskType.UNSUPERVISED)
    with pytest.raises(InvalidDataError):
        flt.fit(np.zeros((20, 1), dtype=np.float32), np.zeros(20, dtype=np.int32), ctx)
