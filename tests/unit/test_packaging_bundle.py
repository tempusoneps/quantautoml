"""ONNXBundleExporter: files, precision gate (RULE-06), train-only stats, no fit()."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from bundle_helper import SPEC, make_bundle
from test_preprocessor_spec import apply_spec_numpy

from quantautoml.common.config import PackagingConfig
from quantautoml.common.exceptions import PrecisionValidationError
from quantautoml.common.types import TaskType
from quantautoml.ensemble import CalibrationResult, EnsembleWeightsResult
from quantautoml.features.base import BaseFeatureFilter
from quantautoml.models.tuner import TunedCandidate
from quantautoml.packaging.bundle import BundleManifest, ONNXBundleExporter
from quantautoml.packaging.schema import FeatureSchemaContract

REQUIRED = {
    "preprocessor.onnx", "model_0.onnx", "model_1.onnx", "ensemble_meta.json",
    "features_schema.json", "predictor.py", "bundle_manifest.json",
}


def _ensemble(weights=(0.6, 0.4), accepted: bool = True) -> EnsembleWeightsResult:
    return EnsembleWeightsResult(
        weights=list(weights), raw_weights=list(weights), shrunk_weights=list(weights),
        bootstrap_std=[0.0, 0.0], is_ensemble_accepted=accepted, ensemble_ic_sharpe=1.0,
        best_single_ic_sharpe=1.0, equal_weights_ic_sharpe=1.0, hurdle_ratio_achieved=0.0,
        fallback_reason=None if accepted else "hurdle",
    )


def _calib(t: float) -> CalibrationResult:
    return CalibrationResult(
        temperature=t, uncalibrated_ece=0.1, calibrated_ece=0.02, rank_ic_preserved=True
    )


def _export(tmp_path: Path, fx, *, X_val_raw=None, weights=(0.6, 0.4), models=None, cfg=None):
    cands = [
        TunedCandidate(model_name=m.name, best_params={}, best_score=0.0, best_ic_sharpe=0.0,
                       complexity_score=0.1, fitted_model=m)
        for m in (models or fx.models)
    ]
    X_val = fx.X_raw[:200] if X_val_raw is None else X_val_raw
    exporter = ONNXBundleExporter(cfg or PackagingConfig())
    return exporter.export(
        out_dir=tmp_path / "out", preprocessor_spec=SPEC, candidates=cands,
        ensemble=_ensemble(weights), calibration=_calib(fx.temperature), task_type=fx.task,
        input_features=[f"x{i}" for i in range(6)], classes=fx.classes,
        X_train_raw=fx.X_raw[200:], X_val_raw=X_val, X_val_model=apply_spec_numpy(SPEC, X_val),
    )


@pytest.fixture
def fx(tmp_path: Path):
    return make_bundle(tmp_path / "ref")


def test_export_writes_required_files(tmp_path: Path, fx) -> None:
    manifest = _export(tmp_path, fx)
    assert isinstance(manifest, BundleManifest)
    assert {p.name for p in (tmp_path / "out").iterdir()} == REQUIRED
    assert set(manifest.files) == REQUIRED - {"bundle_manifest.json"}


def test_manifest_max_abs_diff_below_tol(tmp_path: Path, fx) -> None:
    manifest = _export(tmp_path, fx)
    assert 0.0 <= manifest.max_abs_diff < 1e-4 and manifest.n_models == 2
    on_disk = json.loads((tmp_path / "out" / "bundle_manifest.json").read_text())
    assert on_disk["max_abs_diff"] == manifest.max_abs_diff


def test_meta_carries_weights_temperature_and_task(tmp_path: Path, fx) -> None:
    _export(tmp_path, fx, weights=(0.5, 0.5))
    meta = json.loads((tmp_path / "out" / "ensemble_meta.json").read_text())
    assert meta["weights"] == [0.5, 0.5] and meta["temperature"] == fx.temperature
    assert meta["task_type"] == TaskType.BINARY_CLASSIFICATION.value
    assert meta["models"] == ["model_0.onnx", "model_1.onnx"]


def test_precision_failure_raises_and_leaves_no_bundle(tmp_path: Path, fx) -> None:
    bad = fx.models[0]
    orig = bad.predict_proba
    bad.predict_proba = lambda X: np.clip(orig(X) + 0.01, 0, 1).astype(np.float32)  # type: ignore[method-assign]
    with pytest.raises(PrecisionValidationError):
        _export(tmp_path, fx)
    out = tmp_path / "out"
    assert not (out / "predictor.py").exists()
    assert not out.exists() or not any(out.iterdir())


def test_exporter_never_calls_fit(tmp_path: Path, fx, monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("fit() must not be called during packaging (RULE-01)")

    from quantautoml.ensemble import TemperatureCalibrator
    from quantautoml.models.candidates import LightGBMCandidate, XGBoostCandidate

    for cls in (LightGBMCandidate, XGBoostCandidate, TemperatureCalibrator, BaseFeatureFilter):
        monkeypatch.setattr(cls, "fit", boom)
    _export(tmp_path, fx)


def test_schema_stats_use_train_only(tmp_path: Path, fx) -> None:
    _export(tmp_path, fx)
    a = FeatureSchemaContract.from_json((tmp_path / "out" / "features_schema.json").read_text())
    shifted = fx.X_raw[:200] * 3.0 + 7.0
    (tmp_path / "second").mkdir()
    _export(tmp_path / "second", fx, X_val_raw=shifted.astype(np.float32))
    b = FeatureSchemaContract.from_json(
        (tmp_path / "second" / "out" / "features_schema.json").read_text()
    )
    assert a.train_mean == b.train_mean and a.train_std == b.train_std


def test_multiclass_bundle_exports(tmp_path: Path) -> None:
    fx3 = make_bundle(tmp_path / "ref3", n_classes=3)
    manifest = _export(tmp_path, fx3)
    assert manifest.task_type == TaskType.MULTICLASS_CLASSIFICATION.value


def test_float64_val_input_rejected(tmp_path: Path, fx) -> None:
    from quantautoml.common.exceptions import PackagingError

    with pytest.raises(PackagingError):
        _export(tmp_path, fx, X_val_raw=fx.X_raw[:200].astype(np.float64))


def test_failed_reexport_leaves_previous_bundle_untouched(tmp_path: Path, fx) -> None:
    _export(tmp_path, fx)
    out = tmp_path / "out"
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    bad = fx.models[0]
    orig = bad.predict_proba
    bad.predict_proba = lambda X: np.clip(orig(X) + 0.01, 0, 1).astype(np.float32)  # type: ignore[method-assign]
    with pytest.raises(PrecisionValidationError):
        _export(tmp_path, fx)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert [p.name for p in tmp_path.iterdir()] == ["out", "ref"] or {
        p.name for p in tmp_path.iterdir()
    } == {"out", "ref"}  # no leftover temp build dirs


def test_failed_first_export_leaves_nothing_behind(tmp_path: Path, fx) -> None:
    bad = fx.models[0]
    orig = bad.predict_proba
    bad.predict_proba = lambda X: np.clip(orig(X) + 0.01, 0, 1).astype(np.float32)  # type: ignore[method-assign]
    with pytest.raises(PrecisionValidationError):
        _export(tmp_path, fx)
    assert {p.name for p in tmp_path.iterdir()} == {"ref"}
