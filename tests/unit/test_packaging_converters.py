"""GBDT trio -> ONNX: probabilities match Python predict_proba within 1e-4 (float32)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from quantautoml.common.exceptions import PackagingError
from quantautoml.common.types import TaskType
from quantautoml.models.candidates import (
    BaseGBDCandidate,
    CatBoostCandidate,
    LightGBMCandidate,
    XGBoostCandidate,
)
from quantautoml.packaging.converters import export_candidate_onnx

MODELS = [LightGBMCandidate, XGBoostCandidate, CatBoostCandidate]
PARAMS = {"n_estimators": 40, "learning_rate": 0.1}


def _data(n_classes: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    score = X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=400)
    y = np.digitize(score, np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]))
    X[rng.random(X.shape) < 0.05] = np.nan  # preprocessor output may still hold NaN
    return X, y.astype(np.int32)


def _fit(cls: type[BaseGBDCandidate], n_classes: int) -> tuple[BaseGBDCandidate, np.ndarray]:
    X, y = _data(n_classes)
    task = TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
    return cls(task, dict(PARAMS)).fit(X, y), X


def _run(path: Path, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (out,) = sess.run(["probabilities"], {"X": X})
    return np.asarray(out)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.name)
def test_onnx_probs_close_to_python_predict_proba(cls, n_classes: int, tmp_path: Path) -> None:
    model, X = _fit(cls, n_classes)
    path = tmp_path / "m.onnx"
    export_candidate_onnx(model, X.shape[1], path)
    out = _run(path, X)
    py = model.predict_proba(X)
    assert out.shape == py.shape == (len(X), n_classes)
    assert out.dtype == np.float32
    assert np.max(np.abs(out - py)) < 1e-4


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.name)
def test_single_row_input(cls, tmp_path: Path) -> None:
    model, X = _fit(cls, 2)
    path = tmp_path / "m.onnx"
    export_candidate_onnx(model, X.shape[1], path)
    assert _run(path, X[:1]).shape == (1, 2)


def test_unknown_model_name_raises(tmp_path: Path) -> None:
    model, X = _fit(LightGBMCandidate, 2)
    model.name = "random_forest"  # type: ignore[misc]
    with pytest.raises(PackagingError):
        export_candidate_onnx(model, X.shape[1], tmp_path / "m.onnx")


def test_unfitted_model_raises(tmp_path: Path) -> None:
    from quantautoml.common.exceptions import ModelError

    m = LightGBMCandidate(TaskType.BINARY_CLASSIFICATION, dict(PARAMS))
    with pytest.raises((PackagingError, ModelError)):
        export_candidate_onnx(m, 8, tmp_path / "m.onnx")


def test_lightgbm_boundary_samples_have_float32_thresholds_matching_onnx(
    tmp_path: Path,
) -> None:
    """LightGBM stores split thresholds in float64 by default while ONNX casts them to float32.
    After LightGBMCandidate.fit(), tree thresholds must be quantized to float32 so samples
    sitting on a split boundary match between Python predict_proba and ONNX within 1e-4.
    """
    X, y = _data(3, seed=7)
    model = LightGBMCandidate(
        TaskType.MULTICLASS_CLASSIFICATION,
        {"n_estimators": 60, "learning_rate": 0.05, "num_leaves": 31},
    ).fit(X, y)

    # Probe exact float32 representations of every non-sentinel threshold in the fitted booster
    model_str = model.estimator_.booster_.model_to_string()
    thresholds: list[float] = []
    for line in model_str.splitlines():
        if line.startswith("threshold="):
            for tok in line[len("threshold=") :].split():
                val = float(tok)
                if abs(val + 1e-35) < 1e-40:
                    continue
                # Every non-sentinel threshold stored in the booster must equal its float32 cast
                assert val == float(np.float32(val)), (
                    f"Threshold {val!r} is not float32-quantized ({float(np.float32(val))!r})"
                )
                thresholds.append(val)
    assert thresholds

    # Also construct boundary probe rows where each feature takes its float32 threshold value
    rng = np.random.default_rng(99)
    probe = rng.choice(thresholds, size=(200, X.shape[1])).astype(np.float32)
    path = tmp_path / "lgb_boundary.onnx"
    export_candidate_onnx(model, X.shape[1], path)
    out = _run(path, probe)
    py = model.predict_proba(probe)
    assert np.max(np.abs(out - py)) < 1e-5

