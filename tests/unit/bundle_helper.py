"""Test-only helper: build a tiny but real model bundle (same file formats as the exporter)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from test_preprocessor_spec import apply_spec_numpy

from quantautoml.common.types import PreprocessorSpec, TaskType
from quantautoml.ensemble import TemperatureCalibrator
from quantautoml.models.candidates import LightGBMCandidate, XGBoostCandidate
from quantautoml.packaging.converters import export_candidate_onnx
from quantautoml.packaging.preprocessor import build_preprocessor_onnx
from quantautoml.packaging.schema import FeatureSchemaContract

TEMPLATE = (
    Path(__file__).parents[2] / "src/quantautoml/packaging/templates/predictor_template.py"
)
SPEC = PreprocessorSpec(
    n_in=6, bool_fill=((4, 0.0),), indicator_idx=(2,), keep_idx=(0, 1, 2, 3, 4),
    final_idx=(0, 1, 2, 3, 4, 5),
    output_names=("f0", "f1", "f2", "f3", "f4", "is_nan_f2"),
)
INPUT_NAMES = ["x0", "x1", "x2", "x3", "x4", "Close"]
WEIGHTS = (0.6, 0.4)
TEMPERATURE = 1.7


@dataclass
class BundleFixture:
    dir: Path
    X_raw: np.ndarray
    y: np.ndarray
    task: TaskType
    classes: list[int]
    models: list
    weights: tuple[float, ...]
    temperature: float

    def reference(self, X_raw: np.ndarray) -> np.ndarray:
        """Python reference: spec -> models -> weights -> temperature. Returns calibrated probs."""
        Xp = apply_spec_numpy(SPEC, X_raw)
        cal = TemperatureCalibrator(task_type=self.task)
        cal.temperature_ = self.temperature
        if self.task == TaskType.BINARY_CLASSIFICATION:
            comb = sum(w * m.predict_proba(Xp)[:, 1] for w, m in zip(self.weights, self.models))
            return cal.calibrate(np.asarray(comb, dtype=np.float32))
        comb = sum(w * m.predict_proba(Xp) for w, m in zip(self.weights, self.models))
        return cal.calibrate(np.asarray(comb, dtype=np.float32))


def make_bundle(tmp_path: Path, n_classes: int = 2, seed: int = 0) -> BundleFixture:
    rng = np.random.default_rng(seed)
    n = 600
    X = rng.normal(size=(n, 6)).astype(np.float32)
    s = X[:, 0] + 0.5 * X[:, 1] + 0.5 * rng.normal(size=n)
    y = np.digitize(s, np.quantile(s, np.linspace(0, 1, n_classes + 1)[1:-1])).astype(np.int32)
    X[rng.random(X.shape) < 0.05] = np.nan
    X[rng.random(X.shape) < 0.01] = np.inf
    task = TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
    Xp = apply_spec_numpy(SPEC, X)
    params = {"n_estimators": 30, "learning_rate": 0.1}
    models = [LightGBMCandidate(task, params).fit(Xp, y), XGBoostCandidate(task, params).fit(Xp, y)]

    out = tmp_path / "bundle"
    out.mkdir(parents=True)
    build_preprocessor_onnx(SPEC, out / "preprocessor.onnx")
    for k, m in enumerate(models):
        export_candidate_onnx(m, Xp.shape[1], out / f"model_{k}.onnx")
    classes = list(range(n_classes))
    schema = FeatureSchemaContract.from_train(
        INPUT_NAMES, list(SPEC.output_names), Xp, task, classes, 1e-8
    )
    (out / "features_schema.json").write_text(schema.to_json())
    (out / "ensemble_meta.json").write_text(json.dumps({
        "task_type": task.value, "classes": classes,
        "models": [f"model_{k}.onnx" for k in range(len(models))],
        "weights": list(WEIGHTS), "temperature": TEMPERATURE, "drift_z_threshold": 3.0,
    }))
    shutil.copy(TEMPLATE, out / "predictor.py")
    return BundleFixture(out, X, y, task, classes, models, WEIGHTS, TEMPERATURE)
