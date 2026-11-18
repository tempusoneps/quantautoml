"""ONNXBundleExporter: write the model bundle and gate it on the RULE-06 precision contract."""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from quantautoml.common.config import PackagingConfig
from quantautoml.common.exceptions import PackagingError
from quantautoml.common.types import PreprocessorSpec, TaskType
from quantautoml.ensemble import CalibrationResult, EnsembleWeightsResult, TemperatureCalibrator
from quantautoml.models.tuner import TunedCandidate
from quantautoml.packaging.converters import export_candidate_onnx
from quantautoml.packaging.predictor_loader import load_predictor
from quantautoml.packaging.preprocessor import build_preprocessor_onnx, run_preprocessor_onnx
from quantautoml.packaging.schema import FeatureSchemaContract
from quantautoml.packaging.validator import validate_numerical_precision

_TEMPLATE = Path(__file__).parent / "templates" / "predictor_template.py"


class BundleManifest(BaseModel):
    """Written to ``bundle_manifest.json``."""

    model_config = ConfigDict(frozen=True)

    files: list[str]
    max_abs_diff: float
    n_models: int
    task_type: str


class ONNXBundleExporter:
    """Export only: never fits anything. Train statistics come from the Train matrix passed in."""

    def __init__(self, cfg: PackagingConfig) -> None:
        self.cfg = cfg

    def export(
        self,
        *,
        out_dir: Path,
        preprocessor_spec: PreprocessorSpec,
        candidates: Sequence[TunedCandidate],
        ensemble: EnsembleWeightsResult,
        calibration: CalibrationResult,
        task_type: TaskType,
        input_features: Sequence[str],
        classes: Sequence[int] | None,
        X_train_raw: NDArray[np.float32],
        X_val_raw: NDArray[np.float32],
        X_val_model: NDArray[np.float32],
    ) -> BundleManifest:
        if len(candidates) != len(ensemble.weights):
            raise PackagingError(
                f"{len(candidates)} candidates but {len(ensemble.weights)} ensemble weights"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        try:
            return self._export(
                out_dir, written, preprocessor_spec, candidates, ensemble, calibration,
                task_type, input_features, classes, X_train_raw, X_val_raw, X_val_model,
            )
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            if not any(out_dir.iterdir()):
                out_dir.rmdir()
            raise

    def _export(
        self,
        out_dir: Path,
        written: list[Path],
        spec: PreprocessorSpec,
        candidates: Sequence[TunedCandidate],
        ensemble: EnsembleWeightsResult,
        calibration: CalibrationResult,
        task_type: TaskType,
        input_features: Sequence[str],
        classes: Sequence[int] | None,
        X_train_raw: NDArray[np.float32],
        X_val_raw: NDArray[np.float32],
        X_val_model: NDArray[np.float32],
    ) -> BundleManifest:
        def put(name: str) -> Path:
            path = out_dir / name
            written.append(path)
            return path

        pre_path = put("preprocessor.onnx")
        build_preprocessor_onnx(spec, pre_path)
        x_train_model = run_preprocessor_onnx(pre_path, X_train_raw)
        schema = FeatureSchemaContract.from_train(
            input_features, spec.output_names, x_train_model, task_type, classes,
            self.cfg.std_floor,
        )
        schema.check_input(X_val_raw)
        put("features_schema.json").write_text(schema.to_json(), encoding="utf-8")

        model_files: list[str] = []
        for k, cand in enumerate(candidates):
            name = f"model_{k}.onnx"
            export_candidate_onnx(cand.fitted_model, len(spec.output_names), put(name))
            model_files.append(name)
        meta = {
            "task_type": task_type.value,
            "classes": None if classes is None else [int(c) for c in classes],
            "models": model_files,
            "weights": [float(w) for w in ensemble.weights],
            "temperature": float(calibration.temperature),
            "drift_z_threshold": self.cfg.drift_z_threshold,
        }
        put("ensemble_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        shutil.copyfile(_TEMPLATE, put("predictor.py"))

        reference = _python_reference(candidates, ensemble, calibration, task_type, X_val_model)
        predictor = load_predictor(out_dir / "predictor.py").Predictor(out_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # drift on Val is irrelevant to the precision check
            got = predictor.predict_proba(X_val_raw)
        max_diff = validate_numerical_precision(reference, got, tol=self.cfg.precision_tol)

        manifest = BundleManifest(
            files=[p.name for p in written], max_abs_diff=max_diff,
            n_models=len(candidates), task_type=task_type.value,
        )
        put("bundle_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest


def _python_reference(
    candidates: Sequence[TunedCandidate],
    ensemble: EnsembleWeightsResult,
    calibration: CalibrationResult,
    task_type: TaskType,
    X_model: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Training-side pipeline (models -> weights -> temperature), as ``run_pipeline`` does."""
    weights = np.asarray(ensemble.weights, dtype=np.float32)
    calibrator = TemperatureCalibrator(task_type=task_type)
    calibrator.temperature_ = calibration.temperature
    probs = [c.fitted_model.predict_proba(X_model) for c in candidates]
    if task_type == TaskType.BINARY_CLASSIFICATION:
        score = sum(w * p[:, 1] for w, p in zip(weights, probs))
        p1 = calibrator.calibrate(np.asarray(score, dtype=np.float32))
        return np.column_stack([1.0 - p1, p1]).astype(np.float32)
    combined = sum(w * p for w, p in zip(weights, probs))
    return calibrator.calibrate(np.asarray(combined, dtype=np.float32))
