"""Standalone inference engine for a quantautoml model bundle (numpy + onnxruntime only)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort


class DistributionShiftWarning(UserWarning):
    """An input feature lies more than ``drift_z_threshold`` Train std-devs from its Train mean."""


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class Predictor:
    def __init__(self, bundle_dir: str | Path) -> None:
        d = Path(bundle_dir)
        meta = json.loads((d / "ensemble_meta.json").read_text())
        schema = json.loads((d / "features_schema.json").read_text())
        self._names = list(schema["input_features"])
        self._n_in = len(self._names)
        self._mean = np.asarray(schema["train_mean"], dtype=np.float32)
        self._std = np.asarray(schema["train_std"], dtype=np.float32)
        self._drift_cols = ~np.asarray(schema["is_binary"], dtype=bool)
        self._z = float(meta["drift_z_threshold"])
        self._temp = max(float(meta["temperature"]), 1e-4)
        self._binary = meta["task_type"] == "binary_classification"
        self._classes = np.asarray(meta["classes"], dtype=np.float32)
        self._weights = np.asarray(meta["weights"], dtype=np.float32)
        self._pre = _session(d / "preprocessor.onnx")
        self._models = [_session(d / name) for name in meta["models"]]

    def predict_proba(self, X: np.ndarray, names: list[str] | None = None) -> np.ndarray:
        """Calibrated probabilities ``[N, C]`` (binary: ``[1 - p, p]``), float32.

        Pass ``names`` (the caller's column names) to verify order against the schema.
        """
        if names is not None and list(names) != self._names:
            raise ValueError("input column names/order differ from features_schema.json")
        if X.ndim != 2 or X.shape[1] != self._n_in:
            raise ValueError(f"expected 2-D input with {self._n_in} columns, got shape {X.shape}")
        if X.dtype != np.float32:
            raise ValueError(f"expected float32 input, got {X.dtype}")
        xp = self._pre.run(None, {"X": np.ascontiguousarray(X)})[0]
        self._check_drift(xp)
        probs = [m.run(["probabilities"], {"X": xp})[0] for m in self._models]
        if self._binary:
            p = sum(w * q[:, 1] for w, q in zip(self._weights, probs))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            p1 = 1.0 / (1.0 + np.exp(-np.log(p / (1.0 - p)) / self._temp))
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)
        p = np.clip(sum(w * q for w, q in zip(self._weights, probs)), 1e-7, 1.0)
        z = np.log(p) / self._temp
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict(self, X: np.ndarray, names: list[str] | None = None) -> np.ndarray:
        """Ranking score ``[N]``: P(y=1) for binary, expected class value for multiclass."""
        proba = self.predict_proba(X, names)
        return proba[:, 1] if self._binary else (proba @ self._classes).astype(np.float32)

    def _check_drift(self, xp: np.ndarray) -> None:
        z = np.abs((xp - self._mean) / self._std)[:, self._drift_cols]
        if np.any(z > self._z):
            warnings.warn(
                f"{int((z > self._z).any(axis=0).sum())} feature(s) exceed |z| > {self._z}",
                DistributionShiftWarning,
                stacklevel=3,
            )
