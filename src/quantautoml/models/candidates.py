"""Modern GBDT Trio model wrappers (LightGBM, CatBoost, XGBoost) with unified contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.exceptions import InvalidDataError, ModelError
from quantautoml.common.types import TaskType


def _clip01(value: float) -> float:
    return min(1.0, max(0.01, value))


def _complexity(n_estimators: float, structure: float, learning_rate: float) -> float:
    """Blend of ensemble size, tree structure and learning rate, each scaled to (0, 1]."""
    return float(
        0.4 * _clip01(n_estimators / 500.0)
        + 0.4 * _clip01(structure)
        + 0.2 * _clip01(learning_rate / 0.3)
    )


class BaseGBDCandidate(ABC):
    """Abstract base wrapper for GBDT candidate models."""

    name: str

    def __init__(
        self,
        task_type: TaskType,
        params: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        if task_type == TaskType.UNSUPERVISED:
            raise InvalidDataError(
                f"Candidate model '{self.name}' cannot be trained on UNSUPERVISED tasks"
            )
        self.task_type = task_type
        self.params = dict(params or {})
        self.random_state = random_state
        self.estimator_: Any = None
        self.classes_: NDArray[np.int32] | None = None

    @abstractmethod
    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32 | np.int32]) -> Self:
        """Fit model on training segment. Both X and y must be numpy arrays."""

    @abstractmethod
    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities of shape (n_samples, n_classes) as float32."""

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict continuous scores suitable for Spearman Rank IC evaluation.

        For binary classification, returns P(y=1).
        For multiclass classification, returns expected class value sum(c * P(y=c))
        over the classes seen in training.
        Always returns 1D float32 array of length n_samples.
        """
        probs = self.predict_proba(X)
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return probs[:, 1].copy()
        assert self.classes_ is not None
        return (probs @ self.classes_.astype(np.float32)).astype(np.float32)

    def _encode_labels(self, y: NDArray[np.float32 | np.int32]) -> NDArray[np.int32]:
        """Map labels to contiguous 0..k-1 (required by XGBoost) and remember the originals."""
        classes, encoded = np.unique(np.asarray(y, dtype=np.int32), return_inverse=True)
        if len(classes) < 2:
            raise ModelError(
                f"Candidate '{self.name}' needs at least 2 classes in the training segment, "
                f"got {len(classes)}"
            )
        self.classes_ = classes.astype(np.int32)
        return encoded.astype(np.int32)

    @abstractmethod
    def complexity_score(self) -> float:
        """Return complexity penalty score in (0, 1] based on tree depth and count."""

    def _ensure_fitted(self) -> None:
        if self.estimator_ is None:
            raise ModelError(f"Candidate '{self.name}' has not been fitted yet")


class LightGBMCandidate(BaseGBDCandidate):
    """LightGBM candidate wrapper."""

    name = "lightgbm"

    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32 | np.int32]) -> Self:
        import lightgbm as lgb

        n_estimators = self.params.get("n_estimators", 100)
        learning_rate = self.params.get("learning_rate", 0.05)
        num_leaves = self.params.get("num_leaves", 31)
        max_depth = self.params.get("max_depth", -1)

        default_kwargs = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        for k, v in self.params.items():
            default_kwargs[k] = v

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = self._encode_labels(y)
        assert self.classes_ is not None
        # LightGBM's multiclass objective needs >= 3 classes; a fold seeing only two is binary.
        is_binary = self.task_type == TaskType.BINARY_CLASSIFICATION or len(self.classes_) == 2
        default_kwargs["objective"] = "binary" if is_binary else "multiclass"
        clf = lgb.LGBMClassifier(**default_kwargs)
        clf.fit(X_arr, y_arr)
        self._quantize_booster_to_float32(clf)
        self.estimator_ = clf
        return self

    @staticmethod
    def _quantize_booster_to_float32(clf: Any) -> None:
        """Quantize LightGBM tree thresholds and leaf values to float32 in-place.

        LightGBM stores tree split thresholds in float64 by default while ONNX's
        ``TreeEnsembleClassifier`` represents ``nodes_values`` in float32. Quantizing
        the booster immediately after ``fit()`` guarantees that Python ``predict_proba``
        and ONNX Runtime take identical branch decisions on boundary samples (RULE-06).
        """
        import lightgbm as lgb

        raw_lines = clf.booster_.model_to_string().splitlines()
        out_lines: list[str] = []
        for line in raw_lines:
            if line.startswith("tree_sizes="):
                # Strip byte-length index so LightGBM parses the modified tree blocks cleanly
                continue
            if line.startswith("threshold="):
                vals: list[str] = []
                for v in line[len("threshold=") :].split():
                    fv = float(v)
                    # Preserve LightGBM's sentinel threshold (-1e-35 for zero_as_missing) intact
                    if abs(fv + 1e-35) < 1e-40:
                        vals.append(v)
                    else:
                        vals.append(repr(float(np.float32(fv))))
                out_lines.append("threshold=" + " ".join(vals))
            elif line.startswith("leaf_value="):
                vals = [
                    repr(float(np.float32(float(v))))
                    for v in line[len("leaf_value=") :].split()
                ]
                out_lines.append("leaf_value=" + " ".join(vals))
            else:
                out_lines.append(line)
        clf._Booster = lgb.Booster(model_str="\n".join(out_lines) + "\n")

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        probs = self.estimator_.predict_proba(X_arr)
        return np.asarray(probs, dtype=np.float32)

    def complexity_score(self) -> float:
        n_est = self.params.get("n_estimators", 100)
        n_leaves = self.params.get("num_leaves", 31)
        max_depth = self.params.get("max_depth", -1)
        if max_depth > 0:
            n_leaves = min(n_leaves, 2**max_depth)
        lr = self.params.get("learning_rate", 0.05)
        return _complexity(n_est, n_leaves / 64.0, lr)


class XGBoostCandidate(BaseGBDCandidate):
    """XGBoost candidate wrapper."""

    name = "xgboost"

    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32 | np.int32]) -> Self:
        import xgboost as xgb

        n_estimators = self.params.get("n_estimators", 100)
        learning_rate = self.params.get("learning_rate", 0.05)
        max_depth = self.params.get("max_depth", 6)

        default_kwargs = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": self.random_state,
            "verbosity": 0,
            "n_jobs": -1,
        }
        for k, v in self.params.items():
            default_kwargs[k] = v

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            default_kwargs["eval_metric"] = "logloss"
            clf = xgb.XGBClassifier(**default_kwargs)
        else:
            default_kwargs["eval_metric"] = "mlogloss"
            clf = xgb.XGBClassifier(**default_kwargs)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = self._encode_labels(y)
        clf.fit(X_arr, y_arr)
        self.estimator_ = clf
        return self

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        probs = self.estimator_.predict_proba(X_arr)
        return np.asarray(probs, dtype=np.float32)

    def complexity_score(self) -> float:
        n_est = self.params.get("n_estimators", 100)
        depth = self.params.get("max_depth", 6)
        lr = self.params.get("learning_rate", 0.05)
        return _complexity(n_est, depth / 10.0, lr)


class CatBoostCandidate(BaseGBDCandidate):
    """CatBoost candidate wrapper."""

    name = "catboost"

    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32 | np.int32]) -> Self:
        from catboost import CatBoostClassifier

        n_estimators = self.params.get("n_estimators", 100)
        learning_rate = self.params.get("learning_rate", 0.05)
        depth = self.params.get("depth", 6)

        default_kwargs = {
            "iterations": n_estimators,
            "learning_rate": learning_rate,
            "depth": depth,
            "random_seed": self.random_state,
            "verbose": 0,
            "thread_count": -1,
            "allow_writing_files": False,
        }
        for k, v in self.params.items():
            if k == "n_estimators":
                default_kwargs["iterations"] = v
            else:
                default_kwargs[k] = v
        if "subsample" in default_kwargs:
            # CatBoost's default bootstrap (Bayesian) rejects `subsample`; Bernoulli supports it.
            default_kwargs.setdefault("bootstrap_type", "Bernoulli")

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            default_kwargs["loss_function"] = "Logloss"
            clf = CatBoostClassifier(**default_kwargs)
        else:
            default_kwargs["loss_function"] = "MultiClass"
            clf = CatBoostClassifier(**default_kwargs)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = self._encode_labels(y)
        clf.fit(X_arr, y_arr)
        self.estimator_ = clf
        return self

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        probs = self.estimator_.predict_proba(X_arr)
        return np.asarray(probs, dtype=np.float32)

    def complexity_score(self) -> float:
        n_est = self.params.get("n_estimators", self.params.get("iterations", 100))
        depth = self.params.get("depth", 6)
        lr = self.params.get("learning_rate", 0.05)
        return _complexity(n_est, depth / 10.0, lr)


_CANDIDATE_REGISTRY: dict[str, type[BaseGBDCandidate]] = {
    "lightgbm": LightGBMCandidate,
    "xgboost": XGBoostCandidate,
    "catboost": CatBoostCandidate,
}


def create_candidate(
    name: str,
    task_type: TaskType,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> BaseGBDCandidate:
    """Factory creating a unified GBDT candidate wrapper."""
    cls = _CANDIDATE_REGISTRY.get(name)
    if cls is None:
        raise InvalidDataError(
            f"Unknown candidate model: '{name}'. Supported models: {sorted(_CANDIDATE_REGISTRY.keys())}"
        )
    return cls(task_type=task_type, params=params, random_state=random_state)
