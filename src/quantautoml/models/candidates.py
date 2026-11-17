"""Modern GBDT Trio model wrappers (LightGBM, CatBoost, XGBoost) with unified contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.exceptions import InvalidDataError, ModelError
from quantautoml.common.types import TaskType


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

    @abstractmethod
    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32 | np.int32]) -> Self:
        """Fit model on training segment. Both X and y must be numpy arrays."""

    @abstractmethod
    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities of shape (n_samples, n_classes) as float32."""

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict continuous scores suitable for Spearman Rank IC evaluation.

        For binary classification, returns P(y=1).
        For multiclass classification, returns expected class index sum(c * P(y=c)).
        Always returns 1D float32 array of length n_samples.
        """
        probs = self.predict_proba(X)
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return probs[:, 1].copy()
        classes = np.arange(probs.shape[1], dtype=np.float32)
        return (probs @ classes).astype(np.float32)

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

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            default_kwargs["objective"] = "binary"
            clf = lgb.LGBMClassifier(**default_kwargs)
        else:
            default_kwargs["objective"] = "multiclass"
            clf = lgb.LGBMClassifier(**default_kwargs)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
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
        n_leaves = self.params.get("num_leaves", 31)
        est_part = min(1.0, max(0.01, n_est / 500.0))
        leaves_part = min(1.0, max(0.01, n_leaves / 64.0))
        return float(0.5 * est_part + 0.5 * leaves_part)


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
        y_arr = np.asarray(y, dtype=np.int32)
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
        est_part = min(1.0, max(0.01, n_est / 500.0))
        depth_part = min(1.0, max(0.01, depth / 10.0))
        return float(0.5 * est_part + 0.5 * depth_part)


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

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            default_kwargs["loss_function"] = "Logloss"
            clf = CatBoostClassifier(**default_kwargs)
        else:
            default_kwargs["loss_function"] = "MultiClass"
            clf = CatBoostClassifier(**default_kwargs)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
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
        est_part = min(1.0, max(0.01, n_est / 500.0))
        depth_part = min(1.0, max(0.01, depth / 10.0))
        return float(0.5 * est_part + 0.5 * depth_part)


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
