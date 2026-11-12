"""Base contract for every feature filter (fit on Train only, transform anywhere)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import FilterReport, TaskType


@dataclass(frozen=True)
class FitContext:
    feature_names: tuple[str, ...]
    labels_aux: dict[str, NDArray[np.float32]]
    task_type: TaskType
    discrete_features: tuple[str, ...] = ()


class BaseFeatureFilter(ABC):
    """Template-method base: validation lives here, logic in `_fit`/`_transform`."""

    name: ClassVar[str]

    def __init__(self) -> None:
        self.is_fitted_: bool = False
        self.output_names_: tuple[str, ...] = ()
        self.report_: FilterReport | None = None
        self._n_features_in: int = 0

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> Self:
        if y is None or ctx.task_type == TaskType.UNSUPERVISED:
            raise InvalidDataError(f"{self.name}: feature funnel requires supervised labels")
        if X.ndim != 2:
            raise InvalidDataError(f"{self.name}: X must be 2-D, got ndim={X.ndim}")
        if X.dtype != np.float32:
            raise InvalidDataError(f"{self.name}: X must be float32, got {X.dtype}")
        if len(ctx.feature_names) != X.shape[1]:
            raise InvalidDataError(
                f"{self.name}: {len(ctx.feature_names)} feature names for {X.shape[1]} columns"
            )
        if len(y) != len(X):
            raise InvalidDataError(f"{self.name}: len(y)={len(y)} != len(X)={len(X)}")
        self._fit(X, y, ctx)
        self._n_features_in = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.is_fitted_:
            raise FeatureSelectionError(f"{self.name}: transform called before fit")
        if X.ndim != 2 or X.shape[1] != self._n_features_in:
            raise FeatureSelectionError(
                f"{self.name}: expected {self._n_features_in} columns, got shape {X.shape}"
            )
        out = self._transform(X)
        if out.dtype != np.float32:
            raise FeatureSelectionError(
                f"{self.name}: _transform returned {out.dtype}, float32 required"
            )
        return out

    @abstractmethod
    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        """Set `output_names_` and `report_`."""

    @abstractmethod
    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the fitted selection without mutating state."""
