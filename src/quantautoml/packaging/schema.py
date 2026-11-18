"""features_schema.json data contract: input columns, model columns and Train statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from quantautoml.common.exceptions import SchemaMismatchError
from quantautoml.common.types import TaskType


class FeatureSchemaContract(BaseModel):
    """Contract between the training pipeline and the standalone predictor.

    ``train_mean`` / ``train_std`` describe the preprocessor *output* (``model_features``)
    on the Train segment only (RULE-01) and feed the drift check.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_features: list[str]
    model_features: list[str]
    dtype: Literal["float32"] = "float32"
    train_mean: list[float]
    train_std: list[float]
    task_type: str
    classes: list[int] | None

    @classmethod
    def from_train(
        cls,
        input_features: Sequence[str],
        model_features: Sequence[str],
        X_model_train: NDArray[np.float32],
        task_type: TaskType,
        classes: Sequence[int] | None,
        std_floor: float,
    ) -> Self:
        if X_model_train.shape[1] != len(model_features):
            raise SchemaMismatchError(
                f"{len(model_features)} model features for {X_model_train.shape[1]} columns"
            )
        x = X_model_train.astype(np.float64)
        if x.shape[0] == 0:
            mean = np.zeros(x.shape[1])
            std = np.zeros(x.shape[1])
        else:
            mean = np.nan_to_num(np.nanmean(x, axis=0))
            std = np.nan_to_num(np.nanstd(x, axis=0))
        std = np.maximum(std, std_floor)
        return cls(
            input_features=list(input_features),
            model_features=list(model_features),
            train_mean=[float(v) for v in mean],
            train_std=[float(v) for v in std],
            task_type=task_type.value,
            classes=None if classes is None else [int(c) for c in classes],
        )

    def check_input(self, X: NDArray[np.float32], names: Sequence[str] | None = None) -> None:
        if X.ndim != 2 or X.shape[1] != len(self.input_features):
            raise SchemaMismatchError(
                f"expected 2-D input with {len(self.input_features)} columns, got shape {X.shape}"
            )
        if X.dtype != np.float32:
            raise SchemaMismatchError(f"expected float32 input, got {X.dtype}")
        if names is not None and list(names) != self.input_features:
            raise SchemaMismatchError("input column names/order differ from features_schema.json")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, s: str) -> Self:
        return cls.model_validate_json(s)
