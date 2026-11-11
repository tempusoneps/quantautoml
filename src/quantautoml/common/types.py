"""Core enums, data containers and report dataclasses for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    UNSUPERVISED = "unsupervised"


class SplitMode(str, Enum):
    STATIC_5WAY = "static_5way"
    OOF_ASSISTED = "oof_assisted"


@dataclass(frozen=True)
class IngestionReport:
    n_feature_rows: int
    n_label_rows: int
    n_joined_rows: int
    n_unmatched_dropped: int
    n_trailing_label_nan_dropped: int
    source_timezone: str
    naive_utc_assumed: bool
    n_features: int
    n_nan: int
    n_inf: int
    t_min: str
    t_max: str
    label_nan_counts: dict[str, int]
    max_trailing_nan: int
    horizon_h: int
    timestamp_precision_normalized: bool = False
    timestamp_original_time_unit: str = "us"
    bool_columns_cast: list[str] = field(default_factory=list)
    string_columns_encoded: dict[str, dict[str, int]] = field(default_factory=dict)
    mid_nan_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentSummary:
    name: str
    n_rows: int
    t_start: str
    t_end: str
    rows_dropped_before: int
    class_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class SplitReport:
    mode: SplitMode
    n_rows: int
    horizon_h: int
    segments: tuple[SegmentSummary, ...]
    oof_fold_sizes: tuple[tuple[int, int], ...] | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["segments"] = [dict(s) for s in data["segments"]]
        data["warnings"] = list(self.warnings)
        if self.oof_fold_sizes is not None:
            data["oof_fold_sizes"] = [list(p) for p in self.oof_fold_sizes]
        return data


@dataclass(frozen=True)
class Segment:
    X: NDArray[np.float32]
    y: NDArray[np.int32] | None
    labels_aux: dict[str, NDArray[np.float32]]
    timestamps: NDArray[np.datetime64]
    index: NDArray[np.int64]


@dataclass(frozen=True)
class Dataset:
    X: NDArray[np.float32]
    y: NDArray[np.int32] | None
    labels_aux: dict[str, NDArray[np.float32]]
    timestamps: NDArray[np.datetime64]
    feature_names: tuple[str, ...]
    selected_label: str | None
    report: IngestionReport
    task_type: TaskType | None = None
    classes_: NDArray[np.generic] | None = None


@dataclass(frozen=True)
class SplitResult:
    mode: SplitMode
    train: Segment
    val: Segment
    oos: Segment
    cal_a: Segment | None
    cal_b: Segment | None
    oof_folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]] | None
    report: SplitReport
