"""Infer the learning task (binary / multiclass / unsupervised) from the label vector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import TaskType

MAX_CLASSES = 10


@dataclass(frozen=True)
class TaskInference:
    task_type: TaskType
    classes: NDArray[np.generic] | None
    y: NDArray[np.int32] | None  # labels encoded as contiguous class codes 0..K-1


UNLABELED = -1  # code assigned to rows whose label is missing mid-series


def infer_task_type(y: NDArray[np.generic] | None) -> TaskInference:
    """Infer the task. Missing labels (NaN / None) are unlabeled rows: they are
    ignored when counting classes and encoded as ``UNLABELED`` (-1) in ``y``."""
    if y is None:
        return TaskInference(TaskType.UNSUPERVISED, None, None)

    values = np.asarray(y)
    if values.size == 0:
        raise InvalidDataError("label vector is empty")

    is_text = values.dtype.kind in "USO"
    if is_text:
        missing = np.array([v is None for v in values.tolist()], dtype=bool)
        present = values[~missing].astype(str)
    else:
        as_float = values.astype(np.float64)
        missing = np.isnan(as_float)
        present = values[~missing]
        if np.isinf(present.astype(np.float64)).any():
            raise InvalidDataError("labels contain Inf")

    classes = np.unique(present)
    n_classes = classes.size
    if n_classes < 2:
        raise InvalidDataError("labels contain a single class; cannot infer a task")
    if n_classes > MAX_CLASSES:
        raise InvalidDataError(
            f"{n_classes} unique labels (> {MAX_CLASSES}): regression labels are not supported"
        )
    if not is_text and n_classes > 2 and not np.all(classes == np.round(classes)):
        raise InvalidDataError("multiclass labels must be integer valued")

    codes = np.full(values.shape, UNLABELED, dtype=np.int32)
    codes[~missing] = np.searchsorted(classes, present).astype(np.int32)
    task = (
        TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
    )
    return TaskInference(task, classes, codes)
