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


def infer_task_type(y: NDArray[np.generic] | None) -> TaskInference:
    if y is None:
        return TaskInference(TaskType.UNSUPERVISED, None, None)

    values = np.asarray(y)
    if values.size == 0:
        raise InvalidDataError("label vector is empty")
    if not np.all(np.isfinite(values.astype(np.float64))):
        raise InvalidDataError("labels contain NaN or Inf")

    classes = np.unique(values)
    n_classes = classes.size
    if n_classes < 2:
        raise InvalidDataError("labels contain a single class; cannot infer a task")
    if n_classes > MAX_CLASSES:
        raise InvalidDataError(
            f"{n_classes} unique labels (> {MAX_CLASSES}): regression labels are not supported"
        )
    if n_classes > 2 and not np.all(classes == np.round(classes)):
        raise InvalidDataError("multiclass labels must be integer valued")

    codes = np.searchsorted(classes, values).astype(np.int32)
    task = (
        TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
    )
    return TaskInference(task, classes, codes)
