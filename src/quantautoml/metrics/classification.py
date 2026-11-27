"""Label prediction (classification) metrics for binary and multiclass OOS evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata

from quantautoml.metrics.calibration import calc_ece


class ClassMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    precision: float
    recall: float
    f1: float
    support: int


class ClassificationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float
    roc_auc: float | None = None
    ece: float
    classes: list[str]
    per_class: list[ClassMetrics]
    confusion_matrix: list[list[int]]


def _format_label(val: int | float) -> str:
    fval = float(val)
    return str(int(round(fval))) if fval.is_integer() else f"{fval:g}"


def _binary_roc_auc(y_pos: NDArray[np.bool_], scores: NDArray[np.float64]) -> float | None:
    n_pos = int(y_pos.sum())
    n_neg = len(y_pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    u_stat = float(ranks[y_pos].sum()) - n_pos * (n_pos + 1) / 2.0
    return float(np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0))


def calc_classification_metrics(
    y_true: NDArray[np.int32 | np.float32],
    y_prob: NDArray[np.float32],
    classes: Sequence[int | float] | NDArray[np.float32 | np.int32] | None = None,
) -> ClassificationMetrics:
    """Compute label-prediction classification metrics from calibrated probabilities.

    Parameters
    ----------
    y_true : NDArray[np.int32 | np.float32]
        1D array of ground-truth class labels (e.g. ``{0, 1}`` or ``{-2, -1, 0, 1, 2}``).
    y_prob : NDArray[np.float32]
        2D array of shape ``(N, C)`` with predicted class probabilities.
    classes : Sequence[int | float] | NDArray | None
        Ordered class values corresponding to columns ``0..C-1`` of ``y_prob``.
        Defaults to ``[0, 1, ..., C - 1]`` when omitted.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(y_prob, dtype=np.float32)

    if y_t.size == 0 or probs.size == 0:
        raise ValueError("Empty input provided to calc_classification_metrics.")
    if probs.ndim != 2:
        raise ValueError(f"y_prob must be 2D (N, C), got ndim={probs.ndim}.")
    n_samples, n_classes = probs.shape
    if len(y_t) != n_samples:
        raise ValueError(f"Shape mismatch: len(y_true)={len(y_t)} != y_prob.shape[0]={n_samples}.")
    if n_classes < 2:
        raise ValueError(f"y_prob must have at least 2 classes, got {n_classes}.")
    if np.isnan(y_t).any() or np.isnan(probs).any() or np.isinf(probs).any():
        raise ValueError("Input contains NaN or Inf values.")

    if classes is None:
        cls_arr = np.arange(n_classes, dtype=np.float64)
    else:
        cls_arr = np.asarray(classes, dtype=np.float64)
        if len(cls_arr) != n_classes:
            raise ValueError(
                f"Length of classes ({len(cls_arr)}) does not match y_prob columns ({n_classes})."
            )

    # Map raw label values to column indices 0..C-1.
    y_idx = np.full(n_samples, -1, dtype=np.int32)
    for c_idx, c_val in enumerate(cls_arr):
        y_idx[np.isclose(y_t, c_val, atol=1e-6)] = c_idx
    if (y_idx < 0).any():
        bad = np.unique(y_t[y_idx < 0]).tolist()
        raise ValueError(f"Ground-truth label(s) {bad} not found in classes {cls_arr.tolist()}.")

    probs_clipped = np.clip(probs, 1e-7, 1.0)
    y_pred_idx = np.argmax(probs, axis=1).astype(np.int32)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_idx, y_pred_idx), 1)

    accuracy = float(np.trace(cm) / n_samples)
    per_class: list[ClassMetrics] = []
    recalls_present: list[float] = []
    f1s: list[float] = []
    weighted_f1_num = 0.0

    for c in range(n_classes):
        tp = int(cm[c, c])
        support = int(cm[c, :].sum())
        pred_count = int(cm[:, c].sum())
        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        per_class.append(
            ClassMetrics(
                label=_format_label(cls_arr[c]),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                support=support,
            )
        )
        f1s.append(f1)
        weighted_f1_num += f1 * support
        if support > 0:
            recalls_present.append(recall)

    balanced_accuracy = float(np.mean(recalls_present)) if recalls_present else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(weighted_f1_num / n_samples)

    true_probs = probs_clipped[np.arange(n_samples), y_idx].astype(np.float64)
    log_loss = float(-np.mean(np.log(true_probs)))

    if n_classes == 2:
        roc_auc = _binary_roc_auc(y_idx == 1, probs[:, 1].astype(np.float64))
    else:
        aucs = [
            auc
            for c in range(n_classes)
            if (auc := _binary_roc_auc(y_idx == c, probs[:, c].astype(np.float64))) is not None
        ]
        roc_auc = float(np.mean(aucs)) if aucs else None

    ece = calc_ece(y_idx, probs)

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        log_loss=log_loss,
        roc_auc=roc_auc,
        ece=ece,
        classes=[_format_label(c) for c in cls_arr],
        per_class=per_class,
        confusion_matrix=cm.tolist(),
    )
