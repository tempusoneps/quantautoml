"""Unit tests for OOS label classification metrics (binary and multiclass)."""

from __future__ import annotations

import numpy as np
import pytest

from quantautoml.metrics.classification import (
    ClassificationMetrics,
    calc_classification_metrics,
)


def test_perfect_binary_predictions() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_prob = np.array(
        [[0.95, 0.05], [0.90, 0.10], [0.10, 0.90], [0.05, 0.95]],
        dtype=np.float32,
    )
    m = calc_classification_metrics(y_true, y_prob, classes=[0, 1])
    assert isinstance(m, ClassificationMetrics)
    assert m.accuracy == pytest.approx(1.0)
    assert m.balanced_accuracy == pytest.approx(1.0)
    assert m.macro_f1 == pytest.approx(1.0)
    assert m.weighted_f1 == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.log_loss < 0.15
    assert 0.0 <= m.ece <= 0.15
    assert m.classes == ["0", "1"]
    assert m.confusion_matrix == [[2, 0], [0, 2]]
    assert len(m.per_class) == 2
    assert all(c.f1 == pytest.approx(1.0) and c.support == 2 for c in m.per_class)


def test_multiclass_with_negative_labels_and_imbalance() -> None:
    # Mimics tb5_label classes: [-2, -1, 0, 1, 2]
    classes = [-2.0, -1.0, 0.0, 1.0, 2.0]
    y_true = np.array([-2, -1, 0, 0, 1, 2], dtype=np.int32)
    # Predicts: -2 (correct), 0 (wrong), 0 (correct), 0 (correct), 1 (correct), 1 (wrong)
    y_prob = np.array(
        [
            [0.70, 0.10, 0.10, 0.05, 0.05],  # pred -2 (true -2)
            [0.10, 0.20, 0.50, 0.10, 0.10],  # pred 0  (true -1)
            [0.05, 0.10, 0.70, 0.10, 0.05],  # pred 0  (true 0)
            [0.05, 0.10, 0.65, 0.15, 0.05],  # pred 0  (true 0)
            [0.05, 0.05, 0.10, 0.70, 0.10],  # pred 1  (true 1)
            [0.05, 0.05, 0.10, 0.55, 0.25],  # pred 1  (true 2)
        ],
        dtype=np.float32,
    )
    m = calc_classification_metrics(y_true, y_prob, classes=classes)
    assert m.classes == ["-2", "-1", "0", "1", "2"]
    assert m.accuracy == pytest.approx(4 / 6)
    # Recalls: -2: 1/1, -1: 0/1, 0: 2/2, 1: 1/1, 2: 0/1 -> balanced_acc = 3/5 = 0.6
    assert m.balanced_accuracy == pytest.approx(0.6)
    assert m.confusion_matrix == [
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ]
    assert m.roc_auc is not None and 0.0 <= m.roc_auc <= 1.0
    assert m.log_loss > 0.0
    assert np.isfinite(m.macro_f1) and np.isfinite(m.weighted_f1)
    assert sum(c.support for c in m.per_class) == len(y_true)


def test_single_class_present_returns_none_auc() -> None:
    y_true = np.array([1, 1, 1], dtype=np.int32)
    y_prob = np.array([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]], dtype=np.float32)
    m = calc_classification_metrics(y_true, y_prob, classes=[0, 1])
    assert m.accuracy == pytest.approx(1.0)
    assert m.balanced_accuracy == pytest.approx(1.0)
    assert m.roc_auc is None


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="Empty"):
        calc_classification_metrics(np.array([], dtype=np.int32), np.empty((0, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="2D"):
        calc_classification_metrics(np.array([0, 1], dtype=np.int32), np.array([0.2, 0.8], dtype=np.float32))
    with pytest.raises(ValueError, match="not found in classes"):
        calc_classification_metrics(
            np.array([0, 99], dtype=np.int32),
            np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
            classes=[0, 1],
        )
