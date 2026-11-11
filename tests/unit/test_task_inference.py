import numpy as np
import pytest

from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.data.task_inference import infer_task_type


def test_none_labels_are_unsupervised() -> None:
    res = infer_task_type(None)
    assert res.task_type is TaskType.UNSUPERVISED
    assert res.classes is None
    assert res.y is None


def test_binary_zero_one_float_labels() -> None:
    res = infer_task_type(np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))
    assert res.task_type is TaskType.BINARY_CLASSIFICATION
    assert res.y is not None and res.y.dtype == np.int32
    assert res.y.tolist() == [0, 1, 1, 0]
    assert res.classes is not None and res.classes.tolist() == [0.0, 1.0]


def test_binary_minus_one_one_is_remapped_to_zero_one() -> None:
    res = infer_task_type(np.array([-1, 1, 1, -1]))
    assert res.task_type is TaskType.BINARY_CLASSIFICATION
    assert res.y is not None and res.y.tolist() == [0, 1, 1, 0]
    assert res.classes is not None and res.classes.tolist() == [-1, 1]


@pytest.mark.parametrize("k", [3, 5, 10])
def test_multiclass_between_three_and_ten_integer_classes(k: int) -> None:
    y = np.arange(k * 4) % k
    res = infer_task_type(y.astype(np.float32))
    assert res.task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert res.y is not None and res.y.dtype == np.int32
    assert set(res.y.tolist()) == set(range(k))


def test_multiclass_labels_are_encoded_contiguously() -> None:
    res = infer_task_type(np.array([10, 20, 30, 20, 10]))
    assert res.y is not None and res.y.tolist() == [0, 1, 2, 1, 0]
    assert res.classes is not None and res.classes.tolist() == [10, 20, 30]


def test_single_class_is_rejected() -> None:
    with pytest.raises(InvalidDataError):
        infer_task_type(np.ones(10))


def test_more_than_ten_classes_is_rejected_as_regression() -> None:
    with pytest.raises(InvalidDataError, match="regression"):
        infer_task_type(np.arange(11, dtype=np.float32))


def test_non_integer_multiclass_values_are_rejected() -> None:
    with pytest.raises(InvalidDataError):
        infer_task_type(np.array([0.1, 0.2, 0.3, 0.4]))


def test_inf_labels_are_rejected() -> None:
    with pytest.raises(InvalidDataError):
        infer_task_type(np.array([0.0, 1.0, np.inf]))


def test_nan_labels_are_unlabeled_and_encoded_as_minus_one() -> None:
    res = infer_task_type(np.array([0.0, np.nan, 1.0, 0.0]))
    assert res.task_type is TaskType.BINARY_CLASSIFICATION
    assert res.y is not None and res.y.tolist() == [0, -1, 1, 0]
    assert res.classes is not None and res.classes.tolist() == [0.0, 1.0]


def test_unlabeled_rows_do_not_count_as_a_class() -> None:
    with pytest.raises(InvalidDataError):
        infer_task_type(np.array([1.0, np.nan, np.nan]))


def test_string_labels_are_sorted_alphabetically() -> None:
    res = infer_task_type(np.array(["None", "Short", "Long", None, "None"], dtype=object))
    assert res.task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert res.classes is not None and res.classes.tolist() == ["Long", "None", "Short"]
    assert res.y is not None and res.y.tolist() == [1, 2, 0, -1, 1]


def test_binary_string_labels() -> None:
    res = infer_task_type(np.array(["b", "a", "b"], dtype=object))
    assert res.task_type is TaskType.BINARY_CLASSIFICATION
    assert res.y is not None and res.y.tolist() == [1, 0, 1]


def test_empty_labels_are_rejected() -> None:
    with pytest.raises(InvalidDataError):
        infer_task_type(np.array([], dtype=np.float32))
