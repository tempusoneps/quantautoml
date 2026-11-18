"""Unit tests for FeatureSchemaContract (features_schema.json)."""

import numpy as np
import pytest

from quantautoml.common.exceptions import SchemaMismatchError
from quantautoml.common.types import TaskType
from quantautoml.packaging.schema import FeatureSchemaContract


def _make(X: np.ndarray, floor: float = 1e-8) -> FeatureSchemaContract:
    names = [f"f{i}" for i in range(X.shape[1])]
    return FeatureSchemaContract.from_train(
        input_features=names,
        model_features=names,
        X_model_train=X,
        task_type=TaskType.BINARY_CLASSIFICATION,
        classes=[0, 1],
        std_floor=floor,
    )


def test_from_train_uses_only_given_train_matrix() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(2.0, 3.0, size=(500, 4)).astype(np.float32)
    X[::7, 1] = np.nan
    c = _make(X)
    np.testing.assert_allclose(c.train_mean, np.nanmean(X.astype(np.float64), axis=0), rtol=1e-6)
    np.testing.assert_allclose(c.train_std, np.nanstd(X.astype(np.float64), axis=0), rtol=1e-6)


def test_zero_std_feature_floored() -> None:
    X = np.ones((50, 2), dtype=np.float32)
    X[:, 1] = np.arange(50, dtype=np.float32)
    c = _make(X, floor=1e-8)
    assert c.train_std[0] == 1e-8
    assert c.train_std[1] > 1.0


def test_check_input_accepts_valid() -> None:
    X = np.zeros((3, 4), dtype=np.float32)
    _make(np.random.default_rng(1).normal(size=(20, 4)).astype(np.float32)).check_input(X)


def test_check_input_wrong_width_raises() -> None:
    c = _make(np.random.default_rng(1).normal(size=(20, 4)).astype(np.float32))
    with pytest.raises(SchemaMismatchError):
        c.check_input(np.zeros((3, 5), dtype=np.float32))


def test_check_input_wrong_names_order_raises() -> None:
    c = _make(np.random.default_rng(1).normal(size=(20, 4)).astype(np.float32))
    with pytest.raises(SchemaMismatchError):
        c.check_input(np.zeros((3, 4), dtype=np.float32), names=["f1", "f0", "f2", "f3"])


def test_check_input_wrong_dtype_raises() -> None:
    c = _make(np.random.default_rng(1).normal(size=(20, 4)).astype(np.float32))
    with pytest.raises(SchemaMismatchError):
        c.check_input(np.zeros((3, 4), dtype=np.float64))


def test_json_roundtrip_equal() -> None:
    c = _make(np.random.default_rng(2).normal(size=(30, 3)).astype(np.float32))
    assert FeatureSchemaContract.from_json(c.to_json()) == c
