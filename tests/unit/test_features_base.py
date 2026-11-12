from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
import pytest
from numpy.typing import NDArray

from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import FilterReport, TaskType
from quantautoml.features.base import BaseFeatureFilter, FitContext


class KeepFirstFilter(BaseFeatureFilter):
    name: ClassVar[str] = "keep_first"

    def __init__(self, out_dtype: type = np.float32) -> None:
        super().__init__()
        self.out_dtype = out_dtype

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        self.output_names_ = (ctx.feature_names[0],)
        self.report_ = FilterReport(
            stage=self.name,
            n_in=X.shape[1],
            n_out=1,
            dropped={n: "not first" for n in ctx.feature_names[1:]},
        )

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return X[:, :1].astype(self.out_dtype)


def _ctx(n: int = 3, task: TaskType = TaskType.BINARY_CLASSIFICATION) -> FitContext:
    return FitContext(
        feature_names=tuple(f"f{i}" for i in range(n)), labels_aux={}, task_type=task
    )


def _data(n_rows: int = 20, n_cols: int = 3) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    rng = np.random.default_rng(0)
    return (
        rng.normal(size=(n_rows, n_cols)).astype(np.float32),
        rng.integers(0, 2, n_rows).astype(np.int32),
    )


def test_transform_before_fit_raises() -> None:
    X, _ = _data()
    with pytest.raises(FeatureSelectionError):
        KeepFirstFilter().transform(X)


def test_fit_returns_self_and_sets_state() -> None:
    X, y = _data()
    f = KeepFirstFilter()
    assert f.fit(X, y, _ctx()) is f
    assert f.is_fitted_
    assert f.output_names_ == ("f0",)
    assert f.report_.n_in == 3 and f.report_.n_out == 1


def test_transform_wrong_column_count_raises() -> None:
    X, y = _data()
    f = KeepFirstFilter().fit(X, y, _ctx())
    with pytest.raises(FeatureSelectionError):
        f.transform(X[:, :2])


def test_transform_does_not_mutate_fitted_state() -> None:
    X, y = _data()
    f = KeepFirstFilter().fit(X, y, _ctx())
    names, report = f.output_names_, f.report_
    X2, _ = _data(10, 3)
    f.transform(X2 * 100)
    f.transform(X)
    assert f.output_names_ == names
    assert f.report_ is report


def test_fit_with_none_y_raises() -> None:
    X, _ = _data()
    with pytest.raises(InvalidDataError):
        KeepFirstFilter().fit(X, None, _ctx())  # type: ignore[arg-type]


def test_fit_unsupervised_raises() -> None:
    X, y = _data()
    with pytest.raises(InvalidDataError):
        KeepFirstFilter().fit(X, y, _ctx(task=TaskType.UNSUPERVISED))


def test_fit_rejects_non_float32() -> None:
    X, y = _data()
    with pytest.raises(InvalidDataError):
        KeepFirstFilter().fit(X.astype(np.float64), y, _ctx())


def test_fit_rejects_non_2d() -> None:
    _, y = _data()
    with pytest.raises(InvalidDataError):
        KeepFirstFilter().fit(np.zeros(20, dtype=np.float32), y, _ctx())


def test_fit_rejects_name_count_mismatch() -> None:
    X, y = _data()
    with pytest.raises(InvalidDataError):
        KeepFirstFilter().fit(X, y, _ctx(n=2))


def test_fit_rejects_length_mismatch() -> None:
    X, y = _data()
    with pytest.raises(InvalidDataError):
        KeepFirstFilter().fit(X, y[:-1], _ctx())


def test_transform_output_is_float32() -> None:
    X, y = _data()
    out = KeepFirstFilter().fit(X, y, _ctx()).transform(X)
    assert out.dtype == np.float32
    assert out.shape == (20, 1)


def test_transform_float64_output_raises() -> None:
    X, y = _data()
    f = KeepFirstFilter(out_dtype=np.float64).fit(X, y, _ctx())
    with pytest.raises(FeatureSelectionError):
        f.transform(X)


def test_filter_report_is_frozen() -> None:
    r = FilterReport(stage="s", n_in=1, n_out=1, dropped={})
    assert r.notes == ()
    with pytest.raises(AttributeError):
        r.n_in = 2  # type: ignore[misc]


class WrongWidthFilter(KeepFirstFilter):
    """Declares one output name but `_transform` returns two columns."""

    name: ClassVar[str] = "wrong_width"

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return X[:, :2].astype(np.float32)


def test_transform_output_width_must_match_output_names() -> None:
    X, y = _data()
    f = WrongWidthFilter().fit(X, y, _ctx())
    with pytest.raises(FeatureSelectionError, match="output_names_"):
        f.transform(X)
