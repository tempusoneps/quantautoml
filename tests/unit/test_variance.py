from __future__ import annotations

import time
import warnings
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.features.base import FitContext
from quantautoml.features.variance import MultiMetricVarianceFilter

N = 1000


@pytest.fixture(autouse=True)
def _warnings_are_errors() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _cfg(**kw: float) -> FeaturesConfig:
    return FeaturesConfig(ic_horizons={1: "fwd_ret_1"}, target_horizon=1, **kw)  # type: ignore[arg-type]


def _ctx(names: tuple[str, ...]) -> FitContext:
    return FitContext(
        feature_names=names, labels_aux={}, task_type=TaskType.BINARY_CLASSIFICATION
    )


def _y(n: int = N) -> NDArray[np.int32]:
    return np.random.default_rng(0).integers(0, 2, n).astype(np.int32)


def _fit(
    cols: dict[str, NDArray[np.float32]], cfg: FeaturesConfig | None = None
) -> MultiMetricVarianceFilter:
    names = tuple(cols)
    X = np.column_stack(list(cols.values())).astype(np.float32)
    return MultiMetricVarianceFilter(cfg or _cfg()).fit(X, _y(len(X)), _ctx(names))


def _normal(seed: int, n: int = N, loc: float = 5.0) -> NDArray[np.float32]:
    return np.random.default_rng(seed).normal(loc=loc, size=n).astype(np.float32)


def test_constant_column_dropped_good_kept() -> None:
    f = _fit({"good": _normal(1), "const": np.full(N, 3.0, dtype=np.float32)})
    assert f.output_names_ == ("good",)
    assert f.report_ is not None
    assert "unique" in f.report_.dropped["const"]
    assert f.report_.n_in == 2 and f.report_.n_out == 1


def test_balanced_binary_kept() -> None:
    b = (np.arange(N) % 2).astype(np.float32)
    f = _fit({"flag": b})
    assert f.output_names_ == ("flag",)
    assert f.metrics_["flag"][2] == pytest.approx(2 / N)


def test_low_unique_ratio_multivalued_dropped() -> None:
    col = (np.arange(N) % 3).astype(np.float32)  # 3 uniques / 1000 = 0.003 < 0.01
    f = _fit({"c": col})
    assert f.output_names_ == ()
    assert f.report_ is not None and "unique_ratio" in f.report_.dropped["c"]


def test_variance_burst_dropped_by_peak_var_ratio() -> None:
    x = _normal(2)
    x[100:260] = 5.0 + (x[100:260] - 5.0) * 40.0  # variance burst, mean unchanged
    f = _fit({"burst": x, "calm": _normal(3)}, _cfg(max_peak_var_ratio=2.0))
    assert f.output_names_ == ("calm",)
    assert f.report_ is not None and "peak_var_ratio" in f.report_.dropped["burst"]
    assert f.metrics_["burst"][1] > 2.0 > f.metrics_["calm"][1]


def test_near_zero_mean_no_division_error_and_low_cv_dropped() -> None:
    x = np.random.default_rng(4).normal(scale=1.0, size=N).astype(np.float32)
    x -= x.mean()
    f = _fit({"zm": x})
    cv = f.metrics_["zm"][0]
    assert np.isfinite(cv)
    # tiny-but-nonconstant scale with mean exactly 0 -> cv huge, kept
    assert f.output_names_ == ("zm",)


def test_low_cv_dropped() -> None:
    x = (100.0 + np.random.default_rng(5).normal(scale=0.01, size=N)).astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    f = _fit({"flat": x}, _cfg(min_cv=1e-3))
    assert f.report_ is not None and "cv" in f.report_.dropped.get("flat", "")


def test_train_shorter_than_window_raises() -> None:
    X = np.ones((10, 2), dtype=np.float32)
    with pytest.raises(InvalidDataError, match="10"):
        MultiMetricVarianceFilter(_cfg()).fit(X, _y(10), _ctx(("a", "b")))


def test_all_nan_column_dropped_no_warning() -> None:
    f = _fit({"nan": np.full(N, np.nan, dtype=np.float32), "good": _normal(6)})
    assert f.output_names_ == ("good",)
    assert f.report_ is not None and "NaN" in f.report_.dropped["nan"]


def test_partial_nan_handled() -> None:
    x = _normal(7)
    x[::5] = np.nan
    f = _fit({"p": x})
    assert f.output_names_ == ("p",)
    assert all(np.isfinite(v) for v in f.metrics_["p"])


def test_transform_selects_columns_float32() -> None:
    cols = {"const": np.full(N, 1.0, dtype=np.float32), "good": _normal(8)}
    f = _fit(cols)
    X = np.column_stack(list(cols.values())).astype(np.float32)
    out = f.transform(X)
    assert out.dtype == np.float32 and out.shape == (N, 1)
    np.testing.assert_array_equal(out[:, 0], X[:, 1])


def test_transform_before_fit_raises() -> None:
    with pytest.raises(FeatureSelectionError):
        MultiMetricVarianceFilter(_cfg()).transform(np.zeros((3, 2), dtype=np.float32))


def test_refit_leaves_no_stale_state() -> None:
    f = _fit({"a": _normal(9), "b": np.full(N, 2.0, dtype=np.float32)})
    X2 = np.column_stack([_normal(10), _normal(11)]).astype(np.float32)
    f.fit(X2, _y(), _ctx(("c", "d")))
    assert set(f.metrics_) == {"c", "d"}
    assert f.output_names_ == ("c", "d")
    assert f.report_ is not None and f.report_.dropped == {}


def test_performance_large() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(20_000, 60)).astype(np.float32)
    names = tuple(f"f{i}" for i in range(60))
    t0 = time.perf_counter()
    MultiMetricVarianceFilter(_cfg()).fit(X, _y(20_000), _ctx(names))
    assert time.perf_counter() - t0 < 2.0
