from __future__ import annotations

import time
import tracemalloc
import warnings
from collections.abc import Iterator

import dcor
import numpy as np
import pytest
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.features import collinearity
from quantautoml.features.base import FitContext
from quantautoml.features.collinearity import FastNonlinearCollinearityFilter, fast_dcor

LABEL = "fwd_ret_1"
N = 2000


@pytest.fixture(autouse=True)
def _warnings_are_errors() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _cfg(**kw: float) -> FeaturesConfig:
    return FeaturesConfig(ic_horizons={1: LABEL, 5: "fwd_ret_5"}, target_horizon=1, **kw)  # type: ignore[arg-type]


def _y(n: int) -> NDArray[np.int32]:
    return np.random.default_rng(0).integers(0, 2, n).astype(np.int32)


def _fit(
    cols: dict[str, NDArray[np.float64]],
    label: NDArray[np.float64],
    cfg: FeaturesConfig | None = None,
) -> FastNonlinearCollinearityFilter:
    names = tuple(cols)
    X = np.column_stack(list(cols.values())).astype(np.float32)
    ctx = FitContext(
        feature_names=names,
        labels_aux={LABEL: label.astype(np.float32)},
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    return FastNonlinearCollinearityFilter(cfg or _cfg()).fit(X, _y(len(X)), ctx)


def _f32(a: NDArray[np.float64]) -> NDArray[np.float32]:
    return a.astype(np.float32)


# ---------------------------------------------------------------- fast_dcor


def test_fast_dcor_detects_nonlinear_dependence_pearson_misses() -> None:
    x = np.random.default_rng(1).normal(size=5000)
    assert abs(np.corrcoef(x, x**2)[0, 1]) < 0.05
    assert fast_dcor(_f32(x), _f32(x**2)) > 0.3


def test_fast_dcor_identity_is_one() -> None:
    x = _f32(np.random.default_rng(2).normal(size=5000))
    assert fast_dcor(x, x) == pytest.approx(1.0, abs=1e-6)


def test_fast_dcor_independent_noise_small() -> None:
    rng = np.random.default_rng(3)
    assert fast_dcor(_f32(rng.normal(size=5000)), _f32(rng.normal(size=5000))) < 0.1


def test_fast_dcor_constant_and_too_few_pairs_are_zero() -> None:
    x = _f32(np.random.default_rng(4).normal(size=100))
    assert fast_dcor(np.full(100, 3.0, dtype=np.float32), x) == 0.0
    assert fast_dcor(x, np.full(100, 3.0, dtype=np.float32)) == 0.0
    # dcor's AVL asserts n > 3, so fewer than 4 valid pairs -> 0.0 instead of a crash
    assert fast_dcor(x[:3], x[:3] ** 2) == 0.0
    assert fast_dcor(x[:4], x[:4] ** 2) > 0.0


def test_fast_dcor_drops_nonfinite_pairs() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=500)
    y = x**2 + 0.1 * rng.normal(size=500)
    xb, yb = _f32(x.copy()), _f32(y.copy())
    xb[10:20] = np.nan
    yb[30] = np.inf
    xb[40] = -np.inf
    keep = np.ones(500, dtype=bool)
    keep[[*range(10, 20), 30, 40]] = False
    expected = dcor.distance_correlation(
        xb[keep].astype(np.float64), yb[keep].astype(np.float64), method="AVL"
    )
    assert fast_dcor(xb, yb) == pytest.approx(expected, abs=1e-9)
    all_nan = np.full(500, np.nan, dtype=np.float32)
    assert fast_dcor(all_nan, yb) == 0.0


def test_fast_dcor_casts_float64_so_dcor_emits_no_warning() -> None:
    x = _f32(np.random.default_rng(6).normal(size=1000))
    # raw float32 really does fall back to the uncompiled path with a UserWarning ...
    with pytest.warns(UserWarning, match="uncompiled"):
        dcor.distance_correlation(x, x**2, method="AVL")
    # ... while fast_dcor (warnings are errors here) stays silent
    assert fast_dcor(x, x**2) > 0.3


def test_fast_dcor_shape_mismatch_raises() -> None:
    with pytest.raises(InvalidDataError, match="shape"):
        fast_dcor(np.zeros(10, dtype=np.float32), np.zeros(11, dtype=np.float32))


def test_fast_dcor_nonfinite_result_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(x: NDArray[np.float64], y: NDArray[np.float64], **kw: str) -> float:
        return float("nan")

    monkeypatch.setattr(collinearity.dcor, "distance_correlation", broken)
    x = _f32(np.random.default_rng(7).normal(size=50))
    with pytest.raises(FeatureSelectionError, match="non-finite"):
        fast_dcor(x, x**2)


# ---------------------------------------------------------------- filter


@pytest.mark.parametrize("label_is_square", [False, True])
def test_square_pair_keeps_higher_ic_column(label_is_square: bool) -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=N)
    noise = rng.normal(size=N)
    label = (x**2 if label_is_square else 0.5 * x) + rng.normal(size=N)
    # dCor(x, x^2) ~ 0.53 for symmetric x: lower the threshold so the pair is collinear
    f = _fit({"sq": x**2, "noise": noise, "x": x}, label, _cfg(dcor_threshold=0.5))
    winner, loser = ("sq", "x") if label_is_square else ("x", "sq")
    assert f.output_names_ == tuple(n for n in ("sq", "noise", "x") if n != loser)
    assert f.report_ is not None
    assert set(f.report_.dropped) == {loser}
    reason = f.report_.dropped[loser]
    assert f"'{winner}'" in reason and "dcor=" in reason
    assert abs(f.rank_ic_[winner]) > abs(f.rank_ic_[loser])
    assert f.report_.n_in == 3 and f.report_.n_out == len(f.output_names_) == 2


def test_default_threshold_drops_monotone_duplicate_tie_keeps_earlier() -> None:
    rng = np.random.default_rng(9)
    x = np.abs(rng.normal(size=N)) + 0.1  # x and x^2 share ranks -> identical |IC|
    label = x + rng.normal(size=N)
    f = _fit({"a": x, "b": x**2}, label)
    assert f.rank_ic_["a"] == f.rank_ic_["b"]
    assert f.output_names_ == ("a",)
    f2 = _fit({"b": x**2, "a": x}, label)
    assert f2.output_names_ == ("b",)
    assert f2.report_ is not None and "'b'" in f2.report_.dropped["a"]


def test_only_compared_against_kept_features(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, float]] = []
    real = collinearity.fast_dcor

    def spy(x: NDArray[np.float32], y: NDArray[np.float32]) -> float:
        calls.append((float(x[0]), float(y[0])))
        return real(x, y)

    monkeypatch.setattr(collinearity, "fast_dcor", spy)
    rng = np.random.default_rng(10)
    base = rng.normal(size=N)
    label = base + rng.normal(size=N)
    cols = {
        "s": base,
        "dup1": base + 0.01 * rng.normal(size=N),
        "dup2": base + 0.01 * rng.normal(size=N),
        "n1": rng.normal(size=N),
        "n2": rng.normal(size=N),
    }
    f = _fit(cols, label)
    # s/dup1/dup2 have near-identical |IC|: exactly one of them survives
    assert len(f.output_names_) == 3 and {"n1", "n2"} <= set(f.output_names_)
    # 1st kept: 0 pairs; dup1, dup2 each stop after the 1 kept; then n1 (1), n2 (2)
    assert len(calls) == 5


def test_constant_column_is_never_collinear() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=N)
    f = _fit({"c1": np.full(N, 1.0), "x": x, "c2": np.full(N, 2.0)}, x + rng.normal(size=N))
    assert f.output_names_ == ("c1", "x", "c2")
    assert f.rank_ic_["c1"] == 0.0


def test_missing_target_label_raises_naming_it() -> None:
    X = np.zeros((10, 1), dtype=np.float32)
    ctx = FitContext(feature_names=("a",), labels_aux={}, task_type=TaskType.BINARY_CLASSIFICATION)
    with pytest.raises(InvalidDataError, match=LABEL):
        FastNonlinearCollinearityFilter(_cfg()).fit(X, _y(10), ctx)


def test_label_shape_mismatch_raises() -> None:
    X = np.zeros((10, 1), dtype=np.float32)
    ctx = FitContext(
        feature_names=("a",),
        labels_aux={LABEL: np.zeros(9, dtype=np.float32)},
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    with pytest.raises(InvalidDataError, match=LABEL):
        FastNonlinearCollinearityFilter(_cfg()).fit(X, _y(10), ctx)


def test_deterministic_across_runs() -> None:
    rng = np.random.default_rng(12)
    base = rng.normal(size=(N, 4))
    cols: dict[str, NDArray[np.float64]] = {}
    for j in range(4):
        cols[f"b{j}"] = base[:, j]
        cols[f"d{j}"] = base[:, j] + 0.05 * rng.normal(size=N)
    label = base.sum(axis=1) + rng.normal(size=N)
    f1, f2 = _fit(cols, label), _fit(cols, label)
    assert f1.output_names_ == f2.output_names_
    assert f1.report_ is not None and f2.report_ is not None
    assert f1.report_.dropped == f2.report_.dropped
    assert len(f1.output_names_) == 4


def test_transform_selects_columns_in_original_order_without_reading_aux() -> None:
    class CountingDict(dict[str, NDArray[np.float32]]):
        reads = 0

        def __getitem__(self, key: str) -> NDArray[np.float32]:
            CountingDict.reads += 1
            return super().__getitem__(key)

    rng = np.random.default_rng(13)
    x = rng.normal(size=N)
    noise = rng.normal(size=N)
    X = np.column_stack([noise, x + 0.01 * rng.normal(size=N), x]).astype(np.float32)
    aux = CountingDict({LABEL: (x + rng.normal(size=N)).astype(np.float32)})
    ctx = FitContext(
        feature_names=("noise", "dup", "x"),
        labels_aux=aux,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    f = FastNonlinearCollinearityFilter(_cfg()).fit(X, _y(N), ctx)
    assert len(f.output_names_) == 2 and f.output_names_[0] == "noise"
    CountingDict.reads = 0
    aux.clear()
    Xv = rng.normal(size=(40, 3)).astype(np.float32)
    out = f.transform(Xv)
    assert CountingDict.reads == 0
    assert out.dtype == np.float32 and out.shape == (40, 2)
    kept_idx = [("noise", "dup", "x").index(n) for n in f.output_names_]
    np.testing.assert_array_equal(out, Xv[:, kept_idx])


def test_transform_before_fit_raises() -> None:
    with pytest.raises(FeatureSelectionError):
        FastNonlinearCollinearityFilter(_cfg()).transform(np.zeros((3, 2), dtype=np.float32))


def test_refit_leaves_no_stale_state() -> None:
    rng = np.random.default_rng(14)
    x = rng.normal(size=N)
    label = x + rng.normal(size=N)
    f = _fit({"a": x, "b": x + 0.01 * rng.normal(size=N)}, label)
    assert f.report_ is not None and f.report_.dropped
    X2 = np.column_stack([x]).astype(np.float32)
    ctx = FitContext(
        feature_names=("c",),
        labels_aux={LABEL: label.astype(np.float32)},
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    f.fit(X2, _y(N), ctx)
    assert set(f.rank_ic_) == {"c"}
    assert f.output_names_ == ("c",)
    assert f.report_ is not None and f.report_.dropped == {} and f.report_.n_in == 1
    assert f.transform(X2).shape == (N, 1)


def test_scalability_20k_rows_30_features() -> None:
    """N=20,000 x 30 features: < 60 s and tracemalloc peak < 300 MB.

    A single N x N float64 matrix would be 3.2 GB, so the memory bound proves none is
    built. Caveat: tracemalloc only sees Python-heap / numpy allocations; buffers that
    dcor's numba-compiled AVL kernels allocate internally are not traced (they are
    O(N) by construction of the AVL algorithm).
    """
    n, p = 20_000, 30
    rng = np.random.default_rng(15)
    base = rng.normal(size=(n, 20))
    dups = base[:, :10] + 0.05 * rng.normal(size=(n, 10))
    X = np.column_stack([base, dups])
    label = base[:, :10].sum(axis=1) + rng.normal(size=n)
    cols = {f"f{j}": X[:, j] for j in range(p)}
    _fit({"w0": X[:200, 0], "w1": X[:200, 1]}, label[:200])  # warm up numba JIT
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        f = _fit(cols, label)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert f.report_ is not None
    assert len(f.report_.dropped) == 10 and len(f.output_names_) == 20
    for j in range(10):  # exactly one of each near-duplicate pair survives
        assert (f"f{j}" in f.output_names_) != (f"f{j + 20}" in f.output_names_)
    print(f"\nscalability: {elapsed:.2f} s, tracemalloc peak {peak / 1024**2:.1f} MB")
    assert elapsed < 60.0
    assert peak < 300 * 1024**2
