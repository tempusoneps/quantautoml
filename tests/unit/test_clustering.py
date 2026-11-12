from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.features import clustering
from quantautoml.features.base import FitContext
from quantautoml.features.clustering import (
    ICBehaviorClusterFilter,
    ic_behavior_distance,
)

LABEL = "fwd_ret_1"
W = 63
N = W * 40


@pytest.fixture(autouse=True)
def _warnings_are_errors() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _cfg() -> FeaturesConfig:
    return FeaturesConfig(ic_horizons={1: LABEL, 5: "fwd_ret_5"}, target_horizon=1)


def _label(seed: int = 0, n: int = N) -> NDArray[np.float64]:
    return np.random.default_rng(seed).normal(size=n)


def _loading(seed: int, mean: float = 0.0, n: int = N) -> NDArray[np.float64]:
    """Per-window IC loading a(t), constant inside each window of W rows."""
    a = np.random.default_rng(seed).normal(loc=mean, scale=1.0, size=n // W)
    return np.repeat(a, W)


def _noise(seed: int, n: int = N) -> NDArray[np.float64]:
    return np.random.default_rng(seed).normal(size=n)


def _y(n: int = N, seed: int = 99) -> NDArray[np.int32]:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.int32)


def _ctx(
    names: tuple[str, ...],
    label: NDArray[np.float64],
    aux: dict[str, NDArray[np.float32]] | None = None,
    discrete: tuple[str, ...] = (),
) -> FitContext:
    return FitContext(
        feature_names=names,
        labels_aux=aux if aux is not None else {LABEL: label.astype(np.float32)},
        task_type=TaskType.BINARY_CLASSIFICATION,
        discrete_features=discrete,
    )


def _fit(
    cols: dict[str, NDArray[np.float64]],
    label: NDArray[np.float64],
    y: NDArray[np.int32] | None = None,
) -> ICBehaviorClusterFilter:
    names = tuple(cols)
    X = np.column_stack(list(cols.values())).astype(np.float32)
    return ICBehaviorClusterFilter(_cfg()).fit(
        X, _y(len(X)) if y is None else y, _ctx(names, label)
    )


# ------------------------------------------------------------ distance helper


def test_distance_identical_and_independent_series() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=40)
    roll = np.column_stack([a, -a, rng.normal(size=40)])
    d = ic_behavior_distance(roll)
    assert d.shape == (3, 3) and d.dtype == np.float64
    np.testing.assert_array_equal(d, d.T)
    np.testing.assert_array_equal(np.diag(d), 0.0)
    assert d[0, 1] == pytest.approx(0.0, abs=1e-12)  # |corr| = 1 for anti-correlated too
    assert d[0, 2] > 0.6


def test_distance_constant_nan_and_sparse_series_are_one() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=20)
    sparse = np.full(20, np.nan)
    sparse[:2] = a[:2]  # only 2 common finite windows (< 3)
    roll = np.column_stack([a, np.full(20, 0.5), np.full(20, np.nan), sparse])
    d = ic_behavior_distance(roll)
    assert np.isfinite(d).all()
    off = ~np.eye(4, dtype=bool)
    np.testing.assert_array_equal(d[off], 1.0)
    np.testing.assert_array_equal(np.diag(d), 0.0)


def test_distance_uses_pairwise_common_windows() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=30)
    b = a.copy()
    b[:5] = np.nan
    a2 = a.copy()
    a2[25:] = np.nan
    d = ic_behavior_distance(np.column_stack([a2, b]))
    assert d[0, 1] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------- filter fit


def test_near_duplicate_ic_pattern_pair_same_cluster_one_survives() -> None:
    lab = _label()
    a = _loading(10, mean=0.5)
    cols = {"a1": a * lab + _noise(11), "a2": a * lab + _noise(12)}
    f = _fit(cols, lab)
    assert f.clusters_ == {0: ("a1", "a2")}
    assert len(f.output_names_) == 1
    rep = f.representatives_[0]
    assert f.output_names_ == (rep,)
    assert f.report_ is not None and f.report_.n_in == 2 and f.report_.n_out == 1
    (other,) = {"a1", "a2"} - {rep}
    assert rep in f.report_.dropped[other]


def test_independent_ic_patterns_separate_clusters_both_kept() -> None:
    lab = _label()
    cols = {
        "p": _loading(20, 0.3) * lab + _noise(21),
        "q": _loading(22, 0.3) * lab + _noise(23),
    }
    f = _fit(cols, lab)
    assert f.clusters_ == {0: ("p",), 1: ("q",)}
    assert f.representatives_ == {0: "p", 1: "q"}
    assert f.output_names_ == ("p", "q")
    assert f.report_ is not None and f.report_.dropped == {}


def test_single_feature_pass_through() -> None:
    lab = _label()
    f = _fit({"only": _noise(30)}, lab)
    assert f.output_names_ == ("only",)
    assert f.clusters_ == {0: ("only",)} and f.representatives_ == {0: "only"}
    X = _noise(31, 50).astype(np.float32)[:, None]
    np.testing.assert_array_equal(f.transform(X), X)


def test_zero_features_handled() -> None:
    lab = _label()
    X = np.empty((N, 0), dtype=np.float32)
    f = ICBehaviorClusterFilter(_cfg()).fit(X, _y(), _ctx((), lab))
    assert f.output_names_ == () and f.clusters_ == {} and f.representatives_ == {}
    assert f.report_ is not None and f.report_.n_in == 0 and f.report_.n_out == 0
    assert f.transform(np.empty((5, 0), dtype=np.float32)).shape == (5, 0)


def test_constant_ic_series_distance_one_no_nan() -> None:
    lab = _label()
    a = _loading(40, 0.5)
    cols = {
        "perfect": lab.copy(),  # IC == 1 in every window -> zero-variance IC series
        "const": np.full(N, 2.0),  # every window degenerate -> all-NaN IC series
        "inf": np.where(np.arange(N) < W, np.inf, a * lab + _noise(41)),
        "b": a * lab + _noise(42),
    }
    f = _fit(cols, lab)
    clusters = set(f.clusters_.values())
    assert ("perfect",) in clusters and ("const",) in clusters
    assert set(f.output_names_) >= {"perfect", "const"}
    assert f.report_ is not None and f.report_.n_out == len(f.output_names_)


def test_representative_is_max_mi_not_max_ic() -> None:
    lab = _label()
    y = _y()
    a = _loading(50, mean=0.8)
    cols = {
        "hi_ic": a * lab + _noise(51),
        "hi_mi": a * lab + _noise(52) + 3.0 * y,  # y independent of lab: dilutes IC
    }
    f = _fit(cols, lab, y)
    assert f.clusters_ == {0: ("hi_ic", "hi_mi")}
    # unambiguous, opposing orderings
    assert f.mi_["hi_mi"] > f.mi_["hi_ic"] + 0.1
    assert abs(f.rank_ic_["hi_ic"]) > abs(f.rank_ic_["hi_mi"]) + 0.05
    assert f.representatives_ == {0: "hi_mi"}
    assert f.output_names_ == ("hi_mi",)


def test_mi_tie_breaks_on_abs_rank_ic_then_column_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def zero_mi(X: NDArray[np.float64], *_: object, **__: object) -> NDArray[np.float64]:
        return np.zeros(X.shape[1], dtype=np.float64)

    monkeypatch.setattr(clustering, "mutual_info_classif", zero_mi)
    lab = _label()
    a = _loading(60, mean=0.8)
    base = a * lab + _noise(61)
    cols = {"weak": base + 0.6 * _noise(62), "strong": base}
    f = _fit(cols, lab)
    assert f.clusters_ == {0: ("weak", "strong")}
    assert abs(f.rank_ic_["strong"]) > abs(f.rank_ic_["weak"])
    assert f.representatives_ == {0: "strong"}

    f2 = _fit({"dup0": base, "dup1": base.copy()}, lab)
    assert f2.representatives_ == {0: "dup0"}


def test_mi_masks_mid_nan_labels_and_uses_discrete_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, NDArray[np.bool_]]] = []
    real = clustering.mutual_info_classif

    def spy(
        X: NDArray[np.float64],
        y: NDArray[np.int32],
        *,
        discrete_features: NDArray[np.bool_],
        random_state: int,
    ) -> NDArray[np.float64]:
        assert random_state == 0
        assert not np.isnan(X).any() and (y != -1).all()
        seen.append((len(y), np.asarray(discrete_features).copy()))
        return real(X, y, discrete_features=discrete_features, random_state=random_state)

    monkeypatch.setattr(clustering, "mutual_info_classif", spy)
    lab = _label()
    y = _y()
    y[:100] = -1
    cont = _loading(70, 0.5) * lab + _noise(71)
    cont[5:9] = np.nan
    disc = np.round(np.clip(cont, -2, 2))
    X = np.column_stack([cont, disc]).astype(np.float32)
    ICBehaviorClusterFilter(_cfg()).fit(X, y, _ctx(("c", "d"), lab, discrete=("d",)))
    assert len(seen) == 1
    assert seen[0][0] == N - 100
    np.testing.assert_array_equal(seen[0][1], [False, True])


def test_deterministic_across_runs() -> None:
    lab = _label()
    a, b = _loading(80, 0.5), _loading(81, 0.5)
    cols = {
        "a1": a * lab + _noise(82),
        "b1": b * lab + _noise(83),
        "a2": a * lab + _noise(84),
        "b2": b * lab + _noise(85),
        "c": _loading(86, 0.5) * lab + _noise(87),
    }
    f1, f2 = _fit(cols, lab), _fit(cols, lab)
    assert f1.clusters_ == f2.clusters_
    assert f1.representatives_ == f2.representatives_
    assert f1.mi_ == f2.mi_ and f1.output_names_ == f2.output_names_
    # cluster ids ordered by smallest original column index; output in original order
    assert f1.clusters_ == {0: ("a1", "a2"), 1: ("b1", "b2"), 2: ("c",)}
    order = list(cols)
    assert list(f1.output_names_) == sorted(f1.output_names_, key=order.index)
    assert f1.report_ is not None and set(f1.report_.dropped) | set(f1.output_names_) == set(cols)


def test_missing_target_column_raises_naming_it() -> None:
    lab = _label()
    X = _noise(90)[:, None].astype(np.float32)
    with pytest.raises(InvalidDataError, match=LABEL):
        ICBehaviorClusterFilter(_cfg()).fit(X, _y(), _ctx(("x",), lab, aux={}))


def test_short_train_raises() -> None:
    n = 2 * W - 1
    lab = _label(n=n)
    X = _noise(91, n)[:, None].astype(np.float32)
    with pytest.raises(InvalidDataError, match=str(n)):
        ICBehaviorClusterFilter(_cfg()).fit(X, _y(n), _ctx(("x",), lab))


def test_transform_selects_columns_without_reading_aux() -> None:
    class CountingDict(dict[str, NDArray[np.float32]]):
        reads = 0

        def __getitem__(self, key: str) -> NDArray[np.float32]:
            CountingDict.reads += 1
            return super().__getitem__(key)

    lab = _label()
    a = _loading(100, 0.5)
    X = np.column_stack([a * lab + _noise(101), a * lab + _noise(102)]).astype(np.float32)
    aux = CountingDict({LABEL: lab.astype(np.float32)})
    f = ICBehaviorClusterFilter(_cfg()).fit(X, _y(), _ctx(("u", "v"), lab, aux=aux))
    assert len(f.output_names_) == 1
    keep = ("u", "v").index(f.output_names_[0])
    CountingDict.reads = 0
    aux.clear()
    Xv = np.random.default_rng(103).normal(size=(40, 2)).astype(np.float32)
    out = f.transform(Xv)
    assert CountingDict.reads == 0
    assert out.dtype == np.float32 and out.shape == (40, 1)
    np.testing.assert_array_equal(out[:, 0], Xv[:, keep])


def test_transform_before_fit_raises() -> None:
    with pytest.raises(FeatureSelectionError):
        ICBehaviorClusterFilter(_cfg()).transform(np.zeros((3, 2), dtype=np.float32))


def test_refit_leaves_no_stale_state() -> None:
    lab = _label()
    a = _loading(110, 0.5)
    f = _fit({"a1": a * lab + _noise(111), "a2": a * lab + _noise(112)}, lab)
    assert f.report_ is not None and f.report_.dropped
    X2 = _noise(113)[:, None].astype(np.float32)
    f.fit(X2, _y(), _ctx(("z",), lab))
    assert f.clusters_ == {0: ("z",)} and f.representatives_ == {0: "z"}
    assert set(f.mi_) == {"z"} and set(f.rank_ic_) == {"z"}
    assert f.output_names_ == ("z",)
    assert f.report_ is not None and f.report_.dropped == {} and f.report_.n_in == 1
    assert f.transform(X2).shape == (N, 1)
