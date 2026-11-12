from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.types import TaskType
from quantautoml.features import triage as triage_mod
from quantautoml.features.base import FitContext
from quantautoml.features.triage import NaNInfTriage

N = 2000


def _cfg(max_nan_ratio: float = 0.6) -> FeaturesConfig:
    return FeaturesConfig(ic_horizons={1: "fwd_ret_1"}, target_horizon=1, max_nan_ratio=max_nan_ratio)


def _ctx(names: tuple[str, ...], discrete: tuple[str, ...] = ()) -> FitContext:
    return FitContext(
        feature_names=names,
        labels_aux={},
        task_type=TaskType.BINARY_CLASSIFICATION,
        discrete_features=discrete,
    )


def _labels(seed: int = 0, n: int = N) -> NDArray[np.int32]:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.int32)


def _signal(y: NDArray[np.int32], seed: int = 1) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return (y + rng.normal(scale=0.5, size=len(y))).astype(np.float32)


def _with_nan_ratio(col: NDArray[np.float32], ratio: float, seed: int) -> NDArray[np.float32]:
    out = col.copy()
    idx = np.random.default_rng(seed).permutation(len(col))[: int(round(ratio * len(col)))]
    out[idx] = np.nan
    return out


def _fit(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    names: tuple[str, ...],
    discrete: tuple[str, ...] = (),
    cfg: FeaturesConfig | None = None,
) -> NaNInfTriage:
    return NaNInfTriage(cfg or _cfg()).fit(X, y, _ctx(names, discrete))


# --------------------------------------------------------------------------- #
# Hard Gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("discrete", [(), ("bad_all", "bad_70")])
def test_all_nan_and_high_nan_ratio_columns_dropped(discrete: tuple[str, ...]) -> None:
    y = _labels()
    rng = np.random.default_rng(3)
    good = rng.integers(0, 3, N).astype(np.float32)
    all_nan = np.full(N, np.nan, dtype=np.float32)
    high = _with_nan_ratio(rng.integers(0, 2, N).astype(np.float32), 0.7, seed=4)
    X = np.column_stack([good, all_nan, high]).astype(np.float32)
    f = _fit(X, y, ("good", "bad_all", "bad_70"), discrete=discrete)
    assert "all NaN" in f.report_.dropped["bad_all"]
    assert "nan_ratio" in f.report_.dropped["bad_70"]
    assert "good" not in f.report_.dropped
    assert "bad_all" not in f.output_names_ and "bad_70" not in f.output_names_
    assert f.output_names_ == ("good",)
    assert f.report_.n_in == 3 and f.report_.n_out == 1
    assert f.transform(X).shape == (N, 1)


def test_dropped_categorical_with_nan_ratio_above_gate() -> None:
    y = _labels()
    rng = np.random.default_rng(5)
    cat = _with_nan_ratio(rng.integers(0, 5, N).astype(np.float32), 0.7, seed=6)
    ok = rng.normal(size=N).astype(np.float32)
    X = np.column_stack([cat, ok]).astype(np.float32)
    f = _fit(X, y, ("cat", "ok"), discrete=("cat",))
    assert "cat" in f.report_.dropped
    assert f.output_names_ == ("ok",)


# --------------------------------------------------------------------------- #
# Inf handling / "none"
# --------------------------------------------------------------------------- #
def test_inf_replaced_by_nan_in_transform() -> None:
    y = _labels()
    a = np.random.default_rng(7).normal(size=N).astype(np.float32)
    X = np.column_stack([a]).astype(np.float32)
    f = _fit(X, y, ("a",))
    Xv = X[:5].copy()
    Xv[0, 0], Xv[1, 0] = np.inf, -np.inf
    out = f.transform(Xv)
    assert np.isnan(out[0, 0]) and np.isnan(out[1, 0])
    assert np.isfinite(out[2:, 0]).all()
    assert np.isinf(Xv[:2, 0]).all()  # input not mutated


def test_inf_counts_as_missing_in_fit() -> None:
    y = _labels()
    a = np.random.default_rng(8).normal(size=N).astype(np.float32)
    a[: int(0.7 * N)] = np.inf
    X = np.column_stack([a, _signal(y)]).astype(np.float32)
    f = _fit(X, y, ("a", "s"))
    assert "a" in f.report_.dropped


def test_column_without_nan_is_none_and_no_indicator() -> None:
    y = _labels()
    X = np.column_stack([_signal(y), _signal(y, seed=2)]).astype(np.float32)
    f = _fit(X, y, ("a", "b"))
    assert f.missing_kind_ == {"a": "none", "b": "none"}
    assert f.output_names_ == ("a", "b")
    assert f.transform(X).shape == (N, 2)


# --------------------------------------------------------------------------- #
# MCAR / MNAR / MAR
# --------------------------------------------------------------------------- #
def test_random_nan_independent_of_y_is_mcar() -> None:
    y = _labels()
    rng = np.random.default_rng(9)
    a = _with_nan_ratio(rng.normal(size=N).astype(np.float32), 0.2, seed=10)
    b = rng.normal(size=N).astype(np.float32)
    X = np.column_stack([a, b]).astype(np.float32)
    f = _fit(X, y, ("a", "b"))
    assert f.missing_kind_["a"] == "MCAR"
    assert "is_nan_a" not in f.output_names_


def test_nan_only_when_y_is_one_is_mnar_with_indicator() -> None:
    y = _labels()
    rng = np.random.default_rng(11)
    a = rng.normal(size=N).astype(np.float32)
    ones = np.flatnonzero(y == 1)
    a[ones[: len(ones) // 2]] = np.nan
    b = rng.normal(size=N).astype(np.float32)
    X = np.column_stack([a, b]).astype(np.float32)
    f = _fit(X, y, ("a", "b"))
    assert f.missing_kind_["a"] == "MNAR"
    assert f.output_names_ == ("a", "b", "is_nan_a")
    out = f.transform(X)
    assert out.shape == (N, 3) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:, 2], np.isnan(a).astype(np.float32))
    assert set(np.unique(out[:, 2]).tolist()) == {0.0, 1.0}
    assert f.report_.n_out == len(f.output_names_)


def test_nan_driven_by_other_feature_is_mar() -> None:
    y = _labels()
    rng = np.random.default_rng(12)
    driver = rng.normal(size=N).astype(np.float32)
    a = rng.normal(size=N).astype(np.float32)
    a[driver > 0.8] = np.nan  # missingness explained by another feature, not y
    X = np.column_stack([a, driver]).astype(np.float32)
    f = _fit(X, y, ("a", "driver"))
    assert f.missing_kind_["a"] == "MAR"
    assert f.output_names_ == ("a", "driver", "is_nan_a")


# --------------------------------------------------------------------------- #
# Bool single-value + NaN
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("value", "fill"), [(1.0, 0.0), (0.0, 1.0)])
def test_single_value_bool_imputes_complement_without_indicator(value: float, fill: float) -> None:
    y = _labels()
    flag = np.where(y == 1, value, np.nan).astype(np.float32)  # strongly MNAR-like
    X = np.column_stack([flag, _signal(y)]).astype(np.float32)
    f = _fit(X, y, ("flag", "s"), discrete=("flag",))
    assert not any(n.startswith("is_nan_") for n in f.output_names_)
    assert f.output_names_ == ("flag", "s")
    out = f.transform(X)
    assert not np.isnan(out[:, 0]).any()
    np.testing.assert_array_equal(out[y == 1, 0], value)
    np.testing.assert_array_equal(out[y == 0, 0], fill)
    assert f.scores_["flag"] > 0.0


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def test_score_decreases_with_nan_ratio_for_same_mi(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_mi(X: NDArray[np.float64], y: NDArray[np.int32], **_: object) -> NDArray[np.float64]:
        return np.full(X.shape[1], 0.5)

    monkeypatch.setattr(triage_mod, "mutual_info_classif", fake_mi)
    y = _labels()
    base = np.random.default_rng(13).normal(size=N).astype(np.float32)
    X = np.column_stack(
        [base, _with_nan_ratio(base, 0.1, 14), _with_nan_ratio(base, 0.4, 15)]
    ).astype(np.float32)
    f = _fit(X, y, ("r0", "r1", "r4"))
    assert f.scores_["r0"] == pytest.approx(0.5)
    assert f.scores_["r1"] == pytest.approx(0.5 * 0.9)
    assert f.scores_["r4"] == pytest.approx(0.5 * 0.6)
    assert f.scores_["r0"] > f.scores_["r1"] > f.scores_["r4"]


def test_mid_nan_label_rows_masked_before_mi(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[NDArray[np.int32]] = []
    real = triage_mod.mutual_info_classif

    def spy(X: NDArray[np.float64], y: NDArray[np.int32], **kw: object) -> NDArray[np.float64]:
        seen.append(np.asarray(y).copy())
        assert len(X) == len(y)
        return real(X, y, **kw)

    monkeypatch.setattr(triage_mod, "mutual_info_classif", spy)
    y = _labels()
    y[:300] = -1
    X = np.column_stack([_signal(np.clip(y, 0, 1))]).astype(np.float32)
    f = _fit(X, y, ("s",))
    assert seen and all(-1 not in s for s in seen)
    assert all(len(s) == N - 300 for s in seen)
    assert f.scores_["s"] > 0.0


def test_mid_nan_label_does_not_change_score() -> None:
    y = _labels()
    s = _signal(y)
    X = np.column_stack([s]).astype(np.float32)
    y_marked = y.copy()
    y_marked[:200] = -1
    ref = _fit(X[200:], y[200:], ("s",))
    got = _fit(X, y_marked, ("s",))
    assert got.scores_["s"] == pytest.approx(ref.scores_["s"])


def test_constant_column_scores_zero_without_warnings() -> None:
    y = _labels()
    const = np.full(N, 3.0, dtype=np.float32)
    const_nan = _with_nan_ratio(np.full(N, 2.0, dtype=np.float32), 0.3, seed=16)
    X = np.column_stack([const, const_nan, _signal(y)]).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = _fit(X, y, ("c", "cn", "s"))
    assert f.scores_["c"] == 0.0
    assert f.scores_["cn"] == 0.0
    assert f.scores_["s"] > 0.0
    assert all(np.isfinite(v) for v in f.scores_.values())


def test_discrete_mask_passed_to_mutual_info(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    real = triage_mod.mutual_info_classif

    def spy(X: NDArray[np.float64], y: NDArray[np.int32], **kw: object) -> NDArray[np.float64]:
        calls.append(kw)
        return real(X, y, **kw)

    monkeypatch.setattr(triage_mod, "mutual_info_classif", spy)
    y = _labels()
    rng = np.random.default_rng(17)
    cont = _signal(y)
    cat = rng.integers(0, 4, N).astype(np.float32)
    flag = rng.integers(0, 2, N).astype(np.float32)
    X = np.column_stack([cont, cat, flag]).astype(np.float32)
    _fit(X, y, ("cont", "cat", "flag"), discrete=("cat", "flag"))
    assert len(calls) == 1
    mask = calls[0]["discrete_features"]
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_
    assert mask.tolist() == [False, True, True]
    assert calls[0]["random_state"] == 0


# --------------------------------------------------------------------------- #
# Leakage: transform applies fitted state only
# --------------------------------------------------------------------------- #
def test_transform_val_does_not_refit_state() -> None:
    y = _labels()
    rng = np.random.default_rng(18)
    a = rng.normal(size=N).astype(np.float32)
    b = rng.normal(size=N).astype(np.float32)
    X = np.column_stack([a, b]).astype(np.float32)
    f = _fit(X, y, ("a", "b"))
    kinds, scores, names = dict(f.missing_kind_), dict(f.scores_), f.output_names_
    X_val = rng.normal(size=(500, 2)).astype(np.float32)
    X_val[:400, 0] = np.nan  # Val distribution would be gated/MNAR if refit
    out = f.transform(X_val)
    assert f.missing_kind_ == kinds and f.scores_ == scores and f.output_names_ == names
    assert out.shape == (500, 2)
    assert np.isnan(out[:400, 0]).all()


def test_bool_imputation_uses_train_state_only() -> None:
    y = _labels()
    flag = np.where(y == 1, 1.0, np.nan).astype(np.float32)
    X = np.column_stack([flag, _signal(y)]).astype(np.float32)
    f = _fit(X, y, ("flag", "s"), discrete=("flag",))
    X_val = np.column_stack([np.full(50, np.nan), np.zeros(50)]).astype(np.float32)
    np.testing.assert_array_equal(f.transform(X_val)[:, 0], 0.0)


def test_extra_thresholds_are_keyword_only() -> None:
    f = NaNInfTriage(_cfg(), indicator_mi_min=0.02, mar_corr_min=0.4)
    assert f.indicator_mi_min == 0.02 and f.mar_corr_min == 0.4
    with pytest.raises(TypeError):
        NaNInfTriage(_cfg(), 0.02)  # type: ignore[misc]


def test_refit_resets_scores_and_missing_kind() -> None:
    y = _labels()
    rng = np.random.default_rng(19)
    f = NaNInfTriage(_cfg())
    X1 = rng.normal(size=(N, 2)).astype(np.float32)
    f.fit(X1, y, _ctx(("a", "b")))
    X2 = np.column_stack([_signal(y)]).astype(np.float32)
    f.fit(X2, y, _ctx(("c",)))
    assert set(f.scores_) == {"c"}
    assert set(f.missing_kind_) == set(f.scores_) == {"c"}
    assert f.output_names_ == ("c",)


def test_all_columns_gated_yields_empty_output() -> None:
    y = _labels()
    X = np.column_stack(
        [np.full(N, np.nan), _with_nan_ratio(np.ones(N, dtype=np.float32), 0.7, 20)]
    ).astype(np.float32)
    f = _fit(X, y, ("x", "z"))
    assert f.output_names_ == () and f.report_.n_out == 0
    assert f.scores_ == {} and f.missing_kind_ == {}
    out = f.transform(X)
    assert out.shape == (N, 0) and out.dtype == np.float32
