from __future__ import annotations

import time
import warnings
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import mutual_info_score

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import FeatureSelectionError
from quantautoml.common.types import TaskType
from quantautoml.features import jmi
from quantautoml.features.base import FitContext
from quantautoml.features.jmi import JMIFeatureSelector

N = 5000


@pytest.fixture(autouse=True)
def _warnings_are_errors() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _cfg(max_features: int = 30) -> FeaturesConfig:
    return FeaturesConfig(
        ic_horizons={1: "fwd_ret_1"}, target_horizon=1, max_features=max_features
    )


def _ctx(
    names: tuple[str, ...], aux: dict[str, NDArray[np.float32]] | None = None
) -> FitContext:
    return FitContext(
        feature_names=names,
        labels_aux={} if aux is None else aux,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )


def _noise(seed: int, n: int = N) -> NDArray[np.float64]:
    return np.random.default_rng(seed).normal(size=n)


def _y(n: int = N, seed: int = 99) -> NDArray[np.int32]:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.int32)


def _fit(
    cols: dict[str, NDArray[np.float64]],
    y: NDArray[np.int32],
    max_features: int = 30,
    ctx: FitContext | None = None,
) -> JMIFeatureSelector:
    names = tuple(cols)
    X = np.column_stack(list(cols.values())).astype(np.float32)
    return JMIFeatureSelector(_cfg(max_features)).fit(
        X, y, _ctx(names) if ctx is None else ctx
    )


# -------------------------------------------------------------- MI primitive


def test_contingency_mi_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    for na, nc in ((10, 2), (100, 3), (7, 5)):
        a = rng.integers(0, na - 2, 3000)  # leaves empty top codes on purpose
        yc = rng.integers(0, nc, 3000)
        yc[a < 2] = 0  # dependence
        got = jmi._mi(a.astype(np.intp), na, yc.astype(np.intp), nc)
        assert got == pytest.approx(float(mutual_info_score(a, yc)), abs=1e-12)
    one = np.zeros(50, dtype=np.intp)
    assert jmi._mi(one, 10, rng.integers(0, 2, 50).astype(np.intp), 2) == 0.0
    assert jmi._mi(np.empty(0, np.intp), 10, np.empty(0, np.intp), 2) == 0.0


def test_marginal_mi_matches_sklearn_on_10_quantile_bins() -> None:
    y = _y()
    x = y + _noise(2)
    f = _fit({"x": x}, y)
    x32 = x.astype(np.float32).astype(np.float64)
    edges = np.unique(np.quantile(x32, np.linspace(0, 1, 11)[1:-1]))
    codes = np.searchsorted(edges, x32, side="right")
    assert f.mi_["x"] == pytest.approx(float(mutual_info_score(codes, y)), abs=1e-12)


# ------------------------------------------------------------- selection


def test_max_features_ge_n_features_keeps_all() -> None:
    y = _y()
    cols = {"n": _noise(3), "s": y + 0.5 * _noise(4), "w": y + 2.0 * _noise(5)}
    f = _fit(cols, y, max_features=30)
    assert f.output_names_ == ("n", "s", "w")
    assert set(f.selected_order_) == {"n", "s", "w"} and len(f.selected_order_) == 3
    assert f.selected_order_[0] == "s"
    assert f.report_ is not None and f.report_.dropped == {} and f.report_.n_out == 3
    g = _fit(cols, y, max_features=3)
    assert g.selected_order_ == f.selected_order_ and g.output_names_ == f.output_names_


def test_first_selected_is_max_mi() -> None:
    y = _y()
    cols = {
        "weak": y + 3.0 * _noise(6),
        "noise": _noise(7),
        "strong": y + 0.4 * _noise(8),
        "mid": y + 1.0 * _noise(9),
    }
    f = _fit(cols, y, max_features=1)
    best = max(f.mi_, key=lambda k: f.mi_[k])
    assert best == "strong" and f.selected_order_ == ("strong",)
    assert f.output_names_ == ("strong",)
    assert f.report_ is not None and set(f.report_.dropped) == {"weak", "noise", "mid"}
    assert "max_features" in f.report_.dropped["noise"]
    assert f.report_.notes


def test_near_copy_ranks_after_new_information() -> None:
    """y = 1[s1 + s2 > 0]; ``a`` ~ s1, ``a_copy`` ~ a, ``b`` ~ s2 (noisier).

    Top-k by marginal MI would pick (a, a_copy); JMI picks b second because
    I(a_copy, a; Y) ~ I(a; Y) while I(b, a; Y) adds I(b; Y | a).
    """
    s1, s2 = _noise(10), _noise(11)
    y = (s1 + s2 > 0).astype(np.int32)
    a = s1 + 0.1 * _noise(12)
    cols = {"a_copy": a + 0.01 * _noise(13), "b": s2 + 0.6 * _noise(14), "a": a}
    f = _fit(cols, y, max_features=2)
    # a plain top-2-by-MI selector would keep the copy: the test discriminates
    assert f.mi_["a_copy"] > f.mi_["b"] + 0.02 and f.mi_["a"] > f.mi_["b"] + 0.02
    assert f.selected_order_[0] in {"a", "a_copy"}
    assert f.selected_order_[1] == "b"
    f3 = _fit(cols, y, max_features=3)
    assert f3.selected_order_[1] == "b"


def _xor_member(seed: int, p_pos: float = 0.6) -> NDArray[np.float64]:
    """|N(0,1)| with sign + w.p. ``p_pos`` (independent of everything else)."""
    rng = np.random.default_rng(seed)
    sign = np.where(rng.random(N) < p_pos, 1.0, -1.0)
    return sign * np.abs(rng.normal(size=N))


def test_xor_pair_selected_together() -> None:
    """y = 1[lead * partner > 0]; partner's sign is + w.p. 0.75, lead's w.p. 0.5.

    The partner's skew gives the *lead* marginal MI (P(y=1 | lead>0) = P(partner>0)
    = 0.75 -> ~0.13 nats) while the partner alone carries ~0 (P(y=1 | partner>0) =
    0.5). A distractor ``d = y + 2.5 noise`` has more marginal MI than the partner, so
    a plain top-2-by-MI selector keeps {lead, d}; JMI keeps {lead, partner} because
    I(partner, lead; Y) ~ 0.6 nats >> I(d, lead; Y). Fixed seeds; margins asserted.

    Limitation: the first pick is by marginal MI, so for a *pure* XOR (both members
    ~0 marginal MI) the first slot can go to noise/distractor and greedy JMI may then
    never recover the pair; this test does not (and cannot) cover that case.
    """
    lead, partner = _xor_member(21, p_pos=0.5), _xor_member(20, p_pos=0.75)
    y = (lead * partner > 0).astype(np.int32)
    cols = {"d": y + 2.5 * _noise(23), "lead": lead, "noise": _noise(22), "partner": partner}
    f = _fit(cols, y, max_features=2)
    assert max(f.mi_, key=lambda k: f.mi_[k]) == "lead" and f.mi_["lead"] > 0.1
    assert f.mi_["partner"] < 0.005  # alone: ~no information
    assert f.mi_["d"] > 5 * f.mi_["partner"]  # top-2 by MI would pick {lead, d}
    assert f.selected_order_ == ("lead", "partner")
    assert f.output_names_ == ("lead", "partner")
    assert f.report_ is not None and set(f.report_.dropped) == {"d", "noise"}
    codes = (lead > 0).astype(np.intp) * 2 + (partner > 0).astype(np.intp)
    assert float(mutual_info_score(codes, y)) > 0.6  # the pair is near-deterministic


def test_deterministic_order_and_transform_keeps_original_order() -> None:
    y = _y()
    cols = {
        "w": y + 2.0 * _noise(30),
        "n": _noise(31),
        "s": y + 0.4 * _noise(32),
        "m": y + 1.0 * _noise(33),
    }
    f1, f2 = _fit(cols, y, max_features=3), _fit(cols, y, max_features=3)
    assert f1.selected_order_ == f2.selected_order_ and f1.mi_ == f2.mi_
    assert f1.selected_order_[0] == "s"
    assert f1.selected_order_ != f1.output_names_  # selection order != column order
    names = tuple(cols)
    assert f1.output_names_ == tuple(nm for nm in names if nm in f1.selected_order_)
    Xv = np.random.default_rng(34).normal(size=(40, 4)).astype(np.float32)
    out = f1.transform(Xv)
    idx = [names.index(nm) for nm in f1.output_names_]
    assert out.dtype == np.float32 and out.shape == (40, 3)
    np.testing.assert_array_equal(out, Xv[:, idx])


def test_tie_breaks_to_earlier_column() -> None:
    y = _y()
    s = y + 0.5 * _noise(35)
    f = _fit({"c1": s, "c2": s.copy(), "c3": s.copy()}, y, max_features=2)
    assert f.selected_order_ == ("c1", "c2")


def test_mid_nan_labels_masked() -> None:
    y = _y()
    cols = {"a": y + 0.5 * _noise(40), "b": _noise(41), "c": y + 1.5 * _noise(42)}
    X = np.column_stack(list(cols.values())).astype(np.float32)
    y_mask = y.copy()
    y_mask[:300] = -1
    X_mask = X.copy()
    X_mask[:300] = 1e6  # garbage on masked rows must not move bin edges
    full = JMIFeatureSelector(_cfg(2)).fit(X_mask, y_mask, _ctx(tuple(cols)))
    cut = JMIFeatureSelector(_cfg(2)).fit(X[300:], y[300:], _ctx(tuple(cols)))
    assert full.mi_ == cut.mi_
    assert full.selected_order_ == cut.selected_order_


def test_constant_nan_and_inf_columns_no_warnings() -> None:
    y = _y()
    inf_col = y + 0.5 * _noise(50)
    inf_col[::5] = np.inf
    cols = {"const": np.full(N, 2.0), "nan": np.full(N, np.nan), "inf": inf_col}
    f = _fit(cols, y)
    assert f.mi_["const"] == 0.0 and f.mi_["nan"] == 0.0 and f.mi_["inf"] > 0.05
    assert f.selected_order_[0] == "inf"
    assert f.output_names_ == ("const", "nan", "inf")


def test_zero_features_handled() -> None:
    X = np.empty((N, 0), dtype=np.float32)
    f = JMIFeatureSelector(_cfg()).fit(X, _y(), _ctx(()))
    assert f.output_names_ == () and f.selected_order_ == () and f.mi_ == {}
    assert f.report_ is not None and f.report_.n_in == 0 and f.report_.n_out == 0
    assert f.transform(np.empty((4, 0), dtype=np.float32)).shape == (4, 0)


def test_all_labels_masked_no_crash() -> None:
    y = np.full(N, -1, dtype=np.int32)
    f = _fit({"a": _noise(55), "b": _noise(56)}, y, max_features=1)
    assert f.mi_ == {"a": 0.0, "b": 0.0} and f.selected_order_ == ("a",)


def test_transform_never_reads_labels_aux() -> None:
    class CountingDict(dict[str, NDArray[np.float32]]):
        reads = 0

        def __getitem__(self, key: str) -> NDArray[np.float32]:
            CountingDict.reads += 1
            return super().__getitem__(key)

    y = _y()
    aux = CountingDict({"fwd_ret_1": _noise(60).astype(np.float32)})
    cols = {"n": _noise(61), "s": y + 0.3 * _noise(62)}
    f = _fit(cols, y, max_features=1, ctx=_ctx(tuple(cols), aux=aux))
    CountingDict.reads = 0
    aux.clear()
    Xv = np.random.default_rng(63).normal(size=(10, 2)).astype(np.float32)
    out = f.transform(Xv)
    assert CountingDict.reads == 0
    np.testing.assert_array_equal(out[:, 0], Xv[:, 1])


def test_transform_before_fit_raises() -> None:
    with pytest.raises(FeatureSelectionError):
        JMIFeatureSelector(_cfg()).transform(np.zeros((3, 2), dtype=np.float32))


def test_refit_leaves_no_stale_state() -> None:
    y = _y()
    f = _fit({"a": y + 0.5 * _noise(70), "b": _noise(71), "c": _noise(72)}, y, 1)
    assert f.report_ is not None and f.report_.dropped and f.report_.notes
    f.fit(_noise(73).astype(np.float32)[:, None], y, _ctx(("z",)))
    assert f.selected_order_ == ("z",) and f.output_names_ == ("z",)
    assert f.mi_.keys() == {"z"}
    assert f.report_ is not None and f.report_.dropped == {} and f.report_.notes == ()
    assert f.report_.n_in == 1 and f.transform(np.zeros((5, 1), np.float32)).shape == (5, 1)


def test_performance_60_features_20k_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic cost check: exactly p marginal + sum_{t=1}^{k-1} (p - t) joint MIs.

    Measured ~0.4-0.5 s locally; the wall-clock bound is only a very generous backstop.
    """
    calls = 0
    real = jmi._mi

    def counting(a: NDArray[np.intp], na: int, yc: NDArray[np.intp], nc: int) -> float:
        nonlocal calls
        calls += 1
        return real(a, na, yc, nc)

    monkeypatch.setattr(jmi, "_mi", counting)
    n, p, k = 20_000, 60, 30
    rng = np.random.default_rng(80)
    y = rng.integers(0, 3, n).astype(np.int32)
    Z = rng.normal(size=(n, p)) + 0.05 * np.arange(p)[None, :] % 7 * y[:, None]
    X = Z.astype(np.float32)
    names = tuple(f"f{j}" for j in range(p))
    t0 = time.perf_counter()
    f = JMIFeatureSelector(_cfg(k)).fit(X, y, _ctx(names))
    elapsed = time.perf_counter() - t0
    assert len(f.selected_order_) == k and len(f.output_names_) == k
    assert calls == p + sum(p - t for t in range(1, k))  # 1365: O(F*K), MI cached
    assert elapsed < 30.0
