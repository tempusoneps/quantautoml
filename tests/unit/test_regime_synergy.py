from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.features import regime_synergy
from quantautoml.features.base import FitContext
from quantautoml.features.regime_synergy import RegimeSynergyFilter, label_regimes

LABEL = "fwd_ret_1"
W = 63
N = 3000
BLOCK = 250


@pytest.fixture(autouse=True)
def _warnings_are_errors() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _cfg(horizons: dict[int, str] | None = None) -> FeaturesConfig:
    hz = horizons if horizons is not None else {1: LABEL, 5: "fwd_ret_5"}
    return FeaturesConfig(ic_horizons=hz, target_horizon=max(hz))


def _ret(seed: int = 0, n: int = N) -> NDArray[np.float32]:
    """Forward 1-bar returns with persistent drift blocks: +0.4, -0.4, 0 repeating."""
    drift = np.repeat(np.resize([0.4, -0.4, 0.0], n // BLOCK + 1), BLOCK)[:n]
    return (drift + np.random.default_rng(seed).normal(size=n)).astype(np.float32)


def _noise(seed: int, n: int = N) -> NDArray[np.float64]:
    return np.random.default_rng(seed).normal(size=n)


def _y(n: int = N, seed: int = 99) -> NDArray[np.int32]:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.int32)


def _ctx(
    names: tuple[str, ...],
    ret: NDArray[np.float32] | None = None,
    aux: dict[str, NDArray[np.float32]] | None = None,
) -> FitContext:
    return FitContext(
        feature_names=names,
        labels_aux=aux if aux is not None else {LABEL: _ret() if ret is None else ret},
        task_type=TaskType.BINARY_CLASSIFICATION,
    )


def _fit(
    cols: dict[str, NDArray[np.float64]],
    y: NDArray[np.int32],
    ctx: FitContext | None = None,
    cfg: FeaturesConfig | None = None,
) -> RegimeSynergyFilter:
    names = tuple(cols)
    X = np.column_stack(list(cols.values())).astype(np.float32)
    return RegimeSynergyFilter(_cfg() if cfg is None else cfg).fit(
        X, y, _ctx(names) if ctx is None else ctx
    )


# ------------------------------------------------------------- label_regimes


@pytest.mark.parametrize("t", [0, 1, W - 1, W, W + 1, 500, 1777, N - 1])
def test_label_regimes_is_causal(t: int) -> None:
    ret = _ret()
    base = label_regimes(ret)
    tampered = ret.copy()
    tampered[t:] = np.random.default_rng(t).normal(loc=5.0, scale=10.0, size=N - t)
    tampered[t::7] = np.nan
    np.testing.assert_array_equal(label_regimes(tampered)[: t + 1], base[: t + 1])


def test_label_regimes_values_dtype_and_warmup() -> None:
    lab = label_regimes(_ret())
    assert lab.dtype == np.int8 and lab.shape == (N,)
    assert set(np.unique(lab).tolist()) == {-1, 0, 1}
    np.testing.assert_array_equal(lab[:W], 0)  # insufficient history -> Sideways
    # drift blocks are recovered after the trailing window has filled
    for start, want in ((BLOCK * 3, 1), (BLOCK * 4, -1)):
        seg = lab[start + W : start + BLOCK]
        assert (seg == want).mean() > 0.7


def test_label_regimes_nan_counts_as_zero_and_degenerate_is_sideways() -> None:
    ret = _ret()
    ret_nan = ret.copy()
    ret_nan[100:110] = np.nan
    ret_zero = ret.copy()
    ret_zero[100:110] = 0.0
    np.testing.assert_array_equal(label_regimes(ret_nan), label_regimes(ret_zero))
    np.testing.assert_array_equal(label_regimes(np.full(300, np.nan, np.float32)), 0)
    np.testing.assert_array_equal(label_regimes(np.ones(300, np.float32)), 0)  # sigma == 0
    assert label_regimes(np.empty(0, np.float32)).shape == (0,)


def test_label_regimes_rejects_bad_args() -> None:
    with pytest.raises(InvalidDataError):
        label_regimes(_ret(), window=0)
    with pytest.raises(InvalidDataError):
        label_regimes(_ret(), band=-0.1)
    with pytest.raises(InvalidDataError):
        label_regimes(_ret()[:, None])


# ---------------------------------------------------------------- filter fit


def test_bear_only_signal_kept_and_pure_noise_dropped() -> None:
    y = _y()
    regime = label_regimes(_ret())
    bear = regime == -1
    assert bear.sum() > 300
    signal = np.where(bear, y + 0.3 * _noise(1), _noise(2))
    # pure noise: N=3000, seed 3; best per-regime 5-bin MI stays well below mi_floor
    f = _fit({"noise": _noise(3), "bear_sig": signal}, y)
    mi_bear = f.mi_by_regime_["bear_sig"]
    assert mi_bear[-1] > 0.2 and max(mi_bear[1], mi_bear[0]) < f.mi_floor
    assert max(f.mi_by_regime_["noise"].values()) < f.mi_floor
    assert f.synergy_pairs_ == ()
    assert f.output_names_ == ("bear_sig",)
    assert f.report_ is not None and f.report_.n_in == 2 and f.report_.n_out == 1
    assert "mi_floor" in f.report_.dropped["noise"]


def test_xor_pair_protected_by_synergy() -> None:
    x1, x2 = _noise(10), _noise(11)
    y = (x1 * x2 > 0).astype(np.int32)
    f = _fit({"x1": x1, "noise": _noise(12), "x2": x2}, y)
    for nm in ("x1", "x2"):
        assert max(f.mi_by_regime_[nm].values()) < f.mi_floor  # alone: MI ~ 0
    assert f.synergy_pairs_ == (("x1", "x2"),)
    assert f.output_names_ == ("x1", "x2")
    assert f.report_ is not None and set(f.report_.dropped) == {"noise"}


def test_mid_nan_labels_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    real = regime_synergy.mutual_info_score

    def spy(a: None, b: None, *, contingency: NDArray[np.int64]) -> float:
        assert a is None and b is None
        assert contingency.shape[1] == 2  # classes {0, 1}: label -1 is not a class
        seen.append(int(contingency.sum()))
        return float(real(a, b, contingency=contingency))

    monkeypatch.setattr(regime_synergy, "mutual_info_score", spy)
    y = _y()
    y[:200] = -1
    _fit({"a": _noise(20), "b": _noise(21)}, y)
    assert seen and max(seen) == N - 200


def _small_regime(n_small: int = 100) -> NDArray[np.int8]:
    """Fixed regimes: Bear has ``n_small`` rows, Bull / Sideways split the rest."""
    fixed = np.zeros(N, dtype=np.int8)
    fixed[: N // 2] = 1
    fixed[N // 2 : N // 2 + n_small] = -1
    return fixed


@pytest.mark.parametrize("n_classes", [2, 3])
def test_noise_in_small_regime_mostly_dropped(
    monkeypatch: pytest.MonkeyPatch, n_classes: int
) -> None:
    """Aggregate check: of 400 pure-noise features, <= 5% survive a 100-row regime.

    Fixed seeds (not seed-picked); the assertion is on an aggregate kept fraction.
    Over 8 other seed pairs the corrected gate kept 0.25-1% (binary) / 0.5-2.5% (3-class,
    incl. spurious synergy pairs), versus ~73% / ~99% for uncorrected plug-in MI.
    """
    monkeypatch.setattr(regime_synergy, "label_regimes", lambda *_a, **_k: _small_regime())
    y = np.random.default_rng(500 + n_classes).integers(0, n_classes, N).astype(np.int32)
    Z = np.random.default_rng(600 + n_classes).normal(size=(N, 400))
    cols = {f"z{j}": Z[:, j] for j in range(400)}
    f = _fit(cols, y)
    assert f.report_ is not None
    assert len(f.output_names_) / 400 <= 0.05
    assert all(set(v) == {-1, 0, 1} for v in f.mi_by_regime_.values())


def test_strong_signal_only_in_small_regime_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = _small_regime()
    monkeypatch.setattr(regime_synergy, "label_regimes", lambda *_a, **_k: fixed)
    y = _y()
    sig = np.where(fixed == -1, y + 0.3 * _noise(700), _noise(701))
    f = _fit({"sig": sig, "const": np.full(N, 3.0)}, y)
    mi = f.mi_by_regime_["sig"]
    assert mi[-1] > 0.2 and max(mi[1], mi[0]) < f.mi_floor
    assert f.output_names_ == ("sig",)


def test_small_regime_skipped_and_noted(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = np.zeros(N, dtype=np.int8)
    fixed[:1500] = 1
    fixed[1500:1530] = -1  # 30 bear rows < 50
    monkeypatch.setattr(regime_synergy, "label_regimes", lambda *_a, **_k: fixed)
    y = _y()
    f = _fit({"a": _noise(30), "b": y + 0.3 * _noise(31)}, y)
    assert set(f.mi_by_regime_["a"]) == {0, 1}
    assert f.report_ is not None
    assert any("-1" in note and "30" in note for note in f.report_.notes)
    assert f.output_names_ == ("b",)


def test_missing_h1_column_or_horizon_is_pass_through() -> None:
    y = _y()
    cols = {"a": _noise(40), "b": _noise(41)}
    f = _fit(cols, y, ctx=_ctx(tuple(cols), aux={}))
    assert f.output_names_ == ("a", "b") and f.mi_by_regime_ == {}
    assert f.report_ is not None and f.report_.dropped == {} and f.report_.notes
    assert LABEL in f.report_.notes[0]

    g = _fit(cols, y, cfg=_cfg({5: "fwd_ret_5"}))
    assert g.output_names_ == ("a", "b")
    assert g.report_ is not None and "h=1" in g.report_.notes[0]


def test_misaligned_h1_column_raises() -> None:
    cols = {"a": _noise(42)}
    ctx = _ctx(("a",), aux={LABEL: _ret(n=N - 1)})
    with pytest.raises(InvalidDataError, match=LABEL):
        _fit(cols, _y(), ctx=ctx)


def test_constant_nan_and_inf_columns_no_warnings() -> None:
    y = _y()
    inf_col = _noise(50)
    inf_col[::5] = np.inf
    nan_col = np.full(N, np.nan)
    f = _fit({"const": np.full(N, 2.0), "nan": nan_col, "inf": inf_col}, y)
    assert all(v == 0.0 for v in f.mi_by_regime_["const"].values())
    assert all(v == 0.0 for v in f.mi_by_regime_["nan"].values())
    assert f.report_ is not None and f.report_.n_out == len(f.output_names_)


def test_zero_features_handled() -> None:
    X = np.empty((N, 0), dtype=np.float32)
    f = RegimeSynergyFilter(_cfg()).fit(X, _y(), _ctx(()))
    assert f.output_names_ == () and f.mi_by_regime_ == {} and f.synergy_pairs_ == ()
    assert f.transform(np.empty((4, 0), dtype=np.float32)).shape == (4, 0)


def test_deterministic_and_original_order() -> None:
    x1, x2 = _noise(60), _noise(61)
    y = (x1 * x2 > 0).astype(np.int32)
    cols = {"x2": x2, "n": _noise(62), "sig": y + 0.5 * _noise(63), "x1": x1}
    f1, f2 = _fit(cols, y), _fit(cols, y)
    assert f1.mi_by_regime_ == f2.mi_by_regime_
    assert f1.synergy_pairs_ == f2.synergy_pairs_ == (("x2", "x1"),)
    assert f1.output_names_ == ("x2", "sig", "x1")


def test_transform_selects_columns_without_reading_aux() -> None:
    class CountingDict(dict[str, NDArray[np.float32]]):
        reads = 0

        def __getitem__(self, key: str) -> NDArray[np.float32]:
            CountingDict.reads += 1
            return super().__getitem__(key)

    y = _y()
    aux = CountingDict({LABEL: _ret()})
    cols = {"n": _noise(70), "s": y + 0.3 * _noise(71)}
    f = _fit(cols, y, ctx=_ctx(tuple(cols), aux=aux))
    assert f.output_names_ == ("s",)
    CountingDict.reads = 0
    aux.clear()
    Xv = np.random.default_rng(72).normal(size=(40, 2)).astype(np.float32)
    out = f.transform(Xv)
    assert CountingDict.reads == 0
    assert out.dtype == np.float32 and out.shape == (40, 1)
    np.testing.assert_array_equal(out[:, 0], Xv[:, 1])


def test_transform_before_fit_raises() -> None:
    with pytest.raises(FeatureSelectionError):
        RegimeSynergyFilter(_cfg()).transform(np.zeros((3, 2), dtype=np.float32))


def test_refit_leaves_no_stale_state() -> None:
    x1, x2 = _noise(80), _noise(81)
    y = (x1 * x2 > 0).astype(np.int32)
    f = _fit({"x1": x1, "x2": x2, "const": np.full(N, 1.5)}, y)  # MI == 0 always
    assert f.synergy_pairs_ and f.report_ is not None and f.report_.dropped
    y2 = _y()
    cols = {"z": _noise(83)}
    f.fit(np.asarray(cols["z"], np.float32)[:, None], y2, _ctx(("z",), aux={}))
    assert f.synergy_pairs_ == () and f.mi_by_regime_ == {}
    assert f.output_names_ == ("z",)
    assert f.report_ is not None and f.report_.dropped == {} and f.report_.n_in == 1
    assert f.transform(np.zeros((5, 1), np.float32)).shape == (5, 1)


def test_constructor_rejects_bad_thresholds() -> None:
    with pytest.raises(InvalidDataError):
        RegimeSynergyFilter(_cfg(), mi_floor=-1.0)
    with pytest.raises(InvalidDataError):
        RegimeSynergyFilter(_cfg(), synergy_delta=-1.0)
