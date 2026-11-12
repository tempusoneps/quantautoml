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
from quantautoml.features import ic_stability
from quantautoml.features.base import FitContext
from quantautoml.features.ic_stability import ICStabilityFilter, _rolling_rank_ic
from quantautoml.metrics.fdr import apply_fdr_correction
from quantautoml.metrics.ic import calc_icir, calc_rank_ic, calc_rolling_ic

HORIZONS = (1, 5, 10, 21)
COLS = {h: f"fwd_ret_{h}" for h in HORIZONS}
N = 3150  # 50 non-overlapping windows of 63


@pytest.fixture(autouse=True)
def _warnings_are_errors() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _cfg(horizons: dict[int, str] | None = None, **kw: float) -> FeaturesConfig:
    return FeaturesConfig(ic_horizons=horizons or dict(COLS), target_horizon=1, **kw)  # type: ignore[arg-type]


def _ctx(names: tuple[str, ...], aux: dict[str, NDArray[np.float32]]) -> FitContext:
    return FitContext(
        feature_names=names, labels_aux=aux, task_type=TaskType.BINARY_CLASSIFICATION
    )


def _y(n: int) -> NDArray[np.int32]:
    return np.random.default_rng(0).integers(0, 2, n).astype(np.int32)


def _market(n: int = N, seed: int = 0) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float32]]]:
    """Return next-bar returns r[t+1] and forward-return labels with a NaN tail of h rows."""
    r = np.random.default_rng(seed).normal(size=n + max(HORIZONS) + 1)
    csum = np.concatenate([[0.0], np.cumsum(r)])
    aux: dict[str, NDArray[np.float32]] = {}
    for h in HORIZONS:
        fwd = csum[np.arange(n) + 1 + h] - csum[np.arange(n) + 1]  # sum r[t+1..t+h]
        fwd[n - h :] = np.nan  # realistic unrealized tail
        aux[COLS[h]] = fwd.astype(np.float32)
    return r[1 : n + 1], aux


def _fit(
    cols: dict[str, NDArray[np.float64]],
    aux: dict[str, NDArray[np.float32]],
    cfg: FeaturesConfig | None = None,
) -> ICStabilityFilter:
    names = tuple(cols)
    X = np.column_stack(list(cols.values())).astype(np.float32)
    return ICStabilityFilter(cfg or _cfg()).fit(X, _y(len(X)), _ctx(names, aux))


def test_decaying_signal_kept() -> None:
    nxt, aux = _market()
    sig = 0.3 * nxt + np.random.default_rng(1).normal(size=N)
    f = _fit({"sig": sig}, aux)
    assert f.output_names_ == ("sig",)
    icir = f.icir_by_horizon_["sig"]
    assert list(icir) == list(HORIZONS)
    assert icir[1] > icir[5] > icir[21] > 0.0
    assert f.p_values_["sig"] < 1e-10
    assert f.report_ is not None and f.report_.n_out == 1 and f.report_.dropped == {}


def test_pure_noise_kept_fraction_at_most_5pct() -> None:
    _, aux = _market(seed=2)
    X = np.random.default_rng(3).normal(size=(N, 500))
    cols = {f"n{j}": X[:, j] for j in range(500)}
    t0 = time.perf_counter()
    f = _fit(cols, aux)
    elapsed = time.perf_counter() - t0
    assert len(f.output_names_) / 500 <= 0.05
    assert f.report_ is not None and f.report_.n_in == 500
    assert f.report_.n_out == len(f.output_names_)
    assert elapsed < 5.0


def test_sign_flip_and_non_monotone_dropped() -> None:
    rng = np.random.default_rng(4)
    aux = {c: rng.normal(size=N).astype(np.float32) for c in COLS.values()}
    a1, a5, a10, a21 = (aux[COLS[h]].astype(np.float64) for h in HORIZONS)
    flip = -0.4 * a1 + 1.0 * a5 + rng.normal(size=N)  # h1 < 0, h5 >> 0, h10/h21 ~ 0
    # positive at every horizon (sign gate passes) but |ICIR| rises from h=1 to h=5
    hump = 0.3 * a1 + 1.0 * a5 + 0.2 * a10 + 0.1 * a21 + rng.normal(size=N)
    f = _fit({"flip": flip, "hump": hump}, aux)
    assert f.output_names_ == ()
    assert f.report_ is not None
    assert "sign" in f.report_.dropped["flip"] and "monotonic" in f.report_.dropped["flip"]
    assert "monotonic" in f.report_.dropped["hump"]
    assert "sign" not in f.report_.dropped["hump"]
    assert f.icir_by_horizon_["flip"][1] < -0.3 < 0.3 < f.icir_by_horizon_["flip"][5]


def test_icir_passes_but_fdr_fails_dropped() -> None:
    """Tiny-sample design: 2 windows of 63 rows (N=126, the minimum).

    ICIR = mean/std over only 2 window ICs, so two similar, modest window ICs (~0.15)
    give a large ICIR, while the full-Train Rank IC over 126 rows is not significant
    (p > 0.05). A single tested feature means BH accepts only if p <= 0.05, so the
    feature passes gate 1 (and gate 3, single horizon) but must be dropped by FDR.
    The preconditions are asserted so the test cannot pass vacuously.
    """
    n = 126
    rng = np.random.default_rng(5)
    lab = rng.normal(size=n)
    x = 0.15 * lab + rng.normal(size=n)
    for seed in range(6, 200):  # find a draw with similar window ICs but p > 0.05
        w = calc_rolling_ic(x.astype(np.float32), lab.astype(np.float32), 63)
        p = calc_rank_ic(x.astype(np.float32), lab.astype(np.float32))[1]
        if np.all(w > 0.05) and abs(w[0] - w[1]) < 0.1 and p > 0.05:
            break
        x = 0.15 * lab + np.random.default_rng(seed).normal(size=n)
    aux = {"fwd_ret_1": lab.astype(np.float32)}
    f = _fit({"weak": x}, aux, _cfg({1: "fwd_ret_1"}))
    assert f.icir_by_horizon_["weak"][1] > 0.3
    assert f.p_values_["weak"] > 0.05
    assert f.output_names_ == ()
    assert f.report_ is not None
    reason = f.report_.dropped["weak"]
    assert "fdr" in reason and "icir" not in reason and "monotonic" not in reason


def test_fdr_called_once_with_all_pvalues_including_degenerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[NDArray[np.float64]] = []

    def spy(p: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.bool_]:
        calls.append(np.asarray(p).copy())
        return apply_fdr_correction(p, alpha)

    monkeypatch.setattr(ic_stability, "apply_fdr_correction", spy)
    nxt, aux = _market(seed=6)
    sig = 0.3 * nxt + np.random.default_rng(7).normal(size=N)
    f = _fit({"sig": sig, "const": np.full(N, 2.0)}, aux)
    assert len(calls) == 1 and calls[0].shape == (2,)
    assert calls[0][1] == 1.0
    assert f.p_values_["const"] == 1.0
    assert f.output_names_ == ("sig",)
    assert f.report_ is not None and "fdr" in f.report_.dropped["const"]


def test_missing_aux_column_raises_naming_it() -> None:
    _, aux = _market()
    del aux["fwd_ret_10"]
    with pytest.raises(InvalidDataError, match="fwd_ret_10"):
        _fit({"a": np.random.default_rng(8).normal(size=N)}, aux)


def test_aux_length_mismatch_raises() -> None:
    _, aux = _market()
    aux["fwd_ret_5"] = aux["fwd_ret_5"][:-1]
    with pytest.raises(InvalidDataError, match="fwd_ret_5"):
        _fit({"a": np.random.default_rng(8).normal(size=N)}, aux)


def test_short_train_raises() -> None:
    n = 125  # < 2 * 63
    _, aux = _market(n=n)
    with pytest.raises(InvalidDataError, match="125"):
        _fit({"a": np.random.default_rng(9).normal(size=n)}, aux)


def test_aux_nan_tail_masked_pairwise() -> None:
    nxt, aux = _market(seed=10)
    sig = (0.3 * nxt + np.random.default_rng(11).normal(size=N)).astype(np.float32)
    f = _fit({"sig": sig.astype(np.float64)}, aux)
    lab = aux[COLS[1]]
    assert np.isnan(lab[-1])
    expected_p = calc_rank_ic(sig[:-1], lab[:-1])[1]
    assert f.p_values_["sig"] == pytest.approx(expected_p, rel=1e-9)
    lab21 = aux[COLS[21]]
    expected = calc_icir(calc_rolling_ic(sig, lab21, 63))
    assert np.isfinite(f.icir_by_horizon_["sig"][21])
    assert f.icir_by_horizon_["sig"][21] == pytest.approx(expected, rel=1e-9)


def test_rolling_rank_ic_matches_reference() -> None:
    rng = np.random.default_rng(12)
    n = 63 * 5 + 10
    X = rng.normal(size=(n, 4)).astype(np.float32)
    lab = (0.5 * X[:, 0] + rng.normal(size=n)).astype(np.float32)
    lab[-30:] = np.nan
    X[5:20, 1] = np.nan  # NaN in a feature
    X[63:126, 2] = 1.0  # constant window -> NaN window IC
    X[:, 3] = np.round(X[:, 3])  # heavy ties
    got = _rolling_rank_ic(X, lab, 63)
    for j in range(4):
        ref = calc_rolling_ic(X[:, j], lab, 63)
        np.testing.assert_allclose(got[:, j], ref, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_transform_selects_columns_without_reading_aux() -> None:
    class CountingDict(dict[str, NDArray[np.float32]]):
        reads = 0

        def __getitem__(self, key: str) -> NDArray[np.float32]:
            CountingDict.reads += 1
            return super().__getitem__(key)

    nxt, base = _market(seed=13)
    aux = CountingDict(base)
    sig = 0.3 * nxt + np.random.default_rng(14).normal(size=N)
    noise = np.random.default_rng(15).normal(size=N)
    X = np.column_stack([noise, sig]).astype(np.float32)
    f = ICStabilityFilter(_cfg()).fit(X, _y(N), _ctx(("noise", "sig"), aux))
    assert f.output_names_ == ("sig",)
    CountingDict.reads = 0
    aux.clear()
    Xv = np.random.default_rng(16).normal(size=(40, 2)).astype(np.float32)
    out = f.transform(Xv)
    assert CountingDict.reads == 0
    assert out.dtype == np.float32 and out.shape == (40, 1)
    np.testing.assert_array_equal(out[:, 0], Xv[:, 1])


def test_transform_before_fit_raises() -> None:
    with pytest.raises(FeatureSelectionError):
        ICStabilityFilter(_cfg()).transform(np.zeros((3, 2), dtype=np.float32))


def test_refit_leaves_no_stale_state() -> None:
    nxt, aux = _market(seed=17)
    sig = 0.3 * nxt + np.random.default_rng(18).normal(size=N)
    f = _fit({"a": sig, "b": np.random.default_rng(19).normal(size=N)}, aux)
    assert "b" in f.icir_by_horizon_
    X2 = np.column_stack([sig]).astype(np.float32)
    f.fit(X2, _y(N), _ctx(("c",), aux))
    assert set(f.icir_by_horizon_) == {"c"}
    assert set(f.p_values_) == {"c"}
    assert f.output_names_ == ("c",)
    assert f.report_ is not None and f.report_.dropped == {} and f.report_.n_in == 1
    assert f.transform(X2).shape == (N, 1)
