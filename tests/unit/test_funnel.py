"""FeatureFunnel: F1 -> F2 -> F3+7 -> F4 -> F5 -> F6 -> F8 chaining, fit on Train only."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import DataLeakageError, FeatureSelectionError, InvalidDataError
from quantautoml.common.types import Dataset, IngestionReport, SplitResult, TaskType
from quantautoml.data.splitter import AdaptiveTimeSeriesSplitter
from quantautoml.features import FeatureFunnel
from quantautoml.features.base import BaseFeatureFilter, FitContext

STAGES = (
    "f1_triage",
    "f2_variance",
    "f3_ic_stability",
    "f4_collinearity",
    "f5_clustering",
    "f6_regime_synergy",
    "f8_jmi",
)
H = 21


def _cfg() -> FeaturesConfig:
    return FeaturesConfig(
        ic_horizons={1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21"}, target_horizon=1
    )


def _report(
    n: int, p: int, bools: list[str], strings: dict[str, dict[str, int]]
) -> IngestionReport:
    return IngestionReport(
        n_feature_rows=n, n_label_rows=n, n_joined_rows=n, n_unmatched_dropped=0,
        n_trailing_label_nan_dropped=0, source_timezone="UTC", naive_utc_assumed=False,
        n_features=p, n_nan=0, n_inf=0, t_min="", t_max="", label_nan_counts={},
        max_trailing_nan=H, horizon_h=H, bool_columns_cast=bools,
        string_columns_encoded=strings,
    )


def _fwd(r: NDArray[np.float64], k: int) -> NDArray[np.float32]:
    """Forward k-bar return at t = r[t+1] + ... + r[t+k] (``r[t+1]`` realized at t+1)."""
    c = np.concatenate(([0.0], np.cumsum(r)))
    out = np.full(r.size, np.nan)
    n = r.size
    out[: n - k] = c[1 + k : n + 1] - c[1 : n + 1 - k]
    return out.astype(np.float32)


def _dataset(n: int = 3000, seed: int = 0, constant: bool = False) -> Dataset:
    """Signal ``sig`` drives the next 1-bar return; ``miss`` has MNAR NaN (on y == 1)."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = 0.5 * sig[:-1] + rng.normal(size=n - 1)
    labels = {f"fwd_ret_{k}": _fwd(r, k) for k in (1, 5, 21)}
    y = (np.nan_to_num(labels["fwd_ret_1"]) > 0).astype(np.int32)
    miss = rng.normal(size=n)
    miss[(y == 1) & (rng.random(n) < 0.4)] = np.nan
    cols = {
        "sig": sig,
        "sig_dup": sig + 0.01 * rng.normal(size=n),
        "noise0": rng.normal(size=n),
        "noise1": rng.normal(size=n),
        "miss": miss,
        "flag": (rng.random(n) < 0.3).astype(float),
        "cat": rng.integers(0, 4, n).astype(float),
    }
    names = tuple(cols)
    X = np.column_stack([cols[c] for c in names]).astype(np.float32)
    if constant:
        X = np.ones_like(X)
    ts = np.datetime64("2024-01-01T00:00:00", "ns") + np.arange(n) * np.timedelta64(1, "h")
    return Dataset(
        X=X, y=y, labels_aux=labels, timestamps=ts, feature_names=names,
        selected_label="y_dir", report=_report(n, len(names), ["flag"], {"cat": {"a": 0}}),
        task_type=TaskType.BINARY_CLASSIFICATION, classes_=np.array([0, 1]),
    )


def _split(ds: Dataset) -> SplitResult:
    return AdaptiveTimeSeriesSplitter().split(ds, H)


@pytest.fixture(scope="module")
def fitted() -> tuple[FeatureFunnel, SplitResult, Dataset]:
    ds = _dataset()
    split = _split(ds)
    return FeatureFunnel(_cfg()).fit(split, ds), split, ds


def test_reports_cover_seven_stages_in_order(fitted) -> None:
    funnel, _, ds = fitted
    assert tuple(r.stage for r in funnel.reports_) == STAGES
    assert funnel.reports_[0].n_in == len(ds.feature_names)


def test_report_counts_chain_including_f1_indicators(fitted) -> None:
    funnel, _, _ = fitted
    reps = funnel.reports_
    assert reps[0].n_out == reps[0].n_in + 1  # is_nan_miss (MNAR) appended by F1
    for a, b in zip(reps, reps[1:]):
        assert a.n_out == b.n_in
    for r in reps[1:]:
        assert r.n_out <= r.n_in
    assert reps[-1].n_out == len(funnel.selected_features_)


def test_signal_kept_duplicate_and_noise_dropped(fitted) -> None:
    funnel, _, _ = fitted
    sel = funnel.selected_features_
    assert ("sig" in sel) != ("sig_dup" in sel)  # F4 keeps exactly one of the pair
    assert "noise0" not in sel and "noise1" not in sel
    for r in funnel.reports_:
        assert set(r.dropped).isdisjoint(sel)


def test_transform_all_segments_same_columns_float32(fitted) -> None:
    funnel, split, _ = fitted
    k = len(funnel.selected_features_)
    assert k >= 1
    for seg in (split.train, split.val, split.oos):
        out = funnel.transform(seg)
        assert out is not seg
        assert out.X.dtype == np.float32
        assert out.X.shape == (seg.X.shape[0], k)
        assert out.y is seg.y and out.index is seg.index and out.timestamps is seg.timestamps
        assert out.labels_aux is seg.labels_aux


def test_transform_cal_segments_in_static_mode() -> None:
    ds = _dataset(n=12_000, seed=1)
    split = _split(ds)
    assert split.cal_a is not None and split.cal_b is not None
    funnel = FeatureFunnel(_cfg()).fit(split, ds)
    widths = {
        funnel.transform(s).X.shape[1]
        for s in (split.train, split.val, split.oos, split.cal_a, split.cal_b)
    }
    assert widths == {len(funnel.selected_features_)}
    assert funnel.transform(split.cal_a).X.dtype == np.float32


def test_context_labels_sliced_to_train_and_discrete_propagated(monkeypatch) -> None:
    seen: list[tuple[str, int, FitContext]] = []
    orig = BaseFeatureFilter.fit

    def spy(self: BaseFeatureFilter, X, y, ctx):  # type: ignore[no-untyped-def]
        seen.append((self.name, len(X), ctx))
        return orig(self, X, y, ctx)

    monkeypatch.setattr(BaseFeatureFilter, "fit", spy)
    ds = _dataset()
    split = _split(ds)
    FeatureFunnel(_cfg()).fit(split, ds)
    n_train = len(split.train.X)
    assert [s[0] for s in seen] == list(STAGES)
    for _, n, ctx in seen:
        assert n == n_train
        assert all(v.shape == (n_train,) for v in ctx.labels_aux.values())
        assert set(ctx.discrete_features) <= set(ctx.feature_names)
    f1_ctx, f2_ctx = seen[0][2], seen[1][2]
    assert f1_ctx.feature_names == ds.feature_names
    assert set(f1_ctx.discrete_features) == {"flag", "cat"}
    assert "is_nan_miss" in f2_ctx.feature_names
    assert "is_nan_miss" in f2_ctx.discrete_features


def test_empty_stage_passes_through_remaining_stages() -> None:
    ds = _dataset(constant=True)
    split = _split(ds)
    funnel = FeatureFunnel(_cfg()).fit(split, ds)
    assert funnel.selected_features_ == ()
    assert len(funnel.reports_) == 7
    assert funnel.reports_[1].n_out == 0  # F2 drops every constant column
    for r in funnel.reports_[2:]:
        assert (r.n_in, r.n_out) == (0, 0)
        assert r.notes == ("skipped: no features left",)
    out = funnel.transform(split.val)
    assert out.X.shape == (len(split.val.X), 0)
    assert out.X.dtype == np.float32


def test_refit_rebuilds_state(fitted) -> None:
    _, split, ds = fitted
    funnel = FeatureFunnel(_cfg()).fit(split, ds)
    first = funnel.selected_features_
    assert first
    const = _dataset(constant=True)
    funnel.fit(_split(const), const)
    assert funnel.selected_features_ == ()
    assert len(funnel.reports_) == 7


def test_unsupervised_raises() -> None:
    ds = _dataset()
    split = _split(ds)
    with pytest.raises(InvalidDataError):
        FeatureFunnel(_cfg()).fit(split, replace(ds, task_type=TaskType.UNSUPERVISED))


def test_train_without_y_raises() -> None:
    ds = _dataset()
    split = _split(ds)
    bad = replace(split, train=replace(split.train, y=None))
    with pytest.raises(InvalidDataError):
        FeatureFunnel(_cfg()).fit(bad, ds)


def test_non_monotone_train_index_raises() -> None:
    ds = _dataset()
    split = _split(ds)
    idx = split.train.index.copy()
    idx[[0, 1]] = idx[[1, 0]]
    with pytest.raises(DataLeakageError):
        FeatureFunnel(_cfg()).fit(replace(split, train=replace(split.train, index=idx)), ds)


def test_train_index_overlapping_val_raises() -> None:
    ds = _dataset()
    split = _split(ds)
    val_idx = split.val.index.copy()
    val_idx[0] = split.train.index[-1]
    with pytest.raises(DataLeakageError):
        FeatureFunnel(_cfg()).fit(replace(split, val=replace(split.val, index=val_idx)), ds)


def test_transform_before_fit_raises() -> None:
    ds = _dataset()
    with pytest.raises(FeatureSelectionError):
        FeatureFunnel(_cfg()).transform(_split(ds).val)


def test_transform_wrong_width_raises(fitted) -> None:
    funnel, split, _ = fitted
    with pytest.raises(FeatureSelectionError):
        funnel.transform(replace(split.val, X=split.val.X[:, :3]))


def test_f6_thresholds_passed_from_cfg(fitted) -> None:
    _, split, ds = fitted
    cfg = FeaturesConfig(
        ic_horizons=_cfg().ic_horizons, target_horizon=1, mi_floor=10.0, synergy_delta=10.0
    )
    funnel = FeatureFunnel(cfg).fit(split, ds)
    f6, f8 = funnel.reports_[5], funnel.reports_[6]
    assert f6.n_in >= 1 and f6.n_out == 0
    assert all("mi_floor 10" in why for why in f6.dropped.values())
    assert (f8.n_in, f8.n_out) == (0, 0) and f8.notes == ("skipped: no features left",)
    assert funnel.selected_features_ == ()
