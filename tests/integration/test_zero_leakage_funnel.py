"""End-to-end Phase 2: files -> DataIngestion -> split -> FeatureFunnel, proving no look-ahead.

Synthetic market (deterministic, fixed seeds): ``n_sig`` independent N(0, 1) signals
``s_i`` drive the next 1-bar log return ``r[t+1] = 0.01 * (sum_i a_i s_i[t] + e[t+1])`` with
``e ~ N(0, 1)`` and ``|a_i| = 0.4045`` (alternating sign), so with 5 signals each one has
corr(s_i, r) = 0.30. Labels are forward returns ``fwd_ret_k[t] = logp[t+k] - logp[t]`` for
k in {1, 5, 10, 21} and ``y_dir = 1[fwd_ret_1 > 0]`` (selected label).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig, PipelineConfig
from quantautoml.common.exceptions import DataLeakageError
from quantautoml.common.types import Dataset, Segment, SplitMode, SplitResult
from quantautoml.data.ingestion import DataIngestion
from quantautoml.data.splitter import AdaptiveTimeSeriesSplitter
from quantautoml.features import FeatureFunnel
from quantautoml.features.base import BaseFeatureFilter, FitContext

H = 21
HORIZONS = (1, 5, 10, 21)
SIGNAL_LOADING = 0.4045  # corr(s_i, r) = a / sqrt(5 a^2 + 1) = 0.30 with 5 signals
STAGES = (
    "f1_triage",
    "f2_variance",
    "f3_ic_stability",
    "f4_collinearity",
    "f5_clustering",
    "f6_regime_synergy",
    "f8_jmi",
)


@dataclass(frozen=True)
class Case:
    cfg: PipelineConfig
    dataset: Dataset
    split: SplitResult

    @property
    def features(self) -> FeaturesConfig:
        assert self.cfg.features is not None
        return self.cfg.features


def _write_case(
    root: Path,
    n: int,
    columns: dict[str, NDArray[np.float64]],
    signals: NDArray[np.float64],
    rng: np.random.Generator,
) -> PipelineConfig:
    """Write features/labels parquet (``n + 1`` raw rows; ``y_dir`` drops 1 trailing NaN)."""
    n_raw = n + 1
    loadings = SIGNAL_LOADING * np.where(np.arange(signals.shape[1]) % 2 == 0, 1.0, -1.0)
    r = np.zeros(n_raw)
    r[1:] = 0.01 * (signals[:-1] @ loadings + rng.normal(size=n_raw - 1))
    logp = np.cumsum(r)
    fwd: dict[int, NDArray[np.float64]] = {}
    for k in HORIZONS:
        f = np.full(n_raw, np.nan)
        f[: n_raw - k] = logp[k:] - logp[: n_raw - k]
        fwd[k] = f
    y_dir = np.where(np.isnan(fwd[1]), np.nan, (fwd[1] > 0).astype(float))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = pl.datetime_range(
        start, start + timedelta(hours=n_raw - 1), interval="1h",
        time_unit="ns", time_zone="UTC", eager=True,
    ).alias("timestamp")
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([ts, *[pl.Series(k, v) for k, v in columns.items()]]).write_parquet(
        root / "features.parquet"
    )
    pl.DataFrame(
        [ts, pl.Series("y_dir", y_dir), *[pl.Series(f"fwd_ret_{k}", fwd[k]) for k in HORIZONS]]
    ).write_parquet(root / "labels.parquet")
    return PipelineConfig.model_validate(
        {
            "data": {
                "features_path": str(root / "features.parquet"),
                "labels_path": str(root / "labels.parquet"),
            },
            "labels": {"selected": "y_dir"},
            "split": {"horizon_h": H},
            "features": {
                "ic_horizons": {k: f"fwd_ret_{k}" for k in HORIZONS},
                "target_horizon": 1,
            },
        }
    )


def _load(cfg: PipelineConfig) -> Case:
    ds = DataIngestion(cfg).load()
    return Case(cfg, ds, AdaptiveTimeSeriesSplitter().split(ds, H))


def _signal_case(root: Path, n: int, n_noise: int, seed: int, with_mnar: bool = False) -> Case:
    rng = np.random.default_rng(seed)
    signals = rng.normal(size=(n + 1, 5))
    cols = {f"sig{i}": signals[:, i] for i in range(5)}
    cols |= {f"noise{j:03d}": rng.normal(size=n + 1) for j in range(n_noise)}
    if with_mnar:
        # NaN where the *current* row's 1-bar-ahead signal sum is high -> F1 types it MNAR
        # and appends is_nan_miss, so the funnel width changes after F1.
        miss = rng.normal(size=n + 1)
        high = signals @ np.array([1.0, -1.0, 1.0, -1.0, 1.0]) > 1.0
        miss[high & (rng.random(n + 1) < 0.5)] = np.nan
        cols["miss"] = miss
    return _load(_write_case(root, n, cols, signals, rng))


@pytest.fixture(scope="module")
def small_case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    """15,000 rows: 5 signals + 25 noise + 1 MNAR column (static 5-way split)."""
    return _signal_case(tmp_path_factory.mktemp("small"), 15_000, 25, seed=7, with_mnar=True)


@pytest.fixture(scope="module")
def small_funnel(small_case: Case) -> FeatureFunnel:
    return FeatureFunnel(small_case.features).fit(small_case.split, small_case.dataset)


def _noise_segment(seg: Segment, rng: np.random.Generator) -> Segment:
    y = None if seg.y is None else rng.integers(0, 2, seg.y.size).astype(np.int32)
    return replace(
        seg,
        X=rng.normal(size=seg.X.shape).astype(np.float32),
        y=y,
        labels_aux={
            k: rng.normal(size=v.shape).astype(np.float32) for k, v in seg.labels_aux.items()
        },
    )


# --------------------------------------------------------------------------- #
# (a) behaviour: non-Train data cannot influence the selection
# --------------------------------------------------------------------------- #
def test_noise_in_val_oos_cal_does_not_change_selection(
    small_case: Case, small_funnel: FeatureFunnel
) -> None:
    split, ds = small_case.split, small_case.dataset
    assert split.mode is SplitMode.STATIC_5WAY
    assert split.cal_a is not None and split.cal_b is not None
    assert small_funnel.selected_features_  # non-trivial selection to compare against
    rng = np.random.default_rng(123)
    noisy_split = replace(
        split,
        val=_noise_segment(split.val, rng),
        oos=_noise_segment(split.oos, rng),
        cal_a=_noise_segment(split.cal_a, rng),
        cal_b=_noise_segment(split.cal_b, rng),
    )
    # Also corrupt every non-Train row of the full Dataset arrays.
    outside = np.setdiff1d(np.arange(len(ds.X)), split.train.index)
    X = ds.X.copy()
    X[outside] = rng.normal(size=(outside.size, X.shape[1])).astype(np.float32)
    aux = {k: v.copy() for k, v in ds.labels_aux.items()}
    for v in aux.values():
        v[outside] = rng.normal(size=outside.size).astype(np.float32)
    noisy_ds = replace(ds, X=X, labels_aux=aux)

    refit = FeatureFunnel(small_case.features).fit(noisy_split, noisy_ds)

    assert refit.selected_features_ == small_funnel.selected_features_
    assert [(r.n_in, r.n_out) for r in refit.reports_] == [
        (r.n_in, r.n_out) for r in small_funnel.reports_
    ]


# --------------------------------------------------------------------------- #
# (b) spy: every stage is fitted on exactly the Train rows (RULE-01)
# --------------------------------------------------------------------------- #
def test_every_stage_fits_on_train_rows_only(
    small_case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, NDArray[np.float32], FitContext]] = []
    orig = BaseFeatureFilter.fit

    def spy(
        self: BaseFeatureFilter, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext
    ) -> BaseFeatureFilter:
        calls.append((self.name, X, ctx))
        return orig(self, X, y, ctx)

    monkeypatch.setattr(BaseFeatureFilter, "fit", spy)
    split = small_case.split
    funnel = FeatureFunnel(small_case.features).fit(split, small_case.dataset)

    n_train = len(split.train.X)
    assert [name for name, _, _ in calls] == list(STAGES)
    assert np.array_equal(calls[0][1], split.train.X, equal_nan=True)  # funnel input = Train X
    for name, X, ctx in calls:
        assert len(X) == n_train, name
        assert X.dtype == np.float32, name
        for col, arr in ctx.labels_aux.items():
            assert arr.shape == (n_train,), (name, col)
            assert np.array_equal(arr, split.train.labels_aux[col], equal_nan=True)
    assert funnel.reports_[0].n_out == funnel.reports_[0].n_in + 1  # is_nan_miss appended
    assert "is_nan_miss" in calls[1][2].discrete_features


# --------------------------------------------------------------------------- #
# (c) index-level leakage guards
# --------------------------------------------------------------------------- #
def test_non_monotone_train_index_raises(small_case: Case) -> None:
    split = small_case.split
    idx = split.train.index[::-1].copy()
    bad = replace(split, train=replace(split.train, index=idx))
    with pytest.raises(DataLeakageError):
        FeatureFunnel(small_case.features).fit(bad, small_case.dataset)


@pytest.mark.parametrize("segment", ["val", "oos", "cal_a", "cal_b"])
def test_train_index_overlap_raises(small_case: Case, segment: str) -> None:
    split = small_case.split
    seg = getattr(split, segment)
    idx = seg.index.copy()
    idx[0] = split.train.index[-1]
    bad = replace(split, **{segment: replace(seg, index=idx)})
    with pytest.raises(DataLeakageError):
        FeatureFunnel(small_case.features).fit(bad, small_case.dataset)


# --------------------------------------------------------------------------- #
# (d) BH-FDR end-to-end: 300 noise + 5 real signals
# --------------------------------------------------------------------------- #
def test_bh_fdr_end_to_end_noise_rejected_signals_kept(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """300 iid N(0, 1) noise columns + the 5 signals of the module docstring (seed 2026).

    Why each signal survives every stage (Train = 9,000 rows, 142 windows of 63):
      * F1/F2: no NaN, continuous N(0, 1) -> kept.
      * F3+7: Rank IC(h) ~= 0.30 / sqrt(h) with the same sign at every horizon (s_i[t]
        only predicts r[t+1], which is inside every fwd_ret_h[t]), so
        |ICIR(1)| ~= 2.3 >> 0.30, p ~ 0 passes BH, and |ICIR| decays 1 > 5 > 10 > 21
        with gaps (>= 0.15) well above sampling noise -> monotonic.
      * F4: the signals are mutually independent -> dCor ~ 0 << 0.85.
      * F5: their rolling-IC series are nearly uncorrelated -> distance ~ 1 > 0.30,
        each signal is its own cluster.
      * F6: MI(5-bin s_i, y_dir) ~ 0.02 nats in every regime (signals are iid,
        independent of the past-return regime) > mi_floor 0.01 + chi2 null.
      * F8: <= 30 survivors -> all kept.
    Noise columns have |ICIR| ~ N(0, 0.084): rejected by the ICIR/BH gates.
    """
    case = _signal_case(tmp_path_factory.mktemp("bh"), 15_000, 300, seed=2026)
    funnel = FeatureFunnel(case.features).fit(case.split, case.dataset)
    selected = set(funnel.selected_features_)
    noise_survivors = sorted(n for n in selected if n.startswith("noise"))
    signals_kept = sorted(n for n in selected if n.startswith("sig"))
    print(f"\n[BH e2e] noise survivors={len(noise_survivors)}/300, signals={signals_kept}")
    assert len(noise_survivors) <= 0.05 * 300
    assert len(signals_kept) >= 4


# --------------------------------------------------------------------------- #
# (e) scalability: N = 20,000, 60 features through the full funnel
# --------------------------------------------------------------------------- #
def test_full_funnel_scales_to_20k_rows(tmp_path_factory: pytest.TempPathFactory) -> None:
    """60 columns = 6 noisy copies (noise scale 0.15..1.5) of each of the 5 signals plus
    30 noise columns, so ~30 columns pass F3 and F4 / F5 really evaluate dCor pairs and
    clustering on 12,000 Train rows (no N x N matrix)."""
    n = 20_000
    rng = np.random.default_rng(11)
    signals = rng.normal(size=(n + 1, 5))
    scales = np.linspace(0.15, 1.5, 6)
    cols = {
        f"sig{i}_c{c}": signals[:, i] + scales[c] * rng.normal(size=n + 1)
        for i in range(5)
        for c in range(6)
    }
    cols |= {f"noise{j:02d}": rng.normal(size=n + 1) for j in range(30)}
    case = _load(_write_case(tmp_path_factory.mktemp("scale"), n, cols, signals, rng))
    assert len(case.split.train.X) == 12_000

    t0 = time.perf_counter()
    funnel = FeatureFunnel(case.features).fit(case.split, case.dataset)
    out = funnel.transform(case.split.oos)
    elapsed = time.perf_counter() - t0
    counts = [(r.stage, r.n_in, r.n_out) for r in funnel.reports_]
    print(f"\n[scale] N=20,000 x 60 funnel fit+transform: {elapsed:.2f}s; {counts}")

    assert elapsed < 120.0
    assert funnel.reports_[3].n_in > 5  # F4 had real dCor work to do
    assert out.X.dtype == np.float32
    assert out.X.shape == (len(case.split.oos.X), len(funnel.selected_features_))
