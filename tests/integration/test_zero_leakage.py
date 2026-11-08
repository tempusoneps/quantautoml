"""End-to-end Phase 1: TOML + files -> DataIngestion -> SplitResult, proving no look-ahead.

Leakage is verified against the data itself: labels are ``shift(-k)`` of a price
path, so for any segment A followed by segment B, recomputing A's labels with every
price from B's first row onwards masked out must leave A's labels unchanged.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantautoml.common.config import load_config
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import SplitMode, SplitResult
from quantautoml.data.ingestion import DataIngestion
from quantautoml.data.splitter import AdaptiveTimeSeriesSplitter

H = 21
SELECTED_HORIZON = 5  # y_dir is derived from fwd_ret_5 -> 5 trailing NaN rows are dropped


def _fwd(logp: np.ndarray, k: int) -> np.ndarray:
    out = np.full(logp.size, np.nan)
    out[: logp.size - k] = logp[k:] - logp[: logp.size - k]
    return out


def _write_case(tmp_path: Path, n_dataset_rows: int, horizon_h: int = H) -> tuple[Path, np.ndarray]:
    n_raw = n_dataset_rows + SELECTED_HORIZON
    rng = np.random.default_rng(42)
    logp = np.cumsum(rng.normal(0, 0.01, n_raw))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = pl.datetime_range(
        start, start + timedelta(hours=n_raw - 1), interval="1h",
        time_unit="ns", time_zone="UTC", eager=True,
    ).alias("timestamp")
    fwd5 = _fwd(logp, 5)
    y_dir = np.where(np.isnan(fwd5), np.nan, (fwd5 > 0).astype(float))
    pl.DataFrame([ts, pl.Series("f0", rng.normal(size=n_raw)), pl.Series("f1", rng.normal(size=n_raw))]
                 ).write_parquet(tmp_path / "features.parquet")
    pl.DataFrame(
        [ts, pl.Series("y_dir", y_dir)]
        + [pl.Series(f"fwd_ret_{k}", _fwd(logp, k)) for k in (1, 5, 21)]
    ).write_parquet(tmp_path / "labels.parquet")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'[data]\nfeatures_path = "{tmp_path / "features.parquet"}"\n'
        f'labels_path = "{tmp_path / "labels.parquet"}"\n'
        f'[labels]\nselected = "y_dir"\n[split]\nhorizon_h = {horizon_h}\n',
        encoding="utf-8",
    )
    return cfg, logp


def _run(cfg_path: Path) -> SplitResult:
    cfg = load_config(cfg_path)
    ds = DataIngestion(cfg).load()
    return AdaptiveTimeSeriesSplitter().split(ds, cfg.split.horizon_h)


def _labels_unchanged_when_future_masked(
    logp: np.ndarray, seg_aux: np.ndarray, seg_index: np.ndarray, first_future_row: int
) -> bool:
    masked = logp.copy()
    masked[first_future_row:] = np.nan
    recomputed = _fwd(masked, H)[seg_index].astype(np.float32)  # ingestion stores float32
    return bool(np.array_equal(recomputed, seg_aux, equal_nan=True))


@pytest.mark.parametrize(
    ("n", "mode"),
    [(15_000, SplitMode.STATIC_5WAY), (3_000, SplitMode.OOF_ASSISTED),
     (10_000, SplitMode.STATIC_5WAY), (9_999, SplitMode.OOF_ASSISTED)],
)
def test_no_segment_label_depends_on_a_later_segment(tmp_path: Path, n: int, mode: SplitMode) -> None:
    cfg, logp = _write_case(tmp_path, n)
    res = _run(cfg)
    assert res.mode is mode
    parts = [p for p in (res.train, res.cal_a, res.cal_b, res.val, res.oos) if p is not None]

    for a, b in zip(parts, parts[1:]):
        assert b.index[0] - a.index[-1] - 1 == H
        assert a.timestamps[-1] < b.timestamps[0]
        assert _labels_unchanged_when_future_masked(
            logp, a.labels_aux["fwd_ret_21"], a.index, int(b.index[0])
        ), "labels of an earlier segment depend on prices of a later segment"
        # negative control: without the purge gap the same check must detect leakage
        assert not _labels_unchanged_when_future_masked(
            logp, a.labels_aux["fwd_ret_21"], a.index, int(a.index[-1]) + 1
        ), "control failed: the leak check cannot detect forward-looking labels"
    for p in parts:
        assert p.X.dtype == np.float32
        assert all(v.dtype == np.float32 for v in p.labels_aux.values())


@pytest.mark.parametrize("n", [3_000, 9_999])
def test_oof_folds_do_not_leak_validation_prices_into_training_labels(tmp_path: Path, n: int) -> None:
    cfg, logp = _write_case(tmp_path, n)
    res = _run(cfg)
    assert res.oof_folds is not None and len(res.oof_folds) == 5
    aux = {i: v for i, v in zip(res.train.index, res.train.labels_aux["fwd_ret_21"])}
    for tr, va in res.oof_folds:
        tr_labels = np.array([aux[i] for i in tr])
        assert _labels_unchanged_when_future_masked(logp, tr_labels, tr, int(va[0]))
        assert va[0] - tr[-1] - 1 == H


def test_training_segment_never_contains_a_later_timestamp(tmp_path: Path) -> None:
    cfg, _ = _write_case(tmp_path, 15_000)
    res = _run(cfg)
    later = [res.cal_a, res.cal_b, res.val, res.oos]
    assert all(res.train.timestamps.max() < s.timestamps.min() for s in later if s is not None)
    assert res.report.warnings == ()


def test_horizon_smaller_than_longest_label_is_blocked_before_splitting(tmp_path: Path) -> None:
    cfg, _ = _write_case(tmp_path, 3_000, horizon_h=5)
    with pytest.raises(InvalidDataError, match="horizon_h"):
        _run(cfg)


def test_reports_are_json_serialisable(tmp_path: Path) -> None:
    import json

    cfg, _ = _write_case(tmp_path, 3_000)
    res = _run(cfg)
    json.dumps(res.report.to_dict())
    ds = DataIngestion(load_config(cfg)).load()
    json.dumps(ds.report.to_dict())
