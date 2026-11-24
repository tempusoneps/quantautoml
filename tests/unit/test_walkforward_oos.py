"""Unit tests for Multi-Window Walk-Forward OOS runner (v1.1-fix2 Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantautoml.backtest.walkforward import (
    WalkForwardOOSReport,
    WalkForwardOOSRunner,
    WalkForwardWindowResult,
    generate_walkforward_windows,
)
from quantautoml.common.exceptions import DataLeakageError, InvalidDataError
from quantautoml.common.types import Dataset, IngestionReport, Segment, TaskType


def _dummy_dataset(n: int = 2000, p: int = 4) -> Dataset:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    ts = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(n) * np.timedelta64(5, "m")
    return Dataset(
        X=X,
        y=y,
        labels_aux={},
        timestamps=ts,
        feature_names=tuple(f"f{i}" for i in range(p)),
        selected_label="y",
        report=IngestionReport(
            n_feature_rows=n,
            n_label_rows=n,
            n_joined_rows=n,
            n_unmatched_dropped=0,
            n_trailing_label_nan_dropped=0,
            source_timezone="UTC",
            naive_utc_assumed=True,
            n_features=p,
            n_nan=0,
            n_inf=0,
            t_min=str(ts[0]),
            t_max=str(ts[-1]),
            label_nan_counts={"y": 0},
            max_trailing_nan=0,
            horizon_h=21,
        ),
        task_type=TaskType.BINARY_CLASSIFICATION,
        classes_=np.array([0, 1], dtype=np.int32),
    )


def test_generate_walkforward_windows_enforces_exact_horizon_purging() -> None:
    windows = generate_walkforward_windows(
        n_rows=10_000, n_windows=4, horizon_h=21, min_train_ratio=0.5, min_oos_bars=100
    )
    assert len(windows) == 4
    prev_end = 0
    for k, w in enumerate(windows):
        assert w.window_id == k
        # Exact purging/embargo gap of horizon_h rows between train_end and oos_start
        assert w.oos_start - w.train_end == 21
        assert w.oos_end > w.oos_start
        assert w.oos_start >= prev_end
        prev_end = w.oos_end
    assert windows[-1].oos_end == 10_000


def test_generate_walkforward_windows_rejects_invalid_inputs() -> None:
    with pytest.raises(InvalidDataError):
        generate_walkforward_windows(n_rows=1000, n_windows=1, horizon_h=21)
    with pytest.raises(InvalidDataError):
        generate_walkforward_windows(n_rows=1000, n_windows=3, horizon_h=0)
    with pytest.raises(InvalidDataError):
        generate_walkforward_windows(n_rows=100, n_windows=5, horizon_h=21, min_oos_bars=50)


def test_walkforward_runner_aggregates_windows_and_writes_report(tmp_path: Path) -> None:
    ds = _dummy_dataset(n=2000)
    seen_gaps: list[int] = []

    def fake_evaluator(window, prefix_ds: Dataset, oos_seg: Segment) -> WalkForwardWindowResult:
        # Verify zero-leakage: prefix_ds ends exactly horizon_h bars before oos_seg starts
        gap = int(oos_seg.index[0] - (len(prefix_ds.timestamps) - 1) - 1)
        seen_gaps.append(gap)
        sharpe = 1.5 if window.window_id % 2 == 0 else -0.2
        return WalkForwardWindowResult(
            window_id=window.window_id,
            train_n_rows=len(prefix_ds.timestamps),
            oos_n_rows=len(oos_seg.X),
            t_start=str(oos_seg.timestamps[0]),
            t_end=str(oos_seg.timestamps[-1]),
            status="ok",
            rank_ic=0.05 + 0.01 * window.window_id,
            icir=0.50,
            gross_sharpe=sharpe + 0.3,
            net_sharpe=sharpe,
            daily_net_sharpe=sharpe,
            max_drawdown=0.08 + 0.02 * window.window_id,
            mean_turnover=0.02,
            downside_ic=0.02,
            accepted=sharpe > 1.2,
        )

    out_path = tmp_path / "walkforward_oos_report.json"
    runner = WalkForwardOOSRunner(n_windows=4, horizon_h=21, min_train_ratio=0.5, min_oos_bars=50)
    report = runner.run(ds, window_evaluator=fake_evaluator, out_path=out_path)

    assert isinstance(report, WalkForwardOOSReport)
    assert seen_gaps == [21, 21, 21, 21]
    assert report.n_windows == 4
    assert report.n_completed == 4
    assert report.n_accepted == 2
    assert report.acceptance_rate == pytest.approx(0.5)
    assert report.positive_sharpe_ratio == pytest.approx(0.5)
    assert report.worst_mdd == pytest.approx(0.14)
    assert out_path.is_file()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["n_windows"] == 4 and len(saved["windows"]) == 4


def test_walkforward_runner_rejects_overlapping_timestamps() -> None:
    ds = _dummy_dataset(n=1000)
    bad_ts = ds.timestamps.copy()
    bad_ts[600] = bad_ts[400]  # non-monotonic timestamp violates RULE-01
    bad_ds = Dataset(
        X=ds.X,
        y=ds.y,
        labels_aux={},
        timestamps=bad_ts,
        feature_names=ds.feature_names,
        selected_label=ds.selected_label,
        report=ds.report,
        task_type=ds.task_type,
        classes_=ds.classes_,
    )
    runner = WalkForwardOOSRunner(n_windows=3, horizon_h=21, min_train_ratio=0.5, min_oos_bars=50)
    with pytest.raises(DataLeakageError):
        runner.run(bad_ds, window_evaluator=lambda w, p, o: WalkForwardWindowResult(
            window_id=w.window_id, train_n_rows=0, oos_n_rows=0, t_start="", t_end="", status="skipped"
        ))
