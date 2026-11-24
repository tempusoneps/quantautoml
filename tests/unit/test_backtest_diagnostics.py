"""Unit tests for read-only cross-regime bundle diagnostics (Phase 3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from bundle_helper import make_bundle

from quantautoml.backtest.diagnostics import (
    SegmentDiagnostic,
    diagnose_run_dir,
    evaluate_bundle_segments,
)
from quantautoml.common.config import BacktestConfig
from quantautoml.common.types import Segment

NAMES_WITH_OPEN = ["x0", "x1", "x2", "x3", "Open", "Close"]
NAMES_NO_OPEN = ["x0", "x1", "x2", "x3", "x4", "Close"]
CFG = BacktestConfig(rank_window=100, ic_window=200)


def _segment(n: int = 800, *, seed: int = 1, start: int = 0, with_y: bool = True) -> Segment:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    s = X[:, 0] + 0.5 * X[:, 1]
    y = ((s + 0.5 * rng.normal(size=n)) > np.median(s)).astype(np.int32) if with_y else None
    ret = 0.001 * np.tanh(X[:, 0]) + 0.0005 * rng.normal(size=n)
    close = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + ret[:-1])))
    open_px = close * (1.0 + 0.0002 * rng.normal(size=n))
    X[:, 4] = open_px.astype(np.float32)
    X[:, 5] = close.astype(np.float32)
    ts = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(start, start + n) * np.timedelta64(5, "m")
    return Segment(X=X, y=y, labels_aux={}, timestamps=ts, index=np.arange(start, start + n, dtype=np.int64))


def test_evaluate_bundle_segments_close_and_next_open_read_only(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "b", names=NAMES_WITH_OPEN)

    def snapshot() -> dict[str, str]:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.dir.iterdir()}

    before = snapshot()
    segments = {
        "cal_a": _segment(n=800, seed=1, start=0),
        "cal_b": _segment(n=800, seed=2, start=1000),
        "val": _segment(n=800, seed=3, start=2000),
        "oos": _segment(n=800, seed=4, start=3000),
    }
    res = evaluate_bundle_segments(bundle.dir, segments, NAMES_WITH_OPEN, CFG)

    assert set(res.keys()) == {"cal_a", "cal_b", "val", "oos"}
    for name, diag in res.items():
        assert isinstance(diag, SegmentDiagnostic)
        assert diag.segment == name
        assert diag.status == "ok"
        assert diag.n_rows == 800
        assert diag.close is not None and diag.close.status == "ok"
        assert diag.next_open is not None and diag.next_open.status == "ok"
        assert diag.close.n_bars == 799
        assert diag.next_open.n_bars == 798
        assert "latency" not in diag.close.quant_gates
        assert set(diag.close.quant_gates.keys()) == {
            "rank_ic",
            "icir",
            "net_sharpe",
            "mdd",
            "downside_ic",
        }
    assert snapshot() == before


def test_evaluate_bundle_segments_skips_missing_short_unlabeled_and_bad_prices(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "b", names=NAMES_NO_OPEN)
    bad_px = _segment(n=400, seed=5, start=0)
    X_bad = bad_px.X.copy()
    X_bad[10, 5] = 0.0  # non-positive Close
    bad_seg = Segment(X=X_bad, y=bad_px.y, labels_aux={}, timestamps=bad_px.timestamps, index=bad_px.index)

    segments = {
        "cal_a": None,
        "cal_b": _segment(n=50, seed=6, start=500),  # shorter than rank_window + 2
        "val": _segment(n=400, seed=7, start=1000, with_y=False),
        "bad": bad_seg,
        "oos": _segment(n=400, seed=8, start=2000),  # valid Close, missing Open
    }
    res = evaluate_bundle_segments(bundle.dir, segments, NAMES_NO_OPEN, CFG)

    assert res["cal_a"].status == "skipped"
    assert res["cal_b"].status == "skipped" and res["cal_b"].reason is not None
    assert res["val"].status == "skipped" and "label" in (res["val"].reason or "").lower()
    assert res["bad"].status == "skipped"
    assert res["oos"].status == "ok"
    assert res["oos"].close is not None and res["oos"].close.status == "ok"
    assert res["oos"].next_open is not None and res["oos"].next_open.status == "skipped"


def test_diagnose_run_dir_writes_json_and_falls_back_to_default_backtest_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_sample"
    bundle = make_bundle(run_dir / "05_packaging", names=NAMES_WITH_OPEN)
    assert bundle.dir == run_dir / "05_packaging" / "bundle"

    (run_dir / "01_data" / "segments").mkdir(parents=True, exist_ok=True)
    (run_dir / "01_data" / "02_task.json").write_text(
        json.dumps({"task_type": "binary_classification", "classes": [0, 1], "feature_names": NAMES_WITH_OPEN}),
        encoding="utf-8",
    )
    # config.toml without [backtest] -> falls back to BacktestConfig() (rank_window=2000)
    (run_dir / "config.toml").write_text(
        '[data]\nfeatures_path = "f.parquet"\n[split]\nhorizon_h = 5\n',
        encoding="utf-8",
    )
    seg_oos = _segment(n=2100, seed=9, start=0)
    np.savez(
        run_dir / "01_data" / "segments" / "oos.npz",
        X=seg_oos.X,
        y=seg_oos.y,
        timestamps=seg_oos.timestamps,
        index=seg_oos.index,
    )

    out_file = run_dir / "06_backtest" / "03_regime_diagnostics.json"
    res = diagnose_run_dir(run_dir)
    assert out_file.is_file()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["segments"]["oos"]["status"] == "ok"
    assert payload["segments"]["cal_a"]["status"] == "skipped"
    assert res["oos"].status == "ok"
