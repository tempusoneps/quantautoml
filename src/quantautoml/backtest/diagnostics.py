"""Read-only cross-regime diagnostics for a packaged ONNX model bundle (RULE-01).

Never fits any model or preprocessor and never mutates bundle files. Evaluates a
frozen bundle across chronological segments (e.g., ``cal_a``, ``cal_b``, ``val``, ``oos``)
under both ``close`` and ``next_open`` execution assumptions.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quantautoml.backtest.buffer import HysteresisBuffer
from quantautoml.backtest.runner import OOSBacktestRunner, evaluate_segment_execution
from quantautoml.common.config import BacktestConfig, load_config
from quantautoml.common.exceptions import BacktestError
from quantautoml.common.types import Segment
from quantautoml.packaging.predictor_loader import load_predictor
from quantautoml.packaging.schema import FeatureSchemaContract


class ExecutionDiagnostic(BaseModel):
    """Per-execution-mode diagnostic summary (excludes latency gate)."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "skipped"]
    reason: str | None = None
    rank_ic: float | None = None
    icir: float | None = None
    gross_sharpe: float | None = None
    net_sharpe: float | None = None
    daily_net_sharpe: float | None = None
    deflated_sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    downside_ic: float | None = None
    n_stress_bars: int | None = None
    mean_turnover: float | None = None
    n_bars: int | None = None
    periods_per_year: float | None = None
    n_overnight_gaps: int | None = None
    quant_gates: dict[str, bool] = Field(default_factory=dict)
    quant_accepted: bool | None = None


class SegmentDiagnostic(BaseModel):
    """Cross-regime diagnostic summary for a single chronological segment."""

    model_config = ConfigDict(frozen=True)

    segment: str
    status: Literal["ok", "skipped"]
    reason: str | None = None
    n_rows: int = 0
    close: ExecutionDiagnostic | None = None
    next_open: ExecutionDiagnostic | None = None


def evaluate_bundle_segments(
    bundle_dir: Path,
    segments: Mapping[str, Segment | None],
    feature_names: Sequence[str],
    cfg: BacktestConfig,
) -> dict[str, SegmentDiagnostic]:
    """Evaluate a frozen ONNX bundle across segments (read-only, zero-leakage)."""
    bdir = Path(bundle_dir)
    schema = FeatureSchemaContract.from_json(
        (bdir / "features_schema.json").read_text(encoding="utf-8")
    )
    predictor = load_predictor(bdir / "predictor.py").Predictor(bdir)
    buf = HysteresisBuffer(
        cfg.top_pct,
        cfg.exit_pct,
        cfg.rank_window,
        signal_ema_span=cfg.signal_ema_span,
        min_hold_bars=cfg.min_hold_bars,
    )

    results: dict[str, SegmentDiagnostic] = {}
    for seg_name, seg in segments.items():
        if seg is None:
            results[seg_name] = SegmentDiagnostic(
                segment=seg_name, status="skipped", reason="segment not available"
            )
            continue
        n = len(seg.X)
        if seg.y is None:
            results[seg_name] = SegmentDiagnostic(
                segment=seg_name, status="skipped", reason="segment has no labels", n_rows=n
            )
            continue
        if n < cfg.rank_window + 2:
            results[seg_name] = SegmentDiagnostic(
                segment=seg_name,
                status="skipped",
                reason=f"segment has {n} bars, need >= rank_window + 2 = {cfg.rank_window + 2}",
                n_rows=n,
            )
            continue

        schema.check_input(seg.X, names=feature_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = np.asarray(predictor.predict(seg.X), dtype=np.float32)
        positions = buf.positions(scores)

        close_diag = _eval_mode(
            scores=scores,
            positions=positions,
            seg=seg,
            feature_names=feature_names,
            execution_price="close",
            cfg=cfg,
        )
        open_diag = _eval_mode(
            scores=scores,
            positions=positions,
            seg=seg,
            feature_names=feature_names,
            execution_price="next_open",
            cfg=cfg,
        )
        if close_diag.status == "skipped" and open_diag.status == "skipped":
            results[seg_name] = SegmentDiagnostic(
                segment=seg_name,
                status="skipped",
                reason=close_diag.reason or open_diag.reason,
                n_rows=n,
                close=close_diag,
                next_open=open_diag,
            )
        else:
            results[seg_name] = SegmentDiagnostic(
                segment=seg_name,
                status="ok",
                n_rows=n,
                close=close_diag,
                next_open=open_diag,
            )
    return results


def _eval_mode(
    *,
    scores: np.ndarray,
    positions: np.ndarray,
    seg: Segment,
    feature_names: Sequence[str],
    execution_price: Literal["close", "next_open"],
    cfg: BacktestConfig,
) -> ExecutionDiagnostic:
    assert seg.y is not None
    try:
        close_px = OOSBacktestRunner._price_series(seg.X, feature_names, cfg.price_col)
        open_px = (
            OOSBacktestRunner._price_series(seg.X, feature_names, cfg.open_col)
            if execution_price == "next_open"
            else None
        )
        metrics, periods_per_year, n_gaps = evaluate_segment_execution(
            scores=scores,
            positions=positions,
            y=seg.y,
            timestamps=seg.timestamps,
            close_px=close_px,
            open_px=open_px,
            execution_price=execution_price,
            cfg=cfg,
            latency_ms=None,
        )
    except BacktestError as exc:
        return ExecutionDiagnostic(status="skipped", reason=str(exc))

    quant_gates = {k: bool(v) for k, v in metrics.gates.items() if k != "latency"}
    return ExecutionDiagnostic(
        status="ok",
        rank_ic=metrics.rank_ic,
        icir=metrics.icir,
        gross_sharpe=metrics.gross_sharpe,
        net_sharpe=metrics.net_sharpe,
        daily_net_sharpe=metrics.daily_net_sharpe,
        deflated_sharpe_ratio=metrics.deflated_sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        downside_ic=metrics.downside_ic,
        n_stress_bars=metrics.n_stress_bars,
        mean_turnover=metrics.mean_turnover,
        n_bars=metrics.n_bars,
        periods_per_year=periods_per_year,
        n_overnight_gaps=n_gaps,
        quant_gates=quant_gates,
        quant_accepted=all(quant_gates.values()),
    )


def _load_segment_npz(path: Path) -> Segment | None:
    if not path.is_file():
        return None
    with np.load(path) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int32) if "y" in data.files else None
        timestamps = np.asarray(data["timestamps"])
        index = np.asarray(data["index"], dtype=np.int64)
    return Segment(X=X, y=y, labels_aux={}, timestamps=timestamps, index=index)


def diagnose_run_dir(
    run_dir: str | Path,
    *,
    out_path: str | Path | None = None,
    segment_names: Sequence[str] = ("cal_a", "cal_b", "val", "oos"),
) -> dict[str, SegmentDiagnostic]:
    """Load artifacts from a completed run directory and write ``03_regime_diagnostics.json``."""
    root = Path(run_dir)
    bundle_dir = root / "05_packaging" / "bundle"
    task_path = root / "01_data" / "02_task.json"
    if not bundle_dir.is_dir():
        raise BacktestError(f"bundle directory not found: {bundle_dir}")
    if not task_path.is_file():
        raise BacktestError(f"task metadata not found: {task_path}")

    task_meta = json.loads(task_path.read_text(encoding="utf-8"))
    feature_names: list[str] = list(task_meta["feature_names"])

    cfg_path = root / "config.toml"
    if cfg_path.is_file():
        bt_cfg = load_config(cfg_path).backtest or BacktestConfig()
    else:
        bt_cfg = BacktestConfig()

    seg_dir = root / "01_data" / "segments"
    segments: dict[str, Segment | None] = {
        name: _load_segment_npz(seg_dir / f"{name}.npz") for name in segment_names
    }
    diagnostics = evaluate_bundle_segments(bundle_dir, segments, feature_names, bt_cfg)

    dest = Path(out_path) if out_path is not None else (root / "06_backtest" / "03_regime_diagnostics.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(root),
        "config": bt_cfg.model_dump(),
        "segments": {k: v.model_dump() for k, v in diagnostics.items()},
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return diagnostics
