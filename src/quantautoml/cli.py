"""Command-line entrypoint: ``quantautoml run --config config.toml --out out/``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from quantautoml.backtest.diagnostics import diagnose_run_dir
from quantautoml.common.exceptions import QuantAutoMLError
from quantautoml.pipeline import run_pipeline


def _format_duration(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    if total_ms < 60_000:
        return f"{seconds:.2f}s"
    mins, rem_s = divmod(seconds, 60.0)
    if mins < 60:
        return f"{int(mins)}m {rem_s:.2f}s"
    hours, rem_m = divmod(int(mins), 60)
    return f"{hours}h {rem_m}m {rem_s:.2f}s"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantautoml")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("run", help="run the Quant AutoML pipeline")
    prep.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="path to TOML config (default: config.toml)",
    )
    prep.add_argument(
        "--out",
        type=Path,
        default=Path("runs/default"),
        help="output directory (default: runs/default)",
    )
    diag = sub.add_parser(
        "diagnose", help="run read-only cross-regime diagnostics on a completed run"
    )
    diag.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/default"),
        help="path to a completed run directory (default: runs/default)",
    )
    diag.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output JSON path (default: <run-dir>/06_backtest/03_regime_diagnostics.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    start_dt = datetime.now()
    t0 = time.perf_counter()
    if args.command == "diagnose":
        try:
            diagnostics = diagnose_run_dir(args.run_dir, out_path=args.out)
        except (QuantAutoMLError, ValidationError, OSError) as exc:
            elapsed = time.perf_counter() - t0
            end_dt = datetime.now()
            print(
                f"error: {exc} (end={end_dt.strftime('%Y-%m-%d %H:%M:%S')} duration={_format_duration(elapsed)})",
                file=sys.stderr,
            )
            return 1
        for seg_name, diag_res in diagnostics.items():
            if diag_res.status != "ok" or diag_res.close is None:
                print(f"{seg_name}: skipped ({diag_res.reason})")
                continue
            c = diag_res.close
            o = diag_res.next_open
            open_str = (
                f"next_open_net_sharpe={o.net_sharpe:.3f}"
                if o is not None and o.status == "ok" and o.net_sharpe is not None
                else "next_open=skipped"
            )
            print(
                f"{seg_name}: rank_ic={c.rank_ic:+.4f} icir={c.icir:+.3f} "
                f"close_net_sharpe={c.net_sharpe:.3f} {open_str}"
            )
        return 0
    print(f"start={start_dt.strftime('%Y-%m-%d %H:%M:%S')} config={args.config} out={args.out}", flush=True)
    try:
        result = run_pipeline(args.config, args.out)
    except (QuantAutoMLError, ValidationError, OSError) as exc:
        elapsed = time.perf_counter() - t0
        end_dt = datetime.now()
        print(
            f"error: {exc} (end={end_dt.strftime('%Y-%m-%d %H:%M:%S')} duration={_format_duration(elapsed)})",
            file=sys.stderr,
        )
        return 1
    elapsed = time.perf_counter() - t0
    end_dt = datetime.now()
    rep = result.split.report
    print(
        f"task={result.dataset.task_type.value if result.dataset.task_type else None} "
        f"mode={rep.mode.value} rows={rep.n_rows} -> {args.out}"
    )
    print(
        f"end={end_dt.strftime('%Y-%m-%d %H:%M:%S')} duration={_format_duration(elapsed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
