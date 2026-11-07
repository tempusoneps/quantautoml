"""Command-line entrypoint: ``quantautoml run --config config.toml --out out/``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from quantautoml.common.exceptions import QuantAutoMLError
from quantautoml.pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantautoml")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("run", help="run the implemented pipeline stages (currently: ingestion, task inference, split)")
    prep.add_argument("--config", type=Path, required=True, help="path to TOML config")
    prep.add_argument("--out", type=Path, required=True, help="output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_pipeline(args.config, args.out)
    except (QuantAutoMLError, ValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rep = result.split.report
    print(
        f"task={result.dataset.task_type.value if result.dataset.task_type else None} "
        f"mode={rep.mode.value} rows={rep.n_rows} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
