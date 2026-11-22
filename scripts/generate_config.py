#!/usr/bin/env python3
"""Script to generate a complete, ready-to-run config.toml for Quant AutoML Pipeline.

Can be run with defaults or custom command-line arguments:
Usage:
    python scripts/generate_config.py
    python scripts/generate_config.py --output config.toml --selected-label is_peak_10 --horizon-h 21
    python scripts/generate_config.py --output config.toml --selected-label allow_entry --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Preset directional text-label mappings for known label schemas
_KNOWN_DIRECTIONAL_MAPPINGS: dict[str, dict[str, float]] = {
    "allow_entry": {
        "Yes - Buy": 1,
        "Yes - Sell": -1,
        "No - None": 0,
        "No - Sideway": 0,
    },
    "direction_filter": {
        "Long": 1,
        "Short": -1,
        "None": 0,
    },
}


def detect_file_info(
    features_path: Path,
    labels_path: Path | None,
    selected_label: str | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_col": "Date",
        "price_col": "Close",
        "available_labels": [],
        "suggested_horizon": 21,
        "label_mapping": None,
    }

    # Inspect features file if it exists to verify timestamp_col / price_col
    if features_path.exists():
        try:
            if features_path.suffix.lower() == ".csv":
                with open(features_path, "r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                cols = [c.strip().strip('"') for c in header if c.strip()]
            elif features_path.suffix.lower() == ".parquet":
                import polars as pl

                cols = list(pl.read_parquet_schema(features_path).keys())
            else:
                cols = []
            for c in cols:
                if c.lower() in ("date", "timestamp", "time"):
                    info["timestamp_col"] = c
                    break
            for c in cols:
                if c.lower() == "close":
                    info["price_col"] = c
                    break
        except Exception as e:
            print(f"Warning: could not inspect features file: {e}", file=sys.stderr)

    # Inspect labels file if it exists
    if labels_path and labels_path.exists():
        try:
            if labels_path.suffix.lower() == ".csv":
                with open(labels_path, "r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                cols = [c.strip().strip('"') for c in header if c.strip()]
            elif labels_path.suffix.lower() == ".parquet":
                import polars as pl

                cols = list(pl.read_parquet_schema(labels_path).keys())
            else:
                cols = []
            if cols:
                for c in cols:
                    if c.lower() in ("date", "timestamp", "time"):
                        info["timestamp_col"] = c
                        break
                label_candidates = [c for c in cols if c != info["timestamp_col"]]
                info["available_labels"] = label_candidates
        except Exception as e:
            print(f"Warning: could not inspect labels file: {e}", file=sys.stderr)

    if selected_label and selected_label in _KNOWN_DIRECTIONAL_MAPPINGS:
        info["label_mapping"] = dict(_KNOWN_DIRECTIONAL_MAPPINGS[selected_label])

    return info


def build_config_content(
    features_path: str,
    labels_path: str | None,
    timestamp_col: str,
    min_match_ratio: float,
    selected_label: str | None,
    horizon_h: int,
    *,
    ignore_other_columns: bool = True,
    label_mapping: dict[str, float] | None = None,
    price_col: str = "Close",
    target_horizon: int = 5,
    tournament_trials: int = 30,
    tuning_trials: int = 50,
    realworld_size: str | int | float = "auto",
    include_full_pipeline: bool = True,
) -> str:
    is_supervised = bool(labels_path and selected_label)
    effective_mapping = label_mapping
    if is_supervised and effective_mapping is None and selected_label in _KNOWN_DIRECTIONAL_MAPPINGS:
        effective_mapping = _KNOWN_DIRECTIONAL_MAPPINGS[selected_label]

    lines: list[str] = [
        "# Quant AutoML Pipeline Configuration",
        "# Generated automatically by scripts/generate_config.py",
        "# Reference: docs/CONFIGURATION.md",
        "# Run: uv run quantautoml run --config config.toml --out runs/default",
        "",
        "[data]",
        "# Path to feature matrix file (.parquet or .csv)",
        f'features_path = "{features_path}"',
    ]

    if is_supervised:
        lines.extend(
            [
                "",
                "# Path to labels file (.parquet or .csv). Omit for UNSUPERVISED mode.",
                f'labels_path = "{labels_path}"',
            ]
        )

    lines.extend(
        [
            "",
            "# Name of timestamp column in both features and labels files",
            f'timestamp_col = "{timestamp_col}"',
            "",
            "# Minimum required ratio of matched timestamps during inner join (0.0 to 1.0)",
            f"min_match_ratio = {min_match_ratio}",
        ]
    )

    if is_supervised:
        ignore_str = "true" if ignore_other_columns else "false"
        lines.extend(
            [
                "",
                "[labels]",
                "# Target label column name used for task inference and training",
                f'selected = "{selected_label}"',
                "# Keep only the selected label column (avoids non-numeric sibling label errors)",
                f"ignore_other_columns = {ignore_str}",
            ]
        )
        if effective_mapping:
            lines.extend(
                [
                    "",
                    "[labels.mapping]",
                    "# Text label -> directional number mapping",
                ]
            )
            for k, v in effective_mapping.items():
                val_str = str(int(v)) if float(v).is_integer() else str(v)
                lines.append(f'"{k}" = {val_str}')

    lines.extend(
        [
            "",
            "[split]",
            "# Maximum forward-looking horizon (bars) across all labels and ic_horizons",
            "# Used for Purging & Embargo between Train, Cal-A, Cal-B, Val, and OOS segments.",
            "# Must be >= 1 and >= max(ic_horizons) / trailing NaN count.",
            f"horizon_h = {horizon_h}",
        ]
    )

    if is_supervised and include_full_pipeline:
        lines.extend(
            [
                "",
                "[features]",
                "# Leakage-free feature funnel (F1 -> F8); missing fwd_ret_* are derived from price_col",
                'ic_horizons = { 1 = "fwd_ret_1", 5 = "fwd_ret_5", 10 = "fwd_ret_10", 21 = "fwd_ret_21" }',
                f"target_horizon = {target_horizon}",
                f'price_col = "{price_col}"',
                "icir_min = 0.30",
                "fdr_alpha = 0.05",
                "rolling_window = 63",
                "dcor_threshold = 0.85",
                "cluster_distance = 0.30",
                "max_nan_ratio = 0.60",
                "min_unique_ratio = 0.01",
                "min_cv = 1e-6",
                "max_peak_var_ratio = 50.0",
                "max_features = 30",
                "monotonic_tolerance = 0.05",
                "mi_floor = 0.0",
                "synergy_delta = 0.01",
                "",
                "[models]",
                "# GBDT Trio Tournament & Deep Optuna TPE Tuning",
                f"tournament_trials = {tournament_trials}",
                f"tuning_trials = {tuning_trials}",
                "n_wf_folds = 5",
                "diversity_threshold = 0.88",
                "marginal_gain_threshold = 0.015",
                "wf_consistency_threshold = 0.60",
                "complexity_penalty = 0.05",
                "random_state = 42",
                'enabled_models = ["lightgbm", "xgboost", "catboost"]',
                "",
                "[ensemble]",
                "# Convex SLSQP (C^2 objective on Cal-B) + Bootstrap Shrinkage + Temperature Calibration (Cal-A)",
                "min_weight = 0.05",
                "max_weight = 0.60",
                "n_bootstrap = 200",
                "bootstrap_sample_ratio = 0.80",
                "shrinkage_factor = 0.10",
                "hurdle_ratio = 0.05",
                "max_ece_threshold = 0.05",
                "random_state = 42",
                "",
                "[packaging]",
                "# Standalone ONNX Model Bundle (float32 contract < 1e-4, latency < 5 ms)",
                "precision_tol = 1e-4",
                "drift_z_threshold = 3.0",
                "latency_budget_ms = 5.0",
                "std_floor = 1e-8",
                "latency_repeats = 200",
                "",
                "[backtest]",
                "# Friction-aware blind OOS acceptance with Hysteresis Buffer Bands",
                f'price_col = "{price_col}"',
                "fee_bps = 4.0",
                "slippage_bps = 1.0",
                "top_pct = 0.20",
                "exit_pct = 0.30",
                "rank_window = 2000",
                "ic_window = 500",
                "min_rank_ic = 0.035",
                "min_icir = 0.40",
                "min_net_sharpe = 1.20",
                "max_mdd = 0.18",
                "downside_quantile = 0.20",
                "min_downside_ic = 0.0",
                "",
                "[realworld]",
                '# Tier 2 realistic execution via backtesting.py (size: "auto", "_start_cash", 1, or fraction 0.5)',
                "cash = 1000000.0",
                "commission = 0.0005",
                "margin = 1.0",
                f"size = {realworld_size if isinstance(realworld_size, (int, float)) else f'\"{realworld_size}\"'}",
                "trade_on_close = false",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a ready-to-run config.toml for Quant AutoML Pipeline."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="config.toml",
        help="Output configuration file path (default: config.toml)",
    )
    parser.add_argument(
        "--features-path",
        default="datasets/feature.parquet",
        help="Path to features file (default: datasets/feature.parquet)",
    )
    parser.add_argument(
        "--labels-path",
        default="datasets/label.csv",
        help="Path to labels file (default: datasets/label.csv; empty string for unsupervised)",
    )
    parser.add_argument(
        "--timestamp-col",
        default=None,
        help="Timestamp column name (auto-detected if omitted, default: Date)",
    )
    parser.add_argument(
        "--price-col",
        default=None,
        help="Price column name in features for forward returns & backtest (default: Close)",
    )
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.95,
        help="Minimum timestamp match ratio (default: 0.95)",
    )
    parser.add_argument(
        "--selected-label",
        default=None,
        help="Target label column (e.g. is_peak_10, allow_entry, entry_filter)",
    )
    parser.add_argument(
        "--keep-all-labels",
        action="store_true",
        help="Set labels.ignore_other_columns = false (require all label columns to be numeric)",
    )
    parser.add_argument(
        "--horizon-h",
        type=int,
        default=21,
        help="Purging & Embargo horizon in bars (default: 21, must be >= max ic_horizon)",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=5,
        help="Target horizon in features.ic_horizons (default: 5)",
    )
    parser.add_argument(
        "--tournament-trials",
        type=int,
        default=30,
        help="Tournament random search trials per GBDT model (default: 30)",
    )
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=50,
        help="Deep Optuna TPE tuning trials (default: 50)",
    )
    parser.add_argument(
        "--size",
        default="auto",
        help='Tier 2 position size mode: "auto", "_start_cash", integer >= 1 (e.g. 1), or fraction (0, 1) (default: auto)',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config file without confirmation",
    )

    args = parser.parse_args()

    feat_path = Path(args.features_path)
    lbl_path_str: str | None = args.labels_path.strip() if args.labels_path else None
    lbl_path = Path(lbl_path_str) if lbl_path_str else None

    detected = detect_file_info(feat_path, lbl_path, selected_label=args.selected_label)

    timestamp_col = args.timestamp_col or detected["timestamp_col"]
    price_col = args.price_col or detected["price_col"]
    selected_label: str | None = args.selected_label
    if lbl_path_str and not selected_label:
        if "is_peak_10" in detected["available_labels"]:
            selected_label = "is_peak_10"
        elif "allow_entry" in detected["available_labels"]:
            selected_label = "allow_entry"
        elif detected["available_labels"]:
            selected_label = detected["available_labels"][0]
        else:
            selected_label = "is_peak_10"

    label_mapping = _KNOWN_DIRECTIONAL_MAPPINGS.get(selected_label or "")

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        print(f"File '{out_path}' already exists.")
        ans = input("Do you want to overwrite it? [y/N]: ").strip().lower()
        if ans not in ["y", "yes"]:
            print("Aborted.")
            sys.exit(0)

    raw_size = str(args.size).strip()
    parsed_size: str | int | float = raw_size
    if raw_size not in ("auto", "start_cash", "_start_cash"):
        try:
            fval = float(raw_size)
            parsed_size = int(fval) if fval.is_integer() and fval >= 1.0 else fval
        except ValueError:
            parsed_size = raw_size

    content = build_config_content(
        features_path=args.features_path,
        labels_path=lbl_path_str,
        timestamp_col=timestamp_col,
        min_match_ratio=args.min_match_ratio,
        selected_label=selected_label,
        horizon_h=args.horizon_h,
        ignore_other_columns=not args.keep_all_labels,
        label_mapping=label_mapping,
        price_col=price_col,
        target_horizon=args.target_horizon,
        tournament_trials=args.tournament_trials,
        tuning_trials=args.tuning_trials,
        realworld_size=parsed_size,
    )

    out_path.write_text(content, encoding="utf-8")
    print(f"Successfully generated config at: {out_path.resolve()}")
    print("Configuration preview:")
    print("-" * 50)
    print(content.strip())
    print("-" * 50)


if __name__ == "__main__":
    main()
