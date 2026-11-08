#!/usr/bin/env python3
"""Script to generate config.toml for Quant AutoML Pipeline.

Can be run interactively or with command-line arguments.
Usage:
    python scripts/generate_config.py
    python scripts/generate_config.py --output config.toml --selected-label is_peak_10 --horizon-h 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def detect_file_info(
    features_path: Path, labels_path: Path | None
) -> dict[str, any]:
    info = {
        "timestamp_col": "Date",
        "available_labels": [],
        "suggested_horizon": 10,
    }

    # Inspect labels if file exists
    if labels_path and labels_path.exists():
        try:
            # Quick CSV header read without heavy dependencies
            with open(labels_path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            if header:
                cols = [c.strip().strip('"') for c in header]
                for c in cols:
                    if c.lower() in ["date", "timestamp", "time"]:
                        info["timestamp_col"] = c
                        break
                label_candidates = [
                    c for c in cols if c != info["timestamp_col"]
                ]
                info["available_labels"] = label_candidates
        except Exception as e:
            print(f"Warning: could not inspect labels file: {e}", file=sys.stderr)

    return info


def build_config_content(
    features_path: str,
    labels_path: str | None,
    timestamp_col: str,
    min_match_ratio: float,
    selected_label: str | None,
    horizon_h: int,
) -> str:
    labels_path_str = f'"{labels_path}"' if labels_path else '""'
    selected_str = f'"{selected_label}"' if selected_label else '""'

    content = f"""# Quant AutoML Pipeline Configuration
# Generated automatically by scripts/generate_config.py
# Reference: docs/CONFIGURATION.md

[data]
# Path to feature matrix file (.parquet or .csv)
features_path = "{features_path}"

# Path to labels file (.parquet or .csv). Leave empty ("") for UNSUPERVISED mode.
labels_path = {labels_path_str}

# Name of timestamp column in both features and labels files
timestamp_col = "{timestamp_col}"

# Minimum required ratio of matched timestamps during inner join (0.0 to 1.0)
min_match_ratio = {min_match_ratio}

[labels]
# Target label column name used for task inference and training
# Other label columns in labels_path will be stored in labels_aux
selected = {selected_str}

[split]
# Maximum forward-looking horizon (number of bars/candles) across all labels
# Used for Purging & Embargo between Train, Cal-A, Cal-B, Val, and OOS segments.
# Must be >= 1 and >= max tail NaN count of any label column.
horizon_h = {horizon_h}
"""
    return content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate config.toml for Quant AutoML Pipeline."
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
        help="Path to labels file (default: datasets/label.csv; empty for unsupervised)",
    )
    parser.add_argument(
        "--timestamp-col",
        default=None,
        help="Timestamp column name (auto-detected if omitted, default: Date)",
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
        help="Target label column (e.g. is_peak_10, entry_filter)",
    )
    parser.add_argument(
        "--horizon-h",
        type=int,
        default=10,
        help="Purging & Embargo horizon in bars (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config file without confirmation",
    )

    args = parser.parse_args()

    feat_path = Path(args.features_path)
    lbl_path = Path(args.labels_path) if args.labels_path else None

    detected = detect_file_info(feat_path, lbl_path)

    timestamp_col = args.timestamp_col or detected["timestamp_col"]
    selected_label = args.selected_label
    if not selected_label:
        if "is_peak_10" in detected["available_labels"]:
            selected_label = "is_peak_10"
        elif detected["available_labels"]:
            selected_label = detected["available_labels"][0]
        else:
            selected_label = "selected_label"

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        print(f"File '{out_path}' already exists.")
        ans = input("Do you want to overwrite it? [y/N]: ").strip().lower()
        if ans not in ["y", "yes"]:
            print("Aborted.")
            sys.exit(0)

    content = build_config_content(
        features_path=args.features_path,
        labels_path=args.labels_path,
        timestamp_col=timestamp_col,
        min_match_ratio=args.min_match_ratio,
        selected_label=selected_label,
        horizon_h=args.horizon_h,
    )

    out_path.write_text(content, encoding="utf-8")
    print(f" Successfully generated config at: {out_path.resolve()}")
    print("Configuration preview:")
    print("-" * 50)
    print(content.strip())
    print("-" * 50)


if __name__ == "__main__":
    main()
