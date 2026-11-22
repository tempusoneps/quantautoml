"""Unit tests for scripts/generate_config.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest

from quantautoml.common.config import PipelineConfig, load_config
from quantautoml.data.ingestion import DataIngestion


def _load_generate_config():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "generate_config.py"
    spec = importlib.util.spec_from_file_location("generate_config", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_generated_config_loads_full_pipeline_sections(tmp_path: Path) -> None:
    gen = _load_generate_config()
    content = gen.build_config_content(
        features_path="datasets/feature.parquet",
        labels_path="datasets/label.csv",
        timestamp_col="Date",
        min_match_ratio=0.95,
        selected_label="is_peak_10",
        horizon_h=21,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(content, encoding="utf-8")

    cfg = load_config(cfg_path)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.data.features_path == Path("datasets/feature.parquet")
    assert cfg.data.labels_path == Path("datasets/label.csv")
    assert cfg.data.timestamp_col == "Date"
    assert cfg.labels.selected == "is_peak_10"
    assert cfg.labels.ignore_other_columns is True
    assert cfg.split.horizon_h == 21
    assert cfg.features is not None
    assert cfg.features.ic_horizons == {
        1: "fwd_ret_1",
        5: "fwd_ret_5",
        10: "fwd_ret_10",
        21: "fwd_ret_21",
    }
    assert cfg.features.target_horizon == 5
    assert cfg.features.price_col == "Close"
    assert cfg.models is not None
    assert cfg.ensemble is not None
    assert cfg.packaging is not None
    assert cfg.backtest is not None


def test_generated_config_ingests_labels_with_non_numeric_sibling_columns(
    tmp_path: Path,
) -> None:
    """When label file has mixed string/bool columns (like datasets/label.csv),
    generated config must pass DataIngestion.load() without failing on non-numeric sibling columns.
    """
    gen = _load_generate_config()
    feats_path = tmp_path / "feature.parquet"
    labels_path = tmp_path / "label.csv"

    n = 100
    dates = [f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00" for i in range(n)]
    pl.DataFrame(
        {
            "Date": dates,
            "Close": [100.0 + 0.1 * i for i in range(n)],
            "f1": [float(i) for i in range(n)],
        }
    ).with_columns(pl.col("Date").str.to_datetime()).write_parquet(feats_path)

    pl.DataFrame(
        {
            "Date": dates,
            "entry_filter": ["Yes" if i % 2 == 0 else "No" for i in range(n)],
            "allow_entry": [
                "Yes - Buy" if i % 3 == 0 else ("Yes - Sell" if i % 3 == 1 else "No - None")
                for i in range(n)
            ],
            "is_peak_10": [i % 2 == 0 for i in range(n)],
        }
    ).write_csv(labels_path)

    content = gen.build_config_content(
        features_path=str(feats_path),
        labels_path=str(labels_path),
        timestamp_col="Date",
        min_match_ratio=0.95,
        selected_label="is_peak_10",
        horizon_h=21,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(content, encoding="utf-8")

    cfg = load_config(cfg_path)
    dataset = DataIngestion(cfg).load()
    assert dataset.X.shape[0] == n
    assert set(dataset.labels_aux.keys()) >= {"fwd_ret_1", "fwd_ret_5", "fwd_ret_10", "fwd_ret_21"}


def test_generated_config_for_allow_entry_includes_directional_mapping(
    tmp_path: Path,
) -> None:
    gen = _load_generate_config()
    feats_path = tmp_path / "feature.parquet"
    labels_path = tmp_path / "label.csv"

    n = 100
    dates = [f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00" for i in range(n)]
    pl.DataFrame(
        {
            "Date": dates,
            "Close": [100.0 + 0.1 * i for i in range(n)],
            "f1": [float(i) for i in range(n)],
        }
    ).with_columns(pl.col("Date").str.to_datetime()).write_parquet(feats_path)

    pl.DataFrame(
        {
            "Date": dates,
            "allow_entry": [
                "Yes - Buy"
                if i % 4 == 0
                else ("Yes - Sell" if i % 4 == 1 else ("No - Sideway" if i % 4 == 2 else "No - None"))
                for i in range(n)
            ],
        }
    ).write_csv(labels_path)

    detected = gen.detect_file_info(feats_path, labels_path, selected_label="allow_entry")
    content = gen.build_config_content(
        features_path=str(feats_path),
        labels_path=str(labels_path),
        timestamp_col="Date",
        min_match_ratio=0.95,
        selected_label="allow_entry",
        horizon_h=21,
        label_mapping=detected.get("label_mapping"),
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(content, encoding="utf-8")

    cfg = load_config(cfg_path)
    assert cfg.labels.mapping == {
        "Yes - Buy": 1.0,
        "Yes - Sell": -1.0,
        "No - None": 0.0,
        "No - Sideway": 0.0,
        "NaN": 0.0,
    }
    dataset = DataIngestion(cfg).load()
    assert list(dataset.classes_) == [-1, 0, 1]


def test_unsupervised_generated_config_is_valid(tmp_path: Path) -> None:
    """When labels_path is empty/None, generated config must have no [labels] selected
    and no supervised-only sections so load_config() succeeds.
    """
    gen = _load_generate_config()
    content = gen.build_config_content(
        features_path="datasets/feature.parquet",
        labels_path=None,
        timestamp_col="Date",
        min_match_ratio=0.95,
        selected_label=None,
        horizon_h=21,
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(content, encoding="utf-8")

    cfg = load_config(cfg_path)
    assert cfg.data.labels_path is None
    assert cfg.labels.selected is None


def test_cli_main_generates_runnable_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen = _load_generate_config()
    out_file = tmp_path / "generated.toml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_config.py",
            "--output",
            str(out_file),
            "--force",
        ],
    )
    gen.main()
    assert out_file.is_file()
    cfg = load_config(out_file)
    assert cfg.split.horizon_h >= 21
    assert cfg.features is not None
    assert cfg.models is not None
    assert cfg.ensemble is not None
    assert cfg.packaging is not None
    assert cfg.backtest is not None
