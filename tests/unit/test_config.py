from pathlib import Path

import pydantic
import pytest

from quantautoml.common.config import PipelineConfig, load_config


def test_load_config_parses_valid_toml(write_toml, valid_toml: str) -> None:
    cfg = load_config(write_toml(valid_toml))
    assert isinstance(cfg, PipelineConfig)
    assert cfg.data.features_path == Path("datasets/features.parquet")
    assert cfg.data.labels_path == Path("datasets/labels.parquet")
    assert cfg.labels.selected == "fwd_ret_5"
    assert cfg.split.horizon_h == 21


def test_defaults_are_applied(write_toml, valid_toml: str) -> None:
    cfg = load_config(write_toml(valid_toml))
    assert cfg.data.timestamp_col == "timestamp"
    assert cfg.data.min_match_ratio == 0.95


def test_missing_horizon_is_rejected(write_toml) -> None:
    toml = '[data]\nfeatures_path = "f.parquet"\n'
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(toml))


@pytest.mark.parametrize("h", [0, -3])
def test_horizon_below_one_is_rejected(write_toml, valid_toml: str, h: int) -> None:
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(valid_toml.replace("horizon_h = 21", f"horizon_h = {h}")))


@pytest.mark.parametrize("ratio", ["0", "1.5", "-0.1"])
def test_min_match_ratio_must_be_in_unit_interval(write_toml, valid_toml: str, ratio: str) -> None:
    toml = valid_toml.replace("[labels]", f"min_match_ratio = {ratio}\n\n[labels]")
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(toml))


def test_labels_path_requires_selected(write_toml) -> None:
    toml = (
        '[data]\nfeatures_path = "f.parquet"\nlabels_path = "l.parquet"\n'
        "[split]\nhorizon_h = 5\n"
    )
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(toml))


def test_selected_without_labels_path_is_rejected(write_toml) -> None:
    toml = (
        '[data]\nfeatures_path = "f.parquet"\n'
        '[labels]\nselected = "fwd_ret_5"\n'
        "[split]\nhorizon_h = 5\n"
    )
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(toml))


def test_unsupervised_config_without_labels_is_valid(write_toml) -> None:
    toml = '[data]\nfeatures_path = "f.parquet"\n[split]\nhorizon_h = 5\n'
    cfg = load_config(write_toml(toml))
    assert cfg.data.labels_path is None
    assert cfg.labels.selected is None


def test_unknown_key_is_rejected(write_toml, valid_toml: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(valid_toml + "\n[split_typo]\nx = 1\n"))


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")
