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


_FEATURES_TOML = (
    "\n[features]\n"
    "target_horizon = 5\n"
    "[features.ic_horizons]\n"
    '1 = "fwd_ret_1"\n5 = "fwd_ret_5"\n10 = "fwd_ret_10"\n21 = "fwd_ret_21"\n'
)


def test_features_section_parses(write_toml, valid_toml: str) -> None:
    cfg = load_config(write_toml(valid_toml + _FEATURES_TOML))
    assert cfg.features is not None
    assert cfg.features.ic_horizons == {1: "fwd_ret_1", 5: "fwd_ret_5", 10: "fwd_ret_10", 21: "fwd_ret_21"}
    assert cfg.features.target_horizon == 5
    assert cfg.features.icir_min == 0.30
    assert cfg.features.fdr_alpha == 0.05
    assert cfg.features.rolling_window == 63
    assert cfg.features.dcor_threshold == 0.85
    assert cfg.features.cluster_distance == 0.30
    assert cfg.features.max_nan_ratio == 0.6
    assert cfg.features.min_unique_ratio == 0.01
    assert cfg.features.min_cv == 1e-6
    assert cfg.features.max_peak_var_ratio == 50.0
    assert cfg.features.max_features == 30
    assert cfg.features.monotonic_tolerance == 0.05


def test_target_horizon_must_be_in_ic_horizons(write_toml, valid_toml: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(valid_toml + _FEATURES_TOML.replace("target_horizon = 5", "target_horizon = 7")))


def test_missing_features_section_is_none(write_toml, valid_toml: str) -> None:
    assert load_config(write_toml(valid_toml)).features is None


@pytest.mark.parametrize("alpha", ["0", "-0.01", "0.06", "1.0"])
def test_fdr_alpha_must_be_in_zero_to_005(write_toml, valid_toml: str, alpha: str) -> None:
    toml = valid_toml + _FEATURES_TOML.replace("target_horizon = 5", f"target_horizon = 5\nfdr_alpha = {alpha}")
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(toml))


def test_feature_selection_error_is_quantautoml_error() -> None:
    from quantautoml.common.exceptions import FeatureSelectionError, QuantAutoMLError

    assert issubclass(FeatureSelectionError, QuantAutoMLError)


def test_f6_thresholds_defaults(write_toml, valid_toml: str) -> None:
    cfg = load_config(write_toml(valid_toml + _FEATURES_TOML))
    assert cfg.features is not None
    assert cfg.features.mi_floor == 0.0
    assert cfg.features.synergy_delta == 0.01


def test_f6_thresholds_parse(write_toml, valid_toml: str) -> None:
    extra = "target_horizon = 5\nmi_floor = 0.002\nsynergy_delta = 0.05"
    cfg = load_config(write_toml(valid_toml + _FEATURES_TOML.replace("target_horizon = 5", extra)))
    assert cfg.features is not None
    assert (cfg.features.mi_floor, cfg.features.synergy_delta) == (0.002, 0.05)


@pytest.mark.parametrize("key", ["mi_floor", "synergy_delta"])
def test_f6_thresholds_must_be_non_negative(write_toml, valid_toml: str, key: str) -> None:
    toml = valid_toml + _FEATURES_TOML.replace("target_horizon = 5", f"target_horizon = 5\n{key} = -0.001")
    with pytest.raises(pydantic.ValidationError):
        load_config(write_toml(toml))
