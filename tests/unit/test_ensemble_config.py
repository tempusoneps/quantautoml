"""Unit tests for EnsembleConfig and ensemble-related exceptions."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from quantautoml.common.config import EnsembleConfig, PipelineConfig, load_config
from quantautoml.common.exceptions import (
    CalibrationError,
    EnsembleError,
    OptimizationConvergenceError,
    QuantAutoMLError,
)


def test_ensemble_exceptions_hierarchy() -> None:
    assert issubclass(EnsembleError, QuantAutoMLError)
    assert issubclass(OptimizationConvergenceError, EnsembleError)
    assert issubclass(CalibrationError, EnsembleError)


def test_ensemble_config_defaults() -> None:
    cfg = EnsembleConfig()
    assert cfg.min_weight == 0.05
    assert cfg.max_weight == 0.60
    assert cfg.n_bootstrap == 200
    assert cfg.bootstrap_sample_ratio == 0.80
    assert cfg.shrinkage_factor == 0.10
    assert cfg.hurdle_ratio == 0.05
    assert cfg.max_ece_threshold == 0.05
    assert cfg.random_state == 42


def test_ensemble_config_custom_valid() -> None:
    cfg = EnsembleConfig(
        min_weight=0.10,
        max_weight=0.50,
        n_bootstrap=100,
        bootstrap_sample_ratio=0.75,
        shrinkage_factor=0.20,
        hurdle_ratio=0.08,
        max_ece_threshold=0.03,
        random_state=123,
    )
    assert cfg.min_weight == 0.10
    assert cfg.max_weight == 0.50
    assert cfg.n_bootstrap == 100
    assert cfg.bootstrap_sample_ratio == 0.75
    assert cfg.shrinkage_factor == 0.20
    assert cfg.hurdle_ratio == 0.08
    assert cfg.max_ece_threshold == 0.03
    assert cfg.random_state == 123


def test_ensemble_config_invalid_weights() -> None:
    with pytest.raises(ValidationError):
        # min_weight >= max_weight
        EnsembleConfig(min_weight=0.60, max_weight=0.50)

    with pytest.raises(ValidationError):
        # min_weight < 0
        EnsembleConfig(min_weight=-0.01)

    with pytest.raises(ValidationError):
        # max_weight > 1.0
        EnsembleConfig(max_weight=1.5)


def test_ensemble_config_invalid_bootstrap_and_shrinkage() -> None:
    with pytest.raises(ValidationError):
        # n_bootstrap < 10
        EnsembleConfig(n_bootstrap=5)

    with pytest.raises(ValidationError):
        # bootstrap_sample_ratio <= 0
        EnsembleConfig(bootstrap_sample_ratio=0.0)

    with pytest.raises(ValidationError):
        # bootstrap_sample_ratio > 1.0
        EnsembleConfig(bootstrap_sample_ratio=1.2)

    with pytest.raises(ValidationError):
        # shrinkage_factor < 0
        EnsembleConfig(shrinkage_factor=-0.1)

    with pytest.raises(ValidationError):
        # shrinkage_factor > 1.0
        EnsembleConfig(shrinkage_factor=1.1)

    with pytest.raises(ValidationError):
        # hurdle_ratio < 0
        EnsembleConfig(hurdle_ratio=-0.05)


def test_pipeline_config_with_ensemble(tmp_path: Path) -> None:
    toml_content = """
    [data]
    features_path = "features.parquet"
    labels_path = "labels.parquet"

    [labels]
    selected = "label_h5"

    [split]
    horizon_h = 5

    [ensemble]
    min_weight = 0.05
    max_weight = 0.60
    n_bootstrap = 200
    shrinkage_factor = 0.10
    hurdle_ratio = 0.05
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text(toml_content)

    cfg = load_config(config_file)
    assert cfg.ensemble is not None
    assert isinstance(cfg.ensemble, EnsembleConfig)
    assert cfg.ensemble.min_weight == 0.05
    assert cfg.ensemble.max_weight == 0.60
    assert cfg.ensemble.n_bootstrap == 200


def test_ensemble_config_ic_window_default_and_custom(tmp_path: Path) -> None:
    assert EnsembleConfig().ic_window is None
    assert EnsembleConfig(ic_window=500).ic_window == 500
    toml_content = """
    [data]
    features_path = "features.parquet"
    labels_path = "labels.parquet"

    [labels]
    selected = "label_h5"

    [split]
    horizon_h = 5

    [ensemble]
    ic_window = 300
    """
    cfg_file = tmp_path / "cfg_icw.toml"
    cfg_file.write_text(toml_content, encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.ensemble is not None and cfg.ensemble.ic_window == 300


@pytest.mark.parametrize("w", [19, 0, -5])
def test_ensemble_config_ic_window_below_20_rejected(w: int) -> None:
    with pytest.raises(ValidationError):
        EnsembleConfig(ic_window=w)
