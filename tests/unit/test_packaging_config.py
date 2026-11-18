"""Unit tests for PackagingConfig and packaging/backtest exceptions."""

import pytest
from pydantic import ValidationError

from quantautoml.common.config import PackagingConfig, PipelineConfig, load_config
from quantautoml.common.exceptions import (
    BacktestError,
    PackagingError,
    PrecisionValidationError,
    QuantAutoMLError,
    SchemaMismatchError,
)


def test_packaging_defaults_match_spec() -> None:
    cfg = PackagingConfig()
    assert cfg.precision_tol == 1e-4
    assert cfg.drift_z_threshold == 3.0
    assert cfg.latency_budget_ms == 5.0
    assert cfg.std_floor == 1e-8
    assert cfg.latency_repeats == 200


def test_exception_hierarchy() -> None:
    assert issubclass(PackagingError, QuantAutoMLError)
    assert issubclass(PrecisionValidationError, PackagingError)
    assert issubclass(SchemaMismatchError, PackagingError)
    assert issubclass(BacktestError, QuantAutoMLError)


def test_packaging_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        PackagingConfig(unknown=1)  # type: ignore[call-arg]


def test_packaging_rejects_non_positive_tol() -> None:
    with pytest.raises(ValidationError):
        PackagingConfig(precision_tol=0.0)


def test_pipeline_config_accepts_optional_sections(write_toml, valid_toml) -> None:
    toml = valid_toml + "\n[packaging]\nprecision_tol = 1e-5\n\n[backtest]\nfee_bps = 2.0\n"
    cfg = load_config(write_toml(toml))
    assert isinstance(cfg, PipelineConfig)
    assert cfg.packaging is not None and cfg.packaging.precision_tol == 1e-5
    assert cfg.backtest is not None and cfg.backtest.fee_bps == 2.0


def test_pipeline_config_sections_default_none(write_toml, valid_toml) -> None:
    cfg = load_config(write_toml(valid_toml))
    assert cfg.packaging is None and cfg.backtest is None
