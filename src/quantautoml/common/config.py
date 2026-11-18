"""Pipeline configuration loaded from a TOML file and validated with Pydantic."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_Strict):
    features_path: Path
    labels_path: Path | None = None
    timestamp_col: str = "timestamp"
    min_match_ratio: float = Field(default=0.95, gt=0.0, le=1.0)


class LabelsConfig(_Strict):
    selected: str | None = None


class SplitConfig(_Strict):
    horizon_h: int = Field(ge=1)


class FeaturesConfig(_Strict):
    ic_horizons: dict[int, str]
    target_horizon: int
    icir_min: float = 0.30
    fdr_alpha: float = Field(default=0.05, gt=0.0, le=0.05)
    rolling_window: int = Field(default=63, ge=2)
    dcor_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    cluster_distance: float = Field(default=0.30, gt=0.0, lt=1.0)
    max_nan_ratio: float = Field(default=0.6, gt=0.0, le=1.0)
    min_unique_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    min_cv: float = Field(default=1e-6, ge=0.0)
    max_peak_var_ratio: float = Field(default=50.0, gt=0.0)
    max_features: int = Field(default=30, ge=1)
    monotonic_tolerance: float = Field(default=0.05, ge=0.0)
    mi_floor: float = Field(default=0.0, ge=0.0)
    synergy_delta: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _target_in_horizons(self) -> Self:
        if self.target_horizon not in self.ic_horizons:
            raise ValueError("features.target_horizon must be a key of features.ic_horizons")
        return self


_ALLOWED_MODELS = {"lightgbm", "xgboost", "catboost"}


class ModelsConfig(_Strict):
    tournament_trials: int = Field(default=30, ge=1)
    tuning_trials: int = Field(default=50, ge=1)
    n_wf_folds: int = Field(default=5, ge=2)
    diversity_threshold: float = Field(default=0.88, gt=0.0, le=1.0)
    marginal_gain_threshold: float = Field(default=0.015, ge=0.0)
    wf_consistency_threshold: float = Field(default=0.60, gt=0.0, le=1.0)
    complexity_penalty: float = Field(default=0.05, ge=0.0)
    random_state: int = Field(default=42)
    enabled_models: list[str] = Field(default_factory=lambda: ["lightgbm", "xgboost", "catboost"])

    @field_validator("enabled_models")
    @classmethod
    def _validate_enabled_models(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("enabled_models cannot be empty")
        for m in v:
            if m not in _ALLOWED_MODELS:
                raise ValueError(f"Unknown model '{m}', must be one of {_ALLOWED_MODELS}")
        return v


class EnsembleConfig(_Strict):
    min_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    max_weight: float = Field(default=0.60, gt=0.0, le=1.0)
    n_bootstrap: int = Field(default=200, ge=10)
    bootstrap_sample_ratio: float = Field(default=0.80, gt=0.0, le=1.0)
    shrinkage_factor: float = Field(default=0.10, ge=0.0, le=1.0)
    hurdle_ratio: float = Field(default=0.05, ge=0.0)
    max_ece_threshold: float = Field(default=0.05, gt=0.0, le=1.0)
    random_state: int = Field(default=42)

    @model_validator(mode="after")
    def _weights_consistency(self) -> Self:
        if self.min_weight >= self.max_weight:
            raise ValueError("ensemble.min_weight must be strictly less than ensemble.max_weight")
        return self


class PackagingConfig(_Strict):
    precision_tol: float = Field(default=1e-4, gt=0.0)
    drift_z_threshold: float = Field(default=3.0, gt=0.0)
    latency_budget_ms: float = Field(default=5.0, gt=0.0)
    std_floor: float = Field(default=1e-8, gt=0.0)
    latency_repeats: int = Field(default=200, ge=1)


class BacktestConfig(_Strict):
    price_col: str = "Close"
    fee_bps: float = Field(default=4.0, ge=0.0)
    slippage_bps: float = Field(default=1.0, ge=0.0)
    top_pct: float = Field(default=0.20, gt=0.0, lt=0.5)
    exit_pct: float = Field(default=0.30, gt=0.0, lt=0.5)
    rank_window: int = Field(default=2000, ge=20)
    ic_window: int = Field(default=500, ge=20)
    min_rank_ic: float = 0.035
    min_icir: float = 0.40
    min_net_sharpe: float = 1.20
    max_mdd: float = Field(default=0.18, gt=0.0)
    downside_quantile: float = Field(default=0.20, gt=0.0, lt=0.5)
    min_downside_ic: float = 0.0

    @model_validator(mode="after")
    def _exit_not_tighter_than_entry(self) -> Self:
        if self.exit_pct < self.top_pct:
            raise ValueError("backtest.exit_pct must be >= backtest.top_pct")
        return self


class PipelineConfig(_Strict):
    data: DataConfig
    labels: LabelsConfig = LabelsConfig()
    split: SplitConfig
    features: FeaturesConfig | None = None
    models: ModelsConfig | None = None
    ensemble: EnsembleConfig | None = None
    packaging: PackagingConfig | None = None
    backtest: BacktestConfig | None = None

    @model_validator(mode="after")
    def _labels_consistency(self) -> Self:
        has_path = self.data.labels_path is not None
        has_selected = self.labels.selected is not None
        if has_path and not has_selected:
            raise ValueError("labels.selected is required when data.labels_path is set")
        if has_selected and not has_path:
            raise ValueError("labels.selected given but data.labels_path is not set")
        return self


def load_config(path: str | Path) -> PipelineConfig:
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)
    return PipelineConfig.model_validate(raw)
