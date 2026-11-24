"""Pipeline configuration loaded from a TOML file and validated with Pydantic."""

from __future__ import annotations

import math
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def parse_iso_bound(value: str, *, end_of_day: bool = False) -> datetime:
    """Parse an ISO-8601 date or datetime string into a UTC-aware ``datetime``.

    A date-only string (``YYYY-MM-DD``) becomes ``00:00:00+00:00`` when ``end_of_day=False``
    or ``23:59:59.999999+00:00`` when ``end_of_day=True``. Naive datetimes are assumed UTC.
    """
    s = value.strip()
    if not s:
        raise ValueError("date bound must not be empty")
    is_date_only = len(s) == 10 and "T" not in s and "t" not in s and " " not in s
    dt = datetime.fromisoformat(s)
    if is_date_only and end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999_999)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_Strict):
    features_path: Path
    labels_path: Path | None = None
    timestamp_col: str = "timestamp"
    min_match_ratio: float = Field(default=0.95, gt=0.0, le=1.0)
    date_from: str | None = None
    date_to: str | None = None

    @model_validator(mode="after")
    def _validate_date_range(self) -> Self:
        start = parse_iso_bound(self.date_from, end_of_day=False) if self.date_from is not None else None
        end = parse_iso_bound(self.date_to, end_of_day=True) if self.date_to is not None else None
        if start is not None and end is not None and start > end:
            raise ValueError(f"data.date_from ({self.date_from!r}) must be <= data.date_to ({self.date_to!r})")
        return self


class LabelsConfig(_Strict):
    selected: str | None = None
    # True: ingestion keeps only the selected label column and ignores every other
    # label column (they are neither type-checked nor exposed as labels_aux).
    ignore_other_columns: bool = False
    # Text label -> directional number (e.g. {"Long": 1, "Short": -1, "None": 0}).
    # Several texts may share a number (they merge into one class). Applied in ingestion;
    # `classes_` then holds the mapped numbers in ascending order.
    mapping: dict[str, float] | None = None

    @field_validator("mapping")
    @classmethod
    def _mapping_is_usable(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is not None:
            if not v:
                raise ValueError("labels.mapping must not be empty")
            if not all(math.isfinite(x) for x in v.values()):
                raise ValueError("labels.mapping values must be finite numbers")
        return v


class SplitConfig(_Strict):
    horizon_h: int = Field(ge=1)


class FeaturesConfig(_Strict):
    ic_horizons: dict[int, str]
    target_horizon: int
    # Ingestion derives any ic_horizons column missing from the labels file as
    # fwd_ret_h = price[t+h] / price[t] - 1 using this feature column.
    price_col: str = "Close"
    icir_min: float = 0.30
    fdr_alpha: float = Field(default=0.05, gt=0.0, le=0.05)
    rolling_window: int = Field(default=63, ge=2)
    dcor_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    cluster_distance: float = Field(default=0.30, gt=0.0, lt=1.0)
    max_nan_ratio: float = Field(default=0.6, gt=0.0, le=1.0)
    min_unique_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    max_top_freq_ratio: float = Field(default=0.995, gt=0.0, le=1.0)
    min_cv: float = Field(default=1e-6, ge=0.0)
    max_peak_var_ratio: float | None = Field(default=None, gt=0.0)
    max_temporal_drift: float | None = Field(default=1.5, gt=0.0)
    max_lag1_autocorr: float = Field(default=0.995, gt=0.0, lt=1.0)
    stationarity_blocks: int = Field(default=4, ge=2)
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
    bootstrap_block_size: int = Field(default=20, ge=1)
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
    open_col: str = "Open"
    execution_price: Literal["close", "next_open"] = "close"
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


class RealWorldConfig(_Strict):
    enabled: bool = False
    cash: float = Field(default=1_000_000.0, gt=0.0)
    commission: float = Field(default=0.0005, ge=0.0)
    margin: float = Field(default=1.0, gt=0.0)
    size: str | int | float = "auto"
    trade_on_close: bool = False
    out_html: str | None = None
    out_trades_csv: str | None = None
    ohlcv_open_col: str = "Open"
    ohlcv_high_col: str = "High"
    ohlcv_low_col: str = "Low"
    ohlcv_close_col: str = "Close"
    ohlcv_volume_col: str = "Volume"

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: str | int | float) -> str | int | float:
        if isinstance(v, bool):
            raise ValueError("realworld.size cannot be a boolean")
        if isinstance(v, str):
            s = v.strip()
            if s in ("auto", "start_cash", "_start_cash"):
                return v
            try:
                num = float(s)
            except ValueError as exc:
                raise ValueError(
                    "realworld.size must be 'auto', 'start_cash' ('_start_cash'), "
                    "a positive integer unit count (>= 1), or an equity fraction in (0, 1)"
                ) from exc
            if num <= 0 or (num >= 1.0 and not num.is_integer()):
                raise ValueError(
                    "numeric realworld.size must be an integer >= 1 or a fraction in (0, 1)"
                )
            return v
        num_val = float(v)
        if not math.isfinite(num_val) or num_val <= 0:
            raise ValueError("realworld.size must be > 0")
        if num_val >= 1.0 and not num_val.is_integer():
            raise ValueError(
                "realworld.size >= 1 must be a whole integer number of units/contracts"
            )
        return v


class PipelineConfig(_Strict):
    data: DataConfig
    labels: LabelsConfig = LabelsConfig()
    split: SplitConfig
    features: FeaturesConfig | None = None
    models: ModelsConfig | None = None
    ensemble: EnsembleConfig | None = None
    packaging: PackagingConfig | None = None
    backtest: BacktestConfig | None = None
    realworld: RealWorldConfig | None = None

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
