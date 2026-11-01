"""Pipeline configuration loaded from a TOML file and validated with Pydantic."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class PipelineConfig(_Strict):
    data: DataConfig
    labels: LabelsConfig = LabelsConfig()
    split: SplitConfig

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
