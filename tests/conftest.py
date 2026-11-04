"""Shared pytest fixtures for the quantautoml test-suite."""

from pathlib import Path

import pytest

VALID_TOML = """
[data]
features_path = "datasets/features.parquet"
labels_path = "datasets/labels.parquet"

[labels]
selected = "fwd_ret_5"

[split]
horizon_h = 21
"""


@pytest.fixture
def valid_toml() -> str:
    return VALID_TOML


@pytest.fixture
def write_toml(tmp_path: Path):
    def _write(content: str) -> Path:
        path = tmp_path / "config.toml"
        path.write_text(content, encoding="utf-8")
        return path

    return _write


# --------------------------------------------------------------------------- #
# Synthetic market data helpers (Phase 1)
# --------------------------------------------------------------------------- #
from collections.abc import Callable  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from quantautoml.common.config import PipelineConfig  # noqa: E402

HORIZONS = (1, 5, 21)


@pytest.fixture
def make_frames() -> Callable[..., tuple[pl.DataFrame, pl.DataFrame]]:
    """Build (features, labels) frames with forward-looking labels.

    Labels: ``y_dir`` (binary direction of fwd_ret_5, trailing 5 NaN) as the
    default selected label, plus ``fwd_ret_{1,5,21}`` (``shift(-k)``, k trailing NaN).
    """

    def _make(
        n: int = 200, n_features: int = 3, seed: int = 0
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        rng = np.random.default_rng(seed)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = pl.datetime_range(
            start, start + timedelta(hours=n - 1), interval="1h",
            time_unit="ns", time_zone="UTC", eager=True,
        ).alias("timestamp")
        logp = np.cumsum(rng.normal(0.0, 0.01, n))
        feats = {f"f{i}": rng.normal(size=n) for i in range(n_features)}
        labels: dict[str, np.ndarray] = {}
        for k in HORIZONS:
            fwd = np.full(n, np.nan)
            fwd[: n - k] = logp[k:] - logp[: n - k]
            labels[f"fwd_ret_{k}"] = fwd
        y_dir = np.where(np.isnan(labels["fwd_ret_5"]), np.nan,
                         (labels["fwd_ret_5"] > 0).astype(float))
        features_df = pl.DataFrame([ts, *[pl.Series(k, v) for k, v in feats.items()]])
        labels_df = pl.DataFrame(
            [ts, pl.Series("y_dir", y_dir),
             *[pl.Series(k, v) for k, v in labels.items()]]
        )
        return features_df, labels_df

    return _make


@pytest.fixture
def build_config(tmp_path: Path) -> Callable[..., PipelineConfig]:
    """Write frames to disk and return a PipelineConfig pointing at them."""

    def _build(
        features: pl.DataFrame,
        labels: pl.DataFrame | None,
        horizon_h: int = 21,
        selected: str | None = "y_dir",
        fmt: str = "parquet",
        min_match_ratio: float = 0.95,
    ) -> PipelineConfig:
        def _write(df: pl.DataFrame, name: str) -> Path:
            path = tmp_path / f"{name}.{fmt}"
            if fmt == "parquet":
                df.write_parquet(path)
            else:
                df.write_csv(path)
            return path

        data: dict[str, object] = {
            "features_path": str(_write(features, "features")),
            "min_match_ratio": min_match_ratio,
        }
        raw: dict[str, object] = {"data": data, "split": {"horizon_h": horizon_h}}
        if labels is not None:
            data["labels_path"] = str(_write(labels, "labels"))
            raw["labels"] = {"selected": selected}
        return PipelineConfig.model_validate(raw)

    return _build
