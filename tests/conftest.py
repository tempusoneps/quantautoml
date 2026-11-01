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
