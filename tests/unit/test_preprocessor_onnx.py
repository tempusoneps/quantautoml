"""preprocessor.onnx reproduces FeatureFunnel.transform bit-for-bit (incl. inf/NaN rows)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_funnel import _cfg, _dataset, _split
from test_preprocessor_spec import dirty

from quantautoml.features import FeatureFunnel
from quantautoml.packaging.preprocessor import build_preprocessor_onnx, run_preprocessor_onnx


@pytest.fixture(scope="module")
def fitted():
    ds = _dataset()
    split = _split(ds)
    return FeatureFunnel(_cfg()).fit(split, ds), split


def test_onnx_matches_funnel_transform_clean(fitted, tmp_path: Path) -> None:
    funnel, split = fitted
    path = tmp_path / "preprocessor.onnx"
    build_preprocessor_onnx(funnel.preprocessor_spec(), path)
    out = run_preprocessor_onnx(path, split.val.X)
    expected = funnel.transform(split.val).X
    assert out.dtype == np.float32 and out.shape == expected.shape
    assert np.array_equal(out, expected, equal_nan=True)


def test_onnx_matches_funnel_transform_with_inf_nan(fitted, tmp_path: Path) -> None:
    funnel, split = fitted
    path = tmp_path / "preprocessor.onnx"
    build_preprocessor_onnx(funnel.preprocessor_spec(), path)
    X = dirty(split.val.X, seed=3)
    out = run_preprocessor_onnx(path, X)
    expected = funnel.transform(replace(split.val, X=X)).X
    assert np.array_equal(out, expected, equal_nan=True)


def test_empty_selection_gives_n_by_0(tmp_path: Path) -> None:
    ds = _dataset(constant=True)
    funnel = FeatureFunnel(_cfg()).fit(_split(ds), ds)
    path = tmp_path / "preprocessor.onnx"
    build_preprocessor_onnx(funnel.preprocessor_spec(), path)
    out = run_preprocessor_onnx(path, ds.X[:5])
    assert out.shape == (5, 0) and out.dtype == np.float32


def test_single_row_input(fitted, tmp_path: Path) -> None:
    funnel, split = fitted
    path = tmp_path / "preprocessor.onnx"
    build_preprocessor_onnx(funnel.preprocessor_spec(), path)
    out = run_preprocessor_onnx(path, split.val.X[:1])
    assert out.shape[0] == 1
