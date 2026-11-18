"""FeatureFunnel.preprocessor_spec() reproduces funnel.transform exactly."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_funnel import _cfg, _dataset, _split

from quantautoml.common.types import PreprocessorSpec
from quantautoml.features import FeatureFunnel


def apply_spec_numpy(spec: PreprocessorSpec, X: np.ndarray) -> np.ndarray:
    """Reference implementation of the preprocessing semantics (test-only)."""
    Xc = np.where(np.isinf(X), np.nan, X).astype(np.float32)
    ind = np.isnan(Xc[:, list(spec.indicator_idx)]).astype(np.float32)
    for j, fill in spec.bool_fill:
        Xc[:, j] = np.where(np.isnan(Xc[:, j]), np.float32(fill), Xc[:, j])
    t = np.hstack([Xc[:, list(spec.keep_idx)], ind]).astype(np.float32)
    return t[:, list(spec.final_idx)]


def dirty(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Inject +/-inf and NaN into every column of a copy of X."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    mask = rng.random(out.shape)
    out[mask < 0.02] = np.inf
    out[(mask >= 0.02) & (mask < 0.04)] = -np.inf
    out[(mask >= 0.04) & (mask < 0.07)] = np.nan
    return out


@pytest.fixture(scope="module")
def fitted():
    ds = _dataset()
    split = _split(ds)
    return FeatureFunnel(_cfg()).fit(split, ds), split


def test_spec_reproduces_funnel_transform_on_clean_data(fitted) -> None:
    funnel, split = fitted
    spec = funnel.preprocessor_spec()
    assert spec.n_in == split.val.X.shape[1]
    assert spec.output_names == funnel.selected_features_
    expected = funnel.transform(split.val).X
    assert np.array_equal(apply_spec_numpy(spec, split.val.X), expected, equal_nan=True)


def test_spec_reproduces_funnel_transform_with_inf_nan(fitted) -> None:
    funnel, split = fitted
    spec = funnel.preprocessor_spec()
    X = dirty(split.val.X)
    expected = funnel.transform(replace(split.val, X=X)).X
    assert np.array_equal(apply_spec_numpy(spec, X), expected, equal_nan=True)


def test_spec_before_fit_raises() -> None:
    from quantautoml.common.exceptions import FeatureSelectionError

    with pytest.raises(FeatureSelectionError):
        FeatureFunnel(_cfg()).preprocessor_spec()


def test_empty_selection_spec_has_no_output() -> None:
    ds = _dataset(constant=True)
    funnel = FeatureFunnel(_cfg()).fit(_split(ds), ds)
    spec = funnel.preprocessor_spec()
    assert spec.final_idx == () and spec.output_names == ()
