"""Shared quantile binning and contingency-table MI used by F6 and F8."""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
import pytest
from sklearn.metrics import mutual_info_score

from quantautoml.features._binning import contingency_mi, quantile_bin_codes


@pytest.fixture(autouse=True)
def _warnings_are_errors() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_contingency_mi_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    for na, nc in ((10, 2), (100, 3), (7, 5)):
        a = rng.integers(0, na - 2, 3000)  # leaves empty top codes on purpose
        yc = rng.integers(0, nc, 3000)
        yc[a < 2] = 0  # dependence
        got = contingency_mi(a.astype(np.intp), na, yc.astype(np.intp), nc)
        assert got == pytest.approx(float(mutual_info_score(a, yc)), abs=1e-12)
    one = np.zeros(50, dtype=np.intp)
    assert contingency_mi(one, 10, rng.integers(0, 2, 50).astype(np.intp), 2) == 0.0
    assert contingency_mi(np.empty(0, np.intp), 10, np.empty(0, np.intp), 2) == 0.0


def test_contingency_mi_single_class_is_zero() -> None:
    a = np.random.default_rng(2).integers(0, 5, 200).astype(np.intp)
    assert contingency_mi(a, 5, np.zeros(200, dtype=np.intp), 2) == 0.0


@pytest.mark.parametrize("n_bins", [5, 10])
def test_quantile_bin_codes_match_reference(n_bins: int) -> None:
    x = np.random.default_rng(3).normal(size=1000).astype(np.float32)
    codes, nb = quantile_bin_codes(x[:, None], n_bins)
    x64 = x.astype(np.float64)
    edges = np.unique(np.quantile(x64, np.linspace(0, 1, n_bins + 1)[1:-1]))
    np.testing.assert_array_equal(codes[:, 0], np.searchsorted(edges, x64, side="right"))
    assert codes.dtype == np.intp and nb.dtype == np.intp
    assert nb.tolist() == [n_bins] and set(codes[:, 0].tolist()) == set(range(n_bins))


def test_quantile_bin_codes_degenerate_columns() -> None:
    n = 100
    inf_col = np.arange(n, dtype=np.float32)
    inf_col[::4] = np.inf
    nan_col = np.arange(n, dtype=np.float32)
    nan_col[:10] = np.nan
    X = np.column_stack(
        [np.full(n, 2.0), np.full(n, np.nan), inf_col, nan_col, np.repeat([0.0, 1.0], n // 2)]
    ).astype(np.float32)
    codes, nb = quantile_bin_codes(X, 5)
    assert len(set(codes[:, 0].tolist())) == 1  # constant -> one non-empty bin
    np.testing.assert_array_equal(codes[:, 1], 0)  # all-NaN -> all code 0
    assert nb[1] == 1
    # +-inf -> NaN -> Train median of the finite values
    med = float(np.median(inf_col[np.isfinite(inf_col)]))
    filled = np.where(np.isfinite(inf_col), inf_col, med).astype(np.float64)
    edges = np.unique(np.quantile(filled, np.linspace(0, 1, 6)[1:-1]))
    np.testing.assert_array_equal(codes[:, 2], np.searchsorted(edges, filled, side="right"))
    assert len(set(codes[:10, 3].tolist())) == 1  # NaN rows share the median's bin
    assert nb[4] == 3 and len(set(codes[:, 4].tolist())) == 2  # duplicate edges collapse
    assert quantile_bin_codes(np.empty((0, 2), np.float32), 5)[0].shape == (0, 2)
