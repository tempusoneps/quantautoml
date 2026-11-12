"""Unit tests for quantautoml.metrics.ic."""

import warnings

import numpy as np
import pytest
from scipy.stats import ConstantInputWarning, spearmanr

from quantautoml.common.exceptions import InvalidDataError
from quantautoml.metrics.ic import (
    calc_icir,
    calc_rank_ic,
    calc_rolling_ic,
    calc_rolling_ic_matrix,
)


@pytest.fixture(autouse=True)
def _no_warnings() -> None:
    warnings.simplefilter("error")


def _rand(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).astype(np.float32)
    y = (0.3 * x + rng.normal(size=n)).astype(np.float32)
    return x, y


def test_perfect_positive_and_negative() -> None:
    x, _ = _rand()
    assert calc_rank_ic(x, x)[0] == pytest.approx(1.0)
    assert calc_rank_ic(x, -x)[0] == pytest.approx(-1.0)


def test_matches_scipy() -> None:
    x, y = _rand()
    rho, p = calc_rank_ic(x, y)
    ref = spearmanr(x, y)
    assert rho == pytest.approx(ref.statistic, abs=1e-9)
    assert p == pytest.approx(ref.pvalue, abs=1e-9)


def test_ignore_value_equals_row_drop() -> None:
    x, y = _rand()
    y = y.copy()
    y[::7] = -1.0
    keep = y != -1.0
    assert calc_rank_ic(x, y, ignore_value=-1.0) == calc_rank_ic(x[keep], y[keep])


def test_nan_pairs_dropped() -> None:
    x, y = _rand()
    x = x.copy()
    x[:10] = np.nan
    assert calc_rank_ic(x, y) == calc_rank_ic(x[10:], y[10:])


def test_degenerate_inputs() -> None:
    x, y = _rand()
    assert calc_rank_ic(np.ones_like(x), y) == (0.0, 1.0)
    assert calc_rank_ic(x, np.ones_like(y)) == (0.0, 1.0)
    assert calc_rank_ic(np.full(50, np.nan, dtype=np.float32), y[:50]) == (0.0, 1.0)
    assert calc_rank_ic(x[:2], y[:2]) == (0.0, 1.0)
    assert calc_rank_ic(x[:0], y[:0]) == (0.0, 1.0)


def test_int_labels_with_ignore() -> None:
    x, _ = _rand(100)
    y = np.where(x > 0, 1, 0).astype(np.int32)
    y[:5] = -1
    rho, _ = calc_rank_ic(x, y, ignore_value=-1)
    assert rho > 0.5


def test_rolling_ic_length_and_constant_window() -> None:
    x, y = _rand(200)
    x = x.copy()
    x[63:126] = 1.0
    out = calc_rolling_ic(x, y, 63)
    assert out.shape == (200 // 63,)
    assert out.dtype == np.float64
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])
    assert out[0] == pytest.approx(calc_rank_ic(x[:63], y[:63])[0])


def test_rolling_ic_short_series_returns_empty() -> None:
    x, y = _rand(10)
    assert calc_rolling_ic(x, y, 63).shape == (0,)


def test_rolling_ic_ignore_value() -> None:
    x, y = _rand(126)
    y = y.copy()
    y[:5] = -1.0
    out = calc_rolling_ic(x, y, 63, ignore_value=-1.0)
    assert out[0] == pytest.approx(calc_rank_ic(x[5:63], y[5:63])[0])


def test_rolling_ic_invalid_window() -> None:
    from quantautoml.common.exceptions import InvalidDataError

    x, y = _rand(10)
    with pytest.raises(InvalidDataError):
        calc_rolling_ic(x, y, 0)


def test_icir() -> None:
    s = np.array([0.1, 0.2, np.nan, 0.3])
    assert calc_icir(s) == pytest.approx(0.2 / np.std([0.1, 0.2, 0.3], ddof=1))
    assert calc_icir(np.full(5, 0.1)) == 0.0
    assert calc_icir(np.array([0.1])) == 0.0
    assert calc_icir(np.array([np.nan, np.nan])) == 0.0
    assert calc_icir(np.array([], dtype=np.float64)) == 0.0


def test_rolling_ic_matrix_matches_reference() -> None:
    rng = np.random.default_rng(12)
    n = 63 * 5 + 10
    X = rng.normal(size=(n, 6)).astype(np.float32)
    lab = (0.5 * X[:, 0] + rng.normal(size=n)).astype(np.float32)
    lab[-30:] = np.nan
    X[5:20, 1] = np.nan  # NaN in a feature
    X[63:126, 2] = 1.0  # constant window -> NaN window IC
    X[:, 3] = np.round(X[:, 3])  # heavy ties
    X[:, 4] = rng.normal(size=n)
    X[:63, 4] = np.inf  # all-inf window (reference -> 0.0 IC, not NaN)
    X[:, 5] = rng.normal(size=n)
    X[70:75, 5] = -np.inf  # partial -inf
    X[200:210, 5] = np.inf
    lab[130:133] = np.inf  # +inf in labels of window 2
    # On an all-inf window calc_rank_ic itself emits a numpy 'invalid value' from np.ptp
    # and a scipy ConstantInputWarning before returning 0.0; silence only those two here.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        got = calc_rolling_ic_matrix(X, lab, 63)
        refs = [calc_rolling_ic(X[:, j], lab, 63) for j in range(6)]
    assert got.shape == (n // 63, 6) and got.dtype == np.float64
    for j, ref in enumerate(refs):
        np.testing.assert_allclose(got[:, j], ref, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_rolling_ic_matrix_ignore_value_matches_reference() -> None:
    rng = np.random.default_rng(13)
    n = 63 * 3
    X = rng.normal(size=(n, 3)).astype(np.float32)
    lab = (0.4 * X[:, 0] + rng.normal(size=n)).astype(np.float32)
    lab[::5] = -1.0
    got = calc_rolling_ic_matrix(X, lab, 63, ignore_value=-1.0)
    for j in range(3):
        ref = calc_rolling_ic(X[:, j], lab, 63, ignore_value=-1.0)
        np.testing.assert_allclose(got[:, j], ref, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_rolling_ic_matrix_short_and_invalid_window() -> None:
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    assert calc_rolling_ic_matrix(X, y, 63).shape == (0, 2)
    with pytest.raises(InvalidDataError):
        calc_rolling_ic_matrix(X, y, 0)
