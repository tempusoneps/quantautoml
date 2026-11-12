"""Unit tests for quantautoml.metrics.ic."""

import warnings

import numpy as np
import pytest
from scipy.stats import spearmanr

from quantautoml.metrics.ic import calc_icir, calc_rank_ic, calc_rolling_ic


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
