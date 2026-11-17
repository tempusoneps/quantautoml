import numpy as np
import pytest

from quantautoml.metrics.ic import calc_icir


def test_icir_regular_series_unchanged() -> None:
    v = np.array([0.1, 0.3, 0.2, 0.4, 0.0])
    assert calc_icir(v) == pytest.approx(np.mean(v) / np.std(v, ddof=1))


@pytest.mark.parametrize(
    "series",
    [
        np.full(10, 0.07),
        np.full(10, 0.07) + np.arange(10) * 1e-17,  # float noise only
        np.array([0.05, np.nan]),
        np.array([np.nan, np.nan]),
    ],
)
def test_icir_degenerate_series_returns_zero(series: np.ndarray) -> None:
    assert calc_icir(series) == 0.0


def test_icir_small_but_real_variation_not_zeroed() -> None:
    v = np.array([0.001, 0.002, 0.0015, 0.0012])
    assert calc_icir(v) != 0.0
