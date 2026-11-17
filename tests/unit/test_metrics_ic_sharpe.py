import numpy as np
import pytest

from quantautoml.common.exceptions import InvalidDataError
from quantautoml.metrics.ic import calc_ic_sharpe


def test_calc_ic_sharpe_basic() -> None:
    # ic series with known mean and std
    ics = np.array([0.05, 0.07, 0.03, 0.05], dtype=np.float64)
    # mean = 0.05
    # deviations: -0.02, 0.02, 0.0 -> std(ddof=1) = sqrt((0.0004 + 0.0004 + 0)/3) = sqrt(0.0008/3)
    mean = np.mean(ics)
    std = np.std(ics, ddof=1)
    expected_ir = mean / std
    assert calc_ic_sharpe(ics) == pytest.approx(expected_ir)
    assert calc_ic_sharpe(ics, annualization_factor=252.0) == pytest.approx(expected_ir * np.sqrt(252.0))


def test_calc_ic_sharpe_degenerate_cases() -> None:
    # Constant array -> std is 0 -> returns 0.0
    assert calc_ic_sharpe(np.array([0.05, 0.05, 0.05])) == 0.0
    # Single element -> < 2 points -> returns 0.0
    assert calc_ic_sharpe(np.array([0.05])) == 0.0
    # Empty array -> returns 0.0
    assert calc_ic_sharpe(np.array([])) == 0.0
    # All NaNs -> returns 0.0
    assert calc_ic_sharpe(np.array([np.nan, np.nan])) == 0.0


def test_calc_ic_sharpe_ignores_nan() -> None:
    ics = np.array([0.05, np.nan, 0.07, 0.03, 0.05], dtype=np.float64)
    clean = np.array([0.05, 0.07, 0.03, 0.05], dtype=np.float64)
    assert calc_ic_sharpe(ics) == pytest.approx(calc_ic_sharpe(clean))


def test_calc_ic_sharpe_invalid_annualization() -> None:
    with pytest.raises(InvalidDataError):
        calc_ic_sharpe(np.array([0.05, 0.07]), annualization_factor=0.0)
    with pytest.raises(InvalidDataError):
        calc_ic_sharpe(np.array([0.05, 0.07]), annualization_factor=-1.0)
