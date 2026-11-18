"""FrictionCostModel: R_net = R_gross - Turnover * (fee + slippage), Turnover = 0.5*sum|dw|."""

import numpy as np
import pytest

from quantautoml.backtest.friction import FrictionCostModel


def test_zero_cost_equals_gross() -> None:
    gross = np.array([0.01, -0.02, 0.03])
    pos = np.array([1.0, -1.0, 1.0])
    np.testing.assert_array_equal(FrictionCostModel(0.0, 0.0).net_returns(gross, pos), gross)


def test_known_turnover() -> None:
    pos = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    np.testing.assert_allclose(
        FrictionCostModel(4.0, 1.0).turnover(pos), [0.0, 0.5, 0.0, 1.0, 0.5], atol=1e-12
    )


def test_known_net_returns() -> None:
    pos = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    gross = np.array([0.0, 0.01, 0.01, 0.01, 0.0])
    net = FrictionCostModel(4.0, 1.0).net_returns(gross, pos)
    cost = 5e-4  # (4 + 1) bps
    np.testing.assert_allclose(net, gross - np.array([0.0, 0.5, 0.0, 1.0, 0.5]) * cost, atol=1e-12)


def test_first_bar_entry_pays_cost() -> None:
    net = FrictionCostModel(10.0, 0.0).net_returns(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert net[0] == pytest.approx(-0.5 * 10e-4)
    assert net[1] == 0.0


def test_net_never_exceeds_gross_when_costs_positive() -> None:
    rng = np.random.default_rng(0)
    pos = rng.choice([-1.0, 0.0, 1.0], size=500)
    gross = rng.normal(0, 0.01, size=500)
    net = FrictionCostModel(4.0, 1.0).net_returns(gross, pos)
    assert np.all(net <= gross + 1e-15)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        FrictionCostModel(1.0, 1.0).net_returns(np.zeros(3), np.zeros(4))


def test_negative_cost_rejected() -> None:
    with pytest.raises(ValueError):
        FrictionCostModel(-1.0, 0.0)
    with pytest.raises(ValueError):
        FrictionCostModel(0.0, -0.1)
