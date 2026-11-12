"""Unit tests for quantautoml.metrics.fdr."""

import warnings

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from quantautoml.common.exceptions import InvalidDataError
from quantautoml.metrics.fdr import apply_fdr_correction


@pytest.fixture(autouse=True)
def _no_warnings() -> None:
    warnings.simplefilter("error")


def test_hand_example() -> None:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6], dtype=np.float64)
    mask = apply_fdr_correction(p, alpha=0.05)
    assert mask.dtype == np.bool_
    # Thresholds i/M*alpha = 0.01, 0.02, 0.03, 0.04, 0.05 -> k = 2.
    assert mask.tolist() == [True, True, False, False, False]
    assert np.array_equal(mask, false_discovery_control(p) <= 0.05)


def test_matches_scipy_reference() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.02, 200), rng.uniform(0, 1, 800)])
    rng.shuffle(p)
    expected = false_discovery_control(p) <= 0.05
    assert np.array_equal(apply_fdr_correction(p, alpha=0.05), expected)


def test_preserves_original_order() -> None:
    p = np.array([0.6, 0.001, 0.9, 0.008], dtype=np.float64)
    assert apply_fdr_correction(p).tolist() == [False, True, False, True]


def test_empty() -> None:
    out = apply_fdr_correction(np.array([], dtype=np.float64))
    assert out.shape == (0,)
    assert out.dtype == np.bool_


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_invalid_p_values(bad: float) -> None:
    with pytest.raises(InvalidDataError):
        apply_fdr_correction(np.array([0.01, bad], dtype=np.float64))


def test_pure_noise_false_discovery_rate() -> None:
    rng = np.random.default_rng(42)
    rates = [apply_fdr_correction(rng.uniform(size=1000)).mean() for _ in range(200)]
    assert float(np.mean(rates)) <= 0.05
