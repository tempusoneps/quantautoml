"""HysteresisBuffer: causal rolling percentile + enter/exit bands on the signal rank."""

import numpy as np
import pytest

from quantautoml.backtest.buffer import HysteresisBuffer


def _buf(exit_pct: float = 0.30, window: int = 50) -> HysteresisBuffer:
    return HysteresisBuffer(top_pct=0.20, exit_pct=exit_pct, rank_window=window)


def test_percentile_exact_values() -> None:
    b = HysteresisBuffer(0.2, 0.3, rank_window=5)
    up = b.rolling_percentile(np.arange(1, 6, dtype=np.float32))
    down = b.rolling_percentile(np.arange(5, 0, -1, dtype=np.float32))
    assert up[-1] == 1.0 and down[-1] == 0.0
    assert np.isnan(up[:4]).all()


def test_percentile_ties_use_midrank_constant_is_half() -> None:
    b = HysteresisBuffer(0.2, 0.3, rank_window=10)
    pct = b.rolling_percentile(np.full(30, 0.7, dtype=np.float32))
    assert np.allclose(pct[9:], 0.5)


def test_percentile_is_causal() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(size=300).astype(np.float32)
    b = _buf()
    base = b.rolling_percentile(s)
    s2 = s.copy()
    s2[200:] = rng.normal(size=100).astype(np.float32) * 50
    again = b.rolling_percentile(s2)
    assert np.array_equal(base[:200], again[:200], equal_nan=True)


def test_zero_position_before_window_full() -> None:
    rng = np.random.default_rng(1)
    pos = _buf().positions(rng.normal(size=200).astype(np.float32))
    assert (pos[:49] == 0).all()


def test_hysteresis_holds_between_thresholds() -> None:
    pct = np.array([0.5, 0.85, 0.75, 0.65, 0.5])
    np.testing.assert_array_equal(_buf().positions_from_percentile(pct), [0, 1, 1, 0, 0])
    pct = np.array([0.5, 0.15, 0.25, 0.35, 0.5])
    np.testing.assert_array_equal(_buf().positions_from_percentile(pct), [0, -1, -1, 0, 0])


def test_nan_percentile_is_flat() -> None:
    pct = np.array([np.nan, 0.9, np.nan, 0.9])
    np.testing.assert_array_equal(_buf().positions_from_percentile(pct), [0, 1, 0, 1])


def test_buffer_reduces_trades_vs_no_buffer() -> None:
    rng = np.random.default_rng(2)
    slow = np.convolve(rng.normal(size=3200), np.ones(60) / 60, mode="valid")[:3000]
    s = (slow / slow.std() + 0.25 * rng.normal(size=3000)).astype(np.float32)  # rank wobbles
    changes = lambda p: int(np.count_nonzero(np.diff(p)))  # noqa: E731
    buffered = changes(_buf(exit_pct=0.30).positions(s))
    plain = changes(_buf(exit_pct=0.20).positions(s))
    assert buffered <= 0.8 * plain


def test_exit_equals_top_is_memoryless() -> None:
    pct = np.array([0.85, 0.75, 0.85, 0.1, 0.5])
    np.testing.assert_array_equal(_buf(exit_pct=0.20).positions_from_percentile(pct), [1, 0, 1, -1, 0])


def test_constant_scores_never_trade() -> None:
    pos = _buf().positions(np.full(500, 1.0, dtype=np.float32))
    assert (pos == 0).all() and np.isfinite(pos).all()


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValueError):
        HysteresisBuffer(top_pct=0.3, exit_pct=0.2, rank_window=10)
    with pytest.raises(ValueError):
        HysteresisBuffer(top_pct=0.2, exit_pct=0.3, rank_window=1)
