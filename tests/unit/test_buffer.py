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
    with pytest.raises(ValueError):
        HysteresisBuffer(top_pct=0.2, exit_pct=0.3, rank_window=10, signal_ema_span=0)
    with pytest.raises(ValueError):
        HysteresisBuffer(top_pct=0.2, exit_pct=0.3, rank_window=10, min_hold_bars=-1)


def test_smooth_scores_identity_when_span_one_and_empty() -> None:
    b = HysteresisBuffer(0.2, 0.3, rank_window=10, signal_ema_span=1)
    s = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    out = b.smooth_scores(s)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, s)
    empty = b.smooth_scores(np.empty(0, dtype=np.float32))
    assert empty.shape == (0,) and empty.dtype == np.float32


def test_smooth_scores_causal_and_nan_inf_safe() -> None:
    rng = np.random.default_rng(11)
    s = rng.normal(size=120).astype(np.float32)
    s[30] = np.nan
    s[55] = np.inf
    s[70] = -np.inf
    b = HysteresisBuffer(0.2, 0.3, rank_window=20, signal_ema_span=5)
    sm = b.smooth_scores(s)
    assert sm.dtype == np.float32
    assert np.isnan(sm[30]) and np.isnan(sm[55]) and np.isnan(sm[70])
    # Subsequent finite bars must not be poisoned by NaN/Inf
    assert np.isfinite(sm[31:55]).all()
    assert np.isfinite(sm[56:70]).all()
    assert np.isfinite(sm[71:]).all()

    # Causality: mutating future bars cannot alter smooth[:80]
    s2 = s.copy()
    s2[80:] = 999.0
    sm2 = b.smooth_scores(s2)
    assert np.array_equal(sm[:80], sm2[:80], equal_nan=True)


def test_rolling_percentile_nan_safe_no_false_short() -> None:
    b = HysteresisBuffer(0.2, 0.3, rank_window=10, signal_ema_span=1)
    s = np.linspace(0.0, 1.0, 30, dtype=np.float32)
    s[15] = np.nan
    s[20] = np.inf
    pct = b.rolling_percentile(s)
    assert np.isnan(pct[15]) and np.isnan(pct[20])
    assert np.isfinite(pct[16:20]).all()
    pos = b.positions(s)
    # NaN/Inf bar must go flat (0.0), never open a spurious Short (-1.0)
    assert pos[15] == 0.0 and pos[20] == 0.0


def test_min_hold_bars_exact_boundary_and_nan_reset() -> None:
    # Enter Long at t=1 (pct=0.85), then immediately drop to Short zone (0.10) at t=2..6.
    pct = np.array([0.50, 0.85, 0.10, 0.10, 0.10, 0.10, 0.10, 0.50])
    # min_hold_bars=5: holds bars t=1,2,3,4,5 (5 bars), earliest exit/reversal at t=6.
    b5 = HysteresisBuffer(0.2, 0.3, rank_window=10, min_hold_bars=5)
    np.testing.assert_array_equal(
        b5.positions_from_percentile(pct),
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
    )
    # min_hold_bars=1 matches min_hold_bars=0
    b0 = HysteresisBuffer(0.2, 0.3, rank_window=10, min_hold_bars=0)
    b1 = HysteresisBuffer(0.2, 0.3, rank_window=10, min_hold_bars=1)
    np.testing.assert_array_equal(
        b0.positions_from_percentile(pct), b1.positions_from_percentile(pct)
    )
    # NaN percentile forces flat and resets held counter even during min_hold_bars
    pct_nan = np.array([0.50, 0.85, np.nan, 0.50])
    np.testing.assert_array_equal(b5.positions_from_percentile(pct_nan), [0.0, 1.0, 0.0, 0.0])


def test_ema_and_min_hold_reduce_turnover_and_preserve_causality() -> None:
    rng = np.random.default_rng(21)
    slow = np.convolve(rng.normal(size=1600), np.ones(40) / 40, mode="valid")[:1500]
    s = (slow / slow.std() + 0.4 * rng.normal(size=1500)).astype(np.float32)

    base_buf = HysteresisBuffer(0.2, 0.3, rank_window=100)
    smooth_buf = HysteresisBuffer(
        0.2, 0.3, rank_window=100, signal_ema_span=5, min_hold_bars=5
    )
    p_base = base_buf.positions(s)
    p_smooth = smooth_buf.positions(s)

    changes = lambda p: int(np.count_nonzero(np.diff(p)))  # noqa: E731
    assert changes(p_smooth) < changes(p_base)

    # Strict causality: mutating future scores cannot alter positions[:900]
    s2 = s.copy()
    s2[900:] = rng.normal(size=600).astype(np.float32) * 50.0
    p_smooth2 = smooth_buf.positions(s2)
    np.testing.assert_array_equal(p_smooth[:900], p_smooth2[:900])
