"""Hysteresis Buffer Bands: enter on the top/bottom ``top_pct`` of the signal, exit later.

Single-asset adaptation (D2): the rank of the score at bar ``t`` is its percentile inside the
causal window of the last ``rank_window`` scores (ties: mid-rank; constant window -> 0.5).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

_CHUNK = 1024


class HysteresisBuffer:
    def __init__(
        self,
        top_pct: float,
        exit_pct: float,
        rank_window: int,
        *,
        signal_ema_span: int = 1,
        min_hold_bars: int = 0,
    ) -> None:
        if not 0.0 < top_pct <= exit_pct < 0.5:
            raise ValueError("require 0 < top_pct <= exit_pct < 0.5")
        if rank_window < 2:
            raise ValueError("rank_window must be >= 2")
        if signal_ema_span < 1:
            raise ValueError("signal_ema_span must be >= 1")
        if min_hold_bars < 0:
            raise ValueError("min_hold_bars must be >= 0")
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.rank_window = rank_window
        self.signal_ema_span = int(signal_ema_span)
        self.min_hold_bars = int(min_hold_bars)

    def smooth_scores(self, scores: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply causal EMA smoothing (span=1 returns input unchanged; NaN/Inf-safe)."""
        arr = np.asarray(scores, dtype=np.float32)
        if self.signal_ema_span == 1 or arr.size == 0:
            return arr
        alpha = 2.0 / (self.signal_ema_span + 1.0)
        one_minus = 1.0 - alpha
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        ema: float | None = None
        for t, val in enumerate(arr.astype(np.float64, copy=False)):
            if not np.isfinite(val):
                continue
            ema = float(val) if ema is None else alpha * float(val) + one_minus * ema
            out[t] = ema
        return out.astype(np.float32)

    def rolling_percentile(self, scores: NDArray[np.float32]) -> NDArray[np.float64]:
        """Percentile in [0, 1] of ``scores[t]`` within ``scores[t-w+1 : t+1]``; NaN before full."""
        w, n = self.rank_window, len(scores)
        out = np.full(n, np.nan)
        if n < w:
            return out
        windows = sliding_window_view(scores, w)  # row i ends at bar i + w - 1
        for start in range(0, windows.shape[0], _CHUNK):
            block = windows[start : start + _CHUNK]
            sl = slice(w - 1 + start, w - 1 + start + len(block))
            if np.isfinite(block).all():
                last = block[:, -1:]
                less = (block < last).sum(axis=1)
                equal = (block == last).sum(axis=1)
                out[sl] = (less + 0.5 * (equal - 1)) / (w - 1)
            else:
                finite = np.isfinite(block)
                last = block[:, -1:]
                last_ok = finite[:, -1]
                n_fin = finite.sum(axis=1)
                less = (finite & (block < last)).sum(axis=1)
                equal = (finite & (block == last)).sum(axis=1)
                valid = last_ok & (n_fin >= 2)
                chunk_out = np.full(len(block), np.nan, dtype=np.float64)
                chunk_out[valid] = (less[valid] + 0.5 * (equal[valid] - 1)) / (n_fin[valid] - 1)
                out[sl] = chunk_out
        return out

    def positions(self, scores: NDArray[np.float32]) -> NDArray[np.float64]:
        return self.positions_from_percentile(self.rolling_percentile(self.smooth_scores(scores)))

    def positions_from_percentile(self, pct: NDArray[np.float64]) -> NDArray[np.float64]:
        """State machine over percentiles: +1 long, -1 short, 0 flat (also while pct is NaN)."""
        enter_long, hold_long = 1.0 - self.top_pct, 1.0 - self.exit_pct
        enter_short, hold_short = self.top_pct, self.exit_pct
        pos = np.zeros(len(pct))
        state = 0.0
        held = 0
        for t, p in enumerate(pct):
            if np.isnan(p):
                state = 0.0
                held = 0
            elif state != 0.0 and held < self.min_hold_bars:
                held += 1
            else:
                if state == 1.0 and p < hold_long:
                    state = 0.0
                    held = 0
                elif state == -1.0 and p > hold_short:
                    state = 0.0
                    held = 0
                if state == 0.0:
                    if p >= enter_long:
                        state = 1.0
                        held = 1
                    elif p <= enter_short:
                        state = -1.0
                        held = 1
                else:
                    held += 1
            pos[t] = state
        return pos
