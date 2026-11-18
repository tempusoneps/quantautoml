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
    def __init__(self, top_pct: float, exit_pct: float, rank_window: int) -> None:
        if not 0.0 < top_pct <= exit_pct < 0.5:
            raise ValueError("require 0 < top_pct <= exit_pct < 0.5")
        if rank_window < 2:
            raise ValueError("rank_window must be >= 2")
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.rank_window = rank_window

    def rolling_percentile(self, scores: NDArray[np.float32]) -> NDArray[np.float64]:
        """Percentile in [0, 1] of ``scores[t]`` within ``scores[t-w+1 : t+1]``; NaN before full."""
        w, n = self.rank_window, len(scores)
        out = np.full(n, np.nan)
        if n < w:
            return out
        windows = sliding_window_view(scores, w)  # row i ends at bar i + w - 1
        for start in range(0, windows.shape[0], _CHUNK):
            block = windows[start : start + _CHUNK]
            last = block[:, -1:]
            less = (block < last).sum(axis=1)
            equal = (block == last).sum(axis=1)
            out[w - 1 + start : w - 1 + start + len(block)] = (less + 0.5 * (equal - 1)) / (w - 1)
        return out

    def positions(self, scores: NDArray[np.float32]) -> NDArray[np.float64]:
        return self.positions_from_percentile(self.rolling_percentile(scores))

    def positions_from_percentile(self, pct: NDArray[np.float64]) -> NDArray[np.float64]:
        """State machine over percentiles: +1 long, -1 short, 0 flat (also while pct is NaN)."""
        enter_long, hold_long = 1.0 - self.top_pct, 1.0 - self.exit_pct
        enter_short, hold_short = self.top_pct, self.exit_pct
        pos = np.zeros(len(pct))
        state = 0.0
        for t, p in enumerate(pct):
            if np.isnan(p):
                state = 0.0
            else:
                if state == 1.0 and p < hold_long:
                    state = 0.0
                elif state == -1.0 and p > hold_short:
                    state = 0.0
                if state == 0.0:
                    if p >= enter_long:
                        state = 1.0
                    elif p <= enter_short:
                        state = -1.0
            pos[t] = state
        return pos
