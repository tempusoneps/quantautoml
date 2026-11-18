"""Friction cost model (RULE-09): R_net = R_gross - Turnover * (fee + slippage)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FrictionCostModel:
    """Taker fee + slippage in basis points, charged on ``Turnover_t = 0.5 * sum|w_t - w_(t-1)|``.

    The position before the first bar is flat (0), so entering on bar 0 pays its cost.
    """

    def __init__(self, fee_bps: float, slippage_bps: float) -> None:
        if fee_bps < 0 or slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be >= 0")
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)

    def turnover(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        prev = np.concatenate(([0.0], positions[:-1]))
        return 0.5 * np.abs(positions - prev)

    def net_returns(
        self, gross: NDArray[np.float64], positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if gross.shape != positions.shape:
            raise ValueError(f"length mismatch: gross {gross.shape} vs positions {positions.shape}")
        cost = (self.fee_bps + self.slippage_bps) / 1e4
        return gross - self.turnover(positions) * cost
