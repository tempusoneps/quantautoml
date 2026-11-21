"""Thin backtesting.py adapter: replays pre-computed positions (Tier 2)."""

from __future__ import annotations

try:
    from backtesting import Strategy
except ImportError as e:
    raise ImportError("backtesting package required: uv add --dev backtesting") from e


class BacktestingPyStrategy(Strategy):  # type: ignore[misc]
    """Issues Market orders following the pre-computed ``_position`` column.

    The order is placed at bar t and filled at Open[t+1] (``trade_on_close=False``).
    Positions are computed once in Tier 1 (HysteresisBuffer); nothing is re-derived here.
    """

    def init(self) -> None:
        self._pos = self.I(lambda x: x, self.data["_position"], name="position")

    def next(self) -> None:
        target = int(self._pos[-1])
        current = 1 if self.position.is_long else (-1 if self.position.is_short else 0)
        if target == current:
            return
        if self.position:
            self.position.close()
        if target == 1:
            self.buy()
        elif target == -1:
            self.sell()
