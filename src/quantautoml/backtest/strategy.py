"""Thin tempusonebt adapter: replays pre-computed positions (Tier 2)."""

from __future__ import annotations

try:
    from tempusonebt import Strategy
except ImportError as e:
    raise ImportError(
        'tempusonebt package required: uv add --dev "tempusonebt @ git+https://github.com/tempusoneps/tempusonebt.git"'
    ) from e


class BacktestingPyStrategy(Strategy):  # type: ignore[misc]
    """Issues Market orders following the pre-computed ``_position`` column.

    The order is placed at bar t and filled at Open[t+1] (``trade_on_close=False``).
    Positions are computed once in Tier 1 (HysteresisBuffer); nothing is re-derived here.

    Supported ``size_mode`` values:
    - ``"auto"`` (default): full compounded equity (``tempusonebt`` default).
    - ``"start_cash"`` or ``"_start_cash"``: fixed notional budget equal to ``initial_cash``
      (capped by currently available equity so orders are never rejected on drawdowns).
    - ``int >= 1`` (or numeric string ``"1"``, ``"5"``): fixed number of contracts/units.
    - ``float in (0, 1)`` (or numeric string ``"0.5"``): fraction of current equity.
    """

    size_mode: str | int | float = "auto"
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0

    def init(self) -> None:
        self._pos = self.I(lambda x: x, self.data["_position"], name="position")

    def _order_kwargs(self) -> dict[str, int | float]:
        mode = self.size_mode
        if isinstance(mode, str):
            s = mode.strip()
            if s == "auto":
                return {}
            if s in ("start_cash", "_start_cash"):
                price = float(self.data.Close[-1])
                budget = min(float(self.initial_cash), float(self.equity)) * (
                    1.0 - float(self.commission_rate)
                )
                units = max(1, int(budget // price)) if price > 0 else 1
                return {"size": units}
            num = float(s)
            return {"size": int(num) if num >= 1.0 else num}
        num_val = float(mode)
        return {"size": int(num_val) if num_val >= 1.0 else num_val}

    def next(self) -> None:
        target = int(self._pos[-1])
        current = 1 if self.position.is_long else (-1 if self.position.is_short else 0)
        if target == current:
            return
        if self.position:
            self.position.close()
        kwargs = self._order_kwargs()
        if target == 1:
            self.buy(**kwargs)
        elif target == -1:
            self.sell(**kwargs)
