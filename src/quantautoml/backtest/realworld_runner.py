"""Tier 2 realistic execution: replays Tier 1 positions through backtesting.py.

``backtesting`` is an optional dev dependency, so it is imported lazily inside ``run()``.
Nothing is fitted here (RULE-01); positions come from Tier 1 and are never recomputed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from quantautoml.common.config import RealWorldConfig
from quantautoml.common.exceptions import BacktestError

_STD_COLS = ("Open", "High", "Low", "Close", "Volume")


class RealWorldMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_trades: int
    win_rate: float  # fraction of winning trades in [0, 1]
    profit_factor: float  # gross profit / gross loss
    avg_holding_bars: float
    max_trade_loss_pct: float  # worst single-trade loss as a positive percentage (0 if none)
    equity_final: float
    equity_peak: float
    return_pct: float
    buy_hold_return_pct: float
    sharpe_ratio_bt: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float  # drawdown magnitude as a positive percentage


class RealWorldBacktestRunner:
    def __init__(self, cfg: RealWorldConfig) -> None:
        self.cfg = cfg

    def run(
        self, *, ohlcv_df: pd.DataFrame, positions: NDArray[np.float64]
    ) -> RealWorldMetrics:
        """Inject positions -> Backtest -> extract metrics -> optional HTML / CSV export."""
        try:
            from backtesting import Backtest
        except ImportError as e:
            raise ImportError("backtesting package required: uv add --dev backtesting") from e
        from quantautoml.backtest.strategy import BacktestingPyStrategy

        cfg = self.cfg
        data = self._prepare(ohlcv_df, positions)
        bt = Backtest(
            data,
            BacktestingPyStrategy,
            cash=cfg.cash,
            commission=cfg.commission,
            margin=cfg.margin,
            trade_on_close=cfg.trade_on_close,
            finalize_trades=True,
        )
        stats = bt.run(
            size_mode=cfg.size,
            initial_cash=cfg.cash,
            commission_rate=cfg.commission,
        )
        trades: pd.DataFrame = stats["_trades"]
        if cfg.out_html is not None:
            bt.plot(filename=cfg.out_html, open_browser=False)
        if cfg.out_trades_csv is not None:
            trades.to_csv(cfg.out_trades_csv, index=False)
        return self._metrics(stats, trades)

    def _prepare(self, ohlcv_df: pd.DataFrame, positions: NDArray[np.float64]) -> pd.DataFrame:
        cfg = self.cfg
        if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
            raise BacktestError("ohlcv_df must have a DatetimeIndex")
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape != (len(ohlcv_df),):
            raise ValueError(f"positions length {pos.size} != ohlcv_df length {len(ohlcv_df)}")
        if not np.isin(pos, (-1.0, 0.0, 1.0)).all():
            raise BacktestError("positions must be in {-1, 0, +1}")
        src = (cfg.ohlcv_open_col, cfg.ohlcv_high_col, cfg.ohlcv_low_col,
               cfg.ohlcv_close_col, cfg.ohlcv_volume_col)
        missing = [c for c in src if c not in ohlcv_df.columns]
        if missing:
            raise BacktestError(f"ohlcv_df is missing column(s): {missing}")
        data = ohlcv_df[list(src)].rename(columns=dict(zip(src, _STD_COLS, strict=True)))
        if data[list(_STD_COLS[:4])].isna().any().any():
            raise BacktestError("ohlcv_df contains NaN in Open/High/Low/Close")
        data["_position"] = pos
        return data

    @staticmethod
    def _metrics(stats: pd.Series, trades: pd.DataFrame) -> RealWorldMetrics:
        n = int(stats["# Trades"])
        if n == 0:
            cash = float(stats["Equity Final [$]"])
            return RealWorldMetrics(
                n_trades=0, win_rate=0.0, profit_factor=0.0, avg_holding_bars=0.0,
                max_trade_loss_pct=0.0, equity_final=cash, equity_peak=cash, return_pct=0.0,
                buy_hold_return_pct=0.0, sharpe_ratio_bt=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, max_drawdown_pct=0.0,
            )
        holding = float((trades["ExitBar"] - trades["EntryBar"]).mean())
        return RealWorldMetrics(
            n_trades=n,
            win_rate=_num(stats["Win Rate [%]"]) / 100.0,
            profit_factor=_num(stats["Profit Factor"]),
            avg_holding_bars=holding,
            max_trade_loss_pct=max(0.0, -_num(stats["Worst Trade [%]"])),
            equity_final=_num(stats["Equity Final [$]"]),
            equity_peak=_num(stats["Equity Peak [$]"]),
            return_pct=_num(stats["Return [%]"]),
            buy_hold_return_pct=_num(stats["Buy & Hold Return [%]"]),
            sharpe_ratio_bt=_num(stats["Sharpe Ratio"]),
            sortino_ratio=_num(stats["Sortino Ratio"]),
            calmar_ratio=_num(stats["Calmar Ratio"]),
            max_drawdown_pct=abs(_num(stats["Max. Drawdown [%]"])),
        )


def _num(value: object) -> float:
    """backtesting.py reports undefined ratios as NaN; surface them as 0.0."""
    x = float(value)  # type: ignore[arg-type]
    return 0.0 if math.isnan(x) else x
