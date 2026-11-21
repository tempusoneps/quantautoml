"""Unit tests for BacktestingPyStrategy (adapter for pre-computed positions)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("backtesting")

from backtesting import Backtest  # noqa: E402

from quantautoml.backtest.strategy import BacktestingPyStrategy  # noqa: E402

N_BARS = 100


def _ohlcv(positions: list[int]) -> pd.DataFrame:
    n = len(positions)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = 100.0 + np.sin(np.arange(n) / 5.0)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n, 1000.0),
            "_position": np.asarray(positions, dtype=float),
        },
        index=idx,
    )


def _run(positions: list[int]) -> pd.DataFrame:
    bt = Backtest(
        _ohlcv(positions),
        BacktestingPyStrategy,
        cash=1_000_000.0,
        commission=0.0,
        finalize_trades=True,
    )
    stats = bt.run()
    return stats["_trades"]


def test_strategy_long_signal_opens_position() -> None:
    trades = _run([0] * 20 + [1] * 10 + [0] * 70)
    assert len(trades) == 1
    assert trades["Size"].iloc[0] > 0


def test_strategy_short_signal_opens_short() -> None:
    trades = _run([0] * 20 + [-1] * 10 + [0] * 70)
    assert len(trades) == 1
    assert trades["Size"].iloc[0] < 0


def test_strategy_no_signal_no_trades() -> None:
    trades = _run([0] * N_BARS)
    assert len(trades) == 0


def test_strategy_flip_long_to_short_no_flat_in_between() -> None:
    trades = _run([1] * 50 + [-1] * 50)
    assert len(trades) == 2
    assert trades["Size"].iloc[0] > 0
    assert trades["Size"].iloc[1] < 0
    assert trades["ExitBar"].iloc[0] == trades["EntryBar"].iloc[1]
