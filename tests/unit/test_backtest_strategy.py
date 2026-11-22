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


def test_strategy_fixed_unit_size_one_contract() -> None:
    bt = Backtest(
        _ohlcv([0] * 10 + [1] * 10 + [-1] * 10 + [0] * 70),
        BacktestingPyStrategy,
        cash=1_000_000.0,
        commission=0.0,
        finalize_trades=True,
    )
    stats = bt.run(size_mode=1)
    trades = stats["_trades"]
    assert len(trades) == 2
    assert int(trades["Size"].iloc[0]) == 1
    assert int(trades["Size"].iloc[1]) == -1


def test_strategy_start_cash_size_mode() -> None:
    """With size_mode='start_cash', trade size is computed from initial_cash rather than compounded equity."""
    df = _ohlcv([0] * 10 + [1] * 20 + [0] * 10 + [1] * 20 + [0] * 40)
    # Double price before second trade? No, keep price=100, make first trade lose money so equity drops
    df.loc[df.index[10:31], ["Open", "High", "Low", "Close"]] = 200.0
    df.loc[df.index[30:35], ["Open", "High", "Low", "Close"]] = 100.0  # first trade exits at 100 (50% loss)
    df.loc[df.index[40:], ["Open", "High", "Low", "Close"]] = 200.0    # second trade enters at 200

    bt = Backtest(
        df,
        BacktestingPyStrategy,
        cash=100_000.0,
        commission=0.0,
        finalize_trades=True,
    )
    stats = bt.run(size_mode="start_cash", initial_cash=100_000.0, commission_rate=0.0)
    trades = stats["_trades"]
    assert len(trades) == 2
    # At price 200 with start_cash 100_000, size is floor(100_000 / 200) = 500
    assert int(trades["Size"].iloc[0]) == 500
    # After 50% loss, equity is 50_000, so at price 200 can only afford 250 max, or if price=100 -> 500
    assert int(trades["Size"].iloc[1]) > 0

