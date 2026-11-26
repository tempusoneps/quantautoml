"""Unit tests for RealWorldBacktestRunner / RealWorldMetrics (Tier 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tempusonebt")

from quantautoml.backtest.realworld_runner import (  # noqa: E402
    RealWorldBacktestRunner,
    RealWorldMetrics,
)
from quantautoml.common.config import RealWorldConfig  # noqa: E402
from quantautoml.common.exceptions import BacktestError  # noqa: E402

N = 500


def _ohlcv(n: int = N, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + 0.002 * rng.normal(size=n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.001,
            "Low": np.minimum(open_, close) * 0.999,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def _positions(n: int = N) -> np.ndarray:
    pos = np.zeros(n, dtype=np.float64)
    for start in range(10, n - 20, 40):  # alternating long / short blocks
        pos[start : start + 20] = 1.0 if (start // 40) % 2 == 0 else -1.0
    return pos


def _run(cfg: RealWorldConfig | None = None, df: pd.DataFrame | None = None,
         pos: np.ndarray | None = None) -> RealWorldMetrics:
    return RealWorldBacktestRunner(cfg or RealWorldConfig()).run(
        ohlcv_df=_ohlcv() if df is None else df,
        positions=_positions() if pos is None else pos,
    )


def test_metrics_has_expected_fields() -> None:
    m = _run()
    assert isinstance(m, RealWorldMetrics)
    assert set(RealWorldMetrics.model_fields) == {
        "n_trades", "win_rate", "profit_factor", "avg_holding_bars", "max_trade_loss_pct",
        "equity_final", "equity_peak", "return_pct", "buy_hold_return_pct",
        "sharpe_ratio_bt", "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
    }
    assert m.n_trades > 0


def test_zero_trades_returns_zero_metrics() -> None:
    m = _run(pos=np.zeros(N))
    assert m.n_trades == 0
    assert m.win_rate == 0.0 and m.profit_factor == 0.0 and m.return_pct == 0.0
    assert m.max_trade_loss_pct == 0.0 and m.avg_holding_bars == 0.0


def test_mismatched_length_raises() -> None:
    with pytest.raises(ValueError, match="positions"):
        _run(pos=np.zeros(N - 1))


def test_missing_ohlcv_column_raises() -> None:
    with pytest.raises(BacktestError, match="High"):
        _run(df=_ohlcv().drop(columns=["High"]))


def test_nan_in_close_raises() -> None:
    df = _ohlcv()
    df.iloc[10, df.columns.get_loc("Close")] = np.nan
    with pytest.raises(BacktestError, match="NaN"):
        _run(df=df)


def test_non_datetime_index_raises() -> None:
    with pytest.raises(BacktestError, match="DatetimeIndex"):
        _run(df=_ohlcv().reset_index(drop=True))


def test_win_rate_range() -> None:
    assert 0.0 <= _run().win_rate <= 1.0


def test_equity_final_positive() -> None:
    assert _run().equity_final > 0


def test_custom_column_names_are_mapped() -> None:
    cfg = RealWorldConfig(ohlcv_open_col="o", ohlcv_high_col="h", ohlcv_low_col="l",
                          ohlcv_close_col="c", ohlcv_volume_col="v")
    df = _ohlcv().rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})
    assert _run(cfg, df=df).n_trades == _run().n_trades


def test_html_export_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    _run(RealWorldConfig(out_html=str(out)))
    assert out.exists() and out.stat().st_size > 0


def test_csv_export_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "trades.csv"
    m = _run(RealWorldConfig(out_trades_csv=str(out)))
    trades = pd.read_csv(out)
    assert len(trades) == m.n_trades
    assert {"EntryBar", "ExitBar", "EntryPrice", "ExitPrice", "PnL", "ReturnPct", "Duration"} <= set(
        trades.columns
    )


def test_runner_respects_fixed_size_config(tmp_path: Path) -> None:
    out = tmp_path / "trades_size1.csv"
    _run(RealWorldConfig(size=1, out_trades_csv=str(out)))
    trades = pd.read_csv(out)
    assert len(trades) > 0
    assert set(trades["Size"].abs().unique()) == {1}


def test_runner_respects_start_cash_size_config(tmp_path: Path) -> None:
    out = tmp_path / "trades_start_cash.csv"
    m = _run(RealWorldConfig(size="_start_cash", out_trades_csv=str(out)))
    assert m.n_trades > 0
    assert out.is_file()


def test_returns_curve_export_creates_png(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "returns_curve.png"
    m = _run(RealWorldConfig(out_returns_curve=str(out)))
    assert m.n_trades > 0
    assert out.is_file()
    raw = out.read_bytes()
    assert len(raw) > 0
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_returns_curve_export_handles_zero_trades(tmp_path: Path) -> None:
    out = tmp_path / "returns_curve.png"
    m = _run(RealWorldConfig(out_returns_curve=str(out)), pos=np.zeros(N))
    assert m.n_trades == 0
    assert out.is_file()
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

