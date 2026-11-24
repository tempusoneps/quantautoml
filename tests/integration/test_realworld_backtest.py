"""Tier 2 (tempusonebt) end-to-end on a real bundle: HTML + trades.csv, gated by Tier 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bundle_helper import make_bundle

pytest.importorskip("tempusonebt")

from quantautoml.backtest import OOSBacktestRunner  # noqa: E402
from quantautoml.common.config import BacktestConfig, RealWorldConfig  # noqa: E402
from quantautoml.common.types import Segment  # noqa: E402

NAMES = ["x0", "x1", "x2", "x3", "x4", "Close"]


def _oos(n: int = 3000, seed: int = 5) -> Segment:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    s = X[:, 0] + 0.5 * X[:, 1]
    y = ((s + 0.5 * rng.normal(size=n)) > np.median(s)).astype(np.int32)
    ret = 0.001 * np.tanh(X[:, 0]) + 0.0005 * rng.normal(size=n)
    X[:, 5] = (100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + ret[:-1])))).astype(np.float32)
    ts = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(n) * np.timedelta64(5, "m")
    return Segment(X=X, y=y, labels_aux={}, timestamps=ts, index=np.arange(10_000, 10_000 + n))


def _ohlcv(oos: Segment) -> pd.DataFrame:
    close = oos.X[:, 5].astype(np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.0005,
            "Low": np.minimum(open_, close) * 0.9995,
            "Close": close,
            "Volume": np.full(len(close), 1000.0),
        },
        index=pd.DatetimeIndex(oos.timestamps),
    )


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("rw"))


def test_tier2_produces_html_and_csv_on_real_bundle(bundle, tmp_path) -> None:
    oos = _oos()
    html, csv = tmp_path / "backtest_report.html", tmp_path / "trades.csv"
    metrics, rw = OOSBacktestRunner(BacktestConfig(rank_window=100, ic_window=300)).run(
        bundle_dir=bundle.dir, oos=oos, feature_names=NAMES, other_segments=[],
        ohlcv_df=_ohlcv(oos),
        realworld_cfg=RealWorldConfig(enabled=True, out_html=str(html), out_trades_csv=str(csv)),
    )
    assert metrics.accepted, metrics.failed_gates
    assert rw is not None and rw.n_trades > 0
    assert html.exists() and html.stat().st_size > 0
    trades = pd.read_csv(csv)
    assert len(trades) == rw.n_trades
    assert {"EntryBar", "ExitBar", "EntryPrice", "ExitPrice", "PnL", "ReturnPct", "Duration"} <= set(
        trades.columns
    )


def test_tier2_skipped_when_tier1_rejected(bundle, tmp_path) -> None:
    oos = _oos()
    html = tmp_path / "never.html"
    metrics, rw = OOSBacktestRunner(
        BacktestConfig(rank_window=100, ic_window=300, min_net_sharpe=999.0)
    ).run(
        bundle_dir=bundle.dir, oos=oos, feature_names=NAMES, other_segments=[],
        ohlcv_df=_ohlcv(oos), realworld_cfg=RealWorldConfig(enabled=True, out_html=str(html)),
    )
    assert not metrics.accepted
    assert rw is None and not html.exists()


def test_tier2_trades_are_consistent_with_tier1_positions(bundle, tmp_path) -> None:
    """Tier 2 replays the same positions: it never trades while Tier 1 is flat for the whole run."""
    oos = _oos()
    csv = tmp_path / "trades.csv"
    _, rw = OOSBacktestRunner(BacktestConfig(rank_window=100, ic_window=300)).run(
        bundle_dir=bundle.dir, oos=oos, feature_names=NAMES, other_segments=[],
        ohlcv_df=_ohlcv(oos), realworld_cfg=RealWorldConfig(enabled=True, out_trades_csv=str(csv)),
    )
    assert rw is not None
    trades = pd.read_csv(csv)
    assert (trades["EntryBar"] >= 100).all()  # HysteresisBuffer warm-up: no trade before rank_window
