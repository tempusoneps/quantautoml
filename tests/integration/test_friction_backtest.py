"""Friction + Hysteresis behave as RULE-09 requires on a real bundle (blind OOS run)."""

from __future__ import annotations

import numpy as np
import pytest
from bundle_helper import make_bundle

from quantautoml.backtest import OOSBacktestRunner
from quantautoml.common.config import BacktestConfig
from quantautoml.common.types import Segment

NAMES = ["x0", "x1", "x2", "x3", "x4", "Close"]


def _wobbly_oos(n: int = 4000, seed: int = 11) -> Segment:
    """Slow signal + noise: the percentile rank keeps wobbling around the entry boundary."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    slow = np.convolve(rng.normal(size=n + 59), np.ones(60) / 60, mode="valid")
    X[:, 0] = (slow / slow.std() + 0.25 * rng.normal(size=n)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int32)
    ret = 0.001 * np.tanh(X[:, 0]) + 0.0005 * rng.normal(size=n)
    X[:, 5] = (100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + ret[:-1])))).astype(np.float32)
    ts = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(n) * np.timedelta64(5, "m")
    return Segment(X=X, y=y, labels_aux={}, timestamps=ts, index=np.arange(10_000, 10_000 + n))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("frict"))


def _run(bundle, **cfg_kw):
    cfg = BacktestConfig(rank_window=100, ic_window=300, **cfg_kw)
    return OOSBacktestRunner(cfg).run(
        bundle_dir=bundle.dir, oos=_wobbly_oos(), feature_names=NAMES, other_segments=[]
    )


def test_friction_lowers_sharpe(bundle) -> None:
    free = _run(bundle, fee_bps=0.0, slippage_bps=0.0)
    paid = _run(bundle, fee_bps=4.0, slippage_bps=1.0)
    assert free.net_sharpe == pytest.approx(free.gross_sharpe)
    assert paid.net_sharpe < paid.gross_sharpe
    assert paid.gross_sharpe == pytest.approx(free.gross_sharpe)  # costs never change gross


def test_higher_friction_never_improves_net_sharpe(bundle) -> None:
    low = _run(bundle, fee_bps=1.0, slippage_bps=0.0)
    high = _run(bundle, fee_bps=20.0, slippage_bps=5.0)
    assert high.net_sharpe < low.net_sharpe


def test_buffer_lowers_turnover(bundle) -> None:
    buffered = _run(bundle, top_pct=0.20, exit_pct=0.30)
    plain = _run(bundle, top_pct=0.20, exit_pct=0.20)
    assert buffered.mean_turnover < plain.mean_turnover
