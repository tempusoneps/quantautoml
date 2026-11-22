"""OOSBacktestRunner Tier 2 trigger: realistic execution runs only on accepted Tier 1 + OHLCV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bundle_helper import make_bundle

pytest.importorskip("backtesting")

from quantautoml.backtest import OOSBacktestRunner, OOSMetrics, RealWorldMetrics  # noqa: E402
from quantautoml.common.config import BacktestConfig, RealWorldConfig  # noqa: E402
from quantautoml.common.types import Segment  # noqa: E402

NAMES = ["x0", "x1", "x2", "x3", "x4", "Close"]
CFG = BacktestConfig(rank_window=100, ic_window=300)


def _oos(n: int = 3000, seed: int = 5) -> Segment:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    s = X[:, 0] + 0.5 * X[:, 1]
    y = ((s + 0.5 * rng.normal(size=n)) > np.median(s)).astype(np.int32)
    ret = 0.001 * np.tanh(X[:, 0]) + 0.0005 * rng.normal(size=n)
    close = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + ret[:-1])))
    X[:, 5] = close.astype(np.float32)
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
    return make_bundle(tmp_path_factory.mktemp("tier2"))


def _run(bundle, cfg: BacktestConfig = CFG, **kw):
    oos = _oos()
    if kw.pop("with_ohlcv", False):
        kw["ohlcv_df"] = _ohlcv(oos)
    return OOSBacktestRunner(cfg).run(
        bundle_dir=bundle.dir, oos=oos, feature_names=NAMES, other_segments=[], **kw
    )


def test_run_without_ohlcv_returns_none_rw(bundle) -> None:
    metrics, rw = _run(bundle)
    assert isinstance(metrics, OOSMetrics) and rw is None


def test_run_with_ohlcv_accepted_returns_rw_metrics(bundle) -> None:
    metrics, rw = _run(bundle, with_ohlcv=True)
    assert metrics.accepted, metrics.failed_gates
    assert isinstance(rw, RealWorldMetrics) and rw.n_trades > 0


def test_run_with_ohlcv_rejected_skips_tier2(bundle) -> None:
    metrics, rw = _run(
        bundle, BacktestConfig(rank_window=100, ic_window=300, min_net_sharpe=999.0),
        with_ohlcv=True,
    )
    assert not metrics.accepted
    assert rw is None


def test_backward_compat_old_caller(bundle) -> None:
    oos_metrics, _ = _run(bundle)
    assert oos_metrics.accepted


def test_realworld_cfg_is_forwarded(bundle, tmp_path) -> None:
    csv = tmp_path / "trades.csv"
    _, rw = _run(
        bundle,
        with_ohlcv=True,
        realworld_cfg=RealWorldConfig(enabled=True, out_trades_csv=str(csv)),
    )
    assert rw is not None and csv.exists()


def test_realworld_disabled_skips_tier2_even_when_accepted(bundle, tmp_path) -> None:
    csv = tmp_path / "trades.csv"
    metrics, rw = _run(
        bundle,
        with_ohlcv=True,
        realworld_cfg=RealWorldConfig(enabled=False, out_trades_csv=str(csv)),
    )
    assert metrics.accepted
    assert rw is None
    assert not csv.exists()


def test_tier2_uses_full_length_positions(bundle) -> None:
    """Tier 2 must see all n positions (not the n-1 slice used for Tier 1 returns)."""
    seen: dict[str, int] = {}
    from quantautoml.backtest import runner as runner_mod

    orig = runner_mod.RealWorldBacktestRunner.run

    def spy(self, *, ohlcv_df, positions):  # type: ignore[no-untyped-def]
        seen["n_pos"], seen["n_df"] = len(positions), len(ohlcv_df)
        return orig(self, ohlcv_df=ohlcv_df, positions=positions)

    runner_mod.RealWorldBacktestRunner.run = spy  # type: ignore[method-assign]
    try:
        _run(bundle, with_ohlcv=True)
    finally:
        runner_mod.RealWorldBacktestRunner.run = orig  # type: ignore[method-assign]
    assert seen["n_pos"] == seen["n_df"] == 3000
