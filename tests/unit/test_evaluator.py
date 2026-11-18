"""evaluate_oos_performance: RULE-09 net-of-friction acceptance gates."""

from __future__ import annotations

import numpy as np
import pytest

from quantautoml.backtest.evaluator import OOSMetrics, evaluate_oos_performance
from quantautoml.common.config import BacktestConfig
from quantautoml.metrics.ic import calc_rank_ic

PPY = 1000.0


def _good(n: int = 4000, seed: int = 0) -> dict:
    """Strong, consistent signal with steady positive net returns -> every gate passes."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n).astype(np.float32)
    y = ((scores + 0.5 * rng.normal(size=n)) > 0).astype(np.int32)
    net = 0.001 + 0.0005 * rng.normal(size=n)
    return dict(
        scores=scores, y=y, gross_returns=net + 0.0001, net_returns=net,
        turnover=np.full(n, 0.01), market_returns=rng.normal(0, 0.001, size=n),
        periods_per_year=PPY, latency_ms=1.0, cfg=BacktestConfig(ic_window=500),
    )


def test_sharpe_matches_manual() -> None:
    kw = _good()
    m = evaluate_oos_performance(**kw)
    r = kw["net_returns"]
    assert m.net_sharpe == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(PPY))


def test_all_gates_pass_on_good_run() -> None:
    m = evaluate_oos_performance(**_good())
    assert isinstance(m, OOSMetrics)
    assert m.accepted and m.failed_gates == [] and all(m.gates.values())
    assert set(m.gates) == {"rank_ic", "icir", "net_sharpe", "mdd", "latency", "downside_ic"}


def test_mdd_known_curve() -> None:
    kw = _good(n=4)
    kw["net_returns"] = np.array([0.10, -0.20, 0.0, 0.0])
    kw["gross_returns"] = kw["net_returns"]
    assert evaluate_oos_performance(**kw).max_drawdown == pytest.approx(0.20)


def test_flat_returns_sharpe_zero_not_nan() -> None:
    kw = _good()
    kw["net_returns"] = np.zeros(4000)
    kw["gross_returns"] = np.zeros(4000)
    m = evaluate_oos_performance(**kw)
    assert m.net_sharpe == 0.0 and m.gross_sharpe == 0.0 and m.max_drawdown == 0.0
    assert not m.accepted and "net_sharpe" in m.failed_gates


def test_constant_scores_are_finite_and_rejected() -> None:
    kw = _good()
    kw["scores"] = np.zeros(4000, dtype=np.float32)
    m = evaluate_oos_performance(**kw)
    vals = [m.rank_ic, m.icir, m.downside_ic, m.net_sharpe, m.mean_turnover]
    assert all(np.isfinite(v) for v in vals)
    assert not m.accepted and "rank_ic" in m.failed_gates


def test_gates_use_strict_inequality() -> None:
    kw = _good()
    ic = calc_rank_ic(kw["scores"], kw["y"])[0]
    kw["cfg"] = BacktestConfig(ic_window=500, min_rank_ic=ic)
    m = evaluate_oos_performance(**kw)
    assert m.rank_ic == pytest.approx(ic) and not m.gates["rank_ic"]


def test_failed_gates_listed() -> None:
    kw = _good()
    kw["cfg"] = BacktestConfig(ic_window=500, min_net_sharpe=1e6)
    m = evaluate_oos_performance(**kw)
    assert m.failed_gates == ["net_sharpe"] and not m.accepted


def test_missing_latency_fails_latency_gate() -> None:
    kw = _good()
    kw["latency_ms"] = None
    m = evaluate_oos_performance(**kw)
    assert not m.gates["latency"] and not m.accepted


def test_latency_at_budget_fails() -> None:
    kw = _good()
    kw["latency_ms"] = 5.0
    assert not evaluate_oos_performance(**kw).gates["latency"]


def test_net_sharpe_used_for_gate_not_gross() -> None:
    kw = _good()
    kw["gross_returns"] = np.abs(kw["gross_returns"]) + 0.001
    kw["net_returns"] = -np.abs(kw["net_returns"])
    m = evaluate_oos_performance(**kw)
    assert m.gross_sharpe > 1.2 and m.net_sharpe < 0
    assert not m.accepted and "net_sharpe" in m.failed_gates


def test_negative_downside_ic_fails_gate() -> None:
    kw = _good()
    # Signal flips sign in market-stress bars: rank IC there is negative.
    mkt = kw["market_returns"]
    stress = np.zeros(len(mkt), dtype=bool)
    w = kw["cfg"].ic_window
    cs = np.cumsum(mkt)
    trailing = np.full(len(mkt), np.nan)
    trailing[w:] = cs[w:] - cs[:-w]
    stress[w:] = trailing[w:] <= np.nanquantile(trailing[w:], kw["cfg"].downside_quantile)
    y = kw["y"].copy()
    y[stress] = 1 - y[stress]
    kw["y"] = y
    m = evaluate_oos_performance(**kw)
    assert m.downside_ic < 0 and not m.gates["downside_ic"]


def test_too_few_stress_bars_fails_downside_gate() -> None:
    kw = _good(n=100)
    kw["cfg"] = BacktestConfig(ic_window=20)
    m = evaluate_oos_performance(**kw)
    assert not m.gates["downside_ic"]


def test_length_mismatch_raises() -> None:
    kw = _good()
    kw["turnover"] = np.zeros(10)
    with pytest.raises(ValueError, match="length"):
        evaluate_oos_performance(**kw)


def test_non_positive_periods_per_year_raises() -> None:
    kw = _good()
    kw["periods_per_year"] = 0.0
    with pytest.raises(ValueError):
        evaluate_oos_performance(**kw)


def test_degenerate_downside_ic_fails_gate() -> None:
    kw = _good()
    kw["y"] = np.zeros(4000, dtype=np.int32)  # constant labels: IC is undefined, not "stable"
    m = evaluate_oos_performance(**kw)
    assert m.n_stress_bars >= 30 and not m.gates["downside_ic"]
