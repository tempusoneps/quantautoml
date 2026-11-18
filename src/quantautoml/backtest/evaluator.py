"""OOS acceptance gates (RULE-02 rank metrics, RULE-09 Net Sharpe after friction)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from quantautoml.common.config import BacktestConfig
from quantautoml.metrics.ic import calc_icir, calc_rank_ic, calc_rolling_ic

_MIN_STRESS_BARS = 30


class OOSMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank_ic: float
    icir: float
    gross_sharpe: float
    net_sharpe: float
    max_drawdown: float
    downside_ic: float
    n_stress_bars: int
    mean_turnover: float
    n_bars: int
    latency_ms: float | None
    gates: dict[str, bool]
    accepted: bool
    failed_gates: list[str]


def evaluate_oos_performance(
    *,
    scores: NDArray[np.float32],
    y: NDArray[np.int32],
    gross_returns: NDArray[np.float64],
    net_returns: NDArray[np.float64],
    turnover: NDArray[np.float64],
    market_returns: NDArray[np.float64],
    periods_per_year: float,
    latency_ms: float | None,
    cfg: BacktestConfig,
    latency_budget_ms: float = 5.0,
) -> OOSMetrics:
    """Compute OOS metrics and the acceptance gates (all strict inequalities).

    Downside IC is the Rank IC on *stress bars*: bars whose trailing ``cfg.ic_window`` market
    return lies in the lowest ``cfg.downside_quantile``; it needs >= 30 such bars to count.
    """
    n = len(scores)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    if not all(len(a) == n for a in (y, gross_returns, net_returns, turnover, market_returns)):
        raise ValueError("length mismatch between evaluator inputs")

    rank_ic = calc_rank_ic(scores, y)[0]
    icir = calc_icir(calc_rolling_ic(scores, y, cfg.ic_window))
    stress = _stress_mask(market_returns, cfg.ic_window, cfg.downside_quantile)
    n_stress = int(stress.sum())
    downside_ic = calc_rank_ic(scores[stress], y[stress])[0] if n_stress >= _MIN_STRESS_BARS else 0.0
    net_sharpe = _sharpe(net_returns, periods_per_year)
    mdd = _max_drawdown(net_returns)

    gates = {
        "rank_ic": rank_ic > cfg.min_rank_ic,
        "icir": icir > cfg.min_icir,
        "net_sharpe": net_sharpe > cfg.min_net_sharpe,
        "mdd": mdd < cfg.max_mdd,
        "latency": latency_ms is not None and latency_ms < latency_budget_ms,
        "downside_ic": n_stress >= _MIN_STRESS_BARS and downside_ic >= cfg.min_downside_ic,
    }
    failed = [k for k, ok in gates.items() if not ok]
    return OOSMetrics(
        rank_ic=rank_ic, icir=icir, gross_sharpe=_sharpe(gross_returns, periods_per_year),
        net_sharpe=net_sharpe, max_drawdown=mdd, downside_ic=downside_ic, n_stress_bars=n_stress,
        mean_turnover=float(np.mean(turnover)) if n else 0.0, n_bars=n, latency_ms=latency_ms,
        gates=gates, accepted=not failed, failed_gates=failed,
    )


def _sharpe(returns: NDArray[np.float64], periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-15:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _max_drawdown(returns: NDArray[np.float64]) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peak = np.maximum.accumulate(equity)
    return float(np.clip(np.max(1.0 - equity / peak), 0.0, 1.0))


def _stress_mask(
    market_returns: NDArray[np.float64], window: int, quantile: float
) -> NDArray[np.bool_]:
    mask = np.zeros(len(market_returns), dtype=bool)
    if len(market_returns) <= window:
        return mask
    cs = np.cumsum(market_returns)
    trailing = cs[window:] - cs[:-window]
    mask[window:] = trailing <= np.quantile(trailing, quantile)
    return mask
