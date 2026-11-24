"""OOS acceptance gates (RULE-02 rank metrics, RULE-09 Net Sharpe after friction)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import kurtosis, norm, skew

from quantautoml.common.config import BacktestConfig
from quantautoml.metrics.ic import calc_icir, calc_rank_ic, calc_rolling_ic

_MIN_STRESS_BARS = 30
_TRADING_DAYS_PER_YEAR = 252.0


class OOSMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank_ic: float
    icir: float
    gross_sharpe: float
    net_sharpe: float
    daily_net_sharpe: float | None = None
    deflated_sharpe_ratio: float | None = None
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
    timestamps: NDArray[np.datetime64] | None = None,
) -> OOSMetrics:
    """Compute OOS metrics and the acceptance gates (all strict inequalities).

    Downside IC is the Rank IC on *stress bars*: bars whose trailing ``cfg.ic_window`` market
    return lies in the lowest ``cfg.downside_quantile``; it needs >= 30 such bars with
    non-constant scores and labels to count, else its gate fails. Its gate is ``>=`` (not strict).
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
    # An undefined IC (too few bars, or constant scores/labels on the stress bars) is no evidence.
    downside_defined = (
        n_stress >= _MIN_STRESS_BARS
        and np.ptp(scores[stress]) > 0
        and np.ptp(y[stress]) > 0
    )
    downside_ic = calc_rank_ic(scores[stress], y[stress])[0] if downside_defined else 0.0
    net_sharpe = _sharpe(net_returns, periods_per_year)
    mdd = _max_drawdown(net_returns)
    daily_rets = (
        _daily_compounded_returns(net_returns, timestamps)
        if timestamps is not None and len(timestamps) == n
        else None
    )
    daily_net_sharpe = (
        _sharpe(daily_rets, _TRADING_DAYS_PER_YEAR)
        if daily_rets is not None and len(daily_rets) >= 2
        else None
    )
    dsr_series = daily_rets if (daily_rets is not None and len(daily_rets) >= 5) else net_returns
    dsr = _deflated_sharpe_ratio(dsr_series)

    gates = {
        "rank_ic": rank_ic > cfg.min_rank_ic,
        "icir": icir > cfg.min_icir,
        "net_sharpe": net_sharpe > cfg.min_net_sharpe,
        "mdd": mdd < cfg.max_mdd,
        "latency": latency_ms is not None and latency_ms < latency_budget_ms,
        "downside_ic": downside_defined and downside_ic >= cfg.min_downside_ic,
    }
    failed = [k for k, ok in gates.items() if not ok]
    return OOSMetrics(
        rank_ic=rank_ic, icir=icir, gross_sharpe=_sharpe(gross_returns, periods_per_year),
        net_sharpe=net_sharpe, daily_net_sharpe=daily_net_sharpe, deflated_sharpe_ratio=dsr,
        max_drawdown=mdd, downside_ic=downside_ic, n_stress_bars=n_stress,
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


def _daily_compounded_returns(
    returns: NDArray[np.float64], timestamps: NDArray[np.datetime64]
) -> NDArray[np.float64]:
    """Compound bar returns within each UTC calendar day."""
    if len(returns) == 0:
        return np.empty(0, dtype=np.float64)
    days = timestamps.astype("datetime64[D]")
    _, first_idx, counts = np.unique(days, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    out = np.empty(len(order), dtype=np.float64)
    for i, k in enumerate(order):
        start = int(first_idx[k])
        out[i] = float(np.prod(1.0 + returns[start : start + int(counts[k])]) - 1.0)
    return out


def _deflated_sharpe_ratio(returns: NDArray[np.float64]) -> float:
    """Probabilistic / Deflated Sharpe Ratio P(SR* > 0) adjusting for skew & kurtosis."""
    n = len(returns)
    if n < 3:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-15:
        return 0.0
    sr = float(np.mean(returns)) / std  # non-annualized per-period Sharpe
    g3 = float(skew(returns, bias=False)) if n >= 3 else 0.0
    g4 = float(kurtosis(returns, fisher=False, bias=False)) if n >= 4 else 3.0
    if not (np.isfinite(g3) and np.isfinite(g4)):
        g3, g4 = 0.0, 3.0
    denom_sq = max(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * (sr * sr), 1e-12)
    z = sr * np.sqrt(n - 1.0) / np.sqrt(denom_sq)
    return float(np.clip(norm.cdf(z), 0.0, 1.0))
