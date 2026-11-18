"""Blind OOS backtest: the bundle's predictor scores OOS, Hysteresis sizes it, friction charges it.

Nothing is fitted here. ``oos`` must be index-disjoint from every other segment (RULE-01).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quantautoml.backtest.buffer import HysteresisBuffer
from quantautoml.backtest.evaluator import OOSMetrics, evaluate_oos_performance
from quantautoml.backtest.friction import FrictionCostModel
from quantautoml.common.config import BacktestConfig
from quantautoml.common.exceptions import BacktestError, DataLeakageError
from quantautoml.common.types import Segment
from quantautoml.packaging.predictor_loader import load_predictor

_SECONDS_PER_YEAR = 365.25 * 86400.0
_GAP_FACTOR = 3.0  # a timestamp step > 3x the median step counts as a session gap


class OOSBacktestRunner:
    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        *,
        bundle_dir: Path,
        oos: Segment,
        feature_names: Sequence[str],
        other_segments: Sequence[Segment],
        out_path: Path | None = None,
        latency_repeats: int = 200,
    ) -> OOSMetrics:
        cfg = self.cfg
        for seg in other_segments:
            if np.intersect1d(oos.index, seg.index).size:
                raise DataLeakageError("OOS rows overlap another segment (RULE-01)")
        if oos.y is None:
            raise BacktestError("OOS segment has no labels; Rank IC cannot be computed")
        n = len(oos.X)
        if n < cfg.rank_window + 2:
            raise BacktestError(f"OOS has {n} bars, need >= rank_window + 2 = {cfg.rank_window + 2}")
        close = self._close(oos.X, feature_names)

        predictor = load_predictor(Path(bundle_dir) / "predictor.py").Predictor(bundle_dir)
        scores = np.asarray(predictor.predict(oos.X), dtype=np.float32)
        latency_ms = _latency_ms(predictor, oos.X[:1], latency_repeats)

        # Position decided at bar t earns the return from t to t+1; the last bar has none.
        market = close[1:] / close[:-1] - 1.0
        positions = HysteresisBuffer(cfg.top_pct, cfg.exit_pct, cfg.rank_window).positions(scores)
        pos = positions[:-1]
        friction = FrictionCostModel(cfg.fee_bps, cfg.slippage_bps)
        gross = pos * market
        metrics = evaluate_oos_performance(
            scores=scores[:-1], y=oos.y[:-1], gross_returns=gross,
            net_returns=friction.net_returns(gross, pos), turnover=friction.turnover(pos),
            market_returns=market, periods_per_year=_periods_per_year(oos.timestamps),
            latency_ms=latency_ms, cfg=cfg,
        )
        if out_path is not None:
            report = metrics.model_dump()
            report |= {
                "periods_per_year": _periods_per_year(oos.timestamps),
                "n_overnight_gaps": _count_gaps(oos.timestamps),
                "config": cfg.model_dump(),
            }
            Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return metrics

    def _close(self, X: NDArray[np.float32], feature_names: Sequence[str]) -> NDArray[np.float64]:
        col = self.cfg.price_col
        if col not in feature_names:
            raise BacktestError(f"price column '{col}' not found in feature names")
        close = X[:, list(feature_names).index(col)].astype(np.float64)
        if not (np.isfinite(close).all() and (close > 0).all()):
            raise BacktestError(f"price column '{col}' must be finite and > 0 on every OOS bar")
        return close


def _latency_ms(predictor: object, row: NDArray[np.float32], repeats: int) -> float:
    predictor.predict(row)  # type: ignore[attr-defined]  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predictor.predict(row)  # type: ignore[attr-defined]
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def _periods_per_year(timestamps: NDArray[np.datetime64]) -> float:
    span = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, "s")
    if not span > 0:
        raise BacktestError("OOS timestamps must be strictly increasing to annualize")
    return float((len(timestamps) - 1) / (span / _SECONDS_PER_YEAR))


def _count_gaps(timestamps: NDArray[np.datetime64]) -> int:
    steps = np.diff(timestamps).astype("timedelta64[s]").astype(np.int64)
    return int(np.count_nonzero(steps > _GAP_FACTOR * np.median(steps)))
