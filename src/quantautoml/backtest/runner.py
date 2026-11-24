"""Blind OOS backtest: the bundle's predictor scores OOS, Hysteresis sizes it, friction charges it.

Nothing is fitted here. ``oos`` must be index-disjoint from every other segment (RULE-01).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantautoml.backtest.buffer import HysteresisBuffer
from quantautoml.backtest.evaluator import OOSMetrics, evaluate_oos_performance
from quantautoml.backtest.friction import FrictionCostModel
from quantautoml.backtest.realworld_runner import RealWorldBacktestRunner, RealWorldMetrics
from quantautoml.common.config import BacktestConfig, RealWorldConfig
from quantautoml.common.exceptions import BacktestError, DataLeakageError
from quantautoml.common.types import Segment
from quantautoml.packaging.predictor_loader import load_predictor
from quantautoml.packaging.schema import FeatureSchemaContract

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
        latency_budget_ms: float = 5.0,
        ohlcv_df: pd.DataFrame | None = None,
        realworld_cfg: RealWorldConfig | None = None,
    ) -> tuple[OOSMetrics, RealWorldMetrics | None]:
        cfg = self.cfg
        for seg in other_segments:
            if np.intersect1d(oos.index, seg.index).size:
                raise DataLeakageError("OOS rows overlap another segment (RULE-01)")
        if oos.y is None:
            raise BacktestError("OOS segment has no labels; Rank IC cannot be computed")
        n = len(oos.X)
        if n < cfg.rank_window + 2:
            raise BacktestError(f"OOS has {n} bars, need >= rank_window + 2 = {cfg.rank_window + 2}")
        close = self._price_series(oos.X, feature_names, cfg.price_col)
        open_px = (
            self._price_series(oos.X, feature_names, cfg.open_col)
            if cfg.execution_price == "next_open"
            else None
        )
        schema = FeatureSchemaContract.from_json(
            (Path(bundle_dir) / "features_schema.json").read_text(encoding="utf-8")
        )
        schema.check_input(oos.X, names=feature_names)  # SchemaMismatchError on wrong order/width

        predictor = load_predictor(Path(bundle_dir) / "predictor.py").Predictor(bundle_dir)
        scores = np.asarray(predictor.predict(oos.X), dtype=np.float32)
        latency_ms = _latency_ms(predictor, oos.X[:1], latency_repeats)

        positions = HysteresisBuffer(cfg.top_pct, cfg.exit_pct, cfg.rank_window).positions(scores)
        if cfg.execution_price == "next_open" and open_px is not None:
            # Decision at bar t enters at Open[t+1] and marks/exits at Open[t+2] (matches Tier 2).
            market = open_px[2:] / open_px[1:-1] - 1.0
            pos = positions[:-2]
            eval_scores, eval_y, eval_ts = scores[:-2], oos.y[:-2], oos.timestamps[:-2]
        else:
            # Position decided at bar t earns the return from Close[t] to Close[t+1].
            market = close[1:] / close[:-1] - 1.0
            pos = positions[:-1]
            eval_scores, eval_y, eval_ts = scores[:-1], oos.y[:-1], oos.timestamps[:-1]
        friction = FrictionCostModel(cfg.fee_bps, cfg.slippage_bps)
        gross = pos * market
        metrics = evaluate_oos_performance(
            scores=eval_scores, y=eval_y, gross_returns=gross,
            net_returns=friction.net_returns(gross, pos), turnover=friction.turnover(pos),
            market_returns=market, periods_per_year=_periods_per_year(oos.timestamps),
            latency_ms=latency_ms, cfg=cfg, latency_budget_ms=latency_budget_ms,
            timestamps=eval_ts,
        )
        if out_path is not None:
            report = metrics.model_dump()
            report |= {
                "periods_per_year": _periods_per_year(oos.timestamps),
                "n_overnight_gaps": _count_gaps(oos.timestamps),
                "config": cfg.model_dump(),
            }
            Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        # Tier 2 (realistic execution) only validates strategies Tier 1 already accepted.
        # Respects realworld_cfg.enabled when realworld_cfg is explicitly provided;
        # defaults to enabled when a caller passes ohlcv_df without realworld_cfg.
        rw_metrics: RealWorldMetrics | None = None
        tier2_enabled = True if realworld_cfg is None else realworld_cfg.enabled
        if ohlcv_df is not None and metrics.accepted and tier2_enabled:
            rw_metrics = RealWorldBacktestRunner(realworld_cfg or RealWorldConfig()).run(
                ohlcv_df=ohlcv_df, positions=positions
            )
        return metrics, rw_metrics

    @staticmethod
    def _price_series(
        X: NDArray[np.float32], feature_names: Sequence[str], col: str
    ) -> NDArray[np.float64]:
        if col not in feature_names:
            raise BacktestError(f"price column '{col}' not found in feature names")
        px = X[:, list(feature_names).index(col)].astype(np.float64)
        if not (np.isfinite(px).all() and (px > 0).all()):
            raise BacktestError(f"price column '{col}' must be finite and > 0 on every OOS bar")
        return px


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
