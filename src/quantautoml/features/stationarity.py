"""F2b: Train-only stationarity filter (lag-1 autocorrelation + temporal mean drift)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.types import FilterReport
from quantautoml.features.base import BaseFeatureFilter, FitContext

_EPS = 1e-12


class StationarityFilter(BaseFeatureFilter):
    """Drop non-stationary level/cumulative features using an AND gate on Train only.

    ``metrics_[name] = (lag1_autocorr, temporal_drift)`` where:
      * ``lag1_autocorr`` is the Pearson correlation between contiguous finite adjacent
        observations ``(x[t], x[t+1])`` on Train (unit-root / persistence proxy).
      * ``temporal_drift`` is ``(max_k mean_k - min_k mean_k) / std_train`` across
        ``cfg.stationarity_blocks`` contiguous chronological blocks on Train.

    A column is dropped only when BOTH ``lag1_autocorr > cfg.max_lag1_autocorr`` AND
    ``temporal_drift > cfg.max_temporal_drift``. This catches non-stationary price levels
    and cumulative indices (e.g., ``prev_day_close``, ``pvi``, ``obv``) while preserving
    bounded slow-moving oscillators (e.g., ``ulcer_index``, ``long_moment``).

    Setting ``cfg.max_temporal_drift = None`` disables the filter (keeps all columns).
    """

    name: ClassVar[str] = "f2b_stationarity"

    def __init__(self, cfg: FeaturesConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.metrics_: dict[str, tuple[float, float]] = {}
        self._keep_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        names = ctx.feature_names
        max_drift = self.cfg.max_temporal_drift
        if max_drift is None:
            self.metrics_ = {}
            self._keep_idx = np.arange(len(names), dtype=np.intp)
            self.output_names_ = tuple(names)
            self.report_ = FilterReport(
                stage=self.name,
                n_in=len(names),
                n_out=len(names),
                dropped={},
                notes=("disabled: max_temporal_drift is None",),
            )
            return

        max_rho = self.cfg.max_lag1_autocorr
        n_blocks = self.cfg.stationarity_blocks
        metrics: dict[str, tuple[float, float]] = {}
        dropped: dict[str, str] = {}
        keep: list[int] = []

        for j, name in enumerate(names):
            col = X[:, j].astype(np.float64, copy=False)
            fin = np.isfinite(col)
            if int(fin.sum()) < 2:
                metrics[name] = (0.0, 0.0)
                keep.append(j)
                continue

            valid = col[fin]
            std = float(valid.std(dtype=np.float64))
            if not np.isfinite(std) or std < _EPS:
                metrics[name] = (0.0, 0.0)
                keep.append(j)
                continue

            rho1 = _lag1_autocorr(col, fin)
            drift = _temporal_drift(col, std, n_blocks)
            metrics[name] = (rho1, drift)

            if rho1 > max_rho and drift > max_drift:
                dropped[name] = (
                    f"non-stationary: lag1_autocorr={rho1:.5f} > {max_rho} "
                    f"and temporal_drift={drift:.3f} > {max_drift}"
                )
            else:
                keep.append(j)

        self.metrics_ = metrics
        self._keep_idx = np.asarray(keep, dtype=np.intp)
        self.output_names_ = tuple(names[j] for j in keep)
        self.report_ = FilterReport(
            stage=self.name,
            n_in=len(names),
            n_out=len(self.output_names_),
            dropped=dropped,
        )

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.ascontiguousarray(X[:, self._keep_idx], dtype=np.float32)


def _lag1_autocorr(col: NDArray[np.float64], fin: NDArray[np.bool_]) -> float:
    """Pearson correlation of contiguous finite adjacent pairs (col[:-1], col[1:])."""
    pair_ok = fin[:-1] & fin[1:]
    if int(pair_ok.sum()) < 2:
        return 0.0
    x0 = col[:-1][pair_ok]
    x1 = col[1:][pair_ok]
    c0 = x0 - x0.mean()
    c1 = x1 - x1.mean()
    denom = float(np.sqrt(np.dot(c0, c0) * np.dot(c1, c1)))
    if not np.isfinite(denom) or denom < _EPS:
        return 0.0
    return float(np.dot(c0, c1) / denom)


def _temporal_drift(col: NDArray[np.float64], std: float, n_blocks: int) -> float:
    """Range of contiguous chronological block means normalized by full-Train std."""
    means: list[float] = []
    for block in np.array_split(col, n_blocks):
        fin = np.isfinite(block)
        if fin.any():
            means.append(float(block[fin].mean()))
    if len(means) < 2:
        return 0.0
    return float((max(means) - min(means)) / std)
