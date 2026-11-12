"""F2: multi-metric variance check (CV, peak rolling variance, unique ratio)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import FilterReport
from quantautoml.features.base import BaseFeatureFilter, FitContext

_EPS = 1e-12
_PEAK_PERCENTILE = 90.0


class MultiMetricVarianceFilter(BaseFeatureFilter):
    """Drop degenerate columns using three variance-based metrics (Train only).

    ``metrics_[name] = (cv, peak_var_ratio, unique_ratio)`` where
      * ``cv = nanstd / max(|nanmean|, eps)``,
      * ``peak_var_ratio`` = 90th percentile of the rolling variance (window
        ``cfg.rolling_window``, ddof=0, NaN entries ignored inside each window, windows with fewer than 2
        valid values skipped; ``0.0`` if no window qualifies) divided by the
        full-Train variance (``0.0`` if that variance is zero),
      * ``unique_ratio = n_unique / n_valid``.

    A column is dropped if ``n_unique < 2``, or (``n_unique > 2`` and
    ``unique_ratio < min_unique_ratio``), or ``|cv| < min_cv``, or
    ``peak_var_ratio > max_peak_var_ratio``. All-NaN columns are dropped. Transform only
    selects columns.

    Note:
        The default ``max_peak_var_ratio=50`` is rarely reachable: the 90th percentile of
        rolling variance is bounded near ~10x the full variance for typical series, so the
        threshold needs calibration on real data.

    Args:
        cfg: Feature config; uses ``min_unique_ratio``, ``min_cv``,
            ``max_peak_var_ratio`` and ``rolling_window``.

    Raises:
        InvalidDataError: at fit if Train has fewer rows than ``rolling_window``.
    """

    name: ClassVar[str] = "f2_variance"

    def __init__(self, cfg: FeaturesConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.metrics_: dict[str, tuple[float, float, float]] = {}
        self._keep_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        names = ctx.feature_names
        window = self.cfg.rolling_window
        n = X.shape[0]
        if n < window:
            raise InvalidDataError(
                f"{self.name}: Train has {n} rows, fewer than rolling_window={window}"
            )

        peak = self._peak_var_ratio(X, window)
        metrics: dict[str, tuple[float, float, float]] = {}
        dropped: dict[str, str] = {}
        keep: list[int] = []
        for j, name in enumerate(names):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if valid.size == 0:
                metrics[name] = (float("nan"), float("nan"), 0.0)
                dropped[name] = "all NaN"
                continue
            n_unique = int(np.unique(valid).size)
            unique_ratio = n_unique / valid.size
            mean = float(valid.mean(dtype=np.float64))
            cv = float(valid.std(dtype=np.float64)) / max(abs(mean), _EPS)
            metrics[name] = (cv, float(peak[j]), unique_ratio)

            reason: str | None = None
            if n_unique < 2:
                reason = f"n_unique {n_unique} < 2"
            elif n_unique > 2 and unique_ratio < self.cfg.min_unique_ratio:
                reason = (
                    f"unique_ratio {unique_ratio:.4g} < min_unique_ratio "
                    f"{self.cfg.min_unique_ratio}"
                )
            elif abs(cv) < self.cfg.min_cv:
                reason = f"|cv| {abs(cv):.4g} < min_cv {self.cfg.min_cv}"
            elif peak[j] > self.cfg.max_peak_var_ratio:
                reason = (
                    f"peak_var_ratio {peak[j]:.4g} > max_peak_var_ratio "
                    f"{self.cfg.max_peak_var_ratio}"
                )
            if reason is None:
                keep.append(j)
            else:
                dropped[name] = reason

        self.metrics_ = metrics
        self._keep_idx = np.asarray(keep, dtype=np.intp)
        self.output_names_ = tuple(names[j] for j in keep)
        self.report_ = FilterReport(
            stage=self.name,
            n_in=len(names),
            n_out=len(self.output_names_),
            dropped=dropped,
        )

    @staticmethod
    def _peak_var_ratio(X: NDArray[np.float32], window: int) -> NDArray[np.float64]:
        """P90 of rolling variance / full variance per column, via cumulative sums."""
        n, p = X.shape
        out = np.zeros(p, dtype=np.float64)
        Xd = X.astype(np.float64)
        isnan = np.isnan(Xd)
        center = np.zeros(p, dtype=np.float64)
        has_valid = ~isnan.all(axis=0)
        center[has_valid] = np.nanmean(Xd[:, has_valid], axis=0)
        Z = np.where(isnan, 0.0, Xd - center)
        zeros = np.zeros((1, p), dtype=np.float64)
        s1 = np.vstack([zeros, np.cumsum(Z, axis=0)])
        s2 = np.vstack([zeros, np.cumsum(Z * Z, axis=0)])
        cnt = np.vstack([zeros, np.cumsum(isnan, axis=0, dtype=np.float64)])
        valid_cnt = window - (cnt[window:] - cnt[:-window])
        w1 = s1[window:] - s1[:-window]
        w2 = s2[window:] - s2[:-window]
        full_var = np.zeros(p, dtype=np.float64)
        full_var[has_valid] = np.nanvar(Xd[:, has_valid], axis=0)
        for j in range(p):
            ok = valid_cnt[:, j] >= 2.0
            if not has_valid[j] or not ok.any() or full_var[j] <= 0.0:
                continue
            m = valid_cnt[ok, j]
            rv = np.maximum(w2[ok, j] / m - (w1[ok, j] / m) ** 2, 0.0)
            out[j] = np.percentile(rv, _PEAK_PERCENTILE) / full_var[j]
        return out

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.ascontiguousarray(X[:, self._keep_idx], dtype=np.float32)
