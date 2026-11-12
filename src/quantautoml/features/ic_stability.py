"""F3+7: rolling Rank-IC stability, BH-FDR gate and IC term-structure monotonicity."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import FilterReport
from quantautoml.features.base import BaseFeatureFilter, FitContext
from quantautoml.metrics.fdr import apply_fdr_correction
from quantautoml.metrics.ic import calc_icir, calc_rank_ic, calc_rolling_ic_matrix


class ICStabilityFilter(BaseFeatureFilter):
    """Keep features with stable, significant, monotonically decaying Rank IC (Train only).

    For each horizon ``h`` in ``sorted(cfg.ic_horizons)`` the label is
    ``ctx.labels_aux[cfg.ic_horizons[h]]`` (forward returns; NaN pairs are masked,
    ``y`` is never used). Rolling Rank IC uses non-overlapping windows of
    ``cfg.rolling_window`` rows; ``ICIR(h) = mean / std`` of that series.

    A feature is kept only if all gates hold:
      1. ``icir``: ``|ICIR(h_target)| > cfg.icir_min``; and ``sign``: the mean rolling IC
         has the same sign at every horizon whose mean rolling IC is non-zero
         (horizons with mean exactly 0.0, e.g. all windows degenerate, are ignored).
      2. ``fdr``: the full-Train Rank-IC p-value at ``h_target`` is accepted by one
         Benjamini-Hochberg call (``alpha = cfg.fdr_alpha``) over every tested feature,
         degenerate ones included with ``p = 1.0``.
      3. ``monotonic``: ``|ICIR(h_i)| >= |ICIR(h_{i+1})| - cfg.monotonic_tolerance``
         for consecutive horizons sorted ascending.

    ``report_.dropped[name]`` lists the failed gate(s). Transform only selects columns.

    Raises:
        InvalidDataError: at fit if a horizon's label column is missing or misaligned,
            or Train has fewer than ``2 * cfg.rolling_window`` rows.
    """

    name: ClassVar[str] = "f3_ic_stability"

    def __init__(self, cfg: FeaturesConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.icir_by_horizon_: dict[str, dict[int, float]] = {}
        self.p_values_: dict[str, float] = {}
        self._keep_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def _labels(self, ctx: FitContext, n: int) -> dict[int, NDArray[np.float32]]:
        labels: dict[int, NDArray[np.float32]] = {}
        for h in sorted(self.cfg.ic_horizons):
            col = self.cfg.ic_horizons[h]
            if col not in ctx.labels_aux:
                raise InvalidDataError(
                    f"{self.name}: labels_aux is missing column '{col}' for horizon {h}"
                )
            arr = np.asarray(ctx.labels_aux[col], dtype=np.float32)
            if arr.shape != (n,):
                raise InvalidDataError(
                    f"{self.name}: labels_aux['{col}'] has shape {arr.shape}, expected ({n},)"
                )
            labels[h] = arr
        return labels

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        cfg, names = self.cfg, ctx.feature_names
        n, p = X.shape
        if n < 2 * cfg.rolling_window:
            raise InvalidDataError(
                f"{self.name}: Train has {n} rows, fewer than 2 * rolling_window="
                f"{2 * cfg.rolling_window}"
            )
        labels = self._labels(ctx, n)
        horizons = list(labels)

        icir = np.zeros((len(horizons), p), dtype=np.float64)
        mean_ic = np.zeros((len(horizons), p), dtype=np.float64)
        for k, h in enumerate(horizons):
            roll = calc_rolling_ic_matrix(X, labels[h], cfg.rolling_window)
            for j in range(p):
                series = roll[:, j]
                valid = series[~np.isnan(series)]
                mean_ic[k, j] = float(valid.mean()) if valid.size else 0.0
                icir[k, j] = calc_icir(series)

        target = labels[cfg.target_horizon]
        p_vals = np.array(
            [calc_rank_ic(X[:, j], target)[1] for j in range(p)], dtype=np.float64
        )
        fdr_ok = apply_fdr_correction(p_vals, cfg.fdr_alpha)

        t = horizons.index(cfg.target_horizon)
        abs_icir = np.abs(icir)
        signs = np.sign(mean_ic)
        icir_ok = abs_icir[t] > cfg.icir_min
        sign_ok = ~((signs > 0).any(axis=0) & (signs < 0).any(axis=0))
        mono_ok = np.all(abs_icir[:-1] >= abs_icir[1:] - cfg.monotonic_tolerance, axis=0)

        dropped: dict[str, str] = {}
        keep: list[int] = []
        for j, name in enumerate(names):
            failed: list[str] = []
            if not icir_ok[j]:
                failed.append(f"icir |{icir[t, j]:.4g}| <= icir_min {cfg.icir_min}")
            if not sign_ok[j]:
                failed.append("sign: mean rolling IC changes sign across horizons")
            if not fdr_ok[j]:
                failed.append(f"fdr: p={p_vals[j]:.4g} rejected by BH at alpha={cfg.fdr_alpha}")
            if not mono_ok[j]:
                failed.append("monotonic: |ICIR| increases across horizons beyond tolerance")
            if failed:
                dropped[name] = "; ".join(failed)
            else:
                keep.append(j)

        self.icir_by_horizon_ = {
            name: {h: float(icir[k, j]) for k, h in enumerate(horizons)}
            for j, name in enumerate(names)
        }
        self.p_values_ = {name: float(p_vals[j]) for j, name in enumerate(names)}
        self._keep_idx = np.asarray(keep, dtype=np.intp)
        self.output_names_ = tuple(names[j] for j in keep)
        self.report_ = FilterReport(
            stage=self.name, n_in=len(names), n_out=len(self.output_names_), dropped=dropped
        )

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.ascontiguousarray(X[:, self._keep_idx], dtype=np.float32)
