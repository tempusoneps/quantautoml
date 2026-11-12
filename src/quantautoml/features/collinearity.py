"""F4: fast nonlinear collinearity filter based on distance correlation (dCor, AVL)."""

from __future__ import annotations

from typing import ClassVar

import dcor
import numpy as np
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import FilterReport
from quantautoml.features.base import BaseFeatureFilter, FitContext
from quantautoml.metrics.ic import calc_rank_ic

# dcor's AVL implementation asserts ``n > 3``, so 4 is the smallest usable sample.
_MIN_PAIRS = 4


def _standardize(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z-score (dCor is invariant to shift and positive scaling) to avoid overflow."""
    return (a - a.mean()) / a.std()


def fast_dcor(x: NDArray[np.float32], y: NDArray[np.float32]) -> float:
    """Distance correlation of two 1-D samples in O(N log N) memory-light AVL form.

    Pairs where either value is NaN or +-inf are dropped. Fewer than 4 valid pairs
    (AVL requires n > 3) or a constant side -> ``0.0``. Inputs are cast to ``float64``
    right before the call (RULE-06 exception: with float32 ``dcor`` falls back to an
    uncompiled path and emits a ``UserWarning``); the cast copy is not kept.

    Raises:
        InvalidDataError: ``x`` and ``y`` shapes differ.
        FeatureSelectionError: ``dcor`` returns a non-finite value despite the guards.
    """
    xa = np.asarray(x)
    ya = np.asarray(y)
    if xa.shape != ya.shape:
        raise InvalidDataError(f"fast_dcor: x and y shape mismatch: {xa.shape} vs {ya.shape}")
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < _MIN_PAIRS:
        return 0.0
    xv = xa[mask].astype(np.float64)
    yv = ya[mask].astype(np.float64)
    if np.ptp(xv) == 0.0 or np.ptp(yv) == 0.0:
        return 0.0
    value = float(
        dcor.distance_correlation(_standardize(xv), _standardize(yv), method="AVL")
    )
    if not np.isfinite(value):
        raise FeatureSelectionError(f"fast_dcor: dcor returned non-finite value {value}")
    return min(max(value, 0.0), 1.0)


class FastNonlinearCollinearityFilter(BaseFeatureFilter):
    """Greedy dCor de-duplication ordered by |Rank IC| at the target horizon (Train only).

    ``rank_ic_[name]`` is the Spearman Rank IC of the feature against
    ``ctx.labels_aux[cfg.ic_horizons[cfg.target_horizon]]`` (non-finite feature values
    and NaN labels are masked pairwise; degenerate -> 0.0). Features are visited by
    ``|IC|`` descending (ties: earlier original column first); a feature is dropped
    if ``fast_dcor`` with any already-kept feature exceeds ``cfg.dcor_threshold``,
    otherwise it is kept. Only pairs against kept features are evaluated (O(F*K)),
    no N x N or F x F matrix is built.

    ``output_names_`` keeps the original column order; ``report_.dropped[name]``
    names the kept feature that caused the drop and the dCor value. Transform only
    selects columns.

    Raises:
        InvalidDataError: at fit if the target label column is missing or misaligned.
    """

    name: ClassVar[str] = "f4_collinearity"

    def __init__(self, cfg: FeaturesConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.rank_ic_: dict[str, float] = {}
        self._keep_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def _target(self, ctx: FitContext, n: int) -> NDArray[np.float32]:
        col = self.cfg.ic_horizons[self.cfg.target_horizon]
        if col not in ctx.labels_aux:
            raise InvalidDataError(
                f"{self.name}: labels_aux is missing column '{col}' for target horizon "
                f"{self.cfg.target_horizon}"
            )
        arr = np.asarray(ctx.labels_aux[col], dtype=np.float32)
        if arr.shape != (n,):
            raise InvalidDataError(
                f"{self.name}: labels_aux['{col}'] has shape {arr.shape}, expected ({n},)"
            )
        return arr

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        names = ctx.feature_names
        n, p = X.shape
        target = self._target(ctx, n)
        ic = np.zeros(p, dtype=np.float64)
        for j in range(p):
            col = X[:, j].copy()
            col[~np.isfinite(col)] = np.nan  # +-inf masked like NaN (as in fast_dcor)
            ic[j] = calc_rank_ic(col, target)[0]

        # stable sort on -|IC| -> ties keep original column order
        order = np.argsort(-np.abs(ic), kind="stable")
        thr = self.cfg.dcor_threshold
        kept: list[int] = []
        dropped: dict[str, str] = {}
        for j in order.tolist():
            for k in kept:
                d = fast_dcor(X[:, j], X[:, k])
                if d > thr:
                    dropped[names[j]] = (
                        f"dcor={d:.4f} > {thr} with kept '{names[k]}' "
                        f"(|IC| {abs(ic[j]):.4g} <= {abs(ic[k]):.4g})"
                    )
                    break
            else:
                kept.append(j)

        keep = sorted(kept)
        self.rank_ic_ = {name: float(ic[j]) for j, name in enumerate(names)}
        self._keep_idx = np.asarray(keep, dtype=np.intp)
        self.output_names_ = tuple(names[j] for j in keep)
        self.report_ = FilterReport(
            stage=self.name, n_in=len(names), n_out=len(self.output_names_), dropped=dropped
        )

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.ascontiguousarray(X[:, self._keep_idx], dtype=np.float32)
