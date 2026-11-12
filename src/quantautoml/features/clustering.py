"""F5: IC-behavior hierarchical clustering with a Max-MI representative per cluster."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import FeatureSelectionError, InvalidDataError
from quantautoml.common.types import FilterReport
from quantautoml.features.base import BaseFeatureFilter, FitContext
from quantautoml.metrics.ic import calc_rank_ic, calc_rolling_ic_matrix

_MIN_COMMON_WINDOWS = 3
_MID_NAN_LABEL = -1


def ic_behavior_distance(roll: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pairwise ``d_ij = 1 - |corr(IC_i, IC_j)|`` over rolling-IC columns ``(W, p)``.

    Pearson is taken over windows where both series are finite. Fewer than 3 common
    windows, or zero variance on either side there, gives ``d = 1.0``. The result is
    symmetric, has a zero diagonal and is finite everywhere.
    """
    p = roll.shape[1]
    finite = np.isfinite(roll)
    d = np.ones((p, p), dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    for i in range(p):
        for j in range(i + 1, p):
            m = finite[:, i] & finite[:, j]
            if int(m.sum()) < _MIN_COMMON_WINDOWS:
                continue
            a = roll[m, i] - roll[m, i].mean()
            b = roll[m, j] - roll[m, j].mean()
            denom = float(np.sqrt((a @ a) * (b @ b)))
            if denom == 0.0:
                continue
            r = min(abs(float(a @ b) / denom), 1.0)
            d[i, j] = d[j, i] = 1.0 - r
    return d


class ICBehaviorClusterFilter(BaseFeatureFilter):
    """Cluster features by rolling-IC behavior and keep one Max-MI feature per cluster.

    The behavior series of a feature is its non-overlapping rolling Rank IC
    (``cfg.rolling_window``) against ``ctx.labels_aux[cfg.ic_horizons[cfg.target_horizon]]``
    (+-inf feature values masked like NaN). Distances come from ``ic_behavior_distance``;
    clusters from ``linkage(method="average")`` + ``fcluster(criterion="distance",
    t=cfg.cluster_distance)``. Cluster ids are ``0..k-1`` ordered by each cluster's
    smallest original column index.

    The representative maximizes ``mi_[name]`` = ``mutual_info_classif(random_state=0)``
    with ``y`` on rows ``y != -1`` (non-finite values imputed with the Train median, lower
    median for ``ctx.discrete_features``, which are flagged discrete); ties -> larger
    ``|rank_ic_|`` at the target horizon, then the earlier column. ``output_names_`` keeps
    the original column order; ``report_.dropped[name]`` names its cluster's
    representative. Transform only selects columns.

    Raises:
        InvalidDataError: at fit if the target label column is missing or misaligned,
            or Train has fewer than ``2 * cfg.rolling_window`` rows.
        FeatureSelectionError: the distance matrix is not finite (defensive).
    """

    name: ClassVar[str] = "f5_clustering"

    def __init__(self, cfg: FeaturesConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.clusters_: dict[int, tuple[str, ...]] = {}
        self.representatives_: dict[int, str] = {}
        self.mi_: dict[str, float] = {}
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

    def _cluster(self, roll: NDArray[np.float64]) -> NDArray[np.intp]:
        """Raw fcluster labels, remapped to ids ordered by smallest member column."""
        p = roll.shape[1]
        if p < 2:
            return np.zeros(p, dtype=np.intp)
        d = ic_behavior_distance(roll)
        if not np.isfinite(d).all():
            raise FeatureSelectionError(f"{self.name}: non-finite IC distance matrix")
        z = linkage(squareform(d), method="average")
        raw = fcluster(z, t=self.cfg.cluster_distance, criterion="distance")
        remap: dict[int, int] = {}
        for lab in raw.tolist():
            remap.setdefault(int(lab), len(remap))
        return np.asarray([remap[int(lab)] for lab in raw.tolist()], dtype=np.intp)

    def _mutual_info(
        self, Xc: NDArray[np.float64], y: NDArray[np.int32], discrete: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        p = Xc.shape[1]
        mi = np.zeros(p, dtype=np.float64)
        rows = y != _MID_NAN_LABEL
        yv = y[rows]
        if p == 0 or np.unique(yv).size < 2:
            return mi
        imputed = Xc[rows].copy()
        for j in range(p):
            col = imputed[:, j]
            nan = np.isnan(col)
            observed = col[~nan]
            if observed.size == 0:
                fill = 0.0
            elif discrete[j]:
                fill = float(np.quantile(observed, 0.5, method="lower"))
            else:
                fill = float(np.median(observed))
            col[nan] = fill
        live = np.ptp(imputed, axis=0) > 0.0
        if live.any():
            mi[live] = mutual_info_classif(
                imputed[:, live], yv, discrete_features=discrete[live], random_state=0
            )
        return mi

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        names = ctx.feature_names
        n, p = X.shape
        window = self.cfg.rolling_window
        if n < 2 * window:
            raise InvalidDataError(
                f"{self.name}: Train has {n} rows, fewer than 2 * rolling_window={2 * window}"
            )
        target = self._target(ctx, n)
        Xc = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
        Xc32 = Xc.astype(np.float32)

        roll = calc_rolling_ic_matrix(Xc32, target, window)
        ids = self._cluster(roll)
        disc_set = set(ctx.discrete_features)
        discrete = np.asarray([nm in disc_set for nm in names], dtype=np.bool_)
        mi = self._mutual_info(Xc, y, discrete)
        ic = np.asarray(
            [calc_rank_ic(Xc32[:, j], target)[0] for j in range(p)], dtype=np.float64
        )

        clusters: dict[int, tuple[str, ...]] = {}
        reps: dict[int, str] = {}
        keep: list[int] = []
        dropped: dict[str, str] = {}
        for cid in range(int(ids.max()) + 1 if p else 0):
            members = [j for j in range(p) if ids[j] == cid]
            rep = min(members, key=lambda j: (-mi[j], -abs(ic[j]), j))
            clusters[cid] = tuple(names[j] for j in members)
            reps[cid] = names[rep]
            keep.append(rep)
            for j in members:
                if j != rep:
                    dropped[names[j]] = (
                        f"cluster {cid}: represented by '{names[rep]}' "
                        f"(MI {mi[j]:.4g} vs {mi[rep]:.4g}, |IC| {abs(ic[j]):.4g} vs "
                        f"{abs(ic[rep]):.4g})"
                    )

        keep.sort()
        self.clusters_ = clusters
        self.representatives_ = reps
        self.mi_ = {name: float(mi[j]) for j, name in enumerate(names)}
        self.rank_ic_ = {name: float(ic[j]) for j, name in enumerate(names)}
        self._keep_idx = np.asarray(keep, dtype=np.intp)
        self.output_names_ = tuple(names[j] for j in keep)
        self.report_ = FilterReport(
            stage=self.name, n_in=len(names), n_out=len(self.output_names_), dropped=dropped
        )

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.ascontiguousarray(X[:, self._keep_idx], dtype=np.float32)
