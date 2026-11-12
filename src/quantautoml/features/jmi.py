"""F8: greedy Joint Mutual Information (JMI) feature selection."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.types import FilterReport
from quantautoml.features.base import BaseFeatureFilter, FitContext

_MID_NAN_LABEL = -1
_N_BINS = 10


def _mi(a: NDArray[np.intp], na: int, yc: NDArray[np.intp], nc: int) -> float:
    """Plug-in MI (nats) of codes ``a`` in ``[0, na)`` vs class codes ``yc`` in ``[0, nc)``.

    Integer contingency via ``np.bincount``; same formula as
    ``sklearn.metrics.mutual_info_score`` (empty cells skipped, result clipped at 0).
    """
    if a.size == 0:
        return 0.0
    ct = np.bincount(a * nc + yc, minlength=na * nc).reshape(na, nc)
    rows, cols = ct.sum(axis=1), ct.sum(axis=0)
    if (rows > 0).sum() < 2 or (cols > 0).sum() < 2:
        return 0.0  # a single non-empty bin or class: MI is exactly 0
    nzx, nzy = np.nonzero(ct)
    nz = ct[nzx, nzy].astype(np.float64)
    total = float(a.size)
    outer = rows[nzx].astype(np.float64) * cols[nzy].astype(np.float64)
    mi = (nz / total) * (np.log(nz) + np.log(total) - np.log(outer))
    mi = np.where(np.abs(mi) < np.finfo(np.float64).eps, 0.0, mi)
    return float(max(mi.sum(), 0.0))


def _bins(X: NDArray[np.float32]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Per-column 10-quantile bin codes and bin counts (duplicate edges collapse).

    +-inf -> NaN, NaN imputed with the column median; constant / all-NaN -> a single
    non-empty bin (MI exactly 0).
    """
    Xc = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
    codes = np.zeros(Xc.shape, dtype=np.intp)
    n_bins = np.ones(Xc.shape[1], dtype=np.intp)
    qs = np.linspace(0.0, 1.0, _N_BINS + 1)[1:-1]
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        nan = np.isnan(col)
        if nan.all():
            continue
        col = np.where(nan, float(np.median(col[~nan])), col)
        edges = np.unique(np.quantile(col, qs))
        codes[:, j] = np.searchsorted(edges, col, side="right")
        n_bins[j] = edges.size + 1
    return codes, n_bins


class JMIFeatureSelector(BaseFeatureFilter):
    """Greedy JMI: keep the ``min(cfg.max_features, n_features)`` best features.

    Every feature is discretized into 10 quantile bins (edges from the Train rows with
    ``y != -1``; duplicate edges collapse into fewer bins; constant / all-NaN columns
    become a single non-empty bin with MI 0). Rows with ``y == -1`` are masked out of every MI.
    MI is the plug-in estimate in nats (equal to ``sklearn.metrics.mutual_info_score``).
    The first pick maximizes ``I(f; Y)`` (``mi_``; tie -> earlier column). Each next pick
    maximizes ``J(f) = sum_{s in S} I(f, s; Y)`` over the selected set ``S`` (joint code
    ``bin_f * n_bins_s + bin_s``), ties broken by larger ``I(f; Y)`` then earlier column.
    ``J`` is updated incrementally (one joint MI per candidate per step).

    Standard JMI caveat: ``J`` only sums pairwise terms with *selected* features, so
    redundancy/synergy of higher order (3+ features) is not modelled, and candidates
    compete only against what has already been chosen.

    ``selected_order_`` is the greedy order; ``output_names_`` lists the same features in
    the original column order and ``transform`` only selects those columns.
    """

    name: ClassVar[str] = "f8_jmi"

    def __init__(self, cfg: FeaturesConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.selected_order_: tuple[str, ...] = ()
        self.mi_: dict[str, float] = {}
        self._keep_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    @staticmethod
    def _greedy(
        codes: NDArray[np.intp],
        n_bins: NDArray[np.intp],
        yc: NDArray[np.intp],
        nc: int,
        marginal: NDArray[np.float64],
        k: int,
    ) -> list[int]:
        """Greedy JMI order of ``k`` column indices (cached bin codes, incremental J)."""
        p = marginal.size
        if k == 0:
            return []
        order = [min(range(p), key=lambda j: (-marginal[j], j))]
        remaining = set(range(p)) - set(order)
        score = np.zeros(p, dtype=np.float64)
        while len(order) < k:
            s = order[-1]
            for f in remaining:
                joint = codes[:, f] * n_bins[s] + codes[:, s]
                score[f] += _mi(joint, int(n_bins[f] * n_bins[s]), yc, nc)
            best = min(remaining, key=lambda j: (-score[j], -marginal[j], j))
            order.append(best)
            remaining.discard(best)
        return order

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        names = ctx.feature_names
        p = len(names)
        k = min(self.cfg.max_features, p)
        rows = y != _MID_NAN_LABEL
        classes, yc_raw = np.unique(y[rows], return_inverse=True)
        yc = yc_raw.astype(np.intp)
        nc = max(int(classes.size), 1)
        codes, n_bins = _bins(X[rows])
        marginal = np.asarray(
            [_mi(codes[:, j], int(n_bins[j]), yc, nc) for j in range(p)], dtype=np.float64
        )
        order = self._greedy(codes, n_bins, yc, nc, marginal, k)

        chosen = set(order)
        dropped = {
            nm: f"not selected by JMI (rank > max_features={self.cfg.max_features})"
            for j, nm in enumerate(names)
            if j not in chosen
        }
        notes: tuple[str, ...] = ()
        if dropped:
            notes = (f"truncated {p} features to max_features={self.cfg.max_features}",)

        self.mi_ = {nm: float(marginal[j]) for j, nm in enumerate(names)}
        self.selected_order_ = tuple(names[j] for j in order)
        self._keep_idx = np.asarray(sorted(order), dtype=np.intp)
        self.output_names_ = tuple(names[j] for j in self._keep_idx)
        self.report_ = FilterReport(
            stage=self.name,
            n_in=p,
            n_out=len(self.output_names_),
            dropped=dropped,
            notes=notes,
        )

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.ascontiguousarray(X[:, self._keep_idx], dtype=np.float32)
