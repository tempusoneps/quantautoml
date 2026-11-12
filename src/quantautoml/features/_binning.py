"""Private helpers shared by F6 and F8: quantile bin codes and contingency-table MI."""

from __future__ import annotations

from math import log

import numpy as np
from numpy.typing import NDArray


def quantile_bin_codes(
    X: NDArray[np.float32], n_bins: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Per-column ``n_bins``-quantile bin codes and the code range of each column.

    +-inf -> NaN, NaN imputed with the column median (callers pass Train rows only, so
    edges and median are Train statistics). Edges are ``np.unique(np.quantile(...))`` of
    the interior quantiles (duplicate edges collapse into fewer bins) and codes are
    ``np.searchsorted(edges, x, side="right")`` in ``[0, n_bins_per_column[j])``. A
    constant column has a single non-empty bin; an all-NaN column is all code 0 with
    range 1. Either way its MI with any label is exactly 0.
    """
    Xc = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
    codes = np.zeros(Xc.shape, dtype=np.intp)
    n_per_col = np.ones(Xc.shape[1], dtype=np.intp)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        nan = np.isnan(col)
        if nan.all():
            continue
        col = np.where(nan, float(np.median(col[~nan])), col)
        edges = np.unique(np.quantile(col, qs))
        codes[:, j] = np.searchsorted(edges, col, side="right")
        n_per_col[j] = edges.size + 1
    return codes, n_per_col


def contingency_mi(a: NDArray[np.intp], na: int, yc: NDArray[np.intp], nc: int) -> float:
    """Plug-in MI (nats) of codes ``a`` in ``[0, na)`` vs class codes ``yc`` in ``[0, nc)``.

    Dense integer contingency via ``np.bincount``. Empty rows / columns are ignored; fewer
    than 2 non-empty rows or columns (incl. empty input) gives exactly ``0.0``. Otherwise
    the arithmetic is that of ``sklearn.metrics.mutual_info_score(contingency=...)``
    (empty cells skipped, ``|term| < eps`` -> 0, result clipped at 0), so the value is
    bit-identical to it.
    """
    if a.size == 0:
        return 0.0
    ct = np.bincount(a * nc + yc, minlength=na * nc).reshape(na, nc).astype(np.int64)
    pi, pj = ct.sum(axis=1), ct.sum(axis=0)
    if (pi > 0).sum() < 2 or (pj > 0).sum() < 2:
        return 0.0  # a single non-empty bin or class: MI is exactly 0
    nzx, nzy = np.nonzero(ct)
    nz_val = ct[nzx, nzy]
    total = ct.sum()
    contingency_nm = nz_val / total
    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = contingency_nm * (np.log(nz_val) - log(total)) + contingency_nm * log_outer
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return float(np.clip(mi.sum(), 0.0, None))
