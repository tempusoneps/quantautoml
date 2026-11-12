"""Spearman Rank IC, non-overlapping rolling IC and ICIR (RULE-02)."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from quantautoml.common.exceptions import InvalidDataError

_MIN_PAIRS = 3


def calc_rank_ic(
    x: NDArray[np.float32],
    y: NDArray[np.float32 | np.int32],
    *,
    ignore_value: float | None = None,
) -> tuple[float, float]:
    """Return (Spearman rho, p-value); degenerate inputs yield (0.0, 1.0).

    Pairs with NaN in either array, or with ``y == ignore_value``, are dropped.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape:
        raise InvalidDataError(f"x and y shape mismatch: {xa.shape} vs {ya.shape}")
    mask = ~(np.isnan(xa) | np.isnan(ya))
    if ignore_value is not None:
        mask &= ya != ignore_value
    xv, yv = xa[mask], ya[mask]
    if xv.size < _MIN_PAIRS or np.ptp(xv) == 0.0 or np.ptp(yv) == 0.0:
        return 0.0, 1.0
    res = spearmanr(xv, yv)
    rho, p = float(res.statistic), float(res.pvalue)
    if np.isnan(rho) or np.isnan(p):
        return 0.0, 1.0
    return rho, p


def calc_rolling_ic(
    x: NDArray[np.float32],
    y: NDArray[np.float32 | np.int32],
    window: int,
    *,
    ignore_value: float | None = None,
) -> NDArray[np.float64]:
    """Rank IC on non-overlapping windows; length ``n // window``.

    Degenerate windows give NaN. A series shorter than ``window`` returns an
    empty array.
    """
    if window < 1:
        raise InvalidDataError(f"window must be >= 1, got {window}")
    n_windows = len(x) // window
    out = np.full(n_windows, np.nan, dtype=np.float64)
    for i in range(n_windows):
        sl = slice(i * window, (i + 1) * window)
        xs, ys = x[sl], y[sl]
        xa = np.asarray(xs, dtype=np.float64)
        ya = np.asarray(ys, dtype=np.float64)
        mask = ~(np.isnan(xa) | np.isnan(ya))
        if ignore_value is not None:
            mask &= ya != ignore_value
        if mask.sum() < _MIN_PAIRS or np.ptp(xa[mask]) == 0.0 or np.ptp(ya[mask]) == 0.0:
            continue
        out[i] = calc_rank_ic(xs, ys, ignore_value=ignore_value)[0]
    return out


def calc_icir(ic_series: NDArray[np.float64]) -> float:
    """Return mean/std (ddof=1, NaN ignored); 0.0 if std == 0 or < 2 points."""
    v = np.asarray(ic_series, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return 0.0
    std = float(np.std(v, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(v)) / std
