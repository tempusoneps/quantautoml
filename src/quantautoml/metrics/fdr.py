"""Benjamini-Hochberg FDR control (RULE-08)."""

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.exceptions import InvalidDataError


def apply_fdr_correction(
    p_values: NDArray[np.float64], alpha: float = 0.05
) -> NDArray[np.bool_]:
    """Return a mask of accepted hypotheses, in the original order.

    k = max{i : p_(i) <= i/M * alpha}; every p <= p_(k) is accepted.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return np.zeros(0, dtype=np.bool_)
    if np.isnan(p).any() or (p < 0.0).any() or (p > 1.0).any():
        raise InvalidDataError("p-values must be finite numbers in [0, 1]")
    m = p.size
    sorted_p = np.sort(p)
    passing = np.nonzero(sorted_p <= (np.arange(1, m + 1) / m) * alpha)[0]
    if passing.size == 0:
        return np.zeros(m, dtype=np.bool_)
    return p <= sorted_p[passing[-1]]
