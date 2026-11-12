"""F6: regime-aware mutual information and synergistic-pair protection."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mutual_info_score

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import FilterReport
from quantautoml.features.base import BaseFeatureFilter, FitContext

_MID_NAN_LABEL = -1
_H1 = 1
_N_BINS = 5
_MIN_REGIME_SAMPLES = 50
_TOP_K_SYNERGY = 30
_MIN_SIGMA_OBS = 2
_REGIME_NAMES: dict[int, str] = {1: "Bull", -1: "Bear", 0: "Sideways"}


def label_regimes(
    ret_h1: NDArray[np.float32], window: int = 63, band: float = 0.5
) -> NDArray[np.int8]:
    """Causal Bull (1) / Bear (-1) / Sideways (0) labels from forward 1-bar returns.

    ``ret_h1[s]`` is the return from bar ``s`` to ``s + 1``, realized at ``s + 1``, so at
    bar ``t`` only ``ret_h1[:t]`` is known (the 1-bar shift). The trailing sum is
    ``S_t = sum(ret_h1[t - window : t])`` with NaN/+-inf counted as 0 and not as a
    realized observation. ``S_t`` is valid iff ``t >= window`` (``min_history = window``)
    and the window holds at least one realized observation. ``sigma_t`` is the expanding
    sample std (ddof=1) of the valid sums ``S_s`` with ``s < t``; it needs at least 2 of
    them. Label: ``1`` if ``S_t > band * sigma_t``, ``-1`` if ``S_t < -band * sigma_t``,
    else ``0``; invalid ``S_t``, too few sums or ``sigma_t == 0`` give ``0``. Hence
    ``label[t]`` depends only on ``ret_h1[:t]`` (modifying ``ret_h1[t:]`` never alters
    ``label[: t + 1]``), and the threshold uses Train information only.

    Raises:
        InvalidDataError: ``ret_h1`` not 1-D, ``window < 1`` or ``band < 0``.
    """
    if window < 1:
        raise InvalidDataError(f"label_regimes: window must be >= 1, got {window}")
    if band < 0.0:
        raise InvalidDataError(f"label_regimes: band must be >= 0, got {band}")
    r = np.asarray(ret_h1, dtype=np.float64)
    if r.ndim != 1:
        raise InvalidDataError(f"label_regimes: ret_h1 must be 1-D, got ndim={r.ndim}")
    n = r.shape[0]
    out = np.zeros(n, dtype=np.int8)
    if n <= window:
        return out
    finite = np.isfinite(r)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, r, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    t = np.arange(window, n)
    s = csum[t] - csum[t - window]
    valid = (ccnt[t] - ccnt[t - window]) > 0
    sv = np.where(valid, s, 0.0)
    # expanding stats over valid sums strictly before t (exclusive prefix)
    k = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))[:-1]
    m1 = np.concatenate(([0.0], np.cumsum(sv)))[:-1]
    m2 = np.concatenate(([0.0], np.cumsum(sv * sv)))[:-1]
    ok = valid & (k >= _MIN_SIGMA_OBS)
    kk = np.maximum(k, _MIN_SIGMA_OBS).astype(np.float64)
    var = np.maximum((m2 - m1 * m1 / kk) / (kk - 1.0), 0.0)
    thr = band * np.sqrt(var)
    ok &= var > 0.0
    lab = np.where(s > thr, 1, np.where(s < -thr, -1, 0))
    out[window:] = np.where(ok, lab, 0).astype(np.int8)
    return out


class RegimeSynergyFilter(BaseFeatureFilter):
    """Keep features informative in at least one regime or part of a synergistic pair.

    Regimes come from ``label_regimes(labels_aux[cfg.ic_horizons[1]], cfg.rolling_window)``.
    If horizon 1 is not configured or its column is absent the filter is a pass-through
    (all columns kept, reason in ``report_.notes``). MI is the plug-in
    ``mutual_info_score`` (nats) on 5 quantile bins (edges from the Train rows with
    ``y != -1``; +-inf -> NaN, NaN imputed with the Train median); rows with ``y == -1``
    are masked. ``mi_by_regime_[name][r]`` is the MI with ``y`` inside regime ``r``;
    regimes with < 50 valid rows are skipped and noted. If every regime is skipped the
    MI gate keeps all columns (noted). Synergy is checked among the top-30 features by
    overall MI: ``I(Xi,Xj;Y) - I(Xi;Y) - I(Xj;Y) > synergy_delta`` with the joint label
    ``bin_i * 5 + bin_j``. A feature is kept iff its best regime MI ``>= mi_floor`` or it
    is in ``synergy_pairs_``. ``output_names_`` keeps the original column order;
    transform only selects columns.

    Args:
        mi_floor: minimum best-regime MI (nats). Default 0.01 sits well above the
            plug-in bias ``(5-1)(C-1)/(2 n_r)`` of a pure-noise binary feature in a
            regime of a few hundred rows.
        synergy_delta: minimum interaction information (nats), default 0.01.

    Raises:
        InvalidDataError: negative thresholds, or a misaligned h=1 label column.
    """

    name: ClassVar[str] = "f6_regime_synergy"

    def __init__(
        self, cfg: FeaturesConfig, *, mi_floor: float = 0.01, synergy_delta: float = 0.01
    ) -> None:
        super().__init__()
        if mi_floor < 0.0 or synergy_delta < 0.0:
            raise InvalidDataError(
                f"{self.name}: mi_floor={mi_floor} and synergy_delta={synergy_delta} "
                "must be >= 0"
            )
        self.cfg = cfg
        self.mi_floor = mi_floor
        self.synergy_delta = synergy_delta
        self.mi_by_regime_: dict[str, dict[int, float]] = {}
        self.synergy_pairs_: tuple[tuple[str, str], ...] = ()
        self._keep_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def _h1_returns(self, ctx: FitContext, n: int) -> tuple[NDArray[np.float32] | None, str]:
        col = self.cfg.ic_horizons.get(_H1)
        if col is None:
            return None, f"{self.name}: no h=1 horizon in ic_horizons; pass-through"
        if col not in ctx.labels_aux:
            return None, f"{self.name}: labels_aux lacks h=1 column '{col}'; pass-through"
        arr = np.asarray(ctx.labels_aux[col], dtype=np.float32)
        if arr.shape != (n,):
            raise InvalidDataError(
                f"{self.name}: labels_aux['{col}'] has shape {arr.shape}, expected ({n},)"
            )
        return arr, ""

    @staticmethod
    def _bins(X: NDArray[np.float32]) -> NDArray[np.intp]:
        """Per-column 5-quantile bin codes ``0..4``; constant/all-NaN columns -> all 0."""
        Xc = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
        codes = np.zeros(Xc.shape, dtype=np.intp)
        qs = np.linspace(0.0, 1.0, _N_BINS + 1)[1:-1]
        for j in range(Xc.shape[1]):
            col = Xc[:, j]
            nan = np.isnan(col)
            if nan.all():
                continue
            col = np.where(nan, float(np.median(col[~nan])), col)
            edges = np.unique(np.quantile(col, qs))
            codes[:, j] = np.searchsorted(edges, col, side="right")
        return codes

    def _pass_through(self, names: tuple[str, ...], note: str) -> None:
        self._keep_idx = np.arange(len(names), dtype=np.intp)
        self.output_names_ = names
        self.report_ = FilterReport(
            stage=self.name, n_in=len(names), n_out=len(names), dropped={}, notes=(note,)
        )

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        names = ctx.feature_names
        p = len(names)
        self.mi_by_regime_ = {}
        self.synergy_pairs_ = ()
        ret, note = self._h1_returns(ctx, X.shape[0])
        if ret is None:
            self._pass_through(names, note)
            return

        rows = y != _MID_NAN_LABEL
        yv = y[rows]
        regime = label_regimes(ret, window=self.cfg.rolling_window)[rows]
        codes = self._bins(X[rows])
        notes: list[str] = []
        mi_reg: dict[str, dict[int, float]] = {nm: {} for nm in names}
        for r in (1, -1, 0):
            m = regime == r
            cnt = int(m.sum())
            if cnt < _MIN_REGIME_SAMPLES:
                notes.append(
                    f"regime {r} ({_REGIME_NAMES[r]}) has {cnt} valid samples "
                    f"< {_MIN_REGIME_SAMPLES}: skipped"
                )
                continue
            for j, nm in enumerate(names):
                mi_reg[nm][r] = float(mutual_info_score(codes[m, j], yv[m]))
        evaluated = p > 0 and any(mi_reg[nm] for nm in names)
        if p > 0 and not evaluated:
            notes.append("no regime has enough samples: MI gate keeps all features")

        overall = np.asarray(
            [float(mutual_info_score(codes[:, j], yv)) for j in range(p)], dtype=np.float64
        )
        top = sorted(sorted(range(p), key=lambda j: (-overall[j], j))[:_TOP_K_SYNERGY])
        pairs: list[tuple[int, int]] = []
        for a, i in enumerate(top):
            for j in top[a + 1 :]:
                joint = codes[:, i] * _N_BINS + codes[:, j]
                gain = float(mutual_info_score(joint, yv)) - overall[i] - overall[j]
                if gain > self.synergy_delta:
                    pairs.append((i, j))
        protected = {k for pair in pairs for k in pair}

        keep: list[int] = []
        dropped: dict[str, str] = {}
        for j, nm in enumerate(names):
            best = max(mi_reg[nm].values(), default=0.0)
            if not evaluated or best >= self.mi_floor or j in protected:
                keep.append(j)
            else:
                dropped[nm] = (
                    f"best regime MI {best:.4g} < mi_floor {self.mi_floor:.4g} "
                    "and not in a synergy pair"
                )

        self.mi_by_regime_ = mi_reg
        self.synergy_pairs_ = tuple((names[i], names[j]) for i, j in pairs)
        self._keep_idx = np.asarray(keep, dtype=np.intp)
        self.output_names_ = tuple(names[j] for j in keep)
        self.report_ = FilterReport(
            stage=self.name,
            n_in=p,
            n_out=len(self.output_names_),
            dropped=dropped,
            notes=tuple(notes),
        )

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.ascontiguousarray(X[:, self._keep_idx], dtype=np.float32)
