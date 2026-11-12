"""F6: regime-aware mutual information and synergistic-pair protection."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import FilterReport
from quantautoml.features._binning import contingency_mi, quantile_bin_codes
from quantautoml.features.base import BaseFeatureFilter, FitContext

_MID_NAN_LABEL = -1
_H1 = 1
_N_BINS = 5
_MIN_REGIME_SAMPLES = 50
_TOP_K_SYNERGY = 30
_MIN_SIGMA_OBS = 2
_NULL_QUANTILE = 0.99
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
    (all columns kept, reason in ``report_.notes``). MI is the plug-in contingency MI
    (nats) on 5 quantile bins (edges from the Train rows with ``y != -1``; +-inf -> NaN,
    NaN imputed with the Train median); rows with ``y == -1`` are masked.

    Regime gate (significance test). ``mi_by_regime_[name][r]`` is the *size-corrected*
    MI with ``y`` inside regime ``r``: ``max(0, I_plugin - chi2.ppf(0.99, df) / (2 n_r))``
    with ``df = (B_eff - 1)(C_eff - 1)``, ``B_eff`` the non-empty bins of the feature and
    ``C_eff`` the classes of ``y`` present in that regime. Since ``2 n_r I_plugin`` is the
    G-statistic (~ chi2(df) under independence), the corrected MI is > 0 iff the feature
    is dependent on ``y`` in that regime at the 1% level; it removes the plug-in bias *and*
    its null spread, so noise in a small regime (the max over regimes is the gate) is not
    kept (a plain Miller-Madow ``df / (2 n_r)`` term still kept ~21% / ~32% of noise at
    ``n_r = 100``). ``mi_floor`` is an *additional* effect-size floor on top of that test:
    a feature passes iff its best corrected regime MI is ``> mi_floor`` (strict, so the
    default ``0.0`` means "significant in at least one regime"; with ``>=`` a floor of 0
    would keep everything). Regimes with < 50 valid rows are skipped and noted; if every
    regime is skipped the MI gate keeps all columns (noted).

    Synergy gate. Among the top-30 features by uncorrected overall MI (full valid Train
    sample, ``N`` rows), a pair is synergistic iff the bias-corrected interaction gain
    ``I(Xi,Xj;Y) - I(Xi;Y) - I(Xj;Y) - chi2.ppf(0.99, df_ij) / (2N) > synergy_delta``,
    joint label ``bin_i * 5 + bin_j``, ``df_ij = (B_i - 1)(B_j - 1)(C - 1)`` (see
    ``_fit`` for the derivation). The top-30 cap (spec-mandated O(K^2) bound) can miss
    XOR-type pairs, whose members have ~0 marginal MI, when more than 30 features carry
    marginal signal.

    A feature is kept iff its best corrected regime MI ``> mi_floor`` or it is in
    ``synergy_pairs_``. ``output_names_`` keeps the original column order; transform only
    selects columns.

    Args:
        cfg: ``mi_floor`` (default 0.0) and ``synergy_delta`` (default 0.01, nats) are
            read from it.
        mi_floor: optional override of ``cfg.mi_floor``.
        synergy_delta: optional override of ``cfg.synergy_delta``.

    Raises:
        InvalidDataError: negative thresholds, or a misaligned h=1 label column.
    """

    name: ClassVar[str] = "f6_regime_synergy"

    def __init__(
        self,
        cfg: FeaturesConfig,
        *,
        mi_floor: float | None = None,
        synergy_delta: float | None = None,
    ) -> None:
        super().__init__()
        mi_floor = cfg.mi_floor if mi_floor is None else mi_floor
        synergy_delta = cfg.synergy_delta if synergy_delta is None else synergy_delta
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
        classes, yc = np.unique(y[rows], return_inverse=True)
        nc = max(int(classes.size), 1)

        def mi(a: NDArray[np.intp], na: int, sel: NDArray[np.bool_] | None = None) -> float:
            return contingency_mi(a, na, yc if sel is None else yc[sel], nc)

        regime = label_regimes(ret, window=self.cfg.rolling_window)[rows]
        codes, _ = quantile_bin_codes(X[rows], _N_BINS)
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
            c_eff = int(np.unique(yc[m]).size)
            for j, nm in enumerate(names):
                cm = codes[m, j]
                df = (int(np.unique(cm).size) - 1) * (c_eff - 1)
                null = float(chi2.ppf(_NULL_QUANTILE, df)) / (2 * cnt) if df > 0 else 0.0
                mi_reg[nm][r] = max(0.0, mi(cm, _N_BINS, m) - null)
        evaluated = p > 0 and any(mi_reg[nm] for nm in names)
        if p > 0 and not evaluated:
            notes.append("no regime has enough samples: MI gate keeps all features")

        overall = np.asarray(
            [mi(codes[:, j], _N_BINS) for j in range(p)], dtype=np.float64
        )
        top = sorted(sorted(range(p), key=lambda j: (-overall[j], j))[:_TOP_K_SYNERGY])
        # Null of the interaction gain. With Xi, Xj independent of Y, 2N * I(Xi,Xj;Y) ~
        # chi2((Bi*Bj - 1)(C - 1)) and 2N * I(Xk;Y) ~ chi2((Bk - 1)(C - 1)); the gain is
        # the G-statistic of the Y x (Xi,Xj) interaction left after both main effects, so
        # its df is (Bi*Bj - 1 - (Bi - 1) - (Bj - 1))(C - 1) = (Bi - 1)(Bj - 1)(C - 1)
        # (mean bias df / (2N); 16 / (2N) binary, 32 / (2N) 3-class at 5 bins). B are the
        # non-empty bins of each member and C the classes present after the y == -1 mask.
        # When (Xi, Xj) are dependent some joint cells are empty and the true df is lower,
        # so this full-table df is conservative (it only removes spurious pairs).
        n_valid = int(yc.size)
        b_eff = [int(np.unique(codes[:, j]).size) for j in range(p)]
        pairs: list[tuple[int, int]] = []
        for a, i in enumerate(top):
            for j in top[a + 1 :]:
                df_ij = (b_eff[i] - 1) * (b_eff[j] - 1) * (nc - 1)
                if df_ij <= 0:
                    continue  # a single-bin member or one class: no interaction possible
                joint = codes[:, i] * _N_BINS + codes[:, j]
                gain = mi(joint, _N_BINS * _N_BINS) - overall[i] - overall[j]
                gain -= float(chi2.ppf(_NULL_QUANTILE, df_ij)) / (2 * n_valid)
                if gain > self.synergy_delta:
                    pairs.append((i, j))
        protected = {k for pair in pairs for k in pair}

        keep: list[int] = []
        dropped: dict[str, str] = {}
        for j, nm in enumerate(names):
            best = max(mi_reg[nm].values(), default=0.0)
            if not evaluated or best > self.mi_floor or j in protected:
                keep.append(j)
            else:
                dropped[nm] = (
                    f"best corrected regime MI {best:.4g} <= mi_floor {self.mi_floor:.4g} "
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
