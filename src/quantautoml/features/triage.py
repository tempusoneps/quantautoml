"""F1: NaN/Inf triage, missingness typing (MCAR/MAR/MNAR) and MI scoring."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.types import FilterReport
from quantautoml.features.base import BaseFeatureFilter, FitContext
from quantautoml.metrics.ic import calc_rank_ic

MissingKind = Literal["none", "MCAR", "MAR", "MNAR"]

_MID_NAN_LABEL = -1
_INDICATOR_PREFIX = "is_nan_"


class NaNInfTriage(BaseFeatureFilter):
    """Gate garbage columns, type their missingness and score them by MI.

    Fit (Train only):
      * ``±Inf`` is treated as ``NaN``.
      * Hard Gate: a column that is all NaN or has NaN ratio > ``cfg.max_nan_ratio`` is
        dropped (continuous, bool and categorical alike).
      * ``scores_[name] = MI(X_imputed, y) * (1 - nan_ratio)``; imputation (Train median,
        lower median for discrete columns so the value stays a valid code) is used for MI
        only. MI uses ``mutual_info_classif(..., discrete_features=mask, random_state=0)``
        on rows with ``y != -1`` (mid-NaN labels masked). Constant columns score ``0.0``.
      * Row scope: nan_ratio, MAR correlation and imputation use all Train rows; MI only ``y != -1``.
      * ``missing_kind_``: ``"none"`` without NaN; ``"MNAR"`` if MI(is_nan, y) >
        ``indicator_mi_min``; ``"MAR"`` if max |Spearman(is_nan, other kept feature)| >
        ``mar_corr_min``; otherwise ``"MCAR"``.
      * A discrete column whose observed values are a single ``0.0`` or ``1.0`` is treated
        as a one-sided bool flag: no ``is_nan_`` column; NaN is imputed with the
        complementary value (``1 - v``) in both MI scoring and transform.

    Transform applies only the fitted state: ``±Inf -> NaN``, keep non-gated columns
    (NaN left as-is except one-sided bool flags), then append float32 0/1 ``is_nan_<name>``
    columns for kept ``MAR``/``MNAR`` columns. Column order equals ``output_names_``.

    Args:
        cfg: Feature config; uses ``max_nan_ratio``.
        indicator_mi_min: MI threshold (nats) above which missingness is ``MNAR``.
        mar_corr_min: |Spearman| threshold above which missingness is ``MAR``.
    """

    name: ClassVar[str] = "f1_triage"

    def __init__(
        self,
        cfg: FeaturesConfig,
        *,
        indicator_mi_min: float = 0.01,
        mar_corr_min: float = 0.3,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.indicator_mi_min = indicator_mi_min
        self.mar_corr_min = mar_corr_min
        self.scores_: dict[str, float] = {}
        self.missing_kind_: dict[str, MissingKind] = {}
        self._keep_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._indicator_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._bool_fill: dict[int, float] = {}

    def _fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> None:
        names = ctx.feature_names
        discrete = set(ctx.discrete_features)
        Xc = np.where(np.isinf(X), np.nan, X).astype(np.float32)
        nan_mask = np.isnan(Xc)
        nan_ratio = nan_mask.mean(axis=0)

        dropped: dict[str, str] = {}
        keep: list[int] = []
        for j, name in enumerate(names):
            if nan_mask[:, j].all():
                dropped[name] = "all NaN/Inf"
            elif nan_ratio[j] > self.cfg.max_nan_ratio:
                dropped[name] = (
                    f"nan_ratio {nan_ratio[j]:.3f} > max_nan_ratio {self.cfg.max_nan_ratio}"
                )
            else:
                keep.append(j)

        bool_fill: dict[int, float] = {}
        imputed = np.empty((len(Xc), len(keep)), dtype=np.float64)
        for k, j in enumerate(keep):
            observed = Xc[~nan_mask[:, j], j]
            values = np.unique(observed)
            if names[j] in discrete and values.size == 1 and values[0] in (0.0, 1.0):
                fill = 1.0 - float(values[0])
                bool_fill[j] = fill
            elif names[j] in discrete:
                fill = float(np.quantile(observed, 0.5, method="lower"))
            else:
                fill = float(np.median(observed))
            imputed[:, k] = np.where(nan_mask[:, j], fill, Xc[:, j])

        feat_mi, ind_mi = self._mutual_info(imputed, nan_mask[:, keep], keep, names, discrete, y)

        scores: dict[str, float] = {}
        kinds: dict[str, MissingKind] = {}
        indicators: list[int] = []
        for k, j in enumerate(keep):
            name = names[j]
            scores[name] = float(feat_mi[k] * (1.0 - nan_ratio[j]))
            kind = self._missing_kind(k, ind_mi[k], nan_mask[:, j], Xc, keep)
            kinds[name] = kind
            if kind in ("MAR", "MNAR") and j not in bool_fill:
                indicators.append(j)

        self.scores_ = scores
        self.missing_kind_ = kinds
        self._keep_idx = np.asarray(keep, dtype=np.intp)
        self._indicator_idx = np.asarray(indicators, dtype=np.intp)
        self._bool_fill = bool_fill
        self.output_names_ = tuple(names[j] for j in keep) + tuple(
            f"{_INDICATOR_PREFIX}{names[j]}" for j in indicators
        )
        self.report_ = FilterReport(
            stage=self.name,
            n_in=len(names),
            n_out=len(self.output_names_),
            dropped=dropped,
        )

    def _mutual_info(
        self,
        imputed: NDArray[np.float64],
        kept_nan: NDArray[np.bool_],
        keep: list[int],
        names: tuple[str, ...],
        discrete: set[str],
        y: NDArray[np.int32],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One MI call over [non-constant features | non-constant indicators] on y != -1."""
        rows = y != _MID_NAN_LABEL
        n_keep = len(keep)
        feat_mi = np.zeros(n_keep, dtype=np.float64)
        ind_mi = np.zeros(n_keep, dtype=np.float64)
        yv = y[rows]
        if n_keep == 0 or np.unique(yv).size < 2:
            return feat_mi, ind_mi
        cols = np.hstack([imputed[rows], kept_nan[rows].astype(np.float64)])
        mask = np.array(
            [names[j] in discrete for j in keep] + [True] * n_keep, dtype=np.bool_
        )
        live = np.ptp(cols, axis=0) > 0.0
        if live.any():
            mi = np.zeros(2 * n_keep, dtype=np.float64)
            mi[live] = mutual_info_classif(
                cols[:, live], yv, discrete_features=mask[live], random_state=0
            )
            feat_mi, ind_mi = mi[:n_keep], mi[n_keep:]
        return feat_mi, ind_mi

    def _missing_kind(
        self,
        k: int,
        indicator_mi: float,
        is_nan: NDArray[np.bool_],
        Xc: NDArray[np.float32],
        keep: list[int],
    ) -> MissingKind:
        if not is_nan.any():
            return "none"
        if indicator_mi > self.indicator_mi_min:
            return "MNAR"
        indicator = is_nan.astype(np.float32)
        for other_k, other_j in enumerate(keep):
            if other_k == k:
                continue
            rho, _ = calc_rank_ic(Xc[:, other_j], indicator)
            if abs(rho) > self.mar_corr_min:
                return "MAR"
        return "MCAR"

    def _transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        Xc = np.where(np.isinf(X), np.nan, X).astype(np.float32)
        indicators = np.isnan(Xc[:, self._indicator_idx]).astype(np.float32)
        for j, fill in self._bool_fill.items():
            Xc[:, j] = np.where(np.isnan(Xc[:, j]), np.float32(fill), Xc[:, j])
        return np.hstack([Xc[:, self._keep_idx], indicators]).astype(np.float32)
