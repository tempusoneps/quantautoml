"""FeatureFunnel: chain the seven leakage-free filters F1 -> F2 -> F3+7 -> F4 -> F5 -> F6 -> F8."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Self

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig
from quantautoml.common.exceptions import DataLeakageError, FeatureSelectionError, InvalidDataError
from quantautoml.common.types import Dataset, FilterReport, Segment, SplitResult, TaskType
from quantautoml.features.base import BaseFeatureFilter, FitContext
from quantautoml.features.clustering import ICBehaviorClusterFilter
from quantautoml.features.collinearity import FastNonlinearCollinearityFilter
from quantautoml.features.ic_stability import ICStabilityFilter
from quantautoml.features.jmi import JMIFeatureSelector
from quantautoml.features.regime_synergy import RegimeSynergyFilter
from quantautoml.features.triage import NaNInfTriage
from quantautoml.features.variance import MultiMetricVarianceFilter

StageFactory = Callable[[FeaturesConfig], BaseFeatureFilter]

STAGE_FACTORIES: tuple[StageFactory, ...] = (
    NaNInfTriage,
    MultiMetricVarianceFilter,
    ICStabilityFilter,
    FastNonlinearCollinearityFilter,
    ICBehaviorClusterFilter,
    RegimeSynergyFilter,
    JMIFeatureSelector,
)

SKIPPED_NOTE = "skipped: no features left"


class FeatureFunnel:
    """Fit the seven feature filters in order on Train only, then transform any segment.

    ``fit`` reads only ``split.train`` (X, y, labels_aux, index) plus
    ``dataset.task_type`` / ``dataset.feature_names`` / ``dataset.report`` metadata; the
    other segments are touched only for the index-overlap leakage check. Stage ``k``
    is fitted on the transformed Train X of stage ``k-1`` with a ``FitContext`` whose
    ``feature_names`` are that stage's ``output_names_``, ``labels_aux`` are the Train
    rows of ``split.train.labels_aux`` and ``discrete_features`` are the surviving
    bool / string-encoded ingestion columns plus any ``is_nan_*`` indicators F1 adds.

    If a stage leaves no feature, the remaining stages are not fitted: their reports
    have ``n_in = n_out = 0`` and the note ``"skipped: no features left"``;
    ``selected_features_`` is ``()`` and ``transform`` returns ``(N, 0)`` float32.

    Attributes:
        selected_features_: names of the final columns, in ``transform`` output order.
        reports_: exactly seven ``FilterReport`` in stage order (``n_out`` of stage k
            equals ``n_in`` of stage k+1).

    Raises:
        InvalidDataError: unsupervised task, Train without ``y``, or a feature-name /
            column count mismatch.
        DataLeakageError: Train index not strictly increasing, or overlapping the index
            of Val / OOS / Cal-A / Cal-B (checked before any filter is fitted).
        FeatureSelectionError: ``transform`` before ``fit`` or with a wrong column count.
    """

    def __init__(self, cfg: FeaturesConfig) -> None:
        self.cfg = cfg
        self.selected_features_: tuple[str, ...] = ()
        self.reports_: tuple[FilterReport, ...] = ()
        self.is_fitted_: bool = False
        self._stages: tuple[BaseFeatureFilter, ...] = ()
        self._n_features_in: int = 0

    def fit(self, split: SplitResult, dataset: Dataset) -> Self:
        train = split.train
        if dataset.task_type is None or dataset.task_type == TaskType.UNSUPERVISED:
            raise InvalidDataError("FeatureFunnel: a supervised task is required")
        if train.y is None:
            raise InvalidDataError("FeatureFunnel: Train segment has no labels (y is None)")
        _check_train_index(split)
        names = tuple(dataset.feature_names)
        if len(names) != train.X.shape[1]:
            raise InvalidDataError(
                f"FeatureFunnel: {len(names)} feature names for {train.X.shape[1]} Train columns"
            )

        self.is_fitted_ = False
        self.selected_features_, self.reports_, self._stages = (), (), ()
        labels = {k: np.asarray(v, dtype=np.float32) for k, v in train.labels_aux.items()}
        report = dataset.report
        discrete = set(report.bool_columns_cast) | set(report.string_columns_encoded)
        X: NDArray[np.float32] = train.X
        fitted: list[BaseFeatureFilter] = []
        reports: list[FilterReport] = []
        for factory in STAGE_FACTORIES:
            stage = factory(self.cfg)
            if not names:
                skipped = FilterReport(
                    stage=stage.name, n_in=0, n_out=0, dropped={}, notes=(SKIPPED_NOTE,)
                )
                reports.append(skipped)
                continue
            ctx = FitContext(
                feature_names=names,
                labels_aux=labels,
                task_type=dataset.task_type,
                discrete_features=tuple(n for n in names if n in discrete),
            )
            stage.fit(X, train.y, ctx)
            if stage.report_ is None:
                raise FeatureSelectionError(f"FeatureFunnel: {stage.name} produced no report")
            X = stage.transform(X)
            # Columns a stage adds (only F1's is_nan_* indicators) are 0/1 flags.
            discrete |= set(stage.output_names_) - set(names)
            names = stage.output_names_
            fitted.append(stage)
            reports.append(stage.report_)

        self._stages = tuple(fitted)
        self._n_features_in = train.X.shape[1]
        self.selected_features_ = names
        self.reports_ = tuple(reports)
        self.is_fitted_ = True
        return self

    def transform(self, seg: Segment) -> Segment:
        if not self.is_fitted_:
            raise FeatureSelectionError("FeatureFunnel: transform called before fit")
        if seg.X.ndim != 2 or seg.X.shape[1] != self._n_features_in:
            raise FeatureSelectionError(
                f"FeatureFunnel: expected {self._n_features_in} columns, got shape {seg.X.shape}"
            )
        X = seg.X
        if not self._stages:
            X = np.empty((seg.X.shape[0], 0), dtype=np.float32)
        for stage in self._stages:
            X = stage.transform(X)
        return replace(seg, X=np.ascontiguousarray(X, dtype=np.float32))


def _check_train_index(split: SplitResult) -> None:
    """RULE-01: Train rows are chronological and disjoint from every other segment."""
    idx = np.asarray(split.train.index)
    if idx.ndim != 1 or np.any(np.diff(idx) <= 0):
        raise DataLeakageError("FeatureFunnel: Train index is not strictly increasing")
    others = {"val": split.val, "oos": split.oos, "cal_a": split.cal_a, "cal_b": split.cal_b}
    for name, seg in others.items():
        if seg is not None and np.intersect1d(idx, seg.index).size > 0:
            raise DataLeakageError(f"FeatureFunnel: Train index overlaps the {name} index")
