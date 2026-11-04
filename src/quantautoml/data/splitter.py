"""Adaptive, leakage-free temporal splitter (5-way static or OOF-assisted) with purging."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.exceptions import DataLeakageError, InvalidDataError
from quantautoml.common.types import (
    Dataset,
    Segment,
    SegmentSummary,
    SplitMode,
    SplitReport,
    SplitResult,
)

STATIC_MIN_ROWS = 10_000
MIN_ROWS = 500
OOF_N_FOLDS = 5

# (segment name, cumulative percentage of N)
_STATIC_PLAN = (("train", 60), ("cal_a", 70), ("cal_b", 80), ("val", 90), ("oos", 100))
_OOF_PLAN = (("train", 80), ("val", 90), ("oos", 100))

IndexArray = NDArray[np.int64]


class AdaptiveTimeSeriesSplitter:
    """Chronological splitter. Rows are never shuffled; ``horizon_h`` rows are purged
    at the start of every segment after the first so forward-looking labels of the
    previous segment cannot reach into the next one."""

    def split(self, dataset: Dataset, horizon_h: int) -> SplitResult:
        if horizon_h < 1:
            raise InvalidDataError(f"horizon_h must be >= 1, got {horizon_h}")
        n = dataset.timestamps.shape[0]
        if n < MIN_ROWS:
            raise InvalidDataError(f"{n} rows < minimum of {MIN_ROWS} required to split")

        mode = SplitMode.STATIC_5WAY if n >= STATIC_MIN_ROWS else SplitMode.OOF_ASSISTED
        plan = _STATIC_PLAN if mode is SplitMode.STATIC_5WAY else _OOF_PLAN

        segments: dict[str, Segment] = {}
        bounds: dict[str, tuple[int, int]] = {}
        prev_cut = 0
        for i, (name, cum_pct) in enumerate(plan):
            cut = (n * cum_pct + 50) // 100
            start = prev_cut if i == 0 else prev_cut + horizon_h
            if cut - start <= 0:
                raise InvalidDataError(
                    f"horizon_h={horizon_h} leaves segment {name!r} empty for N={n}"
                )
            idx = np.arange(start, cut, dtype=np.int64)
            segments[name] = _take(dataset, idx)
            bounds[name] = (start, cut)
            prev_cut = cut

        oof_folds = (
            _oof_folds(bounds["train"][1], horizon_h)
            if mode is SplitMode.OOF_ASSISTED
            else None
        )
        result = SplitResult(
            mode=mode,
            train=segments["train"],
            val=segments["val"],
            oos=segments["oos"],
            cal_a=segments.get("cal_a"),
            cal_b=segments.get("cal_b"),
            oof_folds=oof_folds,
            report=_build_report(dataset, mode, n, horizon_h, segments, oof_folds),
        )
        verify_split(result, horizon_h)
        return result


def verify_split(result: SplitResult, horizon_h: int) -> None:
    """Assert the leakage invariants of a split; raise ``DataLeakageError`` otherwise."""
    static = result.mode is SplitMode.STATIC_5WAY
    if static != (result.cal_a is not None and result.cal_b is not None):
        raise DataLeakageError("calibration segments inconsistent with split mode")
    if static == (result.oof_folds is not None):
        raise DataLeakageError("oof_folds inconsistent with split mode")

    ordered = (
        [("train", result.train), ("cal_a", result.cal_a), ("cal_b", result.cal_b),
         ("val", result.val), ("oos", result.oos)]
        if static
        else [("train", result.train), ("val", result.val), ("oos", result.oos)]
    )
    for name, seg in ordered:
        assert seg is not None
        if seg.index.size == 0:
            raise DataLeakageError(f"segment {name!r} is empty")
        if np.any(np.diff(seg.index) <= 0) or np.any(np.diff(seg.timestamps) <= np.timedelta64(0, "ns")):
            raise DataLeakageError(f"segment {name!r} is not strictly chronological")
    for (a_name, a), (b_name, b) in zip(ordered, ordered[1:]):
        assert a is not None and b is not None
        gap = int(b.index[0] - a.index[-1] - 1)
        if gap != horizon_h:
            raise DataLeakageError(
                f"gap between {a_name!r} and {b_name!r} is {gap} rows, expected {horizon_h}"
            )
        if not a.timestamps[-1] < b.timestamps[0]:
            raise DataLeakageError(f"timestamps of {a_name!r} and {b_name!r} overlap")

    if result.oof_folds is not None:
        train_end = int(result.train.index[-1])
        for k, (tr, va) in enumerate(result.oof_folds):
            if tr.size == 0 or va.size == 0:
                raise DataLeakageError(f"OOF fold {k} has an empty part")
            if int(tr[-1]) > train_end or int(va[-1]) > train_end:
                raise DataLeakageError(f"OOF fold {k} reaches outside the train segment")
            gap = int(va[0] - tr[-1] - 1)
            if gap != horizon_h:
                raise DataLeakageError(
                    f"OOF fold {k}: gap {gap} rows between train and validation, expected {horizon_h}"
                )


# --------------------------------------------------------------------------- #
def _take(ds: Dataset, idx: IndexArray) -> Segment:
    return Segment(
        X=ds.X[idx],
        y=None if ds.y is None else ds.y[idx],
        labels_aux={k: v[idx] for k, v in ds.labels_aux.items()},
        timestamps=ds.timestamps[idx],
        index=idx,
    )


def _oof_folds(train_end: int, h: int) -> list[tuple[IndexArray, IndexArray]]:
    """Expanding walk-forward folds on rows [0, train_end): ``OOF_N_FOLDS + 1`` blocks."""
    n_blocks = OOF_N_FOLDS + 1
    edges = [(train_end * j * 2 + n_blocks) // (2 * n_blocks) for j in range(n_blocks + 1)]
    folds: list[tuple[IndexArray, IndexArray]] = []
    for k in range(OOF_N_FOLDS):
        train_stop = edges[k + 1]
        val_start, val_stop = train_stop + h, edges[k + 2]
        if val_stop - val_start <= 0:
            raise InvalidDataError(
                f"horizon_h={h} leaves OOF fold {k} with an empty validation block"
            )
        folds.append(
            (np.arange(0, train_stop, dtype=np.int64),
             np.arange(val_start, val_stop, dtype=np.int64))
        )
    return folds


def _fmt(value: object) -> str:
    f = float(value)  # type: ignore[arg-type]
    return str(int(f)) if f.is_integer() else str(f)


def _build_report(
    ds: Dataset,
    mode: SplitMode,
    n: int,
    h: int,
    segments: dict[str, Segment],
    oof_folds: list[tuple[IndexArray, IndexArray]] | None,
) -> SplitReport:
    summaries: list[SegmentSummary] = []
    warnings: list[str] = []
    for i, (name, seg) in enumerate(segments.items()):
        counts: dict[str, int] | None = None
        if seg.y is not None and ds.classes_ is not None:
            raw = np.bincount(seg.y, minlength=len(ds.classes_))
            counts = {_fmt(c): int(raw[j]) for j, c in enumerate(ds.classes_)}
            warnings.extend(f"class {c} missing in {name}" for c, v in counts.items() if v == 0)
        summaries.append(
            SegmentSummary(
                name=name,
                n_rows=int(seg.index.size),
                t_start=str(np.datetime_as_string(seg.timestamps[0])),
                t_end=str(np.datetime_as_string(seg.timestamps[-1])),
                rows_dropped_before=0 if i == 0 else h,
                class_counts=counts,
            )
        )
    fold_sizes = (
        None if oof_folds is None else tuple((int(t.size), int(v.size)) for t, v in oof_folds)
    )
    return SplitReport(
        mode=mode, n_rows=n, horizon_h=h, segments=tuple(summaries),
        oof_fold_sizes=fold_sizes, warnings=tuple(warnings),
    )
