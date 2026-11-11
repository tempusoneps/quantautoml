import dataclasses
import json

import numpy as np
import pytest

from quantautoml.common.exceptions import (
    DataLeakageError,
    InvalidDataError,
    QuantAutoMLError,
    TimestampAlignmentError,
)
from quantautoml.common.types import (
    Dataset,
    IngestionReport,
    Segment,
    SegmentSummary,
    SplitMode,
    SplitReport,
    SplitResult,
    TaskType,
)


@pytest.mark.parametrize("exc", [InvalidDataError, TimestampAlignmentError, DataLeakageError])
def test_exceptions_inherit_from_base_error(exc: type[Exception]) -> None:
    assert issubclass(exc, QuantAutoMLError)
    with pytest.raises(QuantAutoMLError):
        raise exc("boom")


def test_enum_values_are_stable_strings() -> None:
    assert TaskType.BINARY_CLASSIFICATION.value == "binary_classification"
    assert TaskType.MULTICLASS_CLASSIFICATION.value == "multiclass_classification"
    assert TaskType.UNSUPERVISED.value == "unsupervised"
    assert SplitMode.STATIC_5WAY.value == "static_5way"
    assert SplitMode.OOF_ASSISTED.value == "oof_assisted"


def _segment(n: int = 3) -> Segment:
    return Segment(
        X=np.zeros((n, 2), dtype=np.float32),
        y=np.zeros(n, dtype=np.int32),
        labels_aux={"fwd_ret_21": np.zeros(n, dtype=np.float32)},
        timestamps=np.arange(n).astype("datetime64[ns]"),
        index=np.arange(n),
    )


def test_segment_is_frozen() -> None:
    seg = _segment()
    with pytest.raises(dataclasses.FrozenInstanceError):
        seg.X = np.zeros((1, 1), dtype=np.float32)  # type: ignore[misc]


def test_ingestion_report_to_dict_is_json_serialisable() -> None:
    report = IngestionReport(
        n_feature_rows=100,
        n_label_rows=100,
        n_joined_rows=98,
        n_unmatched_dropped=2,
        n_trailing_label_nan_dropped=5,
        source_timezone="UTC",
        naive_utc_assumed=False,
        n_features=4,
        n_nan=3,
        n_inf=1,
        t_min="2024-01-01T00:00:00",
        t_max="2024-02-01T00:00:00",
        label_nan_counts={"fwd_ret_5": 5, "fwd_ret_21": 21},
        max_trailing_nan=21,
        horizon_h=21,
    )
    d = report.to_dict()
    assert d["n_joined_rows"] == 98
    assert d["label_nan_counts"] == {"fwd_ret_5": 5, "fwd_ret_21": 21}
    json.dumps(d)


def test_split_report_to_dict_converts_enum_and_nests_segments() -> None:
    report = SplitReport(
        mode=SplitMode.STATIC_5WAY,
        n_rows=10_000,
        horizon_h=21,
        segments=(
            SegmentSummary(
                name="train",
                n_rows=6000,
                t_start="2024-01-01T00:00:00",
                t_end="2024-03-01T00:00:00",
                rows_dropped_before=0,
                class_counts={"0": 3000, "1": 3000},
            ),
        ),
        oof_fold_sizes=None,
        warnings=("class 2 missing in oos",),
    )
    d = report.to_dict()
    assert d["mode"] == "static_5way"
    assert d["segments"][0]["name"] == "train"
    assert d["segments"][0]["class_counts"] == {"0": 3000, "1": 3000}
    assert d["warnings"] == ["class 2 missing in oos"]
    json.dumps(d)


def test_dataset_defaults_task_type_and_classes_to_none() -> None:
    report = IngestionReport(
        n_feature_rows=1, n_label_rows=1, n_joined_rows=1, n_unmatched_dropped=0,
        n_trailing_label_nan_dropped=0, source_timezone="UTC", naive_utc_assumed=False,
        n_features=1, n_nan=0, n_inf=0, t_min="a", t_max="b",
        label_nan_counts={}, max_trailing_nan=0, horizon_h=1,
    )
    ds = Dataset(
        X=np.zeros((1, 1), dtype=np.float32),
        y=None,
        labels_aux={},
        timestamps=np.arange(1).astype("datetime64[ns]"),
        feature_names=("f0",),
        selected_label=None,
        report=report,
    )
    assert ds.task_type is None
    assert ds.classes_ is None


def test_split_result_optional_parts_for_oof_mode() -> None:
    seg = _segment()
    report = SplitReport(
        mode=SplitMode.OOF_ASSISTED, n_rows=3, horizon_h=1, segments=(),
        oof_fold_sizes=((2, 1),), warnings=(),
    )
    res = SplitResult(
        mode=SplitMode.OOF_ASSISTED,
        train=seg, val=seg, oos=seg,
        cal_a=None, cal_b=None,
        oof_folds=[(np.array([0, 1]), np.array([2]))],
        report=report,
    )
    assert res.cal_a is None and res.cal_b is None
    assert res.oof_folds is not None
