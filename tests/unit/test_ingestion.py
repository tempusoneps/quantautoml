from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from quantautoml.common.exceptions import InvalidDataError, TimestampAlignmentError
from quantautoml.common.types import TaskType
from quantautoml.data.ingestion import DataIngestion


def _load(cfg):
    return DataIngestion(cfg).load()


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_loads_parquet_and_csv(make_frames, build_config, fmt: str) -> None:
    f, l = make_frames(n=100)
    ds = _load(build_config(f, l, fmt=fmt))
    assert ds.X.dtype == np.float32
    assert ds.feature_names == ("f0", "f1", "f2")
    assert ds.timestamps.dtype == np.dtype("datetime64[ns]")
    assert ds.y is not None and ds.y.dtype == np.int32
    assert ds.task_type is TaskType.BINARY_CLASSIFICATION
    assert ds.selected_label == "y_dir"
    assert set(ds.labels_aux) == {"fwd_ret_1", "fwd_ret_5", "fwd_ret_21"}
    assert all(v.dtype == np.float32 for v in ds.labels_aux.values())


def test_timezone_aware_timestamps_are_converted_to_utc(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    local = lambda df: df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Asia/Bangkok")
    )
    ds = _load(build_config(local(f), local(l)))
    expected = (
        f["timestamp"].dt.replace_time_zone(None).to_numpy().astype("datetime64[ns]")
    )
    np.testing.assert_array_equal(ds.timestamps[:100 - 5], expected[: 100 - 5])
    assert "Asia/Bangkok" in ds.report.source_timezone
    assert ds.report.naive_assumed_utc is False


def test_naive_timestamps_are_assumed_utc_and_flagged(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    naive = lambda df: df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    ds = _load(build_config(naive(f), naive(l)))
    assert ds.report.naive_assumed_utc is True
    assert ds.timestamps[0] == np.datetime64("2024-01-01T00:00:00", "ns")


def test_duplicate_timestamps_are_rejected(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    f = pl.concat([f, f.head(1)]).sort("timestamp")
    with pytest.raises(TimestampAlignmentError):
        _load(build_config(f, l))


def test_shuffled_timestamps_are_rejected_not_sorted(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    with pytest.raises(TimestampAlignmentError):
        _load(build_config(f.sample(fraction=1.0, shuffle=True, seed=1), l))


def test_inner_join_drops_unmatched_rows_and_reports_them(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    l = l.filter(~pl.col("timestamp").is_in(f["timestamp"][10:13].to_list()))
    ds = _load(build_config(f, l))
    assert ds.report.n_feature_rows == 100
    assert ds.report.n_label_rows == 97
    assert ds.report.n_joined_rows == 97
    assert ds.report.n_unmatched_dropped == 3


def test_low_match_ratio_is_rejected(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    shifted = l.with_columns(pl.col("timestamp") + timedelta(minutes=30))
    shifted = pl.concat([shifted.head(10), l.tail(90).with_columns(pl.col("timestamp") + timedelta(days=100))])
    with pytest.raises(TimestampAlignmentError):
        _load(build_config(f, shifted.sort("timestamp"), min_match_ratio=0.95))


def test_non_numeric_feature_is_rejected(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    f = f.with_columns(pl.lit("x").alias("bad"))
    with pytest.raises(InvalidDataError, match="bad"):
        _load(build_config(f, l))


def test_non_numeric_label_column_is_rejected(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    l = l.with_columns(pl.lit("x").alias("regime"))
    with pytest.raises(InvalidDataError, match="regime"):
        _load(build_config(f, l))


def test_float32_overflow_is_rejected(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    f = f.with_columns(
        pl.when(pl.int_range(pl.len()) == 7).then(1e39).otherwise(pl.col("f0")).alias("f0")
    )
    with pytest.raises(InvalidDataError, match="float32"):
        _load(build_config(f, l))


def test_nan_and_inf_in_features_are_preserved_and_counted(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    f = f.with_columns(
        pl.when(pl.int_range(pl.len()) == 3).then(float("nan"))
        .when(pl.int_range(pl.len()) == 4).then(float("inf"))
        .otherwise(pl.col("f1")).alias("f1")
    )
    ds = _load(build_config(f, l))
    assert np.isnan(ds.X[3, 1]) and np.isinf(ds.X[4, 1])
    assert ds.report.n_nan == 1 and ds.report.n_inf == 1


def test_label_columns_must_not_collide_with_feature_columns(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    f = f.with_columns(pl.col("f0").alias("fwd_ret_1"))
    with pytest.raises(InvalidDataError, match="fwd_ret_1"):
        _load(build_config(f, l))


def test_selected_label_missing_lists_available_columns(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    with pytest.raises(InvalidDataError, match="y_dir"):
        _load(build_config(f, l, selected="nope"))


def test_trailing_nan_of_selected_label_is_dropped_and_reported(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    ds = _load(build_config(f, l))
    assert ds.X.shape[0] == 95 and ds.timestamps.shape[0] == 95
    assert ds.y is not None and ds.y.shape[0] == 95
    assert ds.report.n_trailing_label_nan_dropped == 5
    assert ds.report.n_joined_rows == 100
    assert all(v.shape[0] == 95 for v in ds.labels_aux.values())
    assert ds.timestamps[-1] == np.datetime64("2024-01-01T00:00:00", "ns") + np.timedelta64(94, "h")


def test_nan_in_middle_of_selected_label_is_rejected(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    l = l.with_columns(
        pl.when(pl.int_range(pl.len()) == 20).then(None).otherwise(pl.col("y_dir")).alias("y_dir")
    )
    with pytest.raises(InvalidDataError):
        _load(build_config(f, l))


def test_auxiliary_labels_keep_their_nan(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    ds = _load(build_config(f, l))
    # fwd_ret_21 has 21 trailing NaN; 5 rows were dropped, 16 remain
    assert int(np.isnan(ds.labels_aux["fwd_ret_21"]).sum()) == 16
    assert ds.report.label_nan_counts["fwd_ret_21"] == 21
    assert ds.report.max_trailing_nan == 21


def test_horizon_smaller_than_trailing_nan_is_rejected(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    with pytest.raises(InvalidDataError, match="horizon_h"):
        _load(build_config(f, l, horizon_h=5))


def test_without_labels_dataset_is_unsupervised(make_frames, build_config) -> None:
    f, _ = make_frames(n=100)
    ds = _load(build_config(f, None))
    assert ds.y is None and ds.labels_aux == {}
    assert ds.task_type is TaskType.UNSUPERVISED
    assert ds.X.shape == (100, 3)


def test_multiclass_selected_label(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    l = l.with_columns((pl.int_range(pl.len()) % 3).cast(pl.Float64).alias("regime3"))
    ds = _load(build_config(f, l, selected="regime3"))
    assert ds.task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert ds.classes_ is not None and ds.classes_.tolist() == [0.0, 1.0, 2.0]


def test_report_describes_the_load(make_frames, build_config) -> None:
    f, l = make_frames(n=100)
    ds = _load(build_config(f, l))
    r = ds.report
    assert r.n_features == 3 and r.horizon_h == 21
    assert r.t_min.startswith("2024-01-01T00:00:00")
    assert r.t_max.startswith("2024-01-04T22:00:00")  # row 94 -> +94h
    assert r.n_unmatched_dropped == 0
    assert ds.report.to_dict()["n_joined_rows"] == 100
