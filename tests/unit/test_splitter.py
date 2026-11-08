import dataclasses
import inspect

import numpy as np
import pytest

from quantautoml.common.exceptions import DataLeakageError, InvalidDataError
from quantautoml.common.types import Dataset, IngestionReport, Segment, SplitMode, TaskType
from quantautoml.data.splitter import AdaptiveTimeSeriesSplitter, verify_split


def make_dataset(n: int, with_labels: bool = True, y: np.ndarray | None = None) -> Dataset:
    idx = np.arange(n)
    report = IngestionReport(
        n_feature_rows=n, n_label_rows=n, n_joined_rows=n, n_unmatched_dropped=0,
        n_trailing_label_nan_dropped=0, source_timezone="UTC", naive_assumed_utc=False,
        n_features=2, n_nan=0, n_inf=0, t_min="", t_max="", label_nan_counts={},
        max_trailing_nan=0, horizon_h=1,
    )
    if with_labels and y is None:
        y = (idx % 2).astype(np.int32)
    return Dataset(
        X=np.stack([idx, -idx], axis=1).astype(np.float32),
        y=y if with_labels else None,
        labels_aux={"fwd_ret_21": idx.astype(np.float32)},
        timestamps=(np.datetime64("2024-01-01T00:00:00", "ns") + idx.astype("timedelta64[h]")),
        feature_names=("a", "b"),
        selected_label="y" if with_labels else None,
        report=report,
        task_type=TaskType.BINARY_CLASSIFICATION if with_labels else TaskType.UNSUPERVISED,
        classes_=np.array([0, 1]) if with_labels else None,
    )


SPLITTER = AdaptiveTimeSeriesSplitter()


@pytest.mark.parametrize(
    ("n", "mode"),
    [(500, SplitMode.OOF_ASSISTED), (9_999, SplitMode.OOF_ASSISTED),
     (10_000, SplitMode.STATIC_5WAY), (12_345, SplitMode.STATIC_5WAY)],
)
def test_mode_is_chosen_from_row_count(n: int, mode: SplitMode) -> None:
    assert SPLITTER.split(make_dataset(n), horizon_h=5).mode is mode


def test_too_few_rows_is_rejected() -> None:
    with pytest.raises(InvalidDataError):
        SPLITTER.split(make_dataset(499), horizon_h=5)


def test_static_split_boundaries_with_purge() -> None:
    res = SPLITTER.split(make_dataset(10_000), horizon_h=5)
    got = {
        name: (seg.index[0], seg.index[-1] + 1)
        for name, seg in [("train", res.train), ("cal_a", res.cal_a), ("cal_b", res.cal_b),
                          ("val", res.val), ("oos", res.oos)]
    }
    assert got == {
        "train": (0, 6000), "cal_a": (6005, 7000), "cal_b": (7005, 8000),
        "val": (8005, 9000), "oos": (9005, 10000),
    }
    assert res.oof_folds is None


def test_oof_split_boundaries_and_folds() -> None:
    res = SPLITTER.split(make_dataset(1_000), horizon_h=5)
    assert (res.train.index[0], res.train.index[-1] + 1) == (0, 800)
    assert (res.val.index[0], res.val.index[-1] + 1) == (805, 900)
    assert (res.oos.index[0], res.oos.index[-1] + 1) == (905, 1000)
    assert res.cal_a is None and res.cal_b is None
    assert res.oof_folds is not None and len(res.oof_folds) == 5
    for tr, va in res.oof_folds:
        assert tr[0] == 0 and np.array_equal(tr, np.arange(tr[0], tr[-1] + 1))
        assert va[0] - tr[-1] - 1 == 5
        assert va[-1] <= res.train.index[-1]


@pytest.mark.parametrize("n", [1_000, 12_000])
@pytest.mark.parametrize("h", [1, 5, 21])
def test_gap_between_every_adjacent_segment_equals_horizon(n: int, h: int) -> None:
    res = SPLITTER.split(make_dataset(n), horizon_h=h)
    parts: list[Segment] = [res.train, res.cal_a, res.cal_b, res.val, res.oos]
    parts = [p for p in parts if p is not None]
    for a, b in zip(parts, parts[1:]):
        assert b.index[0] - a.index[-1] - 1 == h
        assert a.timestamps[-1] < b.timestamps[0]
    assert all(np.all(np.diff(p.index) == 1) for p in parts)


def test_segments_carry_matching_slices_of_every_array() -> None:
    ds = make_dataset(1_000)
    res = SPLITTER.split(ds, horizon_h=5)
    for seg in (res.train, res.val, res.oos):
        np.testing.assert_array_equal(seg.X, ds.X[seg.index])
        np.testing.assert_array_equal(seg.y, ds.y[seg.index])
        np.testing.assert_array_equal(seg.timestamps, ds.timestamps[seg.index])
        np.testing.assert_array_equal(seg.labels_aux["fwd_ret_21"], ds.labels_aux["fwd_ret_21"][seg.index])
        assert seg.X.dtype == np.float32


def test_horizon_too_large_for_oof_blocks_is_rejected() -> None:
    with pytest.raises(InvalidDataError):
        SPLITTER.split(make_dataset(500), horizon_h=100)


def test_horizon_too_large_for_static_segments_is_rejected() -> None:
    with pytest.raises(InvalidDataError):
        SPLITTER.split(make_dataset(10_000), horizon_h=1_500)


@pytest.mark.parametrize("h", [0, -1])
def test_non_positive_horizon_is_rejected(h: int) -> None:
    with pytest.raises(InvalidDataError):
        SPLITTER.split(make_dataset(1_000), horizon_h=h)


def test_split_api_has_no_shuffle_parameter() -> None:
    params = inspect.signature(AdaptiveTimeSeriesSplitter().split).parameters
    assert set(params) == {"dataset", "horizon_h"}


def test_report_describes_the_split() -> None:
    res = SPLITTER.split(make_dataset(10_000), horizon_h=5)
    r = res.report
    assert r.mode is SplitMode.STATIC_5WAY and r.n_rows == 10_000 and r.horizon_h == 5
    assert [s.name for s in r.segments] == ["train", "cal_a", "cal_b", "val", "oos"]
    assert [s.rows_dropped_before for s in r.segments] == [0, 5, 5, 5, 5]
    assert r.segments[0].n_rows == 6000 and r.segments[1].n_rows == 995
    assert r.segments[0].class_counts == {"0": 3000, "1": 3000}
    assert r.oof_fold_sizes is None and r.warnings == ()


def test_oof_report_lists_fold_sizes() -> None:
    res = SPLITTER.split(make_dataset(1_000), horizon_h=5)
    assert res.report.oof_fold_sizes is not None and len(res.report.oof_fold_sizes) == 5
    assert [s.name for s in res.report.segments] == ["train", "val", "oos"]


def test_missing_class_in_a_segment_raises_a_warning() -> None:
    y = (np.arange(1_000) % 2).astype(np.int32)
    y[900:] = 0
    res = SPLITTER.split(make_dataset(1_000, y=y), horizon_h=5)
    assert any("oos" in w and "1" in w for w in res.report.warnings)


def test_unsupervised_dataset_has_no_labels_in_segments() -> None:
    res = SPLITTER.split(make_dataset(1_000, with_labels=False), horizon_h=5)
    assert res.train.y is None
    assert res.report.segments[0].class_counts is None


def test_verify_split_accepts_a_valid_split() -> None:
    verify_split(SPLITTER.split(make_dataset(10_000), horizon_h=5), horizon_h=5)


def test_verify_split_rejects_missing_gap() -> None:
    res = SPLITTER.split(make_dataset(10_000), horizon_h=5)
    ds = make_dataset(10_000)
    idx = np.arange(res.train.index[-1] + 1, res.cal_a.index[-1] + 1)  # starts at the boundary: gap 0
    leaky = Segment(ds.X[idx], ds.y[idx], {}, ds.timestamps[idx], idx)
    with pytest.raises(DataLeakageError):
        verify_split(dataclasses.replace(res, cal_a=leaky), horizon_h=5)


def test_verify_split_rejects_overlapping_segments() -> None:
    res = SPLITTER.split(make_dataset(10_000), horizon_h=5)
    with pytest.raises(DataLeakageError):
        verify_split(dataclasses.replace(res, val=res.train), horizon_h=5)


def test_verify_split_rejects_oof_fold_without_gap() -> None:
    res = SPLITTER.split(make_dataset(1_000), horizon_h=5)
    tr, va = res.oof_folds[0]
    bad = [(tr, np.arange(tr[-1] + 1, va[-1] + 1))] + res.oof_folds[1:]
    with pytest.raises(DataLeakageError):
        verify_split(dataclasses.replace(res, oof_folds=bad), horizon_h=5)


def test_verify_split_rejects_oof_fold_leaking_beyond_train() -> None:
    res = SPLITTER.split(make_dataset(1_000), horizon_h=5)
    tr, va = res.oof_folds[0]
    bad = [(tr, np.arange(va[0], res.val.index[-1] + 1))] + res.oof_folds[1:]
    with pytest.raises(DataLeakageError):
        verify_split(dataclasses.replace(res, oof_folds=bad), horizon_h=5)
