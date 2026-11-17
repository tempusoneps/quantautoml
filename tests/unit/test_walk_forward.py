import numpy as np
import pytest

from quantautoml.common.exceptions import InvalidDataError
from quantautoml.models.validation import ExpandingWalkForwardSplitter


def test_walk_forward_splitter_splits() -> None:
    n_samples = 1000
    splitter = ExpandingWalkForwardSplitter(n_folds=5, min_train_ratio=0.5, embargo_bars=10)
    splits = list(splitter.split(n_samples))

    assert len(splits) == 5

    prev_train_len = 0
    prev_test_max = -1

    for i, (train_idx, test_idx) in enumerate(splits):
        assert train_idx.dtype == np.int32
        assert test_idx.dtype == np.int32
        assert len(train_idx) > 0
        assert len(test_idx) > 0

        # Monotonicity inside folds
        assert np.all(np.diff(train_idx) == 1)
        assert np.all(np.diff(test_idx) == 1)

        # Expanding window: Train strictly expands
        assert len(train_idx) > prev_train_len
        prev_train_len = len(train_idx)

        # Embargo respected: max(train) + embargo < min(test)
        assert train_idx[-1] + 10 <= test_idx[0]

        # No overlap between consecutive test sets
        assert test_idx[0] > prev_test_max
        prev_test_max = test_idx[-1]

    # Last test fold reaches the end of the samples
    assert splits[-1][1][-1] == n_samples - 1


def test_walk_forward_splitter_validation() -> None:
    with pytest.raises(InvalidDataError):
        ExpandingWalkForwardSplitter(n_folds=1)

    with pytest.raises(InvalidDataError):
        ExpandingWalkForwardSplitter(min_train_ratio=0.0)

    with pytest.raises(InvalidDataError):
        ExpandingWalkForwardSplitter(min_train_ratio=1.0)

    with pytest.raises(InvalidDataError):
        ExpandingWalkForwardSplitter(embargo_bars=-1)


def test_walk_forward_splitter_too_few_samples() -> None:
    splitter = ExpandingWalkForwardSplitter(n_folds=5, min_train_ratio=0.5, embargo_bars=20)
    with pytest.raises(InvalidDataError):
        list(splitter.split(30))
