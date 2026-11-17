"""Walk-forward time-series validation with expanding window and embargo."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from quantautoml.common.exceptions import InvalidDataError


class ExpandingWalkForwardSplitter:
    """Expanding-window walk-forward splitter with embargo for temporal data.

    The dataset is partitioned into an initial training segment and test folds.
    With each fold, the training window expands up to the start of that fold's test set
    minus `embargo_bars`.
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_train_ratio: float = 0.5,
        embargo_bars: int = 0,
    ) -> None:
        if n_folds < 2:
            raise InvalidDataError(f"n_folds must be >= 2, got {n_folds}")
        if min_train_ratio <= 0.0 or min_train_ratio >= 1.0:
            raise InvalidDataError(
                f"min_train_ratio must be in (0, 1), got {min_train_ratio}"
            )
        if embargo_bars < 0:
            raise InvalidDataError(f"embargo_bars must be >= 0, got {embargo_bars}")

        self.n_folds = n_folds
        self.min_train_ratio = min_train_ratio
        self.embargo_bars = embargo_bars

    def split(
        self,
        n_samples: int,
    ) -> Iterator[tuple[NDArray[np.int32], NDArray[np.int32]]]:
        """Yield (train_indices, test_indices) for each expanding fold.

        Both index arrays are strictly increasing np.int32.
        """
        init_train_len = int(n_samples * self.min_train_ratio)
        if init_train_len <= self.embargo_bars:
            raise InvalidDataError(
                f"n_samples={n_samples} too small for min_train_ratio={self.min_train_ratio} "
                f"with embargo_bars={self.embargo_bars}"
            )

        n_test_total = n_samples - init_train_len
        if n_test_total < self.n_folds:
            raise InvalidDataError(
                f"Remaining test samples ({n_test_total}) smaller than n_folds ({self.n_folds})"
            )

        for k in range(self.n_folds):
            test_start = init_train_len + (k * n_test_total) // self.n_folds
            test_end = (
                init_train_len + ((k + 1) * n_test_total) // self.n_folds
                if k < self.n_folds - 1
                else n_samples
            )
            train_end = test_start - self.embargo_bars

            if train_end <= 0:
                raise InvalidDataError(
                    f"Train fold {k} has non-positive sample count (train_end={train_end})"
                )
            if test_end <= test_start:
                raise InvalidDataError(
                    f"Test fold {k} has non-positive sample count (start={test_start}, end={test_end})"
                )

            train_idx = np.arange(0, train_end, dtype=np.int32)
            test_idx = np.arange(test_start, test_end, dtype=np.int32)
            yield train_idx, test_idx
