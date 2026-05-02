"""Inputs to ``skore.evaluate``: the cross-validator for the baseline experiment.

Holds the custom datetime-anchored expanding walk-forward splitter and the
module-level ``splitter`` instance consumed by experiment scripts via
``skore.evaluate(..., splitter=splitter)``.

Per the contract owned by ``evaluate-ml-pipeline``, this module:

- defines only the cross-validator (and any explicit metric overrides — none
  here; we trust skore's regression defaults);
- does NOT call ``skore.evaluate``, open a ``skore.Project``, or persist
  anything — those steps belong in the experiment script.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class DatetimeAnchoredWalkForward:
    """Expanding walk-forward splitter anchored to a datetime column in X.

    Each fold:

    - train: all rows with ``X[time_col] < cutoff_i``
    - test:  rows with ``cutoff_i <= X[time_col] < cutoff_i + test_size``

    where ``cutoff_i = times.min() + initial_train_days + i * step_days``.

    Folds are produced while the test block fits inside the data range.
    With ``keep_partial_tail=True`` a trailing partial test block (shorter
    than ``test_days``) is also yielded.

    Parameters
    ----------
    time_col : str, default "datetime"
        Name of the datetime column in X. The column must be convertible to
        ``numpy.datetime64`` via ``to_numpy()``.
    initial_train_days : int, default 365
        Length of the initial training window in days, anchored at the
        earliest timestamp in X.
    test_days : int, default 183
        Length of each test block in days (≈ 6 months).
    step_days : int, default 183
        Step between consecutive cutoffs in days. Equal to ``test_days`` →
        non-overlapping test blocks.
    keep_partial_tail : bool, default False
        Whether to yield the trailing partial test block when the data ends
        before the next full test block.
    """

    def __init__(
        self,
        time_col: str = "datetime",
        initial_train_days: int = 365,
        test_days: int = 183,
        step_days: int = 183,
        keep_partial_tail: bool = False,
    ):
        self.time_col = time_col
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.step_days = step_days
        self.keep_partial_tail = keep_partial_tail

    def _times(self, X) -> np.ndarray:
        try:
            return X[self.time_col].to_numpy()
        except (KeyError, AttributeError) as e:
            raise ValueError(
                f"{type(self).__name__} expects X to be a DataFrame-like with "
                f"column {self.time_col!r}. Got X of type {type(X).__name__}."
            ) from e

    def _bounds(
        self, times: np.ndarray
    ) -> list[tuple[np.datetime64, np.datetime64]]:
        one_day = np.timedelta64(1, "D")
        initial = self.initial_train_days * one_day
        test = self.test_days * one_day
        step = self.step_days * one_day

        start = times.min()
        end = times.max()
        cutoff = start + initial
        out: list[tuple[np.datetime64, np.datetime64]] = []
        while cutoff < end:
            test_end = cutoff + test
            is_partial = test_end > end
            if is_partial and not self.keep_partial_tail:
                break
            out.append((cutoff, min(test_end, end)))
            cutoff = cutoff + step
        return out

    def split(
        self, X, y=None, groups=None, **kwargs
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        times = self._times(X)
        for cutoff, test_end in self._bounds(times):
            train_mask = times < cutoff
            test_mask = (times >= cutoff) & (times < test_end)
            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        # Fold count depends on the actual datetime range in X. When X is
        # not yet materialized (skore probes for ``== 1`` before resolving
        # the SkrubLearner's env-dict), return 0 so that probe doesn't
        # short-circuit into the single-split branch. The true count is
        # computed once X is available.
        if X is None:
            return 0
        return len(self._bounds(self._times(X)))


# Module-level splitter consumed by ``skore.evaluate(..., splitter=splitter)``
# in the experiment scripts. Defaults match the approved baseline plan:
# 1-year initial training window, half-yearly test blocks, expanding window,
# no partial trailing block.
splitter = DatetimeAnchoredWalkForward()
