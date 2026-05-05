"""Inputs to ``skore.evaluate`` for the t+12 baseline.

Exposes:

- ``CalendarMonthSplit`` — walk-forward CV with a one-calendar-month test
  window per fold; expanding train; embargo equal to the forecast horizon
  to prevent training-row targets from leaking into the test month.
- ``splitter`` — the configured instance fed into both
  ``mark_as_X(cv=...)`` (in ``pipeline.py``) and
  ``skore.evaluate(..., splitter=...)`` (in the experiment script).

This module does not call ``skore.evaluate``, does not open a
``skore.Project``, and does not persist anything.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class CalendarMonthSplit:
    """Walk-forward CV with a one-calendar-month test window per fold.

    For each eligible calendar month ``M``:

    - ``test`` = rows whose timestamp falls inside ``M``.
    - ``train`` = rows whose timestamp is strictly before
      ``start_of_M - embargo_hours``. The embargo prevents a training row
      at time ``s`` (whose target is ``load[s + horizon_hours]``) from
      leaking its target into the test month.

    Folds are produced for every month after a ``min_train_months``
    warm-up, in chronological order.

    Parameters
    ----------
    embargo_hours : int, default=12
        Gap between train end and test start, in hours. Set to the
        forecast horizon used in the supervised frame.
    min_train_months : int, default=12
        Number of leading calendar months to skip before the first test
        fold, so every fold has at least that much training history.
    """

    def __init__(self, embargo_hours: int = 12, min_train_months: int = 12):
        self.embargo_hours = embargo_hours
        self.min_train_months = min_train_months

    @staticmethod
    def _to_ns(timestamps) -> np.ndarray:
        if hasattr(timestamps, "to_numpy"):
            arr = timestamps.to_numpy()
        else:
            arr = np.asarray(timestamps)
        return arr.astype("datetime64[ns]")

    def _eligible_months(self, ts_ns: np.ndarray) -> np.ndarray:
        months = np.unique(ts_ns.astype("datetime64[M]"))
        if len(months) <= self.min_train_months:
            return months[:0]
        return months[self.min_train_months :]

    def split(
        self, X=None, y=None, timestamps=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` per eligible calendar month."""
        if timestamps is None and X is not None:
            try:
                timestamps = X["prediction_time"]
            except (KeyError, TypeError):
                timestamps = None
        if timestamps is None:
            raise ValueError(
                "CalendarMonthSplit needs timestamps. Pass them via "
                "split_kwargs={'timestamps': ...} or include a "
                "'prediction_time' column in X."
            )
        ts_ns = self._to_ns(timestamps)
        positions = np.arange(len(ts_ns))
        embargo = np.timedelta64(self.embargo_hours, "h")
        for month in self._eligible_months(ts_ns):
            month_start = np.datetime64(month, "ns")
            next_month_start = np.datetime64(month + np.timedelta64(1, "M"), "ns")
            test_mask = (ts_ns >= month_start) & (ts_ns < next_month_start)
            train_mask = ts_ns < (month_start - embargo)
            train_idx = positions[train_mask]
            test_idx = positions[test_mask]
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, timestamps=None) -> int:
        """Return the number of eligible monthly folds."""
        if timestamps is None and X is not None:
            try:
                timestamps = X["prediction_time"]
            except (KeyError, TypeError):
                timestamps = None
        if timestamps is None:
            return 2
        return sum(1 for _ in self.split(X=X, y=y, timestamps=timestamps))


splitter = CalendarMonthSplit(embargo_hours=12, min_train_months=12)
