"""Inputs to ``skore.evaluate``.

Strictly limited to objects consumed by the experiment scripts:

- ``splitter`` — a custom walk-forward cross-validator that reads
  its month label directly from ``X[time_col]`` (default
  ``"Date"``), so it works in any caller that hands it the panel
  (``skore.evaluate``, ``skrub`` CV machinery, plain sklearn).

This module does not call ``skore.evaluate``, open a Project, or
persist anything.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class WalkForwardMonthlySplit(BaseCrossValidator):
    """Walk-forward cross-validator stepping by month.

    The first training fold contains the earliest ``min_train_months``
    distinct months. Each test fold is ``interval_months`` wide and
    successive folds advance by ``interval_months`` (no overlap, no
    gap). The per-row month label is read from ``X[time_col]``;
    callers may override by passing ``groups`` to :meth:`split`.

    Parameters
    ----------
    min_train_months : int, default=6
        Number of distinct months in the very first training fold.
        Must be >= 6.
    interval_months : int, default=1
        Width of each test fold in distinct months and step between
        successive folds. Must be >= 1.
    time_col : str, default="Date"
        Name of the column in ``X`` that carries the month label
        (one entry per row). Used when ``groups`` is not supplied.
    """

    def __init__(
        self,
        min_train_months: int = 6,
        interval_months: int = 1,
        time_col: str = "Date",
    ) -> None:
        if min_train_months < 6:
            raise ValueError(f"min_train_months must be >= 6, got {min_train_months}")
        if interval_months < 1:
            raise ValueError(f"interval_months must be >= 1, got {interval_months}")
        self.min_train_months = min_train_months
        self.interval_months = interval_months
        self.time_col = time_col

    def _resolve_groups(self, X, groups):
        if groups is not None:
            return np.asarray(groups)
        if X is None:
            raise ValueError(
                "Either `groups` or `X` (containing the time column) must be provided."
            )
        return np.asarray(X[self.time_col])

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield walk-forward ``(train_idx, test_idx)`` pairs.

        Parameters
        ----------
        X : array-like
            Feature matrix; only its row count is used.
        y : array-like, optional
            Target; ignored.
        groups : array-like of shape (n_samples,)
            Per-row month label (typically a ``Date`` column).

        Yields
        ------
        train_idx, test_idx : numpy.ndarray of int
            Row positions for the training and test folds.
        """
        groups_arr = self._resolve_groups(X, groups)
        unique_months = np.sort(np.unique(groups_arr))
        n_months = len(unique_months)
        train_end = self.min_train_months
        while train_end + self.interval_months <= n_months:
            train_months = unique_months[:train_end]
            test_months = unique_months[train_end : train_end + self.interval_months]
            train_idx = np.where(np.isin(groups_arr, train_months))[0]
            test_idx = np.where(np.isin(groups_arr, test_months))[0]
            yield train_idx, test_idx
            train_end += self.interval_months

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of walk-forward folds.

        Parameters
        ----------
        X, y : array-like, optional
            Ignored.
        groups : array-like of shape (n_samples,), optional
            Per-row month label; required to count distinct months.
            When ``None`` (e.g., during ``skore.evaluate``'s early
            single-split probe before skrub resolves ``split_kwargs``),
            a placeholder ``> 1`` is returned to skip the fast-path —
            the real count is exposed once ``groups`` is wired in.

        Returns
        -------
        int
            Number of ``(train, test)`` pairs :meth:`split` will yield.
        """
        if groups is None and X is None:
            return 2
        groups_arr = self._resolve_groups(X, groups)
        n_months = len(np.unique(groups_arr))
        if n_months <= self.min_train_months:
            return 0
        return (n_months - self.min_train_months) // self.interval_months


splitter = WalkForwardMonthlySplit(min_train_months=12, interval_months=3)
