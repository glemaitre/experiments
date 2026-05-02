"""Feature functions for the Beeristan demand-forecasting panel.

Currently exposes :func:`add_lag_features`, a stateless transform of
the joined panel that adds lag, trailing-rolling-mean, and
year-over-year features over ``Volume`` at the
``(Agency, SKU)`` level. Attached to the skrub DataOps graph via
``.skb.apply_func`` between the loader and the X marker.
"""

from __future__ import annotations

import polars as pl


def add_lag_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Add within-series lag / rolling / year-over-year columns.

    Sorts by ``(Agency, SKU, Date)`` to compute the features safely
    inside each series, then restores the original ``(Date, Agency,
    SKU)`` ordering. All rolling means are *trailing*: the row's own
    ``Volume`` is shifted out before the window aggregates.

    Parameters
    ----------
    panel : polars.DataFrame
        The joined panel produced by :func:`beeristan.data.load_panel`.
        Must carry ``Agency``, ``SKU``, ``Date``, ``Volume``.

    Returns
    -------
    polars.DataFrame
        The input panel with five additional columns:
        ``Volume_lag_1``, ``Volume_lag_12``,
        ``Volume_rolling_mean_3``, ``_6``, ``_12``.
    """
    sorted_panel = panel.sort(["Agency", "SKU", "Date"])

    sorted_panel = sorted_panel.with_columns(
        pl.col("Volume").shift(1).over(["Agency", "SKU"]).alias("Volume_lag_1"),
        pl.col("Volume").shift(12).over(["Agency", "SKU"]).alias("Volume_lag_12"),
        pl.col("Volume")
        .shift(1)
        .rolling_mean(window_size=3)
        .over(["Agency", "SKU"])
        .alias("Volume_rolling_mean_3"),
        pl.col("Volume")
        .shift(1)
        .rolling_mean(window_size=6)
        .over(["Agency", "SKU"])
        .alias("Volume_rolling_mean_6"),
        pl.col("Volume")
        .shift(1)
        .rolling_mean(window_size=12)
        .over(["Agency", "SKU"])
        .alias("Volume_rolling_mean_12"),
    )

    return sorted_panel.sort(["Date", "Agency", "SKU"])
