"""Feature functions for the Beeristan demand-forecasting panel.

Exposes two stateless transforms attached to the skrub DataOps graph
via ``.skb.apply_func`` between the loader and the X marker:

- :func:`add_lag_features` — lag / trailing-rolling-mean /
  year-over-year columns on ``Volume`` at the ``(Agency, SKU)``
  level.
- :func:`add_side_table_lag_features` — lag-1 and trailing
  rolling-mean-3 on six side-table columns
  (``Price``, ``Sales``, ``Promotions``, ``Avg_Max_Temp``,
  ``Industry_Volume``, ``Soda_Volume``).
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


_SIDE_TABLE_COLS: tuple[str, ...] = (
    "Price",
    "Sales",
    "Promotions",
    "Avg_Max_Temp",
    "Industry_Volume",
    "Soda_Volume",
)


def add_side_table_lag_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Add lag-1 and trailing rolling-mean-3 on six side-table columns.

    Within each ``(Agency, SKU)`` partition (sorted by ``Date``),
    builds two new columns per source column: ``<col>_lag_1`` and
    ``<col>_rolling_mean_3``. The rolling mean is trailing
    (``shift(1).rolling_mean(3)``), so the row's own value is shifted
    out before the window aggregates.

    Partitioning by ``(Agency, SKU)`` is correct for all six columns,
    even when the source value does not vary by SKU within a month
    (``Avg_Max_Temp``, ``Industry_Volume``, ``Soda_Volume``): inside
    any single ``(Agency, SKU)`` slice the values still form a
    monthly time series, and shifting within the slice yields the
    same lagged value as partitioning at a coarser level.

    Parameters
    ----------
    panel : polars.DataFrame
        The panel after :func:`add_lag_features` has run. Must carry
        ``Agency``, ``SKU``, ``Date`` and the six source columns.

    Returns
    -------
    polars.DataFrame
        The input panel with twelve additional columns
        (``<col>_lag_1`` and ``<col>_rolling_mean_3`` for each
        source column).
    """
    sorted_panel = panel.sort(["Agency", "SKU", "Date"])

    new_cols = []
    for col in _SIDE_TABLE_COLS:
        new_cols.append(
            pl.col(col).shift(1).over(["Agency", "SKU"]).alias(f"{col}_lag_1")
        )
        new_cols.append(
            pl.col(col)
            .shift(1)
            .rolling_mean(window_size=3)
            .over(["Agency", "SKU"])
            .alias(f"{col}_rolling_mean_3")
        )

    sorted_panel = sorted_panel.with_columns(new_cols)
    return sorted_panel.sort(["Date", "Agency", "SKU"])
