"""Data loading for the Beeristan demand-forecasting panel.

Two entry points:

- :func:`load_panel` — reads the seven training CSVs from a
  directory, parses ``YearMonth`` into a proper ``Date`` column,
  and left-joins everything onto the historical-volume table
  keyed by ``(Agency, SKU, Date)``. Consumed inside the skrub
  DataOps graph via
  ``skrub.var("data_dir").skb.apply_func(load_panel)``.
- :func:`load_cold_start_grid` — builds an in-memory
  ``historical_volume.csv``-shaped frame for the cross-product
  ``agencies × skus × year_months``, with ``Volume`` set to null.
  Used to construct a temporary directory the trained learner
  can predict over for cold-start agencies (see
  ``experiments/04_sku_recommendation.py``).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl


def _parse_yearmonth(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("YearMonth").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m").alias("Date")
    ).drop("YearMonth")


def _read_psp_zip(path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as z:
        inner = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(inner) as f:
            return pl.read_csv(f.read())


def load_panel(data_dir: str | Path) -> pl.DataFrame:
    """Load and join the seven training CSVs into a single panel.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory holding ``historical_volume.csv``,
        ``price_sales_promotion.csv.zip``, ``weather.csv``,
        ``event_calendar.csv``, ``industry_volume.csv``,
        ``industry_soda_sales.csv`` and ``demographics.csv``.

    Returns
    -------
    polars.DataFrame
        One row per ``(Agency, SKU, Date)``, sorted by ``(Date, Agency,
        SKU)``. Carries the target ``Volume`` plus every joined feature.
    """
    data_dir = Path(data_dir)

    historical = _parse_yearmonth(pl.read_csv(data_dir / "historical_volume.csv"))
    psp = _parse_yearmonth(_read_psp_zip(data_dir / "price_sales_promotion.csv.zip"))
    weather = _parse_yearmonth(pl.read_csv(data_dir / "weather.csv"))
    industry_vol = _parse_yearmonth(pl.read_csv(data_dir / "industry_volume.csv"))
    industry_soda = _parse_yearmonth(pl.read_csv(data_dir / "industry_soda_sales.csv"))
    demographics = pl.read_csv(data_dir / "demographics.csv")

    events = pl.read_csv(data_dir / "event_calendar.csv")
    events = events.rename({c: c.strip() for c in events.columns})
    events = _parse_yearmonth(events)

    n_target_rows = historical.height
    panel = (
        historical.join(psp, on=["Agency", "SKU", "Date"], how="left")
        .join(weather, on=["Agency", "Date"], how="left")
        .join(events, on=["Date"], how="left")
        .join(industry_vol, on=["Date"], how="left")
        .join(industry_soda, on=["Date"], how="left")
        .join(demographics, on=["Agency"], how="left")
        .sort(["Date", "Agency", "SKU"])
    )

    if panel.height != n_target_rows:
        raise RuntimeError(
            f"Join cardinality mismatch: historical_volume has {n_target_rows} rows, "
            f"panel has {panel.height} rows. A join key is wrong."
        )
    return panel


def load_cold_start_grid(
    agencies: list[str],
    skus: list[str],
    year_months: list[int],
) -> pl.DataFrame:
    """Build the cross-product of agencies x skus x year_months.

    Returned frame has the same schema as ``historical_volume.csv``
    (``Agency``, ``SKU``, ``YearMonth`` as ``Int64``,
    ``Volume`` as ``Float64`` set to null). Used to write a
    temporary ``historical_volume.csv`` that :func:`load_panel`
    can read at predict time for cold-start agencies — the
    same six side tables in the train directory then get
    left-joined onto these rows by ``learner.predict``.

    Parameters
    ----------
    agencies : list of str
        Agency identifiers, e.g. ``["Agency_06", "Agency_14"]``.
    skus : list of str
        Candidate SKU identifiers, e.g. all SKUs seen in train.
    year_months : list of int
        Year-month integers in ``YYYYMM`` form (e.g. ``201701``).

    Returns
    -------
    polars.DataFrame
        ``len(agencies) * len(skus) * len(year_months)`` rows.
    """
    rows = [
        {"Agency": a, "SKU": s, "YearMonth": ym, "Volume": None}
        for a in agencies
        for s in skus
        for ym in year_months
    ]
    return pl.DataFrame(
        rows,
        schema={
            "Agency": pl.Utf8,
            "SKU": pl.Utf8,
            "YearMonth": pl.Int64,
            "Volume": pl.Float64,
        },
    )
