"""Data loading for the Beeristan demand-forecasting panel.

Reads the seven training CSVs from a directory, parses ``YearMonth``
into a proper ``Date`` column, and left-joins everything onto the
historical-volume table keyed by ``(Agency, SKU, Date)``. The loader
is consumed inside the skrub DataOps graph via
``skrub.var("data_dir").skb.apply_func(load_panel)``.
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
