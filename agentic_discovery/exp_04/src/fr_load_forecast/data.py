"""Data loading and X-marker wiring.

Owns: how raw CSV / parquet files in ``data/`` are materialized into a
single polars frame ready for the pipeline. The datetime column survives
in the returned frame so that ``pipeline.py`` can wire it as
``split_kwargs={"times": ...}`` at the X marker.

Pipeline mechanics live in `build-ml-pipeline`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .features import (
    add_calendar_features,
    add_load_lags,
    add_target,
    aggregate_weather_across_cities,
    shift_future_weather,
)

LOAD_TIME_COL = "Time (UTC)"
LOAD_ACTUAL_COL = "Actual Total Load [MW] - BZN|FR"


def _load_load_csvs(data_dir: Path) -> pl.DataFrame:
    """Concat the yearly ENTSO-E load CSVs into a clean ``[datetime, load]`` frame.

    The raw ``Time (UTC)`` column carries a range string like
    ``"01.01.2021 00:00 - 01.01.2021 01:00"`` — we keep the start of the
    range as the prediction-time index, in UTC. The raw load column comes
    in as a string (the export occasionally contains commas / ``n/e``);
    invalid rows are dropped after coercion.
    """
    csv_files = sorted(data_dir.glob("Total Load*.csv"))
    raw = pl.concat(
        [
            pl.read_csv(
                f,
                null_values=["N/A", "n/e", "-"],
                schema_overrides={LOAD_ACTUAL_COL: pl.String},
            )
            for f in csv_files
        ],
        how="vertical_relaxed",
    )
    return (
        raw.with_columns(
            pl.col(LOAD_TIME_COL)
            .str.split(" - ")
            .list.get(0)
            .str.strptime(pl.Datetime("us", "UTC"), "%d.%m.%Y %H:%M")
            .alias("datetime"),
            pl.col(LOAD_ACTUAL_COL)
            .cast(pl.String, strict=False)
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
            .alias("load"),
        )
        .select(["datetime", "load"])
        .drop_nulls("load")
        .unique(subset="datetime")
        .sort("datetime")
    )


def _load_weather_parquets(data_dir: Path) -> pl.DataFrame:
    """Read all ``weather_<city>.parquet`` files and average across cities."""
    parquet_files = sorted(data_dir.glob("weather_*.parquet"))
    per_city = [pl.read_parquet(f) for f in parquet_files]
    weather = aggregate_weather_across_cities(per_city)
    return weather.with_columns(
        pl.col("datetime").cast(pl.Datetime("us", "UTC"))
    )


def load_dataset(data_dir: str | Path = "data") -> pl.DataFrame:
    """Build the full dataset for h=24 French electricity load forecasting.

    Returned columns:

    - ``datetime``: prediction time t (Datetime UTC). Used by the splitter
      via ``split_kwargs={"times": ...}``; should be dropped from the
      feature matrix before the predictor.
    - ``load_t``: load(t) — current load (past covariate / feature).
    - ``load_lag_<h>h``: load(t − h) for h ∈ {1, 24, 168} (past covariates).
    - 6 weather columns (temperature, precipitation, wind, cloud, soil
      moisture, humidity) — values at t + 24h, mean across the 10 cities
      (future covariates).
    - ``cal_hour`` / ``cal_dow`` / ``cal_month`` / ``cal_is_holiday``:
      calendar features at t + 24h, computed in Europe/Paris local time.
    - ``target``: load(t + 24h).

    Rows with any null (lag warmup at the head, target lookahead at the
    tail, weather coverage gaps) are dropped.
    """
    data_dir = Path(data_dir)
    load = _load_load_csvs(data_dir)
    weather = _load_weather_parquets(data_dir)

    frame = load.join(weather, on="datetime", how="inner").sort("datetime")
    frame = add_load_lags(frame)
    frame = add_target(frame)
    frame = shift_future_weather(frame)
    frame = add_calendar_features(frame)

    return frame.drop_nulls().rename({"load": "load_t"})
