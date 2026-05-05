"""Data loading and supervised-frame assembly for the t+12 forecasting task.

The pipeline binds a `data_dir` source identifier; this module's
``build_supervised_frame`` is attached as the first ``.skb.apply_func`` in
``pipeline.py``. Returns a single polars DataFrame whose rows are
prediction-time samples.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

HORIZON_HOURS = 12

WEATHER_CITIES = (
    "bayonne",
    "brest",
    "lille",
    "limoges",
    "lyon",
    "marseille",
    "nantes",
    "paris",
    "strasbourg",
    "toulouse",
)

WEATHER_COLUMNS = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "soil_moisture_1_to_3cm",
    "relative_humidity_2m",
)


def _parse_load_csv(path: Path) -> pl.DataFrame:
    """Read one ENTSO-E load CSV; return [time, actual_load]."""
    df = pl.read_csv(path, null_values=["", "-", "N/A", "n/e"])
    df = df.rename(
        {
            "Time (UTC)": "interval",
            "Actual Total Load [MW] - BZN|FR": "actual_load",
        }
    )
    df = df.with_columns(
        pl.col("interval").str.split(" - ").list.get(0).alias("interval_start"),
    )
    df = df.with_columns(
        pl.col("interval_start")
        .str.strptime(pl.Datetime("us", time_zone="UTC"), "%d.%m.%Y %H:%M")
        .alias("time"),
        pl.col("actual_load").cast(pl.Float64, strict=False),
    )
    return df.select("time", "actual_load")


def _load_hourly_load(data_dir: Path) -> pl.DataFrame:
    """Load ENTSO-E CSVs, resample sub-hourly years (2025: 15-min) to hourly."""
    parts = sorted(data_dir.glob("Total Load - Day Ahead _ Actual_*.csv"))
    if not parts:
        raise FileNotFoundError(f"No ENTSO-E load CSVs found under {data_dir}")
    df = pl.concat([_parse_load_csv(p) for p in parts]).sort("time")
    return (
        df.group_by_dynamic("time", every="1h")
        .agg(pl.col("actual_load").mean())
        .sort("time")
    )


def _load_weather(data_dir: Path) -> pl.DataFrame:
    """National-mean hourly weather across the 10 cities."""
    paths = [data_dir / f"weather_{city}.parquet" for city in WEATHER_CITIES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing weather parquets: {missing}")
    frames = [pl.read_parquet(p).select("time", *WEATHER_COLUMNS) for p in paths]
    stacked = pl.concat(frames)
    return (
        stacked.group_by("time")
        .agg(pl.col(c).mean() for c in WEATHER_COLUMNS)
        .with_columns(pl.col("time").dt.cast_time_unit("us"))
        .sort("time")
    )


def build_supervised_frame(data_dir: str | Path) -> pl.DataFrame:
    """Assemble the supervised dataset for `load[t+12]` direct forecasting.

    Returns a polars DataFrame with one row per prediction time `t` carrying:

    - ``prediction_time`` — `t` (UTC, hourly); used by the splitter via
      ``split_kwargs={"timestamps": ...}``. Not a feature.
    - ``target_time`` — `t + 12h`; fed to skrub's ``DatetimeEncoder`` for
      calendar features.
    - ``target_<weather>`` columns — national-mean weather at `t + 12h`.
    - ``last_load`` — `load[t]`, the most recent observed load.
    - ``target_load`` — `load[t+12]`; the regression target.

    Rows with missing inputs (incomplete weather coverage at the start of
    the range, missing load values) are dropped.
    """
    data_dir = Path(data_dir)
    load = _load_hourly_load(data_dir)
    weather = _load_weather(data_dir)
    df = load.join(weather, on="time", how="inner").sort("time")

    target_renames = {c: f"target_{c}" for c in WEATHER_COLUMNS}
    target_renames["actual_load"] = "target_load"

    df = df.with_columns(
        pl.col("time").shift(-HORIZON_HOURS).alias("target_time"),
        pl.col("actual_load").alias("last_load"),
        *[
            pl.col(c).shift(-HORIZON_HOURS).alias(target_renames[c])
            for c in (*WEATHER_COLUMNS, "actual_load")
        ],
    )

    feature_cols = [
        "target_time",
        *[f"target_{c}" for c in WEATHER_COLUMNS],
        "last_load",
    ]
    df = (
        df.rename({"time": "prediction_time"})
        .select("prediction_time", *feature_cols, "target_load")
        .drop_nulls()
    )
    return df


def load_dataset(data_dir: str | Path | None = None):
    """Eager (X, y) loader, used for interactive exploration only.

    The pipeline does not call this — it calls ``build_supervised_frame``
    inside the DataOps graph. Kept here so notebooks / scripts can poke
    at the supervised frame without going through skrub.
    """
    if data_dir is None:
        from . import PROJECT_ROOT

        data_dir = PROJECT_ROOT / "data"
    df = build_supervised_frame(data_dir)
    y = df["target_load"]
    X = df.drop(["target_load", "prediction_time"])
    return X, y
