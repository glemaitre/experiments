"""Feature functions and transformers.

Pure (stateless) feature builders for h=24 French electricity load
forecasting. Composition into the learner happens in `pipeline.py`.
See `build-ml-pipeline` for declarative mechanics.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

LOAD_LAGS_HOURS: tuple[int, ...] = (1, 24, 168)
HORIZON_HOURS: int = 24
LOCAL_TZ: str = "Europe/Paris"

WEATHER_COLS: tuple[str, ...] = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "relative_humidity_2m",
)


def _easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm — Easter Sunday for the given year."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def french_holidays(years: range) -> list[dt.date]:
    """French national holidays (fixed-date + Easter-based) for the given years."""
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    out: list[dt.date] = []
    for y in years:
        out.extend(dt.date(y, m, d) for m, d in fixed)
        easter = _easter_sunday(y)
        # Easter Monday (+1), Ascension (+39), Pentecost Monday (+50).
        for delta in (1, 39, 50):
            out.append(easter + dt.timedelta(days=delta))
    return out


def aggregate_weather_across_cities(
    weather_per_city: list[pl.DataFrame],
    weather_cols: tuple[str, ...] = WEATHER_COLS,
) -> pl.DataFrame:
    """Stack per-city weather frames and compute the cross-city mean per variable.

    Each input frame has columns: ``time`` plus ``weather_cols``. The output
    has one row per ``time`` with the mean (null-skipping) of each weather
    column across cities, renamed to ``datetime`` for join consistency.
    """
    stacked = pl.concat(weather_per_city, how="vertical")
    return (
        stacked.group_by("time")
        .agg([pl.col(c).mean() for c in weather_cols])
        .rename({"time": "datetime"})
        .sort("datetime")
    )


def add_load_lags(
    frame: pl.DataFrame, lags_hours: tuple[int, ...] = LOAD_LAGS_HOURS
) -> pl.DataFrame:
    """Add ``load_lag_<h>h`` columns = load(t − h) for h in ``lags_hours``."""
    return frame.with_columns(
        [pl.col("load").shift(h).alias(f"load_lag_{h}h") for h in lags_hours]
    )


def add_target(
    frame: pl.DataFrame, horizon_hours: int = HORIZON_HOURS
) -> pl.DataFrame:
    """Add ``target`` = load(t + horizon_hours)."""
    return frame.with_columns(
        pl.col("load").shift(-horizon_hours).alias("target")
    )


def shift_future_weather(
    frame: pl.DataFrame,
    horizon_hours: int = HORIZON_HOURS,
    weather_cols: tuple[str, ...] = WEATHER_COLS,
) -> pl.DataFrame:
    """Shift weather columns so each row holds weather at t + horizon_hours."""
    return frame.with_columns(
        [pl.col(c).shift(-horizon_hours) for c in weather_cols]
    )


def add_calendar_features(
    frame: pl.DataFrame, horizon_hours: int = HORIZON_HOURS
) -> pl.DataFrame:
    """Add calendar features (hour, weekday, month, holiday flag) at t + horizon_hours.

    All calendar features are computed in Europe/Paris local time so they
    align with French load patterns across DST transitions.
    """
    target_local = (
        pl.col("datetime")
        .dt.offset_by(f"{horizon_hours}h")
        .dt.convert_time_zone(LOCAL_TZ)
    )

    target_year_series = frame.select(target_local.dt.year().alias("y")).to_series()
    year_min = int(target_year_series.min())
    year_max = int(target_year_series.max())
    holidays = french_holidays(range(year_min, year_max + 1))

    return frame.with_columns(
        target_local.dt.hour().alias("cal_hour"),
        target_local.dt.weekday().alias("cal_dow"),
        target_local.dt.month().alias("cal_month"),
        target_local.dt.date().is_in(holidays).alias("cal_is_holiday"),
    )
