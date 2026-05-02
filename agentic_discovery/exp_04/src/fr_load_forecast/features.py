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


def add_load_rollings(
    frame: pl.DataFrame,
    windows_hours: tuple[int, ...] = (),
) -> pl.DataFrame:
    """Add backward rolling statistics over each window size in ``windows_hours``.

    For each window ``W`` (in hours), adds ``load_roll<W>_mean`` and
    ``load_roll<W>_std`` computed over the window ``[t − W, t − 1]`` —
    that is, strictly *before* time ``t``, so the current row is excluded
    (the ``shift(1)`` enforces that). When ``windows_hours`` is empty
    (the baseline default), the frame is returned unchanged.
    """
    if not windows_hours:
        return frame
    cols: list[pl.Expr] = []
    for w in windows_hours:
        cols.append(
            pl.col("load").shift(1).rolling_mean(window_size=w).alias(f"load_roll{w}_mean")
        )
        cols.append(
            pl.col("load").shift(1).rolling_std(window_size=w).alias(f"load_roll{w}_std")
        )
    return frame.with_columns(cols)


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


def add_multi_output_targets(
    frame: pl.DataFrame,
    horizons: tuple[int, ...] = tuple(range(1, HORIZON_HOURS + 1)),
) -> pl.DataFrame:
    """Add one ``target_h<h>`` column per horizon for multi-output regression.

    For each ``h`` in ``horizons``, adds a column ``target_h<h>`` equal to
    ``load(t + h)``. Used by the multi-output regressor framing where the
    learner has 24 outputs (one per horizon) and a single feature vector.
    """
    return frame.with_columns(
        [pl.col("load").shift(-h).alias(f"target_h{h}") for h in horizons]
    )


def add_weather_window_means(
    frame: pl.DataFrame,
    horizon_window: int = HORIZON_HOURS,
    weather_cols: tuple[str, ...] = WEATHER_COLS,
) -> pl.DataFrame:
    """Add mean weather over ``[t + 1, t + horizon_window]`` for each weather var.

    Used by the multi-output regressor framing as a coarse summary of
    upcoming weather, since a single feature vector cannot carry
    per-horizon weather. Implementation: shift the column by
    ``-horizon_window`` (so position ``p`` holds the value at
    ``p + horizon_window``), then a backward rolling mean of the same
    window — the result at row ``t`` is the mean over
    ``[t + 1, t + horizon_window]``.
    """
    cols: list[pl.Expr] = []
    for c in weather_cols:
        cols.append(
            pl.col(c)
            .shift(-horizon_window)
            .rolling_mean(window_size=horizon_window)
            .alias(f"{c}_window_mean")
        )
    return frame.with_columns(cols)


def expand_to_horizons(
    base_frame: pl.DataFrame,
    horizons: tuple[int, ...] = tuple(range(1, HORIZON_HOURS + 1)),
    weather_cols: tuple[str, ...] = WEATHER_COLS,
) -> pl.DataFrame:
    """Replicate ``base_frame`` per horizon in long format for the
    horizon-as-feature framing.

    For each row at time ``t`` in ``base_frame`` and each horizon
    ``h ∈ horizons``, emit one row with:

    - past covariates from ``base_frame`` (load, lags, optional rollings) —
      shared across all replicas of a given ``t``,
    - weather columns shifted to ``t + h``,
    - calendar features computed at ``t + h`` (Europe/Paris local),
    - a numeric ``horizon`` column equal to ``h``,
    - a ``target`` column equal to ``load(t + h)``.

    Rows where any feature is null (lag warmup at the head, target
    look-ahead at the tail) are dropped after the per-horizon concat.
    The output preserves the original ``datetime`` column so the splitter
    can use it; replicas of the same ``t`` share the same datetime, so
    they fall into the same train / test fold by construction.
    """
    if not horizons:
        return base_frame.drop_nulls()
    sub_frames: list[pl.DataFrame] = []
    for h in horizons:
        sub = (
            base_frame.pipe(
                shift_future_weather, horizon_hours=h, weather_cols=weather_cols
            )
            .pipe(add_target, horizon_hours=h)
            .pipe(add_calendar_features, horizon_hours=h)
            .with_columns(pl.lit(h, dtype=pl.Int32).alias("horizon"))
        )
        sub_frames.append(sub)
    return pl.concat(sub_frames, how="vertical").drop_nulls()
