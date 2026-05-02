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
    HORIZON_HOURS,
    LOAD_LAGS_HOURS,
    WEATHER_COLS,
    add_calendar_features,
    add_load_lags,
    add_load_rollings,
    add_multi_output_targets,
    add_target,
    add_weather_window_means,
    aggregate_weather_across_cities,
    expand_to_horizons,
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


def load_dataset(
    data_dir: str | Path = "data",
    lags_hours: tuple[int, ...] = LOAD_LAGS_HOURS,
    rolling_windows_hours: tuple[int, ...] = (),
) -> pl.DataFrame:
    """Build the full dataset for h=24 French electricity load forecasting.

    Parameters
    ----------
    data_dir : str | Path
        Directory holding the ENTSO-E load CSVs and the per-city weather
        parquets. Bound by ``skrub.var("data_dir", ...)`` in the pipeline.
    lags_hours : tuple of int
        Backward lag offsets (in hours) for ``load_lag_<h>h`` features.
        Defaults to ``(1, 24, 168)`` — the baseline set.
    rolling_windows_hours : tuple of int
        Backward rolling-window sizes (in hours) for
        ``load_roll<W>_mean`` / ``load_roll<W>_std`` features. Empty by
        default → no rolling features (baseline).

    Returned columns:

    - ``datetime``: prediction time t (Datetime UTC). Used by the splitter
      directly; dropped from the feature matrix before the predictor.
    - ``load_t``: load(t) — current load (past covariate / feature).
    - ``load_lag_<h>h``: load(t − h) for each h in ``lags_hours``.
    - ``load_roll<W>_mean`` / ``load_roll<W>_std``: backward rolling mean
      and std of load over ``[t − W, t − 1]`` for each W in
      ``rolling_windows_hours`` (omitted when the tuple is empty).
    - 5 weather columns at t + 24h, mean across the 10 cities.
    - ``cal_hour`` / ``cal_dow`` / ``cal_month`` / ``cal_is_holiday``:
      calendar features at t + 24h, in Europe/Paris local time.
    - ``target``: load(t + 24h).

    Rows with any null (lag / rolling warmup at the head, target
    lookahead at the tail, weather coverage gaps) are dropped.
    """
    data_dir = Path(data_dir)
    load = _load_load_csvs(data_dir)
    weather = _load_weather_parquets(data_dir)

    frame = load.join(weather, on="datetime", how="inner").sort("datetime")
    frame = add_load_lags(frame, lags_hours=lags_hours)
    frame = add_load_rollings(frame, windows_hours=rolling_windows_hours)
    frame = add_target(frame)
    frame = shift_future_weather(frame)
    frame = add_calendar_features(frame)

    return frame.drop_nulls().rename({"load": "load_t"})


def load_multi_output_dataset(
    data_dir: str | Path = "data",
    horizons: tuple[int, ...] = tuple(range(1, HORIZON_HOURS + 1)),
    lags_hours: tuple[int, ...] = LOAD_LAGS_HOURS,
    rolling_windows_hours: tuple[int, ...] = (),
) -> pl.DataFrame:
    """Multi-output dataset for the multi-output regressor framing.

    One row per prediction time ``t``. Features (single vector shared
    across all 24 outputs):

    - ``datetime`` (kept for the splitter; dropped before the predictor),
    - ``load_t``, ``load_lag_<h>h``, optional ``load_roll<W>_*`` —
      past covariates,
    - ``cal_hour`` / ``cal_dow`` / ``cal_month`` / ``cal_is_holiday``
      computed at ``t`` (Europe/Paris local) — calendar at the
      *prediction* time, not at any specific horizon,
    - ``<weather>_window_mean`` for each weather variable — mean over
      ``[t + 1, t + max(horizons)]``, a coarse summary of upcoming
      weather.

    Targets (one column per horizon): ``target_h<h>`` = ``load(t + h)``
    for each ``h`` in ``horizons``.

    Rows with any null are dropped at the end (lag warmup, weather
    coverage gap, target / window-mean look-ahead at the tail).
    """
    data_dir = Path(data_dir)
    load = _load_load_csvs(data_dir)
    weather = _load_weather_parquets(data_dir)

    frame = load.join(weather, on="datetime", how="inner").sort("datetime")
    frame = add_load_lags(frame, lags_hours=lags_hours)
    frame = add_load_rollings(frame, windows_hours=rolling_windows_hours)
    frame = add_multi_output_targets(frame, horizons=horizons)
    frame = add_weather_window_means(frame, horizon_window=max(horizons))
    frame = add_calendar_features(frame, horizon_hours=0)

    # Drop the at-t weather columns: only the window means are kept as
    # the future-covariate summary in this framing.
    frame = frame.drop(*WEATHER_COLS)

    return frame.drop_nulls().rename({"load": "load_t"})


def load_horizon_feature_dataset(
    data_dir: str | Path = "data",
    horizons: tuple[int, ...] = tuple(range(1, HORIZON_HOURS + 1)),
    lags_hours: tuple[int, ...] = LOAD_LAGS_HOURS,
    rolling_windows_hours: tuple[int, ...] = (),
) -> pl.DataFrame:
    """Multi-horizon long-format dataset for the horizon-as-feature framing.

    For each prediction time ``t`` in the joined ``[datetime, load,
    weather]`` base frame and each ``h`` in ``horizons``, emit one row
    with past covariates, future covariates aligned to ``t + h``, a
    numeric ``horizon`` column, and a ``target = load(t + h)`` column.
    See :func:`fr_load_forecast.features.expand_to_horizons` for the
    expansion mechanics.

    Returned columns:

    - ``datetime``: prediction time t (UTC). Replicas of the same t
      share the same datetime → splitter assigns them to the same fold.
    - ``load_t`` and ``load_lag_<h>h`` (and optional ``load_roll<W>_*``):
      past covariates, identical across all replicas of a given t.
    - 5 weather columns aligned to t + h.
    - ``cal_hour`` / ``cal_dow`` / ``cal_month`` / ``cal_is_holiday``
      at t + h, in Europe/Paris local time.
    - ``horizon``: numeric h (1..24 by default).
    - ``target``: load(t + h).
    """
    data_dir = Path(data_dir)
    load = _load_load_csvs(data_dir)
    weather = _load_weather_parquets(data_dir)

    base = load.join(weather, on="datetime", how="inner").sort("datetime")
    base = add_load_lags(base, lags_hours=lags_hours)
    base = add_load_rollings(base, windows_hours=rolling_windows_hours)

    expanded = expand_to_horizons(base, horizons=horizons)
    return expanded.rename({"load": "load_t"})
