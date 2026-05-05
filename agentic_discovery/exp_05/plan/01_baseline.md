# 01_baseline

## Question / hypothesis

Does a single skrub `tabular_learner` predicting `load[t+12]` from
calendar features at `t+12`, national-mean weather at `t+12`, and the
last observed load `load[t]` produce a sensible regression error on
France's hourly load — sensible enough to act as the reference any
future feature-engineering or model-class change must beat?

## Motivation

- **Sourcing strategy:** n/a — bootstrap (forced workspace baseline)
- **Source(s):**
  - `data/README.md` (workspace problem statement: ≤ 24 h horizon,
    weather + calendar treated as future covariates known at
    prediction time, load history as past covariate)
- **Why this matters:** Without a baseline number we can't tell
  whether per-city weather, richer lags, holiday flags, or a different
  estimator are paying their complexity. The baseline is deliberately
  simple so the next iteration has clean signal to compare against.

## Method

- **Files touched:**
  - `src/load_forecast/data.py` — load and align ENTSO-E CSVs +
    weather parquets on a UTC hourly index; build `(X, y)` where `y`
    is `load[t+12]` and `X` carries the prediction-time-known
    features. Attach the time column at the X marker via
    `split_kwargs` so `evaluate.py` can pick a time-aware splitter.
  - `src/load_forecast/features.py` — small helpers: shift target by
    12 hours, compute national-mean weather across the 10 cities,
    expose the prediction-time timestamp `t+12` so calendar features
    encode the right hour.
  - `src/load_forecast/pipeline.py` — `build_learner` returns a
    skrub DataOps `tabular_learner("regressor")` learner. Default
    skrub preprocessing (TableVectorizer with DatetimeEncoder for the
    `t+12` timestamp, numeric scaling for weather + lag) feeds the
    skrub default regressor (HistGradientBoostingRegressor).
  - `src/load_forecast/evaluate.py` — exposes a **custom**
    `splitter = CalendarMonthSplit(embargo_hours=12,
    min_train_months=12)` defined in the same module. Each test
    fold is one calendar month; train is everything strictly before
    `(start_of_test_month - embargo_hours)` (expanding window). The
    `embargo_hours=12` matches the forecast horizon and prevents
    targets in the train set from leaking into the test month
    (a row at time `s` carries target `load[s+12]`). `min_train_months=12`
    gives every fold at least one year of history so the early-fold
    error bars aren't dominated by undertraining. The splitter takes
    a `timestamps` array via `split_kwargs`, not numpy positional
    indexing — so the X marker attaches the timestamp column there.
    No metric overrides; rely on skore's regression defaults.
  - `experiments/01_baseline.py` — opens
    `skore.Project(workspace="reports", name="load-forecast",
    mode="local")`, calls `skore.evaluate`, persists under key
    `"01_baseline"`.
- **Change versus baseline (or previous experiment):** n/a — this *is*
  the baseline.
- **Out of scope for this experiment:** ENTSO-E day-ahead forecast as
  a benchmark, per-city weather, richer lags, holiday flags,
  multi-horizon forecasting. All five live in `PLAN.md` Backlog
  (B1–B5).

## Risks / things that could invalidate the result

- **Custom splitter — bug surface.** A handwritten splitter is
  more error-prone than `TimeSeriesSplit`. The first run should
  print one fold's `(timestamps[train].max(), timestamps[test].min())`
  pair to verify the embargo holds and the test window is exactly
  one calendar month.
- **National-mean weather is coarse.** France spans climates that
  drive load differently (heating-dominant Lille, cooling-dominant
  Marseille). The mean may wash out the signal. Promoted to backlog
  (B2).
- **Single load lag.** A standard practice is at least `load[t]`,
  `load[t-24]`, `load[t-168]`. The baseline keeps only `load[t]` to
  start small; B3 covers the extension.
- **Public holidays not encoded.** The DatetimeEncoder gets day-of-week
  and month but not French national / regional holidays, which shift
  load substantially (Christmas, May 1st, etc.). B4 covers it.

## Status

- **State:** done
- **Approved by user on:** 2026-05-04
- **Headline result:** MAE 2621 ± 685 MW · RMSE 3352 ± 777 MW · MAPE 5.19 ± 1.10 % · R² 0.67 ± 0.12 (19 monthly walk-forward folds, Nov 2023 → May 2025; 12 h embargo, 12 month warm-up). Per-fold fit time 1.8 s, predict time 0.03 s.
- **Implication for next iteration:** ~5 % MAPE on a near-trivial baseline; the natural next step is **B1** (benchmark against the ENTSO-E day-ahead forecast already in the load CSVs — that's the operational reference any honest model has to beat). After that, B2 (per-city weather) and B3 (richer load lags) are the most likely structural lifts.
