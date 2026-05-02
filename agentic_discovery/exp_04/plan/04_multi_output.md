# 04_multi_output

<!--
Design note for `experiments/04_multi_output.py`. Same stem,
one-to-one with the script. Owner: `iterate-ml-experiment`. Frozen at
`approved` except for the Status block.
-->

## Question / hypothesis

Does a **multi-output regression** framing (one estimator with 24 outputs via `MultiOutputRegressor` wrapping `HistGradientBoostingRegressor`, with a single feature vector shared across all outputs) yield competitive per-horizon RMSE compared to the horizon-as-feature framing in `03_horizon_as_feature` and to the h=24 result of `01_baseline`?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request after `02_more_load_lags` to implement multi-horizon prediction; user explicitly picked the multi-output regressor as the second framing to compare against horizon-as-feature.
- **Why this matters:** Multi-output regression is the standard alternative to horizon-as-feature for direct multi-step forecasting. It tests whether *parameter sharing across 24 sub-models* (each output gets its own internal HGB inside `MultiOutputRegressor`) generalizes better or worse than horizon-as-feature's *parameter sharing through a single model with `h` as a feature*. The two framings are paired: each is a natural baseline for the other.

## Method

- **Files touched:**
  - `src/fr_load_forecast/features.py` — add `add_multi_output_targets(frame, horizons)` (24 target columns: `target_h1`, …, `target_h24`) and `add_weather_window_means(frame, horizon_window=24)` (mean of each weather variable over `[t+1, t+horizon_window]`).
  - `src/fr_load_forecast/data.py` — add a new loader `load_multi_output_dataset(data_dir, horizons=range(1,25), lags_hours=..., rolling_windows_hours=())` *alongside* the existing `load_dataset`. The existing loader and its defaults are **unchanged**, so re-running `01_baseline.py`, `02_more_load_lags.py`, and `03_horizon_as_feature.py` from the current codebase reproduces their stored metrics exactly.
  - `src/fr_load_forecast/pipeline.py` — add `build_multi_output_learner(horizons=range(1,25), lags_hours=..., rolling_windows_hours=())` alongside the existing `build_learner` and (after `03_horizon_as_feature`) `build_horizon_feature_learner`. Existing builders unchanged.
  - `experiments/04_multi_output.py` — new script; calls `build_multi_output_learner()`, evaluates with the same splitter, persists report under skore key `04_multi_output`.

- **Loader behavior** (`load_multi_output_dataset`):
  - Build the same `[datetime, load, weather]` base frame as `load_dataset`.
  - Compute past-covariate columns (lags, optionally rollings).
  - Add **calendar at t** (not at t + h) — `cal_hour_t`, `cal_dow_t`, `cal_month_t`, `cal_is_holiday_t`. These tell the model *when the forecast is being issued*, which is what a single feature vector can represent.
  - Add **weather window means** — for each weather variable, the mean over `[t + 1, t + 24]`. Coarse summary of upcoming weather; the cleanest way to inject future-covariate signal under a single-vector framing.
  - Add 24 target columns: `target_h1 = load(t + 1)`, …, `target_h24 = load(t + 24)`.
  - Drop nulls (lag-warmup head + target-lookahead tail of 24 h).

- **Learner:** `skrub.tabular_pipeline(MultiOutputRegressor(HistGradientBoostingRegressor(), n_jobs=-1))`. The `MultiOutputRegressor` wrap is **mandatory**: `HistGradientBoostingRegressor` does not support native multi-output as of sklearn 1.8 (verified by direct call — it raises `ValueError: y should be a 1d array`). `n_jobs=-1` parallelizes the 24 sub-fits.

- **Splitter:** same `DatetimeAnchoredWalkForward`. One row per prediction time → no replication, no special handling.

- **Reporting & comparison:**
  - skore's regression metrics on multi-output `y`. Per-output (i.e. per-horizon) metrics are accessible via the `CrossValidationReport`'s per-fold estimator reports.
  - Headline rendered for the user: per-horizon RMSE curve (one point per output), aligned alongside `03_horizon_as_feature`'s curve and `01_baseline`'s h=24 point.

- **Skore key:** `04_multi_output`, in a sibling project `fr-load-forecast-mh` (same `reports/` workspace). Reason: skore constrains each Project to a single ML task; `01`–`03` are `regression` reports, `04` is `multioutput-regression`, so they cannot share the same Project. Cross-project comparison happens post-hoc by loading both projects.

## Risks / things that could invalidate the result

- **Feature-set asymmetry vs. `03_horizon_as_feature`.** This experiment uses calendar at `t` (single value) and a 24h-mean weather summary; `03_horizon_as_feature` uses calendar at `t + h` and weather at `t + h`. Each gets the most natural feature set for its framing, so the 03-vs-04 comparison answers "which framing wins given each framing's natural feature set", not "which model architecture wins under matched features". A matched-feature ablation is a backlog candidate.
- **Information bandwidth per output.** Each output gets the same feature vector, summarizing 24 hours of upcoming weather into a mean. Expected: this experiment underperforms `03_horizon_as_feature` at long horizons where future-weather signal matters most, and possibly outperforms at short horizons where the mean-summary is essentially current weather.
- **Compute.** `MultiOutputRegressor` fits 24 sub-models per fold; `n_jobs=-1` parallelizes. Per-fold time should be in the same order as `03_horizon_as_feature` (~30–60 s). Total ≈ 5 min.
- **h=24 not directly comparable to `01_baseline`.** Feature set differs (mean-window weather vs. point weather at t+24h; calendar at t vs. calendar at t+24h). Expect "in the ballpark" of 1500–1800 MW, not an exact 1635 match.
- **Embargo gap (B6).** Same 24h target overlap as before; symmetric across all experiments.
- **`MultiOutputRegressor` sub-model independence.** Each sub-model is fit independently; there is no learned coupling across horizons. If we want cross-horizon coupling, `RegressorChain` is the natural next step — backlog candidate if 04 underperforms 03.

## Status

- **State:** done
- **Approved by user on:** 2026-05-02
- **Headline result:** Per-horizon RMSE forms a **plausible curve** rising from h=1 (RMSE 843 MW) through a midday peak (h=15: 2547 MW) and back down at h=24 (1971 MW). R² 0.988 at h=1 down to ~0.92 at h=15 and ~0.95 at h=24. Aggregate (24 outputs × 6 folds): RMSE ≈ 2050 MW, MAPE ≈ 3% range. Fit ≈ 1.5 s / fold (24 sub-models, parallelized via `n_jobs=-1`). Skore key: `04_multi_output` in project `fr-load-forecast-mh`.
- **Implication for next iteration:** **Multi-output regression *works* as a multi-horizon framing** (sensible curve, unlike `03_horizon_as_feature`'s flat collapse) but **underperforms `01_baseline` at h=24 by ~330 MW** (1971 vs 1635). Root cause: the single feature vector forces a 24h-mean weather summary and calendar-at-t encoding, both of which discard horizon-specific signal that `01_baseline` exploits via weather/calendar at exactly `t+24h`. The obvious next experiment is **24 independent direct models** (one HGB per horizon, each with weather/calendar at `t+h` like `01_baseline` parameterized) — recovers the per-horizon-feature alignment while still producing a multi-horizon prediction. Captured as backlog item B9.
