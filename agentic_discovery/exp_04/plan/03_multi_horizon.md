# 03_multi_horizon

<!--
Design note for `experiments/03_multi_horizon.py`. Same stem,
one-to-one with the script. Owner: `iterate-ml-experiment`. Frozen at
`approved` except for the Status block.

Umbrella experiment comparing two multi-horizon framings under one
plan / approval / History row, producing two skore keys:
`03_multi_horizon:horizon_feature` and `03_multi_horizon:multi_output`.
Same convention as the skill's "batch re-run" pattern.
-->

## Question / hypothesis

How do two multi-horizon framings — **horizon-as-feature** (one model, rows replicated 24× with horizon `h` as a feature) vs. **multi-output regressor** (one estimator with 24 outputs, wrapped via `sklearn.multioutput.MultiOutputRegressor` since `HistGradientBoostingRegressor` doesn't support native multi-output) — perform across hourly horizons `h ∈ {1, …, 24}`, and how does each compare to `01_baseline` at h=24?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request after `02_more_load_lags`: "implement a way to predict a mode that do multi horizon", followed by an explicit pick to do *both* the recommended single-model framing and a multi-output regressor.
  - `01_baseline` only forecasts at a single fixed horizon. The project goal is "≤ 24h", which implies the full horizon curve.
- **Why this matters:** Multi-horizon forecasting is a structural lever, not a feature-engineering tweak — `02_more_load_lags` confirmed past-covariate engineering is saturated for the h=24 framing, so the next gain must come from how we frame the prediction itself. Comparing two standard direct multi-step methods within one experiment lets us see (a) the per-horizon error curve in two different parameter-sharing regimes and (b) whether `01_baseline`'s h=24 RMSE is recoverable in either regime.

## Method

### Common (both approaches)

- **Same data range, same splitter** (`DatetimeAnchoredWalkForward`, expanding, half-yearly), **same past covariates** (`load_t`, `load_lag_{1,24,168}h`).
- **Files touched:**
  - `src/fr_load_forecast/features.py` — new helpers: `add_calendar_features` already takes a `horizon_hours` argument; add `add_targets_at_horizons` and `add_weather_window_means` for approach B; `expand_to_horizons` (row replication) for approach A.
  - `src/fr_load_forecast/data.py` — two new loaders: `load_horizon_feature_dataset` and `load_multi_output_dataset`. The single-horizon `load_dataset` is unchanged so `01_baseline` and `02_more_load_lags` still work.
  - `src/fr_load_forecast/pipeline.py` — two new learner builders: `build_horizon_feature_learner` and `build_multi_output_learner`. Existing `build_learner` is unchanged.
  - `experiments/03_multi_horizon.py` — runs both, writes two keys (see below).

### Approach A — horizon-as-feature (single model)

- For each prediction time `t`, replicate the row 24× with a `horizon` column ∈ `{1, …, 24}`.
- Per-row feature set:
  - **Past covariates:** `load_t`, `load_lag_1h`, `load_lag_24h`, `load_lag_168h`.
  - **Future covariates aligned to t + h:** weather (5 vars: temperature, precipitation, wind, cloud, humidity — same cross-city mean as baseline), calendar in Europe/Paris (hour, weekday, month, holiday flag).
  - **`horizon`:** numeric `h ∈ {1, …, 24}`. (One-hot vs numeric is a backlog candidate if results disappoint.)
- **Target:** `load(t + h)`.
- **Learner:** `skrub.tabular_pipeline("regressor")` — same `TableVectorizer` + `HistGradientBoostingRegressor` as the baseline.
- **Skore key:** `03_multi_horizon:horizon_feature`.

### Approach B — multi-output regressor

- One row per prediction time `t`. Targets are a 24-column frame `[load(t+1), …, load(t+24)]`.
- Feature set (a *single* vector shared across all 24 outputs):
  - **Past covariates:** `load_t`, `load_lag_1h`, `load_lag_24h`, `load_lag_168h`.
  - **Calendar at t** (not at t+h): hour, weekday, month, holiday — these tell the model *when the forecast is being issued*, which is what a single-vector encoding can carry.
  - **Future-window mean weather:** for each weather variable, the mean over `[t+1, t+24]` — a coarse summary of the upcoming weather, the cleanest way to inject future-covariate signal under a single-vector framing.
- **Targets (24):** `load(t+1), …, load(t+24)`.
- **Learner:** `skrub.tabular_pipeline(MultiOutputRegressor(HistGradientBoostingRegressor(), n_jobs=-1))`. `MultiOutputRegressor` wraps because `HistGradientBoostingRegressor` does not support native multi-output as of sklearn 1.8 (verified by call). `n_jobs=-1` parallelizes the 24 sub-fits.
- **Skore key:** `03_multi_horizon:multi_output`.

### Evaluation & comparison

- Both write `CrossValidationReport`s under the umbrella prefix `03_multi_horizon:*`. Skore picks regression defaults (RMSE / MAE / MAPE / R²).
- **Approach A**: a single global RMSE is reported by skore (averaged across all (t, h) pairs). Per-horizon RMSE is computed *post-hoc* by slicing the CV report's predictions on the `horizon` column. The h=24 slice is what's directly comparable to `01_baseline`.
- **Approach B**: skore's regression metrics on multi-output `y` average across outputs by default; per-output metrics are accessible via the report (each output ≡ one horizon).
- Headline rendered for the user: per-horizon RMSE curve for both approaches, with `01_baseline`'s h=24 point overlaid.

## Risks / things that could invalidate the result

- **Compute.** Approach A trains on 24× rows (~870k vs ~36k) → roughly 24× longer per fold (~40 s vs ~1.7 s). Approach B fits 24 sub-models inside `MultiOutputRegressor` → also ~24× per fold (mitigated by `n_jobs=-1`). Total walk-clock per experiment: 5–10 min on this machine. Acceptable.
- **Feature-set asymmetry.** Approach A uses per-horizon weather and calendar; Approach B uses calendar at t and a 24h-mean weather summary. They are *not* directly comparable feature-by-feature — the comparison answers "which framing wins given each framing's most natural feature set", not "which model architecture wins under matched features". A matched-feature ablation is a backlog candidate.
- **Horizon as numeric vs categorical (Approach A).** A numeric `horizon` feature can let HistGB exploit ordinal smoothness, but if the per-horizon error is highly non-monotonic (e.g., evening vs night), one-hot may help. Worth checking if results disappoint.
- **`load_t` semantics under replication (Approach A).** Every replicated row at time `t` shares the same `load_t`. The model could memorize `(load_t, h)` → `load(t+h)` lookups when training. This is fine — it's not leakage (the test rows use unseen `t`), but it shifts the burden of generalization onto the lag and calendar columns.
- **Multi-output reduces signal per output (Approach B).** Each sub-fit sees the same feature vector for predicting all 24 horizons. The information bandwidth per horizon is lower than approach A's per-row weather. Expected: B underperforms A at long horizons where future weather matters most.
- **h=24 not directly equal to `01_baseline`'s h=24.** Both approaches change the *data shape* and the surrounding training distribution. Even with identical features, the h=24 slice should be in the same ballpark as `01_baseline` (RMSE ~1635 ± 412) but is not guaranteed to match within the noise floor.
- **Same 24h target overlap as before (B6).** Embargo gap unchanged; bias is symmetric across all three experiments so far.
- **Multiple skore keys per experiment.** Follows the skill's "batch re-run" pattern; not a deviation, but worth noting that future `iterate-from-diagnostic` proposals on this experiment will need to specify which key they're reading from.

## Status

- **State:** planned
- **Approved by user on:** TBD
- **Headline result:** TBD
- **Implication for next iteration:** TBD
