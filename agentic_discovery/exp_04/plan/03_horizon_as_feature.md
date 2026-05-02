# 03_horizon_as_feature

<!--
Design note for `experiments/03_horizon_as_feature.py`. Same stem,
one-to-one with the script. Owner: `iterate-ml-experiment`. Frozen at
`approved` except for the Status block.
-->

## Question / hypothesis

Does a single-model **horizon-as-feature** framing (each prediction time replicated 24× with horizon `h ∈ {1, …, 24}` as a numeric feature, target = `load(t + h)`, weather and calendar aligned to `t + h` per row) yield (a) a usable per-horizon error curve for h ∈ {1..24}, and (b) competitive RMSE at the `h = 24` slice compared to `01_baseline`?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request after `02_more_load_lags`: "implement a way to predict a mode that do multi horizon"; user explicitly picked horizon-as-feature as one of two approaches to compare.
  - `02_more_load_lags` confirmed past-covariate engineering is saturated for the h=24 framing — the next gain has to come from a structural change to *how we frame the prediction*.
- **Why this matters:** The project goal is short-horizon forecasting up to 24 h, not just at h=24. Horizon-as-feature is the most parameter-efficient direct multi-step framing — one model, one fit per fold, learns a smooth interpolation across horizons. It's the natural first multi-horizon experiment to run.

## Method

- **Files touched:**
  - `src/fr_load_forecast/features.py` — add `expand_to_horizons(frame, horizons)` (row replication with per-h shifted weather, calendar, target, and a `horizon` column).
  - `src/fr_load_forecast/data.py` — add a new loader `load_horizon_feature_dataset(data_dir, horizons=range(1,25), lags_hours=..., rolling_windows_hours=())` *alongside* the existing `load_dataset`. The existing loader and its defaults are **unchanged**, so re-running `experiments/01_baseline.py` and `experiments/02_more_load_lags.py` from the current codebase reproduces their stored metrics exactly.
  - `src/fr_load_forecast/pipeline.py` — add `build_horizon_feature_learner(horizons=range(1,25), lags_hours=..., rolling_windows_hours=())` alongside the existing `build_learner`. Existing builder unchanged.
  - `experiments/03_horizon_as_feature.py` — new script; calls `build_horizon_feature_learner()`, evaluates with the same splitter, persists report under skore key `03_horizon_as_feature`.

- **Loader behavior** (`load_horizon_feature_dataset`):
  - Build the same `[datetime, load, weather]` base frame as `load_dataset` (sorted, joined inner, post-warmup).
  - Compute past-covariate columns (lags, optionally rollings).
  - For each horizon `h` in the requested set, emit one frame with:
    - past covariates (constant across `h` for a given `t`),
    - weather columns shifted to `t + h`,
    - calendar features at `t + h` in Europe/Paris (existing `add_calendar_features` already takes a `horizon_hours` argument — reuse it),
    - a `horizon` column (numeric, equal to `h`),
    - a `target` column = `load(t + h)`.
  - Concatenate the per-horizon frames vertically (`pl.concat(..., how="vertical")`). Drop nulls (lag-warmup head + target-lookahead tail).
  - Returned shape: roughly `36k × |horizons|` rows (~870k for the full 1..24 set), one target column.

- **Learner:** `skrub.tabular_pipeline("regressor")` — same `TableVectorizer` + `HistGradientBoostingRegressor` as the baseline. The model treats `horizon` as a numeric feature.

- **Splitter:** same `DatetimeAnchoredWalkForward` (1y initial / 6mo test / 6mo step / expanding). All 24 replicas of a prediction time `t` share the same datetime → they fall into the same fold by construction. **No splitter change needed.**

- **Reporting & comparison:**
  - skore reports a global RMSE / MAE / MAPE / R² aggregated across all `(t, h)` pairs.
  - **Per-horizon metrics** are computed *post-hoc* by reading per-fold predictions from the `CrossValidationReport`, joining them back to `X_test` rows, and grouping by the `horizon` column. The h=24 slice is what's directly comparable to `01_baseline` (RMSE 1635 ± 412).
  - The headline rendered for the user: aggregate metrics + a per-horizon RMSE curve over h ∈ {1..24}.

- **Skore key:** `03_horizon_as_feature`.

## Risks / things that could invalidate the result

- **Compute.** ~24× rows → ~24× fit time per fold (~40 s instead of ~1.7 s). Total ≈ 5 min for the experiment. Acceptable on this machine.
- **`horizon` as numeric.** Assumes ordinal smoothness across horizons. If the per-horizon error is highly non-monotonic (e.g., evening peaks vs. overnight troughs), a one-hot or cyclic encoding may help — backlog candidate if results are uneven.
- **`load_t` shared across replicas.** All 24 replicated rows at time `t` share the same `load_t`, `load_lag_*`, etc. The model could partly memorize `(load_t, h) → load(t + h)` mappings on training rows, but test rows have unseen `t`, so this is not leakage — just a soft inductive bias toward looking-up by current load.
- **h=24 not directly comparable to `01_baseline`.** Even though the h=24 slice uses the same target, the surrounding training distribution is now 24× larger and contains all other horizons. RMSE at h=24 should land in the same ballpark (1500–1800 MW) but isn't guaranteed to match `01_baseline`'s 1635 within noise.
- **Embargo gap (B6).** Same 24h target overlap as `01_baseline`. Symmetric across folds and across this experiment vs. the baseline, so relative comparisons stay valid.
- **Per-horizon slicing requires post-hoc work.** Adds a small implementation surface in the experiment script (or a helper); not a methodology risk, but worth flagging that the per-horizon view isn't free from skore.

## Status

- **State:** done
- **Approved by user on:** 2026-05-02
- **Headline result:** Aggregate across 24 horizons + 6 walk-forward folds: **RMSE 2426 ± 746 MW · MAE 1918 ± 669 MW · MAPE 3.98% ± 1.24% · R² 0.902 ± 0.048**. Per-horizon RMSE is **essentially flat at ~2400 MW from h=1 (RMSE 2386) to h=24 (RMSE 2356)** with no monotonic trend. h=24 slice is **markedly worse than `01_baseline`** (2356 vs 1635 MW). Skore key: `03_horizon_as_feature`. Fit ≈ 3 s / fold (single model on 24× rows).
- **Implication for next iteration:** **Horizon-as-feature with default HistGB does not work.** The per-horizon error is flat because the model collapses to predicting the mean target across horizons rather than specializing per-horizon — `horizon` as a single numeric feature is too weak a conditioning signal for HistGB's tree splits to differentiate the 24 sub-problems. Three follow-ups, in order of likely lift:  (a) **one-hot encode `horizon`** so each horizon gets a dedicated split path (cheap fix, candidate for `05_*`); (b) **higher-capacity HGB** (more iterations, deeper trees, more leaves) to make room for h-conditional splits; (c) **the multi-output framing in `04_multi_output`** (already approved + queued) — each output gets its own sub-model, which is structurally what 03 fails to learn implicitly.
