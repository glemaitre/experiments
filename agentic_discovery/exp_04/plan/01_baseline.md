# 01_baseline

<!--
Design note for `experiments/01_baseline.py`. Same stem, one-to-one
with the script. Owner: `iterate-ml-experiment`. Frozen at `approved`
except for the Status block.
-->

## Question / hypothesis

What error level is achievable on 24-hour-ahead French electricity load forecasting using a generic tabular-regression baseline (skrub `tabular_learner`) over load lags, multi-city weather, and calendar features?

## Motivation

- **Sourcing strategy:** bootstrap + user override
- **Source(s):**
  - `data/README.md` — defines the prediction horizon (≤24h), the available signals (load history, weather for 10 cities, calendar/holidays), and the time range (2021-03-23 → 2025-05-31).
  - User override on the splitter — replaced the default positional `TimeSeriesSplit(n_splits=5)` with a custom **datetime-anchored expanding walk-forward** splitter (half-yearly test blocks).
- **Why this matters:** Without a baseline, no subsequent feature, model, or methodology change can be judged. This experiment establishes the starting metric and surfaces the structural choices (horizon framing, splitter, feature aggregation) that future iterations will revisit. The user's splitter override aligns folds to calendar boundaries so each fold tests a contiguous half-year of real wall-clock time — the same regime the production model would face.

## Method

- **Files touched:** `src/fr_load_forecast/data.py`, `src/fr_load_forecast/features.py`, `src/fr_load_forecast/pipeline.py`, `src/fr_load_forecast/evaluate.py`, `experiments/01_baseline.py`.
- **Framing — direct, single-horizon forecast at h=24h.** At each prediction time `t`, target = `load(t + 24h)`. Features available at `t`:
  - **Past covariates (load lags):** `load(t)`, `load(t − 1h)`, `load(t − 24h)`, `load(t − 168h)` (last hour, day, week).
  - **Future covariates (weather):** weather variables at the target time `t + 24h`, aggregated across the 10 cities (initial baseline: simple mean per variable). 5 of the 6 available variables are kept: `temperature_2m`, `precipitation`, `wind_speed_10m`, `cloud_cover`, `relative_humidity_2m`. **`soil_moisture_1_to_3cm` is dropped** — its first non-null value is 2022-11-13, which would truncate the dataset to ~2.5 years and starve the early walk-forward folds. User decision at implementation time; re-introducing it is a backlog candidate.
  - **Future covariates (calendar):** hour-of-day, day-of-week, month, and a French-holiday flag at the target time `t + 24h`.
- **Learner:** skrub `tabular_learner(...)` configured for regression. Per `build-ml-pipeline` Common pattern #2, this is the default starting point for tabular data — it bundles `TableVectorizer` preprocessing with a `HistGradientBoostingRegressor` and handles mixed numeric / categorical / datetime columns out of the box.
- **Cross-validator:** custom **datetime-anchored expanding walk-forward** splitter (lives in `src/fr_load_forecast/evaluate.py`). Per `evaluate-ml-pipeline` rule 5 ("custom splitter — only when sklearn doesn't have it"), the standard `sklearn.TimeSeriesSplit` only knows positional indices; here we want folds aligned to the actual datetime column so each test block is a real half-year of wall-clock time. Parameters:
  - **Initial training window:** 1 year (2021-03-23 → 2022-03-22).
  - **Test block size:** 6 months.
  - **Step size:** 6 months (non-overlapping test blocks).
  - **Window shape:** expanding — each fold trains on **all** history up to the cutoff.
  - **Folds:** ~6 full + 1 partial (the trailing 2025-03-23 → 2025-05-31 stub may be kept or dropped; decided at implementation time).
  - The splitter consumes the datetime column wired in via `.skb.mark_as_X(split_kwargs={"times": ...})`; the implementation contract is a small class with `split` + `get_n_splits` per the `evaluate-ml-pipeline` `references/custom-splitter.md`.
- **Metrics:** skore defaults for regression — MSE, RMSE, MAE, R². Per `evaluate-ml-pipeline` rule 4, do not override unless explicitly requested.
- **Out of scope for this experiment:**
  - multi-step / multi-output forecasting (h=1 through h=24 jointly);
  - recursive forecasting (feeding predictions back as inputs);
  - per-city weather modelling (only the cross-city mean here);
  - hyperparameter tuning;
  - holiday-encoding refinement beyond a binary flag;
  - any custom splitter (purged windows, walk-forward with embargo).

## Risks / things that could invalidate the result

- **Horizon framing mismatch:** the project goal is "≤ 24h"; this baseline only forecasts at h=24. If the downstream consumer needs the full curve h=1..24, this baseline gives a single point on it and is not directly comparable to a multi-step learner trained jointly.
- **Cross-city weather mean is lossy:** averaging 10 cities collapses potentially informative spatial variance (Atlantic vs. Mediterranean, urban vs. rural temperature). The baseline error will likely overstate what's achievable with per-city or population-weighted features.
- **Calendar / holiday encoding is shallow:** a single binary holiday flag misses long-weekends, school-holiday periods, summer-vacation regime shifts, and DST transitions — all of which materially shift French load.
- **Expanding walk-forward → unequal training-window lengths.** Fold 1 trains on ~1 year, fold 6 trains on ~3.5 years. The averaged headline metric mixes folds with very different training conditions; per-fold metrics should be reported alongside the average so a regression in early folds isn't masked.
- **Partial trailing test block (2025-03-23 → 2025-05-31, ~2 months):** if kept, it averages a thin slice with full half-year folds; if dropped, the most recent data is unused for evaluation. Decision made at implementation time.
- **Lag warmup:** the 168h (one-week) lag drops the first week of every fold's training window. Edge effects at fold boundaries should be checked once the run completes.
- **Calendar timezone:** ENTSO-E is UTC; French load is on local time (CET / CEST with DST). If hour-of-day is computed in UTC, it will misalign across the year. Need to confirm the timezone handling in `data.py`.

## Status

- **State:** done
- **Approved by user on:** 2026-05-02
- **Headline result:** RMSE 1635 ± 412 MW · MAE 1213 ± 339 MW · MAPE 2.47% ± 0.56% · R² 0.956 ± 0.019 (6 walk-forward folds, 2022-03 → 2025-03). Per-fold range: RMSE 1138 (fold 4, summer 2024) to RMSE 2283 (fold 1, 2022-09 → 2023-03 winter, overlapping the European energy crisis). Fit ≈ 1.7 s / fold, predict ≈ 0.03 s / fold. Skore key: `01_baseline`.
- **Implication for next iteration:** Per-fold variance is ~2× — fold 1 dominates the spread. The strongest next-iteration seed is a **diagnostic walk on fold 1's residuals** (winter peaks? holidays? specific weeks? specific hours?) before reaching for any model / feature change. Other candidates surfaced as backlog items: sliding-window training, per-city weather, school-holiday encoding, additional load lags.
