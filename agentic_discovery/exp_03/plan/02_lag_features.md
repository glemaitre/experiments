# 02_lag_features

## Question / hypothesis

Once the model has explicit within-series memory at the Agency-SKU level — last-month volume, trailing 3/6/12-month means, and a year-over-year value — does the forecast actually improve, or was the baseline's R² 0.934 already saturating what side-table month-level signal can buy? The split of "added value from lags" vs. "saturation" is the real result; either answer is informative.

## Motivation

- **Sourcing strategy:** backlog:B2
- **Source(s):**
  - `PLAN.md` Backlog row B2 — "Lag / rolling-window features (last-month, 3/6/12-month means, year-over-year) at Agency-SKU level — classical forecasting features the baseline doesn't carry."
  - `01_baseline` headline (R² 0.934 ± 0.021, RMSE 695 ± 130 hL) — the implication block flagged that the baseline's strength may come from *cross-series* month-level signal (industry volume, weather, calendar) rather than *within-series* demand dynamics. Adding lag features is the first cheap test of that hypothesis.
- **Why this matters:** lags / rolling means are the canonical first-step feature engineering for monthly demand forecasting. If R² barely moves, the model is already extracting almost everything from the side-tables — and the next iteration should be a methodology audit (B5, B6) rather than more features. If R² jumps materially, the gain quantifies how much within-series memory was missing in the baseline.

## Method

- **Files touched:** `src/beeristan/features.py` (new — currently empty), `src/beeristan/pipeline.py` (insert one `.skb.apply_func` step between the loader and the X marker). `data.py` and `evaluate.py` are not touched.
- **Change versus `01_baseline`:** add a single stateless function `add_lag_features(panel)` to `features.py` that, after the multi-table join is materialized, computes per-(Agency, SKU) lag/rolling columns over the `Volume` series:
  - `Volume_lag_1` — previous month's volume.
  - `Volume_lag_12` — same month one year ago (year-over-year).
  - `Volume_rolling_mean_3`, `_6`, `_12` — **trailing** rolling means, computed as `Volume.shift(1).rolling(window=k).mean()` so they only see strictly-past months.
  - All five computed within `(Agency, SKU)` groups, sorted by `Date`.
  - Early months for each series will have nulls; left as nulls (skrub's `tabular_pipeline` handles them via the imputer in `TableVectorizer`). No early-row dropping — that would silently shrink the panel and break the cardinality invariant from `data.py`.
- The function is attached via `.skb.apply_func` (stateless: same row receives the same value regardless of which CV subset it's computed on, because it only references *past* rows of the *same* (Agency, SKU)).
- **Out of scope for this experiment:** lags on side-table columns (price/promo lags, weather lags); EWMA / non-uniform weighting; per-series scaling; lag-feature interactions; tuning `tabular_pipeline`'s defaults; B5 (MAPE / robust relative metric); B6 (per-fold-month variance decomposition).

## Risks / things that could invalidate the result

- **Subtle leakage from "rolling including current".** The intent is *trailing* rolling means: `Volume.shift(1).rolling(k).mean()`. If implemented as plain `.rolling(k).mean()` (no `shift(1)`), the current row's own `Volume` enters its own feature — direct target leakage that would massively inflate R². Will be sanity-checked: every `Volume_rolling_mean_k` column for the *first* row of each (Agency, SKU) series must be NaN (proves we shifted before rolling).
- **Cold-start rows have many NaNs.** The first few months of every series will have null `Volume_lag_*` / `Volume_rolling_*` values. With `tabular_pipeline("regressor")` the tail estimator is `HistGradientBoostingRegressor`, which handles NaN natively (each split picks a missing-value direction) — no imputer is in the pipeline, and "cold-start" effectively becomes its own learnable signal rather than being fused into a column mean. If a future iteration swaps in a linear model, skrub will auto-insert an imputer; that's a different regime worth flagging at the time, not now.
- **`(Agency, SKU)` pair appears in only one or two months.** A series with one observation has all-null lag features; the model has nothing within-series to lean on. The score for those rows is dominated by the side-table contribution — same regime as the baseline. Worth surfacing if a slice analysis later shows huge per-pair variance.
- **Sort discipline.** The lag computation requires `(Agency, SKU)` partitioning and `Date` ordering. `data.py` already sorts by `(Date, Agency, SKU)`; the feature function will resort by `(Agency, SKU, Date)` before computing lags, then restore the original order so downstream rows align with the panel.
- **R² may *drop*** if the cold-start NaN signal turns out to be a noisy proxy that the tree picks up but doesn't generalize, or if the lag features are simply redundant with the side-table month signal the baseline already exploits. Either is a real outcome, not a bug; the implication block will record what was learned rather than treat it as failure.

## Status

- **State:** done
- **Approved by user on:** 2026-05-02
- **Headline result:** R² 0.962 ± 0.017, RMSE 520 ± 112 hL, MAE 221 ± 43 hL (16-fold walk-forward, same splitter as `01_baseline`). Versus baseline: ΔRMSE −175 hL (−25 %), ΔMAE −119 hL (−35 %), ΔR² +0.028, fold std also tighter. MAPE still ~4e15, same zero/near-zero `Volume` artifact as the baseline.
- **Implication for next iteration:** the lag features carry real signal, so the baseline was *not* fully saturated on side-table month-level cues — but R² 0.962 is now uncomfortably high for a tree-on-tabular forecast and amplifies the methodology concern raised after `01_baseline`. The single most valuable next step is a methodology audit: confirm none of the joined columns (or the lag computation across the panel before splitting) sneaks future information into the train fold. Once the audit clears, the diagnostic dispatch is ripe — the per-fold-month residual breakdown (B6) and the MAPE artifact (B5) become safe to look at.
