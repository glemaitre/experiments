# 03_side_table_lags

## Question / hypothesis

Once the model already sees `Volume`'s own lags (from `02_lag_features`), do *side-table* lags — last-month price, sales, promotion, weather, and industry volumes — buy any further headline RMSE? The Volume-side lags bought a 25 % RMSE drop; analogous side-table lags either keep paying off (true within-series exogenous dynamics matter) or do almost nothing (`02`'s gain came mostly from the autoregressive `Volume` signal, and the side-tables are most useful at their *current-month* values which are already in the panel).

## Motivation

- **Sourcing strategy:** backlog:B7
- **Source(s):**
  - `PLAN.md` Backlog row B7 — "Lag/rolling on side-table columns (price-lag-1, promo-lag-1, weather-lag-1, industry-volume-lag-1)."
  - `02_lag_features` headline (R² 0.962, RMSE 520 hL) and its implication block — within-series memory turned out to matter; B7 is the parallel question for the *exogenous* columns.
- **Why this matters:** the cheapest possible follow-up to `02`. If side-table lags pay off, we know that price/promo/weather *change* matters, not just current level. If they don't, we've ruled out an obvious feature-engineering direction and can move on to a different lever (calendar/event encoding B3, hierarchical reconciliation B4, robust metric B5, methodology audit) with one less open question.

## Method

- **Files touched:** `src/beeristan/features.py` (add a second function `add_side_table_lag_features`; do not touch `add_lag_features`), `src/beeristan/pipeline.py` (chain a second `.skb.apply_func` after the Volume-lag step). `data.py` and `evaluate.py` are not touched.
- **Change versus `02_lag_features`:** add `lag_1` and trailing `rolling_mean_3` (shift-then-rolling, same discipline as `02`) for six side-table columns:
  - `Price`, `Sales`, `Promotions` — Agency-SKU-month columns; partition by `(Agency, SKU)`.
  - `Avg_Max_Temp` — Agency-month column. Partitioning by `(Agency, SKU)` is still correct because within each `(Agency, SKU)` slice the `Avg_Max_Temp` values are the Agency-level monthly time series; shifting within the slice gives the same answer as partitioning by Agency only.
  - `Industry_Volume`, `Soda_Volume` — month-only columns; same argument as for weather: partition by `(Agency, SKU)` is safe because the values are constant across all `(Agency, SKU)` rows of a given month.
  - 12 new columns total. Brings the feature matrix from 28 columns (post-02) to 40.
- **Out of scope for this experiment:** lag/rolling on the binary event indicators (lagged event flags rarely add anything meaningful); lag-12 (year-over-year) on side-tables — `02`'s `Volume_lag_12` already pulls some of that signal in via the autoregression; per-Agency demographic features (the demographics columns are 2017-only snapshots, time-invariant in this dataset, so a lag is meaningless); methodology audit; tuning `tabular_pipeline`; B3 / B4 / B5 / B6.

## Risks / things that could invalidate the result

- **Trailing-rolling discipline.** Same as `02`: rolling means must use `shift(1).rolling_mean(k)`, never plain `.rolling_mean(k)`. Sanity-checked the same way: every `*_rolling_mean_3` column for the *first* row of each `(Agency, SKU)` series must be NaN.
- **Methodology audit deferred — user override of B7's own caveat.** B7 was added to the backlog with the explicit note "only worth running once the methodology audit clears". The user has chosen to run it before the audit. Recorded here so the result is interpreted with that knowledge: if the headline RMSE drops further, part of the gain may be due to leakage we haven't yet ruled out; the methodology audit (currently the strongest open thread) becomes even more pressing afterwards, regardless of the outcome.
- **Possible feature redundancy with `02`'s `Volume_lag_*`.** If demand at month T is well predicted from `Volume_lag_1` plus current-month price/promo/weather (already in the panel), the *lagged* price/promo/weather may add little marginal information. A flat result here is genuinely informative — it'd say "the autoregressive signal already absorbs most of the exogenous dynamics".
- **Cold-start NaN regime unchanged.** `tabular_pipeline("regressor")` still tails into `HistGradientBoostingRegressor`, which handles NaN natively. New cold-start nulls in the side-table lag columns are treated the same way as in `02`.
- **R² could *decrease slightly* via mild over-fitting noise.** With 40 features instead of 28 on ~21 k rows, the tree has more knobs to chase noise. The fold std (currently ~22 % of the mean) is the right place to look — if it widens noticeably, that's the over-fit signal.

## Status

- **State:** done
- **Approved by user on:** 2026-05-02
- **Headline result:** R² 0.958 ± 0.023, RMSE 542 ± 122 hL, MAE 229 ± 47 hL (16-fold walk-forward, same splitter). Versus `02_lag_features`: ΔRMSE +22 hL (+4 %), ΔMAE +8 hL (+4 %), ΔR² −0.004, fold std *wider* (17 % → 23 % of mean). MAPE still ~4e15.
- **Implication for next iteration:** **null/negative result** — within-series memory of exogenous columns (Price/Sales/Promotions/Avg_Max_Temp/Industry_Volume/Soda_Volume) adds nothing on top of the autoregressive `Volume_lag_*` features and the current-month side-table values that `01_baseline` already had. The slight regression and widened fold std are the standard signature of feature redundancy + mild over-fit (40 features vs. 28 on 21 k rows). Operational consequence: future experiments should branch from `02_lag_features`'s pipeline state, not `03`'s — a one-line revert of `pipeline.py` is the obvious cleanup if the user chooses. Strategically, three feature-engineering threads have now been tried (`02` autoregressive + `03` side-table lags, plus the original side-table values in `01`); the highest-value next moves are *not* more feature engineering — the methodology audit (still open) and the diagnostic dispatch (per-fold-month residual breakdown, MAPE artifact) are the right next directions.
