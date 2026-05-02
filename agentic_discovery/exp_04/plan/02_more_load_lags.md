# 02_more_load_lags

<!--
Design note for `experiments/02_more_load_lags.py`. Same stem,
one-to-one with the script. Owner: `iterate-ml-experiment`. Frozen at
`approved` except for the Status block.
-->

## Question / hypothesis

What is the marginal lift on h=24 French electricity load forecasting from a richer set of past load covariates — additional load lags (`−2h`, `−48h`) and 24h / 168h backward rolling statistics (mean, std) — over the baseline's 4 past covariates (current load + lags at `−1h` / `−24h` / `−168h`)?

## Motivation

- **Sourcing strategy:** backlog:B5
- **Source(s):**
  - `PLAN.md` Backlog row B5, originating from the "minimal past-covariate set in `01_baseline`" finding.
  - User raised a related methodology question (lag-before-`mark_as_X` leakage) which was confirmed safe for stateless backward shifts; the same argument extends to backward rolling reductions, so this experiment can keep the loader-side feature-engineering pattern unchanged.
- **Why this matters:** Past load history is the strongest signal for short-horizon load forecasting. The baseline's 4-value set captures the recent tick, the daily seasonality, and the weekly seasonality, but misses (a) sub-daily momentum (`−2h`, `−48h` are standard second-tier lags), and (b) recent volatility and level (rolling means + stds over 24h and 168h windows). A clean, isolated feature-engineering experiment: same model, same splitter, same calendar / weather, only the past-covariate set changes — so the comparison to `01_baseline` is direct.

## Method

- **Files touched:** `src/fr_load_forecast/features.py` (extend `add_load_lags` and add a new `add_load_rollings` helper), `src/fr_load_forecast/data.py` (call the new helper). `pipeline.py` gains a small parameter passthrough (`lags_hours=`, `rolling_windows_hours=`) so each experiment can request its own past-covariate set without rewriting the graph; baseline defaults are preserved so `01_baseline.py` keeps producing baseline metrics on re-run. `evaluate.py` unchanged.
- **Change versus `01_baseline`:** extend the past-covariate set with:
  - **Additional load lags:** `load_lag_2h`, `load_lag_48h` — full lag set becomes `{t, t−1h, t−2h, t−24h, t−48h, t−168h}` (6 values vs. 4).
  - **24h backward rolling statistics** over `[t − 24h, t − 1h]` (exclusive of `t`, to avoid trivial overlap with `load_t`): `load_roll24_mean`, `load_roll24_std`.
  - **168h backward rolling statistics** over `[t − 168h, t − 1h]`: `load_roll168_mean`, `load_roll168_std`.
- **Stateless argument:** all new features are row-wise shifts (`shift`) or row-wise reductions over a strictly *backward* window (`shift(1).rolling_mean(N)`). They produce the same value for a given row whether computed on the train subset alone or on the full frame, so they are safe to apply inside the loader (before `mark_as_X`) — same argument as the baseline's existing lags.
- **Same learner** (`skrub.tabular_pipeline("regressor")`), **same splitter** (`DatetimeAnchoredWalkForward(initial=365d, test=183d, step=183d, expanding)`), **same metrics** (skore regression defaults), **same data range** (2021-03-22 → 2025-05-30, 36k rows after warmup / target-lookahead drop).
- **Comparison shape:** headline RMSE / MAE / MAPE / R² mean ± std across the same 6 walk-forward folds; per-fold table for direct row-by-row comparison against `01_baseline`. Skore key: `02_more_load_lags`.
- **Out of scope for this experiment:** changing the model, the splitter, the weather aggregation, the calendar features, the data range, the embargo (B6).

## Risks / things that could invalidate the result

- **Correlated features.** `load_lag_1h` and `load_lag_2h` are nearly redundant; rolling means partially overlap with their constituent lags. HistGradientBoosting is generally insensitive to correlation, but feature importances may distribute across correlated features in misleading ways. We're optimizing predictive lift, not interpretation.
- **Warmup is dominated by the 168h lag.** The new 168h rolling window costs no additional dropped rows beyond what the existing `load_lag_168h` already drops.
- **Same 24h target overlap as `01_baseline` (B6).** Both experiments share the same splitter, so absolute metrics carry the same small optimistic bias and the relative comparison stays fair.
- **Null result is informative.** If lift is small or absent, the implication is that the baseline's past-covariate set was already saturated for h=24 forecasting — which redirects the next iteration toward structural items in the backlog (per-city weather, richer calendar, sliding window) rather than further past-covariate engineering.
- **Rolling-window edge across DST.** Polars' rolling on `Datetime[us, UTC]` is purely numerical (UTC has no DST), so the 24h / 168h windows always span a fixed number of clock hours regardless of the local-time DST state. No correction needed.

## Status

- **State:** done
- **Approved by user on:** 2026-05-02
- **Headline result:** RMSE 1633.80 ± 382.52 MW · MAE 1213.27 ± 327.65 MW · MAPE 2.47% ± 0.50% · R² 0.956 ± 0.014 (6 walk-forward folds, key `02_more_load_lags`). Δ vs `01_baseline` is essentially noise: RMSE −1.1 MW (−0.07%); cross-fold std tightened slightly (411 → 382). Per-fold Δ ranges from −111 MW (fold 2, summer 2023) to +104 MW (fold 0, summer 2022); the dominant fold-1 winter spike (RMSE 2256 vs 2283) was barely affected.
- **Implication for next iteration:** **Past-covariate engineering is saturated for h=24 forecasting.** Adding 2 lags + 24h/168h rolling mean+std moves the headline RMSE by 1 MW out of 1635. Future iterations should attack a **structural** lever — per-city weather (B3), richer calendar (B4), sliding-window training (B2), embargo (B6), the fold-1 diagnostic, or **the multi-horizon framing the user has now requested** as the seed for `03_*`.
