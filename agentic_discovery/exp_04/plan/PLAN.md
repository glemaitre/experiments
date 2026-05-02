# PLAN

<!--
This file is the durable index of every experiment in this workspace.
Three sections, in order: Status, History, Backlog. Don't add new
top-level sections; they break the contract that lets future sessions
read this file in two seconds.

Owner: `iterate-ml-experiment` skill. Pair each `plan/NN_short_name.md`
with `experiments/NN_short_name.py` (identical stems).
-->

## Status

- **Project / dataset:** French electricity load forecasting — hourly ENTSO-E load + hourly Open-Meteo weather for 10 French cities + calendar / holidays. Range: 2021-03-23 → 2025-05-31.
- **Goal:** Minimize hourly French electricity load forecast error at a 24-hour-ahead horizon, evaluated via time-series cross-validation.
- **Last experiment:** `04_multi_output` — done
- **Last result:** Per-horizon RMSE curve 843 (h=1) → 2547 (h=15) → 1971 (h=24). Sensible shape, but at h=24 underperforms `01_baseline` (1971 vs 1635) — shared feature vector loses the per-horizon weather signal.

## History

| Stem | Intent (one line) | Status | Headline result | Plan file |
|---|---|---|---|---|
| `01_baseline` | direct h=24 forecast with skrub `tabular_pipeline` + custom datetime-anchored walk-forward CV | done | RMSE 1635 ± 412 MW · R² 0.956 (6 folds) | [plan](01_baseline.md) |
| `02_more_load_lags` | extend past covariates: lags at −2h / −48h + 24h / 168h backward rolling mean & std | done | RMSE 1633.8 ± 382.5 MW · R² 0.956 (Δ vs baseline: −1 MW, flat) | [plan](02_more_load_lags.md) |
| `03_horizon_as_feature` | multi-horizon — single model, rows replicated 24× with horizon as numeric feature, weather/calendar at t+h | done | RMSE 2426 ± 746 MW (per-horizon flat ~2400; h=24 slice 2356 vs baseline 1635 — model collapses to mean) | [plan](03_horizon_as_feature.md) |
| `04_multi_output` | multi-horizon — `MultiOutputRegressor(HGB)` with single feature vector + 24 outputs; weather mean over [t+1, t+24] | done | per-horizon curve 843 (h=1) → 2547 (h=15) → 1971 (h=24); h=24 underperforms baseline by ~330 MW | [plan](04_multi_output.md) |

## Backlog

<!--
Indexed table of ideas not yet committed to a `plan/NN_*.md` file.
Each row carries a stable `B<N>` index so the user can pick by
number when picking the next experiment ("go with B2").
-->

| # | Item | Source |
|---|---|---|
| B1 | Re-introduce `soil_moisture_1_to_3cm` (or impute pre-2022 history) and ablate against the soil-moisture-free baseline | data-gap surfaced during `01_baseline` implementation |
| B2 | Sliding (fixed-size) training window vs the current expanding window — may dampen pre-crisis training rows that are no longer representative | per-fold variance in `01_baseline` (fold 1 winter spike) |
| B3 | Per-city or population-weighted weather features instead of the simple cross-city mean | known-lossy baseline aggregation in `01_baseline` |
| B4 | Richer calendar features: school-holiday and vacation-period flags, long-weekend indicator, DST transition flag | shallow holiday encoding flagged at `01_baseline` planning |
| B6 | Methodology re-run: add a 24h embargo between train and test in the walk-forward splitter (current setup has the last 24h of train labels falling inside the test period — values, not labels, but worth quantifying the bias) | user-raised methodology question on `01_baseline` |
| B7 | One-hot encode the `horizon` feature in horizon-as-feature framing (vs. current numeric encoding) | flat per-horizon RMSE in `03_horizon_as_feature` — model can't differentiate horizons with a single numeric feature |
| B8 | Higher-capacity HGB (more iterations, deeper trees) for horizon-as-feature framing | follow-up to `03_horizon_as_feature` collapse: default HGB lacks capacity to learn h-conditional splits |
| B9 | **24 independent direct models** — one HGB per horizon `h`, each with weather/calendar at `t + h` (like `01_baseline` parameterized over `h`). Recovers the per-horizon-feature alignment that `04_multi_output`'s single feature vector loses, while still producing the full multi-horizon prediction. | `04_multi_output` underperforms `01_baseline` at h=24 by ~330 MW |
| B10 | `RegressorChain(HGB)` instead of `MultiOutputRegressor(HGB)` for multi-output framing — injects cross-horizon coupling (each output sees the previous outputs' predictions) at the cost of sequential sub-fits. | `04_multi_output` outputs are fit independently; chaining might capture the natural sequential structure of a load curve |
