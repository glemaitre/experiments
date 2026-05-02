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
- **Last experiment:** `01_baseline` — done
- **Last result:** RMSE 1635 ± 412 MW · MAPE 2.47% · R² 0.956 (6 folds; fold-1 winter spike at RMSE 2283 dominates spread)

## History

| Stem | Intent (one line) | Status | Headline result | Plan file |
|---|---|---|---|---|
| `01_baseline` | direct h=24 forecast with skrub `tabular_pipeline` + custom datetime-anchored walk-forward CV | done | RMSE 1635 ± 412 MW · R² 0.956 (6 folds) | [plan](01_baseline.md) |

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
| B5 | Additional past covariates: load lags at −2h / −48h, plus 24h / 168h rolling means and stds | minimal past-covariate set in `01_baseline` |
