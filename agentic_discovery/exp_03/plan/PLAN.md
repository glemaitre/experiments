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

- **Project / dataset:** `beeristan` — Stallion & Co. monthly beer demand at Agency-SKU level (Jan 2013 → Dec 2017), with side tables for price/promo, weather, events, demographics, and industry volumes. Test set is a single held-out month (Jan 2018).
- **Goal:** minimize **RMSE on hectoliters** for one-step-ahead monthly demand forecasts at Agency-SKU level on Jan'18, evaluated with a **time-ordered splitter** that mirrors the train→test cut.
- **Last experiment:** `02_lag_features` — approved
- **Last result:** `01_baseline` — R² 0.934 ± 0.021, RMSE 695 ± 130 hL (16-fold walk-forward)

<!--
Secondary task carried in Backlog (not the primary goal): SKU
recommendation for two cold-start agencies (Agency06, Agency14)
from `data/test_*/sku_recommendation.csv`. Different problem
class (ranking under cold start), parked for a later iteration.
-->

## History

| Stem | Intent (one line) | Status | Headline result | Plan file |
|---|---|---|---|---|
| `01_baseline` | skrub `tabular_pipeline` on the joined Agency-SKU-month table with a time-aware splitter | done | R² 0.934 ± 0.021, RMSE 695 ± 130 hL | [plan](01_baseline.md) |
| `02_lag_features` | add within-series lag/rolling/YoY features at Agency-SKU level (promoted from B2) | approved | n/a | [plan](02_lag_features.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | SKU recommendation for new agencies (Agency06, Agency14) — cold-start ranking from `sku_recommendation.csv`. Different problem class; parked. | dataset README |
| B3 | Calendar / event encoding from `event_calendar.csv` (sports, carnivals) joined on month; richer than month-of-year alone. | dataset structure |
| B4 | Hierarchical reconciliation: forecasts at Agency × SKU should sum coherently to per-Agency and per-SKU totals. | forecasting practice |
| B5 | MAPE blew up to ~1e16 in `01_baseline` — investigate zero / near-zero `Volume` rows (legitimate seasonal stockouts vs. data artifacts) and choose a more robust relative metric (sMAPE, WAPE) if relative error is wanted. | diagnostic from `01_baseline` |
| B6 | Fold-to-fold RMSE std is ~19 % of the mean — partly the season effect (Dec vs. summer beer volumes), partly the growing window. Worth confirming via per-fold-month decomposition once the methodology audit is in. | diagnostic from `01_baseline` |
