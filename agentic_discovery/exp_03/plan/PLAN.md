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
- **Last experiment:** `04_sku_recommendation` — done
- **Last result:** Agency_06 → {SKU_01, SKU_04}; Agency_14 → {SKU_01, SKU_04}. See `reports/04_sku_recommendation.csv`.

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
| `02_lag_features` | add within-series lag/rolling/YoY features at Agency-SKU level (promoted from B2) | done | R² 0.962 ± 0.017, RMSE 520 ± 112 hL (−25 % vs baseline) | [plan](02_lag_features.md) |
| `03_side_table_lags` | add lag-1 + trailing-rolling-mean-3 on six side-table columns (Price/Sales/Promotions/Avg_Max_Temp/Industry_Volume/Soda_Volume) — promoted from B7 | done | R² 0.958 ± 0.023, RMSE 542 ± 122 hL (+4 % vs `02`); null/negative | [plan](03_side_table_lags.md) |
| `04_sku_recommendation` | top-2 SKU recommendation for cold-start agencies `Agency_06`, `Agency_14` via baseline-shape learner trained on existing 350 series, predicted over 2017 — promoted from B1 | done | both → {SKU_01, SKU_04}. Demographics in-range. SKU_04 (#5 global) preferred over SKU_02 / 03 / 05 — demographics + weather doing some differentiation. | [plan](04_sku_recommendation.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B10 | Methodological validation of the cold-start recommendation: leave-one-out agency holdout where we hide one existing agency's history, run the same recommendation procedure, and check whether top-K predicted SKUs overlap with that agency's actual best-sellers. Out-of-scope from `04`. | out-of-scope spillover from `04_sku_recommendation` |
| B11 | Demographic-distribution audit: are `Agency_06`/`Agency_14`'s `Avg_Population_2017` / `Avg_Yearly_Household_Income_2017` within the train min/max? If not, recommendations are extrapolations and need a calibrated caveat. The script in `04` will print these as a sanity check; if extrapolation is detected, this becomes the next iteration. | out-of-scope spillover from `04_sku_recommendation` |
| B3 | Calendar / event encoding from `event_calendar.csv` (sports, carnivals) joined on month; richer than month-of-year alone. | dataset structure |
| B4 | Hierarchical reconciliation: forecasts at Agency × SKU should sum coherently to per-Agency and per-SKU totals. | forecasting practice |
| B5 | MAPE blew up to ~1e16 in `01_baseline` — investigate zero / near-zero `Volume` rows (legitimate seasonal stockouts vs. data artifacts) and choose a more robust relative metric (sMAPE, WAPE) if relative error is wanted. | diagnostic from `01_baseline` |
| B6 | Fold-to-fold RMSE std is ~19–22 % of the mean across `01_baseline` and `02_lag_features` — partly the season effect (Dec vs. summer beer volumes), partly the growing window. Worth confirming via per-fold-month decomposition once the methodology audit is in. | diagnostic from `01_baseline` + `02_lag_features` |
| B8 | ~~Side-table year-over-year lags (`*_lag_12`).~~ — Expected value lowered after `03_side_table_lags` showed lag-1 / rolling-3 add nothing on top of `Volume_lag_*`. Year-over-year is conceptually different (seasonality vs. one-month memory) but `Volume_lag_12` already captures that thread. Kept as a strikethrough breadcrumb; not a recommended pick. | out-of-scope spillover from `03_side_table_lags` |
| B9 | ~~Lagged event indicators.~~ — Even more speculative after `03_side_table_lags`'s null result; current-month flags are already in the panel and don't appear to be a load-bearing signal. Kept as a strikethrough breadcrumb. | out-of-scope spillover from `03_side_table_lags` |
