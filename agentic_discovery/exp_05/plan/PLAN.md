# PLAN

## Status

- **Project / dataset:** France hourly electricity load (ENTSO-E, 2021-03 → 2025-05) + historical weather for 10 French cities (Open-Meteo) + calendar.
- **Goal:** Forecast the national hourly load 12 hours ahead (`load[t+12]`) from features known at time `t`. Minimize regression error (skore defaults: MSE/RMSE/MAE/R²); the day-ahead ENTSO-E forecast in the same CSV is a natural future benchmark but is **not** the target of this iteration.
- **Last experiment:** `01_baseline` — done
- **Last result:** MAE 2621 ± 685 MW (≈ 5.2 % MAPE) on 19 monthly walk-forward folds

## History

| Stem | Intent (one line) | Status | Headline result | Plan file |
|---|---|---|---|---|
| `01_baseline` | tabular_learner on calendar + national-mean weather + 1 load lag, t+12 direct forecast, monthly walk-forward CV | done | MAE 2621 ± 685 MW · MAPE 5.19 ± 1.10 % · R² 0.67 | [plan](01_baseline.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Compare the model against ENTSO-E's day-ahead forecast (column already in the CSVs) | bootstrap note |
| B2 | Per-city weather features instead of national mean (Marseille vs Lille climate differs) | bootstrap note |
| B3 | Richer load lag set (24h, 168h, rolling means) | bootstrap note |
| B4 | French public-holiday flag as a calendar feature | bootstrap note |
| B5 | Direct multi-horizon forecasting (t+1 … t+24) instead of t+12 only | bootstrap note |
