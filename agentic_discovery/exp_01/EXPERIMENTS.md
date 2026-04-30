# EXPERIMENTS log

A human-readable timeline of model iterations. Each entry pairs with one or
more runs in the `option-pricing` MLflow experiment (`sqlite:///mlflow.db`).
The ordering is chronological; never edit prior entries — append a new one
when an idea is revisited.

## Conventions

- **Validation**: 80/20 random split of the 1M training rows, `seed=0`. Hold
  this fixed across runs so MSE numbers are directly comparable.
- **Metric**: total squared error is what the leaderboard scores; for
  comparison we report MSE on the 200k validation rows. RMSE/MAE/R² are
  logged as orientation.
- **Provenance**: every run is tagged with `git.sha` / `git.dirty` and the
  experiment script filename. Commit *before* running so SHA points at the
  exact code used.
- **Reproducibility**: any new random source gets a fixed seed, recorded as
  an MLflow param.

## Useful commands

```
pixi run -e ml prepare-data                          # one-time CSV → parquet
pixi run -e ml python experiments/000_baseline_constant.py
pixi run -e ml python experiments/001_lightgbm_default.py
pixi run -e ml mlflow-ui                             # http://127.0.0.1:5001
```

## Log

(Append entries below. Format: `### NNN — short title`, then a paragraph on
the idea, the result, and the takeaway pointing to the next idea.)

### 000 — constant-mean sanity floor

Predict `mean(y_train) = 0.466816` for every validation row. Establishes the
floor any real model must clear.

| metric | value |
|---|---|
| val.mse | 1.280e-3 |
| val.rmse | 3.577e-2 |
| val.mae | 2.367e-2 |
| val.r2 | ~0 |
| val.squared_error_sum | 255.92 |

Takeaway: targets are very tight (std 0.036), so a ~30× RMSE reduction is
already easy; the real game is the last decimal places.

### 001 — LightGBM defaults, no feature engineering

23 raw features → LGBMRegressor, MSE objective, lr=0.05, num_leaves=63,
num_boost_round=5000 with 100-round early stopping on the val fold.

| metric | value |
|---|---|
| val.mse | 1.27e-5 |
| val.rmse | 3.564e-3 |
| val.mae | 2.119e-3 |
| val.r2 | 0.990 |
| val.squared_error_sum | 2.54 |

~100× improvement over the constant baseline. **Model did not converge:**
`best_iteration=5000` hit the cap with val RMSE still falling (0.00437 →
0.00356 over rounds 2000→5000). Two clear next moves:

1. let it run longer (≥20k rounds at lr=0.05, or lower lr with more rounds)
2. add features motivated by the option-pricing structure
   (worst-of-three, basket vol from the correlation matrix, moneyness vs
   each barrier, drift × maturity, etc.)

### 002 — LightGBM + engineered domain features

Same LightGBM hyper-params and budget as 001; added 22 engineered features
in `src/exp_01/features.py`: order-statistics over the three underlyings
(min/max/mean/range of S, mu, sigma), mean pairwise correlation,
equal-weight basket variance from the full Σ matrix, time-scaled drift /
vol (`mu*T`, `vol*sqrt(T)`), signed distances of the worst-of underlying to
each barrier (Yeti / Phoenix / PDI / Strike), a PDI pay-off proxy
incorporating gearing and put/call sign, and reset frequency `NbDates/T`.

| metric | 001 (raw23) | 002 (raw23+engineered) | Δ |
|---|---|---|---|
| val.mse | 1.27e-5 | 9.47e-6 | -25% |
| val.rmse | 3.564e-3 | 3.077e-3 | -14% |
| val.mae | 2.119e-3 | 1.825e-3 | -14% |
| val.r2 | 0.990 | 0.993 | +0.003 |
| val.squared_error_sum | 2.540 | 1.894 | -25% |

Same convergence symptom: `best_iteration=4998/5000`, val RMSE still
dropping ~0.00002 / round at the cap. Next iteration: extend the training
budget so the model actually early-stops, before tuning anything else.

