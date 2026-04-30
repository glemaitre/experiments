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

### 003 — same model + features, 20 000 rounds (patience 200)

Same config as 002, just `NUM_BOOST_ROUND=20000`. Fit time 658 s (~11 min).

| metric | 002 (5k cap) | 003 (20k cap) | Δ |
|---|---|---|---|
| val.mse | 9.47e-6 | 7.33e-6 | -23% |
| val.rmse | 3.077e-3 | 2.708e-3 | -12% |
| val.mae | 1.825e-3 | 1.544e-3 | -15% |
| val.r2 | 0.993 | 0.994 | +0.001 |
| val.squared_error_sum | 1.894 | 1.466 | -23% |
| best_iteration | 4998 | 20000 | hit cap again |

The loss curve is asymptoting (val RMSE drops only ~3e-6 per 100 rounds in
the last 1k rounds) but never triggers early stop with `lr=0.05`. Two
hypotheses:

- **Under-capacity**: with only 45 features but complex barrier interactions,
  `num_leaves=63` may bottleneck the model — try 127 / 255.
- **Lr/budget tradeoff**: lower `lr` with more rounds typically generalizes
  better; but compute per round is constant, so this only pays if num_leaves
  isn't the bottleneck.

The 004 sweep tests both at a shorter budget (12k rounds) to rank the ideas
quickly; the winner gets extended.

### 004 — LightGBM hyper-param sweep (12k-round budget)

Four hypothesis-driven configs, all on raw23+engineered features, single
80/20 holdout (seed=0). All four hit the 12k-round cap.

| config | val RMSE | val MSE | best_iter | fit s |
|---|---|---|---|---|
| 003 reference (lr=0.05, leaves=63, 20k rounds) | **0.002708** | 7.33e-6 | 20000 | 658 |
| lower-lr-more-rounds (lr=0.025) | 0.002830 | 8.01e-6 | 12000 | 416 |
| stronger-reg (min_child=500, ff/bf=0.8) | 0.002917 | 8.51e-6 | 12000 | 460 |
| more-leaves (leaves=127) | 0.002935 | 8.61e-6 | 12000 | 611 |
| more-leaves+stronger-reg | 0.003029 | 9.18e-6 | 12000 | 666 |

Three signals:

1. **Not capacity-limited.** `num_leaves=127` (with or without matching
   regularization) was *worse* than the leaves=63 reference at the same
   budget. The leaf-wise grow policy already exploits the structure;
   doubling capacity dilutes gradient updates per leaf.
2. **Not overfitting.** Stronger regularization (bigger `min_child_samples`,
   lower feature/bagging fractions) hurt val RMSE — we are still in the
   underfitting regime.
3. **Lower LR not yet paying off.** At 12k rounds, lr=0.025 has done
   roughly half the effective training of the lr=0.05 reference at 6k
   rounds (~0.00300 RMSE) — about the same. Lower LR usually only wins
   with a much larger budget.

Therefore the binding constraint is simply **training budget at the current
LR**, not model class or regularization. Next iteration: extend the winning
config (lr=0.05, leaves=63) to 40k rounds; in parallel, run 005 for an
XGBoost second opinion at the same budget as 004.

### 005 — XGBoost (hist) on engineered features [STOPPED EARLY]

**Status:** killed at iteration ~4500 of 12000. No MLflow run was created
(the script logs only at completion). Recorded here from the live stdout
trace so the data point isn't lost.

| round | LightGBM 003 (val RMSE) | XGBoost 005 (val RMSE) |
|---|---|---|
| 500 | 0.00516 | 0.00472 |
| 1000 | 0.00427 | 0.00401 |
| 2000 | 0.00364 | 0.00357 |
| 3000 | 0.00336 | 0.00342 |
| 4000 | 0.00319 | 0.00334 |
| 4500 | ~0.00313 | 0.00332 |

XGBoost descended faster initially (likely due to depth-wise growth filling
in a coarser tree faster) but LightGBM had pulled ahead by round ~3000 and
was widening the gap by 4500. **Inconclusive** — the run was cut short —
but consistent with LightGBM being the right horse to push further.

To re-run cleanly: `pixi run -e ml python experiments/005_xgboost_features.py`
(takes ~30 min at 12k-round cap; wait for completion before drawing conclusions).

### 006 — LightGBM @ 40 000 rounds [SCRIPT READY, NOT EXECUTED]

Script committed at `experiments/006_lightgbm_features_xlong.py`. Same
config as 003 with `NUM_BOOST_ROUND=40000`. Expected wall time ~22–25 min
(extrapolating from 003's 11 min for 20k rounds). Has not been launched.

---

## Where to resume

**Current best:** run `lightgbm-features-long` (003), val RMSE = 0.002708,
R² = 0.994, total squared error on 200k val rows = 1.466.

**Next moves, in priority order:**

1. **Run 006** (extended LightGBM, 40k rounds). Expected outcome: val RMSE
   in the 0.0024–0.0026 range. If the curve still has slope at 40k, queue
   80k rounds or drop lr to 0.025 with proportionally more rounds.
2. **Re-run 005** (XGBoost) to completion for a real second-opinion data
   point. Useful regardless of 006's result because it sets the GBDT-family
   ceiling.
3. **Residual analysis on 003's val predictions.** Are the ~0.05% of
   samples driving most of the squared error in specific regions of feature
   space (e.g., near barriers, deep ITM/OTM)? If so, those regions need
   targeted features or a separate model. To do this, refit 003 with
   `predict()` saved as an MLflow artifact (or just rerun and dump
   `(y_val, y_pred)` to parquet alongside the run).
4. **K-fold or alternate-seed val** to confirm 003's win isn't single-fold
   noise. With 1M rows, a 5-fold CV is ~1 hour; cheaper proxy: re-split
   with `seed=1` and `seed=2`, expect ≤2% RMSE variance.
5. **Different model class.** The README mentions an 11-layer NN as the
   organisers' baseline. A small MLP on the engineered features would
   diversify the ensemble; even if it's not better alone, the residual
   correlations are likely <1 and stacking helps.
6. **Submission pipeline.** `src/exp_01/submission.py` has the helper but
   no script wires it. To submit: refit the chosen config on **all** 1M
   rows (no holdout) for the same number of rounds the holdout fit found,
   predict on `data/test.parquet`, write via `write_submission()`.

**Process discipline that's working — keep it:**

- Each experiment script is self-contained and re-runnable.
- Every run is tagged with `git.sha`, so MLflow rows can be replayed
  exactly. Commit *before* running.
- This `EXPERIMENTS.md` is appended (never edited) per run; numbered
  scripts mirror its ordering.
- `REPORT.md` is the human-facing summary; this file is the granular log.

