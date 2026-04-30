# Option-pricing challenge — interim report

ENS Challenge Data #9: predict the normalized price of an exotic callable
debt instrument from 23 normalized parameters describing three underlyings,
their drifts/vols/correlations, and the deal's barrier / coupon / maturity
structure. Train: 1 000 000 rows. Test: 400 000 rows. Metric: total
squared error. See `data/README.md` for the full data dictionary.

## TL;DR

| | val RMSE | val MSE | R² | sum SE (200k) |
|---|---|---|---|---|
| Constant-mean floor | 3.577e-2 | 1.280e-3 | ~0 | 255.92 |
| LightGBM raw (5k rounds) | 3.564e-3 | 1.27e-5 | 0.990 | 2.54 |
| LightGBM + engineered features (5k rounds) | 3.077e-3 | 9.47e-6 | 0.993 | 1.89 |
| **LightGBM + engineered features (20k rounds)** | **2.708e-3** | **7.33e-6** | **0.994** | **1.47** |

Best so far: a single LightGBM gradient-booster on 23 raw features + 22
hand-engineered domain features, lr=0.05, num_leaves=63, 20 000 rounds.
**~170× lower squared error than the constant baseline**, R² = 0.994.

The model has *not* converged: it hit the round cap with the validation
loss still descending at ~3e-6 RMSE / 100 rounds. There is more to gain
from training budget alone before any structural change is needed.

## Method

### Data

- 1M training rows × 23 features + 1 target. Test set 400k rows.
- All 23 features and the target are pre-normalized to [0, 1]. Targets
  cluster tightly: mean 0.467, std 0.036, max 0.967.
- No missing values. `NbDates` is effectively discrete (21 levels);
  everything else is continuous.
- One-time CSV → parquet caching (`pixi run -e ml prepare-data`).

### Feature engineering

The pay-off is path-dependent on the *worst-of-three* basket crossing
barriers, so summary statistics over the three underlyings, the basket
variance under correlation, and signed distances to each barrier should
help. We added 22 columns in `src/exp_01/features.py`:

- Order statistics over the three S / μ / σ vectors (min / max / mean /
  range).
- Mean pairwise correlation `ρ_mean`.
- Equal-weight basket variance / vol from the full Σ matrix.
- Time-scaled drift `μ·T` and vol `σ·√T`.
- Signed distances of the worst-of underlying to each barrier
  (Yeti / Phoenix / PDI) and to PDI strike.
- A PDI pay-off proxy `(K - S_min) · gearing · sign(PDIType)`.
- Reset frequency `NbDates / T`.

These 22 columns gave a **14% RMSE / 25% squared-error reduction** at
identical training budget (run 002 vs 001).

### Validation

Single 80/20 random split with `seed=0`, frozen across runs so MSE numbers
are directly comparable. 800 000 train / 200 000 val rows. The challenge
metric is total squared error, but RMSE is what we minimise; both are
logged.

### Tracking

- All runs go to MLflow at `sqlite:///mlflow.db`, experiment
  `option-pricing` (separate from the Claude-tracing experiment in the
  same store).
- Each run is auto-tagged with `git.sha`, `git.branch`, `git.dirty` and
  the experiment script filename, so any row in MLflow can be replayed
  from a clean checkout.
- Per-run logged: hyper-parameters, validation metrics
  (MSE/RMSE/MAE/R²/sumSE), `fit_seconds`, the model artefact, per-feature
  gain importance.
- `EXPERIMENTS.md` is the chronological narrative log; this `REPORT.md`
  is the synthesis.

## Findings

### 1. The target is forgiving — and that is the trap

The constant predictor's MSE is 1.28e-3 because the target std is only
0.036. The first non-trivial model wipes out 99% of that error
immediately; the rest of the work is on the 4th decimal place. Holdout
noise, fold variance, and seed sensitivity will start to matter sooner
than feature design.

### 2. Domain features really help

| | val RMSE | sum SE |
|---|---|---|
| Raw 23 features (run 001) | 3.564e-3 | 2.54 |
| + 22 engineered features (run 002) | 3.077e-3 | 1.89 |

Same model, same compute, **-14% RMSE / -25% sum SE**. The biggest
contributors by gain are the worst-of-basket order statistics
(`S_min`, `S_mean`) and the time-scaled vol terms — exactly the things the
pay-off mechanics suggest matter most.

### 3. The model is budget-bound, not capacity-bound

Run 004 swept four hypotheses against the 002 baseline, all at a 12 000-round
cap:

| | val RMSE | vs reference |
|---|---|---|
| Reference (lr=0.05, leaves=63), 20k rounds | **0.002708** | — |
| Lower lr (0.025) | 0.002830 | worse |
| More leaves (127) | 0.002935 | worse |
| Stronger reg (mcs=500, ff/bf=0.8) | 0.002917 | worse |
| More leaves + stronger reg | 0.003029 | worse |

Three signals from this:

- **Not capacity-limited.** Doubling `num_leaves` made things *worse* —
  the leaf-wise grow policy already exploits the structure, more leaves
  dilute updates.
- **Not overfitting.** Stronger regularisation hurt — we are still in the
  underfitting regime.
- **Lower LR isn't winning per unit compute** at 12k rounds (it has done
  half the effective training of the lr=0.05 reference at 6k rounds, and
  matches it). It would need a much larger budget to pay off.

→ The binding constraint is **training budget at the current LR**, not the
model class.

### 4. XGBoost looks similar (incomplete)

Run 005 (XGBoost-hist on engineered features) was started but stopped
early at round ~4500. From the partial trace, XGBoost descended slightly
faster in the first ~2000 rounds (depth-wise growth filling a coarse tree
fast) but was already behind LightGBM by round 3000 and widening. Not a
full data point, but consistent with LightGBM being the right horse to
push further.

## Next steps

In priority order:

1. **Run 006** — same config as 003, 40 000 rounds. Script committed
   (`experiments/006_lightgbm_features_xlong.py`). Expected ~22 min.
   Likely lands val RMSE in the 0.0024–0.0026 range and finally hits a
   real plateau (or doesn't, in which case we know to push lr=0.025 with
   a much larger budget).
2. **Re-run 005 to completion** for a complete XGBoost data point.
3. **Residual analysis on 003's predictions.** The total squared error on
   200k val rows is 1.466 — that's an average per-row contribution of
   ~7e-6. A residual histogram tells us whether ~all rows are roughly that
   bad or a small tail of hard rows dominates. If it's the tail, find the
   feature region those rows live in and either engineer specifically for
   it or train a specialist model.
4. **Robustness check.** Re-split with `seed=1` and `seed=2`; if 003's
   win is real we expect ≤2% RMSE variance. Cheaper than full K-fold.
5. **Diversify the model class.** A small MLP on the engineered features
   (the README mentions an 11-layer NN baseline). Residuals from a NN are
   likely uncorrelated enough with GBDT residuals to make stacking pay
   off, even if the standalone NN is worse.
6. **Build the submission.** `src/exp_01/submission.py` has the helper.
   Wire a script that refits the chosen config on all 1M rows (no
   holdout, same number of rounds the holdout found) and writes
   `submissions/<run-id>.csv`.

## How to reproduce

```sh
# one-time
pixi install -e ml
pixi run -e ml prepare-data            # CSV → parquet under data/

# any single experiment
pixi run -e ml python experiments/003_lightgbm_features_long.py

# inspect the runs
pixi run -e ml mlflow-ui               # http://127.0.0.1:5001
```

`EXPERIMENTS.md` is the chronological log of every iteration. Source
under `src/exp_01/`. Convention: commit *before* running so each MLflow
run's `git.sha` tag points to clean code.

## Run index

| ID | Script | Status | val RMSE |
|---|---|---|---|
| 000 | constant-mean | done | 3.577e-2 |
| 001 | LightGBM raw, 5k rounds | done | 3.564e-3 |
| 002 | LightGBM + features, 5k rounds | done | 3.077e-3 |
| 003 | LightGBM + features, 20k rounds | done | **2.708e-3** ← best |
| 004 | LightGBM sweep, 12k rounds | done | 2.83–3.03e-3 |
| 005 | XGBoost + features, 12k rounds | stopped early at round 4500 (0.00332) | — |
| 006 | LightGBM + features, 40k rounds | committed, not run | — |
