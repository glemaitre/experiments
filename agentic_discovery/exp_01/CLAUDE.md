# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read these first

- **`REPORT.md`** — synthesis of findings to date and the current best result.
- **`EXPERIMENTS.md`** — append-only chronological log of every iteration, with a "Where to resume" section at the bottom that names the next concrete step.

If the user asks "where are we?" or "what's the plan?", these two files answer it. Don't re-derive from MLflow or git history.

## What this repo is

An "agentic discovery" experiment (`exp_01`) where Claude Code itself is the data scientist. The task is the ENS Challenge Data #9: predicting normalized option prices for an exotic callable debt instrument from 23 normalized parameters (3 underlyings, drifts, vols, correlations, barriers, coupons, PDI option terms, maturity, reset dates). Metric: total squared error. Training set is 1M rows; test set is 400k rows. See `data/README.md` for the full data dictionary.

A separate (and orthogonal) part of the setup logs every Claude turn to MLflow via a `Stop` hook in `.claude/settings.json`. Don't change or disable that — but don't worry about it day-to-day either; it's just observability of the agent.

## Code layout

- `src/exp_01/` — installable package (editable install via `pyproject.toml`).
  - `data.py` — paths, parquet caching (`prepare`), `load_train` / `load_test`, deterministic `train_val_split` (`seed=0`, `val_size=0.2`).
  - `features.py` — `add_engineered()` and `feature_matrix()`. The engineered columns encode worst-of basket statistics, basket variance from the Σ matrix, time-scaled drift/vol, signed distances of the worst underlying to each barrier, a PDI pay-off proxy, and reset frequency.
  - `metrics.py` — `mse / rmse / mae / r2 / squared_error_sum / all_metrics`.
  - `tracking.py` — `setup()` and `run(name, ...)` context manager that opens an MLflow run, sets the experiment to `option-pricing`, and tags `git.sha` / `git.branch` / `git.dirty` / `script`.
  - `submission.py` — `write_submission(ids, preds, name=...)` → `submissions/<name>.csv`.
- `experiments/NNN_*.py` — numbered, self-contained scripts. Each opens *one* MLflow run (or several inside a sweep), logs params/metrics/model, and prints final metrics. Numbering matches `EXPERIMENTS.md` entries.

## Conventions

- **One run = one git commit.** Commit *before* running so the run's `git.sha` tag points at clean code. After every experiment, append an entry to `EXPERIMENTS.md` and commit.
- **Validation is fixed.** 80/20 random split, `seed=0`, 800k train / 200k val. Don't change this without an explicit reason — comparability across runs depends on it.
- **MLflow experiment** is `option-pricing`, store is `sqlite:///mlflow.db`. The Claude-tracing experiment lives in the same store; don't merge them.
- **Working in polars by default.** All-numeric tabular ML; we hand `.to_numpy()` to LightGBM/XGBoost. Only switch to pandas if a specific API needs DataFrame-out semantics.
- **No tests, no linter.** This is an experimental scratchpad; the artefact is the model and the report.

## Environment & commands

Pixi manages environments (osx-arm64, conda-forge). Two features:

- `tracing` — MLflow + the Claude autolog plumbing. Pre-existing.
- `ml` — the data-science stack (numpy, polars, pandas, pyarrow, scikit-learn, lightgbm, xgboost, mlflow, matplotlib, scipy, ipykernel). Activates `PYTHONUNBUFFERED=1` so live LightGBM/XGBoost progress prints under `pixi run` (which has no TTY).

```sh
pixi install -e ml                                          # one-time
pixi run -e ml prepare-data                                 # CSV → parquet
pixi run -e ml python experiments/003_lightgbm_features_long.py
pixi run -e ml mlflow-ui                                    # http://127.0.0.1:5001
```

Long runs (≥5 min) should be launched with the Bash tool's `run_in_background=true` rather than blocking the foreground — the wall time exceeds the 10-min Bash tool cap and you'll get a notification when it finishes.

## Data

CSVs live in `data/` and are not committed (large; gitignored):
- `training_input_*.csv` (~273 MB, 1M rows, 24 cols incl. ID)
- `training_output_*.csv` (~19 MB, ID + Target)
- `test_input_*.csv` (~110 MB, 400k rows)

All inputs and the target are normalized to [0, 1]; most prices cluster near 0.467 (std 0.036). `prepare-data` writes `data/train.parquet` and `data/test.parquet` for fast reload.
