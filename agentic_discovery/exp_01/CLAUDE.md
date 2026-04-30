# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An "agentic discovery" experiment (`exp_01`) where Claude Code itself is the modeller. The task is the ENS Challenge Data #9: predicting normalized option prices for an exotic callable debt instrument from 23 normalized parameters (3 underlyings, drifts, vols, correlations, barriers, coupons, PDI option terms, maturity, reset dates). Metric: squared error. Training set is 1M rows; test set is 400k rows. See `data/README.md` for the full data dictionary.

The interesting property of this workspace is that **Claude's own activity is the experiment**: an MLflow `Stop` hook (`.claude/settings.json`) records every Claude turn, and the commit history (`original model` → `add derivative feature` → `add feature related to spots` → `add debug to check size model` → `iter`) is the trace of an agent iterating on a model. None of that modelling code is checked in — only `pixi.toml`, `pixi.lock`, and gitattributes/gitignore are tracked. `data/`, `skore-artifacts/`, `.claude/`, and `*.db` are deliberately gitignored, so the working tree at any moment is the agent's scratch space, not a deliverable.

When asked to "work on the model" or similar, expect to be (re)building Python code from scratch in the working tree — there is no existing module layout to extend.

## Environment & commands

Pixi manages environments (osx-arm64, conda-forge). The base environment has no dependencies declared yet — add ML libraries to `[dependencies]` in `pixi.toml` as the work requires them, then `pixi install`.

The `tracing` feature wires Claude → MLflow:

```
pixi run -e tracing mlflow-init      # enable Claude autolog → sqlite:///mlflow.db (one-time)
pixi run -e tracing mlflow-status    # check whether autolog is on
pixi run -e tracing mlflow-disable   # turn it off
pixi run -e tracing mlflow-server    # launch UI against the local sqlite store
```

The `Stop` hook in `.claude/settings.json` runs `pixi run -e tracing mlflow autolog claude stop-hook` after every turn, and `MLFLOW_CLAUDE_TRACING_ENABLED=true` / `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` are set as env vars. Do not disable this casually — losing the trace defeats the purpose of the experiment.

## Data

CSVs live in `data/` and are not committed (large; gitignored):
- `training_input_*.csv` (~273 MB, 1M rows, 24 cols incl. ID)
- `training_output_*.csv` (~19 MB, ID + Target)
- `test_input_*.csv` (~110 MB, 400k rows)

All inputs and the target are normalized to [0, 1]; most prices cluster near 0.5. Column meanings (S1–S3, mu*, sigma*, rho*, Bonus, YetiBarrier/Coupon, PhoenixBarrier/Coupon, PDIBarrier/Gearing/Strike/Type, Maturity, NbDates) are documented in `data/README.md` and matter for any feature engineering.
