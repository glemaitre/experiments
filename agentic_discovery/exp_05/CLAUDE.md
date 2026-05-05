# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

A fresh experimentation workspace for **short-horizon (≤ 24 h) forecasting of French
electricity load**. Currently only `data/` is populated — no source, no environment,
no experiments yet. Scaffolding is the first task whenever a session starts.

## Data layout (already in place)

Everything lives under `data/`. See `data/README.md` for provenance and full notes;
the modelling-relevant facts are:

- `Total Load - Day Ahead _ Actual_YYYY*.csv` — five yearly CSVs from ENTSO-E covering
  France (BZN|FR), 2021-01 through 2025-12. Hourly actual + day-ahead forecast load.
  This is the **prediction target**.
- `weather_<city>.parquet` — historical weather (Open-Meteo) for 10 French urban areas:
  bayonne, brest, lille, limoges, lyon, marseille, nantes, paris, strasbourg, toulouse.
- Common usable time range across all sources: **2021-03-23 → 2025-05-31**.

Modelling implications baked into the problem:

- Max forecast horizon is 24 h, so weather and calendar features are treated as
  **future covariates** (assumed known at prediction time).
- Load history is used as **past covariates** (lags, rolling aggregations).
- Future load values (relative to the prediction time) are the targets.

## Locally-vendored skills own the workflow

`.claude/skills/` ships project-pinned copies of the ML workflow skills. They are
authoritative — invoke them rather than scaffolding ad-hoc:

- `organize-ml-workspace` — owns the layout (`src/<pkg>/`, `experiments/`, `plan/`)
  and the one-file-per-experiment + `# %%` script convention.
- `python-env-manager` — bootstraps the env. Default is **pixi** unless told otherwise.
- `data-science-python-stack` — picks the libraries (skrub, scikit-learn, skore,
  matplotlib, pandas/polars). Don't reach for `xgboost`, `lightgbm`, `mlflow` for
  tracking, `cross_val_score`, `black`/`isort`/`flake8`, `poetry`/`hatch`.
- `build-ml-pipeline` — pipelines are declared as **skrub DataOps graphs**;
  stateless steps via `.skb.apply_func`, stateful steps via `.skb.apply`.
- `evaluate-ml-pipeline` — evaluation routes through **skore** (`skore.evaluate`,
  `EstimatorReport`, `CrossValidationReport`), not handwritten metric prints.
- `iterate-ml-experiment` + `iterate-from-{user,diagnostic,methodology,literature}` —
  drive the iteration loop; every new experiment file requires a validated
  `plan/NN_short_name.md` first.
- `python-code-style` — ruff for lint + format, numpydoc docstrings.

There is also a stray `.claude/skills/mlflow.db` in the skills folder; mlflow, if
used at all, is for model serving / registry only — never for run tracking
(skore's Project API owns that).

## House rules from prior sessions

- **Each method gets its own experiment file.** Never bundle several modelling
  approaches under a single umbrella plan/experiment. Parametrize shared feature
  engineering in `src/<pkg>/` so older experiments stay reproducible from `main`.
- **Stay scoped to this folder.** Do not read sibling `exp_NN/` directories to
  copy conventions — bootstrap fresh from the skills above and ask the user when
  unsure.
- The user authors acceptance criteria, not Claude. Iteration skills propose;
  the user judges.
