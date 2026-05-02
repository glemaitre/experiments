# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An ML experimentation workspace for **short-horizon (≤24h) forecasting of French
total electricity load**. The data is already on disk; the pipeline, environment,
and experiments are not scaffolded yet.

## Data assets (`data/`)

All sources cover **2021-03-23 → 2025-05-31** in UTC.

- `Total Load - Day Ahead _ Actual_*.csv` — five yearly files (2021–2025) of
  hourly French load, manually exported from the ENTSO-E transparency portal.
  This is the **prediction target**.
- `weather_<city>.parquet` — hourly historical weather for 10 French urban areas
  (Bayonne, Brest, Lille, Limoges, Lyon, Marseille, Nantes, Paris, Strasbourg,
  Toulouse), fetched from the Open-Meteo historical-forecast API.
- See `data/README.md` for retrieval details.

Forecasting framing baked into the data layout:

- Weather and calendar/holiday features are treated as **known at prediction
  time** for the full 24h horizon → use them as **future covariates**.
- Past load values are used to engineer **past covariates** (lags, rolling
  aggregations); future load values are the targets.

## Workspace state

The repo currently contains only `data/` and `.claude/`. There is no Python
package, no `pixi.toml` / `pyproject.toml`, no `experiments/`, no `plan/`.

`.claude/settings.local.json` allow-lists commands hinting at the intended
conventions (`pixi run`, `PYTHONPATH=src`, `experiments/NN_*.py`), but **none of
those files exist on disk yet** — do not assume them. When scaffolding, route
through the relevant skills:

- `organize-ml-workspace` → workspace layout and the per-experiment file rule
- `python-env-manager` → environment manager (default: pixi)
- `data-science-python-stack` → library choices per tier
- `iterate-ml-experiment` → `plan/PLAN.md` + per-experiment `plan/NN_*.md`
- `build-ml-pipeline`, `evaluate-ml-pipeline` → pipeline declaration & evaluation

## Scope rule

Stay inside `exp_04/`. Do **not** read sibling `exp_NN/` folders to infer
conventions — bootstrap fresh and ask the user when a choice is needed.
