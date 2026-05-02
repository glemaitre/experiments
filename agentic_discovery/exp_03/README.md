# exp_03

ML experimentation workspace organized around a single skore Project, a
declarative skrub/scikit-learn pipeline, and one script per experiment.

## Layout

```
exp_03/
├── pixi.toml               # environment + dependencies (scikit-learn, skrub, skore, polars, ruff)
├── ruff.toml               # lint + format config
├── src/beeristan/          # the project's package — reusable code
│   ├── __init__.py
│   ├── data.py             # data loading, splits, split_kwargs wiring
│   ├── features.py         # transformers, encoders, feature functions
│   ├── pipeline.py         # the learner declaration (skrub DataOps)
│   └── evaluate.py         # cross-validator + (optional) metric overrides
├── experiments/            # one `# %%` script per experiment
│   ├── 0x_experiment.py
├── plan/                   # iteration log + per-experiment design notes
│   ├── PLAN.md             # session-start log; index of experiments
│   ├── 0x_experiment.md
├── reports/                # skore Project store (written on first run)
└── data/                   # raw inputs (user-owned)
```

## File contracts

Each file in `src/beeristan/` has a narrow contract — respect it so
experiments compose predictably.

- **`data.py`** — loaders, materialization of `X`, `y`, and any
  `split_kwargs` (groups, time, …) attached at the X marker.
- **`features.py`** — feature functions and transformers used by the
  pipeline.
- **`pipeline.py`** — the learner declaration (typically a
  `SkrubLearner`). Returns the unfit object.
- **`evaluate.py`** — only the inputs to `skore.evaluate`: the
  cross-validator (`splitter = ...`) and optional metric overrides. It
  does **not** call `skore.evaluate`, does **not** open a
  `skore.Project`, does **not** persist anything. Those steps belong in
  the experiment script.

## Experiments

Experiments live under `experiments/` as `.py` scripts with `# %%` cell
markers (recognized by VS Code, PyCharm, and `jupytext`) — not
`.ipynb` notebooks. The script numeric prefix preserves iteration order
in `ls`, and the file's stem is reused as the report key in the skore
Project (`01_baseline.py` → `"01_baseline"`).

Every experiment script:

1. opens (or attaches to) the `skore.Project` rooted at `reports/`,
2. imports the learner from `beeristan.pipeline` and the CV from
   `beeristan.evaluate`,
3. calls `skore.evaluate(...)`,
4. calls `project.put("<experiment-key>", report)` to persist the
   report under a stable key.

## Plan

Each `experiments/NN_short_name.py` has a matching
`plan/NN_short_name.md` written and validated **before** the script is
created. `plan/PLAN.md` is the session-start log and the index of
experiments tried so far, with their outcomes.

## Reports

`reports/` holds the skore Project store. All experiments write into the
same Project (same `name`), which is what enables `ComparisonReport`
across runs.
