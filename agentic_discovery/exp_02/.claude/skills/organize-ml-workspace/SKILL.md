---
name: organize-ml-workspace
description: >
  Decide where files live in an ML experimentation project: where the
  reusable code goes, where each experiment goes, where reports are
  persisted. Owns the layout, the file-creation rules (one file per
  experiment, ask before editing an existing one), and the
  jupytext-style `# %%` script convention. Stops at "the file exists
  in the right place with the right skeleton". Never imposes a
  `data/` layout — the user owns that.

  TRIGGER when: starting a new ML project / scaffolding a workspace;
  about to create the first experiment file in a project; about to
  create `src/<pkg>/data.py`, `features.py`, `pipeline.py`, or
  `evaluate.py` for the first time; about to write a Jupyter
  notebook (`.ipynb`) for experimentation — redirect to a `# %%`
  script under `experiments/`; user asks where something should
  live, how to organize the project, or how to set up the workspace;
  about to add a *new experiment iteration* (must decide: new file
  vs. edit existing — ask the user).

  SKIP when: the file is clearly part of the package's existing
  module (e.g., adding a function to an already-populated
  `features.py`); pure refactor inside a single existing file;
  pipeline declaration mechanics (`build-ml-pipeline`); evaluation
  mechanics (`evaluate-ml-pipeline`); skore symbol lookup
  (`skore-api`).

  HOW TO USE: **first detect whether a workspace is already in
  place**. If yes, glue to its conventions (do not rename or move
  existing folders). If no, scaffold the default layout below
  (omitting `data/`, `models/`, `tests/`). Use the templates in
  `templates/` as the starting content — copy and adapt; do not
  rewrite from scratch. For each new experiment, default to a new
  file in `experiments/`; when the user says "iterate on X", **ask**
  whether to fork into a new file or edit in place.
---

# Organize ML Workspace

Where things live, when to create a new file, what each file is
allowed to contain. Pipeline mechanics, evaluation mechanics, and
skore/skrub/sklearn symbols are out of scope and live in the
sibling skills.

## Scope

- **In scope:** detecting an existing layout, scaffolding a fresh
  one, the `experiments/` script convention (`# %%`, one file per
  experiment), the contract for what `evaluate.py` is allowed to
  contain, the `reports/` location for the skore Project.
- **Out of scope:** what to put inside `pipeline.py` (see
  `build-ml-pipeline`), how to call `skore.evaluate` (see
  `evaluate-ml-pipeline`), skore/skrub/sklearn symbols (see the
  `*-api` skills), data ingestion paths (user-owned).

## Detection — existing workspace first

Before scaffolding anything, look at the project root and infer
whether a layout already exists:

| Signal | Meaning |
|---|---|
| `pyproject.toml` / `pixi.toml` with a project/package name | use that as the package name |
| `src/<pkg>/__init__.py` or `<pkg>/__init__.py` at root | package directory already chosen — keep it |
| `experiments/`, `notebooks/`, `scripts/`, `analyses/` | experiment location already chosen — keep it |
| `reports/`, `results/`, `runs/` | report location already chosen — keep it |
| Existing `.ipynb` files in the experiment folder | user is on notebooks; **do not silently switch to scripts** — surface the convention shift and ask |

If any of these are present, **glue to the existing convention**.
Do not rename or relocate. Add new files in the locations the
project already uses, with names that match the existing pattern.

If none of these are present, the project is fresh — scaffold the
default layout below.

## Default layout (fresh workspace)

```
project/
├── pyproject.toml          # or pixi.toml — already there in most cases
├── src/<pkg>/
│   ├── __init__.py
│   ├── data.py             # data loading, splits, split_kwargs wiring
│   ├── features.py         # transformers, encoders, feature functions
│   ├── pipeline.py         # the learner declaration (skrub DataOps)
│   └── evaluate.py         # ONLY: CV strategy + (optional) metric overrides
├── experiments/            # one `# %%` script per experiment
│   └── 01_baseline.py
└── reports/                # skore Project lives here
```

Notes on what is **deliberately absent**:

- **No `data/` directory.** The user decides where data comes from
  (local mount, remote bucket, fixture, fetched dataset). `data.py`
  exposes a loader; the path is a parameter, not a folder we
  invent.
- **No `models/`.** Persistence is out of scope at this stage.
- **No `tests/`.** Out of scope at this stage.

If the user asks for any of those later, add them — don't pre-empt.

## Files in `src/<pkg>/`

Each file has a narrow contract; respect it so experiments compose
predictably.

- **`data.py`** — loaders, the call to materialize `X`, `y`, and
  any `split_kwargs` (groups, time, …) attached at the X marker.
  Pipeline mechanics: see `build-ml-pipeline`.
- **`features.py`** — feature functions and transformers. Pipeline
  mechanics: see `build-ml-pipeline`.
- **`pipeline.py`** — the learner declaration (typically a
  `SkrubLearner`). Returns the unfit object. Pipeline mechanics:
  see `build-ml-pipeline`.
- **`evaluate.py`** — **only** the inputs to `skore.evaluate`:
  - the cross-validator (`splitter = ...`),
  - optional metric overrides if the user has explicitly asked for
    them.

  `evaluate.py` does **not** call `skore.evaluate`, does **not**
  open a `skore.Project`, does **not** persist anything. Those
  steps belong in the experiment script. See
  `evaluate-ml-pipeline` for cross-validator selection.

## Experiments — one file per experiment

Experiments live under `experiments/` as **`.py` scripts with
`# %%` cell markers**, *not* `.ipynb` notebooks. The `# %%`
convention is recognized by VS Code, PyCharm, and `jupytext`, so
the file opens as a notebook in Jupyter while staying clean under
version control.

### File-creation rule

- **New experiment → new file.** Default to creating a new file:
  `NN_short_name.py` (e.g. `02_text_encoder.py`,
  `03_grouped_cv.py`). The numeric prefix preserves the iteration
  order in `ls`.
- **Iterating on an existing experiment → ask first.** When the
  user says "let's tweak experiment 02" or "iterate on the text
  encoder run", do not assume. Ask:
  > Should this be a new experiment file (e.g.
  > `04_text_encoder_v2.py`) or an in-place edit of
  > `02_text_encoder.py`?

  In-place edits overwrite the prior result in the skore Project
  if the same key is reused — flag this if the user picks
  in-place.

### What an experiment script does

Every experiment script follows the same shape: open the
`skore.Project`, build the learner, evaluate it, store the
report. Use `templates/experiment.py` as the starting content —
copy it, rename it, adapt the imports.

The script is responsible for:

1. opening (or attaching to) the `skore.Project` rooted at
   `reports/` (see "Project parameters" below),
2. importing the learner from `<pkg>.pipeline` and the CV from
   `<pkg>.evaluate`,
3. calling `skore.evaluate(...)`,
4. calling `project.put("<experiment-key>", report)` to persist
   the report under a stable key.

Confirm exact signatures via `skore-api` before writing the call;
do not guess parameter names from memory. Cross-validator choice
is in `evaluate-ml-pipeline`.

### Project parameters

The `skore.Project` constructor takes — at minimum — three things
the experiment script must set explicitly:

| Parameter | Value to use |
|---|---|
| `workspace` | `"reports"` (the folder defined in the layout above; the Project writes its store inside it) |
| `name` | a short, stable project name **inferred from context** — see below |
| `mode` | `"local"` by default |

**Picking `name`.** Do not leave it as a placeholder. Derive it
from whatever is most identifying in the project, in this order:
1. the project / package name from `pyproject.toml` or
   `pixi.toml`;
2. the dataset name if the loader makes it obvious (e.g.
   `"adult-census"`, `"taxi-trips"`);
3. the working-directory name as a last resort.

Use kebab-case, keep it short, and **reuse the same `name` across
all experiments in the workspace** — that's what lets every
experiment's report land in the same Project for later comparison.
If the user has already opened a Project earlier in the
conversation with a different `name`, keep theirs.

**`mode="local"` is the current default.** Don't switch to other
modes (hub, mlflow) unless the user asks. Consult `skore-api` for
the supported values and the full constructor signature.

### Experiment key convention

Use the file's stem as the report key (e.g.
`01_baseline.py` → `"01_baseline"`). One file → one key → one
report. This is what makes `ComparisonReport` across experiments
trivial later.

## Decision flow

1. Read the project root. Does an ML layout already exist
   (signals above)?
   - **Yes** → glue. Add new files in the existing folders with
     names matching the existing pattern. Stop.
   - **No** → scaffold the default layout. Continue.
2. Determine the package name (from `pyproject.toml` /
   `pixi.toml` if present; otherwise ask the user).
3. Create `src/<pkg>/` with the four skeletons (use
   `templates/src_*.py`). Create empty `__init__.py`.
4. Create `experiments/` and seed it with `01_baseline.py` from
   `templates/experiment.py`.
5. Create `reports/` (empty — skore writes into it on first run).
6. Hand back to the relevant sibling skill: `build-ml-pipeline`
   for what goes inside `pipeline.py`, `evaluate-ml-pipeline` for
   what `splitter` should be in `evaluate.py`.

## Templates

- `templates/experiment.py` — the recurring artifact. Copied for
  every new experiment.
- `templates/src_data.py`, `templates/src_features.py`,
  `templates/src_pipeline.py`, `templates/src_evaluate.py` — the
  one-time skeletons for the package.

Copy, don't rewrite. The templates encode the contracts above
(especially the narrow scope of `evaluate.py`).

## Companion skills

- **`build-ml-pipeline`** — what goes inside `pipeline.py`,
  `features.py`, `data.py` (declarative side).
- **`evaluate-ml-pipeline`** — what `splitter` should be in
  `evaluate.py`, and how the experiment script calls
  `skore.evaluate`.
- **`skore-api`** — `skore.Project`, `skore.evaluate`,
  `project.put` signatures. Don't guess from memory.
- **`skrub-api`** / **`sklearn-api`** — symbols used inside the
  `src/<pkg>/` files.
