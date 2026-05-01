---
name: data-science-python-stack
description: >
  Opinionated Python stack for data-science / ML work — one library per
  job. SKILL.md is the index; per-library `references/<library>.md`
  files carry scope, "pick this when" / "pick something else when", and
  pairings.

  TRIGGER when: starting a new Python data-science / ML project; about
  to add or recommend a library in this space (tabular data, numerical
  arrays, classical ML, deep learning, visualization, experiment
  tracking, notebooks, testing, lint/format); the user asks which
  library to use for such a task; the user or current code reaches for
  a substitute outside the stack (xgboost, lightgbm, black, isort,
  flake8, poetry, hatch) — surface the tradeoff against the stack pick
  before going along with it.

  SKIP when: the project is non-Python; the work is web / backend /
  infra unrelated to data science; the library is already chosen and
  the task is implementation inside it (bug fix, feature work, refactor)
  with no new dependency in play.

  HOW TO USE: match the task to a category in SKILL.md's index, then
  **read the linked `references/<library>.md` before installing or
  recommending** — the one-line index entry is not enough to defend the
  choice or know what to pair it with. Don't silently substitute one
  library for another; if no entry fits, surface the gap to the user.
---

# Data Science Python Stack

Opinionated stack — one library per job. Pick the entry that matches the
task and install that library.

## How to use this skill

1. Match the task to an entry below.
2. Read the linked `references/<library>.md` for that library's scope and
   tradeoffs before introducing it.
3. Install via `pixi` by default. If the project already uses a different
   environment manager (pip+venv, uv, conda), follow that instead.
4. Don't substitute libraries silently. If no entry fits the task, surface
   the tradeoff to the user.

## The stack

### Data

For tabular data, **ask the user** at project start whether to use `pandas`
or `polars`. Don't pick silently.

- [`numpy`](references/numpy.md) — N-d arrays, numerical primitives,
  anything that isn't tabular.
- [`scipy`](references/scipy.md) — scientific computing on top of numpy
  (stats, optimize, sparse, signal). Supports the array API.
- [`pandas`](references/pandas.md) / [`polars`](references/polars.md) —
  tabular data. Choice deferred to user (see note above).
- [`pyarrow`](references/pyarrow.md) — pandas's Parquet engine and
  Arrow-backed dtype backend; rarely used directly.

### Machine learning

- [`scikit-learn`](references/scikit-learn.md) — tabular ML: basic
  preprocessing, algorithms, evaluation helpers, model selection. Use
  `HistGradientBoosting{Classifier,Regressor}` instead of pulling in
  xgboost or lightgbm.
- [`skrub`](references/skrub.md) — wrap custom dataframe operations in a
  sklearn-compatible computation graph that replays deterministically
  across train and test splits. Use for the data-cleaning layer that sits
  before the sklearn pipeline.

### Deep learning

For NLP, computer vision, or any task where deep learning is the right
tool, scikit-learn is not the right hammer. Reach for:

- [`pytorch`](references/pytorch.md) — tensor library with GPU / MPS
  support and autograd. Default deep-learning framework. Also the GPU
  alternative to numpy for raw numerical work.
- [`keras`](references/keras.md) — high-level, layer-oriented deep
  learning API. Multi-backend (runs on pytorch, TensorFlow, or JAX).
- [`skorch`](references/skorch.md) — wraps a PyTorch `nn.Module` so it
  behaves like a sklearn estimator (`fit` / `predict`, GridSearchCV,
  pipelines). Bridge between deep models and the sklearn API.

### Visualization

For visualization, **ask the user** at project start whether they need
static plots (reports, publications) or interactive plots (dashboards,
exploratory notebooks). Don't pick silently.

- [`matplotlib`](references/matplotlib.md) — static plotting foundation;
  full control over appearance.
- [`seaborn`](references/seaborn.md) — static statistical plots
  (distributions, regression, faceting). Built on `matplotlib`.
- [`plotly`](references/plotly.md) — interactive plots (hover, zoom, pan);
  browser-based, suited for dashboards and exploratory notebooks.

### Experiment tracking

- [`mlflow`](references/mlflow.md) — track runs, params, metrics, and
  artifacts; model registry. Useful once experimentation scales beyond
  notebook stdout.

### Notebooks & tooling

For notebook-based work, prefer Python files with `# %%` cell markers
(jupytext percent format) over `.ipynb` files. Python files are diffable
and version-control friendly; jupytext handles the conversion to/from
notebook format when needed.

- [`jupyterlab`](references/jupyterlab.md) — browser-based notebook IDE;
  edits and runs notebooks (or jupytext-paired `.py` files).
- [`ipykernel`](references/ipykernel.md) — Python kernel for Jupyter;
  transitive dependency, rarely called directly.
- [`jupytext`](references/jupytext.md) — sync `.ipynb` ↔ `.py` (`# %%`
  markers) so the notebook source-of-truth stays version-control
  friendly.
- [`pytest`](references/pytest.md) — testing.
- [`ruff`](references/ruff.md) — lint + format. Replaces black, isort,
  flake8.

## Conventions

- **Environment manager:** default to `pixi`. If the project already uses
  a different manager, follow that instead.
- **Versions:** don't pin unless the user asks or there's a known
  incompatibility.
- **One tool per job:** don't introduce a second library for a task
  already covered without explicit user request.
- **Line width:** wrap text at 88 chars where natural. Don't compress
  content to fit; long inline links and code spans are fine to leave on
  longer lines.
