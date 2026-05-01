# mlflow

Open-source platform for managing the ML lifecycle: experiment tracking,
model registry, packaging. In this stack mlflow is used primarily for
**tracking** — recording params, metrics, artifacts, and the code
version of every run.

**Pick mlflow when:**
- You're running enough experiments that comparing them across notebooks
  or terminal logs has stopped scaling.
- You need a durable record (params, metrics, plots, model artifacts)
  beyond what stdout or a notebook cell gives.
- You want a server-backed UI to compare runs across collaborators.

**You don't need mlflow for:**
- One-off scripts where the result is consumed once and discarded.
- Toy experiments early in exploration — adding mlflow is friction
  before the work scales.

**Operational notes:**
- mlflow has two halves: a **tracking server** (storage + UI) and the
  **client library** that scripts import. They can be the same process
  for local use, or split with a remote `MLFLOW_TRACKING_URI` for
  shared use.
- The user's project may have mlflow in a separate environment (e.g.
  a `tracing` env) — check before assuming it's installed alongside
  the modelling code.
