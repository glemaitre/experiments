# %% [markdown]
# # Experiment 01: baseline — direct h=24 French electricity load forecast
#
# **Date:** 2026-05-02
# **Goal:** Establish a starting metric for 24-hour-ahead French load
# forecasting using a generic tabular-regression baseline (skrub
# `tabular_pipeline`) over load lags + cross-city weather mean +
# Europe/Paris calendar features.
# **Result:** filled in after the run.
#
# See `plan/01_baseline.md` for the full design contract.

# %%
import skore

from fr_load_forecast import PROJECT_ROOT
from fr_load_forecast.evaluate import splitter
from fr_load_forecast.pipeline import build_learner

# %% [markdown]
# ## Paths
#
# `PROJECT_ROOT` resolves from the package's `__file__` and is therefore
# CWD-independent. The same absolute `DATA_DIR` is passed both as the
# pipeline preview and via `data=` to `skore.evaluate`, so
# `learner.skb.preview()` and the actual fit see the same binding.

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project
#
# One project per workspace; each experiment writes its report under a
# stable key (the file stem). Parameters per `organize-ml-workspace`:
#
# - `workspace="reports"` — folder holding the Project store.
# - `name="fr-load-forecast"` — kebab-case derived from the package
#   name; reused across all experiments in this workspace.
# - `mode="local"` — local file-system store.

# %%
project = skore.Project(workspace="reports", name="fr-load-forecast", mode="local")

# %% [markdown]
# ## Learner
#
# The pipeline binds `skrub.var("data_dir", ...)` as the source identifier
# and loads + feature-engineers inside the graph. The `data_dir_preview`
# keyword sets what `learner.skb.preview()` would resolve against; at fit
# / evaluate time the binding is overridden by the env-dict below. The
# splitter reads the `datetime` column from the materialized X.

# %%
learner = build_learner(data_dir_preview=DATA_DIR)

# %% [markdown]
# ## Evaluate
#
# `skore.evaluate` is the entry point per `evaluate-ml-pipeline`. The
# env-dict-style `data=` form is the right one for a `SkrubLearner`
# (its `fit` takes a single mapping, not `(X, y)`). Cross-validator and
# any metric overrides are imported from `fr_load_forecast.evaluate` —
# the experiment script does not redefine them.

# %%
report = skore.evaluate(
    learner,
    data={"data_dir": str(DATA_DIR)},
    splitter=splitter,
)
report

# %% [markdown]
# ## Persist
#
# Key = file stem. Reusing this key in a future run overwrites the stored
# report — fork into a new experiment file if you want both kept.

# %%
project.put("01_baseline", report)
