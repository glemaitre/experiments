# %% [markdown]
# # Experiment 02: more load lags + rolling stats
#
# **Date:** 2026-05-02
# **Goal:** Quantify the marginal lift on h=24 French electricity load
# forecasting from a richer set of past load covariates — `−2h` and
# `−48h` lags plus 24h / 168h backward rolling mean & std — over the
# `01_baseline` past-covariate set (current load + lags at `−1h` /
# `−24h` / `−168h`).
# **Result:** filled in after the run.
#
# See `plan/02_more_load_lags.md` for the full design contract.

# %%
import skore

from fr_load_forecast.evaluate import splitter
from fr_load_forecast.pipeline import build_learner

# %% [markdown]
# ## Project
#
# Same project as `01_baseline` — both reports live under the
# `fr-load-forecast` Project for direct comparison via
# `project.summarize()`.

# %%
project = skore.Project(workspace="reports", name="fr-load-forecast", mode="local")

# %% [markdown]
# ## Learner
#
# Same model (`skrub.tabular_pipeline("regressor")`), same datetime-anchored
# walk-forward splitter, same data range. Only the past-covariate set
# changes: 5 lags (`1h`, `2h`, `24h`, `48h`, `168h`) and rolling
# statistics over 24h and 168h backward windows (excluding `t`).

# %%
learner = build_learner(
    lags_hours=(1, 2, 24, 48, 168),
    rolling_windows_hours=(24, 168),
)

# %% [markdown]
# ## Evaluate

# %%
report = skore.evaluate(
    learner,
    data={"data_dir": "data"},
    splitter=splitter,
)
report

# %% [markdown]
# ## Persist
#
# Key = file stem. Stored alongside `01_baseline` for side-by-side
# comparison.

# %%
project.put("02_more_load_lags", report)
