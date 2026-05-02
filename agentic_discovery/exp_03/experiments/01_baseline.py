# %% [markdown]
# # Experiment: 01_baseline
#
# **Date:** 2026-05-02
# **Goal:** establish the floor RMSE for monthly Agency-SKU demand on
# the Beeristan panel using `skrub.tabular_pipeline("regressor")` on the
# raw multi-table join, evaluated walk-forward by month.
# **Result:** filled in after the run.

# %%
import skore

from beeristan.evaluate import splitter
from beeristan.pipeline import build_learner

# %% [markdown]
# ## Project
#
# One project per workspace; each experiment writes its report under the
# file's stem as a stable key.

# %%
project = skore.Project(workspace="reports", name="beeristan", mode="local")

# %% [markdown]
# ## Learner and data
#
# The data binding is `data_dir` — the loader inside the DataOps graph
# reads the seven CSVs and returns the joined panel. Swapping data
# sources later is one path change.

# %%
learner = build_learner()
data = {"data_dir": "data/train_OwBvO8W"}

# %% [markdown]
# ## Evaluate
#
# The splitter is a walk-forward cross-validator: 12 months of training
# minimum, then 3-month test folds advancing by 3 months. With 60 months
# of history (Jan 2013 → Dec 2017) that yields 16 folds.

# %%
report = skore.evaluate(learner, data=data, splitter=splitter)
report

# %% [markdown]
# ## Persist

# %%
project.put("01_baseline", report)
