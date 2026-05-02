# %% [markdown]
# # Experiment: 02_lag_features
#
# **Date:** 2026-05-02
# **Goal:** quantify whether explicit within-series memory at the
# Agency-SKU level (lag-1, lag-12, trailing 3/6/12-month rolling means)
# improves the headline RMSE over `01_baseline`, or whether the baseline
# was already saturating what the side-tables can buy.
# **Result:** filled in after the run.

# %%
import skore

from beeristan.evaluate import splitter
from beeristan.pipeline import build_learner

# %% [markdown]
# ## Project

# %%
project = skore.Project(workspace="reports", name="beeristan", mode="local")

# %% [markdown]
# ## Learner and data

# %%
learner = build_learner()
data = {"data_dir": "data/train_OwBvO8W"}

# %% [markdown]
# ## Evaluate

# %%
report = skore.evaluate(learner, data=data, splitter=splitter)
report

# %% [markdown]
# ## Persist

# %%
project.put("02_lag_features", report)
