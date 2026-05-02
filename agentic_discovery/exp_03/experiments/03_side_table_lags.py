# %% [markdown]
# # Experiment: 03_side_table_lags
#
# **Date:** 2026-05-02
# **Goal:** measure the marginal RMSE gain (or lack of it) from lag-1
# and trailing rolling-mean-3 on six side-table columns
# (Price, Sales, Promotions, Avg_Max_Temp, Industry_Volume,
# Soda_Volume), on top of `02_lag_features`.
# **Result:** filled in after the run.

# %%
import skore

from beeristan.evaluate import splitter
from beeristan.features import add_lag_features, add_side_table_lag_features
from beeristan.pipeline import build_learner

# %% [markdown]
# ## Project

# %%
project = skore.Project(workspace="reports", name="beeristan", mode="local")

# %% [markdown]
# ## Learner and data

# %%
learner = build_learner(feature_steps=[add_lag_features, add_side_table_lag_features])
data = {"data_dir": "data/train_OwBvO8W"}

# %% [markdown]
# ## Evaluate

# %%
report = skore.evaluate(learner, data=data, splitter=splitter)
report

# %% [markdown]
# ## Persist

# %%
project.put("03_side_table_lags", report)
