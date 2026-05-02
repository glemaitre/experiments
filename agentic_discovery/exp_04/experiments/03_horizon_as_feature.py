# %% [markdown]
# # Experiment 03: horizon-as-feature multi-horizon forecasting
#
# **Date:** 2026-05-02
# **Goal:** Single-model multi-horizon framing for h ∈ {1, …, 24}.
# Each prediction time t is replicated 24× with horizon as a numeric
# feature; weather + calendar aligned to t + h; target = load(t + h).
# Same `tabular_pipeline` learner and same walk-forward splitter as
# `01_baseline`.
# **Result:** filled in after the run.
#
# See `plan/03_horizon_as_feature.md` for the full design contract.

# %%
import polars as pl
import skore

from fr_load_forecast.evaluate import splitter
from fr_load_forecast.pipeline import build_horizon_feature_learner

# %% [markdown]
# ## Project

# %%
project = skore.Project(workspace="reports", name="fr-load-forecast", mode="local")

# %% [markdown]
# ## Learner
#
# Default horizons = 1..24, default lags = (1, 24, 168), no rolling
# stats — matching `01_baseline`'s past-covariate set under the new
# multi-horizon framing.

# %%
learner = build_horizon_feature_learner()

# %% [markdown]
# ## Evaluate
#
# The custom walk-forward splitter reads `X["datetime"]` to compute fold
# boundaries. With 24× row replication, replicas of the same `t` share
# the same datetime, so they all land in the same fold by construction.

# %%
report = skore.evaluate(
    learner,
    data={"data_dir": "data"},
    splitter=splitter,
)
report

# %% [markdown]
# ## Persist

# %%
project.put("03_horizon_as_feature", report)
