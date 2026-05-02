# %% [markdown]
# # Experiment 04: multi-output regressor
#
# **Date:** 2026-05-02
# **Goal:** Multi-horizon forecasting with one estimator predicting 24
# outputs jointly. `MultiOutputRegressor` wraps
# `HistGradientBoostingRegressor` (HGB doesn't support native
# multi-output). Single feature vector per prediction time: load lags,
# calendar at t, mean weather over [t+1, t+24]. Targets = load(t+1)
# through load(t+24).
# **Result:** filled in after the run.
#
# See `plan/04_multi_output.md` for the full design contract.

# %%
import skore

from fr_load_forecast.evaluate import splitter
from fr_load_forecast.pipeline import build_multi_output_learner

# %% [markdown]
# ## Project

# %% [markdown]
# Note: skore constrains each Project to a single ML task. `01`–`03`
# are regression; `04` is multioutput-regression, which is a distinct
# ML task in skore — so we open a sibling Project under
# `fr-load-forecast-mh` in the same `reports/` workspace. The cross-
# project comparison happens post-hoc by loading both projects.

# %%
project = skore.Project(workspace="reports", name="fr-load-forecast-mh", mode="local")

# %% [markdown]
# ## Learner
#
# Same past-covariate set as `01_baseline` (load_t + lag_{1,24,168}h);
# new features: calendar at t (not t+h), weather mean over [t+1, t+24].
# Target: 24 columns target_h1 .. target_h24.

# %%
learner = build_multi_output_learner()

# %% [markdown]
# ## Evaluate
#
# Same `DatetimeAnchoredWalkForward` splitter; one row per prediction
# time so no replication concerns.

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
project.put("04_multi_output", report)
