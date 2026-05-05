"""01_baseline experiment script — t+12 electricity load forecast."""

# %% [markdown]
# # Experiment: 01_baseline
#
# **Date:** 2026-05-04
# **Goal:** Establish a t+12 forecasting baseline for France's hourly
# electricity load using `skrub.tabular_pipeline("regressor")` over
# calendar features, national-mean weather, and one load lag, evaluated
# with a custom monthly walk-forward CV.
# **Result:** filled in after the run.

# %%
import skore

from load_forecast import PROJECT_ROOT
from load_forecast.evaluate import splitter
from load_forecast.pipeline import build_learner

# %% [markdown]
# ## Paths

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project

# %%
project = skore.Project("load-forecast", workspace=PROJECT_ROOT / "reports")

# %% [markdown]
# ## Splitter sanity check
#
# Per `plan/01_baseline.md` § Risks: print one fold's `(train_max_ts,
# test_min_ts)` to verify the embargo holds (test month start minus 12 h).

# %%
from load_forecast.data import build_supervised_frame  # noqa: E402

_frame = build_supervised_frame(DATA_DIR)
_timestamps = _frame["prediction_time"]
print(f"n_splits: {splitter.get_n_splits(timestamps=_timestamps)}")
for i, (tr, te) in enumerate(splitter.split(timestamps=_timestamps)):
    if i in (0, 1):
        train_max = _timestamps[int(tr.max())]
        test_min = _timestamps[int(te.min())]
        test_max = _timestamps[int(te.max())]
        print(f"fold {i}: train ends {train_max} | test {test_min} → {test_max}")

# %% [markdown]
# ## Build learner and evaluate

# %%
learner = build_learner(data_dir_preview=DATA_DIR)
report = skore.evaluate(
    learner,
    data={"data_dir": str(DATA_DIR)},
    splitter=splitter,
)
report

# %% [markdown]
# ## Persist

# %%
project.put("01_baseline", report)
