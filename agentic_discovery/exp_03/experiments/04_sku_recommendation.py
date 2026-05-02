# %% [markdown]
# # Experiment: 04_sku_recommendation
#
# **Date:** 2026-05-02
# **Goal:** recommend the top-2 SKUs for the cold-start agencies
# `Agency_06` and `Agency_14`. Train a baseline-shape learner
# (no lag features, since they are undefined for cold-start) on
# the full historical panel, predict over a constructed
# `Agency × SKU × month` grid for 2017, average per (Agency, SKU),
# and rank.
# **Result:** filled in after the run.

# %%
import shutil
import tempfile
from pathlib import Path

import polars as pl

from beeristan.data import load_cold_start_grid, load_panel
from beeristan.pipeline import build_learner

# %% [markdown]
# ## Inputs

# %%
TRAIN_DIR = Path("data/train_OwBvO8W")
COLD_START_AGENCIES = ["Agency_06", "Agency_14"]
YEAR_MONTHS_2017 = [201700 + m for m in range(1, 13)]
REPORT_OUT = Path("reports/04_sku_recommendation.csv")
SIDE_TABLE_FILES = [
    "price_sales_promotion.csv.zip",
    "weather.csv",
    "event_calendar.csv",
    "industry_volume.csv",
    "industry_soda_sales.csv",
    "demographics.csv",
]

# %% [markdown]
# ## Train panel: enumerate SKUs + sanity-check demographic range

# %%
train_panel = load_panel(TRAIN_DIR)
all_skus = sorted(train_panel["SKU"].unique().to_list())

demo = pl.read_csv(TRAIN_DIR / "demographics.csv")
cold_start_demo = demo.filter(pl.col("Agency").is_in(COLD_START_AGENCIES))
train_demo = demo.filter(~pl.col("Agency").is_in(COLD_START_AGENCIES))

for col in ["Avg_Population_2017", "Avg_Yearly_Household_Income_2017"]:
    train_min, train_max = train_demo[col].min(), train_demo[col].max()
    cs_values = cold_start_demo.select([pl.col("Agency"), pl.col(col)])
    print(f"{col}: train range [{train_min:,} .. {train_max:,}]")
    for row in cs_values.iter_rows(named=True):
        in_range = train_min <= row[col] <= train_max
        flag = "OK" if in_range else "OUT-OF-RANGE"
        print(f"  {row['Agency']}: {row[col]:,} ({flag})")

# %% [markdown]
# ## Train the cold-start-safe learner on the full panel

# %%
learner = build_learner(feature_steps=[])
learner.fit({"data_dir": str(TRAIN_DIR)})

# %% [markdown]
# ## Build the cold-start prediction grid + predict via temp data dir

# %%
cold_grid = load_cold_start_grid(
    agencies=COLD_START_AGENCIES,
    skus=all_skus,
    year_months=YEAR_MONTHS_2017,
)
print(
    f"cold-start grid: {cold_grid.shape[0]} rows = {len(COLD_START_AGENCIES)} agencies "
    f"× {len(all_skus)} SKUs × {len(YEAR_MONTHS_2017)} months"
)

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    cold_grid.write_csv(tmp_dir / "historical_volume.csv")
    for f in SIDE_TABLE_FILES:
        shutil.copy(TRAIN_DIR / f, tmp_dir / f)
    predictions = learner.predict({"data_dir": str(tmp_dir)})

# %% [markdown]
# ## Match predictions back to the cold-start grid + aggregate
#
# `load_panel` sorts the panel by `(Date, Agency, SKU)`; rebuild the
# same ordering on `cold_grid` so positional alignment with
# `predictions` is exact.

# %%
cold_grid_sorted = cold_grid.with_columns(
    pl.col("YearMonth").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m").alias("Date")
).sort(["Date", "Agency", "SKU"])

cold_grid_sorted = cold_grid_sorted.with_columns(
    pl.Series("predicted_volume", predictions)
)

agg = (
    cold_grid_sorted.group_by(["Agency", "SKU"])
    .agg(pl.col("predicted_volume").mean().alias("predicted_volume_mean_2017"))
    .sort(["Agency", "predicted_volume_mean_2017"], descending=[False, True])
)

top2 = agg.with_columns(
    pl.col("Agency").cum_count().over("Agency").alias("rank")
).filter(pl.col("rank") <= 2)

# %% [markdown]
# ## Persist + surface

# %%
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
top2.write_csv(REPORT_OUT)
print(f"\nwrote {REPORT_OUT}")
print()
print("=== top-2 SKU recommendation per cold-start agency ===")
print(top2)
