# %%
from pathlib import Path

training_input_path = Path("../data/training_input_mtaTRFH.csv")
training_output_path = Path("../data/training_output_aq7NYgj.csv")
testing_input_path = Path("../data/test_input_D77jaRF.csv")

# %%
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    return df


# %%
df = pd.concat(
    [load_data(training_input_path), load_data(training_output_path)], axis=1
)

# %%
import skrub

table_report = skrub.TableReport(df)
table_report

# %%
X = df.drop(columns=["Target"])
y = df["Target"]

# %%
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

hgbdt = HistGradientBoostingRegressor(early_stopping=True)
linear_reg = RidgeCV(alphas=np.logspace(-6, 6, num=100))

# %%
import skore

# hgbdt_report = skore.evaluate(hgbdt, X, y, splitter=10)

# # %%
# log_reg_report = skore.evaluate(linear_reg, X, y, splitter=10)

# %%
X = skrub.var("X", training_input_path)
y = skrub.var("y", training_output_path)

# %%
X = X.skb.apply_func(skrub.deferred(load_data))
y = y.skb.apply_func(skrub.deferred(load_data)).skb.mark_as_y()

# %%
import itertools


@skrub.deferred
def volatility_transform(volatilities: pd.DataFrame) -> pd.DataFrame:
    log_volatility = np.log1p(volatilities)
    log_volatility.columns = [f"log_{c}" for c in volatilities.columns]

    variance = volatilities.pow(2)
    variance.columns = [f"{c}_variance" for c in volatilities.columns]

    ratio_parts = [
        (volatilities[ci] / volatilities[cj]).rename(f"{ci}_div_{cj}")
        for ci, cj in itertools.combinations(volatilities.columns, 2)
    ]
    volatility_ratios = (
        pd.concat(ratio_parts, axis=1)
        if ratio_parts
        else pd.DataFrame(index=volatilities.index)
    )

    average_volatility = volatilities.mean(axis=1).rename("avg_volatility")
    max_volatility = volatilities.max(axis=1).rename("max_volatility")
    min_volatility = volatilities.min(axis=1).rename("min_volatility")
    aggregate_volatility = pd.concat(
        [average_volatility, max_volatility, min_volatility],
        axis=1,
    )

    return pd.concat(
        [log_volatility, variance, volatility_ratios, aggregate_volatility],
        axis=1,
    )


# %%
from skrub import selectors as s

augmented_votalities = X.skb.select(
    s.filter_names(lambda name: name.startswith("sigma"))
).skb.apply_func(volatility_transform)

# %%


@skrub.deferred
def spot_transform(spots: pd.DataFrame) -> pd.DataFrame:
    avg_spot = spots.mean(axis=1).rename("avg_spot")
    min_spot = spots.min(axis=1).rename("min_spot")
    max_spot = spots.max(axis=1).rename("max_spot")
    spot_spread = (max_spot - min_spot).rename("spot_spread")
    return pd.concat([avg_spot, min_spot, max_spot, spot_spread], axis=1)


# %%
augmented_spots = X.skb.select(
    s.filter_names(lambda name: name.startswith("S"))
).skb.apply_func(spot_transform)

# %%


@skrub.deferred
def distance_to_barrier(features: pd.DataFrame) -> pd.DataFrame:
    spots = s.select(features, s.filter_names(lambda name: name.startswith("S")))
    barriers = s.select(features, s.filter_names(lambda name: "Barrier" in name)).copy()
    names = {}
    for col in barriers.columns:
        barriers[col] = spots.min(axis=1) - barriers[col]
        names[col] = f"distance_to_barrier_{col}"
    return barriers.rename(columns=names)


# %%
augmented_barriers = X.skb.select(
    s.filter_names(lambda name: name.startswith("S"))
    | s.filter_names(lambda name: "Barrier" in name)
).skb.apply_func(distance_to_barrier)


# %%
feature_engineering = X.skb.concat(
    [augmented_spots, augmented_votalities, augmented_barriers],
    axis=1,
).skb.mark_as_X()

# %%
pred = feature_engineering.skb.apply(hgbdt, y=y)

# %%
results = pred.skb.cross_validate(cv=10, scoring="neg_root_mean_squared_error")

# %%
results.mean(axis=0)

# %%
