# %%
import skore
from pathlib import Path

workspace = Path("../skore-artifacts")
project = skore.Project(name="natixis", mode="local", workspace=workspace)

# %%

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

features_volatilities = X.skb.select(
    s.filter_names(lambda name: name.startswith("sigma"))
).skb.apply_func(volatility_transform)

# %%
features = X.skb.concat([features_volatilities], axis=1).skb.mark_as_X()

# %%
pred = features.skb.apply(hgbdt, y=y)

# %%
from sklearn.metrics import make_scorer, mean_squared_error

report = skore.evaluate(pred, splitter=10)
report.metrics.add(make_scorer(mean_squared_error, greater_is_better=True))
report

# %%
project.put("hgbdt-report", report)

# %%
