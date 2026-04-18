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
X = skrub.var("X", training_input_path)
y = skrub.var("y", training_output_path)

# %%
X = X.skb.apply_func(skrub.deferred(load_data))
y = y.skb.apply_func(skrub.deferred(load_data)).skb.mark_as_y()

# %%
features = X.skb.mark_as_X()

# %%
pred = features.skb.apply(hgbdt, y=y)

# %%
import skore

workspace = Path("../skore-artifacts")
project = skore.Project(name="natixis", mode="local", workspace=workspace)

# %%
from sklearn.metrics import make_scorer, mean_squared_error

report = skore.evaluate(pred, splitter=10)
report.metrics.add(make_scorer(mean_squared_error, greater_is_better=True))
project.put("hgbdt-report", report)

# %%
report

# %%
