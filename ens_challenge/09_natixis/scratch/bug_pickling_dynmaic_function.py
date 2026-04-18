# %%
import skore
from pathlib import Path

workspace = Path("../skore-artifacts")
project = skore.Project(
    name="bug-pickling-dynamic-function", mode="local", workspace=workspace
)

# %%
from sklearn.datasets import make_classification

data, target = make_classification(n_samples=100, random_state=0)


# %%
import skrub

X = skrub.var("X", data)
y = skrub.var("y", target).skb.mark_as_y()

# %%
import numpy as np


@skrub.deferred
def augment_with_squares(X):
    return np.hstack([X, X**2])


features = X.skb.apply_func(augment_with_squares).skb.mark_as_X()

# %%
from sklearn.linear_model import LogisticRegression

pred = features.skb.apply(LogisticRegression(), y=y)

# %%
report = skore.evaluate(pred, splitter=5, pos_label=1)
report

# %%
project.put("logistic-function-transformer-cv", report)

# %%
