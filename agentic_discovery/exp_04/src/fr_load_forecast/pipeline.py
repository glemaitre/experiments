"""Learner declaration: skrub DataOps graph for h=24 load forecasting.

The pipeline binds the data directory as a source identifier, loads and
feature-engineers inside the graph, marks X / y, and applies a
``skrub.tabular_pipeline("regressor")`` (TableVectorizer +
HistGradientBoostingRegressor) as the predictor.

The ``datetime`` column is kept inside X past the marker so the custom
walk-forward splitter (in ``evaluate.py``) can read fold boundaries from
it; it is dropped just before the predictor so the model never trains
on the absolute timestamp.

See `build-ml-pipeline` for declarative mechanics.
"""

from __future__ import annotations

import skrub

from .data import load_dataset


def build_learner():
    """Return the unfit ``SkrubLearner`` for the baseline experiment."""
    data_dir = skrub.var("data_dir", value="data")
    frame = data_dir.skb.apply_func(load_dataset)

    X = frame.drop("target").skb.mark_as_X()
    y = frame["target"].skb.mark_as_y()

    X_features = X.drop("datetime")
    predictions = X_features.skb.apply(skrub.tabular_pipeline("regressor"), y=y)
    return predictions.skb.make_learner()
