"""Learner declaration for the Beeristan baseline.

Source-bound skrub DataOps graph: a ``data_dir`` path goes in,
:func:`beeristan.data.load_panel` materializes the joined panel,
the ``Volume`` column is the target, every other column is a
feature. The default ``skrub.tabular_pipeline("regressor")`` does
the type-aware encoding and fits a HistGradientBoostingRegressor.
"""

from __future__ import annotations

import skrub

from beeristan.data import load_panel
from beeristan.features import add_lag_features, add_side_table_lag_features


def build_learner() -> skrub.SkrubLearner:
    """Return the unfit baseline learner.

    Returns
    -------
    skrub.SkrubLearner
        DataOps learner whose env-dict expects ``{"data_dir": <path>}``.
        At fit time the loader produces the joined panel, ``Volume`` is
        the target, and ``skrub.tabular_pipeline("regressor")`` handles
        type-aware encoding plus a HistGradientBoostingRegressor tail.
    """
    data_dir = skrub.var("data_dir", value="data/train_OwBvO8W")
    panel = (
        data_dir.skb.apply_func(load_panel)
        .skb.apply_func(add_lag_features)
        .skb.apply_func(add_side_table_lag_features)
    )

    y = panel["Volume"].skb.mark_as_y()
    X = panel.drop("Volume").skb.mark_as_X()

    predictions = X.skb.apply(skrub.tabular_pipeline("regressor"), y=y)
    return predictions.skb.make_learner()
