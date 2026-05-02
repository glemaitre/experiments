"""Learner declaration for the Beeristan demand forecast.

Source-bound skrub DataOps graph: a ``data_dir`` path goes in,
:func:`beeristan.data.load_panel` materializes the joined panel,
each caller-provided feature function is applied in order via
``.skb.apply_func``, and the ``Volume`` column becomes the target.
The default ``skrub.tabular_pipeline("regressor")`` handles
type-aware encoding plus a HistGradientBoostingRegressor tail.

Each experiment script picks the feature pipeline it cares about
by passing an explicit ``feature_steps`` list, so experiment scripts
remain reproducible regardless of what later experiments add to
:mod:`beeristan.features`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl
import skrub

from beeristan.data import load_panel

FeatureStep = Callable[[pl.DataFrame], pl.DataFrame]


def build_learner(feature_steps: Sequence[FeatureStep] = ()) -> skrub.SkrubLearner:
    """Return the unfit learner for an experiment to consume.

    Parameters
    ----------
    feature_steps : sequence of callables, default=()
        Stateless feature functions applied to the joined panel in
        order, before the X / y marker. Each callable takes and
        returns a polars DataFrame. The empty default reproduces
        the ``01_baseline`` pipeline.

    Returns
    -------
    skrub.SkrubLearner
        DataOps learner whose env-dict expects ``{"data_dir": <path>}``.
    """
    data_dir = skrub.var("data_dir", value="data/train_OwBvO8W")
    panel = data_dir.skb.apply_func(load_panel)
    for step in feature_steps:
        panel = panel.skb.apply_func(step)

    y = panel["Volume"].skb.mark_as_y()
    X = panel.drop("Volume").skb.mark_as_X()

    predictions = X.skb.apply(skrub.tabular_pipeline("regressor"), y=y)
    return predictions.skb.make_learner()
