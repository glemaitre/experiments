"""Learner declaration: skrub DataOps graph for h=24 load forecasting.

The pipeline binds the data directory as a source identifier, loads and
feature-engineers inside the graph, marks X / y, and applies a
``skrub.tabular_pipeline("regressor")`` (TableVectorizer +
HistGradientBoostingRegressor) as the predictor.

The ``datetime`` column is kept inside X past the marker so the custom
walk-forward splitter (in ``evaluate.py``) can read fold boundaries from
it; it is dropped just before the predictor so the model never trains
on the absolute timestamp.

``build_learner`` exposes ``lags_hours`` and ``rolling_windows_hours``
parameters so each experiment script can request its own past-covariate
set without rewriting the graph. Defaults preserve the ``01_baseline``
configuration so re-running that experiment still produces the baseline
report.

See `build-ml-pipeline` for declarative mechanics.
"""

from __future__ import annotations

import skrub

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from .data import (
    load_dataset,
    load_horizon_feature_dataset,
    load_multi_output_dataset,
)
from .features import HORIZON_HOURS, LOAD_LAGS_HOURS


def build_learner(
    lags_hours: tuple[int, ...] = LOAD_LAGS_HOURS,
    rolling_windows_hours: tuple[int, ...] = (),
):
    """Return the unfit ``SkrubLearner``.

    Parameters
    ----------
    lags_hours : tuple of int
        Backward load-lag offsets in hours. Defaults to the baseline
        ``(1, 24, 168)``.
    rolling_windows_hours : tuple of int
        Backward rolling-window sizes in hours for mean / std features.
        Empty by default (baseline = no rolling features).
    """
    data_dir = skrub.var("data_dir", value="data")
    frame = data_dir.skb.apply_func(
        load_dataset,
        lags_hours=lags_hours,
        rolling_windows_hours=rolling_windows_hours,
    )

    X = frame.drop("target").skb.mark_as_X()
    y = frame["target"].skb.mark_as_y()

    X_features = X.drop("datetime")
    predictions = X_features.skb.apply(skrub.tabular_pipeline("regressor"), y=y)
    return predictions.skb.make_learner()


def build_horizon_feature_learner(
    horizons: tuple[int, ...] = tuple(range(1, HORIZON_HOURS + 1)),
    lags_hours: tuple[int, ...] = LOAD_LAGS_HOURS,
    rolling_windows_hours: tuple[int, ...] = (),
):
    """Return the unfit ``SkrubLearner`` for the horizon-as-feature
    multi-horizon framing (experiment ``03_horizon_as_feature``).

    Each prediction time is replicated 24× with the horizon ``h`` as a
    numeric feature; weather and calendar are aligned to ``t + h`` per
    replica; target is ``load(t + h)``. Same ``tabular_pipeline``
    learner as the baseline.
    """
    data_dir = skrub.var("data_dir", value="data")
    frame = data_dir.skb.apply_func(
        load_horizon_feature_dataset,
        horizons=horizons,
        lags_hours=lags_hours,
        rolling_windows_hours=rolling_windows_hours,
    )

    X = frame.drop("target").skb.mark_as_X()
    y = frame["target"].skb.mark_as_y()

    X_features = X.drop("datetime")
    predictions = X_features.skb.apply(skrub.tabular_pipeline("regressor"), y=y)
    return predictions.skb.make_learner()


def build_multi_output_learner(
    horizons: tuple[int, ...] = tuple(range(1, HORIZON_HOURS + 1)),
    lags_hours: tuple[int, ...] = LOAD_LAGS_HOURS,
    rolling_windows_hours: tuple[int, ...] = (),
):
    """Return the unfit ``SkrubLearner`` for the multi-output regressor
    framing (experiment ``04_multi_output``).

    Single feature vector, 24 output columns. Wraps
    ``HistGradientBoostingRegressor`` in
    ``sklearn.multioutput.MultiOutputRegressor`` because HGB does not
    support native multi-output as of sklearn 1.8. ``n_jobs=-1``
    parallelizes the per-output sub-fits.
    """
    data_dir = skrub.var("data_dir", value="data")
    frame = data_dir.skb.apply_func(
        load_multi_output_dataset,
        horizons=horizons,
        lags_hours=lags_hours,
        rolling_windows_hours=rolling_windows_hours,
    )

    target_cols = [f"target_h{h}" for h in horizons]
    X = frame.drop(*target_cols).skb.mark_as_X()
    y = frame.select(target_cols).skb.mark_as_y()

    X_features = X.drop("datetime")
    multi_output_estimator = MultiOutputRegressor(
        HistGradientBoostingRegressor(), n_jobs=-1
    )
    predictions = X_features.skb.apply(
        skrub.tabular_pipeline(multi_output_estimator), y=y
    )
    return predictions.skb.make_learner()
