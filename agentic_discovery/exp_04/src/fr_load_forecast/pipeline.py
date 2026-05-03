"""Learner declaration: skrub DataOps graph for h=24 load forecasting.

The pipeline binds the data directory as a source identifier, loads and
feature-engineers inside the graph, marks X / y, and applies a
``skrub.tabular_pipeline("regressor")`` (TableVectorizer +
HistGradientBoostingRegressor) as the predictor.

The ``datetime`` column is kept inside X past the marker so the custom
walk-forward splitter (in ``evaluate.py``) can read fold boundaries from
it; it is dropped just before the predictor so the model never trains
on the absolute timestamp.

Each ``build_*_learner`` exposes ``lags_hours`` and
``rolling_windows_hours`` so each experiment script can request its
own past-covariate set without rewriting the graph. Defaults preserve
the ``01_baseline`` configuration so re-running that experiment still
produces the baseline report.

Source-binding preview. The root ``skrub.var("data_dir", ...)``
accepts an optional ``data_dir_preview`` keyword (an absolute path,
typically ``fr_load_forecast.PROJECT_ROOT / "data"``). The preview is
only consumed by ``learner.skb.preview()`` during interactive
iteration; for fit / cross-validate runs the env-dict passed to
``skore.evaluate(..., data={"data_dir": str(DATA_DIR)})`` supplies
the binding regardless. No relative-path literal is baked into this
module — see `build-ml-pipeline` rule 2 and the source-binding
reference.

See `build-ml-pipeline` for declarative mechanics.
"""

from __future__ import annotations

from pathlib import Path

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
    data_dir_preview: str | Path | None = None,
):
    """Return the unfit ``SkrubLearner`` for the single-horizon framing.

    Parameters
    ----------
    lags_hours : tuple of int
        Backward load-lag offsets in hours. Defaults to the baseline
        ``(1, 24, 168)``.
    rolling_windows_hours : tuple of int
        Backward rolling-window sizes in hours for mean / std features.
        Empty by default (baseline = no rolling features).
    data_dir_preview : str or Path or None, optional
        Preview value for the source-bound ``skrub.var("data_dir",
        ...)`` root. Pass an absolute path (e.g.
        ``fr_load_forecast.PROJECT_ROOT / "data"``) when iterating
        interactively so ``learner.skb.preview()`` works. Leave as
        ``None`` for fit / cross-validate runs — the env-dict passed
        to ``skore.evaluate`` supplies the binding regardless.
    """
    if data_dir_preview is not None:
        data_dir = skrub.var("data_dir", value=str(data_dir_preview))
    else:
        data_dir = skrub.var("data_dir")
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
    data_dir_preview: str | Path | None = None,
):
    """Return the unfit ``SkrubLearner`` for the horizon-as-feature
    multi-horizon framing (experiment ``03_horizon_as_feature``).

    Each prediction time is replicated 24× with the horizon ``h`` as a
    numeric feature; weather and calendar are aligned to ``t + h`` per
    replica; target is ``load(t + h)``. Same ``tabular_pipeline``
    learner as the baseline.

    See :func:`build_learner` for the ``data_dir_preview`` contract.
    """
    if data_dir_preview is not None:
        data_dir = skrub.var("data_dir", value=str(data_dir_preview))
    else:
        data_dir = skrub.var("data_dir")
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
    data_dir_preview: str | Path | None = None,
):
    """Return the unfit ``SkrubLearner`` for the multi-output regressor
    framing (experiment ``04_multi_output``).

    Single feature vector, 24 output columns. Wraps
    ``HistGradientBoostingRegressor`` in
    ``sklearn.multioutput.MultiOutputRegressor`` because HGB does not
    support native multi-output as of sklearn 1.8. ``n_jobs=-1``
    parallelizes the per-output sub-fits.

    See :func:`build_learner` for the ``data_dir_preview`` contract.
    """
    if data_dir_preview is not None:
        data_dir = skrub.var("data_dir", value=str(data_dir_preview))
    else:
        data_dir = skrub.var("data_dir")
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
