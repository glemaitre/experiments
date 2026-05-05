"""Learner declaration for the t+12 baseline.

The pipeline binds ``data_dir`` as the source identifier; the loader
``build_supervised_frame`` runs as the first ``.skb.apply_func`` so the
env-dict at fit / cross-validate time is one binding per source.
"""

from __future__ import annotations

from pathlib import Path

import skrub

from .data import build_supervised_frame


def build_learner(data_dir_preview: str | Path | None = None):
    """Return the unfit `SkrubLearner` for the t+12 baseline.

    Parameters
    ----------
    data_dir_preview : str or Path or None, optional
        Absolute path used as the preview binding for
        ``learner.skb.preview()``. Pass ``PROJECT_ROOT / "data"`` from the
        experiment script. Leave ``None`` for fit / cross-validate runs —
        the env-dict (``data={"data_dir": ...}``) supplies the binding.
    """
    if data_dir_preview is not None:
        data_dir = skrub.var("data_dir", value=str(data_dir_preview))
    else:
        data_dir = skrub.var("data_dir")

    data = data_dir.skb.apply_func(build_supervised_frame)

    # Keep `prediction_time` in X so the custom splitter can read it
    # directly (skore drives CV with `splitter.split(X, y)` and does
    # not forward split_kwargs metadata).
    X = data.drop("target_load").skb.mark_as_X()
    y = data["target_load"].skb.mark_as_y()

    # Drop the timestamp from the model's features after the X marker.
    X_features = X.drop("prediction_time")

    predictor = skrub.tabular_pipeline("regressor")
    predictions = X_features.skb.apply(predictor, y=y)
    return predictions.skb.make_learner()
