"""000 — Sanity baseline: predict the training mean.

Establishes the floor that any real model must beat. The challenge metric is
total squared error; we also report MSE/RMSE/MAE/R^2 for orientation.
"""

from __future__ import annotations

import os
import time

import mlflow
import numpy as np

from exp_01.data import load_train, train_val_split
from exp_01.metrics import all_metrics
from exp_01.tracking import run

os.environ.setdefault("EXP_SCRIPT", __file__)


def main() -> None:
    X, y = load_train()
    train_idx, val_idx = train_val_split(len(y), val_size=0.2, seed=0)
    y_tr, y_val = y[train_idx], y[val_idx]

    t0 = time.perf_counter()
    pred_const = float(np.mean(y_tr))
    fit_seconds = time.perf_counter() - t0

    y_pred = np.full_like(y_val, pred_const)
    metrics = all_metrics(y_val, y_pred)

    with run(
        "constant-mean",
        description="Predict the training-set mean for every row.",
        tags={"model_family": "trivial"},
    ):
        mlflow.log_params({
            "model": "constant_mean",
            "prediction": pred_const,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "split.seed": 0,
            "split.val_size": 0.2,
        })
        mlflow.log_metric("fit_seconds", fit_seconds)
        for k, v in metrics.items():
            mlflow.log_metric(f"val.{k}", v)

    print(f"constant prediction = {pred_const:.6f}")
    for k, v in metrics.items():
        print(f"  val.{k}: {v:.6f}")


if __name__ == "__main__":
    main()
