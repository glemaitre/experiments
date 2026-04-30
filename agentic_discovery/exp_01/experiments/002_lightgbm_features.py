"""002 — Same LightGBM config as 001, plus engineered domain features.

Adds basket order-statistics (worst/best/mean of S, mu, sigma), basket
variance / vol from the 3x3 correlation matrix, time-scaled drift / vol,
and signed distances from the worst-of underlying to each barrier.

Same training budget as 001 so the gain is attributable to the features.
"""

from __future__ import annotations

import os
import time

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from exp_01.data import FEATURE_COLS, load_train, train_val_split
from exp_01.features import feature_matrix
from exp_01.metrics import all_metrics
from exp_01.tracking import run

os.environ.setdefault("EXP_SCRIPT", __file__)

PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 0,
}
NUM_BOOST_ROUND = 5000
EARLY_STOP_ROUNDS = 100


def main() -> None:
    X_raw, y = load_train()
    train_idx, val_idx = train_val_split(len(y), val_size=0.2, seed=0)

    X_np, cols = feature_matrix(X_raw, raw_cols=FEATURE_COLS)
    X_tr, X_val = X_np[train_idx], X_np[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=cols)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=cols)

    t0 = time.perf_counter()
    booster = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    fit_seconds = time.perf_counter() - t0

    y_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    metrics = all_metrics(y_val, y_pred)

    with run(
        "lightgbm-features",
        description="LightGBM defaults + engineered domain features.",
        tags={"model_family": "gbdt", "library": "lightgbm", "features": "engineered"},
    ):
        mlflow.log_params({
            **{f"lgbm.{k}": v for k, v in PARAMS.items()},
            "lgbm.num_boost_round_max": NUM_BOOST_ROUND,
            "lgbm.early_stopping_rounds": EARLY_STOP_ROUNDS,
            "lgbm.best_iteration": booster.best_iteration,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "split.seed": 0,
            "split.val_size": 0.2,
            "n_features": len(cols),
            "feature_set": "raw23+engineered",
        })
        mlflow.log_metric("fit_seconds", fit_seconds)
        for k, v in metrics.items():
            mlflow.log_metric(f"val.{k}", v)
        mlflow.lightgbm.log_model(booster, name="model")
        for name, gain in zip(cols, booster.feature_importance(importance_type="gain")):
            mlflow.log_metric(f"importance.{name}", float(gain))

    print(f"best_iteration = {booster.best_iteration}")
    for k, v in metrics.items():
        print(f"  val.{k}: {v:.6f}")


if __name__ == "__main__":
    main()
