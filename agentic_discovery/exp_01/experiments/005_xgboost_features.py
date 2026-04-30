"""005 — Same engineered features, but XGBoost (`hist` tree method).

A second-opinion GBM. If XGBoost lands within ~1% of LightGBM at the same
budget, the family is the bottleneck and we should try a different model
class (NN, GP, k-NN-on-distances). If one library decisively beats the
other, the loss is being shaped by leaf-policy / split-finder heuristics
and we should tune *that* library further.
"""

from __future__ import annotations

import os
import time

import mlflow
import mlflow.xgboost
import xgboost as xgb

from exp_01.data import FEATURE_COLS, load_train, train_val_split
from exp_01.features import feature_matrix
from exp_01.metrics import all_metrics
from exp_01.tracking import run

os.environ.setdefault("EXP_SCRIPT", __file__)

PARAMS: dict = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "seed": 0,
    "verbosity": 0,
}
NUM_BOOST_ROUND = 12000
EARLY_STOP_ROUNDS = 200


def main() -> None:
    X_raw, y = load_train()
    train_idx, val_idx = train_val_split(len(y), val_size=0.2, seed=0)
    X_np, cols = feature_matrix(X_raw, raw_cols=FEATURE_COLS)
    X_tr, X_val = X_np[train_idx], X_np[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=cols)

    t0 = time.perf_counter()
    booster = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        verbose_eval=500,
    )
    fit_seconds = time.perf_counter() - t0

    y_pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    metrics = all_metrics(y_val, y_pred)

    with run(
        "xgboost-features",
        description="XGBoost hist on engineered features, 12k rounds @ lr=0.05.",
        tags={"model_family": "gbdt", "library": "xgboost", "features": "engineered"},
    ):
        mlflow.log_params({
            **{f"xgb.{k}": v for k, v in PARAMS.items()},
            "xgb.num_boost_round_max": NUM_BOOST_ROUND,
            "xgb.early_stopping_rounds": EARLY_STOP_ROUNDS,
            "xgb.best_iteration": booster.best_iteration,
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
        mlflow.xgboost.log_model(booster, name="model")
        score = booster.get_score(importance_type="gain")
        for name, gain in score.items():
            mlflow.log_metric(f"importance.{name}", float(gain))

    print(f"best_iteration={booster.best_iteration}  fit_seconds={fit_seconds:.0f}")
    for k, v in metrics.items():
        print(f"  val.{k}: {v:.6f}")


if __name__ == "__main__":
    main()
