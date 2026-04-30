"""004 — Hypothesis-driven LightGBM hyper-parameter sweep.

Each config is logged as its own MLflow run, all under the parent
"lightgbm-sweep". Configs are named after the *hypothesis* they test, not
just the param values, so the run name reads as a small experiment in
itself.
"""

from __future__ import annotations

import os
import time
from copy import deepcopy

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from exp_01.data import FEATURE_COLS, load_train, train_val_split
from exp_01.features import feature_matrix
from exp_01.metrics import all_metrics
from exp_01.tracking import run

os.environ.setdefault("EXP_SCRIPT", __file__)

BASE: dict = {
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
NUM_BOOST_ROUND = 12000
EARLY_STOP_ROUNDS = 200


def _override(**kw) -> dict:
    p = deepcopy(BASE)
    p.update(kw)
    return p


CONFIGS: list[tuple[str, dict, str]] = [
    (
        "lower-lr-more-rounds",
        _override(learning_rate=0.025),
        "Halve learning rate; rely on early stop. Often improves generalization.",
    ),
    (
        "more-leaves",
        _override(num_leaves=127),
        "More leaves → captures higher-order interactions; may overfit.",
    ),
    (
        "stronger-reg",
        _override(min_child_samples=500, feature_fraction=0.8, bagging_fraction=0.8),
        "Stronger regularization (bigger min_child, lower feature/bagging fractions).",
    ),
    (
        "more-leaves-stronger-reg",
        _override(num_leaves=127, min_child_samples=500, feature_fraction=0.8),
        "More capacity + matched regularization, in case capacity alone overfits.",
    ),
]


def main() -> None:
    X_raw, y = load_train()
    train_idx, val_idx = train_val_split(len(y), val_size=0.2, seed=0)
    X_np, cols = feature_matrix(X_raw, raw_cols=FEATURE_COLS)
    X_tr, X_val = X_np[train_idx], X_np[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=cols)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=cols)

    for name, params, description in CONFIGS:
        print(f"\n=== {name} ===  {description}")
        t0 = time.perf_counter()
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )
        fit_seconds = time.perf_counter() - t0
        y_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
        metrics = all_metrics(y_val, y_pred)

        with run(
            f"lightgbm-sweep/{name}",
            description=description,
            tags={"model_family": "gbdt", "library": "lightgbm",
                  "features": "engineered", "sweep": "004"},
        ):
            mlflow.log_params({
                **{f"lgbm.{k}": v for k, v in params.items()},
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

        print(f"  best_iteration={booster.best_iteration}  fit_seconds={fit_seconds:.0f}")
        print(f"  val.rmse={metrics['rmse']:.6f}  val.mse={metrics['mse']:.3e}")


if __name__ == "__main__":
    main()
