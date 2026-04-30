"""Data loading, persistence, and splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# Raw CSVs as delivered by the challenge (filenames carry an upload token).
RAW_TRAIN_X = DATA_DIR / "training_input_mtaTRFH.csv"
RAW_TRAIN_Y = DATA_DIR / "training_output_aq7NYgj.csv"
RAW_TEST_X = DATA_DIR / "test_input_D77jaRF.csv"

# Parquet caches built by `prepare()`.
TRAIN_PARQUET = DATA_DIR / "train.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"

TARGET_COL = "Target"
ID_COL = "ID"
FEATURE_COLS: tuple[str, ...] = (
    "S1", "S2", "S3",
    "mu1", "mu2", "mu3",
    "sigma1", "sigma2", "sigma3",
    "rho12", "rho13", "rho23",
    "Bonus",
    "YetiBarrier", "YetiCoupon",
    "PhoenixBarrier", "PhoenixCoupon",
    "PDIBarrier", "PDIGearing", "PDIStrike", "PDIType",
    "Maturity", "NbDates",
)


def prepare(force: bool = False) -> None:
    """Convert raw CSVs to parquet for fast repeated loading."""
    if not TRAIN_PARQUET.exists() or force:
        x = pl.read_csv(RAW_TRAIN_X)
        y = pl.read_csv(RAW_TRAIN_Y)
        if not (x[ID_COL] == y[ID_COL]).all():
            raise RuntimeError("ID columns of training_input/output disagree")
        df = x.with_columns(y[TARGET_COL])
        df.write_parquet(TRAIN_PARQUET)
        print(f"wrote {TRAIN_PARQUET}  rows={df.height}  cols={df.width}")
    else:
        print(f"exists {TRAIN_PARQUET}")

    if not TEST_PARQUET.exists() or force:
        x = pl.read_csv(RAW_TEST_X)
        x.write_parquet(TEST_PARQUET)
        print(f"wrote {TEST_PARQUET}  rows={x.height}  cols={x.width}")
    else:
        print(f"exists {TEST_PARQUET}")


def load_train() -> tuple[pl.DataFrame, np.ndarray]:
    """Return (features as polars DataFrame, target as numpy array)."""
    if not TRAIN_PARQUET.exists():
        prepare()
    df = pl.read_parquet(TRAIN_PARQUET)
    y = df[TARGET_COL].to_numpy()
    X = df.select(FEATURE_COLS)
    return X, y


def load_test() -> pl.DataFrame:
    """Return the held-out test features (with ID column preserved)."""
    if not TEST_PARQUET.exists():
        prepare()
    return pl.read_parquet(TEST_PARQUET)


def train_val_split(
    n: int, val_size: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic random index split. Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(n * (1.0 - val_size)))
    return perm[:cut], perm[cut:]


def _main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="Convert raw CSVs to parquet")
    p.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(force=args.force)


if __name__ == "__main__":
    _main()
