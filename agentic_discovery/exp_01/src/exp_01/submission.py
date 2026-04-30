"""Refit on full training data and produce a leaderboard submission CSV.

The challenge wants a `(ID, Target)` CSV in the same format as
`training_output_*.csv`. Submissions land under `submissions/` with the
MLflow run-id of the parent experiment in the filename so they are
auditable back to a tagged run.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from exp_01.data import ID_COL, ROOT, TARGET_COL

SUBMISSIONS_DIR = ROOT / "submissions"


def write_submission(test_ids, predictions, *, name: str) -> Path:
    """Write a (ID, Target) CSV to submissions/<name>.csv and return the path."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({ID_COL: test_ids, TARGET_COL: predictions})
    out = SUBMISSIONS_DIR / f"{name}.csv"
    df.write_csv(out)
    return out
