"""MLflow setup helpers shared by all experiment scripts."""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow

from exp_01.data import ROOT

EXPERIMENT_NAME = "option-pricing"
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def setup() -> None:
    """Point MLflow at the project's sqlite store + dedicated experiment."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def run(name: str, *, description: str | None = None, tags: dict | None = None):
    """Start an MLflow run with git provenance tags pre-attached."""
    setup()
    base_tags = {
        "git.sha": _git("rev-parse", "HEAD") or "",
        "git.branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "",
        "git.dirty": "1" if _git("status", "--porcelain") else "0",
        "script": Path(os.environ.get("EXP_SCRIPT", "")).name or "",
    }
    if tags:
        base_tags.update(tags)
    with mlflow.start_run(run_name=name, description=description, tags=base_tags) as r:
        yield r
