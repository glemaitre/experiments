"""Package root.

Exposes ``PROJECT_ROOT``: the absolute path to the project root,
derived from this file's location. Modules and experiment scripts
that need a project-relative path (the ``data/`` folder, in
particular) import this constant rather than hard-coding a
CWD-relative string. This is what lets experiments run from any
CWD without breaking.

Works because the package is installed in editable mode (declared
in ``pyproject.toml``, wired into pixi via ``pixi add --pypi
--editable .``); ``__file__`` lives at
``<root>/src/fr_load_forecast/__init__.py``, so ``parents[2]`` is
the project root.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
