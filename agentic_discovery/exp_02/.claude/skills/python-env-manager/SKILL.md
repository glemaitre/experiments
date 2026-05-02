---
name: python-env-manager
description: >
  Single source of truth for "which Python environment manager does
  this project use, and how do I install a package with it?". Owns
  the detection table (pixi / uv / poetry / hatch / conda+mamba /
  pip+venv), the install / remove / upgrade commands per manager,
  and the bootstrap path when no manager is in place (default
  recommendation: pixi). Stops at "the install command was issued
  with the right manager and the package is importable".

  TRIGGER when (any of these):
  (1) **about to install / add / pin / upgrade / remove a Python
      package** — `pip install`, `pixi add`, `uv add`, `poetry add`,
      `conda install`, etc. — under any framing;
  (2) `data-science-python-stack` § "Missing dependency" surfaced a
      missing import and an install is the next step;
  (3) a workflow skill's Stop condition fired on a missing
      dependency (`build-ml-pipeline`, `evaluate-ml-pipeline`,
      `organize-ml-workspace`);
  (4) starting a new Python project and no manager is in place yet
      (bootstrap with pixi unless the user picks otherwise).

  SKIP when: the project is non-Python; the install/add command is
  for a non-Python tool (npm, brew, apt, cargo, gem); the dependency
  is already installed and importable; the work is purely editing
  existing source code with no new dependency in play.

  HOW TO USE: **detect first, then install**. Run the § "Detection"
  table at the project root before issuing any install command. If
  no manager is detected, ask the user before bootstrapping. Never
  install with a different manager than the one the project uses
  (e.g., never `pip install` into a pixi-managed project) — that
  creates env state divergence the manifest won't track. **Read
  the "Stop conditions" block and emit the Pre-flight checklist as
  visible text in your response — both are mandatory before issuing
  any command.**
---

# Python Env Manager

Detect the env manager, install with the right command. Single
authority for `data-science-python-stack` and the workflow skills
when they need a dependency added.

## Stop conditions — read before anything else

- **Wrong-manager install is forbidden.** If the project uses pixi,
  do not `pip install`. If it uses poetry, do not `uv add`. If it
  uses uv, do not `poetry add`. Mixing managers creates environment
  state the project's manifest doesn't track, and the next
  `pixi install` / `poetry install` / `uv sync` will silently undo
  the install. Detection (below) is mandatory before any command.
- **No silent bootstrap.** If detection finds no manager, do not
  pick one and start installing. Ask the user; the default
  *recommendation* is pixi, but the user must approve before
  `pixi init` runs.
- **Don't pin without reason.** Install commands here add packages
  unpinned by default (matching `data-science-python-stack` §
  "Conventions"). Pin only when the user asks or there's a known
  incompatibility.
- **Don't run the bootstrap installer yourself.** When pixi (or any
  manager) is missing, surface the install command and let the
  user run it. `curl | sh` is a system-level action that needs the
  user's hands on it, not Claude's.

## Pre-flight — emit this checklist as visible text before any command

Before running an install / add / remove / upgrade command, output
this block verbatim. Each box must be backed by a real detection
step or an explicit decision documented in the response.

```
Pre-flight (python-env-manager):
- [ ] Detection done; manager identified: <pixi | uv | poetry | hatch
      | conda | pip+venv | none>
- [ ] If "none": user asked which manager to bootstrap (default
      recommendation: pixi)
- [ ] Install command syntax confirmed for that manager (see § "Install
      commands")
- [ ] Feature / group / extras decided (pixi feature, poetry group,
      uv --dev, conda env name) — asked the user when ambiguous
- [ ] Package list ready: <pkg-1, pkg-2, ...>
```

## Detection — figure out the manager first

Run these checks at the project root in order. **The first signal
that matches wins.** If multiple signals are present (a real
possibility — e.g. `pyproject.toml` + `pixi.toml`), surface the
ambiguity to the user before installing.

| Signal at project root | Manager | Notes |
|---|---|---|
| `pixi.toml` or `pixi.lock` | **pixi** | Default for this stack. Likely multi-feature. |
| `uv.lock`, or `pyproject.toml` with `[tool.uv]` | **uv** | Fast Rust-based manager. |
| `poetry.lock`, or `pyproject.toml` with `[tool.poetry]` | **poetry** | Common in older Python projects. |
| `hatch.toml`, or `pyproject.toml` with `[tool.hatch]` | **hatch** | Declarative; install flow varies — ask the user. |
| `environment.yml` (and `conda` / `mamba` on PATH) | **conda / mamba** | Heavy but common in scientific stacks. |
| `requirements.txt` + `.venv/` or `venv/` | **pip + venv** | Plain Python; least integrated. |
| None of the above | **(nothing detected)** | Ask the user. Default *suggestion*: pixi. |

Notes:
- A `pyproject.toml` with **only** `[build-system]` / `[project]` and
  no `[tool.X]` table for any manager is ambiguous. Don't infer a
  manager from `pyproject.toml` alone — ask.
- `hatch` is declarative: dependencies live in `[project]
  dependencies` or `[tool.hatch.envs.<env>.dependencies]` in
  `pyproject.toml`, and `hatch` re-syncs on next `hatch run`. If
  detected, ask the user how they prefer to add deps (edit
  `pyproject.toml` vs. another flow) — there's no universal `hatch
  add` command.
- If both `pixi.toml` and a `pyproject.toml` with another manager's
  `[tool.X]` are present, the project may be transitioning. Ask
  before picking.

## Install commands — by manager

Once detected, use *only* the matching commands. Do not mix.

### pixi

Default for this stack. Pixi typically organizes deps per **feature**
(e.g. `default`, `dev`, `tracing`); confirm which feature the new
package belongs in before running.

| Action | Command |
|---|---|
| Add to default feature | `pixi add <pkg>` |
| Add to a specific feature | `pixi add --feature <feature> <pkg>` |
| Add to a specific environment | `pixi add -e <env> <pkg>` |
| Remove | `pixi remove <pkg>` (or `--feature <feature>`) |
| Upgrade | `pixi upgrade <pkg>` |
| Run inside an env | `pixi run -e <env> <command>` |
| Sync env from manifest | `pixi install` |

When unsure which feature a tool belongs in, ask the user. (Memory
note: in some projects, e.g. `mlflow` lives in a `tracing` feature,
not `default`.)

### uv

| Action | Command |
|---|---|
| Add a runtime dep | `uv add <pkg>` |
| Add a dev dep | `uv add --dev <pkg>` |
| Add to an optional group | `uv add --optional <group> <pkg>` |
| Remove | `uv remove <pkg>` |
| Upgrade a single pkg | `uv lock --upgrade-package <pkg>` |
| Run inside the env | `uv run <command>` |
| Sync env from manifest | `uv sync` |

### poetry

| Action | Command |
|---|---|
| Add a runtime dep | `poetry add <pkg>` |
| Add a dev dep | `poetry add --group dev <pkg>` |
| Add to a named group | `poetry add --group <name> <pkg>` |
| Remove | `poetry remove <pkg>` |
| Upgrade | `poetry update <pkg>` |
| Run inside the env | `poetry run <command>` |
| Sync env from manifest | `poetry install` |

### hatch

Hatch is declarative. There is no universal `hatch add`. Standard
flow:

1. Edit `pyproject.toml`:
   - Project-level dep → add to `[project] dependencies`.
   - Env-specific dep → add to
     `[tool.hatch.envs.<env>.dependencies]`.
2. Re-sync the env: `hatch env prune` (optional, removes stale
   envs), then any `hatch run -e <env> <command>` re-creates it.

Ask the user before editing `pyproject.toml` — the structure varies
per project.

### conda / mamba

`mamba` is a faster drop-in replacement for `conda`. Prefer it if
both are on PATH.

| Action | Command |
|---|---|
| Add a dep (conda-forge channel) | `conda install -n <env> -c conda-forge <pkg>` |
| Same with mamba | `mamba install -n <env> -c conda-forge <pkg>` |
| Remove | `conda remove -n <env> <pkg>` |
| Sync from `environment.yml` | `conda env update -f environment.yml --prune` |

If `environment.yml` is the source of truth for the project, edit
it and run the `env update` rather than installing one-off; this
keeps the manifest in sync.

### pip + venv

The least-integrated path. There is no manifest update — `pip
install` mutates the live env without tracking. Steps:

1. Activate the venv: `source .venv/bin/activate` (Linux/macOS) or
   `.venv\Scripts\activate` (Windows).
2. Install: `pip install <pkg>`.
3. If `requirements.txt` is the project's manifest, regenerate or
   edit it — `pip freeze > requirements.txt` is one option, but
   it captures all transitive pins; for a tighter diff, edit the
   file by hand to add the new top-level dep.

Surface to the user that `pip install` alone leaves no audit trail.
If the project is fresh, offer migration to a managed alternative
(pixi by default).

## Bootstrap — when no manager is detected

If detection found nothing **and the user agrees to use pixi**:

1. Check whether pixi is on PATH: `command -v pixi`.
2. If pixi is not installed, surface the install command and **ask
   the user to run it** (do not run `curl | sh` yourself):
   - Linux/macOS: `curl -fsSL https://pixi.sh/install.sh | sh`
   - Windows: `iwr -useb https://pixi.sh/install.ps1 | iex`
3. Once pixi is available, initialize: `pixi init` (creates
   `pixi.toml` in the current directory).
4. Add the relevant Tier 1 deps for an ML project (per
   `data-science-python-stack` § "Tier 1"):
   `pixi add scikit-learn skrub skore`.
5. Ask the user about the tabular-library choice (per
   `organize-ml-workspace` § "Stop conditions" — pandas vs polars).
   Add the chosen library: `pixi add pandas pyarrow` or
   `pixi add polars`.

If the user wants a different manager (uv / poetry / hatch / conda),
mirror the same flow with that manager's init command (`uv init`,
`poetry init`, `conda env create -f environment.yml`, etc.).

## Cross-references

This skill is the install layer for the rest of the stack. Invoke it
whenever those skills surface a missing dependency or a new install:

- **`data-science-python-stack`** — owns *what* to install (Tier 1
  mandatory, Tier 2 user choice, Tier 3 optional). When that skill
  decides a package is needed, this skill turns the decision into
  the right shell command.
- **`organize-ml-workspace`** — its Stop condition "Tabular library
  is asked, not assumed" produces a pandas-vs-polars decision; this
  skill executes the install.
- **`build-ml-pipeline`** / **`evaluate-ml-pipeline`** — their Stop
  conditions on missing `skrub` / `skore` redirect here for the
  install command. Their Pre-flight checklists include "Tier 1
  importable"; if a box fails, this skill is the next step.

## Conventions

- **One install operation per response.** Don't batch unrelated
  packages into one command. Group related packages (Tier 1
  bootstrap, or a single feature's deps) and confirm before
  continuing.
- **No `--no-deps` or version pins by default.** Match
  `data-science-python-stack` § "Conventions". Pin only on user
  request or known incompatibility.
- **Surface, don't bypass.** If an install fails (network, version
  conflict, missing channel), surface the error and the command —
  don't try alternative managers as a workaround. Wrong-manager
  workarounds are a Stop-condition violation.
