# PLAN

<!--
This file is the durable index of every experiment in this workspace.
Three sections, in order: Status, History, Backlog. Don't add new
top-level sections; they break the contract that lets future sessions
read this file in two seconds.

Owner: `iterate-ml-experiment` skill. Pair each `plan/NN_short_name.md`
with `experiments/NN_short_name.py` (identical stems).
-->

## Status

- **Project / dataset:** <fill in — e.g., `adult-census` classification>
- **Goal:** <one sentence — what would "done" look like for this project?>
- **Last experiment:** <NN_name> — <status: planned | approved | running | done | abandoned>
- **Last result:** <one-line headline metric, or "n/a" if not yet run>

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Status values: planned | approved | running | done | abandoned.
-->

| Stem | Intent (one line) | Status | Headline result | Plan file |
|---|---|---|---|---|
| <!-- e.g. `01_baseline` --> | <!-- "tabular_learner on raw features" --> | <!-- done --> | <!-- "ROC-AUC 0.86 ± 0.01" --> | <!-- [plan](01_baseline.md) --> |

## Backlog

<!--
Ideas that haven't been committed to a `plan/NN_*.md` file yet. When
one graduates into a plan file, move it out of here and add the row to
History.
-->

- <!-- e.g. "try grouped CV by patient_id — current splits may leak across groups" -->
