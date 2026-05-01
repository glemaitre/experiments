---
name: evaluate-ml-pipeline
description: >
  Methodology for evaluating a single sklearn-compatible learner (in
  particular, the `SkrubLearner` produced by `build-ml-pipeline`).
  Owns: which entry point to call (`skore.evaluate` first, the
  explicit report classes when needed), which cross-validator to pick
  from scikit-learn's catalogue, how to consume the structural
  metadata (`groups`, `times`, …) attached at build time via
  `.skb.mark_as_X(split_kwargs=...)`. Stops at "what does the report
  say". Defaults (metrics, plots) come from skore; only override on
  explicit user request.

  TRIGGER when: code calls `cross_val_score`, `cross_validate`,
  `classification_report`, or any handwritten metric print
  (`print(mean_squared_error(...))`); code calls
  `.skb.cross_validate(...)` (route through skore for richer output);
  user asks how to score, evaluate, or compare a single learner;
  user asks how to pick a cross-validator; user wants to see a
  report / metrics / diagnostic plots for a fitted learner.

  SKIP when: declaring the pipeline (use `build-ml-pipeline`);
  hyperparameter / model search (separate skill); fitting,
  persisting, or serving the final model; tracking or comparing
  experiments across multiple runs over time (separate skill).

  HOW TO USE: invoke before any evaluation call. The structural facts
  about the data (group keys, time ordering) should already be
  encoded at the X marker via `split_kwargs` — if they aren't and you
  can't tell from the data, return to `build-ml-pipeline` and ask the
  user. For symbol-level lookups, defer to `skore-api` (skore
  symbols) and `sklearn-api` (splitters); don't guess names from
  memory.
---

# Evaluate ML Pipeline

Pick the entry point, pick the cross-validator, route the metadata,
read the report. The pipeline declaration is out of scope (see
`build-ml-pipeline`).

## Scope

- **In scope:** choosing the evaluation entry point, picking a
  cross-validator, wiring `split_kwargs` into the splitter, reading
  the report, deciding when to escalate to explicit report classes.
- **Out of scope:** pipeline declaration, hyperparameter search,
  persistence, serving, multi-run tracking.

## Core rules

1. **`skore.evaluate(...)` is the entry point.** It is a dispatcher
   that returns the right report for the task and `cv` argument.
   **Never** hand-roll `cross_val_score` + manual metric prints, and
   don't drop back to bare sklearn for evaluation. If you see
   existing `cross_val_score` / `cross_validate` /
   `classification_report` / `mean_squared_error` calls in the diff,
   redirect them through `skore.evaluate`. Consult `skore-api` for
   the exact signature.

2. **Escalate to explicit report classes only when `evaluate` is too
   coarse.** The escalation order:

   - `EstimatorReport` — single fit on a held-out set (no CV); use
     when CV is wasteful (e.g., evaluating the final model on all
     data after CV has already been done).
   - `CrossValidationReport` — k-fold over one learner with access
     to per-fold artifacts.
   - `ComparisonReport` — two or more learners side-by-side.

   See `references/reports.md` for the escalation table; defer all
   API details to `skore-api`.

3. **Pick the cross-validator from the structural facts of the data
   — not by default.** The data tells you what splitter is correct.
   The structural facts arrive at the X marker through
   `split_kwargs` (set by `build-ml-pipeline` at declaration time).
   Mapping rules:

   | `split_kwargs` content | Splitter |
   |---|---|
   | `groups` | `GroupKFold` |
   | temporal ordering | **ask the user** (see "Time-ordered data" below) |
   | none | `KFold` (or `RepeatedKFold` for small / noisy data) |

   Imbalanced classification *does not* change the choice — use
   plain `KFold` / `GroupKFold`. See "Avoid by default" below.

   **Avoid by default:**
   - **Stratified variants** (`StratifiedKFold`,
     `StratifiedGroupKFold`, `StratifiedShuffleSplit`,
     `RepeatedStratifiedKFold`) — they reduce across-fold variance
     by construction, producing over-confident error bars on the
     score. Don't reach for them on imbalance.
   - **`LeaveOneOut` / `LeaveOneGroupOut` / `LeavePGroupsOut`** —
     high per-fold variance; aggregate hides the noise. Use
     `KFold` / `GroupKFold` with 5–10 splits instead.

   See `references/cross-validation.md` § "Avoid" for the reasoning.
   Wiring details: `references/metadata-routing.md`.

   **Time-ordered data — user-question flow.** When the data is
   temporal, ask the user *before* picking a splitter:
   1. Is `TimeSeriesSplit` sufficient, or do they want a custom
      splitter (purged windows, calendar blocks, walk-forward)? If
      custom, see `references/custom-splitter.md`.
   2. Should the time column stay as a covariate or be dropped from
      the feature matrix? Encoders can extract calendar patterns
      from a timestamp; the user's call.

   If `split_kwargs` is empty *and* you cannot confirm there's no
   structure (from build-time checks or from the user), do not
   silently default. Return to `build-ml-pipeline` and ask the user
   first.

4. **Trust skore's metric defaults; override only on explicit user
   request.** `skore.evaluate` picks task-appropriate metrics
   automatically (regression: MSE/RMSE/MAE/R²; binary: accuracy,
   precision, recall, F1, ROC-AUC; multiclass: macro/micro variants;
   multilabel: per-label + averages). Override only when the user
   says so — e.g., "use RMSE", "report ROC-AUC". Don't pre-emptively
   pin metrics or pass a `scoring=...` argument unless asked.

5. **Custom splitter — only when sklearn doesn't have it.** Examples
   that justify one: purged-and-embargoed time-series CV (finance),
   blocked spatial CV. The contract is small: `split` +
   `get_n_splits`. See `references/custom-splitter.md`. Otherwise,
   prefer the sklearn built-in.

## Decision flow

1. Is the goal to *score* one learner, or to *compare* ≥ 2?
   - One → `skore.evaluate(...)` (default), escalate to
     `CrossValidationReport` or `EstimatorReport` only if needed.
   - ≥ 2 → `ComparisonReport`.
2. Read `split_kwargs` at the X marker.
3. Map to a splitter using the table in rule 3.
4. Pass the splitter via `cv=...` to the chosen entry point.
5. Inspect the report; override metrics only on explicit user
   request.

## Companion skills

- **`skore-api`** — every skore symbol used here. Mandatory before
  naming `evaluate`, `EstimatorReport`, `CrossValidationReport`,
  `ComparisonReport`. Don't guess from memory.
- **`sklearn-api`** — every splitter used here. Mandatory before
  naming `KFold`, `GroupKFold`, `TimeSeriesSplit`, etc.
- **`build-ml-pipeline`** — upstream pipeline shape and where
  structural metadata is attached via `split_kwargs`. Return there
  if the metadata you need at evaluation time isn't wired in.
