---
name: build-ml-pipeline
description: >
  Opinionated, Pythonic way to **declare** the pipeline that goes from
  a data source to a predictor: data loading, preprocessing, feature
  engineering, estimator selection, and their composition. The pipeline
  is built as a **skrub DataOps graph**; every step is either a
  pure-Python function (stateless) attached via `.skb.apply_func`, or
  a scikit-learn-compatible estimator (stateful) attached via
  `.skb.apply`. Stops at the declared object. Out of scope: `fit`
  invocation, train/test split, hyperparameter tuning, persistence,
  evaluation. Deep-learning declarations are covered via internal
  `references/*.md`; skrub and scikit-learn mechanics live in
  sibling skills.

  TRIGGER when: writing or editing code that declares any link in the
  chain *data source → predictor* — data readers/loaders feeding a
  model (`read_csv`, `read_parquet`, `Dataset` classes), preprocessing
  or feature-engineering steps (transformers, encoders, imputers,
  scalers, text/image featurizers), **pure-Python data-processing
  functions destined for the pipeline path** (a custom `def` that
  cleans/derives/reshapes data, whether wrapped via
  `FunctionTransformer`, skrub `@deferred` / `skrub.var` expressions,
  a sklearn-compatible custom transformer class, or simply called in
  the training path before the estimator), composition objects
  (`Pipeline`, `make_pipeline`, `ColumnTransformer`, `FeatureUnion`,
  skrub `tabular_learner`, `nn.Module` / `LightningModule`
  definitions), or an estimator instantiated as the tail of the chain;
  a step is added, removed, swapped, or reordered inside an existing
  pipeline declaration; a bare `sklearn.Pipeline` / `make_pipeline` is
  being used as the top-level pipeline (fire to redirect the
  declaration into a skrub DataOps graph instead); the user asks to
  build / declare / set up a pipeline / classifier / regressor for X.

  SKIP when: `.fit(...)` calls, training loops, `Trainer.fit`, epoch
  loops; train/test split or cross-validation splitting; hyperparameter
  search (`GridSearchCV`, Optuna, Ray Tune) — separate skill; model
  persistence (`joblib.dump`, `log_model`, checkpointing); evaluation,
  metrics, scoring, error analysis; inference / serving over a
  pre-trained model; pure EDA with no estimator downstream;
  library-choice questions with no concrete declaration in play (defer
  to `data-science-python-stack`).

  HOW TO USE: consult before the first declarative line and on every
  structural edit (added/swapped step, changed input columns, changed
  estimator family). Don't re-consult for cosmetic edits (renames,
  formatting). For deep-learning declarations specifically, read the
  relevant `references/*.md` inside this skill. For library mechanics,
  defer to the sibling skills rather than inlining detail here — in
  particular, **invoke `skrub-api` whenever you need to recall a
  skrub symbol (DataOps `.skb` methods, `skrub.var`, joiners, column
  selectors, etc.), and `sklearn-api` whenever you need to pick a
  scikit-learn estimator / transformer / utility, confirm its import
  path, or check a signature**. Don't guess names from memory.
---

# Build ML Pipeline (Declaration)

Declarative shape of a Python ML pipeline from data source to predictor.

> **Companion skill (planned): `review-ml-pipeline`** — methodological
> review of an existing declaration (leakage audit, statelessness check,
> step ordering, scope creep). When `review-ml-pipeline` flags a
> problem, return here to fix it: this skill owns the *how* of
> corrections; the review skill owns the *what's wrong*.

## Scope

- **In scope:** how the pipeline *object* is composed in Python — data
  source wiring, preprocessing/feature steps, estimator at the tail, and
  the composition objects that hold them together.
- **Out of scope:** fitting, splitting, tuning, persisting, evaluating —
  those have (or will have) their own skills.

## Core rules

1. **Skrub DataOps is the pipeline entry point.** Declare the pipeline
   as a skrub DataOps graph rooted at one or more `skrub.var(...)`
   calls — not as a bare `sklearn.Pipeline`. See
   https://skrub-data.org/stable/data_ops.html. The `skrub.X(...)` /
   `skrub.y(...)` shortcuts are **not** acceptable roots; see rule 2
   for why. **Consult the `skrub-api` skill** to confirm the symbol
   you want exists and matches the signature you're about to write —
   don't guess from memory.

2. **Bind the source identifier; load inside the graph; mark X / y at
   the split.** When declaring a *new* pipeline, the root
   `skrub.var(name, value=...)` binds a **source identifier** — a
   file path, URL, table name, query — and the loader is the first
   `.skb.apply_func`. Inside the graph, derive X and y, then apply
   `.skb.mark_as_X()` / `.skb.mark_as_y()` at the split.

   ```python
   path = skrub.var("path", "data/train.parquet")
   data = path.skb.apply_func(load_parquet)

   X = data.drop(["id", "target"], axis=1).skb.mark_as_X()
   y = data["target"].skb.mark_as_y()

   X = X.skb.apply_func(feature_engineering_step)
   predictions = X.skb.apply(predictor, y=y)
   ```

   The env-dict at fit / cross-validate time is then one binding per
   source (`learner.fit({"path": "data/test.parquet"})`); swapping
   data sources is one string change.

   **Cross-validation metadata at the X marker.** If the data has
   group structure (subjects, sessions, customer IDs, repeated
   measures) or temporal ordering, attach the relevant column at
   the X marker via `.skb.mark_as_X(split_kwargs={...})`. The keys
   in `split_kwargs` map directly to the keyword arguments of the
   future cross-validator's `split(X, y, **split_kwargs)` (e.g.
   `groups`). See [`skrub.DataOp.skb.mark_as_X`][markx].

   ```python
   X = data.drop([...]).skb.mark_as_X(
       split_kwargs={"groups": data["customer_id"]},
   )
   ```

   **Ask the user when you can't tell from the data alone** whether
   such structure exists — name the suspect columns (anything ending
   in `_id`, columns called `subject` / `session` / `region`, or
   any `date` / `timestamp` for temporal ordering) and ask whether
   to wire them. Do not silently leave `split_kwargs` empty when
   group structure is plausible — that produces optimistic
   evaluations downstream. Choosing the splitter itself is owned by
   `evaluate-ml-pipeline`; this skill only ensures the metadata is
   wired in.

   [markx]: https://skrub-data.org/stable/reference/generated/skrub.DataOp.skb.mark_as_X.html

   The `skrub.X(...)` / `skrub.y(...)` shortcuts are not acceptable
   roots: per `skrub-api` → `data_ops.md` they are literally
   `var("X", value).skb.mark_as_X()` and `var("y", value).skb.mark_as_y()`,
   which bakes in the variable name and the marker at the root and
   forces a pre-loaded, pre-split binding.

   When *editing* an existing pipeline that already binds materialized
   data (or uses the shortcuts), do not auto-rewrite. Surface the
   source-bound alternative and ask whether to refactor.

   Full catalogue (encouraged / discouraged / OK-but-offer-an-upgrade):
   `references/source-binding.md`.

3. **Every data modification is either a function or a
   sklearn-compatible estimator. Nothing else.** Two ways to attach it
   to the graph (via the `.skb` accessor):
   - `.skb.apply_func(fn)` — wraps a callable that transforms data.
   - `.skb.apply(estimator)` — wraps any scikit-learn-compatible
     estimator (a transformer in the middle, or the final predictor).

4. **Stateless → function. Stateful → estimator.** This is the *only*
   decision rule for picking between `apply_func` and `apply`:

   - **Stateless** — output for a row depends only on that row (and
     constants). No information borrowed across rows of the dataset.
     Examples: parsing a date column, dtype casts, `log1p`, substring
     extraction, ratio of two columns, row-wise arithmetic.
     → write a plain `def fn(X): ...` and attach with
     `.skb.apply_func`.

   - **Stateful** — the transform needs a statistic / vocabulary /
     learned parameters fitted on the **training** data and re-applied
     unchanged to the **test** data. Examples: mean/median imputation,
     standard scaling, one-hot or target encoding, PCA, any predictor.
     → use a sklearn estimator (built-in when one exists; otherwise a
     custom `BaseEstimator` / `TransformerMixin` subclass exposing
     `fit` + `transform` / `predict`) and attach with `.skb.apply`.
     **Before naming the estimator, consult the `sklearn-api` skill**
     to confirm the import path and signature — don't guess.

   If a step would silently learn from the test set when called as a
   function, it is stateful — promote it to an estimator.

5. **Leakage rule.** Any computation that uses statistics learned from
   the data (means, medians, quantiles, vocabularies, target
   distribution) MUST be stateful. Calling such a computation as a
   plain function over the whole frame leaks the test set into
   training. Classic traps to call out by name:
   - target encoding (the encoder must `fit` on training y only),
   - target-aware or quantile-based imputation,
   - quantile binning / `KBinsDiscretizer(strategy="quantile")`,
   - any `OrdinalEncoder` / `LabelEncoder` whose categories come from
     the full dataset rather than from `fit` on training only,
   - vocabulary-building text tokenizers, TF-IDF, IDF weights.

   Litmus test: would this output change if I called it on the
   training subset alone vs. on the whole frame? If yes → stateful →
   `.skb.apply` with an estimator, never `.skb.apply_func` with a
   function.

## Decision flow for a new step

1. Does the operation only need the current row (and constants)?
   → **stateless** → pure Python function + `.skb.apply_func`.
2. Otherwise it must learn from training data and reapply on test data.
   → **stateful** → sklearn-compatible estimator + `.skb.apply`.

## Common patterns

A short catalogue of how to express the recurring shapes of a complex
pipeline within the skrub DataOps graph. Look up exact symbols in
`skrub-api` / `sklearn-api`; the patterns below tell you *which*
shape applies, not the precise signature.

1. **Heterogeneous columns (skrub answer to `ColumnTransformer`).**
   Use skrub column selectors with the `cols=` argument of
   `.skb.apply` to apply a transformer to a column subset. One
   `.skb.apply(...)` per group (numeric / string / categorical)
   instead of building a `ColumnTransformer`.

2. **Default starting point for tabular data.** Reach for
   `skrub.tabular_learner(...)` (or `TableVectorizer` + estimator)
   first. Specialize column-by-column only when the default is
   insufficient.

3. **Multi-table inputs.** Declare each input table as its own
   `skrub.var(...)`. Join with skrub's `Joiner` / `AggJoiner` /
   `MultiAggJoiner` via `.skb.apply(...)`. The graph holds the join
   plan deterministically across train and test.

4. **Meta-estimator at the tail.** `StackingClassifier`,
   `CalibratedClassifierCV`, `TransformedTargetRegressor`, etc., are
   regular sklearn estimators — wrap your predictor first, then
   attach the wrapped object with `.skb.apply` as the final step.

5. **Mark hyperparameter knobs in place.** Wrap values you want the
   tuning skill to search over with `skrub.choose_from` /
   `choose_int` / `choose_float` / `optional` directly inside the
   declaration. Don't import `GridSearchCV` here — the tuning skill
   owns search; this skill only exposes the knobs.

6. **Custom sklearn transformer.** Author one only when (a) no
   built-in fits and (b) the operation is stateful. Subclass
   `BaseEstimator` + `TransformerMixin`, implement `fit(self, X,
   y=None)` to learn state and `transform(self, X)` to apply it; add
   `get_feature_names_out` if downstream consumers need feature
   names. For a stateless op, write a function and use
   `.skb.apply_func` — don't author a transformer.

## Companion skills (read for the *how*)

This skill defines the shape and the stateless/stateful rule. For the
mechanics of the libraries themselves, defer to:

- **`sklearn-api`** — authoritative lookup of scikit-learn's public
  API: what's exported from `sklearn.X`, class/function signatures,
  one-line summaries. **Invoke whenever you need to:**
  - pick the right estimator / transformer / metric / utility for the
    stateful step,
  - confirm an import path (`sklearn.preprocessing.X` vs.
    `sklearn.compose.X`),
  - check a constructor signature or parameter name before writing
    the call,
  - verify that a symbol is part of the public API at all.
  Don't guess sklearn names from memory — consult the skill first.
- **`skrub-api`** — authoritative lookup of skrub's public API:
  top-level estimators, joiners, the DataOps lazy-pipeline framework
  (`.skb` accessor methods, `skrub.var`), column selectors, datasets,
  configuration. **Invoke whenever you need to:**
  - confirm a DataOps method exists on `.skb` and recall its
    signature (e.g. `.skb.apply`, `.skb.apply_func`, `.skb.eval`,
    `.skb.cross_validate`, ...),
  - pick a skrub estimator / joiner / column selector for the
    pipeline,
  - find the import path of a skrub symbol,
  - check that a symbol is part of the public API at all.
  Don't guess skrub names from memory — consult the skill first.
- **`evaluate-ml-pipeline`** — methodology for evaluating the
  declared pipeline: `skore.evaluate` as the entry point,
  cross-validator selection, metric defaults, report routing.
  **Defer all evaluation / cross-validation / metric decisions to
  it** — this skill stops at the declared object. Note the contract
  with rule 2's `split_kwargs`: structural metadata wired in here is
  what the evaluate skill consumes downstream.
- **Deep-learning declarations** — PyTorch / Lightning / Keras shapes
  that plug in as the predictor. → `references/*.md` inside this
  skill (TBD).
