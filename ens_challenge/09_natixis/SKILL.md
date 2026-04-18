# skore API Reference

`skore` is a scikit-learn companion for structured model evaluation, comparison, and diagnostics.

## Installation

```bash
pip install skore
```

## Imports

```python
import skore
# or selectively:
from skore import evaluate, compare, train_test_split, EstimatorReport, CrossValidationReport, ComparisonReport
from skore import configuration, Project, login, show_versions
```

---

## Core Functions

### `evaluate()`

One-line estimator evaluation. Returns different report types depending on inputs.

```python
skore.evaluate(
    estimator,                    # BaseEstimator or list[BaseEstimator]
    X=None,                       # ArrayLike or list[ArrayLike | None] or None
    y=None,                       # ArrayLike or None
    data=None,                    # dict or None (for skrub DataOp)
    *,
    splitter=0.2,                 # float | int | str | CrossValidator | Generator
    pos_label=None,               # int | float | bool | str | None
    n_jobs=None,                  # int or None
) -> EstimatorReport | CrossValidationReport | ComparisonReport
```

- `splitter=0.2` (float) -> train/test split -> `EstimatorReport`
- `splitter=5` (int) -> 5-fold CV -> `CrossValidationReport`
- `splitter="prefit"` -> use already-fitted estimator -> `EstimatorReport`
- `splitter=StratifiedKFold(...)` -> sklearn CV splitter -> `CrossValidationReport`
- `estimator=[est1, est2]` -> multiple estimators -> `ComparisonReport`

```python
# Binary classification
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
report = skore.evaluate(LogisticRegression(max_iter=10_000), X, y, splitter=0.2, pos_label=1)

# Cross-validation
cv_report = skore.evaluate(LogisticRegression(), X, y, splitter=5, pos_label=1)

# Comparison of multiple estimators
from sklearn.ensemble import RandomForestClassifier
comparison = skore.evaluate([LogisticRegression(), RandomForestClassifier()], X, y, splitter=0.2)

# Regression
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
report = skore.evaluate(Ridge(), X, y, splitter=0.2)
```

### `compare()`

Consolidate existing reports into a `ComparisonReport`.

```python
skore.compare(
    reports,    # list[EstimatorReport] | dict[str, EstimatorReport]
                # | list[CrossValidationReport] | dict[str, CrossValidationReport]
    *,
    n_jobs=None,
) -> ComparisonReport
```

```python
report1 = skore.evaluate(LogisticRegression(), X, y, splitter=0.2)
report2 = skore.evaluate(RandomForestClassifier(), X, y, splitter=0.2)
comparison = skore.compare([report1, report2])
# or with custom names:
comparison = skore.compare({"LR": report1, "RF": report2})
```

### `train_test_split()`

Enhanced drop-in replacement for `sklearn.model_selection.train_test_split` with diagnostic warnings.

```python
skore.train_test_split(
    *arrays,                      # positional ArrayLike args
    X=None,                       # ArrayLike or None
    y=None,                       # ArrayLike or None
    test_size=None,               # int | float | None
    train_size=None,              # int | float | None
    random_state=None,            # int | RandomState | None
    shuffle=True,                 # bool
    stratify=None,                # ArrayLike or None
    as_dict=False,                # bool
    **keyword_arrays,             # named ArrayLike args
) -> list | dict
```

```python
# Standard usage (returns list like sklearn)
X_train, X_test, y_train, y_test = skore.train_test_split(X, y, test_size=0.2)

# Dict mode
splits = skore.train_test_split(X=X, y=y, test_size=0.2, as_dict=True)
# splits = {"X_train": ..., "X_test": ..., "y_train": ..., "y_test": ...}
```

---

## Report Classes

### `EstimatorReport`

Report for a single fitted estimator on a train/test split.

```python
skore.EstimatorReport(
    estimator,                    # BaseEstimator or skrub.DataOp
    *,
    fit="auto",                   # "auto" | bool
    X_train=None, y_train=None,   # ArrayLike or None
    X_test=None, y_test=None,     # ArrayLike or None
    train_data=None, test_data=None,  # dict or None (for skrub)
    pos_label=None,               # int | float | bool | str | None
)
```

**Attributes:**
- `estimator_` -> fitted BaseEstimator
- `estimator_name_` -> str
- `fit_time_` -> float or None
- `X_train`, `y_train`, `X_test`, `y_test` -> data arrays
- `ml_task` -> str (e.g. "binary-classification", "regression")

**Methods:**
- `cache_predictions(data_source="test")` -> pre-compute predictions
- `clear_cache()` -> clear cached predictions
- `get_state()` -> dict (serializable state)
- `EstimatorReport.from_state(state)` -> restore from state
- `diagnose(*, ignore=None)` -> `DiagnosticDisplay`
- `add_checks(checks)` -> register custom `Check` objects

**Accessors:**
- `report.metrics` -> metrics accessor
- `report.inspection` -> model inspection accessor
- `report.data` -> data analysis accessor

### `CrossValidationReport`

Report for cross-validated estimator evaluation.

```python
skore.CrossValidationReport(
    estimator,                    # BaseEstimator
    X=None, y=None,               # ArrayLike or None
    data=None,                    # dict or None
    pos_label=None,               # int | float | bool | str | None
    splitter=None,                # int | CrossValidator | Generator | None
    n_jobs=None,                  # int or None
)
```

**Attributes:**
- `estimator_` -> cloned BaseEstimator
- `estimator_name_` -> str
- `estimator_reports_` -> list[EstimatorReport] (one per fold)

**Accessors:** same as `EstimatorReport` (`.metrics`, `.inspection`, `.data`)

### `ComparisonReport`

Report for comparing multiple estimators or CV reports.

```python
skore.ComparisonReport(
    reports,    # list[EstimatorReport] | dict[str, EstimatorReport]
                # | list[CrossValidationReport] | dict[str, CrossValidationReport]
    *,
    n_jobs=None,
)
```

**Attributes:**
- `reports_` -> dict[str, EstimatorReport | CrossValidationReport]

**Accessors:** `.metrics`, `.inspection` (no `.data`)

---

## Accessors

### `.metrics` accessor

Available on all report types.

#### Summarize all metrics

```python
report.metrics.summarize(
    *,
    data_source="test",     # "test" | "train" | "both"
    metric=None,            # str | list[str] | None
) -> MetricsSummaryDisplay
```

```python
report.metrics.summarize().frame()
report.metrics.summarize(data_source="both").frame(favorability=True)
report.metrics.summarize(metric=["accuracy", "precision"]).frame()
```

#### Individual scalar metrics

Classification metrics (available when ml_task is classification):

```python
report.metrics.accuracy(*, data_source="test") -> float
report.metrics.precision(*, data_source="test", average=None) -> float | dict
report.metrics.recall(*, data_source="test", average=None) -> float | dict
report.metrics.roc_auc(*, data_source="test", average=None, multi_class="ovr") -> float | dict
report.metrics.log_loss(*, data_source="test") -> float
report.metrics.brier_score(*, data_source="test") -> float
```

Regression metrics (available when ml_task is regression):

```python
report.metrics.r2(*, data_source="test", multioutput="raw_values") -> float | list
report.metrics.rmse(*, data_source="test", multioutput="raw_values") -> float | list
```

Timing metrics:

```python
report.metrics.fit_time(cast=True) -> float | None
report.metrics.predict_time(*, data_source="test", cast=True) -> float | None
report.metrics.timings() -> dict
```

#### Curve displays

```python
report.metrics.roc(*, data_source="test") -> RocCurveDisplay                          # classification
report.metrics.precision_recall(*, data_source="test") -> PrecisionRecallCurveDisplay  # classification
report.metrics.confusion_matrix(*, data_source="test") -> ConfusionMatrixDisplay       # classification
report.metrics.prediction_error(*, data_source="test", subsample=1_000, seed=None) -> PredictionErrorDisplay  # regression
```

#### Add custom metrics

```python
report.metrics.add(
    metric,                       # str | sklearn scorer | callable(y_true, y_pred) -> float
    *,
    name=None,                    # str or None
    response_method="predict",    # str or list[str]
    greater_is_better=True,       # bool
    **kwargs,                     # extra kwargs passed to score function
)
```

```python
from sklearn.metrics import make_scorer, f1_score
report.metrics.add("f1")                                        # by sklearn name
report.metrics.add(make_scorer(f1_score, average="weighted"))   # sklearn scorer
report.metrics.add(lambda y_true, y_pred: (y_true == y_pred).mean(), name="custom_acc")  # callable
```

### `.inspection` accessor

Available on all report types.

```python
# Linear models (estimators with .coef_)
report.inspection.coefficients() -> CoefficientsDisplay

# Tree-based models (estimators with .feature_importances_)
report.inspection.impurity_decrease() -> ImpurityDecreaseDisplay

# Any model (model-agnostic)
report.inspection.permutation_importance(
    *,
    data_source="test",           # "test" | "train"
    at_step=0,                    # int | str (pipeline step)
    metric=None,                  # MetricLike | list | dict | None
    n_repeats=5,                  # int
    max_samples=1.0,              # float
    n_jobs=None,                  # int or None
    seed=None,                    # int or None
) -> PermutationImportanceDisplay
```

### `.data` accessor

Available on `EstimatorReport` and `CrossValidationReport` (not `ComparisonReport`).

```python
report.data.analyze(
    data_source="both",           # "train" | "test" | "both"
    with_y=True,                  # bool
    subsample=None,               # int or None
    subsample_strategy="head",    # "head" | "random"
    seed=None,                    # int or None
) -> TableReportDisplay
```

---

## Display Protocol

All displays follow a common protocol:

```python
display.plot(**kwargs) -> matplotlib.figure.Figure   # render a plot
display.frame(**kwargs) -> pd.DataFrame              # get underlying data
display.set_style(*, policy="update", **kwargs)      # customize plot styling
display.help()                                       # show available attributes
```

### MetricsSummaryDisplay

```python
display.frame(
    *,
    aggregate=("mean", "std"),    # ("mean", "std") | "mean" | "std" | None (CV only)
    favorability=False,           # bool - show arrow indicators
    flat_index=False,             # bool - flatten MultiIndex
) -> pd.DataFrame
```

### RocCurveDisplay

```python
display.plot(*, subplot_by="auto", plot_chance_level=True, despine=True, label=<default>)
display.frame(with_roc_auc=False, label=<default>)
```

### PrecisionRecallCurveDisplay

```python
display.plot(*, subplot_by="auto", despine=True, label=<default>)
display.frame(with_average_precision=False, label=<default>)
```

### ConfusionMatrixDisplay

```python
display.plot(*, normalize=None, threshold_value="default", subplot_by="auto")
display.frame(*, normalize=None, threshold_value="default")
```

### PredictionErrorDisplay

```python
display.plot(*, subplot_by="auto", kind="residual_vs_predicted", despine=True)
display.frame()
```

### CoefficientsDisplay

```python
display.plot(*, include_intercept=True, subplot_by="auto", select_k=None, sorting_order=None)
display.frame(*, aggregate=("mean", "std"), include_intercept=True, select_k=None, sorting_order=None)
```

### ImpurityDecreaseDisplay

```python
display.plot(*, select_k=None, sorting_order=None)
display.frame(*, aggregate=("mean", "std"), select_k=None, sorting_order=None)
```

### PermutationImportanceDisplay

```python
display.plot(*, metric=None, subplot_by="auto", select_k=None, sorting_order=None)
display.frame(*, metric=None, aggregate=("mean", "std"), level="splits", select_k=None, sorting_order=None)
```

### TableReportDisplay

```python
display.plot(*, x=None, y=None, hue=None, kind="dist", top_k_categories=20)
display.frame(*, kind="dataset")  # "dataset" | "top-associations"
```

---

## Diagnostics

```python
# Run diagnostics on any report
diag = report.diagnose(*, ignore=None)  # list[str] or None
diag.frame()   # -> DataFrame with columns: code, title, explanation, documentation_url
diag.issues    # -> dict[str, dict]

# Ignore specific checks globally
skore.configuration.ignore_checks = ["SKD001"]

# Or temporarily
with skore.configuration(ignore_checks=["SKD001"]):
    report.diagnose()

# Register custom checks
report.add_checks([my_check])
```

### Check Protocol

```python
class Check(Protocol):
    code: str                # e.g. "SKD001"
    title: str               # short description
    report_type: ReportType  # "estimator" | "cross-validation" | "comparison-estimator" | "comparison-cross-validation"
    docs_url: str | None
    def check_function(self, report) -> str | None:  # return explanation string if issue found, None if OK
        ...
```

---

## TrainTestSplit (as splitter)

sklearn-compatible single-split cross-validator.

```python
splitter = skore.TrainTestSplit(
    test_size=0.2,
    train_size=None,
    random_state=0,
    shuffle=True,
    stratify=None,
)
splitter.get_n_splits()  # -> 1
splitter.split(X, y)     # -> generator yielding (train_indices, test_indices)
```

---

## Configuration

```python
from skore import configuration

# Read/set globally
configuration.show_progress = False
configuration.plot_backend = "matplotlib"
configuration.ignore_checks = ["SKD001"]

# Temporary override via context manager
with configuration(show_progress=False):
    ...
with configuration(plot_backend="plotly"):
    ...
```

---

## Project

Store and retrieve reports with different backends.

```python
from skore import Project, login

# Hub mode (requires login)
login(mode="hub")
project = Project("my-project", mode="hub")

# Local mode
project = Project("my-project", mode="local")

# MLflow mode
project = Project("my-project", mode="mlflow")

# Store and retrieve
project.put("my_key", report)
restored = project.get("my_id")
summary = project.summarize()

# Delete
Project.delete("my-project", mode="local")
```

---

## Types

```python
DataSource = Literal["test", "train"]
MLTask = Literal[
    "binary-classification", "multiclass-classification",
    "multioutput-binary-classification", "multioutput-multiclass-classification",
    "regression", "multioutput-regression", "clustering", "unknown",
]
ReportType = Literal["estimator", "cross-validation", "comparison-estimator", "comparison-cross-validation"]
Aggregate = Literal["mean", "std"] | Sequence[Literal["mean", "std"]]
MetricLike = str | Callable | sklearn.metrics._scorer._PredictScorer
```

---

## Common Patterns

```python
import skore
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# Quick evaluation
report = skore.evaluate(LogisticRegression(max_iter=10_000), X, y, pos_label=1)

# Metrics summary as DataFrame
report.metrics.summarize().frame()

# Individual metric
report.metrics.accuracy()

# Plot ROC curve
report.metrics.roc().plot()

# Get ROC data
report.metrics.roc().frame(with_roc_auc=True)

# Feature inspection (linear model)
report.inspection.coefficients().frame()
report.inspection.coefficients().plot()

# Permutation importance
report.inspection.permutation_importance().frame()
report.inspection.permutation_importance().plot()

# Data analysis
report.data.analyze().plot()
report.data.analyze().frame()

# Diagnostics
report.diagnose().frame()

# Cross-validation with aggregated metrics
cv_report = skore.evaluate(LogisticRegression(), X, y, splitter=5, pos_label=1)
cv_report.metrics.summarize().frame(aggregate=("mean", "std"))

# Compare models
comparison = skore.evaluate(
    [LogisticRegression(), RandomForestClassifier()], X, y, pos_label=1
)
comparison.metrics.summarize().frame()
comparison.metrics.roc().plot()
```
