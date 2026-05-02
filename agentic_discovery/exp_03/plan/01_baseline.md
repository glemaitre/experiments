# 01_baseline

## Question / hypothesis

How well does a default skrub `tabular_learner`, fed the multi-table join (historical_volume × price/promo × weather × events × demographics × industry × soda) keyed on Agency-SKU-month, forecast the next month's hectoliter volume — and which side-tables already carry signal before any feature engineering?

## Motivation

- **Sourcing strategy:** n/a — bootstrap, baseline forced by workspace defaults
- **Source(s):**
  - workspace defaults (per `iterate-ml-experiment` § 0): tabular regression → skrub `tabular_learner`; splitter default normally `KFold`, **overridden here** to a time-ordered cross-validator because the test cut is a single held-out future month (`data/test_*/volume_forecast.csv` is Jan'18).
  - `data/README.md` — Stallion & Co. demand-forecasting brief (panel of agency-SKU time series, monthly granularity, Jan 2013 → Dec 2017).
- **Why this matters:** establishes the floor metric every later experiment is judged against, and tells us *cheaply* (no feature engineering) whether the side-tables are even useful when joined naively. Picking RMSE on the time-ordered split aligns the cross-validation cut with how Jan'18 will actually be scored.

## Method

- **Files touched:** `src/beeristan/data.py`, `src/beeristan/pipeline.py`, `src/beeristan/evaluate.py`, `experiments/01_baseline.py`.
- **Change versus baseline:** there is no previous experiment — this *is* the baseline.
  - **`data.py`**: read all train CSVs (unzip `price_sales_promotion.csv.zip`) with polars; build a single panel `(Agency, SKU, YearMonth)` table by left-joining historical volume against price/promo, weather (Agency-month), events (month), demographics (Agency, yearly), industry volumes (month), industry soda (month). Target `y` = `Volume`. Mark `X` with skrub's X marker, attach `times=YearMonth` as the time key.
  - **`pipeline.py`**: `skrub.tabular_learner("regressor")` — the stack default for tabular regression. No custom feature work in the baseline.
  - **`evaluate.py`**: `splitter` = a sklearn time-ordered cross-validator (e.g. `TimeSeriesSplit`); concrete choice deferred to `evaluate-ml-pipeline` when the script is written.
  - **`experiments/01_baseline.py`**: open `skore.Project(workspace="reports", name="beeristan", mode="local")`, call `skore.evaluate(learner, data={"X": X, "y": y}, splitter=splitter)`, persist under key `"01_baseline"`.
- **Out of scope for this experiment:** lag / rolling features (B2), calendar/event one-hot beyond raw join (B3), hierarchical reconciliation (B4), SKU recommendation for cold-start agencies (B1), hyperparameter tuning, model selection beyond `tabular_learner`'s defaults.

## Risks / things that could invalidate the result

- **Time leakage if the wrong splitter is used.** The IID `KFold` default would let the model see future months when scoring past ones, inflating the headline metric well beyond what Jan'18 will produce. Forcing a time-ordered split is the guard-rail; the exact cross-validator (`TimeSeriesSplit`, expanding-window, blocked) is delegated to `evaluate-ml-pipeline`.
- **Join cardinality / silent row explosion.** Demographics is yearly (one row per Agency-year), weather is Agency-month, industry tables are month-only. A wrong join key (e.g. forgetting to broadcast the year-only demographics across months) silently inflates or deflates rows; a sanity check on `len(X) == len(historical_volume)` after joins is mandatory.
- **Test-set agencies and SKUs may not all be in train.** Agency06 and Agency14 are explicitly cold-start; if they appear in the Jan'18 forecast file they cannot be predicted by a model trained only on observed (Agency, SKU) pairs. Need to verify the train/test agency-SKU overlap before reading the headline RMSE — it bounds achievable performance.
- **Unit / scale skew.** Volume is in hectoliters across 5 years and ~hundreds of agency-SKU pairs; per-series scales differ by orders of magnitude. RMSE will be dominated by large-volume series. Worth flagging — a follow-up may want a per-series-normalized metric — but not in scope for the baseline.
- **`tabular_learner` default may include a target-aware encoder** (e.g. target encoding of high-cardinality categoricals). Combined with `KFold` it'd leak; with a time-ordered splitter it's safer but worth confirming when the script is written via `build-ml-pipeline`.

## Status

- **State:** done
- **Approved by user on:** 2026-05-02
- **Headline result:** R² 0.934 ± 0.021, RMSE 695 ± 130 hL, MAE 340 ± 64 hL (16-fold walk-forward, min_train=12 months, interval=3 months; fit ~1.7 s/fold). MAPE is reported as ~1.3e16 — diagnostic of zero/near-zero `Volume` rows rather than a usable score.
- **Implication for next iteration:** baseline is unexpectedly strong (R² ≈ 0.93) which is worth pressure-testing before celebrating — the join already pulls in side-tables that carry month-level signal (industry volume, weather, calendar) and the model may be exploiting them more than per-series demand dynamics. Two natural next moves: (i) **methodology audit** of the splitter/feature set to confirm there is no time leakage (e.g., a side-table value derived from the test month), and (ii) once the audit clears, **lag/rolling features at the Agency-SKU level (B2)** to give the model true within-series memory.
