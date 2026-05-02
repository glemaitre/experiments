# 04_sku_recommendation

## Question / hypothesis

Which two SKUs should be recommended to the two cold-start agencies (`Agency_06`, `Agency_14`)? Trained on the existing 350 `(Agency, SKU)` series, can a regression model that uses *only cold-start-safe features* (demographics, weather, calendar events, industry/soda volumes — **no lag features**) generalize to unseen agencies via their shared numerical signals, and produce a defensible per-agency SKU ranking by predicted annual volume?

## Motivation

- **Sourcing strategy:** backlog:B1
- **Source(s):**
  - `PLAN.md` Backlog row B1 — secondary task carried since the dataset README, parked because it is a different problem class (ranking under cold start vs. panel regression).
  - `data/test_8uviCCm/sku_recommendation.csv` — 4-row file: 2 SKU slots each for `Agency_06` and `Agency_14`, with the SKU column empty.
  - Data check confirmed both agencies are **true cold-start**: each has 1 row in `demographics.csv` and 60 rows in `weather.csv`, but 0 rows in `historical_volume.csv` — they have demographic and weather context, no sales history.
- **Why this matters:** closes the secondary deliverable from the dataset brief; uses the regression infrastructure already in place; produces a small, reviewable output (4 recommendations) without requiring a fundamentally new ML problem class.

## Method

- **Files touched:** `src/beeristan/data.py` (add `load_cold_start_grid` — builds a prediction frame for cold-start agencies × candidate SKUs × 2017 months by joining the same side-tables `load_panel` already uses); `experiments/04_sku_recommendation.py` (new). `pipeline.py`, `evaluate.py`, `features.py` are not touched.
- **Pipeline:** `build_learner(feature_steps=[])` — same architecture as `01_baseline`. **Lag features are deliberately excluded** because they are undefined for cold-start agencies (no history). The cold-start-safe learner is therefore a re-use of the baseline shape, which makes its expected error on existing agencies an upper bound on its accuracy for cold-start ones.
- **Procedure in the experiment script:**
  1. Fit the cold-start-safe learner on the full training panel via `learner.fit({"data_dir": "data/train_OwBvO8W"})` (no holdout; we are not measuring CV here, that is `01_baseline`'s job).
  2. Build a cold-start prediction frame: `Agency × SKU × Date` cross-product over `{Agency_06, Agency_14} × all 25 train SKUs × 12 months of 2017 (Jan–Dec)`. Join the same six side-tables as `load_panel` so the row schema matches what the model was fit on.
  3. Predict `Volume` for every row in that frame.
  4. Aggregate to per-`(Agency, SKU)`: mean predicted volume across the 12 months of 2017.
  5. For each cold-start agency, rank the 25 SKUs by mean predicted volume; take the top 2.
  6. Write the four recommendations to `reports/04_sku_recommendation.csv` with columns `Agency, SKU, predicted_volume_mean_2017, rank` (ranks 1 and 2 per agency).
- **Skore report:** the experiment is a one-shot fit + predict, not a cross-validated evaluation. The metric of interest (ranking quality) has no ground truth in this dataset. **No `skore.evaluate` call, no `project.put`** for this experiment — it produces a `reports/04_sku_recommendation.csv` artifact instead, and `PLAN.md` History records "see CSV" as the headline.
- **Out of scope for this experiment:** SKU-level features (the SKU column is treated as a categorical only); agency similarity / k-NN approaches; methodological validation by leave-one-out agency holdout (B10, see backlog spillover); month-specific recommendations (we average across 2017 months); demographic-distribution check that `Agency_06`/`Agency_14` are within the train range (B11, see backlog spillover).

## Risks / things that could invalidate the result

- **Categorical encoding of unseen `Agency_06`/`Agency_14`.** Skrub's `ToCategorical` followed by `HistGradientBoostingRegressor`'s native categorical handling routes unknown categories to the missing-value direction at each split. So the model effectively *ignores agency identity* for the cold-start rows and predicts purely from the numerical features (demographics, weather, calendar, industry). The recommendation is therefore "what would the model expect of an agency with these demographics/weather", not "what would `Agency_06` specifically sell". This is the strongest methodological caveat — flagged so the result is read in that frame.
- **Demographics distribution.** If `Agency_06`/`Agency_14`'s `Avg_Population_2017` or `Avg_Yearly_Household_Income_2017` lie outside the range seen in training, the tree will clip to the boundary leaf and the predictions become extrapolations the model cannot calibrate. The script will print min/max of those features in train vs. the cold-start values as a sanity check; if they extrapolate, the result is recorded with that caveat rather than withdrawn.
- **SKU coverage is uneven.** Not all 25 SKUs appear for all agencies in train; some SKUs are likely regional. The model will reflect that statistically, but the recommendation may propose SKUs that, in business terms, would not be carried by the new agency for non-modelable reasons (distribution agreements, brand exclusivity, etc.). The experiment can't see those constraints.
- **No ground-truth evaluation.** There is no held-out cold-start agency in this dataset to measure recommendation quality against. The recommendation is a methodological output; the user judges defensibility, not a metric.
- **Mean-over-months collapses seasonality.** A SKU strongly demanded in summer and absent in winter would be ranked the same as a steady year-round SKU with the same annual mean. If the user prefers a peak-month or distributional ranking, that is a follow-up.

## Status

- **State:** done
- **Approved by user on:** 2026-05-02
- **Headline result:** see `reports/04_sku_recommendation.csv`. Top-2 SKUs per cold-start agency: `Agency_06` → SKU_01 (7,850 hL/mo), SKU_04 (3,191 hL/mo); `Agency_14` → SKU_01 (8,367 hL/mo), SKU_04 (2,295 hL/mo). Demographic sanity: both agencies' `Avg_Population_2017` (2.0M / 2.1M) and `Avg_Yearly_Household_Income_2017` (204k / 228k) are inside the train range — recommendations are interpolations, not extrapolations.
- **Implication for next iteration:** the model is *not* just returning the global top-2 (which would be SKU_01, SKU_02) — it picks SKU_04 (#5 globally by mean volume) over SKU_02 / SKU_03 / SKU_05 for these high-population, high-income agencies. So demographics + weather + calendar are doing some real differentiation, despite agency-identity being routed to the missing direction. Both cold-start agencies get the *same* top-2 ranking (their demographics are similar), differing only in the predicted magnitudes. Two natural follow-ups: (1) **B10** — leave-one-out methodological validation; this is now the only credible internal check for whether the recommendation generalizes, since the dataset offers no cold-start ground truth. (2) the predicted magnitudes (~8k hL/mo for SKU_01) are large vs. the global SKU_01 mean (3.2k hL/mo); consistent with high-population agencies but worth a slice-by-population sanity check on the trained model.
