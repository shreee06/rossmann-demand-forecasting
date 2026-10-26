# Model protocol (Step 3)

Written and committed **before any model was trained or any Step-3 result was computed**. Deviations must be added as
dated amendments at the bottom, never by silently editing earlier sections. No numbers appear here. Results go to
`evals/results/` and every configuration tried (including dropped ones) is appended to `evals/results/spec_log.csv`.

Builds on `docs/forecast_protocol.md` (task, metrics, folds, bootstrap, baselines). Unchanged from there: forecast task,
Open assumption, evaluated rows (open days with Sales > 0), the four metrics, the 8 backtest folds + VALIDATION,
store-clustered bootstrap (500 resamples, seed 20260101), the TEST lock (2015-06-20..2015-07-31 is never loaded).
`Customers` is never read (explicit column lists; the leakage tests cover the new modules).

Compute limits: `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` everywhere, LightGBM `num_threads=2`, nothing else
in parallel (at most 2 workers overall), memory watchdog on (exit below 1 GB available).

## 1. Design: one global direct multi-horizon LightGBM

One training row per **(store, cutoff c, target day t = c + h)**, `h = 1..42`, with `h` as a feature. The row exists only
if the store has a row on `t` with `Open = 1` (Open of target days is known in advance, as in the baselines). Training
rows additionally need `Sales > 0`.

- **Cutoffs.** Weekly, every **Friday** (all backtest/validation cutoffs are Fridays), from 2013-04-05 (first Friday with
  at least 91 days of history) to 2015-05-08 (end of TRAIN). Friday-only cutoffs mean the model is trained for the
  cutoff weekday it is evaluated on; day of week of the target is then a function of `h`. This is a limitation for
  use at other cutoff weekdays.
- **Per-fold training set.** For a fold with cutoff `c_f`, train only on rows with **target day `t <= c_f`** (the target
  was already observed at the fold cutoff). All evaluation rows of the fold (`c = c_f`, `t > c_f`) are never in its
  training set. Features of every row use data up to its own cutoff only, so the feature table is built once and
  filtered per fold.
- **Row subsample (compute budget, pre-registered).** Training rows are subsampled uniformly at random with a fixed seed
  (`20260101`), keeping **20%**. Adjacent horizons and weekly cutoffs are highly redundant; a measured benchmark on
  synthetic data showed the full row set was about 20 hours of compute. Evaluation rows and the early-stopping holdout
  are never subsampled. If the budget forces a change, the fraction may only be lowered via a dated amendment.
- **Target.** `log1p(Sales)`, objective L2, on open days with `Sales > 0` only. Predictions are `expm1(prediction)`.
  **Known bias:** the back-transformed prediction estimates roughly the conditional median, not the mean, so it
  under-forecasts totals (negative Bias is expected). The primary model is *not* corrected and Bias is reported with every
  result. A **secondary** variant applies a Duan smearing factor `s = mean(exp(log1p(y) − pred))` computed on the inner
  holdout (section 3) as `(1 + expm1(pred))·s − 1`; it is reported next to the primary model but is never used for
  selection.
- **No store-ID feature** (1,115-level categorical would mostly memorise stores); store identity enters only through
  history features and store attributes.

## 2. Features (all computable at the cutoff `c`; target day `t = c + h`)

Anything that depends on dates after `c` may only use information known in advance for the forecast horizon: calendar,
`StateHoliday`, `SchoolHoliday`, planned `Promo`, up to date `c + 42`. Nothing dated after `c + 42` is used, and no
`Sales` after `c`. Missing = NaN (LightGBM handles it).

| Group | Features |
|---|---|
| Horizon | `horizon` (1..42) |
| Calendar | `dow`, `day_of_month`, `week_of_year` (ISO), `month` |
| Holidays | `state_code` (0 none, 1 a, 2 b, 3 c; categorical), `school_flag`, `days_to_state`, `days_since_state`, `days_to_school`, `days_since_school`. State holiday dates = dates on which any store carries a non-zero `StateHoliday`; school flags are per store. "Days to next" only looks at holidays up to `c + 42`; "days since" only up to `t`. Both capped at 60 (60 = none in range). |
| Promo | `promo` (target day), `promo_m1`, `promo_p1` (day before / after), `promo_m7`, `promo_p7` (same weekday one week before / after). A neighbour dated after `c + 42` is NaN. `promo2_active` (store in Promo2, start week reached, target month in PromoInterval), `promo2_weeks` (weeks since Promo2 start). |
| Store attributes | `store_type`, `assortment` (categorical), `comp_distance`, `comp_open_months` (months between competition opening and `t`, signed, NaN if unknown) |
| History at `c` (observations = open days with Sales > 0 on dates `<= c`) | store `mean_28/56/91` and `median_28/56/91` of sales over the last 28/56/91 days; `swd_mean_4/8/12`: mean sales on the same weekday as `t` over the last 4/8/12 weeks (last K occurrences on or before `c`); `swp_mean_4/8/12`: the same restricted to days with the same Promo flag as `t` (the planned flag of `t` selects which of two precomputed means is used); `swp_n_12` (number of matched days); `lag7`, `lag14`: sales on the most recent same weekday at least one lag period back (`t − 7·ceil(h/7)`, `t − 14·ceil(h/14)`, the seasonal-naive source days); `days_since_closure` (days since the last closure on a non-Sunday, where closure = `Open = 0` or no row; capped at 200); `frac_missing_91` (share of the last 91 calendar days without a row), `frac_closed_91` (share with `Open = 0`), `n_obs_91`. |

`Customers` is forbidden as a feature and as an input to any feature.

**Categorical** in LightGBM: `store_type`, `assortment`, `state_code`.

## 3. Hyperparameters and early stopping

Fixed for everything: `learning_rate = 0.05`, `max_bin = 63`, `feature_fraction = 0.8`, `bagging_fraction = 0.5`,
`bagging_freq = 1`, `lambda_l2 = 1.0`, `seed = 20260101`, `num_threads = 2`, maximum 1500 rounds.

**Pre-registered grid** (4 configurations): `num_leaves ∈ {31, 63}` × `min_child_samples ∈ {50, 200}`.

**Number of trees, per (configuration, fold):** early stopping (patience 50, metric L2 on `log1p`) on an **inner holdout**:
the 4 latest Friday cutoffs `c' ∈ {c_f − 63, c_f − 56, c_f − 49, c_f − 42}` (the latest cutoffs whose full 42-day horizon
is observed by `c_f`), all their rows with `t <= c_f` (not subsampled). The inner training set is the subsampled rows
with `t <= c_f − 63` (**purged**: no inner-training target is later than the first holdout cutoff, so the holdout is
strictly "future" for the inner fit). The number of trees found is then used to **refit on all subsampled training rows of
the fold** (`t <= c_f`). The fold's evaluation rows are never used for stopping or selection.

**Selection:** the configuration with the lowest **mean-over-folds WAPE** on the 8 backtest folds. All 4 configurations x
8 folds are logged, with the stopped number of trees. The selected configuration is then frozen
(`evals/results/model_selected.json`) before anything touches VALIDATION. Tuned and scored on the same folds, so
backtest numbers are mildly optimistic; VALIDATION is the clean check.

## 4. Ablations (same folds, selected configuration, same early-stopping procedure)

| Ablation | What is removed / changed |
|---|---|
| no history | all History-at-`c` features |
| no promo | `promo`, `promo_m1/p1/m7/p7`, `promo2_active`, `promo2_weeks`, and `swp_mean_*`, `swp_n_12` (they use the Promo flag) |
| no holiday | `state_code`, `school_flag`, `days_to/since_state/school` |
| horizon fixed to 1 | `horizon` set to the constant 1 for all rows (the features themselves still depend on `h` through the target date; this shows what the explicit horizon feature adds) |

Each ablation is reported against the full model with a paired bootstrap on the WAPE difference.

## 5. Comparison with the best baseline

Full model vs **promo-aware weekday mean, K=12** (the Step-2 winner), recomputed with the same code on the same
evaluated rows (asserted identical). Metrics: RMSPE, WAPE, MASE, Bias with store-clustered bootstrap CIs, plus a
**paired bootstrap** on the WAPE difference (and the other metrics). Breakdowns as in Step 2: horizon bucket, promo vs
non-promo, store type, day of week, folds 2-5 vs folds 1, 6, 7, 8, and the stale/row-coverage diagnostics.

## 6. Prediction intervals

LightGBM quantile models (`alpha = 0.1` and `0.9`, objective `quantile`) on `log1p(Sales)` with the same features, the
**selected configuration** and the same inner-holdout early stopping (metric: pinball loss of that quantile). Quantiles
commute with monotone transforms, so `expm1` of the quantile prediction is a quantile of sales (no back-transform bias).
Crossing (`q10 > q90`) is counted per fold before being fixed by sorting the pair. Reported, with store-clustered
bootstrap CIs: empirical coverage of the 80% interval (target 0.80), average width relative to sales
(`Σ(hi − lo) / Σ y`), coverage by horizon bucket and by fold, share of misses below/above, mean pinball losses. If
coverage is poor, that is reported; **nothing is tuned for coverage on VALIDATION**. Split-conformal calibration is
**not attempted** in this step (the inner holdout is also used for early stopping, so it is not a clean calibration set).

## 7. Promotion analysis (association, not causal)

Scope: rows of TRAIN only (`Date <= 2015-05-08`), open days with `Sales > 0`, Monday-Friday (Promo is never 1 on
Saturdays/Sundays).

1. **Within-store regression of `log(Sales)` on `Promo`.** Fixed effects removed by alternating projections
   (Frisch-Waugh); coefficient and **store-clustered** SE from the cluster-robust sandwich formula; 95% CI; reported as
   `exp(β) − 1`. Specifications fixed in advance:
   - A: store + day-of-week + week-of-year fixed effects;
   - B: A + state-holiday and school-holiday indicators;
   - C: store x day-of-week + week-of-year fixed effects (+ the indicators of B);
   - D: B + `Promo` of the previous/next day and previous/next week (rows with any of these unknown are dropped) as a
     description of timing patterns around promo days.
   Also spec B by store type.
2. **Model-based comparison.** For each backtest fold's model and its evaluated Monday-Friday rows, predict with `promo`
   set to 1 and to 0, with the promo-matched history means switched accordingly and everything else fixed; variant (i)
   keeps the neighbouring-week flags as observed, variant (ii) sets `promo_m7 = promo_p7 = 1 − promo`. Report the
   average predicted ratio (geometric mean of row ratios), its spread (10th-90th percentile of row ratios and of
   store-level means), by store type, and bootstrap CIs over stores.

**What this can and cannot say.** Rossmann chose when promotions ran (and promo timing alternates by week), so every
estimate is an adjusted *association*. Nothing here says anything about profit, margin, cannibalisation of neighbouring
days/products, or what would happen if promotion timing changed.

## 8. Feature importance

On **backtest fold 8** (fixed in advance): gain importance of the fold's selected-configuration full model, and
**permutation importance** on the fold's evaluation rows (3 repeats, seed 20260101): increase of WAPE when one feature
(or one feature group) is shuffled across rows. Importances describe this model, not the world, and are not causal;
correlated features share credit.

## 9. Part B diagnostic: seasonal-naive over-forecasting on VALIDATION

Step 2 reported Bias of about +0.11 / +0.12 for seasonal naive 7 / 14 on VALIDATION. Pre-registered check (uses the
saved Step-2 forecasts; nothing in Step 3's model design depends on it):

- Split evaluated rows by (a) target day is a state holiday, (b) target day has `SchoolHoliday = 1` (and not (a)),
  (c) ordinary (neither); and by whether the **source day** of the forecast (`t − 7·ceil(h/7)` resp. `t − 14·ceil(h/14)`)
  was a state holiday or a school-holiday day. Compare Bias (`Σe / Σy`) with store-clustered bootstrap CIs, also bias by
  target date and by week. The same table is produced for the 8 backtest folds for context.
- Only what the numbers show is reported; if the groups do not explain the bias, that is the finding.

## 10. VALIDATION and freezing

After selection, ablations, intervals and analyses are complete on the backtest folds: **one** run on VALIDATION
(cutoff 2015-05-08, trained on rows with `t <= 2015-05-08`) for the selected full model, its quantile models and the best
baseline. Nothing is tuned or re-selected on it. The selected configuration is frozen; TEST stays locked until Step 4.

## 11. Claims this evaluation cannot support

- A **causal** effect of promotions, or anything about profit or cannibalisation.
- **Performance outside this retailer**, period (2013-2015), horizon (42 days) or setting (Open and Promo known for the
  target days, Friday cutoffs).
- **Why stores closed** (e.g. the Jul-Dec 2014 gap): the data does not say.
- Anything about **unplanned closures** and closed days (excluded from the evaluated rows by construction).

## 12. Amendments

**2026-10-26 - Part B result (seasonal-naive over-forecast on VALIDATION).** Source: `evals/results/snaive_bias_diagnostic.md`
(pre-registered holiday splits in section 9; the promo-state and Saturday cuts below were added afterwards and are labelled
exploratory in that file). No model design decision in this protocol depends on it.

- **Holidays do not explain it.** Seasonal naive 7, VALIDATION: overall Bias +0.1145. Rows whose target day and source day
  are both ordinary (91.0% of rows) have Bias +0.1228 and contribute +0.1114 of the +0.1145; rows with a holiday-affected
  target or source day (9.0%) contribute +0.0030; the 152 state-holiday target rows (0.4%) have Bias +0.0023 with a CI
  that spans zero. No source day falls on a state or school holiday flag. Seasonal naive 14 behaves the same way.
- **What the numbers do show (exploratory).** Every target day copies a source day from the single week before the
  cutoff (2015-05-02..2015-05-08). (a) Source day promo and target day non-promo: 38.8% of rows, Bias +0.2473
  (contribution +0.0858); for seasonal naive 14 the same group has Bias +0.3455 (contribution +0.1199 of +0.1217).
  (b) Saturdays: the target Saturdays all copy 2015-05-02, whose mean sales per open store (7,861) is above every other
  Saturday in 2015-04-11..2015-06-13 (5,655-6,768); Bias on Saturday targets is +0.2836 (contribution +0.0426). Where
  source and target are both promo days the Bias is slightly negative (-0.0273). In the backtest folds the sign of the
  mismatch varies (source non-promo to target promo gives about -0.26), which is why the Bias of seasonal naive 7 pooled over
  all backtest rows is -0.049 instead. The VALIDATION bias is therefore a property of which single week the last cutoff
  happened to copy, not a general over-forecasting tendency. Why 2015-05-02 was high is not examined here.

**2026-10-26 - Feature-table storage and subsample seeding (made after committing this protocol, before any model was trained).**
The first attempt to hold the whole table in memory was stopped by the memory watchdog (about 3 GB of RAM were available).
The table is now written per cutoff (streaming) and only rows that some fold can use are kept: `Sales > 0` and (a member of
the fixed-seed 20% training subsample, or in an evaluation cutoff or in an early-stopping holdout cutoff, which are kept
in full). The subsample draw is seeded per cutoff with `(20260101, cutoff index)` rather than from one global draw, so it does
not depend on which cutoffs are built. The columns that switch the Promo flag of the target day (`swp1_*`, `swp0_*`) are
stored only for evaluation-cutoff rows. Row sets used by every fold (train, inner, holdout, evaluation) are identical to
those of the full table (tested in `tests/test_gbm.py` and `tests/test_model_features.py`). No hyperparameter, feature or
selection rule changed.
