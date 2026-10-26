# Final protocol (Step 4)

Written and committed **before the TEST window (2015-06-20..2015-07-31) is read in any form**. Builds on
`docs/forecast_protocol.md` and `docs/model_protocol.md` (task, metrics, bootstrap, features, leakage rules). Deviations
must be dated amendments at the bottom.

Compute limits as before: `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, LightGBM `num_threads=2`, at most 2 workers, memory
watchdog on. `Customers` is never read (explicit column lists).

## 1. Frozen model (no re-tuning)

- Configuration from `evals/results/model_selected.json`: `num_leaves = 63`, `min_child_samples = 50`, the fixed
  parameters of `docs/model_protocol.md` section 3, the same 45 features, target `log1p(Sales)`, **unsmeared**
  back-transform `expm1(prediction)` (the smearing variant over-forecast on VALIDATION and stays secondary).
- **Tree counts as in the frozen (VALIDATION) run**, no early stopping: point model **1106** trees, q10 model **1500**,
  q90 model **206** (`evals/results/model_validation.json`).
- **Final training data:** everything up to the day before TEST (TRAIN + VALIDATION, through **2015-06-19**). Training
  rows: Friday cutoffs 2013-04-05..2015-06-12, target day `t <= 2015-06-19`, `Sales > 0`, the same fixed-seed 20% row
  subsample (seed `(20260101, cutoff index)`). The features of training rows are computed from data through 2015-06-19
  only (nothing dated in TEST is read to build training rows; calendar/promo look-ahead past 2015-06-19 is treated as
  unknown, exactly as at the end of the data in earlier steps).
- **Forecast:** cutoff **2015-06-19** (a Friday), horizons 1-42 = the TEST window. Test-row features use history through
  2015-06-19 and, for the horizon, only the known-in-advance columns (`Open`, `Promo`, `StateHoliday`, `SchoolHoliday`);
  TEST `Sales` are blanked before features are built and read again only for scoring.

## 2. Interval calibration (pre-registered)

Split-conformal (conformalised quantile regression) with a **multiplicative** score, one factor per horizon bucket
(1-7, 8-14, ..., 36-42):

- Calibration data: the q10/q90 predictions of the **8 backtest folds** from Step 3 (fold models, evaluated rows) **only**;
  never VALIDATION, never TEST (asserted on dates: every calibration target date `<= 2015-05-08`).
- Score per row: `s = max(lo / y, y / hi)` (`s <= 1` means inside the interval). For a bucket with `n` rows the factor is
  `q = the ceil((n + 1) * 0.8)-th smallest score`. Calibrated interval: `[lo / q, hi * q]` (`q > 1` widens, `q < 1` narrows).
- Confirmation: applied **once** to the Step-3 VALIDATION q10/q90 predictions. If VALIDATION coverage is outside
  0.77-0.83 it is reported as is; the factors are **not** re-tuned.
- On TEST, both uncalibrated and calibrated intervals are reported (coverage, below/above shares, relative width, by
  horizon bucket), using the factors fixed here.

## 3. Systems compared on TEST

1. Global LightGBM (frozen, section 1).
2. Best baseline: promo-aware weekday mean, K = 12.
3. Seasonal naive, period 14 (simple reference).

All forecast from cutoff 2015-06-19 with the same Open assumption (Open = 0 days forecast as 0, excluded); evaluated rows
= TEST store-days with `Open = 1` and `Sales > 0`. Metrics (RMSPE, WAPE, MASE, Bias), store-clustered bootstrap 95% CIs
(500 resamples, seed 20260101), breakdowns (horizon bucket, promo vs non-promo, store type, day of week), and paired
bootstrap differences (model − baseline, model − seasonal naive 14, baseline − seasonal naive 14) exactly as in Step 3.

## 4. One-shot rule

- The TEST evaluation runs **once**. Before scoring, the script writes `evals/results/test_run_record.json` (timestamp,
  git commit hash and whether the working tree was clean, SHA-256 of the frozen configuration file, window dates, number
  of evaluated rows, seeds). If the record exists, the script **refuses** to run unless `--override "<reason>"` is given;
  an override is appended to the record and must be reported.
- The result is reported whatever it is, including if the model loses to a baseline.

## 5. What TEST can and cannot say

TEST is **one 42-day summer window (2015-06-20..2015-07-31) of one retailer**, forecast from one cutoff. It does not
establish performance in other seasons (e.g. Christmas, fold 5 in the backtest), other years, other horizons or other
retailers. The bootstrap CIs reflect store sampling only, not period-to-period variation. Unplanned closures are excluded
by the Open assumption. Promotion results remain associations.

## 6. Amendments

_(none yet)_
