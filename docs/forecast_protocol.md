# Forecast protocol

Written and committed **before** any baseline was run. Everything below is fixed in advance; deviations must be
recorded in this file with a dated note, not made silently. No numbers appear here: results live in
`evals/results/` and every configuration tried is logged in `evals/results/spec_log.csv`.

Data and windows: see `docs/data_audit.md` and `evals/results/windows.json`
(TRAIN 2013-01-01..2015-05-08, VALIDATION 2015-05-09..2015-06-19, TEST 2015-06-20..2015-07-31).

## 1. Forecast task

At each **cutoff date** `c`, forecast daily `Sales` for every store for each of the next **42 days** (`c+1 .. c+42`),
using only

- store-day rows with `Date <= c` (history), and
- information known in advance for the target days: calendar (date, day of week), `StateHoliday`, `SchoolHoliday`,
  the planned `Promo` flag, store attributes, and the `Open` schedule (see below).

Never used: `Customers` (unknown at forecast time; see the leakage rule in the README), and any `Sales` after `c`.
`Customers` is excluded structurally: the modelling frame is loaded with an explicit column list that does not
contain it, so no forecaster can even see it.

**Assumption about `Open`.** Forecasts are made for *scheduled open days*. The `Open` flag of the target day is
treated as known in advance (as in the Kaggle test file). Days with `Open = 0` are forecast as 0 and are
**excluded from all error metrics**. Unplanned closures are therefore not part of the forecasting problem here
and this is a limitation, not something the evaluation tests. Store-days missing from the data (no row) are not
forecast-evaluated either.

**Evaluated rows.** Target rows with `Open = 1` and `Sales > 0` (open days with zero sales are excluded because
percentage errors are undefined there).

**Test window is untouched.** The modelling frame drops every row with `Date >= 2015-06-20` at load time, so
step 2 cannot read test-window rows. VALIDATION is used only for the single confirmation run (section 4); nothing is
tuned or selected on it.

## 2. Metrics (on evaluated rows)

With `y` = actual, `ŷ` = forecast, `e = ŷ − y`:

| Metric | Definition |
|---|---|
| RMSPE | `sqrt(mean(((y − ŷ)/y)^2))` (Kaggle competition metric) |
| WAPE | `Σ|e| / Σ y` |
| MASE | mean over evaluated rows of `|e| / s_store`, where `s_store` is the store's in-sample seasonal-naive (period 7) MAE: the mean of `|y_t − y_{t−7}|` over history rows with `Date <= c` where both day `t` and day `t−7` are open with `Sales > 0`. Rows of stores without such pairs are left out of MASE only. |
| Bias | `mean(e) / mean(y) = Σ e / Σ y` (positive = over-forecast) |

All four are always reported together. **Primary metric for choosing between configurations: WAPE** (mean over the
8 backtest folds), because RMSPE is dominated by a few low-sales days; RMSPE is still reported everywhere and ranks
are shown for both.

**Aggregation over folds.** A fold's metric is computed on that fold's evaluated rows. "Mean over folds" is the
unweighted mean of the per-fold metrics (not a pooled metric).

**Confidence intervals.** Store-clustered bootstrap: resample the 1,115 stores with replacement (all of a
store's rows in every fold move together), recompute every metric from the resample, 500 resamples, fixed seed
(20260101), 2.5th/97.5th percentiles. The same resampled store sets are used for all baselines and folds, so
differences between baselines are paired. The bootstrap is computed from per-store sufficient statistics in a single
process (no more than 2 workers are ever used). It captures store sampling uncertainty, not uncertainty about
future periods; with 8 folds the fold-to-fold spread is shown separately.

**Breakdowns** (each with the four metrics and CIs): by horizon bucket (days 1-7, 8-14, 15-21, 22-28, 29-35,
36-42 after the cutoff), by store type (a, b, c, d), by Promo vs non-promo target day, by day of week. Additional
diagnostic: stores with **stale history** (no open positive-sales day in the 56 days up to the cutoff) vs the rest,
because stores that reopen after the Jul-Dec 2014 gap have almost no recent history at some cutoffs.

## 3. Backtest

- The 8 rolling-origin folds in `evals/results/windows.json` (expanding window, 42-day horizon, cutoffs 2014-06-06 ..
  2015-03-27). Each forecaster is fit/prepared only on `Date <= cutoff` of that fold.
- **Folds 2-5** (horizons inside or touching the Jul-Dec 2014 period in which 180 stores have no rows) are reported
  **separately** from folds 1, 6, 7, 8, and also together as "mean over all 8 folds".
- Fold 5's horizon contains Christmas / New Year; fold 1-4 horizons contain no Christmas. None of the baselines
  models yearly seasonality, so this is expected to hurt folds with strong seasonal movements.
- The framework asserts, for every fold, that history dates are `<= cutoff`, target dates are `> cutoff`, and that
  forecaster inputs contain no `Sales`/`Customers` for the target days. A test additionally checks that changing all
  data after the cutoff does not change any forecast.

**Final confirmation.** VALIDATION (2015-05-09..2015-06-19) is forecast once per baseline from cutoff = TRAIN end
(2015-05-08), using the configuration selected on the backtest folds. It is not used for tuning, model selection or
re-running with other settings; if it disagrees with the backtest we report that.

**Optimism note.** Tuned baselines (3, 4, 5) are tuned and evaluated on the same 8 folds, so their backtest numbers are
mildly optimistic (a handful of configurations each); the VALIDATION check is the unbiased one.

## 4. Baselines (specifications fixed here)

Common: "observations" = history rows with `Open = 1` and `Sales > 0`. `dow` = day of week. Forecast for a target
day with `Open = 0` is 0. Forecasts are floored at 0. For every baseline the share of forecasts that used each
fallback level is saved with the results.

**Final fallback chain** (used when a baseline has no value): store x dow mean over all history -> store mean over
all history -> global mean for that dow.

1. **Seasonal naive, period 7.** Target day `t` at horizon `h = t − c`: the sales of the same store at
   `t − 7·k` with `k = ceil(h/7)` (the most recent same weekday on or before the cutoff). If that day is not an
   observation, try `k+1, k+2, k+3`; then the final fallback chain.
2. **Seasonal naive, period 14.** Same with period 14 (`k = ceil(h/14)`, then `k+1..k+3`), to match the alternating
   promo weeks seen in the audit.
3. **Promo-aware weekday mean, window K weeks.** Mean sales of the same store, same dow and same `Promo` flag as the
   target day over observations in `(c − 7K days, c]`. If none: store x dow mean over the same window (any promo);
   then the final fallback chain. **K is chosen from {4, 8, 12} on the backtest folds only** (lowest mean WAPE).
4. **Store median level x global dow/promo index, window L weeks.** Store level = median sales over observations in
   `(c − 7L days, c]` (if none: median over all history, then global median). Index per (dow, Promo) cell = mean over all
   observations in the window of `sales / store level`; if a cell is empty the dow-only index, then 1. Forecast =
   level x index. **L chosen from {13, 26, 52} on the backtest folds only** (lowest mean WAPE).
5. **Per-store exponential smoothing (optional).** statsmodels `ExponentialSmoothing`, no trend, additive seasonality
   with period m, `initialization_method="estimated"`, fit on the last 182 days up to the cutoff. The daily series is
   built from observations; every other day (closed, zero-sales, missing) is filled with the last observed value of
   the same dow within the window (store window mean if there is none), because the `Open` flag already handles
   closed target days. Stores with fewer than 60 observations in the window use the final fallback chain.
   **m is chosen from {7, 14} on the backtest folds only.** Run with at most 2 workers. If the whole backtest grid for
   this baseline would take more than 30 minutes, it is skipped and logged as skipped with the measured reason.

None of the baselines uses the `StateHoliday`/`SchoolHoliday` flags other than through `Open`, store attributes, or
yearly seasonality; these are known gaps for later models.

**Selection rule.** For each tunable baseline, the configuration with the lowest mean-over-folds WAPE on the 8
backtest folds is selected. The best baseline overall is the one with the lowest mean-over-folds WAPE; ties or
overlapping CIs are described as such, not as a win. Paired bootstrap differences are reported for the best vs the
runner-up.

## 5. Logging

Every configuration run (including dropped ones and skipped ones) is appended to `evals/results/spec_log.csv`
with `run_id`, baseline, parameters, split (fold or validation), number of evaluated rows, the four metrics,
runtime, and status (`selected`, `dropped`, `skipped`, `validation`). The log is append-only; the reports use the
latest `run_id`.

## 6. What this evaluation cannot support

- **Causal promotion effects.** Promo days are not randomly assigned; any promo vs non-promo difference in sales or
  in error is descriptive.
- **Why stores closed** (e.g. the Jul-Dec 2014 gap). The data does not say.
- **Performance outside this retailer, period or horizon.** One chain, 2013-2015, 42-day horizon, with `Open` and
  `Promo` known for the target days.
- **Unplanned closures and demand on closed days.** They are outside the evaluated rows by construction.
