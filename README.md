# Retail demand forecasting — Rossmann Store Sales

Portfolio project: forecast daily store sales for the Rossmann drugstore chain with a fixed time-based
evaluation protocol. This repository ships **code and aggregate results only**.

> Status: setup, data audit and fixed evaluation windows are done. No forecasting model has been fitted yet.

## Overview

- Target: daily `Sales` per store.
- Forecast horizon: the last 6 weeks (42 days) of the data are held out as TEST, the 6 weeks before that are
  VALIDATION, everything earlier is TRAIN. Rolling-origin backtest folds inside TRAIN are defined as well.
  Exact dates: `evals/results/windows.json` (definitions in `src/demand_forecast/config.py`).
- Only information known at forecast time may be used (calendar, holidays, planned promotions, store attributes).

### Leakage rule

**`Customers` is not known at forecast time and is never used as a feature, nor to build features** (no lags,
rolling means, ratios of it). `src/demand_forecast/features.py` rejects any feature name containing "customers"
and `tests/test_leakage.py` fails if a Customers-derived column or a module touching it appears.

## Data

- Source: Rossmann store data from the Kaggle competition
  [`rossmann-store-sales`](https://www.kaggle.com/competitions/rossmann-store-sales). Credit to Rossmann and Kaggle.
- **The data is not included in this repository** and must not be redistributed. To get it, accept the competition
  rules on Kaggle and download `train.csv` and `store.csv` (e.g. `kaggle competitions download -c rossmann-store-sales`).
  `test.csv` has no `Sales` and is not used.
- Point the code at the folder holding the CSVs with `ROSSMANN_RAW_DIR` (default `~/Downloads/rossmann-raw`).
  Raw files are read in place, read-only.
- `StateHoliday` is a mixed-type column (the "no holiday" value appears as int `0` or string `"0"`, next to
  `a`/`b`/`c`). It is read as string, normalised, and stored as a categorical over exactly `{0, a, b, c}`;
  unknown or missing values raise an error.
- Audit of the data: [`docs/data_audit.md`](docs/data_audit.md) (generated; numbers in `evals/results/data_audit.json`).

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ROSSMANN_RAW_DIR=/path/to/rossmann-raw
python scripts/prepare_data.py   # CSV -> data/processed/*.parquet (untracked)
python scripts/make_windows.py   # evals/results/windows.json
python scripts/audit_data.py     # docs/data_audit.md + evals/results/data_audit.json
python -m pytest
```

(If ROS 2 is sourced, prefix with `env -u PYTHONPATH`.) Scripts run a memory watchdog that exits when less than 1 GB is available.

## Limitations

_To be written after the modelling steps._

## Results

_To be filled from real runs._ Every number in this repository must come from a run saved under `evals/results/`.
