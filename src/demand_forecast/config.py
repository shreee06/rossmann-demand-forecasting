"""Paths and fixed constants for the Rossmann demand-forecasting project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Raw Kaggle files are read in place and never copied into the repo.
RAW_DIR = Path(os.environ.get("ROSSMANN_RAW_DIR", "~/Downloads/rossmann-raw")).expanduser()
RAW_TRAIN = RAW_DIR / "train.csv"
RAW_STORE = RAW_DIR / "store.csv"  # test.csv has no Sales and is NOT used anywhere

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_TRAIN = PROCESSED_DIR / "train.parquet"
PROCESSED_STORE = PROCESSED_DIR / "store.parquet"
RESULTS_DIR = ROOT / "evals" / "results"
DOCS_DIR = ROOT / "docs"

# Date span of train.csv (verified against the data by scripts/make_windows.py).
DATA_START = "2013-01-01"
DATA_END = "2015-07-31"

# Fixed evaluation windows, in calendar days counted back from DATA_END.
#   TEST = last TEST_DAYS days, VALIDATION = the VAL_DAYS days before that, TRAIN = everything earlier.
TEST_DAYS = 42
VAL_DAYS = 42

# Rolling-origin backtest inside TRAIN: folds of BACKTEST_HORIZON_DAYS, one every BACKTEST_STEP_DAYS,
# keeping only folds whose horizon starts within the last BACKTEST_SPAN_DAYS of TRAIN.
BACKTEST_HORIZON_DAYS = 42
BACKTEST_STEP_DAYS = 42
BACKTEST_SPAN_DAYS = 364

# Runs of consecutive missing/closed days that count as a "long closure".
LONG_CLOSURE_DAYS = 30

# Memory watchdog: exit when MemAvailable drops below this (laptop has ~7.5 GB). Parallel work: <= 2 workers.
MIN_AVAILABLE_KB = 1_000_000
MAX_WORKERS = 2
