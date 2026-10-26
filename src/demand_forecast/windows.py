"""Fixed TRAIN / VALIDATION / TEST windows and rolling-origin backtest folds (calendar days)."""
import pandas as pd

from demand_forecast import config


def make_windows(data_start=config.DATA_START, data_end=config.DATA_END,
                 test_days=config.TEST_DAYS, val_days=config.VAL_DAYS) -> dict:
    start, end = pd.Timestamp(data_start), pd.Timestamp(data_end)
    test_start = end - pd.Timedelta(days=test_days - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=val_days - 1)
    train_end = val_start - pd.Timedelta(days=1)
    if train_end < start:
        raise ValueError("data span is too short for the requested windows")
    return {"train": (start, train_end), "validation": (val_start, val_end), "test": (test_start, end)}


def backtest_folds(train_start, train_end, horizon_days=config.BACKTEST_HORIZON_DAYS,
                   step_days=config.BACKTEST_STEP_DAYS, span_days=config.BACKTEST_SPAN_DAYS) -> list:
    """Folds ending at train_end and stepping back by step_days; each fold trains on dates <= cutoff
    (expanding window) and forecasts the next horizon_days. Only folds whose horizon starts inside the
    last span_days of TRAIN are kept, and every horizon lies entirely inside TRAIN. Chronological order."""
    train_start, train_end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    earliest_horizon_start = train_end - pd.Timedelta(days=span_days - 1)
    folds, horizon_end = [], train_end
    while True:
        horizon_start = horizon_end - pd.Timedelta(days=horizon_days - 1)
        cutoff = horizon_start - pd.Timedelta(days=1)
        if horizon_start < earliest_horizon_start or cutoff < train_start:
            break
        folds.append({"cutoff": cutoff, "horizon_start": horizon_start, "horizon_end": horizon_end})
        horizon_end -= pd.Timedelta(days=step_days)
    folds.reverse()
    for i, f in enumerate(folds, 1):
        f["fold"] = i
    return folds
