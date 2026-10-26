"""Split-conformal (CQR, multiplicative score) widening of q10/q90 per horizon bucket (docs/final_protocol.md section 2).
Calibrated on backtest-fold rows only."""
import numpy as np
import pandas as pd

from demand_forecast.windows import make_windows

LEVEL = 0.8


def scores(y, lo, hi):
    """s = max(lo / y, y / hi); s <= 1 means y is inside [lo, hi]."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return np.maximum(lo / y, y / hi)


def conformal_factor(s, level=LEVEL):
    """The ceil((n + 1) * level)-th smallest score (capped at the largest)."""
    s = np.sort(np.asarray(s, float))
    k = int(np.ceil((len(s) + 1) * level))
    return float(s[min(k, len(s)) - 1])


def fit_factors(cal: pd.DataFrame, level=LEVEL) -> dict:
    """cal: y, lo, hi, horizon_bucket, Date. Refuses any row dated after TRAIN (backtest folds only)."""
    train_end = make_windows()["train"][1]
    if pd.to_datetime(cal["Date"]).max() > train_end:
        raise ValueError(f"calibration rows must be backtest-fold rows dated <= {train_end.date()}")
    s = scores(cal["y"], cal["lo"], cal["hi"])
    return {str(b): conformal_factor(s[(cal["horizon_bucket"] == b).to_numpy()], level)
            for b in sorted(cal["horizon_bucket"].unique(), key=lambda x: int(str(x).split("-")[0]))}


def apply_factors(lo, hi, buckets, factors: dict):
    q = np.array([factors[str(b)] for b in buckets], float)
    return np.asarray(lo, float) / q, np.asarray(hi, float) * q
