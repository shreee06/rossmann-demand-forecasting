"""Backtest runner. Guards: no TEST rows are ever loaded, the leakage column named in the README is never loaded, and a forecaster only
receives history up to the cutoff plus the known-in-advance columns of the target days (no Sales)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_forecast import config
from demand_forecast.metrics import seasonal_naive_scale
from demand_forecast.windows import backtest_folds, make_windows

# Explicit column lists: the unavailable-at-forecast-time column is simply never read.
HISTORY_COLUMNS = ["Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
FUTURE_COLUMNS = ["Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
STALE_DAYS = 56
# Post-hoc diagnostic (see protocol amendment): a store with fewer than SPARSE_MIN_ROWS rows in the last
# SPARSE_WINDOW_DAYS before the cutoff (e.g. stores that just came back after the Jul-Dec 2014 gap).
SPARSE_WINDOW_DAYS = 200
SPARSE_MIN_ROWS = 100


@dataclass(frozen=True)
class Split:
    fold: object  # 1..8 for backtest folds, "validation" for the confirmation window
    cutoff: pd.Timestamp
    horizon_start: pd.Timestamp
    horizon_end: pd.Timestamp


def backtest_splits():
    tr_a, tr_b = make_windows()["train"]
    return [Split(f["fold"], f["cutoff"], f["horizon_start"], f["horizon_end"]) for f in backtest_folds(tr_a, tr_b)]


def validation_split():
    w = make_windows()
    return Split("validation", w["train"][1], *w["validation"])


def assert_split_allowed(split: Split):
    test_start = make_windows()["test"][0]
    if split.horizon_end >= test_start:
        raise ValueError(f"split {split.fold} reaches the TEST window ({test_start.date()})")
    if not split.cutoff < split.horizon_start:
        raise ValueError("cutoff must precede the horizon")


def load_modelling_frame(path=None) -> pd.DataFrame:
    """Train frame without TEST rows and without the unavailable column."""
    df = pd.read_parquet(path or config.PROCESSED_TRAIN, columns=HISTORY_COLUMNS)
    return restrict_to_modelling_period(df)


def restrict_to_modelling_period(df: pd.DataFrame) -> pd.DataFrame:
    test_start = make_windows()["test"][0]
    return df[df["Date"] < test_start].reset_index(drop=True)


def run_splits(forecaster, frame: pd.DataFrame, store: pd.DataFrame, splits, store_ids=None):
    """Run a forecaster over splits. Returns (evaluated-row predictions, per-split diagnostics list)."""
    store_type = store.set_index("Store")["StoreType"].astype(str)
    out, diags = [], []
    for sp in splits:
        assert_split_allowed(sp)
        history = frame[frame["Date"] <= sp.cutoff][HISTORY_COLUMNS]
        target = frame[(frame["Date"] >= sp.horizon_start) & (frame["Date"] <= sp.horizon_end)]
        future = target[FUTURE_COLUMNS].copy()
        assert history["Date"].max() <= sp.cutoff and future["Date"].min() > sp.cutoff
        assert "Sales" not in future.columns
        is_open = future["Open"].to_numpy() == 1
        f_open = np.asarray(forecaster.forecast(history, future[is_open], store, sp.cutoff), float)
        if f_open.shape != (int(is_open.sum()),) or not np.isfinite(f_open).all():
            raise ValueError(f"{forecaster.name}: bad forecast output on split {sp.fold}")
        yhat = np.zeros(len(future))  # Open = 0 days are forecast as 0
        yhat[is_open] = np.maximum(f_open, 0.0)
        ev = (target["Open"].to_numpy() == 1) & (target["Sales"].to_numpy() > 0)
        scale = seasonal_naive_scale(history, 7)
        obs = history[(history["Open"] == 1) & (history["Sales"] > 0)]
        last_obs = obs.groupby("Store")["Date"].max()
        d = pd.DataFrame({
            "fold": sp.fold, "Store": future["Store"].to_numpy(), "Date": future["Date"].to_numpy(),
            "horizon": (future["Date"] - sp.cutoff).dt.days.to_numpy(),
            "y": target["Sales"].to_numpy(float), "yhat": yhat, "Promo": future["Promo"].to_numpy(),
        })[ev]
        d["scale"] = d["Store"].map(scale).to_numpy()
        d["StoreType"] = d["Store"].map(store_type).to_numpy()
        d["DayOfWeek"] = d["Date"].dt.dayofweek.to_numpy() + 1
        d["horizon_bucket"] = ((d["horizon"] - 1) // 7).map(lambda i: f"{7 * i + 1}-{7 * i + 7}")
        d["promo_label"] = np.where(d["Promo"] == 1, "promo", "no promo")
        stale_cut = sp.cutoff - pd.Timedelta(days=STALE_DAYS)
        recent_rows = history[history["Date"] > sp.cutoff - pd.Timedelta(days=SPARSE_WINDOW_DAYS)].groupby("Store").size()
        d["row_coverage"] = np.where(d["Store"].map(recent_rows).fillna(0).to_numpy() < SPARSE_MIN_ROWS, "sparse", "full")
        d["history"] = np.where(d["Store"].map(last_obs).ge(stale_cut).to_numpy(), "recent", "stale")
        out.append(d.reset_index(drop=True))
        diags.append({"fold": sp.fold, "n_target_rows": int(len(future)), "n_evaluated": int(ev.sum()),
                      "n_open_zero_sales": int(((target["Open"] == 1) & (target["Sales"] == 0)).sum()),
                      "n_closed": int((target["Open"] == 0).sum()),
                      "fallback": dict(getattr(forecaster, "fallback_counts", {}) or {})})
    return pd.concat(out, ignore_index=True), diags
