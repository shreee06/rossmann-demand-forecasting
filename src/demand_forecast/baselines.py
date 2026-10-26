"""Baseline forecasters (specs fixed in docs/forecast_protocol.md, section 4).

Interface: forecaster.forecast(history, future, store, cutoff) -> array aligned with `future`.
`history` has Date <= cutoff; `future` has only known-in-advance columns (no Sales). Each forecaster records
`fallback_counts` (how many forecasts came from which rule) for the last call.
"""
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

ETS_WINDOW_DAYS = 182
ETS_MIN_OBS = 60
HORIZON_DAYS = 42


def observations(history: pd.DataFrame) -> pd.DataFrame:
    obs = history.loc[(history["Open"] == 1) & (history["Sales"] > 0), ["Store", "Date", "Sales", "Promo"]].copy()
    obs["dow"] = obs["Date"].dt.dayofweek
    return obs


def _lookup(future: pd.DataFrame, table: pd.Series, keys: list) -> np.ndarray:
    """Left-lookup of a unique-indexed Series on future[keys] (order preserved)."""
    right = table.rename("v").reset_index()
    return future[keys].merge(right, on=keys, how="left")["v"].to_numpy(float)


def final_fallback(future: pd.DataFrame, obs: pd.DataFrame, counts: dict, mask: np.ndarray) -> np.ndarray:
    """store x dow mean (all history) -> store mean -> global dow mean -> global mean, for rows in `mask`."""
    f = future.assign(dow=future["Date"].dt.dayofweek.to_numpy())
    out = np.full(len(f), np.nan)
    chain = [
        ("fb_store_dow_mean", _lookup(f, obs.groupby(["Store", "dow"])["Sales"].mean(), ["Store", "dow"])),
        ("fb_store_mean", _lookup(f, obs.groupby("Store")["Sales"].mean(), ["Store"])),
        ("fb_global_dow_mean", _lookup(f, obs.groupby("dow")["Sales"].mean(), ["dow"])),
        ("fb_global_mean", np.full(len(f), obs["Sales"].mean())),
    ]
    todo = mask.copy()
    for name, vals in chain:
        take = todo & np.isfinite(vals)
        out[take] = vals[take]
        if take.any():
            counts[name] = counts.get(name, 0) + int(take.sum())
        todo &= ~take
    return out


class SeasonalNaive:
    def __init__(self, period: int):
        self.period = period
        self.name = f"seasonal_naive_{period}"
        self.params = {"period": period}

    def forecast(self, history, future, store, cutoff):
        p = self.period
        obs = observations(history)
        s = obs.set_index(["Store", "Date"])["Sales"]
        assert s.index.is_unique
        h = (future["Date"] - cutoff).dt.days.to_numpy()
        k0 = np.ceil(h / p).astype(int)
        yhat = np.full(len(future), np.nan)
        counts = {}
        for j in range(4):
            cand = future["Date"].to_numpy() - pd.to_timedelta(p * (k0 + j), unit="D").to_numpy()
            assert (cand <= np.datetime64(cutoff)).all()
            vals = s.reindex(pd.MultiIndex.from_arrays([future["Store"].to_numpy(), cand])).to_numpy(float)
            take = np.isnan(yhat) & np.isfinite(vals)
            yhat[take] = vals[take]
            counts[f"lag_k+{j}"] = int(take.sum())
        miss = np.isnan(yhat)
        if miss.any():
            yhat[miss] = final_fallback(future, obs, counts, miss)[miss]
        self.fallback_counts = counts
        return yhat


class PromoWeekdayMean:
    def __init__(self, k_weeks: int):
        self.k = k_weeks
        self.name = "promo_weekday_mean"
        self.params = {"K": k_weeks}

    def forecast(self, history, future, store, cutoff):
        obs = observations(history)
        w = obs[obs["Date"] > cutoff - pd.Timedelta(days=7 * self.k)]
        f = future.assign(dow=future["Date"].dt.dayofweek.to_numpy())
        counts = {}
        yhat = _lookup(f, w.groupby(["Store", "dow", "Promo"])["Sales"].mean(), ["Store", "dow", "Promo"])
        counts["store_dow_promo_window"] = int(np.isfinite(yhat).sum())
        miss = np.isnan(yhat)
        v = _lookup(f, w.groupby(["Store", "dow"])["Sales"].mean(), ["Store", "dow"])
        take = miss & np.isfinite(v)
        yhat[take] = v[take]
        counts["store_dow_window"] = int(take.sum())
        miss = np.isnan(yhat)
        if miss.any():
            yhat[miss] = final_fallback(future, obs, counts, miss)[miss]
        self.fallback_counts = counts
        return yhat


class StoreLevelIndex:
    def __init__(self, l_weeks: int):
        self.l = l_weeks
        self.name = "store_median_x_index"
        self.params = {"L": l_weeks}

    def forecast(self, history, future, store, cutoff):
        obs = observations(history)
        w = obs[obs["Date"] > cutoff - pd.Timedelta(days=7 * self.l)]
        counts = {}
        level_w = w.groupby("Store")["Sales"].median()
        level_all = obs.groupby("Store")["Sales"].median()
        level = level_w.combine_first(level_all)
        global_median = float(obs["Sales"].median())
        r = w.assign(ratio=w["Sales"].to_numpy() / w["Store"].map(level_w).to_numpy())
        idx_cell = r.groupby(["dow", "Promo"])["ratio"].mean()
        idx_dow = r.groupby("dow")["ratio"].mean()
        f = future.assign(dow=future["Date"].dt.dayofweek.to_numpy())
        lv = f["Store"].map(level).to_numpy(float)
        counts["level_from_all_history"] = int((np.isnan(f["Store"].map(level_w).to_numpy(float)) & np.isfinite(lv)).sum())
        lv = np.where(np.isfinite(lv), lv, global_median)
        ix = _lookup(f, idx_cell, ["dow", "Promo"])
        counts["index_dow_promo"] = int(np.isfinite(ix).sum())
        ix2 = _lookup(f, idx_dow, ["dow"])
        use2 = np.isnan(ix) & np.isfinite(ix2)
        ix = np.where(use2, ix2, ix)
        counts["index_dow_only"] = int(use2.sum())
        ones = np.isnan(ix)
        ix = np.where(ones, 1.0, ix)
        counts["index_one"] = int(ones.sum())
        self.fallback_counts = counts
        return lv * ix


def _ets_fit(args):
    """One store: fit additive-seasonal ETS on the window series, return the 42-day forecast (or None)."""
    series, m = args
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(series, trend=None, seasonal="add", seasonal_periods=m,
                                       initialization_method="estimated").fit()
            fc = np.asarray(fit.forecast(HORIZON_DAYS), float)
        return fc if np.isfinite(fc).all() else None
    except Exception:
        return None


class ETSWeekly:
    def __init__(self, m: int, workers: int = 2):
        self.m = m
        self.workers = min(workers, 2)
        self.name = "ets_additive_seasonal"
        self.params = {"m": m}

    def _series(self, g: pd.DataFrame, cutoff):
        start = cutoff - pd.Timedelta(days=ETS_WINDOW_DAYS - 1)
        arr = np.full(ETS_WINDOW_DAYS, np.nan)
        pos = (g["Date"] - start).dt.days.to_numpy()
        arr[pos] = g["Sales"].to_numpy(float)
        mean = np.nanmean(arr)
        grid = arr.reshape(ETS_WINDOW_DAYS // 7, 7)  # column = same weekday (window is whole weeks)
        for c in range(7):  # carry the last observed same-weekday value forward
            col = grid[:, c]
            last = np.nan
            for i in range(len(col)):
                if np.isnan(col[i]):
                    col[i] = last
                else:
                    last = col[i]
        grid[np.isnan(grid)] = mean
        return grid.reshape(-1)

    def forecast(self, history, future, store, cutoff):
        assert ETS_WINDOW_DAYS % 7 == 0
        obs = observations(history)
        w = obs[obs["Date"] > cutoff - pd.Timedelta(days=ETS_WINDOW_DAYS)]
        jobs, keys = [], []
        for s, g in w.groupby("Store"):
            if len(g) >= ETS_MIN_OBS:
                jobs.append((self._series(g, cutoff), self.m))
                keys.append(s)
        if self.workers > 1 and len(jobs) > 50:
            with Pool(self.workers) as pool:
                res = pool.map(_ets_fit, jobs, chunksize=25)
        else:
            res = [_ets_fit(j) for j in jobs]
        fcs = {s: r for s, r in zip(keys, res) if r is not None}
        h = (future["Date"] - cutoff).dt.days.to_numpy()
        yhat = np.full(len(future), np.nan)
        stores = future["Store"].to_numpy()
        for s, fc in fcs.items():
            m = stores == s
            yhat[m] = fc[h[m] - 1]
        counts = {"ets": int(np.isfinite(yhat).sum()), "ets_failed_stores": int(len(keys) - len(fcs)),
                  "stores_below_min_obs": int(len(set(future["Store"].unique()) - set(keys)))}
        miss = np.isnan(yhat)
        if miss.any():
            yhat[miss] = final_fallback(future, obs, counts, miss)[miss]
        self.fallback_counts = counts
        return yhat
