"""Store-clustered bootstrap over per-store sufficient statistics (single process, vectorised)."""
import numpy as np
import pandas as pd

from demand_forecast.metrics import METRICS, STATS, add_error_columns, metrics_from_stats

N_BOOT = 500
SEED = 20260101


class StoreBootstrap:
    """Fixed set of store resamples shared by every baseline/fold, so comparisons are paired."""

    def __init__(self, n_stores: int, n_boot: int = N_BOOT, seed: int = SEED):
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n_stores, size=(n_boot, n_stores))
        self.counts = np.stack([np.bincount(r, minlength=n_stores) for r in draws]).astype(float)  # (B, S)
        self.n_boot, self.n_stores = n_boot, n_stores


def stat_tensor(preds: pd.DataFrame, store_ids, fold_ids, group_col=None, groups=None):
    """Sum the statistics per (fold, group, store) -> array [F, G, S, 7], plus the list of group labels."""
    d = add_error_columns(preds)
    if group_col is None:
        d["_g"] = "all"
        group_col = "_g"
    if groups is None:
        groups = sorted(d[group_col].dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
    groups = list(groups)
    fpos = {f: i for i, f in enumerate(fold_ids)}
    gpos = {g: i for i, g in enumerate(groups)}
    spos = {s: i for i, s in enumerate(store_ids)}
    T = np.zeros((len(fold_ids), len(groups), len(store_ids), len(STATS)))
    agg = d.groupby(["fold", group_col, "Store"], observed=True)[STATS].sum().reset_index()
    fi = agg["fold"].map(fpos).to_numpy()
    gi = agg[group_col].map(gpos).to_numpy()
    si = agg["Store"].map(spos).to_numpy()
    T[fi, gi, si] = agg[STATS].to_numpy()
    return T, groups


def _nanmean(a, axis):
    with np.errstate(invalid="ignore"):
        valid = ~np.isnan(a)
        n = valid.sum(axis)
        s = np.where(valid, a, 0.0).sum(axis)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def summarize(T: np.ndarray, fold_ids, boot: StoreBootstrap, views: dict):
    """Point metrics and 95% bootstrap CIs.

    views: name -> list of fold ids; the view metric is the unweighted mean of the per-fold metrics.
    Returns (per_fold, per_view, boot_views) where per_fold[f][g][metric] and per_view[view][g][metric]
    are dicts {value, lo, hi}; boot_views[view] is the raw (B, G, 4) bootstrap array (for paired differences).
    """
    point = metrics_from_stats(T.sum(2))                                  # [F, G, 4]
    sums = np.einsum("bs,fgsk->bfgk", boot.counts, T)                     # [B, F, G, 7]
    bm = metrics_from_stats(sums)                                         # [B, F, G, 4]

    def ci(p, b):
        lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
        return {m: {"value": float(p[..., j]), "lo": float(lo[..., j]), "hi": float(hi[..., j])}
                for j, m in enumerate(METRICS)}

    per_fold = {}
    for fi, f in enumerate(fold_ids):
        per_fold[f] = [ci(point[fi, g], bm[:, fi, g]) for g in range(T.shape[1])]
    per_view, boot_views = {}, {}
    for name, folds in views.items():
        idx = [list(fold_ids).index(f) for f in folds]
        p = _nanmean(point[idx], axis=0)                                  # [G, 4]
        b = _nanmean(bm[:, idx], axis=1)                                  # [B, G, 4]
        per_view[name] = [ci(p[g], b[:, g]) for g in range(T.shape[1])]
        boot_views[name] = b
    return per_fold, per_view, boot_views


def paired_difference(b_a: np.ndarray, b_b: np.ndarray, point_a: float, point_b: float) -> dict:
    """a - b for one metric; b_* are the bootstrap vectors of the same view (shared resamples)."""
    d = b_a - b_b
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"diff": float(point_a - point_b), "lo": float(lo), "hi": float(hi)}
