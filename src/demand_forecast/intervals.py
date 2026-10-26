"""Prediction-interval metrics (coverage, relative width, pinball loss) with store-clustered bootstrap CIs."""
import numpy as np
import pandas as pd

from demand_forecast.bootstrap import StoreBootstrap

NOMINAL = 0.8
ALPHA_LO, ALPHA_HI = 0.1, 0.9
STAT = ["n", "covered", "below", "above", "width", "y", "pin_lo", "pin_hi"]
NAMES = ["coverage", "share_below", "share_above", "rel_width", "pinball_q10", "pinball_q90"]


def add_interval_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Needs y, lo, hi (sales scale, lo <= hi)."""
    y, lo, hi = d["y"].to_numpy(float), d["lo"].to_numpy(float), d["hi"].to_numpy(float)
    out = d.copy()
    out["n"] = 1.0
    out["covered"] = ((y >= lo) & (y <= hi)).astype(float)
    out["below"] = (y < lo).astype(float)
    out["above"] = (y > hi).astype(float)
    out["width"] = hi - lo
    out["pin_lo"] = np.maximum(ALPHA_LO * (y - lo), (ALPHA_LO - 1) * (y - lo))
    out["pin_hi"] = np.maximum(ALPHA_HI * (y - hi), (ALPHA_HI - 1) * (y - hi))
    return out


def metrics_from_stats(S):
    S = np.asarray(S, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([S[..., 1] / S[..., 0], S[..., 2] / S[..., 0], S[..., 3] / S[..., 0], S[..., 4] / S[..., 5],
                         S[..., 6] / S[..., 0], S[..., 7] / S[..., 0]], axis=-1)


def compute(d: pd.DataFrame) -> dict:
    S = add_interval_columns(d)[STAT].sum().to_numpy()
    return dict(zip(NAMES, metrics_from_stats(S).tolist()))


def tensor(preds: pd.DataFrame, store_ids, fold_ids, group_col=None, groups=None):
    d = add_interval_columns(preds)
    if group_col is None:
        d["_g"] = "all"
        group_col = "_g"
    groups = list(groups) if groups is not None else sorted(d[group_col].unique().tolist())
    fpos, gpos = {f: i for i, f in enumerate(fold_ids)}, {g: i for i, g in enumerate(groups)}
    spos = {s: i for i, s in enumerate(store_ids)}
    T = np.zeros((len(fold_ids), len(groups), len(store_ids), len(STAT)))
    agg = d.groupby(["fold", group_col, "Store"], observed=True)[STAT].sum().reset_index()
    T[agg["fold"].map(fpos).to_numpy(), agg[group_col].map(gpos).to_numpy(), agg["Store"].map(spos).to_numpy()] = agg[STAT].to_numpy()
    return T, groups


def summarize(T, fold_ids, boot: StoreBootstrap, views: dict):
    """Per-fold and per-view (unweighted mean of per-fold values) metrics with 95% store-bootstrap CIs."""
    point = metrics_from_stats(T.sum(2))
    bm = metrics_from_stats(np.einsum("bs,fgsk->bfgk", boot.counts, T))

    def nanmean(a, axis):
        valid = ~np.isnan(a)
        n = valid.sum(axis)
        return np.where(n > 0, np.where(valid, a, 0.0).sum(axis) / np.maximum(n, 1), np.nan)

    def ci(p, b):
        lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
        return {m: {"value": float(p[j]), "lo": float(lo[j]), "hi": float(hi[j])} for j, m in enumerate(NAMES)}

    per_fold = {f: [ci(point[i, g], bm[:, i, g]) for g in range(T.shape[1])] for i, f in enumerate(fold_ids)}
    per_view = {}
    for name, folds in views.items():
        idx = [list(fold_ids).index(f) for f in folds]
        p, b = nanmean(point[idx], 0), nanmean(bm[:, idx], 1)
        per_view[name] = [ci(p[g], b[:, g]) for g in range(T.shape[1])]
    return per_fold, per_view
