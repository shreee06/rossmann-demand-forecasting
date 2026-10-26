"""Turn evaluated-row predictions into metric tables with store-clustered bootstrap CIs."""
import numpy as np
import pandas as pd

from demand_forecast.bootstrap import StoreBootstrap, stat_tensor, summarize
from demand_forecast.metrics import METRICS, STATS, add_error_columns, metrics_from_stats

HORIZON_ORDER = ["1-7", "8-14", "15-21", "22-28", "29-35", "36-42"]
BREAKDOWNS = {  # name -> (column, ordered labels or None)
    "horizon": ("horizon_bucket", HORIZON_ORDER),
    "store_type": ("StoreType", ["a", "b", "c", "d"]),
    "promo": ("promo_label", ["promo", "no promo"]),
    "dow": ("DayOfWeek", [1, 2, 3, 4, 5, 6, 7]),
    "history": ("history", ["recent", "stale"]),
    "row_coverage": ("row_coverage", ["full", "sparse"]),  # post-hoc diagnostic
}
BACKTEST_VIEWS = {"all_folds": list(range(1, 9)), "folds_2_5": [2, 3, 4, 5], "other_folds": [1, 6, 7, 8]}


def fold_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Point metrics per fold (no bootstrap)."""
    d = add_error_columns(preds)
    S = d.groupby("fold")[STATS].sum()
    return pd.DataFrame(metrics_from_stats(S.to_numpy()), index=S.index, columns=METRICS).assign(n=S["n"].to_numpy())


def evaluate(preds: pd.DataFrame, store_ids, fold_ids, boot: StoreBootstrap, views: dict):
    """Returns (results dict, raw overall bootstrap arrays per view). All values: {value, lo, hi}."""
    res = {}
    T, _ = stat_tensor(preds, store_ids, fold_ids)
    per_fold, per_view, boot_views = summarize(T, fold_ids, boot, views)
    n_fold = preds.groupby("fold").size()
    res["overall"] = {
        "per_fold": {str(f): {**per_fold[f][0], "n_rows": int(n_fold.get(f, 0))} for f in fold_ids},
        "views": {v: {**per_view[v][0], "n_rows": int(sum(n_fold.get(f, 0) for f in views[v]))} for v in views},
    }
    for name, (col, order) in BREAKDOWNS.items():
        T, groups = stat_tensor(preds, store_ids, fold_ids, col, order)
        _, pv, _ = summarize(T, fold_ids, boot, views)
        n = preds.groupby([col, "fold"]).size()
        res[name] = {"groups": [str(g) for g in groups],
                     "views": {v: {str(g): {**pv[v][gi],
                                            "n_rows": int(sum(n.get((g, f), 0) for f in views[v]))}
                                   for gi, g in enumerate(groups)} for v in views}}
    return res, boot_views


def fallback_summary(diags: list) -> dict:
    tot = {}
    for d in diags:
        for k, v in d["fallback"].items():
            tot[k] = tot.get(k, 0) + v
    return tot


def error_concentration(preds: pd.DataFrame, top_frac: float = 0.001) -> dict:
    """Share of the summed squared percentage error that comes from the worst `top_frac` of rows, per fold."""
    out = {}
    for f, g in preds.groupby("fold"):
        sq = (((g["yhat"] - g["y"]) / g["y"]) ** 2).to_numpy()
        k = max(1, int(np.ceil(top_frac * len(sq))))
        out[str(f)] = {"n_rows": int(len(sq)), "top_k": k, "share_of_sq_pct_error": float(np.sort(sq)[::-1][:k].sum() / sq.sum()),
                       "min_actual_in_top_k": float(g["y"].to_numpy()[np.argsort(sq)[::-1][:k]].min())}
    return out
