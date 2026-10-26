"""Global LightGBM: fold row sets, early-stopped fit + refit, back-transform (docs/model_protocol.md sections 1, 3, 6)."""
import numpy as np
import pandas as pd

from demand_forecast.model_features import CATEGORICAL, FEATURES, HOLDOUT_OFFSETS_DAYS  # noqa: F401

FIXED_PARAMS = dict(learning_rate=0.05, max_bin=63, feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1,
                    lambda_l2=1.0, seed=20260101, num_threads=2, verbose=-1)
MAX_ROUNDS = 1500
PATIENCE = 50
GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (31, 63) for mc in (50, 200)]


def fold_masks(tab: pd.DataFrame, cf_idx: int) -> dict:
    """Row masks for one fold (cutoff index cf_idx). Training rows have target day t <= cf; the inner training rows are
    purged to t <= cf - 63 so that the holdout (cutoffs cf-63..cf-42, all rows with t <= cf, not subsampled) is strictly
    later than the inner training targets. The fold's evaluation rows (cutoff == cf) are never in train/holdout."""
    t, c = tab["t_idx"].to_numpy(), tab["cutoff_idx"].to_numpy()
    pos = (tab["sales"].to_numpy() > 0)
    keep = tab["keep"].to_numpy()
    train = keep & pos & (t <= cf_idx)
    inner = train & (t <= cf_idx - HOLDOUT_OFFSETS_DAYS[0])
    hold = pos & (t <= cf_idx) & np.isin(c, [cf_idx - o for o in HOLDOUT_OFFSETS_DAYS])
    ev = (c == cf_idx) & pos
    assert not (train & (c == cf_idx)).any() and not (hold & (c == cf_idx)).any()
    return {"train": train, "inner": inner, "holdout": hold, "eval": ev}


def design(tab: pd.DataFrame, mask, features, horizon_fixed=False) -> np.ndarray:
    X = tab.loc[mask, features].to_numpy(np.float32)
    if horizon_fixed and "horizon" in features:
        X[:, features.index("horizon")] = 1.0
    return X


def target(tab: pd.DataFrame, mask) -> np.ndarray:
    return np.log1p(tab.loc[mask, "sales"].to_numpy(np.float64)).astype(np.float32)


def backtransform(pred_log):
    return np.expm1(np.asarray(pred_log, float))


def smearing_factor(y_log, pred_log) -> float:
    """Duan smearing on the log1p scale: mean(exp(log1p(y) - pred))."""
    return float(np.mean(np.exp(np.asarray(y_log, float) - np.asarray(pred_log, float))))


def apply_smearing(pred_log, s: float):
    """(1 + expm1(pred)) * s - 1 = exp(pred) * s - 1."""
    return (1.0 + backtransform(pred_log)) * s - 1.0


def lgb_params(cfg: dict, objective="regression", alpha=None) -> dict:
    p = dict(FIXED_PARAMS, objective=objective, **cfg)
    if objective == "quantile":
        p.update(alpha=alpha, metric="quantile")
    else:
        p.update(metric="l2")
    return p


def fit_fold(tab, features, cfg, cf_idx, objective="regression", alpha=None, horizon_fixed=False):
    """Early-stop on the inner holdout, then refit on all training rows of the fold with the stopped number of trees.
    Returns (booster, info)."""
    import lightgbm as lgb
    m = fold_masks(tab, cf_idx)
    params = lgb_params(cfg, objective, alpha)
    cats = [f for f in CATEGORICAL if f in features]
    Xi, yi = design(tab, m["inner"], features, horizon_fixed), target(tab, m["inner"])
    Xh, yh = design(tab, m["holdout"], features, horizon_fixed), target(tab, m["holdout"])
    dtr = lgb.Dataset(Xi, yi, feature_name=features, categorical_feature=cats, free_raw_data=False)
    dho = lgb.Dataset(Xh, yh, reference=dtr, feature_name=features, categorical_feature=cats, free_raw_data=False)
    inner = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dho],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    best = int(inner.best_iteration or MAX_ROUNDS)
    hold_pred = inner.predict(Xh, num_iteration=best)
    info = {"best_iteration": best, "n_inner": int(len(yi)), "n_holdout": int(len(yh)),
            "holdout_score": float(inner.best_score["valid_0"][params["metric"]])}
    if objective == "regression":
        info["smearing_factor"] = smearing_factor(yh, hold_pred)
    del dtr, dho, Xi, yi
    Xt, yt = design(tab, m["train"], features, horizon_fixed), target(tab, m["train"])
    dfull = lgb.Dataset(Xt, yt, feature_name=features, categorical_feature=cats)
    booster = lgb.train(params, dfull, num_boost_round=best)
    info["n_train"] = int(len(yt))
    return booster, info


def predict_eval(booster, tab, features, cf_idx, horizon_fixed=False, X_override=None):
    m = fold_masks(tab, cf_idx)["eval"]
    X = design(tab, m, features, horizon_fixed) if X_override is None else X_override
    return tab.loc[m, ["Store", "t_idx"]].reset_index(drop=True), booster.predict(X)


def fix_crossing(lo, hi):
    """Sort the quantile pair; returns (lo, hi, number of crossings before the fix)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    crossed = int((lo > hi).sum())
    return np.minimum(lo, hi), np.maximum(lo, hi), crossed
