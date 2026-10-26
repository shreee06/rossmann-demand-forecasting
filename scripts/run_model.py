"""Step 3 pipeline (docs/model_protocol.md). Stages, each resumable (per-fit results are cached in data/processed/):

  features   build the (store, cutoff, target day) feature table
  grid       4 configurations x 8 folds (selection by mean WAPE) -> evals/results/model_grid_selection.json
  ablations  4 ablations x 8 folds with the selected configuration
  quantiles  q10 / q90 x 8 folds with the selected configuration
  analysis   promotion analysis (regression + model-based), importance on fold 8
  report     model_backtest / intervals / promo_analysis (.json, .md)
  validation ONE run on VALIDATION: selected model, its quantile models and the best baseline

The TEST window is never loaded; the leakage column named in the README is never read (explicit column lists).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecast import config, gbm, intervals  # noqa: E402
from demand_forecast import model_features as MF  # noqa: E402
from demand_forecast import promo_analysis as PA  # noqa: E402
from demand_forecast.backtest import backtest_splits, load_modelling_frame, run_splits, validation_split  # noqa: E402
from demand_forecast.baselines import PromoWeekdayMean  # noqa: E402
from demand_forecast.bootstrap import N_BOOT, SEED, StoreBootstrap, paired_difference  # noqa: E402
from demand_forecast.data import load_processed  # noqa: E402
from demand_forecast.evaluation import BACKTEST_VIEWS, BREAKDOWNS, evaluate  # noqa: E402
from demand_forecast.memguard import MemGuard  # noqa: E402
from demand_forecast.metrics import METRICS, STATS, add_error_columns, metrics_from_stats  # noqa: E402

RES = config.RESULTS_DIR
PROC = config.PROCESSED_DIR
RUNS = PROC / "model_runs"
BOOSTERS = PROC / "models"
TABLE = PROC / "model_table.parquet"
AUX = PROC / "model_aux.parquet"
SPEC_LOG = RES / "spec_log.csv"
LOG_FIELDS = ["run_id", "stage", "baseline", "params", "split", "n_rows", "rmspe", "wape", "mase", "bias",
              "runtime_s", "status", "note"]
FOLDS = list(range(1, 9))
IMPORTANCE_FOLD = 8
BASELINE_K = 12
VARIANTS = ["full", "full_smearing", "no_history", "no_promo", "no_holiday", "horizon_fixed_1"]
LABEL = {"baseline": f"Baseline: promo-aware weekday mean (K={BASELINE_K})", "full": "Global LightGBM (full)",
         "full_smearing": "Full + smearing (secondary)", "no_history": "Ablation: no history features",
         "no_promo": "Ablation: no Promo features", "no_holiday": "Ablation: no holiday features",
         "horizon_fixed_1": "Ablation: horizon fixed to 1"}
RUN_ID = datetime.now().strftime("%Y%m%dT%H%M%S")


# ------------------------------------------------------------------ helpers
def cfg_id(cfg):
    return f"nl{cfg['num_leaves']}_mc{cfg['min_child_samples']}"


def log_rows(rows):
    new = not SPEC_LOG.exists()
    with open(SPEC_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"run_id": RUN_ID, **{k: r.get(k, "") for k in LOG_FIELDS if k != "run_id"}})


def fmt(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.6g}"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def load_inputs():
    frame = load_modelling_frame()
    assert frame["Date"].max() < pd.Timestamp("2015-06-20"), "TEST rows must not be loaded"
    _, store = load_processed()
    return frame, store


def date0_of(frame):
    return frame["Date"].min()


def base_eval(frame, store, splits, date0):
    """Evaluated rows + baseline forecast (promo-aware weekday mean, K=12) with all breakdown columns."""
    fc = PromoWeekdayMean(BASELINE_K)
    p, _ = run_splits(fc, frame, store, splits)
    p["t_idx"] = (p["Date"] - date0).dt.days.astype(np.int16)
    return p


def attach(base, runs, col="pred"):
    m = base.merge(runs[["fold", "Store", "t_idx", col]], on=["fold", "Store", "t_idx"], how="left", validate="one_to_one")
    assert len(m) == len(base) and m[col].notna().all(), "model predictions do not align with the evaluated rows"
    return m


def score(base, runs, col="pred"):
    m = attach(base, runs, col)
    m = m.assign(yhat=m[col])
    S = add_error_columns(m).groupby("fold")[STATS].sum()
    return pd.DataFrame(metrics_from_stats(S.to_numpy()), index=S.index, columns=METRICS).assign(n=S["n"].to_numpy())


def run_path(kind, key, fold):
    return RUNS / f"{kind}_{key}_f{fold}.parquet"


def fit_and_store(tab, kind, key, features, cfg, fold, cf_idx, base, date0, objective="regression", alpha=None,
                  horizon_fixed=False, status="tried", stage="model-grid", params_extra=None):
    """Fit one (kind, key, fold) unless cached; save predictions + booster; append to the spec log."""
    path = run_path(kind, key, fold)
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    booster, info = gbm.fit_fold(tab, features, cfg, cf_idx, objective, alpha, horizon_fixed)
    keys, pred_log = gbm.predict_eval(booster, tab, features, cf_idx, horizon_fixed)
    keys["fold"] = fold
    keys["pred_log"] = pred_log
    keys["pred"] = gbm.backtransform(pred_log)
    if objective == "regression":
        keys["pred_smear"] = gbm.apply_smearing(pred_log, info["smearing_factor"])
    runs = keys
    BOOSTERS.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(BOOSTERS / f"{kind}_{key}_f{fold}.txt"))
    runs.to_parquet(path, index=False)
    (RUNS / f"{kind}_{key}_f{fold}.json").write_text(json.dumps(clean(info)))
    dt = time.time() - t0
    row = {"stage": stage, "baseline": f"lgbm_{kind}", "params": json.dumps({**cfg, **(params_extra or {})}),
           "split": fold, "runtime_s": f"{dt:.1f}", "status": status}
    if objective == "regression":
        sc = score(base[base["fold"] == fold], runs).iloc[0]
        row.update(n_rows=int(sc["n"]), **{k: fmt(sc[k]) for k in METRICS}, note=f"best_iteration={info['best_iteration']}; n_train={info['n_train']}")
    else:
        b = attach(base[base["fold"] == fold], runs)
        row.update(n_rows=len(b), note=f"alpha={alpha}; best_iteration={info['best_iteration']}; n_train={info['n_train']}")
    log_rows([row])
    return runs


def load_runs(kind, key, folds=FOLDS):
    return pd.concat([pd.read_parquet(run_path(kind, key, f)) for f in folds], ignore_index=True)


def get_table():
    return pd.read_parquet(TABLE)


# ------------------------------------------------------------------ stages
def stage_features(mg):
    frame, store = load_inputs()
    t0 = time.time()
    evals = [sp.cutoff for sp in backtest_splits() + [validation_split()]]
    info = MF.stream_feature_files(frame, store, evals, TABLE, AUX)
    mg.stage(f"feature table written {info['rows_written']:,} of {info['rows_built']:,} rows")
    tab = pd.read_parquet(TABLE, columns=["cutoff_idx", "t_idx", "keep", "sales"])
    d0 = frame["Date"].min()
    assert int(tab["t_idx"].max()) <= (frame["Date"].max() - d0).days
    assert tab["sales"].min() > 0
    cut = set(int(c) for c in tab["cutoff_idx"].unique())
    for c in evals:
        assert int((c - d0).days) in cut, f"missing cutoff {c.date()}"
    info.update({"rows_kept_for_training_subsample": int(tab["keep"].sum()), "seconds": time.time() - t0,
                 "first_cutoff_index": int(tab["cutoff_idx"].min()), "last_cutoff_index": int(tab["cutoff_idx"].max()),
                 "columns_main": list(pd.read_parquet(TABLE).columns[:0]) or None})
    info.pop("columns_main")
    (RES / "model_table_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=1))


def stage_grid(mg, only_first=False):
    frame, store = load_inputs()
    date0 = date0_of(frame)
    tab = get_table()
    splits = backtest_splits()
    base = base_eval(frame, store, splits, date0)
    cf = {s.fold: int((s.cutoff - date0).days) for s in splits}
    mg.stage("grid: inputs ready")
    for cfg in gbm.GRID:
        for f in FOLDS:
            fit_and_store(tab, "full", cfg_id(cfg), MF.features_for(), cfg, f, cf[f], base, date0)
            mg.stage(f"grid {cfg_id(cfg)} fold {f}")
            if only_first:
                return
    # selection by mean WAPE over the 8 folds
    sel = {"criterion": "mean-over-folds WAPE", "configs": []}
    for cfg in gbm.GRID:
        runs = load_runs("full", cfg_id(cfg))
        sc = score(base, runs)
        infos = [json.loads(run_path("full", cfg_id(cfg), f).with_suffix(".json").read_text()) for f in FOLDS]
        sel["configs"].append({"config": cfg, "id": cfg_id(cfg), **{f"mean_{m}": float(sc[m].mean()) for m in METRICS},
                               "per_fold_wape": {str(f): float(sc.loc[f, "wape"]) for f in FOLDS},
                               "best_iteration_per_fold": {str(f): i["best_iteration"] for f, i in zip(FOLDS, infos)}})
    best = min(sel["configs"], key=lambda c: c["mean_wape"])
    sel["selected"] = best["id"]
    sel["selected_config"] = best["config"]
    for c in sel["configs"]:
        c["status"] = "selected" if c["id"] == best["id"] else "dropped"
        log_rows([{"stage": "model-grid", "baseline": "lgbm_full", "params": json.dumps(c["config"]), "split": "mean_over_8_folds",
                   "rmspe": fmt(c["mean_rmspe"]), "wape": fmt(c["mean_wape"]), "mase": fmt(c["mean_mase"]), "bias": fmt(c["mean_bias"]),
                   "status": c["status"], "note": "selected on mean WAPE over the 8 backtest folds" if c["status"] == "selected" else "not best on mean WAPE; dropped"}])
    (RES / "model_grid_selection.json").write_text(json.dumps(clean(sel), indent=2) + "\n")
    print(json.dumps({c["id"]: round(c["mean_wape"], 4) for c in sel["configs"]}), "selected:", sel["selected"])


def selected_cfg():
    sel = json.loads((RES / "model_grid_selection.json").read_text())
    return sel["selected_config"], sel["selected"]


def stage_ablations(mg):
    frame, store = load_inputs()
    date0 = date0_of(frame)
    tab = get_table()
    splits = backtest_splits()
    base = base_eval(frame, store, splits, date0)
    cf = {s.fold: int((s.cutoff - date0).days) for s in splits}
    cfg, cid = selected_cfg()
    for ab in ("no_history", "no_promo", "no_holiday", "horizon_fixed_1"):
        feats = MF.features_for(ab)
        for f in FOLDS:
            fit_and_store(tab, ab, cid, feats, cfg, f, cf[f], base, date0, horizon_fixed=(ab == "horizon_fixed_1"),
                          status="ablation", stage="model-ablation", params_extra={"ablation": ab})
            mg.stage(f"ablation {ab} fold {f}")


def stage_quantiles(mg):
    frame, store = load_inputs()
    date0 = date0_of(frame)
    tab = get_table()
    splits = backtest_splits()
    base = base_eval(frame, store, splits, date0)
    cf = {s.fold: int((s.cutoff - date0).days) for s in splits}
    cfg, cid = selected_cfg()
    for a in (0.1, 0.9):
        for f in FOLDS:
            fit_and_store(tab, f"q{int(a * 100)}", cid, MF.features_for(), cfg, f, cf[f], base, date0, objective="quantile", alpha=a,
                          status="quantile", stage="model-quantile", params_extra={"alpha": a})
            mg.stage(f"quantile {a} fold {f}")


# ---- promotion analysis
def promo_regressions(frame, store, g):
    """Within-store regressions on TRAIN rows (<= 2015-05-08), open Mon-Fri days with Sales > 0."""
    train_end = pd.Timestamp("2015-05-08")
    d0 = g.dates[0]
    f = frame[(frame["Date"] <= train_end) & (frame["Open"] == 1) & (frame["Sales"] > 0) & (frame["Date"].dt.dayofweek < 5)].copy()
    f["si"] = np.searchsorted(g.store_ids, f["Store"].to_numpy())
    f["di"] = (f["Date"] - d0).dt.days.to_numpy()
    y = np.log(f["Sales"].to_numpy(float))
    promo = f["Promo"].to_numpy(float)
    st_c, dow_c = PA.codes(f["Store"].to_numpy()), PA.codes(f["Date"].dt.dayofweek.to_numpy())
    wk_c = PA.codes(f["Date"].dt.isocalendar().week.to_numpy().astype(int))
    sd_c = PA.codes(f["Store"].to_numpy() * 10 + f["Date"].dt.dayofweek.to_numpy())
    hol = np.column_stack([(f["StateHoliday"].astype(str) != "0").to_numpy(float), (f["SchoolHoliday"] == 1).to_numpy(float)])
    cluster = f["Store"].to_numpy()
    stype = store.set_index("Store")["StoreType"].astype(str).reindex(f["Store"]).to_numpy()

    def pack(r, names):
        return {n: {"coef": float(r["coef"][i]), "se": float(r["se"][i]), "lo": float(r["lo"][i]), "hi": float(r["hi"][i]),
                    "pct": float(np.expm1(r["coef"][i]) * 100), "pct_lo": float(np.expm1(r["lo"][i]) * 100),
                    "pct_hi": float(np.expm1(r["hi"][i]) * 100)} for i, n in enumerate(names)} | {"n": r["n"], "n_clusters": r["n_clusters"]}

    out = {"sample": "TRAIN rows (<= 2015-05-08), open Monday-Friday days with Sales > 0",
           "outcome": "log(Sales); coefficient on Promo; pct = exp(coef) - 1 (adjusted association)",
           "se": "store-clustered (cluster-robust sandwich, factor G/(G-1))", "specs": {}}
    out["specs"]["A_store_dow_week"] = {"fixed_effects": "store, day of week, ISO week of year",
                                        **pack(PA.fe_regression(y, promo, [st_c, dow_c, wk_c], cluster), ["promo"])}
    X = np.column_stack([promo, hol])
    names_b = ["promo", "state_holiday", "school_holiday"]
    out["specs"]["B_plus_holidays"] = {"fixed_effects": "store, day of week, ISO week of year; + holiday indicators",
                                       **pack(PA.fe_regression(y, X, [st_c, dow_c, wk_c], cluster), names_b)}
    out["specs"]["C_store_x_dow"] = {"fixed_effects": "store x day of week, ISO week of year; + holiday indicators",
                                     **pack(PA.fe_regression(y, X, [sd_c, wk_c], cluster), names_b)}
    # D: neighbouring-day / neighbouring-week promo flags (rows with any unknown neighbour are dropped)
    nb = {k: g.promo[f["si"].to_numpy(), np.clip(f["di"].to_numpy() + d, 0, g.D - 1)] for k, d in
          {"promo_prev_day": -1, "promo_next_day": 1, "promo_prev_week": -7, "promo_next_week": 7}.items()}
    ok = np.all([~np.isnan(v) for v in nb.values()], axis=0)
    ok &= (f["di"].to_numpy() + 7 <= g.D - 1) & (f["di"].to_numpy() - 7 >= 0)
    Xd = np.column_stack([promo, hol] + [nb[k] for k in nb])
    ids = [PA.codes(v[ok]) for v in (f["Store"].to_numpy(), f["Date"].dt.dayofweek.to_numpy(),
                                     f["Date"].dt.isocalendar().week.to_numpy().astype(int))]
    rd = PA.fe_regression(y[ok], Xd[ok], ids, cluster[ok])
    out["specs"]["D_plus_neighbouring_promo_flags"] = {
        "fixed_effects": "store, day of week, ISO week of year; + holiday indicators and promo flags of previous/next day and week "
                         "(timing pattern around promo days, not effects)", "rows_dropped_for_unknown_neighbours": int((~ok).sum()),
        **pack(rd, names_b + list(nb))}
    out["by_store_type_spec_B"] = {}
    for stp in sorted(set(stype)):
        m = stype == stp
        idm = [PA.codes(v[m]) for v in (f["Store"].to_numpy(), f["Date"].dt.dayofweek.to_numpy(),
                                        f["Date"].dt.isocalendar().week.to_numpy().astype(int))]
        out["by_store_type_spec_B"][stp] = pack(PA.fe_regression(y[m], X[m], idm, cluster[m]), names_b)
    return out


def counterfactual_ratios(tab, store, date0, cfg_key, boot, store_ids):
    """Model-based predicted ratio, Promo=1 vs Promo=0 (everything else fixed), on each backtest fold's evaluated
    Monday-Friday rows with that fold's full model. Two variants for the neighbouring-week flags."""
    import lightgbm as lgb
    feats = MF.features_for()
    stypes = store.set_index("Store")["StoreType"].astype(str)
    splits = {s.fold: s for s in backtest_splits()}
    aux = pd.read_parquet(AUX)
    res = {}
    for variant in ("neighbours_as_observed", "neighbours_alternating"):
        rows = []
        for f in FOLDS:
            cf_idx = int((splits[f].cutoff - date0).days)
            m = gbm.fold_masks(tab, cf_idx)["eval"] & (tab["dow"].to_numpy() < 4.5)
            sub = tab.loc[m].merge(aux, on=["Store", "cutoff_idx", "t_idx"], how="left", validate="one_to_one")
            assert sub["swp1_mean_4"].notna().sum() > 0 and len(sub) == int(m.sum())
            booster = lgb.Booster(model_file=str(BOOSTERS / f"full_{cfg_key}_f{f}.txt"))
            preds = {}
            for p in (1, 0):
                X = sub[feats].copy()
                X["promo"] = float(p)
                for K in (4, 8, 12):
                    X[f"swp_mean_{K}"] = sub[f"swp{p}_mean_{K}"].to_numpy()
                X["swp_n_12"] = sub[f"swp{p}_n_12"].to_numpy()
                if variant == "neighbours_alternating":
                    X["promo_m7"] = float(1 - p)
                    X["promo_p7"] = np.where(sub["promo_p7"].isna(), np.nan, float(1 - p))
                preds[p] = booster.predict(X.to_numpy(np.float32))
            r = np.expm1(preds[1]) / np.maximum(np.expm1(preds[0]), 1e-6)
            rows.append(pd.DataFrame({"fold": f, "Store": sub["Store"].to_numpy(), "log_ratio": np.log(np.maximum(r, 1e-6)),
                                      "observed_promo": sub["promo"].to_numpy()}))
        d = pd.concat(rows, ignore_index=True)
        d["store_type"] = d["Store"].map(stypes).to_numpy()
        res[variant] = summarize_ratios(d, boot, store_ids)
    return res



def summarize_ratios(d, boot, store_ids):
    """Geometric-mean ratio with store-bootstrap CI overall, by store type, and by observed promo; spread of row/store ratios."""
    spos = {s: i for i, s in enumerate(store_ids)}

    def boot_gm(sub):
        sums = np.zeros(len(store_ids))
        cnt = np.zeros(len(store_ids))
        agg = sub.groupby("Store")["log_ratio"].agg(["sum", "size"])
        idx = agg.index.map(spos).to_numpy()
        sums[idx], cnt[idx] = agg["sum"].to_numpy(), agg["size"].to_numpy()
        b = (boot.counts @ sums) / np.maximum(boot.counts @ cnt, 1)
        lo, hi = np.percentile(np.exp(b), [2.5, 97.5])
        store_level = np.exp(agg["sum"] / agg["size"])
        r = np.exp(sub["log_ratio"].to_numpy())
        return {"rows": int(len(sub)), "geometric_mean_ratio": float(np.exp(sums.sum() / max(cnt.sum(), 1))), "ci_lo": float(lo), "ci_hi": float(hi),
                "mean_ratio": float(r.mean()), "row_ratio_p10": float(np.percentile(r, 10)), "row_ratio_p50": float(np.percentile(r, 50)),
                "row_ratio_p90": float(np.percentile(r, 90)), "store_ratio_p10": float(np.percentile(store_level, 10)),
                "store_ratio_p50": float(np.percentile(store_level, 50)), "store_ratio_p90": float(np.percentile(store_level, 90)),
                "n_stores": int(len(store_level))}

    out = {"overall": boot_gm(d), "by_store_type": {k: boot_gm(g) for k, g in sorted(d.groupby("store_type"))},
           "by_observed_promo": {("promo" if k == 1 else "no promo"): boot_gm(g) for k, g in sorted(d.groupby("observed_promo"))}}
    return out


def permutation_importance(tab, fold, cfg_key, date0, repeats=3):
    import lightgbm as lgb
    feats = MF.features_for()
    splits = {s.fold: s for s in backtest_splits()}
    cf_idx = int((splits[fold].cutoff - date0).days)
    m = gbm.fold_masks(tab, cf_idx)["eval"]
    sub = tab.loc[m]
    booster = lgb.Booster(model_file=str(BOOSTERS / f"full_{cfg_key}_f{fold}.txt"))
    X = sub[feats].to_numpy(np.float32)
    y = sub["sales"].to_numpy(float)

    def wape(Xm):
        return float(np.abs(gbm.backtransform(booster.predict(Xm)) - y).sum() / y.sum())

    base = wape(X)
    rng = np.random.default_rng(SEED)
    out = {"fold": fold, "baseline_wape": base, "repeats": repeats, "features": {}, "groups": {}}
    gain = booster.feature_importance("gain")
    for j, ft in enumerate(feats):
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            inc.append(wape(Xp) - base)
        out["features"][ft] = {"gain_share": float(gain[j] / gain.sum()), "permutation_wape_increase": float(np.mean(inc))}
    groups = {"horizon": MF.HORIZON_FEATURES, "calendar": MF.CALENDAR_FEATURES, "holiday": MF.HOLIDAY_FEATURES,
              "promo (flags, Promo2, promo-matched history)": MF.PROMO_FEATURES + MF.HISTORY_PROMO, "store attributes": MF.STORE_FEATURES,
              "history (all)": MF.HISTORY_FEATURES}
    for name, cols in groups.items():
        idx = [feats.index(c) for c in cols]
        inc = []
        for _ in range(repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[:, idx] = Xp[perm][:, idx]
            inc.append(wape(Xp) - base)
        out["groups"][name] = {"features": cols, "permutation_wape_increase": float(np.mean(inc)),
                               "gain_share": float(sum(gain[feats.index(c)] for c in cols) / gain.sum())}
    return out


def stage_analysis(mg):
    frame, store = load_inputs()
    date0 = date0_of(frame)
    tab = get_table()
    g = MF.Grids(frame, store)
    store_ids = sorted(store["Store"].unique().tolist())
    boot = StoreBootstrap(len(store_ids))
    _, cid = selected_cfg()
    mg.stage("analysis: inputs ready")
    promo = {"protocol": "docs/model_protocol.md section 7", "association_not_causal": True}
    promo["regression"] = promo_regressions(frame, store, g)
    mg.stage("promo regressions done")
    promo["model_based"] = counterfactual_ratios(tab, store, date0, cid, boot, store_ids)
    promo["model_based"]["definition"] = ("ratio of predicted sales with the planned Promo flag of the target day set to 1 vs 0 "
                                          "(promo-matched history means switched accordingly), geometric mean over each fold's evaluated "
                                          "Mon-Fri rows, pooled over the 8 backtest folds, with each fold's selected full model")
    mg.stage("model-based comparison done")
    (RES / "promo_analysis_raw.json").write_text(json.dumps(clean(promo), indent=2) + "\n")
    imp = permutation_importance(tab, IMPORTANCE_FOLD, cid, date0)
    (RES / "importance_raw.json").write_text(json.dumps(clean(imp), indent=2) + "\n")
    mg.stage("importance done")


# ------------------------------------------------------------------ reporting
def cell(c, nd=3):
    return "n/a" if c is None or c["value"] is None else f"{c['value']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


def variant_runs(name, cid, folds):
    if name == "full":
        return load_runs("full", cid, folds), "pred"
    if name == "full_smearing":
        return load_runs("full", cid, folds), "pred_smear"
    return load_runs(name, cid, folds), "pred"


def build_variant_frames(base, cid, folds, names):
    out = {"baseline": base}
    for n in names:
        runs, col = variant_runs(n, cid, folds)
        m = attach(base, runs, col)
        out[n] = m.assign(yhat=m[col].clip(lower=0))
    return out


def evaluate_all(frames, store_ids, fold_ids, boot, views):
    res, bv = {}, {}
    for n, p in frames.items():
        res[n], bv[n] = evaluate(p, store_ids, fold_ids, boot, views)
    return res, bv


def paired(bv, res, a, b, view, metrics=METRICS):
    out = {}
    for m in metrics:
        j = METRICS.index(m)
        out[m] = paired_difference(bv[a][view][:, 0, j], bv[b][view][:, 0, j], res[a]["overall"]["views"][view][m]["value"],
                                   res[b]["overall"]["views"][view][m]["value"])
    return out


def render_comparison(title, res, bv, names, views_labels, extra_sections=(), ref="baseline", note_lines=()):
    L, add = [], None
    add = L.append
    first = list(views_labels)[0]
    add(f"# {title}\n")
    for ln in note_lines:
        add(ln + "\n")
    add("Metrics on open days with Sales > 0 (RMSPE, WAPE, MASE, Bias = sum(forecast-actual)/sum(actual)); 95% store-clustered "
        "bootstrap CIs (500 resamples, seed 20260101) as value [lo, hi]. 'Mean over folds' = unweighted mean of per-fold metrics. "
        "Paired differences use the same store resamples.\n")
    add("## Overall\n")
    for v, vl in views_labels.items():
        add(f"**{vl}**\n")
        add("| Forecaster | RMSPE | WAPE | MASE | Bias |\n|---|---|---|---|---|")
        for n in names:
            o = res[n]["overall"]["views"][v]
            add(f"| {LABEL[n]} | " + " | ".join(cell(o[m], 4 if m != "mase" else 3) for m in METRICS) + " |")
        add("")
    folds = list(res[names[0]]["overall"]["per_fold"])
    if len(folds) > 1:
        for m in METRICS:
            add(f"### {m.upper()} by fold\n")
            add("| Fold | " + " | ".join(LABEL[n] for n in names) + " |\n|---|" + "---|" * len(names))
            for f in folds:
                add(f"| {f} | " + " | ".join(cell(res[n]["overall"]["per_fold"][f][m]) for n in names) + " |")
            add("")
    for key, ttl in [("horizon", "Horizon (days after cutoff)"), ("store_type", "Store type"), ("promo", "Promo vs non-promo target day"),
                     ("dow", "Day of week (1 = Monday)")]:
        add(f"## By {ttl} ({views_labels[first]})\n")
        groups = res[names[0]][key]["groups"]
        for m in METRICS:
            add(f"**{m.upper()}**\n")
            add("| Group | rows | " + " | ".join(LABEL[n] for n in names) + " |\n|---|---|" + "---|" * len(names))
            for gname in groups:
                nrows = res[names[0]][key]["views"][first][gname]["n_rows"]
                add(f"| {gname} | {nrows:,} | " + " | ".join(cell(res[n][key]["views"][first][gname][m]) for n in names) + " |")
            add("")
    for s in extra_sections:
        add(s)
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ report stage
def stage_report(mg):
    frame, store = load_inputs()
    date0 = date0_of(frame)
    splits = backtest_splits()
    base = base_eval(frame, store, splits, date0)
    store_ids = sorted(store["Store"].unique().tolist())
    boot = StoreBootstrap(len(store_ids))
    cfg, cid = selected_cfg()
    sel = json.loads((RES / "model_grid_selection.json").read_text())
    frames = build_variant_frames(base, cid, FOLDS, VARIANTS)
    res, bv = evaluate_all(frames, store_ids, FOLDS, boot, BACKTEST_VIEWS)
    mg.stage("backtest variants evaluated")
    views_labels = {"all_folds": "mean over all 8 folds", "folds_2_5": "mean over folds 2-5 (Jul-Dec 2014 gap period)",
                    "other_folds": "mean over folds 1, 6, 7, 8"}
    comp = {"full_vs_baseline": {v: paired(bv, res, "full", "baseline", v) for v in BACKTEST_VIEWS},
            "smearing_vs_baseline": {v: paired(bv, res, "full_smearing", "baseline", v) for v in BACKTEST_VIEWS},
            "ablations_vs_full": {ab: {v: paired(bv, res, ab, "full", v) for v in BACKTEST_VIEWS}
                                  for ab in ("no_history", "no_promo", "no_holiday", "horizon_fixed_1")}}
    best_group = {}
    for key in BREAKDOWNS:
        gs = res["full"][key]["groups"]
        best_group[key] = {}
        for gname in gs:
            a, b = res["full"][key]["views"]["all_folds"][gname]["wape"]["value"], res["baseline"][key]["views"]["all_folds"][gname]["wape"]["value"]
            best_group[key][gname] = "n/a" if a is None or b is None or np.isnan(a) or np.isnan(b) else ("full model" if a < b else "baseline")
    fold_wins = {str(f): ("full model" if res["full"]["overall"]["per_fold"][str(f)]["wape"]["value"] < res["baseline"]["overall"]["per_fold"][str(f)]["wape"]["value"] else "baseline")
                 for f in FOLDS}
    imp = json.loads((RES / "importance_raw.json").read_text())
    infos = {n: [json.loads(run_path(n if n != "full" else "full", cid, f).with_suffix(".json").read_text())["best_iteration"] for f in FOLDS]
             for n in ("full", "no_history", "no_promo", "no_holiday", "horizon_fixed_1")}
    smears = [json.loads(run_path("full", cid, f).with_suffix(".json").read_text())["smearing_factor"] for f in FOLDS]
    out = {"protocol": "docs/model_protocol.md", "run_id": RUN_ID, "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "clustered_by": "Store"},
           "grid": sel, "selected_config": cfg, "features": MF.features_for(), "fixed_params": gbm.FIXED_PARAMS,
           "max_rounds": gbm.MAX_ROUNDS, "early_stopping_patience": gbm.PATIENCE, "training_row_fraction": 0.2,
           "best_iteration_per_fold": infos, "smearing_factor_per_fold": smears, "variants": res, "comparison": comp,
           "full_vs_baseline_winner_by_group_wape": best_group, "full_vs_baseline_winner_by_fold_wape": fold_wins,
           "importance_fold_8": imp}
    out["peak_rss_mb_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    out["peak_rss_mb_children"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    (RES / "model_backtest.json").write_text(json.dumps(clean(out), indent=2) + "\n")

    # ---- markdown
    names = ["baseline"] + VARIANTS
    gl = ["## Hyperparameter grid (mean over the 8 backtest folds; selection criterion: mean WAPE)\n",
          "| Config | Mean RMSPE | Mean WAPE | Mean MASE | Mean Bias | Trees per fold (stopped) | Status |\n|---|---|---|---|---|---|---|"]
    for c in sel["configs"]:
        gl.append(f"| {c['id']} | {c['mean_rmspe']:.4f} | {c['mean_wape']:.4f} | {c['mean_mase']:.3f} | {c['mean_bias']:+.4f} | "
                  f"{list(c['best_iteration_per_fold'].values())} | {c['status']} |")
    gl.append("\nTuned and scored on the same folds, so backtest numbers are mildly optimistic; VALIDATION is the clean check.\n")
    pd_lines = ["## Paired differences (variant minus comparison; negative WAPE/RMSPE/MASE = variant better). Paired store bootstrap.\n"]
    for lbl, d in [("Full model minus baseline", comp["full_vs_baseline"]), ("Full + smearing minus baseline", comp["smearing_vs_baseline"])]:
        pd_lines.append(f"**{lbl}**\n")
        pd_lines.append("| View | RMSPE | WAPE | MASE | Bias |\n|---|---|---|---|---|")
        for v, vl in views_labels.items():
            pd_lines.append(f"| {vl} | " + " | ".join(f"{d[v][m]['diff']:+.4f} [{d[v][m]['lo']:+.4f}, {d[v][m]['hi']:+.4f}]" for m in METRICS) + " |")
        pd_lines.append("")
    pd_lines.append("**Ablation minus full model** (positive WAPE = removing the group hurts)\n")
    pd_lines.append("| Ablation | View | RMSPE | WAPE | MASE | Bias |\n|---|---|---|---|---|---|")
    for ab, dd in comp["ablations_vs_full"].items():
        for v, vl in views_labels.items():
            pd_lines.append(f"| {LABEL[ab]} | {vl} | " + " | ".join(f"{dd[v][m]['diff']:+.4f} [{dd[v][m]['lo']:+.4f}, {dd[v][m]['hi']:+.4f}]" for m in METRICS) + " |")
    pd_lines.append("")
    win = ["## Where does the full model beat the baseline? (WAPE point estimates, mean over all folds)\n",
           f"- By fold: {fold_wins}", "- By group: " + "; ".join(f"{k}: {v}" for k, v in best_group.items()) + "\n"]
    imp_l = [f"## Feature importance on fold {IMPORTANCE_FOLD} (selected full model). Not causal; correlated features share credit.\n",
             f"Baseline WAPE of this fold's model: {imp['baseline_wape']:.4f}. Permutation = mean WAPE increase over {imp['repeats']} shuffles (seed {SEED}); gain = share of total gain.\n",
             "| Feature | gain share | permutation WAPE increase |\n|---|---|---|"]
    for ft, v in sorted(imp["features"].items(), key=lambda kv: -kv[1]["gain_share"]):
        imp_l.append(f"| {ft} | {v['gain_share']:.4f} | {v['permutation_wape_increase']:+.5f} |")
    imp_l.append("\n**Feature groups** (jointly shuffled)\n")
    imp_l.append("| Group | gain share | permutation WAPE increase |\n|---|---|---|")
    for k, v in imp["groups"].items():
        imp_l.append(f"| {k} | {v['gain_share']:.4f} | {v['permutation_wape_increase']:+.5f} |")
    run_info = [f"\n## Run info\n", f"- Smearing factor per fold (exp scale): {[round(s, 3) for s in smears]}",
                f"- Trees per fold: {infos}", f"- Peak RSS (main process): {out['peak_rss_mb_self']} MB; children: {out['peak_rss_mb_children']} MB."]
    md = render_comparison("Global LightGBM — rolling-origin backtest (8 folds inside TRAIN)", res, bv, names, views_labels,
                           extra_sections=["\n".join(gl), "\n".join(pd_lines), "\n".join(win), "\n".join(imp_l), "\n".join(run_info)],
                           note_lines=[f"_Generated by `scripts/run_model.py` (run `{RUN_ID}`); protocol `docs/model_protocol.md`. Selected configuration: "
                                       f"`{cid}`. Training uses a fixed 20% row subsample; the TEST window is untouched._",
                                       "The primary model's back-transformed forecasts are expected to sit below the mean (negative Bias); "
                                       "the smearing variant is secondary and was not used for selection."])
    (RES / "model_backtest.md").write_text(md)
    # ---- frozen selection
    (RES / "model_selected.json").write_text(json.dumps(clean({
        "frozen": True, "note": "selected on the 8 backtest folds only (mean WAPE); VALIDATION and TEST not used", "config": cfg, "id": cid,
        "features": MF.features_for(), "fixed_params": gbm.FIXED_PARAMS, "max_rounds": gbm.MAX_ROUNDS,
        "early_stopping_patience": gbm.PATIENCE, "training_row_fraction": 0.2, "target": "log1p(Sales), open days with Sales > 0",
        "quantile_alphas": [0.1, 0.9], "backtest_trees_per_fold": infos["full"]}), indent=2) + "\n")
    mg.stage("model_backtest written")

    # ---- intervals
    q = {}
    for a in ("q10", "q90"):
        q[a] = load_runs(a, cid)
    lo_hi = attach(base, q["q10"], "pred").rename(columns={"pred": "lo_raw"})
    lo_hi = attach(lo_hi, q["q90"], "pred").rename(columns={"pred": "hi_raw"})
    lo, hi, crossed = gbm.fix_crossing(lo_hi["lo_raw"], lo_hi["hi_raw"])
    iv = lo_hi.assign(lo=lo, hi=hi)
    crossing_by_fold = {str(f): int(((g_["lo_raw"] > g_["hi_raw"])).sum()) for f, g_ in lo_hi.groupby("fold")}
    T, _ = intervals.tensor(iv, store_ids, FOLDS)
    pf, pv = intervals.summarize(T, FOLDS, boot, BACKTEST_VIEWS)
    results = {"per_fold": {str(f): pf[f][0] for f in FOLDS}, "views": {v: pv[v][0] for v in BACKTEST_VIEWS}}
    for key, (col, order) in BREAKDOWNS.items():
        if key in ("dow", "history", "row_coverage"):
            continue
        T, groups = intervals.tensor(iv, store_ids, FOLDS, col, order)
        _, pvv = intervals.summarize(T, FOLDS, boot, BACKTEST_VIEWS)
        results[key] = {str(gname): {v: pvv[v][gi] for v in BACKTEST_VIEWS} for gi, gname in enumerate(groups)}
    ivout = {"protocol": "docs/model_protocol.md section 6", "nominal_coverage": intervals.NOMINAL, "run_id": RUN_ID,
             "quantile_models": {"alphas": [0.1, 0.9], "config": cfg, "trees_per_fold": {a: [json.loads(run_path(a, cid, f).with_suffix(".json").read_text())["best_iteration"] for f in FOLDS] for a in ("q10", "q90")}},
             "crossing_before_fix": {"total": crossed, "by_fold": crossing_by_fold, "rows": int(len(iv))},
             "backtest": results, "conformal": "not attempted (see protocol section 6)"}
    # coverage by horizon bucket within folds 2-5 vs others is available through views; also per fold x horizon bucket
    from demand_forecast.evaluation import HORIZON_ORDER
    Th, gh = intervals.tensor(iv, store_ids, FOLDS, "horizon_bucket", HORIZON_ORDER)
    pfh, _ = intervals.summarize(Th, FOLDS, boot, BACKTEST_VIEWS)
    ivout["backtest"]["horizon_bucket_by_fold_coverage"] = {str(f): {gname: pfh[f][gi]["coverage"] for gi, gname in enumerate(gh)} for f in FOLDS}
    (RES / "intervals_backtest_raw.json").write_text(json.dumps(clean(ivout), indent=2) + "\n")
    mg.stage("intervals evaluated")

    # ---- promo analysis report
    promo = json.loads((RES / "promo_analysis_raw.json").read_text())
    (RES / "promo_analysis.json").write_text(json.dumps(promo, indent=2) + "\n")
    (RES / "promo_analysis.md").write_text(render_promo(promo))
    write_intervals_files(ivout, None)


def render_promo(p):
    L, add = [], None
    add = L.append
    add("# Promotion analysis — association, not causal\n")
    add("_Generated by `scripts/run_model.py` (docs/model_protocol.md section 7). **Rossmann chose when promotions ran** (and timing alternates by week), "
        "so every number below is an adjusted association. It says nothing about profit, margin, cannibalisation of neighbouring days or products, "
        "or what would happen if promotion timing changed._\n")
    r = p["regression"]
    add(f"## 1. Within-store regression of log(Sales) on Promo\n\nSample: {r['sample']}. {r['outcome']}. SEs: {r['se']}.\n")
    add("| Specification | Fixed effects / controls | N | Stores | Promo coefficient [95% CI] | Implied % difference [95% CI] |\n|---|---|---|---|---|---|")
    for k, s in r["specs"].items():
        c = s["promo"]
        add(f"| {k} | {s['fixed_effects']} | {s['n']:,} | {s['n_clusters']:,} | {c['coef']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
            f"{c['pct']:+.1f}% [{c['pct_lo']:+.1f}%, {c['pct_hi']:+.1f}%] |")
    d = r["specs"]["D_plus_neighbouring_promo_flags"]
    add("\n**Spec D: coefficients on the promo flags of neighbouring days/weeks** (timing pattern around promo days, not effects)\n")
    add("| Term | coefficient [95% CI] | implied % |\n|---|---|---|")
    for k, v in d.items():
        if isinstance(v, dict) and "coef" in v:
            add(f"| {k} | {v['coef']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['pct']:+.1f}% |")
    add(f"\nRows dropped in spec D for unknown neighbours: {d['rows_dropped_for_unknown_neighbours']:,}.\n")
    add("**Spec B by store type**\n")
    add("| Store type | Stores | Promo coefficient [95% CI] | Implied % [95% CI] |\n|---|---|---|---|")
    for stp, s in r["by_store_type_spec_B"].items():
        c = s["promo"]
        add(f"| {stp} | {s['n_clusters']:,} | {c['coef']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['pct']:+.1f}% [{c['pct_lo']:+.1f}%, {c['pct_hi']:+.1f}%] |")
    m = p["model_based"]
    add(f"\n## 2. Model-based comparison (Promo = 1 vs 0 in the global model)\n\n{m['definition']}.\n")
    for variant in ("neighbours_as_observed", "neighbours_alternating"):
        desc = {"neighbours_as_observed": "neighbouring-week promo flags kept as observed (everything else fixed)",
                "neighbours_alternating": "neighbouring-week flags set to the alternating pattern (previous/next week = 1 - Promo)"}[variant]
        v = m[variant]
        add(f"**Variant: {desc}**\n")
        add("| Group | rows | stores | geometric-mean ratio [95% CI] | mean ratio | row ratio p10 / p50 / p90 | store ratio p10 / p50 / p90 |\n|---|---|---|---|---|---|---|")
        add(f"| all | {v['overall']['rows']:,} | {v['overall']['n_stores']:,} | {fmt_ratio(v['overall'])} | {v['overall']['mean_ratio']:.3f} | {spread(v['overall'], 'row')} | {spread(v['overall'], 'store')} |")
        for k, s in v["by_store_type"].items():
            add(f"| store type {k} | {s['rows']:,} | {s['n_stores']:,} | {fmt_ratio(s)} | {s['mean_ratio']:.3f} | {spread(s, 'row')} | {spread(s, 'store')} |")
        for k, s in v["by_observed_promo"].items():
            add(f"| rows observed as {k} | {s['rows']:,} | {s['n_stores']:,} | {fmt_ratio(s)} | {s['mean_ratio']:.3f} | {spread(s, 'row')} | {spread(s, 'store')} |")
        add("")
    add("## Reading these numbers\n")
    add("- Both parts are **adjusted associations**. The model-based ratio is a property of the fitted model and is only as good as its features; "
        "toggling Promo while holding the other features fixed can create combinations (for example promo in two consecutive weeks) that the data rarely contains - the "
        "'alternating' variant removes the most obvious one.")
    add("- Spec D shows how sales around promo days differ, which cannot be separated from the alternating-week schedule; no statement about pull-forward or cannibalisation is made.")
    add("- Estimates describe this retailer, 2013-2015, Monday-Friday open days.\n")
    return "\n".join(L) + "\n"


def fmt_ratio(s):
    return f"{s['geometric_mean_ratio']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"


def spread(s, kind):
    return f"{s[kind + '_ratio_p10']:.3f} / {s[kind + '_ratio_p50']:.3f} / {s[kind + '_ratio_p90']:.3f}"


def write_intervals_files(bt, val):
    out = dict(bt)
    if val is not None:
        out["validation"] = val
    (RES / "intervals.json").write_text(json.dumps(clean(out), indent=2) + "\n")
    L, add = [], None
    add = L.append
    add("# Prediction intervals — LightGBM quantile models (q10, q90 = 80% interval)\n")
    add(f"_Generated by `scripts/run_model.py`; protocol `docs/model_protocol.md` section 6. Same features and selected configuration as the point model; "
        f"quantile pair sorted if crossed (crossings before the fix: {out['crossing_before_fix']['total']:,} of {out['crossing_before_fix']['rows']:,} rows). "
        "Nominal coverage 0.80. Coverage is not tuned. Relative width = sum(hi - lo) / sum(actual). 95% store-clustered bootstrap CIs. Split-conformal not attempted._\n")
    f_ = lambda c, nd=3: f"{c['value']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"
    b = out["backtest"]
    add("## Backtest (8 folds)\n")
    add("| View | Coverage | Below | Above | Relative width | Pinball q10 | Pinball q90 |\n|---|---|---|---|---|---|---|")
    labels = {"all_folds": "mean over all 8 folds", "folds_2_5": "folds 2-5", "other_folds": "folds 1, 6, 7, 8"}
    for v, vl in labels.items():
        x = b["views"][v]
        add(f"| {vl} | {f_(x['coverage'])} | {f_(x['share_below'])} | {f_(x['share_above'])} | {f_(x['rel_width'])} | {f_(x['pinball_q10'], 1)} | {f_(x['pinball_q90'], 1)} |")
    add("\n**By fold**\n")
    add("| Fold | Coverage | Below | Above | Relative width | crossings |\n|---|---|---|---|---|---|")
    for fo, x in b["per_fold"].items():
        add(f"| {fo} | {f_(x['coverage'])} | {x['share_below']['value']:.3f} | {x['share_above']['value']:.3f} | {f_(x['rel_width'])} | {out['crossing_before_fix']['by_fold'][fo]} |")
    for key, ttl in (("horizon", "horizon bucket"), ("store_type", "store type"), ("promo", "promo vs non-promo")):
        add(f"\n**Coverage and width by {ttl} (mean over all 8 folds)**\n")
        add("| Group | Coverage | Below | Above | Relative width |\n|---|---|---|---|---|")
        for gname, x in b[key].items():
            x = x["all_folds"]
            add(f"| {gname} | {f_(x['coverage'])} | {x['share_below']['value']:.3f} | {x['share_above']['value']:.3f} | {f_(x['rel_width'])} |")
    add("\n**Coverage by horizon bucket and fold** (point estimates)\n")
    hb = b["horizon_bucket_by_fold_coverage"]
    buckets = list(next(iter(hb.values())))
    add("| Fold | " + " | ".join(buckets) + " |\n|---|" + "---|" * len(buckets))
    for fo, x in hb.items():
        add(f"| {fo} | " + " | ".join("n/a" if x[k]["value"] is None else f"{x[k]['value']:.3f}" for k in buckets) + " |")
    if val is not None:
        x = val["views"]["validation"]
        add("\n## VALIDATION (single run)\n")
        add("| Coverage | Below | Above | Relative width | crossings |\n|---|---|---|---|---|")
        add(f"| {f_(x['coverage'])} | {f_(x['share_below'])} | {f_(x['share_above'])} | {f_(x['rel_width'])} | {val['crossing_before_fix']} |")
        add("\n**By horizon bucket**\n")
        add("| Group | Coverage | Relative width |\n|---|---|---|")
        for gname, y in val["horizon"].items():
            add(f"| {gname} | {f_(y['validation']['coverage'])} | {f_(y['validation']['rel_width'])} |")
    (RES / "intervals.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ validation (ONE run)
def stage_validation(mg):
    frame, store = load_inputs()
    date0 = date0_of(frame)
    tab = get_table()
    vs = validation_split()
    base = base_eval(frame, store, [vs], date0)
    cf_idx = int((vs.cutoff - date0).days)
    store_ids = sorted(store["Store"].unique().tolist())
    boot = StoreBootstrap(len(store_ids))
    sel = json.loads((RES / "model_selected.json").read_text())
    assert sel["frozen"]
    cfg, cid = sel["config"], sel["id"]
    runs = {}
    t0 = time.time()
    for kind, obj, alpha in [("full", "regression", None), ("q10", "quantile", 0.1), ("q90", "quantile", 0.9)]:
        runs[kind] = fit_and_store(tab, f"val_{kind}", cid, MF.features_for(), cfg, "validation", cf_idx, base, date0, objective=obj, alpha=alpha,
                                   status="validation", stage="model-validation", params_extra={"model": kind})
        runs[kind]["fold"] = "validation"
        mg.stage(f"validation fit {kind}")
    views = {"validation": ["validation"]}
    frames = {"baseline": base}
    for name, col in [("full", "pred"), ("full_smearing", "pred_smear")]:
        m = attach(base, runs["full"], col)
        frames[name] = m.assign(yhat=m[col].clip(lower=0))
    res, bv = evaluate_all(frames, store_ids, ["validation"], boot, views)
    comp = {m: paired_difference(bv["full"]["validation"][:, 0, METRICS.index(m)], bv["baseline"]["validation"][:, 0, METRICS.index(m)],
                                 res["full"]["overall"]["views"]["validation"][m]["value"], res["baseline"]["overall"]["views"]["validation"][m]["value"])
            for m in METRICS}
    comp_s = {m: paired_difference(bv["full_smearing"]["validation"][:, 0, METRICS.index(m)], bv["baseline"]["validation"][:, 0, METRICS.index(m)],
                                   res["full_smearing"]["overall"]["views"]["validation"][m]["value"], res["baseline"]["overall"]["views"]["validation"][m]["value"])
              for m in METRICS}
    # intervals
    lo_hi = attach(base, runs["q10"].assign(fold="validation"), "pred").rename(columns={"pred": "lo_raw"})
    lo_hi = attach(lo_hi, runs["q90"].assign(fold="validation"), "pred").rename(columns={"pred": "hi_raw"})
    lo, hi, crossed = gbm.fix_crossing(lo_hi["lo_raw"], lo_hi["hi_raw"])
    iv = lo_hi.assign(lo=lo, hi=hi)
    T, _ = intervals.tensor(iv, store_ids, ["validation"])
    _, pv = intervals.summarize(T, ["validation"], boot, views)
    ival = {"views": {"validation": pv["validation"][0]}, "crossing_before_fix": crossed, "rows": int(len(iv))}
    from demand_forecast.evaluation import HORIZON_ORDER
    Th, gh = intervals.tensor(iv, store_ids, ["validation"], "horizon_bucket", HORIZON_ORDER)
    _, pvh = intervals.summarize(Th, ["validation"], boot, views)
    ival["horizon"] = {str(gname): {"validation": pvh["validation"][gi]} for gi, gname in enumerate(gh)}
    out = {"protocol": "docs/model_protocol.md section 10", "run_id": RUN_ID, "selected_config": cfg, "config_id": cid,
           "single_run": True, "variants": res, "comparison": {"full_vs_baseline": comp, "smearing_vs_baseline": comp_s},
           "intervals": ival, "smearing_factor": json.loads(run_path("val_full", cid, "validation").with_suffix(".json").read_text())["smearing_factor"],
           "trees": {k: json.loads(run_path(f"val_{k}", cid, "validation").with_suffix(".json").read_text())["best_iteration"] for k in ("full", "q10", "q90")},
           "runtime_s": time.time() - t0, "peak_rss_mb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
           "peak_rss_mb_children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024}
    (RES / "model_validation.json").write_text(json.dumps(clean(out), indent=2) + "\n")
    names = ["baseline", "full", "full_smearing"]
    vl = {"validation": "VALIDATION window 2015-05-09..2015-06-19, forecast from cutoff 2015-05-08"}
    pd_lines = ["## Paired differences (variant minus baseline)\n", "| Variant | RMSPE | WAPE | MASE | Bias |\n|---|---|---|---|---|"]
    for lbl, d in (("Full model", comp), ("Full + smearing", comp_s)):
        pd_lines.append(f"| {lbl} | " + " | ".join(f"{d[m]['diff']:+.4f} [{d[m]['lo']:+.4f}, {d[m]['hi']:+.4f}]" for m in METRICS) + " |")
    pd_lines.append(f"\nSmearing factor (holdout): {out['smearing_factor']:.3f}; trees: {out['trees']}; runtime {out['runtime_s']:.0f}s; peak RSS {out['peak_rss_mb_self']} MB (children {out['peak_rss_mb_children']} MB).\n")
    md = render_comparison("Global LightGBM — VALIDATION confirmation (single run, frozen configuration)", res, bv, names, vl,
                           extra_sections=["\n".join(pd_lines)],
                           note_lines=[f"_Generated by `scripts/run_model.py --stage validation` (run `{RUN_ID}`). Configuration `{cid}` frozen from the backtest folds; nothing tuned on VALIDATION._"])
    (RES / "model_validation.md").write_text(md)
    bt = json.loads((RES / "intervals_backtest_raw.json").read_text())
    write_intervals_files(bt, ival)
    mg.stage("validation done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["features", "grid", "grid-probe", "ablations", "quantiles", "analysis", "report", "validation"])
    st = ap.parse_args().stage
    RES.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    mg = MemGuard(RES / f"model_{st}_mem.log")
    mg.start_watchdog()
    t0 = time.time()
    {"features": stage_features, "grid": stage_grid, "grid-probe": lambda m: stage_grid(m, only_first=True), "ablations": stage_ablations,
     "quantiles": stage_quantiles, "analysis": stage_analysis, "report": stage_report, "validation": stage_validation}[st](mg)
    mg.stage(f"{st}: finished in {time.time() - t0:.0f}s; ru_maxrss self {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")


if __name__ == "__main__":
    main()
