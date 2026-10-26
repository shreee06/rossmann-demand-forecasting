"""Step 4 (docs/final_protocol.md).

  calibrate  conformal factors from backtest folds only + ONE confirmation on VALIDATION -> intervals_calibrated.*
  train      frozen final models (point, q10, q90) on data through 2015-06-19
  test       ONE-SHOT TEST evaluation (refuses a second run unless --override "<reason>")
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_model as RM  # noqa: E402
from demand_forecast import calibration as CAL  # noqa: E402
from demand_forecast import config, gbm, intervals  # noqa: E402
from demand_forecast import final as FN  # noqa: E402
from demand_forecast import model_features as MF  # noqa: E402
from demand_forecast.backtest import backtest_splits, load_modelling_frame, run_splits, validation_split  # noqa: E402
from demand_forecast.baselines import PromoWeekdayMean, SeasonalNaive  # noqa: E402
from demand_forecast.bootstrap import N_BOOT, SEED, StoreBootstrap  # noqa: E402
from demand_forecast.data import load_processed  # noqa: E402
from demand_forecast.evaluation import HORIZON_ORDER  # noqa: E402
from demand_forecast.memguard import MemGuard  # noqa: E402
from demand_forecast.metrics import METRICS  # noqa: E402

RES = config.RESULTS_DIR
FINAL_DIR = config.PROCESSED_DIR / "final"
FACTORS = RES / "calibration_factors.json"
RECORD = RES / "test_run_record.json"
RM.LABEL.update({"snaive14": "Seasonal naive 14 (reference)"})


def interval_summary(iv, store_ids, fold_ids, boot, views):
    T, _ = intervals.tensor(iv, store_ids, fold_ids)
    _, pv = intervals.summarize(T, fold_ids, boot, views)
    v = list(views)[0]
    out = {"overall": pv[v][0]}
    Th, gh = intervals.tensor(iv, store_ids, fold_ids, "horizon_bucket", HORIZON_ORDER)
    _, pvh = intervals.summarize(Th, fold_ids, boot, views)
    out["by_horizon"] = {str(g): pvh[v][i] for i, g in enumerate(gh)}
    return out


def with_intervals(base, q10, q90, factors=None):
    m = RM.attach(base, q10, "pred").rename(columns={"pred": "lo_raw"})
    m = RM.attach(m, q90, "pred").rename(columns={"pred": "hi_raw"})
    lo, hi, crossed = gbm.fix_crossing(m["lo_raw"], m["hi_raw"])
    if factors is not None:
        lo, hi = CAL.apply_factors(lo, hi, m["horizon_bucket"], factors)
    return m.assign(lo=lo, hi=hi), crossed


# ------------------------------------------------------------------ calibrate
def stage_calibrate(mg):
    frame, store = RM.load_inputs()
    date0 = frame["Date"].min()
    store_ids = sorted(store["Store"].unique().tolist())
    boot = StoreBootstrap(len(store_ids))
    _, cid = RM.selected_cfg()
    base_bt = RM.base_eval(frame, store, backtest_splits(), date0)
    cal, _ = with_intervals(base_bt, RM.load_runs("q10", cid), RM.load_runs("q90", cid))
    factors = CAL.fit_factors(cal[["y", "lo", "hi", "horizon_bucket", "Date"]])
    FACTORS.write_text(json.dumps({"method": "split-conformal CQR, multiplicative score max(lo/y, y/hi), per horizon bucket",
                                   "level": CAL.LEVEL, "calibration_rows": int(len(cal)),
                                   "calibration_dates": [str(cal["Date"].min().date()), str(cal["Date"].max().date())],
                                   "factors": factors}, indent=2) + "\n")
    mg.stage(f"factors {factors}")
    views_bt = {"all_folds": list(range(1, 9))}
    bt_unc = interval_summary(cal, store_ids, list(range(1, 9)), boot, views_bt)
    calc, _ = with_intervals(base_bt, RM.load_runs("q10", cid), RM.load_runs("q90", cid), factors)
    bt_cal = interval_summary(calc, store_ids, list(range(1, 9)), boot, views_bt)
    # ONE confirmation on VALIDATION with the Step-3 validation quantile predictions (no refit, no re-tuning)
    vs = validation_split()
    base_v = RM.base_eval(frame, store, [vs], date0)
    q10v = pd.read_parquet(RM.run_path("val_q10", cid, "validation")).assign(fold="validation")
    q90v = pd.read_parquet(RM.run_path("val_q90", cid, "validation")).assign(fold="validation")
    vu, _ = with_intervals(base_v, q10v, q90v)
    vc, _ = with_intervals(base_v, q10v, q90v, factors)
    views_v = {"validation": ["validation"]}
    v_unc, v_cal = interval_summary(vu, store_ids, ["validation"], boot, views_v), interval_summary(vc, store_ids, ["validation"], boot, views_v)
    cov = v_cal["overall"]["coverage"]["value"]
    out = {"protocol": "docs/final_protocol.md section 2", "factors": factors,
           "backtest_uncalibrated": bt_unc, "backtest_calibrated_in_sample": bt_cal,
           "validation_uncalibrated": v_unc, "validation_calibrated": v_cal,
           "validation_check": {"coverage": cov, "band": [0.77, 0.83], "inside_band": bool(0.77 <= cov <= 0.83),
                                "action": "none (reported as is; factors not re-tuned)"}}
    (RES / "intervals_calibrated.json").write_text(json.dumps(RM.clean(out), indent=2) + "\n")
    render_intervals(out)
    RM.log_rows([{"stage": "final-calibration", "baseline": "conformal_cqr", "params": json.dumps(factors), "split": "backtest(fit)+validation(check)",
                  "n_rows": len(cal), "status": "calibration", "note": f"validation calibrated coverage={cov:.4f}"}])
    print(json.dumps(out["validation_check"]), factors)


def render_intervals(out):
    f_ = lambda c: f"{c['value']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"
    L = ["# Calibrated prediction intervals (80% nominal)\n",
         "_Generated by `scripts/final_test.py`; protocol `docs/final_protocol.md` section 2. Split-conformal (CQR) factors per horizon bucket, "
         "fitted on the 8 backtest folds only (in-sample there by construction), checked once on VALIDATION, applied unchanged to TEST. "
         "Relative width = sum(hi - lo) / sum(actual). 95% store-clustered bootstrap CIs._\n",
         "## Factors (q > 1 widens)\n", "| Horizon | factor |\n|---|---|"]
    for b, q in out["factors"].items():
        L.append(f"| {b} | {q:.4f} |")
    rows = [("Backtest (8 folds), uncalibrated", "backtest_uncalibrated"), ("Backtest, calibrated (in-sample)", "backtest_calibrated_in_sample"),
            ("VALIDATION, uncalibrated", "validation_uncalibrated"), ("VALIDATION, calibrated (confirmation)", "validation_calibrated"),
            ("TEST, uncalibrated", "test_uncalibrated"), ("TEST, calibrated", "test_calibrated")]
    L += ["\n## Coverage and width\n", "| Set | Coverage | Below | Above | Relative width |\n|---|---|---|---|---|"]
    for lbl, k in rows:
        if k in out:
            o = out[k]["overall"]
            L.append(f"| {lbl} | {f_(o['coverage'])} | {o['share_below']['value']:.3f} | {o['share_above']['value']:.3f} | {f_(o['rel_width'])} |")
    vc = out["validation_check"]
    L.append(f"\nVALIDATION check: calibrated coverage {vc['coverage']:.3f}; pre-registered band 0.77-0.83; inside band: {vc['inside_band']}. {vc['action']}.\n")
    L += ["## Coverage by horizon bucket\n", "| Set | " + " | ".join(HORIZON_ORDER) + " |\n|---|" + "---|" * len(HORIZON_ORDER)]
    for lbl, k in rows:
        if k in out:
            L.append(f"| {lbl} | " + " | ".join(f"{out[k]['by_horizon'][b]['coverage']['value']:.3f}" for b in HORIZON_ORDER) + " |")
    (RES / "intervals_calibrated.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ train
def stage_train(mg):
    frame = load_modelling_frame()  # ends 2015-06-19
    assert frame["Date"].max() == FN.final_cutoff()
    _, store = load_processed()
    sel = json.loads((RES / "model_selected.json").read_text())
    t0 = time.time()
    tab = FN.build_final_training_table(frame, store)
    mg.stage(f"final training table {tab.shape}")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    info = {"rows": int(len(tab)), "last_target_date": str((frame["Date"].min() + pd.Timedelta(days=int(tab["t_idx"].max()))).date()),
            "cutoffs": int(tab["cutoff_idx"].nunique()), "trees": FN.FROZEN_TREES, "config": sel["config"]}
    for kind in ("full", "q10", "q90"):
        t1 = time.time()
        b = FN.fit_final(tab, sel["config"], kind)
        b.save_model(str(FINAL_DIR / f"final_{kind}.txt"))
        RM.log_rows([{"stage": "final-train", "baseline": f"lgbm_final_{kind}", "params": json.dumps({**sel["config"], "trees": FN.FROZEN_TREES[kind]}),
                      "split": "train<=2015-06-19", "n_rows": len(tab), "runtime_s": f"{time.time() - t1:.1f}", "status": "final",
                      "note": "frozen config and tree count; no early stopping"}])
        mg.stage(f"final {kind} trained ({b.num_trees()} trees)")
    info["seconds"] = time.time() - t0
    (RES / "final_training_info.json").write_text(json.dumps(info, indent=2) + "\n")


# ------------------------------------------------------------------ test (one shot)
def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def stage_test(mg, override=None):
    if RECORD.exists() and not override:
        FN.check_and_write_record(RECORD, {}, None)  # raises with the refusal message, before anything is read
    import lightgbm as lgb
    t0 = time.time()
    sp = FN.test_split()
    full = FN.load_frame_including_test()
    full = full[full["Date"] <= sp.horizon_end]
    _, store = load_processed()
    store_ids = sorted(store["Store"].unique().tolist())
    date0 = full["Date"].min()
    # baselines + evaluated rows (actual Sales are only used for scoring)
    base = run_splits(PromoWeekdayMean(RM.BASELINE_K), full, store, [sp], allow_test=True)[0]
    base["t_idx"] = (base["Date"] - date0).dt.days.astype(np.int16)
    sn = run_splits(SeasonalNaive(14), full, store, [sp], allow_test=True)[0]
    sn["t_idx"] = (sn["Date"] - date0).dt.days.astype(np.int16)
    sel_path = RES / "model_selected.json"
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "HEAD"),
              "working_tree_clean": git("status", "--porcelain") == "", "config_sha256": FN.sha256_file(sel_path),
              "calibration_factors_sha256": FN.sha256_file(FACTORS), "window": [str(sp.horizon_start.date()), str(sp.horizon_end.date())],
              "cutoff": str(sp.cutoff.date()), "n_evaluated_rows": int(len(base)), "seeds": {"bootstrap": SEED, "subsample": 20260101},
              "frozen_trees": FN.FROZEN_TREES}
    FN.check_and_write_record(RECORD, record, override)
    mg.stage(f"test record written ({len(base)} rows)")
    # model features: TEST Sales blanked before any feature is built
    feat_frame = FN.blank_sales_after(full, sp.cutoff)
    assert (feat_frame.loc[feat_frame["Date"] > sp.cutoff, "Sales"] == 0).all()
    g = MF.Grids(feat_frame, store)
    ci = MF.cutoff_index(g, sp.cutoff)
    rows = pd.DataFrame(MF.build_cutoff_rows(g, ci))
    feats = MF.features_for()
    X = rows[feats].to_numpy(np.float32)
    preds = {}
    for kind in ("full", "q10", "q90"):
        b = lgb.Booster(model_file=str(FINAL_DIR / f"final_{kind}.txt"))
        assert b.num_trees() == FN.FROZEN_TREES[kind]
        preds[kind] = pd.DataFrame({"fold": "test", "Store": rows["Store"].to_numpy(), "t_idx": rows["t_idx"].to_numpy(),
                                    "pred": gbm.backtransform(b.predict(X))})
    mg.stage("test predictions made")
    m = RM.attach(base, preds["full"], "pred")
    snm = base[["fold", "Store", "t_idx"]].merge(sn[["fold", "Store", "t_idx", "yhat"]], on=["fold", "Store", "t_idx"], how="left", validate="one_to_one")
    assert snm["yhat"].notna().all()
    frames = {"full": m.assign(yhat=m["pred"].clip(lower=0)), "baseline": base, "snaive14": base.assign(yhat=snm["yhat"].to_numpy())}
    boot = StoreBootstrap(len(store_ids))
    views = {"test": ["test"]}
    res, bv = RM.evaluate_all(frames, store_ids, ["test"], boot, views)
    comp = {"full_minus_baseline": RM.paired(bv, res, "full", "baseline", "test"),
            "full_minus_snaive14": RM.paired(bv, res, "full", "snaive14", "test"),
            "baseline_minus_snaive14": RM.paired(bv, res, "baseline", "snaive14", "test")}
    factors = json.loads(FACTORS.read_text())["factors"]
    iu, crossed = with_intervals(base, preds["q10"], preds["q90"])
    ic, _ = with_intervals(base, preds["q10"], preds["q90"], factors)
    t_unc, t_cal = interval_summary(iu, store_ids, ["test"], boot, views), interval_summary(ic, store_ids, ["test"], boot, views)
    out = {"protocol": "docs/final_protocol.md", "record": record, "variants": res, "comparison": comp,
           "intervals": {"uncalibrated": t_unc, "calibrated": t_cal, "crossings_before_fix": crossed},
           "bootstrap": {"n_boot": N_BOOT, "seed": SEED}, "runtime_s": time.time() - t0,
           "peak_rss_mb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
           "peak_rss_mb_children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024,
           "scope_note": "One 42-day summer window of one retailer from one cutoff; not evidence for other seasons, years or retailers."}
    (RES / "final_test.json").write_text(json.dumps(RM.clean(out), indent=2) + "\n")
    ivc = json.loads((RES / "intervals_calibrated.json").read_text())
    ivc["test_uncalibrated"], ivc["test_calibrated"] = t_unc, t_cal
    (RES / "intervals_calibrated.json").write_text(json.dumps(RM.clean(ivc), indent=2) + "\n")
    render_intervals(json.loads((RES / "intervals_calibrated.json").read_text()))
    # store-level predictions stay local (data/ is untracked) for the app viewer
    frames["full"].assign(lo=ic["lo"].to_numpy(), hi=ic["hi"].to_numpy(), baseline=base["yhat"].to_numpy(),
                          snaive14=snm["yhat"].to_numpy())[["Store", "Date", "y", "yhat", "lo", "hi", "baseline", "snaive14", "Promo", "horizon"]] \
        .to_parquet(config.PROCESSED_DIR / "final_test_preds.parquet", index=False)
    pd_lines = ["## Paired differences (95% paired store bootstrap; negative RMSPE/WAPE/MASE = first system better)\n",
                "| Comparison | RMSPE | WAPE | MASE | Bias |\n|---|---|---|---|---|"]
    for k, d in comp.items():
        pd_lines.append(f"| {k.replace('_', ' ')} | " + " | ".join(f"{d[mm]['diff']:+.4f} [{d[mm]['lo']:+.4f}, {d[mm]['hi']:+.4f}]" for mm in METRICS) + " |")
    f_ = lambda c: f"{c['value']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"
    pd_lines += ["\n## Intervals on TEST (80% nominal)\n", "| | Coverage | Below | Above | Relative width |\n|---|---|---|---|---|"]
    for lbl, x in (("uncalibrated", t_unc), ("calibrated (backtest factors)", t_cal)):
        o = x["overall"]
        pd_lines.append(f"| {lbl} | {f_(o['coverage'])} | {o['share_below']['value']:.3f} | {o['share_above']['value']:.3f} | {f_(o['rel_width'])} |")
    pd_lines.append(f"\n**Scope.** {out['scope_note']} CIs reflect store sampling only.\n")
    pd_lines.append(f"Run record: commit `{record['git_commit']}` (clean tree: {record['working_tree_clean']}), {record['n_evaluated_rows']:,} evaluated rows, "
                    f"runtime {out['runtime_s']:.0f}s, peak RSS {out['peak_rss_mb_self']} MB (children {out['peak_rss_mb_children']} MB).")
    md = RM.render_comparison("TEST evaluation (single run, 2015-06-20..2015-07-31)", res, bv, ["full", "baseline", "snaive14"],
                              {"test": "TEST window 2015-06-20..2015-07-31, forecast from cutoff 2015-06-19"},
                              extra_sections=["\n".join(pd_lines)],
                              note_lines=["_Generated by `scripts/final_test.py --stage test`; protocol `docs/final_protocol.md`. Frozen model, evaluated once._"])
    (RES / "final_test.md").write_text(md)
    RM.log_rows([{"stage": "final-test", "baseline": n, "params": "frozen", "split": "test", "n_rows": len(base),
                  **{k: RM.fmt(res[n]["overall"]["views"]["test"][k]["value"]) for k in METRICS}, "status": "test",
                  "note": "single TEST run"} for n in ("full", "baseline", "snaive14")])
    mg.stage("test done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["calibrate", "train", "test"])
    ap.add_argument("--override", default=None, help="documented reason for re-running TEST")
    a = ap.parse_args()
    mg = MemGuard(RES / f"final_{a.stage}_mem.log")
    mg.start_watchdog()
    t0 = time.time()
    {"calibrate": stage_calibrate, "train": stage_train, "test": lambda m: stage_test(m, a.override)}[a.stage](mg)
    mg.stage(f"{a.stage}: finished in {time.time() - t0:.0f}s; maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")


if __name__ == "__main__":
    main()
