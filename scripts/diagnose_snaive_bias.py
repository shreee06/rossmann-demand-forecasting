"""Part B (docs/model_protocol.md section 9): do holidays explain the seasonal-naive over-forecast on VALIDATION?
Uses the saved Step-2 forecasts (data/processed/preds_*.parquet) plus the holiday flags of target and source days."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecast import config  # noqa: E402
from demand_forecast.backtest import HISTORY_COLUMNS  # noqa: E402
from demand_forecast.bootstrap import StoreBootstrap, stat_tensor, summarize  # noqa: E402
from demand_forecast.data import load_processed  # noqa: E402
from demand_forecast.windows import make_windows  # noqa: E402

RES = config.RESULTS_DIR


def grp(state, school):
    return np.where(state, "state holiday", np.where(school, "school holiday (no state holiday)", "ordinary"))


def main():
    test_start = make_windows()["test"][0]
    frame = pd.read_parquet(config.PROCESSED_TRAIN, columns=HISTORY_COLUMNS)
    frame = frame[frame["Date"] < test_start]
    _, store = load_processed()
    store_ids = sorted(store["Store"].unique().tolist())
    flags = frame[["Store", "Date", "StateHoliday", "SchoolHoliday", "Promo"]].rename(columns={"Promo": "Promo_flag"})
    flags["is_state"] = flags["StateHoliday"].astype(str) != "0"
    flags["is_school"] = flags["SchoolHoliday"] == 1
    flags = flags[["Store", "Date", "is_state", "is_school", "Promo_flag"]]
    boot = StoreBootstrap(len(store_ids))
    out = {"protocol": "docs/model_protocol.md section 9", "forecasters": {}}

    for name, period in [("seasonal_naive_7", 7), ("seasonal_naive_14", 14)]:
        out["forecasters"][name] = {}
        for stage in ("validation", "backtest"):
            p = pd.read_parquet(config.PROCESSED_DIR / f"preds_{stage}_{name}.parquet")
            p["fold"] = p["fold"].astype(str)
            fold_ids = sorted(p["fold"].unique(), key=lambda x: (len(x), x))
            p["src_date"] = p["Date"] - pd.to_timedelta(period * np.ceil(p["horizon"] / period).astype(int), unit="D")
            t = p.merge(flags, on=["Store", "Date"], how="left")
            s = p[["Store", "src_date"]].rename(columns={"src_date": "Date"}).merge(flags, on=["Store", "Date"], how="left")
            assert len(t) == len(p) == len(s)
            p["target_group"] = grp(t["is_state"].to_numpy(bool), t["is_school"].to_numpy(bool))
            p["source_group"] = grp(s["is_state"].fillna(False).to_numpy(bool), s["is_school"].fillna(False).to_numpy(bool))
            p["both_ordinary"] = np.where((p["target_group"] == "ordinary") & (p["source_group"] == "ordinary"),
                                          "target and source ordinary", "target or source holiday-affected")
            # EXPLORATORY follow-up (added after the holiday groups showed no explanation): promo state of source vs target day
            p["promo_transition"] = ("source promo=" + s["Promo_flag"].astype("Int64").astype(str).to_numpy() +
                                     " -> target promo=" + t["Promo_flag"].astype("Int64").astype(str).to_numpy())
            p["week"] = p["Date"].dt.strftime("%G-W%V")
            views = {"all": fold_ids}
            res = {"n_rows": int(len(p))}
            tot_y = p["y"].sum()
            for col, label in [("target_group", "target day"), ("source_group", "source day"), ("both_ordinary", "target and source"), ("promo_transition", "promo state")]:
                T, groups = stat_tensor(p, store_ids, fold_ids, col)
                _, pv, _ = summarize(T, fold_ids, boot, views)
                n = p.groupby(col).size()
                sume = p.assign(e=p["yhat"] - p["y"]).groupby(col)["e"].sum()
                res[col] = {g: {"n_rows": int(n[g]), "share_of_rows": float(n[g] / len(p)),
                                "bias": pv["all"][i]["bias"], "wape": pv["all"][i]["wape"],
                                "contribution_to_overall_bias": float(sume[g] / tot_y)} for i, g in enumerate(groups)}
            d = p.assign(e=p["yhat"] - p["y"]).groupby("Date").agg(e=("e", "sum"), y=("y", "sum"), n=("y", "size"))
            d["bias"] = d["e"] / d["y"]
            d["share_of_total_error"] = d["e"] / d["e"].sum()
            res["by_date_top_contributors"] = [
                {"date": str(i.date()), "n_rows": int(r.n), "bias": float(r.bias), "share_of_total_error": float(r.share_of_total_error)}
                for i, r in d.reindex(d["e"].abs().sort_values(ascending=False).index).head(8).iterrows()]
            w = p.assign(e=p["yhat"] - p["y"]).groupby("week").agg(e=("e", "sum"), y=("y", "sum"))
            res["by_iso_week_bias"] = {k: float(v) for k, v in (w["e"] / w["y"]).items()} if stage == "validation" else None
            holiday_dates = sorted({str(x.date()) for x in p.loc[p["target_group"] == "state holiday", "Date"].unique()})
            g = p.assign(e=p["yhat"] - p["y"]).groupby(["promo_transition", "DayOfWeek"]).agg(n=("y", "size"), e=("e", "sum"), y=("y", "sum"))
            g = g[g["n"] >= 200]
            res["promo_transition_by_dow"] = [{"group": a, "dow": int(b), "n_rows": int(r.n), "bias": float(r.e / r.y),
                                               "contribution_to_overall_bias": float(r.e / tot_y)} for (a, b), r in g.iterrows()]
            res["state_holiday_target_dates"] = holiday_dates
            res["overall_bias_all_rows"] = float((p["yhat"] - p["y"]).sum() / tot_y)
            out["forecasters"][name][stage] = res
    sat = frame[(frame["Date"].dt.dayofweek == 5) & (frame["Open"] == 1) & (frame["Sales"] > 0) & (frame["Date"] >= "2015-03-28")]
    sat = sat.groupby("Date")["Sales"].agg(["mean", "size"])
    out["saturday_mean_sales_by_date"] = [{"date": str(i.date()), "mean_sales_per_open_store": float(r["mean"]), "n_stores": int(r["size"])}
                                          for i, r in sat.iterrows()]
    (RES / "snaive_bias_diagnostic.json").write_text(json.dumps(out, indent=2) + "\n")

    L = ["# Part B: seasonal-naive bias diagnostic\n",
         "_Generated by `scripts/diagnose_snaive_bias.py` from the saved Step-2 forecasts; pre-registered in `docs/model_protocol.md` section 9. "
         "Bias = sum(forecast - actual) / sum(actual); CIs are store-clustered bootstrap (500, seed 20260101). "
         "'Contribution' = the group's share of the overall bias (contributions add up to the overall bias). "
         "Source day = the lagged day the forecast copies (t - 7*ceil(h/7) or t - 14*ceil(h/14)); rows that fell back to an earlier lag are classified by their first-choice source day._\n"]
    f = lambda c: f"{c['value']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
    for name, stages in out["forecasters"].items():
        for stage, res in stages.items():
            L.append(f"## {name} - {stage} (overall bias {res['overall_bias_all_rows']:+.4f}, {res['n_rows']:,} rows)\n")
            for col in ("target_group", "source_group", "both_ordinary", "promo_transition"):
                L.append(f"**By {col.replace('_', ' ')}**" + (" (EXPLORATORY, not pre-registered)" if col == "promo_transition" else "") + "\n")
                L.append("| Group | rows | share of rows | Bias [CI] | WAPE [CI] | contribution to overall bias |\n|---|---|---|---|---|---|")
                for g, r in res[col].items():
                    L.append(f"| {g} | {r['n_rows']:,} | {r['share_of_rows']:.3f} | {f(r['bias'])} | "
                             f"{r['wape']['value']:.4f} [{r['wape']['lo']:.4f}, {r['wape']['hi']:.4f}] | {r['contribution_to_overall_bias']:+.4f} |")
                L.append("")
            L.append("**Promo transition x day of week** (EXPLORATORY; groups with >= 200 rows)\n")
            L.append("| Group | day of week | rows | Bias | contribution to overall bias |\n|---|---|---|---|---|")
            for r in res["promo_transition_by_dow"]:
                L.append(f"| {r['group']} | {r['dow']} | {r['n_rows']:,} | {r['bias']:+.4f} | {r['contribution_to_overall_bias']:+.4f} |")
            L.append("")
            L.append(f"State-holiday target dates in these rows: {res['state_holiday_target_dates']}\n")
            L.append("Dates with the largest absolute error contribution:\n")
            L.append("| Date | rows | Bias | share of total error |\n|---|---|---|---|")
            for r in res["by_date_top_contributors"]:
                L.append(f"| {r['date']} | {r['n_rows']:,} | {r['bias']:+.4f} | {r['share_of_total_error']:+.3f} |")
            if res["by_iso_week_bias"]:
                L.append("\nBias by ISO week: " + ", ".join(f"{k}: {v:+.3f}" for k, v in res["by_iso_week_bias"].items()))
            L.append("")
    L.append("## Mean sales per open store on Saturdays (for context; cutoff = 2015-05-08, so the source Saturday is 2015-05-02)\n")
    L.append("| Saturday | mean sales | stores open |\n|---|---|---|")
    for r in out["saturday_mean_sales_by_date"]:
        L.append(f"| {r['date']} | {r['mean_sales_per_open_store']:,.0f} | {r['n_stores']:,} |")
    (RES / "snaive_bias_diagnostic.md").write_text("\n".join(L) + "\n")
    print("\n".join(L)[:200])


if __name__ == "__main__":
    main()
