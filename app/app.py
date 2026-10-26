"""Streamlit app: aggregate results from evals/results, plus an optional local store viewer.

Run: streamlit run app/app.py
The repository ships no store-level data. The store viewer only works if you have downloaded the Kaggle data and run the
pipeline locally (it reads data/processed/final_test_preds.parquet, which git ignores).
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals" / "results"
LOCAL_PREDS = ROOT / "data" / "processed" / "final_test_preds.parquet"
COLORS = {"full": "#2a78d6", "baseline": "#eb6834", "snaive14": "#1baf7a"}
NAMES = {"full": "Global LightGBM", "baseline": "Promo-aware weekday mean (K=12)", "snaive14": "Seasonal naive 14"}
METRICS = ["rmspe", "wape", "mase", "bias"]
HB = ["1-7", "8-14", "15-21", "22-28", "29-35", "36-42"]


@st.cache_data
def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def ci(c, nd=3):
    return f"{c['value']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


def metric_table(variants, view, names):
    rows = []
    for n in names:
        o = variants[n]["overall"]["views"][view]
        rows.append({"System": NAMES.get(n, n), **{m.upper(): ci(o[m], 4) for m in METRICS}})
    return pd.DataFrame(rows)


def group_chart(variants, key, view, groups, title, xlabel):
    fig = go.Figure()
    for n in ("full", "baseline", "snaive14"):
        if n not in variants:
            continue
        g = variants[n][key]["views"][view]
        ys = [g[x]["wape"]["value"] for x in groups]
        err_hi = [g[x]["wape"]["hi"] - g[x]["wape"]["value"] for x in groups]
        err_lo = [g[x]["wape"]["value"] - g[x]["wape"]["lo"] for x in groups]
        fig.add_bar(x=groups, y=ys, name=NAMES[n], marker_color=COLORS[n],
                    error_y=dict(type="data", array=err_hi, arrayminus=err_lo, thickness=1),
                    hovertemplate="%{x}: WAPE %{y:.4f}<extra>" + NAMES[n] + "</extra>")
    fig.update_layout(barmode="group", title=title, xaxis_title=xlabel, yaxis_title="WAPE", height=380, legend_orientation="h")
    return fig


st.set_page_config(page_title="Rossmann demand forecasting", layout="wide")
st.title("Rossmann daily sales forecasting — results")
st.caption("All numbers come from evals/results/*.json (real runs). Data: Kaggle competition rossmann-store-sales, not included in this repository.")

test, val, bt = load("final_test.json"), load("model_validation.json"), load("model_backtest.json")
ivc = load("intervals_calibrated.json")
if test is None:
    st.error("evals/results/final_test.json not found.")
    st.stop()

tabs = st.tabs(["Comparison", "By horizon", "Promo / store type", "Intervals", "Store viewer (local data)", "Limitations"])
with tabs[0]:
    st.subheader("TEST window 2015-06-20..2015-07-31 (evaluated once)")
    st.dataframe(metric_table(test["variants"], "test", ["full", "baseline", "snaive14"]), hide_index=True, width="stretch")
    d = test["comparison"]["full_minus_baseline"]["wape"]
    st.markdown(f"Paired WAPE difference, model minus baseline: **{d['diff']:+.4f}** [{d['lo']:+.4f}, {d['hi']:+.4f}] (95% paired store bootstrap).")
    st.subheader("VALIDATION (single confirmation run)")
    st.dataframe(metric_table(val["variants"], "validation", ["full", "baseline"]), hide_index=True, width="stretch")
    st.subheader("Backtest: mean over 8 rolling-origin folds")
    st.dataframe(metric_table(bt["variants"], "all_folds", ["full", "baseline"]), hide_index=True, width="stretch")
    st.caption("95% store-clustered bootstrap CIs (500 resamples) as value [lo, hi]. Bias = sum(forecast - actual) / sum(actual).")
with tabs[1]:
    st.plotly_chart(group_chart(test["variants"], "horizon", "test", HB, "TEST: WAPE by horizon", "days after cutoff"), width="stretch")
    st.plotly_chart(group_chart(bt["variants"], "horizon", "all_folds", HB, "Backtest (mean of 8 folds): WAPE by horizon", "days after cutoff"),
                    width="stretch")
with tabs[2]:
    st.plotly_chart(group_chart(test["variants"], "promo", "test", ["promo", "no promo"], "TEST: promo vs non-promo days", "target day"),
                    width="stretch")
    st.plotly_chart(group_chart(test["variants"], "store_type", "test", ["a", "b", "c", "d"], "TEST: by store type", "store type"),
                    width="stretch")
    st.caption("Differences between promo and non-promo days describe forecast error, not promotion effects.")
with tabs[3]:
    if ivc:
        rows = []
        for lbl, k in [("Backtest, uncalibrated", "backtest_uncalibrated"), ("Backtest, calibrated (in-sample)", "backtest_calibrated_in_sample"),
                       ("VALIDATION, uncalibrated", "validation_uncalibrated"), ("VALIDATION, calibrated", "validation_calibrated"),
                       ("TEST, uncalibrated", "test_uncalibrated"), ("TEST, calibrated", "test_calibrated")]:
            if k in ivc:
                o = ivc[k]["overall"]
                rows.append({"Set": lbl, "Coverage (nominal 0.80)": ci(o["coverage"]), "Relative width": ci(o["rel_width"]),
                             "Below": round(o["share_below"]["value"], 3), "Above": round(o["share_above"]["value"], 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Calibration factors were fitted on backtest folds only and applied unchanged to VALIDATION and TEST.")
with tabs[4]:
    if not LOCAL_PREDS.exists():
        st.info("The store viewer needs the Kaggle data on your machine. Download train.csv and store.csv from the "
                "rossmann-store-sales competition, set ROSSMANN_RAW_DIR, and run the pipeline (see README). "
                "No store-level data ships with this repository.")
    else:
        p = pd.read_parquet(LOCAL_PREDS)
        s = st.selectbox("Store", sorted(p["Store"].unique()))
        q = p[p["Store"] == s].sort_values("Date")
        fig = go.Figure()
        fig.add_scatter(x=q["Date"], y=q["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=q["Date"], y=q["lo"], fill="tonexty", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="80% interval (calibrated)")
        fig.add_scatter(x=q["Date"], y=q["y"], mode="lines+markers", name="Actual", line=dict(color="#0b0b0b", width=2))
        fig.add_scatter(x=q["Date"], y=q["yhat"], mode="lines", name=NAMES["full"], line=dict(color=COLORS["full"], width=2))
        fig.add_scatter(x=q["Date"], y=q["baseline"], mode="lines", name=NAMES["baseline"], line=dict(color=COLORS["baseline"], width=2, dash="dot"))
        fig.update_layout(title=f"Store {s}: TEST window, open days with sales", height=420, legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
        st.caption("Read from your local data/processed folder; nothing here is part of the repository.")
with tabs[5]:
    st.markdown("""
- **One test window.** TEST is a single 42-day summer window (2015-06-20..2015-07-31) of one retailer, forecast from one cutoff.
  It does not show performance in other seasons (e.g. Christmas), years or retailers. CIs reflect store sampling only.
- **Open flag assumed known.** Days with Open = 0 are forecast as 0 and excluded; unplanned closures are not evaluated.
- **20% training-row sample** and a **1,500-tree cap** (some fits hit it) for compute reasons.
- **Back-transform bias:** forecasts are expm1 of a log-scale prediction, which tends to under-forecast the mean.
- **Data gap:** 180 stores have no rows from July to December 2014.
- **Promotion analysis is association, not causation:** Rossmann chose when promotions ran; nothing here is about profit or cannibalisation.
""")
