"""Data audit: pure functions over the typed train/store frames plus a streaming CSV reconciliation.

Scope rule (set by scripts/audit_data.py): structure and data-quality counts use all rows; Sales-based
descriptive statistics (distributions, promo contrast, outliers, seasonality) use only rows BEFORE the
TEST window, so nothing about test-window sales informs later modelling choices.
"""
import csv

import numpy as np
import pandas as pd

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


# ---------------------------------------------------------------- reconciliation
def source_stats_from_csv(path) -> dict:
    """Independent streaming pass over the raw CSV (csv module, no pandas, read-only)."""
    n = 0
    sales = customers = open_ = promo = school = 0
    state = {}
    stores, dmin, dmax = set(), None, None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            sales += int(r["Sales"])
            customers += int(r["Customers"])
            open_ += int(r["Open"])
            promo += int(r["Promo"])
            school += int(r["SchoolHoliday"])
            k = r["StateHoliday"].strip().lower()
            k = "0" if k in ("0", "0.0") else k
            state[k] = state.get(k, 0) + 1
            stores.add(int(r["Store"]))
            d = r["Date"]
            dmin = d if dmin is None or d < dmin else dmin
            dmax = d if dmax is None or d > dmax else dmax
    return {"rows": n, "sales_sum": sales, "customers_sum": customers, "open_sum": open_, "promo_sum": promo,
            "school_holiday_sum": school, "state_holiday_counts": dict(sorted(state.items())),
            "n_stores": len(stores), "date_min": dmin, "date_max": dmax}


def frame_stats(df: pd.DataFrame) -> dict:
    return {"rows": int(len(df)), "sales_sum": int(df["Sales"].sum()), "customers_sum": int(df["Customers"].sum()),
            "open_sum": int(df["Open"].sum()), "promo_sum": int(df["Promo"].sum()),
            "school_holiday_sum": int(df["SchoolHoliday"].sum()),
            "state_holiday_counts": {str(k): int(v) for k, v in
                                     df["StateHoliday"].astype(str).value_counts().sort_index().items()},
            "n_stores": int(df["Store"].nunique()),
            "date_min": df["Date"].min().strftime("%Y-%m-%d"), "date_max": df["Date"].max().strftime("%Y-%m-%d")}


def reconcile(source: dict, processed: dict) -> dict:
    checks = {k: {"source": source[k], "processed": processed[k], "match": source[k] == processed[k]}
              for k in source}
    return {"all_match": all(c["match"] for c in checks.values()), "checks": checks}


def count_data_lines(path) -> int:
    """Newline-based record count (minus header); equals the row count when no field contains a newline."""
    n = 0
    last = b"\n"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n - 1 + (0 if last == b"\n" else 1)


# ---------------------------------------------------------------- coverage / structure
def _grids(train: pd.DataFrame):
    stores = np.sort(train["Store"].unique())
    d0, d1 = train["Date"].min(), train["Date"].max()
    n_days = (d1 - d0).days + 1
    si = np.searchsorted(stores, train["Store"].to_numpy())
    di = (train["Date"] - d0).dt.days.to_numpy()
    present = np.zeros((len(stores), n_days), dtype=bool)
    present[si, di] = True
    is_open = np.zeros_like(present)
    is_open[si[train["Open"].to_numpy() == 1], di[train["Open"].to_numpy() == 1]] = True
    return stores, d0, n_days, present, is_open


def coverage(train: pd.DataFrame, store: pd.DataFrame) -> dict:
    stores, d0, n_days, present, _ = _grids(train)
    dups = int(train.duplicated(["Store", "Date"]).sum())
    expected = len(stores) * n_days
    missing_mask = ~present
    miss_per_store = missing_mask.sum(1)
    miss_per_day = missing_mask.sum(0)
    days = pd.date_range(d0, periods=n_days)
    by_month = pd.Series(miss_per_day, index=days).groupby(days.to_period("M")).sum()
    dow_ok = bool((train["DayOfWeek"].to_numpy() == train["Date"].dt.dayofweek.to_numpy() + 1).all())
    return {
        "rows": int(len(train)), "n_stores": int(len(stores)), "date_min": str(d0.date()),
        "date_max": str((d0 + pd.Timedelta(days=n_days - 1)).date()), "n_days": int(n_days),
        "expected_store_days": int(expected), "missing_store_days": int(missing_mask.sum()),
        "missing_share": float(missing_mask.sum() / expected), "duplicate_store_date_rows": dups,
        "stores_with_any_missing_day": int((miss_per_store > 0).sum()),
        "stores_with_no_missing_day": int((miss_per_store == 0).sum()),
        "max_missing_days_in_one_store": int(miss_per_store.max()),
        "dates_with_any_missing_store": int((miss_per_day > 0).sum()),
        "max_stores_missing_on_one_date": int(miss_per_day.max()),
        "dayofweek_consistent_with_date": dow_ok,
        "stores_in_train_not_in_store_csv": int(len(set(stores) - set(store["Store"]))),
        "stores_in_store_csv_not_in_train": int(len(set(store["Store"]) - set(stores))),
        "missing_by_month": {str(k): int(v) for k, v in by_month[by_month > 0].items()},
    }


def closed_days(train: pd.DataFrame) -> dict:
    closed = train[train["Open"] == 0]
    opn = train[train["Open"] == 1]
    zero_open = opn[opn["Sales"] == 0]
    return {
        "open_zero_rows": int(len(closed)), "open_zero_share_of_rows": float(len(closed) / len(train)),
        "open_zero_sales_sum": int(closed["Sales"].sum()),
        "open_zero_rows_with_positive_sales": int((closed["Sales"] > 0).sum()),
        "open_zero_by_dayofweek": {int(k): int(v) for k, v in closed.groupby("DayOfWeek").size().items()},
        "open_zero_on_state_holiday_rows": int((closed["StateHoliday"].astype(str) != "0").sum()),
        "open_zero_on_school_holiday_rows": int((closed["SchoolHoliday"] == 1).sum()),
        "open_rows": int(len(opn)),
        "open_rows_with_zero_sales": int(len(zero_open)),
        "open_rows_with_zero_sales_share_of_open": float(len(zero_open) / len(opn)),
        "open_rows_with_zero_sales_n_stores": int(zero_open["Store"].nunique()),
        "open_rows_with_zero_sales_by_year": {int(k): int(v) for k, v in
                                              zero_open.groupby(zero_open["Date"].dt.year).size().items()},
        "open_rows_with_zero_sales_customers_sum": int(zero_open["Customers"].sum()),
        "closed_rows_with_nonzero_customers": int((closed["Customers"] > 0).sum()),
    }


def long_closures(train: pd.DataFrame, min_run: int = 30) -> dict:
    """Runs of >= min_run consecutive 'inactive' days per store, inactive = no row OR Open == 0."""
    stores, d0, n_days, present, is_open = _grids(train)
    inactive = ~is_open  # absent rows are not open either
    runs = []
    for i, s in enumerate(stores):
        x = np.concatenate([[0], inactive[i].astype(np.int8), [0]])
        edges = np.flatnonzero(np.diff(x))
        for a, b in zip(edges[::2], edges[1::2]):  # [a, b) in day index
            if b - a >= min_run:
                n_missing = int((~present[i, a:b]).sum())
                runs.append({"store": int(s), "start": str((d0 + pd.Timedelta(days=int(a))).date()),
                             "end": str((d0 + pd.Timedelta(days=int(b) - 1)).date()), "length": int(b - a),
                             "missing_days": n_missing, "closed_rows": int(b - a - n_missing),
                             "touches_data_start": bool(a == 0), "touches_data_end": bool(b == n_days)})
    r = pd.DataFrame(runs, columns=["store", "start", "end", "length", "missing_days", "closed_rows",
                                    "touches_data_start", "touches_data_end"])
    out = {"min_run_days": min_run, "n_runs": int(len(r)), "n_stores": int(r["store"].nunique()) if len(r) else 0}
    if len(r):
        clusters = r.groupby(["start", "end"]).size().sort_values(ascending=False)
        out["longest_run_days"] = int(r["length"].max())
        out["runs_that_are_entirely_missing_rows"] = int((r["missing_days"] == r["length"]).sum())
        out["runs_that_are_entirely_closed_rows"] = int((r["missing_days"] == 0).sum())
        out["runs_touching_data_end"] = int(r["touches_data_end"].sum())
        out["runs_touching_data_start"] = int(r["touches_data_start"].sum())
        out["top_clusters_same_start_end"] = [{"start": a, "end": b, "n_stores": int(n)}
                                              for (a, b), n in clusters.head(10).items()]
        out["by_start_year_month"] = {str(k): int(v) for k, v in
                                      r.groupby(pd.to_datetime(r["start"]).dt.to_period("M")).size().items()}
    out["runs"] = r.to_dict(orient="records")
    return out


# ---------------------------------------------------------------- sales descriptives
def describe(s: pd.Series) -> dict:
    q = s.quantile(QUANTILES)
    return {"n": int(len(s)), "mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()),
            **{f"p{int(k * 100)}": float(v) for k, v in q.items()}, "max": float(s.max())}


def _pos_open(df):
    return df[(df["Open"] == 1) & (df["Sales"] > 0)]


def sales_distribution(df: pd.DataFrame, store: pd.DataFrame):
    """Sales by day of week / store type / assortment. Returns (json dict, dict of DataFrames for CSV)."""
    d = df.merge(store[["Store", "StoreType", "Assortment"]], on="Store", how="left")
    po = _pos_open(d)
    tables = {}
    for key in ["DayOfWeek", "StoreType", "Assortment"]:
        g_all = d.groupby(key, observed=True)
        g_po = po.groupby(key, observed=True)["Sales"]
        t = pd.DataFrame({"rows": g_all.size(), "open_share": g_all["Open"].mean(),
                          "mean_sales_all_rows": g_all["Sales"].mean(),
                          "open_rows_positive_sales": g_po.size(), "mean": g_po.mean(), "median": g_po.median(),
                          "std": g_po.std(), "p05": g_po.quantile(.05), "p95": g_po.quantile(.95)})
        t["n_stores"] = g_all["Store"].nunique()
        tables[key] = t.reset_index()
    return {"overall_open_positive": describe(po["Sales"]),
            **{k: v.to_dict(orient="records") for k, v in tables.items()}}, tables


def promo_summary(df: pd.DataFrame) -> dict:
    """NAIVE DESCRIPTIVE contrast only: promo days are not randomly assigned (weekday, store, season mix)."""
    po = _pos_open(df)
    g = po.groupby("Promo")["Sales"]
    m = g.mean()
    by_dow = po.groupby(["DayOfWeek", "Promo"])["Sales"].agg(["size", "mean"]).unstack("Promo")
    # Is Promo a week-on / week-off pattern? Daily Mon-Fri promo share autocorrelation + same-store Monday flips.
    wk_share = df[df["DayOfWeek"] <= 5].groupby("Date")["Promo"].mean().asfreq("D")
    mon = df[df["DayOfWeek"] == 1].pivot(index="Date", columns="Store", values="Promo")
    mon = mon.asfreq("7D")  # consecutive Mondays; a skipped Monday becomes NaN rows and is not counted
    both = mon.notna() & mon.shift(1).notna()
    flips = ((mon != mon.shift(1)) & both).to_numpy().sum()
    return {
        "weekday_promo_share_autocorrelation": {f"lag_{k}": float(wk_share.autocorr(k)) for k in [1, 7, 14, 28]},
        "monday_promo_flip_rate_vs_previous_monday": float(flips / both.to_numpy().sum()),
        "monday_pairs_compared": int(both.to_numpy().sum()),
        "label": "naive descriptive difference, NOT a promotion effect",
        "promo_share_all_rows": float((df["Promo"] == 1).mean()),
        "promo_share_open_rows": float((df.loc[df["Open"] == 1, "Promo"] == 1).mean()),
        "promo_share_by_dayofweek": {int(k): float(v) for k, v in df.groupby("DayOfWeek")["Promo"].mean().items()},
        "open_positive_rows": {"promo": int(g.size().get(1, 0)), "no_promo": int(g.size().get(0, 0))},
        "mean_sales_promo": float(m.get(1, np.nan)), "mean_sales_no_promo": float(m.get(0, np.nan)),
        "median_sales_promo": float(g.median().get(1, np.nan)), "median_sales_no_promo": float(g.median().get(0, np.nan)),
        "ratio_of_means": float(m.get(1, np.nan) / m.get(0, np.nan)),
        "by_dayofweek": [
            {"DayOfWeek": int(dw),
             "n_promo": int(r["size"].get(1, 0) if not pd.isna(r["size"].get(1, 0)) else 0),
             "n_no_promo": int(r["size"].get(0, 0) if not pd.isna(r["size"].get(0, 0)) else 0),
             "mean_promo": None if pd.isna(r["mean"].get(1, np.nan)) else float(r["mean"][1]),
             "mean_no_promo": None if pd.isna(r["mean"].get(0, np.nan)) else float(r["mean"][0])}
            for dw, r in by_dow.iterrows()],
    }


def holiday_summary(df: pd.DataFrame, all_rows_for_coverage: pd.DataFrame) -> dict:
    """Coverage uses all rows; the sales contrast (naive descriptive) uses df (pre-test)."""
    c = all_rows_for_coverage
    state = c["StateHoliday"].astype(str)
    po = _pos_open(df)
    po_state = po["StateHoliday"].astype(str)
    out = {
        "school_holiday_share_all_rows": float((c["SchoolHoliday"] == 1).mean()),
        "school_holiday_share_by_year": {int(k): float(v) for k, v in
                                         c.groupby(c["Date"].dt.year)["SchoolHoliday"].mean().items()},
        "dates_with_any_school_holiday_store": int(c.loc[c["SchoolHoliday"] == 1, "Date"].nunique()),
        "state_holiday_rows": {k: int(v) for k, v in state.value_counts().sort_index().items()},
        "state_holiday_distinct_dates": {k: int(c.loc[state == k, "Date"].nunique()) for k in ["a", "b", "c"]},
        "state_holiday_open_share": {k: float(c.loc[state == k, "Open"].mean()) for k in ["0", "a", "b", "c"]
                                     if (state == k).any()},
        "state_holiday_dates": {k: sorted(str(x.date()) for x in c.loc[state == k, "Date"].unique())
                                for k in ["a", "b", "c"]},
        "naive_mean_sales_open_positive_by_state_holiday": {k: float(po.loc[po_state == k, "Sales"].mean())
                                                           for k in ["0", "a", "b", "c"] if (po_state == k).any()},
        "naive_mean_sales_open_positive_school_holiday": {
            str(k): float(v) for k, v in po.groupby("SchoolHoliday")["Sales"].mean().items()},
        "contrast_label": "naive descriptive, not an effect",
    }
    return out


def store_attributes(store: pd.DataFrame) -> dict:
    return {
        "n_stores": int(len(store)),
        "competition_distance_missing": int(store["CompetitionDistance"].isna().sum()),
        "competition_open_since_month_missing": int(store["CompetitionOpenSinceMonth"].isna().sum()),
        "competition_open_since_year_missing": int(store["CompetitionOpenSinceYear"].isna().sum()),
        "promo2_stores": int((store["Promo2"] == 1).sum()),
        "promo2_stores_missing_since_week_or_year": int(((store["Promo2"] == 1) & (
            store["Promo2SinceWeek"].isna() | store["Promo2SinceYear"].isna())).sum()),
        "non_promo2_stores_with_promo_interval": int(((store["Promo2"] == 0) & store["PromoInterval"].notna()).sum()),
        "promo_interval_counts": {str(k): int(v) for k, v in store["PromoInterval"].value_counts(dropna=False).items()},
        "store_type_counts": {str(k): int(v) for k, v in store["StoreType"].value_counts().sort_index().items()},
        "assortment_counts": {str(k): int(v) for k, v in store["Assortment"].value_counts().sort_index().items()},
        "competition_distance": describe(store["CompetitionDistance"].dropna()),
    }


def outliers(df: pd.DataFrame, z_thresh: float = 5.0) -> dict:
    po = _pos_open(df).copy()
    s = po["Sales"]
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    hi3, hi15 = q3 + 3 * iqr, q3 + 1.5 * iqr
    po["lg"] = np.log(po["Sales"])
    med = po.groupby("Store")["lg"].transform("median")
    mad = (po["lg"] - med).abs().groupby(po["Store"]).transform("median") * 1.4826
    z = (po["lg"] - med) / mad.where(mad > 0)
    imax = s.idxmax()
    return {
        "scope": "open days with Sales > 0",
        "global_q1": float(q1), "global_q3": float(q3), "upper_fence_1.5iqr": float(hi15), "upper_fence_3iqr": float(hi3),
        "n_above_1.5iqr_fence": int((s > hi15).sum()), "n_above_3iqr_fence": int((s > hi3).sum()),
        "max_sales": int(s.max()), "max_sales_store": int(po.loc[imax, "Store"]),
        "max_sales_date": str(po.loc[imax, "Date"].date()),
        "p99": float(s.quantile(.99)), "p999": float(s.quantile(.999)),
        "per_store_robust_z_threshold": z_thresh,
        "n_robust_z_high": int((z > z_thresh).sum()), "n_robust_z_low": int((z < -z_thresh).sum()),
        "stores_with_robust_z_high": int(po.loc[z > z_thresh, "Store"].nunique()),
        "stores_with_robust_z_low": int(po.loc[z < -z_thresh, "Store"].nunique()),
        "robust_z_definition": "(log(sales) - store median of log(sales)) / (1.4826 * store MAD of log(sales))",
    }


def seasonality(df: pd.DataFrame):
    """Store-normalised sales index (sales / store mean over open positive days) by period, plus autocorrelations."""
    po = _pos_open(df).copy()
    po["idx"] = po["Sales"] / po.groupby("Store")["Sales"].transform("mean")
    po["year"], po["month"] = po["Date"].dt.year, po["Date"].dt.month
    po["week"] = po["Date"].dt.isocalendar().week.astype(int)
    tables = {}
    dow = po.groupby("DayOfWeek").agg(mean_sales=("Sales", "mean"), index=("idx", "mean"), n=("Sales", "size"))
    tables["dow"] = dow.reset_index()
    mon = po.groupby("month").agg(mean_sales=("Sales", "mean"), index=("idx", "mean"), n=("Sales", "size"),
                                  n_years=("year", "nunique"))
    tables["month"] = mon.reset_index()
    tables["year_month_index"] = po.pivot_table(index="month", columns="year", values="idx", aggfunc="mean").reset_index()
    wk = po.groupby("week").agg(index=("idx", "mean"), n=("Sales", "size"), n_years=("year", "nunique"))
    tables["week_of_year"] = wk.reset_index()
    # Daily mean index, only on days where >= half of all stores are open: on Sundays/holidays a handful of
    # stores gives a noisy mean. Sparse days stay NaN (autocorr drops NaN pairs), nothing is interpolated.
    n_stores = po["Store"].nunique()
    g = po.groupby("Date")
    daily = g["idx"].mean().where(g.size() >= 0.5 * n_stores).asfreq("D")
    acf = {f"lag_{k}": float(daily.autocorr(k)) for k in [1, 7, 14, 28, 364] if len(daily) > k + 10}
    out = {
        "index_definition": "sales / store mean over its open positive-sales days in scope",
        "dow_index": {int(r.DayOfWeek): float(r["index"]) for _, r in tables["dow"].iterrows()},
        "month_index": {int(r.month): float(r["index"]) for _, r in tables["month"].iterrows()},
        "month_n_years": {int(r.month): int(r.n_years) for _, r in tables["month"].iterrows()},
        "week_of_year_index_min": {"week": int(wk["index"].idxmin()), "value": float(wk["index"].min())},
        "week_of_year_index_max": {"week": int(wk["index"].idxmax()), "value": float(wk["index"].max())},
        "daily_mean_index_autocorrelation": acf,
        "daily_series_days": int(len(daily)), "daily_series_days_excluded_sparse_open": int(daily.isna().sum()),
        "daily_series_rule": "days with fewer than half of the stores open are excluded (NaN), not interpolated",
    }
    return out, tables


# ---------------------------------------------------------------- markdown
def md_table(df: pd.DataFrame, floatfmt="{:,.2f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])
