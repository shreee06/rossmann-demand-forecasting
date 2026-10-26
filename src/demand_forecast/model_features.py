"""Feature table for the global direct multi-horizon model (docs/model_protocol.md sections 1-2).

One row per (store, cutoff c, target day t = c + h), h = 1..42, for target days on which the store is open.
History features use observations (open days with Sales > 0) dated <= c only. Calendar, holiday and Promo information
is used only up to c + 42, the end of the forecast horizon (known in advance). Nothing else is read.
"""
import warnings

import numpy as np
import pandas as pd

HORIZON = 42
CAP_HOLIDAY = 60
CAP_CLOSURE = 200
MIN_HISTORY_DAYS = 91
HOLDOUT_OFFSETS_DAYS = (63, 56, 49, 42)  # early-stopping holdout cutoffs = fold cutoff minus these (docs/model_protocol.md section 3)

CATEGORICAL = ["store_type", "assortment", "state_code"]
HORIZON_FEATURES = ["horizon"]
CALENDAR_FEATURES = ["dow", "day_of_month", "week_of_year", "month"]
HOLIDAY_FEATURES = ["state_code", "school_flag", "days_to_state", "days_since_state", "days_to_school", "days_since_school"]
PROMO_FEATURES = ["promo", "promo_m1", "promo_p1", "promo_m7", "promo_p7", "promo2_active", "promo2_weeks"]
STORE_FEATURES = ["store_type", "assortment", "comp_distance", "comp_open_months"]
HISTORY_PLAIN = ["mean_28", "mean_56", "mean_91", "median_28", "median_56", "median_91",
                 "swd_mean_4", "swd_mean_8", "swd_mean_12", "lag7", "lag14",
                 "days_since_closure", "frac_missing_91", "frac_closed_91", "n_obs_91"]
HISTORY_PROMO = ["swp_mean_4", "swp_mean_8", "swp_mean_12", "swp_n_12"]  # use the Promo flag of the target day
HISTORY_FEATURES = HISTORY_PLAIN + HISTORY_PROMO
FEATURES = (HORIZON_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES + PROMO_FEATURES + STORE_FEATURES
            + HISTORY_FEATURES)
# stored per row so the Promo flag of the target day can be switched without recomputing history
SWP_AUX = [f"swp{p}_{k}" for p in (1, 0) for k in ("mean_4", "mean_8", "mean_12", "n_12")]

ABLATIONS = {
    "no_history": HISTORY_FEATURES,
    "no_promo": PROMO_FEATURES + HISTORY_PROMO,
    "no_holiday": HOLIDAY_FEATURES,
    "horizon_fixed_1": [],  # handled by replacing the horizon column with 1
}

MONTHS = {"jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5, "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11}


def features_for(ablation=None):
    """Feature list of the full model or of an ablation."""
    if ablation is None:
        return list(FEATURES)
    drop = set(ABLATIONS[ablation])
    return [f for f in FEATURES if f not in drop]


def _last_idx(flag, axis=-1):
    ar = np.arange(flag.shape[axis])
    shape = [1] * flag.ndim
    shape[axis] = -1
    return np.maximum.accumulate(np.where(flag, ar.reshape(shape), -1), axis=axis)


def _next_idx(flag, big, axis=-1):
    ar = np.arange(flag.shape[axis])
    shape = [1] * flag.ndim
    shape[axis] = -1
    x = np.where(flag, ar.reshape(shape), big)
    return np.flip(np.minimum.accumulate(np.flip(x, axis), axis=axis), axis)


class Grids:
    """Store x date arrays built from the modelling frame."""

    def __init__(self, frame: pd.DataFrame, store: pd.DataFrame):
        self.store = store.sort_values("Store").reset_index(drop=True)
        self.store_ids = self.store["Store"].to_numpy()
        S = len(self.store_ids)
        d0, d1 = frame["Date"].min(), frame["Date"].max()
        self.dates = pd.date_range(d0, d1)
        D = len(self.dates)
        self.S, self.D = S, D
        si = np.searchsorted(self.store_ids, frame["Store"].to_numpy())
        di = (frame["Date"] - d0).dt.days.to_numpy()
        sales = frame["Sales"].to_numpy(np.float32)
        opn = frame["Open"].to_numpy() == 1
        self.present = np.zeros((S, D), bool)
        self.present[si, di] = True
        self.open = np.zeros((S, D), bool)
        self.open[si, di] = opn
        self.sales_raw = np.full((S, D), np.nan, np.float32)
        self.sales_raw[si, di] = sales
        self.sales_obs = np.full((S, D), np.nan, np.float32)
        ok = opn & (sales > 0)
        self.sales_obs[si[ok], di[ok]] = sales[ok]
        self.promo = np.full((S, D), np.nan, np.float32)
        self.promo[si, di] = frame["Promo"].to_numpy(np.float32)
        code = frame["StateHoliday"].astype(str).map({"0": 0, "a": 1, "b": 2, "c": 3}).to_numpy(np.float32)
        self.state_code = np.full((S, D), np.nan, np.float32)
        self.state_code[si, di] = code
        self.school = np.zeros((S, D), bool)
        self.school[si, di] = frame["SchoolHoliday"].to_numpy() == 1
        self.dow = ((d0.dayofweek + np.arange(D)) % 7).astype(np.int8)
        # window sums (prefix sums over dates)
        z = np.where(np.isnan(self.sales_obs), 0.0, self.sales_obs).astype(np.float64)
        self.cs = np.concatenate([np.zeros((S, 1)), np.cumsum(z, axis=1)], axis=1)
        self.cc = np.concatenate([np.zeros((S, 1)), np.cumsum(~np.isnan(self.sales_obs), axis=1)], axis=1)
        self.cmiss = np.concatenate([np.zeros((S, 1)), np.cumsum(~self.present, axis=1)], axis=1)
        self.cclosed = np.concatenate([np.zeros((S, 1)), np.cumsum(self.present & ~self.open, axis=1)], axis=1)
        closure_nonsun = (~self.open) & (self.dow[None, :] != 6)  # no row or Open = 0, on a day other than Sunday
        self.last_closure = _last_idx(closure_nonsun, axis=1)
        # holidays
        self.state_date = np.nan_to_num(self.state_code, nan=0.0).max(axis=0) > 0
        self.last_state = _last_idx(self.state_date)
        self.next_state = _next_idx(self.state_date, 10 ** 6)
        self.last_school = _last_idx(self.school, axis=1)
        self.next_school = _next_idx(self.school, 10 ** 6, axis=1)
        # Promo2 and store attributes
        st = self.store
        self.type_code = st["StoreType"].astype(str).map({c: i for i, c in enumerate(sorted(st["StoreType"].astype(str).unique()))}).to_numpy(np.float32)
        self.assort_code = st["Assortment"].astype(str).map({c: i for i, c in enumerate(sorted(st["Assortment"].astype(str).unique()))}).to_numpy(np.float32)
        self.comp_distance = st["CompetitionDistance"].to_numpy(np.float32)
        cy = st["CompetitionOpenSinceYear"].astype("float64").to_numpy()
        cm = st["CompetitionOpenSinceMonth"].astype("float64").to_numpy()
        self.comp_open_ym = cy * 12 + (cm - 1)  # NaN when unknown
        self.promo2 = (st["Promo2"].to_numpy() == 1)
        wk = st["Promo2SinceWeek"].astype("float64").to_numpy()
        yr = st["Promo2SinceYear"].astype("float64").to_numpy()
        start = np.full(S, np.datetime64("NaT"), dtype="datetime64[ns]")
        for i in range(S):
            if self.promo2[i] and np.isfinite(wk[i]) and np.isfinite(yr[i]):
                start[i] = np.datetime64(pd.Timestamp.fromisocalendar(int(yr[i]), int(wk[i]), 1))
        self.promo2_start = start
        self.promo2_months = np.zeros((S, 12), bool)
        for i, s in enumerate(st["PromoInterval"].astype("object").to_numpy()):
            if isinstance(s, str):
                for tok in s.split(","):
                    self.promo2_months[i, MONTHS[tok.strip()[:3].lower()]] = True


def cutoff_index(grids: Grids, cutoff) -> int:
    return int((pd.Timestamp(cutoff) - grids.dates[0]).days)


def friday_cutoffs(first="2013-04-05", last="2015-05-08"):
    return pd.date_range(first, last, freq="7D")


def _same_weekday_features(g: Grids, ci: int):
    """For each target weekday w: means over the last K occurrences (K = 4, 8, 12) of w on or before ci."""
    out = {}
    dow_c = int(g.dow[ci])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for w in range(7):
            d0 = ci - ((dow_c - w) % 7)
            idx = d0 - 7 * np.arange(12)
            valid = idx >= 0
            idxc = np.where(valid, idx, 0)
            sv = np.where(valid[None, :], g.sales_obs[:, idxc], np.nan)
            pv = np.where(valid[None, :], g.promo[:, idxc], np.nan)
            r = {}
            for K in (4, 8, 12):
                r[f"swd_mean_{K}"] = np.nanmean(sv[:, :K], axis=1)
                for p in (1, 0):
                    m = np.where(pv[:, :K] == p, sv[:, :K], np.nan)
                    r[f"swp{p}_mean_{K}"] = np.nanmean(m, axis=1)
            for p in (1, 0):
                r[f"swp{p}_n_12"] = np.sum(~np.isnan(np.where(pv == p, sv, np.nan)), axis=1).astype(np.float32)
            out[w] = r
    return out


def build_cutoff_rows(g: Grids, ci: int) -> dict:
    """Feature rows for one cutoff (column dict of arrays). Uses sales only up to ci; everything dated after ci + 42
    (and Sales after ci) is never read."""
    S, D = g.S, g.D
    if ci - MIN_HISTORY_DAYS + 1 < 0:
        raise ValueError("not enough history before the cutoff")
    end = min(ci + HORIZON, D - 1)  # last date whose calendar/promo information may be used
    hs = np.arange(1, HORIZON + 1)
    tcols = ci + hs
    tvalid = tcols <= D - 1
    mask = np.zeros((S, HORIZON), bool)
    mask[:, tvalid] = g.present[:, tcols[tvalid]] & g.open[:, tcols[tvalid]]
    s_i, h_i = np.nonzero(mask)
    h = hs[h_i]
    t = ci + h
    n = len(s_i)
    c = {}
    c["Store"] = g.store_ids[s_i].astype(np.int16)
    c["cutoff_idx"] = np.full(n, ci, np.int16)
    c["t_idx"] = t.astype(np.int16)
    c["sales"] = g.sales_raw[s_i, t]
    c["horizon"] = h.astype(np.float32)
    dates_t = g.dates[t]
    c["dow"] = g.dow[t].astype(np.float32)
    c["day_of_month"] = dates_t.day.to_numpy(np.float32)
    c["week_of_year"] = dates_t.isocalendar().week.to_numpy(np.float32)
    c["month"] = dates_t.month.to_numpy(np.float32)
    # holidays (calendar known through ci + 42)
    c["state_code"] = g.state_code[s_i, t]
    c["school_flag"] = g.school[s_i, t].astype(np.float32)
    nxt = g.next_state[np.minimum(t + 1, D - 1)]
    nxt = np.where(t + 1 > D - 1, 10 ** 6, nxt)
    c["days_to_state"] = np.where(nxt <= end, np.minimum(nxt - t, CAP_HOLIDAY), CAP_HOLIDAY).astype(np.float32)
    ls = g.last_state[t]
    c["days_since_state"] = np.where(ls >= 0, np.minimum(t - ls, CAP_HOLIDAY), CAP_HOLIDAY).astype(np.float32)
    nxs = np.where(t + 1 > D - 1, 10 ** 6, g.next_school[s_i, np.minimum(t + 1, D - 1)])
    c["days_to_school"] = np.where(nxs <= end, np.minimum(nxs - t, CAP_HOLIDAY), CAP_HOLIDAY).astype(np.float32)
    lsc = g.last_school[s_i, t]
    c["days_since_school"] = np.where(lsc >= 0, np.minimum(t - lsc, CAP_HOLIDAY), CAP_HOLIDAY).astype(np.float32)
    # promo (planned flags known through ci + 42)
    c["promo"] = g.promo[s_i, t]
    c["promo_m1"] = g.promo[s_i, np.maximum(t - 1, 0)]
    c["promo_m7"] = g.promo[s_i, np.maximum(t - 7, 0)]
    c["promo_p1"] = np.where(t + 1 <= end, g.promo[s_i, np.minimum(t + 1, D - 1)], np.nan).astype(np.float32)
    c["promo_p7"] = np.where(t + 7 <= end, g.promo[s_i, np.minimum(t + 7, D - 1)], np.nan).astype(np.float32)
    start = g.promo2_start[s_i]
    tdt = g.dates.to_numpy()[t]
    months = dates_t.month.to_numpy() - 1
    active = g.promo2[s_i] & (tdt >= start) & g.promo2_months[s_i, months]
    c["promo2_active"] = active.astype(np.float32)
    weeks = (tdt - start) / np.timedelta64(7, "D")
    c["promo2_weeks"] = np.where(g.promo2[s_i] & ~np.isnat(start), weeks, np.nan).astype(np.float32)
    # store attributes
    c["store_type"] = g.type_code[s_i]
    c["assortment"] = g.assort_code[s_i]
    c["comp_distance"] = g.comp_distance[s_i]
    c["comp_open_months"] = ((dates_t.year.to_numpy() * 12 + months) - g.comp_open_ym[s_i]).astype(np.float32)
    # history at the cutoff
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for L in (28, 56, 91):
            cnt = g.cc[:, ci + 1] - g.cc[:, ci + 1 - L]
            tot = g.cs[:, ci + 1] - g.cs[:, ci + 1 - L]
            mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
            c[f"mean_{L}"] = mean[s_i].astype(np.float32)
            c[f"median_{L}"] = np.nanmedian(g.sales_obs[:, ci - L + 1: ci + 1], axis=1)[s_i].astype(np.float32)
    c["n_obs_91"] = (g.cc[:, ci + 1] - g.cc[:, ci + 1 - 91])[s_i].astype(np.float32)
    c["frac_missing_91"] = ((g.cmiss[:, ci + 1] - g.cmiss[:, ci + 1 - 91]) / 91.0)[s_i].astype(np.float32)
    c["frac_closed_91"] = ((g.cclosed[:, ci + 1] - g.cclosed[:, ci + 1 - 91]) / 91.0)[s_i].astype(np.float32)
    lc = g.last_closure[:, ci]
    c["days_since_closure"] = np.where(lc >= 0, np.minimum(ci - lc, CAP_CLOSURE), CAP_CLOSURE)[s_i].astype(np.float32)
    sw = _same_weekday_features(g, ci)
    w_t = g.dow[t].astype(int)
    for name in ["swd_mean_4", "swd_mean_8", "swd_mean_12"] + SWP_AUX:
        col = np.empty(n, np.float32)
        for w in range(7):
            m = w_t == w
            col[m] = sw[w][name][s_i[m]]
        c[name] = col
    p_is1 = c["promo"] == 1
    for K in (4, 8, 12):
        c[f"swp_mean_{K}"] = np.where(p_is1, c[f"swp1_mean_{K}"], c[f"swp0_mean_{K}"])
    c["swp_n_12"] = np.where(p_is1, c["swp1_n_12"], c["swp0_n_12"])
    for name, period in (("lag7", 7), ("lag14", 14)):
        src = t - period * np.ceil(h / period).astype(int)
        assert (src <= ci).all()
        c[name] = np.where(src >= 0, g.sales_obs[s_i, np.maximum(src, 0)], np.nan).astype(np.float32)
    return c


def keep_flags(n, ci, seed=20260101, keep_fraction=0.2):
    """Fixed-seed training subsample, drawn per cutoff so it does not depend on which cutoffs are built."""
    return np.random.default_rng([seed, int(ci)]).random(n) < keep_fraction


def special_cutoff_indices(g: Grids, eval_cutoffs):
    """Evaluation cutoffs and the early-stopping holdout cutoffs of every fold (kept without subsampling)."""
    ev = {cutoff_index(g, c) for c in eval_cutoffs}
    return ev, ev | {e - o for e in ev for o in HOLDOUT_OFFSETS_DAYS}


def build_feature_table(frame: pd.DataFrame, store: pd.DataFrame, cutoffs=None, seed=20260101, keep_fraction=0.2):
    """In-memory table for all given cutoffs (tests / small data). `keep` marks the fixed-seed training subsample."""
    g = Grids(frame, store)
    cutoffs = friday_cutoffs() if cutoffs is None else pd.DatetimeIndex(cutoffs)
    parts = []
    for cd in cutoffs:
        ci = cutoff_index(g, cd)
        p = build_cutoff_rows(g, ci)
        p["keep"] = keep_flags(len(p["Store"]), ci, seed, keep_fraction)
        parts.append(p)
    keys = parts[0].keys()
    tab = pd.DataFrame({k: np.concatenate([p[k] for p in parts]) for k in keys})
    return tab, g


def stream_feature_files(frame, store, eval_cutoffs, path_main, path_aux, cutoffs=None, seed=20260101, keep_fraction=0.2):
    """Memory-lean build: one parquet row group per cutoff. Only rows some fold can use are written:
    Sales > 0 and (in the training subsample, or in an evaluation / holdout cutoff). The promo-switch columns (SWP_AUX)
    are written only for evaluation-cutoff rows (counterfactual analysis). Returns summary counts."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    g = Grids(frame, store)
    cutoffs = friday_cutoffs() if cutoffs is None else pd.DatetimeIndex(cutoffs)
    ev, special = special_cutoff_indices(g, eval_cutoffs)
    writer, aux_parts, n_all, n_kept = None, [], 0, 0
    for cd in cutoffs:
        ci = cutoff_index(g, cd)
        p = build_cutoff_rows(g, ci)
        n = len(p["Store"])
        pos = p["sales"] > 0
        keep = keep_flags(n, ci, seed, keep_fraction)
        need = pos & (keep | (ci in special))
        n_all += n
        n_kept += int(need.sum())
        if ci in ev:
            aux_parts.append(pd.DataFrame({k: p[k][pos] for k in ["Store", "cutoff_idx", "t_idx"] + SWP_AUX}))
        main = pd.DataFrame({k: v[need] for k, v in p.items() if k not in SWP_AUX})
        main["keep"] = keep[need]
        tbl = pa.Table.from_pandas(main, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(path_main), tbl.schema)
        writer.write_table(tbl)
    writer.close()
    pd.concat(aux_parts, ignore_index=True).to_parquet(path_aux, index=False)
    return {"rows_built": n_all, "rows_written": n_kept, "cutoffs": len(cutoffs), "eval_cutoff_indices": sorted(ev)}
