import numpy as np
import pandas as pd
import pytest

from demand_forecast import model_features as MF

START = pd.Timestamp("2013-01-01")  # a Tuesday
N_DAYS = 273
CUTOFF = pd.Timestamp("2013-06-07")  # a Friday, index 157 -> horizon ends at index 199 < 273


def make_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in (1, 2, 3):
        for i, d in enumerate(pd.date_range(START, periods=N_DAYS)):
            if s == 3 and 100 <= i < 120:
                continue  # missing store-days
            opn = int(d.dayofweek != 6 and rng.random() > 0.03)
            rows.append({"Store": s, "Date": d, "Sales": int(rng.integers(200, 900)) * opn, "Open": opn,
                         "Promo": int(d.dayofweek < 5 and d.isocalendar().week % 2 == 0),
                         "StateHoliday": "a" if i in (130, 168, 190) else "0",
                         "SchoolHoliday": int((i // 20 + s) % 4 == 0)})
    return pd.DataFrame(rows)


def make_store():
    return pd.DataFrame({
        "Store": [1, 2, 3], "StoreType": pd.Categorical(["a", "b", "a"]), "Assortment": pd.Categorical(["a", "a", "c"]),
        "CompetitionDistance": [500.0, np.nan, 1200.0],
        "CompetitionOpenSinceMonth": pd.array([3, pd.NA, 12], dtype="Int16"),
        "CompetitionOpenSinceYear": pd.array([2012, pd.NA, 2010], dtype="Int16"),
        "Promo2": [1, 0, 1], "Promo2SinceWeek": pd.array([5, pd.NA, 20], dtype="Int16"),
        "Promo2SinceYear": pd.array([2013, pd.NA, 2012], dtype="Int16"),
        "PromoInterval": pd.array(["Jan,Apr,Jul,Oct", pd.NA, "Mar,Jun,Sept,Dec"], dtype="string")})


def build(frame, cutoff=CUTOFF):
    tab, g = MF.build_feature_table(frame, make_store(), cutoffs=[cutoff], keep_fraction=1.0)
    return tab.drop(columns=["keep"]).sort_values(["Store", "t_idx"]).reset_index(drop=True), g


def feat_equal(a, b, cols=None):
    cols = cols or list(a.columns)
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_feature_names_have_no_customers_and_exist():
    assert not [f for f in MF.FEATURES if "customers" in f.lower()]
    assert len(set(MF.FEATURES)) == len(MF.FEATURES)
    tab, _ = build(make_frame())
    assert set(MF.FEATURES) <= set(tab.columns)
    assert not [c for c in tab.columns if "customers" in c.lower()]


def test_horizon_range_and_target_rows():
    tab, g = build(make_frame())
    ci = MF.cutoff_index(g, CUTOFF)
    assert tab["horizon"].min() >= 1 and tab["horizon"].max() == 42
    assert (tab["horizon"].to_numpy() == (tab["t_idx"] - tab["cutoff_idx"]).to_numpy()).all()
    assert (tab["cutoff_idx"] == ci).all() and (tab["t_idx"] > ci).all() and (tab["t_idx"] <= ci + 42).all()
    # only days on which the store is open and has a row
    f = make_frame()
    open_days = {(r.Store, (r.Date - START).days) for r in f.itertuples() if r.Open == 1}
    assert all((s, t) in open_days for s, t in zip(tab["Store"], tab["t_idx"]))
    assert (tab.groupby("Store")["t_idx"].count() > 0).all()


def test_sales_after_the_cutoff_never_change_any_feature():
    f = make_frame()
    base, g = build(f)
    ci = MF.cutoff_index(g, CUTOFF)
    changed = f.copy()
    after = (changed["Date"] - START).dt.days > ci
    changed.loc[after, "Sales"] = (changed.loc[after, "Sales"] * 37 + 11).astype(int)
    new, _ = build(changed)
    feat_equal(base, new, MF.FEATURES)  # only the (unused) target column differs
    assert not (base["sales"].to_numpy() == new["sales"].to_numpy()).all()


def test_nothing_dated_after_cutoff_plus_42_changes_any_feature():
    f = make_frame()
    base, g = build(f)
    ci = MF.cutoff_index(g, CUTOFF)
    changed = f.copy()
    after = (changed["Date"] - START).dt.days > ci + 42
    n = int(after.sum())
    rng = np.random.default_rng(5)
    changed.loc[after, "Sales"] = rng.integers(1, 5000, n)
    changed.loc[after, "Promo"] = 1 - changed.loc[after, "Promo"]
    changed.loc[after, "StateHoliday"] = np.where(rng.random(n) < 0.5, "a", "0")
    changed.loc[after, "SchoolHoliday"] = 1 - changed.loc[after, "SchoolHoliday"]
    changed.loc[after, "Open"] = 1 - changed.loc[after, "Open"]
    new, _ = build(changed)
    feat_equal(base, new)


def test_history_features_move_with_the_cutoff_only_through_allowed_history():
    f = make_frame()
    base, g = build(f)
    ci = MF.cutoff_index(g, CUTOFF)
    # perturb sales long before the longest window (91 days) and before the lags: nothing changes
    old = f.copy()
    far = (old["Date"] - START).dt.days < ci - 91
    old.loc[far, "Sales"] = old.loc[far, "Sales"] + 1000
    feat_equal(base, build(old)[0], MF.FEATURES)
    # perturb sales inside the last 28 days: mean_28 must change
    recent = f.copy()
    inside = ((recent["Date"] - START).dt.days > ci - 28) & ((recent["Date"] - START).dt.days <= ci) & (recent["Sales"] > 0)
    recent.loc[inside, "Sales"] = recent.loc[inside, "Sales"] + 500
    new, _ = build(recent)
    assert not np.allclose(base["mean_28"], new["mean_28"], equal_nan=True)


def test_history_features_match_a_pandas_reference_for_two_cutoffs():
    f = make_frame()
    for cutoff in (CUTOFF, CUTOFF + pd.Timedelta(days=7)):
        tab, g = build(f, cutoff)
        ci = MF.cutoff_index(g, cutoff)
        obs = f[(f["Open"] == 1) & (f["Sales"] > 0) & (f["Date"] <= cutoff)]
        for s in (1, 2, 3):
            o = obs[obs["Store"] == s]
            for L in (28, 56, 91):
                w = o[o["Date"] > cutoff - pd.Timedelta(days=L)]["Sales"]
                row = tab[tab["Store"] == s].iloc[0]
                assert row[f"mean_{L}"] == pytest.approx(w.mean(), rel=1e-5)
                assert row[f"median_{L}"] == pytest.approx(w.median(), rel=1e-5)
        # same-weekday means and promo-matched means for every row
        for r in tab.sample(40, random_state=1).itertuples():
            t_date = START + pd.Timedelta(days=int(r.t_idx))
            o = obs[(obs["Store"] == r.Store) & (obs["Date"].dt.dayofweek == t_date.dayofweek)]
            for K in (4, 8, 12):
                w = o[o["Date"] > cutoff - pd.Timedelta(days=7 * K)]
                assert getattr(r, f"swd_mean_{K}") == pytest.approx(w["Sales"].mean(), rel=1e-5, nan_ok=True)
                wp = w[w["Promo"] == r.promo]
                assert getattr(r, f"swp_mean_{K}") == pytest.approx(wp["Sales"].mean(), rel=1e-5, nan_ok=True)
            k7 = int(np.ceil(r.horizon / 7))
            src = t_date - pd.Timedelta(days=7 * k7)
            v = obs[(obs["Store"] == r.Store) & (obs["Date"] == src)]["Sales"]
            assert r.lag7 == pytest.approx(v.iloc[0] if len(v) else np.nan, nan_ok=True)


def test_missing_rows_and_holiday_distances_hand_checked():
    f = make_frame()
    tab, g = build(f)
    ci = MF.cutoff_index(g, CUTOFF)
    # store 3 has rows missing at day indices 100..119 -> inside the last 91 days before ci=157 (indices 67..157): 20/91
    r3 = tab[tab["Store"] == 3].iloc[0]
    assert r3["frac_missing_91"] == pytest.approx(20 / 91, rel=1e-5)
    # state holidays at day indices 130, 168, 190 (all stores). For t = 160: next after 160 is 168 (8 days, <= ci+42=199),
    # last on or before 160 is 130 (30 days)
    row = tab[(tab["Store"] == 1) & (tab["t_idx"] == 160)]
    if len(row):
        assert row["days_to_state"].iloc[0] == 8 and row["days_since_state"].iloc[0] == 30
    # t = 195: next holiday after 195 is beyond the data window rules -> none within horizon -> capped 60; last was 190 -> 5
    row = tab[(tab["Store"] == 1) & (tab["t_idx"] == 195)]
    if len(row):
        assert row["days_to_state"].iloc[0] == 60 and row["days_since_state"].iloc[0] == 5


def test_promo_neighbours_beyond_the_horizon_are_unknown():
    tab, g = build(make_frame())
    ci = MF.cutoff_index(g, CUTOFF)
    last = tab[tab["t_idx"] == ci + 42]
    assert last["promo_p1"].isna().all() and last["promo_p7"].isna().all()
    early = tab[tab["t_idx"] <= ci + 10]
    assert early["promo_p7"].notna().all() and early["promo_m7"].notna().mean() > 0.9


def test_swp_selects_the_matching_promo_aux_column():
    tab, _ = build(make_frame())
    p1 = tab["promo"] == 1
    assert np.allclose(tab.loc[p1, "swp_mean_8"], tab.loc[p1, "swp1_mean_8"], equal_nan=True)
    assert np.allclose(tab.loc[~p1, "swp_mean_8"], tab.loc[~p1, "swp0_mean_8"], equal_nan=True)


def test_ablation_lists():
    assert set(MF.features_for("no_history")).isdisjoint(MF.HISTORY_FEATURES)
    assert "promo" not in MF.features_for("no_promo") and "swp_mean_4" not in MF.features_for("no_promo")
    assert "swd_mean_4" in MF.features_for("no_promo")
    assert set(MF.features_for("no_holiday")).isdisjoint(MF.HOLIDAY_FEATURES)
    assert MF.features_for("horizon_fixed_1") == MF.FEATURES


def test_streamed_files_equal_the_filtered_in_memory_table(tmp_path):
    f, store = make_frame(), make_store()
    cutoffs = [CUTOFF - pd.Timedelta(days=7), CUTOFF]
    main, aux = tmp_path / "m.parquet", tmp_path / "a.parquet"
    MF.stream_feature_files(f, store, [CUTOFF], main, aux, cutoffs=cutoffs)
    full, g = MF.build_feature_table(f, store, cutoffs=cutoffs)
    ci = MF.cutoff_index(g, CUTOFF)
    special, _ = {ci}, None
    _, special = MF.special_cutoff_indices(g, [CUTOFF])
    want = full[(full["sales"] > 0) & (full["keep"] | full["cutoff_idx"].isin(special))].drop(columns=MF.SWP_AUX).reset_index(drop=True)
    got = pd.read_parquet(main)
    pd.testing.assert_frame_equal(got[want.columns], want)
    a = pd.read_parquet(aux)
    assert (a["cutoff_idx"] == ci).all() and len(a) == int(((full["cutoff_idx"] == ci) & (full["sales"] > 0)).sum())
    assert set(MF.SWP_AUX) <= set(a.columns)
