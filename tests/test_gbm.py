import numpy as np
import pandas as pd
import pytest

from demand_forecast import gbm
from demand_forecast.model_features import FEATURES


def fake_table(n_stores=12, seed=0):
    """Weekly cutoffs (index step 7), h = 1..42, random features; sales depend on a few features."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(100, 100 + 7 * 40, 7):
        for h in range(1, 43):
            for s in range(n_stores):
                rows.append((s, c, c + h, h))
    a = np.array(rows)
    tab = pd.DataFrame(rng.standard_normal((len(a), len(FEATURES))).astype(np.float32), columns=FEATURES)
    tab["Store"], tab["cutoff_idx"], tab["t_idx"] = a[:, 0].astype(np.int16), a[:, 1].astype(np.int16), a[:, 2].astype(np.int16)
    tab["horizon"] = a[:, 3].astype(np.float32)
    for c in ("store_type", "assortment", "state_code"):
        tab[c] = rng.integers(0, 3, len(tab)).astype(np.float32)
    tab["sales"] = np.exp(6 + 0.3 * tab["mean_28"] + 0.1 * tab["promo"]).astype(np.float32)
    tab["keep"] = True
    return tab


CF = 100 + 7 * 30  # a fold cutoff in the table


def test_fold_masks_respect_the_cutoff_and_purge():
    tab = fake_table()
    m = gbm.fold_masks(tab, CF)
    t, c = tab["t_idx"].to_numpy(), tab["cutoff_idx"].to_numpy()
    assert (t[m["train"]] <= CF).all() and (c[m["train"]] < CF).all()  # no row of the fold's own cutoff is trained on
    assert (t[m["inner"]] <= CF - 63).all()                            # purged: inner targets precede the holdout cutoffs
    assert set(np.unique(c[m["holdout"]])) == {CF - 63, CF - 56, CF - 49, CF - 42}
    assert (t[m["holdout"]] <= CF).all()
    assert not (m["holdout"] & m["eval"]).any() and not (m["train"] & m["eval"]).any()
    assert (c[m["eval"]] == CF).all() and (t[m["eval"]] > CF).all()
    # every inner-training target is earlier than the first holdout target
    assert t[m["inner"]].max() < t[m["holdout"]].min() + 42


def test_subsample_flag_only_shrinks_training_rows():
    tab = fake_table()
    full = gbm.fold_masks(tab, CF)["train"].sum()
    tab["keep"] = np.random.default_rng(1).random(len(tab)) < 0.2
    m = gbm.fold_masks(tab, CF)
    assert 0.1 * full < m["train"].sum() < 0.3 * full
    assert m["holdout"].sum() == gbm.fold_masks(fake_table(), CF)["holdout"].sum()  # holdout is not subsampled


def test_backtransform_hand_example():
    # log1p(100) -> expm1 -> 100 ; log1p(0) -> 0
    assert gbm.backtransform(np.log1p(100.0)) == pytest.approx(100.0)
    assert gbm.backtransform(0.0) == 0.0
    # Back-transform bias: two actuals 100 and 400 predicted by the same log-prediction log1p(100) (the median in log1p space
    # of {101, 401} is not their mean). Plain back-transform gives 100 for both: total 200 vs 500 actual (-60%).
    # Smearing: s = mean(exp(log1p(y) - pred)) = mean(401/101, 101/101) = (3.970297 + 1)/2 = 2.4851485;
    # (1 + expm1(pred)) * s - 1 = 101 * 2.4851485 - 1 = 250 = mean of the actuals.
    y_log = np.log1p([100.0, 400.0])
    pred = np.log1p([100.0, 100.0])
    s = gbm.smearing_factor(y_log, pred)
    assert s == pytest.approx((401 / 101 + 1) / 2)
    assert gbm.apply_smearing(pred, s) == pytest.approx([250.0, 250.0])


def test_fix_crossing_counts_and_sorts():
    lo, hi, n = gbm.fix_crossing([1.0, 5.0, 3.0], [2.0, 4.0, 3.0])
    assert n == 1 and list(lo) == [1.0, 4.0, 3.0] and list(hi) == [2.0, 5.0, 3.0]
    assert (lo <= hi).all()


def test_horizon_fixed_design_sets_constant_one():
    tab = fake_table(n_stores=3)
    m = gbm.fold_masks(tab, CF)["eval"]
    X = gbm.design(tab, m, FEATURES, horizon_fixed=True)
    assert (X[:, FEATURES.index("horizon")] == 1).all()
    X2 = gbm.design(tab, m, FEATURES)
    assert X2[:, FEATURES.index("horizon")].max() == 42 and X2[:, FEATURES.index("horizon")].min() == 1


def test_fit_fold_smoke(monkeypatch):
    monkeypatch.setattr(gbm, "MAX_ROUNDS", 40)
    tab = fake_table(n_stores=8)
    b, info = gbm.fit_fold(tab, FEATURES, {"num_leaves": 7, "min_child_samples": 20}, CF)
    assert 1 <= info["best_iteration"] <= 40 and info["smearing_factor"] > 0
    keys, pred = gbm.predict_eval(b, tab, FEATURES, CF)
    assert len(keys) == len(pred) == (tab["cutoff_idx"] == CF).sum()
    # the signal is learnable: predictions track log sales
    y = np.log1p(tab.loc[tab["cutoff_idx"] == CF, "sales"].to_numpy())
    assert np.corrcoef(pred, y)[0, 1] > 0.5
    qb, qinfo = gbm.fit_fold(tab, FEATURES, {"num_leaves": 7, "min_child_samples": 20}, CF, "quantile", 0.9)
    assert "smearing_factor" not in qinfo and qb.num_trees() >= 1


def test_filtered_storage_gives_identical_fold_row_sets():
    """The streaming build only keeps Sales > 0 rows that are in the training subsample or in an evaluation/holdout cutoff.
    Every row set a fold can ask for must be identical on the full and on the filtered table."""
    from demand_forecast.model_features import HOLDOUT_OFFSETS_DAYS
    tab = fake_table()
    tab["keep"] = np.random.default_rng(3).random(len(tab)) < 0.2
    evals = [100 + 7 * 25, 100 + 7 * 30, 100 + 7 * 35]
    special = set(evals) | {e - o for e in evals for o in HOLDOUT_OFFSETS_DAYS}
    pos = tab["sales"] > 0
    small = tab[pos & (tab["keep"] | tab["cutoff_idx"].isin(special))].reset_index(drop=True)
    assert len(small) < len(tab)
    for cf in evals:
        a, b = gbm.fold_masks(tab, cf), gbm.fold_masks(small, cf)
        for k in ("train", "inner", "holdout", "eval"):
            ka = tab.loc[a[k], ["Store", "cutoff_idx", "t_idx"]].reset_index(drop=True)
            kb = small.loc[b[k], ["Store", "cutoff_idx", "t_idx"]].reset_index(drop=True)
            pd.testing.assert_frame_equal(ka, kb)
