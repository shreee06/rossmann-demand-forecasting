import numpy as np
import pandas as pd
import pytest

from demand_forecast import bootstrap as B
from demand_forecast import metrics as M


def make_preds():
    # two folds, three stores. Per row: y, yhat
    rows = []
    for fold, vals in {1: [(1, 100, 110), (2, 200, 180), (3, 50, 50)], 2: [(1, 100, 100), (2, 200, 220), (3, 50, 60)]}.items():
        for s, y, yh in vals:
            rows.append({"fold": fold, "Store": s, "y": y, "yhat": yh, "scale": 10.0})
    return pd.DataFrame(rows)


def test_counts_shape_and_determinism():
    a, b = B.StoreBootstrap(5, n_boot=20, seed=1), B.StoreBootstrap(5, n_boot=20, seed=1)
    assert a.counts.shape == (20, 5) and (a.counts.sum(1) == 5).all()
    assert (a.counts == b.counts).all()
    assert not (a.counts == B.StoreBootstrap(5, n_boot=20, seed=2).counts).all()


def test_point_values_and_fold_mean_hand_calculated():
    # fold 1: e = [+10, -20, 0]; WAPE = 30/350 ; fold 2: e = [0, +20, +10]; WAPE = 30/350 -> mean 30/350
    # fold 1 bias = (10-20+0)/350 = -10/350 ; fold 2 bias = +30/350 ; mean of fold biases = 10/350
    p = make_preds()
    T, groups = B.stat_tensor(p, [1, 2, 3], [1, 2])
    boot = B.StoreBootstrap(3, n_boot=50, seed=3)
    per_fold, per_view, _ = B.summarize(T, [1, 2], boot, {"all": [1, 2]})
    assert groups == ["all"]
    assert per_fold[1][0]["wape"]["value"] == pytest.approx(30 / 350)
    assert per_view["all"][0]["wape"]["value"] == pytest.approx(30 / 350)
    assert per_fold[1][0]["bias"]["value"] == pytest.approx(-10 / 350)
    assert per_view["all"][0]["bias"]["value"] == pytest.approx(10 / 350)
    for m in M.METRICS:
        c = per_view["all"][0][m]
        assert c["lo"] <= c["hi"]


def test_bootstrap_matches_manual_resample():
    p = make_preds()
    T, _ = B.stat_tensor(p, [1, 2, 3], [1, 2])
    boot = B.StoreBootstrap(3, n_boot=7, seed=4)
    _, _, bv = B.summarize(T, [1, 2], boot, {"f1": [1]})
    # manual: resample b has store multiplicities counts[b]; WAPE = sum(c*|e|) / sum(c*y) on fold 1
    abs_e, y = np.array([10, 20, 0.0]), np.array([100, 200, 50.0])
    for b in range(7):
        c = boot.counts[b]
        want = (c * abs_e).sum() / (c * y).sum()
        assert bv["f1"][b, 0, 1] == pytest.approx(want)


def test_identical_stores_give_degenerate_ci():
    rows = [{"fold": 1, "Store": s, "y": 100.0, "yhat": 90.0, "scale": 5.0} for s in range(10)]
    T, _ = B.stat_tensor(pd.DataFrame(rows), list(range(10)), [1])
    _, pv, _ = B.summarize(T, [1], B.StoreBootstrap(10, n_boot=30, seed=5), {"v": [1]})
    c = pv["v"][0]["wape"]
    assert c["value"] == pytest.approx(0.1) and c["lo"] == pytest.approx(0.1) and c["hi"] == pytest.approx(0.1)


def test_paired_difference():
    a, b = np.array([0.2, 0.3, 0.25]), np.array([0.1, 0.1, 0.1])
    d = B.paired_difference(a, b, 0.25, 0.1)
    assert d["diff"] == pytest.approx(0.15) and d["lo"] > 0
