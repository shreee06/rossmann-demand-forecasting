import numpy as np
import pandas as pd
import pytest

from demand_forecast import intervals as I


def test_interval_metrics_hand_calculated():
    # y = [100, 200, 300]; interval [lo, hi] = [90, 110], [210, 250], [250, 310]
    # covered: yes, no (200 < 210, below), yes -> coverage 2/3, below 1/3, above 0
    # width = 20 + 40 + 60 = 120 ; sum y = 600 -> relative width 0.2
    # pinball q10 (alpha 0.1) on lo: y-lo = [10, -10, 50] -> [max(1,-9)=1, max(-1, 9)=9, max(5,-45)=5]; mean 15/3 = 5
    # pinball q90 on hi: y-hi = [-10, -50, -10] -> [max(-9, 1)=1, max(-45, 5)=5, 1]; mean 7/3
    d = pd.DataFrame({"y": [100.0, 200.0, 300.0], "lo": [90.0, 210.0, 250.0], "hi": [110.0, 250.0, 310.0]})
    m = I.compute(d)
    assert m["coverage"] == pytest.approx(2 / 3) and m["share_below"] == pytest.approx(1 / 3) and m["share_above"] == 0
    assert m["rel_width"] == pytest.approx(0.2)
    assert m["pinball_q10"] == pytest.approx(5.0) and m["pinball_q90"] == pytest.approx(7 / 3)


def test_bootstrap_summary_point_values():
    from demand_forecast.bootstrap import StoreBootstrap
    rows = []
    for fold in (1, 2):
        for s in range(6):
            rows.append({"fold": fold, "Store": s, "y": 100.0, "lo": 90.0, "hi": 110.0 if (s + fold) % 2 else 99.0})
    p = pd.DataFrame(rows)
    T, groups = I.tensor(p, list(range(6)), [1, 2])
    pf, pv = I.summarize(T, [1, 2], StoreBootstrap(6, n_boot=40, seed=1), {"all": [1, 2]})
    # hi = 110 covers y=100; hi = 99 does not (above). Half the rows are covered in each fold.
    assert pf[1][0]["coverage"]["value"] == pytest.approx(0.5) and pv["all"][0]["coverage"]["value"] == pytest.approx(0.5)
    assert pf[1][0]["coverage"]["lo"] <= 0.5 <= pf[1][0]["coverage"]["hi"]
