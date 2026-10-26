import numpy as np
import pandas as pd
import pytest

from demand_forecast import metrics as M


def test_metrics_hand_calculated():
    # y = [100, 200], yhat = [110, 180]  ->  e = yhat - y = [+10, -20]
    # RMSPE: pct errors e/y = [0.10, -0.10]; squares [0.01, 0.01]; mean 0.01; sqrt = 0.10
    # WAPE : (|10| + |-20|) / (100 + 200) = 30/300 = 0.10
    # bias : (10 - 20) / (100 + 200) = -10/300 = -1/30   (mean e / mean y; negative = under-forecast)
    # MASE with scale 10 for both rows: (10/10 + 20/10) / 2 = 1.5
    m = M.compute_metrics([100, 200], [110, 180], scale=[10, 10])
    assert m["rmspe"] == pytest.approx(0.10)
    assert m["wape"] == pytest.approx(0.10)
    assert m["bias"] == pytest.approx(-1 / 30)
    assert m["mase"] == pytest.approx(1.5)


def test_rmspe_penalises_low_actuals():
    # y = [10, 1000], yhat = [20, 1000]: pct errors [1.0, 0.0]; mean of squares 0.5; RMSPE = sqrt(0.5)
    # WAPE = 10 / 1010
    m = M.compute_metrics([10, 1000], [20, 1000])
    assert m["rmspe"] == pytest.approx(np.sqrt(0.5))
    assert m["wape"] == pytest.approx(10 / 1010)


def test_mase_skips_rows_without_scale():
    # row 1 scale 5 -> |e|/s = 10/5 = 2 ; row 2 has no scale (NaN) and is excluded from MASE only
    # MASE = 2 / 1 = 2 ; WAPE still uses both rows: (10 + 30) / (100 + 300) = 0.1
    m = M.compute_metrics([100, 300], [110, 270], scale=[5, np.nan])
    assert m["mase"] == pytest.approx(2.0)
    assert m["wape"] == pytest.approx(0.1)


def test_empty_group_gives_nan():
    assert np.isnan(M.metrics_from_stats(np.zeros(7))).all()


def test_seasonal_naive_scale_hand_calculated():
    # week 1 = [10,20,30,40,50,60,70], week 2 = [12,18,30,44,50,66,70]; pairs (t, t-7) in week 2:
    # |12-10|,|18-20|,|30-30|,|44-40|,|50-50|,|66-60|,|70-70| = 2,2,0,4,0,6,0 -> sum 14, n 7 -> scale 2.0
    d = pd.date_range("2020-01-06", periods=14)
    sales = [10, 20, 30, 40, 50, 60, 70, 12, 18, 30, 44, 50, 66, 70]
    h = pd.DataFrame({"Store": 1, "Date": d, "Sales": sales, "Open": 1})
    assert M.seasonal_naive_scale(h, 7).loc[1] == pytest.approx(2.0)
    # a closed day drops its pairs: close day 8 (Sales 0, Open 0) -> removes |18-20|=2 -> sum 12, n 6 -> 2.0
    h2 = h.copy()
    h2.loc[8, ["Sales", "Open"]] = 0
    assert M.seasonal_naive_scale(h2, 7).loc[1] == pytest.approx(2.0)
