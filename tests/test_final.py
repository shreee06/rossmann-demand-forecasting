import json

import numpy as np
import pandas as pd
import pytest

from demand_forecast import calibration as CAL
from demand_forecast import final as FN
from demand_forecast import model_features as MF
from demand_forecast.backtest import assert_split_allowed


def test_second_test_run_is_refused_without_override(tmp_path):
    p = tmp_path / "record.json"
    FN.check_and_write_record(p, {"timestamp": "t1", "n": 1})
    with pytest.raises(RuntimeError, match="refusing"):
        FN.check_and_write_record(p, {"timestamp": "t2"})
    assert json.loads(p.read_text()) == {"timestamp": "t1", "n": 1}  # untouched by the refused attempt
    rec = FN.check_and_write_record(p, {"timestamp": "t3"}, override="documented reason")
    assert rec["overrides"][0]["reason"] == "documented reason" and rec["timestamp"] == "t1"


def test_test_split_is_blocked_unless_explicitly_allowed():
    sp = FN.test_split()
    assert (sp.cutoff, sp.horizon_start, sp.horizon_end) == (pd.Timestamp("2015-06-19"), pd.Timestamp("2015-06-20"), pd.Timestamp("2015-07-31"))
    with pytest.raises(ValueError, match="TEST"):
        assert_split_allowed(sp)
    assert_split_allowed(sp, allow_test=True)


def test_conformal_factor_hand_example():
    # scores 0.5, 0.8, 0.9, 1.1, 1.3 (n = 5); ceil((5 + 1) * 0.8) = ceil(4.8) = 5 -> 5th smallest = 1.3
    assert CAL.conformal_factor([1.3, 0.5, 1.1, 0.9, 0.8]) == pytest.approx(1.3)
    # n = 9 -> ceil(10 * 0.8) = 8 -> 8th smallest of 0.1..0.9 = 0.8
    assert CAL.conformal_factor(np.arange(1, 10) / 10) == pytest.approx(0.8)
    # score: y = 100 in [80, 125] -> max(0.8, 0.8) = 0.8 ; y = 50 below lo = 80 -> max(1.6, 0.4) = 1.6
    assert list(CAL.scores([100, 50], [80, 80], [125, 125])) == pytest.approx([0.8, 1.6])
    lo, hi = CAL.apply_factors([80, 80], [125, 125], ["1-7", "8-14"], {"1-7": 1.25, "8-14": 0.8})
    assert list(lo) == pytest.approx([64, 100]) and list(hi) == pytest.approx([156.25, 100])


def test_calibration_refuses_rows_after_train():
    ok = pd.DataFrame({"y": [100.0, 90.0], "lo": [80.0, 95.0], "hi": [120.0, 130.0], "horizon_bucket": ["1-7", "1-7"],
                       "Date": pd.to_datetime(["2015-05-01", "2015-05-08"])})
    assert set(CAL.fit_factors(ok)) == {"1-7"}
    for bad in ("2015-05-09", "2015-06-25"):  # a VALIDATION date and a TEST date
        with pytest.raises(ValueError, match="backtest"):
            CAL.fit_factors(ok.assign(Date=pd.to_datetime(["2015-05-01", bad])))


def test_final_training_data_ends_on_2015_06_19():
    assert FN.final_cutoff() == pd.Timestamp("2015-06-19")
    assert FN.training_cutoffs().max() <= pd.Timestamp("2015-06-12")
    frame = pd.DataFrame({"Date": pd.to_datetime(["2015-06-19", "2015-06-20"]), "Store": 1, "Sales": 1, "Open": 1, "Promo": 0,
                          "StateHoliday": "0", "SchoolHoliday": 0})
    with pytest.raises(ValueError, match="must end"):
        FN.build_final_training_table(frame, pd.DataFrame({"Store": [1]}))


def test_final_training_rows_never_target_after_the_cutoff():
    """Synthetic panel 2013-01-01..2015-06-19: every training row targets a day <= 2015-06-19, has Sales > 0, is subsampled."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2013-01-01", "2015-06-19")
    rows = [{"Store": s, "Date": d, "Sales": int(rng.integers(100, 900)) * int(d.dayofweek != 6), "Open": int(d.dayofweek != 6),
             "Promo": int(d.dayofweek < 5 and d.isocalendar().week % 2 == 0), "StateHoliday": "0", "SchoolHoliday": 0}
            for s in (1, 2) for d in dates]
    store = pd.DataFrame({"Store": [1, 2], "StoreType": pd.Categorical(["a", "b"]), "Assortment": pd.Categorical(["a", "a"]),
                          "CompetitionDistance": [100.0, 200.0], "CompetitionOpenSinceMonth": pd.array([1, pd.NA], dtype="Int16"),
                          "CompetitionOpenSinceYear": pd.array([2010, pd.NA], dtype="Int16"), "Promo2": [0, 0],
                          "Promo2SinceWeek": pd.array([pd.NA, pd.NA], dtype="Int16"), "Promo2SinceYear": pd.array([pd.NA, pd.NA], dtype="Int16"),
                          "PromoInterval": pd.array([pd.NA, pd.NA], dtype="string")})
    tab = FN.build_final_training_table(pd.DataFrame(rows), store, keep_fraction=0.5)
    last = (pd.Timestamp("2015-06-19") - pd.Timestamp("2013-01-01")).days
    assert len(tab) > 0 and tab["t_idx"].max() == last and (tab["sales"] > 0).all()
    assert not [c for c in tab.columns if "customers" in c.lower()]
    assert set(MF.FEATURES) <= set(tab.columns)


def test_blanking_removes_test_sales():
    f = pd.DataFrame({"Date": pd.to_datetime(["2015-06-19", "2015-06-20", "2015-07-31"]), "Sales": [5, 6, 7]})
    b = FN.blank_sales_after(f, "2015-06-19")
    assert list(b["Sales"]) == [5, 0, 0] and list(f["Sales"]) == [5, 6, 7]


def test_frozen_trees_match_the_validation_run():
    v = json.loads(open("evals/results/model_validation.json").read())
    assert FN.FROZEN_TREES == v["trees"]
