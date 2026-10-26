import pandas as pd

from demand_forecast import config
from demand_forecast.windows import backtest_folds, make_windows


def test_windows_in_order_and_disjoint():
    w = make_windows()
    (tr_a, tr_b), (va_a, va_b), (te_a, te_b) = w["train"], w["validation"], w["test"]
    assert tr_a < tr_b < va_a <= va_b < te_a <= te_b
    assert va_a - tr_b == pd.Timedelta(days=1) and te_a - va_b == pd.Timedelta(days=1)  # contiguous, no overlap
    assert (te_b - te_a).days + 1 == config.TEST_DAYS
    assert (va_b - va_a).days + 1 == config.VAL_DAYS
    assert te_b == pd.Timestamp(config.DATA_END) and tr_a == pd.Timestamp(config.DATA_START)


def test_windows_synthetic_dates():
    w = make_windows("2020-01-01", "2020-12-31", test_days=7, val_days=7)
    assert w["test"] == (pd.Timestamp("2020-12-25"), pd.Timestamp("2020-12-31"))
    assert w["validation"] == (pd.Timestamp("2020-12-18"), pd.Timestamp("2020-12-24"))
    assert w["train"][1] == pd.Timestamp("2020-12-17")


def test_backtest_folds_inside_train_ordered_and_non_overlapping():
    tr_a, tr_b = make_windows()["train"]
    folds = backtest_folds(tr_a, tr_b)
    assert len(folds) >= 2
    for f in folds:
        assert f["cutoff"] < f["horizon_start"] <= f["horizon_end"] <= tr_b  # never touches validation/test
        assert f["cutoff"] >= tr_a
        assert (f["horizon_end"] - f["horizon_start"]).days + 1 == config.BACKTEST_HORIZON_DAYS
        assert f["horizon_start"] - f["cutoff"] == pd.Timedelta(days=1)
    for a, b in zip(folds, folds[1:]):
        assert a["horizon_end"] < b["horizon_start"] and a["cutoff"] < b["cutoff"]
    assert folds[-1]["horizon_end"] == tr_b
    assert folds[0]["horizon_start"] >= tr_b - pd.Timedelta(days=config.BACKTEST_SPAN_DAYS - 1)


def test_windows_too_short_raises():
    import pytest
    with pytest.raises(ValueError):
        make_windows("2020-01-01", "2020-02-01", test_days=30, val_days=30)
