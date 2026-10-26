import numpy as np
import pandas as pd
import pytest

from demand_forecast import backtest as BT
from demand_forecast.baselines import PromoWeekdayMean, SeasonalNaive, StoreLevelIndex
from demand_forecast.windows import make_windows


def synth_frame(start="2014-01-01", days=200, stores=(1, 2, 3)):
    rng = np.random.default_rng(7)
    rows = []
    for s in stores:
        for d in pd.date_range(start, periods=days):
            open_ = int(d.dayofweek != 6)
            rows.append({"Store": s, "Date": d, "Sales": int(rng.integers(100, 900)) * open_, "Open": open_,
                         "Promo": int(d.isocalendar().week % 2 == 0 and d.dayofweek < 5),
                         "StateHoliday": "0", "SchoolHoliday": 0})
    return pd.DataFrame(rows)


STORE_TABLE = pd.DataFrame({"Store": [1, 2, 3], "StoreType": pd.Categorical(["a", "b", "a"])})
SPLIT = BT.Split(1, pd.Timestamp("2014-05-31"), pd.Timestamp("2014-06-01"), pd.Timestamp("2014-07-12"))
FORECASTERS = [SeasonalNaive(7), SeasonalNaive(14), PromoWeekdayMean(8), StoreLevelIndex(26)]


class Spy:
    name, params = "spy", {}

    def __init__(self):
        self.seen = []

    def forecast(self, history, future, store, cutoff):
        self.seen.append((history, future, cutoff))
        return np.full(len(future), 5.0)


def test_forecaster_inputs_respect_cutoff_and_hide_unavailable_columns():
    spy = Spy()
    BT.run_splits(spy, synth_frame(), STORE_TABLE, [SPLIT])
    history, future, cutoff = spy.seen[0]
    assert history["Date"].max() <= cutoff                       # nothing after the cutoff in history
    assert future["Date"].min() > cutoff and future["Date"].max() == SPLIT.horizon_end
    assert (future["Date"] - cutoff).dt.days.max() == 42
    assert "Sales" not in future.columns                         # no target values for the horizon
    for df in (history, future):
        assert not [c for c in df.columns if "customers" in c.lower()]


@pytest.mark.parametrize("fc", FORECASTERS, ids=lambda f: f"{f.name}-{f.params}")
def test_changing_data_after_cutoff_does_not_change_forecasts(fc):
    frame = synth_frame()
    base, _ = BT.run_splits(fc, frame, STORE_TABLE, [SPLIT])
    after = frame["Date"] > SPLIT.cutoff
    changed = frame.copy()
    changed.loc[after, "Sales"] = (changed.loc[after, "Sales"] * 37 + 11).astype(int)  # wildly different future sales
    new, _ = BT.run_splits(fc, changed, STORE_TABLE, [SPLIT])
    assert len(base) == len(new) and (base["yhat"].to_numpy() == new["yhat"].to_numpy()).all()
    assert not (base["y"].to_numpy() == new["y"].to_numpy()).all()  # the actuals did change


def test_closed_days_forecast_zero_and_not_evaluated():
    frame = synth_frame()
    preds, diag = BT.run_splits(Spy(), frame, STORE_TABLE, [SPLIT])
    assert (preds["yhat"] == 5.0).all()                           # only open evaluated rows remain
    assert (preds["Date"].dt.dayofweek != 6).all()                # Sundays are closed in the synthetic data
    assert diag[0]["n_closed"] > 0 and diag[0]["n_evaluated"] == len(preds)


def test_missing_store_days_are_not_evaluated():
    frame = synth_frame()
    drop = (frame["Store"] == 2) & (frame["Date"] > "2014-06-10")
    preds, _ = BT.run_splits(Spy(), frame[~drop], STORE_TABLE, [SPLIT])
    assert preds[preds["Store"] == 2]["Date"].max() <= pd.Timestamp("2014-06-10")


def test_horizon_buckets_and_labels():
    preds, _ = BT.run_splits(Spy(), synth_frame(), STORE_TABLE, [SPLIT])
    assert set(preds["horizon_bucket"]) == {"1-7", "8-14", "15-21", "22-28", "29-35", "36-42"}
    assert preds.loc[preds["horizon"] == 8, "horizon_bucket"].eq("8-14").all()


def test_test_window_is_never_reachable():
    w = make_windows()
    test_start, test_end = w["test"]
    with pytest.raises(ValueError, match="TEST"):
        BT.assert_split_allowed(BT.Split(9, test_start - pd.Timedelta(days=43), test_start - pd.Timedelta(days=42), test_start))
    for sp in BT.backtest_splits() + [BT.validation_split()]:
        BT.assert_split_allowed(sp)
        assert sp.horizon_end < test_start
    df = pd.DataFrame({"Date": pd.date_range(test_start - pd.Timedelta(days=3), test_end), "Store": 1})
    assert BT.restrict_to_modelling_period(df)["Date"].max() == test_start - pd.Timedelta(days=1)


def test_validation_split_is_the_validation_window_after_train_end():
    w = make_windows()
    v = BT.validation_split()
    assert v.cutoff == w["train"][1] and (v.horizon_start, v.horizon_end) == w["validation"]


def test_modelling_columns_exclude_customers():
    assert not [c for c in BT.HISTORY_COLUMNS + BT.FUTURE_COLUMNS if "customers" in c.lower()]
