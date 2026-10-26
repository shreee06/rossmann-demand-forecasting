import pandas as pd

from demand_forecast import audit as A
from demand_forecast.data import read_raw_train, to_processed_train


def write_csv(df, path):
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def test_reconciliation_matches_synthetic_example(synth_train, tmp_path):
    p = tmp_path / "train.csv"
    write_csv(synth_train, p)
    processed = to_processed_train(read_raw_train(p))
    src = A.source_stats_from_csv(p)
    assert src["rows"] == len(synth_train) == 59
    assert src["sales_sum"] == int(synth_train["Sales"].sum())
    assert src["state_holiday_counts"] == {"0": 56, "a": 3}
    assert A.count_data_lines(p) == 59
    rec = A.reconcile(src, A.frame_stats(processed))
    assert rec["all_match"], rec


def test_reconciliation_detects_a_dropped_row(synth_train, tmp_path):
    p = tmp_path / "train.csv"
    write_csv(synth_train, p)
    processed = to_processed_train(read_raw_train(p)).iloc[1:]
    rec = A.reconcile(A.source_stats_from_csv(p), A.frame_stats(processed))
    assert not rec["all_match"]
    assert not rec["checks"]["rows"]["match"] and not rec["checks"]["sales_sum"]["match"]


def test_coverage_counts_missing_store_days(synth_train):
    assert A.coverage(synth_train, pd.DataFrame({"Store": [1, 2, 3]}))["missing_store_days"] == 1
    c = A.coverage(synth_train, pd.DataFrame({"Store": [1, 2, 3]}))
    assert c["n_stores"] == 3 and c["n_days"] == 20 and c["expected_store_days"] == 60
    assert c["rows"] == 59 and c["duplicate_store_date_rows"] == 0
    assert c["stores_with_any_missing_day"] == 1 and c["dayofweek_consistent_with_date"]


def test_closed_and_zero_sales_days(synth_train):
    df = synth_train.copy()
    i = df.index[(df["Open"] == 1)][0]
    df.loc[i, "Sales"] = 0  # an open day with zero sales
    c = A.closed_days(df)
    n_closed = int((df["Open"] == 0).sum())
    assert c["open_zero_rows"] == n_closed and c["open_zero_sales_sum"] == 0
    assert c["open_rows_with_zero_sales"] == 1


def test_long_closure_runs_counts_missing_and_closed_days():
    dates = pd.date_range("2020-01-01", periods=10)
    rows = []
    for d in dates:
        if d in dates[3:6]:
            continue  # 3 missing days
        rows.append({"Store": 1, "DayOfWeek": d.dayofweek + 1, "Date": d, "Sales": 0 if d in dates[6:8] else 10,
                     "Customers": 1, "Open": 0 if d in dates[6:8] else 1, "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0})
    r = A.long_closures(pd.DataFrame(rows), min_run=5)
    assert r["n_runs"] == 1
    run = r["runs"][0]
    assert (run["start"], run["end"], run["length"]) == ("2020-01-04", "2020-01-08", 5)
    assert run["missing_days"] == 3 and run["closed_rows"] == 2
    assert A.long_closures(pd.DataFrame(rows), min_run=6)["n_runs"] == 0
