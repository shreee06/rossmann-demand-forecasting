import pandas as pd
import pytest

from demand_forecast.data import STATE_HOLIDAY_CODES, normalize_state_holiday


def test_mixed_int_and_str_zero_are_unified():
    s = pd.Series([0, "0", 0.0, "0.0", "a", "B", " c ", 0], dtype=object)
    out = normalize_state_holiday(s)
    assert list(out.astype(str)) == ["0", "0", "0", "0", "a", "b", "c", "0"]
    assert list(out.cat.categories) == STATE_HOLIDAY_CODES


def test_categories_fixed_even_if_some_codes_absent():
    out = normalize_state_holiday(pd.Series(["0", "0"]))
    assert list(out.cat.categories) == STATE_HOLIDAY_CODES


@pytest.mark.parametrize("bad", [["x"], ["0", "d"], ["1"]])
def test_unknown_values_raise(bad):
    with pytest.raises(ValueError, match="Unexpected"):
        normalize_state_holiday(pd.Series(bad))


def test_missing_raises():
    with pytest.raises(ValueError, match="missing"):
        normalize_state_holiday(pd.Series(["0", None], dtype=object))


def test_raw_csv_with_numeric_zero_roundtrip(tmp_path):
    """A copy of the file where the 'no holiday' value is an unquoted number must give the same result."""
    from demand_forecast.data import read_raw_train, to_processed_train
    header = "Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday\n"
    p = tmp_path / "train.csv"
    p.write_text(header + "1,5,2020-01-03,100,10,1,1,0,1\n1,6,2020-01-04,0,0,0,0,a,0\n")
    df = to_processed_train(read_raw_train(p))
    assert list(df["StateHoliday"].astype(str)) == ["0", "a"]
