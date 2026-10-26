"""Feature registry and builder. Only information known at forecast time may enter a feature.

LEAKAGE RULE: `Customers` is only observed together with Sales, so it is NOT known at forecast time.
It must never be a feature, nor be used to build one (no lags, rolling means, ratios, ...).
`assert_no_leakage` rejects any feature name containing "customers" (case-insensitive), and the
builder reads only the allow-listed input columns below.
"""
import pandas as pd

LEAKY_SUBSTRINGS = ("customers",)

# Columns the builder is allowed to read: calendar, holidays, planned promotions, store attributes.
ALLOWED_INPUT_COLUMNS = ["Store", "Date", "Promo", "StateHoliday", "SchoolHoliday"]

FEATURE_COLUMNS = ["Store", "dow", "day_of_month", "month", "week_of_year", "day_of_year",
                   "is_month_start", "is_month_end", "promo", "state_holiday", "school_holiday"]


def assert_no_leakage(columns) -> None:
    for c in columns:
        if any(s in str(c).lower() for s in LEAKY_SUBSTRINGS):
            raise ValueError(f"Leakage: '{c}' is derived from Customers, which is unknown at forecast time")


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar/holiday/promo features. Any extra columns in df (including Customers) are ignored."""
    missing = [c for c in ALLOWED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing input columns: {missing}")
    x = df[ALLOWED_INPUT_COLUMNS]
    d = x["Date"].dt
    out = pd.DataFrame({
        "Store": x["Store"].to_numpy(),
        "dow": d.dayofweek.to_numpy(),
        "day_of_month": d.day.to_numpy(),
        "month": d.month.to_numpy(),
        "week_of_year": d.isocalendar().week.astype("int16").to_numpy(),
        "day_of_year": d.dayofyear.to_numpy(),
        "is_month_start": d.is_month_start.astype("int8").to_numpy(),
        "is_month_end": d.is_month_end.astype("int8").to_numpy(),
        "promo": x["Promo"].astype("int8").to_numpy(),
        "state_holiday": x["StateHoliday"].astype(str).map({"0": 0, "a": 1, "b": 2, "c": 3}).astype("int8").to_numpy(),
        "school_holiday": x["SchoolHoliday"].astype("int8").to_numpy(),
    }, index=df.index)
    assert_no_leakage(out.columns)
    assert list(out.columns) == FEATURE_COLUMNS
    return out
