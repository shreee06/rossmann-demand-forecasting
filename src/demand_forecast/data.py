"""Raw CSV -> typed Parquet. Raw files are only ever opened read-only."""
import pandas as pd

from demand_forecast import config

STATE_HOLIDAY_CODES = ["0", "a", "b", "c"]  # 0 = none, a = public, b = Easter, c = Christmas

TRAIN_DTYPES = {
    "Store": "int16", "DayOfWeek": "int8", "Sales": "int32", "Customers": "int32",
    "Open": "int8", "Promo": "int8", "SchoolHoliday": "int8",
}
STORE_NULLABLE_INT = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                      "Promo2SinceWeek", "Promo2SinceYear"]


def normalize_state_holiday(s: pd.Series) -> pd.Series:
    """StateHoliday is mixed type in Kaggle copies: the 'no holiday' value shows up as int 0, "0" or 0.0
    next to the strings a/b/c. Everything is cast to string, "0.0" is folded into "0", letters are
    lower-cased, and the result is a categorical over exactly {0, a, b, c}. Missing or unknown values
    raise instead of being silently coerced."""
    if s.isna().any():
        raise ValueError("StateHoliday contains missing values")
    out = s.astype("string").str.strip().str.lower().replace({"0.0": "0"})
    bad = sorted(set(out.unique()) - set(STATE_HOLIDAY_CODES))
    if bad:
        raise ValueError(f"Unexpected StateHoliday values: {bad}")
    return pd.Series(pd.Categorical(out.astype(object), categories=STATE_HOLIDAY_CODES), index=s.index)


def read_raw_train(path=None) -> pd.DataFrame:
    # StateHoliday read as str so the mixed int/str values never get type-guessed by pandas.
    return pd.read_csv(path or config.RAW_TRAIN, dtype={"StateHoliday": "string"})


def read_raw_store(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.RAW_STORE)


def to_processed_train(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "Store": raw["Store"], "DayOfWeek": raw["DayOfWeek"],
        "Date": pd.to_datetime(raw["Date"], format="%Y-%m-%d"),
        "Sales": raw["Sales"], "Customers": raw["Customers"],
        "Open": raw["Open"], "Promo": raw["Promo"],
        "StateHoliday": normalize_state_holiday(raw["StateHoliday"]),
        "SchoolHoliday": raw["SchoolHoliday"],
    })
    if df.isna().any().any():
        raise ValueError("train has missing values: " + str(df.isna().sum()[lambda x: x > 0].to_dict()))
    return df.astype(TRAIN_DTYPES)


def to_processed_store(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["StoreType"] = df["StoreType"].astype("category")
    df["Assortment"] = df["Assortment"].astype("category")
    df["CompetitionDistance"] = df["CompetitionDistance"].astype("float32")  # NaN kept as missing
    for c in STORE_NULLABLE_INT:
        df[c] = df[c].astype("Int16")
    df["Promo2"] = df["Promo2"].astype("int8")
    df["Store"] = df["Store"].astype("int16")
    df["PromoInterval"] = df["PromoInterval"].astype("string")  # NaN when the store has no Promo2
    return df


def convert_to_parquet(raw_train=None, raw_store=None, out_train=None, out_store=None):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train = to_processed_train(read_raw_train(raw_train))
    store = to_processed_store(read_raw_store(raw_store))
    train.to_parquet(out_train or config.PROCESSED_TRAIN, index=False)
    store.to_parquet(out_store or config.PROCESSED_STORE, index=False)
    return train, store


def load_processed():
    return pd.read_parquet(config.PROCESSED_TRAIN), pd.read_parquet(config.PROCESSED_STORE)
