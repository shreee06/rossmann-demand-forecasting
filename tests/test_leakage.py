"""Customers is unknown at forecast time: it must never be a feature or feed a feature builder."""
from pathlib import Path

import pandas as pd
import pytest

from demand_forecast import features
from demand_forecast.features import FEATURE_COLUMNS, assert_no_leakage, build_calendar_features

SRC = Path(features.__file__).parent
# Modules that legitimately mention Customers: loading/typing the raw data, auditing it, and the guard itself.
MAY_MENTION_CUSTOMERS = {"data.py", "audit.py", "features.py"}


def test_feature_list_has_no_customers():
    assert not [c for c in FEATURE_COLUMNS if "customers" in c.lower()]
    assert "Customers" not in features.ALLOWED_INPUT_COLUMNS
    assert_no_leakage(FEATURE_COLUMNS)


@pytest.mark.parametrize("name", ["Customers", "customers", "CUSTOMERS", "Customers_lag7", "roll_mean_customers_28"])
def test_guard_rejects_customers_columns(name):
    with pytest.raises(ValueError, match="Leakage"):
        assert_no_leakage(["Store", name])


def test_builder_output_has_no_customers_and_ignores_the_column(synth_train):
    a = build_calendar_features(synth_train)
    changed = synth_train.assign(Customers=synth_train["Customers"] * 1000 + 7)
    b = build_calendar_features(changed)
    assert list(a.columns) == FEATURE_COLUMNS
    assert not [c for c in a.columns if "customers" in c.lower()]
    pd.testing.assert_frame_equal(a, b)  # features identical whatever Customers contains
    pd.testing.assert_frame_equal(a, build_calendar_features(synth_train.drop(columns=["Customers"])))


def test_builder_fails_if_a_customers_feature_were_added(monkeypatch, synth_train):
    monkeypatch.setattr(features, "FEATURE_COLUMNS", FEATURE_COLUMNS + ["customers_lag7"])
    with pytest.raises(ValueError, match="Leakage"):
        assert_no_leakage(features.FEATURE_COLUMNS)


def test_no_other_module_references_customers():
    offenders = [p.name for p in SRC.glob("*.py")
                 if p.name not in MAY_MENTION_CUSTOMERS and "customers" in p.read_text().lower()]
    assert offenders == [], f"modules reference Customers: {offenders}; review for leakage and extend the allow-list deliberately"


NEW_MODULES = ["metrics.py", "bootstrap.py", "backtest.py", "baselines.py", "evaluation.py", "model_features.py", "gbm.py",
               "intervals.py", "promo_analysis.py"]


@pytest.mark.parametrize("name", NEW_MODULES)
def test_forecasting_modules_never_mention_customers(name):
    path = SRC / name
    assert path.exists()
    assert name not in MAY_MENTION_CUSTOMERS
    assert "customers" not in path.read_text().lower()


def test_modelling_frame_loader_reads_an_explicit_column_list_without_customers(tmp_path):
    """load_modelling_frame must not be able to return Customers, even if the parquet contains it."""
    from demand_forecast import backtest
    df = pd.DataFrame({"Store": [1], "Date": pd.to_datetime(["2014-01-01"]), "Sales": [1], "Customers": [9],
                       "Open": [1], "Promo": [0], "StateHoliday": ["0"], "SchoolHoliday": [0], "DayOfWeek": [3]})
    p = tmp_path / "t.parquet"
    df.to_parquet(p)
    out = backtest.load_modelling_frame(p)
    assert not [c for c in out.columns if "customers" in c.lower()]
    assert list(out.columns) == backtest.HISTORY_COLUMNS
