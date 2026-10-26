import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_train():
    """Small synthetic panel: 3 stores x 20 days, with one missing store-day, a closure and a holiday."""
    dates = pd.date_range("2020-01-01", periods=20)
    rows = []
    rng = np.random.default_rng(0)
    for store in (1, 2, 3):
        for d in dates:
            if store == 3 and d == dates[5]:
                continue  # missing store-day
            is_open = 0 if d.dayofweek == 6 else 1
            sales = int(rng.integers(100, 500)) if is_open else 0
            rows.append({"Store": store, "DayOfWeek": d.dayofweek + 1, "Date": d, "Sales": sales,
                         "Customers": sales // 10, "Open": is_open, "Promo": int(d.day % 2 == 0),
                         "StateHoliday": "a" if d == dates[0] else "0", "SchoolHoliday": int(d.day < 5)})
    return pd.DataFrame(rows)
