import numpy as np
import pandas as pd
import pytest

from demand_forecast.baselines import ETSWeekly, PromoWeekdayMean, SeasonalNaive, StoreLevelIndex


def frame(rows):
    """rows: (store, 'YYYY-MM-DD', sales, promo[, open])"""
    return pd.DataFrame([{"Store": r[0], "Date": pd.Timestamp(r[1]), "Sales": r[2], "Open": r[4] if len(r) > 4 else 1,
                          "Promo": r[3], "StateHoliday": "0", "SchoolHoliday": 0} for r in rows])


def fut(rows):
    f = frame([(s, d, 0, p) for s, d, p in rows]).drop(columns=["Sales"])
    return f


STORES = pd.DataFrame({"Store": [1, 2]})


def test_seasonal_naive_7_and_14_hand_calculated():
    # 2020-01-06 (Mon) is day 0, sales = 100 + day index; history through cutoff 2020-01-19 (day 13, Sunday)
    days = pd.date_range("2020-01-06", "2020-01-19")
    hist = frame([(1, d.strftime("%Y-%m-%d"), 100 + i, 0) for i, d in enumerate(days)])
    cutoff = pd.Timestamp("2020-01-19")
    f = fut([(1, "2020-01-20", 0), (1, "2020-01-26", 0), (1, "2020-01-27", 0), (1, "2020-02-03", 0)])
    # period 7: h=1 (Mon 01-20): k=ceil(1/7)=1 -> 01-13 = day 7 -> 107
    #           h=7 (Sun 01-26): k=1 -> 01-19 = day 13 -> 113
    #           h=8 (Mon 01-27): k=ceil(8/7)=2 -> 01-13 -> 107
    #           h=15 (Mon 02-03): k=3 -> 01-13 -> 107
    assert list(SeasonalNaive(7).forecast(hist, f, STORES, cutoff)) == [107, 113, 107, 107]
    # period 14: h=1: k=1 -> 01-20 - 14 = 01-06 = day 0 -> 100 ; h=7: k=1 -> 01-12 = day 6 -> 106
    #            h=8: k=1 -> 01-13 = day 7 -> 107 ; h=15: k=2 -> 02-03 - 28 = 01-06 -> 100
    assert list(SeasonalNaive(14).forecast(hist, f, STORES, cutoff)) == [100, 106, 107, 100]


def test_seasonal_naive_steps_back_when_lag_day_was_closed_and_uses_fallback():
    days = pd.date_range("2020-01-06", "2020-01-19")
    rows = [(1, d.strftime("%Y-%m-%d"), 100 + i, 0, 0 if i == 7 else 1) for i, d in enumerate(days)]
    hist = frame(rows)
    hist.loc[hist["Open"] == 0, "Sales"] = 0  # 01-13 closed
    cutoff = pd.Timestamp("2020-01-19")
    # Mon 01-20: lag day 01-13 closed -> k+1 -> 01-06 = 100
    sn = SeasonalNaive(7)
    assert sn.forecast(hist, fut([(1, "2020-01-20", 0)]), STORES, cutoff)[0] == 100
    # store 2 has no history at all -> global dow mean of Mondays: store 1 Mondays observed: 01-06 (100) only -> 100
    out = sn.forecast(hist, fut([(2, "2020-01-20", 0)]), STORES, cutoff)
    assert out[0] == 100 and sn.fallback_counts["fb_global_dow_mean"] == 1


def test_promo_weekday_mean_hand_calculated_and_fallbacks():
    # Mondays: 12-23 promo 200, 12-30 no promo 100, 01-06 promo 220, 01-13 no promo 120 ; cutoff 2020-01-19
    hist = frame([(1, "2019-12-23", 200, 1), (1, "2019-12-30", 100, 0), (1, "2020-01-06", 220, 1), (1, "2020-01-13", 120, 0)])
    cutoff = pd.Timestamp("2020-01-19")
    f = fut([(1, "2020-01-20", 1), (1, "2020-01-20", 0), (1, "2020-01-21", 0)])
    # K=2 weeks: window is Date > 01-19 - 14d = 01-05 -> {01-06 promo 220, 01-13 no-promo 120}
    #   Monday promo -> 220 ; Monday no promo -> 120
    #   Tuesday: no obs anywhere -> store mean over all history = (200+100+220+120)/4 = 160
    m2 = PromoWeekdayMean(2)
    assert list(m2.forecast(hist, f, STORES, cutoff)) == [220, 120, 160]
    # K=4 weeks: window Date > 12-22 -> all four. promo: (200+220)/2 = 210 ; no promo: (100+120)/2 = 110
    assert list(PromoWeekdayMean(4).forecast(hist, f.iloc[:2], STORES, cutoff)) == [210, 110]
    # K=1: window Date > 01-12 -> only 01-13 (no promo). Monday promo has no match in window ->
    #   falls back to store x dow mean within window = 120 ; Monday no promo -> 120
    assert list(PromoWeekdayMean(1).forecast(hist, f.iloc[:2], STORES, cutoff)) == [120, 120]


def test_store_level_index_hand_calculated():
    # store 1: Mon(promo)=150, Tue=100, Wed=100 -> median 100, ratios 1.5, 1, 1
    # store 2: Mon(promo)=300, Tue=200, Wed=200 -> median 200, ratios 1.5, 1, 1
    # index (Mon,promo)=1.5, (Tue,0)=1, (Wed,0)=1. Forecast Mon promo store 1 = 100*1.5 = 150 ; Tue store 2 = 200*1
    hist = frame([(1, "2020-01-06", 150, 1), (1, "2020-01-07", 100, 0), (1, "2020-01-08", 100, 0),
                  (2, "2020-01-06", 300, 1), (2, "2020-01-07", 200, 0), (2, "2020-01-08", 200, 0)])
    cutoff = pd.Timestamp("2020-01-08")
    f = fut([(1, "2020-01-13", 1), (2, "2020-01-14", 0), (1, "2020-01-15", 1)])
    out = StoreLevelIndex(13).forecast(hist, f, STORES, cutoff)
    # third row: (Wed, promo=1) cell is empty -> dow-only index for Wed = 1 -> 100
    assert list(out) == [150, 200, 100]


def test_ets_recovers_exact_weekly_pattern():
    pattern = np.array([10, 20, 30, 40, 50, 60, 70.0])
    days = pd.date_range("2019-07-22", periods=182)  # starts Monday, 26 whole weeks
    hist = frame([(1, d.strftime("%Y-%m-%d"), pattern[d.dayofweek], 0) for d in days])
    cutoff = days[-1]
    f = fut([(1, (cutoff + pd.Timedelta(days=h)).strftime("%Y-%m-%d"), 0) for h in (1, 2, 7, 8)])
    out = ETSWeekly(7, workers=1).forecast(hist, f, STORES, cutoff)
    # cutoff is a Sunday: h=1 Mon 10, h=2 Tue 20, h=7 Sun 70, h=8 Mon 10
    assert out == pytest.approx([10, 20, 70, 10], rel=0.02)
