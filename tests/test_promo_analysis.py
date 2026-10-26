import numpy as np
import pytest

from demand_forecast import promo_analysis as PA


def panel(noise=0.0, beta=0.2, seed=0, n_stores=40, n_days=60):
    rng = np.random.default_rng(seed)
    store = np.repeat(np.arange(n_stores), n_days)
    day = np.tile(np.arange(n_days), n_stores)
    dow = day % 7
    promo = (rng.random(len(day)) < 0.4).astype(float)
    y = beta * promo + rng.normal(0, 1, n_stores)[store] + np.array([0, .1, .2, .3, .1, -.2, .4])[dow] + noise * rng.standard_normal(len(day))
    return y, promo, store, dow


def test_fe_regression_recovers_known_coefficient_without_noise():
    y, promo, store, dow = panel(noise=0.0)
    fe = [PA.codes(store), PA.codes(dow)]
    r = PA.fe_regression(y, promo, fe, store)
    assert r["coef"][0] == pytest.approx(0.2, abs=1e-8)  # exact: y is linear in promo + store + dow effects
    assert r["n_clusters"] == 40


def test_fe_regression_ci_covers_truth_and_matches_dummy_ols():
    y, promo, store, dow = panel(noise=0.5, seed=3)
    fe = [PA.codes(store), PA.codes(dow)]
    r = PA.fe_regression(y, promo, fe, store)
    # same coefficient as OLS with explicit dummies
    D = np.column_stack([promo] + [(store == s).astype(float) for s in range(40)] + [(dow == d).astype(float) for d in range(1, 7)])
    beta = np.linalg.lstsq(D, y, rcond=None)[0][0]
    assert r["coef"][0] == pytest.approx(beta, abs=1e-7)
    assert r["lo"][0] < 0.2 < r["hi"][0] and r["se"][0] > 0


def test_demean_removes_group_means():
    rng = np.random.default_rng(1)
    g1, g2 = rng.integers(0, 5, 200), rng.integers(0, 7, 200)
    M = PA.demean(rng.standard_normal((200, 2)), [PA.codes(g1), PA.codes(g2)])
    for g in (g1, g2):
        for k in np.unique(g):
            assert abs(M[g == k, 0].mean()) < 1e-7
