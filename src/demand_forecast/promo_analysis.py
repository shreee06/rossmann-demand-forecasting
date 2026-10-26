"""Promotion analysis helpers: high-dimensional fixed-effects regression with store-clustered errors.

ASSOCIATION only: promo timing was chosen by Rossmann; nothing here is a causal effect.
"""
import numpy as np


def codes(values):
    """Integer codes 0..G-1 for any array-like (tuples allowed via a structured key)."""
    uniq, inv = np.unique(values, return_inverse=True)
    return inv.astype(np.int64), len(uniq)


def demean(M, fe, tol=1e-10, max_iter=500):
    """Alternating projections: remove the group means of every fixed-effect factor from the columns of M.
    fe: list of (codes, n_groups). Converges to the residual of M on all fixed-effect dummies."""
    R = np.array(M, dtype=np.float64, copy=True)
    for _ in range(max_iter):
        worst = 0.0
        for c, g in fe:
            cnt = np.bincount(c, minlength=g).astype(np.float64)
            for j in range(R.shape[1]):
                mean = np.bincount(c, weights=R[:, j], minlength=g) / np.maximum(cnt, 1)
                R[:, j] -= mean[c]
                worst = max(worst, float(np.abs(mean).max()))
        if worst < tol:
            break
    return R


def fe_regression(y, X, fe, cluster):
    """OLS of y on X with fixed effects (Frisch-Waugh), cluster-robust SEs (clusters = `cluster` codes; factor G/(G-1)).
    Returns dict with coef, se, lo, hi per column of X, plus n and number of clusters."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    M = demean(np.column_stack([y, X]), fe)
    yt, Xt = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.inv(Xt.T @ Xt)
    beta = XtX_inv @ (Xt.T @ yt)
    e = yt - Xt @ beta
    cc, G = codes(cluster)
    scores = np.zeros((G, Xt.shape[1]))
    for j in range(Xt.shape[1]):
        scores[:, j] = np.bincount(cc, weights=Xt[:, j] * e, minlength=G)
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv * (G / (G - 1))
    se = np.sqrt(np.diag(V))
    return {"coef": beta, "se": se, "lo": beta - 1.96 * se, "hi": beta + 1.96 * se, "n": int(len(yt)), "n_clusters": int(G)}
