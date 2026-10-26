"""Forecast metrics (RMSPE, WAPE, MASE, bias) built from additive per-row sufficient statistics.

Additive statistics make store-clustered bootstrapping cheap: sum the statistics of the resampled stores, then
turn the sums into metrics. All definitions are fixed in docs/forecast_protocol.md.
"""
import numpy as np
import pandas as pd

METRICS = ["rmspe", "wape", "mase", "bias"]
# columns of the statistic vector
STATS = ["n", "sq_pct", "abs_err", "y", "err", "abs_scaled", "n_scaled"]


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Needs y (actual > 0), yhat and scale (store in-sample seasonal-naive MAE; NaN/0 = no MASE for the row)."""
    e = df["yhat"].to_numpy(float) - df["y"].to_numpy(float)
    y = df["y"].to_numpy(float)
    sc = df["scale"].to_numpy(float)
    ok = np.isfinite(sc) & (sc > 0)
    out = df.copy()
    out["n"] = 1.0
    out["sq_pct"] = (e / y) ** 2
    out["abs_err"] = np.abs(e)
    out["err"] = e
    out["abs_scaled"] = np.where(ok, np.abs(e) / np.where(ok, sc, 1.0), 0.0)
    out["n_scaled"] = ok.astype(float)
    return out


def metrics_from_stats(S: np.ndarray) -> np.ndarray:
    """S[..., 7] (order STATS) -> [..., 4] (order METRICS). Empty groups give NaN."""
    S = np.asarray(S, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rmspe = np.sqrt(S[..., 1] / S[..., 0])
        wape = S[..., 2] / S[..., 3]
        mase = S[..., 5] / S[..., 6]
        bias = S[..., 4] / S[..., 3]
    return np.stack([rmspe, wape, mase, bias], axis=-1)


def compute_metrics(y, yhat, scale=None) -> dict:
    df = pd.DataFrame({"y": np.asarray(y, float), "yhat": np.asarray(yhat, float),
                       "scale": np.nan if scale is None else np.asarray(scale, float)})
    S = add_error_columns(df)[STATS].sum().to_numpy()
    return dict(zip(METRICS, metrics_from_stats(S).tolist()))


def seasonal_naive_scale(history: pd.DataFrame, period: int = 7) -> pd.Series:
    """Per-store mean |y_t - y_{t-period}| over history pairs where both days are open with Sales > 0."""
    obs = history.loc[(history["Open"] == 1) & (history["Sales"] > 0), ["Store", "Date", "Sales"]]
    lag = obs.assign(Date=obs["Date"] + pd.Timedelta(days=period)).rename(columns={"Sales": "lag"})
    m = obs.merge(lag, on=["Store", "Date"])
    return (m["Sales"] - m["lag"]).abs().groupby(m["Store"]).mean()
