"""Write the fixed TRAIN / VALIDATION / TEST windows and backtest folds to evals/results/windows.json.
Only dates and row counts are computed here; no sales statistics, no model."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecast import config  # noqa: E402
from demand_forecast.windows import backtest_folds, make_windows  # noqa: E402


def d(x):
    return str(pd.Timestamp(x).date())


def main():
    train = pd.read_parquet(config.PROCESSED_TRAIN, columns=["Date", "Store"])
    assert d(train["Date"].min()) == config.DATA_START and d(train["Date"].max()) == config.DATA_END, \
        "config DATA_START/DATA_END do not match the data"
    w = make_windows()

    def rows(a, b):
        return int(train["Date"].between(a, b).sum())

    out = {"unit": "calendar days", "data_start": config.DATA_START, "data_end": config.DATA_END,
           "test_days": config.TEST_DAYS, "validation_days": config.VAL_DAYS, "windows": {}}
    for name, (a, b) in w.items():
        out["windows"][name] = {"start": d(a), "end": d(b), "n_days": (b - a).days + 1, "n_rows": rows(a, b)}
    tr_a, tr_b = w["train"]
    folds = backtest_folds(tr_a, tr_b)
    out["backtest"] = {
        "scheme": "rolling origin, expanding train window (dates <= cutoff), horizon forecast right after cutoff",
        "horizon_days": config.BACKTEST_HORIZON_DAYS, "step_days": config.BACKTEST_STEP_DAYS,
        "span_days": config.BACKTEST_SPAN_DAYS, "n_folds": len(folds),
        "folds": [{"fold": f["fold"], "cutoff": d(f["cutoff"]), "horizon_start": d(f["horizon_start"]),
                   "horizon_end": d(f["horizon_end"]), "n_train_rows": int((train["Date"] <= f["cutoff"]).sum()),
                   "n_horizon_rows": rows(f["horizon_start"], f["horizon_end"])} for f in folds],
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "windows.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
