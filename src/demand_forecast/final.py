"""Final training (TRAIN + VALIDATION, through 2015-06-19) and the guarded one-shot TEST evaluation helpers."""
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from demand_forecast import config, gbm
from demand_forecast import model_features as MF
from demand_forecast.backtest import HISTORY_COLUMNS, Split
from demand_forecast.windows import make_windows

FROZEN_TREES = {"full": 1106, "q10": 1500, "q90": 206}  # docs/final_protocol.md section 1


def final_cutoff():
    return make_windows()["validation"][1]  # 2015-06-19, the day before TEST


def test_split():
    a, b = make_windows()["test"]
    return Split("test", final_cutoff(), a, b)


def training_cutoffs():
    """Friday cutoffs whose first target day is on or before the final cutoff."""
    return pd.date_range("2013-04-05", final_cutoff() - pd.Timedelta(days=7), freq="7D")


def build_final_training_table(frame: pd.DataFrame, store: pd.DataFrame, seed=20260101, keep_fraction=0.2) -> pd.DataFrame:
    """Training rows: subsample, Sales > 0, target day <= final cutoff. `frame` must end on or before the final cutoff."""
    fc = final_cutoff()
    if frame["Date"].max() > fc:
        raise ValueError(f"final training frame must end by {fc.date()}")
    g = MF.Grids(frame, store)
    last_t = MF.cutoff_index(g, fc)
    parts = []
    for cd in training_cutoffs():
        ci = MF.cutoff_index(g, cd)
        p = MF.build_cutoff_rows(g, ci)
        keep = MF.keep_flags(len(p["Store"]), ci, seed, keep_fraction)
        need = keep & (p["sales"] > 0) & (p["t_idx"] <= last_t)
        parts.append(pd.DataFrame({k: v[need] for k, v in p.items() if k not in MF.SWP_AUX}))
    tab = pd.concat(parts, ignore_index=True)
    assert tab["t_idx"].max() <= last_t
    return tab


def fit_final(tab: pd.DataFrame, cfg: dict, kind: str):
    import lightgbm as lgb
    feats = MF.features_for()
    objective, alpha = ("regression", None) if kind == "full" else ("quantile", {"q10": 0.1, "q90": 0.9}[kind])
    params = gbm.lgb_params(cfg, objective, alpha)
    mask = np.ones(len(tab), bool)
    ds = lgb.Dataset(gbm.design(tab, mask, feats), gbm.target(tab, mask), feature_name=feats,
                     categorical_feature=[f for f in MF.CATEGORICAL if f in feats])
    return lgb.train(params, ds, num_boost_round=FROZEN_TREES[kind])


def blank_sales_after(frame: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Copy of frame with Sales after the cutoff set to 0, so feature building cannot see them."""
    f = frame.copy()
    f.loc[f["Date"] > pd.Timestamp(cutoff), "Sales"] = 0
    return f


def load_frame_including_test(path=None) -> pd.DataFrame:
    """ONLY for the one-shot TEST script. Explicit column list (no unavailable columns)."""
    return pd.read_parquet(path or config.PROCESSED_TRAIN, columns=HISTORY_COLUMNS)


def sha256_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_and_write_record(path, record: dict, override: str | None = None) -> dict:
    """Refuse a second TEST run unless an override reason is given; the override is appended to the record."""
    from pathlib import Path
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if not override:
            raise RuntimeError(f"TEST was already evaluated at {old.get('timestamp')}; refusing to run again "
                               "(pass --override '<reason>' to document a deliberate re-run)")
        old.setdefault("overrides", []).append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                                                "reason": override, "new_run": record})
        path.write_text(json.dumps(old, indent=2) + "\n")
        return old
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record
