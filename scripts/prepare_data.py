"""Convert the raw Kaggle CSVs (read in place, read-only) to typed Parquet in data/processed/ (untracked)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecast import config  # noqa: E402
from demand_forecast.data import convert_to_parquet  # noqa: E402
from demand_forecast.memguard import MemGuard  # noqa: E402


def main():
    mg = MemGuard(config.RESULTS_DIR / "prepare_data_mem.log")
    mg.start_watchdog()
    mg.stage("start")
    train, store = convert_to_parquet()
    mg.stage("converted")
    print(f"raw dir: {config.RAW_DIR}")
    print(f"train: {train.shape}  ->  {config.PROCESSED_TRAIN}")
    print(train.dtypes.to_string())
    print(f"store: {store.shape}  ->  {config.PROCESSED_STORE}")
    print(store.dtypes.to_string())


if __name__ == "__main__":
    main()
