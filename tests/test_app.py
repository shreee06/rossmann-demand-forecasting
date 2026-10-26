from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_app_renders_aggregate_results_without_errors():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.dataframe) >= 3  # TEST, VALIDATION, backtest comparison tables (+ intervals)
