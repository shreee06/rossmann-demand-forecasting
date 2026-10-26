import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_matches_the_result_files():
    spec = importlib.util.spec_from_file_location("make_readme", ROOT / "scripts" / "make_readme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "README.md").read_text() == mod.render(), "README.md drifted from evals/results; run scripts/make_readme.py"
