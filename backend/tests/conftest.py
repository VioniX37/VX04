import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

# Isolate tests from any developer .env / workspace before the app is imported.
_TMP = Path(tempfile.mkdtemp(prefix="automl-agent-tests-"))
os.environ["WORKSPACE_DIR"] = str(_TMP / "workspace")
os.environ["LLM_PROVIDER"] = "fake"
os.environ["LLM_MODEL"] = ""
os.environ["MAX_REVISIONS"] = "0"
os.environ["EXEC_TIMEOUT_S"] = "120"

from automl_agent.config import Settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen", REPO_ROOT / "data" / "samples" / "generate_samples.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def sample_csvs(tmp_path_factory) -> dict[str, Path]:
    gen = _load_generator()
    out = tmp_path_factory.mktemp("samples")
    paths = {
        "churn": out / "customer_churn.csv",
        "houses": out / "house_prices.csv",
        "reviews": out / "product_reviews.csv",
    }
    gen.churn(400).to_csv(paths["churn"], index=False)
    gen.houses(400).to_csv(paths["houses"], index=False)
    gen.reviews(500).to_csv(paths["reviews"], index=False)
    return paths


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(
        workspace_dir=tmp_path / "ws", llm_provider="fake", max_revisions=0, n_plans=3, exec_timeout_s=120
    )
    s.ensure_dirs()
    return s
