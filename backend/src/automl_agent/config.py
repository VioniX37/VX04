"""Application settings, loaded from environment variables and `.env` files.

Every field maps to an upper-case environment variable of the same name
(e.g. ``gemini_model_smart`` -> ``GEMINI_MODEL_SMART``). The full reference
lives in ``docs/reference/configuration.md``; a test keeps the two in sync.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent

LLMProvider = Literal["gemini", "fake"]
ModelRole = Literal["smart", "fast"]


class Settings(BaseSettings):
    """Typed configuration for the backend, the pipeline and the experiments."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ LLM (Gemini)
    llm_provider: LLMProvider = Field(
        "gemini", description="`gemini` for real runs, `fake` for the offline deterministic backend."
    )
    gemini_api_key: str | None = Field(
        None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key"),
        description="Google AI Studio API key (`GOOGLE_API_KEY` is accepted as an alias).",
    )
    gemini_model_smart: str = Field(
        "gemini-3.8-flash", description="Model for reasoning-heavy roles: Manager and Operation Agent."
    )
    gemini_model_fast: str = Field(
        "gemini-3.1-flash-lite", description="Model for high-volume roles: Prompt, Data and Model agents."
    )
    gemini_use_vertexai: bool = Field(False, description="Use Vertex AI instead of the Gemini Developer API.")
    google_cloud_project: str | None = Field(None, description="GCP project id (Vertex AI mode only).")
    google_cloud_location: str = Field("global", description="GCP region (Vertex AI mode only).")
    gemini_rpm_smart: int = Field(5, description="Requests-per-minute cap for the smart model.")
    gemini_rpm_fast: int = Field(15, description="Requests-per-minute cap for the fast model.")
    gemini_max_concurrency: int = Field(2, description="Max simultaneous in-flight Gemini requests.")
    gemini_max_retries: int = Field(6, description="Retries on 429/5xx with exponential backoff.")
    llm_temperature: float = Field(0.2, description="Sampling temperature for all agents.")
    llm_cache: bool = Field(True, description="Cache LLM responses on disk (reproducible, saves quota).")
    llm_cache_root: Path | None = Field(
        None,
        description="Cache location (default <workspace>/llm_cache); share it across experiment workspaces.",
    )
    llm_cache_namespace: str = Field(
        "default", description="Cache partition; change it (e.g. per seed) to force fresh responses."
    )
    agent_fusion: bool = Field(
        False, description="Merge the Data and Model agent calls into one call per plan (halves calls)."
    )
    search_grounding: bool = Field(
        True, description="Retrieve web knowledge through Gemini's Google Search grounding tool."
    )

    # ------------------------------------------------------------------ pipeline
    n_plans: int = Field(3, description="Number of candidate plans per planning round.")
    max_revisions: int = Field(2, description="Extra planning rounds when requirements are not met.")
    max_debug_attempts: int = Field(3, description="Operation Agent run/fix attempts per plan.")
    exec_timeout_s: int = Field(1800, description="Hard timeout for one full-data training script.")
    codegen_mode: Literal["llm", "template"] = Field(
        "llm", description="`llm`: Operation Agent edits the template; `template`: run the template as-is."
    )

    verification_mode: Literal["pseudo", "grounded"] = Field(
        "grounded",
        description="`pseudo`: rank plans on LLM-predicted scores (paper-faithful); "
        "`grounded`: successive halving with real runs on growing data subsamples.",
    )
    grounding_min_rows: int = Field(20_000, description="Training rows used at the first grounding rung.")
    grounding_growth: int = Field(4, description="Factor by which training rows grow between rungs.")
    grounding_eta: int = Field(2, description="Keep the best 1/eta of plans after each rung.")
    grounding_valid_rows: int = Field(100_000, description="Validation rows used to score grounding runs.")
    memory_enabled: bool = Field(
        True, description="Experience memory: reuse plans, scores and fixes from past runs on similar data."
    )
    memory_k: int = Field(3, description="Number of similar past runs recalled as planning knowledge.")
    memory_exclude_same_dataset: bool = Field(
        False,
        description="Never recall runs on the same dataset (set true for leave-one-dataset-out evaluation).",
    )
    budget_wall_s: int = Field(0, description="Wall-clock budget per run in seconds (0 = unlimited).")
    budget_llm_calls: int = Field(0, description="LLM call budget per run (0 = unlimited).")
    budget_tokens: int = Field(0, description="LLM token budget per run (0 = unlimited).")
    exec_max_mem_mb: int = Field(
        0, description="Kill a training script above this resident memory (MB); 0 = 80% of system RAM."
    )
    exec_n_jobs: int = Field(-1, description="CPU threads for model training (-1 = all cores).")

    # ------------------------------------------------------------------ data
    max_upload_mb: int = Field(5120, description="Largest accepted browser upload (MB).")
    allow_path_registration: bool = Field(
        True, description="Allow registering datasets by server-side file path (disable on shared servers)."
    )
    split_valid_fraction: float = Field(0.15, description="Share of rows in the validation split.")
    split_test_fraction: float = Field(0.15, description="Share of rows in the held-out test split.")
    split_seed: int = Field(42, description="Seed for the train/valid/test assignment.")
    profile_sample_rows: int = Field(100_000, description="Rows sampled for text/identifier detection.")

    # ------------------------------------------------------------------ storage / server
    workspace_dir: Path = Field(BACKEND_ROOT / "workspace", description="Datasets, runs, database, cache.")
    cors_origins: list[str] = Field(["http://localhost:3000"], description="Origins allowed to call the API.")

    @property
    def datasets_dir(self) -> Path:
        """Directory holding one sub-folder per registered dataset."""
        return self.workspace_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        """Directory holding one sub-folder per pipeline run."""
        return self.workspace_dir / "runs"

    @property
    def llm_cache_dir(self) -> Path:
        """Directory of the on-disk LLM response cache."""
        return (self.llm_cache_root or self.workspace_dir / "llm_cache") / self.llm_cache_namespace

    @property
    def db_url(self) -> str:
        """SQLAlchemy URL of the SQLite database."""
        return f"sqlite:///{(self.workspace_dir / 'automl.db').as_posix()}"

    def model_for(self, role: ModelRole) -> str:
        """Return the Gemini model id configured for an agent role."""
        return self.gemini_model_smart if role == "smart" else self.gemini_model_fast

    def ensure_dirs(self) -> None:
        """Create the workspace directory tree if it does not exist."""
        for d in (self.workspace_dir, self.datasets_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
