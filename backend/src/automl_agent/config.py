from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent

LLMProvider = Literal["fake", "openai", "anthropic", "gemini", "groq", "ollama", "openai_compatible"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: LLMProvider = "fake"
    llm_model: str = ""
    llm_base_url: str | None = None
    llm_temperature: float = 0.2
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None

    # Pipeline
    n_plans: int = 3
    max_revisions: int = 2
    max_debug_attempts: int = 3
    exec_timeout_s: int = 600
    codegen_mode: Literal["llm", "template"] = "llm"

    # Storage / server
    workspace_dir: Path = BACKEND_ROOT / "workspace"
    cors_origins: list[str] = ["http://localhost:3000"]

    @property
    def datasets_dir(self) -> Path:
        return self.workspace_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.workspace_dir / "runs"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.workspace_dir / 'automl.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (self.workspace_dir, self.datasets_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
