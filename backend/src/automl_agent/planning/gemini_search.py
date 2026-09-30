"""Web knowledge retrieval through Gemini's Google Search grounding.

Plays the role of the paper's web/arXiv/Papers-with-Code retrieval in
retrieval-augmented planning, using a single grounded call instead of several
third-party APIs. Results (answer + cited URLs) become one ``KnowledgeItem``.
"""

from __future__ import annotations

import logging

from automl_agent.llm.base import LLMError
from automl_agent.llm.router import LLMRouter
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import TaskSpec

from .retrieval import KnowledgeItem

log = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 2500


def build_query(spec: TaskSpec, profile: DatasetProfile) -> str:
    """Compose the search question from the task and the dataset's shape."""
    domain = f" in the {spec.domain} domain" if spec.domain else ""
    n_cat = sum(c.kind == "categorical" for c in profile.columns)
    n_num = sum(c.kind == "numeric" for c in profile.columns)
    task = spec.task_type.value.replace("_", " ")
    return (
        f"What are the best-performing machine learning approaches for {task}"
        f"{domain} on a dataset with {profile.n_rows:,} rows and {profile.n_cols} columns "
        f"({n_num} numeric, {n_cat} categorical, {len(profile.text_columns)} free-text), "
        f"optimising {spec.metric}? Recommend specific models, preprocessing and hyperparameter ranges "
        "that work on a CPU, and mention pitfalls. Be concise."
    )


class GeminiSearchRetriever:
    """Retriever that asks Gemini with Google Search grounding enabled."""

    def __init__(self, llm: LLMRouter) -> None:
        self.llm = llm

    async def retrieve(
        self, spec: TaskSpec, profile: DatasetProfile, prompt: str, k: int = 5
    ) -> list[KnowledgeItem]:
        """Return at most one knowledge item; failures degrade to no web knowledge."""
        try:
            result = await self.llm.search(build_query(spec, profile))
        except LLMError as e:
            log.warning("search grounding failed, continuing without web knowledge: %s", e)
            return []
        urls = [s["url"] for s in result.sources]
        return [
            KnowledgeItem(
                id="web-search",
                task_types=[spec.task_type.value],
                title="Web search: current best practices",
                content=result.text[:MAX_CONTENT_CHARS],
                tags=["web"],
                source="google-search",
                urls=urls,
            )
        ]
