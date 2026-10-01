"""Knowledge retrieval for retrieval-augmented planning (paper §3.2).

The paper retrieves from web search, papers and model hubs. We start with a
curated local knowledge base ranked by simple lexical overlap; additional
retrievers (web, Hugging Face Hub, arXiv) can implement `Retriever` and be
combined in `retrieve_all`.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import TaskSpec

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
_WORD = re.compile(r"[a-z0-9]+")


class KnowledgeItem(BaseModel):
    """A piece of planning knowledge and where it came from."""

    id: str
    task_types: list[str]
    title: str
    content: str
    tags: list[str] = []
    source: str = "local-kb"
    urls: list[str] = []
    data: dict[str, Any] | None = None  # structured payload (e.g. from experience memory)


class Retriever(Protocol):
    """Anything that can return knowledge for a task (local KB, web search, memory, ...)."""

    async def retrieve(
        self, spec: TaskSpec, profile: DatasetProfile, prompt: str, k: int
    ) -> list[KnowledgeItem]:
        """Return up to `k` knowledge items relevant to the task."""
        ...


@cache
def _load_kb() -> tuple[KnowledgeItem, ...]:
    items: list[KnowledgeItem] = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.json")):
        items.extend(KnowledgeItem.model_validate(x) for x in json.loads(f.read_text(encoding="utf-8")))
    return tuple(items)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class LocalKnowledgeRetriever:
    """Ranks the curated local knowledge base by lexical overlap with the task."""

    async def retrieve(
        self, spec: TaskSpec, profile: DatasetProfile, prompt: str, k: int = 5
    ) -> list[KnowledgeItem]:
        """Return the `k` best-matching entries of the curated knowledge base."""
        size = "small" if profile.n_rows < 5000 else "large"
        query = _tokens(
            f"{prompt} {spec.metric} {spec.notes} {size} {spec.task_type.value.replace('_', ' ')}"
        )
        if spec.task_type.value.startswith("tabular") and any(c.n_missing for c in profile.columns):
            query |= {"missing", "imputation"}
        scored = []
        for item in _load_kb():
            if spec.task_type.value not in item.task_types:
                continue
            overlap = len(query & (_tokens(item.title + " " + item.content) | set(item.tags)))
            scored.append((overlap + 2 * len(query & set(item.tags)), item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]


async def retrieve_all(
    retrievers: list[Retriever], spec: TaskSpec, profile: DatasetProfile, prompt: str, k: int = 5
) -> list[KnowledgeItem]:
    """Query every retriever and merge the results (first occurrence of an id wins)."""
    seen: dict[str, KnowledgeItem] = {}
    for r in retrievers:
        for item in await r.retrieve(spec, profile, prompt, k):
            seen.setdefault(item.id, item)
    return list(seen.values())
