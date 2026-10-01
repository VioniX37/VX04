"""Recall past runs on similar datasets as planning knowledge."""

from __future__ import annotations

import json

from automl_agent.config import Settings
from automl_agent.planning.retrieval import KnowledgeItem
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import TaskSpec
from automl_agent.storage.db import ExperienceRecord

from .meta_features import dataset_fingerprint, distance, meta_features
from .store import ExperienceStore


def _fmt(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.4f}"


def describe(record: ExperienceRecord, dist: float) -> KnowledgeItem:
    """Render one past run as a knowledge item (text for the LLM, structured data for tools)."""
    best = record.best_plan or {}
    lines = []
    if best:
        hp = json.dumps(best.get("hyperparameters") or {})
        outcome = "met the user's target" if record.target_met else "did not meet the target"
        lines.append(
            f"Best plan: {best.get('model_family')} with hyperparameters {hp} scored "
            f"{record.metric}={_fmt(record.best_score)} on the held-out test split ({outcome})."
        )
        if best.get("preprocessing"):
            lines.append("Its preprocessing: " + "; ".join(best["preprocessing"][:6]) + ".")
    tried = [p for p in record.plans if p.get("observed_score") is not None]
    if tried:
        ranked = sorted(tried, key=lambda p: p["observed_score"], reverse=record.higher_is_better)
        lines.append(
            "Observed validation scores of candidates: "
            + ", ".join(f"{p['model_family']}={_fmt(p['observed_score'])}" for p in ranked[:5])
            + "."
        )
    failed = sorted({p["model_family"] for p in record.plans if p.get("ok") is False})
    if failed:
        lines.append("Failed to run: " + ", ".join(failed) + ".")
    return KnowledgeItem(
        id=f"memory-{record.run_id}",
        task_types=[record.task_type],
        title=f"Past run on a similar dataset ({record.n_rows:,} rows, similarity distance {dist:.2f})",
        content=" ".join(lines) or "No successful plan recorded.",
        tags=["memory", "experience"],
        source=f"memory:{record.run_id}",
        data={
            "best_model_family": best.get("model_family"),
            "best_hyperparameters": best.get("hyperparameters") or {},
            "best_score": record.best_score,
            "distance": round(dist, 4),
            "failed_families": failed,
        },
    )


class MemoryRetriever:
    """k-nearest past runs of the same task type, by meta-feature distance."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = ExperienceStore(settings)

    async def retrieve(
        self, spec: TaskSpec, profile: DatasetProfile, prompt: str, k: int = 5
    ) -> list[KnowledgeItem]:
        """Return up to ``MEMORY_K`` knowledge items describing the most similar past runs."""
        exclude = dataset_fingerprint(profile) if self.settings.memory_exclude_same_dataset else None
        records = self.store.for_task(spec.task_type.value, exclude_fingerprint=exclude)
        if not records:
            return []
        query = meta_features(profile, spec)
        candidates = [(distance(query, r.meta), r) for r in records if r.best_plan or r.plans]
        # Nearest first; equally similar runs (e.g. repeats on the same dataset) -> most recent first.
        candidates.sort(key=lambda pair: (pair[0], -pair[1].created_at.timestamp()))
        return [describe(r, d) for d, r in candidates[: self.settings.memory_k]]
