"""Create pipeline runs, execute them, and persist their outcome.

Used by the API (runs in the background) and by the CLI / benchmark harness
(awaited directly). Every run is recorded in the database, so runs started
from a notebook also appear in the web UI.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlmodel import Session

from automl_agent.agents import AgentManager, PipelineResult, RunContext
from automl_agent.config import Settings, get_settings
from automl_agent.llm import LLMRouter, create_llm
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.events import Stage
from automl_agent.schemas.run import RunStatus
from automl_agent.storage.db import (
    DatasetRecord,
    PlanObservation,
    RunRecord,
    get_engine,
    init_db,
    utcnow,
)

from .event_bus import EventBus, bus

log = logging.getLogger(__name__)
_tasks: set[asyncio.Task] = set()


@dataclass(frozen=True)
class DatasetRef:
    """Plain-value view of a dataset (safe to hand to background tasks)."""

    id: str
    path: Path
    profile: DatasetProfile

    @classmethod
    def from_record(cls, rec: DatasetRecord) -> DatasetRef:
        """Copy the values out of an ORM record."""
        return cls(rec.id, Path(rec.path), DatasetProfile.model_validate(rec.profile))


def events_log_path(settings: Settings, run_id: str) -> Path:
    """Where the JSONL event log of a run is written."""
    return settings.runs_dir / run_id / "events.jsonl"


def run_config(settings: Settings, llm: LLMRouter) -> dict[str, Any]:
    """Snapshot of the settings that define an experimental condition."""
    return {
        "llm_provider": settings.llm_provider,
        "models": llm.models(),
        "agent_fusion": settings.agent_fusion,
        "search_grounding": settings.search_grounding,
        "codegen_mode": settings.codegen_mode,
        "n_plans": settings.n_plans,
        "max_revisions": settings.max_revisions,
        "verification_mode": settings.verification_mode,
        "grounding": {
            "min_rows": settings.grounding_min_rows,
            "growth": settings.grounding_growth,
            "eta": settings.grounding_eta,
            "valid_rows": settings.grounding_valid_rows,
        },
        "budget": {
            "wall_s": settings.budget_wall_s or None,
            "llm_calls": settings.budget_llm_calls or None,
            "tokens": settings.budget_tokens or None,
        },
    }


def new_run(session: Session, dataset_id: str, prompt: str) -> RunRecord:
    """Insert a pending run record."""
    run = RunRecord(id=uuid.uuid4().hex[:12], dataset_id=dataset_id, prompt=prompt)
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def create_run(session: Session, dataset: DatasetRecord, prompt: str) -> RunRecord:
    """Insert a run and start it in the background (API use)."""
    ref = DatasetRef.from_record(dataset)  # copy before commit() expires the ORM object
    run = new_run(session, dataset.id, prompt)
    task = asyncio.create_task(execute_run(run.id, ref, prompt))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return run


def save_observations(settings: Settings, run_id: str, dataset_id: str, rows: list[dict[str, Any]]) -> None:
    """Persist the predicted-vs-observed rows of a run."""
    if not rows:
        return
    allowed = set(PlanObservation.model_fields)
    with Session(get_engine(settings.db_url)) as session:
        for row in rows:
            data = {k: v for k, v in row.items() if k in allowed}
            session.add(PlanObservation(run_id=run_id, dataset_id=dataset_id, **data))
        session.commit()


def _update(settings: Settings, run_id: str, **fields: Any) -> None:
    with Session(get_engine(settings.db_url)) as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            return
        for k, v in fields.items():
            setattr(run, k, v)
        session.add(run)
        session.commit()


async def execute_run(
    run_id: str,
    dataset: DatasetRef,
    prompt: str,
    *,
    settings: Settings | None = None,
    event_bus: EventBus = bus,
) -> PipelineResult | None:
    """Run the full pipeline for an existing run record and store the outcome.

    Returns:
        The pipeline result, or None if the run crashed (the error is stored on the record).
    """
    settings = settings or get_settings()
    init_db(settings.db_url)
    workdir = settings.runs_dir / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    event_bus.attach_log(run_id, events_log_path(settings, run_id))
    _update(settings, run_id, status=RunStatus.running.value)

    llm: LLMRouter | None = None
    try:
        llm = create_llm(settings)
        _update(settings, run_id, config=run_config(settings, llm))
        ctx = RunContext(
            run_id=run_id,
            prompt=prompt,
            dataset_path=dataset.path,
            profile=dataset.profile,
            workdir=workdir,
            settings=settings,
            llm=llm,
            bus=event_bus,
            dataset_id=dataset.id,
        )
        await ctx.emit(Stage.parse, "system", f"Run started with {llm.provider}: {llm.model}", kind="status")
        result = await AgentManager(ctx).run()
        _update(
            settings,
            run_id,
            status=(RunStatus.succeeded if result.success else RunStatus.failed).value,
            finished_at=utcnow(),
            task_spec=result.task_spec.model_dump(mode="json") if result.task_spec else None,
            plan=result.plan.model_dump(mode="json") if result.plan else None,
            metrics={
                **(result.metrics or {}),
                "target_met": result.target_met,
                "attempts": result.attempts,
                "stop_reason": result.stop_reason,
                "budget": result.budget,
            },
            code=result.code,
            error=result.error,
            llm_usage=llm.usage.to_dict(),
        )
        save_observations(settings, run_id, dataset.id, result.observations)
        await ctx.emit(
            Stage.done,
            "system",
            "Run finished successfully" if result.success else f"Run failed: {result.error}",
            kind="status" if result.success else "error",
            payload={"success": result.success, "metrics": result.metrics, "target_met": result.target_met},
        )
        return result
    except Exception as e:  # keep the server alive and report the failure to the UI
        log.exception("run %s crashed", run_id)
        _update(
            settings,
            run_id,
            status=RunStatus.failed.value,
            finished_at=utcnow(),
            error=f"{type(e).__name__}: {e}",
            llm_usage=llm.usage.to_dict() if llm else None,
        )
        await event_bus.publish(
            run_id,
            stage=Stage.done,
            agent="system",
            kind="error",
            message=f"Run crashed: {type(e).__name__}: {e}",
        )
        return None
    finally:
        await event_bus.close(run_id)
