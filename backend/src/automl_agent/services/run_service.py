"""Starts pipeline runs in the background and persists their outcome."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from sqlmodel import Session

from automl_agent.agents import AgentManager, RunContext
from automl_agent.config import Settings, get_settings
from automl_agent.llm import create_llm
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.events import Stage
from automl_agent.schemas.run import RunStatus
from automl_agent.storage.db import DatasetRecord, RunRecord, get_engine, utcnow

from .event_bus import EventBus, bus

log = logging.getLogger(__name__)
_tasks: set[asyncio.Task] = set()


def events_log_path(settings: Settings, run_id: str) -> Path:
    return settings.runs_dir / run_id / "events.jsonl"


def create_run(session: Session, dataset: DatasetRecord, prompt: str) -> RunRecord:
    # Copy plain values first: commit() expires ORM objects the background task can't reload.
    dataset_path, profile = Path(dataset.path), DatasetProfile.model_validate(dataset.profile)
    run = RunRecord(id=uuid.uuid4().hex[:12], dataset_id=dataset.id, prompt=prompt)
    session.add(run)
    session.commit()
    session.refresh(run)
    task = asyncio.create_task(execute_run(run.id, dataset_path, profile, prompt))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return run


def _update(run_id: str, **fields) -> None:
    with Session(get_engine()) as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            return
        for k, v in fields.items():
            setattr(run, k, v)
        session.add(run)
        session.commit()


async def execute_run(
    run_id: str,
    dataset_path: Path,
    profile: DatasetProfile,
    prompt: str,
    *,
    settings: Settings | None = None,
    event_bus: EventBus = bus,
) -> None:
    settings = settings or get_settings()
    workdir = settings.runs_dir / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    event_bus.attach_log(run_id, events_log_path(settings, run_id))
    _update(run_id, status=RunStatus.running.value)

    llm = None
    try:
        llm = create_llm(settings)
        ctx = RunContext(
            run_id=run_id,
            prompt=prompt,
            dataset_path=dataset_path,
            profile=profile,
            workdir=workdir,
            settings=settings,
            llm=llm,
            bus=event_bus,
        )
        await ctx.emit(
            Stage.parse, "system", f"Run started with LLM {llm.provider}:{llm.model}", kind="status"
        )
        result = await AgentManager(ctx).run()
        _update(
            run_id,
            status=(RunStatus.succeeded if result.success else RunStatus.failed).value,
            finished_at=utcnow(),
            task_spec=result.task_spec.model_dump(mode="json") if result.task_spec else None,
            plan=result.plan.model_dump(mode="json") if result.plan else None,
            metrics={**(result.metrics or {}), "target_met": result.target_met, "attempts": result.attempts},
            code=result.code,
            error=result.error,
            llm_usage=llm.usage.to_dict(),
        )
        await ctx.emit(
            Stage.done,
            "system",
            "Run finished successfully" if result.success else f"Run failed: {result.error}",
            kind="status" if result.success else "error",
            payload={"success": result.success, "metrics": result.metrics, "target_met": result.target_met},
        )
    except Exception as e:  # keep the server alive and report the failure to the UI
        log.exception("run %s crashed", run_id)
        _update(
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
    finally:
        await event_bus.close(run_id)
