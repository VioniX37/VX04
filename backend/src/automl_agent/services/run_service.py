"""Create pipeline runs, execute them, and persist their outcome.

Used by the API (runs in the background) and by the CLI / benchmark harness
(awaited directly). Every run is recorded in the database, so runs started
from a notebook also appear in the web UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
from sqlmodel import Session, select

from automl_agent.agents import AgentManager, PipelineResult, RunContext
from automl_agent.config import Settings, get_settings
from automl_agent.execution.sandbox import kill_run_processes
from automl_agent.extensions.approval import cancel_pending_approval, resolve_approval
from automl_agent.llm import LLMRouter, create_llm
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.events import AgentEvent, Stage
from automl_agent.schemas.run import PlanApprovalRequest, RunStatus
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
_active_tasks: dict[str, asyncio.Task] = {}


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
        "pid": os.getpid(),  # owning process; used to detect runs orphaned by a restart
        "llm_provider": settings.llm_provider,
        "models": llm.models(),
        "agent_fusion": settings.agent_fusion,
        "search_grounding": settings.search_grounding,
        "codegen_mode": settings.codegen_mode,
        "n_plans": settings.n_plans,
        "max_revisions": settings.max_revisions,
        "verification_mode": settings.verification_mode,
        "memory": {"enabled": settings.memory_enabled, "k": settings.memory_k},
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


INTERRUPTED_MESSAGE = (
    "Interrupted: the process running this pipeline stopped (for example, the server restarted). "
    "Start the run again; cached LLM answers make the repeat cheap."
)


def recover_interrupted_runs(settings: Settings) -> list[str]:
    """Mark runs whose owning process is gone as failed, and close their event logs.

    Runs started by another live process (e.g. a CLI run while the API restarts) are left alone.

    Returns:
        Ids of the runs that were marked as interrupted.
    """
    interrupted: list[str] = []
    with Session(get_engine(settings.db_url)) as session:
        unfinished = [RunStatus.pending.value, RunStatus.running.value]
        active = select(RunRecord).where(RunRecord.status.in_(unfinished))
        for run in session.exec(active).all():
            pid = (run.config or {}).get("pid")
            if pid and pid != os.getpid() and psutil.pid_exists(pid):
                continue
            run.status = RunStatus.failed.value
            run.finished_at = utcnow()
            run.error = INTERRUPTED_MESSAGE
            session.add(run)
            interrupted.append(run.id)
            log_path = events_log_path(settings, run.id)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            seq = sum(1 for _ in log_path.open(encoding="utf-8")) + 1 if log_path.exists() else 1
            event = AgentEvent(
                seq=seq,
                run_id=run.id,
                stage=Stage.done,
                agent="system",
                kind="error",
                message=INTERRUPTED_MESSAGE,
            )
            with log_path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        session.commit()
    if interrupted:
        log.warning("Marked %d interrupted run(s) as failed: %s", len(interrupted), ", ".join(interrupted))
    return interrupted


def new_run(session: Session, dataset_id: str, prompt: str, approval: str = "auto") -> RunRecord:
    """Insert a pending run record."""
    run = RunRecord(id=uuid.uuid4().hex[:12], dataset_id=dataset_id, prompt=prompt, approval=approval)
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def create_run(session: Session, dataset: DatasetRecord, prompt: str, approval: str = "auto") -> RunRecord:
    """Insert a run and start it in the background (API use)."""
    ref = DatasetRef.from_record(dataset)  # copy before commit() expires the ORM object
    run = new_run(session, dataset.id, prompt, approval=approval)
    run_id = run.id
    task = asyncio.create_task(execute_run(run_id, ref, prompt, approval=approval))
    _active_tasks[run_id] = task
    _tasks.add(task)
    task.add_done_callback(lambda t: _active_tasks.pop(run_id, None))
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
    approval: str = "auto",
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
    _update(settings, run_id, status=RunStatus.running.value, config={"pid": os.getpid()})

    if approval == "auto":
        with Session(get_engine(settings.db_url)) as session:
            rec = session.get(RunRecord, run_id)
            if rec and rec.approval:
                approval = rec.approval

    llm: LLMRouter | None = None
    ctx: RunContext | None = None
    try:
        current_task = asyncio.current_task()
        if current_task is not None:
            _active_tasks[run_id] = current_task

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
            approval=approval,
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
            human_override=ctx.state.get("human_override", False),
            human_decision=ctx.state.get("human_decision"),
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
    except asyncio.CancelledError:
        log.info("run %s was cancelled", run_id)
        kill_run_processes(run_id)
        if ctx and ctx.observations:
            save_observations(settings, run_id, dataset.id, ctx.observations)
        _update(
            settings,
            run_id,
            status=RunStatus.cancelled.value,
            finished_at=utcnow(),
            error="Cancelled by user",
            llm_usage=llm.usage.to_dict() if llm else None,
        )
        await event_bus.publish(
            run_id,
            stage=Stage.done,
            agent="system",
            kind="warning",
            message="Run cancelled",
            payload={"event": "run_cancelled", "cancelled": True, "success": False},
        )
        raise
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
        _active_tasks.pop(run_id, None)
        await event_bus.close(run_id)


async def cancel_run(run_id: str, settings: Settings | None = None) -> RunRecord | None:
    """Cancel an active or paused run: kills sandbox processes, stops task, and persists status."""
    settings = settings or get_settings()
    kill_run_processes(run_id)
    cancel_pending_approval(run_id)

    task = _active_tasks.get(run_id)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    with Session(get_engine(settings.db_url)) as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            return None
        if run.status not in (RunStatus.succeeded.value, RunStatus.failed.value, RunStatus.cancelled.value):
            run.status = RunStatus.cancelled.value
            run.finished_at = utcnow()
            run.error = "Cancelled by user"
            session.add(run)
            session.commit()
            session.refresh(run)
            await bus.publish(
                run_id,
                stage=Stage.done,
                agent="system",
                kind="warning",
                message="Run cancelled",
                payload={"event": "run_cancelled", "cancelled": True, "success": False},
            )
            await bus.close(run_id)
        return run


async def approve_run(
    run_id: str,
    request: PlanApprovalRequest,
    settings: Settings | None = None,
) -> RunRecord | None:
    """Approve, pick, or edit a plan for a run waiting in awaiting_input."""
    settings = settings or get_settings()
    with Session(get_engine(settings.db_url)) as session:
        run = session.get(RunRecord, run_id)
        if run is None or run.status != RunStatus.awaiting_input.value:
            return None

    if resolve_approval(run_id, request):
        await asyncio.sleep(0.02)  # yield control to let resumed task update status and DB
        with Session(get_engine(settings.db_url)) as session:
            return session.get(RunRecord, run_id)

    # Recovery scenario (e.g. backend restarted while run was awaiting input)
    workdir = settings.runs_dir / run_id
    pause_file = workdir / "pause_state.json"
    if pause_file.exists():
        try:
            data = json.loads(pause_file.read_text(encoding="utf-8"))
            data["decision"] = request.model_dump(mode="json")
            pause_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            log.exception("Failed updating pause_state.json for run %s", run_id)

    with Session(get_engine(settings.db_url)) as session:
        run_record = session.get(RunRecord, run_id)
        if run_record is None:
            return None
        dataset_record = session.get(DatasetRecord, run_record.dataset_id)
        if dataset_record is None:
            return None
        ref = DatasetRef.from_record(dataset_record)
        run_record.status = RunStatus.running.value
        session.add(run_record)
        session.commit()
        session.refresh(run_record)

    task = asyncio.create_task(
        execute_run(
            run_record.id,
            ref,
            run_record.prompt,
            approval=run_record.approval,
            settings=settings,
        )
    )
    _active_tasks[run_record.id] = task
    _tasks.add(task)
    task.add_done_callback(lambda t: _active_tasks.pop(run_record.id, None))
    task.add_done_callback(_tasks.discard)
    return run_record
