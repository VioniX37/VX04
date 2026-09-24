from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from automl_agent.config import get_settings
from automl_agent.schemas.events import AgentEvent
from automl_agent.schemas.run import RunCreate, RunOut
from automl_agent.services.event_bus import bus
from automl_agent.services.run_service import create_run, events_log_path
from automl_agent.storage.db import DatasetRecord, RunRecord, get_session

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunOut, status_code=201)
async def start_run(body: RunCreate, session: Session = Depends(get_session)) -> RunRecord:
    dataset = session.get(DatasetRecord, body.dataset_id)
    if dataset is None:
        raise HTTPException(404, "Dataset not found")
    return create_run(session, dataset, body.prompt)


@router.get("", response_model=list[RunOut])
def list_runs(session: Session = Depends(get_session)) -> list[RunRecord]:
    return list(session.exec(select(RunRecord).order_by(RunRecord.created_at.desc())).all())


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: str, session: Session = Depends(get_session)) -> RunRecord:
    run = session.get(RunRecord, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


def _ensure_history(run: RunRecord) -> None:
    if run.status in ("succeeded", "failed"):
        bus.load_history(run.id, events_log_path(get_settings(), run.id))


@router.get("/{run_id}/events/history", response_model=list[AgentEvent])
def run_events(run_id: str, session: Session = Depends(get_session)) -> list[AgentEvent]:
    run = get_run(run_id, session)
    _ensure_history(run)
    return bus.history(run_id)


@router.get("/{run_id}/events")
async def stream_events(
    run_id: str, after: int = Query(0, ge=0), session: Session = Depends(get_session)
) -> EventSourceResponse:
    """Server-Sent Events: replays events with seq > `after`, then streams live ones."""
    run = get_run(run_id, session)
    _ensure_history(run)

    async def gen():
        async for event in bus.subscribe(run_id, after=after):
            yield {"event": "agent_event", "id": str(event.seq), "data": event.model_dump_json()}
        yield {"event": "end", "data": "{}"}

    return EventSourceResponse(gen())
