"""In-process pub/sub of AgentEvents per run, with replay (drives the SSE stream)."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from automl_agent.schemas.events import AgentEvent, EventKind, Stage


class EventBus:
    def __init__(self) -> None:
        self._events: dict[str, list[AgentEvent]] = defaultdict(list)
        self._conds: dict[str, asyncio.Condition] = {}
        self._closed: set[str] = set()
        self._logs: dict[str, Path] = {}

    def _cond(self, run_id: str) -> asyncio.Condition:
        if run_id not in self._conds:
            self._conds[run_id] = asyncio.Condition()
        return self._conds[run_id]

    def attach_log(self, run_id: str, path: Path) -> None:
        """Also append every event of this run to a JSONL file (survives restarts)."""
        self._logs[run_id] = path

    async def publish(
        self,
        run_id: str,
        *,
        stage: Stage,
        agent: str,
        message: str,
        kind: EventKind = "info",
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        cond = self._cond(run_id)
        async with cond:
            event = AgentEvent(
                seq=len(self._events[run_id]) + 1,
                run_id=run_id,
                stage=stage,
                agent=agent,
                kind=kind,
                message=message,
                payload=payload,
            )
            self._events[run_id].append(event)
            if log := self._logs.get(run_id):
                with log.open("a", encoding="utf-8") as f:
                    f.write(event.model_dump_json() + "\n")
            cond.notify_all()
        return event

    async def close(self, run_id: str) -> None:
        cond = self._cond(run_id)
        async with cond:
            self._closed.add(run_id)
            cond.notify_all()

    def history(self, run_id: str) -> list[AgentEvent]:
        return list(self._events.get(run_id, []))

    def load_history(self, run_id: str, path: Path) -> None:
        """Rehydrate a finished run's events from its JSONL log (e.g. after a server restart)."""
        if run_id in self._events or not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        self._events[run_id] = [AgentEvent.model_validate_json(line) for line in lines if line.strip()]
        self._closed.add(run_id)

    async def subscribe(self, run_id: str, after: int = 0) -> AsyncIterator[AgentEvent]:
        """Yield events with seq > `after`, then live ones, until the run is closed."""
        cond = self._cond(run_id)
        while True:
            async with cond:
                while len(self._events[run_id]) <= after and run_id not in self._closed:
                    await cond.wait()
                pending = self._events[run_id][after:]
                closed = run_id in self._closed
            for event in pending:
                yield event
                after = event.seq
            if closed and not pending:
                return


bus = EventBus()
