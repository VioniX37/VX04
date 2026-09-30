from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl_agent.config import Settings
from automl_agent.llm.router import LLMRouter
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.events import EventKind, Stage
from automl_agent.services.event_bus import EventBus


@dataclass
class RunContext:
    """Everything an agent needs for one pipeline run."""

    run_id: str
    prompt: str
    dataset_path: Path
    profile: DatasetProfile
    workdir: Path
    settings: Settings
    llm: LLMRouter
    bus: EventBus
    state: dict[str, Any] = field(default_factory=dict)  # scratch space for extensions

    async def emit(
        self,
        stage: Stage,
        agent: str,
        message: str,
        *,
        kind: EventKind = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self.bus.publish(
            self.run_id, stage=stage, agent=agent, message=message, kind=kind, payload=payload
        )
