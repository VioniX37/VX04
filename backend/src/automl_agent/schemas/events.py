from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Stage(StrEnum):
    """Pipeline stages, in order (mirrors Fig. 2 of the paper)."""

    parse = "parse"  # Prompt Agent: request -> TaskSpec
    verify_request = "verify_request"  # request verification
    prepare = "prepare"  # Parquet conversion + fixed train/valid/test split
    retrieve = "retrieve"  # knowledge retrieval
    plan = "plan"  # retrieval-augmented planning (N plans)
    execute_plans = "execute_plans"  # decomposition + Data/Model agents in parallel
    select = "select"  # execution verification, pick best plan
    implement = "implement"  # Operation Agent: code + run + debug
    verify_impl = "verify_impl"  # implementation verification
    done = "done"


EventKind = Literal["status", "info", "llm", "artifact", "warning", "error"]


class AgentEvent(BaseModel):
    seq: int
    run_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: Stage
    agent: str
    kind: EventKind = "info"
    message: str
    payload: dict[str, Any] | None = None
