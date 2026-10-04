"""API schemas for pipeline runs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    """Lifecycle of a run."""

    pending = "pending"
    running = "running"
    awaiting_input = "awaiting_input"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class ApprovalMode(StrEnum):
    """Approval gate for pipeline decisions."""

    auto = "auto"
    plans = "plans"
    plans_code = "plans+code"


class PlanApprovalAction(StrEnum):
    """Action taken by the user on a paused run."""

    approve = "approve"
    pick = "pick"
    edit = "edit"


class PlanApprovalRequest(BaseModel):
    """Payload to approve, pick, or edit a plan for a paused run."""

    action: PlanApprovalAction = PlanApprovalAction.approve
    plan_id: str | None = None
    edited_plan: dict[str, Any] | None = None
    edited_code: str | None = None
    feedback: str | None = None


class RunCreate(BaseModel):
    """Request body to start a run."""

    dataset_id: str
    prompt: str = Field(min_length=3)
    approval: ApprovalMode = Field(default=ApprovalMode.auto)


class RunOut(BaseModel):
    """API representation of a run and its outcome."""

    id: str
    dataset_id: str
    prompt: str
    status: RunStatus
    approval: ApprovalMode = ApprovalMode.auto
    human_override: bool = False
    human_decision: dict[str, Any] | None = None
    created_at: datetime
    finished_at: datetime | None = None
    task_spec: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    code: str | None = None
    error: str | None = None
    llm_usage: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
