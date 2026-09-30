"""Plans, decomposed sub-tasks and agent results (paper §3.2-3.4)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    id: str = Field(description="Short unique id, e.g. 'p1'")
    title: str
    rationale: str = Field(description="Why this plan should work, citing retrieved knowledge")
    preprocessing: list[str] = Field(description="Ordered data-processing steps")
    model_family: str = Field(description="One of the supported model families")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    validation: str = Field(default="stratified 80/20 hold-out", description="Validation strategy")


class PlanSet(BaseModel):
    plans: list[Plan]


class SubTask(BaseModel):
    id: str
    agent: Literal["data", "model"]
    instruction: str


class DataAgentResult(BaseModel):
    """Data Agent's (pseudo-)execution of the data sub-tasks of a plan."""

    summary: str
    steps: list[str] = Field(description="Concrete preprocessing steps to implement")
    risks: list[str] = Field(default_factory=list, description="Data issues that could hurt the model")


class ModelAgentResult(BaseModel):
    """Model Agent's (pseudo-)execution of the model sub-tasks of a plan."""

    summary: str
    model_family: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    predicted_score: float = Field(description="Expected value of the task metric on hold-out data")
    predicted_train_time_s: float = Field(default=60.0)


class PlanAnalysis(BaseModel):
    """Data and Model agent results produced in a single fused call (``AGENT_FUSION=true``)."""

    data: DataAgentResult
    model: ModelAgentResult


class Observation(BaseModel):
    """A real training run of a plan at one fidelity (grounded verification)."""

    fidelity_rows: int = Field(description="Training rows actually used")
    score: float | None = Field(description="Validation score (None if the run failed)")
    ok: bool
    duration_s: float
    error: str | None = None


class PlanEvaluation(BaseModel):
    """A plan with its agents' pseudo-execution results and, if grounded, real observations."""

    plan: Plan
    data: DataAgentResult
    model: ModelAgentResult
    rank: int | None = None
    observations: list[Observation] = Field(default_factory=list)

    @property
    def last_observation(self) -> Observation | None:
        """Observation at the highest fidelity this plan reached."""
        return self.observations[-1] if self.observations else None


class CodeDraft(BaseModel):
    code: str = Field(description="Complete runnable Python script")
    explanation: str = ""


class ExecutionResult(BaseModel):
    ok: bool
    returncode: int | None
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] | None = None
    timed_out: bool = False
    memory_exceeded: bool = False
    peak_memory_mb: float | None = None
