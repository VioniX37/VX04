"""Human-in-the-loop plan and code approval hooks."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from sqlmodel import Session

from automl_agent.extensions import PipelineHooks
from automl_agent.schemas.events import Stage
from automl_agent.schemas.plan import Plan, PlanEvaluation
from automl_agent.schemas.run import PlanApprovalAction, PlanApprovalRequest, RunStatus
from automl_agent.storage.db import RunRecord, get_engine

if TYPE_CHECKING:
    from automl_agent.agents.context import RunContext
    from automl_agent.config import Settings

log = logging.getLogger(__name__)

_pending_approvals: dict[str, asyncio.Future[PlanApprovalRequest]] = {}


def resolve_approval(run_id: str, request: PlanApprovalRequest) -> bool:
    """Resolve a waiting approval future for run_id."""
    future = _pending_approvals.get(run_id)
    if future is not None and not future.done():
        future.set_result(request)
        return True
    return False


def cancel_pending_approval(run_id: str) -> None:
    """Cancel and remove any waiting approval future for run_id."""
    future = _pending_approvals.pop(run_id, None)
    if future is not None and not future.done():
        future.cancel()


def has_pending_approval(run_id: str) -> bool:
    """Check if run_id is actively waiting on approval."""
    future = _pending_approvals.get(run_id)
    return future is not None and not future.done()


def _update_run(settings: Settings, run_id: str, **fields: Any) -> None:
    with Session(get_engine(settings.db_url)) as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            return
        for k, v in fields.items():
            setattr(run, k, v)
        session.add(run)
        session.commit()


class ApprovalHooks(PipelineHooks):
    """Pauses pipeline execution for user approval of plans and/or code."""

    async def on_plans_ranked(
        self, ctx: RunContext, ranked: list[PlanEvaluation]
    ) -> list[PlanEvaluation]:
        """Pause run and await user approval/pick/edit when approval mode is enabled."""
        if ctx.approval not in ("plans", "plans+code") or not ranked:
            return ranked

        pause_file = ctx.workdir / "pause_state.json"
        if pause_file.exists():
            try:
                data = json.loads(pause_file.read_text(encoding="utf-8"))
                if data.get("step") == "plans" and "decision" in data:
                    req = PlanApprovalRequest.model_validate(data["decision"])
                    pause_file.unlink(missing_ok=True)
                    return self._apply_plan_decision(ctx, ranked, req)
            except Exception:
                log.exception("Failed reading recovered approval decision from %s", pause_file)

        # Pause run
        _update_run(ctx.settings, ctx.run_id, status=RunStatus.awaiting_input.value)

        pause_data = {
            "step": "plans",
            "run_id": ctx.run_id,
            "ranked": [ev.model_dump(mode="json") for ev in ranked],
            "selected": ranked[0].plan.id,
        }
        pause_file.write_text(json.dumps(pause_data, indent=2), encoding="utf-8")

        await ctx.emit(
            Stage.select,
            "manager",
            "Pipeline paused: awaiting plan approval",
            kind="status",
            payload={
                "approval_step": "plans",
                "ranked": [ev.model_dump(mode="json") for ev in ranked],
                "selected": ranked[0].plan.id,
            },
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[PlanApprovalRequest] = loop.create_future()
        _pending_approvals[ctx.run_id] = future

        try:
            req = await future
        finally:
            _pending_approvals.pop(ctx.run_id, None)
            if pause_file.exists():
                with contextlib.suppress(Exception):
                    pause_file.unlink()

        ranked = self._apply_plan_decision(ctx, ranked, req)
        await ctx.emit(
            Stage.select,
            "manager",
            f"Plan approved: action={req.action.value}",
            kind="info",
            payload={"action": req.action.value, "plan_id": ranked[0].plan.id},
        )
        return ranked

    async def on_code_generated(
        self, ctx: RunContext, code: str, template_code: str, plan: Plan
    ) -> str:
        """Pause run and await user code review when approval mode is plans+code."""
        if ctx.approval != "plans+code":
            return code

        pause_file = ctx.workdir / "pause_state.json"
        if pause_file.exists():
            try:
                data = json.loads(pause_file.read_text(encoding="utf-8"))
                if data.get("step") == "code" and "decision" in data:
                    req = PlanApprovalRequest.model_validate(data["decision"])
                    pause_file.unlink(missing_ok=True)
                    return self._apply_code_decision(ctx, code, req)
            except Exception:
                log.exception("Failed reading recovered code decision from %s", pause_file)

        # Pause run
        _update_run(ctx.settings, ctx.run_id, status=RunStatus.awaiting_input.value)

        pause_data = {
            "step": "code",
            "run_id": ctx.run_id,
            "code": code,
            "template_code": template_code,
            "plan": plan.model_dump(mode="json"),
        }
        pause_file.write_text(json.dumps(pause_data, indent=2), encoding="utf-8")

        await ctx.emit(
            Stage.implement,
            "operation",
            "Pipeline paused: awaiting code review",
            kind="status",
            payload={
                "approval_step": "code",
                "code": code,
                "template_code": template_code,
                "plan_id": plan.id,
            },
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[PlanApprovalRequest] = loop.create_future()
        _pending_approvals[ctx.run_id] = future

        try:
            req = await future
        finally:
            _pending_approvals.pop(ctx.run_id, None)
            if pause_file.exists():
                with contextlib.suppress(Exception):
                    pause_file.unlink()

        code = self._apply_code_decision(ctx, code, req)
        await ctx.emit(
            Stage.implement,
            "operation",
            f"Code review approved: action={req.action.value}",
            kind="info",
            payload={"action": req.action.value},
        )
        return code

    def _apply_plan_decision(
        self, ctx: RunContext, ranked: list[PlanEvaluation], req: PlanApprovalRequest
    ) -> list[PlanEvaluation]:
        ranked = list(ranked)
        top_plan_id = ranked[0].plan.id if ranked else None
        human_override = False

        if req.action == PlanApprovalAction.pick:
            if req.plan_id and req.plan_id != top_plan_id:
                idx = next((i for i, ev in enumerate(ranked) if ev.plan.id == req.plan_id), None)
                if idx is not None:
                    chosen_ev = ranked.pop(idx)
                    ranked.insert(0, chosen_ev)
                    human_override = True
        elif req.action == PlanApprovalAction.edit:
            if req.edited_plan and ranked:
                curr = ranked[0].plan.model_dump()
                curr.update(req.edited_plan)
                updated_plan = Plan.model_validate(curr)
                updated_model = ranked[0].model.model_copy()
                if "model_family" in req.edited_plan:
                    updated_model.model_family = req.edited_plan["model_family"]
                if "hyperparameters" in req.edited_plan:
                    updated_model.hyperparameters = req.edited_plan["hyperparameters"]
                ranked[0] = ranked[0].model_copy(update={"plan": updated_plan, "model": updated_model})
                human_override = True
        elif req.action == PlanApprovalAction.approve:
            human_override = False

        prior_override = ctx.state.get("human_override", False)
        ctx.state["human_override"] = human_override or prior_override
        decision = req.model_dump(mode="json")
        ctx.state["human_decision"] = decision

        _update_run(
            ctx.settings,
            ctx.run_id,
            status=RunStatus.running.value,
            plan=ranked[0].plan.model_dump(mode="json"),
            human_override=ctx.state["human_override"],
            human_decision=decision,
        )
        return ranked

    def _apply_code_decision(
        self, ctx: RunContext, code: str, req: PlanApprovalRequest
    ) -> str:
        prior_override = ctx.state.get("human_override", False)
        human_override = False
        if req.edited_code and req.edited_code != code:
            code = req.edited_code
            human_override = True

        ctx.state["human_override"] = human_override or prior_override
        decision = req.model_dump(mode="json")
        ctx.state["human_decision"] = decision

        _update_run(
            ctx.settings,
            ctx.run_id,
            status=RunStatus.running.value,
            code=code,
            human_override=ctx.state["human_override"],
            human_decision=decision,
        )
        return code
