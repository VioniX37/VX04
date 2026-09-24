from __future__ import annotations

from pathlib import Path

from automl_agent.execution.renderer import render_template
from automl_agent.execution.sandbox import run_script
from automl_agent.schemas.events import Stage
from automl_agent.schemas.plan import CodeDraft, ExecutionResult, PlanEvaluation
from automl_agent.schemas.task_spec import TaskSpec

from .base import BaseAgent


class OperationAgent(BaseAgent):
    """Writes the training code for the selected plan, runs it, and debugs it on failure."""

    name = "operation_agent"
    prompt_name = "operation_agent"

    async def implement(
        self, spec: TaskSpec, ev: PlanEvaluation, workdir: Path
    ) -> tuple[str, ExecutionResult]:
        settings = self.ctx.settings
        plan = ev.plan.model_copy(update={"model_family": ev.model.model_family})
        base_code = render_template(
            spec,
            plan,
            self.ctx.dataset_path,
            hyperparameters=ev.model.hyperparameters or plan.hyperparameters,
        )
        context = {
            "task_spec": spec.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "data_agent": ev.data.model_dump(mode="json"),
            "model_agent": ev.model.model_dump(mode="json"),
        }

        use_llm = settings.codegen_mode == "llm"
        code = base_code
        if use_llm:
            draft = await self.ask_json(
                Stage.implement,
                "Write the training script for this plan.",
                {**context, "base_code": base_code},
                CodeDraft,
            )
            code = draft.code

        result = ExecutionResult(ok=False, returncode=None, duration_s=0.0, stderr="not run")
        for attempt in range(1, settings.max_debug_attempts + 1):
            await self.ctx.emit(
                Stage.implement,
                self.name,
                f"Running training script (attempt {attempt})",
                kind="status",
                payload={"attempt": attempt, "code": code},
            )
            result = await run_script(code, workdir, timeout_s=settings.exec_timeout_s)
            await self.ctx.emit(
                Stage.implement,
                self.name,
                "Script succeeded" if result.ok else f"Script failed: {result.stderr.strip()[-300:]}",
                kind="artifact" if result.ok else "warning",
                payload={"attempt": attempt, "result": result.model_dump(mode="json")},
            )
            if result.ok or not use_llm or attempt == settings.max_debug_attempts:
                break
            draft = await self.ask_json(
                Stage.implement,
                "The script failed. Fix it and return the full corrected script.",
                {**context, "base_code": code, "error": result.stderr[-4000:]},
                CodeDraft,
            )
            code = draft.code

        if not result.ok and code != base_code:
            # Safety net: the template alone is known to satisfy the contract.
            await self.ctx.emit(
                Stage.implement, self.name, "Falling back to the unmodified template", kind="warning"
            )
            code = base_code
            result = await run_script(code, workdir, timeout_s=settings.exec_timeout_s)
        return code, result
