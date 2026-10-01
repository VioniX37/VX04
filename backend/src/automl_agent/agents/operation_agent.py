"""Operation Agent: turns the selected plan into code, runs it on the full data, and debugs it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl_agent.execution.renderer import render_template
from automl_agent.execution.sandbox import run_script
from automl_agent.schemas.events import Stage
from automl_agent.schemas.plan import CodeDraft, ExecutionResult, PlanEvaluation
from automl_agent.schemas.task_spec import TaskSpec

from .base import BaseAgent


def error_signature(stderr: str) -> str:
    """Normalise the last exception line of a traceback into a reusable signature."""
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    exc = next((ln for ln in reversed(lines) if re.match(r"^[\w.]+(Error|Exception|Warning)\b", ln)), None)
    text = exc or (lines[-1] if lines else "unknown error")
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", text)  # strip quoted, data-specific values
    return re.sub(r"\d+", "N", text)[:200]


@dataclass
class ImplementationOutcome:
    """Final code, its execution result, and the debug history (used by experience memory)."""

    code: str
    result: ExecutionResult
    attempts: int = 0
    fixes: list[dict[str, Any]] = field(default_factory=list)
    used_template_fallback: bool = False


class OperationAgent(BaseAgent):
    """Writes the training code for the selected plan, runs it, and debugs it on failure."""

    name = "operation_agent"
    prompt_name = "operation_agent"
    model_role = "smart"

    def base_code(self, spec: TaskSpec, ev: PlanEvaluation) -> str:
        """Render the template for a final, full-data run scored on the held-out test split."""
        ctx = self.ctx
        if ctx.split is None:
            raise RuntimeError("data must be prepared (split created) before implementation")
        plan = ev.plan.model_copy(update={"model_family": ev.model.model_family})
        return render_template(
            spec,
            plan,
            ctx.dataset_path,
            ctx.split.path,
            hyperparameters=ev.model.hyperparameters or plan.hyperparameters,
            eval_split="test",
            n_jobs=ctx.settings.exec_n_jobs,
            n_rows=ctx.train_rows,
        )

    async def _run(self, code: str, workdir: Path, attempt: int) -> ExecutionResult:
        ctx = self.ctx

        async def on_progress(item: dict[str, Any]) -> None:
            if item.get("telemetry"):
                await ctx.emit(
                    Stage.implement, self.name, "resources", kind="telemetry",
                    payload={"attempt": attempt, "telemetry": item, "job": "final"},
                )  # fmt: skip
                return
            await ctx.emit(
                Stage.implement,
                self.name,
                f"Training: {item.get('stage', 'progress')}",
                kind="info",
                payload={"attempt": attempt, "progress": item},
            )

        return await run_script(
            code,
            workdir,
            timeout_s=ctx.settings.exec_timeout_s,
            max_mem_mb=ctx.settings.exec_max_mem_mb or None,
            on_progress=on_progress,
        )

    async def implement(
        self,
        spec: TaskSpec,
        ev: PlanEvaluation,
        workdir: Path,
        *,
        past_fixes: list[dict[str, Any]] | None = None,
    ) -> ImplementationOutcome:
        """Generate, run and (if needed) repair the training script for plan `ev`.

        Args:
            past_fixes: Error->fix notes from similar past runs (experience memory), shown to the
                model when a script fails with a matching error signature.
        """
        settings = self.ctx.settings
        base = self.base_code(spec, ev)
        plan = ev.plan.model_copy(update={"model_family": ev.model.model_family})
        context = {
            "task_spec": spec.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "data_agent": ev.data.model_dump(mode="json"),
            "model_agent": ev.model.model_dump(mode="json"),
        }

        use_llm = settings.codegen_mode == "llm"
        code = base
        if use_llm:
            draft = await self.ask_json(
                Stage.implement,
                "Write the training script for this plan.",
                {**context, "base_code": base},
                CodeDraft,
            )
            code = draft.code

        outcome = ImplementationOutcome(
            code=code, result=ExecutionResult(ok=False, returncode=None, duration_s=0.0)
        )
        last_fix: dict[str, str] | None = None  # the most recent error->fix, recorded if it works
        for attempt in range(1, settings.max_debug_attempts + 1):
            outcome.attempts = attempt
            await self.ctx.emit(
                Stage.implement,
                self.name,
                f"Running training script (attempt {attempt})",
                kind="status",
                payload={"attempt": attempt, "code": code},
            )
            result = await self._run(code, workdir, attempt)
            outcome.code, outcome.result = code, result
            await self.ctx.emit(
                Stage.implement,
                self.name,
                "Script succeeded" if result.ok else f"Script failed: {result.stderr.strip()[-300:]}",
                kind="artifact" if result.ok else "warning",
                payload={"attempt": attempt, "result": result.model_dump(mode="json")},
            )
            if result.ok:
                if last_fix is not None:
                    outcome.fixes.append(last_fix)
                break
            if not use_llm or attempt == settings.max_debug_attempts:
                break
            signature = error_signature(result.stderr)
            hints = [f for f in (past_fixes or []) if f.get("error") == signature]
            fix_context = {**context, "base_code": code, "error": result.stderr[-4000:]}
            if hints:
                fix_context["past_fixes"] = hints[:3]
            draft = await self.ask_json(
                Stage.implement,
                "The script failed. Fix it and return the full corrected script.",
                fix_context,
                CodeDraft,
            )
            last_fix = {"error": signature, "fix": draft.explanation or "rewrote the failing section"}
            code = draft.code

        if not outcome.result.ok and code != base:
            # Safety net: the template alone is known to satisfy the contract.
            await self.ctx.emit(
                Stage.implement, self.name, "Falling back to the unmodified template", kind="warning"
            )
            outcome.code = base
            outcome.result = await self._run(base, workdir, outcome.attempts + 1)
            outcome.used_template_fallback = True
        return outcome
