"""Extension hook points - where our own contribution plugs into the paper's pipeline.

Subclass `PipelineHooks`, override only what you need, and register it with
`register_hooks(...)` (e.g. from an extension module imported in main.py).
The Manager awaits every registered hook at the matching pipeline point.

Examples of what fits here (see docs/extension-ideas.md):
  * human-in-the-loop: `on_plans_ranked` waits for the user to pick/edit a plan
  * plan memory: `on_run_finished` stores the winning plan, `on_plans_generated` injects past ones
  * cost-aware planning: `on_plans_ranked` re-ranks with a token/compute budget
  * explainability: `on_run_finished` writes a model card / feature importances
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from automl_agent.agents.context import RunContext
    from automl_agent.agents.manager import PipelineResult
    from automl_agent.planning.retrieval import KnowledgeItem
    from automl_agent.schemas.plan import Plan, PlanEvaluation
    from automl_agent.schemas.task_spec import TaskSpec


class PipelineHooks:
    async def on_task_parsed(self, ctx: RunContext, spec: TaskSpec) -> TaskSpec:
        return spec

    async def on_knowledge_retrieved(
        self, ctx: RunContext, items: list[KnowledgeItem]
    ) -> list[KnowledgeItem]:
        return items

    async def on_plans_generated(self, ctx: RunContext, plans: list[Plan]) -> list[Plan]:
        return plans

    async def on_plans_ranked(self, ctx: RunContext, ranked: list[PlanEvaluation]) -> list[PlanEvaluation]:
        return ranked

    async def on_run_finished(self, ctx: RunContext, result: PipelineResult) -> None:
        return None


_registry: list[PipelineHooks] = []


def register_hooks(hooks: PipelineHooks) -> None:
    _registry.append(hooks)


def registered_hooks() -> list[PipelineHooks]:
    return list(_registry)


async def run_hook(name: str, ctx: RunContext, value: Any, *args: Any) -> Any:
    """Thread `value` through every registered hook's `name` method."""
    for hooks in _registry:
        value = await getattr(hooks, name)(ctx, value, *args)
    return value
