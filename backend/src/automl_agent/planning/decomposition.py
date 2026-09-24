"""Plan decomposition (paper §3.3): split a plan into data- and model-specific sub-tasks.

The paper prompts the manager to decompose; since our Plan schema already
separates preprocessing from modelling we decompose deterministically, which
is cheaper and removes one LLM call per plan.
"""

from __future__ import annotations

import json

from automl_agent.schemas.plan import Plan, SubTask


def decompose(plan: Plan) -> tuple[list[SubTask], list[SubTask]]:
    data = [
        SubTask(id=f"{plan.id}-d{i + 1}", agent="data", instruction=step)
        for i, step in enumerate(plan.preprocessing)
    ]
    data.append(
        SubTask(id=f"{plan.id}-d{len(data) + 1}", agent="data", instruction=f"Split: {plan.validation}")
    )
    hp = json.dumps(plan.hyperparameters) if plan.hyperparameters else "sensible defaults"
    model = [
        SubTask(id=f"{plan.id}-m1", agent="model", instruction=f"Build a {plan.model_family} model"),
        SubTask(id=f"{plan.id}-m2", agent="model", instruction=f"Choose hyperparameters (start from {hp})"),
        SubTask(
            id=f"{plan.id}-m3", agent="model", instruction="Estimate hold-out performance and training time"
        ),
    ]
    return data, model
