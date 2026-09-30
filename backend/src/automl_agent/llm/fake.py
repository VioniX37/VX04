"""Deterministic offline LLM used for tests, CI and demos without API keys.

Agents embed their inputs as a `<context>{json}</context>` block in the user
message (see `automl_agent.agents.base.render_context`). The fake reads that
block and answers each schema with simple heuristics, so the whole pipeline
runs end to end with realistic structure.
"""

from __future__ import annotations

import json
import re

from automl_agent.execution.model_registry import supported_models
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import TaskType
from automl_agent.tools.heuristics import guess_task_spec

from .base import LLMClient, LLMResponse, Message

_CONTEXT = re.compile(r"<context>\s*(.*?)\s*</context>", re.DOTALL)

# Rough prior for how well each family does, used to fake the Model Agent's prediction.
_PRIOR = {
    "lightgbm": 0.89,
    "xgboost": 0.885,
    "sgd_hashing": 0.82,
    "hist_gradient_boosting": 0.88,
    "gradient_boosting": 0.87,
    "random_forest": 0.86,
    "extra_trees": 0.85,
    "svm": 0.83,
    "logistic_regression": 0.82,
    "linear_svm": 0.84,
    "sgd": 0.81,
    "naive_bayes": 0.8,
    "ridge": 0.78,
    "lasso": 0.76,
    "knn": 0.78,
}


def _context(messages: list[Message]) -> dict:
    for m in reversed(messages):
        if m["role"] == "user" and (match := _CONTEXT.search(m["content"])):
            return json.loads(match.group(1))
    return {}


class FakeLLM(LLMClient):
    """Heuristic stand-in for Gemini that needs no network or API key."""

    provider = "fake"

    def __init__(self, model: str = "fake-heuristic", temperature: float = 0.0) -> None:
        super().__init__(model, temperature)

    async def _complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        json_mode: bool,
        schema: dict | None = None,
    ) -> LLMResponse:
        """Answer from the `<context>` block using the handler for the expected schema."""
        ctx = _context(messages)
        handler = getattr(self, f"_answer_{ctx.get('expected', '')}", None)
        payload = handler(ctx) if handler else {"text": "ok"}
        text = json.dumps(payload) if json_mode or handler else str(payload)
        prompt_chars = sum(len(m["content"]) for m in messages)
        return LLMResponse(
            text=text, model=self.model, input_tokens=prompt_chars // 4, output_tokens=len(text) // 4
        )

    # --- one handler per response schema -------------------------------------------------------

    def _answer_TaskSpec(self, ctx: dict) -> dict:
        profile = DatasetProfile.model_validate(ctx["dataset_profile"])
        return guess_task_spec(ctx["user_prompt"], profile).model_dump(mode="json")

    def _answer_PlanSet(self, ctx: dict) -> dict:
        task_type = TaskType(ctx["task_spec"]["task_type"])
        families = list(ctx.get("allowed_models") or supported_models(task_type))
        # Like a sensible planner, try families that worked on similar past datasets first.
        remembered = [
            k["data"]["best_model_family"]
            for k in ctx.get("knowledge", [])
            if str(k.get("source", "")).startswith("memory:")
            and (k.get("data") or {}).get("best_model_family")
        ]
        families = list(dict.fromkeys([f for f in remembered if f in families] + families))
        n = int(ctx.get("n_plans", 3))
        offset = int(ctx.get("revision", 0)) * n  # revisions explore different families
        plans = []
        for i in range(n):
            fam = families[(offset + i) % len(families)]
            steps = (
                ["clean text (lowercase, strip)", "TF-IDF word 1-2 grams"]
                if task_type == TaskType.text_classification
                else [
                    "drop identifier columns",
                    "impute missing values",
                    "one-hot encode categoricals",
                    "standardize numeric features",
                ]
            )
            plans.append(
                {
                    "id": f"p{i + 1}",
                    "title": f"{fam.replace('_', ' ').title()} pipeline",
                    "rationale": f"{fam} is a strong choice for {task_type.value} per retrieved knowledge.",
                    "preprocessing": steps,
                    "model_family": fam,
                    "hyperparameters": {},
                    "validation": "stratified 80/20 hold-out",
                }
            )
        return {"plans": plans}

    def _answer_DataAgentResult(self, ctx: dict) -> dict:
        plan = ctx["plan"]
        return {
            "summary": f"Data pipeline for plan {plan['id']} looks feasible on this dataset.",
            "steps": plan["preprocessing"],
            "risks": ["class imbalance may affect minority classes"],
        }

    def _answer_ModelAgentResult(self, ctx: dict) -> dict:
        plan = ctx["plan"]
        fam = plan["model_family"]
        higher = ctx["task_spec"]["metric"] not in {"rmse", "mae", "mape"}
        prior = _PRIOR.get(fam, 0.75)
        return {
            "summary": f"{fam} with default hyperparameters.",
            "model_family": fam,
            "hyperparameters": plan.get("hyperparameters", {}),
            "predicted_score": prior if higher else round(1 - prior, 3),
            "predicted_train_time_s": 30.0,
        }

    def _answer_PlanAnalysis(self, ctx: dict) -> dict:
        return {"data": self._answer_DataAgentResult(ctx), "model": self._answer_ModelAgentResult(ctx)}

    def _answer_CodeDraft(self, ctx: dict) -> dict:
        return {"code": ctx["base_code"], "explanation": "Using the rendered template unchanged."}
