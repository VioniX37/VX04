"""End-to-end pipeline runs with the offline FakeLLM (no network, no API keys)."""

import asyncio

import pytest

from automl_agent.agents import AgentManager, RunContext
from automl_agent.llm import create_llm
from automl_agent.schemas.events import Stage
from automl_agent.services.event_bus import EventBus
from automl_agent.tools import profile_file


def _run(settings, csv, prompt):
    bus = EventBus()
    ctx = RunContext(
        run_id="test",
        prompt=prompt,
        dataset_path=csv,
        profile=profile_file(csv),
        workdir=settings.runs_dir / "test",
        settings=settings,
        llm=create_llm(settings),
        bus=bus,
    )
    result = asyncio.run(AgentManager(ctx).run())
    return result, bus.history("test"), ctx


@pytest.mark.parametrize(
    ("key", "prompt", "metric"),
    [
        ("churn", "Predict whether a customer will churn. Use f1 score.", "f1_macro"),
        ("houses", "Predict the house price.", "rmse"),
        ("reviews", "Classify the sentiment of product reviews.", "f1_macro"),
    ],
)
def test_pipeline_end_to_end(settings, sample_csvs, key, prompt, metric):
    result, events, ctx = _run(settings, sample_csvs[key], prompt)

    assert result.success, result.error
    assert result.task_spec.metric == metric
    assert isinstance(result.metrics["score"], float)
    assert (ctx.workdir / "attempt_1" / "metrics.json").exists()
    assert (ctx.workdir / "attempt_1" / "model.joblib").exists()

    stages = {e.stage for e in events}
    for stage in (
        Stage.parse,
        Stage.verify_request,
        Stage.retrieve,
        Stage.plan,
        Stage.execute_plans,
        Stage.select,
        Stage.implement,
        Stage.verify_impl,
    ):
        assert stage in stages
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert ctx.llm.usage.calls >= 1 + 1 + 2 * settings.n_plans + 1


def test_unreachable_target_triggers_revisions(settings, sample_csvs):
    settings.max_revisions = 1
    result, events, _ = _run(settings, sample_csvs["churn"], "Predict churn with 99.9% accuracy")
    assert result.success  # best attempt is still reported
    assert not result.target_met
    assert len(result.attempts) == 2
    assert any("Revising" in e.message for e in events)
