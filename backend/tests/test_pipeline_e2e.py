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


def test_agent_fusion_halves_analysis_calls(settings, sample_csvs):
    _, _, separate = _run(settings, sample_csvs["churn"], "Predict churn")
    settings.agent_fusion = True
    result, events, fused = _run(settings, sample_csvs["churn"], "Predict churn")
    assert result.success
    assert separate.llm.usage.calls - fused.llm.usage.calls == settings.n_plans
    assert any(e.agent == "plan_analyst" for e in events)


def test_pseudo_mode_skips_grounding(settings, sample_csvs):
    settings.verification_mode = "pseudo"
    result, events, _ = _run(settings, sample_csvs["churn"], "Predict churn")
    assert result.success
    assert not any(e.stage == Stage.ground for e in events)
    assert all(o["final"] or "observed_score" not in o for o in result.observations)


def test_grounded_mode_runs_every_plan_and_logs_observations(settings, sample_csvs):
    result, events, ctx = _run(settings, sample_csvs["churn"], "Predict churn")
    assert result.success
    rung_events = [e for e in events if e.stage == Stage.ground and e.payload and "rung" in e.payload]
    assert len(rung_events) == 1 and len(rung_events[0].payload["rung"]["results"]) == settings.n_plans
    grounding_rows = [o for o in result.observations if not o["final"]]
    assert len(grounding_rows) == settings.n_plans
    assert all(o["observed_score"] is not None and o["predicted_score"] is not None for o in grounding_rows)
    final = [o for o in result.observations if o["final"]]
    assert final and final[0]["split"] == "test"
    selected = next(e for e in events if e.stage == Stage.select).payload["selected"]
    best = max(grounding_rows, key=lambda o: o["observed_score"])
    assert selected == best["plan_id"]


def test_budget_stops_revisions(settings, sample_csvs):
    settings.max_revisions = 3
    settings.budget_llm_calls = 1  # exhausted after the first round
    result, _, _ = _run(settings, sample_csvs["churn"], "Predict churn with 99.9% accuracy")
    assert result.stop_reason == "llm_calls_budget"
    assert len(result.attempts) == 1


def test_events_carry_llm_metadata_and_grounding_progress(settings, sample_csvs):
    _, events, _ = _run(settings, sample_csvs["churn"], "Predict churn")
    llm = [e for e in events if e.kind == "llm"]
    assert llm and all({"role", "duration_s", "calls", "input_tokens"} <= set(e.payload["meta"]) for e in llm)
    ground_progress = [e for e in events if e.stage == Stage.ground and e.kind == "telemetry"]
    assert {e.payload["plan_id"] for e in ground_progress} == {"r1p1", "r1p2", "r1p3"}
