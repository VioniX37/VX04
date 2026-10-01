"""Experience memory: meta-features, storage, recall, leakage guard and fix reuse."""

import asyncio

from automl_agent.agents import AgentManager, RunContext
from automl_agent.agents.operation_agent import error_signature
from automl_agent.llm import create_llm
from automl_agent.memory import ExperienceStore, MemoryRetriever, dataset_fingerprint, distance, meta_features
from automl_agent.schemas.events import Stage
from automl_agent.services.event_bus import EventBus
from automl_agent.tools import guess_task_spec, profile_file


def _run(settings, csv, prompt, run_id):
    bus = EventBus()
    ctx = RunContext(
        run_id=run_id,
        prompt=prompt,
        dataset_path=csv,
        profile=profile_file(csv),
        workdir=settings.runs_dir / run_id,
        settings=settings,
        llm=create_llm(settings),
        bus=bus,
        dataset_id=csv.stem,
    )
    result = asyncio.run(AgentManager(ctx).run())
    return result, bus.history(run_id)


def _other_churn(sample_csvs, tmp_path):
    """A second, similar-but-different churn dataset (different fingerprint)."""
    import importlib.util

    from tests.conftest import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "gen", REPO_ROOT / "data" / "samples" / "generate_samples.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    path = tmp_path / "churn_b.csv"
    gen.churn(350).to_csv(path, index=False)
    return path


def test_meta_features_and_fingerprint(sample_csvs):
    profile = profile_file(sample_csvs["churn"])
    spec = guess_task_spec("Predict churn", profile)
    m = meta_features(profile, spec)
    assert m["log_classes"] == 1.0 and m["frac_text"] == 0 and m["log_imbalance"] > 0
    assert distance(m, m) == 0
    assert dataset_fingerprint(profile) == dataset_fingerprint(profile_file(sample_csvs["churn"]))
    assert dataset_fingerprint(profile) != dataset_fingerprint(profile_file(sample_csvs["houses"]))


def test_runs_are_stored_and_recalled_for_similar_data(settings, sample_csvs, tmp_path):
    first, _ = _run(settings, sample_csvs["churn"], "Predict churn", "run1")
    records = ExperienceStore(settings).for_task("tabular_classification")
    assert len(records) == 1 and records[0].best_plan["model_family"] == first.plan.model_family
    assert records[0].plans and all(p["observed_score"] is not None for p in records[0].plans)

    second, events = _run(settings, _other_churn(sample_csvs, tmp_path), "Predict churn", "run2")
    assert any(s.startswith("memory:run1") for s in second.knowledge_sources)
    first_plan = next(e for e in events if e.stage == Stage.plan and e.kind == "llm").payload["output"][
        "plans"
    ][0]
    assert first_plan["model_family"] == first.plan.model_family  # planner reused the remembered winner


def test_memory_is_task_type_specific(settings, sample_csvs):
    _run(settings, sample_csvs["churn"], "Predict churn", "clf")
    reg, _ = _run(settings, sample_csvs["houses"], "Predict the house price", "reg")
    assert not any(s.startswith("memory:") for s in reg.knowledge_sources)


def test_leave_one_dataset_out_guard(settings, sample_csvs):
    _run(settings, sample_csvs["churn"], "Predict churn", "a")
    profile = profile_file(sample_csvs["churn"])
    spec = guess_task_spec("Predict churn", profile)
    settings.memory_exclude_same_dataset = False
    assert asyncio.run(MemoryRetriever(settings).retrieve(spec, profile, ""))
    settings.memory_exclude_same_dataset = True
    assert asyncio.run(MemoryRetriever(settings).retrieve(spec, profile, "")) == []


def test_memory_disabled_stores_nothing(settings, sample_csvs):
    settings.memory_enabled = False
    _run(settings, sample_csvs["churn"], "Predict churn", "off")
    assert ExperienceStore(settings).for_task("tabular_classification") == []


def test_error_signature_normalises_data_specific_details():
    a = "Traceback...\n  File \"x.py\", line 12\nKeyError: 'monthly_charges'"
    b = "Traceback...\n  File \"y.py\", line 99\nKeyError: 'tenure'"
    assert error_signature(a) == error_signature(b) == "KeyError: '…'"
    assert (
        error_signature("ValueError: could not convert 3 values") == "ValueError: could not convert N values"
    )


def test_recall_handles_equally_similar_runs(settings, sample_csvs):
    """Repeated runs on the same dataset have identical distance; recall must not compare records."""
    for i in range(3):
        _run(settings, sample_csvs["churn"], "Predict churn", f"repeat{i}")
    profile = profile_file(sample_csvs["churn"])
    spec = guess_task_spec("Predict churn", profile)
    items = asyncio.run(MemoryRetriever(settings).retrieve(spec, profile, ""))
    assert [i.source for i in items] == ["memory:repeat2", "memory:repeat1", "memory:repeat0"]
