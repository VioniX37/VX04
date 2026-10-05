"""Tests for Phase 3, 4 & 5: Human-in-the-loop plan and code approval."""

import asyncio
import contextlib
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from automl_agent.config import Settings
from automl_agent.extensions.approval import (
    ApprovalHooks,
    has_pending_approval,
    resolve_approval,
)
from automl_agent.main import create_app
from automl_agent.memory.hooks import MemoryHooks
from automl_agent.memory.retriever import describe
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.plan import (
    DataAgentResult,
    ModelAgentResult,
    Plan,
    PlanEvaluation,
)
from automl_agent.schemas.run import (
    PlanApprovalAction,
    PlanApprovalRequest,
    RunStatus,
)
from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.services.run_service import approve_run, cancel_run
from automl_agent.storage.db import (
    ExperienceRecord,
    RunRecord,
    get_engine,
    init_db,
)


def _make_sample_plans() -> list[PlanEvaluation]:
    p1 = Plan(
        id="p1",
        title="LightGBM baseline",
        rationale="fast and strong",
        preprocessing=["impute"],
        model_family="lightgbm",
        hyperparameters={"n_estimators": 50},
    )
    p2 = Plan(
        id="p2",
        title="Random Forest alternative",
        rationale="robust to outliers",
        preprocessing=["scale"],
        model_family="random_forest",
        hyperparameters={"n_estimators": 80},
    )
    data = DataAgentResult(summary="ok", steps=["impute"])
    m1 = ModelAgentResult(summary="lgb", model_family="lightgbm", predicted_score=0.85)
    m2 = ModelAgentResult(summary="rf", model_family="random_forest", predicted_score=0.82)
    return [
        PlanEvaluation(plan=p1, data=data, model=m1, rank=1),
        PlanEvaluation(plan=p2, data=data, model=m2, rank=2),
    ]


def test_approve_endpoint_validations(sample_csvs):
    """Test validation and error handling on POST /api/runs/{id}/approve."""
    with TestClient(create_app()) as client:
        with sample_csvs["churn"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")}).json()

        # Nonexistent run -> 404
        resp = client.post(
            "/api/runs/nonexistent_123/approve",
            json={"action": "approve"},
        )
        assert resp.status_code == 404

        # Start an auto run (not awaiting input)
        resp = client.post(
            "/api/runs",
            json={"dataset_id": dataset["id"], "prompt": "Predict churn", "approval": "auto"},
        )
        assert resp.status_code == 201
        run_id = resp.json()["id"]

        # Run is not awaiting_input -> 400
        approve_resp = client.post(
            f"/api/runs/{run_id}/approve",
            json={"action": "approve"},
        )
        assert approve_resp.status_code == 400


def test_approval_hook_plans_approve(settings, tmp_path):
    """Test ApprovalHooks.on_plans_ranked pauses and resumes on approve."""
    init_db(settings.db_url)
    run_id = "test_hook_approve"
    workdir = tmp_path / run_id
    workdir.mkdir(parents=True)

    with Session(get_engine(settings.db_url)) as s:
        s.add(
            RunRecord(id=run_id, dataset_id="d1", prompt="Predict churn", approval="plans", status="running")
        )
        s.commit()

    class DummyContext:
        def __init__(self):
            self.run_id = run_id
            self.approval = "plans"
            self.workdir = workdir
            self.settings = settings
            self.state = {}
            self.events = []

        async def emit(self, stage, agent, message, kind="info", payload=None):
            self.events.append({"stage": stage, "message": message, "payload": payload})

    ctx = DummyContext()
    hooks = ApprovalHooks()
    ranked = _make_sample_plans()

    async def run_flow():
        # Start hook in background
        task = asyncio.create_task(hooks.on_plans_ranked(ctx, ranked))

        # Wait until run has paused
        for _ in range(50):
            if has_pending_approval(run_id):
                break
            await asyncio.sleep(0.02)
        assert has_pending_approval(run_id)

        # Check DB status is awaiting_input
        with Session(get_engine(settings.db_url)) as s:
            rec = s.get(RunRecord, run_id)
            assert rec.status == RunStatus.awaiting_input

        # Check pause file was created
        assert (workdir / "pause_state.json").exists()

        # Approve the top plan
        ok = resolve_approval(run_id, PlanApprovalRequest(action=PlanApprovalAction.approve))
        assert ok

        result_ranked = await task
        assert len(result_ranked) == 2
        assert result_ranked[0].plan.id == "p1"
        assert ctx.state.get("human_override") is False

        # Status restored to running
        with Session(get_engine(settings.db_url)) as s:
            rec = s.get(RunRecord, run_id)
            assert rec.status == RunStatus.running
            assert rec.human_override is False

    asyncio.run(run_flow())


def test_approval_hook_plans_pick(settings, tmp_path):
    """Test picking a different plan overrides top choice and flags human_override."""
    init_db(settings.db_url)
    run_id = "test_hook_pick"
    workdir = tmp_path / run_id
    workdir.mkdir(parents=True)

    with Session(get_engine(settings.db_url)) as s:
        s.add(
            RunRecord(id=run_id, dataset_id="d1", prompt="Predict churn", approval="plans", status="running")
        )
        s.commit()

    class DummyContext:
        def __init__(self):
            self.run_id = run_id
            self.approval = "plans"
            self.workdir = workdir
            self.settings = settings
            self.state = {}
            self.events = []

        async def emit(self, stage, agent, message, kind="info", payload=None):
            self.events.append({"stage": stage, "message": message, "payload": payload})

    ctx = DummyContext()
    hooks = ApprovalHooks()
    ranked = _make_sample_plans()

    async def run_flow():
        task = asyncio.create_task(hooks.on_plans_ranked(ctx, ranked))
        for _ in range(50):
            if has_pending_approval(run_id):
                break
            await asyncio.sleep(0.02)

        # User picks 'p2'
        resolve_approval(
            run_id,
            PlanApprovalRequest(action=PlanApprovalAction.pick, plan_id="p2"),
        )
        result_ranked = await task

        # 'p2' is now index 0
        assert result_ranked[0].plan.id == "p2"
        assert result_ranked[0].model.model_family == "random_forest"
        assert ctx.state.get("human_override") is True
        assert ctx.state.get("human_decision")["action"] == "pick"

        with Session(get_engine(settings.db_url)) as s:
            rec = s.get(RunRecord, run_id)
            assert rec.status == RunStatus.running
            assert rec.human_override is True
            assert rec.human_decision["plan_id"] == "p2"

    asyncio.run(run_flow())


def test_approval_hook_plans_edit(settings, tmp_path):
    """Test editing a plan modifies the plan parameters and sets human_override."""
    init_db(settings.db_url)
    run_id = "test_hook_edit"
    workdir = tmp_path / run_id
    workdir.mkdir(parents=True)

    with Session(get_engine(settings.db_url)) as s:
        s.add(
            RunRecord(id=run_id, dataset_id="d1", prompt="Predict churn", approval="plans", status="running")
        )
        s.commit()

    class DummyContext:
        def __init__(self):
            self.run_id = run_id
            self.approval = "plans"
            self.workdir = workdir
            self.settings = settings
            self.state = {}
            self.events = []

        async def emit(self, stage, agent, message, kind="info", payload=None):
            self.events.append({"stage": stage, "message": message, "payload": payload})

    ctx = DummyContext()
    hooks = ApprovalHooks()
    ranked = _make_sample_plans()

    async def run_flow():
        task = asyncio.create_task(hooks.on_plans_ranked(ctx, ranked))
        for _ in range(50):
            if has_pending_approval(run_id):
                break
            await asyncio.sleep(0.02)

        # User edits p1 hyperparameters and title
        resolve_approval(
            run_id,
            PlanApprovalRequest(
                action=PlanApprovalAction.edit,
                edited_plan={"title": "Custom Tuned LightGBM", "hyperparameters": {"n_estimators": 250}},
            ),
        )
        result_ranked = await task

        assert result_ranked[0].plan.title == "Custom Tuned LightGBM"
        assert result_ranked[0].plan.hyperparameters["n_estimators"] == 250
        assert ctx.state.get("human_override") is True

    asyncio.run(run_flow())


def test_approval_hook_code_review(settings, tmp_path):
    """Test on_code_generated pauses and allows code editing under plans+code mode."""
    init_db(settings.db_url)
    run_id = "test_hook_code"
    workdir = tmp_path / run_id
    workdir.mkdir(parents=True)

    with Session(get_engine(settings.db_url)) as s:
        s.add(
            RunRecord(
                id=run_id, dataset_id="d1", prompt="Predict churn", approval="plans+code", status="running"
            )
        )
        s.commit()

    class DummyContext:
        def __init__(self):
            self.run_id = run_id
            self.approval = "plans+code"
            self.workdir = workdir
            self.settings = settings
            self.state = {}
            self.events = []

        async def emit(self, stage, agent, message, kind="info", payload=None):
            self.events.append({"stage": stage, "message": message, "payload": payload})

    ctx = DummyContext()
    hooks = ApprovalHooks()
    original_code = "print('original')"
    template_code = "print('template')"
    plan = _make_sample_plans()[0].plan

    async def run_flow():
        task = asyncio.create_task(hooks.on_code_generated(ctx, original_code, template_code, plan))
        for _ in range(50):
            if has_pending_approval(run_id):
                break
            await asyncio.sleep(0.02)

        assert has_pending_approval(run_id)

        # User provides edited code
        edited_code = "print('human reviewed code')"
        resolve_approval(
            run_id,
            PlanApprovalRequest(action=PlanApprovalAction.edit, edited_code=edited_code),
        )
        final_code = await task
        assert final_code == edited_code
        assert ctx.state.get("human_override") is True

    asyncio.run(run_flow())


def test_cancel_during_awaiting_input(settings, tmp_path):
    """Test that cancel_run successfully cancels a run waiting in awaiting_input."""
    init_db(settings.db_url)
    run_id = "test_cancel_awaiting"
    workdir = tmp_path / run_id
    workdir.mkdir(parents=True)

    with Session(get_engine(settings.db_url)) as s:
        s.add(
            RunRecord(id=run_id, dataset_id="d1", prompt="Predict churn", approval="plans", status="running")
        )
        s.commit()

    class DummyContext:
        def __init__(self):
            self.run_id = run_id
            self.approval = "plans"
            self.workdir = workdir
            self.settings = settings
            self.state = {}

        async def emit(self, stage, agent, message, kind="info", payload=None):
            pass

    ctx = DummyContext()
    hooks = ApprovalHooks()
    ranked = _make_sample_plans()

    async def run_flow():
        task = asyncio.create_task(hooks.on_plans_ranked(ctx, ranked))
        for _ in range(50):
            if has_pending_approval(run_id):
                break
            await asyncio.sleep(0.02)

        # Cancel while paused
        cancelled_run = await cancel_run(run_id, settings)
        assert cancelled_run is not None
        assert cancelled_run.status == RunStatus.cancelled

        # Future should be cancelled
        assert not has_pending_approval(run_id)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run_flow())


def test_end_to_end_api_approval_flow(sample_csvs):
    """Full end-to-end integration test via API client with approval='plans'."""
    with TestClient(create_app()) as client:
        with sample_csvs["churn"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")}).json()

        # Start a run with approval='plans'
        resp = client.post(
            "/api/runs",
            json={"dataset_id": dataset["id"], "prompt": "Predict churn", "approval": "plans"},
        )
        assert resp.status_code == 201
        run_id = resp.json()["id"]

        # Poll until the run pauses in awaiting_input
        deadline = time.time() + 35
        paused = False
        while time.time() < deadline:
            run_data = client.get(f"/api/runs/{run_id}").json()
            if run_data["status"] == "awaiting_input":
                paused = True
                break
            time.sleep(0.2)
        assert paused, f"Run did not enter awaiting_input within 35s (status={run_data['status']})"

        # Check events history has the pause event
        events = client.get(f"/api/runs/{run_id}/events/history").json()
        pause_events = [e for e in events if (e.get("payload") or {}).get("approval_step") == "plans"]
        assert len(pause_events) > 0

        # Approve the run via API
        approve_resp = client.post(
            f"/api/runs/{run_id}/approve",
            json={"action": "approve"},
        )
        assert approve_resp.status_code == 200

        # Poll until the run finishes
        deadline = time.time() + 60
        finished = False
        while time.time() < deadline:
            run_data = client.get(f"/api/runs/{run_id}").json()
            if run_data["status"] in ("succeeded", "failed"):
                finished = True
                break
            time.sleep(0.2)
        assert finished, f"Run did not finish within 30s (status={run_data['status']})"
        assert run_data["status"] == "succeeded"


def test_experience_memory_records_human_override(settings):
    """Test Phase 5: MemoryHooks stores human_override in ExperienceRecord and retriever describes it."""
    init_db(settings.db_url)
    run_id = "test_memory_override"

    profile = DatasetProfile(
        n_rows=500,
        n_cols=5,
        columns=[],
        guessed_target="target",
    )
    spec = TaskSpec(
        task_type=TaskType.tabular_classification,
        target_column="target",
        metric="roc_auc",
        higher_is_better=True,
    )

    class DummyResult:
        success = True
        target_met = True
        task_spec = spec
        plan = Plan(
            id="p1",
            title="Custom Plan",
            rationale="chosen by expert",
            preprocessing=[],
            model_family="lightgbm",
        )
        metrics = {"score": 0.91}
        observations = []
        fixes = []

    class DummyContext:
        def __init__(self):
            self.run_id = run_id
            self.dataset_id = "d1"
            self.profile = profile
            self.settings = settings
            self.state = {"human_override": True}

    hooks = MemoryHooks()
    asyncio.run(hooks.on_run_finished(DummyContext(), DummyResult()))

    # Verify ExperienceRecord in DB has human_override = True
    with Session(get_engine(settings.db_url)) as s:
        rec = s.exec(select(ExperienceRecord).where(ExperienceRecord.run_id == run_id)).first()
        assert rec is not None
        assert rec.human_override is True

        # Test retriever description
        knowledge_item = describe(rec, 0.15)
        assert knowledge_item.data.get("human_override") is True
        assert "human expert" in knowledge_item.content.lower()


def test_crash_recovery_resumes_paused_run(sample_csvs, settings):
    """Test that a paused run without in-memory state (post-restart) resumes via approve_run."""
    from automl_agent.storage.db import DatasetRecord
    from automl_agent.tools import profile_file

    init_db(settings.db_url)
    dataset_id = "test_ds_restart"
    with Session(get_engine(settings.db_url)) as s:
        s.add(
            DatasetRecord(
                id=dataset_id,
                filename="churn.csv",
                path=str(sample_csvs["churn"]),
                profile=profile_file(sample_csvs["churn"]).model_dump(mode="json"),
            )
        )
        s.commit()

    # Create a paused run record directly in DB as if a server restarted while awaiting_input
    run_id = "test_restart_run"
    workdir = settings.runs_dir / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    pause_data = {
        "step": "plans",
        "run_id": run_id,
        "ranked": [ev.model_dump(mode="json") for ev in _make_sample_plans()],
        "selected": "p1",
    }
    (workdir / "pause_state.json").write_text(json.dumps(pause_data), encoding="utf-8")

    with Session(get_engine(settings.db_url)) as s:
        s.add(
            RunRecord(
                id=run_id,
                dataset_id=dataset_id,
                prompt="Predict churn",
                approval="plans",
                status=RunStatus.awaiting_input.value,
            )
        )
        s.commit()

    # Ensure no in-memory future exists
    assert not has_pending_approval(run_id)

    # Call approve_run and wait for task on the running loop
    req = PlanApprovalRequest(action=PlanApprovalAction.approve)

    async def run_flow():
        resumed = await approve_run(run_id, req, settings=settings)
        assert resumed is not None

        deadline = time.time() + 90
        finished = False
        while time.time() < deadline:
            with Session(get_engine(settings.db_url)) as s:
                run_rec = s.get(RunRecord, run_id)
                if run_rec and run_rec.status in ("succeeded", "failed"):
                    finished = True
                    break
            await asyncio.sleep(0.5)

        st = run_rec.status if run_rec else None
        err = run_rec.error if run_rec else None
        assert finished, f"Run did not finish in time; status={st}, error={err}"
        assert run_rec.status == "succeeded"

    asyncio.run(run_flow())


def test_approval_timeout_falls_back_to_auto(tmp_path: Path):
    """When approval_timeout_s expires, the hook falls back to auto approval without hanging."""
    settings = Settings(
        workspace_dir=tmp_path / "ws",
        db_url=f"sqlite:///{(tmp_path / 'ws' / 'automl.db').as_posix()}",
        approval_timeout_s=1,
    )
    init_db(settings.db_url)
    run_id = "test_timeout_fallback"
    with Session(get_engine(settings.db_url)) as s:
        s.add(
            RunRecord(
                id=run_id,
                dataset_id="d1",
                prompt="test prompt",
                status="running",
                approval="plans",
            )
        )
        s.commit()

    ranked = _make_sample_plans()

    workdir = tmp_path / "ws" / "runs" / run_id
    workdir.mkdir(parents=True)

    class DummyContext:
        def __init__(self):
            self.run_id = run_id
            self.approval = "plans"
            self.workdir = workdir
            self.settings = settings
            self.state = {}
            self.events = []

        async def emit(self, stage, agent, message, kind="info", payload=None):
            self.events.append({"stage": stage, "message": message, "payload": payload})

    ctx = DummyContext()

    hooks = ApprovalHooks()

    async def run():
        # Should pause, wait 1s, time out, and return ranked with top plan approved
        start = time.time()
        res = await hooks.on_plans_ranked(ctx, ranked)
        elapsed = time.time() - start
        assert len(res) == len(ranked)
        assert res[0].plan.id == "p1"
        assert 0.8 <= elapsed < 4.0

    asyncio.run(run())


def test_recovered_decision_applies_to_the_plans_the_user_reviewed(settings, tmp_path):
    """After a restart, a pick refers to the saved plans, even if re-planning ranks different ones."""
    init_db(settings.db_url)
    run_id = "test_recovered_pick"
    workdir = tmp_path / run_id
    workdir.mkdir(parents=True)
    with Session(get_engine(settings.db_url)) as s:
        s.add(RunRecord(id=run_id, dataset_id="d1", prompt="p", approval="plans", status="running"))
        s.commit()
    reviewed = _make_sample_plans()
    (workdir / "pause_state.json").write_text(
        json.dumps(
            {
                "step": "plans",
                "run_id": run_id,
                "ranked": [ev.model_dump(mode="json") for ev in reviewed],
                "decision": PlanApprovalRequest(action=PlanApprovalAction.pick, plan_id="p2").model_dump(
                    mode="json"
                ),
            }
        ),
        encoding="utf-8",
    )

    class Ctx:
        approval = "plans"
        state: dict = {}

        def __init__(self):
            self.run_id, self.workdir, self.settings = run_id, workdir, settings

    replanned = [
        ev.model_copy(update={"plan": ev.plan.model_copy(update={"id": f"new-{ev.plan.id}"})})
        for ev in reviewed
    ]
    ranked = asyncio.run(ApprovalHooks().on_plans_ranked(Ctx(), replanned))
    assert ranked[0].plan.id == "p2"
    assert ranked[0].model.model_family == "random_forest"


def test_edit_to_unsupported_model_family_is_normalized(settings):
    """An edited model name the registry does not know is mapped to a supported family before training."""
    from automl_agent.agents import AgentManager, RunContext
    from automl_agent.llm import create_llm
    from automl_agent.services.event_bus import EventBus
    from automl_agent.tools import profile_file

    path = Path(__file__).resolve().parents[2] / "data" / "samples" / "customer_churn.csv"
    ctx = RunContext(
        run_id="edit_norm",
        prompt="Predict churn",
        dataset_path=path,
        profile=profile_file(path),
        workdir=settings.runs_dir / "edit_norm",
        settings=settings,
        llm=create_llm(settings),
        bus=EventBus(),
    )
    manager = AgentManager(ctx)
    spec = TaskSpec(task_type=TaskType.tabular_classification, target_column="churn", metric="accuracy")
    edited = _make_sample_plans()
    edited[0] = edited[0].model_copy(
        update={"model": edited[0].model.model_copy(update={"model_family": "LightGBM-DART"})}
    )
    ranked = asyncio.run(manager._validated_choice(spec, edited))
    assert ranked[0].model.model_family == "lightgbm"
    assert ranked[0].plan.model_family == "lightgbm"
