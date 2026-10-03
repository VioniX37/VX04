"""Tests for Phase 1: Run schemas, approval modes, and SQLite database migration."""

import sqlite3

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect
from sqlmodel import Session, create_engine

from automl_agent.schemas.run import (
    ApprovalMode,
    PlanApprovalAction,
    PlanApprovalRequest,
    RunCreate,
    RunOut,
    RunStatus,
)
from automl_agent.storage.db import (
    RunRecord,
    init_db,
    utcnow,
)


def test_run_status_enum():
    assert RunStatus.pending == "pending"
    assert RunStatus.running == "running"
    assert RunStatus.awaiting_input == "awaiting_input"
    assert RunStatus.succeeded == "succeeded"
    assert RunStatus.failed == "failed"
    assert RunStatus.cancelled == "cancelled"


def test_approval_mode_enum():
    assert ApprovalMode.auto == "auto"
    assert ApprovalMode.plans == "plans"
    assert ApprovalMode.plans_code == "plans+code"


def test_plan_approval_request_validation():
    req = PlanApprovalRequest()
    assert req.action == PlanApprovalAction.approve
    assert req.plan_id is None

    req_pick = PlanApprovalRequest(action="pick", plan_id="r1p2")
    assert req_pick.action == PlanApprovalAction.pick
    assert req_pick.plan_id == "r1p2"

    req_edit = PlanApprovalRequest(
        action="edit",
        plan_id="r1p1",
        edited_plan={"title": "Custom LightGBM"},
        edited_code="import lightgbm",
        feedback="Reduced n_estimators to 50",
    )
    assert req_edit.action == PlanApprovalAction.edit
    assert req_edit.edited_plan == {"title": "Custom LightGBM"}
    assert req_edit.edited_code == "import lightgbm"

    with pytest.raises(ValidationError):
        PlanApprovalRequest(action="invalid_action")


def test_run_create_and_run_out_defaults():
    create = RunCreate(dataset_id="ds1", prompt="Predict churn")
    assert create.approval == ApprovalMode.auto

    create_plans = RunCreate(dataset_id="ds1", prompt="Predict churn", approval=ApprovalMode.plans)
    assert create_plans.approval == "plans"

    create_code = RunCreate(dataset_id="ds1", prompt="Predict churn", approval="plans+code")
    assert create_code.approval == ApprovalMode.plans_code

    out = RunOut(
        id="run123",
        dataset_id="ds1",
        prompt="Predict churn",
        status=RunStatus.awaiting_input,
        created_at=utcnow(),
    )
    assert out.status == RunStatus.awaiting_input
    assert out.approval == ApprovalMode.auto
    assert out.human_override is False
    assert out.human_decision is None


def test_sqlite_forward_migration(tmp_path):
    """Simulate an existing SQLite database with an older schema missing new columns."""
    db_file = tmp_path / "old.db"
    db_url = f"sqlite:///{db_file}"

    # Create the old table without approval, human_override, human_decision
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE runrecord (
            id VARCHAR PRIMARY KEY,
            dataset_id VARCHAR NOT NULL,
            prompt VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            created_at TIMESTAMP,
            finished_at TIMESTAMP,
            task_spec JSON,
            plan JSON,
            metrics JSON,
            code VARCHAR,
            error VARCHAR,
            llm_usage JSON,
            config JSON
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE experiencerecord (
            run_id VARCHAR PRIMARY KEY,
            dataset_fingerprint VARCHAR NOT NULL,
            dataset_name VARCHAR NOT NULL,
            task_type VARCHAR NOT NULL,
            metric VARCHAR NOT NULL,
            higher_is_better BOOLEAN NOT NULL,
            n_rows INTEGER NOT NULL,
            success BOOLEAN NOT NULL,
            target_met BOOLEAN NOT NULL,
            best_score FLOAT,
            meta JSON NOT NULL,
            best_plan JSON,
            plans JSON,
            fixes JSON,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO runrecord (id, dataset_id, prompt, status) "
        "VALUES ('old_run', 'ds0', 'Old prompt', 'succeeded')"
    )
    conn.commit()
    conn.close()

    # Now apply the forward migration via init_db
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    init_db(db_url)

    inspector = inspect(engine)
    run_cols = {c["name"]: c for c in inspector.get_columns("runrecord")}
    assert "approval" in run_cols
    assert "human_override" in run_cols
    assert "human_decision" in run_cols

    exp_cols = {c["name"]: c for c in inspector.get_columns("experiencerecord")}
    assert "human_override" in exp_cols

    # Query the existing old record to ensure it has valid default values
    with Session(engine) as session:
        old = session.get(RunRecord, "old_run")
        assert old is not None
        assert old.approval == "auto"
        assert old.human_override is False
        assert old.human_decision is None

        # Insert a new record with the new fields
        new = RunRecord(
            id="new_run",
            dataset_id="ds1",
            prompt="New prompt",
            status=RunStatus.awaiting_input.value,
            approval=ApprovalMode.plans.value,
            human_override=True,
            human_decision={"action": "pick", "plan_id": "r1p2"},
        )
        session.add(new)
        session.commit()

        retrieved = session.get(RunRecord, "new_run")
        assert retrieved is not None
        assert retrieved.status == "awaiting_input"
        assert retrieved.approval == "plans"
        assert retrieved.human_override is True
        assert retrieved.human_decision == {"action": "pick", "plan_id": "r1p2"}
