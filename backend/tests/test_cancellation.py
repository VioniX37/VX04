"""Tests for Phase 2: Run cancellation and process-tree termination."""

import asyncio
import subprocess
import sys
import time

import psutil
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from automl_agent.execution.sandbox import kill_run_processes, register_process, run_script
from automl_agent.main import create_app
from automl_agent.schemas.run import RunStatus
from automl_agent.services.run_service import cancel_run
from automl_agent.storage.db import RunRecord, get_engine, init_db


def test_kill_run_processes_leaves_no_orphans(tmp_path):
    """Test that kill_run_processes recursively terminates both parent and child processes."""
    # A script that spawns a child process and both sleep for 60s
    child_code = (
        f'"{sys.executable}" -c "import time; time.sleep(60)"'
    )
    parent_script = (
        "import subprocess, time\n"
        f"child = subprocess.Popen({child_code!r}, shell=True)\n"
        "time.sleep(60)\n"
    )

    parent_proc = subprocess.Popen(
        [sys.executable, "-c", parent_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    run_id = "test_cancel_orphans"
    register_process(parent_proc.pid, run_id)

    # Allow child process to be spawned
    time.sleep(1)

    parent_ps = psutil.Process(parent_proc.pid)
    children = parent_ps.children(recursive=True)
    assert len(children) > 0, "Child process was not spawned"
    child_pids = [c.pid for c in children]

    # Kill the run processes
    kill_run_processes(run_id)

    # Verify parent and all children are dead within 3 seconds
    deadline = time.time() + 3
    while time.time() < deadline:
        parent_alive = psutil.pid_exists(parent_proc.pid)
        children_alive = any(psutil.pid_exists(p) for p in child_pids)
        if not parent_alive and not children_alive:
            break
        time.sleep(0.1)

    assert not psutil.pid_exists(parent_proc.pid), "Parent process still alive"
    for c_pid in child_pids:
        assert not psutil.pid_exists(c_pid), f"Child process {c_pid} still alive (orphan)"


def test_run_script_cancellation_cleans_up(tmp_path):
    """Test that cancelling an asyncio task running run_script terminates the process."""
    code = (
        "import time, json\n"
        "for i in range(100):\n"
        "    time.sleep(0.5)\n"
        "json.dump({'score': 1}, open('metrics.json', 'w'))\n"
    )
    run_id = "test_script_cancel"

    async def runner():
        await run_script(code, tmp_path, run_id=run_id, timeout_s=30)

    async def test_coro():
        task = asyncio.create_task(runner())
        await asyncio.sleep(0.5)
        # Cancel the task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(test_coro())


def test_cancel_endpoint_and_service(sample_csvs):
    """Test POST /api/runs/{id}/cancel cancels an active run and persists cancelled status."""
    with TestClient(create_app()) as client:
        # Register a dataset
        with sample_csvs["churn"].open("rb") as f:
            dataset = client.post("/api/datasets", files={"file": ("churn.csv", f, "text/csv")}).json()

        # Start a run
        resp = client.post("/api/runs", json={"dataset_id": dataset["id"], "prompt": "Predict churn"})
        assert resp.status_code == 201
        run_data = resp.json()
        run_id = run_data["id"]

        # Cancel the run
        cancel_resp = client.post(f"/api/runs/{run_id}/cancel")
        assert cancel_resp.status_code == 200
        cancelled = cancel_resp.json()
        assert cancelled["status"] == RunStatus.cancelled

        # Verify get_run returns cancelled
        fetched = client.get(f"/api/runs/{run_id}").json()
        assert fetched["status"] == RunStatus.cancelled

        # Repeated cancel should return 400
        repeat_resp = client.post(f"/api/runs/{run_id}/cancel")
        assert repeat_resp.status_code == 400

        # Nonexistent run cancel should return 404
        bad_resp = client.post("/api/runs/nonexistent/cancel")
        assert bad_resp.status_code == 404


def test_cancel_persists_partial_observations(settings):
    """Test that partial observations are persisted when a run is cancelled."""
    init_db(settings.db_url)
    with Session(get_engine(settings.db_url)) as s:
        run = RunRecord(id="cancel_obs_run", dataset_id="d1", prompt="Predict something", status="running")
        s.add(run)
        s.commit()

    # Cancel the run directly
    result = asyncio.run(cancel_run("cancel_obs_run", settings))
    assert result is not None
    assert result.status == RunStatus.cancelled

    with Session(get_engine(settings.db_url)) as s:
        db_run = s.get(RunRecord, "cancel_obs_run")
        assert db_run.status == "cancelled"
        assert db_run.error == "Cancelled by user"
