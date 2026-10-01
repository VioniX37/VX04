"""Runs orphaned by a server restart are detected and closed on startup."""

import os

import psutil
from sqlmodel import Session

from automl_agent.schemas.events import AgentEvent
from automl_agent.services.run_service import events_log_path, recover_interrupted_runs
from automl_agent.storage.db import RunRecord, get_engine, init_db


def _dead_pid() -> int:
    pid = 4_000_000
    while psutil.pid_exists(pid):
        pid += 1
    return pid


def test_orphaned_runs_are_marked_interrupted(settings):
    init_db(settings.db_url)
    with Session(get_engine(settings.db_url)) as s:
        s.add(
            RunRecord(id="orphan", dataset_id="d", prompt="p", status="running", config={"pid": _dead_pid()})
        )
        s.add(RunRecord(id="never-started", dataset_id="d", prompt="p", status="pending"))
        s.add(
            RunRecord(id="alive", dataset_id="d", prompt="p", status="running", config={"pid": os.getppid()})
        )
        s.add(RunRecord(id="done", dataset_id="d", prompt="p", status="succeeded"))
        s.commit()
    log = events_log_path(settings, "orphan")
    log.parent.mkdir(parents=True)
    log.write_text(
        AgentEvent(seq=1, run_id="orphan", stage="ground", agent="manager", message="x").model_dump_json()
        + "\n"
    )

    assert sorted(recover_interrupted_runs(settings)) == ["never-started", "orphan"]

    with Session(get_engine(settings.db_url)) as s:
        assert s.get(RunRecord, "orphan").status == "failed"
        assert "Interrupted" in s.get(RunRecord, "orphan").error
        assert s.get(RunRecord, "alive").status == "running"  # owned by a live process (e.g. a CLI run)
        assert s.get(RunRecord, "done").status == "succeeded"
    events = [AgentEvent.model_validate_json(line) for line in log.read_text().splitlines()]
    assert [e.seq for e in events] == [1, 2] and events[-1].stage == "done"


def test_serve_command_is_available():
    from automl_agent.cli import build_parser

    args = build_parser().parse_args(["serve", "--reload", "--port", "8123"])
    assert args.reload and args.port == 8123
