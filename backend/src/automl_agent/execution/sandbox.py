"""Run generated training scripts in an isolated working directory.

NOTE: this is process isolation (separate interpreter, cwd, timeout), not a
security sandbox. LLM-generated code runs with the server's permissions -
only run this locally / in a container you trust.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from automl_agent.schemas.plan import ExecutionResult

MAX_LOG_CHARS = 8000


def static_check(code: str) -> str | None:
    """Return an error message if the script obviously can't satisfy the contract."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    if "metrics.json" not in code:
        return "script must write its results to metrics.json"
    return None


def _tail(text: str) -> str:
    return text if len(text) <= MAX_LOG_CHARS else "...[truncated]...\n" + text[-MAX_LOG_CHARS:]


def _run_blocking(script: Path, timeout_s: int) -> ExecutionResult:
    env = {**os.environ, "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    metrics_file = script.parent / "metrics.json"
    metrics_file.unlink(missing_ok=True)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, script.name],
            cwd=script.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return ExecutionResult(
            ok=False,
            returncode=None,
            duration_s=time.perf_counter() - start,
            stdout=_tail(e.stdout if isinstance(e.stdout, str) else ""),
            stderr=f"Timed out after {timeout_s}s",
            timed_out=True,
        )
    metrics = None
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = None
    stderr = proc.stderr
    if proc.returncode == 0 and metrics is None:
        stderr += "\nScript finished but did not write a valid metrics.json"
    return ExecutionResult(
        ok=proc.returncode == 0 and metrics is not None,
        returncode=proc.returncode,
        duration_s=round(time.perf_counter() - start, 3),
        stdout=_tail(proc.stdout),
        stderr=_tail(stderr),
        metrics=metrics,
    )


async def run_script(
    code: str, workdir: Path, *, filename: str = "train.py", timeout_s: int = 600
) -> ExecutionResult:
    """Write `code` to `workdir/filename` and execute it with the current interpreter.

    Uses a worker thread + blocking subprocess so it works on any event loop
    (uvicorn's reload mode on Windows uses a selector loop without subprocess support).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / filename
    script.write_text(code, encoding="utf-8")
    if err := static_check(code):
        return ExecutionResult(ok=False, returncode=None, duration_s=0.0, stderr=err)
    return await asyncio.to_thread(_run_blocking, script, timeout_s)
