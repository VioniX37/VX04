"""Run generated training scripts as supervised child processes.

Each script runs in its own working directory with a wall-clock timeout and a
resident-memory ceiling (checked for the whole process tree). Lines printed
as ``PROGRESS {json}`` are forwarded live to the caller.

NOTE: this is process isolation, not a security sandbox. LLM-generated code
runs with the server's permissions; only run it on machines you trust.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import psutil

from automl_agent.schemas.plan import ExecutionResult

MAX_LOG_CHARS = 8000
PROGRESS_PREFIX = "PROGRESS "
POLL_S = 0.25
TELEMETRY_S = 2.0  # how often resource samples are forwarded to the caller

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


def static_check(code: str) -> str | None:
    """Return an error message if the script obviously can't satisfy the contract."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    if "metrics.json" not in code:
        return "script must write its results to metrics.json"
    return None


def default_memory_limit_mb() -> int:
    """80% of physical RAM."""
    return int(psutil.virtual_memory().total / 1024**2 * 0.8)


def _tail(text: str) -> str:
    return text if len(text) <= MAX_LOG_CHARS else "...[truncated]...\n" + text[-MAX_LOG_CHARS:]


def _tree_rss_mb(proc: psutil.Process) -> float:
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0.0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            continue
    return total / 1024**2


def _tree_cpu_seconds(proc: psutil.Process) -> float:
    """Total user + system CPU seconds consumed by a process tree so far."""
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0.0
    total = 0.0
    for p in procs:
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.Error:
            continue
    return total


def _kill_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.Error:
        pass


def _run_blocking(
    script: Path,
    timeout_s: int,
    max_mem_mb: int,
    forward: Callable[[dict[str, Any]], None] | None,
) -> ExecutionResult:
    env = {**os.environ, "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    metrics_file = script.parent / "metrics.json"
    metrics_file.unlink(missing_ok=True)
    start = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", script.name],
        cwd=script.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    ps = psutil.Process(proc.pid)
    stderr_parts: list[str] = []
    stdout_lines: deque[str] = deque(maxlen=500)
    state = {"killed": None, "peak_mb": 0.0}

    def read_stderr() -> None:
        stderr_parts.append(proc.stderr.read())

    def watchdog() -> None:
        last_sample, last_cpu = time.perf_counter(), 0.0
        while proc.poll() is None:
            rss = _tree_rss_mb(ps)
            state["peak_mb"] = max(state["peak_mb"], rss)
            now = time.perf_counter()
            if forward is not None and now - last_sample >= TELEMETRY_S:
                cpu = _tree_cpu_seconds(ps)
                cores = max(0.0, (cpu - last_cpu) / (now - last_sample)) if last_cpu else 0.0
                forward({"telemetry": True, "t": round(now - start, 1), "rss_mb": round(rss, 1),
                         "cores": round(cores, 2)})  # fmt: skip
                last_sample, last_cpu = now, cpu
            if time.perf_counter() - start > timeout_s:
                state["killed"] = "timeout"
            elif max_mem_mb and rss > max_mem_mb:
                state["killed"] = "memory"
            if state["killed"]:
                _kill_tree(proc.pid)
                return
            time.sleep(POLL_S)

    threads = [
        threading.Thread(target=read_stderr, daemon=True),
        threading.Thread(target=watchdog, daemon=True),
    ]
    for t in threads:
        t.start()
    for line in proc.stdout:
        if line.startswith(PROGRESS_PREFIX) and forward is not None:
            try:
                forward(json.loads(line[len(PROGRESS_PREFIX) :]))
            except json.JSONDecodeError:
                stdout_lines.append(line)
        else:
            stdout_lines.append(line)
    proc.wait()
    for t in threads:
        t.join(timeout=5)

    duration = round(time.perf_counter() - start, 3)
    stderr = "".join(stderr_parts)
    if state["killed"] == "timeout":
        stderr += f"\nKilled: exceeded the {timeout_s}s time limit."
    elif state["killed"] == "memory":
        stderr += (
            f"\nKilled: exceeded the {max_mem_mb} MB memory limit. Use a lower fidelity or a leaner model."
        )

    metrics = None
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = None
    if proc.returncode == 0 and metrics is None:
        stderr += "\nScript finished but did not write a valid metrics.json"
    return ExecutionResult(
        ok=proc.returncode == 0 and metrics is not None and state["killed"] is None,
        returncode=proc.returncode,
        duration_s=duration,
        stdout=_tail("".join(stdout_lines)),
        stderr=_tail(stderr),
        metrics=metrics,
        timed_out=state["killed"] == "timeout",
        memory_exceeded=state["killed"] == "memory",
        peak_memory_mb=round(state["peak_mb"], 1),
    )


async def run_script(
    code: str,
    workdir: Path,
    *,
    filename: str = "train.py",
    timeout_s: int = 600,
    max_mem_mb: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> ExecutionResult:
    """Write `code` to `workdir/filename`, run it, and return the outcome.

    Runs in a worker thread (so it works on any event loop, including uvicorn's
    reload mode on Windows). Progress lines are delivered to `on_progress` on
    the caller's event loop as they are printed.

    Args:
        max_mem_mb: Resident-memory ceiling for the process tree (None = 80% of RAM, 0 = unlimited).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / filename
    script.write_text(code, encoding="utf-8")
    if err := static_check(code):
        return ExecutionResult(ok=False, returncode=None, duration_s=0.0, stderr=err)
    limit = default_memory_limit_mb() if max_mem_mb is None else max_mem_mb

    if on_progress is None:
        return await asyncio.to_thread(_run_blocking, script, timeout_s, limit, None)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def forward(item: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    task = asyncio.ensure_future(asyncio.to_thread(_run_blocking, script, timeout_s, limit, forward))
    while not (task.done() and queue.empty()):
        try:
            item = await asyncio.wait_for(queue.get(), timeout=POLL_S)
        except TimeoutError:
            continue
        await on_progress(item)
    return task.result()
