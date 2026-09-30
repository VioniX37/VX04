"""Benchmark harness: run the full pipeline on a set of tasks and record the paper's metrics.

    python evaluation/run_benchmark.py                       # uses LLM_PROVIDER from .env
    python evaluation/run_benchmark.py --provider fake --repeats 3

Records, per task and repeat: success (pipeline produced working code), whether
the user's target was met, the achieved score, wall-clock time, and LLM
calls/tokens. Writes a CSV + JSON summary to evaluation_results/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from automl_agent.agents import AgentManager, RunContext
from automl_agent.config import get_settings
from automl_agent.llm import create_llm
from automl_agent.services.event_bus import EventBus
from automl_agent.tools import profile_file

HERE = Path(__file__).resolve().parent


async def run_task(task: dict, settings, run_id: str) -> dict:
    csv_path = (HERE / task["path"]).resolve()
    llm = create_llm(settings)
    ctx = RunContext(
        run_id=run_id,
        prompt=task["prompt"],
        dataset_path=csv_path,
        profile=profile_file(csv_path),
        workdir=settings.runs_dir / "benchmarks" / run_id,
        settings=settings,
        llm=llm,
        bus=EventBus(),
    )
    start = time.perf_counter()
    try:
        result = await AgentManager(ctx).run()
        error = result.error
    except Exception as e:
        result, error = None, f"{type(e).__name__}: {e}"
    return {
        "task": task["name"],
        "run_id": run_id,
        "provider": llm.provider,
        "model": llm.model,
        "success": bool(result and result.success),
        "target_met": bool(result and result.target_met),
        "metric": result.task_spec.metric if result and result.task_spec else None,
        "score": (result.metrics or {}).get("score") if result else None,
        "model_family": result.plan.model_family if result and result.plan else None,
        "revisions": len(result.attempts) if result else 0,
        "wall_time_s": round(time.perf_counter() - start, 2),
        "llm_calls": llm.usage.calls,
        "input_tokens": llm.usage.input_tokens,
        "output_tokens": llm.usage.output_tokens,
        "error": error,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=HERE / "benchmarks.json")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--provider", help="override LLM_PROVIDER")
    ap.add_argument("--model", help="override LLM_MODEL")
    args = ap.parse_args()

    settings = get_settings()
    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        settings.gemini_model_smart = settings.gemini_model_fast = args.model
    settings.ensure_dirs()

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    rows = []
    for task in tasks:
        for rep in range(args.repeats):
            run_id = f"{task['name']}-{rep + 1}-{datetime.now():%H%M%S}"
            row = await run_task(task, settings, run_id)
            print(json.dumps(row))
            rows.append(row)

    out_dir = HERE.parent / "evaluation_results"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    with (out_dir / f"results-{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_runs": len(rows),
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "target_met_rate": sum(r["target_met"] for r in rows) / len(rows),
        "avg_llm_calls": sum(r["llm_calls"] for r in rows) / len(rows),
    }
    (out_dir / f"summary-{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("summary:", json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
