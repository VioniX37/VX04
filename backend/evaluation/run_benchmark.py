"""Benchmark harness: run the ablation matrix of a config and record the paper's metrics.

    cd backend
    python -m evaluation.run_benchmark --config evaluation/configs/smoke.json
    python -m evaluation.run_benchmark --config evaluation/configs/paper_tabular.json --variants grounded,full

A config lists datasets (with ground-truth task specs and one or more prompts),
pipeline variants (setting overrides such as VERIFICATION_MODE / MEMORY_ENABLED),
seeds and baselines. Each variant gets its own workspace, so experience memory
accumulates across the tasks of that variant only, in the listed task order. The
LLM response cache is shared across variants (keyed by seed), so identical
requests are paid for once.

Outputs (in ``--out``, default ``evaluation_results/<config>-<timestamp>/``):
``results.csv`` (one row per run), ``observations.csv`` (predicted vs observed
per plan and fidelity), ``summary.json`` and a copy of the config.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from automl_agent.config import Settings, get_settings
from automl_agent.schemas.dataset import DatasetProfile
from automl_agent.schemas.task_spec import TaskSpec, TaskType
from automl_agent.services.dataset_service import register_path
from automl_agent.services.event_bus import EventBus
from automl_agent.services.run_service import DatasetRef, execute_run, new_run
from automl_agent.storage.db import DatasetRecord, PlanObservation, RunRecord, get_engine, init_db
from automl_agent.tools.splits import ensure_split

from .baselines.runners import run_optuna_lgbm, run_zero_shot
from .metrics import calibration, comprehensive_score, normalized_performance, success_rate

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent

RESULT_FIELDS = [
    "kind", "variant", "task", "prompt_kind", "seed", "run_id", "success", "target_met", "sr", "nps", "cs",
    "metric", "score", "model_family", "spec_correct", "revisions", "stop_reason", "wall_s", "llm_calls",
    "cache_hits", "tokens", "memory_hits", "error",
]  # fmt: skip


def load_config(path: Path) -> dict[str, Any]:
    """Read a benchmark config and resolve dataset paths relative to it."""
    config = json.loads(path.read_text(encoding="utf-8"))
    for task in config["datasets"]:
        task["path"] = str((path.parent / task["path"]).resolve())
    return config


def make_settings(base: Settings, overrides: dict[str, Any]) -> Settings:
    """Copy `base` with ``KEY=value`` overrides (environment-variable names)."""
    values = base.model_dump()
    values.update({k.lower(): v for k, v in overrides.items()})
    return Settings(**values)


def truth_spec(task: dict[str, Any], metric_target: float | None) -> TaskSpec:
    """Ground-truth task spec of a benchmark task."""
    return TaskSpec(**{**task["truth"], "metric_target": metric_target})


def _prompts(task: dict[str, Any]) -> list[tuple[str, str, float | None]]:
    out = []
    for kind, p in task["prompts"].items():
        if isinstance(p, str):
            out.append((kind, p, None))
        else:
            out.append((kind, p["text"], p.get("metric_target")))
    return out


def _dataset(settings: Settings, cache: dict[str, DatasetRecord], task: dict[str, Any]) -> DatasetRecord:
    if task["name"] not in cache:
        with Session(get_engine(settings.db_url)) as session:
            rec = register_path(session, settings, task["path"], task["name"])
            session.expunge(rec)
        cache[task["name"]] = rec
    return cache[task["name"]]


def _score_row(row: dict[str, Any], metric: str | None, metrics: dict[str, Any] | None) -> dict[str, Any]:
    row["sr"] = success_rate(bool(row.get("success")), bool(row.get("target_met")))
    row["nps"] = normalized_performance(metric, row.get("score"), metrics)
    row["cs"] = comprehensive_score(row["sr"], row["nps"])
    return row


async def run_pipeline_task(
    settings: Settings, dataset: DatasetRecord, task: dict[str, Any], prompt: str, metric_target: float | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the agent pipeline once and return (result row, observation rows)."""
    with Session(get_engine(settings.db_url)) as session:
        run_id = new_run(session, dataset.id, prompt).id
    start = time.perf_counter()
    ref = DatasetRef.from_record(dataset)
    result = await execute_run(run_id, ref, prompt, settings=settings, event_bus=EventBus())
    wall = round(time.perf_counter() - start, 2)
    with Session(get_engine(settings.db_url)) as session:
        record = session.get(RunRecord, run_id)
        usage = (record.llm_usage if record else None) or {}
        query = select(PlanObservation).where(PlanObservation.run_id == run_id)
        obs = [o.model_dump(mode="json") for o in session.exec(query)]

    truth = truth_spec(task, metric_target)
    spec = result.task_spec if result else None
    spec_correct = bool(
        spec and spec.target_column == truth.target_column and spec.task_type == truth.task_type
    )
    metrics = (result.metrics or {}) if result else {}
    sources = result.knowledge_sources if result else []
    row = {
        "kind": "pipeline",
        "run_id": run_id,
        "success": bool(result and result.success),
        "target_met": bool(result and result.target_met),
        "metric": spec.metric if spec else truth.metric,
        "score": metrics.get("score"),
        "model_family": result.plan.model_family if result and result.plan else None,
        "spec_correct": spec_correct,
        "revisions": len(result.attempts) if result else 0,
        "stop_reason": result.stop_reason if result else "crashed",
        "wall_s": wall,
        "llm_calls": usage.get("calls", 0),
        "cache_hits": usage.get("cache_hits", 0),
        "tokens": usage.get("total_tokens", 0),
        "memory_hits": sum(1 for s in sources if s.startswith("memory:")),
        "error": result.error if result else "crashed",
    }
    return _score_row(row, row["metric"], metrics.get("metrics")), obs


async def run_baseline(
    name: str,
    settings: Settings,
    dataset: DatasetRecord,
    task: dict[str, Any],
    prompt: str,
    metric_target: float | None,
    workdir: Path,
    budget_s: float,
) -> dict[str, Any] | None:
    """Run one baseline on one task/prompt (None when the baseline does not apply)."""
    truth = truth_spec(task, metric_target)
    data_path = Path(dataset.path)
    split = ensure_split(
        data_path,
        truth.target_column,
        stratify=truth.task_type != TaskType.tabular_regression,
        valid_fraction=settings.split_valid_fraction,
        test_fraction=settings.split_test_fraction,
        seed=settings.split_seed,
    )
    if name == "zero_shot":
        profile = DatasetProfile.model_validate(dataset.profile)
        out = await run_zero_shot(
            settings,
            spec=truth,
            prompt=prompt,
            profile=profile,
            data_path=data_path,
            split=split,
            workdir=workdir,
        )
    elif name == "optuna_lgbm":
        out = await run_optuna_lgbm(
            settings, spec=truth, data_path=data_path, split=split, workdir=workdir, time_budget_s=budget_s
        )
    else:
        raise ValueError(f"unknown baseline {name}")
    if out is None:
        return None
    row = {
        "kind": "baseline",
        "run_id": "",
        "metric": truth.metric,
        "spec_correct": True,
        "revisions": 0,
        "stop_reason": "",
        "memory_hits": 0,
        **out,
    }
    return _score_row(row, truth.metric, out.get("metrics"))


def summarize(rows: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean metrics per variant/baseline, plus calibration per variant."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["variant"], []).append(r)
    summary: dict[str, Any] = {}
    for name, rs in groups.items():

        def mean(key: str, rs: list[dict[str, Any]] = rs) -> float:
            return round(sum(float(r.get(key) or 0) for r in rs) / len(rs), 4)

        summary[name] = {
            "runs": len(rs),
            "SR": mean("sr"),
            "NPS": mean("nps"),
            "CS": mean("cs"),
            "wall_s": mean("wall_s"),
            "llm_calls": mean("llm_calls"),
            "tokens": mean("tokens"),
            "spec_accuracy": mean("spec_correct"),
            "calibration": calibration(o for o in observations if o.get("variant") == name),
        }
    return summary


def _variant_settings(
    base: Settings, overrides: dict[str, Any], workspace: Path, cache: Path, seed: int
) -> Settings:
    settings = make_settings(
        base,
        {
            **overrides,
            "WORKSPACE_DIR": str(workspace),
            "LLM_CACHE_ROOT": str(cache),
            "LLM_CACHE_NAMESPACE": f"seed{seed}",
        },
    )
    settings.ensure_dirs()
    init_db(settings.db_url)
    return settings


async def main_async(args: argparse.Namespace) -> Path:
    """Run the benchmark described by the CLI arguments; returns the output directory."""
    config = load_config(args.config)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (BACKEND.parent / "evaluation_results" / f"{config['name']}-{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    base = get_settings()
    global_overrides = {**config.get("settings", {}), **dict(kv.split("=", 1) for kv in args.set)}
    cache_root = out / "llm_cache" if args.isolated_cache else BACKEND / "workspace" / "benchmark_llm_cache"
    variants = [v for v in config["variants"] if not args.variants or v["name"] in args.variants]
    tasks = [t for t in config["datasets"] if not args.tasks or t["name"] in args.tasks]
    seeds = config.get("seeds", [0])
    budget_s = float(global_overrides.get("BUDGET_WALL_S") or config.get("baseline_budget_s", 300))

    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    def record(row: dict[str, Any], **extra: Any) -> None:
        row.update(extra)
        rows.append(row)
        keys = ("variant", "task", "prompt_kind", "seed", "cs", "score", "error")
        print(json.dumps({k: row.get(k) for k in keys}), flush=True)

    for variant in variants:
        for seed in seeds:
            workspace = out / "workspaces" / variant["name"] / f"seed{seed}"
            overrides = {**global_overrides, **variant.get("settings", {})}
            settings = _variant_settings(base, overrides, workspace, cache_root, seed)
            datasets: dict[str, DatasetRecord] = {}
            for task in tasks:
                dataset = _dataset(settings, datasets, task)
                for kind, prompt, target in _prompts(task):
                    row, obs = await run_pipeline_task(settings, dataset, task, prompt, target)
                    for o in obs:
                        o.update(variant=variant["name"], task=task["name"], seed=seed, prompt_kind=kind)
                    observations.extend(obs)
                    record(row, variant=variant["name"], task=task["name"], prompt_kind=kind, seed=seed)

    for name in [] if args.no_baselines else config.get("baselines", []):
        for seed in seeds:
            workspace = out / "workspaces" / f"baseline-{name}" / f"seed{seed}"
            settings = _variant_settings(base, global_overrides, workspace, cache_root, seed)
            datasets = {}
            for task in tasks:
                dataset = _dataset(settings, datasets, task)
                for kind, prompt, target in _prompts(task):
                    workdir = settings.runs_dir / f"{task['name']}-{kind}"
                    row = await run_baseline(name, settings, dataset, task, prompt, target, workdir, budget_s)
                    if row is not None:
                        record(
                            row, variant=f"baseline:{name}", task=task["name"], prompt_kind=kind, seed=seed
                        )

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if observations:
        fields = sorted({k for o in observations for k in o})
        with (out / "observations.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(observations)
    summary = summarize(rows, observations)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"results written to {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=HERE / "configs" / "smoke.json")
    ap.add_argument("--out", type=Path, help="output directory")
    ap.add_argument("--variants", type=lambda s: s.split(","), help="comma-separated subset of variants")
    ap.add_argument("--tasks", type=lambda s: s.split(","), help="comma-separated subset of tasks")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="global setting override")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--isolated-cache", action="store_true", help="do not reuse the shared LLM cache")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    asyncio.run(main_async(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
