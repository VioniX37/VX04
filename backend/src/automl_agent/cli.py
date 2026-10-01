"""Command-line interface for headless use (laptops, Colab, Kaggle notebooks).

Examples::

    automl-agent ingest --data /kaggle/input/higgs/HIGGS.csv
    automl-agent run --data data/samples/customer_churn.csv --prompt "Predict churn, optimise F1"
    automl-agent run --dataset-id 3f2a... --prompt "..." --set VERIFICATION_MODE=pseudo
    automl-agent datasets
    automl-agent models

Runs started here are stored in the same workspace database as the web UI,
so they can be inspected there afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sqlmodel import Session, select

from automl_agent.config import Settings, get_settings
from automl_agent.schemas.events import AgentEvent
from automl_agent.services.dataset_service import DatasetError, register_path, register_url
from automl_agent.services.event_bus import EventBus
from automl_agent.services.run_service import DatasetRef, execute_run, new_run
from automl_agent.storage.db import DatasetRecord, get_engine, init_db


def _settings_with(overrides: list[str]) -> Settings:
    """Apply ``KEY=VALUE`` overrides (same names as the environment variables)."""
    base = get_settings()
    if not overrides:
        return base
    values = base.model_dump()
    for item in overrides:
        key, _, value = item.partition("=")
        values[key.strip().lower()] = value
    return Settings(**values)


def _register(session: Session, settings: Settings, data: str) -> DatasetRecord:
    if data.startswith(("http://", "https://")):
        return register_url(session, settings, data)
    return register_path(session, settings, data)


def _print_event(event: AgentEvent) -> None:
    if event.kind == "llm":
        return
    mark = {"error": "x", "warning": "!", "artifact": "+"}.get(event.kind, "-")
    print(f"[{event.stage:>14}] {mark} {event.agent}: {event.message}", flush=True)


async def _follow(event_bus: EventBus, run_id: str) -> None:
    async for event in event_bus.subscribe(run_id):
        _print_event(event)


async def _run(settings: Settings, dataset: DatasetRecord, prompt: str) -> int:
    with Session(get_engine(settings.db_url)) as session:
        run = new_run(session, dataset.id, prompt)
        run_id = run.id
    event_bus = EventBus()
    follower = asyncio.create_task(_follow(event_bus, run_id))
    result = await execute_run(
        run_id, DatasetRef.from_record(dataset), prompt, settings=settings, event_bus=event_bus
    )
    await follower
    summary = {
        "run_id": run_id,
        "success": bool(result and result.success),
        "target_met": bool(result and result.target_met),
        "metric": result.task_spec.metric if result and result.task_spec else None,
        "score": (result.metrics or {}).get("score") if result else None,
        "model_family": result.plan.model_family if result and result.plan else None,
        "artifacts": (result.metrics or {}).get("artifact_dir") if result else None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    """Register a dataset and print its id and profile summary."""
    settings = _settings_with(args.set)
    init_db(settings.db_url)
    with Session(get_engine(settings.db_url)) as session:
        rec = _register(session, settings, args.data)
        p = rec.profile
        print(
            json.dumps(
                {
                    "dataset_id": rec.id,
                    "rows": p["n_rows"],
                    "cols": p["n_cols"],
                    "scale_tier": p["scale_tier"],
                    "guessed_target": p["guessed_target"],
                },
                indent=2,
            )
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline on a dataset (registering it first when given by path/URL)."""
    settings = _settings_with(args.set)
    settings.ensure_dirs()
    init_db(settings.db_url)
    with Session(get_engine(settings.db_url)) as session:
        if args.dataset_id:
            dataset = session.get(DatasetRecord, args.dataset_id)
            if dataset is None:
                print(f"dataset {args.dataset_id} not found", file=sys.stderr)
                return 2
        else:
            dataset = _register(session, settings, args.data)
        session.expunge(dataset)
    return asyncio.run(_run(settings, dataset, args.prompt))


def cmd_datasets(args: argparse.Namespace) -> int:
    """List registered datasets."""
    settings = _settings_with(args.set)
    init_db(settings.db_url)
    with Session(get_engine(settings.db_url)) as session:
        for rec in session.exec(select(DatasetRecord).order_by(DatasetRecord.created_at.desc())):
            p = rec.profile
            print(
                f"{rec.id}  {rec.filename:40.40}  {p['n_rows']:>12,} rows  {p['scale_tier']:>6}  {rec.source}"
            )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List Gemini models visible to the configured key and check the configured ones."""
    from automl_agent.llm import create_llm, validate_models

    settings = _settings_with(args.set)
    router = create_llm(settings)
    if router.provider != "gemini":
        print("LLM_PROVIDER is not gemini")
        return 1
    names = asyncio.run(router.for_role("smart").list_models())  # type: ignore[attr-defined]
    for name in sorted(n for n in names if "gemini" in n):
        print(name)
    print(json.dumps(asyncio.run(validate_models(settings)), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all sub-commands."""
    parser = argparse.ArgumentParser(prog="automl-agent", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a setting for this command (repeatable), e.g. --set N_PLANS=2",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Register a dataset from a path or URL")
    p.add_argument("--data", required=True, help="File path or http(s) URL (CSV/TSV/Parquet/JSONL, .gz ok)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("run", help="Run the AutoML pipeline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="File path or URL to register and use")
    src.add_argument("--dataset-id", help="Id of an already registered dataset")
    p.add_argument("--prompt", required=True, help="The task in plain language")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("datasets", help="List registered datasets")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("models", help="List available Gemini models and validate the configuration")
    p.set_defaults(func=cmd_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
