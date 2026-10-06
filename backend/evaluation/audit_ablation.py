"""Leakage ablation: does the pre-training audit keep a planted leak out of the selected model?

    python -m evaluation.audit_ablation ../data/samples/customer_churn_leaky.csv \
        --prompt "Predict customer churn. Optimise ROC AUC." --leak cancellation_confirmation

Runs the full pipeline twice on the same data, with ``DATA_AUDIT`` off and on, using the
offline backend (planning is a deterministic prior, so the comparison isolates the audit).
Reports the test score, whether the leaking column was dropped, and its share of the
selected model's feature importance.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from automl_agent.agents import AgentManager, RunContext
from automl_agent.config import Settings
from automl_agent.llm import create_llm
from automl_agent.services.event_bus import EventBus
from automl_agent.tools import profile_file


def run_once(data: Path, prompt: str, audit: bool) -> dict:
    """One offline pipeline run with the audit on or off."""
    settings = Settings(
        workspace_dir=Path(tempfile.mkdtemp()),
        llm_provider="fake",
        data_audit=audit,
        max_revisions=0,
        exec_timeout_s=600,
    )
    settings.ensure_dirs()
    ctx = RunContext(
        run_id=f"audit-{'on' if audit else 'off'}",
        prompt=prompt,
        dataset_path=data,
        profile=profile_file(data),
        workdir=settings.runs_dir / "run",
        settings=settings,
        llm=create_llm(settings),
        bus=EventBus(),
    )
    result = asyncio.run(AgentManager(ctx).run())
    card = (result.metrics or {}).get("model_card") or {}
    return {
        "success": result.success,
        "metric": result.task_spec.metric if result.task_spec else None,
        "score": (result.metrics or {}).get("score"),
        "dropped": list(result.task_spec.drop_columns) if result.task_spec else [],
        "importance": {f["feature"]: f["importance"] for f in card.get("feature_importances", [])},
    }


def main() -> None:
    """Command-line entry point: print a Markdown table comparing audit off and on."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--leak", required=True, help="the planted leaking column")
    args = parser.parse_args()
    print("| Audit | Test score | Leak dropped | Leak importance share |")
    print("|---|---|---|---|")
    for audit in (False, True):
        r = run_once(args.data, args.prompt, audit)
        share = r["importance"].get(args.leak, 0.0)
        print(
            f"| {'on' if audit else 'off'} | {r['metric']} = {r['score']:.4f} | "
            f"{'yes' if args.leak in r['dropped'] else 'no'} | {share:.1%} |"
        )


if __name__ == "__main__":
    main()
