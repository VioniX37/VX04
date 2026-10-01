"""Turn benchmark outputs into paper tables and figures.

    python -m evaluation.analysis evaluation_results/smoke-20261001-120000 \
        --figures ../paper/figures --markdown ../docs/research/results.md

Produces:
  * ``table_main.md``      - SR / NPS / CS / wall time / LLM calls per variant and baseline (RQ2, RQ4)
  * ``table_calibration.md`` - Spearman, top-1 hit rate and MAE of predicted vs observed scores (RQ1)
  * ``fig_cs_by_variant.pdf/png`` - comprehensive score per variant with per-task points
  * ``fig_calibration.pdf/png``   - predicted vs observed validation scores at the first grounding rung
  * ``fig_memory_curve.pdf/png``  - LLM calls and CS over the task sequence, memory off vs on (RQ3)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import calibration  # noqa: E402

PALETTE = ["#4f46e5", "#0d9488", "#d97706", "#db2777", "#64748b", "#16a34a", "#9333ea"]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
            elif v in ("", None):
                r[k] = None
            else:
                with contextlib.suppress(ValueError):
                    r[k] = float(v)
    return rows


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table of the headline metrics per variant."""
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    lines = ["| Variant | Runs | SR | NPS | CS | Wall (s) | LLM calls | Tokens |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        lines.append(
            f"| {v} | {len(rs)} | {_mean([r['sr'] for r in rs]):.3f} | {_mean([r['nps'] for r in rs]):.3f} | "
            f"{_mean([r['cs'] for r in rs]):.3f} | {_mean([r['wall_s'] for r in rs]):.1f} | "
            f"{_mean([r['llm_calls'] for r in rs]):.1f} | {_mean([r['tokens'] for r in rs]):,.0f} |"
        )
    return "\n".join(lines)


def calibration_table(obs: list[dict[str, Any]]) -> str:
    """Markdown table of RQ1 calibration metrics per variant."""
    lines = ["| Variant | Groups | Spearman ρ | Top-1 hit | MAE |", "|---|---:|---:|---:|---:|"]
    for v in dict.fromkeys(o["variant"] for o in obs):
        c = calibration(o for o in obs if o["variant"] == v)
        if not c["groups"]:
            continue

        def fmt(x: float | None) -> str:
            return "–" if x is None else f"{x:.3f}"

        lines.append(
            f"| {v} | {c['groups']} | {fmt(c['spearman'])} | {fmt(c['top1_hit'])} | {fmt(c['mae'])} |"
        )
    return "\n".join(lines)


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cs_by_variant(rows: list[dict[str, Any]], out: Path) -> None:
    """Bar of mean CS per variant with one dot per task."""
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    fig, ax = plt.subplots(figsize=(max(4, 1.1 * len(variants)), 3))
    for i, v in enumerate(variants):
        rs = [r for r in rows if r["variant"] == v]
        ax.bar(i, _mean([r["cs"] for r in rs]), color=PALETTE[i % len(PALETTE)], alpha=0.85, width=0.6)
        ax.scatter([i] * len(rs), [r["cs"] for r in rs], color="black", s=8, zorder=3)
    ax.set_xticks(range(len(variants)), variants, rotation=20, ha="right")
    ax.set_ylabel("Comprehensive score (CS)")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig_cs_by_variant")


def fig_calibration(obs: list[dict[str, Any]], out: Path) -> None:
    """Scatter of predicted vs observed validation scores (bounded metrics, first rung)."""
    loss = {"rmse", "mae", "mape", "rmsle"}
    pts = [
        o
        for o in obs
        if not o.get("final")
        and o.get("observed_score") is not None
        and o.get("predicted_score") is not None
        and o.get("metric") not in loss
    ]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.scatter(
        [p["predicted_score"] for p in pts],
        [p["observed_score"] for p in pts],
        s=12,
        color=PALETTE[0],
        alpha=0.7,
    )
    lo = min(min(p["predicted_score"] for p in pts), min(p["observed_score"] for p in pts))
    ax.plot([lo, 1], [lo, 1], ls="--", lw=1, color="#94a3b8")
    ax.set_xlabel("Predicted score (pseudo-execution)")
    ax.set_ylabel("Observed validation score")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig_calibration")


def fig_memory_curve(rows: list[dict[str, Any]], out: Path, pairs: list[tuple[str, str]]) -> None:
    """LLM calls per run over the task sequence, for (without, with) memory variant pairs."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    drawn = False
    for i, (off, on) in enumerate(pairs):
        for name, style in ((off, "--"), (on, "-")):
            rs = [r for r in rows if r["variant"] == name]
            if not rs:
                continue
            ax.plot(range(1, len(rs) + 1), [r["llm_calls"] for r in rs], style, marker="o", ms=3,
                    color=PALETTE[i % len(PALETTE)], label=name)  # fmt: skip
            drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("Task index (sequence order)")
    ax.set_ylabel("LLM calls")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig_memory_curve")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--figures", type=Path, help="also write figures here (e.g. ../paper/figures)")
    ap.add_argument("--markdown", type=Path, help="write a results page (e.g. ../docs/research/results.md)")
    args = ap.parse_args(argv)

    rows = _read_csv(args.results_dir / "results.csv")
    obs = _read_csv(args.results_dir / "observations.csv")
    config = json.loads((args.results_dir / "config.json").read_text(encoding="utf-8"))
    tables = {"table_main.md": main_table(rows), "table_calibration.md": calibration_table(obs)}
    for name, text in tables.items():
        (args.results_dir / name).write_text(text + "\n", encoding="utf-8")

    pairs = [("paper", "memory"), ("grounded", "full")]
    for out in filter(None, [args.results_dir / "figures", args.figures]):
        fig_cs_by_variant(rows, out)
        fig_calibration(obs, out)
        fig_memory_curve(rows, out, pairs)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            f"# Results: {config['name']}\n\n{config.get('description', '')}\n\n"
            f"## Main results\n\n{tables['table_main.md']}\n\n"
            f"## Calibration of pseudo-execution (RQ1)\n\n{tables['table_calibration.md']}\n",
            encoding="utf-8",
        )
    print(tables["table_main.md"])
    print()
    print(tables["table_calibration.md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
