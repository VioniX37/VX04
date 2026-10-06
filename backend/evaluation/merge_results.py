"""Assemble the paper's final result set from several benchmark output directories.

    python -m evaluation.merge_results evaluation/results/local_large \
        --take evaluation_results/local_large_v2 --except-runs full:microsoft_malware \
        --except-variants baseline:zero_shot \
        --take evaluation_results/local_large --only-runs full:microsoft_malware,baseline:optuna_lgbm:*

Each ``--take`` names a source directory (with ``results.csv`` and ``observations.csv``);
the filters after it select rows by ``variant:task`` (``*`` matches any task). Rows from
later sources are added after earlier ones. Every row gets a ``source`` column, and the
selection is written to ``PROVENANCE.md`` so the merge is auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def _key_matches(row: dict[str, str], patterns: list[str]) -> bool:
    for pattern in patterns:
        variant, _, task = pattern.rpartition(":")
        if row.get("variant") == variant and task in ("*", row.get("task")):
            return True
    return False


def _selected(row: dict[str, str], spec: dict[str, list[str]]) -> bool:
    if spec["only_runs"] and not _key_matches(row, spec["only_runs"]):
        return False
    if _key_matches(row, spec["except_runs"]):
        return False
    return row.get("variant") not in spec["except_variants"]


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse(argv: list[str]) -> tuple[Path, list[dict]]:
    """Parse ``OUT (--take DIR [filters])+`` into the output path and per-source specs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", type=Path)
    args, rest = parser.parse_known_args(argv)
    sources: list[dict] = []
    flags = {
        "--only-runs": "only_runs",
        "--except-runs": "except_runs",
        "--except-variants": "except_variants",
    }
    i = 0
    while i < len(rest):
        if rest[i] == "--take":
            sources.append(
                {"dir": Path(rest[i + 1]), "only_runs": [], "except_runs": [], "except_variants": []}
            )
        elif rest[i] in flags and sources:
            sources[-1][flags[rest[i]]] = rest[i + 1].split(",")
        else:
            parser.error(f"unexpected argument {rest[i]!r}")
        i += 2
    if not sources:
        parser.error("at least one --take DIR is required")
    return args.out, sources


def main(argv: list[str] | None = None) -> None:
    """Merge the selected rows and write results, observations, config and provenance."""
    out, sources = parse(sys.argv[1:] if argv is None else argv)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    observations: list[dict[str, str]] = []
    lines = ["# Provenance", "", "Rows of this result set, by source directory:", ""]
    for spec in sources:
        taken = [r for r in _read(spec["dir"] / "results.csv") if _selected(r, spec)]
        keys = {(r["variant"], r["task"], r.get("seed", "")) for r in taken}
        obs = [
            o
            for o in _read(spec["dir"] / "observations.csv")
            if (o.get("variant"), o.get("task"), o.get("seed", "")) in keys
        ]
        for row in (*taken, *obs):
            row["source"] = spec["dir"].name
        results.extend(taken)
        observations.extend(obs)
        lines.append(
            f"- `{spec['dir'].as_posix()}`: " + ", ".join(f"{r['variant']} / {r['task']}" for r in taken)
        )
        if (spec["dir"] / "config.json").exists() and not (out / "config.json").exists():
            shutil.copy(spec["dir"] / "config.json", out / "config.json")
    _write(out / "results.csv", results)
    _write(out / "observations.csv", observations)
    (out / "PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results": len(results), "observations": len(observations), "out": str(out)}))


if __name__ == "__main__":
    main()
