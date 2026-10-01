"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { formatDuration } from "@/lib/run-model";
import { familyLabel, formatMetric, higherIsBetter } from "@/lib/stages";
import type { Run } from "@/lib/types";
import { Card, ErrorNote, LiveDot, Spinner, Stat, StatusBadge } from "./ui";

function duration(r: Run): number | null {
  if (!r.finished_at) return r.status === "running" ? (Date.now() - Date.parse(r.created_at)) / 1000 : null;
  return (Date.parse(r.finished_at) - Date.parse(r.created_at)) / 1000;
}

/** Run history with summary tiles; refreshes itself while any run is in progress. */
export function RunList() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => api.listRuns().then((r) => !cancelled && setRuns(r), (e) => !cancelled && setError((e as Error).message));
    load();
    const id = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (error) return <ErrorNote>{error}</ErrorNote>;
  if (!runs)
    return (
      <p className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> Loading runs…
      </p>
    );
  if (runs.length === 0)
    return (
      <Card>
        <p className="text-sm text-muted">
          No runs yet.{" "}
          <Link href="/" className="text-accent hover:underline">
            Start one
          </Link>
          .
        </p>
      </Card>
    );

  const done = runs.filter((r) => r.status === "succeeded" || r.status === "failed");
  const ok = runs.filter((r) => r.status === "succeeded");
  const live = runs.filter((r) => r.status === "running" || r.status === "pending");
  const calls = runs.reduce((s, r) => s + (r.llm_usage?.calls ?? 0), 0);
  const scored = runs.filter((r) => r.metrics?.score != null);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        <Stat label="runs" value={runs.length} caption={`${live.length} in progress`} />
        <Stat label="success rate" value={done.length ? `${Math.round((ok.length / done.length) * 100)}%` : "—"}
          caption={`${ok.length} of ${done.length} finished`} tone="success" />
        <Stat label="models trained" value={scored.length} caption="scored on held-out test" />
        <Stat label="gemini calls" value={calls.toLocaleString()} caption="all runs" />
      </div>

      <Card className="overflow-hidden p-0">
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-border">
                {["Run", "Request", "Status", "Score", "Model", "Duration", "Calls", "Started"].map((h, i) => (
                  <th key={h} className={`eyebrow px-4 py-3 font-normal ${i >= 3 && i !== 4 ? "text-right" : ""}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const score = r.metrics?.score;
                const metric = r.metrics?.metric;
                const bounded = metric && higherIsBetter(metric) && score != null && score <= 1;
                return (
                  <tr key={r.id} className="border-b border-border/60 transition-colors last:border-0 hover:bg-surface-muted/50">
                    <td className="px-4 py-3 font-mono text-xs">
                      <Link href={`/runs/${r.id}`} className="inline-flex items-center gap-2 text-accent hover:underline">
                        {(r.status === "running" || r.status === "pending") && <LiveDot className="h-1.5 w-1.5" />}
                        {r.id}
                      </Link>
                    </td>
                    <td className="max-w-[16rem] truncate px-4 py-3 text-muted">{r.prompt}</td>
                    <td className="px-4 py-3"><StatusBadge status={r.status} /></td>
                    <td className="px-4 py-3 text-right">
                      {score != null ? (
                        <div className="inline-flex items-center gap-2">
                          {bounded && (
                            <span className="hidden h-1 w-14 overflow-hidden rounded-full bg-surface-muted sm:inline-block">
                              <span className="block h-full rounded-full bg-linear-to-r from-accent to-accent-2" style={{ width: `${score * 100}%` }} />
                            </span>
                          )}
                          <span className="tabular font-medium">{formatMetric(score)}</span>
                          <span className="font-mono text-[10.5px] text-faint">{metric}</span>
                        </div>
                      ) : (
                        <span className="text-faint">—</span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-xs text-muted">{r.plan?.model_family ? familyLabel(r.plan.model_family) : "—"}</td>
                    <td className="tabular px-4 py-3 text-right text-xs text-muted">{formatDuration(duration(r))}</td>
                    <td className="tabular px-4 py-3 text-right text-xs text-muted">{r.llm_usage?.calls ?? "—"}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-right text-xs text-faint">
                      {new Date(r.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
