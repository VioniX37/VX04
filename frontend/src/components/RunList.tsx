"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { formatMetric } from "@/lib/stages";
import type { Run } from "@/lib/types";
import { Card, ErrorNote, Spinner, StatusBadge } from "./ui";

/** Table of past runs with status and final score. */
export function RunList() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listRuns().then(setRuns, (e) => setError((e as Error).message));
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

  return (
    <Card className="p-0">
      <table className="w-full text-left text-sm">
        <thead className="text-xs text-muted">
          <tr>
            <th className="px-5 py-3 font-medium">Run</th>
            <th className="px-5 py-3 font-medium">Request</th>
            <th className="px-5 py-3 font-medium">Status</th>
            <th className="px-5 py-3 text-right font-medium">Score</th>
            <th className="px-5 py-3 text-right font-medium">Started</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="border-t border-border hover:bg-surface-muted">
              <td className="px-5 py-3 font-mono text-xs">
                <Link href={`/runs/${r.id}`} className="text-accent hover:underline">
                  {r.id}
                </Link>
              </td>
              <td className="max-w-md truncate px-5 py-3">{r.prompt}</td>
              <td className="px-5 py-3">
                <StatusBadge status={r.status} />
              </td>
              <td className="px-5 py-3 text-right tabular-nums">
                {r.metrics?.score != null ? (
                  <>
                    {formatMetric(r.metrics.score)} <span className="text-xs text-muted">{r.metrics.metric}</span>
                  </>
                ) : (
                  "—"
                )}
              </td>
              <td className="px-5 py-3 text-right text-xs text-muted">{new Date(r.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
