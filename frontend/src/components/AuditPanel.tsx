"use client";

import type { AuditFinding, AuditReport } from "@/lib/types";
import { Badge, Card, CardTitle } from "./ui";

interface AuditPanelProps {
  audit: AuditReport;
}

export function AuditPanel({ audit }: AuditPanelProps) {
  const { findings, dropped_columns, train_test_duplicates, duplicate_pct, has_leakage } = audit;

  const severityTone = (severity: string): "neutral" | "accent" | "success" | "warning" | "danger" => {
    switch (severity) {
      case "critical":
        return "danger";
      case "high":
        return "warning";
      case "medium":
        return "accent";
      default:
        return "neutral";
    }
  };

  return (
    <Card>
      <CardTitle
        eyebrow="data audit · pre-training verification"
        aside={
          has_leakage ? (
            <Badge tone="danger">Target Leakage Detected</Badge>
          ) : (
            <Badge tone="success">Clean Dataset</Badge>
          )
        }
      >
        Pre-Training Data Audit
      </CardTitle>

      <div className="space-y-4">
        {dropped_columns.length > 0 && (
          <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3 text-xs text-red-400">
            <span className="font-semibold">Auto-Dropped Leaky Columns: </span>
            {dropped_columns.map((c) => (
              <span key={c} className="mr-1.5 inline-block font-mono bg-red-500/10 px-1.5 py-0.5 rounded">
                {c}
              </span>
            ))}
          </div>
        )}

        {train_test_duplicates > 0 && (
          <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-3 text-xs text-yellow-400">
            <span className="font-semibold">Split Leakage Warning: </span>
            {train_test_duplicates.toLocaleString()} test rows ({duplicate_pct.toFixed(2)}%) duplicate training rows exactly.
          </div>
        )}

        {findings.length > 0 ? (
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-muted/50 text-muted uppercase tracking-wider font-semibold border-b border-border">
                <tr>
                  <th className="py-2 px-3">Severity</th>
                  <th className="py-2 px-3">Check</th>
                  <th className="py-2 px-3">Column</th>
                  <th className="py-2 px-3">Issue Description</th>
                  <th className="py-2 px-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {findings.map((f: AuditFinding, idx: number) => (
                  <tr key={`${f.check}-${f.column ?? idx}`} className="hover:bg-surface-muted/30">
                    <td className="py-2.5 px-3">
                      <Badge tone={severityTone(f.severity)} className="uppercase text-[10px]">
                        {f.severity}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-3 font-mono text-muted">{f.check}</td>
                    <td className="py-2.5 px-3 font-mono font-medium text-foreground">
                      {f.column ?? "—"}
                    </td>
                    <td className="py-2.5 px-3 text-muted max-w-xs">{f.message}</td>
                    <td className="py-2.5 px-3">
                      <Badge tone={f.suggested_action === "drop" ? "danger" : "neutral"}>
                        {f.suggested_action}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-muted">
            All checks passed.
          </p>
        )}
      </div>
    </Card>
  );
}
