"use client";

import { formatDuration } from "@/lib/run-model";
import { useNow } from "@/lib/use-now";
import { cn } from "../ui";

export type IngestPhase = "upload" | "process" | "ready" | "error";

const STEPS = [
  { id: "upload", label: "Transfer", hint: "streamed to the server" },
  { id: "convert", label: "Parquet", hint: "streaming conversion, zstd" },
  { id: "profile", label: "Profile", hint: "types, missing values, tiers" },
  { id: "ready", label: "Ready", hint: "registered" },
] as const;

/**
 * Visualises dataset registration. Conversion and profiling happen in one server request,
 * so they animate together until it returns.
 */
export function IngestProgress({
  phase,
  startedAt,
  uploadFraction,
  viaPath,
}: {
  phase: IngestPhase;
  startedAt: number;
  uploadFraction: number | null;
  viaPath: boolean;
}) {
  const now = useNow(phase === "upload" || phase === "process", 500);
  const state = (i: number): "done" | "active" | "pending" | "skipped" => {
    if (i === 0 && viaPath) return "skipped";
    if (phase === "ready") return "done";
    if (phase === "upload") return i === 0 ? "active" : "pending";
    if (phase === "process") return i === 0 ? "done" : i <= 2 ? "active" : "pending";
    return "pending";
  };
  return (
    <div className="rounded-xl border border-border bg-surface-muted/30 p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="eyebrow">dataset ingest</p>
        <p className="tabular font-mono text-[11px] text-muted">
          {phase === "ready" ? "done" : formatDuration((now - startedAt) / 1000)}
        </p>
      </div>
      <ol className="grid grid-cols-4 gap-2">
        {STEPS.map((s, i) => {
          const st = state(i);
          return (
            <li key={s.id} className="min-w-0">
              <div
                className={cn(
                  "h-1.5 overflow-hidden rounded-full bg-surface-muted",
                  st === "active" && i !== 0 && "shimmer",
                )}
              >
                <div
                  className="h-full rounded-full bg-linear-to-r from-accent to-accent-2 transition-all duration-300"
                  style={{
                    width:
                      st === "done" ? "100%" : st === "active" && i === 0 ? `${(uploadFraction ?? 0) * 100}%` : st === "active" ? "35%" : "0%",
                    opacity: st === "skipped" ? 0 : 1,
                  }}
                />
              </div>
              <p className={cn("mt-1.5 text-xs font-medium", st === "pending" || st === "skipped" ? "text-faint" : "text-foreground")}>
                {s.label}
                {st === "active" && i === 0 && uploadFraction !== null ? ` · ${Math.round(uploadFraction * 100)}%` : ""}
              </p>
              <p className="truncate text-[10.5px] text-faint">{st === "skipped" ? "read in place" : s.hint}</p>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
