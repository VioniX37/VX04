"use client";

import { formatDuration } from "@/lib/run-model";
import { formatBytes } from "@/lib/stages";
import type { IngestJob } from "@/lib/types";
import { useNow } from "@/lib/use-now";
import { Badge, cn, LiveDot } from "../ui";

/** Client-side state of a browser upload, before the server-side job exists. */
export interface UploadState {
  filename: string;
  loaded: number;
  total: number;
}

const LABELS: Record<string, string> = {
  upload: "Upload",
  download: "Download",
  decompress: "Decompress",
  convert: "Convert to Parquet",
  profile: "Profile columns",
};

/** Relative share of the total time each phase usually takes, for the overall bar. */
const WEIGHTS: Record<string, number> = { upload: 2, download: 2, decompress: 1, convert: 4, profile: 1 };

type StepState = "done" | "active" | "pending" | "failed";

/**
 * Live dataset registration: one bar per phase with real progress from the server
 * (rows converted, columns profiled), an overall bar, time left in the current phase, and any error.
 */
export function IngestProgress({
  name,
  job,
  upload,
  startedAt,
  error,
}: {
  name: string;
  job: IngestJob | null;
  upload: UploadState | null;
  startedAt: number;
  error: string | null;
}) {
  const failed = Boolean(error) || job?.status === "failed";
  const done = job?.status === "succeeded";
  const running = !failed && !done;
  const now = useNow(running, 500);

  const steps = job?.steps ?? (upload ? ["upload", "convert", "profile"] : ["convert", "profile"]);
  const phase = job ? job.phase : "upload";
  const phaseIndex = done ? steps.length : phase === "save" ? steps.length - 1 : Math.max(steps.indexOf(phase), 0);
  const fractionOf = (i: number): number | null => {
    if (done || i < phaseIndex) return 1;
    if (i > phaseIndex) return 0;
    if (phase === "save") return 1;
    if (!job) return upload && upload.total ? upload.loaded / upload.total : 0;
    return job.fraction;
  };
  const stateOf = (i: number): StepState =>
    i < phaseIndex || done ? "done" : i === phaseIndex ? (failed ? "failed" : "active") : "pending";

  const totalWeight = steps.reduce((s, id) => s + (WEIGHTS[id] ?? 1), 0);
  const overall = steps.reduce((s, id, i) => s + (WEIGHTS[id] ?? 1) * (fractionOf(i) ?? 0), 0) / totalWeight;

  const elapsed = job && !running ? job.elapsed_s : (now - startedAt) / 1000;
  const current = fractionOf(phaseIndex);
  const phaseElapsed = job ? job.phase_elapsed_s : elapsed;
  const eta = running && current !== null && current > 0.03 && phaseElapsed > 2 ? (phaseElapsed * (1 - current)) / current : null;

  const message = failed
    ? null
    : done
      ? job?.message
      : !job && upload
        ? `${formatBytes(upload.loaded)} of ${formatBytes(upload.total)}`
        : job?.message;
  const filename = job?.filename ?? name;

  return (
    <div
      className={cn(
        "rounded-xl border p-3.5",
        failed ? "border-danger/40 bg-danger-soft/40" : "border-border bg-surface-muted/30",
      )}
      role="status"
      aria-live="polite"
    >
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          {running && <LiveDot className="h-1.5 w-1.5 text-accent-2" />}
          <p className="truncate text-sm font-medium">{filename}</p>
          <Badge tone={failed ? "danger" : done ? "success" : "accent"}>
            {failed ? "Failed" : done ? "Ready" : "Registering"}
          </Badge>
        </div>
        <p className="tabular shrink-0 font-mono text-[11px] text-muted">
          {Math.round(overall * 100)}% · {formatDuration(elapsed)}
        </p>
      </div>

      <div className="mb-3 h-2 overflow-hidden rounded-full bg-surface-muted">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            failed ? "bg-danger" : "bg-linear-to-r from-accent to-accent-2",
          )}
          style={{ width: `${Math.max(overall * 100, running ? 2 : 0)}%` }}
        />
      </div>

      <ol className="grid gap-2" style={{ gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` }}>
        {steps.map((id, i) => {
          const st = stateOf(i);
          const f = fractionOf(i);
          return (
            <li key={id} className="min-w-0">
              <div className={cn("h-1 overflow-hidden rounded-full bg-surface-muted", st === "active" && f === null && "shimmer")}>
                <div
                  className={cn("h-full rounded-full transition-all duration-500", st === "failed" ? "bg-danger" : "bg-accent")}
                  style={{ width: `${(f ?? 0) * 100}%` }}
                />
              </div>
              <p className={cn("mt-1.5 truncate text-xs font-medium", st === "pending" ? "text-faint" : st === "failed" ? "text-danger" : "text-foreground")}>
                {LABELS[id] ?? id}
              </p>
              <p className="tabular text-[10.5px] text-faint">
                {st === "done" ? "done" : st === "pending" ? "waiting" : st === "failed" ? "failed" : f === null ? "working…" : `${Math.round(f * 100)}%`}
              </p>
            </li>
          );
        })}
      </ol>

      {message && (
        <p className="tabular mt-3 truncate font-mono text-[11px] text-muted">
          {phase === "save" ? "Saving dataset record" : message}
          {eta !== null && ` · ~${formatDuration(eta)} left`}
        </p>
      )}
      {failed && (
        <p className="mt-3 break-words rounded-lg border border-danger/25 bg-danger-soft px-3 py-2 text-xs text-danger">
          {error ?? job?.error ?? "Registration failed."}
        </p>
      )}
    </div>
  );
}
