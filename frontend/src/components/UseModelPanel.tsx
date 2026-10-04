"use client";

import { useCallback, useRef, useState } from "react";

import { api, ApiError } from "@/lib/api";
import type { BundleSchema, PredictResult, Run } from "@/lib/types";
import { Button, Card, CardTitle, ErrorNote, Spinner } from "./ui";

/** The "Use this model" panel shown on a finished, succeeded run. */
export function UseModelPanel({ run }: { run: Run }) {
  if (run.status !== "succeeded") return null;

  return (
    <Card>
      <CardTitle eyebrow="deployment">Use this model</CardTitle>
      <div className="space-y-6">
        <SinglePredictSection runId={run.id} run={run} />
        <hr className="border-border" />
        <BatchScoreSection runId={run.id} />
        <hr className="border-border" />
        <BundleDownloadSection runId={run.id} />
      </div>
    </Card>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Single prediction form
// ─────────────────────────────────────────────────────────────────────────────

function SinglePredictSection({ runId, run }: { runId: string; run: Run }) {
  const spec = run.task_spec;
  // Build the list of input fields from the task spec.
  const target = spec?.target_column ?? null;
  const drop = new Set(spec?.drop_columns ?? []);
  const allCols = run.metrics?.artifact_dir
    ? null // artifact_dir doesn't carry schema; we rely on spec
    : null;

  // Derive feature list from task spec (may be null if spec absent).
  const featureCols: string[] = spec
    ? (spec.feature_columns ?? []).filter((c) => !drop.has(c) && c !== target)
    : [];

  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(featureCols.map((c) => [c, ""])),
  );
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setLoading(true);
      setError(null);
      setResult(null);
      try {
        // Convert string values to numbers where possible.
        const record: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(values)) {
          const num = Number(v);
          record[k] = v === "" ? null : Number.isNaN(num) ? v : num;
        }
        const res = await api.predictRecords(runId, [record]);
        setResult(res);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    },
    [runId, values],
  );

  if (featureCols.length === 0) {
    return (
      <div>
        <SectionHeading>Single prediction</SectionHeading>
        <p className="text-sm text-muted">
          Feature columns are not available in the task specification. Use the batch scoring or CLI instead.
        </p>
      </div>
    );
  }

  return (
    <div>
      <SectionHeading>Single prediction</SectionHeading>
      <p className="mb-3 text-xs text-muted">
        Fill in the feature values and get an instant prediction from the model.
      </p>
      <form onSubmit={handleSubmit} id={`predict-form-${runId}`} className="space-y-3">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {featureCols.map((col) => (
            <div key={col} className="flex flex-col gap-1">
              <label htmlFor={`field-${runId}-${col}`} className="eyebrow truncate">
                {col}
              </label>
              <input
                id={`field-${runId}-${col}`}
                type="text"
                value={values[col] ?? ""}
                onChange={(e) => setValues((prev) => ({ ...prev, [col]: e.target.value }))}
                placeholder={col}
                className="rounded-lg border border-border bg-surface-muted/60 px-3 py-1.5 text-sm outline-none
                           focus:border-accent focus:ring-1 focus:ring-accent/30 placeholder:text-muted"
              />
            </div>
          ))}
        </div>
        <Button type="submit" disabled={loading} id={`predict-submit-${runId}`}>
          {loading ? <Spinner /> : "Predict"}
        </Button>
      </form>

      {error && <ErrorNote>{error}</ErrorNote>}

      {result && (
        <PredictResultCard result={result} />
      )}
    </div>
  );
}

function PredictResultCard({ result }: { result: PredictResult }) {
  const pred = result.predictions[0];
  const isClf = result.classes !== undefined;

  return (
    <div className="mt-3 rounded-xl border border-success/25 bg-success-soft p-4">
      <p className="eyebrow mb-1">prediction</p>
      <p className="text-xl font-semibold text-success">{String(pred)}</p>
      {isClf && result.probabilities && result.classes && (
        <div className="mt-3 space-y-1.5">
          <p className="eyebrow">class probabilities</p>
          {result.classes.map((cls, i) => {
            const prob = result.probabilities![0]?.[i] ?? 0;
            return (
              <div key={cls} className="flex items-center gap-2">
                <span className="w-24 truncate text-xs text-muted">{cls}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-muted">
                  <div
                    className="h-full rounded-full bg-accent transition-all"
                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-12 text-right text-xs tabular">{(prob * 100).toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch scoring (file upload)
// ─────────────────────────────────────────────────────────────────────────────

function BatchScoreSection({ runId }: { runId: string }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const handleScore = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setDone(false);
    try {
      const blob = await api.predictBatch(runId, file);
      // Trigger a download of the scored Parquet.
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${runId}_scored.parquet`;
      a.click();
      URL.revokeObjectURL(url);
      setDone(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [runId, file]);

  return (
    <div>
      <SectionHeading>Batch scoring</SectionHeading>
      <p className="mb-3 text-xs text-muted">
        Upload a CSV or Parquet file. The result is returned as a scored Parquet file with a{" "}
        <code className="rounded bg-surface-muted px-1 py-0.5 font-mono text-[11px]">prediction</code> column.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <label
          htmlFor={`batch-file-${runId}`}
          className="cursor-pointer rounded-lg border border-border bg-surface-muted/60 px-3 py-1.5 text-sm
                     hover:border-border-strong hover:bg-surface-muted transition-colors"
        >
          {file ? file.name : "Choose file (CSV / Parquet)"}
        </label>
        <input
          id={`batch-file-${runId}`}
          ref={fileRef}
          type="file"
          accept=".csv,.tsv,.parquet,.pq,.jsonl"
          className="sr-only"
          onChange={(e) => {
            setFile(e.target.files?.[0] ?? null);
            setDone(false);
            setError(null);
          }}
        />
        <Button
          onClick={handleScore}
          disabled={!file || loading}
          variant="ghost"
          id={`batch-score-btn-${runId}`}
        >
          {loading ? <Spinner /> : "Score and download"}
        </Button>
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      {done && !error && (
        <p className="mt-2 text-xs text-success">Scored file downloaded successfully.</p>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Bundle download
// ─────────────────────────────────────────────────────────────────────────────

function BundleDownloadSection({ runId }: { runId: string }) {
  return (
    <div>
      <SectionHeading>Export model bundle</SectionHeading>
      <p className="mb-3 text-xs text-muted">
        Download a self-contained zip with <code className="font-mono text-[11px]">model.joblib</code>,{" "}
        <code className="font-mono text-[11px]">predict.py</code>,{" "}
        <code className="font-mono text-[11px]">requirements.txt</code>,{" "}
        <code className="font-mono text-[11px]">schema.json</code> and{" "}
        <code className="font-mono text-[11px]">metrics.json</code>. Runs in a fresh venv with only its own requirements.
      </p>
      <a
        href={api.bundleUrl(runId)}
        download={`automl-bundle-${runId}.zip`}
        id={`bundle-download-${runId}`}
        className="inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-surface-muted/60
                   px-4 py-2 text-sm font-medium text-foreground transition-all
                   hover:border-border-strong hover:bg-surface-muted"
      >
        ↓ Download bundle (.zip)
      </a>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared
// ─────────────────────────────────────────────────────────────────────────────

function SectionHeading({ children }: { children: React.ReactNode }) {
  return <h3 className="mb-2 text-sm font-semibold tracking-tight">{children}</h3>;
}
