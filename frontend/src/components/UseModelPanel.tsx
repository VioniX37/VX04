"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "@/lib/api";
import type { ModelInput, ModelSchema, PredictResult, Run } from "@/lib/types";
import { Button, Card, CardTitle, cn, ErrorNote, Spinner } from "./ui";

/** The "Use this model" panel shown on a finished, succeeded run. */
export function UseModelPanel({ run }: { run: Run }) {
  if (run.status !== "succeeded") return null;

  return (
    <Card>
      <CardTitle eyebrow="deployment">Use this model</CardTitle>
      <div className="space-y-6">
        <SinglePredictSection runId={run.id} />
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

function SinglePredictSection({ runId }: { runId: string }) {
  // The form is generated from the model's own input schema (the same one shipped in schema.json).
  const [schema, setSchema] = useState<ModelSchema | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.getModelSchema(runId).then(
      (s) => {
        if (cancelled) return;
        setSchema(s);
        setValues(Object.fromEntries(s.features.map((f) => [f.name, ""])));
      },
      (e) => !cancelled && setSchemaError(e instanceof ApiError ? e.message : String(e)),
    );
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!schema) return;
      setLoading(true);
      setError(null);
      setResult(null);
      try {
        // Numeric fields are sent as numbers; everything else as entered. Empty fields mean "missing".
        const record: Record<string, unknown> = {};
        for (const f of schema.features) {
          const v = values[f.name] ?? "";
          record[f.name] = v === "" ? null : f.kind === "numeric" && !Number.isNaN(Number(v)) ? Number(v) : v;
        }
        setResult(await api.predictRecords(runId, [record]));
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    },
    [runId, schema, values],
  );

  if (schemaError) {
    return (
      <div>
        <SectionHeading>Single prediction</SectionHeading>
        <ErrorNote>{schemaError}</ErrorNote>
      </div>
    );
  }
  if (!schema) {
    return (
      <div>
        <SectionHeading>Single prediction</SectionHeading>
        <Spinner />
      </div>
    );
  }

  return (
    <div>
      <SectionHeading>Single prediction</SectionHeading>
      <p className="mb-3 text-xs text-muted">
        Fill in the feature values and get an instant prediction from the model. Leave a field empty to treat it as
        missing.
      </p>
      <form onSubmit={handleSubmit} id={`predict-form-${runId}`} className="space-y-3">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {schema.features.map((f) => (
            <div key={f.name} className={cn("flex flex-col gap-1", f.kind === "text" && "sm:col-span-2 lg:col-span-3")}>
              <label htmlFor={`field-${runId}-${f.name}`} className="eyebrow truncate">
                {f.name}
              </label>
              <FieldInput
                id={`field-${runId}-${f.name}`}
                input={f}
                value={values[f.name] ?? ""}
                onChange={(v) => setValues((prev) => ({ ...prev, [f.name]: v }))}
              />
            </div>
          ))}
        </div>
        <Button type="submit" disabled={loading} id={`predict-submit-${runId}`}>
          {loading ? <Spinner /> : "Predict"}
        </Button>
      </form>

      {error && <ErrorNote>{error}</ErrorNote>}

      {result && <PredictResultCard result={result} />}
    </div>
  );
}

const fieldClass =
  "rounded-lg border border-border bg-surface-muted/60 px-3 py-1.5 text-sm outline-none focus:border-accent focus:ring-1 focus:ring-accent/30 placeholder:text-muted";

function FieldInput({
  id,
  input,
  value,
  onChange,
}: {
  id: string;
  input: ModelInput;
  value: string;
  onChange: (v: string) => void;
}) {
  if (input.kind === "category" && input.categories.length > 0) {
    return (
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)} className={fieldClass}>
        <option value="">(missing)</option>
        {input.categories.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    );
  }
  if (input.kind === "text") {
    return (
      <textarea
        id={id}
        rows={3}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Text to classify"
        className={fieldClass}
      />
    );
  }
  return (
    <input
      id={id}
      type="text"
      inputMode={input.kind === "numeric" ? "decimal" : "text"}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={input.kind === "numeric" ? "number" : input.name}
      className={fieldClass}
    />
  );
}

function PredictResultCard({ result }: { result: PredictResult }) {
  const pred = result.predictions[0];
  const isClf = result.classes !== undefined;

  return (
    <div className="mt-3 rounded-xl border border-success/25 bg-success-soft p-4">
      <p className="eyebrow mb-1">{result.forecast_dates ? `forecast for ${result.forecast_dates[0].slice(0, 10)}` : "prediction"}</p>
      <p className="text-xl font-semibold text-success">
        {typeof pred === "number" ? Number(pred.toFixed(4)) : String(pred)}
      </p>
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
