"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { ApprovalMode, Dataset, IngestJob } from "@/lib/types";
import { DatasetProfileTable } from "./DatasetProfileTable";
import { IngestProgress, type UploadState } from "./viz/IngestProgress";
import { Button, Card, CardTitle, ErrorNote, Spinner, cn } from "./ui";

const EXAMPLES = [
  "Predict whether a customer will churn. The classes are imbalanced, so optimise macro F1.",
  "Predict the house price from its features and report RMSE.",
  "Classify the sentiment of product reviews with at least 85% accuracy.",
  "Forecast the next 14 days of store sales per store. Optimise sMAPE.",
];

type SourceTab = "upload" | "register" | "existing";

/** A registration in progress (or just finished): browser upload first, then the server-side job. */
interface Ingest {
  name: string;
  startedAt: number;
  upload: UploadState | null;
  job: IngestJob | null;
  error: string | null;
}

const POLL_MS = 500;
/** Consecutive failed polls before we tell the user the backend is gone (e.g. it crashed). */
const MAX_POLL_FAILURES = 6;

const TABS: { id: SourceTab; label: string }[] = [
  { id: "upload", label: "Upload" },
  { id: "register", label: "Path or URL" },
  { id: "existing", label: "Registered" },
];

/** Form to choose a dataset (upload, register by path/URL, or reuse) and describe the task. */
export function NewRunForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<SourceTab>("upload");
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [known, setKnown] = useState<Dataset[] | null>(null);
  const [dragging, setDragging] = useState(false);
  const [location, setLocation] = useState("");
  const [prompt, setPrompt] = useState("");
  const [approval, setApproval] = useState<ApprovalMode>("auto");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ingest, setIngest] = useState<Ingest | null>(null);
  const busy = ingest !== null && !ingest.error && ingest.job?.status !== "succeeded" && ingest.job?.status !== "failed";

  useEffect(() => {
    if (tab === "existing" && known === null) api.listDatasets().then(setKnown, (e) => setError(e.message));
  }, [tab, known]);

  // Poll the server-side job until it finishes; a run of failed polls means the backend went away.
  const jobId = ingest?.job?.status === "running" && !ingest.error ? ingest.job.id : null;
  useEffect(() => {
    if (!jobId) return;
    let failures = 0;
    const id = setInterval(() => {
      api.getIngestJob(jobId).then(
        (job) => {
          failures = 0;
          setIngest((s) => (s && s.job?.id === job.id ? { ...s, job } : s));
          if (job.status === "succeeded" && job.dataset) {
            setDataset(job.dataset);
            setKnown(null);
          }
        },
        (e: Error) => {
          failures += 1;
          const gone = e.message.includes("not found") || failures >= MAX_POLL_FAILURES;
          if (gone)
            setIngest((s) =>
              s && {
                ...s,
                error: e.message.includes("not found")
                  ? "The backend restarted and lost this registration. Check the backend terminal, then try again."
                  : `Lost contact with the backend while it was processing the file (${e.message}). It may have crashed; check the backend terminal.`,
              },
            );
        },
      );
    }, POLL_MS);
    return () => clearInterval(id);
  }, [jobId]);

  async function upload(file: File) {
    setError(null);
    setIngest({
      name: file.name,
      startedAt: Date.now(),
      upload: { filename: file.name, loaded: 0, total: file.size },
      job: null,
      error: null,
    });
    try {
      const job = await api.uploadDataset(file, (f) =>
        setIngest((s) => s && s.upload && { ...s, upload: { ...s.upload, loaded: f * s.upload.total } }),
      );
      setIngest((s) => s && { ...s, job });
    } catch (e) {
      setIngest((s) => s && { ...s, error: (e as Error).message });
    }
  }

  async function register() {
    const value = location.trim();
    if (!value) return;
    setError(null);
    setIngest({ name: value.split(/[\\/]/).pop() || value, startedAt: Date.now(), upload: null, job: null, error: null });
    try {
      const isUrl = /^https?:\/\//i.test(value);
      const job = await api.registerDataset(isUrl ? { url: value } : { path: value });
      setIngest((s) => s && { ...s, job });
    } catch (e) {
      setIngest((s) => s && { ...s, error: (e as Error).message });
    }
  }

  async function submit() {
    if (!dataset || prompt.trim().length < 3) return;
    setError(null);
    setSubmitting(true);
    try {
      const run = await api.createRun(dataset.id, prompt.trim(), approval);
      router.push(`/runs/${run.id}`);
    } catch (e) {
      setError((e as Error).message);
      setSubmitting(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <Card className="lg:col-span-3">
        <CardTitle
          aside={
            <div role="tablist" className="flex rounded-lg bg-surface-muted p-0.5 text-xs">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  role="tab"
                  aria-selected={tab === t.id}
                  onClick={() => setTab(t.id)}
                  className={cn(
                    "rounded-md px-2.5 py-1",
                    tab === t.id ? "bg-surface font-medium shadow-sm" : "text-muted hover:text-foreground",
                  )}
                >
                  {t.label}
                </button>
              ))}
            </div>
          }
        >
          Dataset
        </CardTitle>

        {tab === "upload" && (
          <div
            role="button"
            tabIndex={0}
            onClick={() => !busy && inputRef.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const file = e.dataTransfer.files[0];
              if (file && !busy) upload(file);
            }}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border border-dashed px-4 py-8 text-center text-sm transition-colors",
              dragging ? "border-accent bg-accent-soft" : "border-border hover:bg-surface-muted",
            )}
          >
            {busy ? (
              <span className="flex items-center justify-center gap-2 text-muted">
                <Spinner /> Registering…
              </span>
            ) : (
              <>
                <span className="font-medium">
                  {dataset ? `${dataset.filename} — drop another file to replace` : "Drop a file here, or click to browse"}
                </span>
                <span className="text-xs text-muted">CSV, TSV, Parquet or JSONL (optionally .gz), up to 5 GB</span>
              </>
            )}
            <input
              ref={inputRef}
              type="file"
              accept=".csv,.tsv,.parquet,.jsonl,.ndjson,.gz"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) upload(file);
                e.target.value = "";
              }}
            />
          </div>
        )}

        {tab === "register" && (
          <div className="space-y-2">
            <div className="flex gap-2">
              <input
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && register()}
                placeholder="/kaggle/input/higgs/HIGGS.csv.gz  or  https://…/data.parquet"
                className="min-w-0 flex-1 rounded-lg border border-border bg-surface px-3 py-2 font-mono text-xs outline-none focus:border-accent"
              />
              <Button variant="ghost" onClick={register} disabled={!location.trim() || !!busy}>
                {busy && <Spinner />}
                Register
              </Button>
            </div>
          </div>
        )}

        {tab === "existing" && (
          <div className="max-h-56 space-y-1 overflow-auto">
            {known === null && (
              <p className="flex items-center gap-2 text-sm text-muted">
                <Spinner /> Loading datasets…
              </p>
            )}
            {known?.length === 0 && <p className="text-sm text-muted">No datasets registered yet.</p>}
            {known?.map((d) => (
              <button
                key={d.id}
                onClick={() => setDataset(d)}
                className={cn(
                  "flex w-full items-center justify-between gap-3 rounded-md px-3 py-2 text-left text-sm",
                  dataset?.id === d.id ? "bg-accent-soft" : "hover:bg-surface-muted",
                )}
              >
                <span className="truncate font-medium">{d.filename}</span>
                <span className="shrink-0 text-xs text-muted">
                  {d.profile.n_rows.toLocaleString()} rows · {d.source}
                </span>
              </button>
            ))}
          </div>
        )}

        {ingest && tab !== "existing" && (
          <div className="mt-4">
            <IngestProgress
              name={ingest.name}
              job={ingest.job}
              upload={ingest.upload}
              startedAt={ingest.startedAt}
              error={ingest.error}
            />
          </div>
        )}

        {dataset && (
          <div className="mt-5">
            <DatasetProfileTable profile={dataset.profile} />
          </div>
        )}
      </Card>

      <Card className="flex flex-col lg:col-span-2">
        <CardTitle>Task</CardTitle>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={6}
          placeholder="e.g. Predict which customers will churn next month, optimising F1…"
          className="w-full resize-y rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <div className="mt-3 space-y-1.5">
          <p className="text-xs text-muted">Examples</p>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setPrompt(ex)}
              className="block w-full rounded-md px-2 py-1.5 text-left text-xs text-muted hover:bg-surface-muted hover:text-foreground"
            >
              {ex}
            </button>
          ))}
        </div>

        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-foreground">Human-in-the-Loop</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 rounded-lg bg-surface-muted p-1 text-xs">
            {(
              [
                { id: "auto", label: "Auto" },
                { id: "plans", label: "Plans" },
                { id: "plans+code", label: "Plans+Code" },
              ] as const
            ).map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setApproval(opt.id)}
                className={cn(
                  "flex flex-col items-center rounded-md p-2 text-center transition-all",
                  approval === opt.id
                    ? "bg-surface font-semibold text-accent shadow-xs"
                    : "text-muted hover:bg-surface/50 hover:text-foreground",
                )}
              >
                <span>{opt.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-auto space-y-3 pt-5">
          {error && <ErrorNote>{error}</ErrorNote>}
          <Button className="w-full" onClick={submit} disabled={!dataset || prompt.trim().length < 3 || submitting}>
            {submitting && <Spinner />}
            Start run
          </Button>
        </div>
      </Card>
    </div>
  );
}
