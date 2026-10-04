"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { formatBytes } from "@/lib/stages";
import type { ApprovalMode, Dataset } from "@/lib/types";
import { DatasetProfileTable } from "./DatasetProfileTable";
import { IngestProgress, type IngestPhase } from "./viz/IngestProgress";
import { Button, Card, CardTitle, ErrorNote, Spinner, cn } from "./ui";

const EXAMPLES = [
  "Predict whether a customer will churn. The classes are imbalanced, so optimise macro F1.",
  "Predict the house price from its features and report RMSE.",
  "Classify the sentiment of product reviews with at least 85% accuracy.",
];

type SourceTab = "upload" | "register" | "existing";

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
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [location, setLocation] = useState("");
  const [prompt, setPrompt] = useState("");
  const [approval, setApproval] = useState<ApprovalMode>("auto");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ingest, setIngest] = useState<{ phase: IngestPhase; startedAt: number; viaPath: boolean } | null>(null);

  useEffect(() => {
    if (tab === "existing" && known === null) api.listDatasets().then(setKnown, (e) => setError(e.message));
  }, [tab, known]);

  async function upload(file: File) {
    setError(null);
    setProgress(0);
    setBusy(`Uploading ${file.name} (${formatBytes(file.size)})`);
    setIngest({ phase: "upload", startedAt: Date.now(), viaPath: false });
    try {
      const ds = await api.uploadDataset(file, (f) => {
        setProgress(f);
        if (f >= 1) {
          setBusy("Converting to Parquet and profiling…");
          setIngest((s) => (s ? { ...s, phase: "process" } : s));
        }
      });
      setDataset(ds);
      setIngest((s) => (s ? { ...s, phase: "ready" } : s));
    } catch (e) {
      setError((e as Error).message);
      setIngest(null);
    } finally {
      setBusy(null);
      setProgress(null);
    }
  }

  async function register() {
    const value = location.trim();
    if (!value) return;
    setError(null);
    setBusy("Registering, converting to Parquet and profiling… (large files can take a minute)");
    setIngest({ phase: "process", startedAt: Date.now(), viaPath: true });
    try {
      const isUrl = /^https?:\/\//i.test(value);
      setDataset(await api.registerDataset(isUrl ? { url: value } : { path: value }));
      setIngest((s) => (s ? { ...s, phase: "ready" } : s));
    } catch (e) {
      setError((e as Error).message);
      setIngest(null);
    } finally {
      setBusy(null);
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
          eyebrow="step 1 · data"
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
              if (file) upload(file);
            }}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border border-dashed px-4 py-8 text-center text-sm transition-colors",
              dragging ? "border-accent bg-accent-soft" : "border-border hover:bg-surface-muted",
            )}
          >
            {busy ? (
              <div className="w-full max-w-sm space-y-2">
                <span className="flex items-center justify-center gap-2 text-muted">
                  <Spinner /> {busy}
                </span>
                {progress !== null && progress < 1 && (
                  <div className="h-1.5 overflow-hidden rounded-full bg-surface-muted">
                    <div className="h-full bg-accent transition-all" style={{ width: `${progress * 100}%` }} />
                  </div>
                )}
              </div>
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
            <p className="text-xs text-muted">
              For multi-gigabyte data, register a file that is already on the server (e.g. a Kaggle or Colab input
              path) or an http(s) URL. Nothing passes through the browser.
            </p>
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
            {busy && <p className="text-xs text-muted">{busy}</p>}
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

        {ingest && (busy || ingest.phase === "ready") && tab !== "existing" && (
          <div className="mt-4">
            <IngestProgress phase={ingest.phase} startedAt={ingest.startedAt} uploadFraction={progress} viaPath={ingest.viaPath} />
          </div>
        )}

        {dataset && (
          <div className="mt-5">
            <DatasetProfileTable profile={dataset.profile} />
          </div>
        )}
      </Card>

      <Card className="flex flex-col lg:col-span-2">
        <CardTitle eyebrow="step 2 · intent">Describe the task</CardTitle>
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
            <span className="text-[11px] text-muted">Approval Mode</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 rounded-lg bg-surface-muted p-1 text-xs">
            {(
              [
                { id: "auto", label: "Auto", desc: "No review" },
                { id: "plans", label: "Plans", desc: "Review plans" },
                { id: "plans+code", label: "Plans+Code", desc: "Review code" },
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
                <span className="text-[10px] font-normal text-muted">{opt.desc}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-auto space-y-3 pt-5">
          {error && <ErrorNote>{error}</ErrorNote>}
          <Button className="w-full" onClick={submit} disabled={!dataset || prompt.trim().length < 3 || submitting}>
            {submitting && <Spinner />}
            Start AutoML run
          </Button>
        </div>
      </Card>
    </div>
  );
}
