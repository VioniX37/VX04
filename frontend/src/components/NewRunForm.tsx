"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { api } from "@/lib/api";
import type { Dataset } from "@/lib/types";
import { DatasetProfileTable } from "./DatasetProfileTable";
import { Button, Card, CardTitle, ErrorNote, Spinner, cn } from "./ui";

const EXAMPLES = [
  "Predict whether a customer will churn. The classes are imbalanced, so optimise macro F1.",
  "Predict the house price from its features and report RMSE.",
  "Classify the sentiment of product reviews with at least 85% accuracy.",
];

export function NewRunForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload(file: File) {
    setError(null);
    setUploading(true);
    try {
      setDataset(await api.uploadDataset(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function submit() {
    if (!dataset || prompt.trim().length < 3) return;
    setError(null);
    setSubmitting(true);
    try {
      const run = await api.createRun(dataset.id, prompt.trim());
      router.push(`/runs/${run.id}`);
    } catch (e) {
      setError((e as Error).message);
      setSubmitting(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <Card className="lg:col-span-3">
        <CardTitle>1 · Dataset</CardTitle>
        <div
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
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
          {uploading ? (
            <span className="flex items-center gap-2 text-muted">
              <Spinner /> Uploading and profiling…
            </span>
          ) : dataset ? (
            <>
              <span className="font-medium">{dataset.filename}</span>
              <span className="text-xs text-muted">Click or drop to replace</span>
            </>
          ) : (
            <>
              <span className="font-medium">Drop a CSV file here, or click to browse</span>
              <span className="text-xs text-muted">Samples are in data/samples/</span>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.tsv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) upload(file);
              e.target.value = "";
            }}
          />
        </div>
        {dataset && (
          <div className="mt-5">
            <DatasetProfileTable profile={dataset.profile} />
          </div>
        )}
      </Card>

      <Card className="flex flex-col lg:col-span-2">
        <CardTitle>2 · Describe the task</CardTitle>
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
        <div className="mt-auto space-y-3 pt-5">
          {error && <ErrorNote>{error}</ErrorNote>}
          <Button
            className="w-full"
            onClick={submit}
            disabled={!dataset || prompt.trim().length < 3 || submitting}
          >
            {submitting && <Spinner />}
            Start AutoML run
          </Button>
        </div>
      </Card>
    </div>
  );
}
