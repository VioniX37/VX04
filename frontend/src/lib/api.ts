import type { AgentEvent, BundleSchema, Dataset, Health, PredictResult, Run } from "./types";

/** Base URL of the FastAPI backend (set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`). */
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

/** Error carrying the HTTP status and the backend's `detail` message. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function detailOf(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(0, `Cannot reach the backend at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = detailOf(await res.json(), detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

/**
 * Upload a file with progress reporting (fetch cannot report upload progress, so this uses XHR).
 * `onProgress` receives a fraction in [0, 1]; the promise resolves after server-side ingest and profiling.
 */
function uploadWithProgress(file: File, onProgress?: (fraction: number) => void): Promise<Dataset> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/datasets`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as Dataset);
      else reject(new ApiError(xhr.status, detailOf(body, xhr.statusText || "Upload failed")));
    };
    xhr.onerror = () => reject(new ApiError(0, `Cannot reach the backend at ${API_URL}. Is it running?`));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

/** Typed client for the backend REST API. */
export const api = {
  health: () => request<Health>("/health"),
  uploadDataset: uploadWithProgress,
  /** Register a dataset that already exists on the server's disk or at an http(s) URL. */
  registerDataset: (source: { path?: string; url?: string; name?: string }) =>
    request<Dataset>("/datasets/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(source),
    }),
  listDatasets: () => request<Dataset[]>("/datasets"),
  createRun: (datasetId: string, prompt: string) =>
    request<Run>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, prompt }),
    }),
  listRuns: () => request<Run[]>("/runs"),
  getRun: (id: string) => request<Run>(`/runs/${id}`),
  /** Score JSON records against a finished run's model. */
  predictRecords: (runId: string, records: Record<string, unknown>[]) =>
    request<PredictResult>(`/runs/${runId}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(records),
    }),
  /** Return the bundle download URL (opens in a new tab or triggers <a> download). */
  bundleUrl: (runId: string) => `${API_URL}/api/runs/${runId}/artifacts/bundle`,
  /** Score a file upload; returns the response blob for saving. */
  predictBatch: async (runId: string, file: File): Promise<Blob> => {
    const form = new FormData();
    form.append("file", file);
    let res: Response;
    try {
      res = await fetch(`${API_URL}/api/runs/${runId}/predict/batch`, { method: "POST", body: form, cache: "no-store" });
    } catch {
      throw new ApiError(0, `Cannot reach the backend at ${API_URL}. Is it running?`);
    }
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = detailOf(await res.json(), detail); } catch { /* non-JSON */ }
      throw new ApiError(res.status, detail);
    }
    return res.blob();
  },
};

/** Subscribe to a run's live agent events (SSE). Returns an unsubscribe function. */
export function subscribeToRun(
  runId: string,
  handlers: { onEvent: (e: AgentEvent) => void; onEnd?: () => void; onError?: () => void },
): () => void {
  const source = new EventSource(`${API_URL}/api/runs/${runId}/events`);
  source.addEventListener("agent_event", (msg) => {
    handlers.onEvent(JSON.parse((msg as MessageEvent).data) as AgentEvent);
  });
  source.addEventListener("end", () => {
    source.close();
    handlers.onEnd?.();
  });
  source.onerror = () => {
    // The browser reconnects automatically; only report if the stream is closed for good.
    if (source.readyState === EventSource.CLOSED) handlers.onError?.();
  };
  return () => source.close();
}
