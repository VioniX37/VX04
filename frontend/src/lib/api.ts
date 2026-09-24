import type { AgentEvent, Dataset, Health, Run } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
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
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  uploadDataset: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Dataset>("/datasets", { method: "POST", body: form });
  },
  listDatasets: () => request<Dataset[]>("/datasets"),
  createRun: (datasetId: string, prompt: string) =>
    request<Run>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, prompt }),
    }),
  listRuns: () => request<Run[]>("/runs"),
  getRun: (id: string) => request<Run>(`/runs/${id}`),
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
