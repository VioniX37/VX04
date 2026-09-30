import type { Stage } from "./types";

/** Pipeline stages in execution order, with the agent responsible for each. */
export const STAGES: { id: Stage; label: string; agent: string }[] = [
  { id: "parse", label: "Parse request", agent: "Prompt Agent" },
  { id: "verify_request", label: "Verify request", agent: "Manager" },
  { id: "prepare", label: "Prepare data", agent: "Manager" },
  { id: "retrieve", label: "Retrieve knowledge", agent: "Manager" },
  { id: "plan", label: "Plan", agent: "Manager" },
  { id: "execute_plans", label: "Analyse plans", agent: "Data + Model Agents" },
  { id: "ground", label: "Ground on data", agent: "Manager" },
  { id: "select", label: "Select plan", agent: "Manager" },
  { id: "implement", label: "Implement & run", agent: "Operation Agent" },
  { id: "verify_impl", label: "Verify result", agent: "Manager" },
];

export const AGENT_LABELS: Record<string, string> = {
  manager: "Manager",
  prompt_agent: "Prompt Agent",
  data_agent: "Data Agent",
  model_agent: "Model Agent",
  plan_analyst: "Plan Analyst",
  operation_agent: "Operation Agent",
  system: "System",
};

/** Format a metric value compactly (4 decimals, or thousands separators for large values). */
export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.abs(value) >= 100 ? value.toLocaleString(undefined, { maximumFractionDigits: 0 }) : value.toFixed(4);
}

/** Human-readable byte size. */
export function formatBytes(bytes: number): string {
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

/** Compact row count (e.g. 3.5M, 20k). */
export function formatRows(rows: number | null | undefined): string {
  if (rows === null || rows === undefined) return "all";
  if (rows >= 1_000_000) return `${(rows / 1_000_000).toFixed(rows % 1_000_000 ? 1 : 0)}M`;
  if (rows >= 1_000) return `${Math.round(rows / 1_000)}k`;
  return String(rows);
}

/** Whether a higher value of `metric` is better. */
export function higherIsBetter(metric: string | undefined): boolean {
  return !["rmse", "mae", "mape", "rmsle"].includes(metric ?? "");
}
