import type { Stage } from "./types";

export const STAGES: { id: Stage; label: string; agent: string }[] = [
  { id: "parse", label: "Parse request", agent: "Prompt Agent" },
  { id: "verify_request", label: "Verify request", agent: "Manager" },
  { id: "retrieve", label: "Retrieve knowledge", agent: "Manager" },
  { id: "plan", label: "Plan", agent: "Manager" },
  { id: "execute_plans", label: "Evaluate plans", agent: "Data + Model Agents" },
  { id: "select", label: "Select plan", agent: "Manager" },
  { id: "implement", label: "Implement & run", agent: "Operation Agent" },
  { id: "verify_impl", label: "Verify result", agent: "Manager" },
];

export const AGENT_LABELS: Record<string, string> = {
  manager: "Manager",
  prompt_agent: "Prompt Agent",
  data_agent: "Data Agent",
  model_agent: "Model Agent",
  operation_agent: "Operation Agent",
  system: "System",
};

export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.abs(value) >= 100 ? value.toLocaleString(undefined, { maximumFractionDigits: 0 }) : value.toFixed(4);
}
