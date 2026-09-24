// TypeScript mirrors of the backend Pydantic schemas (backend/src/automl_agent/schemas).

export type ColumnKind = "numeric" | "categorical" | "text" | "datetime" | "identifier" | "boolean";

export interface ColumnProfile {
  name: string;
  dtype: string;
  kind: ColumnKind;
  n_unique: number;
  n_missing: number;
  sample_values: string[];
  mean_length: number | null;
}

export interface DatasetProfile {
  n_rows: number;
  n_cols: number;
  columns: ColumnProfile[];
  guessed_target: string | null;
  text_columns: string[];
}

export interface Dataset {
  id: string;
  filename: string;
  created_at: string;
  profile: DatasetProfile;
}

export type TaskType = "tabular_classification" | "tabular_regression" | "text_classification";

export interface TaskSpec {
  task_type: TaskType;
  target_column: string;
  text_column: string | null;
  feature_columns: string[] | null;
  drop_columns: string[];
  metric: string;
  metric_target: number | null;
  max_train_time_s: number | null;
  domain: string | null;
  notes: string;
}

export interface Plan {
  id: string;
  title: string;
  rationale: string;
  preprocessing: string[];
  model_family: string;
  hyperparameters: Record<string, unknown>;
  validation: string;
}

export interface PlanEvaluation {
  plan: Plan;
  data: { summary: string; steps: string[]; risks: string[] };
  model: {
    summary: string;
    model_family: string;
    hyperparameters: Record<string, unknown>;
    predicted_score: number;
    predicted_train_time_s: number;
  };
  rank: number | null;
}

export interface ExecutionMetrics {
  metric?: string;
  score?: number | null;
  metrics?: Record<string, number>;
  model_family?: string;
  train_time_s?: number;
  n_train?: number;
  n_test?: number;
  target_met?: boolean;
  attempts?: { revision: number; plan_id: string; score: number | null; ok: boolean; issues: string[] }[];
  artifact_dir?: string;
}

export type RunStatus = "pending" | "running" | "succeeded" | "failed";

export interface Run {
  id: string;
  dataset_id: string;
  prompt: string;
  status: RunStatus;
  created_at: string;
  finished_at: string | null;
  task_spec: TaskSpec | null;
  plan: Plan | null;
  metrics: ExecutionMetrics | null;
  code: string | null;
  error: string | null;
  llm_usage: { calls: number; input_tokens: number; output_tokens: number } | null;
}

export type Stage =
  | "parse"
  | "verify_request"
  | "retrieve"
  | "plan"
  | "execute_plans"
  | "select"
  | "implement"
  | "verify_impl"
  | "done";

export type EventKind = "status" | "info" | "llm" | "artifact" | "warning" | "error";

export interface AgentEvent {
  seq: number;
  run_id: string;
  ts: string;
  stage: Stage;
  agent: string;
  kind: EventKind;
  message: string;
  payload: Record<string, unknown> | null;
}

export interface Health {
  status: string;
  version: string;
  llm_provider: string;
  llm_model: string;
  codegen_mode: string;
}
