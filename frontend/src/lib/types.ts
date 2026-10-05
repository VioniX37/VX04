/**
 * TypeScript mirrors of the backend Pydantic schemas (`backend/src/automl_agent/schemas`).
 * Keep these in sync with the backend; the API contract is documented in `docs/reference/rest-api.md`.
 */

export type ColumnKind = "numeric" | "categorical" | "text" | "datetime" | "identifier" | "boolean";

/** Size bucket that drives model choice and grounding (small < 100k rows, medium < 2M, large). */
export type ScaleTier = "small" | "medium" | "large";

/** Statistics for one column of a dataset. */
export interface ColumnProfile {
  name: string;
  dtype: string;
  kind: ColumnKind;
  n_unique: number;
  n_missing: number;
  sample_values: string[];
  mean_length: number | null;
  /** Share of the most frequent values (low-cardinality columns only). */
  top_values: Record<string, number> | null;
}

/** Summary of a dataset; never contains raw rows beyond a few samples. */
export interface DatasetProfile {
  n_rows: number;
  n_cols: number;
  columns: ColumnProfile[];
  guessed_target: string | null;
  text_columns: string[];
  size_bytes: number;
  memory_estimate_mb: number;
  scale_tier: ScaleTier;
  /** True when distinct counts are HyperLogLog estimates (datasets above 2M rows). */
  approximate_counts: boolean;
}

/** A registered dataset. */
export interface Dataset {
  id: string;
  filename: string;
  /** How the dataset was added: browser upload, server path or URL. */
  source: "upload" | "path" | "url";
  created_at: string;
  profile: DatasetProfile;
}

export type TaskType = "tabular_classification" | "tabular_regression" | "text_classification";

/** Structured task produced by the Prompt Agent. */
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
  user_expertise: "beginner" | "intermediate" | "expert";
  assumptions: string[];
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

/** A real training run of a plan at one fidelity (grounded verification). */
export interface Observation {
  fidelity_rows: number;
  score: number | null;
  ok: boolean;
  duration_s: number;
  error: string | null;
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
  observations: Observation[];
}

/** Budget consumption snapshot (null budget fields mean unlimited). */
export interface BudgetSnapshot {
  elapsed_s: number;
  wall_budget_s: number | null;
  wall_remaining_s: number | null;
  llm_calls: number;
  llm_call_budget: number | null;
  tokens: number;
  token_budget: number | null;
}

export type AuditSeverity = "critical" | "high" | "medium" | "low";
export type AuditAction = "drop" | "flag" | "impute" | "stratify" | "none";

export interface AuditFinding {
  check: string;
  severity: AuditSeverity;
  column: string | null;
  message: string;
  suggested_action: AuditAction;
  metric_name?: string | null;
  metric_value?: number | null;
  details?: Record<string, unknown>;
}

export interface AuditReport {
  findings: AuditFinding[];
  dropped_columns: string[];
  train_test_duplicates: number;
  duplicate_pct: number;
  has_leakage: boolean;
  duration_s: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  method: "native_gain" | "permutation";
}

export interface CalibrationPoint {
  prob_pred: number;
  prob_true: number;
}

export interface CalibrationReport {
  brier_score: number;
  points: CalibrationPoint[];
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
  per_class: Record<string, { precision: number; recall: number; "f1-score": number; support: number }>;
}

export interface ResidualsData {
  mae: number;
  rmse: number;
  r2: number;
  max_error: number;
  quantiles: Record<string, number>;
  sample_residuals: number[];
}

export interface ModelCard {
  task_type: TaskType;
  model_family: string;
  primary_metric: string;
  primary_score: number | null;
  feature_importances: FeatureImportance[];
  confusion_matrix?: ConfusionMatrixData | null;
  residuals?: ResidualsData | null;
  calibration?: CalibrationReport | null;
  audit_summary: string[];
  why_this_model: string;
  metadata?: Record<string, unknown>;
}

/** Contents of the final `metrics.json` plus run bookkeeping. */
export interface ExecutionMetrics {
  metric?: string;
  score?: number | null;
  metrics?: Record<string, number>;
  metrics_valid?: Record<string, number>;
  split?: "valid" | "test";
  model_family?: string;
  train_time_s?: number;
  n_train?: number;
  n_eval?: number;
  target_met?: boolean;
  stop_reason?: string | null;
  budget?: BudgetSnapshot | null;
  attempts?: { revision: number; plan_id: string; score: number | null; ok: boolean; issues: string[] }[];
  artifact_dir?: string;
  model_card?: ModelCard | null;
}

/** Settings that define a run's experimental condition. */
export interface RunConfig {
  llm_provider: string;
  models: Record<string, string>;
  agent_fusion: boolean;
  search_grounding: boolean;
  codegen_mode: string;
  n_plans: number;
  max_revisions: number;
  verification_mode: "pseudo" | "grounded";
  memory?: { enabled: boolean; k: number };
  grounding: { min_rows: number; growth: number; eta: number; valid_rows: number };
  budget: { wall_s: number | null; llm_calls: number | null; tokens: number | null };
}

export type RunStatus = "pending" | "running" | "awaiting_input" | "succeeded" | "failed" | "cancelled";

export type ApprovalMode = "auto" | "plans" | "plans+code";

export type PlanApprovalAction = "approve" | "pick" | "edit";

export interface PlanApprovalRequest {
  action: PlanApprovalAction;
  plan_id?: string | null;
  edited_plan?: Record<string, unknown> | null;
  edited_code?: string | null;
  feedback?: string | null;
}

export interface Run {
  id: string;
  dataset_id: string;
  prompt: string;
  status: RunStatus;
  approval?: ApprovalMode;
  human_override?: boolean;
  human_decision?: Record<string, unknown> | null;
  created_at: string;
  finished_at: string | null;
  task_spec: TaskSpec | null;
  plan: Plan | null;
  metrics: ExecutionMetrics | null;
  code: string | null;
  error: string | null;
  llm_usage: { calls: number; cache_hits: number; input_tokens: number; output_tokens: number; total_tokens: number } | null;
  config: RunConfig | null;
}

export type Stage =
  | "parse"
  | "verify_request"
  | "prepare"
  | "retrieve"
  | "plan"
  | "execute_plans"
  | "ground"
  | "select"
  | "implement"
  | "verify_impl"
  | "done";

export type EventKind = "status" | "info" | "llm" | "artifact" | "warning" | "error" | "telemetry";

/** One entry of a run's live event stream. */
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

/** Retrieved planning knowledge as reported by the `retrieve` stage. */
export interface KnowledgeRef {
  id: string;
  title: string;
  /** `local-kb`, `google-search`, or `memory:<run id>` for experience memory. */
  source: string;
  urls: string[];
}

/** Results of one successive-halving rung. */
export interface RungResult {
  rows: number | null;
  results: (Observation & { plan_id: string; model_family: string })[];
}

export interface Health {
  status: string;
  version: string;
  llm_provider: string;
  llm_model: string;
  models: Record<string, string>;
  models_available: Record<string, boolean>;
  codegen_mode: string;
}
