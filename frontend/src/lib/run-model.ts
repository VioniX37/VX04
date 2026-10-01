/**
 * Derives everything the run visualizations need from the raw event stream:
 * stage timing, per-agent activity spans, LLM calls, grounding results, sandbox
 * jobs and resource telemetry. Pure functions; memoise the result in components.
 */
import { STAGES } from "./stages";
import type { AgentEvent, BudgetSnapshot, KnowledgeRef, PlanEvaluation, RungResult, Stage, TaskSpec } from "./types";

export type StageStatus = "pending" | "active" | "done" | "failed" | "skipped";

export interface StageInfo {
  id: Stage;
  label: string;
  agent: string;
  status: StageStatus;
  start: number | null; // seconds since run start
  end: number | null;
}

export type Lane =
  | "prompt_agent"
  | "manager"
  | "data_agent"
  | "model_agent"
  | "plan_analyst"
  | "operation_agent"
  | "sandbox";

export interface Span {
  lane: Lane;
  start: number;
  end: number;
  label: string;
  detail?: string;
  kind: "llm" | "cached" | "train" | "ground";
}

export interface LlmCall {
  t: number;
  duration: number;
  agent: string;
  schema: string;
  role: string;
  models: string[];
  inputTokens: number;
  outputTokens: number;
  cached: boolean;
}

export interface TelemetryPoint {
  t: number; // seconds since run start
  rssMb: number;
  cores: number;
  job: string; // "final" or "<plan>@<rows>"
}

export interface SandboxJob {
  id: string;
  label: string;
  kind: "ground" | "final";
  planId?: string;
  rows?: number | null;
  start: number;
  end: number;
  phases: { stage: string; t: number }[];
  done: boolean;
}

export interface GroundingRound {
  round: number;
  predicted: Record<string, number>;
  families: Record<string, string>;
  schedule: (number | null)[];
  nTrain: number;
  rungs: RungResult[];
  stoppedForBudget: boolean;
}

export interface RunModel {
  t0: number; // epoch ms of the first event
  now: number; // seconds since t0 of the last event
  stages: StageInfo[];
  current: Stage | null;
  spans: Span[];
  llmCalls: LlmCall[];
  telemetry: TelemetryPoint[];
  jobs: SandboxJob[];
  grounding: GroundingRound | null;
  spec?: TaskSpec;
  split?: { train: number; valid: number; test: number };
  knowledge?: KnowledgeRef[];
  ranked?: PlanEvaluation[];
  selected: string | null;
  liveCode?: string;
  budget?: BudgetSnapshot;
  finalProgress?: { stage: string; n_train?: number };
}

const AGENT_LANES: Lane[] = ["prompt_agent", "manager", "data_agent", "model_agent", "plan_analyst", "operation_agent"];

function secs(e: AgentEvent, t0: number) {
  return (Date.parse(e.ts) - t0) / 1000;
}

function last<T>(events: AgentEvent[], stage: Stage, key: string): T | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const p = events[i].payload;
    if (events[i].stage === stage && p && key in p) return p[key] as T;
  }
  return undefined;
}

/** Build the run model. `finished`/`failed` come from the run record. */
export function buildRunModel(events: AgentEvent[], finished: boolean, failed: boolean): RunModel {
  const t0 = events.length ? Date.parse(events[0].ts) : Date.now();
  const nonTelemetry = events.filter((e) => e.kind !== "telemetry");

  // ---- stages
  const firstSeen = new Map<Stage, number>();
  const lastSeen = new Map<Stage, number>();
  for (const e of events) {
    const t = secs(e, t0);
    if (!firstSeen.has(e.stage)) firstSeen.set(e.stage, t);
    lastSeen.set(e.stage, t);
  }
  const progressed = nonTelemetry.filter((e) => e.stage !== "done");
  const current = progressed.length ? progressed[progressed.length - 1].stage : null;
  const currentIdx = STAGES.findIndex((s) => s.id === current);
  const stages: StageInfo[] = STAGES.map((s, i) => {
    const start = firstSeen.get(s.id) ?? null;
    let status: StageStatus = "pending";
    if (start !== null) status = "done";
    if (s.id === current && !finished) status = "active";
    if (s.id === current && failed) status = "failed";
    if (start === null && currentIdx > i) status = "skipped"; // e.g. grounding in paper (pseudo) mode
    const nextStart = STAGES.slice(i + 1)
      .map((n) => firstSeen.get(n.id))
      .find((v) => v !== undefined);
    return { ...s, status, start, end: start === null ? null : (nextStart ?? lastSeen.get(s.id) ?? start) };
  });

  // ---- LLM calls and agent spans
  const llmCalls: LlmCall[] = [];
  const spans: Span[] = [];
  for (const e of events) {
    if (e.kind !== "llm" || !e.payload) continue;
    const meta = e.payload.meta as
      | { schema: string; role: string; models: string[]; calls: number; cache_hits: number;
          input_tokens: number; output_tokens: number; duration_s: number }
      | undefined;
    if (!meta) continue;
    const end = secs(e, t0);
    const cached = meta.calls === 0 && meta.cache_hits > 0;
    llmCalls.push({
      t: end - meta.duration_s,
      duration: meta.duration_s,
      agent: e.agent,
      schema: meta.schema,
      role: meta.role,
      models: meta.models,
      inputTokens: meta.input_tokens,
      outputTokens: meta.output_tokens,
      cached,
    });
    if (AGENT_LANES.includes(e.agent as Lane)) {
      spans.push({
        lane: e.agent as Lane,
        start: end - Math.max(meta.duration_s, 0.05),
        end,
        label: meta.schema,
        detail: cached ? "cached" : `${meta.models.join(", ") || meta.role} · ${(meta.input_tokens + meta.output_tokens).toLocaleString()} tok`,
        kind: cached ? "cached" : "llm",
      });
    }
  }

  // ---- sandbox jobs (grounding runs + final training) and telemetry
  const jobs = new Map<string, SandboxJob>();
  const telemetry: TelemetryPoint[] = [];
  for (const e of events) {
    const p = e.payload;
    if (!p) continue;
    const t = secs(e, t0);
    let id: string | null = null;
    if (e.stage === "ground" && e.kind === "telemetry" && "plan_id" in p) {
      const rows = p.rows as number | null;
      id = `r${p.round}:${p.plan_id}@${rows ?? "all"}`;
      if (!jobs.has(id)) {
        jobs.set(id, {
          id, kind: "ground", planId: p.plan_id as string, rows,
          label: `${p.plan_id} · ${rows ? `${Math.round(rows / 1000)}k` : "all"} rows`,
          start: t, end: t, phases: [], done: false,
        });
      }
    } else if (e.stage === "implement" && p && ("progress" in p || "telemetry" in p || ("attempt" in p && "code" in p))) {
      id = `final:${p.attempt ?? 1}`;
      if (!jobs.has(id)) {
        jobs.set(id, {
          id, kind: "final", label: `final training · attempt ${p.attempt ?? 1}`,
          start: t, end: t, phases: [], done: false,
        });
      }
    } else if (e.stage === "implement" && p && "result" in p) {
      const job = jobs.get(`final:${p.attempt ?? 1}`);
      if (job) {
        job.end = t;
        job.done = true;
      }
      continue;
    }
    if (!id) continue;
    const job = jobs.get(id)!;
    job.end = Math.max(job.end, t);
    if ("telemetry" in p) {
      const tel = p.telemetry as { rss_mb: number; cores: number };
      telemetry.push({ t, rssMb: tel.rss_mb, cores: tel.cores, job: id });
    }
    if ("progress" in p) {
      const stage = (p.progress as { stage: string }).stage;
      job.phases.push({ stage, t });
      if (stage === "done") job.done = true;
    }
  }
  for (const job of jobs.values()) {
    spans.push({
      lane: "sandbox",
      start: job.start,
      end: Math.max(job.end, job.start + 0.3),
      label: job.label,
      kind: job.kind === "final" ? "train" : "ground",
    });
  }

  // ---- grounding (latest round)
  let grounding: GroundingRound | null = null;
  let startIdx = -1;
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].stage === "ground" && events[i].payload && "predicted" in events[i].payload!) {
      startIdx = i;
      break;
    }
  }
  if (startIdx >= 0) {
    const head = events[startIdx].payload!;
    const round = events.slice(startIdx).filter((e) => e.stage === "ground" && e.kind !== "telemetry");
    const rungs = round.filter((e) => e.payload && "rung" in e.payload).map((e) => e.payload!.rung as RungResult);
    const families: Record<string, string> = {};
    rungs.forEach((r) => r.results.forEach((x) => (families[x.plan_id] = x.model_family)));
    const roundNo = Number(
      (events.slice(startIdx).find((e) => e.kind === "telemetry" && e.payload && "round" in e.payload)?.payload?.round as
        | number
        | undefined) ?? 1,
    );
    grounding = {
      round: roundNo,
      predicted: head.predicted as Record<string, number>,
      families,
      schedule: (head.schedule as (number | null)[]) ?? [],
      nTrain: (head.n_train as number) ?? 0,
      rungs,
      stoppedForBudget: round.some((e) => e.kind === "warning" && e.message.includes("budget")),
    };
  }

  const finalProgressEvent = [...events]
    .reverse()
    .find((e) => e.stage === "implement" && e.kind === "info" && e.payload && "progress" in e.payload);

  return {
    t0,
    now: events.length ? secs(events[events.length - 1], t0) : 0,
    stages,
    current,
    spans: spans.sort((a, b) => a.start - b.start),
    llmCalls,
    telemetry,
    jobs: [...jobs.values()].sort((a, b) => a.start - b.start),
    grounding,
    spec: last<TaskSpec>(events, "verify_request", "task_spec"),
    split: last(events, "prepare", "split"),
    knowledge: last<KnowledgeRef[]>(events, "retrieve", "knowledge"),
    ranked: last<PlanEvaluation[]>(events, "select", "ranked"),
    selected: last<string>(events, "select", "selected") ?? null,
    liveCode: last<string>(events, "implement", "code"),
    budget: last<BudgetSnapshot>(events, "verify_impl", "budget"),
    finalProgress: finalProgressEvent?.payload?.progress as { stage: string; n_train?: number } | undefined,
  };
}

/** mm:ss (or h:mm:ss) for a number of seconds. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
}

/** Stable chart colour for a plan/series index. */
export const SERIES = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)", "var(--chart-6)"];

export const LANE_META: Record<Lane, { label: string; color: string }> = {
  prompt_agent: { label: "Prompt Agent", color: "var(--chart-6)" },
  manager: { label: "Manager", color: "var(--chart-1)" },
  data_agent: { label: "Data Agent", color: "var(--chart-2)" },
  model_agent: { label: "Model Agent", color: "var(--chart-4)" },
  plan_analyst: { label: "Plan Analyst", color: "var(--chart-2)" },
  operation_agent: { label: "Operation Agent", color: "var(--chart-3)" },
  sandbox: { label: "Sandbox", color: "var(--chart-5)" },
};
