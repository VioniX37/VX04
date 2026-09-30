"use client";

import { useEffect, useMemo, useState } from "react";

import { api, subscribeToRun } from "@/lib/api";
import type {
  AgentEvent,
  BudgetSnapshot,
  ExecutionMetrics,
  KnowledgeRef,
  PlanEvaluation,
  Run,
  RungResult,
  Stage,
  TaskSpec,
} from "@/lib/types";
import { AgentTimeline } from "./AgentTimeline";
import { BudgetPanel } from "./BudgetPanel";
import { CodeViewer } from "./CodeViewer";
import { GroundingPanel } from "./GroundingPanel";
import { KnowledgePanel } from "./KnowledgePanel";
import { MetricsPanel } from "./MetricsPanel";
import { PlanCards } from "./PlanCards";
import { StageStepper } from "./StageStepper";
import { Badge, Card, CardTitle, ErrorNote, Spinner, StatusBadge } from "./ui";

function lastPayload<T>(events: AgentEvent[], stage: Stage, key: string): T | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const p = events[i].payload;
    if (events[i].stage === stage && p && key in p) return p[key] as T;
  }
  return undefined;
}

/** Grounding rungs of the most recent planning round. */
function latestGrounding(events: AgentEvent[]) {
  let start = -1;
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].stage === "ground" && events[i].payload && "predicted" in events[i].payload!) {
      start = i;
      break;
    }
  }
  if (start < 0) return null;
  const round = events.slice(start).filter((e) => e.stage === "ground");
  return {
    predicted: round[0].payload!.predicted as Record<string, number>,
    rungs: round.filter((e) => e.payload && "rung" in e.payload).map((e) => e.payload!.rung as RungResult),
    stoppedForBudget: round.some((e) => e.kind === "warning" && e.message.includes("budget")),
  };
}

/** Live and historical view of one pipeline run. */
export function RunView({ runId }: { runId: string }) {
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const refresh = () =>
      api.getRun(runId).then(
        (r) => !cancelled && setRun(r),
        (e) => !cancelled && setError((e as Error).message),
      );
    refresh();
    const unsubscribe = subscribeToRun(runId, {
      onEvent: (e) => {
        setEvents((prev) => (prev.some((p) => p.seq === e.seq) ? prev : [...prev, e]));
        if (e.stage === "done") refresh();
      },
      onEnd: refresh,
      onError: () => !cancelled && setError("Lost connection to the event stream."),
    });
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [runId]);

  const derived = useMemo(() => {
    const reached = new Set<Stage>(events.map((e) => e.stage));
    const nonDone = events.filter((e) => e.stage !== "done");
    const progress = [...events].reverse().find((e) => e.stage === "implement" && e.payload && "progress" in e.payload);
    return {
      reached,
      current: nonDone.length ? nonDone[nonDone.length - 1].stage : null,
      spec: lastPayload<TaskSpec>(events, "verify_request", "task_spec"),
      split: lastPayload<{ train: number; valid: number; test: number }>(events, "prepare", "split"),
      knowledge: lastPayload<KnowledgeRef[]>(events, "retrieve", "knowledge"),
      ranked: lastPayload<PlanEvaluation[]>(events, "select", "ranked"),
      selected: lastPayload<string>(events, "select", "selected") ?? null,
      grounding: latestGrounding(events),
      liveCode: lastPayload<string>(events, "implement", "code"),
      liveMetrics: lastPayload<ExecutionMetrics>(events, "verify_impl", "metrics"),
      liveBudget: lastPayload<BudgetSnapshot>(events, "verify_impl", "budget"),
      progress: progress?.payload?.progress as { stage: string; n_train?: number } | undefined,
    };
  }, [events]);

  if (error && !run) return <ErrorNote>{error}</ErrorNote>;
  if (!run)
    return (
      <p className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> Loading run…
      </p>
    );

  const finished = run.status === "succeeded" || run.status === "failed";
  const spec = run.task_spec ?? derived.spec ?? null;
  const metrics = run.metrics ?? derived.liveMetrics;
  const code = run.code ?? derived.liveCode;
  const budget = run.metrics?.budget ?? derived.liveBudget;
  const cfg = run.config;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight">Run {run.id}</h1>
            <StatusBadge status={run.status} />
          </div>
          <p className="mt-1 max-w-3xl text-sm text-muted">“{run.prompt}”</p>
          {cfg && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              <Badge tone={cfg.verification_mode === "grounded" ? "accent" : "neutral"}>
                {cfg.verification_mode === "grounded" ? "grounded verification" : "pseudo verification (paper)"}
              </Badge>
              <Badge tone={cfg.memory?.enabled ? "success" : "neutral"}>
                memory {cfg.memory?.enabled ? "on" : "off"}
              </Badge>
              {cfg.agent_fusion && <Badge>fused agents</Badge>}
              {Object.entries(cfg.models ?? {}).map(([role, model]) => (
                <Badge key={role}>
                  {role}: {model}
                </Badge>
              ))}
            </div>
          )}
        </div>
        {run.llm_usage && (
          <div className="text-right text-xs text-muted">
            <div>{run.llm_usage.calls} LLM calls</div>
            <div>{(run.llm_usage.total_tokens ?? run.llm_usage.input_tokens + run.llm_usage.output_tokens).toLocaleString()} tokens</div>
          </div>
        )}
      </div>

      <StageStepper reached={derived.reached} current={derived.current} finished={finished} failed={run.status === "failed"} />

      {!finished && derived.current === "implement" && derived.progress && (
        <p className="flex items-center gap-2 text-sm text-muted">
          <Spinner className="text-accent" /> Final training on the full data: {derived.progress.stage}
          {derived.progress.n_train ? ` (${derived.progress.n_train.toLocaleString()} rows)` : ""}
        </p>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
      {run.error && <ErrorNote>{run.error}</ErrorNote>}

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          {metrics && metrics.score !== undefined && (
            <Card>
              <CardTitle aside={metrics.split === "test" ? <Badge>held-out test split</Badge> : undefined}>Result</CardTitle>
              <MetricsPanel metrics={metrics} spec={spec} />
            </Card>
          )}

          {derived.grounding && (
            <Card>
              <CardTitle aside={<Badge tone="accent">successive halving</Badge>}>Grounded verification</CardTitle>
              <GroundingPanel
                rungs={derived.grounding.rungs}
                predicted={derived.grounding.predicted}
                selected={derived.selected}
                metric={spec?.metric}
                stoppedForBudget={derived.grounding.stoppedForBudget}
              />
            </Card>
          )}

          {derived.ranked && (
            <Card>
              <CardTitle>Candidate plans</CardTitle>
              <PlanCards ranked={derived.ranked} selectedId={derived.selected} metric={spec?.metric} />
            </Card>
          )}

          {code && (
            <Card>
              <CardTitle>Generated code</CardTitle>
              <CodeViewer code={code} />
            </Card>
          )}
        </div>

        <div className="space-y-6">
          {spec && (
            <Card>
              <CardTitle>Task specification</CardTitle>
              <dl className="space-y-2 text-sm">
                <Row label="Task" value={spec.task_type.replaceAll("_", " ")} />
                <Row label="Target" value={spec.target_column} mono />
                {spec.text_column && <Row label="Text column" value={spec.text_column} mono />}
                <Row label="Metric" value={spec.metric} mono />
                {spec.metric_target != null && <Row label="Target value" value={String(spec.metric_target)} />}
                {spec.drop_columns.length > 0 && <Row label="Dropped" value={spec.drop_columns.join(", ")} mono />}
                {derived.split && (
                  <Row
                    label="Split"
                    value={`${derived.split.train.toLocaleString()} / ${derived.split.valid.toLocaleString()} / ${derived.split.test.toLocaleString()}`}
                  />
                )}
              </dl>
              {spec.assumptions?.length > 0 && (
                <ul className="mt-3 list-disc space-y-1 pl-4 text-xs text-muted">
                  {spec.assumptions.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              )}
            </Card>
          )}

          {budget && (
            <Card>
              <CardTitle>Budget</CardTitle>
              <BudgetPanel budget={budget} cacheHits={run.llm_usage?.cache_hits} />
            </Card>
          )}

          {derived.knowledge && (
            <Card>
              <CardTitle>Planning knowledge</CardTitle>
              <KnowledgePanel items={derived.knowledge} />
            </Card>
          )}

          <Card>
            <CardTitle aside={!finished && <Spinner className="text-accent" />}>Agent activity</CardTitle>
            <AgentTimeline events={events} />
          </Card>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-muted">{label}</dt>
      <dd className={mono ? "truncate font-mono text-xs leading-5" : "text-right"}>{value}</dd>
    </div>
  );
}
