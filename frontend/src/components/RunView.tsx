"use client";

import { useEffect, useMemo, useState } from "react";

import { api, subscribeToRun } from "@/lib/api";
import type { AgentEvent, ExecutionMetrics, PlanEvaluation, Run, Stage, TaskSpec } from "@/lib/types";
import { AgentTimeline } from "./AgentTimeline";
import { CodeViewer } from "./CodeViewer";
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
    return {
      reached,
      current: nonDone.length ? nonDone[nonDone.length - 1].stage : null,
      spec: lastPayload<TaskSpec>(events, "verify_request", "task_spec"),
      knowledge: lastPayload<{ id: string; title: string; source: string }[]>(events, "retrieve", "knowledge"),
      ranked: lastPayload<PlanEvaluation[]>(events, "select", "ranked"),
      selected: lastPayload<string>(events, "select", "selected") ?? null,
      liveCode: lastPayload<string>(events, "implement", "code"),
      liveMetrics: lastPayload<ExecutionMetrics>(events, "verify_impl", "metrics"),
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

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight">Run {run.id}</h1>
            <StatusBadge status={run.status} />
          </div>
          <p className="mt-1 max-w-3xl text-sm text-muted">“{run.prompt}”</p>
        </div>
        {run.llm_usage && (
          <div className="text-right text-xs text-muted">
            <div>{run.llm_usage.calls} LLM calls</div>
            <div>
              {(run.llm_usage.input_tokens + run.llm_usage.output_tokens).toLocaleString()} tokens
            </div>
          </div>
        )}
      </div>

      <StageStepper
        reached={derived.reached}
        current={derived.current}
        finished={finished}
        failed={run.status === "failed"}
      />

      {error && <ErrorNote>{error}</ErrorNote>}
      {run.error && <ErrorNote>{run.error}</ErrorNote>}

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          {metrics && metrics.score !== undefined && (
            <Card>
              <CardTitle>Result</CardTitle>
              <MetricsPanel metrics={metrics} spec={spec} />
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
              </dl>
            </Card>
          )}

          {derived.knowledge && (
            <Card>
              <CardTitle>Retrieved knowledge</CardTitle>
              <ul className="space-y-2 text-sm">
                {derived.knowledge.map((k) => (
                  <li key={k.id} className="flex items-start justify-between gap-2">
                    <span>{k.title}</span>
                    <Badge>{k.source}</Badge>
                  </li>
                ))}
              </ul>
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
