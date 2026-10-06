"use client";

import { useEffect, useMemo, useState } from "react";

import { api, subscribeToRun } from "@/lib/api";
import { buildRunModel, formatDuration } from "@/lib/run-model";
import { formatMetric } from "@/lib/stages";
import type { AgentEvent, ExecutionMetrics, Run } from "@/lib/types";
import { useNow } from "@/lib/use-now";
import { AgentTimeline } from "./AgentTimeline";
import { ApprovalCard } from "./ApprovalCard";
import { AuditPanel } from "./AuditPanel";
import { CodeViewer } from "./CodeViewer";
import { GroundingPanel } from "./GroundingPanel";
import { KnowledgePanel } from "./KnowledgePanel";
import { MetricsPanel } from "./MetricsPanel";
import { ModelCardPanel } from "./ModelCardPanel";
import { PlanCards } from "./PlanCards";
import { Badge, Button, Card, CardTitle, cn, ErrorNote, LiveDot, Spinner, Stat, StatusBadge } from "./ui";
import { UseModelPanel } from "./UseModelPanel";
import { AgentGantt } from "./viz/AgentGantt";
import { BudgetGauges, SplitBar } from "./viz/Gauges";
import { GroundingChart } from "./viz/GroundingChart";
import { LlmActivity } from "./viz/LlmActivity";
import { PipelineGraph } from "./viz/PipelineGraph";
import { ResourceMonitor } from "./viz/ResourceMonitor";

/** Live "mission control" view of one pipeline run. */
export function RunView({ runId }: { runId: string }) {
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [groundView, setGroundView] = useState<"chart" | "table">("chart");

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
        setEvents((prev) => (prev.length && prev[prev.length - 1].seq >= e.seq ? prev : [...prev, e]));
        if (
          e.stage === "done" ||
          (e.payload && Boolean((e.payload as Record<string, unknown>).approval_step)) ||
          e.message?.toLowerCase().includes("resumed")
        ) {
          refresh();
        }
      },
      onEnd: refresh,
      onError: () => !cancelled && setError("Lost connection to the event stream."),
    });
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [runId]);

  const isAwaitingInput = useMemo(() => {
    if (run?.status === "succeeded" || run?.status === "failed" || run?.status === "cancelled") {
      return false;
    }
    if (run?.status === "running") return false;
    if (run?.status === "awaiting_input") return true;
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (
        e.stage === "done" ||
        e.stage === "implement" ||
        e.message?.toLowerCase().includes("approved") ||
        e.message?.toLowerCase().includes("resumed")
      ) {
        return false;
      }
      if ((e.payload as Record<string, unknown>)?.approval_step) return true;
    }
    return false;
  }, [run?.status, events]);

  const finished = run?.status === "succeeded" || run?.status === "failed" || run?.status === "cancelled";
  const isUnsuccessful = run?.status === "failed" || run?.status === "cancelled";
  const model = useMemo(() => buildRunModel(events, finished, isUnsuccessful), [events, finished, isUnsuccessful]);
  const nowMs = useNow(!finished && !!run);
  const nowSec = finished ? model.now : Math.max(model.now, (nowMs - model.t0) / 1000);

  if (error && !run) return <ErrorNote>{error}</ErrorNote>;
  if (!run)
    return (
      <p className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> Loading run…
      </p>
    );

  const spec = run.task_spec ?? model.spec ?? null;
  const metrics: ExecutionMetrics | undefined = run.metrics ?? undefined;
  const code = run.code ?? model.liveCode;
  const cfg = run.config;
  const currentStage = model.stages.find((s) => s.status === "active");
  const groundBest = model.grounding?.rungs
    .flatMap((r) => r.results)
    .filter((r) => r.ok && r.score !== null)
    .map((r) => r.score as number);
  const peakMb = model.telemetry.length ? Math.max(...model.telemetry.map((p) => p.rssMb)) : null;
  const tokens = model.llmCalls.reduce((s, c) => s + c.inputTokens + c.outputTokens, 0);
  const nPlans = cfg?.n_plans ?? 3;
  const modelCard = run.metrics?.model_card ?? model.modelCard;

  return (
    <div className="space-y-5">
      {/* hero */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <p className="eyebrow mb-2 flex items-center gap-2">
            {!finished && <LiveDot className="h-1.5 w-1.5 text-accent-2" />}
            pipeline run
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-gradient font-mono text-3xl font-semibold tracking-tight">{run.id}</h1>
            <StatusBadge status={isAwaitingInput ? "awaiting_input" : run.status} />
            {run.human_override && (
              <Badge tone="accent" title="A human expert intervened in plan or code selection">
                👤 Human Guided
              </Badge>
            )}
          </div>
          <p className="mt-2 max-w-3xl text-sm text-muted">“{run.prompt}”</p>
          {cfg && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              <Badge tone={cfg.verification_mode === "grounded" ? "accent" : "neutral"}>
                {cfg.verification_mode === "grounded" ? "grounded verification" : "pseudo verification (paper)"}
              </Badge>
              <Badge tone={cfg.memory?.enabled ? "success" : "neutral"}>memory {cfg.memory?.enabled ? "on" : "off"}</Badge>
              {cfg.agent_fusion && <Badge>fused agents</Badge>}
              {Object.entries(cfg.models ?? {}).map(([role, m]) => (
                <Badge key={role}>
                  <span className="text-faint">{role}</span> {m}
                </Badge>
              ))}
            </div>
          )}
        </div>

        {!finished && (
          <Button
            variant="outline"
            size="sm"
            onClick={async () => {
              if (confirm("Are you sure you want to cancel this run?")) {
                try {
                  const cancelledRun = await api.cancelRun(run.id);
                  setRun(cancelledRun);
                } catch (e) {
                  setError((e as Error).message);
                }
              }
            }}
            className="text-danger hover:bg-danger-soft/20"
          >
            Cancel Run
          </Button>
        )}
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="elapsed" value={formatDuration(nowSec)} caption={finished ? "total" : "live"} tone={finished ? undefined : "accent"} />
        <Stat
          label="stage"
          value={
            finished
              ? run.status === "succeeded"
                ? "Complete"
                : run.status === "cancelled"
                  ? "Cancelled"
                  : "Stopped"
              : isAwaitingInput
                ? "Needs your input"
                : currentStage?.label ?? "Starting"
          }
          caption={finished ? undefined : isAwaitingInput ? "Human approval" : currentStage?.agent}
          tone={isAwaitingInput ? "warning" : undefined}
        />
        <Stat
          label={metrics?.score != null ? `test ${metrics.metric ?? ""}` : "best validation"}
          value={metrics?.score != null ? formatMetric(metrics.score) : groundBest?.length ? formatMetric(Math.max(...groundBest)) : "—"}
          caption={metrics?.target_met ? "requirements met" : metrics?.score != null ? "held-out test split" : "from grounding"}
          tone={metrics?.score != null ? "success" : undefined}
        />
        <Stat label="llm calls" value={String(run.llm_usage?.calls ?? model.llmCalls.filter((c) => !c.cached).length)}
          caption={run.llm_usage?.cache_hits ? `+${run.llm_usage.cache_hits} cached` : "Gemini"} />
        <Stat label="tokens" value={(run.llm_usage?.total_tokens ?? tokens).toLocaleString()} caption="input + output" />
        <Stat label="peak memory" value={peakMb ? `${(peakMb / 1024).toFixed(2)} GB` : "—"} caption="training sandbox" />
      </div>

      {error && <ErrorNote>{error}</ErrorNote>}
      {run.error && <ErrorNote>{run.error}</ErrorNote>}

      {/* Human approval review card when awaiting input */}
      {isAwaitingInput && (
        <ApprovalCard
          run={run}
          events={events}
          onUpdated={(updatedRun) => setRun(updatedRun)}
        />
      )}

      {/* pipeline graph */}
      <Card glow={!finished}>
        <CardTitle
          eyebrow="orchestration"
          aside={
            !finished && model.finalProgress && currentStage?.id === "implement" ? (
              <Badge tone="accent">
                <LiveDot className="h-1.5 w-1.5" /> training: {model.finalProgress.stage}
                {model.finalProgress.n_train ? ` · ${model.finalProgress.n_train.toLocaleString()} rows` : ""}
              </Badge>
            ) : undefined
          }
        >
          Pipeline
        </CardTitle>
        <PipelineGraph model={model} nPlans={nPlans} />
      </Card>

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          {metrics && metrics.score !== undefined && (
            <Card>
              <CardTitle eyebrow="outcome" aside={metrics.split === "test" ? <Badge>held-out test split</Badge> : undefined}>
                Result
              </CardTitle>
              <MetricsPanel metrics={metrics} spec={spec} />
            </Card>
          )}

          {run.status === "succeeded" && <UseModelPanel run={run} />}

          {modelCard && <ModelCardPanel card={modelCard} />}

          {model.audit && <AuditPanel audit={model.audit} />}

          {model.grounding && (
            <Card>
              <CardTitle
                eyebrow={`grounded verification · round ${model.grounding.round}`}
                aside={
                  <div role="tablist" className="flex rounded-lg border border-border bg-surface-muted/50 p-0.5 text-xs">
                    {(["chart", "table"] as const).map((v) => (
                      <button key={v} role="tab" aria-selected={groundView === v} onClick={() => setGroundView(v)}
                        className={cn("rounded-md px-2.5 py-1 capitalize", groundView === v ? "bg-surface font-medium shadow-sm" : "text-muted")}>
                        {v}
                      </button>
                    ))}
                  </div>
                }
              >
                Predicted vs observed
              </CardTitle>
              {groundView === "chart" ? (
                <GroundingChart grounding={model.grounding} selected={model.selected} metric={spec?.metric} />
              ) : (
                <GroundingPanel
                  rungs={model.grounding.rungs}
                  predicted={model.grounding.predicted}
                  selected={model.selected}
                  metric={spec?.metric}
                  stoppedForBudget={model.grounding.stoppedForBudget}
                />
              )}
            </Card>
          )}

          <Card>
            <CardTitle eyebrow="who did what, when">Agent timeline</CardTitle>
            <AgentGantt model={model} nowSec={nowSec} live={!finished} />
          </Card>

          <Card>
            <CardTitle eyebrow="sandbox telemetry">Compute</CardTitle>
            <ResourceMonitor model={model} />
          </Card>

          {model.ranked && (
            <Card>
              <CardTitle eyebrow="planning">Candidate plans</CardTitle>
              <PlanCards ranked={model.ranked} selectedId={model.selected} metric={spec?.metric} />
            </Card>
          )}

          {code && (
            <Card>
              <CardTitle eyebrow="operation agent">Generated code</CardTitle>
              <CodeViewer code={code} />
            </Card>
          )}
        </div>

        <div className="space-y-5">
          {spec && (
            <Card>
              <CardTitle eyebrow="prompt agent">Task specification</CardTitle>
              <dl className="space-y-2 text-sm">
                <Row label="Task" value={spec.task_type.replaceAll("_", " ")} />
                <Row label="Target" value={spec.target_column} mono />
                {spec.time_column && <Row label="Time column" value={spec.time_column} mono />}
                {spec.horizon != null && <Row label="Horizon" value={`${spec.horizon} steps`} />}
                {spec.frequency && <Row label="Frequency" value={spec.frequency} mono />}
                {spec.series_id_columns && spec.series_id_columns.length > 0 && (
                  <Row label="Series IDs" value={spec.series_id_columns.join(", ")} mono />
                )}
                {spec.text_column && <Row label="Text column" value={spec.text_column} mono />}
                <Row label="Metric" value={spec.metric} mono />
                {spec.metric_target != null && <Row label="Target value" value={String(spec.metric_target)} />}
                {spec.drop_columns.length > 0 && <Row label="Dropped" value={spec.drop_columns.join(", ")} mono />}
              </dl>
              {model.split && (
                <div className="mt-4 border-t border-border pt-4">
                  <SplitBar split={model.split} />
                </div>
              )}
              {spec.assumptions?.length > 0 && (
                <ul className="mt-4 list-disc space-y-1 border-t border-border pt-3 pl-4 text-xs text-muted">
                  {spec.assumptions.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              )}
            </Card>
          )}

          <Card>
            <CardTitle eyebrow="limits">Budget</CardTitle>
            <BudgetGauges budget={run.metrics?.budget ?? model.budget} elapsed={nowSec} />
          </Card>

          <Card>
            <CardTitle eyebrow="gemini">LLM activity</CardTitle>
            <LlmActivity calls={model.llmCalls} />
          </Card>

          {model.knowledge && (
            <Card>
              <CardTitle eyebrow="retrieval-augmented planning">Planning knowledge</CardTitle>
              <KnowledgePanel items={model.knowledge} />
            </Card>
          )}

          <Card>
            <CardTitle eyebrow="event stream" aside={!finished && <LiveDot className="mt-1 h-2 w-2 text-accent-2" />}>
              Agent activity
            </CardTitle>
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
