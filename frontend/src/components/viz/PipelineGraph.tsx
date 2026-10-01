"use client";

import { formatDuration, type RunModel, type StageInfo, type StageStatus } from "@/lib/run-model";
import { familyLabel } from "@/lib/stages";
import type { Stage } from "@/lib/types";

/** Layout constants (SVG user units). */
const W = 1180;
const H = 300;
const MID = 150;
const NODE_W = 84;
const NODE_H = 46;

type NodeDef = { id: Stage; x: number; short: string };

const MAIN: NodeDef[] = [
  { id: "parse", x: 54, short: "Parse" },
  { id: "verify_request", x: 150, short: "Verify" },
  { id: "prepare", x: 246, short: "Prepare" },
  { id: "retrieve", x: 342, short: "Retrieve" },
  { id: "plan", x: 438, short: "Plan" },
  // execute_plans is drawn as the plan fan-out at x=566
  { id: "ground", x: 700, short: "Ground" },
  { id: "select", x: 796, short: "Select" },
  { id: "implement", x: 900, short: "Train" },
  { id: "verify_impl", x: 1004, short: "Verify" },
];
const FAN_X = 566;
const FAN_W = 140;
const FAN_H = 50;
const DONE_X = 1112;

const STROKE: Record<StageStatus, string> = {
  done: "var(--accent)",
  active: "var(--accent-2)",
  failed: "var(--danger)",
  pending: "var(--border-strong)",
  skipped: "var(--border)",
};

function edgeState(to: StageStatus): "done" | "active" | "pending" {
  if (to === "active") return "active";
  if (to === "done" || to === "failed") return "done";
  return "pending";
}

function Edge({ d, state }: { d: string; state: "done" | "active" | "pending" }) {
  return (
    <g>
      <path d={d} fill="none" stroke="var(--border)" strokeWidth={2} />
      {state !== "pending" && (
        <path
          d={d}
          fill="none"
          stroke={state === "active" ? "var(--accent-2)" : "url(#edge-grad)"}
          strokeWidth={2}
          className={state === "active" ? "flow" : undefined}
          opacity={state === "active" ? 1 : 0.85}
        />
      )}
    </g>
  );
}

function StageNode({ def, info, index }: { def: NodeDef; info: StageInfo; index: number }) {
  const x = def.x - NODE_W / 2;
  const y = MID - NODE_H / 2;
  const stroke = STROKE[info.status];
  const dur = info.start !== null && info.end !== null ? info.end - info.start : null;
  return (
    <g>
      <title>{`${info.label} (${info.agent}) — ${info.status}`}</title>
      {info.status === "active" && (
        <rect x={x - 5} y={y - 5} width={NODE_W + 10} height={NODE_H + 10} rx={16} fill="none"
          stroke="var(--accent-2)" strokeWidth={6} className="breathe" opacity={0.4} />
      )}
      <rect x={x} y={y} width={NODE_W} height={NODE_H} rx={12} fill="var(--surface-raised)" stroke={stroke}
        strokeWidth={info.status === "pending" || info.status === "skipped" ? 1 : 1.5}
        strokeDasharray={info.status === "skipped" ? "4 4" : undefined} />
      <text x={x + 10} y={y + 17} fontSize={9} fontFamily="var(--font-geist-mono)" fill="var(--faint)"
        letterSpacing={1}>{String(index + 1).padStart(2, "0")}</text>
      {info.status === "done" && (
        <g transform={`translate(${x + NODE_W - 16}, ${y + 12})`}>
          <circle r={6} fill="var(--accent)" opacity={0.18} />
          <path d="M-2.6 0.2 L-0.6 2.2 L2.8 -1.8" stroke="var(--accent)" strokeWidth={1.4} fill="none"
            strokeLinecap="round" strokeLinejoin="round" />
        </g>
      )}
      {info.status === "failed" && (
        <text x={x + NODE_W - 16} y={y + 16} fontSize={11} fill="var(--danger)" textAnchor="middle">!</text>
      )}
      <text x={x + 10} y={y + 33} fontSize={12.5} fontWeight={600}
        fill={info.status === "pending" || info.status === "skipped" ? "var(--faint)" : "var(--foreground)"}>
        {def.short}
      </text>
      <text x={def.x} y={y + NODE_H + 16} fontSize={10} textAnchor="middle" fontFamily="var(--font-geist-mono)"
        fill={info.status === "active" ? "var(--accent-2)" : "var(--faint)"}>
        {info.status === "skipped" ? "skipped" : dur !== null ? formatDuration(dur) : info.status === "active" ? "running" : ""}
      </text>
    </g>
  );
}

/** Live DAG of the pipeline: stages, plan fan-out, knowledge sources and the revision loop. */
export function PipelineGraph({ model, nPlans }: { model: RunModel; nPlans: number }) {
  const byId = Object.fromEntries(model.stages.map((s) => [s.id, s])) as Record<Stage, StageInfo>;
  const exec = byId.execute_plans;
  const ground = byId.ground;

  // Plans: prefer the ranked evaluations, then grounding families, then placeholders.
  type PlanNode = { id: string; family: string; state: "selected" | "eliminated" | "normal" };
  const plans: PlanNode[] = ((): PlanNode[] => {
    if (model.ranked?.length) {
      const lastRung = model.grounding?.rungs.at(-1);
      return model.ranked.map((ev) => ({
        id: ev.plan.id,
        family: ev.model.model_family,
        state:
          ev.plan.id === model.selected
            ? "selected"
            : lastRung && !lastRung.results.some((r) => r.plan_id === ev.plan.id)
              ? "eliminated"
              : "normal",
      }));
    }
    if (model.grounding) {
      return Object.keys(model.grounding.predicted).map((id) => ({
        id, family: model.grounding!.families[id] ?? "analysing…", state: "normal" as const,
      }));
    }
    return Array.from({ length: nPlans }, (_, i) => ({ id: `p${i + 1}`, family: "—", state: "normal" as const }));
  })().slice(0, 4);

  const fanYs = plans.map((_, i) => MID + (i - (plans.length - 1) / 2) * (FAN_H + 12));
  const sources = new Set((model.knowledge ?? []).map((k) => (k.source.startsWith("memory:") ? "memory" : k.source)));
  const rounds = model.llmCalls.filter((c) => c.schema === "PlanSet").length;
  const finished = byId.verify_impl.status === "done" && model.stages.every((s) => s.status !== "active");

  const main = (id: Stage) => MAIN.find((m) => m.id === id)!;
  const right = (id: Stage) => main(id).x + NODE_W / 2;
  const left = (id: Stage) => main(id).x - NODE_W / 2;

  return (
    <div className="scroll-thin overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="min-w-[960px]" role="img" aria-label="Pipeline graph">
        <defs>
          <linearGradient id="edge-grad" x1="0" x2="1">
            <stop offset="0" stopColor="var(--accent)" />
            <stop offset="1" stopColor="var(--accent-2)" />
          </linearGradient>
          <radialGradient id="node-glow">
            <stop offset="0" stopColor="var(--accent-2)" stopOpacity={0.35} />
            <stop offset="1" stopColor="var(--accent-2)" stopOpacity={0} />
          </radialGradient>
        </defs>

        {/* main chain edges */}
        {MAIN.slice(0, 4).map((n, i) => (
          <Edge key={n.id} d={`M${right(n.id)} ${MID} L${left(MAIN[i + 1].id)} ${MID}`}
            state={edgeState(byId[MAIN[i + 1].id].status)} />
        ))}
        {/* plan fan-out and fan-in */}
        {fanYs.map((y, i) => (
          <g key={`fan-${i}`}>
            <Edge d={`M${right("plan")} ${MID} C ${right("plan") + 40} ${MID}, ${FAN_X - FAN_W / 2 - 40} ${y}, ${FAN_X - FAN_W / 2} ${y}`}
              state={edgeState(exec.status)} />
            <Edge d={`M${FAN_X + FAN_W / 2} ${y} C ${FAN_X + FAN_W / 2 + 40} ${y}, ${left("ground") - 40} ${MID}, ${left("ground")} ${MID}`}
              state={edgeState(ground.status === "skipped" ? byId.select.status : ground.status)} />
          </g>
        ))}
        {(["ground", "select", "implement", "verify_impl"] as Stage[]).slice(0, 3).map((id, i) => {
          const next = (["select", "implement", "verify_impl"] as Stage[])[i];
          return <Edge key={id} d={`M${right(id)} ${MID} L${left(next)} ${MID}`} state={edgeState(byId[next].status)} />;
        })}
        <Edge d={`M${right("verify_impl")} ${MID} L${DONE_X - 22} ${MID}`} state={finished ? "done" : "pending"} />

        {/* revision loop */}
        {rounds > 1 && (
          <g>
            <path d={`M${main("verify_impl").x} ${MID + NODE_H / 2} C ${main("verify_impl").x} ${H - 8}, ${main("plan").x} ${H - 8}, ${main("plan").x} ${MID + NODE_H / 2}`}
              fill="none" stroke="var(--warning)" strokeWidth={1.5} strokeDasharray="5 5" opacity={0.8} />
            <text x={(main("verify_impl").x + main("plan").x) / 2} y={H - 12} fontSize={10} textAnchor="middle"
              fontFamily="var(--font-geist-mono)" fill="var(--warning)">{`revised ×${rounds - 1}`}</text>
          </g>
        )}

        {/* knowledge sources feeding Retrieve */}
        {(["local-kb", "google-search", "memory"] as const).map((src, i) => {
          const sx = main("retrieve").x - 40 + i * 40;
          const sy = 42;
          const on = sources.has(src);
          const label = src === "local-kb" ? "KB" : src === "google-search" ? "Web" : "Mem";
          return (
            <g key={src}>
              <title>{`${src}: ${on ? "used" : "not used"}`}</title>
              <path d={`M${sx} ${sy + 12} C ${sx} ${sy + 40}, ${main("retrieve").x} ${MID - NODE_H / 2 - 30}, ${main("retrieve").x} ${MID - NODE_H / 2}`}
                fill="none" stroke={on ? "var(--accent-2)" : "var(--border)"} strokeWidth={1.2}
                strokeDasharray={on ? undefined : "3 4"} opacity={on ? 0.7 : 1} />
              <circle cx={sx} cy={sy} r={12} fill="var(--surface-raised)" stroke={on ? "var(--accent-2)" : "var(--border-strong)"} />
              <text x={sx} y={sy + 3.5} fontSize={8.5} textAnchor="middle" fontFamily="var(--font-geist-mono)"
                fill={on ? "var(--accent-2)" : "var(--faint)"}>{label}</text>
            </g>
          );
        })}

        {/* stage nodes */}
        {MAIN.map((def) => (
          <StageNode key={def.id} def={def} info={byId[def.id]} index={model.stages.findIndex((s) => s.id === def.id)} />
        ))}

        {/* plan nodes (execute_plans) */}
        {plans.map((p, i) => {
          const y = fanYs[i] - FAN_H / 2;
          const x = FAN_X - FAN_W / 2;
          const active = exec.status === "active";
          const sel = p.state === "selected";
          const elim = p.state === "eliminated";
          return (
            <g key={p.id} opacity={elim ? 0.45 : 1}>
              <title>{`${p.id}: ${p.family}${sel ? " (selected)" : elim ? " (eliminated)" : ""}`}</title>
              {active && <rect x={x - 4} y={y - 4} width={FAN_W + 8} height={FAN_H + 8} rx={14} fill="none"
                stroke="var(--accent-2)" strokeWidth={5} className="breathe" opacity={0.35} />}
              <rect x={x} y={y} width={FAN_W} height={FAN_H} rx={12} fill="var(--surface-raised)"
                stroke={sel ? "var(--success)" : exec.status === "pending" ? "var(--border-strong)" : "var(--accent)"}
                strokeWidth={sel ? 2 : 1.2} />
              <text x={x + 10} y={y + 18} fontSize={9.5} fontFamily="var(--font-geist-mono)" fill="var(--faint)">
                {p.id}
                {sel && <tspan fill="var(--success)"> · SELECTED</tspan>}
                {elim && <tspan> · DROPPED</tspan>}
              </text>
              <text x={x + 10} y={y + 37} fontSize={12} fontWeight={600} fill="var(--foreground)">
                {familyLabel(p.family)}
              </text>
              {/* data + model agent pips */}
              {(["D", "M"] as const).map((k, j) => (
                <g key={k} transform={`translate(${x + FAN_W - 30 + j * 13}, ${y + 33})`}>
                  <circle r={5} fill={exec.status === "pending" ? "var(--surface-muted)" : j ? "var(--chart-4)" : "var(--chart-2)"}
                    opacity={exec.status === "pending" ? 1 : 0.85} />
                  <text y={2.6} fontSize={6.5} textAnchor="middle" fill="var(--background)" fontWeight={700}>{k}</text>
                </g>
              ))}
            </g>
          );
        })}
        <text x={FAN_X} y={fanYs[0] - FAN_H / 2 - 10} fontSize={10} textAnchor="middle" fontFamily="var(--font-geist-mono)"
          fill={exec.status === "active" ? "var(--accent-2)" : "var(--faint)"}>
          {exec.start !== null && exec.end !== null ? `analyse · ${formatDuration(exec.end - exec.start)}` : "analyse"}
        </text>

        {/* done node */}
        <g>
          {finished && <circle cx={DONE_X} cy={MID} r={30} fill="url(#node-glow)" />}
          <circle cx={DONE_X} cy={MID} r={20} fill="var(--surface-raised)"
            stroke={finished ? "var(--success)" : "var(--border-strong)"} strokeWidth={finished ? 2 : 1} />
          <path d={`M${DONE_X - 6} ${MID} L${DONE_X - 1.5} ${MID + 4.5} L${DONE_X + 7} ${MID - 5}`} fill="none"
            stroke={finished ? "var(--success)" : "var(--faint)"} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
          <text x={DONE_X} y={MID + 40} fontSize={10} textAnchor="middle" fontFamily="var(--font-geist-mono)" fill="var(--faint)">
            {finished ? "model ready" : "result"}
          </text>
        </g>
      </svg>
    </div>
  );
}
