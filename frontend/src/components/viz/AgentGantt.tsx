"use client";

import { formatDuration, LANE_META, type Lane, type RunModel } from "@/lib/run-model";

const W = 1000;
const LABEL_W = 128;
const LANE_H = 30;
const TOP = 34;

function niceStep(span: number): number {
  const target = span / 6;
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  return steps.find((s) => s >= target) ?? 3600;
}

/**
 * Swim-lane timeline of every agent call and sandbox job. Bars are real durations;
 * parallel bars in different lanes show agents working concurrently.
 */
export function AgentGantt({ model, nowSec, live }: { model: RunModel; nowSec: number; live: boolean }) {
  const lanes = (Object.keys(LANE_META) as Lane[]).filter((l) => model.spans.some((s) => s.lane === l));
  if (!lanes.length) {
    return <p className="text-sm text-muted">Agent activity appears here as soon as the first agent responds.</p>;
  }
  const end = Math.max(nowSec, model.now, ...model.spans.map((s) => s.end), 1);
  const plotW = W - LABEL_W - 12;
  const x = (t: number) => LABEL_W + (Math.max(0, t) / end) * plotW;
  const height = TOP + lanes.length * LANE_H + 26;
  const step = niceStep(end);
  const ticks = Array.from({ length: Math.floor(end / step) + 1 }, (_, i) => i * step);
  const bands = model.stages.filter((s) => s.start !== null);

  return (
    <div className="scroll-thin overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${height}`} className="min-w-[760px]" role="img" aria-label="Agent timeline">
        <defs>
          <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="var(--surface-muted)" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--border-strong)" strokeWidth="2" />
          </pattern>
        </defs>

        {/* stage bands */}
        {bands.map((s, i) => {
          const x0 = x(s.start ?? 0);
          const x1 = x(s.end ?? s.start ?? 0);
          return (
            <g key={s.id}>
              <rect x={x0} y={TOP - 6} width={Math.max(x1 - x0, 1)} height={lanes.length * LANE_H + 6}
                fill={i % 2 ? "var(--surface-muted)" : "transparent"} opacity={0.45} />
              {x1 - x0 > 46 && (
                <text x={x0 + 4} y={TOP - 12} fontSize={9} fontFamily="var(--font-geist-mono)" fill="var(--faint)">
                  {s.label.toUpperCase()}
                </text>
              )}
            </g>
          );
        })}

        {/* lanes */}
        {lanes.map((lane, i) => {
          const y = TOP + i * LANE_H;
          const meta = LANE_META[lane];
          return (
            <g key={lane}>
              <line x1={LABEL_W} x2={W - 12} y1={y + LANE_H} y2={y + LANE_H} stroke="var(--border)" />
              <circle cx={10} cy={y + LANE_H / 2} r={3.5} fill={meta.color} />
              <text x={20} y={y + LANE_H / 2 + 4} fontSize={11.5} fill="var(--muted)">{meta.label}</text>
              {model.spans
                .filter((s) => s.lane === lane)
                .map((s, j) => {
                  const bx = x(s.start);
                  const bw = Math.max(x(s.end) - bx, 2.5);
                  const fill = s.kind === "cached" ? "url(#hatch)" : meta.color;
                  return (
                    <g key={j}>
                      <title>{`${s.label}${s.detail ? ` — ${s.detail}` : ""}\n${formatDuration(s.start)} → ${formatDuration(s.end)} (${(s.end - s.start).toFixed(1)}s)`}</title>
                      <rect x={bx} y={y + 7} width={bw} height={LANE_H - 14} rx={4} fill={fill}
                        opacity={s.kind === "cached" ? 1 : 0.85} />
                      {bw > 70 && (
                        <text x={bx + 6} y={y + LANE_H / 2 + 3.5} fontSize={9.5} fill="var(--background)" fontWeight={600}>
                          {s.label.length > bw / 6 ? `${s.label.slice(0, Math.floor(bw / 6) - 1)}…` : s.label}
                        </text>
                      )}
                    </g>
                  );
                })}
            </g>
          );
        })}

        {/* time axis */}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={TOP - 4} y2={TOP + lanes.length * LANE_H} stroke="var(--border)" strokeDasharray="2 4" />
            <text x={x(t)} y={height - 8} fontSize={9.5} textAnchor="middle" fontFamily="var(--font-geist-mono)" fill="var(--faint)">
              {formatDuration(t)}
            </text>
          </g>
        ))}

        {/* now cursor */}
        {live && (
          <g>
            <line x1={x(nowSec)} x2={x(nowSec)} y1={TOP - 8} y2={TOP + lanes.length * LANE_H + 2} stroke="var(--accent-2)" strokeWidth={1.5} />
            <circle cx={x(nowSec)} cy={TOP - 8} r={3} fill="var(--accent-2)" className="breathe" />
          </g>
        )}
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-4 rounded-sm" style={{ background: "var(--chart-1)" }} /> LLM call
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="16" height="8" aria-hidden><rect width="16" height="8" rx="2" fill="url(#hatch)" /></svg> served from cache
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-4 rounded-sm" style={{ background: "var(--chart-5)" }} /> training job (sandbox)
        </span>
      </div>
    </div>
  );
}
