"use client";

import { familyLabel, formatMetric, formatRows, higherIsBetter } from "@/lib/stages";
import { SERIES, type GroundingRound } from "@/lib/run-model";

const W = 640;
const H = 300;
const PAD = { l: 52, r: 128, t: 18, b: 40 };
const PRED_X = PAD.l + 26; // column where predictions are drawn

/**
 * Successive halving as a chart: each plan's predicted score (hollow diamond) next to its
 * observed validation score at every rung (log-scaled training rows). The gap between the
 * diamond and the first dot is the pseudo-execution error the paper never measures.
 */
export function GroundingChart({
  grounding,
  selected,
  metric,
}: {
  grounding: GroundingRound;
  selected: string | null;
  metric: string | undefined;
}) {
  const plans = Object.keys(grounding.predicted);
  const higher = higherIsBetter(metric);
  const points = grounding.rungs.flatMap((r) =>
    r.results.filter((x) => x.ok && x.score !== null).map((x) => ({ plan: x.plan_id, rows: x.fidelity_rows, score: x.score as number })),
  );
  const rowsSeen = points.map((p) => p.rows).filter((r) => r > 0);
  if (!rowsSeen.length && !plans.length) return null;

  const minRows = Math.max(1, Math.min(...(rowsSeen.length ? rowsSeen : [1000])) / 1.6);
  const maxRows = Math.max(...(rowsSeen.length ? rowsSeen : [grounding.nTrain || 10_000])) * 1.35;
  const lx = (rows: number) =>
    PRED_X + 34 + ((Math.log10(rows) - Math.log10(minRows)) / (Math.log10(maxRows) - Math.log10(minRows) || 1)) * (W - PAD.r - PRED_X - 34);

  const values = [...points.map((p) => p.score), ...plans.map((p) => grounding.predicted[p])].filter(Number.isFinite);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const pad = (hi - lo) * 0.15 || 0.02;
  lo -= pad;
  hi += pad;
  const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const xTicks = [...new Set(grounding.rungs.map((r) => r.results[0]?.fidelity_rows).filter(Boolean) as number[])];

  return (
    <div className="scroll-thin overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="min-w-[560px]" role="img" aria-label="Grounding chart">
        <defs>
          <filter id="line-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* grid */}
        {yTicks.map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="var(--border)" strokeDasharray="2 4" />
            <text x={PAD.l - 8} y={y(v) + 3.5} fontSize={9.5} textAnchor="end" fontFamily="var(--font-geist-mono)" fill="var(--faint)">
              {formatMetric(v)}
            </text>
          </g>
        ))}
        <line x1={PRED_X + 17} x2={PRED_X + 17} y1={PAD.t} y2={H - PAD.b} stroke="var(--border-strong)" />
        <text x={PRED_X} y={H - PAD.b + 16} fontSize={9.5} textAnchor="middle" fontFamily="var(--font-geist-mono)" fill="var(--faint)">
          predicted
        </text>
        {xTicks.map((r) => (
          <g key={r}>
            <line x1={lx(r)} x2={lx(r)} y1={PAD.t} y2={H - PAD.b} stroke="var(--border)" strokeDasharray="2 4" />
            <text x={lx(r)} y={H - PAD.b + 16} fontSize={9.5} textAnchor="middle" fontFamily="var(--font-geist-mono)" fill="var(--faint)">
              {formatRows(r)}
            </text>
          </g>
        ))}
        <text x={(PRED_X + W - PAD.r) / 2 + 20} y={H - 6} fontSize={10} textAnchor="middle" fill="var(--muted)">
          training rows (log scale) →
        </text>
        <text x={14} y={PAD.t + (H - PAD.t - PAD.b) / 2} fontSize={10} textAnchor="middle" fill="var(--muted)"
          transform={`rotate(-90 14 ${PAD.t + (H - PAD.t - PAD.b) / 2})`}>
          {`${metric ?? "score"} ${higher ? "↑" : "↓"}`}
        </text>

        {/* series */}
        {plans.map((pid, i) => {
          const color = SERIES[i % SERIES.length];
          const obs = points.filter((p) => p.plan === pid).sort((a, b) => a.rows - b.rows);
          const sel = pid === selected;
          const pred = grounding.predicted[pid];
          const lastObs = obs.at(-1);
          const path = obs.map((p, j) => `${j ? "L" : "M"}${lx(p.rows)} ${y(p.score)}`).join(" ");
          return (
            <g key={pid} opacity={selected && !sel ? 0.75 : 1}>
              {Number.isFinite(pred) && obs[0] && (
                <path d={`M${PRED_X} ${y(pred)} L${lx(obs[0].rows)} ${y(obs[0].score)}`} stroke={color} strokeWidth={1}
                  strokeDasharray="3 4" fill="none" opacity={0.6} />
              )}
              {path && <path d={path} stroke={color} strokeWidth={sel ? 2.6 : 1.6} fill="none"
                filter={sel ? "url(#line-glow)" : undefined} />}
              {Number.isFinite(pred) && (
                <g>
                  <title>{`${pid} predicted ${formatMetric(pred)}`}</title>
                  <rect x={PRED_X - 5} y={y(pred) - 5} width={10} height={10} transform={`rotate(45 ${PRED_X} ${y(pred)})`}
                    fill="var(--surface)" stroke={color} strokeWidth={1.5} />
                </g>
              )}
              {obs.map((p) => (
                <g key={p.rows}>
                  <title>{`${pid} · ${formatRows(p.rows)} rows · ${formatMetric(p.score)}`}</title>
                  <circle cx={lx(p.rows)} cy={y(p.score)} r={sel ? 5 : 4} fill={color} stroke="var(--surface)" strokeWidth={1.5} />
                </g>
              ))}
              {lastObs && (
                <text x={lx(lastObs.rows) + 10} y={y(lastObs.score) + 3.5} fontSize={10.5} fill={color} fontWeight={sel ? 700 : 500}>
                  {`${pid} ${formatMetric(lastObs.score)}${sel ? " ★" : ""}`}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
        {plans.map((pid, i) => (
          <span key={pid} className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: SERIES[i % SERIES.length] }} />
            <span className="font-mono">{pid}</span> {familyLabel(grounding.families[pid])}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <svg width="10" height="10" aria-hidden><rect x="2" y="2" width="6" height="6" transform="rotate(45 5 5)" fill="none" stroke="var(--muted)" /></svg>
          LLM prediction
        </span>
      </div>
    </div>
  );
}
