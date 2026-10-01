"use client";

import { formatDuration, type RunModel } from "@/lib/run-model";

const W = 1000;
const H = 210;
const PAD = { l: 46, r: 46, t: 16, b: 46 };

/**
 * Live CPU and memory of the training sandbox: memory as a filled area (left axis),
 * CPU cores in use as a line (right axis), with each grounding run and the final
 * training shown as segments underneath.
 */
export function ResourceMonitor({ model }: { model: RunModel }) {
  const pts = model.telemetry;
  const jobs = model.jobs;
  if (!pts.length && !jobs.length) {
    return (
      <p className="text-sm text-muted">
        CPU and memory of every training job stream here while the grounding runs and the final training execute.
      </p>
    );
  }
  const t0 = Math.min(...jobs.map((j) => j.start), ...(pts.length ? [pts[0].t] : []));
  const t1 = Math.max(...jobs.map((j) => j.end), ...pts.map((p) => p.t), t0 + 1);
  const memMax = Math.max(1, ...pts.map((p) => p.rssMb)) / 1024;
  const coreMax = Math.max(1, ...pts.map((p) => p.cores));
  const x = (t: number) => PAD.l + ((t - t0) / (t1 - t0)) * (W - PAD.l - PAD.r);
  const yMem = (gb: number) => PAD.t + (1 - gb / (memMax * 1.15)) * (H - PAD.t - PAD.b);
  const yCpu = (c: number) => PAD.t + (1 - c / (coreMax * 1.15)) * (H - PAD.t - PAD.b);
  const base = H - PAD.b;

  // Draw each job's samples as its own segment so gaps between jobs stay empty.
  const byJob = new Map<string, typeof pts>();
  pts.forEach((p) => byJob.set(p.job, [...(byJob.get(p.job) ?? []), p]));

  const peak = pts.length ? Math.max(...pts.map((p) => p.rssMb)) : 0;
  const maxCores = pts.length ? Math.max(...pts.map((p) => p.cores)) : 0;
  const lastPt = pts.at(-1);

  return (
    <div>
      <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Mini label="Peak memory" value={`${(peak / 1024).toFixed(2)} GB`} color="var(--chart-2)" />
        <Mini label="Max CPU" value={`${maxCores.toFixed(1)} cores`} color="var(--chart-3)" />
        <Mini label="Training jobs" value={String(jobs.length)} color="var(--chart-5)" />
        <Mini label="Last sample" value={lastPt ? `${(lastPt.rssMb / 1024).toFixed(2)} GB · ${lastPt.cores.toFixed(1)}c` : "—"} color="var(--muted)" />
      </div>
      <div className="scroll-thin overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="min-w-[700px]" role="img" aria-label="Sandbox resources">
          <defs>
            <linearGradient id="mem-fill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="var(--chart-2)" stopOpacity={0.45} />
              <stop offset="1" stopColor="var(--chart-2)" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={PAD.l} x2={W - PAD.r} y1={yMem(memMax * 1.15 * f)} y2={yMem(memMax * 1.15 * f)} stroke="var(--border)" strokeDasharray="2 4" />
              <text x={PAD.l - 6} y={yMem(memMax * 1.15 * f) + 3.5} fontSize={9.5} textAnchor="end" fontFamily="var(--font-geist-mono)" fill="var(--chart-2)">
                {(memMax * 1.15 * f).toFixed(1)}G
              </text>
              <text x={W - PAD.r + 6} y={yCpu(coreMax * 1.15 * f) + 3.5} fontSize={9.5} fontFamily="var(--font-geist-mono)" fill="var(--chart-3)">
                {(coreMax * 1.15 * f).toFixed(0)}c
              </text>
            </g>
          ))}
          {[...byJob.entries()].map(([job, series]) => {
            const area = `M${x(series[0].t)} ${base} ${series.map((p) => `L${x(p.t)} ${yMem(p.rssMb / 1024)}`).join(" ")} L${x(series.at(-1)!.t)} ${base} Z`;
            const cpu = series.map((p, i) => `${i ? "L" : "M"}${x(p.t)} ${yCpu(p.cores)}`).join(" ");
            return (
              <g key={job}>
                <path d={area} fill="url(#mem-fill)" stroke="var(--chart-2)" strokeWidth={1.4} />
                <path d={cpu} fill="none" stroke="var(--chart-3)" strokeWidth={1.4} />
              </g>
            );
          })}
          {/* job segments */}
          {jobs.map((j, i) => {
            const x0 = x(j.start);
            const w = Math.max(x(j.end) - x0, 3);
            return (
              <g key={j.id}>
                <title>{`${j.label}: ${formatDuration(j.end - j.start)}${j.done ? "" : " (running)"}`}</title>
                <rect x={x0} y={base + 8} width={w} height={12} rx={3}
                  fill={j.kind === "final" ? "var(--chart-5)" : i % 2 ? "var(--chart-1)" : "var(--chart-6)"}
                  opacity={j.done ? 0.75 : 1} className={j.done ? undefined : "breathe"} />
                {w > 64 && (
                  <text x={x0 + 5} y={base + 17.5} fontSize={8.5} fill="var(--background)" fontWeight={600}>
                    {j.label}
                  </text>
                )}
              </g>
            );
          })}
          <text x={PAD.l} y={H - 6} fontSize={9.5} fontFamily="var(--font-geist-mono)" fill="var(--faint)">{formatDuration(t0)}</text>
          <text x={W - PAD.r} y={H - 6} fontSize={9.5} textAnchor="end" fontFamily="var(--font-geist-mono)" fill="var(--faint)">
            {formatDuration(t1)}
          </text>
        </svg>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-4 rounded-sm" style={{ background: "var(--chart-2)" }} /> memory (process tree)</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-4" style={{ background: "var(--chart-3)" }} /> CPU cores in use</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-4 rounded-sm" style={{ background: "var(--chart-6)" }} /> grounding run</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-4 rounded-sm" style={{ background: "var(--chart-5)" }} /> final training</span>
      </div>
    </div>
  );
}

function Mini({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface-muted/40 px-3 py-2">
      <p className="eyebrow">{label}</p>
      <p className="tabular mt-0.5 text-sm font-semibold" style={{ color }}>{value}</p>
    </div>
  );
}
