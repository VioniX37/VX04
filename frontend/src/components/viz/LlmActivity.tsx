import { LANE_META, type Lane, type LlmCall } from "@/lib/run-model";

/** Tokens per agent and calls per Gemini model (including which fallback model answered). */
export function LlmActivity({ calls }: { calls: LlmCall[] }) {
  if (!calls.length) return <p className="text-sm text-muted">No LLM calls yet.</p>;

  const byAgent = new Map<string, { inTok: number; outTok: number; calls: number; cached: number }>();
  const byModel = new Map<string, number>();
  for (const c of calls) {
    const a = byAgent.get(c.agent) ?? { inTok: 0, outTok: 0, calls: 0, cached: 0 };
    a.inTok += c.inputTokens;
    a.outTok += c.outputTokens;
    a.calls += 1;
    a.cached += c.cached ? 1 : 0;
    byAgent.set(c.agent, a);
    for (const m of c.cached ? ["cache"] : c.models.length ? c.models : [c.role]) byModel.set(m, (byModel.get(m) ?? 0) + 1);
  }
  const maxTok = Math.max(1, ...[...byAgent.values()].map((a) => a.inTok + a.outTok));
  const totalCalls = calls.length;
  const totalTok = calls.reduce((s, c) => s + c.inputTokens + c.outputTokens, 0);
  const avgLatency = calls.filter((c) => !c.cached).reduce((s, c, _, arr) => s + c.duration / arr.length, 0);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2 text-center">
        {[
          ["calls", totalCalls.toString()],
          ["tokens", totalTok >= 1000 ? `${(totalTok / 1000).toFixed(1)}k` : String(totalTok)],
          ["avg latency", `${avgLatency.toFixed(1)}s`],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg border border-border bg-surface-muted/40 px-2 py-2">
            <p className="eyebrow">{k}</p>
            <p className="tabular text-sm font-semibold">{v}</p>
          </div>
        ))}
      </div>

      <div className="space-y-2">
        <p className="eyebrow">tokens by agent</p>
        {[...byAgent.entries()].map(([agent, a]) => {
          const color = LANE_META[agent as Lane]?.color ?? "var(--chart-1)";
          const total = a.inTok + a.outTok;
          return (
            <div key={agent}>
              <div className="mb-1 flex justify-between text-xs">
                <span className="text-muted">{LANE_META[agent as Lane]?.label ?? agent}</span>
                <span className="tabular text-faint">
                  {a.calls} call{a.calls > 1 ? "s" : ""} · {total.toLocaleString()} tok
                </span>
              </div>
              <div className="flex h-1.5 overflow-hidden rounded-full bg-surface-muted">
                <div style={{ width: `${(a.inTok / maxTok) * 100}%`, background: color, opacity: 0.55 }} />
                <div style={{ width: `${(a.outTok / maxTok) * 100}%`, background: color }} />
              </div>
            </div>
          );
        })}
      </div>

      <div>
        <p className="eyebrow mb-2">answered by</p>
        <div className="flex flex-wrap gap-1.5">
          {[...byModel.entries()].map(([m, n]) => (
            <span key={m} className="rounded-full border border-border bg-surface-muted/50 px-2 py-0.5 font-mono text-[10.5px] text-muted">
              {m} <span className="text-foreground">×{n}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
