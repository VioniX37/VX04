import { formatDuration } from "@/lib/run-model";
import type { BudgetSnapshot } from "@/lib/types";

function Ring({ label, value, limit, display }: { label: string; value: number; limit: number | null; display: string }) {
  const r = 30;
  const c = 2 * Math.PI * r;
  const share = limit ? Math.min(value / limit, 1) : null;
  const color = share === null ? "var(--accent)" : share > 0.9 ? "var(--danger)" : share > 0.7 ? "var(--warning)" : "var(--accent-2)";
  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg viewBox="0 0 80 80" className="h-20 w-20" aria-hidden>
        <circle cx={40} cy={40} r={r} fill="none" stroke="var(--surface-muted)" strokeWidth={6} />
        <circle cx={40} cy={40} r={r} fill="none" stroke={color} strokeWidth={6} strokeLinecap="round"
          strokeDasharray={share === null ? "2 5" : `${share * c} ${c}`} transform="rotate(-90 40 40)" opacity={share === null ? 0.5 : 1} />
        <text x={40} y={44} fontSize={12} textAnchor="middle" fontWeight={600} fill="var(--foreground)" className="tabular">
          {share === null ? "∞" : `${Math.round(share * 100)}%`}
        </text>
      </svg>
      <p className="eyebrow">{label}</p>
      <p className="tabular -mt-1 text-xs text-muted">{display}</p>
    </div>
  );
}

/** Radial gauges for the run's wall-clock, LLM-call and token budgets. */
export function BudgetGauges({ budget, elapsed }: { budget: BudgetSnapshot | undefined; elapsed: number }) {
  const b = budget ?? { elapsed_s: elapsed, wall_budget_s: null, llm_calls: 0, llm_call_budget: null, tokens: 0, token_budget: null, wall_remaining_s: null };
  return (
    <div className="grid grid-cols-3 gap-2">
      <Ring label="time" value={elapsed} limit={b.wall_budget_s} display={b.wall_budget_s ? `${formatDuration(elapsed)} / ${formatDuration(b.wall_budget_s)}` : formatDuration(elapsed)} />
      <Ring label="llm calls" value={b.llm_calls} limit={b.llm_call_budget} display={b.llm_call_budget ? `${b.llm_calls} / ${b.llm_call_budget}` : String(b.llm_calls)} />
      <Ring label="tokens" value={b.tokens} limit={b.token_budget} display={b.token_budget ? `${(b.tokens / 1000).toFixed(0)}k / ${(b.token_budget / 1000).toFixed(0)}k` : `${(b.tokens / 1000).toFixed(1)}k`} />
    </div>
  );
}

/** Stacked bar of the fixed train / validation / test split. */
export function SplitBar({ split }: { split: { train: number; valid: number; test: number } }) {
  const total = split.train + split.valid + split.test || 1;
  const parts = [
    { k: "train", v: split.train, c: "var(--chart-1)", note: "fit + grounding samples" },
    { k: "valid", v: split.valid, c: "var(--chart-2)", note: "early stopping + ranking" },
    { k: "test", v: split.test, c: "var(--chart-3)", note: "final score only" },
  ];
  return (
    <div>
      <div className="flex h-2.5 overflow-hidden rounded-full">
        {parts.map((p) => (
          <div key={p.k} style={{ width: `${(p.v / total) * 100}%`, background: p.c }} title={`${p.k}: ${p.v.toLocaleString()}`} />
        ))}
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2">
        {parts.map((p) => (
          <div key={p.k}>
            <p className="eyebrow flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: p.c }} />
              {p.k}
            </p>
            <p className="tabular text-sm font-semibold">{p.v.toLocaleString()}</p>
            <p className="text-[10.5px] text-faint">{p.note}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
