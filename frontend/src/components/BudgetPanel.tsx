import type { BudgetSnapshot } from "@/lib/types";
import { cn } from "./ui";

function Meter({ label, used, limit, unit }: { label: string; used: number; limit: number | null; unit?: string }) {
  const share = limit ? Math.min(used / limit, 1) : null;
  return (
    <div>
      <div className="flex justify-between text-xs">
        <span className="text-muted">{label}</span>
        <span className="tabular-nums">
          {used.toLocaleString()}
          {unit}
          {limit ? ` / ${limit.toLocaleString()}${unit ?? ""}` : " (no limit)"}
        </span>
      </div>
      {share !== null && (
        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-muted">
          <div
            className={cn("h-full", share > 0.9 ? "bg-danger" : share > 0.7 ? "bg-warning" : "bg-accent")}
            style={{ width: `${share * 100}%` }}
          />
        </div>
      )}
    </div>
  );
}

/** Time, LLM-call and token consumption against the run's budgets. */
export function BudgetPanel({ budget, cacheHits }: { budget: BudgetSnapshot; cacheHits?: number }) {
  return (
    <div className="space-y-3">
      <Meter label="Wall clock" used={Math.round(budget.elapsed_s)} limit={budget.wall_budget_s} unit="s" />
      <Meter label="LLM calls" used={budget.llm_calls} limit={budget.llm_call_budget} />
      <Meter label="Tokens" used={budget.tokens} limit={budget.token_budget} />
      {cacheHits !== undefined && cacheHits > 0 && (
        <p className="text-xs text-muted">{cacheHits} responses served from the LLM cache (no quota used).</p>
      )}
    </div>
  );
}
