import { formatMetric, formatRows, higherIsBetter } from "@/lib/stages";
import type { RungResult } from "@/lib/types";
import { Badge, cn } from "./ui";

/**
 * Successive-halving results: one row per plan, one column per rung (training rows),
 * next to the score the agents predicted before any code ran.
 */
export function GroundingPanel({
  rungs,
  predicted,
  selected,
  metric,
  stoppedForBudget,
}: {
  rungs: RungResult[];
  predicted: Record<string, number>;
  selected: string | null;
  metric: string | undefined;
  stoppedForBudget: boolean;
}) {
  const higher = higherIsBetter(metric);
  const planIds = Object.keys(predicted);
  const families: Record<string, string> = {};
  rungs.forEach((r) => r.results.forEach((x) => (families[x.plan_id] = x.model_family)));
  const best = (values: number[]) => (higher ? Math.max(...values) : Math.min(...values));
  const predictedBest = planIds.length ? best(planIds.map((p) => predicted[p])) : null;

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-2 pr-3 font-medium">Plan</th>
              <th className="px-3 py-2 text-right font-medium">Predicted</th>
              {rungs.map((r, i) => (
                <th key={i} className="px-3 py-2 text-right font-medium">
                  {formatRows(r.rows)} rows
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {planIds.map((pid) => (
              <tr key={pid} className={cn("border-t border-border", pid === selected && "bg-accent-soft/50")}>
                <td className="py-2 pr-3">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs">{pid}</span>
                    {pid === selected && <Badge tone="accent">selected</Badge>}
                  </div>
                  <div className="text-xs text-muted">{families[pid]}</div>
                </td>
                <td className={cn("px-3 py-2 text-right tabular-nums text-muted", predicted[pid] === predictedBest && "font-semibold")}>
                  {formatMetric(predicted[pid])}
                </td>
                {rungs.map((r, i) => {
                  const cell = r.results.find((x) => x.plan_id === pid);
                  const scores = r.results.filter((x) => x.ok && x.score !== null).map((x) => x.score as number);
                  const isBest = cell?.ok && cell.score !== null && scores.length > 0 && cell.score === best(scores);
                  return (
                    <td key={i} className="px-3 py-2 text-right tabular-nums">
                      {!cell ? (
                        <span className="text-xs text-muted">eliminated</span>
                      ) : !cell.ok ? (
                        <span className="text-xs text-danger" title={cell.error ?? undefined}>
                          failed
                        </span>
                      ) : (
                        <span className={cn(isBest && "font-semibold text-success")}>
                          {formatMetric(cell.score)}
                          <span className="ml-1.5 text-xs font-normal text-muted">{cell.duration_s.toFixed(0)}s</span>
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {stoppedForBudget && <p className="text-xs text-muted">Stopped early: time budget reached.</p>}
    </div>
  );
}
