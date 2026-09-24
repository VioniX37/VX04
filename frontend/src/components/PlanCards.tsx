import { formatMetric } from "@/lib/stages";
import type { PlanEvaluation } from "@/lib/types";
import { Badge, cn } from "./ui";

export function PlanCards({
  ranked,
  selectedId,
  metric,
}: {
  ranked: PlanEvaluation[];
  selectedId: string | null;
  metric: string | undefined;
}) {
  return (
    <div className="grid gap-3 md:grid-cols-3">
      {ranked.map((ev) => {
        const selected = ev.plan.id === selectedId;
        return (
          <article
            key={ev.plan.id}
            className={cn("rounded-lg border p-4", selected ? "border-accent bg-accent-soft/40" : "border-border")}
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-xs text-muted">
                #{ev.rank} · {ev.plan.id}
              </span>
              {selected && <Badge tone="accent">selected</Badge>}
            </div>
            <h3 className="text-sm font-semibold">{ev.plan.title}</h3>
            <p className="mt-1 font-mono text-xs text-muted">{ev.model.model_family}</p>
            <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div>
                <dt className="text-muted">Predicted {metric ?? "score"}</dt>
                <dd className="font-medium tabular-nums">{formatMetric(ev.model.predicted_score)}</dd>
              </div>
              <div>
                <dt className="text-muted">Est. train time</dt>
                <dd className="font-medium tabular-nums">{Math.round(ev.model.predicted_train_time_s)}s</dd>
              </div>
            </dl>
            <p className="mt-3 text-xs text-muted">{ev.plan.rationale}</p>
            <details className="mt-3 text-xs">
              <summary className="cursor-pointer text-muted hover:text-foreground">Data steps & risks</summary>
              <ul className="mt-2 list-disc space-y-0.5 pl-4">
                {ev.data.steps.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
              {ev.data.risks.length > 0 && (
                <ul className="mt-2 list-disc space-y-0.5 pl-4 text-warning">
                  {ev.data.risks.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              )}
            </details>
          </article>
        );
      })}
    </div>
  );
}
