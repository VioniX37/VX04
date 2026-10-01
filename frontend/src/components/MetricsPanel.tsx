import { formatMetric } from "@/lib/stages";
import type { ExecutionMetrics, TaskSpec } from "@/lib/types";
import { Badge } from "./ui";

/** Final metrics of a run, the requirement verdict and training details. */
export function MetricsPanel({ metrics, spec }: { metrics: ExecutionMetrics; spec: TaskSpec | null }) {
  const all = metrics.metrics ?? {};
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
        <div>
          <p className="text-xs text-muted">{metrics.metric ?? spec?.metric ?? "score"}</p>
          <p className="text-3xl font-semibold tabular-nums">{formatMetric(metrics.score)}</p>
        </div>
        {spec?.metric_target != null && (
          <div>
            <p className="text-xs text-muted">Target</p>
            <p className="text-lg tabular-nums">{formatMetric(spec.metric_target)}</p>
          </div>
        )}
        {metrics.target_met !== undefined && (
          <Badge tone={metrics.target_met ? "success" : "warning"}>
            {metrics.target_met ? "requirements met" : "target not met"}
          </Badge>
        )}
      </div>
      <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        {Object.entries(all).map(([k, v]) => (
          <div key={k} className="rounded-lg bg-surface-muted px-3 py-2">
            <dt className="text-xs text-muted">{k}</dt>
            <dd className="font-medium tabular-nums">{formatMetric(v)}</dd>
          </div>
        ))}
        {metrics.model_family && (
          <div className="rounded-lg bg-surface-muted px-3 py-2">
            <dt className="text-xs text-muted">model</dt>
            <dd className="truncate font-mono text-xs font-medium">{metrics.model_family}</dd>
          </div>
        )}
        {metrics.train_time_s !== undefined && (
          <div className="rounded-lg bg-surface-muted px-3 py-2">
            <dt className="text-xs text-muted">train time</dt>
            <dd className="font-medium tabular-nums">{metrics.train_time_s.toFixed(2)}s</dd>
          </div>
        )}
      </dl>
    </div>
  );
}
