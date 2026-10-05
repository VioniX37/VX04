"use client";

import { useState } from "react";
import type { ModelCard } from "@/lib/types";
import { Badge, Card, CardTitle, cn, Stat } from "./ui";

interface ModelCardPanelProps {
  card: ModelCard;
}

export function ModelCardPanel({ card }: ModelCardPanelProps) {
  const [tab, setTab] = useState<"features" | "diagnostics" | "calibration">("features");

  const {
    model_family,
    task_type,
    primary_metric,
    primary_score,
    why_this_model,
    feature_importances,
    confusion_matrix,
    residuals,
    calibration,
  } = card;

  return (
    <Card>
      <CardTitle
        eyebrow={`post-training verification · ${task_type.replaceAll("_", " ")}`}
        aside={
          <div className="flex items-center gap-2">
            {primary_metric && primary_score !== null && (
              <Badge tone="accent">
                {primary_metric}: {primary_score.toFixed(4)}
              </Badge>
            )}
            <div role="tablist" className="flex rounded-lg border border-border bg-surface-muted/50 p-0.5 text-xs">
              {(["features", "diagnostics", ...(calibration ? ["calibration"] : [])] as const).map((v) => (
                <button
                  key={v}
                  role="tab"
                  aria-selected={tab === v}
                  onClick={() => setTab(v as typeof tab)}
                  className={cn(
                    "rounded-md px-2.5 py-1 capitalize",
                    tab === v ? "bg-surface font-medium shadow-sm" : "text-muted hover:text-foreground"
                  )}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>
        }
      >
        Model Card: {model_family}
      </CardTitle>

      <div className="space-y-4">
        {/* Why this model section */}
        {why_this_model && (
          <div className="rounded-lg border border-accent/20 bg-accent/5 p-3.5 text-xs text-foreground">
            <span className="font-semibold text-accent">Why this model was selected: </span>
            <span className="text-muted leading-relaxed">{why_this_model}</span>
          </div>
        )}

        {/* Tab 1: Feature Importances */}
        {tab === "features" && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-muted">
              <span>Top Predictive Features</span>
              <span>Method: {feature_importances[0]?.method ?? "native"}</span>
            </div>
            <div className="space-y-2">
              {feature_importances.slice(0, 10).map((fi) => {
                const pct = Math.max(2, Math.round(fi.importance * 100));
                return (
                  <div key={fi.feature} className="space-y-1">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="truncate text-foreground">{fi.feature}</span>
                      <span className="text-muted">{(fi.importance * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-surface-muted overflow-hidden">
                      <div
                        className="h-full rounded-full bg-accent transition-all duration-300"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tab 2: Diagnostics (Confusion Matrix / Per-Class or Residuals) */}
        {tab === "diagnostics" && (
          <div className="space-y-4 text-xs">
            {confusion_matrix && (
              <div className="space-y-3">
                <p className="font-medium text-muted">Confusion Matrix & Per-Class Performance</p>
                <div className="overflow-x-auto rounded-lg border border-border">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-surface-muted/50 text-muted uppercase font-semibold border-b border-border">
                      <tr>
                        <th className="py-2 px-3">Class</th>
                        <th className="py-2 px-3">Precision</th>
                        <th className="py-2 px-3">Recall</th>
                        <th className="py-2 px-3">F1-Score</th>
                        <th className="py-2 px-3">Support</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {confusion_matrix.labels.map((lbl) => {
                        const stats = confusion_matrix.per_class[lbl] ?? {};
                        return (
                          <tr key={lbl} className="hover:bg-surface-muted/30">
                            <td className="py-2 px-3 font-mono font-medium">{lbl}</td>
                            <td className="py-2 px-3 font-mono">{stats.precision?.toFixed(3) ?? "—"}</td>
                            <td className="py-2 px-3 font-mono">{stats.recall?.toFixed(3) ?? "—"}</td>
                            <td className="py-2 px-3 font-mono">{stats["f1-score"]?.toFixed(3) ?? "—"}</td>
                            <td className="py-2 px-3 font-mono text-muted">{stats.support ?? "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="overflow-x-auto rounded-lg border border-border">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-surface-muted/50 text-muted font-semibold border-b border-border">
                      <tr>
                        <th className="py-2 px-3 uppercase">Actual \ predicted</th>
                        {confusion_matrix.labels.map((lbl) => (
                          <th key={lbl} className="py-2 px-3 font-mono">
                            {lbl}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {confusion_matrix.matrix.map((row, i) => (
                        <tr key={confusion_matrix.labels[i]}>
                          <td className="py-2 px-3 font-mono font-medium">{confusion_matrix.labels[i]}</td>
                          {row.map((count, j) => (
                            <td
                              key={j}
                              className={cn("py-2 px-3 font-mono tabular", i === j ? "text-success font-semibold" : "text-muted")}
                            >
                              {count}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {residuals && (
              <div className="space-y-3">
                <p className="font-medium text-muted">Residual Diagnostics</p>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="MAE" value={residuals.mae.toFixed(4)} />
                  <Stat label="RMSE" value={residuals.rmse.toFixed(4)} />
                  <Stat label="R² Score" value={residuals.r2.toFixed(4)} />
                  <Stat label="Max Error" value={residuals.max_error.toFixed(4)} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Calibration */}
        {tab === "calibration" && calibration && (
          <div className="space-y-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-muted">Probability Calibration (Reliability Diagram)</span>
              <Badge tone="accent">Brier Score: {calibration.brier_score.toFixed(4)}</Badge>
            </div>
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-left text-xs">
                <thead className="bg-surface-muted/50 text-muted uppercase font-semibold border-b border-border">
                  <tr>
                    <th className="py-2 px-3">Bin Mean Predicted Prob</th>
                    <th className="py-2 px-3">Observed True Positive Rate</th>
                    <th className="py-2 px-3">Calibration Difference</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border font-mono">
                  {calibration.points.map((pt, idx) => {
                    const diff = Math.abs(pt.prob_pred - pt.prob_true);
                    return (
                      <tr key={idx} className="hover:bg-surface-muted/30">
                        <td className="py-2 px-3">{(pt.prob_pred * 100).toFixed(1)}%</td>
                        <td className="py-2 px-3">{(pt.prob_true * 100).toFixed(1)}%</td>
                        <td className="py-2 px-3 text-muted">{(diff * 100).toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
