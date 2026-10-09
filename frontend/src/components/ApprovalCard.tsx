"use client";

import { useState } from "react";

import { api } from "@/lib/api";
import { familyLabel, formatMetric } from "@/lib/stages";
import type { AgentEvent, PlanEvaluation, Run } from "@/lib/types";
import { Badge, Button, Card, CardTitle, ErrorNote, LiveDot, Spinner, cn } from "./ui";

interface ApprovalCardProps {
  run: Run;
  events: AgentEvent[];
  onUpdated: (run: Run) => void;
}

export function ApprovalCard({ run, events, onUpdated }: ApprovalCardProps) {
  // Find the latest pause event
  const pauseEvent = [...events]
    .reverse()
    .find((e) => e.kind === "status" && e.payload && (e.payload as Record<string, unknown>).approval_step);

  const payload = (pauseEvent?.payload as Record<string, unknown>) || {};
  const approvalStep = (payload.approval_step as string) || "plans";

  // Plan approval state
  const rankedCandidates = (payload.ranked as PlanEvaluation[]) || [];
  const defaultSelectedId = (payload.selected as string) || rankedCandidates[0]?.plan.id || "";
  const [selectedPlanId, setSelectedPlanId] = useState<string>(defaultSelectedId);
  const [isEditingPlan, setIsEditingPlan] = useState(false);
  const [editedTitle, setEditedTitle] = useState("");
  const [editedModelFamily, setEditedModelFamily] = useState("");
  const [editedHpJson, setEditedHpJson] = useState("");

  // Code review state
  const generatedCode = (payload.code as string) || run.code || "";
  const templateCode = (payload.template_code as string) || "";
  const [activeCodeTab, setActiveCodeTab] = useState<"code" | "diff" | "edit">("code");
  const [editedCode, setEditedCode] = useState(generatedCode);

  const [submitting, setSubmitting] = useState(false);
  const [approved, setApproved] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize edit fields when user expands editor
  function handleStartEdit(ev: PlanEvaluation) {
    setSelectedPlanId(ev.plan.id);
    setEditedTitle(ev.plan.title);
    setEditedModelFamily(ev.model.model_family);
    setEditedHpJson(JSON.stringify(ev.model.hyperparameters || ev.plan.hyperparameters || {}, null, 2));
    setIsEditingPlan(true);
  }

  async function handleApprovePlan() {
    setError(null);
    setSubmitting(true);
    try {
      let requestBody;
      const isTop = rankedCandidates.length > 0 && selectedPlanId === rankedCandidates[0].plan.id;

      if (isEditingPlan) {
        let parsedHp = {};
        try {
          if (editedHpJson.trim()) parsedHp = JSON.parse(editedHpJson);
        } catch {
          throw new Error("Invalid JSON in hyperparameters field.");
        }
        requestBody = {
          action: "edit" as const,
          plan_id: selectedPlanId,
          edited_plan: {
            title: editedTitle,
            model_family: editedModelFamily,
            hyperparameters: parsedHp,
          },
        };
      } else if (!isTop) {
        requestBody = {
          action: "pick" as const,
          plan_id: selectedPlanId,
        };
      } else {
        requestBody = {
          action: "approve" as const,
        };
      }

      setApproved(true);
      const updated = await api.approveRun(run.id, requestBody);
      onUpdated(updated);
    } catch (e) {
      setApproved(false);
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleApproveCode() {
    setError(null);
    setSubmitting(true);
    try {
      const isCodeEdited = editedCode.trim() !== generatedCode.trim();
      const requestBody = isCodeEdited
        ? { action: "edit" as const, edited_code: editedCode }
        : { action: "approve" as const };

      setApproved(true);
      const updated = await api.approveRun(run.id, requestBody);
      onUpdated(updated);
    } catch (e) {
      setApproved(false);
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCancelRun() {
    if (!confirm("Are you sure you want to cancel this run?")) return;
    setError(null);
    setCancelling(true);
    try {
      const updated = await api.cancelRun(run.id);
      onUpdated(updated);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCancelling(false);
    }
  }

  return (
    <Card className="border-warning/50 bg-gradient-to-br from-warning/10 via-surface to-surface shadow-lg shadow-warning/5">
      <CardTitle
        eyebrow="human-in-the-loop review"
        aside={
          <Badge tone="warning" className="animate-pulse">
            <LiveDot className="h-1.5 w-1.5 text-warning" /> Awaiting Your Decision
          </Badge>
        }
      >
        {approvalStep === "code" ? "Review & Approve Training Code" : "Review & Select Model Plan"}
      </CardTitle>


      {error && <div className="mt-3"><ErrorNote>{error}</ErrorNote></div>}

      {/* PLAN APPROVAL STEP */}
      {approvalStep === "plans" && (
        <div className="mt-5 space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {rankedCandidates.map((ev, idx) => {
              const isSelected = (selectedPlanId || rankedCandidates[0]?.plan.id) === ev.plan.id;
              const isTop = idx === 0;
              const lastObs = ev.observations && ev.observations.length > 0 ? ev.observations[ev.observations.length - 1] : null;
              const score = lastObs?.score;

              return (
                <div
                  key={ev.plan.id}
                  onClick={() => {
                    setSelectedPlanId(ev.plan.id);
                    if (isEditingPlan) handleStartEdit(ev);
                  }}
                  className={cn(
                    "relative flex cursor-pointer flex-col justify-between rounded-xl border p-4 transition-all",
                    isSelected
                      ? "border-accent bg-accent-soft/30 shadow-md ring-2 ring-accent/30"
                      : "border-border bg-surface-muted/40 hover:border-border-strong hover:bg-surface-muted/70",
                  )}
                >
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-semibold text-accent">#{idx + 1}</span>
                        <Badge tone={isTop ? "accent" : "neutral"}>
                          {familyLabel(ev.model.model_family)}
                        </Badge>
                      </div>
                      {isTop && <Badge tone="success">Top Ranked</Badge>}
                    </div>

                    <h4 className="mt-2 text-sm font-semibold text-foreground">{ev.plan.title}</h4>
                    <p className="mt-1 line-clamp-2 text-xs text-muted">{ev.plan.rationale}</p>
                  </div>

                  <div className="mt-4 border-t border-border/50 pt-3">
                    <div className="flex items-baseline justify-between text-xs">
                      <span className="text-muted">Observed score:</span>
                      <span className="font-mono font-semibold text-foreground">
                        {score != null ? formatMetric(score) : "—"}
                      </span>
                    </div>
                    <div className="mt-1 flex items-baseline justify-between text-xs">
                      <span className="text-muted">Predicted score:</span>
                      <span className="font-mono text-faint">
                        {ev.model.predicted_score != null ? formatMetric(ev.model.predicted_score) : "—"}
                      </span>
                    </div>

                    <div className="mt-3 flex items-center justify-between">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStartEdit(ev);
                        }}
                        className="text-xs text-accent hover:underline"
                      >
                        Customize plan
                      </button>
                      <input
                        type="radio"
                        checked={isSelected}
                        onChange={() => setSelectedPlanId(ev.plan.id)}
                        className="accent-accent"
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Plan Customizer / Editor */}
          {isEditingPlan && (
            <div className="rounded-xl border border-accent/30 bg-surface-muted/60 p-4">
              <div className="flex items-center justify-between">
                <h5 className="text-xs font-semibold uppercase tracking-wider text-accent">
                  Customize Plan {selectedPlanId}
                </h5>
                <button
                  type="button"
                  onClick={() => setIsEditingPlan(false)}
                  className="text-xs text-muted hover:text-foreground"
                >
                  Cancel customization
                </button>
              </div>

              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <div>
                  <label className="eyebrow block text-xs">Plan Title</label>
                  <input
                    type="text"
                    value={editedTitle}
                    onChange={(e) => setEditedTitle(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-border bg-surface px-3 py-1.5 text-xs text-foreground focus:border-accent focus:outline-none"
                  />
                </div>
                <div>
                  <label className="eyebrow block text-xs">Model Family</label>
                  <select
                    value={editedModelFamily}
                    onChange={(e) => setEditedModelFamily(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-border bg-surface px-3 py-1.5 text-xs text-foreground focus:border-accent focus:outline-none"
                  >
                    {["lightgbm", "xgboost", "random_forest", "linear", "mlp"].map((fam) => (
                      <option key={fam} value={fam}>{familyLabel(fam)}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="mt-3">
                <label className="eyebrow block text-xs">Hyperparameters (JSON)</label>
                <textarea
                  rows={4}
                  value={editedHpJson}
                  onChange={(e) => setEditedHpJson(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-border bg-surface p-2.5 font-mono text-xs text-foreground focus:border-accent focus:outline-none"
                />
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleCancelRun}
              disabled={cancelling || submitting || approved}
              className="text-danger hover:bg-danger-soft/20"
            >
              {cancelling ? <Spinner /> : null} Cancel Run
            </Button>

            <div className="flex items-center gap-2">
              <Button
                variant="primary"
                onClick={handleApprovePlan}
                disabled={submitting || cancelling || approved}
                className={cn(
                  "text-white",
                  approved ? "bg-success/80" : "bg-success hover:bg-success/90",
                )}
              >
                {approved ? (
                  <>
                    <span className="text-white">✓</span> Approved! Resuming pipeline…
                  </>
                ) : submitting ? (
                  <>
                    <Spinner /> Resuming…
                  </>
                ) : (
                  "✓ Approve & Implement Plan"
                )}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* CODE REVIEW STEP */}
      {approvalStep === "code" && (
        <div className="mt-5 space-y-4">
          {/* Tabs */}
          <div className="flex items-center gap-2 border-b border-border pb-2">
            <button
              type="button"
              onClick={() => setActiveCodeTab("code")}
              className={cn(
                "rounded-lg px-3 py-1 text-xs font-medium transition-colors",
                activeCodeTab === "code" ? "bg-accent text-white" : "text-muted hover:text-foreground",
              )}
            >
              Generated Script
            </button>
            {templateCode && (
              <button
                type="button"
                onClick={() => setActiveCodeTab("diff")}
                className={cn(
                  "rounded-lg px-3 py-1 text-xs font-medium transition-colors",
                  activeCodeTab === "diff" ? "bg-accent text-white" : "text-muted hover:text-foreground",
                )}
              >
                Base Template Comparison
              </button>
            )}
            <button
              type="button"
              onClick={() => setActiveCodeTab("edit")}
              className={cn(
                "rounded-lg px-3 py-1 text-xs font-medium transition-colors",
                activeCodeTab === "edit" ? "bg-accent text-white" : "text-muted hover:text-foreground",
              )}
            >
              Edit Script
            </button>
          </div>

          {activeCodeTab === "code" && (
            <pre className="scroll-thin max-h-96 overflow-auto rounded-xl border border-border bg-surface-muted p-4 font-mono text-xs text-foreground">
              {generatedCode}
            </pre>
          )}

          {activeCodeTab === "diff" && templateCode && (
            <div className="grid gap-3 lg:grid-cols-2">
              <div>
                <span className="eyebrow block pb-1 text-xs text-faint">Base Scaffold Template</span>
                <pre className="scroll-thin max-h-96 overflow-auto rounded-xl border border-border bg-surface-muted/60 p-3 font-mono text-xs text-muted">
                  {templateCode}
                </pre>
              </div>
              <div>
                <span className="eyebrow block pb-1 text-xs text-accent">Operation Agent Generated Script</span>
                <pre className="scroll-thin max-h-96 overflow-auto rounded-xl border border-accent/30 bg-accent-soft/10 p-3 font-mono text-xs text-foreground">
                  {generatedCode}
                </pre>
              </div>
            </div>
          )}

          {activeCodeTab === "edit" && (
            <div>
              <span className="eyebrow block pb-1 text-xs">Direct Python Script Editor</span>
              <textarea
                rows={14}
                value={editedCode}
                onChange={(e) => setEditedCode(e.target.value)}
                className="scroll-thin w-full rounded-xl border border-accent/40 bg-surface p-3 font-mono text-xs text-foreground focus:border-accent focus:outline-none"
              />
            </div>
          )}

          {/* Action buttons */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleCancelRun}
              disabled={cancelling || submitting || approved}
              className="text-danger hover:bg-danger-soft/20"
            >
              {cancelling ? <Spinner /> : null} Cancel Run
            </Button>

            <Button
              variant="primary"
              onClick={handleApproveCode}
              disabled={submitting || cancelling || approved}
              className={cn(
                "text-white",
                approved ? "bg-success/80" : "bg-success hover:bg-success/90",
              )}
            >
              {approved ? (
                <>
                  <span className="text-white">✓</span> Code approved! Running sandbox…
                </>
              ) : submitting ? (
                <>
                  <Spinner /> Resuming…
                </>
              ) : (
                "✓ Approve & Run Training Script"
              )}
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}
