"use client";

import { useEffect, useRef, useState } from "react";

import { AGENT_LABELS } from "@/lib/stages";
import type { AgentEvent, EventKind } from "@/lib/types";
import { cn } from "./ui";

const DOT: Record<EventKind, string> = {
  status: "bg-accent",
  info: "bg-muted",
  llm: "bg-border",
  artifact: "bg-success",
  warning: "bg-warning",
  error: "bg-danger",
};

/** Chronological agent activity; raw LLM outputs can be expanded for inspection. */
export function AgentTimeline({ events }: { events: AgentEvent[] }) {
  const [showLlm, setShowLlm] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const visible = showLlm ? events : events.filter((e) => e.kind !== "llm");

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "nearest" });
  }, [visible.length]);

  return (
    <div>
      <label className="mb-3 flex items-center gap-2 text-xs text-muted">
        <input type="checkbox" checked={showLlm} onChange={(e) => setShowLlm(e.target.checked)} />
        Show raw LLM outputs
      </label>
      <ol className="max-h-[32rem] space-y-3 overflow-auto pr-1">
        {visible.map((e) => (
          <li key={e.seq} className="flex gap-3">
            <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", DOT[e.kind])} />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2 text-xs text-muted">
                <span className="font-medium text-foreground">{AGENT_LABELS[e.agent] ?? e.agent}</span>
                <span>{new Date(e.ts).toLocaleTimeString()}</span>
              </div>
              <p className={cn("text-sm break-words", e.kind === "error" && "text-danger")}>{e.message}</p>
              {showLlm && e.kind === "llm" && e.payload && (
                <pre className="mt-1 max-h-48 overflow-auto rounded-md bg-surface-muted p-2 text-xs">
                  {JSON.stringify(e.payload.output, null, 2)}
                </pre>
              )}
            </div>
          </li>
        ))}
        <div ref={bottom} />
      </ol>
    </div>
  );
}
