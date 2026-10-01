"use client";

import { useEffect, useRef, useState } from "react";

import { AGENT_LABELS } from "@/lib/stages";
import type { AgentEvent, EventKind } from "@/lib/types";
import { cn } from "./ui";

const DOT: Record<EventKind, string> = {
  status: "bg-accent",
  info: "bg-faint",
  llm: "bg-border-strong",
  artifact: "bg-success",
  warning: "bg-warning",
  error: "bg-danger",
  telemetry: "bg-faint",
};

/** Chronological agent activity; raw LLM outputs can be expanded for inspection. */
export function AgentTimeline({ events }: { events: AgentEvent[] }) {
  const [showLlm, setShowLlm] = useState(false);
  const list = useRef<HTMLOListElement>(null);
  const visible = events.filter((e) => e.kind !== "telemetry" && (showLlm || e.kind !== "llm"));

  // Keep the newest entry in view by scrolling the list itself (never the page),
  // and only when the reader is already near the bottom.
  useEffect(() => {
    const el = list.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 120) el.scrollTop = el.scrollHeight;
  }, [visible.length]);

  return (
    <div>
      <label className="mb-3 flex items-center gap-2 text-xs text-muted">
        <input type="checkbox" checked={showLlm} onChange={(e) => setShowLlm(e.target.checked)} className="accent-accent" />
        Show raw LLM outputs
      </label>
      <ol ref={list} className="scroll-thin relative max-h-[30rem] space-y-3 overflow-auto pr-1">
        <span aria-hidden className="absolute left-[3.5px] top-1 bottom-1 w-px bg-border" />
        {visible.map((e) => (
          <li key={e.seq} className="relative flex gap-3">
            <span className={cn("relative mt-1.5 h-2 w-2 shrink-0 rounded-full ring-4 ring-surface", DOT[e.kind])} />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2 text-[11px] text-faint">
                <span className="font-medium text-muted">{AGENT_LABELS[e.agent] ?? e.agent}</span>
                <span className="font-mono">{new Date(e.ts).toLocaleTimeString()}</span>
              </div>
              <p className={cn("text-[13px] leading-snug break-words", e.kind === "error" && "text-danger")}>{e.message}</p>
              {showLlm && e.kind === "llm" && e.payload && (
                <pre className="scroll-thin mt-1 max-h-48 overflow-auto rounded-lg border border-border bg-code-bg p-2 text-[11px] text-code-fg">
                  {JSON.stringify(e.payload.output, null, 2)}
                </pre>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
