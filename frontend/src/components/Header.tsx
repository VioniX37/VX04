"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { Health, Run } from "@/lib/types";
import { ThemeToggle } from "./ThemeToggle";
import { cn, LiveDot } from "./ui";

const NAV = [
  { href: "/", label: "New run" },
  { href: "/runs", label: "Runs" },
];

/** Brand mark: a small node graph, echoing the pipeline. */
function Mark() {
  return (
    <svg viewBox="0 0 28 28" className="h-7 w-7" aria-hidden>
      <defs>
        <linearGradient id="mark-g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="var(--accent)" />
          <stop offset="1" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="26" height="26" rx="8" fill="var(--surface-muted)" stroke="var(--border-strong)" />
      <path d="M8 18 L14 9 L20 18" fill="none" stroke="url(#mark-g)" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="8" cy="18" r="2.4" fill="var(--accent)" />
      <circle cx="14" cy="9" r="2.4" fill="url(#mark-g)" />
      <circle cx="20" cy="18" r="2.4" fill="var(--accent-2)" />
    </svg>
  );
}

/** Top navigation with live run activity, backend status and the Gemini models in use. */
export function Header() {
  const pathname = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [active, setActive] = useState<Run[]>([]);

  useEffect(() => {
    api.health().then(setHealth, () => setOffline(true));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = () =>
      api.listRuns().then(
        (runs) => !cancelled && setActive(runs.filter((r) => r.status === "running" || r.status === "pending")),
        () => undefined,
      );
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const models = health?.models ?? {};
  return (
    <header className="sticky top-0 z-30 border-b border-border/70 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex h-15 max-w-7xl items-center justify-between gap-4 px-4 py-2.5">
        <div className="flex items-center gap-7">
          <Link href="/" className="flex items-center gap-2.5">
            <Mark />
            <span className="leading-tight">
              <span className="block text-sm font-semibold tracking-tight">GroundML</span>
            </span>
          </Link>
          <nav className="flex gap-1 rounded-xl border border-border bg-surface-muted/40 p-1">
            {NAV.map((item) => {
              const on = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "rounded-lg px-3 py-1 text-sm transition-colors",
                    on ? "bg-surface font-medium text-foreground shadow-sm" : "text-muted hover:text-foreground",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
        <div className="flex items-center gap-3">
          {active.length > 0 && (
            <Link
              href={`/runs/${active[0].id}`}
              className="hidden items-center gap-2 rounded-full border border-accent/30 bg-accent-soft px-3 py-1 text-xs text-accent sm:inline-flex"
            >
              <LiveDot className="h-1.5 w-1.5" />
              {active.length} run{active.length > 1 ? "s" : ""} in progress
            </Link>
          )}
          <div
            className="flex items-center gap-2 rounded-full border border-border bg-surface-muted/40 px-3 py-1 text-xs text-muted"
            title={Object.entries(models).map(([r, m]) => `${r}: ${m}`).join("\n") || undefined}
          >
            <span className={cn("h-1.5 w-1.5 rounded-full", offline ? "bg-danger" : health ? "bg-success" : "bg-faint")} />
            {offline
              ? "backend offline"
              : health
                ? health.llm_provider === "fake"
                  ? "offline fake LLM"
                  : `${models.smart ?? health.llm_model}${models.fast && models.fast !== models.smart ? ` · ${models.fast}` : ""}`
                : "connecting…"}
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
