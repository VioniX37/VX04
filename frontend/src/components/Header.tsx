"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { Health } from "@/lib/types";
import { cn } from "./ui";

const NAV = [
  { href: "/", label: "New run" },
  { href: "/runs", label: "Runs" },
];

export function Header() {
  const pathname = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    api.health().then(setHealth, () => setOffline(true));
  }, []);

  return (
    <header className="border-b border-border bg-surface">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4">
        <div className="flex items-center gap-6">
          <Link href="/" className="text-sm font-semibold tracking-tight">
            AutoML-Agent
          </Link>
          <nav className="flex gap-1">
            {NAV.map((item) => {
              const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-sm",
                    active ? "bg-surface-muted font-medium" : "text-muted hover:text-foreground",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted">
          <span className={cn("h-2 w-2 rounded-full", offline ? "bg-danger" : health ? "bg-success" : "bg-border")} />
          {offline ? "backend offline" : health ? `${health.llm_provider} · ${health.llm_model}` : "connecting…"}
        </div>
      </div>
    </header>
  );
}
