import type { KnowledgeRef } from "@/lib/types";
import { Badge } from "./ui";

function sourceBadge(source: string) {
  if (source.startsWith("memory:")) return <Badge tone="success">memory</Badge>;
  if (source === "google-search") return <Badge tone="accent">web</Badge>;
  return <Badge>{source}</Badge>;
}

/**
 * Knowledge used for planning, with experience-memory recalls (past runs on similar
 * datasets) listed first and web sources linked.
 */
export function KnowledgePanel({ items }: { items: KnowledgeRef[] }) {
  const memory = items.filter((k) => k.source.startsWith("memory:"));
  const other = items.filter((k) => !k.source.startsWith("memory:"));
  return (
    <div className="space-y-3 text-sm">
      {memory.length > 0 && (
        <p className="rounded-lg bg-success-soft px-3 py-2 text-xs text-success">
          Recalled {memory.length} past run{memory.length > 1 ? "s" : ""} on similar datasets.
        </p>
      )}
      <ul className="space-y-2">
        {[...memory, ...other].map((k) => (
          <li key={k.id}>
            <div className="flex items-start justify-between gap-2">
              <span>{k.title}</span>
              {sourceBadge(k.source)}
            </div>
            {k.urls?.length > 0 && (
              <ul className="mt-1 space-y-0.5">
                {k.urls.slice(0, 4).map((u) => (
                  <li key={u} className="truncate text-xs">
                    <a href={u} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                      {u.replace(/^https?:\/\//, "")}
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
