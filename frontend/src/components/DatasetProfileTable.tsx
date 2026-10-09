import { formatBytes } from "@/lib/stages";
import type { ColumnKind, DatasetProfile, ScaleTier } from "@/lib/types";
import { Badge } from "./ui";

const TIER_TONE: Record<ScaleTier, "neutral" | "accent" | "warning"> = {
  small: "neutral",
  medium: "accent",
  large: "warning",
};

const KIND_TONE: Record<ColumnKind, "neutral" | "accent" | "success" | "warning"> = {
  numeric: "accent",
  categorical: "success",
  text: "warning",
  datetime: "neutral",
  identifier: "neutral",
  boolean: "success",
};

/** Column-level profile of a dataset, with its size and scale tier. */
export function DatasetProfileTable({ profile }: { profile: DatasetProfile }) {
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted">
        <Badge tone={TIER_TONE[profile.scale_tier]}>{profile.scale_tier} data</Badge>
        <span>
          {profile.n_rows.toLocaleString()} rows · {profile.n_cols} columns
        </span>
        {profile.size_bytes > 0 && <span>· {formatBytes(profile.size_bytes)} on disk</span>}
        {profile.memory_estimate_mb > 0 && <span>· ~{formatBytes(profile.memory_estimate_mb * 1024 ** 2)} in memory</span>}
        {profile.guessed_target && (
          <span>
            · likely target <span className="font-medium text-foreground">{profile.guessed_target}</span>
          </span>
        )}
      </div>
      <div className="max-h-80 overflow-auto rounded-lg border border-border">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-surface-muted text-xs text-muted">
            <tr>
              <th className="px-3 py-2 font-medium">Column</th>
              <th className="px-3 py-2 font-medium">Kind</th>
              <th className="px-3 py-2 text-right font-medium">Unique</th>
              <th className="px-3 py-2 text-right font-medium">Missing</th>
              <th className="px-3 py-2 font-medium">Sample</th>
            </tr>
          </thead>
          <tbody>
            {profile.columns.map((c) => (
              <tr key={c.name} className="border-t border-border">
                <td className="px-3 py-2 font-mono text-xs">{c.name}</td>
                <td className="px-3 py-2">
                  <Badge tone={KIND_TONE[c.kind]}>{c.kind}</Badge>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{c.n_unique.toLocaleString()}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${c.n_missing ? "text-warning" : "text-muted"}`}>
                  {c.n_missing.toLocaleString()}
                </td>
                <td className="max-w-64 truncate px-3 py-2 text-xs text-muted">{c.sample_values.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
