import type { ColumnKind, DatasetProfile } from "@/lib/types";
import { Badge } from "./ui";

const KIND_TONE: Record<ColumnKind, "neutral" | "accent" | "success" | "warning"> = {
  numeric: "accent",
  categorical: "success",
  text: "warning",
  datetime: "neutral",
  identifier: "neutral",
  boolean: "success",
};

export function DatasetProfileTable({ profile }: { profile: DatasetProfile }) {
  return (
    <div>
      <p className="mb-3 text-sm text-muted">
        {profile.n_rows.toLocaleString()} rows · {profile.n_cols} columns
        {profile.guessed_target && (
          <>
            {" "}
            · likely target <span className="font-medium text-foreground">{profile.guessed_target}</span>
          </>
        )}
      </p>
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
