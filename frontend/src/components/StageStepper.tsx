import { STAGES } from "@/lib/stages";
import type { Stage } from "@/lib/types";
import { Spinner, cn } from "./ui";

export function StageStepper({
  reached,
  current,
  finished,
  failed,
}: {
  reached: Set<Stage>;
  current: Stage | null;
  finished: boolean;
  failed: boolean;
}) {
  return (
    <ol className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
      {STAGES.map((s, i) => {
        const active = !finished && current === s.id;
        const done = reached.has(s.id) && !active;
        const failedHere = failed && current === s.id;
        return (
          <li
            key={s.id}
            className={cn(
              "rounded-lg border px-3 py-2",
              failedHere
                ? "border-danger bg-danger-soft"
                : active
                  ? "border-accent bg-accent-soft"
                  : done
                    ? "border-border bg-surface"
                    : "border-dashed border-border opacity-60",
            )}
          >
            <div className="flex items-center gap-1.5 text-xs text-muted">
              {active ? <Spinner className="h-2.5 w-2.5 text-accent" /> : <span className="tabular-nums">{i + 1}</span>}
              <span className="truncate">{s.agent}</span>
            </div>
            <div className="mt-0.5 text-sm font-medium">{s.label}</div>
          </li>
        );
      })}
    </ol>
  );
}
