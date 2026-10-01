import type { ButtonHTMLAttributes, ReactNode } from "react";

/** Join class names, skipping falsy values. */
export function cn(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}

/** Glass panel used for every section; `glow` adds the gradient edge used for live/primary panels. */
export function Card({
  children,
  className,
  glow = false,
}: {
  children: ReactNode;
  className?: string;
  glow?: boolean;
}) {
  return <section className={cn("panel fade-up p-5", glow && "panel-glow", className)}>{children}</section>;
}

/** Panel heading: small mono eyebrow, a title, and an optional right-aligned element. */
export function CardTitle({ children, eyebrow, aside }: { children: ReactNode; eyebrow?: string; aside?: ReactNode }) {
  return (
    <div className="mb-4 flex items-start justify-between gap-3">
      <div>
        {eyebrow && <p className="eyebrow mb-1">{eyebrow}</p>}
        <h2 className="text-[15px] font-semibold tracking-tight">{children}</h2>
      </div>
      {aside}
    </div>
  );
}

type Tone = "neutral" | "accent" | "success" | "warning" | "danger";

const TONES: Record<Tone, string> = {
  neutral: "border-border bg-surface-muted text-muted",
  accent: "border-accent/25 bg-accent-soft text-accent",
  success: "border-success/25 bg-success-soft text-success",
  warning: "border-warning/25 bg-warning-soft text-warning",
  danger: "border-danger/25 bg-danger-soft text-danger",
};

/** Small pill label; `tone` selects the semantic colour. */
export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium",
        TONES[tone],
      )}
    >
      {children}
    </span>
  );
}

/** Primary or ghost button. */
export function Button({
  children,
  variant = "primary",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition-all",
        "disabled:cursor-not-allowed disabled:opacity-45",
        variant === "primary"
          ? "bg-linear-to-r from-accent to-accent-2 text-white shadow-[0_8px_24px_-10px_var(--accent)] hover:brightness-110"
          : "border border-border bg-surface-muted/60 text-foreground hover:border-border-strong hover:bg-surface-muted",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

/** Inline loading indicator that inherits the text colour. */
export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cn("inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent", className)}
    />
  );
}

/** Pulsing dot for "live" states. */
export function LiveDot({ className }: { className?: string }) {
  return <span aria-hidden className={cn("live-dot", className)} />;
}

/** Inline error message. */
export function ErrorNote({ children }: { children: ReactNode }) {
  return <p className="rounded-xl border border-danger/25 bg-danger-soft px-3 py-2 text-sm text-danger">{children}</p>;
}

/** Badge for a run status (pending, running, succeeded, failed). */
export function StatusBadge({ status }: { status: string }) {
  const tone: Tone =
    status === "succeeded" ? "success" : status === "failed" ? "danger" : status === "running" ? "accent" : "neutral";
  return (
    <Badge tone={tone}>
      {status === "running" && <LiveDot className="h-1.5 w-1.5" />}
      {status}
    </Badge>
  );
}

/** Compact metric tile: mono label, large value, optional caption. */
export function Stat({
  label,
  value,
  caption,
  tone,
}: {
  label: string;
  value: ReactNode;
  caption?: ReactNode;
  tone?: "accent" | "success" | "warning";
}) {
  return (
    <div className="min-w-0 rounded-xl border border-border bg-surface-muted/40 px-3.5 py-3">
      <p className="eyebrow truncate">{label}</p>
      <p
        className={cn(
          "tabular mt-1 truncate text-xl font-semibold tracking-tight",
          tone === "accent" && "text-accent",
          tone === "success" && "text-success",
          tone === "warning" && "text-warning",
        )}
      >
        {value}
      </p>
      {caption && <p className="mt-0.5 truncate text-[11px] text-muted">{caption}</p>}
    </div>
  );
}
