import type { ButtonHTMLAttributes, ReactNode } from "react";

/** Join class names, skipping falsy values. */
export function cn(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}

/** Bordered surface used for every panel. */
export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={cn("rounded-xl border border-border bg-surface p-5", className)}>{children}</section>;
}

/** Panel heading with an optional right-aligned element. */
export function CardTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-4 flex items-center justify-between gap-3">
      <h2 className="text-sm font-semibold tracking-tight">{children}</h2>
      {aside}
    </div>
  );
}

type Tone = "neutral" | "accent" | "success" | "warning" | "danger";

const TONES: Record<Tone, string> = {
  neutral: "bg-surface-muted text-muted",
  accent: "bg-accent-soft text-accent",
  success: "bg-success-soft text-success",
  warning: "bg-warning-soft text-warning",
  danger: "bg-danger-soft text-danger",
};

/** Small pill label; `tone` selects the semantic colour. */
export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return (
    <span className={cn("inline-flex shrink-0 items-center whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium", TONES[tone])}>
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
        "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary"
          ? "bg-accent text-white hover:opacity-90"
          : "border border-border bg-surface text-foreground hover:bg-surface-muted",
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

/** Inline error message. */
export function ErrorNote({ children }: { children: ReactNode }) {
  return <p className="rounded-lg bg-danger-soft px-3 py-2 text-sm text-danger">{children}</p>;
}

/** Badge for a run status (pending, running, succeeded, failed). */
export function StatusBadge({ status }: { status: string }) {
  const tone: Tone =
    status === "succeeded" ? "success" : status === "failed" ? "danger" : status === "running" ? "accent" : "neutral";
  return (
    <Badge tone={tone}>
      {status === "running" && <Spinner className="mr-1 h-2.5 w-2.5" />}
      {status}
    </Badge>
  );
}
