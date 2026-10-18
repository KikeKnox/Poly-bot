import { cn } from "@/lib/utils";

/**
 * Level bar — the signature element of the dashboard. Everything in this product
 * is either a price-as-probability (0 to 1; a YES at 0.31 IS 31%
 * implied) or a consumption against a limit (exposure vs. cap). Both answer
 * "how full is the bar", so they share this component instead of plain text
 * "X / Y $" — it is reused in Summary, Risk (exposure) and in the positions
 * tables (implied probability).
 */
export function Gauge({
  value,
  max,
  tone = "up",
  className,
}: {
  value: number;
  max: number;
  tone?: "up" | "down" | "danger";
  className?: string;
}) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  const fill = tone === "danger" ? "bg-destructive" : tone === "down" ? "bg-down" : "bg-up";
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-black/30", className)}>
      <div className={cn("h-full transition-[width] duration-500", fill)} style={{ width: `${pct}%` }} />
    </div>
  );
}

/** Inline probability dot for table rows — a miniature YES price. */
export function ProbabilityDot({ p }: { p: number }) {
  const pct = Math.min(100, Math.max(0, p * 100));
  return (
    <span className="inline-flex items-center gap-1.5" title={`${pct.toFixed(1)}% implied`}>
      <span className="relative inline-block h-1.5 w-8 overflow-hidden rounded-full bg-black/30">
        <span className="absolute inset-y-0 left-0 bg-up" style={{ width: `${pct}%` }} />
      </span>
    </span>
  );
}

/**
 * Level ring — variant of Gauge for the header, where a bar's width does not
 * fit. Same data (consumption against a limit), different geometry: a ring
 * that closes as the remaining exposure shrinks. The gap between the stroke
 * and a full 360° IS the available margin, not decoration.
 */
export function ArcGauge({
  value,
  max,
  tone = "up",
  size = 34,
  stroke = 4,
}: {
  value: number;
  max: number;
  tone?: "up" | "down" | "danger";
  size?: number;
  stroke?: number;
}) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const color = tone === "danger" ? "var(--danger)" : tone === "down" ? "var(--cyan)" : "var(--amber)";
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90 shrink-0" aria-hidden>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--hair)" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={c - (pct / 100) * c}
        className="transition-[stroke-dashoffset] duration-500"
      />
    </svg>
  );
}
