import type { ReactNode } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkline } from "@/components/ui/sparkline";
import { cn } from "@/lib/utils";

/**
 * Hero-figure tile — the block repeated in Summary/Money/Metrics for any
 * number that matters most. It replaces the earlier pattern of
 * Card+CardHeader+CardTitle repeated by hand on each page with a different
 * font size each time; here the large size (`size="lg"`) is the one that
 * is actually readable from afar, the rest are supporting figures.
 */
export function StatTile({
  label,
  value,
  tone,
  sub,
  trend,
  trendTone = "up",
  size = "default",
  title,
  numericValue,
}: {
  label: string;
  value: ReactNode;
  tone?: "gain" | "danger" | "muted";
  sub?: ReactNode;
  trend?: number[];
  trendTone?: "up" | "down" | "gain" | "danger";
  size?: "default" | "lg";
  title?: string;
  /** Only for size="lg": the real sign behind `value` (which may be an
   * already formatted ReactNode) — determines the background wash. Without it,
   * the hero tile stays neutral rather than faking a sign it does not know. */
  numericValue?: number;
}) {
  const toneClass = tone === "gain" ? "text-gain" : tone === "danger" ? "text-destructive" : tone === "muted" ? "text-muted-foreground" : "";
  const isHero = size === "lg";
  const wash =
    isHero && numericValue !== undefined
      ? numericValue >= 0
        ? "radial-gradient(ellipse 420px 160px at 0% 0%, oklch(0.78 0.140 70 / 8%), transparent 70%)"
        : "radial-gradient(ellipse 420px 160px at 0% 0%, oklch(0.63 0.210 25 / 8%), transparent 70%)"
      : undefined;

  return (
    <Card style={wash ? { backgroundImage: wash } : undefined}>
      <CardHeader className="flex flex-row items-start justify-between gap-2 space-y-0">
        <CardTitle>{label}</CardTitle>
        {trend && trend.length > 1 && <Sparkline points={trend} tone={trendTone} />}
      </CardHeader>
      <CardContent title={title}>
        <div
          className={cn(
            "font-mono tabular-nums",
            isHero ? "text-5xl font-semibold tracking-tight sm:text-6xl" : "text-xl",
            isHero && !toneClass && "glow-amber-text",
            toneClass
          )}
        >
          {value}
        </div>
        {sub && <div className="mt-1 text-xs text-muted-foreground">{sub}</div>}
      </CardContent>
    </Card>
  );
}
