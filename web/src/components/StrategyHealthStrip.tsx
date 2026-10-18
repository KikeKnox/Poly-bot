import type { UiStrategyRow } from "@/lib/types";

export function StrategyHealthStrip({ strategies }: { strategies: UiStrategyRow[] }) {
  if (!strategies.length) {
    return <span className="text-sm text-muted-foreground">No strategies configured</span>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      {strategies.map((s) => {
        const pnl = s.pnl ? parseFloat(s.pnl) : null;
        return (
          <span
            key={s.strategy}
            title={s.killed ? "killed" : "running"}
            className={`inline-flex items-center gap-1.5 rounded-sm border px-2 py-1 font-mono text-xs ${
              s.killed ? "border-destructive/30 text-muted-foreground" : "border-border bg-panel-3/40 text-foreground"
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${s.killed ? "bg-destructive" : "bg-gain"}`} aria-hidden />
            {s.strategy}
            {pnl !== null && (
              <span className={`tabular-nums ${pnl > 0 ? "text-gain" : pnl < 0 ? "text-destructive" : "text-muted-foreground"}`}>
                {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}$
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}
