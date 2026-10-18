import type { UiPosition } from "@/lib/types";
import { fmtClear } from "@/lib/format";

/**
 * Bracket ticker: the product's central object (an up/down bracket, with its
 * YES/NO split and its countdown) made visual rather than yet another abstract
 * number. Each capsule splits its fill into amber (the fraction given by
 * `avg_price`, the same data `ProbabilityDot` already uses) and cyan (the
 * rest) — a two-tone generalization of the probability dot, which used to
 * paint only one side.
 *
 * Deliberately static (wrap, never scroll/marquee): a continuously moving
 * strip during a live trading session would distract from what matters.
 * If there are no positions nothing is rendered — an empty strip is worse
 * than none.
 */
export function BracketTicker({ positions }: { positions: UiPosition[] }) {
  if (!positions.length) return null;

  return (
    <div className="flex flex-wrap gap-1.5 border-b border-border bg-panel-2/30 px-4 py-1.5">
      {positions.map((p) => {
        const pct = Math.min(100, Math.max(0, parseFloat(p.avg_price) * 100));
        const label = p.outcome_label || p.slug || p.market_cid.slice(0, 10);
        return (
          <span
            key={`${p.token_id}-${p.strategy}`}
            title={`${p.slug || p.market_cid} · ${p.strategy}`}
            className="inline-flex items-center gap-1.5 overflow-hidden rounded-full border border-border bg-panel-3/50 py-0.5 pl-0.5 pr-2 font-mono text-[10px] tabular-nums"
          >
            <span className="relative inline-block h-3 w-6 shrink-0 overflow-hidden rounded-full" aria-hidden>
              <span className="absolute inset-0 bg-down" />
              <span className="absolute inset-y-0 left-0 bg-up" style={{ width: `${pct}%` }} />
            </span>
            <span className="max-w-24 truncate text-foreground/80">{label}</span>
            <span className="text-muted-foreground">{pct.toFixed(0)}%</span>
            <span className={p.secs_to_clear <= 0 ? "text-up" : "text-muted-foreground/70"}>
              {p.bail_sold ? "bail" : fmtClear(p.secs_to_clear)}
            </span>
          </span>
        );
      })}
    </div>
  );
}
