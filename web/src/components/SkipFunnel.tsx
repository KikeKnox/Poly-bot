import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";
import type { FunnelCounter } from "@/lib/types";

export function SkipFunnel({ strategy }: { strategy: string }) {
  const [counters, setCounters] = useState<FunnelCounter[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiGet<{ counters: FunnelCounter[] }>(`/ui/strategy_funnel?strategy=${strategy}`)
      .then((r) => { if (!cancelled) setCounters(r.counters); })
      .catch((e) => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [strategy]);

  if (error) return <div className="text-xs text-destructive">Error: {error}</div>;
  if (!counters) return <div className="text-xs text-muted-foreground">Loading…</div>;
  if (counters.length === 0) return <div className="text-xs text-muted-foreground">No signals evaluated yet</div>;

  const max = Math.max(...counters.map((c) => c.value), 1);
  return (
    <div className="space-y-1">
      {counters.map((c) => {
        // `c.name` may carry attached labels, e.g. `hrf_pcs2_skip_total{reason=late}`
        // (counters with the same metric but different labels, see strategy_funnel.rs).
        // The `{...}` suffix is split off before stripping the prefix/suffix of the base name,
        // and reformatted as "(reason=late)" so it stays readable.
        const [base, labelPart] = c.name.split(/(?=\{)/, 2);
        const label = labelPart ? ` (${labelPart.slice(1, -1)})` : "";
        const display = base.replace(/^hrf_\w+?_skip_/, "").replace(/_total$/, "") + label;
        return (
          <div key={c.name} className="flex items-center gap-2 text-xs">
            <span className="w-40 truncate text-muted-foreground" title={c.name}>{display}</span>
            <div className="h-2 flex-1 overflow-hidden rounded bg-muted">
              <div className="h-full bg-primary" style={{ width: `${(c.value / max) * 100}%` }} />
            </div>
            <span className="w-12 text-right">{c.value}</span>
          </div>
        );
      })}
    </div>
  );
}
