import { apiGet } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import type { ModePermission, UiStrategyRow } from "@/lib/types";
import { computeModePermission, permissionVariant } from "@/lib/strategyAuth";
import { StrategyCard } from "@/components/StrategyCard";
import { PageHeader } from "@/components/PageHeader";
import { Badge } from "@/components/ui/badge";

/**
 * Strategies page: the single place where the operator checks the state of
 * all strategies and turns them on or off.
 *
 * The current mode (`paper`/`live`) comes from `/ui/data` (which already
 * returns it), not from `/status` — `/ui/data` is on the same polling path
 * as the strategies and refreshes at the same rate.
 *
 * If the backend does NOT expose `enabled`/`paper_only` in
 * `/ui/strategies` (a legacy scenario), the page shows an "Unknown
 * authorization" note and the running count is based on `killed`: the
 * permission state is never invented. The backend exposes these fields
 * since 2026-09-14.
 */
export function Strategies() {
  const strategies = usePolling(() => apiGet<{ strategies: UiStrategyRow[] }>("/ui/strategies"), 10_000);
  const dashboard  = usePolling(() => apiGet<{ mode: string }>("/ui/data"), 10_000);
  const mode = dashboard.data?.mode ?? "paper";
  const rows = strategies.data?.strategies ?? [];
  // Real state: the backend exposes it as `running` (live task + not killed).
  // If absent, fall back to `!killed` so older backends keep working.
  const isRunning = (s: UiStrategyRow) => s.running ?? !s.killed;
  const running_count = rows.filter((s) => isRunning(s)).length;

  // Count by effective permission: uses `mode_permission` if the backend
  // exposes it; otherwise recomputes it on the client from the flags received.
  // If the flags are missing too (the "unknown" scenario), the row counts in
  // no bucket. `rows` is reassigned on every poll (not stable across renders),
  // so the count is computed inline instead of in `useMemo`: it is a single
  // pass over 13 strategies.
  let blocked = 0;
  let unknown = 0;
  let ok = 0;
  for (const s of rows) {
    const p: ModePermission = s.mode_permission ?? computeModePermission(s, mode);
    if (p === "ok") ok++;
    else if (p === "unknown") unknown++;
    else blocked++;
  }
  const anyUnknown = unknown > 0;

  return (
    <div className="space-y-4">
      <PageHeader
        group="Management"
        title="Strategies"
        action={
          rows.length > 0 ? (
            <span className="font-mono text-xs text-muted-foreground">
              {running_count}/{rows.length} running_count
            </span>
          ) : undefined
        }
      />

      {strategies.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Failed to load strategies: {strategies.error}
        </div>
      )}

      {anyUnknown && rows.length > 0 && (
        <div
          role="status"
          className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          {unknown} strategy{unknown === 1 ? "" : "s"} without authorization data
          (`enabled` / `paper_only` not exposed by the
          backend's `/ui/strategies`). The "Enable" button is disabled for those
          rows, since a toggle could be rejected by the barrier without the UI
          knowing in advance.
        </div>
      )}

      {rows.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="font-mono uppercase tracking-wide text-muted-foreground">mode {mode}:</span>
          <Badge variant={permissionVariant("ok")}>{ok} authorized</Badge>
          {blocked > 0 && (
            <Badge variant="secondary">{blocked} blocked by barrier</Badge>
          )}
          {unknown > 0 && (
            <Badge variant="outline">{unknown} with unknown authorization</Badge>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {rows.map((s) => (
          <StrategyCard
            key={s.strategy}
            s={s}
            mode={mode}
            onChanged={() => { /* the next poll refreshes */ }}
          />
        ))}
      </div>
    </div>
  );
}