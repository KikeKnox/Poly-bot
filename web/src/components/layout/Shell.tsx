import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import type { PageId } from "@/pages";
import { apiPost, apiGet } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import type { UiData } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { ArcGauge } from "@/components/ui/gauge";
import { BracketTicker } from "@/components/BracketTicker";
import { fmtUsd } from "@/lib/format";

/**
 * Cockpit strip: the header's signature element. Earlier pages buried
 * equity, PnL and exposure in a summary tab. They live here instead, visible
 * from any tab, because they are the only three numbers an operator must be
 * able to read at a glance at any moment of a live trading session.
 */
export function Shell({
  current,
  onNavigate,
  children,
}: {
  current: PageId;
  onNavigate: (p: PageId) => void;
  children: ReactNode;
}) {
  const { data } = usePolling(() => apiGet<UiData>("/ui/data"), 10_000);
  const exposure = data ? parseFloat(data.open_cost) : 0;
  const expMax = data ? parseFloat(data.max_total_exposure) : 0;
  const expPct = expMax > 0 ? (exposure / expMax) * 100 : 0;
  const pnl = data ? parseFloat(data.pnl_total) : 0;
  const equity = data ? parseFloat(data.equity) : 0;
  const wallet = data?.wallet_balance ? parseFloat(data.wallet_balance) : null;

  async function toggleKill() {
    try {
      if (data?.global_killed) {
        await apiPost("/control/kill_switches");
      } else {
        if (!window.confirm("Activate global kill switch?")) return;
        await apiPost("/control/kill");
      }
    } catch (e) {
      alert(`Error changing kill switch state: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <div className="flex h-screen text-foreground">
      {/* No bg-background here on purpose: it would paint an opaque layer over
          the radial wash of `body` (index.css) and hide it completely. */}
      <Sidebar current={current} onNavigate={onNavigate} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex min-h-14 flex-wrap items-center gap-x-5 gap-y-2 border-b border-border bg-panel-2/50 px-4 py-2">
          <div className="flex items-center gap-2">
            <span className="relative flex h-1.5 w-1.5">
              {!data?.global_killed && (
                <span className="absolute inline-flex h-full w-full animate-led-pulse rounded-full bg-gain" />
              )}
              <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${data?.global_killed ? "bg-destructive" : "bg-gain"}`} />
            </span>
            <span className="font-mono text-[11px] tracking-wide text-muted-foreground">
              {data?.global_killed ? "HALTED" : "RUNNING"}
            </span>
            <span
              className={`rounded-sm border px-1.5 py-0.5 font-mono text-[10px] tracking-wider ${
                data?.mode === "live" ? "border-destructive/50 text-destructive" : "border-down/40 text-down"
              }`}
            >
              {(data?.mode ?? "…").toUpperCase()}
            </span>
          </div>

          <div className="h-6 w-px bg-border" aria-hidden />

          <div className="flex items-baseline gap-1.5" title="Available cash + market value of open positions">
            <span className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">equity</span>
            <span className="font-mono text-sm font-semibold tabular-nums">{data ? fmtUsd(equity) : "…"}</span>
          </div>
          <div className="flex items-baseline gap-1.5" title="Realized PnL — full history">
            <span className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">pnl</span>
            <span className={`font-mono text-sm tabular-nums ${pnl > 0 ? "text-gain" : pnl < 0 ? "text-destructive" : ""}`}>
              {data ? fmtUsd(pnl) : "…"}
            </span>
          </div>
          {wallet !== null && (
            <div className="hidden items-baseline gap-1.5 md:flex" title="On-chain wallet balance (live)">
              <span className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">wallet</span>
              <span className="font-mono text-sm tabular-nums">{wallet.toFixed(2)} $</span>
            </div>
          )}

          <div className="h-6 w-px bg-border" aria-hidden />

          <div className="flex items-center gap-2" title="Open exposure against the global limit">
            <ArcGauge value={exposure} max={expMax || 1} size={26} stroke={3} tone={expPct > 90 ? "danger" : "up"} />
            <div className="leading-tight">
              <div className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">exposure</div>
              <div className="font-mono text-xs tabular-nums">
                {exposure.toFixed(0)} <span className="text-muted-foreground">/ {expMax.toFixed(0)}$</span>
              </div>
            </div>
          </div>

          <div className="flex-1" />

          <Button
            size="sm"
            variant={data?.global_killed ? "outline" : "destructive"}
            className={`font-mono text-xs tracking-wider transition-shadow ${
              data?.global_killed ? "" : "hover:shadow-[0_0_18px_oklch(0.63_0.210_25/45%)]"
            }`}
            onClick={toggleKill}
          >
            {data?.global_killed ? "REARM" : "KILL"}
          </Button>
        </header>
        <div className="hairline-updown" aria-hidden />
        <BracketTicker positions={data?.positions ?? []} />
        <main className="flex-1 overflow-auto p-4">{children}</main>
      </div>
    </div>
  );
}
