import { apiGet } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import type { AlertEntry, MoneyData, UiData, UiPosition, UiStrategyRow } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StrategyHealthStrip } from "@/components/StrategyHealthStrip";
import { StatTile } from "@/components/StatTile";
import { PageHeader } from "@/components/PageHeader";
import { ArcGauge, ProbabilityDot } from "@/components/ui/gauge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fmtUsd, fmtClear } from "@/lib/format";

interface MarketHealth { assets: { asset: string; last_price: string; secs_since_update: number }[] }

const SEVERITY_DOT: Record<AlertEntry["severity"], string> = {
  critical: "bg-destructive",
  warning: "bg-up",
  info: "bg-muted-foreground",
};

function PositionsTable({ positions }: { positions: UiPosition[] }) {
  if (!positions.length) {
    return <span className="text-sm text-muted-foreground">No open positions</span>;
  }
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Market</TableHead>
            <TableHead>Strategy</TableHead>
            <TableHead className="text-right">Size</TableHead>
            <TableHead className="text-right">Avg price</TableHead>
            <TableHead className="text-right">Cost</TableHead>
            <TableHead>Close</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {positions.map((p) => (
            <TableRow key={`${p.token_id}-${p.strategy}`}>
              <TableCell className="max-w-[220px] truncate" title={p.slug || p.market_cid}>
                {p.slug || p.market_cid.slice(0, 12)}
                {p.outcome_label && <span className="ml-1 text-muted-foreground">({p.outcome_label})</span>}
              </TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground">{p.strategy}</TableCell>
              <TableCell className="text-right font-mono tabular-nums">{parseFloat(p.size).toFixed(2)}</TableCell>
              <TableCell className="text-right">
                <span className="inline-flex items-center justify-end gap-2 font-mono tabular-nums">
                  {parseFloat(p.avg_price).toFixed(3)}
                  <ProbabilityDot p={parseFloat(p.avg_price)} />
                </span>
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">{fmtUsd(p.cost_basis)}</TableCell>
              <TableCell className={`font-mono text-xs ${p.secs_to_clear <= 0 ? "text-up" : "text-muted-foreground"}`}>
                {p.bail_sold ? "bail sold" : fmtClear(p.secs_to_clear)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function MarketPulse({ assets, error }: { assets: MarketHealth["assets"] | undefined; error: string | null }) {
  if (error) return <div className="text-sm text-destructive">Error: {error}</div>;
  if (!assets) return <span className="text-sm text-muted-foreground">Loading…</span>;
  if (!assets.length) return <span className="text-sm text-muted-foreground">No assets configured</span>;
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-2">
      {assets.map((a) => {
        const tone = a.secs_since_update < 15 ? "bg-gain" : a.secs_since_update < 60 ? "bg-up" : "bg-destructive";
        return (
          <div key={a.asset} className="flex items-center gap-2 font-mono text-xs">
            <span className={`h-1.5 w-1.5 rounded-full ${tone}`} aria-hidden />
            <span className="font-medium uppercase">{a.asset}</span>
            <span className="tabular-nums text-muted-foreground">
              {parseFloat(a.last_price).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            <span className="text-muted-foreground/60">{a.secs_since_update}s ago</span>
          </div>
        );
      })}
    </div>
  );
}

export function Summary() {
  const data = usePolling(() => apiGet<UiData>("/ui/data"), 10_000);
  const strategies = usePolling(() => apiGet<{ strategies: UiStrategyRow[] }>("/ui/strategies"), 15_000);
  const alerts = usePolling(() => apiGet<{ alerts: AlertEntry[] }>("/ui/alerts"), 15_000);
  const money = usePolling(() => apiGet<MoneyData>("/ui/money"), 20_000);
  const health = usePolling(() => apiGet<MarketHealth>("/ui/market_health"), 10_000);

  const d = data.data;
  // The sparkline follows the real PnL, not the gross equity (which steps
  // +$1000 at every automatic paper top-up and distorts the trend).
  const pnlTrend = (money.data?.equity_series ?? []).slice(-24).map((p) => parseFloat(p.pnl));
  const pnl = d ? parseFloat(d.pnl_total) : 0;
  const exposure = d ? parseFloat(d.open_cost) : 0;
  const expMax = d ? parseFloat(d.max_total_exposure) : 0;
  const wallet = d?.wallet_balance ? parseFloat(d.wallet_balance) : null;

  return (
    <div className="space-y-4">
      <PageHeader group="Monitoring" title="Summary" />

      {data.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Error loading data: {data.error}
        </div>
      )}
      {d?.cash_warning && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Negative cash — the simulator is below zero.
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
        <div className="md:col-span-2">
          <StatTile
            label="Portfolio (equity)"
            value={d ? fmtUsd(d.equity) : "…"}
            numericValue={d ? parseFloat(d.equity) : undefined}
            size="lg"
            trend={pnlTrend}
            trendTone="up"
            title="Available cash + mark-to-market value of open positions: the real account value, not the initial deposit. The sparkline shows PnL (equity minus deposits), not gross equity."
          />
        </div>
        <StatTile label="Realized PnL" value={d ? fmtUsd(d.pnl_total) : "…"} tone={pnl > 0 ? "gain" : pnl < 0 ? "danger" : undefined} />
        <StatTile label="Available cash" value={d ? fmtUsd(d.cash) : "…"} />
        <StatTile
          label="Exposure / limit"
          value={
            <span className="inline-flex items-center gap-2">
              <ArcGauge value={exposure} max={expMax || 1} size={20} stroke={3} tone={expMax > 0 && exposure / expMax > 0.9 ? "danger" : "up"} />
              {fmtUsd(exposure)} <span className="text-sm text-muted-foreground">/ {fmtUsd(expMax)}</span>
            </span>
          }
        />
        <StatTile label={wallet !== null ? "Wallet on-chain" : "Open positions"} value={wallet !== null ? `${wallet.toFixed(2)} $` : (d?.positions.length ?? "…")} />
      </div>

      <Card>
        <CardHeader><CardTitle>Market pulse</CardTitle></CardHeader>
        <CardContent><MarketPulse assets={health.data?.assets} error={health.error} /></CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Health by strategy</CardTitle></CardHeader>
        <CardContent>
          {strategies.error && (
            <div className="mb-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              Error loading strategies: {strategies.error}
            </div>
          )}
          {strategies.data ? <StrategyHealthStrip strategies={strategies.data.strategies} /> : strategies.error ? null : "Loading…"}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Open positions</CardTitle></CardHeader>
        <CardContent>
          {d ? <PositionsTable positions={d.positions} /> : !data.error && "Loading…"}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Latest alerts</CardTitle></CardHeader>
        <CardContent className="space-y-1.5 text-sm">
          {alerts.error && (
            <div className="mb-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              Error loading alerts: {alerts.error}
            </div>
          )}
          {alerts.data?.alerts.length
            ? alerts.data.alerts.slice(0, 5).map((a, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${SEVERITY_DOT[a.severity]}`} aria-hidden />
                  <span className="shrink-0 font-mono text-xs text-muted-foreground">{new Date(a.ts_ms).toLocaleString("en-US")}</span>
                  <span>{a.message}</span>
                </div>
              ))
            : !alerts.error && <span className="text-muted-foreground">No alerts</span>}
        </CardContent>
      </Card>
    </div>
  );
}
