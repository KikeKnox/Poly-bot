import { apiGet } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/PageHeader";
import { StatTile } from "@/components/StatTile";
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from "recharts";

interface PnlSeries { ts_ms: number[]; pnl: string[] }
interface StrategyStat {
  strategy: string; n: number; pnl: string; wins: number; won: string; lost: string; best: string; worst: string;
}
interface MetricsStats { by_strategy: StrategyStat[]; total_volume: string }
interface PnlAnalytics {
  max_drawdown: string;
  by_hour: { hour: number; pnl: string }[];
  markout_by_strategy: { strategy: string; n: number; markout_5s_avg_bps: string }[];
}

const tooltipStyle = { background: "var(--panel-3)", border: "1px solid var(--hair)", borderRadius: 6, fontSize: 12 };
const axisProps = { stroke: "var(--dim)", fontSize: 11 };

export function Metrics() {
  const series = usePolling(() => apiGet<PnlSeries>("/ui/pnl_series"), 15_000);
  const stats = usePolling(() => apiGet<MetricsStats>("/ui/metrics_stats"), 15_000);
  const analytics = usePolling(() => apiGet<PnlAnalytics>("/ui/pnl_analytics"), 30_000);

  const loadError = series.error ?? stats.error ?? analytics.error;
  const rows = stats.data?.by_strategy ?? [];
  const n = rows.reduce((s, r) => s + r.n, 0);
  const wins = rows.reduce((s, r) => s + r.wins, 0);
  const wr = n > 0 ? ((wins / n) * 100).toFixed(1) : "—";
  const totalPnl = rows.reduce((s, r) => s + parseFloat(r.pnl), 0);
  const avg = n > 0 ? (totalPnl / n).toFixed(4) : "—";
  const drawdown = analytics.data ? parseFloat(analytics.data.max_drawdown) : 0;

  let cum = 0;
  const chartData = (series.data?.ts_ms ?? []).map((t, i) => {
    cum += parseFloat(series.data!.pnl[i]);
    return { t, pnl: Number(cum.toFixed(6)) };
  });

  return (
    <div className="space-y-4">
      <PageHeader group="Analysis" title="Metrics" />
      {loadError && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Error loading metrics: {loadError}
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <StatTile label="Win rate" value={n > 0 ? `${wr}%` : "—"} tone={n > 0 && parseFloat(wr) >= 50 ? "gain" : undefined} />
        <StatTile label="Closed trades" value={n || "—"} />
        <StatTile label="Avg PnL/trade" value={avg} />
        <StatTile label="Volume USDC" value={stats.data ? `${parseFloat(stats.data.total_volume).toFixed(2)} $` : "…"} />
        <StatTile
          label="Max drawdown"
          value={analytics.data ? (drawdown > 0 ? `-${drawdown.toFixed(2)} $` : "0.00 $") : "…"}
          tone={drawdown > 0 ? "danger" : "muted"}
        />
      </div>

      <Card>
        <CardHeader><CardTitle>Cumulative PnL</CardTitle></CardHeader>
        <CardContent style={{ height: 240 }}>
          {chartData.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="pnlFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.78 0.140 70)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="oklch(0.78 0.140 70)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--hair)" vertical={false} />
                <XAxis dataKey="t" tickFormatter={(t) => new Date(t).toLocaleDateString("en-US")} {...axisProps} />
                <YAxis {...axisProps} width={56} />
                <Tooltip contentStyle={tooltipStyle} labelFormatter={(t) => new Date(t as number).toLocaleString("en-US")} />
                <Area type="monotone" dataKey="pnl" stroke="oklch(0.78 0.140 70)" fill="url(#pnlFill)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <span className="text-sm text-muted-foreground">No closed trades yet</span>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <CardTitle>Stats by strategy</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 text-sm">
          {rows.map((r) => {
            const wr = r.n > 0 ? (r.wins / r.n) * 100 : 0;
            const pnl = parseFloat(r.pnl);
            return (
              <div key={r.strategy} className="flex items-center justify-between gap-3 border-b border-border py-1.5 font-mono text-xs tabular-nums">
                <span className="w-32 shrink-0 truncate text-muted-foreground">{r.strategy}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-black/30">
                  <div className="h-full bg-up" style={{ width: `${wr}%` }} />
                </div>
                <span className="w-24 shrink-0 text-right">{r.wins}/{r.n} ({wr.toFixed(0)}%)</span>
                <span className={`w-20 shrink-0 text-right ${pnl >= 0 ? "text-gain" : "text-destructive"}`}>{pnl >= 0 ? "+" : ""}{r.pnl}</span>
              </div>
            );
          })}
          {rows.length === 0 && <span className="text-muted-foreground">Not enough data</span>}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>PnL by hour of day (UTC)</CardTitle></CardHeader>
          <CardContent style={{ height: 200 }}>
            {analytics.data?.by_hour.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.data.by_hour}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--hair)" vertical={false} />
                  <XAxis dataKey="hour" {...axisProps} />
                  <YAxis {...axisProps} width={48} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                    {analytics.data.by_hour.map((h, i) => (
                      <Cell key={i} fill={parseFloat(h.pnl) >= 0 ? "oklch(0.72 0.160 155)" : "oklch(0.63 0.210 25)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <span className="text-sm text-muted-foreground">Not enough data</span>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Markout by strategy (5s, bps)</CardTitle></CardHeader>
          <CardContent className="space-y-1.5 text-sm">
            {analytics.data?.markout_by_strategy.length
              ? analytics.data.markout_by_strategy.map((m) => (
                  <div key={m.strategy} className="flex items-center justify-between font-mono text-xs tabular-nums">
                    <span className="text-muted-foreground">{m.strategy}</span>
                    <span className="text-muted-foreground/60">n={m.n}</span>
                    <span className={parseFloat(m.markout_5s_avg_bps) >= 0 ? "text-gain" : "text-destructive"}>{m.markout_5s_avg_bps} bps</span>
                  </div>
                ))
              : <span className="text-muted-foreground">Not enough data</span>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
