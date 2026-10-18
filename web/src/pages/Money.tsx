import { apiGet } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import type { MoneyData } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/PageHeader";
import { StatTile } from "@/components/StatTile";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from "recharts";
import { fmtUsd, pnlTone } from "@/lib/format";

export function Money() {
  const { data, error } = usePolling(() => apiGet<MoneyData>("/ui/money"), 15_000);

  // Plots `pnl` (equity minus accumulated deposits), not `equity`.
  // Gross equity jumps +$1000 at every automatic paper top-up, so the historical
  // run (−$1,204 real) was drawn as +280%.
  const chartData = (data?.equity_series ?? []).map((p) => ({
    t: p.ts_ms,
    pnl: parseFloat(p.pnl),
    equity: parseFloat(p.equity),
    cash: parseFloat(p.cash),
  }));
  const totalReturn = data ? parseFloat(data.total_return) : 0;
  const pnlTrend = chartData.slice(-24).map((p) => p.pnl);

  return (
    <div className="space-y-4">
      <PageHeader group="Analysis" title="Money" />
      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Error loading money data: {error}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <StatTile
          label="Equity (cash + positions)"
          value={data ? fmtUsd(data.equity) : "…"}
          numericValue={data ? parseFloat(data.equity) : undefined}
          size="lg"
          trend={pnlTrend}
        />
        <StatTile
          label="Total return"
          value={data ? fmtUsd(data.total_return) : "…"}
          tone={data ? pnlTone(totalReturn) : undefined}
          sub={data?.total_return_pct ? `${data.total_return_pct}%` : undefined}
        />
        <StatTile label="Available cash" value={data ? fmtUsd(data.cash) : "…"} />
        <StatTile label="Value in positions" value={data ? fmtUsd(data.positions_value) : "…"} />
        <StatTile
          label="Contributed capital (total)"
          value={data ? fmtUsd(data.deposits) : "…"}
          title="Configured initial contribution + automatic paper top-ups. The latter simulate the money you would have had to deposit for real so the follower could keep trading at this pace, so they count as actual contributed capital."
        />
      </div>

      {data && data.topups_count > 0 && (
        <div className="rounded-md border border-up/30 bg-up/5 px-3 py-2 font-mono text-xs text-muted-foreground">
          Of the {fmtUsd(data.deposits).replace("+", "")} contributed in total: {fmtUsd(data.initial_capital).replace("+", "")} was the configured initial contribution,
          and {fmtUsd(data.topups_total).replace("+", "")} came from {data.topups_count} {data.topups_count === 1 ? "automatic top-up" : "automatic top-ups"} on paper
          (the simulator adds +$1000 when cash drops below $100) — this simulates what you would have had to contribute for real to sustain the follower's pace,
          so it DOES count as contributed capital: "Total return" is already calculated over the total, top-ups included.
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>PnL curve</CardTitle>
          <p className="text-xs text-muted-foreground">
            Real PnL = equity minus accumulated deposits. Starts at 0 and does not jump with the
            automatic paper top-ups. The gross equity is shown as a dotted reference.
          </p>
        </CardHeader>
        <CardContent style={{ height: 260 }}>
          {chartData.length > 1 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.78 0.140 70)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="oklch(0.78 0.140 70)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--hair)" vertical={false} />
                <XAxis dataKey="t" tickFormatter={(t) => new Date(t).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })} stroke="var(--dim)" fontSize={11} />
                <YAxis domain={["auto", "auto"]} stroke="var(--dim)" fontSize={11} width={56} />
                <Tooltip
                  contentStyle={{ background: "var(--panel-3)", border: "1px solid var(--hair)", borderRadius: 6, fontSize: 12 }}
                  labelFormatter={(t) => new Date(t as number).toLocaleString("en-US")}
                  formatter={(v) => `${Number(v).toFixed(2)} $`}
                />
                <ReferenceLine y={0} stroke="var(--dim)" strokeWidth={1} />
                <Area type="monotone" dataKey="equity" name="Gross equity (incl. top-ups)" stroke="var(--dim)" fill="none" strokeWidth={1.25} strokeDasharray="4 3" dot={false} />
                <Area type="monotone" dataKey="pnl" name="PnL real" stroke="oklch(0.78 0.140 70)" fill="url(#equityFill)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <span className="text-sm text-muted-foreground">
              Not enough history yet — one point is saved every ~5 min, check back later.
            </span>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Total PnL by strategy (realized + unrealized)</CardTitle></CardHeader>
        <CardContent>
          {data?.by_strategy.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Strategy</TableHead>
                  <TableHead className="text-right">Realized</TableHead>
                  <TableHead className="text-right">Unrealized</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  <TableHead className="text-right">Exposure</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.by_strategy.map((r) => (
                  <TableRow key={r.strategy}>
                    <TableCell className="font-mono">{r.strategy}</TableCell>
                    <TableCell className={`text-right font-mono tabular-nums ${pnlTone(parseFloat(r.realized)) === "gain" ? "text-gain" : pnlTone(parseFloat(r.realized)) === "danger" ? "text-destructive" : ""}`}>{fmtUsd(r.realized)}</TableCell>
                    <TableCell className={`text-right font-mono tabular-nums ${pnlTone(parseFloat(r.unrealized)) === "gain" ? "text-gain" : pnlTone(parseFloat(r.unrealized)) === "danger" ? "text-destructive" : ""}`}>{fmtUsd(r.unrealized)}</TableCell>
                    <TableCell className={`text-right font-mono tabular-nums ${pnlTone(parseFloat(r.total)) === "gain" ? "text-gain" : pnlTone(parseFloat(r.total)) === "danger" ? "text-destructive" : ""}`}>{fmtUsd(r.total)}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-muted-foreground">{parseFloat(r.exposure).toFixed(2)} $</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <span className="text-sm text-muted-foreground">No activity yet</span>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
