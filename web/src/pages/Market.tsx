import { apiGet } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/PageHeader";
import { StatTile } from "@/components/StatTile";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface MarketHealth { assets: { asset: string; sigma_short: string | null; sigma_long: string | null; last_price: string; secs_since_update: number }[] }

function healthLed(secs: number) {
  const tone = secs < 15 ? "bg-gain" : secs < 60 ? "bg-up" : "bg-destructive";
  const label = secs < 15 ? "up to date" : secs < 60 ? "delayed" : "stale";
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-xs">
      <span className={`h-1.5 w-1.5 rounded-full ${tone}`} />
      {label}
    </span>
  );
}

// sigma arrives as a Rust Decimal with full precision (~28 digits) —
// these are per-tick volatility values, unreadable in fixed notation (0.0000435818...);
// scientific notation with 3 significant figures is the standard for this magnitude.
function fmtSigma(v: string): string {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n.toExponential(2) : v;
}

function fmtPrice(v: string): string {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n.toLocaleString("en-US",{ minimumFractionDigits: 2, maximumFractionDigits: 4 }) : v;
}

export function Market() {
  const { data, error } = usePolling(() => apiGet<MarketHealth>("/ui/market_health"), 10_000);
  const assets = data?.assets ?? [];
  const stale = assets.filter((a) => a.secs_since_update >= 60).length;

  return (
    <div className="space-y-4">
      <PageHeader group="Monitoring" title="Market" />
      {error && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">Error: {error}</div>}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <StatTile label="Monitored assets" value={data ? assets.length : "…"} />
        <StatTile label="Up to date (<15s)" value={data ? assets.filter((a) => a.secs_since_update < 15).length : "…"} tone="gain" />
        <StatTile label="Stale (≥60s)" value={data ? stale : "…"} tone={stale > 0 ? "danger" : "muted"} />
      </div>

      <Card>
        <CardHeader><CardTitle>Data health per asset</CardTitle></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Asset</TableHead><TableHead>Status</TableHead><TableHead>Last price</TableHead>
                <TableHead>Short σ (60s)</TableHead><TableHead>Long σ (300s)</TableHead><TableHead>Last update</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.assets.map((a) => (
                <TableRow key={a.asset}>
                  <TableCell className="font-mono font-medium uppercase">{a.asset}</TableCell>
                  <TableCell>{healthLed(a.secs_since_update)}</TableCell>
                  <TableCell className="font-mono tabular-nums">{fmtPrice(a.last_price)}</TableCell>
                  <TableCell className="font-mono tabular-nums">{a.sigma_short ? fmtSigma(a.sigma_short) : <span className="font-sans text-muted-foreground">insufficient sample</span>}</TableCell>
                  <TableCell className="font-mono tabular-nums">{a.sigma_long ? fmtSigma(a.sigma_long) : <span className="font-sans text-muted-foreground">insufficient sample</span>}</TableCell>
                  <TableCell className="font-mono text-xs tabular-nums text-muted-foreground">{a.secs_since_update}s ago</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
