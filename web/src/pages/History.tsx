import { useEffect, useRef, useState } from "react";
import { apiGet } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/PageHeader";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface Trade {
  id: number; ts_ms: number; slug: string; market_cid: string; token_id: string;
  outcome_label: string | null; kind: string; shares: string; price: string;
  usdc: string; pnl: string | null; strategy: string;
}

export function History() {
  const [rows, setRows] = useState<Trade[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  const [exhausted, setExhausted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadMore() {
    if (exhausted) return;
    try {
      const q = cursor ? `?limit=500&before_id=${cursor}` : "?limit=500";
      const d = await apiGet<{ trades: Trade[] }>(`/ui/history${q}`);
      if (d.trades.length < 500) setExhausted(true);
      if (d.trades.length) {
        setCursor(d.trades[d.trades.length - 1].id);
        setRows((prev) => [...prev, ...d.trades]);
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  // Initial load in an effect (not in the render body). The guard uses
  // `useRef` (not `useState`): under StrictMode React re-invokes the effect after
  // cleaning it up, keeping the component state. A `useState` guard would still be
  // `false` in the second invocation (the first call's promise has not resolved
  // yet) and would fire two concurrent `loadMore()` calls, duplicating the first
  // rows. The ref does survive that cycle.
  const started = useRef(false);
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    loadMore();
    // Empty deps on purpose: it must run only once, on mount.
  }, []);

  async function exportCsv() {
    try {
      const all: Trade[] = [];
      let c: number | null = null;
      for (let i = 0; i < 500; i++) {
        const q: string = c ? `?limit=2000&before_id=${c}` : "?limit=2000";
        const d: { trades: Trade[] } = await apiGet<{ trades: Trade[] }>(`/ui/history${q}`);
        if (!d.trades.length) break;
        all.push(...d.trades);
        c = d.trades[d.trades.length - 1].id;
        if (d.trades.length < 2000) break;
      }
      // Sanitizes CSV values against formula injection: prefixes ' when it starts with =, +, -, @, tab or CR
      const sanitizeCsvCell = (v: unknown): string => {
        let s = String(v ?? "");
        if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
        return s;
      };
      const cols = ["date", "market", "token_id", "kind", "shares", "price", "usdc", "pnl", "strategy"];
      const q2 = (v: unknown) => `"${sanitizeCsvCell(v).replace(/"/g, '""')}"`;
      const csvRows = all.map((t) => [
        new Date(t.ts_ms).toISOString(), t.slug || t.market_cid, t.token_id,
        t.kind, t.shares, t.price, t.usdc, t.pnl ?? "", t.strategy,
      ].map(q2).join(","));
      const blob = new Blob([[cols.join(","), ...csvRows].join("\n")], { type: "text/csv" });
      const a = Object.assign(document.createElement("a"), {
        href: URL.createObjectURL(blob),
        download: `hrf_trades_${new Date().toISOString().slice(0, 10)}.csv`,
      });
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      alert(`Error exporting CSV: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <div className="space-y-2">
      <PageHeader
        group="Analysis"
        title="History"
        action={
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-muted-foreground">{rows.length}{exhausted ? "" : "+"} trades</span>
            <Button size="sm" variant="outline" onClick={exportCsv}>⬇ CSV</Button>
          </div>
        }
      />
      {error && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">Error: {error}</div>}
      {rows.length === 0 && !error && <span className="text-sm text-muted-foreground">No trades yet</span>}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date</TableHead><TableHead>Market</TableHead><TableHead>Type</TableHead>
            <TableHead>Shares</TableHead><TableHead>Price</TableHead><TableHead>USDC</TableHead>
            <TableHead>PnL</TableHead><TableHead>Strategy</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((t) => (
            <TableRow key={t.id}>
              <TableCell className="font-mono text-xs tabular-nums text-muted-foreground">{new Date(t.ts_ms).toLocaleString("en-US")}</TableCell>
              <TableCell className="max-w-[200px] truncate">{t.slug || t.market_cid.slice(0, 12)}</TableCell>
              <TableCell className="font-mono text-xs">{t.kind}</TableCell>
              <TableCell className="font-mono tabular-nums">{t.shares}</TableCell>
              <TableCell className="font-mono tabular-nums">{t.price}</TableCell>
              <TableCell className="font-mono tabular-nums">{t.usdc}</TableCell>
              <TableCell
                className={`font-mono tabular-nums ${
                  t.pnl === null ? "text-muted-foreground" : parseFloat(t.pnl) > 0 ? "text-gain" : parseFloat(t.pnl) < 0 ? "text-destructive" : ""
                }`}
              >
                {t.pnl ?? "—"}
              </TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground">{t.strategy}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {!exhausted && <Button size="sm" variant="outline" onClick={loadMore}>Load more</Button>}
    </div>
  );
}
