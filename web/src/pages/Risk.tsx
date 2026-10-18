import { useState } from "react";
import { apiGet, apiPost } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import type { RiskData } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Gauge, ArcGauge } from "@/components/ui/gauge";
import { PageHeader } from "@/components/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

// Editor for the global exposure limit — a separate component so the
// `useState` for editing is initialized once with the value received
// from the first poll (same pattern as `StrategyCard.tsx` with the per-strategy
// limits), and is not overwritten while the user is typing.
function ExposureLimitEditor({ initial }: { initial: string }) {
  const [value, setValue] = useState(initial);

  async function save() {
    try {
      const resp = await apiPost<{ max_total_exposure_usdc: string; env_override: boolean }>(
        "/control/exposure_limit",
        { max_total_exposure_usdc: value },
      );
      if (resp.env_override) {
        alert("Saved, but the environment variable HRF_RISK__MAX_TOTAL_EXPOSURE_USDC is set: it will override this value on the next restart.");
      }
    } catch (e) {
      alert(`Error saving the exposure limit: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <div className="mt-2 flex items-center gap-2">
      <Input className="w-32" value={value} onChange={(e) => setValue(e.target.value)} placeholder="Limit $" />
      <Button size="sm" variant="outline" onClick={save}>Save</Button>
    </div>
  );
}

export function Risk() {
  const { data, error } = usePolling(() => apiGet<RiskData>("/ui/risk"), 3_000);

  const current = data ? parseFloat(data.exposure_current) : 0;
  const max = data ? parseFloat(data.exposure_max) : 0;

  async function resetKill(strategy: string) {
    try {
      await apiPost("/control/strategy_kill", { strategy, action: "reset" });
    } catch (e) {
      alert(`Error resetting the kill switch for ${strategy}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  const dangerZone = current / (max || 1) > 0.9;

  return (
    <div className="space-y-4">
      <PageHeader group="Monitoring" title="Risk" />
      {error && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">Error: {error}</div>}

      <Card>
        <CardHeader><CardTitle>Global exposure</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <ArcGauge value={current} max={max || 1} size={56} stroke={6} tone={dangerZone ? "danger" : "up"} />
            <div>
              <div className={`font-mono text-2xl font-semibold tabular-nums ${dangerZone ? "text-destructive" : ""}`}>
                {current.toFixed(2)} $ <span className="text-base font-normal text-muted-foreground">/ {max.toFixed(2)} $</span>
              </div>
              <Gauge value={current} max={max || 1} tone={dangerZone ? "danger" : "up"} className="mt-2 w-56" />
            </div>
          </div>
          {data && parseFloat(data.bracket_cap) > 0 && (
            <div className="mt-3 font-mono text-xs text-muted-foreground">Per-bracket cap configured: {data.bracket_cap} $</div>
          )}
          {data && <ExposureLimitEditor initial={data.exposure_max} />}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Kill switches by strategy</CardTitle></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Strategy</TableHead><TableHead>State</TableHead><TableHead>Reason</TableHead>
                <TableHead>Orders/h</TableHead><TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.strategies.map((s) => (
                <TableRow key={s.strategy}>
                  <TableCell className="font-mono">{s.strategy}</TableCell>
                  <TableCell>
                    <span className="inline-flex items-center gap-1.5 font-mono text-xs">
                      <span className={`h-1.5 w-1.5 rounded-full ${s.killed ? "bg-destructive" : "bg-gain"}`} />
                      {s.killed ? "KILLED" : "active"}
                    </span>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">{s.kill_reason ?? "—"}</TableCell>
                  <TableCell className="font-mono text-xs tabular-nums">{s.orders_sent_last_hour} / {s.max_orders_per_hour || "∞"}</TableCell>
                  <TableCell>{s.killed && <Button size="sm" variant="secondary" onClick={() => resetKill(s.strategy)}>Reset</Button>}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
