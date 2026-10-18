import { apiGet } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/PageHeader";
import { StatTile } from "@/components/StatTile";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

interface RiskSnapshot {
  global_killed: boolean;
  global_reason: string | null;
  session_pnl: string;
  daily_loss_stop: string;
  max_total_exposure: string;
}
interface StatusTool { tool: string; kind: "scan" | "recorder"; enabled: boolean; running: boolean; is_tool: boolean }
interface StatusData {
  mode: string;
  global_killed: boolean;
  risk: RiskSnapshot;
  tools: StatusTool[];
  dead_letter_fills_unresolved: number;
}

export function System() {
  const status = usePolling(() => apiGet<StatusData>("/status"), 10_000);
  const s = status.data;
  const deadLetters = s?.dead_letter_fills_unresolved ?? 0;

  return (
    <div className="space-y-4">
      <PageHeader group="System" title="System" />
      {status.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Error loading state: {status.error}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Modo" value={s ? s.mode.toUpperCase() : "…"} tone={s?.mode === "live" ? "danger" : "muted"} />
        <StatTile label="Kill global" value={s ? (s.global_killed ? "ASSET" : "off") : "…"} tone={s?.global_killed ? "danger" : undefined} />
        <StatTile
          label="Unresolved fills"
          value={s ? deadLetters : "…"}
          tone={deadLetters > 0 ? "danger" : "muted"}
          sub={deadLetters > 0 ? "require manual review" : "queue clear"}
          title="Fills that arrived but could not be reconciled automatically against a position — table dead_letter_fills."
        />
        <StatTile label="Tools active" value={s ? `${s.tools.filter((t) => t.running).length}/${s.tools.length}` : "…"} />
      </div>

      <Card>
        <CardHeader><CardTitle>Risk global</CardTitle></CardHeader>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Session PnL</div>
            <div className={`font-mono text-lg tabular-nums ${s && parseFloat(s.risk.session_pnl) < 0 ? "text-destructive" : s && parseFloat(s.risk.session_pnl) > 0 ? "text-gain" : ""}`}>
              {s ? `${parseFloat(s.risk.session_pnl).toFixed(2)} $` : "…"}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Daily stop</div>
            <div className="font-mono text-lg tabular-nums">{s ? `${parseFloat(s.risk.daily_loss_stop).toFixed(2)} $` : "…"}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Max exposure</div>
            <div className="font-mono text-lg tabular-nums">{s ? `${parseFloat(s.risk.max_total_exposure).toFixed(2)} $` : "…"}</div>
          </div>
          {s?.risk.global_reason && (
            <div className="col-span-full text-xs text-destructive">Kill reason: {s.risk.global_reason}</div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Tools (scan-only)</CardTitle></CardHeader>
        <CardContent className="flex flex-wrap gap-x-4 gap-y-2">
          {s?.tools.map((t) => (
            <span key={t.tool} className="inline-flex items-center gap-1.5 font-mono text-xs text-muted-foreground">
              <span className={`h-1.5 w-1.5 rounded-full ${t.running ? "bg-gain" : "bg-muted-foreground"}`} aria-hidden />
              {t.tool}
            </span>
          ))}
          {s && s.tools.length === 0 && <span className="text-sm text-muted-foreground">None configured</span>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Prometheus</CardTitle></CardHeader>
        <CardContent>
          <a href="/metrics" target="_blank" rel="noreferrer" className="font-mono text-sm text-primary underline decoration-primary/30 underline-offset-4 hover:decoration-primary">
            /metrics ↗
          </a>
        </CardContent>
      </Card>

      <Card size="sm">
        <CardContent>
          <Accordion>
            <AccordionItem value="raw">
              <AccordionTrigger className="font-mono text-xs text-muted-foreground">View raw /status payload</AccordionTrigger>
              <AccordionContent>
                <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs text-muted-foreground">{JSON.stringify(s, null, 2)}</pre>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>
    </div>
  );
}
