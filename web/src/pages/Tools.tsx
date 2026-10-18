import { apiGet, apiPost } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import type { ToolRow } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/PageHeader";
import { Button } from "@/components/ui/button";

export function Tools() {
  const tools = usePolling(() => apiGet<{ tools: ToolRow[] }>("/ui/tools"), 10_000);
  const rows = tools.data?.tools ?? [];
  const scan = rows.filter((t) => t.kind === "scan");
  const recorders = rows.filter((t) => t.kind === "recorder");

  async function toggleTool(t: ToolRow) {
    try {
      await apiPost("/control/strategy", { strategy: t.tool, enabled: !t.running });
    } catch (e) {
      alert(`Error changing status of ${t.tool}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function renderGroup(title: string, group: ToolRow[]) {
    return (
      <section className="space-y-2">
        <h2 className="text-sm font-semibold text-foreground">{title}</h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {group.map((t) => (
            <Card key={t.tool}>
              <CardHeader className="flex flex-row items-center gap-2 space-y-0">
                <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${t.running ? "bg-gain" : "bg-muted-foreground"}`} aria-hidden />
                <CardTitle className="font-mono text-sm normal-case tracking-tight text-foreground">{t.tool}</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-between">
                <span className="font-mono text-xs text-muted-foreground">
                  {t.running ? "running" : "stopped"}
                  <span className="ml-2 text-muted-foreground/70">{t.enabled ? "config: enabled" : "config: disabled"}</span>
                </span>
                <Button size="sm" variant={t.running ? "secondary" : "outline"} onClick={() => toggleTool(t)}>
                  {t.running ? "Stop" : "Start"}
                </Button>
              </CardContent>
            </Card>
          ))}
          {group.length === 0 && <span className="text-sm text-muted-foreground">No tools in this group</span>}
        </div>
      </section>
    );
  }

  return (
    <div className="space-y-4">
      <PageHeader group="Management" title="Tools" />
      {tools.error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Error loading tools: {tools.error}
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        They only scan and record signals — they do not go through RiskGuard and do not send orders.
      </p>
      {renderGroup("Scan", scan)}
      {renderGroup("Recorders", recorders)}
      {tools.data && rows.length === 0 && <span className="text-sm text-muted-foreground">No tools configured</span>}
    </div>
  );
}
