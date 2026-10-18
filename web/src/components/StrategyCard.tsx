import { useState } from "react";
import type { UiStrategyRow } from "@/lib/types";
import { ApiError, apiPost } from "@/lib/api";
import {
  canActivateFromUi,
  computeModePermission,
  flagLabel,
  isCurrentlyRunning,
  permissionExplain,
  permissionLabel,
  permissionVariant,
} from "@/lib/strategyAuth";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { SkipFunnel } from "./SkipFunnel";

/**
 * State of the Enable/Disable toggle.
 * - `idle`    → no request in flight
 * - `pending` → request in progress: both buttons stay disabled and the
 *               Enable button shows "Switching…"
 * - `error`   → the backend rejected it (HTTP 4xx/5xx). The strategy state
 *               ON SCREEN stays as it was (it is not flipped locally to
 *               "success") and the message appears in an accessible
 *               `role="alert"` banner, never in a modal `alert()`.
 *
 * The backend always allows DISABLING (see the early return `if !enabled` in
 * `StrategySupervisor::set_enabled`), so "Disable" is never blocked by the
 * permission state — even if the client thinks Enable is blocked.
 * Keeping that asymmetry avoids the "my strategy is running and I cannot
 * stop it" path.
 */
type ToggleState =
  | { kind: "idle" }
  | { kind: "pending"; target: "on" | "off" }
  | { kind: "error"; message: string; attempted: "on" | "off" };

export function StrategyCard({
  s,
  mode,
  onChanged,
}: {
  s: UiStrategyRow;
  /** Current bot mode (`"paper"` or `"live"`), read from the page
   *  payload — so the effective permission is evaluated against the real
   *  mode, not an invented value. */
  mode: string;
  onChanged: () => void;
}) {
  const [usdc, setUsdc]   = useState(s.max_order_usdc);
  const [rate, setRate]   = useState(s.max_orders_per_hour);
  const [dloss, setDloss] = useState(s.daily_loss_stop);
  const [toggle, setToggle] = useState<ToggleState>({ kind: "idle" });

  // REAL state of the strategy (boot + on-demand spawn), not the TOML's
  // boolean flag: if the backend exposes it (`running`) we use it,
  // otherwise we fall back to the previous behavior (`!killed`). It is never used
  // to decide the toggle — only to label the card and decide
  // whether the button says "Enable" or "Disable".
  const isOn = isCurrentlyRunning(s);
  // `enabled` is still shown as a badge (it is the TOML CONFIG, not the real
  // state), but it takes no part in the toggle decision. Whether
  // the backend exposes it only affects the badge style.
  const enabledKnown = s.enabled !== undefined && s.enabled !== null;

  // Effective permission: preferred from the backend (`s.mode_permission`, already
  // computed with the right policy, which assumes `enabled=true`); if
  // missing, replicate it locally with `computeModePermission`, which follows
  // the same matrix as `mode_permission_for_toggle` in the backend.
  const perm = s.mode_permission ?? computeModePermission(s, mode);
  const activateAllowed = canActivateFromUi(perm);

  async function toggleEnabled() {
    if (toggle.kind === "pending") return;
    // UX guard: if the UI already KNOWS that Enable will fail (perm ∈
    // {blocked_disabled, blocked_paper_only}),
    // we disable the button and this guard is just defense in depth.
    // It does NOT apply to "Disable": switching off is never blocked.
    const target: "on" | "off" = isOn ? "off" : "on";
    if (target === "on" && !activateAllowed) return;

    setToggle({ kind: "pending", target });
    try {
      await apiPost("/control/strategy", { strategy: s.strategy, enabled: target === "on" });
      setToggle({ kind: "idle" });
      onChanged();
    } catch (e) {
      // 403 (live barrier) or another toggle error: the state ON SCREEN
      // stays as it was. The next poll refreshes the backend's truth.
      const message =
        e instanceof ApiError
          ? e.status === 403
            ? `Rejected by the paper/live barrier: ${e.message}`
            : `Error ${e.status ?? "?"}: ${e.message}`
          : `Error switching the state of ${s.strategy}: ${String(e)}`;
      setToggle({ kind: "error", message, attempted: target });
    }
  }
  function dismissError() {
    setToggle({ kind: "idle" });
  }

  async function kill() {
    try {
      await apiPost("/control/strategy_kill", { strategy: s.strategy, action: s.killed ? "reset" : "kill" });
      onChanged();
    } catch (e) {
      alert(`Error applying kill on ${s.strategy}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  async function saveLimits() {
    try {
      await apiPost("/control/strategy_limits", {
        strategy: s.strategy,
        max_order_usdc: usdc || null,
        max_orders_per_hour: rate ? parseInt(rate, 10) : null,
        daily_loss_usdc_stop: dloss || null,
      });
      onChanged();
    } catch (e) {
      alert(`Error saving limits of ${s.strategy}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  const pnl = s.pnl ? parseFloat(s.pnl) : null;
  const isPending = toggle.kind === "pending";
  const isError = toggle.kind === "error";

  return (
    <Card className={s.killed ? "border-destructive/30 opacity-60" : undefined}>
      <CardHeader className="flex flex-row items-center gap-2 space-y-0 pb-2">
        <span className="relative flex h-1.5 w-1.5 shrink-0">
          {!s.killed && <span className="absolute inline-flex h-full w-full animate-led-pulse rounded-full bg-gain" />}
          <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${s.killed ? "bg-destructive" : "bg-gain"}`} />
        </span>
        <span className="font-mono text-sm font-medium tracking-tight">{s.strategy}</span>
        <span className={`ml-auto font-mono text-base font-semibold tabular-nums ${pnl && pnl > 0 ? "text-gain" : pnl && pnl < 0 ? "text-destructive" : "text-muted-foreground"}`}>
          {s.pnl ? `${pnl! >= 0 ? "+" : ""}${pnl!.toFixed(2)} $` : "—"}
        </span>
      </CardHeader>

      <CardContent className="space-y-2">
        {/* Permission strip: raw flags + effective permission.
            The effective permission governs the "Enable" button: if it is
            "ok" it can be pressed; any other reason disables the
            button with a tooltip explaining why. The "Disable" action stays
            always enabled (the backend guarantees it). */}
        <div className="flex flex-wrap items-center gap-1.5" aria-label="Strategy authorization status">
          <Badge
            variant={enabledKnown ? (s.enabled ? "default" : "secondary") : "outline"}
            title={enabledKnown
              ? `enabled = ${s.enabled} (config/strategies/${s.strategy}.toml)`
              : "Backend did not expose `enabled` in /ui/strategies"}
          >
            enabled: {flagLabel(s.enabled)}
          </Badge>
          <Badge
            variant={s.paper_only === null || s.paper_only === undefined
              ? "outline"
              : s.paper_only ? "destructive" : "secondary"}
            title={s.paper_only === null || s.paper_only === undefined
              ? "Backend did not expose `paper_only`"
              : s.paper_only
                ? "paper_only = true: this experiment never trades with real money"
                : "paper_only = false"}
          >
            paper_only: {flagLabel(s.paper_only)}
          </Badge>
          <Badge variant={permissionVariant(perm)} title={permissionExplain(perm, mode)}>
            mode {mode}: {permissionLabel(perm)}
          </Badge>
        </div>

        {/* Toggle error banner. role="alert" so screen readers
            announce it when it appears, without needing focus. It is NOT
            a modal `alert()`: the operator keeps control of the
            cursor and can read the message calmly. */}
        {isError && (
          <div
            role="alert"
            className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
          >
            <span className="flex-1">{toggle.message}</span>
            <button
              type="button"
              onClick={dismissError}
              className="font-mono text-xs uppercase tracking-wide underline underline-offset-2 hover:no-underline"
              aria-label="Close error message"
            >
              close
            </button>
          </div>
        )}

        <div className="grid grid-cols-3 gap-2">
          <div>
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-muted-foreground">Max $/order</span>
            <Input className="font-mono tabular-nums" value={usdc} onChange={(e) => setUsdc(e.target.value)} />
          </div>
          <div>
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-muted-foreground">Max orders/h</span>
            <Input className="font-mono tabular-nums" value={rate} onChange={(e) => setRate(e.target.value)} />
          </div>
          <div>
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-muted-foreground">Daily stop $</span>
            <Input className="font-mono tabular-nums" value={dloss} onChange={(e) => setDloss(e.target.value)} />
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={toggleEnabled}
            disabled={isPending || (!isOn && !activateAllowed)}
            aria-busy={isPending}
            title={
              isOn
                ? "Disable — the backend always allows it (even if Enable was rejected)"
                : activateAllowed
                  ? "Enable — the backend will accept it according to the paper/live barrier"
                  : permissionExplain(perm, mode)
            }
          >
            {isPending
              ? "Switching…"
              : isOn
                ? "Disable"
                : activateAllowed
                  ? "Enable"
                  : "Not authorizable"}
          </Button>
          <Button size="sm" variant={s.killed ? "secondary" : "destructive"} onClick={kill}>{s.killed ? "Reset kill" : "Kill"}</Button>
          <Button size="sm" variant="ghost" onClick={saveLimits}>Save limits</Button>
        </div>

        <Accordion>
          <AccordionItem value="diag">
            <AccordionTrigger className="text-xs">Diagnostics</AccordionTrigger>
            <AccordionContent><SkipFunnel strategy={s.strategy} /></AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}