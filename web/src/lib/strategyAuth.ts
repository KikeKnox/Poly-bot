/**
 * Client-side (optional) replica of the `StrategyLimits::can_run` barrier,
 * modeled as `mode_permission_for_toggle` in the backend
 * (`src/api/handlers.rs`). It is only invoked when the backend does not yet
 * expose `mode_permission` in `/ui/strategies` (a historical scenario). As soon
 * as the handler returns the field, the UI uses it directly and this function
 * stays as a fallback.
 *
 * Key difference from a "naive replica" of `can_run`:
 *
 *   `can_run` evaluates (enabled, paper_only) → 3 branches
 *   and returns `Disabled` if `!enabled`. But the operator's "Enable" toggle
 *   is NOT asking to evaluate the current state — it is asking
 *   "will the supervisor accept my `set_enabled(true)`?". And the supervisor,
 *   in `src/strategy/supervisor.rs`, forces `enabled=true` before
 *   consulting the barrier. Returning `"blocked_disabled"` here would
 *   therefore disable the "Enable" button for an `enabled=false` strategy
 *   that the barrier WOULD ACCEPT once `enabled` is switched on — it would
 *   confuse current state with authorization.
 *
 * The correct policy is therefore:
 *   1. `enabled` takes no part (because the supervisor overrides it).
 *   2. `paper_only=true && mode==="live"` → `"blocked_paper_only"`.
 *   3. If any flag is missing (older backend) → `"unknown"` and "Enable"
 *      disabled. If all are present → `"ok"` and "Enable" enabled.
 *
 * Same evaluation order and same mapping as the backend helper; if
 *   either changes, both must change together.
 *
 * Current state vs. authorization: the running state (is the strategy
 * running?) is read by the client from the `s.running` field returned by the
 * backend — it is not inferred from `enabled`. If the backend does not expose it yet,
 * `s.running === undefined` and the UI falls back to the previous behavior
 * (`!s.killed`) only for display, never to decide the toggle.
 */
import type { ModePermission, UiStrategyRow } from "./types";

export type StrategyFlags = Pick<
  UiStrategyRow,
  "enabled" | "paper_only"
>;

/**
 * Barrier decision ASSUMING `enabled=true` (the toggle's view). If
 * any flag is missing (`null`/`undefined`), it returns `"unknown"` and leaves the
 * operator with the "Enable" button disabled so as not to expose an
 * action whose result the client cannot predict.
 */
export function computeModePermission(
  flags: StrategyFlags,
  mode: string,
): ModePermission {
  // If any is missing, we assert nothing. "unknown" is not the same as
  // "blocked_disabled": the client cannot tell "strategy
  // currently disabled" from "I do not know what enabled is".
  if (
    flags.enabled === undefined || flags.enabled === null ||
    flags.paper_only === undefined || flags.paper_only === null
  ) {
    return "unknown";
  }
  // Correct policy: the supervisor's toggle forces `enabled=true`
  // before consulting the barrier, so here we do not look at
  // `flags.enabled` either — only the combination (paper_only, mode).
  if (mode === "live" && flags.paper_only) return "blocked_paper_only";
  return "ok";
}

/** Short label for an effective-permission badge. */
export function permissionLabel(p: ModePermission): string {
  switch (p) {
    case "ok":                   return "Authorized";
    case "blocked":              return "Blocked";
    case "blocked_disabled":     return "Disabled";
    case "blocked_paper_only":   return "Paper only";
    case "unknown":              return "Authorization unknown";
  }
}

/** Explanatory tooltip — the text the operator reads on hover. */
export function permissionExplain(p: ModePermission, mode: string): string {
  switch (p) {
    case "ok":
      return `Can run in ${mode} mode.`;
    case "blocked":
      // Forward-compat: the backend rejected the toggle with a new
      // reason this client does not know yet. It is treated as
      // "not authorized" — "Enable" is never enabled with this value.
      return "The barrier rejected the toggle with a reason this client version does not recognize (newer backend). It is treated as not authorized.";
    case "blocked_disabled":
      // Only reached via the fallback when the flags are missing AND
      // enabled=false is explicitly assumed. In the supervisor's real flow
      // this branch is unreachable (because the supervisor forces
      // enabled=true). It stays here only for backward compatibility with the
      // previous version of the module, where this label had a direction.
      return "enabled = false in config/strategies/<id>.toml — the strategy does not start and cannot be switched on live.";
    case "blocked_paper_only":
      return "paper_only = true: an experiment that never trades with real money. In live the barrier blocks it.";
    case "unknown":
      return "The backend did not expose `enabled`/`paper_only` in /ui/strategies — permission unknown. " +
             "While they are missing, the UI cannot predict whether a toggle would be accepted.";
  }
}

/** Visual variant of `Badge` that maps to the permission. */
export function permissionVariant(p: ModePermission): "default" | "secondary" | "destructive" | "outline" {
  switch (p) {
    case "ok":                   return "default";
    case "blocked":              return "destructive";
    case "blocked_disabled":     return "secondary";
    case "blocked_paper_only":   return "outline";
    case "unknown":              return "outline";
  }
}

/** Compact label for a boolean value that may be absent. */
export function flagLabel(value: boolean | null | undefined): string {
  if (value === null || value === undefined) return "?";
  return value ? "yes" : "no";
}

/**
 * Should the "Enable" button be enabled in the UI?
 *
 * Rule: "Enable" may only be pressed if the client knows the
 * backend will accept it. If the flags are missing (`unknown`), it is disabled.
 *
 * Disable is NEVER blocked from the UI: if the barrier rejects Enable
 * but the strategy is ALREADY running because the operator bypassed it, that
 * operator must also be able to stop it. (The backend also always allows it —
 * see the `StrategySupervisor::set_enabled` `if !enabled`
 * early return.)
 */
export function canActivateFromUi(perm: ModePermission): boolean {
  return perm === "ok";
}

/**
 * Is the strategy currently running? The backend decides it
 * (`s.running`) when available; otherwise, fall back to the
 * previous behavior (`!s.killed`).
 *
 * NOTE: never use this result to authorize the toggle — only to
 * label the card and decide whether the button says "Enable" or
 * "Disable".
 */
export function isCurrentlyRunning(s: UiStrategyRow): boolean {
  if (s.running === true)  return true;
  if (s.running === false) return false;
  // Fallback while the backend does not expose `running`: do not invent
  // state, do not assume. Returns `!killed` to preserve the
  // previous behavior and not break interactions.
  return !s.killed;
}
