export interface UiPosition {
  token_id: string; market_cid: string; slug: string; strategy: string;
  size: string; avg_price: string; cost_basis: string; secs_to_clear: number;
  bail_sold: boolean; outcome_label: string | null;
}
export interface UiStrategyPnl { strategy: string; pnl: string | null; running: boolean }
export interface UiData {
  mode: string; global_killed: boolean; max_total_exposure: string;
  deposits: string; open_cost: string; pnl_total: string; cash: string;
  cash_warning: boolean; wallet_balance: string | null;
  positions_value: string; equity: string;
  by_strategy: UiStrategyPnl[]; positions: UiPosition[];
}

/**
 * Authorization state per strategy, following `StrategyLimits` in the backend
 * (`src/config.rs:769-869`).
 *
 * The flags are `boolean | null`: `null` = "the backend did not expose this
 * field in `/ui/strategies`" (the handler exposes them now; `null` remains for
 * backward compatibility with older backend versions). The UI shows `?` and
 * labels the set as "unknown authorization" in that case. It NEVER claims that
 * the strategy is authorized for live.
 *
 * `mode_permission` is the effective decision the barrier would take if the
 * operator pressed "Activate" (assumes `enabled=true`, as the supervisor does in
 * `set_enabled`). The backend computes it with
 * `mode_permission_for_toggle(paper_only, mode)`
 * — see `src/api/handlers.rs`. If the flags are missing or the backend response
 * is stale, the client recomputes it from the received flags. If the flags are
 * also missing → `"unknown"`.
 *
 * `running` (boolean) is the REAL state of boot + on-demand spawn. The backend
 * exposes it as `s.strategies.has_task(id) && !killed` so the front-end toggle
 * reflects reality, not an assumption. If the backend does not expose it
 * (`undefined`), `isCurrentlyRunning` falls back to `!killed`.
 */
export type ModePermission =
  | "ok"
  // `"blocked"`: forward-compat for the backend — `can_run` rejected with a
  // reason this client version does not know. The UI treats it as "not
  // authorized" and never enables "Activate" with it.
  | "blocked"
  | "blocked_disabled"
  | "blocked_paper_only"
  | "unknown";

export interface UiStrategyRow {
  strategy: string; killed: boolean; max_order_usdc: string;
  max_orders_per_hour: string; daily_loss_stop: string; pnl: string;
  // Real state: is the task alive (boot or on-demand spawn) and not killed?
  // `null` = "the backend did not expose it" (backward compatibility).
  running?: boolean | null;
  // The backend may not expose these yet — `null` = absent.
  enabled?: boolean | null;
  paper_only?: boolean | null;
  /** Effective decision `can_run(mode)` for the strategy, assuming
   *  `enabled=true` (the toggle's view). If the backend computes and exposes
   *  it, the UI uses it as is; otherwise the client recomputes it from the
   *  flags. If the flags are also missing → `"unknown"`. */
  mode_permission?: ModePermission;
}

export interface RiskStrategyRow {
  strategy: string; killed: boolean; kill_reason: string | null;
  max_orders_per_hour: number; orders_sent_last_hour: number;
  max_order_usdc: string; daily_loss_stop: string;
}
export interface RiskData {
  global_killed: boolean; global_reason: string | null;
  exposure_current: string; exposure_max: string; bracket_cap: string;
  strategies: RiskStrategyRow[];
}
export interface AlertEntry { ts_ms: number; severity: "critical" | "warning" | "info"; message: string }
export interface FunnelCounter { name: string; value: number }
export interface ToolRow { tool: string; kind: "scan" | "recorder"; enabled: boolean; running: boolean }
export interface AssetHealth {
  asset: string; sigma_short: string | null; sigma_long: string | null;
  last_price: string; secs_since_update: number;
}
export interface MoneyByStrategy {
  strategy: string; realized: string; unrealized: string; total: string; exposure: string;
}
// `pnl` = equity − cumulative deposits at that instant. Charts must plot `pnl`:
// `equity` includes the +$1000 step of each automatic paper top-up and makes a
// loss look like a gain.
export interface EquityPoint { ts_ms: number; cash: string; positions_value: string; equity: string; deposits: string; pnl: string }
export interface MoneyData {
  deposits: string; initial_capital: string; topups_count: number; topups_total: string;
  cash: string; positions_value: string; equity: string;
  total_return: string; total_return_pct: string | null;
  by_strategy: MoneyByStrategy[]; equity_series: EquityPoint[];
}