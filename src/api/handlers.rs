//! HTTP API handlers.

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::json;

use super::state::ApiState;
use crate::{
    config::BlockReason,
    tools::ToolId,
    types::{Mode, StrategyId},
};

/// Extracts `deposits`/`open_cost`/`pnl_total` from their SQLite queries, logging
/// any failure as an error instead of silencing it with `unwrap_or_default()`.
/// Previously a DB failure in any of the three made `paper_account` and
/// `ui_data` show 0 without warning — in a bot with real capital, a `cash`/
/// `cash_warning` fabricated from silent zeros can hide a negative real
/// cash position.
fn resolve_cash_fields(
    deposits: anyhow::Result<rust_decimal::Decimal>,
    cost: anyhow::Result<rust_decimal::Decimal>,
    pnl_total: anyhow::Result<rust_decimal::Decimal>,
) -> (
    rust_decimal::Decimal,
    rust_decimal::Decimal,
    rust_decimal::Decimal,
) {
    let deposits = deposits.unwrap_or_else(|e| {
        tracing::error!("resolve_cash_fields: deposits_total failed: {e}");
        rust_decimal::Decimal::ZERO
    });
    let cost = cost.unwrap_or_else(|e| {
        tracing::error!("resolve_cash_fields: total_open_cost failed: {e}");
        rust_decimal::Decimal::ZERO
    });
    let pnl_total = pnl_total.unwrap_or_else(|e| {
        tracing::error!("resolve_cash_fields: realized_pnl_total failed: {e}");
        rust_decimal::Decimal::ZERO
    });
    (deposits, cost, pnl_total)
}

#[cfg(test)]
mod cash_fields_tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn ok_passes_values_through_intact() {
        let (d, c, p) = resolve_cash_fields(Ok(dec!(100)), Ok(dec!(40)), Ok(dec!(5)));
        assert_eq!((d, c, p), (dec!(100), dec!(40), dec!(5)));
    }

    #[test]
    fn error_in_any_query_falls_to_zero_without_losing_the_rest() {
        // Previously (unwrap_or_default at the call site) a failure in total_open_cost
        // did not distinguish "no open cost" (real 0) from "the query failed" (fabricated
        // 0) — with deposits/pnl_total OK, cash would end up inflated instead
        // of reflecting that the cost figure is unknown.
        let (d, c, p) =
            resolve_cash_fields(Ok(dec!(100)), Err(anyhow::anyhow!("db lock")), Ok(dec!(5)));
        assert_eq!((d, c, p), (dec!(100), rust_decimal::Decimal::ZERO, dec!(5)));
    }
}

/// "Running" for real requires a live task (boot or on-demand spawn
/// via `StrategySupervisor::has_task`) AND no kill switch — previously
/// `ui_data` only looked at the kill switch, so a strategy with
/// `enabled=false` at startup was reported as "running" as soon as the
/// kill switch was lifted via the API, with no real task behind it.
fn strategy_is_running(task_alive: bool, killed: bool) -> bool {
    task_alive && !killed
}

/// `/paper/deposit` only makes conceptual sense in Paper mode —
/// see the comment next to its only call site in `paper_deposit`.
fn paper_deposit_allowed(mode: crate::types::Mode) -> bool {
    mode == crate::types::Mode::Paper
}

#[cfg(test)]
mod bug_077_tests {
    use super::*;

    #[test]
    fn running_requires_live_task_and_no_kill() {
        assert!(
            strategy_is_running(true, false),
            "live task + no kill = running"
        );
        assert!(
            !strategy_is_running(true, true),
            "live task + killed = not running"
        );
        assert!(
            !strategy_is_running(false, false),
            "no task, even if not killed = not running"
        );
        assert!(
            !strategy_is_running(false, true),
            "no task + killed = not running"
        );
    }
}

/// `/paper/deposit` closed outside Paper mode.
#[cfg(test)]
mod bug_083_tests {
    use super::*;

    #[test]
    fn deposit_only_allowed_in_paper() {
        assert!(paper_deposit_allowed(crate::types::Mode::Paper));
        assert!(!paper_deposit_allowed(crate::types::Mode::Live));
    }
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

pub async fn healthz() -> impl IntoResponse {
    "ok"
}

pub async fn status(State(s): State<ApiState>) -> impl IntoResponse {
    // Tools in a separate section: they do not go through RiskGuard (no kill switch,
    // no rate limits) — mixing them into "risk" would fake a control that does not exist.
    let tools: Vec<_> = ToolId::ALL
        .iter()
        .map(|&id| {
            json!({
                "tool":      id.to_string(),
                "kind":      id.kind().to_string(),
                "enabled":   s.tools.enabled(id),
                "running":   s.tools.is_running(id),
                "is_tool":   true,
            })
        })
        .collect();
    // visibility of the `dead_letter_fills` content — a table
    // nobody reads is as blind as the `debug!` this replaced.
    let dead_letters = s
        .state
        .count_unresolved_dead_letters()
        .await
        .unwrap_or_else(|e| {
            tracing::error!("status: count_unresolved_dead_letters failed: {e}");
            0
        });
    Json(json!({
        "mode":          s.mode.to_string(),
        "global_killed": s.risk.is_killed(),
        "risk":          s.risk.status_snapshot(),
        "tools":         tools,
        "dead_letter_fills_unresolved": dead_letters,
    }))
}

pub async fn metrics_summary(State(s): State<ApiState>) -> impl IntoResponse {
    s.prometheus_handle.render()
}

pub async fn settings(State(s): State<ApiState>) -> impl IntoResponse {
    Json(json!({ "mode": s.mode.to_string() }))
}

// ── Recent fills ─────────────────────────────────────────────────────────────

pub async fn recent(State(s): State<ApiState>) -> impl IntoResponse {
    match s.state.recent_transactions(50, None).await {
        Ok(txs) => Json(json!({ "fills": txs })).into_response(),
        Err(e) => {
            tracing::error!("recent: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

// ── Control ───────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct StrategyToggleBody {
    strategy: String,
    enabled: bool,
}

pub async fn control_strategy(
    State(s): State<ApiState>,
    Json(body): Json<StrategyToggleBody>,
) -> impl IntoResponse {
    // Tools first: same name ("vol_surface"/"ladder_arb") but without
    // going through RiskGuard — the toggle really starts/stops the task instead of
    // a kill switch. If this were not checked first, a tool's name
    // does not parse as a `StrategyId` (`FromStr` is STRICT)
    // and the toggle would end in the 400 below instead of starting the tool.
    if let Ok(id) = body.strategy.parse::<ToolId>() {
        // `set_enabled` is async and fails if the tool is `Stopping`
        // (a stop in progress) or if it lacks `RecorderDeps` — in that case
        // nothing is persisted (same criterion as the barrier
        // below: if the runtime effect did not happen, the config must not
        // lie about what is running).
        if let Err(reason) = s.tools.set_enabled(id, body.enabled).await {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "error": reason })),
            )
                .into_response();
        }
        let name = id.to_string();
        let enabled = body.enabled;
        return match tokio::task::spawn_blocking(move || {
            crate::config::persist_strategy_toggle(&name, enabled)
        })
        .await
        {
            Ok(Ok(())) => {
                // `/ui/tools` and `/status` read
                // `s.tools.enabled(id)`, not the boot `Arc<Config>` — without
                // this, a successfully persisted toggle would not show up there
                // until the next process restart.
                s.tools.set_desired_enabled(id, enabled);
                Json(json!({ "tool": body.strategy, "enabled": body.enabled })).into_response()
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    "control_strategy (tool): runtime toggled, TOML persist failed: {e}"
                );
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
            Err(e) => {
                tracing::error!("control_strategy (tool): {e}");
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        };
    }

    let Ok(id) = body.strategy.parse::<StrategyId>() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "unknown strategy" })),
        )
            .into_response();
    };
    // Previously this only touched `RiskGuard`'s kill switch,
    // which has nothing to "unpause" if the task was never spawned
    // (`enabled=false` in the TOML loaded at startup). `set_enabled` now
    // also spawns the task on demand if needed — see
    // `strategy::StrategySupervisor`.
    // Barrier: in live, a `paper_only` strategy cannot be turned on. The rejection persists NOTHING: if it did, the
    // next startup would inherit an `enabled=true` that the barrier blocks again,
    // leaving the config lying about what is running.
    if let Err(reason) = s.strategies.set_enabled(id, body.enabled) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": format!(
                    "cannot turn on {id} in mode {}: {reason}",
                    s.mode
                ),
            })),
        )
            .into_response();
    }
    let name = id.to_string();
    let enabled = body.enabled;
    match tokio::task::spawn_blocking(move || {
        crate::config::persist_strategy_toggle(&name, enabled)
    })
    .await
    {
        Ok(Ok(())) => {
            Json(json!({ "strategy": body.strategy, "enabled": body.enabled })).into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!("control_strategy: runtime toggled, TOML persist failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
        Err(e) => {
            tracing::error!("control_strategy: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

pub async fn control_kill(State(s): State<ApiState>) -> impl IntoResponse {
    s.risk.kill_global("kill manual since API");
    Json(json!({ "killed": true }))
}

pub async fn control_kill_switches(State(s): State<ApiState>) -> impl IntoResponse {
    s.risk.reset_global_kill();
    Json(json!({ "reset": true }))
}

#[derive(Deserialize)]
pub struct StrategyKillBody {
    strategy: String,
    /// "kill" | "reset"
    action: String,
}

pub async fn control_strategy_kill(
    State(s): State<ApiState>,
    Json(body): Json<StrategyKillBody>,
) -> impl IntoResponse {
    if body.strategy.parse::<ToolId>().is_ok() {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "is a tool, it has no kill switch — use /control/strategy (toggle)",
        }))).into_response();
    }
    let Ok(id) = body.strategy.parse::<StrategyId>() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "unknown strategy" })),
        )
            .into_response();
    };
    match body.action.as_str() {
        "kill" => {
            s.risk.kill_strategy(id, "kill manual since API");
            Json(json!({ "killed": true })).into_response()
        }
        "reset" => {
            s.risk.reset_strategy_kill(id);
            Json(json!({ "reset": true })).into_response()
        }
        _ => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "action must be 'kill' or 'reset'" })),
        )
            .into_response(),
    }
}

// ── Strategy limits ─────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct StrategyLimitsBody {
    strategy: String,
    max_order_usdc: Option<String>,
    max_orders_per_hour: Option<u32>,
    daily_loss_usdc_stop: Option<String>,
}

/// Already-parsed `/control/strategy_limits` limits.
#[derive(Debug)]
struct RequestedLimits {
    max_orders_per_hour: u32,
    max_order_usdc: rust_decimal::Decimal,
    daily_loss_usdc_stop: rust_decimal::Decimal,
}

/// Field absent or cleared from the UI (`""`) = no limit (0). A value
/// that is present but does NOT parse is an operator error, not a 0: returning it as
/// zero turned a typo into "no limit" / "no stop".
fn parse_limit_decimal(value: Option<&str>, field: &str) -> Result<rust_decimal::Decimal, String> {
    match value.map(str::trim) {
        None | Some("") => Ok(rust_decimal::Decimal::ZERO),
        Some(v) => v
            .parse::<rust_decimal::Decimal>()
            .map_err(|_| format!("{field}: \"{v}\" is not a valid number")),
    }
}

/// Parses and validates what arrives through `/control/strategy_limits`.
///
/// Closes the holes in the hot path:
/// - A `.parse().unwrap_or(ZERO)` turned a UI typo into 0, which in
///   `RiskGuard` means "no limit" — and in `daily_loss_usdc_stop`, "no loss
///   stop" (`risk.rs`: `if stop > Decimal::ZERO`). It is now a 400 in
///   both modes: widening limits can never be the effect of a typing error.
/// - `Config::validate_live_loss_stop` only protects STARTUP: this path
///   could disable the stop of a strategy already trading with real money. In
///   live the stop is mandatory and > 0; omitting the field (which also meant 0) is
///   rejected just the same. In paper 0 remains legitimate: there is no real capital.
/// - Same hole in `max_order_usdc` and `max_orders_per_hour`: both mean
///   "limit disabled" at 0 (`executor.rs`: `if max_per_hour > 0 && ...`,
///   `if max_order_usdc > ZERO && ...`), and a POST in live with the field empty/at 0
///   left the per-order size or the rate limit with no brake other than the global
///   exposure cap. Closed the same way as the loss stop: mandatory and > 0
///   in live, 0 remains legitimate in paper.
fn parse_strategy_limits(body: &StrategyLimitsBody, mode: Mode) -> Result<RequestedLimits, String> {
    let max_order_usdc = parse_limit_decimal(body.max_order_usdc.as_deref(), "max_order_usdc")?;
    let daily_loss_usdc_stop =
        parse_limit_decimal(body.daily_loss_usdc_stop.as_deref(), "daily_loss_usdc_stop")?;
    let max_orders_per_hour = body.max_orders_per_hour.unwrap_or(0);

    if mode == Mode::Live && daily_loss_usdc_stop <= rust_decimal::Decimal::ZERO {
        return Err(
            "daily_loss_usdc_stop: in live the daily loss stop is mandatory and must be > 0 \
             — a 0 (or an empty field) disables it and leaves the strategy trading with real money \
             with no automatic brake".to_string()
        );
    }
    if mode == Mode::Live && max_order_usdc <= rust_decimal::Decimal::ZERO {
        return Err(
            "max_order_usdc: in live the per-order cap is mandatory and must be > 0 \
             — a 0 (or an empty field) disables it and leaves each order's size with no brake other \
             than the global exposure cap"
                .to_string(),
        );
    }
    if mode == Mode::Live && max_orders_per_hour == 0 {
        return Err(
            "max_orders_per_hour: in live the hourly rate limit is mandatory and must be > 0 \
             — a 0 (or an absent field) disables it entirely"
                .to_string(),
        );
    }

    Ok(RequestedLimits {
        max_orders_per_hour,
        max_order_usdc,
        daily_loss_usdc_stop,
    })
}

pub async fn control_strategy_limits(
    State(s): State<ApiState>,
    Json(body): Json<StrategyLimitsBody>,
) -> impl IntoResponse {
    if body.strategy.parse::<ToolId>().is_ok() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "is a tool, with no risk limits (it does not go through RiskGuard)",
            })),
        )
            .into_response();
    }
    let Ok(id) = body.strategy.parse::<StrategyId>() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "unknown strategy" })),
        )
            .into_response();
    };

    let limits = match parse_strategy_limits(&body, s.mode) {
        Ok(l) => l,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e }))).into_response(),
    };
    let new_rate = limits.max_orders_per_hour;
    let new_usdc = limits.max_order_usdc;
    let new_dloss = limits.daily_loss_usdc_stop;

    // Apply hot: update RiskGuard without a restart.
    s.risk
        .set_strategy_limits(id, new_rate, new_usdc, new_dloss);

    // Persist to the TOML (always, including 0).
    let name = body.strategy.clone();
    let rate_save = new_rate;
    let usdc_save = new_usdc.to_string();
    let dloss = body
        .daily_loss_usdc_stop
        .as_ref()
        .map(|_| new_dloss.to_string());
    let r = tokio::task::spawn_blocking(move || {
        crate::config::persist_strategy_limits(
            &name,
            Some(usdc_save.as_str()),
            Some(rate_save),
            dloss.as_deref(),
        )
    })
    .await;
    match r {
        Ok(Ok(())) => Json(json!({ "ok": true, "max_orders_per_hour": new_rate, "max_order_usdc": new_usdc.to_string() })).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
        Err(e)     => { tracing::error!("strategy_limits: {e}"); StatusCode::INTERNAL_SERVER_ERROR.into_response() }
    }
}

#[cfg(test)]
mod strategy_limits_for_tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn body(usdc: Option<&str>, dloss: Option<&str>) -> StrategyLimitsBody {
        body_with_rate(usdc, dloss, Some(20))
    }

    fn body_with_rate(
        usdc: Option<&str>,
        dloss: Option<&str>,
        rate: Option<u32>,
    ) -> StrategyLimitsBody {
        StrategyLimitsBody {
            strategy: "arb".into(),
            max_order_usdc: usdc.map(str::to_string),
            max_orders_per_hour: rate,
            daily_loss_usdc_stop: dloss.map(str::to_string),
        }
    }

    #[test]
    fn typo_in_max_order_usdc_is_error_not_zero() {
        let err = parse_strategy_limits(&body(Some("5o"), Some("10")), Mode::Paper)
            .expect_err("an unparseable value must not fall to 0 (= no limit)");
        assert!(
            err.contains("max_order_usdc"),
            "the message must name the field: {err}"
        );
    }

    #[test]
    fn typo_in_loss_stop_is_error_not_zero() {
        let err = parse_strategy_limits(&body(Some("50"), Some("1O")), Mode::Paper)
            .expect_err("a typo must not turn into 'stop disabled'");
        assert!(
            err.contains("daily_loss_usdc_stop"),
            "the message must name the field: {err}"
        );
    }

    /// Review scenario A: process already in live, a POST with the stop at 0
    /// disabled it hot and persisted it.
    #[test]
    fn stop_at_zero_in_live_is_rejected() {
        let err = parse_strategy_limits(&body(Some("50"), Some("0")), Mode::Live)
            .expect_err("in live a stop at 0 leaves the strategy with no brake");
        assert!(
            err.contains("daily_loss_usdc_stop"),
            "the message must name the field: {err}"
        );
    }

    #[test]
    fn absent_stop_in_live_is_rejected() {
        // Omitting the field meant 0 via `unwrap_or`: same hole, no typo.
        parse_strategy_limits(&body(Some("50"), None), Mode::Live)
            .expect_err("in live the stop is mandatory, not optional");
    }

    /// Same hole as the loss stop, now in
    /// `max_order_usdc` — a 0 in live leaves the per-order size with no brake
    /// other than the global exposure cap.
    #[test]
    fn max_order_usdc_at_zero_in_live_is_rejected() {
        let err = parse_strategy_limits(&body(Some("0"), Some("10")), Mode::Live)
            .expect_err("in live a max_order_usdc of 0 leaves the order with no per-order cap");
        assert!(
            err.contains("max_order_usdc"),
            "the message must name the field: {err}"
        );
    }

    #[test]
    fn max_order_usdc_absent_in_live_is_rejected() {
        parse_strategy_limits(&body(None, Some("10")), Mode::Live)
            .expect_err("in live the per-order cap is mandatory, not optional");
    }

    /// Same hole in `max_orders_per_hour` — a 0 (or the absent field, which
    /// is 0 via `unwrap_or`) disables the rate limit entirely.
    #[test]
    fn max_orders_per_hour_at_zero_in_live_is_rejected() {
        let err =
            parse_strategy_limits(&body_with_rate(Some("50"), Some("10"), Some(0)), Mode::Live)
                .expect_err("in live a rate limit of 0 disables it entirely");
        assert!(
            err.contains("max_orders_per_hour"),
            "the message must name the field: {err}"
        );
    }

    #[test]
    fn max_orders_per_hour_absent_in_live_is_rejected() {
        parse_strategy_limits(&body_with_rate(Some("50"), Some("10"), None), Mode::Live)
            .expect_err("in live the rate limit is mandatory, not optional");
    }

    #[test]
    fn max_order_usdc_and_rate_at_zero_in_paper_remain_legitimate() {
        let l = parse_strategy_limits(&body_with_rate(Some("0"), Some("10"), Some(0)), Mode::Paper)
            .expect("in paper there is no real capital: 0 = no limit is legitimate");
        assert_eq!(l.max_order_usdc, rust_decimal::Decimal::ZERO);
        assert_eq!(l.max_orders_per_hour, 0);
    }

    #[test]
    fn stop_at_zero_in_paper_remains_legitimate() {
        let l = parse_strategy_limits(&body(Some("50"), Some("0")), Mode::Paper)
            .expect("in paper there is no real capital: 0 = no stop is legitimate");
        assert_eq!(l.daily_loss_usdc_stop, rust_decimal::Decimal::ZERO);
        assert_eq!(l.max_order_usdc, dec!(50));
    }

    #[test]
    fn valid_limits_in_live_are_accepted() {
        let l = parse_strategy_limits(&body(Some("50"), Some("25")), Mode::Live)
            .expect("stop > 0 in live is exactly what is required");
        assert_eq!(l.daily_loss_usdc_stop, dec!(25));
        assert_eq!(l.max_orders_per_hour, 20);
    }

    #[test]
    fn empty_field_means_no_limit_not_a_parse_error() {
        // The UI sends "" when the operator clears the field.
        let l = parse_strategy_limits(&body(Some(""), Some("")), Mode::Paper)
            .expect("cleared field = no limit, as before");
        assert_eq!(l.max_order_usdc, rust_decimal::Decimal::ZERO);
        assert_eq!(l.daily_loss_usdc_stop, rust_decimal::Decimal::ZERO);
    }
}

// ── Global exposure limit ───────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ExposureLimitBody {
    max_total_exposure_usdc: String,
}

pub async fn control_exposure_limit(
    State(s): State<ApiState>,
    Json(body): Json<ExposureLimitBody>,
) -> impl IntoResponse {
    let v = match crate::config::parse_exposure_limit(&body.max_total_exposure_usdc) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e }))).into_response(),
    };

    // Apply hot: the executor reads max_total_exposure() fresh per intent.
    s.risk.set_max_total_exposure(v);

    // Persist to config/general.toml (always).
    let v_save = v.to_string();
    let r = tokio::task::spawn_blocking(move || crate::config::persist_max_total_exposure(&v_save))
        .await;

    let env_override = crate::config::exposure_env_override_present();
    match r {
        Ok(Ok(())) => Json(json!({
            "max_total_exposure_usdc": v.to_string(),
            "env_override": env_override,
        }))
        .into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
        Err(e) => {
            tracing::error!("exposure_limit: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[derive(Deserialize)]
pub struct BlacklistBody {
    cid: String,
}

pub async fn control_blacklist(
    State(s): State<ApiState>,
    Json(body): Json<BlacklistBody>,
) -> impl IntoResponse {
    match s.state.add_to_blacklist(&body.cid).await {
        Ok(()) => Json(json!({ "blacklisted": body.cid })).into_response(),
        Err(e) => {
            tracing::error!("control_blacklist: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

pub async fn control_shutdown(State(s): State<ApiState>) -> impl IntoResponse {
    let mut lock = s.shutdown_tx.lock().await;
    if let Some(tx) = lock.take() {
        let _ = tx.send(());
        Json(json!({ "shutdown": true }))
    } else {
        Json(json!({ "shutdown": false, "reason": "already in progress" }))
    }
}

// ── Paper / account ────────────────────────────────────────────────────────────

pub async fn paper_account(State(s): State<ApiState>) -> impl IntoResponse {
    let (deposits, cost, pnl_total) = tokio::join!(
        s.state.deposits_total(),
        s.state.total_open_cost(),
        s.state.realized_pnl_total(),
    );
    let (deposits, cost, pnl_total) = resolve_cash_fields(deposits, cost, pnl_total);
    let cash = deposits - cost + pnl_total;
    Json(json!({
        "deposits":     deposits.to_string(),
        "open_cost":    cost.to_string(),
        "pnl_total":    pnl_total.to_string(),
        "cash":         cash.to_string(),
        "cash_warning": cash < rust_decimal::Decimal::ZERO,
    }))
}

#[derive(Deserialize)]
pub struct DepositBody {
    usdc: String,
}

pub async fn paper_deposit(
    State(s): State<ApiState>,
    Json(body): Json<DepositBody>,
) -> impl IntoResponse {
    // `/paper/deposit` MUTATES `deposits_total` (and therefore `cash` and the
    // `cash_warning` alarm) — in live with an empty `api.token` (allowed if
    // `api.bind` is loopback), the `auth.rs` exception for `/paper/*`
    // (meant for diagnostic reads) let this POST through without
    // authentication. "Paper" deposits have no conceptual meaning in live
    // in any mode: rejecting the whole call if the current mode is not Paper
    // closes the hole without touching the auth logic (which still lets the
    // `/paper/account`, `/paper/pnl`, `/paper/positions` reads through — harmless).
    if !paper_deposit_allowed(s.mode) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "paper_deposit only available in paper mode",
            })),
        )
            .into_response();
    }
    let Ok(amount) = body.usdc.parse::<rust_decimal::Decimal>() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "invalid usdc" })),
        )
            .into_response();
    };
    match s.state.add_deposit(amount, "manual deposit via API").await {
        Ok(()) => Json(json!({ "deposited": body.usdc })).into_response(),
        Err(e) => {
            tracing::error!("paper_deposit: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

pub async fn paper_pnl(State(s): State<ApiState>) -> impl IntoResponse {
    match s.state.realized_pnl_by_strategy().await {
        Ok(map) => {
            let by_strat: serde_json::Value = map
                .iter()
                .map(|(k, v)| (k.clone(), json!(v.to_string())))
                .collect::<serde_json::Map<_, _>>()
                .into();
            Json(json!({ "by_strategy": by_strat }))
        }
        Err(e) => {
            tracing::error!("paper_pnl: {e}");
            Json(json!({ "by_strategy": {} }))
        }
    }
}

pub async fn paper_positions(State(s): State<ApiState>) -> impl IntoResponse {
    match s.state.list_open_positions().await {
        Ok(pos) => {
            let items: Vec<_> = pos
                .iter()
                .map(|p| {
                    json!({
                        "token_id":      p.token_id,
                        "market_cid":    p.market_cid,
                        "strategy":      p.strategy,
                        "size":          p.size.to_string(),
                        "avg_price":     p.avg_price.to_string(),
                        "cost_basis":    p.cost_basis.to_string(),
                        "opened_ts_ms":  p.opened_ts_ms,
                        "clear_at":      p.clear_at,
                        "bail_sold":     p.bail_sold,
                        "outcome_label": p.outcome_label,
                        "winner_label":  p.winner_label,
                    })
                })
                .collect();
            Json(json!({ "count": items.len(), "positions": items })).into_response()
        }
        Err(e) => {
            tracing::error!("paper_positions: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

// ── PnL, alerts ─────────────────────────────────────────────────────────────

pub async fn pnl_strat(State(s): State<ApiState>) -> impl IntoResponse {
    match s.state.realized_pnl_by_strategy().await {
        Ok(map) => {
            let items: Vec<_> = map
                .iter()
                .map(|(k, v)| json!({ "strategy": k, "pnl": v.to_string() }))
                .collect();
            Json(json!({ "strategies": items }))
        }
        Err(e) => {
            tracing::error!("pnl_strat: {e}");
            Json(json!({ "strategies": [] }))
        }
    }
}

// ── UI data endpoints ─────────────────────────────────────────────────────────

/// State of all strategies: enabled, killed, limits, barrier flags
/// and PnL.
///
/// Contract:
/// - `enabled` / `paper_only`: exact copy of the in-memory
///   TOML `StrategyLimits` (via
///   `crate::strategy::registry::strategy_limits`). Never absent.
/// - `mode_permission`: barrier decision for the current mode (`s.mode`),
///   computed as `StrategyLimits::can_run(mode)` with `enabled`
///   assumed `true` (because `set_enabled(true)` in
///   `StrategySupervisor::set_enabled` already forces `enabled=true` before
///   consulting the barrier — see `supervisor.rs`). This tells the current
///   state (what the "Enabled" badge shows) apart from the barrier
///   (what decides whether the "Enable" button would end in 200 or 403).
/// - `running`: real state, boot + on-demand spawn (`has_task` and not
///   `killed`). The front-end toggle must reflect it — it used to fall back to
///   `!killed` or `s.enabled`, which confused state with the barrier.
pub async fn ui_strategies(State(s): State<ApiState>) -> impl IntoResponse {
    let pnl_map = s.state.realized_pnl_by_strategy().await.unwrap_or_default();

    let rows: Vec<_> = StrategyId::ALL
        .iter()
        .map(|&id| {
            let name = id.to_string();
            let killed = s.risk.is_strategy_killed(id);
            let pnl = pnl_map
                .get(&*name)
                .map(|d| d.to_string())
                .unwrap_or_default();
            let (rate, usdc) = s.risk.get_strategy_limits(id);
            let dloss = s.risk.get_strategy_daily_loss_stop(id);
            // 0 = no limit → empty field in the UI
            let rate_str = if rate == 0 {
                "".to_string()
            } else {
                rate.to_string()
            };
            let usdc_str = if usdc == rust_decimal::Decimal::ZERO {
                "".to_string()
            } else {
                usdc.to_string()
            };
            let dloss_str = if dloss == rust_decimal::Decimal::ZERO {
                "".to_string()
            } else {
                dloss.to_string()
            };

            // Barrier flags — single source `StrategyLimits` (via registry,
            // the same point `StrategySupervisor::set_enabled` consumes).
            let limits = crate::strategy::registry::strategy_limits(id, &s.config);
            let enabled = limits.enabled;
            let paper_only = limits.paper_only;
            // Effective permission assuming the operator is going to press "Enable"
            // (enabled=true in the barrier). The supervisor does exactly this
            // in `set_enabled(true)`, so here we only mirror its question,
            // not confuse it with the current state.
            let mode_permission = mode_permission_for_toggle(paper_only, s.mode);

            // Real state: task alive from boot or on demand, and not killed.
            let task_alive = s.strategies.has_task(id);
            let running = task_alive && !killed;

            json!({
                "strategy":            name,
                "killed":              killed,
                "running":             running,
                "max_order_usdc":      usdc_str,
                "max_orders_per_hour": rate_str,
                "daily_loss_stop":     dloss_str,
                "pnl":                 pnl,
                "enabled":             enabled,
                "paper_only":          paper_only,
                "mode_permission":     mode_permission,
            })
        })
        .collect();

    Json(json!({ "strategies": rows }))
}

/// Expectancy/edge per strategy over the WHOLE persisted history —
/// economic evaluation of a strategy (idea from Freqtrade's
/// Edge module). The closed trades are rebuilt from inventory
/// event by event inside `StateStore::strategy_edge_stats` — see its
/// docstring for the exact definition and the approximations it makes over
/// the real schema (partial sells aggregated, re-buy = new trade,
/// late standalone sell with cost 0, open trades excluded).
///
/// Response: flat object `{strategy_id: {stats}}` — only strategies with
/// transactions; a strategy without trades does NOT appear (no block
/// of zeros is returned that a reader would confuse with "traded and ended at zero").
/// `Decimal`s go as strings (`EquityPoint`/`ui_money` convention) and
/// absent `Option<Decimal>`s as `null` — never a fabricated 0.
pub async fn ui_strategies_edge(State(s): State<ApiState>) -> impl IntoResponse {
    match s.state.strategy_edge_stats().await {
        Ok(map) => {
            let obj: serde_json::Value = map
                .iter()
                .map(|(name, st)| {
                    (
                        name.clone(),
                        json!({
                            "closed_trades":        st.closed_trades,
                            "open_trades":          st.open_trades,
                            "wins":                 st.wins,
                            "losses":               st.losses,
                            "flat_trades":          st.flat_trades,
                            "win_rate":             st.win_rate.map(|d| d.to_string()),
                            "avg_win":              st.avg_win.map(|d| d.to_string()),
                            "avg_loss":             st.avg_loss.map(|d| d.to_string()),
                            "profit_factor":        st.profit_factor.map(|d| d.to_string()),
                            "expectancy_per_trade": st.expectancy_per_trade.map(|d| d.to_string()),
                            "expectancy_pct":       st.expectancy_pct.map(|d| d.to_string()),
                            "total_pnl":            st.total_pnl.to_string(),
                            "total_cost_deployed":  st.total_cost_deployed.to_string(),
                            "max_exposure":         st.max_exposure.to_string(),
                        }),
                    )
                })
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into();
            Json(obj).into_response()
        }
        Err(e) => {
            tracing::error!("ui_strategies_edge: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// Barrier decision for the "Enable" toggle, expressed as a stable
/// string the client can show directly. Matches the
/// variant of the TypeScript counterpart in `web/src/lib/strategyAuth.ts`.
///
/// This helper does NOT reimplement the barrier policy: it delegates TO
/// `StrategyLimits::can_run` (the single operational source already used by
/// `registry::spawn_all`, `StrategySupervisor::set_enabled` and
/// `main.rs`). It does so by building a `StrategyLimits` with `enabled=true`
/// — exactly what the supervisor does internally in
/// `set_enabled(true)` (it clones the limits and forces `enabled=true`
/// before invoking the barrier). That way, if the
/// barrier adds a new `BlockReason` in the future, this helper
/// picks it up with just a recompile.
///
/// The current `enabled` is deliberately NOT an input because the supervisor
/// always evaluates with `enabled=true` (the toggle's view), not with the
/// TOML flag of the moment. A current `enabled=false` does NOT block the toggle: the
/// barrier decides, just like the supervisor.
///
/// Mapping of `Result<(), BlockReason>` to string:
///
/// | `can_run` with `enabled=true`         | string returned          |
/// |---------------------------------------|--------------------------|
/// | `Ok(())`                              | `"ok"`                   |
/// | `Err(BlockReason::PaperOnly)`         | `"blocked_paper_only"`   |
/// | `Err(BlockReason::Disabled)`          | `"blocked_disabled"` (should not happen: we force `enabled=true`. If it did, fail-safe: we do NOT grant permission.) |
/// | any future `BlockReason`              | `"blocked"` (forward-compat: the REPLICA CANNOT grant a permission that the source denies). |
///
/// Strings emitted: `"ok" | "blocked_paper_only" | "blocked_disabled" |
/// "blocked"`. The last one is new and
/// exists precisely so that adding a reason to `can_run` does not
/// silently turn a rejected toggle into one accepted by the
/// UI.
fn mode_permission_for_toggle(paper_only: bool, mode: Mode) -> &'static str {
    // The operator is going to ask for "Enable", so the current `enabled` is
    // not an input — an `enabled=false` does NOT block the toggle: the
    // barrier decides, just like the supervisor.

    // Toggle view: `StrategyLimits` with `enabled=true` and the barrier
    // flags the TOML loaded at boot. This is the SAME construction that
    // `StrategySupervisor::set_enabled` does right before invoking the
    // barrier, which is why the single source decides here the same as
    // it decides there.
    let view = crate::config::StrategyLimits {
        enabled: true,
        paper_only,
        ..Default::default()
    };
    // `BlockReason` is today a closed enum (no `#[non_exhaustive]`),
    // so `Err(_)` is unreachable and the compiler flags it with an
    // `unreachable_patterns` warning. It is left explicit anyway as a
    // forward-compat contract: if in the future someone adds a new variant
    // to `BlockReason`, the helper treats it as
    // `"blocked"` by default — the REPLICA CANNOT grant a permission
    // that the source denies. Without this `Err(_)` and its `#[allow]`, adding
    // a new variant would force touching this `match` to compile
    // (the opposite of the desired effect: we want a new reason
    // NOT to be silently translated to `"ok"`).
    #[allow(unreachable_patterns)]
    match view.can_run(mode) {
        Ok(()) => "ok",
        Err(BlockReason::PaperOnly) => "blocked_paper_only",
        Err(BlockReason::NoLossStop) => "blocked_no_loss_stop",
        Err(BlockReason::Disabled) => "blocked_disabled", // fail-safe (should not happen: enabled=true)
        Err(_) => "blocked",                              // forward-compat: new unmapped reason
    }
}

/// State of the tools (scan + recorders): no RiskGuard — `kind`
/// ("scan"/"recorder"), `enabled` (desired state persisted in
/// `config/strategies/<tool>.toml`) and `running` (live task), with no limits or
/// kill switch to show.
pub async fn ui_tools(State(s): State<ApiState>) -> impl IntoResponse {
    let rows: Vec<_> = ToolId::ALL
        .iter()
        .map(|&id| {
            json!({
                "tool":    id.to_string(),
                "kind":    id.kind().to_string(),
                "enabled": s.tools.enabled(id),
                "running": s.tools.is_running(id),
            })
        })
        .collect();
    Json(json!({ "tools": rows }))
}

/// Recent alerts (global/strategy kills) for the "Recent
/// alerts" panel of the UI — fixes the historical bug of the nonexistent `/alerts`.
pub async fn ui_alerts(State(s): State<ApiState>) -> impl IntoResponse {
    Json(json!({ "alerts": s.alert_log.recent(20) }))
}

// ── RL predictor (online runtime) ────────────────────────────────────────────

/// GET `/ui/rl_learner` — RL runtime state: loaded policy
/// (hash/version/artifact), online learner (cohort counters,
/// updates applied/rejected, checkpoints) and, if disabled,
/// the explicit reason. Never exposes weights.
pub async fn ui_rl_learner() -> impl IntoResponse {
    let strat = crate::strategy::rl_predictor::rl_strategy_status();
    let learner = crate::rl::learner::learner_control().map(|c| c.status());
    Json(json!({
        "strategy": {
            "active": strat.active,
            "disabled_reason": strat.disabled_reason,
            "policy_sha256": strat.policy_sha256,
            "artifact_dir": strat.artifact_dir,
            "learning_enabled": strat.learning_enabled,
        },
        "learner": learner,
    }))
}

#[derive(Deserialize)]
pub struct RlLearnerControlBody {
    /// "freeze" | "unfreeze" | "rollback".
    action: String,
}

/// POST `/control/rl_learner` — online learner controls (paper-only).
/// `freeze` stops updates while keeping inference; `unfreeze` resumes
/// them; `rollback` publishes the approved base policy as a NEW version
/// (the checkpoint chain moves forward, it never rewrites history).
/// `learning_enabled=false` or a strategy without a policy → 409 (there is no learner
/// to control; the response does not lie about the runtime state).
pub async fn control_rl_learner(Json(body): Json<RlLearnerControlBody>) -> impl IntoResponse {
    let Some(ctl) = crate::rl::learner::learner_control() else {
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "RL learner not registered — strategy without an active policy or learning_enabled=false"
            })),
        )
            .into_response();
    };
    if !ctl.is_alive() {
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "RL learner down — the current policy remains the last \
                          snapshot; restart the strategy to recover the learner"
            })),
        )
            .into_response();
    }
    match body.action.as_str() {
        "freeze" => {
            ctl.set_frozen(true);
            tracing::info!("control_rl_learner: freeze-learning enabled via API");
            Json(json!({ "frozen": true })).into_response()
        }
        "unfreeze" | "resume" => {
            ctl.set_frozen(false);
            tracing::info!("control_rl_learner: freeze-learning disabled via API");
            Json(json!({ "frozen": false })).into_response()
        }
        "rollback" => {
            ctl.request_rollback();
            tracing::warn!("control_rl_learner: rollback to base policy requested via API");
            Json(json!({ "rollback_requested": true })).into_response()
        }
        other => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("unknown action '{other}' — valid: freeze, unfreeze, rollback")
            })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct StrategyFunnelQuery {
    strategy: String,
}

/// Skip funnel per strategy: groups the Prometheus counters
/// `hrf_<prefix>_skip_*_total` according to the `strategy_funnel` table, for the
/// diagnostic panel on why a strategy is not trading.
pub async fn ui_strategy_funnel(
    State(s): State<ApiState>,
    Query(q): Query<StrategyFunnelQuery>,
) -> impl IntoResponse {
    let text = s.prometheus_handle.render();
    let metrics = crate::api::prom_parse::parse_prometheus_text(&text);
    let counters =
        crate::api::strategy_funnel::funnel_for_strategy(&metrics, &q.strategy).unwrap_or_default();
    Json(json!({ "strategy": q.strategy, "counters": counters }))
}

/// Live risk/exposure — Risk page of the dashboard.
pub async fn ui_risk(State(s): State<ApiState>) -> impl IntoResponse {
    // Source of truth: the same BracketLedger that gates buys in the real
    // executor — it includes resting GTDs as well as already-filled positions, unlike
    // `state.total_open_cost()` which only counts SQLite and underestimates
    // exposure while there are live GTDs (normal mode of parity_maker/reward_farm).
    let exposure_current = s.ledger.total_exposure();

    let strategies: Vec<_> = StrategyId::ALL
        .iter()
        .map(|&id| {
            let (rate, usdc) = s.risk.get_strategy_limits(id);
            json!({
                "strategy":               id.to_string(),
                "killed":                 s.risk.is_strategy_killed(id),
                "kill_reason":            s.risk.strategy_kill_reason(id),
                "max_orders_per_hour":    rate,
                "orders_sent_last_hour":  s.risk.orders_sent_last_hour(id),
                "max_order_usdc":         usdc.to_string(),
                "daily_loss_stop":        s.risk.get_strategy_daily_loss_stop(id).to_string(),
            })
        })
        .collect();

    Json(json!({
        "global_killed":    s.risk.is_killed(),
        "global_reason":    s.risk.global_kill_reason(),
        "exposure_current": exposure_current.to_string(),
        "exposure_max":     s.risk.max_total_exposure().to_string(),
        "bracket_cap":      s.risk.max_bracket_exposure().to_string(),
        "strategies":       strategies,
    }))
}

/// Short/long σ per asset — for the future "Market" page.
/// `sigma_short`/`sigma_long` as `None` serialize as `null`, not `0`: the
/// frontend explicitly distinguishes "not enough samples" from a real σ of 0.
pub async fn ui_market_health(State(s): State<ApiState>) -> impl IntoResponse {
    let mut assets: Vec<_> = s
        .vol_rxs
        .iter()
        .map(|(asset, rx)| {
            let snap = rx.borrow();
            json!({
                "asset":              asset,
                "sigma_short":       snap.sigma_short.map(|v| v.to_string()),
                "sigma_long":        snap.sigma_long.map(|v| v.to_string()),
                "last_price":         snap.last_price.to_string(),
                "secs_since_update":  snap.updated_at.elapsed().as_secs(),
            })
        })
        .collect();
    assets.sort_by(|a, b| a["asset"].as_str().cmp(&b["asset"].as_str()));
    Json(json!({ "assets": assets }))
}

/// Main dashboard: equity, cash, open positions, PnL per strategy.
pub async fn ui_data(State(s): State<ApiState>) -> impl IntoResponse {
    let (deposits, cost, pnl_total, pnl_by_strat, open_pos) = tokio::join!(
        s.state.deposits_total(),
        s.state.total_open_cost(),
        s.state.realized_pnl_total(),
        s.state.realized_pnl_by_strategy(),
        s.state.list_open_positions(),
    );
    let (deposits, cost, pnl_total) = resolve_cash_fields(deposits, cost, pnl_total);
    let cash = deposits - cost + pnl_total;

    // Visible if running (not killed) or if it has historical PnL — previously only
    // strategies with realized PnL were listed and a running strategy
    // with no fills yet was invisible on the dashboard (calibration scenario).
    // `pnl: null` = running with no trades; the front-end renders it as "no fills".
    let pnl_map = pnl_by_strat.unwrap_or_default();
    let by_strat: Vec<_> = StrategyId::ALL
        .iter()
        .filter_map(|&id| {
            let name = id.to_string();
            let pnl = pnl_map.get(&*name);
            // "running" used to only check whether `spawn_strategy!`
            // had started it at boot — a strategy turned on dynamically
            // via `control_strategy` (without having had `enabled=true` at startup)
            // was reported as "not running" despite already having a live task.
            // `has_task` reflects the real state: boot OR on-demand spawn.
            let task_alive = s.strategies.has_task(id);
            let running = strategy_is_running(task_alive, s.risk.is_strategy_killed(id));
            if pnl.is_none() && !running {
                return None;
            }
            Some(json!({
                "strategy": name,
                "pnl":      pnl.map(|v| v.to_string()),
                "running":  running,
            }))
        })
        .collect();

    let open_pos = open_pos.unwrap_or_default();

    // Real equity = cash + mark-to-market value of open positions — it does
    // not fire network I/O, it only reads the cache that accounting::mark_price::run refreshes in the
    // background (see the module doc-comment). Positions with no price
    // cached yet use avg_price as the reference (what was paid).
    let positions_value: rust_decimal::Decimal = open_pos
        .iter()
        .map(|p| s.mark_price.get(&p.token_id).unwrap_or(p.avg_price) * p.size)
        .sum();
    let equity = cash + positions_value;

    let positions: Vec<_> = open_pos
        .into_iter()
        .map(|p| {
            let secs = p.clear_at - chrono::Utc::now().timestamp();
            json!({
                "token_id":      p.token_id,
                "market_cid":    p.market_cid,
                "slug":          p.slug,
                "strategy":      p.strategy,
                "size":          p.size.to_string(),
                "avg_price":     p.avg_price.to_string(),
                "cost_basis":    p.cost_basis.to_string(),
                "secs_to_clear": secs,
                "bail_sold":     p.bail_sold,
                "outcome_label": p.outcome_label,
            })
        })
        .collect();

    let wallet_balance = *s.wallet_balance.borrow();
    Json(json!({
        "mode":           s.mode.to_string(),
        "global_killed":  s.risk.is_killed(),
        "max_total_exposure": s.risk.max_total_exposure().to_string(),
        "deposits":       deposits.to_string(),
        "open_cost":      cost.to_string(),
        "pnl_total":      pnl_total.to_string(),
        "cash":           cash.to_string(),
        "cash_warning":   cash < rust_decimal::Decimal::ZERO,
        "positions_value": positions_value.to_string(),
        "equity":         equity.to_string(),
        "wallet_balance": wallet_balance.map(|b| b.to_string()),
        "by_strategy":    by_strat,
        "positions":      positions,
    }))
}

/// Aggregate statistics per strategy over the WHOLE history.
/// The Metrics tab used to compute them from /ui/history (last 200 tx) and with
/// more volume it diverged from the Dashboard and the DB.
pub async fn ui_metrics_stats(State(s): State<ApiState>) -> impl IntoResponse {
    let (stats, volume) = tokio::join!(s.state.strategy_stats(), s.state.total_volume_usdc());
    match stats {
        Ok(rows) => {
            let by_strategy: Vec<_> = rows
                .into_iter()
                .map(|(strategy, n, pnl, wins, won, lost, best, worst)| {
                    json!({
                        "strategy": strategy,
                        "n":        n,
                        "pnl":      pnl.to_string(),
                        "wins":     wins,
                        "won":      won.to_string(),
                        "lost":     lost.to_string(),
                        "best":     best.to_string(),
                        "worst":    worst.to_string(),
                    })
                })
                .collect();
            Json(json!({
                "by_strategy":  by_strategy,
                "total_volume": volume.unwrap_or_default().to_string(),
            }))
            .into_response()
        }
        Err(e) => {
            tracing::error!("ui_metrics_stats: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[derive(Deserialize)]
pub struct HistoryQuery {
    limit: Option<u32>,
    before_id: Option<i64>,
}

/// History of closed trades, paginated. Previously limited to 200 rows —
/// before_id (id of the last row of the previous page) allows loading more
/// backwards; default limit 500.
pub async fn ui_history(
    State(s): State<ApiState>,
    Query(q): Query<HistoryQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(500);
    match s.state.recent_transactions(limit, q.before_id).await {
        Ok(txs) => Json(json!({ "trades": txs })).into_response(),
        Err(e) => {
            tracing::error!("ui_history: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// Recent transactions (fills + deposits).
pub async fn ui_transactions(State(s): State<ApiState>) -> impl IntoResponse {
    match s.state.recent_transactions(100, None).await {
        Ok(txs) => Json(json!({ "transactions": txs })).into_response(),
        Err(e) => {
            tracing::error!("ui_transactions: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// Full series of closed PnL (the whole history) for the curve
/// of the Metrics tab. Columnar format (ts_ms[]/pnl[]) to minimize
/// the payload with thousands of points.
pub async fn ui_pnl_series(State(s): State<ApiState>) -> impl IntoResponse {
    match s.state.pnl_series().await {
        Ok(points) => {
            let (ts_ms, pnl): (Vec<i64>, Vec<String>) =
                points.into_iter().map(|(t, p)| (t, p.to_string())).unzip();
            Json(json!({ "ts_ms": ts_ms, "pnl": pnl })).into_response()
        }
        Err(e) => {
            tracing::error!("ui_pnl_series: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// PnL analytics for the Metrics tab: maximum drawdown, hourly
/// distribution and average markout in bps (at 5s) per strategy. Combines
/// `max_drawdown`, `pnl_by_hour_of_day` and one call to
/// `markouts_for_strategy` per `StrategyId::ALL`.
pub async fn ui_pnl_analytics(State(s): State<ApiState>) -> impl IntoResponse {
    let (dd, by_hour) = tokio::join!(s.state.max_drawdown(), s.state.pnl_by_hour_of_day());

    let dd = dd.unwrap_or_else(|e| {
        tracing::error!("ui_pnl_analytics: max_drawdown failed: {e}");
        rust_decimal::Decimal::ZERO
    });
    let by_hour = by_hour.unwrap_or_else(|e| {
        tracing::error!("ui_pnl_analytics: pnl_by_hour_of_day failed: {e}");
        Vec::new()
    });
    let by_hour_json: Vec<_> = by_hour
        .into_iter()
        .map(|(h, pnl)| json!({ "hour": h, "pnl": pnl.to_string() }))
        .collect();

    let mut markout_rows = Vec::new();
    for &id in StrategyId::ALL {
        let name = id.to_string();
        match s.state.markouts_for_strategy(&name).await {
            Ok(rows) => {
                // markout in bps relative to the fill: (mid_5s - fill_price) / fill_price * 10_000
                // Only rows with mid_5s = Some(...) enter the average; the divisor
                // and the reported `n` must be the count of THAT subset, not
                // of the unfiltered `rows` (fills without a book snapshot at +5s are
                // plausible in illiquid brackets and must not bias the average
                // towards 0 nor pass off a sample of 0 rows as a "neutral markout").
                let with_mid5s: Vec<rust_decimal::Decimal> = rows
                    .iter()
                    // `fill_price = 0` is unlikely with real prices but not
                    // impossible with corrupt data — Decimal panics when
                    // dividing by zero, so it is discarded before the computation.
                    .filter(|(_, fill, ..)| !fill.is_zero())
                    .filter_map(|(_, fill, mid_5s, _)| {
                        mid_5s.map(|m| (m - fill) / fill * rust_decimal::Decimal::from(10_000))
                    })
                    .collect();
                if !with_mid5s.is_empty() {
                    let n = with_mid5s.len();
                    let avg_bps = with_mid5s.iter().sum::<rust_decimal::Decimal>()
                        / rust_decimal::Decimal::from(n as i64);
                    markout_rows.push(json!({ "strategy": name, "n": n, "markout_5s_avg_bps": avg_bps.round_dp(1).to_string() }));
                }
            }
            Err(e) => tracing::error!("ui_pnl_analytics: markouts_for_strategy({name}) failed: {e}"),
        }
    }

    Json(json!({
        "max_drawdown":        dd.to_string(),
        "by_hour":             by_hour_json,
        "markout_by_strategy": markout_rows,
    }))
}

/// Money tab: real equity (cash + mark-to-market value of open
/// positions), its historical curve, and total PnL (realized + unrealized) and
/// exposure broken down per strategy — realized only counts what is already
/// closed, so a strategy with large open positions but no
/// sells yet would look "inactive" if the unrealized were not summed.
pub async fn ui_money(State(s): State<ApiState>) -> impl IntoResponse {
    let (deposits, cost, pnl_total, pnl_realized_by_strat, open_pos, series, topups) = tokio::join!(
        s.state.deposits_total(),
        s.state.total_open_cost(),
        s.state.realized_pnl_total(),
        s.state.realized_pnl_by_strategy(),
        s.state.list_open_positions(),
        s.state.equity_series(),
        s.state.topups_total(),
    );
    let (deposits, cost, pnl_total) = resolve_cash_fields(deposits, cost, pnl_total);
    let cash = deposits - cost + pnl_total;
    let (topups_count, topups_sum) = topups.unwrap_or_else(|e| {
        tracing::error!("ui_money: topups_total failed: {e}");
        (0, rust_decimal::Decimal::ZERO)
    });
    // The automatic `paper_topup` reloads DO count as contributed capital —
    // they simulate exactly what would happen in real trading if the balance ran short: the user
    // would have to put in more real USDC for the strategy to keep
    // trading at the observed pace. Excluding them underestimated the capital
    // this strategy really needs to sustain itself. `initial_capital`
    // remains only as an informative reference (the configured initial
    // contribution, before any reload) — the return is computed
    // over the full `deposits`, not over this value.
    let initial_capital = deposits - topups_sum;

    let open_pos = open_pos.unwrap_or_else(|e| {
        tracing::error!("ui_money: list_open_positions failed: {e}");
        Vec::new()
    });
    let positions_value: rust_decimal::Decimal = open_pos
        .iter()
        .map(|p| s.mark_price.get(&p.token_id).unwrap_or(p.avg_price) * p.size)
        .sum();
    let equity = cash + positions_value;

    // Unrealized + exposure per strategy, in a single pass over the
    // open positions (same source as positions_value above).
    let mut unrealized_by_strat: std::collections::HashMap<String, rust_decimal::Decimal> =
        std::collections::HashMap::new();
    let mut exposure_by_strat: std::collections::HashMap<String, rust_decimal::Decimal> =
        std::collections::HashMap::new();
    for p in &open_pos {
        let mark = s.mark_price.get(&p.token_id).unwrap_or(p.avg_price);
        *unrealized_by_strat.entry(p.strategy.clone()).or_default() += mark * p.size - p.cost_basis;
        *exposure_by_strat.entry(p.strategy.clone()).or_default() += p.cost_basis;
    }

    let realized_by_strat = pnl_realized_by_strat.unwrap_or_else(|e| {
        tracing::error!("ui_money: realized_pnl_by_strategy failed: {e}");
        std::collections::HashMap::new()
    });

    let mut strategy_names: Vec<String> = realized_by_strat
        .keys()
        .chain(unrealized_by_strat.keys())
        .chain(exposure_by_strat.keys())
        .cloned()
        .collect();
    strategy_names.sort_unstable();
    strategy_names.dedup();

    let by_strategy: Vec<_> = strategy_names
        .into_iter()
        .map(|name| {
            let realized = realized_by_strat.get(&name).copied().unwrap_or_default();
            let unrealized = unrealized_by_strat.get(&name).copied().unwrap_or_default();
            let exposure = exposure_by_strat.get(&name).copied().unwrap_or_default();
            json!({
                "strategy":   name,
                "realized":   realized.to_string(),
                "unrealized": unrealized.to_string(),
                "total":      (realized + unrealized).to_string(),
                "exposure":   exposure.to_string(),
            })
        })
        .collect();

    let series = series.unwrap_or_else(|e| {
        tracing::error!("ui_money: equity_series failed: {e}");
        Vec::new()
    });
    // the series exposes `pnl` (equity − cumulative deposits at that
    // instant) in addition to gross equity. Charts must plot `pnl`: raw
    // equity adds a vertical step of +$1000 at every `paper_topup`,
    // which made a −24% run read as if it were +280%.
    let equity_series_json: Vec<_> = series
        .into_iter()
        .map(|p| {
            json!({
                "ts_ms":           p.ts_ms,
                "cash":            p.cash.to_string(),
                "positions_value": p.positions_value.to_string(),
                "equity":          p.equity.to_string(),
                "deposits":        p.deposits.to_string(),
                "pnl":             p.pnl.to_string(),
            })
        })
        .collect();

    // Total return = current equity − everything ever deposited, automatic
    // reloads included — they are capital that in real trading the user would have had to
    // contribute, not a gift from the simulator (see the comment on
    // `initial_capital` above). The % is computed over the full `deposits`
    // for the same reason: it is "how does the capital that was really
    // needed to sustain this perform?", not an artificially inflated version
    // hiding how much capital the follower's pace really demands.
    let total_return = equity - deposits;
    let total_return_pct = if deposits > rust_decimal::Decimal::ZERO {
        Some(
            (total_return / deposits * rust_decimal::Decimal::from(100))
                .round_dp(2)
                .to_string(),
        )
    } else {
        None
    };

    Json(json!({
        "deposits":         deposits.to_string(),
        "initial_capital":  initial_capital.to_string(),
        "topups_count":     topups_count,
        "topups_total":     topups_sum.to_string(),
        "cash":             cash.to_string(),
        "positions_value":  positions_value.to_string(),
        "equity":           equity.to_string(),
        "total_return":     total_return.to_string(),
        "total_return_pct": total_return_pct,
        "by_strategy":      by_strategy,
        "equity_series":    equity_series_json,
    }))
}

// ── Tests of `mode_permission_for_toggle` ─────────────────────────────────
//
// This helper is the ONLY new logic that `ui_strategies` adds. It lives
// here as a pure function so tests can cover it without having to
// build a full `ApiState` (some fields require a real token,
// metric handle, live task, etc.). Its contract:
//
//   mode_permission_for_toggle(paper_only, mode)
//   == "ok"                           if the barrier does NOT block a hypothetical
//                                      `set_enabled(true)` in mode `mode`.
//   == "blocked_paper_only"           if mode == Live && paper_only.
//
// The current `enabled` is not an input: the supervisor's toggle
// (`StrategySupervisor::set_enabled(_, true)`) already forces `enabled=true`
// before consulting `can_run`. The UI thereby tells the "current state"
// (the "Enabled" badge, based on `running`) apart from the barrier (the
// mode badge, based on `mode_permission`).
#[cfg(test)]
mod mode_permission_for_toggle_tests {
    use super::mode_permission_for_toggle;
    use crate::types::Mode;

    #[test]
    fn paper_never_blocks() {
        assert_eq!(mode_permission_for_toggle(false, Mode::Paper), "ok");
        // paper_only strategies can run in paper, though never in live
        assert_eq!(mode_permission_for_toggle(true, Mode::Paper), "ok");
    }

    #[test]
    fn live_without_paper_only_is_ok() {
        assert_eq!(mode_permission_for_toggle(false, Mode::Live), "ok");
    }

    #[test]
    fn live_with_paper_only_blocks() {
        assert_eq!(
            mode_permission_for_toggle(true, Mode::Live),
            "blocked_paper_only"
        );
    }

    /// Cross coverage against `StrategyLimits::can_run` — the single
    /// operational source, applied to a `StrategyLimits` with `enabled=true`
    /// (the scenario the supervisor actually evaluates in `set_enabled(true)`).
    #[test]
    fn matches_can_run_assuming_enabled_true() {
        use crate::config::BlockReason;

        for paper_only in [true, false] {
            for mode in [Mode::Paper, Mode::Live] {
                let limits = crate::config::StrategyLimits {
                    enabled: true, // what the supervisor forces
                    paper_only,
                    ..Default::default()
                };
                let expected = match limits.can_run(mode) {
                    Ok(()) => "ok",
                    Err(BlockReason::PaperOnly) => "blocked_paper_only",
                    Err(BlockReason::ModelPredictionTxMissing) => {
                        unreachable!("can_run does not check runtime channels")
                    }
                    Err(BlockReason::NoLossStop) => {
                        unreachable!("the default daily_loss_usdc_stop is > 0")
                    }
                    Err(BlockReason::Disabled) => unreachable!(
                        "we force enabled=true so the barrier does not return Disabled here"
                    ),
                };
                let got = mode_permission_for_toggle(paper_only, mode);
                assert_eq!(
                    got, expected,
                    "mismatch for paper_only={paper_only}, mode={mode:?}"
                );
            }
        }
    }
}
