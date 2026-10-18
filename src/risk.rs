//! Risk guard — merge of the global `RiskGuard` and per-strategy kills.
//!
//! V3 consolidates the two separate modules (`risk.rs` + `strategy_kills.rs`) into
//! one. The public API is the same one the `Executor` consumes:
//!
//! - `is_killed()` / `kill_global(reason)` — global kill switch
//! - `is_strategy_killed(id)` / `kill_strategy(id, reason)` — per-strategy kill
//! - `check_order_limit(id) -> bool` — per-strategy rate limit (does not record)
//! - `record_order(id)` — records an emitted order (after the real submit)
//! - `record_bracket_loss(cid)` — per-bracket entry cooldown after a losing exit
//! - `in_entry_cooldown(id, cid, now_ms) -> bool` — ENTRY gate by cooldown
//! - `paper_disable_kill_switch_active() -> bool` — in paper, kills disabled
//!
//! Cooldowns after losses (equivalent to Freqtrade's `CooldownPeriod` + `StoplossGuard`):
//! they only block ENTRIES (Buy); sells/cancels/unwinds are never
//! affected. Everything at 0 in `RiskConfig` = disabled.

use crate::time::Instant;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rust_decimal::Decimal;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::watch;

use crate::{
    config::RiskConfig,
    types::{Mode, StrategyId},
};

// ── Global kill switch ───────────────────────────────────────────────────────

struct GlobalKill {
    killed: AtomicBool,
    reason: Mutex<Option<String>>,
    /// Allows subscribing to the global kill (watch) so tasks stop instantly.
    tx: watch::Sender<bool>,
}

// ── Per-strategy rate limit ────────────────────────────────────────────────

/// REAL sliding window of orders per hour — it used to be a
/// TUMBLING window (`rollover_if_needed` reset `count=0` every 3600s since
/// startup), which allowed up to 2× `max_per_hour` straddling the
/// reset boundary. Now the timestamp of each order is stored and everything
/// older than `now − 1h` is pruned on every query/record.
struct OrderWindow {
    timestamps: VecDeque<Instant>,
}

impl OrderWindow {
    fn new() -> Self {
        Self {
            timestamps: VecDeque::new(),
        }
    }

    /// Checks whether one more order can be emitted within the hourly limit.
    /// Does not modify the counter (check / record separation).
    /// 0 = no limit (the UI uses it to indicate an empty field).
    fn can_send(&mut self, max_per_hour: u32) -> bool {
        if max_per_hour == 0 {
            return true;
        }
        self.prune();
        (self.timestamps.len() as u32) < max_per_hour
    }

    /// Records an emitted order.
    fn record(&mut self) {
        self.prune();
        self.timestamps.push_back(Instant::now());
    }

    /// Discards timestamps older than one hour — the window is sliding, not tumbling.
    fn prune(&mut self) {
        let now = Instant::now();
        while let Some(&front) = self.timestamps.front() {
            if now.duration_since(front).as_secs() >= 3600 {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }
    }
}

struct StrategyState {
    killed: AtomicBool,
    kill_reason: Mutex<Option<String>>,
    window: Mutex<OrderWindow>,
    kill_tx: watch::Sender<bool>,
    session_pnl: Mutex<Decimal>,
    /// Baseline of the PER-STRATEGY daily loss stop — same pattern
    /// as `RiskGuard::loss_stop_baseline` but at strategy level.
    /// Without this, `reset_strategy_kill` only cleared the `killed` flag and the
    /// next `record_strategy_pnl` killed it again instantly because
    /// `session_pnl` was still below −stop.
    loss_stop_baseline: Mutex<Decimal>,
    /// Streak of consecutive losing exits (StoplossGuard). Fed by
    /// `record_strategy_pnl`: pnl<0 adds, pnl>0 resets, pnl==0 leaves it
    /// untouched. It is not reset when the cooldown fires nor on the daily reset —
    /// the streak only dies with a win.
    loss_strikes: Mutex<u32>,
    /// End of the entry cooldown caused by a loss streak (epoch ms;
    /// 0 = no active cooldown). Only affects entries (Buy).
    entry_cooldown_until_ms: Mutex<i64>,
}

impl StrategyState {
    fn new() -> Arc<Self> {
        let (tx, _) = watch::channel(false);
        Arc::new(Self {
            killed: AtomicBool::new(false),
            kill_reason: Mutex::new(None),
            window: Mutex::new(OrderWindow::new()),
            kill_tx: tx,
            session_pnl: Mutex::new(Decimal::ZERO),
            loss_stop_baseline: Mutex::new(Decimal::ZERO),
            loss_strikes: Mutex::new(0),
            entry_cooldown_until_ms: Mutex::new(0),
        })
    }
}

// ── RiskGuard ────────────────────────────────────────────────────────────────

/// Risk guard shared between the Executor and the Supervisor.
///
/// Sync access (parking_lot) so it can be called from any context.
pub struct RiskGuard {
    global: GlobalKill,
    strategies: Mutex<HashMap<StrategyId, Arc<StrategyState>>>,
    mode: Mode,
    cfg: RiskConfig,
    /// Realized PnL of the current day (for the daily loss stop).
    session_pnl: Mutex<Decimal>,
    /// Baseline of the daily loss stop after a manual re-arm — the stop
    /// evaluates `session_pnl − baseline` so it does not re-kill over the old loss.
    loss_stop_baseline: Mutex<Decimal>,
    /// Next UTC midnight at which session_pnl resets.
    daily_reset_at: Mutex<DateTime<Utc>>,
    /// Per-strategy operating limits, hot-modifiable.
    /// (max_orders_per_hour, max_order_usdc)
    runtime_limits: Mutex<HashMap<StrategyId, (u32, Decimal)>>,
    /// daily_loss_usdc_stop per strategy (0 = no limit).
    strategy_loss_stops: Mutex<HashMap<StrategyId, Decimal>>,
    /// Global exposure limit, hot-modifiable (UI/Telegram).
    /// Initialized from `cfg.max_total_exposure_usdc`; the executor reads it
    /// fresh on every intent (`max_total_exposure()`, see executor.rs:348).
    runtime_max_total_exposure: Mutex<Decimal>,
    /// Per-strategy global exposure reserve, ONLY for strategies
    /// with an almost deterministic outcome (PCS-2 today). `exposure > max` rejections
    /// lost entries with an almost certain outcome that
    /// compete for the same global cap with directional strategies — the
    /// carve-out adds to the global cap exclusively for the reserved strategy.
    /// Do NOT extend to directional strategies (their risk is not deterministic).
    strategy_exposure_carveout: Mutex<HashMap<StrategyId, Decimal>>,
    /// Optional handle to the StateStore to persist kill switches and
    /// session PnL on every mutation. `None` (default) = entirely in
    /// memory, behavior identical to before this fix (used by
    /// all tests and offline tools that do not call
    /// `attach_persistence`). Only `main.rs` wires it up at real startup.
    persist: Mutex<Option<Arc<crate::state::StateStore>>>,
    /// In-memory alert log (UI) — injected after `new()` because `new()`
    /// returns `Arc<Self>`, so a mutable setter does not work; `OnceLock`
    /// allows setting it once from `main.rs` without blocking reads.
    alert_log: std::sync::OnceLock<Arc<crate::api::alert_log::AlertLog>>,
    /// Per-bracket entry cooldown (CooldownPeriod): `market_cid` → end
    /// of the cooldown in epoch ms. Fed by `record_bracket_loss` from the
    /// executor sell path (the only point with cid + sign of the
    /// loss within the immediate-exit path). Volatile like the rest
    /// of the cooldown state: a restart clears it — deliberate, it is a short
    /// pause of minutes, not a durable protection like the kills.
    bracket_cooldowns: Mutex<HashMap<String, i64>>,
}

impl RiskGuard {
    pub fn new(cfg: RiskConfig, mode: Mode) -> Arc<Self> {
        let (global_tx, _) = watch::channel(false);
        let runtime_max_total_exposure = Mutex::new(cfg.max_total_exposure_usdc);
        Arc::new(Self {
            global: GlobalKill {
                killed: AtomicBool::new(false),
                reason: Mutex::new(None),
                tx: global_tx,
            },
            strategies: Mutex::new(HashMap::new()),
            mode,
            cfg,
            session_pnl: Mutex::new(Decimal::ZERO),
            loss_stop_baseline: Mutex::new(Decimal::ZERO),
            daily_reset_at: Mutex::new(next_midnight_utc()),
            runtime_limits: Mutex::new(HashMap::new()),
            strategy_loss_stops: Mutex::new(HashMap::new()),
            runtime_max_total_exposure,
            strategy_exposure_carveout: Mutex::new(HashMap::new()),
            persist: Mutex::new(None),
            alert_log: std::sync::OnceLock::new(),
            bracket_cooldowns: Mutex::new(HashMap::new()),
        })
    }

    // ── Persistence of kill switches + session PnL ──────────────────────────

    /// Wires disk persistence — call ONCE from `main.rs` after
    /// restoring the saved state. Without calling this (tests, offline
    /// tools) the guard keeps working entirely in memory, as before.
    pub fn attach_persistence(&self, state: Arc<crate::state::StateStore>) {
        *self.persist.lock() = Some(state);
    }

    /// Restores persisted kill switches and session PnL after a restart.
    /// Fail-open was the bug: `RiskGuard::new` always started clean without
    /// reading anything from disk, so an active kill switch or daily-loss-stop
    /// was "forgotten" on every restart. `main.rs` decides what to do if
    /// READING `state.load_risk_state()` fails (fail-closed recommended).
    pub fn restore_persisted(&self, rows: Vec<crate::state::RiskStateRow>) {
        // In paper with `paper_disable_kill_switch` persisted kills are NOT
        // restored. Without this the `check_daily_loss_stop` fix is useless in practice: the run
        // of 3-sep left `risk_state` with `global killed=1` saved, so the
        // next startup brought the bot back already dead and not trading —
        // the two failures cancelled out and "fixing" only one changed nothing.
        // PnL and the baseline ARE restored (metrics and UI stay correct), and
        // MANUAL kills via Telegram/API keep working hot.
        let ignore_kills = self.paper_disable_kill_switch_active();
        if ignore_kills && rows.iter().any(|r| r.killed) {
            tracing::warn!(
                "risk: persisted kills IGNORED (paper + paper_disable_kill_switch) — \
                 paper mode exists to collect data, not to stop by itself"
            );
        }
        for row in rows {
            let killed = row.killed && !ignore_kills;
            if row.scope == "global" {
                *self.session_pnl.lock() = row.session_pnl;
                *self.loss_stop_baseline.lock() = row.loss_stop_baseline;
                if killed {
                    self.global.killed.store(true, Ordering::Release);
                    *self.global.reason.lock() = row.reason.clone();
                    let _ = self.global.tx.send(true);
                }
                tracing::info!(
                    killed, pnl = %row.session_pnl,
                    "risk: global state restored after restart"
                );
            } else {
                match row.scope.parse::<StrategyId>() {
                    Ok(id) => {
                        let st = self.get_or_init(id);
                        *st.session_pnl.lock() = row.session_pnl;
                        *st.loss_stop_baseline.lock() = row.loss_stop_baseline;
                        if killed {
                            st.killed.store(true, Ordering::Release);
                            *st.kill_reason.lock() = row.reason.clone();
                            let _ = st.kill_tx.send(true);
                        }
                        tracing::info!(
                            strategy = %id, killed, pnl = %row.session_pnl,
                            "risk: strategy state restored after restart"
                        );
                    }
                    Err(()) => {
                        tracing::warn!(scope = %row.scope, "risk: unknown scope in risk_state — ignored")
                    }
                }
            }
        }
    }

    /// Dumps the current state of a scope ("global" or the name of a
    /// strategy) to disk in a separate task — all `RiskGuard`
    /// mutations are synchronous (parking_lot), so the real persistence
    /// (sqlx, async) cannot block the caller. No-op if `attach_persistence`
    /// was not called (tests, offline tools).
    fn persist_scope(
        &self,
        scope: String,
        killed: bool,
        reason: Option<String>,
        session_pnl: Decimal,
        baseline: Decimal,
    ) {
        let Some(state) = self.persist.lock().clone() else {
            return;
        };
        tokio::spawn(async move {
            if let Err(e) = state
                .save_risk_state(&scope, killed, reason.as_deref(), session_pnl, baseline)
                .await
            {
                tracing::error!(scope, "risk: failed persisting risk_state: {e}");
            }
        });
    }

    /// Injects the UI alert log (once, from `main.rs` after
    /// building the `RiskGuard`). Later calls are ignored.
    pub fn attach_alert_log(&self, log: Arc<crate::api::alert_log::AlertLog>) {
        let _ = self.alert_log.set(log);
    }

    /// Pure logging side effect for the UI — does not affect the kill logic.
    fn log_alert(&self, severity: &'static str, message: String) {
        if let Some(log) = self.alert_log.get() {
            log.push(severity, message);
        }
    }

    /// Initializes or updates a strategy's hot limits.
    pub fn set_strategy_limits(
        &self,
        id: StrategyId,
        max_per_hour: u32,
        max_order_usdc: Decimal,
        daily_loss_stop: Decimal,
    ) {
        self.runtime_limits
            .lock()
            .insert(id, (max_per_hour, max_order_usdc));
        self.strategy_loss_stops.lock().insert(id, daily_loss_stop);
    }

    /// Returns (max_orders_per_hour, max_order_usdc) with defaults if not configured.
    pub fn get_strategy_limits(&self, id: StrategyId) -> (u32, Decimal) {
        self.runtime_limits
            .lock()
            .get(&id)
            .copied()
            .unwrap_or((20, rust_decimal_macros::dec!(50)))
    }

    /// Hot daily loss stop for a strategy. 0 = no stop.
    pub fn get_strategy_daily_loss_stop(&self, id: StrategyId) -> Decimal {
        self.strategy_loss_stops
            .lock()
            .get(&id)
            .copied()
            .unwrap_or(Decimal::ZERO)
    }

    // ── Global kill switch ───────────────────────────────────────────────────

    pub fn is_killed(&self) -> bool {
        self.global.killed.load(Ordering::Relaxed)
    }

    pub fn kill_global(&self, reason: impl Into<String>) {
        let reason = reason.into();
        // Alert message computed before the move of `reason` (the next line
        // consumes it inside the Mutex) so the existing logic is not altered.
        let alert_msg = format!("global kill: {reason}");
        tracing::warn!(reason, "risk: GLOBAL KILL activated");
        *self.global.reason.lock() = Some(reason.clone());
        self.global.killed.store(true, Ordering::Release);
        let _ = self.global.tx.send(true);
        self.persist_scope(
            "global".into(),
            true,
            Some(reason),
            self.session_pnl(),
            *self.loss_stop_baseline.lock(),
        );
        self.log_alert("critical", alert_msg);
    }

    pub fn reset_global_kill(&self) {
        // Effective re-arm of the daily loss stop. Without this, after a manual reset
        // the PnL was still below −stop and the next record_pnl re-killed
        // instantly (the comment of check_daily_loss promised otherwise).
        // The loss already accumulated becomes the baseline; the stop goes back to
        // protecting against the NEXT losses.
        *self.loss_stop_baseline.lock() = *self.session_pnl.lock();
        self.global.killed.store(false, Ordering::Release);
        *self.global.reason.lock() = None;
        let _ = self.global.tx.send(false);
        self.persist_scope(
            "global".into(),
            false,
            None,
            self.session_pnl(),
            *self.loss_stop_baseline.lock(),
        );
    }

    /// Subscription to the global kill — the executor uses it to cancel
    /// resting GTDs as soon as the kill is activated, without waiting for the next intent.
    pub fn subscribe_global_kill(&self) -> watch::Receiver<bool> {
        self.global.tx.subscribe()
    }

    pub fn global_kill_reason(&self) -> Option<String> {
        self.global.reason.lock().clone()
    }

    // ── Per-strategy kill switch ───────────────────────────────────────────

    pub fn is_strategy_killed(&self, id: StrategyId) -> bool {
        self.strategies
            .lock()
            .get(&id)
            .map(|s| s.killed.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// UI-Risk: reason of the strategy's last kill (read-only, no side effects).
    pub fn strategy_kill_reason(&self, id: StrategyId) -> Option<String> {
        self.get_or_init(id).kill_reason.lock().clone()
    }

    pub fn kill_strategy(&self, id: StrategyId, reason: impl Into<String>) {
        let reason = reason.into();
        // Alert message computed before the move of `reason` (the next line
        // consumes it inside the Mutex) so the existing logic is not altered.
        let alert_msg = format!("kill {id}: {reason}");
        tracing::warn!(%id, reason, "risk: strategy kill activated");
        let st = self.get_or_init(id);
        *st.kill_reason.lock() = Some(reason.clone());
        st.killed.store(true, Ordering::Release);
        let _ = st.kill_tx.send(true);
        let pnl = *st.session_pnl.lock();
        let baseline = *st.loss_stop_baseline.lock();
        self.persist_scope(id.to_string(), true, Some(reason), pnl, baseline);
        self.log_alert("warning", alert_msg);
    }

    pub fn reset_strategy_kill(&self, id: StrategyId) {
        if let Some(st) = self.strategies.lock().get(&id) {
            // Replicate the baseline of the daily loss stop also on
            // the per-strategy path — before, `reset_strategy_kill` only
            // cleared the `killed` flag without touching `session_pnl`/`baseline`, so
            // the next fill could kill it again instantly because
            // the PnL was still below −stop.
            *st.loss_stop_baseline.lock() = *st.session_pnl.lock();
            st.killed.store(false, Ordering::Release);
            *st.kill_reason.lock() = None;
            let _ = st.kill_tx.send(false);
            let pnl = *st.session_pnl.lock();
            let baseline = *st.loss_stop_baseline.lock();
            self.persist_scope(id.to_string(), false, None, pnl, baseline);
        }
    }

    pub fn subscribe_strategy_kill(&self, id: StrategyId) -> watch::Receiver<bool> {
        self.get_or_init(id).kill_tx.subscribe()
    }

    // ── Per-strategy rate limit ────────────────────────────────────────────

    /// Checks whether one more order can be emitted for this strategy.
    /// Does NOT record the order (check / record separation, avoids TOCTOU).
    ///
    /// The caller (Executor) passes the `max_per_hour` extracted from the
    /// strategy config so RiskGuard does not need to know the individual parameters.
    pub fn check_order_limit(&self, id: StrategyId, max_per_hour: u32) -> bool {
        let st = self.get_or_init(id);
        let result = st.window.lock().can_send(max_per_hour);
        result
    }

    /// Records that an order was emitted for this strategy.
    /// Call only when the order was sent to the router.
    pub fn record_order(&self, id: StrategyId) {
        let st = self.get_or_init(id);
        st.window.lock().record();
    }

    /// UI-Risk: orders consumed in the current sliding window (read-only).
    pub fn orders_sent_last_hour(&self, id: StrategyId) -> u32 {
        let st = self.get_or_init(id);
        let mut window = st.window.lock();
        window.prune();
        window.timestamps.len() as u32
    }

    /// Global exposure limit — read fresh on every intent (executor.rs:348),
    /// so `set_max_total_exposure` is enough for the hot change to apply
    /// immediately. Caching this value in the executor in the future would break that.
    pub fn max_total_exposure(&self) -> Decimal {
        *self.runtime_max_total_exposure.lock()
    }

    /// Hot-changes the global exposure limit (UI/Telegram).
    pub fn set_max_total_exposure(&self, v: Decimal) {
        *self.runtime_max_total_exposure.lock() = v;
        tracing::info!(new = %v, "risk: max_total_exposure_usdc changed at runtime");
    }

    /// Sets (or removes with `Decimal::ZERO`) the global exposure reserve
    /// of a strategy. Reserved ONLY for strategies with an almost
    /// deterministic outcome (PCS-2); do not extend to directional ones — see the field doc.
    pub fn set_strategy_exposure_carveout(&self, id: StrategyId, v: Decimal) {
        self.strategy_exposure_carveout.lock().insert(id, v);
        tracing::info!(strategy = %id, carveout = %v, "risk: exposure carve-out set");
    }

    /// Effective global exposure cap for `id`: the global one plus its reserve
    /// (0 if it has none configured). The executor uses it instead of
    /// plain `max_total_exposure()` so as not to penalize the reserved
    /// strategy for the exposure of the others.
    pub fn effective_max_total_exposure(&self, id: StrategyId) -> Decimal {
        let carveout = self
            .strategy_exposure_carveout
            .lock()
            .get(&id)
            .copied()
            .unwrap_or(Decimal::ZERO);
        self.max_total_exposure() + carveout
    }

    /// Net directional cap per bracket (0 = disabled). See `coordination::BracketLedger`.
    pub fn max_bracket_exposure(&self) -> Decimal {
        self.cfg.max_bracket_usdc
    }

    // ── Cooldowns after losses (CooldownPeriod + StoplossGuard) ───────────────

    /// Per-bracket cooldown (Freqtrade's CooldownPeriod): marks `cid` after a
    /// losing exit to block NEW entries (Buy) on that bracket
    /// for `entry_cooldown_secs`. Called by the executor on the
    /// sell path when it detects the losing exit (`handle_sell`, before emitting
    /// the fill — see executor.rs). `entry_cooldown_secs == 0` = disabled
    /// (no-op). Only affects entries; sells/cancels are never blocked.
    pub fn record_bracket_loss(&self, cid: &str) {
        if self.cfg.entry_cooldown_secs == 0 {
            return;
        }
        let now_ms = crate::time::now().timestamp_millis();
        let until = now_ms + self.cfg.entry_cooldown_secs as i64 * 1000;
        let mut map = self.bracket_cooldowns.lock();
        // Opportunistic pruning: up/down brackets rotate every few minutes and
        // their cids are not reused — without this the mapping would accumulate
        // dead entries during the whole session.
        map.retain(|_, u| *u > now_ms);
        map.insert(cid.to_string(), until);
        tracing::warn!(
            cid,
            cooldown_secs = self.cfg.entry_cooldown_secs,
            "risk: entry cooldown on bracket after losing exit"
        );
    }

    /// `true` if an ENTRY (Buy) of `id` on `cid` must be blocked by
    /// cooldown. Two independent scopes:
    ///
    /// - `bracket` — recent losing exit on that `cid` (CooldownPeriod):
    ///   blocks ALL strategies on that bracket, not only the one that
    ///   lost (the bracket is the designated market, like Freqtrade's pair).
    /// - `strategy` — streak of `loss_strike` consecutive losing exits
    ///   of `id` (StoplossGuard): blocks only that strategy's entries.
    ///
    /// `now_ms` is epoch ms (`crate::time::now().timestamp_millis()` in production;
    /// injected in tests to control expiry). Each `true`
    /// increments `hrf_risk_cooldown_blocked_total{strategy,scope}` — the
    /// executor queries this gate once per Buy, so one block = +1.
    pub fn in_entry_cooldown(&self, id: StrategyId, cid: &str, now_ms: i64) -> bool {
        if self.cfg.entry_cooldown_secs > 0 {
            let blocked = self
                .bracket_cooldowns
                .lock()
                .get(cid)
                .is_some_and(|&until| now_ms < until);
            if blocked {
                metrics::counter!(
                    "hrf_risk_cooldown_blocked_total",
                    "strategy" => id.to_string(),
                    "scope" => "bracket",
                )
                .increment(1);
                return true;
            }
        }
        if self.cfg.loss_strike > 0 && self.cfg.loss_strike_cooldown_secs > 0 {
            let until = self
                .strategies
                .lock()
                .get(&id)
                .map(|s| *s.entry_cooldown_until_ms.lock())
                .unwrap_or(0);
            if until > 0 && now_ms < until {
                metrics::counter!(
                    "hrf_risk_cooldown_blocked_total",
                    "strategy" => id.to_string(),
                    "scope" => "strategy",
                )
                .increment(1);
                return true;
            }
        }
        false
    }

    // ── Session PnL / daily loss stop ────────────────────────────────────────

    pub fn record_pnl(&self, pnl: Decimal) {
        // Accumulate BEFORE evaluating the stop — before, `check_daily_loss()`
        // was called with the PnL PRIOR to this fill, so the event that should have
        // triggered the stop was only detected on the NEXT call to
        // `record_pnl` (compare `record_strategy_pnl`, which did accumulate
        // first). If the daily reset fell exactly on this call, the PnL of
        // this fill already belongs to the new day — do not evaluate the stop of the closing day.
        if self.maybe_daily_reset() {
            *self.session_pnl.lock() += pnl;
            self.persist_scope(
                "global".into(),
                self.is_killed(),
                self.global_kill_reason(),
                self.session_pnl(),
                *self.loss_stop_baseline.lock(),
            );
            return;
        }
        *self.session_pnl.lock() += pnl;
        self.check_daily_loss_stop();
        self.persist_scope(
            "global".into(),
            self.is_killed(),
            self.global_kill_reason(),
            self.session_pnl(),
            *self.loss_stop_baseline.lock(),
        );
    }

    /// Accumulates PnL for a specific strategy and fires its kill if it exceeds the stop.
    /// Call together with `record_pnl` so both counters stay in sync.
    pub fn record_strategy_pnl(&self, id: StrategyId, pnl: Decimal) {
        let st = self.get_or_init(id);
        let current = {
            let mut g = st.session_pnl.lock();
            *g += pnl;
            *g
        };
        let baseline = *st.loss_stop_baseline.lock();
        // Evaluate the stop FROM the baseline of the strategy's last re-arm
        // (same pattern as the global stop) — without this, after a
        // `reset_strategy_kill`, the old loss kept counting and could
        // re-kill on the first new fill.
        let adjusted = current - baseline;
        let stop = self
            .strategy_loss_stops
            .lock()
            .get(&id)
            .copied()
            .unwrap_or(Decimal::ZERO);
        if stop > Decimal::ZERO && adjusted < -stop && !self.paper_disable_kill_switch_active() {
            self.kill_strategy(id, format!(
                "strategy daily loss stop: PnL {adjusted:.2} < −{stop:.2} (since the last re-arm)"
            ));
        }
        // Cooldown by loss streak (Freqtrade's StoplossGuard): each
        // exit with pnl<0 adds to the streak; on reaching `loss_strike` the strategy's
        // ENTRIES are paused for `loss_strike_cooldown_secs`.
        // A winning exit (pnl>0) resets the streak; pnl==0 leaves it untouched —
        // a tie neither extends nor breaks it. The streak is NOT reset when
        // the cooldown fires: it stays alive until a win, so each new
        // consecutive loss re-arms the pause. This hook runs AFTER the
        // daily loss stop: the cooldown is the soft brake; the kill, the hard one.
        if self.cfg.loss_strike > 0 {
            if pnl < Decimal::ZERO {
                let strikes = {
                    let mut s = st.loss_strikes.lock();
                    *s += 1;
                    *s
                };
                if strikes >= self.cfg.loss_strike {
                    let until = crate::time::now().timestamp_millis()
                        + self.cfg.loss_strike_cooldown_secs as i64 * 1000;
                    *st.entry_cooldown_until_ms.lock() = until;
                    tracing::warn!(
                        strategy = %id, strikes, pnl = %pnl,
                        cooldown_secs = self.cfg.loss_strike_cooldown_secs,
                        "risk: entry cooldown due to loss streak"
                    );
                }
            } else if pnl > Decimal::ZERO {
                *st.loss_strikes.lock() = 0;
            }
        }
        let killed = st.killed.load(Ordering::Relaxed);
        let reason = st.kill_reason.lock().clone();
        self.persist_scope(id.to_string(), killed, reason, current, baseline);
    }

    pub fn session_pnl(&self) -> Decimal {
        *self.session_pnl.lock()
    }

    /// Automatic reset at the start of each UTC day. Returns `true` if it reset
    /// in this call (the PnL that triggered it belongs to the NEW day, must not
    /// be evaluated against the stop of the day that just closed).
    fn maybe_daily_reset(&self) -> bool {
        let mut reset_at = self.daily_reset_at.lock();
        if crate::time::now() >= *reset_at {
            *self.session_pnl.lock() = Decimal::ZERO;
            *self.loss_stop_baseline.lock() = Decimal::ZERO;
            // Also reset the per-strategy counters.
            for st in self.strategies.lock().values() {
                *st.session_pnl.lock() = Decimal::ZERO;
                *st.loss_stop_baseline.lock() = Decimal::ZERO;
            }
            *reset_at = next_midnight_utc();
            tracing::info!("risk: daily PnL reset (new session)");
            true
        } else {
            false
        }
    }

    /// Evaluates the daily loss stop with the ALREADY accumulated PnL of
    /// the current fill (separate from `maybe_daily_reset`, which only handles the
    /// rollover). The stop evaluates the loss FROM the baseline of the
    /// last manual re-arm (0 if there was none).
    fn check_daily_loss_stop(&self) {
        let pnl = *self.session_pnl.lock() - *self.loss_stop_baseline.lock();
        let stop = self.cfg.daily_loss_usdc_stop;
        // The AUTOMATIC loss stop must not kill in paper — it is
        // the same criterion `record_strategy_pnl` already applied,
        // which did consult `paper_disable_kill_switch_active()`. The asymmetry
        // made the global stop kill the whole bot in paper (stopping data
        // collection, which is the purpose of the mode) while the per-strategy
        // stops stayed silent. PnL keeps being
        // accumulated for UI/metrics; MANUAL kills (Telegram/API) and the
        // `fill_sink` integrity kill do not go through here and stay active.
        if stop > Decimal::ZERO
            && pnl < -stop
            && !self.is_killed()
            && !self.paper_disable_kill_switch_active()
        {
            self.kill_global(format!(
                "daily loss stop: day PnL {pnl:.2} < −{stop:.2} (since the last re-arm)"
            ));
        }
    }

    // ── Paper mode / kill switches ───────────────────────────────────────────

    /// `true` if we are in paper and paper_disable_kill_switch is active.
    /// In that scenario sells are not blocked by global kill switches.
    pub fn paper_disable_kill_switch_active(&self) -> bool {
        self.mode == Mode::Paper && self.cfg.paper_disable_kill_switch
    }

    /// `true` if Polymarket redeems automatically (magic/pro account).
    /// The orphan checker only marks in the DB without sending an on-chain transaction.
    pub fn auto_redeem(&self) -> bool {
        self.cfg.auto_redeem
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    // ── State (for UI/Telegram) ──────────────────────────────────────────────

    /// Summary for the UI.
    pub fn status_snapshot(&self) -> RiskSnapshot {
        RiskSnapshot {
            global_killed: self.is_killed(),
            global_reason: self.global_kill_reason(),
            session_pnl: self.session_pnl(),
            daily_loss_stop: self.cfg.daily_loss_usdc_stop,
            max_total_exposure: self.max_total_exposure(),
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn get_or_init(&self, id: StrategyId) -> Arc<StrategyState> {
        self.strategies
            .lock()
            .entry(id)
            .or_insert_with(StrategyState::new)
            .clone()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn next_midnight_utc() -> DateTime<Utc> {
    let tomorrow = (crate::time::now() + chrono::Duration::days(1)).date_naive();
    tomorrow.and_hms_opt(0, 0, 0).unwrap().and_utc()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn guard_with_stop(stop: Decimal) -> Arc<RiskGuard> {
        let cfg = RiskConfig {
            daily_loss_usdc_stop: stop,
            paper_disable_kill_switch: false,
            ..RiskConfig::default()
        };
        RiskGuard::new(cfg, Mode::Live)
    }

    /// The manual re-arm sets the baseline — the old loss does not re-kill,
    /// but a NEW loss exceeding the stop since the re-arm does.
    ///
    /// With the fix, a SINGLE call to `record_pnl` that crosses the stop
    /// kills on the spot (before TWO calls were needed: the first only
    /// accumulated, the second —with any pnl, even 0— was the one that evaluated
    /// the stop with the PnL of the earlier call).
    #[test]
    fn daily_loss_rearm_sets_baseline() {
        let g = guard_with_stop(dec!(10));
        g.record_pnl(dec!(-15));
        assert!(
            g.is_killed(),
            "a single call crossing the stop must kill immediately"
        );

        g.reset_global_kill();
        assert!(!g.is_killed());
        g.record_pnl(dec!(-1));
        assert!(
            !g.is_killed(),
            "after the re-arm, the old loss must not re-kill"
        );

        g.record_pnl(dec!(-10));
        assert!(
            g.is_killed(),
            "new loss > stop since the re-arm must kill"
        );
    }

    /// Explicit, isolated scenario — before the fix, `record_pnl`
    /// evaluated the stop WITH THE PRIOR PNL (check before accumulating), so
    /// the event that should have triggered the stop was only detected on the
    /// NEXT call. A single catastrophic loss must kill in its own call.
    #[test]
    fn record_pnl_evaluates_stop_with_already_accumulated_pnl() {
        let g = guard_with_stop(dec!(10));
        assert!(!g.is_killed());
        g.record_pnl(dec!(-20)); // crosses the stop in this very call
        assert!(
            g.is_killed(),
            "the stop must be evaluated AFTER accumulating, not before"
        );
    }

    /// The GLOBAL daily loss stop ignored `paper_disable_kill_switch`,
    /// while the PER-STRATEGY stop (`record_strategy_pnl`) did
    /// honor it. In a paper run (27-aug→3-sep-2026) that left
    /// `risk_state` with `global killed=1` ("daily loss stop: day PnL
    /// -11.68 < −10.00") and ALL strategies at `killed=0` — the bot stopped
    /// trading 1h before dying, ruining the data collection that is
    /// precisely the purpose of paper mode.
    #[test]
    fn daily_loss_stop_global_does_not_kill_in_paper() {
        let cfg = RiskConfig {
            daily_loss_usdc_stop: dec!(10),
            paper_disable_kill_switch: true,
            ..RiskConfig::default()
        };
        let g = RiskGuard::new(cfg, Mode::Paper);
        g.record_pnl(dec!(-50)); // 5× the stop
        assert!(
            !g.is_killed(),
            "in paper with paper_disable_kill_switch the global stop must NOT kill"
        );
        assert_eq!(
            g.session_pnl(),
            dec!(-50),
            "PnL must keep being accounted for the UI/metrics"
        );
    }

    /// The counterpart of the previous one: in LIVE the global stop still kills
    /// as before, and in paper with the lever disabled too.
    #[test]
    fn daily_loss_stop_global_keeps_killing_in_live_and_in_paper_without_lever() {
        let g = guard_with_stop(dec!(10)); // Mode::Live, lever set to false
        g.record_pnl(dec!(-50));
        assert!(g.is_killed(), "in live the global stop must kill");

        let cfg = RiskConfig {
            daily_loss_usdc_stop: dec!(10),
            paper_disable_kill_switch: false,
            ..RiskConfig::default()
        };
        let g2 = RiskGuard::new(cfg, Mode::Paper);
        g2.record_pnl(dec!(-50));
        assert!(
            g2.is_killed(),
            "in paper with the lever set to false the global stop must kill"
        );
    }

    /// The MANUAL kill (Telegram/API) and the integrity kill (`fill_sink`) must
    /// keep working in paper — `paper_disable_kill_switch` only silences
    /// the AUTOMATIC loss stop, not the emergency stop.
    #[test]
    fn manual_kill_still_works_in_paper() {
        let cfg = RiskConfig {
            daily_loss_usdc_stop: dec!(10),
            paper_disable_kill_switch: true,
            ..RiskConfig::default()
        };
        let g = RiskGuard::new(cfg, Mode::Paper);
        g.kill_global("manual kill via Telegram".to_string());
        assert!(
            g.is_killed(),
            "the manual kill must always work, also in paper"
        );
    }

    /// The real run left `risk_state` with
    /// `global killed=1` on disk. If `restore_persisted` brings it back as is,
    /// the `check_daily_loss_stop` fix is useless — the bot starts already dead.
    /// In paper with the lever active, PnL is restored but the kill is NOT.
    #[test]
    fn restore_persisted_ignores_kills_in_paper() {
        use crate::state::RiskStateRow;
        let cfg = RiskConfig {
            daily_loss_usdc_stop: dec!(10),
            paper_disable_kill_switch: true,
            ..RiskConfig::default()
        };
        let g = RiskGuard::new(cfg, Mode::Paper);
        g.restore_persisted(vec![RiskStateRow {
            scope: "global".into(),
            killed: true,
            reason: Some("daily loss stop: day PnL -11.68 < −10.00".into()),
            session_pnl: dec!(-168.09),
            loss_stop_baseline: Decimal::ZERO,
        }]);
        assert!(
            !g.is_killed(),
            "in paper the persisted kill must NOT come back to life"
        );
        assert_eq!(
            g.session_pnl(),
            dec!(-168.09),
            "PnL is restored (metrics/UI)"
        );
    }

    /// Counterpart: in LIVE the persisted kill must come back to life as before
    /// — a daily-loss-stop cannot be forgotten by a restart.
    #[test]
    fn restore_persisted_keeps_kills_in_live() {
        use crate::state::RiskStateRow;
        let g = guard_with_stop(dec!(10)); // Mode::Live
        g.restore_persisted(vec![RiskStateRow {
            scope: "global".into(),
            killed: true,
            reason: Some("daily loss stop".into()),
            session_pnl: dec!(-50),
            loss_stop_baseline: Decimal::ZERO,
        }]);
        assert!(
            g.is_killed(),
            "in live the persisted kill must stay alive"
        );
    }

    /// Indefinite paper: the PER-STRATEGY daily loss stop must
    /// be silenced in paper with `paper_disable_kill_switch=true`, just like
    /// the global stop. The `ModelBtc5m` strategy runs an indefinite-duration
    /// paper experiment over `crypto-updown-short` with stop
    /// `daily_loss_usdc_stop=10` and max exposure 10 USDC — the bot must NOT
    /// die when crossing the strategy's stop (collecting the
    /// model's data is the purpose of paper mode).
    ///
    /// Mirror of the global test `daily_loss_stop_global_does_not_kill_in_paper`
    /// applied to the `record_strategy_pnl` path — without this test, a
    /// regression that removed the `paper_disable_kill_switch_active()` guard
    /// in the per-strategy branch would pass the existing tests and kill the
    /// strategy on the experiment's first losing fill.
    #[test]
    fn daily_loss_stop_strategy_does_not_kill_in_paper() {
        let cfg = RiskConfig {
            paper_disable_kill_switch: true,
            ..RiskConfig::default()
        };
        let g = RiskGuard::new(cfg, Mode::Paper);
        let id = crate::types::StrategyId::ModelBtc5m;
        // ModelBtc5m strategy stop = 10 USDC (config/strategies/model_btc5m.toml).
        g.set_strategy_limits(id, 60, dec!(5), dec!(10));

        // Accumulated PnL that crosses the stop by a wide margin (5× stop).
        g.record_strategy_pnl(id, dec!(-50));
        assert!(
            !g.is_strategy_killed(id),
            "in paper with paper_disable_kill_switch the strategy stop must NOT kill"
        );
        assert!(
            !g.is_killed(),
            "the global kill must not fire either"
        );
        // The strategy's kill_reason stays empty (kill_strategy was not invoked).
        assert!(
            g.strategy_kill_reason(id).is_none(),
            "in paper the strategy stop must NOT leave a recorded kill reason"
        );
    }

    /// Counterpart: the per-strategy stop DOES kill in live (paper is not the
    /// only mode) — guard that the previous test does not silence the stop
    /// universally.
    #[test]
    fn daily_loss_stop_strategy_keeps_killing_in_live() {
        let g = RiskGuard::new(RiskConfig::default(), Mode::Live);
        let id = crate::types::StrategyId::ModelBtc5m;
        g.set_strategy_limits(id, 60, dec!(5), dec!(10));

        g.record_strategy_pnl(id, dec!(-50));
        assert!(
            g.is_strategy_killed(id),
            "in live the per-strategy stop must kill even if the strategy \
             is paper-only by construction (the paper guard is by mode, \
             not by strategy)"
        );
    }

    /// `reset_strategy_kill` must set a per-strategy baseline
    /// (like the global-level one) — without this, the next
    /// `record_strategy_pnl` re-killed instantly because the accumulated PnL
    /// was still below −stop.
    #[test]
    fn reset_strategy_kill_sets_baseline_per_strategy() {
        let g = RiskGuard::new(RiskConfig::default(), Mode::Live);
        let id = crate::types::StrategyId::NearCertain;
        g.set_strategy_limits(id, 20, dec!(50), dec!(10));

        g.record_strategy_pnl(id, dec!(-15));
        assert!(
            g.is_strategy_killed(id),
            "loss > stop must kill the strategy"
        );

        g.reset_strategy_kill(id);
        assert!(!g.is_strategy_killed(id));

        g.record_strategy_pnl(id, dec!(-1));
        assert!(
            !g.is_strategy_killed(id),
            "after the re-arm, the old loss must not re-kill"
        );

        g.record_strategy_pnl(id, dec!(-10));
        assert!(
            g.is_strategy_killed(id),
            "new loss > stop since the re-arm must kill"
        );
    }

    /// The rate-limit window must be SLIDING, not tumbling.
    /// With a tumbling window, exhausting the quota right before the "rollover" and
    /// exhausting it again right after allows 2× `max_per_hour` in a
    /// real time window much shorter than an hour.
    #[test]
    fn order_window_is_sliding_not_tumbling() {
        let mut w = OrderWindow::new();
        // Quota of 3: the first 3 pass, the 4th does not.
        assert!(w.can_send(3));
        w.record();
        assert!(w.can_send(3));
        w.record();
        assert!(w.can_send(3));
        w.record();
        assert!(
            !w.can_send(3),
            "quota exhausted within the same hour must block"
        );
    }

    /// `restore_persisted` must recreate the global kill AND the session
    /// PnL from the persisted rows — without this, a restart with the bot
    /// killed started "fail-open" (clean, no kill).
    #[test]
    fn restore_persisted_recovers_global_kill_and_pnl() {
        let g = RiskGuard::new(RiskConfig::default(), Mode::Live);
        assert!(!g.is_killed());

        g.restore_persisted(vec![crate::state::RiskStateRow {
            scope: "global".into(),
            killed: true,
            reason: Some("daily loss stop".into()),
            session_pnl: dec!(-42),
            loss_stop_baseline: dec!(-10),
        }]);

        assert!(
            g.is_killed(),
            "the global kill must be restored after restart"
        );
        assert_eq!(g.global_kill_reason(), Some("daily loss stop".into()));
        assert_eq!(g.session_pnl(), dec!(-42));
    }

    /// Same but for the per-strategy kill.
    #[test]
    fn restore_persisted_recovers_strategy_kill() {
        let g = RiskGuard::new(RiskConfig::default(), Mode::Live);
        let id = crate::types::StrategyId::ParityMaker;
        assert!(!g.is_strategy_killed(id));

        g.restore_persisted(vec![crate::state::RiskStateRow {
            scope: id.to_string(),
            killed: true,
            reason: Some("daily loss stop strategy".into()),
            session_pnl: dec!(-7),
            loss_stop_baseline: Decimal::ZERO,
        }]);

        assert!(
            g.is_strategy_killed(id),
            "the strategy kill must be restored after restart"
        );
    }

    /// The carve-out must only add to the global cap for the
    /// reserved strategy (PCS-2); any other strategy keeps seeing the cap as is.
    #[test]
    fn risk_carveout_adds_to_global_cap() {
        let cfg = RiskConfig {
            max_total_exposure_usdc: dec!(100),
            ..RiskConfig::default()
        };
        let g = RiskGuard::new(cfg, Mode::Paper);

        g.set_strategy_exposure_carveout(crate::types::StrategyId::PostCloseSweep, dec!(30));

        assert_eq!(
            g.effective_max_total_exposure(crate::types::StrategyId::PostCloseSweep),
            dec!(130),
            "PCS must see the global + its reserve"
        );
        assert_eq!(
            g.effective_max_total_exposure(crate::types::StrategyId::NearCertain),
            dec!(100),
            "another strategy without carve-out must see only the global cap"
        );
    }

    /// UI-Risk: exposes the reason of the last per-strategy kill (read-only).
    #[test]
    fn strategy_kill_reason_returns_reason_of_last_kill() {
        let g = RiskGuard::new(RiskConfig::default(), Mode::Live);
        let id = crate::types::StrategyId::NearCertain;
        assert_eq!(g.strategy_kill_reason(id), None);
        g.kill_strategy(id, "test");
        assert_eq!(g.strategy_kill_reason(id), Some("test".to_string()));
    }

    /// UI-Risk: exposes the current consumption of the rate-limit window (read-only).
    #[test]
    fn orders_sent_last_hour_counts_recorded_orders() {
        let g = RiskGuard::new(RiskConfig::default(), Mode::Live);
        let id = crate::types::StrategyId::NearCertain;
        assert_eq!(g.orders_sent_last_hour(id), 0);
        g.record_order(id);
        g.record_order(id);
        assert_eq!(g.orders_sent_last_hour(id), 2);
    }

    // ── Cooldowns after losses (CooldownPeriod + StoplossGuard) ───────────────

    fn guard_with_cooldowns(entry_secs: u64, strike: u32, strike_secs: u64) -> Arc<RiskGuard> {
        let cfg = RiskConfig {
            entry_cooldown_secs: entry_secs,
            loss_strike: strike,
            loss_strike_cooldown_secs: strike_secs,
            ..RiskConfig::default()
        };
        RiskGuard::new(cfg, Mode::Paper)
    }

    /// CooldownPeriod: a losing exit on a bracket blocks NEW entries
    /// (Buy) on that cid — for any strategy, not only the one that lost —
    /// until `entry_cooldown_secs` expires.
    #[test]
    fn cooldown_per_bracket_blocks_entries_and_expires() {
        let g = guard_with_cooldowns(60, 0, 0);
        let id = StrategyId::NearCertain;
        let now = Utc::now().timestamp_millis();

        g.record_bracket_loss("cid-a");

        assert!(
            g.in_entry_cooldown(id, "cid-a", now),
            "the bracket with a losing exit must block entries"
        );
        assert!(
            g.in_entry_cooldown(StrategyId::ParityMaker, "cid-a", now),
            "the block is per bracket: other strategies see it too"
        );
        assert!(
            !g.in_entry_cooldown(id, "cid-b", now),
            "another bracket without a loss is not affected"
        );
        assert!(
            !g.in_entry_cooldown(id, "cid-a", now + 61_000),
            "after the cooldown, re-entry is allowed"
        );
    }

    /// StoplossGuard: N consecutive losing exits pause the strategy's
    /// entries for `loss_strike_cooldown_secs`.
    #[test]
    fn losing_streak_pauses_strategy_after_n_exits() {
        let g = guard_with_cooldowns(0, 2, 60);
        let id = StrategyId::NearCertain;
        let now = Utc::now().timestamp_millis();

        g.record_strategy_pnl(id, dec!(-1));
        assert!(
            !g.in_entry_cooldown(id, "cid-a", now),
            "a single loss is not enough with loss_strike=2"
        );

        g.record_strategy_pnl(id, dec!(-1));
        assert!(
            g.in_entry_cooldown(id, "cid-a", now),
            "the 2nd consecutive loss must trigger the pause"
        );
        assert!(
            !g.in_entry_cooldown(StrategyId::ParityMaker, "cid-a", now),
            "the streak pause applies only to the losing strategy"
        );
        assert!(
            !g.in_entry_cooldown(id, "cid-a", now + 61_000),
            "the pause expires after loss_strike_cooldown_secs"
        );
    }

    /// A winning exit resets the streak: N consecutive losses must be
    /// accumulated again for the pause to fire again.
    #[test]
    fn winning_exit_resets_losing_streak() {
        let g = guard_with_cooldowns(0, 2, 60);
        let id = StrategyId::NearCertain;
        let now = Utc::now().timestamp_millis();

        g.record_strategy_pnl(id, dec!(-1));
        g.record_strategy_pnl(id, dec!(5)); // win → streak to 0
        g.record_strategy_pnl(id, dec!(-1)); // streak = 1 < loss_strike
        assert!(
            !g.in_entry_cooldown(id, "cid-a", now),
            "an intermediate win must break the streak (StoplossGuard)"
        );
    }

    /// A break-even exit (pnl == 0) neither adds to the streak nor breaks it.
    #[test]
    fn tied_exit_neither_adds_nor_breaks_streak() {
        let g = guard_with_cooldowns(0, 2, 60);
        let id = StrategyId::NearCertain;
        let now = Utc::now().timestamp_millis();

        g.record_strategy_pnl(id, dec!(-1));
        g.record_strategy_pnl(id, dec!(0)); // break-even: streak stays at 1
        assert!(!g.in_entry_cooldown(id, "cid-a", now));
        g.record_strategy_pnl(id, dec!(-1)); // streak = 2 → fires
        assert!(
            g.in_entry_cooldown(id, "cid-a", now),
            "the tie must not break the streak: the 2nd loss fires"
        );
    }

    /// Defaults (all 0) = feature off: neither the bracket nor the streak block —
    /// behavior is exactly what it was before the protection existed.
    #[test]
    fn cooldowns_disabled_with_zero_defaults() {
        let g = RiskGuard::new(RiskConfig::default(), Mode::Paper);
        let id = StrategyId::NearCertain;
        let now = Utc::now().timestamp_millis();

        g.record_bracket_loss("cid-a");
        for _ in 0..10 {
            g.record_strategy_pnl(id, dec!(-1));
        }
        assert!(
            !g.in_entry_cooldown(id, "cid-a", now),
            "with everything at 0 nothing must block"
        );
    }

    /// `loss_strike` configured but 0s pause: the streak is counted but
    /// never blocks (a zero-duration pause cannot contain `now`).
    #[test]
    fn streak_without_pause_duration_does_not_block() {
        let g = guard_with_cooldowns(0, 1, 0);
        let id = StrategyId::NearCertain;
        g.record_strategy_pnl(id, dec!(-1));
        assert!(!g.in_entry_cooldown(id, "cid-a", Utc::now().timestamp_millis()));
    }
}

// ── Snapshot for the UI ──────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct RiskSnapshot {
    pub global_killed: bool,
    pub global_reason: Option<String>,
    pub session_pnl: Decimal,
    pub daily_loss_stop: Decimal,
    pub max_total_exposure: Decimal,
}
