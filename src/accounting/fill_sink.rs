//! Fill sink: persists fills in SQLite and re-emits them on the strategies' broadcast.
//!
//! Single fan-out point for fills: every path (paper executor, live user_ws)
//! feeds this mpsc channel. fill_sink persists → deduplicates → resolves the
//! strategy → broadcasts to the shared strategies/Telegram channel.
//!
//! Updates:
//! - `fills` (raw record, deduplicated with INSERT OR IGNORE)
//! - `positions` (opening / averaging / closing of positions)
//! - `transactions` (trade ledger for PnL and the history UI)

use std::collections::HashMap;
use std::sync::Arc;

use rust_decimal::Decimal;
use tokio::sync::{broadcast, mpsc, watch};
use tracing::{info, warn};

use crate::{
    coordination::BracketLedger,
    risk::RiskGuard,
    state::{AppliedFill, EconomicFlows, StateStore},
    types::{DeadLetterFill, Fill, FillSide, StrategyId},
};

/// Time cap for the final fill drain on shutdown.
///
/// It is checked BETWEEN fills, never halfway through one: cancelling a
/// `process` midway would lose a fill already taken off the channel, which is
/// exactly what this drain exists to prevent. The real deadline can be exceeded
/// by the cost of one fill (up to 3 attempts with backoff, ~0.6 s + SQLite);
/// what it guarantees is that shutdown does not hang indefinitely.
const SHUTDOWN_DRAIN_CAP: std::time::Duration = std::time::Duration::from_secs(5);

/// Period of the quarantine reconciler: on each tick it retries owner
/// resolution for the fills in `dead_letter_fills`.
/// The first tick is immediate — it covers the startup after a restart.
const QUARANTINE_RECONCILE_PERIOD: std::time::Duration = std::time::Duration::from_secs(15);

// The shutdown `fin_tx` brings it to 8 arguments; same approach as the other
// long-lived tasks in the tree (executor, sim_router, supervisor…).
#[allow(clippy::too_many_arguments)]
pub async fn run(
    mut fill_rx: mpsc::UnboundedReceiver<Fill>,
    state: Arc<StateStore>,
    risk: Arc<RiskGuard>,
    ledger: Arc<BracketLedger>,
    fill_tx: broadcast::Sender<Fill>,
    order_map: Arc<parking_lot::Mutex<HashMap<String, StrategyId>>>,
    mut shutdown: watch::Receiver<bool>,
    done_tx: tokio::sync::oneshot::Sender<()>,
) {
    info!("fill_sink: starting");
    // Quarantine fill reconciler — retries owner attribution
    // (e.g. the race between the WS fill and the POST response, which
    // resolves itself once `record_order_index` lands). The first tick is
    // immediate: after a restart, whatever was quarantined before the crash is
    // reviewed in the first step.
    let mut reconciler = tokio::time::interval(QUARANTINE_RECONCILE_PERIOD);
    loop {
        tokio::select! {
            result = fill_rx.recv() => {
                match result {
                    Some(fill) => process_fill(&state, &risk, &ledger, &fill_tx, &order_map, fill).await,
                    None => {
                        info!("fill_sink: channel closed, exiting");
                        break;
                    }
                }
            }
            _ = reconciler.tick() => {
                reconcile_quarantine(&state, &risk, &ledger, &fill_tx, &order_map).await;
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("fill_sink: shutdown received — draining in-flight fills");
                    // This used to be a bare `return` and every fill
                    // already queued in the mpsc was lost (only the next
                    // startup's `backfill_missed_trades` recovered it, which is
                    // best-effort and does not even exist in paper).
                    drain_pending(
                        &mut fill_rx, &state, &risk, &ledger, &fill_tx, &order_map,
                        SHUTDOWN_DRAIN_CAP,
                    ).await;
                    break;
                }
            }
        }
    }
    // Tell `main` that nothing is left pending to persist, so
    // the shutdown `abort_all()` does not fire too early.
    let _ = done_tx.send(());
}

/// Drains the fills that were ALREADY in the channel when shutdown arrived.
/// Returns how many it persisted.
///
/// The drain waits for the channel to CLOSE (`recv() → None`), not for it to
/// empty. With `try_recv` there was a residual window: a still-live producer
/// (executor/user_ws) could queue a fill between the last empty
/// `try_recv` and the `fin_tx.send(())`, and that fill was lost without a log.
/// `main` now aborts the producers before waiting for the sink
/// (shutdown producer grace period + abort via AbortHandle), so
/// the senders drop during the drain and `recv` returns `None` — happens-before
/// ordering: every `send()` precedes the drop of its sender, so
/// whatever was queued before the close is persisted and whatever was not queued
/// never existed (it is covered by the intent's Unknown journal).
///
/// The `cap` remains as a safety net in case a sender survives
/// (a producer bug that does not die): it is checked between fills with
/// `timeout` over the `recv` — wait, never cancel a `process` midway.
async fn drain_pending(
    fill_rx: &mut mpsc::UnboundedReceiver<Fill>,
    state: &StateStore,
    risk: &RiskGuard,
    ledger: &BracketLedger,
    fill_tx: &broadcast::Sender<Fill>,
    order_map: &parking_lot::Mutex<HashMap<String, StrategyId>>,
    cap: std::time::Duration,
) -> usize {
    let start = std::time::Instant::now();
    let mut drained = 0usize;
    let mut by_cap = false;
    loop {
        let remaining = cap.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            by_cap = true;
            break;
        }
        match tokio::time::timeout(remaining, fill_rx.recv()).await {
            Ok(Some(fill)) => {
                process_fill(state, risk, ledger, fill_tx, order_map, fill).await;
                drained += 1;
            }
            // Channel closed: all senders dropped → drain complete.
            Ok(None) => break,
            // Cap expired while waiting for fills or for the close.
            Err(_) => {
                by_cap = true;
                break;
            }
        }
    }
    if by_cap {
        warn!(
            drained,
            pending = fill_rx.len(),
            cap_ms = cap.as_millis(),
            "fill_sink: drain cap expired — fills remain UNPERSISTED"
        );
    } else {
        info!(drained, "fill_sink: fills channel drained on shutdown");
    }
    drained
}

/// Persists a fill and re-emits it on the broadcast. Extracted from `run` so the
/// shutdown drain uses exactly the same path (retries included).
///
/// `pub` for the replay engine: the driver processes the channel's fills
/// in the virtual order of their emission, through this SAME function (owner
/// resolution, retries, kill on durable failure and broadcast included) —
/// no parallel accounting.
pub async fn process_fill(
    state: &StateStore,
    risk: &RiskGuard,
    ledger: &BracketLedger,
    fill_tx: &broadcast::Sender<Fill>,
    order_map: &parking_lot::Mutex<HashMap<String, StrategyId>>,
    mut fill: Fill,
) {
    // In live, the Fill's `strategy` is a placeholder (handle_trade stamps
    // Arb by default — the real WS carries no strategy). Owner resolution
    // goes by identity, not by the market:token pair: `local_order_id` →
    // Unknown journal; `order_id` → `order_index`/`gtd_orders` (durable,
    // survives a restart); `order_map` only as a last resort within the session.
    // If NONE resolves, the fill is NOT accounted for nor broadcast: it stays
    // in durable quarantine (`dead_letter_fills` + a pending `unknown_attempts`,
    // which blocks bracket buys via `has_unresolved_unknown`). It used to fall
    // to the Arb catch-all and be accounted to the wrong owner.
    // Rejected/Cancelled are internal executor signals — they are not persisted
    // and need no resolvable owner.
    if fill.mode == crate::types::Mode::Live
        && matches!(fill.side, FillSide::Buy | FillSide::Sell)
        && !resolve_owner(state, order_map, &mut fill).await
    {
        let reason = format!(
            "live fill without resolvable owner (exec_id={:?} order_id={:?} \
             local_order_id={:?} declared strategy={})",
            fill.exec_id, fill.order_id, fill.local_order_id, fill.strategy,
        );
        metrics::counter!("hrf_fill_sink_unowned_quarantined_total").increment(1);
        tracing::error!(
            market_cid = %fill.market_cid, token_id = %fill.token_id,
            side = %fill.side, price = %fill.price, size = %fill.size,
            "fill_sink: FILL QUARANTINED — no safe attribution; not applied \
             to positions nor broadcast; bracket blocked until reconciliation",
        );
        // If the quarantine cannot be written, the fill is left without a trace:
        // same reasoning as above — better to kill the bot than lose money
        // without persistence.
        if let Err(e) = state.quarantine_unowned_fill(&fill, &reason).await {
            tracing::error!(
                "fill_sink: failed writing quarantine of unowned fill \
                 — GLOBAL KILL: {e}"
            );
            risk.kill_global(format!(
                "failed writing fill quarantine: {e}"
            ));
        }
        return;
    }
    let resolved = fill.clone();
    let t0 = std::time::Instant::now();
    // A fill that fails to persist is real money
    // without a trace — previously a single failure was logged with `warn!`
    // and the fill was lost forever, with no retry. Minimum
    // viable (no elaborate retry queue): 3
    // attempts with short backoff; if all 3 fail, kill the
    // bot (`risk.kill_global`) instead of silently losing the trace.
    let mut attempt = 0u32;
    let outcome = loop {
        match process(state, risk, ledger, fill.clone()).await {
            Ok(v) => break Ok(v),
            Err(e) if attempt < 2 => {
                attempt += 1;
                tracing::error!(
                    attempt = attempt, order_id = ?fill.order_id, token = %fill.token_id,
                    "fill_sink: failed persisting fill, retrying: {e}"
                );
                tokio::time::sleep(std::time::Duration::from_millis(200 * attempt as u64)).await;
            }
            Err(e) => break Err(e),
        }
    };
    match outcome {
        Ok(true) => {
            let _ = fill_tx.send(resolved);
        }
        Ok(false) => {} // duplicate — already broadcast on the first event
        Err(e) => {
            tracing::error!(
                market_cid = %resolved.market_cid, token_id = %resolved.token_id,
                side = %resolved.side, price = %resolved.price, size = %resolved.size,
                order_id = ?resolved.order_id, strategy = %resolved.strategy,
                "fill_sink: failed persisting fill after 3 attempts — GLOBAL KILL: {e}"
            );
            risk.kill_global(format!(
                "failed persisting fill after 3 attempts: {e}"
            ));
        }
    }
    // End-to-end sink latency (includes the fill's SQLite queries).
    metrics::histogram!("hrf_fill_sink_seconds").record(t0.elapsed().as_secs_f64());
}

/// Resolves the REAL owner of a live fill by identity, mutating
/// `fill.strategy` (and `fill.local_order_id` when the index knows it and the
/// fill did not carry it). Returns `true` if it was resolved.
///
/// Chain, from most to least reliable:
/// 1. `local_order_id` → `unknown_attempts` (the executor journals the
///    attempt_id pre-POST and uses it as local_order_id) → durable owner
///    even without `order_id`.
/// 2. `local_order_id` → `order_index` (accepted orders indexed at submit).
/// 3. `order_id` → `order_index` (taker and resting, survives a restart).
/// 4. `order_id` → `gtd_orders` (legacy GTD table).
/// 5. `order_map` `market:token` (memory, last writer wins — cross-strategy
///    collision is a known live limitation; only used if identity fails).
///
/// A DB error is treated as unresolved (warn + next link): if the
/// DB is broken, the later quarantine write will fail anyway and
/// trigger the global kill.
async fn resolve_owner(
    state: &StateStore,
    order_map: &parking_lot::Mutex<HashMap<String, StrategyId>>,
    fill: &mut Fill,
) -> bool {
    if let Some(local) = fill.local_order_id.clone() {
        match state.get_unknown_attempt_strategy(&local).await {
            Ok(Some(strat)) => {
                fill.strategy = strat;
                return true;
            }
            Ok(None) => {}
            Err(e) => warn!("fill_sink: error querying journal by local_order_id: {e}"),
        }
        match state.order_index_lookup_local(&local).await {
            Ok(Some((strat, oid))) => {
                fill.strategy = strat;
                if fill.order_id.is_none() {
                    fill.order_id = oid;
                }
                return true;
            }
            Ok(None) => {}
            Err(e) => warn!("fill_sink: error in order_index by local_order_id: {e}"),
        }
    }
    if let Some(oid) = fill.order_id.clone() {
        match state.order_index_lookup(&oid).await {
            Ok(Some((strat, local))) => {
                fill.strategy = strat;
                if fill.local_order_id.is_none() {
                    fill.local_order_id = local;
                }
                return true;
            }
            Ok(None) => {}
            Err(e) => warn!("fill_sink: error in order_index by order_id: {e}"),
        }
        match state.get_gtd_strategy(&oid).await {
            Ok(Some(strat)) => {
                fill.strategy = strat;
                return true;
            }
            Ok(None) => {}
            Err(e) => warn!("fill_sink: error in gtd_orders by order_id: {e}"),
        }
    }
    let key = format!("{}:{}", fill.market_cid, fill.token_id);
    if let Some(&strat) = order_map.lock().get(&key) {
        fill.strategy = strat;
        return true;
    }
    false
}

/// Quarantine reconciler.
///
/// A live fill can be quarantined by a TRANSIENT condition: the typical
/// race is the user-WS reporting the trade before the executor's
/// `submit` returns the `order_id` and `order_index` is written.
/// This periodic pass retries `resolve_owner` over each pending
/// row; if it resolves, it applies the fill through the SAME normal path
/// (`process` → `apply_fill`, dedup by `exec_id` — a fill applied by hand
/// earlier is not duplicated) and closes the quarantine + the companion
/// unknown, which unblocks the bracket.
///
/// If persisting the close fails the row is left pending: the fill is already
/// in `fills` and the next tick will deduplicate it and close it — a retry
/// cannot account twice.
///
/// The broadcast to strategies happens only if the execution was NEW
/// (`applied == true`) — same as in `process_fill`, where a durable duplicate
/// is not re-emitted. It is also emitted BEFORE closing the quarantine: if
/// the close fails, the retry will see `applied == false` and close without
/// re-emitting — so the broadcast is neither lost nor duplicated by that failure.
/// (Without a durable outbox there is no exactly-once between entry→broadcast→close
/// across a crash; this ordering only narrows the window, it does not remove it.)
///
/// `pub`: besides the internal periodic tick, an operator/API can force a
/// reconciliation pass after a manual intervention.
pub async fn reconcile_quarantine(
    state: &StateStore,
    risk: &RiskGuard,
    ledger: &BracketLedger,
    fill_tx: &broadcast::Sender<Fill>,
    order_map: &parking_lot::Mutex<HashMap<String, StrategyId>>,
) {
    let pending = match state.list_quarantined_fills().await {
        Ok(v) => v,
        Err(e) => {
            warn!("fill_sink: error listing quarantine to reconcile: {e}");
            return;
        }
    };
    for (dl_id, mut fill, attempt_id) in pending {
        if !resolve_owner(state, order_map, &mut fill).await {
            continue; // still unowned — stays quarantined and blocking
        }
        match process(state, risk, ledger, fill.clone()).await {
            Ok(applied) => {
                let resolution = format!(
                    "quarantined fill settled with owner {}",
                    fill.strategy
                );
                // Only a NEW execution is re-emitted — `applied ==
                // false` means the `exec_id` was already in `fills` (e.g.
                // a re-delivery through the normal channel or the first of two duplicate
                // quarantine rows) and its broadcast already went out then.
                // Re-emitting it would make every strategy without its own dedup
                // (`parity_maker::on_fill`, etc.) count it twice
                // in memory. Emit BEFORE the durable close: if
                // `resolve_quarantined_fill` fails, the next pass deduplicates
                // (`applied == false`) and only retries the close.
                if applied {
                    let _ = fill_tx.send(fill.clone());
                }
                // The close is correct in both cases (the quarantine row
                // is resolved whether the fill was new or duplicate) — what
                // was not idempotent was the broadcast.
                match state
                    .resolve_quarantined_fill(dl_id, &attempt_id, &resolution)
                    .await
                {
                    Ok(()) => {
                        metrics::counter!("hrf_fill_sink_quarantine_reconciled_total").increment(1);
                        info!(
                            market_cid = %fill.market_cid, token_id = %fill.token_id,
                            order_id = ?fill.order_id, strategy = %fill.strategy, applied,
                            "fill_sink: quarantine reconciled — fill settled and bracket unblocked"
                        );
                    }
                    Err(e) => warn!(
                        "fill_sink: quarantine applied but failed closing rows \
                         (the close will be retried; the fill is already deduplicated): {e}"
                    ),
                }
            }
            Err(e) => warn!(
                market_cid = %fill.market_cid, token_id = %fill.token_id,
                order_id = ?fill.order_id,
                "fill_sink: failed applying reconciled fill — stays in quarantine: {e}"
            ),
        }
    }
}

/// Consumer of the channel of orphan LIVE fills (market found neither in the
/// asset registry nor in `clear_at_cache`) — they used to be lost with a plain
/// `error!` in the log. Best-effort: if the write fails it is logged and we
/// move on (no `kill_global` here, unlike `run` above — the fill is already in
/// a degraded state without reliable `neg_risk`/`clear_at`; killing the bot
/// over a failed write to the DEAD-LETTER table adds no real safety).
pub async fn run_dead_letters(
    mut rx: mpsc::UnboundedReceiver<DeadLetterFill>,
    state: Arc<StateStore>,
    mut shutdown: watch::Receiver<bool>,
) {
    info!("fill_sink: dead-letter sink starting");
    loop {
        tokio::select! {
            dl = rx.recv() => {
                let Some(dl) = dl else {
                    info!("fill_sink: dead-letter channel closed, exiting");
                    return;
                };
                if let Err(e) = state.record_dead_letter_fill(&dl).await {
                    tracing::error!(
                        order_id = %dl.order_id, market_cid = %dl.market_cid, token_id = %dl.token_id,
                        "fill_sink: failed persisting dead-letter fill: {e}"
                    );
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("fill_sink: dead-letter sink — shutdown received, draining queue");
                    // Same drain-until-close as the main sink.
                    // This used to be a bare `return`: dead-letters
                    // already queued were lost — the durable evidence of unowned
                    // fills is exactly what must not be thrown away on shutdown. The
                    // senders (user_ws) drop with the producer abort of the
                    // phased shutdown; the cap is the net in case one survives.
                    let start = std::time::Instant::now();
                    loop {
                        let remaining = SHUTDOWN_DRAIN_CAP.saturating_sub(start.elapsed());
                        if remaining.is_zero() {
                            warn!(
                                pending = rx.len(),
                                "fill_sink: drain cap expired in dead-letters"
                            );
                            break;
                        }
                        match tokio::time::timeout(remaining, rx.recv()).await {
                            Ok(Some(dl)) => {
                                if let Err(e) = state.record_dead_letter_fill(&dl).await {
                                    tracing::error!(
                                        order_id = %dl.order_id, market_cid = %dl.market_cid,
                                        token_id = %dl.token_id,
                                        "fill_sink: failed persisting dead-letter during drain: {e}"
                                    );
                                }
                            }
                            Ok(None) | Err(_) => break,
                        }
                    }
                    return;
                }
            }
        }
    }
}

/// Persists and applies a fill to positions. Returns `true` if it was new, `false` if duplicate.
///
/// All the SQLite mutation (fills + positions + transactions) happens in ONE
/// transaction within `state.apply_fill`; only the in-memory effects
/// (ledger, risk) that depend on the result remain here.
pub async fn process(
    state: &StateStore,
    risk: &RiskGuard,
    ledger: &BracketLedger,
    fill: Fill,
) -> anyhow::Result<bool> {
    // Rejections: internal signal — broadcast to strategies to clear state; not persisted.
    // `GateRejected` travels the same way (it is not a market fact either).
    if matches!(
        fill.side,
        FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected
    ) {
        return Ok(true);
    }

    match state.apply_fill(&fill).await? {
        AppliedFill::Duplicate => {
            tracing::debug!(
                order_id = ?fill.order_id,
                token    = %fill.token_id,
                "fill_sink: duplicate fill discarded"
            );
            Ok(false)
        }
        AppliedFill::Buy => Ok(true),
        AppliedFill::SellWithoutPosition => {
            warn!(
                token    = %fill.token_id,
                strategy = %fill.strategy,
                "fill_sink: sell without a live position in DB — discarded"
            );
            Ok(true) // it was a new fill, we just have no position
        }
        AppliedFill::Sell { pnl, cost_released } => {
            // Ledger: release the sold cost basis (not the proceeds — the claim was the cost).
            // `cost_released` is the cost the accounting actually removed
            // from the position (micro-level proration, already clipped to the size in
            // DB if the fill was larger). Previously `avg_price × size` was recomputed, which
            // differs from that proration and over-released exposure.
            // The release is deducted from the fill owner's bucket — a
            // sell by A can no longer eat B's claim over the same token.
            ledger.release_exposure(
                &fill.market_cid,
                &fill.token_id,
                cost_released,
                fill.strategy,
            );
            risk.record_pnl(pnl);
            risk.record_strategy_pnl(fill.strategy, pnl);
            // Per-bracket cooldown on a losing exit: this path covers
            // resting sells filled via user_ws, which do not go through the executor's
            // cost check (that only sees immediate taker fills). The
            // call is idempotent — if the executor already armed it for the
            // same fill, rewriting the same `until` changes nothing.
            if pnl < rust_decimal::Decimal::ZERO {
                risk.record_bracket_loss(&fill.market_cid);
            }
            Ok(true)
        }
    }
}

// ── Reconciliation of economic return against the ledger ─────────────────────

/// Result of reconciling the portfolio return against the durable
/// ledger — the return must be reconstructible from the persisted
/// entries, not just from the algorithm's reward.
///
/// Two invariants, both exact (Decimal, micro-USDC):
/// - `cash_drift == 0`: cash moved EXACTLY by
///   `sells + redeem − buys` of the ledger (plus external contributions
///   accounted separately). If settlement does not credit the payout to the
///   paper cash, the drift reveals it.
/// - `pnl_drift == 0`: `pnl_ledger == cash_return + open_cost` — the
///   realized PnL plus the still-open cost reconstruct the return.
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnReconciliation {
    /// `final_wallet − initial_wallet − contributions` — return measured by
    /// the cash.
    pub cash_return: Decimal,
    /// `sells + redeem − buys` of the ledger.
    pub ledger_net_flow: Decimal,
    /// Realized PnL of the ledger (`sell_pnl + redeem_pnl`).
    pub pnl_ledger: Decimal,
    /// Cost still open in positions (`cost_basis` with size > 0).
    pub open_cost: Decimal,
    /// `cash_return − ledger_net_flow` — 0 if cash and ledger count
    /// the same.
    pub cash_drift: Decimal,
    /// `pnl_ledger − (cash_return + open_cost)` — 0 if the realized PnL
    /// reconstructs the return together with the open cost.
    pub pnl_drift: Decimal,
}

impl ReturnReconciliation {
    /// Both drifts at zero: economic return reconcilable with the ledger.
    pub fn reconciled(&self) -> bool {
        self.cash_drift == Decimal::ZERO && self.pnl_drift == Decimal::ZERO
    }
}

/// Reconciles the portfolio return against the ledger flows.
///
/// `initial_wallet`/`final_wallet` are the observed balances (sim or real);
/// `contributions` are funds added externally during the window
/// (`add_funds` of the sim, deposits) — they are not trading PnL.
///
/// Pure function — reading the ledger is done by
/// `StateStore::economic_flows` and reading the cash by whoever coordinates (replay,
/// calibration or tests); here they are only compared.
pub fn reconcile_economic_return(
    flows: &EconomicFlows,
    initial_wallet: Decimal,
    contributions: Decimal,
    wallet_final: Decimal,
) -> ReturnReconciliation {
    let cash_return = wallet_final - initial_wallet - contributions;
    let net_flow = flows.net_cash_flow();
    let pnl_ledger = flows.realized_pnl();
    ReturnReconciliation {
        cash_return,
        ledger_net_flow: net_flow,
        pnl_ledger,
        open_cost: flows.open_cost,
        cash_drift: cash_return - net_flow,
        pnl_drift: pnl_ledger - (cash_return + flows.open_cost),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Mode;

    async fn state() -> Arc<StateStore> {
        StateStore::open_memory(Mode::Paper).await.unwrap()
    }

    fn guard() -> Arc<RiskGuard> {
        RiskGuard::new(crate::config::RiskConfig::default(), Mode::Paper)
    }

    fn buy(token_id: &str) -> Fill {
        Fill {
            market_cid: "cid-shutdown".into(),
            token_id: token_id.into(),
            side: FillSide::Buy,
            price: "0.50".parse().unwrap(),
            size: "2".parse().unwrap(),
            order_id: None,
            ts: chrono::Utc::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::NearCertain,
            clear_at: chrono::Utc::now().timestamp() + 3600,
            outcome_label: None,
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
        }
    }

    /// Fills already queued when shutdown arrives must be
    /// persisted, not lost. Before the fix the `shutdown.changed()` branch
    /// did an immediate `return` and the `select!` chose between both branches at
    /// random: with 40 queued fills the probability that all of them survived
    /// was negligible.
    #[tokio::test]
    async fn shutdown_persists_already_queued_fills() {
        const N: usize = 40;
        let state = state().await;
        let risk = guard();
        let ledger = BracketLedger::new();
        let (fill_tx, fill_rx) = mpsc::unbounded_channel();
        let (bcast_tx, _bcast_rx) = broadcast::channel(N * 2);
        let order_map = Arc::new(parking_lot::Mutex::new(HashMap::new()));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();

        for i in 0..N {
            fill_tx.send(buy(&format!("token-{i}"))).unwrap();
        }
        shutdown_tx.send(true).unwrap();

        let sink = tokio::spawn(run(
            fill_rx,
            state.clone(),
            risk.clone(),
            ledger,
            bcast_tx,
            order_map,
            shutdown_rx,
            done_tx,
        ));
        tokio::time::timeout(std::time::Duration::from_secs(10), sink)
            .await
            .expect("fill_sink shutdown must not hang")
            .unwrap();

        let txs = state.recent_transactions(N as u32 * 2, None).await.unwrap();
        assert_eq!(
            txs.len(),
            N,
            "shutdown must persist ALL the already-queued fills"
        );
        done_rx
            .await
            .expect("the sink must tell main it finished draining");
    }

    /// The drain empties the channel and reports how many it persisted.
    #[tokio::test]
    async fn drain_empties_channel() {
        let state = state().await;
        let risk = guard();
        let ledger = BracketLedger::new();
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (bcast_tx, _bcast_rx) = broadcast::channel(16);
        let order_map = parking_lot::Mutex::new(HashMap::new());

        for i in 0..3 {
            fill_tx.send(buy(&format!("token-{i}"))).unwrap();
        }
        // The drain ends when the channel CLOSES (all senders
        // dropped), not when it empties — with a live sender `recv` would wait.
        drop(fill_tx);
        let drained = drain_pending(
            &mut fill_rx,
            &state,
            &risk,
            &ledger,
            &bcast_tx,
            &order_map,
            std::time::Duration::from_secs(10),
        )
        .await;

        assert_eq!(drained, 3);
        assert_eq!(fill_rx.len(), 0, "the channel is left empty");
        assert_eq!(state.recent_transactions(10, None).await.unwrap().len(), 3);
    }

    /// The residual producer→sink window. A fill that reaches the channel
    /// AFTER the shutdown signal (producer still alive during the drain)
    /// is persisted: the drain does not finish until the channel really closes
    /// (all senders dropped). With `try_recv` this fill was lost
    /// without a log if the drain saw the buffer empty an instant before the send.
    #[tokio::test]
    async fn shutdown_persists_fill_arriving_during_drain() {
        let state = state().await;
        let risk = guard();
        let ledger = BracketLedger::new();
        let (fill_tx, fill_rx) = mpsc::unbounded_channel();
        let (bcast_tx, _bcast_rx) = broadcast::channel(8);
        let order_map = Arc::new(parking_lot::Mutex::new(HashMap::new()));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();

        let sink = tokio::spawn(run(
            fill_rx,
            state.clone(),
            risk.clone(),
            ledger,
            bcast_tx,
            order_map,
            shutdown_rx,
            done_tx,
        ));

        // Producer that emits AFTER the signal: models an executor/user_ws
        // that still had a job in flight when the kill arrived.
        let producer = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            fill_tx.send(buy("token-late")).unwrap();
            // The sender drops when the task ends — just like when main
            // aborts the producer on a real shutdown.
        });

        shutdown_tx.send(true).unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(10), done_rx)
            .await
            .expect("the sink must signal after draining")
            .expect("the sink must not die without signalling");
        producer.await.unwrap();
        sink.await.unwrap();

        let txs = state.recent_transactions(10, None).await.unwrap();
        assert_eq!(
            txs.len(),
            1,
            "the fill queued during the drain must be persisted, not silently lost"
        );
    }

    /// If the cap expires, the drain still ENDS (does not hang shutdown),
    /// loses no half-processed fill and leaves a log record of how many
    /// were left unpersisted.
    #[test]
    fn drain_hitting_cap_does_not_hang_and_logs_it() {
        use tracing_subscriber::layer::SubscriberExt;

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let log = crate::log_buffer::new(16);
        let sub =
            tracing_subscriber::registry().with(crate::log_buffer::LogLayer::new(log.clone(), 16));

        tracing::subscriber::with_default(sub, || {
            rt.block_on(async {
                let state = state().await;
                let risk = guard();
                let ledger = BracketLedger::new();
                let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
                let (bcast_tx, _bcast_rx) = broadcast::channel(16);
                let order_map = parking_lot::Mutex::new(HashMap::new());

                for i in 0..3 {
                    fill_tx.send(buy(&format!("token-{i}"))).unwrap();
                }
                // Cap already expired on entry: the drain exits without processing any.
                let drained = drain_pending(
                    &mut fill_rx,
                    &state,
                    &risk,
                    &ledger,
                    &bcast_tx,
                    &order_map,
                    std::time::Duration::ZERO,
                )
                .await;

                assert_eq!(drained, 0);
                assert_eq!(
                    fill_rx.len(),
                    3,
                    "the cap is checked BETWEEN fills: none is taken off the channel just to be lost"
                );
            })
        });

        let lines = log.lock().unwrap().clone();
        assert!(
            lines
                .iter()
                .any(|l| l.contains("drain cap expired") && l.contains("pending=3")),
            "the expired cap must be logged with how many fills were left: {lines:?}"
        );
    }

    /// The per-bracket cooldown must also be armed from fills that arrive
    /// via user_ws (resting sells): previously only the executor's taker path
    /// did it. `pnl < 0` on the applied `Sell` = losing exit.
    #[tokio::test]
    async fn losing_sell_arms_bracket_cooldown() {
        let state = state().await;
        let cfg = crate::config::RiskConfig {
            entry_cooldown_secs: 60,
            ..crate::config::RiskConfig::default()
        };
        let risk = RiskGuard::new(cfg, Mode::Paper);
        let ledger = BracketLedger::new();

        let mut entry = buy("tok-x");
        entry.market_cid = "cid-loss".into();
        process(&state, &risk, &ledger, entry).await.unwrap();

        let mut output = buy("tok-x");
        output.market_cid = "cid-loss".into();
        output.side = FillSide::Sell;
        output.price = "0.40".parse().unwrap(); // bought at 0.50 → loss
        process(&state, &risk, &ledger, output).await.unwrap();

        let now = chrono::Utc::now().timestamp_millis();
        assert!(
            risk.in_entry_cooldown(StrategyId::NearCertain, "cid-loss", now),
            "the losing exit via fill_sink must arm the bracket cooldown"
        );
    }

    /// And a winning sell does NOT arm the cooldown.
    #[tokio::test]
    async fn winning_sell_does_not_arm_cooldown() {
        let state = state().await;
        let cfg = crate::config::RiskConfig {
            entry_cooldown_secs: 60,
            ..crate::config::RiskConfig::default()
        };
        let risk = RiskGuard::new(cfg, Mode::Paper);
        let ledger = BracketLedger::new();

        let mut entry = buy("tok-y");
        entry.market_cid = "cid-win".into();
        process(&state, &risk, &ledger, entry).await.unwrap();

        let mut output = buy("tok-y");
        output.market_cid = "cid-win".into();
        output.side = FillSide::Sell;
        output.price = "0.60".parse().unwrap();
        process(&state, &risk, &ledger, output).await.unwrap();

        let now = chrono::Utc::now().timestamp_millis();
        assert!(
            !risk.in_entry_cooldown(StrategyId::NearCertain, "cid-win", now),
            "a winning exit must not arm the cooldown"
        );
    }
}
