//! Unified order executor.
//!
//! **Single entry point for every order the bot places.** No strategy calls
//! `clob_sdk` directly. The full flow:
//!
//! ```text
//! Strategy → mpsc<QueuedIntent> → Executor → dyn OrderRouter → fill_persist_tx → fill_sink
//! ```
//!
//! The executor applies these checks, in this order:
//! 1. Blacklist (the market is on the blacklist → skip)
//! 2. Global kill switch (RiskGuard::is_killed)
//! 3. Per-strategy kill switch
//! 4. Per-strategy rate limit (check_order_limit)
//! 5. Position and exposure limits (buys only)
//! 6. Register (market_cid:token_id) → strategy in order_map
//! 7. Delegate to the OrderRouter (paper or live)
//! 8. On success: send the fill to fill_sink (which persists it and re-broadcasts it)
//! 9. For hedges: execute the second leg if the first one was accepted or filled

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashSet;
use rust_decimal_macros::dec;
use tokio::sync::{mpsc, watch};
use tracing::{debug, info, warn};

use crate::{
    coordination::BracketLedger,
    order::audit,
    order::router::{
        resting_claim_usdc, OpenOrderSide, OrderKind, OrderRequest, OrderRouter, RouterOpenOrder,
        RouterOrderStatus, TimeInForce,
    },
    risk::RiskGuard,
    state::StateStore,
    types::{Fill, FillSide, OrderIntent, OrderSide, QueuedIntent, StrategyId},
};

/// Unknown taker reconciliation: grace period before querying the
/// destination's trade tape. The trades API takes a few seconds to publish a
/// new execution, so checking earlier proves nothing.
const TAKER_TRADES_GRACE_MS: i64 = 90_000;

/// If there is no own trade within the window once this age has passed, the POST never
/// reached the venue: a taker (FOK/market) executes instantly or never. The claim is released
/// instead of waiting for the bracket's `clear_at` (which could be days away).
const TAKER_NO_TRADE_EXPIRY_MS: i64 = 300_000;

/// Trade↔intent matching window: the `match_ts` must fall within
/// `[ts_ms − 60 s, ts_ms + 300 s]` (clock skew on the left, publication latency
/// on the right).
const TAKER_TRADE_WINDOW_PRE_MS: i64 = 60_000;
const TAKER_TRADE_WINDOW_POST_MS: i64 = 300_000;

/// Trade tape lookback when cancelling a GTD that `order_status` reports as executed:
/// the order may stay live for hours (resting), so the query looks 24 h back;
/// the real filter is by `order_id`, not by time.
const TRADES_LOOKBACK_SECS: i64 = 86_400;

/// Retention of `known_order_ids` in the periodic prune. An id is only needed while its order
/// can still generate trades: live (`active_gtd`) or recorded less than 24 h ago in
/// `order_index` (late user_ws deliveries after fill/cancel). The repo's brackets live for
/// hours; 24 h is a generous margin that bounds the set to a day of orders instead of
/// letting it grow indefinitely.
const KNOWN_ORDER_IDS_RETENTION_MS: i64 = 86_400_000;

// ── Unknown state ─────────────────────────────────────────────────────────

/// Local marker of a send attempt whose result is unknown: the POST may have been
/// accepted (only its ACK was lost) or rejected before it reached the
/// matcher.
///
/// **Durable source of truth**: the matching row in
/// `state::StateStore::unknown_attempts` (SQLite). This struct is the in-memory
/// mirror that `Executor::unknown_attempts()` exposes for inspection and tests.
/// It is repopulated from the journal on startup
/// (`Executor::seed_unknown_attempts_from_journal_pub`). If the DB is lost, the cache is too,
/// but `block_by_unresolved_unknown` ALWAYS reads the durable journal on every
/// `handle_buy`, so the blocker does not depend on the cache.
/// Losing the cache therefore does not lift the blocker.
///
/// **Important**: this record is LOCAL: it is not sent to the exchange and is NOT
/// a remote idempotency key. `attempt_id` is unique within the instance
/// (monotonic counter plus the `SystemTime` nanosecond timestamp).
///
/// **How it is closed**: `state::StateStore::mark_unknown_resolved` with one of the
/// reasons: `"rejected"`, `"accepted"`, or free text for manual reconciliation.
/// There is no auto-resolution: the `unresolved` state needs a future worker that
/// queries the exchange's `/orders` endpoint, or an explicit decision by the operator.
/// Nothing else resolves it automatically.
#[derive(Clone, Debug)]
pub struct UnknownAttempt {
/// Unique local ID of the attempt (nanosecond timestamp plus monotonic counter).
/// Not sent to the exchange. Same key as the durable row.
    pub attempt_id: String,
    pub cid: String,
    pub token_id: String,
    pub strategy: StrategyId,
    pub cost: rust_decimal::Decimal,
    pub resting: bool,
    pub reason: String,
    pub ts_ms: i64,
    /// Unknown reconciler: declared price and size of the intent, used to
    /// match the order against the exchange's live orders when the `order_id` was
    /// lost with the POST response. `None` for old rows or quarantine rows
    /// (`qfill-*`), which are not journaled intents.
    pub price: Option<rust_decimal::Decimal>,
    pub size: Option<rust_decimal::Decimal>,
    /// Bracket close: the claim horizon kept in the ledger.
    /// `None` only in mirrors rebuilt from old or quarantine rows.
    pub clear_at: Option<i64>,
    /// How many consecutive cycles have queried the trade tape and got an EMPTY result
    /// within the window for a taker that has already expired. Releasing the claim on the
    /// first empty query treated "no trade published" as "never executed", so a transient
    /// API lag would turn into a false negative. Two confirmations (two queries separated
    /// by one reconciliation cycle) are required before releasing.
    /// In memory on purpose: a restart resets it to 0, which errs on the side of
    /// caution: it is more conservative, never less so.
    /// (a restart can only delay the release).
    pub no_trade_strikes: u8,
}

/// Generator of local `attempt_id`s: combines nanoseconds from the monotonic clock with
/// an atomic counter to guarantee uniqueness within the instance even when two attempts
/// happen in the same nanosecond. It is NOT cryptographically secure: it is a LOCAL
/// uncertainty marker, not a remote token.
fn next_attempt_id(counter: &AtomicU64) -> String {
    let n = counter.fetch_add(1, Ordering::Relaxed);
    // Uses `crate::time::now()` (not SystemTime): in replay the id carries the timestamp of the
    // current event, so the tape plus seed reproduce the same
    // attempt_ids (uniqueness is still guaranteed by the counter, as before).
    let ts = crate::time::now().timestamp_nanos_opt().unwrap_or(0);
    format!("unk-{ts}-{n}")
}

/// Local identity of an order that does NOT go through the Unknown journal
/// (today: sells, which reserve no cash, so they are not journaled before the POST).
/// It still reaches the `Fill` via `OrderRequest::local_order_id` for traceability, and
/// `order_index` for durable attribution. Its prefix differs from `unk-` so it is not
/// confused with a journaled attempt.
fn next_local_order_id(counter: &AtomicU64) -> String {
    let n = counter.fetch_add(1, Ordering::Relaxed);
    // Same reason as `next_attempt_id`: uses the replay clock when armed.
    let ts = crate::time::now().timestamp_nanos_opt().unwrap_or(0);
    format!("loc-{ts}-{n}")
}

/// Identity of a synthetic `GateRejected`/`Rejected` event (pre-submit).
/// Each rejection is a DISTINCT event (one per discarded intent): without its own
/// identity, two identical rejections in the same millisecond collide in the
/// strategies' dedup, and the second does not release its marked state
/// (observed in the paper smoke test: `pending_timeout` → RL episode invalidated).
fn next_gate_reject_id(counter: &AtomicU64) -> String {
    let n = counter.fetch_add(1, Ordering::Relaxed);
    let ts = crate::time::now().timestamp_nanos_opt().unwrap_or(0);
    format!("gate-{ts}-{n}")
}

// ── Executor ──────────────────────────────────────────────────────────────────

pub struct Executor {
    intent_rx: mpsc::Receiver<QueuedIntent>,
    fill_persist_tx: mpsc::UnboundedSender<Fill>,
    router: Arc<dyn OrderRouter>,
    risk: Arc<RiskGuard>,
    state: Arc<StateStore>,
    /// Map of (market_cid:token_id) → StrategyId. Populated before submit;
    /// fill_sink reads it to attribute the correct strategy to user_ws fills.
    order_map: Arc<parking_lot::Mutex<HashMap<String, StrategyId>>>,
    /// Map of `limit_key(market_cid:token_id:strategy)` → (order_id, cost_usdc,
    /// clear_at, strategy) for GTD. The owner is PART of the key (per-owner reservations):
    /// a `Cancel` from strategy B does not find A's entry, so it neither cancels A's order
    /// nor releases A's claim in the ledger. clear_at allows purging expired and
    /// strategy allows cancelling a strategy's GTDs when it is killed.
    /// (the kill switch is per strategy).
    active_gtd: HashMap<String, (String, rust_decimal::Decimal, i64, StrategyId)>,
    /// Available wallet balance (paper: sim; live: polled from the CLOB every 30 s).
    /// `None` = not yet known → do not trim.
    wallet_bal_rx: watch::Receiver<Option<rust_decimal::Decimal>>,
    /// Set of order_ids emitted by THIS bot in THIS session.
    /// user_ws filters MATCHED events to ignore orders from previous sessions
    /// or other external sources that arrive through the WS.
    pub known_order_ids: Arc<DashSet<String>>,
    /// Strategies that use FAK (Fill-And-Kill) instead of FOK.
    /// Controlled by `use_fak` in each strategy's config.
    fak_strategies: HashSet<StrategyId>,
    /// Ledger of net directional exposure per bracket.
    /// Claim before each submit; release on rejects, cancels and sells (fill_sink).
    ledger: Arc<BracketLedger>,
    /// Local Unknown markers (POST without a response), integrated close.
    /// Keyed by `attempt_id` (unique within the instance). See `UnknownAttempt`.
    /// In memory only: if the process dies, the markers are lost (documented).
    unknown_attempts: HashMap<String, UnknownAttempt>,
    /// Monotonic counter for `next_attempt_id`.
    attempt_counter: AtomicU64,
    /// Producer for the research recorder. Emits
    /// `EventKind::OrderSubmit` with the serialized `OrderRequest` just before
    /// each `audit::submit*`: the intent as sent, pre-result.
    /// Always present: an `EventProducer::sink()` by default discards without
    /// cost; `main` injects the producer of the real hub.
    research_prod: crate::research::events::EventProducer,
}

impl Executor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        intent_rx: mpsc::Receiver<QueuedIntent>,
        fill_persist_tx: mpsc::UnboundedSender<Fill>,
        router: Arc<dyn OrderRouter>,
        risk: Arc<RiskGuard>,
        state: Arc<StateStore>,
        order_map: Arc<parking_lot::Mutex<HashMap<String, StrategyId>>>,
        wallet_bal_rx: watch::Receiver<Option<rust_decimal::Decimal>>,
        known_order_ids: Arc<DashSet<String>>,
        fak_strategies: HashSet<StrategyId>,
        ledger: Arc<BracketLedger>,
    ) -> Self {
        Self {
            intent_rx,
            fill_persist_tx,
            router,
            risk,
            state,
            order_map,
            active_gtd: HashMap::new(),
            wallet_bal_rx,
            known_order_ids,
            fak_strategies,
            ledger,
            unknown_attempts: HashMap::new(),
            attempt_counter: AtomicU64::new(0),
            research_prod: crate::research::events::EventProducer::sink(),
        }
    }

    /// Injects the research recorder's producer. A setter instead of a
    /// `new` parameter: the constructor already takes 10 arguments.
    /// Its test call sites must not change.
    pub fn set_research_producer(&mut self, prod: crate::research::events::EventProducer) {
        self.research_prod = prod;
    }

    /// Emits `EventKind::OrderSubmit` with the `OrderRequest` as sent
    /// (pre-result). Never blocks (`register` is `try_send`) and never
    /// alters the flow: a discard only leaves a gap in the ledger's seq.
    ///
    /// Semantics: "dispatched to the router", not "reached the venue". In
    /// `submit_pair` leg 2 is recorded even if the router cuts it when leg
    /// 1 is rejected (dataset consumers must cross it with the fills
    /// to know what actually arrived).
    fn record_order_submit(&self, req: &OrderRequest) {
        // `research_prod` is ALWAYS present (hub injected in main):
        // return before serializing the request if the hub is inactive.
        if !self.research_prod.is_active() {
            return;
        }
        let prod = &self.research_prod;
        let now_ms = crate::time::now().timestamp_millis();
        let mut e = crate::research::events::EventInput::new(
            crate::research::contract::EventSource::Internal,
            crate::research::contract::EventKind::OrderSubmit,
            serde_json::to_value(req).unwrap_or(serde_json::Value::Null),
            now_ms,
        );
        e.market_cid = Some(req.market_cid.clone());
        e.token_id = Some(req.token_id.clone());
        e.stream = Some("executor".to_string());
        prod.record(e);
    }

    /// Emits an `EventKind::Sim(SimEventKind)`: the router's RESPONSE as observed by the
    /// executor (Accept/Reject/Fill, `Notify` for the ambiguous `unknown` response,
    /// CancelRequest/CancelAck for cancels).
    /// It is the counterpart of `OrderSubmit`: together they rebuild the submit→outcome
    /// cycle of each intent on the research tape.
    ///
    /// Source `InternalSim` in paper (the router is `SimulatedRouter`) and
    /// `PolymarketRest` in live (the response comes from the CLOB over REST). The real
    /// mode is preserved in `detail.router_mode`. Never blocks
    /// (`register` is `try_send`) and never alters the flow.
    ///
    /// Known limitation (documented for the contract): the sim's INTERNAL transitions
    /// (`Arrive` after the modelled latency, partial fills per level) happen within
    /// `sim_router` and are not observable here: this hook records the aggregated
    /// result that the router returns. `seed`/`local_seq` of the internal `SimEvent`
    /// are set to 0; the ledger's canonical sequence is the envelope's `local_seq`,
    /// which the producer assigns when sealing the entry.
    #[allow(clippy::too_many_arguments)]
    fn record_sim_event(
        &self,
        kind: crate::research::contract::SimEventKind,
        market_cid: &str,
        token_id: &str,
        local_order_id: Option<&str>,
        order_id: Option<&str>,
        exec_id: Option<&str>,
        reason: Option<String>,
        detail: serde_json::Value,
    ) {
        // Same guard: without an attached writer the `SimEvent` is built only
        // to be dropped by `record`.
        if !self.research_prod.is_active() {
            return;
        }
        let prod = &self.research_prod;
        use crate::research::contract::{EventKind, EventSource, SimEvent};
        let now_ms = crate::time::now().timestamp_millis();
        let mode = self.router.mode();
        let source = match mode {
            crate::types::Mode::Paper => EventSource::InternalSim,
            crate::types::Mode::Live => EventSource::PolymarketRest,
        };
        let detail = serde_json::json!({
            "router_mode": mode.to_string(),
            "data": detail,
        });
        let sim_ev = SimEvent::new(
            kind,
            now_ms,
            0,
            0,
            Some(market_cid.to_string()),
            Some(token_id.to_string()),
            local_order_id.map(str::to_string),
            order_id.map(str::to_string),
            exec_id.map(str::to_string),
            reason,
            detail,
        );
        let mut e = crate::research::events::EventInput::new(
            source,
            EventKind::Sim(kind),
            serde_json::to_value(&sim_ev).unwrap_or(serde_json::Value::Null),
            now_ms,
        );
        e.market_cid = Some(market_cid.to_string());
        e.token_id = Some(token_id.to_string());
        e.stream = Some("executor".to_string());
        prod.record(e);
    }

    /// Translates a `RouterResult` into the `Sim` event for the observed outcome:
    /// `Reject` (definitive rejection), `Fill` (execution, with `exec_id`, NEVER
    /// an ACK mistaken for a fill), `Notify` (`unknown`: POST response
    /// lost, ambiguous: neither a rejection nor a fill), and `Accept` (accepted
    /// without an immediate fill: resting or eligible).
    fn record_router_result(
        &self,
        market_cid: &str,
        token_id: &str,
        local_order_id: Option<&str>,
        result: &crate::order::router::RouterResult,
    ) {
        use crate::research::contract::SimEventKind;
        // Return before serializing the fill/`reason`:
        // with the hub inactive, `record_sim_event` would discard the built event.
        if !self.research_prod.is_active() {
            return;
        }
        if result.rejected {
            self.record_sim_event(
                SimEventKind::Reject,
                market_cid,
                token_id,
                local_order_id,
                result.order_id.as_deref(),
                None,
                result.reason.clone(),
                serde_json::Value::Null,
            );
        } else if result.unknown {
            self.record_sim_event(
                SimEventKind::Notify,
                market_cid,
                token_id,
                local_order_id,
                result.order_id.as_deref(),
                None,
                Some(format!(
                    "submit response lost (unknown): {}",
                    result.reason.as_deref().unwrap_or("")
                )),
                serde_json::Value::Null,
            );
        } else if let Some(f) = &result.fill {
            self.record_sim_event(
                SimEventKind::Fill,
                market_cid,
                token_id,
                local_order_id.or(f.local_order_id.as_deref()),
                f.order_id.as_deref().or(result.order_id.as_deref()),
                f.exec_id.as_deref(),
                None,
                serde_json::to_value(f).unwrap_or(serde_json::Value::Null),
            );
        } else {
            self.record_sim_event(
                SimEventKind::Accept,
                market_cid,
                token_id,
                local_order_id,
                result.order_id.as_deref(),
                None,
                result.reason.clone(),
                serde_json::Value::Null,
            );
        }
    }

    /// Cancel sent to the router: `CancelRequest` before the call.
    /// `order_id` is the venue id of the order to cancel.
    fn record_cancel_request(&self, market_cid: &str, token_id: &str, order_id: &str) {
        self.record_sim_event(
            crate::research::contract::SimEventKind::CancelRequest,
            market_cid,
            token_id,
            None,
            Some(order_id),
            None,
            None,
            serde_json::Value::Null,
        );
    }

    /// Outcome of the cancel: `CancelAck` if the router confirmed it, `Notify`
    /// with the error if it failed (the order may still be live, so it is not an ack).
    fn record_cancel_result(
        &self,
        market_cid: &str,
        token_id: &str,
        order_id: &str,
        result: &anyhow::Result<()>,
    ) {
        use crate::research::contract::SimEventKind;
        // Same guard: do not format the failure reason
        // if the hub is inactive.
        if !self.research_prod.is_active() {
            return;
        }
        match result {
            Ok(()) => self.record_sim_event(
                SimEventKind::CancelAck,
                market_cid,
                token_id,
                None,
                Some(order_id),
                None,
                None,
                serde_json::Value::Null,
            ),
            Err(e) => self.record_sim_event(
                SimEventKind::Notify,
                market_cid,
                token_id,
                None,
                Some(order_id),
                None,
                Some(format!("cancel failed: {e}")),
                serde_json::Value::Null,
            ),
        }
    }

    /// Blocking Unknown fix: seeds the in-memory cache with the
    /// `unresolved` entries of the durable journal and leaves them
    /// available for `handle_buy` to consult (via `block_by_unresolved_unknown`).
    /// It is `async` (it touches SQLite) and must be called RIGHT AFTER
    /// `Executor::new`, before the first `handle_buy` or `run`.
    ///
    /// The tests call it explicitly; `main` also calls it
    /// at startup.
    pub async fn seed_unknown_attempts_from_journal_pub(&mut self) {
        self.seed_unknown_attempts_from_journal().await;
    }

    /// Blocking Unknown fix: snapshot of the Unknown markers
    /// recorded during the life of the instance. Only for inspection and tests;
    /// the full journal lives in SQLite (`unknown_attempts`) and is kept
    /// across purges and cancels (it is not deleted automatically: an Unknown is
    /// neither a rejection nor a confirmed fill, so there is no "natural" event that
    /// closes it). `state::StateStore::list_unresolved_unknowns` is the only
    /// durable source of truth; this HashMap is an in-memory mirror.
    pub fn unknown_attempts(&self) -> Vec<UnknownAttempt> {
        self.unknown_attempts.values().cloned().collect()
    }

    /// Blocking Unknown fix: before the POST, records the pending entry in
    /// SQLite. If the write fails it returns `Err` and the caller must NOT
    /// invoke `router.submit` (fail-closed).
    ///
    /// The `attempt_id` is generated LOCALLY (it is not sent to the exchange) and is
    /// monotonic within the instance (see `next_attempt_id`). It is NOT a remote
    /// idempotency key, only a local uncertainty marker.
    async fn record_pending_unknown(
        &self,
        intent: &OrderIntent,
        cost: rust_decimal::Decimal,
        reason: &str,
    ) -> anyhow::Result<String> {
        let attempt_id = next_attempt_id(&self.attempt_counter);
        let cost_um = crate::state::to_micros(cost);
        // The intent's declared side is stored durably in the SAME
        // insert as the pending: without it the reconciler could only match by
        // (token, price), and a foreign resting sell would steal a pending BUY.
        let side = match &intent.side {
            OrderSide::Buy => Some("buy"),
            OrderSide::Sell { .. } => Some("sell"),
            // A cancel pending does not know the original order's side:
            // NULL, never invented.
            OrderSide::Cancel => None,
        };
        self.state
            .record_unknown_attempt_pending_meta(
                &attempt_id,
                &intent.market_cid,
                &intent.token_id,
                &intent.strategy.to_string(),
                cost_um,
                intent.resting,
                reason,
                Some(crate::state::to_micros(intent.price)),
                Some(crate::state::to_micros(intent.size)),
                Some(intent.clear_at),
                side,
            )
            .await?;
        Ok(attempt_id)
    }

    /// Journals the Unknown `pending` of a GTD order whose
    /// `cancel` was left uncertain (Ok + indeterminate `order_status`, or with an
    /// executed part without durable fills).
    ///
    /// The `Cancel` intent does not carry the original order's metadata: its
    /// `price`/`size` are `0` by protocol. Stamping them as real would make the
    /// reconciler compute `remaining × 0 = 0` and release the claim. `price_um`,
    /// `size_um` and `side` are journaled as NULL. The durable claim is `cost` (the
    /// original claim in `active_gtd`), and the reconciler uses the destination's
    /// evidence (the exact open order) or keeps that cost; it never reduces it for lack of data.
    /// `clear_at` is kept from the order: the real expiry of `active_gtd`, not the one
    /// of the cancel intent.
    async fn record_pending_cancel(
        &self,
        intent: &OrderIntent,
        cost: rust_decimal::Decimal,
        order_clear_at: i64,
        reason: &str,
    ) -> anyhow::Result<String> {
        let attempt_id = next_attempt_id(&self.attempt_counter);
        self.state
            .record_unknown_attempt_pending_meta(
                &attempt_id,
                &intent.market_cid,
                &intent.token_id,
                &intent.strategy.to_string(),
                crate::state::to_micros(cost),
                true, // the order was a resting GTD
                reason,
                None, // real price unknown: do not stamp the Cancel's 0
                None, // real size unknown
                Some(order_clear_at),
                None, // side unknown: the order may be BUY or SELL
            )
            .await?;
        Ok(attempt_id)
    }

    /// In-memory mirror of the cancel pending: `price`/`size` set to
    /// `None`, same as in the journal (see `record_pending_cancel`).
    fn register_cancel_pending(
        &mut self,
        intent: &OrderIntent,
        cost: rust_decimal::Decimal,
        order_clear_at: i64,
        reason: &str,
        attempt_id: &str,
    ) {
        let entry = UnknownAttempt {
            attempt_id: attempt_id.to_string(),
            cid: intent.market_cid.clone(),
            token_id: intent.token_id.clone(),
            strategy: intent.strategy,
            cost,
            resting: true,
            reason: reason.to_string(),
            ts_ms: crate::time::now().timestamp_millis(),
            price: None,
            size: None,
            clear_at: Some(order_clear_at),
            no_trade_strikes: 0,
        };
        self.unknown_attempts.insert(attempt_id.to_string(), entry);
    }

    /// Marks a `pending` as `resolved`. The reason describes the observed outcome:
    /// `rejected` for definitive rejections from the exchange, `accepted`
    /// for a confirmed response (with or without a fill), `manual` for an
    /// explicit reconciliation by the operator.
    async fn resolve_pending_unknown(&self, attempt_id: &str, resolution: &str) {
        if let Err(e) = self
            .state
            .mark_unknown_resolved(attempt_id, resolution)
            .await
        {
            tracing::error!(
                attempt_id,
                error = %e,
                "executor: failed to mark unknown as resolved; the durable journal \
                 may be inconsistent with the in-memory marker",
            );
        }
    }

    /// Blocking Unknown fix: records in memory an uncertainty marker when the POST
    /// has just returned `RouterResult::unknown`.
    /// The durable journal ALREADY HAS its `pending` entry from before the submit
    /// (`record_pending_unknown`); only the in-memory mirror is updated here,
    /// so the Executor can query it without touching the DB on the hot path.
    ///
    /// If the in-memory entry already existed (same `attempt_id`), its
    /// `reason` and `ts_ms` are updated. It is NOT duplicated.
    fn register_unknown_attempt(
        &mut self,
        intent: &OrderIntent,
        cost: rust_decimal::Decimal,
        reason: &str,
        attempt_id: &str,
    ) {
        let entry = UnknownAttempt {
            attempt_id: attempt_id.to_string(),
            cid: intent.market_cid.clone(),
            token_id: intent.token_id.clone(),
            strategy: intent.strategy,
            cost,
            resting: intent.resting,
            reason: reason.to_string(),
            ts_ms: crate::time::now().timestamp_millis(),
            price: Some(intent.price),
            size: Some(intent.size),
            clear_at: Some(intent.clear_at),
            no_trade_strikes: 0,
        };
        self.unknown_attempts.insert(attempt_id.to_string(), entry);
    }

    /// Blocking Unknown fix: seeds the in-memory cache with the
    /// `unresolved` entries that the durable journal had at startup (models a
    /// restart). Called ONCE after `Executor::new`, after loading the
    /// `StateStore`. If the read fails, the Executor does NOT abort: the
    /// per-attempt `has_unresolved_unknown` check keeps blocking (direct
    /// DB query), and the warn logs leave a trace of the failure.
    async fn seed_unknown_attempts_from_journal(&mut self) {
        match self.state.list_unresolved_unknowns().await {
            Ok(rows) => {
                for r in rows {
                    let cost = crate::state::from_micros(r.cost_um);
                    let entry = UnknownAttempt {
                        attempt_id: r.attempt_id,
                        cid: r.cid,
                        token_id: r.token_id,
                        strategy: match crate::types::StrategyId::from_str(&r.strategy) {
                            Ok(s) => s,
                            Err(_) => continue, // unknown strategy: skip
                        },
                        cost,
                        resting: r.resting,
                        reason: r.reason,
                        ts_ms: r.ts_ms,
                        price: r.price_um.map(crate::state::from_micros),
                        size: r.size_um.map(crate::state::from_micros),
                        clear_at: r.clear_at,
                        no_trade_strikes: 0, // restart ⇒ back to 0, more conservative
                    };
                    self.unknown_attempts
                        .insert(entry.attempt_id.clone(), entry);
                }
            }
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "executor: failed to read the durable unknowns journal at \
                     startup; handle_buy will keep blocking via a DB query",
                );
            }
        }
    }

    /// Blocking Unknown fix: queries the durable journal (source of truth) whether
    /// the `cid` has an `unresolved` entry. Used by `handle_buy` at the start to
    /// BLOCK new submits on ANY token of the same bracket while an earlier Unknown
    /// is not yet reconciled. Decision documented in
    /// `StateStore::has_unresolved_unknown`: the scope is per bracket, not per token,
    /// and it is not global.
    async fn block_by_unresolved_unknown(&self, cid: &str, token_id: &str) -> bool {
        match self.state.has_unresolved_unknown(cid).await {
            Ok(true) => {
                metrics::counter!("hrf_executor_unknown_blocked_total",).increment(1);
                tracing::warn!(
                    cid, token = %token_id,
                    "executor: skip — blocked by an unresolved unknown of the \
                     same cid (any token of the bracket) until \
                     explicit reconciliation",
                );
                true
            }
            Ok(false) => false,
            Err(e) => {
                // Fail-closed: if the DB query fails, we treat it as if
                // there were an unresolved entry (we do not send). The reverse (fail-open)
                // is exactly the risk this fix targets.
                tracing::error!(
                    cid, token = %token_id, error = %e,
                    "executor: error querying the durable journal — fail-closed, \
                     skip (the POST is not sent)",
                );
                true
            }
        }
    }

    /// Registers a GTD from an earlier session that is still live on the CLOB.
    /// Called during startup reconciliation (live only).
    ///
    /// Besides the local mapping, the `order_id` is added to
    /// `known_order_ids` (so a user_ws maker fill can match the order in
    /// `maker_orders`) and to `order_index` (durable owner attribution; without it a
    /// live fill after restart would fall into quarantine).
    pub async fn restore_gtd(
        &mut self,
        market_cid: &str,
        token_id: &str,
        order_id: String,
        cost: rust_decimal::Decimal,
        clear_at: i64,
        strategy: StrategyId,
    ) {
        let key = limit_key(market_cid, token_id, strategy);
        self.known_order_ids.insert(order_id.clone());
        self.active_gtd
            .insert(key, (order_id.clone(), cost, clear_at, strategy));
        if let Err(e) = self
            .state
            .record_order_index(&order_id, None, strategy, market_cid, token_id)
            .await
        {
            tracing::error!(
                order_id,
                "executor: failed to record order_index in restore_gtd: {e}"
            );
        }
    }

    /// Tells the strategy that one of its orders died without filling.
    ///
    /// It travels through the fill channel (like `Rejected`), and the fill_sink does NOT persist it: it is an
    /// internal signal, not a market fact. Without it, a maker that cancels and reposts has to treat its
    /// order as dead when the cancel is EMITTED, which is false (the cancel is async and the order may fill
    /// meanwhile), and ends up stacking GTDs.
    #[allow(clippy::too_many_arguments)]
    fn emit_cancel_ack(
        &self,
        market_cid: &str,
        token_id: &str,
        strategy: StrategyId,
        order_id: Option<String>,
        neg_risk: bool,
        clear_at: i64,
    ) {
        self.emit_fill(Fill {
            market_cid: market_cid.to_string(),
            token_id: token_id.to_string(),
            side: FillSide::Cancelled,
            price: rust_decimal::Decimal::ZERO,
            size: rust_decimal::Decimal::ZERO,
            order_id,
            ts: crate::time::now(),
            mode: self.router.mode(),
            neg_risk,
            strategy,
            clear_at,
            outcome_label: None,
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
        });
    }

    /// Notifies the strategy that one of its intents was discarded by a gate BEFORE it
    /// existed as an order. Without this, the risk gates in `handle_buy` returned
    /// silently while the strategy had already marked its state on emit (`yes_live += size`,
    /// `posted_bid = Some(p)`): a ghost state. A leg goes mute forever if the ghost `live`
    /// consumed the cap (`remaining = 0`: a cancel against nothing yields
    /// no ack that could release it).
    ///
    /// The fill travels as `GateRejected` with the `size`/`price` OF THE INTENT: the
    /// strategy undoes ONLY that contribution (a decrement), never resets the leg to zero
    /// (there may be another live order).
    /// `marked_size` is the size the strategy marked on emit (the original
    /// `intent.size`, before the balance trim below).
    fn emit_gate_rejected(
        &self,
        intent: &OrderIntent,
        marked_size: rust_decimal::Decimal,
        gate: &'static str,
    ) {
        metrics::counter!(
            "hrf_executor_gate_rejected_total",
            "strategy" => intent.strategy.to_string(),
            "gate" => gate,
        )
        .increment(1);
        self.emit_fill(Fill {
            market_cid: intent.market_cid.clone(),
            token_id: intent.token_id.clone(),
            side: FillSide::GateRejected,
            price: intent.price,
            size: marked_size,
            order_id: None,
            ts: crate::time::now(),
            mode: self.router.mode(),
            neg_risk: intent.neg_risk,
            strategy: intent.strategy,
            clear_at: intent.real_clear_at.unwrap_or(intent.clear_at),
            outcome_label: None,
            hedge_id: None,
            // Each rejection has its own identity: two identical intents discarded
            // in the same ms are NOT the same event. Without `exec_id` the strategy's
            // dedup merges them and the second does not release its pending.
            exec_id: Some(next_gate_reject_id(&self.attempt_counter)),
            local_order_id: None,
        });
    }

    /// Same signal as `emit_gate_rejected`, but for the whole intent:
    /// if it carries a `second_leg`, that leg was also marked by the strategy on
    /// emit and must be released with its own keyed ack (token, price and size of
    /// the leg). Always use this version in the pre-submit gates of `handle_buy`:
    /// a gate blocks the whole intent, legs included.
    /// It is the variant to use for gates.
    fn emit_gate_rejected_intent(
        &self,
        intent: &OrderIntent,
        marked_size: rust_decimal::Decimal,
        marked_second_size: Option<rust_decimal::Decimal>,
        gate: &'static str,
    ) {
        self.emit_gate_rejected(intent, marked_size, gate);
        if let Some(ref leg) = intent.second_leg {
            self.emit_gate_rejected(leg, marked_second_size.unwrap_or(leg.size), gate);
        }
    }

    /// Freshness budget of a queued intent, derived from the default of
    /// `max_book_age_ms`: if a book is staler than that, the runner discards it, and an intent
    /// that waits in the executor queue longer than that was decided on a book that no longer
    /// exists. It applies only to entries (Buy, including `second_leg`); Sell and Cancel
    /// are exits and are never discarded.
    const MAX_INTENT_QUEUE_WAIT: std::time::Duration =
        std::time::Duration::from_millis(crate::config::defaults::max_book_age_ms());

    /// Does the intent contain a Buy entry (the first leg or the hedge `second_leg`)?
    fn intent_has_buy(intent: &OrderIntent) -> bool {
        matches!(intent.side, OrderSide::Buy)
            || intent
                .second_leg
                .as_deref()
                .is_some_and(|l| matches!(l.side, OrderSide::Buy))
    }

    /// Sends a fill to the fill_sink (which persists it, deduplicates it and re-broadcasts it to strategies).
    fn emit_fill(&self, fill: Fill) {
        if let Err(e) = self.fill_persist_tx.send(fill) {
            tracing::error!("executor: fill_sink closed — fill lost: {e}");
        }
    }

    /// Main loop. Spawn it in a dedicated task.
    pub async fn run(mut self) {
        // React to the global kill immediately. Before, resting GTDs
        // survived the kill (only those of killed strategies were cancelled,
        // and only when the next intent arrived), and could fill later.
        let mut global_kill_rx = self.risk.subscribe_global_kill();
        // The purge/resync runs in its own select branch; before, it ran inline in
        // handle() and put SQLite queries and cancels ahead of the intent that
        // arrived right after the 30 s mark (and without intents it never ran).
        let mut purge_interval = tokio::time::interval(std::time::Duration::from_secs(30));
        purge_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let intent = tokio::select! {
                biased;
                changed = global_kill_rx.changed() => {
                    if changed.is_err() { continue; }
                    if *global_kill_rx.borrow() && !self.risk.paper_disable_kill_switch_active() {
                        self.cancel_all_gtds("global kill").await;
                    }
                    continue;
                }
                queued = self.intent_rx.recv() => match queued {
                    Some(q) => q,
                    None => break,
                },
                _ = purge_interval.tick() => {
                    self.purge_and_resync().await;
                    continue;
                }
            };
            let t0 = std::time::Instant::now();
            let strat = intent.0.strategy.to_string();
            // Occupancy of the intents channel (512 slots): if it nears the cap,
            // strategies start blocking in send().
            metrics::gauge!("hrf_intent_channel_len").set(self.intent_rx.len() as f64);
            // Before processing a buy, drain the channel and prioritize Sell/Cancel.
            // Otherwise a GTD with high HTTP latency delays urgent bail-sells.
            if matches!(intent.0.side, OrderSide::Buy) {
                let mut pending = vec![intent];
                while let Ok(extra) = self.intent_rx.try_recv() {
                    pending.push(extra);
                }
                // `sort_unstable_by_key` does NOT preserve the relative order of elements
                // with the same key. Sell and Cancel share key `0u8`. The PM unwind batch is
                // `[Cancel(yes), Cancel(no), Sell-resting(yes)]`. If it were reordered to
                // `[Sell(yes), Cancel(yes), ...]`, the Cancel would cancel the ask just posted by the Sell
                // instead of the original GTD, leaving the original bid alive. `sort_by_key` (stable)
                // keeps the arrival order among elements with the same key.
                pending.sort_by_key(|(i, _)| match i.side {
                    OrderSide::Sell { .. } | OrderSide::Cancel => 0u8,
                    OrderSide::Buy => 1u8,
                });
                for (p, queued_at) in pending {
                    // Real latency of the strategy→executor queue (wall clock, not replay).
                    let wait = queued_at.elapsed();
                    metrics::histogram!("hrf_intent_queue_wait_seconds",
                        "strategy" => p.strategy.to_string())
                    .record(wait.as_secs_f64());
                    // Entry intent decided on a dead book.
                    if wait > Self::MAX_INTENT_QUEUE_WAIT && Self::intent_has_buy(&p) {
                        metrics::counter!("hrf_intent_stale_dropped_total",
                            "strategy" => p.strategy.to_string())
                        .increment(1);
                        tracing::warn!(strategy = %p.strategy, cid = %p.market_cid,
                            wait_ms = %wait.as_millis(),
                            "executor: Buy intent discarded — too long in the queue (stale book)");
                        self.emit_gate_rejected_intent(
                            &p,
                            p.size,
                            p.second_leg.as_deref().map(|l| l.size),
                            "queue_stale",
                        );
                        continue;
                    }
                    self.handle(p).await;
                }
            } else {
                let (i, queued_at) = intent;
                let wait = queued_at.elapsed();
                metrics::histogram!("hrf_intent_queue_wait_seconds",
                    "strategy" => i.strategy.to_string())
                .record(wait.as_secs_f64());
                // Same gate in the non-Buy branch: a Sell/Cancel leg is never discarded,
                // but a Buy `second_leg` is an entry.
                if wait > Self::MAX_INTENT_QUEUE_WAIT && Self::intent_has_buy(&i) {
                    metrics::counter!("hrf_intent_stale_dropped_total",
                        "strategy" => i.strategy.to_string())
                    .increment(1);
                    tracing::warn!(strategy = %i.strategy, cid = %i.market_cid,
                        wait_ms = %wait.as_millis(),
                        "executor: intent with Buy discarded — too long in the queue (stale book)");
                    self.emit_gate_rejected_intent(
                        &i,
                        i.size,
                        i.second_leg.as_deref().map(|l| l.size),
                        "queue_stale",
                    );
                } else {
                    self.handle(i).await;
                }
            }
            metrics::histogram!("hrf_executor_handle_seconds", "strategy" => strat)
                .record(t0.elapsed().as_secs_f64());
        }
        info!("executor: intents channel closed, exiting");
    }

    /// Cancels ALL GTDs (global kill). It releases ledger exposure only when the cancel
    /// is confirmed; if it fails, the order may still be live on the CLOB and the claim is kept
    /// (conservatively) for a retry.
    async fn cancel_all_gtds(&mut self, reason: &str) {
        if self.active_gtd.is_empty() {
            return;
        }
        warn!(
            reason,
            n = self.active_gtd.len(),
            "executor: cancelling all GTDs"
        );
        let entries: Vec<(String, String, rust_decimal::Decimal, StrategyId)> = self
            .active_gtd
            .iter()
            .map(|(key, (oid, cost, _, strat))| (key.clone(), oid.clone(), *cost, *strat))
            .collect();
        for (key, oid, cost, strat) in entries {
            let mut parts = key.split(':');
            let market_cid = parts.next().unwrap_or("").to_string();
            let token_id = parts.next().unwrap_or("").to_string();
            self.record_cancel_request(&market_cid, &token_id, &oid);
            let cancel_res = self.router.cancel(&market_cid, &oid).await;
            self.record_cancel_result(&market_cid, &token_id, &oid, &cancel_res);
            match cancel_res {
                Ok(_) => {
                    self.active_gtd.remove(&key);
                    self.ledger
                        .release_exposure(&market_cid, &token_id, cost, strat);
                    if let Err(e) = self.state.delete_gtd_order(&oid).await {
                        tracing::error!(oid, "executor: failed to delete gtd_orders: {e}");
                    }
                    metrics::counter!("hrf_gtd_kill_cancels_total").increment(1);
                }
                Err(e) => {
                    warn!(oid, "executor: cancel failed during kill; the GTD may still be live, claim kept: {e}");
                }
            }
        }
    }

    /// Periodic maintenance (every 30 s, from the run() select):
    /// cancels GTDs of killed strategies, purges expired and resyncs
    /// the ledger with the DB. Off the critical path of the intent.
    ///
    /// `pub` for the replay engine: the driver schedules it as a periodic event of the
    /// virtual queue with the SAME cadence as production;
    /// the code path is identical to the wall-clock interval tick.
    pub async fn purge_and_resync(&mut self) {
        let now_ts = crate::time::now().timestamp();
        // Before releasing or rebuilding claims, refresh the real remainder
        // of each live GTD according to the destination (see `refresh_gtd_claims`).
        self.refresh_gtd_claims().await;
        // The other kill paths (the global kill in run(), steps 2/3 of
        // handle_buy, cancel_all_gtds) check `paper_disable_kill_switch_active()`
        // before treating a strategy as killed; this filter does not
        // did, so in paper with `paper_disable_kill_switch=true` this
        // periodic cycle cancelled GTDs of "killed" strategies that the
        // rest of the system kept treating as running (active).
        let paper_kill_disabled = self.risk.paper_disable_kill_switch_active();
        // Cancel GTDs of strategies disabled by the kill switch.
        // The mapping key carries the owner (cid:token:strategy); it is kept
        // in the tuple so that ONLY the killed strategy's entry is removed.
        let to_cancel: Vec<(
            String,
            String,
            String,
            String,
            rust_decimal::Decimal,
            StrategyId,
        )> = self
            .active_gtd
            .iter()
            .filter(|(_, (_, _, ca, strat))| {
                !paper_kill_disabled && self.risk.is_strategy_killed(*strat) && now_ts < *ca + 65
            })
            .map(|(key, (oid, cost, _, strat))| {
                let mut parts = key.split(':');
                let market_cid = parts.next().unwrap_or("").to_string();
                let token_id = parts.next().unwrap_or("").to_string();
                (
                    key.clone(),
                    market_cid,
                    oid.clone(),
                    token_id,
                    *cost,
                    *strat,
                )
            })
            .collect();
        for (key, market_cid, oid, token_id, cost, strat) in to_cancel {
            // Release exposure ONLY if the cancel is confirmed. If the cancel fails,
            // the order may still be live on the CLOB; releasing the claim would leave the
            // real exposure without accounting coverage. The entry is kept
            // and the next purge cycle retries.
            self.record_cancel_request(&market_cid, &token_id, &oid);
            let cancel_res = self.router.cancel(&market_cid, &oid).await;
            self.record_cancel_result(&market_cid, &token_id, &oid, &cancel_res);
            match cancel_res {
                Ok(_) => {
                    self.active_gtd.remove(&key);
                    self.ledger
                        .release_exposure(&market_cid, &token_id, cost, strat);
                    if let Err(e) = self.state.delete_gtd_order(&oid).await {
                        tracing::error!(oid, "executor: failed to delete gtd_orders: {e}");
                    }
                }
                Err(e) => {
                    warn!(oid, "executor: error cancelling the GTD of a disabled strategy; claim kept: {e}");
                }
            }
        }
        // Remove from the local mapping any GTD that is NO LONGER live on the
        // destination (it filled) but was never requoted or cancelled. Without this, its
        // claim counts TWICE against the position resync below (once as a live GTD,
        // and again as an already-open position in the DB).
        // `live_gtd_order_ids` returns `None` if the router does not support it today
        // (ClobRouter); in that case nothing is purged here, which is the conservative choice.
        if let Some(live_ids) = self.router.live_gtd_order_ids().await {
            let filled: Vec<(String, rust_decimal::Decimal, String, StrategyId)> = self
                .active_gtd
                .iter()
                .filter(|(_, (oid, ..))| !live_ids.contains(oid))
                .map(|(key, (oid, cost, _, strat))| (key.clone(), *cost, oid.clone(), *strat))
                .collect();
            for (key, cost, oid, strat) in filled {
                self.active_gtd.remove(&key);
                if cost > rust_decimal::Decimal::ZERO {
                    let mut parts = key.split(':');
                    let cid = parts.next().unwrap_or("");
                    let token = parts.next().unwrap_or("");
                    self.ledger.release_exposure(cid, token, cost, strat);
                }
                if let Err(e) = self.state.delete_gtd_order(&oid).await {
                    tracing::error!(oid, "executor: failed to delete gtd_orders: {e}");
                }
                tracing::debug!(
                    oid,
                    "executor: filled GTD removed from the local mapping"
                );
            }
        }
        // Purge expired: the GTD expired on the CLOB (clear_at+65 has passed),
        // so there is no live order to cancel; this only cleans up the local mapping.
        // Its `gtd_orders` row must also be deleted: the other purge paths (cancel_all_gtds,
        // the kill filter above, handle_cancel) do this, but this one used to leave orphan rows
        // in the table.
        let expired_oids: Vec<String> = self
            .active_gtd
            .iter()
            .filter(|(_, (_, _, ca, _))| now_ts >= *ca + 65)
            .map(|(_, (oid, ..))| oid.clone())
            .collect();
        self.active_gtd.retain(|_, (_, _, ca, _)| now_ts < *ca + 65);
        for oid in expired_oids {
            if let Err(e) = self.state.delete_gtd_order(&oid).await {
                tracing::error!(
                    oid,
                    "executor: failed to delete expired gtd_orders: {e}"
                );
            }
        }
        // Unknown reconciler: queries the destination for each pending attempt
        // of the durable journal and resolves it with evidence:
        // accepted-live / matched / cancelled / not-found. Whatever cannot be
        // proven stays pending (fail-closed).
        self.reconcile_unknowns().await;
        // Quarantine↔journal link: a quarantined fill whose order_id matches
        // the ONLY pending attempt of its (cid, token) becomes attributable.
        // See `link_quarantine_to_journal`.
        self.link_quarantine_to_journal().await;
        // Unknown watchdog: expires pendings whose bracket has already closed
        // (clear_at+65 s, same margin as the active_gtd purge) and alerts on those
        // unresolved for more than 30 min. It runs before
        // `ledger.purge_expired`, so the claim is released by the watchdog (with its
        // journal resolution), not by the blind purge.
        self.watchdog_unknowns().await;
        // Ledger: resolved brackets release residual claims (FOK without fill, partial FAK).
        self.ledger.purge_expired(now_ts);
        // Resync: rebuilds from the DB truth plus our own GTDs.
        // Corrects accumulated drift (FOK claims without fill, fills at a price different
        // from the claim). An in-flight order whose fill is not yet in the DB
        // loses its claim during a window of under 30 s, bounded by rate limits.
        match self.state.list_open_positions().await {
            Ok(open) => {
                self.ledger.clear();
                for p in &open {
                    self.ledger.force_add_exposure(
                        &p.market_cid,
                        p.clear_at,
                        &p.token_id,
                        p.cost_basis,
                        StrategyId::from_db_str(&p.strategy),
                    );
                }
                for (key, (_, cost, clear_at, strat)) in &self.active_gtd {
                    let mut parts = key.split(':');
                    let cid = parts.next().unwrap_or("");
                    let token = parts.next().unwrap_or("");
                    self.ledger
                        .force_add_exposure(cid, *clear_at, token, *cost, *strat);
                }
                // Claims from `unknown_attempts` still pending: without this, the
                // `clear()` above erased the protection of a POST with a lost response
                // after ~30 s, even though the order could still be live on the exchange;
                // the real exposure was left without coverage
                // accounting coverage (fail-open). Conservative: the claim
                // persists until the reconciler closes the attempt with
                // evidence from the destination.
                for att in self.unknown_attempts.values() {
                    if let Some(clear_at) = att.clear_at {
                        self.ledger.force_add_exposure(
                            &att.cid,
                            clear_at,
                            &att.token_id,
                            att.cost,
                            att.strategy,
                        );
                    }
                }
            }
            Err(e) => {
                warn!("executor: ledger resync failed (current state kept): {e}")
            }
        }
        // Safety net: purges orphan `gtd_orders` rows.
        // (1 h cutoff, well above any real GTD TTL in the repo: PCS-2 uses
        // gtd_ttl_secs=90 and PM uses pair_ttl_secs, both in seconds or minutes).
        // If this purges anything, it signals a real leak at another call site
        // that skipped its normal cancellation, so it stays visible in the logs.
        match self.state.purge_stale_gtd_orders(3600).await {
            Ok(purged) if purged > 0 => {
                warn!(
                    purged,
                    "executor: purged orphan gtd_orders rows"
                );
            }
            Ok(_) => {}
            Err(e) => warn!("executor: failed to purge orphan gtd_orders: {e}"),
        }
        // `known_order_ids` only grew. The ids of live orders
        // (active_gtd) and those recorded in order_index in the last
        // 24 h (a user_ws trade can arrive late after fill/cancel) are kept.
        // The rest cannot generate new trades. If the query fails nothing is
        // pruned: fail-closed, the set just keeps growing one more cycle.
        match self
            .state
            .order_ids_since(now_ts * 1000 - KNOWN_ORDER_IDS_RETENTION_MS)
            .await
        {
            Ok(recent) => {
                let live: HashSet<&str> = self
                    .active_gtd
                    .values()
                    .map(|(oid, _, _, _)| oid.as_str())
                    .collect();
                let before = self.known_order_ids.len();
                self.known_order_ids
                    .retain(|oid| live.contains(oid.as_str()) || recent.contains(oid));
                let pruned = before - self.known_order_ids.len();
                if pruned > 0 {
                    debug!(pruned, "executor: known_order_ids pruned");
                }
            }
            Err(e) => warn!("executor: failed to list order_index for the known_order_ids prune (not pruned): {e}"),
        }
    }

    /// Refreshes the claim of each active GTD with the REAL remainder
    /// the destination reports, instead of the original cost. The formula is no longer
    /// `remaining × price` for everything: the contractual rule applies per side
    /// (`resting_claim_usdc`): BUY = remaining×price, SELL
    /// of inventory = 0, short TC SELL = downside of the remainder.
    ///
    /// Without this, the ledger resync counted a partial fill twice: the position
    /// is already in the DB and the stored claim was the original whole cost, so
    /// `position + original_cost` inflated the exposure. With the refreshed claim,
    /// `accounted_position + remainder` rebuilds the order's total exposure exactly, and the
    /// exposure of the order exactly, and the cancels of this
    /// same cycle release only what is still really open.
    ///
    /// An order MISSING from the listing is NOT touched: it may be a fill not yet
    /// accounted for, or a gap in the API's pagination. Its claim is kept
    /// (fail-closed), and the filled-GTD purge and the Unknown reconciler decide with
    /// their own evidence. `None` from the router (unsupported or indeterminate query)
    /// means nothing is refreshed.
    async fn refresh_gtd_claims(&mut self) {
        let Some(orders) = self.router.open_gtd_orders().await else {
            return;
        };
        // The claim is no longer `remaining × price` for EVERY order: that formula
        // only describes BUY. The contractual rule (`resting_claim_usdc`) is applied with
        // the destination's side and the LOCAL owner of `active_gtd`.
        // `None` (indeterminate side or invalid data) means the order
        // contributes no evidence and the previous claim is kept.
        for (oid, cost, _clear_at, strategy) in self.active_gtd.values_mut() {
            let Some(order) = orders.iter().find(|o| o.order_id == *oid) else {
                continue; // missing from the snapshot: keep (fail-closed)
            };
            if let Some(claim) =
                resting_claim_usdc(order.side, order.remaining, order.price, *strategy)
            {
                *cost = claim;
            }
        }
    }

    /// Unknown reconciler (the piece that was missing). It walks the `pending` entries of
    /// the durable journal and asks the destination what happened to each order:
    /// (the destination is the source of truth for the outcome).
    ///
    /// - `order_index_lookup_local(attempt_id)` resolves the `order_id`
    ///   when the response arrived but the journal stayed pending (crash
    ///   between the response and `mark_unknown_resolved`).
    /// - If there is no `order_id` and the intent was `resting`, it is matched against the
    ///   destination's live orders by `(cid, token_id, side, price)`. It requires ONE single FREE
    ///   candidate (no other durable adjudication, and no other compatible pending that
    ///   disputes it); zero, several, or an already-adjudicated candidate leave the attempt pending.
    ///   Pendings without a declared side
    ///   (legacy/cancel) never match by eye.
    /// - Taker (`resting=0`) without `order_id`: there is no safe way to prove
    ///   its outcome without the id, so it stays pending. If it filled, the fill arrives
    ///   via user_ws and `link_quarantine_to_journal` unblocks it.
    ///
    /// Outcomes: `accepted-live` (registers the GTD in `active_gtd` with the
    /// remainder's claim), `matched` (the fill does the accounting via
    /// dedup by `exec_id`), `cancelled`/`not-found` (releases the owner's claim).
    /// Indeterminate query (`None` from the router) stays pending.
    /// Fail-closed: NOTHING is resolved without evidence from the destination.
    async fn reconcile_unknowns(&mut self) {
        let pending = match self.state.list_unresolved_unknowns().await {
            Ok(v) => v,
            Err(e) => {
                warn!("executor: reconcile_unknowns failed reading the journal: {e}");
                return;
            }
        };
        if pending.is_empty() {
            return;
        }
        // The destination's live orders are queried ONCE per cycle,
        // lazily: only if some resting pending lacks an order_id.
        let mut open_orders: Option<Vec<RouterOpenOrder>> = None;
        let mut open_orders_fetched = false;
        // Same for the trade tape (taker Unknowns): one query per cycle,
        // starting from the oldest pending taker attempt.
        let mut trades: Option<Vec<crate::order::router::RouterTrade>> = None;
        let mut trades_fetched = false;
        let taker_pendings: Vec<(String, String, i64)> = pending
            .iter()
            .filter(|r| !r.resting && !r.attempt_id.starts_with("qfill-"))
            .map(|r| (r.attempt_id.clone(), r.token_id.clone(), r.ts_ms))
            .collect();
        let min_taker_ts_ms = taker_pendings.iter().map(|r| r.2).min();
        // Snapshot of the RESTING pendings, used to dispute matches
        // without order_id: if two compatible pendings (same cid+token+
        // price+side) would claim the same live order, the candidate is
        // ambiguous and neither keeps it.
        let resting_pendings: Vec<&crate::state::UnknownAttemptRow> = pending
            .iter()
            .filter(|r| r.resting && !r.attempt_id.starts_with("qfill-"))
            .collect();

        for row in &pending {
            // Quarantine rows (`qfill-*`) are not journaled intents:
            // they are resolved by `link_quarantine_to_journal` and the
            // `fill_sink` reconciler, not by this pass.
            if row.attempt_id.starts_with("qfill-") {
                continue;
            }
            let Ok(strat) = StrategyId::from_str(&row.strategy) else {
                continue;
            };
            let cost = crate::state::from_micros(row.cost_um);

            // 1) order_id already known via order_index (late response or
            //    crash between the response and the journal resolution).
            //    The index evidence is validated AGAINST the pending
            //    (cid + token + strategy) before the oid is used. A contradictory row
            //    or one with an unrecognised strategy is indeterminate, not "no order_id":
            //    the pending is kept
            //    and that oid is not used.
            let order_id = match self.state.order_index_row_by_local(&row.attempt_id).await {
                Ok(Some(ev)) => match StrategyId::from_str(&ev.strategy) {
                    Ok(ev_strat)
                        if ev_strat == strat
                            && ev.cid == row.cid
                            && ev.token_id == row.token_id =>
                    {
                        Some(ev.order_id)
                    }
                    _ => {
                        warn!(
                            attempt_id = %row.attempt_id,
                            order_id = %ev.order_id,
                            "executor: order_index contradicts the pending entry — pending kept"
                        );
                        continue;
                    }
                },
                Ok(None) => None,
                Err(e) => {
                    warn!(attempt_id = %row.attempt_id, "executor: order_index_row_by_local failed: {e}");
                    continue;
                }
            };

            // 2) resting without order_id: match against the live orders of the
            //    destination. It is NO LONGER enough to match (token_id, price): the candidate
            //    must be UNIQUE, from the SAME cid, with the intent's declared side
            //    and free of any other adjudication; two compatible pendings from
            //    different attempts dispute the order, and neither steals it.
            if order_id.is_none() && row.resting {
                if !open_orders_fetched {
                    open_orders = self.router.open_gtd_orders().await;
                    open_orders_fetched = true;
                }
                let Some(list) = &open_orders else {
                    continue; // query unsupported or failed → pending
                };
                // Minimal durable metadata: without a declared price, or with an
                // invalid price (e.g. the `0` marker of a legacy cancel pending), no
                // match is possible: the attempt stays pending and the claim is
                // kept.
                let Some(price_um) = row.price_um else {
                    continue;
                };
                let price = crate::state::from_micros(price_um);
                if price <= rust_decimal::Decimal::ZERO || price > rust_decimal::Decimal::ONE {
                    continue;
                }
                // Without a declared side (NULL: legacy rows, quarantine,
                // cancel pendings) there is NO match by eye: the venue does not declare
                // the strategy, and an invented side could attribute the order to the
                // wrong attempt.
                let Some(row_side) = journal_side(row) else {
                    continue;
                };
                let mut candidates = list.iter().filter(|o| {
                    o.token_id == row.token_id
                        && o.market_cid == row.cid
                        && o.price == price
                        && o.side == row_side
                });
                let (Some(o), None) = (candidates.next(), candidates.next()) else {
                    // 0 candidates (already filled or cancelled? the next pass will see it) or
                    // ambiguous → pending (conservative).
                    continue;
                };
                let oid = o.order_id.clone();
                // Remainder claim via the contractual rule, computed
                // BEFORE writing any attribution: a candidate with invalid data (negative
                // remainder, indeterminate side, price out of range) does not leave a
                // half-seeded `order_index` row that the next cycle would take as "already owned".
                let Some(claim) = resting_claim_usdc(o.side, o.remaining, o.price, strat)
                else {
                    continue;
                };
                // The candidate order cannot already be adjudicated to ANOTHER
                // attempt or strategy: it must not be known in memory, nor have a row in
                // `order_index`, nor a legacy owner in `gtd_orders`
                // (an earlier restore may have left only this table).
                // A row present, even for the SAME owner, means the order already has its
                // own record: it is not reassigned to a different pending. A query error → pending.
                if self.known_order_ids.contains(&oid) {
                    continue;
                }
                match self.state.order_index_row(&oid).await {
                    Ok(Some(_)) => continue, // already owned in the index
                    Ok(None) => {}
                    Err(e) => {
                        warn!(oid, "executor: order_index_row failed during matching: {e}");
                        continue;
                    }
                }
                match self.state.gtd_order_row(&oid).await {
                    Ok(Some(_)) => continue, // legacy owner: do not reassign
                    Ok(None) => {}
                    Err(e) => {
                        warn!(oid, "executor: gtd_order_row failed during matching: {e}");
                        continue;
                    }
                }
                // Dispute: another COMPATIBLE resting pending could claim the same order,
                // making it ambiguous; both stay pending. A side or price that is UNKNOWN on
                // the other pending (legacy rows, cancel pendings) counts as compatible.
                // "Unknown" is not "different".
                // In doubt, nobody steals the order.
                let disputed = resting_pendings.iter().any(|p| {
                    p.attempt_id != row.attempt_id
                        && p.cid == row.cid
                        && p.token_id == row.token_id
                        && (p.price_um.is_none() || p.price_um == Some(price_um))
                        && journal_side(*p).map_or(true, |s| s == row_side)
                });
                if disputed {
                    continue;
                }
                // Unambiguous match: the order IS live on the destination.
                // Register it and resolve it without a second query. The attribution
                // (`order_index`) is durable and CRITICAL: if it fails, the pending is kept
                // and the next cycle retries.
                // It is never resolved without a durable owner.
                if let Err(e) = self
                    .state
                    .record_order_index(&oid, Some(&row.attempt_id), strat, &row.cid, &row.token_id)
                    .await
                {
                    tracing::error!(
                        oid,
                        "executor: failed to record order_index (reconciler); pending kept: {e}"
                    );
                    continue;
                }
                self.known_order_ids.insert(oid.clone());
                let clear_at = row.clear_at.unwrap_or(0);
                let key = limit_key(&row.cid, &row.token_id, strat);
                self.active_gtd
                    .insert(key, (oid.clone(), claim, clear_at, strat));
                if let Err(e) = self.state.record_gtd_order(&oid, &row.cid, strat).await {
                    tracing::error!(
                        oid,
                        "executor: failed to record gtd_orders (reconciler): {e}"
                    );
                }
                info!(
                    attempt_id = %row.attempt_id,
                    oid, strategy = %strat, cid = %row.cid,
                    "executor: Unknown reconciled — order LIVE on the destination (matched by cid+token+side+price)"
                );
                self.resolve_pending_unknown(&row.attempt_id, "accepted-live")
                    .await;
                self.unknown_attempts.remove(&row.attempt_id);
                continue;
            }

            // 3) Taker without order_id: the order cannot be left resting. If the
            //    POST arrived, it executed instantly and the trade is in the tape of the
            //    destination with `taker_order_id` = the lost identity. If there is no own
            //    trade after the grace period, the POST never arrived and the claim is released
            //    (before, it stayed pending until `clear_at`, blocking the claim for
            //    hours or days on long brackets).
            if order_id.is_none() && !row.resting {
                let now_ms = crate::time::now().timestamp_millis();
                if now_ms - row.ts_ms < TAKER_TRADES_GRACE_MS {
                    // The trades API takes a few seconds to publish: nothing
                    // can be concluded yet.
                    continue;
                }
                if !trades_fetched {
                    if let Some(min_ts) = min_taker_ts_ms {
                        trades = self.router.trades_since(min_ts / 1000 - 60).await;
                    }
                    trades_fetched = true;
                }
                let Some(list) = &trades else {
                    continue; // query unsupported or failed → pending
                };
                // A FOK/market can fill against several makers → N trades
                // with the SAME taker_order_id. They are grouped by order_id: a single
                // distinct order_id = proven outcome; several or none prove nothing about
                // THIS attempt.
                let mut by_oid: HashMap<&str, i64> = HashMap::new();
                for t in list.iter().filter(|t| {
                    t.is_buy
                        && t.token_id == row.token_id
                        && t.match_ts_ms >= row.ts_ms - TAKER_TRADE_WINDOW_PRE_MS
                        && t.match_ts_ms <= row.ts_ms + TAKER_TRADE_WINDOW_POST_MS
                }) {
                    if let Some(oid) = t.order_id.as_deref() {
                        by_oid.entry(oid).or_insert(t.match_ts_ms);
                    }
                }
                match by_oid.len() {
                    1 => {
                        let (oid, match_ts) = {
                            let (o, ts) = by_oid.iter().next().expect("len==1");
                            ((*o).to_string(), *ts)
                        };
                        // Ambiguity: could another pending taker attempt for the same
                        // token claim this same trade? If so, stay pending.
                        let disputed = taker_pendings.iter().any(|(aid, tok, ts)| {
                            aid != &row.attempt_id
                                && tok == &row.token_id
                                && match_ts >= ts - TAKER_TRADE_WINDOW_PRE_MS
                                && match_ts <= ts + TAKER_TRADE_WINDOW_POST_MS
                        });
                        // And an order_id already known or indexed cannot be the lost one
                        // (that order already has an owner).
                        let already_known = self.known_order_ids.contains(&oid);
                        if disputed || already_known {
                            continue;
                        }
                        // The durable attribution is CRITICAL: if the
                        // `order_index` is not written, the order would run with no persisted
                        // owner. The pending is kept;
                        // the next cycle retries.
                        if let Err(e) = self
                            .state
                            .record_order_index(
                                &oid,
                                Some(&row.attempt_id),
                                strat,
                                &row.cid,
                                &row.token_id,
                            )
                            .await
                        {
                            tracing::error!(
                                oid,
                                "executor: failed to record order_index (taker trades); pending kept: {e}"
                            );
                            continue;
                        }
                        self.known_order_ids.insert(oid.clone());
                        // The trade IS already evidence of execution. If the fill did not land
                        // (user_ws down), the position would be missing until the next backfill, and the
                        // strategy, with its state cleared by timeout, would re-emit the
                        // same edge (double position). So a Fill is synthesized per trade of the
                        // order_id: `exec_id = trade_id` deduplicates against the later user_ws
                        // or backfill delivery, and the broadcast recreates the state that was
                        // orphaned in the strategy.
                        // The journal is only resolved once the synthesized fills are DURABLE
                        // in `fills`. Emitting to the channel and resolving right away would leave a
                        // crash window (fill in flight + closed pending = position invisible
                        // forever). Fills already settled are not re-emitted; the missing
                        // ones are emitted, and the pending stays alive until the next
                        // cycle.
                        let mut missing = 0u32;
                        for t in list.iter().filter(|t| {
                            t.order_id.as_deref() == Some(oid.as_str())
                                && t.is_buy
                                && t.token_id == row.token_id
                                && t.match_ts_ms >= row.ts_ms - TAKER_TRADE_WINDOW_PRE_MS
                                && t.match_ts_ms <= row.ts_ms + TAKER_TRADE_WINDOW_POST_MS
                        }) {
                            match self.state.fill_exists_exec(&t.trade_id).await {
                                Ok(true) => continue, // already durable
                                Ok(false) | Err(_) => {
                                    self.emit_fill(Fill {
                                        market_cid: row.cid.clone(),
                                        token_id: row.token_id.clone(),
                                        side: FillSide::Buy,
                                        price: t.price,
                                        size: t.size,
                                        order_id: Some(oid.clone()),
                                        exec_id: Some(t.trade_id.clone()),
                                        local_order_id: Some(row.attempt_id.clone()),
                                        ts: chrono::DateTime::from_timestamp_millis(t.match_ts_ms)
                                            .unwrap_or_else(crate::time::now),
                                        mode: self.router.mode(),
                                        // Same as the backfill: an unknown neg_risk,
                                        // false is safe (it only affects manual redeem).
                                        neg_risk: false,
                                        strategy: strat,
                                        clear_at: row.clear_at.unwrap_or(t.match_ts_ms / 1000),
                                        outcome_label: None,
                                        hedge_id: None,
                                    });
                                    missing += 1;
                                }
                            }
                        }
                        if missing > 0 {
                            // Emitted to the channel but still without a durable record:
                            // the next cycle re-verifies them and closes them.
                            continue;
                        }
                        // The claim is kept until here: the position is already in
                        // `positions` (durable fills verified), so resolving the pending does not open
                        // a double-entry window.
                        info!(
                            attempt_id = %row.attempt_id, oid, strategy = %strat,
                            "executor: Unknown taker reconciled — durable fills verified"
                        );
                        self.resolve_pending_unknown(&row.attempt_id, "matched-trade")
                            .await;
                        self.unknown_attempts.remove(&row.attempt_id);
                    }
                    0 if now_ms - row.ts_ms >= TAKER_NO_TRADE_EXPIRY_MS => {
                        // "No trade on the tape" is not "never executed" until TWO queries
                        // in a row (separated by one reconciliation cycle) confirm it. A transient
                        // API lag turned a single empty query into a false negative
                        // that released the claim while the position was open.
                        let strikes = match self.unknown_attempts.get_mut(&row.attempt_id) {
                            Some(e) => {
                                e.no_trade_strikes = e.no_trade_strikes.saturating_add(1);
                                e.no_trade_strikes
                            }
                            None => 1,
                        };
                        if strikes < 2 {
                            debug!(
                                attempt_id = %row.attempt_id, strikes,
                                "executor: empty tape in the window — pending, awaiting confirmation"
                            );
                            continue;
                        }
                        self.ledger
                            .release_exposure(&row.cid, &row.token_id, cost, strat);
                        info!(
                            attempt_id = %row.attempt_id, strategy = %strat, cid = %row.cid,
                            "executor: Unknown taker with no trade after {} ms and {} confirmations — POST never arrived, claim released",
                            TAKER_NO_TRADE_EXPIRY_MS, strikes
                        );
                        self.resolve_pending_unknown(&row.attempt_id, "no-trade")
                            .await;
                        self.unknown_attempts.remove(&row.attempt_id);
                    }
                    _ => {
                        // ≥2 distinct order_ids in the window, or still within the
                        // long grace period → stays pending (fail-closed).
                    }
                }
                continue;
            }

            let Some(oid) = order_id else {
                continue;
            };

            match self.router.order_status(&oid).await {
                Some(RouterOrderStatus::Live { remaining }) => {
                    self.known_order_ids.insert(oid.clone());
                    if row.resting {
                        // The claim cannot be rebuilt blindly from the journal's metadata: a cancel
                        // pending carries a NULL `price_um`, and `remaining × 0` would release the exposure.
                        // `live_claim_evidence` looks up the exact open order on the destination (coherent
                        // oid, cid and token, side compatible with the journal), then the durable metadata.
                        // Without evidence the original claim is kept; with CONTRADICTORY evidence the
                        // pending is left open and the order is not adjudicated.
                        if !open_orders_fetched {
                            open_orders = self.router.open_gtd_orders().await;
                            open_orders_fetched = true;
                        }
                        let claim = match live_claim_evidence(
                            row,
                            &oid,
                            remaining,
                            strat,
                            open_orders.as_deref(),
                        ) {
                            LiveClaimEvidence::Determined(c) => c,
                            LiveClaimEvidence::KeepCost => cost,
                            LiveClaimEvidence::Contradictory => {
                                warn!(
                                    attempt_id = %row.attempt_id, oid,
                                    "executor: live order contradicts the pending — pending kept, not adjudicated"
                                );
                                continue;
                            }
                        };
                        let clear_at = row.clear_at.unwrap_or(0);
                        let key = limit_key(&row.cid, &row.token_id, strat);
                        self.active_gtd
                            .insert(key, (oid.clone(), claim, clear_at, strat));
                        if let Err(e) = self.state.record_gtd_order(&oid, &row.cid, strat).await {
                            tracing::error!(
                                oid,
                                "executor: failed to record gtd_orders (reconciler): {e}"
                            );
                        }
                    }
                    info!(attempt_id = %row.attempt_id, oid, "executor: Unknown reconciled — order live");
                    self.resolve_pending_unknown(&row.attempt_id, "accepted-live")
                        .await;
                    self.unknown_attempts.remove(&row.attempt_id);
                }
                Some(RouterOrderStatus::Gone { matched })
                    if matched > rust_decimal::Decimal::ZERO =>
                {
                    // The order executed (fully or partially) and left the book.
                    // `order_index` makes it attributable. A proven execution also needs the fill to
                    // land in `positions`: if the user_ws fill was lost, the strategy (state already
                    // cleared by timeout) could re-emit the same edge with the real position open.
                    // Fills are synthesized from the venue's trade tape (exec_id = trade_id, deduplicated
                    // against a later delivery). If the tape does not publish them yet (~90 s of
                    // latency), the pending stays open and is retried in the next cycle.
                    // The `clear_at` watchdog is the backstop if the venue never publishes them.
                    // The pending stays open in the meantime.
                    // The `clear_at` watchdog is the backstop.
                    // It fires if the venue never publishes the trades.
                    self.known_order_ids.insert(oid.clone());
                    if let Err(e) = self
                        .state
                        .record_order_index(
                            &oid,
                            Some(&row.attempt_id),
                            strat,
                            &row.cid,
                            &row.token_id,
                        )
                        .await
                    {
                        tracing::error!(
                            oid,
                            "executor: failed to record order_index (reconciler): {e}"
                        );
                    }
                    // Same rule as the taker branch:
                    // resolve only once the fills are durable.
                    let mut legs = 0u32;
                    let mut missing = 0u32;
                    if let Some(list) = self.router.trades_since(row.ts_ms / 1000 - 60).await {
                        for t in list.iter().filter(|t| {
                            t.order_id.as_deref() == Some(oid.as_str())
                                && t.token_id == row.token_id
                        }) {
                            legs += 1;
                            match self.state.fill_exists_exec(&t.trade_id).await {
                                Ok(true) => {} // already durable
                                Ok(false) | Err(_) => {
                                    self.emit_fill(Fill {
                                        market_cid: row.cid.clone(),
                                        token_id: row.token_id.clone(),
                                        side: if t.is_buy {
                                            FillSide::Buy
                                        } else {
                                            FillSide::Sell
                                        },
                                        price: t.price,
                                        size: t.size,
                                        order_id: Some(oid.clone()),
                                        exec_id: Some(t.trade_id.clone()),
                                        local_order_id: Some(row.attempt_id.clone()),
                                        ts: chrono::DateTime::from_timestamp_millis(t.match_ts_ms)
                                            .unwrap_or_else(crate::time::now),
                                        mode: self.router.mode(),
                                        neg_risk: false,
                                        strategy: strat,
                                        clear_at: row.clear_at.unwrap_or(t.match_ts_ms / 1000),
                                        outcome_label: None,
                                        hedge_id: None,
                                    });
                                    missing += 1;
                                }
                            }
                        }
                    }
                    if legs == 0 || missing > 0 {
                        // No trades published yet, or emitted but still without a durable record:
                        // pending; the claim is kept
                        // (the resync re-adds it from the mirror).
                        continue;
                    }
                    info!(attempt_id = %row.attempt_id, oid, matched = %matched, fills = legs,
                          "executor: Unknown reconciled — order executed, durable fills verified");
                    self.resolve_pending_unknown(&row.attempt_id, "matched")
                        .await;
                    self.unknown_attempts.remove(&row.attempt_id);
                }
                Some(RouterOrderStatus::Gone { .. }) | Some(RouterOrderStatus::NotFound) => {
                    // Definitive outcome WITHOUT execution: cancelled, expired or
                    // never accepted → release the owner's claim.
                    self.ledger
                        .release_exposure(&row.cid, &row.token_id, cost, strat);
                    let key = limit_key(&row.cid, &row.token_id, strat);
                    self.active_gtd.remove(&key);
                    if let Err(e) = self.state.delete_gtd_order(&oid).await {
                        tracing::error!(
                            oid,
                            "executor: failed to delete gtd_orders (reconciler): {e}"
                        );
                    }
                    info!(attempt_id = %row.attempt_id, oid,
                          "executor: Unknown reconciled — no live order, claim released");
                    self.resolve_pending_unknown(&row.attempt_id, "cancelled-not-live")
                        .await;
                    self.unknown_attempts.remove(&row.attempt_id);
                }
                None => {
                    // Indeterminate query → stays pending; the claim is kept
                    // (the resync re-adds it from the mirror).
                }
            }
        }
    }

    /// Quarantine↔journal link: a quarantined fill whose `order_id` is not
    /// indexed may belong to an `unk-*` attempt whose POST was lost: the fill IS the
    /// evidence that the order exists. If there is ONE single pending attempt for the same
    /// `(cid, token_id)`, `order_index` is written, and the `fill_sink` reconciler applies the fill
    /// with the correct owner on its next tick. The following `reconcile_unknowns` pass
    /// closes the attempt as `matched` via
    /// `order_status`.
    async fn link_quarantine_to_journal(&mut self) {
        let quarantine = match self.state.list_quarantined_fills().await {
            Ok(v) => v,
            Err(e) => {
                warn!("executor: link_quarantine failed reading quarantine: {e}");
                return;
            }
        };
        if quarantine.is_empty() {
            return;
        }
        let pending: Vec<(String, String, String, StrategyId)> = self
            .unknown_attempts
            .values()
            .filter(|a| a.attempt_id.starts_with("unk-"))
            .map(|a| {
                (
                    a.attempt_id.clone(),
                    a.cid.clone(),
                    a.token_id.clone(),
                    a.strategy,
                )
            })
            .collect();
        if pending.is_empty() {
            return;
        }
        for (_row_id, fill, _reason) in quarantine {
            let Some(oid) = fill.order_id.clone() else {
                continue;
            };
            // Already indexed → the fill_sink reconciler resolves it on its own.
            match self.state.order_index_lookup(&oid).await {
                Ok(Some(_)) => continue,
                Ok(None) => {}
                Err(e) => {
                    warn!(
                        oid,
                        "executor: link_quarantine: order_index_lookup failed: {e}"
                    );
                    continue;
                }
            }
            let mut candidates = pending
                .iter()
                .filter(|(_, cid, token, _)| *cid == fill.market_cid && *token == fill.token_id);
            let (Some(att), None) = (candidates.next(), candidates.next()) else {
                continue; // 0 (fill with no pending attempt) or ambiguous → conservative
            };
            let (attempt_id, cid, token, strat) = att.clone();
            info!(
                oid, attempt_id = %attempt_id, strategy = %strat,
                "executor: quarantined fill linked to a pending Unknown attempt"
            );
            if let Err(e) = self
                .state
                .record_order_index(&oid, Some(&attempt_id), strat, &cid, &token)
                .await
            {
                tracing::error!(oid, "executor: failed to link quarantine→journal: {e}");
            }
            self.known_order_ids.insert(oid);
        }
    }

    /// Unknown watchdog. Two rules over the durable journal (the
    /// source of truth, not the mirror: a manual `mark_unknown_resolved` may have closed a
    /// row without touching the mirror):
    ///
    /// 1. **Expired**: `clear_at + 65 s <= now` (same margin as the purge
    ///    of `active_gtd`). The bracket has already closed: the order cannot fill and,
    ///    if it was left resting, it expires on its own at the exchange. Without this rule an
    ///    Unknown taker with no fill (neither `order_id` nor a trade that proves it)
    ///    would block the bracket FOREVER via `has_unresolved_unknown`:
    ///    this was the missing exit from the conservative quarantine. It marks
    ///    `resolved="expired_no_reconcile"`, removes the attempt from the mirror,
    ///    releases the owner's claim, and clears `active_gtd`/`gtd_orders` if the
    ///    reconciler had managed to register the order.
    /// 2. **Old and not expired**: `ts_ms` more than 30 min ago while the bracket is still open:
    ///    automatic reconciliation is not catching it. Metric + warn to force
    ///    operator intervention; it is NOT resolved automatically
    ///    (no evidence, fail-closed).
    ///
    /// `qfill-*` rows (fill quarantine) are handled by the `fill_sink` reconciler,
    /// not by this pass. `clear_at=0/None` (no known end) never expires here, it only alerts.
    async fn watchdog_unknowns(&mut self) {
        let now_ts = crate::time::now().timestamp();
        let pending = match self.state.list_unresolved_unknowns().await {
            Ok(v) => v,
            Err(e) => {
                warn!("executor: watchdog_unknowns failed reading the journal: {e}");
                return;
            }
        };
        for row in pending {
            if row.attempt_id.starts_with("qfill-") {
                continue;
            }
            let expired = row
                .clear_at
                .map(|ca| ca > 0 && now_ts >= ca + 65)
                .unwrap_or(false);
            if !expired {
                if row.ts_ms < (now_ts * 1000) - 30 * 60 * 1000 {
                    metrics::counter!("hrf_unknown_watchdog_stale_total").increment(1);
                    warn!(
                        attempt_id = %row.attempt_id,
                        cid = %row.cid,
                        token = %row.token_id,
                        "executor: Unknown pending for >30 min, neither resolved nor expired — needs manual review"
                    );
                }
                continue;
            }
            let Ok(strat) = StrategyId::from_str(&row.strategy) else {
                continue;
            };
            let cost = crate::state::from_micros(row.cost_um);
            self.ledger
                .release_exposure(&row.cid, &row.token_id, cost, strat);
            self.unknown_attempts.remove(&row.attempt_id);
            let key = limit_key(&row.cid, &row.token_id, strat);
            if let Some((oid, _, _, _)) = self.active_gtd.remove(&key) {
                if let Err(e) = self.state.delete_gtd_order(&oid).await {
                    tracing::error!(
                        oid,
                        "executor: failed to delete gtd_orders (watchdog_unknowns): {e}"
                    );
                }
            }
            metrics::counter!("hrf_unknown_watchdog_expired_total").increment(1);
            warn!(
                attempt_id = %row.attempt_id,
                cid = %row.cid,
                token = %row.token_id,
                strategy = %strat,
                cost = %cost,
                "executor: Unknown expired (clear_at+65s, bracket closed) — claim released and journal marked resolved"
            );
            self.resolve_pending_unknown(&row.attempt_id, "expired_no_reconcile")
                .await;
        }
    }

    async fn handle(&mut self, intent: OrderIntent) {
        match intent.side {
            OrderSide::Cancel => {
                self.handle_cancel(&intent).await;
            }
            OrderSide::Sell { .. } => {
                self.handle_sell(intent).await;
            }
            OrderSide::Buy => {
                self.handle_buy(intent).await;
            }
        }
    }

    // ── Replay engine entry point ──────────────────────────────────────────────
    //
    // The `crate::replay` engine does NOT use `run()` (its `tokio::time::interval`
    // purge and its `queued_at.elapsed()` would be wall-clock time): it drives the executor
    // with messages and passes it the queue wait measured on the virtual clock. The
    // logic is the SAME as in `run()`: staleness gate → `handle`.

    /// Replica of the `run()` gate for a single intent: it discards Buy entries
    /// decided on an already dead book (`queue_wait` measured by the driver in
    /// replay time) and dispatches by `side` with `handle`: the same code
    /// as production, without loops or wall-clock timers.
    pub async fn dispatch_replay(&mut self, intent: OrderIntent, queue_wait: std::time::Duration) {
        metrics::histogram!("hrf_intent_queue_wait_seconds",
            "strategy" => intent.strategy.to_string())
        .record(queue_wait.as_secs_f64());
        if queue_wait > Self::MAX_INTENT_QUEUE_WAIT && Self::intent_has_buy(&intent) {
            metrics::counter!("hrf_intent_stale_dropped_total",
                "strategy" => intent.strategy.to_string())
            .increment(1);
            tracing::warn!(strategy = %intent.strategy, cid = %intent.market_cid,
                wait_ms = %queue_wait.as_millis(),
                "executor: Buy intent discarded — too long in the queue (replay)");
            self.emit_gate_rejected_intent(
                &intent,
                intent.size,
                intent.second_leg.as_deref().map(|l| l.size),
                "queue_stale",
            );
            return;
        }
        self.handle(intent).await;
    }

    /// Replica of the `global_kill_rx.changed()` branch of `run()` for the replay engine:
    /// the driver schedules this check as an event when the tape (or a test) triggers the kill
    /// — same predicate and same effect.
    pub async fn global_kill_check_replay(&mut self) {
        if self.risk.is_killed() && !self.risk.paper_disable_kill_switch_active() {
            self.cancel_all_gtds("global kill").await;
        }
    }

    // ── Buy ───

    /// Single entry point for buy orders, also used by the vertical tests
    /// in `tests/order_uncertainty.rs`.
    /// The main loop (`run`) still calls it via `handle()`.
    pub async fn handle_buy(&mut self, mut intent: OrderIntent) {
        let cid = &intent.market_cid;
        // The size the strategy marked when it emitted this intent, before the balance trim
        // below. It is what `GateRejected` must carry, so that the strategy undoes exactly its
        // contribution: a bigger undo would eat the `live` of a real order,
        // and a smaller one would leave a ghost residue.
        let marked_size = intent.size;
        let marked_second_size = intent.second_leg.as_ref().map(|l| l.size);

        // 1. Blacklist
        // Previously the `Err` arm only logged and FELL THROUGH (without
        // `return`), treating the SQLite failure as "not blacklisted":
        // fail-OPEN. A market blacklisted for some reason (settlement
        // broken, dispute, etc.) could be traded again as soon as the DB
        // hiccupped. Fail-closed: if it cannot be confirmed that the market is NOT
        // blacklisted, it is treated as if it were.
        match self.state.is_blacklisted(cid).await {
            Ok(true) => {
                debug!(cid, "executor: skip — market is blacklisted");
                self.emit_gate_rejected_intent(
                    &intent,
                    marked_size,
                    marked_second_size,
                    "blacklist",
                );
                return;
            }
            Err(e) => {
                tracing::error!(
                    cid,
                    "executor: error querying blacklist — fail-closed, skip: {e}"
                );
                self.emit_gate_rejected_intent(
                    &intent,
                    marked_size,
                    marked_second_size,
                    "blacklist_error",
                );
                return;
            }
            _ => {}
        }

        // 1b. Unknown blocking fix: if the durable journal has an `unresolved` entry for the
        // SAME `cid:token_id`, this attempt is NOT sent. A blind resend on a bracket whose
        // earlier order is in an unknown state is exactly the window that opens a double buy
        // without accounting coverage. The blocker is fail-closed:
        // if the DB query fails, we block as well.
        if self
            .block_by_unresolved_unknown(cid, &intent.token_id)
            .await
        {
            self.emit_gate_rejected_intent(
                &intent,
                marked_size,
                marked_second_size,
                "unresolved_unknown",
            );
            return;
        }

        // 2. Global kill switch (ignored in paper with paper_disable_kill_switch)
        if self.risk.is_killed() && !self.risk.paper_disable_kill_switch_active() {
            info!(strategy = %intent.strategy, "executor: skip — global kill switch active");
            self.emit_gate_rejected_intent(&intent, marked_size, marked_second_size, "kill_global");
            return;
        }

        // 3. Per-strategy kill switch (ignored in paper with paper_disable_kill_switch)
        if self.risk.is_strategy_killed(intent.strategy)
            && !self.risk.paper_disable_kill_switch_active()
        {
            info!(strategy = %intent.strategy, "executor: skip — strategy kill switch active");
            self.emit_gate_rejected_intent(
                &intent,
                marked_size,
                marked_second_size,
                "kill_strategy",
            );
            return;
        }

        // 3b. Entry cooldown after losses (CooldownPeriod + StoplossGuard
        //     from Freqtrade): scope `bracket` (a recent losing exit on this cid)
        //     or scope `strategy` (a streak of losing exits of the strategy).
        //     All zeros in [risk] = disabled. Risk gates emit `GateRejected` keyed to the
        //     intent (size/price), never a per-token `Rejected` (it would zero `live` with a real
        //     GTD still alive). Entries only: sells and cancels
        //     go through handle_sell/handle_cancel, not here.
        if self
            .risk
            .in_entry_cooldown(intent.strategy, cid, crate::time::now().timestamp_millis())
        {
            info!(strategy = %intent.strategy, cid, "executor: skip — entry cooldown after losses");
            self.emit_gate_rejected_intent(
                &intent,
                marked_size,
                marked_second_size,
                "entry_cooldown",
            );
            return;
        }

        // 4. Hot limits: per-order rate and maximum size (0 = no limit).
        let (max_per_hour, max_order_usdc) = self.risk.get_strategy_limits(intent.strategy);
        if max_per_hour > 0 && !self.risk.check_order_limit(intent.strategy, max_per_hour) {
            info!(strategy = %intent.strategy, max_per_hour, "executor: skip — rate limit exceeded");
            self.emit_gate_rejected_intent(&intent, marked_size, marked_second_size, "rate_limit");
            return;
        }
        let planned_cost = intent.price * intent.size;
        if max_order_usdc > rust_decimal::Decimal::ZERO && planned_cost > max_order_usdc {
            info!(strategy = %intent.strategy, cost = %planned_cost, max = %max_order_usdc,
                "executor: skip — order size exceeds max_order_usdc");
            self.emit_gate_rejected_intent(
                &intent,
                marked_size,
                marked_second_size,
                "max_order_usdc",
            );
            return;
        }
        // Trim to the available balance if it is below the planned cost.
        // If there is a second Buy leg (ARB: YES+NO), the combined price sets the trim
        // so that both legs fit the balance; otherwise the second order would be rejected.
        let second_buy_price = intent
            .second_leg
            .as_ref()
            .filter(|l| matches!(l.side, OrderSide::Buy))
            .map_or(rust_decimal::Decimal::ZERO, |l| l.price);
        let price_total = intent.price + second_buy_price;
        let order_cost = if let Some(bal) = *self.wallet_bal_rx.borrow() {
            let total_planned = price_total * intent.size;
            if bal < total_planned {
                if bal < dec!(1) {
                    info!(strategy = %intent.strategy, bal = %bal,
                        "executor: skip — insufficient balance for the minimum order");
                    self.emit_gate_rejected_intent(
                        &intent,
                        marked_size,
                        marked_second_size,
                        "balance",
                    );
                    return;
                }
                intent.size = bal / price_total;
                if let Some(ref mut leg) = intent.second_leg {
                    if matches!(leg.side, OrderSide::Buy) {
                        leg.size = intent.size;
                    }
                }
                info!(strategy = %intent.strategy, available = %bal,
                    new_size = %intent.size, "executor: size trimmed to the available balance");
                intent.price * intent.size
            } else {
                planned_cost
            }
        } else {
            planned_cost
        };

        // 5. Global exposure limit from the ledger: its claims cover
        //    in-flight GTDs and open positions, with no SQLite query per order.
        //    The periodic resync in the purge (30 s) corrects drift (FOK without fill, etc.).
        // If there is a second Buy leg (ARB YES+NO), its cost is ALSO committed
        // to the ledger (see `try_add_pair_exposure` below). Before, this check only saw
        // the `order_cost` of the first leg, leaving the other half of the pair invisible to
        // `max_total_exposure_usdc` and letting it be exceeded silently. It uses `l.size` (already
        // trimmed to the available balance above, like `intent.size`), not `intent.size`.
        let second_leg_cost = intent
            .second_leg
            .as_ref()
            .filter(|l| matches!(l.side, OrderSide::Buy))
            .map_or(rust_decimal::Decimal::ZERO, |l| l.price * l.size);
        let exposure = self.ledger.total_exposure();
        // The effective cap subtracts the strategy's reserved carve-out (0 if it has none)
        // (today only PCS-2 uses it; see risk.rs).
        let max_total = self.risk.effective_max_total_exposure(intent.strategy);
        // If this order is going to re-quote over an existing GTD of ours already live
        // on the same token (see the duplicate check in step 8 below), its earlier claim is
        // released BEFORE comparing against the global cap;
        // otherwise this check would count the same exposure TWICE (the old GTD, still live
        // in the ledger, plus the new one about to be reserved) and could reject an order that
        // actually fits. It is safe: either the cancel of the old one confirms below and it is
        // really released, or it fails and (for resting orders) the function returns without adding
        // the new one. A real double claim never remains through this path.
        // (For resting orders the new claim is not added on failure.)
        let dup_key = limit_key(&intent.market_cid, &intent.token_id, intent.strategy);
        let prior_claim = self
            .active_gtd
            .get(&dup_key)
            .map(|(_, cost, _, _)| *cost)
            .unwrap_or_default();
        let net_exposure = (exposure - prior_claim).max(rust_decimal::Decimal::ZERO);
        if net_exposure + order_cost + second_leg_cost > max_total {
            info!(strategy = %intent.strategy, cost = %(order_cost + second_leg_cost),
                exposure = %net_exposure, max = %max_total,
                "executor: skip — global exposure limit");
            self.emit_gate_rejected_intent(
                &intent,
                marked_size,
                marked_second_size,
                "exposure_cap",
            );
            return;
        }

        // 6. Register YES in order_map before the submit.
        let map_key = format!("{}:{}", intent.market_cid, intent.token_id);
        self.order_map.lock().insert(map_key, intent.strategy);

        // 8. Submit: simultaneous YES+NO hedge if there is a second Buy leg (post_orders in
        //    batch, WITHOUT real cross-atomicity; see the doc of submit_pair).
        //    For GTDs (resting) the simple path is used: the second leg waits for the async fill.
        let maybe_second = if let Some(ref l) = intent.second_leg {
            if matches!(l.side, OrderSide::Buy) && !intent.resting {
                intent.second_leg.take()
            } else {
                None
            }
        } else {
            None
        };

        if let Some(second) = maybe_second {
            // Atomic claim of the pair in the ledger: the net is evaluated with BOTH legs,
            // so a covered pair (net ≈ 0) is never blocked by the cap.
            let yes_cost = intent.price * intent.size;
            let no_cost = second.price * second.size;
            // REAL close of the bracket, not the synthetic GTD one.
            // ARB-15M5M: each leg is claimed under ITS `market_cid` (in the YES+NO pair of
            // `arb`, both agree and the behaviour is the same as always).
            let Some(net) = self.ledger.try_add_pair_exposure_per_market(
                (
                    cid,
                    intent.real_clear_at.unwrap_or(intent.clear_at),
                    &intent.token_id,
                    yes_cost,
                    intent.strategy,
                ),
                (
                    &second.market_cid,
                    second.real_clear_at.unwrap_or(second.clear_at),
                    &second.token_id,
                    no_cost,
                    second.strategy,
                ),
                self.risk.max_bracket_exposure(),
            ) else {
                metrics::counter!("hrf_bracket_cap_rejects_total", "strategy" => intent.strategy.to_string()).increment(1);
                info!(strategy = %intent.strategy, cid, "executor: skip — directional cap per bracket");
                // The whole pair was never sent: ack both legs.
                self.emit_gate_rejected_intent(
                    &intent,
                    marked_size,
                    marked_second_size,
                    "bracket_cap",
                );
                self.emit_gate_rejected(
                    &second,
                    marked_second_size.unwrap_or(second.size),
                    "bracket_cap",
                );
                return;
            };
            record_net_exposure(net);

            // Register NO in order_map before the submit.
            let no_key = format!("{}:{}", second.market_cid, second.token_id);
            self.order_map.lock().insert(no_key, second.strategy);

            // Unknown blocking fix: pre-POST journal entry per leg.
            // Each leg has its own attempt_id because they are independent submits
            // to the exchange. If the pending insert of ANY leg fails, the whole pair is NOT
            // sent (fail-closed).
            let yes_attempt_id = match self
                .record_pending_unknown(&intent, yes_cost, "pre-POST journal leg YES (pending)")
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!(
                        cid, token = %intent.token_id, leg = "YES", error = %e,
                        "executor: failed to write pending unknown YES \
                         pre-POST — pair not sent (fail-closed)",
                    );
                    self.emit_gate_rejected_intent(
                        &intent,
                        marked_size,
                        marked_second_size,
                        "journal_yes",
                    );
                    self.emit_gate_rejected(
                        &second,
                        marked_second_size.unwrap_or(second.size),
                        "journal_yes",
                    );
                    return;
                }
            };
            let no_attempt_id = match self
                .record_pending_unknown(&second, no_cost, "pre-POST journal leg NO (pending)")
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!(
                        cid, token = %second.token_id, leg = "NO", error = %e,
                        "executor: failed to write pending unknown NO \
                         pre-POST — pair not sent (fail-closed); \
                         YES already recorded with a pending mark",
                    );
                    // YES stays with its pending entry in the journal: the operator
                    // will have to reconcile it explicitly. Conservative.
                    self.emit_gate_rejected_intent(
                        &intent,
                        marked_size,
                        marked_second_size,
                        "journal_no",
                    );
                    self.emit_gate_rejected(
                        &second,
                        marked_second_size.unwrap_or(second.size),
                        "journal_no",
                    );
                    return;
                }
            };

            let mut req1 = self.build_request(&intent, false);
            let mut req2 = self.build_request(&second, false);
            // The local identity of each leg is its journal
            // `attempt_id` from before the POST: it is already persisted BEFORE the
            // submit, so a future fill can resolve its owner via `local_order_id → unknown_attempts`
            // even after a restart.
            req1.local_order_id = Some(yes_attempt_id.clone());
            req2.local_order_id = Some(no_attempt_id.clone());

            // Both legs as sent, pre-result.
            self.record_order_submit(&req1);
            self.record_order_submit(&req2);
            let submit_t0 = std::time::Instant::now();
            let (r_yes, r_no) = audit::submit_pair(&*self.router, req1, req2).await;
            metrics::histogram!("hrf_router_submit_seconds", "strategy" => intent.strategy.to_string())
                .record(submit_t0.elapsed().as_secs_f64());
            // OBSERVED outcome of each leg (accept/reject/fill/
            // unknown), counterpart of the OrderSubmit events emitted above.
            self.record_router_result(
                &intent.market_cid,
                &intent.token_id,
                Some(&yes_attempt_id),
                &r_yes,
            );
            self.record_router_result(
                &second.market_cid,
                &second.token_id,
                Some(&no_attempt_id),
                &r_no,
            );

            if r_yes.order_id.is_some() || r_no.order_id.is_some() {
                self.risk.record_order(intent.strategy);
            }
            // known_order_ids: user_ws uses them to match `maker_orders` of a
            // live maker fill with our order (the taker's order_id is not ours).
            // order_index: durable order_id→strategy attribution that survives the
            // restart (order_map is memory only).
            if let Some(oid) = &r_yes.order_id {
                self.known_order_ids.insert(oid.clone());
                if let Err(e) = self
                    .state
                    .record_order_index(
                        oid,
                        Some(&yes_attempt_id),
                        intent.strategy,
                        &intent.market_cid,
                        &intent.token_id,
                    )
                    .await
                {
                    tracing::error!(
                        oid,
                        "executor: failed to record order_index (YES leg): {e}"
                    );
                }
            }
            if let Some(oid) = &r_no.order_id {
                self.known_order_ids.insert(oid.clone());
                if let Err(e) = self
                    .state
                    .record_order_index(
                        oid,
                        Some(&no_attempt_id),
                        second.strategy,
                        &second.market_cid,
                        &second.token_id,
                    )
                    .await
                {
                    tracing::error!(
                        oid,
                        "executor: failed to record order_index (NO leg): {e}"
                    );
                }
            }

            if r_yes.rejected {
                // post_orders sends the pair in a single POST, but each FOK is matched
                // INDEPENDENTLY against its own book: there is no cross-atomicity. If NO
                // was not rejected, it may have filled anyway; releasing its claim here
                // would leave real exposure without coverage in the ledger.
                // Only the claim of the side that was actually rejected is released.
                self.ledger
                    .release_exposure(cid, &intent.token_id, yes_cost, intent.strategy);
                if r_no.rejected {
                    self.ledger
                        .release_exposure(&second.market_cid, &second.token_id, no_cost, second.strategy);
                } else {
                    // If NO returned Unknown, its journal pending stays open (correct, since it is not
                    // definitive), and its in-memory mirror must ALSO be registered: the `return`
                    // below makes the `r_no.unknown` branch at the end unreachable, since it only runs
                    // when YES was accepted. Without the mirror, the resync would drop the claim of a
                    // leg that may still be live on the CLOB, and the quarantine would not link its fill
                    // to its attempt.
                    if r_no.unknown {
                        self.register_unknown_attempt(
                            &second,
                            no_cost,
                            r_no.reason.as_deref().unwrap_or(""),
                            &no_attempt_id,
                        );
                        metrics::counter!(
                            "hrf_executor_unknown_orders_total",
                            "strategy" => second.strategy.to_string(),
                            "resting"  => "false",
                        )
                        .increment(1);
                    }
                    warn!(strategy = %intent.strategy, cid,
                          "executor: YES rejected but NO not; NO cannot be left orphaned, exposure kept");
                }
                // Unknown blocking fix: journal pendings are closed with
                // `rejected`. The pending is only closed if NO was ALSO rejected
                // (otherwise NO may still be live; it is not definitive).
                self.resolve_pending_unknown(&yes_attempt_id, "rejected")
                    .await;
                if r_no.rejected {
                    self.resolve_pending_unknown(&no_attempt_id, "rejected")
                        .await;
                }
                info!(strategy = %intent.strategy, cid, reason = ?r_yes.reason,
                      "executor: hedge YES rejected");
                self.emit_fill(Fill {
                    market_cid: intent.market_cid.clone(),
                    token_id: intent.token_id.clone(),
                    side: FillSide::Rejected,
                    price: rust_decimal::Decimal::ZERO,
                    size: rust_decimal::Decimal::ZERO,
                    order_id: r_yes.order_id,
                    ts: crate::time::now(),
                    mode: self.router.mode(),
                    neg_risk: intent.neg_risk,
                    strategy: intent.strategy,
                    clear_at: intent.clear_at,
                    outcome_label: None,
                    hedge_id: None,
                    exec_id: None,
                    local_order_id: Some(yes_attempt_id.clone()),
                });
                return;
            }
            // Integrated close: see the analogous comment in the simple path of `handle_buy`.
            // YES with a lost response: we keep its claim and record a local mark.
            // NO is only released if it ALSO was definitively rejected (the `r_no.rejected`
            // branch below). If NO is `unknown` or `accepted`, its claim is kept
            // (no cross-atomicity between the legs).
            if r_yes.unknown {
                // Unknown blocking fix: the `attempt_id` was already recorded pre-POST in the journal;
                // here we only update the in-memory mirror.
                self.register_unknown_attempt(
                    &intent,
                    yes_cost,
                    r_yes.reason.as_deref().unwrap_or(""),
                    &yes_attempt_id,
                );
                metrics::counter!(
                    "hrf_executor_unknown_orders_total",
                    "strategy" => intent.strategy.to_string(),
                    "resting"  => "false",
                )
                .increment(1);
                warn!(
                    strategy = %intent.strategy,
                    cid,
                    token = %intent.token_id,
                    leg = "YES",
                    cost = %yes_cost,
                    attempt_id = %yes_attempt_id,
                    reason = ?r_yes.reason,
                    "executor: ARB pair — leg YES POST without response (Unknown), exposure kept",
                );
                // NO may remain Unknown or Accepted: if Accepted, its order_id was already
                // recorded above (not released); if Unknown, its pending stays open and its mirror is
                // registered in the `r_no.unknown` branch just below. Do not leave this to the
                // final branch: this `return` makes it unreachable.
                if r_no.rejected {
                    // YES Unknown, NO rejected → we keep YES (it may be live on the CLOB),
                    // and release NO (definitive rejection).
                    self.ledger
                        .release_exposure(&second.market_cid, &second.token_id, no_cost, second.strategy);
                    // Unknown blocking fix: NO's pending is closed with `rejected`
                    // (definitive rejection).
                    self.resolve_pending_unknown(&no_attempt_id, "rejected")
                        .await;
                } else if r_no.unknown {
                    // YES and NO both Unknown: NO's pending stays open
                    // (correct, since it is not a definitive outcome), and its in-memory mirror is
                    // registered here. Without it, the first resync would drop the claim of a leg
                    // that may be live, and the quarantine could not link its late fill.
                    self.register_unknown_attempt(
                        &second,
                        no_cost,
                        r_no.reason.as_deref().unwrap_or(""),
                        &no_attempt_id,
                    );
                    metrics::counter!(
                        "hrf_executor_unknown_orders_total",
                        "strategy" => second.strategy.to_string(),
                        "resting"  => "false",
                    )
                    .increment(1);
                    warn!(
                        strategy = %intent.strategy,
                        cid,
                        token = %second.token_id,
                        leg = "NO",
                        cost = %no_cost,
                        attempt_id = %no_attempt_id,
                        reason = ?r_no.reason,
                        "executor: ARB pair — leg NO POST without response (Unknown), exposure kept",
                    );
                } else {
                    // YES Unknown, NO accepted → both live; we close
                    // NO's pending with `accepted`.
                    self.resolve_pending_unknown(&no_attempt_id, "accepted")
                        .await;
                }
                return;
            }
            // YES is neither rejected nor unknown → it is either accepted or unknown
            // (the unknown branch already returned above). YES's pending is
            // closed with `accepted`.
            self.resolve_pending_unknown(&yes_attempt_id, "accepted")
                .await;
            if r_no.rejected {
                self.ledger
                    .release_exposure(&second.market_cid, &second.token_id, no_cost, second.strategy);
                // Unknown blocking fix: NO pending → rejected.
                self.resolve_pending_unknown(&no_attempt_id, "rejected")
                    .await;
                warn!(strategy = %intent.strategy, cid, "executor: hedge NO rejected — YES orphaned");
                self.emit_fill(Fill {
                    market_cid: second.market_cid.clone(),
                    token_id: second.token_id.clone(),
                    side: FillSide::Rejected,
                    price: rust_decimal::Decimal::ZERO,
                    size: rust_decimal::Decimal::ZERO,
                    order_id: r_no.order_id,
                    ts: crate::time::now(),
                    mode: self.router.mode(),
                    neg_risk: second.neg_risk,
                    strategy: second.strategy,
                    clear_at: second.clear_at,
                    outcome_label: None,
                    hedge_id: None,
                    exec_id: None,
                    local_order_id: Some(no_attempt_id.clone()),
                });
            }
            // Unknown blocking fix (integrated close): NO with a lost response, YES already
            // handled above (it is neither rejected nor unknown, so it is accepted or unknown).
            // If YES is already accepted, we keep NO (it may also have been accepted) and
            // record a local mark.
            if r_no.unknown {
                self.register_unknown_attempt(
                    &second,
                    no_cost,
                    r_no.reason.as_deref().unwrap_or(""),
                    &no_attempt_id,
                );
                metrics::counter!(
                    "hrf_executor_unknown_orders_total",
                    "strategy" => second.strategy.to_string(),
                    "resting"  => "false",
                )
                .increment(1);
                warn!(
                    strategy = %intent.strategy,
                    cid,
                    token = %second.token_id,
                    leg = "NO",
                    cost = %no_cost,
                    attempt_id = %no_attempt_id,
                    reason = ?r_no.reason,
                    "executor: ARB pair — leg NO POST without response (Unknown), exposure kept",
                );
            }
            // Fills arrive asynchronously via user_ws; no emit_fill here.
            return;
        }

        // 8 (simple path: no second Buy leg, or GTD).
        //
        // Anti-duplicate: if a GTD is already live on this token, cancel it
        // BEFORE sending the new order, whatever its kind.
        //
        // This used to run only for NO-resting intents (FOK retries). For GTDs
        // (every maker's scenario: PM, MM, TC, PCS) the cancellation of the earlier order happened
        // AFTER the submit, so during the whole `submit → cancel` window there were TWO live GTDs on
        // the same token, and the book could fill both. The strategy's size cap meant nothing: PM, with a cap of
        // 3 shares, accumulated 33 net shares in one leg and chained 6 fills of 3 in 4 seconds
        // (calibration of 14 Jul; the late anti-duplicate check fired 1,509 times in 30 min).
        // The warn itself said "cancelling the earlier one before reposting" while the code did it afterwards.
        // `dup_key` was already computed above (step 5, to subtract its
        // claim from the exposure check): reuse the same key.
        if let Some((old_oid, old_cost, old_clear, old_strat)) = self.active_gtd.remove(&dup_key) {
            self.record_cancel_request(&intent.market_cid, &intent.token_id, &old_oid);
            let cancel_res = self.router.cancel(&intent.market_cid, &old_oid).await;
            self.record_cancel_result(&intent.market_cid, &intent.token_id, &old_oid, &cancel_res);
            match cancel_res {
                Ok(_) => {
                    self.ledger.release_exposure(
                        &intent.market_cid,
                        &intent.token_id,
                        old_cost,
                        old_strat,
                    );
                    if let Err(e) = self.state.delete_gtd_order(&old_oid).await {
                        tracing::error!(
                            old_oid,
                            "executor: failed to delete gtd_orders: {e}"
                        );
                    }
                    debug!(old_oid, token = %intent.token_id,
                           "executor: previous GTD cancelled before the new order (anti-duplicate)");
                    // The strategy has to find out that this order died,
                    // even though it did not ask for the cancel itself.
                    self.emit_cancel_ack(
                        &intent.market_cid,
                        &intent.token_id,
                        old_strat,
                        Some(old_oid.clone()),
                        intent.neg_risk,
                        old_clear,
                    );
                }
                Err(e) => {
                    // The oldest GTD may still be live on the CLOB → keep its claim.
                    // If the new order is another GTD, do NOT post on top: stacking two live orders on
                    // the same token is exactly what this guard prevents.
                    // A non-resting order (FOK retry) continues: it either executes or dies, and it never
                    // sits in the book competing with itself.
                    warn!(
                        old_oid,
                        resting = intent.resting,
                        "executor: anti-duplicate cancel failed — claim kept: {e}"
                    );
                    self.active_gtd
                        .insert(dup_key, (old_oid, old_cost, old_clear, old_strat));
                    if intent.resting {
                        metrics::counter!("hrf_executor_gtd_repost_aborted_total",
                                          "strategy" => intent.strategy.to_string())
                        .increment(1);
                        // The new intent was never posted (the oldest GTD
                        // is still live, which is what prevents the repost). The strategy
                        // only undoes the mark of THIS intent.
                        self.emit_gate_rejected_intent(
                            &intent,
                            marked_size,
                            marked_second_size,
                            "repost_aborted",
                        );
                        return;
                    }
                }
            }
        }
        // Ledger claim before the submit; released if the router rejects it.
        // The REAL bracket close (not the synthetic GTD one used by PCS-2) sets the ledger
        // entry's `clear_at`, as already done for fills.
        let Some(net) = self.ledger.try_add_exposure(
            cid,
            intent.real_clear_at.unwrap_or(intent.clear_at),
            &intent.token_id,
            order_cost,
            self.risk.max_bracket_exposure(),
            intent.strategy,
        ) else {
            metrics::counter!("hrf_bracket_cap_rejects_total", "strategy" => intent.strategy.to_string()).increment(1);
            info!(strategy = %intent.strategy, cid, cost = %order_cost,
                  "executor: skip — directional cap per bracket");
            self.emit_gate_rejected_intent(&intent, marked_size, marked_second_size, "bracket_cap");
            return;
        };
        record_net_exposure(net);
        let mut req = self.build_request(&intent, false);
        // Unknown blocking fix: pre-POST journal. If the insert fails, we do NOT send
        // (fail-closed).
        let attempt_id = match self
            .record_pending_unknown(
                &intent,
                order_cost,
                "pre-POST journal (awaiting response)",
            )
            .await
        {
            Ok(id) => id,
            Err(e) => {
                tracing::error!(
                    cid, token = %intent.token_id, error = %e,
                    "executor: failed to write pending unknown to the journal \
                     pre-POST — submit not sent (fail-closed)",
                );
                metrics::counter!("hrf_executor_unknown_journal_pre_post_failures_total",)
                    .increment(1);
                self.emit_gate_rejected_intent(&intent, marked_size, marked_second_size, "journal");
                return;
            }
        };
        // Integrated close: the `attempt_id` already persisted in the journal is
        // the LOCAL identity of the order: it goes in the request so that the fill
        // brings it back (sim) or it stays resolvable via `order_index` (live).
        req.local_order_id = Some(attempt_id.clone());
        self.record_order_submit(&req);
        let submit_t0 = std::time::Instant::now();
        let result = audit::submit(&*self.router, req).await;
        metrics::histogram!("hrf_router_submit_seconds", "strategy" => intent.strategy.to_string())
            .record(submit_t0.elapsed().as_secs_f64());
        // Observed outcome of the submit (accept/reject/fill/unknown).
        self.record_router_result(
            &intent.market_cid,
            &intent.token_id,
            Some(&attempt_id),
            &result,
        );

        if result.rejected {
            self.ledger
                .release_exposure(cid, &intent.token_id, order_cost, intent.strategy);
            // A router-rejected order ALSO consumes the strategy's rate limit: before,
            // `record_order` was only called on the success path, so a retry loop against a
            // book that systematically rejects was not bounded by
            // `max_orders_per_hour`.
            self.risk.record_order(intent.strategy);
            // Unknown blocking fix: the journal pending is closed with
            // `rejected` (it is not Unknown).
            self.resolve_pending_unknown(&attempt_id, "rejected").await;
            info!(
                strategy = %intent.strategy,
                cid,
                reason = ?result.reason,
                "executor: order rejected by the router"
            );
            self.emit_fill(Fill {
                market_cid: intent.market_cid.clone(),
                token_id: intent.token_id.clone(),
                side: FillSide::Rejected,
                price: rust_decimal::Decimal::ZERO,
                size: rust_decimal::Decimal::ZERO,
                order_id: result.order_id.clone(),
                ts: crate::time::now(),
                mode: self.router.mode(),
                neg_risk: intent.neg_risk,
                strategy: intent.strategy,
                clear_at: intent.clear_at,
                outcome_label: None,
                hedge_id: None,
                exec_id: None,
                local_order_id: Some(attempt_id.clone()),
            });
            return;
        }

        // Integrated close: a POST with a lost response. The exchange may have accepted
        // or rejected the order, but the `RouterResult` has no way to know which. The exposure
        // is kept (the claim is not released: if the exchange accepted, a double buy would
        // be left without accounting coverage), and a durable local uncertainty mark
        // (`attempt_id`) is recorded so that a blind resend of the same intent can detect it
        // and does NOT call the router again.
        //
        // For GTD (resting) without a known `order_id` we cannot cancel via the exchange,
        // so the claim stays at its live cost in the ledger until the next `purge_and_resync`
        // (clear_at+65). Conservative: if the order filled on the CLOB, the position resync picks
        // it up from the DB and releases the claim via `fill_sink`. If it did not fill,
        // it expires and is purged.
        //
        // For a taker (not resting) the exposure is kept, but without an `order_id` there is no
        // way to cancel or reconcile the order: it stays a "potentially live" position in the
        // ledger until the bracket closes. The metric
        // `hrf_executor_unknown_orders_total` lets an operator
        // detect the scenario and decide manually.
        if result.unknown {
            // Unknown blocking fix: the `attempt_id` was already recorded pre-POST in the durable
            // journal; here we only update the in-memory mirror (used for inspection and tests).
            self.register_unknown_attempt(
                &intent,
                order_cost,
                result.reason.as_deref().unwrap_or(""),
                &attempt_id,
            );
            metrics::counter!(
                "hrf_executor_unknown_orders_total",
                "strategy" => intent.strategy.to_string(),
                "resting"  => intent.resting.to_string(),
            )
            .increment(1);
            warn!(
                strategy = %intent.strategy,
                cid,
                token = %intent.token_id,
                resting = intent.resting,
                cost    = %order_cost,
                attempt_id = %attempt_id,
                reason  = ?result.reason,
                "executor: POST without a response — exposure kept, recorded as Unknown (fail-closed)",
            );
            // We do not emit `Fill::Rejected`, `Fill::Cancelled` or `Fill::Buy`:
            // the exchange may have done ANY of the three. The mark in the durable journal
            // (`unknown_attempts`) is the ONLY persistent trace until an explicit reconciliation.
            return;
        }

        // Unknown blocking fix: the POST accepted the order (with or without
        // a fill). The pending entry of the journal is closed with `accepted`.
        self.resolve_pending_unknown(&attempt_id, "accepted").await;

        // Register the order_id for matching maker_orders in user_ws and in the
        // durable index (fill attribution after a restart; order_map lives in memory only).
        if let Some(oid) = &result.order_id {
            self.known_order_ids.insert(oid.clone());
            if let Err(e) = self
                .state
                .record_order_index(
                    oid,
                    Some(&attempt_id),
                    intent.strategy,
                    &intent.market_cid,
                    &intent.token_id,
                )
                .await
            {
                tracing::error!(oid, "executor: failed to record order_index: {e}");
            }
        }

        // 9a. Register order_id + cost for GTD orders.
        // The earlier GTD was already cancelled BEFORE the submit (step 8), so here
        // no live one can remain on this token: only the new one has to be registered.
        if intent.resting {
            if let Some(oid) = &result.order_id {
                let key = limit_key(&intent.market_cid, &intent.token_id, intent.strategy);
                self.active_gtd.insert(
                    key,
                    (oid.clone(), order_cost, intent.clear_at, intent.strategy),
                );
                // Persist order_id → strategy. Startup reconciliation
                // uses it so that restored GTDs are not attributed to Arb.
                if let Err(e) = self
                    .state
                    .record_gtd_order(oid, &intent.market_cid, intent.strategy)
                    .await
                {
                    tracing::error!(oid, "executor: failed to record gtd_orders: {e}");
                }
            }
        }

        // 9b. Immediate fill (paper).
        let has_fill = result.fill.is_some();
        let has_order = result.order_id.is_some();

        if has_fill || has_order {
            self.risk.record_order(intent.strategy);
        }

        if let Some(mut fill) = result.fill {
            // `intent.clear_at` may be a synthetic value (a GTD with its own expiry, e.g. PCS-2)
            // that does not match the market's real close. `real_clear_at`, when present, is the
            // value to persist: `src/settlement/orphan.rs::list_expired_unresolved` relies on it
            // to know when a position is ready to redeem.
            fill.clear_at = intent.real_clear_at.unwrap_or(intent.clear_at);
            info!(
                strategy = %intent.strategy, cid,
                token = %intent.token_id, price = %fill.price, size = %fill.size, mode = %fill.mode,
                "executor: fill"
            );
            self.emit_fill(fill);
        }

        // 10. Second hedge leg for GTDs (resting): sent after acceptance.
        if has_fill || (has_order && !intent.resting) {
            if let Some(second) = intent.second_leg {
                let no_key = format!("{}:{}", second.market_cid, second.token_id);
                self.order_map.lock().insert(no_key, second.strategy);
                // Coverage leg claim: it reduces the net, so with the asset cap it is never
                // blocked if it truly covers the first one.
                let second_cost = second.price * second.size;
                // The REAL bracket close, not the synthetic GTD one.
                if let Some(net) = self.ledger.try_add_exposure(
                    cid,
                    second.real_clear_at.unwrap_or(second.clear_at),
                    &second.token_id,
                    second_cost,
                    self.risk.max_bracket_exposure(),
                    second.strategy,
                ) {
                    record_net_exposure(net);
                } else {
                    metrics::counter!("hrf_bracket_cap_rejects_total", "strategy" => second.strategy.to_string()).increment(1);
                    warn!(strategy = %intent.strategy, cid,
                          "executor: skip second leg — directional cap per bracket");
                    return;
                }
                // This path (resting with a second Buy leg sent after the fill: no emitter today,
                // but reachable) must also be journaled pre-POST: a `submit` with a lost response and no
                // mark would keep the claim WITHOUT a journal entry, leaving no bracket blocker and no
                // possible reconciliation.
                // If writing the pending fails, the claim just taken is released and the
                // leg is not sent (fail-closed).
                let attempt2 = match self
                    .record_pending_unknown(
                        &second,
                        second_cost,
                        "pre-POST journal second leg (pending)",
                    )
                    .await
                {
                    Ok(id) => id,
                    Err(e) => {
                        tracing::error!(
                            cid, token = %second.token_id, error = %e,
                            "executor: failed to write pending unknown for the \
                             second leg — not sent, claim released (fail-closed)",
                        );
                        self.ledger.release_exposure(
                            cid,
                            &second.token_id,
                            second_cost,
                            second.strategy,
                        );
                        return;
                    }
                };
                let mut req2 = self.build_request(&second, false);
                req2.local_order_id = Some(attempt2.clone());
                self.record_order_submit(&req2);
                let r2 = audit::submit(&*self.router, req2).await;
                // Observed outcome of the second leg.
                self.record_router_result(
                    &second.market_cid,
                    &second.token_id,
                    Some(&attempt2),
                    &r2,
                );
                if let Some(mut fill2) = r2.fill {
                    // No strategy today uses a `second_leg` with a `real_clear_at` that differs
                    // from `clear_at` (only ARB emits a second_leg and does not adopt the field).
                    // It gets the same treatment as the first leg, for consistency, so the invariant
                    // is not broken if a future strategy combines both.
                    fill2.clear_at = second.real_clear_at.unwrap_or(second.clear_at);
                    // Immediate fill: the order is closed, so pending → accepted.
                    self.resolve_pending_unknown(&attempt2, "accepted").await;
                    self.emit_fill(fill2);
                } else if r2.rejected {
                    self.ledger.release_exposure(
                        cid,
                        &second.token_id,
                        second_cost,
                        second.strategy,
                    );
                    self.resolve_pending_unknown(&attempt2, "rejected").await;
                    warn!(strategy = %intent.strategy, cid,
                          "executor: second leg rejected — first leg orphaned");
                    self.emit_fill(Fill {
                        market_cid: second.market_cid.clone(),
                        token_id: second.token_id.clone(),
                        side: FillSide::Rejected,
                        price: rust_decimal::Decimal::ZERO,
                        size: rust_decimal::Decimal::ZERO,
                        order_id: r2.order_id.clone(),
                        ts: crate::time::now(),
                        mode: self.router.mode(),
                        neg_risk: second.neg_risk,
                        strategy: second.strategy,
                        clear_at: second.clear_at,
                        outcome_label: None,
                        hedge_id: None,
                        exec_id: None,
                        local_order_id: Some(attempt2.clone()),
                    });
                } else if r2.unknown {
                    // Lost response: keep the claim, record the mirror and leave the
                    // pending OPEN, and the reconciler resolves it with the destination's evidence.
                    self.register_unknown_attempt(
                        &second,
                        second_cost,
                        r2.reason.as_deref().unwrap_or(""),
                        &attempt2,
                    );
                    metrics::counter!(
                        "hrf_executor_unknown_orders_total",
                        "strategy" => second.strategy.to_string(),
                        "resting"  => "false",
                    )
                    .increment(1);
                    warn!(
                        strategy = %intent.strategy, cid,
                        token = %second.token_id,
                        attempt_id = %attempt2,
                        reason = ?r2.reason,
                        "executor: second leg POST without a response (Unknown) — exposure kept",
                    );
                } else {
                    // Accepted without fill: pending → accepted and, if there is an
                    // order_id, it is indexed for durable attribution.
                    self.resolve_pending_unknown(&attempt2, "accepted").await;
                    if let Some(oid) = &r2.order_id {
                        self.known_order_ids.insert(oid.clone());
                        if let Err(e) = self
                            .state
                            .record_order_index(
                                oid,
                                Some(&attempt2),
                                second.strategy,
                                &second.market_cid,
                                &second.token_id,
                            )
                            .await
                        {
                            tracing::error!(
                                oid,
                                "executor: failed to record order_index (second leg): {e}"
                            );
                        }
                    }
                }
            }
        }
    }

    // ── Sell ─────────────────────────────────────────────────────────────────

    async fn handle_sell(&mut self, intent: OrderIntent) {
        let mut req = self.build_request(&intent, true);
        // Integrated close: own local id (sells are not journaled:
        // they do not reserve cash). It reaches the Fill for traceability and to order_index.
        let sell_local_id = next_local_order_id(&self.attempt_counter);
        req.local_order_id = Some(sell_local_id.clone());
        self.record_order_submit(&req);
        let result = audit::submit(&*self.router, req).await;
        // Observed outcome of the sell submit.
        self.record_router_result(
            &intent.market_cid,
            &intent.token_id,
            Some(&sell_local_id),
            &result,
        );

        // Integrated close: index EVERY accepted order that has an order_id,
        // resting and taker, so that a live fill after a restart resolves its owner
        // without depending on the in-memory order_map.
        if let Some(oid) = &result.order_id {
            self.known_order_ids.insert(oid.clone());
            if let Err(e) = self
                .state
                .record_order_index(
                    oid,
                    Some(&sell_local_id),
                    intent.strategy,
                    &intent.market_cid,
                    &intent.token_id,
                )
                .await
            {
                tracing::error!(oid, "executor: failed to record order_index (sell): {e}");
            }
        }

        if let Some(mut fill) = result.fill {
            // Same clear_at/real_clear_at treatment as in handle_buy.
            fill.clear_at = intent.real_clear_at.unwrap_or(intent.clear_at);
            // Per-bracket cooldown (CooldownPeriod): if this sell closes at a loss, mark the cid
            // so that re-entries are blocked for `entry_cooldown_secs`. The `positions` row still
            // reflects the previous cost: the fill is not applied until fill_sink processes it
            // from the channel. So the sign of the exit is `fill.price < cost_basis/size`, the same
            // as the `pnl = proceeds − prorated cost` of `apply_fill`. Without a position (a naked
            // TC short, an orphan sell) there is no attributable loss, so it is a no-op.
            // This path only sees immediate (taker) fills.
            // Resting sell fills arrive via user_ws → fill_sink, which arms the same cooldown
            // from the `pnl` of `apply_fill` (idempotent).
            // (A resting fill is handled in fill_sink.)
            match self
                .state
                .get_position(&intent.token_id, &intent.strategy.to_string())
                .await
            {
                Ok(Some(pos)) if pos.size > rust_decimal::Decimal::ZERO => {
                    let cost_per_share = pos.cost_basis / pos.size;
                    if cost_per_share > rust_decimal::Decimal::ZERO && fill.price < cost_per_share {
                        self.risk.record_bracket_loss(&intent.market_cid);
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    // Deliberate fail-open: without reading the cost there is no cooldown,
                    // so behaviour stays as it was before this protection existed (blocking
                    // blindly would be worse).
                    warn!(
                        cid = %intent.market_cid, token = %intent.token_id,
                        "executor: could not read the cost for the bracket cooldown — skipped: {e}"
                    );
                }
            }
            info!(
                strategy = %intent.strategy,
                cid   = %intent.market_cid,
                token = %intent.token_id,
                price = %fill.price,
                size  = %fill.size,
                "executor: fill sell"
            );
            self.emit_fill(fill);
            return;
        }

        // For the sell side, the router rejection is the TERMINAL event of the order, and only a
        // `Fill` communicates it to the strategy. PM and RF set `unwind_sent` when they emit their
        // unwind sell (taker or maker), and that flag is only released by a terminal event. Without the
        // `Rejected`, a rejected ask/dump left the losing leg mute forever, and the position rolled to
        // settlement with no management.
        //
        // `unknown` is NOT translated to Rejected: the sell may still be live on the CLOB, and a blind
        // resend would be a double sell (oversell). It is only alerted on. If it fills, the fill arrives via
        // user_ws and the quarantine holds it until it can be attributed. If it never existed, the position
        // stays and strategies with a time deadline (NE/NC/OM/DB/ARB) retry.
        // (No `Rejected` is emitted in that case.)
        if result.rejected {
            warn!(
                strategy = %intent.strategy,
                cid   = %intent.market_cid,
                token = %intent.token_id,
                reason = ?result.reason,
                "executor: sell rejected by the router — emitting terminal Rejected"
            );
            self.emit_fill(Fill {
                market_cid: intent.market_cid.clone(),
                token_id: intent.token_id.clone(),
                side: FillSide::Rejected,
                price: rust_decimal::Decimal::ZERO,
                size: rust_decimal::Decimal::ZERO,
                order_id: result.order_id.clone(),
                ts: crate::time::now(),
                mode: self.router.mode(),
                neg_risk: intent.neg_risk,
                strategy: intent.strategy,
                clear_at: intent.real_clear_at.unwrap_or(intent.clear_at),
                outcome_label: None,
                hedge_id: None,
                exec_id: None,
                local_order_id: Some(sell_local_id.clone()),
            });
            return;
        }
        if result.unknown {
            // Separate metric (not `hrf_executor_unknown_orders_total`): sells carry no
            // `attempt_id` or claim, so mixing them with the buy series would blur the journal alerts.
            metrics::counter!(
                "hrf_executor_unknown_sell_orders_total",
                "strategy" => intent.strategy.to_string(),
                "resting"  => intent.resting.to_string(),
            )
            .increment(1);
            warn!(
                strategy = %intent.strategy,
                cid   = %intent.market_cid,
                token = %intent.token_id,
                resting = intent.resting,
                reason = ?result.reason,
                "executor: sell POST without a response (Unknown) — may be live on the CLOB; \
                 NOT retried blindly and no state released (if it fills, the quarantine holds it)"
            );
        }

        // The GTD maker ask (resting) is registered the same way as on the buy path
        // (see handle_buy above): without this the order cannot be cancelled, does not
        // appear in `gtd_orders`, and startup reconciliation does not see it.
        //
        // Accounting note: the `cost` of a resting sell is 0 in the exposure ledger for PM
        // (unwind: selling one's own inventory releases the position). But ThetaCapture with
        // `sell_loser_side` sells the LOSING side without ever having bought it (a naked short, with
        // downside of up to $1/share if it resolves against us). That exposure was TOTALLY invisible
        // to the global cap. It is told apart without adding a new field to `OrderIntent` (that would
        // touch the ~69 call sites in 16 files): within TC, the ONLY resting sell is that of the
        // losing side (`theta_capture::ask_maker_intent`); its winning-side sell (bail,
        // `sell_intent`) is always `resting=false` (market).
        // So `(strategy == ThetaCapture) && resting` identifies exactly the naked short
        // without touching `types.rs`.
        if intent.resting {
            if let Some(oid) = &result.order_id {
                // known_order_ids and order_index are already registered above, next to the submit.
                let key = limit_key(&intent.market_cid, &intent.token_id, intent.strategy);

                // If there is ALREADY a live entry on this token (e.g. a buy GTD re-inserted
                // after a failed cancel; see handle_cancel and purge_and_resync),
                // the `insert` below would silently overwrite it, losing its order_id and
                // leaving it no way to be cancelled. Cancel it explicitly
                // BEFORE registering the new one, just like the anti-duplicate check on the
                // buy side (see the anti-duplicate block in handle_buy).
                if let Some((old_oid, old_cost, old_clear, old_strat)) =
                    self.active_gtd.remove(&key)
                {
                    tracing::error!(
                        old_oid, old_strategy = %old_strat, new_oid = oid,
                        "executor: handle_sell was about to overwrite a live GTD on the same token — cancelling it explicitly"
                    );
                    self.record_cancel_request(&intent.market_cid, &intent.token_id, &old_oid);
                    let cancel_res = self.router.cancel(&intent.market_cid, &old_oid).await;
                    self.record_cancel_result(
                        &intent.market_cid,
                        &intent.token_id,
                        &old_oid,
                        &cancel_res,
                    );
                    match cancel_res {
                        Ok(_) => {
                            self.ledger.release_exposure(
                                &intent.market_cid,
                                &intent.token_id,
                                old_cost,
                                old_strat,
                            );
                            if let Err(e) = self.state.delete_gtd_order(&old_oid).await {
                                tracing::error!(
                                    old_oid,
                                    "executor: failed to delete gtd_orders: {e}"
                                );
                            }
                        }
                        Err(e) => {
                            // It may still be live on the CLOB: keep the claim
                            // and re-register it so the next cycle retries.
                            warn!(old_oid, "executor: cancel of orphan GTD failed — claim kept: {e}");
                            self.active_gtd
                                .insert(key.clone(), (old_oid, old_cost, old_clear, old_strat));
                        }
                    }
                }

                // Naked TC short: commit $1/share of downside to the ledger BEFORE leaving the GTD
                // live, so the per-bracket directional cap and the global cap see it. If it exceeds the cap,
                // cancel immediately (the order was already posted: on this path there is no atomicity
                // between reserving capacity and posting, unlike the buy side where the claim
                // is taken BEFORE the submit).
                let is_naked_short = intent.strategy == StrategyId::ThetaCapture;
                let claim_cost = if is_naked_short {
                    intent.size
                } else {
                    rust_decimal::Decimal::ZERO
                };
                if is_naked_short {
                    match self.ledger.try_add_exposure(
                        &intent.market_cid,
                        intent.real_clear_at.unwrap_or(intent.clear_at),
                        &intent.token_id,
                        claim_cost,
                        self.risk.max_bracket_exposure(),
                        intent.strategy,
                    ) {
                        Some(net) => record_net_exposure(net),
                        None => {
                            metrics::counter!("hrf_bracket_cap_rejects_total", "strategy" => intent.strategy.to_string()).increment(1);
                            warn!(strategy = %intent.strategy, cid = %intent.market_cid,
                                  "executor: naked short maker ask (TC) exceeds the directional cap — cancelling");
                            self.record_cancel_request(&intent.market_cid, &intent.token_id, oid);
                            let cancel_res = self.router.cancel(&intent.market_cid, oid).await;
                            self.record_cancel_result(
                                &intent.market_cid,
                                &intent.token_id,
                                oid,
                                &cancel_res,
                            );
                            return;
                        }
                    }
                }
                // Known limitation: once this GTD fills, the risk moves from "pending order" to "open short"
                // position, but `state.rs` has no short-position tracking
                // (`list_open_positions` only models the bot's own inventory). The resync
                // of the filled-GTD purge removes the `active_gtd` entry as soon as it fills,
                // implicitly releasing this claim even though the short stays open until settlement.
                // Fixing it requires short-position tracking in `state.rs`, which is out of scope here
                // (`state.rs` and `apply_fill` are owned elsewhere).
                // The losing side's bail, if the oracle reverts, is not implemented here either:
                // the bail logic lives in `theta_capture.rs`.
                // A short position is not tracked anywhere until settlement.
                self.active_gtd.insert(
                    key,
                    (oid.clone(), claim_cost, intent.clear_at, intent.strategy),
                );
                if let Err(e) = self
                    .state
                    .record_gtd_order(oid, &intent.market_cid, intent.strategy)
                    .await
                {
                    tracing::error!(oid, "executor: failed to record gtd_orders: {e}");
                }
                metrics::counter!("hrf_executor_limit_sell_total", "strategy" => intent.strategy.to_string()).increment(1);
            }
        }

        if result.order_id.is_some() {
            info!(
                strategy = %intent.strategy,
                cid   = %intent.market_cid,
                token = %intent.token_id,
                order_id = ?result.order_id,
                "executor: maker ask accepted"
            );
        } else {
            warn!(
                strategy = %intent.strategy,
                cid   = %intent.market_cid,
                token = %intent.token_id,
                "executor: sell without fill — router rejected it or there is no liquidity"
            );
        }
    }

    // ── Cancel ────────────────────────────────────────────────────────────────

    async fn handle_cancel(&mut self, intent: &OrderIntent) {
        // The key carries the intent's strategy: a `Cancel` from B does NOT reach
        // A's GTD on the same token (it would hit here under the old cid:token key);
        // without its own entry it is ignored and nothing is released.
        let key = limit_key(&intent.market_cid, &intent.token_id, intent.strategy);
        let (order_id, cost, clear_at, strategy) = match self.active_gtd.remove(&key) {
            Some(v) => v,
            None => {
                // Do not ignore this silently: if the strategy believes it has a live order here
                // but the executor never recorded it (the intent was discarded by a gate, or already
                // finished), it is a ghost state that ONLY a terminal ack releases. `Cancelled` is the
                // truth from the executor's authoritative point of view:
                // there is no live order for this key. Without this ack the leg
                // would stay mute forever (a cancel against nothing).
                warn!(
                    cid   = %intent.market_cid,
                    token = %intent.token_id,
                    "executor: cancel with no registered order — Cancelled ack to release the ghost state"
                );
                self.emit_cancel_ack(
                    &intent.market_cid,
                    &intent.token_id,
                    intent.strategy,
                    None,
                    intent.neg_risk,
                    intent.real_clear_at.unwrap_or(intent.clear_at),
                );
                return;
            }
        };

        self.record_cancel_request(&intent.market_cid, &intent.token_id, &order_id);
        let cancel_res = self.router.cancel(&intent.market_cid, &order_id).await;
        self.record_cancel_result(&intent.market_cid, &intent.token_id, &order_id, &cancel_res);
        match cancel_res {
            Ok(_) => {
                debug!(order_id, "executor: order cancelled");
                // `cancel` is idempotent on the CLOB: `Ok` does not prove that the order
                // had no filled quantity (in full or in part) before the cancel. Releasing the whole claim
                // would leave the executed part without exposure until the user_ws fill landed, and
                // if user_ws lost it, the position would be invisible with nothing to restore it.
                // So the final state is queried:
                match self.router.order_status(&order_id).await {
                    Some(RouterOrderStatus::Live { .. }) => {
                        // Cancel accepted but the order is still live: same rule as the Err arm:
                        // claim kept and entry re-registered for a retry.
                        warn!(
                            order_id,
                            "executor: cancel Ok but the order is still live — claim kept"
                        );
                        self.active_gtd
                            .insert(key, (order_id, cost, clear_at, strategy));
                    }
                    Some(RouterOrderStatus::Gone { matched })
                        if matched > rust_decimal::Decimal::ZERO =>
                    {
                        // Executed part: emit, from the venue's trade tape, the fills that are not yet durable
                        // (exec_id = composite trade_id, deduplicated). Opportunistic: it speeds up the landing
                        // but closes nothing; see below.
                        if let Some(list) = self
                            .router
                            .trades_since(crate::time::now().timestamp() - TRADES_LOOKBACK_SECS)
                            .await
                        {
                            for t in list.iter().filter(|t| {
                                t.order_id.as_deref() == Some(order_id.as_str())
                                    && t.token_id == intent.token_id
                            }) {
                                if matches!(
                                    self.state.fill_exists_exec(&t.trade_id).await,
                                    Ok(true)
                                ) {
                                    continue; // already durable
                                }
                                self.emit_fill(Fill {
                                    market_cid: intent.market_cid.clone(),
                                    token_id: intent.token_id.clone(),
                                    side: if t.is_buy {
                                        FillSide::Buy
                                    } else {
                                        FillSide::Sell
                                    },
                                    price: t.price,
                                    size: t.size,
                                    order_id: Some(order_id.clone()),
                                    exec_id: Some(t.trade_id.clone()),
                                    local_order_id: None,
                                    ts: chrono::DateTime::from_timestamp_millis(t.match_ts_ms)
                                        .unwrap_or_else(crate::time::now),
                                    mode: self.router.mode(),
                                    neg_risk: intent.neg_risk,
                                    strategy,
                                    clear_at,
                                    outcome_label: None,
                                    hedge_id: None,
                                });
                            }
                        }
                        // The claim is NOT released when emitting: a crash between enqueue and persistence
                        // would leave the pending closed and the position invisible.
                        // It is always an Unknown pending plus an order_index: the reconciler resolves it
                        // once the fills are durable (or once the tape publishes them, if they are not yet),
                        // and only then is the exposure released.
                        // The pending goes through `record_pending_cancel`: the `Cancel` intent carries
                        // `price=0` by protocol, and journaling it with that metadata made the reconciler
                        // release the claim as `remaining*0`.
                        // The original claim (`cost`) and the order's real `clear_at` are kept;
                        // the pending's price, size and side stay NULL (unknown, not invented).
                        match self
                            .record_pending_cancel(
                                intent,
                                cost,
                                clear_at,
                                "cancel Ok with matched>0 (B3)",
                            )
                            .await
                        {
                            Ok(attempt_id) => {
                                self.register_cancel_pending(
                                    intent,
                                    cost,
                                    clear_at,
                                    "cancel Ok with matched>0 (B3)",
                                    &attempt_id,
                                );
                                self.known_order_ids.insert(order_id.clone());
                                if let Err(e) = self
                                    .state
                                    .record_order_index(
                                        &order_id,
                                        Some(&attempt_id),
                                        strategy,
                                        &intent.market_cid,
                                        &intent.token_id,
                                    )
                                    .await
                                {
                                    tracing::error!(
                                        order_id,
                                        "executor: failed to record order_index (cancel): {e}"
                                    );
                                }
                                warn!(order_id, matched = %matched,
                                    "executor: cancel Ok with an executed part — claim kept via pending; the reconciler releases it once the fills are durable");
                            }
                            Err(e) => {
                                // Fail-closed: without a journal there is no
                                // durable reconciliation: keep the claim
                                // and re-register it so that the next
                                // resync/cancel retries it.
                                tracing::error!(order_id,
                                    "executor: failed to journal the cancel pending — claim kept: {e}");
                                self.active_gtd
                                    .insert(key, (order_id.clone(), cost, clear_at, strategy));
                            }
                        }
                        if let Err(e) = self.state.delete_gtd_order(&order_id).await {
                            tracing::error!(
                                order_id,
                                "executor: failed to delete gtd_orders: {e}"
                            );
                        }
                        self.emit_cancel_ack(
                            &intent.market_cid,
                            &intent.token_id,
                            strategy,
                            Some(order_id),
                            intent.neg_risk,
                            clear_at,
                        );
                    }
                    Some(RouterOrderStatus::Gone { .. }) | Some(RouterOrderStatus::NotFound) => {
                        // Nothing filled: clean outcome.
                        self.ledger.release_exposure(
                            &intent.market_cid,
                            &intent.token_id,
                            cost,
                            strategy,
                        );
                        if let Err(e) = self.state.delete_gtd_order(&order_id).await {
                            tracing::error!(
                                order_id,
                                "executor: failed to delete gtd_orders: {e}"
                            );
                        }
                        self.emit_cancel_ack(
                            &intent.market_cid,
                            &intent.token_id,
                            strategy,
                            Some(order_id),
                            intent.neg_risk,
                            clear_at,
                        );
                    }
                    None => {
                        // Indeterminate state after an Ok cancel: it may still be live or have a filled quantity.
                        // Do NOT release the claim: the pending becomes an Unknown entry plus an order_index,
                        // and the reconciler closes it with evidence in the next cycle.
                        // `record_pending_cancel` is used: the `price=0` of the Cancel intent is not the
                        // order's price; the pending keeps the original claim (`cost`) with NULL metadata,
                        // and the reconciler resolves them with the destination's evidence.
                        match self
                            .record_pending_cancel(
                                intent,
                                cost,
                                clear_at,
                                "cancel Ok with indeterminate order_status (B3)",
                            )
                            .await
                        {
                            Ok(attempt_id) => {
                                self.register_cancel_pending(
                                    intent,
                                    cost,
                                    clear_at,
                                    "cancel Ok with indeterminate order_status (B3)",
                                    &attempt_id,
                                );
                                self.known_order_ids.insert(order_id.clone());
                                if let Err(e) = self
                                    .state
                                    .record_order_index(
                                        &order_id,
                                        Some(&attempt_id),
                                        strategy,
                                        &intent.market_cid,
                                        &intent.token_id,
                                    )
                                    .await
                                {
                                    tracing::error!(
                                        order_id,
                                        "executor: failed to record order_index (cancel): {e}"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::error!(order_id,
                                    "executor: failed to journal the cancel pending — claim kept: {e}");
                                self.active_gtd
                                    .insert(key, (order_id.clone(), cost, clear_at, strategy));
                            }
                        }
                        self.emit_cancel_ack(
                            &intent.market_cid,
                            &intent.token_id,
                            strategy,
                            Some(order_id),
                            intent.neg_risk,
                            clear_at,
                        );
                    }
                }
            }
            Err(e) => {
                // It may still be live on the CLOB: keep the claim
                // and re-register the entry so the next Cancel or anti-duplicate check
                // retries it (it used to be released blindly).
                warn!(order_id, "executor: error cancelling (the order may still be active on the CLOB) — claim kept: {e}");
                self.active_gtd
                    .insert(key, (order_id, cost, clear_at, strategy));
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn build_request(&self, intent: &OrderIntent, is_sell: bool) -> OrderRequest {
        let kind = if is_sell {
            let size = match &intent.side {
                OrderSide::Sell { size } => *size,
                _ => None,
            };
            match (intent.limit_price, intent.resting) {
                // Maker GTD ask: the price and the resting nature are the whole point of the order.
                // Previously `is_sell` was checked FIRST and both fields were discarded: every sell
                // went out as `OrderKind::Sell` (an unconditional market sell taker), even when
                // `parity_maker::ask_maker_intent` asked for cost+1¢ resting.
                // Invariant: `OrderKind::Sell` is reserved for market sells ON PURPOSE (orphan bail,
                // `sell_intent`); any intent with `limit_price: Some(_) && resting` is a maker by definition.
                // The limit price and the resting flag are honoured as written.
                (Some(limit), true) => OrderKind::LimitSell {
                    price: limit,
                    size: size.unwrap_or(intent.size),
                    tif: TimeInForce::Gtd,
                },
                _ => OrderKind::Sell { size },
            }
        } else if let Some(limit) = intent.limit_price {
            let tif = if intent.resting {
                TimeInForce::Gtd
            } else if self.fak_strategies.contains(&intent.strategy) {
                TimeInForce::Fak
            } else {
                TimeInForce::Fok
            };
            OrderKind::Limit {
                price: limit,
                size: intent.size,
                tif,
            }
        } else {
            OrderKind::Market {
                usdc_amount: intent.price * intent.size,
            }
        };

        // MM with GTD resting = pure maker: post_only guarantees 0% fee and a possible rebate.
        // The CLOB rejects the order if it would cross the spread (the desired behaviour for MM).
        let post_only = intent.resting && intent.strategy == StrategyId::MarketMaking;

        OrderRequest {
            market_cid: intent.market_cid.clone(),
            token_id: intent.token_id.clone(),
            kind,
            neg_risk: intent.neg_risk,
            strategy: intent.strategy,
            clear_at: intent.clear_at,
            signal: intent.signal,
            post_only,
            real_clear_at: intent.real_clear_at,
            local_order_id: None,
        }
    }
}

/// Key of `active_gtd`: `market:token:strategy`. Per-owner reservations:
/// the owner is IN the key, so that a strategy's `Cancel` cannot reach the order (nor release its claim)
/// of another strategy on the same token. NOTE: `order_map` is still keyed by `market:token`
/// (a known cross-strategy collision: durable attribution uses
/// `order_index`, not this mapping).
/// (Attribution is therefore never taken from this key.)
#[inline]
fn limit_key(market_cid: &str, token_id: &str, strategy: StrategyId) -> String {
    format!("{market_cid}:{token_id}:{strategy}")
}

/// Durable side of a row of the Unknown journal. `None` = unknown side
/// (legacy, quarantine, cancel): it is not inferred from
/// `cost`/`price`/`reason`: without evidence there is no match and no new claim.
fn journal_side(row: &crate::state::UnknownAttemptRow) -> Option<OpenOrderSide> {
    match row.side.as_deref() {
        Some("buy") => Some(OpenOrderSide::Buy),
        Some("sell") => Some(OpenOrderSide::Sell),
        // NULL or an unrecognised value (a future format?) → indeterminate.
        _ => None,
    }
}

/// Outcome of evaluating the evidence for the claim of an order that the destination
/// reports as LIVE while reconciling a pending.
///
/// Three outcomes, deliberately distinct: "missing data" is NOT the same as
/// "contradictory data": the first keeps the claim and resolves; the
/// last one does NOT close the pending nor adjudicate the order.
enum LiveClaimEvidence {
    /// Claim determined from coherent evidence (the open order of the
    /// destination, or the journal's durable metadata).
    Determined(rust_decimal::Decimal),
    /// Not enough evidence: keep the original durable claim
    /// (`cost_um`) and resolve the pending. The order IS live (`order_status` proved it),
    /// only its valuation is missing. Never
    /// `remaining × 0` with the price marker of a cancel pending.
    KeepCost,
    /// The destination's evidence CONTRADICTS the pending (same `order_id`
    /// but different cid/token, or a declared side different from the journal):
    /// it is not a fallback, it is real uncertainty. The pending stays
    /// open and the order is NOT adjudicated to `active_gtd`.
    Contradictory,
}

/// Claim of a live order when reconciling a pending.
/// Chain of evidence, from the freshest to the least fresh:
///
/// 1. The exact open order from the destination's listing: requires a coherent `order_id`
///    + `cid` + `token_id` matching the pending, and the SAME side if the journal
///    declares one (cancel pendings carry a NULL `side`, so the destination's BUY/SELL
///    evidence is accepted). Any contradiction → `Contradictory`.
///    Never a reduction.
/// 2. Durable metadata of the journal: an explicit `side` + a valid `price`,
///    with the `remaining` from `order_status`.
/// 3. Not enough evidence (incomplete data, a wire side `Unknown`, invalid price):
///    `KeepCost`: the original claim is kept.
///    It is never reduced for lack of data.
fn live_claim_evidence(
    row: &crate::state::UnknownAttemptRow,
    oid: &str,
    remaining: rust_decimal::Decimal,
    strategy: StrategyId,
    open_orders: Option<&[RouterOpenOrder]>,
) -> LiveClaimEvidence {
    if let Some(list) = open_orders {
        if let Some(o) = list.iter().find(|o| o.order_id == oid) {
            // The oid is correct but the rest of the identity is not: a different cid or
            // token, or a declared side that differs from the journal's.
            // Corrupt or conflicting data: it authorizes neither a claim nor a close.
            if o.market_cid != row.cid || o.token_id != row.token_id {
                return LiveClaimEvidence::Contradictory;
            }
            if let Some(js) = journal_side(row) {
                if o.side != OpenOrderSide::Unknown && o.side != js {
                    return LiveClaimEvidence::Contradictory;
                }
            }
            if let Some(claim) = resting_claim_usdc(o.side, o.remaining, o.price, strategy)
            {
                return LiveClaimEvidence::Determined(claim);
            }
            // Incomplete data (wire side Unknown, invalid price): not a contradiction;
            // it falls through to the durable metadata or to the cost.
        }
    }
    if let (Some(side), Some(price_um)) = (journal_side(row), row.price_um) {
        if let Some(claim) =
            resting_claim_usdc(side, remaining, crate::state::from_micros(price_um), strategy)
        {
            return LiveClaimEvidence::Determined(claim);
        }
    }
    LiveClaimEvidence::KeepCost
}

/// Distribution of the directional net per bracket after each accepted claim.
/// No cid label on purpose: the cardinality would be unbounded (new brackets every hour).
fn record_net_exposure(net: rust_decimal::Decimal) {
    use rust_decimal::prelude::ToPrimitive;
    metrics::histogram!("hrf_bracket_net_exposure_usdc").record(net.to_f64().unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RiskConfig;
    use crate::order::router::{OrderRequest, RouterOrderStatus, RouterTrade};
    use crate::types::BookSnapshot;
    use rust_decimal::Decimal;

    /// Test router: always rejects the first leg (YES) and always accepts
    /// the second leg (NO). It models the real scenario where `post_orders` is not
    /// atomic across legs.
    struct AsymmetricRouter;

    impl OrderRouter for AsymmetricRouter {
        fn submit(
            &self,
            _req: OrderRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::order::router::RouterResult> + Send + '_>,
        > {
            Box::pin(async { crate::order::router::RouterResult::no_fill(Some("oid".into())) })
        }

        fn submit_pair(
            &self,
            _req1: OrderRequest,
            _req2: OrderRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = (
                            crate::order::router::RouterResult,
                            crate::order::router::RouterResult,
                        ),
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                (
                    crate::order::router::RouterResult::rejected(
                        "FOK rejected: insufficient liquidity",
                    ),
                    crate::order::router::RouterResult::no_fill(Some("no-oid".into())),
                )
            })
        }

        fn cancel(
            &self,
            _market_cid: &str,
            _order_id: &str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>
        {
            Box::pin(async { Ok(()) })
        }

        fn mode(&self) -> crate::types::Mode {
            crate::types::Mode::Paper
        }
    }

    /// Router that records the SEQUENCE of operations: the only thing that allows
    /// telling apart "cancel the oldest and then post" from "post and then cancel the
    /// oldest", which is exactly the bug.
    struct RecordingRouter {
        events: Arc<parking_lot::Mutex<Vec<String>>>,
        n: Arc<parking_lot::Mutex<u32>>,
    }

    impl OrderRouter for RecordingRouter {
        fn submit(
            &self,
            _req: OrderRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::order::router::RouterResult> + Send + '_>,
        > {
            Box::pin(async move {
                let oid = {
                    let mut n = self.n.lock();
                    *n += 1;
                    format!("oid-{n}")
                };
                self.events.lock().push(format!("submit:{oid}"));
                crate::order::router::RouterResult::no_fill(Some(oid))
            })
        }

        fn submit_pair(
            &self,
            _req1: OrderRequest,
            _req2: OrderRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = (
                            crate::order::router::RouterResult,
                            crate::order::router::RouterResult,
                        ),
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                (
                    crate::order::router::RouterResult::no_fill(None),
                    crate::order::router::RouterResult::no_fill(None),
                )
            })
        }

        fn cancel(
            &self,
            _market_cid: &str,
            order_id: &str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>
        {
            let ev = self.events.clone();
            let oid = order_id.to_string();
            Box::pin(async move {
                ev.lock().push(format!("cancel:{oid}"));
                Ok(())
            })
        }

        fn mode(&self) -> crate::types::Mode {
            crate::types::Mode::Paper
        }

        /// The executor queries the final state after an Ok cancel.
        /// `Gone{matched:0}` models "nothing filled": the clean outcome that
        /// releases the claim (tests with an executed part use the sim).
        fn order_status(
            &self,
            _order_id: &str,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Option<crate::order::router::RouterOrderStatus>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Some(crate::order::router::RouterOrderStatus::Gone {
                    matched: rust_decimal::Decimal::ZERO,
                })
            })
        }
    }

    /// Test router for the pair path: each leg answers with what it is told.
    /// `Ok` = accepted without an immediate fill (it stays claimed), `No` = rejected.
    struct PairRouter {
        leg1_ok: bool,
        leg2_ok: bool,
    }

    impl PairRouter {
        fn result(ok: bool, oid: &str) -> crate::order::router::RouterResult {
            if ok {
                crate::order::router::RouterResult::no_fill(Some(oid.into()))
            } else {
                crate::order::router::RouterResult::rejected("test")
            }
        }
    }

    impl OrderRouter for PairRouter {
        fn submit(
            &self,
            _req: OrderRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::order::router::RouterResult> + Send + '_>,
        > {
            Box::pin(async { crate::order::router::RouterResult::rejected("only submit_pair is used") })
        }

        fn submit_pair(
            &self,
            _req1: OrderRequest,
            _req2: OrderRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = (
                            crate::order::router::RouterResult,
                            crate::order::router::RouterResult,
                        ),
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                (
                    Self::result(self.leg1_ok, "oid-1"),
                    Self::result(self.leg2_ok, "oid-2"),
                )
            })
        }

        fn cancel(
            &self,
            _market_cid: &str,
            _order_id: &str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>
        {
            Box::pin(async { Ok(()) })
        }

        fn mode(&self) -> crate::types::Mode {
            crate::types::Mode::Paper
        }
    }

    /// ARB-15M5M: pair of TWO distinct markets. Returns the ledger after
    /// running `handle_buy` with the given outcome for each leg.
    async fn pair_two_markets(leg1_ok: bool, leg2_ok: bool) -> Arc<BracketLedger> {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper).await.unwrap();
        let risk = RiskGuard::new(
            RiskConfig { max_total_exposure_usdc: dec!(1000), ..Default::default() },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(PairRouter { leg1_ok, leg2_ok }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );
        let clear_at = chrono::Utc::now().timestamp() + 300;
        let leg = |cid: &str, tok: &str, second_order: Option<OrderIntent>| OrderIntent {
            market_cid: cid.into(),
            token_id: tok.into(),
            side: OrderSide::Buy,
            price: dec!(0.4),
            size: dec!(10),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.4)),
            second_leg: second_order.map(Box::new),
            strategy: StrategyId::Arb15m5m,
            resting: false,
            real_clear_at: None,
        };
        let intent = leg("cid-15m", "up-15m", Some(leg("cid-5m", "down-5m", None)));
        executor.handle_buy(intent).await;
        ledger
    }

    #[tokio::test]
    async fn two_market_pair_claims_each_leg_under_its_cid() {
        let l = pair_two_markets(true, true).await;
        let s = StrategyId::Arb15m5m;
        assert_eq!(l.owner_exposure("cid-15m", "up-15m", s), dec!(4));
        assert_eq!(l.owner_exposure("cid-5m", "down-5m", s), dec!(4));
        // Nothing from the second leg hangs off the first leg's bracket.
        assert_eq!(l.owner_exposure("cid-15m", "down-5m", s), dec!(0));
    }

    #[tokio::test]
    async fn rejected_two_market_pair_releases_both_brackets() {
        let l = pair_two_markets(false, false).await;
        assert_eq!(l.total_exposure(), dec!(0));
        assert_eq!(l.gross_exposure("cid-15m"), dec!(0));
        assert_eq!(l.gross_exposure("cid-5m"), dec!(0));
    }

    #[tokio::test]
    async fn two_market_orphan_releases_only_the_rejected_leg() {
        let s = StrategyId::Arb15m5m;
        let l = pair_two_markets(true, false).await;
        assert_eq!(l.owner_exposure("cid-15m", "up-15m", s), dec!(4));
        assert_eq!(l.gross_exposure("cid-5m"), dec!(0), "leg 2 rejected is released in ITS cid");
        let l = pair_two_markets(false, true).await;
        assert_eq!(l.gross_exposure("cid-15m"), dec!(0));
        assert_eq!(l.owner_exposure("cid-5m", "down-5m", s), dec!(4));
    }

    /// Re-posting a GTD over a token that already has another live one must CANCEL
    /// THE OLD ONE BEFORE sending the new one. If it is sent first and cancelled afterwards,
    /// there is a window with TWO live GTDs on the same token, and the book can fill
    /// both: the strategy's size cap stops meaning anything (PM, with a cap of 3 shares,
    /// accumulated 33 net shares in the 14 Jul calibration).
    #[tokio::test]
    async fn gtd_repost_cancels_old_before_posting_new() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        let gtd = || OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.5),
            size: dec!(3),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.5)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        };

        executor.handle_buy(gtd()).await; // first GTD
        executor.handle_buy(gtd()).await; // requote over the same token

        let ev = events.lock().clone();
        assert_eq!(
            ev,
            vec!["submit:oid-1", "cancel:oid-1", "submit:oid-2"],
            "the oldest GTD must be cancelled BEFORE the new order's submit; observed sequence: {ev:?}"
        );
    }

    /// A Buy intent that waits more than `MAX_INTENT_QUEUE_WAIT` in the
    /// executor queue is discarded with `GateRejected` and NEVER reaches the
    /// router: it was decided on a book that no longer exists. Exits
    /// (Sell/Cancel) are not dropped: a bail-sell that arrives late is still
    /// a valid exit.
    #[tokio::test]
    async fn stale_intent_in_queue_buy_gate_rejected_without_submit() {
        let (intent_tx, intent_rx) = mpsc::channel(4);
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );
        let task = tokio::spawn(async move { executor.run().await });

        let buy = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.5),
            size: dec!(3),
            clear_at: chrono::Utc::now().timestamp() + 300,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.5)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        };
        // Queued 1 s ago, above the freshness budget (500 ms).
        intent_tx
            .send((
                buy,
                std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_secs(1))
                    .unwrap(),
            ))
            .await
            .unwrap();
        drop(intent_tx);

        tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("executor.run() hung")
            .unwrap();

        let fill = fill_rx.recv().await.expect("must emit GateRejected");
        assert!(
            matches!(fill.side, FillSide::GateRejected),
            "expected GateRejected, got {:?}",
            fill.side
        );
        assert!(
            events.lock().is_empty(),
            "a stale intent must not reach the router: {:?}",
            events.lock()
        );
    }

    /// Cooldown after losses (CooldownPeriod): a bracket marked by
    /// `record_bracket_loss` makes `handle_buy` skip BEFORE the submit
    /// (same pattern as the other risk gates: log + return, no fill).
    /// Sells in the same bracket are NOT affected: the gate only
    /// blocks entries.
    #[tokio::test]
    async fn cooldown_after_loss_blocks_buy_but_not_sell() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                entry_cooldown_secs: 3600,
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk.clone(),
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        let buy = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.5),
            size: dec!(3),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.5)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: false,
            real_clear_at: None,
        };
        let sell = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Sell {
                size: Some(dec!(3)),
            },
            price: dec!(0.4),
            size: dec!(3),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: None,
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: false,
            real_clear_at: None,
        };

        risk.record_bracket_loss("cid-1");
        executor.handle_buy(buy).await;
        assert!(
            events.lock().is_empty(),
            "the Buy on a bracket with cooldown must not reach the router"
        );

        executor.handle_sell(sell).await;
        assert_eq!(
            events.lock().clone(),
            vec!["submit:oid-1"],
            "the sell in the same bracket must NOT be blocked: entries only"
        );
    }

    /// If YES is rejected but NO is not (post_orders is not atomic), the ledger
    /// must only release the claim of the leg that was actually rejected. Releasing
    /// NO's too would leave a real position without accounting coverage in the ledger.
    #[tokio::test]
    async fn rejected_hedge_yes_does_not_release_no_exposure_if_no_was_not_rejected() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AsymmetricRouter),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        let no_leg = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "no-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.4),
            size: dec!(10),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.4)),
            second_leg: None,
            strategy: StrategyId::Arb,
            resting: false,
            real_clear_at: None,
        };
        let yes_leg = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.5),
            size: dec!(10),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.5)),
            second_leg: Some(Box::new(no_leg)),
            strategy: StrategyId::Arb,
            resting: false,
            real_clear_at: None,
        };

        executor.handle_buy(yes_leg).await;

        // NO was not rejected (RouterResult::no_fill), so its claim (0.4*10=4) must
        // stay in the ledger; only YES's (0.5*10=5) was released.
        assert_eq!(
            ledger.total_exposure(),
            dec!(4),
            "NO's claim must stay alive"
        );
    }

    /// With global exposure almost saturated, a Buy from PostCloseSweep must
    /// pass thanks to the carve-out (effective_max_total_exposure), while the
    /// same Buy for another strategy without a reserve is rejected.
    #[tokio::test]
    async fn executor_pcs_enters_with_saturated_global() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(100),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        risk.set_strategy_exposure_carveout(StrategyId::PostCloseSweep, dec!(30));
        let ledger = BracketLedger::new();

        // Exposure already almost saturated (95 of 100) in another bracket.
        ledger.force_add_exposure(
            "cid-other",
            chrono::Utc::now().timestamp() + 300,
            "other-tok",
            dec!(95),
            StrategyId::NearCertain,
        );

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AsymmetricRouter),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        let pcs_intent = OrderIntent {
            market_cid: "cid-pcs".into(),
            token_id: "pcs-tok".into(),
            side: OrderSide::Buy,
            price: dec!(1),
            size: dec!(20),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(1)),
            second_leg: None,
            strategy: StrategyId::PostCloseSweep,
            resting: false,
            real_clear_at: None,
        };
        executor.handle_buy(pcs_intent).await;
        // 95 + 20 = 115 <= 130 (100 + carve-out 30) → passes and its claim is added.
        assert_eq!(
            ledger.total_exposure(),
            dec!(115),
            "PCS must enter with the carve-out"
        );

        let other_intent = OrderIntent {
            market_cid: "cid-nc".into(),
            token_id: "nc-tok".into(),
            side: OrderSide::Buy,
            price: dec!(1),
            size: dec!(20),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(1)),
            second_leg: None,
            strategy: StrategyId::NearCertain,
            resting: false,
            real_clear_at: None,
        };
        executor.handle_buy(other_intent).await;
        // 115 + 20 = 135 > 100 (no carve-out for NearCertain) → rejected without adding.
        assert_eq!(
            ledger.total_exposure(),
            dec!(115),
            "another strategy without a carve-out must be rejected"
        );
    }

    /// The test that would have caught the bug. Before, `build_request`
    /// checked `is_sell` BEFORE looking at `limit_price`/`resting`, so EVERY
    /// sell, including PM's unwind maker GTD ask, went out as
    /// `OrderKind::Sell` (an unconditional market sell taker).
    #[tokio::test]
    async fn build_request_respects_limit_price_and_resting_on_sells() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();

        let executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AsymmetricRouter),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;

        // Maker GTD ask (PM unwind): limit_price + resting → OrderKind::LimitSell.
        let ask_maker = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Sell {
                size: Some(dec!(3)),
            },
            price: dec!(0.45),
            size: dec!(3),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.45)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        };
        match executor.build_request(&ask_maker, true).kind {
            OrderKind::LimitSell { price, size, tif } => {
                assert_eq!(price, dec!(0.45));
                assert_eq!(size, dec!(3));
                assert_eq!(tif, TimeInForce::Gtd);
            }
            other => panic!("expected OrderKind::LimitSell, got {other:?}"),
        }

        // Bail of an orphan, or a deliberate market sell: no limit_price, no
        // resting → it stays OrderKind::Sell (unchanged behaviour).
        let bail = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Sell {
                size: Some(dec!(3)),
            },
            price: dec!(0.30),
            size: dec!(3),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: None,
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: false,
            real_clear_at: None,
        };
        match executor.build_request(&bail, true).kind {
            OrderKind::Sell { size } => assert_eq!(size, Some(dec!(3))),
            other => panic!("expected OrderKind::Sell, got {other:?}"),
        }
    }

    /// Router that ALWAYS rejects, for the rate limit on rejections.
    struct AlwaysRejectRouter;
    impl OrderRouter for AlwaysRejectRouter {
        fn submit(
            &self,
            _req: OrderRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::order::router::RouterResult> + Send + '_>,
        > {
            Box::pin(async {
                crate::order::router::RouterResult::rejected("always rejects (test)")
            })
        }
        fn cancel(
            &self,
            _market_cid: &str,
            _order_id: &str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>
        {
            Box::pin(async { Ok(()) })
        }
        fn mode(&self) -> crate::types::Mode {
            crate::types::Mode::Paper
        }
    }

    /// Test router reporting a FIXED set of "live" order_ids, for
    /// the purge_and_resync path that removes filled GTDs from the local mapping.
    struct FakeLiveIdsRouter {
        live_ids: std::collections::HashSet<String>,
    }
    impl OrderRouter for FakeLiveIdsRouter {
        fn submit(
            &self,
            _req: OrderRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::order::router::RouterResult> + Send + '_>,
        > {
            Box::pin(async { crate::order::router::RouterResult::no_fill(Some("oid-live".into())) })
        }
        fn cancel(
            &self,
            _market_cid: &str,
            _order_id: &str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>
        {
            Box::pin(async { Ok(()) })
        }
        fn live_gtd_order_ids(
            &self,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Option<std::collections::HashSet<String>>>
                    + Send
                    + '_,
            >,
        > {
            let ids = self.live_ids.clone();
            Box::pin(async move { Some(ids) })
        }
        fn mode(&self) -> crate::types::Mode {
            crate::types::Mode::Paper
        }
    }

    /// The global exposure cap must ALSO see the second leg
    /// (ARB YES+NO). Before, only the first leg was evaluated, leaving the other half
    /// of the pair invisible to `max_total_exposure_usdc` and letting it be
    /// exceeded silently.
    #[tokio::test]
    async fn cap_global_counts_second_leg_of_pair() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(10),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AsymmetricRouter),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        // YES=6 + NO=6 = 12 > cap(10). Before, only YES=6 was seen (it passed the
        // step 5 check and was only stopped later, inside try_add_pair_exposure, against
        // a DIFFERENT cap: the per-bracket directional one, not the global one).
        let no_leg = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "no-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.6),
            size: dec!(10),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.6)),
            second_leg: None,
            strategy: StrategyId::Arb,
            resting: false,
            real_clear_at: None,
        };
        let yes_leg = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.6),
            size: dec!(10),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.6)),
            second_leg: Some(Box::new(no_leg)),
            strategy: StrategyId::Arb,
            resting: false,
            real_clear_at: None,
        };
        executor.handle_buy(yes_leg).await;

        assert_eq!(
            ledger.total_exposure(),
            rust_decimal::Decimal::ZERO,
            "6+6=12 > cap(10) must be rejected BEFORE the submit, seeing both legs"
        );
    }

    /// Re-quoting a GTD over the SAME token must not count the earlier claim
    /// twice against the global cap: the stale claim (still live in the ledger at the time
    /// of the check) must be subtracted.
    #[tokio::test]
    async fn requote_does_not_count_previous_claim_twice() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        // Cap set EXACTLY to the cost of one GTD: a requote over the same
        // token should never hit it (it is the same exposure, not a new one).
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1.5),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        let gtd = || OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.5),
            size: dec!(3),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.5)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        };
        executor.handle_buy(gtd()).await; // first GTD: cost=1.5, exactly the cap
        executor.handle_buy(gtd()).await; // requote over the same token: must pass

        let ev = events.lock().clone();
        assert_eq!(ev, vec!["submit:oid-1", "cancel:oid-1", "submit:oid-2"],
            "the requote must pass the exposure gate instead of being rejected for counting twice; sequence: {ev:?}");
        assert_eq!(
            ledger.total_exposure(),
            dec!(1.5),
            "only the claim of the new GTD must remain"
        );
    }

    /// `purge_and_resync` must respect `paper_disable_kill_switch`
    /// like the other kill paths (the global kill in run(), steps 2/3 of handle_buy,
    /// cancel_all_gtds): before, it cancelled GTDs of "killed" strategies that
    /// the rest of the system kept treating as running in paper.
    #[tokio::test]
    async fn purge_and_resync_respects_paper_disable_kill_switch() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                paper_disable_kill_switch: true,
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        risk.kill_strategy(StrategyId::ParityMaker, "test kill");
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        // paper_disable_kill_switch=true → the step 3 check of handle_buy ignores the kill,
        // so the GTD is posted anyway (needed to observe
        // whether purge_and_resync cancels it improperly).
        executor
            .handle_buy(OrderIntent {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                side: OrderSide::Buy,
                price: dec!(0.5),
                size: dec!(3),
                clear_at,
                signal: dec!(0),
                neg_risk: false,
                limit_price: Some(dec!(0.5)),
                second_leg: None,
                strategy: StrategyId::ParityMaker,
                resting: true,
                real_clear_at: None,
            })
            .await;

        executor.purge_and_resync().await;

        let ev = events.lock().clone();
        assert!(!ev.iter().any(|e| e.starts_with("cancel:")),
            "with paper_disable_kill_switch=true, purge_and_resync must NOT cancel GTDs of 'killed' strategies; events: {ev:?}");
    }

    /// Owner-scoped reservations: two strategies with a live GTD on the SAME
    /// `market_cid:token_id`. The `Cancel` of one must reach ONLY its own order and
    /// release ONLY its own claim. With the old `cid:token` key, B's cancel found A's
    /// entry, cancelled its order on the CLOB, and released exposure that was not its own.
    #[tokio::test]
    async fn cancel_of_one_strategy_does_not_touch_gtd_or_claim_of_another_same_token() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        // A (ParityMaker) and B (NearCertain) each have a live GTD on the same token.
        executor
            .restore_gtd(
                "cid-1",
                "yes-tok",
                "oid-A".into(),
                dec!(1.5),
                clear_at,
                StrategyId::ParityMaker,
            )
            .await;
        executor
            .restore_gtd(
                "cid-1",
                "yes-tok",
                "oid-B".into(),
                dec!(2.0),
                clear_at,
                StrategyId::NearCertain,
            )
            .await;
        // Owner-aware claims: 1.5 from A (PM) + 2.0 from B (NC) on the same token.
        ledger.force_add_exposure(
            "cid-1",
            clear_at,
            "yes-tok",
            dec!(1.5),
            StrategyId::ParityMaker,
        );
        ledger.force_add_exposure(
            "cid-1",
            clear_at,
            "yes-tok",
            dec!(2.0),
            StrategyId::NearCertain,
        );

        let cancel_b = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Cancel,
            price: dec!(0),
            size: dec!(0),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: None,
            second_leg: None,
            strategy: StrategyId::NearCertain,
            resting: true,
            real_clear_at: None,
        };
        executor.handle_cancel(&cancel_b).await;

        let ev = events.lock().clone();
        assert_eq!(
            ev,
            vec!["cancel:oid-B".to_string()],
            "only B's order must be cancelled; events: {ev:?}"
        );
        assert_eq!(
            ledger.total_exposure(),
            dec!(1.5),
            "only B's claim (2.0) is released; A's (1.5) stays alive"
        );
        assert!(
            executor
                .active_gtd
                .contains_key("cid-1:yes-tok:parity_maker"),
            "A's entry must stay in active_gtd"
        );

        // A's cancel does reach its own order.
        let cancel_a = OrderIntent {
            strategy: StrategyId::ParityMaker,
            ..cancel_b.clone()
        };
        executor.handle_cancel(&cancel_a).await;
        let ev = events.lock().clone();
        assert_eq!(
            ev,
            vec!["cancel:oid-B".to_string(), "cancel:oid-A".to_string()]
        );
        assert_eq!(ledger.total_exposure(), rust_decimal::Decimal::ZERO);

        // A duplicate Cancel for the same order (e.g. a replayed intent after a requote)
        // finds no entry in active_gtd and releases nothing: no double release, no drift.
        executor.handle_cancel(&cancel_a).await;
        let ev = events.lock().clone();
        assert_eq!(
            ev,
            vec!["cancel:oid-B".to_string(), "cancel:oid-A".to_string()],
            "a duplicate cancel must not reach the router nor release again"
        );
        assert_eq!(ledger.total_exposure(), rust_decimal::Decimal::ZERO);
    }

    /// Router with configurable `order_status`/`trades_since` to
    /// exercise the cancel↔fill race: cancel always returns `Ok`
    /// (idempotent on the CLOB), and the truth about the executed part only comes
    /// from the final state plus the trade tape.
    struct CancelProbeRouter {
        status: Option<RouterOrderStatus>,
        trades: Option<Vec<RouterTrade>>,
    }

    impl OrderRouter for CancelProbeRouter {
        fn submit(
            &self,
            _req: OrderRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::order::router::RouterResult> + Send + '_>,
        > {
            Box::pin(async { crate::order::router::RouterResult::no_fill(None) })
        }

        fn cancel(
            &self,
            _market_cid: &str,
            _order_id: &str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>
        {
            Box::pin(async { Ok(()) })
        }

        fn order_status(
            &self,
            _order_id: &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Option<RouterOrderStatus>> + Send + '_>,
        > {
            let st = self.status.clone();
            Box::pin(async move { st })
        }

        fn trades_since(
            &self,
            _after_ts_secs: i64,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Option<Vec<RouterTrade>>> + Send + '_>,
        > {
            let trades = self.trades.clone();
            Box::pin(async move { trades })
        }

        fn mode(&self) -> crate::types::Mode {
            crate::types::Mode::Paper
        }
    }

    /// Sets up an executor with `CancelProbeRouter`, one live PM GTD over
    /// yes-tok (claim of 1.5), and returns the pieces to assert on.
    async fn executor_cancel_probe(
        status: Option<RouterOrderStatus>,
        trades: Option<Vec<RouterTrade>>,
    ) -> (
        Executor,
        Arc<BracketLedger>,
        Arc<StateStore>,
        mpsc::UnboundedReceiver<Fill>,
        OrderIntent,
    ) {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(CancelProbeRouter { status, trades }),
            risk,
            state.clone(),
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );
        let clear_at = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd(
                "cid-1",
                "yes-tok",
                "oid-1".into(),
                dec!(1.5),
                clear_at,
                StrategyId::ParityMaker,
            )
            .await;
        ledger.force_add_exposure(
            "cid-1",
            clear_at,
            "yes-tok",
            dec!(1.5),
            StrategyId::ParityMaker,
        );
        let cancel = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Cancel,
            price: dec!(0),
            size: dec!(0),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: None,
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        };
        (executor, ledger, state, fill_rx, cancel)
    }

    fn trade_probe(oid: &str, token: &str, price: Decimal, size: Decimal) -> RouterTrade {
        RouterTrade {
            trade_id: "trade-1".into(),
            order_id: Some(oid.into()),
            token_id: token.into(),
            is_buy: true,
            price,
            size,
            match_ts_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Regression router with a listing of open orders,
    /// `order_status` and trade tape MUTABLE between phases: a test can
    /// do `cancel → indeterminate status → journal` and then change the
    /// destination to `Live + open_order` before calling the reconciler.
    struct D1ReconRouter {
        open_orders: parking_lot::Mutex<Option<Vec<RouterOpenOrder>>>,
        status: parking_lot::Mutex<Option<RouterOrderStatus>>,
        trades: Option<Vec<RouterTrade>>,
    }

    impl D1ReconRouter {
        fn new() -> Self {
            Self {
                open_orders: parking_lot::Mutex::new(None),
                status: parking_lot::Mutex::new(None),
                trades: Some(vec![]),
            }
        }
    }

    impl OrderRouter for D1ReconRouter {
        fn submit(
            &self,
            _req: OrderRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::order::router::RouterResult> + Send + '_>,
        > {
            Box::pin(async { crate::order::router::RouterResult::rejected("d1-test") })
        }

        fn cancel(
            &self,
            _market_cid: &str,
            _order_id: &str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>
        {
            Box::pin(async { Ok(()) })
        }

        fn order_status(
            &self,
            _order_id: &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Option<RouterOrderStatus>> + Send + '_>,
        > {
            let st = self.status.lock().clone();
            Box::pin(async move { st })
        }

        fn open_gtd_orders(
            &self,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Option<Vec<RouterOpenOrder>>> + Send + '_,
            >,
        > {
            let v = self.open_orders.lock().clone();
            Box::pin(async move { v })
        }

        fn trades_since(
            &self,
            _after_ts_secs: i64,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Option<Vec<RouterTrade>>> + Send + '_>,
        > {
            let trades = self.trades.clone();
            Box::pin(async move { trades })
        }

        fn mode(&self) -> crate::types::Mode {
            crate::types::Mode::Paper
        }
    }

    fn open_order(
        oid: &str,
        cid: &str,
        token: &str,
        side: OpenOrderSide,
        price: Decimal,
        remaining: Decimal,
    ) -> RouterOpenOrder {
        RouterOpenOrder {
            order_id: oid.into(),
            market_cid: cid.into(),
            token_id: token.into(),
            side,
            price,
            remaining,
            clear_at: chrono::Utc::now().timestamp() + 300,
        }
    }

    /// Executor over `D1ReconRouter`; also returns the router so that the
    /// test can move the destination's listing and state between phases.
    async fn executor_with_d1_router(
        router: Arc<D1ReconRouter>,
    ) -> (
        Executor,
        Arc<BracketLedger>,
        Arc<StateStore>,
        mpsc::UnboundedReceiver<Fill>,
    ) {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let executor = Executor::new(
            intent_rx,
            fill_tx,
            router,
            risk,
            state.clone(),
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );
        (executor, ledger, state, fill_rx)
    }

    fn cancel_intent(strategy: StrategyId) -> OrderIntent {
        OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Cancel,
            price: dec!(0),
            size: dec!(0),
            clear_at: chrono::Utc::now().timestamp() + 300,
            signal: dec!(0),
            neg_risk: false,
            limit_price: None,
            second_leg: None,
            strategy,
            resting: true,
            real_clear_at: None,
        }
    }

    /// Clean path: `cancel` Ok + `order_status` Gone{matched=0}
    /// releases the claim and emits the `Cancelled` ack (nothing filled).
    #[tokio::test]
    async fn cancel_ok_without_match_releases_claim() {
        let (mut executor, ledger, _state, mut fill_rx, cancel) = executor_cancel_probe(
            Some(RouterOrderStatus::Gone {
                matched: Decimal::ZERO,
            }),
            Some(vec![]),
        )
        .await;
        executor.handle_cancel(&cancel).await;

        assert_eq!(ledger.total_exposure(), Decimal::ZERO);
        assert!(executor.active_gtd.is_empty());
        let f = fill_rx
            .recv()
            .await
            .expect("the Cancelled ack must be emitted");
        assert_eq!(f.side, FillSide::Cancelled);
        assert_eq!(f.order_id.as_deref(), Some("oid-1"));
    }

    /// Real race: `cancel` Ok but the order filled completely
    /// first. The fills are emitted from the tape (exec_id = trade_id,
    /// deduplicated) and the claim is kept via an Unknown pending. It is only released
    /// once the reconciler confirms that the fills are DURABLE in
    /// `fills` (a crash between enqueue and persist must not lose the
    /// position with the journal already closed).
    #[tokio::test]
    async fn cancel_ok_with_executed_portion_emits_fills_from_tape() {
        let (mut executor, ledger, state, mut fill_rx, cancel) = executor_cancel_probe(
            Some(RouterOrderStatus::Gone { matched: dec!(3) }),
            Some(vec![trade_probe("oid-1", "yes-tok", dec!(0.5), dec!(3))]),
        )
        .await;
        executor.handle_cancel(&cancel).await;

        let f = fill_rx
            .recv()
            .await
            .expect("fill synthesized from the tape");
        assert_eq!(f.side, FillSide::Buy);
        assert_eq!(f.order_id.as_deref(), Some("oid-1"));
        assert_eq!(f.exec_id.as_deref(), Some("trade-1"));
        assert_eq!(f.size, dec!(3));
        assert_eq!(f.price, dec!(0.5));
        assert_eq!(f.strategy, StrategyId::ParityMaker);
        let ack = fill_rx.recv().await.expect("Cancelled ack after the fills");
        assert_eq!(ack.side, FillSide::Cancelled);
        // The claim is kept until the fill is durable: the release
        // is done by the reconciler (pending + order_index).
        assert_eq!(ledger.total_exposure(), dec!(1.5));
        assert!(executor.active_gtd.is_empty());
        assert!(!executor.unknown_attempts().is_empty());
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());

        // The fill lands (fill_sink applies it) → the reconciler resolves it.
        state.apply_fill(&f).await.unwrap();
        executor.reconcile_unknowns().await;
        assert!(
            !state.has_unresolved_unknown("cid-1").await.unwrap(),
            "durable fills ⇒ the pending is resolved"
        );
        assert!(executor.unknown_attempts().is_empty());
    }

    /// The original gap: `cancel` Ok + matched>0 but the tape does not yet
    /// publish the trade. Previously the claim was released and the position
    /// was invisible if user_ws lost the fill. Now: the claim is kept, plus a
    /// durable Unknown pending and an order_index; the reconciler lands the
    /// fills in the next cycle.
    #[tokio::test]
    async fn cancel_ok_with_matched_without_trades_keeps_claim() {
        let (mut executor, ledger, state, mut fill_rx, cancel) = executor_cancel_probe(
            Some(RouterOrderStatus::Gone { matched: dec!(3) }),
            Some(vec![]),
        )
        .await;
        executor.handle_cancel(&cancel).await;

        assert_eq!(
            ledger.total_exposure(),
            dec!(1.5),
            "claim kept: matched>0 without trades cannot release exposure"
        );
        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "durable Unknown pending so that the reconciler lands the fills"
        );
        assert!(!executor.unknown_attempts().is_empty());
        assert!(
            executor.known_order_ids.contains("oid-1"),
            "order_index must link the oid for reconciliation"
        );
        let ack = fill_rx.recv().await.expect("ack Cancelled");
        assert_eq!(ack.side, FillSide::Cancelled);
    }

    /// Cancel Ok but the order is still live (the venue accepted the
    /// cancel without applying it): the claim is kept and the entry is
    /// re-registered so that the next cancel or resync retries it.
    #[tokio::test]
    async fn cancel_ok_but_live_order_re_registers() {
        let (mut executor, ledger, _state, _fill_rx, cancel) = executor_cancel_probe(
            Some(RouterOrderStatus::Live { remaining: dec!(3) }),
            Some(vec![]),
        )
        .await;
        executor.handle_cancel(&cancel).await;

        assert_eq!(ledger.total_exposure(), dec!(1.5));
        assert!(
            executor
                .active_gtd
                .contains_key("cid-1:yes-tok:parity_maker"),
            "the order is still live — it must be re-registered"
        );
    }

    /// Indeterminate `order_status` (`None`: network error, status not
    /// mappable, router without support): fail-closed. The claim is kept via a
    /// durable Unknown pending; it is never released blindly.
    #[tokio::test]
    async fn cancel_ok_with_indeterminate_status_keeps_claim() {
        let (mut executor, ledger, state, _fill_rx, cancel) =
            executor_cancel_probe(None, None).await;
        executor.handle_cancel(&cancel).await;

        assert_eq!(ledger.total_exposure(), dec!(1.5));
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        assert!(!executor.unknown_attempts().is_empty());
    }

    /// Pruning of `known_order_ids`: the ids of orders that are live
    /// on the destination (even if their order_index row is old) and those
    /// recorded less than 24 h ago (late user_ws trades) are kept; the rest
    /// is pruned, so the set stops growing without limit.
    #[tokio::test]
    async fn purge_prunes_known_order_ids_keeps_live_and_recent() {
        let (mut executor, sim, _ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        // Real live order in the sim (without it the filled-GTD purge would remove it).
        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::Limit {
                    price: dec!(0.5),
                    size: dec!(3),
                    tif: TimeInForce::Gtd,
                },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: None,
            })
            .await;
        let live_oid = res.order_id.expect("GTD must be accepted by the sim");
        executor
            .restore_gtd(
                "cid-1",
                "yes-tok",
                live_oid.clone(),
                dec!(1.5),
                intent.clear_at,
                StrategyId::ParityMaker,
            )
            .await;

        for oid in [&live_oid, "oid-recent", "oid-stale-ts"] {
            state
                .record_order_index(oid, None, StrategyId::ParityMaker, "cid-1", "yes-tok")
                .await
                .unwrap();
        }
        let old = chrono::Utc::now().timestamp_millis() - 2 * KNOWN_ORDER_IDS_RETENTION_MS;
        state
            .test_set_order_index_ts_ms(&live_oid, old)
            .await
            .unwrap();
        state
            .test_set_order_index_ts_ms("oid-stale-ts", old)
            .await
            .unwrap();

        for oid in [&live_oid, "oid-recent", "oid-stale-ts", "oid-no-index"] {
            executor.known_order_ids.insert((*oid).to_string());
        }

        executor.purge_and_resync().await;

        assert!(
            executor.known_order_ids.contains(&live_oid),
            "live order on the destination: kept even if its order_index is stale"
        );
        assert!(
            executor.known_order_ids.contains("oid-recent"),
            "recent order_index: a late user_ws trade may still arrive"
        );
        assert!(
            !executor.known_order_ids.contains("oid-stale-ts"),
            "order_index >24 h and dead order: pruned"
        );
        assert!(
            !executor.known_order_ids.contains("oid-no-index"),
            "id with no row in order_index and no live order: pruned"
        );
    }

    /// `purge_and_resync` must call `purge_stale_gtd_orders` so that
    /// orphan `gtd_orders` rows (1 h cutoff) do not stay around forever:
    /// the safety net existed but was never hooked into the executor's
    /// periodic cycle.
    #[tokio::test]
    async fn purge_and_resync_purges_orphan_gtd_orders() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        // Row deliberately old (well above the 3600 s cutoff).
        let old_ts = chrono::Utc::now().timestamp_millis() - 999_999_000;
        state
            .insert_gtd_order_row_for_test("old-1", "cid-orphan_fill", "arb", old_ts)
            .await
            .unwrap();

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state.clone(),
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );

        executor.purge_and_resync().await;

        assert!(state.get_gtd_strategy("old-1").await.unwrap().is_none(),
            "purge_and_resync must purge old gtd_orders rows via purge_stale_gtd_orders");
    }

    /// `handle_sell` must not silently overwrite an entry in
    /// `active_gtd` of a live BUY GTD on the same token: it must
    /// cancel it explicitly before registering the new one.
    #[tokio::test]
    async fn handle_sell_does_not_silently_overwrite_live_buy_gtd() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        let buy_gtd = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Buy,
            price: dec!(0.5),
            size: dec!(3),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.5)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        };
        executor.handle_buy(buy_gtd).await; // active_gtd[key] = oid-1

        let sell_gtd = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Sell {
                size: Some(dec!(3)),
            },
            price: dec!(0.55),
            size: dec!(3),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.55)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        };
        executor.handle_sell(sell_gtd).await;

        let ev = events.lock().clone();
        assert_eq!(ev, vec!["submit:oid-1", "submit:oid-2", "cancel:oid-1"],
            "the oldest buy GTD must be cancelled explicitly, not silently overwritten; sequence: {ev:?}");
    }

    /// ThetaCapture selling the LOSING side (`sell_loser_side`) is
    /// a naked short: it never bought those shares. It must add exposure
    /// ($1/share, the maximum downside) to the ledger, invisible before the fix.
    /// Contrast: an unwind of ParityMaker (selling its OWN inventory) still
    /// adds nothing.
    #[tokio::test]
    async fn tc_sell_losing_side_adds_exposure_to_ledger() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AsymmetricRouter),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        let tc_naked_sell = OrderIntent {
            market_cid: "cid-tc".into(),
            token_id: "loser-tok".into(),
            side: OrderSide::Sell {
                size: Some(dec!(5)),
            },
            price: dec!(0.10),
            size: dec!(5),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.10)),
            second_leg: None,
            strategy: StrategyId::ThetaCapture,
            resting: true,
            real_clear_at: None,
        };
        executor.handle_sell(tc_naked_sell).await;
        assert_eq!(
            ledger.total_exposure(),
            dec!(5),
            "the naked TC short must add $1/share of downside to the ledger"
        );

        let pm_unwind = OrderIntent {
            market_cid: "cid-pm".into(),
            token_id: "pm-tok".into(),
            side: OrderSide::Sell {
                size: Some(dec!(5)),
            },
            price: dec!(0.45),
            size: dec!(5),
            clear_at,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(dec!(0.45)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        };
        executor.handle_sell(pm_unwind).await;
        assert_eq!(ledger.total_exposure(), dec!(5),
            "a PM unwind (own inventory) must NOT add new exposure: it stays at 5, only the TC one");
    }

    /// An order rejected by the router must ALSO consume the
    /// strategy's rate limit: before, `record_order` was only called on
    /// the success path.
    #[tokio::test]
    async fn rejected_order_consumes_rate_limit() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        risk.set_strategy_limits(StrategyId::Arb, 1, dec!(1000), rust_decimal::Decimal::ZERO); // 1 order/hour
        let ledger = BracketLedger::new();

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AlwaysRejectRouter),
            risk.clone(),
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        executor
            .handle_buy(OrderIntent {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                side: OrderSide::Buy,
                price: dec!(0.5),
                size: dec!(3),
                clear_at,
                signal: dec!(0),
                neg_risk: false,
                limit_price: Some(dec!(0.5)),
                second_leg: None,
                strategy: StrategyId::Arb,
                resting: false,
                real_clear_at: None,
            })
            .await;

        assert!(
            !risk.check_order_limit(StrategyId::Arb, 1),
            "after ONE rejected order, the 1/hour quota must be exhausted"
        );
    }

    /// The ledger must set the bracket's `clear_at` from the REAL
    /// close (`real_clear_at`), not from a synthetic GTD expiry buffer
    /// (e.g. PCS-2): otherwise `purge_expired` could delete the claim long
    /// before the bracket actually closes.
    #[tokio::test]
    async fn ledger_uses_real_clear_at_not_the_synthetic() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AsymmetricRouter),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let now = chrono::Utc::now().timestamp();
        let synthetic_clear_at = now + 60; // synthetic GTD expiry buffer (e.g. PCS-2)
        let real_clear_at = now + 99_999; // real market close

        executor
            .handle_buy(OrderIntent {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                side: OrderSide::Buy,
                price: dec!(0.5),
                size: dec!(3),
                clear_at: synthetic_clear_at,
                signal: dec!(0),
                neg_risk: false,
                limit_price: Some(dec!(0.5)),
                second_leg: None,
                strategy: StrategyId::PostCloseSweep,
                resting: true,
                real_clear_at: Some(real_clear_at),
            })
            .await;

        // Purge at t = now+70: the synthetic one has passed, but NOT the real one. The bracket
        // must stay alive if the ledger used real_clear_at.
        ledger.purge_expired(now + 70);
        assert_eq!(
            ledger.total_exposure(),
            dec!(1.5),
            "the REAL clear_at, not the synthetic one, must govern the bracket purge"
        );
    }

    /// `purge_and_resync` must remove from the local mapping (`active_gtd`)
    /// any GTD that is NO LONGER live on the destination (it filled) but was never
    /// requoted or cancelled. Without this, the final resync below (which
    /// rebuilds from `active_gtd` + DB positions) adds it back again,
    /// double-counting the claim.
    #[tokio::test]
    async fn purge_and_resync_removes_filled_gtd_from_local_map() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();

        // The router reports ZERO live GTDs, so the "ghost" GTD below
        // counts as filled.
        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(FakeLiveIdsRouter {
                live_ids: std::collections::HashSet::new(),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd(
                "cid-1",
                "yes-tok",
                "oid-phantom".into(),
                dec!(1.5),
                clear_at,
                StrategyId::NearCertain,
            )
            .await;
        ledger.force_add_exposure(
            "cid-1",
            clear_at,
            "yes-tok",
            dec!(1.5),
            StrategyId::NearCertain,
        );
        assert_eq!(ledger.total_exposure(), dec!(1.5));

        executor.purge_and_resync().await;

        assert_eq!(ledger.total_exposure(), rust_decimal::Decimal::ZERO,
            "the claim of the filled GTD (no longer live on the router) must be released, not re-added");
    }

    // ── Unknown reconciler ──────────────────────────────────────────

    /// Executor over `SimulatedRouter`: the sim knows the real state of each
    /// order (`pending`/`exec_log`), so `order_status`/`open_gtd_orders`
    /// exercise the same path as `ClobRouter` in live.
    async fn executor_with_sim() -> (
        Executor,
        Arc<crate::order::sim_router::SimulatedRouter>,
        Arc<BracketLedger>,
        Arc<StateStore>,
        mpsc::UnboundedReceiver<Fill>,
    ) {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let (sim_fill_tx, _sim_fill_rx) = mpsc::unbounded_channel();
        let (sim_wallet_tx, _sim_wallet_rx) = watch::channel(None);
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let sim = crate::order::sim_router::SimulatedRouter::new(
            0,
            1.0,
            0,
            rust_decimal::Decimal::ZERO,
            1.0,
            sim_fill_tx,
            Arc::new(crate::market_feed::MarketRegistry::new()),
            dec!(100000),
            sim_wallet_tx,
        );
        let router: Arc<dyn OrderRouter> = sim.clone();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let executor = Executor::new(
            intent_rx,
            fill_tx,
            router,
            risk,
            state.clone(),
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );
        (executor, sim, ledger, state, fill_rx)
    }

    fn intent_gtd(
        token: &str,
        price: rust_decimal::Decimal,
        size: rust_decimal::Decimal,
    ) -> OrderIntent {
        OrderIntent {
            market_cid: "cid-1".into(),
            token_id: token.into(),
            side: OrderSide::Buy,
            price,
            size,
            clear_at: chrono::Utc::now().timestamp() + 300,
            signal: dec!(0),
            neg_risk: false,
            limit_price: Some(price),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: true,
            real_clear_at: None,
        }
    }

    /// A resting GTD whose POST lost its response (pending journal entry,
    /// without a known `order_id`) is matched against the destination's live orders
    /// by `(cid, token, side, price)`: it moves to `active_gtd` with the claim of the
    /// remainder and the bracket is unblocked. The pending carries the declared
    /// side `buy` (the match now requires an explicit side).
    #[tokio::test]
    async fn reconcile_unknown_matches_live_gtd_without_order_id() {
        let (mut executor, sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        // The order exists on the destination: this models a POST that arrived but whose
        // response was lost (the journal stayed pending without order_index).
        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::Limit {
                    price: dec!(0.5),
                    size: dec!(3),
                    tif: TimeInForce::Gtd,
                },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: Some("unk-test-1".into()),
            })
            .await;
        let oid = res.order_id.expect("GTD must be accepted by the sim");

        state
            .record_unknown_attempt_pending_meta(
                "unk-test-1",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                true,
                "test",
                Some(crate::state::to_micros(dec!(0.5))),
                Some(crate::state::to_micros(dec!(3))),
                Some(intent.clear_at),
                Some("buy"),
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-test-1");
        ledger
            .try_add_exposure(
                "cid-1",
                intent.clear_at,
                "yes-tok",
                dec!(1.5),
                dec!(1000),
                StrategyId::ParityMaker,
            )
            .unwrap();
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());

        executor.reconcile_unknowns().await;

        assert!(
            !state.has_unresolved_unknown("cid-1").await.unwrap(),
            "the matched live order must resolve the pending and unblock the bracket"
        );
        assert!(executor.unknown_attempts().is_empty());
        let key = limit_key("cid-1", "yes-tok", StrategyId::ParityMaker);
        let (registered_oid, claim, _, _) = executor
            .active_gtd
            .get(&key)
            .expect("the reconciled GTD must end up in active_gtd");
        assert_eq!(*registered_oid, oid);
        assert_eq!(*claim, dec!(1.5), "remainder 3 × price 0.5");
        assert!(executor.known_order_ids.contains(&oid));
    }

    /// Known `order_id` (order_index) + order already executed on the
    /// destination → `matched` resolution; the claim is NOT released (the position
    /// picks it up when the fill lands, deduplicated by exec_id).
    #[tokio::test]
    async fn reconcile_unknown_matched_keeps_claim() {
        let (mut executor, sim, ledger, state, mut fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::Limit {
                    price: dec!(0.5),
                    size: dec!(3),
                    tif: TimeInForce::Gtd,
                },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: Some("unk-test-2".into()),
            })
            .await;
        let oid = res.order_id.unwrap();

        // The order fills in the sim: a book with an ask at our price consumes the queue.
        sim.tick_book(&BookSnapshot {
            market_condition_id: "cid-1".into(),
            yes: crate::types::TokenBook {
                token_id: "yes-tok".into(),
                asks: crate::types::HalfBook {
                    levels: vec![crate::types::Level {
                        price: dec!(0.5),
                        size: dec!(10),
                    }],
                },
                ..Default::default()
            },
            no: crate::types::TokenBook {
                token_id: "no-tok".into(),
                ..Default::default()
            },
            clear_at: intent.clear_at,
            slug: "test".into(),
            ..Default::default()
        });

        // Pending journal entry + order_index (the response arrived but the journal
        // stayed pending, e.g. a crash before `mark_unknown_resolved`).
        state
            .record_unknown_attempt_pending(
                "unk-test-2",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                true,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        state
            .record_order_index(
                &oid,
                Some("unk-test-2"),
                StrategyId::ParityMaker,
                "cid-1",
                "yes-tok",
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-test-2");
        ledger
            .try_add_exposure(
                "cid-1",
                intent.clear_at,
                "yes-tok",
                dec!(1.5),
                dec!(1000),
                StrategyId::ParityMaker,
            )
            .unwrap();

        executor.reconcile_unknowns().await;

        // The pending is NOT resolved when emitting: it waits for the fill
        // to be durable in `fills` (a crash between enqueue and persist must not
        // leave the journal closed without a position).
        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "fill emitted but not yet durable ⇒ the pending stays alive"
        );
        assert_eq!(
            ledger.total_exposure(),
            dec!(1.5),
            "the claim is NOT released — the position picks it up when the fill lands"
        );
        // The fill is synthesized from the venue's trade tape, so it does not
        // depend on user_ws delivering it.
        let fill = fill_rx
            .try_recv()
            .expect("matched must emit the fill synthesized from the tape");
        assert_eq!(fill.side, FillSide::Buy);
        assert_eq!(fill.order_id.as_deref(), Some(oid.as_str()));
        assert!(fill.exec_id.is_some(), "exec_id = trade_id for dedup");
        assert_eq!(fill.local_order_id.as_deref(), Some("unk-test-2"));

        // The fill lands (fill_sink) → the next cycle resolves it.
        state.apply_fill(&fill).await.unwrap();
        executor.reconcile_unknowns().await;
        assert!(
            !state.has_unresolved_unknown("cid-1").await.unwrap(),
            "durable fill ⇒ the pending is resolved"
        );
        assert!(executor.unknown_attempts().is_empty());
        assert!(
            fill_rx.try_recv().is_err(),
            "the fill already durable is not re-emitted"
        );
    }

    /// Unknown taker reconciliation: a FOK whose POST lost its response is
    /// resolved from the trade tape: the trade's `taker_order_id` returns the lost
    /// order_id, `order_index` is written and the attempt becomes
    /// `matched-trade`. The claim is kept (the fill lands via user_ws/backfill
    /// with dedup by exec_id).
    #[tokio::test]
    async fn reconcile_unknown_taker_with_trade_resolves_matched() {
        let (mut executor, sim, ledger, state, mut fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        // The order DID arrive and executed: the sim has it in its executions tape
        // (what /data/trades would be in live).
        sim.tick_book(&BookSnapshot {
            market_condition_id: "cid-1".into(),
            yes: crate::types::TokenBook {
                token_id: "yes-tok".into(),
                asks: crate::types::HalfBook {
                    levels: vec![crate::types::Level {
                        price: dec!(0.5),
                        size: dec!(10),
                    }],
                },
                ..Default::default()
            },
            no: crate::types::TokenBook {
                token_id: "no-tok".into(),
                ..Default::default()
            },
            clear_at: intent.clear_at,
            slug: "test".into(),
            ..Default::default()
        });
        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::Limit {
                    price: dec!(0.5),
                    size: dec!(3),
                    tif: TimeInForce::Fok,
                },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: Some("unk-taker-1".into()),
            })
            .await;
        let oid = res.order_id.expect("FOK must fill in the sim");

        // Taker journal pending (resting=0) without order_id: ACK lost.
        state
            .record_unknown_attempt_pending(
                "unk-taker-1",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                false,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        // Past the trades API publication grace period.
        state
            .test_set_unknown_ts_ms(
                "unk-taker-1",
                chrono::Utc::now().timestamp_millis() - TAKER_TRADES_GRACE_MS - 1_000,
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-taker-1");
        ledger
            .try_add_exposure(
                "cid-1",
                intent.clear_at,
                "yes-tok",
                dec!(1.5),
                dec!(1000),
                StrategyId::ParityMaker,
            )
            .unwrap();

        executor.reconcile_unknowns().await;

        // Emitting is not enough: the pending stays alive until the fill
        // is durable in `fills` (crash window closed).
        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "fill emitted but not yet durable ⇒ pending"
        );
        assert!(executor.known_order_ids.contains(&oid));
        assert_eq!(
            state
                .order_index_lookup(&oid)
                .await
                .unwrap()
                .map(|(_, local)| local),
            Some(Some("unk-taker-1".to_string())),
            "order_index must link the recovered order_id to the attempt"
        );
        assert_eq!(
            ledger.total_exposure(),
            dec!(1.5),
            "matched-trade keeps the claim until the fill lands"
        );

        // The reconciliation synthesizes the Fill from the evidence of the
        // trade: it closes the window in which the strategy would re-emit the
        // same edge with the real position already open on the venue.
        let fill = fill_rx
            .try_recv()
            .expect("matched-trade must emit the synthesized fill");
        assert_eq!(fill.side, FillSide::Buy);
        assert_eq!(fill.market_cid, "cid-1");
        assert_eq!(fill.token_id, "yes-tok");
        assert_eq!(fill.order_id.as_deref(), Some(oid.as_str()));
        assert!(
            fill.exec_id.is_some(),
            "exec_id = trade_id for dedup against user_ws/backfill"
        );
        assert_eq!(
            fill.local_order_id.as_deref(),
            Some("unk-taker-1"),
            "local_order_id = attempt_id of the journal"
        );
        assert_eq!(fill.strategy, StrategyId::ParityMaker);
        assert!(fill_rx.try_recv().is_err(), "only one fill per trade");

        // Lands in `fills` (fill_sink) → the next cycle resolves it and does not
        // re-emit (it is already durable).
        state.apply_fill(&fill).await.unwrap();
        executor.reconcile_unknowns().await;
        assert!(
            !state.has_unresolved_unknown("cid-1").await.unwrap(),
            "durable fill ⇒ the taker pending is resolved"
        );
        assert!(executor.unknown_attempts().is_empty());
        assert!(fill_rx.try_recv().is_err(), "already durable: not re-emitted");
    }

    /// A taker Unknown WITHOUT an own trade in the window, once the long grace period has passed,
    /// proves the POST never arrived → claim released, journal `no-trade`.
    #[tokio::test]
    async fn reconcile_unknown_taker_without_trade_releases_claim() {
        let (mut executor, _sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        // No submit to the sim: the tape is empty, so the order never existed.
        state
            .record_unknown_attempt_pending(
                "unk-taker-2",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                false,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        state
            .test_set_unknown_ts_ms(
                "unk-taker-2",
                chrono::Utc::now().timestamp_millis() - TAKER_NO_TRADE_EXPIRY_MS - 1_000,
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-taker-2");
        ledger
            .try_add_exposure(
                "cid-1",
                intent.clear_at,
                "yes-tok",
                dec!(1.5),
                dec!(1000),
                StrategyId::ParityMaker,
            )
            .unwrap();

        // A single empty query is NOT enough: two consecutive cycles
        // without a trade in the window are needed before releasing (a transient
        // API lag cannot produce a false "never arrived").
        executor.reconcile_unknowns().await;
        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "first empty query = only one strike — still pending"
        );
        assert_eq!(ledger.total_exposure(), dec!(1.5));

        executor.reconcile_unknowns().await;
        assert!(!state.has_unresolved_unknown("cid-1").await.unwrap());
        assert!(executor.unknown_attempts().is_empty());
        assert_eq!(
            ledger.total_exposure(),
            dec!(0),
            "no trade after the grace period and two confirmations — claim released"
        );
    }

    /// The short grace period protects against the publication latency of
    /// /data/trades: a recent taker attempt with an empty tape is NOT
    /// resolved yet, since it could be an executed fill that the API does not index yet.
    #[tokio::test]
    async fn reconcile_unknown_taker_recent_stays_pending() {
        let (mut executor, _sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        state
            .record_unknown_attempt_pending(
                "unk-taker-3",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                false,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        // ts_ms = now (just journaled), so it is within the grace period.
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-taker-3");
        ledger
            .try_add_exposure(
                "cid-1",
                intent.clear_at,
                "yes-tok",
                dec!(1.5),
                dec!(1000),
                StrategyId::ParityMaker,
            )
            .unwrap();

        executor.reconcile_unknowns().await;

        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "within the grace period the attempt stays pending"
        );
        assert_eq!(ledger.total_exposure(), dec!(1.5));
    }

    /// Two pending taker attempts for the same token with overlapping windows and
    /// ONE single trade on the tape: the match is ambiguous → neither is resolved
    /// (fail-closed: better pending than a wrong attribution).
    #[tokio::test]
    async fn reconcile_unknown_taker_ambiguous_stays_pending() {
        let (mut executor, sim, _ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        sim.tick_book(&BookSnapshot {
            market_condition_id: "cid-1".into(),
            yes: crate::types::TokenBook {
                token_id: "yes-tok".into(),
                asks: crate::types::HalfBook {
                    levels: vec![crate::types::Level {
                        price: dec!(0.5),
                        size: dec!(10),
                    }],
                },
                ..Default::default()
            },
            no: crate::types::TokenBook::default(),
            clear_at: intent.clear_at,
            slug: "test".into(),
            ..Default::default()
        });
        sim.submit(OrderRequest {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            kind: OrderKind::Limit {
                price: dec!(0.5),
                size: dec!(3),
                tif: TimeInForce::Fok,
            },
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: intent.clear_at,
            signal: dec!(0),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        })
        .await;

        let old = chrono::Utc::now().timestamp_millis() - TAKER_TRADES_GRACE_MS - 1_000;
        for aid in ["unk-taker-A", "unk-taker-B"] {
            state
                .record_unknown_attempt_pending(
                    aid,
                    "cid-1",
                    "yes-tok",
                    "parity_maker",
                    crate::state::to_micros(dec!(1.5)),
                    false,
                    "test",
                    crate::state::to_micros(dec!(0.5)),
                    crate::state::to_micros(dec!(3)),
                    intent.clear_at,
                )
                .await
                .unwrap();
            state.test_set_unknown_ts_ms(aid, old).await.unwrap();
        }

        executor.reconcile_unknowns().await;

        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "one trade disputed by two attempts resolves neither"
        );
    }

    /// `trades_since` unsupported (the trait default → `None`) is
    /// indeterminate: the taker pending is NOT resolved, not even to
    /// `no-trade`, once it is past the long grace period. Without tape evidence
    /// there is no proof that the POST did not arrive. Fail-closed.
    #[tokio::test]
    async fn reconcile_unknown_taker_without_trades_since_stays_pending() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));
        // RecordingRouter does NOT implement trades_since → trait default (None).
        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events,
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state.clone(),
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        state
            .record_unknown_attempt_pending(
                "unk-taker-4",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                false,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        // Well beyond the long grace period: if None were treated as "no
        // trades", it would resolve to no-trade here.
        state
            .test_set_unknown_ts_ms(
                "unk-taker-4",
                chrono::Utc::now().timestamp_millis() - TAKER_NO_TRADE_EXPIRY_MS - 60_000,
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-taker-4");
        ledger
            .try_add_exposure(
                "cid-1",
                intent.clear_at,
                "yes-tok",
                dec!(1.5),
                dec!(1000),
                StrategyId::ParityMaker,
            )
            .unwrap();

        executor.reconcile_unknowns().await;

        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "without a trades query there is no evidence of no-trade — still pending"
        );
        assert_eq!(
            ledger.total_exposure(),
            dec!(1.5),
            "the claim is kept: indeterminate is not absence"
        );
    }

    /// Known `order_id` + the destination does NOT know the order (it never
    /// arrived) → definitive outcome, owner's claim released, bracket unblocked.
    #[tokio::test]
    async fn reconcile_unknown_not_found_releases_claim() {
        let (mut executor, _sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        state
            .record_unknown_attempt_pending(
                "unk-test-3",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                true,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        state
            .record_order_index(
                "oid-phantom",
                Some("unk-test-3"),
                StrategyId::ParityMaker,
                "cid-1",
                "yes-tok",
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-test-3");
        ledger
            .try_add_exposure(
                "cid-1",
                intent.clear_at,
                "yes-tok",
                dec!(1.5),
                dec!(1000),
                StrategyId::ParityMaker,
            )
            .unwrap();

        executor.reconcile_unknowns().await;

        assert!(!state.has_unresolved_unknown("cid-1").await.unwrap());
        assert_eq!(
            ledger.total_exposure(),
            rust_decimal::Decimal::ZERO,
            "NotFound = the order never existed → the claim is released"
        );
    }

    /// A GTD with a partial fill must NOT be counted twice in the
    /// resync. The partial position is already in the DB and the claim must be the
    /// real REMAINDER (`remaining × price` from the destination), not the original
    /// cost: `position + remainder` = exact total exposure.
    #[tokio::test]
    async fn resync_claim_gtd_uses_remainder_after_partial_fill() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let (sim_fill_tx, mut sim_fill_rx) = mpsc::unbounded_channel();
        let (sim_wallet_tx, _sim_wallet_rx) = watch::channel(None);
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let sim = crate::order::sim_router::SimulatedRouter::new(
            0,
            1.0,
            0,
            rust_decimal::Decimal::ZERO,
            1.0,
            sim_fill_tx,
            Arc::new(crate::market_feed::MarketRegistry::new()),
            dec!(100000),
            sim_wallet_tx,
        );
        let router: Arc<dyn OrderRouter> = sim.clone();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            router,
            risk,
            state.clone(),
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );

        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));
        executor.handle_buy(intent.clone()).await;
        let key = limit_key("cid-1", "yes-tok", StrategyId::ParityMaker);
        let (_, cost, ..) = executor
            .active_gtd
            .get(&key)
            .expect("the GTD must end up in active_gtd");
        assert_eq!(*cost, dec!(1.5), "initial claim = original cost 3×0.5");

        // Partial fill: the book offers 1 share at our price → 2 remain
        // resting. The partial position is recorded in the DB (as fill_sink would)
        // BEFORE the resync.
        sim.tick_book(&BookSnapshot {
            market_condition_id: "cid-1".into(),
            yes: crate::types::TokenBook {
                token_id: "yes-tok".into(),
                asks: crate::types::HalfBook {
                    levels: vec![crate::types::Level {
                        price: dec!(0.5),
                        size: dec!(1),
                    }],
                },
                ..Default::default()
            },
            no: crate::types::TokenBook {
                token_id: "no-tok".into(),
                ..Default::default()
            },
            clear_at: intent.clear_at,
            slug: "test".into(),
            ..Default::default()
        });
        let fill = sim_fill_rx
            .recv()
            .await
            .expect("the tick must emit the partial fill");
        assert_eq!(fill.size, dec!(1));
        state.apply_fill(&fill).await.unwrap();

        executor.purge_and_resync().await;

        // position 0.5 + remainder 2×0.5 = 1.5. Before the fix the resync
        // gave 0.5 + 1.5 (original cost) = 2.0: the fill was counted twice.
        assert_eq!(ledger.total_exposure(), dec!(1.5));
        let (_, cost, ..) = executor.active_gtd.get(&key).unwrap();
        assert_eq!(*cost, dec!(1.0), "refreshed claim = remainder 2×0.5");
    }

    /// Regression: the resync must NOT delete the claim of an Unknown still
    /// pending. Before, `ledger.clear()` only rebuilt positions and
    /// `active_gtd`, so the conservative protection of a lost POST
    /// disappeared after ~30 s even though the order could still be live.
    #[tokio::test]
    async fn resync_keeps_claim_of_pending_unknown() {
        // AlwaysRejectRouter does not implement order_status (default `None`) →
        // the attempt is indeterminate and must stay pending WITH its claim.
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(1000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AlwaysRejectRouter),
            risk,
            state.clone(),
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));
        state
            .record_unknown_attempt_pending(
                "unk-test-4",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                true,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-test-4");
        ledger
            .try_add_exposure(
                "cid-1",
                intent.clear_at,
                "yes-tok",
                dec!(1.5),
                dec!(1000),
                StrategyId::ParityMaker,
            )
            .unwrap();

        executor.purge_and_resync().await;

        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "indeterminate query → still pending (fail-closed)"
        );
        assert_eq!(
            ledger.total_exposure(),
            dec!(1.5),
            "the claim of a pending Unknown must survive the resync"
        );
    }

    /// A quarantined fill whose `order_id` is not indexed is linked to the
    /// ONLY pending attempt of its `(cid, token)`: the fill is the evidence
    /// that the order exists, and `order_index` makes it attributable for the
    /// `fill_sink` reconciler.
    #[tokio::test]
    async fn link_quarantine_with_journal_attributes_fill() {
        let (mut executor, _sim, _ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));

        // Taker attempt (not resting) whose POST was lost: no order_id.
        state
            .record_unknown_attempt_pending(
                "unk-test-5",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                false,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        let mut intent_taker = intent.clone();
        intent_taker.resting = false;
        executor.register_unknown_attempt(&intent_taker, dec!(1.5), "test", "unk-test-5");

        // The fill arrived via user_ws BEFORE the order_id was known → quarantine.
        state
            .quarantine_unowned_fill(
                &Fill {
                    market_cid: "cid-1".into(),
                    token_id: "yes-tok".into(),
                    side: FillSide::Buy,
                    price: dec!(0.5),
                    size: dec!(3),
                    order_id: Some("oid-late".into()),
                    ts: chrono::Utc::now(),
                    mode: crate::types::Mode::Live,
                    neg_risk: false,
                    strategy: StrategyId::Arb, // declared, not verified
                    clear_at: intent.clear_at,
                    outcome_label: None,
                    hedge_id: None,
                    exec_id: Some("exec-late".into()),
                    local_order_id: None,
                },
                "test: owner not resolvable",
            )
            .await
            .unwrap();

        executor.link_quarantine_to_journal().await;

        let attribution = state.order_index_lookup("oid-late").await.unwrap();
        assert_eq!(
            attribution,
            Some((StrategyId::ParityMaker, Some("unk-test-5".to_string()))),
            "the quarantined fill must be linked to the single pending attempt"
        );
    }

    /// An Unknown whose `clear_at` has already passed (bracket closed) is resolved
    /// `expired_no_reconcile`, the owner's claim is released and it disappears from the
    /// mirror: it is the exit from the permanent blocker via `has_unresolved_unknown`.
    #[tokio::test]
    async fn watchdog_expires_unknown_whose_bracket_already_closed() {
        let (mut executor, _sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let past = chrono::Utc::now().timestamp() - 120;
        state
            .record_unknown_attempt_pending(
                "unk-watchdog-1",
                "cid-1",
                "yes-tok",
                "near_certain",
                crate::state::to_micros(dec!(5)),
                false,
                "lost POST",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(10)),
                past,
            )
            .await
            .unwrap();
        executor.seed_unknown_attempts_from_journal_pub().await;
        ledger.force_add_exposure("cid-1", past, "yes-tok", dec!(5), StrategyId::NearCertain);
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        assert_eq!(ledger.total_exposure(), dec!(5));
        assert_eq!(executor.unknown_attempts().len(), 1);

        executor.watchdog_unknowns().await;

        assert!(
            !state.has_unresolved_unknown("cid-1").await.unwrap(),
            "bracket closed → the watchdog resolves the pending and unblocks"
        );
        assert_eq!(
            ledger.total_exposure(),
            rust_decimal::Decimal::ZERO,
            "the expired Unknown's claim must be released"
        );
        assert!(
            executor.unknown_attempts().is_empty(),
            "the in-memory mirror must also be cleaned"
        );
    }

    /// A RECENT pending with its bracket still open is NOT resolved or released
    /// — the watchdog does not auto-resolve without evidence (fail-closed).
    #[tokio::test]
    async fn watchdog_does_not_touch_recent_pending_with_live_bracket() {
        let (mut executor, _sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let future = chrono::Utc::now().timestamp() + 600;
        state
            .record_unknown_attempt_pending(
                "unk-watchdog-2",
                "cid-1",
                "yes-tok",
                "near_certain",
                crate::state::to_micros(dec!(5)),
                false,
                "lost POST",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(10)),
                future,
            )
            .await
            .unwrap();
        executor.seed_unknown_attempts_from_journal_pub().await;
        ledger.force_add_exposure("cid-1", future, "yes-tok", dec!(5), StrategyId::NearCertain);

        executor.watchdog_unknowns().await;

        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        assert_eq!(ledger.total_exposure(), dec!(5));
        assert_eq!(executor.unknown_attempts().len(), 1);
    }

    /// A sell rejected by the router must emit a terminal
    /// `Rejected` Fill. Strategies that latch state when emitting a sell
    /// (PM/RF `unwind_sent`) are released only by that event.
    /// Without it, a rejected unwind sell left the leg mute until the
    /// settlement.
    #[tokio::test]
    async fn rejected_sell_emits_terminal_fill_rejected() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(10_000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                max_total_exposure_usdc: dec!(10_000),
                ..Default::default()
            },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(AlwaysRejectRouter),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger,
        );

        let intent = OrderIntent {
            market_cid: "cid-1".into(),
            token_id: "yes-tok".into(),
            side: OrderSide::Sell {
                size: Some(dec!(5)),
            },
            price: dec!(0.4),
            size: dec!(5),
            clear_at: chrono::Utc::now().timestamp() + 300,
            signal: dec!(0),
            neg_risk: false,
            limit_price: None,
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting: false,
            real_clear_at: None,
        };
        executor.handle_sell(intent).await;

        let fill = fill_rx
            .try_recv()
            .expect("a rejected sell must emit a terminal Rejected Fill");
        assert!(matches!(fill.side, FillSide::Rejected));
        assert_eq!(fill.token_id, "yes-tok");
        assert_eq!(fill.strategy, StrategyId::ParityMaker);
        assert!(
            fill.local_order_id.is_some(),
            "the Rejected must carry the sell's local identity"
        );
        // And nothing must remain registered as a live GTD.
        assert!(executor.active_gtd.is_empty());
    }

    /// A pre-submit gate must emit `GateRejected` keyed to the intent
    /// (token/price/size of the intent, not a generic per-token one) so that the
    /// strategy undoes exactly what it marked when emitting.
    #[tokio::test]
    async fn gate_kill_emits_gate_rejected_keyed_to_intent() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                // Without this, the kill gate is ignored in paper (default true).
                paper_disable_kill_switch: false,
                ..RiskConfig::default()
            },
            crate::types::Mode::Paper,
        );
        risk.kill_strategy(StrategyId::ParityMaker, "test kill");
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: events.clone(),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            BracketLedger::new(),
        );

        executor
            .handle_buy(OrderIntent {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                side: OrderSide::Buy,
                price: dec!(0.42),
                size: dec!(7),
                clear_at: chrono::Utc::now().timestamp() + 300,
                signal: dec!(0),
                neg_risk: false,
                limit_price: Some(dec!(0.42)),
                second_leg: None,
                strategy: StrategyId::ParityMaker,
                resting: true,
                real_clear_at: None,
            })
            .await;

        let fill = fill_rx
            .try_recv()
            .expect("a pre-submit gate must emit GateRejected");
        assert!(matches!(fill.side, FillSide::GateRejected));
        assert_eq!(fill.token_id, "yes-tok");
        assert_eq!(
            fill.price,
            dec!(0.42),
            "the ack must carry the intent's price"
        );
        assert_eq!(fill.size, dec!(7), "the ack must carry the marked size");
        assert_eq!(fill.strategy, StrategyId::ParityMaker);
        assert!(
            fill.order_id.is_none(),
            "the order never existed: no order_id"
        );
        assert!(events.lock().is_empty(), "there must be no submit at all");
    }

    /// A pair blocked by an early gate must acknowledge receipt for
    /// BOTH legs: the strategy marked state for each one when emitting.
    #[tokio::test]
    async fn gate_rejected_on_pair_emits_ack_for_both_legs() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(
            RiskConfig {
                // Without this, the kill gate is ignored in paper (default true).
                paper_disable_kill_switch: false,
                ..RiskConfig::default()
            },
            crate::types::Mode::Paper,
        );
        risk.kill_strategy(StrategyId::Arb, "test kill");

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: Arc::new(parking_lot::Mutex::new(Vec::new())),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            BracketLedger::new(),
        );

        let clear_at = chrono::Utc::now().timestamp() + 300;
        executor
            .handle_buy(OrderIntent {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                side: OrderSide::Buy,
                price: dec!(0.48),
                size: dec!(5),
                clear_at,
                signal: dec!(0),
                neg_risk: false,
                limit_price: None,
                second_leg: Some(Box::new(OrderIntent {
                    market_cid: "cid-1".into(),
                    token_id: "no-tok".into(),
                    side: OrderSide::Buy,
                    price: dec!(0.44),
                    size: dec!(5),
                    clear_at,
                    signal: dec!(0),
                    neg_risk: false,
                    limit_price: None,
                    second_leg: None,
                    strategy: StrategyId::Arb,
                    resting: false,
                    real_clear_at: None,
                })),
                strategy: StrategyId::Arb,
                resting: false,
                real_clear_at: None,
            })
            .await;

        let f1 = fill_rx.try_recv().expect("ack for the YES leg");
        let f2 = fill_rx.try_recv().expect("ack for the NO leg");
        assert!(matches!(f1.side, FillSide::GateRejected));
        assert!(matches!(f2.side, FillSide::GateRejected));
        let (yes, no) = if f1.token_id == "yes-tok" {
            (f1, f2)
        } else {
            (f2, f1)
        };
        assert_eq!(
            (yes.token_id, yes.price, yes.size),
            ("yes-tok".into(), dec!(0.48), dec!(5))
        );
        assert_eq!(
            (no.token_id, no.price, no.size),
            ("no-tok".into(), dec!(0.44), dec!(5))
        );
        assert!(fill_rx.try_recv().is_err(), "only two acks, one per leg");
    }

    /// The balance trim must NOT contaminate the ack: the
    /// `GateRejected` carries the size the strategy marked (the original intent's),
    /// not the trimmed one.
    #[tokio::test]
    async fn gate_balance_emits_marked_original_size() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        // Balance < 1 USDC → "balance" gate before any trim.
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(0.5)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(RiskConfig::default(), crate::types::Mode::Paper);

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: Arc::new(parking_lot::Mutex::new(Vec::new())),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            BracketLedger::new(),
        );

        executor
            .handle_buy(OrderIntent {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                side: OrderSide::Buy,
                price: dec!(0.5),
                size: dec!(9),
                clear_at: chrono::Utc::now().timestamp() + 300,
                signal: dec!(0),
                neg_risk: false,
                limit_price: Some(dec!(0.5)),
                second_leg: None,
                strategy: StrategyId::ParityMaker,
                resting: true,
                real_clear_at: None,
            })
            .await;

        let fill = fill_rx.try_recv().expect("ack for the balance gate");
        assert!(matches!(fill.side, FillSide::GateRejected));
        assert_eq!(
            fill.size,
            dec!(9),
            "the ack carries the marked size, not a trim"
        );
    }

    /// A `Cancel` against a key with no registered order must emit
    /// a `Cancelled` ack: it is the authoritative truth (there is no live order), and it is
    /// the only thing that releases the strategy's ghost state.
    #[tokio::test]
    async fn cancel_without_order_emits_ack_cancelled() {
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let risk = RiskGuard::new(RiskConfig::default(), crate::types::Mode::Paper);

        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(RecordingRouter {
                events: Arc::new(parking_lot::Mutex::new(Vec::new())),
                n: Arc::new(parking_lot::Mutex::new(0)),
            }),
            risk,
            state,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            BracketLedger::new(),
        );

        executor
            .handle_cancel(&OrderIntent {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                side: OrderSide::Cancel,
                price: dec!(0),
                size: dec!(0),
                clear_at: chrono::Utc::now().timestamp() + 300,
                signal: dec!(0),
                neg_risk: false,
                limit_price: None,
                second_leg: None,
                strategy: StrategyId::ParityMaker,
                resting: true,
                real_clear_at: None,
            })
            .await;

        let fill = fill_rx
            .try_recv()
            .expect("a cancel against nothing must be acknowledged with Cancelled");
        assert!(matches!(fill.side, FillSide::Cancelled));
        assert_eq!(fill.token_id, "yes-tok");
        assert_eq!(fill.strategy, StrategyId::ParityMaker);
    }

    // ── Coherent GTD claims per side ───────────────────────────────────────

    /// The claim refresh applies the contractual rule per SIDE:
    /// BUY = remaining×price, SELL of inventory (non-TC) = 0, short TC SELL
    /// = remainder (downside). Before, `remaining × price` for everything gave a
    /// claim >0 to the inventory sells and an undervalued claim to the short TC.
    #[tokio::test]
    async fn refresh_claims_applies_rule_per_side() {
        let router = Arc::new(D1ReconRouter::new());
        let (mut executor, _ledger, _state, _fill_rx) =
            executor_with_d1_router(router.clone()).await;
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "tok-buy", "oid-buy".into(), dec!(1.5), clear, StrategyId::ParityMaker)
            .await;
        executor
            .restore_gtd("cid-1", "tok-sell-pm", "oid-sell-pm".into(), dec!(0), clear, StrategyId::ParityMaker)
            .await;
        executor
            .restore_gtd("cid-1", "tok-sell-tc", "oid-sell-tc".into(), dec!(4), clear, StrategyId::ThetaCapture)
            .await;
        *router.open_orders.lock() = Some(vec![
            open_order("oid-buy", "cid-1", "tok-buy", OpenOrderSide::Buy, dec!(0.5), dec!(3)),
            open_order("oid-sell-pm", "cid-1", "tok-sell-pm", OpenOrderSide::Sell, dec!(0.7), dec!(2)),
            open_order("oid-sell-tc", "cid-1", "tok-sell-tc", OpenOrderSide::Sell, dec!(0.4), dec!(4)),
        ]);

        executor.refresh_gtd_claims().await;

        let k_buy = limit_key("cid-1", "tok-buy", StrategyId::ParityMaker);
        let k_pm = limit_key("cid-1", "tok-sell-pm", StrategyId::ParityMaker);
        let k_tc = limit_key("cid-1", "tok-sell-tc", StrategyId::ThetaCapture);
        assert_eq!(executor.active_gtd[&k_buy].1, dec!(1.5), "BUY: 3×0.5");
        assert_eq!(executor.active_gtd[&k_pm].1, dec!(0), "SELL inventory PM: claim 0");
        assert_eq!(executor.active_gtd[&k_tc].1, dec!(4), "short TC SELL: downside 4");

        // Partial fill on the destination: the remainder drops and the claim follows it.
        *router.open_orders.lock() = Some(vec![
            open_order("oid-buy", "cid-1", "tok-buy", OpenOrderSide::Buy, dec!(0.5), dec!(1)),
            open_order("oid-sell-pm", "cid-1", "tok-sell-pm", OpenOrderSide::Sell, dec!(0.7), dec!(1.25)),
            open_order("oid-sell-tc", "cid-1", "tok-sell-tc", OpenOrderSide::Sell, dec!(0.4), dec!(2.5)),
        ]);
        executor.refresh_gtd_claims().await;
        assert_eq!(executor.active_gtd[&k_buy].1, dec!(0.5), "partial BUY: 1×0.5");
        assert_eq!(executor.active_gtd[&k_pm].1, dec!(0));
        assert_eq!(executor.active_gtd[&k_tc].1, dec!(2.5), "partial TC: downside of the remainder");

        // Second stable pass: no invented or trimmed exposure.
        executor.refresh_gtd_claims().await;
        assert_eq!(executor.active_gtd[&k_buy].1, dec!(0.5));
        assert_eq!(executor.active_gtd[&k_pm].1, dec!(0));
        assert_eq!(executor.active_gtd[&k_tc].1, dec!(2.5));
    }

    /// Without valid evidence the claim is NOT rewritten: an order missing
    /// from the snapshot, an indeterminate side on the wire, or a `None` from the router
    /// keep the previous claim (fail-closed).
    #[tokio::test]
    async fn refresh_claims_indeterminate_evidence_keeps() {
        let router = Arc::new(D1ReconRouter::new());
        let (mut executor, _ledger, _state, _fill_rx) =
            executor_with_d1_router(router.clone()).await;
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "tok-a", "oid-a".into(), dec!(1.5), clear, StrategyId::ParityMaker)
            .await;
        executor
            .restore_gtd("cid-1", "tok-b", "oid-b".into(), dec!(1.5), clear, StrategyId::ParityMaker)
            .await;
        let k_a = limit_key("cid-1", "tok-a", StrategyId::ParityMaker);
        let k_b = limit_key("cid-1", "tok-b", StrategyId::ParityMaker);

        // oid-b with an indeterminate wire side, and oid-a missing from the snapshot.
        *router.open_orders.lock() = Some(vec![open_order(
            "oid-b", "cid-1", "tok-b", OpenOrderSide::Unknown, dec!(0.5), dec!(1),
        )]);
        executor.refresh_gtd_claims().await;
        assert_eq!(executor.active_gtd[&k_a].1, dec!(1.5), "missing → kept");
        assert_eq!(executor.active_gtd[&k_b].1, dec!(1.5), "unknown side → kept");

        // Listing not supported or failed → nothing is touched.
        *router.open_orders.lock() = None;
        executor.refresh_gtd_claims().await;
        assert_eq!(executor.active_gtd[&k_a].1, dec!(1.5));
        assert_eq!(executor.active_gtd[&k_b].1, dec!(1.5));
    }

    /// Sim projection. `open_gtd_orders` keeps the side (Limit=BUY,
    /// LimitSell=SELL). Before, they were merged and a sell matched as a buy.
    #[tokio::test]
    async fn sim_open_gtd_orders_preserves_side() {
        let (_executor, sim, _ledger, _state, _fill_rx) = executor_with_sim().await;
        let clear = chrono::Utc::now().timestamp() + 300;
        for (kind, tok) in [
            (
                OrderKind::Limit { price: dec!(0.5), size: dec!(3), tif: TimeInForce::Gtd },
                "tok-b",
            ),
            (
                OrderKind::LimitSell { price: dec!(0.7), size: dec!(2), tif: TimeInForce::Gtd },
                "tok-s",
            ),
        ] {
            let res = sim
                .submit(OrderRequest {
                    market_cid: "cid-1".into(),
                    token_id: tok.into(),
                    kind,
                    neg_risk: false,
                    strategy: StrategyId::ParityMaker,
                    clear_at: clear,
                    signal: dec!(0),
                    post_only: false,
                    real_clear_at: None,
                    local_order_id: None,
                })
                .await;
            assert!(res.order_id.is_some(), "the order must rest");
        }
        let orders = sim.open_gtd_orders().await.expect("the sim supports the listing");
        let buy = orders.iter().find(|o| o.token_id == "tok-b").unwrap();
        let sell = orders.iter().find(|o| o.token_id == "tok-s").unwrap();
        assert_eq!(buy.side, OpenOrderSide::Buy);
        assert_eq!(sell.side, OpenOrderSide::Sell);
        assert_eq!(buy.market_cid, "cid-1");
        assert_eq!(sell.market_cid, "cid-1");
    }

    /// A BUY pending does NOT match a resting SELL on the same
    /// token+price: before the fix the match ignored the side and assigned the
    /// other party's order to the buying attempt.
    #[tokio::test]
    async fn reconcile_pending_buy_does_not_match_sell_same_token_price() {
        let (mut executor, sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));
        // The venue has a resting SELL at that same price.
        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::LimitSell { price: dec!(0.5), size: dec!(3), tif: TimeInForce::Gtd },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: None,
            })
            .await;
        let sell_oid = res.order_id.expect("resting LimitSell GTD in the sim");
        state
            .record_unknown_attempt_pending_meta(
                "unk-sell-clash",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                true,
                "test",
                Some(crate::state::to_micros(dec!(0.5))),
                Some(crate::state::to_micros(dec!(3))),
                Some(intent.clear_at),
                Some("buy"),
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-sell-clash");
        ledger
            .try_add_exposure("cid-1", intent.clear_at, "yes-tok", dec!(1.5), dec!(1000), StrategyId::ParityMaker)
            .unwrap();

        executor.reconcile_unknowns().await;

        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "BUY pending vs live SELL = no match → still pending"
        );
        assert!(
            state.order_index_lookup(&sell_oid).await.unwrap().is_none(),
            "the foreign SELL must not be indexed to this attempt"
        );
        assert!(executor.active_gtd.is_empty(), "no GTD is adjudicated");
        assert_eq!(ledger.total_exposure(), dec!(1.5), "claim kept");
    }

    /// A pending WITHOUT a durable side (NULL: rows from before
    /// the side was stored, quarantine, cancel) is never matched by eye to a live order
    /// even if token+price coincide: it stays pending with its claim.
    #[tokio::test]
    async fn reconcile_pending_legacy_without_side_does_not_match() {
        let (mut executor, sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));
        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::Limit { price: dec!(0.5), size: dec!(3), tif: TimeInForce::Gtd },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: None,
            })
            .await;
        let oid = res.order_id.unwrap();
        // API legacy → side NULL.
        state
            .record_unknown_attempt_pending(
                "unk-legacy",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                true,
                "test",
                crate::state::to_micros(dec!(0.5)),
                crate::state::to_micros(dec!(3)),
                intent.clear_at,
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-legacy");
        ledger
            .try_add_exposure("cid-1", intent.clear_at, "yes-tok", dec!(1.5), dec!(1000), StrategyId::ParityMaker)
            .unwrap();

        executor.reconcile_unknowns().await;

        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "without a declared side there is no match — the pending is kept"
        );
        assert!(state.order_index_lookup(&oid).await.unwrap().is_none());
        assert_eq!(ledger.total_exposure(), dec!(1.5));
    }

    /// Two compatible resting pendings (same cid+token+price+side)
    /// from different attempts/strategies dispute the SAME live order:
    /// ambiguous: neither steals it, and both stay pending with their claim.
    #[tokio::test]
    async fn reconcile_two_compatible_pendings_neither_steals_order() {
        let (mut executor, sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));
        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::Limit { price: dec!(0.5), size: dec!(3), tif: TimeInForce::Gtd },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: None,
            })
            .await;
        let oid = res.order_id.unwrap();
        for (aid, strat) in [("unk-dup-a", "parity_maker"), ("unk-dup-b", "near_certain")] {
            state
                .record_unknown_attempt_pending_meta(
                    aid,
                    "cid-1",
                    "yes-tok",
                    strat,
                    crate::state::to_micros(dec!(1.5)),
                    true,
                    "test",
                    Some(crate::state::to_micros(dec!(0.5))),
                    Some(crate::state::to_micros(dec!(3))),
                    Some(intent.clear_at),
                    Some("buy"),
                )
                .await
                .unwrap();
        }
        let mut intent_nc = intent.clone();
        intent_nc.strategy = StrategyId::NearCertain;
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-dup-a");
        executor.register_unknown_attempt(&intent_nc, dec!(1.5), "test", "unk-dup-b");
        ledger
            .try_add_exposure("cid-1", intent.clear_at, "yes-tok", dec!(1.5), dec!(1000), StrategyId::ParityMaker)
            .unwrap();
        ledger
            .try_add_exposure("cid-1", intent.clear_at, "yes-tok", dec!(1.5), dec!(1000), StrategyId::NearCertain)
            .unwrap();

        executor.reconcile_unknowns().await;

        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "match disputed → both pendings are kept"
        );
        assert!(
            state.order_index_lookup(&oid).await.unwrap().is_none(),
            "the order is not adjudicated to either of them"
        );
        assert!(executor.active_gtd.is_empty());
        assert_eq!(ledger.total_exposure(), dec!(3), "both claims kept");
    }

    /// A live order already adjudicated in `order_index` to ANOTHER attempt
    /// cannot be taken by a different pending even if everything matches.
    #[tokio::test]
    async fn reconcile_does_not_reuse_oid_already_awarded() {
        let (mut executor, sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));
        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::Limit { price: dec!(0.5), size: dec!(3), tif: TimeInForce::Gtd },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: None,
            })
            .await;
        let oid = res.order_id.unwrap();
        // The order ALREADY has a durable owner: another attempt of another strategy.
        state
            .record_order_index(&oid, Some("unk-other"), StrategyId::NearCertain, "cid-1", "yes-tok")
            .await
            .unwrap();
        state
            .record_unknown_attempt_pending_meta(
                "unk-claim",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                true,
                "test",
                Some(crate::state::to_micros(dec!(0.5))),
                Some(crate::state::to_micros(dec!(3))),
                Some(intent.clear_at),
                Some("buy"),
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-claim");
        ledger
            .try_add_exposure("cid-1", intent.clear_at, "yes-tok", dec!(1.5), dec!(1000), StrategyId::ParityMaker)
            .unwrap();

        executor.reconcile_unknowns().await;

        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        assert_eq!(
            state.order_index_lookup(&oid).await.unwrap().map(|(_, l)| l),
            Some(Some("unk-other".to_string())),
            "the previous adjudication is not overwritten"
        );
        assert!(executor.active_gtd.is_empty());
    }

    /// Not even `known_order_ids` in memory is enough for a
    /// pending to take ownership of an order: the entry is already registered as
    //  known through another route (same principle as order_index).
    #[tokio::test]
    async fn reconcile_does_not_match_oid_already_known_in_memory() {
        let (mut executor, sim, ledger, state, _fill_rx) = executor_with_sim().await;
        let intent = intent_gtd("yes-tok", dec!(0.5), dec!(3));
        let res = sim
            .submit(OrderRequest {
                market_cid: "cid-1".into(),
                token_id: "yes-tok".into(),
                kind: OrderKind::Limit { price: dec!(0.5), size: dec!(3), tif: TimeInForce::Gtd },
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: intent.clear_at,
                signal: dec!(0),
                post_only: false,
                real_clear_at: None,
                local_order_id: None,
            })
            .await;
        let oid = res.order_id.unwrap();
        // Live order already registered as active (a different owner).
        executor
            .restore_gtd("cid-1", "yes-tok", oid.clone(), dec!(1.5), intent.clear_at, StrategyId::NearCertain)
            .await;
        state
            .record_unknown_attempt_pending_meta(
                "unk-late",
                "cid-1",
                "yes-tok",
                "parity_maker",
                crate::state::to_micros(dec!(1.5)),
                true,
                "test",
                Some(crate::state::to_micros(dec!(0.5))),
                Some(crate::state::to_micros(dec!(3))),
                Some(intent.clear_at),
                Some("buy"),
            )
            .await
            .unwrap();
        executor.register_unknown_attempt(&intent, dec!(1.5), "test", "unk-late");
        ledger
            .try_add_exposure("cid-1", intent.clear_at, "yes-tok", dec!(1.5), dec!(1000), StrategyId::ParityMaker)
            .unwrap();

        executor.reconcile_unknowns().await;

        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        // The restored GTD keeps its owner and its claim: nothing is usurped.
        let key = limit_key("cid-1", "yes-tok", StrategyId::NearCertain);
        assert_eq!(executor.active_gtd[&key].1, dec!(1.5));
        assert_eq!(executor.active_gtd[&key].0, oid);
    }

    // ── Uncertain cancel keeps claim and metadata ──────────────────────────

    /// `Ok + None` arm: cancel accepted but `order_status`
    /// indeterminate → durable pending with the original `cost` and NULL metadata
    /// (the `price=0` of the Cancel intent is NOT stamped). When the destination
    /// reports the order as live, the claim is re-evaluated with the evidence of the
    /// open order: never `remaining × 0`.
    #[tokio::test]
    async fn cancel_indeterminate_buy_keeps_claim_and_reconciles_with_evidence() {
        let router = Arc::new(D1ReconRouter::new());
        let (mut executor, ledger, state, _fill_rx) =
            executor_with_d1_router(router.clone()).await;
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "yes-tok", "oid-1".into(), dec!(1.5), clear, StrategyId::ParityMaker)
            .await;
        ledger.force_add_exposure("cid-1", clear, "yes-tok", dec!(1.5), StrategyId::ParityMaker);
        let cancel = cancel_intent(StrategyId::ParityMaker);

        // Phase 1: cancel Ok + indeterminate order_status → pending.
        executor.handle_cancel(&cancel).await;
        assert!(executor.active_gtd.is_empty(), "the order leaves active_gtd");
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        assert_eq!(ledger.total_exposure(), dec!(1.5), "claim kept in pending");
        let row = state
            .list_unresolved_unknowns()
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(row.cost_um, crate::state::to_micros(dec!(1.5)));
        assert_eq!(row.price_um, None, "the Cancel's price=0 is not stamped");
        assert_eq!(row.size_um, None);
        assert_eq!(row.side, None, "the original order's side is unknown");
        assert_eq!(row.clear_at, Some(clear), "the order's real clear_at");

        // Phase 2: the destination confirms it is still live with a partial remainder:
        // the exact open order yields the claim by the rule (2×0.5 = 1.0),
        // not remaining×0 nor the original cost.
        *router.status.lock() = Some(RouterOrderStatus::Live { remaining: dec!(2) });
        *router.open_orders.lock() = Some(vec![open_order(
            "oid-1", "cid-1", "yes-tok", OpenOrderSide::Buy, dec!(0.5), dec!(2),
        )]);
        executor.reconcile_unknowns().await;

        assert!(!state.has_unresolved_unknown("cid-1").await.unwrap());
        let key = limit_key("cid-1", "yes-tok", StrategyId::ParityMaker);
        assert_eq!(
            executor.active_gtd[&key].1,
            dec!(1.0),
            "claim re-evaluated with the destination's open order"
        );
    }

    /// Same branch but without enough evidence afterwards: if the
    /// destination's listing does not respond, the claim does NOT fall to `remaining×0`:
    /// the original durable claim is kept.
    #[tokio::test]
    async fn cancel_indeterminate_without_listing_keeps_original_cost() {
        let router = Arc::new(D1ReconRouter::new());
        let (mut executor, ledger, state, _fill_rx) =
            executor_with_d1_router(router.clone()).await;
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "yes-tok", "oid-1".into(), dec!(1.5), clear, StrategyId::ParityMaker)
            .await;
        ledger.force_add_exposure("cid-1", clear, "yes-tok", dec!(1.5), StrategyId::ParityMaker);
        let cancel = cancel_intent(StrategyId::ParityMaker);

        executor.handle_cancel(&cancel).await;
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());

        // The destination reports it live but the open-orders listing fails:
        // with no price/side evidence, the claim is kept (previously here it
        // stayed remaining × price_um(NULL→0) = 0, so exposure was released).
        *router.status.lock() = Some(RouterOrderStatus::Live { remaining: dec!(3) });
        *router.open_orders.lock() = None;
        executor.reconcile_unknowns().await;

        assert!(!state.has_unresolved_unknown("cid-1").await.unwrap());
        let key = limit_key("cid-1", "yes-tok", StrategyId::ParityMaker);
        assert_eq!(
            executor.active_gtd[&key].1,
            dec!(1.5),
            "no evidence: original claim kept, never remaining×0"
        );
    }

    /// `Ok + Gone{matched>0}` arm with an empty tape: the pending is
    /// journaled with NULL metadata and the original claim, the same as the
    /// indeterminate arm (same shared helper).
    #[tokio::test]
    async fn cancel_matched_without_tape_keeps_claim_in_pending() {
        let router = Arc::new(D1ReconRouter::new());
        let (mut executor, ledger, state, _fill_rx) =
            executor_with_d1_router(router.clone()).await;
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "yes-tok", "oid-1".into(), dec!(1.5), clear, StrategyId::ParityMaker)
            .await;
        ledger.force_add_exposure("cid-1", clear, "yes-tok", dec!(1.5), StrategyId::ParityMaker);
        *router.status.lock() = Some(RouterOrderStatus::Gone { matched: dec!(3) });
        // `trades` is already empty in D1ReconRouter::new().
        let cancel = cancel_intent(StrategyId::ParityMaker);

        executor.handle_cancel(&cancel).await;

        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        assert_eq!(ledger.total_exposure(), dec!(1.5));
        let row = state
            .list_unresolved_unknowns()
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(row.cost_um, crate::state::to_micros(dec!(1.5)));
        assert_eq!(row.price_um, None);
        assert_eq!(row.side, None);
    }

    /// A SELL of inventory (PM) with an uncertain cancel keeps its
    /// claim at 0 — no exposure is invented on a sell of inventory.
    #[tokio::test]
    async fn cancel_indeterminate_sell_inventory_stays_zero() {
        let router = Arc::new(D1ReconRouter::new());
        let (mut executor, ledger, state, _fill_rx) =
            executor_with_d1_router(router.clone()).await;
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "yes-tok", "oid-sell".into(), dec!(0), clear, StrategyId::ParityMaker)
            .await;
        ledger.force_add_exposure("cid-1", clear, "yes-tok", dec!(0), StrategyId::ParityMaker);
        let cancel = cancel_intent(StrategyId::ParityMaker);

        executor.handle_cancel(&cancel).await;
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        assert_eq!(ledger.total_exposure(), dec!(0));

        // The order is still live: SELL evidence + PM owner → claim 0.
        *router.status.lock() = Some(RouterOrderStatus::Live { remaining: dec!(2) });
        *router.open_orders.lock() = Some(vec![open_order(
            "oid-sell", "cid-1", "yes-tok", OpenOrderSide::Sell, dec!(0.7), dec!(2),
        )]);
        executor.reconcile_unknowns().await;

        let key = limit_key("cid-1", "yes-tok", StrategyId::ParityMaker);
        assert_eq!(executor.active_gtd[&key].1, dec!(0), "SELL inventory: claim 0");
    }

    /// A short TC SELL with an uncertain cancel keeps the contractual
    /// downside: the pending stores `cost`, and the live reconciliation
    /// re-evaluates it at the remainder (never cost/size, never 0).
    #[tokio::test]
    async fn cancel_indeterminate_sell_tc_keeps_downside() {
        let router = Arc::new(D1ReconRouter::new());
        let (mut executor, ledger, state, _fill_rx) =
            executor_with_d1_router(router.clone()).await;
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "yes-tok", "oid-tc".into(), dec!(4), clear, StrategyId::ThetaCapture)
            .await;
        ledger.force_add_exposure("cid-1", clear, "yes-tok", dec!(4), StrategyId::ThetaCapture);
        let cancel = cancel_intent(StrategyId::ThetaCapture);

        executor.handle_cancel(&cancel).await;
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());
        assert_eq!(ledger.total_exposure(), dec!(4), "downside kept");

        *router.status.lock() = Some(RouterOrderStatus::Live { remaining: dec!(4) });
        *router.open_orders.lock() = Some(vec![open_order(
            "oid-tc", "cid-1", "yes-tok", OpenOrderSide::Sell, dec!(0.4), dec!(4),
        )]);
        executor.reconcile_unknowns().await;

        let key = limit_key("cid-1", "yes-tok", StrategyId::ThetaCapture);
        assert_eq!(
            executor.active_gtd[&key].1,
            dec!(4),
            "short TC SELL: full downside of the remainder"
        );
    }

    /// A cancel error (the order may still be live) keeps the
    /// claim and re-registers the entry: no false pending and no release.
    #[tokio::test]
    async fn cancel_error_keeps_claim_and_order() {
        struct CancelErrRouter;
        impl OrderRouter for CancelErrRouter {
            fn submit(
                &self,
                _req: OrderRequest,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = crate::order::router::RouterResult> + Send + '_>,
            > {
                Box::pin(async { crate::order::router::RouterResult::rejected("d1-test") })
            }
            fn cancel(
                &self,
                _m: &str,
                _o: &str,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>>
            {
                Box::pin(async { Err(anyhow::anyhow!("venue indeterminate")) })
            }
            fn mode(&self) -> crate::types::Mode {
                crate::types::Mode::Paper
            }
        }
        let (_intent_tx, intent_rx) = mpsc::channel(1);
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(1000)));
        let state = StateStore::open_memory(crate::types::Mode::Paper).await.unwrap();
        let risk = RiskGuard::new(
            RiskConfig { max_total_exposure_usdc: dec!(1000), ..Default::default() },
            crate::types::Mode::Paper,
        );
        let ledger = BracketLedger::new();
        let mut executor = Executor::new(
            intent_rx,
            fill_tx,
            Arc::new(CancelErrRouter),
            risk,
            state.clone(),
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            wallet_rx,
            Arc::new(DashSet::new()),
            HashSet::new(),
            ledger.clone(),
        );
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "yes-tok", "oid-1".into(), dec!(1.5), clear, StrategyId::ParityMaker)
            .await;
        ledger.force_add_exposure("cid-1", clear, "yes-tok", dec!(1.5), StrategyId::ParityMaker);
        let cancel = cancel_intent(StrategyId::ParityMaker);

        executor.handle_cancel(&cancel).await;

        let key = limit_key("cid-1", "yes-tok", StrategyId::ParityMaker);
        assert_eq!(executor.active_gtd[&key].1, dec!(1.5), "claim kept");
        assert!(!state.has_unresolved_unknown("cid-1").await.unwrap(),
            "a cancel error does not fabricate a false pending");
    }

    /// Restart after an uncertain cancel: the durable pending re-seeds
    /// the protection after `seed + resync` (full claim).
    #[tokio::test]
    async fn cancel_indeterminate_restart_keeps_protection() {
        let router = Arc::new(D1ReconRouter::new());
        let (mut executor, ledger, state, _fill_rx) =
            executor_with_d1_router(router.clone()).await;
        let clear = chrono::Utc::now().timestamp() + 300;
        executor
            .restore_gtd("cid-1", "yes-tok", "oid-1".into(), dec!(1.5), clear, StrategyId::ParityMaker)
            .await;
        ledger.force_add_exposure("cid-1", clear, "yes-tok", dec!(1.5), StrategyId::ParityMaker);
        let cancel = cancel_intent(StrategyId::ParityMaker);
        executor.handle_cancel(&cancel).await;
        assert!(state.has_unresolved_unknown("cid-1").await.unwrap());

        // "Restart": a new executor over the SAME DB. The journal is the
        // only source of truth that survives.
        let (mut executor2, ledger2, _frx2) = {
            let (_itx, irx) = mpsc::channel(1);
            let (ftx, frx) = mpsc::unbounded_channel();
            let (_wtx, wrx) = watch::channel(Some(dec!(1000)));
            let risk = RiskGuard::new(
                RiskConfig { max_total_exposure_usdc: dec!(1000), ..Default::default() },
                crate::types::Mode::Paper,
            );
            let ledger2 = BracketLedger::new();
            let e2 = Executor::new(
                irx,
                ftx,
                router.clone(),
                risk,
                state.clone(), // same in-memory StateStore = same SQLite
                Arc::new(parking_lot::Mutex::new(HashMap::new())),
                wrx,
                Arc::new(DashSet::new()),
                HashSet::new(),
                ledger2.clone(),
            );
            (e2, ledger2, frx)
        };
        executor2.seed_unknown_attempts_from_journal_pub().await;
        // order_status is still indeterminate after the restart.
        executor2.purge_and_resync().await;

        assert!(
            state.has_unresolved_unknown("cid-1").await.unwrap(),
            "the pending survives the restart"
        );
        assert_eq!(
            ledger2.total_exposure(),
            dec!(1.5),
            "the cancel pending's claim is re-seeded after restart"
        );
    }
}
