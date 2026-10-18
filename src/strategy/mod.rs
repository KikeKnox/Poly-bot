//! `Strategy` trait + common runner for all strategies.
//!
//! Implemented strategies:
//! - `near_certain` — NC: z-score of the underlying's displacement
//! - `near_expiry`  — NE: % distance to the strike near the close

pub mod arb;
pub mod arb_15m5m;
pub mod directional;
pub mod drift_box;
pub mod follower;
pub mod g_rw1_maker;
pub mod market_making;
pub mod model_btc5m;
pub mod near_certain;
pub mod near_expiry;
pub mod open_momentum;
pub mod parity_maker;
pub mod post_close_sweep;
pub mod registry;
pub mod reward_farm;
pub mod rl_predictor;
pub mod supervisor;
pub mod theta_capture;

pub use arb::Arb;
pub use arb_15m5m::{Arb15m5m, Arb15m5mDeps};
pub use directional::Directional;
pub use drift_box::DriftBox;
pub use g_rw1_maker::GRw1Maker;
pub use market_making::MarketMaking;
pub use model_btc5m::ModelBtc5m;
pub use near_certain::NearCertain;
pub use near_expiry::NearExpiry;
pub use open_momentum::OpenMomentum;
pub use parity_maker::ParityMaker;
pub use post_close_sweep::PostCloseSweep;
pub use registry::{BuildCtx, StrategyConfigs};
pub use reward_farm::RewardFarm;
pub use rl_predictor::RlPredictor;
pub use supervisor::StrategySupervisor;
pub use theta_capture::ThetaCapture;

// LadderArb and VolSurface moved to `crate::tools` — they were pure observers
// (a `Strategy` that always returned `vec![]`) and that was not the right
// abstraction. See `crate::tools` for the `Tool` trait and the runner without RiskGuard.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering as AtomicOrdering},
        Arc,
    },
};

use tokio::sync::{broadcast, mpsc, watch};
use tracing::debug;

use crate::{
    market_feed::MarketRegistry,
    risk::RiskGuard,
    types::{BookSnapshot, Fill, OrderIntent, OrderSide, QueuedIntent, StrategyId, VolSnapshot},
};

/// Maximum bracket age to accept `vol.last_price` as a reference.
pub(crate) const REFERENCE_FALLBACK_MAX_AGE_SECS: i64 = 30;

/// Reference fallback to the oracle's current price —
/// only valid near the bracket's open. Starting mid-bracket the
/// current price is NOT the opening one: it would give a permanent dist ≈ 0 (dead
/// strategy) or a false signal. The guard requires `0 <= open_age < MAX`: it rejects
/// both the late capture (`open_age >= MAX`, mid-bracket startup) and the
/// early one (`open_age < 0`, bracket not yet open — before, it passed the check
/// anyway and cached as "strike" a price from up to ~1h before the real
/// open).
pub(crate) fn fallback_reference(
    vol: &VolSnapshot,
    book: &BookSnapshot,
    now: i64,
    cid: &str,
    tag: &str,
) -> Option<rust_decimal::Decimal> {
    let p = vol.last_price;
    if p <= rust_decimal::Decimal::ZERO {
        return None;
    }
    let open_age = now - book.market_open_ts;
    if !(0..REFERENCE_FALLBACK_MAX_AGE_SECS).contains(&open_age) {
        debug!(
            cid, open_age,
            "{tag}: no reference — outside the opening window, oracle fallback discarded"
        );
        return None;
    }
    Some(p)
}

/// Order size from the config caps (`max_order_usdc` converted to shares at the given `price`, and
/// `max_size_shares` directly), previously copied by hand in 7 strategies before this
/// helper. Both caps apply when > 0 (0 = no cap in that
/// dimension); if both apply, the minimum is taken. With no cap set, falls back
/// to `fallback` (typically the available book depth, or 0).
///
/// `shares_divisor`: ARB trades two simultaneous legs and splits
/// `max_size_shares` between both — `Some(dec!(2))` reproduces that split without
/// forcing the other strategies (call with `None`) into the same semantics.
pub(crate) fn size_from_caps(
    max_order_usdc: rust_decimal::Decimal,
    max_size_shares: rust_decimal::Decimal,
    price: rust_decimal::Decimal,
    shares_divisor: Option<rust_decimal::Decimal>,
    fallback: Option<rust_decimal::Decimal>,
) -> rust_decimal::Decimal {
    use rust_decimal::Decimal;
    let cap_usdc = (max_order_usdc > Decimal::ZERO).then(|| max_order_usdc / price);
    let max_size_shares = match shares_divisor {
        Some(d) if d > Decimal::ZERO => max_size_shares / d,
        _ => max_size_shares,
    };
    let cap_shares = (max_size_shares > Decimal::ZERO).then_some(max_size_shares);
    match (cap_usdc, cap_shares) {
        (Some(u), Some(s)) => u.min(s),
        (Some(u), None) => u,
        (None, Some(s)) => s,
        (None, None) => fallback.unwrap_or(Decimal::ZERO),
    }
}

/// Cancel `OrderIntent` — the same literal (everything at `Decimal::ZERO`/`None`
/// except what identifies what to cancel) previously copied by hand in 6 strategies before
/// this helper. A new required field in `OrderIntent` meant touching
/// all 6 sites; with this, just one.
pub(crate) fn cancel_intent(
    market_cid: &str,
    token_id: &str,
    clear_at: i64,
    neg_risk: bool,
    strategy: StrategyId,
) -> OrderIntent {
    use rust_decimal::Decimal;
    OrderIntent {
        market_cid: market_cid.to_string(),
        token_id: token_id.to_string(),
        side: OrderSide::Cancel,
        price: Decimal::ZERO,
        size: Decimal::ZERO,
        clear_at,
        signal: Decimal::ZERO,
        neg_risk,
        limit_price: None,
        second_leg: None,
        strategy,
        resting: false,
        real_clear_at: None,
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Strategy signal. Implementations in `strategy::arb`, `::near_certain`, etc.
pub trait Strategy: Send + 'static {
    /// Unique identifier (for logs, risk gating, kill switch).
    fn id(&self) -> StrategyId;

    /// Processes a book and returns zero or more order intents.
    ///
    /// - Pure with respect to external state: does not call the executor directly.
    /// - May mutate internal state (price ring buffers, cursor position, etc.).
    /// - Receives the most recent VolSnapshot of the book's asset; it may be
    ///   `sigma_short/60s = None` if the oracle has been running only briefly.
    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent>;

    /// Fill notification (for exit/requote management).
    /// Default: no-op.
    fn on_fill(&mut self, _fill: &Fill) {}

    /// Periodic watchdog tick: called even when no books arrive.
    /// It should only contain logic that does not require a live price (time
    /// deadlines, housekeeping). Default: no-op.
    fn on_timer(&mut self) -> Vec<OrderIntent> {
        vec![]
    }

    /// `true` if the strategy needs an underlying with an oracle feed
    /// to decide its signal (default). `false` for the agnostic ones
    /// (RF, ARB, PM — see `strategy::registry::strategy_requires_oracle`).
    ///
    /// NOTE: the runner (`run`, below) does NOT call this method — it uses
    /// `strategy::registry::strategy_requires_oracle(id)` (a centralized
    /// table in `strategy/registry.rs`) because `arb.rs`,
    /// `parity_maker.rs` and `reward_farm.rs`
    /// were out of scope of that change and could not override this
    /// default. It stays declared in the trait so that
    /// a new strategy can override it
    /// directly the day it gets a universe without an oracle.
    fn requires_oracle(&self) -> bool {
        true
    }
}

/// Config of a strategy, loaded from its own TOML section
/// (`config/strategies/<name_str>.toml` or `general.toml`). Implemented by
/// each `XxxConfig` — see `strategy::registry` for how it is registered.
pub trait StrategyConfig: serde::de::DeserializeOwned + Default {
    /// Shared limits (kill switch, rate limit, universe).
    fn limits(&self) -> &crate::config::StrategyLimits;
}

// ── Common runner ─────────────────────────────────────────────────────────────

/// Execution context of a strategy (what the supervisor injects).
pub struct StrategyCtx {
    pub book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    /// Vol snapshots per asset (key = slug prefix, e.g. "btc", "eth").
    pub vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    pub intent_tx: mpsc::Sender<QueuedIntent>,
    pub fill_rx: broadcast::Receiver<Fill>,
    pub risk: Arc<RiskGuard>,
    pub shutdown: watch::Receiver<bool>,
    /// Maximum age of a book to process it. Older books are skipped:
    /// during WS gaps the last snapshot stays frozen and its prices no longer exist.
    pub max_book_age: std::time::Duration,
    /// Shared registry to resolve which universe each book's
    /// `market_condition_id` belongs to and whether its asset has an oracle
    /// (`registry::UniverseInfo`).
    pub registry: Arc<MarketRegistry>,
    /// Universes where this strategy trades (`StrategyLimits.universe`).
    /// Default = current behavior: the two crypto-updown universes.
    pub universe: Vec<String>,
    /// Whether this strategy requires an oracle to decide its signal
    /// (`strategy::registry::strategy_requires_oracle(id)` — see the note on the
    /// `Strategy` trait).
    pub requires_oracle: bool,
}

/// Main loop of the runner. Spawn one task per strategy.
///
/// Exits when:
/// - `shutdown` emits `true`
/// - The `book_rx` channel is closed (no more feeds)
/// - The strategy's kill switch is activated (pause until it is reset)
pub async fn run(mut strategy: Box<dyn Strategy>, mut ctx: StrategyCtx) {
    let id = strategy.id();
    let mut kill_rx = ctx.risk.subscribe_strategy_kill(id);

    tracing::info!(%id, "strategy runner: starting");

    // Time watchdog — the loop historically only woke up with books,
    // so a quiet feed left time-based management unexecuted (ARB orphan
    // legs with no sell retry or close bail) and the shutdown
    // unattended until the next tick. `Skip`: after a burst of books there is no
    // point catching up on missed watchdog ticks.
    let mut watchdog = tokio::time::interval(std::time::Duration::from_secs(WATCHDOG_SECS));
    watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        let books = tokio::select! {
            // Intra-cid coalesce: drain the channel and keep the latest
            // book of EACH market (the global latest used to blind the runner
            // to all markets but one during bursts).
            books = drain_latest(&mut ctx.book_rx) => match books {
                Some(v) => v,
                None => {
                    tracing::warn!(%id, "strategy runner: books channel closed, exiting");
                    return;
                }
            },
            _ = watchdog.tick() => {
                drain_pending_fills(&mut *strategy, &mut ctx.fill_rx, id);
                let intents = strategy.on_timer();
                for intent in intents {
                    if ctx
                        .intent_tx
                        .send((intent, std::time::Instant::now()))
                        .await
                        .is_err()
                    {
                        tracing::warn!(%id, "strategy runner: executor closed, exiting");
                        return;
                    }
                }
                Vec::new()
            }
            _ = ctx.shutdown.changed() => {
                tracing::info!(%id, "strategy runner: shutdown received");
                return;
            }
        };

        for book in books {
            // Shutdown check
            if *ctx.shutdown.borrow() {
                tracing::info!(%id, "strategy runner: shutdown received");
                return;
            }

            // Process fills even if the book is degraded or out of universe.
            drain_pending_fills(&mut *strategy, &mut ctx.fill_rx, id);
            let entry_block = match book.recv_instant {
                None => Some("missing_receive_time"),
                Some(recv) => match crate::time::Instant::now().checked_duration_since(recv) {
                    None => Some("future_receive_time"),
                    Some(age) if age > ctx.max_book_age => Some("stale_book"),
                    _ => None,
                },
            };

            // Dispatch by universe — BEFORE any other job
            // (vol, fills, on_book), so a strategy
            // never spends cycles on a book from a universe it did not declare.
            // No registry entry for the cid (discovery/eviction race) → conservative
            // skip ("not my universe" until proven otherwise).
            let uni_info = ctx
                .registry
                .universe_info
                .get(&book.market_condition_id)
                .map(|r| r.clone());
            match uni_info {
                Some(info) => {
                    if !ctx.universe.iter().any(|u| u == &info.universe) {
                        continue; // not my universe
                    }
                    if ctx.requires_oracle && !info.has_oracle {
                        continue; // no underlying, does not quote
                    }
                }
                None => {
                    metrics::counter!("hrf_book_no_universe_info_skipped_total", "strategy" => id.to_string())
                    .increment(1);
                    continue;
                }
            }

            // Kill switch check (pause; the executor also filters, but better not to process).
            // In paper with paper_disable_kill_switch the flag controls whether the kill blocks here too.
            if *kill_rx.borrow() && !ctx.risk.paper_disable_kill_switch_active() {
                tracing::debug!(%id, "strategy runner: kill switch active, waiting for reset");
                // Wait until the kill is reset, shutdown, or a new book.
                tokio::select! {
                    _ = kill_rx.changed() => continue,
                    _ = ctx.shutdown.changed() => return,
                }
            }

            // Vol snapshot of the asset matching this book (canonical parsing,
            // same as post_close: lowercase + empty-slug rejection — before, a
            // slug "BTC-..." silently failed the lookup and one without '-' fell to the
            // "btc" fallback with no metric).
            // No cross-asset fallback — deciding a doge bracket with btc's vol
            // corrupts FV/z-score/cushion. If the asset has no feed, the book is skipped.
            // Empty map (tests / no oracle) → VolSnapshot::default() (stale, the
            // strategies that depend on the oracle already abstain on their own).
            let Some(asset) = crate::oracle::asset_from_slug(&book.slug) else {
                metrics::counter!("hrf_book_slug_without_asset_total", "strategy" => id.to_string())
                    .increment(1);
                continue;
            };
            let vol = match ctx.vol_rxs.get(asset.as_str()) {
                Some(rx) => rx.borrow().clone(),
                None if ctx.vol_rxs.is_empty() => VolSnapshot::default(),
                None => {
                    metrics::counter!("hrf_vol_asset_missing_total", "asset" => asset.clone())
                        .increment(1);
                    continue;
                }
            };

            // Process fills before on_book: guarantees ever_entered and have_position
            // are up to date before evaluating the signal. Without this, if the executor sends the fill
            // while books are piled up, on_book can run N times before seeing the fill
            // and reset entered via an exhausted entry_pending_ticks → double entry.
            drain_pending_fills(&mut *strategy, &mut ctx.fill_rx, id);

            let intents = strategy.on_book(&book, &vol);

            // hrf_signal_latency_seconds: total time from WS receipt to the end of on_book.
            if let Some(recv) = book.recv_instant {
                if let Some(elapsed) = crate::time::Instant::now().checked_duration_since(recv) {
                    metrics::histogram!("hrf_signal_latency_seconds", "strategy" => id.to_string())
                        .record(elapsed.as_secs_f64());
                }
            }

            // Send the intents (error = executor closed → exit)
            for intent in intents {
                if let Some(reason) = entry_block {
                    if intent_contains_buy(&intent) {
                        reject_degraded_entry(&mut *strategy, &intent, reason);
                        continue;
                    }
                }
                if ctx
                    .intent_tx
                    .send((intent, std::time::Instant::now()))
                    .await
                    .is_err()
                {
                    tracing::warn!(%id, "strategy runner: executor closed, exiting");
                    return;
                }
            }

            // Yield to the scheduler so fill_sink can process pending fills before
            // the next on_book. Without this, books piled up in the broadcast buffer can
            // make the loop process 20+ ticks without fill_rx receiving the confirmed fill,
            // triggering the TC bail-sell repeatedly (a cheap yield vs. 32 SELLs).
            tokio::task::yield_now().await;
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// `pub(crate)` for the replay engine: the driver reproduces the runner's
/// `entry_block` gate and needs the same predicate.
pub(crate) fn intent_contains_buy(intent: &OrderIntent) -> bool {
    matches!(intent.side, OrderSide::Buy)
        || intent
            .second_leg
            .as_deref()
            .is_some_and(intent_contains_buy)
}

/// Unique identity per local degraded rejection — each blocked intent is a
/// distinct event: without it, two identical rejections in the same ms collide
/// in the strategy's dedup and the second does not release its marked state.
static DEGRADED_REJECT_SEQ: AtomicU64 = AtomicU64::new(0);

/// LOCAL notification: unlocks the strategy's entry latches.
/// Not published to fill_sink or SQLite: no order left for the exchange.
/// `pub(crate)` for the replay engine: local rejection identical to the
/// runner's (notifies the strategy with `GateRejected`; does not persist).
pub(crate) fn reject_degraded_entry(
    strategy: &mut dyn Strategy,
    intent: &OrderIntent,
    reason: &'static str,
) {
    metrics::counter!("hrf_entry_data_blocked_total", "strategy" => intent.strategy.to_string(), "reason" => reason).increment(1);
    tracing::info!(target: "decision_audit", event = "entry_blocked", reason,
        strategy = %intent.strategy, cid = %intent.market_cid, token_id = %intent.token_id,
        signal = %intent.signal, price = %intent.price, size = %intent.size,
        "entry not sent due to data quality");
    let reject_id = format!(
        "gate-degraded-{}-{}",
        crate::time::now().timestamp_nanos_opt().unwrap_or(0),
        DEGRADED_REJECT_SEQ.fetch_add(1, AtomicOrdering::Relaxed)
    );
    strategy.on_fill(&Fill {
        market_cid: intent.market_cid.clone(),
        token_id: intent.token_id.clone(),
        // `GateRejected`, not `Rejected` — this intent never existed
        // as an order (blocked by degraded data BEFORE being sent), so
        // it travels with the intent's size/price so the strategy
        // undoes ONLY its contribution to the marked state. A `Rejected` per
        // token would set `live = 0` even if a real GTD were live on the
        // same leg (early-repost + violated-cap pattern).
        side: crate::types::FillSide::GateRejected,
        price: intent.price,
        size: intent.size,
        order_id: None,
        exec_id: Some(reject_id),
        local_order_id: None,
        // Replay clock if present — a wall-clock `Utc::now`
        // here would put this synthetic fill decades ahead of the replay
        // time and break the cooldowns/deadlines that read `fill.ts`.
        ts: crate::time::now(),
        mode: crate::types::Mode::Paper,
        neg_risk: intent.neg_risk,
        strategy: intent.strategy,
        clear_at: intent.clear_at,
        outcome_label: None,
        hedge_id: None,
    });
    if let Some(second) = intent.second_leg.as_deref() {
        reject_degraded_entry(strategy, second, reason);
    }
}

/// Cadence of the runner's watchdog — `Strategy::on_timer` ticks even when
/// the book feed is quiet. 5s << `orphan_sell_retry_secs` (10s) and <<
/// `ORPHAN_BAIL_SECS` (30s), so no ARB deadline misses its window.
const WATCHDOG_SECS: u64 = 5;

/// Drains the pending fills from the broadcast into the strategy. Shared by
/// the books arm and the watchdog arm — fills do not depend on the books feed.
fn drain_pending_fills(
    strategy: &mut dyn Strategy,
    fill_rx: &mut broadcast::Receiver<Fill>,
    id: StrategyId,
) {
    loop {
        match fill_rx.try_recv() {
            Ok(fill) => strategy.on_fill(&fill),
            // Lagged = skipped fills this strategy will no longer see.
            // They are not recoverable from the broadcast — make it visible (before it was
            // a silent `continue` that left positions out of sync).
            Err(broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(%id, skipped = n,
                    "strategy runner: fills lost to broadcast lag — possible position desync");
                metrics::counter!("hrf_strategy_fill_lag_total", "strategy" => id.to_string())
                    .increment(n);
            }
            Err(_) => break,
        }
    }
}

/// Drains the broadcast channel and returns the latest book **of each market**.
///
/// - If the channel has accumulated messages, coalesces ONLY within the same
///   `condition_id`: a market sees its most recent tick, but ticks of
///   different markets are all kept (keeping the global
///   latest silently dropped observations of other markets
///   during bursts — each strategy went blind to all but one).
/// - If empty, waits for the next one (blocking but cancellation-safe).
/// - Returns `None` only if the channel is closed and nothing is left
///   to deliver.
///
/// Delivery order: first appearance of each cid in the backlog (the
/// oldest markets first). The counter `hrf_strategy_book_coalesce_total`
/// records the ticks discarded by intra-cid coalescing.
///
/// `pub(crate)`: `model_observer::run_observer` reuses it — a
/// third copy of the drain (there is already one in `tools::run_tool`) would be another
/// place where `Lagged` handling can diverge.
pub(crate) async fn drain_latest(
    rx: &mut broadcast::Receiver<Arc<BookSnapshot>>,
) -> Option<Vec<Arc<BookSnapshot>>> {
    let mut latest: Vec<Arc<BookSnapshot>> = Vec::new();
    let mut pos: HashMap<String, usize> = HashMap::new();

    loop {
        match rx.try_recv() {
            Ok(book) => match pos.get(book.market_condition_id.as_str()) {
                // Intra-cid coalesce — only the most recent tick of
                // each market counts; those of other markets are kept.
                Some(&i) => {
                    metrics::counter!("hrf_strategy_book_coalesce_total").increment(1);
                    latest[i] = book;
                }
                None => {
                    pos.insert(book.market_condition_id.clone(), latest.len());
                    latest.push(book);
                }
            },
            Err(broadcast::error::TryRecvError::Empty) => {
                if !latest.is_empty() {
                    return Some(latest);
                }
                // Channel empty and nothing received → wait for the next one.
                // `recv().await` can also return Lagged: a burst that
                // exceeds the buffer BETWEEN the empty try_recv and the await must NOT
                // kill the runner — the cursor was already repositioned to the
                // oldest retained element; keep draining.
                match rx.recv().await {
                    Ok(b) => {
                        pos.insert(b.market_condition_id.clone(), 0);
                        latest.push(b);
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        debug!(
                            "strategy runner: lagged {n} books in the await (coalesce in effect)"
                        );
                        metrics::counter!("hrf_strategy_book_lag_total").increment(n);
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(n)) => {
                debug!("strategy runner: lagged {n} books (coalesce in effect)");
                metrics::counter!("hrf_strategy_book_lag_total").increment(n);
                // Keep draining to reach the latest
            }
            Err(broadcast::error::TryRecvError::Closed) => {
                // Deliver what was drained before closing — in-flight books
                // are still valid information.
                return if latest.is_empty() {
                    None
                } else {
                    Some(latest)
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, OrderSide, TokenBook};
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    /// Dummy strategy: emits one Buy intent for each processed book.
    struct AlwaysBuy;
    impl Strategy for AlwaysBuy {
        fn id(&self) -> StrategyId {
            StrategyId::NearCertain
        }
        fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
            vec![OrderIntent {
                market_cid: book.market_condition_id.clone(),
                token_id: book.yes.token_id.clone(),
                side: OrderSide::Buy,
                price: dec!(0.5),
                size: dec!(1),
                clear_at: book.clear_at,
                signal: dec!(0.5),
                neg_risk: false,
                limit_price: Some(dec!(0.5)),
                second_leg: None,
                strategy: StrategyId::NearCertain,
                resting: false,
                real_clear_at: None,
            }]
        }
    }

    /// Dummy strategy that emits an intent on every watchdog tick —
    /// the books channel stays quiet in the test.
    struct TimerSell;
    impl Strategy for TimerSell {
        fn id(&self) -> StrategyId {
            StrategyId::Arb
        }
        fn on_book(&mut self, _book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
            vec![]
        }
        fn on_timer(&mut self) -> Vec<OrderIntent> {
            vec![OrderIntent {
                market_cid: "cid-stale".into(),
                token_id: "tok-yes".into(),
                side: OrderSide::Sell {
                    size: Some(dec!(1)),
                },
                price: dec!(0.4),
                size: dec!(1),
                clear_at: 9_999_999_999,
                signal: Decimal::ZERO,
                neg_risk: false,
                limit_price: None,
                second_leg: None,
                strategy: StrategyId::Arb,
                resting: false,
                real_clear_at: None,
            }]
        }
    }

    /// Registry with a single `universe_info` entry for "cid-stale"
    /// (the cid `make_book` uses). The dispatch tests override it or
    /// use it as is to simulate the "pass" scenario (declared universe +
    /// oracle present — same as today's updown).
    fn registry_with_universe(universe: &str, has_oracle: bool) -> Arc<MarketRegistry> {
        let registry = Arc::new(MarketRegistry::new());
        registry.universe_info.insert(
            "cid-stale".to_string(),
            crate::market_feed::registry::UniverseInfo {
                universe: universe.to_string(),
                has_oracle,
            },
        );
        registry
    }

    fn make_book(age: std::time::Duration) -> Arc<BookSnapshot> {
        let half = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        Arc::new(BookSnapshot {
            market_condition_id: "cid-stale".into(),
            yes: TokenBook {
                token_id: "tok-yes".into(),
                bids: half("0.49"),
                asks: half("0.51"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-no".into(),
                bids: half("0.49"),
                asks: half("0.51"),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: 9_999_999_999,
            neg_risk: false,
            slug: "btc-updown-test".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: crate::time::Instant::now().checked_sub(age),
        })
    }

    /// A burst that overflows the buffer BETWEEN the `try_recv(Empty)` and the
    /// `recv().await` returns `RecvError::Lagged` — recoverable, not a close:
    /// the runner must keep draining and return the latest retained book.
    /// (Before, `Err(_) => return None` killed the runner permanently.)
    #[tokio::test]
    async fn drain_latest_survives_lagged_in_await() {
        let (tx, mut rx) = broadcast::channel::<Arc<BookSnapshot>>(4);
        let drain = tokio::spawn(async move { drain_latest(&mut rx).await });
        // Let the drain reach the `recv().await` with the channel empty.
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        // 10 sends of the same cid with buffer 4 → Lagged(6) in the await;
        // intra-cid coalesce → a single book delivered.
        for _ in 0..10 {
            tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        }
        let got = tokio::time::timeout(std::time::Duration::from_secs(5), drain)
            .await
            .expect("drain_latest hung")
            .expect("task panic");
        assert!(
            got.as_ref().is_some_and(|v| !v.is_empty()),
            "drain_latest returned None/empty after Lagged — runner dead"
        );
    }

    /// `RecvError::Closed` IS terminal: the runner must terminate.
    #[tokio::test]
    async fn drain_latest_returns_none_only_if_channel_closes() {
        let (tx, mut rx) = broadcast::channel::<Arc<BookSnapshot>>(4);
        drop(tx);
        assert!(drain_latest(&mut rx).await.is_none());
    }

    /// Coalescing is intra-cid. A backlog with ticks from
    /// several markets must deliver the latest book of EACH one — keeping
    /// the global latest blinded the strategy to all markets
    /// but one during bursts (the worst time to lose ticks).
    #[tokio::test]
    async fn drain_latest_keeps_one_book_per_market() {
        let (tx, mut rx) = broadcast::channel::<Arc<BookSnapshot>>(16);
        let with_cid = |cid: &str| {
            let mut b = (*make_book(std::time::Duration::ZERO)).clone();
            b.market_condition_id = cid.into();
            Arc::new(b)
        };
        // Backlog: cid-A (old), cid-B, cid-stale, cid-A (new) →
        // delivers [cid-A(new), cid-B, cid-stale] in order of first
        // appearance.
        tx.send(with_cid("cid-a")).unwrap();
        tx.send(with_cid("cid-b")).unwrap();
        tx.send(make_book(std::time::Duration::ZERO)).unwrap(); // "cid-stale"
        tx.send(with_cid("cid-a")).unwrap(); // replaces the first cid-a
        let got = drain_latest(&mut rx)
            .await
            .expect("open channel with backlog");
        let cids: Vec<&str> = got.iter().map(|b| b.market_condition_id.as_str()).collect();
        assert_eq!(cids, vec!["cid-a", "cid-b", "cid-stale"]);
    }

    /// A degraded entry must not prevent Sell/Cancel nor leave the Buy latch set.
    #[tokio::test]
    async fn degraded_preserves_outputs_and_recovers_with_fresh_book() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Mixed(Arc<AtomicUsize>);
        impl Strategy for Mixed {
            fn id(&self) -> StrategyId {
                StrategyId::NearCertain
            }
            fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
                let buy = AlwaysBuy.on_book(book, vol).remove(0);
                let mut sell = buy.clone();
                sell.side = OrderSide::Sell {
                    size: Some(dec!(1)),
                };
                let mut cancel = buy.clone();
                cancel.side = OrderSide::Cancel;
                vec![buy, sell, cancel]
            }
            fn on_fill(&mut self, fill: &Fill) {
                // The degraded entry is acknowledged as `GateRejected`
                // (the intent never existed as an order), not `Rejected`.
                if matches!(fill.side, crate::types::FillSide::GateRejected) {
                    self.0.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let rejects = Arc::new(AtomicUsize::new(0));
        let (book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (shutdown_tx, shutdown) = watch::channel(false);
        let ctx = StrategyCtx {
            book_rx,
            intent_tx,
            fill_rx: fill_tx.subscribe(),
            shutdown,
            vol_rxs: HashMap::new(),
            risk: RiskGuard::new(
                crate::config::RiskConfig::default(),
                crate::types::Mode::Paper,
            ),
            max_book_age: std::time::Duration::from_secs(5),
            registry: registry_with_universe("crypto-updown-short", true),
            universe: vec!["crypto-updown-short".into()],
            requires_oracle: false,
        };
        let task = tokio::spawn(run(Box::new(Mixed(rejects.clone())), ctx));
        book_tx
            .send(make_book(std::time::Duration::from_secs(10)))
            .unwrap();
        let sell = tokio::time::timeout(std::time::Duration::from_secs(1), intent_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(sell.0.side, OrderSide::Sell { .. }));
        assert!(matches!(
            intent_rx.recv().await.unwrap().0.side,
            OrderSide::Cancel
        ));
        assert_eq!(rejects.load(Ordering::Relaxed), 1);
        assert!(intent_rx.try_recv().is_err());
        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        let buy = tokio::time::timeout(std::time::Duration::from_secs(1), intent_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(buy.0.side, OrderSide::Buy));
        shutdown_tx.send(true).unwrap();
        task.await.unwrap();
    }

    /// A book whose asset has no vol feed must NOT be processed with another
    /// asset's vol (before it fell back to the first VolSnapshot in the mapping).
    #[tokio::test]
    async fn book_without_vol_feed_is_skipped() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let risk = crate::risk::RiskGuard::new(
            crate::config::RiskConfig::default(),
            crate::types::Mode::Paper,
        );
        // Non-empty map but without the book's asset ("btc"): only eth has a feed.
        let (_vol_tx, vol_rx) = watch::channel(VolSnapshot::default());
        let mut vol_rxs = HashMap::new();
        vol_rxs.insert("eth".to_string(), vol_rx);
        let ctx = StrategyCtx {
            book_rx,
            vol_rxs,
            intent_tx,
            fill_rx: fill_tx.subscribe(),
            risk,
            shutdown: kill_rx,
            max_book_age: std::time::Duration::from_secs(5),
            registry: registry_with_universe("crypto-updown-short", true),
            universe: vec!["crypto-updown-short".into(), "crypto-updown-long".into()],
            requires_oracle: true,
        };
        tokio::spawn(run(Box::new(AlwaysBuy), ctx));

        // btc book (slug "btc-updown-test") without a btc feed → skip, no intent.
        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        let res =
            tokio::time::timeout(std::time::Duration::from_millis(150), intent_rx.recv()).await;
        assert!(
            res.is_err(),
            "a book without a vol feed for its asset must not produce an intent"
        );
    }

    /// The runner's watchdog wakes up the strategy even if no
    /// book arrives — the interval's first tick is immediate.
    #[tokio::test]
    async fn watchdog_wakes_strategy_without_books() {
        let (_book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let risk = crate::risk::RiskGuard::new(
            crate::config::RiskConfig::default(),
            crate::types::Mode::Paper,
        );
        let ctx = StrategyCtx {
            book_rx,
            vol_rxs: HashMap::new(),
            intent_tx,
            fill_rx: fill_tx.subscribe(),
            risk,
            shutdown: kill_rx,
            max_book_age: std::time::Duration::from_secs(5),
            registry: Arc::new(MarketRegistry::new()),
            universe: vec![],
            requires_oracle: false,
        };
        tokio::spawn(run(Box::new(TimerSell), ctx));

        let intent = tokio::time::timeout(std::time::Duration::from_millis(500), intent_rx.recv())
            .await
            .expect("on_timer must dispatch intents without books")
            .unwrap();
        assert!(matches!(intent.0.side, OrderSide::Sell { .. }));
        assert_eq!(intent.0.strategy, StrategyId::Arb);
    }

    #[tokio::test]
    async fn guard_staleness_skips_old_books() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let risk = crate::risk::RiskGuard::new(
            crate::config::RiskConfig::default(),
            crate::types::Mode::Paper,
        );
        let ctx = StrategyCtx {
            book_rx,
            vol_rxs: HashMap::new(),
            intent_tx,
            fill_rx: fill_tx.subscribe(),
            risk,
            shutdown: kill_rx,
            max_book_age: std::time::Duration::from_millis(500),
            registry: registry_with_universe("crypto-updown-short", true),
            universe: vec!["crypto-updown-short".into(), "crypto-updown-long".into()],
            requires_oracle: true,
        };
        tokio::spawn(run(Box::new(AlwaysBuy), ctx));

        // Stale book (10s) → the guard skips it, no intent.
        book_tx
            .send(make_book(std::time::Duration::from_secs(10)))
            .unwrap();
        let res =
            tokio::time::timeout(std::time::Duration::from_millis(100), intent_rx.recv()).await;
        assert!(res.is_err(), "a stale book must not produce an intent");

        // Fresh book → intent.
        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        let intent = tokio::time::timeout(std::time::Duration::from_millis(200), intent_rx.recv())
            .await
            .expect("a fresh book must produce an intent")
            .unwrap();
        assert_eq!(intent.0.market_cid, "cid-stale");
    }

    // ── Dispatch by universe ────────────────────────────────────────────────

    /// Common setup of the dispatch tests: channels + risk + base ctx (without
    /// registry/universe/requires_oracle, which each test customizes).
    fn basic_ctx(
        book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
        fill_rx: broadcast::Receiver<Fill>,
        intent_tx: mpsc::Sender<QueuedIntent>,
        kill_rx: watch::Receiver<bool>,
        registry: Arc<MarketRegistry>,
        universe: Vec<String>,
        requires_oracle: bool,
    ) -> StrategyCtx {
        let risk = crate::risk::RiskGuard::new(
            crate::config::RiskConfig::default(),
            crate::types::Mode::Paper,
        );
        StrategyCtx {
            book_rx,
            vol_rxs: HashMap::new(),
            intent_tx,
            fill_rx,
            risk,
            shutdown: kill_rx,
            max_book_age: std::time::Duration::from_secs(5),
            registry,
            universe,
            requires_oracle,
        }
    }

    /// The current "pass" scenario: declared crypto-updown-short universe, with
    /// oracle — a book from such a market MUST reach `on_book`
    /// (sanity check that the dispatch does not break today's behavior).
    #[tokio::test]
    async fn book_of_declared_universe_with_oracle_passes() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let ctx = basic_ctx(
            book_rx,
            fill_tx.subscribe(),
            intent_tx,
            kill_rx,
            registry_with_universe("crypto-updown-short", true),
            vec!["crypto-updown-short".into(), "crypto-updown-long".into()],
            true,
        );
        tokio::spawn(run(Box::new(AlwaysBuy), ctx));

        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        let intent = tokio::time::timeout(std::time::Duration::from_millis(200), intent_rx.recv())
            .await
            .expect("declared universe + oracle present must produce an intent")
            .unwrap();
        assert_eq!(intent.0.market_cid, "cid-stale");
    }

    /// Central invariant: a strategy with the default universe
    /// (crypto-updown-*) must NEVER receive a book from a market registered
    /// in "rewards-longdated" — 0 orders in new universes even if the
    /// strategy does not change its config.
    #[tokio::test]
    async fn book_of_undeclared_universe_is_skipped() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let ctx = basic_ctx(
            book_rx,
            fill_tx.subscribe(),
            intent_tx,
            kill_rx,
            registry_with_universe("rewards-longdated", false),
            vec!["crypto-updown-short".into(), "crypto-updown-long".into()], // default — does not include rewards-longdated
            true,
        );
        tokio::spawn(run(Box::new(AlwaysBuy), ctx));

        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        let res =
            tokio::time::timeout(std::time::Duration::from_millis(150), intent_rx.recv()).await;
        assert!(
            res.is_err(),
            "a book from an undeclared universe must not produce an intent"
        );
    }

    /// Oracle-driven strategy (`requires_oracle=true`) + market without oracle
    /// (`has_oracle=false`) even though the universe IS declared → skip.
    #[tokio::test]
    async fn book_without_oracle_is_skipped_if_strategy_requires_it() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let ctx = basic_ctx(
            book_rx,
            fill_tx.subscribe(),
            intent_tx,
            kill_rx,
            registry_with_universe("crypto-updown-short", false), // declared, but has_oracle=false
            vec!["crypto-updown-short".into(), "crypto-updown-long".into()],
            true, // requires_oracle
        );
        tokio::spawn(run(Box::new(AlwaysBuy), ctx));

        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        let res =
            tokio::time::timeout(std::time::Duration::from_millis(150), intent_rx.recv()).await;
        assert!(
            res.is_err(),
            "an oracle-driven strategy without an oracle on the market must not produce an intent"
        );
    }

    /// Counterpart of the above: an agnostic strategy (`requires_oracle=false`)
    /// MUST process a book without an oracle, as long as the universe is declared.
    #[tokio::test]
    async fn book_without_oracle_passes_if_strategy_is_agnostic() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let ctx = basic_ctx(
            book_rx,
            fill_tx.subscribe(),
            intent_tx,
            kill_rx,
            registry_with_universe("crypto-updown-short", false),
            vec!["crypto-updown-short".into(), "crypto-updown-long".into()],
            false, // requires_oracle = false (agnostic, e.g. ARB/PM/RF)
        );
        tokio::spawn(run(Box::new(AlwaysBuy), ctx));

        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        let intent = tokio::time::timeout(std::time::Duration::from_millis(200), intent_rx.recv())
            .await
            .expect("an agnostic strategy must process the book even without an oracle")
            .unwrap();
        assert_eq!(intent.0.market_cid, "cid-stale");
    }

    /// Discovery/eviction race: no entry in `universe_info` for the cid
    /// → conservative skip, never assume "it is my universe".
    #[tokio::test]
    async fn book_without_registry_entry_is_skipped() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (intent_tx, mut intent_rx) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let ctx = basic_ctx(
            book_rx,
            fill_tx.subscribe(),
            intent_tx,
            kill_rx,
            Arc::new(MarketRegistry::new()), // empty — no entry for "cid-stale"
            vec!["crypto-updown-short".into(), "crypto-updown-long".into()],
            true,
        );
        tokio::spawn(run(Box::new(AlwaysBuy), ctx));

        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        let res =
            tokio::time::timeout(std::time::Duration::from_millis(150), intent_rx.recv()).await;
        assert!(
            res.is_err(),
            "a registry without an entry must be skipped conservatively"
        );
    }

    /// `fallback_reference` must reject both the late capture
    /// (mid-bracket startup, `open_age >= MAX`) and the early one
    /// (bracket not yet open, `open_age < 0`) — it only accepts the window
    /// `0 <= open_age < MAX` around the real open.
    #[test]
    fn fallback_reference_rejects_negative_and_late_open_age() {
        let vol = VolSnapshot {
            last_price: dec!(100_000),
            ..Default::default()
        };
        let book = BookSnapshot {
            market_open_ts: 1_000,
            ..Default::default()
        };

        // Bracket not yet open (now < market_open_ts): open_age = -5.
        assert_eq!(fallback_reference(&vol, &book, 995, "cid", "T"), None);
        // Right at the open: open_age = 0, within the window.
        assert_eq!(
            fallback_reference(&vol, &book, 1_000, "cid", "T"),
            Some(dec!(100_000))
        );
        // Inside the window: open_age = 29.
        assert_eq!(
            fallback_reference(&vol, &book, 1_029, "cid", "T"),
            Some(dec!(100_000))
        );
        // Mid-bracket startup: open_age = 30 (== MAX, out of range).
        assert_eq!(fallback_reference(&vol, &book, 1_030, "cid", "T"), None);
    }

    // ── size_from_caps ──

    /// Reproduces, for the common pattern of the 6 strategies without a divisor
    /// (`(cap_usdc > 0).then(...) / (cap_shares > 0).then_some(...) / fallback`),
    /// the same 4 cases (both caps, only usdc, only shares, neither) that
    /// each inline formula had before this helper.
    #[test]
    fn size_from_caps_matches_inline_formula_without_divisor() {
        let price = dec!(0.40);
        let fallback = Some(dec!(7));

        // Both caps set: min(u, s).
        assert_eq!(
            size_from_caps(dec!(20), dec!(30), price, None, fallback),
            dec!(30)
        );
        // Only usdc.
        assert_eq!(
            size_from_caps(dec!(20), Decimal::ZERO, price, None, fallback),
            dec!(50)
        );
        // Only shares.
        assert_eq!(
            size_from_caps(Decimal::ZERO, dec!(12), price, None, fallback),
            dec!(12)
        );
        // Neither: falls back to the fallback (book depth or 0).
        assert_eq!(
            size_from_caps(Decimal::ZERO, Decimal::ZERO, price, None, fallback),
            dec!(7)
        );
        assert_eq!(
            size_from_caps(Decimal::ZERO, Decimal::ZERO, price, None, None),
            Decimal::ZERO
        );
    }

    /// arb.rs: `max_size_shares` is split between the 2 legs (`shares_divisor
    /// = Some(2)`) and the depth is a hard cap in ALL branches — reproduced
    /// by calling `.min(depth)` over the helper's result (fallback =
    /// depth too). Checks against arb.rs's original inline formula:
    /// `match (cap_usdc, cap_shares) { ... => u.min(s).min(depth), ... }`.
    #[test]
    fn size_from_caps_reproduces_arb_variant_with_divisor_and_depth() {
        let cost_per_share = dec!(0.90);
        let depth = dec!(15);

        // Both caps, but depth is the most restrictive.
        let cap_usdc = dec!(50) / cost_per_share; // ≈55.55
        let cap_shares = dec!(40) / dec!(2); // 20 (half for each of the 2 legs)
        let inline = cap_usdc.min(cap_shares).min(depth);
        let helper = size_from_caps(
            dec!(50),
            dec!(40),
            cost_per_share,
            Some(dec!(2)),
            Some(depth),
        )
        .min(depth);
        assert_eq!(helper, inline);
        assert_eq!(helper, depth);

        // Neither cap: falls back to depth on both.
        let inline_none = depth;
        let helper_none = size_from_caps(
            Decimal::ZERO,
            Decimal::ZERO,
            cost_per_share,
            Some(dec!(2)),
            Some(depth),
        )
        .min(depth);
        assert_eq!(helper_none, inline_none);

        // Only cap_usdc, smaller than depth.
        let cap_usdc_2 = dec!(9) / cost_per_share; // = 10
        let inline_usdc = cap_usdc_2.min(depth);
        let helper_usdc = size_from_caps(
            dec!(9),
            Decimal::ZERO,
            cost_per_share,
            Some(dec!(2)),
            Some(depth),
        )
        .min(depth);
        assert_eq!(helper_usdc, inline_usdc);
    }

    #[derive(Debug, Clone, serde::Deserialize, Default)]
    struct DummyConfig {
        #[serde(flatten)]
        limits: crate::config::StrategyLimits,
    }
    impl StrategyConfig for DummyConfig {
        fn limits(&self) -> &crate::config::StrategyLimits {
            &self.limits
        }
    }

    #[test]
    fn strategy_config_trait_exposes_limits() {
        let cfg = DummyConfig::default();
        assert!(!cfg.limits().enabled);
    }
}
