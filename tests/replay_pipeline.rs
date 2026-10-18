//! Replay tests driven by the REAL PIPELINE: the recorded tape runs through
//! Executor → RiskGuard → SimulatedRouter → fill_sink → in-memory SQLite,
//! over a queue of events on a virtual clock (no wall-clock sleeps).
//!
//! Verified properties:
//! 1. An observation the agent receives AFTER making a decision cannot
//!    influence it (arrival at the exchange ≠ observation by the agent).
//! 2. Same tape + same seed → same economic ledger, byte for byte.
//! 3. A risk limit rejects IDENTICALLY in realtime paper and in replay.
//! 4. Caps, Unknown, TTL (GTD expiry), partial fills, the cancel/fill race and
//!    close+restore (restart) are exercised through the production path,
//!    not through a parallel ledger.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{broadcast, mpsc, watch};

use poly_bot::{
    accounting::fill_sink,
    market_feed::MarketRegistry,
    order::{
        executor::Executor,
        sim_router::{SimClock, SimulatedRouter},
    },
    replay::{ReplayEngine, ReplayParams, Tape, TapeBook},
    risk::RiskGuard,
    state::StateStore,
    strategy::Strategy,
    types::{
        BookSnapshot, Fill, HalfBook, Level, Mode, OrderIntent, OrderSide, StrategyId, TokenBook,
        VolSnapshot,
    },
};

// ── Tape fixture ─────────────────────────────────────────────────────────

/// The replay clock (`crate::time::REPLAY_NOW_MS`) is GLOBAL to the process:
/// two concurrent runs would clobber each other (one test's `ReplayClockGuard::drop`
/// would disarm the clock midway through another run, and `now()` would fall back to wall time).
/// The tests in this file are serialized with this mutex; it is the only
/// coordination they need, since within each run the engine is deterministic.
static REPLAY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const T0: i64 = 1_700_000_000_000; // ms epoch — fixed tape, deterministic
const CLEAR_AT: i64 = (T0 / 1000) + 7_200; // bracket open for the whole replay

fn token_book(token_id: &str, bid: &str, ask: &str, size: &str) -> TokenBook {
    TokenBook {
        token_id: token_id.into(),
        bids: HalfBook {
            levels: vec![Level {
                price: bid.parse().unwrap(),
                size: size.parse().unwrap(),
            }],
        },
        asks: HalfBook {
            levels: vec![Level {
                price: ask.parse().unwrap(),
                size: size.parse().unwrap(),
            }],
        },
        ..Default::default()
    }
}

fn book(cid: &str, slug: &str, yes_tok: &str, no_tok: &str, ask: &str) -> BookSnapshot {
    BookSnapshot {
        market_condition_id: cid.into(),
        slug: slug.into(),
        clear_at: CLEAR_AT,
        market_open_ts: T0 / 1000 - 3_600,
        neg_risk: false,
        reference_price: Some(dec!(50_000)),
        fee_rate_bps: None,
        parabolic_fee_rate: None,
        tick_size: None,
        yes: token_book(yes_tok, "0.49", ask, "1000"),
        no: token_book(no_tok, "0.49", ask, "1000"),
        recv_instant: None,
    }
}

fn tape_book(recv_ms: i64, snap: BookSnapshot) -> TapeBook {
    TapeBook {
        recv_ms,
        avail_ms: None,
        tape_seq: None,
        snap,
    }
}

fn default_params() -> ReplayParams {
    ReplayParams {
        seed: 0xC0FF_EE15,
        initial_balance: dec!(1000),
        risk: poly_bot::config::RiskConfig::default(),
        obs_delay_ms: 0,
        latency_ms: 0,
        maintenance_ms: 30_000,
        timer_ms: 5_000,
        fak_strategies: HashSet::new(),
        // Large enough not to filter out the fixture books (age
        // = obs_delay + queue time; tape timestamps are in small ms).
        max_book_age_ms: 300_000,
        research: None,
        settlement_proxy_diagnostic: false,
    }
}

async fn new_engine(params: ReplayParams) -> (ReplayEngine, Arc<StateStore>) {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let engine = ReplayEngine::new(params, state.clone())
        .await
        .expect("engine");
    (engine, state)
}

// ── Test strategies ───────────────────────────────────────────────────────

/// Emits ONE Limit FOK buy at the ask price of the first book it sees; records
/// how many books it had observed when it emitted (for the causality test).
struct BuyOnce {
    emitted: bool,
    seen_books: usize,
    /// Books observed when the intent was emitted — the test's assertion.
    seen_on_emit: Arc<Mutex<usize>>,
    token: String,
    size: Decimal,
    resting: bool,
}

impl BuyOnce {
    fn new(token: &str, size: Decimal) -> Self {
        Self {
            emitted: false,
            seen_books: 0,
            seen_on_emit: Arc::new(Mutex::new(0)),
            token: token.into(),
            size,
            resting: false,
        }
    }
}

impl Strategy for BuyOnce {
    fn id(&self) -> StrategyId {
        StrategyId::Arb
    }
    fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
        self.seen_books += 1;
        if self.emitted {
            return vec![];
        }
        self.emitted = true;
        *self.seen_on_emit.lock() = self.seen_books;
        let ask = book
            .yes
            .asks
            .levels
            .first()
            .map(|l| l.price)
            .unwrap_or(dec!(0.50));
        vec![OrderIntent {
            market_cid: book.market_condition_id.clone(),
            token_id: self.token.clone(),
            side: OrderSide::Buy,
            price: ask,
            size: self.size,
            clear_at: book.clear_at,
            signal: ask,
            neg_risk: false,
            limit_price: Some(ask),
            second_leg: None,
            strategy: StrategyId::Arb,
            resting: self.resting,
            real_clear_at: None,
        }]
    }
}

/// Buys N times (once per observed book), same leg — for caps.
struct BuyAlways {
    token: String,
    size: Decimal,
    price: Decimal,
    cid: String,
    max: usize,
    n: usize,
}

impl BuyAlways {
    fn new(cid: &str, token: &str, price: Decimal, size: Decimal, max: usize) -> Self {
        Self {
            token: token.into(),
            size,
            price,
            cid: cid.into(),
            max,
            n: 0,
        }
    }
}

impl Strategy for BuyAlways {
    fn id(&self) -> StrategyId {
        StrategyId::Arb
    }
    fn on_book(&mut self, _book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
        if self.n >= self.max {
            return vec![];
        }
        self.n += 1;
        vec![OrderIntent {
            market_cid: self.cid.clone(),
            token_id: self.token.clone(),
            side: OrderSide::Buy,
            price: self.price,
            size: self.size,
            clear_at: CLEAR_AT,
            signal: self.price,
            neg_risk: false,
            limit_price: Some(self.price),
            second_leg: None,
            strategy: StrategyId::Arb,
            resting: false,
            real_clear_at: None,
        }]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Late observation cannot influence a decision already made
// ═══════════════════════════════════════════════════════════════════════════

/// The exchange receives book1 (ask 0.50) at t=0 and book2 (ask 0.99, not fillable
/// at book1's limit) at t=5ms. The agent observes with a 50ms delay: it
/// decides on book1 at t=50 — BEFORE seeing book2 (t=55). If the engine
/// delivered book2 to the strategy "too early" (or the decision
/// used book2), the intent would carry limit 0.99 and fill. With the correct
/// semantics: the decision uses book1 (limit 0.50) and the FOK is rejected against the
/// exchange's live book on arrival (ask 0.99 > 0.50).
#[tokio::test(flavor = "current_thread")]
async fn late_observation_does_not_influence_already_made_decision() {
    let _serial = REPLAY_LOCK.lock().await;
    let mut params = default_params();
    params.obs_delay_ms = 50;
    params.latency_ms = 60; // the order reaches the venue at t=110, with book2 already live
    let (mut engine, state) = new_engine(params).await;

    let cid = "cid-obs";
    let mut b1 = book(cid, "btc-updown-x", "tok-y1", "tok-n1", "0.50");
    b1.yes.asks.levels[0].price = dec!(0.50);
    let mut b2 = book(cid, "btc-updown-x", "tok-y1", "tok-n1", "0.99");
    b2.yes.asks.levels[0].price = dec!(0.99);

    let seen = Arc::new(Mutex::new(0usize));
    let mut strat = BuyOnce::new("tok-y1", dec!(10));
    strat.seen_on_emit = seen.clone();

    let tape = Tape {
        books: vec![tape_book(T0, b1), tape_book(T0 + 5, b2)],
        spots: vec![],
        ..Default::default()
    };
    let out = engine.run(tape, &mut strat).await.unwrap();

    assert_eq!(
        *seen.lock(),
        1,
        "the intent was decided having seen ONLY book1"
    );
    assert_eq!(
        out.market_fills, 0,
        "the FOK at book1's limit (0.50) cannot fill against ask 0.99"
    );
    assert!(
        out.rejected_router >= 1,
        "the router must reject the FOK outside price"
    );
    assert!(state.get_position("tok-y1", "arb").await.unwrap().is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// Same tape + same seed → same economic ledger, byte for byte
// ═══════════════════════════════════════════════════════════════════════════

fn deterministic_tape() -> Tape {
    let cid = "cid-det";
    let mut books = Vec::new();
    // 6 books: the ask oscillates so fills and rejections alternate.
    for (i, ask) in ["0.50", "0.52", "0.48", "0.55", "0.50", "0.51"]
        .iter()
        .enumerate()
    {
        books.push(tape_book(T0 + i as i64 * 1_000, {
            let mut b = book(cid, "btc-updown-det", "tok-dy", "tok-dn", ask);
            b.yes.asks.levels[0].price = ask.parse().unwrap();
            b
        }));
    }
    Tape {
        books,
        spots: vec![],
        ..Default::default()
    }
}

#[tokio::test(flavor = "current_thread")]
async fn same_tape_and_seed_same_ledger_byte_for_byte() {
    let _serial = REPLAY_LOCK.lock().await;
    let mut params = default_params();
    params.latency_ms = 7; // exercises deferred arrivals at the venue
    params.obs_delay_ms = 3;

    let run_once = |params: ReplayParams| async move {
        let (mut engine, _state) = new_engine(params).await;
        let mut strat = BuyAlways::new("cid-det", "tok-dy", dec!(0.55), dec!(5), 6);
        engine
            .run(deterministic_tape(), &mut strat)
            .await
            .unwrap()
            .digest
    };

    let d1 = run_once(default_params()).await;
    let d2 = run_once({
        let mut p = default_params();
        p.latency_ms = 7;
        p.obs_delay_ms = 3;
        p
    })
    .await;
    let d3 = run_once({
        let mut p = default_params();
        p.latency_ms = 7;
        p.obs_delay_ms = 3;
        p
    })
    .await;
    // d1 has different latencies: equality with d2/d3 is NOT required (different
    // effective tape); d2 and d3 share EVERYTHING and must be identical.
    assert_eq!(d2, d3, "same tape + seed + latencies ⇒ same ledger");
    // d1 also runs twice, to cover latency=0.
    let d1b = run_once(default_params()).await;
    assert_eq!(d1, d1b);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bracket cap rejects identically in realtime paper and in replay
// ═══════════════════════════════════════════════════════════════════════════

/// Realtime pipeline (Executor::run + accounting::fill_sink::run + sim with real sleeps)
/// fed with the SAME event sequence as the replay. Returns the
/// number of open positions and the ledger net — the comparison is
/// economic (order_ids differ: the sim's boot_ms is wall time).
async fn fixture_realtime_two_buys() -> (Decimal, Decimal) {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let (fill_tx, _bcast_rx) = broadcast::channel::<Fill>(64);
    let (fill_persist_tx, fill_persist_rx) = mpsc::unbounded_channel::<Fill>();
    let risk_cfg = poly_bot::config::RiskConfig {
        max_bracket_usdc: dec!(8),
        ..Default::default()
    };
    let risk = RiskGuard::new(risk_cfg.clone(), Mode::Paper);
    let registry = Arc::new(MarketRegistry::new());
    let (wallet_tx, wallet_rx) = watch::channel(None);
    // SAME sim params as the replay leg (risk_cfg = RiskConfig::default
    // + cap — same origin as `default_params().risk`): slippage, fok_rate,
    // fee and correlation identical, and the SAME seed so the FOK draw
    // sequence is the same in both legs — with different seeds another
    // draw could flip the differential result. The clock IS wall time
    // (it is the realtime fixture); order_ids differ by `boot_ms`,
    // so the comparison is economic.
    let sim_clock: SimClock = Arc::new(|| chrono::Utc::now().timestamp_millis());
    let router = SimulatedRouter::with_deps(
        risk_cfg.paper_slippage_bps,
        risk_cfg.paper_fok_fill_rate,
        2, // real latency 2ms — same as the replay's virtual latency
        risk_cfg.paper_taker_fee_rate,
        risk_cfg.paper_pair_fill_correlation,
        fill_persist_tx.clone(),
        registry.clone(),
        dec!(1000),
        wallet_tx,
        0xC0FF_EE15, // same seed as default_params().seed
        sim_clock,
    );
    let (intent_tx, intent_rx) = mpsc::channel(16);
    let order_map = Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    let ledger = poly_bot::coordination::BracketLedger::new();
    let executor = Executor::new(
        intent_rx,
        fill_persist_tx,
        router.clone(),
        risk.clone(),
        state.clone(),
        order_map.clone(),
        wallet_rx,
        Arc::new(dashmap::DashSet::new()),
        HashSet::new(),
        ledger.clone(),
    );
    let risk2 = RiskGuard::new(risk_cfg, Mode::Paper);
    let (_, kill_rx) = watch::channel(false);
    tokio::spawn(fill_sink::run(
        fill_persist_rx,
        state.clone(),
        risk2,
        ledger.clone(),
        fill_tx,
        order_map,
        kill_rx,
        tokio::sync::oneshot::channel().0,
    ));
    tokio::spawn(executor.run());

    // Same tape event: book with ask 0.75, recorded as in the replay.
    let snap = book("cid-cap", "btc-updown-cap", "tok-cy", "tok-cn", "0.75");
    registry.remember_clear_at("cid-cap", false, CLEAR_AT, "tok-cy", "tok-cn");
    router.tick_book(&snap);
    router.tick_book(&snap); // second tick: same book (second event of the tape)

    let mk_intent = || OrderIntent {
        market_cid: "cid-cap".into(),
        token_id: "tok-cy".into(),
        side: OrderSide::Buy,
        price: dec!(0.75),
        size: dec!(10), // 7.5 USDC; twice = 15 > cap 8
        clear_at: CLEAR_AT,
        signal: dec!(0.75),
        neg_risk: false,
        limit_price: Some(dec!(0.75)),
        second_leg: None,
        strategy: StrategyId::Arb,
        resting: false,
        real_clear_at: None,
    };
    intent_tx
        .send((mk_intent(), std::time::Instant::now()))
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    intent_tx
        .send((mk_intent(), std::time::Instant::now()))
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    let pos = state
        .get_position("tok-cy", "arb")
        .await
        .unwrap()
        .map(|p| p.size)
        .unwrap_or_default();
    (pos, ledger.net_exposure("cid-cap"))
}

#[tokio::test(flavor = "current_thread")]
async fn cap_bracket_rejects_equally_in_realtime_and_replay() {
    let _serial = REPLAY_LOCK.lock().await;
    // Realtime paper: same tape (two identical books, two buys of 7.5 USDC
    // against a bracket cap of 8).
    let (pos_rt, net_rt) = fixture_realtime_two_buys().await;

    // Replay: same sequence — two books (the second is the same observation); the
    // strategy emits one buy per book, bracket cap 8 USDC.
    let mut params = default_params();
    params.latency_ms = 2; // same as the realtime fixture
    params.risk.max_bracket_usdc = dec!(8);
    let (mut engine, _state) = new_engine(params).await;
    let mut strat = BuyAlways::new("cid-cap", "tok-cy", dec!(0.75), dec!(10), 2);
    let snap = book("cid-cap", "btc-updown-cap", "tok-cy", "tok-cn", "0.75");
    // The third book (T0+31s) extends the tape past the Maintenance at
    // T0+30s: `purge_and_resync` rewrites the ledger with the persisted
    // `cost_basis` (executed price with slippage+fee), matching it with the
    // resync the realtime executor runs on its first immediate tick.
    // Without it the replay would compare the claim at intent price (7.5) against the
    // real cost (~7.63) — a known resync drift, not a divergence.
    let out = engine
        .run(
            Tape {
                books: vec![
                    tape_book(T0, snap.clone()),
                    tape_book(T0 + 100, snap.clone()),
                    tape_book(T0 + 31_000, snap),
                ],
                spots: vec![],
                ..Default::default()
            },
            &mut strat,
        )
        .await
        .unwrap();

    assert_eq!(pos_rt, dec!(10), "realtime: only the first buy filled");
    assert_eq!(
        out.market_fills, 1,
        "replay: only the first buy must fill — the second is rejected by the cap"
    );
    assert_eq!(
        out.gate_rejected, 1,
        "the second buy is rejected by bracket_cap"
    );
    // The differential assertion is the EQUALITY between legs after the resync:
    // both ledgers converge to the real `cost_basis` (executed price with
    // slippage 5bps + parabolic fee — symmetric params).
    assert_eq!(
        net_rt,
        out.net_exposure("cid-cap"),
        "same net exposure in realtime and replay (post-resync)"
    );
    assert!(
        out.net_exposure("cid-cap") > dec!(7.5),
        "the executed cost includes slippage+fee (symmetric params)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Coverage of the pipeline paths in replay
// ═══════════════════════════════════════════════════════════════════════════

/// Strategy that emits a resting buy (GTD) at the book's bid and then cancels it
/// on the next call — exercises GTD + cancel.
struct RestingThenCancels {
    token: String,
    phase: usize,
}

impl Strategy for RestingThenCancels {
    fn id(&self) -> StrategyId {
        StrategyId::ParityMaker
    }
    fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
        let mk = |side: OrderSide, resting: bool| OrderIntent {
            market_cid: book.market_condition_id.clone(),
            token_id: self.token.clone(),
            side,
            price: dec!(0.40),
            size: dec!(10),
            clear_at: book.clear_at,
            signal: dec!(0.40),
            neg_risk: false,
            limit_price: Some(dec!(0.40)),
            second_leg: None,
            strategy: StrategyId::ParityMaker,
            resting,
            real_clear_at: None,
        };
        let r = match self.phase {
            0 => vec![mk(OrderSide::Buy, true)],
            1 => vec![mk(OrderSide::Cancel, false)],
            _ => vec![],
        };
        self.phase += 1;
        r
    }
}

/// A resting GTD that never fills (the ask stays above the bid) expires
/// when the replay crosses its clear_at — TTL exercised by the sim's tick_book
/// against the virtual clock.
#[tokio::test(flavor = "current_thread")]
async fn gtd_expires_by_ttl_on_virtual_clock() {
    let _serial = REPLAY_LOCK.lock().await;
    let mut params = default_params();
    params.latency_ms = 0;
    let (mut engine, state) = new_engine(params).await;

    let cid = "cid-ttl";
    let short_clear = T0 / 1000 + 10; // closes 10s after the first book
    let mut b = book(cid, "btc-updown-ttl", "tok-ty", "tok-tn", "0.90");
    b.clear_at = short_clear;
    // Ask at 0.90: the GTD bid at 0.40 never crosses.
    b.yes.bids.levels[0].price = dec!(0.39);

    let mut strat = RestingThenCancels {
        token: "tok-ty".into(),
        phase: 0,
    };
    // Books 20s apart — the second event is already past clear_at.
    let mut b2 = b.clone();
    b2.yes.asks.levels[0].price = dec!(0.91);
    let tape = Tape {
        books: vec![tape_book(T0, b), tape_book(T0 + 20_000, b2)],
        spots: vec![],
        ..Default::default()
    };
    let out = engine.run(tape, &mut strat).await.unwrap();

    // The GTD was accepted (resting in the book) and expired on the virtual clock:
    // no fills, no position, and the maintenance purge removed it.
    assert_eq!(out.market_fills, 0);
    assert!(state
        .get_position("tok-ty", "parity_maker")
        .await
        .unwrap()
        .is_none());
    assert_eq!(out.final_live_gtds(), 0, "the GTD expired by TTL");
}

/// Unknown taker: the sim loses the ACK (`set_unknown_rate(1.0)`), the order
/// executes at the venue but the executor never learns it → pending in the journal and
/// the bracket is blocked for the next buy (GateRejected unresolved_unknown).
#[tokio::test(flavor = "current_thread")]
async fn unknown_blocks_bracket_in_replay() {
    let _serial = REPLAY_LOCK.lock().await;
    let params = default_params();
    let (mut engine, state) = new_engine(params).await;
    engine.sim().set_unknown_rate(1.0); // ALL ACKs are lost

    let cid = "cid-unk";
    let snap = book(cid, "btc-updown-unk", "tok-uy", "tok-un", "0.50");
    let mut strat = BuyAlways::new(cid, "tok-uy", dec!(0.50), dec!(5), 2);
    let out = engine
        .run(
            Tape {
                books: vec![tape_book(T0, snap.clone()), tape_book(T0 + 1_000, snap)],
                spots: vec![],
                ..Default::default()
            },
            &mut strat,
        )
        .await
        .unwrap();

    // The first buy's fill WAS applied (the order reached the venue; only
    // the ACK was lost) — the position exists despite the Unknown.
    let pos = state.get_position("tok-uy", "arb").await.unwrap();
    assert_eq!(pos.map(|p| p.size), Some(dec!(5)));

    // The second buy of the same bracket was blocked by the journal.
    assert!(
        out.gate_rejected >= 1,
        "the second buy must be rejected by unresolved_unknown"
    );
    assert!(out.unresolved_unknowns >= 1, "journal with pending");
}

/// Unknown resolution on the virtual clock (complements the test above):
/// the ACK is lost but the order executed at the venue → the journal stays
/// pending until, once the taker grace period has passed (`TAKER_TRADES_GRACE_MS` = 90s,
/// run on the virtual clock), maintenance queries `trades_since`,
/// matches the execution by `order_id` and resolves `matched-trade` (the fill is
/// already durable — `fill_exists_exec` does not need to synthesize it). Afterwards the
/// bracket is UNBLOCKED: the next buy passes the gate and fills.
/// This proves the journal reconciles with evidence, not just blocks.
#[tokio::test(flavor = "current_thread")]
async fn unknown_taker_resolves_by_trades_after_grace() {
    let _serial = REPLAY_LOCK.lock().await;
    let params = default_params(); // maintenance_ms = 30s
    let (mut engine, state) = new_engine(params).await;
    engine.sim().set_unknown_rate(1.0);

    let cid = "cid-unkres";
    let snap = book(cid, "btc-updown-ures", "tok-uy2", "tok-un2", "0.50");
    let mut strat = BuyAlways::new(cid, "tok-uy2", dec!(0.50), dec!(5), 2);
    let out1 = engine
        .run(
            Tape {
                books: vec![tape_book(T0, snap.clone())],
                spots: vec![],
                ..Default::default()
            },
            &mut strat,
        )
        .await
        .unwrap();
    assert_eq!(
        out1.market_fills, 1,
        "the order executed at the venue — only the ACK was lost"
    );
    assert!(
        state.count_unresolved_unknowns().await.unwrap() >= 1,
        "journal pending after the lost ACK"
    );

    // The venue stops losing ACKs and the tape advances past the grace period:
    // the Maintenance runs at T0+30/60/90s rescheduled in this run trigger
    // `purge_and_resync` → `reconcile_unknowns`; on the 90s tick
    // (now − ts_ms ≥ grace) `trades_since` is queried, the matched exec
    // resolves the pending and the bracket unblocks before the 95s book.
    engine.sim().set_unknown_rate(0.0);
    let out2 = engine
        .run(
            Tape {
                books: vec![
                    tape_book(T0 + 95_000, snap.clone()),
                    tape_book(T0 + 96_000, snap),
                ],
                spots: vec![],
                ..Default::default()
            },
            &mut strat,
        )
        .await
        .unwrap();

    assert_eq!(
        state.count_unresolved_unknowns().await.unwrap(),
        0,
        "the pending was resolved by matched-trade in the T0+90s maintenance"
    );
    assert_eq!(out2.unresolved_unknowns, 0);
    // Real unblock: the second buy (emitted when observing the T0+95s book,
    // after the T0+90s resolution) passed the gate and filled.
    assert_eq!(
        out2.market_fills, 1,
        "the second buy fills once the Unknown resolved"
    );
    let pos = state.get_position("tok-uy2", "arb").await.unwrap().unwrap();
    assert_eq!(pos.size, dec!(10), "two fills of 5 shares accumulated");
}

/// Cap coverage in replay (in addition to the `max_bracket_usdc` differential):
/// the GLOBAL cap (`max_total_exposure_usdc`, read fresh by the executor on
/// each Buy) and the per-strategy rate limit (`max_orders_per_hour`, sliding
/// window over `crate::time::Instant` — replay time) reject through
/// the same production gate.
#[tokio::test(flavor = "current_thread")]
async fn cap_global_and_rate_limit_reject_in_replay() {
    let _serial = REPLAY_LOCK.lock().await;
    let cid = "cid-glob";
    let snap = book(cid, "btc-updown-glob", "tok-gy", "tok-gn", "0.75");
    let tape = || Tape {
        books: vec![
            tape_book(T0, snap.clone()),
            tape_book(T0 + 100, snap.clone()),
        ],
        spots: vec![],
        ..Default::default()
    };

    // (a) Global cap: 10 USDC total — two buys of 7.5 do not fit together.
    let mut params = default_params();
    params.risk.max_total_exposure_usdc = dec!(10);
    let (mut engine, _state) = new_engine(params).await;
    let mut strat = BuyAlways::new(cid, "tok-gy", dec!(0.75), dec!(10), 2);
    let out = engine.run(tape(), &mut strat).await.unwrap();
    assert_eq!(
        out.market_fills, 1,
        "only the first buy fits within max_total_exposure_usdc=10"
    );
    assert!(
        out.gate_rejected >= 1,
        "the second buy is rejected by the global exposure gate"
    );

    // (b) Rate limit: 1 order/hour for Arb — the second buy is rejected by
    //     rate_limit even though the caps have room (the two books are 100ms
    //     apart in replay, within the same sliding window).
    let (mut engine, _state) = new_engine(default_params()).await;
    engine
        .risk()
        .set_strategy_limits(StrategyId::Arb, 1, dec!(1000), Decimal::ZERO);
    let mut strat = BuyAlways::new(cid, "tok-gy", dec!(0.75), dec!(10), 2);
    let out = engine.run(tape(), &mut strat).await.unwrap();
    assert_eq!(out.market_fills, 1, "only the first order consumes the quota");
    assert!(
        out.gate_rejected >= 1,
        "the second buy is rejected by rate_limit"
    );
}

/// Partial FAK fill: ask with less depth than the requested size — the FAK
/// fills what is there and the rest dies; the position reflects only what was executed.
#[tokio::test(flavor = "current_thread")]
async fn fak_partial_fill_counts_only_executed() {
    let _serial = REPLAY_LOCK.lock().await;
    let mut params = default_params();
    // NearCertain is in fak_strategies → the Limit goes out as a FAK.
    params.fak_strategies.insert(StrategyId::NearCertain);
    let (mut engine, state) = new_engine(params).await;

    let cid = "cid-partial";
    let mut snap = book(cid, "btc-updown-pair", "tok-py", "tok-pn", "0.50");
    snap.yes.asks.levels[0].size = dec!(4); // only 4 shares available

    struct OneFak {
        done: bool,
    }
    impl Strategy for OneFak {
        fn id(&self) -> StrategyId {
            StrategyId::NearCertain
        }
        fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
            if self.done {
                return vec![];
            }
            self.done = true;
            vec![OrderIntent {
                market_cid: book.market_condition_id.clone(),
                token_id: "tok-py".into(),
                side: OrderSide::Buy,
                price: dec!(0.50),
                size: dec!(10), // asks for 10, only 4 available
                clear_at: book.clear_at,
                signal: dec!(0.50),
                neg_risk: false,
                limit_price: Some(dec!(0.50)),
                second_leg: None,
                strategy: StrategyId::NearCertain,
                resting: false,
                real_clear_at: None,
            }]
        }
    }
    let mut strat = OneFak { done: false };
    let out = engine
        .run(
            Tape {
                books: vec![tape_book(T0, snap)],
                spots: vec![],
                ..Default::default()
            },
            &mut strat,
        )
        .await
        .unwrap();

    let pos = state
        .get_position("tok-py", "near_certain")
        .await
        .unwrap()
        .expect("partial position");
    assert_eq!(pos.size, dec!(4), "only 4 shares filled (depth)");
    assert_eq!(out.market_fills, 1);
}

/// Cancel/fill race: a resting GTD fills on the SAME tick its cancel
/// arrives — the fill is already durable when the cancel confirms.
/// Exercises the executor's cancel path (cancel Ok with matched>0 → pending
/// Unknown + reconciliation with trade evidence).
#[tokio::test(flavor = "current_thread")]
async fn cancel_fill_in_same_tick_is_deterministic() {
    let _serial = REPLAY_LOCK.lock().await;
    let mut params = default_params();
    params.latency_ms = 50; // the cancel is slow to arrive: the fill wins
    params.maintenance_ms = 500; // frequent purge/resync to resolve the pending
    let (mut engine, state) = new_engine(params).await;

    let cid = "cid-race";
    // Book1: a resting bid at 0.40 is posted. Book2: the ask DROPS to 0.40 → the GTD
    // fills (our bid is the best) on the same event in which the agent
    // decides to cancel upon seeing book2.
    let mut b1 = book(cid, "btc-updown-race", "tok-ry", "tok-rn", "0.60");
    b1.yes.bids.levels[0].price = dec!(0.39);
    let mut b2 = book(cid, "btc-updown-race", "tok-ry", "tok-rn", "0.40");
    b2.yes.asks.levels[0].price = dec!(0.40);
    b2.yes.bids.levels[0].price = dec!(0.39);

    struct PostsAndCancels {
        phase: usize,
    }
    impl Strategy for PostsAndCancels {
        fn id(&self) -> StrategyId {
            StrategyId::ParityMaker
        }
        fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
            let r = match self.phase {
                0 => vec![OrderIntent {
                    market_cid: book.market_condition_id.clone(),
                    token_id: "tok-ry".into(),
                    side: OrderSide::Buy,
                    price: dec!(0.40),
                    size: dec!(10),
                    clear_at: book.clear_at,
                    signal: dec!(0.40),
                    neg_risk: false,
                    limit_price: Some(dec!(0.40)),
                    second_leg: None,
                    strategy: StrategyId::ParityMaker,
                    resting: true,
                    real_clear_at: None,
                }],
                1 => vec![OrderIntent {
                    market_cid: book.market_condition_id.clone(),
                    token_id: "tok-ry".into(),
                    side: OrderSide::Cancel,
                    price: dec!(0.40),
                    size: dec!(10),
                    clear_at: book.clear_at,
                    signal: dec!(0.40),
                    neg_risk: false,
                    limit_price: None,
                    second_leg: None,
                    strategy: StrategyId::ParityMaker,
                    resting: false,
                    real_clear_at: None,
                }],
                _ => vec![],
            };
            self.phase += 1;
            r
        }
    }

    let tape = Tape {
        books: vec![tape_book(T0, b1), tape_book(T0 + 100, b2)],
        spots: vec![],
        ..Default::default()
    };
    let out = engine
        .run(tape, &mut PostsAndCancels { phase: 0 })
        .await
        .unwrap();

    // The GTD filled against book2 (ask touched our bid) before the
    // cancel reached the venue — the position exists and is durable.
    let pos = state.get_position("tok-ry", "parity_maker").await.unwrap();
    assert!(
        pos.is_some(),
        "the fill must have won the race against the cancel"
    );
    assert!(out.market_fills >= 1);
}

/// Close + restore: an Unknown stays pending; the executor restarts
/// over the SAME ephemeral StateStore with a different RiskConfig — the new executor
/// seeds the durable journal and keeps blocking the bracket (conservative
/// behavior: it decides from what was persisted, not from the process config).
#[tokio::test(flavor = "current_thread")]
async fn restart_with_unknown_journal_blocks_bracket_under_other_config() {
    let _serial = REPLAY_LOCK.lock().await;
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let mut params = default_params();
    let mut engine = ReplayEngine::new(params.clone(), state.clone())
        .await
        .unwrap();
    engine.sim().set_unknown_rate(1.0);

    let cid = "cid-restart";
    let snap = book(cid, "btc-updown-rs", "tok-ry2", "tok-rn2", "0.50");
    let mut strat = BuyAlways::new(cid, "tok-ry2", dec!(0.50), dec!(5), 1);
    engine
        .run(
            Tape {
                books: vec![tape_book(T0, snap)],
                spots: vec![],
                ..Default::default()
            },
            &mut strat,
        )
        .await
        .unwrap();
    assert!(state.count_unresolved_unknowns().await.unwrap() >= 1);

    // Restart: new executor over the SAME state, with a different config
    // (caps changed — the blocker must come from the JOURNAL, not from the config).
    params.risk.max_total_exposure_usdc = dec!(999_999);
    params.risk.max_bracket_usdc = dec!(999_999);
    engine.set_params(params);
    engine.restart_executor().await.unwrap();

    let mut strat2 = BuyAlways::new(cid, "tok-ry2", dec!(0.50), dec!(5), 1);
    let snap2 = book(cid, "btc-updown-rs", "tok-ry2", "tok-rn2", "0.50");
    let out = engine
        .run(
            Tape {
                books: vec![tape_book(T0 + 2_000, snap2)],
                spots: vec![],
                ..Default::default()
            },
            &mut strat2,
        )
        .await
        .unwrap();
    assert_eq!(
        out.market_fills, 0,
        "after restart the bracket stays blocked by the durable journal"
    );
    assert!(out.gate_rejected >= 1);
}

/// Settlement: a position still open at clear_at is settled against the last spot
/// versus the reference_price — via the real `state.apply_settlement`.
#[tokio::test(flavor = "current_thread")]
async fn settlement_close_liquidates_position_with_spot() {
    let _serial = REPLAY_LOCK.lock().await;
    use poly_bot::replay::TapeSpot;
    // Settlement via spot proxy: explicit diagnostic mode — the certified path
    // requires an official label; there is none here, so this test
    // documents the historical proxy.
    let mut params = default_params();
    params.settlement_proxy_diagnostic = true;
    let (mut engine, state) = new_engine(params).await;

    let cid = "cid-settle";
    let clear = T0 / 1000 + 60;
    let mut b = book(cid, "btc-updown-st", "tok-sy", "tok-sn", "0.60");
    b.clear_at = clear;
    b.reference_price = Some(dec!(50_000));

    let mut strat = BuyOnce::new("tok-sy", dec!(10));
    strat.resting = false;

    let tape = Tape {
        books: vec![
            tape_book(T0, b.clone()),
            tape_book(clear * 1000 + 2_000, b), // after the close
        ],
        spots: vec![TapeSpot {
            ts_ms: clear * 1000 + 1_000,
            tape_seq: None,
            asset: "BTC".into(),
            price: dec!(51_000), // > strike 50000 → YES wins
        }],
        ..Default::default()
    };
    let out = engine.run(tape, &mut strat).await.unwrap();

    assert_eq!(out.settlements, 1, "the position was settled at clear_at");
    let pos = state.get_position("tok-sy", "arb").await.unwrap();
    // apply_settlement leaves the row with size=0 and outcome_won — the position
    // no longer counts as open.
    let still_open = pos.map(|p| p.size > Decimal::ZERO).unwrap_or(false);
    assert!(!still_open);
    // Payout: YES won → 10 shares at $1 = +10 USDC redeemed (vs cost ~6).
    assert!(out.realized_pnl() > Decimal::ZERO);
}

// ═══════════════════════════════════════════════════════════════════════════
// Latency legs of the research profile ([research_sim.latencies])
// ═══════════════════════════════════════════════════════════════════════════

/// Nominal research profile with the given stages — the rest stays at the
/// profile's nominal defaults (they do not alter the sim, only the latencies).
fn research_with_latencies(
    f: impl Fn(&mut poly_bot::config::LatencyProfile),
) -> poly_bot::config::ResearchSimConfig {
    let mut r = poly_bot::config::ResearchSimConfig::nominal();
    f(&mut r.latencies);
    r
}

/// The `submit` stage delays arrival at the venue: a FOK sent with the
/// ask at 0.50 but arriving once the book has moved to 0.99 must be REJECTED
/// by the router, not the filled quantity against the stale book.
#[tokio::test(flavor = "current_thread")]
async fn submit_leg_delays_arrival_at_venue() {
    let _serial = REPLAY_LOCK.lock().await;
    let cid = "cid-lat";
    let mut b1 = book(cid, "btc-updown-lat", "tok-ly", "tok-ln", "0.50");
    b1.yes.asks.levels[0].price = dec!(0.50);
    let mut b2 = book(cid, "btc-updown-lat", "tok-ly", "tok-ln", "0.99");
    b2.yes.asks.levels[0].price = dec!(0.99);
    let tape = |recv_gap: i64| Tape {
        books: vec![
            tape_book(T0, b1.clone()),
            tape_book(T0 + recv_gap, b2.clone()),
        ],
        spots: vec![],
        ..Default::default()
    };

    // Without a submit stage: the FOK arrives before book2 → fills at 0.50.
    let (mut e0, _s0) = new_engine({
        let mut p = default_params();
        p.research = Some(research_with_latencies(|_| {}));
        p
    })
    .await;
    let mut s0 = BuyOnce::new("tok-ly", dec!(10));
    let out0 = e0.run(tape(200), &mut s0).await.unwrap();
    assert_eq!(out0.market_fills, 1, "without submit latency the FOK fills");

    // With submit_ms > gap: it arrives with book2 live (ask 0.99) → rejected.
    let (mut e1, _s1) = new_engine({
        let mut p = default_params();
        p.research = Some(research_with_latencies(|l| l.submit_ms = 500.0));
        p
    })
    .await;
    let mut s1 = BuyOnce::new("tok-ly", dec!(10));
    let out1 = e1.run(tape(200), &mut s1).await.unwrap();
    assert_eq!(
        out1.market_fills, 0,
        "the submit reaches the venue after book2 → the FOK at 0.50 cannot fill"
    );
    assert!(
        out1.rejected_router + out1.gate_rejected >= 1,
        "the router/gate must reject the FOK outside price"
    );
}

/// The `feed` stage delays the OBSERVATION of the book: if the delay exceeds
/// `max_book_age_ms`, the runner's stale-book gate blocks the entry.
#[tokio::test(flavor = "current_thread")]
async fn feed_leg_triggers_stale_book_gate() {
    let _serial = REPLAY_LOCK.lock().await;
    let cid = "cid-feed";
    let mut b = book(cid, "btc-updown-feed", "tok-fy", "tok-fn", "0.50");
    b.yes.asks.levels[0].price = dec!(0.50);
    let tape = || Tape {
        books: vec![tape_book(T0, b.clone())],
        spots: vec![],
        ..Default::default()
    };

    let mut base = default_params();
    base.max_book_age_ms = 500;

    // feed=0: observes fresh → intent → fill.
    let (mut e0, _s0) = new_engine({
        let mut p = base.clone();
        p.research = Some(research_with_latencies(|_| {}));
        p
    })
    .await;
    let mut s0 = BuyOnce::new("tok-fy", dec!(10));
    let out0 = e0.run(tape(), &mut s0).await.unwrap();
    assert_eq!(out0.market_fills, 1);

    // feed=2000 > max_book_age=500: the observation arrives degraded → gate.
    let (mut e1, _s1) = new_engine({
        let mut p = base.clone();
        p.research = Some(research_with_latencies(|l| l.feed_ms = 2_000.0));
        p
    })
    .await;
    let mut s1 = BuyOnce::new("tok-fy", dec!(10));
    let out1 = e1.run(tape(), &mut s1).await.unwrap();
    assert_eq!(
        out1.market_fills, 0,
        "with feed > max_book_age the entry stays blocked as stale"
    );
}

/// Stage jitter is deterministic per seed: same tape +
/// profile with jitter ⇒ same digest (and different from a neighboring seed).
#[tokio::test(flavor = "current_thread")]
async fn latencies_with_jitter_are_deterministic_per_seed() {
    let _serial = REPLAY_LOCK.lock().await;
    let params = |seed: u64| ReplayParams {
        seed,
        research: Some(research_with_latencies(|l| {
            l.submit_ms = 40.0;
            l.ack_ms = 20.0;
            l.feed_ms = 5.0;
            l.decision_ms = 3.0;
            l.jitter_ms = 10.0;
            l.load_mult = 0.5;
        })),
        ..default_params()
    };
    let run_once = |p: ReplayParams| async move {
        let (mut engine, _state) = new_engine(p).await;
        let mut strat = BuyAlways::new("cid-det", "tok-dy", dec!(0.55), dec!(5), 6);
        engine
            .run(deterministic_tape(), &mut strat)
            .await
            .unwrap()
            .digest
    };
    let a1 = run_once(params(0xC0FF_EE15)).await;
    let a2 = run_once(params(0xC0FF_EE15)).await;
    let b1 = run_once(params(0xDEAD_BEEF)).await;
    assert_eq!(a1, a2, "same seed + jitter ⇒ same ledger");
    assert_ne!(a1, b1, "different seed ⇒ different jitter sample");
}

/// The `ack` stage delays DELIVERY of the response to the executor: with a large
/// ack the executor stays parked waiting for the responses of its calls
/// before the submit, the intent ages in its queue and the entry
/// is dropped by a gate — exactly the effect of a slow round trip in live.
/// Observable: a high `ack_ms` changes the outcome vs `ack_ms=0`.
#[tokio::test(flavor = "current_thread")]
async fn ack_leg_delays_response_to_executor() {
    let _serial = REPLAY_LOCK.lock().await;
    let cid = "cid-ack";
    let mut b = book(cid, "btc-updown-ack", "tok-ay", "tok-an", "0.50");
    b.yes.asks.levels[0].price = dec!(0.50);
    let tape = || Tape {
        books: vec![tape_book(T0, b.clone())],
        spots: vec![],
        ..Default::default()
    };

    // ack=0: immediate response → normal submit and fill.
    let (mut e0, _s0) = new_engine({
        let mut p = default_params();
        p.research = Some(research_with_latencies(|_| {}));
        p
    })
    .await;
    let mut s0 = BuyOnce::new("tok-ay", dec!(10));
    let out0 = e0.run(tape(), &mut s0).await.unwrap();
    assert_eq!(out0.market_fills, 1, "ack=0 → normal fill");

    // Large ack: the executor does not see the response in time → the entry does not
    // execute (stale-intent gate / degraded flow — no fill).
    let (mut e1, _s1) = new_engine({
        let mut p = default_params();
        p.research = Some(research_with_latencies(|l| l.ack_ms = 5_000.0));
        p
    })
    .await;
    let mut s1 = BuyOnce::new("tok-ay", dec!(10));
    let out1 = e1.run(tape(), &mut s1).await.unwrap();
    assert_eq!(
        out1.market_fills, 0,
        "a high ack degrades the pipeline: the intent ages before the submit"
    );
    assert!(
        out1.rejected_router + out1.gate_rejected >= 1,
        "the rejection is recorded"
    );
}
