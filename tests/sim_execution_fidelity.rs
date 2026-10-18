//! Execution-fidelity tests for the simulator (T5): execution based on
//! depth/tape and queue scenarios over `SimulatedRouter`.
//!
//! Cases:
//! - A depth reduction caused by a cancel does NOT create a fill on its own; the
//!   attribution of the shrink (cancels ahead vs. behind) is an explicit
//!   hypothesis of the profile (`cancel_attr`), not a hidden constant.
//! - A trade plus its twin L2 delta do not count the queue twice.
//! - Two of our own orders do not consume the same displayed volume twice.
//! - An in-flight cancel can lose a race against a fill.
//! - An expired order does not fill and releases its reserve.
//! - Tape gaps invalidate the episode (abstention) or degrade the
//!   confidence of the fills they produce.
//! - Funds, reserves and inventory are never fabricated.
//! - Every simulated fill carries evidence, hypothesis and cost traceability.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{mpsc, watch};

use poly_bot::{
    config::ResearchSimConfig,
    market_feed::MarketRegistry,
    order::{
        router::{OrderKind, OrderRequest, OrderRouter, RouterOrderStatus, TimeInForce},
        sim_router::{SimClock, SimConfidence, SimEvidence, SimHypothesis, SimulatedRouter},
    },
    types::{BookSnapshot, Fill, HalfBook, Level, StrategyId, TokenBook},
};

// ── Fixtures ─────────────────────────────────────────────────────────────────

const T0_MS: i64 = 1_700_000_000_000;
const CLEAR_AT: i64 = (T0_MS / 1000) + 7_200;

fn clock() -> Arc<AtomicI64> {
    Arc::new(AtomicI64::new(T0_MS))
}

fn lv(p: &str, s: &str) -> Level {
    Level {
        price: p.parse().unwrap(),
        size: s.parse().unwrap(),
    }
}

/// Book for "tok-yes" (asks asc, bids desc) with a symmetric, static "tok-no".
fn book_levels(asks: Vec<Level>, bids: Vec<Level>) -> BookSnapshot {
    book_pair(asks, bids, vec![lv("0.50", "100")], vec![lv("0.49", "100")])
}

/// Complete book: `asks_yes`/`bids_yes` for tok-yes and `asks_no`/`bids_no`
/// for tok-no (the cross YES↔NO consensus is checked by the stale marker).
fn book_pair(
    asks_yes: Vec<Level>,
    bids_yes: Vec<Level>,
    asks_no: Vec<Level>,
    bids_no: Vec<Level>,
) -> BookSnapshot {
    BookSnapshot {
        market_condition_id: "cid-sim".into(),
        yes: TokenBook {
            token_id: "tok-yes".into(),
            bids: HalfBook { levels: bids_yes },
            asks: HalfBook { levels: asks_yes },
            ..Default::default()
        },
        no: TokenBook {
            token_id: "tok-no".into(),
            bids: HalfBook { levels: bids_no },
            asks: HalfBook { levels: asks_no },
            ..Default::default()
        },
        market_open_ts: T0_MS / 1000 - 3_600,
        clear_at: CLEAR_AT,
        neg_risk: false,
        slug: "test-sim".into(),
        reference_price: Some(dec!(50_000)),
        fee_rate_bps: None,
        parabolic_fee_rate: None,
        tick_size: None,
        recv_instant: None,
    }
}

fn req_kind(kind: OrderKind) -> OrderRequest {
    OrderRequest {
        market_cid: "cid-sim".into(),
        token_id: "tok-yes".into(),
        kind,
        neg_risk: false,
        strategy: StrategyId::Arb,
        clear_at: CLEAR_AT,
        signal: dec!(0.5),
        // post_only by default: resting orders are genuine makers (fee 0), and a
        // post_only order that crosses at submit is rejected by the venue.
        post_only: true,
        real_clear_at: None,
        local_order_id: Some("loc-1".into()),
    }
}

fn limit(price: Decimal, size: Decimal, tif: TimeInForce) -> OrderRequest {
    req_kind(OrderKind::Limit { price, size, tif })
}

/// Deterministic sim: no fee, fok_fill_rate=1, injectable clock.
fn sim_with_clock(
    clock: Arc<AtomicI64>,
    balance: Decimal,
    latency_ms: u64,
) -> (
    Arc<SimulatedRouter>,
    mpsc::UnboundedReceiver<Fill>,
    watch::Receiver<Option<Decimal>>,
) {
    let (fill_tx, fill_rx) = mpsc::unbounded_channel();
    let (wallet_tx, wallet_rx) = watch::channel(None);
    let clk: SimClock = {
        let c = clock.clone();
        Arc::new(move || c.load(Ordering::Relaxed))
    };
    let sim = SimulatedRouter::with_deps(
        0,
        1.0,
        latency_ms,
        Decimal::ZERO,
        1.0,
        fill_tx,
        Arc::new(MarketRegistry::new()),
        balance,
        wallet_tx,
        0xC0FF_EE15,
        clk,
    );
    (sim, fill_rx, wallet_rx)
}

// ── Behavioural cases (phase 1, existing API) ──────────────────

/// A reduction of the level WITHOUT a cross does not create a fill on its own — the shrink
/// only drains the external queue (nominal hypothesis `cancel_attr = 1`: the whole
/// shrink is attributed to cancels/consumption ahead). Without evidence of
/// trade-through we never execute.
#[tokio::test]
async fn shrink_without_cross_creates_no_fill_but_drains_nominal_queue() {
    let (sim, mut rx, _w) = sim_with_clock(clock(), dec!(1000), 0);
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "5")]));
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
        .await
        .order_id
        .expect("GTD accepted");

    // The level disappears without the ask crossing: it could be cancels — never
    // enough evidence for a fill.
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.49", "8")]));
    assert!(
        rx.try_recv().is_err(),
        "a shrink without a cross never creates a fill (cancel is not liquidity)"
    );

    // Nominal hypothesis: the shrink drained the queue ahead → a small
    // cross already executes us.
    sim.tick_book(&book_levels(vec![lv("0.50", "3")], vec![lv("0.49", "8")]));
    assert_eq!(
        rx.try_recv().expect("queue drained → fill by cross").size,
        dec!(3)
    );
}

/// Trade + twin L2 delta: what an OWN taker takes from a resting order's level
/// removes the queue ONCE (via `consume_depth`); when the
/// next snapshot reflects that same consumption (the displayed level dropped), the
/// L2 delta cannot deduct it again — that would count the same flow
/// twice.
#[tokio::test]
async fn trade_and_twin_delta_are_not_counted_twice() {
    let (sim, mut rx, _w) = sim_with_clock(clock(), dec!(1000), 0);
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "5")]));
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
        .await
        .order_id
        .expect("GTD accepted");

    // Our taker sells 3 against bids@0.50 → the external queue is left at 2.
    let s = sim
        .submit(req_kind(OrderKind::Sell {
            size: Some(dec!(3)),
        }))
        .await;
    assert!(!s.rejected);
    let _ = rx.try_recv().expect("fill of our own sell taker");

    // The venue reflects the real consumption: bids@0.50 now shows 2. The delta
    // twin (5 → 2) CANNOT drain the queue again: it would end at 0 due to a
    // double count of the same flow.
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "2")]));
    assert!(rx.try_recv().is_err(), "twin delta without a cross does not fill");

    // Cross of 3 with the real queue at 2 → only the excess (1) executes us.
    sim.tick_book(&book_levels(vec![lv("0.50", "3")], vec![lv("0.50", "2")]));
    let f = rx
        .try_recv()
        .expect("the excess over the real queue executes us");
    assert_eq!(
        f.size,
        dec!(1),
        "twin trade+delta drained the queue twice (expected excess 1)"
    );
    assert!(rx.try_recv().is_err(), "only one partial fill");
}

/// Two own resting orders at the same level are serialized: the same flow cannot
/// fill both with the same volume (the second puts the
/// first one ahead in its queue).
#[tokio::test]
async fn two_own_resting_do_not_consume_same_volume_twice() {
    let (sim, mut rx, _w) = sim_with_clock(clock(), dec!(1000), 0);
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "5")]));
    let a = sim
        .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
        .await
        .order_id
        .expect("GTD A");
    let b = sim
        .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
        .await
        .order_id
        .expect("GTD B");

    // Cross of 8: 5 empty the external queue and the excess (3) executes ONLY A —
    // B has A ahead of it; if the same flow filled both, the displayed volume
    // would have been consumed twice.
    sim.tick_book(&book_levels(vec![lv("0.50", "8")], vec![lv("0.50", "5")]));
    let f = rx.try_recv().expect("only A is fill-eligible");
    assert_eq!(f.order_id.as_deref(), Some(a.as_str()));
    assert_eq!(f.size, dec!(3));
    assert!(
        rx.try_recv().is_err(),
        "B cannot fill with the same flow as A"
    );

    // Cross of 12: A completes its 7 remaining; the excess (5) reaches B.
    sim.tick_book(&book_levels(vec![lv("0.50", "12")], vec![lv("0.50", "5")]));
    let mut fills = [rx.try_recv().unwrap(), rx.try_recv().unwrap()];
    fills.sort_by_key(|f| f.order_id.clone());
    assert_eq!(fills[0].order_id.as_deref(), Some(a.as_str()));
    assert_eq!(fills[0].size, dec!(7), "A completes first");
    assert_eq!(fills[1].order_id.as_deref(), Some(b.as_str()));
    assert_eq!(fills[1].size, dec!(5), "only the excess serializes to B");
    assert!(rx.try_recv().is_err());
}

/// An IN-FLIGHT cancellation can lose against a fill: the venue processes
/// orders in arrival order and the cancel travels with the same latency as the
/// submit — if the book crosses during flight, the fill wins the race.
/// (Replica of the live scenario: `cancel` returns Ok but the order was already
/// executed — verified by `order_status`, not by the ack.)
#[tokio::test(flavor = "current_thread")]
async fn in_flight_cancel_can_lose_to_fill() {
    let (sim, mut rx, _w) = sim_with_clock(clock(), dec!(1000), 50);
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "5")]));
    let oid = sim
        .submit(limit(dec!(0.50), dec!(5), TimeInForce::Gtd))
        .await
        .order_id
        .expect("GTD accepted");

    // We emit the cancel (it stays in flight `latency_ms`) and MEANWHILE
    // a book arrives that crosses our level with ample volume.
    let sim2 = sim.clone();
    let oid2 = oid.clone();
    let handle = tokio::spawn(async move { sim2.cancel("cid-sim", &oid2).await });
    tokio::task::yield_now().await; // the cancel starts and stays in flight
    sim.tick_book(&book_levels(vec![lv("0.50", "10")], vec![lv("0.50", "5")]));

    let res = handle.await.expect("cancel completed");
    assert!(res.is_ok(), "the venue processed the cancel (too late)");
    let f = rx
        .try_recv()
        .expect("the fill reached the venue before the cancel — it loses the race");
    assert_eq!(f.size, dec!(5));
    match sim.order_status(&oid).await {
        Some(RouterOrderStatus::Gone { matched }) => assert_eq!(matched, dec!(5)),
        other => panic!("an order already executed cannot remain Live: {other:?}"),
    }
}

/// An expired order does not fill even if the book crosses later: the CLOB purge
/// (replicated with the live margin `clear_at + 60s`) removes it
/// before the tick's matching — and its reserve is freed (funds are neither
/// fabricated nor held in excess).
#[tokio::test]
async fn expired_order_does_not_fill_and_releases_reserve() {
    let clock = clock();
    let (sim, mut rx, _w) = sim_with_clock(clock.clone(), dec!(10), 0);
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "5")]));
    // GTD of 10 shares at 0.50 (post_only → reserve 5 USDC, no fee).
    let mut req = limit(dec!(0.50), dec!(10), TimeInForce::Gtd);
    req.clear_at = T0_MS / 1000 + 10;
    let oid = sim.submit(req).await.order_id.expect("GTD accepted");
    // The reserve leaves 5 free: a second GTD order of 20 shares (10 USDC) does not fit.
    let mut blocked = limit(dec!(0.50), dec!(20), TimeInForce::Gtd);
    blocked.post_only = false;
    blocked.clear_at = T0_MS / 1000 + 3_600;
    assert!(
        sim.submit(blocked.clone()).await.rejected,
        "the reserve of the live GTD blocks the second order"
    );

    // We advance the clock past clear_at + 60s and a book arrives that
    // would fill it entirely if it were still alive.
    clock.store(T0_MS + 71_000, Ordering::Relaxed);
    sim.tick_book(&book_levels(vec![lv("0.50", "50")], vec![lv("0.50", "5")]));
    assert!(rx.try_recv().is_err(), "an expired order cannot fill");
    assert!(
        !sim.live_gtd_order_ids().await.unwrap().contains(&oid),
        "expired = off the book"
    );
    match sim.order_status(&oid).await {
        Some(RouterOrderStatus::Gone { matched }) => {
            assert_eq!(matched, Decimal::ZERO, "expired without executing")
        }
        other => panic!("the expired order must be reported as Gone{{matched:0}}: {other:?}"),
    }

    // The reserve was fully released: the 20-share GTD now fits.
    assert!(
        sim.submit(blocked).await.order_id.is_some(),
        "expiration must release the full reserve"
    );
}

// ── Research profile (κ, participation, stale, gap, inventory) ────────

/// Conservative hypothesis κ=0 (`cancel_attr`): a shrink without a cross does NOT drain
/// the queue — the whole shrink is assumed to be behind our position. The same
/// scenario that with κ=1 filled at the next cross now requires the flow to
/// exceed the intact queue; and the participation cap (0.25 of the flow
/// crossed at the tick) caps the resulting fill.
#[tokio::test]
async fn conservative_hypothesis_shrink_does_not_drain_queue_and_caps_participation() {
    let (sim, mut rx, _w) = sim_with_clock(clock(), dec!(1000), 0);
    sim.apply_research_profile(ResearchSimConfig::conservative());
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "5")]));
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
        .await
        .order_id
        .expect("GTD accepted");

    // bids@0.50: 5→2 without a cross — κ=0: the shrink is attributed to cancels BEHIND.
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "2")]));
    // A cross of 3 is not enough for us: the queue stays at 5 (nominal would have
    // drained it to 2 and this cross would have executed us).
    sim.tick_book(&book_levels(vec![lv("0.50", "3")], vec![lv("0.50", "2")]));
    assert!(rx.try_recv().is_err(), "κ=0: the shrink did not drain the queue");

    // Cross of 8: remaining queue 2 → excess 6, capped at participation
    // 0.25 × 8 = 2.
    sim.tick_book(&book_levels(vec![lv("0.50", "8")], vec![lv("0.50", "2")]));
    let f = rx
        .try_recv()
        .expect("the excess over the intact queue executes");
    assert_eq!(f.size, dec!(2), "participation 0.25 × cross 8 = 2");
}

/// A PERSISTENT quote outside consensus is not free liquidity: the taker fill
/// against it is marked `Stressed` (adverse selection).
/// Persistence =
/// displayed ≥ `quote_stale_ms` with unchanged size; outside consensus =
/// deviates from the complement to 1 of the best crossed level of the sibling token.
#[tokio::test]
async fn fill_against_stale_quote_outside_consensus_marks_stressed() {
    let clock = clock();
    let (sim, mut rx, _w) = sim_with_clock(clock.clone(), dec!(1000), 0);
    let mut profile = ResearchSimConfig::nominal();
    profile.quote_stale_ms = 10_000;
    profile.consensus_margin = dec!(0.02);
    sim.apply_research_profile(profile);

    // ask yes 0.30 vs crossed consensus 1 − bid_no 0.49 = 0.51 → deviation 0.21.
    let book = book_levels(vec![lv("0.30", "5")], vec![lv("0.20", "5")]);
    sim.tick_book(&book);
    // Re-tick of the SAME content 15s later: the quote remains identical —
    // accumulated persistence (the non-fresh footprint does not reset its age).
    clock.store(T0_MS + 15_000, Ordering::Relaxed);
    sim.tick_book(&book);

    let res = sim
        .submit(limit(dec!(0.30), dec!(5), TimeInForce::Fok))
        .await;
    assert!(!res.rejected);
    let _ = rx.try_recv().expect("taker fill against the stale quote");
    let exec = sim
        .executions()
        .last()
        .expect("execution recorded")
        .clone();
    assert!(
        exec.levels.iter().all(|l| l.stale),
        "the fills against the persistent off-consensus quote are marked"
    );
    assert_eq!(
        exec.confidence,
        SimConfidence::Stressed,
        "adverse selection: the fill against a stale quote is low confidence"
    );
}

/// Tape gap with `gap_abstain` (conservative): the post-gap tick neither fills
/// nor advances the queue — it resyncs the L2 delta base and degrades the
/// evidence of the order. A cross that survives the resync executes afterwards,
/// marked `Stressed` (episode invalidated, not invented).
#[tokio::test]
async fn tape_gap_abstains_and_degrades_evidence() {
    let clock = clock();
    let (sim, mut rx, _w) = sim_with_clock(clock.clone(), dec!(1000), 0);
    sim.apply_research_profile(ResearchSimConfig::conservative());
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "5")]));
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
        .await
        .order_id
        .expect("GTD accepted");

    // 61s jump > gap_ms (60s) → the post-gap tick ABSTAINS even though the book
    // crosses our level with ample volume.
    clock.store(T0_MS + 61_000, Ordering::Relaxed);
    sim.tick_book(&book_levels(vec![lv("0.50", "10")], vec![lv("0.50", "5")]));
    assert!(
        rx.try_recv().is_err(),
        "post-gap tick with abstention: the cross produces no fill"
    );

    // Next tick with no gap: the cross persists (different content →
    // fresh) and executes — but the order was left degraded by the gap.
    clock.store(T0_MS + 61_100, Ordering::Relaxed);
    sim.tick_book(&book_levels(vec![lv("0.50", "11")], vec![lv("0.50", "5")]));
    let f = rx.try_recv().expect("post-resync cross executes");
    // excess = 11 − queue 5 = 6, capped at participation 0.25 × 11 = 2.75.
    assert_eq!(f.size, dec!(2.75));
    let exec = sim
        .executions()
        .last()
        .expect("execution recorded")
        .clone();
    assert_eq!(
        exec.confidence,
        SimConfidence::Stressed,
        "the degraded evidence from the gap marks the fill"
    );
}

/// `track_inventory` (conservative): sells require simulated inventory —
/// shares are never fabricated. Buying accumulates; selling consumes; once
/// inventory is exhausted the sell is rejected.
#[tokio::test]
async fn track_inventory_requires_shares_to_sell() {
    let (sim, mut rx, _w) = sim_with_clock(clock(), dec!(1000), 0);
    sim.apply_research_profile(ResearchSimConfig::conservative());
    sim.tick_book(&book_levels(vec![lv("0.50", "10")], vec![lv("0.49", "10")]));

    // Selling without inventory → rejection: shares are not fabricated.
    let r = sim
        .submit(req_kind(OrderKind::Sell {
            size: Some(dec!(5)),
        }))
        .await;
    assert!(r.rejected, "without inventory there is no sell (track_inventory)");

    // Buy 2 (FOK; participation 0.25 × 10 = 2.5 covers it) → inventory 2.
    let ok = sim
        .submit(limit(dec!(0.50), dec!(2), TimeInForce::Fok))
        .await;
    assert!(!ok.rejected);
    let _ = rx.try_recv().expect("buy fill");
    assert_eq!(
        sim.simulated_inventory()
            .get("tok-yes")
            .copied()
            .unwrap_or_default(),
        dec!(2)
    );

    // Sell 2 → ok; sell 1 more → rejection (inventory exhausted).
    let ok = sim
        .submit(req_kind(OrderKind::Sell {
            size: Some(dec!(2)),
        }))
        .await;
    assert!(!ok.rejected, "with inventory the sell executes");
    let _ = rx.try_recv().expect("sell fill");
    let r = sim
        .submit(req_kind(OrderKind::Sell {
            size: Some(dec!(1)),
        }))
        .await;
    assert!(r.rejected, "inventory exhausted → rejection");
}

/// Each simulated fill keeps full traceability: causal evidence (which
/// book observation produced it), queue model hypothesis applied, and
/// costs kept separate (notional + fee = cash moved). The exec_id is unique per
/// execution and never equal to the order_id.
#[tokio::test]
async fn each_fill_carries_evidence_hypothesis_and_costs() {
    let (sim, mut rx, _w) = sim_with_clock(clock(), dec!(1000), 0);
    sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![lv("0.50", "5")]));

    // Resting first (bid@0.50, external queue 5).
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
        .await
        .order_id
        .expect("GTD accepted");
    // Cross of 8: 5 queue + 3 excess → fill 3 with evidence of a cross past the queue.
    sim.tick_book(&book_levels(vec![lv("0.50", "8")], vec![lv("0.50", "5")]));
    let _ = rx.try_recv().expect("resting fill from the queue excess");
    let exec = &sim.executions()[0];
    match &exec.evidence {
        SimEvidence::CrossBehindQueue {
            cross,
            queue_ahead,
            own_ahead,
        } => {
            assert_eq!(*cross, dec!(8));
            assert_eq!(*queue_ahead, dec!(5));
            assert_eq!(*own_ahead, Decimal::ZERO);
        }
        other => panic!("the resting order must register queue evidence: {other:?}"),
    }
    assert_eq!(exec.hypothesis, SimHypothesis::QueueExcess);
    assert_eq!(exec.confidence, SimConfidence::Normal);
    assert_eq!(exec.notional + exec.fee_usdc, exec.cash);
    assert_ne!(exec.exec_id, exec.order_id);

    // Taker afterwards: evidence DepthOnArrival + TakerDepth.
    sim.submit(limit(dec!(0.60), dec!(3), TimeInForce::Fok))
        .await;
    let _ = rx.try_recv().expect("fill taker");
    let exec = sim
        .executions()
        .last()
        .expect("execution recorded")
        .clone();
    assert!(
        matches!(exec.evidence, SimEvidence::DepthAtArrival { levels: 1 }),
        "the taker records the depth walked on arrival: {:?}",
        exec.evidence
    );
    assert_eq!(exec.hypothesis, SimHypothesis::TakerDepth);
    assert_eq!(exec.notional + exec.fee_usdc, exec.cash);
}

/// Multiple seeds: the `rng_seed` parameter of `with_deps` separates the
/// RNG sequences per instance — the SAME seed reproduces exactly the
/// same run (replay reproducibility by seed). With
/// fok_fill_rate=0.5 the draw decides whether to accept/reject each FOK; the seed
/// fixes the sequence.
#[tokio::test]
async fn multiple_seeds_are_reproducible() {
    async fn run(seed: u64) -> Vec<String> {
        let (fill_tx, mut rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(None);
        let clk: SimClock = Arc::new(|| T0_MS);
        let sim = SimulatedRouter::with_deps(
            0,
            0.5,
            0,
            Decimal::ZERO,
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(1000),
            wallet_tx,
            seed,
            clk,
        );
        sim.tick_book(&book_levels(
            vec![lv("0.50", "100")],
            vec![lv("0.49", "100")],
        ));
        for _ in 0..8 {
            let _ = sim
                .submit(limit(dec!(0.50), dec!(1), TimeInForce::Fok))
                .await;
        }
        let mut v = Vec::new();
        while let Ok(f) = rx.try_recv() {
            v.push(format!("{}@{}", f.size, f.price));
        }
        v
    }
    for s in [1u64, 7, 42] {
        assert_eq!(
            run(s).await,
            run(s).await,
            "seed {s} does not reproduce the same run"
        );
    }
}
