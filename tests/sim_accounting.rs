//! Reproducible costs, settlement and accounting.
//!
//! Independent fixtures per price/size/role/effective fee, with rates
//! cited to the official source and date:
//!
//! - `RATE_CRYPTO = 0.07`: crypto taker — official docs
//!   <https://docs.polymarket.com/trading/fees> (table "Crypto", consulted
//!   22-sep-2026) and `feeSchedule.rate` from the Gamma API (`crypto_fees_v2`,
//!   verified 18-jul-2026). Official formula `fee = C × rate ×
//!   p × (1−p)` per fill; makers NEVER pay fee.
//! - `RATE_FINANCE = 0.04`: Finance/Politics/Mentions/Tech category of the
//!   same official table (22-sep-2026); also `finance_prices_fees`
//!   (02-ago-2026).
//! - Rebates (maker 20%, taker tiers): NOT confirmed per execution — the
//!   conservative scenario excludes them (`rate ≥ 0` validated; never
//!   credits a rebate in these tests).
//! - Official rounding: "Fees are rounded to 5 decimal places. The smallest
//!   fee charged is 0.00001 USDC" (same page, "Fee Precision"). The
//!   venue does not publish the tie-break; the sim uses `Decimal::round_dp(5)`
//!   (MidpointNearestEven) — documented in `fair_value::fee_charged_5dp`.
//!
//! The books carry `parabolic_fee_rate` on purpose, DIFFERENT from the schedule
//! frozen in `fee_windows_*`: the schedule by validity must win.

use std::collections::HashSet;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{mpsc, watch};

use poly_bot::{
    coordination::BracketLedger,
    fair_value::{self, FeeWindowsError},
    accounting::fill_sink,
    market_feed::MarketRegistry,
    order::{
        router::{OrderKind, OrderRequest, OrderRouter, TimeInForce},
        sim_router::{SimClock, SimRole, SimulatedRouter},
    },
    research::contract::RulesWindow,
    risk::RiskGuard,
    state::{to_micros, StateStore},
    types::{BookSnapshot, Fill, FillSide, HalfBook, Level, Mode, StrategyId, TokenBook},
};

// ── Fixtures ─────────────────────────────────────────────────────────────────

const T0_MS: i64 = 1_700_000_000_000;
const CLEAR_AT: i64 = (T0_MS / 1000) + 7_200;

/// Official crypto taker rate (docs + Gamma `feeSchedule`, see header).
const RATE_CRYPTO: Decimal = dec!(0.07);
/// Official Finance/Politics taker rate (docs, see header).
const RATE_FINANCE: Decimal = dec!(0.04);

fn clock() -> Arc<AtomicI64> {
    Arc::new(AtomicI64::new(T0_MS))
}

fn lv(p: &str, s: &str) -> Level {
    Level {
        price: p.parse().unwrap(),
        size: s.parse().unwrap(),
    }
}

/// Book for "tok-yes" with an explicit `parabolic_fee_rate` (or None) — the cid is
/// "cid-acc", the key of the frozen fee schedule.
fn book_fee(asks: Vec<Level>, bids: Vec<Level>, rate: Option<Decimal>) -> BookSnapshot {
    BookSnapshot {
        market_condition_id: "cid-acc".into(),
        yes: TokenBook {
            token_id: "tok-yes".into(),
            bids: HalfBook { levels: bids },
            asks: HalfBook { levels: asks },
            ..Default::default()
        },
        no: TokenBook {
            token_id: "tok-no".into(),
            bids: HalfBook {
                levels: vec![lv("0.49", "100")],
            },
            asks: HalfBook {
                levels: vec![lv("0.50", "100")],
            },
            ..Default::default()
        },
        market_open_ts: T0_MS / 1000 - 3_600,
        clear_at: CLEAR_AT,
        neg_risk: false,
        slug: "test-acc".into(),
        reference_price: Some(dec!(50_000)),
        fee_rate_bps: None,
        parabolic_fee_rate: rate,
        tick_size: None,
        recv_instant: None,
    }
}

fn req(kind: OrderKind) -> OrderRequest {
    OrderRequest {
        market_cid: "cid-acc".into(),
        token_id: "tok-yes".into(),
        kind,
        neg_risk: false,
        strategy: StrategyId::Arb,
        clear_at: CLEAR_AT,
        signal: dec!(0.5),
        post_only: false,
        real_clear_at: None,
        local_order_id: None,
    }
}

fn limit(price: Decimal, size: Decimal, tif: TimeInForce) -> OrderRequest {
    req(OrderKind::Limit { price, size, tif })
}

/// Deterministic sim with configurable fee: slippage/fok/latency controlled
/// by parameter; injectable clock (AtomicI64) for the validity windows.
fn sim_cfg(
    clock: Arc<AtomicI64>,
    balance: Decimal,
    slippage_bps: u32,
    taker_fee_rate: Decimal,
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
        slippage_bps,
        1.0, // fok_fill_rate — liquidity is decided by the book, not the draw
        0,   // latency 0: immediate arrival
        taker_fee_rate,
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

fn window(from_ms: i64, to_ms: Option<i64>, rate: Option<Decimal>) -> RulesWindow {
    RulesWindow {
        valid_from_ms: from_ms,
        valid_to_ms: to_ms,
        parabolic_fee_rate: rate,
        fee_rate_bps: None,
        tick_size: None,
        min_order_size: None,
    }
}

async fn state_paper() -> (Arc<StateStore>, Arc<RiskGuard>, Arc<BracketLedger>) {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let risk = RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper);
    (state, risk, BracketLedger::new())
}

// ═══════════════════════════════════════════════════════════════════════════
// Taker fee by price (extremes 0.02/0.50/0.98), official formula
//          and separation of negotiated price / fee / impact in each execution.
// ═══════════════════════════════════════════════════════════════════════════

/// Taker buy at each extreme of the parabola: `fee = 0.07 × p × (1−p) × C`,
/// quantized to 5dp per fill (official precision rule). The breakdown keeps
/// the negotiated price (`notional`), the commission and the impact ALREADY realized
/// separated — with slippage 0 the impact is 0 and `cash = notional + fee`.
#[tokio::test]
async fn taker_fee_price_extremes_parabolic_formula() {
    for (price, shares) in [
        (dec!(0.02), dec!(10)),
        (dec!(0.50), dec!(10)),
        (dec!(0.98), dec!(10)),
    ] {
        let (sim, mut rx, wallet) = sim_cfg(clock(), dec!(1000), 0, RATE_CRYPTO);
        sim.tick_book(&book_fee(
            vec![lv(&price.to_string(), "100")],
            vec![lv("0.01", "100")],
            Some(RATE_CRYPTO),
        ));
        let r = sim.submit(limit(price, shares, TimeInForce::Fok)).await;
        assert!(!r.rejected, "FOK at {price} must fill");
        let _ = rx.try_recv().expect("fill taker");
        let exec = sim.executions().last().unwrap().clone();

        let expected_fee = (RATE_CRYPTO * price * (Decimal::ONE - price) * shares).round_dp(5);
        assert_eq!(
            exec.fee_usdc, expected_fee,
            "official fee `C×rate×p×(1−p)` at 5dp for p={price}"
        );
        assert_eq!(exec.notional, price * shares, "notional = negotiated price");
        assert_eq!(exec.cash, exec.notional + exec.fee_usdc);

        let d = exec.cost_breakdown();
        assert_eq!(d.traded_notional, exec.notional);
        assert_eq!(d.fee_usdc, exec.fee_usdc);
        assert_eq!(d.impact_usdc, Decimal::ZERO, "slippage 0 → no impact");
        assert_eq!(
            d.penalty_usdc,
            Decimal::ZERO,
            "penalties live in the reward, not in the execution"
        );
        assert_eq!(
            d.residual_usdc,
            Decimal::ZERO,
            "cash = notional + fee + impact"
        );
        assert!(!d.is_maker, "a crossing FOK is a taker");
        assert_eq!(exec.role, SimRole::Taker);

        let expected_wallet = dec!(1000) - exec.cash;
        assert_eq!(
            *wallet.borrow(),
            Some(expected_wallet),
            "the wallet moves by all-in cash (p={price})"
        );
    }
    // Official reference: 100 shares at p=0.50 crypto → $1.75 fee (table
    // "Fee Tables" in the docs — with our fixture: 0.07×0.25×100 = 1.75).
    assert_eq!(
        fair_value::fee_charged_5dp(RATE_CRYPTO, dec!(0.50), dec!(100)),
        dec!(1.75)
    );
}

/// Documented rounding: the fee is charged quantized to 5 decimals PER FILL
/// (`fee = C×rate×p×(1−p)`, official "Fee Precision" rule). Cases:
/// - 3 shares at 0.31: raw 0.044919 → 0.04492.
/// - 0.002 shares at 0.50: raw 0.000035 (midpoint) → 0.00004 with
///   MidpointNearestEven (documented tie-break — the venue does not publish it;
///   the maximum effect of the tie-break is 0.000005 USDC per fill).
/// - raw fee < 0.000005 → 0 (the docs say "anything smaller [than 0.00001]
///   rounds to zero": 4.9e-6 → 0).
#[tokio::test]
async fn fee_rounding_5dp_documented() {
    // Case at a real fill level (FAK over one level: fee quantized per segment).
    let (sim, mut rx, wallet) = sim_cfg(clock(), dec!(1000), 0, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.31", "100")],
        vec![lv("0.30", "100")],
        Some(RATE_CRYPTO),
    ));
    let r = sim
        .submit(limit(dec!(0.31), dec!(3), TimeInForce::Fak))
        .await;
    assert!(!r.rejected);
    let _ = rx.try_recv().expect("fill FAK");
    let exec = sim.executions().last().unwrap().clone();
    // 0.07 × 0.31 × 0.69 × 3 = 0.044919 → 0.04492 (5dp, per segment/fill).
    assert_eq!(exec.fee_usdc, dec!(0.04492));
    assert_eq!(exec.cash, dec!(0.93) + dec!(0.04492));
    assert_eq!(*wallet.borrow(), Some(dec!(1000) - exec.cash));

    // Pure cases via the shared helper (same quantization point
    // the sim uses — `fair_value::fee_charged_5dp`).
    assert_eq!(
        fair_value::fee_charged_5dp(RATE_CRYPTO, dec!(0.31), dec!(3)),
        dec!(0.04492)
    );
    assert_eq!(
        fair_value::fee_charged_5dp(RATE_CRYPTO, dec!(0.50), dec!(0.002)),
        dec!(0.00004),
        "midpoint 0.000035 → MidpointNearestEven → 0.00004"
    );
    assert_eq!(
        fair_value::fee_charged_5dp(RATE_CRYPTO, dec!(0.50), dec!(0.00028)),
        Decimal::ZERO,
        "raw fee 4.9e-6 < 0.000005 → 0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// parameters frozen per market and validity; an invalid schedule is
//          rejected; unconfirmed rebates (negative rate) are not admitted.
// ═══════════════════════════════════════════════════════════════════════════

/// The schedule frozen per market+validity decides the fill's rate — the
/// `parabolic_fee_rate` the book carries CANNOT override it (this is what makes
/// the run reproducible: the validity lives in the fixture, not in the tape).
#[tokio::test]
async fn fee_windows_freeze_rate_by_validity() {
    let clock = clock();
    let (sim, mut rx, _w) = sim_cfg(clock.clone(), dec!(1000), 0, dec!(0.99));
    // The book declares a DIFFERENT rate (0.10): it must be ignored while
    // a frozen window covers the instant of the fill.
    sim.tick_book(&book_fee(
        vec![lv("0.50", "100")],
        vec![lv("0.49", "100")],
        Some(dec!(0.10)),
    ));
    sim.set_fee_windows(
        "cid-acc",
        vec![
            window(T0_MS, Some(T0_MS + 60_000), Some(RATE_CRYPTO)),
            window(T0_MS + 60_000, None, Some(RATE_FINANCE)),
        ],
    )
    .unwrap();

    // t0: crypto 0.07 in force → fee = 0.07×0.25×10 = 0.175.
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Fok))
        .await;
    let _ = rx.try_recv().expect("fill t0");
    assert_eq!(sim.executions().last().unwrap().fee_usdc, dec!(0.175));

    // t1 = T0+61s: finance 0.04 in force → fee = 0.04×0.25×10 = 0.10.
    clock.store(T0_MS + 61_000, Ordering::Relaxed);
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Fok))
        .await;
    let _ = rx.try_recv().expect("fill t1");
    assert_eq!(sim.executions().last().unwrap().fee_usdc, dec!(0.1));
}

/// Window with rate `None` (the source did not report it for that segment) → the
/// book fallback is used; a 0 or a made-up rate is never invented.
#[tokio::test]
async fn window_without_rate_falls_back_to_book_fallback() {
    let (sim, mut rx, _w) = sim_cfg(clock(), dec!(1000), 0, dec!(0.99));
    sim.tick_book(&book_fee(
        vec![lv("0.50", "100")],
        vec![lv("0.49", "100")],
        Some(RATE_FINANCE), // the market's real rate from the tape
    ));
    sim.set_fee_windows(
        "cid-acc",
        vec![window(T0_MS - 1, None, None)], // covers now, no rate
    )
    .unwrap();
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Fok))
        .await;
    let _ = rx.try_recv().expect("fill");
    assert_eq!(
        sim.executions().last().unwrap().fee_usdc,
        (RATE_FINANCE * dec!(0.25) * dec!(10)).round_dp(5),
        "no rate in the window in force → fallback = book rate"
    );
}

/// The schedule is validated when frozen: ordering, no overlap, no empty window and
/// rates in [0,1]. A NEGATIVE rate would be a rebate — not confirmed by the
/// venue, excluded from the conservative scenario → rejected.
#[tokio::test]
async fn invalid_fee_windows_are_rejected() {
    let (sim, _rx, _w) = sim_cfg(clock(), dec!(1000), 0, RATE_CRYPTO);

    // Overlapping: the second window starts before the first one closes.
    let overlapping = vec![
        window(T0_MS, Some(T0_MS + 60_000), Some(RATE_CRYPTO)),
        window(T0_MS + 30_000, None, Some(RATE_FINANCE)),
    ];
    assert_eq!(
        fair_value::validate_fee_windows(&overlapping),
        Err(FeeWindowsError::Overlapped { idx: 1 })
    );
    assert!(sim.set_fee_windows("cid-acc", overlapping).is_err());

    // Out of order by valid_from.
    let disordered = vec![
        window(T0_MS + 60_000, None, Some(RATE_FINANCE)),
        window(T0_MS, Some(T0_MS + 60_000), Some(RATE_CRYPTO)),
    ];
    assert_eq!(
        fair_value::validate_fee_windows(&disordered),
        Err(FeeWindowsError::Unordered { idx: 1 })
    );

    // Inverted: valid_to ≤ valid_from.
    let inverted = vec![window(T0_MS, Some(T0_MS), Some(RATE_CRYPTO))];
    assert_eq!(
        fair_value::validate_fee_windows(&inverted),
        Err(FeeWindowsError::InvertedWindow { idx: 0 })
    );

    // Unconfirmed rebate → negative rate → rejected (conservative scenario).
    let rebate = vec![window(T0_MS, None, Some(dec!(-0.01)))];
    assert_eq!(
        fair_value::validate_fee_windows(&rebate),
        Err(FeeWindowsError::InvalidRate { idx: 0 })
    );
    assert!(sim.set_fee_windows("cid-acc", rebate).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Maker/taker role by the REAL cross: a maker never pays a fee (official
//          rule); a resting post_only order fills as maker.
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn maker_post_only_role_does_not_pay_fee() {
    let (sim, mut rx, wallet) = sim_cfg(clock(), dec!(1000), 0, RATE_CRYPTO);
    // Resting bid@0.45 post_only (external queue 0: no bids at 0.45).
    sim.tick_book(&book_fee(
        vec![lv("0.60", "100")],
        vec![lv("0.44", "5")],
        Some(RATE_CRYPTO),
    ));
    let mut gtd = limit(dec!(0.45), dec!(10), TimeInForce::Gtd);
    gtd.post_only = true;
    let order_id = sim.submit(gtd).await.order_id.expect("GTD accepted");

    // Seller flow crosses up to our bid: asks at 0.45 for 8 shares →
    // cross 8 > queue 0 → maker fill of 8 at our price.
    sim.tick_book(&book_fee(
        vec![lv("0.45", "8")],
        vec![lv("0.44", "5")],
        Some(RATE_CRYPTO),
    ));
    let fill = rx.try_recv().expect("maker fill of the resting order");
    assert_eq!(fill.size, dec!(8));
    let exec = sim.executions().last().unwrap().clone();
    assert_eq!(exec.order_id, order_id);
    assert_eq!(
        exec.fee_usdc,
        Decimal::ZERO,
        "genuine maker (post_only resting): fee 0 — official rule"
    );
    assert_eq!(exec.role, SimRole::Maker);
    assert!(exec.cost_breakdown().is_maker);
    assert_eq!(exec.cash, dec!(0.45) * dec!(8), "cash = notional only");
    let _ = wallet.borrow();
}

/// The marketable part of a GTD crosses on submit → taker, with fee; the
/// resting post_only later fills as maker without fee. The role is decided by what
/// actually crossed in each fill, not by the order's flag.
#[tokio::test]
async fn role_by_real_cross_not_by_flag() {
    let (sim, mut rx, _w) = sim_cfg(clock(), dec!(1000), 0, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.50", "4")],
        vec![lv("0.49", "100")],
        Some(RATE_CRYPTO),
    ));
    // Marketable GTD (limit 0.50 = best_ask): the 4 at the level execute right away
    // as taker; the remainder (6) rests.
    sim.submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
        .await;
    let _ = rx.try_recv().expect("taker fill on submit");
    let exec = sim.executions().last().unwrap().clone();
    assert_eq!(exec.role, SimRole::Taker, "the part that crossed pays taker");
    assert_eq!(
        exec.fee_usdc,
        (RATE_CRYPTO * dec!(0.50) * dec!(0.50) * dec!(4)).round_dp(5)
    );

    // New book with seller flow over our 0.50 bid (queue: bids
    // displayed at 0.50 on entry = 0 in this fixture → the remainder fills).
    sim.tick_book(&book_fee(
        vec![lv("0.50", "6")],
        vec![lv("0.49", "100")],
        Some(RATE_CRYPTO),
    ));
    let _ = rx.try_recv().expect("fill of the resting order later");
    let exec = sim.executions().last().unwrap().clone();
    // The non-post_only buy GTD fills as taker in this sim
    // (live, that order would cross): it pays fee even though it came from the book.
    assert_eq!(exec.role, SimRole::Taker);
    assert!(exec.fee_usdc > Decimal::ZERO);
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-level partial fills: each segment carries negotiated price,
//          quantized fee and its own impact; the sell deducts fee+impact.
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn partial_fak_breaks_down_each_level() {
    let (sim, mut rx, _w) = sim_cfg(clock(), dec!(1000), 0, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.50", "4"), lv("0.55", "4")],
        vec![lv("0.49", "100")],
        Some(RATE_CRYPTO),
    ));
    // FAK of 10 with depth for 8 → real partial, 2 levels.
    let r = sim
        .submit(limit(dec!(0.55), dec!(10), TimeInForce::Fak))
        .await;
    assert!(!r.rejected);
    let fill = rx.try_recv().expect("partial FAK fill");
    assert_eq!(fill.size, dec!(8), "FAK keeps the partial fill");
    let exec = sim.executions().last().unwrap().clone();
    assert_eq!(exec.shares, dec!(8));
    assert_eq!(exec.levels.len(), 2);

    for (level, p) in exec.levels.iter().zip([dec!(0.50), dec!(0.55)]) {
        assert_eq!(level.price, p, "negotiated price of the level (book)");
        assert_eq!(level.size, dec!(4));
        assert_eq!(
            level.fee_usdc,
            (RATE_CRYPTO * p * (Decimal::ONE - p) * dec!(4)).round_dp(5)
        );
        assert_eq!(level.impact_usdc, Decimal::ZERO, "slippage 0");
    }
    let d = exec.cost_breakdown();
    let fee_sum: Decimal = exec.levels.iter().map(|l| l.fee_usdc).sum();
    assert_eq!(exec.fee_usdc, fee_sum);
    assert_eq!(exec.cash, exec.notional + fee_sum);
    assert_eq!(d.residual_usdc, Decimal::ZERO);

    // Resume: new book with more depth → the next FAK fills what
    // is missing (second execution, different exec_id).
    sim.tick_book(&book_fee(
        vec![lv("0.50", "4"), lv("0.55", "6")],
        vec![lv("0.49", "100")],
        Some(RATE_CRYPTO),
    ));
    sim.submit(limit(dec!(0.55), dec!(10), TimeInForce::Fak))
        .await;
    let _ = rx.try_recv().expect("second FAK fill");
    let execs = sim.executions();
    assert_ne!(
        execs[0].exec_id, execs[1].exec_id,
        "exec_id unique per execution"
    );
    assert!(
        sim.accounting().is_consistent(),
        "conservation after partial fills"
    );
}

/// Taker sell with slippage: `cash = notional − fee − impact` and the impact
/// (spread/slippage ALREADY realized) is kept separate from the fee — never charged
/// twice nor merged into the price.
#[tokio::test]
async fn taker_sell_breaks_down_fee_and_impact_separately() {
    // slippage 25bps: slipped = bid / 1.0025.
    let (sim, mut rx, wallet) = sim_cfg(clock(), dec!(1000), 25, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.60", "100")],
        vec![lv("0.48", "10")],
        Some(RATE_CRYPTO),
    ));
    let before = *wallet.borrow();
    let r = sim
        .submit(req(OrderKind::Sell {
            size: Some(dec!(5)),
        }))
        .await;
    assert!(!r.rejected);
    let _ = rx.try_recv().expect("fill sell");
    let exec = sim.executions().last().unwrap().clone();
    assert_eq!(exec.side, FillSide::Sell);

    let slipped = dec!(0.48) / dec!(1.0025);
    let impact = (dec!(0.48) - slipped) * dec!(5);
    let fee = (RATE_CRYPTO
        * slipped.min(Decimal::ONE)
        * (Decimal::ONE - slipped.min(Decimal::ONE))
        * dec!(5))
    .round_dp(5);
    let d = exec.cost_breakdown();
    assert_eq!(d.traded_notional, dec!(0.48) * dec!(5));
    assert_eq!(d.fee_usdc, fee, "fee over the already slipped price, 5dp");
    assert_eq!(d.impact_usdc, impact, "impact = realized slippage");
    // Equality is checked at ledger resolution — the cash
    // moved by the wallet is `to_micros(notional − fee − impact)`, not the
    // full-precision value (the sub-micro remainder does not exist in the DB).
    assert_eq!(
        exec.cash,
        poly_bot::state::from_micros(to_micros(d.traded_notional - d.fee_usdc - d.impact_usdc)),
        "sell: cash = notional − fee − impact (once, to the micro)"
    );
    assert_eq!(d.residual_usdc, Decimal::ZERO);
    assert_eq!(*wallet.borrow(), before.map(|b| b + exec.cash));
    assert!(sim.accounting().is_consistent());
}

// ═══════════════════════════════════════════════════════════════════════════
// Micro-level conservation: wallet + reserves + inventory match the
//          execution log and the redemptions; the released reserve uses the
//          stored unit (not a recalculated cost).
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn micro_box_reserves_inventory_conservation() {
    let (sim, mut rx, wallet) = sim_cfg(clock(), dec!(100), 0, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.60", "100")],
        vec![lv("0.44", "5"), lv("0.49", "10")],
        Some(RATE_CRYPTO),
    ));

    // 1) FOK taker 10@0.60 → cash out 6 + fee 0.168 = 6.168.
    sim.submit(limit(dec!(0.60), dec!(10), TimeInForce::Fok))
        .await;
    let _ = rx.try_recv().unwrap();

    // 2) GTD post_only 20@0.45 → reserve 9.0 (maker: unit without fee).
    let mut gtd = limit(dec!(0.45), dec!(20), TimeInForce::Gtd);
    gtd.post_only = true;
    let oid = sim.submit(gtd).await.order_id.unwrap();
    {
        let c = sim.accounting();
        assert_eq!(c.reserved_buy, dec!(9));
        assert_eq!(c.expected_reserved_buy, dec!(9));
        assert!(c.is_consistent());
    }

    // 3) Seller flow 8@0.45 → maker fill 8; reserve left 0.45×12 = 5.4.
    sim.tick_book(&book_fee(
        vec![lv("0.45", "8")],
        vec![lv("0.44", "5"), lv("0.49", "10")],
        Some(RATE_CRYPTO),
    ));
    let _ = rx.try_recv().unwrap();
    {
        let c = sim.accounting();
        assert_eq!(c.reserved_buy, dec!(5.4));
        assert!(c.is_consistent());
    }

    // 4) Cancel: releases the remainder by the STORED unit → reserve at 0.
    sim.cancel("cid-acc", &oid).await.unwrap();
    {
        let c = sim.accounting();
        assert_eq!(c.reserved_buy, Decimal::ZERO, "the reserve goes back to 0");
        assert_eq!(c.expected_reserved_buy, Decimal::ZERO);
        assert!(c.is_consistent());
    }

    // 5) Taker sell 4 against bid 0.49 → cash in 1.96 − fee(0.06997) = 1.89003.
    sim.submit(req(OrderKind::Sell {
        size: Some(dec!(4)),
    }))
    .await;
    let _ = rx.try_recv().unwrap();

    // 6) Redemption of the remaining inventory (18−4 = 14 winning shares at $1).
    assert!(sim
        .credit_redemption("red-1", "tok-yes", dec!(14), dec!(14))
        .unwrap());

    let c = sim.accounting();
    assert!(c.is_consistent(), "drift: {c:?}");
    let expected = dec!(100) - dec!(6.168) - dec!(3.6) + dec!(1.89003) + dec!(14);
    assert_eq!(*wallet.borrow(), Some(expected));
    assert_eq!(c.balance, expected);
    assert_eq!(
        c.expected_balance, expected,
        "wallet reconstructed from the log"
    );
    assert_eq!(
        c.inventory_state.get("tok-yes").copied().unwrap_or_default(),
        Decimal::ZERO,
        "inventory closed: 10+8−4−14"
    );
    assert_eq!(c.redemptions_usdc, dec!(14));
    assert_eq!(c.initial_balance, dec!(100));
}

/// Reserve released per stored unit: if the rate in force changes between
/// submit and fill, the reserve is released by the stored `reserved_unit` — the
/// difference from the real charge comes from free cash, never "adjusts" the reserve
/// below zero nor creates cash.
#[tokio::test]
async fn reserve_releases_per_stored_unit() {
    let clock = clock();
    let (sim, mut rx, _w) = sim_cfg(clock.clone(), dec!(1000), 0, dec!(0.99));
    sim.tick_book(&book_fee(
        vec![lv("0.60", "100")],
        vec![lv("0.44", "5")],
        Some(dec!(0.10)),
    ));
    // Schedule: rate 0.07 until t1, 0.00 afterwards (the market went fee-free,
    // geopolitical scenario documented in the official table).
    sim.set_fee_windows(
        "cid-acc",
        vec![
            window(T0_MS, Some(T0_MS + 60_000), Some(RATE_CRYPTO)),
            window(T0_MS + 60_000, None, Some(Decimal::ZERO)),
        ],
    )
    .unwrap();

    // GTD no-post_only 10@0.45: reserve = 0.45 + fee(0.07) = 0.467325/share.
    let gtd = limit(dec!(0.45), dec!(10), TimeInForce::Gtd);
    let oid = sim.submit(gtd).await.order_id.unwrap();
    let fee_share_submit = RATE_CRYPTO * dec!(0.45) * dec!(0.55);
    let reserved = (dec!(0.45) + fee_share_submit) * dec!(10);
    assert_eq!(sim.accounting().reserved_buy, reserved);

    // The rate in force drops to 0 before the fill: the real charge is only notional,
    // but the RELEASED reserve is the stored unit (0.467325/share).
    clock.store(T0_MS + 61_000, Ordering::Relaxed);
    sim.tick_book(&book_fee(
        vec![lv("0.45", "4")],
        vec![lv("0.44", "5")],
        Some(dec!(0.10)),
    ));
    let _ = rx.try_recv().unwrap();
    let c = sim.accounting();
    let execs = sim.executions();
    let exec = execs.last().unwrap();
    assert_eq!(exec.fee_usdc, Decimal::ZERO, "rate in force 0 at fill");
    assert_eq!(
        c.reserved_buy,
        reserved - (dec!(0.45) + fee_share_submit) * dec!(4),
        "the reserve is released by the stored unit, not by the real charge"
    );
    assert!(
        c.is_consistent(),
        "conservation holds internally even with a changed rate"
    );
    let _ = oid;
}

// ═══════════════════════════════════════════════════════════════════════════
// Idempotency: redemption, resolution and duplicate execution do not
//          move cash or the ledger twice.
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn redemption_idempotent_by_redeem_id() {
    let (sim, mut rx, wallet) = sim_cfg(clock(), dec!(100), 0, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.60", "100")],
        vec![lv("0.59", "100")],
        Some(RATE_CRYPTO),
    ));
    sim.submit(limit(dec!(0.60), dec!(10), TimeInForce::Fok))
        .await;
    let _ = rx.try_recv().unwrap();
    let after_buy = *wallet.borrow();

    // Losing side: payout 0 — debits inventory, does not move cash.
    assert!(sim
        .credit_redemption("red-x", "tok-no", dec!(2), Decimal::ZERO)
        .unwrap());
    assert_eq!(*wallet.borrow(), after_buy, "payout 0 does not move cash");

    // Winning side: +10 payout, −10 inventory. Duplicate → no-op.
    assert!(sim
        .credit_redemption("red-1", "tok-yes", dec!(10), dec!(10))
        .unwrap());
    let after_redeem = *wallet.borrow();
    assert_eq!(after_redeem, after_buy.map(|b| b + dec!(10)));
    assert!(
        !sim.credit_redemption("red-1", "tok-yes", dec!(10), dec!(10))
            .unwrap(),
        "same redeem_id → no-op (idempotent resolution)"
    );
    assert_eq!(
        *wallet.borrow(),
        after_redeem,
        "the duplicate does not credit twice"
    );
    assert!(sim.accounting().is_consistent());
}

/// Idempotent resolution at the durable level: `apply_settlement` twice only
/// applies once (existing guard) and the aggregate ledger does not move.
#[tokio::test]
async fn duplicate_resolution_does_not_duplicate_ledger() {
    let (state, risk, ledger) = state_paper().await;
    let mut f = fill_paper("tok-set", FillSide::Buy, "0.60", "10");
    f.exec_id = Some("exec-set-1".into());
    fill_sink::process(&state, &risk, &ledger, f).await.unwrap();
    let pos = state.get_position("tok-set", "arb").await.unwrap().unwrap();

    assert!(state
        .apply_settlement(
            &pos,
            true,
            "YES",
            dec!(10),
            dec!(4),
            StrategyId::Arb,
            Some("paper-auto")
        )
        .await
        .unwrap());
    let flows1 = state.economic_flows().await.unwrap();
    assert!(
        !state
            .apply_settlement(
                &pos,
                true,
                "YES",
                dec!(10),
                dec!(4),
                StrategyId::Arb,
                Some("paper-auto")
            )
            .await
            .unwrap(),
        "second resolution is a no-op"
    );
    let flows2 = state.economic_flows().await.unwrap();
    assert_eq!(flows1, flows2, "the ledger does not move on the duplicate");
}

/// A duplicate execution (same exec_id) does not duplicate fill or flows — the
/// durable dedup key already existed; here it is verified over the aggregate.
#[tokio::test]
async fn duplicate_execution_does_not_duplicate_flows() {
    let (state, risk, ledger) = state_paper().await;
    let f = fill_paper("tok-dup", FillSide::Buy, "0.50", "4");
    assert!(fill_sink::process(&state, &risk, &ledger, f.clone())
        .await
        .unwrap());
    let flows1 = state.economic_flows().await.unwrap();
    assert!(
        !fill_sink::process(&state, &risk, &ledger, f).await.unwrap(),
        "same exec_id → duplicate"
    );
    let flows2 = state.economic_flows().await.unwrap();
    assert_eq!(flows1, flows2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Identical reopening after settlement: the new position does not
//          revive the closed metadata (outcome/redeemed) nor the old cost.
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn reopening_after_settlement_does_not_revive_metadata() {
    let (state, risk, ledger) = state_paper().await;
    let mut f1 = fill_paper("tok-re", FillSide::Buy, "0.60", "10");
    f1.exec_id = Some("exec-re-1".into());
    fill_sink::process(&state, &risk, &ledger, f1)
        .await
        .unwrap();
    let pos = state.get_position("tok-re", "arb").await.unwrap().unwrap();
    state
        .apply_settlement(
            &pos,
            true,
            "YES",
            dec!(10),
            dec!(4),
            StrategyId::Arb,
            Some("paper-auto"),
        )
        .await
        .unwrap();

    // Reopening on the same token/strategy: NEW position, without the
    // metadata of the closed one — size/cost only from the new fill.
    let mut f2 = fill_paper("tok-re", FillSide::Buy, "0.30", "5");
    f2.exec_id = Some("exec-re-2".into());
    fill_sink::process(&state, &risk, &ledger, f2)
        .await
        .unwrap();
    let p2 = state.get_position("tok-re", "arb").await.unwrap().unwrap();
    assert_eq!(p2.size, dec!(5));
    assert_eq!(p2.cost_basis, dec!(1.5), "cost only from the new fill");
    assert_eq!(
        p2.outcome_won, None,
        "does not revive the outcome of the earlier cycle"
    );
    assert!(!p2.redeemed, "no revive redeemed");
    assert!(p2.redeem_tx.is_none(), "no revive redeem_tx");
}

// ═══════════════════════════════════════════════════════════════════════════
// Acceptance: the economic return reconciles with the durable ledger
//          (BUY/SELL/REDEEM flows + open cost), not only with the reward.
// ═══════════════════════════════════════════════════════════════════════════

fn fill_paper(token: &str, side: FillSide, price: &str, size: &str) -> Fill {
    Fill {
        hedge_id: None,
        exec_id: Some(format!("exec-{token}-{}", size)),
        local_order_id: None,
        market_cid: "cid-acc".into(),
        token_id: token.into(),
        side,
        price: price.parse().unwrap(),
        size: size.parse().unwrap(),
        order_id: Some(format!("ord-{token}")),
        ts: chrono::Utc::now(),
        mode: Mode::Paper,
        neg_risk: false,
        strategy: StrategyId::Arb,
        clear_at: CLEAR_AT,
        outcome_label: Some("YES".into()),
    }
}

/// Complete paper cycle: taker buy + maker buy + partial sell +
/// winning settlement. The sim's final wallet must reconcile EXACTLY with
/// the durable ledger flows (micros), and the open cost must conserve
/// `buys − withdrawn − open = 0`.
#[tokio::test]
async fn economic_return_reconciles_with_ledger() {
    let (state, risk, ledger) = state_paper().await;
    let (sim, mut rx, wallet) = sim_cfg(clock(), dec!(100), 0, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.60", "100")],
        vec![lv("0.44", "5"), lv("0.49", "10")],
        Some(RATE_CRYPTO),
    ));

    // Taker buy 10@0.60 and maker buy 8@0.45 (post_only resting).
    sim.submit(limit(dec!(0.60), dec!(10), TimeInForce::Fok))
        .await;
    let mut gtd = limit(dec!(0.45), dec!(8), TimeInForce::Gtd);
    gtd.post_only = true;
    sim.submit(gtd).await;
    sim.tick_book(&book_fee(
        vec![lv("0.45", "8")],
        vec![lv("0.44", "5"), lv("0.49", "10")],
        Some(RATE_CRYPTO),
    ));
    // Partial taker sell 4@0.49.
    sim.submit(req(OrderKind::Sell {
        size: Some(dec!(4)),
    }))
    .await;

    // Fills → durable ledger (same boundary as production).
    let mut n_fills = 0;
    while let Ok(f) = rx.try_recv() {
        assert!(fill_sink::process(&state, &risk, &ledger, f).await.unwrap());
        n_fills += 1;
    }
    assert_eq!(n_fills, 3, "taker + maker + sell");

    // Settlement: YES wins → payout = live shares (14) at $1. The durable entry
    // is made by apply_settlement; the paper CASH is credited by the sim via
    // an idempotent redeem_id (the same resolution).
    let pos = state.get_position("tok-yes", "arb").await.unwrap().unwrap();
    assert_eq!(pos.size, dec!(14));
    let payout = pos.size;
    assert!(state
        .apply_settlement(
            &pos,
            true,
            "YES",
            payout,
            payout - pos.cost_basis,
            StrategyId::Arb,
            Some("paper-auto")
        )
        .await
        .unwrap());
    assert!(sim
        .credit_redemption("settle:cid-acc:tok-yes", "tok-yes", pos.size, payout)
        .unwrap());

    // Ledger ↔ wallet reconciliation: return by cash == net ledger flow
    // ledger; realized pnl == return + open cost (0 here).
    let flows = state.economic_flows().await.unwrap();
    assert_eq!(flows.n_buys, 2);
    assert_eq!(flows.n_sells, 1);
    assert_eq!(flows.n_redeem, 1);
    assert_eq!(flows.open_cost_drift(), Decimal::ZERO);

    // The ledger's buys are the wallet movements of the sim — to the micro PER
    // FILL (the ledger records `to_micros` per transaction; comparing the
    // aggregate `to_micros(Σ cashᵢ)` would hide cross rounding).
    let buys_micros: i64 = sim
        .executions()
        .iter()
        .filter(|e| e.side == FillSide::Buy)
        .map(|e| to_micros(e.cash))
        .sum();
    assert_eq!(
        flows.buys_usdc,
        poly_bot::state::from_micros(buys_micros),
        "ledger buys = Σ to_micros(cashᵢ) per sim fill"
    );

    let c = fill_sink::reconcile_economic_return(
        &flows,
        dec!(100),     // initial wallet
        Decimal::ZERO, // contributions
        wallet.borrow().expect("wallet published"),
    );
    assert_eq!(c.cash_drift, Decimal::ZERO, "cash == net ledger flow");
    assert_eq!(c.pnl_drift, Decimal::ZERO, "pnl == return + open cost");
    assert!(c.reconciled());
    assert!(sim.accounting().is_consistent());
}

/// Fractional fill with slippage — `slipped×size` has a sub-micro
/// remainder (0.997 × 1.0025 × 3.5). Without quantizing to the ledger's
/// resolution, `exec.cash` carried that remainder and reconciliation gave
/// `cash_drift ≠ 0` in a healthy run. Now the wallet moves EXACTLY
/// what `to_micros` records: `cash_drift == 0` and `residual_usdc == 0` at the
/// micro granularity.
#[tokio::test]
async fn cash_at_micro_leaves_no_sub_micro_drift() {
    let (state, risk, ledger) = state_paper().await;
    // 25 bps of slippage: slipped = 0.997 × 1.0025 = 0.9994925.
    let (sim, mut rx, wallet) = sim_cfg(clock(), dec!(100), 25, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.997", "3.5")],
        vec![lv("0.99", "100")],
        Some(RATE_CRYPTO),
    ));
    // FAK size 7 against a 3.5 level → fractional partial fill.
    sim.submit(limit(dec!(1.00), dec!(7), TimeInForce::Fak))
        .await;
    let mut n = 0;
    while let Ok(f) = rx.try_recv() {
        fill_sink::process(&state, &risk, &ledger, f).await.unwrap();
        n += 1;
    }
    assert_eq!(n, 1, "one partial fill");

    let execs = sim.executions();
    let exec = execs.last().unwrap();
    assert_eq!(exec.shares, dec!(3.5));
    assert!(
        exec.levels[0].impact_usdc > Decimal::ZERO,
        "realized slippage"
    );
    // The cash is exact to the micro — what the ledger records.
    assert_eq!(
        exec.cash,
        poly_bot::state::from_micros(to_micros(exec.cash)),
        "exec.cash quantized to the ledger resolution"
    );
    assert_eq!(
        exec.cost_breakdown().residual_usdc,
        Decimal::ZERO,
        "residual at micro resolution"
    );

    let flows = state.economic_flows().await.unwrap();
    let accounting = sim.accounting();
    let c = fill_sink::reconcile_economic_return(
        &flows,
        accounting.initial_balance,
        accounting.contributions_usdc,
        wallet.borrow().expect("wallet published"),
    );
    assert_eq!(
        c.cash_drift,
        Decimal::ZERO,
        "without a sub-micro remainder, reconciliation closes exactly"
    );
    assert!(c.reconciled());
    assert!(accounting.is_consistent());
}

/// A second `credit_redemption` with the SAME id but a divergent payload
/// does not credit twice nor panic — it is ignored (and logged).
#[tokio::test]
async fn duplicate_redemption_with_divergent_payload_does_not_duplicate() {
    let (sim, _rx, wallet) = sim_cfg(clock(), dec!(100), 0, RATE_CRYPTO);
    assert!(sim
        .credit_redemption("red-div", "tok-yes", dec!(10), dec!(10))
        .unwrap());
    let after = *wallet.borrow();
    // Same redeem_id, different payload → no-op (warn in logs).
    assert!(!sim
        .credit_redemption("red-div", "tok-yes", dec!(99), dec!(99))
        .unwrap());
    assert_eq!(*wallet.borrow(), after, "the divergent one does not move cash");
    assert!(sim.accounting().is_consistent());
}

/// Without fills there are no flows: the aggregate returns zeros and the reconciliation of
/// an untouched portfolio trivially balances.
#[tokio::test]
async fn empty_flows_balance() {
    let (state, _r, _l) = state_paper().await;
    let f = state.economic_flows().await.unwrap();
    assert_eq!(f.net_cash_flow(), Decimal::ZERO);
    assert_eq!(f.open_cost_drift(), Decimal::ZERO);
    let c = fill_sink::reconcile_economic_return(&f, dec!(50), Decimal::ZERO, dec!(50));
    assert!(c.reconciled());
}

/// The exec_ids observed by the ledger must be unique per execution — the
/// durable dedup depends on it (partials of the same order share the
/// order_id, not exec_id).
#[tokio::test]
async fn unique_exec_ids_in_partials() {
    let (sim, mut rx, _w) = sim_cfg(clock(), dec!(1000), 0, RATE_CRYPTO);
    sim.tick_book(&book_fee(
        vec![lv("0.50", "4"), lv("0.55", "4")],
        vec![lv("0.49", "100")],
        Some(RATE_CRYPTO),
    ));
    sim.submit(limit(dec!(0.55), dec!(10), TimeInForce::Fak))
        .await;
    let _ = rx.try_recv().unwrap();
    sim.tick_book(&book_fee(
        vec![lv("0.50", "4"), lv("0.55", "6")],
        vec![lv("0.49", "100")],
        Some(RATE_CRYPTO),
    ));
    sim.submit(limit(dec!(0.55), dec!(10), TimeInForce::Fak))
        .await;
    let _ = rx.try_recv().unwrap();
    let ids: HashSet<String> = sim.executions().iter().map(|e| e.exec_id.clone()).collect();
    assert_eq!(ids.len(), 2, "two executions → two exec_ids");
}
