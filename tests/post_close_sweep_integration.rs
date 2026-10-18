//! Verifies the load-bearing detail of the synthetic `clear_at` of the maker GTD.
//!
//! `PostCloseSweep` posts its GTD bid with `clear_at = now + gtd_ttl_secs + 60`
//! (future), NEVER with the book's real `clear_at` (which has already passed post-close):
//! `sim_router.rs` purges on every `tick_book` the GTDs whose `clear_at` <= now,
//! and `clob_router.rs` would compute an already-expired expiration — with the real
//! `clear_at` the strategy would silently get 0 maker fills.
//!
//! Pipeline modeled on `tests/pipeline.rs::build_pipeline`, but keeping the
//! handle to the concrete `SimulatedRouter` so `tick_book` can be called by hand
//! (`build_pipeline` does not expose it).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{broadcast, mpsc, watch};

use poly_bot::{
    accounting::fill_sink,
    market_feed::MarketRegistry,
    order::{executor::Executor, sim_router::SimulatedRouter},
    risk::RiskGuard,
    state::StateStore,
    types::{
        BookSnapshot, Fill, HalfBook, Level, Mode, OrderIntent, OrderSide, QueuedIntent,
        StrategyId, TokenBook,
    },
};

/// Builds the full in-memory pipeline and returns the concrete `SimulatedRouter`
/// (not `Arc<dyn OrderRouter>`) so books can be pushed with `tick_book`.
async fn build_pipeline_with_sim() -> (
    mpsc::Sender<QueuedIntent>,
    broadcast::Receiver<Fill>,
    Arc<SimulatedRouter>,
) {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let (fill_tx, fill_rx) = broadcast::channel::<Fill>(64);
    let (fill_persist_tx, fill_persist_rx) = mpsc::unbounded_channel::<Fill>();
    let risk = RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper);
    let registry = Arc::new(MarketRegistry::new());
    let (wallet_tx, wallet_rx) = watch::channel::<Option<Decimal>>(None);
    // fok_fill_rate=1.0, latency=0, no fee or slippage — same as pipeline.rs::make_sim.
    let sim = SimulatedRouter::new(
        0,
        1.0,
        0,
        Decimal::ZERO,
        1.0,
        fill_persist_tx.clone(),
        registry,
        dec!(1000),
        wallet_tx,
    );
    let (intent_tx, intent_rx) = mpsc::channel(16);
    let order_map = Arc::new(parking_lot::Mutex::new(HashMap::new()));

    let executor = Executor::new(
        intent_rx,
        fill_persist_tx,
        sim.clone(),
        risk.clone(),
        state.clone(),
        order_map.clone(),
        wallet_rx,
        Arc::new(dashmap::DashSet::new()),
        HashSet::new(),
        poly_bot::coordination::BracketLedger::new(),
    );

    let risk2 = RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper);
    let (_, kill_rx) = watch::channel(false);
    tokio::spawn(fill_sink::run(
        fill_persist_rx,
        state,
        risk2,
        poly_bot::coordination::BracketLedger::new(),
        fill_tx,
        order_map,
        kill_rx,
        tokio::sync::oneshot::channel().0, // drain-complete warning, unused here
    ));
    tokio::spawn(executor.run());

    (intent_tx, fill_rx, sim)
}

/// GTD bid as `PostCloseSweep::gtd_bid_intent` would emit it (resting=true).
/// `real_clear_at`: `None` for the existing tests (which only exercise the
/// purge by `clear_at`, unrelated to the bug); `Some(t)` for the regression test of the
/// clear_at/real_clear_at bug further below.
fn gtd_intent(token_id: &str, clear_at: i64, real_clear_at: Option<i64>) -> OrderIntent {
    OrderIntent {
        market_cid: "pcs2-integ-cid".into(),
        token_id: token_id.into(),
        side: OrderSide::Buy,
        price: dec!(0.99),
        size: dec!(10),
        clear_at,
        signal: dec!(0.99),
        neg_risk: false,
        limit_price: Some(dec!(0.99)),
        second_leg: None,
        strategy: StrategyId::PostCloseSweep,
        resting: true,
        real_clear_at,
    }
}

/// Post-close book with the given token's ask below the GTD bid (0.99) —
/// enough to cross and fill the order if `tick_book` still holds it pending.
fn make_crossing_book(yes_token: &str, yes_ask: &str) -> BookSnapshot {
    let ask = HalfBook {
        levels: vec![Level {
            price: yes_ask.parse().unwrap(),
            size: dec!(100),
        }],
    };
    let bid = HalfBook {
        levels: vec![Level {
            price: dec!(0.01),
            size: dec!(100),
        }],
    };
    let now = chrono::Utc::now().timestamp();
    BookSnapshot {
        market_condition_id: "pcs2-integ-cid".into(),
        yes: TokenBook {
            token_id: yes_token.into(),
            bids: bid,
            asks: ask,
            ..Default::default()
        },
        no: TokenBook {
            token_id: "tok-no".into(),
            bids: HalfBook::default(),
            asks: HalfBook::default(),
            ..Default::default()
        },
        market_open_ts: now - 300,
        // Post-close book: the real clear_at has already passed (independent of the
        // synthetic clear_at carried by the GTD's OrderIntent).
        clear_at: now - 5,
        neg_risk: false,
        slug: "btc-updown-5m-pcs2-integ".into(),
        reference_price: Some(dec!(100000)),
        fee_rate_bps: None,
        parabolic_fee_rate: None,
        tick_size: None,
        recv_instant: None,
    }
}

#[tokio::test]
async fn gtd_with_future_clear_at_receives_fill() {
    let (intent_tx, mut fill_rx, sim) = build_pipeline_with_sim().await;

    // SYNTHETIC future clear_at (now + gtd_ttl_secs + 60), as PostCloseSweep emits.
    let synthetic_clear_at = chrono::Utc::now().timestamp() + 90 + 60;
    intent_tx
        .send((
            gtd_intent("tok-yes", synthetic_clear_at, None),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();

    // Give the executor time to register the GTD in the router before tick_book.
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    let book = make_crossing_book("tok-yes", "0.50");
    sim.tick_book(&book);

    let fill = tokio::time::timeout(std::time::Duration::from_millis(300), fill_rx.recv())
        .await
        .expect("timeout — the GTD with a synthetic future clear_at should have filled")
        .expect("fills channel closed");
    assert_eq!(fill.token_id, "tok-yes");
    assert_eq!(fill.size, dec!(10));
}

#[tokio::test]
async fn gtd_with_past_clear_at_does_not_receive_fill() {
    let (intent_tx, mut fill_rx, sim) = build_pipeline_with_sim().await;

    // Twin with the book's REAL clear_at (post-close → already passed). It must
    // be purged on the first tick_book before it can fill.
    // The GTD purge in sim_router now uses the same margin as
    // live (`clear_at + 60s`, parity with `clob_router.rs`/`executor.rs`) instead
    // of purging flat at the exact `clear_at` — so the offset has to
    // exceed that margin (-5s used to be enough).
    let real_past_clear_at = chrono::Utc::now().timestamp() - 65;
    intent_tx
        .send((
            gtd_intent("tok-yes", real_past_clear_at, None),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    let book = make_crossing_book("tok-yes", "0.50");
    sim.tick_book(&book);

    let res = tokio::time::timeout(std::time::Duration::from_millis(200), fill_rx.recv()).await;
    assert!(
        res.is_err(),
        "GTD with a real (past) clear_at must not get a fill — purged before the match"
    );
}

/// REGRESSION for the clear_at/real_clear_at bug (fix applied to `executor.rs::handle_buy`
/// and `handle_sell`): previously, the executor ALWAYS copied `intent.clear_at` to
/// `fill.clear_at` — for the post-close sweep GTD that is the SYNTHETIC future buffer
/// (`now + gtd_ttl_secs + 60`), not the bracket's real close. Since `fill.clear_at`
/// is persisted in `positions.clear_at` and `src/settlement/orphan.rs::list_expired_unresolved` filters
/// by `clear_at < now` to know when a position is ready to redeem, the
/// bug delayed that detection by 150-260s per position.
///
/// This end-to-end test verifies that, with an `OrderIntent` that carries `real_clear_at`
/// separate from `clear_at` (as emitted by `PostCloseSweep::gtd_bid_intent` via
/// `OrderIntent::resting_custom_expiry` after the fix), the resulting `fill.clear_at`
/// is the REAL close, not the synthetic one — back to how it was shown above that the synthetic
/// `clear_at` DOES avoid the purge (`gtd_with_future_clear_at_receives_fill`),
/// but the fill must no longer inherit that synthetic value.
#[tokio::test]
async fn fill_uses_real_clear_at_not_gtd_synthetic_buffer() {
    let (intent_tx, mut fill_rx, sim) = build_pipeline_with_sim().await;

    // SYNTHETIC future clear_at (so the exchange does not expire the GTD
    // instantly) different from real_clear_at (the bracket's real close, already passed
    // — post-close, as the sweep really operates).
    let synthetic_clear_at = chrono::Utc::now().timestamp() + 90 + 60;
    let real_clear_at = chrono::Utc::now().timestamp() - 5;
    intent_tx
        .send((
            gtd_intent("tok-yes", synthetic_clear_at, Some(real_clear_at)),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    let book = make_crossing_book("tok-yes", "0.50");
    sim.tick_book(&book);

    let fill = tokio::time::timeout(std::time::Duration::from_millis(300), fill_rx.recv())
        .await
        .expect("timeout — the GTD with a synthetic future clear_at should have filled")
        .expect("fills channel closed");
    assert_eq!(fill.token_id, "tok-yes");
    assert_eq!(
        fill.clear_at, real_clear_at,
        "fill.clear_at must be the market's REAL close (real_clear_at), not the GTD's synthetic expiration buffer"
    );
    assert_ne!(
        fill.clear_at, synthetic_clear_at,
        "the fixed bug did exactly this: leak the synthetic clear_at into the fill"
    );
}
