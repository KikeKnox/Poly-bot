//! Integration test of the full pipeline.
//!
//!   OrderIntent → Executor → (no_fill) → order_map
//!                                        ↓ (handle_trade via sim)
//!   handle_trade(UserTrade MATCHED) → fill_persist_tx → fill_sink → SQLite + broadcast<Fill>
//!
//! Verifies the paper path is identical to live: the fill always arrives async
//! via handle_trade, never synchronously through the executor's return.

use std::sync::Arc;

use tokio::sync::{broadcast, mpsc, watch};

use poly_bot::{
    accounting::fill_sink,
    market_feed::MarketRegistry,
    order::{executor::Executor, sim_router::SimulatedRouter},
    risk::RiskGuard,
    state::StateStore,
    types::{Mode, OrderIntent, OrderSide, QueuedIntent, StrategyId},
};
use rust_decimal_macros::dec;

fn make_sim(
    fill_persist_tx: mpsc::UnboundedSender<poly_bot::types::Fill>,
    registry: Arc<MarketRegistry>,
    balance: rust_decimal::Decimal,
    wallet_tx: watch::Sender<Option<rust_decimal::Decimal>>,
) -> Arc<SimulatedRouter> {
    // fok_fill_rate=1.0 → always fills; latency=0; no fee or slippage for tests.
    SimulatedRouter::new(
        0,
        1.0,
        0,
        rust_decimal::Decimal::ZERO,
        1.0,
        fill_persist_tx,
        registry,
        balance,
        wallet_tx,
    )
}

fn make_order_map() -> Arc<parking_lot::Mutex<std::collections::HashMap<String, StrategyId>>> {
    Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()))
}

/// The simulator requires real depth — no book means no fill. Seeds
/// a `BookSnapshot` for each token PAIR (yes/no), with asks at the given
/// price and ample depth for the tests.
fn seed_books(sim: &SimulatedRouter, asks: &[(&str, &str)]) {
    use poly_bot::types::{BookSnapshot, HalfBook, Level, TokenBook};
    for pair in asks.chunks(2) {
        let mk = |tok: &str, ask: &str| TokenBook {
            token_id: tok.into(),
            bids: HalfBook {
                levels: vec![Level {
                    price: dec!(0.01),
                    size: dec!(1000),
                }],
            },
            asks: HalfBook {
                levels: vec![Level {
                    price: ask.parse().unwrap(),
                    size: dec!(1000),
                }],
            },
            ..Default::default()
        };
        let (t1, a1) = pair[0];
        let (t2, a2) = pair.get(1).copied().unwrap_or(("tok-no-dummy", "0.50"));
        sim.tick_book(&BookSnapshot {
            market_condition_id: "integ-cid".into(),
            yes: mk(t1, a1),
            no: mk(t2, a2),
            market_open_ts: 0,
            clear_at: 9_999_999_999,
            neg_risk: false,
            slug: "test".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        });
    }
}

/// Construye el pipeline complete en memoria.
/// Returns (intent_tx, state, wallet_rx, fill_rx, router) — the router is
/// returned so books can be seeded (no book means no fill).
async fn build_pipeline() -> (
    mpsc::Sender<QueuedIntent>,
    Arc<StateStore>,
    watch::Receiver<Option<rust_decimal::Decimal>>,
    broadcast::Receiver<poly_bot::types::Fill>,
    Arc<SimulatedRouter>,
) {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let (fill_tx, fill_rx) = broadcast::channel::<poly_bot::types::Fill>(64);
    let (fill_persist_tx, fill_persist_rx) = mpsc::unbounded_channel::<poly_bot::types::Fill>();
    let risk = RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper);
    let registry = Arc::new(MarketRegistry::new());
    let (wallet_tx, wallet_rx) = watch::channel::<Option<rust_decimal::Decimal>>(None);
    let router = make_sim(fill_persist_tx.clone(), registry, dec!(1000), wallet_tx);
    let (intent_tx, intent_rx) = mpsc::channel(16);
    let order_map = make_order_map();

    let executor = Executor::new(
        intent_rx,
        fill_persist_tx,
        router.clone(),
        risk.clone(),
        state.clone(),
        order_map.clone(),
        wallet_rx.clone(),
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
        poly_bot::coordination::BracketLedger::new(),
    );

    let risk2 = RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper);
    let (_, kill_rx) = watch::channel(false);
    tokio::spawn(fill_sink::run(
        fill_persist_rx,
        state.clone(),
        risk2,
        poly_bot::coordination::BracketLedger::new(),
        fill_tx,
        order_map,
        kill_rx,
        tokio::sync::oneshot::channel().0, // drain-complete warning, unused here
    ));
    tokio::spawn(executor.run());

    (intent_tx, state, wallet_rx, fill_rx, router)
}

fn market_buy_intent(token_id: &str, price: &str, size: &str) -> OrderIntent {
    let p: rust_decimal::Decimal = price.parse().unwrap();
    let s: rust_decimal::Decimal = size.parse().unwrap();
    OrderIntent {
        market_cid: "integ-cid".into(),
        token_id: token_id.into(),
        side: OrderSide::Buy,
        price: p,
        size: s,
        clear_at: chrono::Utc::now().timestamp() + 600,
        signal: p,
        neg_risk: false,
        limit_price: Some(p),
        second_leg: None,
        strategy: StrategyId::NearCertain,
        resting: false,
        real_clear_at: None,
    }
}

#[tokio::test]
async fn pipeline_executor_emits_fill() {
    // Verifies the fill arrives via handle_trade → fill_sink → broadcast (same as live).
    let (intent_tx, _, _, mut fill_rx, sim) = build_pipeline().await;
    seed_books(&sim, &[("tok-yes", "0.75")]);

    intent_tx
        .send((
            market_buy_intent("tok-yes", "0.75", "10"),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();

    // The fill must arrive within <200ms: executor → no_fill → sim handle_trade → fill_persist_tx
    // → fill_sink → broadcast. Same path as live.
    let fill = tokio::time::timeout(tokio::time::Duration::from_millis(200), fill_rx.recv())
        .await
        .expect("timeout — fill did not arrive via handle_trade")
        .expect("channel closed");

    assert_eq!(fill.token_id, "tok-yes");
    assert_eq!(fill.size, dec!(10));
}

#[tokio::test]
async fn pipeline_buy_creates_position_in_db() {
    let (intent_tx, state, _, _, sim) = build_pipeline().await;
    seed_books(&sim, &[("tok-yes", "0.75")]);

    intent_tx
        .send((
            market_buy_intent("tok-yes", "0.75", "10"),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    let pos = state.get_position("tok-yes", "near_certain").await.unwrap();
    assert!(pos.is_some(), "position must exist after the fill");
    let pos = pos.unwrap();
    assert_eq!(pos.size, dec!(10));
    assert_eq!(pos.avg_price, dec!(0.75));
}

#[tokio::test]
async fn pipeline_balance_is_deducted() {
    // Verifies the paper balance is debited after a fill — a new property post-refactor.
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let (fill_tx, _) = broadcast::channel::<poly_bot::types::Fill>(64);
    let (fill_persist_tx, fill_persist_rx) = mpsc::unbounded_channel::<poly_bot::types::Fill>();
    let risk = RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper);
    let registry = Arc::new(MarketRegistry::new());
    let (wallet_tx, wallet_rx) = watch::channel(None);
    let initial = dec!(500);
    let router = make_sim(fill_persist_tx.clone(), registry, initial, wallet_tx);
    let (intent_tx, intent_rx) = mpsc::channel(16);
    let order_map = make_order_map();

    let executor = Executor::new(
        intent_rx,
        fill_persist_tx,
        router.clone(),
        risk.clone(),
        state.clone(),
        order_map.clone(),
        wallet_rx.clone(),
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
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

    // Initial balance must be 500 (published by the sim in new()).
    let initial_bal = *wallet_rx.borrow();
    assert_eq!(initial_bal, Some(initial));

    // No book means no fill — seed asks at 0.75 for tok-bal.
    seed_books(&router, &[("tok-bal", "0.75")]);

    // Compra 10 shares a 0.75 = 7.50 USDC
    intent_tx
        .send((
            market_buy_intent("tok-bal", "0.75", "10"),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let new_bal = *wallet_rx.borrow();
    assert!(new_bal.is_some());
    assert!(
        new_bal.unwrap() < initial,
        "balance must have been debited after the purchase"
    );
}

#[tokio::test]
async fn pipeline_cap_bracket_blocks_second_stacking() {
    // With max_bracket_usdc=8, two buys of 7.5 USDC on the same side of the same
    // bracket: the first passes, the second exceeds the net (15 > 8) and never reaches the router.
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let (fill_tx, _) = broadcast::channel::<poly_bot::types::Fill>(64);
    let (fill_persist_tx, fill_persist_rx) = mpsc::unbounded_channel::<poly_bot::types::Fill>();
    let risk_cfg = poly_bot::config::RiskConfig {
        max_bracket_usdc: dec!(8),
        ..Default::default()
    };
    let risk = RiskGuard::new(risk_cfg.clone(), Mode::Paper);
    let registry = Arc::new(MarketRegistry::new());
    let (wallet_tx, wallet_rx) = watch::channel(None);
    let router = make_sim(fill_persist_tx.clone(), registry, dec!(1000), wallet_tx);
    let (intent_tx, intent_rx) = mpsc::channel(16);
    let order_map = make_order_map();
    let ledger = poly_bot::coordination::BracketLedger::new();

    let executor = Executor::new(
        intent_rx,
        fill_persist_tx,
        router.clone(),
        risk.clone(),
        state.clone(),
        order_map.clone(),
        wallet_rx.clone(),
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
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
        tokio::sync::oneshot::channel().0, // drain-complete warning, unused here
    ));
    tokio::spawn(executor.run());

    seed_books(&router, &[("tok-cap", "0.75")]);
    intent_tx
        .send((
            market_buy_intent("tok-cap", "0.75", "10"),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    intent_tx
        .send((
            market_buy_intent("tok-cap", "0.75", "10"),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let pos = state
        .get_position("tok-cap", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        pos.size,
        dec!(10),
        "the second buy should have been blocked by the bracket cap"
    );
    assert_eq!(ledger.net_exposure("integ-cid"), dec!(7.5));
}

#[tokio::test]
async fn pipeline_global_exposure_via_ledger_blocks_without_sqlite() {
    // The global check reads claims from the ledger, not SQLite. Two buys of
    // 7.5 USDC on DIFFERENT brackets sent back to back (without waiting for the fill):
    // with max_total_exposure=10, the first's in-flight claim blocks the
    // second even though the position is not persisted yet.
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let (fill_tx, _) = broadcast::channel::<poly_bot::types::Fill>(64);
    let (fill_persist_tx, fill_persist_rx) = mpsc::unbounded_channel::<poly_bot::types::Fill>();
    let risk_cfg = poly_bot::config::RiskConfig {
        max_total_exposure_usdc: dec!(10),
        ..Default::default()
    };
    let risk = RiskGuard::new(risk_cfg.clone(), Mode::Paper);
    let registry = Arc::new(MarketRegistry::new());
    let (wallet_tx, wallet_rx) = watch::channel(None);
    let router = make_sim(fill_persist_tx.clone(), registry, dec!(1000), wallet_tx);
    let (intent_tx, intent_rx) = mpsc::channel(16);
    let order_map = make_order_map();
    let ledger = poly_bot::coordination::BracketLedger::new();

    let executor = Executor::new(
        intent_rx,
        fill_persist_tx,
        router.clone(),
        risk.clone(),
        state.clone(),
        order_map.clone(),
        wallet_rx.clone(),
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
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
        tokio::sync::oneshot::channel().0, // drain-complete warning, unused here
    ));
    tokio::spawn(executor.run());

    let mut i1 = market_buy_intent("tok-glob-a", "0.75", "10");
    i1.market_cid = "cid-glob-a".into();
    let mut i2 = market_buy_intent("tok-glob-b", "0.75", "10");
    i2.market_cid = "cid-glob-b".into();
    // No book means no fill — both tokens in the same snapshot.
    seed_books(&router, &[("tok-glob-a", "0.75"), ("tok-glob-b", "0.75")]);
    // No sleep between sends: the second can only be blocked if the first's
    // claim (still without a persisted fill) counts in the global check.
    intent_tx
        .send((i1, std::time::Instant::now()))
        .await
        .unwrap();
    intent_tx
        .send((i2, std::time::Instant::now()))
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    assert!(state
        .get_position("tok-glob-a", "near_certain")
        .await
        .unwrap()
        .is_some());
    assert!(
        state
            .get_position("tok-glob-b", "near_certain")
            .await
            .unwrap()
            .is_none(),
        "the second buy should have been blocked by global exposure via the ledger"
    );
    assert_eq!(ledger.total_exposure(), dec!(7.5));
}

#[tokio::test]
async fn pipeline_two_buys_averaging() {
    let (intent_tx, state, _, _, sim) = build_pipeline().await;
    // Ask at 0.60: covers both limits (0.60 and 0.80); the average fill comes out below.
    seed_books(&sim, &[("tok-avg", "0.60")]);

    intent_tx
        .send((
            market_buy_intent("tok-avg", "0.60", "10"),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
    intent_tx
        .send((
            market_buy_intent("tok-avg", "0.80", "10"),
            std::time::Instant::now(),
        ))
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let pos = state
        .get_position("tok-avg", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert!(
        pos.size >= dec!(10),
        "there must be at least the first buy"
    );
}

// ── Global kill cancels the GTDs of all assets ───────────────────────────────────

/// Fake router: accepts GTDs (no_fill + order_id) and records the cancels.
struct FakeGtdRouter {
    cancels: Arc<parking_lot::Mutex<Vec<String>>>,
}

impl poly_bot::order::router::OrderRouter for FakeGtdRouter {
    fn submit(
        &self,
        _req: poly_bot::order::router::OrderRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = poly_bot::order::router::RouterResult> + Send + '_>,
    > {
        Box::pin(async { poly_bot::order::router::RouterResult::no_fill(Some("gtd-fake-1".into())) })
    }

    fn cancel(
        &self,
        _market_cid: &str,
        order_id: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>> {
        self.cancels.lock().push(order_id.to_string());
        Box::pin(async { Ok(()) })
    }

    fn mode(&self) -> Mode {
        Mode::Paper
    }
}

#[tokio::test]
async fn kill_global_cancels_active_gtds() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let (fill_persist_tx, _fill_persist_rx) = mpsc::unbounded_channel::<poly_bot::types::Fill>();
    // paper_disable_kill_switch=false so the kill also acts in paper.
    let risk = RiskGuard::new(
        poly_bot::config::RiskConfig {
            paper_disable_kill_switch: false,
            ..poly_bot::config::RiskConfig::default()
        },
        Mode::Paper,
    );
    let cancels = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let router = Arc::new(FakeGtdRouter {
        cancels: cancels.clone(),
    });
    let (_wallet_tx, wallet_rx) = watch::channel::<Option<rust_decimal::Decimal>>(None);
    let (intent_tx, intent_rx) = mpsc::channel(16);

    let executor = Executor::new(
        intent_rx,
        fill_persist_tx,
        router,
        risk.clone(),
        state,
        make_order_map(),
        wallet_rx,
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
        poly_bot::coordination::BracketLedger::new(),
    );
    tokio::spawn(executor.run());

    // Post a GTD (resting) and give it time to register in active_gtd.
    let mut intent = market_buy_intent("tok-gtd", "0.50", "10");
    intent.resting = true;
    intent_tx
        .send((intent, std::time::Instant::now()))
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    assert!(cancels.lock().is_empty(), "without a kill there must be no cancels");

    // Global kill → the executor must cancel the GTD without waiting for another intent.
    risk.kill_global("test");
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    assert_eq!(
        cancels.lock().as_slice(),
        ["gtd-fake-1"],
        "the global kill must cancel the active GTDs"
    );
}
