//! ARB-15M5M en el pipeline paper complete:
//!   books de DOS markets → strategy → OrderIntent (+ second_leg) → Executor
//!   → SimulatedRouter → fill_sink → SQLite; the fills return to the strategy
//!   and the streak is recorded in `arb15m5m_streaks`.

use std::sync::Arc;

use tokio::sync::{broadcast, mpsc, watch};

use poly_bot::{
    accounting::fill_sink,
    market_feed::MarketRegistry,
    oracle::chainlink_twap::TwapStore,
    order::{executor::Executor, sim_router::SimulatedRouter},
    risk::RiskGuard,
    state::StateStore,
    strategy::{
        arb_15m5m::{Arb15m5m, Arb15m5mConfig},
        Arb15m5mDeps, Strategy,
    },
    types::{BookSnapshot, Fill, HalfBook, Level, Mode, StrategyId, TokenBook, VolSnapshot},
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

fn token(id: &str, ask: Decimal) -> TokenBook {
    TokenBook {
        token_id: id.into(),
        bids: HalfBook { levels: vec![Level { price: dec!(0.10), size: dec!(1000) }] },
        asks: HalfBook { levels: vec![Level { price: ask, size: dec!(1000) }] },
        ..Default::default()
    }
}

fn book(dur: &str, seg: i64, t1: i64, up: Decimal, down: Decimal) -> BookSnapshot {
    BookSnapshot {
        market_condition_id: format!("cid-{dur}"),
        slug: format!("btc-updown-{dur}-{}", t1 - seg),
        clear_at: t1,
        yes: token(&format!("u{dur}"), up),
        no: token(&format!("d{dur}"), down),
        ..Default::default()
    }
}

#[tokio::test]
async fn two_market_basket_fills_and_is_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("hrf_state.sqlite");
    let state = StateStore::open(db.to_str().unwrap(), Mode::Paper).await.unwrap();
    let db_real = StateStore::resolve_path(db.to_str().unwrap(), Mode::Paper);
    let (fill_tx, mut fill_rx) = broadcast::channel::<Fill>(64);
    let (fill_persist_tx, fill_persist_rx) = mpsc::unbounded_channel::<Fill>();
    let risk = RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper);
    let (wallet_tx, wallet_rx) = watch::channel::<Option<Decimal>>(None);
    let router = SimulatedRouter::new(
        0, 1.0, 0, Decimal::ZERO, 1.0,
        fill_persist_tx.clone(),
        Arc::new(MarketRegistry::new()),
        dec!(1000),
        wallet_tx,
    );
    let (intent_tx, intent_rx) = mpsc::channel(16);
    let order_map = Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    let executor = Executor::new(
        intent_rx,
        fill_persist_tx,
        router.clone(),
        risk,
        state.clone(),
        order_map.clone(),
        wallet_rx,
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
        poly_bot::coordination::BracketLedger::new(),
    );
    let (_, kill_rx) = watch::channel(false);
    tokio::spawn(fill_sink::run(
        fill_persist_rx,
        state.clone(),
        RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper),
        poly_bot::coordination::BracketLedger::new(),
        fill_tx,
        order_map,
        kill_rx,
        tokio::sync::oneshot::channel().0,
    ));
    tokio::spawn(executor.run());

    // Exact strikes: s15 < s5 → Up 15m + Down 5m.
    let t1 = chrono::Utc::now().timestamp() + 200;
    let twap = TwapStore::new();
    twap.insert("btc", (t1 - 900) * 1000, dec!(100));
    twap.insert("btc", (t1 - 300) * 1000, dec!(101));
    let mut strategy = Arb15m5m::new(
        Arb15m5mConfig { cooldown_ms: 0, ..Default::default() },
        Arb15m5mDeps { twap, state: Some(state.clone()), filled_per_book: true },
    );

    let b15 = book("15m", 900, t1, dec!(0.40), dec!(0.55));
    let b5 = book("5m", 300, t1, dec!(0.55), dec!(0.45));
    router.tick_book(&b15);
    router.tick_book(&b5);

    assert!(strategy.on_book(&b15, &VolSnapshot::default()).is_empty());
    let intents = strategy.on_book(&b5, &VolSnapshot::default());
    assert_eq!(intents.len(), 1);
    for intent in intents {
        intent_tx.send((intent, std::time::Instant::now())).await.unwrap();
    }

    // Two fills (one per market) come back to the strategy over the broadcast.
    for _ in 0..2 {
        let f = tokio::time::timeout(std::time::Duration::from_secs(2), fill_rx.recv())
            .await
            .expect("fill did not arrive")
            .unwrap();
        assert_eq!(f.strategy, StrategyId::Arb15m5m);
        strategy.on_fill(&f);
    }

    // Positions in the TWO markets under the strategy.
    let p15 = state.get_position("u15m", "arb_15m5m").await.unwrap().expect("pos Up 15m");
    let p5 = state.get_position("d5m", "arb_15m5m").await.unwrap().expect("pos Down 5m");
    assert_eq!(p15.size, p5.size, "same q on both legs");

    // Telemetry: the write is a separate task; wait for it to show up.
    let reader = sqlx::SqlitePool::connect(&format!("sqlite://{}?mode=ro", db_real.display()))
        .await
        .unwrap();
    let mut row_entry = None;
    for _ in 0..40 {
        row_entry = sqlx::query_as::<_, (String, String, String, String)>(
            "SELECT result, direction, cid_15m, cid_5m FROM arb15m5m_streaks",
        )
        .fetch_optional(&reader)
        .await
        .unwrap();
        if row_entry.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let (result, direction, cid_15m, cid_5m) = row_entry.expect("streak without record");
    assert_eq!((result.as_str(), direction.as_str()), ("both", "up15_down5"));
    assert_eq!((cid_15m.as_str(), cid_5m.as_str()), ("cid-15m", "cid-5m"));
}
