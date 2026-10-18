//! Tests for `StateStore::strategy_edge_stats`, the data layer behind
//! `GET /ui/strategies/edge` (expectancy/edge per strategy for the economic
//! evaluation of a strategy, an idea from Freqtrade's Edge
//! module).
//!
//! Like the rest of `tests/api_*.rs`, this exercises the `StateStore` layer:
//! there is no harness in this repo that builds a full `ApiState`. The
//! handler only serializes what this query returns (Decimals as strings,
//! an absent `Option` as `null`).
//!
//! The synthetic data is written through the real path (`apply_fill` +
//! `apply_settlement`), not by direct INSERT, so it also exercises the
//! convention that `transactions.pnl_um` only carries a value on SELL/REDEEM
//! and that `apply_settlement` settles the whole position with ONE REDEEM.

use poly_bot::{
    state::StateStore,
    types::{Fill, FillSide, Mode, StrategyId},
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Synthetic fill through the real persistence path. `order_id` must be
/// unique per test: it is the dedup key of `record_fill_conn` when the
/// fill carries no `exec_id`.
fn fill(
    strategy: StrategyId,
    token: &str,
    side: FillSide,
    price: Decimal,
    size: Decimal,
    ts_ms: i64,
    order_id: &str,
) -> Fill {
    Fill {
        market_cid: format!("cid-{token}"),
        token_id: token.into(),
        side,
        price,
        size,
        order_id: Some(order_id.into()),
        ts: chrono::DateTime::from_timestamp_millis(ts_ms).unwrap(),
        mode: Mode::Paper,
        neg_risk: false,
        strategy,
        clear_at: 9_999_999_999,
        outcome_label: Some("YES".into()),
        hedge_id: None,
        exec_id: None,
        local_order_id: None,
    }
}

async fn buy(
    store: &StateStore,
    strategy: StrategyId,
    token: &str,
    price: Decimal,
    size: Decimal,
    ts_ms: i64,
    order_id: &str,
) {
    store
        .apply_fill(&fill(
            strategy,
            token,
            FillSide::Buy,
            price,
            size,
            ts_ms,
            order_id,
        ))
        .await
        .unwrap();
}

async fn sell(
    store: &StateStore,
    strategy: StrategyId,
    token: &str,
    price: Decimal,
    size: Decimal,
    ts_ms: i64,
    order_id: &str,
) {
    store
        .apply_fill(&fill(
            strategy,
            token,
            FillSide::Sell,
            price,
            size,
            ts_ms,
            order_id,
        ))
        .await
        .unwrap();
}

/// Settles the open position of `token` through the real path of
/// `apply_settlement` (the REDEEM is stamped with `Utc::now()`, later than
/// any explicit ts_ms of the test fills).
async fn settle(store: &StateStore, strategy: StrategyId, token: &str, won: bool) {
    let pos = store
        .get_position(token, &strategy.to_string())
        .await
        .unwrap()
        .expect("an open position must exist to settle");
    store
        .apply_settlement(&pos, won, "YES", dec!(0), dec!(0), strategy, None)
        .await
        .unwrap();
}

/// Two strategies with different results: correct stats per strategy
/// and an honest absence of the strategy without trades.
#[tokio::test]
async fn edge_stats_two_strategies_with_correct_stats() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();

    // arb: winning trade (+2, cost 5) and losing trade (-1, cost 6).
    buy(
        &store,
        StrategyId::Arb,
        "arb-t1",
        dec!(0.50),
        dec!(10),
        1000,
        "b1",
    )
    .await;
    sell(
        &store,
        StrategyId::Arb,
        "arb-t1",
        dec!(0.70),
        dec!(10),
        2000,
        "s1",
    )
    .await;
    buy(
        &store,
        StrategyId::Arb,
        "arb-t2",
        dec!(0.60),
        dec!(10),
        3000,
        "b2",
    )
    .await;
    sell(
        &store,
        StrategyId::Arb,
        "arb-t2",
        dec!(0.50),
        dec!(10),
        4000,
        "s2",
    )
    .await;

    // near_certain: trade closed by a winning settlement (+2, cost 18) and
    // trade with a partial sell + losing settlement (-4.4 total, cost 8).
    buy(
        &store,
        StrategyId::NearCertain,
        "nc-t1",
        dec!(0.90),
        dec!(20),
        1000,
        "b3",
    )
    .await;
    buy(
        &store,
        StrategyId::NearCertain,
        "nc-t2",
        dec!(0.80),
        dec!(10),
        2000,
        "b4",
    )
    .await;
    sell(
        &store,
        StrategyId::NearCertain,
        "nc-t2",
        dec!(0.90),
        dec!(4),
        3000,
        "s3",
    )
    .await;
    settle(&store, StrategyId::NearCertain, "nc-t1", true).await;
    settle(&store, StrategyId::NearCertain, "nc-t2", false).await;

    let map = store.strategy_edge_stats().await.unwrap();

    // The strategy without trades does not appear: neither zeros nor an empty row.
    assert!(
        !map.contains_key("directional"),
        "no trades = absent, not zeros"
    );
    assert_eq!(map.len(), 2, "only strategies with transactions");

    let arb = &map["arb"];
    assert_eq!(arb.closed_trades, 2);
    assert_eq!(arb.open_trades, 0);
    assert_eq!((arb.wins, arb.losses, arb.flat_trades), (1, 1, 0));
    assert_eq!(arb.win_rate, Some(dec!(0.5)));
    assert_eq!(arb.avg_win, Some(dec!(2)));
    assert_eq!(arb.avg_loss, Some(dec!(-1)));
    assert_eq!(arb.profit_factor, Some(dec!(2)), "gross 2 / |gross −1|");
    assert_eq!(
        arb.expectancy_per_trade,
        Some(dec!(0.5)),
        "(+2 −1) / 2 trades"
    );
    assert_eq!(arb.total_pnl, dec!(1));
    assert_eq!(arb.total_cost_deployed, dec!(11), "5 + 6 of the two trades");
    assert_eq!(
        arb.expectancy_pct.unwrap().round_dp(4),
        (dec!(1) / dec!(11)).round_dp(4)
    );
    // Exposure: +5 (t1) -> 0 (t2) -> +6 (t3, peak) -> 0 (t4).
    assert_eq!(arb.max_exposure, dec!(6));

    let nc = &map["near_certain"];
    assert_eq!(nc.closed_trades, 2);
    assert_eq!(nc.open_trades, 0);
    assert_eq!((nc.wins, nc.losses, nc.flat_trades), (1, 1, 0));
    assert_eq!(nc.avg_win, Some(dec!(2)));
    // The partial sell (+0.4) and the losing REDEEM (-4.8) are a SINGLE trade.
    assert_eq!(nc.avg_loss, Some(dec!(-4.4)));
    assert_eq!(nc.total_pnl, dec!(-2.4), "+2 − 4.4");
    assert_eq!(nc.total_cost_deployed, dec!(26), "18 + 8");
    assert_eq!(nc.expectancy_per_trade, Some(dec!(-1.2)));
    assert_eq!(
        nc.profit_factor.unwrap().round_dp(4),
        (dec!(2) / dec!(4.4)).round_dp(4)
    );
    // Peak with both positions open at the same time (t2): 18 + 8.
    assert_eq!(nc.max_exposure, dec!(26));

    // Reconciliation invariant: the per-strategy sum matches the existing
    // global aggregate (two independent paths to the same number).
    let total_map: Decimal = map.values().map(|s| s.total_pnl).sum();
    assert_eq!(total_map, store.realized_pnl_total().await.unwrap());
}

/// Re-buying after a full close of the SAME token = two trades, not one,
/// even though the `positions` row is reused (a late-sell fix does the same
/// with the position: the re-entry is a new position).
#[tokio::test]
async fn edge_stats_reentry_after_full_close_counts_two_trades() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();

    buy(
        &store,
        StrategyId::Arb,
        "re-t",
        dec!(0.50),
        dec!(5),
        1000,
        "b1",
    )
    .await;
    sell(
        &store,
        StrategyId::Arb,
        "re-t",
        dec!(0.60),
        dec!(5),
        2000,
        "s1",
    )
    .await;
    buy(
        &store,
        StrategyId::Arb,
        "re-t",
        dec!(0.50),
        dec!(5),
        3000,
        "b2",
    )
    .await;
    sell(
        &store,
        StrategyId::Arb,
        "re-t",
        dec!(0.40),
        dec!(5),
        4000,
        "s2",
    )
    .await;

    let arb = &store.strategy_edge_stats().await.unwrap()["arb"];
    assert_eq!(
        arb.closed_trades, 2,
        "inventory at 0 closes; a rebuy opens another"
    );
    assert_eq!((arb.wins, arb.losses), (1, 1));
    assert_eq!(arb.total_pnl, dec!(0), "+0.5 − 0.5");
    assert_eq!(arb.total_cost_deployed, dec!(5), "2.5 + 2.5");
    assert_eq!(arb.expectancy_per_trade, Some(dec!(0)));
}

/// Open positions only: the strategy appears (it has transactions) but all
/// the Options are None; no rates are fabricated over 0 trades.
#[tokio::test]
async fn edge_stats_only_open_optional_fields_null() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();
    buy(
        &store,
        StrategyId::MarketMaking,
        "mm-t",
        dec!(0.50),
        dec!(10),
        1000,
        "b1",
    )
    .await;

    let mm = &store.strategy_edge_stats().await.unwrap()["market_making"];
    assert_eq!(mm.closed_trades, 0);
    assert_eq!(mm.open_trades, 1, "the BUY without a close stays open");
    assert_eq!(mm.win_rate, None);
    assert_eq!(mm.avg_win, None);
    assert_eq!(mm.avg_loss, None);
    assert_eq!(mm.profit_factor, None);
    assert_eq!(mm.expectancy_per_trade, None);
    assert_eq!(mm.expectancy_pct, None, "cost deployed in closed = 0");
    assert_eq!(mm.total_pnl, dec!(0));
    assert_eq!(mm.max_exposure, dec!(5), "the peak does reflect what is open");
}

/// All winners: `profit_factor` is None (infinity is not fabricated); the
/// scenario is read from `wins`/`closed_trades`.
#[tokio::test]
async fn edge_stats_all_winners_profit_factor_null() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();
    buy(
        &store,
        StrategyId::DriftBox,
        "db-t",
        dec!(0.50),
        dec!(10),
        1000,
        "b1",
    )
    .await;
    sell(
        &store,
        StrategyId::DriftBox,
        "db-t",
        dec!(0.80),
        dec!(10),
        2000,
        "s1",
    )
    .await;

    let db = &store.strategy_edge_stats().await.unwrap()["drift_box"];
    assert_eq!(db.closed_trades, 1);
    assert_eq!(db.wins, 1);
    assert_eq!(db.losses, 0);
    assert_eq!(db.win_rate, Some(dec!(1)));
    assert_eq!(db.profit_factor, None, "no losses = None, not infinity");
    assert_eq!(db.avg_loss, None);
    assert_eq!(db.expectancy_pct, Some(dec!(0.6)), "+3 over 5 deployed");
}

/// Late sell on an already-settled position (late-sell branch of
/// `apply_fill`): the proceeds are genuinely realized and count as a
/// standalone trade of cost 0, distinct from the trade closed by the REDEEM.
#[tokio::test]
async fn edge_stats_late_post_settlement_sell_is_zero_cost_trade() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();
    buy(
        &store,
        StrategyId::ParityMaker,
        "pm-t",
        dec!(0.50),
        dec!(10),
        1000,
        "b1",
    )
    .await;
    settle(&store, StrategyId::ParityMaker, "pm-t", true).await;
    // Late fill after settlement: proceeds with assigned cost 0.
    sell(
        &store,
        StrategyId::ParityMaker,
        "pm-t",
        dec!(0.30),
        dec!(10),
        5000,
        "s1",
    )
    .await;

    let pm = &store.strategy_edge_stats().await.unwrap()["parity_maker"];
    assert_eq!(
        pm.closed_trades, 2,
        "REDEEM closes one; the late sell is another"
    );
    assert_eq!(pm.wins, 2);
    assert_eq!(pm.total_pnl, dec!(8), "+5 from the REDEEM +3 from the late sell");
    assert_eq!(
        pm.total_cost_deployed,
        dec!(5),
        "only the BUY of the first trade"
    );
    assert_eq!(pm.expectancy_per_trade, Some(dec!(4)));
    assert_eq!(pm.expectancy_pct, Some(dec!(1.6)), "+8 over 5 deployed");
    assert_eq!(pm.max_exposure, dec!(5));
}

/// Several buys and partial sells within the same token aggregate into ONE
/// trade as long as the inventory does not reach 0.
#[tokio::test]
async fn edge_stats_partials_and_rebuys_are_a_single_trade() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();
    buy(
        &store,
        StrategyId::ThetaCapture,
        "tc-t",
        dec!(0.50),
        dec!(10),
        1000,
        "b1",
    )
    .await;
    sell(
        &store,
        StrategyId::ThetaCapture,
        "tc-t",
        dec!(0.60),
        dec!(5),
        2000,
        "s1",
    )
    .await;
    buy(
        &store,
        StrategyId::ThetaCapture,
        "tc-t",
        dec!(0.55),
        dec!(5),
        3000,
        "b2",
    )
    .await;
    sell(
        &store,
        StrategyId::ThetaCapture,
        "tc-t",
        dec!(0.70),
        dec!(10),
        4000,
        "s2",
    )
    .await;

    let tc = &store.strategy_edge_stats().await.unwrap()["theta_capture"];
    assert_eq!(
        tc.closed_trades, 1,
        "the inventory did not return to 0 until the end"
    );
    assert_eq!(tc.open_trades, 0);
    // Trade: cost 5 + 2.75 = 7.75; proceeds 3 + 7 = 10 -> pnl +2.25.

    assert_eq!(tc.total_pnl, dec!(2.25));
    assert_eq!(tc.total_cost_deployed, dec!(7.75));
    assert_eq!(tc.wins, 1);
    assert_eq!(tc.expectancy_per_trade, Some(dec!(2.25)));
}
