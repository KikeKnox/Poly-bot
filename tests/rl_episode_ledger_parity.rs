//! E2E parity of `EpisodeLedger` (the RL episode view) ↔ `StateStore`
//! (durable SQLite ledger) over the SAME sequence of fills.
//!
//! The `Fill` contract in paper/sim: `price` is the ALL-IN price
//! (`cash/shares` — fee and impact included; see
//! `SimulatedRouter::emit_execution`) and the durable ledger books
//! `usdc = price × size`. The episode ledger must reproduce that
//! arithmetic EXACTLY — this test proves it transition by transition:
//! cash, residual cost, realized PnL and settlement, comparing against the real
//! `apply_fill`/`apply_settlement`/`economic_flows`.

use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use poly_bot::{
    rl::episode::EpisodeLedger,
    state::{to_micros, AppliedFill, StateStore},
    types::{Fill, FillSide, Mode, StrategyId},
};

const CID: &str = "0xcond-parity";
const TOK: &str = "tok-yes-parity";
const BUDGET: Decimal = dec!(10);

fn mk_fill(seq: u64, side: FillSide, price: Decimal, size: Decimal) -> Fill {
    Fill {
        market_cid: CID.into(),
        token_id: TOK.into(),
        side,
        price,
        size,
        order_id: Some(format!("ord-{seq}")),
        ts: Utc::now(),
        mode: Mode::Paper,
        neg_risk: false,
        strategy: StrategyId::RlPredictor,
        clear_at: 9_999_999_999,
        outcome_label: Some("YES".into()),
        hedge_id: None,
        exec_id: Some(format!("exec-{seq}")),
        local_order_id: Some(format!("loc-{seq}")),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn episode_matches_durable_ledger_transition_by_transition() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let mut ep = EpisodeLedger::new(CID, BUDGET, 0);
    let fee_rate = dec!(0.07);
    let strat = StrategyId::RlPredictor.to_string();
    let budget_um = to_micros(BUDGET);

    // ── T1: BUY 10 @ 0.55 (all-in price) ───────────────────────────
    let f1 = mk_fill(1, FillSide::Buy, dec!(0.55), dec!(10));
    assert_eq!(state.apply_fill(&f1).await.unwrap(), AppliedFill::Buy);
    ep.apply_buy_fill(TOK, f1.price, f1.size, 0, fee_rate);
    let pos = state.get_position(TOK, &strat).await.unwrap().unwrap();
    assert_eq!(pos.size, dec!(10));
    assert_eq!(pos.cost_basis, dec!(5.50));
    assert_eq!(ep.cash_um, budget_um - to_micros(dec!(5.50)));
    assert_eq!(ep.positions[TOK].size_um, to_micros(dec!(10)));
    assert_eq!(ep.positions[TOK].cost_um, to_micros(dec!(5.50)));

    // ── T2: BUY 6 @ 0.40 (averaging) ────────────────────────────────
    let f2 = mk_fill(2, FillSide::Buy, dec!(0.40), dec!(6));
    assert_eq!(state.apply_fill(&f2).await.unwrap(), AppliedFill::Buy);
    ep.apply_buy_fill(TOK, f2.price, f2.size, 0, fee_rate);
    let pos = state.get_position(TOK, &strat).await.unwrap().unwrap();
    assert_eq!(pos.size, dec!(16));
    assert_eq!(pos.cost_basis, dec!(7.90));
    assert_eq!(ep.cash_um, budget_um - to_micros(dec!(7.90)));
    assert_eq!(ep.positions[TOK].cost_um, to_micros(dec!(7.90)));

    // ── T3: partial SELL 4 @ 0.60 ───────────────────────────────────
    // allocated durable = 7.90 × 4/16 = 1.975 → pnl = 2.40 − 1.975.
    let f3 = mk_fill(3, FillSide::Sell, dec!(0.60), dec!(4));
    let af = state.apply_fill(&f3).await.unwrap();
    let AppliedFill::Sell { pnl, cost_released } = af else {
        panic!("unexpected sell: {af:?}")
    };
    let real_before = ep.realized_pnl_um;
    assert!(ep.apply_sell_fill(TOK, f3.price, f3.size, fee_rate));
    assert_eq!(ep.realized_pnl_um - real_before, to_micros(pnl));
    let pos = state.get_position(TOK, &strat).await.unwrap().unwrap();
    assert_eq!(to_micros(pos.cost_basis), ep.positions[TOK].cost_um);
    assert_eq!(to_micros(pos.size), ep.positions[TOK].size_um);
    assert_eq!(ep.positions[TOK].cost_um, to_micros(dec!(7.90)) - to_micros(cost_released));
    assert_eq!(
        ep.cash_um,
        budget_um - to_micros(dec!(7.90)) + to_micros(dec!(2.40))
    );

    // ── T4: full SELL 12 @ 0.50 (complete close) ──────────────────
    let f4 = mk_fill(4, FillSide::Sell, dec!(0.50), dec!(12));
    let af = state.apply_fill(&f4).await.unwrap();
    let AppliedFill::Sell { pnl, .. } = af else {
        panic!("unexpected sell: {af:?}")
    };
    let real_before = ep.realized_pnl_um;
    assert!(ep.apply_sell_fill(TOK, f4.price, f4.size, fee_rate));
    assert_eq!(ep.realized_pnl_um - real_before, to_micros(pnl));
    let pos = state.get_position(TOK, &strat).await.unwrap().unwrap();
    assert_eq!(pos.size, Decimal::ZERO);
    assert_eq!(ep.positions[TOK].size_um, 0);
    assert_eq!(ep.positions[TOK].cost_um, 0);

    // ── Flow identity: episode cash == durable flows ─
    let flows = state.economic_flows().await.unwrap();
    let expected_um = budget_um - to_micros(flows.buys_usdc) + to_micros(flows.sells_usdc);
    assert_eq!(ep.cash_um, expected_um, "episode cash != durable");
    assert_eq!(ep.realized_pnl_um, to_micros(flows.sells_pnl));
    // And with positions at zero, equity == cash → the episode return
    // (Σ rewards × budget, lambdas=0) is exactly the cash delta.
    assert_eq!(ep.equity_um(0), ep.cash_um);
}

#[tokio::test(flavor = "current_thread")]
async fn episode_matches_settlement_winner_and_loser() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let mut ep = EpisodeLedger::new(CID, BUDGET, 0);
    let strat = StrategyId::RlPredictor.to_string();

    // Position 3 @ 0.30 all-in; winning settlement pays size (1.0/share).
    let f = mk_fill(1, FillSide::Buy, dec!(0.30), dec!(3));
    assert_eq!(state.apply_fill(&f).await.unwrap(), AppliedFill::Buy);
    ep.apply_buy_fill(TOK, f.price, f.size, 0, dec!(0.07));

    let pos = state.get_position(TOK, &strat).await.unwrap().unwrap();
    let payout = pos.size; // winner: 1.0/share
    assert!(
        state
            .apply_settlement(&pos, true, "YES", payout, payout - pos.cost_basis,
                              StrategyId::RlPredictor, None)
            .await
            .unwrap()
    );
    let cash_before = ep.cash_um;
    assert!(ep.apply_settlement(TOK, true));
    assert_eq!(ep.cash_um, cash_before + to_micros(dec!(3)));

    let flows = state.economic_flows().await.unwrap();
    let expected_um =
        to_micros(BUDGET) - to_micros(flows.buys_usdc) + to_micros(flows.redeem_usdc);
    assert_eq!(ep.cash_um, expected_um);
    assert_eq!(
        ep.realized_pnl_um,
        to_micros(flows.sells_pnl + flows.redeem_pnl)
    );
}
