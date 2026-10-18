//! Integration test for the fill → SQLite cycle.
//!
//! Verifies that accounting::fill_sink::process writes to `fills`, creates the position and updates it
//! correctly in an in-memory SQLite.

use chrono::Utc;
use rust_decimal_macros::dec;

use poly_bot::{
    coordination::BracketLedger,
    accounting::fill_sink,
    risk::RiskGuard,
    state::StateStore,
    types::{Fill, FillSide, Mode, StrategyId},
};

async fn make_state() -> std::sync::Arc<StateStore> {
    StateStore::open_memory(Mode::Paper).await.unwrap()
}

fn make_risk() -> std::sync::Arc<RiskGuard> {
    RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper)
}

fn make_fill(token_id: &str, side: FillSide, price: &str, size: &str) -> Fill {
    Fill {
        hedge_id: None,
        exec_id: None,
        local_order_id: None,
        market_cid: "test-cid".into(),
        token_id: token_id.into(),
        side,
        price: price.parse().unwrap(),
        size: size.parse().unwrap(),
        order_id: None,
        ts: Utc::now(),
        mode: Mode::Paper,
        neg_risk: false,
        strategy: StrategyId::NearCertain,
        clear_at: Utc::now().timestamp() + 3600,
        outcome_label: None,
    }
}

#[tokio::test]
async fn fill_sink_buy_creates_position() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Buy, "0.70", "10"),
    )
    .await
    .unwrap();

    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .expect("position created");
    assert_eq!(pos.size, dec!(10));
    assert_eq!(pos.avg_price, dec!(0.70));
    assert_eq!(pos.cost_basis, dec!(7.0));
    assert_eq!(pos.strategy, "near_certain");

    let txs = state.recent_transactions(5, None).await.unwrap();
    assert_eq!(txs.len(), 1);
    assert_eq!(txs[0]["kind"], "BUY");
}

#[tokio::test]
async fn fill_sink_buy_averaging() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Buy, "0.60", "10"),
    )
    .await
    .unwrap();
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Buy, "0.80", "10"),
    )
    .await
    .unwrap();

    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(pos.size, dec!(20));
    // avg = (0.60*10 + 0.80*10) / 20 = 14 / 20 = 0.70
    assert_eq!(pos.avg_price, dec!(0.70));
    assert_eq!(pos.cost_basis, dec!(14.0));
}

#[tokio::test]
async fn fill_sink_sell_records_pnl() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Buy, "0.60", "10"),
    )
    .await
    .unwrap();
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Sell, "0.90", "10"),
    )
    .await
    .unwrap();

    let txs = state.recent_transactions(10, None).await.unwrap();
    let sell_tx = txs
        .iter()
        .find(|t| t["kind"] == "SELL")
        .expect("SELL transaction");
    // PnL expected: (0.90 - 0.60) * 10 = 3.0
    let pnl: rust_decimal::Decimal = sell_tx["pnl"]
        .as_str()
        .expect("pnl present")
        .parse()
        .expect("pnl parseable");
    assert_eq!(pnl, dec!(3.0));

    // The daily loss stop must reflect the session PnL
    assert_eq!(risk.session_pnl(), dec!(3.0));
}

#[tokio::test]
async fn fill_sink_same_order_id_is_idempotent() {
    // apply_fill deduplicates by order_id within the transaction — an
    // identical second fill must not duplicate the position or transactions.
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();
    let mut fill = make_fill("token-yes", FillSide::Buy, "0.70", "10");
    fill.order_id = Some("order-abc".into());

    let first = fill_sink::process(&state, &risk, &ledger, fill.clone())
        .await
        .unwrap();
    let second_result = fill_sink::process(&state, &risk, &ledger, fill)
        .await
        .unwrap();
    assert!(first, "the first fill is new");
    assert!(
        !second_result,
        "the second fill with the same order_id is a duplicate"
    );

    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(pos.size, dec!(10), "the position must not be duplicated");
    assert_eq!(state.recent_transactions(10, None).await.unwrap().len(), 1);
}

#[tokio::test]
async fn fill_sink_oversell_trims_to_position() {
    // A sell larger than the position in the DB (drift) is trimmed: the PnL enters
    // the daily-loss-stop based on the real size, not the fill's size.
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Buy, "0.60", "10"),
    )
    .await
    .unwrap();
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Sell, "0.90", "25"),
    )
    .await
    .unwrap();

    // Trimmed PnL: (0.90 − 0.60) × 10 = 3.0, not × 25 = 7.5.
    assert_eq!(risk.session_pnl(), dec!(3.0));
    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(pos.size, rust_decimal::Decimal::ZERO);
}

/// `accounting::fill_sink::run` (the real loop, not a direct `process` call) must keep
/// persisting and re-emitting a normal fill after the fix (retries + kill on
/// persistent failure): happy-path regression. The failure path (3 retries exhausted ->
/// `risk.kill_global`) is not covered here: forcing a real `apply_fill` error requires
/// access to the `StateStore` pool (private, outside this package's scope).
#[tokio::test]
async fn fill_sink_run_processes_and_emits_normal_fill() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();
    let (fill_persist_tx, fill_persist_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broadcast_tx, mut broadcast_rx) = tokio::sync::broadcast::channel(16);
    let order_map = std::sync::Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let (done_tx, _end_rx) = tokio::sync::oneshot::channel();

    let handle = tokio::spawn(fill_sink::run(
        fill_persist_rx,
        state.clone(),
        risk.clone(),
        ledger,
        broadcast_tx,
        order_map,
        shutdown_rx,
        done_tx,
    ));

    fill_persist_tx
        .send(make_fill("token-yes", FillSide::Buy, "0.70", "10"))
        .unwrap();

    let broadcasted = tokio::time::timeout(std::time::Duration::from_secs(2), broadcast_rx.recv())
        .await
        .expect("timeout waiting for the re-emitted fill")
        .unwrap();
    assert_eq!(broadcasted.price, dec!(0.70));

    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .expect("position created");
    assert_eq!(pos.size, dec!(10));
    assert!(
        !risk.is_killed(),
        "a fill that persists correctly must not kill the bot"
    );

    drop(fill_persist_tx);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
}

#[tokio::test]
async fn fill_sink_sell_releases_ledger_exposure() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();
    // Claim as the executor would make when buying: cost basis 0.60 × 10 = 6 USDC.
    ledger.try_add_exposure(
        "test-cid",
        9_999_999_999,
        "token-yes",
        dec!(6.0),
        dec!(0),
        poly_bot::types::StrategyId::NearCertain,
    );
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Buy, "0.60", "10"),
    )
    .await
    .unwrap();
    fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::Sell, "0.90", "10"),
    )
    .await
    .unwrap();

    // The sell releases the sold cost basis (avg 0.60 × 10), not the proceeds.
    assert_eq!(ledger.net_exposure("test-cid"), rust_decimal::Decimal::ZERO);
}

/// The ledger released `avg_price × size` (average price rounded to the
/// micro) while accounting withdrew the prorated cost: two numbers for the
/// same thing. Reviewer's case: BUY1 10 @0.33 -> SELL 3 -> BUY2 5 @0.40 ->
/// SELL 12. Accounting withdraws 0.99 + 4.31 = 5.30; the ledger released
/// 0.99 + 4.310004 = 5.300004, i.e. 4 µ of over-release.
#[tokio::test]
async fn fill_sink_sell_releases_exactly_the_withdrawn_cost() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();
    // Executor claim: the cost of both buys (3.30 + 2.00) plus a live GTD
    // of 1.00 that never reaches a fill. Without that GTD, the clamp to 0 in
    // `release_exposure` would hide the over-release.
    ledger.try_add_exposure(
        "test-cid",
        9_999_999_999,
        "token-yes",
        dec!(6.30),
        dec!(0),
        poly_bot::types::StrategyId::NearCertain,
    );

    for (side, price, size) in [
        (FillSide::Buy, "0.33", "10"),
        (FillSide::Sell, "0.50", "3"),
        (FillSide::Buy, "0.40", "5"),
        (FillSide::Sell, "0.50", "12"),
    ] {
        fill_sink::process(
            &state,
            &risk,
            &ledger,
            make_fill("token-yes", side, price, size),
        )
        .await
        .unwrap();
    }

    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .expect("position");
    assert_eq!(
        pos.cost_basis,
        rust_decimal::Decimal::ZERO,
        "full close: entire cost consumed"
    );
    assert_eq!(
        ledger.net_exposure("test-cid"),
        dec!(1.00),
        "only the live GTD must keep claiming exposure"
    );
}

/// Integration check: a GTD/FAK order can produce SEVERAL partial executions
/// that share `order_id` but carry a different `exec_id`. The durable dedup is
/// by `exec_id`: each partial must be counted (previously the UNIQUE index over
/// `order_id` silently discarded the second partial).
#[tokio::test]
async fn partial_fills_same_order_id_different_exec_id_count_each() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();

    let mut f1 = make_fill("token-yes", FillSide::Buy, "0.70", "3");
    f1.order_id = Some("ord-gtd".into());
    f1.exec_id = Some("exec-1".into());
    let mut f2 = f1.clone();
    f2.size = "4".parse().unwrap();
    f2.exec_id = Some("exec-2".into());
    let mut f3 = f1.clone();
    f3.size = "5".parse().unwrap();
    f3.exec_id = Some("exec-3".into());

    assert!(fill_sink::process(&state, &risk, &ledger, f1)
        .await
        .unwrap());
    assert!(
        fill_sink::process(&state, &risk, &ledger, f2)
            .await
            .unwrap(),
        "the second partial (same order_id, different exec_id) is NOT a duplicate"
    );
    assert!(fill_sink::process(&state, &risk, &ledger, f3)
        .await
        .unwrap());

    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        pos.size,
        dec!(12),
        "the three partials must add up to the position"
    );
    assert_eq!(state.recent_transactions(10, None).await.unwrap().len(), 3);
}

/// Integration check: the dedup by `exec_id` is DURABLE. A duplicate delivery
/// of the same execution after reopening the database (restart) must be
/// discarded just like a live one.
#[tokio::test]
async fn dedup_by_exec_id_survives_sqlite_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("dedup.sqlite")
        .to_str()
        .unwrap()
        .to_string();

    let mut fill = make_fill("token-yes", FillSide::Buy, "0.70", "10");
    fill.order_id = Some("ord-1".into());
    fill.exec_id = Some("exec-unique".into());

    {
        let state = StateStore::open(&path, Mode::Paper).await.unwrap();
        let risk = make_risk();
        let ledger = BracketLedger::new();
        assert!(fill_sink::process(&state, &risk, &ledger, fill.clone())
            .await
            .unwrap());
    }
    // Reopening: simulates the bot restart over the same DB.
    let state = StateStore::open(&path, Mode::Paper).await.unwrap();
    let risk = make_risk();
    let ledger = BracketLedger::new();
    assert!(
        !fill_sink::process(&state, &risk, &ledger, fill.clone())
            .await
            .unwrap(),
        "the duplicate delivery of exec-unique after reopening must be discarded"
    );

    // A late partial of the SAME order but with a new exec_id does count.
    let mut late = fill.clone();
    late.exec_id = Some("exec-late".into());
    late.size = "2".parse().unwrap();
    assert!(
        fill_sink::process(&state, &risk, &ledger, late)
            .await
            .unwrap(),
        "late fill with a different exec_id must be counted after reopening"
    );

    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(pos.size, dec!(12));
}

/// Integration check, legacy fallback: rows written before the migration stored
/// the execution identity in `order_id` (exec_id = NULL). A fill that reappears
/// with `exec_id = X` must match the oldest row with `order_id = X`; and a
/// fill WITHOUT exec_id keeps deduplicating by `order_id` as always.
#[tokio::test]
async fn dedup_fallback_legacy_order_id() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();

    // The "oldest" row: no exec_id, the execution went in order_id.
    let mut old = make_fill("token-yes", FillSide::Buy, "0.70", "10");
    old.order_id = Some("trade-abc".into());
    old.exec_id = None;
    assert!(fill_sink::process(&state, &risk, &ledger, old)
        .await
        .unwrap());

    // The same execution reappears post-migration with exec_id = trade-abc.
    let mut replay = make_fill("token-yes", FillSide::Buy, "0.70", "10");
    replay.order_id = Some("ord-clob".into());
    replay.exec_id = Some("trade-abc".into());
    assert!(
        !fill_sink::process(&state, &risk, &ledger, replay)
            .await
            .unwrap(),
        "exec_id must match the legacy row that stored it in order_id"
    );

    // And a fill without exec_id deduplicates by order_id as always.
    let mut dup = make_fill("token-yes", FillSide::Buy, "0.70", "10");
    dup.order_id = Some("trade-abc".into());
    dup.exec_id = None;
    assert!(!fill_sink::process(&state, &risk, &ledger, dup)
        .await
        .unwrap());

    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(pos.size, dec!(10), "only the first fill counts");
}

/// Pieces returned by `spawn_sink`: persistence tx, broadcast rx,
/// shared order_map and the task handle.
type SinkPieces = (
    tokio::sync::mpsc::UnboundedSender<Fill>,
    tokio::sync::broadcast::Receiver<Fill>,
    std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<String, StrategyId>>>,
    tokio::task::JoinHandle<()>,
);

/// Starts the real `accounting::fill_sink::run` and returns its pieces for the
/// quarantine/attribution tests in live.
fn spawn_sink(state: &std::sync::Arc<StateStore>, risk: &std::sync::Arc<RiskGuard>) -> SinkPieces {
    let (fill_persist_tx, fill_persist_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broadcast_tx, broadcast_rx) = tokio::sync::broadcast::channel(16);
    let order_map = std::sync::Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let (done_tx, _end_rx) = tokio::sync::oneshot::channel();
    let handle = tokio::spawn(fill_sink::run(
        fill_persist_rx,
        state.clone(),
        risk.clone(),
        BracketLedger::new(),
        broadcast_tx,
        order_map.clone(),
        shutdown_rx,
        done_tx,
    ));
    (fill_persist_tx, broadcast_rx, order_map, handle)
}

fn make_live_fill(token_id: &str, exec_id: &str) -> Fill {
    let mut f = make_fill(token_id, FillSide::Buy, "0.70", "10");
    f.mode = Mode::Live;
    f.exec_id = Some(exec_id.into());
    // Placeholder from handle_trade when the WS does not carry a strategy: Arb.
    // It is NOT attribution: fill_sink must resolve it or quarantine it.
    f.strategy = StrategyId::Arb;
    f
}

/// Integration check: a LIVE fill with no resolvable owner is NOT accounted for
/// nor attributed to the Arb catch-all. It goes to durable quarantine
/// (`dead_letter_fills`) and leaves a pending `unknown_attempts` entry that blocks
/// the bracket's buys until explicit reconciliation.
#[tokio::test]
async fn fill_live_without_owner_goes_to_quarantine_and_blocks_bracket() {
    let state = make_state().await;
    let risk = make_risk();
    let (tx, mut bcast, _map, _h) = spawn_sink(&state, &risk);

    let orphan_fill = make_live_fill("token-yes", "exec-orphan_fill");
    tx.send(orphan_fill).unwrap();

    // Control paper fill AFTER: its broadcast guarantees (mpsc order) that
    // the orphan has already been processed.
    tx.send(make_fill("token-no", FillSide::Buy, "0.60", "1"))
        .unwrap();
    let b = tokio::time::timeout(std::time::Duration::from_secs(2), bcast.recv())
        .await
        .expect("timeout waiting for the broadcast of the control fill")
        .unwrap();
    assert_eq!(b.token_id, "token-no");

    // The orphan: no position, no transaction of its own, no broadcast of its own.
    assert!(
        state
            .get_position("token-yes", "arb")
            .await
            .unwrap()
            .is_none(),
        "a fill without an owner must NOT be accounted as an Arb position"
    );
    assert!(state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .is_none());
    assert!(
        bcast.try_recv().is_err(),
        "the quarantined fill is not re-emitted"
    );

    // Durable trail: dead-letter + pending unknown that blocks the bracket.
    assert_eq!(state.count_unresolved_dead_letters().await.unwrap(), 1);
    assert!(
        state.has_unresolved_unknown("test-cid").await.unwrap(),
        "the bracket of the orphan fill must stay blocked for new buys"
    );
    assert!(
        !state.has_unresolved_unknown("other-cid").await.unwrap(),
        "the blocker is per bracket: other brackets remain tradable"
    );

    // Quarantine survives a manual resolution by the operator.
    state
        .mark_unknown_resolved("qfill-exec-orphan_fill", "reconciled against CLOB")
        .await
        .unwrap();
    assert!(!state.has_unresolved_unknown("test-cid").await.unwrap());
}

/// Integration check: a live fill whose `order_id` is in `order_index`
/// (written by the executor at submit; survives restart) is attributed to the
/// REAL strategy, not to the WS Arb placeholder.
#[tokio::test]
async fn fill_live_resolves_strategy_via_order_index() {
    let state = make_state().await;
    let risk = make_risk();
    state
        .record_order_index(
            "ord-mia",
            Some("attempt-7"),
            StrategyId::NearCertain,
            "test-cid",
            "token-yes",
        )
        .await
        .unwrap();

    let (tx, mut bcast, _map, _h) = spawn_sink(&state, &risk);
    let mut fill = make_live_fill("token-yes", "exec-attributed");
    fill.order_id = Some("ord-mia".into());
    tx.send(fill).unwrap();

    let b = tokio::time::timeout(std::time::Duration::from_secs(2), bcast.recv())
        .await
        .expect("the attributable fill must be broadcast")
        .unwrap();
    assert_eq!(
        b.strategy,
        StrategyId::NearCertain,
        "the strategy resolved via order_index overrides the Arb placeholder"
    );
    let pos = state
        .get_position("token-yes", "near_certain")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(pos.size, dec!(10));
    assert_eq!(state.count_unresolved_dead_letters().await.unwrap(), 0);
}

/// Integration check: the race "fill via user-WS before `submit` returns the
/// order_id and writes order_index" leaves the fill in quarantine, but it is
/// TRANSIENT. When `record_order_index` lands, the reconciler must apply the
/// fill only once, close dead-letter + unknown and unblock the bracket.
#[tokio::test]
async fn quarantine_reconciles_when_order_index_arrives_late() {
    let state = make_state().await;
    let risk = make_risk();
    let (tx, mut bcast, order_map, _h) = spawn_sink(&state, &risk);
    let (recon_tx, mut recon_rx) = tokio::sync::broadcast::channel(16);

    // 1. The WS fill arrives BEFORE submit responds: no owner yet.
    let mut fill = make_live_fill("token-yes", "exec-race");
    fill.order_id = Some("ord-late".into());
    tx.send(fill).unwrap();

    // Control fill: its broadcast guarantees that the orphan was already processed.
    tx.send(make_fill("token-no", FillSide::Buy, "0.60", "1"))
        .unwrap();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), bcast.recv())
        .await
        .expect("timeout waiting for control fill")
        .unwrap();
    assert_eq!(state.count_unresolved_dead_letters().await.unwrap(), 1);
    assert!(state.has_unresolved_unknown("test-cid").await.unwrap());

    // 2. Submit responds late: order_index can now attribute the fill.
    state
        .record_order_index(
            "ord-late",
            Some("attempt-42"),
            StrategyId::Directional,
            "test-cid",
            "token-yes",
        )
        .await
        .unwrap();

    // 3. Reconciler pass (the periodic tick would do the same).
    fill_sink::reconcile_quarantine(&state, &risk, &BracketLedger::new(), &recon_tx, &order_map)
        .await;

    // The fill was applied to the real owner, exactly once.
    let pos = state
        .get_position("token-yes", "directional")
        .await
        .unwrap()
        .expect("the reconciled fill must open a position for the real owner");
    assert_eq!(pos.size, dec!(10));
    assert!(
        state
            .get_position("token-yes", "arb")
            .await
            .unwrap()
            .is_none(),
        "the Arb placeholder must never receive the position"
    );

    // Quarantine closed: dead-letter resolved, unknown resolved, bracket free.
    assert_eq!(state.count_unresolved_dead_letters().await.unwrap(), 0);
    assert!(
        !state.has_unresolved_unknown("test-cid").await.unwrap(),
        "reconciliation must unblock the bracket"
    );

    // The reconciled fill is re-emitted to the strategies broadcast.
    let b = tokio::time::timeout(std::time::Duration::from_secs(2), recon_rx.recv())
        .await
        .expect("the reconciled fill must be broadcast")
        .unwrap();
    assert_eq!(b.strategy, StrategyId::Directional);
    assert_eq!(b.exec_id.as_deref(), Some("exec-race"));

    // 4. A second reconcile pass is idempotent: dedup by exec_id, nothing is duplicated.
    fill_sink::reconcile_quarantine(&state, &risk, &BracketLedger::new(), &recon_tx, &order_map)
        .await;
    assert_eq!(pos.size, dec!(10));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), recon_rx.recv())
            .await
            .is_err(),
        "a second reconcile pass must not re-emit anything"
    );
}

/// Integration check: if the owner still does not appear, the reconciler
/// leaves the quarantine intact and the bracket stays blocked (fail-closed).
#[tokio::test]
async fn quarantine_without_owner_stays_blocked_after_reconcile() {
    let state = make_state().await;
    let risk = make_risk();
    let (tx, mut bcast, order_map, _h) = spawn_sink(&state, &risk);
    let (recon_tx, _recon_rx) = tokio::sync::broadcast::channel(16);

    tx.send(make_live_fill("token-yes", "exec-no-owner"))
        .unwrap();
    tx.send(make_fill("token-no", FillSide::Buy, "0.60", "1"))
        .unwrap();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), bcast.recv())
        .await
        .unwrap();

    // No order_index nor journal: the reconciler cannot attribute, so nothing changes.
    fill_sink::reconcile_quarantine(&state, &risk, &BracketLedger::new(), &recon_tx, &order_map)
        .await;
    assert_eq!(state.count_unresolved_dead_letters().await.unwrap(), 1);
    assert!(state.has_unresolved_unknown("test-cid").await.unwrap());
    assert!(state
        .get_position("token-yes", "arb")
        .await
        .unwrap()
        .is_none());
}

/// `GateRejected` is an internal signal (the intent never existed as an order).
/// `process` returns it as new so the sink re-broadcasts it to strategies, but it
/// is NEVER persisted nor moves positions.
#[tokio::test]
async fn fill_sink_gate_rejected_is_not_persisted() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();

    let is_new = fill_sink::process(
        &state,
        &risk,
        &ledger,
        make_fill("token-yes", FillSide::GateRejected, "0.42", "7"),
    )
    .await
    .unwrap();
    assert!(is_new, "must be returned as new for the rebroadcast");

    assert!(
        state
            .get_position("token-yes", "near_certain")
            .await
            .unwrap()
            .is_none(),
        "GateRejected does not create a position"
    );
    assert!(
        state.recent_transactions(5, None).await.unwrap().is_empty(),
        "GateRejected leaves no transaction"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Ledger economic flows and return reconciliation.
// ═══════════════════════════════════════════════════════════════════════════

/// The durable aggregate preserves the cost to the micro: buys = cost withdrawn
/// (by sells and redeems) + open cost. Partial sell -> proration.
#[tokio::test]
async fn economic_flows_preserve_cost_to_the_micro() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();

    // Buy 10@0.60 (cost 6.0) and 10@0.40 (cost 4.0): total cost 10.0.
    let mut f1 = make_fill("token-t6", FillSide::Buy, "0.60", "10");
    f1.exec_id = Some("exec-t6-a".into());
    let mut f2 = make_fill("token-t6", FillSide::Buy, "0.40", "10");
    f2.exec_id = Some("exec-t6-b".into());
    fill_sink::process(&state, &risk, &ledger, f1)
        .await
        .unwrap();
    fill_sink::process(&state, &risk, &ledger, f2)
        .await
        .unwrap();

    // Partial sell 8@0.70 -> proceeds 5.6; cost withdrawn = 10×8/20 = 4.0;
    // pnl = +1.6; open cost = 6.0.
    let mut f3 = make_fill("token-t6", FillSide::Sell, "0.70", "8");
    f3.exec_id = Some("exec-t6-c".into());
    fill_sink::process(&state, &risk, &ledger, f3)
        .await
        .unwrap();

    let f = state.economic_flows().await.unwrap();
    assert_eq!(f.buys_usdc, dec!(10));
    assert_eq!(f.sells_usdc, dec!(5.6));
    assert_eq!(f.sells_pnl, dec!(1.6));
    assert_eq!(f.open_cost, dec!(6));
    assert_eq!(
        f.open_cost_drift(),
        rust_decimal::Decimal::ZERO,
        "buys − withdrawn − open = 0 (to the micro)"
    );
    assert_eq!(f.n_buys, 2);
    assert_eq!(f.n_sells, 1);
}

/// Winning settlement: the REDEEM records payout/pnl and leaves open cost at
/// 0; return reconciliation balances cash <-> ledger <-> pnl.
#[tokio::test]
async fn redeem_flows_and_return_reconciliation() {
    let state = make_state().await;
    let risk = make_risk();
    let ledger = BracketLedger::new();

    let mut f = make_fill("token-t6r", FillSide::Buy, "0.60", "10");
    f.exec_id = Some("exec-t6r".into());
    fill_sink::process(&state, &risk, &ledger, f).await.unwrap();
    let pos = state
        .get_position("token-t6r", "near_certain")
        .await
        .unwrap()
        .unwrap();
    state
        .apply_settlement(
            &pos,
            true,
            "YES",
            dec!(10),
            dec!(4),
            StrategyId::NearCertain,
            Some("paper-auto"),
        )
        .await
        .unwrap();

    let f = state.economic_flows().await.unwrap();
    assert_eq!(f.redeem_usdc, dec!(10));
    assert_eq!(f.redeem_pnl, dec!(4));
    assert_eq!(f.open_cost, rust_decimal::Decimal::ZERO);
    assert_eq!(f.open_cost_drift(), rust_decimal::Decimal::ZERO);

    // Initial wallet 100: −6.0 from the buy, +10 from the redeem → 104.
    let c =
        fill_sink::reconcile_economic_return(&f, dec!(100), rust_decimal::Decimal::ZERO, dec!(104));
    assert_eq!(c.cash_return, dec!(4));
    assert_eq!(c.ledger_net_flow, dec!(4));
    assert_eq!(c.pnl_ledger, dec!(4));
    assert!(c.reconciled(), "return reconcilable with the ledger: {c:?}");

    // If cash did not receive the payout (wallet = 94), reconciliation must
    // FAIL: this is the guard against settlements that do not move cash.
    let broken =
        fill_sink::reconcile_economic_return(&f, dec!(100), rust_decimal::Decimal::ZERO, dec!(94));
    assert!(!broken.reconciled());
}

/// Two quarantine rows with the same `exec_id`, plus a re-delivery of the fill
/// via the NORMAL path before reconciliation: the execution is applied only once
/// (ledger/positions) and ONE broadcast is emitted; both rows are closed without
/// the reconciler re-emitting the already-settled fill.
///
/// With the earlier code (`Ok(_) => { ... resolve ...; send(fill) }`), the
/// reconciliation of the second row returned `Ok(false)` (dedup by exec_id)
/// but still emitted the Fill a second time. Strategies without their own dedup
/// (`parity_maker::on_fill`, etc.) counted it twice in memory.
///
/// Declared limit: a failure of `resolve_quarantined_fill` after the write is not
/// injected here; it would require a faulty DB under `StateStore` (private pool,
/// outside this test's scope). The emit-before-close ordering is designed so that
/// this failure neither loses nor duplicates the broadcast, but that path has no
/// automated evidence.
#[tokio::test]
async fn reconcile_does_not_re_emit_fill_already_settled_via_normal_path() {
    let state = make_state().await;
    let risk = make_risk();
    // Unlike `spawn_sink`, here we need the broadcast Sender to pass it to the
    // reconciler and measure the emissions of the SAME channel.
    let (fill_persist_tx, fill_persist_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broadcast_tx, mut broadcast_rx) = tokio::sync::broadcast::channel(16);
    let order_map =
        std::sync::Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let (done_tx, _done_rx) = tokio::sync::oneshot::channel();
    let handle = tokio::spawn(fill_sink::run(
        fill_persist_rx,
        state.clone(),
        risk.clone(),
        BracketLedger::new(),
        broadcast_tx.clone(),
        order_map.clone(),
        shutdown_rx,
        done_tx,
    ));

    // 1. Two deliveries of the SAME live fill without owner -> two quarantine
    //    rows (the table does not dedup by exec_id), a single unknown_attempt.
    let mut fill = make_live_fill("token-yes", "exec-dup");
    fill.order_id = Some("ord-dup".into());
    fill_persist_tx.send(fill.clone()).unwrap();
    fill_persist_tx.send(fill.clone()).unwrap();
    // Barrier: the control fill's broadcast guarantees (mpsc order) that both
    // quarantines were already written.
    fill_persist_tx
        .send(make_fill("token-no", FillSide::Buy, "0.60", "1"))
        .unwrap();
    let barrier = tokio::time::timeout(std::time::Duration::from_secs(2), broadcast_rx.recv())
        .await
        .expect("timeout waiting for the control fill")
        .unwrap();
    assert_eq!(barrier.token_id, "token-no");
    assert_eq!(
        state.count_unresolved_dead_letters().await.unwrap(),
        2,
        "the two ownerless deliveries produce two quarantine rows"
    );
    assert!(state.has_unresolved_unknown("test-cid").await.unwrap());

    // 2. The order_index lands and the fill is re-delivered via the normal path:
    //    it is applied and emitted exactly once.
    state
        .record_order_index(
            "ord-dup",
            Some("att-dup"),
            StrategyId::Directional,
            "test-cid",
            "token-yes",
        )
        .await
        .unwrap();
    fill_persist_tx.send(fill.clone()).unwrap();
    let b = tokio::time::timeout(std::time::Duration::from_secs(2), broadcast_rx.recv())
        .await
        .expect("the re-delivery via the normal path must be emitted")
        .unwrap();
    assert_eq!(b.exec_id.as_deref(), Some("exec-dup"));
    assert_eq!(b.strategy, StrategyId::Directional);

    // 3. Reconciliation over the SAME broadcast channel: both rows are
    //    resolved without re-emission.
    fill_sink::reconcile_quarantine(
        &state,
        &risk,
        &BracketLedger::new(),
        &broadcast_tx,
        &order_map,
    )
    .await;
    assert_eq!(
        state.count_unresolved_dead_letters().await.unwrap(),
        0,
        "both quarantine rows are resolved"
    );
    assert!(
        !state.has_unresolved_unknown("test-cid").await.unwrap(),
        "the bracket is unblocked after closing the rows"
    );
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(150), broadcast_rx.recv())
            .await
            .is_err(),
        "the already-settled fill must NOT be re-broadcast by reconciliation"
    );

    // 4. Accounting only once: position 10 (not 20/30) and one transaction for
    //    the fill + one for the control, with no third one from the re-delivery.
    let pos = state
        .get_position("token-yes", "directional")
        .await
        .unwrap()
        .expect("position of the real owner");
    assert_eq!(pos.size, dec!(10));
    assert!(state
        .get_position("token-yes", "arb")
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        state.recent_transactions(10, None).await.unwrap().len(),
        2,
        "control + fill: neither the re-delivery nor the reconciliation adds a third"
    );
    assert_eq!(state.economic_flows().await.unwrap().n_buys, 2);

    drop(fill_persist_tx);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
}
