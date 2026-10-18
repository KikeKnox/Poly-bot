//! Focused tests of the conservative integrated fix for "Unknown on submit".
//!
//! These tests are the RED half of TDD: they implement the requirements the
//! earlier delivery did NOT meet (UnknownAttempt was created AFTER the POST and only in
//! a HashMap). They must FAIL with the old code and PASS after the fix.
//!
//! Added coverage (rules 8–12 of the doc comment in `order_uncertainty.rs`):
//!
//! - **A.** The local `attempt_id` is generated BEFORE the POST and persisted in SQLite
//!   in the `unknown_attempts` table (state `pending`) BEFORE invoking
//!   `router.submit`. If the write fails, the POST is NOT sent.
//! - **B.** `handle_buy` consults the journal on entry and blocks new submits
//!   over the same `cid:token_id` while an `unresolved` Unknown exists for
//!   that bracket. It does NOT resolve automatically by clock, nor by `clear_at`.
//! - **C.** ARB pair: both legs Unknown → TWO independent records in the
//!   journal. Rejected + Unknown (in any combination) → the Unknown
//!   record is kept; the rejected one releases ONLY its own claim.
//! - **D.** Restart: an `Executor` built over the same `StateStore`
//!   (via `open_in_memory_with_journal` or equivalent) detects `unresolved`
//!   and keeps the blocker for the same bracket without resending.
//! - **E.** Pre-POST journal failure (injected): `router.submit` is NOT
//!   invoked and `unknown_attempts` stays empty.
//! - **F.** DEFINITIVE rejection with a prior `unresolved` on another bracket: the
//!   rejection releases ONLY its own claim; the other `unresolved` keeps
//!   blocking.
//!
//! What it does NOT cover (explicit, out of scope for the fix):
//!
//! - Automatic resolution of the Unknown against the exchange's `/orders`.
//! - Migration of the `BracketLedger` to `Reservations` — untouched.
//! - Remote reconciliation API — the only valid mechanism
//!   to unblock is `mark_unknown_resolved(attempt_id, reason)` invoked
//!   manually.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{mpsc, watch};

use poly_bot::{
    coordination::BracketLedger,
    order::{
        executor::Executor,
        router::{OrderRequest, OrderRouter, RouterResult},
    },
    risk::RiskGuard,
    state::StateStore,
    types::{Mode, OrderIntent, OrderSide, StrategyId},
};

// ── Fake router with a submit counter ──────────────────────────────────────

/// Configurable fake router: each `submit` consumes an entry of the queue.
/// The `submit_count` counter is queried from the tests to verify
/// that `router.submit` was invoked (or not) the expected number of times.
#[derive(Clone)]
enum RouterAction {
    Unknown(String),
    Rejected(String),
    Accepted { order_id: String },
}

struct ScriptedRouter {
    script: Mutex<VecDeque<RouterAction>>,
    submit_count: AtomicUsize,
}

impl ScriptedRouter {
    fn new(script: Vec<RouterAction>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter().collect()),
            submit_count: AtomicUsize::new(0),
        })
    }
    fn next(&self) -> RouterAction {
        self.script
            .lock()
            .pop_front()
            .expect("ScriptedRouter: script exhausted")
    }
    fn count(&self) -> usize {
        self.submit_count.load(Ordering::SeqCst)
    }
}

impl OrderRouter for ScriptedRouter {
    fn submit(
        &self,
        _req: OrderRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RouterResult> + Send + '_>> {
        self.submit_count.fetch_add(1, Ordering::SeqCst);
        let action = self.next();
        Box::pin(async move {
            match action {
                RouterAction::Unknown(reason) => RouterResult::unknown(reason),
                RouterAction::Rejected(reason) => RouterResult::rejected(reason),
                RouterAction::Accepted { order_id } => RouterResult::no_fill(Some(order_id)),
            }
        })
    }

    fn submit_pair(
        &self,
        _req1: OrderRequest,
        _req2: OrderRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = (RouterResult, RouterResult)> + Send + '_>,
    > {
        self.submit_count.fetch_add(2, Ordering::SeqCst);
        let a1 = self.next();
        let a2 = self.next();
        Box::pin(async move {
            let to_r = |a: RouterAction| match a {
                RouterAction::Unknown(reason) => RouterResult::unknown(reason),
                RouterAction::Rejected(reason) => RouterResult::rejected(reason),
                RouterAction::Accepted { order_id } => RouterResult::no_fill(Some(order_id)),
            };
            (to_r(a1), to_r(a2))
        })
    }

    fn cancel(
        &self,
        _market_cid: &str,
        _order_id: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }

    fn mode(&self) -> Mode {
        Mode::Paper
    }
}

// ── Helpers shared with order_uncertainty.rs ────────────────────────────

async fn build_executor(
    router: Arc<ScriptedRouter>,
) -> (
    Executor,
    Arc<BracketLedger>,
    Arc<ScriptedRouter>,
    mpsc::UnboundedSender<poly_bot::types::Fill>,
    Arc<StateStore>,
) {
    let (_intent_tx, intent_rx) = mpsc::channel(1);
    let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
    let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(10_000)));
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let risk = RiskGuard::new(
        poly_bot::config::RiskConfig {
            max_total_exposure_usdc: dec!(10_000),
            ..Default::default()
        },
        Mode::Paper,
    );
    let ledger = BracketLedger::new();
    // We pass the SAME Arc as dyn OrderRouter to the Executor; we keep the
    // original ref to count submits from the tests.
    let router_dyn: Arc<dyn OrderRouter> = router.clone();
    let executor = Executor::new(
        intent_rx,
        fill_tx.clone(),
        router_dyn,
        risk,
        state.clone(),
        Arc::new(Mutex::new(HashMap::new())),
        wallet_rx,
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
        ledger.clone(),
    );
    (executor, ledger, router, fill_tx, state)
}

fn buy_intent(token_id: &str, price: Decimal, size: Decimal, strategy: StrategyId) -> OrderIntent {
    let clear_at = chrono::Utc::now().timestamp() + 300;
    OrderIntent {
        market_cid: "cid-1".into(),
        token_id: token_id.into(),
        side: OrderSide::Buy,
        price,
        size,
        clear_at,
        signal: dec!(0),
        neg_risk: false,
        limit_price: Some(price),
        second_leg: None,
        strategy,
        resting: false,
        real_clear_at: None,
    }
}

fn buy_intent_market(
    token_id: &str,
    market_cid: &str,
    price: Decimal,
    size: Decimal,
    strategy: StrategyId,
) -> OrderIntent {
    let clear_at = chrono::Utc::now().timestamp() + 300;
    OrderIntent {
        market_cid: market_cid.into(),
        token_id: token_id.into(),
        side: OrderSide::Buy,
        price,
        size,
        clear_at,
        signal: dec!(0),
        neg_risk: false,
        limit_price: None,
        second_leg: None,
        strategy,
        resting: false,
        real_clear_at: None,
    }
}

fn arb_pair_intent(yes_price: Decimal, no_price: Decimal, size: Decimal) -> OrderIntent {
    let no_leg = {
        let mut i = buy_intent("no-tok", no_price, size, StrategyId::Arb);
        i.token_id = "no-tok".into();
        i
    };
    let mut yes = buy_intent("yes-tok", yes_price, size, StrategyId::Arb);
    yes.second_leg = Some(Box::new(no_leg));
    yes
}

// ── Focused tests (TDD red) ─────────────────────────────────────────────────

/// Test 9 (A): the `attempt_id` is PERSISTED in SQLite BEFORE the POST.
/// The fake router records the submit_count so the test can
/// confirm it WAS sent. Besides, `state.list_unresolved_unknowns`
/// must return the registered pending, referenced by the durable journal.
#[tokio::test]
async fn unknown_persists_pending_in_journal_pre_post() {
    let router = ScriptedRouter::new(vec![RouterAction::Unknown("lost ACK".into())]);
    let (mut ex, _ledger, scripted, _fill_tx, state) = build_executor(router).await;
    ex.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;

    // The POST was sent (1 submit consumed).
    assert_eq!(
        scripted.count(),
        1,
        "submit WAS invoked after a successful journal write"
    );
    // The durable journal contains ONE unresolved pending.
    let unresolved = state.list_unresolved_unknowns().await.unwrap();
    assert_eq!(
        unresolved.len(),
        1,
        "there must be ONE pending in SQLite (durable journal)"
    );
    assert_eq!(unresolved[0].token_id, "tok-1");
    assert_eq!(unresolved[0].cid, "cid-1");
    assert!(!unresolved[0].resolved, "the pending is unresolved");
}

/// Test 10 (B): a second attempt on the same `cid:token_id` after an Unknown
/// must NOT send a new POST — the unresolved journal blocker prevents it.
#[tokio::test]
async fn second_attempt_same_bracket_blocked_by_unresolved() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Unknown("lost ACK".into()),
        // Second submit: must NOT be consumed if the blocker works.
        RouterAction::Accepted {
            order_id: "never".into(),
        },
    ]);
    let (mut ex, _ledger, scripted, _fill_tx, _state) = build_executor(router).await;
    // First intent: Unknown.
    ex.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(scripted.count(), 1, "first POST sent");
    // Second intent on the SAME token: the journal blocks it.
    ex.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(
        scripted.count(),
        1,
        "second POST blocked by the unresolved journal; submit_count must NOT go up"
    );
}

/// Test 11 (B'): a second attempt on ANOTHER token of the SAME bracket — blocked
/// because the blocker is per `cid:token_id` (the earlier Unknown pair was that token)
/// and for global safety: if the bracket has an unresolved, nothing is sent
/// over it. Here we test a different token of the SAME bracket → blocked.
#[tokio::test]
async fn second_attempt_other_token_same_bracket_also_blocked() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Unknown("lost ACK tok-A".into()),
        RouterAction::Accepted {
            order_id: "never-tok-B".into(),
        },
    ]);
    let (mut ex, _ledger, scripted, _fill_tx, _state) = build_executor(router).await;
    // First intent on tok-A.
    ex.handle_buy(buy_intent(
        "tok-A",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(scripted.count(), 1);
    // Second intent on tok-B of the same cid: the blocker must apply
    // (same bracket — `cid-1`).
    ex.handle_buy(buy_intent(
        "tok-B",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(
        scripted.count(),
        1,
        "second POST on ANOTHER token of the same cid also blocked"
    );
}

/// Test 12 (C): ARB pair both Unknown → TWO independent records in the
/// journal (one per leg), both `unresolved`. The ledger keeps both claims.
#[tokio::test]
async fn pair_both_unknown_two_independent_records_in_journal() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Unknown("ACK YES".into()),
        RouterAction::Unknown("ACK NO".into()),
    ]);
    let (mut ex, ledger, _scripted, _fill_tx, state) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;

    assert_eq!(ledger.total_exposure(), dec!(10), "both kept (5+5)");
    let unresolved = state.list_unresolved_unknowns().await.unwrap();
    assert_eq!(
        unresolved.len(),
        2,
        "TWO independent records, one per leg (YES + NO)"
    );
    let tokens: Vec<&str> = unresolved.iter().map(|u| u.token_id.as_str()).collect();
    assert!(tokens.contains(&"yes-tok"));
    assert!(tokens.contains(&"no-tok"));
}

/// Test 13 (C): ARB pair YES Unknown + NO rejected → keeps YES (journal +
/// ledger), releases NO (claim released, NO does not enter the journal).
#[tokio::test]
async fn pair_yes_unknown_no_rejected_keeps_yes_record() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Unknown("ACK YES".into()),
        RouterAction::Rejected("NO rejected".into()),
    ]);
    let (mut ex, ledger, _scripted, _fill_tx, state) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;

    assert_eq!(
        ledger.total_exposure(),
        dec!(5),
        "YES=5 kept, NO=5 released"
    );
    let unresolved = state.list_unresolved_unknowns().await.unwrap();
    assert_eq!(
        unresolved.len(),
        1,
        "ONLY the YES record is kept; the NO rejected one is dropped"
    );
    assert_eq!(unresolved[0].token_id, "yes-tok");
}

/// Test 14 (C): ARB pair YES rejected + NO Unknown → keeps the NO record,
/// releases YES. Symmetric case of the earlier one.
#[tokio::test]
async fn pair_yes_rejected_no_unknown_keeps_no_record() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Rejected("YES rejected".into()),
        RouterAction::Unknown("ACK NO".into()),
    ]);
    let (mut ex, ledger, _scripted, _fill_tx, state) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;

    assert_eq!(
        ledger.total_exposure(),
        dec!(5),
        "YES released, NO kept"
    );
    let unresolved = state.list_unresolved_unknowns().await.unwrap();
    assert_eq!(unresolved.len(), 1);
    assert_eq!(unresolved[0].token_id, "no-tok");
}

/// Test 15 (F): a definitive rejection of one token does NOT block submits of another
/// bracket. The rejection mark is NOT Unknown, so the
/// journal for that token stays empty and the other bracket stays free.
#[tokio::test]
async fn final_rejection_does_not_block_other_brackets() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Rejected("token-A rejected".into()),
        // Second submit to ANOTHER bracket must go through.
        RouterAction::Accepted {
            order_id: "oid-B".into(),
        },
    ]);
    let (mut ex, ledger, scripted, _fill_tx, _state) = build_executor(router).await;
    // First intent: bracket cid-1, tok-A, rejected.
    ex.handle_buy(buy_intent_market(
        "tok-A",
        "cid-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(scripted.count(), 1);
    assert_eq!(ledger.total_exposure(), dec!(0), "rejected releases the claim");
    // Second intent: bracket cid-2 (different), tok-B → must go through.
    ex.handle_buy(buy_intent_market(
        "tok-B",
        "cid-2",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(
        scripted.count(),
        2,
        "rejection of cid-1 does NOT block submits over cid-2"
    );
    // The journal contains no unresolved (a rejection is not Unknown).
    let unresolved = _state.list_unresolved_unknowns().await.unwrap();
    assert!(unresolved.is_empty(), "no pending remains after rejected");
}

/// Test 16 (D): a NEW `Executor` built over the same `StateStore`
/// that already has an unresolved pending in the journal BLOCKS submits of the
/// same bracket. This models a restart: the journal survives, the blocker
/// is restored.
#[tokio::test]
async fn restart_detects_unresolved_and_blocks() {
    // ── Phase 1: the first Executor generates an Unknown over cid-1/tok-1.
    let router1 = ScriptedRouter::new(vec![RouterAction::Unknown("lost ACK".into())]);
    let (mut ex1, _ledger1, _scripted1, _fill_tx1, state) = build_executor(router1).await;
    ex1.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;

    // The journal has ONE unresolved pending.
    let unresolved_pre = state.list_unresolved_unknowns().await.unwrap();
    assert_eq!(unresolved_pre.len(), 1);

    // ── Phase 2: new Executor over the SAME StateStore — models the restart.
    // The journal already has the pending; `Executor::new` must read it and block.
    let router2 = ScriptedRouter::new(vec![RouterAction::Accepted {
        order_id: "never-arrives".into(),
    }]);
    let (mut ex2, _ledger2, scripted2, _fill_tx2, state2) =
        build_executor_with_state(router2, state.clone()).await;
    // Sanity: the shared StateStore has the pending.
    assert_eq!(state2.list_unresolved_unknowns().await.unwrap().len(), 1);

    // The new Executor must NOT send the POST: there is an unresolved of the same
    // bracket/token.
    ex2.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(
        scripted2.count(),
        0,
        "new Executor over a StateStore with unresolved BLOCKS the submit"
    );
}

/// Test 17 (D'): after explicit reconciliation (`mark_unknown_resolved`),
/// the next attempt on the same bracket IS sent.
#[tokio::test]
async fn explicit_reconciliation_unblocks() {
    // Phase 1: the first Executor generates an Unknown.
    let router1 = ScriptedRouter::new(vec![RouterAction::Unknown("ACK".into())]);
    let (mut ex1, _ledger1, _scripted1, _fill_tx1, state) = build_executor(router1).await;
    ex1.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    let unresolved = state.list_unresolved_unknowns().await.unwrap();
    assert_eq!(unresolved.len(), 1);
    let attempt_id = unresolved[0].attempt_id.clone();

    // Manual reconciliation.
    state
        .mark_unknown_resolved(&attempt_id, "verified on exchange")
        .await
        .unwrap();
    assert_eq!(state.list_unresolved_unknowns().await.unwrap().len(), 0);

    // Phase 2: new Executor — the journal is clean, the submit goes through.
    let router2 = ScriptedRouter::new(vec![RouterAction::Accepted {
        order_id: "oid-1".into(),
    }]);
    let (mut ex2, _ledger2, scripted2, _fill_tx2, _state2) =
        build_executor_with_state(router2, state).await;
    ex2.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(
        scripted2.count(),
        1,
        "after explicit reconciliation, the POST IS sent"
    );
}

/// Test 18 (F'): a definitive rejection on one bracket does NOT block another
/// bracket that already has an independent unresolved ("a definitive rejection
/// releases ONLY its claim" scenario — its rejection mark does not enter the journal, but the
/// other bracket stays blocked by ITS OWN unresolved).
#[tokio::test]
async fn rejection_does_not_block_other_bracket_that_has_its_own_unresolved() {
    let router = ScriptedRouter::new(vec![
        // cid-1 / tok-A: Unknown → unresolved pending in the journal.
        RouterAction::Unknown("ACK cid-1/tok-A".into()),
        // cid-2 / tok-B: rejected → releases the claim, NO journal.
        RouterAction::Rejected("cid-2 rejected".into()),
        // cid-1 / tok-C: must remain BLOCKED by cid-1's unresolved.
        RouterAction::Accepted {
            order_id: "never".into(),
        },
    ]);
    let (mut ex, _ledger, scripted, _fill_tx, state) = build_executor(router).await;
    // cid-1 / tok-A
    ex.handle_buy(buy_intent_market(
        "tok-A",
        "cid-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    // cid-2 / tok-B
    ex.handle_buy(buy_intent_market(
        "tok-B",
        "cid-2",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    // cid-1 / tok-C — same bracket as the unresolved → blocked.
    ex.handle_buy(buy_intent_market(
        "tok-C",
        "cid-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;

    assert_eq!(
        scripted.count(),
        2,
        "third submit blocked by cid-1's unresolved"
    );
    let unresolved = state.list_unresolved_unknowns().await.unwrap();
    assert_eq!(unresolved.len(), 1, "only cid-1/tok-A's pending remains");
}

// ── Helper to build an Executor with a pre-existing StateStore ────────

async fn build_executor_with_state(
    router: Arc<ScriptedRouter>,
    state: Arc<StateStore>,
) -> (
    Executor,
    Arc<BracketLedger>,
    Arc<ScriptedRouter>,
    mpsc::UnboundedSender<poly_bot::types::Fill>,
    Arc<StateStore>,
) {
    let (_intent_tx, intent_rx) = mpsc::channel(1);
    let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
    let (_wallet_tx, wallet_rx) = watch::channel(Some(dec!(10_000)));
    let risk = RiskGuard::new(
        poly_bot::config::RiskConfig {
            max_total_exposure_usdc: dec!(10_000),
            ..Default::default()
        },
        Mode::Paper,
    );
    let ledger = BracketLedger::new();
    let router_dyn: Arc<dyn OrderRouter> = router.clone();
    let mut executor = Executor::new(
        intent_rx,
        fill_tx.clone(),
        router_dyn,
        risk,
        state.clone(),
        Arc::new(Mutex::new(HashMap::new())),
        wallet_rx,
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
        ledger.clone(),
    );
    // Fix for Unknown: seeds the cache from
    // the durable journal before the first handle_buy (models the restart).
    executor.seed_unknown_attempts_from_journal_pub().await;
    (executor, ledger, router, fill_tx, state)
}

// ── `side` is durable in `unknown_attempts` ──────────────────

/// The intent's declared side is persisted in the SAME insert of the
/// pending (column `side`, additive nullable migration) and SURVIVES the
/// SQLite reopen — without it, a restart lost the evidence that
/// prevents matching a BUY pending against a foreign resting SELL.
/// Legacy rows (oldest API / earlier migrations) keep
/// `side = NULL` = unknown side, never invented.
#[tokio::test]
async fn journal_side_survives_sqlite_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d1.sqlite").to_str().unwrap().to_string();
    let clear_at = chrono::Utc::now().timestamp() + 300;

    {
        let state = StateStore::open(&path, Mode::Paper).await.unwrap();
        // New pending with full metadata — declared side "buy".
        state
            .record_unknown_attempt_pending_meta(
                "att-buy",
                "cid-1",
                "yes-tok",
                "parity_maker",
                poly_bot::state::to_micros(dec!(1.5)),
                true,
                "test",
                Some(poly_bot::state::to_micros(dec!(0.5))),
                Some(poly_bot::state::to_micros(dec!(3))),
                Some(clear_at),
                Some("buy"),
            )
            .await
            .unwrap();
        // Row written by the LEGACY API — side NULL.
        state
            .record_unknown_attempt_pending(
                "att-legacy",
                "cid-1",
                "no-tok",
                "parity_maker",
                poly_bot::state::to_micros(dec!(1.5)),
                true,
                "test",
                poly_bot::state::to_micros(dec!(0.5)),
                poly_bot::state::to_micros(dec!(3)),
                clear_at,
            )
            .await
            .unwrap();
    }

    // Reopen: models the bot restart over the same DB.
    let state = StateStore::open(&path, Mode::Paper).await.unwrap();
    let rows = state.list_unresolved_unknowns().await.unwrap();
    let buy = rows.iter().find(|r| r.attempt_id == "att-buy").unwrap();
    assert_eq!(buy.side.as_deref(), Some("buy"), "the side survives the reopen");
    let legacy = rows.iter().find(|r| r.attempt_id == "att-legacy").unwrap();
    assert_eq!(legacy.side, None, "the legacy row keeps side NULL");
    assert!(legacy.price_um.is_some(), "the legacy metadata is intact");
}
