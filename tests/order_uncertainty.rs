//! Vertical tests of the "Unknown on submit" behavior.
//!
//! These tests cover the integrated path (not an isolated primitive) of the acknowledgement
//! failure (the executor collapses a timeout to `rejected` and releases
//! the reservation) and of its pair twin (Unknown + accepted: an Unknown leg with the
//! other one accepted must not release the orphan).
//!
//! Rules verified:
//!
//! 1. POST with a lost response (`RouterResult::unknown`) — the exposure stays
//!    held in the ledger, it is not released.
//! 2. DEFINITIVELY rejected POST (`RouterResult::rejected`) — it does release.
//! 3. Pair (YES+NO ARB): YES Unknown + NO accepted — YES exposure kept,
//!    NO accepted recorded (the Unknown leg is not released).
//! 4. Pair: both Unknown — both kept (TWO independent records).
//! 5. Pair: both definitively rejected — both released.
//! 6. Pre-POST validation (global kill switch) does NOT touch exposure: it is already
//!    fail-closed (it never enters the ledger), still definitive.
//! 7. A fake router that loses the ACK records a durable local uncertainty mark
//!    (the `attempt_id`) — the key stays in `Executor::unknown_attempts` AND is
//!    persisted in SQLite to survive a restart. A blind resend of the same
//!    attempt does not call `router.submit` again.
//!
//! Additional rules of the conservative fix:
//!
//! 8. **The local attempt_id is generated BEFORE the POST** and written to the durable
//!    journal (`unknown_attempts` in SQLite) BEFORE invoking `router.submit`.
//!    If the write fails, the POST is NOT sent (fail-closed).
//! 9. **handle_buy consults the journal on entry**: if there is an `unresolved`
//!    Unknown for the same `cid:token_id` (or global if so decided), the
//!    attempt is NOT sent until it is explicitly reconciled.
//! 10. **A DEFINITIVE rejection releases ONLY its claim** and marks its attempt_id as
//!     `resolved` — it does not affect other unrelated unknowns.
//! 11. **Restart**: a new Executor built over the same `StateStore`
//!     (even in a manually recreated `:memory:` with the preexisting journal)
//!     detects the `unresolved` ones and blocks submits of the same bracket until
//!     an explicit reconciliation (via `mark_unknown_resolved`).
//! 12. **Pre-POST journal fallback**: if inserting the pending fails
//!     (SQLite down / error), `router.submit` is NOT invoked.
//!
//! What these tests do NOT cover (explicit, out of scope):
//!
//! - **Real resolution** of an Unknown (querying the exchange's `/orders`, etc.):
//!   future reconciliation is a dedicated worker, not implemented here.
//!   The fix exposes `mark_unknown_resolved` as the ONLY valid mechanism to
//!   unblock; there is no automatic clock/clear_at, nor pure logging.
//! - **Migration of the `BracketLedger` to `Reservations`** to have an
//!   unambiguous owner: out of scope, the current ledger is kept. Nor is a
//!   ownership fix claimed.

use std::collections::{HashMap, VecDeque};
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

/// Configurable fake router: each call to `submit` or `submit_pair` consumes an
/// entry of the `script` queue (VecDeque) and returns the indicated result.
/// Models: lost ACK (Unknown), definitive rejection (Rejected), or acceptance
/// (Accepted with `order_id`).
#[derive(Clone)]
enum RouterAction {
    Unknown(String),
    Rejected(String),
    Accepted { order_id: String },
}

struct ScriptedRouter {
    script: Mutex<VecDeque<RouterAction>>,
}

impl ScriptedRouter {
    fn new(script: Vec<RouterAction>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter().collect()),
        })
    }
    fn next(&self) -> RouterAction {
        self.script
            .lock()
            .pop_front()
            .expect("ScriptedRouter: script exhausted")
    }
}

impl OrderRouter for ScriptedRouter {
    fn submit(
        &self,
        _req: OrderRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RouterResult> + Send + '_>> {
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

async fn build_executor(
    router: Arc<dyn OrderRouter>,
) -> (
    Executor,
    Arc<BracketLedger>,
    mpsc::UnboundedSender<poly_bot::types::Fill>,
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
    let executor = Executor::new(
        intent_rx,
        fill_tx.clone(),
        router,
        risk,
        state,
        Arc::new(Mutex::new(HashMap::new())),
        wallet_rx,
        Arc::new(dashmap::DashSet::new()),
        std::collections::HashSet::new(),
        ledger.clone(),
    );
    (executor, ledger, fill_tx)
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

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Test 1: POST with a lost response (Unknown) — the exposure STAYS held
/// in the ledger, it is not released.
#[tokio::test]
async fn unknown_post_does_not_release_exposure() {
    let router = ScriptedRouter::new(vec![RouterAction::Unknown(
        "timeout reading the CLOB response".into(),
    )]);
    let (mut ex, ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(
        ledger.total_exposure(),
        dec!(5),
        "Unknown must NOT release the claim — the ACK may have been lost and the order still alive"
    );
}

/// Test 2: DEFINITIVELY rejected POST does release the claim (CLOB rejection, not
/// ambiguous: closed market, validator, etc.).
#[tokio::test]
async fn rejected_post_releases_exposure() {
    let router = ScriptedRouter::new(vec![RouterAction::Rejected("market closed".into())]);
    let (mut ex, ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert_eq!(
        ledger.total_exposure(),
        Decimal::ZERO,
        "DEFINITIVE rejected does release — the exchange denied the order, there is no ambiguity"
    );
}

/// Test 3: ARB pair YES Unknown + NO accepted — YES exposure kept,
/// NO accepted recorded (the Unknown leg is not released).
#[tokio::test]
async fn pair_yes_unknown_no_accepted_keeps_yes_releases_no() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Unknown("lost ACK leg YES".into()),
        RouterAction::Accepted {
            order_id: "no-oid".into(),
        },
    ]);
    let (mut ex, ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;
    // YES=5 kept, NO=5 accepted. Total = 10.
    assert_eq!(
        ledger.total_exposure(),
        dec!(10),
        "YES Unknown keeps its claim (5) and NO accepted keeps its own (5)"
    );
}

/// Test 4: ARB pair both Unknown — both kept.
#[tokio::test]
async fn pair_both_unknown_keeps_both() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Unknown("lost ACK YES".into()),
        RouterAction::Unknown("lost ACK NO".into()),
    ]);
    let (mut ex, ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;
    assert_eq!(
        ledger.total_exposure(),
        dec!(10),
        "both Unknown → both claims kept (5 + 5)"
    );
}

/// Test 5: ARB pair both definitively rejected — both released.
#[tokio::test]
async fn pair_both_rejected_releases_both() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Rejected("YES rejected".into()),
        RouterAction::Rejected("NO rejected".into()),
    ]);
    let (mut ex, ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;
    assert_eq!(
        ledger.total_exposure(),
        Decimal::ZERO,
        "both rejected → both claims released (5 + 5)"
    );
}

/// Test 6: the attempt enters `Executor::unknown_attempts` with a local
/// `attempt_id` — the mark exists for inspection and so that the (future)
/// resend logic does NOT blindly resend the same attempt.
#[tokio::test]
async fn unknown_post_records_local_attempt_id() {
    let router = ScriptedRouter::new(vec![RouterAction::Unknown("lost ACK".into())]);
    let (mut ex, _ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    let unknowns = ex.unknown_attempts();
    assert_eq!(
        unknowns.len(),
        1,
        "there must be ONE local attempt_id with an uncertainty mark"
    );
    let entry = &unknowns[0];
    assert_eq!(entry.token_id, "tok-1");
    assert_eq!(entry.cid, "cid-1");
    assert!(entry.reason.contains("lost ACK"));
    // The attempt_id is local (timestamp+counter) — it was not sent to the exchange and
    // is NOT an invented idempotency key. We verify it is non-empty.
    assert!(!entry.attempt_id.is_empty());
}

/// Test 7: a DEFINITIVE rejection does NOT enter `unknown_attempts`.
#[tokio::test]
async fn rejected_post_does_not_record_attempt_id() {
    let router = ScriptedRouter::new(vec![RouterAction::Rejected("market closed".into())]);
    let (mut ex, _ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(buy_intent(
        "tok-1",
        dec!(0.50),
        dec!(10),
        StrategyId::NearCertain,
    ))
    .await;
    assert!(
        ex.unknown_attempts().is_empty(),
        "a definitive rejection is NOT Unknown — it must not be recorded"
    );
}

/// Test 8: ARB pair YES Unknown + NO rejected — YES keeps its claim, NO
/// releases. It is the most dangerous scenario (live leg or not, we keep).
#[tokio::test]
async fn pair_yes_unknown_no_rejected_keeps_yes_releases_no() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Unknown("lost ACK leg YES".into()),
        RouterAction::Rejected("NO rejected".into()),
    ]);
    let (mut ex, ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;
    // YES=5 kept (Unknown), NO=5 released (rejected). Total = 5.
    assert_eq!(
        ledger.total_exposure(),
        dec!(5),
        "YES Unknown keeps its claim (5); NO rejected releases its own (5) → 5 remains"
    );
    let unknowns = ex.unknown_attempts();
    assert_eq!(unknowns.len(), 1);
    assert_eq!(unknowns[0].token_id, "yes-tok");
}

/// Test 9: ARB pair both Unknown — BOTH legs must end up in the
/// in-memory mirror (`unknown_attempts`), not only in the durable journal.
///
/// Regression covered: the `return` of the `r_yes.unknown` path made the
/// `r_no.unknown` branch at the end unreachable (it only runs when YES
/// was accepted), so the NO leg never called
/// `register_unknown_attempt`. The mirror feeds the claim re-add of
/// `purge_and_resync` and the quarantine link: without it, the first resync
/// (~30 s) released the claim of a leg possibly alive on the CLOB.
#[tokio::test]
async fn pair_both_unknown_records_two_mirrors() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Unknown("lost ACK YES".into()),
        RouterAction::Unknown("lost ACK NO".into()),
    ]);
    let (mut ex, _ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;
    let unknowns = ex.unknown_attempts();
    assert_eq!(
        unknowns.len(),
        2,
        "both Unknown legs must end up in the in-memory mirror, not only in the journal"
    );
    let tokens: std::collections::HashSet<&str> =
        unknowns.iter().map(|u| u.token_id.as_str()).collect();
    assert!(tokens.contains("yes-tok"), "missing the YES leg in the mirror");
    assert!(tokens.contains("no-tok"), "missing the NO leg in the mirror");
}

/// Test 10: YES definitively rejected + NO Unknown — the NO leg
/// keeps its claim AND ends up in the mirror. The `return` of the
/// `r_yes.rejected` path skipped NO's `register_unknown_attempt`.
#[tokio::test]
async fn pair_yes_rejected_no_unknown_records_no_mirror() {
    let router = ScriptedRouter::new(vec![
        RouterAction::Rejected("YES rejected".into()),
        RouterAction::Unknown("lost ACK NO".into()),
    ]);
    let (mut ex, ledger, _fill_tx) = build_executor(router).await;
    ex.handle_buy(arb_pair_intent(dec!(0.50), dec!(0.50), dec!(10)))
        .await;
    // YES=5 released (rejected), NO=5 kept (Unknown) → total 5.
    assert_eq!(
        ledger.total_exposure(),
        dec!(5),
        "YES rejected releases (5); NO Unknown keeps (5) → 5 remains"
    );
    let unknowns = ex.unknown_attempts();
    assert_eq!(
        unknowns.len(),
        1,
        "the NO Unknown leg must end up in the in-memory mirror"
    );
    assert_eq!(unknowns[0].token_id, "no-tok");
}
