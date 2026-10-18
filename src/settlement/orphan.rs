//! Orphan position checker — resolves positions of expired markets by querying the CLOB.
//!
//! Every `poll_secs` it looks for positions with `clear_at < now()` and `outcome_won IS NULL`.
//! It queries `clob.polymarket.com/markets/{condition_id}`, which includes the `closed`,
//! `winner` and post-settlement `price` fields per token — the only reliable source for brackets
//! whose winner is decided by on-chain resolution, not by the last traded price.
//!
//! If the market is not yet `closed=true` or the token has no `winner` field defined,
//! resolution is deferred to the next cycle (UMA resolution pending).
//!
//! Simulated PnL:
//!   - won  → size × $1 − cost_basis
//!   - lost → −cost_basis

use std::{collections::HashMap, sync::Arc};

use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::{
    risk::RiskGuard,
    state::StateStore,
    types::{Mode, StrategyId},
};

const CLOB_BASE: &str = "https://clob.polymarket.com";
const DEFAULT_POLL_SECS: u64 = 60; // 1 minute

// ── CLOB /markets/{cid} ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ClobMarket {
    #[serde(default)]
    closed: bool,
    tokens: Vec<ClobToken>,
}

#[derive(Deserialize)]
struct ClobToken {
    token_id: String,
    outcome: Option<String>,
    winner: Option<bool>,
}

// ── Public task ───────────────────────────────────────────────────────────────

/// `sim` is the paper router — when present, each settlement
/// booked in the ledger credits its payout to the sim's cash and debits the
/// inventory (`credit_redemption`, idempotent by `redeem_id`). In live
/// it is `None`: the real income comes from the on-chain redeem (or the operator's
/// sweep), not from the simulated wallet.
pub async fn run(
    state: Arc<StateStore>,
    risk: Arc<RiskGuard>,
    poll_secs: Option<u64>,
    mut shutdown: watch::Receiver<bool>,
    redeemer: Option<Arc<dyn crate::settlement::redeem::ChainRedeemer>>,
    sim: Option<Arc<crate::order::sim_router::SimulatedRouter>>,
) {
    let period = std::time::Duration::from_secs(poll_secs.unwrap_or(DEFAULT_POLL_SECS));
    let http = reqwest::Client::builder()
        .user_agent(crate::POLY_USER_AGENT)
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .expect("orphan: reqwest client");

    info!(
        "orphan: checker started (poll every {}s)",
        period.as_secs()
    );

    let mut ticker = tokio::time::interval(period);
    ticker.tick().await; // skip the first, immediate tick

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => { if *shutdown.borrow() { break; } }
            _ = ticker.tick() => { poll(&http, &state, &risk, redeemer.as_deref(), sim.as_deref()).await; }
        }
    }
    info!("orphan: checker stopped");
}

// ── Resolution cycle ──────────────────────────────────────────────────────────

/// Exposed for integration tests: runs one complete resolution cycle.
pub async fn poll_once(state: &StateStore, risk: &RiskGuard) -> anyhow::Result<()> {
    poll_once_with_redeemer(state, risk, None, None).await
}

/// `poll_once` with an injectable redeemer — tests fix the on-chain outcome
/// without a real chain. `sim`: paper router to credit with
/// the settlement payout; `None` in tests without a sim and in live.
pub async fn poll_once_with_redeemer(
    state: &StateStore,
    risk: &RiskGuard,
    redeemer: Option<&dyn crate::settlement::redeem::ChainRedeemer>,
    sim: Option<&crate::order::sim_router::SimulatedRouter>,
) -> anyhow::Result<()> {
    let http = reqwest::Client::builder()
        .user_agent(crate::POLY_USER_AGENT)
        .timeout(std::time::Duration::from_secs(15))
        .build()?;
    poll(&http, state, risk, redeemer, sim).await;
    Ok(())
}

async fn poll(
    http: &reqwest::Client,
    state: &StateStore,
    risk: &RiskGuard,
    redeemer: Option<&dyn crate::settlement::redeem::ChainRedeemer>,
    sim: Option<&crate::order::sim_router::SimulatedRouter>,
) {
    let positions = match state.list_expired_unresolved().await {
        Ok(v) => v,
        Err(e) => {
            warn!("orphan: error querying positions: {e}");
            return;
        }
    };

    if positions.is_empty() {
        debug!("orphan: no expired unresolved positions");
        return;
    }

    info!("orphan: checking {} expired positions", positions.len());

    // Group by market_cid to make a single CLOB call per market.
    let mut by_cid: HashMap<String, Vec<_>> = HashMap::new();
    for pos in positions {
        by_cid.entry(pos.market_cid.clone()).or_default().push(pos);
    }

    let mut first = true;
    for (cid, positions) in by_cid {
        if !first {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        first = false;
        let market = match fetch_market(http, &cid).await {
            Ok(m) => m,
            Err(e) => {
                warn!(cid = %cid, "orphan: error querying CLOB: {e}");
                continue;
            }
        };

        if !market.closed {
            debug!(cid = %cid, "orphan: market not closed yet according to CLOB — retrying next cycle");
            continue;
        }

        // Fast token_id → winner index
        let token_winners: HashMap<&str, bool> = market
            .tokens
            .iter()
            .filter_map(|t| t.winner.map(|w| (t.token_id.as_str(), w)))
            .collect();

        for pos in positions {
            // If the `winner` field has not arrived yet (UMA resolution pending), wait.
            let Some(&won) = token_winners.get(pos.token_id.as_str()) else {
                debug!(
                    token = %pos.token_id,
                    cid   = %cid,
                    "orphan: `winner` field missing — resolution pending, retrying next cycle"
                );
                continue;
            };

            if let Err(e) = settle(state, risk, &pos, won, &market, redeemer, sim).await {
                warn!(token = %pos.token_id, "orphan: error settling: {e}");
            }
        }
    }
}

async fn settle(
    state: &StateStore,
    risk: &RiskGuard,
    pos: &crate::types::Position,
    won: bool,
    market: &ClobMarket,
    redeemer: Option<&dyn crate::settlement::redeem::ChainRedeemer>,
    sim: Option<&crate::order::sim_router::SimulatedRouter>,
) -> anyhow::Result<()> {
    // Label of the winning outcome (e.g. "Up" / "Down", "Yes" / "No").
    let winner_label = market
        .tokens
        .iter()
        .find(|t| t.winner == Some(true))
        .and_then(|t| t.outcome.as_deref())
        .unwrap_or(if won { "Yes" } else { "No" });

    let payout = if won { pos.size } else { Decimal::ZERO };
    let pnl = payout - pos.cost_basis;

    let strategy_id = pos.strategy.parse().unwrap_or(StrategyId::Arb);
    let redeem_tx = match risk.mode() {
        Mode::Paper => Some("paper-auto".to_string()),
        Mode::Live if risk.auto_redeem() => Some("live-auto".to_string()),
        // (auto_redeem=false, EOA custody): the on-chain redeem is
        // attempted BEFORE booking — if it fails, `Err` leaves `outcome_won` as
        // NULL and the next poll retries. redeemed=1 is never set without a
        // mined tx (or a balance of 0 verified on-chain).
        Mode::Live => {
            let idx = market
                .tokens
                .iter()
                .position(|t| t.token_id == pos.token_id)
                .unwrap_or(0);
            crate::settlement::redeem::try_redeem(redeemer, pos, won, idx, market.tokens.len()).await?
        }
    };

    // A single SQLite transaction: outcome + REDEEM entry + redeemed — all or nothing.
    let settled = state
        .apply_settlement_detail(
            pos,
            won,
            winner_label,
            payout,
            pnl,
            strategy_id,
            redeem_tx.as_deref(),
        )
        .await?;

    let Some(applied) = settled else {
        // Another race already settled it — its pnl was booked there; repeating it
        // would inflate the daily-loss-stop and the strategy statistics.
        info!(
            token = %pos.token_id,
            cid   = %pos.market_cid,
            "orphan: position already settled by another branch — skipping pnl"
        );
        return Ok(());
    };

    // In paper, resolution also moves the sim's CASH and
    // consumes the inventory — with the FRESH values booked by the ledger,
    // not the `pos` snapshot. A stable `redeem_id` makes it idempotent across
    // re-polls/reopenings; an Err is logged but does not undo the settlement
    // (the ledger is already booked — drift shows up in `accounting()`).
    if let Some(sim) = sim {
        let redeem_id = format!(
            "settle:{}:{}:{}",
            pos.market_cid, pos.token_id, pos.strategy
        );
        if let Err(e) =
            sim.credit_redemption(&redeem_id, &pos.token_id, applied.size, applied.payout)
        {
            warn!(
                token = %pos.token_id,
                cid   = %pos.market_cid,
                "orphan: credit_redemption on the sim failed: {e}"
            );
        }
    }

    risk.record_pnl(pnl);
    risk.record_strategy_pnl(strategy_id, pnl);

    // A winner without redeem_tx in live only happens if there was no redeemer —
    // `Config::validate` + the fail-closed startup make it unreachable; the
    // warn stays as a safeguard. Losers have nothing to claim.
    if redeem_tx.is_none() && won && risk.mode() == Mode::Live {
        warn!(
            token    = %pos.token_id,
            cid      = %pos.market_cid,
            "orphan: winner booked WITHOUT on-chain redemption — check DB"
        );
    }

    info!(
        token      = %pos.token_id,
        cid        = %pos.market_cid,
        strategy = %pos.strategy,
        won_flag     = won,
        winner     = winner_label,
        pnl        = %pnl,
        "orphan: position resolved"
    );

    Ok(())
}

// ── CLOB API ──────────────────────────────────────────────────────────────────

async fn fetch_market(http: &reqwest::Client, condition_id: &str) -> anyhow::Result<ClobMarket> {
    let url = format!("{CLOB_BASE}/markets/{condition_id}");
    let resp = http.get(&url).send().await?.error_for_status()?;
    Ok(resp.json::<ClobMarket>().await?)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RiskConfig;
    use crate::settlement::redeem::{ChainRedeemer, RedeemOutcome};
    use crate::types::{Fill, FillSide, Position};
    use rust_decimal_macros::dec;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Fake redeemer with a preset result and a call counter.
    struct FakeRedeemer {
        result: Mutex<anyhow::Result<RedeemOutcome>>,
        calls: AtomicUsize,
    }

    impl FakeRedeemer {
        fn sent(hash: &str) -> Self {
            Self {
                result: Mutex::new(Ok(RedeemOutcome::Sent(hash.to_string()))),
                calls: AtomicUsize::new(0),
            }
        }
        fn with(result: anyhow::Result<RedeemOutcome>) -> Self {
            Self {
                result: Mutex::new(result),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl ChainRedeemer for FakeRedeemer {
        fn redeem<'a>(
            &'a self,
            _pos: &'a Position,
            _token_index: usize,
            _outcome_slots: usize,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = anyhow::Result<RedeemOutcome>> + Send + 'a>,
        > {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let r = std::mem::replace(
                &mut *self.result.lock().unwrap(),
                Err(anyhow::anyhow!("redeemer exhausted")),
            );
            Box::pin(async move { r })
        }
    }

    fn risk_live_without_auto_redeem() -> Arc<RiskGuard> {
        let cfg = RiskConfig {
            auto_redeem: false,
            ..RiskConfig::default()
        };
        RiskGuard::new(cfg, Mode::Live)
    }

    /// Winning position bought on `token-yes` (expired — clear_at in the past).
    async fn winning_position(state: &StateStore) -> Position {
        let fill = Fill {
            market_cid: "0x".to_string() + &"a".repeat(64),
            token_id: "111".to_string(),
            side: FillSide::Buy,
            price: dec!(0.60),
            size: dec!(10),
            order_id: Some("o-1".to_string()),
            ts: chrono::Utc::now(),
            mode: Mode::Live,
            neg_risk: false,
            strategy: StrategyId::Arb,
            clear_at: 1_000_000,
            outcome_label: Some("YES".to_string()),
            hedge_id: None,
            exec_id: Some("exec-1".to_string()),
            local_order_id: None,
        };
        state.apply_fill(&fill).await.unwrap();
        state.get_position("111", "arb").await.unwrap().unwrap()
    }

    fn resolved_binary_market() -> ClobMarket {
        ClobMarket {
            closed: true,
            tokens: vec![
                ClobToken {
                    token_id: "111".to_string(),
                    outcome: Some("Up".to_string()),
                    winner: Some(true),
                },
                ClobToken {
                    token_id: "222".to_string(),
                    outcome: Some("Down".to_string()),
                    winner: Some(false),
                },
            ],
        }
    }

    #[tokio::test]
    async fn settle_live_sends_tx_and_saves_hash() {
        let state = StateStore::open_memory(Mode::Live).await.unwrap();
        let risk = risk_live_without_auto_redeem();
        let pos = winning_position(&state).await;
        let fake = FakeRedeemer::sent("0xdeadbeef");

        settle(
            &state,
            &risk,
            &pos,
            true,
            &resolved_binary_market(),
            Some(&fake),
            None,
        )
        .await
        .unwrap();

        assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
        let p = state.get_position("111", "arb").await.unwrap().unwrap();
        assert_eq!(p.outcome_won, Some(true));
        assert!(p.redeemed);
        assert_eq!(p.redeem_tx.as_deref(), Some("0xdeadbeef"));
        // The settled position does not re-enter the poll.
        assert!(state.list_expired_unresolved().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn settle_live_redeem_failure_leaves_pending_for_retry() {
        let state = StateStore::open_memory(Mode::Live).await.unwrap();
        let risk = risk_live_without_auto_redeem();
        let pos = winning_position(&state).await;
        let fake = FakeRedeemer::with(Err(anyhow::anyhow!("rpc down")));

        assert!(settle(
            &state,
            &risk,
            &pos,
            true,
            &resolved_binary_market(),
            Some(&fake),
            None
        )
        .await
        .is_err());

        let p = state.get_position("111", "arb").await.unwrap().unwrap();
        // Fail-closed: neither outcome nor redeemed — the next poll retries.
        assert_eq!(p.outcome_won, None);
        assert!(!p.redeemed);
        assert!(!state.list_expired_unresolved().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn settle_live_proxy_custody_leaves_pending() {
        let state = StateStore::open_memory(Mode::Live).await.unwrap();
        let risk = risk_live_without_auto_redeem();
        let pos = winning_position(&state).await;
        let fake = FakeRedeemer::with(Ok(RedeemOutcome::ProxyCustody));

        assert!(settle(
            &state,
            &risk,
            &pos,
            true,
            &resolved_binary_market(),
            Some(&fake),
            None
        )
        .await
        .is_err());
        assert_eq!(
            state
                .get_position("111", "arb")
                .await
                .unwrap()
                .unwrap()
                .outcome_won,
            None
        );
    }

    #[tokio::test]
    async fn settle_live_zero_balance_settles_with_forensic_mark() {
        let state = StateStore::open_memory(Mode::Live).await.unwrap();
        let risk = risk_live_without_auto_redeem();
        let pos = winning_position(&state).await;
        let fake = FakeRedeemer::with(Ok(RedeemOutcome::NoBalance));

        settle(
            &state,
            &risk,
            &pos,
            true,
            &resolved_binary_market(),
            Some(&fake),
            None,
        )
        .await
        .unwrap();

        let p = state.get_position("111", "arb").await.unwrap().unwrap();
        assert!(p.redeemed);
        assert_eq!(p.redeem_tx.as_deref(), Some("onchain-no-balance"));
    }

    #[tokio::test]
    async fn settle_live_losing_does_not_call_redeemer() {
        let state = StateStore::open_memory(Mode::Live).await.unwrap();
        let risk = risk_live_without_auto_redeem();
        let pos = winning_position(&state).await;
        let fake = FakeRedeemer::sent("0xno");

        settle(
            &state,
            &risk,
            &pos,
            false,
            &resolved_binary_market(),
            Some(&fake),
            None,
        )
        .await
        .unwrap();

        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
        let p = state.get_position("111", "arb").await.unwrap().unwrap();
        assert_eq!(p.outcome_won, Some(false));
        assert!(!p.redeemed); // nothing to claim on-chain
    }

    #[tokio::test]
    async fn settle_paper_does_not_call_redeemer() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let risk = RiskGuard::new(RiskConfig::default(), Mode::Paper);
        let pos = winning_position(&state).await;
        let fake = FakeRedeemer::sent("0xno");

        settle(
            &state,
            &risk,
            &pos,
            true,
            &resolved_binary_market(),
            Some(&fake),
            None,
        )
        .await
        .unwrap();

        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
        let p = state.get_position("111", "arb").await.unwrap().unwrap();
        assert!(p.redeemed);
        assert_eq!(p.redeem_tx.as_deref(), Some("paper-auto"));
    }
}
