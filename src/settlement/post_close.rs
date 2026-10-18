//! Passive post-close observer.
//!
//! After a bracket's `clear_at`, Polymarket takes a while to resolve the
//! market (window ≈ `evict_grace_secs` before the registry evicts it).
//! During that time the book keeps streaming — the losing side sometimes
//! quotes above 0 and the winner is slow to rise to $1. This task samples
//! those post-close books, computes a deterministic verdict (oracle spot
//! vs. the bracket's `reference_price`) and persists the observation. The goal
//! is to measure how much of the winning side can be bought/sold after the close before
//! the market corrects — input for a possible "post-close
//! sweep" strategy, not for trading yet.
//!
//! The verdict uses the FROZEN pre-close spot: a `SpotRef` is kept per market,
//! refreshed with the fresh vol of any book of the same asset while the bracket
//! is still pre-close. Previously the verdict was per sample (live spot at the
//! instant of the observation) and in micro-move brackets the spot 80-119s late
//! fell on the opposite side of the strike — the 11 historical observations with a
//! recorded ask had the verdict inverted with respect to the real winner because of this.
//! If there is no fresh frozen spot (≤ MAX_SPOT_AGE_SECS at close) the observation is
//! marked stale → Indeterminate.
//!
//! Dedicated task modeled on `accounting::markout::run`, not a `Strategy`: the
//! runner in `strategy/mod.rs` coalesces books via `drain_latest` and would lose post-close
//! observations (infrequent per bracket). It emits no orders, is not hot-toggleable,
//! and does not go through `RiskGuard`/strategy attribution.

use crate::time::Instant;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{broadcast, watch};
use tracing::warn;

use crate::{
    research::{
        contract::{EventSource, OutcomeSide},
        labels::{self, LabelRule, PriceObservation, RuleRegistration},
    },
    state::StateStore,
    types::{BookSnapshot, VolSnapshot},
};

/// Minimum relative move of the spot vs. `reference_price` to
/// declare a winner. Below this, oracle noise does not allow
/// telling the winning side apart (constant, not configurable).
const MIN_REL_MOVE: Decimal = dec!(0.0005);

/// Purges old entries of the throttle mapping every N loop iterations
/// (avoids traversing the whole mapping on every book).
const PURGE_EVERY: u32 = 200;

/// Maximum age (seconds relative to the CLOSE) of the frozen spot to
/// accept it as the verdict reference — same criterion as the pre-close
/// candidate (`max_spot_age_secs`, default 8).
const MAX_SPOT_AGE_SECS: i64 = 8;

/// Frozen pre-close spot of a market (same pattern as the pre-close `Candidate`).
struct SpotRef {
    clear_at: i64,
    asset: String,
    /// (spot, wall_ts of the capture) — only updated pre-close with fresh vol.
    spot: Option<(Decimal, i64)>,
}

/// Picks the spot for the post-close verdict: the frozen one if its age
/// at close is acceptable; if it is stale or missing, returns whatever there is with
/// `stale=true` (→ `deterministic_winner` declares it Indeterminate).
fn spot_for_verdict(
    frozen: Option<(i64, (Decimal, i64))>,
    live: Option<Decimal>,
) -> (Option<Decimal>, bool) {
    match frozen {
        Some((clear_at, (s, ts))) if clear_at.saturating_sub(ts) <= MAX_SPOT_AGE_SECS => {
            (Some(s), false)
        }
        Some((_, (s, _))) => (Some(s), true),
        None => (live, true),
    }
}

/// Deterministic verdict of an updown bracket at the instant of the sample.
#[derive(PartialEq, Debug)]
pub enum Verdict {
    Yes,
    No,
    Indeterminate,
}

impl Verdict {
    fn as_str(&self) -> &'static str {
        match self {
            Verdict::Yes => "yes",
            Verdict::No => "no",
            Verdict::Indeterminate => "indeterminate",
        }
    }
}

/// Deterministic winner of the updown bracket from the oracle spot.
///
/// Returns `(verdict, rel_move)` where `rel_move = (spot − reference) / reference`
/// (signed). `rel_move` is always returned when computable, even if the
/// verdict ends up `Indeterminate` (stale or move below threshold) — it is useful for
/// offline aggregate analysis.
///
/// Rules: `reference` missing/`<=0`, or `spot` missing → `Indeterminate` without
/// `rel_move`. `stale=true` → `Indeterminate` (with `rel_move` if computable).
/// `|rel_move| < min_move` → `Indeterminate`. `spot > reference` → `Yes`,
/// otherwise → `No`.
pub fn deterministic_winner(
    reference: Option<Decimal>,
    spot: Option<Decimal>,
    stale: bool,
    min_move: Decimal,
) -> (Verdict, Option<Decimal>) {
    let rel_move = match (reference, spot) {
        (Some(r), Some(s)) if r > Decimal::ZERO => Some((s - r) / r),
        _ => None,
    };

    let Some(rel_move) = rel_move else {
        return (Verdict::Indeterminate, None);
    };

    if stale || rel_move.abs() < min_move {
        return (Verdict::Indeterminate, Some(rel_move));
    }

    if rel_move > Decimal::ZERO {
        (Verdict::Yes, Some(rel_move))
    } else {
        (Verdict::No, Some(rel_move))
    }
}

/// Passive post-close observation task.
///
/// `throttle` limits sampling to ≤1 row per market every N ms (main passes
/// `config.post_close.sample_throttle_ms`). Modeled on `accounting::markout::run`: it
/// subscribes directly to the book broadcast, with no intermediate channel.
pub async fn run(
    state: Arc<StateStore>,
    mut book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    throttle: Duration,
    mut shutdown: watch::Receiver<bool>,
) {
    tracing::info!("post_close: starting");
    let mut last_sample: HashMap<String, Instant> = HashMap::new();
    let mut spot_refs: HashMap<String, SpotRef> = HashMap::new();
    let mut iters: u32 = 0;

    loop {
        tokio::select! {
            book = book_rx.recv() => match book {
                Ok(b) => {
                    let now_wall = chrono::Utc::now().timestamp();
                    // Same derivation criterion as the rest of the project
                    // (lowercase + None on an empty slug) — a manual split('-')
                    // without normalizing left the vol_rxs lookup failing
                    // silently if the slug arrived in uppercase.
                    let asset = crate::oracle::asset_from_slug(&b.slug).unwrap_or_default();
                    let vol = vol_rxs.get(&asset).map(|rx| rx.borrow().clone());

                    // ── Frozen spot maintenance (ALL books) ──
                    // Pre-close upsert of the market itself + cross-market refresh
                    // per asset, as in the pre-close candidate logic: the oracle's fresh spot
                    // is valid for any bracket of the asset, and is frozen only
                    // while the bracket is still pre-close.
                    if b.secs_to_clear() > 0 {
                        let e = spot_refs.entry(b.market_condition_id.clone()).or_insert(SpotRef {
                            clear_at: b.clear_at, asset: asset.to_string(), spot: None,
                        });
                        e.clear_at = b.clear_at;
                        e.asset    = asset.to_string();
                    }
                    if let Some(v) = &vol {
                        if !v.is_stale_for_signal() && v.last_price > Decimal::ZERO {
                            for r in spot_refs.values_mut() {
                                if r.asset == asset && now_wall < r.clear_at {
                                    r.spot = Some((v.last_price, now_wall));
                                }
                            }
                        }
                    }

                    // We only observe post-close books (secs_to_clear <= 0).
                    if b.secs_to_clear() > 0 {
                        continue;
                    }

                    let now = Instant::now();
                    if let Some(last) = last_sample.get(&b.market_condition_id) {
                        if now.duration_since(*last) < throttle {
                            continue;
                        }
                    }
                    last_sample.insert(b.market_condition_id.clone(), now);

                    iters = iters.wrapping_add(1);
                    if iters.is_multiple_of(PURGE_EVERY) {
                        last_sample.retain(|_, t| now.duration_since(*t) < Duration::from_secs(300));
                        spot_refs.retain(|_, r| now_wall < r.clear_at + 300);
                    }

                    let (spot, stale) = spot_for_verdict(
                        spot_refs.get(&b.market_condition_id).and_then(|r| r.spot.map(|s| (r.clear_at, s))),
                        vol.as_ref().map(|v| v.last_price),
                    );

                    let (verdict, rel_move) =
                        deterministic_winner(b.reference_price, spot, stale, MIN_REL_MOVE);

                    let (winner_ask, winner_ask_sz) = match verdict {
                        Verdict::Yes           => (b.yes.best_ask(), b.yes.asks.best_size()),
                        Verdict::No            => (b.no.best_ask(),  b.no.asks.best_size()),
                        Verdict::Indeterminate => (None, None),
                    };

                    // Post-close: secs_to_clear() <= 0, so its negative is
                    // the time elapsed since the close.
                    let secs_since_close = -b.secs_to_clear();

                    if let Err(e) = state.record_post_close(
                        chrono::Utc::now().timestamp_millis(),
                        secs_since_close,
                        &b.market_condition_id,
                        &b.slug,
                        b.clear_at,
                        b.reference_price,
                        spot,
                        stale,
                        rel_move,
                        verdict.as_str(),
                        b.yes.best_bid(), b.yes.best_ask(), b.yes.bids.best_size(), b.yes.asks.best_size(),
                        b.no.best_bid(),  b.no.best_ask(),  b.no.bids.best_size(),  b.no.asks.best_size(),
                        winner_ask, winner_ask_sz,
                    ).await {
                        // Best-effort: do not abort the task over a one-off write failure.
                        warn!("post_close: error persisting observation: {e}");
                    }

                    // The verdict feeds the labels pipeline
                    // as a PROVISIONAL resolution — an explicit proxy
                    // (`PROVENANCE_PROXY_POST_CLOSE`), excluded from
                    // certified evaluation, never a confirmed label. The
                    // market's rule is also registered (idempotent) and
                    // the spot observation used (window evidence).
                    let winner = match verdict {
                        Verdict::Yes => Some(OutcomeSide::Yes),
                        Verdict::No => Some(OutcomeSide::No),
                        Verdict::Indeterminate => None,
                    };
                    if let Some(winner) = winner {
                        // `crate::time::now` (injectable clock) — consistent
                        // with the rest of the labels pipeline timestamps.
                        let now_ms = crate::time::now().timestamp_millis();
                        let reg = RuleRegistration {
                            market_cid: b.market_condition_id.clone(),
                            slug: b.slug.clone(),
                            rule: LabelRule::UpDownReference,
                            rule_version: labels::RULE_VERSION,
                            resolution_source: EventSource::PolymarketRest,
                            // Concrete reference in the source: the slug is
                            // the key by which Gamma/CLOB identify the
                            // market.
                            resolution_source_ref: Some(format!("gamma:{}", b.slug)),
                            reference_price: b.reference_price,
                            market_open_ts: Some(b.market_open_ts),
                            clear_at: Some(b.clear_at),
                        };
                        if let Err(e) =
                            labels::register_rule(&state, &reg, now_ms).await
                        {
                            warn!("post_close: error registering resolution rule: {e}");
                        }
                        // The spot used in the verdict: if it is the frozen one,
                        // its ts is that of the pre-close capture (matters for
                        // TWAP windows); if it fell back to the live one, the ts is now.
                        if let Some(s) = spot {
                            let obs_ts_ms = spot_refs
                                .get(&b.market_condition_id)
                                .and_then(|r| r.spot.map(|(_, ts)| ts * 1000))
                                .unwrap_or(now_ms);
                            let obs = PriceObservation {
                                ts_ms: obs_ts_ms,
                                price: s,
                                source: EventSource::BinanceWs,
                            };
                            if let Err(e) = labels::record_price_observation(
                                &state,
                                &b.market_condition_id,
                                &obs,
                                now_ms,
                            )
                            .await
                            {
                                warn!("post_close: error registering resolution observation: {e}");
                            }
                        }
                        if let Err(e) = labels::apply_provisional(
                            &state,
                            &b.market_condition_id,
                            EventSource::BinanceWs,
                            winner,
                            labels::PROVENANCE_PROXY_POST_CLOSE,
                            now_ms,
                            Some(LabelRule::UpDownReference.tag()),
                        )
                        .await
                        {
                            warn!("post_close: error registering provisional resolution: {e}");
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(n, "post_close: books lost to broadcast lag");
                    metrics::counter!("hrf_post_close_book_lag_total").increment(n);
                }
                Err(broadcast::error::RecvError::Closed) => return,
            },
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    tracing::info!("post_close: shutdown received");
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── spot_for_verdict ──────────────────────────────────────────────────

    /// The post-close verdict must use the FROZEN pre-close spot; the live
    /// spot only enters as a last resort and always marked stale (the 11
    /// historical observations with an inverted verdict came from using the
    /// live spot 78-119s after the close in micro-move brackets).
    #[test]
    fn fresh_frozen_spot_is_used_without_stale() {
        let clear_at = 1_000_000;
        let (s, stale) = spot_for_verdict(
            Some((clear_at, (dec!(77.10), clear_at - 3))),
            Some(dec!(99.0)),
        );
        assert_eq!(s, Some(dec!(77.10)));
        assert!(!stale);
    }

    #[test]
    fn old_frozen_spot_goes_stale() {
        let clear_at = 1_000_000;
        let (s, stale) = spot_for_verdict(
            Some((clear_at, (dec!(77.10), clear_at - 20))),
            Some(dec!(99.0)),
        );
        assert_eq!(
            s,
            Some(dec!(77.10)),
            "the frozen one is kept even if stale (for the record), but marked stale"
        );
        assert!(
            stale,
            "20s > MAX_SPOT_AGE_SECS(8) → Indeterminate via stale flag"
        );
    }

    #[test]
    fn without_frozen_falls_back_to_live_but_stale() {
        let (s, stale) = spot_for_verdict(None, Some(dec!(99.0)));
        assert_eq!(s, Some(dec!(99.0)));
        assert!(stale, "live post-close spot never produces a verdict");
    }

    // ── deterministic_winner ─────────────────────────────────────────────────

    #[test]
    fn winner_spot_rises_above_threshold_is_yes() {
        // +0.1% > MIN_REL_MOVE (0.05%)
        let (v, mv) =
            deterministic_winner(Some(dec!(100000)), Some(dec!(100100)), false, MIN_REL_MOVE);
        assert_eq!(v, Verdict::Yes);
        assert_eq!(mv, Some(dec!(0.001)));
    }

    #[test]
    fn winner_move_below_threshold_is_indeterminate() {
        // +0.04% < MIN_REL_MOVE (0.05%)
        let (v, mv) =
            deterministic_winner(Some(dec!(100000)), Some(dec!(100040)), false, MIN_REL_MOVE);
        assert_eq!(v, Verdict::Indeterminate);
        assert_eq!(mv, Some(dec!(0.0004)));
    }

    #[test]
    fn winner_stale_is_indeterminate_even_if_move_is_large() {
        let (v, mv) =
            deterministic_winner(Some(dec!(100000)), Some(dec!(101000)), true, MIN_REL_MOVE);
        assert_eq!(v, Verdict::Indeterminate);
        assert_eq!(mv, Some(dec!(0.01)));
    }

    #[test]
    fn winner_without_reference_is_indeterminate_without_rel_move() {
        let (v, mv) = deterministic_winner(None, Some(dec!(100100)), false, MIN_REL_MOVE);
        assert_eq!(v, Verdict::Indeterminate);
        assert_eq!(mv, None);
    }

    #[test]
    fn winner_spot_drops_below_threshold_is_no() {
        // -0.1% < -MIN_REL_MOVE
        let (v, mv) =
            deterministic_winner(Some(dec!(100000)), Some(dec!(99900)), false, MIN_REL_MOVE);
        assert_eq!(v, Verdict::No);
        assert_eq!(mv, Some(dec!(-0.001)));
    }

    #[test]
    fn winner_rel_move_keeps_sign() {
        let (_, mv_up) =
            deterministic_winner(Some(dec!(100)), Some(dec!(101)), false, MIN_REL_MOVE);
        let (_, mv_down) =
            deterministic_winner(Some(dec!(100)), Some(dec!(99)), false, MIN_REL_MOVE);
        assert!(mv_up.unwrap() > Decimal::ZERO);
        assert!(mv_down.unwrap() < Decimal::ZERO);
    }

    // ── run (integration) ────────────────────────────────────────────────────

    use crate::types::{HalfBook, Level, Mode, TokenBook};

    fn make_book(reference: Option<&str>, clear_delta_secs: i64) -> Arc<BookSnapshot> {
        let side = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        let now = chrono::Utc::now().timestamp();
        Arc::new(BookSnapshot {
            market_condition_id: "pcs-cid".into(),
            yes: TokenBook {
                token_id: "yes-tok".into(),
                bids: side("0.97"),
                asks: side("0.99"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-tok".into(),
                bids: side("0.01"),
                asks: side("0.03"),
                ..Default::default()
            },
            market_open_ts: now - 300,
            clear_at: now + clear_delta_secs,
            neg_risk: false,
            slug: "btc-updown-5m-pcs".into(),
            reference_price: reference.map(|r| r.parse().unwrap()),
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        })
    }

    fn make_vol(live: &str) -> VolSnapshot {
        VolSnapshot {
            asset: "btc".into(),
            sigma_short: Some(dec!(0.001)),
            sigma_long: Some(dec!(0.001)),
            last_price: live.parse().unwrap(),
            updated_at: Instant::now(),
            ..VolSnapshot::default()
        }
    }

    #[tokio::test]
    async fn post_close_persists_one_observation_per_post_close_book() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (book_tx, book_rx) = broadcast::channel(16);
        let (_vol_tx, vol_rx) = watch::channel(make_vol("100100")); // +0.1% over reference
        let mut vol_rxs = HashMap::new();
        vol_rxs.insert("btc".to_string(), vol_rx);
        let (_kill_tx, kill_rx) = watch::channel(false);

        tokio::spawn(run(
            state.clone(),
            book_rx,
            vol_rxs,
            Duration::from_millis(0),
            kill_rx,
        ));

        // Book already closed (clear_at = now - 5) with reference_price so the
        // verdict is determinable.
        book_tx.send(make_book(Some("100000"), -5)).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let n = state.post_close_observations_count().await.unwrap();
        assert_eq!(n, 1);
    }

    /// A determinable verdict also persists the PROVISIONAL
    /// resolution (post-close proxy) and the market's rule — never a confirmed
    /// label (that requires official evidence via `apply_official`).
    #[tokio::test]
    async fn post_close_records_provisional_resolution_and_rule() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (book_tx, book_rx) = broadcast::channel(16);
        let (_vol_tx, vol_rx) = watch::channel(make_vol("100100")); // +0.1% over reference
        let mut vol_rxs = HashMap::new();
        vol_rxs.insert("btc".to_string(), vol_rx);
        let (_kill_tx, kill_rx) = watch::channel(false);

        tokio::spawn(run(
            state.clone(),
            book_rx,
            vol_rxs,
            Duration::from_millis(0),
            kill_rx,
        ));

        // Pre-close book first (freezes the spot) and then the post-close one.
        book_tx.send(make_book(Some("100000"), 5)).unwrap();
        book_tx.send(make_book(Some("100000"), -5)).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let recs = state.resolutions_for("pcs-cid").await.unwrap();
        assert_eq!(recs.len(), 1, "a single idempotent provisional");
        assert_eq!(
            recs[0].status,
            crate::research::labels::ResolutionStatus::Provisional
        );
        assert_eq!(
            recs[0].provenance,
            crate::research::labels::PROVENANCE_PROXY_POST_CLOSE
        );
        assert_eq!(recs[0].winner, crate::research::contract::OutcomeSide::Yes);

        let reg = state.resolution_rule("pcs-cid").await.unwrap().unwrap();
        assert_eq!(reg.rule, crate::research::labels::LabelRule::UpDownReference);

        // The provisional resolution NEVER produces a confirmed label.
        assert!(
            crate::research::labels::confirmed_label(&state, "pcs-cid", None)
                .await
                .unwrap()
                .is_none()
        );
    }
}
