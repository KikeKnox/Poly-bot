//! PostCloseSweep (PCS-2) strategy.
//!
//! After a bracket's `clear_at`, Polymarket takes a while to resolve the
//! market and the book keeps streaming (see `src/settlement/post_close.rs`). This
//! strategy trades that window: it freezes the oracle spot right before
//! close, computes the deterministic verdict (`settlement::post_close::deterministic_winner`)
//! and buys the winning side: a maker GTD at `maker_bid_price` (0.99 by default) and,
//! optionally, a taker FOK sweep if the winner's ask is still cheap.
//!
//! There is no bail: the post-close verdict is deterministic (spot vs. an already
//! known reference_price), not directional; the only exit is automatic redemption
//! when the market resolves. The risk is asymmetric (we buy at ~0.99, win ~0.01
//! or lose ~0.99 if the verdict is wrong, a 1:100 ratio), mitigated by
//! `min_rel_move`: a spot move that is too small is declared
//! `Indeterminate` and the strategy does not enter.

use crate::time::Instant;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{mpsc, watch};

use crate::{
    config::defaults,
    settlement::post_close::{deterministic_winner, Verdict},
    types::{BookSnapshot, Fill, FillSide, OrderIntent, OrderSide, StrategyId, VolSnapshot},
};

use super::{Strategy, StrategyConfig};

/// PostCloseSweep (PCS-2) strategy configuration.
///
/// Buys the deterministic winning side after the bracket's `clear_at`, while
/// Polymarket has not yet corrected the book (see `settlement::post_close::deterministic_winner`).
/// Maker GTD at `maker_bid_price` + optional taker sweep if the ask is still cheap.
/// Exit only by automatic redemption (no bail: the post-close verdict
/// is deterministic, not directional).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PostCloseSweepConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    /// Minimum relative spot move vs. reference_price to declare a verdict.
    #[serde(default = "defaults::pcs2_min_rel_move")]
    pub min_rel_move: Decimal,
    /// Price of the maker GTD bid on the winning side.
    #[serde(default = "defaults::pcs2_maker_bid_price")]
    pub maker_bid_price: Decimal,
    /// Enables the taker (FOK) sweep when the winner's ask is still cheap.
    #[serde(default)]
    pub taker_enabled: bool,
    /// Maximum ask of the winning side to trigger the taker sweep.
    #[serde(default = "defaults::pcs2_taker_max_ask")]
    pub taker_max_ask: Decimal,
    /// TTL of the maker GTD in seconds (after this, cancel if there is no fill).
    #[serde(default = "defaults::pcs2_gtd_ttl_secs")]
    pub gtd_ttl_secs: u64,
    /// Maximum post-close window to trade (seconds since clear_at).
    #[serde(default = "defaults::pcs2_max_sweep_secs")]
    pub max_sweep_secs: u64,
    /// Maximum age of the pre-close frozen spot to accept it as a reference.
    #[serde(default = "defaults::pcs2_max_spot_age_secs")]
    pub max_spot_age_secs: u64,
    /// Maximum size in shares. 0 = no cap (primary limit: max_order_usdc).
    #[serde(default)]
    pub max_size_shares: Decimal,
    /// Global exposure reserve exclusive to PCS-2 (`RiskGuard::
    /// effective_max_total_exposure`). `exposure > max` rejections used to
    /// lose near-certain-verdict entries that
    /// competed for the same global cap with directional strategies; this
    /// carve-out adds to the cap ONLY for PCS-2, without touching the other strategies' limit.
    #[serde(default = "defaults::pcs2_exposure_carveout_usdc")]
    pub exposure_carveout_usdc: Decimal,
}

impl Default for PostCloseSweepConfig {
    fn default() -> Self {
        Self {
            limits: crate::config::StrategyLimits::default(),
            min_rel_move: defaults::pcs2_min_rel_move(),
            maker_bid_price: defaults::pcs2_maker_bid_price(),
            taker_enabled: false,
            taker_max_ask: defaults::pcs2_taker_max_ask(),
            gtd_ttl_secs: defaults::pcs2_gtd_ttl_secs(),
            max_sweep_secs: defaults::pcs2_max_sweep_secs(),
            max_spot_age_secs: defaults::pcs2_max_spot_age_secs(),
            max_size_shares: Decimal::ZERO,
            exposure_carveout_usdc: defaults::pcs2_exposure_carveout_usdc(),
        }
    }
}

impl StrategyConfig for PostCloseSweepConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

const CLOB_BASE: &str = "https://clob.polymarket.com";
/// Cadence of the async poller: the useful window of post-close dumps has a
/// median of 2s (measured), and 400ms leaves several attempts within that
/// without hammering the CLOB (a single in-flight token at a time per bracket).
const TAKER_POLL_INTERVAL_MS: u64 = 400;

// ── Per-bracket state ────────────────────────────────────────────────────────

struct PcsState {
    real_clear_at: i64,
    // Chosen winning side (true=YES).
    is_yes: bool,
    winner_token: String,
    // Needed to build the taker sweep intent fired by the
    // async poller (`taker_fok_intent_synthetic`), which has no
    // `&BookSnapshot` at hand (see the comment on `gtd_bid_intent`).
    neg_risk: bool,
    // GTD bid in flight.
    have_bid: bool,
    // Taker sweep already fired (avoids repeating it).
    taker_fired: bool,
    // Fill confirmed → position open (exit only by redemption).
    have_position: bool,
    position_size: Decimal,
    // Instant at which the currently in-flight GTD was placed (for the TTL).
    bid_placed_at: Option<Instant>,
    // Bracket discarded (permanent skip or TTL exhausted) — do not re-evaluate.
    done: bool,
}

impl PcsState {
    /// Terminal state for brackets discarded on entry (no reference,
    /// stale spot, indeterminate verdict or window already closed). Avoids
    /// recomputing the verdict on every remaining post-close tick.
    fn skipped(real_clear_at: i64) -> Self {
        Self {
            real_clear_at,
            is_yes: false,
            winner_token: String::new(),
            neg_risk: false,
            have_bid: false,
            taker_fired: false,
            have_position: false,
            position_size: Decimal::ZERO,
            bid_placed_at: None,
            done: true,
        }
    }
}

// ── Pre-close candidate ──────────────────────────────────────────────────────

/// The market's own post-close book takes 53s on average to arrive
/// (range 0-124s, measured): books are only emitted when there is
/// activity. Waiting for that book to enter misses the useful window of dumps
/// (p50=2s). Instead, every bracket with a pre-close book is kept as a
/// `Candidate` here; the entry is fired by wall clock on EVERY
/// `on_book` (whichever market the book comes from), not on the first real
/// post-close book.
struct Candidate {
    clear_at: i64,
    yes_token: String,
    no_token: String,
    neg_risk: bool,
    reference_price: Option<Decimal>,
    /// Asset of the bracket (`VolSnapshot::asset` of the book that created it); lets us
    /// refresh the frozen spot with ticks from OTHER markets of the same asset.
    asset: String,
    /// Pre-close frozen spot and timestamp (seconds) of that capture.
    /// `None` if a fresh spot was never seen before close. Updated
    /// ONLY with fresh vol (`!is_stale_for_signal`) and ONLY while the bracket
    /// is still pre-close, so post-close oracle noise cannot
    /// flip the entry relative to the real spot at `clear_at`. The refresh
    /// comes from any book of the same asset, not just the market's own
    /// (books are only emitted on activity, and a quiet market
    /// let its spot age even though the oracle was fresh).
    spot_ref: Option<(Decimal, i64)>,
}

// ── Async poller for the taker sweep ───────────────────────────────────────

/// Watch request for `run_taker_poller`: a bracket that has just
/// entered by wall clock (maker GTD placed) and for which we still
/// have no real ask from its own market (the clock scan does not bring one,
/// by design; see the doc of `Candidate`). The poller asks the CLOB over
/// REST, outside the synchronous `Strategy` thread, and fires this bracket without
/// waiting for the market's own book (same ~53s latency problem the
/// maker entry already solved, here for the taker sweep).
pub struct TakerWatch {
    market_cid: String,
    token_id: String,
    max_ask: Decimal,
    deadline: Instant,
}

// ── PostCloseSweep ────────────────────────────────────────────────────────────

pub struct PostCloseSweep {
    cfg: PostCloseSweepConfig,
    states: HashMap<String, PcsState>,
    candidates: HashMap<String, Candidate>,
    last_retain: Instant,
    /// Confirmed asks <= `taker_max_ask` by `market_cid`, written by
    /// `run_taker_poller` (separate task, network I/O) and consumed here on
    /// ANY `on_book`, never the other way around: `Strategy` remains the only one
    /// that decides and emits `OrderIntent` (same pattern as `MarkPriceCache`).
    taker_ready: Arc<DashMap<String, Decimal>>,
    poll_tx: mpsc::UnboundedSender<TakerWatch>,
    poll_rx: Option<mpsc::UnboundedReceiver<TakerWatch>>,
}

impl PostCloseSweep {
    pub fn new(cfg: PostCloseSweepConfig) -> Self {
        let (poll_tx, poll_rx) = mpsc::unbounded_channel();
        Self {
            cfg,
            states: HashMap::new(),
            candidates: HashMap::new(),
            last_retain: Instant::now(),
            taker_ready: Arc::new(DashMap::new()),
            poll_tx,
            poll_rx: Some(poll_rx),
        }
    }

    /// Extracts the watch channel receiver and the shared cache of
    /// confirmed asks. Call ONCE from `main.rs`, right after `new()`,
    /// to spawn `run_taker_poller` in a separate task (same pattern as
    /// `accounting::mark_price::MarkPriceCache`: network I/O lives outside the `Strategy`,
    /// which stays synchronous/pure). Tests never call this: the
    /// channel is simply left without a consumer, which is harmless.
    pub fn taker_poller_handles(
        &mut self,
    ) -> (
        mpsc::UnboundedReceiver<TakerWatch>,
        Arc<DashMap<String, Decimal>>,
    ) {
        (
            self.poll_rx
                .take()
                .expect("taker_poller_handles: already extracted"),
            self.taker_ready.clone(),
        )
    }

    /// Entry by wall clock: evaluates an already-expired candidate (`now >= clear_at`)
    /// and, if appropriate, places the GTD bid on the winning side. `cand` has already left the map
    /// of candidates (drained by the scan in `on_book`); this method decides the
    /// verdict and records the corresponding `PcsState` (active or `skipped`), without
    /// touching `self.candidates` again.
    fn try_enter(&mut self, cid: &str, cand: &Candidate, now: i64) -> Option<OrderIntent> {
        let secs_since_close = now - cand.clear_at;

        if secs_since_close > self.cfg.max_sweep_secs as i64 {
            metrics::counter!("hrf_pcs2_skip_total", "reason" => "late").increment(1);
            self.states
                .insert(cid.to_string(), PcsState::skipped(cand.clear_at));
            return None;
        }

        let Some((spot, spot_wall_ts)) = cand.spot_ref else {
            metrics::counter!("hrf_pcs2_skip_total", "reason" => "no_ref").increment(1);
            self.states
                .insert(cid.to_string(), PcsState::skipped(cand.clear_at));
            return None;
        };

        // Freshness relative to CLOSE, not relative to `now`: a spot frozen
        // too long before `clear_at` is no good as a reference even if the
        // clock evaluation happens immediately at close. `spot_wall_ts` is only
        // updated pre-close so in theory it is always <= clear_at, but
        // we saturate for safety instead of assuming it.
        let spot_age_at_close = cand.clear_at.saturating_sub(spot_wall_ts);
        if spot_age_at_close > self.cfg.max_spot_age_secs as i64 {
            metrics::counter!("hrf_pcs2_skip_total", "reason" => "stale_spot").increment(1);
            self.states
                .insert(cid.to_string(), PcsState::skipped(cand.clear_at));
            return None;
        }

        let (verdict, _) = deterministic_winner(
            cand.reference_price,
            Some(spot),
            false,
            self.cfg.min_rel_move,
        );
        if verdict == Verdict::Indeterminate {
            metrics::counter!("hrf_pcs2_skip_total", "reason" => "indeterminate").increment(1);
            self.states
                .insert(cid.to_string(), PcsState::skipped(cand.clear_at));
            return None;
        }

        let is_yes = verdict == Verdict::Yes;
        let winner_token = if is_yes {
            cand.yes_token.clone()
        } else {
            cand.no_token.clone()
        };

        let bid_size = calc_size(&self.cfg, self.cfg.maker_bid_price, true);
        if bid_size < Decimal::ONE || bid_size * self.cfg.maker_bid_price < Decimal::ONE {
            self.states
                .insert(cid.to_string(), PcsState::skipped(cand.clear_at));
            return None;
        }

        // Insert state before emitting: anti-double-entry (TC pattern).
        self.states.insert(
            cid.to_string(),
            PcsState {
                real_clear_at: cand.clear_at,
                is_yes,
                winner_token: winner_token.clone(),
                neg_risk: cand.neg_risk,
                have_bid: true,
                taker_fired: false,
                have_position: false,
                position_size: Decimal::ZERO,
                bid_placed_at: Some(Instant::now()),
                done: false,
            },
        );

        metrics::counter!("hrf_pcs2_bid_placed_total").increment(1);
        metrics::counter!("hrf_pcs2_entries_total").increment(1);
        tracing::info!(
            cid = %cid, is_yes, winner_token = %winner_token, secs_since_close,
            "PCS-2: post-close clock entry, GTD bid placed"
        );

        // There is no ask to look at here (no market book at this instant).
        // The initial taker sweep fires as soon as the first real post-close
        // book of the market arrives (management branch) OR as soon as the async
        // poller (`run_taker_poller`) finds a cheap ask over REST, whichever
        // happens first. Without this second path, a market with no book activity
        // would be at the mercy of the same delay (~53s on average) that the
        // maker entry already eliminated, reintroduced here for the
        // taker sweep.
        if self.cfg.taker_enabled {
            let budget = self
                .cfg
                .max_sweep_secs
                .saturating_sub(secs_since_close.max(0) as u64);
            let _ = self.poll_tx.send(TakerWatch {
                market_cid: cid.to_string(),
                token_id: winner_token.clone(),
                max_ask: self.cfg.taker_max_ask,
                deadline: Instant::now() + Duration::from_secs(budget),
            });
        }

        Some(gtd_bid_intent(
            cid,
            cand.neg_risk,
            &winner_token,
            self.cfg.maker_bid_price,
            bid_size,
            self.cfg.gtd_ttl_secs,
            cand.clear_at,
        ))
    }
}

impl Strategy for PostCloseSweep {
    fn id(&self) -> StrategyId {
        StrategyId::PostCloseSweep
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        let cid = &book.market_condition_id;

        if self.last_retain.elapsed().as_secs() >= 30 {
            self.states.retain(|_, s| now < s.real_clear_at + 300);
            self.candidates.retain(|_, c| now < c.clear_at + 300);
            self.last_retain = Instant::now();
        }

        let remaining = book.secs_to_clear();

        // ── Pre-close: upsert the candidate with the data of the current book ──
        if remaining > 0 {
            let entry = self
                .candidates
                .entry(cid.clone())
                .or_insert_with(|| Candidate {
                    clear_at: book.clear_at,
                    yes_token: book.yes.token_id.clone(),
                    no_token: book.no.token_id.clone(),
                    neg_risk: book.neg_risk,
                    reference_price: book.reference_price,
                    asset: vol.asset.clone(),
                    spot_ref: None,
                });
            entry.clear_at = book.clear_at;
            entry.yes_token = book.yes.token_id.clone();
            entry.no_token = book.no.token_id.clone();
            entry.neg_risk = book.neg_risk;
            entry.reference_price = book.reference_price;
            entry.asset = vol.asset.clone();
        }

        // ── Refresh of the frozen spot for ALL pre-close candidates ──
        // of the same asset. The oracle's fresh spot is valid for any
        // bracket of the asset, not just the market whose book arrived. Without
        // this, a market with no book activity in the last
        // max_spot_age_secs pre-close ended up in skip stale_spot even though the
        // oracle had a fresh price (46% of closes in the calibration).
        // The `now < clear_at` guard preserves the invariant of
        // PRE-close frozen spot. The runner guarantees `vol` belongs to the
        // asset of the received book (strategy/mod.rs), and an empty `asset` (no
        // oracle) never refreshes because the default vol always comes stale.
        // The stored timestamp must be the REAL instant at which the
        // oracle updated the price (`vol.updated_at`, monotonic), not the
        // tick clock (`now`). `is_stale_for_signal()` accepts up to
        // `VOL_SIGNAL_STALE_SECS` (5s) of age; with plain `now`, a
        // spot that was already 5s old when captured was recorded as
        // "from this instant", and `max_spot_age_secs` (8s) in `try_enter` measured
        // freshness wrongly: up to 8+5=13s of real age, not the 8s
        // the config declares. `vol.updated_at.elapsed()` is the real
        // monotonic age of the datum; it is subtracted from `now` (wall clock) to
        // obtain the real instant of the update.
        if !vol.is_stale_for_signal() && vol.last_price > Decimal::ZERO {
            let spot_wall_ts = now - vol.updated_at.elapsed().as_secs() as i64;
            for c in self.candidates.values_mut() {
                if c.asset == vol.asset && now < c.clear_at {
                    c.spot_ref = Some((vol.last_price, spot_wall_ts));
                }
            }
        }

        // ── Wall-clock scan: enters ALL expired candidates, ──
        // whether or not they belong to the market whose book arrived on this tick. The first
        // real post-close book of the market itself takes 53s on average to arrive
        // (measured); the useful window of dumps (p50=2s) is already lost
        // by then. The wall clock, fed by the continuous flow of
        // books from ALL markets, allows entering at the instant of close.
        let due: Vec<String> = self
            .candidates
            .iter()
            .filter(|(_, c)| now >= c.clear_at)
            .map(|(k, _)| k.clone())
            .collect();
        let mut intents = Vec::new();
        for due_cid in due {
            if let Some(cand) = self.candidates.remove(&due_cid) {
                if let Some(intent) = self.try_enter(&due_cid, &cand, now) {
                    intents.push(intent);
                }
            }
        }

        // ── Taker sweep via the async poller ────────────────────────────────
        // `run_taker_poller` writes here (REST, outside this synchronous thread)
        // as soon as it detects a cheap ask for a bracket that already entered by
        // clock. Like the scan above, ANY book from ANY
        // market works as a trigger; without this, the poller's finding would
        // sit waiting for a real book of the market itself to be applied.
        if self.cfg.taker_enabled && !self.taker_ready.is_empty() {
            // ALWAYS consume the entry from the DashMap: previously only
            // those that reached `hits` were removed (active state), so a
            // confirmed ask for a bracket that was already `done`, with a
            // position, or without state (e.g. TTL expired before the finding)
            // kept being re-polled and accumulating on every on_book.
            let keys: Vec<String> = self.taker_ready.iter().map(|e| e.key().clone()).collect();
            for hit_cid in keys {
                let fire = self
                    .states
                    .get(&hit_cid)
                    .is_some_and(|s| !s.taker_fired && !s.have_position && !s.done);
                if let Some((_, ask)) = self.taker_ready.remove(&hit_cid) {
                    if !fire {
                        continue;
                    }
                    if let Some(state) = self.states.get_mut(&hit_cid) {
                        let size = calc_size(&self.cfg, ask, false);
                        state.taker_fired = true;
                        // Do not emit a size-0 intent (caps at 0 or
                        // null ask); it was an empty FOK with taker_fired already set.
                        if size < Decimal::ONE || size * ask < Decimal::ONE {
                            continue;
                        }
                        metrics::counter!("hrf_pcs2_taker_fired_total").increment(1);
                        tracing::info!(cid = %hit_cid, ask = %ask,
                            "PCS-2: taker sweep fired by the async REST poller");
                        intents.push(taker_fok_intent_synthetic(
                            &hit_cid,
                            state.neg_risk,
                            &state.winner_token,
                            ask,
                            size,
                            state.real_clear_at,
                        ));
                    }
                }
            }
        }

        if remaining > 0 {
            return intents;
        }

        let secs_since_close = -remaining;

        // ── Existing state: management (only if the received book IS the market's) ──
        if let Some(state) = self.states.get_mut(cid.as_str()) {
            if state.done || state.have_position {
                return intents;
            }

            // TTL of the in-flight GTD: no fill after gtd_ttl_secs → cancel and do not retry.
            if state.have_bid {
                if let Some(placed_at) = state.bid_placed_at {
                    if placed_at.elapsed().as_secs() >= self.cfg.gtd_ttl_secs {
                        state.done = true;
                        metrics::counter!("hrf_pcs2_ttl_cancel_total").increment(1);
                        intents.push(cancel_intent(book, &state.winner_token));
                        return intents;
                    }
                }
            }

            // Late sweep: the taker has not fired yet and the winner's ask is still cheap.
            // This is now the only taker path: the clock entry has no
            // market book (there is no ask to look at); the first real post-close
            // book to arrive here does fire it.
            if !state.taker_fired
                && self.cfg.taker_enabled
                && secs_since_close <= self.cfg.max_sweep_secs as i64
            {
                let winner_ask = if state.is_yes {
                    book.yes.best_ask()
                } else {
                    book.no.best_ask()
                };
                if let Some(ask) = winner_ask {
                    if ask <= self.cfg.taker_max_ask {
                        let size = calc_size(&self.cfg, ask, false);
                        state.taker_fired = true;
                        metrics::counter!("hrf_pcs2_taker_fired_total").increment(1);
                        // The executor cancels the active GTD for anti-duplication when
                        // it receives this non-resting order; no explicit Cancel needed.
                        intents.push(taker_fok_intent(book, &state.winner_token, ask, size));
                        return intents;
                    }
                }
            }

            // Re-place the GTD if there is no bid in flight and we are still within the window.
            if !state.have_bid
                && !state.taker_fired
                && secs_since_close <= self.cfg.max_sweep_secs as i64
            {
                let price = self.cfg.maker_bid_price;
                let size = calc_size(&self.cfg, price, true);
                if size >= Decimal::ONE && size * price >= Decimal::ONE {
                    state.have_bid = true;
                    state.bid_placed_at = Some(Instant::now());
                    intents.push(gtd_bid_intent(
                        &book.market_condition_id,
                        book.neg_risk,
                        &state.winner_token,
                        price,
                        size,
                        self.cfg.gtd_ttl_secs,
                        state.real_clear_at,
                    ));
                    return intents;
                }
            }

            return intents;
        }

        // ── No state and no candidate: there was never a pre-close tick for this ──
        // market (startup mid-bracket, or the candidate was evicted) →
        // no reference for the verdict, nothing to evaluate.
        metrics::counter!("hrf_pcs2_skip_total", "reason" => "no_ref").increment(1);
        self.states
            .insert(cid.clone(), PcsState::skipped(book.clear_at));
        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::PostCloseSweep {
            return;
        }
        let Some(state) = self.states.get_mut(&fill.market_cid) else {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "PCS-2: Buy fill without state, position left to the orphan checker (restart mid-bracket)");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "post_close_sweep").increment(1);
            }
            return;
        };
        if fill.token_id != state.winner_token {
            return;
        }

        match fill.side {
            FillSide::Buy => {
                state.have_position = true;
                state.position_size += fill.size;
                state.have_bid = false;
                if state.taker_fired {
                    metrics::counter!("hrf_pcs2_taker_fills_total").increment(1);
                } else {
                    metrics::counter!("hrf_pcs2_maker_fills_total").increment(1);
                }
                tracing::info!(
                    cid = %fill.market_cid, price = %fill.price, size = %fill.size,
                    "PCS-2: winning-side fill → position open"
                );
            }
            // The confirmed cancellation clears the same state as the rejection.
            FillSide::Rejected | FillSide::Cancelled => {
                if state.taker_fired {
                    state.taker_fired = false;
                    state.have_bid = false;
                    metrics::counter!("hrf_pcs2_taker_rejected_total").increment(1);
                } else {
                    state.have_bid = false;
                }
            }
            // The intent was blocked by a pre-submit gate: the order
            // never existed. The emission latches (`have_bid`,
            // `taker_fired`) must be released or the bracket goes silent; if an
            // earlier order were still alive, the executor's LIVE-8 anti-dup reconciles when reposting.
            FillSide::GateRejected => {
                if state.taker_fired {
                    state.taker_fired = false;
                }
                if !state.have_position {
                    state.have_bid = false;
                }
            }
            FillSide::Sell => {}
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Natural size according to the config caps (same pattern as ThetaCapture).
/// `trunc_int=true` truncates to whole shares (required for GTD); the taker sweep does not need it.
fn calc_size(cfg: &PostCloseSweepConfig, price: Decimal, trunc_int: bool) -> Decimal {
    if price <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    let cap_usdc =
        (cfg.limits.max_order_usdc > Decimal::ZERO).then(|| cfg.limits.max_order_usdc / price);
    let cap_sh = (cfg.max_size_shares > Decimal::ZERO).then_some(cfg.max_size_shares);
    let nat = match (cap_usdc, cap_sh) {
        (Some(u), Some(s)) => u.min(s),
        (Some(u), None) => u,
        (None, Some(s)) => s,
        (None, None) => dec!(10),
    };
    if trunc_int {
        nat.trunc_with_scale(0)
    } else {
        nat
    }
}

/// Maker GTD bid on the winning side. SYNTHETIC FUTURE `clear_at` (`now + gtd_ttl_secs
/// + 60`), NOT the bracket's real `clear_at` (already past, it is post-close): sim_router.rs
/// purges GTDs with a past `clear_at` on every tick, and clob_router.rs computes the
/// expiry based on that `clear_at`; with the real (past) one the CLOB would expire it
/// instantly and the strategy would silently get 0 maker fills.
///
/// That synthetic value must NOT leak into the `Fill`; hence
/// `real_clear_at` (`OrderIntent::resting_custom_expiry`), the bracket's REAL close,
/// which the executor uses when building the fill instead of the synthetic one.
///
/// Takes `market_cid`/`neg_risk` separately (not `&BookSnapshot`) because the
/// clock entry (`try_enter`) has no market book at the instant of close, only
/// the `Candidate` with the data captured pre-close.
fn gtd_bid_intent(
    market_cid: &str,
    neg_risk: bool,
    token_id: &str,
    price: Decimal,
    size: Decimal,
    gtd_ttl_secs: u64,
    real_clear_at: i64,
) -> OrderIntent {
    let synthetic_clear_at = crate::time::now().timestamp() + gtd_ttl_secs as i64 + 60;
    OrderIntent::resting_custom_expiry(
        market_cid.to_string(),
        token_id.to_string(),
        OrderSide::Buy,
        price,
        size,
        price, // signal
        StrategyId::PostCloseSweep,
        neg_risk,
        synthetic_clear_at,
        real_clear_at,
    )
}

/// Taker FOK sweep on the winning side. It uses the book's REAL `clear_at` (already past, which
/// does not matter: FOK is not resting, no GTD cron purges it). It does not have the bug
/// of `gtd_bid_intent`: it never used a synthetic `clear_at`, so `taker()` (which
/// always takes `book.clear_at`) is equivalent without needing `real_clear_at`.
fn taker_fok_intent(
    book: &BookSnapshot,
    token_id: &str,
    price: Decimal,
    size: Decimal,
) -> OrderIntent {
    OrderIntent::taker(
        book.market_condition_id.clone(),
        token_id.to_string(),
        OrderSide::Buy,
        price,
        size,
        price, // signal
        StrategyId::PostCloseSweep,
        book,
    )
}

/// Taker FOK sweep fired by the async poller (`run_taker_poller`), without
/// `&BookSnapshot` at hand, same as `gtd_bid_intent` (see its doc): the cheap
/// ask was detected over REST in a separate task, not on the current tick.
fn taker_fok_intent_synthetic(
    market_cid: &str,
    neg_risk: bool,
    token_id: &str,
    price: Decimal,
    size: Decimal,
    clear_at: i64,
) -> OrderIntent {
    OrderIntent {
        market_cid: market_cid.to_string(),
        token_id: token_id.to_string(),
        side: OrderSide::Buy,
        price,
        size,
        clear_at,
        signal: price,
        neg_risk,
        limit_price: Some(price),
        second_leg: None,
        strategy: StrategyId::PostCloseSweep,
        resting: false,
        real_clear_at: None,
    }
}

fn cancel_intent(book: &BookSnapshot, token_id: &str) -> OrderIntent {
    super::cancel_intent(
        &book.market_condition_id,
        token_id,
        book.clear_at,
        book.neg_risk,
        StrategyId::PostCloseSweep,
    )
}

// ── Async REST poller for the taker sweep ──────────────────────────────────
//
// Lives outside the synchronous `Strategy` thread (same reason as
// `src/accounting/mark_price.rs`): network I/O cannot sneak into `on_book`, which runs in the
// shared loop of the strategy runner. This poller ONLY writes to the
// `taker_ready` cache; it never emits `OrderIntent` directly, so as not to
// race with the synchronous management branch (which can already fire
// the same sweep if the market's real book arrives first). Spawn from
// `main.rs` with `PostCloseSweep::taker_poller_handles()`.

#[derive(serde::Deserialize)]
struct PollerRawLevel {
    price: String,
    size: String,
}

#[derive(serde::Deserialize)]
struct PollerBookResponse {
    #[serde(default)]
    asks: Vec<PollerRawLevel>,
}

/// Best ask of the book: the API does not guarantee order (verified with real data
/// in `src/accounting/mark_price.rs::mid_from_raw`, same scenario here), so it is
/// sorted explicitly before taking the minimum.
async fn fetch_best_ask(http: &reqwest::Client, token_id: &str) -> anyhow::Result<Option<Decimal>> {
    let body: PollerBookResponse = http
        .get(format!("{CLOB_BASE}/book"))
        .query(&[("token_id", token_id)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let mut asks: Vec<Decimal> = body
        .asks
        .iter()
        .filter_map(|l| {
            let price = Decimal::from_str(&l.price).ok()?;
            let size = Decimal::from_str(&l.size).ok()?;
            (size > Decimal::ZERO).then_some(price)
        })
        .collect();
    asks.sort_unstable();
    Ok(asks.first().copied())
}

/// Watches a single bracket until an ask <= `w.max_ask` appears, `w.deadline`
/// expires, or another path (synchronous management branch) has already resolved it
/// (`ready.contains_key`; we avoid hitting the CLOB needlessly).
async fn poll_one(w: TakerWatch, ready: Arc<DashMap<String, Decimal>>, http: Arc<reqwest::Client>) {
    let mut interval = tokio::time::interval(Duration::from_millis(TAKER_POLL_INTERVAL_MS));
    loop {
        interval.tick().await;
        if Instant::now() >= w.deadline || ready.contains_key(&w.market_cid) {
            return;
        }
        match fetch_best_ask(&http, &w.token_id).await {
            Ok(Some(ask)) if ask <= w.max_ask => {
                ready.insert(w.market_cid.clone(), ask);
                return;
            }
            Ok(_) => {}
            Err(e) => tracing::debug!(cid = %w.market_cid, error = %e,
                "PCS-2 poller: error querying book, will retry on the next cycle"),
        }
    }
}

/// Background task: for each bracket that enters by clock (`TakerWatch` received
/// via `poll_tx`), spawns an independent `poll_one`, self-limited by its
/// own `deadline`, with no shared state or single ticker to manage.
pub async fn run_taker_poller(
    mut watch_rx: mpsc::UnboundedReceiver<TakerWatch>,
    ready: Arc<DashMap<String, Decimal>>,
    mut shutdown: watch::Receiver<bool>,
) {
    let http = Arc::new(
        reqwest::Client::builder()
            .user_agent(crate::POLY_USER_AGENT)
            .timeout(Duration::from_secs(5))
            .build()
            .expect("pcs2 taker poller: reqwest client"),
    );

    tracing::info!("PCS-2: async REST poller for the taker sweep started");

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => { if *shutdown.borrow() { break; } }
            watch = watch_rx.recv() => {
                match watch {
                    Some(w) => { tokio::spawn(poll_one(w, ready.clone(), http.clone())); }
                    None => break, // PostCloseSweep dropped, nothing left to watch
                }
            }
        }
    }
    tracing::info!("PCS-2: async REST poller for the taker sweep stopped");
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};
    use std::thread::sleep;
    use std::time::Duration;

    /// The clock scan (`try_enter`) compares the candidate's `clear_at`
    /// against the real wall clock. Unlike the earlier design (which
    /// read `book.secs_to_clear()` from the book that just arrived), here it is NOT enough
    /// to pass `make_book` a negative `remaining` to simulate "already closed":
    /// real time must have passed the `clear_at` captured on
    /// the pre-close tick. The tests that exercise the entry use a short
    /// `remaining` (1s) for the pre-close book and wait that real time.
    fn wait_past_close() {
        sleep(Duration::from_millis(1100));
    }

    fn make_book(yes_ask: &str, no_ask: &str, remaining: i64, reference: &str) -> BookSnapshot {
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let now = crate::time::now().timestamp();
        BookSnapshot {
            market_condition_id: "pcs2-cid".into(),
            yes: TokenBook {
                token_id: "yes-tok".into(),
                bids: bids("0.01"),
                asks: asks(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-tok".into(),
                bids: bids("0.01"),
                asks: asks(no_ask),
                ..Default::default()
            },
            market_open_ts: now - 300,
            clear_at: now + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-pcs2".into(),
            reference_price: Some(reference.parse().unwrap()),
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        }
    }

    fn make_vol(live: &str) -> VolSnapshot {
        VolSnapshot {
            asset: "btc".into(),
            sigma_short: Some(dec!(0.001)),
            sigma_long: Some(dec!(0.001)),
            last_price: live.parse().unwrap(),
            updated_at: Instant::now(),
            // No CVD in this fixture.
            ..VolSnapshot::default()
        }
    }

    fn test_cfg() -> PostCloseSweepConfig {
        PostCloseSweepConfig {
            limits: crate::config::StrategyLimits {
                enabled: true,
                max_orders_per_hour: 20,
                max_order_usdc: dec!(10),
                daily_loss_usdc_stop: dec!(20),
                ..crate::config::StrategyLimits::default()
            },
            ..PostCloseSweepConfig::default()
        }
    }

    // ── on_book ───────────────────────────────────────────────────────────────

    /// CRITICAL: the post-close verdict must use the spot frozen BEFORE
    /// close, not the live spot of the post-close tick. Pre-close: spot 100100
    /// (> strike 100000 → YES). Post-close: vol inverted to 99900 (< strike →
    /// would be NO live); the entry must still be on YES. The entry
    /// is now fired by wall clock, so real time must pass
    /// between the pre-close tick and the post-close one.
    #[test]
    fn freezes_pre_close_spot_and_does_not_invert_verdict() {
        let mut pcs = PostCloseSweep::new(test_cfg());

        let book_pre = make_book("0.60", "0.40", 1, "100000");
        let vol_pre = make_vol("100100");
        assert!(
            pcs.on_book(&book_pre, &vol_pre).is_empty(),
            "pre-close must not emit intents"
        );

        wait_past_close();

        let book_post = make_book("0.99", "0.01", -2, "100000");
        let vol_post = make_vol("99900"); // inverted: must not be used
        let intents = pcs.on_book(&book_post, &vol_post);

        assert_eq!(intents.len(), 1);
        assert_eq!(
            intents[0].token_id, "yes-tok",
            "must enter on YES (frozen spot), not on NO (live spot)"
        );
    }

    /// The clock entry does not need the market's own book: any
    /// book that arrives after the candidate's `clear_at` has passed
    /// (by wall clock) fires it.
    #[test]
    fn entry_by_clock_without_market_book() {
        let mut pcs = PostCloseSweep::new(test_cfg());

        let book_pre = make_book("0.60", "0.40", 1, "100000");
        let vol_pre = make_vol("100100");
        assert!(pcs.on_book(&book_pre, &vol_pre).is_empty());

        wait_past_close();

        // Book of ANOTHER market (different cid), still pre-close for itself.
        let mut other_book = make_book("0.55", "0.45", 30, "50000");
        other_book.market_condition_id = "other-cid".into();
        let vol_other = make_vol("50000");
        let intents = pcs.on_book(&other_book, &vol_other);

        assert_eq!(
            intents.len(),
            1,
            "must emit the expired candidate's GTD even though the book is from another market"
        );
        assert_eq!(intents[0].token_id, "yes-tok");
        assert_eq!(
            intents[0].market_cid, "pcs2-cid",
            "the GTD belongs to the candidate's market, not to the received book"
        );
    }

    /// Freshness of the frozen spot relative to CLOSE, not relative to the instant
    /// of evaluation: a spot captured 20s before `clear_at` is too
    /// stale (max_spot_age_secs=8 by default) even though the clock scan
    /// evaluates the candidate right at the instant of close.
    #[test]
    fn stale_spot_is_by_age_at_close_not_by_evaluation_delay() {
        let mut pcs = PostCloseSweep::new(test_cfg()); // max_spot_age_secs default = 8

        let clear_at = crate::time::now().timestamp();
        pcs.candidates.insert(
            "pcs2-cid".into(),
            Candidate {
                clear_at,
                yes_token: "yes-tok".into(),
                no_token: "no-tok".into(),
                neg_risk: false,
                reference_price: Some(dec!(100000)),
                asset: "btc".into(),
                spot_ref: Some((dec!(100100), clear_at - 20)), // spot from 20s ago
            },
        );

        // Any book can fire the clock scan; the candidate
        // is already expired (now >= clear_at) as soon as on_book is called.
        let book = make_book("0.99", "0.01", -1, "100000");
        let vol = make_vol("100100");
        let intents = pcs.on_book(&book, &vol);

        assert!(
            intents.is_empty(),
            "20s-old spot > max_spot_age_secs(8) → skip stale_spot"
        );
    }

    /// The frozen spot of a candidate whose book went
    /// inactive must be refreshed with ticks from OTHER markets of the same
    /// asset; the oracle belongs to the asset, not to the market. Without this
    /// that candidate would end up in skip stale_spot (20s spot > max of 8s).
    #[test]
    fn refreshes_spot_with_ticks_of_another_market_of_same_asset() {
        let mut pcs = PostCloseSweep::new(test_cfg());

        let clear_at = crate::time::now().timestamp() + 1;
        pcs.candidates.insert(
            "pcs2-cid".into(),
            Candidate {
                clear_at,
                yes_token: "yes-tok".into(),
                no_token: "no-tok".into(),
                neg_risk: false,
                reference_price: Some(dec!(100000)),
                asset: "btc".into(),
                spot_ref: Some((dec!(99000), clear_at - 20)), // stale AND on the opposite side
            },
        );

        // Pre-close tick of ANOTHER btc market with fresh vol → refreshes the frozen
        // spot of the candidate (and with it the verdict becomes YES).
        let mut other_book = make_book("0.55", "0.45", 30, "50000");
        other_book.market_condition_id = "other-cid".into();
        assert!(pcs.on_book(&other_book, &make_vol("100100")).is_empty());

        wait_past_close();

        // Any other tick fires the clock entry: without the refresh
        // it would have been skip stale_spot; with it, it enters on YES (spot 100100 > strike).
        let intents = pcs.on_book(&other_book, &make_vol("99000")); // post-close vol must not be used
        assert_eq!(
            intents.len(),
            1,
            "the candidate refreshed by another market must enter"
        );
        assert_eq!(intents[0].token_id, "yes-tok");
        assert_eq!(intents[0].market_cid, "pcs2-cid");
    }

    /// `vol.updated_at` can be up to `VOL_SIGNAL_STALE_SECS`
    /// (5s) old and still pass `is_stale_for_signal()`. If
    /// `spot_ref` is stamped with the TICK clock (`now`) instead of the oracle's
    /// real timestamp, the real age of the datum is underestimated, by up to
    /// 5s more than `max_spot_age_secs` declares (8s by default → up to
    /// 13s real). The fix uses `vol.updated_at.elapsed()` to rebuild the
    /// real instant of the update.
    #[test]
    fn spot_ref_uses_real_oracle_timestamp_not_the_tick() {
        let mut pcs = PostCloseSweep::new(test_cfg());
        let mut vol = make_vol("100100");
        // Oracle with 4s of real age (< 5s of VOL_SIGNAL_STALE_SECS,
        // passes is_stale_for_signal) but NOT freshly captured on this tick.
        vol.updated_at = crate::time::Instant::now() - std::time::Duration::from_secs(4);

        let book = make_book("0.55", "0.45", 120, "100000");
        let before = crate::time::now().timestamp();
        assert!(
            pcs.on_book(&book, &vol).is_empty(),
            "pre-close must not emit intents"
        );
        let after = crate::time::now().timestamp();

        let (_, spot_wall_ts) = pcs
            .candidates
            .get("pcs2-cid")
            .expect("candidate must exist")
            .spot_ref
            .expect("spot_ref must be set — the oracle is fresh");
        assert!(spot_wall_ts <= before - 3,
            "spot_wall_ts must reflect the oracle's real age (~4s ago), not the tick clock");
        assert!(spot_wall_ts <= after);
    }

    #[test]
    fn skip_without_ref() {
        let mut pcs = PostCloseSweep::new(test_cfg());
        // Post-close book with no prior pre-close tick → no candidate.
        let book = make_book("0.99", "0.01", -2, "100000");
        let vol = make_vol("100100");
        assert!(pcs.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn skip_indeterminate() {
        let mut pcs = PostCloseSweep::new(test_cfg());

        let book_pre = make_book("0.60", "0.40", 1, "100000");
        // rel_move = 0.0007 < min_rel_move (0.001) → indeterminate.
        let vol_pre = make_vol("100070");
        pcs.on_book(&book_pre, &vol_pre);

        wait_past_close();

        let book_post = make_book("0.99", "0.01", -2, "100000");
        let vol_post = make_vol("100070");
        assert!(pcs.on_book(&book_post, &vol_post).is_empty());
    }

    #[test]
    fn gtd_maker_price_and_size() {
        let mut pcs = PostCloseSweep::new(test_cfg());

        let book_pre = make_book("0.60", "0.40", 1, "100000");
        let vol_pre = make_vol("100100");
        pcs.on_book(&book_pre, &vol_pre);

        wait_past_close();

        let book_post = make_book("0.99", "0.01", -2, "100000");
        let vol_post = make_vol("100100");
        let intents = pcs.on_book(&book_post, &vol_post);

        assert_eq!(intents.len(), 1);
        let intent = &intents[0];
        assert!(intent.resting, "the entry bid must be a GTD (resting)");
        assert_eq!(intent.price, dec!(0.99));
        // size = max_order_usdc(10) / 0.99 = 10.10... truncated a 10.
        assert_eq!(intent.size, dec!(10));
        let now = crate::time::now().timestamp();
        assert!(
            intent.clear_at > now + 60,
            "clear_at must be a synthetic future one, not the real (past) clear_at"
        );
        // The real close (for the fill/orphan checker)
        // must be kept separate, in real_clear_at; it must not leak into the synthetic one.
        assert_eq!(
            intent.real_clear_at,
            Some(book_pre.clear_at),
            "real_clear_at must be the bracket's real close, not the GTD's synthetic buffer"
        );
    }

    #[test]
    fn taker_fok_when_ask_cheap() {
        let mut cfg = test_cfg();
        cfg.taker_enabled = true;
        let mut pcs = PostCloseSweep::new(cfg);

        let book_pre = make_book("0.60", "0.40", 1, "100000");
        let vol_pre = make_vol("100100");
        pcs.on_book(&book_pre, &vol_pre);

        wait_past_close();

        // yes_ask=0.95 <= taker_max_ask (0.98 by default) → fires the taker sweep
        // in the management of the same tick (the post-close book DOES have a real ask).
        let book_post = make_book("0.95", "0.02", -2, "100000");
        let vol_post = make_vol("100100");
        let intents = pcs.on_book(&book_post, &vol_post);

        assert_eq!(
            intents.len(),
            2,
            "must emit GTD (clock entry) + FOK (management of the same tick)"
        );
        assert!(intents[0].resting);
        assert!(!intents[1].resting);
        assert_eq!(intents[1].price, dec!(0.95));
        assert_eq!(
            intents[1].clear_at, book_post.clear_at,
            "the taker sweep uses the book's REAL clear_at"
        );
    }

    /// The taker leg used to fire only from the
    /// management branch, which needs a real book from the market ITSELF. When
    /// the candidate expires via the book of ANOTHER market (the dominant scenario of the
    /// wall clock, which is what it was designed for), a cheap ask detected by the
    /// async REST poller (`run_taker_poller`, whose network logic is
    /// tested separately; here its effect is simulated by inserting directly into
    /// `taker_ready`) must be applied on the next tick of ANY
    /// market, without waiting for the market's own book, which is exactly the ~53s delay
    /// already eliminated for the maker side.
    #[test]
    fn poller_taker_ready_applies_on_any_tick() {
        let mut cfg = test_cfg();
        cfg.taker_enabled = true;
        let mut pcs = PostCloseSweep::new(cfg);

        let book_pre = make_book("0.60", "0.40", 1, "100000");
        let vol_pre = make_vol("100100");
        assert!(pcs.on_book(&book_pre, &vol_pre).is_empty());

        wait_past_close();

        // Enters by clock via the book of ANOTHER market, with no own ask at hand;
        // the taker sweep is left to the poller.
        let mut other_book = make_book("0.55", "0.45", 30, "50000");
        other_book.market_condition_id = "other-cid".into();
        let intents1 = pcs.on_book(&other_book, &make_vol("50000"));
        assert_eq!(
            intents1.len(),
            1,
            "only the entry GTD, no own ask at hand"
        );
        assert!(intents1[0].resting);

        // The poller "found" a cheap ask: simulates what run_taker_poller
        // would write after querying the CLOB over REST.
        pcs.taker_ready.insert("pcs2-cid".into(), dec!(0.95));

        // Any tick from ANY market (here, a third one) must pick up
        // the finding and fire the taker sweep, without waiting for the market's own book.
        let mut third_book = make_book("0.50", "0.50", 30, "70000");
        third_book.market_condition_id = "third-cid".into();
        let intents2 = pcs.on_book(&third_book, &make_vol("70000"));

        assert_eq!(
            intents2.len(),
            1,
            "must fire the taker sweep for the poller's finding"
        );
        assert!(!intents2[0].resting, "it is a FOK, not a GTD");
        assert_eq!(intents2[0].market_cid, "pcs2-cid");
        assert_eq!(intents2[0].token_id, "yes-tok");
        assert_eq!(intents2[0].price, dec!(0.95));
    }

    #[test]
    fn ttl_cancels_gtd() {
        let mut cfg = test_cfg();
        cfg.gtd_ttl_secs = 0;
        let mut pcs = PostCloseSweep::new(cfg);

        let book_pre = make_book("0.60", "0.40", 1, "100000");
        let vol_pre = make_vol("100100");
        pcs.on_book(&book_pre, &vol_pre);

        wait_past_close();

        let book_post = make_book("0.99", "0.01", -2, "100000");
        let vol_post = make_vol("100100");
        let i1 = pcs.on_book(&book_post, &vol_post);
        // With TTL=0 the clock entry (due-scan) and the management of the
        // market's own book run in the SAME call: it places the GTD and, right
        // after, cancels it for expired TTL; there is no separate "next
        // tick" from the entry one.
        assert_eq!(
            i1.len(),
            2,
            "places the GTD and cancels it on the same tick (TTL=0)"
        );
        assert!(i1[0].resting, "first intent: entry GTD");
        assert!(
            matches!(i1[1].side, OrderSide::Cancel),
            "second intent: cancel due to expired TTL"
        );

        let i2 = pcs.on_book(&book_post, &vol_post);
        assert!(i2.is_empty(), "state already done — do not retry");
    }

    #[test]
    fn no_double_entry() {
        let mut pcs = PostCloseSweep::new(test_cfg());

        let book_pre = make_book("0.60", "0.40", 1, "100000");
        let vol_pre = make_vol("100100");
        pcs.on_book(&book_pre, &vol_pre);

        wait_past_close();

        let book_post = make_book("0.99", "0.01", -2, "100000");
        let vol_post = make_vol("100100");
        let i1 = pcs.on_book(&book_post, &vol_post);
        assert_eq!(i1.len(), 1);

        let i2 = pcs.on_book(&book_post, &vol_post);
        assert!(
            i2.is_empty(),
            "must not re-place the GTD while it is still in flight"
        );
    }
}
