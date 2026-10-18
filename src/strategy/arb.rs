//! ARB strategy (parity arbitrage).
//!
//! Signal: `1 − yes_ask − no_ask − parity_fee > fee_threshold + min_edge_after_fee`
//! where `parity_fee = Σ 0.07 × p × (1−p)` is the exact parabolic taker fee of
//! both legs (`fair_value::fee_taker_parabolic_with_rate`; the earlier triangular
//! model overestimated ~2x in the mid zone where ARB operates).
//!
//! Entry: simultaneous buy of YES + NO with `second_leg` in the executor.
//!
//! If YES fills but NO fails (or vice versa), ARB detects the orphan after 2 ticks:
//!   1. Retries the missing leg as resting at the originally observed price.
//!   2. Trailing stop (bid drops ≥5% from the peak, 2 consecutive ticks) + bail
//!      on adverse oracle or time. 10 s gate before considering a FOK rejected.

use std::collections::HashMap;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    config::defaults,
    types::{BookSnapshot, Fill, FillSide, OrderIntent, OrderSide, StrategyId, VolSnapshot},
};

use super::Strategy;

/// Configuration of the ARB strategy (parity arbitrage).
/// Describes the parity arbitrage on YES + NO legs.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ArbConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    /// Additional flat margin over the exact fee. With `taker_fee_rate` computing
    /// the pair's real fee, this is only extra cushion (0 = no cushion).
    #[serde(default = "defaults::arb_fee_threshold")]
    pub fee_threshold: Decimal,
    /// Polymarket taker fee rate: `fee = rate × min(p, 1−p)` per leg.
    /// The pair's fee depends on price (≈5¢ at p=0.5, ≈0.1¢ at the extremes) —
    /// a flat threshold lets EV-negative trades through in the mid zone.
    #[serde(default = "defaults::arb_taker_fee_rate")]
    pub taker_fee_rate: Decimal,
    #[serde(default = "defaults::arb_min_edge")]
    pub min_edge_after_fee: Decimal,
    #[serde(default = "defaults::arb_max_size_shares")]
    pub max_size_shares: Decimal,
    /// Seconds without a fill on the orphan sell before retrying. Time
    /// deadline: N coalesced ticks do not guarantee real time.
    #[serde(default = "defaults::arb_sell_retry_secs")]
    pub orphan_sell_retry_secs: u64,
    /// Seconds after one leg's fill before treating it as an orphan
    /// (covers the async arrival of the opposite fill via user_ws, ~0.7-2s). Once
    /// confirmed, ALL bails are enabled — previously there was a blind window
    /// of 20 ticks with no stop after a partial fill.
    #[serde(default = "defaults::arb_orphan_confirm_secs")]
    pub orphan_confirm_secs: u64,
    /// Maximum tolerable bid-ask spread on EACH leg. 0 = no filter.
    /// Wide spread → illiquidity → high risk on orphan bail.
    #[serde(default)]
    pub max_bid_ask_spread: Decimal,
    /// Slippage tolerance for BOTH legs: each FOK is signed at
    /// `ask + leg_slippage` and requires `edge >= min_edge + 2×leg_slippage`.
    /// Absorbs the 1-tick move between reading the WS book and the POST
    /// to the CLOB. Previously only the NO leg carried slippage — YES died on
    /// a 1-tick move and produced NO orphans.
    #[serde(default, alias = "no_leg_slippage")]
    pub leg_slippage: Decimal,
    /// Hard stop-loss for orphan legs — sell if the bid falls below
    /// `entry × (1 − orphan_stop_loss_pct)`. The trailing stop only
    /// takes profits (`bid > entry`); without this a losing orphan had no
    /// exit until the oracle bail (−2%) or the time bail. 0 = disabled.
    #[serde(default = "defaults::arb_orphan_stop_loss")]
    pub orphan_stop_loss_pct: Decimal,
    /// Minimum remaining time of the bracket to enter. In the live pilot
    /// all 5 entries were on books collapsing near
    /// expiry (pairs completeness 0/5): the parity "edge" was
    /// the collapse itself. Previously a hardcoded 10s.
    #[serde(default = "defaults::arb_min_remaining_secs")]
    pub min_remaining_secs: i64,
    /// Safety factor over the depth at the best ask — we only
    /// take `depth / factor` shares, requiring spare liquidity.
    /// A thin quote that disappears before the POST cannot fill us. ≥ 1.
    #[serde(default = "defaults::arb_depth_safety_factor")]
    pub depth_safety_factor: Decimal,
    /// Maximum credible edge — above it, the "arbitrage" is the dead
    /// part of the book (phantom ask), not an opportunity. 0 = no filter.
    #[serde(default = "defaults::arb_max_plausible_edge")]
    pub max_plausible_edge: Decimal,
}

impl crate::strategy::StrategyConfig for ArbConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

/// Seconds before the close at which the orphan leg's sell is forced.
const ORPHAN_BAIL_SECS: i64 = 30;

/// Minimum sellable: the router truncates shares to 2dp; a smaller remainder
/// can never be sold and only produces endless retries. It is written off.
const DUST_SHARES: Decimal = dec!(0.01);

// ── Per-bracket state ─────────────────────────────────────────────────────────

struct BracketState {
    clear_at: i64,
    yes_token: String,
    no_token: String,
    /// NO ask price observed when the intent was emitted — used exactly for the resting order.
    no_price: Decimal,
    /// Size of the NO fill (0 until on_fill confirms).
    no_size: Decimal,
    /// Price and size of the YES fill (0 until on_fill confirms).
    yes_entry: Decimal,
    yes_size: Decimal,
    have_yes: bool,
    have_no: bool,
    /// Instant the state was created (cleanup of doubly rejected FOKs).
    created_at: crate::time::Instant,
    /// Instant at which only one leg was left filled. After `orphan_confirm_secs`
    /// (covers the async fill of the other leg) ALL bails are enabled — before
    /// it was a 20-tick counter: a blind window with no stop under coalescing.
    orphan_since: Option<crate::time::Instant>,
    /// We already sent the NO resting retry.
    retry_sent: bool,
    /// We already sent the sell of the orphan leg (YES or NO).
    sell_sent: bool,
    /// YES sell fill confirmed — the resting NO must be canceled if it exists.
    yes_sold: bool,
    /// Any sell confirmed (YES or NO) — freezes the state.
    sell_confirmed: bool,
    /// We already emitted Cancel for the resting NO after selling YES.
    cancel_sent: bool,
    /// Instant of the in-flight sell — time deadline for the retry.
    sell_sent_at: Option<crate::time::Instant>,
    /// Maximum YES bid seen after the fill (for the trailing stop).
    peak_bid: Decimal,
    /// Consecutive ticks in which the bid is below the trailing stop
    /// (2-tick noise filter; the large gate is already time-based — orphan_since).
    trail_ticks: u32,
    /// Last observed bid of the orphan leg — informational reference
    /// for the watchdog's sells when the feed is quiet.
    last_bid: Decimal,
    /// neg_risk flag of the book that originated the state — the watchdog's
    /// intents need it and no fresh book is available.
    neg_risk: bool,
}

/// Seconds before cleaning up a state with both legs rejected (the YES fill
/// may arrive async via user_ws up to ~2s late; 10s gives ample margin).
const REJECTED_CLEANUP_SECS: u64 = 10;

// ── Arb ───────────────────────────────────────────────────────────────────────

pub struct Arb {
    cfg: ArbConfig,
    states: HashMap<String, BracketState>,
    last_retain: crate::time::Instant,
}

impl Arb {
    pub fn new(cfg: ArbConfig) -> Self {
        Self {
            cfg,
            states: HashMap::new(),
            last_retain: crate::time::Instant::now(),
        }
    }
}

impl Strategy for Arb {
    fn id(&self) -> StrategyId {
        StrategyId::Arb
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        let cid = &book.market_condition_id;
        let remaining = book.secs_to_clear();

        // Throttle to 30s to amortize the cost of retain over all names on every tick.
        if self.last_retain.elapsed() >= std::time::Duration::from_secs(30) {
            self.states.retain(|_, s| {
                // Expired → clean up.
                if s.clear_at <= now {
                    return false;
                }
                // FOK rejected on both legs and YES never arrived (time
                // deadline): clean up to allow retrying on the next edge.
                if !s.have_yes
                    && !s.have_no
                    && s.created_at.elapsed().as_secs() >= REJECTED_CLEANUP_SECS
                {
                    return false;
                }
                true
            });
            self.last_retain = crate::time::Instant::now();
        }

        let mut intents = vec![];

        // ── Orphan leg management (YES open, NO unfilled) ───────────────────

        if let Some(state) = self.states.get_mut(cid.as_str()) {
            // !have_yes && !have_no: FOK rejected. But the YES fill arrives via user_ws
            // late (700ms-2s): do not delete the state yet — states.retain cleans it
            // by time deadline (REJECTED_CLEANUP_SECS) if YES never arrives.
            if !state.have_yes && !state.have_no {
                return intents;
            }

            // YES sold → cancel the resting NO only if there is an active resting (retry_sent).
            if state.yes_sold && state.retry_sent && !state.cancel_sent {
                intents.push(OrderIntent {
                    market_cid: cid.clone(),
                    token_id: state.no_token.clone(),
                    side: OrderSide::Cancel,
                    price: Decimal::ZERO,
                    size: Decimal::ZERO,
                    clear_at: state.clear_at,
                    signal: Decimal::ZERO,
                    neg_risk: book.neg_risk,
                    limit_price: None,
                    second_leg: None,
                    strategy: StrategyId::Arb,
                    resting: false,
                    real_clear_at: None,
                });
                state.cancel_sent = true;
                tracing::info!(cid = %cid, "ARB: canceling resting NO after selling the orphan YES");
                return intents;
            }
            if state.sell_confirmed {
                return intents;
            } // wait for expiry to clean up

            // If the sell FOK was rejected (no fill after the time
            // deadline), reset to retry — before it was N coalesced ticks.
            if state.sell_sent && !state.sell_confirmed {
                if let Some(sent_at) = state.sell_sent_at {
                    if sent_at.elapsed().as_secs() >= self.cfg.orphan_sell_retry_secs {
                        state.sell_sent = false;
                        state.sell_sent_at = None;
                        tracing::debug!(cid = %cid, "ARB: sell FOK rejected — reset to retry");
                    }
                }
            }

            // Comparison by QUANTITY, not by boolean flag. A PARTIAL fill
            // of the NO retry (e.g. YES=10, NO=3) set `have_no=true` without
            // covering the whole orphan — with `!state.have_no` neither of the two
            // management blocks (this one and the symmetric one below) ran, leaving the
            // remainder with no trailing stop, no stop-loss and no oracle bail.
            if state.have_yes && state.no_size < state.yes_size && !state.sell_sent {
                // Dust write-off BEFORE emitting. The write-off in
                // on_fill (Sell branch) only acted after a PARTIAL sell fill —
                // but if the orphan is born below DUST_SHARES the sell is never
                // sent (`submit_sell` aborts with "shares after truncating to 2dp is 0"), there is
                // no fill to process, and `orphan_sell_retry_secs` re-arms the attempt in an
                // infinite loop (seen in a live run: 29 errors for the same cid).
                if state.yes_size < DUST_SHARES {
                    tracing::info!(cid = %cid, dust = %state.yes_size,
                        "ARB: write-off of unsellable YES dust — no sell attempt");
                    state.yes_size = Decimal::ZERO;
                    state.sell_confirmed = true;
                    return intents;
                }

                let orphan_since = *state
                    .orphan_since
                    .get_or_insert_with(crate::time::Instant::now);

                let yes_is_up = state.yes_token == book.yes.token_id;
                let yes_bid = if yes_is_up {
                    book.yes.best_bid().unwrap_or(dec!(0.01))
                } else {
                    book.no.best_bid().unwrap_or(dec!(0.01))
                };
                state.last_bid = yes_bid;

                // Update the maximum for the trailing stop.
                if yes_bid > state.peak_bid {
                    state.peak_bid = yes_bid;
                    state.trail_ticks = 0;
                }

                // Orphan confirmed by TIME (the NO fill arrives async
                // within ≤2s); from here on retry and all the bails act.
                if orphan_since.elapsed().as_secs() >= self.cfg.orphan_confirm_secs {
                    // 1. Retry NO as resting at the CURRENT ask: the entry
                    //    price goes stale if the market moved and would never fill.
                    //    Cap: the complete pair must remain profitable — the resting is
                    //    maker (0% fee), so yes_entry + no_retry ≤ 1 − min_edge is enough.
                    //    If the ask exceeds the cap, do not repost (re-evaluated every tick) and
                    //    let the bails below act.
                    if !state.retry_sent {
                        let no_ask_actual = if yes_is_up {
                            book.no.best_ask()
                        } else {
                            book.yes.best_ask()
                        };
                        let retry_cap = dec!(1) - state.yes_entry - self.cfg.min_edge_after_fee;
                        match no_ask_actual {
                            Some(ask) if ask <= retry_cap => {
                                intents.push(OrderIntent {
                                    market_cid: cid.clone(),
                                    token_id: state.no_token.clone(),
                                    side: OrderSide::Buy,
                                    price: ask,
                                    size: state.yes_size,
                                    clear_at: state.clear_at,
                                    signal: ask,
                                    neg_risk: book.neg_risk,
                                    limit_price: Some(ask),
                                    second_leg: None,
                                    strategy: StrategyId::Arb,
                                    resting: true,
                                    real_clear_at: None,
                                });
                                state.retry_sent = true;
                                tracing::info!(
                                    cid = %cid, no_ask = %ask, cap = %retry_cap,
                                    "ARB: retrying NO leg as resting at the current ask"
                                );
                            }
                            _ => tracing::debug!(
                                cid = %cid, ask = ?no_ask_actual, cap = %retry_cap,
                                "ARB: NO retry skipped — ask over the cap or absent"
                            ),
                        }
                    }

                    // 2. Trailing stop for the orphan YES.
                    let trail_level = state.peak_bid * dec!(0.95);
                    if yes_bid < trail_level {
                        state.trail_ticks += 1;
                    } else {
                        state.trail_ticks = 0;
                    }
                    // 2 consecutive ticks below the level to filter market noise.
                    // This block only runs under
                    // `have_yes && !have_no` (orphan YES, uncovered), never with
                    // both legs bought. The condition `yes_bid > yes_entry` is
                    // intentional to keep trailing-stop and stop-loss orthogonal:
                    // the trailing ONLY takes profits (bid rose over the entry and
                    // then pulled back ≥5% from the peak) — the exit at a loss
                    // (bid below the entry) is covered by the hard stop-loss
                    // below, not the trailing. Without this guard both gates could
                    // fire on the same drop and overwrite the reported reason.
                    let trailing_stop = state.trail_ticks >= 2 && yes_bid > state.yes_entry;

                    // Hard stop-loss — the trailing only takes profits; without this the
                    // losing orphan had no exit until the oracle/time bail.
                    let stop_loss = self.cfg.orphan_stop_loss_pct > Decimal::ZERO
                        && yes_bid < state.yes_entry * (dec!(1) - self.cfg.orphan_stop_loss_pct);

                    // Oracle and time as additional bail (independent of the trailing).
                    let live = vol.last_price;
                    let reference = book.reference_price.unwrap_or(Decimal::ZERO);
                    let cushion = if reference > Decimal::ZERO && live > Decimal::ZERO {
                        if yes_is_up {
                            (live - reference) / reference
                        } else {
                            (reference - live) / reference
                        }
                    } else {
                        Decimal::ZERO
                    };
                    let force_time = remaining < ORPHAN_BAIL_SECS;
                    let oracle_bad = cushion < dec!(-0.02);

                    if trailing_stop || stop_loss || force_time || oracle_bad {
                        let reason = if trailing_stop {
                            "trailing stop"
                        } else if stop_loss {
                            "stop-loss"
                        } else if force_time {
                            "time"
                        } else {
                            "oracle against"
                        };
                        tracing::info!(
                            cid = %cid, yes_bid = %yes_bid, peak = %state.peak_bid,
                            yes_entry = %state.yes_entry, cushion = %cushion, remaining, reason,
                            "ARB: selling orphan YES leg"
                        );
                        metrics::counter!("hrf_arb_orphan_total", "leg" => "yes").increment(1);
                        intents.push(OrderIntent {
                            market_cid: cid.clone(),
                            token_id: state.yes_token.clone(),
                            side: OrderSide::Sell {
                                size: Some(state.yes_size),
                            },
                            price: yes_bid,
                            size: state.yes_size,
                            clear_at: state.clear_at,
                            signal: cushion,
                            neg_risk: book.neg_risk,
                            limit_price: None,
                            second_leg: None,
                            strategy: StrategyId::Arb,
                            resting: false,
                            real_clear_at: None,
                        });
                        state.sell_sent = true;
                        state.sell_sent_at = Some(crate::time::Instant::now());
                    }
                }
            }

            // ── NO-only leg management (NO filled, YES did not) ─────────────
            // Same QUANTITY criterion as the symmetric block above.
            if state.have_no && state.yes_size < state.no_size && !state.sell_sent {
                // Same pre-emission dust write-off as on the YES leg.
                if state.no_size < DUST_SHARES {
                    tracing::info!(cid = %cid, dust = %state.no_size,
                        "ARB: write-off of unsellable NO dust — no sell attempt");
                    state.no_size = Decimal::ZERO;
                    state.sell_confirmed = true;
                    return intents;
                }

                let orphan_since = *state
                    .orphan_since
                    .get_or_insert_with(crate::time::Instant::now);

                let no_is_up = state.no_token == book.no.token_id;
                let no_bid = if no_is_up {
                    book.no.best_bid().unwrap_or(dec!(0.01))
                } else {
                    book.yes.best_bid().unwrap_or(dec!(0.01))
                };
                state.last_bid = no_bid;

                if no_bid > state.peak_bid {
                    state.peak_bid = no_bid;
                    state.trail_ticks = 0;
                }

                if orphan_since.elapsed().as_secs() >= self.cfg.orphan_confirm_secs
                    && state.no_size > Decimal::ZERO
                {
                    let trail_level = state.peak_bid * dec!(0.95);
                    if no_bid < trail_level {
                        state.trail_ticks += 1;
                    } else {
                        state.trail_ticks = 0;
                    }
                    let trailing_stop = state.trail_ticks >= 2 && no_bid > state.no_price;
                    // Same hard stop-loss as the YES leg (no_price ≈ entry + slippage).
                    let stop_loss = self.cfg.orphan_stop_loss_pct > Decimal::ZERO
                        && no_bid < state.no_price * (dec!(1) - self.cfg.orphan_stop_loss_pct);
                    let force_time = remaining < ORPHAN_BAIL_SECS;

                    let live = vol.last_price;
                    let reference = book.reference_price.unwrap_or(Decimal::ZERO);
                    // The orphan NO wins if the underlying does NOT rise (no_is_up=true) or rises (no_is_up=false).
                    // cushion < 0: oracle moves against the NO token.
                    let cushion = if reference > Decimal::ZERO && live > Decimal::ZERO {
                        if no_is_up {
                            (reference - live) / reference
                        } else {
                            (live - reference) / reference
                        }
                    } else {
                        Decimal::ZERO
                    };
                    let oracle_bad = cushion < dec!(-0.02);

                    if trailing_stop || stop_loss || force_time || oracle_bad {
                        let reason = if trailing_stop {
                            "trailing stop"
                        } else if stop_loss {
                            "stop-loss"
                        } else if force_time {
                            "time"
                        } else {
                            "oracle against"
                        };
                        tracing::info!(
                            cid = %cid, no_bid = %no_bid, peak = %state.peak_bid,
                            cushion = %cushion, remaining, reason,
                            "ARB: selling orphan NO leg"
                        );
                        metrics::counter!("hrf_arb_orphan_total", "leg" => "no").increment(1);
                        intents.push(OrderIntent {
                            market_cid: cid.clone(),
                            token_id: state.no_token.clone(),
                            side: OrderSide::Sell {
                                size: Some(state.no_size),
                            },
                            price: no_bid,
                            size: state.no_size,
                            clear_at: state.clear_at,
                            signal: no_bid - state.no_price,
                            neg_risk: book.neg_risk,
                            limit_price: None,
                            second_leg: None,
                            strategy: StrategyId::Arb,
                            resting: false,
                            real_clear_at: None,
                        });
                        state.sell_sent = true;
                        state.sell_sent_at = Some(crate::time::Instant::now());
                    }
                }
            }
            return intents;
        }

        // ── Entry signal (bracket with no previous state) ──────────────────

        // Do not enter near expiry — in the live pilot
        // all 5 entries were on collapsing books (completeness 0/5):
        // the parity "edge" was the collapse of the losing side itself.
        if remaining < self.cfg.min_remaining_secs {
            metrics::counter!("hrf_arb_skip_remaining_total").increment(1);
            return intents;
        }

        let yes_ask = match book.yes.best_ask() {
            Some(a) => a,
            None => return intents,
        };
        let no_ask = match book.no.best_ask() {
            Some(a) => a,
            None => return intents,
        };

        // EXACT fee of the pair: both legs are taker and Polymarket charges
        // `rate × p × (1−p)` per leg (verified against the Gamma
        // `feeSchedule` and the official fee table). The earlier triangular
        // model (`rate × min(p, 1−p)`) OVERESTIMATES the fee ~2x right
        // in the mid zone where ARB operates; the paper sensor was the one
        // miscalibrated, not the strategy.
        // `rate` uses the market's real rate
        // (`book.parabolic_fee_rate`, from Gamma's `feeSchedule.rate`)
        // when available — there are ≥2 coexisting regimes (0.07 crypto,
        // 0.04 actions) and the fixed constant overestimates the fee ~75% in the
        // second family. Falls back to `TAKER_FEE_RATE_PARABOLIC` (0.07, conservative
        // fallback for crypto, where ARB mostly operates) if Gamma
        // did not bring a `feeSchedule` for this market.
        // The earlier flat threshold (fee_threshold=0.01) let trades in
        // with guaranteed loss into the mid zone; fee_threshold remains as an
        // optional additional cushion over the real fee.
        let parabolic_rate =
            book.parabolic_fee_rate_or(crate::fair_value::TAKER_FEE_RATE_PARABOLIC);
        let fee_pair =
            crate::fair_value::fee_taker_parabolic_with_rate(parabolic_rate, yes_ask, Decimal::ONE)
                + crate::fair_value::fee_taker_parabolic_with_rate(
                    parabolic_rate,
                    no_ask,
                    Decimal::ONE,
                );
        let edge = dec!(1) - yes_ask - no_ask - fee_pair - self.cfg.fee_threshold;
        // The FOKs are signed at the WS book price, but the CLOB sees them
        // ~100-300ms later. leg_slippage widens the signed limit of BOTH
        // legs (before only NO — YES died on a 1-tick move) and
        // enough edge is required to cover the extra cost of both.
        if edge < self.cfg.min_edge_after_fee + self.cfg.leg_slippage * dec!(2) {
            metrics::counter!("hrf_arb_skip_edge_total").increment(1);
            return intents;
        }

        // An implausibly large edge is not arbitrage — it is the dead part
        // of the book (live: "edge" 0.39 after a WS reset;
        // the real ask had disappeared → FOK with no counterparty → orphan).
        // Real edges in the updown universe are 1-3 cents.
        if self.cfg.max_plausible_edge > Decimal::ZERO && edge > self.cfg.max_plausible_edge {
            metrics::counter!("hrf_arb_skip_edge_implausible_total").increment(1);
            tracing::warn!(
                cid = %cid, yes_ask = %yes_ask, no_ask = %no_ask, edge = %edge,
                "ARB: skip — implausible edge (book suspected to be partly dead)"
            );
            return intents;
        }

        // Spread filter: wide spread on any leg = illiquid market.
        // If we end up orphaned, selling at the bid with spread > max_bid_ask_spread can wipe out the profit.
        if self.cfg.max_bid_ask_spread > Decimal::ZERO {
            let yes_bid = book.yes.best_bid().unwrap_or(Decimal::ZERO);
            let no_bid = book.no.best_bid().unwrap_or(Decimal::ZERO);
            let yes_spread = yes_ask - yes_bid;
            let no_spread = no_ask - no_bid;
            if yes_spread > self.cfg.max_bid_ask_spread || no_spread > self.cfg.max_bid_ask_spread {
                metrics::counter!("hrf_arb_skip_spread_total").increment(1);
                tracing::debug!(
                    cid = %cid, yes_spread = %yes_spread, no_spread = %no_spread,
                    limit = %self.cfg.max_bid_ask_spread,
                    "ARB: skip — wide spread"
                );
                return intents;
            }
        }

        let cost_per_share = (yes_ask + no_ask).max(dec!(0.01));
        // Both limits apply when > 0; 0 = no cap on that dimension.
        // max_size_shares is the YES+NO total; each leg gets half
        // (divisor of `size_from_caps`, the same across the 7 strategies).
        // depth = size at the best ask on both legs (WS aggregates makers at the same price).
        // FOK with limit_price = best_ask only fills up to this depth; if size > depth → fails.
        // We only take depth/factor — require spare liquidity on both
        // sides so that a thin quote that disappears before the POST does not
        // leave us with an orphan leg (live completeness was 0/5).
        let factor = self.cfg.depth_safety_factor.max(Decimal::ONE);
        let depth = book
            .yes
            .asks
            .best_size()
            .unwrap_or(Decimal::ZERO)
            .min(book.no.asks.best_size().unwrap_or(Decimal::ZERO))
            / factor;
        // depth is a hard cap in ALL branches (unlike the other
        // strategies, where the fallback only applies without any asset cap) —
        // hence `.min(depth)` is applied after the helper rather than within.
        let size = super::size_from_caps(
            self.cfg.limits.max_order_usdc,
            self.cfg.max_size_shares,
            cost_per_share,
            Some(dec!(2)),
            Some(depth),
        )
        .min(depth);
        if size <= Decimal::ZERO {
            return intents;
        }

        // Polymarket minimum: $1 USDC per leg for marketable orders.
        // The SDK truncates size to whole before sending; we check here to avoid
        // the HTTP round-trip that would end in 400 "invalid amount".
        let size_int = size.trunc_with_scale(0);
        if size_int < Decimal::ONE || size_int * yes_ask < dec!(1) || size_int * no_ask < dec!(1) {
            tracing::debug!(
                cid = %cid, yes_ask = %yes_ask, no_ask = %no_ask, size = %size_int,
                "ARB: skip — amount per leg < $1 (Polymarket minimum)"
            );
            metrics::counter!("hrf_arb_skip_min_amount_total").increment(1);
            return intents;
        }

        // Signs both legs with slippage: tolerates the ask rising up to leg_slippage
        // between the WS read and the POST reaching the CLOB.
        let no_limit = no_ask + self.cfg.leg_slippage;
        let yes_limit = yes_ask + self.cfg.leg_slippage;
        let no_leg = Box::new(OrderIntent {
            market_cid: cid.clone(),
            token_id: book.no.token_id.clone(),
            side: OrderSide::Buy,
            price: no_limit,
            size,
            clear_at: book.clear_at,
            signal: edge,
            neg_risk: book.neg_risk,
            limit_price: Some(no_limit),
            second_leg: None,
            strategy: StrategyId::Arb,
            resting: false,
            real_clear_at: None,
        });

        let yes_leg = OrderIntent {
            market_cid: cid.clone(),
            token_id: book.yes.token_id.clone(),
            side: OrderSide::Buy,
            price: yes_limit,
            size,
            clear_at: book.clear_at,
            signal: edge,
            neg_risk: book.neg_risk,
            limit_price: Some(yes_limit),
            second_leg: Some(no_leg),
            strategy: StrategyId::Arb,
            resting: false,
            real_clear_at: None,
        };

        self.states.insert(
            cid.clone(),
            BracketState {
                clear_at: book.clear_at,
                yes_token: book.yes.token_id.clone(),
                no_token: book.no.token_id.clone(),
                no_price: no_limit,
                no_size: Decimal::ZERO,
                yes_entry: Decimal::ZERO,
                yes_size: Decimal::ZERO,
                have_yes: false,
                have_no: false,
                created_at: crate::time::Instant::now(),
                orphan_since: None,
                retry_sent: false,
                sell_sent: false,
                yes_sold: false,
                sell_confirmed: false,
                cancel_sent: false,
                sell_sent_at: None,
                peak_bid: Decimal::ZERO,
                trail_ticks: 0,
                last_bid: Decimal::ZERO,
                neg_risk: book.neg_risk,
            },
        );

        metrics::counter!("hrf_arb_signals_total").increment(1);
        tracing::info!(
            cid = %cid, yes_ask = %yes_ask, no_ask = %no_ask, edge = %edge,
            "ARB: edge detected"
        );

        intents.push(yes_leg);
        intents
    }

    /// Time watchdog. All orphan management runs within
    /// `on_book`; with a cid's feed quiet, the leg was left without a sell
    /// retry or time bail until expiry. Sells are market
    /// FOK (`OrderKind::Sell` walks the CLOB's real book — `price` is only
    /// informative), so what acts here is what needs no live price:
    /// cleanup on expiry, cancel of the resting NO after selling YES,
    /// retry of rejected sells and forced bail in the final window.
    /// Trailing stop / stop-loss / oracle bail remain in `on_book`.
    fn on_timer(&mut self) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();

        // Same cleanup as on_book's retain (expired + doubly
        // rejected FOK) — with a dead feed nobody else runs it.
        self.states.retain(|_, s| {
            if s.clear_at <= now {
                return false;
            }
            !(s.created_at.elapsed().as_secs() >= REJECTED_CLEANUP_SECS
                && !s.have_yes
                && !s.have_no)
        });

        let mut intents = vec![];
        for (cid, state) in self.states.iter_mut() {
            if state.sell_confirmed || (!state.have_yes && !state.have_no) {
                continue;
            }

            // Cancel of the resting NO after selling YES — needs no book.
            if state.yes_sold && state.retry_sent && !state.cancel_sent {
                intents.push(OrderIntent {
                    market_cid: cid.clone(),
                    token_id: state.no_token.clone(),
                    side: OrderSide::Cancel,
                    price: Decimal::ZERO,
                    size: Decimal::ZERO,
                    clear_at: state.clear_at,
                    signal: Decimal::ZERO,
                    neg_risk: state.neg_risk,
                    limit_price: None,
                    second_leg: None,
                    strategy: StrategyId::Arb,
                    resting: false,
                    real_clear_at: None,
                });
                state.cancel_sent = true;
                tracing::info!(cid = %cid, "ARB watchdog: canceling resting NO with a quiet feed");
                continue;
            }

            // Orphan by quantity (same criterion as on_book).
            let (token, size, leg) = if state.have_yes && state.no_size < state.yes_size {
                (state.yes_token.clone(), state.yes_size, "yes")
            } else if state.have_no && state.yes_size < state.no_size {
                (state.no_token.clone(), state.no_size, "no")
            } else {
                continue; // complete pair or no fills yet
            };

            // The confirmation window also runs without books.
            let orphan_since = *state
                .orphan_since
                .get_or_insert_with(crate::time::Instant::now);
            if orphan_since.elapsed().as_secs() < self.cfg.orphan_confirm_secs {
                continue;
            }

            // Rejected sell retry: in on_book the reset only re-arms for
            // the next tick — with a dead feed there is no next tick, so the
            // watchdog re-emits directly (the decision to sell was already made).
            let mut is_retry = false;
            if state.sell_sent {
                let deadline = match state.sell_sent_at {
                    Some(t) => t.elapsed().as_secs() >= self.cfg.orphan_sell_retry_secs,
                    None => true, // flag without timestamp: unknown age → retry
                };
                if !deadline {
                    continue;
                }
                state.sell_sent = false;
                state.sell_sent_at = None;
                is_retry = true;
            }

            let remaining = state.clear_at - now;
            if !is_retry && remaining >= ORPHAN_BAIL_SECS {
                continue; // without a book only the time bail justifies selling
            }

            // Same dust write-off as on_book (impossible sell < 2dp).
            if size < DUST_SHARES {
                tracing::info!(cid = %cid, dust = %size, leg,
                    "ARB watchdog: write-off of unsellable dust");
                if leg == "yes" {
                    state.yes_size = Decimal::ZERO;
                } else {
                    state.no_size = Decimal::ZERO;
                }
                state.sell_confirmed = true;
                continue;
            }

            let reason = if is_retry {
                "sell retry"
            } else {
                "time"
            };
            tracing::info!(
                cid = %cid, leg, last_bid = %state.last_bid, remaining, reason,
                "ARB watchdog: selling orphan leg without a fresh book"
            );
            metrics::counter!("hrf_arb_orphan_total", "leg" => leg).increment(1);
            intents.push(OrderIntent {
                market_cid: cid.clone(),
                token_id: token,
                side: OrderSide::Sell { size: Some(size) },
                price: state.last_bid,
                size,
                clear_at: state.clear_at,
                signal: Decimal::ZERO,
                neg_risk: state.neg_risk,
                limit_price: None,
                second_leg: None,
                strategy: StrategyId::Arb,
                resting: false,
                real_clear_at: None,
            });
            state.sell_sent = true;
            state.sell_sent_at = Some(crate::time::Instant::now());
        }
        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::Arb {
            return;
        }

        // Buy fill with no state in `states` (restart mid-pair, or state already
        // purged by retain): before it was ignored and the position was left unmanaged until
        // the expiry's orphan checker. Recreate it as a confirmed orphan so that
        // the bails (stop-loss/trailing/oracle/time) manage it. We do not know which is
        // the opposite token → no retry or cancel possible (retry_sent/cancel_sent=true).
        if !self.states.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(
                    cid = %fill.market_cid, token = %fill.token_id, price = %fill.price,
                    "ARB: Buy fill without state — recreating as an orphan leg"
                );
                metrics::counter!("hrf_arb_orphan_total", "leg" => "recreated").increment(1);
                self.states.insert(
                    fill.market_cid.clone(),
                    BracketState {
                        clear_at: fill.clear_at,
                        yes_token: fill.token_id.clone(),
                        no_token: String::new(),
                        no_price: Decimal::ZERO,
                        no_size: Decimal::ZERO,
                        yes_entry: fill.price,
                        yes_size: fill.size,
                        have_yes: true,
                        have_no: false,
                        created_at: crate::time::Instant::now(),
                        // Confirmed orphan — bails enabled from now on: move the
                        // instant back to skip the orphan_confirm_secs wait.
                        orphan_since: crate::time::Instant::now()
                            .checked_sub(std::time::Duration::from_secs(3600))
                            .or(Some(crate::time::Instant::now())),
                        retry_sent: true,
                        sell_sent: false,
                        yes_sold: false,
                        sell_confirmed: false,
                        cancel_sent: true,
                        sell_sent_at: None,
                        peak_bid: Decimal::ZERO,
                        trail_ticks: 0,
                        last_bid: fill.price,
                        neg_risk: fill.neg_risk,
                    },
                );
            }
            return;
        }

        let remove = if let Some(state) = self.states.get_mut(&fill.market_cid) {
            match fill.side {
                FillSide::Buy => {
                    if fill.token_id == state.yes_token {
                        // Size-weighted average, not the last
                        // partial — yes_entry is the reference for the hard stop-loss,
                        // the trailing stop and the cap of the NO retry; overwriting it
                        // with the last fill miscalibrates those three gates when the
                        // taker fills across several trades at different prices.
                        let prev_size = state.yes_size;
                        let new_size = prev_size + fill.size;
                        state.yes_entry = if new_size > Decimal::ZERO {
                            (state.yes_entry * prev_size + fill.price * fill.size) / new_size
                        } else {
                            fill.price
                        };
                        state.yes_size = new_size;
                        state.have_yes = true;
                        // YES arrived: if NO was already there, the pair is complete.
                        if state.have_no {
                            state.orphan_since = None;
                        }
                    } else {
                        // NO fill (original entry or resting retry).
                        state.have_no = true;
                        state.no_size += fill.size; // add up: possible partial GTD fill
                        state.sell_sent = false;
                        state.sell_sent_at = None;
                        state.orphan_since = None; // pair complete (or NO-only starts being counted again)
                                                   // If YES was already sold, prevent on_book from emitting Cancel for an order
                                                   // that already filled — it would make the executor release exposure twice.
                        if state.yes_sold {
                            state.cancel_sent = true;
                        }
                    }
                    false
                }
                FillSide::Sell => {
                    // Partial sell (FAK capped at the bid's depth): reduce the size and
                    // re-arm the sell of the rest — before, sell_confirmed froze the
                    // state and the remainder was left unmanaged until settlement.
                    // A remainder < DUST_SHARES is not sellable (the router
                    // truncates shares to 2dp → "shares after truncating to 2dp is 0" in an endless
                    // retry loop in live). Write-off: consider it closed.
                    if fill.token_id == state.yes_token {
                        state.yes_size = (state.yes_size - fill.size).max(Decimal::ZERO);
                        // Cancel the resting NO even if the sell is partial: we already decided to exit.
                        state.yes_sold = true;
                        if state.yes_size < DUST_SHARES && state.yes_size > Decimal::ZERO {
                            tracing::info!(cid = %fill.market_cid, dust = %state.yes_size,
                                "ARB: write-off of unsellable YES dust");
                            state.yes_size = Decimal::ZERO;
                        }
                        state.sell_confirmed = state.yes_size == Decimal::ZERO;
                    } else {
                        state.no_size = (state.no_size - fill.size).max(Decimal::ZERO);
                        if state.no_size < DUST_SHARES && state.no_size > Decimal::ZERO {
                            tracing::info!(cid = %fill.market_cid, dust = %state.no_size,
                                "ARB: write-off of unsellable NO dust");
                            state.no_size = Decimal::ZERO;
                        }
                        state.sell_confirmed = state.no_size == Decimal::ZERO;
                    }
                    state.sell_sent_at = None;
                    if !state.sell_confirmed {
                        state.sell_sent = false;
                    }
                    false
                }
                // Never delete on Rejected: the YES fill may arrive via user_ws up to ~2s
                // later (async). states.retain cleans up by time deadline if it never arrives.
                // Same for GateRejected — the pair's state comes from fills,
                // not from emission marks; latches with a deadline self-heal.
                FillSide::Rejected | FillSide::GateRejected => false,
                // No live-order accounting — nothing to clean up.
                FillSide::Cancelled => false,
            }
        } else {
            false
        };
        if remove {
            self.states.remove(&fill.market_cid);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};

    fn make_book(yes_ask: &str, no_ask: &str) -> BookSnapshot {
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        BookSnapshot {
            market_condition_id: "arb-cid".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids("0.40"),
                asks: asks(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids("0.10"),
                asks: asks(no_ask),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + 300,
            neg_risk: false,
            slug: "btc-updown-arb".into(),
            reference_price: Some(dec!(100000)),
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        }
    }

    fn make_vol() -> VolSnapshot {
        VolSnapshot {
            asset: "btc".into(),
            sigma_short: None,
            sigma_long: None,
            last_price: dec!(100000),
            updated_at: crate::time::Instant::now(),
            ..VolSnapshot::default()
        }
    }

    fn test_cfg() -> ArbConfig {
        ArbConfig {
            fee_threshold: dec!(0.02),
            min_edge_after_fee: dec!(0.005),
            max_size_shares: dec!(10),
            // Immediate orphan confirmation in tests (no sleeps)
            // and a distant sell retry so sells are not re-emitted in tick loops.
            orphan_confirm_secs: 0,
            orphan_sell_retry_secs: 60,
            ..ArbConfig::default()
        }
    }

    fn make_fill(cid: &str, token: &str, side: FillSide, price: &str, size: &str) -> Fill {
        Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: cid.into(),
            token_id: token.into(),
            side,
            price: price.parse().unwrap(),
            size: size.parse().unwrap(),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::Arb,
            clear_at: 0,
            outcome_label: None,
        }
    }

    #[test]
    fn arb_detects_edge() {
        let mut arb = Arb::new(test_cfg());
        // pair_fee = 2 × 0.07×0.45×0.55 = 0.03465.
        // edge = 1-0.45-0.45-0.03465-0.02(fee_threshold) = 0.04535 > 0.005
        let intents = arb.on_book(&make_book("0.45", "0.45"), &make_vol());
        assert_eq!(intents.len(), 1);
        assert!(intents[0].second_leg.is_some());
    }

    #[test]
    fn arb_exact_fee_blocks_negative_ev_in_mid_zone() {
        // Real pair fee at 0.48/0.48 with `0.07 × p × (1−p)` per leg
        // (fair_value::fee_taker_parabolic_with_rate — replaces the triangular
        // `rate × min(p,1−p)`, which overestimated ~2x in this zone):
        // 0.07×0.48×0.52 × 2 = 0.034944. Raw edge = 1 − 0.96 = 0.04 → net almost
        // zero (0.04 − 0.034944 − 0.005 cushion = 0.000056 < min_edge_after_fee)
        // — the EV≈0/− trade must be blocked.
        let cfg = ArbConfig {
            fee_threshold: dec!(0.005), // cushion
            min_edge_after_fee: dec!(0.005),
            max_size_shares: dec!(20), // 10/leg → cheap NO leg ≥ $1 (Polymarket minimum)
            ..ArbConfig::default()
        };
        let mut arb = Arb::new(cfg.clone());
        assert!(
            arb.on_book(&make_book("0.48", "0.48"), &make_vol())
                .is_empty(),
            "at mid prices the real fee must block the 4¢ raw edge"
        );
        // A 5¢ raw edge at the extremes DOES pass: lower fee at the extremes.
        // 0.85/0.10 → fee = 0.07×0.85×0.15 + 0.07×0.10×0.90 = 0.008925 + 0.0063
        // = 0.015225; edge = 0.05 − 0.015225 − 0.005 = 0.029775 ≥ min_edge_after_fee (0.005).
        let mut arb2 = Arb::new(cfg);
        assert_eq!(
            arb2.on_book(&make_book("0.85", "0.10"), &make_vol()).len(),
            1,
            "at the extremes a similar raw edge is EV+ and must enter"
        );
    }

    /// `book.parabolic_fee_rate` (read from Gamma's
    /// `feeSchedule.rate`) replaces the fallback constant when the
    /// market provides it — there are ≥2 coexisting regimes (`crypto_fees_v2`
    /// rate=0.07, `finance_prices_fees` rate=0.04). At 0.48/0.48 the fallback
    /// (0.07) blocks the trade (see earlier test); with an explicit rate=0.04 the
    /// same 4¢ raw edge IS EV+ and must pass.
    #[test]
    fn arb_uses_market_real_parabolic_fee_rate_if_available() {
        let cfg = ArbConfig {
            fee_threshold: dec!(0.005),
            min_edge_after_fee: dec!(0.005),
            max_size_shares: dec!(20),
            ..ArbConfig::default()
        };

        // Without parabolic_fee_rate (None) it falls back to 0.07 and blocks.
        let mut arb_fallback = Arb::new(cfg.clone());
        assert!(
            arb_fallback
                .on_book(&make_book("0.48", "0.48"), &make_vol())
                .is_empty(),
            "without parabolic_fee_rate it must fall back to 0.07 and block"
        );

        // With explicit rate=0.04: pair_fee = 2 × 0.04×0.48×0.52 = 0.019968.
        // edge = 0.04 − 0.019968 − 0.005(fee_threshold) = 0.015032 ≥ min_edge_after_fee (0.005).
        let mut book = make_book("0.48", "0.48");
        book.parabolic_fee_rate = Some(dec!(0.04));
        let mut arb_real_rate = Arb::new(cfg);
        assert_eq!(
            arb_real_rate.on_book(&book, &make_vol()).len(),
            1,
            "with parabolic_fee_rate=0.04 (finance_prices_fees) the same raw edge must pass"
        );
    }

    /// A partial sell that leaves < 0.01 shares is not sellable (the router
    /// truncates to 2dp) — write off the remainder instead of retrying forever.
    #[test]
    fn arb_write_off_dust_after_partial_sell() {
        let mut arb = Arb::new(test_cfg());
        // Buy with no previous state → recreated as an orphan YES leg.
        arb.on_fill(&make_fill(
            "arb-cid",
            "y",
            FillSide::Buy,
            "0.34",
            "10.588234",
        ));
        // Partial sell: the router truncated to 10.58 → 0.008234 remains (dust).
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Sell, "0.18", "10.58"));
        let st = arb.states.get("arb-cid").unwrap();
        assert_eq!(
            st.yes_size,
            Decimal::ZERO,
            "the dust must be written off"
        );
        assert!(
            st.sell_confirmed,
            "state closed — no sell retry loop"
        );
    }

    #[test]
    fn arb_skip_without_edge() {
        let mut arb = Arb::new(test_cfg());
        // pair_fee = 2 × 0.07×0.50×0.50 = 0.035.
        // yes_ask=0.50, no_ask=0.50 → edge = 1-0.50-0.50-0.035-0.02 = -0.055 < 0.005
        assert!(arb
            .on_book(&make_book("0.50", "0.50"), &make_vol())
            .is_empty());
    }

    #[test]
    fn arb_orphan_retry_not_small() {
        // pair_fee = 0.07×0.75×0.25 + 0.07×0.20×0.80 = 0.013125+0.0112 =
        // 0.024325. edge = 1-0.75-0.20-0.024325-0.02(fee_threshold) = 0.005675
        // ≥ min_edge_after_fee (0.005) → NO is small → must retry as resting
        let mut arb = Arb::new(test_cfg());
        let book = make_book("0.75", "0.20");

        // Tick 1: emits intent
        let intents = arb.on_book(&book, &make_vol());
        assert_eq!(intents.len(), 1);

        // YES fills
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.75", "10"));

        // With orphan_confirm_secs=0 the retry goes out on the first tick after the fill.
        let i = arb.on_book(&book, &make_vol());
        let retries: Vec<_> = i
            .iter()
            .filter(|i| i.resting && matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(
            retries.len(),
            1,
            "must retry NO as resting once the orphan is confirmed"
        );
        assert_eq!(retries[0].token_id, "n");
        assert_eq!(retries[0].price, dec!(0.20));
    }

    #[test]
    fn arb_orphan_waits_for_temporal_confirmation() {
        // With a high orphan_confirm_secs, NO must not act (retry/bail) yet —
        // the other leg's fill may arrive async via user_ws.
        let cfg = ArbConfig {
            orphan_confirm_secs: 3600,
            ..test_cfg()
        };
        let mut arb = Arb::new(cfg);
        let book = make_book("0.75", "0.20");
        let _ = arb.on_book(&book, &make_vol());
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.75", "10"));

        for _ in 0..10 {
            let i = arb.on_book(&book, &make_vol());
            assert!(
                i.is_empty(),
                "within the confirmation window there must be no actions"
            );
        }
    }

    #[test]
    fn arb_orphan_sells_yes_oracle_against() {
        // yes=0.45, no=0.45 → edge=0.08 ✓.
        // After the YES fill, oracle falls -5% → oracle_bad → must sell YES.
        let mut arb = Arb::new(test_cfg());
        let mut book = make_book("0.45", "0.45");
        book.reference_price = Some(dec!(100000));

        let _ = arb.on_book(&book, &make_vol());
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.45", "10"));

        let mut vol_bad = make_vol();
        vol_bad.last_price = dec!(95000); // -5% → cushion = -0.05 < -0.02

        // Advance to orphan_ticks=20 with the oracle against
        let mut sells = vec![];
        for _ in 0..20 {
            let i = arb.on_book(&book, &vol_bad);
            sells.extend(
                i.into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            1,
            "must sell YES when the oracle is against (at tick 20)"
        );
    }

    /// A DUST orphan (< DUST_SHARES) must not generate any sell attempt.
    /// `submit_sell` truncates shares to 2dp → 0 and aborts, no fill arrives to
    /// process, and `orphan_sell_retry_secs` re-armed the attempt in an endless loop
    /// (a live run: 29 errors "shares after truncating to 2dp is 0 (0.005554)"
    /// for the same cid). The earlier write-off only covered dust left
    /// after a PARTIAL sell fill, not dust that arrived from the start.
    #[test]
    fn arb_orphan_dust_does_not_attempt_sell() {
        let mut arb = Arb::new(test_cfg());
        let mut book = make_book("0.45", "0.45");
        book.reference_price = Some(dec!(100000));

        let _ = arb.on_book(&book, &make_vol());
        // Dust fill: 0.005554 shares < DUST_SHARES (0.01) — the real scenario from the log.
        arb.on_fill(&make_fill(
            "arb-cid",
            "y",
            FillSide::Buy,
            "0.45",
            "0.005554",
        ));

        let mut vol_bad = make_vol();
        vol_bad.last_price = dec!(95000); // oracle against → would trigger the sell

        let mut sells = vec![];
        for _ in 0..20 {
            let i = arb.on_book(&book, &vol_bad);
            sells.extend(
                i.into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert!(
            sells.is_empty(),
            "dust is not sellable: must not emit a Sell even once"
        );
    }

    #[test]
    fn arb_orphan_trailing_stop() {
        // Trailing stop: YES rises to a peak and then falls ≥5% for 2 ticks → sells.
        let mut arb = Arb::new(test_cfg());
        let mut book = make_book("0.45", "0.45");
        book.reference_price = Some(dec!(100000));

        let _ = arb.on_book(&book, &make_vol());
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.45", "10"));

        let mut book_up = book.clone();
        book_up.yes.bids = crate::types::HalfBook {
            levels: vec![crate::types::Level {
                price: dec!(0.70),
                size: dec!(50),
            }],
        };
        let mut book_drop = book.clone();
        book_drop.yes.bids = crate::types::HalfBook {
            levels: vec![crate::types::Level {
                price: dec!(0.62),
                size: dec!(50),
            }],
        };

        // Ticks 1-19: peak at 0.70, orphan < 20 → no trailing stop yet
        for _ in 0..19 {
            let _ = arb.on_book(&book_up, &make_vol());
        }

        // Tick 20: orphan_ticks=20, bid falls to 0.62 → trail_ticks=1 (does not sell yet)
        let i20 = arb.on_book(&book_drop, &make_vol());
        assert!(
            i20.iter()
                .filter(|i| matches!(i.side, OrderSide::Sell { .. }))
                .count()
                == 0,
            "trail_ticks=1, does not sell yet"
        );

        // Tick 21: trail_ticks=2 → trailing_stop fires
        let i21 = arb.on_book(&book_drop, &make_vol());
        let sells: Vec<_> = i21
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Sell { .. }))
            .collect();
        assert_eq!(
            sells.len(),
            1,
            "trailing stop must fire after 2 ticks below the level"
        );
    }

    #[test]
    fn arb_orphan_retry_repriced_to_current_ask() {
        // The market moved after the entry — the retry must go out at NO's CURRENT
        // ask (not the stale entry price) as long as it fits within the cap.
        let mut arb = Arb::new(test_cfg());
        let _ = arb.on_book(&make_book("0.75", "0.20"), &make_vol());
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.75", "10"));

        // NO ask rises from 0.20 to 0.24 — below the cap 1 − 0.75 − 0.005 = 0.245.
        let moved_book = make_book("0.75", "0.24");
        let i = arb.on_book(&moved_book, &make_vol());
        let retries: Vec<_> = i.iter().filter(|i| i.resting).collect();
        assert_eq!(retries.len(), 1);
        assert_eq!(
            retries[0].price,
            dec!(0.24),
            "retry at the current ask, not the entry 0.20"
        );
    }

    #[test]
    fn arb_orphan_retry_skipped_if_ask_above_cap() {
        // NO ask rose so much that the complete pair would no longer be profitable
        // (0.30 > cap 0.245) → do not repost; the bails manage the orphan.
        let mut arb = Arb::new(test_cfg());
        let _ = arb.on_book(&make_book("0.75", "0.20"), &make_vol());
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.75", "10"));

        let expensive_book = make_book("0.75", "0.30");
        for _ in 0..25 {
            let i = arb.on_book(&expensive_book, &make_vol());
            assert!(
                i.iter().all(|x| !x.resting),
                "must not repost over the cap"
            );
        }
    }

    #[test]
    fn arb_orphan_hard_stop_loss() {
        // Entry YES at 0.45, bid at 0.40 < 0.45 × 0.95 = 0.4275 → stop-loss.
        // Neutral oracle and no time pressure: the only trigger is the stop-loss
        // (the trailing requires bid > entry and here bid < entry).
        let cfg = ArbConfig {
            orphan_stop_loss_pct: dec!(0.05),
            ..test_cfg()
        };
        let mut arb = Arb::new(cfg);
        let book = make_book("0.45", "0.45"); // yes bid = 0.40 (make_book default)
        let _ = arb.on_book(&book, &make_vol());
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.45", "10"));

        let mut sells = vec![];
        for _ in 0..20 {
            let i = arb.on_book(&book, &make_vol());
            sells.extend(
                i.into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            1,
            "hard stop-loss must sell the losing orphan"
        );
        assert_eq!(sells[0].token_id, "y");
    }

    #[test]
    fn arb_fill_without_state_recreates_orphan() {
        // Buy fill with no previous state (restart mid-pair) — before it was
        // ignored; now it is recreated as an orphan and the bails manage it.
        let mut arb = Arb::new(test_cfg());
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.45", "10"));

        // Time bail: bracket about to expire.
        let mut book_expiry = make_book("0.45", "0.45");
        book_expiry.clear_at = crate::time::now().timestamp() + 10; // < ORPHAN_BAIL_SECS
        let i = arb.on_book(&book_expiry, &make_vol());
        let sells: Vec<_> = i
            .iter()
            .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
            .collect();
        assert_eq!(
            sells.len(),
            1,
            "the recreated orphan must be sold by the time bail"
        );
        assert_eq!(sells[0].token_id, "y");
        assert_eq!(sells[0].size, dec!(10));
        // And it must never emit a retry or Cancel (opposite token unknown).
        assert!(i
            .iter()
            .all(|x| !x.resting && !matches!(x.side, OrderSide::Cancel)));
    }

    #[test]
    fn arb_skip_spread_too_wide() {
        // max_bid_ask_spread=0.04; NO bid=0.10, NO ask=0.45 → spread=0.35 > 0.04 → skip.
        let cfg = ArbConfig {
            max_bid_ask_spread: dec!(0.04),
            ..test_cfg()
        };
        let mut arb = Arb::new(cfg);
        assert!(
            arb.on_book(&make_book("0.45", "0.45"), &make_vol())
                .is_empty(),
            "must skip entry when the spread of any leg exceeds the limit"
        );
    }

    #[test]
    fn arb_no_orphan_oracle_against() {
        // NO fills (YES does not). Oracle rises 4% → NO-of-UP loses → oracle_bad → sell.
        let mut arb = Arb::new(test_cfg());
        let mut book = make_book("0.45", "0.45");
        book.reference_price = Some(dec!(100000));
        let _ = arb.on_book(&book, &make_vol());
        arb.on_fill(&make_fill("arb-cid", "n", FillSide::Buy, "0.45", "5"));

        // BTC +4%: cushion_NO = (100k - 104k)/100k = -0.04 < -0.02 → oracle_bad
        let mut vol_bad = make_vol();
        vol_bad.last_price = dec!(104000);

        let mut sells = vec![];
        for _ in 0..20 {
            let i = arb.on_book(&book, &vol_bad);
            sells.extend(
                i.into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            1,
            "must sell the orphan NO when the oracle goes against it"
        );
        assert_eq!(sells[0].token_id, "n");
    }

    #[test]
    fn arb_no_orphan_sell_without_spurious_cancel() {
        // NO fills but YES does not (defensive scenario). When selling NO:
        //  - no Cancel must be emitted for NO (there is no active resting)
        //  - the state must freeze (sell_confirmed) without generating more intents
        let mut arb = Arb::new(test_cfg());
        let mut book = make_book("0.45", "0.45");
        book.reference_price = Some(dec!(100000));

        let _ = arb.on_book(&book, &make_vol());
        // Simulate that only NO filled
        arb.on_fill(&make_fill("arb-cid", "n", FillSide::Buy, "0.45", "5"));

        // Wait for orphan_ticks >= 20 with force_time to force the sell
        let mut book_expiry = book.clone();
        book_expiry.clear_at = crate::time::now().timestamp() + 10; // < ORPHAN_BAIL_SECS
        let mut sells = vec![];
        for _ in 0..20 {
            let i = arb.on_book(&book_expiry, &make_vol());
            sells.extend(
                i.into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            1,
            "must sell the orphan NO by time at tick 20"
        );
        assert_eq!(sells[0].token_id, "n");

        // Confirm the NO sell
        arb.on_fill(&make_fill("arb-cid", "n", FillSide::Sell, "0.45", "5"));

        // Following ticks: must emit no Cancel or new intents
        let i3 = arb.on_book(&book_expiry, &make_vol());
        let cancels: Vec<_> = i3
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Cancel))
            .collect();
        assert!(
            cancels.is_empty(),
            "a sold orphan NO must not emit a spurious Cancel"
        );
        assert!(
            i3.is_empty(),
            "state must be frozen after sell_confirmed"
        );
    }

    #[test]
    fn arb_orphan_management_runs_with_partial_no() {
        // A PARTIAL NO fill (3 of 10 YES shares) must not
        // disable management of the orphan YES. Before, the boolean guard
        // `!state.have_no` became `false` with ANY partial fill and
        // neither the YES-only block nor the NO-only one (`have_yes` stayed `true`)
        // ran again — the uncovered remainder was left with no trailing
        // stop, no stop-loss and no oracle bail. With the comparison by
        // QUANTITY (`no_size < yes_size`) the YES-only block must keep
        // managing the remainder.
        let cfg = ArbConfig {
            orphan_stop_loss_pct: dec!(0.05),
            ..test_cfg()
        };
        let mut arb = Arb::new(cfg);
        let book = make_book("0.45", "0.45"); // yes bid = 0.40 (make_book default)
        let _ = arb.on_book(&book, &make_vol());
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.45", "10"));
        // NO retry fills only partially (3 of 10).
        arb.on_fill(&make_fill("arb-cid", "n", FillSide::Buy, "0.45", "3"));

        let mut sells = vec![];
        for _ in 0..20 {
            let i = arb.on_book(&book, &make_vol());
            sells.extend(
                i.into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            1,
            "stop-loss must keep managing the YES remainder even if NO has a partial fill"
        );
        assert_eq!(sells[0].token_id, "y");
    }

    #[test]
    fn arb_yes_entry_averages_partial_fills() {
        // yes_entry must be the size-weighted average of
        // ALL partial fills, not just the last — it is the reference for the
        // hard stop-loss, the trailing stop and the NO retry cap.
        let mut arb = Arb::new(test_cfg());
        let book = make_book("0.45", "0.45");
        let _ = arb.on_book(&book, &make_vol());

        // Two partial fills at different prices: 6@0.40 + 4@0.50 → average 0.44.
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.40", "6"));
        arb.on_fill(&make_fill("arb-cid", "y", FillSide::Buy, "0.50", "4"));

        let state = arb
            .states
            .get("arb-cid")
            .expect("state must exist after the fills");
        assert_eq!(
            state.yes_entry,
            dec!(0.44),
            "yes_entry must be the weighted average, not the last fill (0.50)"
        );
        assert_eq!(state.yes_size, dec!(10));
    }

    // ── Time watchdog (on_timer) ─────────────────────────────────────────────

    /// Orphan recreated by a fill (confirmed-orphan path)
    /// with `clear_at` `secs` from now.
    fn arb_with_orphan(secs_to_close: i64) -> Arb {
        let mut arb = Arb::new(test_cfg());
        let mut fill = make_fill("cid-w", "tok-y", FillSide::Buy, "0.45", "5");
        fill.clear_at = crate::time::now().timestamp() + secs_to_close;
        arb.on_fill(&fill);
        arb
    }

    #[test]
    fn watchdog_sells_orphan_in_final_window_without_books() {
        // With a quiet feed the orphan reached expiry unmanaged.
        // The time bail needs no live price (the sell is a market FOK).
        let mut arb = arb_with_orphan(10); // < ORPHAN_BAIL_SECS (30)
        let intents = arb.on_timer();
        assert_eq!(intents.len(), 1, "the watchdog must sell the orphan");
        assert!(matches!(intents[0].side, OrderSide::Sell { .. }));
        assert_eq!(intents[0].token_id, "tok-y");
        assert_eq!(intents[0].price, dec!(0.45), "reference = last_bid (fill)");
    }

    #[test]
    fn watchdog_does_not_sell_orphan_far_from_close() {
        // Far from the close and with no rejected sell there is no time action: the
        // price bails (trailing/stop/oracle) require a live book.
        let mut arb = arb_with_orphan(300);
        assert!(arb.on_timer().is_empty());
    }

    #[test]
    fn watchdog_retries_rejected_sell_without_waiting_for_book() {
        // The sell FOK was rejected; in on_book the reset only re-arms for the
        // next tick. With a dead feed the watchdog re-emits directly after
        // `orphan_sell_retry_secs` — the decision to sell had already been made.
        let mut arb = arb_with_orphan(300);
        {
            let s = arb.states.get_mut("cid-w").unwrap();
            s.sell_sent = true;
            s.sell_sent_at =
                Some(crate::time::Instant::now() - std::time::Duration::from_secs(120));
        }
        let intents = arb.on_timer();
        assert_eq!(intents.len(), 1, "a rejected sell must be retried");
        assert!(matches!(intents[0].side, OrderSide::Sell { .. }));
        // Sell in flight within the deadline → do not duplicate.
        let intents = arb.on_timer();
        assert!(intents.is_empty(), "do not retry before the deadline");
    }

    #[test]
    fn watchdog_cancels_resting_not_after_yes_sold() {
        let mut arb = arb_with_orphan(300);
        {
            let s = arb.states.get_mut("cid-w").unwrap();
            s.no_token = "tok-n".into();
            s.yes_sold = true;
            s.retry_sent = true;
            s.cancel_sent = false;
        }
        let intents = arb.on_timer();
        assert_eq!(intents.len(), 1);
        assert!(matches!(intents[0].side, OrderSide::Cancel));
        assert_eq!(intents[0].token_id, "tok-n");
    }

    #[test]
    fn watchdog_cleans_expired_state_and_writes_off_dust() {
        // Expired → retain removes it even if no books arrive.
        let mut arb = arb_with_orphan(-5);
        assert!(arb.on_timer().is_empty());
        assert!(arb.states.is_empty(), "expired state must be purged");

        // Dust < 2dp → write-off without attempting a sell (impossible sell).
        let mut arb = arb_with_orphan(10);
        {
            let s = arb.states.get_mut("cid-w").unwrap();
            s.yes_size = dec!(0.005);
        }
        assert!(arb.on_timer().is_empty());
        let s = arb.states.get("cid-w").unwrap();
        assert_eq!(s.yes_size, Decimal::ZERO);
        assert!(s.sell_confirmed);
    }
}
