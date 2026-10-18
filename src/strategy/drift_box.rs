//! DriftBox strategy — biased entry + deferred parity box.
//!
//! Phase 1: buys the leg favored by the oracle (FOK taker) when
//!         `|fair_value_yes − 0.5| > min_confidence`.
//!
//! Phase 2: waits for the opposite leg to get cheaper. If
//!         `cost_a + opp_ask ≤ 1 − fee − min_box_profit`, completes the box
//!         (YES+NO) → risk-free profit guaranteed at settlement.
//!
//! Contingency: if the oracle reverts against us for ≥ 3 consecutive
//!               checks, bail-sell leg A to bound the loss.
//!               Also bails on time if `remaining < bail_secs` and
//!               the position is at a loss.
//!
//! With the box complete the state is frozen; `src/settlement/orphan.rs` handles the
//! settlement payout after the close (no changes needed).

use std::collections::HashMap;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::config::defaults;
use crate::{
    fair_value::{
        breakeven_price_parabolic, fair_value_yes, fee_taker_parabolic_with_rate,
        TAKER_FEE_RATE_PARABOLIC,
    },
    types::{BookSnapshot, Fill, FillSide, OrderIntent, OrderSide, StrategyId, VolSnapshot},
};

use super::{fallback_reference, Strategy, StrategyConfig};

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct DriftBoxConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    /// Minimum |fv − 0.5| threshold to pick a side. Higher = more confidence required.
    #[serde(default = "defaults::db_min_confidence")]
    pub min_confidence: Decimal,
    /// Maximum ask of leg A on entry. Leaves margin so the box is profitable.
    #[serde(default = "defaults::db_max_entry_price")]
    pub max_entry_price: Decimal,
    /// Minimum profit required when completing the box: cost_a + opp_ask ≤ 1 − fee − this.
    #[serde(default = "defaults::db_min_box_profit")]
    pub min_box_profit: Decimal,
    /// Fee assumed in the box threshold calculation (same semantics as ARB).
    #[serde(default = "defaults::arb_fee_threshold")]
    pub fee_threshold: Decimal,
    /// Relative adverse retracement of the underlying that triggers bail-sell (fraction, not %).
    #[serde(default = "defaults::db_bail_threshold_pct")]
    pub bail_threshold_pct: Decimal,
    /// Minimum bid of leg A to execute bail-sell. 0 = no minimum.
    /// Protects against selling into a dry market (bid = $0.01).
    #[serde(default = "defaults::nc_min_bail_bid")]
    pub min_bail_bid: Decimal,
    /// Minimum remaining time to enter (seconds). Leaves margin for B to get cheaper.
    #[serde(default = "defaults::db_min_lead_secs")]
    pub min_lead_secs: u64,
    /// Maximum remaining time to enter (seconds). Do not enter too far from the close.
    #[serde(default = "defaults::db_max_window_secs")]
    pub max_window_secs: u64,
    /// Seconds before the close to force bail-sell if the position is at a loss.
    #[serde(default = "defaults::db_bail_secs")]
    pub bail_secs: u64,
    /// Maximum size in shares per order. 0 = no cap; the primary limit is max_order_usdc.
    #[serde(default)]
    pub max_size_shares: Decimal,
    /// Maximum tolerable bid-ask spread on entry. 0 = no filter.
    #[serde(default)]
    pub max_bid_ask_spread: Decimal,
}

impl Default for DriftBoxConfig {
    fn default() -> Self {
        Self {
            limits: crate::config::StrategyLimits::default(),
            min_confidence: defaults::db_min_confidence(),
            max_entry_price: defaults::db_max_entry_price(),
            min_box_profit: defaults::db_min_box_profit(),
            fee_threshold: defaults::arb_fee_threshold(),
            bail_threshold_pct: defaults::db_bail_threshold_pct(),
            min_bail_bid: defaults::nc_min_bail_bid(),
            min_lead_secs: defaults::db_min_lead_secs(),
            max_window_secs: defaults::db_max_window_secs(),
            bail_secs: defaults::db_bail_secs(),
            max_size_shares: Decimal::ZERO,
            max_bid_ask_spread: Decimal::ZERO,
        }
    }
}

impl StrategyConfig for DriftBoxConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

// ── Per-bracket state ─────────────────────────────────────────────────────────

struct BracketState {
    clear_at: i64,
    /// Token of leg A (chosen by the oracle's prediction).
    token_a: String,
    /// Token of leg B (the opposite; to be completed when it gets cheaper).
    token_b: String,
    /// Price at which we bought A (updated when the fill is confirmed).
    cost_a: Decimal,
    /// Size of position A (updated when the fill is confirmed).
    size_a: Decimal,
    /// Buy fill of A confirmed.
    have_a: bool,
    /// Buy fill of B confirmed (box complete → hold to settlement).
    have_b: bool,
    /// Instant the A intent was emitted. If no fill arrives within
    /// IN_FLIGHT_RETRY_SECS → FOK rejected, retain clears it so it is re-evaluated.
    /// (Previously 20 ticks — under coalescing they do not guarantee real time.)
    entry_sent_at: crate::time::Instant,
    /// Bail-sell confirmed: do not re-enter this bracket even if the signal returns.
    bail_confirmed: bool,
    /// Instant of the in-flight box FOK (B). None = no active intent.
    box_sent_at: Option<crate::time::Instant>,
    /// Consecutive checks with cushion < bail_threshold.
    /// Per-tick on purpose — under coalescing the failure mode is a later
    /// bail (more widely spaced observations), not a duplicate; convert to time
    /// if calibration shows systematically slow bails.
    bail_checks: u32,
    /// Instant of the in-flight bail-sell. None = no active sell.
    bail_sent_at: Option<crate::time::Instant>,
    /// Reference price of the underlying when the entry was emitted (captured once).
    reference: Decimal,
}

/// Seconds without a fill for an in-flight order before considering it rejected.
/// Must exceed the async latency of user_ws (~0.7-2s); same value as
/// bail_retry_secs of NC/NE.
const IN_FLIGHT_RETRY_SECS: u64 = 10;

// ── DriftBox ──────────────────────────────────────────────────────────────────

pub struct DriftBox {
    cfg: DriftBoxConfig,
    /// Additive haircut over `fv_side` in the net-EV gate
    /// of leg A (`fair_value.confidence_haircut`, shared across strategies).
    haircut: Decimal,
    states: HashMap<String, BracketState>,
    last_retain: crate::time::Instant,
}

impl DriftBox {
    pub fn new(cfg: DriftBoxConfig, haircut: Decimal) -> Self {
        Self {
            cfg,
            haircut,
            states: HashMap::new(),
            last_retain: crate::time::Instant::now(),
        }
    }
}

impl Strategy for DriftBox {
    fn id(&self) -> StrategyId {
        StrategyId::DriftBox
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        let cid = &book.market_condition_id;
        let remaining = book.secs_to_clear();

        // Never trade expired brackets.
        if remaining <= 0 {
            return vec![];
        }

        // Cleanup throttled to 30s to amortize the cost of going over all names every tick.
        if self.last_retain.elapsed() >= std::time::Duration::from_secs(30) {
            self.states.retain(|_, s| {
                if s.clear_at <= now {
                    return false;
                }
                // Entry FOK rejected (no position or bail): clear so it is re-evaluated.
                if !s.have_a
                    && !s.bail_confirmed
                    && s.entry_sent_at.elapsed().as_secs() >= IN_FLIGHT_RETRY_SECS
                {
                    return false;
                }
                true
            });
            self.last_retain = crate::time::Instant::now();
        }

        // ── Existing state management ────────────────────────────────────────

        if let Some(state) = self.states.get_mut(cid.as_str()) {
            // No position in A.
            if !state.have_a {
                // A intent in flight (or bail confirmed): wait for the fill or for
                // retain to clear it via the time deadline.
                return vec![];
            }

            // Box complete: state frozen until expiry.
            if state.have_b {
                return vec![];
            }

            // ── Leg A filled, B pending ──────────────────────────────────────

            let a_is_yes_side = state.token_a == book.yes.token_id;

            // 1. Complete the box.
            if let Some(sent_at) = state.box_sent_at {
                // Time deadline — previously 20 coalesced ticks.
                if sent_at.elapsed().as_secs() >= IN_FLIGHT_RETRY_SECS {
                    state.box_sent_at = None;
                    metrics::counter!("hrf_driftbox_box_fok_rejected_total").increment(1);
                    tracing::debug!(cid = %cid, "DB: box FOK rejected — reset to retry");
                }
                return vec![];
            }

            let opp_ask = if a_is_yes_side {
                book.no.best_ask()
            } else {
                book.yes.best_ask()
            };
            if let Some(opp_ask) = opp_ask {
                let box_cost = state.cost_a + opp_ask;
                // Leg B also pays a taker fee — discount it with
                // the exact PARABOLIC CLOB fee (the market's real rate if
                // the book carries it, otherwise TAKER_FEE_RATE_PARABOLIC) instead of the flat
                // fee_threshold. fee_threshold becomes an optional additional cushion,
                // as in ARB. The MARGINAL gate (decides whether to complete the box) is
                // correct as is: A's fee is already a sunk cost at this point —
                // only B's fee matters for the decision to complete or not.
                let rate = book.parabolic_fee_rate_or(TAKER_FEE_RATE_PARABOLIC);
                let fee_b = fee_taker_parabolic_with_rate(rate, opp_ask, Decimal::ONE);
                let threshold = dec!(1) - fee_b - self.cfg.fee_threshold - self.cfg.min_box_profit;
                if box_cost <= threshold && state.size_a > Decimal::ZERO {
                    // The earlier log ("risk-free profit blocked") only
                    // discounted B's fee and ignored A's fee, claiming
                    // guaranteed profitability when the TOTAL cost (with both fees)
                    // may exceed 1. Here A's fee is included to report the
                    // real P&L of the box — the decision gate above does not change.
                    let fee_a = fee_taker_parabolic_with_rate(rate, state.cost_a, Decimal::ONE);
                    let total_cost = box_cost + fee_a + fee_b;
                    let pnl_total = Decimal::ONE - total_cost;
                    tracing::info!(
                        cid = %cid, cost_a = %state.cost_a, opp_ask = %opp_ask,
                        box_cost = %box_cost, marginal_margin = %(threshold - box_cost),
                        fee_a = %fee_a, fee_b = %fee_b, pnl_total = %pnl_total,
                        "DB: completing box — total box P&L after fees of both legs"
                    );
                    state.box_sent_at = Some(crate::time::Instant::now());
                    metrics::counter!("hrf_driftbox_box_attempts_total").increment(1);
                    return vec![OrderIntent {
                        market_cid: cid.clone(),
                        token_id: state.token_b.clone(),
                        side: OrderSide::Buy,
                        price: opp_ask,
                        size: state.size_a,
                        clear_at: state.clear_at,
                        signal: threshold - box_cost,
                        neg_risk: book.neg_risk,
                        limit_price: Some(opp_ask),
                        second_leg: None,
                        strategy: StrategyId::DriftBox,
                        resting: false,
                        real_clear_at: None,
                    }];
                }
            }

            // 2. Bail-sell (oracle against us or time).
            if let Some(sent_at) = state.bail_sent_at {
                // Time deadline — previously 20 ticks, fewer than the
                // user_ws latency under load → risk of a duplicate sell.
                if sent_at.elapsed().as_secs() >= IN_FLIGHT_RETRY_SECS {
                    state.bail_sent_at = None;
                    state.bail_checks = 0;
                    tracing::warn!(cid = %cid, secs = IN_FLIGHT_RETRY_SECS,
                        "DB: bail-sell without fill after the deadline — reset to retry");
                }
                return vec![];
            }

            let live = vol.last_price;
            let oracle_fresh = live > Decimal::ZERO && !vol.is_stale_for_signal();

            if oracle_fresh && state.reference > Decimal::ZERO {
                let cushion = if a_is_yes_side {
                    (live - state.reference) / state.reference
                } else {
                    (state.reference - live) / state.reference
                };

                let oracle_bad = cushion < -self.cfg.bail_threshold_pct;
                let force_time = remaining < self.cfg.bail_secs as i64 && cushion < Decimal::ZERO;

                if oracle_bad {
                    state.bail_checks += 1;
                } else {
                    state.bail_checks = 0;
                }

                if state.bail_checks >= 3 || force_time {
                    let bid_a = if a_is_yes_side {
                        book.yes.best_bid().unwrap_or(dec!(0.01))
                    } else {
                        book.no.best_bid().unwrap_or(dec!(0.01))
                    };

                    if bid_a < self.cfg.min_bail_bid {
                        tracing::warn!(
                            cid = %cid, bid = %bid_a, min = %self.cfg.min_bail_bid,
                            "DB: bail cancelled — low bid, hold-to-expiry"
                        );
                        state.bail_checks = 0;
                    } else {
                        let reason = if force_time {
                            "time"
                        } else {
                            "oracle against"
                        };
                        tracing::info!(
                            cid = %cid, cushion = %cushion, remaining, reason,
                            "DB: bail-sell — prediction reversed"
                        );
                        state.bail_sent_at = Some(crate::time::Instant::now());
                        state.bail_checks = 0;
                        metrics::counter!("hrf_driftbox_bail_total").increment(1);
                        return vec![sell_intent(book, &state.token_a, bid_a, state.size_a)];
                    }
                }
            }

            // 3. Hold: we are winning but B is not yet cheap enough,
            //    or oracle stale. At settlement the winning leg pays $1.
            return vec![];
        }

        // ── Entry signal (leg A) — bracket with no prior state ─────────────

        // Trading window: [min_lead_secs, max_window_secs].
        if remaining > self.cfg.max_window_secs as i64 {
            return vec![];
        }
        if remaining < self.cfg.min_lead_secs as i64 {
            return vec![];
        }

        // Resolve the underlying's reference (the bracket's strike). The fallback to the
        // oracle's current price is only valid near the open: past
        // that margin, reference ≈ live → fv ≈ 0.5 → the strategy would die
        // silently or enter on a false signal.
        let reference = match book
            .reference_price
            .or_else(|| fallback_reference(vol, book, now, cid, "DriftBox"))
        {
            Some(r) if r > Decimal::ZERO => r,
            _ => return vec![],
        };

        if vol.is_stale_for_signal() {
            return vec![];
        }
        let sigma = match vol.sigma_long.or(vol.sigma_short) {
            Some(s) if s > Decimal::ZERO => s,
            _ => return vec![],
        };
        let live = vol.last_price;
        if live <= Decimal::ZERO {
            return vec![];
        }

        let fv = match fair_value_yes(live, reference, sigma, remaining) {
            Some(v) => v,
            None => return vec![],
        };

        // Pick a side by confidence: |fv − 0.5| > min_confidence.
        let half = dec!(0.5);
        let is_yes = if fv > half + self.cfg.min_confidence {
            true
        } else if fv < half - self.cfg.min_confidence {
            false
        } else {
            metrics::counter!("hrf_driftbox_skip_confidence_total").increment(1);
            tracing::debug!(
                cid = %cid, fv = %fv, threshold = %self.cfg.min_confidence,
                "DB: skip — insufficient confidence"
            );
            return vec![];
        };

        let (ask, token_a, token_b) = if is_yes {
            match book.yes.best_ask() {
                Some(a) => (a, book.yes.token_id.clone(), book.no.token_id.clone()),
                None => return vec![],
            }
        } else {
            match book.no.best_ask() {
                Some(a) => (a, book.no.token_id.clone(), book.yes.token_id.clone()),
                None => return vec![],
            }
        };

        // Margin for the box: if A is too expensive there is no room for a profitable box.
        if ask > self.cfg.max_entry_price {
            metrics::counter!("hrf_driftbox_skip_price_total").increment(1);
            tracing::debug!(
                cid = %cid, ask = %ask, max = %self.cfg.max_entry_price,
                "DB: skip — leg A too expensive for a profitable box"
            );
            return vec![];
        }

        // The FV of the bought side must cover the ask PLUS the PARABOLIC taker
        // fee of leg A (previously the gate was only
        // min_confidence — nominal edge ≠ net). The market's real rate if the book
        // carries it, otherwise TAKER_FEE_RATE_PARABOLIC.
        let fv_side = if is_yes { fv } else { dec!(1) - fv };
        let breakeven = breakeven_price_parabolic(book, ask);
        // Additive haircut — ALWAYS subtracts in a
        // BUY gate (moves the decision toward not trading). Sign trap: do not add.
        if fv_side - self.haircut <= breakeven {
            if fv_side > breakeven {
                metrics::counter!("hrf_fv_haircut_block_total", "strategy" => "drift_box")
                    .increment(1);
            }
            metrics::counter!("hrf_driftbox_skip_fee_edge_total").increment(1);
            tracing::debug!(cid = %cid, fv = %fv_side, ask = %ask, breakeven = %breakeven, haircut = %self.haircut,
                "DB: skip — edge does not cover the taker fee of leg A");
            return vec![];
        }

        // Bid-ask spread filter. With no bid (`None`) it must fail CLOSED
        // (reject the entry), like directional.rs — previously a book with no
        // bid let the entry through without applying the filter.
        if self.cfg.max_bid_ask_spread > Decimal::ZERO {
            let bid = if is_yes {
                book.yes.best_bid()
            } else {
                book.no.best_bid()
            };
            if !bid.is_some_and(|b| ask - b <= self.cfg.max_bid_ask_spread) {
                tracing::debug!(cid = %cid, "DB: skip — wide spread or no bid");
                return vec![];
            }
        }

        // Sizing: caps max_order_usdc / max_size_shares ∧ available depth.
        let depth = if is_yes {
            book.yes.asks.best_size()
        } else {
            book.no.asks.best_size()
        };
        let mut size = super::size_from_caps(
            self.cfg.limits.max_order_usdc,
            self.cfg.max_size_shares,
            ask,
            None,
            depth,
        );
        // FOK: cap to the available depth instead of rejecting everything.
        match depth {
            None => return vec![],
            Some(d) if d < size => {
                size = d;
                if size <= Decimal::ZERO {
                    return vec![];
                }
            }
            _ => {}
        }
        // Minimum $1 USDC per order (Polymarket requirement).
        let size_int = size.trunc_with_scale(0);
        if size_int < Decimal::ONE || size_int * ask < dec!(1) {
            tracing::debug!(
                cid = %cid, ask = %ask, size = %size_int,
                "DB: skip — amount < $1 Polymarket minimum"
            );
            return vec![];
        }

        let signal = (fv - half).abs(); // magnitude of the prediction

        self.states.insert(
            cid.clone(),
            BracketState {
                clear_at: book.clear_at,
                token_a: token_a.clone(),
                token_b,
                cost_a: ask, // updated when the fill is confirmed
                size_a: Decimal::ZERO,
                have_a: false,
                have_b: false,
                entry_sent_at: crate::time::Instant::now(),
                bail_confirmed: false,
                box_sent_at: None,
                bail_checks: 0,
                bail_sent_at: None,
                reference,
            },
        );

        metrics::counter!("hrf_driftbox_signals_total").increment(1);
        tracing::info!(
            cid = %cid, is_yes, fv = %fv, ask = %ask, remaining,
            "DB: entry signal — leg A"
        );

        vec![OrderIntent {
            market_cid: cid.clone(),
            token_id: token_a,
            side: OrderSide::Buy,
            price: ask,
            size: size_int,
            clear_at: book.clear_at,
            signal,
            neg_risk: book.neg_risk,
            limit_price: Some(ask),
            second_leg: None,
            strategy: StrategyId::DriftBox,
            resting: false,
            real_clear_at: None,
        }]
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::DriftBox {
            return;
        }
        // Buy fill without state (restart mid-bracket or state already purged) —
        // DriftBox cannot rebuild the management (reference/token_b unknown);
        // the orphan checker will settle the position at settlement. Make it visible.
        if !self.states.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "DB: Buy fill without state — position left to the orphan checker (restart mid-bracket)");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "drift_box")
                    .increment(1);
            }
            return;
        }
        if let Some(state) = self.states.get_mut(&fill.market_cid) {
            match fill.side {
                FillSide::Buy => {
                    if fill.token_id == state.token_a {
                        // Fill of leg A: enable box and bail management.
                        state.have_a = true;
                        state.cost_a = fill.price;
                        state.size_a += fill.size; // sum: possible partial GTD fill
                        metrics::counter!("hrf_driftbox_fill_total").increment(1);
                    } else {
                        // Fill of leg B: box complete, hold to settlement.
                        state.have_b = true;
                        state.box_sent_at = None;
                        metrics::counter!("hrf_driftbox_box_completed_total").increment(1);
                        tracing::info!(
                            cid = %fill.market_cid, cost_a = %state.cost_a,
                            cost_b = %fill.price, total = %(state.cost_a + fill.price),
                            "DB: box complete — hold to settlement"
                        );
                    }
                }
                FillSide::Sell => {
                    // Bail-sell confirmed. Partial sell (FAK capped at depth):
                    // the rest stays under management until everything is closed.
                    state.size_a = (state.size_a - fill.size).max(Decimal::ZERO);
                    state.bail_sent_at = None;
                    if state.size_a == Decimal::ZERO {
                        state.have_a = false;
                        state.bail_confirmed = true;
                        tracing::info!(cid = %fill.market_cid, "DB: bail-sell confirmed");
                    }
                }
                // Async fill: the real fill may arrive up to ~2s later.
                // Do not delete the state yet; the entry_sent_at deadline handles it.
                // Same for GateRejected — the latch has a timestamp and
                // retries only on expiry (IN_FLIGHT_RETRY_SECS).
                FillSide::Rejected | FillSide::GateRejected => {}
                // They carry no live-order accounting — nothing to clean up.
                FillSide::Cancelled => {}
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sell_intent(book: &BookSnapshot, token_id: &str, price: Decimal, size: Decimal) -> OrderIntent {
    OrderIntent {
        market_cid: book.market_condition_id.clone(),
        token_id: token_id.to_string(),
        side: OrderSide::Sell { size: Some(size) },
        price,
        size,
        clear_at: book.clear_at,
        signal: price,
        neg_risk: book.neg_risk,
        limit_price: None,
        second_leg: None,
        strategy: StrategyId::DriftBox,
        resting: false,
        real_clear_at: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};

    fn make_book(
        yes_ask: &str,
        yes_bid: &str,
        no_ask: &str,
        no_bid: &str,
        remaining: i64,
        reference: Option<&str>,
    ) -> BookSnapshot {
        let level = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        BookSnapshot {
            market_condition_id: "db-cid".into(),
            yes: TokenBook {
                token_id: "yes-tok".into(),
                bids: level(yes_bid),
                asks: level(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-tok".into(),
                bids: level(no_bid),
                asks: level(no_ask),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-db".into(),
            reference_price: reference.map(|p| p.parse().unwrap()),
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        }
    }

    fn make_vol(live: &str, sigma: &str) -> VolSnapshot {
        VolSnapshot {
            asset: "btc".into(),
            sigma_short: Some(sigma.parse().unwrap()),
            sigma_long: Some(sigma.parse().unwrap()),
            last_price: live.parse().unwrap(),
            updated_at: crate::time::Instant::now(),
            ..VolSnapshot::default()
        }
    }

    fn make_fill(token: &str, side: FillSide, price: &str, size: &str) -> Fill {
        Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "db-cid".into(),
            token_id: token.into(),
            side,
            price: price.parse().unwrap(),
            size: size.parse().unwrap(),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::DriftBox,
            clear_at: 0,
            outcome_label: None,
        }
    }

    fn test_cfg() -> DriftBoxConfig {
        DriftBoxConfig::default()
    }

    #[test]
    fn db_enters_leg_a_with_confidence() {
        // live=110k, reference=100k, sigma=0.01/√s, remaining=120s
        // denom = 0.01*√120 ≈ 0.1095; d = ln(1.1)/0.1095 ≈ 0.0953/0.1095 ≈ 0.87
        // Φ(0.87) ≈ 0.808 > 0.5 + 0.08 = 0.58 → enters YES
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.50", "0.46", 120, Some("100000"));
        let vol = make_vol("110000", "0.01");
        let intents = db.on_book(&book, &vol);
        assert_eq!(intents.len(), 1);
        assert!(matches!(intents[0].side, OrderSide::Buy));
        assert_eq!(intents[0].token_id, "yes-tok");
    }

    #[test]
    fn db_without_reference_price_and_old_open_ts_does_not_trade() {
        // With no reference_price from the feed and the bracket having been open for a while
        // (market_open_ts=0), the oracle fallback is discarded → 0 intents,
        // even though the book and vol would allow entering with a false reference.
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.50", "0.46", 120, None);
        let vol = make_vol("110000", "0.01");
        assert!(
            db.on_book(&book, &vol).is_empty(),
            "mid-bracket without a reference must not trade"
        );
    }

    #[test]
    fn db_skip_without_confidence() {
        // live=reference → fv=0.5 → no confidence
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.50", "0.46", 120, Some("100000"));
        let vol = make_vol("100000", "0.01");
        assert!(db.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn db_skip_outside_window() {
        // remaining > max_window_secs (default 240)
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.50", "0.46", 3600, Some("100000"));
        let vol = make_vol("110000", "0.01");
        assert!(db.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn db_completes_box_when_b_cheap() {
        // With the real parabolic fee (`0.07 × p × (1−p)`, fallback
        // TAKER_FEE_RATE_PARABOLIC) fee(0.35) ≈ 0.0159, and 0.52+0.35=0.87 ≤
        // 1−0.0159−0.02−0.03 = 0.9341 ✓ — the box is still profitable.
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.60", "0.56", 120, Some("100000"));
        let vol = make_vol("110000", "0.01");

        // Entry signal
        let _ = db.on_book(&book, &vol);
        db.on_fill(&make_fill("yes-tok", FillSide::Buy, "0.52", "10"));

        // B's price drops (correct prediction: YES rises, NO drops)
        let book2 = make_book("0.75", "0.70", "0.35", "0.31", 90, Some("100000"));
        let intents = db.on_book(&book2, &vol);
        assert_eq!(
            intents.len(),
            1,
            "must complete the box when B gets cheaper"
        );
        assert_eq!(intents[0].token_id, "no-tok");
        assert!(matches!(intents[0].side, OrderSide::Buy));
    }

    #[test]
    fn db_does_not_complete_box_if_unprofitable() {
        // cost_a=0.52, opp_ask=0.52 → sum=1.04 > 0.95 → does not buy B
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.52", "0.48", 120, Some("100000"));
        let vol = make_vol("110000", "0.01");

        let _ = db.on_book(&book, &vol);
        db.on_fill(&make_fill("yes-tok", FillSide::Buy, "0.52", "10"));

        let intents = db.on_book(&book, &vol);
        assert!(
            intents.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "must not complete the box if the combined cost exceeds the threshold"
        );
    }

    #[test]
    fn db_bail_sell_prediction_against() {
        // After the YES fill, oracle reverts -5% → 3 consecutive checks → bail
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.60", "0.56", 120, Some("100000"));
        let vol = make_vol("110000", "0.01");

        let _ = db.on_book(&book, &vol);
        db.on_fill(&make_fill("yes-tok", FillSide::Buy, "0.52", "10"));

        // oracle reverts: live < reference → cushion_YES < 0
        let vol_bad = make_vol("90000", "0.01"); // -10% → cushion < -0.015
        let mut sells = vec![];
        for _ in 0..5 {
            let i = db.on_book(&book, &vol_bad);
            sells.extend(
                i.into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            1,
            "must emit bail-sell after 3 consecutive checks against us"
        );
        assert_eq!(sells[0].token_id, "yes-tok");
    }

    #[test]
    fn db_no_reentry_after_confirmed_bail() {
        // After a confirmed bail-sell, it must not emit a new entry in the same bracket.
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.60", "0.56", 120, Some("100000"));
        let vol = make_vol("110000", "0.01");

        let _ = db.on_book(&book, &vol);
        db.on_fill(&make_fill("yes-tok", FillSide::Buy, "0.52", "10"));

        let vol_bad = make_vol("90000", "0.01");
        for _ in 0..5 {
            let _ = db.on_book(&book, &vol_bad);
        }

        db.on_fill(&make_fill("yes-tok", FillSide::Sell, "0.48", "10"));

        // Now the signal returns (oracle recovered): must not re-enter
        let intents = db.on_book(&book, &vol);
        assert!(intents.is_empty(), "must not re-enter after confirmed bail");
    }

    /// With a nominal edge of 3pp (< haircut 4pp) DriftBox must NOT
    /// enter leg A; with a nominal edge of 5pp (> haircut) it MUST enter.
    /// Same method as the analogous NC test: `ask` fixed within
    /// `max_entry_price` (0.58) and `parabolic_fee_rate` (parabolic
    /// model, not the triangular `fee_rate_bps`) derived so the
    /// breakeven falls exactly at `fv_side - 0.03`/`-0.05`.
    #[test]
    fn db_haircut_blocks_edge_3pp_and_lets_edge_5pp_through() {
        let live = dec!(105000);
        let reference = dec!(100000);
        let sigma = dec!(0.001);
        let remaining = 60i64;
        let fv_side =
            fair_value_yes(live, reference, sigma, remaining).expect("fv must be computable");

        let ask = dec!(0.50); // <= max_entry_price (0.58)
        let haircut = dec!(0.04);

        // Inverse of fee_taker_parabolic_with_rate: breakeven = ask + rate*ask*(1-ask).
        let rate_for_breakeven =
            |breakeven: Decimal| -> Decimal { (breakeven - ask) / (ask * (Decimal::ONE - ask)) };

        let vol = make_vol("105000", "0.001");

        // Case 1: nominal edge 3pp < haircut 4pp → must BLOCK.
        let mut book_3pp = make_book("0.50", "0.46", "0.50", "0.46", remaining, Some("100000"));
        book_3pp.parabolic_fee_rate = Some(rate_for_breakeven(fv_side - dec!(0.03)));
        let mut db_3pp = DriftBox::new(test_cfg(), haircut);
        assert!(
            db_3pp.on_book(&book_3pp, &vol).is_empty(),
            "nominal edge 3pp < haircut 4pp — DriftBox must not enter leg A"
        );

        // Case 2: nominal edge 5pp > haircut 4pp → must PASS.
        let mut book_5pp = make_book("0.50", "0.46", "0.50", "0.46", remaining, Some("100000"));
        book_5pp.parabolic_fee_rate = Some(rate_for_breakeven(fv_side - dec!(0.05)));
        let mut db_5pp = DriftBox::new(test_cfg(), haircut);
        assert_eq!(
            db_5pp.on_book(&book_5pp, &vol).len(),
            1,
            "nominal edge 5pp > haircut 4pp — DriftBox must enter leg A"
        );
    }

    /// A book WITHOUT a bid on leg A must fail CLOSED in the spread
    /// filter, not let the entry through without applying the filter.
    #[test]
    fn db_without_bid_fails_closed_in_spread_filter() {
        let cfg = DriftBoxConfig {
            max_bid_ask_spread: dec!(0.05),
            ..DriftBoxConfig::default()
        };
        let mut db = DriftBox::new(cfg, Decimal::ZERO);
        let mut book = make_book("0.52", "0.48", "0.50", "0.46", 120, Some("100000"));
        book.yes.bids = HalfBook { levels: vec![] }; // no bid on the YES leg (candidate for A)
        let vol = make_vol("110000", "0.01");
        assert!(
            db.on_book(&book, &vol).is_empty(),
            "with no bid the spread filter must reject the entry, not let it through"
        );
    }

    /// The "box complete" log must reflect the TOTAL P&L (including
    /// leg A's fee), not just the marginal margin that discounts B's fee.
    /// There is no log assert here (tracing is not captured in unit tests) — the test
    /// documents that the box is completed anyway (marginal gate unchanged) even though
    /// the total cost with both fees may be higher than the marginal margin reported
    /// before the fix.
    #[test]
    fn db_completes_box_with_fee_of_both_legs_counted_in_pnl() {
        let mut db = DriftBox::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.52", "0.48", "0.60", "0.56", 120, Some("100000"));
        let vol = make_vol("110000", "0.01");

        let _ = db.on_book(&book, &vol);
        db.on_fill(&make_fill("yes-tok", FillSide::Buy, "0.52", "10"));

        let book2 = make_book("0.75", "0.70", "0.35", "0.31", 90, Some("100000"));
        let intents = db.on_book(&book2, &vol);
        assert_eq!(
            intents.len(),
            1,
            "the marginal gate (B's fee) still decides whether to complete the box"
        );
    }
}
