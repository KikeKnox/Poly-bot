//! Near-Certain (NC) strategy.
//!
//! Signal: when the underlying has moved enough sigmas (z >= z_threshold)
//! towards the winning bracket side, buy the favorite side before close.
//!
//! Bail-sell: if the price reverts and the cushion drops below bail_threshold_pct
//! for N consecutive checks, sell the position.

use std::collections::HashMap;

use rust_decimal::{Decimal, MathematicalOps};
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    config::defaults,
    fair_value::{breakeven_price_parabolic, fair_value_yes},
    types::{
        BookSnapshot, Fill, FillSide, HalfBook, OrderIntent, OrderSide, StrategyId, VolSnapshot,
    },
};

use super::{fallback_reference, Strategy, StrategyConfig};

// ── NearCertainConfig ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct NearCertainConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    #[serde(default = "defaults::nc_z_threshold")]
    pub z_threshold: Decimal,
    #[serde(default = "defaults::nc_window_secs")]
    pub window_secs: u64,
    #[serde(default = "defaults::nc_min_ask")]
    pub min_ask: Decimal,
    #[serde(default = "defaults::nc_max_ask")]
    pub max_ask: Decimal,
    #[serde(default = "defaults::nc_min_book_imbalance")]
    pub min_book_imbalance: Decimal,
    #[serde(default = "defaults::nc_bail_threshold_pct")]
    pub bail_threshold_pct: Decimal,
    /// Maximum size in shares per order. 0 = no cap; the limit is max_order_usdc.
    #[serde(default)]
    pub max_size_shares: Decimal,
    /// Minimum bid to run a bail-sell. 0 = no minimum (sells at the available bid).
    /// Protects against selling into a dry market where the bid = $0.01.
    #[serde(default = "defaults::nc_min_bail_bid")]
    pub min_bail_bid: Decimal,
    /// Maximum tolerable bid-ask spread on entry. 0 = no filter.
    /// Avoids entering low-liquidity brackets where the spread reflects high uncertainty.
    #[serde(default)]
    pub max_bid_ask_spread: Decimal,
    /// If true, uses FAK (Fill-And-Kill) on entry: fills the available depth and cancels the rest.
    /// Default false (FOK). FAK captures partial fills in brackets with limited liquidity.
    #[serde(default)]
    pub use_fak: bool,
    /// Seconds to wait before giving up on an in-flight bail-sell and
    /// allowing a retry. Time-based deadline (not book ticks, whose cadence
    /// varies): it must exceed the async user_ws latency in live (700ms-2s).
    #[serde(default = "defaults::nc_bail_retry_secs")]
    pub bail_retry_secs: u64,
}

impl Default for NearCertainConfig {
    fn default() -> Self {
        Self {
            limits: crate::config::StrategyLimits::default(),
            z_threshold: defaults::nc_z_threshold(),
            window_secs: defaults::nc_window_secs(),
            min_ask: defaults::nc_min_ask(),
            max_ask: defaults::nc_max_ask(),
            min_book_imbalance: defaults::nc_min_book_imbalance(),
            bail_threshold_pct: defaults::nc_bail_threshold_pct(),
            max_size_shares: Decimal::ZERO,
            min_bail_bid: defaults::nc_min_bail_bid(),
            max_bid_ask_spread: Decimal::ZERO,
            use_fak: false,
            bail_retry_secs: defaults::nc_bail_retry_secs(),
        }
    }
}

impl StrategyConfig for NearCertainConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

// ── Per-bracket state ────────────────────────────────────────────────────────

struct MarketState {
    clear_at: i64,
    /// Reference price of the underlying when the bracket opened.
    reference: Option<Decimal>,
    /// Whether we already hold an open position (waiting for bail or expiry).
    have_position: bool,
    /// Token we bought (to know which one to sell on bail-sell).
    open_token: Option<String>,
    /// Size (shares) of the open position, for the bail-sell.
    open_size: Decimal,
    /// Consecutive checks with cushion < bail_threshold.
    /// Per-tick on purpose: under `drain_latest` coalescing the failure mode
    /// is a later bail (fewer checks per second), not a duplicated
    /// bail; switch to a time deadline if calibration shows systematically
    /// slow bails (TC documents the same limitation in its own counter).
    bail_checks: u32,
    /// Instant at which the bail-sell was emitted (None = no sell in flight).
    /// Time-based deadline, not tick-based: book cadence varies and at
    /// ~10 ticks/s the old counter (20 ticks ~ 2s) could reset before the
    /// fill arrived via user_ws (700ms-2s) and re-emit a duplicate sell.
    bail_sent_at: Option<crate::time::Instant>,
    /// Entry emitted in this bracket (only once per bracket, set when the
    /// intent is emitted, not in on_fill, to close the FOK-timeout race).
    ever_entered: bool,
}

// ── NearCertain ───────────────────────────────────────────────────────────────

pub struct NearCertain {
    cfg: NearCertainConfig,
    /// Additive haircut over `fv_side` in the net-EV gate
    /// (`fair_value.confidence_haircut`, shared across strategies).
    haircut: Decimal,
    markets: HashMap<String, MarketState>,
    last_retain: crate::time::Instant,
}

impl NearCertain {
    pub fn new(cfg: NearCertainConfig, haircut: Decimal) -> Self {
        Self {
            cfg,
            haircut,
            markets: HashMap::new(),
            last_retain: crate::time::Instant::now(),
        }
    }
}

impl Strategy for NearCertain {
    fn id(&self) -> StrategyId {
        StrategyId::NearCertain
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        if self.last_retain.elapsed() >= std::time::Duration::from_secs(30) {
            self.markets.retain(|_, s| s.clear_at > now);
            self.last_retain = crate::time::Instant::now();
        }

        let mut intents = Vec::new();
        let remaining = book.secs_to_clear();

        // Trading window: [min_lead_secs, window_secs]. This gate must NOT
        // bound the bail-sell (below, under `state.have_position`), only the
        // entry; the real lower bound for the ENTRY is applied further down
        // (`NC_MIN_LEAD_SECS`) so the bail is not blocked near
        // `remaining=0`.
        if remaining <= 0 || remaining > self.cfg.window_secs as i64 {
            return intents;
        }

        let cid = &book.market_condition_id;
        let state = self
            .markets
            .entry(cid.clone())
            .or_insert_with(|| MarketState {
                clear_at: book.clear_at,
                reference: book.reference_price,
                have_position: false,
                open_token: None,
                open_size: Decimal::ZERO,
                bail_checks: 0,
                bail_sent_at: None,
                ever_entered: false,
            });

        // Resolve reference: first book.reference_price, then the oracle ONLY near
        // the open (same guard as NE/DriftBox: mid-bracket the current price
        // is not the opening price and the z-score would be garbage).
        if state.reference.is_none() {
            state.reference = book
                .reference_price
                .or_else(|| fallback_reference(vol, book, now, cid, "NC"));
        }
        let reference = match state.reference {
            Some(r) if r > Decimal::ZERO => r,
            _ => return intents,
        };

        let live = vol.last_price;
        let oracle_fresh = live > Decimal::ZERO && !vol.is_stale_for_signal();

        // ── Bail-sell for an open position ────────────────────────────────────
        if state.have_position {
            if let Some(token) = state.open_token.as_deref() {
                if let Some(sent_at) = state.bail_sent_at {
                    // Sell already in flight: time-based deadline. If it takes longer than
                    // bail_retry_secs (sell rejected or network timeout), reset.
                    if sent_at.elapsed().as_secs() >= self.cfg.bail_retry_secs {
                        state.bail_sent_at = None;
                        state.bail_checks = 0;
                        tracing::warn!(
                            cid = %cid, secs = self.cfg.bail_retry_secs,
                            "NC: bail-sell without fill after the deadline; reset for retry"
                        );
                    }
                } else if oracle_fresh {
                    let bought_yes = token == book.yes.token_id;
                    // cushion > 0: underlying in our favor; < 0: it has reverted below the reference.
                    let cushion = if bought_yes {
                        (live - reference) / reference
                    } else {
                        (reference - live) / reference
                    };

                    if cushion < -self.cfg.bail_threshold_pct {
                        state.bail_checks += 1;
                        if state.bail_checks >= 3 {
                            let token_bid = if bought_yes {
                                book.yes.best_bid().unwrap_or(dec!(0.01))
                            } else {
                                book.no.best_bid().unwrap_or(dec!(0.01))
                            };
                            if token_bid < self.cfg.min_bail_bid {
                                tracing::warn!(
                                    cid = %cid, bid = %token_bid, min = %self.cfg.min_bail_bid,
                                    "NC: bail-sell cancelled — bid too low, hold-to-expiry"
                                );
                                state.bail_checks = 0;
                            } else {
                                intents.push(sell_intent(book, token, token_bid, state.open_size));
                                state.bail_checks = 0;
                                state.bail_sent_at = Some(crate::time::Instant::now());
                                metrics::counter!("hrf_nc_bail_total").increment(1);
                                tracing::info!(
                                    cid = %cid,
                                    cushion = %cushion,
                                    threshold = %(-self.cfg.bail_threshold_pct),
                                    "NC: bail-sell, underlying reverted below the threshold"
                                );
                            }
                        }
                    } else {
                        state.bail_checks = 0;
                    }
                }
                // oracle stale and no sell in flight -> no new decision; the position is
                // held until the oracle recovers or the bracket expires.
            }
            return intents;
        }

        // At most one entry per bracket: ever_entered is set when the intent is EMITTED
        // (not in on_fill) to close the signal -> FOK -> late fill -> double buy race.
        // A rejected FOK leaves the bracket consumed; this is accepted as a cost.
        if state.ever_entered {
            return intents;
        }

        if !oracle_fresh {
            return intents;
        }

        // The window comment above declared a lower bound (`min_lead_secs`)
        // that did NOT exist in `NearCertainConfig`: NC could enter as taker at
        // `remaining=1s`, with no margin for the bail-sell to act if the price
        // reverted right after the entry. Applies ONLY to the entry (the
        // bail-sell above does not share it and keeps covering until
        // `remaining=1`, same as NE).
        // The limit is applied here as a local constant, with the same default
        // value as `NearExpiryConfig::min_lead_secs` (`ne_min_lead_secs` in
        // config.rs, 5s). Migrating it to a real field of
        // `NearCertainConfig` (with its entry in
        // `config/strategies/near_certain.toml`) is pending.
        const NC_MIN_LEAD_SECS: i64 = 5;
        if remaining < NC_MIN_LEAD_SECS {
            metrics::counter!("hrf_nc_skip_insufficient_lead_total").increment(1);
            tracing::debug!(cid = %cid, remaining, min_lead = NC_MIN_LEAD_SECS,
                "NC: skip, remaining below the minimum lead, no margin for the bail-sell");
            return intents;
        }

        // ── Entry signal ──────────────────────────────────────────────────

        // Favorite side: the one whose ask is within [min_ask, max_ask] with tolerable spread.
        let (is_yes, ask) = match favor_side(
            book,
            self.cfg.min_ask,
            self.cfg.max_ask,
            self.cfg.max_bid_ask_spread,
        ) {
            Some(v) => v,
            None => return intents,
        };

        // Direction consistent with the underlying's move.
        // dist_frac: fractional return (dimensionless), same space as sigma.
        // Historical bug: an absolute USD difference (dist = live - reference) was used,
        // which made z proportional to the asset price and defeated the filter.
        if reference <= Decimal::ZERO {
            return intents;
        }
        let dist_frac = if is_yes {
            (live - reference) / reference
        } else {
            (reference - live) / reference
        };
        if dist_frac <= Decimal::ZERO {
            return intents;
        }

        // Sigma from the vol oracle (slow window = more stable for brackets >= 60 s).
        let sigma = match vol.sigma_long.or(vol.sigma_short) {
            Some(s) if s > Decimal::ZERO => s,
            _ => return intents,
        };

        // Z-score: fractional return in units of σ√t (dimensionless).
        let sqrt_t = Decimal::from(remaining.max(1)).sqrt().unwrap_or_default();
        let z = dist_frac / (sigma * sqrt_t);
        if z < self.cfg.z_threshold {
            metrics::counter!("hrf_nc_skip_z_total").increment(1);
            tracing::debug!(cid = %cid, z = %z, threshold = %self.cfg.z_threshold, dist_frac = %dist_frac, "NC: skip insufficient z");
            return intents;
        }

        // Net-EV gate: the probability implied by the model
        // (Φ(z) via fair_value) must cover the ask PLUS the PARABOLIC taker fee.
        // With a low z_threshold (calibration) NC used to buy asks above
        // its own probability. The rate uses the market's real fee
        // (`book.parabolic_fee_rate`, Gamma's `feeSchedule.rate`) if
        // discovery provided it; otherwise it falls back to the constant
        // TAKER_FEE_RATE_PARABOLIC. Previously the triangular model
        // (`breakeven_wr`/`fee_rate_bps`, deprecated source) was used, which overestimates the
        // fee ~1.5-2.9x and demands more fv than needed to enter.
        if let Some(fv) = fair_value_yes(live, reference, sigma, remaining) {
            let fv_side = if is_yes { fv } else { Decimal::ONE - fv };
            let breakeven = breakeven_price_parabolic(book, ask);
            // Additive haircut: ALWAYS subtracted in a BUY gate
            // (moves the decision towards not trading). Sign trap: do
            // not add here.
            if fv_side - self.haircut <= breakeven {
                // The haircut blocked a signal that would have passed without it;
                // numerator to evaluate whether 0.04 is the right value.
                if fv_side > breakeven {
                    metrics::counter!("hrf_fv_haircut_block_total", "strategy" => "near_certain")
                        .increment(1);
                }
                metrics::counter!("hrf_nc_skip_fee_edge_total").increment(1);
                tracing::debug!(cid = %cid, fv = %fv_side, ask = %ask, breakeven = %breakeven, haircut = %self.haircut,
                    "NC: skip, model probability does not cover ask + fee");
                return intents;
            }
        }

        // Imbalance: if the opposite depth exceeds N× the favorite's → skip.
        let fav_depth = if is_yes {
            book.yes.asks.best_size()
        } else {
            book.no.asks.best_size()
        };
        let opp_depth = if is_yes {
            book.no.bids.best_size()
        } else {
            book.yes.bids.best_size()
        };

        // `min_book_imbalance` was set to 1.5 with no empirical distribution
        // (1,052 of 1,114 candidates died here with 0.5). BOTH branches are
        // logged: logging only the rejections is what produced the false
        // "max z = 0.990". `depth3` = sum of the size of the 3 best levels: it allows
        // evaluating a less noisy statistic than top-of-book offline WITHOUT another run.
        let depth3 = |h: &HalfBook| -> Decimal { h.levels.iter().take(3).map(|l| l.size).sum() };
        let (fav_half, opp_half) = if is_yes {
            (&book.yes.asks, &book.no.bids)
        } else {
            (&book.no.asks, &book.yes.bids)
        };
        let ratio = match (opp_depth, fav_depth) {
            (Some(od), Some(fd)) if fd > Decimal::ZERO => Some(od / fd),
            _ => None,
        };
        let action = match (fav_depth, opp_depth) {
            (Some(fd), Some(od)) if od > fd * self.cfg.min_book_imbalance => "skip",
            (Some(_), Some(_)) => "pass",
            _ => "no_data",
        };
        metrics::counter!("hrf_nc_imbalance_eval_total", "action" => action).increment(1);
        let fav_side_bid = if is_yes {
            book.yes.best_bid()
        } else {
            book.no.best_bid()
        };
        let fav_side_ask = Some(ask);
        tracing::info!(target: "nc_imbalance",
            ts_ms       = crate::time::now().timestamp_millis(),
            cid         = %cid,
            slug        = %book.slug,
            asset       = %vol.asset,
            remaining   = remaining,
            is_yes      = is_yes,
            action      = action,
            z           = %z,
            dist_frac   = %dist_frac,
            ask         = %ask,
            fav_bid     = %opt_dec(fav_side_bid),      // to reconstruct spread and mid
            fav_ask     = %opt_dec(fav_side_ask),
            opp_bid_px  = %opt_dec(opp_half.best_bid()), // the opposite bid PRICE, not just its size
            fd_top      = %opt_dec(fav_depth),
            od_top      = %opt_dec(opp_depth),
            fd_top3     = %depth3(fav_half),
            od_top3     = %depth3(opp_half),
            ratio       = %opt_dec(ratio),
            threshold   = %self.cfg.min_book_imbalance,
            "NC: imbalance evaluation");
        if action == "skip" {
            metrics::counter!("hrf_nc_skip_imbalance_total").increment(1);
            return intents;
        }

        // No restriction: natural size = depth available at the favorite ask.
        let mut size = super::size_from_caps(
            self.cfg.limits.max_order_usdc,
            self.cfg.max_size_shares,
            ask,
            None,
            fav_depth,
        );
        if size <= Decimal::ZERO {
            return intents;
        }

        // FOK is all-or-nothing: if there is less depth than size, we cap to the available depth
        // instead of skipping the whole trade; more fills at smaller size > nothing.
        if !self.cfg.use_fak {
            match fav_depth {
                None => return intents,
                Some(d) if d < size => {
                    tracing::debug!(cid = %cid, size = %size, depth = %d, "NC: capping size to the available depth for FOK");
                    size = d;
                    if size <= Decimal::ZERO {
                        return intents;
                    }
                }
                _ => {}
            }
        }

        let token_id = if is_yes {
            book.yes.token_id.clone()
        } else {
            book.no.token_id.clone()
        };

        intents.push(OrderIntent {
            market_cid: cid.clone(),
            token_id: token_id.clone(),
            side: OrderSide::Buy,
            price: ask,
            size,
            clear_at: book.clear_at,
            signal: z,
            neg_risk: book.neg_risk,
            limit_price: Some(ask),
            second_leg: None,
            strategy: StrategyId::NearCertain,
            resting: false,
            real_clear_at: None,
        });

        state.ever_entered = true;

        metrics::counter!("hrf_nc_signals_total").increment(1);
        tracing::info!(
            cid = %cid, is_yes, ask = %ask, z = %z, remaining,
            "NC: entry signal"
        );

        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::NearCertain {
            return;
        }
        // Buy fill without state (restart mid-bracket): recreate the position
        // (same pattern as ARB). Without a reference the bail stays disabled (hold),
        // but Sells are accounted for and ever_entered blocks re-entry.
        if !self.markets.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "NC: Buy fill without state, recreating position (restart mid-bracket, bail disabled without reference)");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "near_certain").increment(1);
                self.markets.insert(
                    fill.market_cid.clone(),
                    MarketState {
                        clear_at: fill.clear_at,
                        reference: None,
                        have_position: true,
                        open_token: Some(fill.token_id.clone()),
                        open_size: fill.size,
                        bail_checks: 0,
                        bail_sent_at: None,
                        ever_entered: true,
                    },
                );
            }
            return;
        }
        if let Some(state) = self.markets.get_mut(&fill.market_cid) {
            match fill.side {
                // Buy also reconciles the LATE fill: even if it arrives
                // seconds after the signal, the position is recorded and
                // the bail-sell is operational; ever_entered already blocked re-entry.
                FillSide::Buy => {
                    state.have_position = true;
                    state.ever_entered = true;
                    state.bail_checks = 0;
                    // A taker can fill across several trades (live): accumulate into the
                    // same token; overwriting made the bail sell only
                    // the last partial.
                    if state.open_token.as_deref() == Some(fill.token_id.as_str()) {
                        state.open_size += fill.size;
                    } else {
                        state.open_size = fill.size;
                        state.open_token = Some(fill.token_id.clone());
                    }
                }
                FillSide::Sell => {
                    // Partial sell (FAK capped to the bid depth): the rest stays
                    // under management; previously the whole position was treated as closed.
                    state.open_size = (state.open_size - fill.size).max(Decimal::ZERO);
                    state.bail_sent_at = None;
                    if state.open_size == Decimal::ZERO {
                        state.have_position = false;
                        state.open_token = None;
                    }
                }
                // Same semantics as `Rejected` (a single entry per
                // bracket; the gate blocker is equivalent to a rejection).
                FillSide::Rejected | FillSide::GateRejected => {}
                // They carry no live-order accounting, so there is nothing to clear.
                FillSide::Cancelled => {}
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns (is_yes, ask) of the side whose ask falls in [min_ask, max_ask] and spread <= max_spread.
/// Prefers YES if both qualify. max_spread = 0 → no spread filter.
fn favor_side(
    book: &BookSnapshot,
    min_ask: Decimal,
    max_ask: Decimal,
    max_spread: Decimal,
) -> Option<(bool, Decimal)> {
    let spread_ok = |ask: Decimal, bid: Option<Decimal>| -> bool {
        max_spread == Decimal::ZERO || bid.is_some_and(|b| ask - b <= max_spread)
    };
    if let Some(yes_ask) = book.yes.best_ask() {
        if yes_ask >= min_ask && yes_ask <= max_ask && spread_ok(yes_ask, book.yes.best_bid()) {
            return Some((true, yes_ask));
        }
    }
    if let Some(na) = book.no.best_ask() {
        if na >= min_ask && na <= max_ask && spread_ok(na, book.no.best_bid()) {
            return Some((false, na));
        }
    }
    None
}

/// Optional datum for the `nc_imbalance` log: empty string
/// if there is no value. Distinguishes "no data" from "zero": a queue of 0 shares is
/// information; an unavailable signal is not.
fn opt_dec(d: Option<Decimal>) -> String {
    d.map_or(String::new(), |v| v.to_string())
}

fn sell_intent(book: &BookSnapshot, token_id: &str, price: Decimal, size: Decimal) -> OrderIntent {
    OrderIntent {
        market_cid: book.market_condition_id.clone(),
        token_id: token_id.to_string(),
        side: OrderSide::Sell { size: Some(size) },
        price,
        size,
        clear_at: book.clear_at,
        signal: price, // token price (0-1), not the underlying's
        neg_risk: book.neg_risk,
        limit_price: None,
        second_leg: None,
        strategy: StrategyId::NearCertain,
        resting: false,
        real_clear_at: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};
    use rust_decimal_macros::dec;

    fn make_book(
        yes_ask: &str,
        no_ask: &str,
        remaining: i64,
        reference: Option<&str>,
    ) -> BookSnapshot {
        let ask_level = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        BookSnapshot {
            market_condition_id: "cid-test".into(),
            yes: TokenBook {
                token_id: "yes-token".into(),
                bids: HalfBook { levels: vec![] },
                asks: ask_level(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-token".into(),
                bids: HalfBook { levels: vec![] },
                asks: ask_level(no_ask),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-test".into(),
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
            market_cid: "cid-test".into(),
            token_id: token.into(),
            side,
            price: price.parse().unwrap(),
            size: size.parse().unwrap(),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::NearCertain,
            clear_at: 0,
            outcome_label: None,
        }
    }

    #[test]
    fn late_nc_fill_does_not_cause_second_buy() {
        // Signal -> FOK with no immediate fill -> many ticks -> late fill via
        // user_ws. At no point may a second buy order be emitted.
        let mut nc = NearCertain::new(NearCertainConfig::default(), Decimal::ZERO);
        let book = make_book("0.92", "0.08", 60, Some("100000"));
        let vol = make_vol("105000", "0.001");
        assert_eq!(nc.on_book(&book, &vol).len(), 1, "initial signal");

        for _ in 0..50 {
            assert!(
                nc.on_book(&book, &vol).is_empty(),
                "without a fill the entry must not be re-emitted"
            );
        }

        // Late fill: reconcile the position without a second buy order.
        nc.on_fill(&make_fill("yes-token", FillSide::Buy, "0.92", "10"));
        for _ in 0..10 {
            let i = nc.on_book(&book, &vol);
            assert!(
                i.iter().all(|x| !matches!(x.side, OrderSide::Buy)),
                "after the late fill there must be no second buy order"
            );
        }
    }

    #[test]
    fn nc_bail_temporal_deadline_allows_retry() {
        // The bail-sell pending expires by TIME (bail_retry_secs), not
        // by book ticks. With a 0 deadline the retry is enabled immediately.
        let cfg = NearCertainConfig {
            bail_retry_secs: 0,
            ..NearCertainConfig::default()
        };
        let mut nc = NearCertain::new(cfg, Decimal::ZERO);
        let mut book = make_book("0.92", "0.08", 60, Some("100000"));
        book.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.85),
                size: dec!(100),
            }],
        };
        let _ = nc.on_book(&book, &make_vol("105000", "0.001"));
        nc.on_fill(&make_fill("yes-token", FillSide::Buy, "0.92", "10"));

        // Oracle reverts −5% (< −bail_threshold 4%): on the 3rd check it emits the sell.
        let vol_down = make_vol("95000", "0.001");
        let count_sells = |i: Vec<OrderIntent>| {
            i.iter()
                .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
                .count()
        };
        let mut sells = 0;
        for _ in 0..3 {
            sells += count_sells(nc.on_book(&book, &vol_down));
        }
        assert_eq!(sells, 1, "bail-sell after 3 checks of adverse cushion");

        // Deadline 0 → one tick resets the pending and 3 more checks re-emit.
        let mut sells2 = 0;
        for _ in 0..5 {
            sells2 += count_sells(nc.on_book(&book, &vol_down));
        }
        assert_eq!(
            sells2, 1,
            "once the deadline has expired the sell must be allowed to retry"
        );
    }

    #[test]
    fn pending_nc_bail_is_not_duplicated_within_deadline() {
        // With the default deadline (10s) the in-flight sell is NOT re-emitted no matter how many
        // ticks pass; previously 20 ticks (~2s at 10/s) reset before user_ws.
        let mut nc = NearCertain::new(NearCertainConfig::default(), Decimal::ZERO);
        let mut book = make_book("0.92", "0.08", 60, Some("100000"));
        book.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.85),
                size: dec!(100),
            }],
        };
        let _ = nc.on_book(&book, &make_vol("105000", "0.001"));
        nc.on_fill(&make_fill("yes-token", FillSide::Buy, "0.92", "10"));

        let vol_down = make_vol("95000", "0.001");
        let mut sells = 0;
        for _ in 0..60 {
            sells += nc
                .on_book(&book, &vol_down)
                .iter()
                .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
                .count();
        }
        assert_eq!(
            sells, 1,
            "a single sell while the time deadline has not expired"
        );
    }

    #[test]
    fn nc_buys_accumulate_and_partial_sell_keeps_the_rest() {
        // A taker can fill across several trades and a FAK bail-sell can be
        // partial: the position must accumulate buys and the remainder of the
        // partial sell must stay under management (re-emit the bail for the rest).
        let cfg = NearCertainConfig {
            bail_retry_secs: 0,
            ..NearCertainConfig::default()
        };
        let mut nc = NearCertain::new(cfg, Decimal::ZERO);
        let mut book = make_book("0.92", "0.08", 60, Some("100000"));
        book.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.85),
                size: dec!(100),
            }],
        };
        let _ = nc.on_book(&book, &make_vol("105000", "0.001"));
        // Two trades by the same taker: 10 + 5 = 15 shares.
        nc.on_fill(&make_fill("yes-token", FillSide::Buy, "0.92", "10"));
        nc.on_fill(&make_fill("yes-token", FillSide::Buy, "0.92", "5"));
        // Partial bail: only 4 of 15 filled → 11 remain.
        nc.on_fill(&make_fill("yes-token", FillSide::Sell, "0.85", "4"));

        let vol_down = make_vol("95000", "0.001");
        let mut sells = vec![];
        for _ in 0..5 {
            sells.extend(
                nc.on_book(&book, &vol_down)
                    .into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(sells.len(), 1, "must re-emit the bail for the remainder");
        assert_eq!(sells[0].size, dec!(11), "remainder = 10 + 5 − 4");
    }

    #[test]
    fn nc_enters_with_sufficient_z() {
        let cfg = NearCertainConfig::default();
        let mut nc = NearCertain::new(cfg.clone(), Decimal::ZERO);

        // reference=100k, live=105k → dist_frac=0.05, sigma=0.001/√s, remaining=60s
        // z = 0.05 / (0.001 × √60) ≈ 0.05 / 0.00775 ≈ 6.45 > z_threshold=2.5 → must enter YES
        let book = make_book("0.92", "0.08", 60, Some("100000"));
        let vol = make_vol("105000", "0.001");
        let intents = nc.on_book(&book, &vol);

        // An entry must have been emitted
        assert_eq!(intents.len(), 1);
        assert!(matches!(intents[0].side, OrderSide::Buy));
        assert_eq!(intents[0].token_id, "yes-token");
    }

    #[test]
    fn nc_skip_outside_window() {
        let mut nc = NearCertain::new(NearCertainConfig::default(), Decimal::ZERO);
        // remaining > window_secs (default 600s = 10min)
        let book = make_book("0.92", "0.08", 7200, Some("100000"));
        let vol = make_vol("101000", "0.001");
        assert!(nc.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn nc_skip_insufficient_z() {
        let mut nc = NearCertain::new(NearCertainConfig::default(), Decimal::ZERO);
        // small dist → low z
        let book = make_book("0.92", "0.08", 60, Some("100000"));
        let vol2 = make_vol("100001", "1.0"); // sigma=1 → z = 1/(1*sqrt(60)) ≈ 0.13 < threshold
        assert!(nc.on_book(&book, &vol2).is_empty());
    }

    /// Same fv (~0.905) and same ask (0.90): with the market's real fee
    /// at 1% (`parabolic_fee_rate = Some(0.01)`) the entry DOES pass the
    /// breakeven gate, whereas with the TAKER_FEE_RATE_PARABOLIC fallback
    /// (7%, book without `parabolic_fee_rate`) it is blocked. breakeven(0.90, 0.07) =
    /// 0.90 + 0.07×0.90×0.10 = 0.9063 > fv; breakeven(0.90, 0.01) =
    /// 0.90 + 0.01×0.90×0.10 = 0.9009 < fv. z_threshold is low on purpose (see the
    /// guard comment in on_book): with the default z_threshold=2.5, a sufficient z
    /// implies fv≈1 and the fee gate never gets to block anything.
    #[test]
    fn nc_uses_market_fee() {
        let cfg = NearCertainConfig {
            z_threshold: dec!(0.5),
            ..NearCertainConfig::default()
        };
        let vol = make_vol("101020.34", "0.001");

        let mut high_fee_book = make_book("0.90", "0.10", 60, Some("100000"));
        high_fee_book.parabolic_fee_rate = None; // falls back to TAKER_FEE_RATE_PARABOLIC = 0.07
        let mut nc_blocked = NearCertain::new(cfg.clone(), Decimal::ZERO);
        assert!(
            nc_blocked.on_book(&high_fee_book, &vol).is_empty(),
            "with the 0.07 fallback fv does not cover breakeven, so it must block"
        );

        let mut low_fee_book = high_fee_book.clone();
        low_fee_book.parabolic_fee_rate = Some(dec!(0.01)); // 1% real market fee
        let mut nc = NearCertain::new(cfg, Decimal::ZERO);
        let intents = nc.on_book(&low_fee_book, &vol);
        assert_eq!(
            intents.len(),
            1,
            "with the market's 1% fee fv DOES cover breakeven, so it must enter"
        );
    }

    /// MANDATORY unit test: with `fv_side = breakeven +
    /// 0.03` (nominal edge 3pp < haircut 4pp) NC must NOT emit; with
    /// `fv_side = breakeven + 0.05` (nominal edge 5pp > haircut 4pp) it MUST
    /// emit. The book's `parabolic_fee_rate` (Decimal, a fraction, not
    /// `fee_rate_bps`, a u32 in bps, the source declared deprecated) is
    /// computed so that the parabolic breakeven falls exactly at
    /// `fv_side - 0.03`/`-0.05`.
    #[test]
    fn nc_haircut_blocks_edge_3pp_and_lets_edge_5pp_through() {
        let live = dec!(105000);
        let reference = dec!(100000);
        let sigma = dec!(0.001);
        let remaining = 60i64;
        let fv_side =
            fair_value_yes(live, reference, sigma, remaining).expect("fv must be computable");

        let ask = dec!(0.80);
        let haircut = dec!(0.04);

        // Inverse of fee_taker_parabolic_with_rate: breakeven = ask + rate*ask*(1-ask).
        let rate_for_breakeven =
            |breakeven: Decimal| -> Decimal { (breakeven - ask) / (ask * (Decimal::ONE - ask)) };

        // z_threshold is low on purpose: the gate under test is the fee-edge one,
        // not the z-score (same pattern as nc_uses_market_fee).
        let cfg = NearCertainConfig {
            z_threshold: dec!(0.1),
            ..NearCertainConfig::default()
        };
        let vol = make_vol("105000", "0.001");

        // Case 1: nominal edge 3pp < haircut 4pp → must BLOCK.
        let mut book_edge_3pp = make_book("0.80", "0.20", remaining, Some("100000"));
        book_edge_3pp.parabolic_fee_rate = Some(rate_for_breakeven(fv_side - dec!(0.03)));
        let mut nc_3pp = NearCertain::new(cfg.clone(), haircut);
        assert!(
            nc_3pp.on_book(&book_edge_3pp, &vol).is_empty(),
            "nominal edge 3pp < haircut 4pp — NC must not emit an intent"
        );

        // Case 2: nominal edge 5pp > haircut 4pp → must PASS.
        let mut book_edge_5pp = make_book("0.80", "0.20", remaining, Some("100000"));
        book_edge_5pp.parabolic_fee_rate = Some(rate_for_breakeven(fv_side - dec!(0.05)));
        let mut nc_5pp = NearCertain::new(cfg, haircut);
        assert_eq!(
            nc_5pp.on_book(&book_edge_5pp, &vol).len(),
            1,
            "nominal edge 5pp > haircut 4pp — NC must emit an intent"
        );
    }

    /// With `remaining < NC_MIN_LEAD_SECS` (5s) NC must not enter
    /// even though the other gates (z-score, fee-edge) would pass; without this limit
    /// there is no margin for the bail-sell to act if the price reverts right
    /// after the entry.
    #[test]
    fn nc_skip_remaining_below_min_lead() {
        let mut nc = NearCertain::new(NearCertainConfig::default(), Decimal::ZERO);
        // Same parameters as nc_enters_with_sufficient_z (z≈6.45) but
        // remaining=3s < NC_MIN_LEAD_SECS=5s.
        let book = make_book("0.92", "0.08", 3, Some("100000"));
        let vol = make_vol("105000", "0.001");
        assert!(
            nc.on_book(&book, &vol).is_empty(),
            "remaining below the minimum lead must not enter even if z is sufficient"
        );
    }
}
