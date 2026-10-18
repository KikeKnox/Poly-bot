//! ThetaCapture (TC) strategy.
//!
//! Maker variant of NC: in the last `max_window_secs` before close, when
//! the winning side quotes at `yes_ask >= min_yes_price`, posts a GTD bid at price
//! `yes_ask - bid_discount`. If someone wants to exit before settlement (panic,
//! liquidity), it fills at a better price than NC taker and with no taker fee.
//!
//! If the oracle reverts > bail_sigma × σ√t while the GTD is in flight, it emits
//! a cancel intent to avoid fills in a market moving against it.
//!
//! Optionally (`sell_loser_side = true`): also posts a maker ask on the losing
//! side (NO when YES >= min_yes_price). The premium received is the additional edge
//! if NO resolves to $0.

use crate::time::Instant;
use std::collections::HashMap; // used in last_retain and tests

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    config::defaults,
    fair_value::{fair_value_yes, net_of_fee_sell_parabolic},
    types::{BookSnapshot, Fill, FillSide, OrderIntent, OrderSide, StrategyId, VolSnapshot},
};

use super::{Strategy, StrategyConfig};

// ── Configuration ─────────────────────────────────────────────────────────────

/// ThetaCapture (TC) strategy configuration.
///
/// Maker variant of NC: posts a GTD bid on the winning side below the current ask.
/// When someone wants to exit before settlement (panic/liquidity), it fills at a
/// better price than NC taker, with no taker fee. If the oracle reverts, it cancels the GTD.
/// Optionally sells the losing side as maker.
#[derive(Debug, Clone, Deserialize)]
pub struct ThetaCaptureConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    /// Minimum ask of the winning side to activate. YES >= min_yes_price → we enter YES.
    #[serde(default = "defaults::tc_min_yes_price")]
    pub min_yes_price: Decimal,
    /// Discount over the ask when placing the GTD bid (improves price vs taker).
    #[serde(default = "defaults::tc_bid_discount")]
    pub bid_discount: Decimal,
    /// Activation window: only trade in the last N seconds before close.
    #[serde(default = "defaults::tc_max_window_secs")]
    pub max_window_secs: u64,
    /// Maximum tolerable bid-ask spread. 0 = no filter.
    #[serde(default)]
    pub max_bid_ask_spread: Decimal,
    /// Minimum bid to run a bail-sell. 0 = no minimum.
    #[serde(default = "defaults::nc_min_bail_bid")]
    pub min_bail_bid: Decimal,
    /// Enable selling the losing side as maker (passive ask on NO when YES >= min_yes_price).
    #[serde(default)]
    pub sell_loser_side: bool,
    /// Minimum ask of the losing side for the maker sell. Only applies if sell_loser_side=true.
    #[serde(default = "defaults::tc_min_loser_ask")]
    pub min_loser_ask: Decimal,
    /// σ × √t threshold to cancel an in-flight GTD if the oracle reverts.
    #[serde(default = "defaults::om_bail_sigma")]
    pub bail_sigma: Decimal,
    /// Consecutive ticks with reversal to trigger cancel/bail.
    #[serde(default = "defaults::om_bail_checks")]
    pub bail_checks_required: u32,
    /// Maximum size in shares. 0 = no cap (primary limit max_order_usdc).
    #[serde(default)]
    pub max_size_shares: Decimal,
}

impl Default for ThetaCaptureConfig {
    fn default() -> Self {
        Self {
            limits: crate::config::StrategyLimits::default(),
            min_yes_price: defaults::tc_min_yes_price(),
            bid_discount: defaults::tc_bid_discount(),
            max_window_secs: defaults::tc_max_window_secs(),
            max_bid_ask_spread: Decimal::ZERO,
            min_bail_bid: defaults::nc_min_bail_bid(),
            sell_loser_side: false,
            min_loser_ask: defaults::tc_min_loser_ask(),
            bail_sigma: defaults::om_bail_sigma(),
            bail_checks_required: defaults::om_bail_checks(),
            max_size_shares: Decimal::ZERO,
        }
    }
}

impl StrategyConfig for ThetaCaptureConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

// ── Per-bracket state ────────────────────────────────────────────────────────

struct BracketState {
    clear_at: i64,
    // Chosen winning side (true=YES).
    is_yes: bool,
    // Reference price of the underlying when the bid was placed.
    reference: Decimal,
    // GTD bid in flight.
    have_bid: bool,
    bid_token: String,
    // Fill confirmed → position open.
    have_position: bool,
    position_size: Decimal,
    // Bail/cancel state.
    // Per-tick on purpose: bail_checks, the failure mode under coalescing is
    // a later bail, not a duplicated one; switch to time if calibration shows it.
    bail_checks: u32,
    cancel_pending: bool,
    /// Instant of the in-flight bail-sell (time deadline; the
    /// 20-tick counter caused 32 SELLs for 1 BUY in a calibration run).
    bail_sent_at: Option<Instant>,
    // Losing side (sell_loser_side).
    loser_token: String,
    loser_size: Decimal,
}

/// Seconds without a bail-sell fill before treating it as rejected (> the
/// async user_ws latency; same value as NC/NE bail_retry_secs).
const BAIL_RETRY_SECS: u64 = 10;

// ── ThetaCapture ──────────────────────────────────────────────────────────────

pub struct ThetaCapture {
    cfg: ThetaCaptureConfig,
    /// Additive haircut over `fv_side` in the bail-sell gate.
    /// It is a SELL gate: sign INVERTED relative to buy gates:
    /// it is ADDED (favors holding over selling, which is the
    /// "don't trade" direction for this strategy).
    haircut: Decimal,
    states: HashMap<String, BracketState>,
    last_retain: Instant,
}

impl ThetaCapture {
    pub fn new(cfg: ThetaCaptureConfig, haircut: Decimal) -> Self {
        Self {
            cfg,
            haircut,
            states: HashMap::new(),
            last_retain: Instant::now(),
        }
    }
}

impl Strategy for ThetaCapture {
    fn id(&self) -> StrategyId {
        StrategyId::ThetaCapture
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        let cid = &book.market_condition_id;
        let remaining = book.secs_to_clear();

        if self.last_retain.elapsed().as_secs() >= 30 {
            self.states.retain(|_, s| now < s.clear_at);
            self.last_retain = Instant::now();
        }

        // Outside the trading window.
        if remaining <= 0 {
            return vec![];
        }

        // ── Existing state ──────────────────────────────────────────────────

        if let Some(state) = self.states.get_mut(cid.as_str()) {
            // If we leave the window and a GTD is in flight, cancel to avoid
            // post-window fills (they should not happen, this is a safeguard).
            if remaining > self.cfg.max_window_secs as i64 {
                if state.have_bid && !state.cancel_pending && !state.have_position {
                    state.cancel_pending = true;
                    metrics::counter!("hrf_tc_cancel_total", "reason" => "out_of_window")
                        .increment(1);
                    return vec![cancel_intent(book, &state.bid_token)];
                }
                return vec![];
            }

            // ── Bail-sell for an open position ────────────────────────────────

            if state.have_position {
                if let Some(sent_at) = state.bail_sent_at {
                    // Time-based deadline. The previous 20 ticks (~2s at 10
                    // ticks/s) were less than the user_ws latency, so the reset
                    // arrived before the fill and the sell was re-emitted over and over
                    // (32 SELLs for 1 BUY in a calibration run).
                    if sent_at.elapsed().as_secs() >= BAIL_RETRY_SECS {
                        state.bail_sent_at = None;
                        state.bail_checks = 0;
                        tracing::warn!(cid = %cid, secs = BAIL_RETRY_SECS,
                            "TC: bail-sell without fill after the deadline; reset");
                    }
                    return vec![];
                }

                let live = vol.last_price;
                if live > Decimal::ZERO
                    && !vol.is_stale_for_signal()
                    && state.reference > Decimal::ZERO
                {
                    let cushion = if state.is_yes {
                        (live - state.reference) / state.reference
                    } else {
                        (state.reference - live) / state.reference
                    };

                    let sigma = vol.sigma_long.or(vol.sigma_short).unwrap_or(dec!(0.001));
                    let sqrt_t = Decimal::from_f64_retain((remaining.max(1) as f64).sqrt())
                        .unwrap_or(dec!(1));
                    let threshold = -(self.cfg.bail_sigma * sigma * sqrt_t);

                    if cushion < threshold {
                        state.bail_checks += 1;
                    } else {
                        state.bail_checks = 0;
                    }

                    if state.bail_checks >= self.cfg.bail_checks_required {
                        let bid = if state.is_yes {
                            book.yes.best_bid().unwrap_or(dec!(0.01))
                        } else {
                            book.no.best_bid().unwrap_or(dec!(0.01))
                        };

                        if bid < self.cfg.min_bail_bid {
                            tracing::warn!(cid = %cid, bid = %bid, "TC: bail cancelled — bid low, hold-to-expiry");
                            state.bail_checks = 0;
                        } else {
                            // cushion+bail_checks is a cheap trigger (it only
                            // looks at the oracle) but does not know whether the BOOK already reacted. The bail
                            // is only profitable if the bid (sometimes stale-high, not yet repriced
                            // after the oracle move) exceeds the real
                            // probability of winning (fv_side). Selling below that probability
                            // gives away the EV that holding to settlement would recover.
                            let (should_sell, fv_side_dbg) = match fair_value_yes(
                                live,
                                state.reference,
                                sigma,
                                remaining,
                            ) {
                                Some(fv) => {
                                    let fv_side = if state.is_yes { fv } else { Decimal::ONE - fv };
                                    // The triangular model + fee_rate_bps (deprecated)
                                    // was replaced by the verified parabolic one with the market's
                                    // real rate. Direction is DANGEROUS here, not conservative:
                                    // the real fee is lower, ev_sell goes up, and the bail gate
                                    // (ev_sell > fv_side + haircut) becomes MORE sensitive; previously
                                    // the stale model delayed/avoided bails that should have fired.
                                    let ev_sell = net_of_fee_sell_parabolic(book, bid);
                                    // SELL gate: sign INVERTED. The
                                    // haircut ALWAYS moves the decision towards not trading; here
                                    // "trading" is selling, so it is ADDED to fv_side (subtracting
                                    // would be the error: it would make it sell MORE).
                                    let should_sell = ev_sell > fv_side + self.haircut;
                                    if !should_sell && ev_sell > fv_side {
                                        metrics::counter!("hrf_fv_haircut_block_total", "strategy" => "theta_capture").increment(1);
                                    }
                                    (should_sell, Some(fv_side))
                                }
                                // Without a model (reference/sigma absent or vol stale) there is nothing
                                // to compare the EV against: keep the previous behavior (sell).
                                None => (true, None),
                            };

                            if should_sell {
                                tracing::info!(cid = %cid, cushion = %cushion, "TC: bail-sell, oracle reverted");
                                state.bail_sent_at = Some(Instant::now());
                                state.bail_checks = 0;
                                metrics::counter!("hrf_tc_bail_total").increment(1);
                                return vec![sell_intent(
                                    book,
                                    &state.bid_token,
                                    bid,
                                    state.position_size,
                                )];
                            } else {
                                state.bail_checks = 0;
                                metrics::counter!("hrf_tc_bail_hold_total").increment(1);
                                tracing::info!(cid = %cid, bid = %bid, fv_side = ?fv_side_dbg,
                                    "TC: hold — sell no bate FV");
                            }
                        }
                    }
                }
                return vec![];
            }

            // ── In-flight GTD: check for reversal to cancel ───────────────────

            if state.have_bid && !state.cancel_pending {
                let live = vol.last_price;
                if live > Decimal::ZERO
                    && !vol.is_stale_for_signal()
                    && state.reference > Decimal::ZERO
                {
                    let cushion = if state.is_yes {
                        (live - state.reference) / state.reference
                    } else {
                        (state.reference - live) / state.reference
                    };
                    let sigma = vol.sigma_long.or(vol.sigma_short).unwrap_or(dec!(0.001));
                    let sqrt_t = Decimal::from_f64_retain((remaining.max(1) as f64).sqrt())
                        .unwrap_or(dec!(1));
                    let threshold = -(self.cfg.bail_sigma * sigma * sqrt_t);

                    if cushion < threshold {
                        state.bail_checks += 1;
                    } else {
                        state.bail_checks = 0;
                    }

                    if state.bail_checks >= self.cfg.bail_checks_required {
                        state.cancel_pending = true;
                        state.bail_checks = 0;
                        tracing::info!(cid = %cid, cushion = %cushion, "TC: cancelling GTD, oracle reverted");
                        metrics::counter!("hrf_tc_cancel_total", "reason" => "oracle_reversal")
                            .increment(1);
                        return vec![cancel_intent(book, &state.bid_token)];
                    }
                }
                return vec![];
            }

            // GTD cancelled or already processed; do not re-enter in this bracket.
            return vec![];
        }

        // ── No prior state: check the entry signal ─────────────────────────

        // Only trade in the window of the last max_window_secs.
        if remaining > self.cfg.max_window_secs as i64 {
            return vec![];
        }

        // Oracle fresh.
        if vol.is_stale_for_signal() {
            return vec![];
        }
        let live = vol.last_price;
        if live <= Decimal::ZERO {
            return vec![];
        }

        // Winning side: the one whose ask >= min_yes_price.
        let (is_yes, ask, bid_tok, loser_ask_opt, loser_tok) = {
            let yes_ask = book.yes.best_ask();
            let na = book.no.best_ask();
            if yes_ask.is_some_and(|a| a >= self.cfg.min_yes_price) {
                let la = na; // loser = NO
                (
                    true,
                    yes_ask.unwrap(),
                    book.yes.token_id.clone(),
                    la,
                    book.no.token_id.clone(),
                )
            } else if na.is_some_and(|a| a >= self.cfg.min_yes_price) {
                let la = yes_ask; // loser = YES
                (
                    false,
                    na.unwrap(),
                    book.no.token_id.clone(),
                    la,
                    book.yes.token_id.clone(),
                )
            } else {
                return vec![];
            }
        };

        // Filtro de spread.
        if self.cfg.max_bid_ask_spread > Decimal::ZERO {
            let bid = if is_yes {
                book.yes.best_bid()
            } else {
                book.no.best_bid()
            };
            if let Some(b) = bid {
                if ask - b > self.cfg.max_bid_ask_spread {
                    return vec![];
                }
            }
        }

        // GTD bid price: below the ask to be a maker.
        let bid_price = ask - self.cfg.bid_discount;
        if bid_price <= Decimal::ZERO {
            return vec![];
        }

        // Sizing.
        let size = {
            let nat = super::size_from_caps(
                self.cfg.limits.max_order_usdc,
                self.cfg.max_size_shares,
                bid_price,
                None,
                Some(dec!(10)),
            );
            nat.trunc_with_scale(0)
        };
        if size < Decimal::ONE || size * bid_price < dec!(1) {
            return vec![];
        }

        // The reference must be the bracket's OPENING price. Using `live`
        // as a fallback left cushion ≈ 0 permanently → the oracle cancel/bail
        // never fired. TC trades in the final window (bracket already stale), so
        // the opening fallback does not apply: without a real reference_price, do not enter.
        let reference = match book.reference_price {
            Some(r) if r > Decimal::ZERO => r,
            _ => {
                tracing::debug!(cid = %cid, "TC: no reference_price from the feed, skip (a bail without a reference would be inoperative)");
                return vec![];
            }
        };

        // Insert state before emitting to avoid a double entry.
        self.states.insert(
            cid.clone(),
            BracketState {
                clear_at: book.clear_at,
                is_yes,
                reference,
                have_bid: true,
                bid_token: bid_tok.clone(),
                have_position: false,
                position_size: Decimal::ZERO,
                bail_checks: 0,
                cancel_pending: false,
                bail_sent_at: None,
                loser_token: loser_tok.clone(),
                loser_size: Decimal::ZERO,
            },
        );

        let mut intents = vec![bid_intent(book, &bid_tok, bid_price, size)];

        // Optional: sell the losing side as maker.
        if self.cfg.sell_loser_side {
            if let Some(la) = loser_ask_opt {
                if la >= self.cfg.min_loser_ask {
                    // Emit as maker ask (resting sell).
                    intents.push(ask_maker_intent(book, &loser_tok, la, size));
                    metrics::counter!("hrf_tc_loser_placed_total").increment(1);
                }
            }
        }

        metrics::counter!("hrf_tc_bid_placed_total").increment(1);
        tracing::info!(
            cid = %cid, is_yes, ask = %ask, bid_price = %bid_price, remaining,
            "TC: GTD bid placed"
        );

        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::ThetaCapture {
            return;
        }
        // Buy fill without state (restart mid-bracket): without a reference no
        // bail is possible; the orphan checker will liquidate at settlement. Make it visible.
        if !self.states.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "TC: Buy fill without state, position left to the orphan checker (restart mid-bracket)");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "theta_capture").increment(1);
            }
            return;
        }
        if let Some(state) = self.states.get_mut(&fill.market_cid) {
            match fill.side {
                FillSide::Buy => {
                    if fill.token_id == state.bid_token {
                        state.have_position = true;
                        state.position_size += fill.size;
                        state.have_bid = false;
                        state.bail_checks = 0;
                        metrics::counter!("hrf_tc_bid_fills_total").increment(1);
                        tracing::info!(
                            cid = %fill.market_cid, price = %fill.price, size = %fill.size,
                            "TC: GTD bid filled → position open"
                        );
                    } else if fill.token_id == state.loser_token {
                        // Fill on the losing side's bid (if the logic was inverted): ignore.
                    }
                }
                FillSide::Sell => {
                    if fill.token_id == state.bid_token {
                        // Winning-side bail-sell. Partial sell (FAK capped to
                        // depth): the rest stays under management.
                        state.position_size = (state.position_size - fill.size).max(Decimal::ZERO);
                        state.bail_sent_at = None;
                        if state.position_size == Decimal::ZERO {
                            state.have_position = false;
                        }
                    } else if fill.token_id == state.loser_token {
                        // Fill of the maker ask on the losing side.
                        state.loser_size += fill.size;
                        metrics::counter!("hrf_tc_loser_sold_total").increment(1);
                        tracing::info!(
                            cid = %fill.market_cid, price = %fill.price, size = %fill.size,
                            "TC: theta collected from the losing side"
                        );
                    }
                }
                // `Cancelled` is the REAL ack from the router; `Rejected` still
                // arrives via the FOK. Same effect on the bid state.
                FillSide::Rejected | FillSide::Cancelled => {
                    // GTD cancelled or rejected.
                    if fill.token_id == state.bid_token {
                        state.have_bid = false;
                        state.cancel_pending = false;
                    }
                }
                // The intent was blocked by a pre-submit gate, so the
                // order never existed. If the bid marked by THIS emission dies,
                // `have_bid` must be released or the bracket goes silent. If an
                // earlier order were still alive (blocked requote), the executor's
                // LIVE-8 anti-dup reconciles when reposting.
                FillSide::GateRejected => {
                    if fill.token_id == state.bid_token && !state.have_position {
                        state.have_bid = false;
                        state.cancel_pending = false;
                    }
                }
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn bid_intent(book: &BookSnapshot, token_id: &str, price: Decimal, size: Decimal) -> OrderIntent {
    OrderIntent {
        market_cid: book.market_condition_id.clone(),
        token_id: token_id.to_string(),
        side: OrderSide::Buy,
        price,
        size,
        clear_at: book.clear_at,
        signal: price,
        neg_risk: book.neg_risk,
        limit_price: Some(price),
        second_leg: None,
        strategy: StrategyId::ThetaCapture,
        resting: true,
        real_clear_at: None,
    }
}

fn ask_maker_intent(
    book: &BookSnapshot,
    token_id: &str,
    price: Decimal,
    size: Decimal,
) -> OrderIntent {
    OrderIntent {
        market_cid: book.market_condition_id.clone(),
        token_id: token_id.to_string(),
        side: OrderSide::Sell { size: Some(size) },
        price,
        size,
        clear_at: book.clear_at,
        signal: price,
        neg_risk: book.neg_risk,
        limit_price: Some(price),
        second_leg: None,
        strategy: StrategyId::ThetaCapture,
        resting: true,
        real_clear_at: None,
    }
}

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
        strategy: StrategyId::ThetaCapture,
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
        StrategyId::ThetaCapture,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, Mode, TokenBook};

    fn make_book(yes_ask: &str, no_ask: &str, remaining: i64) -> BookSnapshot {
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
            market_condition_id: "tc-cid".into(),
            yes: TokenBook {
                token_id: "yes-tok".into(),
                bids: bids("0.92"),
                asks: asks(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-tok".into(),
                bids: bids("0.05"),
                asks: asks(no_ask),
                ..Default::default()
            },
            market_open_ts: now - 200,
            clear_at: now + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-tc".into(),
            reference_price: Some(dec!(100000)),
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
            ..VolSnapshot::default()
        }
    }

    fn test_cfg() -> ThetaCaptureConfig {
        ThetaCaptureConfig {
            limits: crate::config::StrategyLimits {
                enabled: true,
                max_orders_per_hour: 20,
                max_order_usdc: dec!(10),
                daily_loss_usdc_stop: dec!(20),
                ..crate::config::StrategyLimits::default()
            },
            ..ThetaCaptureConfig::default()
        }
    }

    #[test]
    fn tc_places_gtd_bid_in_window() {
        let mut tc = ThetaCapture::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.95", "0.07", 60);
        let vol = make_vol("105000");
        let intents = tc.on_book(&book, &vol);
        assert_eq!(intents.len(), 1);
        assert!(matches!(intents[0].side, OrderSide::Buy));
        assert!(intents[0].resting);
        // bid = 0.95 - 0.015 = 0.935
        assert_eq!(intents[0].price, dec!(0.935));
        assert_eq!(intents[0].token_id, "yes-tok");
    }

    #[test]
    fn tc_skip_outside_window() {
        let mut tc = ThetaCapture::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.95", "0.07", 200); // remaining > max_window_secs=90
        let vol = make_vol("105000");
        assert!(tc.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn tc_skip_without_winning_side() {
        let mut tc = ThetaCapture::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.80", "0.25", 60); // none >= 0.93
        let vol = make_vol("100000");
        assert!(tc.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn tc_does_not_replace_after_gtd() {
        let mut tc = ThetaCapture::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.95", "0.07", 60);
        let vol = make_vol("105000");
        let i1 = tc.on_book(&book, &vol);
        assert_eq!(i1.len(), 1);
        let i2 = tc.on_book(&book, &vol);
        assert!(i2.is_empty(), "must not re-place an in-flight GTD");
    }

    #[test]
    fn tc_cancels_gtd_on_oracle_reversion() {
        let mut tc = ThetaCapture::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.95", "0.07", 60);
        let vol_ok = make_vol("105000");
        tc.on_book(&book, &vol_ok);

        // Oracle drops 10% → very negative cushion → must cancel after bail_checks_required ticks
        let vol_bad = make_vol("90000");
        let mut cancels = vec![];
        for _ in 0..5 {
            let intents = tc.on_book(&book, &vol_bad);
            cancels.extend(
                intents
                    .into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Cancel)),
            );
        }
        assert_eq!(cancels.len(), 1, "must emit exactly one Cancel");
    }

    /// Strong oracle reversal with the book STILL stale-high
    /// (bid = 0.92, not repriced). ev_sell (bid − fee) far exceeds
    /// fv_side (~0, given the collapse) → the bail must sell.
    #[test]
    fn tc_bail_sells_if_sell_exceeds_fv() {
        let mut tc = ThetaCapture::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.95", "0.07", 60);
        let vol_ok = make_vol("105000");
        tc.on_book(&book, &vol_ok);

        // Confirm the buy fill.
        tc.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "tc-cid".into(),
            token_id: "yes-tok".into(),
            side: FillSide::Buy,
            price: dec!(0.935),
            size: dec!(10),
            order_id: None,
            ts: crate::time::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ThetaCapture,
            clear_at: 0,
            outcome_label: None,
        });

        // Oracle reverts strongly; YES book still has a stale-high bid (0.92).
        let vol_bad = make_vol("85000");
        let mut sells = vec![];
        for _ in 0..5 {
            let intents = tc.on_book(&book, &vol_bad);
            sells.extend(
                intents
                    .into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            1,
            "ev_sell (stale-high bid) > fv_side (~0) → must sell"
        );
    }

    /// Reversal that fires the cushion trigger, but with the
    /// book ALREADY repriced to a very low bid (consistent with the collapse). ev_sell
    /// (bid − fee) falls below fv_side (residual probability of winning
    /// per the model) → selling would give away EV; it must keep the position.
    #[test]
    fn tc_bail_hold_if_fv_exceeds_sell() {
        let mut cfg = test_cfg();
        cfg.min_bail_bid = Decimal::ZERO; // no minimum: isolate the EV gate
        let mut tc = ThetaCapture::new(cfg, Decimal::ZERO);

        let book = make_book("0.95", "0.07", 60);
        let vol_ok = make_vol("105000");
        tc.on_book(&book, &vol_ok);

        tc.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "tc-cid".into(),
            token_id: "yes-tok".into(),
            side: FillSide::Buy,
            price: dec!(0.935),
            size: dec!(10),
            order_id: None,
            ts: crate::time::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ThetaCapture,
            clear_at: 0,
            outcome_label: None,
        });

        // YES book with an already-repriced very low bid (0.002), consistent with the
        // oracle reversal (cushion = -2%, just past the threshold of
        // bail_sigma=2.5 with sigma=0.001 and t=60s). fv_side ≈ Phi(-2.6) ≈ 0.0046
        // exceeds ev_sell ≈ 0.0018 (0.002 − fee) → hold.
        let mut book_low_bid = book.clone();
        book_low_bid.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.002),
                size: dec!(100),
            }],
        };

        let vol_bad = make_vol("98000");
        let mut sells = vec![];
        for _ in 0..5 {
            let intents = tc.on_book(&book_low_bid, &vol_bad);
            sells.extend(
                intents
                    .into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            0,
            "ev_sell < fv_side → must not sell (hold-to-settlement)"
        );
    }

    /// MANDATORY unit test (symmetric to NC's): with
    /// `ev_sell = fv_side + 0.03` (nominal edge 3pp < haircut 4pp) TC must
    /// keep the position (NOT sell); with `ev_sell = fv_side + 0.05` (nominal edge
    /// 5pp > haircut) it MUST sell. Sign INVERTED relative to a buy
    /// gate: here "trading" is selling, so the haircut works in
    /// favor of holding (it is ADDED to fv_side, not subtracted).
    ///
    /// `rate_for_ev` uses the inverse of the PARABOLIC model
    /// (`ev_sell = bid - rate*bid*(1-bid)`), not the earlier triangular one, and sets
    /// `parabolic_fee_rate` (Decimal, a fraction) instead of `fee_rate_bps` (u32,
    /// bps), the source declared deprecated.
    #[test]
    fn tc_haircut_blocks_sell_edge_3pp_and_lets_edge_5pp_through() {
        let mut cfg = test_cfg();
        cfg.min_bail_bid = Decimal::ZERO; // isolate the EV gate
        let haircut = dec!(0.04);

        let live = dec!(98000);
        let reference = dec!(100000);
        let sigma = dec!(0.001);
        let remaining = 60i64;
        let fv_side =
            fair_value_yes(live, reference, sigma, remaining).expect("fv must be computable");

        let bid = dec!(0.5);
        // Inverse of fee_taker_parabolic_with_rate: ev_sell = bid - rate*bid*(1-bid).
        let rate_for_ev =
            |target: Decimal| -> Decimal { (bid - target) / (bid * (Decimal::ONE - bid)) };

        let enter_and_confirm = |tc: &mut ThetaCapture| {
            let book = make_book("0.95", "0.07", 60);
            tc.on_book(&book, &make_vol("105000"));
            tc.on_fill(&Fill {
                hedge_id: None,
                market_cid: "tc-cid".into(),
                token_id: "yes-tok".into(),
                exec_id: None,
                local_order_id: None,
                side: FillSide::Buy,
                price: dec!(0.935),
                size: dec!(10),
                order_id: None,
                ts: crate::time::now(),
                mode: Mode::Paper,
                neg_risk: false,
                strategy: StrategyId::ThetaCapture,
                clear_at: 0,
                outcome_label: None,
            });
        };

        // Case 1: nominal edge 3pp < haircut 4pp → must keep the position.
        {
            let mut tc = ThetaCapture::new(cfg.clone(), haircut);
            enter_and_confirm(&mut tc);

            let mut book_bid = make_book("0.95", "0.07", 60);
            book_bid.yes.bids = HalfBook {
                levels: vec![Level {
                    price: bid,
                    size: dec!(100),
                }],
            };
            book_bid.parabolic_fee_rate = Some(rate_for_ev(fv_side + dec!(0.03)));

            let vol_bad = make_vol("98000");
            let mut sells = 0;
            for _ in 0..5 {
                sells += tc
                    .on_book(&book_bid, &vol_bad)
                    .into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
                    .count();
            }
            assert_eq!(
                sells, 0,
                "nominal edge 3pp < haircut 4pp — TC must not sell"
            );
        }

        // Case 2: nominal edge 5pp > haircut 4pp → must sell.
        {
            let mut tc = ThetaCapture::new(cfg, haircut);
            enter_and_confirm(&mut tc);

            let mut book_bid = make_book("0.95", "0.07", 60);
            book_bid.yes.bids = HalfBook {
                levels: vec![Level {
                    price: bid,
                    size: dec!(100),
                }],
            };
            book_bid.parabolic_fee_rate = Some(rate_for_ev(fv_side + dec!(0.05)));

            let vol_bad = make_vol("98000");
            let mut sells = 0;
            for _ in 0..5 {
                sells += tc
                    .on_book(&book_bid, &vol_bad)
                    .into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
                    .count();
            }
            assert_eq!(sells, 1, "nominal edge 5pp > haircut 4pp — TC must sell");
        }
    }
}
