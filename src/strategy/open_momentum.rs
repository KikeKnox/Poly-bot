//! OpenMomentum (OM) strategy.
//!
//! Enters in the first `max_open_secs` of a freshly opened bracket when
//! the oracle already shows a skewed fair value (|fv − 0.5| > min_confidence).
//!
//! Initial market makers post quotes near 0.51/0.51; if spot is already
//! far from the strike, the real FV differs from the initial ask → an edge the
//! market corrects within seconds. OM captures that first tick.
//!
//! No box, no completing leg B. Pure directional bet, held to settlement.
//! Bail-sell only if the oracle reverts strongly (> bail_sigma × σ√t) for
//! bail_checks_required consecutive ticks.

use std::collections::HashMap;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    config::{defaults, StrategyLimits},
    fair_value::{breakeven_price_parabolic, fair_value_yes},
    types::{BookSnapshot, Fill, FillSide, OrderIntent, OrderSide, StrategyId, VolSnapshot},
};

use super::{Strategy, StrategyConfig};

// ── Configuration ─────────────────────────────────────────────────────────────

/// OpenMomentum strategy configuration.
///
/// Enters on the first viable tick of a freshly opened bracket when the oracle
/// already shows a skewed FV. Pure directional bet held to settlement.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenMomentumConfig {
    #[serde(flatten)]
    pub limits: StrategyLimits,
    /// Seconds since the bracket opened during which entry is allowed.
    #[serde(default = "defaults::om_max_open_secs")]
    pub max_open_secs: u64,
    /// Minimum |fv − 0.5| threshold to choose a side.
    #[serde(default = "defaults::om_min_confidence")]
    pub min_confidence: Decimal,
    /// Maximum ask of the chosen leg. If the market already corrected, skip.
    #[serde(default = "defaults::om_max_entry_price")]
    pub max_entry_price: Decimal,
    /// Minimum remaining time to enter. Do not enter if the bracket is almost closed.
    #[serde(default = "defaults::om_min_remaining")]
    pub min_remaining: u64,
    /// Bail-sell threshold: σ multiplied by √t against the chosen side.
    #[serde(default = "defaults::om_bail_sigma")]
    pub bail_sigma: Decimal,
    /// Consecutive ticks with cushion < 0 to trigger a bail-sell.
    #[serde(default = "defaults::om_bail_checks")]
    pub bail_checks_required: u32,
    /// Minimum leg bid to run a bail-sell. 0 = no minimum.
    #[serde(default = "defaults::nc_min_bail_bid")]
    pub min_bail_bid: Decimal,
    /// Maximum size in shares. 0 = no cap; primary limit is max_order_usdc.
    #[serde(default)]
    pub max_size_shares: Decimal,
    /// Maximum tolerable bid-ask spread on entry. 0 = no filter.
    #[serde(default)]
    pub max_bid_ask_spread: Decimal,
}

impl Default for OpenMomentumConfig {
    fn default() -> Self {
        Self {
            limits: StrategyLimits::default(),
            max_open_secs: defaults::om_max_open_secs(),
            min_confidence: defaults::om_min_confidence(),
            max_entry_price: defaults::om_max_entry_price(),
            min_remaining: defaults::om_min_remaining(),
            bail_sigma: defaults::om_bail_sigma(),
            bail_checks_required: defaults::om_bail_checks(),
            min_bail_bid: defaults::nc_min_bail_bid(),
            max_size_shares: Decimal::ZERO,
            max_bid_ask_spread: Decimal::ZERO,
        }
    }
}

impl StrategyConfig for OpenMomentumConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

// ── Per-bracket state ────────────────────────────────────────────────────────

struct BracketState {
    clear_at: i64,
    have_position: bool,
    token_id: String,
    size: Decimal,
    /// Per-tick on purpose: the failure mode under coalescing is a later
    /// bail, not a duplicated one; switch to time if calibration shows it is slow.
    bail_checks: u32,
    /// Instant of the in-flight bail-sell (time deadline, previously 20 ticks).
    bail_sent_at: Option<crate::time::Instant>,
    /// Once the bail is confirmed, do not re-enter even if the signal returns.
    bail_confirmed: bool,
    /// Reference price of the underlying when the entry was emitted.
    reference: Decimal,
}

/// Seconds without a bail-sell fill before treating it as rejected (> the
/// async user_ws latency; same value as NC/NE bail_retry_secs).
const BAIL_RETRY_SECS: u64 = 10;

// ── OpenMomentum ──────────────────────────────────────────────────────────────

pub struct OpenMomentum {
    cfg: OpenMomentumConfig,
    /// Additive haircut over `fv_side` in the net-EV gate
    /// (`fair_value.confidence_haircut`, shared across strategies).
    haircut: Decimal,
    states: HashMap<String, BracketState>,
    last_retain: crate::time::Instant,
}

impl OpenMomentum {
    pub fn new(cfg: OpenMomentumConfig, haircut: Decimal) -> Self {
        Self {
            cfg,
            haircut,
            states: HashMap::new(),
            last_retain: crate::time::Instant::now(),
        }
    }
}

impl Strategy for OpenMomentum {
    fn id(&self) -> StrategyId {
        StrategyId::OpenMomentum
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        let cid = &book.market_condition_id;
        let remaining = book.secs_to_clear();
        if remaining <= 0 {
            return vec![];
        }

        // Clean up expired states every 30s.
        if self.last_retain.elapsed() >= std::time::Duration::from_secs(30) {
            self.states.retain(|_, s| now < s.clear_at);
            self.last_retain = crate::time::Instant::now();
        }

        // ── Existing state management ────────────────────────────────────────

        if let Some(state) = self.states.get_mut(cid.as_str()) {
            if state.bail_confirmed {
                return vec![];
            }

            if !state.have_position {
                // FOK in flight or rejected — do not retry (OM: a single chance).
                return vec![];
            }

            // ── Bail-sell ─────────────────────────────────────────────────────

            if let Some(sent_at) = state.bail_sent_at {
                // Time deadline — previously 20 coalesced ticks (< user_ws latency
                // under load → risk of a duplicate sell).
                if sent_at.elapsed().as_secs() >= BAIL_RETRY_SECS {
                    state.bail_sent_at = None;
                    state.bail_checks = 0;
                    tracing::warn!(cid = %cid, secs = BAIL_RETRY_SECS,
                        "OM: bail-sell without fill after the deadline; reset");
                }
                return vec![];
            }

            let live = vol.last_price;
            if live > Decimal::ZERO && !vol.is_stale_for_signal() && state.reference > Decimal::ZERO
            {
                let bought_yes = state.token_id == book.yes.token_id;
                // Relative pullback of the underlying against the chosen side.
                let cushion = if bought_yes {
                    (live - state.reference) / state.reference
                } else {
                    (state.reference - live) / state.reference
                };

                // Bail when the pullback exceeds bail_sigma × σ√t.
                let sigma = vol.sigma_long.or(vol.sigma_short).unwrap_or(dec!(0.001));
                let sqrt_t =
                    Decimal::from_f64_retain((remaining.max(1) as f64).sqrt()).unwrap_or(dec!(1));
                let bail_threshold = -(self.cfg.bail_sigma * sigma * sqrt_t);

                if cushion < bail_threshold {
                    state.bail_checks += 1;
                } else {
                    state.bail_checks = 0;
                }

                if state.bail_checks >= self.cfg.bail_checks_required {
                    let bid = if bought_yes {
                        book.yes.best_bid().unwrap_or(dec!(0.01))
                    } else {
                        book.no.best_bid().unwrap_or(dec!(0.01))
                    };

                    if bid < self.cfg.min_bail_bid {
                        tracing::warn!(
                            cid = %cid, bid = %bid, min = %self.cfg.min_bail_bid,
                            "OM: bail cancelled — bid low, hold-to-expiry"
                        );
                        state.bail_checks = 0;
                    } else {
                        tracing::info!(
                            cid = %cid, cushion = %cushion, threshold = %bail_threshold,
                            "OM: bail-sell, oracle reverted"
                        );
                        state.bail_sent_at = Some(crate::time::Instant::now());
                        state.bail_checks = 0;
                        metrics::counter!("hrf_om_bail_total").increment(1);
                        return vec![sell_intent(book, &state.token_id, bid, state.size)];
                    }
                }
            }

            return vec![];
        }

        // ── Entry signal: bracket with no prior state ──────────────────────

        // Only enter in the first max_open_secs of the bracket's life.
        // market_open_ts = the API's start_date ≈ 24h before close, useless here.
        // The slug epoch (btc-updown-5m-<epoch>) is the real opening time.
        let open_ts = book
            .slug
            .rsplit('-')
            .next()
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|&t| t > 0)
            .unwrap_or(book.market_open_ts);
        let age_secs = now.saturating_sub(open_ts);
        if age_secs > self.cfg.max_open_secs as i64 {
            metrics::counter!("hrf_om_skip_age_total").increment(1);
            return vec![];
        }

        // Minimum remaining time: do not enter if the bracket is almost closed.
        if remaining < self.cfg.min_remaining as i64 {
            return vec![];
        }

        // Fresh oracle and valid reference.
        if vol.is_stale_for_signal() {
            return vec![];
        }
        let live = vol.last_price;
        if live <= Decimal::ZERO {
            return vec![];
        }

        let reference = match book.reference_price.or_else(|| {
            if live > Decimal::ZERO {
                Some(live)
            } else {
                None
            }
        }) {
            Some(r) if r > Decimal::ZERO => r,
            _ => return vec![],
        };

        let sigma = match vol.sigma_long.or(vol.sigma_short) {
            Some(s) if s > Decimal::ZERO => s,
            _ => return vec![],
        };

        let fv = match fair_value_yes(live, reference, sigma, remaining) {
            Some(v) => v,
            None => return vec![],
        };

        let half = dec!(0.5);
        let is_yes = if fv > half + self.cfg.min_confidence {
            true
        } else if fv < half - self.cfg.min_confidence {
            false
        } else {
            metrics::counter!("hrf_om_skip_confidence_total").increment(1);
            return vec![];
        };

        let (ask, token_id) = if is_yes {
            match book.yes.best_ask() {
                Some(a) => (a, book.yes.token_id.clone()),
                None => return vec![],
            }
        } else {
            match book.no.best_ask() {
                Some(a) => (a, book.no.token_id.clone()),
                None => return vec![],
            }
        };

        // If the market already corrected (high ask), the edge is gone.
        if ask > self.cfg.max_entry_price {
            metrics::counter!("hrf_om_skip_price_total").increment(1);
            return vec![];
        }

        // The bought side's FV must cover the ask PLUS the PARABOLIC
        // taker fee, using the market's real rate
        // (`book.parabolic_fee_rate`) if discovery provided it, otherwise the
        // constant TAKER_FEE_RATE_PARABOLIC. Previously the triangular model
        // with `fee_rate_bps` (deprecated source) was used, which overestimates
        // the fee and discards genuinely +EV entries. Before that the gate was
        // only |fv−0.5| > min_confidence: it could buy with a positive
        // nominal edge but negative net of the fee.
        let fv_side = if is_yes { fv } else { dec!(1) - fv };
        let breakeven = breakeven_price_parabolic(book, ask);
        // Additive haircut: ALWAYS subtracted in a BUY gate
        // (moves the decision towards not trading). Sign trap: do not add.
        if fv_side - self.haircut <= breakeven {
            if fv_side > breakeven {
                metrics::counter!("hrf_fv_haircut_block_total", "strategy" => "open_momentum")
                    .increment(1);
            }
            metrics::counter!("hrf_om_skip_fee_edge_total").increment(1);
            tracing::debug!(cid = %cid, fv = %fv_side, ask = %ask, breakeven = %breakeven, haircut = %self.haircut,
                "OM: skip, edge does not cover the taker fee");
            return vec![];
        }

        // Spread filter. A missing bid (`None`) must fail CLOSED
        // (reject the entry), same as directional.rs; previously a book
        // with no bid let the entry through without applying the filter.
        if self.cfg.max_bid_ask_spread > Decimal::ZERO {
            let bid = if is_yes {
                book.yes.best_bid()
            } else {
                book.no.best_bid()
            };
            if !bid.is_some_and(|b| ask - b <= self.cfg.max_bid_ask_spread) {
                return vec![];
            }
        }

        // Sizing: caps max_order_usdc / max_size_shares ∧ available depth (FOK).
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
        let size_int = size.trunc_with_scale(0);
        if size_int < Decimal::ONE || size_int * ask < dec!(1) {
            return vec![];
        }

        // Insert state BEFORE emitting the intent: guarantees the next
        // tick does not emit a second entry order even if the fill has not arrived yet.
        self.states.insert(
            cid.clone(),
            BracketState {
                clear_at: book.clear_at,
                have_position: false,
                token_id: token_id.clone(),
                size: Decimal::ZERO,
                bail_checks: 0,
                bail_sent_at: None,
                bail_confirmed: false,
                reference,
            },
        );

        metrics::counter!("hrf_om_signals_total").increment(1);
        tracing::info!(
            cid = %cid, is_yes, fv = %fv, ask = %ask, age_secs, remaining,
            "OM: opening signal"
        );

        vec![OrderIntent {
            market_cid: cid.clone(),
            token_id,
            side: OrderSide::Buy,
            price: ask,
            size: size_int,
            clear_at: book.clear_at,
            signal: (fv - half).abs(),
            neg_risk: book.neg_risk,
            limit_price: Some(ask),
            second_leg: None,
            strategy: StrategyId::OpenMomentum,
            resting: false,
            real_clear_at: None,
        }]
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::OpenMomentum {
            return;
        }
        // Buy fill without state (restart mid-bracket): without a reference no
        // bail is possible; the orphan checker will liquidate at settlement. Make it visible.
        if !self.states.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "OM: Buy fill without state, position left to the orphan checker (restart mid-bracket)");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "open_momentum").increment(1);
            }
            return;
        }
        if let Some(state) = self.states.get_mut(&fill.market_cid) {
            match fill.side {
                FillSide::Buy => {
                    state.have_position = true;
                    state.size += fill.size;
                    state.bail_checks = 0;
                    metrics::counter!("hrf_om_fill_total").increment(1);
                }
                FillSide::Sell => {
                    // Partial sell (FAK capped to depth): the rest stays under management.
                    state.size = (state.size - fill.size).max(Decimal::ZERO);
                    state.bail_sent_at = None;
                    if state.size == Decimal::ZERO {
                        state.have_position = false;
                        state.bail_confirmed = true;
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
        strategy: StrategyId::OpenMomentum,
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
        no_ask: &str,
        remaining: i64,
        reference: Option<&str>,
        age_secs: i64,
    ) -> BookSnapshot {
        let level = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let now = crate::time::now().timestamp();
        BookSnapshot {
            market_condition_id: "om-cid".into(),
            yes: TokenBook {
                token_id: "yes-tok".into(),
                bids: level("0.48"),
                asks: level(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-tok".into(),
                bids: level("0.46"),
                asks: level(no_ask),
                ..Default::default()
            },
            market_open_ts: now - age_secs,
            clear_at: now + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-om".into(),
            reference_price: reference.map(|p| p.parse().unwrap()),
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
            updated_at: crate::time::Instant::now(),
            ..VolSnapshot::default()
        }
    }

    #[test]
    fn om_enters_on_open_with_confidence() {
        let mut om = OpenMomentum::new(OpenMomentumConfig::default(), Decimal::ZERO);
        // live=110k, reference=100k → fv >> 0.58 → enters YES at open (age=5s)
        let book = make_book("0.52", "0.50", 240, Some("100000"), 5);
        let vol = make_vol("110000");
        let intents = om.on_book(&book, &vol);
        assert_eq!(intents.len(), 1);
        assert!(matches!(intents[0].side, OrderSide::Buy));
        assert_eq!(intents[0].token_id, "yes-tok");
    }

    #[test]
    fn om_skip_outside_open_window() {
        let mut om = OpenMomentum::new(OpenMomentumConfig::default(), Decimal::ZERO);
        // age=60s > max_open_secs=30 → does not enter
        let book = make_book("0.52", "0.50", 240, Some("100000"), 60);
        let vol = make_vol("110000");
        assert!(om.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn om_skip_without_confidence() {
        let mut om = OpenMomentum::new(OpenMomentumConfig::default(), Decimal::ZERO);
        // live=reference → fv=0.5 → no confidence
        let book = make_book("0.52", "0.50", 240, Some("100000"), 5);
        let vol = make_vol("100000");
        assert!(om.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn om_no_reentry_after_insert() {
        let mut om = OpenMomentum::new(OpenMomentumConfig::default(), Decimal::ZERO);
        let book = make_book("0.52", "0.50", 240, Some("100000"), 5);
        let vol = make_vol("110000");
        // First call → enters
        let intents = om.on_book(&book, &vol);
        assert_eq!(intents.len(), 1);
        // Second call (same opening) → no retry
        let intents2 = om.on_book(&book, &vol);
        assert!(intents2.is_empty());
    }

    #[test]
    fn om_slug_epoch_overwrites_market_open_ts() {
        // market_open_ts = 24h ago (like the real updown API), but the slug has
        // the real opening epoch (5s ago). OM must enter using the slug epoch.
        let now = crate::time::now().timestamp();
        let slug_epoch = now - 5; // bracket opened 5s ago
        let level = |p: &str| crate::types::HalfBook {
            levels: vec![crate::types::Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "om-slug-cid".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: level("0.48"),
                asks: level("0.52"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: level("0.46"),
                asks: level("0.50"),
                ..Default::default()
            },
            market_open_ts: now - 86400, // simulates start_date ≈ 24h before
            clear_at: now + 240,
            neg_risk: false,
            slug: format!("btc-updown-5m-{slug_epoch}"),
            reference_price: Some(dec!(100000)),
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        };
        let vol = make_vol("110000");
        let intents =
            OpenMomentum::new(OpenMomentumConfig::default(), Decimal::ZERO).on_book(&book, &vol);
        assert_eq!(
            intents.len(),
            1,
            "must enter even though market_open_ts is 24h back"
        );
    }

    #[test]
    fn om_bail_sell_after_reversion() {
        let mut om = OpenMomentum::new(OpenMomentumConfig::default(), Decimal::ZERO);
        let book = make_book("0.52", "0.50", 240, Some("100000"), 5);
        let vol = make_vol("110000");

        let _ = om.on_book(&book, &vol);
        // Confirm the buy fill
        om.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "om-cid".into(),
            token_id: "yes-tok".into(),
            side: FillSide::Buy,
            price: dec!(0.52),
            size: dec!(10),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::OpenMomentum,
            clear_at: 0,
            outcome_label: None,
        });

        // Oracle reverts sharply: -10% → very negative cushion
        let vol_bad = make_vol("90000");
        let mut sells = vec![];
        for _ in 0..5 {
            let i = om.on_book(&book, &vol_bad);
            sells.extend(
                i.into_iter()
                    .filter(|x| matches!(x.side, OrderSide::Sell { .. })),
            );
        }
        assert_eq!(
            sells.len(),
            1,
            "must emit bail-sell after bail_checks_required ticks"
        );
        assert_eq!(sells[0].token_id, "yes-tok");
    }

    /// With a nominal edge of 3pp (< haircut 4pp) OM must NOT
    /// enter; with a nominal edge of 5pp (> haircut) it MUST enter. `ask` fixed
    /// within `max_entry_price` (0.60) and `parabolic_fee_rate`
    /// (parabolic model, not triangular `fee_rate_bps`) derived so that the
    /// breakeven falls exactly at `fv_side - 0.03`/`-0.05`.
    #[test]
    fn om_haircut_blocks_edge_3pp_and_lets_edge_5pp_through() {
        let live = dec!(105000);
        let reference = dec!(100000);
        let sigma = dec!(0.001);
        let remaining = 180i64; // >= min_remaining (120)
        let fv_side =
            fair_value_yes(live, reference, sigma, remaining).expect("fv must be computable");

        let ask = dec!(0.55); // <= max_entry_price (0.60)
        let haircut = dec!(0.04);

        // Inverse of fee_taker_parabolic_with_rate: breakeven = ask + rate*ask*(1-ask).
        let rate_for_breakeven =
            |breakeven: Decimal| -> Decimal { (breakeven - ask) / (ask * (Decimal::ONE - ask)) };

        let vol = make_vol("105000");

        // age_secs=5 (within de max_open_secs=30).
        let mut book_3pp = make_book("0.55", "0.45", remaining, Some("100000"), 5);
        book_3pp.parabolic_fee_rate = Some(rate_for_breakeven(fv_side - dec!(0.03)));
        let mut om_3pp = OpenMomentum::new(OpenMomentumConfig::default(), haircut);
        assert!(
            om_3pp.on_book(&book_3pp, &vol).is_empty(),
            "nominal edge 3pp < haircut 4pp — OM must not enter"
        );

        let mut book_5pp = make_book("0.55", "0.45", remaining, Some("100000"), 5);
        book_5pp.parabolic_fee_rate = Some(rate_for_breakeven(fv_side - dec!(0.05)));
        let mut om_5pp = OpenMomentum::new(OpenMomentumConfig::default(), haircut);
        assert_eq!(
            om_5pp.on_book(&book_5pp, &vol).len(),
            1,
            "nominal edge 5pp > haircut 4pp — OM must enter"
        );
    }

    /// A book WITHOUT a bid (`best_bid()==None`) must fail CLOSED in the
    /// spread filter, not let the entry through without applying the filter.
    #[test]
    fn om_without_bid_fails_closed_in_spread_filter() {
        let cfg = OpenMomentumConfig {
            max_bid_ask_spread: dec!(0.05),
            ..OpenMomentumConfig::default()
        };
        let mut om = OpenMomentum::new(cfg, Decimal::ZERO);
        let mut book = make_book("0.55", "0.45", 180, Some("100000"), 5);
        book.yes.bids = crate::types::HalfBook { levels: vec![] }; // no bid
        let vol = make_vol("105000");
        assert!(
            om.on_book(&book, &vol).is_empty(),
            "without a bid the spread filter must reject the entry, not let it through"
        );
    }
}
