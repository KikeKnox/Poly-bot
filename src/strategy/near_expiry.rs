//! Near-Expiry (NE) strategy.
//!
//! Signal: near close, the underlying has moved > min_dist_pct away from the strike
//! in the direction of the winning bracket side → buy that side.
//!
//! Sizing: logarithmic scaling by distance (more distance → more size).
//! Bail-sell: if the price reverts below bail_threshold_pct for N checks → sell.

use std::collections::HashMap;

use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    config::{defaults, StrategyLimits},
    types::{BookSnapshot, Fill, FillSide, OrderIntent, OrderSide, StrategyId, VolSnapshot},
};

use super::{fallback_reference, Strategy, StrategyConfig};

// ── Per-bracket state ────────────────────────────────────────────────────────

struct MarketState {
    clear_at: i64,
    reference: Option<Decimal>,
    have_position: bool,
    open_token: Option<String>,
    open_size: Decimal,
    /// Per-tick on purpose: under `drain_latest` coalescing the failure mode
    /// is a later bail (fewer checks per second), not a duplicated
    /// bail; switch to a time deadline if calibration shows systematically
    /// slow bails (TC documents the same limitation in its own counter).
    bail_checks: u32,
    /// Instant of the in-flight bail-sell (None = none). TIME-based deadline as
    /// in NC: 20 ticks ~ 2s at 10 ticks/s was less than the user_ws latency
    /// (700ms-2s) and could re-emit a duplicate sell.
    bail_sent_at: Option<crate::time::Instant>,
    /// Entry emitted (set when the intent is EMITTED, closing the FOK-timeout
    /// race). A Sell fill resets it: NE allows re-entry after a bail.
    ever_entered: bool,
    /// Re-entries already consumed in this bracket after a complete bail
    /// (see `NE_MAX_REENTRIES_PER_BRACKET`); bounds whipsaw in
    /// markets oscillating near the reference (same spirit as
    /// `dump_count`/`max_dumps_per_bracket` in PM).
    reentries: u32,
}

/// Maximum re-entries per bracket after a complete bail. Without
/// this limit, NE re-entered with no cooldown or counter every time the price
/// crossed back over the minimum distance; a market oscillating near the
/// reference could generate unlimited entries and bails (whipsaw), each
/// paying fee and spread. The limit is applied here as a local
/// constant. Migrating it to a real field of `NearExpiryConfig`
/// (`max_reentries_per_bracket`, with its entry in
/// `config/strategies/near_expiry.toml`) is pending.
const NE_MAX_REENTRIES_PER_BRACKET: u32 = 2;

// ── NearExpiry ────────────────────────────────────────────────────────────────

pub struct NearExpiry {
    cfg: NearExpiryConfig,
    markets: HashMap<String, MarketState>,
    last_retain: crate::time::Instant,
}

impl NearExpiry {
    pub fn new(cfg: NearExpiryConfig) -> Self {
        Self {
            cfg,
            markets: HashMap::new(),
            last_retain: crate::time::Instant::now(),
        }
    }
}

impl Strategy for NearExpiry {
    fn id(&self) -> StrategyId {
        StrategyId::NearExpiry
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        if self.last_retain.elapsed() >= std::time::Duration::from_secs(30) {
            self.markets.retain(|_, s| s.clear_at > now);
            self.last_retain = crate::time::Instant::now();
        }

        let mut intents = Vec::new();
        let remaining = book.secs_to_clear();

        let cid = &book.market_condition_id;
        // Capture the reference at the bracket's first tick (ideally near the open,
        // remaining ≈ 295s), not in the final window. If captured at T-45s the reference
        // reflects only the last 45s and dist_pct never reaches the threshold.
        let is_new_market = !self.markets.contains_key(cid);
        let state = self.markets.entry(cid.clone()).or_insert_with(|| {
            let reference = book
                .reference_price
                .or_else(|| fallback_reference(vol, book, now, cid, "NE"));
            MarketState {
                clear_at: book.clear_at,
                reference,
                have_position: false,
                open_token: None,
                open_size: Decimal::ZERO,
                bail_checks: 0,
                bail_sent_at: None,
                ever_entered: false,
                reentries: 0,
            }
        });
        // Structured log at the EXACT moment the reference is set.
        // Failure mode if the capture comes late: dist_pct never
        // reaches the threshold (see the `fallback_reference` comments above).
        // Lets you grep out the distribution of `ref_capture_remaining`.
        if is_new_market {
            if let Some(r) = state.reference {
                tracing::info!(
                    cid = %cid, slug = %book.slug, ref_price = %r, remaining_secs = remaining,
                    target = "ne_ref",
                    "NE: price reference set"
                );
            }
        }

        // The ENTRY window gate [min_lead_secs, window_secs]
        // is split in two: here the tick is only dropped if the bracket
        // already closed (`remaining<=0`), a condition that also applies to the bail-sell.
        // The rest of the window (`window_secs`/`min_lead_secs`) is applied FURTHER
        // DOWN, after the bail-sell block, so as not to block the
        // EXIT as well: previously, with `remaining < min_lead_secs` (the last seconds
        // before close) this `return` cut the WHOLE tick, including the
        // bail-sell, leaving the exit unreachable just when it could be
        // most needed. Compare with NC, which covers the bail until `remaining=1`.
        if remaining <= 0 {
            return intents;
        }

        // Safety fallback: if the first tick arrived without vol (late startup or resync),
        // retry with the oracle's current price, same freshness gate.
        if state.reference.is_none() {
            state.reference = book
                .reference_price
                .or_else(|| fallback_reference(vol, book, now, cid, "NE"));
            // Same structured log as above: this branch sets the
            // reference later (after the window gate), so the
            // remaining_secs recorded here IS already the "late capture" candidate.
            if let Some(r) = state.reference {
                tracing::info!(
                    cid = %cid, slug = %book.slug, ref_price = %r, remaining_secs = remaining,
                    target = "ne_ref",
                    "NE: price reference set (late fallback)"
                );
            }
        }
        let reference = match state.reference {
            Some(r) if r > Decimal::ZERO => r,
            _ => return intents,
        };

        let live = vol.last_price;
        let oracle_fresh = live > Decimal::ZERO && !vol.is_stale_for_signal();

        // ── Bail-sell ─────────────────────────────────────────────────────────
        if state.have_position {
            if let Some(token) = state.open_token.as_deref() {
                if let Some(sent_at) = state.bail_sent_at {
                    // Sell already in flight: time-based deadline (see NC).
                    if sent_at.elapsed().as_secs() >= self.cfg.bail_retry_secs {
                        state.bail_sent_at = None;
                        state.bail_checks = 0;
                        tracing::warn!(
                            cid = %cid, secs = self.cfg.bail_retry_secs,
                            "NE: bail-sell without fill after the deadline; reset for retry"
                        );
                    }
                } else if oracle_fresh {
                    let bought_yes = token == book.yes.token_id;
                    // In PERCENT (×100), the same unit as dist_pct/min_dist_pct.
                    // Previously the FRACTION was compared against bail_threshold_pct=0.20, i.e.
                    // a 20% reversal in 45s: the bail was inert because of units.
                    let cushion_pct = if bought_yes {
                        (live - reference) / reference * dec!(100)
                    } else {
                        (reference - live) / reference * dec!(100)
                    };

                    if cushion_pct < -self.cfg.bail_threshold_pct {
                        state.bail_checks += 1;
                        if state.bail_checks >= 2 {
                            let token_bid = if bought_yes {
                                book.yes.best_bid().unwrap_or(dec!(0.01))
                            } else {
                                book.no.best_bid().unwrap_or(dec!(0.01))
                            };
                            if token_bid < self.cfg.min_bail_bid {
                                tracing::warn!(
                                    cid = %cid, bid = %token_bid, min = %self.cfg.min_bail_bid,
                                    "NE: bail-sell cancelled — bid too low, hold-to-expiry"
                                );
                                state.bail_checks = 0;
                            } else {
                                intents.push(sell_intent(book, token, token_bid, state.open_size));
                                state.bail_checks = 0;
                                state.bail_sent_at = Some(crate::time::Instant::now());
                                metrics::counter!("hrf_ne_bail_total").increment(1);
                                tracing::info!(
                                    cid = %cid,
                                    cushion_pct = %cushion_pct,
                                    threshold = %(-self.cfg.bail_threshold_pct),
                                    "NE: bail-sell, underlying reverted below the threshold"
                                );
                            }
                        }
                    } else {
                        state.bail_checks = 0;
                    }
                }
            }
            return intents;
        }

        // Rest of the ENTRY window, applied only from
        // here on, AFTER the bail-sell above (which already handled the
        // `state.have_position` scenario and returned). An open position never reaches
        // this point in the same tick.
        if remaining > self.cfg.window_secs as i64 || remaining < self.cfg.min_lead_secs as i64 {
            return intents;
        }

        // ever_entered is set on emit (closes the FOK-timeout race) and only a
        // Sell fill resets it; the old entry_pending_ticks block was
        // unreachable (ever_entered always returned earlier) and was removed.
        if state.ever_entered {
            return intents;
        }

        if !oracle_fresh {
            return intents;
        }

        // ── Entry signal ──────────────────────────────────────────────────

        let dist = (live - reference).abs();
        let dist_pct = dist / reference * dec!(100);
        if dist_pct < self.cfg.min_dist_pct {
            metrics::counter!("hrf_ne_skip_dist_total").increment(1);
            return intents;
        }

        let is_yes = live > reference;

        // Sigma filter (optional — None or 0 = disabled).
        if self.cfg.min_sigma_for_entry > Decimal::ZERO {
            let sigma = vol.sigma_short.unwrap_or_default();
            if sigma < self.cfg.min_sigma_for_entry {
                metrics::counter!("hrf_ne_skip_sigma_total").increment(1);
                return intents;
            }
        }

        // Ask of the winning side with a spread filter (0 = disabled).
        // The central hypothesis of the calibration finding
        // is that THIS gate, not min_dist_pct, kills almost all the signal
        // (98.4% of what passes the distance filter dies here); it has its
        // own counter, clearly distinguishable from the rest.
        let side_book = if is_yes { &book.yes } else { &book.no };
        let ask = match side_book.best_ask() {
            Some(a) if a >= self.cfg.min_ask && a <= self.cfg.max_ask => {
                if self.cfg.max_bid_ask_spread > Decimal::ZERO {
                    let bid = side_book.best_bid().unwrap_or(Decimal::ZERO);
                    if a - bid > self.cfg.max_bid_ask_spread {
                        metrics::counter!("hrf_ne_skip_spread_total").increment(1);
                        return intents;
                    }
                }
                a
            }
            _ => {
                metrics::counter!("hrf_ne_skip_ask_range_total").increment(1);
                return intents;
            }
        };

        // Sizing: ln(1 + ratio)/ln(5) scaled to [0.25, 0.85].
        let ratio_f64 = (dist_pct / self.cfg.min_dist_pct).to_f64().unwrap_or(1.0);
        let scale_f64 = ((1.0 + ratio_f64).ln() / 5_f64.ln()).clamp(0.25, 0.85);
        let scale = Decimal::from_f64(scale_f64).unwrap_or(dec!(0.25));
        let scaled_usdc = self.cfg.order_usdc * scale;
        let fav_depth = if is_yes {
            book.yes.asks.best_size()
        } else {
            book.no.asks.best_size()
        };
        let mut size =
            super::size_from_caps(scaled_usdc, self.cfg.max_size_shares, ask, None, fav_depth);
        if size <= Decimal::ZERO {
            metrics::counter!("hrf_ne_skip_size_total").increment(1);
            return intents;
        }

        // FOK: cap to the available depth instead of skipping entirely; captures the trade at a smaller size.
        if !self.cfg.use_fak {
            match fav_depth {
                None => {
                    metrics::counter!("hrf_ne_skip_depth_total").increment(1);
                    return intents;
                }
                Some(d) if d < size => {
                    tracing::debug!(cid = %cid, size = %size, depth = %d, "NE: capping size to the available depth for FOK");
                    size = d;
                    if size <= Decimal::ZERO {
                        metrics::counter!("hrf_ne_skip_depth_total").increment(1);
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
            signal: dist_pct,
            neg_risk: book.neg_risk,
            limit_price: Some(ask),
            second_leg: None,
            strategy: StrategyId::NearExpiry,
            resting: false,
            real_clear_at: None,
        });

        state.ever_entered = true;

        metrics::counter!("hrf_ne_signals_total").increment(1);
        tracing::info!(
            cid = %cid, is_yes, ask = %ask, dist_pct = %dist_pct, scale = %scale, remaining,
            "NE: entry signal"
        );

        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::NearExpiry {
            return;
        }
        // Buy fill without state (restart mid-bracket): recreate the position
        // (same pattern as ARB). Without a reference the bail stays disabled (hold),
        // but Sells are accounted for and the bracket is not re-entered.
        if !self.markets.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "NE: Buy fill without state, recreating position (restart mid-bracket, bail disabled without reference)");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "near_expiry").increment(1);
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
                        reentries: 0,
                    },
                );
            }
            return;
        }
        if let Some(state) = self.markets.get_mut(&fill.market_cid) {
            match fill.side {
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
                        // NE allows re-entry after a complete bail-sell: the oracle
                        // can confirm the winning side again if the price reverts.
                        // Bounded by NE_MAX_REENTRIES_PER_BRACKET: without
                        // this cap, a market oscillating near the reference could
                        // generate unlimited entries and bails (whipsaw). Once the
                        // quota is exhausted, the bracket is "burned" for NE (no more entries)
                        // until it expires, same spirit as the burned bracket in
                        // PM (dump_count).
                        if state.reentries < NE_MAX_REENTRIES_PER_BRACKET {
                            state.reentries += 1;
                            state.ever_entered = false;
                        } else {
                            metrics::counter!("hrf_ne_burned_bracket_total").increment(1);
                            tracing::warn!(cid = %fill.market_cid, reentries = state.reentries,
                                "NE: bracket burned, maximum re-entries reached, no more re-entries");
                        }
                    }
                }
                // Same semantics as `Rejected`: `ever_entered` is
                // deliberately one-entry-per-bracket (a gate blocker
                // is equivalent to a rejection: this bracket no longer enters).
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
        signal: price, // token price (0-1)
        neg_risk: book.neg_risk,
        limit_price: None,
        second_leg: None,
        strategy: StrategyId::NearExpiry,
        resting: false,
        real_clear_at: None,
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct NearExpiryConfig {
    #[serde(flatten)]
    pub limits: StrategyLimits,
    #[serde(default = "defaults::ne_min_dist_pct")]
    pub min_dist_pct: Decimal,
    #[serde(default = "defaults::ne_window_secs")]
    pub window_secs: u64,
    #[serde(default = "defaults::ne_min_lead_secs")]
    pub min_lead_secs: u64,
    #[serde(default = "defaults::ne_min_ask")]
    pub min_ask: Decimal,
    #[serde(default = "defaults::ne_max_ask")]
    pub max_ask: Decimal,
    #[serde(default = "defaults::ne_order_usdc")]
    pub order_usdc: Decimal,
    #[serde(default = "defaults::ne_bail_threshold_pct")]
    pub bail_threshold_pct: Decimal,
    #[serde(default = "defaults::ne_min_sigma")]
    pub min_sigma_for_entry: Decimal,
    /// Maximum size in shares per order. 0 = no cap; the limit is order_usdc.
    #[serde(default)]
    pub max_size_shares: Decimal,
    /// Minimum bid to run a bail-sell. 0 = no minimum (sells at the available bid).
    /// Protects against selling into a dry market where the bid = $0.01.
    #[serde(default = "defaults::ne_min_bail_bid")]
    pub min_bail_bid: Decimal,
    /// Maximum tolerable bid-ask spread on entry. 0 = no filter.
    #[serde(default)]
    pub max_bid_ask_spread: Decimal,
    /// If true, uses FAK on entry: fills the available depth and cancels the rest.
    #[serde(default)]
    pub use_fak: bool,
    /// Seconds before giving up on an in-flight bail-sell and allowing a
    /// retry (same time deadline as NC).
    #[serde(default = "defaults::ne_bail_retry_secs")]
    pub bail_retry_secs: u64,
}

impl Default for NearExpiryConfig {
    fn default() -> Self {
        Self {
            limits: StrategyLimits::default(),
            min_dist_pct: defaults::ne_min_dist_pct(),
            window_secs: defaults::ne_window_secs(),
            min_lead_secs: defaults::ne_min_lead_secs(),
            min_ask: defaults::ne_min_ask(),
            max_ask: defaults::ne_max_ask(),
            order_usdc: defaults::ne_order_usdc(),
            bail_threshold_pct: defaults::ne_bail_threshold_pct(),
            min_sigma_for_entry: defaults::ne_min_sigma(),
            max_size_shares: Decimal::ZERO,
            min_bail_bid: defaults::ne_min_bail_bid(),
            max_bid_ask_spread: Decimal::ZERO,
            use_fak: false,
            bail_retry_secs: defaults::ne_bail_retry_secs(),
        }
    }
}

impl StrategyConfig for NearExpiryConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
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
    ) -> BookSnapshot {
        let ask_level = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        BookSnapshot {
            market_condition_id: "ne-test-cid".into(),
            yes: TokenBook {
                token_id: "yes-tok".into(),
                bids: HalfBook { levels: vec![] },
                asks: ask_level(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-tok".into(),
                bids: HalfBook { levels: vec![] },
                asks: ask_level(no_ask),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-ne".into(),
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
    fn ne_enters_with_sufficient_distance() {
        let mut ne = NearExpiry::new(NearExpiryConfig::default());

        // reference=100k, live=103k → dist_pct=3% (> min_dist_pct default ~2%)
        let book = make_book("0.85", "0.15", 30, Some("100000"));
        let vol = make_vol("103000");
        let intents = ne.on_book(&book, &vol);

        assert_eq!(intents.len(), 1);
        assert!(matches!(intents[0].side, OrderSide::Buy));
        assert_eq!(intents[0].token_id, "yes-tok");
    }

    /// The earlier fixture (ask=0.51, remaining=60) actually
    /// failed at the WINDOW gate (remaining=60 > default window_secs=45)
    /// or, had it gone further, at the ASK gate (0.51 <
    /// min_ask=0.60); it never exercised the distance gate the test
    /// name claims to cover. Here remaining=30 (within the window) and ask=0.65
    /// (within [min_ask, max_ask]) so the only gate that blocks is
    /// the distance one: dist_pct=0.002% < min_dist_pct=0.003% (defaults).
    #[test]
    fn ne_skip_insufficient_distance() {
        let mut ne = NearExpiry::new(NearExpiryConfig::default());
        let book = make_book("0.65", "0.35", 30, Some("100000"));
        let vol = make_vol("100002"); // dist_pct = 0.002% < min_dist_pct (0.003%)
        assert!(ne.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn ne_skip_outside_window() {
        let mut ne = NearExpiry::new(NearExpiryConfig::default());
        let book = make_book("0.85", "0.15", 7200, Some("100000"));
        let vol = make_vol("103000");
        assert!(ne.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn pending_ne_bail_is_not_duplicated_within_deadline() {
        // The in-flight sell must not be re-emitted no matter how many ticks pass while
        // the time deadline (bail_retry_secs) has not expired; previously 20 ticks (~2s at
        // 10 ticks/s) reset before the fill arrived via user_ws.
        let mut ne = NearExpiry::new(NearExpiryConfig::default());
        let mut book = make_book("0.85", "0.15", 30, Some("100000"));
        book.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.80),
                size: dec!(50),
            }],
        };
        // YES entry + fill.
        assert_eq!(ne.on_book(&book, &make_vol("103000")).len(), 1);
        ne.on_fill(&crate::types::Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "ne-test-cid".into(),
            token_id: "yes-tok".into(),
            side: crate::types::FillSide::Buy,
            price: dec!(0.85),
            size: dec!(10),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::NearExpiry,
            clear_at: 0,
            outcome_label: None,
        });

        // Oracle reverts well below the threshold (default bail_threshold 0.20):
        // reference 100k → live 75k = −25%.
        let vol_down = make_vol("75000");
        let mut sells = 0;
        for _ in 0..60 {
            sells += ne
                .on_book(&book, &vol_down)
                .iter()
                .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
                .count();
        }
        assert_eq!(
            sells, 1,
            "a single bail-sell while the time deadline has not expired"
        );
    }

    /// The bail-sell must keep working in the last seconds
    /// before close (`remaining < min_lead_secs`, 5s by default); previously
    /// the ENTRY window gate also cut off the EXIT in that stretch.
    #[test]
    fn ne_bail_reachable_below_min_lead_secs() {
        let mut ne = NearExpiry::new(NearExpiryConfig::default());
        let mut book = make_book("0.85", "0.15", 30, Some("100000"));
        book.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.80),
                size: dec!(50),
            }],
        };
        assert_eq!(
            ne.on_book(&book, &make_vol("103000")).len(),
            1,
            "initial entry"
        );
        ne.on_fill(&crate::types::Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "ne-test-cid".into(),
            token_id: "yes-tok".into(),
            side: crate::types::FillSide::Buy,
            price: dec!(0.85),
            size: dec!(10),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::NearExpiry,
            clear_at: 0,
            outcome_label: None,
        });

        // Clock advances to remaining=3s (< min_lead_secs=5) — oracle reverts sharply.
        let mut book_final = book.clone();
        book_final.clear_at = crate::time::now().timestamp() + 3;
        let vol_down = make_vol("75000");
        let mut sells = 0;
        for _ in 0..5 {
            sells += ne
                .on_book(&book_final, &vol_down)
                .iter()
                .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
                .count();
        }
        assert_eq!(
            sells, 1,
            "the bail-sell must keep working even if remaining < min_lead_secs"
        );
    }

    /// After `NE_MAX_REENTRIES_PER_BRACKET` (2) consumed
    /// re-entries, the bracket is "burned" for NE: it must not
    /// re-enter even if the distance signal holds again (whipsaw).
    #[test]
    fn ne_reentry_cap_blocks_whipsaw_after_max_reentries() {
        let mut ne = NearExpiry::new(NearExpiryConfig::default());
        let mut book = make_book("0.85", "0.15", 30, Some("100000"));
        book.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.80),
                size: dec!(50),
            }],
        };

        let buy_fill = |ne: &mut NearExpiry| {
            ne.on_fill(&crate::types::Fill {
                hedge_id: None,
                exec_id: None,
                local_order_id: None,
                market_cid: "ne-test-cid".into(),
                token_id: "yes-tok".into(),
                side: crate::types::FillSide::Buy,
                price: dec!(0.85),
                size: dec!(10),
                order_id: None,
                ts: crate::time::now(),
                mode: crate::types::Mode::Paper,
                neg_risk: false,
                strategy: StrategyId::NearExpiry,
                clear_at: 0,
                outcome_label: None,
            });
        };
        let sell_fill_full = |ne: &mut NearExpiry| {
            ne.on_fill(&crate::types::Fill {
                hedge_id: None,
                exec_id: None,
                local_order_id: None,
                market_cid: "ne-test-cid".into(),
                token_id: "yes-tok".into(),
                side: crate::types::FillSide::Sell,
                price: dec!(0.80),
                size: dec!(10),
                order_id: None,
                ts: crate::time::now(),
                mode: crate::types::Mode::Paper,
                neg_risk: false,
                strategy: StrategyId::NearExpiry,
                clear_at: 0,
                outcome_label: None,
            });
        };

        // Initial entry + bail complete → reentries goes from 0 to 1.
        assert_eq!(
            ne.on_book(&book, &make_vol("103000")).len(),
            1,
            "initial entry"
        );
        buy_fill(&mut ne);
        sell_fill_full(&mut ne);

        // Re-entry 1 (reentries=1<2, allowed) + bail → reentries goes to 2.
        assert_eq!(
            ne.on_book(&book, &make_vol("103000")).len(),
            1,
            "re-entry 1 allowed"
        );
        buy_fill(&mut ne);
        sell_fill_full(&mut ne);

        // Re-entry 2 (reentries=2<2 is false: this bail no longer increments,
        // but the entry preceding it was still allowed with reentries=2).
        assert_eq!(
            ne.on_book(&book, &make_vol("103000")).len(),
            1,
            "re-entry 2 allowed (quota runs out after this)"
        );
        buy_fill(&mut ne);
        sell_fill_full(&mut ne);

        // Quota exhausted: NE must not re-enter in this bracket.
        assert!(
            ne.on_book(&book, &make_vol("103000")).is_empty(),
            "after exhausting NE_MAX_REENTRIES_PER_BRACKET it must not re-enter"
        );
    }

    #[test]
    fn ne_without_reference_price_and_old_open_ts_does_not_trade() {
        // Startup mid-bracket without reference_price from the feed: the fallback
        // to the oracle's current price is discarded (stale open_ts) and NE does not trade.
        let mut ne = NearExpiry::new(NearExpiryConfig::default());
        let book = make_book("0.85", "0.15", 30, None); // market_open_ts = 0 (stale)
        let vol = make_vol("103000");
        for _ in 0..5 {
            assert!(
                ne.on_book(&book, &vol).is_empty(),
                "without a valid reference it must not emit intents"
            );
        }
    }

    #[test]
    fn ne_valid_oracle_fallback_near_open() {
        // Same scenario but with the bracket just opened → the fallback is accepted.
        let mut ne = NearExpiry::new(NearExpiryConfig::default());
        let mut book = make_book("0.85", "0.15", 30, None);
        book.market_open_ts = crate::time::now().timestamp() - 5; // opened 5s ago
                                                                  // First tick captures reference = 100k (oracle); then the oracle moves to 103k.
        let _ = ne.on_book(&book, &make_vol("100000"));
        let intents = ne.on_book(&book, &make_vol("103000"));
        assert_eq!(
            intents.len(),
            1,
            "with a fresh fallback it must operate as before"
        );
    }
}
