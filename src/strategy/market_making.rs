//! Market Making (MM) strategy.
//!
//! Posts GTD bids on YES and NO centered on the fair value:
//!   bid_YES = FV - half_spread
//!   bid_NO  = (1 - FV) - half_spread
//!
//! Requotes when FV moves > requote_threshold_bps since the last post,
//! or when the order exceeds its TTL.

use crate::time::Instant;
use std::collections::HashMap;
use std::time::Duration;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    config::defaults,
    fair_value::fair_value_yes,
    types::{
        truncate_to_tick, BookSnapshot, Fill, FillSide, OrderIntent, OrderSide, StrategyId,
        VolSnapshot,
    },
};

use super::{fallback_reference, Strategy, StrategyConfig};

// ── Configuration ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MarketMakingConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    #[serde(default = "defaults::mm_spread_bps")]
    pub spread_bps: u32,
    #[serde(default = "defaults::mm_min_spread_bps")]
    pub min_spread_bps: u32,
    #[serde(default = "defaults::mm_max_spread_bps")]
    pub max_spread_bps: u32,
    #[serde(default = "defaults::mm_vol_factor")]
    pub vol_factor: Decimal,
    #[serde(default = "defaults::mm_order_size_shares")]
    pub order_size_shares: Decimal,
    #[serde(default = "defaults::mm_max_inventory_shares")]
    pub max_inventory_shares: Decimal,
    #[serde(default = "defaults::mm_requote_threshold_bps")]
    pub requote_threshold_bps: u32,
    /// Requote tolerance in market ticks (order refresh tolerance):
    /// if the bid already resting on a leg is at most N ticks from the desired
    /// price, that leg is NOT canceled+reposted. 0 = disabled (requote
    /// whenever the FV threshold or the TTL trips, the earlier behavior).
    #[serde(default)]
    pub requote_tolerance_ticks: u32,
    #[serde(default = "defaults::mm_order_ttl_secs")]
    pub order_ttl_secs: u64,
    /// Maximum σ_5s/σ_60s ratio before pausing quotes (adverse selection). 0 = disabled.
    #[serde(default)]
    pub vol_ratio_threshold: Decimal,
    /// Minimum bid-ask spread to quote. Below it, no margin for MM. 0 = disabled.
    #[serde(default)]
    pub min_spread_to_quote: Decimal,
    /// Minimum bid to unwind the inventory at window close. Dust bids
    /// save nothing and settlement may still pay 1 — hold in that scenario.
    #[serde(default = "defaults::mm_min_unwind_bid")]
    pub min_unwind_bid: Decimal,
}

impl StrategyConfig for MarketMakingConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

// ── Per-bracket state ────────────────────────────────────────────────────────

struct MMState {
    clear_at: i64,
    reference: Option<Decimal>,
    posted_fv: Option<Decimal>,
    posted_at: Option<Instant>,
    /// Price at which each leg's bid was left resting at the last post —
    /// what the requote tolerance consumes. `None` = we do not believe there is a
    /// live GTD on that token: cleared together with `posted_fv`, on any
    /// terminal event (Cancelled/Rejected) and on every Buy fill (conservative:
    /// a partial fill leaves the order live but reposting it is covered by the requote's
    /// prior cancel; a full fill avoids keeping a phantom order).
    posted_bid_yes: Option<Decimal>,
    posted_bid_no: Option<Decimal>,
    yes_shares: Decimal,
    no_shares: Decimal,
    yes_token: String,
    no_token: String,
}

// ── MarketMaking ──────────────────────────────────────────────────────────────

pub struct MarketMaking {
    cfg: MarketMakingConfig,
    markets: HashMap<String, MMState>,
    last_retain: Instant,
}

impl MarketMaking {
    pub fn new(cfg: MarketMakingConfig) -> Self {
        Self {
            cfg,
            markets: HashMap::new(),
            last_retain: Instant::now(),
        }
    }

    fn spread_frac(&self, sigma: Option<Decimal>) -> Decimal {
        let base = Decimal::from(self.cfg.spread_bps) / dec!(10000);
        // `Ord::clamp` asserts `min<=max` — a panic
        // here brings the whole bot down. The current config has min<max, but nothing
        // in `config.rs::validate()` guarantees it.
        // Defensive: sort the pair
        // before clamping, without assuming min<=max whatever the
        // config brings.
        let min = Decimal::from(self.cfg.min_spread_bps) / dec!(10000);
        let max = Decimal::from(self.cfg.max_spread_bps) / dec!(10000);
        let (lo, hi) = (min.min(max), min.max(max));
        // Widen spread with vol: spread += vol_factor * sigma (both in /sec units)
        if let Some(s) = sigma {
            let extra = self.cfg.vol_factor * s * dec!(10);
            let spread = base + extra;
            return spread.clamp(lo, hi);
        }
        // The branch without sigma used to return `base` WITHOUT clamping —
        // now it also respects [min_spread_bps, max_spread_bps].
        base.clamp(lo, hi)
    }
}

impl Strategy for MarketMaking {
    fn id(&self) -> StrategyId {
        StrategyId::MarketMaking
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        if self.last_retain.elapsed() >= std::time::Duration::from_secs(30) {
            self.markets.retain(|_, s| s.clear_at > now);
            self.last_retain = Instant::now();
        }

        let mut intents = Vec::new();
        let cid = &book.market_condition_id;
        let remaining = book.secs_to_clear();
        if remaining <= 30 || remaining > 3600 {
            if let Some(st) = self.markets.get_mut(cid) {
                // Cancel live GTDs before leaving the window to avoid orphan
                // fills that MM cannot track (posted_fv is set to None below).
                if st.posted_fv.is_some() {
                    intents.push(cancel_intent(book, &book.yes.token_id));
                    intents.push(cancel_intent(book, &book.no.token_id));
                }
                // Unwind the inventory when leaving the window. Holding it to
                // settlement caused a −$716 loss in one calibration run (482 buys /
                // 52 sells): the losing side goes to 0. Sold taker at the current bid
                // except dust bids (min_unwind_bid) where there is nothing to save
                // and settlement may still pay 1.
                //
                // The in-memory inventory (`yes_shares`/
                // `no_shares`) is zeroed ONLY AFTER confirming that the
                // unwind of that leg was emitted — before it was zeroed
                // unconditionally on leaving the window, even if the unwind
                // was NEVER emitted (no bid, bid < min_unwind_bid, or
                // remaining<=0 — the block below does not even run), losing track
                // of a real position that the executor could also
                // reject later due to an exposure limit.
                if remaining > 0 && remaining <= 30 {
                    if st.yes_shares > Decimal::ZERO {
                        if let Some(bid) = book.yes.best_bid() {
                            if bid >= self.cfg.min_unwind_bid {
                                metrics::counter!("hrf_mm_unwind_total").increment(1);
                                tracing::info!(cid = %cid, token = %st.yes_token, shares = %st.yes_shares, bid = %bid,
                                    "MM: unwinding YES inventory at window close");
                                intents.push(OrderIntent {
                                    market_cid: cid.clone(),
                                    token_id: st.yes_token.clone(),
                                    side: OrderSide::Sell {
                                        size: Some(st.yes_shares),
                                    },
                                    price: bid,
                                    size: st.yes_shares,
                                    clear_at: book.clear_at,
                                    signal: Decimal::ZERO,
                                    neg_risk: book.neg_risk,
                                    limit_price: None,
                                    second_leg: None,
                                    strategy: StrategyId::MarketMaking,
                                    resting: false,
                                    real_clear_at: None,
                                });
                                st.yes_shares = Decimal::ZERO;
                            }
                            // bid < min_unwind_bid: dust, nothing to save — but
                            // NO unwind was emitted, so the inventory stays intact.
                        }
                        // no bid: no unwind emitted — inventory intact.
                    }
                    if st.no_shares > Decimal::ZERO {
                        if let Some(bid) = book.no.best_bid() {
                            if bid >= self.cfg.min_unwind_bid {
                                metrics::counter!("hrf_mm_unwind_total").increment(1);
                                tracing::info!(cid = %cid, token = %st.no_token, shares = %st.no_shares, bid = %bid,
                                    "MM: unwinding NO inventory at window close");
                                intents.push(OrderIntent {
                                    market_cid: cid.clone(),
                                    token_id: st.no_token.clone(),
                                    side: OrderSide::Sell {
                                        size: Some(st.no_shares),
                                    },
                                    price: bid,
                                    size: st.no_shares,
                                    clear_at: book.clear_at,
                                    signal: Decimal::ZERO,
                                    neg_risk: book.neg_risk,
                                    limit_price: None,
                                    second_leg: None,
                                    strategy: StrategyId::MarketMaking,
                                    resting: false,
                                    real_clear_at: None,
                                });
                                st.no_shares = Decimal::ZERO;
                            }
                        }
                    }
                }
                st.posted_fv = None;
                st.posted_bid_yes = None;
                st.posted_bid_no = None;
            }
            return intents;
        }

        // Extract everything from self BEFORE the mutable borrow of self.markets
        let sigma = vol.sigma_short.or(vol.sigma_long);
        let half_spread = self.spread_frac(sigma) / dec!(2);
        let rq_threshold = Decimal::from(self.cfg.requote_threshold_bps) / dec!(10000);
        let ttl = Duration::from_secs(self.cfg.order_ttl_secs);
        let max_inv = self.cfg.max_inventory_shares;
        let size = self.cfg.order_size_shares;

        let state = self.markets.entry(cid.clone()).or_insert_with(|| MMState {
            clear_at: book.clear_at,
            reference: book.reference_price,
            posted_fv: None,
            posted_at: None,
            posted_bid_yes: None,
            posted_bid_no: None,
            yes_shares: Decimal::ZERO,
            no_shares: Decimal::ZERO,
            yes_token: book.yes.token_id.clone(),
            no_token: book.no.token_id.clone(),
        });

        // Reference fallback with an opening guard (before it used the oracle's
        // current price with no age limit → false FV when quoting mid-bracket).
        if state.reference.is_none() {
            state.reference = book
                .reference_price
                .or_else(|| fallback_reference(vol, book, now, cid, "MM"));
        }
        let reference = match state.reference {
            Some(r) if r > Decimal::ZERO => r,
            _ => return intents,
        };

        let yes_maxed = state.yes_shares >= max_inv;
        let no_maxed = state.no_shares >= max_inv;
        if yes_maxed && no_maxed {
            // With both sides maxed there is nothing left to
            // quote, but live GTDs (if any) must be canceled here
            // just like the requote path does (below) — before, this early
            // exit returned without canceling and the GTDs stayed live,
            // able to fill above max_inventory_shares.
            if state.posted_fv.is_some() {
                intents.push(cancel_intent(book, &book.yes.token_id));
                intents.push(cancel_intent(book, &book.no.token_id));
                state.posted_fv = None;
                state.posted_at = None;
                state.posted_bid_yes = None;
                state.posted_bid_no = None;
            }
            return intents;
        }

        // Adverse selection filter — pause on a vol spike or a spread that is too tight.
        if self.cfg.vol_ratio_threshold > Decimal::ZERO {
            let vol_spike = vol
                .sigma_short
                .zip(vol.sigma_long)
                .is_some_and(|(s5, s60)| {
                    s60 > Decimal::ZERO && s5 / s60 > self.cfg.vol_ratio_threshold
                });
            if vol_spike {
                metrics::counter!("hrf_mm_paused_total", "reason" => "vol_spike").increment(1);
                return intents;
            }
        }
        if self.cfg.min_spread_to_quote > Decimal::ZERO {
            let spread = book
                .yes
                .best_ask()
                .zip(book.yes.best_bid())
                .map(|(a, b)| a - b);
            if spread.is_none_or(|s| s < self.cfg.min_spread_to_quote) {
                metrics::counter!("hrf_mm_paused_total", "reason" => "tight_spread").increment(1);
                return intents;
            }
        }

        if vol.is_stale_for_signal() {
            return intents;
        }
        let fv = match fair_value_yes(
            vol.last_price,
            reference,
            sigma.unwrap_or(dec!(0.001)),
            remaining,
        ) {
            Some(v) => v,
            None => return intents,
        };
        if fv < dec!(0.05) || fv > dec!(0.95) {
            // FV in the extreme zone: cancel stale GTDs, if any, before exiting.
            if state.posted_fv.is_some() {
                metrics::counter!("hrf_mm_skip_fv_extreme_total").increment(1);
                intents.push(cancel_intent(book, &book.yes.token_id));
                intents.push(cancel_intent(book, &book.no.token_id));
                state.posted_fv = None;
                state.posted_at = None;
                state.posted_bid_yes = None;
                state.posted_bid_no = None;
            }
            return intents;
        }

        let needs_requote = state.posted_fv.is_none() || {
            let fv_shift = state
                .posted_fv
                .is_none_or(|prev| (fv - prev).abs() > rq_threshold);
            let expired = state.posted_at.is_none_or(|at| at.elapsed() > ttl);
            fv_shift || expired
        };
        if !needs_requote {
            return intents;
        }

        // Prices are truncated to the market's REAL tick
        // (`book.tick()`, 0.01 in the mid zone / 0.001 above 0.96 or below
        // 0.04) before building the intent — `fv` carries 6 decimals and without
        // truncating the CLOB rejects the GTD for an invalid price (same pattern
        // already used by `parity_maker.rs` and `reward_farm.rs`).
        let tick = book.tick();

        // Desired state per leg: `None` = no order should remain resting on that
        // token (side maxed or no size gap up to max_inventory_shares) —
        // its live GTD is canceled as before; the tolerance never keeps
        // a leg the strategy no longer wants.
        let desired_yes = if yes_maxed {
            None
        } else {
            let bid = truncate_to_tick((fv - half_spread).max(dec!(0.01)).min(dec!(0.99)), tick);
            // Cap to the remaining gap so as not to exceed max_inventory_shares with the GTD.
            let headroom = size.min(max_inv - state.yes_shares);
            (headroom > Decimal::ZERO).then_some((bid, headroom))
        };
        let desired_no = if no_maxed {
            None
        } else {
            let bid = truncate_to_tick(
                (dec!(1) - fv - half_spread).max(dec!(0.01)).min(dec!(0.99)),
                tick,
            );
            let headroom = size.min(max_inv - state.no_shares);
            (headroom > Decimal::ZERO).then_some((bid, headroom))
        };

        // Requote tolerance (order refresh tolerance): if the bid already resting
        // on a leg is at most `requote_tolerance_ticks` ticks from the desired
        // price, that leg is NOT canceled+reposted — the resting order is already where
        // we want it and saving the cancel/ack/post cycle reduces involuntary fills
        // and pressure on the rate limit. Applies only to the requote: the
        // danger paths (window exit, extreme FV, both sides maxed) always
        // cancel, without going through here.
        let tol = Decimal::from(self.cfg.requote_tolerance_ticks) * tick;
        let tol_on = self.cfg.requote_tolerance_ticks > 0;
        let keep_yes = tol_on
            && desired_yes
                .is_some_and(|(b, _)| state.posted_bid_yes.is_some_and(|r| (b - r).abs() <= tol));
        let keep_no = tol_on
            && desired_no
                .is_some_and(|(b, _)| state.posted_bid_no.is_some_and(|r| (b - r).abs() <= tol));

        // Always cancel before reposting, but only the legs NOT
        // kept by tolerance. If only the non-maxed side is canceled, the
        // maxed side's GTD stays live on the CLOB and may keep filling
        // above max_inventory_shares.
        if state.posted_fv.is_some() {
            if !keep_yes {
                intents.push(cancel_intent(book, &book.yes.token_id));
            }
            if !keep_no {
                intents.push(cancel_intent(book, &book.no.token_id));
            }
        }
        if !keep_yes {
            state.posted_bid_yes = None;
        }
        if !keep_no {
            state.posted_bid_no = None;
        }
        if keep_yes {
            metrics::counter!("hrf_requote_skipped_total", "strategy" => "market_making")
                .increment(1);
        }
        if keep_no {
            metrics::counter!("hrf_requote_skipped_total", "strategy" => "market_making")
                .increment(1);
        }

        let mut posted_any = false;
        if let Some((bid_yes, headroom)) = desired_yes {
            if !keep_yes {
                intents.push(bid_intent(book, &book.yes.token_id, bid_yes, headroom, fv));
                state.posted_bid_yes = Some(bid_yes);
                posted_any = true;
            }
        }
        if let Some((bid_no, headroom)) = desired_no {
            if !keep_no {
                intents.push(bid_intent(book, &book.no.token_id, bid_no, headroom, fv));
                state.posted_bid_no = Some(bid_no);
                posted_any = true;
            }
        }

        // The anchors are re-anchored even if nothing was posted (pair
        // kept by tolerance): without this the FV/TTL trigger would stay
        // fired on every tick and the skip metric would spike in a loop.
        state.posted_fv = Some(fv);
        state.posted_at = Some(Instant::now());

        if posted_any {
            metrics::counter!("hrf_mm_quotes_total").increment(1);
            tracing::debug!(cid = %cid, fv = %fv, half_spread = %half_spread, "MM: requote");
        } else {
            tracing::debug!(cid = %cid, fv = %fv, "MM: requote deferred by tolerance");
        }

        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::MarketMaking {
            return;
        }
        // Fill with no state (restart with a live GTD) — the inventory cannot be rebuilt
        // without knowing which token is YES; the orphan checker settles at
        // settlement and the startup reconciliation cancels GTDs with no position.
        if !self.markets.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "MM: Buy fill without state — inventory not tracked (restart), left to the orphan checker");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "market_making").increment(1);
            }
            return;
        }
        if let Some(state) = self.markets.get_mut(&fill.market_cid) {
            match fill.side {
                FillSide::Buy => {
                    if fill.token_id == state.yes_token {
                        state.yes_shares += fill.size;
                        // The resting mark is cleared on any Buy fill
                        // (partial or full): without live-size accounting it
                        // cannot be distinguished, and keeping the mark would risk
                        // leaving a leg unquoted believing a live GTD exists.
                        // The requote reposts it anyway (the prior cancel covers
                        // the live remainder if the fill was partial).
                        state.posted_bid_yes = None;
                        metrics::counter!("hrf_mm_fill_total", "side" => "yes").increment(1);
                    } else if fill.token_id == state.no_token {
                        state.no_shares += fill.size;
                        state.posted_bid_no = None;
                        metrics::counter!("hrf_mm_fill_total", "side" => "no").increment(1);
                    }
                }
                FillSide::Sell => {
                    if fill.token_id == state.yes_token {
                        state.yes_shares = (state.yes_shares - fill.size).max(Decimal::ZERO);
                    } else if fill.token_id == state.no_token {
                        state.no_shares = (state.no_shares - fill.size).max(Decimal::ZERO);
                    }
                }
                // The intent was blocked by a gate BEFORE existing
                // as an order — its resting mark dies with it, but ONLY if
                // it points to that intent's price (if a later repost already
                // moved it to another bid, the mark belongs to the new intent and
                // is kept — never delete the mark of a potentially live order).
                FillSide::GateRejected => {
                    if fill.token_id == state.yes_token && state.posted_bid_yes == Some(fill.price)
                    {
                        state.posted_bid_yes = None;
                    } else if fill.token_id == state.no_token
                        && state.posted_bid_no == Some(fill.price)
                    {
                        state.posted_bid_no = None;
                    }
                }
                // Terminal events of an order that never got
                // filled — that leg's resting mark dies with it.
                FillSide::Rejected | FillSide::Cancelled => {
                    if fill.token_id == state.yes_token {
                        state.posted_bid_yes = None;
                    } else if fill.token_id == state.no_token {
                        state.posted_bid_no = None;
                    }
                }
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn bid_intent(
    book: &BookSnapshot,
    token_id: &str,
    price: Decimal,
    size: Decimal,
    fv: Decimal,
) -> OrderIntent {
    OrderIntent {
        market_cid: book.market_condition_id.clone(),
        token_id: token_id.to_string(),
        side: OrderSide::Buy,
        price,
        size,
        clear_at: book.clear_at,
        signal: fv,
        neg_risk: book.neg_risk,
        limit_price: Some(price),
        second_leg: None,
        strategy: StrategyId::MarketMaking,
        resting: true,
        real_clear_at: None,
    }
}

fn cancel_intent(book: &BookSnapshot, token_id: &str) -> OrderIntent {
    super::cancel_intent(
        &book.market_condition_id,
        token_id,
        book.clear_at,
        book.neg_risk,
        StrategyId::MarketMaking,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};

    fn make_book(remaining: i64, reference: Option<&str>) -> BookSnapshot {
        let level = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        BookSnapshot {
            market_condition_id: "mm-cid".into(),
            yes: TokenBook {
                token_id: "yes-t".into(),
                bids: level("0.50"),
                asks: level("0.55"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-t".into(),
                bids: level("0.45"),
                asks: level("0.48"),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-mm".into(),
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
            sigma_short: Some(dec!(0.0005)),
            sigma_long: Some(dec!(0.0005)),
            last_price: live.parse().unwrap(),
            updated_at: Instant::now(),
            ..VolSnapshot::default()
        }
    }

    fn test_cfg() -> MarketMakingConfig {
        MarketMakingConfig {
            spread_bps: 30,
            min_spread_bps: 10,
            max_spread_bps: 100,
            vol_factor: dec!(1.5),
            order_size_shares: dec!(5),
            max_inventory_shares: dec!(50),
            requote_threshold_bps: 15,
            order_ttl_secs: 60,
            ..MarketMakingConfig::default()
        }
    }

    #[test]
    fn mm_posts_initial_quotes() {
        let mut mm = MarketMaking::new(test_cfg());
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500"); // 0.5% above → FV > 0.5

        let intents = mm.on_book(&book, &vol);
        let bids: Vec<_> = intents
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(bids.len(), 2);
        assert!(bids.iter().all(|i| i.resting));
    }

    #[test]
    fn mm_skip_very_near_expiry() {
        let mut mm = MarketMaking::new(test_cfg());
        let book = make_book(20, Some("100000")); // < 30s
        let vol = make_vol("100500");
        assert!(mm.on_book(&book, &vol).is_empty());
    }

    /// `fv` carries 6 decimals — without truncating to the market tick
    /// the CLOB would reject the GTD for an invalid price (0% fill rate).
    #[test]
    fn mm_bid_truncated_to_tick() {
        let mut mm = MarketMaking::new(test_cfg());
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");

        let intents = mm.on_book(&book, &vol);
        let bids: Vec<_> = intents
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(bids.len(), 2);
        for b in bids {
            assert!(
                b.price.scale() <= book.tick().scale(),
                "the bid price must be truncated to the market tick (0.01 by default)"
            );
        }
    }

    /// `Ord::clamp` panics if min>max — `spread_frac` must
    /// not panic regardless of the order of `min_spread_bps`/`max_spread_bps`,
    /// even if the config brings them inverted.
    #[test]
    fn mm_spread_frac_does_not_panic_with_min_greater_than_max() {
        let cfg = MarketMakingConfig {
            spread_bps: 30,
            min_spread_bps: 500, // inverted on purpose (min > max)
            max_spread_bps: 50,
            vol_factor: dec!(1.5),
            ..MarketMakingConfig::default()
        };
        let mm = MarketMaking::new(cfg);
        // Must not panic in either branch (with and without sigma).
        let _ = mm.spread_frac(Some(dec!(0.001)));
        let _ = mm.spread_frac(None);
    }

    /// The branch WITHOUT sigma used to return the base spread without
    /// clamping — now it also respects [min_spread_bps, max_spread_bps].
    #[test]
    fn mm_spread_frac_branch_without_sigma_also_clamps() {
        let cfg = MarketMakingConfig {
            spread_bps: 500, // 5% — above max_spread_bps
            min_spread_bps: 10,
            max_spread_bps: 100, // 1%
            vol_factor: dec!(1.5),
            ..MarketMakingConfig::default()
        };
        let mm = MarketMaking::new(cfg);
        let spread = mm.spread_frac(None);
        assert_eq!(
            spread,
            dec!(0.01),
            "the branch without sigma must also clamp to max_spread_bps"
        );
    }

    /// With both sides at max_inventory_shares, MM must
    /// cancel the live GTDs instead of leaving them hanging — before, this early
    /// exit returned without canceling.
    #[test]
    fn mm_cancels_gtds_when_both_sides_maxed() {
        let mut mm = MarketMaking::new(test_cfg());
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");

        let intents = mm.on_book(&book, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "must post initial quotes on both sides"
        );

        let fill = |token: &str, size: &str| crate::types::Fill {
            hedge_id: None,
            market_cid: "mm-cid".into(),
            token_id: token.into(),
            exec_id: None,
            local_order_id: None,
            side: FillSide::Buy,
            price: dec!(0.50),
            size: size.parse().unwrap(),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::MarketMaking,
            clear_at: 0,
            outcome_label: None,
        };
        // Fill both sides up to max_inventory_shares (50 in test_cfg).
        mm.on_fill(&fill("yes-t", "50"));
        mm.on_fill(&fill("no-t", "50"));

        let intents2 = mm.on_book(&book, &vol);
        let cancels: Vec<_> = intents2
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Cancel))
            .collect();
        assert_eq!(
            cancels.len(),
            2,
            "with both sides maxed it must cancel the live GTDs, not leave them hanging"
        );
    }

    /// If the unwind at window close was NOT emitted (no
    /// bid available), the in-memory inventory must NOT be reset to zero —
    /// losing track of a real position leaves the orphan checker with no
    /// visibility of the pending leg.
    #[test]
    fn mm_does_not_reset_inventory_without_emitted_unwind() {
        let mut mm = MarketMaking::new(test_cfg());
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");
        let _ = mm.on_book(&book, &vol);
        mm.on_fill(&crate::types::Fill {
            hedge_id: None,
            market_cid: "mm-cid".into(),
            token_id: "yes-t".into(),
            exec_id: None,
            local_order_id: None,
            side: FillSide::Buy,
            price: dec!(0.50),
            size: dec!(10),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::MarketMaking,
            clear_at: 0,
            outcome_label: None,
        });

        // Window closing (remaining=20 <= 30) but NO bid available on YES.
        let mut book_close = book.clone();
        book_close.clear_at = crate::time::now().timestamp() + 20;
        book_close.yes.bids = HalfBook { levels: vec![] };

        let intents = mm.on_book(&book_close, &vol);
        assert!(
            intents
                .iter()
                .all(|i| !matches!(i.side, OrderSide::Sell { .. })),
            "without a bid no unwind sell must be emitted"
        );

        let state = mm
            .markets
            .get("mm-cid")
            .expect("the bracket state must exist");
        assert_eq!(
            state.yes_shares,
            dec!(10),
            "without an emitted unwind the inventory must NOT be reset to zero"
        );
    }

    // ── Requote tolerance (order refresh tolerance) ─────────────────────────

    /// With the TTL expired the requote fires on every tick; if the desired bid
    /// is within ≤ `requote_tolerance_ticks` of the resting, the leg is NOT canceled nor
    /// reposted.
    #[test]
    fn mm_tolerance_skips_requote_within_ticks() {
        let mut mm = MarketMaking::new(MarketMakingConfig {
            requote_tolerance_ticks: 1,
            order_ttl_secs: 0, // TTL=0 → needs_requote on every tick
            ..test_cfg()
        });
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");

        let first = mm.on_book(&book, &vol);
        assert_eq!(
            first
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "first quote: two resting bids"
        );

        // Same book and same vol: the desired bids are identical to the resting ones
        // (0 ticks difference ≤ tolerance) → no cancels or reposts.
        let intents = mm.on_book(&book, &vol);
        assert!(
            intents.is_empty(),
            "within tolerance it must neither cancel nor repost: {intents:?}"
        );
    }

    /// Control: with tolerance 0 (default) the same scenario cancels and reposts
    /// on every tick — the churn the tolerance avoids.
    #[test]
    fn mm_without_tolerance_always_reposts() {
        let mut mm = MarketMaking::new(MarketMakingConfig {
            order_ttl_secs: 0,
            ..test_cfg()
        });
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");
        let _ = mm.on_book(&book, &vol);

        let intents = mm.on_book(&book, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "without tolerance the requote cancels both legs even though the price did not move"
        );
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2
        );
    }

    /// Outside tolerance (> N ticks) the requote cancels and reposts normally.
    #[test]
    fn mm_tolerance_exceeded_reposts() {
        let mut mm = MarketMaking::new(MarketMakingConfig {
            requote_tolerance_ticks: 1,
            order_ttl_secs: 0,
            ..test_cfg()
        });
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");
        let _ = mm.on_book(&book, &vol);

        // The underlying jumps ~0.9% → the FV and the desired bids move several
        // ticks above the 1-tick tolerance.
        let moved_vol = make_vol("100900");
        let intents = mm.on_book(&book, &moved_vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "outside tolerance both legs are canceled and reposted"
        );
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2
        );
    }

    /// A maxed leg is NOT kept by tolerance: there is no desired bid for
    /// it, so its resting order is canceled as before — while the other
    /// leg, within tolerance, is kept.
    #[test]
    fn mm_tolerance_does_not_keep_maxed_leg() {
        let mut mm = MarketMaking::new(MarketMakingConfig {
            requote_tolerance_ticks: 1,
            order_ttl_secs: 0,
            ..test_cfg()
        });
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");
        let _ = mm.on_book(&book, &vol);

        // Fill YES up to max_inventory_shares (50 in test_cfg).
        mm.on_fill(&crate::types::Fill {
            hedge_id: None,
            market_cid: "mm-cid".into(),
            token_id: "yes-t".into(),
            exec_id: None,
            local_order_id: None,
            side: FillSide::Buy,
            price: dec!(0.50),
            size: dec!(50),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::MarketMaking,
            clear_at: 0,
            outcome_label: None,
        });

        let intents = mm.on_book(&book, &vol);
        let cancels: Vec<_> = intents
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Cancel))
            .collect();
        assert_eq!(cancels.len(), 1, "only the maxed leg is canceled");
        assert_eq!(cancels[0].token_id, "yes-t");
        assert!(
            intents.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "the leg NOT within tolerance is kept: it is not reposted"
        );
    }

    /// Danger paths intact: on leaving the window GTDs are canceled
    /// always — the tolerance only applies to the requote path.
    #[test]
    fn mm_tolerance_does_not_hold_back_cancel_at_close() {
        let mut mm = MarketMaking::new(MarketMakingConfig {
            requote_tolerance_ticks: 5, // absurd slack: even so it must save nothing
            ..test_cfg()
        });
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");
        let _ = mm.on_book(&book, &vol);

        let book_close = make_book(20, Some("100000"));
        let intents = mm.on_book(&book_close, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "the window-exit cancel never goes through the tolerance"
        );
    }

    /// `GateRejected` releases the resting mark ONLY if it points to the
    /// blocked intent's price — an ack at a stale price must not delete the
    /// mark of a later repost (it belongs to another live order).
    #[test]
    fn gate_rejected_releases_mark_only_if_price_matches() {
        let mut mm = MarketMaking::new(test_cfg());
        let book = make_book(300, Some("100000"));
        let vol = make_vol("100500");
        let intents = mm.on_book(&book, &vol);
        let yes = intents
            .iter()
            .find(|i| i.token_id == "yes-t" && matches!(i.side, OrderSide::Buy))
            .expect("must post a YES bid");
        let px = yes.price;

        let mk_fill = |price: Decimal| Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "mm-cid".into(),
            token_id: "yes-t".into(),
            side: FillSide::GateRejected,
            price,
            size: dec!(5),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::MarketMaking,
            clear_at: 0,
            outcome_label: None,
        };

        // Ack at a price that is not the intent's → the mark survives.
        mm.on_fill(&mk_fill(px + dec!(0.01)));
        assert_eq!(
            mm.markets.get("mm-cid").unwrap().posted_bid_yes,
            Some(px),
            "a GateRejected at another price must not delete the mark"
        );

        // Ack at the intent's price → the mark dies with its intent.
        mm.on_fill(&mk_fill(px));
        assert_eq!(
            mm.markets.get("mm-cid").unwrap().posted_bid_yes,
            None,
            "the intent's GateRejected releases its mark"
        );
    }
}
