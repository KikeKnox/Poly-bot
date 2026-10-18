//! Directional (DIR) strategy — Option B.
//!
//! Only enters when the ask is in [min_entry_price, max_entry_price] (default 0.95-0.97).
//! With an ask of 0.95, the maximum loss is 5¢/$ even if the bracket goes against us.
//!
//! Signal: FV > ask + fair_value_min_edge (the market undervalues the winning side).
//! FV = fair_value_yes(spot, reference, sigma, t_remaining).
//!
//! Live reactivation gate: n≥100 clean trades with WR > average ask.

use std::collections::HashMap;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::config::defaults;
use crate::{
    config::StrategyLimits,
    fair_value::{breakeven_price_parabolic, fair_value_yes},
    types::{BookSnapshot, Fill, FillSide, OrderIntent, OrderSide, StrategyId, VolSnapshot},
};

use super::{fallback_reference, Strategy, StrategyConfig};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct DirectionalConfig {
    #[serde(flatten)]
    pub limits: StrategyLimits,
    #[serde(default = "defaults::dir_min_edge")]
    pub fair_value_min_edge: Decimal,
    #[serde(default = "defaults::dir_max_entry")]
    pub max_entry_price: Decimal,
    #[serde(default = "defaults::dir_min_entry")]
    pub min_entry_price: Decimal,
    #[serde(default = "defaults::dir_decision_window")]
    pub decision_window_secs: u64,
    /// Maximum size in shares per order. 0 = no cap; the limit is max_order_usdc.
    #[serde(default)]
    pub max_size_shares: Decimal,
    /// Maximum tolerable bid-ask spread (ask − bid) on entry. 0 = no filter.
    /// Avoids entering illiquid markets where the spread reflects high uncertainty.
    #[serde(default = "defaults::dir_max_spread")]
    pub max_bid_ask_spread: Decimal,
}

impl StrategyConfig for DirectionalConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

struct DirState {
    clear_at: i64,
    reference: Option<Decimal>,
    /// true = do not enter: fill confirmed OR FOK pending confirmation.
    locked: bool,
    /// `Some(t)` = temporary lock for a pending FOK, expires at t.
    /// `None`    = fill confirmed → permanent lock (never re-enters this bracket).
    fok_until: Option<crate::time::Instant>,
}

pub struct Directional {
    cfg: DirectionalConfig,
    /// Additive haircut over `fv_side` in `find_entry`
    /// (`fair_value.confidence_haircut`, shared across strategies).
    haircut: Decimal,
    markets: HashMap<String, DirState>,
    last_retain: crate::time::Instant,
}

impl Directional {
    pub fn new(cfg: DirectionalConfig, haircut: Decimal) -> Self {
        Self {
            cfg,
            haircut,
            markets: HashMap::new(),
            last_retain: crate::time::Instant::now(),
        }
    }
}

impl Strategy for Directional {
    fn id(&self) -> StrategyId {
        StrategyId::Directional
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        if self.last_retain.elapsed() >= std::time::Duration::from_secs(30) {
            self.markets.retain(|_, s| s.clear_at > now);
            self.last_retain = crate::time::Instant::now();
        }

        let remaining = book.secs_to_clear();
        if remaining <= 0 || remaining > self.cfg.decision_window_secs as i64 {
            return vec![];
        }

        // Extract config BEFORE the mutable borrow of self.markets
        let min_entry = self.cfg.min_entry_price;
        let max_entry = self.cfg.max_entry_price;
        let edge_req = self.cfg.fair_value_min_edge;
        let order_usdc = self.cfg.limits.max_order_usdc;

        let cid = &book.market_condition_id;
        let state = self.markets.entry(cid.clone()).or_insert_with(|| DirState {
            clear_at: book.clear_at,
            reference: book.reference_price,
            locked: false,
            fok_until: None,
        });

        if state.locked {
            // If there is a temporary timeout and it already expired → FOK rejected, unlock.
            if let Some(until) = state.fok_until {
                if crate::time::Instant::now() >= until {
                    state.locked = false;
                    state.fok_until = None;
                    metrics::counter!("hrf_dir_fok_rejected_total").increment(1);
                    tracing::debug!(cid = %cid, "DIR: FOK expired without fill — reset");
                }
            }
            // If fok_until=None: fill confirmed → permanent lock, never unlock.
            return vec![];
        }

        // Reference fallback with an open guard (previously it used the oracle's current
        // price with no age limit → permanent FV≈0.5 mid-bracket).
        if state.reference.is_none() {
            state.reference = book
                .reference_price
                .or_else(|| fallback_reference(vol, book, now, cid, "DIR"));
        }
        let reference = match state.reference {
            Some(r) if r > Decimal::ZERO => r,
            _ => return vec![],
        };

        if vol.is_stale_for_signal() {
            return vec![];
        }
        let sigma = vol.sigma_long.or(vol.sigma_short);
        let fv = match fair_value_yes(
            vol.last_price,
            reference,
            sigma.unwrap_or(dec!(0.001)),
            remaining,
        ) {
            Some(v) => v,
            None => return vec![],
        };

        let max_spread = self.cfg.max_bid_ask_spread;
        let (is_yes, ask) = match find_entry(
            min_entry,
            max_entry,
            edge_req,
            max_spread,
            book,
            fv,
            self.haircut,
        ) {
            Some(v) => v,
            None => {
                metrics::counter!("hrf_dir_skip_no_entry_total").increment(1);
                return vec![];
            }
        };

        let token_id = if is_yes {
            book.yes.token_id.clone()
        } else {
            book.no.token_id.clone()
        };
        let fav_depth = if is_yes {
            book.yes.asks.best_size()
        } else {
            book.no.asks.best_size()
        };
        let mut size =
            super::size_from_caps(order_usdc, self.cfg.max_size_shares, ask, None, fav_depth);
        if size <= Decimal::ZERO {
            return vec![];
        }

        // DIR always uses FOK: cap to the available depth instead of rejecting everything.
        match fav_depth {
            None => return vec![],
            Some(d) if d < size => {
                tracing::debug!(cid = %cid, size = %size, depth = %d, "DIR: size capped to available depth for FOK");
                size = d;
                if size <= Decimal::ZERO {
                    return vec![];
                }
            }
            _ => {}
        }

        state.locked = true;
        // 10s covers the async latency of user_ws (live fills > 500ms); avoids a double entry.
        state.fok_until = Some(crate::time::Instant::now() + std::time::Duration::from_secs(10));

        metrics::counter!("hrf_dir_signals_total").increment(1);
        tracing::info!(
            cid = %cid, is_yes, ask = %ask, fv = %fv,
            edge = %(fv - ask), "DIR: entry Option B"
        );

        vec![OrderIntent {
            market_cid: cid.clone(),
            token_id,
            side: OrderSide::Buy,
            price: ask,
            size,
            clear_at: book.clear_at,
            signal: fv - ask,
            neg_risk: book.neg_risk,
            limit_price: Some(ask),
            second_leg: None,
            strategy: StrategyId::Directional,
            resting: false,
            real_clear_at: None,
        }]
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::Directional {
            return;
        }
        // Buy fill without state (restart mid-bracket) — DIR does not manage the
        // position after the fill (hold to settlement), so making it
        // visible is enough; the orphan checker will settle it at settlement.
        if !self.markets.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "DIR: Buy fill without state — position left to the orphan checker (restart mid-bracket)");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "directional").increment(1);
            }
            return;
        }
        if let Some(state) = self.markets.get_mut(&fill.market_cid) {
            // fok_until=None with locked=true = permanent lock; never re-enters this bracket.
            state.locked = true;
            state.fok_until = None;
            if matches!(fill.side, FillSide::Buy) {
                metrics::counter!("hrf_dir_fill_total").increment(1);
            }
        }
    }
}

/// Returns (is_yes, ask) if some side has an ask in [min_entry, max_entry],
/// FV_side − haircut > ask + min_edge + fee, and the bid-ask spread ≤ max_spread
/// (0 = no filter).
///
/// `haircut` is the additive confidence term — it ALWAYS subtracts in
/// this BUY gate (moves the decision toward not trading). It is applied to
/// the oriented `fv_side` (`fv` for YES, `1 − fv` for NO), one side at a time.
fn find_entry(
    min: Decimal,
    max: Decimal,
    edge_req: Decimal,
    max_spread: Decimal,
    book: &BookSnapshot,
    fv: Decimal,
    haircut: Decimal,
) -> Option<(bool, Decimal)> {
    let spread_ok = |ask: Decimal, bid: Option<Decimal>| -> bool {
        max_spread == Decimal::ZERO || bid.is_some_and(|b| ask - b <= max_spread)
    };
    // The edge must also cover the CLOB taker fee
    // (previously nominal) — with the market's real PARABOLIC rate
    // (`book.parabolic_fee_rate`, Gamma's `feeSchedule.rate`) if
    // discovery brought it, otherwise the TAKER_FEE_RATE_PARABOLIC constant. Previously
    // the triangular model with `fee_rate_bps` was used (a source deprecated since
    // FEE-2), which overestimates the fee 1.5-2.9x and discards truly EV+ entries.
    if let Some(yes_ask) = book.yes.best_ask() {
        let threshold = breakeven_price_parabolic(book, yes_ask) + edge_req;
        let passes_without_haircut = fv > threshold;
        let passes_with_haircut = (fv - haircut) > threshold;
        if yes_ask >= min
            && yes_ask <= max
            && passes_with_haircut
            && spread_ok(yes_ask, book.yes.best_bid())
        {
            return Some((true, yes_ask));
        }
        if yes_ask >= min
            && yes_ask <= max
            && passes_without_haircut
            && !passes_with_haircut
            && spread_ok(yes_ask, book.yes.best_bid())
        {
            metrics::counter!("hrf_fv_haircut_block_total", "strategy" => "directional")
                .increment(1);
        }
    }
    let fv_no = dec!(1) - fv;
    if let Some(na) = book.no.best_ask() {
        let threshold = breakeven_price_parabolic(book, na) + edge_req;
        let passes_without_haircut = fv_no > threshold;
        let passes_with_haircut = (fv_no - haircut) > threshold;
        if na >= min && na <= max && passes_with_haircut && spread_ok(na, book.no.best_bid()) {
            return Some((false, na));
        }
        if na >= min
            && na <= max
            && passes_without_haircut
            && !passes_with_haircut
            && spread_ok(na, book.no.best_bid())
        {
            metrics::counter!("hrf_fv_haircut_block_total", "strategy" => "directional")
                .increment(1);
        }
    }
    None
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
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        // bid = ask - 0.02 so the spread is narrow by default in the tests
        let bids = |p: &str| {
            let ask: Decimal = p.parse().unwrap();
            HalfBook {
                levels: vec![Level {
                    price: ask - dec!(0.02),
                    size: dec!(50),
                }],
            }
        };
        BookSnapshot {
            market_condition_id: "dir-cid".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids(yes_ask),
                asks: asks(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids(no_ask),
                asks: asks(no_ask),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-dir".into(),
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
            sigma_short: Some(dec!(0.0001)),
            sigma_long: Some(dec!(0.0001)),
            last_price: live.parse().unwrap(),
            updated_at: crate::time::Instant::now(),
            ..VolSnapshot::default()
        }
    }

    fn test_cfg() -> DirectionalConfig {
        DirectionalConfig {
            fair_value_min_edge: dec!(0.03),
            max_entry_price: dec!(0.97),
            min_entry_price: dec!(0.95),
            decision_window_secs: 60,
            ..DirectionalConfig::default()
        }
    }

    #[test]
    fn dir_enters_with_higher_fv() {
        let mut dir = Directional::new(test_cfg(), Decimal::ZERO);
        // reference=100000, live=103000 → spot/strike=1.03 → Φ(ln(1.03)/tiny) ≈ 1.0 → FV YES ≈ 1
        // yes_ask=0.96 ∈ [0.95, 0.97], FV >> 0.96 + 0.03 → must enter YES
        let book = make_book("0.96", "0.04", 45, Some("100000"));
        let vol = make_vol("103000");
        let intents = dir.on_book(&book, &vol);
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].token_id, "y");
    }

    #[test]
    fn dir_skip_ask_out_of_range() {
        let mut dir = Directional::new(test_cfg(), Decimal::ZERO);
        // ask=0.80 < min_entry=0.95 → does not enter
        let book = make_book("0.80", "0.20", 45, Some("100000"));
        let vol = make_vol("103000");
        assert!(dir.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn dir_skip_outside_window() {
        let mut dir = Directional::new(test_cfg(), Decimal::ZERO);
        let book = make_book("0.96", "0.04", 300, Some("100000")); // > 60s
        let vol = make_vol("103000");
        assert!(dir.on_book(&book, &vol).is_empty());
    }

    #[test]
    fn dir_skip_spread_too_wide() {
        // max_bid_ask_spread = 0.05; the helper creates spread=0.02 by default.
        // We force a wide spread by modifying the book manually.
        let cfg = DirectionalConfig {
            fair_value_min_edge: dec!(0.03),
            max_entry_price: dec!(0.97),
            min_entry_price: dec!(0.95),
            decision_window_secs: 60,
            max_bid_ask_spread: dec!(0.05),
            ..DirectionalConfig::default()
        };
        let mut dir = Directional::new(cfg, Decimal::ZERO);
        // ask=0.96, bid=0.85 → spread=0.11 > max_bid_ask_spread=0.05 → skip
        let asks = crate::types::HalfBook {
            levels: vec![crate::types::Level {
                price: dec!(0.96),
                size: dec!(50),
            }],
        };
        let bids = crate::types::HalfBook {
            levels: vec![crate::types::Level {
                price: dec!(0.85),
                size: dec!(50),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "dir-spread-cid".into(),
            yes: crate::types::TokenBook {
                token_id: "y".into(),
                bids: bids.clone(),
                asks: asks.clone(),
                ..Default::default()
            },
            no: crate::types::TokenBook {
                token_id: "n".into(),
                bids,
                asks,
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + 45,
            neg_risk: false,
            slug: "btc-updown-5m-spread".into(),
            reference_price: Some(dec!(100000)),
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        };
        let vol = make_vol("103000");
        assert!(
            dir.on_book(&book, &vol).is_empty(),
            "must skip a market with a wide spread"
        );
    }

    /// With a nominal edge of 3pp (< haircut 4pp) DIR must NOT
    /// enter; with a nominal edge of 5pp (> haircut) it MUST enter. `edge_req=0`
    /// isolates the haircut gate; `parabolic_fee_rate` (parabolic
    /// model, not the triangular `fee_rate_bps`) is derived so the threshold
    /// (`ask + fee`) falls exactly at `fv - 0.03`/`-0.05`.
    #[test]
    fn dir_haircut_blocks_edge_3pp_and_lets_edge_5pp_through() {
        let cfg = DirectionalConfig {
            fair_value_min_edge: Decimal::ZERO,
            max_entry_price: dec!(0.97),
            min_entry_price: dec!(0.50),
            decision_window_secs: 60,
            ..DirectionalConfig::default()
        };
        // make_vol sets sigma=0.0001; reference=100000 via the book.
        let live = dec!(103000);
        let reference = dec!(100000);
        let sigma = dec!(0.0001);
        let remaining = 45i64;
        let fv = fair_value_yes(live, reference, sigma, remaining).expect("fv must be computable");

        let ask = dec!(0.80);
        let haircut = dec!(0.04);

        // Inverse of fee_taker_parabolic_with_rate: threshold = ask + rate*ask*(1-ask)
        // → rate = (threshold - ask) / (ask*(1-ask)).
        let rate_for_threshold =
            |target: Decimal| -> Decimal { (target - ask) / (ask * (Decimal::ONE - ask)) };

        let vol = make_vol("103000");

        // Case 1: nominal edge 3pp < haircut 4pp → must BLOCK.
        let mut book_3pp = make_book("0.80", "0.19", remaining, Some("100000"));
        book_3pp.parabolic_fee_rate = Some(rate_for_threshold(fv - dec!(0.03)));
        let mut dir_3pp = Directional::new(cfg.clone(), haircut);
        assert!(
            dir_3pp.on_book(&book_3pp, &vol).is_empty(),
            "nominal edge 3pp < haircut 4pp — DIR must not enter"
        );

        // Case 2: nominal edge 5pp > haircut 4pp → must PASS.
        let mut book_5pp = make_book("0.80", "0.19", remaining, Some("100000"));
        book_5pp.parabolic_fee_rate = Some(rate_for_threshold(fv - dec!(0.05)));
        let mut dir_5pp = Directional::new(cfg, haircut);
        assert_eq!(
            dir_5pp.on_book(&book_5pp, &vol).len(),
            1,
            "nominal edge 5pp > haircut 4pp — DIR must enter"
        );
    }

    /// With the market's real 4% fee (parabolic, `finance_prices_fees`)
    /// an entry that the triangular model with a 10% fee would block by overestimating
    /// the fee must PASS with the correct parabolic model.
    #[test]
    fn dir_uses_market_parabolic_fee() {
        let cfg = DirectionalConfig {
            fair_value_min_edge: Decimal::ZERO,
            max_entry_price: dec!(0.97),
            min_entry_price: dec!(0.50),
            decision_window_secs: 60,
            ..DirectionalConfig::default()
        };
        let vol = make_vol("103000");

        // fv ≈ 1 with these parameters (live far above reference, small
        // sigma) — we use an ask very close to the limit so the fee decides.
        let mut book_without_fee_real = make_book("0.96", "0.03", 45, Some("100000"));
        book_without_fee_real.parabolic_fee_rate = None; // falls back to TAKER_FEE_RATE_PARABOLIC=0.07
        let mut dir_fallback = Directional::new(cfg.clone(), Decimal::ZERO);
        let intents_fallback = dir_fallback.on_book(&book_without_fee_real, &vol);
        assert_eq!(
            intents_fallback.len(),
            1,
            "with fv≈1 even the 0.07 fallback must pass"
        );

        // With a real rate of 0.04 the threshold is lower — the same entry must still pass.
        let mut book_fee_real = book_without_fee_real.clone();
        book_fee_real.parabolic_fee_rate = Some(dec!(0.04));
        let mut dir_real = Directional::new(cfg, Decimal::ZERO);
        assert_eq!(
            dir_real.on_book(&book_fee_real, &vol).len(),
            1,
            "with a real fee of 0.04 the entry must also pass (even looser threshold)"
        );
    }
}
