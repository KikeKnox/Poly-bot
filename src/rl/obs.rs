//! Observation, mask and action translation for the `policy_v1` contract.
//!
//! ONE implementation shared by the runtime strategy
//! (`crate::strategy::rl_predictor`) and the offline driver
//! (`src/bin/rl_policy_env.rs`): the contract math lives here so that
//! there is no train/serve skew. The replay produces exactly the same
//! features, masks and limits as the online runtime.

use rust_decimal::Decimal;

use crate::fair_value::fee_charged_5dp;
use crate::state::to_micros;
use crate::types::{BookSnapshot, Level, TokenBook};

use super::contract::{
    action_kind, ActionKind, ActionMask, TokenSide, BUY_NO_FULL, BUY_NO_SMALL, BUY_YES_FULL,
    BUY_YES_SMALL, MASK_HOLD_ONLY, NUM_FEATURES, SELL_NO_ALL, SELL_NO_HALF, SELL_YES_ALL,
    SELL_YES_HALF,
};
use super::episode::EpisodeLedger;

/// Shared operational parameters (a subset of `RlPredictorConfig` that the
/// contract math needs; the driver receives them via CLI with the same
/// defaults as the runtime).
#[derive(Debug, Clone)]
pub struct PolicyTunables {
    /// Margin of the protective limit over the ask/bid (USDC/share).
    pub slippage_margin: Decimal,
    /// Minimum executable size (shares).
    pub min_order_shares: Decimal,
    /// Notional cap per buy action (USDC).
    pub max_order_usdc: Decimal,
    /// ENTRY window in remaining seconds (SELL does not use it).
    pub min_remaining_secs: i64,
    pub max_remaining_secs: i64,
    /// Maximum tolerable spread on the bought leg (0 = no filter).
    pub max_spread: Decimal,
}

/// Minimal view of the episode's market state. The strategy builds it from
/// `MarketState`; the driver builds it from its own `to_record`. Token
/// identity ALWAYS comes from the market metadata.
pub struct MarketView<'a> {
    pub episode: &'a EpisodeLedger,
    pub yes_token: &'a str,
    pub no_token: &'a str,
    /// Parabolic fee rate of the market (fraction, e.g. 0.07).
    pub fee_rate: Decimal,
    /// Order in flight without a confirmed resolution.
    pub pending: bool,
    /// Uncertain state (expired pending order, not reconciled).
    pub uncertain: bool,
}

/// Visible depth in USDC within the `±slippage_margin` band of the best
/// level (visible depth within the slippage limit).
pub fn depth_usdc(levels: &[Level], band: Decimal) -> Decimal {
    let Some(best) = levels.first().map(|l| l.price) else {
        return Decimal::ZERO;
    };
    levels
        .iter()
        .take_while(|l| (l.price - best).abs() <= band)
        .fold(Decimal::ZERO, |acc, l| acc + l.price * l.size)
}

/// Buy size in shares: `min(max_order_usdc, free_cash) / unit_cost`
/// truncated to 2dp (the CLOB quantum). `small` = half of FULL.
/// `unit_cost = limit + parabolic_fee(limit)`: all-in cost per share.
pub fn buy_size_shares(
    t: &PolicyTunables,
    ep: &EpisodeLedger,
    ask: Decimal,
    rate: Decimal,
    small: bool,
) -> Decimal {
    let limit = ask + t.slippage_margin;
    if limit >= Decimal::ONE || rate < Decimal::ZERO {
        return Decimal::ZERO;
    }
    let unit_cost = limit + rate * limit * (Decimal::ONE - limit);
    if unit_cost <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    let budget = t.max_order_usdc.min(ep.free_cash());
    if budget <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    let full = (budget / unit_cost).trunc_with_scale(2);
    if small {
        (full / Decimal::TWO).trunc_with_scale(2)
    } else {
        full
    }
}

/// Buy gates per leg: valid ask and bounded spread on that leg.
fn buy_leg_ok(t: &PolicyTunables, leg: &TokenBook) -> bool {
    let Some(ask) = leg.best_ask() else {
        return false;
    };
    if ask <= Decimal::ZERO || ask >= Decimal::ONE {
        return false;
    }
    if t.max_spread > Decimal::ZERO {
        let bid = leg.best_bid().unwrap_or(Decimal::ZERO);
        if ask - bid > t.max_spread {
            return false;
        }
    }
    true
}

/// Legal mask (identity by token_id, never by item order).
/// `p_usable` = the predictor produced a valid `p_yes` for this book.
/// `boundary` = cohort boundary closed to new entries (always `false` in
/// the offline driver, where the trainer cuts the cohorts).
pub fn mask_legal(
    t: &PolicyTunables,
    v: &MarketView<'_>,
    book: &BookSnapshot,
    p_usable: bool,
    boundary: bool,
) -> ActionMask {
    let mut mask = MASK_HOLD_ONLY;
    if v.uncertain || v.pending {
        return mask; // uncertain state or order in flight: HOLD only
    }
    let secs_left = book.secs_to_clear();
    if secs_left <= 0 {
        return mask; // market closed: no further decisions
    }

    // SELL: own inventory + book of the side with a bid. Independent of
    // the predictor (`p_missing` serves SELL without a loaded model) and of the
    // cohort boundary ("never block SELL of existing positions").
    for (side, a_half, a_all) in [
        (TokenSide::Yes, SELL_YES_HALF, SELL_YES_ALL),
        (TokenSide::No, SELL_NO_HALF, SELL_NO_ALL),
    ] {
        let (token, leg) = match side {
            TokenSide::Yes => (v.yes_token, &book.yes),
            TokenSide::No => (v.no_token, &book.no),
        };
        if token.is_empty() {
            continue;
        }
        let shares = v.episode.shares_of(token);
        if shares >= t.min_order_shares && leg.best_bid().is_some() {
            let half = (shares / Decimal::TWO).trunc_with_scale(2);
            // HALF is invalid if the rounded half does not reach the minimum, or
            // if the remainder falls below it without being zero.
            mask[a_half] = half >= t.min_order_shares
                && (shares - half == Decimal::ZERO || shares - half >= t.min_order_shares);
            mask[a_all] = true;
        }
    }

    // BUY: usable predictor, entry window and cohort boundary closed to new
    // entries (time gate at the boundary; SELL is not affected).
    if !p_usable
        || boundary
        || secs_left < t.min_remaining_secs
        || secs_left > t.max_remaining_secs
    {
        return mask;
    }
    for (side, a_small, a_full) in [
        (TokenSide::Yes, BUY_YES_SMALL, BUY_YES_FULL),
        (TokenSide::No, BUY_NO_SMALL, BUY_NO_FULL),
    ] {
        let leg = match side {
            TokenSide::Yes => &book.yes,
            TokenSide::No => &book.no,
        };
        if let Some(ask) = leg.best_ask().filter(|_| buy_leg_ok(t, leg)) {
            mask[a_small] =
                buy_size_shares(t, v.episode, ask, v.fee_rate, true) >= t.min_order_shares;
            mask[a_full] =
                buy_size_shares(t, v.episode, ask, v.fee_rate, false) >= t.min_order_shares;
        }
    }
    mask
}

/// Observation of 22 features (contractual order `FEATURE_NAMES`).
/// `None` if any feature is not finite: the whole observation is invalidated,
/// never imputed.
pub fn build_obs(
    t: &PolicyTunables,
    v: &MarketView<'_>,
    book: &BookSnapshot,
    p_yes: Option<f64>,
    book_age_secs: f64,
) -> Option<[f64; NUM_FEATURES]> {
    use rust_decimal::prelude::ToPrimitive;
    let f = |d: Decimal| d.to_f64().unwrap_or(f64::NAN);
    let d2f = |o: Option<Decimal>| o.map(f).unwrap_or(0.0);
    let yes_bid = book.yes.best_bid();
    let yes_ask = book.yes.best_ask();
    let no_bid = book.no.best_bid();
    let no_ask = book.no.best_ask();
    let band = t.slippage_margin;
    let obs = [
        p_yes.unwrap_or(0.5),                                        // p_yes
        if p_yes.is_none() { 1.0 } else { 0.0 },                     // p_missing
        d2f(yes_bid),                                                // yes_bid
        d2f(yes_ask),                                                // yes_ask
        d2f(no_bid),                                                 // no_bid
        d2f(no_ask),                                                 // no_ask
        f(depth_usdc(&book.yes.bids.levels, band)),                  // yes_bid_depth_usdc
        f(depth_usdc(&book.yes.asks.levels, band)),                  // yes_ask_depth_usdc
        f(depth_usdc(&book.no.bids.levels, band)),                   // no_bid_depth_usdc
        f(depth_usdc(&book.no.asks.levels, band)),                   // no_ask_depth_usdc
        (book.secs_to_clear() as f64 / 300.0).clamp(0.0, 1.0),       // time_remaining_frac
        f(v.episode.shares_of(v.yes_token)),                         // yes_position_shares
        f(v.episode.shares_of(v.no_token)),                          // no_position_shares
        f(v.episode.cost_of(v.yes_token)),                           // yes_cost_basis_usdc
        f(v.episode.cost_of(v.no_token)),                            // no_cost_basis_usdc
        f(v.episode.free_cash()),                                    // free_budget_usdc
        f(v.episode.reserved()),                                     // reserved_budget_usdc
        if v.pending { 1.0 } else { 0.0 },                           // pending_order
        if yes_bid.is_none() && yes_ask.is_none() { 1.0 } else { 0.0 }, // yes_book_missing
        if no_bid.is_none() && no_ask.is_none() { 1.0 } else { 0.0 },   // no_book_missing
        book_age_secs,                                               // book_age_seconds
        f(v.fee_rate),                                               // buy_fee_rate
    ];
    obs.iter().all(|v| v.is_finite()).then_some(obs)
}

/// Parameters of an `OrderIntent` translated from a contractual action: the
/// pure part of `emit_action`. It does not reserve or set `pending`; the
/// caller does that, since it owns the episode state.
#[derive(Debug, Clone)]
pub struct IntentSpec {
    pub action: usize,
    pub token_id: String,
    /// `true` = buy, `false` = sell.
    pub buy: bool,
    /// Observed price (ask for buys, bid for sells): the intent's `price`.
    pub price: Decimal,
    pub size: Decimal,
    /// Protective limit truncated to the tick (the intent's `limit_price`).
    pub limit: Decimal,
    /// Estimated all-in reserve in micros (BUY only; 0 for SELL).
    pub reserve_um: i64,
}

/// Translates `action` (0..8, contractual) into the intent parameters, with
/// the SAME size and limit semantics as the runtime: buy at `ask + slippage`
/// with size from the budget; sell at `bid − slippage` with HALF/ALL of the
/// own inventory. `None` = HOLD, or an action that is no longer
/// executable (no ask/bid, or size below the minimum).
pub fn action_params(
    t: &PolicyTunables,
    v: &MarketView<'_>,
    book: &BookSnapshot,
    action: usize,
) -> Option<IntentSpec> {
    // The tick can arrive malformed from the feed (`tick_size_change` with "0"
    // or a negative value is accepted without validation at the boundary).
    // Dividing by it panics in `rust_decimal`, or yields garbage limits when
    // tick < 0. `None` = action not executable; the caller treats it as a safe HOLD.
    let tick = book.tick();
    if tick <= Decimal::ZERO {
        return None;
    }
    match action_kind(action)? {
        ActionKind::Hold => None,
        ActionKind::Buy { side, small } => {
            let (token, leg) = match side {
                TokenSide::Yes => (v.yes_token, &book.yes),
                TokenSide::No => (v.no_token, &book.no),
            };
            let ask = leg.best_ask()?;
            let size = buy_size_shares(t, v.episode, ask, v.fee_rate, small);
            if size < t.min_order_shares {
                return None;
            }
            // Protective limit: ask + margin, truncated to the tick, < 1.
            let limit = ((ask + t.slippage_margin) / tick).floor() * tick;
            let limit = limit.min(Decimal::ONE - tick).max(tick);
            // All-in reserve (notional + fee estimated at the limit).
            let reserve_um =
                to_micros(limit * size + fee_charged_5dp(v.fee_rate, limit, size));
            Some(IntentSpec {
                action,
                token_id: token.to_string(),
                buy: true,
                price: ask,
                size,
                limit,
                reserve_um,
            })
        }
        ActionKind::Sell { side, half } => {
            let (token, leg) = match side {
                TokenSide::Yes => (v.yes_token, &book.yes),
                TokenSide::No => (v.no_token, &book.no),
            };
            let bid = leg.best_bid()?;
            let owned = v.episode.shares_of(token);
            let size = if half {
                (owned / Decimal::TWO).trunc_with_scale(2)
            } else {
                owned
            };
            if size < t.min_order_shares {
                return None;
            }
            // Protective sell limit = bid − margin (minimum acceptable),
            // truncated to the tick, with a floor of one tick.
            let limit = ((bid - t.slippage_margin) / tick).floor() * tick;
            let limit = limit.max(tick);
            Some(IntentSpec {
                action,
                token_id: token.to_string(),
                buy: false,
                price: bid,
                size,
                limit,
                reserve_um: 0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::episode::TokenPosition;
    use crate::types::HalfBook;
    use rust_decimal_macros::dec;
    use std::collections::HashMap;

    fn tunables() -> PolicyTunables {
        PolicyTunables {
            slippage_margin: dec!(0.01),
            min_order_shares: dec!(1),
            max_order_usdc: dec!(10),
            min_remaining_secs: 30,
            max_remaining_secs: 240,
            max_spread: dec!(0.05),
        }
    }

    fn book_with_tick(tick: Option<Decimal>) -> BookSnapshot {
        let lvl = |p: Decimal| HalfBook {
            levels: vec![Level {
                price: p,
                size: dec!(50),
            }],
        };
        let now = crate::time::now().timestamp();
        BookSnapshot {
            market_condition_id: "cid-obs".into(),
            yes: TokenBook {
                token_id: "tok-yes".into(),
                bids: lvl(dec!(0.49)),
                asks: lvl(dec!(0.50)),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-no".into(),
                bids: lvl(dec!(0.49)),
                asks: lvl(dec!(0.50)),
                ..Default::default()
            },
            market_open_ts: now - 180,
            clear_at: now + 120,
            neg_risk: false,
            slug: "btc-updown-5m-x".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: Some(dec!(0.07)),
            tick_size: tick,
            recv_instant: None,
        }
    }

    fn view_with_position(ep: &EpisodeLedger) -> MarketView<'_> {
        MarketView {
            episode: ep,
            yes_token: "tok-yes",
            no_token: "tok-no",
            fee_rate: dec!(0.07),
            pending: false,
            uncertain: false,
        }
    }

    /// `tick_size = Some(0)` (accepted without validation by the feed):
    /// `action_params` divided by the tick and panicked in `rust_decimal`.
    /// With the guard in place, BUY and SELL return `None` (safe HOLD).
    #[test]
    fn zero_tick_does_not_panic_buy_or_sell() {
        let book = book_with_tick(Some(Decimal::ZERO));
        // BUY: enough free cash to reach the division by tick.
        let ep_buy = EpisodeLedger::new("cid-obs", dec!(10), 0);
        let v = view_with_position(&ep_buy);
        assert!(
            action_params(&tunables(), &v, &book, BUY_YES_FULL).is_none(),
            "tick=0 must return None (safe HOLD), not panic"
        );
        // SELL: own inventory to reach the division by tick.
        let mut pos = HashMap::new();
        pos.insert(
            "tok-yes".to_string(),
            TokenPosition {
                size_um: 10_000_000,
                cost_um: 5_000_000,
            },
        );
        let ep_sell = EpisodeLedger::recovered("cid-obs", dec!(10), pos, 0);
        let v = view_with_position(&ep_sell);
        assert!(
            action_params(&tunables(), &v, &book, SELL_YES_ALL).is_none(),
            "tick=0 must return None (safe HOLD), not panic"
        );
    }

    /// A negative tick must not divide either: without the guard, `floor` over
    /// a negative divisor silently produced garbage limits.
    #[test]
    fn negative_tick_does_not_panic_buy_or_sell() {
        let book = book_with_tick(Some(dec!(-0.01)));
        let ep_buy = EpisodeLedger::new("cid-obs", dec!(10), 0);
        let v = view_with_position(&ep_buy);
        assert!(action_params(&tunables(), &v, &book, BUY_YES_FULL).is_none());
        let mut pos = HashMap::new();
        pos.insert(
            "tok-yes".to_string(),
            TokenPosition {
                size_um: 10_000_000,
                cost_um: 5_000_000,
            },
        );
        let ep_sell = EpisodeLedger::recovered("cid-obs", dec!(10), pos, 0);
        let v = view_with_position(&ep_sell);
        assert!(action_params(&tunables(), &v, &book, SELL_YES_ALL).is_none());
    }

    /// Regression for the healthy path: a valid tick still produces intents.
    #[test]
    fn valid_tick_keeps_translating_actions() {
        let book = book_with_tick(Some(dec!(0.01)));
        let ep = EpisodeLedger::new("cid-obs", dec!(10), 0);
        let v = view_with_position(&ep);
        let spec = action_params(&tunables(), &v, &book, BUY_YES_FULL)
            .expect("tick healthy + ask present must produce a BUY intent");
        assert!(spec.buy && spec.size >= dec!(1));
    }
}
