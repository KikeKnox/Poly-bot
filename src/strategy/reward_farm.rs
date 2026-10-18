//! RewardFarm strategy: scan-only mode plus real quote posting.
//!
//! Polymarket pays out daily (pool in pUSD) to whoever keeps limit orders
//! within the scoring band around the mid (score = `((v−s)/v)² · size`; two-sided
//! quoting is mandatory if the mid leaves 0.10–0.90; it pays even if the order
//! is never filled). Scan-only mode measures the pool and the competition before
//! risking a single dollar; it keeps working exactly as before (it is the G0 gate).
//!
//! Quote posting adds the real orders: **RF is a FILL-AVOIDANCE strategy**,
//! not a spread-capture one. The 20% taker-fee rebate is peanuts
//! next to the cost of a toxic fill (markout measured on PM: −1326 bps), so
//! the WHOLE design is subordinated to the orders NOT getting filled:
//! placement at the EDGE of the band (minimum score, minimum fill risk, behind
//! the existing queue if there is one), a single level per side, requote only
//! when the mid moves enough, immediate withdrawal on toxic-flow signals
//! (CVD/liquidations/vol-ratio from `VolSnapshot`, the same machinery PM
//! already uses), and a bracket BURNED after the first fill (do not reopen: the
//! lesson from PM is that all the loss comes from re-entering after a dump).
//!
//! Posting has TWO keys: (1) `enabled=false` (StrategyLimits): the strategy is
//! not even spawned unless explicitly enabled; (2) a hot G0 gate, per bracket:
//! without a confirmed `rewards_daily_rate` > 0 for that duration, RF cancels
//! and posts nothing (it does not farm what does not pay).
//!
//! Design note: the rewards config (min_size/max_spread/daily_rate) does NOT
//! travel in `BookSnapshot`; it is read directly from the shared
//! `Arc<MarketRegistry>` by `market_condition_id` (an alternative the spec
//! explicitly allows when the scan code leaves the rewards only in
//! `MarketEntry`). This avoids touching `types.rs`, `market_feed/handle.rs` and
//! `resync.rs`.

use crate::time::Instant;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    config::defaults,
    coordination::BracketLedger,
    fair_value::fair_value_yes,
    market_feed::registry::MarketRegistry,
    types::{
        BookSnapshot, Fill, FillSide, Level, OrderIntent, OrderSide, StrategyId, TokenBook,
        VolSnapshot,
    },
};

use super::Strategy;

// ── RewardFarm configuration ─────────────────────────────────────────────────

/// Configuration of the RewardFarm strategy (scan plus quote posting).
///
/// Before posting a single GTD for the Liquidity Rewards Program, the pool and
/// the competition must be measured (scan-only mode keeps working the same;
/// it is the G0 gate). Posting adds real GTD bids at the edge of the scoring
/// band, conditional on the scan confirming `rewards_daily_rate > 0` for the
/// bracket duration. Posting is never activated if `enabled=false`
/// (StrategyLimits, process-level gate) NOR if the rate is not confirmed (hot
/// G0 gate, per bracket).
#[derive(Debug, Clone, Deserialize)]
pub struct RewardFarmConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    /// Minimum interval between `rf_scan` logs for the same market (seconds).
    #[serde(default = "defaults::rf_scan_log_interval_secs")]
    pub scan_log_interval_secs: u64,

    // ── Market selection ────────────────────────────────────────────────────
    /// Nominal bracket durations allowed for farming. NEVER "5m"/
    /// "hourly" by default: the mid moves too fast there and the
    /// band edge stops protecting against toxic fills.
    #[serde(default = "defaults::rf_durations")]
    pub rf_durations: Vec<String>,
    /// Simultaneous brackets (also limited by `rf_max_total_usdc` and the global
    /// `max_total_exposure_usdc` cap).
    #[serde(default = "defaults::rf_max_markets")]
    pub rf_max_markets: u32,
    /// Minimum pool rate in $/day to enter. `rewards_daily_rate` absent or
    /// below this ⇒ the G0 gate fails, RF cancels and posts nothing (do not
    /// farm what does not pay).
    #[serde(default)]
    pub rf_min_daily_rate: Decimal,
    /// Do not farm with mid < this: two-sided quoting is mandatory outside the band.
    #[serde(default = "defaults::rf_mid_min")]
    pub rf_mid_min: Decimal,
    /// Do not farm with mid > this.
    #[serde(default = "defaults::rf_mid_max")]
    pub rf_mid_max: Decimal,

    // ── Quoting policy ───────────────────────────────────────────────────────
    /// Inside the band edge: `s_target = rewards_max_spread − this`.
    #[serde(default = "defaults::rf_edge_buffer_cents")]
    pub rf_edge_buffer_cents: Decimal,
    /// Requote if the mid moves at least this much (cents) since the last post.
    #[serde(default = "defaults::rf_requote_threshold_cents")]
    pub rf_requote_threshold_cents: Decimal,
    /// Place behind the existing queue inside the band.
    #[serde(default = "defaults::rf_queue_behind")]
    pub rf_queue_behind: bool,
    /// One level per side, never a ladder (v1 only implements this; the flag
    /// documents the design invariant and does not enable an alternative mode).
    #[serde(default = "defaults::rf_single_level")]
    pub rf_single_level: bool,

    // ── Sizing / capital ─────────────────────────────────────────────────────
    /// Only mode implemented in v1: `order_size_shares = rewards_min_size` exactly.
    #[serde(default = "defaults::rf_size_mode")]
    pub rf_size_mode: String,
    /// RF's own sub-cap (USDC committed in live GTDs) within the global
    /// `max_total_exposure_usdc` cap, so that RF does not monopolize it.
    #[serde(default = "defaults::rf_max_total_usdc")]
    pub rf_max_total_usdc: Decimal,
    /// Bracket exclusivity versus PM/MM: if another strategy already has
    /// exposure in the bracket, RF skips it.
    #[serde(default = "defaults::rf_require_no_pm_mm")]
    pub rf_require_no_pm_mm: bool,

    // ── Toxic-flow withdrawal ────────────────────────────────────────────────
    /// `|cvd_ratio_60s|` that condemns a leg.
    #[serde(default = "defaults::rf_cvd_toxic")]
    pub rf_cvd_toxic: Decimal,
    /// `|liq_net_60s|` USDT that condemns a leg. 0 = disabled until calibrated.
    #[serde(default)]
    pub rf_liq_toxic_notional: Decimal,
    /// σ5/σ60 that pauses BOTH legs (volatility spike).
    #[serde(default = "defaults::rf_vol_ratio_pause")]
    pub rf_vol_ratio_pause: Decimal,

    // ── Fill management ──────────────────────────────────────────────────────
    /// After N fills the bracket is BURNED (do not reopen).
    #[serde(default = "defaults::rf_max_fills_per_bracket")]
    pub rf_max_fills_per_bracket: u32,
    /// FV of the retained leg for HOLD to settlement.
    #[serde(default = "defaults::rf_unwind_hold_fv")]
    pub rf_unwind_hold_fv: Decimal,
    /// Below this → taker DUMP.
    #[serde(default = "defaults::rf_unwind_dump_fv")]
    pub rf_unwind_dump_fv: Decimal,
    /// Minimum remaining time (seconds) to repost a maker ask in the neutral zone.
    #[serde(default = "defaults::rf_repost_min_secs")]
    pub rf_repost_min_secs: u64,

    // ── Common limits ────────────────────────────────────────────────────────
    /// Do not enter closer to the close than this (daily/weekly markets have room to breathe).
    #[serde(default = "defaults::rf_min_secs_to_clear")]
    pub min_secs_to_clear: i64,
    /// 0 = no ceiling.
    #[serde(default)]
    pub max_secs_to_clear: i64,
}

impl Default for RewardFarmConfig {
    fn default() -> Self {
        Self {
            limits: crate::config::StrategyLimits::default(),
            scan_log_interval_secs: defaults::rf_scan_log_interval_secs(),
            rf_durations: defaults::rf_durations(),
            rf_max_markets: defaults::rf_max_markets(),
            rf_min_daily_rate: Decimal::ZERO,
            rf_mid_min: defaults::rf_mid_min(),
            rf_mid_max: defaults::rf_mid_max(),
            rf_edge_buffer_cents: defaults::rf_edge_buffer_cents(),
            rf_requote_threshold_cents: defaults::rf_requote_threshold_cents(),
            rf_queue_behind: defaults::rf_queue_behind(),
            rf_single_level: defaults::rf_single_level(),
            rf_size_mode: defaults::rf_size_mode(),
            rf_max_total_usdc: defaults::rf_max_total_usdc(),
            rf_require_no_pm_mm: defaults::rf_require_no_pm_mm(),
            rf_cvd_toxic: defaults::rf_cvd_toxic(),
            rf_liq_toxic_notional: Decimal::ZERO,
            rf_vol_ratio_pause: defaults::rf_vol_ratio_pause(),
            rf_max_fills_per_bracket: defaults::rf_max_fills_per_bracket(),
            rf_unwind_hold_fv: defaults::rf_unwind_hold_fv(),
            rf_unwind_dump_fv: defaults::rf_unwind_dump_fv(),
            rf_repost_min_secs: defaults::rf_repost_min_secs(),
            min_secs_to_clear: defaults::rf_min_secs_to_clear(),
            max_secs_to_clear: 0,
        }
    }
}

impl super::StrategyConfig for RewardFarmConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

/// Price band [0.10, 0.90]: outside it the Liquidity Rewards
/// Program requires two-sided quoting (docs.polymarket.com/market-makers/liquidity-rewards).
const DOUBLE_SIDED_LOW: Decimal = dec!(0.10);
const DOUBLE_SIDED_HIGH: Decimal = dec!(0.90);

/// Gamma's `rewardsMaxSpread` is expressed in CENTS ("max spread config of
/// 3 cents" in the official docs), not as a price fraction: divide by 100
/// to compare against prices on a 0-1 scale.
const CENTS_TO_PRICE: Decimal = dec!(100);

/// Penalty for quoting a single side in the middle zone (`Q_one / c`, c = 3.0,
/// a scale fixed across all markets by the official scoring formula).
const DOUBLE_SIDE_PENALTY: Decimal = dec!(3);

// ── Per-bracket state ─────────────────────────────────────────────────────────

/// Live GTD level on one side of the bracket (YES or NO).
#[derive(Debug, Clone, Copy)]
struct LiveQuote {
    price: Decimal,
    size: Decimal,
    /// Token mid at the moment this level was posted (or requoted);
    /// triggers the displacement-based requote.
    posted_mid: Decimal,
}

struct RfState {
    clear_at: i64,
    last_logged: Option<Instant>,

    yes_token: String,
    no_token: String,
    /// Reference (opening) price of the underlying: only used for the
    /// settlement FV of an unwanted fill, never for placement.
    reference: Option<Decimal>,

    yes_quote: Option<LiveQuote>,
    no_quote: Option<LiveQuote>,

    /// Inventory accumulated through fills (shares). RF never chases completing the
    /// box: a fill is evidence that this bracket has flow that picks us off.
    yes_filled: Decimal,
    no_filled: Decimal,
    /// Size-weighted average entry price of the purchases
    /// (Buy fills) accumulated on each side; used by `RepostMaker` so as not to
    /// repost below cost.
    yes_avg_entry: Option<Decimal>,
    no_avg_entry: Option<Decimal>,
    /// Number of fills received in this bracket. `>= rf_max_fills_per_bracket` ⇒ BURNED.
    fills: u32,
    burned: bool,
    /// Unwind decision (hold/dump/repost) already emitted for the current imbalance.
    unwind_sent: bool,
    /// RF has posted a GTD in this bracket at some point; used by bracket
    /// exclusivity: the first time the bracket is evaluated it checks
    /// whether another strategy already has exposure; once RF owns it, it is not
    /// re-evaluated (its own exposure in the ledger must not block itself).
    ever_posted: bool,

    /// Accumulated proxy of expected reward (the sim does not pay real rewards).
    reward_proxy_pusd_accrued: Decimal,
}

impl RfState {
    fn new(book: &BookSnapshot) -> Self {
        Self {
            clear_at: book.clear_at,
            last_logged: None,
            yes_token: book.yes.token_id.clone(),
            no_token: book.no.token_id.clone(),
            reference: None,
            yes_quote: None,
            no_quote: None,
            yes_filled: Decimal::ZERO,
            no_filled: Decimal::ZERO,
            yes_avg_entry: None,
            no_avg_entry: None,
            fills: 0,
            burned: false,
            unwind_sent: false,
            ever_posted: false,
            reward_proxy_pusd_accrued: Decimal::ZERO,
        }
    }

    /// USDC committed in live GTDs of this bracket (sub-cap `rf_max_total_usdc`).
    fn committed_cost(&self) -> Decimal {
        self.yes_quote.map_or(Decimal::ZERO, |q| q.price * q.size)
            + self.no_quote.map_or(Decimal::ZERO, |q| q.price * q.size)
    }
}

/// Unwind decision for the inventory state after an unwanted fill.
/// Redefined here (same pattern as PM, `parity_maker::UnwindAction`);
/// that version is private to its module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnwindActionRf {
    /// FV favors the retained leg: cancel the opposite one and wait for settlement.
    Hold,
    /// FV is unfavorable (or no FV available): sell the excess as taker right away.
    DumpTaker,
    /// Neutral zone with enough time: repost a maker ask instead of going taker.
    RepostMaker,
}

/// Same decision tree as `parity_maker::decide_unwind`,
/// with RF's config names.
fn decide_unwind_rf(
    fv_ret: Option<Decimal>,
    remaining: i64,
    cfg: &RewardFarmConfig,
) -> UnwindActionRf {
    match fv_ret {
        Some(fv) if fv >= cfg.rf_unwind_hold_fv => UnwindActionRf::Hold,
        Some(fv) if fv <= cfg.rf_unwind_dump_fv => UnwindActionRf::DumpTaker,
        Some(_) if remaining > cfg.rf_repost_min_secs as i64 => UnwindActionRf::RepostMaker,
        Some(_) => UnwindActionRf::DumpTaker,
        None => UnwindActionRf::DumpTaker,
    }
}

/// FV for the SETTLEMENT decision (hold vs dump of a
/// single leg). Uses the Binance spot, the bot's only oracle source.
fn fv_yes_settlement_rf(
    vol: &VolSnapshot,
    reference: Option<Decimal>,
    remaining: i64,
) -> Option<Decimal> {
    if vol.is_stale_for_signal() {
        return None;
    }
    let reference = reference?;
    let sigma = vol.sigma_long.or(vol.sigma_short)?;
    fair_value_yes(vol.last_price, reference, sigma, remaining)
}

/// Action to apply to one side (YES or NO) on this placement/requote tick.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SideAction {
    /// Nothing to do: already well placed, or placing/keeping does not apply.
    Noop,
    /// Cancel the live level (no longer quoted, or fell out of range).
    Cancel,
    /// Cancel the live level (if any) and post a new one at the given price.
    Requote(Decimal),
}

/// Places the bid at the EDGE of the scoring band
/// (`s_target = v_c − buffer_c`), optionally behind the existing queue
/// if the best_bid falls within the band. Never crosses the mid.
///
/// Requote (cancel+repost) only when the mid has moved at least
/// `requote_threshold_c` since the last post: hysteresis that minimizes
/// cancellations (the time factor of the scoring rewards live orders).
#[allow(clippy::too_many_arguments)]
fn decide_side_action(
    quote: bool,
    mid: Decimal,
    best_bid: Option<Decimal>,
    v_c: Decimal,
    buffer_c: Decimal,
    tick: Decimal,
    queue_behind: bool,
    requote_threshold_c: Decimal,
    live: Option<(Decimal /* price */, Decimal /* posted_mid */)>,
) -> SideAction {
    if !quote {
        return if live.is_some() {
            SideAction::Cancel
        } else {
            SideAction::Noop
        };
    }

    let s_tgt = (v_c - buffer_c) / CENTS_TO_PRICE;
    let mut px = crate::types::truncate_to_tick(mid - s_tgt, tick);

    if queue_behind {
        if let Some(bb) = best_bid {
            let band_lo = mid - v_c / CENTS_TO_PRICE;
            if bb >= band_lo {
                px = px.min(bb - tick);
            }
        }
    }

    if px <= Decimal::ZERO || px >= mid {
        return if live.is_some() {
            SideAction::Cancel
        } else {
            SideAction::Noop
        };
    }

    let need_requote = match live {
        None => true,
        Some((_, posted_mid)) => (mid - posted_mid).abs() >= requote_threshold_c / CENTS_TO_PRICE,
    };

    if need_requote {
        SideAction::Requote(px)
    } else {
        SideAction::Noop
    }
}

/// `true` if the flow condemns the NO leg (underlying rising:
/// aggressive buyer CVD, or a bullish liquidation cascade — `liq_net_60s`
/// positive = shorts forced to buy, see `oracle/liquidations.rs`).
fn toxic_flow_favors_no(
    cvd_r: Option<Decimal>,
    liq: Option<Decimal>,
    cfg: &RewardFarmConfig,
) -> bool {
    cvd_r.is_some_and(|c| c >= cfg.rf_cvd_toxic)
        || (cfg.rf_liq_toxic_notional > Decimal::ZERO
            && liq.is_some_and(|l| l >= cfg.rf_liq_toxic_notional))
}

/// Analogous to `toxic_flow_favors_no` for the YES leg (underlying falling).
fn toxic_flow_favors_yes(
    cvd_r: Option<Decimal>,
    liq: Option<Decimal>,
    cfg: &RewardFarmConfig,
) -> bool {
    cvd_r.is_some_and(|c| c <= -cfg.rf_cvd_toxic)
        || (cfg.rf_liq_toxic_notional > Decimal::ZERO
            && liq.is_some_and(|l| l <= -cfg.rf_liq_toxic_notional))
}

/// Redefined here (private to `parity_maker.rs`, outside
/// scope): normalizes the raw signed CVD (USDT) to a ratio ∈ [−1,1].
fn cvd_ratio(cvd: Option<Decimal>, vol_quote_60s: Option<Decimal>) -> Option<Decimal> {
    match (cvd, vol_quote_60s) {
        (Some(c), Some(v)) if v > Decimal::ZERO => Some(c / v),
        _ => None,
    }
}

/// Redefined here for the same reason: regime ratio σ5s/σ60s.
fn vol_ratio(sigma_short: Option<Decimal>, sigma_long: Option<Decimal>) -> Option<Decimal> {
    match (sigma_short, sigma_long) {
        (Some(s5), Some(s60)) if s60 > Decimal::ZERO => Some(s5 / s60),
        _ => None,
    }
}

/// Score of an order at distance `s` (cents) from the mid, with
/// `v` = `rewards_max_spread` (cents). `S(s) = ((v−s)/v)² · size` if
/// `0 ≤ s ≤ v`, otherwise `0`.
fn score(s: Decimal, v: Decimal, size: Decimal) -> Decimal {
    if v <= Decimal::ZERO || s < Decimal::ZERO || s > v {
        return Decimal::ZERO;
    }
    let factor = (v - s) / v;
    factor * factor * size
}

/// Two-sided rule of the official scoring. In the mean zone
/// ([mid_min, mid_max]) a single side scores, penalized ÷`DOUBLE_SIDE_PENALTY`;
/// outside that zone only both sides together count (`min`).
fn q_min(
    q_one: Decimal,
    q_two: Decimal,
    mid: Decimal,
    mid_min: Decimal,
    mid_max: Decimal,
) -> Decimal {
    if mid >= mid_min && mid <= mid_max {
        let single = (q_one / DOUBLE_SIDE_PENALTY).max(q_two / DOUBLE_SIDE_PENALTY);
        q_one.min(q_two).max(single)
    } else {
        q_one.min(q_two)
    }
}

/// Own fraction of the pool — `q_self / (q_self + q_comp)`. `0` if there is no
/// own score nor competitor score (nothing to share).
fn share_proxy(q_self: Decimal, q_comp: Decimal) -> Decimal {
    let total = q_self + q_comp;
    if total > Decimal::ZERO {
        q_self / total
    } else {
        Decimal::ZERO
    }
}

/// Increment of the accumulated reward proxy over a sample of `dt_secs`
/// seconds — `rewards_daily_rate · share · (dt/86400)`.
fn reward_proxy_delta(daily_rate: Decimal, share: Decimal, dt_secs: u64) -> Decimal {
    daily_rate * share * Decimal::from(dt_secs) / dec!(86400)
}

/// Size-weighted average entry price after accumulating a new
/// Buy fill — `(avg_prev·size_prev + price_fill·size_fill) / (size_prev+size_fill)`.
/// Without a previous entry (or previous size 0), the average price is that of the fill itself.
fn weighted_avg_entry(
    prev_avg: Option<Decimal>,
    prev_size: Decimal,
    fill_price: Decimal,
    fill_size: Decimal,
) -> Decimal {
    match prev_avg {
        Some(avg) if prev_size > Decimal::ZERO => {
            (avg * prev_size + fill_price * fill_size) / (prev_size + fill_size)
        }
        _ => fill_price,
    }
}

/// Maker repost price after an unwanted fill
/// in a neutral FV zone — the spec asks for `max(best_ask, avg_entry + edge)`,
/// never a bare `best_ask` (it would give away the spread below cost). Without
/// a dedicated edge parameter in the config (KISS): a minimum of 1 tick above the
/// `avg_entry`. `None` if the result falls outside `(0,1)` (price
/// invalid after truncating to the tick) — the caller must skip that tick.
fn repost_maker_price(
    best_ask: Decimal,
    avg_entry: Option<Decimal>,
    tick: Decimal,
) -> Option<Decimal> {
    let floor = avg_entry.map_or(best_ask, |e| e + tick);
    let price = crate::types::truncate_to_tick(best_ask.max(floor), tick);
    if price > Decimal::ZERO && price < Decimal::ONE {
        Some(price)
    } else {
        None
    }
}

/// Nominal bracket duration from `clear_at − market_open_ts`,
/// bucketed with a 20% tolerance (avoids losing brackets to rounding in
/// discovery). `None` if `market_open_ts` is not populated or the duration does not
/// fit any known bucket — an unknown duration is NOT farmed
/// (conservative stance: this spec's default risk is to not trade
/// 5m/hourly by mistake, never the reverse).
fn bracket_duration_label(book: &BookSnapshot) -> Option<&'static str> {
    if book.market_open_ts <= 0 || book.clear_at <= book.market_open_ts {
        return None;
    }
    let secs = book.clear_at - book.market_open_ts;
    let close_to = |target: i64| (secs - target).abs() <= target / 5;
    if close_to(300) {
        Some("5m")
    } else if close_to(3_600) {
        Some("hourly")
    } else if close_to(86_400) {
        Some("daily")
    } else if close_to(604_800) {
        Some("weekly")
    } else {
        None
    }
}

/// `true` if the bracket duration is in the allowed list
/// (`rf_durations`, default `["daily","weekly"]` — NEVER 5m/hourly by default).
fn duration_allowed(book: &BookSnapshot, allowed: &[String]) -> bool {
    match bracket_duration_label(book) {
        Some(label) => allowed.iter().any(|d| d == label),
        None => false,
    }
}

/// `true` if the mid falls outside [0.10, 0.90] — the rewards program
/// requires two-sided quoting mandatorily in that range (official docs).
fn double_sided_required(mid: Decimal) -> bool {
    mid < DOUBLE_SIDED_LOW || mid > DOUBLE_SIDED_HIGH
}

/// Depth (sum of sizes) of the bid+ask levels of `token` whose
/// price falls within `[mid − band, mid + band]` — the current competition
/// for the rewards score on that side.
fn depth_in_band(token: &TokenBook, mid: Decimal, band: Decimal) -> Decimal {
    let lo = mid - band;
    let hi = mid + band;
    let sum_side = |levels: &[Level]| -> Decimal {
        levels
            .iter()
            .filter(|l| l.price >= lo && l.price <= hi)
            .map(|l| l.size)
            .sum()
    };
    sum_side(&token.bids.levels) + sum_side(&token.asks.levels)
}

/// Same `opt_dec` pattern as `pm_entry` (`parity_maker.rs`) — optional Decimal
/// to String, absent signal = empty. Redefined here because the
/// original is private to its module and that file is out of scope for
/// this change (must not be touched).
fn opt_dec(d: Option<Decimal>) -> String {
    d.map_or(String::new(), |v| v.to_string())
}

// ── Intents ───────────────────────────────────────────────────────────────────

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
        strategy: StrategyId::RewardFarm,
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
        strategy: StrategyId::RewardFarm,
        resting: false,
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
        strategy: StrategyId::RewardFarm,
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
        StrategyId::RewardFarm,
    )
}

/// Mid out of range, volatility spike pause, or no confirmed rate (G0):
/// cancels both live legs and clears the bracket's posting state.
fn cancel_all(state: &mut RfState, book: &BookSnapshot, intents: &mut Vec<OrderIntent>) {
    if state.yes_quote.is_some() {
        intents.push(cancel_intent(book, &state.yes_token));
        state.yes_quote = None;
    }
    if state.no_quote.is_some() {
        intents.push(cancel_intent(book, &state.no_token));
        state.no_quote = None;
    }
}

// ── RewardFarm ────────────────────────────────────────────────────────────────

pub struct RewardFarm {
    cfg: RewardFarmConfig,
    registry: Arc<MarketRegistry>,
    /// Shared exposure ledger — RF only uses it to READ
    /// (`net_exposure`) and decides bracket exclusivity against PM/MM; the
    /// executor is the one that writes to it when routing RF's GTDs (as with
    /// any other strategy, without touching `order/executor.rs`).
    ledger: Arc<BracketLedger>,
    markets: HashMap<String, RfState>,
    last_retain: Instant,
    /// Count of scans actually logged (after passing the throttle).
    pub scans_logged: u64,
}

impl RewardFarm {
    pub fn new(
        cfg: RewardFarmConfig,
        registry: Arc<MarketRegistry>,
        ledger: Arc<BracketLedger>,
    ) -> Self {
        Self {
            cfg,
            registry,
            ledger,
            markets: HashMap::new(),
            last_retain: Instant::now(),
            scans_logged: 0,
        }
    }
}

impl Strategy for RewardFarm {
    fn id(&self) -> StrategyId {
        StrategyId::RewardFarm
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        if self.last_retain.elapsed() >= Duration::from_secs(30) {
            self.markets.retain(|_, s| s.clear_at > now);
            self.last_retain = Instant::now();
        }

        // The market's rewards config lives in the registry (see
        // module comment), not in the BookSnapshot. Without an entry, or without
        // configured rewards (Gamma reported 0 → None in discovery), there
        // is nothing to scan or post.
        let (rewards_min_size, rewards_max_spread, rewards_daily_rate) =
            match self.registry.markets.get(&book.market_condition_id) {
                Some(entry) => (
                    entry.rewards_min_size,
                    entry.rewards_max_spread,
                    entry.rewards_daily_rate,
                ),
                None => return Vec::new(),
            };
        if rewards_min_size.is_none() && rewards_max_spread.is_none() {
            return Vec::new();
        }

        // Totals taken BEFORE borrowing this bracket's mutable state
        // (the exclusive borrow of `self.markets` via `entry()` below prevents
        // iterating the map at the same time). They include this bracket's own current exposure
        // if it already existed — it is subtracted at the point of use.
        let mut total_committed_before = Decimal::ZERO;
        let mut active_markets_before = 0usize;
        for s in self.markets.values() {
            total_committed_before += s.committed_cost();
            if s.ever_posted {
                active_markets_before += 1;
            }
        }

        let cid = book.market_condition_id.clone();
        let mut just_logged = false;
        let state = self
            .markets
            .entry(cid.clone())
            .or_insert_with(|| RfState::new(book));

        // ── (a) Scan/log — same throttle and same fields as always,
        // plus the score/reward proxy for paper (the sim does not pay
        // real rewards). ──────────────────────────────────────────────────
        let interval = Duration::from_secs(self.cfg.scan_log_interval_secs);
        if should_log(state.last_logged, interval) {
            if let Some(mid) = book.yes_mid() {
                let dt_secs = state
                    .last_logged
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);
                let max_spread_price = rewards_max_spread.map(|c| c / CENTS_TO_PRICE);
                let double_required = double_sided_required(mid);

                let (depth_yes_in_band, depth_no_in_band) = match max_spread_price {
                    Some(band) => (
                        depth_in_band(&book.yes, mid, band),
                        book.no_mid()
                            .map(|no_mid| depth_in_band(&book.no, no_mid, band))
                            .unwrap_or(Decimal::ZERO),
                    ),
                    None => (Decimal::ZERO, Decimal::ZERO),
                };

                let v_c = rewards_max_spread.unwrap_or(Decimal::ZERO);
                let score_yes = state.yes_quote.map_or(Decimal::ZERO, |q| {
                    score((mid - q.price).abs() * CENTS_TO_PRICE, v_c, q.size)
                });
                let score_no = book
                    .no_mid()
                    .zip(state.no_quote)
                    .map_or(Decimal::ZERO, |(no_mid, q)| {
                        score((no_mid - q.price).abs() * CENTS_TO_PRICE, v_c, q.size)
                    });
                let q_min_self = q_min(
                    score_yes,
                    score_no,
                    mid,
                    self.cfg.rf_mid_min,
                    self.cfg.rf_mid_max,
                );
                let q_comp = depth_yes_in_band + depth_no_in_band;
                let share = share_proxy(q_min_self, q_comp);
                if let Some(rate) = rewards_daily_rate {
                    let delta = reward_proxy_delta(rate, share, dt_secs);
                    state.reward_proxy_pusd_accrued += delta;
                    metrics::gauge!("hrf_rf_reward_proxy_pusd_accrued")
                        .increment(delta.to_f64().unwrap_or(0.0));
                }

                state.last_logged = Some(Instant::now());
                just_logged = true;

                metrics::counter!("hrf_rf_scan_total").increment(1);
                tracing::info!(
                    target: "rf_scan",
                    cid               = %cid,
                    asset             = %vol.asset,
                    ts_ms             = crate::time::now().timestamp_millis(),
                    mid               = %mid,
                    max_spread        = %opt_dec(rewards_max_spread),
                    min_size          = %opt_dec(rewards_min_size),
                    daily_rate        = %opt_dec(rewards_daily_rate),
                    depth_yes_in_band = %depth_yes_in_band,
                    depth_no_in_band  = %depth_no_in_band,
                    best_yes_bid      = %opt_dec(book.yes.best_bid()),
                    best_no_bid       = %opt_dec(book.no.best_bid()),
                    double_sided_required = double_required,
                    remaining_secs    = book.secs_to_clear(),
                    score_self        = %q_min_self,
                    share_proxy       = %share,
                    reward_proxy_pusd_accrued = %state.reward_proxy_pusd_accrued,
                    "RF: rewards scan"
                );
            }
        }
        if just_logged {
            self.scans_logged += 1;
        }

        // ── (b) Real posting ──────────────────────────────────────────────
        let mut intents = Vec::new();

        // Management of an unwanted fill — top priority. While the
        // inventory is unbalanced, RF evaluates nothing else: it cancels the
        // opposing leg immediately (avoids turning into PM by accident) and
        // decides hold/dump/repost by FV. It never chases completing the box.
        let imbalance = state.yes_filled - state.no_filled;
        if imbalance != Decimal::ZERO && !state.unwind_sent {
            let retained_is_yes = imbalance > Decimal::ZERO;

            if retained_is_yes {
                if state.no_quote.is_some() {
                    intents.push(cancel_intent(book, &state.no_token));
                    state.no_quote = None;
                }
            } else if state.yes_quote.is_some() {
                intents.push(cancel_intent(book, &state.yes_token));
                state.yes_quote = None;
            }

            if state.reference.is_none() {
                state.reference = book
                    .reference_price
                    .or_else(|| super::fallback_reference(vol, book, now, &cid, "RF"));
            }
            let remaining = book.secs_to_clear();
            let fv_ret = fv_yes_settlement_rf(vol, state.reference, remaining).map(|f| {
                if retained_is_yes {
                    f
                } else {
                    Decimal::ONE - f
                }
            });
            let action = decide_unwind_rf(fv_ret, remaining, &self.cfg);

            match action {
                UnwindActionRf::Hold => {
                    metrics::counter!("hrf_rf_unwind_total", "reason" => "hold_fv").increment(1);
                    tracing::info!(cid = %cid, fv_ret = ?fv_ret,
                        "RF: unwanted fill — hold-to-settlement (FV favors the retained leg)");
                    state.unwind_sent = true;
                }
                UnwindActionRf::DumpTaker => {
                    let (token, excess, bid) = if retained_is_yes {
                        (state.yes_token.clone(), imbalance, book.yes.best_bid())
                    } else {
                        (state.no_token.clone(), -imbalance, book.no.best_bid())
                    };
                    if let Some(bid) = bid {
                        let reason = if fv_ret.is_none() {
                            "dump_no_fv"
                        } else {
                            "dump_fv"
                        };
                        metrics::counter!("hrf_rf_unwind_total", "reason" => reason).increment(1);
                        tracing::info!(cid = %cid, token = %token, excess = %excess, bid = %bid,
                            "RF: unwanted fill — taker dump of the excess leg");
                        intents.push(sell_intent(book, &token, bid, excess));
                        state.unwind_sent = true;
                    }
                    // No valid bid: do not mark unwind_sent — retry on the next tick.
                }
                UnwindActionRf::RepostMaker => {
                    let (token, excess, best_ask, avg_entry) = if retained_is_yes {
                        (
                            state.yes_token.clone(),
                            imbalance,
                            book.yes.best_ask(),
                            state.yes_avg_entry,
                        )
                    } else {
                        (
                            state.no_token.clone(),
                            -imbalance,
                            book.no.best_ask(),
                            state.no_avg_entry,
                        )
                    };
                    if let Some(ask) = best_ask {
                        // Repost at
                        // `max(best_ask, avg_entry + edge)`, not at a bare `best_ask` —
                        // never give away the spread below cost.
                        if let Some(price) = repost_maker_price(ask, avg_entry, book.tick()) {
                            metrics::counter!("hrf_rf_unwind_total", "reason" => "repost_maker")
                                .increment(1);
                            tracing::info!(cid = %cid, token = %token, price = %price, excess = %excess,
                                avg_entry = ?avg_entry,
                                "RF: unwanted fill — neutral FV zone with time left, maker repost instead of dump");
                            intents.push(ask_maker_intent(book, &token, price, excess));
                            state.unwind_sent = true;
                        }
                        // Invalid price (≥1 after adding the edge): do not mark
                        // unwind_sent — retry on the next tick.
                    }
                }
            }

            return intents;
        }

        // Unwind already sent (or hold decided) but the inventory STILL is
        // unbalanced — the sell may be in flight.
        // Without this cut, execution fell through to the posting section and RF re-placed
        // GTDs (the PM behavior the spec forbids). Only reachable with
        // rf_max_fills_per_bracket > 1 (with the default 1 the bracket is already
        // burned), but the invariant "unbalanced ⇒ do not post" must hold
        // with any config.
        if imbalance != Decimal::ZERO {
            return intents;
        }

        // Bracket BURNED after `rf_max_fills_per_bracket` fills — do not reopen.
        if state.burned {
            cancel_all(state, book, &mut intents);
            return intents;
        }

        // ── Gate G0: pool rate confirmed by the scan ─────────────────────
        let Some(rate) = rewards_daily_rate else {
            cancel_all(state, book, &mut intents);
            return intents;
        };
        if rate < self.cfg.rf_min_daily_rate {
            cancel_all(state, book, &mut intents);
            return intents;
        }

        // Allowed duration (NEVER 5m/hourly by default).
        if !duration_allowed(book, &self.cfg.rf_durations) {
            return intents;
        }

        let Some(min_size) = rewards_min_size else {
            return intents;
        };
        let Some(v_c) = rewards_max_spread else {
            return intents;
        };

        let remaining = book.secs_to_clear();
        if remaining < self.cfg.min_secs_to_clear {
            cancel_all(state, book, &mut intents);
            return intents;
        }
        if self.cfg.max_secs_to_clear > 0 && remaining > self.cfg.max_secs_to_clear {
            return intents;
        }

        // Diversification — do not open a NEW bracket above
        // `rf_max_markets` (already open ones keep being managed without limit).
        if !state.ever_posted
            && self.cfg.rf_max_markets > 0
            && active_markets_before >= self.cfg.rf_max_markets as usize
        {
            metrics::counter!("hrf_rf_skip_max_markets_total").increment(1);
            return intents;
        }

        // Bracket exclusivity versus PM/MM. Only evaluated before
        // RF has ever posted here — RF's own exposure must not
        // self-block it on later requotes. Uses GROSS exposure
        // (not net): a balanced PM box (YES=NO) yields `net_exposure == 0`
        // but another strategy IS operating the bracket — netting would let
        // RF post on top of an already covered box.
        if self.cfg.rf_require_no_pm_mm
            && !state.ever_posted
            && self.ledger.gross_exposure(&cid) > Decimal::ZERO
        {
            metrics::counter!("hrf_rf_skip_busy_bracket_total").increment(1);
            return intents;
        }

        // ── Mid and two-sided rule ──────────────────────────────────────────
        let Some(mid_yes) = book.yes_mid() else {
            return intents;
        };
        if mid_yes < self.cfg.rf_mid_min || mid_yes > self.cfg.rf_mid_max {
            // Extremes: qualifying requires buying the almost-losing leg — the
            // most toxic fill possible. RF abandons the bracket completely.
            metrics::counter!("hrf_rf_abandon_extreme_total").increment(1);
            cancel_all(state, book, &mut intents);
            return intents;
        }

        // ── Toxic-flow signals ─────────────────────────────────────
        if vol.is_stale_for_signal() {
            // Missing signal ≠ toxic signal: conservative stance, do not touch what
            // is already live and place nothing new until the oracle
            // recovers.
            return intents;
        }
        let cvd_r = cvd_ratio(vol.cvd_60s, vol.vol_quote_60s);
        let volr = vol_ratio(vol.sigma_short, vol.sigma_long);
        let liq = vol.liq_net_60s;

        if volr.is_some_and(|r| r >= self.cfg.rf_vol_ratio_pause) {
            metrics::counter!("hrf_rf_pause_vol_spike_total").increment(1);
            cancel_all(state, book, &mut intents);
            return intents;
        }

        let quote_yes = !toxic_flow_favors_yes(cvd_r, liq, &self.cfg);
        let quote_no = !toxic_flow_favors_no(cvd_r, liq, &self.cfg);
        if !quote_yes {
            metrics::counter!("hrf_rf_toxic_withdrawal_total", "side" => "yes").increment(1);
        }
        if !quote_no {
            metrics::counter!("hrf_rf_toxic_withdrawal_total", "side" => "no").increment(1);
        }

        // ── Placement at the edge of the band / requote ─────────────────
        let tick = book.tick();

        // Accumulator updated after each leg
        // posted in THIS tick — `total_committed_before` only reflects the
        // state prior to the tick; without this, the NO leg was checked against the
        // sub-cap without counting the YES that the same tick has just committed.
        let mut total_committed = total_committed_before;

        let existing_cost_yes = state.yes_quote.map_or(Decimal::ZERO, |q| q.price * q.size);
        match decide_side_action(
            quote_yes,
            mid_yes,
            book.yes.best_bid(),
            v_c,
            self.cfg.rf_edge_buffer_cents,
            tick,
            self.cfg.rf_queue_behind,
            self.cfg.rf_requote_threshold_cents,
            state.yes_quote.map(|q| (q.price, q.posted_mid)),
        ) {
            SideAction::Noop => {}
            SideAction::Cancel => {
                if state.yes_quote.is_some() {
                    intents.push(cancel_intent(book, &state.yes_token));
                    state.yes_quote = None;
                }
            }
            SideAction::Requote(px) => {
                let order_cost = px * min_size;
                let projected = total_committed - existing_cost_yes + order_cost;
                if self.cfg.rf_max_total_usdc > Decimal::ZERO
                    && projected > self.cfg.rf_max_total_usdc
                {
                    metrics::counter!("hrf_rf_skip_cap_total_total").increment(1);
                    // A requote blocked by the sub-cap must not leave
                    // the oldest GTD alive (it could end up crossed with the new mid)
                    // — better no order than an order at a dead price.
                    if state.yes_quote.is_some() {
                        intents.push(cancel_intent(book, &state.yes_token));
                        state.yes_quote = None;
                    }
                } else if self.cfg.limits.max_order_usdc > Decimal::ZERO
                    && order_cost > self.cfg.limits.max_order_usdc
                {
                    // The executor silently discards the order if
                    // price*size > max_order_usdc (order/executor.rs);
                    // if we marked the quota anyway, a ghost quota would remain that
                    // is never retried and keeps counting against the sub-cap.
                    // Skip without touching the state — it will be retried if the mid moves.
                    tracing::debug!(cid = %cid, token = %state.yes_token, px = %px, size = %min_size,
                        cost = %order_cost, max_order_usdc = %self.cfg.limits.max_order_usdc,
                        "RF: skip — YES leg exceeds max_order_usdc, the executor would reject it silently");
                    metrics::counter!("hrf_rf_skip_max_order_usdc_total").increment(1);
                } else {
                    if state.yes_quote.is_some() {
                        intents.push(cancel_intent(book, &state.yes_token));
                    }
                    intents.push(bid_intent(book, &state.yes_token, px, min_size));
                    state.yes_quote = Some(LiveQuote {
                        price: px,
                        size: min_size,
                        posted_mid: mid_yes,
                    });
                    state.ever_posted = true;
                }
            }
        }
        total_committed = total_committed - existing_cost_yes
            + state.yes_quote.map_or(Decimal::ZERO, |q| q.price * q.size);

        match book.no_mid() {
            None => {
                if state.no_quote.is_some() {
                    intents.push(cancel_intent(book, &state.no_token));
                    state.no_quote = None;
                }
            }
            Some(mid_no) => {
                let existing_cost_no = state.no_quote.map_or(Decimal::ZERO, |q| q.price * q.size);
                match decide_side_action(
                    quote_no,
                    mid_no,
                    book.no.best_bid(),
                    v_c,
                    self.cfg.rf_edge_buffer_cents,
                    tick,
                    self.cfg.rf_queue_behind,
                    self.cfg.rf_requote_threshold_cents,
                    state.no_quote.map(|q| (q.price, q.posted_mid)),
                ) {
                    SideAction::Noop => {}
                    SideAction::Cancel => {
                        if state.no_quote.is_some() {
                            intents.push(cancel_intent(book, &state.no_token));
                            state.no_quote = None;
                        }
                    }
                    SideAction::Requote(px) => {
                        let order_cost = px * min_size;
                        let projected = total_committed - existing_cost_no + order_cost;
                        if self.cfg.rf_max_total_usdc > Decimal::ZERO
                            && projected > self.cfg.rf_max_total_usdc
                        {
                            metrics::counter!("hrf_rf_skip_cap_total_total").increment(1);
                            // Same as YES — do not leave the oldest GTD alive.
                            if state.no_quote.is_some() {
                                intents.push(cancel_intent(book, &state.no_token));
                                state.no_quote = None;
                            }
                        } else if self.cfg.limits.max_order_usdc > Decimal::ZERO
                            && order_cost > self.cfg.limits.max_order_usdc
                        {
                            // Same as YES.
                            tracing::debug!(cid = %cid, token = %state.no_token, px = %px, size = %min_size,
                                cost = %order_cost, max_order_usdc = %self.cfg.limits.max_order_usdc,
                                "RF: skip — NO leg exceeds max_order_usdc, the executor would reject it silently");
                            metrics::counter!("hrf_rf_skip_max_order_usdc_total").increment(1);
                        } else {
                            if state.no_quote.is_some() {
                                intents.push(cancel_intent(book, &state.no_token));
                            }
                            intents.push(bid_intent(book, &state.no_token, px, min_size));
                            state.no_quote = Some(LiveQuote {
                                price: px,
                                size: min_size,
                                posted_mid: mid_no,
                            });
                            state.ever_posted = true;
                        }
                    }
                }
                // (there is no third leg that needs the total updated
                // after NO — unlike the YES side, it is not reassigned here).
            }
        }

        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::RewardFarm {
            return;
        }
        let Some(state) = self.markets.get_mut(&fill.market_cid) else {
            // Mid-bracket restart: without state there is no
            // opposing leg to cancel nor FV to compute with confidence.
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "RF: buy fill without state — bracket left to the orphan checker (mid-bracket restart)");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "reward_farm").increment(1);
            }
            return;
        };
        match fill.side {
            FillSide::Buy => {
                if fill.token_id == state.yes_token {
                    state.yes_avg_entry = Some(weighted_avg_entry(
                        state.yes_avg_entry,
                        state.yes_filled,
                        fill.price,
                        fill.size,
                    ));
                    state.yes_filled += fill.size;
                    state.yes_quote = None;
                } else if fill.token_id == state.no_token {
                    state.no_avg_entry = Some(weighted_avg_entry(
                        state.no_avg_entry,
                        state.no_filled,
                        fill.price,
                        fill.size,
                    ));
                    state.no_filled += fill.size;
                    state.no_quote = None;
                }
                state.fills += 1;
                metrics::counter!("hrf_rf_fill_total").increment(1);
                // Bracket BURNED after N fills — a fill is hard evidence
                // of flow that picks us off; do not reopen.
                if state.fills >= self.cfg.rf_max_fills_per_bracket.max(1) {
                    state.burned = true;
                }
            }
            FillSide::Sell => {
                // Unwind (taker dump or maker repost) — reduces the inventory.
                if fill.token_id == state.yes_token {
                    state.yes_filled = (state.yes_filled - fill.size).max(Decimal::ZERO);
                    if state.yes_filled == Decimal::ZERO {
                        state.yes_avg_entry = None;
                    }
                } else if fill.token_id == state.no_token {
                    state.no_filled = (state.no_filled - fill.size).max(Decimal::ZERO);
                    if state.no_filled == Decimal::ZERO {
                        state.no_avg_entry = None;
                    }
                }
                // Bug fix: unconditional reset, not only when the
                // imbalance reaches zero. The dump is FAK (paper: cut to
                // `cached_bid_depth` in sim_router; live: clob_sdk::submit_sell
                // with OrderType::FAK) and the repost is a GTD that, if it fills, does so
                // for the size available in that tick — in both cases a
                // PARTIAL fill leaves nothing retrying on its own: the order no longer
                // exists. Without this reset the remainder would be orphaned forever
                // (blocked after the `return` of the "unwind in flight" section of
                // `on_book`, without going back through `decide_unwind_rf`).
                state.unwind_sent = false;
            }
            // Ack of an intent blocked by a pre-submit gate — the
            // order never existed. The quote mark is only released if the
            // price matches that of the blocked intent: a more recent requote
            // would have overwritten the mark and must not be touched.
            FillSide::GateRejected => {
                if fill.token_id == state.yes_token
                    && state.yes_quote.is_some_and(|q| q.price == fill.price)
                {
                    state.yes_quote = None;
                }
                if fill.token_id == state.no_token
                    && state.no_quote.is_some_and(|q| q.price == fill.price)
                {
                    state.no_quote = None;
                }
            }
            FillSide::Cancelled | FillSide::Rejected => {
                if fill.token_id == state.yes_token {
                    state.yes_quote = None;
                }
                if fill.token_id == state.no_token {
                    state.no_quote = None;
                }
                // A `Rejected` on the RETAINED leg during the
                // unwind can only be the unwind sell rejected (RF does not
                // quote on that leg while `unwind_sent`). Without this
                // reset the flag would stay stuck and the unwanted fill would roll
                // to settlement unmanaged. NOT with `Cancelled`: the acks of
                // the quote cancels travel through the same channel and an
                // indiscriminate reset would trigger spurious re-decisions.
                if matches!(fill.side, FillSide::Rejected) && state.unwind_sent {
                    let excess_yes = state.yes_filled - state.no_filled;
                    let retained_token = if excess_yes > Decimal::ZERO {
                        &state.yes_token
                    } else {
                        &state.no_token
                    };
                    if fill.token_id == *retained_token {
                        state.unwind_sent = false;
                        tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                            "RF: unwind sell rejected — unwind_sent reset, re-decided on the next tick");
                    }
                }
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// `true` if at least `interval` has passed since `last_logged` (or if
/// it was never logged). Extracted as a pure function to be able to test the
/// throttle without log-capture mechanisms.
fn should_log(last_logged: Option<Instant>, interval: Duration) -> bool {
    last_logged.is_none_or(|t| t.elapsed() >= interval)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_feed::registry::MarketEntry;
    use crate::types::HalfBook;
    use rust_decimal_macros::dec;

    fn registry_with_rewards(
        cid: &str,
        min_size: Option<Decimal>,
        max_spread: Option<Decimal>,
        daily_rate: Option<Decimal>,
    ) -> Arc<MarketRegistry> {
        let registry = Arc::new(MarketRegistry::new());
        registry.markets.insert(
            cid.into(),
            MarketEntry {
                slug: "btc-updown-daily-test".into(),
                yes_token_id: "yes-tok".into(),
                no_token_id: "no-tok".into(),
                clear_at: crate::time::now().timestamp() + 86_400,
                market_open_ts: 0,
                neg_risk: false,
                reference_price: None,
                fee_rate_bps: None,
                parabolic_fee_rate: None,
                tick_size: None,
                rewards_min_size: min_size,
                rewards_max_spread: max_spread,
                rewards_daily_rate: daily_rate,
            },
        );
        registry
    }

    fn make_book(
        cid: &str,
        yes_bid: &str,
        yes_ask: &str,
        no_bid: &str,
        no_ask: &str,
        remaining: i64,
    ) -> BookSnapshot {
        let level = |p: &str| Level {
            price: p.parse().unwrap(),
            size: dec!(100),
        };
        BookSnapshot {
            market_condition_id: cid.into(),
            yes: TokenBook {
                token_id: "yes-tok".into(),
                bids: HalfBook {
                    levels: vec![level(yes_bid)],
                },
                asks: HalfBook {
                    levels: vec![level(yes_ask)],
                },
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-tok".into(),
                bids: HalfBook {
                    levels: vec![level(no_bid)],
                },
                asks: HalfBook {
                    levels: vec![level(no_ask)],
                },
                ..Default::default()
            },
            clear_at: crate::time::now().timestamp() + remaining,
            slug: "btc-updown-5m-test".into(),
            ..Default::default()
        }
    }

    /// Variant of `make_book` with `market_open_ts` populated so that
    /// `bracket_duration_label` recognizes a "daily" bracket (~86400s).
    fn make_daily_book(
        cid: &str,
        yes_bid: &str,
        yes_ask: &str,
        no_bid: &str,
        no_ask: &str,
        remaining: i64,
    ) -> BookSnapshot {
        let mut book = make_book(cid, yes_bid, yes_ask, no_bid, no_ask, remaining);
        book.clear_at = crate::time::now().timestamp() + remaining;
        book.market_open_ts = book.clear_at - 86_400;
        book.slug = "btc-updown-daily-test".into();
        book.reference_price = Some(dec!(100_000));
        book
    }

    fn make_vol() -> VolSnapshot {
        VolSnapshot {
            asset: "btc".into(),
            ..Default::default()
        }
    }

    /// "Fresh" VolSnapshot (not stale) without toxic signals — needed so
    /// RF reaches placement (the staleness gate is conservative).
    fn make_fresh_vol() -> VolSnapshot {
        VolSnapshot {
            asset: "btc".into(),
            sigma_short: Some(dec!(0.0005)),
            sigma_long: Some(dec!(0.0005)),
            last_price: dec!(100_000),
            updated_at: Instant::now(),
            ..VolSnapshot::default()
        }
    }

    fn rf_with(cfg: RewardFarmConfig, registry: Arc<MarketRegistry>) -> RewardFarm {
        RewardFarm::new(cfg, registry, BracketLedger::new())
    }

    // ── (b) depth in band ──────────────────────────────────────────

    #[test]
    fn depth_in_band_sums_only_levels_within_band() {
        let token = TokenBook {
            token_id: "t".into(),
            bids: HalfBook {
                levels: vec![
                    Level {
                        price: dec!(0.48),
                        size: dec!(10),
                    }, // within: [0.47, 0.53]
                    Level {
                        price: dec!(0.40),
                        size: dec!(999),
                    }, // outside
                ],
            },
            asks: HalfBook {
                levels: vec![
                    Level {
                        price: dec!(0.52),
                        size: dec!(5),
                    }, // within
                    Level {
                        price: dec!(0.60),
                        size: dec!(999),
                    }, // outside
                ],
            },
            ..Default::default()
        };
        let depth = depth_in_band(&token, dec!(0.50), dec!(0.03));
        assert_eq!(
            depth,
            dec!(15),
            "only the levels within [0.47, 0.53] must add up"
        );
    }

    #[test]
    fn depth_in_band_empty_if_no_level_falls_within() {
        let token = TokenBook {
            token_id: "t".into(),
            bids: HalfBook {
                levels: vec![Level {
                    price: dec!(0.10),
                    size: dec!(50),
                }],
            },
            asks: HalfBook {
                levels: vec![Level {
                    price: dec!(0.90),
                    size: dec!(50),
                }],
            },
            ..Default::default()
        };
        assert_eq!(depth_in_band(&token, dec!(0.50), dec!(0.03)), Decimal::ZERO);
    }

    // ── (c) two-sided rule (scan) ──────────────────────────────

    #[test]
    fn double_sided_required_inside_and_outside_band() {
        assert!(
            !double_sided_required(dec!(0.50)),
            "mid 0.50 does not require two-sided quoting"
        );
        assert!(
            !double_sided_required(dec!(0.10)),
            "lower edge included in the band"
        );
        assert!(
            !double_sided_required(dec!(0.90)),
            "upper edge included in the band"
        );
        assert!(
            double_sided_required(dec!(0.95)),
            "mid 0.95 requires two-sided quoting"
        );
        assert!(
            double_sided_required(dec!(0.05)),
            "mid 0.05 requires two-sided quoting"
        );
    }

    // ── (d) log throttle per interval ──────────────────────────────

    #[test]
    fn throttle_limits_to_a_single_scan_within_interval() {
        let cfg = RewardFarmConfig {
            scan_log_interval_secs: 60,
            ..RewardFarmConfig::default()
        };
        let registry = registry_with_rewards("cid-rf", Some(dec!(50)), Some(dec!(0.2)), None);
        let mut rf = rf_with(cfg, registry);
        let book = make_book("cid-rf", "0.49", "0.51", "0.49", "0.51", 60);
        let vol = make_vol();

        for _ in 0..5 {
            assert!(
                rf.on_book(&book, &vol).is_empty(),
                "without a confirmed rate (G0) it must never emit an OrderIntent"
            );
        }
        assert_eq!(
            rf.scans_logged, 1,
            "the throttle must limit it to a single logged scan in 60s"
        );
    }

    #[test]
    fn throttle_at_zero_logs_every_book() {
        let cfg = RewardFarmConfig {
            scan_log_interval_secs: 0,
            ..RewardFarmConfig::default()
        };
        let registry = registry_with_rewards("cid-rf", Some(dec!(50)), Some(dec!(0.2)), None);
        let mut rf = rf_with(cfg, registry);
        let book = make_book("cid-rf", "0.49", "0.51", "0.49", "0.51", 60);
        let vol = make_vol();

        for _ in 0..3 {
            rf.on_book(&book, &vol);
        }
        assert_eq!(
            rf.scans_logged, 3,
            "interval 0 must not suppress any scan"
        );
    }

    // ── Gates for missing config / entry ──────────────────────────

    #[test]
    fn without_entry_in_registry_does_not_scan() {
        let cfg = RewardFarmConfig::default();
        let registry = Arc::new(MarketRegistry::new()); // empty
        let mut rf = rf_with(cfg, registry);
        let book = make_book("cid-unknown", "0.49", "0.51", "0.49", "0.51", 60);
        assert!(rf.on_book(&book, &make_vol()).is_empty());
        assert_eq!(rf.scans_logged, 0);
    }

    #[test]
    fn market_without_rewards_is_not_scanned() {
        // rewards_min_size/max_spread both None (Gamma reported 0 → discovery
        // mapped them to None) — RF must abstain without logging anything.
        let cfg = RewardFarmConfig::default();
        let registry = registry_with_rewards("cid-no-rewards", None, None, None);
        let mut rf = rf_with(cfg, registry);
        let book = make_book("cid-no-rewards", "0.49", "0.51", "0.49", "0.51", 60);
        assert!(rf.on_book(&book, &make_vol()).is_empty());
        assert_eq!(rf.scans_logged, 0);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Real posting
    // ══════════════════════════════════════════════════════════════════════

    /// Standard asset posting config for the posting tests: rate/gates
    /// open, fine tick so the placement can be tested exactly to the cent.
    fn rf2_cfg() -> RewardFarmConfig {
        RewardFarmConfig {
            limits: crate::config::StrategyLimits {
                enabled: true,
                ..Default::default()
            },
            min_secs_to_clear: 60,
            ..RewardFarmConfig::default()
        }
    }

    // 1. score_at_edge_is_low_but_positive
    #[test]
    fn score_at_edge_is_low_but_positive() {
        let v = dec!(4.5);
        let at_the_edge = score(v - dec!(0.5), v, dec!(50)); // s = 4.0c
        let near_mid = score(dec!(1.0), v, dec!(50));
        assert!(at_the_edge > Decimal::ZERO, "the edge must score > 0");
        assert!(
            at_the_edge < near_mid,
            "the edge must score much less than near the mid"
        );
    }

    // 2. edge_placement
    #[test]
    fn edge_placement() {
        // mid=0.50, v=4.5c, buffer=0.5c → s_target=4.0c → px = 0.50 - 0.04 = 0.46
        let action = decide_side_action(
            true,
            dec!(0.50),
            Some(dec!(0.30)), /* best_bid far from the band */
            dec!(4.5),
            dec!(0.5),
            dec!(0.01),
            true,
            dec!(1.0),
            None,
        );
        assert_eq!(action, SideAction::Requote(dec!(0.46)));
    }

    #[test]
    fn placement_never_crosses_or_equals_mid() {
        // v so large that s_target would fall beyond or above the mid — must be rejected (Cancel/Noop, never Requote >= mid).
        let action = decide_side_action(
            true,
            dec!(0.50),
            None,
            dec!(0.1), /* small v: s_target negative after the buffer */
            dec!(0.5),
            dec!(0.01),
            true,
            dec!(1.0),
            None,
        );
        assert_ne!(action, SideAction::Requote(dec!(0.50)));
        if let SideAction::Requote(px) = action {
            assert!(px < dec!(0.50), "the bid must never reach the mid");
        }
    }

    // 3. queue_behind_places_behind_best_bid
    #[test]
    fn queue_behind_places_behind_best_bid() {
        // mid=0.50, v=4.5c, buffer=0.5c → target=0.46; band=[0.455,0.50].
        // best_bid=0.458 (fine tick 0.001) falls INSIDE the band and below
        // the natural target → we must place behind it.
        let action = decide_side_action(
            true,
            dec!(0.50),
            Some(dec!(0.458)),
            dec!(4.5),
            dec!(0.5),
            dec!(0.001),
            true,
            dec!(1.0),
            None,
        );
        assert_eq!(
            action,
            SideAction::Requote(dec!(0.457)),
            "px = best_bid - tick"
        );
    }

    #[test]
    fn queue_behind_disabled_ignores_best_bid() {
        let action = decide_side_action(
            true,
            dec!(0.50),
            Some(dec!(0.458)),
            dec!(4.5),
            dec!(0.5),
            dec!(0.001),
            false,
            dec!(1.0),
            None,
        );
        assert_eq!(
            action,
            SideAction::Requote(dec!(0.46)),
            "without queue_behind the existing queue is ignored"
        );
    }

    // 4. two_sided_rule_mid_zone_one_side_scores
    #[test]
    fn two_sided_rule_mid_zone_one_side_scores() {
        let q = q_min(dec!(30), Decimal::ZERO, dec!(0.50), dec!(0.10), dec!(0.90));
        assert_eq!(
            q,
            dec!(10),
            "mean zone with a single live side must give Q_one/3, not 0"
        );
    }

    // 5. two_sided_rule_extreme_requires_both
    #[test]
    fn two_sided_rule_extreme_requires_both() {
        let q = q_min(dec!(30), Decimal::ZERO, dec!(0.95), dec!(0.10), dec!(0.90));
        assert_eq!(
            q,
            Decimal::ZERO,
            "at the extreme a single side scores nothing"
        );
    }

    // 6. skip_mid_out_of_range
    #[test]
    fn skip_mid_out_of_range() {
        let cfg = rf2_cfg();
        let registry =
            registry_with_rewards("cid-ext", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(10)));
        let mut rf = rf_with(cfg, registry);
        // mid_yes = (0.94+0.96)/2 = 0.95 > rf_mid_max=0.90
        let book = make_daily_book("cid-ext", "0.94", "0.96", "0.03", "0.05", 7_200);
        let intents = rf.on_book(&book, &make_fresh_vol());
        assert!(
            intents.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "mid outside [0.10,0.90] must not post any GTD"
        );
    }

    // 7. condemns_leg_by_cvd
    #[test]
    fn condemns_leg_by_cvd() {
        let cfg = RewardFarmConfig::default();
        assert!(
            toxic_flow_favors_no(Some(dec!(0.40)), None, &cfg),
            "cvd_r >= rf_cvd_toxic must condemn NO"
        );
        assert!(
            !toxic_flow_favors_yes(Some(dec!(0.40)), None, &cfg),
            "YES must remain healthy"
        );
    }

    // 8. condemns_leg_by_liq
    #[test]
    fn condemns_leg_by_liq() {
        let cfg = RewardFarmConfig {
            rf_liq_toxic_notional: dec!(1000),
            ..RewardFarmConfig::default()
        };
        assert!(
            toxic_flow_favors_yes(None, Some(dec!(-1500)), &cfg),
            "very negative liq (bearish cascade) must condemn YES"
        );
        assert!(
            !toxic_flow_favors_no(None, Some(dec!(-1500)), &cfg),
            "NO must remain healthy"
        );
    }

    #[test]
    fn liq_toxic_disabled_by_default() {
        let cfg = RewardFarmConfig::default(); // rf_liq_toxic_notional = 0 = disabled
        assert!(!toxic_flow_favors_yes(None, Some(dec!(-999_999)), &cfg));
        assert!(!toxic_flow_favors_no(None, Some(dec!(999_999)), &cfg));
    }

    // 9. pause_by_vol_ratio
    #[test]
    fn pause_by_vol_ratio() {
        let cfg = rf2_cfg();
        let registry =
            registry_with_rewards("cid-volr", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(10)));
        let mut rf = rf_with(cfg.clone(), registry.clone());
        let book = make_daily_book("cid-volr", "0.30", "0.70", "0.30", "0.70", 7_200);

        // First tick: places both levels with healthy vol.
        let i1 = rf.on_book(&book, &make_fresh_vol());
        assert_eq!(
            i1.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2
        );

        // Vol spike (σ5/σ60 >= rf_vol_ratio_pause=2.0) → cancel BOTH legs.
        let vol_spike = VolSnapshot {
            asset: "btc".into(),
            sigma_short: Some(dec!(0.01)),
            sigma_long: Some(dec!(0.001)),
            last_price: dec!(100_000),
            updated_at: Instant::now(),
            ..VolSnapshot::default()
        };
        let i2 = rf.on_book(&book, &vol_spike);
        assert_eq!(
            i2.iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "the vol spike must cancel YES and NO"
        );
        assert!(i2.iter().all(|i| !matches!(i.side, OrderSide::Buy)));
    }

    // 10. requote_only_if_mid_moved
    #[test]
    fn requote_only_if_mid_moved() {
        let cfg = rf2_cfg();
        let registry =
            registry_with_rewards("cid-req", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(10)));
        let mut rf = rf_with(cfg, registry);
        let book = make_daily_book("cid-req", "0.30", "0.70", "0.30", "0.70", 7_200);
        let vol = make_fresh_vol();

        let i1 = rf.on_book(&book, &vol);
        assert!(i1
            .iter()
            .any(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "yes-tok"));

        // Same book (stable mid) → 0 new intents.
        let i2 = rf.on_book(&book, &vol);
        assert!(i2.is_empty(), "without a mid move it must not re-place");

        // YES mid moves >= rf_requote_threshold_cents (1c) → cancel+repost of the YES side.
        let mut book_moved = book.clone();
        book_moved.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.32),
                size: dec!(100),
            }],
        };
        book_moved.yes.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.72),
                size: dec!(100),
            }],
        };
        let i3 = rf.on_book(&book_moved, &vol);
        assert_eq!(
            i3.iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel) && i.token_id == "yes-tok")
                .count(),
            1
        );
        assert_eq!(
            i3.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "yes-tok")
                .count(),
            1
        );
    }

    // 11. on_fill_cancels_opposite_leg_and_marks_burned
    #[test]
    fn on_fill_cancels_opposite_leg_and_marks_burned() {
        let cfg = rf2_cfg(); // rf_max_fills_per_bracket = 1 (default)
        let registry =
            registry_with_rewards("cid-fill", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(10)));
        let mut rf = rf_with(cfg, registry);
        let book = make_daily_book("cid-fill", "0.30", "0.70", "0.30", "0.70", 7_200);
        let vol = make_fresh_vol();

        let i1 = rf.on_book(&book, &vol);
        assert_eq!(
            i1.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "both legs must remain alive"
        );

        rf.on_fill(&Fill {
            hedge_id: None,
            market_cid: "cid-fill".into(),
            token_id: "yes-tok".into(),
            exec_id: None,
            local_order_id: None,
            side: FillSide::Buy,
            price: dec!(0.46),
            size: dec!(50),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RewardFarm,
            clear_at: book.clear_at,
            outcome_label: None,
        });

        let i2 = rf.on_book(&book, &vol);
        assert_eq!(
            i2.iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel) && i.token_id == "no-tok")
                .count(),
            1,
            "must immediately cancel the opposing leg (NO)"
        );

        // Resolve the imbalance (dump/hold already emitted) → simulate the sell of the excess.
        rf.on_fill(&Fill {
            hedge_id: None,
            market_cid: "cid-fill".into(),
            token_id: "yes-tok".into(),
            exec_id: None,
            local_order_id: None,
            side: FillSide::Sell,
            price: dec!(0.46),
            size: dec!(50),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RewardFarm,
            clear_at: book.clear_at,
            outcome_label: None,
        });

        let i3 = rf.on_book(&book, &vol);
        assert!(
            i3.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "the BURNED bracket must not post again after the inventory is resolved"
        );
    }

    // With rf_max_fills_per_bracket > 1 the
    // bracket does not burn on the first fill; with the unwind already sent and the sell
    // still in flight the inventory remains unbalanced — posting must not
    // be reached (invariant "unbalanced ⇒ do not post").
    #[test]
    fn imbalance_with_unwind_in_flight_does_not_repost() {
        let cfg = RewardFarmConfig {
            rf_max_fills_per_bracket: 2,
            ..rf2_cfg()
        };
        let registry =
            registry_with_rewards("cid-in-flight", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(10)));
        let mut rf = rf_with(cfg, registry);
        let book = make_daily_book("cid-in-flight", "0.30", "0.70", "0.30", "0.70", 7_200);
        let vol = make_fresh_vol();

        let i1 = rf.on_book(&book, &vol);
        assert_eq!(
            i1.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2
        );

        rf.on_fill(&Fill {
            hedge_id: None,
            market_cid: "cid-in-flight".into(),
            token_id: "yes-tok".into(),
            exec_id: None,
            local_order_id: None,
            side: FillSide::Buy,
            price: dec!(0.46),
            size: dec!(50),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RewardFarm,
            clear_at: book.clear_at,
            outcome_label: None,
        });

        // First tick after the fill: cancels the opposing leg and emits the unwind.
        let i2 = rf.on_book(&book, &vol);
        assert_eq!(
            i2.iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel) && i.token_id == "no-tok")
                .count(),
            1
        );

        // Second tick with the unwind sell still in flight (inventory still
        // unbalanced): must not repost ANY buy GTD.
        let i3 = rf.on_book(&book, &vol);
        assert!(
            i3.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "with unwind in flight it must not repost (PM behavior prohibited by the spec)"
        );
    }

    // Bug fix: a PARTIAL fill of the dump (real FAK: it fills what
    // there is in depth and the rest is lost, it is not retried on its own) left
    // `unwind_sent` stuck at `true` forever because the reset required
    // `yes_filled == no_filled` exactly — the remainder was orphaned without
    // `on_book` evaluating it again. Verifies that a partial Sell DOES
    // retry the unwind of the rest on the next tick.
    #[test]
    fn unwind_with_partial_fill_retries_remainder() {
        let cfg = rf2_cfg();
        let registry = registry_with_rewards(
            "cid-partial",
            Some(dec!(50)),
            Some(dec!(4.5)),
            Some(dec!(10)),
        );
        let mut rf = rf_with(cfg, registry);
        let book = make_daily_book("cid-partial", "0.30", "0.70", "0.30", "0.70", 7_200);
        let vol = make_fresh_vol();

        rf.on_book(&book, &vol);

        rf.on_fill(&Fill {
            hedge_id: None,
            market_cid: "cid-partial".into(),
            token_id: "yes-tok".into(),
            exec_id: None,
            local_order_id: None,
            side: FillSide::Buy,
            price: dec!(0.46),
            size: dec!(50),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RewardFarm,
            clear_at: book.clear_at,
            outcome_label: None,
        });

        // First tick: cancels the opposing leg and triggers the dump (no FV → DumpTaker).
        let i2 = rf.on_book(&book, &vol);
        assert!(
            i2.iter()
                .any(|i| matches!(i.side, OrderSide::Sell { .. }) && i.token_id == "yes-tok"),
            "must emit the taker dump of the excess"
        );

        // The dump only fills 20 of the 50 shares of excess (FAK with limited depth).
        rf.on_fill(&Fill {
            hedge_id: None,
            market_cid: "cid-partial".into(),
            token_id: "yes-tok".into(),
            exec_id: None,
            local_order_id: None,
            side: FillSide::Sell,
            price: dec!(0.46),
            size: dec!(20),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RewardFarm,
            clear_at: book.clear_at,
            outcome_label: None,
        });

        // Next tick: the inventory is still unbalanced (30 of excess) — it must
        // retry the dump, not stay in the dead "unwind in flight" branch.
        let i3 = rf.on_book(&book, &vol);
        assert!(
            i3.iter()
                .any(|i| matches!(i.side, OrderSide::Sell { .. }) && i.token_id == "yes-tok"),
            "a partial dump fill must retry the remainder, not abandon it"
        );
    }

    // 12. unwind_hold_if_fv_high / unwind_dump_if_fv_low
    #[test]
    fn unwind_hold_if_fv_high() {
        let cfg = RewardFarmConfig::default(); // rf_unwind_hold_fv = 0.85
        assert_eq!(
            decide_unwind_rf(Some(dec!(0.90)), 3600, &cfg),
            UnwindActionRf::Hold
        );
    }

    #[test]
    fn unwind_dump_if_fv_low() {
        let cfg = RewardFarmConfig::default(); // rf_unwind_dump_fv = 0.55
        assert_eq!(
            decide_unwind_rf(Some(dec!(0.30)), 3600, &cfg),
            UnwindActionRf::DumpTaker
        );
    }

    #[test]
    fn unwind_repost_in_neutral_zone_with_time() {
        let cfg = RewardFarmConfig::default();
        assert_eq!(
            decide_unwind_rf(Some(dec!(0.70)), 3600, &cfg),
            UnwindActionRf::RepostMaker
        );
    }

    #[test]
    fn unwind_without_fv_dumps() {
        let cfg = RewardFarmConfig::default();
        assert_eq!(
            decide_unwind_rf(None, 3600, &cfg),
            UnwindActionRf::DumpTaker
        );
    }

    // 13. gate_G0_without_rate_does_not_post
    #[test]
    fn gate_without_rate_does_not_post() {
        let cfg = rf2_cfg();
        let registry = registry_with_rewards("cid-g0", Some(dec!(50)), Some(dec!(4.5)), None); // no daily_rate
        let mut rf = rf_with(cfg, registry);
        let book = make_daily_book("cid-g0", "0.30", "0.70", "0.30", "0.70", 7_200);
        let intents = rf.on_book(&book, &make_fresh_vol());
        assert!(
            intents.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "without a confirmed rewards_daily_rate (G0) it must not post anything"
        );
    }

    #[test]
    fn gate_rate_below_min_does_not_post() {
        let cfg = RewardFarmConfig {
            rf_min_daily_rate: dec!(5),
            ..rf2_cfg()
        };
        let registry =
            registry_with_rewards("cid-g0b", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(1)));
        let mut rf = rf_with(cfg, registry);
        let book = make_daily_book("cid-g0b", "0.30", "0.70", "0.30", "0.70", 7_200);
        let intents = rf.on_book(&book, &make_fresh_vol());
        assert!(intents.iter().all(|i| !matches!(i.side, OrderSide::Buy)));
    }

    // 14. pm_mm_exclusivity
    #[test]
    fn pm_mm_exclusivity() {
        let cfg = rf2_cfg(); // rf_require_no_pm_mm = true (default)
        let registry =
            registry_with_rewards("cid-excl", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(10)));
        let ledger = BracketLedger::new();
        // Another strategy (e.g. PM/MM) already has exposure in this bracket.
        ledger.try_add_exposure(
            "cid-excl",
            crate::time::now().timestamp() + 86_400,
            "other-token",
            dec!(20),
            Decimal::ZERO,
            crate::types::StrategyId::ParityMaker,
        );
        let mut rf = RewardFarm::new(cfg, registry, ledger);
        let book = make_daily_book("cid-excl", "0.30", "0.70", "0.30", "0.70", 7_200);
        let intents = rf.on_book(&book, &make_fresh_vol());
        assert!(
            intents.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "bracket with prior exposure from another strategy must skip"
        );
    }

    // 14b. pm_mm_exclusivity with a balanced box:
    // `net_exposure` of a YES=NO box yields 0 but there IS another strategy owning
    // the bracket — GROSS exposure (`gross_exposure`) must be used.
    #[test]
    fn pm_mm_exclusivity_balanced_box_does_not_post() {
        let cfg = rf2_cfg();
        let registry = registry_with_rewards(
            "cid-excl-bal",
            Some(dec!(50)),
            Some(dec!(4.5)),
            Some(dec!(10)),
        );
        let ledger = BracketLedger::new();
        // PM balanced box: YES=23/NO=23 → net_exposure = 0, but gross > 0.
        ledger.try_add_pair_exposure(
            "cid-excl-bal",
            crate::time::now().timestamp() + 86_400,
            ("yes-tok", dec!(23), crate::types::StrategyId::ParityMaker),
            ("no-tok", dec!(23), crate::types::StrategyId::ParityMaker),
            Decimal::ZERO,
        );
        assert_eq!(
            ledger.net_exposure("cid-excl-bal"),
            Decimal::ZERO,
            "precondition: the box's net is 0"
        );
        let mut rf = RewardFarm::new(cfg, registry, ledger);
        let book = make_daily_book("cid-excl-bal", "0.30", "0.70", "0.30", "0.70", 7_200);
        let intents = rf.on_book(&book, &make_fresh_vol());
        assert!(
            intents.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "balanced box of another strategy (net=0, gross>0) must keep blocking RF"
        );
    }

    // 15. sizing_is_exact_min_size
    #[test]
    fn sizing_is_exact_min_size() {
        let cfg = rf2_cfg();
        let registry =
            registry_with_rewards("cid-size", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(10)));
        let mut rf = rf_with(cfg, registry);
        let book = make_daily_book("cid-size", "0.30", "0.70", "0.30", "0.70", 7_200);
        let intents = rf.on_book(&book, &make_fresh_vol());
        let buys: Vec<_> = intents
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(buys.len(), 2);
        for b in buys {
            assert_eq!(
                b.size,
                dec!(50),
                "order_size_shares must be exactly rewards_min_size"
            );
        }
    }

    // 16. reward_proxy_share_correct
    #[test]
    fn reward_proxy_share_correct() {
        assert_eq!(share_proxy(dec!(30), dec!(70)), dec!(0.3));
        assert_eq!(share_proxy(Decimal::ZERO, Decimal::ZERO), Decimal::ZERO);
    }

    #[test]
    fn reward_proxy_delta_scales_by_dt_86400() {
        // rate=10, share=0.5, dt=8640s (=86400/10) → delta = 10*0.5/10 = 0.5
        let delta = reward_proxy_delta(dec!(10), dec!(0.5), 8_640);
        assert_eq!(delta, dec!(0.5));
    }

    // ── Bracket duration ─────────────────────────────────────────

    #[test]
    fn duration_5m_not_allowed_by_default() {
        let cfg = RewardFarmConfig::default();
        let book = make_book("cid-5m", "0.49", "0.51", "0.49", "0.51", 200);
        let mut b = book;
        b.clear_at = crate::time::now().timestamp() + 200;
        b.market_open_ts = b.clear_at - 300; // 5m
        assert!(!duration_allowed(&b, &cfg.rf_durations));
    }

    #[test]
    fn duration_daily_allowed_by_default() {
        let cfg = RewardFarmConfig::default();
        let book = make_daily_book("cid-d", "0.49", "0.51", "0.49", "0.51", 7_200);
        assert!(duration_allowed(&book, &cfg.rf_durations));
    }

    // ── Abandon at extremes (mid outside [0.10,0.90]) after having posted ──

    #[test]
    fn abandons_both_legs_on_extreme_cross() {
        let cfg = rf2_cfg();
        let registry =
            registry_with_rewards("cid-abn", Some(dec!(50)), Some(dec!(4.5)), Some(dec!(10)));
        let mut rf = rf_with(cfg, registry);
        let book = make_daily_book("cid-abn", "0.30", "0.70", "0.30", "0.70", 7_200);
        let vol = make_fresh_vol();
        let i1 = rf.on_book(&book, &vol);
        assert_eq!(
            i1.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2
        );

        // The bracket drifts toward the extreme (mid_yes = 0.95).
        let book_ext = make_daily_book("cid-abn", "0.94", "0.96", "0.03", "0.05", 7_200);
        let i2 = rf.on_book(&book_ext, &vol);
        assert_eq!(
            i2.iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "must cancel both legs when crossing the extreme"
        );
    }

    // ── Bug review findings ────────────────────────────────────────

    // leg that does not fit in max_order_usdc: skip without touching the state
    // (the executor would silently reject it, see order/executor.rs).
    #[test]
    fn leg_that_does_not_fit_in_max_order_usdc_does_not_post_and_leaves_state_none() {
        let cfg = RewardFarmConfig {
            limits: crate::config::StrategyLimits {
                enabled: true,
                max_order_usdc: dec!(10),
                ..Default::default()
            },
            min_secs_to_clear: 60,
            ..RewardFarmConfig::default()
        };
        let registry = registry_with_rewards(
            "cid-max-usdc",
            Some(dec!(50)),
            Some(dec!(4.5)),
            Some(dec!(10)),
        );
        let mut rf = rf_with(cfg, registry);
        // mid=0.70 (YES) / 0.30 (NO) → edges ~0.66/~0.26 → cost ~33/~13,
        // both above max_order_usdc=10.
        let book = make_daily_book("cid-max-usdc", "0.68", "0.72", "0.28", "0.32", 7_200);
        let intents = rf.on_book(&book, &make_fresh_vol());
        assert!(
            intents.iter().all(|i| !matches!(i.side, OrderSide::Buy)),
            "a leg exceeding max_order_usdc must not post a GTD"
        );

        let state = rf
            .markets
            .get("cid-max-usdc")
            .expect("the bracket must have state after the tick");
        assert!(state.yes_quote.is_none(), "no ghost quote on YES");
        assert!(state.no_quote.is_none(), "no ghost quote on NO");
    }

    // rf_max_total_usdc sub-cap counted within the SAME tick: with a cap
    // just enough for one leg, the second leg of the same tick must be rejected.
    #[test]
    fn subcap_same_tick_rejects_second_leg() {
        let cfg = RewardFarmConfig {
            rf_max_total_usdc: dec!(25),
            ..rf2_cfg()
        };
        let registry = registry_with_rewards(
            "cid-subcap",
            Some(dec!(50)),
            Some(dec!(4.5)),
            Some(dec!(10)),
        );
        let mut rf = rf_with(cfg, registry);
        // mid=0.50 on both sides, best_bid far from the band → both legs
        // would cost ~22-23 each; the cap of 25 only allows the first.
        let book = make_daily_book("cid-subcap", "0.40", "0.60", "0.40", "0.60", 7_200);
        let intents = rf.on_book(&book, &make_fresh_vol());
        let buys: Vec<_> = intents
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(
            buys.len(),
            1,
            "with the sub-cap just enough for one leg, the second leg of the SAME tick must be rejected"
        );
        assert_eq!(
            buys[0].token_id, "yes-tok",
            "YES is evaluated first and does fit within the cap"
        );
    }

    // A requote blocked by the sub-cap must not leave the oldest GTD alive
    // (it could end up crossed with the new mid) — better no order than one at a dead price.
    #[test]
    fn requote_blocked_by_subcap_cancels_old_gtd() {
        let cfg = RewardFarmConfig {
            rf_max_total_usdc: dec!(47),
            ..rf2_cfg()
        };
        let registry = registry_with_rewards(
            "cid-req-cap",
            Some(dec!(50)),
            Some(dec!(4.5)),
            Some(dec!(10)),
        );
        let mut rf = rf_with(cfg, registry);
        let vol = make_fresh_vol();

        // Tick 1: mid=0.50 on both sides, both legs fit (~23+23=46 <= 47).
        let book1 = make_daily_book("cid-req-cap", "0.40", "0.60", "0.40", "0.60", 7_200);
        let i1 = rf.on_book(&book1, &vol);
        assert_eq!(
            i1.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2
        );

        // Tick 2: the YES mid rises to 0.60 → more expensive requote (~28 instead of
        // ~23); the projected (46 − 23 + 28 = 51) exceeds the sub-cap (47).
        let mut book2 = book1.clone();
        book2.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.50),
                size: dec!(100),
            }],
        };
        book2.yes.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.70),
                size: dec!(100),
            }],
        };
        let i2 = rf.on_book(&book2, &vol);

        assert!(
            i2.iter()
                .all(|i| !(matches!(i.side, OrderSide::Buy) && i.token_id == "yes-tok")),
            "the requote blocked by the sub-cap must not emit a new Buy"
        );
        assert_eq!(
            i2.iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel) && i.token_id == "yes-tok")
                .count(),
            1,
            "must cancel the oldest GTD instead of leaving it alive at a dead price"
        );

        let state = rf.markets.get("cid-req-cap").unwrap();
        assert!(
            state.yes_quote.is_none(),
            "the YES quote state must be clean after the cancel"
        );
    }

    // RepostMaker uses max(best_ask, avg_entry + edge).
    #[test]
    fn repost_maker_uses_avg_entry_if_greater_than_best_ask() {
        // avg_entry=0.50, tick=0.01 → floor=0.51 > best_ask=0.48 → uses the floor.
        assert_eq!(
            repost_maker_price(dec!(0.48), Some(dec!(0.50)), dec!(0.01)),
            Some(dec!(0.51))
        );
    }

    #[test]
    fn repost_maker_uses_best_ask_if_already_above_floor() {
        // best_ask=0.60 > avg_entry+edge=0.51 → uses best_ask (never gives away spread).
        assert_eq!(
            repost_maker_price(dec!(0.60), Some(dec!(0.50)), dec!(0.01)),
            Some(dec!(0.60))
        );
    }

    #[test]
    fn repost_maker_without_avg_entry_uses_best_ask() {
        assert_eq!(
            repost_maker_price(dec!(0.55), None, dec!(0.01)),
            Some(dec!(0.55))
        );
    }

    #[test]
    fn repost_maker_invalid_price_gives_none() {
        // avg_entry + tick falls outside (0,1) after truncation → None, no repost.
        assert_eq!(
            repost_maker_price(dec!(0.999), Some(dec!(0.999)), dec!(0.01)),
            None
        );
    }

    // ── The real repo TOML must parse with the exact names/defaults ──

    #[test]
    fn reward_farm_toml_parses_with_exact_names_from_spec() {
        use figment::providers::{Format, Toml};
        use figment::Figment;

        let cfg: RewardFarmConfig = Figment::new()
            .merge(Toml::file("config/strategies/reward_farm.toml").nested())
            .select("reward_farm")
            .extract()
            .expect("reward_farm.toml must deserialize into RewardFarmConfig");

        assert!(
            !cfg.limits.enabled,
            "enabled=false by default (process-level gate)"
        );
        assert_eq!(cfg.limits.max_order_usdc, dec!(46));
        assert_eq!(
            cfg.rf_durations,
            vec!["daily".to_string(), "weekly".to_string()]
        );
        assert_eq!(cfg.rf_max_markets, 5);
        assert_eq!(cfg.rf_mid_min, dec!(0.10));
        assert_eq!(cfg.rf_mid_max, dec!(0.90));
        assert_eq!(cfg.rf_edge_buffer_cents, dec!(0.5));
        assert_eq!(cfg.rf_requote_threshold_cents, dec!(1.0));
        assert!(cfg.rf_queue_behind);
        assert!(cfg.rf_single_level);
        assert_eq!(cfg.rf_size_mode, "min_size");
        assert_eq!(cfg.rf_max_total_usdc, dec!(300));
        assert!(cfg.rf_require_no_pm_mm);
        assert_eq!(cfg.rf_cvd_toxic, dec!(0.35));
        assert_eq!(cfg.rf_liq_toxic_notional, Decimal::ZERO);
        // Lowered from 2.0 to 1.6, recalibrated against realized_vol_per_sec
        // (the new σ̂ estimator — see config/strategies/reward_farm.toml).
        assert_eq!(cfg.rf_vol_ratio_pause, dec!(1.6));
        assert_eq!(cfg.rf_max_fills_per_bracket, 1);
        assert_eq!(cfg.rf_unwind_hold_fv, dec!(0.85));
        assert_eq!(cfg.rf_unwind_dump_fv, dec!(0.55));
        assert_eq!(cfg.rf_repost_min_secs, 300);
        assert_eq!(cfg.min_secs_to_clear, 3600);
        assert_eq!(cfg.max_secs_to_clear, 0);
    }
}
