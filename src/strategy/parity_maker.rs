//! Parity Maker (PM) strategy.
//!
//! Posts GTD bids on YES and NO under the constraint:
//!   bid_YES + bid_NO ≤ 1 - margin
//!
//! If both legs fill → guaranteed profit = 1 - (bid_YES + bid_NO) > 0.
//! Periodic requote when the mid moves by more than requote_threshold_bps.
//!
//! The oracle is no longer ignored. The fair value (`fair_value_yes`) is used
//! to (a) skew the bid of the favored leg without breaking the parity
//! constraint, (b) avoid quoting the disfavored leg when the edge is extreme,
//! (c) pause quotes on a volatility spike, and (d) decide the unwind of a
//! single unfilled leg: hold to settlement if the model favors it
//! (selling taker gives away 10% of the upside to the fee), dump taker if it
//! disfavors it, or re-post a maker ask if there is time and the zone is neutral.

use crate::time::Instant;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    config::{defaults, StrategyLimits},
    fair_value::fair_value_yes,
    types::{
        BookSnapshot, Fill, FillSide, HalfBook, OrderIntent, OrderSide, StrategyId, VolSnapshot,
    },
};

use super::Strategy;

// ── Parity Maker configuration ──────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ParityMakerConfig {
    #[serde(flatten)]
    pub limits: StrategyLimits,
    /// Seconds with only one leg filled before giving up: cancel the
    /// other GTD and sell the excess (taker). Adverse selection fills the
    /// losing leg first; without this exit PM was left directional (WR 28.6%).
    /// 0 = disabled. While waiting, the requote tries to complete the pair.
    #[serde(default = "defaults::pm_single_leg_unwind_secs")]
    pub single_leg_unwind_secs: u64,
    /// Minimum bid for the unwind: with dust bids there is nothing to save.
    #[serde(default = "defaults::pm_min_unwind_bid")]
    pub min_unwind_bid: Decimal,
    #[serde(default = "defaults::pm_margin")]
    pub margin: Decimal,
    #[serde(default = "defaults::pm_min_leg_price")]
    pub min_leg_price: Decimal,
    #[serde(default = "defaults::pm_order_size_shares")]
    pub order_size_shares: Decimal,
    #[serde(default = "defaults::pm_requote_threshold_bps")]
    pub requote_threshold_bps: u32,
    /// Requote tolerance in market ticks (order refresh tolerance):
    /// if the bid already resting on a leg is at most N ticks from the
    /// desired price, that leg is NOT cancelled and re-posted. The tolerance never
    /// keeps an invalid pair: the sum of what stays resting must still be ≤ 1 −
    /// margin. 0 = disabled (previous behavior).
    #[serde(default)]
    pub requote_tolerance_ticks: u32,
    #[serde(default = "defaults::pm_pair_ttl_secs")]
    pub pair_ttl_secs: u64,
    /// Minimum |FV-0.5| to skew the bid toward the favored leg
    /// (parity-neutral: whatever is added to one side is subtracted from the other). 0 = disabled.
    #[serde(default = "defaults::pm_fv_skew_threshold")]
    pub fv_skew_threshold: Decimal,
    /// Skew offset in bps (100 = ±0.01) applied to the bid of each leg.
    #[serde(default = "defaults::pm_fv_skew_bps")]
    pub fv_skew_bps: u32,
    /// |FV-0.5| from which the disfavored leg is not quoted (single-leg PM),
    /// also skipping the sum constraint. 0 = disabled.
    #[serde(default = "defaults::pm_fv_no_quote_threshold")]
    pub fv_no_quote_threshold: Decimal,
    /// Maximum σ_5s/σ_60s ratio before pausing quotes (adverse selection).
    /// 0 = disabled (same pattern as MarketMakingConfig).
    #[serde(default)]
    pub vol_ratio_threshold: Decimal,
    /// FV of the retained leg (probability it wins) from which we hold to settlement
    /// instead of selling taker (selling taker hands the upside
    /// over to the 10% fee).
    #[serde(default = "defaults::pm_unwind_hold_fv")]
    pub unwind_hold_fv: Decimal,
    /// FV of the retained leg below which we sell taker (dump).
    /// Between this threshold and `unwind_hold_fv` (neutral zone) we re-post maker if
    /// there is enough time left (`unwind_repost_min_secs`).
    #[serde(default = "defaults::pm_unwind_dump_fv")]
    pub unwind_dump_fv: Decimal,
    /// Minimum remaining seconds to re-post a maker ask instead of
    /// selling taker in the unwind's neutral zone.
    #[serde(default = "defaults::pm_unwind_repost_min_secs")]
    pub unwind_repost_min_secs: u64,
    /// Edge over avg_entry when re-posting the unwind's maker ask
    /// (`price = avg_entry + this value`, capped by the best ask and 0.99).
    #[serde(default = "defaults::pm_unwind_maker_edge")]
    pub unwind_maker_edge: Decimal,
    /// Window (seconds) over which the oracle spot's movement is measured.
    /// This separates the impatient seller (spot flat, book falls → benign
    /// fill) from informed flow (spot running, book follows → we get filled
    /// on the leg about to lose). 0 = directional gate disabled.
    #[serde(default = "defaults::pm_oracle_drift_secs")]
    pub oracle_drift_secs: u64,
    /// |movement| of the spot over that window, in bps, from which the fill
    /// is considered informed and the doomed leg (the one the movement is sinking) is NOT quoted.
    /// Measured on the underlying, not on the token.
    #[serde(default = "defaults::pm_oracle_drift_bps")]
    pub oracle_drift_bps: u32,
    /// Minimum expected movement of the underlying (in bps) to OPEN a position
    /// in a new bracket: the time barrier, derived instead of set by eye.
    ///
    /// `expected move = σ · √t` with the time remaining until the close. When
    /// that gets small the bracket is already decided: the price will not move
    /// any more, the loser can only fall toward 0 through FV convergence, and no
    /// flow is left to complete the box. That is where PM must stop entering.
    ///
    /// It is preferable to a threshold in seconds because it adapts by itself: with high volatility
    /// the price stays alive near the close (entry can be later); with low volatility
    /// the bracket dies earlier. A "do not enter with <240s" only holds
    /// for the σ of the day it was measured on and for brackets of that duration.
    ///
    /// Calibration (5s buy markout by remaining time): -820 bps with
    /// >240s and box ratio 19/23; -4622 bps with <60s and ratio 2/5.
    /// > Completing a box already started remains allowed below the threshold.
    /// > 0 = no restriction.
    #[serde(default = "defaults::pm_min_expected_move_bps")]
    pub min_expected_move_bps: u32,
    /// Minimum spread of each leg to OPEN. The spread is the maker's
    /// premium: with 1¢ there is no room for an impatient seller to cross to
    /// our bid, so only informed flow arrives. Without spread there is no
    /// business, only adverse selection. It does not apply to completing a box
    /// already started. 0 = disabled.
    #[serde(default = "defaults::pm_min_spread")]
    pub min_spread: Decimal,
    /// Shares of bids ahead of ours (at a price ≥) from which we do not OPEN: a bid
    /// behind a lot of liquidity never fills; that is what sank an earlier
    /// version of this strategy. 0 = disabled.
    #[serde(default = "defaults::pm_max_queue_ahead_shares")]
    pub max_queue_ahead_shares: Decimal,
    /// Maximum ratio between one leg's queue and the other's for OPENING.
    /// If one leg has much more queue ahead than the other, it fills late or never
    /// while the other is already filling: the box is born unbalanced, which is the
    /// orphan (−11¢) to avoid. 0 = disabled.
    #[serde(default = "defaults::pm_queue_imbalance_ratio")]
    pub queue_imbalance_ratio: Decimal,
    /// Window (seconds) over which this bracket's movement is compared with its
    /// siblings of the same asset. If all siblings move at once, the underlying moved
    /// (informed flow, the opposite leg is doomed);
    /// if only this book moves it is idiosyncratic flow, the impatient seller's, which is what a maker lives on.
    /// 0 = signal disabled.
    #[serde(default = "defaults::pm_xbracket_window_secs")]
    pub xbracket_window_secs: u64,
    /// Minimum own movement (in probability bps; 100 = 1 point) for the signal
    /// to be evaluated at all. Below that it is tick noise.
    #[serde(default = "defaults::pm_xbracket_move_bps")]
    pub xbracket_move_bps: u32,
    /// Fraction of its own movement that a sibling bracket must replicate
    /// (same sign) to count as "in agreement". With half of the siblings in
    /// agreement, the movement is declared common.
    #[serde(default = "defaults::pm_xbracket_common_ratio")]
    pub xbracket_common_ratio: Decimal,
    /// Seconds with a single leg before chasing the missing one by quoting it
    /// at the ceiling of the residual budget instead of the proportional split.
    /// The extra margin is given up to maximize P(complete): the box pays +margin,
    /// the orphan costs −11c. It never exceeds the ceiling — the box closes at `1 − margin`.
    ///
    /// MUST be smaller than `single_leg_unwind_secs` or it never acts (the
    /// unwind resolves the leg first). 0 = chase disabled.
    #[serde(default = "defaults::pm_chase_after_secs")]
    pub chase_after_secs: u64,
    /// Number of unwind dumps (forced sell of the losing leg) after which a
    /// bracket is considered BURNED and is not reopened. A dump is hard evidence of
    /// toxic flow: the losing leg filled and the winning one did not arrive. Reopening is
    /// buying the losing leg again. Measured: all of PM's loss was in
    /// the unwind SELLs; boxes held to settlement won. It only blocks
    /// OPENING, never completing a box already started. 0 = disabled.
    #[serde(default = "defaults::pm_max_dumps_per_bracket")]
    pub max_dumps_per_bracket: u32,
    /// Maximum number of results (boxed/orphan) retained per
    /// bucket (asset, duration) to mine whether there is a directional or
    /// streak bias that separates boxes from orphans ex-ante. The earlier diagnosis
    /// found no single price feature that did so. 0 = disabled
    /// (does not accumulate history; the `hist_*` fields of the log stay empty).
    #[serde(default = "defaults::pm_bracket_history_len")]
    pub bracket_history_len: usize,
    /// Minimum nominal duration of the bracket
    /// (`clear_at - market_open_ts`) to OPEN a new box. It excludes the
    /// 5-minute brackets (300s), whose clock barrier leaves them
    /// orphaned much more often than the 15-minute ones (900s) — mining
    /// duration AUC 0.774, median 900s in complete boxes vs 300s in
    /// orphans. It only blocks OPENING; completing a box already started
    /// goes through it (same pattern as `min_expected_move_bps`). duration == 0
    /// (unknown mark) never blocks. 0 = disabled.
    /// Revert if the markout does not improve by ≥ 300 bps with n ≥ 80 boxes.
    #[serde(default = "defaults::pm_min_duration_secs")]
    pub pm_min_duration_secs: u64,
}

impl super::StrategyConfig for ParityMakerConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

// ── Per-bracket state ────────────────────────────────────────────────────────

struct PMState {
    clear_at: i64,
    posted_at: Option<Instant>,
    posted_mid: Option<Decimal>,
    yes_token: String,
    no_token: String,
    yes_filled: Decimal,
    no_filled: Decimal,
    /// Instant at which the state became unbalanced (one leg more filled).
    unbalanced_since: Option<Instant>,
    /// Terminal unwind decision already emitted (hold/dump/repost): do not
    /// repeat while that decision still holds (sell in flight, or hold decided).
    unwind_sent: bool,
    /// Cached reference price of the underlying (same pattern as near_certain and market_making).
    reference: Option<Decimal>,
    /// Last posted FV: triggers a requote if it moves (same threshold in
    /// bps as the mid). `None` if no FV was available when posting.
    posted_fv: Option<Decimal>,
    /// Implied probability of YES at the time of the last post: the requote trigger
    /// that was missing. `posted_mid` only detects PARITY moves; this one detects the
    /// DIRECTIONAL ones, which are the ones that leave our bids stale. `None` if the
    /// YES book was empty when posting.
    posted_prob: Option<Decimal>,
    /// Requote tolerance: price at which each leg's bid was left resting at the
    /// last post. `None` = we do not believe a GTD is live on that token: it is
    /// cleared by the fill that exhausts `yes_live`/`no_live`, by the terminal ack
    /// (Cancelled/Rejected) and by any cancel the strategy emits.
    posted_bid_yes: Option<Decimal>,
    posted_bid_no: Option<Decimal>,
    /// Accumulated cost of the YES leg (Σ price×size on Buy, reduced
    /// proportionally on Sell) — for the avg_entry of the maker repost.
    yes_cost: Decimal,
    no_cost: Decimal,
    /// YES+NO box completed at least once (avoids double counting the metric).
    box_counted: bool,
    /// Number of times this bracket has forced an unwind with a dump (sell of the
    /// losing leg). A dump is hard evidence of toxic flow: the losing leg filled
    /// and the winning one did not arrive. Calibration: 5 brackets with 6+
    /// re-entries accounted for half of the hour's loss, and ALL of it
    /// was in the unwind SELLs (−$21.93) while boxes held to settlement
    /// won (+$1.08). After N dumps the bracket is not reopened.
    dump_count: u32,
    /// Shares posted and still live on the book (GTD emitted, no fill nor
    /// confirmed cancel). Without this the `order_size_shares` cap capped nothing:
    /// `yes_filled` only rises when the fill comes back (async), so PM re-posted
    /// 3 shares on every tick believing it had nothing posted and piled up dozens
    /// of live GTDs — 129 shares on a bracket with a cap of 3.
    yes_live: Decimal,
    no_live: Decimal,
    /// Bracket asset (same value as `VolSnapshot::asset`) — key
    /// of the history together with `duration`.
    asset: String,
    /// Nominal duration of the bracket in seconds (`clear_at −
    /// market_open_ts` when the state is created). `0` = unknown (invalid mark,
    /// `market_open_ts` not yet populated) — never negative.
    duration: i64,
    /// The YES leg had AT LEAST one Buy fill at some point. Do not use
    /// `yes_cost > 0` as a proxy: a full dump of the inventory returns it to 0 and
    /// would erase the evidence that the leg DID get touched.
    ever_yes: bool,
    /// Analogous to `ever_yes` for the NO leg.
    ever_no: bool,
}

/// Result of an already-expired bracket, for the (asset, duration) history.
/// `boxed`/`touched` are read from `PMState` at the moment
/// of expiry; `final_prob` is the last observed value of the implied
/// probability of YES (v1: does not wait for the real on-chain settlement).
#[derive(Debug, Clone, Copy)]
struct BracketRecord {
    boxed: bool,
    touched: bool,
    final_prob: Option<Decimal>,
}

/// Unwind decision for a single unfilled leg (private to PM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnwindAction {
    /// FV favors the retained leg: cancel the opposite one and wait for settlement.
    Hold,
    /// FV disfavors it (or no FV is available): sell the excess taker.
    DumpTaker,
    /// Neutral zone with enough time: re-post a maker ask instead of taker.
    RepostMaker,
}

/// Decides the unwind action from `fv_ret` — the FV probability that
/// the RETAINED leg wins (for a retained NO it is `1 − fv_yes`, see `fv_of`).
/// `None` = no confidence (stale oracle or no reference/sigma) → dump, same
/// as the earlier behavior.
fn decide_unwind(fv_ret: Option<Decimal>, remaining: i64, cfg: &ParityMakerConfig) -> UnwindAction {
    match fv_ret {
        Some(fv) if fv >= cfg.unwind_hold_fv => UnwindAction::Hold,
        Some(fv) if fv <= cfg.unwind_dump_fv => UnwindAction::DumpTaker,
        Some(_) if remaining > cfg.unwind_repost_min_secs as i64 => UnwindAction::RepostMaker,
        Some(_) => UnwindAction::DumpTaker,
        None => UnwindAction::DumpTaker,
    }
}

/// Optional value for the entry log: empty string if there is no value.
/// The calibration parser distinguishes "no data" from "zero", which is not the same:
/// a queue of 0 shares is information; a signal that is unavailable is not.
fn opt_dec(d: Option<Decimal>) -> String {
    d.map_or(String::new(), |v| v.to_string())
}

/// Optional fraction in basis points, for the entry log.
fn opt_bps(d: Option<Decimal>) -> String {
    d.map_or(String::new(), |v| (v * dec!(10000)).round().to_string())
}

/// Seconds until the next futures funding window
/// (00/08/16 UTC = every 8h = 28_800s). Derived from the clock, no waiting. It is a
/// FEATURE for offline mining, NOT a gate: informed flow clusters around
/// the funding settlement, so the distance to the window can separate
/// boxes from orphans. The distance to the PREVIOUS window is `28800 − value` and is
/// folded in the offline analysis.
fn secs_to_funding(now_ts: i64) -> i64 {
    const FUNDING_PERIOD: i64 = 28_800;
    (FUNDING_PERIOD - now_ts.rem_euclid(FUNDING_PERIOD)) % FUNDING_PERIOD
}

/// Volatility-regime ratio σ5s/σ60s. `None` if σ60 is missing or 0 (avoids
/// dividing by zero). >1 = recent volatility above the background (agitated regime);
/// <1 = recent calm. An earlier diagnosis could not test this signal because
/// it was not logged.
fn vol_ratio(sigma_short: Option<Decimal>, sigma_long: Option<Decimal>) -> Option<Decimal> {
    match (sigma_short, sigma_long) {
        (Some(s5), Some(s60)) if s60 > Decimal::ZERO => Some(s5 / s60),
        _ => None,
    }
}

/// Normalizes the raw signed CVD (USDT) to a ratio ∈ [−1,1]
/// comparable across assets, dividing by the gross volume of the long window
/// (`vol_quote_60s` is the only total available in `VolSnapshot`; it is used as
/// the denominator for both `cvd_ratio_5s` and `cvd_ratio_60s`). `None` if
/// the CVD or the total is missing, or the total is zero (avoids dividing by zero).
fn cvd_ratio(cvd: Option<Decimal>, vol_quote_60s: Option<Decimal>) -> Option<Decimal> {
    match (cvd, vol_quote_60s) {
        (Some(c), Some(v)) if v > Decimal::ZERO => Some(c / v),
        _ => None,
    }
}

/// Implied probability of YES in this bracket (mid of the YES token).
///
/// **Careful with the requote "mid".** That is `(yes_ask + no_ask) / 2`, which is half the
/// COST OF THE BOX. YES and NO are two independent books: no identity ties them,
/// which is why their sum drifts from 1 and why the arbitrage phases that the arb strategy chases exist,
/// so that number does move, but it moves with the **parity deviation**
/// rather than with direction: if the underlying rises, `yes_ask` rises and `no_ask`
/// falls, and the sum only changes by whatever the two moves differ in (residual and
/// latency between the two books). Its sign does not say whether YES is rising.
///
/// The requote needs the other thing: the **probability**, which is the price of YES on its own.
/// One side of the book is enough: the best price is an approximation, not the signal itself.
fn implied_yes(book: &BookSnapshot) -> Option<Decimal> {
    match (book.yes.best_bid(), book.yes.best_ask()) {
        (Some(b), Some(a)) => Some((b + a) / dec!(2)),
        (Some(b), None) => Some(b),
        (None, Some(a)) => Some(a),
        (None, None) => None,
    }
}

/// Shares of bids that will fill BEFORE ours: those at a strictly better price,
/// plus those at our same price (at an equal price FIFO rules and we have just arrived:
/// all that queue goes ahead).
///
/// Bid levels arrive sorted descending, guaranteed by the feed
/// (`market_feed/handle.rs` and `resync.rs`, see the doc comment of `HalfBook`), so
/// it is enough to traverse while the price stays ≥ ours.
fn queue_ahead(bids: &HalfBook, our_bid: Decimal) -> Decimal {
    bids.levels
        .iter()
        .take_while(|l| l.price >= our_bid)
        .map(|l| l.size)
        .sum()
}

/// FV that YES wins (same pattern as near_certain and market_making: only if the oracle is not stale and there is
/// reference + sigma). `None` if any of those requirements is missing.
fn fv_yes(vol: &VolSnapshot, reference: Option<Decimal>, remaining: i64) -> Option<Decimal> {
    if vol.is_stale_for_signal() {
        return None;
    }
    let reference = reference?;
    let sigma = vol.sigma_long.or(vol.sigma_short)?;
    fair_value_yes(vol.last_price, reference, sigma, remaining)
}

/// FV for SETTLEMENT decisions (hold vs dump of a single leg).
/// Uses the oracle spot (Binance), the bot's only price source.
fn fv_yes_settlement(
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

/// Number of consecutive results, counting from the MOST RECENT
/// (`back()`), with the same direction. Direction of a record = `final_prob >
/// 0.5` ⇒ UP. The sign of the returned value encodes the streak direction:
/// positive = UP streak, negative = DOWN streak, `0` = no streak (empty deque
/// or the most recent record has no `final_prob`). A record WITHOUT
/// `final_prob` ends the streak as soon as it is reached; its direction is not assumed.
fn hist_streak(deque: &VecDeque<BracketRecord>) -> i32 {
    let mut streak = 0i32;
    let mut dir: Option<bool> = None;
    for record in deque.iter().rev() {
        let Some(prob) = record.final_prob else { break };
        let up = prob > dec!(0.5);
        match dir {
            None => {
                dir = Some(up);
                streak = 1;
            }
            Some(d) if d == up => streak += 1,
            Some(_) => break,
        }
    }
    match dir {
        Some(true) => streak,
        Some(false) => -streak,
        None => 0,
    }
}

// ── ParityMaker ───────────────────────────────────────────────────────────────

pub struct ParityMaker {
    cfg: ParityMakerConfig,
    markets: HashMap<String, PMState>,
    last_retain: Instant,
    /// Short history of the spot per asset. `VolSnapshot` only publishes the
    /// last price; the **movement** (the signal that separates toxic from benign flow)
    /// has to be measured here.
    spot_hist: HashMap<String, VecDeque<(Instant, Decimal)>>,
    /// Short history of the implied probability of YES of EACH bracket,
    /// grouped by asset. It is what allows comparing a bracket with its siblings
    /// and deciding whether the movement is common (underlying) or idiosyncratic (flow).
    bracket_probs:
        HashMap<String /*asset*/, HashMap<String /*cid*/, VecDeque<(Instant, Decimal)>>>,
    /// History of results (boxed/orphan) of brackets already in expired,
    /// grouped by (asset, duration). The same nominal duration tends to
    /// behave alike (a 5-minute updown is not comparable to a 1-hour one).
    history: HashMap<(String, i64), VecDeque<BracketRecord>>,
}

impl ParityMaker {
    pub fn new(cfg: ParityMakerConfig) -> Self {
        Self {
            cfg,
            markets: HashMap::new(),
            last_retain: Instant::now(),
            spot_hist: HashMap::new(),
            bracket_probs: HashMap::new(),
            history: HashMap::new(),
        }
    }

    /// COMMON vs IDIOSYNCRATIC MOVEMENT: the signal that separates the two flows.
    ///
    /// All updown brackets of an asset share the underlying. Therefore:
    ///
    ///   · If ALL of them move at once and in the same direction → the spot moved →
    ///     INFORMED flow. The leg opposite the movement is doomed: its price
    ///     is going to fall, and our passive bid can only fill because the price drops
    ///     down to it. They will give us the losing leg.
    ///   · If ONLY this book moves → IDIOSYNCRATIC flow: someone is selling in a hurry
    ///     in this particular market. **That is the flow a maker lives on.** Quote.
    ///
    /// The spot-drift gate tried the same by watching Binance and did not move the needle:
    /// the book moves at its own pace. This signal watches the books themselves, which is where
    /// the fill happens.
    ///
    /// Returns `(own_movement, is_common)`. `None` = signal unavailable (no
    /// history covering the window, or no sibling brackets with data) → does not block.
    fn cross_bracket_move(
        &mut self,
        asset: &str,
        cid: &str,
        prob: Decimal,
        secs: u64,
    ) -> Option<(Decimal, bool)> {
        if secs == 0 {
            return None;
        }
        let now = Instant::now();

        let by_cid = self.bracket_probs.entry(asset.to_string()).or_default();
        let hist = by_cid.entry(cid.to_string()).or_default();
        hist.push_back((now, prob));
        // Retain twice the window (same pattern as `spot_drift`).
        while hist
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t).as_secs() > secs * 2)
        {
            hist.pop_front();
        }

        // Movement of a bracket = current probability − the one from `secs` ago. In absolute
        // probability points: the prob is ALREADY a probability, not a price from which
        // a relative variation has to be derived.
        let move_of = |h: &VecDeque<(Instant, Decimal)>| -> Option<Decimal> {
            let (_, past) = h
                .iter()
                .find(|(t, _)| now.duration_since(*t).as_secs() >= secs)?;
            let (_, last) = h.back()?;
            Some(*last - *past)
        };

        let own_move = move_of(by_cid.get(cid)?)?;

        // Siblings: the other brackets of the SAME asset with history covering the
        // window. Without siblings that have data there is no signal, and a missing signal never
        // blocks (updowns usually have 2-4 brackets at once, but right after startup
        // or with a single live bracket this is intentionally inert).
        let sibling_moves: Vec<Decimal> = by_cid
            .iter()
            .filter(|(other, _)| other.as_str() != cid)
            .filter_map(|(_, h)| move_of(h))
            .collect();
        if sibling_moves.is_empty() {
            return None;
        }

        let move_thr = Decimal::from(self.cfg.xbracket_move_bps) / dec!(10000);
        let mut common = false;
        if own_move.abs() >= move_thr && move_thr > Decimal::ZERO {
            // Same sign and comparable magnitude: the signature that the underlying moved
            // and not this particular book.
            let up = own_move > Decimal::ZERO;
            let agree = sibling_moves
                .iter()
                .filter(|m| {
                    (**m > Decimal::ZERO) == up
                        && m.abs() >= self.cfg.xbracket_common_ratio * own_move.abs()
                })
                .count();
            common = agree * 2 >= sibling_moves.len();
        }
        Some((own_move, common))
    }

    /// Relative change of the spot over the last `secs` seconds.
    /// `None` while there is no history covering the window (startup).
    fn spot_drift(&mut self, vol: &VolSnapshot, secs: u64) -> Option<Decimal> {
        if secs == 0 || vol.last_price <= Decimal::ZERO || vol.is_stale_for_signal() {
            return None;
        }
        let now = Instant::now();
        let hist = self.spot_hist.entry(vol.asset.clone()).or_default();
        hist.push_back((now, vol.last_price));
        // Retain twice the window: enough to find the reference point
        // without letting the buffer grow.
        while hist
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t).as_secs() > secs * 2)
        {
            hist.pop_front();
        }
        // Oldest price that already covers the requested window.
        let (_, past) = hist
            .iter()
            .find(|(t, _)| now.duration_since(*t).as_secs() >= secs)?;
        if *past <= Decimal::ZERO {
            return None;
        }
        Some((vol.last_price - past) / past)
    }
}

impl Strategy for ParityMaker {
    fn id(&self) -> StrategyId {
        StrategyId::ParityMaker
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        if self.last_retain.elapsed() >= std::time::Duration::from_secs(30) {
            // Register the result of each bracket that expires BEFORE
            // sweeping it out of `self.markets`, and BEFORE `bracket_probs` loses its
            // entry (that happens further down, in the same block block). The order matters:
            // reversing it would leave `final_prob` always `None` (or, worse, the bracket itself
            // already gone from `self.markets` with nothing registered).
            if self.cfg.bracket_history_len > 0 {
                for (cid, state) in self.markets.iter() {
                    if state.clear_at > now {
                        continue;
                    }
                    // Bugfix: duration == 0 is the invalid mark (market_open_ts not
                    // populated when the state was created). Do not pollute bucket 0 of the
                    // history with brackets of unknown real duration.
                    if state.duration == 0 {
                        continue;
                    }
                    let final_prob = self
                        .bracket_probs
                        .get(&state.asset)
                        .and_then(|by_cid| by_cid.get(cid))
                        .and_then(|h| h.back())
                        .map(|(_, p)| *p);
                    let record = BracketRecord {
                        boxed: state.box_counted,
                        touched: state.ever_yes || state.ever_no,
                        final_prob,
                    };
                    let deque = self
                        .history
                        .entry((state.asset.clone(), state.duration))
                        .or_default();
                    deque.push_back(record);
                    while deque.len() > self.cfg.bracket_history_len {
                        deque.pop_front();
                    }
                }
            }
            self.markets.retain(|_, s| s.clear_at > now);
            // Dead cids also leave the cross-bracket history, or `bracket_probs` grows without limit
            // (a new bracket every few minutes).
            let live = &self.markets;
            for by_cid in self.bracket_probs.values_mut() {
                by_cid.retain(|cid, _| live.contains_key(cid));
            }
            self.bracket_probs.retain(|_, by_cid| !by_cid.is_empty());
            self.last_retain = Instant::now();
        }

        let mut intents = Vec::new();
        let remaining = book.secs_to_clear();
        let cid = &book.market_condition_id;
        // The tick travels in the BookSnapshot: 0.001 above 0.96 and below 0.04,
        // where truncating to 2 decimals would wreck the prices. It is not a constant.
        let tick = book.tick();

        if remaining <= 30 || remaining > 3600 {
            // On leaving the window, cancel the GTDs and liquidate an unbalanced inventory.
            // Previously it returned without touching anything and the single filled leg was held
            // to settlement (same leak as market_making).
            if let Some(st) = self.markets.get_mut(cid) {
                cancel_live_orders(st, book, &mut intents);
                if remaining > 0 && remaining <= 30 && !st.unwind_sent {
                    let imbalance = st.yes_filled - st.no_filled;
                    if imbalance != Decimal::ZERO {
                        let retained_is_yes = imbalance > Decimal::ZERO;
                        let fv_ret = fv_yes_settlement(vol, st.reference, remaining).map(|f| {
                            if retained_is_yes {
                                f
                            } else {
                                Decimal::ONE - f
                            }
                        });
                        let action = decide_unwind(fv_ret, remaining, &self.cfg);
                        match action {
                            UnwindAction::Hold => {
                                metrics::counter!("hrf_pm_unwind_total", "reason" => "hold_fv")
                                    .increment(1);
                                tracing::info!(cid = %cid, fv_ret = ?fv_ret,
                                    "PM: window close — hold to settlement, FV favors the retained leg");
                                st.unwind_sent = true;
                            }
                            UnwindAction::DumpTaker => {
                                let (token, excess, bid) = if retained_is_yes {
                                    (st.yes_token.clone(), imbalance, book.yes.best_bid())
                                } else {
                                    (st.no_token.clone(), -imbalance, book.no.best_bid())
                                };
                                let reason = if fv_ret.is_none() {
                                    "dump_no_fv"
                                } else {
                                    "dump_fv"
                                };
                                if let Some(bid) = bid.filter(|b| *b >= self.cfg.min_unwind_bid) {
                                    metrics::counter!("hrf_pm_unwind_total", "reason" => reason)
                                        .increment(1);
                                    tracing::info!(cid = %cid, token = %token, excess = %excess, bid = %bid,
                                        "PM: liquidating unbalanced leg at window close");
                                    intents.push(sell_intent(book, &token, bid, excess));
                                    st.unwind_sent = true;
                                    // Consistent with the per-bracket dump limit: a dump at window close
                                    // also counts as a toxic bracket for `max_dumps_per_bracket`, the same
                                    // as the DumpTaker of the single-leg unwind.
                                    // (see the single-leg unwind in this file).
                                    st.dump_count += 1;
                                }
                            }
                            UnwindAction::RepostMaker => {
                                let (token, excess, cost, filled, best_ask) = if retained_is_yes {
                                    (
                                        st.yes_token.clone(),
                                        imbalance,
                                        st.yes_cost,
                                        st.yes_filled,
                                        book.yes.best_ask(),
                                    )
                                } else {
                                    (
                                        st.no_token.clone(),
                                        -imbalance,
                                        st.no_cost,
                                        st.no_filled,
                                        book.no.best_ask(),
                                    )
                                };
                                if filled > Decimal::ZERO {
                                    let avg_entry = cost / filled;
                                    let mut price = avg_entry + self.cfg.unwind_maker_edge;
                                    if let Some(ba) = best_ask {
                                        price = price.max(ba);
                                    }
                                    price = price.min(Decimal::ONE - tick); // Ceiling = 1 − tick of the market
                                    metrics::counter!("hrf_pm_unwind_total", "reason" => "repost_maker").increment(1);
                                    tracing::info!(cid = %cid, token = %token, price = %price, excess = %excess,
                                        "PM: window close — re-posting maker for the excess (neutral zone, enough time)");
                                    intents.push(ask_maker_intent(book, &token, price, excess));
                                    st.unwind_sent = true;
                                }
                            }
                        }
                    }
                }
            }
            return intents;
        }

        // The spot movement is measured before taking the bracket's state
        // (`spot_hist` lives in the strategy, not in the bracket: the underlying is the
        // same for every bracket of the asset).
        let drift = self.spot_drift(vol, self.cfg.oracle_drift_secs);
        let drift_thr = Decimal::from(self.cfg.oracle_drift_bps) / dec!(10000);
        // If the spot rises, YES (up) gains probability and NO sinks: the NO bid is doomed
        // to be swept by informed flow. And symmetrically the other way round.
        let (mut drift_condemns_yes, mut drift_condemns_no) = match drift {
            Some(d) if drift_thr > Decimal::ZERO && d >= drift_thr => {
                metrics::counter!("hrf_pm_no_quote_total", "side" => "no", "reason" => "drift")
                    .increment(1);
                (false, true)
            }
            Some(d) if drift_thr > Decimal::ZERO && d <= -drift_thr => {
                metrics::counter!("hrf_pm_no_quote_total", "side" => "yes", "reason" => "drift")
                    .increment(1);
                (true, false)
            }
            _ => (false, false),
        };

        // The cross-bracket signal is also computed BEFORE taking the state
        // (`bracket_probs` lives in the strategy, not in the bracket: it precisely
        // compares this bracket with its siblings).
        // It is also reused as the directional requote trigger.
        let prob_yes = implied_yes(book);
        let xbracket = prob_yes.and_then(|p| {
            self.cross_bracket_move(&vol.asset, cid, p, self.cfg.xbracket_window_secs)
        });

        // COMMON movement across the asset's brackets = the underlying moved = informed
        // flow. The leg opposite the movement is doomed: they are going to sweep it from us.
        // The drift variables are reused with OR; the whole machinery of
        // skip + immediate cancellation of the doomed leg already exists and is tested;
        // this signal only adds a source. It has its own metric reason so the two can be
        // separated in calibration.
        if let Some((own_move, true)) = xbracket {
            if own_move > Decimal::ZERO {
                metrics::counter!("hrf_pm_no_quote_total", "side" => "no", "reason" => "xbracket")
                    .increment(1);
                drift_condemns_no = true;
            } else if own_move < Decimal::ZERO {
                metrics::counter!("hrf_pm_no_quote_total", "side" => "yes", "reason" => "xbracket")
                    .increment(1);
                drift_condemns_yes = true;
            }
        }

        // History of results of sibling brackets (same asset +
        // nominal duration): read BEFORE `self.markets.entry()`, for the same borrow
        // reason as the signals above: `entry()` borrows `self.markets` exclusively,
        // and `self.history` cannot be read through a method that borrows all of `self`.
        // Scalar locals: `None`/zeros if the signal is disabled or has no history,
        // so a missing signal
        // never blocks.
        // Bugfix: with `market_open_ts == 0` (field not yet populated, see the tracing log below)
        // the subtraction gave `clear_at` in full (~1.7e9) instead of the invalid mark `0`,
        // documented in the doc comment of `duration`. Each bracket fell into its own history
        // bucket, so `self.history` never accumulated and gained a new key per bracket with no
        // pruning (memory leak).
        // Now aligned with the formula the tracing log already uses.
        let bracket_duration = if book.market_open_ts > 0 && book.clear_at > book.market_open_ts {
            book.clear_at - book.market_open_ts
        } else {
            0
        };
        let hist_stats: Option<(usize, usize, usize, i32)> =
            if self.cfg.bracket_history_len == 0 || bracket_duration == 0 {
                // Invalid duration (0): do not query bucket 0, it would mix brackets
                // of different real durations that ended up there because of the same bug.
                None
            } else {
                self.history
                    .get(&(vol.asset.clone(), bracket_duration))
                    .map(|deque| {
                        let n = deque.len();
                        let boxes = deque.iter().filter(|r| r.touched && r.boxed).count();
                        let orphans = deque.iter().filter(|r| r.touched && !r.boxed).count();
                        (n, boxes, orphans, hist_streak(deque))
                    })
            };

        let state = self.markets.entry(cid.clone()).or_insert_with(|| PMState {
            clear_at: book.clear_at,
            posted_at: None,
            posted_mid: None,
            yes_token: book.yes.token_id.clone(),
            no_token: book.no.token_id.clone(),
            yes_filled: Decimal::ZERO,
            no_filled: Decimal::ZERO,
            unbalanced_since: None,
            unwind_sent: false,
            reference: None,
            posted_fv: None,
            posted_prob: None,
            posted_bid_yes: None,
            posted_bid_no: None,
            yes_cost: Decimal::ZERO,
            no_cost: Decimal::ZERO,
            box_counted: false,
            dump_count: 0,
            yes_live: Decimal::ZERO,
            no_live: Decimal::ZERO,
            asset: vol.asset.clone(),
            duration: bracket_duration,
            ever_yes: false,
            ever_no: false,
        });

        // Cache the underlying reference (same pattern as near_certain and market_making: oracle fallback only near the open,
        // see `fallback_reference`).
        if state.reference.is_none() {
            state.reference = book
                .reference_price
                .or_else(|| super::fallback_reference(vol, book, now, cid, "PM"));
        }

        // FV(YES) of the model: None if the oracle is stale or the reference/sigma is missing.
        // Reused for the single-leg unwind and the skew.
        let fv = fv_yes(vol, state.reference, remaining);

        // Single-leg management. Parity only pays when BOTH legs fill; adverse selection
        // fills the losing leg first. While the grace window runs the requote tries to complete
        // the pair; once it expires, we choose between hold to settlement, dump taker or
        // repost maker according to the FV of the retained leg
        // (it used to always sell taker, giving 10% of the upside to the fee).
        let imbalance = state.yes_filled - state.no_filled;
        if imbalance != Decimal::ZERO {
            let since = *state.unbalanced_since.get_or_insert_with(Instant::now);
            if !state.unwind_sent
                && self.cfg.single_leg_unwind_secs > 0
                && since.elapsed().as_secs() >= self.cfg.single_leg_unwind_secs
            {
                let retained_is_yes = imbalance > Decimal::ZERO;
                let fv_ret = fv_yes_settlement(vol, state.reference, remaining).map(|f| {
                    if retained_is_yes {
                        f
                    } else {
                        Decimal::ONE - f
                    }
                });
                let action = decide_unwind(fv_ret, remaining, &self.cfg);

                // Cancels are conditional on something being live to cancel (same guard as
                // `cancel_live_orders`). They used to run unconditionally on EVERY tick while
                // `imbalance != 0 && !unwind_sent`. In the DumpTaker branch with no valid bid
                // (toxic zone, <min_unwind_bid) `unwind_sent` was never set, so this
                // `block_bytes` block re-entered tick after tick
                // emitting 2 `Cancel`s with nothing live to cancel, indefinitely.
                cancel_live_orders(state, book, &mut intents);
                // The live counters are not cleared here: the cancel ack releases them.

                match action {
                    UnwindAction::Hold => {
                        metrics::counter!("hrf_pm_unwind_total", "reason" => "hold_fv")
                            .increment(1);
                        tracing::info!(cid = %cid, fv_ret = ?fv_ret,
                            "PM: hold to settlement — FV favors the retained leg (selling taker gives away the upside)");
                        state.unwind_sent = true;
                    }
                    UnwindAction::DumpTaker => {
                        let (token, excess, bid) = if retained_is_yes {
                            (state.yes_token.clone(), imbalance, book.yes.best_bid())
                        } else {
                            (state.no_token.clone(), -imbalance, book.no.best_bid())
                        };
                        let reason = if fv_ret.is_none() {
                            "dump_no_fv"
                        } else {
                            "dump_fv"
                        };
                        if let Some(bid) = bid.filter(|b| *b >= self.cfg.min_unwind_bid) {
                            metrics::counter!("hrf_pm_unwind_total", "reason" => reason)
                                .increment(1);
                            tracing::info!(cid = %cid, token = %token, excess = %excess, bid = %bid,
                                secs = self.cfg.single_leg_unwind_secs,
                                "PM: unfilled single leg — cancelling and selling the excess (taker)");
                            intents.push(sell_intent(book, &token, bid, excess));
                            state.unwind_sent = true;
                            state.dump_count += 1; // Toxic bracket: count the dump
                        }
                        // No valid bid: do not mark unwind_sent; retry on the next tick.
                    }
                    UnwindAction::RepostMaker => {
                        let (token, excess, cost, filled, best_ask) = if retained_is_yes {
                            (
                                state.yes_token.clone(),
                                imbalance,
                                state.yes_cost,
                                state.yes_filled,
                                book.yes.best_ask(),
                            )
                        } else {
                            (
                                state.no_token.clone(),
                                -imbalance,
                                state.no_cost,
                                state.no_filled,
                                book.no.best_ask(),
                            )
                        };
                        if filled > Decimal::ZERO {
                            let avg_entry = cost / filled;
                            let mut price = avg_entry + self.cfg.unwind_maker_edge;
                            if let Some(ba) = best_ask {
                                price = price.max(ba);
                            }
                            price = price.min(Decimal::ONE - tick); // Ceiling = 1 − tick of the market
                            metrics::counter!("hrf_pm_unwind_total", "reason" => "repost_maker")
                                .increment(1);
                            tracing::info!(cid = %cid, token = %token, price = %price, excess = %excess,
                                "PM: single leg — neutral FV zone with time to spare, re-posting maker instead of dump");
                            intents.push(ask_maker_intent(book, &token, price, excess));
                            state.unwind_sent = true;
                        }
                    }
                }
                return intents;
            }
        } else {
            state.unbalanced_since = None;
            state.unwind_sent = false;
        }

        // No inventory means this bracket is about to OPEN. All opening gates
        // (the barrier, spread and queue gates) are conditioned on this: completing a box already started
        // is sacred. The missing leg is what turns an orphan (−11¢) into a box (+margin), and no
        // filter may prevent it.
        let no_inventory = state.yes_filled == Decimal::ZERO && state.no_filled == Decimal::ZERO;

        // BURNED BRACKET. If this bracket already forced us to dump the losing leg, it is toxic
        // and we do not reopen it: adverse selection would fill the losing leg again. Measured:
        // boxes held to settlement win (+$1.08); all the loss is in re-entering and dumping again
        // (SELL −$21.93), concentrated in 5 brackets with 6+ entries. It only blocks OPENING
        // (no_inventory): completing a box already started remains sacred. Updowns live about
        // 5 min, so burning the bracket after the first dump barely costs any legitimate
        // opportunity.
        if self.cfg.max_dumps_per_bracket > 0
            && no_inventory
            && state.dump_count >= self.cfg.max_dumps_per_bracket
        {
            metrics::counter!("hrf_pm_skip_burned_bracket_total").increment(1);
            cancel_live_orders(state, book, &mut intents);
            return intents;
        }

        // Remaining movement of the underlying (σ·√t). It is used by the entry barrier
        // (time barrier), the chase and the feature log.
        let exp_move = vol
            .sigma_long
            .or(vol.sigma_short)
            .and_then(|s| crate::fair_value::expected_move(s, remaining));

        // CHASE OF THE MISSING LEG.
        //
        // With one leg held, every second without completing brings the orphan closer
        // (−11c measured, against the +4c the box pays). The residual budget is a CEILING:
        // paying up to `target_sum − already_paid` leaves the box exactly at `1 − margin`,
        // i.e. the configured margin, not a cent less. Quoting below that ceiling only earns
        // EXTRA margin *if* the fill arrives; when the probability of completing falls, that
        // extra is a bet that loses against 11 cents. So we give it up and bid up to the ceiling.
        //
        // It triggers on any of the three:
        // (a), (b) or (c) below:
        let chase = self.cfg.chase_after_secs > 0 && imbalance != Decimal::ZERO && {
            let retained_is_yes = imbalance > Decimal::ZERO;
            // (a) we have been unbalanced for too long.
            let overdue = state
                .unbalanced_since
                .is_some_and(|t| t.elapsed().as_secs() >= self.cfg.chase_after_secs);
            // (b) the bracket is already decided: no more flow is going to arrive.
            let barrier_closed = self.cfg.min_expected_move_bps > 0
                && exp_move.is_none_or(|m| {
                    m < Decimal::from(self.cfg.min_expected_move_bps) / dec!(10000)
                });
            // (c) the missing leg is getting more expensive. YES/UP and NO/DOWN add up to 1 WHEN
            // CLAIMED, since the payout becomes 1 for the winner and 0 for the
            // loser (their quoted prices do NOT add up to 1: they are independent books). So
            // an underlying move that condemns one leg makes the other more expensive: the
            // missing one rises exactly when the RETAINED leg is condemned. If we do not
            // pay now, it moves away from our bid and the box never closes.
            let condemned_retention =
                (retained_is_yes && drift_condemns_yes) || (!retained_is_yes && drift_condemns_no);
            overdue || barrier_closed || condemned_retention
        };

        // Adverse selection filter: pause and cancel live GTDs if there is
        // a volatility spike (deliberately duplicated inline from market_making: they are guards
        // independent by design; a shared helper is not worth it for 3 lines).
        if self.cfg.vol_ratio_threshold > Decimal::ZERO {
            let vol_spike = vol
                .sigma_short
                .zip(vol.sigma_long)
                .is_some_and(|(s5, s60)| {
                    s60 > Decimal::ZERO && s5 / s60 > self.cfg.vol_ratio_threshold
                });
            if vol_spike {
                metrics::counter!("hrf_pm_paused_total", "reason" => "vol_spike").increment(1);
                cancel_live_orders(state, book, &mut intents);
                return intents;
            }
        }

        // Compute the best bids of the book to center our quotes.
        // If the ask falls below min_leg_price (the toxic zone the clock
        // sinks) PM stops quoting, but it must NOT leave the GTDs hanging.
        let yes_ask = match book.yes.best_ask() {
            Some(a) if a >= self.cfg.min_leg_price => a,
            _ => {
                cancel_live_orders(state, book, &mut intents);
                return intents;
            }
        };
        let no_ask = match book.no.best_ask() {
            Some(a) if a >= self.cfg.min_leg_price => a,
            _ => {
                cancel_live_orders(state, book, &mut intents);
                return intents;
            }
        };

        // SPREAD GATE: the cheapest filter of all.
        //
        // The spread is the maker's premium. With a 1¢ spread there is no room for
        // an impatient seller to cross to our bid: the only flow that reaches us is informed
        // flow, exactly the kind we do not want. Without spread
        // there is no business, only adverse selection.
        //
        // A missing bid = empty book on that side = "infinite" spread: it does not block
        // (general rule: a missing signal must never leave the bot unable to trade).
        let spread_yes = book.yes.best_bid().map(|b| yes_ask - b);
        let spread_no = book.no.best_bid().map(|b| no_ask - b);
        if no_inventory && self.cfg.min_spread > Decimal::ZERO {
            let narrow = |s: Option<Decimal>| s.is_some_and(|s| s < self.cfg.min_spread);
            if narrow(spread_yes) || narrow(spread_no) {
                metrics::counter!("hrf_pm_skip_spread_total").increment(1);
                cancel_live_orders(state, book, &mut intents);
                return intents;
            }
        }

        // Box budget (1 - margin) split between the two legs in
        // proportion to their ask: the parity constraint holds by construction.
        //
        // Previously the bids were `ask - margin/2` each, so their sum was
        // `ask_yes + ask_no - margin` and the check `sum_value <= 1 - margin` was
        // reduced to `ask_yes + ask_no <= 1`, regardless of margin: PM only
        // quoted when the book was already a taker arb. 3 quotes in 30 min.
        let half_margin = self.cfg.margin / dec!(2);
        let ask_sum = yes_ask + no_ask;
        if ask_sum <= Decimal::ZERO {
            return intents;
        }
        let target_sum = (dec!(1) - self.cfg.margin).max(Decimal::ZERO);
        // Truncated (not rounded) to the market tick: the sum never exceeds the target.
        let mut bid_yes = crate::types::truncate_to_tick(yes_ask / ask_sum * target_sum, tick);
        let mut bid_no = crate::types::truncate_to_tick(no_ask / ask_sum * target_sum, tick);

        // If the book gives away an arb (ask_sum < 1 − margin) the split produces
        // bids ABOVE the ask: the GTD would cross and execute as taker paying the fee
        // (~10¢ per pair at p≈0.5) while believing it is a maker. Clamp to top-of-book; parity
        // is preserved because the clamp can only lower the bids.
        bid_yes = bid_yes.min(yes_ask - tick);
        bid_no = bid_no.min(no_ask - tick);

        // RESIDUAL budget. The parity constraint of the split only links
        // two bids of the same requote, but the legs fill at different instants:
        // YES fills at 0.50, the book moves, and the next requote splits from zero and
        // quotes NO at 0.52 → the box ends up costing 1.02 and loses 2 cents for sure at
        // settlement. Measured: 38 of 49 boxes above 1.00, average cost 1.0407.
        // With one leg already held, the other cannot pay more than what is left of the
        // budget. If it does not fit, we do not quote: the single leg is resolved by the unwind,
        // which is a manageable loss — an expensive box is a guaranteed loss.
        //
        // When it is time to chase the box (`chase`), the bid of the missing leg
        // is set AT the ceiling instead of being capped by it: we give up the extra margin
        // of the split to maximize the probability of completing. It never exceeds it —
        // the box still closes at exactly `1 − margin`.
        if state.yes_filled > Decimal::ZERO {
            let avg_yes = state.yes_cost / state.yes_filled;
            let ceiling = crate::types::truncate_to_tick(target_sum - avg_yes, tick);
            bid_no = if chase && imbalance > Decimal::ZERO {
                ceiling
            } else {
                bid_no.min(ceiling)
            };
            bid_no = bid_no.min(no_ask - tick); // The chase cannot cross the book either
        }
        if state.no_filled > Decimal::ZERO {
            let avg_no = state.no_cost / state.no_filled;
            let ceiling = crate::types::truncate_to_tick(target_sum - avg_no, tick);
            bid_yes = if chase && imbalance < Decimal::ZERO {
                ceiling
            } else {
                bid_yes.min(ceiling)
            };
            bid_yes = bid_yes.min(yes_ask - tick);
        }
        if chase {
            metrics::counter!("hrf_pm_completion_chase_total").increment(1);
        }

        if bid_yes < self.cfg.min_leg_price || bid_no < self.cfg.min_leg_price {
            metrics::counter!("hrf_pm_skip_budget_total").increment(1);
            cancel_live_orders(state, book, &mut intents);
            return intents;
        }

        // QUEUE AND DEPTH: "will they fill us, and in a balanced way?"
        //
        // The data was already in the book and the strategy was discarding it (the simulator
        // does model `queue_ahead`). It says two different things and both matter:
        //
        //   · Absolute queue: a bid behind a lot of liquidity never fills. That is
        //     what sank an earlier version (320k shares ahead at 0.999).
        //   · Asymmetric queue: if one leg has much more queue than the other, the other
        //     will fill by itself and this one will not. That is not a box, it is an orphan with
        //     a head start — and the orphan costs −11¢ against the +4¢ of a box.
        let q_yes = queue_ahead(&book.yes.bids, bid_yes);
        let q_no = queue_ahead(&book.no.bids, bid_no);
        if no_inventory {
            let q_max = q_yes.max(q_no);
            let q_min = q_yes.min(q_no);
            let absolute = self.cfg.max_queue_ahead_shares > Decimal::ZERO
                && q_max > self.cfg.max_queue_ahead_shares;
            // Ratio evaluated as multiplication, never division: q_min can be 0
            // (empty queue = we are alone at the level). The .max(1) prevents a queue
            // of 0 on one leg from triggering the asymmetry against any queue on the other.
            let asymmetry = self.cfg.queue_imbalance_ratio > Decimal::ZERO
                && q_max > self.cfg.queue_imbalance_ratio * q_min.max(Decimal::ONE);
            if absolute || asymmetry {
                let reason = if absolute { "absolute" } else { "asymmetry" };
                metrics::counter!("hrf_pm_skip_queue_total", "reason" => reason).increment(1);
                cancel_live_orders(state, book, &mut intents);
                return intents;
            }
        }

        // ENTRY barrier: is there still room in this bracket?
        //
        // The losing leg does not sink because the spot moves; it sinks because the
        // CLOCK runs: as expiry approaches, the FV converges to 0/1 and the token that is losing falls on its own.
        // Measured buy markout by remaining time: -820 bps with >240s, -4622 bps with <60s;
        // box ratio 19/23 when entering early, 2/5 when late.
        //
        // The barrier is NOT a threshold in seconds (that would only be valid for the σ of the day
        // it was measured on): it is the **expected move** left for the underlying, σ·√t.
        // When that drops below the threshold, the bracket is decided — the price no longer moves,
        // no flow completes the box, and only the loser falling is left.
        // With high σ the bracket stays alive near the close and PM can enter later;
        // with low σ it dies sooner. It adapts by itself.
        //
        // Outside the barrier we do not OPEN, but we still quote to COMPLETE a box
        // already started: the missing leg is what turns an orphan (a loss)
        // into a box (a gain).
        if self.cfg.min_expected_move_bps > 0 && no_inventory {
            let min_move_threshold = Decimal::from(self.cfg.min_expected_move_bps) / dec!(10000);
            // Without σ there is no movement estimate: we do not open blindly.
            if exp_move.is_none_or(|m| m < min_move_threshold) {
                metrics::counter!("hrf_pm_skip_without_path_total").increment(1);
                cancel_live_orders(state, book, &mut intents);
                return intents;
            }
        }

        // DURATION GATE: "does this bracket still have life left for the
        // box to complete before the clock condemns it?"
        //
        // 5-minute brackets (300s) are left orphaned much more often than the 15-minute
        // ones (900s): mining (duration AUC 0.774)
        // — median of 900s in complete boxes vs 300s in orphans. The nominal duration
        // of the bracket (`clear_at - market_open_ts`, computed above in
        // `bracket_duration` and set in `state.duration` when the state is created) is a
        // cheap ex-ante signal: it depends on neither price nor σ, only on the bracket
        // itself.
        //
        // Only blocks OPENING (no_inventory), like the barrier: completing a box already
        // started goes through it — the missing leg turns an orphan into a box.
        // duration == 0 is the "unknown" mark (market_open_ts not populated
        // yet): a missing signal never blocks.
        //
        // Target metric: BUY markout from -1174 to
        // ≥ -400 bps and box ratio ≥ 60%. Revert if the markout does not improve by
        // ≥ 300 bps with n ≥ 80 boxes.
        if self.cfg.pm_min_duration_secs > 0
            && no_inventory
            && state.duration > 0
            && state.duration < self.cfg.pm_min_duration_secs as i64
        {
            metrics::counter!("hrf_pm_skip_duration_total").increment(1);
            cancel_live_orders(state, book, &mut intents);
            return intents;
        }

        // Edge of the model relative to the parity neutral point (0.5).
        let edge = fv.map(|f| f - dec!(0.5));

        // No-quote: with an extreme FV do not risk the disfavored leg — PM quotes only
        // the favored side and skips the sum constraint (a single leg has no parity
        // to protect).
        let mut skip_yes_leg = false;
        let mut skip_no_leg = false;
        if self.cfg.fv_no_quote_threshold > Decimal::ZERO {
            if let Some(e) = edge {
                if e.abs() >= self.cfg.fv_no_quote_threshold {
                    if e > Decimal::ZERO {
                        skip_no_leg = true;
                    } else {
                        skip_yes_leg = true;
                    }
                    let side = if skip_yes_leg { "yes" } else { "no" };
                    metrics::counter!("hrf_pm_no_quote_total", "side" => side, "reason" => "fv_level").increment(1);
                }
            }
        }

        // ORACLE DIRECTIONAL GATE: the underlying fix for adverse selection.
        //
        // The `fv_no_quote` above looks at the LEVEL of the fair value (|FV-0.5| large),
        // and the level does not distinguish a toxic fill from a benign one: a bracket can
        // spend hours at FV=0.75 with a still spot, and there a fill on NO at 0.25 is
        // perfectly healthy (it is priced in). What poisons is the MOVEMENT.
        //
        // YES/UP and NO/DOWN add up to 1 WHEN CLAIMED, since the payout becomes 1
        // for the winner and 0 for the loser (their quoted prices do NOT add up to 1: they are
        // independent books, hence the arbitrage phases). So, when the spot
        // runs up, NO sinks, and a passive bid
        // on NO CAN ONLY fill because the price falls to it. In other words, the
        // underlying's movement guarantees that the losing leg gets filled on us
        // before the book reflects it. Measured: buy markout
        // -1293 bps at 5s. The book lags behind; the oracle leads.
        //
        //   flat spot + falling book → impatient seller → QUOTE (this is what a maker lives on)
        //   running spot + book lagging → informed flow → WITHDRAW the doomed leg
        if drift_condemns_yes {
            skip_yes_leg = true;
        }
        if drift_condemns_no {
            skip_no_leg = true;
        }

        // With a live maker ask on the retained leg, quoting
        // a bid on that same token too (a) has no economic direction (we would be both
        // sides of the same token) and (b) the executor's anti-duplicate, keyed by
        // (cid, token) without side, would cancel the ask. Only the missing leg is quoted.
        if state.unwind_sent && imbalance > Decimal::ZERO {
            skip_yes_leg = true;
        }
        if state.unwind_sent && imbalance < Decimal::ZERO {
            skip_no_leg = true;
        }

        // Not re-posting the doomed leg is not enough: if it has a LIVE bid they will sweep it
        // right now. Cancel without waiting for the requote.
        //
        // `live` is not cleared here (the ack releases it). While the ack has not come back,
        // this cancel is re-emitted on every tick: it is idempotent (a cancel with no GTD
        // registered is ignored by the executor), and the window lasts as long as the round-trip.
        // Re-posting is blocked too, because the cap subtracts `live`.
        if drift_condemns_yes && state.yes_live > Decimal::ZERO {
            intents.push(cancel_intent(book, &book.yes.token_id));
        }
        if drift_condemns_no && state.no_live > Decimal::ZERO {
            intents.push(cancel_intent(book, &book.no.token_id));
        }

        let single_leg = skip_yes_leg || skip_no_leg;

        // Skew by FV (parity-neutral): whatever is added to the favored bid is subtracted
        // from the disfavored one — the sum does not change, the parity constraint
        // holds. It only has direction when quoting both legs.
        if !single_leg && self.cfg.fv_skew_threshold > Decimal::ZERO {
            if let Some(e) = edge {
                if e.abs() >= self.cfg.fv_skew_threshold {
                    let s =
                        (Decimal::from(self.cfg.fv_skew_bps) / dec!(10000)).min(half_margin - tick);
                    if e > Decimal::ZERO {
                        bid_yes = (bid_yes + s).min(yes_ask - tick);
                        bid_no -= s;
                        metrics::counter!("hrf_pm_skew_total", "side" => "yes").increment(1);
                    } else {
                        bid_no = (bid_no + s).min(no_ask - tick);
                        bid_yes -= s;
                        metrics::counter!("hrf_pm_skew_total", "side" => "no").increment(1);
                    }
                }
            }
        }

        // The min_leg check ran before the skew, and the skew could leave
        // the disfavored leg's bid below the minimum.
        if (!skip_yes_leg && bid_yes < self.cfg.min_leg_price)
            || (!skip_no_leg && bid_no < self.cfg.min_leg_price)
        {
            metrics::counter!("hrf_pm_skip_min_leg_total").increment(1);
            cancel_live_orders(state, book, &mut intents);
            return intents;
        }

        // Parity constraint: sum cannot exceed 1 - margin (with a single
        // leg there is no sum to protect — no-quote already avoided the risk).
        if !single_leg {
            let max_sum = dec!(1) - self.cfg.margin;
            if bid_yes + bid_no > max_sum {
                return intents;
            }
        }

        // Does it need a requote?
        //
        // `mid` is `(yes_ask + no_ask) / 2`, i.e. half the BOX COST. YES and
        // NO are not algorithmically tied: they are two independent books, which is why their
        // sum drifts from 1 and why the arbitrage phases the arb strategy chases exist, so
        // that mid does move, but it moves with the PARITY, not with the DIRECTION:
        // when the underlying rises, `yes_ask` rises and `no_ask` falls, and the sum only
        // changes by whatever the two moves differ in. As a result PM could NOT requote
        // during a clean directional move (the mid barely budges) and left the
        // two bids hanging exactly when adverse selection collects.
        // The only brakes were the FV shift, which requires a fresh oracle, σ and
        // a reference, and the 10 s TTL: up to 10 s of stale bids without an oracle.
        //
        // The fix is to add the missing trigger: the implied PROBABILITY
        // (`implied_yes`, the price of YES on its own), with the same threshold in bps. The box
        // mid stays: it is still the right trigger for
        // parity changes, which also affect us.
        let mid = (yes_ask + no_ask) / dec!(2);
        let threshold = Decimal::from(self.cfg.requote_threshold_bps) / dec!(10000);
        let needs_requote = state
            .posted_at
            .is_none_or(|at| at.elapsed() > Duration::from_secs(self.cfg.pair_ttl_secs))
            || state
                .posted_mid
                .is_none_or(|prev| (mid - prev).abs() > threshold)
            || match (state.posted_prob, prob_yes) {
                // DIRECTIONAL shift: same threshold in bps as the mid.
                (Some(prev), Some(p)) => (p - prev).abs() > threshold,
                // Without a probability (empty YES book) there is no signal: do not force a requote
                // for the absence of data (project-wide rule).
                _ => false,
            }
            || match (state.posted_fv, fv) {
                // FV shift: same threshold in bps as the mid.
                (Some(prev), Some(f)) => (f - prev).abs() > threshold,
                (None, None) => false,
                _ => true, // FV availability changed (before/now differ)
            };

        if !needs_requote {
            return intents;
        }

        let size = self.cfg.order_size_shares;
        // size=0 → no cap; infer from the max USDC or the ask depth (if USDC=0).
        // size>0 → respect what is left up to the configured limit.
        let usdc_cap = self.cfg.limits.max_order_usdc;
        let fallback_size = |bid: Decimal| -> Decimal {
            if usdc_cap > Decimal::ZERO && bid > Decimal::ZERO {
                usdc_cap / bid
            } else {
                Decimal::ZERO
            }
        };
        // The cap subtracts what is already filled AND what is still live on the book.
        let yes_remaining = if size > Decimal::ZERO {
            (size - state.yes_filled - state.yes_live).max(Decimal::ZERO)
        } else {
            fallback_size(bid_yes)
        };
        let no_remaining = if size > Decimal::ZERO {
            (size - state.no_filled - state.no_live).max(Decimal::ZERO)
        } else {
            fallback_size(bid_no)
        };

        // Requote tolerance (order refresh tolerance): if the bid already resting
        // on a leg is at most `requote_tolerance_ticks` ticks away from the
        // desired price, that leg is NOT cancelled and re-posted — the resting order is already where we
        // want it, and skipping the cancel/ack/post cycle reduces unintended fills
        // and rate-limit pressure. It is also what breaks the churn of
        // the earlier re-posting: with the cap consumed by the live order (`remaining == 0`) the
        // resting order is still wanted: keeping it does not violate the cap, it IS the cap.
        // A leg that is NOT wanted (skip_*: drift, fv_no_quote, retained leg in
        // unwind) is never kept: those paths are dangerous.
        let tol = Decimal::from(self.cfg.requote_tolerance_ticks) * tick;
        let tol_on = self.cfg.requote_tolerance_ticks > 0;
        let mut keep_yes = tol_on
            && !skip_yes_leg
            && state.yes_live > Decimal::ZERO
            && state
                .posted_bid_yes
                .is_some_and(|r| (bid_yes - r).abs() <= tol);
        let mut keep_no = tol_on
            && !skip_no_leg
            && state.no_live > Decimal::ZERO
            && state
                .posted_bid_no
                .is_some_and(|r| (bid_no - r).abs() <= tol);

        // Tolerance NEVER keeps an invalid pair: the sum of what stays
        // resting (kept legs at their stale price + re-posted ones at the new price)
        // must still be ≤ 1 − margin. If the combination violates it, the kept ones
        // fall back to re-posting — the new bids have already passed the parity check.
        // With a single leg there is no sum to protect (the `single_leg` criterion).
        let bid_final = |keep: bool,
                         resting: Option<Decimal>,
                         skip: bool,
                         remaining: Decimal,
                         bid: Decimal|
         -> Option<Decimal> {
            if keep {
                resting
            } else if !skip && remaining > Decimal::ZERO {
                Some(bid)
            } else {
                None
            }
        };
        if let (Some(y), Some(n)) = (
            bid_final(
                keep_yes,
                state.posted_bid_yes,
                skip_yes_leg,
                yes_remaining,
                bid_yes,
            ),
            bid_final(
                keep_no,
                state.posted_bid_no,
                skip_no_leg,
                no_remaining,
                bid_no,
            ),
        ) {
            if y + n > dec!(1) - self.cfg.margin {
                keep_yes = false;
                keep_no = false;
            }
        }
        if keep_yes {
            metrics::counter!("hrf_requote_skipped_total", "strategy" => "parity_maker")
                .increment(1);
        }
        if keep_no {
            metrics::counter!("hrf_requote_skipped_total", "strategy" => "parity_maker")
                .increment(1);
        }

        // Cancelling is IMMEDIATE: adverse selection is paid during the
        // seconds that a stale bid stays hanging (markout -1293 bps at 5 s).
        // Only the legs NOT kept by tolerance are cancelled.
        if state.posted_at.is_some() {
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

        let mut posted_any = false;
        if !skip_yes_leg && !keep_yes && yes_remaining > Decimal::ZERO {
            intents.push(bid_intent(book, &book.yes.token_id, bid_yes, yes_remaining));
            state.yes_live += yes_remaining;
            state.posted_bid_yes = Some(bid_yes);
            posted_any = true;
        }
        if !skip_no_leg && !keep_no && no_remaining > Decimal::ZERO {
            intents.push(bid_intent(book, &book.no.token_id, bid_no, no_remaining));
            state.no_live += no_remaining;
            state.posted_bid_no = Some(bid_no);
            posted_any = true;
        }

        // If no bid went out (the cap is consumed because we are still
        // waiting for the cancel ack) and no resting order was kept by tolerance either,
        // do NOT mark the state as "posted": PM
        // would believe it has live orders it never sent, and on the next tick
        // would emit cancels against nothing. With no bid and no resting order there is no quote.
        if !posted_any && !keep_yes && !keep_no {
            state.posted_at = None;
            return intents;
        }

        // The anchors are re-set even when the pair was kept entirely by
        // tolerance (posted_any == false): otherwise the trigger would stay armed
        // on every tick and `hrf_requote_skipped_total` would fire in a loop.
        state.posted_at = Some(Instant::now());
        state.posted_mid = Some(mid);
        state.posted_fv = fv;
        state.posted_prob = prob_yes;

        if posted_any {
            metrics::counter!("hrf_pm_quotes_total").increment(1);
            tracing::debug!(
                cid = %cid, bid_yes = %bid_yes, bid_no = %bid_no,
                sum = %(bid_yes + bid_no), "PM: requote"
            );
        } else {
            tracing::debug!(cid = %cid, "PM: requote deferred by tolerance (pair kept)");
        }

        // Record the conditions of EACH ENTRY (one entry row per new pair;
        // completion re-quotes are not entries). Without this the calibration
        // cannot estimate P(complete | signals) and the spread/queue/cross-bracket thresholds stay
        // guesswork forever.
        //
        // It goes to the log, not to SQLite, on purpose: strategies are pure tasks
        // book→intents and have no handle on the database (only the fill_sink writes). The
        // bot RECORDS; the offline calibration decides. Learning online would be a
        // perpetual cold start — the history is wiped when migrating machines.
        // `posted_any`: a requote kept entirely by tolerance is not an
        // entry — no new bid went out.
        if posted_any && no_inventory {
            tracing::info!(
                target: "pm_entry",
                ts_ms        = crate::time::now().timestamp_millis(),
                cid          = %cid,
                asset        = %vol.asset,
                exp_move_bps = %opt_bps(exp_move),
                spread_yes   = %opt_dec(spread_yes),
                spread_no    = %opt_dec(spread_no),
                queue_yes    = %q_yes,
                queue_no     = %q_no,
                own_move_bps = %opt_bps(xbracket.map(|(m, _)| m)),
                common       = %xbracket.map_or(String::new(), |(_, c)| c.to_string()),
                bid_yes      = %bid_yes,
                bid_no       = %bid_no,
                size         = %yes_remaining,
                // Volatility regime + calendar: free features.
                sigma5       = %opt_dec(vol.sigma_short),
                sigma60      = %opt_dec(vol.sigma_long),
                vol_ratio    = %opt_dec(vol_ratio(vol.sigma_short, vol.sigma_long)),
                secs_to_funding = secs_to_funding(crate::time::now().timestamp()),
                // Informed directional flow (Binance CVD). Normalized ratio
                // ∈ [−1,1] (denominator vol_quote_60s in both windows,
                // see `cvd_ratio`) plus the total, to tell whether the flow was thin or not.
                cvd_ratio_5s  = %opt_dec(cvd_ratio(vol.cvd_5s, vol.vol_quote_60s)),
                cvd_ratio_60s = %opt_dec(cvd_ratio(vol.cvd_60s, vol.vol_quote_60s)),
                vol_quote_60s = %opt_dec(vol.vol_quote_60s),
                // Signed flow from the Polymarket tape (shares,
                // window `market_feed.tape_window_secs`) per token of the pair itself.
                tape_yes_net   = %opt_dec(book.yes.flow_net),
                tape_yes_total = %opt_dec(book.yes.flow_total),
                tape_no_net    = %opt_dec(book.no.flow_net),
                tape_no_total  = %opt_dec(book.no.flow_total),
                remaining    = remaining,
                // Bracket duration: if market_open_ts==0 the mark is invalid →
                // empty = unknown (the offline analysis ignores it).
                duration     = %(if book.market_open_ts > 0 {
                    (book.clear_at - book.market_open_ts).to_string()
                } else { String::new() }),
                // History of results of sibling brackets (same
                // asset + duration) — empty if bracket_history_len=0 or no
                // history yet (never blocks).
                hist_n       = %hist_stats.map_or(String::new(), |(n, _, _, _)| n.to_string()),
                hist_boxes   = %hist_stats.map_or(String::new(), |(_, b, _, _)| b.to_string()),
                hist_orphans = %hist_stats.map_or(String::new(), |(_, _, o, _)| o.to_string()),
                hist_streak  = %hist_stats.map_or(String::new(), |(_, _, _, s)| s.to_string()),
                // Net/gross notional liquidated in futures (60s window) —
                // liquidation-cascade pressure. Empty when the feed is off
                // (`oracle.futures_liq_enabled=false`, the default) or without
                // enough history yet; never blocks the entry.
                liq_net_60s   = %opt_dec(vol.liq_net_60s),
                liq_total_60s = %opt_dec(vol.liq_total_60s),
                "PM: entry"
            );
        }

        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::ParityMaker {
            return;
        }
        // Fill without state (restart with a live GTD): without knowing which token is
        // YES, the imbalance cannot be attributed; surface it and defer to the
        // orphan checker and the startup reconciliation.
        if !self.markets.contains_key(&fill.market_cid) {
            if fill.side == FillSide::Buy {
                tracing::warn!(cid = %fill.market_cid, token = %fill.token_id,
                    "PM: Buy fill without state — leg not tracked (restart), left to the orphan checker");
                metrics::counter!("hrf_strategy_fill_without_state_total", "strategy" => "parity_maker").increment(1);
            }
            return;
        }
        if let Some(state) = self.markets.get_mut(&fill.market_cid) {
            match fill.side {
                FillSide::Buy => {
                    // A fill consumes a live order: subtract it from the in-flight size.
                    if fill.token_id == state.yes_token {
                        state.yes_filled += fill.size;
                        state.yes_cost += fill.price * fill.size;
                        state.yes_live = (state.yes_live - fill.size).max(Decimal::ZERO);
                        // Full fill: the GTD is dead, and without a resting mark tolerance
                        // would not keep it as a ghost. Partial fill →
                        // it stays live at the same price, so the mark stays.
                        if state.yes_live == Decimal::ZERO {
                            state.posted_bid_yes = None;
                        }
                        // The YES leg WAS touched. Do not use `yes_cost > 0` as a
                        // proxy: a full dump of the inventory returns it to 0, and that
                        // evidence (it got filled) would be lost.
                        state.ever_yes = true;
                        metrics::counter!("hrf_pm_fill_total", "side" => "yes").increment(1);
                    } else if fill.token_id == state.no_token {
                        state.no_filled += fill.size;
                        state.no_cost += fill.price * fill.size;
                        state.no_live = (state.no_live - fill.size).max(Decimal::ZERO);
                        if state.no_live == Decimal::ZERO {
                            state.posted_bid_no = None;
                        }
                        state.ever_no = true;
                        metrics::counter!("hrf_pm_fill_total", "side" => "no").increment(1);
                    }
                    // First time both legs have a fill → complete box.
                    if !state.box_counted
                        && state.yes_filled > Decimal::ZERO
                        && state.no_filled > Decimal::ZERO
                    {
                        state.box_counted = true;
                        metrics::counter!("hrf_pm_box_completed_total").increment(1);
                    }
                }
                FillSide::Sell => {
                    if fill.token_id == state.yes_token {
                        if state.yes_filled > Decimal::ZERO {
                            let sold_frac = (fill.size / state.yes_filled).min(Decimal::ONE);
                            state.yes_cost -= state.yes_cost * sold_frac;
                        }
                        state.yes_filled = (state.yes_filled - fill.size).max(Decimal::ZERO);
                    } else if fill.token_id == state.no_token {
                        if state.no_filled > Decimal::ZERO {
                            let sold_frac = (fill.size / state.no_filled).min(Decimal::ONE);
                            state.no_cost -= state.no_cost * sold_frac;
                        }
                        state.no_filled = (state.no_filled - fill.size).max(Decimal::ZERO);
                    }
                }
                // The intent was blocked by a gate BEFORE existing as an order: undo ONLY its
                // contribution to `live` (`fill.size`/`fill.price` are those of the blocked intent).
                // Never `live = 0`: there may be ANOTHER real GTD live on the same leg, and the
                // reset would fabricate an early re-post and violate the cap (same pattern as
                // the cancel ack below). The resting mark is only cleared if it still points at the
                // blocked intent's price. If a later re-post already moved it to another
                // bid, it is kept: it belongs to a different intent.
                // (the GateRejected branch below does this).
                FillSide::GateRejected => {
                    if fill.token_id == state.yes_token {
                        state.yes_live = (state.yes_live - fill.size).max(Decimal::ZERO);
                        if state.posted_bid_yes == Some(fill.price) {
                            state.posted_bid_yes = None;
                        }
                    } else if fill.token_id == state.no_token {
                        state.no_live = (state.no_live - fill.size).max(Decimal::ZERO);
                        if state.posted_bid_no == Some(fill.price) {
                            state.posted_bid_no = None;
                        }
                    }
                }
                // TERMINAL events of an order that never filled:
                // a cancellation confirmed by the router, or a rejection. Only here (and in the
                // fill) is it known that the order is dead, and only here can its in-flight size be released.
                // Emitting the cancel tells us nothing: assuming it was done
                // was what piled up the GTDs (12 orders, 36 shares, with a cap of 3).
                //
                // The rejection goes in the same branch because, if it is not released, the leg stays
                // silent forever (`live` blocking the cap with no live order).
                FillSide::Cancelled | FillSide::Rejected => {
                    if fill.token_id == state.yes_token {
                        state.yes_live = Decimal::ZERO;
                        state.posted_bid_yes = None; // resting order is dead: no mark
                    } else if fill.token_id == state.no_token {
                        state.no_live = Decimal::ZERO;
                        state.posted_bid_no = None;
                    }
                    // A `Rejected` on the RETAINED leg during the
                    // unwind can only be the unwind sell rejected by the
                    // router (PM does not quote bids on that leg while
                    // `unwind_sent` is set. Without this reset, the flag
                    // stayed latched forever and the losing leg
                    // rolled to settlement with no management.
                    // It is NOT done with `Cancelled`: during the unwind PM cancels
                    // its live bids (`cancel_live_orders`) and their acks also travel
                    // over the retained token: resetting there would re-open the problem
                    // by cancelling the freshly placed ask.
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
                                "PM: unwind sell rejected — unwind_sent reset, re-decided on the next tick");
                        }
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
        strategy: StrategyId::ParityMaker,
        resting: true,
        real_clear_at: None,
    }
}

fn sell_intent(book: &BookSnapshot, token_id: &str, bid: Decimal, size: Decimal) -> OrderIntent {
    OrderIntent {
        market_cid: book.market_condition_id.clone(),
        token_id: token_id.to_string(),
        side: OrderSide::Sell { size: Some(size) },
        price: bid,
        size,
        clear_at: book.clear_at,
        signal: Decimal::ZERO,
        neg_risk: book.neg_risk,
        limit_price: None,
        second_leg: None,
        strategy: StrategyId::ParityMaker,
        resting: false,
        real_clear_at: None,
    }
}

/// Maker ask GTD to re-post the excess of a single leg — same
/// shape as `sell_intent` but with `resting: true` and a `limit_price` (the
/// `ask_maker_intent` pattern from theta_capture).
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
        strategy: StrategyId::ParityMaker,
        resting: true,
        real_clear_at: None,
    }
}

/// Cancels the live GTDs of both legs and resets the in-flight accounting. Every path that stops
/// managing the bracket MUST go through here:
/// an early return without cancelling leaves bids hanging right in the toxic zone
/// (<0.20, markout −3310 bps).
fn cancel_live_orders(state: &mut PMState, book: &BookSnapshot, intents: &mut Vec<OrderIntent>) {
    if state.posted_at.is_some() {
        intents.push(cancel_intent(book, &book.yes.token_id));
        intents.push(cancel_intent(book, &book.no.token_id));
        state.posted_at = None;
        // Resting marks of the requote tolerance: dead together with
        // posted_at — this path (dangerous) always cancels both legs.
        state.posted_bid_yes = None;
        state.posted_bid_no = None;
        // `yes_live`/`no_live` are NOT touched here. Emitting a cancel does not kill the
        // order: the cancel travels async and the order may fill in the meantime. Treating it
        // as dead here is what voided the very protection of the in-flight cap: PM would re-post
        // believing it had nothing live, and the fill of the earlier order arrived
        // after the next requote. Measured: 12 GTDs and 36 shares with a cap of 3.
        // The in-flight size is only released by a confirmed TERMINAL event (fill,
        // cancel ack or rejection), in `on_fill`.
    }
}

fn cancel_intent(book: &BookSnapshot, token_id: &str) -> OrderIntent {
    super::cancel_intent(
        &book.market_condition_id,
        token_id,
        book.clear_at,
        book.neg_risk,
        StrategyId::ParityMaker,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};
    use rust_decimal_macros::dec;

    // ── Free features ────────────────────────────────────
    #[test]
    fn funding_distance_at_edges() {
        // exact multiple of 8h → 0 (we are RIGHT at the window)
        assert_eq!(secs_to_funding(0), 0);
        assert_eq!(secs_to_funding(28_800), 0);
        assert_eq!(secs_to_funding(3 * 28_800), 0);
        // 1s before the window → 1
        assert_eq!(secs_to_funding(28_800 - 1), 1);
        // half the period → 14_400
        assert_eq!(secs_to_funding(28_800 / 2), 14_400);
    }

    #[test]
    fn funding_never_negative() {
        for t in [0_i64, 1, 12_345, 28_799, 28_800, 100_000, 1_700_000_000] {
            let d = secs_to_funding(t);
            assert!((0..28_800).contains(&d), "t={t} gave {d}");
        }
    }

    #[test]
    fn vol_ratio_none_safe() {
        // σ60 = 0 or None → None (never divides by zero)
        assert_eq!(vol_ratio(Some(dec!(0.01)), Some(dec!(0))), None);
        assert_eq!(vol_ratio(Some(dec!(0.01)), None), None);
        assert_eq!(vol_ratio(None, Some(dec!(0.02))), None);
        // both present → quotient
        assert_eq!(vol_ratio(Some(dec!(0.02)), Some(dec!(0.01))), Some(dec!(2)));
    }

    fn make_book(remaining: i64) -> BookSnapshot {
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        BookSnapshot {
            market_condition_id: "pm-cid".into(),
            yes: TokenBook {
                token_id: "yes-t".into(),
                bids: bids("0.50"),
                asks: asks("0.55"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "no-t".into(),
                bids: bids("0.43"),
                asks: asks("0.47"),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-pm".into(),
            reference_price: None,
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

    /// VolSnapshot with FV available: fresh oracle, with sigma and an explicit `last_price`.
    /// Used by the tests that need a real FV, not `None`.
    fn make_vol_fv(last_price: &str, sigma: &str) -> VolSnapshot {
        VolSnapshot {
            asset: "btc".into(),
            sigma_short: Some(sigma.parse().unwrap()),
            sigma_long: Some(sigma.parse().unwrap()),
            last_price: last_price.parse().unwrap(),
            updated_at: crate::time::Instant::now(),
            ..VolSnapshot::default()
        }
    }

    /// The executor confirms to the router that the GTDs of both legs are dead.
    /// In production it comes back within milliseconds; in the tests there is no executor, so
    /// it must be delivered by hand, and if it is not, PM does NOT re-post. That is not a defect
    /// of the test: it is exactly the protection that prevents stacking GTDs.
    fn ack_cancels(pm: &mut ParityMaker, book: &BookSnapshot) {
        for token in [book.yes.token_id.clone(), book.no.token_id.clone()] {
            pm.on_fill(&Fill {
                hedge_id: None,
                exec_id: None,
                local_order_id: None,
                market_cid: book.market_condition_id.clone(),
                token_id: token,
                side: FillSide::Cancelled,
                price: dec!(0),
                size: dec!(0),
                order_id: Some("oid".into()),
                ts: crate::time::now(),
                mode: crate::types::Mode::Paper,
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: 0,
                outcome_label: None,
            });
        }
    }

    fn test_cfg() -> ParityMakerConfig {
        ParityMakerConfig {
            margin: dec!(0.05),
            min_leg_price: dec!(0.05),
            order_size_shares: dec!(3),
            requote_threshold_bps: 20,
            pair_ttl_secs: 10,
            ..ParityMakerConfig::default()
        }
    }

    /// The entry barrier is the movement left for the underlying (σ·√t),
    /// not a number of seconds. Without room to move it does not open; but a box already started
    /// keeps being quoted (the missing leg turns an orphan into a box).
    #[test]
    fn pm_does_not_open_without_path_but_completes_box() {
        // threshold 20 bps. σ = 0.0002/√s → σ·√t with t=40s ≈ 13 bps: bracket already decided.
        // pair_ttl_secs = 0 → the requote fires on every tick, with no waiting.
        let cfg = ParityMakerConfig {
            min_expected_move_bps: 20,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let no_move_book = make_book(40);
        let vol = make_vol_fv("100000", "0.0002");

        assert_eq!(
            pm.on_book(&no_move_book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0,
            "without expected movement it must not open: the loser can only fall"
        );

        // Same σ but with time to spare (t=3600s → σ·√t ≈ 120 bps): it does open.
        let live_book = make_book(3000);
        assert_eq!(
            pm.on_book(&live_book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "with movement ahead it must quote both legs"
        );

        // And with one leg already held, it completes the box even with no movement left.
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: no_move_book.market_condition_id.clone(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.45),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });
        // The NO bid is still live on the book; the requote withdraws it and we have to
        // wait for the confirmation before re-posting it.
        pm.on_book(&no_move_book, &vol);
        ack_cancels(&mut pm, &no_move_book);
        assert!(
            pm.on_book(&no_move_book, &vol)
                .iter()
                .any(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t"),
            "with one leg held it must keep quoting the missing one to close the box"
        );
    }

    /// A 5-minute bracket (nominal duration 300s, below the
    /// 900s threshold) must not OPEN a new box.
    #[test]
    fn pm_does_not_open_5m_bracket_due_to_duration_gate() {
        let cfg = ParityMakerConfig {
            pm_min_duration_secs: 900,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let mut book = make_book(200);
        book.market_open_ts = book.clear_at - 300; // duration=300 < 900: 5-minute bracket
        let vol = make_vol();

        assert_eq!(
            pm.on_book(&book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0,
            "5-minute bracket (duration=300 < pm_min_duration_secs=900) must not open a box"
        );
    }

    /// A 15-minute bracket (nominal duration 900s, at the threshold)
    /// must open — the gate only blocks BELOW the threshold.
    #[test]
    fn pm_opens_15m_bracket_despite_duration_gate() {
        let cfg = ParityMakerConfig {
            pm_min_duration_secs: 900,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let mut book = make_book(1000);
        book.market_open_ts = book.clear_at - 900; // duration=900: at the threshold, not below it
        let vol = make_vol();

        assert_eq!(
            pm.on_book(&book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "15-minute bracket (duration=900, not < pm_min_duration_secs) must open both legs"
        );
    }

    /// Unknown duration (`market_open_ts` not populated, `duration=0`)
    /// is a missing signal: it never blocks, not even with the gate on.
    #[test]
    fn pm_unknown_duration_does_not_block_open() {
        let cfg = ParityMakerConfig {
            pm_min_duration_secs: 900,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300); // make_book() leaves market_open_ts=0 -> duration=0 (unknown)
        let vol = make_vol();

        assert_eq!(
            pm.on_book(&book, &vol).iter().filter(|i| matches!(i.side, OrderSide::Buy)).count(), 2,
            "unknown duration (market_open_ts=0) is a missing signal: it must not block the opening"
        );
    }

    /// The gate only blocks OPENING: a box already started in a bracket
    /// of 5 minutes must still be completable (the missing leg turns an
    /// orphan into a box, as with the entry barrier).
    #[test]
    fn pm_completes_box_in_5m_bracket_despite_duration_gate() {
        let cfg = ParityMakerConfig {
            pm_min_duration_secs: 900,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let mut book = make_book(200);
        book.market_open_ts = book.clear_at - 300; // duration=300 < 900: 5-minute bracket
        let vol = make_vol();

        // Without inventory, the gate blocks the opening.
        assert_eq!(
            pm.on_book(&book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0,
            "without inventory, the 5-minute bracket must not open"
        );

        // A fill of the YES leg arrives (e.g. a live order from before the
        // bracket crossed the threshold, or a fill outside the band) — with one leg in
        // the portfolio the gate must no longer prevent quoting the missing one.
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: book.market_condition_id.clone(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.45),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });
        // The NO bid is still live on the book; the requote withdraws it and we have to
        // wait for the confirmation before re-posting it.
        pm.on_book(&book, &vol);
        ack_cancels(&mut pm, &book);
        assert!(
            pm.on_book(&book, &vol)
                .iter()
                .any(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t"),
            "with one leg held, the duration gate must let the box complete"
        );
    }

    /// A bracket that forced an unwind dump (sell of the losing leg) is
    /// toxic and is not reopened — adverse selection would fill the losing leg again.
    /// It only blocks OPENING; completing a box already started is still allowed.
    #[test]
    fn pm_does_not_reopen_burned_bracket_after_dump() {
        // single_leg_unwind_secs = 1 to force a fast unwind; without FV (make_vol) →
        // decide_unwind falls into DumpTaker (None → dump). max_dumps_per_bracket = 1.
        let cfg = ParityMakerConfig {
            single_leg_unwind_secs: 1,
            max_dumps_per_bracket: 1,
            min_unwind_bid: dec!(0.05),
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = make_vol();

        // Opens and fills ONLY YES → single leg.
        pm.on_book(&book, &vol);
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: book.market_condition_id.clone(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.40),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });
        pm.on_book(&book, &vol); // mark unbalanced_since
        std::thread::sleep(std::time::Duration::from_millis(1100));

        // The unwind window runs out → dump of the single leg (there is a bid in the fixture, 0.50).
        let out = pm.on_book(&book, &vol);
        assert!(
            out.iter().any(|i| matches!(i.side, OrderSide::Sell { .. })),
            "must dump the losing leg when the unwind window runs out"
        );

        // The dump executes and the position returns to zero (sell fill).
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: book.market_condition_id.clone(),
            token_id: "yes-t".into(),
            side: FillSide::Sell,
            price: dec!(0.34),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });
        ack_cancels(&mut pm, &book);

        // Burned bracket: even if the book is still quotable, PM does NOT reopen it.
        let out = pm.on_book(&book, &vol);
        assert_eq!(
            out.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0,
            "after the dump the bracket is burned: it is not reopened"
        );
    }

    /// Regression: with the unwind's maker ask live on the retained
    /// leg (`unwind_sent && imbalance != 0`), the requote of the next tick
    /// must NOT quote a buy bid on that same leg again:
    /// the executor's anti-duplicate, keyed by (cid, token) WITHOUT side,
    /// (`limit_key`), would cancel the ask ~1 s after placing it (executor.rs
    /// The missing leg (chase) must keep being quoted.
    #[test]
    fn pm_does_not_quote_bid_on_retained_leg_after_repost_maker() {
        let cfg = ParityMakerConfig {
            single_leg_unwind_secs: 1,
            min_unwind_bid: dec!(0.01),
            unwind_hold_fv: dec!(0.60),
            unwind_dump_fv: dec!(0.40),
            unwind_repost_min_secs: 60,
            unwind_maker_edge: dec!(0.01),
            pair_ttl_secs: 0,
            // Larger than what gets filled, so there is leftover size ("remaining") with which
            // the original bug would re-post too much on the retained leg.
            order_size_shares: dec!(5),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let mut book = make_book(300); // remaining=300 > unwind_repost_min_secs=60
        book.reference_price = Some(dec!(100000)); // == last_price → FV≈0.5, neutral zone
        let vol = make_vol_fv("100000", "0.0002");

        // Opens and fills ONLY YES (3 of the 5 PM would ask for) → single leg.
        pm.on_book(&book, &vol);
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: book.market_condition_id.clone(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.40),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });
        pm.on_book(&book, &vol); // mark unbalanced_since
        std::thread::sleep(std::time::Duration::from_millis(1100));

        // The unwind window runs out: neutral FV (≈0.5) + time to spare → RepostMaker.
        let out = pm.on_book(&book, &vol);
        assert!(
            out.iter().any(|i| matches!(i.side, OrderSide::Sell { .. })
                && i.token_id == "yes-t"
                && i.resting),
            "must re-post a resting maker ask on the retained leg"
        );

        ack_cancels(&mut pm, &book);

        // Next tick: the maker ask is still live (unwind_sent=true), imbalance
        // is still != 0 (its fill has not come back yet). WITHOUT the fix, PM would
        // quote a bid on "yes-t" again (2 shares are left under order_size_shares=5)
        // and the executor's anti-duplicate would cancel the freshly placed ask.
        let out = pm.on_book(&book, &vol);
        assert_eq!(
            out.iter().filter(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "yes-t").count(), 0,
            "must not quote a bid on the retained leg again while the unwind ask is still live"
        );
        assert!(
            out.iter()
                .any(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t"),
            "the missing leg (chase) must keep being quoted"
        );
    }

    /// If the ROUTER rejects the unwind sell (maker ask or dump), the
    /// executor emits a `Fill` Rejected and `unwind_sent` must be released — if
    /// not, the retained leg stays silent forever (it rolls to settlement). The
    /// next tick must re-decide the unwind and re-emit the sell.
    /// A `Cancelled` on the same token (ack of the bids cancelled by
    /// `cancel_live_orders`) must NOT release it.
    #[test]
    fn pm_unwind_rejected_unblocks_unwind_sent_but_not_cancel() {
        let cfg = ParityMakerConfig {
            single_leg_unwind_secs: 1,
            min_unwind_bid: dec!(0.01),
            unwind_hold_fv: dec!(0.60),
            unwind_dump_fv: dec!(0.40),
            unwind_repost_min_secs: 60,
            unwind_maker_edge: dec!(0.01),
            pair_ttl_secs: 0,
            order_size_shares: dec!(5),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let mut book = make_book(300);
        book.reference_price = Some(dec!(100000)); // FV≈0.5 → neutral zone → RepostMaker
        let vol = make_vol_fv("100000", "0.0002");
        let fill_of = |side: FillSide, token: &str| Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: book.market_condition_id.clone(),
            token_id: token.into(),
            side,
            price: dec!(0.40),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        };

        // Single YES leg filled → unwind window → RepostMaker emits the ask.
        pm.on_book(&book, &vol);
        pm.on_fill(&fill_of(FillSide::Buy, "yes-t"));
        pm.on_book(&book, &vol);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let out = pm.on_book(&book, &vol);
        assert!(
            out.iter()
                .any(|i| matches!(i.side, OrderSide::Sell { .. }) && i.resting),
            "precondition: the unwind ask was emitted"
        );

        // (a) A Cancelled on the retained token (ack of the bids) does NOT release
        // the unwind: the next tick must not re-emit the sell or quote a bid
        // on the retained leg.
        ack_cancels(&mut pm, &book);
        let out = pm.on_book(&book, &vol);
        assert!(
            out.iter()
                .all(|i| !(matches!(i.side, OrderSide::Sell { .. }) && i.token_id == "yes-t")),
            "a Cancelled of a bid must not restart the unwind"
        );

        // (b) A Rejected on the retained token DOES release: the unwind sell
        // died at the router without reaching the book.
        pm.on_fill(&fill_of(FillSide::Rejected, "yes-t"));
        let out = pm.on_book(&book, &vol);
        assert!(
            out.iter()
                .any(|i| matches!(i.side, OrderSide::Sell { .. }) && i.token_id == "yes-t"),
            "after a Rejected unwind sell, PM must re-decide and re-emit the sell"
        );
    }

    /// Emitting a cancel does NOT kill the order: the cancel travels async and the order
    /// may fill in the meantime. Until the ack arrives, the size stays in flight and PM
    /// must NOT re-post: if it does, it piles up GTDs (12 orders and 36 shares with a cap
    /// of 3, measured in calibration). It is not a time wait: it is
    /// waiting for a CONFIRMATION.
    #[test]
    fn pm_does_not_repost_until_cancellation_ack() {
        // pair_ttl_secs = 0 → requote on every tick, no artificial waits.
        let cfg = ParityMakerConfig {
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = make_vol();

        let buys = |v: &[OrderIntent]| {
            v.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count()
        };
        let cancels = |v: &[OrderIntent]| {
            v.iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count()
        };

        let out = pm.on_book(&book, &vol);
        assert_eq!(buys(&out), 2, "initial posting of both legs");

        // Requote: it cancels… but the ack has not come back yet. It must NOT re-post.
        let out = pm.on_book(&book, &vol);
        assert_eq!(cancels(&out), 2, "the requote cancels both legs");
        assert_eq!(
            buys(&out),
            0,
            "without the cancel ack the size is still in flight: re-posting here stacks GTDs"
        );

        // And it still does not re-post while the ack has not arrived, however many ticks pass.
        let out = pm.on_book(&book, &vol);
        assert_eq!(
            buys(&out),
            0,
            "the blocker depends on the confirmation, not on time"
        );

        // The router's confirmation arrives for both legs.
        for token in ["yes-t", "no-t"] {
            pm.on_fill(&Fill {
                hedge_id: None,
                exec_id: None,
                local_order_id: None,
                market_cid: book.market_condition_id.clone(),
                token_id: token.into(),
                side: FillSide::Cancelled,
                price: dec!(0),
                size: dec!(0),
                order_id: Some("oid".into()),
                ts: crate::time::now(),
                mode: crate::types::Mode::Paper,
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: 0,
                outcome_label: None,
            });
        }

        let out = pm.on_book(&book, &vol);
        assert_eq!(
            buys(&out),
            2,
            "once the death of the orders is confirmed, PM quotes again"
        );
    }

    /// A fill is also a terminal event: it releases the in-flight size, but
    /// then the cap is already consumed by `filled` and PM must not re-post that leg either.
    #[test]
    fn pm_does_not_repost_already_filled_leg() {
        let cfg = ParityMakerConfig {
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = make_vol();

        pm.on_book(&book, &vol); // posts 3 shares on each leg
        pm.on_fill(&Fill {
            // YES fills entirely
            hedge_id: None,
            market_cid: book.market_condition_id.clone(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.48),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
            exec_id: None,
            local_order_id: None,
        });

        let out = pm.on_book(&book, &vol);
        assert_eq!(
            out.iter()
                .filter(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "yes-t")
                .count(),
            0,
            "the YES leg already has its 3 shares: the cap is consumed, no re-post"
        );
    }

    /// A clean DIRECTIONAL move (YES rises by the same amount NO falls) leaves the
    /// box mid `(yes_ask + no_ask)/2` unchanged, because that mid measures parity, not
    /// direction. Before the probability trigger existed, PM did not requote and left the
    /// two bids hanging during the move that was going to sweep them.
    #[test]
    fn pm_requotes_on_directional_move_at_constant_parity() {
        // t0: yes_ask 0.55 / no_ask 0.47 → sum 1.02, mid 0.51.
        // t1: yes_ask 0.62 / no_ask 0.40 → sum 1.02, mid 0.51 (identical!).
        //     But the probability of YES has moved 7 points.
        let t0 = make_book(300);
        let mut t1 = make_book(300);
        t1.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.57),
                size: dec!(100),
            }],
        };
        t1.yes.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.62),
                size: dec!(100),
            }],
        };
        t1.no.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.36),
                size: dec!(100),
            }],
        };
        t1.no.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.40),
                size: dec!(100),
            }],
        };

        let mid =
            |b: &BookSnapshot| (b.yes.best_ask().unwrap() + b.no.best_ask().unwrap()) / dec!(2);
        assert_eq!(
            mid(&t0),
            mid(&t1),
            "the fixture must leave the box mid unchanged"
        );

        // High TTL and a 150 bps threshold: without the probability trigger nothing would fire the requote.
        // Without an oracle (make_vol) there is no FV to save the situation either.
        let cfg = ParityMakerConfig {
            pair_ttl_secs: 3600,
            requote_threshold_bps: 150,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let vol = make_vol();

        assert_eq!(
            pm.on_book(&t0, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "initial posting"
        );
        // Same book: nothing to requote.
        let out = pm.on_book(&t0, &vol);
        assert_eq!(
            out.len(),
            0,
            "no movement, no requote: neither bids nor cancels"
        );

        // Directional move with constant parity: PM must REACT, and reacting
        // first means withdrawing the stale bids (re-posting requires the ack).
        let out = pm.on_book(&t1, &vol);
        assert_eq!(
            out.iter().filter(|i| matches!(i.side, OrderSide::Cancel)).count(), 2,
            "the bracket moved 7 probability points: the old bids are stale and are withdrawn"
        );

        // Once the withdrawal is confirmed, re-posts at the new price.
        ack_cancels(&mut pm, &t1);
        assert_eq!(
            pm.on_book(&t1, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "with the old orders already dead, PM quotes again"
        );
    }

    /// With a single leg, the bid of the missing one is set AT the CEILING of the
    /// residual budget — the extra margin is given up to close the box, but
    /// the ceiling is never exceeded (the box still closes at `1 − margin`).
    #[test]
    fn pm_chase_quotes_at_residual_ceiling() {
        // margin 0.05 → box budget 0.95. YES bought at 0.45 → NO ceiling = 0.50.
        let mut book = make_book(300);
        book.no.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.60),
                size: dec!(100),
            }],
        };
        let vol = make_vol();

        let buy_yes = |pm: &mut ParityMaker| {
            pm.on_fill(&Fill {
                hedge_id: None,
                exec_id: None,
                local_order_id: None,
                market_cid: "pm-cid".into(),
                token_id: "yes-t".into(),
                side: FillSide::Buy,
                price: dec!(0.45),
                size: dec!(3),
                order_id: None,
                ts: crate::time::now(),
                mode: crate::types::Mode::Paper,
                neg_risk: false,
                strategy: StrategyId::ParityMaker,
                clear_at: 0,
                outcome_label: None,
            })
        };
        let bid_of_no = |intents: &[OrderIntent]| {
            intents
                .iter()
                .find(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t")
                .map(|i| i.price)
        };

        // Without chase: proportional split, capped by the ceiling → 0.49.
        let cfg = ParityMakerConfig {
            chase_after_secs: 0,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        pm.on_book(&book, &vol);
        buy_yes(&mut pm);
        pm.on_book(&book, &vol); // requote: withdraws the live GTD of NO
        ack_cancels(&mut pm, &book); // the router confirms
        assert_eq!(
            bid_of_no(&pm.on_book(&book, &vol)),
            Some(dec!(0.49)),
            "without chase, the missing leg is quoted at the proportional split"
        );

        // With chase and the time elapsed: at the ceiling → 0.50.
        let cfg = ParityMakerConfig {
            chase_after_secs: 1,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        pm.on_book(&book, &vol);
        buy_yes(&mut pm);
        pm.on_book(&book, &vol); // mark unbalanced_since and withdraws the GTD of NO
        ack_cancels(&mut pm, &book); // the router confirms the death
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert_eq!(
            bid_of_no(&pm.on_book(&book, &vol)),
            Some(dec!(0.50)),
            "with chase, the missing leg is quoted at the residual ceiling (0.95 − 0.45)"
        );
    }

    /// The chase gives up the extra margin, never the configured margin — the
    /// residual budget ceiling is impassable even if the ask is much higher.
    #[test]
    fn pm_chase_respects_budget() {
        let mut book = make_book(300);
        book.no.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.90),
                size: dec!(100),
            }],
        };
        let vol = make_vol();

        let cfg = ParityMakerConfig {
            chase_after_secs: 1,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        pm.on_book(&book, &vol);
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-cid".into(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.45),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });
        pm.on_book(&book, &vol);
        ack_cancels(&mut pm, &book);
        std::thread::sleep(std::time::Duration::from_millis(1100));

        let bid = pm
            .on_book(&book, &vol)
            .iter()
            .find(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t")
            .map(|i| i.price);
        assert_eq!(
            bid,
            Some(dec!(0.50)),
            "the chase never exceeds the ceiling: the box closes at 1 − margin (0.45 + 0.50 = 0.95)"
        );
    }

    /// The unwind hold-vs-dump decision uses the oracle spot.
    /// Binance well below the
    /// strike → FV(YES) ≈ 0 → the retained YES leg is dropped; well above → hold.
    #[test]
    fn pm_unwind_decides_with_spot_oracle() {
        let mut book = make_book(300);
        book.reference_price = Some(dec!(100000)); // strike
        let cfg = ParityMakerConfig {
            single_leg_unwind_secs: 1,
            unwind_hold_fv: dec!(0.60),
            unwind_dump_fv: dec!(0.40),
            pair_ttl_secs: 0,
            ..test_cfg()
        };

        let fill_yes = Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-cid".into(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.45),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        };

        // Spot well below the strike → the YES leg is lost → dump.
        let bad_vol = make_vol_fv("95000", "0.001");
        let mut pm = ParityMaker::new(cfg.clone());
        pm.on_book(&book, &bad_vol);
        pm.on_fill(&fill_yes);
        pm.on_book(&book, &bad_vol); // mark unbalanced_since
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert!(
            pm.on_book(&book, &bad_vol)
                .iter()
                .any(|i| matches!(i.side, OrderSide::Sell { .. })),
            "with the spot well below the strike the retained YES leg is dropped"
        );

        // Spot well above the strike → FV ≈ 1 → hold, no sell.
        let good_vol = make_vol_fv("105000", "0.001");
        let mut pm = ParityMaker::new(cfg);
        pm.on_book(&book, &good_vol);
        pm.on_fill(&fill_yes);
        pm.on_book(&book, &good_vol);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert!(
            !pm.on_book(&book, &good_vol)
                .iter()
                .any(|i| matches!(i.side, OrderSide::Sell { .. })),
            "with a favorable spot the retained leg is held to settlement"
        );
    }

    /// Book of a specific bracket of the same asset, with the probability of
    /// YES given to it (the NO side stays as in the fixture: `implied_yes`
    /// only looks at YES).
    fn book_bracket(cid: &str, yes_bid: &str, yes_ask: &str) -> BookSnapshot {
        let mut b = make_book(300);
        b.market_condition_id = cid.into();
        b.yes.token_id = format!("{cid}-yes");
        b.no.token_id = format!("{cid}-no");
        b.yes.bids = HalfBook {
            levels: vec![Level {
                price: yes_bid.parse().unwrap(),
                size: dec!(100),
            }],
        };
        b.yes.asks = HalfBook {
            levels: vec![Level {
                price: yes_ask.parse().unwrap(),
                size: dec!(100),
            }],
        };
        b
    }

    fn cfg_xbracket() -> ParityMakerConfig {
        ParityMakerConfig {
            xbracket_window_secs: 1,
            xbracket_move_bps: 100, // 1 probability point
            xbracket_common_ratio: dec!(0.5),
            oracle_drift_secs: 0, // isolate from the spot-drift gate
            pair_ttl_secs: 0,     // requote on every tick, no waiting
            ..test_cfg()
        }
    }

    /// If ALL the brackets of the asset move at once, the underlying moved
    /// → informed flow → the leg opposite the movement is doomed
    /// and must not be quoted.
    #[test]
    fn pm_common_xbracket_condemns_leg() {
        let mut pm = ParityMaker::new(cfg_xbracket());
        let vol = make_vol();

        // t0: both brackets of the same asset, still.
        pm.on_book(&book_bracket("a", "0.50", "0.55"), &vol);
        pm.on_book(&book_bracket("b", "0.50", "0.55"), &vol);

        std::thread::sleep(std::time::Duration::from_millis(1100));

        // The GTDs from the initial posting are still live until the router confirms
        // their death. The ack is delivered to isolate the cross-bracket signal, which is what this
        // test measures (without it, PM would not re-post because of the cap and the test would prove nothing).
        ack_cancels(&mut pm, &book_bracket("a", "0.50", "0.55"));
        ack_cancels(&mut pm, &book_bracket("b", "0.50", "0.55"));

        // t1: BOTH rise ~10 probability points → the spot moved.
        // The sibling first, so that its new point is in the history.
        pm.on_book(&book_bracket("b", "0.60", "0.65"), &vol);
        let intents = pm.on_book(&book_bracket("a", "0.60", "0.65"), &vol);

        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "a-no")
                .count(),
            0,
            "common upward move: the NO bid is doomed, it must not be quoted"
        );
        assert!(
            intents
                .iter()
                .any(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "a-yes"),
            "the leg favored by the movement is quoted"
        );
    }

    /// If ONLY this book moves and the siblings stay still, it is idiosyncratic flow
    /// — the impatient seller a maker lives on. Quote both.
    #[test]
    fn pm_idiosyncratic_xbracket_does_not_block() {
        let mut pm = ParityMaker::new(cfg_xbracket());
        let vol = make_vol();

        pm.on_book(&book_bracket("a", "0.50", "0.55"), &vol);
        pm.on_book(&book_bracket("b", "0.50", "0.55"), &vol);

        std::thread::sleep(std::time::Duration::from_millis(1100));

        ack_cancels(&mut pm, &book_bracket("a", "0.50", "0.55"));
        ack_cancels(&mut pm, &book_bracket("b", "0.50", "0.55"));

        // The sibling does NOT move; only A does.
        pm.on_book(&book_bracket("b", "0.50", "0.55"), &vol);
        let intents = pm.on_book(&book_bracket("a", "0.60", "0.65"), &vol);

        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "idiosyncratic move: this is the good flow, both legs are quoted"
        );
    }

    /// With a single live bracket there are no siblings → the signal does not exist → it
    /// cannot block anything (the invariant the PM tests rely on).
    #[test]
    fn pm_xbracket_without_siblings_is_inert() {
        let mut pm = ParityMaker::new(cfg_xbracket());
        let vol = make_vol();

        pm.on_book(&book_bracket("a", "0.50", "0.55"), &vol);
        ack_cancels(&mut pm, &book_bracket("a", "0.50", "0.55"));
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let intents = pm.on_book(&book_bracket("a", "0.60", "0.65"), &vol);

        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "without siblings with data the signal is unavailable and does not block"
        );
    }

    /// Without spread there is no maker premium, only adverse selection. The gate
    /// blocks OPENING, but never the completion of a box already started.
    #[test]
    fn pm_does_not_open_with_narrow_spread() {
        // Fixture: YES spread = 0.55-0.50 = 0.05; NO spread = 0.47-0.43 = 0.04.
        let book = make_book(300);
        let vol = make_vol();

        // min_spread 0.05 → the NO leg (0.04) does not reach it: neither of the two opens.
        let cfg = ParityMakerConfig {
            min_spread: dec!(0.05),
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        assert_eq!(
            pm.on_book(&book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0,
            "with one leg under the minimum spread it must not open"
        );

        // min_spread 0.03 → both legs have room: quotes normally.
        let cfg = ParityMakerConfig {
            min_spread: dec!(0.03),
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        assert_eq!(
            pm.on_book(&book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "with enough spread on both legs it must quote both"
        );

        // With one leg already held, the gate must NOT prevent completing the box: the leg
        // missing is what turns an orphan (-11c) into a box (+margin).
        let cfg = ParityMakerConfig {
            min_spread: dec!(0.05),
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        pm.on_book(&book, &vol); // registers the bracket
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: book.market_condition_id.clone(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.45),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });
        assert!(
            pm.on_book(&book, &vol)
                .iter()
                .any(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t"),
            "the spread gate must not block completing a box already started"
        );
    }

    /// If one leg has much more queue ahead than the other, the other fills
    /// by itself and this one does not: the box is born an orphan. Better not to open it.
    #[test]
    fn pm_does_not_open_with_asymmetric_queue() {
        // With margin 0.05 the computed bids are YES 0.51 and NO 0.43 (split
        // of 0.95 across asks 0.55/0.47).
        // YES queue: 400 shares at 0.52 (ahead of our 0.51).
        // NO queue:    5 shares at 0.43 (our same price → FIFO, they go ahead).
        let mut book = make_book(300);
        book.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.52),
                size: dec!(400),
            }],
        };
        book.no.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.43),
                size: dec!(5),
            }],
        };
        let vol = make_vol();

        let cfg = ParityMakerConfig {
            queue_imbalance_ratio: dec!(5),
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        assert_eq!(
            pm.on_book(&book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0,
            "queue 400 vs 5 (ratio 80 > 5): the box would be born unbalanced, do not open"
        );

        // Same book with the gate disabled (0 = off): quotes as usual.
        let cfg = ParityMakerConfig {
            queue_imbalance_ratio: Decimal::ZERO,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        assert_eq!(
            pm.on_book(&book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "with the gate at 0 the queue blocks nothing"
        );

        // Absolute queue: 400 shares ahead exceed the maximum of 100 → does not open either.
        let cfg = ParityMakerConfig {
            max_queue_ahead_shares: dec!(100),
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        assert_eq!(
            pm.on_book(&book, &vol)
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0,
            "a bid behind 400 shares does not fill: do not open"
        );
    }

    /// With the oracle spot running up, the NO leg sinks and its bid
    /// is doomed to be swept by informed flow: PM must not quote it, and if it has one live
    /// it must cancel it NOW (without waiting for the requote or the cooldown).
    #[test]
    fn pm_directional_gate_withdraws_condemned_leg() {
        let cfg = ParityMakerConfig {
            oracle_drift_secs: 1,
            oracle_drift_bps: 5,                  // 5 bps of the underlying
            fv_no_quote_threshold: Decimal::ZERO, // isolate the directional gate
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);

        // Spot still: PM quotes both legs (benign flow) and they stay live.
        let calm_intents = pm.on_book(&book, &make_vol_fv("100000", "0.02"));
        assert_eq!(
            calm_intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "with the spot still it must quote both legs"
        );

        // The spot jumps +50 bps over the window: the NO leg is doomed.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let running_intents = pm.on_book(&book, &make_vol_fv("100500", "0.02"));

        assert!(
            !running_intents
                .iter()
                .any(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t"),
            "with the spot rising it cannot quote NO: informed flow sweeps it"
        );
        assert!(
            running_intents
                .iter()
                .any(|i| matches!(i.side, OrderSide::Cancel) && i.token_id == "no-t"),
            "the live NO bid must be cancelled immediately"
        );
    }

    /// With one leg already bought, the bid of the missing one is capped to the budget
    /// left in the box. Without this PM completed boxes above 1.00 (average cost
    /// measured: 1.0407) — a guaranteed loss at settlement.
    #[test]
    fn pm_bid_bounded_by_residual_budget() {
        // pair_ttl_secs = 0 → the requote fires on the next tick, with no waiting.
        let cfg = ParityMakerConfig {
            pair_ttl_secs: 0,
            ..test_cfg()
        }; // margin 0.05 → budget 0.95
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300); // asks: yes 0.55, no 0.47
        pm.on_book(&book, &make_vol()); // creates the state

        // We already bought YES at 0.60: the box only has 0.95 - 0.60 = 0.35 left for NO.
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: book.market_condition_id.clone(),
            token_id: "yes-t".into(),
            side: FillSide::Buy,
            price: dec!(0.60),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });

        // Requote: without the cap, the proportional split would ask for NO at ~0.43.
        let bids: Vec<_> = pm
            .on_book(&book, &make_vol())
            .into_iter()
            .filter(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t")
            .collect();
        for b in &bids {
            let px = b.limit_price.unwrap();
            assert!(
                px <= dec!(0.35),
                "NO bid {px} would complete the box above the budget"
            );
        }
    }

    /// Consecutive books (several per second, like the real feed) must not
    /// stack GTDs. Previously each tick re-posted 3 shares because `yes_filled` stayed
    /// at 0 until the fill came back: 129 shares piled up with a cap of 3.
    #[test]
    fn pm_does_not_stack_gtd_on_consecutive_ticks() {
        let mut pm = ParityMaker::new(test_cfg());
        let book = make_book(300);

        let first_intents = pm.on_book(&book, &make_vol());
        assert_eq!(
            first_intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2
        );

        // 20 more books at the same instant: not a single new bid (live size + requote threshold).
        let extra: usize = (0..20)
            .map(|_| {
                pm.on_book(&book, &make_vol())
                    .iter()
                    .filter(|i| matches!(i.side, OrderSide::Buy))
                    .count()
            })
            .sum();
        assert_eq!(
            extra, 0,
            "PM re-posted {extra} bids without waiting for the previous fills"
        );
    }

    #[test]
    fn pm_posts_if_sum_less_than_max() {
        // bid_yes = 0.52 - 0.025 = 0.495, bid_no = 0.43 - 0.025 = 0.405 → sum=0.90 < 0.95 → posts
        let mut pm = ParityMaker::new(test_cfg());
        let book = {
            let bids = |p: &str| HalfBook {
                levels: vec![Level {
                    price: p.parse().unwrap(),
                    size: dec!(10),
                }],
            };
            let asks = |p: &str| HalfBook {
                levels: vec![Level {
                    price: p.parse().unwrap(),
                    size: dec!(10),
                }],
            };
            BookSnapshot {
                market_condition_id: "pm-cid2".into(),
                yes: TokenBook {
                    token_id: "y".into(),
                    bids: bids("0.50"),
                    asks: asks("0.52"),
                    ..Default::default()
                },
                no: TokenBook {
                    token_id: "n".into(),
                    bids: bids("0.40"),
                    asks: asks("0.43"),
                    ..Default::default()
                },
                market_open_ts: 0,
                clear_at: crate::time::now().timestamp() + 300,
                neg_risk: false,
                slug: "btc-updown-5m-pm2".into(),
                reference_price: None,
                fee_rate_bps: None,
                parabolic_fee_rate: None,
                tick_size: None,
                recv_instant: None,
            }
        };
        let intents = pm.on_book(&book, &make_vol());
        let bids: Vec<_> = intents
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(bids.len(), 2);
        assert!(bids.iter().all(|i| i.resting));
        assert!(bids.iter().all(|i| i.size > Decimal::ZERO));
    }

    /// The proportional split quotes even when the asks add up to > 1 (the
    /// normal state of the book) and the sum of the bids respects the budget 1 - margin.
    /// Previously this did not post: the old formula `ask - margin/2` de facto required
    /// `ask_yes + ask_no <= 1`, i.e. a taker arb already served.
    #[test]
    fn pm_proportional_split_respects_budget() {
        let mut pm = ParityMaker::new(test_cfg()); // margin 0.05 → budget 0.95
        let book = make_book(300); // asks 0.55 + 0.47 = 1.02 > 1
        let bids: Vec<_> = pm
            .on_book(&book, &make_vol())
            .into_iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(bids.len(), 2, "must quote both legs");
        let sum: Decimal = bids.iter().map(|i| i.limit_price.unwrap()).sum();
        assert!(sum <= dec!(0.95), "sum {sum} exceeds the budget");
        // And the split follows the book: the expensive leg gets the higher bid.
        let yes = bids.iter().find(|i| i.token_id == "yes-t").unwrap();
        let no = bids.iter().find(|i| i.token_id == "no-t").unwrap();
        assert!(yes.limit_price.unwrap() > no.limit_price.unwrap());
    }

    /// If the split leaves a leg below min_leg_price, nothing is quoted
    /// — a half box is not a box.
    #[test]
    fn pm_skip_if_leg_below_min_leg_price() {
        let cfg = ParityMakerConfig {
            min_leg_price: dec!(0.12),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = {
            let bids = |p: &str| HalfBook {
                levels: vec![Level {
                    price: p.parse().unwrap(),
                    size: dec!(10),
                }],
            };
            let asks = |p: &str| HalfBook {
                levels: vec![Level {
                    price: p.parse().unwrap(),
                    size: dec!(10),
                }],
            };
            BookSnapshot {
                // asks 0.12 + 0.90 = 1.02 → bid_yes = 0.12/1.02*0.95 = 0.11 < 0.12
                yes: TokenBook {
                    token_id: "y".into(),
                    bids: bids("0.10"),
                    asks: asks("0.12"),
                    ..Default::default()
                },
                no: TokenBook {
                    token_id: "n".into(),
                    bids: bids("0.88"),
                    asks: asks("0.90"),
                    ..Default::default()
                },
                ..make_book(300)
            }
        };
        assert_eq!(
            pm.on_book(&book, &make_vol())
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0
        );
    }

    /// With only one leg filled for longer than single_leg_unwind_secs, PM cancels
    /// the GTDs and sells the excess instead of staying directional to settlement.
    /// Without FV available (make_vol has no sigmas) the decision takes the
    /// dump_no_fv branch of decide_unwind — same observable result as before.
    #[test]
    fn pm_single_leg_is_liquidated_after_timeout() {
        let cfg = ParityMakerConfig {
            single_leg_unwind_secs: 1,
            min_unwind_bid: dec!(0.01),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(10),
            }],
        };
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(10),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "pm-orphan".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids("0.50"),
                asks: asks("0.52"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids("0.40"),
                asks: asks("0.43"),
                ..Default::default()
            },
            clear_at: crate::time::now().timestamp() + 300,
            slug: "btc-updown-5m-pmo".into(),
            ..BookSnapshot::default()
        };

        // Posts both legs; only YES fills.
        let _ = pm.on_book(&book, &make_vol());
        pm.on_fill(&crate::types::Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-orphan".into(),
            token_id: "y".into(),
            side: FillSide::Buy,
            price: dec!(0.495),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });

        // Tick that starts the imbalance clock; still within the grace period.
        let i = pm.on_book(&book, &make_vol());
        assert!(
            i.iter().all(|x| !matches!(x.side, OrderSide::Sell { .. })),
            "within the grace period it must not sell"
        );

        std::thread::sleep(std::time::Duration::from_millis(1100));
        let i = pm.on_book(&book, &make_vol());
        let sells: Vec<_> = i
            .iter()
            .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
            .collect();
        let cancels = i
            .iter()
            .filter(|x| matches!(x.side, OrderSide::Cancel))
            .count();
        assert_eq!(
            sells.len(),
            1,
            "after the grace period it must sell the excess (dump_no_fv without FV available)"
        );
        assert_eq!(sells[0].token_id, "y");
        assert_eq!(sells[0].size, dec!(3));
        assert!(
            !sells[0].resting,
            "without FV available it must sell taker, not maker"
        );
        assert_eq!(cancels, 2, "must cancel both GTDs before selling");
    }

    /// If the retained leg's bid is below `min_unwind_bid`
    /// (toxic zone), the DumpTaker branch does not set `unwind_sent` and retries on
    /// the next tick — but the `cancel_intent`s of both legs must NOT
    /// be re-emitted indefinitely once there are no live GTDs left to
    /// cancel (they used to be unconditional on every tick, 2 `Cancel`s per tick
    /// forever).
    #[test]
    fn pm_single_leg_toxic_zone_no_cancel_loop() {
        let cfg = ParityMakerConfig {
            single_leg_unwind_secs: 1,
            min_unwind_bid: dec!(0.90), // above the real bid (0.50) → toxic zone
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(10),
            }],
        };
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(10),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "pm-toxic".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids("0.50"),
                asks: asks("0.52"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids("0.40"),
                asks: asks("0.43"),
                ..Default::default()
            },
            clear_at: crate::time::now().timestamp() + 300,
            slug: "btc-updown-5m-pmt".into(),
            ..BookSnapshot::default()
        };

        let _ = pm.on_book(&book, &make_vol());
        pm.on_fill(&crate::types::Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-toxic".into(),
            token_id: "y".into(),
            side: FillSide::Buy,
            price: dec!(0.495),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });

        // Tick that starts the imbalance clock (unbalanced_since); still
        // within the grace period of single_leg_unwind_secs, it must not cancel.
        let i0 = pm.on_book(&book, &make_vol());
        assert!(
            i0.iter().all(|x| !matches!(x.side, OrderSide::Cancel)),
            "within the grace period it must not cancel yet"
        );

        std::thread::sleep(std::time::Duration::from_millis(1100));
        // First tick after the timeout: cancels the live GTDs (2 cancels), without
        // a valid bid (toxic zone) it does not sell.
        let i1 = pm.on_book(&book, &make_vol());
        let cancels1 = i1
            .iter()
            .filter(|x| matches!(x.side, OrderSide::Cancel))
            .count();
        assert_eq!(
            cancels1, 2,
            "first tick after the timeout must cancel the live GTDs"
        );
        assert!(
            i1.iter().all(|x| !matches!(x.side, OrderSide::Sell { .. })),
            "without a valid bid it must not sell"
        );

        // Following ticks: with nothing live to cancel, it must NOT re-emit cancels in a loop.
        for _ in 0..10 {
            let i = pm.on_book(&book, &make_vol());
            assert!(
                i.iter().all(|x| !matches!(x.side, OrderSide::Cancel)),
                "without live GTDs it must not re-emit cancels indefinitely"
            );
        }
    }

    #[test]
    fn pm_size_zero_uses_usdc_cap() {
        // order_size_shares=0 → fallback to max_order_usdc / bid
        // bid_yes ≈ 0.495, max_order_usdc=10 → size ≈ 20.2 shares
        use crate::config::StrategyLimits;
        let cfg = ParityMakerConfig {
            margin: dec!(0.05),
            min_leg_price: dec!(0.05),
            order_size_shares: dec!(0),
            requote_threshold_bps: 20,
            pair_ttl_secs: 10,
            limits: StrategyLimits {
                max_order_usdc: dec!(10),
                ..StrategyLimits::default()
            },
            ..ParityMakerConfig::default()
        };
        let mut pm = ParityMaker::new(cfg);
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "pm-zero-cid".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids("0.495"),
                asks: asks("0.52"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids("0.40"),
                asks: asks("0.43"),
                ..Default::default()
            },
            clear_at: crate::time::now().timestamp() + 300,
            ..BookSnapshot::default()
        };
        let intents = pm.on_book(&book, &make_vol());
        let buy_intents: Vec<_> = intents
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(buy_intents.len(), 2, "must post both legs");
        assert!(buy_intents.iter().all(|i| i.size > Decimal::ZERO));
    }

    /// With spot≫strike (high FV, favors YES) the skew raises bid_yes and lowers
    /// bid_no compared with the symmetric scenario (spot=strike, FV=0.5, no skew), and the
    /// parity constraint holds (the sum does not change).
    #[test]
    fn pm_skew_favors_high_fv_side() {
        let cfg = ParityMakerConfig {
            fv_skew_threshold: dec!(0.05),
            fv_skew_bps: 100,               // ±0.01
            fv_no_quote_threshold: dec!(0), // isolate the effect of the skew
            ..test_cfg()
        };
        let max_sum = dec!(1) - cfg.margin;

        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "pm-skew-cid".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids("0.50"),
                asks: asks("0.55"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids("0.40"),
                asks: asks("0.45"),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: crate::time::now().timestamp() + 300,
            neg_risk: false,
            slug: "btc-updown-5m-skew".into(),
            reference_price: Some(dec!(100000)),
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        };

        // Symmetric case: spot = strike → FV = 0.5 exactly → no skew.
        let mut pm_sym = ParityMaker::new(cfg.clone());
        let vol_sym = make_vol_fv("100000", "0.0005");
        let intents_sym = pm_sym.on_book(&book, &vol_sym);
        let bid_yes_sym = intents_sym
            .iter()
            .find(|i| i.token_id == "y")
            .unwrap()
            .price;
        let bid_no_sym = intents_sym
            .iter()
            .find(|i| i.token_id == "n")
            .unwrap()
            .price;

        // Skewed case: spot >> strike → high FV → favors YES.
        let mut pm_skew = ParityMaker::new(cfg);
        let vol_skew = make_vol_fv("110000", "0.0005");
        let intents_skew = pm_skew.on_book(&book, &vol_skew);
        let bid_yes_skew = intents_skew
            .iter()
            .find(|i| i.token_id == "y")
            .unwrap()
            .price;
        let bid_no_skew = intents_skew
            .iter()
            .find(|i| i.token_id == "n")
            .unwrap()
            .price;

        assert!(
            bid_yes_skew > bid_yes_sym,
            "bid_yes must rise with high FV: {bid_yes_skew} vs {bid_yes_sym}"
        );
        assert!(
            bid_no_skew < bid_no_sym,
            "bid_no must fall with high FV: {bid_no_skew} vs {bid_no_sym}"
        );
        assert!(
            bid_yes_skew + bid_no_skew <= max_sum,
            "parity must be preserved after the skew"
        );
    }

    /// With |FV-0.5| >= fv_no_quote_threshold, PM quotes only the leg
    /// favored (a single leg) — it does not risk the disfavored one.
    #[test]
    fn pm_does_not_quote_disfavored_side_if_extreme_fv() {
        let cfg = ParityMakerConfig {
            fv_no_quote_threshold: dec!(0.05),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "pm-noquote-cid".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids("0.50"),
                asks: asks("0.55"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids("0.40"),
                asks: asks("0.45"),
                ..Default::default()
            },
            clear_at: crate::time::now().timestamp() + 300,
            reference_price: Some(dec!(100000)),
            slug: "btc-updown-5m-nq".into(),
            ..BookSnapshot::default()
        };
        let vol = make_vol_fv("110000", "0.0005"); // spot >> strike → extreme FV, favors YES

        let intents = pm.on_book(&book, &vol);
        let buys: Vec<_> = intents
            .iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(buys.len(), 1, "must quote only the favored leg");
        assert_eq!(buys[0].token_id, "y");
    }

    /// With FV_ret >= unwind_hold_fv when the grace period runs out, PM keeps the
    /// position to settlement (no taker sell) and only cancels the opposite GTD.
    #[test]
    fn pm_unwind_hold_if_fv_favors_retained_leg() {
        let cfg = ParityMakerConfig {
            single_leg_unwind_secs: 1,
            min_unwind_bid: dec!(0.01),
            unwind_hold_fv: dec!(0.60),
            unwind_dump_fv: dec!(0.40),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "pm-hold-cid".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids("0.50"),
                asks: asks("0.55"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids("0.40"),
                asks: asks("0.45"),
                ..Default::default()
            },
            clear_at: crate::time::now().timestamp() + 300,
            reference_price: Some(dec!(100000)),
            slug: "btc-updown-5m-hold".into(),
            ..BookSnapshot::default()
        };
        let vol = make_vol_fv("110000", "0.0005"); // FV_ret (YES) ≈ 1.0 >= 0.60

        // Posts; only YES fills.
        let _ = pm.on_book(&book, &vol);
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-hold-cid".into(),
            token_id: "y".into(),
            side: FillSide::Buy,
            price: dec!(0.53),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });

        // Tick that starts the imbalance clock (unbalanced_since).
        let _ = pm.on_book(&book, &vol);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let i = pm.on_book(&book, &vol);
        let sells = i
            .iter()
            .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
            .count();
        let cancels = i
            .iter()
            .filter(|x| matches!(x.side, OrderSide::Cancel))
            .count();
        assert_eq!(sells, 0, "FV favors the retained leg → hold, do not sell");
        assert_eq!(
            cancels, 2,
            "must cancel the opposite GTD (and its own, which is no longer re-quoted)"
        );
    }

    /// In the neutral zone (FV_ret between dump and hold) with time to spare,
    /// PM re-posts a GTD (maker) ask instead of selling taker.
    #[test]
    fn pm_unwind_repost_maker_if_fv_neutral() {
        let cfg = ParityMakerConfig {
            single_leg_unwind_secs: 1,
            min_unwind_bid: dec!(0.01),
            unwind_hold_fv: dec!(0.60),
            unwind_dump_fv: dec!(0.40),
            unwind_repost_min_secs: 60,
            unwind_maker_edge: dec!(0.01),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "pm-repost-cid".into(),
            yes: TokenBook {
                token_id: "y".into(),
                bids: bids("0.50"),
                asks: asks("0.52"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "n".into(),
                bids: bids("0.40"),
                asks: asks("0.43"),
                ..Default::default()
            },
            clear_at: crate::time::now().timestamp() + 300, // remaining > 60
            reference_price: Some(dec!(100000)),
            slug: "btc-updown-5m-repost".into(),
            ..BookSnapshot::default()
        };
        let vol = make_vol_fv("100000", "0.0005"); // spot = strike → FV_ret = exactly 0.5 (neutral)

        // Posts; only YES fills at 0.50.
        let _ = pm.on_book(&book, &vol);
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-repost-cid".into(),
            token_id: "y".into(),
            side: FillSide::Buy,
            price: dec!(0.50),
            size: dec!(3),
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });

        // Tick that starts the imbalance clock (unbalanced_since).
        let _ = pm.on_book(&book, &vol);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let i = pm.on_book(&book, &vol);
        let sells: Vec<_> = i
            .iter()
            .filter(|x| matches!(x.side, OrderSide::Sell { .. }))
            .collect();
        assert_eq!(
            sells.len(),
            1,
            "neutral zone with time to spare must re-post, not oversell"
        );
        assert_eq!(sells[0].token_id, "y");
        assert!(sells[0].resting, "the repost must be maker (GTD), not taker");
        assert!(
            sells[0].price >= dec!(0.50),
            "the price must cover at least the avg_entry"
        );
    }

    /// With σ_5s/σ_60s above the threshold, PM pauses
    /// (emits no new quotes) until the volatility spike subsides.
    #[test]
    fn pm_pause_on_vol_spike() {
        let cfg = ParityMakerConfig {
            vol_ratio_threshold: dec!(3.0),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = VolSnapshot {
            asset: "btc".into(),
            sigma_short: Some(dec!(0.003)),
            sigma_long: Some(dec!(0.0005)), // ratio = 6 > 3
            last_price: dec!(100000),
            updated_at: crate::time::Instant::now(),
            ..VolSnapshot::default()
        };
        let intents = pm.on_book(&book, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0,
            "vol spike must pause without quoting"
        );
    }

    /// With a 0.001 market tick, the bids keep the third decimal
    /// instead of being wrecked to 2 (0.029 → 0.02 with the old truncation).
    #[test]
    fn pm_truncates_to_market_tick() {
        let cfg = ParityMakerConfig {
            min_leg_price: dec!(0.01),
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let mut book = make_book(300);
        book.tick_size = Some(dec!(0.001));
        // Thin zone: asks 0.975 + 0.031. Split of 0.95:
        //   bid_no = 0.031/1.006×0.95 ≈ 0.0292 → 0.029 with tick 0.001 (0.02 with the bug)
        book.yes.asks = asks("0.975");
        book.no.asks = asks("0.031");
        let bids: Vec<_> = pm
            .on_book(&book, &make_vol())
            .into_iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(bids.len(), 2, "must quote both legs in a thin zone");
        let bid_no = bids
            .iter()
            .find(|i| i.token_id == "no-t")
            .unwrap()
            .limit_price
            .unwrap();
        assert_eq!(
            bid_no,
            dec!(0.029),
            "the third decimal must survive the truncation"
        );
    }

    /// If the book gives away an arb (asks summing to < 1 − margin), the split
    /// would produce bids above the ask — the GTD would cross and
    /// execute as taker paying the fee. The clamp keeps them at top-of-book.
    #[test]
    fn pm_clamp_does_not_cross_book() {
        let mut pm = ParityMaker::new(test_cfg()); // margin 0.05 → budget 0.95
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let mut book = make_book(300);
        book.yes.asks = asks("0.45");
        book.no.asks = asks("0.40"); // sum 0.85 < 0.95: without the clamp, the bids cross
        let bids: Vec<_> = pm
            .on_book(&book, &make_vol())
            .into_iter()
            .filter(|i| matches!(i.side, OrderSide::Buy))
            .collect();
        assert_eq!(bids.len(), 2);
        for b in &bids {
            let ask = if b.token_id == "yes-t" {
                dec!(0.45)
            } else {
                dec!(0.40)
            };
            let px = b.limit_price.unwrap();
            assert!(
                px <= ask - dec!(0.01),
                "bid {px} crosses the ask {ask}: it would execute as taker"
            );
        }
    }

    /// An early return (ask below min_leg_price) must not leave the GTDs
    /// live and hanging — it must cancel them, as the entry barrier does.
    #[test]
    fn pm_early_return_cancels_live_gtds() {
        let mut pm = ParityMaker::new(test_cfg()); // min_leg_price 0.05
        let book = make_book(300);
        let posted = pm.on_book(&book, &make_vol());
        assert_eq!(
            posted
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2
        );

        // The YES ask collapses below min_leg_price: PM stops quoting
        // the bracket, but its live bids must come off the book NOW.
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        let mut crashed_book = make_book(300);
        crashed_book.yes.asks = asks("0.04");
        let intents = pm.on_book(&crashed_book, &make_vol());
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "must cancel both GTDs when leaving the bracket"
        );
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            0
        );
    }

    // ── Bracket history ───────────────────────────────────────────────

    /// Inserts an already expired `PMState` directly (the test submodule can see the
    /// private fields of `PMState`/`ParityMaker` because it is a descendant of the module that declares them)
    /// together with its last probability
    /// in `bracket_probs`, forces the 30s sweep to be due and
    /// fires ONE call to `on_book` — the sweep runs before anything else,
    /// so any book/vol is enough to trigger it.
    #[allow(clippy::too_many_arguments)]
    fn push_expired(
        pm: &mut ParityMaker,
        cid: &str,
        asset: &str,
        duration: i64,
        box_counted: bool,
        ever_yes: bool,
        ever_no: bool,
        final_prob: Option<Decimal>,
    ) {
        let now = crate::time::now().timestamp();
        pm.markets.insert(
            cid.to_string(),
            PMState {
                clear_at: now - 1, // already expired
                posted_at: None,
                posted_mid: None,
                yes_token: format!("{cid}-yes"),
                no_token: format!("{cid}-no"),
                yes_filled: Decimal::ZERO,
                no_filled: Decimal::ZERO,
                unbalanced_since: None,
                unwind_sent: false,
                reference: None,
                posted_fv: None,
                posted_prob: None,
                posted_bid_yes: None,
                posted_bid_no: None,
                yes_cost: Decimal::ZERO,
                no_cost: Decimal::ZERO,
                box_counted,
                dump_count: 0,
                yes_live: Decimal::ZERO,
                no_live: Decimal::ZERO,
                asset: asset.to_string(),
                duration,
                ever_yes,
                ever_no,
            },
        );
        if let Some(p) = final_prob {
            pm.bracket_probs
                .entry(asset.to_string())
                .or_default()
                .entry(cid.to_string())
                .or_default()
                .push_back((Instant::now(), p));
        }
        // Force the 30s sweep gate to be already due.
        pm.last_retain = Instant::now() - Duration::from_secs(31);
        pm.on_book(&make_book(3000), &make_vol());
    }

    #[test]
    fn history_records_boxed_and_orphan_on_expiry() {
        let cfg = ParityMakerConfig {
            bracket_history_len: 20,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);

        // Full box: both legs touched and counted as boxed.
        push_expired(
            &mut pm,
            "cid-a",
            "btc",
            300,
            true,
            true,
            true,
            Some(dec!(0.9)),
        );
        // Orphan: only one leg was touched, never boxed.
        push_expired(
            &mut pm,
            "cid-b",
            "btc",
            300,
            false,
            true,
            false,
            Some(dec!(0.2)),
        );

        let hist = pm
            .history
            .get(&("btc".to_string(), 300))
            .expect("a history for (btc, 300) must exist");
        assert_eq!(hist.len(), 2);
        assert!(
            hist.iter()
                .any(|r| r.boxed && r.touched && r.final_prob == Some(dec!(0.9))),
            "the record of the full box is missing"
        );
        assert!(
            hist.iter()
                .any(|r| !r.boxed && r.touched && r.final_prob == Some(dec!(0.2))),
            "the record of the orphan is missing"
        );

        // The order matters: the expired brackets left `markets`, but
        // only AFTER being recorded (if the order were reversed, the
        // earlier block step would have failed: there would be nothing to record.
        assert!(!pm.markets.contains_key("cid-a"));
        assert!(!pm.markets.contains_key("cid-b"));
    }

    #[test]
    fn history_respects_max_length() {
        let cfg = ParityMakerConfig {
            bracket_history_len: 3,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);

        for (i, p) in [dec!(0.1), dec!(0.2), dec!(0.3), dec!(0.4), dec!(0.5)]
            .into_iter()
            .enumerate()
        {
            push_expired(
                &mut pm,
                &format!("cid-{i}"),
                "btc",
                300,
                true,
                true,
                true,
                Some(p),
            );
        }

        let hist = pm
            .history
            .get(&("btc".to_string(), 300))
            .expect("a history must exist");
        assert_eq!(hist.len(), 3, "must truncate to bracket_history_len");
        let probs: Vec<Decimal> = hist.iter().map(|r| r.final_prob.unwrap()).collect();
        // FIFO: the 3 LAST inserted are kept, in arrival order.
        assert_eq!(probs, vec![dec!(0.3), dec!(0.4), dec!(0.5)]);
    }

    #[test]
    fn history_directional_streak_with_final_prob() {
        // UP streak: the 3 most recent (0.9, 0.8, 0.7) are above 0.5;
        // the oldest (0.3) is DOWN and cuts the streak there.
        let mut up = VecDeque::new();
        up.push_back(BracketRecord {
            boxed: false,
            touched: true,
            final_prob: Some(dec!(0.3)),
        });
        up.push_back(BracketRecord {
            boxed: true,
            touched: true,
            final_prob: Some(dec!(0.9)),
        });
        up.push_back(BracketRecord {
            boxed: true,
            touched: true,
            final_prob: Some(dec!(0.8)),
        });
        up.push_back(BracketRecord {
            boxed: false,
            touched: true,
            final_prob: Some(dec!(0.7)),
        });
        assert_eq!(hist_streak(&up), 3, "UP streak must be positive");

        // DOWN streak: negative sign.
        let mut down = VecDeque::new();
        down.push_back(BracketRecord {
            boxed: false,
            touched: true,
            final_prob: Some(dec!(0.9)),
        });
        down.push_back(BracketRecord {
            boxed: false,
            touched: true,
            final_prob: Some(dec!(0.3)),
        });
        down.push_back(BracketRecord {
            boxed: false,
            touched: true,
            final_prob: Some(dec!(0.2)),
        });
        assert_eq!(hist_streak(&down), -2, "DOWN streak must be negative");

        // A record without final_prob cuts the streak right when it is reached — only
        // the most recent one counts (0.8, UP).
        let mut cut = VecDeque::new();
        cut.push_back(BracketRecord {
            boxed: false,
            touched: true,
            final_prob: Some(dec!(0.9)),
        });
        cut.push_back(BracketRecord {
            boxed: false,
            touched: false,
            final_prob: None,
        });
        cut.push_back(BracketRecord {
            boxed: true,
            touched: true,
            final_prob: Some(dec!(0.8)),
        });
        assert_eq!(
            hist_streak(&cut),
            1,
            "the record without final_prob must cut the streak"
        );

        // Empty deque → no streak.
        assert_eq!(hist_streak(&VecDeque::new()), 0);
    }

    #[test]
    fn history_disabled_with_len_zero() {
        let cfg = ParityMakerConfig {
            bracket_history_len: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);

        push_expired(
            &mut pm,
            "cid-a",
            "btc",
            300,
            true,
            true,
            true,
            Some(dec!(0.9)),
        );

        assert!(
            pm.history.is_empty(),
            "bracket_history_len=0 must not accumulate anything"
        );
    }

    #[test]
    fn history_separates_by_asset_and_duration() {
        let cfg = ParityMakerConfig {
            bracket_history_len: 20,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);

        push_expired(
            &mut pm,
            "cid-btc-300",
            "btc",
            300,
            true,
            true,
            true,
            Some(dec!(0.9)),
        );
        push_expired(
            &mut pm,
            "cid-eth-300",
            "eth",
            300,
            false,
            true,
            false,
            Some(dec!(0.1)),
        );
        push_expired(
            &mut pm,
            "cid-btc-900",
            "btc",
            900,
            false,
            true,
            false,
            Some(dec!(0.2)),
        );

        assert_eq!(
            pm.history.get(&("btc".to_string(), 300)).map(VecDeque::len),
            Some(1)
        );
        assert_eq!(
            pm.history.get(&("eth".to_string(), 300)).map(VecDeque::len),
            Some(1)
        );
        assert_eq!(
            pm.history.get(&("btc".to_string(), 900)).map(VecDeque::len),
            Some(1)
        );
        assert!(
            !pm.history.contains_key(&("eth".to_string(), 900)),
            "must not mix buckets of different (asset, duration)"
        );
    }

    /// Bugfix: with `market_open_ts == 0` (field not yet populated) the bracket
    /// duration must stay at the invalid mark `0` — previously it was computed
    /// as `clear_at - 0 = clear_at` (~1.7e9), which put every bracket in a single
    /// history bucket (it never accumulated) and left a new key with no per-bracket
    /// pruning (memory leak). It verifies the real computation in
    /// `on_book` (not the `push_expired` shortcut, which sets `duration` by hand) and
    /// that on expiry nothing is recorded in the history.
    #[test]
    fn invalid_duration_with_market_open_ts_zero_does_not_contaminate_history() {
        let cfg = ParityMakerConfig {
            bracket_history_len: 20,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);

        // make_book() already sets market_open_ts: 0 (real scenario: field not populated).
        let book = make_book(300);
        pm.on_book(&book, &make_vol());
        let state = pm
            .markets
            .get(&book.market_condition_id)
            .expect("the state must exist");
        assert_eq!(
            state.duration, 0,
            "market_open_ts=0 must leave duration at 0 (invalid mark)"
        );

        // Force the bracket to expire and trigger the 30s sweep.
        pm.markets
            .get_mut(&book.market_condition_id)
            .unwrap()
            .clear_at = crate::time::now().timestamp() - 1;
        pm.last_retain = Instant::now() - Duration::from_secs(31);
        // different cid in the tick's book so as not to recreate the same state that
        // the sweep just swept (the sweep operates over `self.markets`, not
        // over the book argument — see block at the start of `on_book`).
        let mut other_book = make_book(300);
        other_book.market_condition_id = "pm-cid-other".into();
        pm.on_book(&other_book, &make_vol());

        assert!(
            !pm.markets.contains_key(&book.market_condition_id),
            "the expired bracket must leave markets"
        );
        assert!(
            pm.history.is_empty(),
            "duration=0 (invalid mark) must not generate entries in the history"
        );
    }

    /// Bugfix (counterpart of the test above): with a real `market_open_ts > 0` the
    /// duration is computed as before (`clear_at - market_open_ts`) and the
    /// bracket is still recorded in the history on expiry.
    #[test]
    fn valid_duration_with_positive_market_open_ts_is_recorded() {
        let cfg = ParityMakerConfig {
            bracket_history_len: 20,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);

        let mut book = make_book(300);
        let now = crate::time::now().timestamp();
        book.market_open_ts = now - 300; // bracket "opened" 300s ago → nominal duration 600s

        pm.on_book(&book, &make_vol());
        let state = pm
            .markets
            .get(&book.market_condition_id)
            .expect("the state must exist");
        assert_eq!(
            state.duration, 600,
            "duration must be clear_at - market_open_ts with market_open_ts>0"
        );

        pm.markets
            .get_mut(&book.market_condition_id)
            .unwrap()
            .clear_at = crate::time::now().timestamp() - 1;
        pm.last_retain = Instant::now() - Duration::from_secs(31);
        let mut other_book = make_book(300);
        other_book.market_condition_id = "pm-cid-other2".into();
        pm.on_book(&other_book, &make_vol());

        let hist = pm
            .history
            .get(&("btc".to_string(), 600))
            .expect("a valid duration must be recorded in the history on expiry");
        assert_eq!(hist.len(), 1);
    }

    // ── Requote tolerance (order refresh tolerance) ─────────────────────

    /// With `pair_ttl_secs = 0` the requote fires on every tick; if the desired bids
    /// are ≤ `requote_tolerance_ticks` away from the resting ones, PM does NOT cancel
    /// nor re-post — the pair stays as it is.
    #[test]
    fn pm_tolerance_skips_requote_within_ticks() {
        let cfg = ParityMakerConfig {
            requote_tolerance_ticks: 1,
            pair_ttl_secs: 0, // requote fired on every tick (pattern of pm_does_not_open_without_path)
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = make_vol();

        let first = pm.on_book(&book, &vol);
        assert_eq!(
            first
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "first quote: two resting bids"
        );

        // Same book: the desired bids are identical to the resting ones (0 ticks) →
        // pair kept: no cancels nor re-posts even though the requote fired.
        let intents = pm.on_book(&book, &vol);
        assert!(
            intents.is_empty(),
            "within tolerance it must neither cancel nor re-post: {intents:?}"
        );
    }

    /// Control: with tolerance 0 (default) the same scenario produces the usual churn
    /// — it cancels both legs and only re-posts when the acks come back.
    #[test]
    fn pm_without_tolerance_cancels_and_reposts() {
        let cfg = ParityMakerConfig {
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = make_vol();
        let _ = pm.on_book(&book, &vol);

        // Second tick: cancels on both legs; no re-post because the cap
        // is still consumed by the live GTDs until the ack arrives.
        let intents = pm.on_book(&book, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "without tolerance the requote cancels both legs even though the price did not move"
        );

        ack_cancels(&mut pm, &book);
        let intents = pm.on_book(&book, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "after the acks it re-posts — the cancel/ack/post cycle completes"
        );
    }

    /// Outside tolerance (> N ticks per leg) the requote cancels and re-posts
    /// normally, as before.
    #[test]
    fn pm_tolerance_exceeded_reposts() {
        let cfg = ParityMakerConfig {
            requote_tolerance_ticks: 1,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = make_vol();
        let _ = pm.on_book(&book, &vol);

        // yes_ask 0.55→0.60: desired bid_yes goes from 0.51 to 0.53 and bid_no from
        // 0.43 to 0.41 — both legs 2 ticks from the resting ones (> tolerance of 1).
        let mut book2 = make_book(300);
        book2.yes.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.60),
                size: dec!(100),
            }],
        };
        let intents = pm.on_book(&book2, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "outside tolerance both legs are cancelled"
        );

        ack_cancels(&mut pm, &book2);
        let intents = pm.on_book(&book2, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Buy))
                .count(),
            2,
            "and they are re-posted at the new prices"
        );
    }

    /// Tolerance NEVER keeps an invalid pair: keeping YES at its resting price
    /// (0.51) while NO is re-posted at 0.45 would leave the sum at 0.96 >
    /// 1 − margin (0.95) → the kept leg is cancelled anyway.
    #[test]
    fn pm_tolerance_does_not_keep_invalid_pair() {
        let cfg = ParityMakerConfig {
            requote_tolerance_ticks: 2,
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = make_vol();
        let posted = pm.on_book(&book, &vol);
        let bid_yes = posted
            .iter()
            .find(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "yes-t")
            .expect("must post the YES bid")
            .price;
        assert_eq!(bid_yes, dec!(0.51));

        // The NO GTD dies (terminal ack from the executor) — it no longer has
        // a resting order to keep, and it is re-posted at the new price.
        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-cid".into(),
            token_id: "no-t".into(),
            side: FillSide::Cancelled,
            price: dec!(0),
            size: dec!(0),
            order_id: Some("oid".into()),
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });

        // no_ask 0.47→0.50: desired bid_yes 0.49 (2 ticks from the resting 0.51,
        // within tolerance → tentative keep) and desired bid_no 0.45. But
        // resting_yes + new bid_no = 0.51 + 0.45 = 0.96 > 0.95: invalid pair.
        let mut book2 = make_book(300);
        book2.no.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.50),
                size: dec!(100),
            }],
        };
        let intents = pm.on_book(&book2, &vol);

        assert!(
            intents
                .iter()
                .any(|i| matches!(i.side, OrderSide::Cancel) && i.token_id == "yes-t"),
            "the resting YES order cannot stay: with the new NO bid the sum violates 1-margin: {intents:?}"
        );
        let buy_no = intents
            .iter()
            .find(|i| matches!(i.side, OrderSide::Buy) && i.token_id == "no-t")
            .expect("NO is re-posted at the new price");
        assert_eq!(buy_no.price, dec!(0.45));
        assert!(
            intents
                .iter()
                .all(|i| !(matches!(i.side, OrderSide::Buy) && i.token_id == "yes-t")),
            "YES is not re-posted (cap consumed by the resting order): it is only cancelled"
        );
    }

    /// Dangerous paths stay intact: an ask below min_leg_price always cancels the
    /// live GTDs — tolerance only applies to the requote path.
    #[test]
    fn pm_tolerance_does_not_hold_back_dangerous_cancel() {
        let cfg = ParityMakerConfig {
            requote_tolerance_ticks: 5, // absurd slack: even this must save nothing
            pair_ttl_secs: 0,
            ..test_cfg()
        };
        let mut pm = ParityMaker::new(cfg);
        let book = make_book(300);
        let vol = make_vol();
        let _ = pm.on_book(&book, &vol);

        let mut crashed = make_book(300);
        crashed.yes.asks = HalfBook {
            levels: vec![Level {
                price: dec!(0.04), // < min_leg_price (0.05)
                size: dec!(100),
            }],
        };
        let intents = pm.on_book(&crashed, &vol);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.side, OrderSide::Cancel))
                .count(),
            2,
            "the dangerous early return cancels both legs, tolerance or not"
        );
    }

    /// A `GateRejected` undoes ONLY its intent's contribution:
    /// `yes_live` drops by `fill.size` and the resting mark is released because
    /// it points at the blocked intent's price. The other leg is not touched.
    #[test]
    fn gate_rejected_decrements_only_its_intent() {
        let mut pm = ParityMaker::new(test_cfg());
        let book = make_book(300);
        let intents = pm.on_book(&book, &make_vol());
        let yes = intents
            .iter()
            .find(|i| i.token_id == "yes-t" && matches!(i.side, OrderSide::Buy))
            .expect("must post a YES bid");
        let (px, sz) = (yes.price, yes.size);

        {
            let st = pm.markets.get("pm-cid").unwrap();
            assert_eq!(st.yes_live, sz);
            assert_eq!(st.posted_bid_yes, Some(px));
            assert!(st.no_live > Decimal::ZERO);
        }

        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-cid".into(),
            token_id: "yes-t".into(),
            side: FillSide::GateRejected,
            price: px,
            size: sz,
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });

        let st = pm.markets.get("pm-cid").unwrap();
        assert_eq!(
            st.yes_live,
            Decimal::ZERO,
            "the intent's live size dies with it"
        );
        assert_eq!(st.posted_bid_yes, None, "its price's mark is released");
        assert!(st.no_live > Decimal::ZERO, "the NO leg is not touched");
        assert!(st.posted_bid_no.is_some(), "the NO mark is not touched");
    }

    /// `GateRejected` NEVER resets the leg to zero: if `live` remains from another
    /// real order only the blocked intent's size is subtracted; and if
    /// the resting mark already points at a NEW bid (later requote), it is not
    /// deleted: it may belong to a potentially live order.
    #[test]
    fn gate_rejected_does_not_reset_or_delete_foreign_mark() {
        let mut pm = ParityMaker::new(test_cfg());
        let book = make_book(300);
        let intents = pm.on_book(&book, &make_vol());
        let yes = intents
            .iter()
            .find(|i| i.token_id == "yes-t" && matches!(i.side, OrderSide::Buy))
            .unwrap();
        let (old_px, sz) = (yes.price, yes.size);
        let new_px = old_px + dec!(0.01);

        // State as if ANOTHER real GTD were live (live = sz of the intent +
        // 3 from the real order) and a later re-post moved the mark.
        {
            let st = pm.markets.get_mut("pm-cid").unwrap();
            st.yes_live += dec!(3);
            st.posted_bid_yes = Some(new_px);
        }

        pm.on_fill(&Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "pm-cid".into(),
            token_id: "yes-t".into(),
            side: FillSide::GateRejected,
            price: old_px,
            size: sz,
            order_id: None,
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 0,
            outcome_label: None,
        });

        let st = pm.markets.get("pm-cid").unwrap();
        assert_eq!(
            st.yes_live,
            dec!(3),
            "only the intent's size is subtracted — the real order is still counted"
        );
        assert_eq!(
            st.posted_bid_yes,
            Some(new_px),
            "the mark points to another bid — it belongs to the re-post, it is not deleted"
        );
    }
}
