//! Virtual ledger per market episode.
//!
//! Each market in scope opens ONE episode with its own virtual budget
//! (`initial_episode_budget`). The virtual budget separates the "paper
//! infinite" (automatic top-ups from the sim) from learned ruin: deposits
//! and top-ups NEVER enter this box — they are neutral by construction, not
//! by correction (`net_pnl_delta` no longer needs to subtract them).
//!
//! Conventions of the real ledger replicated here:
//! - Everything in micros (`i64`, `state::to_micros`/`from_micros`) — float only
//!   for the reward normalization.
//! - `cash` may go negative (it is not clamped — historical sim_balance bug).
//! - `Fill.price` is the ALL-INCLUDED price (`cash/shares` in the sim —
//!   fee and impact ALREADY included; see `SimulatedRouter::emit_execution` and
//!   `EconomicFlows`: `usdc = fill.price × fill.size`). The durable ledger
//!   books `cost_basis = usdc` — here it is replicated EXACTLY:
//!   BUY charges `price·size` and SELL credits `price·size`, without estimating or
//!   adding/subtracting the fee a second time (historical bug: double fee per fill).
//! - `fees_paid_um` accumulates the ESTIMATED fee embedded in the all-in price
//!   (`fee_charged_5dp` over the fill aggregate) — audit only,
//!   never moves cash nor feeds back into the reward.
//! - Partial sell splits the cost with the formula `reduce_position_conn`
//!   (truncated, remainder stays in the position) — `cost_sold = cost × sold/size`.
//! - SELL of own inventory only: selling more than what is left
//!   marks `ownership_ambiguous` and excludes the episode from training.

use std::collections::HashMap;

use rust_decimal::Decimal;

use crate::fair_value::fee_charged_5dp;
use crate::state::{from_micros, to_micros};

/// Book level already converted to micros `(price_um, size_um)`.
/// Bids sorted from best to worst (price descending).
pub type BookLevelsUm = Vec<(i64, i64)>;

/// Own position of a token within the episode.
#[derive(Debug, Clone, Default)]
pub struct TokenPosition {
    pub size_um: i64,
    /// All-in residual cost (Σ `fill.price×size` live) — same convention as
    /// `positions.cost_basis_um` (fee and impact included in the price).
    pub cost_um: i64,
}

/// Auditable result of an accounting point.
#[derive(Debug, Clone, Copy)]
pub struct RewardBreakdown {
    /// Equity delta in micros (net of costs; no external flows — the virtual ledger
    /// has no deposits).
    pub net_pnl_delta_um: i64,
    /// Normalized reward carried to the cohort.
    pub reward_norm: f64,
    /// Applied inventory penalty (λ_inventory).
    pub inventory_penalty: f64,
    /// Applied incremental drawdown penalty (λ_drawdown).
    pub drawdown_penalty: f64,
    /// Current equity in micros.
    pub equity_um: i64,
}

#[derive(Debug)]
pub struct EpisodeLedger {
    pub episode_id: String,
    /// Initial virtual budget of the episode (micros).
    pub budget_um: i64,
    /// Free cash (may go negative if a conservative estimate falls
    /// short — the real sim is the one that stops it; this only accounts).
    pub cash_um: i64,
    /// Budget reserved by the in-flight buy (≤1 pending per market).
    pub reserved_um: i64,
    pub positions: HashMap<String, TokenPosition>,
    /// Estimated fees charged (audit — not fed back into the reward).
    pub fees_paid_um: i64,
    /// Realized PnL from sells (audit — not fed back into the reward).
    pub realized_pnl_um: i64,
    /// Maximum observed equity (for incremental drawdown).
    peak_equity_um: i64,
    /// Equity at the last accounting point.
    last_equity_um: i64,
    /// Adjusted equity at the last point (≡ last_equity_um — the virtual ledger
    /// has no external flows to adjust).
    /// Drawdown at the last point.
    last_drawdown_um: i64,
    /// Last accounting timestamp (ms) — used for `elapsed_ms` in the reward.
    last_ts_ms: i64,
    /// `false` if the episode cannot feed cohorts (stale/gap,
    /// ownership ambiguous, truncated, recovered mid-flight…). The episode
    /// KEEPS MANAGING positions — it is only excluded from training.
    pub trainable: bool,
    pub invalid_reason: Option<String>,
    /// Tokens whose settlement is ALREADY applied (idempotency + episode close).
    pub settled_tokens: HashMap<String, bool>,
}

impl EpisodeLedger {
    pub fn new(episode_id: impl Into<String>, budget: Decimal, now_ms: i64) -> Self {
        let budget_um = to_micros(budget);
        Self {
            episode_id: episode_id.into(),
            budget_um,
            cash_um: budget_um,
            reserved_um: 0,
            positions: HashMap::new(),
            fees_paid_um: 0,
            realized_pnl_um: 0,
            peak_equity_um: budget_um,
            last_equity_um: budget_um,
            last_drawdown_um: 0,
            last_ts_ms: now_ms,
            trainable: true,
            invalid_reason: None,
            settled_tokens: HashMap::new(),
        }
    }

    /// Episode recovered after a restart: real ledger positions with
    /// cash rebuilt as `budget − open_cost` (prior history
    /// lost → the episode is managed but NOT trained).
    pub fn recovered(
        episode_id: impl Into<String>,
        budget: Decimal,
        positions: HashMap<String, TokenPosition>,
        now_ms: i64,
    ) -> Self {
        let mut ep = Self::new(episode_id, budget, now_ms);
        let open_cost: i64 = positions.values().map(|p| p.cost_um).sum();
        ep.positions = positions;
        ep.cash_um = ep.budget_um - open_cost;
        ep.trainable = false;
        ep.invalid_reason = Some("recovered_restart".into());
        ep
    }

    pub fn invalidate(&mut self, reason: &str) {
        if self.trainable {
            self.trainable = false;
            self.invalid_reason = Some(reason.to_string());
        }
    }

    /// Own shares of a token.
    pub fn shares_of(&self, token_id: &str) -> Decimal {
        from_micros(self.positions.get(token_id).map(|p| p.size_um).unwrap_or(0))
    }

    /// Residual own cost of a token.
    pub fn cost_of(&self, token_id: &str) -> Decimal {
        from_micros(self.positions.get(token_id).map(|p| p.cost_um).unwrap_or(0))
    }

    /// Reserves the estimated all-in cost of an in-flight BUY.
    /// Returns `false` if the free cash (`cash − reserved`) does not cover it
    /// (nothing is reserved). The reservation does NOT debit cash: `cash_um` stays
    /// the total cash and `reserved_um` the committed part — equity stays
    /// `cash + exit`, never `cash + reserved + exit` (reserving
    /// must not create fictitious equity).
    pub fn reserve_buy(&mut self, all_in_um: i64) -> bool {
        if all_in_um <= 0 || all_in_um > self.cash_um - self.reserved_um {
            return false;
        }
        self.reserved_um += all_in_um;
        true
    }

    /// Confirmed BUY fill: releases the intent's reservation and charges the
    /// REAL cost of the fill — `usdc = price × size`, the all-in amount that
    /// the durable ledger books (`fill.price` already carries fee and impact
    /// included; charging the estimated fee separately would double-count it (former bug,
    /// since fixed). `reserved_um` may be left slightly >0 if the
    /// estimate was lower — it is never negative.
    pub fn apply_buy_fill(
        &mut self,
        token_id: &str,
        price: Decimal,
        size: Decimal,
        reserved_um: i64,
        fee_rate: Decimal,
    ) {
        let usdc_um = to_micros(price * size);
        self.reserved_um = (self.reserved_um - reserved_um).max(0);
        self.cash_um -= usdc_um;
        // Audit: estimate of the fee embedded in the all-in price.
        self.fees_paid_um += to_micros(fee_charged_5dp(fee_rate, price, size));
        let pos = self.positions.entry(token_id.to_string()).or_default();
        pos.size_um += to_micros(size);
        pos.cost_um += usdc_um;
    }

    /// Confirmed SELL fill: reduces the position, splitting the cost
    /// (`cost × sold/size`, truncated — same formula as
    /// `StateStore::reduce_position_conn`) and credits the proceeds
    /// all-in `price × size` (same expression as `apply_fill`
    /// on the durable side: `from_micros(to_micros(fill.price * sold))`).
    /// `false` + `ownership_ambiguous` if more than is held is sold.
    pub fn apply_sell_fill(
        &mut self,
        token_id: &str,
        price: Decimal,
        size: Decimal,
        fee_rate: Decimal,
    ) -> bool {
        let sold_um = to_micros(size);
        let Some(pos) = self.positions.get_mut(token_id) else {
            self.invalidate("sell_without_position");
            return false;
        };
        if sold_um <= 0 || sold_um > pos.size_um {
            self.invalidate("sell_over_inventory");
            return false;
        }
        let allocated_um =
            i64::try_from(i128::from(pos.cost_um) * i128::from(sold_um) / i128::from(pos.size_um))
                .unwrap_or(0);
        let proceeds_um = to_micros(price * size);
        pos.size_um -= sold_um;
        pos.cost_um -= allocated_um;
        if pos.size_um == 0 {
            pos.cost_um = 0;
        }
        self.cash_um += proceeds_um;
        // Audit: estimate of the fee embedded in the all-in price.
        self.fees_paid_um += to_micros(fee_charged_5dp(fee_rate, price, size));
        self.realized_pnl_um += proceeds_um - allocated_um;
        true
    }

    /// Releases a reservation without a fill (rejection/cancel/gate).
    pub fn release_reservation(&mut self, reserved_um: i64) {
        self.reserved_um = (self.reserved_um - reserved_um).max(0);
    }

    /// Total exit value in micros: `Σ exit_value(positions)` —
    /// does NOT include cash or the reservation (that is combined by [`Self::equity_um`]).
    ///
    /// `bids_by_token`: token_id → bid levels `(price_um, size_um)` from best
    /// to worst. Conservative exit value: walks the bid book until
    /// covering the shares, deducting the estimated sell fee per slice;
    /// any remainder that cannot be sold is worth ZERO (neither original cost nor
    /// an optimistic mid). `None` (token without a book) → the position is worth 0 and the caller
    /// must flag the quality.
    pub fn exit_value_um(
        &self,
        bids_by_token: &HashMap<String, BookLevelsUm>,
        fee_rate: Decimal,
    ) -> (i64, bool) {
        let mut total: i64 = 0;
        let mut complete = true;
        for (token, pos) in &self.positions {
            if pos.size_um <= 0 {
                continue;
            }
            match bids_by_token.get(token) {
                Some(levels) if !levels.is_empty() => {
                    total += compute_exit_value_um(pos.size_um, levels, fee_rate);
                }
                _ => {
                    complete = false; // no visible liquidity → worth 0
                }
            }
        }
        (total, complete)
    }

    /// Current equity: `cash + exit value`. `cash_um` is the TOTAL cash
    /// (it includes the reserved part — reserving does not debit cash), so
    /// adding `reserved_um` here would count it twice. `exit_um`
    /// is the result of [`Self::exit_value_um`].
    pub fn equity_um(&self, exit_um: i64) -> i64 {
        self.cash_um + exit_um
    }

    /// Accounting point: computes the normalized reward since the last point
    /// and updates `last_*`. `exit_um` must come from
    /// `exit_value_um` over the fresh book at the current instant.
    ///
    /// `lambda_inventory`/`lambda_drawdown` ≥ 0 (defaults 0 in v1).
    /// `reward = Δequity/budget − λ_inv·(exposure/budget)·(elapsed/300000)
    ///          − λ_dd·max(0, Δdrawdown)/budget`.
    pub fn account(
        &mut self,
        exit_um: i64,
        now_ms: i64,
        lambda_inventory: f64,
        lambda_drawdown: f64,
    ) -> RewardBreakdown {
        let equity = self.equity_um(exit_um);
        let delta_um = equity - self.last_equity_um;
        let elapsed_ms = (now_ms - self.last_ts_ms).max(0) as f64;
        let exposure_um: i64 = self.positions.values().map(|p| p.cost_um).sum();
        let budget = self.budget_um.max(1) as f64;
        let net = delta_um as f64 / budget;
        let inv_pen = lambda_inventory * (exposure_um.max(0) as f64 / budget) * (elapsed_ms / 300_000.0);
        let dd = (self.peak_equity_um - equity).max(0);
        let dd_pen = lambda_drawdown * ((dd - self.last_drawdown_um).max(0) as f64 / budget);
        self.peak_equity_um = self.peak_equity_um.max(equity);
        self.last_equity_um = equity;
        self.last_drawdown_um = dd;
        self.last_ts_ms = now_ms;
        RewardBreakdown {
            net_pnl_delta_um: delta_um,
            reward_norm: net - inv_pen - dd_pen,
            inventory_penalty: inv_pen,
            drawdown_penalty: dd_pen,
            equity_um: equity,
        }
    }

    /// Observed official settlement (via `positions.outcome_won`/`redeemed`
    /// — the authoritative source is the SQLite ledger, not a synthetic fill).
    /// The winning side pays `size` (1.0/share), the losing side 0. Idempotent per token.
    /// Returns `false` if it was already booked or the token is not ours.
    pub fn apply_settlement(&mut self, token_id: &str, won: bool) -> bool {
        if self.settled_tokens.contains_key(token_id) {
            return false;
        }
        let Some(pos) = self.positions.get_mut(token_id) else {
            return false;
        };
        if won {
            self.cash_um += pos.size_um;
            self.realized_pnl_um += pos.size_um - pos.cost_um;
        } else {
            self.realized_pnl_um -= pos.cost_um;
        }
        pos.size_um = 0;
        pos.cost_um = 0;
        self.settled_tokens.insert(token_id.to_string(), won);
        true
    }

    /// Free cash available for new buys (excludes the reserved part).
    pub fn free_cash(&self) -> Decimal {
        from_micros((self.cash_um - self.reserved_um).max(0))
    }

    pub fn reserved(&self) -> Decimal {
        from_micros(self.reserved_um)
    }
}

/// Conservative exit value of `size_um` shares walking `bids`
/// (best→worst, `(price_um, size_um)`): proceeds − estimated parabolic fee
/// per slice, in micros. Shares not covered by the book are worth 0.
fn compute_exit_value_um(size_um: i64, bids: &BookLevelsUm, fee_rate: Decimal) -> i64 {
    let mut remaining = size_um;
    let mut value: i64 = 0;
    for &(price_um, lvl_size_um) in bids {
        if remaining <= 0 {
            break;
        }
        let take = remaining.min(lvl_size_um);
        let price = from_micros(price_um);
        let shares = from_micros(take);
        let gross = to_micros(price * shares);
        let fee = to_micros(fee_charged_5dp(fee_rate, price, shares));
        value += gross - fee;
        remaining -= take;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    const RATE: Decimal = dec!(0.07);

    fn ledger() -> EpisodeLedger {
        EpisodeLedger::new("ep1", dec!(10), 1_000)
    }

    #[test]
    fn buy_fill_charges_usdc_all_in_without_extra_fee() {
        // `fill.price` is already all-in (`cash/shares`): charging the fee
        // separately would double it (former bug). cash −= price×size
        // and cost_basis = usdc — same convention as the durable `apply_fill`.
        let mut ep = ledger();
        assert!(ep.reserve_buy(to_micros(dec!(5.20))));
        ep.apply_buy_fill("yes", dec!(0.5), dec!(10), to_micros(dec!(5.20)), RATE);
        assert_eq!(ep.cash_um, to_micros(dec!(10)) - to_micros(dec!(5)));
        assert_eq!(ep.positions["yes"].size_um, to_micros(dec!(10)));
        assert_eq!(ep.positions["yes"].cost_um, to_micros(dec!(5)));
        // The embedded fee estimate is kept for audit only.
        let fee = to_micros(fee_charged_5dp(RATE, dec!(0.5), dec!(10)));
        assert_eq!(ep.fees_paid_um, fee);
    }

    #[test]
    fn partial_sell_splits_cost() {
        let mut ep = ledger();
        ep.apply_buy_fill("yes", dec!(0.5), dec!(10), 0, RATE);
        assert!(ep.apply_sell_fill("yes", dec!(0.6), dec!(4), RATE));
        // sold 4/10 → allocated cost = 5*0.4 = 2 → 6 shares / 3 USDC remain.
        assert_eq!(ep.positions["yes"].size_um, to_micros(dec!(6)));
        assert_eq!(ep.positions["yes"].cost_um, to_micros(dec!(3)));
    }

    #[test]
    fn sell_over_inventory_invalidates() {
        let mut ep = ledger();
        ep.apply_buy_fill("yes", dec!(0.5), dec!(10), 0, RATE);
        assert!(!ep.apply_sell_fill("yes", dec!(0.5), dec!(20), RATE));
        assert!(!ep.trainable);
        assert_eq!(ep.invalid_reason.as_deref(), Some("sell_over_inventory"));
    }

    #[test]
    fn winning_settlement_credits_once() {
        let mut ep = ledger();
        ep.apply_buy_fill("yes", dec!(0.5), dec!(10), 0, RATE);
        let cash_before = ep.cash_um;
        assert!(ep.apply_settlement("yes", true));
        assert_eq!(ep.cash_um, cash_before + to_micros(dec!(10)));
        // Idempotent: a second settlement event does not pay again.
        assert!(!ep.apply_settlement("yes", true));
        assert_eq!(ep.cash_um, cash_before + to_micros(dec!(10)));
    }

    #[test]
    fn losing_settlement_does_not_credit() {
        let mut ep = ledger();
        ep.apply_buy_fill("yes", dec!(0.5), dec!(10), 0, RATE);
        let cash = ep.cash_um;
        assert!(ep.apply_settlement("yes", false));
        assert_eq!(ep.cash_um, cash);
    }

    #[test]
    fn topup_structural_neutrality() {
        // The virtual ledger has no deposit API: the reward's neutrality
        // to top-ups holds by construction. This test documents it:
        // there is no entry that can move cash without a fill or settlement.
        let ep = ledger();
        assert_eq!(ep.cash_um, to_micros(dec!(10)));
    }

    #[test]
    fn reward_zero_without_movement() {
        let mut ep = ledger();
        let r = ep.account(0, 2_000, 0.0, 0.0);
        assert_eq!(r.net_pnl_delta_um, 0);
        assert_eq!(r.reward_norm, 0.0);
    }

    #[test]
    fn reward_marks_fee_on_buy() {
        let mut ep = ledger();
        ep.apply_buy_fill("yes", dec!(0.5), dec!(10), 0, RATE);
        // Book: bid 0.5 with full depth → exit ≈ notional − exit fee.
        let mut bids = HashMap::new();
        bids.insert(
            "yes".to_string(),
            vec![(to_micros(dec!(0.5)), to_micros(dec!(10)))],
        );
        let (exit, complete) = ep.exit_value_um(&bids, RATE);
        assert!(complete);
        let r = ep.account(exit, 2_000, 0.0, 0.0);
        // Δequity ≈ −estimated exit fee < 0 (the buy was booked all-in;
        // the live cost is usdc=5 and the 0.5 bid mark minus the exit fee).
        assert!(r.net_pnl_delta_um < 0);
    }

    #[test]
    fn position_without_book_is_worth_zero_and_flag() {
        let mut ep = ledger();
        ep.apply_buy_fill("yes", dec!(0.5), dec!(10), 0, RATE);
        let (exit, complete) = ep.exit_value_um(&HashMap::new(), RATE);
        assert_eq!(exit, 0);
        assert!(!complete);
    }

    #[test]
    fn free_reserve_deducts_free() {
        let mut ep = ledger();
        assert!(ep.reserve_buy(to_micros(dec!(4))));
        assert_eq!(ep.free_cash(), dec!(6));
        ep.release_reservation(to_micros(dec!(4)));
        assert_eq!(ep.free_cash(), dec!(10));
    }

    // ── Reservation is cash already held — never added to equity ────────

    #[test]
    fn reserve_does_not_create_fictitious_equity() {
        // Reserving and releasing WITHOUT a fill must move neither equity nor reward:
        // total cash already includes the reserved part (a reservation is not new income).
        let mut ep = ledger();
        let e0 = ep.equity_um(0);
        assert!(ep.reserve_buy(to_micros(dec!(5))));
        assert_eq!(ep.equity_um(0), e0, "reserve_buy inflated equity");
        ep.release_reservation(to_micros(dec!(5)));
        assert_eq!(ep.equity_um(0), e0);
    }

    #[test]
    fn multiple_reserves_bounded_by_free_cash() {
        // With cash=10: 4+4 fit; the third reservation exceeds the free cash
        // (cash−reserved=2) and must be rejected without reserving anything.
        let mut ep = ledger();
        assert!(ep.reserve_buy(to_micros(dec!(4))));
        assert!(ep.reserve_buy(to_micros(dec!(4))));
        assert!(!ep.reserve_buy(to_micros(dec!(4))), "over-reserve > free cash");
        assert_eq!(ep.reserved_um, to_micros(dec!(8)));
    }

    #[test]
    fn conservation_rejection_after_reserve() {
        // reserve → reject: equity returns intact to the budget (no gain nor
        // loss), and the total cash never moved.
        let mut ep = ledger();
        let e0 = ep.equity_um(0);
        assert!(ep.reserve_buy(to_micros(dec!(5))));
        ep.release_reservation(to_micros(dec!(5)));
        assert_eq!(ep.cash_um, to_micros(dec!(10)));
        assert_eq!(ep.equity_um(0), e0);
    }

    #[test]
    fn conservation_partial_fill_releases_entire_reserve() {
        // Non-resting intents (FOK/FAK): execution is terminal, so
        // a partial fill releases the WHOLE intent reservation and only charges
        // what was executed. Equity = cash + exit, with no reserved remainder.
        let mut ep = ledger();
        let reserve = to_micros(dec!(5.20));
        assert!(ep.reserve_buy(reserve));
        ep.apply_buy_fill("yes", dec!(0.52), dec!(4), reserve, RATE);
        assert_eq!(ep.reserved_um, 0, "unfilled remainder cancelled → reservation released");
        let mut bids = HashMap::new();
        bids.insert(
            "yes".to_string(),
            vec![(to_micros(dec!(0.52)), to_micros(dec!(4)))],
        );
        let (exit, complete) = ep.exit_value_um(&bids, RATE);
        assert!(complete);
        assert_eq!(ep.equity_um(exit), ep.cash_um + exit);
        // A partial fill does not charge the remainder: cash = 10 − 4·0.52 all-in.
        assert_eq!(ep.cash_um, to_micros(dec!(10)) - to_micros(dec!(2.08)));
    }

    #[test]
    fn conservation_buy_sell_settlement_cycle() {
        // Full conservation: reserve → buy fill → partial sell →
        // winning settlement of the remainder. The final cash is exactly
        // the real cash flows and equity never counted the reserved part.
        let mut ep = ledger();
        let reserve = to_micros(dec!(5.20));
        assert!(ep.reserve_buy(reserve));
        ep.apply_buy_fill("yes", dec!(0.52), dec!(10), reserve, RATE);
        assert_eq!(ep.reserved_um, 0);
        assert!(ep.apply_sell_fill("yes", dec!(0.60), dec!(5), RATE));
        assert!(ep.apply_settlement("yes", true));
        let expected = to_micros(dec!(10)) - to_micros(dec!(5.20))
            + to_micros(dec!(3.0)) + to_micros(dec!(5));
        assert_eq!(ep.cash_um, expected);
        assert_eq!(ep.equity_um(0), expected);
    }
}
