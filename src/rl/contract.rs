//! Action and observation contract of the RL predictor.
//!
//! The names, order and dimensions are CONTRACTUAL: the `policy.json`
//! artifact declares `action_names`/`feature_names` and the loader compares
//! them literally with these tables. Changing them breaks compatibility
//! with trained artifacts — that is deliberate (fail-closed).

/// Number of actions in the contractual space.
pub const NUM_ACTIONS: usize = 9;

/// Contractual indices (fixed order — do NOT reorder).
pub const HOLD: usize = 0;
pub const BUY_YES_SMALL: usize = 1;
pub const BUY_YES_FULL: usize = 2;
pub const BUY_NO_SMALL: usize = 3;
pub const BUY_NO_FULL: usize = 4;
pub const SELL_YES_HALF: usize = 5;
pub const SELL_YES_ALL: usize = 6;
pub const SELL_NO_HALF: usize = 7;
pub const SELL_NO_ALL: usize = 8;

/// Exact contract names, in index order.
pub const ACTION_NAMES: [&str; NUM_ACTIONS] = [
    "HOLD",
    "BUY_YES_SMALL",
    "BUY_YES_FULL",
    "BUY_NO_SMALL",
    "BUY_NO_FULL",
    "SELL_YES_HALF",
    "SELL_YES_ALL",
    "SELL_NO_HALF",
    "SELL_NO_ALL",
];

/// Contractual `feature_schema` of the observation vector.
pub const FEATURE_SCHEMA: &str = "rl_predictor.features.v1";

/// Number of observation features.
pub const NUM_FEATURES: usize = 22;

/// Exact names of the 22 features (HERMES_SPEC §4), in order.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "p_yes",
    "p_missing",
    "yes_bid",
    "yes_ask",
    "no_bid",
    "no_ask",
    "yes_bid_depth_usdc",
    "yes_ask_depth_usdc",
    "no_bid_depth_usdc",
    "no_ask_depth_usdc",
    "time_remaining_frac",
    "yes_position_shares",
    "no_position_shares",
    "yes_cost_basis_usdc",
    "no_cost_basis_usdc",
    "free_budget_usdc",
    "reserved_budget_usdc",
    "pending_order",
    "yes_book_missing",
    "no_book_missing",
    "book_age_seconds",
    "buy_fee_rate",
];

// Feature indices (to build the observation without strings).
pub const F_P_YES: usize = 0;
pub const F_P_MISSING: usize = 1;
pub const F_YES_BID: usize = 2;
pub const F_YES_ASK: usize = 3;
pub const F_NO_BID: usize = 4;
pub const F_NO_ASK: usize = 5;
pub const F_YES_BID_DEPTH: usize = 6;
pub const F_YES_ASK_DEPTH: usize = 7;
pub const F_NO_BID_DEPTH: usize = 8;
pub const F_NO_ASK_DEPTH: usize = 9;
pub const F_TIME_REMAINING: usize = 10;
pub const F_YES_SHARES: usize = 11;
pub const F_NO_SHARES: usize = 12;
pub const F_YES_COST: usize = 13;
pub const F_NO_COST: usize = 14;
pub const F_FREE_BUDGET: usize = 15;
pub const F_RESERVED: usize = 16;
pub const F_PENDING: usize = 17;
pub const F_YES_BOOK_MISSING: usize = 18;
pub const F_NO_BOOK_MISSING: usize = 19;
pub const F_BOOK_AGE: usize = 20;
pub const F_BUY_FEE_RATE: usize = 21;

/// Legal-action mask (`true` = legal). Indexed by contractual action
/// index — NEVER by position in a reduced "legal" list
/// (`action-mask-order-is-not-identity`): an action's identity
/// (which token it buys/sells) does not depend on which other actions are legal.
pub type ActionMask = [bool; NUM_ACTIONS];

/// Mask with only HOLD legal — the safe default state under
/// any uncertain condition (pending recovery, uncertain market, etc.).
pub const MASK_HOLD_ONLY: ActionMask = [true, false, false, false, false, false, false, false, false];

/// Token side an action refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenSide {
    Yes,
    No,
}

/// Operational class of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    Hold,
    /// Buy of `TokenSide` with a fraction of the FULL budget
    /// (`true` = SMALL = half, `false` = FULL).
    Buy { side: TokenSide, small: bool },
    /// Sell of `TokenSide`: `true` = HALF (half of the position),
    /// `false` = ALL (everything available that is not reserved).
    Sell { side: TokenSide, half: bool },
}

/// Classifies a contractual action index. `None` if out of range.
pub fn action_kind(action: usize) -> Option<ActionKind> {
    Some(match action {
        HOLD => ActionKind::Hold,
        BUY_YES_SMALL => ActionKind::Buy { side: TokenSide::Yes, small: true },
        BUY_YES_FULL => ActionKind::Buy { side: TokenSide::Yes, small: false },
        BUY_NO_SMALL => ActionKind::Buy { side: TokenSide::No, small: true },
        BUY_NO_FULL => ActionKind::Buy { side: TokenSide::No, small: false },
        SELL_YES_HALF => ActionKind::Sell { side: TokenSide::Yes, half: true },
        SELL_YES_ALL => ActionKind::Sell { side: TokenSide::Yes, half: false },
        SELL_NO_HALF => ActionKind::Sell { side: TokenSide::No, half: true },
        SELL_NO_ALL => ActionKind::Sell { side: TokenSide::No, half: false },
        _ => return None,
    })
}

/// Is the action an entry (buy)?
pub fn is_buy(action: usize) -> bool {
    matches!(action_kind(action), Some(ActionKind::Buy { .. }))
}

/// Is the action an exit (sell)?
pub fn is_sell(action: usize) -> bool {
    matches!(action_kind(action), Some(ActionKind::Sell { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contract: exactly 9 actions with the names and order of
    /// the spec — the loader compares the artifact's `action_names`
    /// against this literal table.
    #[test]
    fn action_order_contractual() {
        assert_eq!(ACTION_NAMES[HOLD], "HOLD");
        assert_eq!(ACTION_NAMES[1], "BUY_YES_SMALL");
        assert_eq!(ACTION_NAMES[2], "BUY_YES_FULL");
        assert_eq!(ACTION_NAMES[3], "BUY_NO_SMALL");
        assert_eq!(ACTION_NAMES[4], "BUY_NO_FULL");
        assert_eq!(ACTION_NAMES[5], "SELL_YES_HALF");
        assert_eq!(ACTION_NAMES[6], "SELL_YES_ALL");
        assert_eq!(ACTION_NAMES[7], "SELL_NO_HALF");
        assert_eq!(ACTION_NAMES[8], "SELL_NO_ALL");
        assert_eq!(ACTION_NAMES.len(), NUM_ACTIONS);
    }

    #[test]
    fn feature_order_contractual() {
        assert_eq!(FEATURE_NAMES.len(), NUM_FEATURES);
        assert_eq!(FEATURE_NAMES[0], "p_yes");
        assert_eq!(FEATURE_NAMES[1], "p_missing");
        assert_eq!(FEATURE_NAMES[10], "time_remaining_frac");
        assert_eq!(FEATURE_NAMES[17], "pending_order");
        assert_eq!(FEATURE_NAMES[21], "buy_fee_rate");
        assert_eq!(FEATURE_SCHEMA, "rl_predictor.features.v1");
    }

    #[test]
    fn mask_hold_only_just_hold() {
        assert!(MASK_HOLD_ONLY[HOLD]);
        assert!(MASK_HOLD_ONLY[1..].iter().all(|l| !l));
    }

    #[test]
    fn action_kind_covers_all() {
        for a in 0..NUM_ACTIONS {
            assert!(action_kind(a).is_some());
        }
        assert!(action_kind(NUM_ACTIONS).is_none());
        assert!(is_buy(BUY_YES_FULL) && is_buy(BUY_NO_SMALL));
        assert!(is_sell(SELL_YES_ALL) && is_sell(SELL_NO_HALF));
        assert!(!is_buy(HOLD) && !is_sell(HOLD));
    }
}
