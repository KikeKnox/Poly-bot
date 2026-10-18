//! Startup decisions, kept apart from `main.rs` so they can be tested.
//!
//! They live in the library (not the binary) because `cargo test --lib` does not
//! cover `main.rs`: the startup policy is exactly what must be verifiable
//! without network or credentials.

use crate::types::Mode;

/// Must startup abort because the CLOB GTD order reconciliation
/// could not be completed?
///
/// The paper/live asymmetry is deliberate, like the one in
/// `StrategyLimits::can_run`:
///
/// - **Live**: fails CLOSED. Starting without knowing which orders are open on
///   the CLOB is exactly the scenario that produces orphan legs and
///   miscounted exposure — with real money at stake. It used to log
///   *"operating without reconciliation"* and carry on.
/// - **Paper**: does not abort. There are no remote orders to reconcile (the path
///   is not even executed: `main.rs` only reconciles if a live CLOB client
///   exists) and blocking startup would only prevent trading with no money at stake.
pub fn must_abort_startup(mode: Mode, ok_reconciliation: bool) -> bool {
    matches!(mode, Mode::Live) && !ok_reconciliation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_with_failed_reconciliation_aborts() {
        assert!(
            must_abort_startup(Mode::Live, false),
            "in live, without reconciliation no strategy can be spawned"
        );
    }

    #[test]
    fn paper_with_failed_reconciliation_starts() {
        assert!(
            !must_abort_startup(Mode::Paper, false),
            "in paper there are no remote orders to reconcile: startup continues"
        );
    }

    #[test]
    fn live_with_correct_reconciliation_starts() {
        assert!(!must_abort_startup(Mode::Live, true));
    }
}
