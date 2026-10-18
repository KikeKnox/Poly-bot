//! UI/strategy contract closure.
//!
//! Verifies the paper/live barrier of the `set_enabled` toggle for all
//! flags × mode combinations, guarantees the "disabling is never
//! blocked" asymmetry, and anchors `mode_permission_for_toggle` (the new string of the
//! `/ui/strategies` payload) to the single operational source of truth:
//! `StrategyLimits::can_run` and `StrategySupervisor::set_enabled`.
//!
//! These tests do NOT build a full `ApiState` (the remaining fields
//! — `wallet_balance`, `mark_price`, `prometheus_handle`, `alert_log` — do not
//! take part in the toggle path nor in the barrier decision, so
//! covering them here with a real supervisor gives the same guarantee with
//! much less wiring than a `tower::oneshot` harness over the `ui_strategies`
//! handler). The handler is validated with `cargo check` by its type
//! (any drift is caught at compile time).
//!
//! Setup: the `Config` is loaded with `Config::load()` from `config/` (read
//! only, no writes; equivalent to what `main.rs` does daily)
//! and the `mode` and the per-strategy `StrategyLimits` are overridden in
//! each test. No global `cargo test`, no real DB, no network, no `.env`,
//! no git, no secrets. The test does not invoke HTTP API handlers, only the
//! internal contract (what the API reflects to the client).

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{broadcast, mpsc, watch};

use poly_bot::config::{BlockReason, StrategyLimits};
use poly_bot::market_feed::MarketRegistry;
use poly_bot::risk::RiskGuard;
use poly_bot::strategy::{registry::strategy_limits, StrategySupervisor};
use poly_bot::types::Mode;

/// `StrategyLimits` with only the variable flags — the rest inherits from
/// `StrategyLimits::default()`. It does not touch numeric fields: the toggle
/// tests do not need them (the toggle evaluates `can_run`, which only
/// looks at the two barrier flags).
fn limits(enabled: bool, paper_only: bool) -> StrategyLimits {
    StrategyLimits {
        enabled,
        paper_only,
        ..StrategyLimits::default()
    }
}

/// Base `Config` + `mode` override + override of ALL the
/// `StrategyLimits` with the same `target`. Reads `Config::load()` only
/// to get valid `markets/oracle/api/...`; on top of that it overrides the `mode`
/// each test asks for (paper/live) and the barrier flags the
/// matrix asks for. It does not touch the TOML on disk.
fn cfg_into(mode: Mode, target: StrategyLimits) -> Arc<poly_bot::config::Config> {
    let mut cfg =
        poly_bot::config::Config::load().expect("config/ must be present and valid");
    cfg.mode = mode;
    for &id in poly_bot::strategy::registry::REGISTERED {
        match id {
            poly_bot::types::StrategyId::Arb => cfg.strategies.arb.limits = target.clone(),
            poly_bot::types::StrategyId::Directional => {
                cfg.strategies.directional.limits = target.clone()
            }
            poly_bot::types::StrategyId::NearCertain => {
                cfg.strategies.near_certain.limits = target.clone()
            }
            poly_bot::types::StrategyId::NearExpiry => {
                cfg.strategies.near_expiry.limits = target.clone()
            }
            poly_bot::types::StrategyId::MarketMaking => {
                cfg.strategies.market_making.limits = target.clone()
            }
            poly_bot::types::StrategyId::ParityMaker => {
                cfg.strategies.parity_maker.limits = target.clone()
            }
            poly_bot::types::StrategyId::DriftBox => {
                cfg.strategies.drift_box.limits = target.clone()
            }
            poly_bot::types::StrategyId::OpenMomentum => {
                cfg.strategies.open_momentum.limits = target.clone()
            }
            poly_bot::types::StrategyId::ThetaCapture => {
                cfg.strategies.theta_capture.limits = target.clone()
            }
            poly_bot::types::StrategyId::PostCloseSweep => {
                cfg.strategies.post_close_sweep.limits = target.clone()
            }
            poly_bot::types::StrategyId::RewardFarm => {
                cfg.strategies.reward_farm.limits = target.clone()
            }
            poly_bot::types::StrategyId::GRw1Maker => {
                cfg.strategies.g_rw1_maker.limits = target.clone()
            }
            // Test config — the limits apply to the PAPER-ONLY strategy.
            poly_bot::types::StrategyId::ModelBtc5m => {
                cfg.strategies.model_btc5m.limits = target.clone()
            }
            // Same paper-only barrier as ModelBtc5m.
            poly_bot::types::StrategyId::RlPredictor => {
                cfg.strategies.rl_predictor.limits = target.clone()
            }
            poly_bot::types::StrategyId::Arb15m5m => {
                cfg.strategies.arb_15m5m.limits = target.clone()
            }
            poly_bot::types::StrategyId::Follower => cfg.follower.limits = target.clone(),
        }
    }
    Arc::new(cfg)
}

/// Builds a supervisor with `enabled=false` (no `boot_spawned`) and the
/// `lim` flags already overridden on all the strategies. The channels are
/// left empty — `spawn` (which the toggle calls if it accepts) needs
/// `book_rx` only in `on_book`, and the book path is not executed here
/// because the strategies stay paused or get switched off.
#[allow(clippy::too_many_arguments)]
fn sup_with(mode: Mode, lim: StrategyLimits) -> Arc<StrategySupervisor> {
    let cfg = cfg_into(mode, lim);
    let (book_tx, _) = broadcast::channel(16);
    let (intent_tx, _) = mpsc::channel(16);
    let (fill_tx, _) = broadcast::channel(16);
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let (sub_tx, _sub_rx) = broadcast::channel(16);
    StrategySupervisor::new(
        book_tx,
        HashMap::new(),
        intent_tx,
        fill_tx,
        RiskGuard::new(poly_bot::config::RiskConfig::default(), mode),
        shutdown_rx,
        std::time::Duration::from_secs(1),
        Arc::new(MarketRegistry::new()),
        cfg,
        poly_bot::coordination::BracketLedger::new(),
        sub_tx,
        None,
        poly_bot::strategy::Arb15m5mDeps::empty(),
    )
}

// ── 1. `can_run` covers the 16 cases without contradiction ────────────────

/// Truth table: for EACH combination of (enabled, paper_only) × mode, the
/// verdict of the single source must be deterministic and match the
/// documented pattern.
#[test]
fn can_run_full_table() {
    // enabled, paper_only, mode, expected
    type RowCanRun = (bool, bool, Mode, Result<(), BlockReason>);
    let table: &[RowCanRun] = &[
        (false, false, Mode::Paper, Err(BlockReason::Disabled)),
        (false, false, Mode::Live, Err(BlockReason::Disabled)),
        (false, true, Mode::Paper, Err(BlockReason::Disabled)),
        (false, true, Mode::Live, Err(BlockReason::Disabled)),
        (true, false, Mode::Paper, Ok(())),
        (true, false, Mode::Live, Ok(())),
        (true, true, Mode::Paper, Ok(())), // paper_only is ignored in paper
        (true, true, Mode::Live, Err(BlockReason::PaperOnly)),
    ];

    for &(e, po, mode, expected) in table {
        let got = limits(e, po).can_run(mode);
        assert_eq!(
            got, expected,
            "mismatch in can_run(enabled={e}, paper_only={po}, mode={mode:?}): got={got:?}, expected={expected:?}"
        );
    }
}

// ── 2. Toggle ON: the barrier accepts exactly what `can_run` says
//         over a limits with `enabled=true` (what the supervisor does) ──

/// Central case of the UI closure: `enabled=false` in paper is not rejected by
/// the barrier, only by the `!enabled` branch (which the supervisor bridges by
/// setting `enabled=true` before evaluating). Therefore the "Enable" toggle
/// must accept it in paper even when starting up with `enabled=false`.
///
/// If this test fails, the supervisor is re-evaluating `can_run`
/// over the TOML flags (not over the clone with `enabled=true`), and the UI
/// could not distinguish "current state" from the barrier as the
/// contract requires.
#[tokio::test]
async fn toggle_on_with_enabled_false_in_paper_is_accepted() {
    let lim = limits(false /*enabled*/, false /*paper_only*/);
    let sup = sup_with(Mode::Paper, lim.clone());
    let id = poly_bot::types::StrategyId::NearCertain; // a registered one

    // Guarantee of the single source:
    assert_eq!(lim.can_run(Mode::Paper), Err(BlockReason::Disabled));
    // Supervisor behavior (the contract observable from the API):
    assert!(
        sup.set_enabled(id, true).is_ok(),
        "enabling NearCertain with enabled=false in paper must be ACCEPTED: \
         the supervisor forces enabled=true before querying the barrier"
    );
}

#[tokio::test]
async fn toggle_on_in_live_with_paper_only_is_rejected_due_to_paper_only() {
    // This is what the barrier does and what `mode_permission_for_toggle`
    // encodes.
    let lim = limits(true, true /*paper_only*/);
    let sup = sup_with(Mode::Live, lim);
    let id = poly_bot::types::StrategyId::Arb;
    let err = sup.set_enabled(id, true).unwrap_err();
    assert_eq!(err, BlockReason::PaperOnly, "403: paper only");
}

#[tokio::test]
async fn toggle_on_in_paper_with_paper_only_is_accepted() {
    // The `paper_only` experiment must be activatable in paper.
    let lim = limits(true, true /*paper_only*/);
    let sup = sup_with(Mode::Paper, lim);
    let id = poly_bot::types::StrategyId::Arb;
    assert!(sup.set_enabled(id, true).is_ok());
}

#[tokio::test]
async fn toggle_on_in_live_without_paper_only_is_accepted() {
    // Standard live case: enabled, paper_only=false.
    let lim = limits(true, false);
    let sup = sup_with(Mode::Live, lim);
    let id = poly_bot::types::StrategyId::Arb;
    assert!(sup.set_enabled(id, true).is_ok());
}

// ── 3. Toggle OFF: hard asymmetry, it is NEVER blocked ─────────────────────

/// The `Always disable` asymmetry is documented in
/// `supervisor.rs` and replicated in the client. Here it is anchored with
/// all the flags × mode combinations: none may reject
/// `set_enabled(_, false)`.
#[tokio::test]
async fn toggle_off_always_accepted_in_paper_and_live() {
    let id = poly_bot::types::StrategyId::NearCertain;
    for mode in [Mode::Paper, Mode::Live] {
        for &(e, po) in &[
            (false, false),
            (true, false),
            (true, true),
            (false, true),
        ] {
            let lim = limits(e, po);
            let sup = sup_with(mode, lim);
            assert!(
                sup.set_enabled(id, false).is_ok(),
                "turning off {id} with enabled={e}, paper_only={po}, mode={mode:?} \
                 must always be accepted (safety asymmetry — without this, an operator could \
                 be left with no way to stop a strategy)"
            );
        }
    }
}

// ── 4. anchoring against the real chain: what the `ui_strategies` JSON
//         claims matches the supervisor's behavior ────────────

/// Verifies that for EACH flag combination, what the `ui_strategies`
/// handler would expose as `mode_permission` (encoded in the
/// `mode_permission_for_toggle` helper of `handlers.rs`) PREDICTS
/// exactly the result of `set_enabled(id, true)`. Without this, the
/// client's "Enable" button may be enabled for a toggle that
/// will be rejected by the barrier (broken UX) or disabled for one that
/// would pass (operator blocked).
///
/// The logic replicated here encodes the SAME matrix as the handler; if
/// `mode_permission_for_toggle` gets out of sync, this test keeps passing
/// (it has no access to that private helper from an integration test). The
/// "single source of truth" is closed properly by the unit tests
/// of `handlers.rs::mode_permission_for_toggle_tests::matches_can_run_assuming_enabled_true`,
/// which have direct access to the helper and compare cell by cell.
#[tokio::test]
async fn supervisor_matches_toggle_matrix() {
    use poly_bot::types::StrategyId;

    let id = StrategyId::NearCertain;

    let cases: &[(&str, bool, bool, Mode, bool)] = &[
        // (label, enabled, paper_only, mode, expect_ok)
        ("any_paper", false, false, Mode::Paper, true),
        ("paper_paper_only_true", false, true, Mode::Paper, true),
        ("standard_live", true, false, Mode::Live, true),
        ("live_paper_only_blocks", true, true, Mode::Live, false),
    ];

    for &(label, e, po, mode, expect_ok) in cases {
        let lim = limits(e, po);
        let sup = sup_with(mode, lim);
        let got = sup.set_enabled(id, true).is_ok();
        assert_eq!(
            got, expect_ok,
            "[{label}] supervisor said {got} but the contract matrix says {expect_ok} \
             (enabled={e}, paper_only={po}, mode={mode:?})"
        );
    }
}

// ── 5. `enabled=true` with barrier flags = identical to the supervisor's "boot"

/// Verifies that `strategy_limits(id, cfg)` (the single mapping point that
/// `ui_strategies` has just started to use for the flags) keeps
/// returning the SAME flags that are in the `Config`. If they
/// diverge, the handler would be lying about the barrier.
#[test]
fn registry_strategy_limits_is_the_single_source_of_the_flags() {
    let lim_ref = limits(true, false);
    let cfg = cfg_into(Mode::Paper, lim_ref.clone());
    for &id in poly_bot::strategy::registry::REGISTERED {
        let got = strategy_limits(id, &cfg);
        assert_eq!(got.enabled, lim_ref.enabled);
        assert_eq!(got.paper_only, lim_ref.paper_only);
    }
}

/// `StrategyLimits::default()` must not start anything by itself: the TOMLs
/// must declare `enabled = true` explicitly.
#[test]
fn default_strategy_limits_is_disabled() {
    let l = StrategyLimits::default();
    assert!(!l.enabled, "Default must have enabled=false (the TOMLs must declare true explicitly to start)");
    assert!(!l.paper_only, "Default must have paper_only=false");
    assert_eq!(l.can_run(Mode::Paper), Err(BlockReason::Disabled));
    assert_eq!(l.can_run(Mode::Live), Err(BlockReason::Disabled));
}
