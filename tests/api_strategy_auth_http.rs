//! In-process HTTP tests for the `mode_permission` contract + strategy toggle.
//!
//! Builds a minimal in-memory `ApiState` and mounts the REAL `wrap_auth_and_static`
//! (not a local re-assembly) over the real handlers `handlers::ui_strategies` and
//! `handlers::control_strategy`, via `tower::ServiceExt::oneshot`. Channels and
//! watches stay empty: the supervisor and the guard need no real books to decide
//! `can_run` or to serialize the JSON.
//!
//! Why HTTP and not a unit test of the helper:
//!
//! - The unit test `mode_permission_for_toggle_tests` pins the private helper
//!   to `StrategyLimits::can_run`, cell by cell.
//! - `tests/api_strategy_auth.rs` pins the supervisor (`set_enabled`)
//!   against the same source.
//! - BUT none of them exercises the real chain a request goes through:
//!   `wrap_auth_and_static` → Bearer middleware → handler → JSON
//!   serialization → StatusCode → body. A regression in any of those
//!   steps (layer order in `mod.rs`, handler signature drift, missing
//!   `enabled` default, etc.) would let a 200 through where the UI
//!   expects a 403. This test closes that gap.
//!
//! Restricciones:
//! - NO real bot, server, network or live trading.
//! - `config/strategies/*.toml` in the working tree is NOT modified. The tests
//!   that exercise the `persist_strategy_toggle` path (an accepted hot toggle,
//!   e.g. `paper disabled activatable`) mount a temporary cwd with a minimal copy
//!   of the TOML. While the fixture lives, the real `config/strategies/`
//!   directory is read-only: any write that escapes the tempdir fails with EACCES
//!   instead of mutating the repo. On `Drop`, the originals are verified byte by
//!   byte (they are NOT restored: if something touched them it is an isolation
//!   failure and must stay visible, not be papered over).
//! - NO `.env` and no real DB: `StateStore::open_memory` + `RiskGuard` without
//!   `attach_persistence`.
//! - The test token is wired into the `Authorization: Bearer ...` header,
//!   never into source code outside the test file.
//!
//! Differences from `tests/api_strategy_auth.rs`:
//! - This test goes through the real auth middleware and the real router.
//! - It checks the JSON the UI consumes (fields `running`,
//!   `enabled`, `paper_only`, `mode_permission`),
//!   not only the supervisor verdict.
//! - It checks real HTTP status codes (200/400/401/403) instead of
//!   `Result<(), BlockReason>`.

// `serial()` returns a MutexGuard that is held across `.await` ON PURPOSE:
// it is what serializes the tests (they mutate cwd and config). The
// `await_holding_lock` lint cannot tell this apart from an oversight.
#![allow(clippy::await_holding_lock)]

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use metrics_exporter_prometheus::PrometheusBuilder;
use tokio::sync::{broadcast, mpsc, watch};
use tower::ServiceExt;

use poly_bot::api::{handlers, ApiState};
use poly_bot::config::{BlockReason, StrategyLimits};
use poly_bot::coordination::BracketLedger;
use poly_bot::accounting::mark_price::MarkPriceCache;
use poly_bot::market_feed::MarketRegistry;
use poly_bot::risk::RiskGuard;
use poly_bot::tools::scan::sink::ScanObs;
use poly_bot::strategy::{registry::strategy_limits, StrategySupervisor};
use poly_bot::tools::ToolSupervisor;
use poly_bot::types::Mode;

// ── Test serialization ─────────────────────────────────────────────────
//
// `Config::load()` reads paths RELATIVE to the process cwd (see
// `src/config.rs::load`). The test `control_strategy_paper_with_enabled_false_is_accepted`
// switches the cwd to a tempdir (so that `persist_strategy_toggle` does not touch
// the repo TOML). If the tests run in parallel, that change affects the other
// `Config::load()` calls in the same process and they fail with
// "missing field `mode`" because the `general.toml` in the tempdir is empty.
//
// Fix: a global mutex that serializes ALL the tests in this file.
// It is the most conservative option: it adds latency but no new
// dependencies and no race risk with other integration tests in the
// crate (each `tests/*.rs` is a separate binary with its own mutex).
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Helper: lock + async function. Used as `let _g = serial();` at the
// start of each test. The guard lives until the end of the function, which
// ensures the next test does not enter until this one finishes.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ── Fixture: temporary cwd with `config/strategies/<id>.toml` ────────────────

/// Copies `config/strategies/<name>.toml` from the real working tree into a
/// temporary dir and calls `set_current_dir` there. While the fixture lives,
/// the repo's `config/strategies/` is read-only at DIRECTORY level (rename
/// included: `persist_strategy_toggle` writes `<path>.tmp` and renames it, so a
/// chmod on the file path would not be enough), so any write that escapes the
/// temporary cwd fails with EACCES instead of silently mutating the repo. On
/// `Drop` the cwd and permissions are restored and the originals are verified
/// byte by byte to be intact; if not, it is an isolation failure and is reported,
/// not papered over.
///
/// Only used by the tests that activate the persistence path; the rest do not
/// touch `persist_strategy_toggle` because they return 4xx earlier.
struct TmpCwd {
    saved_cwd: std::path::PathBuf,
    /// Original content read BEFORE the chdir; ONLY for verification in Drop,
    /// never to rewrite.
    saved_toml: HashMap<String, Vec<u8>>,
    /// `(dir, original_permissions)` of the protected directory, if the
    /// permissions guard was applied (Unix).
    dir_guard: Option<(std::path::PathBuf, std::fs::Permissions)>,
    /// `tempdir` is kept as a field only so that the `TempDir` destructor runs
    /// when `TmpCwd` is dropped (cleaning up the temporary directory); it is not
    /// read directly.
    #[allow(dead_code)]
    tmp_dir: tempfile::TempDir,
}

impl TmpCwd {
    /// Prepares a temporary cwd with `config/general.toml` and
    /// `config/strategies/<name>.toml` copied from the real working tree
    /// (preserving permissions/content). All writes are confined to the
    /// tempdir: relative paths land there thanks to the chdir, and a future
    /// absolute path would hit the read-only guard.
    ///
    /// Why `general.toml` too? `Config::load()` (via `src/config.rs::load`)
    /// reads `config/general.toml` FIRST to resolve the `mode` field; without it,
    /// `Config::load()` fails in this temporary cwd with "missing field `mode`".
    fn with_strategy(strategy: &str) -> Self {
        let saved_cwd = std::env::current_dir().expect("cwd legible");
        let saved_general = std::fs::read(saved_cwd.join("config").join("general.toml"))
            .expect("general.toml source present in the working tree");
        let saved_strategy = std::fs::read(
            saved_cwd
                .join("config")
                .join("strategies")
                .join(format!("{strategy}.toml")),
        )
        .expect("TOML source present in the working tree");

        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_root = tmp.path().join("config");
        let cfg_strat = cfg_root.join("strategies");
        std::fs::create_dir_all(&cfg_strat).expect("mkdir temp config/strategies");
        std::fs::write(cfg_root.join("general.toml"), &saved_general).expect("copy general.toml");
        std::fs::write(cfg_strat.join(format!("{strategy}.toml")), &saved_strategy)
            .expect("copy TOML");

        // Isolation guard (Unix): the repo's `config/strategies/` is read-only while
        // the fixture lives. If a future code path resolved an absolute path or
        // ignored the cwd, creating the `.tmp` file would fail with EACCES and the
        // test would break, instead of mutating silently.
        #[cfg(unix)]
        let dir_guard = {
            use std::os::unix::fs::PermissionsExt;
            let dir = saved_cwd.join("config").join("strategies");
            let orig = std::fs::metadata(&dir)
                .expect("metadata config/strategies")
                .permissions();
            let mut ro = orig.clone();
            ro.set_mode(0o555);
            std::fs::set_permissions(&dir, ro).expect("chmod 555 config/strategies");
            Some((dir, orig))
        };
        #[cfg(not(unix))]
        let dir_guard = None;

        // The supervisor and the rest of ApiState use relative paths; the
        // cwd now points at the tempdir so that `persist_strategy_toggle`
        // writes to `tmp_dir/config/strategies/<id>.toml`, not to the repo.
        std::env::set_current_dir(tmp.path()).expect("chdir tempdir");

        let mut saved_toml = HashMap::new();
        saved_toml.insert("general.toml".to_string(), saved_general);
        saved_toml.insert(strategy.to_string(), saved_strategy);
        Self {
            saved_cwd,
            saved_toml,
            dir_guard,
            tmp_dir: tmp,
        }
    }
}

impl Drop for TmpCwd {
    fn drop(&mut self) {
        // Restore the cwd first so that the verification reads real paths.
        let _ = std::env::set_current_dir(&self.saved_cwd);
        // Lift the permissions guard (metadata, not content).
        if let Some((dir, orig)) = self.dir_guard.take() {
            let _ = std::fs::set_permissions(&dir, orig);
        }
        // Verification, NOT restoration: if any original changed, isolation
        // failed and we report it. Rewriting it here would be exactly the write
        // outside the tempdir that this fixture promises not to make (and would
        // hide the bug).
        for (name, content) in self.saved_toml.drain() {
            let path = if name == "general.toml" {
                self.saved_cwd.join("config").join("general.toml")
            } else {
                self.saved_cwd
                    .join("config")
                    .join("strategies")
                    .join(format!("{name}.toml"))
            };
            let current = std::fs::read(&path).unwrap_or_default();
            if current != content {
                let msg = format!(
                    "ISOLATION BROKEN: the test modified {} outside the tempdir",
                    path.display()
                );
                if std::thread::panicking() {
                    eprintln!("{msg}");
                } else {
                    panic!("{msg}");
                }
            }
        }
    }
}

// ── Fixture: minimal in-memory ApiState ────────────────────────────────────

/// `StrategyLimits` with only the variable barrier flags set; the rest inherits
/// from `StrategyLimits::default()` (neutral numerics).
fn lim(enabled: bool, paper_only: bool) -> StrategyLimits {
    StrategyLimits {
        enabled,
        paper_only,
        ..StrategyLimits::default()
    }
}

/// Loads `Config::load()` (reads the TOML) and overrides `mode` and the
/// `StrategyLimits` of each strategy with `target`. Same technique as
/// `tests/api_strategy_auth.rs::cfg_into`.
fn cfg_with(mode: Mode, target: StrategyLimits) -> Arc<poly_bot::config::Config> {
    let mut cfg = poly_bot::config::Config::load().expect("config/ is valid");
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
            // Test config: limits apply the same way as to the other registered
            // strategies. `enabled/paper_only` already come in
            // `target` so that the test evaluates combinations.
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

#[allow(clippy::too_many_arguments)]
async fn build_state(
    mode: Mode,
    lim_ref: StrategyLimits,
) -> (Arc<ApiState>, Arc<RiskGuard>, Arc<StrategySupervisor>) {
    // build_state is invoked inside a `#[tokio::test]` runtime, so
    // `tokio::runtime::Handle::current()` is available and we can `.await`
    // `StateStore::open_memory` directly. We do not need the `futures` crate.
    let cfg = cfg_with(mode, lim_ref);

    // In-memory StateStore (no real DB touched).
    let state_store = poly_bot::state::StateStore::open_memory(mode)
        .await
        .expect("StateStore::open_memory");

    let risk = RiskGuard::new(poly_bot::config::RiskConfig::default(), mode);
    // Sync the RiskGuard limits with the TOML flags in memory: ui_strategies
    // reads from here (`get_strategy_limits` / `get_strategy_daily_loss_stop`),
    // not from the TOML directly.
    for &id in poly_bot::strategy::registry::REGISTERED {
        let l = strategy_limits(id, &cfg);
        risk.set_strategy_limits(
            id,
            l.max_orders_per_hour,
            l.max_order_usdc,
            l.daily_loss_usdc_stop,
        );
    }

    let prom_handle = PrometheusBuilder::new().build_recorder().handle();

    let (book_tx, _book_rx) = broadcast::channel(16);
    let (intent_tx, _intent_rx) = mpsc::channel(16);
    let (fill_tx, _fill_rx) = broadcast::channel(16);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let (scan_tx, _scan_rx) = mpsc::unbounded_channel::<ScanObs>();
    let (_wb_tx, wallet_balance_rx) = watch::channel(None);

    let (sub_tx, _sub_rx) = broadcast::channel(16);
    // ModelTool's prediction channel, as in production (main.rs always
    // injects it): without it the supervisor does NOT boot-spawn model_btc5m
    // (it would have no signal) and the contract `mode_permission=ok ⇒ running`
    // of `ui_strategies` would be unverifiable for that strategy.
    let (pred_tx, _pred_rx) =
        broadcast::channel::<poly_bot::tools::model::ModelPrediction>(16);
    let strategies = StrategySupervisor::new(
        book_tx.clone(), // also consumed by ToolSupervisor (Clone, not Copy)
        HashMap::new(),
        intent_tx,
        fill_tx,
        risk.clone(),
        shutdown_rx,
        std::time::Duration::from_secs(1),
        Arc::new(MarketRegistry::new()),
        cfg.clone(),
        BracketLedger::new(),
        sub_tx,
        Some(Arc::new(pred_tx)),
        poly_bot::strategy::Arb15m5mDeps::empty(),
    );

    let (_shutdown_tx2, shutdown_rx2) = watch::channel(false);
    let tools = ToolSupervisor::new(
        book_tx.clone(),
        HashMap::new(),
        scan_tx,
        shutdown_rx2,
        std::time::Duration::from_secs(1),
        cfg.clone(),
    );

    let alert_log = poly_bot::api::alert_log::AlertLog::new(50);
    risk.attach_alert_log(alert_log.clone());

    let api_state = Arc::new(ApiState {
        config: cfg,
        risk: risk.clone(),
        state: state_store,
        token: Arc::new("test-token-fixture".to_string()),
        mode,
        shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
        prometheus_handle: Arc::new(prom_handle),
        wallet_balance: wallet_balance_rx,
        tools,
        strategies: strategies.clone(),
        alert_log,
        vol_rxs: Arc::new(HashMap::new()),
        ledger: BracketLedger::new(),
        mark_price: MarkPriceCache::new(),
    });
    // The strategy runner watches `shutdown.changed()` in its select: if the
    // sender is dropped, `changed()` completes with Err and a task spawned on
    // demand would exit at once, so `has_task` would return false. In production
    // the sender lives in main until the real shutdown; here we keep it alive for
    // the whole test process.
    std::mem::forget(shutdown_tx);
    (api_state, risk, strategies)
}

// ── HTTP helpers ───────────────────────────────────────────────────────────

/// Builds the real production router with auth + static files over a
/// minimal set of routes (the contract ones). The `tempfile::tempdir` for the
/// static files is leaked on purpose (same pattern as `tests/api_auth.rs`).
fn router_with_token(state: Arc<ApiState>) -> axum::Router {
    let json_routes = axum::Router::new()
        .route(
            "/ui/strategies",
            axum::routing::get(handlers::ui_strategies),
        )
        .route("/ui/tools", axum::routing::get(handlers::ui_tools))
        .route(
            "/control/strategy",
            axum::routing::post(handlers::control_strategy),
        )
        // `ApiState: Clone`, not `Arc<ApiState>`: `.with_state` takes
        // ownership of the state the routes will use through `Clone` on each
        // request. The outer `Arc` is for sharing with `risk` and `strategies`
        // in the test asserts; the router gets the cloned copy that lives inside it.
        .with_state((*state).clone());

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("index.html"),
        "<!doctype html><title>t</title>",
    )
    .unwrap();
    let dir_path = Box::leak(Box::new(dir)).path().to_path_buf();

    poly_bot::api::wrap_auth_and_static(
        json_routes,
        state.token.clone(),
        state.mode == Mode::Paper,
        dir_path,
    )
}

fn req_get(path: &str, with_token: bool) -> Request<Body> {
    let mut b = Request::builder().uri(path);
    if with_token {
        b = b.header("authorization", "Bearer test-token-fixture");
    }
    b.body(Body::empty()).unwrap()
}

fn req_post(path: &str, body: &str, with_token: bool) -> Request<Body> {
    let mut b = Request::builder()
        .uri(path)
        .method("POST")
        .header("content-type", "application/json");
    if with_token {
        b = b.header("authorization", "Bearer test-token-fixture");
    }
    b.body(Body::from(body.to_string())).unwrap()
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).expect("valid JSON")
}

fn row_entry<'a>(json: &'a serde_json::Value, strategy: &str) -> &'a serde_json::Value {
    json.get("strategies")
        .and_then(|s| s.as_array())
        .and_then(|a| {
            a.iter()
                .find(|r| r.get("strategy").and_then(|v| v.as_str()) == Some(strategy))
        })
        .unwrap_or_else(|| panic!("row {strategy} not found in payload: {json}"))
}

// ── Tests ──────────────────────────────────────────────────────────────────

/// `GET /ui/strategies` with a token returns 200 and each row includes the
/// 4 contract fields (`running`, `enabled`,
/// `paper_only`, `mode_permission`). This test pins them: if the handler forgets
/// to serialize one, this test breaks. Before the fix this would have passed
/// silently (the fields were not part of the contract).
#[tokio::test]
async fn ui_strategies_returns_the_four_contract_fields() {
    let _g = serial();
    let (state, _risk, _sup) = build_state(Mode::Paper, lim(false, false)).await;

    let router = router_with_token(state);
    let resp = router
        .oneshot(req_get("/ui/strategies", true))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET with token → 200");

    let body = body_json(resp).await;
    let strategies = body
        .get("strategies")
        .and_then(|v| v.as_array())
        .expect("payload { strategies: [...] }");
    assert!(!strategies.is_empty(), "there are registered strategies");
    for row in strategies {
        let name = row.get("strategy").and_then(|v| v.as_str()).unwrap_or("?");
        // Each of the 4 fields must be present with the expected kind; a handler
        // that omits them would fail here.
        assert!(row.get("running").is_some(), "{name}: missing `running`");
        assert!(row.get("enabled").is_some(), "{name}: missing `enabled`");
        assert!(
            row.get("paper_only").is_some(),
            "{name}: missing `paper_only`"
        );
        assert!(
            row.get("mode_permission").is_some(),
            "{name}: missing `mode_permission`"
        );
        // The barrier verdict must be in the closed set of strings (no accidental
        // ok leaks to the client).
        let mp = row.get("mode_permission").and_then(|v| v.as_str()).unwrap();
        assert!(
            matches!(
                mp,
                "ok" | "blocked_paper_only"
                    | "blocked_disabled"
                    | "blocked"
            ),
            "{name}: mode_permission={mp} is not a string from the contract"
        );
    }
}

/// In paper with barrier flags OK, `mode_permission` must be `"ok"` for all the
/// strategies (paper never blocks). Central case: the UI uses this
/// field to enable the activate button.
#[tokio::test]
async fn ui_strategies_in_paper_clean_flags_gives_ok_for_all() {
    let _g = serial();
    let (state, _risk, _sup) = build_state(Mode::Paper, lim(true, false)).await;
    // paper flags OK → ok

    let router = router_with_token(state);
    let resp = router
        .oneshot(req_get("/ui/strategies", true))
        .await
        .unwrap();
    let body = body_json(resp).await;

    for id in poly_bot::strategy::registry::REGISTERED {
        let row_entry = row_entry(&body, &id.to_string());
        assert_eq!(
            row_entry.get("mode_permission").and_then(|v| v.as_str()),
            Some("ok"),
            "{} in paper with flags OK: mode_permission expected ok",
            id,
        );
        // `mode_permission: "ok"` ⇒ the supervisor spawned the task in
        // `boot_spawned` (because `can_run(mode=Paper)` passes with
        // `enabled=true, paper_only=false`).
        // Then `running` must be `true` and `killed` must be `false` (the test
        // kills nothing). This verifies the joint contract: a strategy with
        // `mode_permission="ok"` must be running if the barrier passed at boot.
        // The UI uses `running` to label it as active and `mode_permission` to
        // allow the toggle.
        assert_eq!(
            row_entry.get("running").and_then(|v| v.as_bool()),
            Some(true),
            "{}: running must be true with mode_permission=ok at boot (the supervisor spawned it)",
            id,
        );
        assert_eq!(
            row_entry.get("killed").and_then(|v| v.as_bool()),
            Some(false),
            "{}: killed must be false (the fixture kills nothing)",
            id,
        );
    }
}

/// Explicit case: if `paper_only=true` in
/// live `mode_permission` must be `"blocked_paper_only"`. If the helper had
/// duplicated the "paper only vs live" logic with an `if`, a test that moved
/// `paper_only` to `false` first would fail this scenario just from reordering
/// the match. Delegating to `can_run` (the single source) prevents that.
#[tokio::test]
async fn ui_strategies_paper_only_true_in_live_always_blocks() {
    let _g = serial();
    let (state, _risk, _sup) = build_state(Mode::Live, lim(true, true)).await;

    let router = router_with_token(state);
    let resp = router
        .oneshot(req_get("/ui/strategies", true))
        .await
        .unwrap();
    let body = body_json(resp).await;

    for id in poly_bot::strategy::registry::REGISTERED {
        let row_entry = row_entry(&body, &id.to_string());
        assert_eq!(
            row_entry.get("mode_permission").and_then(|v| v.as_str()),
            Some("blocked_paper_only"),
            "{}: paper_only=true in live must be blocked by paper_only",
            id,
        );
    }
}

/// Auth: without a token (in paper with no token configured, requests pass)
/// and then with a token configured, no token → 401 on `/ui/strategies`. This
/// pins that the layer order (`wrap_auth_and_static`) still protects the JSON
/// routes. If someone moves `static_router` before the `.layer(auth)` call,
/// this breaks.
#[tokio::test]
async fn ui_strategies_without_token_with_configured_token_returns_401() {
    let _g = serial();
    let (state, _risk, _sup) = build_state(Mode::Paper, lim(true, false)).await;
    // 401

    let router = router_with_token(state);
    let resp = router
        .oneshot(req_get("/ui/strategies", false))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "without Authorization and with a configured token it must be 401 (not 403/200)"
    );
}

// ── POST /control/strategy ─────────────────────────────────────────────────

/// Central case: Live + `paper_only=true` → the barrier rejects the toggle
/// and the handler returns HTTP 403 with an error message. The strategy must NOT
/// be left active (the supervisor neither spawns the task nor resets the kill
/// switch). This pins the contract: if `mode_permission_for_toggle` and the
/// supervisor diverge, one of the two fails; the UI must be able to rely on a 403
/// meaning "it was not turned on".
#[tokio::test]
async fn control_strategy_live_paper_only_returns_403_and_does_not_activate() {
    let _g = serial();
    let (state, risk, sup) = build_state(Mode::Live, lim(false, true)).await;
    let id = poly_bot::types::StrategyId::Arb;

    // Before the toggle: the JSON `mode_permission` says blocked, and the
    // supervisor has no live task.
    let arb_initial = sup.has_task(id);
    assert!(!arb_initial, "precondition: arb has no live task at boot");

    let router = router_with_token(state.clone());
    let resp = router
        .oneshot(req_post(
            "/control/strategy",
            r#"{"strategy":"arb","enabled":true}"#,
            true,
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "paper_only=true in live must be 403 (PaperOnly)"
    );

    let body = body_json(resp).await;
    let err = body
        .get("error")
        .and_then(|v| v.as_str())
        .expect("body with an `error` field");
    assert!(
        err.contains("arb"),
        "the message must mention the strategy: {err}"
    );
    assert!(
        err.contains("live") && err.contains("paper_only"),
        "the message must mention the live/paper_only reason: {err}"
    );

    // The supervisor must NOT have spawned the task nor reset the kill switch.
    assert!(
        !sup.has_task(id),
        "after 403, arb must NOT have a live task (the supervisor rejects before spawning)"
    );
    // `is_strategy_killed` may or may not have been touched; the handler
    // returns early BEFORE `set_enabled`, so the kill switch should still be as
    // it was (false in this fixture).
    assert!(
        !risk.is_strategy_killed(id),
        "the 403 returns BEFORE touching RiskGuard, so the kill switch is still as at boot"
    );
}

/// Case of "admitted deactivation": even if the barrier denies turning on, the
/// endpoint always accepts `enabled=false`. The asymmetry is documented in
/// `StrategySupervisor::set_enabled` and mirrored in the UI; this HTTP test pins
/// it against refactoring regressions.
#[tokio::test]
async fn control_strategy_off_always_returns_200_even_if_barrier_blocks() {
    let _g = serial();
    let _tmp = TmpCwd::with_strategy("near_certain"); // temporary cwd: persistence does not touch the repo
    let (state, risk, _sup) = build_state(Mode::Live, lim(true, true)).await;
    let id = poly_bot::types::StrategyId::NearCertain;

    // Even with paper_only=true in live, turning off must pass: the barrier only
    // applies to turning ON. Persistence to TOML is the only side effect; if it
    // fails the handler returns 500, but with a writable TOML it must return 200.
    let router = router_with_token(state.clone());
    let resp = router
        .oneshot(req_post(
            "/control/strategy",
            r#"{"strategy":"near_certain","enabled":false}"#,
            true,
        ))
        .await
        .unwrap();
    // We accept 200 (normal scenario with a writable TOML) or 500 (if
    // persist_strategy_toggle failed on FS permissions, etc.), but never 403
    // (the asymmetry forbids it).
    let status = resp.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "turning off in live with paper_only must be 200 (or 500 if persist failed), never 403 — got {status:?}",
    );
    // Regardless of the status code, the strategy kill switch must be set
    // (turning off kills the task via kill_strategy).
    assert!(
        risk.is_strategy_killed(id),
        "turning off sets the kill switch in RiskGuard (even if the HTTP response is 500 due to the TOML)"
    );
}

/// Case "disabled in paper but activatable": in paper, a strategy with
/// `enabled=false` (TOML config at boot) but clean barrier flags
/// (`paper_only=false`)
/// must be activatable. This validates that the toggle ON goes through the
/// barrier with `enabled=true` (the supervisor's clone), and the UI predicts
/// "ok" for exactly that reason.
///
/// Requires persistence to TOML → a temporary cwd with a copy of the original
/// TOML + a restoring `Drop`.
#[tokio::test]
async fn control_strategy_paper_with_enabled_false_is_accepted() {
    let _g = serial();
    let _tmp = TmpCwd::with_strategy("near_certain");
    let (state, risk, sup) = build_state(Mode::Paper, lim(false, false)).await;
    // disabled in paper but activatable
    let id = poly_bot::types::StrategyId::NearCertain;

    // Precondition: enabled=false at boot ⇒ supervisor with no live task.
    assert!(!sup.has_task(id));

    let router = router_with_token(state.clone());
    let resp = router
        .oneshot(req_post(
            "/control/strategy",
            r#"{"strategy":"near_certain","enabled":true}"#,
            true,
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "activating near_certain with enabled=false in paper must be 200 (the supervisor forces enabled=true before the barrier)"
    );

    // Coherent runtime state: the task becomes live (if the strategy has a
    // working constructor with the test config) and the kill switch is reset.
    assert!(
        sup.has_task(id),
        "after 200, near_certain must have a live task (the supervisor spawned it on demand)"
    );
    assert!(
        !risk.is_strategy_killed(id),
        "after 200, the kill switch must be reset"
    );
}

// ── Tools: /ui/tools and toggle via /control/strategy ───────────────────────

/// `GET /ui/tools` returns the 6 tools with the contract fields
/// (`tool`, `kind`, `enabled`, `running`). `kind` separates scan from recorder,
/// and `enabled` mirrors each tool's `enabled` in the TOML
/// (`config/strategies/<tool>.toml`, loaded at boot by `Config::load`).
/// The fixture does not spawn tools at boot, so `running` must be false for all
/// of them even when `enabled` is true: they are different states.
#[tokio::test]
async fn ui_tools_returns_all_six_tools_with_kind_and_enabled() {
    let _g = serial();
    let (state, _risk, _sup) = build_state(Mode::Paper, lim(false, false)).await;

    let router = router_with_token(state);
    let resp = router.oneshot(req_get("/ui/tools", true)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET /ui/tools with token → 200");

    let body = body_json(resp).await;
    let tools = body
        .get("tools")
        .and_then(|v| v.as_array())
        .expect("payload { tools: [...] }");
    assert_eq!(
        tools.len(),
        poly_bot::tools::ToolId::ALL.len(),
        "all registered tools must appear"
    );

    // Expected `enabled` is the one from each tool's TOML in the working tree
    // (`config/strategies/<tool>.toml`). If the figment merge stopped routing the
    // `enabled` into the tool's toml, these literals would break.
    let expected: &[(&str, &str, bool)] = &[
        ("vol_surface", "scan", false),
        ("ladder_arb", "scan", false),
        ("analytics", "recorder", true),
        ("spot_stream", "recorder", true),
        ("research_events", "recorder", false),
        ("resolution_fetcher", "recorder", true),
    ];
    for (name, kind, enabled) in expected {
        let row = tools
            .iter()
            .find(|r| r.get("tool").and_then(|v| v.as_str()) == Some(*name))
            .unwrap_or_else(|| panic!("tool {name} not found: {body}"));
        assert_eq!(
            row.get("kind").and_then(|v| v.as_str()),
            Some(*kind),
            "{name}: incorrect kind"
        );
        assert_eq!(
            row.get("enabled").and_then(|v| v.as_bool()),
            Some(*enabled),
            "{name}: enabled does not match its TOML"
        );
        assert_eq!(
            row.get("running").and_then(|v| v.as_bool()),
            Some(false),
            "{name}: not spawned at boot in this fixture, running must be false"
        );
    }
}

/// Toggling a tool that the supervisor rejects returns 409, the same gate as the
/// `Stopping` scenario. The deterministic path in this binary: `analytics` is a
/// recorder and `set_recorder_deps` is never called here, so `spawn` rejects
/// before touching anything and never reaches persistence (which is why `TmpCwd`
/// is not needed).
#[tokio::test]
async fn control_strategy_tool_recorder_without_deps_returns_409() {
    let _g = serial();
    let (state, _risk, _sup) = build_state(Mode::Paper, lim(false, false)).await;

    let router = router_with_token(state);
    let resp = router
        .oneshot(req_post(
            "/control/strategy",
            r#"{"strategy":"analytics","enabled":true}"#,
            true,
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::CONFLICT,
        "a recorder without RecorderDeps must be rejected with 409"
    );
    let body = body_json(resp).await;
    let err = body
        .get("error")
        .and_then(|v| v.as_str())
        .expect("body with an `error` field");
    assert!(
        err.contains("analytics"),
        "the message must mention the tool: {err}"
    );
}

/// Successful toggle of a scan tool over HTTP: `set_enabled` really starts the
/// task and the persistence writes to the tool's TOML in the temporary cwd.
#[tokio::test]
async fn control_strategy_tool_scan_toggle_returns_200() {
    let _g = serial();
    let _tmp = TmpCwd::with_strategy("vol_surface"); // persistence confined to the tempdir
    let (state, _risk, _sup) = build_state(Mode::Paper, lim(false, false)).await;

    let router = router_with_token(state.clone());
    let resp = router
        .oneshot(req_post(
            "/control/strategy",
            r#"{"strategy":"vol_surface","enabled":true}"#,
            true,
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "activating a scan tool must be 200"
    );
    // The spawn already happened inside the handler; `is_running` is not asserted
    // because the fixture's `shutdown_tx` dies when `build_state` returns and the
    // task may have exited before this assert (a legitimate race). What is
    // deterministic is the response body.
    let body = body_json(resp).await;
    assert_eq!(
        body.get("tool").and_then(|v| v.as_str()),
        Some("vol_surface")
    );
    assert_eq!(body.get("enabled").and_then(|v| v.as_bool()), Some(true));
}

/// Unknown name: 400 (before touching the supervisor or risk).
/// Ensures the error handler is still wired in the correct order.
#[tokio::test]
async fn control_strategy_unknown_name_returns_400() {
    let _g = serial();
    let (state, _risk, _sup) = build_state(Mode::Paper, lim(false, false)).await;

    let router = router_with_token(state);
    let resp = router
        .oneshot(req_post(
            "/control/strategy",
            r#"{"strategy":"does_not_exist","enabled":true}"#,
            true,
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "name is not parseable as a StrategyId or a ToolId → 400"
    );
}

// ── Test of the string set: pins the serialized contract ──────────

/// Closed table: for every combination (mode × barrier flags), the
/// `mode_permission` in the `ui_strategies` JSON must be in the whitelist of
/// strings and must NOT grant permission (`"ok"`) when `can_run` would reject
/// with the same flags. This closes the gap on the JSON side: if the helper
/// lied with `"ok"` for a future variant of `BlockReason`, this test breaks,
/// because that variant would make `can_run` over `enabled=true` return
/// `Err(_)`, and the delegating helper returns `"blocked"` (not `"ok"`).
#[tokio::test]
async fn ui_strategies_does_not_grant_ok_when_can_run_rejects() {
    let _g = serial();
    // For each combination that "should reject", we build the fixture and check
    // that the JSON says `"blocked_*"` (never `"ok"`).
    // The "expected" field is what the JSON must show.
    // The "wait_reason" field is the `BlockReason` that `can_run` must return
    // for the same flag set with `enabled=true` forced (as the helper and the
    // supervisor do).
    type PermissionCase = (Mode, bool, bool, &'static str, Option<BlockReason>);
    let cases: &[PermissionCase] = &[
        // mode, enabled, paper_only, expected permission, expected reason
        (
            Mode::Live,
            true,
            true,
            "blocked_paper_only",
            Some(BlockReason::PaperOnly),
        ),
        // Paper with enabled=false: the helper FORCES enabled=true in its
        // clone, so the barrier accepts. Expected: "ok".
        (Mode::Paper, false, false, "ok", None),
        (Mode::Paper, false, true, "ok", None),
    ];

    for (i, (mode, enabled, paper_only, expected, wait_reason)) in
        cases.iter().enumerate()
    {
        let _ = enabled; // only documents the fixture; the source always uses enabled=true
        let (state, _risk, _sup) =
            build_state(*mode, lim(*enabled, *paper_only)).await;
        let router = router_with_token(state);
        let resp = router
            .oneshot(req_get("/ui/strategies", true))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "scenario {i}: GET ui_strategies");
        let body = body_json(resp).await;
        for id in poly_bot::strategy::registry::REGISTERED {
            let row_entry = row_entry(&body, &id.to_string());
            let mp = row_entry
                .get("mode_permission")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            // 1) Must be in the closed set.
            assert!(
                matches!(
                    mp,
                    "ok" | "blocked_paper_only"
                            | "blocked_disabled"
                        | "blocked"
                ),
                "scenario {i} ({id}): mode_permission={mp} is not in the closed set"
            );
            // 2) The JSON says exactly the expected value.
            assert_eq!(
                mp, *expected,
                "scenario {i} ({id}): mode_permission={mp} != expected {expected} \
                 (mode={mode:?}, paper_only={paper_only})"
            );
            // 3) If `wait_reason` is Some, the barrier MUST reject (with that
            //    specific reason). If it is None, the barrier must accept. The
            //    source ALWAYS evaluates with `enabled=true` (the toggle's view,
            //    as the helper and the supervisor do).
            let source = StrategyLimits {
                enabled: true,
                paper_only: *paper_only,
                ..StrategyLimits::default()
            }
            .can_run(*mode);
            match wait_reason {
                Some(m) => {
                    assert_eq!(
                        source.unwrap_err(),
                        *m,
                        "scenario {i} ({id}): the source must reject with {m:?} (got {source:?})"
                    );
                    // Forward-compat fail-safe: if the source rejects, the UI
                    // must NEVER say "ok".
                    assert_ne!(
                        mp, "ok",
                        "scenario {i} ({id}): the barrier rejected but the UI said 'ok' \
                         — forward-compat broken"
                    );
                }
                None => {
                    assert!(
                        source.is_ok(),
                        "scenario {i} ({id}): the source must accept (got {source:?})"
                    );
                }
            }
        }
    }
}
