//! Servidor API axum — V3.

pub mod alert_log;
pub mod auth;
pub mod handlers;
pub mod prom_parse;
pub mod state;
pub mod strategy_funnel;

pub use state::ApiState;

use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use std::path::Path;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::services::{ServeDir, ServeFile};

use crate::types::Mode;

/// Router for the SPA's static files only, kept separate from `serve()` so it
/// can be tested with `tower::ServiceExt::oneshot` without starting a TcpListener.
pub fn static_router(dist_dir: impl AsRef<Path>) -> Router {
    let dist_dir = dist_dir.as_ref();
    let index = dist_dir.join("index.html");
    let serve_dir = ServeDir::new(dist_dir).fallback(ServeFile::new(index));
    Router::new().nest_service("/ui", serve_dir)
}

/// Wraps `json_routes` (routes that require `ApiState`, without statics) with
/// the auth middleware and merges the statics AFTER it. This function is the
/// ONLY one responsible for the security invariant "statics public by
/// construction, JSON protected": the `.layer()` here only wraps the
/// routes already registered in `json_routes`, so `static_router` (merged
/// afterwards) never goes through `bearer_auth`. Extracted from `build_router` so
/// this invariant can be tested (`tests/api_auth.rs`) without needing a
/// full `ApiState`: the test passes it a router of decoy routes instead
/// of the real routes, but exercises the same assembly code.
pub fn wrap_auth_and_static(
    json_routes: Router,
    token: Arc<String>,
    is_paper: bool,
    dist_dir: impl AsRef<Path>,
) -> Router {
    json_routes
        .layer(middleware::from_fn(move |req, next| {
            auth::bearer_auth(token.clone(), is_paper, req, next)
        }))
        .merge(static_router(dist_dir))
}

/// Builds the full router: registers the JSON routes with `ApiState` and
/// delegates to `wrap_auth_and_static` for the auth-before-statics ordering.
/// Kept separate from `serve()` so it can be tested without starting a `TcpListener`.
pub fn build_router(api_state: ApiState) -> Router {
    let token = api_state.token.clone();
    let is_paper = api_state.mode == Mode::Paper;

    let json_routes = Router::new()
        // Diagnostics
        .route("/healthz", get(handlers::healthz))
        .route("/status", get(handlers::status))
        .route("/metrics", get(handlers::metrics_summary))
        .route("/recent", get(handlers::recent))
        .route("/settings", get(handlers::settings))
        // Control
        .route("/control/strategy", post(handlers::control_strategy))
        .route(
            "/control/strategy_limits",
            post(handlers::control_strategy_limits),
        )
        .route(
            "/control/exposure_limit",
            post(handlers::control_exposure_limit),
        )
        .route("/control/kill", post(handlers::control_kill))
        .route(
            "/control/kill_switches",
            post(handlers::control_kill_switches),
        )
        .route(
            "/control/strategy_kill",
            post(handlers::control_strategy_kill),
        )
        .route("/control/blacklist", post(handlers::control_blacklist))
        .route("/control/shutdown", post(handlers::control_shutdown))
        // Cuenta / PnL
        .route("/paper/account", get(handlers::paper_account))
        .route("/paper/deposit", post(handlers::paper_deposit))
        .route("/paper/pnl", get(handlers::paper_pnl))
        .route("/paper/positions", get(handlers::paper_positions))
        .route("/pnl_strat", get(handlers::pnl_strat))
        // UI: data endpoints consumed by the SPA (the static files
        // are served via static_router further below)
        .route("/ui/strategies", get(handlers::ui_strategies))
        .route("/ui/strategies/edge", get(handlers::ui_strategies_edge))
        .route("/ui/tools", get(handlers::ui_tools))
        .route("/ui/data", get(handlers::ui_data))
        .route("/ui/history", get(handlers::ui_history))
        .route("/ui/pnl_series", get(handlers::ui_pnl_series))
        .route("/ui/pnl_analytics", get(handlers::ui_pnl_analytics))
        .route("/ui/metrics_stats", get(handlers::ui_metrics_stats))
        .route("/ui/transactions", get(handlers::ui_transactions))
        .route("/ui/alerts", get(handlers::ui_alerts))
        .route("/ui/strategy_funnel", get(handlers::ui_strategy_funnel))
        .route("/ui/rl_learner", get(handlers::ui_rl_learner))
        .route("/control/rl_learner", post(handlers::control_rl_learner))
        .route("/ui/risk", get(handlers::ui_risk))
        .route("/ui/market_health", get(handlers::ui_market_health))
        .route("/ui/money", get(handlers::ui_money))
        .with_state(api_state);

    wrap_auth_and_static(
        json_routes,
        token,
        is_paper,
        std::env::var("HRF_UI_DIST_DIR").unwrap_or_else(|_| "src/api/static/dist".to_string()),
    )
}

pub async fn serve(api_state: ApiState, listener: Option<TcpListener>) {
    let bind_addr = format!(
        "{}:{}",
        api_state.config.api.bind, api_state.config.api.port
    );
    let router = build_router(api_state);

    let listener = match listener {
        Some(l) => l,
        None => {
            let addr = bind_addr;
            match TcpListener::bind(&addr).await {
                Ok(l) => {
                    tracing::info!("api: listening on http://{addr}/ui/");
                    l
                }
                Err(e) => {
                    tracing::error!("api: bind {addr} failed: {e}");
                    return;
                }
            }
        }
    };

    if let Err(e) = axum::serve(listener, router).await {
        tracing::error!("api: serve error: {e}");
    }
}
