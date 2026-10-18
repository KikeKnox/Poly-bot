//! Verifies that the `bearer_auth` middleware requires a token on the `/ui/*`
//! routes that serve operational data (they are not public SPA statics).
//!
//! Protection no longer depends on an exception list in `auth.rs` but on the
//! ORDER in which the router is assembled: `poly_bot::api::wrap_auth_and_static`
//! applies `.layer(auth)` to the JSON routes BEFORE merging `static_router`.
//! These tests call that real function (not a local reimplementation) so a
//! future regression in the ordering in `mod.rs` breaks them; previously,
//! `router_with_auth` reimplemented the ordering by hand and would not have
//! detected anything if `build_router` were reordered incorrectly.
//!
//! Original regression: `/ui/alerts` was left out of the old exclusion list
//! `is_public_ui`, allowing alerts (including strategy kill reasons) to be
//! read without a token on a non-loopback bind. `/ui/strategy_funnel`,
//! `/ui/risk`, `/ui/market_health` and `/ui/pnl_analytics` have the same
//! operational sensitivity and were added to this file for the same reason.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    routing::get,
    Router,
};
use tower::ServiceExt;

async fn dummy_handler() -> &'static str {
    "ok"
}

/// Builds the real production router (`wrap_auth_and_static`) over decoy
/// JSON routes: it exercises the same assembly code as `build_router`
/// without needing a full `ApiState`.
fn router_with_auth(token: &str, is_paper: bool) -> Router {
    let json_routes = Router::new()
        .route("/ui/alerts", get(dummy_handler))
        .route("/ui/data", get(dummy_handler))
        .route("/ui/strategy_funnel", get(dummy_handler))
        .route("/ui/risk", get(dummy_handler))
        .route("/ui/market_health", get(dummy_handler))
        .route("/ui/pnl_analytics", get(dummy_handler));

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("index.html"),
        "<!doctype html><title>t</title>",
    )
    .unwrap();
    // The tempdir is intentionally leaked: it lives as long as the test
    // process, enough for ServeDir to serve it during the request; it avoids
    // a premature `Drop` if it were returned together with the router.
    let dir_path = Box::leak(Box::new(dir)).path().to_path_buf();

    poly_bot::api::wrap_auth_and_static(json_routes, Arc::new(token.to_string()), is_paper, dir_path)
}

/// Anchors the security invariant: the SPA statics sit outside the auth
/// middleware by construction, even with a token configured and without
/// sending `Authorization`. If someone merges `static_router` BEFORE
/// `.layer(auth)` (the bug this round fixed), this test would fail with 401
/// instead of 200.
#[tokio::test]
async fn static_files_without_token_return_200_with_configured_token() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(Request::builder().uri("/ui").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn ui_alerts_without_token_returns_401_with_configured_token() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/alerts")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn ui_alerts_with_correct_token_returns_200() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/alerts")
                .header("authorization", "Bearer test-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

/// Same behavior as its sibling path `/ui/data`, to record that both are
/// protected consistently.
#[tokio::test]
async fn ui_data_without_token_returns_401_with_configured_token() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/data")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// `/ui/strategy_funnel` exposes internal skip counters per strategy; same
/// sensitivity as `/ui/alerts`, so it must require a token.
#[tokio::test]
async fn ui_strategy_funnel_without_token_returns_401_with_configured_token() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/strategy_funnel?strategy=near_certain")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn ui_strategy_funnel_with_correct_token_returns_200() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/strategy_funnel?strategy=near_certain")
                .header("authorization", "Bearer test-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

/// `/ui/risk` exposes kill reasons and exposure; same sensitivity as
/// `/ui/alerts` and `/ui/strategy_funnel`, so it must require a token.
#[tokio::test]
async fn ui_risk_without_token_returns_401_with_configured_token() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/risk")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn ui_risk_with_correct_token_returns_200() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/risk")
                .header("authorization", "Bearer test-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

/// `/ui/market_health` exposes short/long sigma per asset; it must require
/// a token.
#[tokio::test]
async fn ui_market_health_without_token_returns_401_with_configured_token() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/market_health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn ui_market_health_with_correct_token_returns_200() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/market_health")
                .header("authorization", "Bearer test-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

/// `/ui/pnl_analytics` exposes maximum drawdown, hourly distribution and
/// average markout per strategy; it must require a token.

#[tokio::test]
async fn ui_pnl_analytics_without_token_returns_401_with_configured_token() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/pnl_analytics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn ui_pnl_analytics_with_correct_token_returns_200() {
    let router = router_with_auth("test-secret", false);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/ui/pnl_analytics")
                .header("authorization", "Bearer test-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}
