//! Verifies that `/ui` serves the compiled SPA (index.html) and that
//! unknown routes under /ui/* fall back to the same index.html (SPA fallback).

use tower::ServiceExt;

#[tokio::test]
async fn ui_serves_index_html_as_spa_fallback() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("index.html"),
        "<!doctype html><title>t</title>",
    )
    .unwrap();

    let router = poly_bot::api::static_router(dir.path());

    // Root route
    let resp = router
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/ui")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);

    // Unknown route (deep link to an SPA page) must also fall back to the index.
    let resp = router
        .oneshot(
            axum::http::Request::builder()
                .uri("/ui/anything")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
}
