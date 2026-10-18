//! Bearer authentication middleware.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use subtle::ConstantTimeEq;

/// Normalizes the path by removing empty segments and `..`, as defense in
/// depth so the exact comparisons below (`/healthz`,
/// `/metrics`, `/paper/*`) cannot be bypassed with a path such as
/// `/control/kill/../../healthz`.
fn normalize_path(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    format!("/{}", parts.join("/"))
}

/// Middleware Bearer.
///
/// - Token configured: requires `Authorization: Bearer <token>` on all endpoints
///   except `/healthz` and `/metrics`. The SPA's static files are outside
///   this middleware by router construction (`serve()` applies `.layer()`
///   BEFORE merging `static_router`, see `mod.rs`): they never reach `bearer_auth`,
///   so they are unconditionally public without needing an exceptions list.
///   (An earlier version of this middleware used an allowlist of literal routes
///   for the vanilla dashboard's HTML/assets, which went stale
///   with every new asset. The current SPA serves assets with Vite-hashed names,
///   so a literal list would no longer be viable anyway.)
/// - Empty token + paper: all endpoints are accessible without authentication.
/// - Empty token + live: only `/paper/*` passes; everything else returns 503.
pub async fn bearer_auth(
    token: Arc<String>,
    is_paper: bool,
    req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let raw_path = req.uri().path();
    let path = normalize_path(raw_path);
    let path = path.as_str();

    if path == "/healthz" || path == "/metrics" {
        return Ok(next.run(req).await);
    }

    if token.is_empty() {
        if is_paper {
            return Ok(next.run(req).await);
        }
        // This exception lets ALL of `/paper/*` through without a token in live
        // (including `/paper/deposit`, a POST that mutates `deposits_total` and therefore
        // `cash`/`cash_warning`). It stays this way because the READ endpoints
        // (`paper_account`, `paper_pnl`, `paper_positions`) must remain
        // accessible for quick diagnostics without a token in live; the real fix
        // for the `/paper/deposit` hole lives in `handlers::paper_deposit`, which
        // rejects the whole call if `Mode != Paper` (see the comment there).
        if path.starts_with("/paper/") {
            return Ok(next.run(req).await);
        }
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let provided = req
        .headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match provided {
        Some(p) if p.as_bytes().ct_eq(token.as_bytes()).into() => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_path_collapses_traversal() {
        assert_eq!(normalize_path("/ui/../control/kill"), "/control/kill");
        assert_eq!(normalize_path("/ui/"), "/ui");
        assert_eq!(normalize_path("//ui//data"), "/ui/data");
    }
}
