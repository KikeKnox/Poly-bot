// ── Core of the strategy bot ──────────────────────────────────────────────
pub mod config;
pub mod coordination;
pub mod fair_value;
pub mod market_feed;
pub mod metrics;
pub mod oracle;
pub mod order;
pub mod risk;
pub mod state;
pub mod strategy;
pub mod time;
pub mod types;

pub mod accounting; // fills, markouts, mark price
pub mod api;
pub mod startup;
pub mod clob_sdk;
pub mod log_buffer;
pub mod research; // research data layer (contract, tape, labels)
pub mod rl; // RL predictor: contract, policy, episodes and online learner
pub mod settlement; // orphans, redeem, post-close, reconciliation
pub mod telegram;

// ── In-process tools (never emit orders; toggled by ToolId) ──────────────────
pub mod tools;

// ── Offline: backtest and replay ─────────────────────────────────────────────
pub mod backtest;
pub mod replay;

// Short aliases used by the research modules (model_observer, research_replay,
// hrf_research_env) and their tests.
pub use research::contract as research_contract;
pub use research::events as research_events;
pub use research::labels as research_labels;

// Research: model observer and research-tape replay environment.
pub mod model_observer;
pub mod research_replay;

/// Browser-like User-Agent shared by every client that calls Polymarket's public
/// HTTP endpoints (Gamma, CLOB, Data API: market_feed, follower, orphan, mark_price).
/// Their CDN has rejected the default UA of some HTTP libraries with 403.
pub(crate) const POLY_USER_AGENT: &str =
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";
