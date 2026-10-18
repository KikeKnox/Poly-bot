//! Shared state of the API server — V3.

use metrics_exporter_prometheus::PrometheusHandle;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{oneshot, watch, Mutex};

use crate::{
    config::Config,
    accounting::mark_price::MarkPriceCache,
    risk::RiskGuard,
    state::StateStore,
    strategy::StrategySupervisor,
    tools::ToolSupervisor,
    types::{Mode, VolSnapshot},
};

/// State injected into all the axum handlers.
#[derive(Clone)]
pub struct ApiState {
    pub config: Arc<Config>,
    pub risk: Arc<RiskGuard>,
    pub state: Arc<StateStore>,
    pub token: Arc<String>,
    pub mode: Mode,
    pub shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    pub prometheus_handle: Arc<PrometheusHandle>,
    /// USDC wallet balance: in paper = initial capital (static),
    /// in live = real CLOB balance polled every 30s.
    pub wallet_balance: watch::Receiver<Option<Decimal>>,
    /// Scan-only tools (vol_surface, ladder_arb), outside RiskGuard,
    /// hot toggle via real spawn/stop (no kill switch to fake).
    pub tools: Arc<ToolSupervisor>,
    /// On-demand spawn of a strategy with `enabled=false` at
    /// boot; see the doc comment of `StrategySupervisor`.
    pub strategies: Arc<StrategySupervisor>,
    /// In-memory alert log for the UI's "Recent alerts" panel.
    pub alert_log: Arc<crate::api::alert_log::AlertLog>,
    /// Short/long σ per asset: a copy of the mapping of receivers that
    /// the strategies also consume. `Arc<HashMap<...>>` instead of cloning
    /// the whole `HashMap` on every `ApiState::clone()` per request; each
    /// inner `watch::Receiver` is still independently cloneable.
    pub vol_rxs: Arc<HashMap<String, watch::Receiver<VolSnapshot>>>,
    /// Directional exposure ledger per bracket: the same source the real
    /// executor uses to gate buys. `/ui/risk` uses it instead of
    /// `state.total_open_cost()` because the latter only counts positions already
    /// filled in SQLite, understating exposure while live GTDs exist.
    pub ledger: Arc<crate::coordination::BracketLedger>,
    /// Mark-to-market price cache for real equity; see the doc comment
    /// of `src/accounting/mark_price.rs`. Refreshed by a background task; the handlers
    /// only READ from here and never trigger network I/O on their own.
    pub mark_price: Arc<MarkPriceCache>,
}
