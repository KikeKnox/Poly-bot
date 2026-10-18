//! `Tool` trait + common runner for observation (scan-only) tools.
//!
//! A tool NEVER emits `OrderIntent`: it only observes books/vol and persists
//! `ScanObs` into `scan_observations` via `scan_tx`. Unlike a `Strategy`
//! (see `crate::strategy`), it does not go through `RiskGuard` — no kill switch, no rate
//! limit, no exposure cap — because it can never place an order.
//!
//! Implemented tools:
//! - `vol_surface` — VS: cross-expiry implied vol divergence
//! - `ladder_arb`  — ladder arbitrage scan between brackets of the same asset
//!
//! Migrated from `crate::strategy` (they used to implement `Strategy` but
//! always returned `vec![]` — the `Strategy` trait was not the right
//! abstraction for a pure observer). The `LadderArb`/`VolSurface` variants were
//! removed from `StrategyId` (they never emitted orders, so they never
//! appeared in `gtd_orders`/`positions`/`fills`; `scan_observations` only
//! stores the string via `ToolId::Display`, which keeps the same text
//! "vol_surface"/"ladder_arb" — data backward compatibility is preserved).

pub mod model;
pub mod recorders;
pub mod scan;

pub use scan::ladder_arb::LadderArb;
pub use scan::vol_surface::VolSurface;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex as StdMutex, OnceLock},
    time::Duration,
};

use tokio::sync::{broadcast, mpsc, watch};
use tracing::debug;

use crate::{
    config::Config,
    market_feed::registry::MarketRegistry,
    research::events::Hub,
    state::StateStore,
    tools::scan::sink::ScanObs,
    types::{BookSnapshot, VolSnapshot},
};

// ── Tool identifier ────────────────────────────────────────────────────

/// Identifier of a tool (scan-only observer, or research
/// recorder — see [`ToolKind`]).
///
/// Separate from `StrategyId`: tools do not go through `RiskGuard` (no
/// kill switch, no limits), so mixing them into `StrategyId::ALL` forced
/// us to feign a kill switch that never did anything real.
///
/// The recorders (`analytics`, `spot_stream`, `research_events`,
/// `resolution_fetcher`) joined `ToolId` — before, they started only once
/// at boot from `main.rs` according to their `enabled` in `general.toml` (now
/// `enabled` lives in `config/strategies/<tool>.toml`, which is where the hot
/// toggle persists it), with no hot toggle and no orderly stop (an
/// `abort()` halfway through a flush). The name of
/// each variant (`Display`/`FromStr`) is that of its section in the TOML so
/// `persist_strategy_toggle` keeps working unchanged.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ToolId {
    VolSurface,
    LadderArb,
    Analytics,
    SpotStream,
    ResearchEvents,
    ResolutionFetcher,
    /// Tool of the numeric model (`src/tools/model/`). Producer of
    /// P(YES) probabilities; the `ModelBtc5m` strategy is the one that translates them
    /// into `OrderIntent`. Follows the `ResearchEvents` pattern: starts via
    /// `ToolSupervisor`, never emits orders.
    ModelTool,
}

/// Scan-only (no state to flush: `stop()` can abort without ceremony)
/// vs recorder (has a writer with its own buffer/thread: `stop()` MUST
/// wait for the flush before aborting — see `ToolSupervisor::stop`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolKind {
    Scan,
    Recorder,
}

impl std::fmt::Display for ToolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ToolKind::Scan => "scan",
            ToolKind::Recorder => "recorder",
        };
        write!(f, "{s}")
    }
}

impl ToolId {
    pub const ALL: &'static [Self] = &[
        Self::VolSurface,
        Self::LadderArb,
        Self::Analytics,
        Self::SpotStream,
        Self::ResearchEvents,
        Self::ResolutionFetcher,
        Self::ModelTool,
    ];

    pub fn kind(self) -> ToolKind {
        match self {
            ToolId::VolSurface | ToolId::LadderArb => ToolKind::Scan,
            ToolId::Analytics
            | ToolId::SpotStream
            | ToolId::ResearchEvents
            | ToolId::ResolutionFetcher
            | ToolId::ModelTool => ToolKind::Recorder,
        }
    }
}

impl std::fmt::Display for ToolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ToolId::VolSurface => "vol_surface",
            ToolId::LadderArb => "ladder_arb",
            ToolId::Analytics => "analytics",
            ToolId::SpotStream => "spot_stream",
            ToolId::ResearchEvents => "research_events",
            ToolId::ResolutionFetcher => "resolution_fetcher",
            ToolId::ModelTool => "model_tool",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for ToolId {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "vol_surface" => Ok(ToolId::VolSurface),
            "ladder_arb" => Ok(ToolId::LadderArb),
            "analytics" => Ok(ToolId::Analytics),
            "spot_stream" => Ok(ToolId::SpotStream),
            "research_events" => Ok(ToolId::ResearchEvents),
            "resolution_fetcher" => Ok(ToolId::ResolutionFetcher),
            "model_tool" => Ok(ToolId::ModelTool),
            _ => Err(()),
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Pure observation tool. Implementations in `tools::scan::vol_surface`,
/// `tools::scan::ladder_arb`.
pub trait Tool: Send + 'static {
    /// Unique identifier (for logs and control surfaces).
    fn id(&self) -> ToolId;

    /// Processes a book. No return: a tool never places orders,
    /// it only updates its internal state and persists observations via `scan_tx`.
    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot);
}

// ── Common runner ─────────────────────────────────────────────────────────────

/// Execution context of a tool (what the supervisor injects).
///
/// Deliberately smaller than `StrategyCtx`: no `intent_tx` (never emits
/// orders), no `fill_rx` (does not manage positions), no `risk` (no kill
/// switch). `scan_tx` lives here instead of being injected by constructor
/// as before — fixes the inconsistency of `VolSurface::new`/`LadderArb::new`
/// receiving the sender separately from the rest of the execution context.
pub struct ToolCtx {
    pub book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    pub vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    pub scan_tx: mpsc::UnboundedSender<ScanObs>,
    pub shutdown: watch::Receiver<bool>,
    pub max_book_age: Duration,
}

/// Main loop of the tool runner. Spawn in one task per tool.
///
/// Same coalesce + staleness guard as `strategy::run`, but with no kill switch
/// (tools have none) and no sending of intents/processing of fills.
///
/// Exits when:
/// - `shutdown` emits `true`
/// - The `book_rx` channel is closed (no more feeds)
///
/// `shutdown.changed()` runs in a `select!` against `drain_latest` — without
/// this, a tool with no books in flight stayed blocked indefinitely on the
/// `.await` of `drain_latest` and never saw the signal (the check further down,
/// inside the `for book in books`, is only reached once a book HAS arrived).
/// `ToolSupervisor::stop` needs the signal alone to be enough for the
/// task to exit — otherwise, every `stop()` of a tool with no traffic would wait the
/// full `STOP_TIMEOUT` and end up aborting instead of joining.
pub async fn run_tool(mut tool: Box<dyn Tool>, mut ctx: ToolCtx) {
    let id = tool.id();

    tracing::info!(%id, "tool runner: starting");

    loop {
        let books = tokio::select! {
            biased;
            changed = ctx.shutdown.changed() => {
                if changed.is_err() || *ctx.shutdown.borrow() {
                    tracing::info!(%id, "tool runner: shutdown received");
                    return;
                }
                continue;
            }
            drained = drain_latest(&mut ctx.book_rx) => match drained {
                Some(v) => v,
                None => {
                    tracing::warn!(%id, "tool runner: books channel closed, exiting");
                    return;
                }
            },
        };

        for book in books {
            if *ctx.shutdown.borrow() {
                tracing::info!(%id, "tool runner: shutdown received");
                return;
            }

            // Staleness guard (same criterion as in strategy::run): do not evaluate
            // over a stale book (broadcast backlog, WS gap).
            if let Some(recv) = book.recv_instant {
                if recv.elapsed() > ctx.max_book_age {
                    metrics::counter!("hrf_tool_book_stale_skipped_total", "tool" => id.to_string())
                        .increment(1);
                    continue;
                }
            }

            // Vol snapshot of the book's asset. Same criterion as strategy::run:
            // no cross-asset fallback. Empty map (tests / no oracle) → default.
            // Canonical parsing (lowercase + empty slug → skip), same as the runner.
            let Some(asset) = crate::oracle::asset_from_slug(&book.slug) else {
                continue;
            };
            let vol = match ctx.vol_rxs.get(asset.as_str()) {
                Some(rx) => rx.borrow().clone(),
                None if ctx.vol_rxs.is_empty() => VolSnapshot::default(),
                None => {
                    metrics::counter!(
                        "hrf_tool_vol_asset_missing_total",
                        "asset" => asset.clone()
                    )
                    .increment(1);
                    continue;
                }
            };

            tool.on_book(&book, &vol);

            tokio::task::yield_now().await;
        }
    }
}

/// Drains the broadcast channel and returns the last book **of each market**.
/// Copy of `strategy::drain_latest` — deliberately duplicated: `Tool` and
/// `Strategy` are independent runners, not coupled through a
/// private function shared between modules. Same intra-cid coalesce fix:
/// books of different markets are kept.
async fn drain_latest(
    rx: &mut broadcast::Receiver<Arc<BookSnapshot>>,
) -> Option<Vec<Arc<BookSnapshot>>> {
    let mut latest: Vec<Arc<BookSnapshot>> = Vec::new();
    let mut pos: HashMap<String, usize> = HashMap::new();

    loop {
        match rx.try_recv() {
            Ok(book) => match pos.get(book.market_condition_id.as_str()) {
                Some(&i) => {
                    metrics::counter!("hrf_tool_book_coalesce_total").increment(1);
                    latest[i] = book;
                }
                None => {
                    pos.insert(book.market_condition_id.clone(), latest.len());
                    latest.push(book);
                }
            },
            Err(broadcast::error::TryRecvError::Empty) => {
                if !latest.is_empty() {
                    return Some(latest);
                }
                // Same as in strategy::drain_latest: a `Lagged` here is
                // transient (burst between the empty try_recv and the await) —
                // it is not a channel close; keep draining.
                match rx.recv().await {
                    Ok(b) => {
                        pos.insert(b.market_condition_id.clone(), 0);
                        latest.push(b);
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        debug!("tool runner: lagged {n} books at the await (coalesce in effect)");
                        metrics::counter!("hrf_tool_book_lag_total").increment(n);
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(n)) => {
                debug!("tool runner: lagged {n} books (coalesce in effect)");
                metrics::counter!("hrf_tool_book_lag_total").increment(n);
            }
            Err(broadcast::error::TryRecvError::Closed) => {
                return if latest.is_empty() {
                    None
                } else {
                    Some(latest)
                };
            }
        }
    }
}

// ── Recorder dependencies ─────────────────────────────────────────────────────

/// What a `ToolId::kind() == Recorder` needs to build its task and that
/// `ToolSupervisor::new` does not receive (its signature does not change — it is used by
/// `tests/api_strategy_auth_http.rs`). It is set with
/// [`ToolSupervisor::set_recorder_deps`] before `spawn_enabled_at_boot`; the
/// rest of a recorder's dependencies (its config, `book_tx`, `vol_rxs`, the
/// mode) are already in `ToolSupervisor` and are not duplicated here.
pub struct RecorderDeps {
    pub registry: Arc<MarketRegistry>,
    pub state: Arc<StateStore>,
    pub research_hub: Arc<Hub>,
}

// ── Supervisor (hot spawn/stop) ───────────────────────────────────────────────

/// Deadline `stop()` waits for the result of the real stop before
/// returning `Err` without having confirmed the shutdown. The real stop runs in
/// a background task independent of this deadline (see the doc comment of
/// `stop()`) — exceeding this timeout does NOT cancel it, it only says "it has not
/// finished yet".
const STOP_TIMEOUT: Duration = Duration::from_secs(10);

/// Task in the `running` map:
/// - `Starting`: `spawn()` is building the task — for a recorder
///   it may take a while (blocking I/O via `spawn_blocking`, see
///   `spawn_recorder_task`). No `running` lock is ever held
///   under `Starting`: it is marked, released, the task is built, and it is retaken to
///   insert `Running` or clean up.
/// - `Running`: live task.
/// - `Stopping`: a background task is awaiting the real join and will clean up
///   the slot when it finishes — see the doc comment of `stop()`.
///
/// A `spawn`/`stop` on an id already in `Starting`/`Stopping` is rejected:
/// never two writers over the same SQLite/JSONL, never two concurrent
/// stops of the same task.
enum SlotState {
    Starting,
    Running(watch::Sender<bool>, tokio::task::JoinHandle<()>),
    Stopping,
}

/// Manages the lifecycle of the tools' tasks: initial startup
/// gated by `config.<tool>.enabled` (same pattern as `spawn_strategy!`) and
/// hot toggle (API/Telegram) that really starts/stops the task —
/// tools have no kill switch to "pause" them instead.
pub struct ToolSupervisor {
    book_tx: broadcast::Sender<Arc<BookSnapshot>>,
    vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    scan_tx: mpsc::UnboundedSender<ScanObs>,
    shutdown: watch::Receiver<bool>,
    max_book_age: Duration,
    config: Arc<Config>,
    running: StdMutex<HashMap<ToolId, SlotState>>,
    /// CURRENT desired state per tool, for `/status`/`/ui/tools` — seeded
    /// from `config_enabled` in `new()` and updated by
    /// [`Self::set_desired_enabled`] when a hot toggle succeeded
    /// AND the caller confirmed it persisted it (the supervisor knows nothing about TOML;
    /// without this map, the API ALWAYS reported the boot `Arc<Config>`
    /// snapshot, blind to any later toggle).
    desired_enabled: StdMutex<HashMap<ToolId, bool>>,
    /// Recorder dependencies — an instance field, not a `static`
    /// (a process-global `OnceLock` used to mean that at most ONE
    /// test in the whole binary could set them). It is still a `OnceLock`
    /// because it only makes sense to set them once per supervisor.
    recorder_deps: OnceLock<RecorderDeps>,
    /// Shared `ModelPrediction` channel between `ToolId::ModelTool` and the
    /// `ModelBtc5m` strategy. It is injected AFTER building the supervisor
    /// (via [`Self::set_model_prediction_tx`]) because its size must
    /// match the one the strategies expect in `BuildCtx`. Without
    /// this injection, `spawn(ModelTool)` is rejected with a clear error
    /// (never a panic, never silently starts without predictions).
    model_prediction_tx: OnceLock<Arc<broadcast::Sender<crate::tools::model::ModelPrediction>>>,
    /// Writer + handle of the live model tool. We keep it to do an
    /// orderly shutdown (drain JSONL, no abort) in `stop()` —
    /// see [`Self::stop`] and [`Self::shutdown_all`].
    model_handle: StdMutex<Option<crate::tools::model::ToolHandle>>,
}

impl ToolSupervisor {
    pub fn new(
        book_tx: broadcast::Sender<Arc<BookSnapshot>>,
        vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
        scan_tx: mpsc::UnboundedSender<ScanObs>,
        shutdown: watch::Receiver<bool>,
        max_book_age: Duration,
        config: Arc<Config>,
    ) -> Arc<Self> {
        let sup = Self {
            book_tx,
            vol_rxs,
            scan_tx,
            shutdown,
            max_book_age,
            config,
            running: StdMutex::new(HashMap::new()),
            desired_enabled: StdMutex::new(HashMap::new()),
            recorder_deps: OnceLock::new(),
            model_prediction_tx: OnceLock::new(),
            model_handle: StdMutex::new(None),
        };
        {
            let mut desired = sup.desired_enabled.lock().unwrap();
            for &id in ToolId::ALL {
                desired.insert(id, sup.config_enabled(id));
            }
        }
        Arc::new(sup)
    }

    /// Sets the recorders' dependencies — call once from
    /// `main.rs`, before `spawn_enabled_at_boot`. Without this call, `spawn`
    /// of a `ToolId::kind() == Recorder` is rejected (warn, no panic: a missing
    /// recorder must not bring down startup). A second attempt
    /// is also ignored with a `warn!` — there is no real use case for
    /// replacing them hot.
    pub fn set_recorder_deps(&self, deps: RecorderDeps) {
        if self.recorder_deps.set(deps).is_err() {
            tracing::warn!("tool supervisor: set_recorder_deps called more than once — ignored");
        }
    }

    /// Sets the `broadcast::Sender<ModelPrediction>` that `ToolId::ModelTool`
    /// emits and the `ModelBtc5m` strategy consumes. Call ONCE from
    /// `main.rs` before `spawn_enabled_at_boot` or before any
    /// `set_enabled(ModelTool, true)`. The `Arc` is shared with
    /// `BuildCtx.model_prediction_tx` of the strategy supervisor — they are
    /// clones of the same sender, not separate channels.
    ///
    /// If not set, `spawn(ModelTool)` is rejected with an explicit `Err`
    /// (never a panic, never silently starts). The supervisor can
    /// remain useful for all the other tools without this injection.
    pub fn set_model_prediction_tx(
        &self,
        tx: Arc<broadcast::Sender<crate::tools::model::ModelPrediction>>,
    ) {
        if self.model_prediction_tx.set(tx).is_err() {
            tracing::warn!(
                "tool supervisor: set_model_prediction_tx called more than once — ignored"
            );
        }
    }

    /// Returns the `Arc<Sender>` set by [`Self::set_model_prediction_tx`].
    /// Useful so `main.rs` can populate `BuildCtx.model_prediction_tx`
    /// from the same sender the tool consumes, without desynchronizing.
    pub fn model_prediction_tx(
        &self,
    ) -> Option<Arc<broadcast::Sender<crate::tools::model::ModelPrediction>>> {
        self.model_prediction_tx.get().cloned()
    }

    /// `enabled` from the tool's TOML AT BOOT (`config/strategies/<tool>.toml`
    /// — see `Config::load`): the snapshot the process started with. It is
    /// used by `spawn_enabled_at_boot` and seeds `desired_enabled` in `new()`;
    /// for the CURRENT state (which does reflect the hot toggles already
    /// persisted) use [`Self::enabled`] — the API/UI must read that one, not
    /// this one.
    pub fn config_enabled(&self, id: ToolId) -> bool {
        match id {
            ToolId::VolSurface => self.config.vol_surface.limits.enabled,
            ToolId::LadderArb => self.config.ladder_arb.limits.enabled,
            ToolId::Analytics => self.config.analytics.enabled,
            ToolId::SpotStream => self.config.spot_stream.enabled,
            ToolId::ResearchEvents => self.config.research_events.enabled,
            ToolId::ResolutionFetcher => self.config.resolution_fetcher.enabled,
            ToolId::ModelTool => self.config.model_tool.enabled,
        }
    }

    /// CURRENT desired state — what `/status` and `/ui/tools` report.
    pub fn enabled(&self, id: ToolId) -> bool {
        self.desired_enabled
            .lock()
            .unwrap()
            .get(&id)
            .copied()
            .unwrap_or(false)
    }

    /// The caller (API/Telegram) calls it after successfully persisting a toggle
    /// — the supervisor knows nothing about TOML, it only reflects what the caller
    /// confirms will survive a restart. See `api::handlers::control_strategy`.
    pub fn set_desired_enabled(&self, id: ToolId, enabled: bool) {
        self.desired_enabled.lock().unwrap().insert(id, enabled);
    }

    fn build_scan_tool(&self, id: ToolId) -> Box<dyn Tool> {
        match id {
            ToolId::VolSurface => Box::new(VolSurface::new(
                self.config.vol_surface.clone(),
                self.scan_tx.clone(),
            )),
            ToolId::LadderArb => Box::new(LadderArb::new(
                self.config.ladder_arb.clone(),
                self.scan_tx.clone(),
            )),
            _ => unreachable!("build_scan_tool is only called for ToolKind::Scan"),
        }
    }

    /// Starts a recorder's task over `stop_rx`. Requires
    /// [`RecorderDeps`] to be set — without them, `spawn` never gets here.
    ///
    /// `async` because `ResearchEvents` does blocking I/O (the preflight
    /// fsync) via `spawn_blocking`: `spawn()` already released the `running`
    /// lock before calling here, so that fsync never runs with the
    /// mutex held.
    async fn spawn_recorder_task(
        &self,
        id: ToolId,
        deps: &RecorderDeps,
        stop_rx: watch::Receiver<bool>,
    ) -> Result<tokio::task::JoinHandle<()>, String> {
        match id {
            ToolId::Analytics => {
                let cfg = self.config.analytics.clone();
                Ok(tokio::spawn(recorders::analytics::run(
                    self.book_tx.subscribe(),
                    cfg.db_path,
                    deps.registry.clone(),
                    self.vol_rxs.clone(),
                    self.config.mode,
                    cfg.retention_hours,
                    cfg.purge_interval_secs,
                    cfg.purge_batch_size,
                    stop_rx,
                )))
            }
            ToolId::SpotStream => {
                let cfg = self.config.spot_stream.clone();
                let db_path = StateStore::resolve_path(&cfg.db_path, self.config.mode);
                Ok(tokio::spawn(recorders::spot_stream::run(
                    db_path,
                    cfg.channel_capacity,
                    self.vol_rxs.clone(),
                    deps.research_hub.producer(),
                    stop_rx,
                )))
            }
            ToolId::ResolutionFetcher => {
                let cfg = self.config.resolution_fetcher.clone();
                Ok(tokio::spawn(recorders::resolution::run_with_recorder(
                    deps.state.clone(),
                    cfg,
                    stop_rx,
                    deps.research_hub.producer(),
                )))
            }
            // The model tool emits predictions via
            // `broadcast::Sender<ModelPrediction>` directly to the
            // subscribed strategies. Its spawn is done in `spawn_model_tool`
            // (see below), NEVER by `spawn_recorder_task`. We return an
            // explicit `Err` as a safety net — if this arm were
            // reached in production, the diagnosis would point to the
            // wrong place instead of failing silently.
            ToolId::ModelTool => Err(
                "ToolId::ModelTool is not a recorder — its spawn goes through \
                 `spawn_model_tool` and is dispatched by \
                 `ToolSupervisor::spawn` for the right dispatch."
                    .to_string(),
            ),
            ToolId::ResearchEvents => {
                let cfg = crate::research::events::ResearchRecorderConfig {
                    dir: std::path::PathBuf::from(&self.config.research_events.dir),
                    channel_capacity: self.config.research_events.channel_capacity,
                    dedup_capacity: self.config.research_events.dedup_capacity,
                };
                let cfg_dir = cfg.dir.clone();
                let hub = deps.research_hub.clone();
                // The `attach()` must have happened ALREADY by the time this
                // function (and hence `spawn()`) returns: the WS tape
                // subscription evaluates `hub.is_active()` ONCE per
                // session (see the long comment in `market_feed::ws`), so
                // if `market_feed::run`/`oracle::price` had already
                // started by the time the hub is activated, the first session
                // would be born without a tape until the next reconnection.
                // `spawn_blocking` + `.await` keeps that guarantee (it does not
                // return from here until the preflight fsync finishes)
                // WITHOUT running it under the `running` lock — `spawn()` already
                // released it before getting here.
                let start_result = tokio::task::spawn_blocking(move || {
                    crate::research::events::ResearchRecorder::start(&cfg, &hub)
                })
                .await
                .map_err(|e| format!("{id}: panic in the startup thread: {e}"))?;
                match start_result {
                    Ok(recorder) => {
                        tracing::info!(dir = %cfg_dir.display(), "research_events: recorder active");
                        Ok(tokio::spawn(recorders::research_tape::run(recorder, stop_rx)))
                    }
                    // A startup failure used to
                    // be swallowed as `Ok` (an `async {}` task that ends
                    // by itself) — the caller (API/Telegram) persisted
                    // `enabled=true` for a recorder that never got to
                    // record anything. Now `spawn()` propagates this `Err` and the
                    // caller does not persist.
                    Err(e) => Err(format!(
                        "{id}: could not start the recorder ({dir}): {e}",
                        dir = cfg_dir.display()
                    )),
                }
            }
            ToolId::VolSurface | ToolId::LadderArb => {
                unreachable!("spawn_recorder_task is only called for ToolKind::Recorder")
            }
        }
    }

    /// Spawn of `ToolId::ModelTool`: starts `run_model_tool` with the
    /// `prediction_tx` set by `set_model_prediction_tx`. Returns the
    /// `JoinHandle` of the runner ONLY — the JSONL writer is drained in
    /// `stop()` via `ToolHandle::join_and_drain` (no abort).
    ///
    /// The check that `prediction_tx` is set is done BEFORE releasing the
    /// `Starting` lock: if the injection is missing, we undo the `Starting` and
    /// return a clear `Err`. NEVER a panic, never silently starts.
    async fn spawn_model_tool(self: &Arc<Self>, id: ToolId) -> Result<(), String> {
        debug_assert!(id == ToolId::ModelTool);

        let Some(pred_tx) = self.model_prediction_tx.get().cloned() else {
            self.running.lock().unwrap().remove(&id);
            return Err(format!(
                "{id}: model_prediction_tx not set — call `set_model_prediction_tx` \
                 in `main.rs` before `spawn_enabled_at_boot`"
            ));
        };

        let (stop_tx, stop_rx) = watch::channel(false);
        // We forward the global kill just like for a scan tool: `run_model_tool`
        // consumes books until it receives `true` or the channel closes.
        let mut global_kill = self.shutdown.clone();
        let forward_tx = stop_tx.clone();
        tokio::spawn(async move {
            tokio::select! {
                r = global_kill.changed() => {
                    if r.is_ok() && *global_kill.borrow() {
                        let _ = forward_tx.send(true);
                    }
                }
                _ = forward_tx.closed() => {}
            }
        });

        let cfg = self.config.model_tool.clone();
        let book_rx = self.book_tx.subscribe();
        let vol_rxs = self.vol_rxs.clone();
        let max_book_age = self.max_book_age;
        let prediction_tx = (*pred_tx).clone();

        let mut handle = match crate::tools::model::ToolHandle::spawn(
            cfg,
            book_rx,
            vol_rxs,
            prediction_tx,
            stop_rx,
            max_book_age,
        ) {
            Ok(h) => h,
            Err(e) => {
                self.running.lock().unwrap().remove(&id);
                return Err(e);
            }
        };

        // We take the runner's `JoinHandle` to put it in the `running`
        // map (queried with `is_finished` for `is_running`). The rest
        // of the `ToolHandle` (JSONL writer) is kept in `model_handle`
        // to drain it in `stop`/`shutdown_all`.
        let runner_join = match handle.take_runner() {
            Some(h) => h,
            None => {
                self.running.lock().unwrap().remove(&id);
                return Err(format!(
                    "{id}: internal failure — runner already consumed when spawning"
                ));
            }
        };
        *self.model_handle.lock().unwrap() = Some(handle);
        let mut running = self.running.lock().unwrap();
        running.insert(id, SlotState::Running(stop_tx, runner_join));
        drop(running);
        tracing::info!(%id, "tool supervisor: task started");
        Ok(())
    }

    /// Starts the tools with `enabled = true` in their TOML — call once
    /// at bot start (equivalent to the initial `spawn_strategy!`). A
    /// failure of a specific recorder (missing deps, startup error) is only
    /// logged — the rest keep starting.
    pub async fn spawn_enabled_at_boot(self: &Arc<Self>) {
        for &id in ToolId::ALL {
            if self.config_enabled(id) {
                if let Err(e) = self.spawn(id).await {
                    tracing::warn!(%id, error = %e, "tool supervisor: could not start at boot");
                }
            }
        }
    }

    /// Starts the task of `id` if it is not already running.
    ///
    /// The `running` `std::Mutex` is NEVER held during blocking I/O:
    /// it is taken to check/mark
    /// `Starting`, released, the task is built (for `ResearchEvents`
    /// that includes the preflight fsync, via `spawn_blocking` inside
    /// `spawn_recorder_task`), and retaken only to insert `Running` or
    /// undo the `Starting` if it failed. `Starting`/`Stopping` reject a
    /// concurrent `spawn` on the same id.
    async fn spawn(self: &Arc<Self>, id: ToolId) -> Result<(), String> {
        {
            let mut running = self.running.lock().unwrap();
            match running.get(&id) {
                Some(SlotState::Stopping) => {
                    return Err(format!("{id}: stopping, retry when it finishes"));
                }
                Some(SlotState::Starting) => {
                    return Err(format!("{id}: already starting"));
                }
                Some(SlotState::Running(_, h)) if !h.is_finished() => {
                    debug!(%id, "tool supervisor: already running, spawn ignored");
                    return Ok(());
                }
                _ => {}
            }
            running.insert(id, SlotState::Starting);
        }

        // ModelTool has its own path: it uses `prediction_tx` (not
        // `recorder_deps`) and emits predictions, not scan observations.
        // The check is done BEFORE the scan/recorder flow below so as not to
        // contaminate the common logic.
        if id == ToolId::ModelTool {
            return self.spawn_model_tool(id).await;
        }

        // Recorder deps, checked BEFORE building anything — a rejection
        // here undoes the `Starting` without leaving anything half-done.
        if id.kind() == ToolKind::Recorder && self.recorder_deps.get().is_none() {
            self.running.lock().unwrap().remove(&id);
            return Err(format!(
                "{id}: RecorderDeps not set (set_recorder_deps was never called)"
            ));
        }

        let (stop_tx, stop_rx) = watch::channel(false);
        let build_result: Result<tokio::task::JoinHandle<()>, String> = match id.kind() {
            ToolKind::Scan => {
                // Global kill forwarding — ONLY scan tools.
                // A recorder does NOT follow the global kill: it is only
                // stopped by `set_enabled`/`shutdown_all`. `main` sends
                // `kill_tx.send(true)` BEFORE the grace and the `abort_all` of
                // the producers (`tasks`) — a recorder that followed that
                // signal would disconnect right there, losing what those
                // producers still emit during that grace. `run_tool`
                // can follow it: a scan tool persists nothing of its own.
                let mut global_kill = self.shutdown.clone();
                let forward_tx = stop_tx.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        r = global_kill.changed() => {
                            if r.is_ok() && *global_kill.borrow() {
                                let _ = forward_tx.send(true);
                            }
                        }
                        _ = forward_tx.closed() => {}
                    }
                });
                let ctx = ToolCtx {
                    book_rx: self.book_tx.subscribe(),
                    vol_rxs: self.vol_rxs.clone(),
                    scan_tx: self.scan_tx.clone(),
                    shutdown: stop_rx,
                    max_book_age: self.max_book_age,
                };
                let tool = self.build_scan_tool(id);
                Ok(tokio::spawn(run_tool(tool, ctx)))
            }
            ToolKind::Recorder => {
                let deps = self
                    .recorder_deps
                    .get()
                    .expect("already checked above before building the stop channel");
                self.spawn_recorder_task(id, deps, stop_rx).await
            }
        };

        let mut running = self.running.lock().unwrap();
        match build_result {
            Ok(handle) => {
                running.insert(id, SlotState::Running(stop_tx, handle));
                tracing::info!(%id, "tool supervisor: task started");
                Ok(())
            }
            Err(e) => {
                running.remove(&id); // undo the Starting — nothing was left half-done
                Err(e)
            }
        }
    }

    /// Stops the task of `id` if it is running.
    ///
    /// The REAL stop (signal + wait for the task to finish) runs in a
    /// background task INDEPENDENT of the caller: if the future of this `stop()`
    /// is dropped (HTTP client disconnects, `shutdown_all` is slow on another tool
    /// and the runtime cancels this wait, etc.), that background task stays alive and
    /// clears the slot all the same when the real join finishes — it is never left
    /// `Stopping` forever.
    ///
    /// A recorder is NEVER aborted: an `abort()` halfway through a flush would leave the
    /// writer OS thread alive writing to the same file while a
    /// later `spawn` (after `is_running()==false`) opens a second
    /// writer — the background task waits for the real join no matter how long it
    /// takes. A scan tool is aborted after `STOP_TIMEOUT`: it has nothing
    /// to flush, so forcing its exit is safe.
    ///
    /// This method only WAITS for the background task's result with ITS
    /// OWN timeout — if it expires, it returns `Err` ("stop in progress") without
    /// that meaning the background stop was abandoned; a `set_enabled`
    /// that receives that `Err` must not persist `enabled=false`.
    async fn stop(self: &Arc<Self>, id: ToolId) -> Result<(), String> {
        // ModelTool has its own drain (JSONL writer) — we handle its
        // stop outside the common flow so as not to contaminate the
        // scan/recorder logic.
        if id == ToolId::ModelTool {
            return self.stop_model_tool().await;
        }

        let (stop_tx, handle) = {
            let mut running = self.running.lock().unwrap();
            match running.get(&id) {
                None => return Ok(()), // already stopped
                Some(SlotState::Stopping) => return Err(format!("{id}: stop in progress")),
                Some(SlotState::Starting) => {
                    return Err(format!("{id}: still starting, retry shortly"));
                }
                Some(SlotState::Running(_, _)) => {}
            }
            match running.insert(id, SlotState::Stopping) {
                Some(SlotState::Running(tx, h)) => (tx, h),
                _ => unreachable!("the match above already confirmed Running"),
            }
        };

        let _ = stop_tx.send(true);

        let kind = id.kind();
        let sup = self.clone();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            let join_result = match kind {
                ToolKind::Scan => {
                    let abort_handle = handle.abort_handle();
                    match tokio::time::timeout(STOP_TIMEOUT, handle).await {
                        Ok(r) => r,
                        Err(_) => {
                            tracing::warn!(
                                %id,
                                timeout_s = STOP_TIMEOUT.as_secs(),
                                "tool supervisor: did not respond to the signal in time, aborting"
                            );
                            abort_handle.abort();
                            Ok(())
                        }
                    }
                }
                // Never abort — wait for the real join even if it takes longer than
                // `STOP_TIMEOUT` (the caller of `stop()` only waits for THIS
                // result with its own timeout, further below).
                ToolKind::Recorder => handle.await,
            };
            match join_result {
                Ok(()) => tracing::info!(%id, "tool supervisor: task stopped"),
                Err(e) => tracing::warn!(%id, error = %e, "tool supervisor: panic while stopping"),
            }
            sup.running.lock().unwrap().remove(&id);
            let _ = done_tx.send(());
        });

        match tokio::time::timeout(STOP_TIMEOUT, done_rx).await {
            Ok(_) => Ok(()),
            Err(_) => Err(format!(
                "{id}: stop in progress (more than {}s, still running in the background)",
                STOP_TIMEOUT.as_secs()
            )),
        }
    }

    /// Hot toggle: starts or stops the task. Persistence to TOML —
    /// and the update of [`Self::enabled`] via
    /// [`Self::set_desired_enabled`] — are done by the caller after an `Ok` (same
    /// pattern as `control_strategy`: runtime effect first, then
    /// `persist_strategy_toggle`).
    pub async fn set_enabled(self: &Arc<Self>, id: ToolId, enabled: bool) -> Result<(), String> {
        if enabled {
            self.spawn(id).await
        } else {
            self.stop(id).await
        }
    }

    /// Running right now (for API/Telegram status). `Starting`/
    /// `Stopping` count as not running — it is the right reading for an
    /// operator ("can I trust that it is recording RIGHT NOW?").
    pub fn is_running(&self, id: ToolId) -> bool {
        matches!(
            self.running.lock().unwrap().get(&id),
            Some(SlotState::Running(_, h)) if !h.is_finished()
        )
    }

    /// For `ToolId::ModelTool`, with an orderly drain of the JSONL writer.
    /// Steps:
    ///   1. Mark `Stopping` in the `running` map, EXTRACTING the real
    ///      `JoinHandle` of the runner (do not lose it, as the earlier
    ///      `insert(_,_)` did — the runner's handle was left
    ///      orphaned, so `ToolHandle::join_and_drain`
    ///      waited for nobody and shutdown reported Ok before the
    ///      real task finished).
    ///   2. Send `true` to `stop_tx` so `run_model_tool` exits.
    ///   3. Wait for the real runner with `STOP_TIMEOUT` (same deadline as the
    ///      scan tools; a recorder-style without `STOP_TIMEOUT` would apply only
    ///      to recorders that only hold in-memory state, not to a task that
    ///      writes to a file and must close in time).
    ///   4. Consume the `ToolHandle` (flush the JSONL writer); if the
    ///      runner already finished on its own, the writer still has data in
    ///      flight and must be drained.
    ///   5. Clear the `running` slot.
    /// Never aborts the writer: an abort halfway through a flush would leave the
    /// writer OS thread alive writing to the same file.
    async fn stop_model_tool(self: &Arc<Self>) -> Result<(), String> {
        let id = ToolId::ModelTool;
        // 1) We extract stop_tx + the real JoinHandle from the Running slot. If the
        //    state is not Running, same handling as the scan/recorder `stop()`.
        let (stop_tx, runner_join) = {
            let mut running = self.running.lock().unwrap();
            match running.get(&id) {
                None => return Ok(()),
                Some(SlotState::Stopping) => return Err(format!("{id}: stop in progress")),
                Some(SlotState::Starting) => {
                    return Err(format!("{id}: still starting, retry shortly"));
                }
                Some(SlotState::Running(_, _)) => {}
            }
            match running.insert(id, SlotState::Stopping) {
                Some(SlotState::Running(tx, h)) => (tx, h),
                _ => unreachable!("the match above already confirmed Running"),
            }
        };

        // 2) The `ToolHandle` keeps its own `writer`; the inner `runner`
        //    is already `None` because `spawn_model_tool` called
        //    `take_runner()`. We wait for the real runner with the
        //    `JoinHandle` we just took from the slot — NOT with the
        //    inner one of the `ToolHandle` (that was the original bug).
        //    We take the `ToolHandle` WITHOUT holding the `MutexGuard` across an
        //    `.await` (the `StdMutex` is not `Send`); if it is not there, we wait
        //    for the real runner with `runner_join` and clear the slot.
        let handle_exists = {
            let guard = self.model_handle.lock().unwrap();
            guard.is_some()
        };
        if !handle_exists {
            // The `ToolHandle` was no longer there: the runner stays alive
            // in `runner_join` and we must wait for it anyway. If the
            // runner already finished, `is_finished() == true` and the await
            // returns instantly — the writer (if it existed) we
            // lose, but the JSONL writer is inside
            // `model_handle` which was already emptied, so we assume that
            // another path (an earlier `shutdown_all`) already drained it.
            let _ = stop_tx.send(true);
            let join_result = runner_join.await;
            self.running.lock().unwrap().remove(&id);
            return match join_result {
                Ok(()) => Ok(()),
                Err(_) => Err(format!("{id}: runner panicked during the stop")),
            };
        }
        let handle = self.model_handle.lock().unwrap().take().expect(
            "is_some() check above under the same mutex — it cannot have disappeared",
        );

        let _ = stop_tx.send(true);

        // 3) We wait for the real runner. If `STOP_TIMEOUT` expires we abort:
        //    a hung runner must not prevent the supervisor's shutdown
        //    (same criterion as `stop()` with `Scan`/`Recorder` → see the
        //    `match kind` above). The `ToolHandle` is drained in the background
        //    — a `join_and_drain` with `runner=None` skips the runner
        //    await (which we already did here) and only flushes the writer.
        let runner_outcome = {
            let abort_handle = runner_join.abort_handle();
            match tokio::time::timeout(STOP_TIMEOUT, runner_join).await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(join_err)) => {
                    tracing::warn!(
                        %id,
                        error = %join_err,
                        "tool supervisor: model_tool runner panicked during the stop"
                    );
                    Err(())
                }
                Err(_) => {
                    tracing::warn!(
                        %id,
                        timeout_s = STOP_TIMEOUT.as_secs(),
                        "tool supervisor: model_tool runner did not respond in time, aborting"
                    );
                    abort_handle.abort();
                    Err(())
                }
            }
        };

        // 4) Drain of the JSONL writer — fire-and-forget in the background
        //    so as not to block the caller for an extra `STOP_TIMEOUT`. If the
        //    earlier wait ended Ok, `writer.close()` should be
        //    instant. If we aborted, the writer may hang: we
        //    bound it with a second `STOP_TIMEOUT` so the bot's shutdown
        //    never depends on a slow flush after an abort.
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<(u64, u64)>();
        let id_for_task = id;
        tokio::spawn(async move {
            let stats = handle.join_and_drain().await;
            tracing::info!(
                tool = %id_for_task,
                written = stats.0,
                discarded = stats.1,
                "tool supervisor: model_tool writer drained"
            );
            let _ = done_tx.send(stats);
        });

        match tokio::time::timeout(STOP_TIMEOUT, done_rx).await {
            Ok(_) => {
                self.running.lock().unwrap().remove(&id);
                if runner_outcome.is_err() {
                    // We report the runner that failed, but the write
                    // was drained.
                    Err(format!("{id}: runner ended with an error (writer drained)"))
                } else {
                    Ok(())
                }
            }
            Err(_) => {
                self.running.lock().unwrap().remove(&id);
                Err(format!(
                    "{id}: writer drain in progress (more than {}s, still running in the background)",
                    STOP_TIMEOUT.as_secs()
                ))
            }
        }
    }

    /// Orderly shutdown of ALL the tools — call once in the shutdown
    /// of `main`, AFTER the producers have finished (grace +
    /// `abort_all` of `tasks`) and after `cancel_all` (money first). Never
    /// before: the recorders no longer follow the global kill, so
    /// this is the ONLY way they are stopped, and it has to run
    /// when the queue of what those producers emitted during the grace
    /// is already complete — if called earlier, it would stop over a queue
    /// still growing.
    ///
    /// Budget: the scan tools and the recorders that are not
    /// `research_events` stop IN PARALLEL (`tokio::join!`), each with its
    /// own `STOP_TIMEOUT` (`stop()` already applies it internally) — so a slow
    /// one never consumes the budget of the others. `research_events`
    /// goes last, with its OWN separate `stop()`: before, they shared an
    /// aggregate timeout that could run out on the first ones and leave
    /// `research_events` without `stop()` even being called.
    pub async fn shutdown_all(self: &Arc<Self>) {
        let (r_vs, r_la, r_an, r_ss, r_rf, r_mt) = tokio::join!(
            self.stop(ToolId::VolSurface),
            self.stop(ToolId::LadderArb),
            self.stop(ToolId::Analytics),
            self.stop(ToolId::SpotStream),
            self.stop(ToolId::ResolutionFetcher),
            self.stop(ToolId::ModelTool),
        );
        for (id, r) in [
            (ToolId::VolSurface, r_vs),
            (ToolId::LadderArb, r_la),
            (ToolId::Analytics, r_an),
            (ToolId::SpotStream, r_ss),
            (ToolId::ResolutionFetcher, r_rf),
            (ToolId::ModelTool, r_mt),
        ] {
            if let Err(e) = r {
                tracing::warn!(%id, error = %e, "tool supervisor: failed to stop in shutdown_all");
            }
        }
        if let Err(e) = self.stop(ToolId::ResearchEvents).await {
            tracing::warn!(
                error = %e,
                "tool supervisor: failed to stop research_events in shutdown_all"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};
    use rust_decimal_macros::dec;

    /// Dummy tool: counts how many books it processed (emitting nothing).
    struct CountingTool {
        count: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl Tool for CountingTool {
        fn id(&self) -> ToolId {
            ToolId::VolSurface
        }
        fn on_book(&mut self, _book: &BookSnapshot, _vol: &VolSnapshot) {
            self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn make_book(age: std::time::Duration) -> Arc<BookSnapshot> {
        let half = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        Arc::new(BookSnapshot {
            market_condition_id: "cid-tool".into(),
            yes: TokenBook {
                token_id: "tok-yes".into(),
                bids: half("0.49"),
                asks: half("0.51"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-no".into(),
                bids: half("0.49"),
                asks: half("0.51"),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: 9_999_999_999,
            neg_risk: false,
            slug: "btc-updown-test".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: crate::time::Instant::now().checked_sub(age),
        })
    }

    /// Same as in `strategy::drain_latest`: a `RecvError::Lagged` at the
    /// `await` is transient — the drain must survive and return a book.
    #[tokio::test]
    async fn drain_latest_survives_lagged_in_await() {
        let (tx, mut rx) = broadcast::channel::<Arc<BookSnapshot>>(4);
        let drain = tokio::spawn(async move { drain_latest(&mut rx).await });
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        for _ in 0..10 {
            tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        }
        let got = tokio::time::timeout(std::time::Duration::from_secs(5), drain)
            .await
            .expect("drain_latest hung")
            .expect("task panic");
        assert!(
            got.as_ref().is_some_and(|v| !v.is_empty()),
            "drain_latest returned None/empty after Lagged — runner dead"
        );
    }

    #[tokio::test]
    async fn tool_processes_fresh_books_without_emitting_anything() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let ctx = ToolCtx {
            book_rx,
            vol_rxs: HashMap::new(),
            scan_tx,
            shutdown: kill_rx,
            max_book_age: std::time::Duration::from_secs(5),
        };
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn(run_tool(
            Box::new(CountingTool {
                count: count.clone(),
            }),
            ctx,
        ));

        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn tool_guard_staleness_skips_old_books() {
        let (book_tx, book_rx) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let ctx = ToolCtx {
            book_rx,
            vol_rxs: HashMap::new(),
            scan_tx,
            shutdown: kill_rx,
            max_book_age: std::time::Duration::from_millis(500),
        };
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn(run_tool(
            Box::new(CountingTool {
                count: count.clone(),
            }),
            ctx,
        ));

        book_tx
            .send(make_book(std::time::Duration::from_secs(10)))
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "stale book must not be processed"
        );

        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn tool_id_display_from_str_roundtrip() {
        for &id in ToolId::ALL {
            let s = id.to_string();
            assert_eq!(s.parse::<ToolId>().unwrap(), id);
        }
    }

    #[test]
    fn tool_id_from_str_rejects_unknown() {
        assert!("arb".parse::<ToolId>().is_err());
        assert!("".parse::<ToolId>().is_err());
    }

    #[test]
    fn tool_id_kind_separates_scan_from_recorder() {
        assert_eq!(ToolId::VolSurface.kind(), ToolKind::Scan);
        assert_eq!(ToolId::LadderArb.kind(), ToolKind::Scan);
        for &id in &[
            ToolId::Analytics,
            ToolId::SpotStream,
            ToolId::ResearchEvents,
            ToolId::ResolutionFetcher,
        ] {
            assert_eq!(id.kind(), ToolKind::Recorder, "{id} should be a Recorder");
        }
    }

    fn test_supervisor() -> Arc<ToolSupervisor> {
        let (book_tx, _book_rx) = broadcast::channel(16);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let (_kill_tx, kill_rx) = watch::channel(false);
        ToolSupervisor::new(
            book_tx,
            HashMap::new(),
            scan_tx,
            kill_rx,
            std::time::Duration::from_secs(5),
            Arc::new(crate::config::base_config_with_universe(true)),
        )
    }

    /// `spawn` must reject while the map marks `Stopping` — checked
    /// by setting the state by hand (same module, `SlotState` is private) instead
    /// of forcing a real race against `stop()`: a tool with no books in
    /// flight (as in this test, with no traffic) can take up to
    /// `STOP_TIMEOUT` to resolve that race, and depending on winning it would be a
    /// timing-fragile test to prove exactly the same
    /// precondition that `spawn` already checks synchronously.
    #[tokio::test]
    async fn spawn_rejects_while_stopping() {
        let sup = test_supervisor();
        {
            let mut running = sup.running.lock().unwrap();
            running.insert(ToolId::VolSurface, SlotState::Stopping);
        }
        let err = sup
            .spawn(ToolId::VolSurface)
            .await
            .expect_err("must reject spawn in Stopping");
        assert!(err.contains("stopping"), "unexpected message: {err}");
    }

    /// Same criterion as above but for `Starting` (the gap
    /// between releasing the lock and retaking it to
    /// insert `Running` must also reject a concurrent `spawn`/`stop`).
    #[tokio::test]
    async fn spawn_and_stop_reject_while_starting() {
        let sup = test_supervisor();
        {
            let mut running = sup.running.lock().unwrap();
            running.insert(ToolId::VolSurface, SlotState::Starting);
        }
        let err = sup
            .spawn(ToolId::VolSurface)
            .await
            .expect_err("must reject spawn in Starting");
        assert!(err.contains("starting"), "unexpected message: {err}");
        let err = sup
            .stop(ToolId::VolSurface)
            .await
            .expect_err("must reject stop in Starting");
        assert!(err.contains("starting"), "unexpected message: {err}");
    }

    /// `stop()` on a tool that is not running (never started) is
    /// idempotent: there is nothing to join, no error.
    #[tokio::test]
    async fn stop_on_non_running_tool_is_noop() {
        let sup = test_supervisor();
        assert!(sup.stop(ToolId::VolSurface).await.is_ok());
    }

    /// Full spawn→stop→spawn cycle of a scan tool through the
    /// real supervisor: it starts, processes a book, stops by signal (join
    /// within `STOP_TIMEOUT`, not by `abort()`) and starts again without
    /// being left marked `Stopping`. Anchors the `run_tool` fix above: without
    /// the `select!` against `shutdown.changed()`, this `stop()` would hang the
    /// full `STOP_TIMEOUT` (10s) waiting for a book that never arrives after
    /// the first one.
    #[tokio::test]
    async fn scan_tool_toggle_on_off_on_through_supervisor() {
        let (book_tx, _book_rx) = broadcast::channel(16);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let (_kill_tx, kill_rx) = watch::channel(false);
        let sup = ToolSupervisor::new(
            book_tx.clone(),
            HashMap::new(),
            scan_tx,
            kill_rx,
            std::time::Duration::from_secs(5),
            Arc::new(crate::config::base_config_with_universe(true)),
        );

        sup.spawn(ToolId::VolSurface).await.unwrap();
        assert!(sup.is_running(ToolId::VolSurface));

        // An in-flight book so the task is awake when the
        // signal arrives (not needed for `stop()` to work with the fix above, but
        // it exercises the common path: stopping while traffic is flowing).
        book_tx.send(make_book(std::time::Duration::ZERO)).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let start = std::time::Instant::now();
        sup.stop(ToolId::VolSurface).await.unwrap();
        assert!(!sup.is_running(ToolId::VolSurface));
        assert!(
            start.elapsed() < STOP_TIMEOUT,
            "stop() should not have needed the full timeout — the join on the signal failed"
        );

        sup.spawn(ToolId::VolSurface).await.unwrap();
        assert!(sup.is_running(ToolId::VolSurface));
        sup.stop(ToolId::VolSurface).await.unwrap();
    }

    /// `shutdown_all` stops everything that is running, without the caller needing to
    /// know which ones: for tools never started it is a full no-op.
    #[tokio::test]
    async fn shutdown_all_on_empty_supervisor_does_not_fail() {
        let sup = test_supervisor();
        sup.shutdown_all().await;
        for &id in ToolId::ALL {
            assert!(!sup.is_running(id));
        }
    }

    /// `enabled()`/`set_desired_enabled()`: the
    /// desired state starts equal to `config_enabled` (snapshot of the TOML
    /// at boot) and only changes when the caller updates it explicitly —
    /// never just through `spawn`/`stop` (the supervisor knows nothing about persistence,
    /// that decision belongs to the caller after a successful `persist_strategy_toggle`).
    #[tokio::test]
    async fn enabled_originates_from_config_and_only_changes_with_set_desired_enabled() {
        let sup = test_supervisor();
        assert_eq!(
            sup.enabled(ToolId::VolSurface),
            sup.config_enabled(ToolId::VolSurface)
        );

        sup.spawn(ToolId::VolSurface).await.unwrap();
        assert_eq!(
            sup.enabled(ToolId::VolSurface),
            sup.config_enabled(ToolId::VolSurface),
            "spawn() alone must not touch the desired state"
        );

        sup.set_desired_enabled(ToolId::VolSurface, true);
        assert!(sup.enabled(ToolId::VolSurface));
        sup.set_desired_enabled(ToolId::VolSurface, false);
        assert!(!sup.enabled(ToolId::VolSurface));
    }

    /// Builds the minimal `RecorderDeps` for a test. Each test creates its own
    /// `ToolSupervisor`/`OnceLock` instance (not a global `static` of the binary),
    /// so several tests can set them without stepping on each other
    /// (the old shared `static` no longer exists).
    async fn test_deps(dir: &std::path::Path) -> (RecorderDeps, Arc<Hub>) {
        let state_path = dir.join("state.sqlite");
        let state = crate::state::StateStore::open(state_path.to_str().unwrap(), crate::types::Mode::Paper)
            .await
            .unwrap();
        let research_hub = Hub::new();
        (
            RecorderDeps {
                registry: Arc::new(MarketRegistry::new()),
                state,
                research_hub: research_hub.clone(),
            },
            research_hub,
        )
    }

    fn recorder_supervisor(config: crate::config::Config) -> Arc<ToolSupervisor> {
        let (book_tx, _book_rx) = broadcast::channel(4);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let (_kill_tx, kill_rx) = watch::channel(false);
        ToolSupervisor::new(
            book_tx,
            HashMap::new(),
            scan_tx,
            kill_rx,
            std::time::Duration::from_secs(5),
            Arc::new(config),
        )
    }

    /// Startup order regression: the `attach()` on the hub must have already
    /// happened when `spawn()` returns — never at some later moment within
    /// the spawned task (`research_tape::run` receives the already-started
    /// `ResearchRecorder`). It is checked WITHOUT any intermediate `sleep` on purpose:
    /// if `spawn_recorder_task` went back to calling `start()` inside the (async,
    /// spawned) task instead of via `spawn_blocking().await` right here, this
    /// `assert!` would fail intermittently depending on the scheduler.
    /// (`spawn_blocking` keeps the startup synchronous.)
    #[tokio::test]
    async fn research_events_spawn_returns_with_hub_already_active() {
        let dir = tempfile::tempdir().unwrap();
        let (deps, research_hub) = test_deps(dir.path()).await;
        let mut config = crate::config::base_config_with_universe(true);
        config.research_events.dir = dir.path().join("research_events").to_str().unwrap().to_string();
        let sup = recorder_supervisor(config);
        sup.set_recorder_deps(deps);

        sup.spawn(ToolId::ResearchEvents).await.unwrap();
        assert!(
            research_hub.is_active(),
            "the hub must be active IMMEDIATELY when spawn() returns, \
             before market_feed/oracle can start"
        );
        sup.stop(ToolId::ResearchEvents).await.unwrap();
        assert!(!research_hub.is_active());
    }

    /// A startup failure of `ResearchEvents`
    /// (here `dir` points to an EXISTING FILE, not a directory — the
    /// preflight of `ResearchRecorder::start` cannot create/write there)
    /// must propagate as `Err` from `spawn`/`set_enabled` — the caller
    /// (API/Telegram) must not persist `enabled=true` for something that never
    /// started. The slot must not stay stuck: a later `spawn` (after
    /// fixing the problem) must be able to retry.
    #[tokio::test]
    async fn research_events_startup_failure_propagates_err_and_does_not_jam_slot() {
        let dir = tempfile::tempdir().unwrap();
        let (deps, research_hub) = test_deps(dir.path()).await;
        let bad_dir = dir.path().join("is_not_a_directory");
        std::fs::write(&bad_dir, b"a regular file, not a directory").unwrap();

        let mut config = crate::config::base_config_with_universe(true);
        config.research_events.dir = bad_dir.to_str().unwrap().to_string();
        let sup = recorder_supervisor(config);
        sup.set_recorder_deps(deps);

        let err = sup
            .spawn(ToolId::ResearchEvents)
            .await
            .expect_err("an invalid dir must make the startup fail");
        assert!(
            err.contains("research_events") || err.contains("could not start"),
            "unexpected message: {err}"
        );
        assert!(!research_hub.is_active());
        assert!(!sup.is_running(ToolId::ResearchEvents));

        // The slot was not left stuck in `Starting`/`Stopping`: a second
        // attempt (even if it fails again for the same reason) is accepted without
        // rejection for "already starting"/"stopping".
        let err2 = sup.spawn(ToolId::ResearchEvents).await.unwrap_err();
        assert!(!err2.contains("starting") && !err2.contains("stopping"));
    }

    /// A recorder does NOT follow the global kill — only
    /// `set_enabled`/`shutdown_all` stop it. `main` sends the global kill
    /// BEFORE the grace/`abort_all` of the producers; if a recorder followed it,
    /// it would lose what those producers emit during that grace
    /// (the real regression that motivated this fix).
    #[tokio::test]
    async fn recorder_does_not_follow_global_kill_only_shutdown_all_stops_it() {
        let dir = tempfile::tempdir().unwrap();
        let (deps, research_hub) = test_deps(dir.path()).await;
        let mut config = crate::config::base_config_with_universe(true);
        config.research_events.dir = dir.path().join("research_events").to_str().unwrap().to_string();

        let (book_tx, _book_rx) = broadcast::channel(4);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let (kill_tx, kill_rx) = watch::channel(false);
        let sup = ToolSupervisor::new(
            book_tx,
            HashMap::new(),
            scan_tx,
            kill_rx,
            std::time::Duration::from_secs(5),
            Arc::new(config),
        );
        sup.set_recorder_deps(deps);

        sup.spawn(ToolId::ResearchEvents).await.unwrap();
        assert!(sup.is_running(ToolId::ResearchEvents));

        // The GLOBAL kill must not stop it — unlike a scan tool.
        kill_tx.send(true).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(
            sup.is_running(ToolId::ResearchEvents),
            "a recorder must not follow the global kill"
        );
        assert!(research_hub.is_active());

        // shutdown_all() does stop it, for real (hub inactive on return).
        sup.shutdown_all().await;
        assert!(!sup.is_running(ToolId::ResearchEvents));
        assert!(!research_hub.is_active());
    }

    /// `shutdown_all` must stop all six tools, `research_events` included
    /// — regression: they used to share an aggregate timeout that
    /// could run out before reaching `research_events`, the last of the
    /// list).
    #[tokio::test]
    async fn shutdown_all_stops_all_six_tools_including_research_events() {
        let dir = tempfile::tempdir().unwrap();
        let (deps, research_hub) = test_deps(dir.path()).await;
        let mut config = crate::config::base_config_with_universe(true);
        config.research_events.dir = dir.path().join("research_events").to_str().unwrap().to_string();
        config.spot_stream.db_path = dir.path().join("spot.sqlite").to_str().unwrap().to_string();
        let sup = recorder_supervisor(config);
        sup.set_recorder_deps(deps);
        // `ToolId::ModelTool` requires `model_prediction_tx` to be set
        // before `spawn` (see the doc of `set_model_prediction_tx` and the
        // barrier of `spawn_model_tool`). Without this, `sup.spawn(ModelTool)`
        // returns `Err(...)` and the `unwrap()` panics — the test predates
        // ModelTool and benefited from `ToolId::ALL` not including
        // `ModelTool`. Since the barrier is structural (it cannot be
        // relaxed without losing the guarantee of the defensive check), we
        // fix it by injecting a dummy `broadcast::Sender`: the tool does not
        // emit predictions, it only needs the setter to have been
        // called.
        let (model_tx, _model_rx) =
            broadcast::channel::<crate::tools::model::ModelPrediction>(4);
        sup.set_model_prediction_tx(Arc::new(model_tx));

        for &id in ToolId::ALL {
            sup.spawn(id).await.unwrap();
        }
        for &id in ToolId::ALL {
            assert!(sup.is_running(id), "{id} should be running");
        }

        sup.shutdown_all().await;

        for &id in ToolId::ALL {
            assert!(!sup.is_running(id), "{id} should have stopped");
        }
        assert!(!research_hub.is_active());
    }

    /// End-to-end integration of `SpotStream`: `stop()` waits for the
    /// real flush (the row must already be in the file as soon as `stop()`
    /// returns, with no additional polling) and the toggle on→off→on does not clash with a
    /// writer still alive from before on the same file.
    #[tokio::test]
    async fn spot_stream_toggle_on_off_on_stop_waits_for_flush() {
        let dir = tempfile::tempdir().unwrap();
        let (deps, _hub) = test_deps(dir.path()).await;
        let db_path = dir.path().join("spot_stream_tool.sqlite");

        let (book_tx, _book_rx) = broadcast::channel(4);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let (_kill_tx, kill_rx) = watch::channel(false);
        let (vol_tx, vol_rx) = watch::channel(VolSnapshot::default());
        let mut vol_rxs = HashMap::new();
        vol_rxs.insert("BTC/USDT".to_string(), vol_rx);

        let mut config = crate::config::base_config_with_universe(true);
        config.spot_stream.db_path = db_path.to_str().unwrap().to_string();
        let sup = ToolSupervisor::new(
            book_tx,
            vol_rxs,
            scan_tx,
            kill_rx,
            std::time::Duration::from_secs(5),
            Arc::new(config),
        );
        sup.set_recorder_deps(deps);

        sup.set_enabled(ToolId::SpotStream, true).await.unwrap();
        assert!(sup.is_running(ToolId::SpotStream));

        // A spot change so the recorder has something to persist
        // (`asset` + `last_price > 0`: `from_vol_snapshot` does not record the
        // cold-start of `VolSnapshot::default()`).
        vol_tx.send_replace(VolSnapshot {
            asset: "BTC/USDT".to_string(),
            last_price: rust_decimal_macros::dec!(50000),
            ..VolSnapshot::default()
        });
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        sup.set_enabled(ToolId::SpotStream, false).await.unwrap();
        assert!(!sup.is_running(ToolId::SpotStream));
        let resolved = crate::state::StateStore::resolve_path(
            db_path.to_str().unwrap(),
            crate::types::Mode::Paper,
        );
        let count_after_stop: i64 = rusqlite::Connection::open(&resolved)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM spot_samples", [], |r| r.get(0))
            .unwrap();
        assert!(
            count_after_stop > 0,
            "stop() should have waited for the writer to flush the sample"
        );

        // Toggle on→off→on: reopening the SAME DB must not clash with a
        // writer still alive from before (it would have failed at the `unwrap()`
        // above or at this second startup).
        sup.set_enabled(ToolId::SpotStream, true).await.unwrap();
        assert!(sup.is_running(ToolId::SpotStream));
        sup.set_enabled(ToolId::SpotStream, false).await.unwrap();
    }

    /// `stop_model_tool` fix: after `set_enabled(false)`,
    /// the `JoinHandle` of the runner in the `running` slot must have finished
    /// (`is_finished() == true`), not merely been extracted from the slot.
    /// Before the fix, `stop_model_tool` inserted `Stopping`, discarding the
    /// real `JoinHandle`; `ToolHandle::join_and_drain` waited for nobody
    /// (its internal `runner` was already `None` after `take_runner`) and the
    /// shutdown reported Ok before the real task finished — an
    /// `is_running()==false` with the task still alive remained consistent
    /// afterwards. We verify the precondition that join_and_drain CANNOT
    /// detect on its own: the slot's handle must be `Finished`.
    #[tokio::test]
    async fn model_tool_stop_waits_for_real_runner_not_just_the_slot() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::config::base_config_with_universe(true);
        config.model_tool.enabled = true;
        // Points to a nonexistent artifact: the predictor degrades to
        // `NullPredictor` (abstentions), but the runner stays alive and
        // consumes books until it receives the stop signal. This gives us
        // a stable task that we can time.
        config.model_tool.artifact_dir = dir.path().join("no_model").to_string_lossy().into_owned();
        config.model_tool.predictions_dir = dir
            .path()
            .join("predictions")
            .to_string_lossy()
            .into_owned();

        let (book_tx, _book_rx) = broadcast::channel(4);
        let vol_rxs: HashMap<String, watch::Receiver<crate::types::VolSnapshot>> = HashMap::new();
        let (_kill_tx, kill_rx) = watch::channel(false);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let sup = ToolSupervisor::new(
            book_tx.clone(),
            vol_rxs,
            scan_tx,
            kill_rx,
            Duration::from_millis(500),
            Arc::new(config),
        );
        let (tx_raw, _rx) =
            broadcast::channel::<crate::tools::model::ModelPrediction>(64);
        let tx = Arc::new(tx_raw);
        sup.set_model_prediction_tx(tx);

        // We start ModelTool with a valid prediction_tx. Since the
        // artifact does not exist, the predictions will be abstentions —
        // enough to keep the task awake.
        sup.set_enabled(ToolId::ModelTool, true)
            .await
            .expect("startup with a populated prediction_tx must work");
        assert!(sup.is_running(ToolId::ModelTool));

        // Capture the JoinHandle from the slot to inspect it
        // after `stop` — this verifies the real wiring of the supervisor,
        // not the public face (`is_running`), which is already known to be misleading.
        let join_handle = {
            let running = sup.running.lock().unwrap();
            match running.get(&ToolId::ModelTool) {
                Some(SlotState::Running(_, h)) => Some(h.abort_handle()),
                _ => panic!("unexpected state after set_enabled(true)"),
            }
        };
        let join_handle = join_handle.expect("captured join_handle");
        assert!(
            !join_handle.is_finished(),
            "the runner must be alive after starting"
        );

        // Timing: `set_enabled(false)` must take at least the
        // time the task takes to respond to the signal (it cannot be
        // instantaneous). If the implementation returned without waiting for the
        // runner, this assert would hold even with the task alive.
        let t0 = std::time::Instant::now();
        let stop_res = sup.set_enabled(ToolId::ModelTool, false).await;
        let elapsed = t0.elapsed();
        // After stopping, the slot is released and `is_running` drops.
        assert!(!sup.is_running(ToolId::ModelTool));
        assert!(
            stop_res.is_ok(),
            "stop must report Ok if the writer drained, got: {stop_res:?}"
        );

        // The CRITICAL check of the fix: the original `JoinHandle` of the
        // slot must have finished. Without the fix, `join_and_drain` did not
        // wait for the runner (already `None` inside the
        // `ToolHandle`), the `SlotState` was erased on return, and this
        // check would see `is_finished() == false`.
        // We give a brief margin for the scheduler to do the `poll`
        // final of the task after the signal.
        let mut finished = join_handle.is_finished();
        for _ in 0..20 {
            if finished {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
            finished = join_handle.is_finished();
        }
        assert!(
            finished,
            "the runner's JoinHandle must be finished after stop() \
             (took {elapsed:?}); bug: stop_model_tool discarded the handle \
             and join_and_drain did not wait for the real runner"
        );
    }

    /// Combination: with the defensive barrier of mode active,
    /// `set_enabled(true)` in Live is rejected without spawning; we verify that
    /// the supervisor is LEFT in a consistent state (empty slot, no orphan task,
    /// no open writer). This protects against the
    /// `panic!` of the `ModelBtc5m::new` constructor if the barrier failed.
    /// Complements `model_btc5m_live_rejected_even_with_paper_only_false`
    /// in `strategy::supervisor::tests` with the runtime check that
    /// the supervisor spawns nothing on the Live path.
    #[tokio::test]
    async fn model_tool_does_not_require_strategy_supervisor() {
        // This test only verifies the isolation of the `ToolSupervisor`:
        // a set_enabled(true) without `prediction_tx` must fail clearly,
        // not panic, and must not leave the slot stuck in `Starting`.
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::config::base_config_with_universe(true);
        config.model_tool.enabled = true;
        config.model_tool.artifact_dir = dir.path().join("artifact").to_string_lossy().into_owned();
        config.model_tool.predictions_dir = dir
            .path()
            .join("predictions")
            .to_string_lossy()
            .into_owned();

        let (book_tx, _book_rx) = broadcast::channel(4);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let sup = ToolSupervisor::new(
            book_tx,
            HashMap::new(),
            scan_tx,
            kill_rx,
            Duration::from_millis(500),
            Arc::new(config),
        );
        // We do not call `set_model_prediction_tx` → the spawn must fail
        // before any task.
        let res = sup.set_enabled(ToolId::ModelTool, true).await;
        assert!(
            res.is_err(),
            "without prediction_tx the spawn of ModelTool must fail"
        );
        assert!(
            !sup.is_running(ToolId::ModelTool),
            "the slot must be free after a failed spawn"
        );
        // A second attempt (now with prediction_tx) must be able to
        // try without staying stuck in `Starting`.
        let (tx_raw, _rx) =
            broadcast::channel::<crate::tools::model::ModelPrediction>(64);
        sup.set_model_prediction_tx(Arc::new(tx_raw));
        assert!(sup.set_enabled(ToolId::ModelTool, true).await.is_ok());
        assert!(sup.is_running(ToolId::ModelTool));
        let _ = sup.set_enabled(ToolId::ModelTool, false).await;
        assert!(!sup.is_running(ToolId::ModelTool));
    }
}
