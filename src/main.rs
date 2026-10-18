//! Entry point of Poly-bot.
//!
//! Assembles all the components and starts the JoinSet of tasks.
//! The lifecycle of the strategy tasks is managed by `supervisor.rs`.
//! Infrastructure compiling + functional order pipeline.

use std::sync::Arc;
use std::str::FromStr;
// edition 2018

use tokio::sync::{mpsc, oneshot};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use poly_bot::{
    api::{self, ApiState},
    clob_sdk,
    config::Config,
    market_feed::{self, MarketRegistry},
    metrics, oracle,
    order::{
        executor::Executor,
        router::{resting_claim_usdc, OrderRouter},
        sim_router::SimulatedRouter,
    },
    risk::RiskGuard,
    state::StateStore,
    strategy::{self, BuildCtx, PostCloseSweep, StrategyCtx},
    tools::ToolSupervisor,
    types::{BookSnapshot, Fill, Mode, QueuedIntent, StrategyId},
};

/// Maximum time shutdown waits for the `fill_sink` before
/// aborting the tasks. Looser than `accounting::fill_sink::SHUTDOWN_DRAIN_CAP`
/// (5 s) so that the sink itself is the one that cuts off and records what was left.
const FILLS_SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(7);

/// Grace period for the fill producers that listen to `kill_rx`
/// to exit on their own before being aborted (those that do not listen, such as
/// `market_feed`, are always aborted). It fits comfortably within the sink's window.
const PRODUCERS_SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_millis(1500);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // TLS provider — must be installed before any TLS socket.
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls CryptoProvider (aws-lc-rs)");

    // Load .env if it exists (fallback for `cargo run`; launch.sh already sources it).
    // HRF_NO_DOTENV=1 skips it entirely: the dedicated paper profile
    // (launch_rl_paper.sh) runs isolated from credentials and from any
    // HRF_* overrides that .env might carry.
    if std::env::var_os("HRF_NO_DOTENV").is_none() {
        let _ = dotenvy::dotenv();
    }

    // Fail-fast if config/ does not exist in the working directory.
    if !std::path::Path::new("config").is_dir() {
        eprintln!(
            "FATAL: config/ directory not found in ({}).\n\
             Run the bot from the project directory, or use launch.sh.",
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "?".to_string())
        );
        std::process::exit(1);
    }

    let config = Arc::new(Config::load().unwrap_or_else(|e| {
        eprintln!("Config error: {e}");
        std::process::exit(1);
    }));

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let log_buf = poly_bot::log_buffer::new(500);
    let log_layer = poly_bot::log_buffer::LogLayer::new(log_buf.clone(), 500);
    // LOG_FORMAT=json → structured output for log aggregators (production).
    // Default: compact format readable in a terminal.
    if std::env::var("LOG_FORMAT").as_deref() == Ok("json") {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().json())
            .with(log_layer)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().compact())
            .with(log_layer)
            .init();
    }

    tracing::info!(mode = ?config.mode, "poly-bot starting");
    // Trace of HRF_<TOOL>__ENABLED env vars that would override the `enabled`
    // persisted in config/strategies/<tool>.toml. Goes AFTER the init of
    // tracing (a warn! inside Config::load would be lost without a subscriber).
    poly_bot::config::warn_tool_enabled_env_overrides();

    let prometheus_handle = match metrics::install_prometheus_recorder() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("metrics: failed to install recorder: {e}");
            std::process::exit(1);
        }
    };

    let state = StateStore::open(&config.db_path, config.mode)
        .await
        .unwrap_or_else(|e| {
            tracing::error!("Failed to open state DB: {e}");
            std::process::exit(1);
        });

    let risk = RiskGuard::new(config.risk.clone(), config.mode);
    // Restore persisted kill switches + session PnL BEFORE
    // connecting persistence — without this the bot started "fail-open" after
    // any restart (kill switches and daily-loss-stop were forgotten).
    // Fail-closed if the disk READ fails: start killed rather than
    // pretend everything is clean.
    match state.load_risk_state().await {
        Ok(rows) => risk.restore_persisted(rows),
        Err(e) => {
            tracing::error!("risk: failed restoring risk_state — starting KILLED (fail-closed): {e}");
            risk.kill_global(format!("failed restoring risk_state after restart: {e}"));
        }
    }
    risk.attach_persistence(state.clone());

    // In-memory alert log (UI) — fixes /ui/alerts, which was broken until now.
    let alert_log = poly_bot::api::alert_log::AlertLog::new(200);
    risk.attach_alert_log(alert_log.clone());

    // Per-bracket directional exposure ledger: rebuild from the
    // open positions so a restart does not lose the exposure already committed.
    let ledger = poly_bot::coordination::BracketLedger::new();
    match state.list_open_positions().await {
        Ok(open) => {
            for p in &open {
                // Attribution by owner — the position carries its strategy in the DB.
                ledger.force_add_exposure(
                    &p.market_cid,
                    p.clear_at,
                    &p.token_id,
                    p.cost_basis,
                    poly_bot::types::StrategyId::from_db_str(&p.strategy),
                );
            }
            if !open.is_empty() {
                tracing::info!(
                    n = open.len(),
                    "ledger: exposure rebuilt from open positions"
                );
            }
        }
        Err(e) => tracing::warn!("ledger: error rebuilding exposure from DB: {e}"),
    }

    // Initial paper deposit: only if the account is empty (not re-added on restarts).
    if config.mode == Mode::Paper {
        let initial = config.risk.paper_initial_deposit_usdc;
        if initial > rust_decimal::Decimal::ZERO
            && state.deposits_total().await.unwrap_or_default() == rust_decimal::Decimal::ZERO
        {
            if let Err(e) = state
                .add_deposit(initial, "initial paper capital (config)")
                .await
            {
                tracing::warn!("main: initial deposit failed: {e}");
            } else {
                tracing::info!(usdc = %initial, "main: initial paper deposit recorded");
            }
        }
    }

    // Shutdown channel (API → main).
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_tx = Arc::new(tokio::sync::Mutex::new(Some(shutdown_tx)));

    // Oracle shutdown channel (watch to broadcast to all the price tasks).
    let (kill_tx, kill_rx) = tokio::sync::watch::channel(false);

    // Order pipeline channels.
    let (intent_tx, intent_rx) = mpsc::channel::<QueuedIntent>(512);
    let (fill_tx, _) = tokio::sync::broadcast::channel::<Fill>(1024);
    // Fill_sink exclusive channel: unbounded so it never loses fills under SQLite pressure.
    let (fill_persist_tx, fill_persist_rx) = mpsc::unbounded_channel::<Fill>();
    // LIVE orphan fills (market not found in the registry nor
    // in clear_at_cache) → dead-letter in the DB instead of just a log.
    let (dead_letter_tx, dead_letter_rx) =
        mpsc::unbounded_channel::<poly_bot::types::DeadLetterFill>();
    // Mapping (market_cid:token_id) → StrategyId, shared between executor and fill_sink.
    let order_map = std::sync::Arc::new(parking_lot::Mutex::new(std::collections::HashMap::<
        String,
        StrategyId,
    >::new()));

    // Book broadcast channel.
    // 4096 slots: ~725 live cids emit 1-2 msg/s each in bursts; 1024
    // only covered ~1s of burst and a lagging consumer fell into `Lagged`
    // (lost books). 4096 gives ~3s of headroom for ~32KB of memory.
    let (book_tx, _) = tokio::sync::broadcast::channel::<Arc<BookSnapshot>>(4096);

    // Market registry shared between market_feed and strategies.
    let registry = Arc::new(MarketRegistry::new());

    // ── Router ────────────────────────────────────────────────────────────────

    let mut sim_pump: Option<Arc<SimulatedRouter>> = None;
    let mut live_creds: Option<Arc<poly_bot::types::DerivedCreds>> = None;

    // Wallet balance watch: paper = sim deducts per fill, live = polled CLOB.
    let (wallet_bal_tx, wallet_bal_rx) =
        tokio::sync::watch::channel::<Option<rust_decimal::Decimal>>(None);
    // Option for selective consumption: Paper moves it to the sim, Live to the balance task.
    let mut wallet_bal_tx = Some(wallet_bal_tx);

    // In live: we keep the CLOB client's Arc to spawn the balance poller after
    // creating the JoinSet (tasks do not exist yet at this point).
    let mut live_clob_for_balance: Option<Arc<poly_bot::clob_sdk::AuthedClobClient>> = None;
    let mut live_clob_for_shutdown: Option<Arc<poly_bot::clob_sdk::AuthedClobClient>> = None;
    let mut live_clob_for_reconcile: Option<Arc<poly_bot::clob_sdk::AuthedClobClient>> = None;
    // On-chain redeemer for the orphan checker (live only with
    // auto_redeem=false; prior validation in Config::validate requires rpc_url).
    let mut redeemer_for_orphan: Option<Arc<dyn poly_bot::settlement::redeem::ChainRedeemer>> = None;

    let router: Arc<dyn OrderRouter> = match config.mode {
        Mode::Live => {
            let (bundle, creds) = clob_sdk::bootstrap(&config).await.unwrap_or_else(|e| {
                tracing::error!("clob_sdk::bootstrap failed in live mode: {e}");
                std::process::exit(1);
            });
            // `own_address` (EOA or proxy depending on wallet_mode) — needed by
            // `trades_since` to reconcile taker Unknowns via /data/trades.
            let own_address = creds
                .funder_address
                .clone()
                .unwrap_or_else(|| creds.eoa_address.clone());
            live_creds = Some(Arc::new(creds));
            let bundle_arc = Arc::new(bundle);
            live_clob_for_balance = Some(Arc::clone(&bundle_arc));
            live_clob_for_shutdown = Some(Arc::clone(&bundle_arc));
            live_clob_for_reconcile = Some(Arc::clone(&bundle_arc));
            // With auto_redeem=false the winners are redeemed
            // on-chain from the EOA. `Config::validate` already required rpc_url.
            if !config.risk.auto_redeem {
                let funder_addr = live_creds
                    .as_ref()
                    .and_then(|c| c.funder_address.as_deref())
                    .and_then(|h| h.parse::<alloy::primitives::Address>().ok());
                match poly_bot::settlement::redeem::CtfRedeemer::new(
                    &config.redeem.rpc_url,
                    Arc::clone(&bundle_arc.signer),
                    funder_addr,
                    &config.redeem.collateral,
                ) {
                    Ok(r) => {
                        redeemer_for_orphan = Some(Arc::new(r));
                    }
                    Err(e) => {
                        tracing::error!("redeem: could not build the redeemer: {e}");
                        std::process::exit(1);
                    }
                }
            }
            poly_bot::order::clob_router::ClobRouter::new(bundle_arc, own_address)
        }
        Mode::Paper => {
            let sim = SimulatedRouter::new(
                config.risk.paper_slippage_bps,
                config.risk.paper_fok_fill_rate,
                config.risk.paper_latency_ms,
                config.risk.paper_taker_fee_rate,
                config.risk.paper_pair_fill_correlation,
                fill_persist_tx.clone(), // fills → fill_sink (same as live user_ws)
                registry.clone(),        // neg_risk/clear_at via handle_trade
                config.risk.paper_initial_deposit_usdc,
                wallet_bal_tx.take().unwrap(), // the sim manages the paper balance
            );
            // Sim chaos: `paper_unknown_rate` injects Unknowns into
            // accepted submits; 0.0 (default) = the usual deterministic sim.
            sim.set_unknown_rate(config.risk.paper_unknown_rate);
            // Per-strategy FOK filled overrides (see general.toml).
            sim.set_fok_fill_rate_per_strategy(
                config
                    .risk
                    .paper_fok_fill_rate_per_strategy
                    .iter()
                    .map(|(name_str, rate)| {
                        let id = StrategyId::from_str(name_str).unwrap_or_else(|_| {
                            tracing::error!(strategy = %name_str,
                                "paper_fok_fill_rate_per_strategy: unknown strategy");
                            std::process::exit(1);
                        });
                        (id, *rate)
                    })
                    .collect(),
            );
            sim_pump = Some(sim.clone());
            sim
        }
    };

    // ── Executor ──────────────────────────────────────────────────────────────

    // VolSurface and LadderArb are NOT here: they are tools (`crate::tools`),
    // outside RiskGuard by design (no kill switch, no rate limits).
    //
    // `Follower` is not in `strategy::registry` (it does not
    // implement `Strategy`, standalone task — see the doc comment of
    // `strategy::follower`), so `register_limits` does not cover it. Its
    // limits are still registered by hand here, as before this change.
    risk.set_strategy_limits(
        StrategyId::Follower,
        config.follower.limits.max_orders_per_hour,
        config.follower.limits.max_order_usdc,
        config.follower.limits.daily_loss_usdc_stop,
    );
    // Follower does not go through `register_limits` (it is not in the
    // `register_strategies!` table), so the barrier is applied here — same
    // decision function, without duplicating the logic.
    if let Err(reason) = config.follower.limits.can_run(config.mode) {
        risk.kill_strategy(
            StrategyId::Follower,
            format!("on startup in {} mode: {reason}", config.mode),
        );
    }

    // Registration of the real strategies' limits (12, all except
    // Follower) — MUST happen here, before spawning executor/API/orphan,
    // not inside `strategy::registry::spawn_all` (which runs much later,
    // later, after those tasks). If it were registered late, `RiskGuard::
    // get_strategy_limits` would return hardcoded defaults during that
    // window, and a strategy with `enabled=false` in TOML would not have its
    // `kill_strategy` applied until the spawn — real risk with money at
    // stake. Restored to its original position.
    strategy::registry::register_limits(&config, &risk);

    // Global exposure carve-out exclusive to post_close_sweep (almost
    // deterministic verdict) — see the doc in risk.rs and config/strategies/post_close_sweep.toml.
    risk.set_strategy_exposure_carveout(
        StrategyId::PostCloseSweep,
        config.strategies.post_close_sweep.exposure_carveout_usdc,
    );

    // Set of order_ids emitted by the bot in this session.
    // The executor registers them; user_ws filters out foreign MATCHED (old orders, etc.).
    let known_order_ids = Arc::new(dashmap::DashSet::<String>::new());

    // The research tape producer hub (src/research/events.rs) is created
    // ALWAYS, enabled or not — it is the stable identity that survives any
    // writer startup/stop. The wiring below (executor,
    // oracle, market_feed…) injects `research_hub.producer()` without having to be
    // touched again when the tool is switched on/off hot:
    // `EventProducer::record` drops at no cost while there is no `attach`ed
    // writer. `ResearchRecorder::start`/`stop` over
    // this hub is now managed by `ToolSupervisor` as `ToolId::ResearchEvents`
    // (see `tool_supervisor.set_recorder_deps`/`spawn_enabled_at_boot` further
    // down) — here only the `Hub` is created, never the recorder.
    let research_hub = poly_bot::research::events::Hub::new();
    let research_prod = research_hub.producer();

    // Strategies configured with use_fak=true → the executor uses TimeInForce::Fak.
    let mut fak_strategies = std::collections::HashSet::new();
    if config.strategies.near_certain.use_fak {
        fak_strategies.insert(poly_bot::types::StrategyId::NearCertain);
    }
    if config.strategies.near_expiry.use_fak {
        fak_strategies.insert(poly_bot::types::StrategyId::NearExpiry);
    }

    let mut executor = Executor::new(
        intent_rx,
        fill_persist_tx.clone(),
        router,
        risk.clone(),
        state.clone(),
        order_map.clone(),
        wallet_bal_rx.clone(),
        known_order_ids.clone(),
        fak_strategies,
        ledger.clone(),
    );
    // The executor emits `OrderSubmit` (the pre-result OrderRequest) to the
    // recorder when it is enabled.
    executor.set_research_producer(research_prod.clone());

    // Seed the in-memory mirror of
    // `unknown_attempts` with the durable journal's `pending` entries BEFORE the
    // startup reconciliation, any strategy spawn and the
    // first `purge_and_resync`. Without this seeding the mirror starts empty
    // after a restart: the resync rebuilds claims from the mirror and
    // would release those of Unknowns that may still be live on the CLOB
    // (fail-open), and `link_quarantine_to_journal` could not link
    // late fills with their attempt. If the read fails, the executor does not
    // abort — the `has_unresolved_unknown` blocker queries the DB
    // directly and stays active.
    executor.seed_unknown_attempts_from_journal_pub().await;

    // Startup reconciliation (live): re-register GTDs with an active position,
    // cancel stale ones. In paper: no-op (there is no CLOB client, so this whole
    // block is skipped — hence the asymmetry of `must_abort_startup`
    // has no practical effect in paper; see its doc comment).
    if let Some(ref bundle) = live_clob_for_reconcile {
        let reconciliation_ok = reconcile_open_orders(bundle, &state, &mut executor).await;
        // Fail CLOSED. Previously "operating without
        // reconciliation" was logged and startup continued: strategies were spawned
        // without knowing which orders were open on the CLOB. We abort HERE,
        // before any spawn (including the manual ones of Follower and
        // post_close_sweep) and without touching the `StrategyLimits::can_run` barrier.
        if poly_bot::startup::must_abort_startup(config.mode, reconciliation_ok) {
            tracing::error!(
                "FATAL: could not fetch the open GTD orders from the CLOB. In live, \
                 startup fails CLOSED: no strategy is spawned. Check \
                 CLOB connectivity and credentials and start again; if there are resting GTDs \
                 you want to withdraw in the meantime, cancel them from the Polymarket website."
            );
            std::process::exit(1);
        }
        // Backfill of fills lost during downtime — a GTD that filled
        // while the bot was down left a real unregistered position that the orphan
        // checker would never liquidate. The fill_sink's dedup by trade-id makes
        // repeating trades already seen by user_ws harmless.
        if let Some(ref creds) = live_creds {
            backfill_missed_trades(bundle, &state, creds, &fill_persist_tx).await;
        }
    }

    // ── Oracle channels ───────────────────────────────────────────────────────

    let (price_txs, price_rxs) =
        oracle::price::create_price_channels(&config.market_feed.bracket_slug_patterns);
    let assets: Vec<String> = price_txs.keys().cloned().collect();
    let (vol_txs, vol_rxs) = oracle::vol::create_vol_channels(&assets);

    // Shared store of strikes (Chainlink 60 s TWAP). Filled by
    // `oracle::chainlink_twap` and read by the strategy; the streak telemetry
    // goes to the same SQLite as the rest of the accounting.
    let arb15m5m_deps = strategy::Arb15m5mDeps::from_config(
        oracle::chainlink_twap::TwapStore::new(),
        state.clone(),
        &config,
    );

    // ── Tools (scan + recorders, outside RiskGuard) ───────────────────
    // Scan observations sink: LadderArb/VolSurface → scan_observations.
    // Created here (before ApiState) because the ToolSupervisor —which the API and
    // Telegram need for the hot toggle— requires it in its constructor.
    let (scan_tx, scan_rx) = tokio::sync::mpsc::unbounded_channel();
    // Broadcast channel shared between `ToolId::ModelTool`
    // (producer of P(YES) predictions) and the `ModelBtc5m` strategy
    // (consumer). Injected into the supervisor and the registry's BuildCtx
    // BEFORE spawning strategies — they are clones of the same `Arc<Sender>`,
    // not separate channels. Channel size: `broadcast_capacity` of
    // `model_tool` (16 by default). The tool emits predictions at this
    // rate and the strategy consumes every book; 16 leaves ample headroom
    // for 5-min up/down brackets.
    let (model_prediction_tx_raw, _) = tokio::sync::broadcast::channel::<
        poly_bot::tools::model::ModelPrediction,
    >(config.model_tool.broadcast_capacity);
    let model_prediction_tx = Arc::new(model_prediction_tx_raw);
    let tool_supervisor = ToolSupervisor::new(
        book_tx.clone(),
        vol_rxs.clone(),
        scan_tx.clone(),
        kill_rx.clone(),
        std::time::Duration::from_millis(config.market_feed.max_book_age_ms),
        config.clone(),
    );
    // Inject the `Arc<Sender>` BEFORE `spawn_enabled_at_boot`
    // so that `ToolId::ModelTool` can be spawned if it is enabled in
    // the TOML. The call is idempotent (internal singleton).
    tool_supervisor.set_model_prediction_tx(model_prediction_tx.clone());

    // Startup of the enabled tools
    // (`spawn_enabled_at_boot`, scan + recorders) must happen HERE,
    // BEFORE spawning `oracle::price`/`market_feed::run` further down — not at the
    // end of startup like the rest of the tasks. Reason (the regression this
    // reordering fixes): the research tape (`research_events`)
    // depends on the hub being `active` in two order-sensitive places:
    // (1) the market meta emitted by `market_feed::discover` in its
    // initial discovery and the first books/ticks of the oracle are
    // lost forever if the hub starts inactive — without that meta the
    // replay cannot reconstruct those markets; (2) `market_feed::ws`
    // decides whether to subscribe the WS tape (`record_tape`) ONCE per session,
    // so if that first session starts with the hub inactive, there is no
    // tape until the NEXT reconnect of the shard (silent regression,
    // with no visible error). `ToolSupervisor::spawn` for
    // `ToolId::ResearchEvents` does the `attach()` synchronously (via
    // `spawn_blocking().await` inside `spawn_recorder_task`), so
    // just `.await`ing it before the two tasks above is enough — no
    // additional synchronization channel (oneshot, barrier…) is needed
    // between this call and the later ones. `set_recorder_deps` is now an
    // instance method of `tool_supervisor` itself (previously a
    // process-wide `static` `OnceLock`).
    tool_supervisor.set_recorder_deps(poly_bot::tools::RecorderDeps {
        registry: registry.clone(),
        state: state.clone(),
        research_hub: research_hub.clone(),
    });
    tool_supervisor.spawn_enabled_at_boot().await;

    // Dynamic adds/removes of the market WS (`operation:subscribe`/
    // `operation:unsubscribe` — it turned out to be an artifact of the
    // `type:subscribe` format, verified with a live probe). Producers: discover
    // (new WS-bound brackets + evictions), prefetch (anticipated
    // up/down brackets) and the strategy supervisor (poll↔WS migration
    // on toggle). The shard manager resubscribes on each iteration of the
    // feed's restart loop — hence broadcast and not mpsc.
    let (token_event_tx, _) =
        tokio::sync::broadcast::channel::<poly_bot::market_feed::TokenEvent>(256);

    // Same reason as `tool_supervisor` above — created here
    // (before `ApiState`) because the API and Telegram need it for the
    // hot toggle of strategies that start with `enabled=false`.
    let strategy_supervisor = strategy::StrategySupervisor::new(
        book_tx.clone(),
        vol_rxs.clone(),
        intent_tx.clone(),
        fill_tx.clone(),
        risk.clone(),
        kill_rx.clone(),
        std::time::Duration::from_millis(config.market_feed.max_book_age_ms),
        registry.clone(),
        config.clone(),
        ledger.clone(),
        token_event_tx.clone(),
        // Same `Arc<Sender>` as the tool — if a hot
        // `set_enabled(ModelBtc5m, true)` arrives, the supervisor
        // builds the strategy on demand with this sender.
        Some(model_prediction_tx.clone()),
        arb15m5m_deps.clone(),
    );

    // ── API state ─────────────────────────────────────────────────────────────

    let topup_bal_rx = wallet_bal_rx.clone(); // for the paper top-up task
    let mark_price_cache = poly_bot::accounting::mark_price::MarkPriceCache::new();

    let api_state = ApiState {
        config: config.clone(),
        risk: risk.clone(),
        state: state.clone(),
        token: Arc::new(config.api.token.clone()),
        mode: config.mode,
        shutdown_tx: shutdown_tx.clone(),
        prometheus_handle: Arc::new(prometheus_handle),
        wallet_balance: wallet_bal_rx,
        tools: tool_supervisor.clone(),
        strategies: strategy_supervisor.clone(),
        alert_log: alert_log.clone(),
        vol_rxs: Arc::new(vol_rxs.clone()),
        ledger: ledger.clone(),
        mark_price: mark_price_cache.clone(),
    };

    // ── Task spawning ─────────────────────────────────────────────────────────

    let mut tasks = tokio::task::JoinSet::new();
    // `task::Id` is an internal tokio number, undiagnosable at a
    // glance in the shutdown log further down — this mapping translates it
    // to a readable name at spawn time, without depending on
    // `tokio::task::Builder::name` (which would require rewriting every
    // `tasks.spawn(...)` to `tasks.build_task().name(...).spawn(...)` AND that the
    // name survive until the final `JoinError`/output, which tokio's stable API
    // does not expose).
    let mut task_names: std::collections::HashMap<tokio::task::Id, String> =
        std::collections::HashMap::new();
    // Handles by name for the phased shutdown — the fill_sink is
    // aborted LAST (after draining), the fill producers before.
    let mut abort_handles: Vec<(String, tokio::task::AbortHandle)> = Vec::new();
    macro_rules! spawn_named {
        ($name:expr, $fut:expr) => {{
            let h = tasks.spawn($fut);
            task_names.insert(h.id(), $name.to_string());
            abort_handles.push(($name.to_string(), h));
        }};
    }

    // CLOB balance polling in live (guarded before creating tasks).
    if let (Some(bal_clob), Some(bal_tx)) = (live_clob_for_balance, wallet_bal_tx) {
        let mut bal_kill = kill_rx.clone();
        let state_for_dep = state.clone();
        spawn_named!("wallet_balance_poll", async move {
            let mut delay = std::time::Duration::from_secs(30);
            let mut capital_saved = false;
            loop {
                tokio::select! {
                    _ = bal_kill.changed() => { if *bal_kill.borrow() { return; } }
                    _ = tokio::time::sleep(delay) => {
                        match clob_sdk::fetch_balance(&bal_clob).await {
                            Ok(b) => {
                                // On the first successful fetch, save as initial capital if the DB
                                // has no registered deposits (first startup in live).
                                if !capital_saved {
                                    capital_saved = true;
                                    match state_for_dep.deposits_total().await {
                                        Ok(d) if d == rust_decimal::Decimal::ZERO => {
                                            if let Err(e) = state_for_dep.add_deposit(b, "initial live capital").await {
                                                tracing::warn!(error = %e, "main: could not register initial capital");
                                            } else {
                                                tracing::info!(capital = %b, "main: initial live capital recorded");
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                let _ = bal_tx.send(Some(b));
                                delay = std::time::Duration::from_secs(30);
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "main: fetch_balance failed");
                                delay = (delay * 2).min(std::time::Duration::from_secs(300));
                            }
                        }
                    }
                }
            }
        });
    }

    {
        let oracle_cfg = Arc::new(config.oracle.clone());
        spawn_named!(
            "oracle_price",
            // With the recorder enabled every raw Binance tick enters the
            // dataset (venue seq + generation per reconnect).
            oracle::price::run_with_recorder(
                oracle_cfg,
                price_txs,
                kill_rx.clone(),
                research_prod.clone(),
            )
        );
    }

    // Chainlink 60 s TWAP feed. Runs whenever
    // L2 credentials exist, even if the strategy is switched off: a
    // pair's strike is the TWAP at its `t0`, so the feed must be alive BEFORE
    // the market opens for a hot toggle to have strikes.
    match oracle::chainlink_twap::Credentials::from_env() {
        Some(creds) => {
            spawn_named!(
                "oracle_chainlink_twap",
                oracle::chainlink_twap::run(
                    arb15m5m_deps.twap.clone(),
                    creds,
                    assets.clone(),
                    kill_rx.clone(),
                )
            );
        }
        None => tracing::warn!(
            "oracle::chainlink_twap: missing POLY_API_KEY/SECRET/PASSPHRASE — the TWAP feed does not start \
             and arb_15m5m will have no strikes (it will not trade)"
        ),
    }

    // ── Binance futures liquidations (observation) ───────────────────────────
    // Map that oracle::vol uses for the net/total liquidations
    // window (`VolSnapshot::liq_*`). Empty if futures_liq_enabled=false → behavior
    // identical to before this pipe.
    let mut liq_rxs_for_vol: std::collections::HashMap<
        String,
        tokio::sync::watch::Receiver<poly_bot::types::LiqCum>,
    > = std::collections::HashMap::new();
    if config.oracle.futures_liq_enabled {
        let (liq_txs, liq_rxs) =
            oracle::liquidations::create_liq_channels(&config.market_feed.bracket_slug_patterns);
        liq_rxs_for_vol = liq_rxs;

        {
            let oracle_cfg = Arc::new(config.oracle.clone());
            spawn_named!(
                "oracle_liquidations",
                oracle::liquidations::run(oracle_cfg, liq_txs, kill_rx.clone())
            );
        }

        tracing::info!("oracle::liquidations: futures liquidations feed enabled");
    }

    oracle::vol::spawn_all(price_rxs, vol_txs, liq_rxs_for_vol, kill_rx.clone());

    // Position reconciler (live only): periodically compares against
    // the Polymarket Data API to detect drift between what we believe we hold
    // and what is really on-chain — safety net independent of
    // any internal accounting bug.
    if let Some(ref creds) = live_creds {
        let own_address = creds
            .funder_address
            .clone()
            .unwrap_or_else(|| creds.eoa_address.clone());
        spawn_named!(
            "position_reconciler",
            poly_bot::settlement::position_reconciler::run(own_address, state.clone(), kill_rx.clone())
        );
    }

    spawn_named!(
        "market_feed",
        market_feed::run(
            book_tx.clone(),
            config.clone(),
            registry.clone(),
            state.clone(),
            live_creds,
            fill_persist_tx.clone(),
            known_order_ids.clone(),
            live_clob_for_reconcile.clone(), // enables backfill_gap in user_ws
            Some(dead_letter_tx.clone()),    // dead-letter of orphan fills
            token_event_tx.clone(),
            research_prod.clone(), // WS/REST hooks of the recorder
        )
    );

    // Stamps reference_price from the oracle when each bracket opens —
    // Gamma gives no openPrice on up/down; without this NC/NE/MM/… have no strike.
    spawn_named!(
        "market_feed_reference",
        market_feed::reference::run(registry.clone(), vol_rxs.clone(), kill_rx.clone(),)
    );

    spawn_named!("executor", executor.run());

    spawn_named!("api_serve", api::serve(api_state, None));

    // ── GTD driver for SimulatedRouter (paper only) ───────────────────────────
    // Advances the GTD queue on every book tick. Fills go to fill_sink via handle_trade
    // (same code as live user_ws — full paper/live parity).
    // `sim_pump.clone()` — the orphan checker (spawned further down) also
    // receives the Arc to credit redemptions in the paper cash.
    if let Some(sim) = sim_pump.clone() {
        // ── Automatic paper top-up: if balance < $100 → add $1000 ─────────────
        {
            use rust_decimal_macros::dec;
            let sim_ref = sim.clone();
            let state_ref = state.clone();
            let mut bal_rx = topup_bal_rx;
            let mut kill_tp = kill_rx.clone();
            spawn_named!("paper_topup", async move {
                loop {
                    tokio::select! {
                        res = bal_rx.changed() => {
                            if res.is_err() { break; }
                            // Copy the value out of the Ref before the .await (Ref is not Send).
                            let bal_opt: Option<rust_decimal::Decimal> = *bal_rx.borrow_and_update();
                            if let Some(bal) = bal_opt {
                                if bal < dec!(100) {
                                    sim_ref.add_funds(dec!(1000));
                                    if let Err(e) = state_ref.add_deposit(dec!(1000), poly_bot::state::TOPUP_NOTE).await {
                                        tracing::warn!("paper topup: error recording deposit: {e}");
                                    }
                                    tracing::info!(previous_balance = %bal, "paper topup: balance < $100 — added $1000");
                                }
                            }
                        }
                        _ = kill_tp.changed() => { if *kill_tp.borrow() { break; } }
                    }
                }
            });
        }

        let mut pump_rx = book_tx.subscribe();
        spawn_named!("sim_pump", async move {
            // `while let Ok(...)` treated ANY `Err` of
            // `recv()` as the end of the task — but `RecvError::Lagged(n)` is
            // RECOVERABLE (the consumer fell behind the 1024-book
            // buffer), not a channel close. Since any task that returns
            // triggers `kill_tx.send(true)` + `abort_all()` in the shutdown
            // select further down (without distinguishing the reason), a mere lag of the
            // pump shut down the WHOLE bot — a plausible cause of calibration
            // stops with no apparent cause. Same criterion as
            // `drain_latest` in `strategy/mod.rs`: Lagged is ignored and we continue,
            // only `Closed` ends the task.
            loop {
                match pump_rx.recv().await {
                    Ok(book) => sim.tick_book(&book),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(n, "sim pump: book(s) lost due to lag, continuing");
                        continue;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        tracing::info!("sim pump: books channel closed, exiting");
                        break;
                    }
                }
            }
        });
    }

    // ── Fill sink ─────────────────────────────────────────────────────────────

    // The fill_sink signals through this oneshot when it has finished
    // draining the in-flight fills. Shutdown waits for it (with a deadline) BEFORE
    // `abort_all()` — see `FILLS_SHUTDOWN_GRACE` further down.
    let (fill_sink_done_tx, fill_sink_done_rx) = oneshot::channel::<()>();
    spawn_named!(
        "fill_sink",
        poly_bot::accounting::fill_sink::run(
            fill_persist_rx,
            state.clone(),
            risk.clone(),
            ledger.clone(),    // release of bracket exposure on sell
            fill_tx.clone(),   // fill_sink re-broadcast a strategies/Telegram (C1)
            order_map.clone(), // strategy resolution for user_ws fills
            kill_rx.clone(),
            fill_sink_done_tx,
        )
    );

    // Dead-letter of orphan LIVE fills — consumer separate from the
    // fill_sink above, best-effort (see the doc comment of `run_dead_letters`).
    spawn_named!(
        "dead_letter_sink",
        poly_bot::accounting::fill_sink::run_dead_letters(dead_letter_rx, state.clone(), kill_rx.clone(),)
    );

    // scan_tx/scan_rx were created earlier (together with the ToolSupervisor) — here only
    // the consumer that persists into scan_observations is spawned.
    spawn_named!(
        "scan_sink",
        poly_bot::tools::scan::sink::run(state.clone(), scan_rx, kill_rx.clone())
    );

    // Post-fill markouts: mid at +5s/+30s of each fill → markouts table.
    spawn_named!(
        "markout",
        poly_bot::accounting::markout::run(
            state.clone(),
            fill_tx.subscribe(),
            book_tx.subscribe(),
            kill_rx.clone(),
            std::time::Duration::from_secs(5),
            std::time::Duration::from_secs(30),
        )
    );

    // Passive post-close observer (emits no orders).
    if config.post_close.enabled {
        spawn_named!(
            "post_close",
            poly_bot::settlement::post_close::run(
                state.clone(),
                book_tx.subscribe(),
                vol_rxs.clone(),
                std::time::Duration::from_millis(config.post_close.sample_throttle_ms),
                kill_rx.clone(),
            )
        );
    }

    // Official resolutions fetcher — turns the closed
    // brackets into `confirmed` labels by querying Gamma (provisional →
    // confirmed, correction → supersede). HTTP read-only; never trades.
    // It is now `ToolId::ResolutionFetcher`, managed by
    // `ToolSupervisor` (see `spawn_enabled_at_boot` further up, next to the
    // construction of `tool_supervisor`) — no longer spawned here.

    // Expired-positions checker: queries Gamma every 5 min and resolves outcome + PnL.
    // In paper the sim is passed so settlement credits the payout to
    // the cash and debits the inventory (`credit_redemption`, idempotent).
    spawn_named!(
        "orphan",
        poly_bot::settlement::orphan::run(
            state.clone(),
            risk.clone(),
            None,
            kill_rx.clone(),
            redeemer_for_orphan,
            sim_pump.clone(),
        )
    );

    // Real equity (cash + mark-to-market value of open positions) — see the doc comment
    // of src/accounting/mark_price.rs on why this lives in a cached background task and
    // is not computed on the fly on every UI request.
    spawn_named!(
        "mark_price",
        poly_bot::accounting::mark_price::run(
            mark_price_cache.clone(),
            state.clone(),
            registry.clone(),
            kill_rx.clone(),
        )
    );

    // Wallet follower — standalone task (NOT a `Strategy`,
    // see the doc comment of `strategy::follower`), direct spawn conditioned on
    // `enabled` (same criterion as `post_close` above). Sends its OrderIntents
    // through the same `intent_tx` as the real strategies — kill switch/rate
    // limit/daily loss stop were already registered above along with the others.
    if config.follower.limits.can_run(config.mode).is_ok() {
        spawn_named!(
            "follower",
            strategy::follower::run(
                state.clone(),
                intent_tx.clone(),
                config.follower.clone(),
                kill_rx.clone(),
            )
        );
    }

    // ── Telegram bot (optional) ──────────────────────────────────────────────────
    // The token comes from the env, never from the TOML config.

    let tg_token = std::env::var("TELEGRAM_BOT_TOKEN").unwrap_or_default();
    if !tg_token.is_empty() {
        let tg_allowed: Vec<i64> = std::env::var("TELEGRAM_ALLOWED_IDS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        spawn_named!(
            "telegram",
            poly_bot::telegram::run(
                tg_token,
                tg_allowed,
                config.telegram.clone(),
                state.clone(),
                risk.clone(),
                fill_tx.subscribe(),
                kill_rx.clone(),
                log_buf.clone(),
                tool_supervisor.clone(),
                strategy_supervisor.clone(),
            )
        );
    }

    // ── Strategies ───────────────────────────────────────────────────────────
    // Single registry in `strategy::registry` — adding/removing a strategy
    // no longer touches this file except for the special case below (post_close_sweep).

    let build_ctx = BuildCtx {
        config: config.clone(),
        registry: registry.clone(),
        ledger: ledger.clone(),
        // The `model_prediction_tx` broadcast channel is created
        // BEFORE the tool and injected into the supervisor and the BuildCtx —
        // they are clones of the same `Arc<Sender>`, NOT separate channels.
        // The loaded-model tool (`tools::model::ToolModel`) emits
        // predictions through this channel and `ModelBtc5m` subscribes
        // via the registry's BuildCtx. Without this injection,
        // `registry::spawn_all` would abort loudly when building
        // `ModelBtc5m` (with `enabled=true`) and the `ToolSupervisor`
        // would reject the spawn of `ToolId::ModelTool`. Since
        // `model_btc5m.limits.enabled=false` in the default TOML,
        // the registry does NOT build the strategy — but we still
        // populate the channel so the runtime supervisor can
        // accept `set_enabled(true)` or `spawn_enabled_at_boot` when
        // the operator enables it without a restart.
        model_prediction_tx: tool_supervisor.model_prediction_tx(),
        arb15m5m: arb15m5m_deps.clone(),
    };

    // Builds the instance BEFORE the generic spawn to extract
    // `taker_poller_handles()` from that same instance — it cannot be
    // rebuilt twice. That is why it stays out of the automatic spawn
    // (`manual_spawn` below) although it DOES take part in the table for
    // config/limits/requires_oracle (see `strategy/registry.rs`).
    let pcs2_lim = &config.strategies.post_close_sweep.limits;
    // post_close_sweep is spawned by hand (it is in `manual_spawn`), so the
    // barrier that `spawn_all` applies would not cover it — it is evaluated once here
    // and reused in the two `if`s below.
    let pcs2_allowed = pcs2_lim.can_run(config.mode).is_ok();
    let mut pcs2 = PostCloseSweep::new(config.strategies.post_close_sweep.clone());
    if pcs2_allowed && config.strategies.post_close_sweep.taker_enabled {
        let (pcs2_poll_rx, pcs2_taker_ready) = pcs2.taker_poller_handles();
        spawn_named!(
            "pcs2_taker_poller",
            poly_bot::strategy::post_close_sweep::run_taker_poller(
                pcs2_poll_rx,
                pcs2_taker_ready,
                kill_rx.clone(),
            )
        );
    }

    strategy::registry::spawn_all(
        &build_ctx,
        &risk,
        &mut tasks,
        &mut task_names,
        &book_tx,
        &vol_rxs,
        &intent_tx,
        &fill_tx,
        &kill_rx,
        config.market_feed.max_book_age_ms,
        &[StrategyId::PostCloseSweep],
    );

    if pcs2_allowed {
        let strategy_ctx = StrategyCtx {
            book_rx: book_tx.subscribe(),
            vol_rxs: vol_rxs.clone(),
            intent_tx: intent_tx.clone(),
            fill_rx: fill_tx.subscribe(),
            risk: risk.clone(),
            shutdown: kill_rx.clone(),
            max_book_age: std::time::Duration::from_millis(config.market_feed.max_book_age_ms),
            registry: registry.clone(),
            universe: pcs2_lim.universe.clone(),
            requires_oracle: strategy::registry::strategy_requires_oracle(
                StrategyId::PostCloseSweep,
            ),
        };
        spawn_named!(
            format!("strategy:{}", StrategyId::PostCloseSweep),
            strategy::run(Box::new(pcs2), strategy_ctx)
        );
    }

    // Shadow observer for the numeric model (`model_observer.rs`).
    // Pure observer: it consumes the book broadcast and the vol watch and logs VIRTUAL decisions to its own JSONL.
    // It never emits OrderIntent nor touches RiskGuard/executor/ledger (its `ShadowCtx` does not carry those channels: this is structural, not a convention).
    // No gate enables live: the shadow record is evidence for a manual go/no-go decision, not a path to real orders.
    // An invalid artifact or a writer that fails to start only logs an error: the observer never aborts the bot.
    let mut shadow_writer = None;
    if config.model_shadow.enabled {
        let artifact_dir = std::path::PathBuf::from(&config.model_shadow.artifact_dir);
        match poly_bot::model_observer::ShadowModel::load(&artifact_dir) {
            Ok(loaded_model) => {
                match poly_bot::model_observer::ShadowWriter::start(
                    std::path::Path::new(&config.model_shadow.decisions_dir),
                    config.model_shadow.channel_capacity,
                ) {
                    Ok(writer) => {
                        let shadow_ctx = poly_bot::model_observer::ShadowCtx {
                            book_rx: book_tx.subscribe(),
                            vol_rxs: vol_rxs.clone(),
                            shutdown: kill_rx.clone(),
                            max_book_age: std::time::Duration::from_millis(
                                config.market_feed.max_book_age_ms,
                            ),
                        };
                        spawn_named!(
                            "model_shadow",
                            poly_bot::model_observer::run_observer(
                                std::sync::Arc::new(loaded_model),
                                config.model_shadow.clone(),
                                shadow_ctx,
                                writer.producer(),
                            )
                        );
                        shadow_writer = Some(writer);
                        tracing::info!(
                            artifact = %config.model_shadow.artifact_dir,
                            decisions_dir = %config.model_shadow.decisions_dir,
                            "model_shadow: observer attached (shadow mode, never trades)"
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            dir = %config.model_shadow.decisions_dir,
                            "model_shadow: could not start the writer, observer disabled"
                        );
                    }
                }
            }
            Err(e) => {
                tracing::error!(
                    error = %e,
                    artifact = %config.model_shadow.artifact_dir,
                    "model_shadow: artifact rejected — observer disabled"
                );
            }
        }
    }

    // book_tx and fill_tx can be dropped: the subscribers already hold their receivers.
    // intent_tx is NOT dropped here: if no strategy is active, the channel would have no senders and the executor would exit immediately. It is dropped after shutdown.
    drop(book_tx);
    drop(fill_tx);
    drop(fill_persist_tx);

    tracing::info!("poly-bot v3: all tasks started");

    // OS signals. Without this, `systemctl stop|restart` (SIGTERM) and Ctrl-C (SIGINT) used the DEFAULT disposition:
    // immediate process death, GTD resting orders not cancelled on the CLOB, the analytics writer queue not drained,
    // and the SQLite WAL left dirty. With the bot under systemd (`deploy/poly-bot.service`, `Restart=always`) this would
    // happen on EVERY restart, so an orderly shutdown is mandatory.
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| anyhow::anyhow!("could not register the SIGTERM handler: {e}"))?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .map_err(|e| anyhow::anyhow!("could not register the SIGINT handler: {e}"))?;

    // Wait for a shutdown signal (OS or API) or for the first task to exit.
    tokio::select! {
        _ = shutdown_rx => {
            tracing::info!("poly-bot: shutdown signal received, stopping");
        }
        _ = sigterm.recv() => {
            tracing::info!("poly-bot: SIGTERM received, shutting down in order");
        }
        _ = sigint.recv() => {
            tracing::info!("poly-bot: SIGINT (Ctrl-C) received, shutting down in order");
        }
        // Name the exiting task: a bare JoinSet id makes a spontaneous shutdown impossible to diagnose.
        // `task_names` (filled by `spawn_named!` at each spawn above) maps the numeric `task::Id` to a readable name.
        res = tasks.join_next_with_id() => {
            match res {
                Some(Ok((id, ()))) => {
                    let name = task_names.get(&id).map(String::as_str).unwrap_or("unknown");
                    tracing::warn!(task_id = %id, task = name, "poly-bot: a task exited unexpectedly");
                }
                Some(Err(e)) => {
                    let name = task_names.get(&e.id()).map(String::as_str).unwrap_or("unknown");
                    tracing::error!(task_id = %e.id(), task = name, "poly-bot: task panicked: {e}");
                }
                None => tracing::info!("poly-bot: all tasks finished"),
            }
        }
    }

    let _ = kill_tx.send(true);
    drop(intent_tx); // closes the executor channel after the shutdown signal

    // Phased shutdown, to close the residual window between producer and sink.
    // fill_sink drains until the channel CLOSES (all senders gone), so the producers must die first: a short grace period
    // lets those that listen on kill_rx exit on their own (the executor exits when `intent_rx` closes; strategies, top-up,
    // analytics... in their shutdown branch), then the rest are aborted (market_feed has no kill channel, and a hung task must not delay shutdown).
    // Meanwhile the sink keeps draining with `recv`: every fill a producer enqueues before dying is persisted; nothing is lost silently.
    tokio::time::sleep(PRODUCERS_SHUTDOWN_GRACE).await;
    for (name, handle) in &abort_handles {
        if name != "fill_sink" {
            handle.abort();
        }
    }

    // Grace period BEFORE `abort_all()`. The one thing that must not be lost is fill persistence: a fill already received by
    // user_ws/sim and queued in the mpsc is money without a trace if the sink is aborted half-way. Strategy tasks are aborted without ceremony.
    //
    // Wait for the fill_sink's own completion signal (not `tasks.join_next()`), so we do not hang on the other JoinSet tasks,
    // and so the sink can still report its own panics and unexpected exit.
    // The deadline is deliberately longer than `accounting::fill_sink::SHUTDOWN_DRAIN_CAP`: the sink goes first, since it knows
    // how many fills are left; this is only the safety net if it gets stuck. It never hangs shutdown: on timeout it aborts and logs it
    // (systemd allows 90 s via `TimeoutStopSec`, plenty of slack).
    match tokio::time::timeout(FILLS_SHUTDOWN_GRACE, fill_sink_done_rx).await {
        Ok(Ok(())) => tracing::info!("poly-bot: fill_sink drained, no fills pending"),
        Ok(Err(_)) => tracing::warn!(
        "poly-bot: fill_sink exited without notice (panic or closed channel), \
             fills may remain unpersisted"
        ),
        Err(_) => tracing::error!(
            grace_s = FILLS_SHUTDOWN_GRACE.as_secs(),
        "poly-bot: fill_sink did not finish within the grace period, aborting anyway; \
             fills may remain unpersisted"
        ),
    }

    tasks.abort_all();

    // Cancel resting orders on shutdown so no GTD assets are left on the CLOB.
    // If the bot crashes before reaching this point, the next startup's cancel_all cleans up the rest.
    if let Some(sdk) = live_clob_for_shutdown {
        match clob_sdk::cancel_all_orders(&sdk).await {
            Ok(resp) => tracing::info!(
                cancelled = resp.canceled.len(),
                "poly-bot: orders cancelled on shutdown"
            ),
            Err(e) => tracing::warn!(
                error = %e,
                "poly-bot: cancel_all on shutdown failed, GTD orders may remain resting"
            ),
        }
    }

    // Orderly shutdown of ALL tools (scan + recorders): signal + join per tool, never an `abort()` in the middle of a recorder's flush
    // (the `ToolSupervisor` tasks are not in `tasks`, so the `abort_all()` above does not reach them). It runs AFTER `cancel_all` on purpose:
    // pulling resting orders off the CLOB is money, while this is research/observation. If shutdown gets stuck, what must not hang is the money path.
    // Recorders NO LONGER follow the global kill (`kill_tx.send(true)` above): if they did, they would disconnect right there,
    // BEFORE the grace/`abort_all` of the producers above, losing the queue of what those producers emit during that grace.
    // `shutdown_all` is the ONLY way a recorder stops, and it must run here, once that queue is complete, not earlier.
    // Internal order (scan + analytics/spot_stream/resolution_fetcher in parallel, research_events last with its own budget)
    // lives in `ToolSupervisor::shutdown_all`.
    tool_supervisor.shutdown_all().await;

    // Close the shadow writer, same approach as research_events: the observer is already dead (kill_rx/abort_all)
    // and its shadow producer dropped with the task; `close()` drains the queue and waits for the writer thread.
    if let Some(writer) = shadow_writer {
        let res = writer.close().await;
        tracing::info!(
            enqueued = res.enqueued,
            written = res.written,
            discarded = res.discarded,
            "poly-bot: model_shadow closed"
        );
    }

    tracing::info!("poly-bot: shutdown complete");
    Ok(())
}

/// Recovers trades executed while the bot was down (live).
///
/// Queries the CLOB since the timestamp of the last fill in the DB and re-injects the trades as `Fill`s through the fill_sink.
/// The unique index on order_id (trade UUID, the same one user_ws sends) discards the ones already recorded.
async fn backfill_missed_trades(
    bundle: &poly_bot::clob_sdk::AuthedClobClient,
    state: &StateStore,
    creds: &poly_bot::types::DerivedCreds,
    fill_tx: &mpsc::UnboundedSender<Fill>,
) {
    let last_ts_ms = match state.last_fill_ts_ms().await {
        Ok(Some(ts)) => ts,
        Ok(None) => {
            tracing::info!("startup: DB has no previous fills, backfill skipped");
            return;
        }
        Err(e) => {
            tracing::warn!(error = %e, "startup: could not load the last fill, backfill skipped");
            return;
        }
    };
    let own_address = creds
        .funder_address
        .as_deref()
        .unwrap_or(&creds.eoa_address);
    // Bounded retry: this is the only recovery of downtime fills (the position reconciler detects drift but does not re-inject fills),
    // and a transient CLOB error should not silence it. The retry is idempotent: dedup by exec_id in fill_sink absorbs repeats.
    let mut trades = None;
    for attempt in 1..=3u32 {
        match clob_sdk::fetch_trades_since(bundle, last_ts_ms / 1000, own_address).await {
            Ok(t) => {
                trades = Some(t);
                break;
            }
            Err(e) => {
                tracing::warn!(error = %e, attempt,
                    "startup: trade backfill failed, retrying");
                tokio::time::sleep(std::time::Duration::from_secs(2 * attempt as u64)).await;
            }
        }
    }
    let Some(trades) = trades else {
        tracing::warn!(
            "startup: trade backfill failed after 3 attempts, downtime fills may be missing"
        );
        return;
    };
    if trades.is_empty() {
        tracing::info!("startup: backfill, no new trades since the last fill");
        return;
    }
    tracing::info!(
        n = trades.len(),
        "startup: backfill, re-injecting CLOB trades (deduplicated in fill_sink)"
    );
    for t in trades {
        let strategy = match &t.own_order_id {
            Some(oid) => state
                .get_gtd_strategy(oid)
                .await
                .ok()
                .flatten()
                .unwrap_or(StrategyId::Arb),
            None => StrategyId::Arb,
        };
        let fill = Fill {
            market_cid: t.market_cid,
            token_id: t.token_id,
            side: if t.is_buy {
                poly_bot::types::FillSide::Buy
            } else {
                poly_bot::types::FillSide::Sell
            },
            price: t.price,
            size: t.size,
            // `trade_id` is the execution (exec_id, the dedup key); `own_order_id` is the order (attribution via order_index).
            order_id: t.own_order_id.clone(),
            // `exec_id` already carries the :order_id suffix on maker legs (one aggregated trade can match several of our orders).
            exec_id: Some(t.exec_id),
            ts: chrono::DateTime::from_timestamp(t.match_ts, 0).unwrap_or_else(chrono::Utc::now),
            mode: Mode::Live,
            // neg_risk unknown for markets already out of the registry: false is safe, it only affects the manual redeem path, not the accounting.
            neg_risk: false,
            strategy,
            // clear_at = match_ts: the bracket has already expired or will expire soon, so the orphan checker picks it up in the next cycle and resolves the winner.
            clear_at: t.match_ts,
            outcome_label: None,
            hedge_id: None,
            local_order_id: None,
        };
        if let Err(e) = fill_tx.send(fill) {
            tracing::error!(error = %e, "startup: fill_sink closed during backfill");
            return;
        }
    }
}

/// Reconciles the GTD orders (active) on the CLOB against the DB at startup in live mode.
///
/// - GTD with an open position in DB → re-register in `executor.active_gtd`.
/// - GTD without a matching position → stale (earlier session without the opposite leg's fill) → cancel.
///
/// Returns `false` if it could not determine which orders are open.
/// The caller decides what to do (in live mode, abort the startup).
async fn reconcile_open_orders(
    bundle: &poly_bot::clob_sdk::AuthedClobClient,
    state: &StateStore,
    executor: &mut Executor,
) -> bool {
    let orders = match clob_sdk::fetch_open_gtd_orders(bundle).await {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = %e, "startup: could not fetch open orders from the CLOB");
            return false;
        }
    };

    if orders.is_empty() {
        tracing::info!("startup: no GTD orders on the CLOB");
        return true;
    }

    tracing::info!(
        count = orders.len(),
        "startup: reconciling GTD orders from the CLOB"
    );

    let (mut restored, mut canceled) = (0usize, 0usize);
    for order in orders {
        let has_pos = state
            .has_open_position_in_market(&order.market_cid)
            .await
            .unwrap_or(false);
        if has_pos {
            // L01-1/B10: the OWNER is resolved BEFORE the claim, with COMPLETE evidence from the journal: `order_index` with cid/token
            // consistent with the venue's order, and `gtd_orders` (legacy owner) with a consistent cid. A row present with invalid data
            // or in conflict with the other table is indeterminate, not absent. An absent owner fails closed (registering it under Arb
            // gave a SELL a claim of 0). A query error aborts the startup (same rule as a failed fetch).
            let owner = match startup_gtd_owner(state, &order).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(
                        error = %e, order_id = %order.order_id,
                        "startup: owner indeterminate or contradictory, aborting"
                    );
                    return false;
                }
            };
            let Some((cost, strategy)) = startup_gtd_claim(&order, owner) else {
                tracing::error!(
                    order_id = %order.order_id,
                    market   = %order.market_cid,
                    token    = %order.token_id,
                    "startup: GTD order with a position but no recoverable owner/side, \
                     indeterminate reconciliation, aborting startup"
                );
                return false;
            };
            executor
                .restore_gtd(
                    &order.market_cid,
                    &order.token_id,
                    order.order_id.clone(),
                    cost,
                    order.clear_at,
                    strategy,
                )
                .await;
            restored += 1;
            tracing::info!(
                order_id  = %order.order_id,
                market    = %order.market_cid,
                token     = %order.token_id,
                remaining = %order.remaining,
                price     = %order.price,
                strategy  = %strategy,
                "startup: GTD re-registered (associated open position)"
            );
        } else {
            match clob_sdk::cancel_order(bundle, &order.order_id).await {
                Ok(_) => canceled += 1,
                Err(e) => {
                    tracing::warn!(error = %e, order_id = %order.order_id, "startup: error cancelling stale GTD")
                }
            }
            tracing::info!(
                order_id = %order.order_id,
                market   = %order.market_cid,
                "startup: stale GTD cancelled (no position in DB)"
            );
        }
    }
    tracing::info!(
        restored,
        canceled,
        "startup: order reconciliation complete"
    );
    true
}

/// Claim decision for a live GTD order on the CLOB that has an associated open position in the DB. It is the SAME contractual
/// rule used by the periodic refresh and the Unknown reconciler (`resting_claim_usdc` in `order::router`), except that the owner
/// here comes from the journal (`order_index`/`gtd_orders`) rather than from `active_gtd`.
///
/// Returns `Some((claim, owner))` only when BOTH values are certain:
///
/// - `owner` `None` → `None`. The venue does not declare a strategy; it used to be fabricated as `Arb`, which gave a SELL of
///   inventory a claim of 0 and dropped the order from the exposure radar. Fail-closed: aborting the startup is better than losing money.
/// - `side` `Unknown` from the wire or invalid data → `resting_claim_usdc` returns `None` → `None`. No claim is invented.
fn startup_gtd_claim(
    order: &clob_sdk::GtdOrder,
    owner: Option<StrategyId>,
) -> Option<(rust_decimal::Decimal, StrategyId)> {
    let strategy = owner?;
    let claim =
        resting_claim_usdc((&order.side).into(), order.remaining, order.price, strategy)?;
    Some((claim, strategy))
}

/// Durable owner of a live GTD order on the CLOB, backed by COMPLETE evidence. Knowing the strategy is not enough: the row
/// must refer to the SAME order (`cid`/`token_id` consistent with the venue's data).
///
/// Chain: `order_index` (modern attribution) → `gtd_orders` (legacy owner, leftovers from earlier restorations). Rules:
///
/// - A row present with an unrecognised strategy, a `cid`/`token` that does not match the venue's order, or a conflict between
///   `order_index` and `gtd_orders` → `Err`: indeterminate, NOT absent. The startup aborts instead of guessing the owner.
/// - No row → `Ok(None)` (no owner: the caller fails closed, never defaults to `Arb`).
/// - Query error → propagates (the caller aborts the startup).
async fn startup_gtd_owner(
    state: &StateStore,
    order: &clob_sdk::GtdOrder,
) -> anyhow::Result<Option<StrategyId>> {
    let idx = state.order_index_row(&order.order_id).await?;
    if let Some(ev) = idx {
        let strategy = StrategyId::from_str(&ev.strategy).map_err(|_| {
            anyhow::anyhow!("order_index with unrecognised strategy: {}", ev.strategy)
        })?;
        if ev.cid != order.market_cid || ev.token_id != order.token_id {
            return Err(anyhow::anyhow!(
                "order_index contradicts the venue's order (different cid/token)"
            ));
        }
        // If there is also a legacy row in gtd_orders it must be CONSISTENT: a conflict between tables does not turn into certainty.
        if let Some(g) = state.gtd_order_row(&order.order_id).await? {
            if g.market_cid != order.market_cid {
                return Err(anyhow::anyhow!(
                    "gtd_orders contradicts the venue's order (different cid)"
                ));
            }
            let gs = StrategyId::from_str(&g.strategy).map_err(|_| {
                anyhow::anyhow!("gtd_orders with unrecognised strategy: {}", g.strategy)
            })?;
            if gs != strategy {
                return Err(anyhow::anyhow!(
                    "order_index and gtd_orders disagree on the owner"
                ));
            }
        }
        return Ok(Some(strategy));
    }
    if let Some(g) = state.gtd_order_row(&order.order_id).await? {
        if g.market_cid != order.market_cid {
            return Err(anyhow::anyhow!(
                "gtd_orders contradicts the venue's order (different cid)"
            ));
        }
        let strategy = StrategyId::from_str(&g.strategy).map_err(|_| {
            anyhow::anyhow!("gtd_orders with unrecognised strategy: {}", g.strategy)
        })?;
        return Ok(Some(strategy));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    //! Startup claim rule: `startup_gtd_claim` is the pure piece; `reconcile_open_orders` feeds it the owner from the journal.
    //! Covered cases: BUY, SELL-PM and SELL-TC with an owner, a missing owner (no fallback to Arb), and an indeterminate side.
    use super::*;
    use polymarket_client_sdk_v2::clob::types::Side;
    use rust_decimal_macros::dec;

    fn gtd(
        side: Side,
        price: rust_decimal::Decimal,
        remaining: rust_decimal::Decimal,
    ) -> clob_sdk::GtdOrder {
        clob_sdk::GtdOrder {
            order_id: "oid-1".into(),
            market_cid: "0xabc".into(),
            token_id: "tok-yes".into(),
            side,
            price,
            remaining,
            clear_at: 1_700_000_000,
        }
    }

    #[test]
    fn startup_claim_buy_with_owner_is_remainder_by_price() {
        let o = gtd(Side::Buy, dec!(0.5), dec!(3));
        assert_eq!(
            startup_gtd_claim(&o, Some(StrategyId::ParityMaker)),
            Some((dec!(1.5), StrategyId::ParityMaker))
        );
    }

    #[test]
    fn startup_claim_sell_pm_is_zero_with_pm_owner() {
        let o = gtd(Side::Sell, dec!(0.7), dec!(2));
        assert_eq!(
            startup_gtd_claim(&o, Some(StrategyId::ParityMaker)),
            Some((dec!(0), StrategyId::ParityMaker))
        );
    }

    #[test]
    fn startup_claim_sell_tc_is_downside_remainder() {
        let o = gtd(Side::Sell, dec!(0.4), dec!(4));
        assert_eq!(
            startup_gtd_claim(&o, Some(StrategyId::ThetaCapture)),
            Some((dec!(4), StrategyId::ThetaCapture))
        );
        // Partial remainder: the downside is re-evaluated with what is left.
        let o = gtd(Side::Sell, dec!(0.4), dec!(1.5));
        assert_eq!(
            startup_gtd_claim(&o, Some(StrategyId::ThetaCapture)),
            Some((dec!(1.5), StrategyId::ThetaCapture))
        );
    }

    #[test]
    fn startup_claim_without_owner_does_not_fabricate_arb() {
        // Even a harmless BUY is never attributed without an owner:
        // `None` = fail-closed (the startup aborts).
        let buy = gtd(Side::Buy, dec!(0.5), dec!(3));
        assert_eq!(startup_gtd_claim(&buy, None), None);
        let sell = gtd(Side::Sell, dec!(0.7), dec!(2));
        assert_eq!(startup_gtd_claim(&sell, None), None);
    }

    #[test]
    fn startup_claim_unknown_side_is_indeterminate() {
        // `Side::Unknown` from the wire is not turned into a BUY: without a side there is
        // no computable claim → None.
        let o = gtd(Side::Unknown, dec!(0.5), dec!(3));
        assert_eq!(startup_gtd_claim(&o, Some(StrategyId::ParityMaker)), None);
        let o = gtd(Side::Unknown, dec!(0.5), dec!(3));
        assert_eq!(startup_gtd_claim(&o, None), None);
    }

    #[test]
    fn startup_claim_invalid_price_does_not_reduce() {
        // A price of 0 or >1 cannot fabricate a spurious reduction of the claim.
        let o = gtd(Side::Buy, dec!(0), dec!(3));
        assert_eq!(startup_gtd_claim(&o, Some(StrategyId::ParityMaker)), None);
        let o = gtd(Side::Buy, dec!(1.5), dec!(3));
        assert_eq!(startup_gtd_claim(&o, Some(StrategyId::ParityMaker)), None);
    }

    #[test]
    fn open_order_side_maps_sdk_side() {
        use poly_bot::order::router::OpenOrderSide as OOS;
        assert_eq!(OOS::from(&Side::Buy), OOS::Buy);
        assert_eq!(OOS::from(&Side::Sell), OOS::Sell);
        assert_eq!(OOS::from(&Side::Unknown), OOS::Unknown);
    }
}
