use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};

use dashmap::DashSet;

use crate::config::Config;
use crate::types::{BookSnapshot, DeadLetterFill, DerivedCreds, Fill};

mod discover;
mod events;
// `pub(crate)`: the `research_replay` rebuilder deserializes the
// recorder payloads with the same wire types (`WsMessage`,
// `RawLevel`, `RawPriceChange`) and reuses `parse_levels` — a single
// implementation of venue parsing, no duplicated format.
pub(crate) mod handle;
pub mod poll;
mod prefetch;
pub mod reference;
pub mod registry;
mod resync;
mod user_ws;
mod ws;

pub use events::TokenEvent;
pub use registry::MarketRegistry;
pub use user_ws::{handle_trade, UserTrade};

#[allow(clippy::too_many_arguments)]
pub async fn run(
    book_tx: broadcast::Sender<Arc<BookSnapshot>>,
    config: Arc<Config>,
    shared_registry: Arc<registry::MarketRegistry>,
    state: Arc<crate::state::StateStore>,
    creds: Option<Arc<DerivedCreds>>,
    fill_tx: mpsc::UnboundedSender<Fill>,
    known_order_ids: Arc<DashSet<String>>,
    // Authenticated CLOB bundle, `Some` only in live (travels
    // coupled to `creds`) — enables the reconnection gap backfill in
    // `user_ws::run`.
    bundle: Option<Arc<crate::clob_sdk::AuthedClobClient>>,
    // Orphan LIVE fills → dead-letter in DB.
    dead_letter_tx: Option<mpsc::UnboundedSender<DeadLetterFill>>,
    // WS dynamic add/remove channel — `broadcast` (not `mpsc`) because it
    // has several producers (discover, prefetch, StrategySupervisor) and the
    // consumer resubscribes on every iteration of the loop: with mpsc the receiver
    // would be lost along with the task aborted on a feed restart.
    // The registry is authoritative at manager boot (`ws_token_ids`), so
    // events emitted without live receivers don't lose real state.
    sub_tx: broadcast::Sender<TokenEvent>,
    // Research recorder producer — feeds the WS dispatcher's edge hook
    // and the REST sweeper. Always exists: with the hub inactive
    // `record` discards at no cost.
    recorder: crate::research::events::EventProducer,
) {
    let registry = shared_registry;
    let mut backoff_secs = 2u64;

    // Seed of `ws_universes`: universes declared by `enabled=true`
    // strategies at boot — those tokens go to the WS; the rest of the scan-only
    // universes are served by `poll.rs` via batched REST. The venue
    // limits by total subscribed tokens, so the ~1,200
    // tokens with no real-time consumer are taken out of the stream.
    for u in config.ws_universe_names() {
        registry.ws_universes.insert(u);
    }

    // Restart loop. If any subtask panics or exits unexpectedly,
    // all the others are aborted and the whole feed is restarted with backoff.
    // Without this loop, a crashed subtask left the bot running with no book feed
    // (blind: strategies stopped receiving snapshots with no visible error).
    loop {
        tracing::info!("market_feed: starting");

        let http = match reqwest::Client::builder()
            .user_agent(crate::POLY_USER_AGENT)
            .timeout(std::time::Duration::from_secs(10))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "market_feed: failed to build HTTP client — retrying in {backoff_secs}s"
                );
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(60);
                continue;
            }
        };

        // Initial synchronous discovery so the WS supervisor has tokens on first connect
        discover::initial_discover(&http, &config, &registry, &state, &recorder).await;

        let registry_ws = registry.clone();
        let registry_disc = registry.clone();
        let config_ws = config.clone();
        let config_disc = config.clone();

        let session_start = std::time::Instant::now();
        let mut tasks = tokio::task::JoinSet::new();
        tasks.spawn(ws::run_supervisor(
            sub_tx.subscribe(),
            registry_ws,
            book_tx.clone(),
            config_ws,
            recorder.clone(),
        ));
        // REST sweeper for stragglers (resync.rs) — only if enabled.
        // There is no per-token resync queue anymore: the registry's `needs_snapshot`
        // set IS the queue; the sweeper sweeps in batches (POST /books).
        if config.market_feed.resync_on_reconnect {
            // Variant with recorder — the sweeper's REST books enter
            // the dataset tagged `RESYNC` (without the hook they were left out).
            tasks.spawn(resync::run_resync_task_with_recorder(
                registry.clone(),
                book_tx.clone(),
                http.clone(),
                resync::ResyncCfg {
                    sweep_interval: std::time::Duration::from_millis(
                        config.market_feed.resync_sweep_ms,
                    ),
                    grace: std::time::Duration::from_millis(config.market_feed.resync_grace_ms),
                    batch_size: config.market_feed.resync_batch_size,
                    min_request_interval: std::time::Duration::from_millis(
                        config.market_feed.resync_min_interval_ms,
                    ),
                    cooldown_base: std::time::Duration::from_millis(
                        config.market_feed.resync_cooldown_base_ms,
                    ),
                    cooldown_max: std::time::Duration::from_millis(
                        config.market_feed.resync_cooldown_max_ms,
                    ),
                },
                recorder.clone(),
            ));
        }
        tasks.spawn(discover::run_discover_task(
            http.clone(),
            config_disc,
            registry_disc,
            sub_tx.clone(),
            state.clone(),
            recorder.clone(),
        ));
        // Batched REST poll of scan-only universes (see `poll.rs`) — takes
        // out of the WS the tokens no enabled strategy consumes.
        if config.market_feed.scan_poll_interval_secs > 0 {
            tasks.spawn(poll::run_scan_poll_task(
                registry.clone(),
                book_tx.clone(),
                http.clone(),
                config.market_feed.scan_poll_interval_secs,
                config.market_feed.scan_poll_batch_size,
            ));
        }
        // Predictive prefetch: the next bracket of each `*-updown-*` series
        // (deterministic slug `{asset}-updown-{dur}-{epoch}`) is queried on
        // Gamma by direct slug — the brackets exist before opening, so
        // they are discovered and subscribed in advance (window ~1-3s after
        // appearing vs up to `market_poll_interval_secs` of the general poll).
        if config.market_feed.prefetch_interval_secs > 0 {
            tasks.spawn(prefetch::run_prefetch_task(
                http.clone(),
                config.clone(),
                registry.clone(),
                sub_tx.clone(),
                state.clone(),
                recorder.clone(),
            ));
        }
        if let Some(c) = creds.clone() {
            tasks.spawn(user_ws::run(
                c,
                registry.clone(),
                fill_tx.clone(),
                known_order_ids.clone(),
                bundle.clone(),
                Some(state.clone()),
                dead_letter_tx.clone(),
            ));
        }

        // Wait for any subtask to exit (cleanly or by panic).
        if let Some(res) = tasks.join_next().await {
            match res {
                Ok(()) => tracing::warn!("market_feed: subtask exited unexpectedly"),
                Err(e) => tracing::error!(error = %e, "market_feed: subtask panicked"),
            }
            // Abort the remaining ones to clean up and prepare the restart.
            tasks.abort_all();
        }

        // Increment first, then reset if the session was stable ≥60 s.
        // That way, after a stable restart, the next backoff also starts at 2 s
        // (not at 4 s as happened when incrementing after the reset).
        backoff_secs = (backoff_secs * 2).min(60);
        if session_start.elapsed().as_secs() >= 60 {
            backoff_secs = 2;
        }
        tracing::warn!("market_feed: restarting all subtasks in {backoff_secs}s");
        ::metrics::counter!("hrf_market_feed_restart_total").increment(1);
        let jitter_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_millis() % (backoff_secs as u32 * 200 + 1))
            .unwrap_or(0) as u64;
        tokio::time::sleep(
            std::time::Duration::from_secs(backoff_secs)
                + std::time::Duration::from_millis(jitter_ms),
        )
        .await;
    }
}
