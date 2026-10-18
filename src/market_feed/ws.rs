use crate::time::Instant;
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use super::events::TokenEvent;
use super::handle;
use super::registry::MarketRegistry;
use crate::config::Config;
use crate::research::events::{EventProducer, SourceContext};
use crate::types::BookSnapshot;

// metrics crate re-exported via the top-level `metrics` dep (same as paper.rs usage)
use metrics;

const MARKET_WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const MAX_BACKOFF_SECS: u64 = 30;

/// Capacity of the reader→dispatcher queue. With ~2,800 tokens
/// price_change bursts exceed the dispatch rate — the buffer absorbs
/// the burst in memory instead of letting the server's TCP buffer grow
/// (which answered with code=1013 "slow consumer: send buffer full").
const DISPATCH_QUEUE_CAP: usize = 8_192;

/// Aborts the dispatcher task on any exit from `ws_loop` (return, break
/// or `?`): frames queued from a dead session must not be dispatched
/// after `mark_all_needs_snapshot` — they would be pre-cut data
/// stamped as freshly arrived.
struct AbortOnDrop(tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Maximum tokens per WS connection. A single connection with ~2,800
/// tokens received the initial book dump and the price_change bursts
/// over one socket — the SERVER's send buffer filled up and it closed
/// (code=1013 "slow consumer: send buffer full", ~840 reconnects/6h; and at
/// ~384/connection it still kept dying). There is no documented limit, but the ecosystem
/// (nautilus_trader `WS_DEFAULT_SUBSCRIPTIONS`, community collectors)
/// converges on ~200 as an empirical bound: above it the connection "stalls".
/// With K = ceil(tokens/200) connections, the dump and traffic of each server
/// buffer stay under the bound; also, a shard failure only marks
/// ITS tokens stale and the rest of the feed keeps flowing.
/// Live measurement: the AGGREGATE kill rate is ~constant (~0.7-0.9/min)
/// at ~384, ~193 and ~100 tokens/conn alike — venue throttling
/// is proportional to the TOTAL of subscribed tokens, not to the per-connection burst.
/// Kills concentrate (77%) in minutes 0-2 after each 5 min
/// boundary (the updown-5m cohort expiring → synchronized storm). 100 is chosen
/// for blast radius: each kill stales ~100 tokens for ~2-4 s, not ~200.
const MAX_TOKENS_PER_SHARD: usize = 100;

type WsWrite = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;

/// Minimum connection uptime (seconds) before treating a disconnection as a
/// "clean session" and resetting the reconnection backoff.  A reset that happens
/// after e.g. 5 minutes of healthy data should not keep the backoff at 30 s.
const HEALTHY_SESSION_SECS: u64 = 60;

/// Splits the tokens into `k` round-robin lists. A token lives in exactly
/// one shard → per-token order is preserved even though each shard has its
/// own dispatcher.
fn shard_tokens(token_ids: Vec<String>, k: usize) -> Vec<Vec<String>> {
    let mut shards: Vec<Vec<String>> = (0..k).map(|_| Vec::new()).collect();
    for (i, t) in token_ids.into_iter().enumerate() {
        shards[i % k].push(t);
    }
    shards.retain(|s| !s.is_empty());
    shards
}

/// Dynamic command to an already connected shard — add/remove without reconnecting.
/// Verified live (`ws_probe` probe): `{"operation":"subscribe"}` delivers
/// the token's initial dump in ~1 RTT and `{"operation":"unsubscribe"}` cuts
/// its events instantly — an earlier belief that "the server ignores later
/// subscriptions" was an artifact of using `type:"subscribe"` instead of the
/// documented `operation:*` format.
enum ShardCmd {
    Add(Vec<String>),
    Remove(Vec<String>),
}

impl ShardCmd {
    fn is_empty(&self) -> bool {
        match self {
            ShardCmd::Add(ts) | ShardCmd::Remove(ts) => ts.is_empty(),
        }
    }
}

/// Filters a pending command against the shard's desired set (`tokens` of the
/// bookkeeping). An `Add` keeps only tokens still desired; a `Remove` only
/// tokens that are NO longer desired — so a command queued because the queue was full does not
/// execute a decision that a later event already reverted.
fn retain_desired(desired: &std::collections::HashSet<String>, cmd: &mut ShardCmd) {
    match cmd {
        ShardCmd::Add(ts) => ts.retain(|t| desired.contains(t)),
        ShardCmd::Remove(ts) => ts.retain(|t| !desired.contains(t)),
    }
}

/// Manager's view of a live shard: its command channel and its set of
/// tokens (the manager's copy is authoritative for the distribution; the shard
/// keeps it in sync by applying each ShardCmd).
struct ShardCtl {
    cmd_tx: mpsc::Sender<ShardCmd>,
    tokens: std::collections::HashSet<String>,
    /// Handle of the shard's live task — lets us locate it when the
    /// JoinSet returns a `JoinError` (panic): the error carries no index,
    /// but its handle is the only `is_finished` among the live shards.
    abort_handle: tokio::task::AbortHandle,
}

/// Shard manager with DYNAMIC subscription: each new token is added
/// with `operation:subscribe` on an existing shard (~1 RTT, no reconnect) and
/// each evicted token is removed with `operation:unsubscribe`. Global epoch
/// restarts are gone — previously each discovery killed the K
/// connections and staled ALL the tokens.
pub async fn run_supervisor(
    mut sub_rx: broadcast::Receiver<TokenEvent>,
    registry: Arc<MarketRegistry>,
    book_tx: broadcast::Sender<Arc<BookSnapshot>>,
    config: Arc<Config>,
    // Research recorder producer. Always exists:
    // with the hub inactive `record` discards at no cost.
    recorder: EventProducer,
) {
    let mut shards: Vec<ShardCtl> = Vec::new();
    let mut tasks: tokio::task::JoinSet<usize> = tokio::task::JoinSet::new();
    // token_id → index of the shard that has it subscribed.
    let mut token_shard: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    // Commands that did not fit in their shard's queue (queue full) — they are
    // retried at the start of each iteration. Without this the bookkeeping would say
    // "subscribed" for a token whose `operation:subscribe` was never sent.
    let mut pending_cmds: std::collections::VecDeque<(usize, ShardCmd)> =
        std::collections::VecDeque::new();

    // Boot: only WS-bound tokens (universes declared by enabled
    // strategies). Scan-only ones are served by `poll.rs` via REST.
    let initial = registry.ws_token_ids();
    if !initial.is_empty() {
        let n = initial.len().div_ceil(MAX_TOKENS_PER_SHARD);
        for shard in shard_tokens(initial, n) {
            let idx = spawn_shard(
                &mut shards,
                &mut tasks,
                shard,
                &registry,
                &book_tx,
                &config,
                &recorder,
            );
            for t in &shards[idx].tokens {
                token_shard.insert(t.clone(), idx);
            }
        }
        tracing::info!(
            tokens = token_shard.len(),
            shards = shards.len(),
            "market_feed: shard manager started (dynamic subscription)"
        );
    } else {
        tracing::warn!("market_feed: no WS tokens at startup — waiting for discovery");
    }

    loop {
        // Drain pending commands from previous iterations (full queues).
        let mut n = pending_cmds.len();
        while n > 0 {
            let (idx, mut cmd) = pending_cmds.pop_front().unwrap();
            n -= 1;
            // Reordering: between enqueueing and now the opposite event
            // for the same token may have arrived (pending Add + later
            // Unsubscribe, or vice versa). The `shards[idx].tokens` bookkeeping is
            // the DESIRED state — filter against it so as not to execute an
            // obsolete command (subscribing a dead token or unsubscribing one
            // that was re-added afterwards).
            retain_desired(&shards[idx].tokens, &mut cmd);
            if cmd.is_empty() {
                continue;
            }
            match shards[idx].cmd_tx.try_send(cmd) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(cmd)) => {
                    pending_cmds.push_back((idx, cmd));
                }
                Err(mpsc::error::TrySendError::Closed(cmd)) => {
                    // Dead shard — its respawn keeps `shards[idx].tokens`,
                    // which already includes/reflects this command: no need to
                    // resend it (the initial subscription covers the set).
                    let _ = cmd;
                }
            }
        }

        tokio::select! {
            // Retry pending commands even if no events arrive —
            // without this arm a full queue would leave tokens unsubscribed
            // until the next discovery/toggle.
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)),
                if !pending_cmds.is_empty() => {}

            event = sub_rx.recv() => {
                let event = match event {
                    Ok(ev) => ev,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        // Broadcast lost events — the registry remains
                        // authoritative: reconcile against `ws_token_ids` instead
                        // of trusting the event stream.
                        tracing::warn!(missed = n, "market_feed: TokenEvent lag — reconciling against registry");
                        reconcile(&registry, &mut shards, &mut tasks, &mut token_shard, &mut pending_cmds, &book_tx, &config, &recorder);
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::error!("market_feed: TokenEvent channel closed — manager dead");
                        return;
                    }
                };
                match event {
                    TokenEvent::Subscribe { token_ids } => {
                        // Dynamic add: distribute to the least loaded shard
                        // that still has room; overflow → new shard.
                        let mut per_shard: std::collections::HashMap<usize, Vec<String>> =
                            std::collections::HashMap::new();
                        let mut added = 0usize;
                        for t in token_ids {
                            if token_shard.contains_key(&t) {
                                continue; // already subscribed
                            }
                            let idx = match shards
                                .iter()
                                .enumerate()
                                .filter(|(_, s)| s.tokens.len() < MAX_TOKENS_PER_SHARD)
                                .min_by_key(|(_, s)| s.tokens.len())
                                .map(|(i, _)| i)
                            {
                                Some(i) => i,
                                None => spawn_shard(
                                    &mut shards,
                                    &mut tasks,
                                    Vec::new(),
                                    &registry,
                                    &book_tx,
                                    &config,
                                    &recorder,
                                ),
                            };
                            shards[idx].tokens.insert(t.clone());
                            token_shard.insert(t.clone(), idx);
                            per_shard.entry(idx).or_default().push(t);
                            added += 1;
                        }
                        for (idx, batch) in per_shard {
                            if let Err(e) = shards[idx].cmd_tx.try_send(ShardCmd::Add(batch)) {
                                match e {
                                    mpsc::error::TrySendError::Full(cmd) => {
                                        tracing::warn!(shard = idx, "market_feed: command queue full — pending retry");
                                        pending_cmds.push_back((idx, cmd));
                                    }
                                    mpsc::error::TrySendError::Closed(_) => {
                                        // Respawn covers the set via `shards[idx].tokens`.
                                    }
                                }
                            }
                        }
                        if added > 0 {
                            metrics::counter!("hrf_market_feed_dyn_subscribe_total").increment(added as u64);
                            tracing::info!(new_tokens = added, "market_feed: tokens subscribed dynamically");
                        }
                    }
                    TokenEvent::Unsubscribe { token_ids } => {
                        let mut per_shard: std::collections::HashMap<usize, Vec<String>> =
                            std::collections::HashMap::new();
                        let mut removed = 0usize;
                        for t in token_ids {
                            let Some(idx) = token_shard.remove(&t) else {
                                continue; // was never subscribed (poll-only)
                            };
                            shards[idx].tokens.remove(&t);
                            per_shard.entry(idx).or_default().push(t);
                            removed += 1;
                        }
                        for (idx, batch) in per_shard {
                            if let Err(e) = shards[idx].cmd_tx.try_send(ShardCmd::Remove(batch)) {
                                match e {
                                    mpsc::error::TrySendError::Full(cmd) => {
                                        pending_cmds.push_back((idx, cmd));
                                    }
                                    mpsc::error::TrySendError::Closed(_) => {}
                                }
                            }
                        }
                        if removed > 0 {
                            metrics::counter!("hrf_market_feed_dyn_unsubscribe_total").increment(removed as u64);
                            tracing::info!(tokens = removed, "market_feed: tokens unsubscribed dynamically");
                        }
                    }
                }
            }
            res = tasks.join_next() => {
                // A shard_supervisor only exits by panic or closed command
                // channel — it is respawned keeping its token set.
                match res {
                    Some(Ok(idx)) => {
                        // If the Err arm already respawned this index, the registered
                        // handle points to the NEW (live) task — respawning
                        // again would leave two tasks for the same shard.
                        if shards[idx].abort_handle.is_finished() {
                            respawn_shard(idx, &mut shards, &mut tasks, &registry, &book_tx, &config, &recorder);
                        }
                    }
                    Some(Err(e)) => {
                        // Task panic: the JoinError carries no index,
                        // but its AbortHandle is the only finished one among the
                        // live shards. Without respawn its tokens would stay stale
                        // forever — `token_shard` believes them served and
                        // `reconcile` doesn't touch them.
                        tracing::error!(error = %e, "market_feed: shard supervisor panicked — respawning");
                        if let Some(idx) = shards
                            .iter()
                            .position(|s| s.abort_handle.is_finished())
                        {
                            respawn_shard(idx, &mut shards, &mut tasks, &registry, &book_tx, &config, &recorder);
                        }
                    }
                    None => {
                        // No live shards: may be a startup with no tokens.
                        // Wait for the next event instead of spinning hot.
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    }
                }
            }
        }
    }
}

/// Spawns a new shard with its initial set (may be empty — then it doesn't
/// connect until it receives `ShardCmd::Add`). Returns its index in `shards`.
fn spawn_shard(
    shards: &mut Vec<ShardCtl>,
    tasks: &mut tokio::task::JoinSet<usize>,
    token_ids: Vec<String>,
    registry: &Arc<MarketRegistry>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    config: &Arc<Config>,
    recorder: &EventProducer,
) -> usize {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ShardCmd>(256);
    let idx = shards.len();
    let tokens: std::collections::HashSet<String> = token_ids.into_iter().collect();
    let reg = registry.clone();
    let tx = book_tx.clone();
    let cfg = config.clone();
    let rec = recorder.clone();
    let tokens_set = tokens.clone();
    let abort_handle = tasks.spawn(async move {
        shard_supervisor(tokens_set, cmd_rx, reg, tx, cfg, idx, rec).await;
        idx
    });
    shards.push(ShardCtl {
        cmd_tx,
        tokens,
        abort_handle,
    });
    idx
}

/// Respawns shard `idx` keeping its token set (the initial subscription
/// of the new connection covers the whole set — pending commands
/// toward the dead task are covered by this).
fn respawn_shard(
    idx: usize,
    shards: &mut [ShardCtl],
    tasks: &mut tokio::task::JoinSet<usize>,
    registry: &Arc<MarketRegistry>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    config: &Arc<Config>,
    recorder: &EventProducer,
) {
    let tokens: std::collections::HashSet<String> = shards[idx].tokens.iter().cloned().collect();
    tracing::warn!(
        shard = idx,
        tokens = tokens.len(),
        "market_feed: shard exited — respawning with its set"
    );
    let (cmd_tx, cmd_rx) = mpsc::channel::<ShardCmd>(256);
    let reg = registry.clone();
    let tx = book_tx.clone();
    let cfg = config.clone();
    let rec = recorder.clone();
    let tokens_set = tokens.clone();
    let abort_handle = tasks.spawn(async move {
        shard_supervisor(tokens_set, cmd_rx, reg, tx, cfg, idx, rec).await;
        idx
    });
    shards[idx].cmd_tx = cmd_tx;
    shards[idx].abort_handle = abort_handle;
}

/// Reconciliation after a broadcast `Lagged`: the registry is the source of
/// truth — subscribes the WS-bound tokens that are missing and unsubscribes those that
/// no longer are (evicted or migrated to poll). Same mechanics as the loop's
/// Subscribe/Unsubscribe arms, applied to a full diff.
fn reconcile(
    registry: &Arc<MarketRegistry>,
    shards: &mut Vec<ShardCtl>,
    tasks: &mut tokio::task::JoinSet<usize>,
    token_shard: &mut std::collections::HashMap<String, usize>,
    pending_cmds: &mut std::collections::VecDeque<(usize, ShardCmd)>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    config: &Arc<Config>,
    recorder: &EventProducer,
) {
    let want: std::collections::HashSet<String> = registry.ws_token_ids().into_iter().collect();
    // Removals: subscribed tokens that are no longer WS-bound.
    let mut per_shard_rm: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    token_shard.retain(|t, idx| {
        if want.contains(t) {
            true
        } else {
            shards[*idx].tokens.remove(t);
            per_shard_rm.entry(*idx).or_default().push(t.clone());
            false
        }
    });
    for (idx, batch) in per_shard_rm {
        if let Err(mpsc::error::TrySendError::Full(cmd)) =
            shards[idx].cmd_tx.try_send(ShardCmd::Remove(batch))
        {
            pending_cmds.push_back((idx, cmd));
        }
    }
    // Additions: WS-bound with no assigned shard.
    for t in want {
        if token_shard.contains_key(&t) {
            continue;
        }
        let idx = match shards
            .iter()
            .enumerate()
            .filter(|(_, s)| s.tokens.len() < MAX_TOKENS_PER_SHARD)
            .min_by_key(|(_, s)| s.tokens.len())
            .map(|(i, _)| i)
        {
            Some(i) => i,
            None => spawn_shard(
                shards,
                tasks,
                Vec::new(),
                registry,
                book_tx,
                config,
                recorder,
            ),
        };
        shards[idx].tokens.insert(t.clone());
        token_shard.insert(t.clone(), idx);
        if let Err(mpsc::error::TrySendError::Full(cmd)) =
            shards[idx].cmd_tx.try_send(ShardCmd::Add(vec![t]))
        {
            pending_cmds.push_back((idx, cmd));
        }
    }
}

/// Supervisor of ONE shard: keeps its connection alive, marks only its
/// tokens stale on each cut and applies its own backoff — one shard's failure doesn't
/// contaminate the others. With an empty set it doesn't connect: it waits for commands
/// (a shard with no tokens receives no messages and the staleness guard would kill it
/// in a loop).
async fn shard_supervisor(
    mut tokens: std::collections::HashSet<String>,
    mut cmd_rx: mpsc::Receiver<ShardCmd>,
    registry: Arc<MarketRegistry>,
    book_tx: broadcast::Sender<Arc<BookSnapshot>>,
    config: Arc<Config>,
    shard_idx: usize,
    recorder: EventProducer,
) {
    let mut backoff = 2u64;
    loop {
        // No tokens, no connection — wait for commands (Add) idle.
        while tokens.is_empty() {
            match cmd_rx.recv().await {
                Some(ShardCmd::Add(ts)) => tokens.extend(ts),
                Some(ShardCmd::Remove(ts)) => {
                    for t in ts {
                        tokens.remove(&t);
                    }
                }
                None => return,
            }
        }

        // Drain pending commands BEFORE connecting: the initial subscription
        // covers the whole set and redundant `operation:*` aren't needed.
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                ShardCmd::Add(ts) => tokens.extend(ts),
                ShardCmd::Remove(ts) => {
                    for t in ts {
                        tokens.remove(&t);
                    }
                }
            }
        }
        if tokens.is_empty() {
            continue;
        }

        let session_start = Instant::now();
        match ws_loop(
            &mut tokens,
            &mut cmd_rx,
            &registry,
            &book_tx,
            &config,
            shard_idx,
            recorder.clone(),
        )
        .await
        {
            Ok(()) => {
                backoff = 2;
                metrics::counter!("hrf_market_feed_ws_reconnects_total", "reason" => "clean")
                    .increment(1);
                tracing::warn!(shard = shard_idx, "market_feed: WS closed, reconnecting");
            }
            Err(e) => {
                // Long session → short backoff; sessions that die instantly
                // accumulate backoff (same rule as the global supervisor).
                let session_secs = session_start.elapsed().as_secs();
                if session_secs >= HEALTHY_SESSION_SECS {
                    backoff = 2;
                }
                metrics::counter!("hrf_market_feed_ws_reconnects_total", "reason" => "error")
                    .increment(1);
                tracing::warn!(shard = shard_idx, error = %e, backoff_secs = backoff, session_secs, "market_feed: WS error");
            }
        }

        // Only THIS shard's tokens go stale — the other shards keep
        // emitting normally (previously a cut staled all ~2,800).
        for t in &tokens {
            registry.needs_snapshot.insert(t.clone());
        }

        let jitter_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_millis() % (backoff as u32 * 200 + 1))
            .unwrap_or(0) as u64;
        tokio::time::sleep(
            tokio::time::Duration::from_secs(backoff)
                + tokio::time::Duration::from_millis(jitter_ms),
        )
        .await;
        backoff = (backoff * 2).min(MAX_BACKOFF_SECS);
    }
}

/// Builds the subscription JSON for a chunk of tokens. Extracted
/// from `send_subscribe_all` to be able to test the `custom_feature_enabled` flag
/// without opening a real WS. `tape=true` adds the flag that enables the trade tape
/// (`last_trade_price`) for those tokens.
fn build_subscribe_msg(chunk: &[String], tape: bool) -> serde_json::Value {
    let mut sub_msg = serde_json::json!({
        "type": "subscribe",
        "channel": "book",
        "assets_ids": chunk
    });
    if tape {
        sub_msg["custom_feature_enabled"] = serde_json::Value::Bool(true);
    }
    sub_msg
}

/// Polymarket's market WS only honors the FIRST `subscribe` message
/// of each connection — any later `subscribe` on the same
/// connection is silently ignored (verified live: disjoint
/// chunks A/B of 100 tokens, A-then-B yielded 186 messages from A and ZERO from B).
/// This function used to split into `chunk_size` (100 by default) and send one
/// `subscribe` per piece — with more than 100 tokens (the normal scenario), all
/// the pieces after the first were silently lost. Verified that a single
/// message with 20,615 tokens works without problem — there is no real size
/// ceiling that justifies splitting. Now it always sends everything in a single message.
async fn send_subscribe_all(
    write: &mut WsWrite,
    token_ids: &[String],
    // Enables the trade tape (last_trade_price) in the subscription.
    tape: bool,
) -> anyhow::Result<()> {
    let sub_msg = build_subscribe_msg(token_ids, tape);
    write.send(Message::Text(sub_msg.to_string())).await?;
    Ok(())
}

async fn ws_loop(
    tokens: &mut std::collections::HashSet<String>,
    cmd_rx: &mut mpsc::Receiver<ShardCmd>,
    registry: &Arc<MarketRegistry>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    config: &Config,
    shard_idx: usize,
    recorder: EventProducer,
) -> anyhow::Result<()> {
    let (ws_stream, _) = tokio::time::timeout(
        tokio::time::Duration::from_secs(10),
        connect_async(MARKET_WS_URL),
    )
    .await
    .map_err(|_| anyhow::anyhow!("market WS connect timed out"))?
    .map_err(|e| anyhow::anyhow!("market WS connect failed: {e}"))?;
    tracing::info!(shard = shard_idx, "market_feed: WS connected");
    let (mut write, mut read) = ws_stream.split();

    // 0 = tape disabled (the flag is not enabled and no flow is accumulated).
    // `custom_feature_enabled` also enables best_bid_ask/new_market/
    // market_resolved — events that handle.rs doesn't even parse and that multiply
    // per-connection traffic (they contributed to the 1013 "slow consumer"). The tape
    // is only consumed by parity_maker and g_rw1_maker (flow_net/flow_total): if
    // none is running, it isn't requested — useless traffic eliminated. A hot
    // enable takes effect on the shard's next reconnect.
    let tape_window_secs = config.market_feed.tape_window_secs;
    let tape = (tape_window_secs > 0
        && (config.strategies.parity_maker.limits.enabled
            || config.strategies.g_rw1_maker.limits.enabled))
        // With recording ACTIVE and `research_events.record_tape`, the
        // flag is requested even if no strategy consumes the tape — without it the
        // venue doesn't emit `last_trade_price` and the tape doesn't reach the dataset.
        // The producer always exists, so the gate is
        // `is_active()`, not `is_some()`. Evaluated once per WS session:
        // turning recording on live takes effect on the next
        // shard reconnect.
        //
        // Known limitations of the live toggle of
        // `ToolId::ResearchEvents` (enabling/disabling with the bot already
        // running, via API/Telegram — NOT startup, which does hook in
        // before any WS session exists, see `main.rs`):
        // 1. This condition is only evaluated when the shard (re)connects —
        //    enabling `research_events` mid-way through a live WS session does NOT
        //    add the flag until the next reconnect (above). Meanwhile
        //    the other sources (executor, oracle, discover, resync)
        //    already start recording immediately — only the WS trade
        //    TAPE lags one cycle behind.
        // 2. The market meta (`EventKind::MarketMeta`) is NOT re-emitted on
        //    live enable: `market_feed::discover::register_if_new`
        //    only emits it when a bracket enters the `MarketRegistry` for the
        //    first time in the process. Markets already discovered before
        //    `attach()` (with the hub inactive) don't pass through
        //    there again, so a live-enabled tape starts without its meta
        //    — replay can't reconstruct those specific markets until
        //    they close/are rediscovered or the bot restarts with the
        //    tool already enabled at boot.
        || (recorder.is_active() && config.research_events.record_tape);

    // Recorder context for this SESSION — a new generation per
    // (re)connection (the venue's seq is only compared within the same
    // epoch) and the stream identifies the shard.
    let ctx = SourceContext {
        producer: recorder,
        generation: crate::research::events::new_generation(),
        stream: format!("poly-ws:shard-{shard_idx}"),
    };
    let initial: Vec<String> = tokens.iter().cloned().collect();
    send_subscribe_all(&mut write, &initial, tape).await?;
    tracing::info!(
        shard = shard_idx,
        "market_feed: subscribed to {} tokens",
        initial.len()
    );

    // Decouple reading from processing. Previously each frame went through
    // handle::dispatch synchronously inside the read loop — with ~2,800
    // tokens price_change bursts exceeded the read rate, the
    // TCP buffer filled up and the server closed on us: code=1013
    // "slow consumer: send buffer full" (~840 reconnects/6h). Now the
    // reader only enqueues (arrival Instant + text) and a dispatcher task
    // consumes; if the queue saturates it's structural overload — we reconnect
    // rather than let the session die from backpressure.
    let (msg_tx, mut msg_rx) = mpsc::channel::<(Instant, String)>(DISPATCH_QUEUE_CAP);
    let registry_d = registry.clone();
    let book_tx_d = book_tx.clone();
    let ctx_d = ctx.clone();
    let mut dispatcher = AbortOnDrop(tokio::spawn(async move {
        while let Some((recv_instant, text)) = msg_rx.recv().await {
            // Liveness: epoch of the last WS message received — allows alerting
            // "feed quiet > Ns" without relying on per-event-kind counters.
            metrics::gauge!("hrf_market_feed_last_msg_unix_seconds")
                .set(crate::time::now().timestamp() as f64);
            handle::dispatch_with_recorder(
                &text,
                &registry_d,
                &book_tx_d,
                recv_instant,
                tape_window_secs,
                &ctx_d,
            );
        }
    }));

    let mut ping_int = tokio::time::interval(tokio::time::Duration::from_secs(
        config.market_feed.ws_ping_interval_secs,
    ));
    // Without this, the default `Burst` fires a burst of delayed
    // pings all at once if the scheduler lags (e.g. under load) — as
    // already fixed in user_ws.rs for its own keepalive.
    ping_int.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let stale_secs = config.market_feed.ws_stale_threshold_secs;
    let mut last_msg = Instant::now();

    loop {
        tokio::select! {
            biased;

            // Manager commands: dynamic add/remove without reconnecting
            // (`operation:subscribe`/`operation:unsubscribe` — verified live
            // with the `ws_probe` probe).
            cmd = cmd_rx.recv() => {
                let Some(cmd) = cmd else {
                    // Manager down — clean close so the shard gets respawned.
                    return Err(anyhow::anyhow!("market_feed: command channel closed"));
                };
                match cmd {
                    ShardCmd::Add(ts) => {
                        let fresh: Vec<String> =
                            ts.into_iter().filter(|t| !tokens.contains(t)).collect();
                        if !fresh.is_empty() {
                            let msg = serde_json::json!({
                                "operation": "subscribe",
                                "assets_ids": fresh,
                            });
                            write.send(Message::Text(msg.to_string())).await?;
                            for t in fresh {
                                tokens.insert(t);
                            }
                            metrics::counter!("hrf_market_feed_shard_dyn_sub_total")
                                .increment(1);
                        }
                    }
                    ShardCmd::Remove(ts) => {
                        let owned: Vec<String> =
                            ts.iter().filter(|t| tokens.contains(*t)).cloned().collect();
                        if !owned.is_empty() {
                            let msg = serde_json::json!({
                                "operation": "unsubscribe",
                                "assets_ids": owned,
                            });
                            write.send(Message::Text(msg.to_string())).await?;
                            for t in owned {
                                tokens.remove(&t);
                            }
                        }
                        if tokens.is_empty() {
                            // Empty shard — go back to waiting for commands without a connection.
                            return Ok(());
                        }
                    }
                }
            }

            msg_result = read.next() => {
                let Some(msg_result) = msg_result else {
                    // Distinguish "the server hung up without a close frame"
                    // from a normal Close — the churn (~2 drops/min) needed
                    // this visibility to be diagnosed.
                    tracing::warn!("market_feed: WS stream ended without close frame");
                    break;
                };
                let msg = msg_result?;
                let text: String = match msg {
                    // Reply to the textual keepalive — not JSON, don't dispatch.
                    Message::Text(t) if t == "PONG" => { last_msg = Instant::now(); continue; }
                    Message::Text(t) => { last_msg = Instant::now(); t.to_string() }
                    Message::Binary(b) => { last_msg = Instant::now(); String::from_utf8_lossy(&b).into_owned() }
                    Message::Close(frame) => {
                        let code = frame.as_ref().map(|f| f.code.into()).unwrap_or(1000u16);
                        let reason = frame.as_ref().map(|f| f.reason.to_string()).unwrap_or_default();
                        // Fatal codes (rate-limit, policy violation): propagate as error
                        // so the supervisor applies exponential backoff instead of resetting.
                        if matches!(code, 1008 | 4001 | 4003) {
                            return Err(anyhow::anyhow!("WS closed with fatal code {code}: {reason}"));
                        }
                        metrics::counter!("hrf_market_feed_ws_close_frames_total",
                            "code" => code.to_string()).increment(1);
                        tracing::warn!(shard = shard_idx, code, %reason, "market_feed: WS close frame received");
                        return Ok(());
                    }
                    Message::Ping(p) => {
                        // Docs: the server sends a protocol ping every 5s and
                        // closes if there is no pong in ~10s — explicit reply,
                        // same as user_ws (tungstenite also queues an
                        // automatic one; duplicating is harmless).
                        last_msg = Instant::now();
                        write.send(Message::Pong(p)).await?;
                        continue;
                    }
                    Message::Pong(_) | Message::Frame(_) => {
                        last_msg = Instant::now();
                        continue;
                    }
                };
                metrics::gauge!("hrf_market_feed_dispatch_queue_len")
                    .set((msg_tx.max_capacity() - msg_tx.capacity()) as f64);
                if msg_tx.try_send((Instant::now(), text)).is_err() {
                    // Saturated queue = structural overload, not a burst.
                    // Never drop deltas halfway (it would silently corrupt the book):
                    // clean reconnect + needs_snapshot, like any
                    // other cut — fail-closed.
                    metrics::counter!("hrf_market_feed_dispatch_queue_full_total").increment(1);
                    return Err(anyhow::anyhow!(
                        "market_feed: dispatch queue saturated ({DISPATCH_QUEUE_CAP} msgs) — reconnecting"
                    ));
                }
            }

            res = &mut dispatcher.0 => {
                // The dispatcher only ends if it panics or the channel closes
                // — both are a feed defect; supervised restart.
                return Err(anyhow::anyhow!(
                    "market_feed: dispatcher ended unexpectedly: {res:?}"
                ));
            }

            _ = ping_int.tick() => {
                write.send(Message::Ping(vec![])).await?;
                // Textual keepalive — the market channel docs
                // require the literal "PING" every 10s (at 30s the server
                // closed the session every ~10-60s; user_ws was already at 10s). An
                // intermediate proxy may absorb protocol pings without
                // the server's idle-tracker counting them.
                write.send(Message::Text("PING".into())).await?;
                if last_msg.elapsed().as_secs() > stale_secs {
                    tracing::warn!(
                        stale_secs,
                        "market_feed: WS stale, forcing reconnect"
                    );
                    break;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the tape enabled, the subscription JSON adds the flag that
    /// enables `last_trade_price` on Polymarket's WS.
    #[test]
    fn subscription_includes_custom_feature_enabled_with_active_tape() {
        let chunk = vec!["tok-1".to_string(), "tok-2".to_string()];

        let active_msg = build_subscribe_msg(&chunk, true);
        assert_eq!(
            active_msg["custom_feature_enabled"],
            serde_json::json!(true)
        );
        assert_eq!(active_msg["type"], serde_json::json!("subscribe"));
        assert_eq!(active_msg["channel"], serde_json::json!("book"));

        let inactive_msg = build_subscribe_msg(&chunk, false);
        assert!(
            inactive_msg.get("custom_feature_enabled").is_none(),
            "without tape the flag must not be added"
        );
    }

    /// Sharding is exhaustive, disjoint and balanced —
    /// each token lives in exactly one shard (per-token order invariant
    /// with independent dispatchers) and there are no leftover empty shards.
    #[test]
    fn shard_tokens_distributes_disjoint_and_balanced() {
        let tokens: Vec<String> = (0..10).map(|i| format!("tok-{i}")).collect();

        let shards = shard_tokens(tokens.clone(), 4);

        assert_eq!(shards.len(), 4);
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, 10);
        for s in &shards {
            assert!((2..=3).contains(&s.len()), "unbalanced shard: {s:?}");
        }
        // Disjoint: no token in two shards.
        let mut seen = std::collections::HashSet::new();
        for s in &shards {
            for t in s {
                assert!(seen.insert(t.clone()), "token duplicated across shards: {t}");
            }
        }
        // Fewer tokens than shards → no empty shards.
        let shards = shard_tokens(vec!["a".into(), "b".into()], 4);
        assert_eq!(shards.len(), 2);
    }

    /// Polymarket's WS only honors the FIRST `subscribe` of the
    /// connection — if splitting into several messages is ever reintroduced
    /// (`subscribe_chunk_size`, already removed), everything not in the
    /// first message would be silently lost. This test pins the invariant:
    /// a single `build_subscribe_msg` message must carry ALL the tokens
    /// given, even above the old default `chunk_size` (100).
    #[test]
    fn subscription_carries_all_tokens_in_a_single_message_without_chunking() {
        let tokens: Vec<String> = (0..250).map(|i| format!("tok-{i}")).collect();

        let msg = build_subscribe_msg(&tokens, false);

        let assets_ids = msg["assets_ids"]
            .as_array()
            .expect("assets_ids must be an array");
        assert_eq!(
            assets_ids.len(),
            250,
            "all 250 tokens must go in the same message, without splitting"
        );
    }

    /// Reordering of pending commands: a queued Add whose token was
    /// unsubscribed afterwards must not subscribe it (it ends up empty), and a queued Remove
    /// whose token was re-added afterwards must not unsubscribe it.
    #[test]
    fn retain_desired_filters_stale_commands() {
        let desired: std::collections::HashSet<String> =
            ["alive".to_string(), "revived".to_string()]
                .into_iter()
                .collect();

        // Pending Add: "dead" was unsubscribed afterwards → filtered out.
        let mut add = ShardCmd::Add(vec!["alive".into(), "dead".into()]);
        retain_desired(&desired, &mut add);
        match &add {
            ShardCmd::Add(ts) => assert_eq!(ts, &vec!["alive".to_string()]),
            _ => panic!(),
        }

        // Pending Remove: "revived" was re-added afterwards → filtered out.
        let mut rm = ShardCmd::Remove(vec!["revived".into(), "expired".into()]);
        retain_desired(&desired, &mut rm);
        match &rm {
            ShardCmd::Remove(ts) => assert_eq!(ts, &vec!["expired".to_string()]),
            _ => panic!(),
        }

        // Entirely obsolete command → empty (the caller discards it).
        let mut obsolete = ShardCmd::Add(vec!["dead".into()]);
        retain_desired(&desired, &mut obsolete);
        assert!(obsolete.is_empty());
    }
}
