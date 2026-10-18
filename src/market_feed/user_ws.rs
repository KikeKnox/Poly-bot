//! Polymarket user WS channel.
//!
//! Receives events for our own trades (MATCHED → MINED → CONFIRMED) and persists them
//! in fill_sink. Only MATCHED is processed, to avoid duplicates with MINED/CONFIRMED.

use dashmap::DashSet;
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use rust_decimal::Decimal;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};

use super::registry::MarketRegistry;
use crate::types::{DeadLetterFill, DerivedCreds, Fill, FillSide, Mode, StrategyId};

const USER_WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/user";
const RECONNECT_SECS: u64 = 5;
/// Cadence of the textual "PING" keepalive (the usual value).
const USER_WS_PING_SECS: u64 = 10;
/// Maximum silence tolerated before the channel is declared zombie. The venue
/// answers each "PING" with a textual "PONG" (~10s), so 45s ≈ 4 pings
/// lost — a half-open socket is detected here, not when TCP breaks.
const USER_WS_STALE_SECS: u64 = 45;
/// Bound on any write to the socket (auth, PING, Pong). A TCP
/// half-open also hangs `send()`, not only `read.next()` — without this
/// timeout the ping-tick watchdog would never reach the `last_msg` check.
const USER_WS_WRITE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// Margin applied to the gap-backfill anchor (`since - margin`) to
/// absorb clock jitter/drift between the local instant of the last frame and
/// what the CLOB considers "executed" for that trade.
const BACKFILL_MARGIN_MS: i64 = 5_000;

/// Retry cadence of the gap backfill while the
/// session stays alive — bounded exponential backoff (`max`) and a bound per fetch
/// (`fetch_timeout`) so a hung REST call does not delay reconnection.
/// Injectable in tests.
#[derive(Debug, Clone, Copy)]
struct BackfillRetryCfg {
    /// Wait after the first failed attempt.
    initial: std::time::Duration,
    /// Upper bound of the backoff.
    max: std::time::Duration,
    /// Bound for ONE complete REST fetch of the backfill.
    fetch_timeout: std::time::Duration,
}

impl Default for BackfillRetryCfg {
    fn default() -> Self {
        Self {
            initial: std::time::Duration::from_secs(2),
            max: std::time::Duration::from_secs(60),
            fetch_timeout: std::time::Duration::from_secs(30),
        }
    }
}

/// Result of one user WS session: how the loop ended and the
/// wall-clock watermark (ms epoch) of the LAST frame received. That instant — not
/// the moment the cut was detected — is the gap-backfill anchor: a zombie
/// channel can take minutes to die after it stops delivering traffic, and
/// anchoring the backfill to `Utc::now()` left the real gap uncovered.
struct SessionEnd {
    result: anyhow::Result<()>,
    last_traffic_ms: i64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct UserTrade {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub asset_id: String, // token_id
    #[serde(default)]
    pub market: String, // condition_id
    #[serde(default)]
    pub side: String, // BUY | SELL
    #[serde(default)]
    pub size: String,
    #[serde(default)]
    pub price: String,
    #[serde(default)]
    pub status: String, // MATCHED | MINED | CONFIRMED | FAILED
    #[serde(default)]
    pub trader_side: String, // taker | maker
    /// ARB pair id — only emitted by the sim (submit_pair); the real WS
    /// does not carry this field → None in live.
    #[serde(default)]
    pub hedge_id: Option<i64>,
    /// Authoritative strategy — only emitted by the sim (which always knows it
    /// from the OrderRequest). The real WS does not carry this field → None in
    /// live, where fill_sink resolves it via order_map. Without this, two
    /// strategies on the same token collided in the order_map
    /// (key market:token) and fills were attributed to whichever wrote last.
    #[serde(default)]
    pub strategy: Option<crate::types::StrategyId>,
    /// Real bracket close, when the router knows it (it comes
    /// from `OrderRequest::real_clear_at`, populated since `OrderIntent::real_clear_at`
    /// — only post_close_sweep uses it today, with its synthetic GTD `clear_at`). Only
    /// the sim (paper) emits it; the real WS does not carry this field → `None` in live.
    /// `handle_trade` uses it as a last resort if the market is in neither the
    /// registry nor the `clear_at_cache`.
    #[serde(default)]
    pub real_clear_at: Option<i64>,
    /// Integrated close: id of the ORDER this execution belongs to.
    /// The real WS carries it as `taker_order_id` (taker side); the simulator
    /// stamps it here explicitly. `id` remains the execution identity (trade id).
    /// CAUTION: on a fill where WE are the maker, `taker_order_id`
    /// is the COUNTERPARTY's order — ours is in `maker_orders`.
    #[serde(default, alias = "taker_order_id")]
    pub order_id: Option<String>,
    /// Integrated close: local identity of the order — only emitted by the sim.
    #[serde(default)]
    pub local_order_id: Option<String>,
    /// Integrated close: maker orders of the real trade (live user channel).
    /// When `trader_side == "maker"` our order is HERE, not in
    /// `taker_order_id`. `handle_trade` matches it against `known_order_ids`.
    #[serde(default)]
    pub maker_orders: Vec<MakerOrderRef>,
}

/// Reference to one of our maker orders within the user channel's trade.
/// The real payload carries `matched_amount`, `price`, `asset_id` and
/// `side` per leg — needed to emit ONE fill for each of OUR orders
/// matched in the aggregate trade (previously only one was emitted).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MakerOrderRef {
    #[serde(default)]
    pub order_id: String,
    /// Shares executed on THIS leg (not the trade total).
    #[serde(default)]
    pub matched_amount: Option<String>,
    /// Price of THIS leg.
    #[serde(default)]
    pub price: Option<String>,
    /// Token of THIS leg (may differ between legs of the same trade).
    #[serde(default)]
    pub asset_id: Option<String>,
    /// Side of THIS leg ("BUY"/"SELL", maker perspective). The wire carries it
    /// (same field as `MakerOrder.side` of the REST SDK used in
    /// `clob_sdk::fetch_trades_since`). Authoritative when present: in a
    /// complement match (our BUY NO matched against a taker BUY YES) it is NOT
    /// the inversion of the taker's side — see `resolve_leg_side` (explicit
    /// resolution, fail-closed).
    #[serde(default)]
    pub side: Option<String>,
}

/// `bundle`/`state` enable the gap backfill on every reconnection — the
/// startup backfill (`main.rs::backfill_missed_trades`) only covers the
/// downtime BEFORE the first connect of this session, not a WS drop
/// mid-session. `None` (impossible today: `market_feed::run` only spawns
/// this task when `creds` is `Some`, and both travel together since
/// `live_clob_for_reconcile` in `main.rs`) simply disables the gap backfill
/// without affecting the rest of the WS.
pub async fn run(
    creds: Arc<DerivedCreds>,
    registry: Arc<MarketRegistry>,
    fill_tx: mpsc::UnboundedSender<Fill>,
    known_order_ids: Arc<DashSet<String>>,
    bundle: Option<Arc<crate::clob_sdk::AuthedClobClient>>,
    state: Option<Arc<crate::state::StateStore>>,
    // Orphan LIVE fills (unknown market) → dead-letter in the DB
    // instead of being dropped with just a log. `None` in tests/paper.
    dead_letter_tx: Option<mpsc::UnboundedSender<DeadLetterFill>>,
) {
    // Gap-backfill anchor = ms epoch (wall clock) of the LAST
    // traffic received by the session earlier — NOT the instant the
    // cut was detected (previously `disconnected_at_ms = Utc::now()` on leaving the
    // loop: a zombie channel could stay silent for minutes and the backfill only
    // covered the last ~5s). The OLDEST mark is kept while there is an
    // unrecovered gap: if the backfill fails, the watermark does not advance
    // and the next session retries the whole gap (dedup in fill_sink).
    // `None` on the first iteration — main.rs already did the startup backfill.
    let mut gap_since_ms: Option<i64> = None;
    loop {
        let end = connect_and_run(
            &creds,
            &registry,
            &fill_tx,
            &known_order_ids,
            &dead_letter_tx,
            &mut gap_since_ms,
            bundle.as_deref(),
            state.as_deref(),
        )
        .await;
        match end.result {
            Ok(()) => {
                tracing::warn!("user_ws: connection closed, reconnecting in {RECONNECT_SECS}s")
            }
            Err(e) => tracing::warn!(error = %e, reconnect_secs = RECONNECT_SECS, "user_ws: error"),
        }
        gap_since_ms = Some(match gap_since_ms {
            Some(prev) => prev.min(end.last_traffic_ms),
            None => end.last_traffic_ms,
        });
        tokio::time::sleep(std::time::Duration::from_secs(RECONNECT_SECS)).await;
    }
}

async fn connect_and_run(
    creds: &DerivedCreds,
    registry: &MarketRegistry,
    fill_tx: &mpsc::UnboundedSender<Fill>,
    known_order_ids: &Arc<DashSet<String>>,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
    gap_since_ms: &mut Option<i64>,
    bundle: Option<&crate::clob_sdk::AuthedClobClient>,
    state: Option<&crate::state::StateStore>,
) -> SessionEnd {
    // Real CLOB fetcher — in `run_session` it is injectable so that tests
    // exercise the backfill retries without network or credentials.
    let fetch = bundle.map(|b| {
        let own_address = creds
            .funder_address
            .clone()
            .unwrap_or_else(|| creds.eoa_address.clone());
        move |after_ts: i64| {
            let own = own_address.clone();
            async move { crate::clob_sdk::fetch_trades_since(b, after_ts, &own).await }
        }
    });
    run_session(
        USER_WS_URL,
        creds,
        registry,
        fill_tx,
        known_order_ids,
        dead_letter_tx,
        gap_since_ms,
        fetch,
        state,
        BackfillRetryCfg::default(),
        BACKFILL_MARGIN_MS,
        std::time::Duration::from_secs(USER_WS_PING_SECS),
        std::time::Duration::from_secs(USER_WS_STALE_SECS),
    )
    .await
}

/// One user WS session: connect + auth + `join!` of the read loop with the
/// gap backfill. Injectable (`url`, `fetch`, `retry`, timeouts) for tests
/// with a local server/stream and a simulated REST fetch.
///
/// The backfill is no longer a single attempt inside the join! —
/// `backfill_gap_supervised` retries with bounded backoff WHILE the session
/// stays alive (a fetch that fails once no longer leaves the gap unrecovered
/// until the next reconnection). When `session_loop` ends (Close, error,
/// watchdog), `session_done` is signalled and the supervisor stops waiting
/// immediately — `join!` never delays reconnection by more than one in-flight
/// attempt (each fetch is bounded by `cfg.fetch_timeout`). If the gap remains
/// pending, `gap_since_ms` is NOT released: the next session retries the
/// same window.
#[allow(clippy::too_many_arguments)]
async fn run_session<F, Fut>(
    url: &str,
    creds: &DerivedCreds,
    registry: &MarketRegistry,
    fill_tx: &mpsc::UnboundedSender<Fill>,
    known_order_ids: &Arc<DashSet<String>>,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
    gap_since_ms: &mut Option<i64>,
    fetch: Option<F>,
    state: Option<&crate::state::StateStore>,
    retry: BackfillRetryCfg,
    backfill_margin_ms: i64,
    ping_every: std::time::Duration,
    stale_after: std::time::Duration,
) -> SessionEnd
where
    F: Fn(i64) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<Vec<crate::clob_sdk::ExecutedTrade>>>,
{
    // If there is not even a connect, this value is "now" — run() combines it
    // with min() against the previous anchor, which thus never moves forward.
    let mut last_traffic_ms = chrono::Utc::now().timestamp_millis();
    let result = async {
        let (ws_stream, _) = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            connect_async(url),
        )
        .await
        .map_err(|_| anyhow::anyhow!("user_ws: connection timeout"))??;

        let (mut write, mut read) = ws_stream.split();

        let auth_msg = serde_json::json!({
            "type":    "subscribe",
            "channel": "user",
            "auth": {
                "apiKey":     creds.api_key,
                "secret":     creds.api_secret,
                "passphrase": creds.api_passphrase,
            }
        });
        send_bounded(&mut write, Message::Text(auth_msg.to_string())).await?;
        tracing::info!("user_ws: connected and authenticated");

        // End-of-session signal for the backfill supervisor: when the
        // read loop ends, retries stop waiting so as not to
        // delay reconnection.
        let (done_tx, done_rx) = watch::channel(false);

        // The gap backfill runs CONCURRENTLY with the session loop (not before the
        // connect, as in the earlier fix): this covers the interval
        // [last_traffic, reconnection_ok] fully — previously the stretch between
        // the backfill itself and the new subscription was left uncovered. join! (not
        // spawn): the loop keeps answering the venue's pings while the REST fetch is
        // in flight, and the result decides whether the anchor is released.
        // If the backfill stays pending, `gap_since_ms` is NOT touched — the next
        // session retries the same gap.
        let backfill = async {
            if let (Some(since_ms), Some(f), Some(st)) = (*gap_since_ms, fetch.as_ref(), state) {
                return backfill_gap_supervised(
                    f,
                    st,
                    registry,
                    fill_tx,
                    dead_letter_tx,
                    since_ms - backfill_margin_ms,
                    retry,
                    done_rx,
                )
                .await;
            }
            // Only "covered" if there was nothing to cover. A pending gap without a
            // fetcher/state cannot be recovered, and releasing the anchor here would
            // lose it forever.
            gap_since_ms.is_none()
        };
        let session = async {
            let res = session_loop(
                &mut write,
                &mut read,
                registry,
                fill_tx,
                known_order_ids,
                dead_letter_tx,
                ping_every,
                stale_after,
                &mut last_traffic_ms,
            )
            .await;
            let _ = done_tx.send(true);
            res
        };
        let (res, backfill_ok) = tokio::join!(session, backfill);
        if backfill_ok {
            *gap_since_ms = None;
        }
        res
    }
    .await;
    SessionEnd {
        result,
        last_traffic_ms,
    }
}

/// Liveness watchdog loop of the user WS session, with a stale-socket check.
///
/// `last_msg` (monotonic) is refreshed by ANY frame received — the
/// textual "PONG" is the sign of life that survives legitimate silence (the
/// user channel only speaks when there are fills), same pattern as
/// `oracle/liquidations.rs` and `ws.rs`. If the venue stops delivering for more than
/// `stale_after`, the socket is half-open and it bails out to reconnect
/// instead of waiting for the TCP timeout (which might never arrive).
///
/// `last_traffic_ms` (out-param) receives the wall-clock ms epoch of the last frame
/// of EVENTS (non-PONG JSON) — `run` uses it as the gap-backfill anchor
/// (see `SessionEnd`). Err/Close and keepalives do NOT move it: a channel with
/// a desynchronized subscription may keep answering PONGs while it
/// no longer delivers events; anchoring at the last real event keeps the gap
/// conservative (the extra window is re-queried — idempotent dedup —, but
/// no fill from the silence is left out).
#[allow(clippy::too_many_arguments)]
async fn session_loop<W, R>(
    write: &mut W,
    read: &mut R,
    registry: &MarketRegistry,
    fill_tx: &mpsc::UnboundedSender<Fill>,
    known_order_ids: &Arc<DashSet<String>>,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
    ping_every: std::time::Duration,
    stale_after: std::time::Duration,
    last_traffic_ms: &mut i64,
) -> anyhow::Result<()>
where
    W: Sink<Message, Error = WsError> + Unpin,
    R: Stream<Item = Result<Message, WsError>> + Unpin,
{
    let mut ping_int = tokio::time::interval(ping_every);
    ping_int.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_msg = std::time::Instant::now();
    *last_traffic_ms = chrono::Utc::now().timestamp_millis();

    loop {
        tokio::select! {
            msg = read.next() => {
                let Some(msg) = msg else { break };
                // Liveness vs gap evidence: any frame the socket DELIVERS (including
                // a Close or a protocol error) proves the channel is alive and
                // refreshes the `last_msg` watchdog. But `last_traffic_ms` is the
                // backfill ANCHOR: it advances only with EVENT traffic, so an
                // Err/Close after a silence must NOT move the anchor to the
                // detection instant, because the real gap (since the last valid
                // frame) would otherwise stay uncovered. Hence the anchor only
                // advances with events.
                last_msg = std::time::Instant::now();
                let m = match msg {
                    Ok(m) => m,
                    Err(e) => return Err(e.into()),
                };
                if matches!(m, Message::Close(_)) {
                    break;
                }
                match m {
                    // "PONG" text reply to the keepalive: liveness only, nothing
                    // more. It already refreshed `last_msg` above (it is the sign of
                    // life during legitimate silence), but it does NOT move the
                    // anchor: the channel might be alive yet not delivering events
                    // (desynchronized subscription), and the real uncovered window
                    // starts at the last event received.
                    Message::Text(t) if t == "PONG" => {}
                    Message::Text(t)  => {
                        *last_traffic_ms = chrono::Utc::now().timestamp_millis();
                        dispatch(&t, registry, fill_tx, known_order_ids, Mode::Live, dead_letter_tx);
                    }
                    Message::Ping(p)  => { send_bounded(write, Message::Pong(p)).await?; }
                    Message::Close(_) => unreachable!("Close filtered above"),
                    _                 => {}
                }
            }
            _ = ping_int.tick() => {
                if last_msg.elapsed() > stale_after {
                    metrics::counter!("hrf_user_ws_zombie_total").increment(1);
                    anyhow::bail!(
                        "user_ws: zombie connection (no messages for {}s) — reconnecting",
                        stale_after.as_secs()
                    );
                }
                send_bounded(write, Message::Text("PING".into())).await?;
            }
        }
    }
    Ok(())
}

/// `send` to the socket with a time bound: a half-open TCP also hangs the
/// writes; without the timeout the watchdog would never run.
async fn send_bounded<W>(write: &mut W, msg: Message) -> anyhow::Result<()>
where
    W: Sink<Message, Error = WsError> + Unpin,
{
    match tokio::time::timeout(USER_WS_WRITE_TIMEOUT, write.send(msg)).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e.into()),
        Err(_) => Err(anyhow::anyhow!("user_ws: socket write timeout")),
    }
}

/// Supervisor of the gap backfill: retries `backfill_gap_once` with bounded
/// exponential backoff (`cfg.initial`, doubling up to `cfg.max`) while the WS
/// session stays alive. This is a cadence, not a silent give-up: a failed fetch
/// no longer leaves the gap unrecovered until the next reconnection.
///
/// Returns `true` when the gap is covered; `false` if the session died first
/// (`session_done`): the `gap_since_ms` anchor is then left intact and the next
/// session retries the same window. The `select!` against `session_done` ensures
/// the `join!` in `run_session` never waits on a backoff sleep to reconnect; an
/// in-flight attempt is bounded by `cfg.fetch_timeout` inside `backfill_gap_once`.
#[allow(clippy::too_many_arguments)]
async fn backfill_gap_supervised<F, Fut>(
    fetch: &F,
    state: &crate::state::StateStore,
    registry: &MarketRegistry,
    fill_tx: &mpsc::UnboundedSender<Fill>,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
    since_ts_ms: i64,
    cfg: BackfillRetryCfg,
    mut session_done: watch::Receiver<bool>,
) -> bool
where
    F: Fn(i64) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<Vec<crate::clob_sdk::ExecutedTrade>>>,
{
    let mut delay = cfg.initial;
    // Evidence dedup across retries: a market that stays permanently
    // unknown must not insert the same dead-letter on every cycle (the
    // fetch is repeated with the SAME window while the gap is pending).
    let mut dead_lettered: HashSet<String> = HashSet::new();
    loop {
        if *session_done.borrow() {
            return false;
        }
        let covered = tokio::select! {
            r = backfill_gap_once(
                fetch, state, registry, fill_tx, dead_letter_tx,
                since_ts_ms, cfg.fetch_timeout, &mut dead_lettered,
            ) => r,
            _ = session_done.changed() => return false,
        };
        if covered {
            return true;
        }
        tokio::select! {
            _ = tokio::time::sleep(delay) => {
                delay = (delay * 2).min(cfg.max);
            }
            _ = session_done.changed() => return false,
        }
    }
}

/// One backfill attempt for the reconnection gap of the user WS (same logic as
/// `main.rs::backfill_missed_trades`, which runs ONLY ONCE at process startup;
/// this covers intermediate reconnections, via `backfill_gap_supervised` inside
/// the `join!` of `run_session`): bounded REST fetch (`fetch_timeout`) plus
/// re-injection of each trade as a `Fill`. Idempotent: the `UNIQUE(exec_id)`
/// dedup in fill_sink discards repeats between retries and against WS delivery.
///
/// Returns `true` only if ALL trades were resolved (re-injected
/// with real metadata, already persisted, or empty window); `false` = gap
/// pending (fetch failed or timed out, dedup check failed, `fill_tx.send`
/// failed — a queued send is not durable — or unresolvable metadata → dead-letter),
/// and the caller keeps the `gap_since_ms` anchor to retry the SAME
/// window.
///
/// Metadata comes from `lookup_market_info` (registry asset → `clear_at_cache`),
/// not from `match_ts`: a gap of seconds almost always falls on markets that are
/// still LIVE, and `clear_at = match_ts` would persist them as expired.
/// If there is no durable evidence either (`durable_market_info`), `clear_at`
/// and `neg_risk` are NOT fabricated: the trade goes to `dead_letter_fills`
/// (the dead-letter channel) for manual reconciliation.
#[allow(clippy::too_many_arguments)]
async fn backfill_gap_once<F, Fut>(
    fetch: &F,
    state: &crate::state::StateStore,
    registry: &MarketRegistry,
    fill_tx: &mpsc::UnboundedSender<Fill>,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
    since_ts_ms: i64,
    fetch_timeout: std::time::Duration,
    dead_lettered: &mut HashSet<String>,
) -> bool
where
    F: Fn(i64) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<Vec<crate::clob_sdk::ExecutedTrade>>>,
{
    // The fetch is bounded: a hung REST call cannot block either the
    // session or the reconnection (run_session's join! waits on this branch).
    let trades = match tokio::time::timeout(fetch_timeout, fetch(since_ts_ms / 1000)).await {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "user_ws: gap backfill on reconnection failed — fills from the gap may be missing");
            return false;
        }
        Err(_) => {
            tracing::warn!(
                timeout_secs = fetch_timeout.as_secs(),
                "user_ws: gap backfill fetch exceeded the timeout — will retry",
            );
            return false;
        }
    };
    if trades.is_empty() {
        tracing::debug!("user_ws: reconnection gap backfill — no new trades");
        return true;
    }
    tracing::info!(n = trades.len(), "user_ws: reconnection gap backfill — re-injecting CLOB trades (dedup in fill_sink)");
    let mut pending = false;
    for t in trades {
        // Idempotent across retries and with WS delivery: an exec_id already
        // persisted is neither re-queued nor re-dead-lettered. If the lookup fails,
        // there is no evidence — the trade is processed anyway (the durable dedup in
        // fill_sink absorbs repeats) but the gap stays marked pending.
        match state.fill_exists_exec(&t.exec_id).await {
            Ok(true) => continue,
            Ok(false) => {}
            Err(e) => {
                tracing::warn!(
                    error = %e, exec_id = %t.exec_id,
                    "user_ws: could not verify dedup of the backfilled trade",
                );
                pending = true;
            }
        }
        let strategy = match &t.own_order_id {
            Some(oid) => state
                .get_gtd_strategy(oid)
                .await
                .ok()
                .flatten()
                .unwrap_or(StrategyId::Arb),
            None => StrategyId::Arb,
        };
        // clear_at/neg_risk/label come from the registry asset or `clear_at_cache`
        // if the market is still there; otherwise from durable evidence (persisted
        // positions, `markets_observed`). If nothing answers, NO clear_at is
        // fabricated: neither `clear_at=match_ts` (a false close assumption) nor
        // `clear_at=0` (eternal position). The trade goes to dead-letter for
        // reconciliation and the gap stays pending. Known limitation: an evicted
        // market without durable evidence stays blocked.
        let market_info = match lookup_market_info(registry, &t.market_cid, &t.token_id) {
            Some(info) => Some(info),
            None => durable_market_info(state, &t, strategy).await,
        };
        let Some((neg_risk, clear_at, outcome_label)) = market_info else {
            pending = true;
            metrics::counter!("hrf_user_ws_backfill_metadata_unknown_total").increment(1);
            tracing::error!(
                market = %t.market_cid,
                token_id = %t.token_id,
                trade_id = %t.trade_id,
                exec_id = %t.exec_id,
                "user_ws: gap backfill — market without metadata in registry/cache \
                 nor durable evidence: trade to dead-letter, gap kept for retry"
            );
            // Only one dead-letter per exec_id and process: the supervisor's
            // retries bring back the same trade with the same window.
            if dead_lettered.insert(t.exec_id.clone()) {
                let dl = DeadLetterFill {
                    order_id: t.own_order_id.clone().unwrap_or_default(),
                    market_cid: t.market_cid.clone(),
                    token_id: t.token_id.clone(),
                    side: if t.is_buy {
                        FillSide::Buy
                    } else {
                        FillSide::Sell
                    },
                    price: t.price,
                    size: t.size,
                    status: "MATCHED".to_string(),
                    role: if t.exec_id == t.trade_id {
                        "taker".to_string()
                    } else {
                        "maker".to_string()
                    },
                    reason: format!(
                        "gap backfill: market without metadata in registry/clear_at_cache \
                         nor durable evidence; gap kept for retry (trade_id={})",
                        t.trade_id
                    ),
                    ts: chrono::Utc::now(),
                };
                match dead_letter_tx {
                    Some(tx) => {
                        if let Err(e) = tx.send(dl) {
                            tracing::error!(error = %e, "user_ws: dead-letter channel closed during gap backfill");
                        }
                    }
                    None => tracing::warn!(
                        trade_id = %t.trade_id,
                        "user_ws: no dead-letter channel — unknown trade only left in the log",
                    ),
                }
            }
            continue;
        };
        let fill = Fill {
            market_cid: t.market_cid,
            token_id: t.token_id,
            side: if t.is_buy {
                FillSide::Buy
            } else {
                FillSide::Sell
            },
            price: t.price,
            size: t.size,
            // Integrated close: `trade_id` is the EXECUTION (exec_id) and
            // `own_order_id` is the ORDER — previously order_id carried the trade id and
            // attribution by order_index/gtd_orders never matched.
            order_id: t.own_order_id.clone(),
            // `exec_id` now carries the :order_id suffix on maker legs
            // (an aggregated trade may match several of our orders).
            exec_id: Some(t.exec_id),
            ts: chrono::DateTime::from_timestamp(t.match_ts, 0).unwrap_or_else(chrono::Utc::now),
            mode: Mode::Live,
            neg_risk,
            strategy,
            clear_at,
            outcome_label,
            hedge_id: None,
            local_order_id: None,
        };
        if let Err(e) = fill_tx.send(fill) {
            tracing::error!(error = %e, "user_ws: fill_sink closed during gap backfill");
            return false;
        }
    }
    !pending
}

/// Market metadata from DURABLE evidence when neither the registry asset nor
/// `clear_at_cache` has it (TTL eviction, restart). Only the existing
/// `StateStore` queries are used, without extending the schema:
///
/// 1. A persisted position on the SAME token+strategy (`clear_at > 0`) —
///    its `clear_at`/`neg_risk`/`outcome_label` are real (set by
///    `apply_fill` with good metadata when the market was known).
/// 2. Any open position of the bracket with `clear_at > 0` and a matching cid —
///    sets `clear_at`/`neg_risk`; the label is reused only if the
///    position is on the same token (a position on the opposite token
///    does not determine the label of THIS leg).
///
/// `markets_observed` is no longer enough on its own: the table stores
/// `neg_risk`, but the only existing read (`markets_pending_official`)
/// does not project it — returning `false` as a "safe default" turned the
/// ABSENCE of data into evidence. Without a verifiable `neg_risk` the trade stays
/// pending (dead-letter + retry, see caller) until a query exposing it
/// exists (extending `StateStore` is out of scope here).
///
/// `None` = no complete evidence — the caller does NOT invent the close.
async fn durable_market_info(
    state: &crate::state::StateStore,
    t: &crate::clob_sdk::ExecutedTrade,
    strategy: StrategyId,
) -> Option<(bool, i64, Option<String>)> {
    if let Ok(Some(p)) = state.get_position(&t.token_id, &strategy.to_string()).await {
        if p.clear_at > 0 {
            return Some((p.neg_risk, p.clear_at, p.outcome_label.clone()));
        }
    }
    if let Ok(positions) = state.list_open_positions().await {
        if let Some(p) = positions
            .iter()
            .find(|p| p.market_cid == t.market_cid && p.clear_at > 0)
        {
            let label = if p.token_id == t.token_id {
                p.outcome_label.clone()
            } else {
                None
            };
            return Some((p.neg_risk, p.clear_at, label));
        }
    }
    None
}

fn dispatch(
    text: &str,
    registry: &MarketRegistry,
    fill_tx: &mpsc::UnboundedSender<Fill>,
    known_order_ids: &DashSet<String>,
    mode: Mode,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
) {
    let val = match serde_json::from_str::<serde_json::Value>(text) {
        Ok(v) => v,
        Err(_) => {
            tracing::debug!(raw = %text, "user_ws: unparseable message");
            return;
        }
    };

    let items: Vec<serde_json::Value> = if val.is_array() {
        val.as_array().cloned().unwrap_or_default()
    } else {
        vec![val]
    };

    for item in items {
        let event_type = item
            .get("event_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        match event_type {
            "trade" => match serde_json::from_value::<UserTrade>(item) {
                Ok(t) => handle_trade(t, registry, fill_tx, known_order_ids, mode, dead_letter_tx),
                Err(e) => tracing::warn!(error = %e, "user_ws: unparseable trade"),
            },
            "order" => handle_order_cancel(&item, fill_tx),
            _ => tracing::debug!(event_type, "user_ws: event ignored"),
        }
    }
}

/// `neg_risk`/`clear_at` and the token_id pair (YES, NO) of `market_cid`,
/// looking first in the ASSET registry (`registry.markets`) and, if absent,
/// in the long-lived `clear_at_cache`.
/// `None` if it is in neither. Returns the ids so that the
/// outcome_label is derived PER LEG with `label_for` — an aggregated maker trade
/// may match our legs on both tokens of the bracket.
fn lookup_market_meta(
    registry: &MarketRegistry,
    market_cid: &str,
) -> Option<(bool, i64, String, String)> {
    if let Some(m) = registry.markets.get(market_cid) {
        return Some((
            m.neg_risk,
            m.clear_at,
            m.yes_token_id.clone(),
            m.no_token_id.clone(),
        ));
    }
    if let Some(cached) = registry.clear_at_cache.get(market_cid) {
        let (neg_risk, clear_at, yes_token_id, no_token_id) = cached.value();
        return Some((
            *neg_risk,
            *clear_at,
            yes_token_id.clone(),
            no_token_id.clone(),
        ));
    }
    None
}

/// `neg_risk`/`clear_at`/YES-NO label of `market_cid` for ONE concrete
/// token (`asset_id`), via `lookup_market_meta`. `None` if the
/// market is in neither the asset registry nor `clear_at_cache`.
fn lookup_market_info(
    registry: &MarketRegistry,
    market_cid: &str,
    asset_id: &str,
) -> Option<(bool, i64, Option<String>)> {
    let (neg_risk, clear_at, yes_token_id, no_token_id) =
        lookup_market_meta(registry, market_cid)?;
    Some((
        neg_risk,
        clear_at,
        label_for(asset_id, &yes_token_id, &no_token_id),
    ))
}

fn label_for(asset_id: &str, yes_token_id: &str, no_token_id: &str) -> Option<String> {
    if asset_id == yes_token_id {
        Some("YES".to_string())
    } else if asset_id == no_token_id {
        Some("NO".to_string())
    } else {
        None
    }
}

/// Side of ONE maker leg — EXPLICIT, fail-closed resolution.
/// `Some` only if the side is verifiable:
///
/// - `maker_orders[].side` with "BUY"/"SELL" (maker perspective — the same
///   field as `MakerOrder.side` of the REST SDK used in
///   `clob_sdk::fetch_trades_since`): authoritative. In a complement match
///   (taker BUY YES ↔ maker BUY NO) the NO leg carries the side
///   inverted relative to the taker, and the wire resolves it.
/// - No `side` on the wire (absent or empty): it is only deduced by inverting
///   the taker's side when the leg's token IS the taker's
///   (`t.asset_id`) — the taker side is already validated in `handle_trade`
///   (returns early if it is not BUY/SELL).
///
/// `None` = side ambiguous: complement leg or unknown token without a
/// `side` on the wire (there is no safe way to deduce it), or a `side` present
/// but unreadable — an explicitly invalid value does not fall back, because
/// that would prefer the deduction over contradictory real data. The
/// caller does NOT book the leg: it goes to dead-letter as uncertain evidence.
fn resolve_leg_side(
    m: &MakerOrderRef,
    taker_token: &str,
    deduced: &FillSide,
) -> Option<FillSide> {
    match m.side.as_deref().filter(|s| !s.is_empty()) {
        Some(s) if s.eq_ignore_ascii_case("BUY") => Some(FillSide::Buy),
        Some(s) if s.eq_ignore_ascii_case("SELL") => Some(FillSide::Sell),
        Some(_) => None,
        None => match m.asset_id.as_deref() {
            Some(tok) if !tok.is_empty() && tok == taker_token => Some(deduced.clone()),
            _ => None,
        },
    }
}

/// Side for a dead-letter ROW — the schema requires
/// `FillSide`, but a DL is EVIDENCE, not an executable fill. If
/// `resolve_leg_side` does not resolve it, `deduced` is kept as a placeholder and
/// the `reason` explicitly declares it NOT verified ("side ambiguous"),
/// so that nobody reads it as the execution's real side.
fn dl_leg_side(
    m: &MakerOrderRef,
    taker_token: &str,
    deduced: &FillSide,
) -> (FillSide, &'static str) {
    match resolve_leg_side(m, taker_token, deduced) {
        Some(s) => (s, ""),
        None => (
            deduced.clone(),
            "; side ambiguous (no valid BUY/SELL on the wire, no verifiable taker inversion) — \
             this row's side is a PLACEHOLDER, \
             NOT verified",
        ),
    }
}

// Integrated close: `known_order_ids` is no longer decorative — a live fill
// where WE are maker carries in `maker_orders` the passive-side orders (ours
// among them); matching against the order_ids the executor registered at
// submit (or restored at startup) is how OURS is identified
// without falling into `taker_order_id`, which there is the counterparty's order.
pub fn handle_trade(
    t: UserTrade,
    registry: &MarketRegistry,
    fill_tx: &mpsc::UnboundedSender<Fill>,
    known_order_ids: &DashSet<String>,
    mode: Mode,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
) {
    tracing::info!(
        id     = %t.id,
        market = %t.market,
        side   = %t.side,
        size   = %t.size,
        price  = %t.price,
        status = %t.status,
        role   = %t.trader_side,
        "user_ws: trade"
    );

    // Only MATCHED, so as not to duplicate MINED/CONFIRMED of the same fill.
    if t.status != "MATCHED" {
        return;
    }

    // B: the trade's `side` is the TAKER's (the message is the same for both
    // parties). If our leg was maker (GTD filled), our side is the
    // opposite: a taker SELLING against our bid means we BUY.
    // Without this inversion, every maker fill in live was booked backwards.
    let taker_side = match t.side.as_str() {
        "BUY" => FillSide::Buy,
        "SELL" => FillSide::Sell,
        _ => {
            tracing::warn!(side = %t.side, "user_ws: unknown side");
            return;
        }
    };
    let side = if t.trader_side.eq_ignore_ascii_case("maker") {
        match taker_side {
            FillSide::Buy => FillSide::Sell,
            FillSide::Sell => FillSide::Buy,
            other => other,
        }
    } else {
        taker_side
    };

    let price = match t.price.parse::<Decimal>() {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(raw = %t.price, "user_ws: unparseable price");
            return;
        }
    };
    let size = match t.size.parse::<Decimal>() {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(raw = %t.size, "user_ws: unparseable size");
            return;
        }
    };

    // Market data (neg_risk, clear_at and the YES/NO token_id pair):
    // first the registry ASSET (short-TTL working set), and if it is not
    // there, the long-lived clear_at_cache — a legitimate fill of the
    // CURRENT session (e.g. the late GTD of post_close_sweep) can arrive
    // after `evict_expired` has already evicted the market from the short
    // working set. The outcome_label is NOT resolved here: it is derived per
    // leg with `label_for` over these ids, because one of our maker legs may
    // belong to the OTHER token of the bracket (complement match) and the
    // label of the taker's token (`t.asset_id`) does not apply to it.
    // In live: if it is NOT in the long-lived cache either, the fill is ignored.
    // It remains the real protection against GTDs from a PREVIOUS session (the
    // cache, like the asset registry, starts empty on every process restart).
    // In paper: same scheme, but if both fail it falls back to the usual
    // fallback, using `t.real_clear_at` (if the router brought it, e.g. for
    // post_close_sweep) instead of forcing clear_at=0 — needed in tests that
    // never go through real discovery and therefore populate neither the
    // registry nor the cache. Without token ids the label stays None per leg —
    // no metadata is invented.
    let market_meta = lookup_market_meta(registry, &t.market);
    let (neg_risk, clear_at, token_ids) = if mode == Mode::Live {
        let Some((neg_risk, clear_at, yes_token_id, no_token_id)) = market_meta else {
            // Previously this was a `debug!` (later an `error!` with no
            // persistence), indistinguishable between "fill from a PREVIOUS
            // session" (the original reason of the guard) and "fill of the
            // CURRENT session whose market dropped out of registry/clear_at_cache
            // through an eviction race or a short TTL" (real money already spent or
            // collected, with no trace). Now, besides the `error!`, the fill goes to
            // the `dead_letter_fills` table (`StateStore::record_dead_letter_fill`,
            // via the dedicated channel) for manual reconciliation later against
            // the CLOB by `order_id`. Discarding the position itself is still
            // required: without reliable neg_risk/clear_at/outcome_label there is
            // no safe way to book it as a normal `Fill`.
            //
            // The dead letter is emitted PER OWN LEG, verifiable (each with its
            // order_id/token/matched_amount/price), never `maker_orders.first()`
            // (which could belong to ANOTHER maker), nor the trade totals as if
            // they were the leg's. Without a verifiable own leg: a single evidence
            // NOT attributed to any order, with the trade and its legs serialized
            // in `reason`. See `dead_letter_orphan_trade`.
            // `dead_letter_orphan_trade`.
            metrics::counter!("hrf_user_ws_fill_orphaned_total").increment(1);
            dead_letter_orphan_trade(
                &t,
                &side,
                price,
                size,
                known_order_ids,
                dead_letter_tx,
                "market not found in registry or clear_at_cache",
            );
            return;
        };
        (neg_risk, clear_at, Some((yes_token_id, no_token_id)))
    } else {
        match market_meta {
            Some((neg_risk, clear_at, yes_token_id, no_token_id)) => {
                (neg_risk, clear_at, Some((yes_token_id, no_token_id)))
            }
            None => (false, t.real_clear_at.unwrap_or(0), None),
        }
    };

    // Identity separation for fills:
    // - `exec_id`  = identity of OUR execution: `t.id` (trade UUID) for taker
    //   or a single leg; `t.id:order_id` for each of our maker legs — an
    //   aggregated trade can match SEVERAL of our orders and they share
    //   trade_id. Without the suffix, the second order's leg collided on
    //   UNIQUE(exec_id) and was silently discarded (same format in the
    //   backfill/reconciler).
    // - `order_id` = the ORDER. If we are taker: `t.order_id` (`taker_order_id`).
    //   If we are maker in live: `taker_order_id` is the counterparty's order —
    //   ours is in `maker_orders`, matched against `known_order_ids`.
    //   If NO leg matches, no Fill is emitted: the trade goes to dead-letter as
    //   unattributed (see below). In paper the sim always stamps `t.order_id`.
    // - `local_order_id` = the local identity (attempt_id of the journal), only
    //   emitted by the sim; in live fill_sink resolves it via `order_index`.
    struct Leg {
        order_id: Option<String>,
        exec_id: String,
        token_id: String,
        /// Side of THIS leg (see `resolve_leg_side`): the wire one if it was
        /// valid; otherwise the taker's inversion, ONLY for legs of the same
        /// token. Any other case never reaches here: the leg is dead-lettered
        /// as ambiguous (fail-closed).
        side: FillSide,
        price: Decimal,
        size: Decimal,
    }

    let legs: Vec<Leg> = if mode == Mode::Live && t.trader_side.eq_ignore_ascii_case("maker") {
        let ours: Vec<&MakerOrderRef> = t
            .maker_orders
            .iter()
            .filter(|m| known_order_ids.contains(&m.order_id))
            .collect();
        if ours.is_empty() {
            // Without a verifiable own leg the trade does NOT reach the
            // accounting channel — not even with `order_id = None`
            // (`order_map[cid:token]` in fill_sink would resolve it and the
            // foreign aggregate would be applied to the ledger). Evidence that is
            // unambiguously NOT attributed is kept: empty order_id, the trade
            // totals declared and ALL the legs serialized in `reason`.
            metrics::counter!("hrf_user_ws_maker_unverifiable_total").increment(1);
            dead_letter_orphan_trade(
                &t,
                &side,
                price,
                size,
                known_order_ids,
                dead_letter_tx,
                "live maker trade without a verifiable own leg in known_order_ids",
            );
            return;
        }
        // One Fill per own leg, grouping repeated entries of the same
        // order_id (an order can only match one price per trade).
        // An own leg WITHOUT its fields (matched_amount, price or asset_id
        // absent/unparseable) is NOT booked with the trade aggregates: it goes
        // to dead-letter as uncertain evidence, with the reason stated in `reason`.
        let mut legs: Vec<Leg> = Vec::new();
        for &m in &ours {
            let leg_price = m.price.as_deref().and_then(|p| p.parse::<Decimal>().ok());
            let leg_size = m
                .matched_amount
                .as_deref()
                .and_then(|s| s.parse::<Decimal>().ok());
            let leg_token = m.asset_id.clone().filter(|s| !s.is_empty());
            let (Some(leg_price), Some(leg_size), Some(leg_token)) =
                (leg_price, leg_size, leg_token)
            else {
                dead_letter_incomplete_maker_leg(&t, &side, m, price, size, dead_letter_tx);
                continue;
            };
            // The side is fail-closed too: without a verifiable resolution
            // (complement leg or other token without `side` on the wire, or a
            // `side` present but unreadable) the leg is NOT booked with a
            // fabricated side: it goes to uncertain dead-letter, and the
            // remaining healthy legs keep emitting their fill.
            let Some(leg_side_res) = resolve_leg_side(m, &t.asset_id, &side) else {
                dead_letter_ambiguous_maker_leg(&t, &side, m, price, size, dead_letter_tx);
                continue;
            };
            if let Some(p) = legs
                .iter_mut()
                .find(|p| p.order_id.as_deref() == Some(m.order_id.as_str()))
            {
                p.size += leg_size;
            } else {
                legs.push(Leg {
                    order_id: Some(m.order_id.clone()),
                    exec_id: crate::clob_sdk::maker_exec_id(&t.id, &m.order_id),
                    token_id: leg_token,
                    side: leg_side_res,
                    price: leg_price,
                    size: leg_size,
                });
            }
        }
        legs
    } else {
        vec![Leg {
            order_id: t.order_id.clone(),
            exec_id: t.id.clone(),
            token_id: t.asset_id.clone(),
            side,
            price,
            size,
        }]
    };

    for leg in legs {
        if leg.size <= Decimal::ZERO {
            continue;
        }
        // outcome_label PER LEG: `label_for` of the leg's token against the
        // cid's ids, not the taker's token (`t.asset_id`), since one of our legs
        // may be of the OTHER token of the bracket. If the leg's token matches
        // neither yes nor no, the label is None (no metadata is invented).
        let outcome_label = token_ids
            .as_ref()
            .and_then(|(yes_token_id, no_token_id)| {
                label_for(&leg.token_id, yes_token_id, no_token_id)
            });
        let fill = Fill {
            market_cid: t.market.clone(),
            token_id: leg.token_id,
            side: leg.side,
            price: leg.price,
            size: leg.size,
            order_id: leg.order_id,
            ts: chrono::Utc::now(),
            mode,
            neg_risk,
            // In paper the sim stamps the real strategy; in live it comes as
            // None. The placeholder here (Arb) is NOT attribution: fill_sink
            // resolves it via the journal/order_index/gtd_orders/order_map chain
            // and, if none answers, QUARANTINES the fill instead of giving it to Arb.
            strategy: t.strategy.unwrap_or(StrategyId::Arb),
            clear_at,
            outcome_label,
            hedge_id: t.hedge_id,
            exec_id: Some(leg.exec_id),
            local_order_id: t.local_order_id.clone(),
        };

        if let Err(e) = fill_tx.send(fill) {
            tracing::error!(error = %e, "user_ws: error sending fill to fill_sink");
        }
    }
}

/// Fields of a maker leg broken down for a dead letter. Absent or unparseable
/// values become 0/empty and are DECLARED in `reason_extra` alongside the
/// trade totals — never silently replaced by the aggregate (the trade's
/// `asset_id` is the TAKER's token; assigning it to the leg would be a
/// fabricated attribution).
/// Returns `(token_id, price, size, reason_extra)`.
fn maker_leg_dl_fields(
    m: &MakerOrderRef,
    trade_price: Decimal,
    trade_size: Decimal,
) -> (String, Decimal, Decimal, String) {
    let mut missing: Vec<&str> = Vec::new();
    let size = m
        .matched_amount
        .as_deref()
        .and_then(|s| s.parse::<Decimal>().ok())
        .unwrap_or_else(|| {
            missing.push("matched_amount");
            Decimal::ZERO
        });
    let price = m
        .price
        .as_deref()
        .and_then(|p| p.parse::<Decimal>().ok())
        .unwrap_or_else(|| {
            missing.push("price");
            Decimal::ZERO
        });
    let token = m
        .asset_id
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            missing.push("asset_id");
            String::new()
        });
    let reason_extra = if missing.is_empty() {
        String::new()
    } else {
        format!(
            "; own leg with incomplete/unreadable fields ({}) — trade totals: size={} price={}",
            missing.join(","),
            trade_size,
            trade_price
        )
    };
    (token, price, size, reason_extra)
}

/// Common emission of the dead letter of ONE own maker leg (`order_id` already
/// verified against `known_order_ids`): ERROR log with the complete reason and
/// a send to the channel if it exists. `reason` already carries reason +
/// trade_id + missing fields + placeholder-side warning when applicable.
#[allow(clippy::too_many_arguments)]
fn emit_maker_leg_dl(
    t: &UserTrade,
    m: &MakerOrderRef,
    token_id: String,
    side: FillSide,
    price: Decimal,
    size: Decimal,
    reason: String,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
) {
    tracing::error!(
        order_id = %m.order_id,
        trade_id = %t.id,
        market   = %t.market,
        "user_ws: own maker leg NOT accounted — dead-letter \
         (uncertain evidence), no fabricated fill: {reason}"
    );
    if let Some(tx) = dead_letter_tx {
        let dl = DeadLetterFill {
            order_id: m.order_id.clone(),
            market_cid: t.market.clone(),
            token_id,
            side,
            price,
            size,
            status: t.status.clone(),
            role: t.trader_side.clone(),
            reason,
            ts: chrono::Utc::now(),
        };
        if let Err(e) = tx.send(dl) {
            tracing::error!(error = %e, "user_ws: dead-letter channel closed");
        }
    }
}

/// An OWN maker leg (verified by `known_order_ids`) whose payload does not
/// carry intact `matched_amount`/`price`/`asset_id` cannot become a `Fill` with
/// the trade totals — dead-letter with its own `order_id` and the missing
/// fields declared in `reason`.
/// The `side` is resolved with `dl_leg_side`: if it is not verifiable either,
/// the row carries a declared placeholder, never a silently invented side.
fn dead_letter_incomplete_maker_leg(
    t: &UserTrade,
    side: &FillSide,
    m: &MakerOrderRef,
    trade_price: Decimal,
    trade_size: Decimal,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
) {
    let (token_id, price, size, reason_extra) =
        maker_leg_dl_fields(m, trade_price, trade_size);
    let (dl_side, side_extra) = dl_leg_side(m, &t.asset_id, side);
    metrics::counter!("hrf_user_ws_maker_leg_incomplete_total").increment(1);
    emit_maker_leg_dl(
        t,
        m,
        token_id,
        dl_side,
        price,
        size,
        format!(
            "own maker leg with incomplete fields in trade (trade_id={}){reason_extra}{side_extra}",
            t.id
        ),
        dead_letter_tx,
    );
}

/// An OWN maker leg whose `side` is NOT resolvable (`resolve_leg_side` = None:
/// complement/other-token leg without `side` on the wire, or a `side` present
/// but unreadable) is not booked with a fabricated side: dead-letter with ITS
/// fields, reason 'side ambiguous' and the `FillSide` placeholder declared as
/// NOT verified in `reason` (the schema requires the field; the DL is evidence,
/// not an executable fill).
fn dead_letter_ambiguous_maker_leg(
    t: &UserTrade,
    deduced_side: &FillSide,
    m: &MakerOrderRef,
    trade_price: Decimal,
    trade_size: Decimal,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
) {
    let (token_id, price, size, reason_extra) =
        maker_leg_dl_fields(m, trade_price, trade_size);
    let (dl_side, side_extra) = dl_leg_side(m, &t.asset_id, deduced_side);
    metrics::counter!("hrf_user_ws_maker_leg_side_ambiguous_total").increment(1);
    emit_maker_leg_dl(
        t,
        m,
        token_id,
        dl_side,
        price,
        size,
        format!(
            "own maker leg with ambiguous side — side not resolvable in a \
             verifiable way (trade_id={}){reason_extra}{side_extra}",
            t.id
        ),
        dead_letter_tx,
    );
}

/// Dead letter of a live fill that cannot enter the accounting channel:
/// market without metadata, or a maker trade without a verifiable own leg.
/// `reason` describes the concrete cause and goes at the start of `reason`
/// together with the trade_id. Identity rules:
///
/// - Taker: `t.order_id` (`taker_order_id`) IS our order — a single DL.
/// - Maker with verifiable own legs (matched against `known_order_ids`):
///   ONE DL per leg, with ITS `order_id`/`asset_id`/`matched_amount`/
///   `price`/`side` — never `maker_orders.first()` (which could belong to
///   ANOTHER maker), nor the trade totals as if they were the leg's. A leg with
///   missing fields produces uncertain evidence with the reason in `reason`.
/// - Maker without a verifiable own leg: an UNATTRIBUTED DL — `order_id`
///   empty, no identity is invented nor is a foreign leg assigned — with the
///   trade and ALL its legs serialized in `reason` (the current schema of
///   `dead_letter_fills` has no per-leg fields).
#[allow(clippy::too_many_arguments)]
fn dead_letter_orphan_trade(
    t: &UserTrade,
    side: &FillSide,
    price: Decimal,
    size: Decimal,
    known_order_ids: &DashSet<String>,
    dead_letter_tx: &Option<mpsc::UnboundedSender<DeadLetterFill>>,
    reason: &str,
) {
    let base_reason = format!("{reason} (trade_id={})", t.id);

        /// A dead-letter row already broken down per leg.
    struct DlRow {
        order_id: String,
        token_id: String,
        side: FillSide,
        price: Decimal,
        size: Decimal,
        reason_extra: String,
    }

    let rows: Vec<DlRow> = if t.trader_side.eq_ignore_ascii_case("maker") {
        let ours: Vec<&MakerOrderRef> = t
            .maker_orders
            .iter()
            .filter(|m| known_order_ids.contains(&m.order_id))
            .collect();
        if ours.is_empty() {
            // No verifiable owner: evidence explicitly NOT attributed — keeps
            // the trade totals and ALL the legs serialized (including their
            // wire `side`) for manual reconciliation.
            let legs = serde_json::to_string(
                &t.maker_orders
                    .iter()
                    .map(|m| {
                        serde_json::json!({
                            "order_id": m.order_id,
                            "matched_amount": m.matched_amount,
                            "price": m.price,
                            "asset_id": m.asset_id,
                            "side": m.side,
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_else(|_| "[]".to_string());
            vec![DlRow {
                order_id: String::new(),
                token_id: t.asset_id.clone(),
                side: side.clone(),
                price,
                size,
                reason_extra: format!(
                    "; NO VERIFIABLE OWNER — order_id not attributed; \
                     size/price are the trade TOTALS; side = deduced by inverting \
                     the taker's side, NOT verified per leg; legs \
                     of the trade: {legs}"
                ),
            }]
        } else {
            ours.iter()
                .map(|&m| {
                    let (token_id, leg_price, leg_size, reason_extra) =
                        maker_leg_dl_fields(m, price, size);
                    // The leg's side if it is verifiable; otherwise a declared
                    // placeholder — never a silently invented side.
                    let (dl_side, side_extra) = dl_leg_side(m, &t.asset_id, side);
                    DlRow {
                        order_id: m.order_id.clone(),
                        token_id,
                        side: dl_side,
                        price: leg_price,
                        size: leg_size,
                        reason_extra: format!("{reason_extra}{side_extra}"),
                    }
                })
                .collect()
        }
    } else {
        // Taker: keeps our own order; explicitly empty if it did not come.
        vec![DlRow {
            order_id: t.order_id.clone().unwrap_or_default(),
            token_id: t.asset_id.clone(),
            side: side.clone(),
            price,
            size,
            reason_extra: if t.order_id.as_deref().unwrap_or("").is_empty() {
                "; taker_order_id missing from the payload".to_string()
            } else {
                String::new()
            },
        }]
    };

    for row in rows {
        if row.order_id.is_empty() {
            tracing::warn!(
                trade_id = %t.id,
                "user_ws: dead-letter without a verifiable order_id — can be reconciled \
                 only by trade_id/legs in `reason`"
            );
        }
        tracing::error!(
            order_id = %row.order_id,
            trade_id = %t.id,
            market   = %t.market,
            token_id = %row.token_id,
            side     = ?row.side,
            price    = %row.price,
            size     = %row.size,
            status   = %t.status,
            role     = %t.trader_side,
            "user_ws: LIVE FILL DISCARDED — {reason}. Sent to \
             dead_letter_fills for manual reconciliation"
        );
        if let Some(tx) = dead_letter_tx {
            let dl = DeadLetterFill {
                order_id: row.order_id,
                market_cid: t.market.clone(),
                token_id: row.token_id,
                side: row.side,
                price: row.price,
                size: row.size,
                status: t.status.clone(),
                role: t.trader_side.clone(),
                // The trade UUID goes in `reason` so as not to lose the execution's
                // identity (DeadLetterFill has no field of its own).
                reason: format!("{base_reason}{}", row.reason_extra),
                ts: chrono::Utc::now(),
            };
            if let Err(e) = tx.send(dl) {
                tracing::error!(error = %e, "user_ws: dead-letter channel closed — orphan fill only left in the log");
            }
        }
    }
}

/// Handles the `"order"` event of the user WS.
/// Processes cancellations → emits Fill::Rejected so that ARB/PM clear their state.
///
/// The original guard required `status == "CANCELLED"` (two L's) and never looked
/// at `type` — probably dead code: in SDK v2 a cancellation is identified by
/// `type == "CANCELLATION"` (`OrderMessageType::Cancellation`), and
/// `OrderStatusType` serializes `Canceled` as `"CANCELED"` (one L, American
/// spelling). Both spellings are accepted in `status` AND the `type`
/// discriminant — the real wire spelling still has to be verified with a
/// capture, so the tolerance is deliberate.
fn handle_order_cancel(item: &serde_json::Value, fill_tx: &mpsc::UnboundedSender<Fill>) {
    let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("");
    let msg_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let is_cancel = msg_type.eq_ignore_ascii_case("CANCELLATION")
        || status.eq_ignore_ascii_case("CANCELLED")
        || status.eq_ignore_ascii_case("CANCELED");
    if !is_cancel {
        return;
    }

    let market_cid = item.get("market").and_then(|v| v.as_str()).unwrap_or("");
    let token_id = item.get("asset_id").and_then(|v| v.as_str()).unwrap_or("");
    let order_id = item.get("id").and_then(|v| v.as_str()).map(str::to_string);

    if market_cid.is_empty() || token_id.is_empty() {
        return;
    }

    tracing::info!(order_id = ?order_id, market = %market_cid, status = %status, msg_type = %msg_type, "user_ws: order cancelled → Rejected signal");

    // Hardcoded `mode: Mode::Live` and `neg_risk: false` — correct TODAY because
    // `handle_order_cancel` is only called from the real user WS (always live),
    // and `FillSide::Rejected` needs no neg_risk (it computes no payout). If this
    // event is ever simulated in paper, or neg_risk starts to matter for
    // `Rejected`, review here first — not worth generalizing without a real
    // usage scenario.
    let fill = Fill {
        market_cid: market_cid.to_string(),
        token_id: token_id.to_string(),
        side: FillSide::Rejected,
        price: Decimal::ZERO,
        size: Decimal::ZERO,
        order_id,
        ts: chrono::Utc::now(),
        mode: Mode::Live,
        neg_risk: false,
        strategy: StrategyId::Arb,
        clear_at: 0,
        outcome_label: None,
        hedge_id: None,
        exec_id: None,
        local_order_id: None,
    };
    if let Err(e) = fill_tx.send(fill) {
        tracing::error!(error = %e, "user_ws: error sending Rejected to fill_sink");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(side: &str, role: &str) -> UserTrade {
        UserTrade {
            id: format!("t-{side}-{role}"),
            asset_id: "tokA".into(),
            market: "cid-t".into(),
            side: side.into(),
            size: "5".into(),
            price: "0.9".into(),
            status: "MATCHED".into(),
            trader_side: role.into(),
            hedge_id: None,
            strategy: None,
            real_clear_at: None,
            order_id: None,
            local_order_id: None,
            maker_orders: Vec::new(),
        }
    }

    /// The trade's side is the taker's — as maker, our side is the opposite.
    #[test]
    fn fill_maker_inverts_taker_side() {
        let registry = MarketRegistry::new();
        let known = DashSet::new();
        let (tx, mut rx) = mpsc::unbounded_channel();

        // Taker sells against our GTD bid → we BUY.
        handle_trade(
            trade("SELL", "MAKER"),
            &registry,
            &tx,
            &known,
            Mode::Paper,
            &None,
        );
        let fill = rx.try_recv().expect("fill maker");
        assert!(
            matches!(fill.side, FillSide::Buy),
            "maker with taker SELL = our Buy"
        );

        // As taker, the side is ours directly.
        handle_trade(
            trade("SELL", "TAKER"),
            &registry,
            &tx,
            &known,
            Mode::Paper,
            &None,
        );
        let fill = rx.try_recv().expect("fill taker");
        assert!(
            matches!(fill.side, FillSide::Sell),
            "taker SELL = our Sell"
        );
    }

    /// Market evicted from the ASSET registry (short-TTL working set,
    /// `evict_expired`) but still within the long-lived `clear_at_cache` —
    /// the fill must be processed with the correct `clear_at`/`neg_risk`,
    /// NOT discarded nor fall back to (false, 0, None), in live or in paper.
    #[test]
    fn fill_uses_clear_at_cache_when_active_registry_already_evicted_market() {
        let registry = MarketRegistry::new();
        // Simulates: the market was registered (same point as `markets`), then
        // `evict_expired` removed it from `markets` due to the short TTL, but
        // `clear_at_cache` (long TTL) still keeps it.
        registry.remember_clear_at("cid-evicted", true, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        // In live the maker branch requires an own leg that is verifiable —
        // `order_id` in `known_order_ids` and complete fields (side/token/price/
        // matched_amount). Without it the fill is not booked: dead-letter, which
        // is not the behavior under test here.
        known.insert("oid-ours".to_string());

        for mode in [Mode::Live, Mode::Paper] {
            let (tx, mut rx) = mpsc::unbounded_channel();
            let mut t = trade("SELL", "MAKER");
            t.market = "cid-evicted".into();
            t.maker_orders = vec![MakerOrderRef {
                order_id: "oid-ours".into(),
                matched_amount: Some("5".into()),
                price: Some("0.9".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            }];
            handle_trade(t, &registry, &tx, &known, mode, &None);
            let fill = rx
                .try_recv()
                .expect("the fill must not be discarded — it is in clear_at_cache");
            assert_eq!(
                fill.clear_at, 1_700_000_000,
                "must use the clear_at from the long-lived cache, not 0"
            );
        assert!(fill.neg_risk, "must use neg_risk from the long-lived cache");
            assert_eq!(
                fill.outcome_label,
                Some("YES".to_string()),
                "must resolve the YES label from the long-lived cache"
            );
        }
    }

    /// A fill for a `market_cid` that this process NEVER saw (neither in the
    /// asset registry nor in `clear_at_cache`) is still NOT re-sent to fill_sink
    /// in live (real protection against GTDs from a PREVIOUS session, before a
    /// crash/restart: both maps start empty in every new process), and falls
    /// back to the usual fallback in paper.
    ///
    /// This test used to assert that the live discard was "silent": THAT was the
    /// bug. A fill from the CURRENT session (market evicted by an eviction race,
    /// not from an earlier session) was lost without any trace, with real money
    /// already spent or collected. Discarding the POSITION itself is still
    /// required (without reliable neg_risk/clear_at/outcome_label there is no
    /// safe way to book it). What changed: besides the `ERROR` event (checked
    /// with a minimal `Subscriber`), the fill is now sent through the dead-letter
    /// channel to be persisted in `dead_letter_fills` and reconciled by hand
    /// against the CLOB.
    #[test]
    fn fill_of_never_seen_market_is_not_resent_in_live_and_logs_error_falls_back_in_paper() {
        let registry = MarketRegistry::new();
        let known = DashSet::new();

        let error_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sub = ErrorEventCounter(error_count.clone());
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();
        tracing::subscriber::with_default(sub, || {
            handle_trade(
                trade("SELL", "MAKER"),
                &registry,
                &tx,
                &known,
                Mode::Live,
                &Some(dl_tx),
            );
        });
        assert!(rx.try_recv().is_err(),
            "live: fill of a never-seen market is still not re-sent to fill_sink (the position is not booked)");
        assert_eq!(error_count.load(std::sync::atomic::Ordering::SeqCst), 1,
            "must log an ERROR event (previously a silent debug!) to leave a trace of the orphan fill");
        let dl = dl_rx
            .try_recv()
            .expect("the orphan fill must be sent to the dead-letter channel");
        assert_eq!(dl.market_cid, "cid-t");
        assert!(
            matches!(dl.side, FillSide::Buy),
            "maker with taker SELL = our Buy, same as in the normal Fill"
        );

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_trade(
            trade("SELL", "MAKER"),
            &registry,
            &tx,
            &known,
            Mode::Paper,
            &None,
        );
        let fill = rx
            .try_recv()
            .expect("paper: must not be discarded, falls back to the usual fallback");
        assert_eq!(
            fill.clear_at, 0,
            "paper: no real_clear_at in the trade, falls back to 0 as always"
        );
        assert!(!fill.neg_risk);
        assert_eq!(fill.outcome_label, None);
    }

    /// In a live fill where WE are maker, `t.order_id`
    /// (taker_order_id) is the COUNTERPARTY's order — ours is in
    /// `maker_orders`, matched against `known_order_ids`. `exec_id` is the id of
    /// the trade (the execution), distinct from the order_id.
    #[test]
    fn fill_maker_live_resolves_order_id_from_known_maker_orders() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        known.insert("oid-ours".to_string());
        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut t = trade("SELL", "MAKER");
        t.order_id = Some("oid-counterparty".into()); // taker_order_id — NOT ours
        t.maker_orders = vec![
            MakerOrderRef {
                order_id: "oid-other-maker".into(),
                matched_amount: None,
                price: None,
                asset_id: None,
                side: None,
            },
            MakerOrderRef {
                order_id: "oid-ours".into(),
                matched_amount: Some("4".into()),
                price: Some("0.62".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &None);

        let fill = rx.try_recv().expect("fill maker");
        assert_eq!(
            fill.order_id.as_deref(),
            Some("oid-ours"),
            "must pick the maker_order that matches known_order_ids, not the taker's"
        );
        assert_eq!(
            fill.exec_id.as_deref(),
            Some("t-SELL-MAKER:oid-ours"),
            "exec_id = trade_id:order_id — the leg, not the aggregated trade"
        );
        assert_eq!(
            fill.size,
            rust_decimal_macros::dec!(4),
            "size = the leg's matched_amount, not the trade total"
        );
        assert_eq!(fill.price, rust_decimal_macros::dec!(0.62));
        // The other maker's leg is NOT ours: no fill is emitted for it.
        assert!(rx.try_recv().is_err(), "only own legs are emitted");
    }

    /// An aggregated trade can match SEVERAL of our orders — each leg must emit
    /// its own Fill with a composite exec_id, so that the `UNIQUE(exec_id)`
    /// dedup does not discard the second order onwards.
    #[test]
    fn fill_maker_live_emits_one_leg_per_own_order() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        known.insert("oid-A".to_string());
        known.insert("oid-B".to_string());
        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut t = trade("SELL", "MAKER");
        t.maker_orders = vec![
            MakerOrderRef {
                order_id: "oid-A".into(),
                matched_amount: Some("2".into()),
                price: Some("0.60".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            },
            MakerOrderRef {
                order_id: "oid-B".into(),
                matched_amount: Some("5".into()),
                price: Some("0.58".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &None);

        let f1 = rx.try_recv().expect("leg A");
        let f2 = rx.try_recv().expect("leg B");
        assert_eq!(f1.exec_id.as_deref(), Some("t-SELL-MAKER:oid-A"));
        assert_eq!(f2.exec_id.as_deref(), Some("t-SELL-MAKER:oid-B"));
        assert_eq!(f1.size, rust_decimal_macros::dec!(2));
        assert_eq!(f2.size, rust_decimal_macros::dec!(5));
        assert_eq!(f1.price, rust_decimal_macros::dec!(0.60));
        assert_eq!(f2.price, rust_decimal_macros::dec!(0.58));
        assert!(rx.try_recv().is_err(), "exactly two legs");
    }

    /// KNOWN market but NO maker_order matches `known_order_ids` → NO `Fill` goes
    /// to the accounting channel. An aggregate with `order_id = None` used to be
    /// resolved by `order_map[cid:token]` in `fill_sink`, so a FOREIGN aggregate
    /// (trade totals, taker's token) could be applied to the ledger. Now: one
    /// unattributed dead-letter with ALL the legs in `reason`, and `rx` empty.
    #[test]
    fn fill_maker_live_without_own_leg_does_not_reach_accounting_channel() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new(); // empty — nothing matches
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();

        let mut t = trade("SELL", "MAKER");
        t.order_id = Some("oid-counterparty".into());
        t.maker_orders = vec![
            MakerOrderRef {
                order_id: "oid-foreign-1".into(),
                matched_amount: Some("2".into()),
                price: Some("0.60".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            },
            MakerOrderRef {
                order_id: "oid-foreign-2".into(),
                matched_amount: Some("3".into()),
                price: Some("0.55".into()),
                asset_id: Some("tok-no".into()),
                side: Some("SELL".into()),
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &Some(dl_tx));

        assert!(
            rx.try_recv().is_err(),
            "without a verifiable own leg, NO Fill reaches the accounting channel"
        );
        let d = dl_rx
            .try_recv()
            .expect("unattributed evidence via dead-letter");
        assert!(
            d.order_id.is_empty(),
            "no identity is invented nor taken from a foreign leg"
        );
        assert!(d.reason.contains("without a verifiable own leg"));
        assert!(d.reason.contains("NO VERIFIABLE OWNER"));
        assert!(d.reason.contains("oid-foreign-1"));
        assert!(d.reason.contains("oid-foreign-2"), "all the legs in reason");
        assert!(d.reason.contains("t-SELL-MAKER"), "trade_id in reason");
        // Trade totals kept as evidence (declared in reason).
        assert_eq!(d.size, rust_decimal_macros::dec!(5));
        assert_eq!(d.price, rust_decimal_macros::dec!(0.9));
        assert!(
            dl_rx.try_recv().is_err(),
            "a single unattributed evidence, not one per foreign leg"
        );
    }

    /// Label, side and token are PER LEG, not the taker's.
    /// Taker BUY over `tokA` (YES): one of our complement legs is BUY over
    /// `tok-no` (the wire carries its `side` — NOT the taker's inversion, which
    /// would give Sell), and another one of the same token with SELL. A third
    /// leg with a token foreign to the cid gets label `None` — no metadata is
    /// invented. That leg carries an explicit `side` from the wire: without it,
    /// today it would be dead-lettered as an ambiguous side (there is no
    /// fallback to inversion for a token other than the taker's).
    #[test]
    fn fill_maker_leg_own_label_side_and_token() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        known.insert("oid-no".to_string());
        known.insert("oid-yes".to_string());
        known.insert("oid-other".to_string());
        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut t = trade("BUY", "MAKER"); // taker buys YES (tokA)
        t.maker_orders = vec![
            // Complement: our BUY NO matched against the taker's BUY YES.
            MakerOrderRef {
                order_id: "oid-no".into(),
                matched_amount: Some("2".into()),
                price: Some("0.40".into()),
                asset_id: Some("tok-no".into()),
                side: Some("BUY".into()),
            },
            // Same token as the taker: SELL YES.
            MakerOrderRef {
                order_id: "oid-yes".into(),
                matched_amount: Some("5".into()),
                price: Some("0.58".into()),
                asset_id: Some("tokA".into()),
                side: Some("SELL".into()),
            },
            // Token that is neither yes nor no of the cid → label None; the wire
            // `side` is respected as is (without a `side` this leg would go to
            // dead-letter as ambiguous: the inversion does not apply to a token
            // other than the taker's).
            MakerOrderRef {
                order_id: "oid-other".into(),
                matched_amount: Some("1".into()),
                price: Some("0.50".into()),
                asset_id: Some("tok-x".into()),
                side: Some("SELL".into()),
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &None);

        let f_no = rx.try_recv().expect("leg NO");
        let f_yes = rx.try_recv().expect("leg YES");
        let f_x = rx.try_recv().expect("leg of unknown token");

        assert_eq!(f_no.token_id, "tok-no");
        assert_eq!(
            f_no.outcome_label.as_deref(),
            Some("NO"),
            "label of the leg's token, not the taker's (YES)"
        );
        assert!(
            matches!(f_no.side, FillSide::Buy),
            "wire side per leg — the taker's inversion would give Sell"
        );
        assert_eq!(f_no.size, rust_decimal_macros::dec!(2));
        assert_eq!(f_no.price, rust_decimal_macros::dec!(0.40));
        assert_eq!(f_no.exec_id.as_deref(), Some("t-BUY-MAKER:oid-no"));

        assert_eq!(f_yes.token_id, "tokA");
        assert_eq!(f_yes.outcome_label.as_deref(), Some("YES"));
        assert!(matches!(f_yes.side, FillSide::Sell));
        assert_eq!(f_yes.size, rust_decimal_macros::dec!(5));
        assert_eq!(f_yes.price, rust_decimal_macros::dec!(0.58));
        assert_eq!(f_yes.exec_id.as_deref(), Some("t-BUY-MAKER:oid-yes"));

        assert_eq!(f_x.token_id, "tok-x");
        assert_eq!(
            f_x.outcome_label, None,
            "token that matches neither yes nor no of the cid → label None, not invented"
        );
        assert!(
            matches!(f_x.side, FillSide::Sell),
            "wire side respected even though the token is foreign to the cid"
        );
        assert_eq!(f_x.exec_id.as_deref(), Some("t-BUY-MAKER:oid-other"));

        assert!(rx.try_recv().is_err(), "exactly three own legs");
    }

    /// Complement maker leg of ours (another token of the bracket) WITHOUT a
    /// `side` on the wire: the side cannot be deduced safely (the taker's
    /// inversion only holds for the same token). No Fill reaches the accounting
    /// channel, and the leg stays as a dead-letter of uncertain evidence with
    /// reason 'side ambiguous'. The complementary scenario WITH a valid `side` is
    /// covered by `fill_maker_leg_own_label_side_and_token` (BUY over `tok-no`).
    #[test]
    fn fill_maker_complement_without_side_is_not_counted_and_goes_to_dead_letter() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        known.insert("oid-no".to_string());
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();

        // Taker BUY over YES (tokA); our leg matched in tok-no without side.
        let mut t = trade("BUY", "MAKER");
        t.maker_orders = vec![MakerOrderRef {
            order_id: "oid-no".into(),
            matched_amount: Some("2".into()),
            price: Some("0.40".into()),
            asset_id: Some("tok-no".into()),
            side: None,
        }];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &Some(dl_tx));

        assert!(
            rx.try_recv().is_err(),
            "leg with ambiguous side is NOT booked — no fabricated side"
        );
        let d = dl_rx.try_recv().expect("dead-letter for ambiguous side");
        assert_eq!(d.order_id, "oid-no", "own order_id verified");
        assert_eq!(d.token_id, "tok-no");
        assert_eq!(d.size, rust_decimal_macros::dec!(2));
        assert_eq!(d.price, rust_decimal_macros::dec!(0.40));
        assert!(d.reason.contains("side ambiguous"), "explicit reason");
        assert!(
            d.reason.contains("PLACEHOLDER"),
            "the row's side is declared as a placeholder, NOT verified"
        );
        assert!(d.reason.contains("t-BUY-MAKER"), "trade_id in reason");
        assert!(dl_rx.try_recv().is_err(), "a single dead-letter");
    }

    /// Our maker leg of the SAME token as the taker, WITHOUT `side` on the wire:
    /// the inversion is verifiable (same token, opposite side) and is applied.
    #[test]
    fn fill_maker_same_token_without_side_infers_taker_inversion() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        known.insert("oid-yes".to_string());
        let (tx, mut rx) = mpsc::unbounded_channel();

        // Taker SELL over tokA → our leg in tokA = BUY by inversion.
        let mut t = trade("SELL", "MAKER");
        t.maker_orders = vec![MakerOrderRef {
            order_id: "oid-yes".into(),
            matched_amount: Some("5".into()),
            price: Some("0.58".into()),
            asset_id: Some("tokA".into()),
            side: None,
        }];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &None);

        let fill = rx.try_recv().expect("fill with side inferred by inversion");
        assert_eq!(fill.token_id, "tokA");
        assert!(
            matches!(fill.side, FillSide::Buy),
            "same token as the taker: SELL→BUY inversion is valid"
        );
        assert_eq!(fill.outcome_label.as_deref(), Some("YES"));
        assert!(rx.try_recv().is_err(), "a single leg");
    }

    /// `side` on the wire PRESENT but unreadable (neither BUY nor SELL) → uncertain
    /// even when the token matches the taker's: contradictory real data is not
    /// covered by the inversion fallback.
    #[test]
    fn fill_maker_unreadable_wire_side_does_not_fall_back() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        known.insert("oid-unreadable".to_string());
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();

        let mut t = trade("SELL", "MAKER");
        t.maker_orders = vec![MakerOrderRef {
            order_id: "oid-unreadable".into(),
            matched_amount: Some("5".into()),
            price: Some("0.58".into()),
            asset_id: Some("tokA".into()),
            side: Some("HOLD".into()), // present but invalid
        }];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &Some(dl_tx));

        assert!(
            rx.try_recv().is_err(),
            "invalid wire side → uncertain, NO fallback to the inversion"
        );
        let d = dl_rx.try_recv().expect("dead-letter for unreadable side");
        assert_eq!(d.order_id, "oid-unreadable");
        assert!(d.reason.contains("side ambiguous"));
    }

    /// Mixed trade: the HEALTHY leg (valid wire side) is emitted to the
    /// accounting channel, and the AMBIGUOUS one (complement without side) stays
    /// in dead-letter; one does not cover the other.
    #[test]
    fn fill_maker_mix_of_healthy_and_ambiguous_leg_emits_only_the_healthy() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        known.insert("oid-healthy".to_string());
        known.insert("oid-ambiguous".to_string());
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();

        let mut t = trade("BUY", "MAKER");
        t.maker_orders = vec![
            MakerOrderRef {
                order_id: "oid-healthy".into(),
                matched_amount: Some("5".into()),
                price: Some("0.58".into()),
                asset_id: Some("tokA".into()),
                side: Some("SELL".into()),
            },
            MakerOrderRef {
                order_id: "oid-ambiguous".into(),
                matched_amount: Some("2".into()),
                price: Some("0.40".into()),
                asset_id: Some("tok-no".into()),
                side: None,
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &Some(dl_tx));

        let fill = rx.try_recv().expect("the healthy leg does emit its fill");
        assert_eq!(fill.order_id.as_deref(), Some("oid-healthy"));
        assert!(matches!(fill.side, FillSide::Sell), "wire side respected");
        assert_eq!(fill.exec_id.as_deref(), Some("t-BUY-MAKER:oid-healthy"));
        assert!(rx.try_recv().is_err(), "the ambiguous leg does not fabricate a fill");

        let d = dl_rx.try_recv().expect("dead-letter of the ambiguous leg");
        assert_eq!(d.order_id, "oid-ambiguous");
        assert!(d.reason.contains("side ambiguous"));
        assert!(dl_rx.try_recv().is_err(), "a single dead-letter");
    }

    /// In a taker fill the order_id is `taker_order_id` directly.
    #[test]
    fn fill_taker_live_uses_taker_order_id() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut t = trade("BUY", "TAKER");
        t.order_id = Some("oid-ours-taker".into());
        handle_trade(t, &registry, &tx, &known, Mode::Live, &None);

        let fill = rx.try_recv().expect("fill taker");
        assert_eq!(fill.order_id.as_deref(), Some("oid-ours-taker"));
        assert_eq!(fill.exec_id.as_deref(), Some("t-BUY-TAKER"));
    }

    // ── User WS liveness watchdog ─────────────────────────────────────────────

    /// Local WS server: accepts one connection and, if `pong`, replies with the
    /// literal "PONG" to each "PING"; otherwise it reads and stays silent (zombie connection).
    async fn ws_server(pong: bool) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            let (mut w, mut r) = ws.split();
            while let Some(Ok(m)) = r.next().await {
                if pong && matches!(&m, Message::Text(t) if t == "PING") {
                    if w.send(Message::Text("PONG".into())).await.is_err() {
                        break;
                    }
                }
            }
        });
        format!("ws://{addr}")
    }

    /// A channel that stops delivering any frame (not even the PONG of the
    /// keepalives) is declared a zombie after `stale_after`, and the loop bails
    /// out with an error — before, `read.next()` hung forever.
    #[tokio::test]
    async fn zombie_connection_is_detected_and_returns_error() {
        let url = ws_server(false).await;
        let (ws, _) = connect_async(url).await.unwrap();
        let (mut w, mut r) = ws.split();
        let (fill_tx, _rx) = mpsc::unbounded_channel();
        let known = Arc::new(DashSet::new());
        let registry = MarketRegistry::new();
        let mut last_traffic_ms = 0i64;

        let res = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            session_loop(
                &mut w,
                &mut r,
                &registry,
                &fill_tx,
                &known,
                &None,
                std::time::Duration::from_millis(50),
                std::time::Duration::from_millis(300),
                &mut last_traffic_ms,
            ),
        )
        .await
        .expect("the watchdog must cut before the external timeout");
        let err = res.expect_err("a silent connection must return Err");
        assert!(err.to_string().contains("zombi"), "{err}");
        assert!(last_traffic_ms > 0, "the watermark is stamped even without frames");
    }

    /// The textual "PONG" from the venue IS a sign of life: a channel with no
    /// fills but a healthy keepalive does NOT reconnect (silence is normal on the
    /// user channel).
    #[tokio::test]
    async fn legitimate_silence_with_pong_does_not_reconnect() {
        let url = ws_server(true).await;
        let (ws, _) = connect_async(url).await.unwrap();
        let (mut w, mut r) = ws.split();
        let (fill_tx, _rx) = mpsc::unbounded_channel();
        let known = Arc::new(DashSet::new());
        let registry = MarketRegistry::new();
        let mut last_traffic_ms = 0i64;
        let t0 = chrono::Utc::now().timestamp_millis();

        // stale_after=300ms and the server answers PONG to each PING (50ms) —
        // the session must stay alive well beyond the threshold.
        let res = tokio::time::timeout(
            std::time::Duration::from_millis(1500),
            session_loop(
                &mut w,
                &mut r,
                &registry,
                &fill_tx,
                &known,
                &None,
                std::time::Duration::from_millis(50),
                std::time::Duration::from_millis(300),
                &mut last_traffic_ms,
            ),
        )
        .await;
        assert!(
            res.is_err(),
            "with live PONGs the session must not end on its own: {res:?}"
        );
        assert!(
            last_traffic_ms >= t0,
            "the watermark is stamped at the session start"
        );
        // PONGs refresh liveness but NOT the backfill anchor: without events the
        // anchor stays at the session startup.
        assert!(
            last_traffic_ms - t0 < 1500,
            "keepalives must not advance the anchor: {last_traffic_ms} vs t0={t0}"
        );
    }

    // ── Cancellations tolerant to spelling and type ───────────────────────────

    fn assert_rejected(item: serde_json::Value, expected: bool, ctx: &str) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_order_cancel(&item, &tx);
        match rx.try_recv() {
            Ok(fill) => {
                assert!(expected, "{ctx}: Rejected must not be emitted");
                assert!(
                    matches!(fill.side, FillSide::Rejected),
                    "{ctx}: the fill must be Rejected"
                );
                assert_eq!(fill.market_cid, "cid-x");
                assert_eq!(fill.token_id, "tok-x");
                assert_eq!(fill.order_id.as_deref(), Some("oid-1"));
            }
            Err(_) => assert!(!expected, "{ctx}: Rejected must be emitted"),
        }
    }

    /// `type:"CANCELLATION"` identifies the cancellation even if `status` is not
    /// "CANCELLED" — this is how the SDK v2 emits the event.
    #[test]
    fn cancel_by_type_cancellation_emits_rejected() {
        for status in ["CANCELED", "CANCELLED", "UNMATCHED", "MATCHED", ""] {
            let item = serde_json::json!({
                "event_type": "order",
                "type": "CANCELLATION",
                "status": status,
                "id": "oid-1",
                "market": "cid-x",
                "asset_id": "tok-x",
            });
            assert_rejected(item, true, &format!("type=CANCELLATION status={status}"));
        }
    }

    /// Both spellings of `status` (one and two L's) trigger the Rejected without
    /// needing `type`.
    #[test]
    fn cancel_by_status_both_spellings_emits_rejected() {
        for status in ["CANCELLED", "CANCELED"] {
            let item = serde_json::json!({
                "event_type": "order",
                "type": "UPDATE",
                "status": status,
                "id": "oid-1",
                "market": "cid-x",
                "asset_id": "tok-x",
            });
            assert_rejected(item, true, &format!("status={status}"));
        }
    }

    /// Order events that are NOT cancellations (placement, update, order fill)
    /// do not emit Rejected: the tolerance must not swallow other types.
    #[test]
    fn order_events_without_cancel_do_not_emit() {
        for (ty, status) in [
            ("PLACEMENT", "LIVE"),
            ("UPDATE", "MATCHED"),
            ("", "LIVE"),
            ("", ""),
        ] {
            let item = serde_json::json!({
                "event_type": "order",
                "type": ty,
                "status": status,
                "id": "oid-1",
                "market": "cid-x",
                "asset_id": "tok-x",
            });
            assert_rejected(item, false, &format!("type={ty} status={status}"));
        }
    }

    /// The dispatch routes `event_type:"order"` with `type:"CANCELLATION"` to the
    /// handler: the full pipeline emits the Rejected.
    #[test]
    fn dispatch_routes_cancel_by_type() {
        let registry = MarketRegistry::new();
        let known = DashSet::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let msg = r#"{"event_type":"order","type":"CANCELLATION","status":"CANCELED","id":"oid-9","market":"cid-x","asset_id":"tok-x"}"#;
        dispatch(msg, &registry, &tx, &known, Mode::Live, &None);
        let fill = rx.try_recv().expect("Rejected must be emitted");
        assert!(matches!(fill.side, FillSide::Rejected));
        assert_eq!(fill.order_id.as_deref(), Some("oid-9"));
    }

    // ── Dead-letter with the real order_id ────────────────────────────────────

    /// The dead letter stores the order_id of OUR order — the key for
    /// reconciliation against the CLOB — not the trade UUID (which stays in the
    /// `reason` field and in the log as `trade_id`).
    #[test]
    fn dead_letter_stores_order_id_not_trade_uuid() {
        let registry = MarketRegistry::new(); // cid-t not registered → dead-letter
        let known = DashSet::new();
        known.insert("oid-maker-ours".to_string());
        let (tx, _rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();
        let dl = Some(dl_tx);

        // Taker: the order_id is t.order_id (taker_order_id).
        let mut t = trade("BUY", "TAKER");
        t.order_id = Some("oid-taker".into());
        handle_trade(t, &registry, &tx, &known, Mode::Live, &dl);
        let d = dl_rx.try_recv().expect("dead-letter taker");
        assert_eq!(d.order_id, "oid-taker");
        assert!(
            d.reason.contains("t-BUY-TAKER"),
            "the trade UUID stays accessible in reason"
        );

        // Maker: our leg is the one matched against known_order_ids — NOT
        // t.order_id (which there is the counterparty's order), nor blindly the first.
        let mut t = trade("SELL", "MAKER");
        t.order_id = Some("oid-counterparty".into());
        t.maker_orders = vec![
            MakerOrderRef {
                order_id: "oid-another-maker".into(),
                matched_amount: None,
                price: None,
                asset_id: None,
                side: None,
            },
            MakerOrderRef {
                order_id: "oid-maker-ours".into(),
                matched_amount: Some("2".into()),
                price: Some("0.61".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &dl);
        let d = dl_rx.try_recv().expect("dead-letter maker");
        assert_eq!(d.order_id, "oid-maker-ours");
        // The leg carries ITS OWN data, not the trade totals
        // (trade: size=5 price=0.9).
        assert_eq!(d.size, rust_decimal_macros::dec!(2));
        assert_eq!(d.price, rust_decimal_macros::dec!(0.61));
        assert_eq!(d.token_id, "tokA");
        assert!(dl_rx.try_recv().is_err(), "a single dead-letter for the only own leg");
    }

    // ── Conservative anchor of the gap backfill ──────────────────────────────

    /// Single-event server: reads the auth, sends ONE event frame, answers "PONG"
    /// to the PINGs during `gap` (liveness alive during the silence of events),
    /// and ends the session — orderly Close (`clean_close`) or raw bytes with a
    /// reserved opcode, which the client's parser delivers as `Some(Err(_))` (a
    /// real protocol error on the stream).
    async fn server_event_then_terminal(
        gap: std::time::Duration,
        clean_close: bool,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            // No split: WebSocketStream is already Sink+Stream, and `get_mut()` gives
            // the raw TcpStream to inject the invalid frame of the Err scenario.
            let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            let _ = ws.next().await; // client auth
            let _ = ws
                .send(Message::Text(
                    r#"{"event_type":"order","type":"UPDATE","status":"LIVE","id":"oid-x","market":"cid-x","asset_id":"tok-x"}"#
                        .into(),
                ))
                .await;
            let deadline = std::time::Instant::now() + gap;
            while std::time::Instant::now() < deadline {
                if let Ok(Some(Ok(Message::Text(t)))) = tokio::time::timeout(
                    std::time::Duration::from_millis(40),
                    ws.next(),
                )
                .await
                {
                    if t == "PING" {
                        let _ = ws.send(Message::Text("PONG".into())).await;
                    }
                }
            }
            if clean_close {
                let _ = ws.send(Message::Close(None)).await;
            } else {
                use tokio::io::AsyncWriteExt;
                // FIN=1 + opcode 0x3 (reserved): invalid frame — the client
                // receives `Some(Err(Protocol(_)))`, not a clean Close.
                let _ = ws.get_mut().write_all(&[0x83, 0x00]).await;
            }
            // Leave the socket open for a while so the client processes the frame
            // (if we closed right away, an in-flight PING could return a write Err
            // before the terminal frame is read).
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        });
        format!("ws://{addr}")
    }

    /// A Close after a silence of events must NOT move the backfill anchor to the
    /// detection instant: it stays at the last valid event (~t0), even though the
    /// PONGs kept the session alive in between. stale_after is loose so that the
    /// watchdog does not interfere.
    #[tokio::test]
    async fn anchor_does_not_advance_with_close_after_silence() {
        let gap = std::time::Duration::from_millis(1200);
        let url = server_event_then_terminal(gap, true).await;
        let (ws, _) = connect_async(url).await.unwrap();
        let (mut w, mut r) = ws.split();
        let (fill_tx, _rx) = mpsc::unbounded_channel();
        let known = Arc::new(DashSet::new());
        let registry = MarketRegistry::new();
        let mut last_traffic_ms = 0i64;
        let t0 = chrono::Utc::now().timestamp_millis();

        let res = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            session_loop(
                &mut w,
                &mut r,
                &registry,
                &fill_tx,
                &known,
                &None,
                std::time::Duration::from_millis(50),
                std::time::Duration::from_secs(30),
                &mut last_traffic_ms,
            ),
        )
        .await
        .expect("a clean Close must end the session without hanging it");
        res.expect("orderly Close → Ok(())");
        assert!(last_traffic_ms >= t0, "anchor at least at startup");
        assert!(
            last_traffic_ms - t0 < gap.as_millis() as i64 / 2,
            "the anchor must stay at the last event (~t0={t0}), not at the \
             detection after {}ms of silence: {last_traffic_ms}",
            gap.as_millis()
        );
    }

    /// Same invariant when the terminal is a stream ERROR (invalid frame →
    /// `Some(Err(_))`), not an orderly Close: the corrupt frame does not advance
    /// the anchor either.
    #[tokio::test]
    async fn anchor_does_not_advance_with_stream_error_after_silence() {
        let gap = std::time::Duration::from_millis(1200);
        let url = server_event_then_terminal(gap, false).await;
        let (ws, _) = connect_async(url).await.unwrap();
        let (mut w, mut r) = ws.split();
        let (fill_tx, _rx) = mpsc::unbounded_channel();
        let known = Arc::new(DashSet::new());
        let registry = MarketRegistry::new();
        let mut last_traffic_ms = 0i64;
        let t0 = chrono::Utc::now().timestamp_millis();

        let res = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            session_loop(
                &mut w,
                &mut r,
                &registry,
                &fill_tx,
                &known,
                &None,
                std::time::Duration::from_millis(50),
                std::time::Duration::from_secs(30),
                &mut last_traffic_ms,
            ),
        )
        .await
        .expect("the protocol error must propagate without hanging the session");
        assert!(res.is_err(), "invalid frame → Err, not Ok");
        assert!(last_traffic_ms >= t0);
        assert!(
            last_traffic_ms - t0 < gap.as_millis() as i64 / 2,
            "a stream Err does not move the anchor: {last_traffic_ms} vs t0={t0}"
        );
    }

    /// A failed REST fetch is retried WHILE the session stays alive: it recovers
    /// the gap without a second connection and without stopping attention to the
    /// WS traffic (a live fill arrives during the retries).
    /// Server: one connection, a trade after the auth, PONG per PING, and a clean Close.
    #[tokio::test]
    async fn backfill_retries_during_session_without_reconnecting() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let connections = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        {
            let connections = connections.clone();
            tokio::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                connections.fetch_add(1, Ordering::SeqCst);
                let ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                let (mut w, mut r) = ws.split();
                let _ = r.next().await; // auth
                let trade = serde_json::json!({
                    "event_type": "trade", "id": "tr-ws-1",
                    "asset_id": "tok-bf", "market": "cid-bf",
                    "side": "BUY", "size": "2", "price": "0.5",
                    "status": "MATCHED", "trader_side": "taker",
                    "taker_order_id": "oid-taker-bf",
                });
                let _ = w.send(Message::Text(trade.to_string())).await;
                let mut pings = 0u32;
                while let Some(Ok(m)) = r.next().await {
                    if matches!(&m, Message::Text(t) if t == "PING") {
                        pings += 1;
                        if pings >= 12 {
                            let _ = w.send(Message::Close(None)).await;
                            // Socket left open for a while: the client must read the
                            // clean Close, not a reset in the middle of a PING.
                            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                            return;
                        }
                        if w.send(Message::Text("PONG".into())).await.is_err() {
                            return;
                        }
                    }
                }
            });
        }
        let url = format!("ws://{addr}");

        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-bf", false, 1_900_000_000, "tok-bf", "tok-no-bf");
        let known = Arc::new(DashSet::new());
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let creds = DerivedCreds {
            eoa_address: "0x0".into(),
            funder_address: None,
            api_key: "k".into(),
            api_secret: "s".into(),
            api_passphrase: "p".into(),
        };
        // Simulated REST fetch: the first attempt fails; the next ones return
        // empty items — the gap is covered within the SAME session.
        let calls = Arc::new(AtomicUsize::new(0));
        let fetch = {
            let calls = calls.clone();
            move |_after_ts: i64| {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    if n == 0 {
                        Err(anyhow::anyhow!("simulated REST failure"))
                    } else {
                        Ok(Vec::new())
                    }
                }
            }
        };
        let state = crate::state::StateStore::open_memory(Mode::Live).await.unwrap();
        let mut gap = Some(chrono::Utc::now().timestamp_millis() - 60_000);

        let end = run_session(
            &url,
            &creds,
            &registry,
            &fill_tx,
            &known,
            &None,
            &mut gap,
            Some(fetch),
            Some(state.as_ref()),
            BackfillRetryCfg {
                initial: std::time::Duration::from_millis(15),
                max: std::time::Duration::from_millis(30),
                fetch_timeout: std::time::Duration::from_secs(5),
            },
            5_000,
            std::time::Duration::from_millis(50),
            std::time::Duration::from_secs(30),
        )
        .await;

        end.result.expect("the session ended with a clean Close");
        assert!(
            calls.load(Ordering::SeqCst) >= 2,
            "the failed fetch was retried within the same session"
        );
        assert!(gap.is_none(), "the gap was released after the successful backfill");
        assert_eq!(
            connections.load(Ordering::SeqCst),
            1,
            "the recovery did not need a second connection"
        );
        let fill = fill_rx
            .try_recv()
            .expect("the WS trade was handled during the retries");
        assert_eq!(fill.order_id.as_deref(), Some("oid-taker-bf"));
        assert_eq!(fill.clear_at, 1_900_000_000);
    }

    // ── Backfill metadata: durable or dead-letter ────────────────────────────

    /// Backfill trade for a market absent from the registry, the clear_at_cache
    /// and durable evidence: NO `clear_at=match_ts` nor `clear_at=0` is invented.
    /// Explicit dead-letter + pending gap (`false`), and the evidence is not
    /// duplicated between supervisor retries.
    #[tokio::test]
    async fn backfill_unknown_market_does_not_invent_clear_at() {
        let registry = MarketRegistry::new();
        let state = crate::state::StateStore::open_memory(Mode::Live).await.unwrap();
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();
        let dl = Some(dl_tx);
        let fetch = |_ts: i64| async move {
            Ok(vec![crate::clob_sdk::ExecutedTrade {
                trade_id: "tr-x".into(),
                exec_id: "tr-x".into(),
                market_cid: "cid-unknown".into(),
                token_id: "tok-x".into(),
                is_buy: true,
                price: rust_decimal_macros::dec!(0.5),
                size: rust_decimal_macros::dec!(3),
                match_ts: 1_700_000_000,
                own_order_id: Some("oid-x".into()),
            }])
        };
        let mut dead_lettered = HashSet::new();

        let covered = backfill_gap_once(
            &fetch,
            state.as_ref(),
            &registry,
            &fill_tx,
            &dl,
            0,
            std::time::Duration::from_secs(5),
            &mut dead_lettered,
        )
        .await;
        assert!(!covered, "gap pending — unresolvable metadata");
        assert!(
            fill_rx.try_recv().is_err(),
            "no fill is fabricated with an invented clear_at"
        );
        let d = dl_rx.try_recv().expect("dead-letter of the unknown trade");
        assert_eq!(d.order_id, "oid-x");
        assert!(d.reason.contains("tr-x"), "the trade_id stays in reason");
        assert!(d.reason.contains("metadata"), "reason explains the cause");

        // Supervisor retry: same exec_id does not repeat the evidence.
        let covered = backfill_gap_once(
            &fetch,
            state.as_ref(),
            &registry,
            &fill_tx,
            &dl,
            0,
            std::time::Duration::from_secs(5),
            &mut dead_lettered,
        )
        .await;
        assert!(!covered, "still pending between retries");
        assert!(
            dl_rx.try_recv().is_err(),
            "the dead-letter is not duplicated between retries"
        );
    }

    /// A row in `markets_observed` ALONE no longer completes the metadata:
    /// the table stores `neg_risk`, but the only existing read
    /// (`markets_pending_official`) does not project it, and fabricating
    /// `neg_risk=false` would turn the absence of data into evidence. Without a
    /// verifiable `neg_risk`: pending gap, no fill, dead-letter.
    #[tokio::test]
    async fn backfill_markets_observed_without_verifiable_neg_risk_stays_pending() {
        let registry = MarketRegistry::new();
        let state = crate::state::StateStore::open_memory(Mode::Live).await.unwrap();
        state
            .upsert_market(
                "cid-v",
                "slug-v",
                "tok-yes-v",
                "tok-no-v",
                1_600_000_000,
                1_700_000_000, // already expired → markets_pending_official sees it
                false,
            )
            .await
            .unwrap();
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();
        let fetch = |_ts: i64| async move {
            Ok(vec![crate::clob_sdk::ExecutedTrade {
                trade_id: "tr-v".into(),
                exec_id: "tr-v".into(),
                market_cid: "cid-v".into(),
                token_id: "tok-yes-v".into(),
                is_buy: true,
                price: rust_decimal_macros::dec!(0.4),
                size: rust_decimal_macros::dec!(2),
                match_ts: 1_699_999_000,
                own_order_id: Some("oid-v".into()),
            }])
        };
        let mut dead_lettered = HashSet::new();
        let covered = backfill_gap_once(
            &fetch,
            state.as_ref(),
            &registry,
            &fill_tx,
            &Some(dl_tx),
            0,
            std::time::Duration::from_secs(5),
            &mut dead_lettered,
        )
        .await;
        assert!(
            !covered,
            "markets_observed alone is not enough: without a verifiable neg_risk the gap stays pending"
        );
        assert!(
            fill_rx.try_recv().is_err(),
            "no fill is fabricated with an invented neg_risk"
        );
        let d = dl_rx.try_recv().expect("dead-letter of the trade without complete metadata");
        assert_eq!(d.order_id, "oid-v");
        assert!(d.reason.contains("tr-v"), "the trade_id stays in reason");
        assert!(d.reason.contains("metadata"), "reason explains the cause");
    }

    /// Market outside registry/cache, NOT yet expired (outside the scope of
    /// `markets_pending_official`), but with a persisted position over the leg's
    /// token: its real `clear_at`/`neg_risk`/`outcome_label` are the durable evidence.
    #[tokio::test]
    async fn backfill_uses_durable_position_metadata() {
        let registry = MarketRegistry::new();
        let state = crate::state::StateStore::open_memory(Mode::Live).await.unwrap();
        // Discovery row of the market (realistic: the bracket existed when the
        // position was opened). clear_at in the future → outside
        // markets_pending_official, so the resolution still depends on the position.
        state
            .upsert_market(
                "cid-p", "slug-p", "tok-p", "tok-no-p",
                1_700_000_000, 1_900_000_000, true,
            )
            .await
            .unwrap();
        // Previous position over the token (same strategy the backfill resolves:
        // Arb, since there is no row in gtd_orders for oid-new).
        state
            .apply_fill(&Fill {
                market_cid: "cid-p".into(),
                token_id: "tok-p".into(),
                side: FillSide::Buy,
                price: rust_decimal_macros::dec!(0.5),
                size: rust_decimal_macros::dec!(4),
                order_id: Some("oid-prev".into()),
                ts: chrono::Utc::now(),
                mode: Mode::Live,
                neg_risk: true,
                strategy: StrategyId::Arb,
                clear_at: 1_900_000_000,
                outcome_label: Some("NO".into()),
                hedge_id: None,
                exec_id: Some("exec-prev".into()),
                local_order_id: None,
            })
            .await
            .unwrap();

        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let fetch = |_ts: i64| async move {
            Ok(vec![crate::clob_sdk::ExecutedTrade {
                trade_id: "tr-p".into(),
                exec_id: "tr-p".into(),
                market_cid: "cid-p".into(),
                token_id: "tok-p".into(),
                is_buy: true,
                price: rust_decimal_macros::dec!(0.55),
                size: rust_decimal_macros::dec!(1),
                match_ts: 1_800_000_000,
                own_order_id: Some("oid-new".into()),
            }])
        };
        let mut dead_lettered = HashSet::new();
        let covered = backfill_gap_once(
            &fetch,
            state.as_ref(),
            &registry,
            &fill_tx,
            &None,
            0,
            std::time::Duration::from_secs(5),
            &mut dead_lettered,
        )
        .await;
        assert!(covered);
        let fill = fill_rx
            .try_recv()
            .expect("fill with the durable metadata of the position");
        assert_eq!(fill.clear_at, 1_900_000_000, "clear_at of the position, not match_ts");
        assert!(fill.neg_risk);
        assert_eq!(fill.outcome_label.as_deref(), Some("NO"));
    }

    /// A "queued" send is not "covered": if `fill_tx.send` fails, the attempt
    /// returns `false` and the anchor stays pending.
    #[tokio::test]
    async fn backfill_failed_send_keeps_gap_pending() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-s", false, 1_900_000_000, "tok-s", "tok-no-s");
        let state = crate::state::StateStore::open_memory(Mode::Live).await.unwrap();
        let (fill_tx, fill_rx) = mpsc::unbounded_channel();
        drop(fill_rx); // channel closed → send() fails
        let fetch = |_ts: i64| async move {
            Ok(vec![crate::clob_sdk::ExecutedTrade {
                trade_id: "tr-s".into(),
                exec_id: "tr-s".into(),
                market_cid: "cid-s".into(),
                token_id: "tok-s".into(),
                is_buy: true,
                price: rust_decimal_macros::dec!(0.4),
                size: rust_decimal_macros::dec!(2),
                match_ts: 1_800_000_000,
                own_order_id: Some("oid-s".into()),
            }])
        };
        let mut dead_lettered = HashSet::new();
        let covered = backfill_gap_once(
            &fetch,
            state.as_ref(),
            &registry,
            &fill_tx,
            &None,
            0,
            std::time::Duration::from_secs(5),
            &mut dead_lettered,
        )
        .await;
        assert!(!covered, "failed send → gap pending, not covered");
    }

    // ── Dead-letter per own leg / unattributed ────────────────────────────────

    /// Maker with NO leg matched in `known_order_ids`: the dead-letter is NOT
    /// attributed to the first leg (it could belong to another maker): `order_id`
    /// empty, and the trade with ALL its legs serialized in `reason` for manual
    /// reconciliation.
    #[test]
    fn dead_letter_maker_without_own_leg_stays_unattributed() {
        let registry = MarketRegistry::new(); // cid-t not registered → dead-letter
        let known = DashSet::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();
        let dl = Some(dl_tx);

        let mut t = trade("SELL", "MAKER");
        t.maker_orders = vec![
            MakerOrderRef {
                order_id: "oid-foreign-1".into(),
                matched_amount: Some("2".into()),
                price: Some("0.60".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            },
            MakerOrderRef {
                order_id: "oid-foreign-2".into(),
                matched_amount: Some("3".into()),
                price: Some("0.55".into()),
                asset_id: Some("tok-no".into()),
                side: Some("SELL".into()),
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &dl);

        let d = dl_rx.try_recv().expect("an unattributed dead-letter");
        assert!(
            d.order_id.is_empty(),
            "no verifiable owner: nothing is invented nor taken from a foreign leg"
        );
        assert!(d.reason.contains("NO VERIFIABLE OWNER"));
        assert!(d.reason.contains("oid-foreign-1"));
        assert!(d.reason.contains("oid-foreign-2"), "all the legs stay in reason");
        // Trade totals kept as evidence (declared in reason).
        assert_eq!(d.size, rust_decimal_macros::dec!(5));
        assert_eq!(d.price, rust_decimal_macros::dec!(0.9));
        assert!(
            dl_rx.try_recv().is_err(),
            "a single unattributed evidence, not one per foreign leg"
        );
    }

    /// TWO own legs + one foreign: one dead-letter per own leg with ITS
    /// `order_id`/token/matched_amount/price; the foreign leg produces nothing.
    #[test]
    fn dead_letter_maker_one_row_per_own_leg() {
        let registry = MarketRegistry::new();
        let known = DashSet::new();
        known.insert("oid-A".to_string());
        known.insert("oid-B".to_string());
        let (tx, _rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();
        let dl = Some(dl_tx);

        let mut t = trade("SELL", "MAKER");
        t.maker_orders = vec![
            MakerOrderRef {
                order_id: "oid-A".into(),
                matched_amount: Some("2".into()),
                price: Some("0.60".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            },
            MakerOrderRef {
                order_id: "oid-foreign".into(),
                matched_amount: Some("9".into()),
                price: Some("0.50".into()),
                asset_id: Some("tok-no".into()),
                side: None,
            },
            MakerOrderRef {
                order_id: "oid-B".into(),
                matched_amount: Some("5".into()),
                price: Some("0.58".into()),
                asset_id: Some("tok-no".into()),
                side: Some("SELL".into()),
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &dl);

        let d1 = dl_rx.try_recv().expect("dead-letter leg A");
        let d2 = dl_rx.try_recv().expect("dead-letter leg B");
        assert_eq!(d1.order_id, "oid-A");
        assert_eq!(d1.size, rust_decimal_macros::dec!(2));
        assert_eq!(d1.price, rust_decimal_macros::dec!(0.60));
        assert_eq!(d1.token_id, "tokA");
        assert_eq!(d2.order_id, "oid-B");
        assert_eq!(d2.size, rust_decimal_macros::dec!(5));
        assert_eq!(d2.price, rust_decimal_macros::dec!(0.58));
        assert_eq!(d2.token_id, "tok-no");
        assert!(
            dl_rx.try_recv().is_err(),
            "the foreign leg produces no dead-letter"
        );
    }

    /// An own leg with incomplete fields produces UNCERTAIN evidence (own
    /// order_id verified, missing fields at 0 and declared in `reason`), never
    /// the trade totals silently.
    #[test]
    fn dead_letter_incomplete_leg_is_uncertain_evidence() {
        let registry = MarketRegistry::new(); // unknown market → DL
        let known = DashSet::new();
        known.insert("oid-incomplete".to_string());
        let (tx, _rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();
        let dl = Some(dl_tx);

        let mut t = trade("SELL", "MAKER");
        t.maker_orders = vec![MakerOrderRef {
            order_id: "oid-incomplete".into(),
            matched_amount: None, // the leg's size is missing
            price: Some("0.60".into()),
            asset_id: Some("tokA".into()),
            side: None,
        }];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &dl);

        let d = dl_rx.try_recv().expect("dead-letter of the incomplete leg");
        assert_eq!(d.order_id, "oid-incomplete", "own order_id verified");
        assert_eq!(
            d.size,
            Decimal::ZERO,
            "the missing size stays at 0 — not the trade total (5)"
        );
        assert_eq!(d.price, rust_decimal_macros::dec!(0.60));
        assert!(
            d.reason.contains("matched_amount"),
            "reason declares the missing field"
        );
        assert!(
            d.reason.contains("incomplete"),
            "reason marks the evidence as uncertain"
        );
    }

    /// In the FILLS branch (known market), an incomplete own leg is not
    /// fabricated from aggregates (uncertain dead-letter), and the HEALTHY own
    /// leg of the same trade does emit its fill with its own data.
    #[test]
    fn fill_maker_incomplete_leg_does_not_use_totals_and_does_not_mask_the_healthy() {
        let registry = MarketRegistry::new();
        registry.remember_clear_at("cid-t", false, 1_700_000_000, "tokA", "tok-no");
        let known = DashSet::new();
        known.insert("oid-inc".to_string());
        known.insert("oid-healthy".to_string());
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (dl_tx, mut dl_rx) = mpsc::unbounded_channel();

        let mut t = trade("SELL", "MAKER");
        t.maker_orders = vec![
            MakerOrderRef {
                order_id: "oid-inc".into(),
                matched_amount: None, // incomplete
                price: Some("0.60".into()),
                asset_id: Some("tokA".into()),
                side: None,
            },
            MakerOrderRef {
                order_id: "oid-healthy".into(),
                matched_amount: Some("3".into()),
                price: Some("0.58".into()),
                asset_id: Some("tokA".into()),
                side: Some("BUY".into()),
            },
        ];
        handle_trade(t, &registry, &tx, &known, Mode::Live, &Some(dl_tx));

        let d = dl_rx
            .try_recv()
            .expect("dead-letter of the incomplete leg");
        assert_eq!(d.order_id, "oid-inc");
        assert_eq!(d.size, Decimal::ZERO);
        assert!(d.reason.contains("matched_amount"));

        let fill = rx.try_recv().expect("the healthy leg does emit its fill");
        assert_eq!(fill.order_id.as_deref(), Some("oid-healthy"));
        assert_eq!(fill.size, rust_decimal_macros::dec!(3));
        assert_eq!(fill.price, rust_decimal_macros::dec!(0.58));
        assert_eq!(
            fill.exec_id.as_deref(),
            Some("t-SELL-MAKER:oid-healthy"),
            "exec_id per own leg (trade_id:order_id)"
        );
        assert!(rx.try_recv().is_err(), "the incomplete one does not fabricate a fill");
    }

    /// Minimal `Subscriber` (without depending on `tracing-subscriber`'s `Layer`,
    /// which needs more boilerplate) that only counts events of level `ERROR`:
    /// used to verify that the orphan fill leaves a trace.
    struct ErrorEventCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl tracing::Subscriber for ErrorEventCounter {
        fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            if *event.metadata().level() == tracing::Level::ERROR {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
    }
}
