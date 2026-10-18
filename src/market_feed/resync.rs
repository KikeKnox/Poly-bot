//! REST book resync — batched sweeper.
//!
//! Previous design: each trigger (WS reconnect, persistent crossed book)
//! queued a `GET /book` per token — ~2,800 tokens after every reconnect
//! (~1 every 26s, Polymarket closes aggressively), 10 concurrent HTTP requests with no
//! rate limit or backoff → 16k 429 responses and 4k 404s in 6h, with tokens
//! re-queued in a loop because `needs_snapshot` was never cleared.
//!
//! New design: the WS is the primary channel — on resubscribe, the server
//! pushes `book` snapshots of all live tokens and `handle.rs` clears
//! `needs_snapshot` on its own. REST is left for the STRAGGLERS: tokens that stay
//! pending past a grace window (dead/illiquid/desynchronized).
//! A periodic sweeper collects them and requests them in ONE `POST /books` per batch
//! (up to `batch_size` tokens per request), with a global rate limit and exponential
//! per-token backoff. In the common scenario — healthy WS — it costs 0 requests.

use crate::time::Instant;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

use super::handle::RawLevel;
use super::registry::{book_is_crossed, MarketRegistry, TokenState};
use crate::research::contract::{EventKind, EventSource, QualityFlags};
use crate::research::events::{EventInput, EventProducer, SourceContext};
use crate::types::{BookSnapshot, HalfBook, Level, TokenBook};

const CLOB_BASE: &str = "https://clob.polymarket.com";

/// Sweeper parameters (derived from `config.market_feed` in mod.rs).
#[derive(Debug, Clone)]
pub struct ResyncCfg {
    /// Sweep period over `needs_snapshot`.
    pub sweep_interval: Duration,
    /// Window given to the WS to repopulate a pending token before
    /// requesting it via REST — avoids duplicating the snapshot the server already
    /// pushes after resubscription.
    pub grace: Duration,
    /// Maximum tokens per `POST /books` (the endpoint accepts up to 500).
    pub batch_size: usize,
    /// Minimum interval between HTTP requests — process-wide rate limit.
    pub min_request_interval: Duration,
    /// Base cooldown after a failed attempt; grows exponentially per token.
    pub cooldown_base: Duration,
    /// Cooldown ceiling per token (dead tokens: one attempt every so often).
    pub cooldown_max: Duration,
}

impl Default for ResyncCfg {
    fn default() -> Self {
        Self {
            sweep_interval: Duration::from_millis(2_000),
            grace: Duration::from_millis(1_500),
            batch_size: 400,
            min_request_interval: Duration::from_millis(250),
            cooldown_base: Duration::from_secs(5),
            cooldown_max: Duration::from_secs(300),
        }
    }
}

/// Typed response of `POST /books`: one OrderBookSummary per requested token.
/// Tokens the CLOB doesn't know (dead) simply do NOT appear in the
/// array — they are treated as a failed attempt and the backoff self-throttles them.
#[derive(serde::Deserialize, serde::Serialize)]
pub(crate) struct BookSummary {
    #[serde(default, alias = "token_id")]
    pub(crate) asset_id: String,
    #[serde(default)]
    pub(crate) bids: Vec<RawLevel>,
    #[serde(default)]
    pub(crate) asks: Vec<RawLevel>,
}

/// Mutable sweeper state — per-token cooldowns and windows.
#[derive(Default)]
struct SweepState {
    /// First time the sweeper saw the token pending (grace anchor).
    pending_since: HashMap<String, Instant>,
    /// Instant from which the token may be retried.
    retry_at: HashMap<String, Instant>,
    /// Consecutive failed attempts — cooldown exponent.
    fail_streak: HashMap<String, u32>,
}

fn cooldown_for(streak: u32, cfg: &ResyncCfg) -> Duration {
    // base × 2^streak, exponent capped at 8 (×256) — with a 5s base it reaches the ceiling in ~7 failures.
    let shift = streak.min(8);
    let scaled = cfg
        .cooldown_base
        .checked_mul(1u32 << shift)
        .unwrap_or(cfg.cooldown_max);
    scaled.min(cfg.cooldown_max)
}

/// Collects the pending tokens eligible for REST: pending for at
/// least `grace` (so the WS wins the race) and with no active cooldown.
/// Those that show up pending for the first time are registered and NOT yet
/// eligible — their WS `book` may be in flight.
fn collect_eligible(
    registry: &Arc<MarketRegistry>,
    st: &mut SweepState,
    now: Instant,
    cfg: &ResyncCfg,
) -> Vec<String> {
    let mut eligible = Vec::new();
    for entry in registry.needs_snapshot.iter() {
        let token = entry.key();
        let Some(&since) = st.pending_since.get(token) else {
            // Newly pending — anchor it and let the grace elapse.
            st.pending_since.insert(token.clone(), now);
            continue;
        };
        if now.duration_since(since) < cfg.grace {
            continue;
        }
        if st.retry_at.get(token).is_some_and(|&r| now < r) {
            continue;
        }
        eligible.push((since, token.clone()));
    }
    // Oldest first; capped to the batch size.
    eligible.sort_by_key(|(since, _)| *since);
    eligible
        .into_iter()
        .take(cfg.batch_size)
        .map(|(_, t)| t)
        .collect()
}

/// Purge the state of tokens that are no longer pending (cleared by the WS or
/// evicted by the registry) — keeps the maps from growing unbounded.
fn prune_state(registry: &Arc<MarketRegistry>, st: &mut SweepState) {
    let alive = |t: &String| registry.needs_snapshot.contains(t);
    st.pending_since.retain(|t, _| alive(t));
    st.retry_at.retain(|t, _| alive(t));
    st.fail_streak.retain(|t, _| alive(t));
}

/// Per-token outcome of processing a received book.
pub(crate) enum BookOutcome {
    /// Written to the registry — clear flag and emit.
    Written,
    /// The WS wrote fresher data in parallel — clear the flag anyway.
    WsNewer,
    /// Empty book, crossed, or token absent from the response — stays pending,
    /// counts as a failed attempt (backoff).
    NotWritten,
}

/// Writes the book to `token_books` if it is fresher than whatever is there
/// (check+write atomic within the shard lock — the WS↔REST TOCTOU of the
/// earlier version is preserved). `bids`/`asks` arrive already normalized.
pub(crate) fn write_book_if_fresher(
    token_id: &str,
    bids: Vec<Level>,
    asks: Vec<Level>,
    fetch_started: chrono::DateTime<Utc>,
    registry: &Arc<MarketRegistry>,
) -> BookOutcome {
    if bids.is_empty() && asks.is_empty() {
        // Empty REST (illiquid/just-closed bracket) — if the WS
        // wrote something more recent the flag can be cleared; if not, it
        // stays pending for the next sweep.
        let ws_won = registry
            .token_books
            .get(token_id)
            .map(|s| s.updated_at >= fetch_started)
            .unwrap_or(false);
        return if ws_won {
            BookOutcome::WsNewer
        } else {
            BookOutcome::NotWritten
        };
    }

    use dashmap::mapref::entry::Entry;
    let received_at = Utc::now();
    let wrote = match registry.token_books.entry(token_id.to_string()) {
        Entry::Occupied(mut occ) => {
            if occ.get().updated_at < fetch_started {
                occ.insert(TokenState {
                    bids,
                    asks,
                    updated_at: received_at,
                });
                true
            } else {
                false
            }
        }
        Entry::Vacant(vac) => {
            vac.insert(TokenState {
                bids,
                asks,
                updated_at: received_at,
            });
            true
        }
    };
    if wrote {
        BookOutcome::Written
    } else {
        BookOutcome::WsNewer
    }
}

/// Emits the REST book to the recorder as it arrived — BEFORE any
/// decision (empty, crossed, empty token): the payload is the server's
/// response, not the result of accepting it. `origin_ts_ms=None` (POST
/// /books carries no server timestamp → mark NO_REMOTE_TS, honest);
/// `quality_extra=RESYNC` because this data comes from a resync after a
/// WS desync, not from the primary stream.
fn emit_rest_book(
    ctx: &SourceContext,
    registry: &Arc<MarketRegistry>,
    book: &BookSummary,
    recv_ts_ms: i64,
) {
    // The producer always exists — don't build the event
    // or serialize the book if the hub is inactive.
    if !ctx.producer.is_active() {
        return;
    }
    let token_id = if book.asset_id.is_empty() {
        None
    } else {
        Some(book.asset_id.clone())
    };
    let market_cid = token_id
        .as_ref()
        .and_then(|t| registry.token_to_cid.get(t).map(|c| c.clone()));
    let payload = serde_json::to_value(book).unwrap_or(serde_json::Value::Null);
    let mut e = EventInput::new(
        EventSource::PolymarketRest,
        EventKind::BookSnapshot,
        payload,
        recv_ts_ms,
    );
    e.token_id = token_id;
    e.market_cid = market_cid;
    e.stream = Some(ctx.stream.clone());
    e.generation = Some(ctx.generation);
    e.quality_extra = QualityFlags::RESYNC;
    ctx.producer.record(e);
}

/// Processes a batch response: writes each book, clears the resolved
/// flags and applies backoff to the unresolved ones. Separated from HTTP so it
/// can be tested with fabricated responses. A `rec` with producer
/// `EventProducer::sink()` keeps the no-recorder behavior.
#[allow(clippy::too_many_arguments)]
fn apply_books_response(
    requested: &[String],
    books: Vec<BookSummary>,
    fetch_started: chrono::DateTime<Utc>,
    registry: &Arc<MarketRegistry>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    st: &mut SweepState,
    cfg: &ResyncCfg,
    rec: &SourceContext,
) {
    let now = Instant::now();
    // `crate::time::now` (injectable clock) — same stamp domain as the
    // rest of the research pipeline (wall clock in production, tape time in replay).
    let recv_ts_ms = crate::time::now().timestamp_millis();
    let mut answered: std::collections::HashSet<String> = std::collections::HashSet::new();

    for book in books {
        emit_rest_book(rec, registry, &book, recv_ts_ms);
        let token_id = book.asset_id;
        if token_id.is_empty() {
            continue;
        }
        answered.insert(token_id.clone());

        let mut bids = parse_levels_from_raw(&book.bids);
        let mut asks = parse_levels_from_raw(&book.asks);
        bids.sort_unstable_by_key(|l| std::cmp::Reverse(l.price));
        asks.sort_unstable_by_key(|a| a.price);

        // A REST response can also be captured mid-trade and arrive
        // crossed. It is not written; the next sweep retries (the per-token
        // backoff avoids hammering it). Previously it was retried immediately one
        // more time — with the sweeper the wait is ~sweep_interval and costs a
        // fraction of a batched request.
        if book_is_crossed(&bids, &asks) {
            tracing::warn!(
                token = %token_id,
                "market_feed: resync returned crossed book — stays pending"
            );
            metrics::counter!("hrf_book_crossed_total", "source" => "resync").increment(1);
            bump_fail(&token_id, now, st, cfg);
            continue;
        }

        match write_book_if_fresher(&token_id, bids, asks, fetch_started, registry) {
            BookOutcome::Written => {
                registry.needs_snapshot.remove(&token_id);
                // the authoritative write also resets the "persistent crossing"
                // window — the pending describes the stale book.
                registry.crossed_pending.remove(&token_id);
                clear_token(&token_id, st);
                metrics::counter!("hrf_resync_tokens_written_total").increment(1);
                emit_snapshot_if_ready(&token_id, registry, book_tx);
            }
            BookOutcome::WsNewer => {
                registry.needs_snapshot.remove(&token_id);
                registry.crossed_pending.remove(&token_id);
                clear_token(&token_id, st);
            }
            BookOutcome::NotWritten => {
                bump_fail(&token_id, now, st, cfg);
            }
        }
    }

    // Requested tokens that did NOT come back in the response (dead/illiquid):
    // failed attempt — the exponential backoff self-throttles them up to
    // cooldown_max instead of hammering them on every reconnect (the ~4k
    // 404s/6h of the earlier version).
    for t in requested {
        if !answered.contains(t) {
            metrics::counter!("hrf_resync_tokens_missing_total").increment(1);
            bump_fail(t, now, st, cfg);
        }
    }
}

fn bump_fail(token_id: &str, now: Instant, st: &mut SweepState, cfg: &ResyncCfg) {
    let streak = st.fail_streak.entry(token_id.to_string()).or_insert(0);
    let wait = cooldown_for(*streak, cfg);
    *streak = (*streak + 1).min(64);
    st.retry_at.insert(token_id.to_string(), now + wait);
}

fn clear_token(token_id: &str, st: &mut SweepState) {
    st.pending_since.remove(token_id);
    st.retry_at.remove(token_id);
    st.fail_streak.remove(token_id);
}

pub(crate) async fn fetch_books(
    http: &reqwest::Client,
    token_ids: &[String],
) -> anyhow::Result<Vec<BookSummary>> {
    let body: Vec<serde_json::Value> = token_ids
        .iter()
        .map(|t| serde_json::json!({ "token_id": t }))
        .collect();
    let resp = http
        .post(format!("{CLOB_BASE}/books"))
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    Ok(resp.json().await?)
}

/// Periodic sweeper: the only consumer of `needs_snapshot` via REST.
/// There is no request queue — `needs_snapshot` itself IS the queue: the
/// triggers (`handle.rs`, `ws.rs`) just insert the token and the set deduplicates
/// bursts on its own. A genuine crossed book is repaired in ~one sweep
/// plus the grace instead of ~100ms — a deliberate trade-off: the token is already
/// blocked for the strategies in the meantime (fail-closed).
// `run_resync_task` (recorder-less wrapper) was left with no callers when
// `mod.rs` moved to the recorder variant — removed; a producer
// `EventProducer::sink()` reproduces the previous behavior.

/// REST sweeper of `needs_snapshot`; each book returned by the
/// `POST /books` is emitted to the research recorder with
/// `EventSource::PolymarketRest` + mark `RESYNC` — without this hook mid-session
/// resyncs (persistent crossing without reconnect) did not show up in
/// the dataset and `PolymarketRest` went unused. With the hub inactive,
/// `record` discards at no cost.
/// Wired in `market_feed/mod.rs`.
pub async fn run_resync_task_with_recorder(
    registry: Arc<MarketRegistry>,
    book_tx: broadcast::Sender<Arc<BookSnapshot>>,
    http: reqwest::Client,
    cfg: ResyncCfg,
    recorder: EventProducer,
) {
    // One generation per task — the REST sweeper has no connection epoch
    // of its own; the resync mark is carried by `quality_extra=RESYNC`.
    let ctx = SourceContext {
        producer: recorder,
        generation: crate::research::events::new_generation(),
        stream: "polymarket_rest".to_string(),
    };
    let mut tick = tokio::time::interval(cfg.sweep_interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut st = SweepState::default();
    let mut last_request = Instant::now() - cfg.min_request_interval;

    loop {
        tick.tick().await;

        let now = Instant::now();
        // Stale-backlog gauge: if it doesn't drain to ~0 between events there are
        // stuck tokens that never emit (the half_stale counter only measures suppressed
        // attempts, it doesn't tell churn from permanent stuckness).
        metrics::gauge!("hrf_market_feed_needs_snapshot").set(registry.needs_snapshot.len() as f64);
        let eligible = collect_eligible(&registry, &mut st, now, &cfg);
        if eligible.is_empty() {
            prune_state(&registry, &mut st);
            continue;
        }

        // Global rate limit: space out requests even if a job is pending.
        let delay = cfg
            .min_request_interval
            .saturating_sub(last_request.elapsed());
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }

        let fetch_started = Utc::now();
        last_request = Instant::now();
        metrics::counter!("hrf_resync_batch_requests_total").increment(1);
        metrics::histogram!("hrf_resync_batch_tokens").record(eligible.len() as f64);

        match fetch_books(&http, &eligible).await {
            Ok(books) => {
                apply_books_response(
                    &eligible,
                    books,
                    fetch_started,
                    &registry,
                    &book_tx,
                    &mut st,
                    &cfg,
                    &ctx,
                );
            }
            Err(e) => {
                // One warn per batch (not per token — log flooding was
                // part of the problem). The per-token backoff does the rest.
                tracing::warn!(
                    tokens = eligible.len(),
                    error = %e,
                    "market_feed: resync batch failed — per-token cooldown"
                );
                for t in &eligible {
                    bump_fail(t, Instant::now(), &mut st, &cfg);
                }
            }
        }
        prune_state(&registry, &mut st);
    }
}

pub(crate) fn parse_levels_from_raw(raw: &[RawLevel]) -> Vec<Level> {
    use std::str::FromStr;
    raw.iter()
        .filter_map(|item| {
            let price = rust_decimal::Decimal::from_str(&item.price).ok()?;
            let size = rust_decimal::Decimal::from_str(&item.size).ok()?;
            if size <= rust_decimal::Decimal::ZERO {
                return None;
            }
            Some(Level { price, size })
        })
        .collect()
}

pub(crate) fn emit_snapshot_if_ready(
    token_id: &str,
    registry: &Arc<MarketRegistry>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
) {
    let cid = match registry.token_to_cid.get(token_id) {
        Some(c) => c.clone(),
        None => return,
    };
    let entry = match registry.markets.get(&*cid) {
        Some(e) => e.clone(),
        None => return,
    };
    // Same guard as try_emit_snapshot (handle.rs) — don't emit if the
    // opposite side is still waiting for its snapshot after a WS gap (it would be
    // frozen from before the cut and would fabricate a phantom edge).
    if registry.needs_snapshot.contains(&entry.yes_token_id)
        || registry.needs_snapshot.contains(&entry.no_token_id)
    {
        metrics::counter!("hrf_market_feed_emit_half_stale_total").increment(1);
        return;
    }
    let yes_state = match registry.token_books.get(&entry.yes_token_id) {
        Some(s) => s.clone(),
        None => return,
    };
    let no_state = match registry.token_books.get(&entry.no_token_id) {
        Some(s) => s.clone(),
        None => return,
    };

    let tick_size = super::registry::effective_tick(
        &[
            yes_state.bids.first().map(|l| l.price),
            yes_state.asks.first().map(|l| l.price),
            no_state.bids.first().map(|l| l.price),
            no_state.asks.first().map(|l| l.price),
        ],
        entry.tick_size,
    );

    let snap = Arc::new(BookSnapshot {
        market_condition_id: cid.clone(),
        yes: TokenBook {
            token_id: entry.yes_token_id.clone(),
            bids: HalfBook {
                levels: yes_state.bids,
            },
            asks: HalfBook {
                levels: yes_state.asks,
            },
            flow_net: None,
            flow_total: None,
        },
        no: TokenBook {
            token_id: entry.no_token_id.clone(),
            bids: HalfBook {
                levels: no_state.bids,
            },
            asks: HalfBook {
                levels: no_state.asks,
            },
            flow_net: None,
            flow_total: None,
        },
        market_open_ts: entry.market_open_ts,
        clear_at: entry.clear_at,
        neg_risk: entry.neg_risk,
        slug: entry.slug.clone(),
        reference_price: entry.reference_price,
        fee_rate_bps: entry.fee_rate_bps,
        parabolic_fee_rate: entry.parabolic_fee_rate,
        tick_size: Some(tick_size),
        // The snapshot's freshness is that of its OLDEST side — a
        // book resynced via REST may be mixed with the side frozen
        // since a WS gap; stamping it with `now()` would fabricate a
        // phantom edge (a phantom YES ask → FOK with no counterparty).
        recv_instant: {
            let oldest = yes_state.updated_at.min(no_state.updated_at);
            // If `oldest` ends up in the FUTURE (clock jumped back —
            // NTP, VM suspend), `to_std()` returns Err and `unwrap_or_default()`
            // produced Duration::ZERO → the stalest book was stamped as
            // just arrived and skipped the strategies' `max_book_age`
            // gate. Fail-closed: negative age = maximally stale;
            // `checked_sub_or_floor` already degrades Duration::MAX to the floor.
            let age = (chrono::Utc::now() - oldest)
                .to_std()
                .unwrap_or(Duration::MAX);
            Some(checked_sub_or_floor(
                Instant::now(),
                age,
                earliest_instant(),
            ))
        },
    });

    if let Ok(receivers) = book_tx.send(snap) {
        metrics::gauge!("hrf_market_feed_broadcast_receivers").set(receivers as f64);
    }
}

/// Oldest instant representable by this process — captured
/// on the first call to `emit_snapshot_if_ready`. Safe floor for
/// `checked_sub_or_floor`.
static EARLIEST_INSTANT: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn earliest_instant() -> Instant {
    *EARLIEST_INSTANT.get_or_init(Instant::now)
}

/// `now - age`, or `floor` if `age` is not representable (clock
/// anomaly — NTP, system suspend…). The PREVIOUS fallback was
/// `Instant::now()` (would look freshly arrived and skip the
/// `max_book_age` gate — exactly what the freshness stamping avoids).
fn checked_sub_or_floor(now: Instant, age: Duration, floor: Instant) -> Instant {
    match now.checked_sub(age) {
        Some(t) => t,
        None => {
            metrics::counter!("hrf_resync_instant_underflow_total").increment(1);
            tracing::warn!(
                "market_feed: resync — book age not representable (possible \
                  clock anomaly); treated as maximally stale instead of freshly arrived"
            );
            floor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    /// Sink context for the `apply_books_response` tests — the tape hook
    /// discards at no cost (the producer is no longer an `Option`).
    fn ctx_sink() -> SourceContext {
        SourceContext {
            producer: EventProducer::sink(),
            generation: 0,
            stream: "polymarket_rest".into(),
        }
    }

    fn cfg_fast() -> ResyncCfg {
        ResyncCfg {
            sweep_interval: Duration::from_millis(10),
            grace: Duration::from_millis(50),
            batch_size: 3,
            min_request_interval: Duration::ZERO,
            cooldown_base: Duration::from_millis(100),
            cooldown_max: Duration::from_secs(10),
        }
    }

    fn book_summary(token: &str, bid: &str, ask: &str) -> BookSummary {
        BookSummary {
            asset_id: token.to_string(),
            bids: vec![RawLevel {
                price: bid.to_string(),
                size: "10".to_string(),
            }],
            asks: vec![RawLevel {
                price: ask.to_string(),
                size: "10".to_string(),
            }],
        }
    }

    #[test]
    fn cooldown_grows_exponentially_and_caps() {
        let cfg = ResyncCfg::default();
        assert_eq!(cooldown_for(0, &cfg), cfg.cooldown_base);
        assert_eq!(cooldown_for(1, &cfg), cfg.cooldown_base * 2);
        assert_eq!(cooldown_for(4, &cfg), cfg.cooldown_base * 16);
        assert_eq!(cooldown_for(64, &cfg), cfg.cooldown_max, "ceiling");
    }

    #[test]
    fn eligible_requires_grace_and_respects_retry() {
        let registry = Arc::new(MarketRegistry::new());
        let cfg = cfg_fast();
        let mut st = SweepState::default();
        registry.needs_snapshot.insert("tok-a".to_string());

        // First sighting: anchored, not eligible yet.
        let t0 = Instant::now();
        assert!(collect_eligible(&registry, &mut st, t0, &cfg).is_empty());

        // Before the grace: still not eligible.
        let t1 = t0 + cfg.grace - Duration::from_millis(10);
        assert!(collect_eligible(&registry, &mut st, t1, &cfg).is_empty());

        // Past the grace: eligible.
        let t2 = t0 + cfg.grace + Duration::from_millis(1);
        assert_eq!(
            collect_eligible(&registry, &mut st, t2, &cfg),
            vec!["tok-a"]
        );

        // With an active cooldown after a failure: not eligible even after the grace.
        bump_fail("tok-a", t2, &mut st, &cfg);
        assert!(collect_eligible(&registry, &mut st, t2, &cfg).is_empty());
        let t3 = t2 + cfg.cooldown_base + Duration::from_millis(1);
        assert_eq!(
            collect_eligible(&registry, &mut st, t3, &cfg),
            vec!["tok-a"]
        );
    }

    #[test]
    fn eligible_capped_at_batch_size_by_age() {
        let registry = Arc::new(MarketRegistry::new());
        let cfg = cfg_fast();
        let mut st = SweepState::default();
        let t0 = Instant::now();
        for i in 0..10 {
            let t = format!("tok-{i}");
            registry.needs_snapshot.insert(t.clone());
            st.pending_since.insert(t, t0 + Duration::from_millis(i));
        }
        let now = t0 + cfg.grace + Duration::from_secs(1);
        let eligibles = collect_eligible(&registry, &mut st, now, &cfg);
        assert_eq!(eligibles.len(), cfg.batch_size);
        assert_eq!(
            eligibles,
            vec!["tok-0", "tok-1", "tok-2"],
            "oldest first"
        );
    }

    #[test]
    fn batch_response_writes_clears_flag_and_emits() {
        let registry = Arc::new(MarketRegistry::new());
        let (book_tx, mut rx) = broadcast::channel(4);
        let cfg = cfg_fast();
        let mut st = SweepState::default();

        // Registry with a complete yes/no market.
        let cid = "cid-1";
        registry.markets.insert(
            cid.to_string(),
            crate::market_feed::registry::MarketEntry {
                slug: "test".into(),
                yes_token_id: "tok-yes".into(),
                no_token_id: "tok-no".into(),
                clear_at: 9_999_999_999,
                market_open_ts: 0,
                neg_risk: false,
                reference_price: None,
                fee_rate_bps: None,
                parabolic_fee_rate: None,
                tick_size: None,
                rewards_min_size: None,
                rewards_max_spread: None,
                rewards_daily_rate: None,
            },
        );
        registry
            .token_to_cid
            .insert("tok-yes".into(), cid.to_string());
        registry
            .token_to_cid
            .insert("tok-no".into(), cid.to_string());
        registry.needs_snapshot.insert("tok-yes".to_string());
        registry.needs_snapshot.insert("tok-no".to_string());

        let requested = vec!["tok-yes".to_string(), "tok-no".to_string()];
        let books = vec![
            book_summary("tok-yes", "0.48", "0.50"),
            book_summary("tok-no", "0.48", "0.50"),
        ];
        apply_books_response(
            &requested,
            books,
            Utc::now(),
            &registry,
            &book_tx,
            &mut st,
            &cfg,
            &ctx_sink(),
        );

        assert!(!registry.needs_snapshot.contains("tok-yes"));
        assert!(!registry.needs_snapshot.contains("tok-no"));
        let snap = rx.try_recv().expect("complete book → snapshot emitted");
        assert_eq!(snap.yes.bids.levels[0].price, dec!(0.48));
    }

    #[test]
    fn token_absent_in_response_backoff_and_stays_pending() {
        let registry = Arc::new(MarketRegistry::new());
        let (book_tx, _rx) = broadcast::channel(4);
        let cfg = cfg_fast();
        let mut st = SweepState::default();
        registry.needs_snapshot.insert("tok-dead".to_string());

        apply_books_response(
            &["tok-dead".to_string()],
            vec![],
            Utc::now(),
            &registry,
            &book_tx,
            &mut st,
            &cfg,
            &ctx_sink(),
        );

        assert!(registry.needs_snapshot.contains("tok-dead"));
        assert_eq!(*st.fail_streak.get("tok-dead").unwrap(), 1);
        assert!(st.retry_at.contains_key("tok-dead"), "backoff armed");

        // Second failure → larger cooldown.
        let r1 = *st.retry_at.get("tok-dead").unwrap();
        apply_books_response(
            &["tok-dead".to_string()],
            vec![],
            Utc::now(),
            &registry,
            &book_tx,
            &mut st,
            &cfg,
            &ctx_sink(),
        );
        assert!(*st.retry_at.get("tok-dead").unwrap() > r1);
    }

    #[test]
    fn crossed_response_does_not_write_and_stays_pending() {
        let registry = Arc::new(MarketRegistry::new());
        let (book_tx, _rx) = broadcast::channel(4);
        let cfg = cfg_fast();
        let mut st = SweepState::default();
        registry.needs_snapshot.insert("tok-x".to_string());

        apply_books_response(
            &["tok-x".to_string()],
            vec![book_summary("tok-x", "0.55", "0.50")], // bid > ask → crossed
            Utc::now(),
            &registry,
            &book_tx,
            &mut st,
            &cfg,
            &ctx_sink(),
        );

        assert!(registry.needs_snapshot.contains("tok-x"));
        assert!(
            registry.token_books.get("tok-x").is_none(),
            "crossed is not written"
        );
        assert_eq!(*st.fail_streak.get("tok-x").unwrap(), 1);
    }

    /// The CLOB returns the id as `asset_id` in OrderBookSummary; the `token_id`
    /// alias covers payload variants. Both must parse.
    #[test]
    fn book_summary_parses_asset_id_and_token_id_alias() {
        let with_asset: BookSummary = serde_json::from_str(
            r#"{"asset_id":"tok-1","bids":[{"price":"0.48","size":"10"}],"asks":[{"price":"0.50","size":"10"}]}"#,
        )
        .expect("asset_id must parse");
        assert_eq!(with_asset.asset_id, "tok-1");
        assert_eq!(with_asset.bids.len(), 1);

        let with_token: BookSummary =
            serde_json::from_str(r#"{"token_id":"tok-2","bids":[],"asks":[]}"#)
                .expect("token_id alias must parse");
        assert_eq!(with_token.asset_id, "tok-2");
    }

    /// If the WS wrote a fresher book than the start of the REST
    /// fetch, the resync must NOT overwrite it — but it must clear needs_snapshot (the
    /// token is already healthy via the WS).
    #[test]
    fn fresher_ws_wins_does_not_overwrite_and_clears_flag() {
        let registry = Arc::new(MarketRegistry::new());
        let (book_tx, _rx) = broadcast::channel(4);
        let cfg = cfg_fast();
        let mut st = SweepState::default();
        registry.needs_snapshot.insert("tok-x".to_string());

        // The WS writes a book NOW (fresher than the simulated REST fetch,
        // which started 60s ago).
        let fetch_started = Utc::now() - chrono::Duration::seconds(60);
        registry.token_books.insert(
            "tok-x".to_string(),
            TokenState {
                bids: vec![Level {
                    price: dec!(0.40),
                    size: dec!(5),
                }],
                asks: vec![Level {
                    price: dec!(0.60),
                    size: dec!(5),
                }],
                updated_at: Utc::now(),
            },
        );

        apply_books_response(
            &["tok-x".to_string()],
            vec![book_summary("tok-x", "0.10", "0.90")], // staler REST
            fetch_started,
            &registry,
            &book_tx,
            &mut st,
            &cfg,
            &ctx_sink(),
        );

        let book = registry.token_books.get("tok-x").unwrap();
        assert_eq!(book.bids[0].price, dec!(0.40), "the WS book is not overwritten");
        drop(book);
        assert!(
            !registry.needs_snapshot.contains("tok-x"),
            "WS won → token healthy → flag cleared"
        );
        assert!(
            !st.fail_streak.contains_key("tok-x"),
            "does not count as a failure"
        );
    }

    /// REST returns the token but with an empty book after filtering levels with
    /// size<=0 (illiquid/closed): nothing is written and it stays pending with
    /// backoff.
    #[test]
    fn empty_book_in_response_stays_pending() {
        let registry = Arc::new(MarketRegistry::new());
        let (book_tx, _rx) = broadcast::channel(4);
        let cfg = cfg_fast();
        let mut st = SweepState::default();
        registry.needs_snapshot.insert("tok-empty".to_string());

        let empty = BookSummary {
            asset_id: "tok-empty".to_string(),
            bids: vec![RawLevel {
                price: "0.50".into(),
                size: "0".into(),
            }],
            asks: vec![],
        };
        apply_books_response(
            &["tok-empty".to_string()],
            vec![empty],
            Utc::now(),
            &registry,
            &book_tx,
            &mut st,
            &cfg,
            &ctx_sink(),
        );

        assert!(registry.needs_snapshot.contains("tok-empty"));
        assert!(registry.token_books.get("tok-empty").is_none());
        assert_eq!(*st.fail_streak.get("tok-empty").unwrap(), 1);
    }

    #[test]
    fn prune_discards_tokens_no_longer_pending() {
        let registry = Arc::new(MarketRegistry::new());
        let mut st = SweepState::default();
        registry.needs_snapshot.insert("tok-a".to_string());
        st.pending_since.insert("tok-a".into(), Instant::now());
        st.retry_at.insert("tok-a".into(), Instant::now());
        st.fail_streak.insert("tok-a".into(), 3);
        st.pending_since.insert("tok-z".into(), Instant::now()); // never pending

        prune_state(&registry, &mut st);
        assert!(st.pending_since.contains_key("tok-a"));
        assert!(!st.pending_since.contains_key("tok-z"));
    }

    // ── checked_sub_or_floor ───────────────────────────────────────────────────

    #[test]
    fn checked_sub_or_floor_subtracts_normally_when_representable() {
        let now = Instant::now();
        let floor = now;
        let age = Duration::from_secs(5);
        let result = checked_sub_or_floor(now, age, floor);
        assert!(result <= now, "must be earlier than (or equal to) now");
    }

    #[test]
    fn checked_sub_or_floor_uses_floor_when_age_is_not_representable() {
        let now = Instant::now();
        let floor = now; // any valid anchor works for the test
                         // Duration::MAX forces checked_sub to underflow on any platform.
        let result = checked_sub_or_floor(now, Duration::MAX, floor);
        assert_eq!(result, floor,
            "an unrepresentable age must fall to the floor, NOT to `now` (which would look freshly arrived)");
    }
}
