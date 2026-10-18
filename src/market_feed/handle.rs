use crate::time::Instant;
use chrono::Utc;
use rust_decimal::Decimal;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::broadcast;

use super::registry::{book_is_crossed, crossed_escalation_wait, MarketRegistry, TokenState};
use crate::research::contract::{EventKind, EventSource};
use crate::research::events::{stable_hash, EventInput, SourceContext};
use crate::types::{BookSnapshot, HalfBook, Level, TokenBook};

// ── Typed structs for WS parsing ─────────────────────────────────────────────
// Before: serde_json::Value (one allocation per field of each JSON object).
// Now: Serde structs with skip_unknown_fields, so memory is only reserved for
// the fields the bot actually uses.

// Polymarket publishes the timestamp as a string ("1780198004663") instead of a number.
// This helper accepts both representations and normalises them to Option<i64>.
fn de_opt_i64_flexible<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Number(n) => Ok(n.as_i64()),
        serde_json::Value::String(s) => Ok(s.parse::<i64>().ok()),
        serde_json::Value::Null => Ok(None),
        _ => Ok(None),
    }
}

/// Price level in an order book (bid or ask).
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct RawLevel {
    pub price: String,
    pub size: String,
}

/// Entry of `price_changes` in a `price_change` message.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct RawPriceChange {
    pub asset_id: String,
    pub price: String,
    pub size: String,
    pub side: String,
}

/// Message from the Polymarket WS. The field that indicates the kind may be named
/// "type" (snapshots) or "event_type" (price_change), hence the alias.
///
/// Derives `Serialize` so the recorder stores the raw payload of the
/// message as it arrived; the `skip_serializing_if` attributes keep the JSON
/// compact and faithful (no noise from empty fields).
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct WsMessage {
    #[serde(
        rename = "type",
        alias = "event_type",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub msg_type: Option<String>,
    /// Token ID (only in book/initial_book messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    /// Timestamp in milliseconds published by Polymarket.
    /// The exchange sends it as a string ("1780198004663") or as a whole number.
    #[serde(
        default,
        deserialize_with = "de_opt_i64_flexible",
        skip_serializing_if = "Option::is_none"
    )]
    pub timestamp: Option<i64>,
    /// Bid levels (only in book/initial_book).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bids: Vec<RawLevel>,
    /// Ask levels (only in book/initial_book).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub asks: Vec<RawLevel>,
    /// Incremental price changes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub price_changes: Vec<RawPriceChange>,
    /// New minimum tick published in the `tick_size_change` event
    /// (string, e.g. "0.001"). Only present in that kind of message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_tick_size: Option<String>,
    /// Trade price (only in `last_trade_price`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub price: Option<String>,
    /// Side of the TAKER ("BUY"/"SELL") of the trade (only in `last_trade_price`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub side: Option<String>,
    /// Trade size in shares (only in `last_trade_price`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Catch-all: preserves the wire fields that the struct does not
    /// declare (`market`, `old_tick_size`, `fee_rate_bps`, `status`,
    /// `transaction_hash`, `order_id`…). Without it, `serde_json::to_value(msg)`
    /// in the recorder would lose those fields: the recorded payload would not be the
    /// ORIGINAL message, and a token evicted from the registry could no longer
    /// resolve its `market_cid` from the message's `market`.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ── Main dispatch ────────────────────────────────────────────────────────────

/// Wrapper without recorder: only used by this module's unit tests;
/// production always dispatches through [`dispatch_with_recorder`].
#[cfg_attr(not(test), allow(dead_code))]
pub fn dispatch(
    text: &str,
    registry: &Arc<MarketRegistry>,
    // O1: channel of Arc<BookSnapshot>: a single BookSnapshot in shared memory
    // shared by all receivers, without cloning the structure on each broadcast.
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    // Arrival instant of the frame, stamped by the WS reader (ws.rs) when it
    // enqueues it (not by the dispatcher when it processes it), so that `max_book_age`
    // measures the real age of the data even when the dispatch queue accumulates bursts.
    recv_instant: Instant,
    // Window of the trade tape (config `market_feed.tape_window_secs`).
    // 0 = disabled: no flow is accumulated and flow_net/flow_total are not filled.
    tape_window_secs: u64,
) {
    // F3.5: the producer is no longer an `Option`: an `EventProducer::sink()`
    // (hub never attached, so `is_active()` is false) reproduces "no recording".
    let ctx = SourceContext {
        producer: crate::research::events::EventProducer::sink(),
        generation: 0,
        stream: "poly-ws".into(),
    };
    dispatch_with_recorder(
        text,
        registry,
        book_tx,
        recv_instant,
        tape_window_secs,
        &ctx,
    );
}

/// T2: same as [`dispatch`], but also emits each received WS message to the
/// research recorder (`rec`) at the EDGE: what arrived, not what the bot
/// accepted (a crossed book or a discarded delta is also recorded;
/// the replay later reproduces the same filtering). A `SourceContext` with
/// producer `EventProducer::sink()` keeps the behaviour identical to
/// `dispatch`.
///
/// Wiring: `ws.rs` creates the source context per session
/// (`generation: new_generation()`, `stream: "poly-ws:shard-N"`) and calls
/// this function from the dispatcher.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_with_recorder(
    text: &str,
    registry: &Arc<MarketRegistry>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    recv_instant: Instant,
    tape_window_secs: u64,
    rec: &SourceContext,
) {
    // Discard plain-text keepalives (e.g. "PONG") that arrive as Message::Text.
    let text = text.trim();
    if text.is_empty() || (!text.starts_with('{') && !text.starts_with('[')) {
        return;
    }
    if text.starts_with('[') {
        match serde_json::from_str::<Vec<WsMessage>>(text) {
            Ok(msgs) => {
                for msg in &msgs {
                    handle_message(msg, registry, book_tx, recv_instant, tape_window_secs, rec);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "market_feed: unparseable JSON array from exchange");
                metrics::counter!("hrf_market_feed_unparseable_total").increment(1);
            }
        }
    } else {
        match serde_json::from_str::<WsMessage>(text) {
            Ok(msg) => handle_message(&msg, registry, book_tx, recv_instant, tape_window_secs, rec),
            Err(e) => {
                tracing::warn!(error = %e, "market_feed: unparseable JSON object from exchange");
                metrics::counter!("hrf_market_feed_unparseable_total").increment(1);
            }
        }
    }
}

/// T2: emits the input event (`EventInput`) for the WS message just dispatched.
///
/// - `recv_ts_ms` is rebuilt by subtracting the frame's age (the monotonic
///   `Instant` stamped by the reader on enqueue) from the current wall clock:
///   same technique as `tools::recorders::spot_stream::from_vol_snapshot`,
///   so the recorded `recv_ts`
///   is the one of ARRIVAL, not the one of processing.
/// - `avail_ts_ms` = wall clock at the moment of emission (≥ recv).
/// - `origin_ts_ms` = the `timestamp` published by Polymarket, or `None` if the
///   message does not carry it (never invented: marks NO_REMOTE_TS).
/// - `dedup_key`: only in `last_trade_price`. Hash of the stable fields
///   of the trade (token+price+side+size+venue timestamp), so that a
///   re-send after reconnection does not duplicate volume. NEVER recv/local_seq.
fn emit_ws_event(
    ctx: &SourceContext,
    registry: &Arc<MarketRegistry>,
    msg: &WsMessage,
    kind: EventKind,
    token_id: Option<&str>,
    dedup_key: Option<u64>,
    recv_instant: Instant,
) {
    // F3.2: since F3.1 the producer always exists; without an asset writer the
    // message would be serialised only for `record` to discard it.
    if !ctx.producer.is_active() {
        return;
    }
    // `crate::time::now` (injectable clock): the same stamp as the rest of the
    // research pipeline; in production it is wall time, in replay it is the tape.
    let now_ms = crate::time::now().timestamp_millis();
    let recv_ts_ms = now_ms.saturating_sub(recv_instant.elapsed().as_millis() as i64);
    let payload = serde_json::to_value(msg).unwrap_or(serde_json::Value::Null);
    let mut e = EventInput::new(EventSource::PolymarketWs, kind, payload, recv_ts_ms);
    e.avail_ts_ms = now_ms;
    e.origin_ts_ms = msg.timestamp;
    e.token_id = token_id.map(String::from);
    e.market_cid = token_id.and_then(|t| registry.token_to_cid.get(t).map(|c| c.clone()));
    e.stream = Some(ctx.stream.clone());
    e.generation = Some(ctx.generation);
    e.dedup_key = dedup_key;
    ctx.producer.record(e);
}

fn handle_message(
    msg: &WsMessage,
    registry: &Arc<MarketRegistry>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    recv_instant: Instant,
    tape_window_secs: u64,
    rec: &SourceContext,
) {
    // Exchange-to-bot latency
    if let Some(poly_ts_ms) = msg.timestamp {
        let latency_ms = Utc::now().timestamp_millis() - poly_ts_ms;
        if latency_ms >= 0 {
            metrics::histogram!("hrf_market_feed_exchange_to_bot_seconds")
                .record(latency_ms as f64 / 1000.0);
        }
    }

    let msg_type = msg.msg_type.as_deref().unwrap_or("");

    match msg_type {
        "book" | "initial_book" => {
            // T2: record the raw snapshot ON ARRIVAL, before ANY validation:
            // a book without `asset_id`, crossed, or discarded by resync also arrived
            // through the socket and is part of the dataset.
            emit_ws_event(
                rec,
                registry,
                msg,
                EventKind::BookSnapshot,
                msg.asset_id.as_deref().filter(|s| !s.is_empty()),
                None,
                recv_instant,
            );

            let asset_id = match msg.asset_id.as_deref().filter(|s| !s.is_empty()) {
                Some(id) => id,
                None => return,
            };

            // Dedup by price BEFORE sorting: if the exchange repeats
            // a price in the same snapshot (seen in practice), without this one of
            // the two levels stayed orphaned in the book forever (sort_unstable
            // does not remove duplicates). We keep the LAST occurrence (the more
            // authoritative one, according to the order of the raw exchange array).
            let mut bids = dedup_levels_by_price(parse_levels(&msg.bids));
            let mut asks = dedup_levels_by_price(parse_levels(&msg.asks));
            // CLOB returns bids ascending, asks ascending; normalise to bids desc.
            bids.sort_unstable_by_key(|l| std::cmp::Reverse(l.price));
            asks.sort_unstable_by_key(|a| a.price);

            // Any authoritative snapshot resets the "persistent crossing" window:
            // the `crossed_pending` state describes the PREVIOUS book, not the new
            // one. Without this, after each resync the token was left just one
            // crossed delta away from escalating again (was_already_crossed=true on
            // the first crossing), which produced a resync loop: ~35k WARN in 5.5 h
            // (~10× the baseline) on hot tokens.
            registry.crossed_pending.remove(asset_id);

            // Check for a crossed book before emitting, as for price_change.
            if book_is_crossed(&bids, &asks) {
                metrics::counter!("hrf_book_crossed_total").increment(1);
                tracing::warn!(
                    token = asset_id,
                    "market_feed: crossed initial snapshot — resync requested"
                );
                // Without this, a token with constant traffic always
                // loses the race between "the resync is enqueued" and "the next"
                // legitimate price_change arrives". That price_change updates
                // `updated_at` without needs_snapshot being set, so
                // `resync_token` (resync.rs) sees `updated_at >= fetch_started` and
                // concludes "the WS won" even though the book was still crossed. Marking
                // it here (as the persistent-crossing branch below already does) forces
                // intermediate price_change messages to be discarded until
                // the fresh resync snapshot arrives. Inserting into
                // `needs_snapshot` IS the request: the resync.rs sweeper
                // collects it in batches on the next sweep.
                registry.needs_snapshot.insert(asset_id.to_string());
                return;
            }

            registry.token_books.insert(
                asset_id.to_string(),
                TokenState {
                    bids,
                    asks,
                    updated_at: Utc::now(),
                },
            );
            registry.needs_snapshot.remove(asset_id);

            let pre_emit = Instant::now();
            try_emit_snapshot(
                asset_id,
                registry,
                book_tx,
                recv_instant,
                true,
                tape_window_secs,
            );
            metrics::histogram!("hrf_market_feed_book_to_emit_seconds")
                .record(pre_emit.elapsed().as_secs_f64());
        }

        "price_change" => {
            // The raw delta is recorded on arrival, even for a message with an empty
            // `price_changes` or whose entries are later discarded
            // (needs_snapshot / crossed book). Token_id at message level:
            // the wire field if it exists; otherwise, if ALL entries belong to the
            // same asset (the common case), that one. If the message mixes tokens it is
            // left as None (the per-entry asset_id values live in the payload, which is
            // the complete message thanks to `extra`).
            let token_id = msg
                .asset_id
                .as_deref()
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    let mut ids = msg
                        .price_changes
                        .iter()
                        .map(|c| c.asset_id.as_str())
                        .filter(|s| !s.is_empty());
                    let first = ids.next()?;
                    ids.all(|s| s == first).then_some(first)
                });
            emit_ws_event(
                rec,
                registry,
                msg,
                EventKind::BookDelta,
                token_id,
                None,
                recv_instant,
            );

            if msg.price_changes.is_empty() {
                return;
            }

            let mut updated_tokens: HashSet<&str> = HashSet::new();

            // Apply ALL entries of the message before checking for a crossing.
            // A trade that moves the top across the spread arrives as several
            // entries of the same token ("new ask" + "remove bid"); checking
            // after each entry saw a transient crossing, set needs_snapshot
            // and discarded the entries remaining (the ones that uncrossed it), causing
            // ~1,700 resyncs in 3 h with degraded data windows.
            for entry in &msg.price_changes {
                let asset_id = entry.asset_id.as_str();
                if asset_id.is_empty() {
                    continue;
                }
                if registry.needs_snapshot.contains(asset_id) {
                    tracing::debug!(
                        token = asset_id,
                        "market_feed: discarding price_change — awaiting snapshot"
                    );
                    continue;
                }
                if let Some(mut state) = registry.token_books.get_mut(asset_id) {
                    apply_price_change_entry(&mut state, entry);
                    state.updated_at = Utc::now();
                    updated_tokens.insert(asset_id);
                } else {
                    // Delta over a token without `TokenState` (its initial `book`
                    // never landed, e.g. the WS array was discarded
                    // whole because ONE element failed to parse). Previously it was dropped
                    // silently and the token stayed mute until the next shard
                    // reconnection; now it is queued in `needs_snapshot`
                    // so the REST sweeper repairs it.
                    // Only if the token is registered: a delta from a foreign token
                    // (residual subscription of an evicted market) must not
                    // create a repair job for a dead one.
                    metrics::counter!("hrf_market_feed_delta_without_state_total").increment(1);
                    if registry.token_to_cid.contains_key(asset_id) {
                        registry.needs_snapshot.insert(asset_id.to_string());
                        tracing::warn!(
                            token = asset_id,
                            "market_feed: price_change without prior token state, needs_snapshot to repair via REST"
                        );
                    } else {
                        tracing::debug!(
                            token = asset_id,
                            "market_feed: price_change for an unregistered token, ignored"
                        );
                    }
                }
            }

            for asset_id in updated_tokens {
                // A late or out-of-order delta can leave the book crossed
                // (best bid ≥ best ask) at the END of the message.
                let crossed = registry
                    .token_books
                    .get(asset_id)
                    .map(|state| book_is_crossed(&state.bids, &state.asks));
                // Shard reference released here, before touching other maps
                match crossed {
                    Some(true) => {
                        // Most of these crossings are transient
                        // — the same trade arrives as "remove from the ask" + "add to the"
                        // bid in separate WS messages, and the NEXT price_change
                        // of this same token corrects it on its own. Previously it was escalated to a
                        // REST resync + WARN on the first tick, and in the meantime the legitimate deltas
                        // that would have corrected the crossing were discarded (needs_snapshot drops them).
                        // This produced 22k
                        // resyncs in 18 h during calibration.
                        //
                        // Instead of guessing how long that delta takes with a timeout,
                        // we wait for the real signal: if the next price_change of
                        // THIS token (however late it arrives) also leaves the book
                        // crossed, then it is a genuine desynchronisation. If no other delta
                        // ever arrives, the staleness guard of each strategy
                        // (`ctx.max_book_age`) already prevents acting on the last valid
                        // snapshot, so no timeout is needed here either.
                        let was_already_crossed =
                            !registry.crossed_pending.insert(asset_id.to_string());
                        if was_already_crossed {
                            // Per-token backoff: if the venue re-crosses the book a few deltas after each
                            // fresh snapshot (markets in resolution, or extreme top-of-book churn; the REST
                            // confirms books that are not crossed or are empty: the crossing IS the market,
                            // not a desynchronisation), so escalating on every cycle only
                            // produces WARN spam and REST churn without repairing anything.
                            // While the token is in cooldown, the escalation is suppressed:
                            // `crossed_pending` stays set, so the next crossed delta
                            // re-evaluates the book.
                            // The deltas keep being applied, so the stream can heal itself
                            // instead of being discarded by needs_snapshot.
                            // Escalation is limited by the per-token backoff below
                            // (`crossed_escalation`), so it does not repeat on every delta.
                            let suppressed = registry
                                .crossed_escalation
                                .get(asset_id)
                                .and_then(|e| e.retry_at)
                                .is_some_and(|t| Instant::now() < t);
                            if suppressed {
                                metrics::counter!("hrf_book_crossed_suppressed_total").increment(1);
                                continue;
                            }
                            registry.crossed_pending.remove(asset_id);
                            registry.needs_snapshot.insert(asset_id.to_string());
                            {
                                let mut esc = registry
                                    .crossed_escalation
                                    .entry(asset_id.to_string())
                                    .or_default();
                                esc.retry_at =
                                    Some(Instant::now() + crossed_escalation_wait(esc.streak));
                                esc.streak = (esc.streak + 1).min(16);
                            }
                            metrics::counter!("hrf_book_crossed_total").increment(1);
                            tracing::warn!(
                                token = asset_id,
                                "market_feed: persistent crossed book after price_change, resync requested"
                            );
                        }
                        continue;
                    }
                    Some(false) => {
                        registry.crossed_pending.remove(asset_id);
                        // The book un-crossed on its own: it forgives ONE STEP
                        // of the backoff, not a full reset. Markets in resolution oscillate
                        // between crossed and healthy because of venue churn, and a reset to zero
                        // would leave the backoff at the floor (5 s) forever (~400 escalations on one
                        // token in 11 min). With decay, the oscillation keeps the streak high and
                        // the escalation converges to ~1/300 s per token.
                        // A book that stays healthy is fully pardoned after
                        // `streak` consecutive healthy deltas.
                        // The streak is decremented by one per healthy delta.
                        let mut forgive = false;
                        if let Some(mut esc) = registry.crossed_escalation.get_mut(asset_id) {
                            esc.streak = esc.streak.saturating_sub(1);
                            forgive =
                                esc.streak == 0 && esc.retry_at.is_none_or(|t| Instant::now() >= t);
                        }
                        if forgive {
                            registry.crossed_escalation.remove(asset_id);
                        }
                    }
                    None => continue,
                }

                let pre_emit = Instant::now();
                try_emit_snapshot(
                    asset_id,
                    registry,
                    book_tx,
                    recv_instant,
                    false,
                    tape_window_secs,
                );
                metrics::histogram!("hrf_market_feed_book_to_emit_seconds")
                    .record(pre_emit.elapsed().as_secs_f64());
            }
        }

        // Price crossed 0.96/0.04 and the market's minimum tick changed.
        // The registry is updated and the snapshot is re-emitted so that
        // the strategies see the new tick without waiting for the next book event.
        "tick_size_change" => {
            // Book rule change (tick): recorded BEFORE validating asset_id/tick
            // (an unreadable tick_size_change also arrived), as a BookDelta:
            // there is no dedicated variant in `EventKind`, and `MarketMeta`
            // requires a typed payload; here the payload is
            // the complete raw msg (including `market`/`old_tick_size` via
            // `extra`).
            emit_ws_event(
                rec,
                registry,
                msg,
                EventKind::BookDelta,
                msg.asset_id.as_deref().filter(|s| !s.is_empty()),
                None,
                recv_instant,
            );
            let Some(asset_id) = msg.asset_id.as_deref().filter(|s| !s.is_empty()) else {
                return;
            };
            let Some(new_tick_str) = msg.new_tick_size.as_deref() else {
                return;
            };
            let new_tick = match Decimal::from_str(new_tick_str) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(
                        token = asset_id,
                        new_tick = new_tick_str,
                        error = %e,
                        "market_feed: tick_size_change with unreadable tick, ignored"
                    );
                    return;
                }
            };
            // A tick ≤ 0 ("0", "-0.01"…) is REJECTED here:
            // `Decimal::from_str` accepts it, and `effective_tick`/`min()` would
            // propagate it to `BookSnapshot::tick()`; rl_predictor divides by the
            // tick without a guard, and a 0 makes it panic. An invalid tick never
            // enters the registry.
            if new_tick <= Decimal::ZERO {
                metrics::counter!("hrf_market_feed_tick_invalid_total").increment(1);
                tracing::warn!(
                    token = asset_id,
                    new_tick = new_tick_str,
                    "market_feed: tick_size_change with non-positive tick, ignored"
                );
                return;
            }
            let Some(cid) = registry.token_to_cid.get(asset_id).map(|c| c.clone()) else {
                return;
            };
            if let Some(mut entry) = registry.markets.get_mut(&cid) {
                entry.tick_size = Some(new_tick);
            }
            try_emit_snapshot(
                asset_id,
                registry,
                book_tx,
                recv_instant,
                true,
                tape_window_secs,
            );
        }

        // Executed trade (aggressor = TAKER). It only accumulates signed flow
        // per token; it does NOT emit a snapshot: the twin `price_change` of this same
        // trade (it always travels with it and consumes a book level) already emits,
        // and the updated flow travels in that emission.
        "last_trade_price" => {
            // The trade tape is recorded at the edge: before the cut for a missing
            // asset_id, tape_window_secs=0 and the guard for an unregistered token:
            // the trade DID arrive even if the strategy does not accumulate it.
            // Dedup by the venue's stable fields (a re-send after reconnection
            // does not duplicate volume); the wire carries no trade id of its own
            // — caveat (minor): two DISTINCT trades with identical
            // token+price+side+size+timestamp (same venue ms) collapse into one;
            // an ambiguity inherent in the message, documented: better than duplicating a re-send.
            // (the other choice would be to duplicate a re-send).
            emit_ws_event(
                rec,
                registry,
                msg,
                EventKind::Trade,
                msg.asset_id.as_deref().filter(|s| !s.is_empty()),
                Some(stable_hash((
                    "poly-trade",
                    &msg.asset_id,
                    &msg.price,
                    &msg.side,
                    &msg.size,
                    msg.timestamp,
                ))),
                recv_instant,
            );
            let Some(asset_id) = msg.asset_id.as_deref().filter(|s| !s.is_empty()) else {
                return;
            };
            if tape_window_secs == 0 {
                return;
            } // 0 = tape disabled, nothing is accumulated
              // Polymarket's WS subscription is cumulative per
              // connection (see ws.rs:~192): late trades of a bracket already
              // evicted (evict_expired) keep arriving, and `record_trade` would
              // re-create the entry in `token_flow` via `.or_default()`, an entry
              // that is never cleaned again (evict_expired only removes tokens of
              // markets present at that moment), an unbounded memory leak over
              // uptime. The trade is discarded if the token is no longer registered.
            if !registry.token_to_cid.contains_key(asset_id) {
                return;
            }
            let Some(size_str) = msg.size.as_deref() else {
                return;
            };
            let Some(side_str) = msg.side.as_deref() else {
                return;
            };
            let Ok(size) = Decimal::from_str(size_str) else {
                return;
            };
            let signed = if side_str.eq_ignore_ascii_case("BUY") {
                size
            } else if side_str.eq_ignore_ascii_case("SELL") {
                -size
            } else {
                return;
            };
            let ts_ms = msg
                .timestamp
                .unwrap_or_else(|| Utc::now().timestamp_millis());
            // Retention = 2x the query window: margin so as not to lose
            // coverage right at the edge of the window.
            registry.record_trade(asset_id, ts_ms, signed, tape_window_secs * 2);
            tracing::trace!(
                token = asset_id, price = msg.price.as_deref().unwrap_or(""), side = side_str, size = %size,
                "market_feed: tape trade accumulated"
            );
        }
        _ => {}
    }
}

/// Converts the items of RawLevel into Vec<Level>, discarding levels with size=0.
pub fn parse_levels(raw: &[RawLevel]) -> Vec<Level> {
    raw.iter()
        .filter_map(|item| {
            let price = Decimal::from_str(&item.price).ok()?;
            let size = Decimal::from_str(&item.size).ok()?;
            if size <= Decimal::ZERO {
                return None;
            }
            Some(Level { price, size })
        })
        .collect()
}

/// Dedup by price, keeping the LAST occurrence in the raw exchange array
/// (more authoritative than the first). A `book` snapshot with a repeated price
/// used to leave an orphaned level in the book forever
/// (`sort_unstable` does not remove duplicates).
fn dedup_levels_by_price(levels: Vec<Level>) -> Vec<Level> {
    let mut by_price: HashMap<Decimal, Decimal> = HashMap::with_capacity(levels.len());
    let mut order: Vec<Decimal> = Vec::with_capacity(levels.len());
    for l in levels {
        if !by_price.contains_key(&l.price) {
            order.push(l.price);
        }
        by_price.insert(l.price, l.size);
    }
    order
        .into_iter()
        .map(|price| Level {
            price,
            size: by_price[&price],
        })
        .collect()
}

/// Apply a single `price_changes` entry to the local token book state.
fn apply_price_change_entry(state: &mut TokenState, entry: &RawPriceChange) {
    let Ok(price) = Decimal::from_str(&entry.price) else {
        return;
    };
    let Ok(size) = Decimal::from_str(&entry.size) else {
        return;
    };

    let is_bid = entry.side.eq_ignore_ascii_case("BUY");
    let levels = if is_bid {
        &mut state.bids
    } else if entry.side.eq_ignore_ascii_case("SELL") {
        &mut state.asks
    } else {
        return;
    };

    // Sorted insertion (O(log n) search + O(n) insert) instead of a
    // full re-sort O(n log n) after each event.
    if size.is_zero() {
        if let Ok(pos) = levels.binary_search_by(|l| {
            if is_bid {
                l.price.cmp(&price).reverse()
            } else {
                l.price.cmp(&price)
            }
        }) {
            levels.remove(pos);
        }
    } else if let Ok(pos) = levels.binary_search_by(|l| {
        if is_bid {
            l.price.cmp(&price).reverse()
        } else {
            l.price.cmp(&price)
        }
    }) {
        levels[pos].size = size;
    } else {
        let insert_pos = levels.partition_point(|l| {
            if is_bid {
                l.price > price
            } else {
                l.price < price
            }
        });
        levels.insert(insert_pos, Level { price, size });
    }
}

fn try_emit_snapshot(
    token_id: &str,
    registry: &Arc<MarketRegistry>,
    book_tx: &broadcast::Sender<Arc<BookSnapshot>>,
    recv_instant: Instant,
    // force=true on book/initial_book (full resync): always emit.
    // force=false on price_change: suppress if only depth changed, not the top.
    force: bool,
    // Tape window used to fill flow_net/flow_total. 0 = disabled.
    tape_window_secs: u64,
) {
    let cid = match registry.token_to_cid.get(token_id) {
        Some(c) => c.clone(),
        None => return,
    };
    // Extract only the fields needed from the Ref: previously the whole
    // MarketEntry was cloned, then its Strings again when building the snapshot,
    // and whole TokenState values were cloned instead of just their levels.
    let (
        yes_token_id,
        no_token_id,
        market_open_ts,
        clear_at,
        neg_risk,
        slug,
        reference_price,
        fee_rate_bps,
        parabolic_fee_rate,
        entry_tick_size,
    ) = match registry.markets.get(&*cid) {
        Some(e) => (
            e.yes_token_id.clone(),
            e.no_token_id.clone(),
            e.market_open_ts,
            e.clear_at,
            e.neg_risk,
            e.slug.clone(),
            e.reference_price,
            e.fee_rate_bps,
            e.parabolic_fee_rate,
            e.tick_size,
        ),
        None => return,
    };
    // Do not emit while EITHER half is waiting for a snapshot
    // (pending needs_snapshot after a WS gap or a crossed book): mixing
    // a fresh half with a frozen one from before the cut manufactures
    // phantom edges that the max_book_age gate cannot see (live incident: 2 pairs
    // with FOK and no counterpart on the leg of the frozen side).
    if registry.needs_snapshot.contains(&yes_token_id)
        || registry.needs_snapshot.contains(&no_token_id)
    {
        metrics::counter!("hrf_market_feed_emit_half_stale_total").increment(1);
        return;
    }

    let (yes_bids, yes_asks) = match registry.token_books.get(&yes_token_id) {
        Some(s) => (s.bids.clone(), s.asks.clone()),
        None => return,
    };
    let (no_bids, no_asks) = match registry.token_books.get(&no_token_id) {
        Some(s) => (s.bids.clone(), s.asks.clone()),
        None => return,
    };

    // Suppress the broadcast if the top-of-book did not change (depth only).
    // Resyncs (force=true) always emit, to guarantee consistency.
    let current_top = [
        yes_bids.first().map(|l| l.price),
        yes_asks.first().map(|l| l.price),
        no_bids.first().map(|l| l.price),
        no_asks.first().map(|l| l.price),
    ];
    if !force
        && registry
            .last_top
            .get(&*cid)
            .map(|t| *t == current_top)
            .unwrap_or(false)
    {
        metrics::counter!("hrf_market_feed_emit_skipped_total").increment(1);
        return;
    }
    registry.last_top.insert(cid.clone(), current_top);

    // min(derived from the book, WS event): a tick that is too fine gets
    // truncated without warning in clob_sdk (authoritative backstop); one that is too
    // coarse would reject 0.999.
    let tick_size = super::registry::effective_tick(&current_top, entry_tick_size);

    // Tape flow per token in the configured window. 0 = disabled
    // → None in both fields (signal absent, never blocks).
    let now_ms = Utc::now().timestamp_millis();
    let (yes_flow_net, yes_flow_total) = if tape_window_secs > 0 {
        match registry.flow_window(&yes_token_id, now_ms, tape_window_secs) {
            Some((net, total)) => (Some(net), Some(total)),
            None => (None, None),
        }
    } else {
        (None, None)
    };
    let (no_flow_net, no_flow_total) = if tape_window_secs > 0 {
        match registry.flow_window(&no_token_id, now_ms, tape_window_secs) {
            Some((net, total)) => (Some(net), Some(total)),
            None => (None, None),
        }
    } else {
        (None, None)
    };

    let snap = Arc::new(BookSnapshot {
        market_condition_id: cid,
        yes: TokenBook {
            token_id: yes_token_id,
            bids: HalfBook { levels: yes_bids },
            asks: HalfBook { levels: yes_asks },
            flow_net: yes_flow_net,
            flow_total: yes_flow_total,
        },
        no: TokenBook {
            token_id: no_token_id,
            bids: HalfBook { levels: no_bids },
            asks: HalfBook { levels: no_asks },
            flow_net: no_flow_net,
            flow_total: no_flow_total,
        },
        market_open_ts,
        clear_at,
        neg_risk,
        slug,
        reference_price,
        fee_rate_bps, // fee taker of the market (Gamma feeRateBps or /fee-rate from the CLOB)
        parabolic_fee_rate, // real parabolic taker fee (Gamma feeSchedule.rate)
        tick_size: Some(tick_size),
        recv_instant: Some(recv_instant),
    });

    match book_tx.send(snap) {
        Ok(receivers) => {
            metrics::gauge!("hrf_market_feed_broadcast_receivers").set(receivers as f64);
        }
        Err(_) => {
            metrics::gauge!("hrf_market_feed_broadcast_receivers").set(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_feed::registry::MarketEntry;
    use rust_decimal_macros::dec;

    fn registry_with_token(token: &str) -> Arc<MarketRegistry> {
        let registry = Arc::new(MarketRegistry::new());
        registry.markets.insert(
            "cid-1".into(),
            MarketEntry {
                slug: "btc-updown-5m-test".into(),
                yes_token_id: token.into(),
                no_token_id: "tok-no".into(),
                clear_at: 9_999_999_999,
                market_open_ts: 0,
                neg_risk: false,
                reference_price: Some(dec!(100000)),
                fee_rate_bps: None,
                parabolic_fee_rate: None,
                tick_size: None,
                rewards_min_size: None,
                rewards_max_spread: None,
                rewards_daily_rate: None,
            },
        );
        registry.token_to_cid.insert(token.into(), "cid-1".into());
        registry
            .token_to_cid
            .insert("tok-no".into(), "cid-1".into());
        let level = |p: &str, s: &str| Level {
            price: p.parse().unwrap(),
            size: s.parse().unwrap(),
        };
        registry.token_books.insert(
            token.into(),
            TokenState {
                bids: vec![level("0.55", "10")],
                asks: vec![level("0.57", "10")],
                updated_at: Utc::now(),
            },
        );
        registry.token_books.insert(
            "tok-no".into(),
            TokenState {
                bids: vec![level("0.43", "10")],
                asks: vec![level("0.45", "10")],
                updated_at: Utc::now(),
            },
        );
        registry
    }

    /// A price_change whose intermediate state crosses the book, but whose
    /// FINAL state is valid, must not request a resync or set needs_snapshot.
    /// (Before: "new ask 0.54" + "remove bid 0.55" in the same message triggered
    /// the resync after the first entry and discarded the second one.)
    #[test]
    fn transient_intra_message_cross_does_not_request_resync() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        // The new ask (0.54) sits below the current bid (0.55) until the
        // second entry removes that bid. Final state: bid 0.53 / ask 0.54.
        let msg = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.54","size":"5","side":"SELL"},
            {"asset_id":"tok-yes","price":"0.55","size":"0","side":"BUY"},
            {"asset_id":"tok-yes","price":"0.53","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        assert!(
            !registry.needs_snapshot.contains("tok-yes"),
            "a final state that is not crossed must not set needs_snapshot"
        );
        let snap = book_rx.try_recv().expect("must emit the final snapshot");
        assert_eq!(snap.yes.best_bid(), Some(dec!(0.53)));
        assert_eq!(snap.yes.best_ask(), Some(dec!(0.54)));
    }

    /// A crossing that was just detected does not escalate to a resync immediately:
    /// it is granted a grace margin (most self-correct with the next delta,
    /// split across different WS messages for the same trade). It must not emit
    /// the crossed book in the meantime, but nor request a resync yet.
    #[test]
    fn freshly_detected_cross_does_not_escalate_immediately() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.60","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        assert!(
            !registry.needs_snapshot.contains("tok-yes"),
            "the first crossed tick must not escalate yet"
        );
        assert!(
            book_rx.try_recv().is_err(),
            "crossed book must not be emitted"
        );
    }

    /// A book that stays crossed on the NEXT price_change of the same token
    /// (the real desynchronisation signal, not a timeout) does request a resync and does not emit.
    #[test]
    fn persistent_cross_on_next_delta_requests_resync() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.60","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);
        // Second delta of the SAME token that also fails to fix the crossing: it is the
        // real signal (not a clock) that this is not the usual transient split.
        let msg2 = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.61","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg2, &registry, &book_tx, Instant::now(), 60);

        assert!(registry.needs_snapshot.contains("tok-yes"));
        assert!(
            book_rx.try_recv().is_err(),
            "a crossed book must not be emitted"
        );
    }

    /// The scenario that motivates the whole fix: a crossing that self-corrects with the
    /// next delta (a separate WS message, not the same entry) must not request
    /// a resync, and the already healthy snapshot is emitted normally again.
    #[test]
    fn cross_that_self_corrects_on_next_delta_does_not_request_resync() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.60","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        // SEPARATE WS message (not the same entry) that removes the crossed ask and
        // publishes a new one above the bid: the split trade that motivated the fix.
        let msg2 = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.57","size":"0","side":"SELL"},
            {"asset_id":"tok-yes","price":"0.62","size":"5","side":"SELL"}
        ]}"#;
        dispatch(msg2, &registry, &book_tx, Instant::now(), 60);

        assert!(
            !registry.needs_snapshot.contains("tok-yes"),
            "it self-corrected, must not escalate"
        );
        let snap = book_rx.try_recv().expect("must emit the already healthy snapshot");
        assert_eq!(snap.yes.best_bid(), Some(dec!(0.60)));
        assert_eq!(snap.yes.best_ask(), Some(dec!(0.62)));
    }

    /// An authoritative `book` snapshot resets the persistent-crossing window:
    /// the `crossed_pending` state describes the previous book,
    /// not the new one. Without the reset, after each resync the token escalated to
    /// `needs_snapshot` with a single crossed delta (a loop of ~10× WARNs).
    #[test]
    fn fresh_book_resets_persistent_cross_window() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, _book_rx) = broadcast::channel(8);

        // First crossing: crossed_pending set (needs_snapshot not yet).
        let msg = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.60","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);
        assert!(registry.crossed_pending.contains("tok-yes"));
        assert!(!registry.needs_snapshot.contains("tok-yes"));

        // Fresh, healthy book snapshot: resets the window.
        let book = r#"{"type":"book","asset_id":"tok-yes","bids":[{"price":"0.55","size":"5"}],"asks":[{"price":"0.57","size":"5"}]}"#;
        dispatch(book, &registry, &book_tx, Instant::now(), 60);
        assert!(
            !registry.crossed_pending.contains("tok-yes"),
            "the fresh book must reset crossed_pending"
        );

        // A single crossed delta no longer escalates: two in a row are needed again.
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);
        assert!(
            !registry.needs_snapshot.contains("tok-yes"),
            "a single crossing after a fresh book must not escalate"
        );
        let msg2 = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.61","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg2, &registry, &book_tx, Instant::now(), 60);
        assert!(
            registry.needs_snapshot.contains("tok-yes"),
            "a second consecutive crossing does escalate"
        );
    }

    /// After escalating, re-escalations of the same token are
    /// suppressed during the cooldown: the oscillating token throttles itself
    /// instead of looping resync→re-cross→WARN every ~4.5 s. `crossed_pending`
    /// stays set while the suppression lasts (the next crossing re-evaluates)
    /// and a non-crossed delta fully resets the backoff.
    #[test]
    fn cross_escalation_applies_backoff_per_token() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, _book_rx) = broadcast::channel(8);

        let cross = |p: &str| {
            let msg = format!(
                r#"{{"event_type":"price_change","price_changes":[{{"asset_id":"tok-yes","price":"{p}","size":"5","side":"BUY"}}]}}"#
            );
            dispatch(&msg, &registry, &book_tx, Instant::now(), 60);
        };

        // First pair of consecutive crosses → first escalation (immediate).
        cross("0.60");
        cross("0.61");
        assert!(registry.needs_snapshot.contains("tok-yes"));
        assert!(registry
            .crossed_escalation
            .get("tok-yes")
            .and_then(|e| e.retry_at)
            .is_some_and(|t| Instant::now() < t));

        // Simulate that the resync resolved: the book is written and clean,
        // just as the sweeper would do (needs_snapshot + crossed_pending removed);
        // the escalation backoff SURVIVES, since it is what cuts the loop.
        registry.needs_snapshot.remove("tok-yes");
        registry.crossed_pending.remove("tok-yes");

        // Two consecutive crossings within the cooldown → NO re-escalation: pending
        // stays set to re-evaluate on the next delta, not needs_snapshot.
        cross("0.60");
        cross("0.61");
        assert!(
            !registry.needs_snapshot.contains("tok-yes"),
            "within the cooldown must not re-escalate"
        );
        assert!(
            registry.crossed_pending.contains("tok-yes"),
            "pending stays set during the suppression"
        );

        // A delta that un-crosses the book FORGIVES ONE STEP (streak 1→0),
        // but the entry survives while the cooldown is still active:
        // decay, not reset: a token that oscillates crossed↔healthy does not regain
        // the immediate escalation on each cycle (that oscillation IS the venue churn
        // that the backoff has to absorb).
        let healthy = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.57","size":"0","side":"SELL"},
            {"asset_id":"tok-yes","price":"0.62","size":"5","side":"SELL"}
        ]}"#;
        dispatch(healthy, &registry, &book_tx, Instant::now(), 60);
        {
            let esc = registry
                .crossed_escalation
                .get("tok-yes")
                .expect("a healthy delta only forgives one step: the entry survives");
            assert_eq!(esc.streak, 0);
        }
        // And it stays suppressed: retry_at has not expired. After the healthy delta the ask
        // is at 0.62; crossings need bids ≥ 0.62.
        cross("0.63");
        cross("0.64");
        assert!(
            !registry.needs_snapshot.contains("tok-yes"),
            "decay is not a reset: within the cooldown it stays suppressed"
        );

        // Full pardon: cooldown expired + a delta that un-crosses the book
        // (bid 0.64 < ask 0.70) → the entry is removed entirely.
        {
            let mut esc = registry.crossed_escalation.get_mut("tok-yes").unwrap();
            esc.retry_at = Some(Instant::now() - std::time::Duration::from_secs(1));
        }
        let healthy2 = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.62","size":"0","side":"SELL"},
            {"asset_id":"tok-yes","price":"0.70","size":"5","side":"SELL"}
        ]}"#;
        dispatch(healthy2, &registry, &book_tx, Instant::now(), 60);
        assert!(
            !registry.crossed_escalation.contains_key("tok-yes"),
            "streak=0 + expired cooldown + healthy book → full pardon"
        );

        // And after the pardon, the next desynchronisation escalates immediately
        // again (fail-closed intact). Ask at 0.70 → bids ≥ 0.70 cross.
        cross("0.71");
        cross("0.72");
        assert!(registry.needs_snapshot.contains("tok-yes"));
    }

    /// LIVE-5b: while the OPPOSITE half awaits a snapshot (needs_snapshot after
    /// a WS gap), the pair is not emitted: it would mix a fresh book with a
    /// frozen one from before the cut (a phantom edge that max_book_age does not see).
    #[test]
    fn does_not_emit_while_other_half_awaits_snapshot() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        registry.needs_snapshot.insert("tok-no".into());
        let msg = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.54","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);
        assert!(
            book_rx.try_recv().is_err(),
            "with tok-no pending a snapshot, the pair must not be emitted"
        );

        // Once the flag is cleared (its initial_book/resync arrived), it emits again.
        registry.needs_snapshot.remove("tok-no");
        let msg2 = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.53","size":"5","side":"BUY"}
        ]}"#;
        dispatch(msg2, &registry, &book_tx, Instant::now(), 60);
        assert!(
            book_rx.try_recv().is_ok(),
            "with no pending flags it must emit"
        );
    }

    /// TICK-1: the `tick_size_change` event updates the MarketEntry and forces
    /// the snapshot to be re-emitted with the new tick, without waiting for the next book.
    #[test]
    fn tick_size_change_updates_registry_and_re_emits() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"tick_size_change","asset_id":"tok-yes","market":"cid-1","old_tick_size":"0.01","new_tick_size":"0.001"}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        let entry_tick = registry.markets.get("cid-1").unwrap().tick_size;
        assert_eq!(entry_tick, Some(dec!(0.001)));

        let snap = book_rx
            .try_recv()
            .expect("must re-emit the snapshot with the new tick");
        assert_eq!(snap.tick(), dec!(0.001));
    }

    /// A `tick_size_change` with tick ≤ 0 is REJECTED: it never
    /// enters the `MarketEntry` nor forces a re-emission. `Decimal::from_str` accepts
    /// "0" and "-0.01" without error; without this guard `BookSnapshot::tick()` could
    /// return 0 and `rl_predictor` would panic dividing by it.
    #[test]
    fn tick_size_change_non_positive_is_rejected() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        // Initial healthy state as a reference.
        let ok = r#"{"event_type":"tick_size_change","asset_id":"tok-yes","market":"cid-1","new_tick_size":"0.01"}"#;
        dispatch(ok, &registry, &book_tx, Instant::now(), 60);
        assert_eq!(
            registry.markets.get("cid-1").unwrap().tick_size,
            Some(dec!(0.01))
        );
        let _ = book_rx.try_recv();

        for bad in ["0", "-0.01", "0.000"] {
            let msg = format!(
                r#"{{"event_type":"tick_size_change","asset_id":"tok-yes","market":"cid-1","new_tick_size":"{bad}"}}"#
            );
            dispatch(&msg, &registry, &book_tx, Instant::now(), 60);
            assert_eq!(
                registry.markets.get("cid-1").unwrap().tick_size,
                Some(dec!(0.01)),
                "tick {bad} must not overwrite the valid value"
            );
        }
        assert!(
            book_rx.try_recv().is_err(),
            "invalid ticks do not force a re-emission"
        );
    }

    /// A `price_change` for a REGISTERED token without `TokenState`
    /// (its initial `book` never landed) marks `needs_snapshot` so that the
    /// REST sweeper repairs it. Previously it was silently discarded and the token
    /// stayed mute until the next reconnection.
    #[test]
    fn price_change_without_token_state_marks_needs_snapshot() {
        let registry = registry_with_token("tok-yes");
        // The initial book of tok-yes never arrived.
        registry.token_books.remove("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.54","size":"5","side":"SELL"}
        ]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        assert!(
            registry.needs_snapshot.contains("tok-yes"),
            "the delta without state must queue a REST repair"
        );
        assert!(book_rx.try_recv().is_err(), "there is nothing to emit");
    }

    /// The same discard for an UNREGISTERED token (residual subscription
    /// of an evicted market) is NOT queued in `needs_snapshot`:
    /// there is no market to repair and the sweeper would only spend REST calls.
    #[test]
    fn price_change_of_unregistered_token_is_not_queued() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, _book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-phantom","price":"0.54","size":"5","side":"SELL"}
        ]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        assert!(
            !registry.needs_snapshot.contains("tok-phantom"),
            "a token not in the registry must not enter needs_snapshot"
        );
    }

    /// Without a previous WS event, a best above 0.96 derives a fine tick.
    #[test]
    fn emit_derives_fine_tick_when_crossing_096() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let msg = r#"{"type":"book","asset_id":"tok-yes","bids":[{"price":"0.95","size":"5"}],"asks":[{"price":"0.97","size":"5"}]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        let snap = book_rx.try_recv().expect("must emit snapshot");
        assert_eq!(snap.tick(), dec!(0.001));
    }

    /// Mid-zone bests (not crossing 0.96/0.04) derive the standard coarse tick.
    #[test]
    fn emit_coarse_tick_in_mid_zone() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let msg = r#"{"type":"book","asset_id":"tok-yes","bids":[{"price":"0.55","size":"5"}],"asks":[{"price":"0.57","size":"5"}]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        let snap = book_rx.try_recv().expect("must emit snapshot");
        assert_eq!(snap.tick(), dec!(0.01));
    }

    // ── Trade tape ────────────────────────────────────────────────────────────

    /// A `last_trade_price` accumulates the signed flow in `token_flow`,
    /// queryable via `flow_window`.
    #[test]
    fn last_trade_price_accumulates_flow_per_token() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.55","side":"BUY","size":"5","timestamp":"1000000"}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        let (net, total) = registry
            .flow_window("tok-yes", 1_000_000, 60)
            .expect("must have history after the trade");
        assert_eq!(net, dec!(5));
        assert_eq!(total, dec!(5));
        assert!(
            book_rx.try_recv().is_err(),
            "last_trade_price must not emit a snapshot on its own"
        );
    }

    /// Sign = side of the TAKER: an aggressive sell subtracts from the net
    /// (while still adding to the gross total), an aggressive buy adds.
    #[test]
    fn aggressive_sell_subtracts_buy_adds_in_tape() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, _book_rx) = broadcast::channel(8);

        let buy = r#"{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.55","side":"BUY","size":"5","timestamp":"1000000"}"#;
        let sell = r#"{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.55","side":"SELL","size":"3","timestamp":"1000500"}"#;
        dispatch(buy, &registry, &book_tx, Instant::now(), 60);
        dispatch(sell, &registry, &book_tx, Instant::now(), 60);

        let (net, total) = registry.flow_window("tok-yes", 1_000_500, 60).unwrap();
        assert_eq!(net, dec!(2));
        assert_eq!(total, dec!(8));
    }

    /// The retention (2× window) prunes a stale trade, even though the QUERY window
    /// is wide enough to have covered it: pruning happens on the insert of the
    /// next trade, not on the query.
    #[test]
    fn out_of_window_flow_expires() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, _book_rx) = broadcast::channel(8);

        let t1 = 1_000_000i64;
        let t2 = t1 + 130_000; // 130 s later: exceeds the 2×60 s = 120 s retention
        let old = format!(
            r#"{{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.55","side":"BUY","size":"10","timestamp":"{t1}"}}"#
        );
        let recent = format!(
            r#"{{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.55","side":"BUY","size":"3","timestamp":"{t2}"}}"#
        );
        dispatch(&old, &registry, &book_tx, Instant::now(), 60);
        dispatch(&recent, &registry, &book_tx, Instant::now(), 60);

        // A 200 s window would cover t1 if it had not been pruned on the insert of t2.
        let (net, total) = registry.flow_window("tok-yes", t2, 200).unwrap();
        assert_eq!(net, dec!(3));
        assert_eq!(total, dec!(3));
    }

    /// The snapshot that IS emitted (the twin `price_change` of the trade) carries
    /// the flow of both tokens already filled in.
    #[test]
    fn emitted_snapshot_carries_flow_of_both_tokens() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        let now_ms = chrono::Utc::now().timestamp_millis();
        let buy_yes = format!(
            r#"{{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.55","side":"BUY","size":"5","timestamp":"{now_ms}"}}"#
        );
        let sell_no = format!(
            r#"{{"event_type":"last_trade_price","asset_id":"tok-no","price":"0.45","side":"SELL","size":"2","timestamp":"{now_ms}"}}"#
        );
        dispatch(&buy_yes, &registry, &book_tx, Instant::now(), 60);
        dispatch(&sell_no, &registry, &book_tx, Instant::now(), 60);
        assert!(
            book_rx.try_recv().is_err(),
            "tape trades do not emit on their own"
        );

        // Legitimate price_change (it does not cross the book) that triggers the emission of the pair.
        let pc = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"tok-yes","price":"0.53","size":"5","side":"BUY"}
        ]}"#;
        dispatch(pc, &registry, &book_tx, Instant::now(), 60);

        let snap = book_rx
            .try_recv()
            .expect("must emit snapshot after the price_change");
        assert_eq!(snap.yes.flow_net, Some(dec!(5)));
        assert_eq!(snap.yes.flow_total, Some(dec!(5)));
        assert_eq!(snap.no.flow_net, Some(dec!(-2)));
        assert_eq!(snap.no.flow_total, Some(dec!(2)));
    }

    /// tape_window_secs=0 disables the tape entirely: no flow is accumulated.
    #[test]
    fn disabled_tape_does_not_accumulate() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, _book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.55","side":"BUY","size":"5","timestamp":"1000000"}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 0);

        assert!(
            registry.flow_window("tok-yes", 1_000_000, 60).is_none(),
            "with tape_window_secs=0 no trade must be accumulated"
        );
    }

    /// Bugfix: a `last_trade_price` for a token NOT registered (e.g. a bracket
    /// already evicted by `evict_expired`, whose late trade arrives through the
    /// cumulative WS subscription) is discarded BEFORE `record_trade`: it must not
    /// create an entry in `token_flow`, or the memory leak returns.
    #[test]
    fn last_trade_price_of_unregistered_token_creates_no_token_flow_entry() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, _book_rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"last_trade_price","asset_id":"tok-evicted","price":"0.55","side":"BUY","size":"5","timestamp":"1000000"}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        assert!(
            registry.flow_window("tok-evicted", 1_000_000, 60).is_none(),
            "the unregistered token must not have an entry in token_flow"
        );
        assert!(
            !registry.token_flow.contains_key("tok-evicted"),
            "record_trade must not have been called for an unregistered token"
        );
    }

    // ── Crossed initial book ──────────────────────────────────────────────────

    /// A crossed `book`/`initial_book` snapshot must mark `needs_snapshot`
    /// BEFORE (or together with) requesting the REST resync; otherwise a
    /// legitimate `price_change` arriving while the resync is in flight refreshes
    /// `updated_at` without the flag being set, and `resync_token`
    /// (resync.rs) concludes "the WS won the race" even though the book was still
    /// crossed (a permanent no-op on any token with traffic).
    #[test]
    fn crossed_initial_book_marks_needs_snapshot() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        // bid >= ask ⇒ crossed.
        let msg = r#"{"type":"book","asset_id":"tok-yes","bids":[{"price":"0.60","size":"5"}],"asks":[{"price":"0.55","size":"5"}]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        assert!(
            registry.needs_snapshot.contains("tok-yes"),
            "a crossed initial book must mark needs_snapshot so that the \
             intermediate price_change messages are discarded until the resync"
        );
        assert!(
            book_rx.try_recv().is_err(),
            "crossed book must not be emitted"
        );
    }

    // ── Repeated price levels ─────────────────────────────────────────────────

    /// A `book` snapshot with a repeated price must not leave an orphaned
    /// level: the LAST occurrence is kept (most recent size).
    #[test]
    fn book_with_repeated_price_dedupes_keeping_last() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);

        // Two bid levels at the same price 0.50: the exchange repeats the price,
        // so the second occurrence (size=7) must win.
        let msg = r#"{"type":"book","asset_id":"tok-yes","bids":[
            {"price":"0.50","size":"3"},
            {"price":"0.48","size":"2"},
            {"price":"0.50","size":"7"}
        ],"asks":[{"price":"0.57","size":"5"}]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);

        let snap = book_rx.try_recv().expect("must emit snapshot");
        // Only ONE level at 0.50 must remain (with size=7), and the 0.48 one separately.
        let levels_050: Vec<_> = snap
            .yes
            .bids
            .levels
            .iter()
            .filter(|l| l.price == dec!(0.50))
            .collect();
        assert_eq!(
            levels_050.len(),
            1,
            "there must be no duplicate levels at the same price"
        );
        assert_eq!(
            levels_050[0].size,
            dec!(7),
            "must keep the size of the LAST occurrence"
        );
        assert_eq!(snap.yes.bids.levels.len(), 2, "0.50 (deduplicated) + 0.48");
    }

    // ── Research recording hooks (research::events) ────────────────────────────

    use crate::research::contract::{EventKind, EventSource, QualityFlags};
    use crate::research::events::{
        RecordedLine, ResearchRecorder, ResearchRecorderConfig, SourceContext,
    };

    /// Reads all the JSONL lines the recorder wrote under `dir` (one file per
    /// source/day), in `local_seq` order.
    fn read_recorded_events(dir: &std::path::Path) -> Vec<RecordedLine> {
        let mut lines = Vec::new();
        for source in std::fs::read_dir(dir).unwrap() {
            let source = source.unwrap();
            if !source.file_type().unwrap().is_dir() {
                continue;
            }
            for f in std::fs::read_dir(source.path()).unwrap() {
                let f = f.unwrap();
                if f.path().extension().is_some_and(|e| e == "jsonl") {
                    for line in std::fs::read_to_string(f.path()).unwrap().lines() {
                        if !line.trim().is_empty() {
                            lines.push(serde_json::from_str::<RecordedLine>(line).unwrap());
                        }
                    }
                }
            }
        }
        lines.sort_by_key(|l| l.event.local_seq);
        lines
    }

    fn recorder_and_ctx(dir: &std::path::Path) -> (ResearchRecorder, SourceContext) {
        let hub = crate::research::events::Hub::new();
        let recorder = ResearchRecorder::start(
            &ResearchRecorderConfig {
                dir: dir.to_path_buf(),
                channel_capacity: 64,
                ..ResearchRecorderConfig::default()
            },
            &hub,
        )
        .unwrap();
        let ctx = SourceContext {
            producer: hub.producer(),
            generation: 7,
            stream: "poly-ws:test".into(),
        };
        (recorder, ctx)
    }

    /// A `last_trade_price` that does NOT move the book (a trade without a
    /// book change) is still recorded as a `Trade` event: the trade tape is kept
    /// even when no snapshot is emitted.
    #[tokio::test]
    async fn trade_without_book_change_is_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let (recorder, ctx) = recorder_and_ctx(dir.path());
        let registry = registry_with_token("tok-yes");
        let (book_tx, _rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.61","side":"BUY","size":"12.5","timestamp":"1780000000000"}"#;
        dispatch_with_recorder(msg, &registry, &book_tx, Instant::now(), 60, &ctx);

        drop(ctx);
        let res = recorder.stop().await;
        assert_eq!(res.written, 1);

        let events = read_recorded_events(dir.path());
        assert_eq!(events.len(), 1);
        let l = &events[0];
        assert_eq!(l.event.kind, EventKind::Trade);
        assert_eq!(l.event.source, EventSource::PolymarketWs);
        assert_eq!(l.event.token_id.as_deref(), Some("tok-yes"));
        assert_eq!(l.event.market_cid.as_deref(), Some("cid-1"));
        assert_eq!(l.event.origin_ts_ms, Some(1_780_000_000_000));
        assert_eq!(l.generation, Some(7));
        assert_eq!(l.stream.as_deref(), Some("poly-ws:test"));
        assert_eq!(l.event.payload["price"], serde_json::json!("0.61"));
        assert_eq!(l.event.payload["side"], serde_json::json!("BUY"));
        assert_eq!(l.event.payload["size"], serde_json::json!("12.5"));
    }

    /// Several trades of the same token between two books are ALL recorded:
    /// the tape does not collapse to "last per key".
    #[tokio::test]
    async fn several_trades_between_books_are_all_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let (recorder, ctx) = recorder_and_ctx(dir.path());
        let registry = registry_with_token("tok-yes");
        let (book_tx, _rx) = broadcast::channel(8);

        let book = r#"{"type":"book","asset_id":"tok-yes","bids":[{"price":"0.55","size":"5"}],"asks":[{"price":"0.57","size":"5"}],"timestamp":"1780000000000"}"#;
        dispatch_with_recorder(book, &registry, &book_tx, Instant::now(), 60, &ctx);
        for (i, ts) in [1u64, 2, 3].iter().enumerate() {
            let msg = format!(
                r#"{{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.6{i}","side":"BUY","size":"{i}","timestamp":"178000000100{i}"}}"#
            );
            let _ = ts;
            dispatch_with_recorder(&msg, &registry, &book_tx, Instant::now(), 60, &ctx);
        }

        drop(ctx);
        let res = recorder.stop().await;
        assert_eq!(res.written, 4, "1 book + 3 trades");

        let events = read_recorded_events(dir.path());
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].event.kind, EventKind::BookSnapshot);
        for l in &events[1..] {
            assert_eq!(l.event.kind, EventKind::Trade);
        }
    }

    /// The same trade re-sent (a re-send after reconnection, even with a
    /// different generation) is deduplicated by stable content: volume is not
    /// duplicated. The key is (token, price, side, size, timestamp):
    /// NEVER recv_ts or local_seq.
    #[tokio::test]
    async fn trade_resent_after_reconnection_is_deduplicated() {
        let dir = tempfile::tempdir().unwrap();
        let (recorder, ctx) = recorder_and_ctx(dir.path());
        let mut ctx2 = ctx.clone();
        ctx2.generation = 8; // new connection: same trade re-sent
        let registry = registry_with_token("tok-yes");
        let (book_tx, _rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.61","side":"BUY","size":"12.5","timestamp":"1780000000000"}"#;
        dispatch_with_recorder(msg, &registry, &book_tx, Instant::now(), 60, &ctx);
        dispatch_with_recorder(msg, &registry, &book_tx, Instant::now(), 60, &ctx2);

        drop(ctx);
        drop(ctx2);
        let res = recorder.stop().await;
        assert_eq!(res.written, 1, "the re-send does not write twice");
        assert_eq!(res.deduped, 1);
    }

    /// A `price_change` is recorded as a `BookDelta` with the raw payload.
    #[tokio::test]
    async fn price_change_is_recorded_as_delta() {
        let dir = tempfile::tempdir().unwrap();
        let (recorder, ctx) = recorder_and_ctx(dir.path());
        let registry = registry_with_token("tok-yes");
        let (book_tx, _rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"price_change","timestamp":"1780000000000","price_changes":[
            {"asset_id":"tok-yes","price":"0.53","size":"5","side":"BUY"}
        ]}"#;
        dispatch_with_recorder(msg, &registry, &book_tx, Instant::now(), 60, &ctx);

        drop(ctx);
        let res = recorder.stop().await;
        assert_eq!(res.written, 1);

        let events = read_recorded_events(dir.path());
        let l = &events[0];
        assert_eq!(l.event.kind, EventKind::BookDelta);
        assert_eq!(l.event.origin_ts_ms, Some(1_780_000_000_000));
        assert_eq!(
            l.event.payload["price_changes"][0]["asset_id"],
            serde_json::json!("tok-yes")
        );
    }

    /// A missing remote timestamp: a message without `timestamp` records
    /// `origin_ts_ms = null` with the NO_REMOTE_TS flag (it is never
    /// invented from the local clock).
    #[tokio::test]
    async fn message_without_timestamp_marks_missing_remote_ts() {
        let dir = tempfile::tempdir().unwrap();
        let (recorder, ctx) = recorder_and_ctx(dir.path());
        let registry = registry_with_token("tok-yes");
        let (book_tx, _rx) = broadcast::channel(8);

        let msg = r#"{"event_type":"last_trade_price","asset_id":"tok-yes","price":"0.61","side":"BUY","size":"1"}"#;
        dispatch_with_recorder(msg, &registry, &book_tx, Instant::now(), 60, &ctx);

        drop(ctx);
        recorder.stop().await;

        let events = read_recorded_events(dir.path());
        assert_eq!(events[0].event.origin_ts_ms, None);
        assert!(events[0]
            .event
            .quality
            .contains(QualityFlags::MISSING_REMOTE_TS));
    }

    /// Without a recorder (plain `dispatch`) the behaviour does not change:
    /// the hook is strictly additive.
    #[test]
    fn dispatch_without_recorder_keeps_working() {
        let registry = registry_with_token("tok-yes");
        let (book_tx, mut book_rx) = broadcast::channel(8);
        let msg = r#"{"type":"book","asset_id":"tok-yes","bids":[{"price":"0.55","size":"5"}],"asks":[{"price":"0.57","size":"5"}]}"#;
        dispatch(msg, &registry, &book_tx, Instant::now(), 60);
        assert!(book_rx.try_recv().is_ok());
    }

    /// The recorded payload is the ORIGINAL message of the
    /// wire: fields that `WsMessage` does not declare (`market`,
    /// `old_tick_size`, `fee_rate_bps`, `status`, `transaction_hash`…) are
    /// preserved via the `extra` catch-all. Losing `market` prevented resolving
    /// `market_cid` for tokens evicted from the registry.
    #[tokio::test]
    async fn payload_ws_keeps_fields_not_declared_in_wire() {
        let dir = tempfile::tempdir().unwrap();
        let (recorder, ctx) = recorder_and_ctx(dir.path());
        let registry = registry_with_token("tok-yes");
        let (book_tx, _rx) = broadcast::channel(8);

        // last_trade_price with the extra fields that the real venue sends.
        let wire_trade = r#"{
            "event_type": "last_trade_price",
            "asset_id": "tok-yes",
            "price": "0.62", "size": "40", "side": "BUY",
            "market": "cid-abc", "fee_rate_bps": "25",
            "status": "matched", "transaction_hash": "0xdead",
            "timestamp": "1700000000001"
        }"#;
        dispatch_with_recorder(
            wire_trade,
            &registry,
            &book_tx,
            Instant::now(),
            0,
            &ctx,
        );

        // tick_size_change with `market`/`old_tick_size` (not declared).
        let wire_tick = r#"{
            "type": "tick_size_change",
            "asset_id": "tok-yes", "new_tick_size": "0.001",
            "market": "cid-abc", "old_tick_size": "0.01",
            "timestamp": 1700000000002
        }"#;
        dispatch_with_recorder(
            wire_tick,
            &registry,
            &book_tx,
            Instant::now(),
            0,
            &ctx,
        );

        // book WITHOUT a valid asset_id: it is recorded anyway (emitted before validation).
        let wire_book = r#"{
            "type": "book", "asset_id": "",
            "bids": [], "asks": [], "timestamp": "1700000000003"
        }"#;
        dispatch_with_recorder(
            wire_book,
            &registry,
            &book_tx,
            Instant::now(),
            0,
            &ctx,
        );

        drop(ctx);
        let res = recorder.stop().await;
        assert_eq!(res.written, 3, "all three messages arrived and were recorded");

        let events = read_recorded_events(dir.path());
        let trade = events
            .iter()
            .find(|l| l.event.kind == EventKind::Trade)
            .unwrap();
        assert_eq!(trade.event.payload["market"], serde_json::json!("cid-abc"));
        assert_eq!(trade.event.payload["fee_rate_bps"], serde_json::json!("25"));
        assert_eq!(
            trade.event.payload["transaction_hash"],
            serde_json::json!("0xdead")
        );
        assert_eq!(trade.event.payload["status"], serde_json::json!("matched"));

        let tick = events
            .iter()
            .find(|l| {
                l.event.kind == EventKind::BookDelta
                    && l.event.payload["new_tick_size"] == serde_json::json!("0.001")
            })
            .unwrap();
        assert_eq!(tick.event.payload["market"], serde_json::json!("cid-abc"));
        assert_eq!(
            tick.event.payload["old_tick_size"],
            serde_json::json!("0.01")
        );

        let book = events
            .iter()
            .find(|l| l.event.kind == EventKind::BookSnapshot)
            .unwrap();
        assert_eq!(book.event.token_id, None, "empty asset_id → no token");
    }
}
