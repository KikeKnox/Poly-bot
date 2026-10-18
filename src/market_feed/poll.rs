//! Batched REST poll of scan-only universe tokens (`poll_token_ids`).
//!
//! The venue limits the market WS in proportion to the TOTAL of subscribed
//! tokens (measured: rate of 1013 closes ~constant at 4/15/24 connections) —
//! the ~1,200 tokens that no enabled strategy consumes are taken out of the
//! stream and served via `POST /books` every `scan_poll_interval_secs`.
//!
//! Consumers of these books: `tools` (LadderArb/VolSurface — mid only),
//! `tools::recorders::analytics` (crypto-updown* universes), post_close/markout.
//! Snapshots are emitted through the SAME `broadcast<BookSnapshot>` as the WS
//! (`emit_snapshot_if_ready` from resync) — consumers don't tell the
//! source apart. `recv_instant` is stamped with the book's real freshness
//! (updated_at of the oldest side).
//!
//! Differences from `resync.rs`:
//! - There is no `needs_snapshot` or per-token backoff: the poll is the ONLY writer
//!   of these tokens — a failure just leaves the data one more tick stale.
//! - A crossed REST book IS written (with a metric): REST is the authoritative source
//!   for these tokens — resync's "don't write crossed" gate exists so as
//!   not to overwrite a desynchronized WS, a scenario that doesn't exist here. "Locked"
//!   books (bid=ask≈1.0) of resolving markets are valid data
//!   for the scan — their mid=1.0 IS the signal.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;

use super::registry::{book_is_crossed, MarketRegistry};
use super::resync::{
    emit_snapshot_if_ready, fetch_books, parse_levels_from_raw, write_book_if_fresher, BookOutcome,
};
use crate::types::BookSnapshot;

/// Periodic task: sweeps `registry.poll_token_ids()` in `POST /books` batches,
/// writes `token_books` and emits a `BookSnapshot` once per touched market.
pub async fn run_scan_poll_task(
    registry: Arc<MarketRegistry>,
    book_tx: broadcast::Sender<Arc<BookSnapshot>>,
    http: reqwest::Client,
    interval_secs: u64,
    batch_size: usize,
) {
    let mut tick = tokio::time::interval(Duration::from_secs(interval_secs.max(1)));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Pacing between requests within a single sweep — reuses the
    // sweeper's global rate limit (same endpoint, same order of magnitude).
    let min_request_interval = Duration::from_millis(250);
    let mut last_request = std::time::Instant::now() - min_request_interval;

    loop {
        tick.tick().await;

        let tokens = registry.poll_token_ids();
        if tokens.is_empty() {
            continue;
        }

        let mut written = 0usize;
        let mut emitted_cids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for chunk in tokens.chunks(batch_size.max(1)) {
            let delay = min_request_interval.saturating_sub(last_request.elapsed());
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let fetch_started = chrono::Utc::now();
            last_request = std::time::Instant::now();
            metrics::counter!("hrf_scan_poll_requests_total").increment(1);
            metrics::histogram!("hrf_scan_poll_batch_tokens").record(chunk.len() as f64);

            match fetch_books(&http, chunk).await {
                Ok(books) => {
                    for book in books {
                        let token_id = book.asset_id;
                        if token_id.is_empty() {
                            continue;
                        }
                        let mut bids = parse_levels_from_raw(&book.bids);
                        let mut asks = parse_levels_from_raw(&book.asks);
                        bids.sort_unstable_by_key(|l| std::cmp::Reverse(l.price));
                        asks.sort_unstable_by_key(|a| a.price);
                        if book_is_crossed(&bids, &asks) {
                            // REST is the authoritative source for these tokens
                            // (see module doc): written anyway — the
                            // metric leaves the trace for observability.
                            metrics::counter!("hrf_scan_poll_crossed_written_total").increment(1);
                        }
                        match write_book_if_fresher(&token_id, bids, asks, fetch_started, &registry)
                        {
                            BookOutcome::Written => {
                                // In case it was left marked stale by a WS cut
                                // before its migration to poll.
                                registry.needs_snapshot.remove(&token_id);
                                written += 1;
                                if let Some(cid) = registry.token_to_cid.get(&token_id) {
                                    emitted_cids.insert(cid.clone());
                                }
                            }
                            BookOutcome::WsNewer => {
                                registry.needs_snapshot.remove(&token_id);
                            }
                            BookOutcome::NotWritten => {}
                        }
                    }
                }
                Err(e) => {
                    metrics::counter!("hrf_scan_poll_errors_total").increment(1);
                    tracing::warn!(
                        tokens = chunk.len(),
                        error = %e,
                        "market_feed: scan poll batch failed — next tick retries"
                    );
                }
            }
        }

        // Emit once per market with some side written.
        for cid in &emitted_cids {
            if let Some(entry) = registry.markets.get(cid) {
                emit_snapshot_if_ready(&entry.yes_token_id, &registry, &book_tx);
            }
        }
        metrics::counter!("hrf_scan_poll_tokens_written_total").increment(written as u64);
        metrics::gauge!("hrf_scan_poll_tokens_active").set(tokens.len() as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_feed::registry::UniverseInfo;

    /// The WS/poll partition is governed by `ws_universes`: tokens of universes
    /// declared by enabled strategies go to the WS; the rest to the poller.
    #[test]
    fn ws_poll_partition_per_universe() {
        let registry = MarketRegistry::new();
        registry
            .ws_universes
            .insert("crypto-updown-short".to_string());
        registry
            .token_to_cid
            .insert("tok-ws".into(), "cid-ws".into());
        registry
            .token_to_cid
            .insert("tok-poll".into(), "cid-poll".into());
        registry.universe_info.insert(
            "cid-ws".into(),
            UniverseInfo {
                universe: "crypto-updown-short".into(),
                has_oracle: true,
            },
        );
        registry.universe_info.insert(
            "cid-poll".into(),
            UniverseInfo {
                universe: "crypto-multistrike".into(),
                has_oracle: true,
            },
        );

        let ws = registry.ws_token_ids();
        let poll = registry.poll_token_ids();
        assert!(ws.contains(&"tok-ws".to_string()));
        assert!(!ws.contains(&"tok-poll".to_string()));
        assert!(poll.contains(&"tok-poll".to_string()));
        assert!(!poll.contains(&"tok-ws".to_string()));
    }

    /// Without `universe_info` (registration race) the token is WS-bound by
    /// default — conservative: better to over-subscribe than to lose data.
    #[test]
    fn token_without_universe_info_is_ws_bound() {
        let registry = MarketRegistry::new();
        registry.token_to_cid.insert("tok-x".into(), "cid-x".into());
        assert!(registry.ws_token_ids().contains(&"tok-x".to_string()));
        assert!(registry.poll_token_ids().is_empty());
    }

    /// `mark_all_needs_snapshot` only marks WS-bound tokens stale — poll ones are not
    /// degraded on a WS cut (they keep emitting via REST).
    #[test]
    fn mark_all_only_stales_ws_tokens() {
        let registry = MarketRegistry::new();
        registry.ws_universes.insert("oper".to_string());
        registry
            .token_to_cid
            .insert("tok-ws".into(), "cid-ws".into());
        registry
            .token_to_cid
            .insert("tok-poll".into(), "cid-poll".into());
        registry.universe_info.insert(
            "cid-ws".into(),
            UniverseInfo {
                universe: "oper".into(),
                has_oracle: true,
            },
        );
        registry.universe_info.insert(
            "cid-poll".into(),
            UniverseInfo {
                universe: "scan".into(),
                has_oracle: false,
            },
        );

        registry.mark_all_needs_snapshot();

        assert!(registry.needs_snapshot.contains("tok-ws"));
        assert!(
            !registry.needs_snapshot.contains("tok-poll"),
            "the poll-only token must not be marked stale by a WS cut"
        );
    }
}
