//! Predictive prefetch of `*-updown-{dur}-{epoch}` brackets.
//!
//! Slugs are deterministic: the next bracket of a series is
//! `{asset}-updown-{dur}-{current_epoch + dur}` and the market already exists on Gamma
//! (with token IDs) before opening — verified live: the
//! +2 bracket was published >10 min before its boundary.
//!
//! This task predicts the successor slug of each registered series, queries it on
//! Gamma every `prefetch_interval_secs` and, as soon as it appears, registers it with the
//! same owning universe and fires a dynamic `TokenEvent::Subscribe`
//! (`operation:subscribe` — verified live that the initial dump arrives in
//! ~1 RTT). Result: bracket data from its first tick instead of up to
//! `market_poll_interval_secs` (30s) late.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::broadcast;

use super::discover::{
    fetch_market_by_slug, persist_market_observed, refresh_taker_fee, register_if_new,
};
use super::events::TokenEvent;
use super::registry::MarketRegistry;
use crate::config::Config;
use crate::state::StateStore;

/// Backoff after a Gamma miss ("doesn't exist yet"): exponential 2s→60s —
/// a delisted series stops hammering the API without being abandoned entirely.
const MISS_BACKOFF_BASE_SECS: u64 = 2;
const MISS_BACKOFF_MAX_SECS: u64 = 60;

/// Generates the candidate slugs of each series: current boundary + next one,
/// minus those already registered. Extracted to test the bounding.
fn next_candidates(
    series: &HashMap<String, (u64, String)>,
    known_slugs: &HashSet<String>,
    now: i64,
) -> HashMap<String, String> {
    let mut candidates = HashMap::new();
    for (stem, (dur_secs, universe)) in series {
        let dur = *dur_secs as i64;
        let base = now / dur * dur;
        for epoch in [base, base + dur] {
            let slug = format!("{stem}-{epoch}");
            if !known_slugs.contains(&slug) {
                candidates.insert(slug, universe.clone());
            }
        }
    }
    candidates
}

/// Parses `{asset}-updown-{dur_str}-{epoch}` → (prefix `{asset}-updown-{dur}`, dur_secs, epoch).
/// Rejects slugs foreign to the convention (ladders, tags, rewards…).
fn parse_updown_slug(slug: &str) -> Option<(String, u64, i64)> {
    let (stem, epoch_s) = slug.rsplit_once('-')?;
    let epoch: i64 = epoch_s.parse().ok()?;
    let (prefix, dur_s) = stem.rsplit_once('-')?;
    let dur_secs = match dur_s {
        "5m" => 300,
        "15m" => 900,
        "1h" => 3600,
        "4h" => 14400,
        "1d" => 86400,
        _ => return None,
    };
    if !prefix.ends_with("-updown") {
        return None;
    }
    Some((format!("{prefix}-{dur_s}"), dur_secs, epoch))
}

/// Brackets that prefetch registers in advance also
/// emit `MarketMeta` to the research tape. The producer always exists;
/// with the hub inactive `record` discards at no cost.
pub async fn run_prefetch_task(
    http: reqwest::Client,
    config: Arc<Config>,
    registry: Arc<MarketRegistry>,
    sub_tx: broadcast::Sender<TokenEvent>,
    state: Arc<StateStore>,
    recorder: crate::research::events::EventProducer,
) {
    let interval_secs = config.market_feed.prefetch_interval_secs.max(1);
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval.tick().await; // first cycle immediate (the registry comes from initial_discover)

    // Predicted slug → (instant of last miss, consecutive misses) —
    // exponential backoff so as not to hammer Gamma with dead series.
    let mut miss_backoff: HashMap<String, (Instant, u32)> = HashMap::new();

    loop {
        interval.tick().await;

        // Observed series: stem `{asset}-updown-{dur}` → (dur_secs, owning
        // universe). Deriving the series from ANY registered bracket — not from the
        // latest — covers the gap where discover lost the current bracket due to
        // Gamma's pagination ceiling (offset≈1800): the direct
        // slug query doesn't paginate.
        let mut series: HashMap<String, (u64, String)> = HashMap::new();
        let mut known_slugs: HashSet<String> = HashSet::new();
        for item in registry.universe_info.iter() {
            let cid = item.key().clone();
            if let Some(entry) = registry.markets.get(&cid) {
                if let Some((stem, dur, _epoch)) = parse_updown_slug(&entry.slug) {
                    series.insert(stem, (dur, item.value().universe.clone()));
                }
                known_slugs.insert(entry.slug.clone());
            }
        }

        // Candidates per series: the brackets of the CURRENT boundary and the
        // next — bounded to two per series by construction. The current one
        // is also looked up: if discover lost it to Gamma's pagination
        // ceiling, nobody else registers it and the market would run
        // up to `dur` minutes without data despite existing.
        let now = chrono::Utc::now().timestamp();
        let candidates = next_candidates(&series, &known_slugs, now);

        // Prune the backoff of slugs that were registered or stopped being
        // predicted (the series died).
        miss_backoff.retain(|slug, _| candidates.contains_key(slug));

        let oracle_assets = super::discover::oracle_asset_set(&config);
        for (slug, uname) in candidates {
            // Exponential backoff after a miss — dead series don't hammer.
            if let Some((t, count)) = miss_backoff.get(&slug) {
                let wait = (MISS_BACKOFF_BASE_SECS << (*count).min(5)).min(MISS_BACKOFF_MAX_SECS);
                if t.elapsed().as_secs() < wait {
                    continue;
                }
            }
            let Some(uni) = config.universes.get(&uname) else {
                continue;
            };
            match fetch_market_by_slug(&http, &slug).await {
                Ok(Some(gm)) => {
                    miss_backoff.remove(&slug);
                    let cid = gm.condition_id.clone();
                    let Some((token_ids, entry)) = register_if_new(
                        &gm,
                        &registry,
                        &uname,
                        uni,
                        &oracle_assets,
                        &recorder,
                    ) else {
                        continue; // already registered by the general poll (benign race)
                    };
                    metrics::counter!("hrf_prefetch_markets_total").increment(1);
                    tracing::info!(
                        slug = %slug,
                        universe = %uname,
                        cid = %cid,
                        "market_feed: prefetch registered bracket in advance"
                    );
                    persist_market_observed(&state, &cid, &entry).await;
                    // Real taker fee (fire-and-forget, same pattern as discover).
                    {
                        let http = http.clone();
                        let config = config.clone();
                        let registry = registry.clone();
                        let token = token_ids[0].clone();
                        tokio::spawn(async move {
                            refresh_taker_fee(&http, &config, &registry, &cid, &token).await;
                        });
                    }
                    // Immediate subscription only if the universe is WS-bound;
                    // scan-only universes are picked up by the REST poller.
                    if registry.ws_universes.contains(&uname) {
                        metrics::counter!("hrf_prefetch_ws_subscribe_total").increment(1);
                        if sub_tx.send(TokenEvent::Subscribe { token_ids }).is_err() {
                            tracing::warn!(
                                "market_feed: prefetch could not send Subscribe — ws supervisor down"
                            );
                        }
                    }
                }
                Ok(None) => {
                    let prev = miss_backoff.get(&slug).map(|(_, c)| *c).unwrap_or(0);
                    miss_backoff.insert(slug.clone(), (Instant::now(), prev + 1));
                }
                Err(e) => {
                    metrics::counter!("hrf_prefetch_errors_total").increment(1);
                    let prev = miss_backoff.get(&slug).map(|(_, c)| *c).unwrap_or(0);
                    miss_backoff.insert(slug.clone(), (Instant::now(), prev + 1));
                    tracing::debug!(error = %e, slug = %slug, "market_feed: prefetch slug failed");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_slug_ok() {
        let (stem, dur, epoch) = parse_updown_slug("btc-updown-5m-1789765500").unwrap();
        assert_eq!(stem, "btc-updown-5m");
        assert_eq!(dur, 300);
        assert_eq!(epoch, 1789765500);
    }

    #[test]
    fn parse_slug_rejects_non_updown() {
        assert!(parse_updown_slug("will-btc-hit-100k-1789765500").is_none());
        assert!(parse_updown_slug("btc-updown-7m-1789765500").is_none());
        assert!(parse_updown_slug("btc-updown-5m").is_none());
        assert!(parse_updown_slug("not-a-slug-abc").is_none());
    }

    /// Candidates are the current boundary + the next one, bounded to 2 per
    /// series. The current one is also looked up: a pagination miss in discover
    /// on the live bracket must not leave the market without data.
    #[test]
    fn candidates_current_and_next() {
        let mut series = HashMap::new();
        series.insert("btc-updown-5m".to_string(), (300u64, "u1".to_string()));
        let known = HashSet::new();

        // now = 1789765607 → current boundary 1789765500, next 1789765800.
        let c = next_candidates(&series, &known, 1_789_765_607);
        assert_eq!(c.len(), 2);
        assert!(c.contains_key("btc-updown-5m-1789765500"));
        assert!(c.contains_key("btc-updown-5m-1789765800"));
        assert_eq!(c["btc-updown-5m-1789765500"], "u1");
    }

    /// Those already registered are not queried again.
    #[test]
    fn candidates_excludes_registered() {
        let mut series = HashMap::new();
        series.insert("eth-updown-15m".to_string(), (900u64, "u2".to_string()));
        let mut known = HashSet::new();
        known.insert("eth-updown-15m-1789764300".to_string());

        // now = 1789764500 → base 1789764300 (registered), next 1789765200.
        let c = next_candidates(&series, &known, 1_789_764_500);
        assert_eq!(c.len(), 1);
        assert!(c.contains_key("eth-updown-15m-1789765200"));
    }
}
