//! "Mark to market" price cache used to compute real equity (cash + value
//! of open positions at the current price) — not just REALIZED PnL like
//! `pnl_series`. It is the number that actually answers "am I winning or
//! losing right now?", analogous to the total portfolio value Polymarket
//! itself shows.
//!
//! Refreshed by a periodic background task, NEVER on the path of a UI HTTP
//! request: with tens or hundreds of open follower positions (which does not
//! live in `MarketRegistry` by design, see the `strategy::follower` doc),
//! fetching the price on every dashboard load would trigger a burst of calls
//! to the CLOB with a real rate-limit risk — it could even affect the follower
//! itself, which hits the same endpoint for its own buy/sell decisions.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::market_feed::registry::MarketRegistry;
use crate::state::StateStore;
use crate::types::Level;

const CLOB_BASE: &str = "https://clob.polymarket.com";
const REFRESH_INTERVAL_SECS: u64 = 45;
/// Every how many refresh cycles an equity snapshot is persisted — 7 × 45s
/// ≈ 5.25 min, similar to the bot's other time series (nothing critical
/// needs more resolution than that for an equity curve).
const SNAPSHOT_INTERVAL_CYCLES: u32 = 7;
/// Throttle between individual CLOB calls within a single cycle — with
/// many untracked tokens (follower), it avoids a burst of simultaneous
/// requests to the same host.
const FETCH_STAGGER_MS: u64 = 120;

/// Maximum acceptable age of a mark. The refresh runs every
/// `REFRESH_INTERVAL_SECS` (45 s); three cycles (~135 s) tolerate up to two
/// consecutive network/registry failures without degrading — past that
/// deadline the cached mid stops being evidence and `get` returns `None` so
/// the caller falls back to `avg_price` (never to an inferred outcome or a
/// fictitious price). The main scenario — a resolved/evicted market with an
/// empty book — is resolved BEFORE the TTL by explicit invalidation in
/// `refresh_once`; the TTL is the net for silent staleness.
const MARK_TTL: Duration = Duration::from_secs(REFRESH_INTERVAL_SECS * 3);

/// Reference price (book mid) per token_id. Each entry carries the
/// MONOTONIC instant at which the DATA was observed (not when it was re-read) —
/// `get()` returns `None` if the mark exceeded `MARK_TTL` and does not distinguish
/// "could never be obtained" from "stale": the caller decides the fallback
/// (the position's avg_price is reasonable: it is what was actually paid).
pub struct MarkPriceCache {
    prices: DashMap<String, (Decimal, Instant)>,
}

impl MarkPriceCache {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            prices: DashMap::new(),
        })
    }

    /// Current mark of the token, or `None` if absent or past `MARK_TTL`.
    /// A stale mid is NOT a price — the caller's honest fallback
    /// (`avg_price`) only applies when the mark is absent or expired.
    pub fn get(&self, token_id: &str) -> Option<Decimal> {
        self.prices.get(token_id).and_then(|v| {
            let (price, seen) = *v;
            (seen.elapsed() <= MARK_TTL).then_some(price)
        })
    }

    /// Mark observed NOW (fresh REST response from the CLOB).
    fn set(&self, token_id: String, price: Decimal) {
        self.set_observed(token_id, price, Instant::now());
    }

    /// Mark with the real observation instant of the data — different from `set`
    /// when the source was already stale when re-read (a registry
    /// `TokenState` whose `updated_at` is stale does not rejuvenate the
    /// mark no matter how often the refresh re-reads it each cycle).
    fn set_observed(&self, token_id: String, price: Decimal, seen: Instant) {
        self.prices.insert(token_id, (price, seen));
    }

    /// Discards the mark — the source reported "no book" (market
    /// resolved/evicted). Keeping the last mid would feign a price that no
    /// longer exists; without a mark, `compute_equity` falls back to `avg_price`.
    fn invalidate(&self, token_id: &str) {
        self.prices.remove(token_id);
    }
}

/// Translates `updated_at` (wall clock of the registry's `TokenState`) to the
/// monotonic `Instant` with the SAME age, to stamp the mark with the
/// real freshness of the data. `None` = no reliable freshness evidence:
/// `updated_at` in the future (corrupt clock — not accepted as fresh) or
/// an age the monotonic clock cannot represent.
fn seen_of(updated_at: DateTime<Utc>) -> Option<Instant> {
    let age = Utc::now().signed_duration_since(updated_at).to_std().ok()?;
    Instant::now().checked_sub(age)
}

fn mid_from_levels(bids: &[Level], asks: &[Level]) -> Option<Decimal> {
    match (bids.first(), asks.first()) {
        (Some(b), Some(a)) => Some((b.price + a.price) / Decimal::TWO),
        (Some(b), None) => Some(b.price),
        (None, Some(a)) => Some(a.price),
        (None, None) => None,
    }
}

// ── CLOB book (same pattern as strategy::follower::fetch_book — not
// worth exposing that private function just for this) ───────────────────

#[derive(Debug, Deserialize)]
struct RawBookLevel {
    price: String,
    size: String,
}

#[derive(Debug, Deserialize)]
struct ClobBookResponse {
    #[serde(default)]
    bids: Vec<RawBookLevel>,
    #[serde(default)]
    asks: Vec<RawBookLevel>,
}

fn parse_levels(raw: &[RawBookLevel]) -> Vec<Level> {
    use std::str::FromStr;
    raw.iter()
        .filter_map(|l| {
            let price = Decimal::from_str(&l.price).ok()?;
            let size = Decimal::from_str(&l.size).ok()?;
            if size <= Decimal::ZERO {
                return None;
            }
            Some(Level { price, size })
        })
        .collect()
}

/// The API does not guarantee "best price first" (verified with real data:
/// it returns bids in ASCENDING order, worst first) — without sorting,
/// `.first()` in `mid_from_levels` picks a garbage price from the wrong
/// end (real bug: it inflated positions_value ~1.8× over the real cost
/// on the first run with follower positions). Same sorting criterion
/// as `strategy::follower::fetch_book`. Pure function (no I/O)
/// so the scenario can be tested with unordered data.
fn mid_from_raw(raw_bids: &[RawBookLevel], raw_asks: &[RawBookLevel]) -> Option<Decimal> {
    let mut bids = parse_levels(raw_bids);
    let mut asks = parse_levels(raw_asks);
    bids.sort_unstable_by_key(|l| std::cmp::Reverse(l.price)); // desc: best bid first
    asks.sort_unstable_by_key(|a| a.price); // asc: best ask first
    mid_from_levels(&bids, &asks)
}

async fn fetch_mid_from_clob(
    http: &reqwest::Client,
    token_id: &str,
) -> anyhow::Result<Option<Decimal>> {
    let body: ClobBookResponse = http
        .get(format!("{CLOB_BASE}/book"))
        .query(&[("token_id", token_id)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(mid_from_raw(&body.bids, &body.asks))
}

/// One refresh cycle: the price of each distinct token with an open position.
/// The free path takes priority (`MarketRegistry`, already in memory because the
/// bot tracks those markets); only hits the CLOB over REST for the
/// tokens it does not track (follower positions, by design).
async fn refresh_once(
    cache: &MarkPriceCache,
    registry: &MarketRegistry,
    http: &reqwest::Client,
    token_ids: &[String],
) {
    for token_id in token_ids {
        if let Some(ts) = registry.token_books.get(token_id) {
            match mid_from_levels(&ts.bids, &ts.asks) {
                Some(mid) => match seen_of(ts.updated_at) {
                    // The mark is stamped with the freshness of the DATA
                    // (`updated_at`), not that of this re-read — re-reading the
                    // same stale TokenState every 45 s does not rejuvenate it.
                    Some(seen) => cache.set_observed(token_id.clone(), mid, seen),
                    None => debug!(
                        token_id,
                        "mark_price: updated_at unreliable (future or not representable) — mark not renewed"
                    ),
                },
                // Registry with an empty book — no reliable price, the
                // previous mark is not kept.
                None => cache.invalidate(token_id),
            }
            continue;
        }
        match fetch_mid_from_clob(http, token_id).await {
            Ok(Some(mid)) => cache.set(token_id.clone(), mid),
            // The venue responds with an empty book (market resolved/evicted)
            // — the mark stops being evidence: it is invalidated and `get` falls to the
            // caller's fallback (`avg_price`), never to an inferred outcome.
            Ok(None) => cache.invalidate(token_id),
            Err(e) => {
                // Transient error: keeps the previous mark ONLY until its
                // TTL — `get` expires it even if the refresh keeps failing.
                debug!(token_id, error = %e, "mark_price: failed querying book — mark kept until TTL")
            }
        }
        tokio::time::sleep(Duration::from_millis(FETCH_STAGGER_MS)).await;
    }
}

/// Cash (deposits − open cost + realized PnL) + market value of the
/// open positions = real equity. Exposed so `/ui/data` can compute it
/// on demand reading ONLY the cache (no network I/O) — network I/O lives
/// exclusively in `run()`.
pub async fn compute_equity(
    state: &StateStore,
    cache: &MarkPriceCache,
) -> anyhow::Result<(Decimal, Decimal, Decimal)> {
    let (deposits, cost, pnl_total, positions) = tokio::try_join!(
        state.deposits_total(),
        state.total_open_cost(),
        state.realized_pnl_total(),
        state.list_open_positions(),
    )?;
    let cash = deposits - cost + pnl_total;
    let positions_value: Decimal = positions
        .iter()
        .map(|p| cache.get(&p.token_id).unwrap_or(p.avg_price) * p.size)
        .sum();
    let equity = cash + positions_value;
    Ok((cash, positions_value, equity))
}

pub async fn run(
    cache: Arc<MarkPriceCache>,
    state: Arc<StateStore>,
    registry: Arc<MarketRegistry>,
    mut shutdown: watch::Receiver<bool>,
) {
    let http = reqwest::Client::builder()
        .user_agent(crate::POLY_USER_AGENT)
        .timeout(Duration::from_secs(10))
        .build()
        .expect("mark_price: reqwest client");

    info!("mark_price: equity cache started (refresh every {REFRESH_INTERVAL_SECS}s)");

    let mut ticker = tokio::time::interval(Duration::from_secs(REFRESH_INTERVAL_SECS));
    ticker.tick().await; // skip the first immediate tick
    let mut cycle: u32 = 0;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => { if *shutdown.borrow() { break; } }
            _ = ticker.tick() => {
                let positions = match state.list_open_positions().await {
                    Ok(p) => p,
                    Err(e) => { warn!(error = %e, "mark_price: failed listing positions — cycle skipped"); continue; }
                };
                let mut token_ids: Vec<String> = positions.iter().map(|p| p.token_id.clone()).collect();
                token_ids.sort_unstable();
                token_ids.dedup();

                refresh_once(&cache, &registry, &http, &token_ids).await;

                cycle += 1;
                if cycle.is_multiple_of(SNAPSHOT_INTERVAL_CYCLES) {
                    match compute_equity(&state, &cache).await {
                        Ok((cash, positions_value, equity)) => {
                            if let Err(e) = state.record_equity_snapshot(
                                chrono::Utc::now().timestamp_millis(), cash, positions_value, equity,
                            ).await {
                                warn!(error = %e, "mark_price: failed saving equity snapshot");
                            }
                        }
                        Err(e) => warn!(error = %e, "mark_price: failed computing equity for snapshot"),
                    }
                }
            }
        }
    }
    info!("mark_price: equity cache stopped");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn lvl(price: Decimal, size: Decimal) -> Level {
        Level { price, size }
    }

    fn raw(price: &str, size: &str) -> RawBookLevel {
        RawBookLevel {
            price: price.into(),
            size: size.into(),
        }
    }

    /// Regression: the API returns bids in ASCENDING order (worst first),
    /// verified with real CLOB data (token with a one-sided book).
    /// Without sorting, taking `.first()` gives 0.001 instead of the real bid of 0.999.
    #[test]
    fn mid_from_raw_sorts_ascending_bids_from_real_api() {
        let bids = vec![
            raw("0.001", "99999"),
            raw("0.002", "5000"),
            raw("0.999", "417.79"),
        ];
        assert_eq!(mid_from_raw(&bids, &[]), Some(dec!(0.999)));
    }

    #[test]
    fn mid_from_raw_sorts_descending_asks() {
        let asks = vec![raw("0.60", "10"), raw("0.50", "10"), raw("0.90", "10")];
        assert_eq!(mid_from_raw(&[], &asks), Some(dec!(0.50)));
    }

    #[test]
    fn mid_with_both_sides() {
        let bids = vec![lvl(dec!(0.40), dec!(10))];
        let asks = vec![lvl(dec!(0.44), dec!(10))];
        assert_eq!(mid_from_levels(&bids, &asks), Some(dec!(0.42)));
    }

    #[test]
    fn mid_with_single_side_uses_that_side() {
        let bids = vec![lvl(dec!(0.40), dec!(10))];
        assert_eq!(mid_from_levels(&bids, &[]), Some(dec!(0.40)));
        let asks = vec![lvl(dec!(0.44), dec!(10))];
        assert_eq!(mid_from_levels(&[], &asks), Some(dec!(0.44)));
    }

    #[test]
    fn mid_with_no_side_is_none() {
        assert_eq!(mid_from_levels(&[], &[]), None);
    }

    #[test]
    fn cache_get_without_refresh_is_none() {
        let cache = MarkPriceCache::new();
        assert_eq!(cache.get("tok-1"), None);
    }

    #[test]
    fn cache_set_then_get_returns_price() {
        let cache = MarkPriceCache::new();
        cache.set("tok-1".into(), dec!(0.55));
        assert_eq!(cache.get("tok-1"), Some(dec!(0.55)));
    }

    // ── Freshness of the marks ──────────────────────────────────────────────

    use crate::market_feed::registry::TokenState;
    use crate::types::{Fill, FillSide, Mode, StrategyId};

    fn token_state(bid: &str, ask: &str, updated_at: DateTime<Utc>) -> TokenState {
        TokenState {
            bids: vec![lvl(bid.parse().unwrap(), dec!(10))],
            asks: vec![lvl(ask.parse().unwrap(), dec!(10))],
            updated_at,
        }
    }

    /// Instant already expired relative to MARK_TTL (local test helper — the
    /// `seen` field is private by design; the module's tests can set it).
    fn stale_seen() -> Instant {
        Instant::now()
            .checked_sub(MARK_TTL + Duration::from_secs(1))
            .expect("the monotonic clock exceeds MARK_TTL of uptime")
    }

    #[test]
    fn fresh_mark_returns_price() {
        let cache = MarkPriceCache::new();
        cache.set("tok-1".into(), dec!(0.55));
        assert_eq!(cache.get("tok-1"), Some(dec!(0.55)));
    }

    #[test]
    fn stale_mark_returns_none() {
        // Mark written but already past its TTL → absent for the caller's purposes.
        let cache = MarkPriceCache::new();
        cache.set_observed("tok-1".into(), dec!(0.55), stale_seen());
        assert_eq!(
            cache.get("tok-1"),
            None,
            "a mark past MARK_TTL stops being a valid price"
        );
    }

    #[tokio::test]
    async fn registry_with_empty_book_invalidates_mark() {
        // Token present in the registry but with an empty book → no reliable
        // price: the previous mark is discarded (not kept "just in case").
        let cache = MarkPriceCache::new();
        let registry = MarketRegistry::new();
        let http = reqwest::Client::new();
        cache.set("tok-1".into(), dec!(0.55));
        registry.token_books.insert(
            "tok-1".into(),
            TokenState {
                bids: vec![],
                asks: vec![],
                updated_at: Utc::now(),
            },
        );

        refresh_once(&cache, &registry, &http, &["tok-1".to_string()]).await;
        assert_eq!(cache.get("tok-1"), None);
    }

    #[tokio::test]
    async fn fresh_registry_updates_mark() {
        let cache = MarkPriceCache::new();
        let registry = MarketRegistry::new();
        let http = reqwest::Client::new();
        registry
            .token_books
            .insert("tok-1".into(), token_state("0.60", "0.62", Utc::now()));

        refresh_once(&cache, &registry, &http, &["tok-1".to_string()]).await;
        assert_eq!(cache.get("tok-1"), Some(dec!(0.61)));
    }

    #[tokio::test]
    async fn stale_registry_does_not_rejuvenate_mark() {
        // The bracket resolved but the TokenState is still in the registry with an
        // updated_at already past the TTL: re-reading it each cycle must NOT keep
        // the price alive forever — the mark is stamped with the real age
        // of the data and `get` expires it.
        let cache = MarkPriceCache::new();
        let registry = MarketRegistry::new();
        let http = reqwest::Client::new();
        let stale_ts = Utc::now() - chrono::Duration::seconds((MARK_TTL.as_secs() + 60) as i64);
        registry
            .token_books
            .insert("tok-1".into(), token_state("0.60", "0.62", stale_ts));

        refresh_once(&cache, &registry, &http, &["tok-1".to_string()]).await;
        assert_eq!(
            cache.get("tok-1"),
            None,
            "a stale TokenState does not produce a fresh mark"
        );
    }

    #[tokio::test]
    async fn future_updated_at_does_not_renew_mark() {
        // Corrupt wall clock: a future `updated_at` is not evidence
        // of freshness — the stale mark is not renewed with that data.
        let cache = MarkPriceCache::new();
        let registry = MarketRegistry::new();
        let http = reqwest::Client::new();
        cache.set_observed("tok-1".into(), dec!(0.55), stale_seen());
        let future = Utc::now() + chrono::Duration::hours(1);
        registry
            .token_books
            .insert("tok-1".into(), token_state("0.60", "0.62", future));

        refresh_once(&cache, &registry, &http, &["tok-1".to_string()]).await;
        assert_eq!(
            cache.get("tok-1"),
            None,
            "future updated_at does not rejuvenate an already expired mark"
        );
    }

    /// `invalidate` is what the "empty book" paths of refresh_once call
    /// (registry and REST Ok(None)) — the mark disappears and the caller falls to its
    /// fallback. The REST path is only exercised with a real network; here the
    /// primitive's contract is pinned down.
    #[test]
    fn invalidate_discards_mark() {
        let cache = MarkPriceCache::new();
        cache.set("tok-1".into(), dec!(0.55));
        cache.invalidate("tok-1");
        assert_eq!(cache.get("tok-1"), None);
    }

    /// Fallback evidence: with a stale mark, `compute_equity` values
    /// the position at `avg_price` (what was paid), not at the expired mid nor at an
    /// invented outcome.
    #[tokio::test]
    async fn equity_with_stale_mark_falls_back_to_avg_price() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        // Open position: 10 shares at 0.50.
        state
            .apply_fill(&Fill {
                hedge_id: None,
                exec_id: Some("exec-eq".into()),
                local_order_id: None,
                market_cid: "cid-eq".into(),
                token_id: "tok-eq".into(),
                side: FillSide::Buy,
                price: dec!(0.50),
                size: dec!(10),
                order_id: None,
                ts: Utc::now(),
                mode: Mode::Paper,
                neg_risk: false,
                strategy: StrategyId::NearCertain,
                clear_at: Utc::now().timestamp() + 3600,
                outcome_label: None,
            })
            .await
            .unwrap();

        let cache = MarkPriceCache::new();
        // Stale mark of 0.99 — if get returned it, positions_value
        // would be 9.90; with the honest fallback it must be 5.00 (avg_price).
        cache.set_observed("tok-eq".into(), dec!(0.99), stale_seen());

        let (_cash, positions_value, _equity) = compute_equity(&state, &cache).await.unwrap();
        assert_eq!(
            positions_value,
            dec!(5),
            "stale mark → fallback to avg_price (10 × 0.50), not to the expired mid"
        );
    }
}
