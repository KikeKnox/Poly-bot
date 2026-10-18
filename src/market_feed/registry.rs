use chrono::{DateTime, Utc};
use dashmap::{DashMap, DashSet};
use rust_decimal::Decimal;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::types::Level;

#[derive(Debug, Clone)]
pub struct TokenState {
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
    pub updated_at: DateTime<Utc>,
}

/// best_bid >= best_ask — invalid state that must never be emitted to the
/// strategies. Shared between the WS delta (`handle.rs`) and the REST resync
/// (`resync.rs`): both sources can produce a crossed book and must
/// apply exactly the same criterion before accepting it as valid.
pub fn book_is_crossed(bids: &[Level], asks: &[Level]) -> bool {
    match (bids.first(), asks.first()) {
        (Some(b), Some(a)) => b.price >= a.price,
        _ => false,
    }
}

#[cfg(test)]
mod book_is_crossed_tests {
    use super::*;

    fn level(p: &str) -> Level {
        Level {
            price: p.parse().unwrap(),
            size: rust_decimal_macros::dec!(1),
        }
    }

    #[test]
    fn bid_less_than_ask_is_not_crossed() {
        assert!(!book_is_crossed(&[level("0.40")], &[level("0.45")]));
    }

    #[test]
    fn bid_greater_or_equal_to_ask_is_crossed() {
        assert!(book_is_crossed(&[level("0.45")], &[level("0.45")]));
        assert!(book_is_crossed(&[level("0.50")], &[level("0.45")]));
    }

    #[test]
    fn an_empty_side_is_not_crossed() {
        assert!(!book_is_crossed(&[], &[level("0.45")]));
        assert!(!book_is_crossed(&[level("0.45")], &[]));
    }
}

#[derive(Debug, Clone)]
pub struct MarketEntry {
    pub slug: String,
    pub yes_token_id: String,
    pub no_token_id: String,
    pub clear_at: i64,
    pub market_open_ts: i64,
    pub neg_risk: bool,
    /// Underlying price at bracket open (from the Gamma API if available).
    pub reference_price: Option<Decimal>,
    /// Taker fee in basis points (0 = no fee). `None` if the Gamma API did not return it.
    pub fee_rate_bps: Option<u32>,
    /// Real rate of the market's parabolic taker fee,
    /// read from the Gamma API's `feeSchedule.rate`. At least
    /// two verified regimes coexist (`crypto_fees_v2` with rate=0.07,
    /// `finance_prices_fees` with rate=0.04) — using the fixed constant
    /// `TAKER_FEE_RATE_PARABOLIC` for all overestimates the fee ~75% on
    /// markets of the second family. `None` if Gamma has no `feeSchedule`
    /// (falls back to the constant).
    pub parabolic_fee_rate: Option<Decimal>,
    /// Last tick reported by the WS tick_size_change event; None until
    /// the first event.
    pub tick_size: Option<Decimal>,
    /// Minimum size (shares) an order must have to qualify for the
    /// Liquidity Rewards Program (Gamma `rewardsMinSize`). The Gamma API reports 0
    /// when the market has no active rewards program — translated to
    /// `None` in discovery (never a faked 0; see `market_feed::discover`).
    pub rewards_min_size: Option<Decimal>,
    /// Maximum spread allowed from the adjusted midpoint for an
    /// order to qualify (Gamma `rewardsMaxSpread`, in CENTS — see
    /// docs.polymarket.com/market-makers/liquidity-rewards). `None` = no
    /// rewards program.
    pub rewards_max_spread: Option<Decimal>,
    /// Daily rate of the market's rewards pool, if the Gamma API
    /// returns it in `clobRewards[].rewardsDailyRate` (observed `null` on
    /// crypto updown even though min_size/max_spread are populated — signal
    /// absent, never blocks the scan).
    pub rewards_daily_rate: Option<Decimal>,
}

/// Which universe a market belongs to and whether its
/// asset has an oracle feed (Binance). Lives OUTSIDE `MarketEntry` (apart
/// from what the design asks for) because `reward_farm.rs` builds exhaustive
/// `MarketEntry` literals in its tests and that file is out of
/// scope for this change — a new field there would break its compilation.
/// `discover.rs` populates this table alongside each new `MarketEntry`; the
/// dispatch of `strategy::run` queries it by `market_condition_id`.
#[derive(Debug, Clone)]
pub struct UniverseInfo {
    pub universe: String,
    pub has_oracle: bool,
}

/// Effective bracket tick = min(tick derived from the book's bests,
/// tick reported by the last WS `tick_size_change` event) — the finer one wins.
/// A tick that is too fine is harmlessly truncated in clob_sdk (authoritative backstop);
/// one that is too coarse would reject prices such as 0.999. Shared between the
/// delta emitter (`handle.rs`) and the REST resync one (`resync.rs`).
///
/// Note, not fixed here — depends on work out of
/// scope for this change: the `min()` is unidirectional by design — a
/// `tick_size_change` that COARSENS the tick is never reflected if `derived` (from the
/// book) is still finer. Same root pattern as the unidirectional tick cache
/// of `clob_sdk.rs`; don't touch without coordinating both fixes.
pub(crate) fn effective_tick(bests: &[Option<Decimal>], ws_tick: Option<Decimal>) -> Decimal {
    let derived = bests
        .iter()
        .filter_map(|p| *p)
        .map(crate::types::tick_from_price)
        .min()
        .unwrap_or(rust_decimal_macros::dec!(0.01));
    match ws_tick {
        Some(t) => derived.min(t),
        None => derived,
    }
}

pub struct MarketRegistry {
    pub markets: Arc<DashMap<String, MarketEntry>>,
    pub token_to_cid: Arc<DashMap<String, String>>,
    pub token_books: Arc<DashMap<String, TokenState>>,
    /// Tokens that must receive a fresh `book` snapshot before `price_change`
    /// messages are applied. Set on every reconnect; cleared on snapshot receipt.
    pub needs_snapshot: Arc<DashSet<String>>,
    /// Last top-of-book emitted per CID: [yes_bid, yes_ask, no_bid, no_ask].
    /// Used to suppress broadcasts when only depth changes (not the top).
    pub last_top: Arc<DashMap<String, [Option<Decimal>; 4]>>,
    /// Tokens whose most recent `price_change` left the book crossed.
    /// Most crossings are transient (a trade arrives as two
    /// entries — remove from the ask, add to the bid — spread over different
    /// WS messages) and self-correct with the NEXT delta of that same token.
    /// Instead of guessing how long that delta would take with a timeout, we wait for
    /// the real signal: if the next `price_change` of this token is still crossed,
    /// then it is a genuine desync and is escalated to resync.
    pub crossed_pending: Arc<DashSet<String>>,
    /// Per-token exponential backoff on the
    /// "persistent crossing → resync" escalation. Markets that OSCILLATE crossed on the venue
    /// (resolution, extreme top-of-book churn) re-cross the book a few
    /// deltas after each fresh snapshot — without backoff, each
    /// resync→recross→WARN cycle lasted ~4.5s per token (53 WARN/4min on a single
    /// token; REST confirmed uncrossed or empty books:
    /// the crossing is the market, not a desync). The backoff spaces out
    /// the noisy token's re-escalations (5s→300s) and lets the WS stream
    /// self-heal in place of hammering REST over venue noise.
    /// A real desync is repaired all the same — just spaced out.
    /// Reset when a delta leaves the book uncrossed (`Some(false)`)
    /// and when the market is evicted; NOT on authoritative writes —
    /// resetting there would defeat the throttling on the oscillating token.
    pub crossed_escalation: Arc<DashMap<String, CrossedEscalation>>,
    /// Trade tape per token — `(ts_ms, signed_size)`, sign = side
    /// of the TAKER (+buy, −sell). Auto-pruned in `record_trade` to 2× the
    /// query window; also cleared in `evict_expired` (memory leak
    /// otherwise — there is a new bracket every ~5 min).
    pub token_flow: Arc<DashMap<String, VecDeque<(i64, Decimal)>>>,
    /// Universe + has_oracle per condition_id.
    /// See the `UniverseInfo` doc for why it lives apart from `MarketEntry`.
    pub universe_info: Arc<DashMap<String, UniverseInfo>>,
    /// LONG-lived cache (independent of the working set of
    /// `markets`, which has a short TTL via `evict_grace_secs`) with the 4 minimum
    /// fields that `user_ws::handle_trade` needs to rebuild a `Fill`:
    /// `(neg_risk, clear_at, yes_token_id, no_token_id)`. Populated at the same
    /// point as `markets` (`discover::register_if_new`) and survives much longer
    /// than the working set — so a legitimate fill from the CURRENT session (e.g. the
    /// late GTD of post_close_sweep) that arrives after `evict_expired`
    /// already evicted the market from the short working set still finds its data,
    /// instead of being dropped (live) or recording `clear_at=0` in limbo (paper).
    /// Starts empty on each process restart — like `markets` — so it does
    /// NOT reintroduce the original problem that motivated the registry guard: a
    /// fill from a PREVIOUS session (before a crash) still has no entry
    /// here and is dropped just as today.
    pub clear_at_cache: Arc<DashMap<String, (bool, i64, String, String)>>,
    /// Insertion instant of each `condition_id` in `markets`.
    /// Lives apart from `MarketEntry` for the same reason as `UniverseInfo`
    /// (avoid breaking exhaustive `MarketEntry` literals out of scope,
    /// e.g. in `reward_farm.rs`). Only queried for `scan=true` universes
    /// WITHOUT an `end_date` ceiling (`rewards-longdated`): those are never evicted
    /// by `clear_at` (it may be years in the future) and without this extra purge
    /// the working set grows unbounded with uptime.
    pub scan_inserted_at: Arc<DashMap<String, DateTime<Utc>>>,
    /// Universes whose tokens go over WebSocket (real time). The other
    /// universes (scan-only, with no enabled strategy declaring them)
    /// are served by batched REST poll — the venue limits by TOTAL subscribed
    /// tokens, so the ~1,200 tokens nobody trades are taken out of the stream.
    /// Seeded at boot with the universes declared by `enabled=true`
    /// strategies and `StrategySupervisor::set_enabled` keeps it updated
    /// live: enabling a strategy on a scan universe migrates it to the
    /// WS in ~1 RTT (dynamic `operation:subscribe` — an earlier failure turned out to be an
    /// artifact of the `type:subscribe` format, verified with a live probe).
    pub ws_universes: Arc<DashSet<String>>,
}

/// State of the "persistent crossing → resync" escalation backoff.
#[derive(Debug, Default, Clone, Copy)]
pub struct CrossedEscalation {
    /// Consecutive escalations — cooldown exponent.
    pub streak: u32,
    /// Instant from which the token may escalate to resync again.
    /// `None` = never escalated (the first escalation is immediate).
    pub retry_at: Option<crate::time::Instant>,
}

/// Wait after escalation number `streak` — 5s × 2^streak, cap 300s.
/// Same shape as the per-token cooldown of the REST sweeper (resync.rs).
pub(crate) fn crossed_escalation_wait(streak: u32) -> std::time::Duration {
    const BASE_MS: u64 = 5_000;
    const MAX_MS: u64 = 300_000;
    let scaled = BASE_MS.checked_mul(1u64 << streak.min(6)).unwrap_or(MAX_MS);
    std::time::Duration::from_millis(scaled.min(MAX_MS))
}

impl Default for MarketRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MarketRegistry {
    pub fn new() -> Self {
        Self {
            markets: Arc::new(DashMap::new()),
            token_to_cid: Arc::new(DashMap::new()),
            token_books: Arc::new(DashMap::new()),
            needs_snapshot: Arc::new(DashSet::new()),
            last_top: Arc::new(DashMap::new()),
            crossed_pending: Arc::new(DashSet::new()),
            crossed_escalation: Arc::new(DashMap::new()),
            token_flow: Arc::new(DashMap::new()),
            universe_info: Arc::new(DashMap::new()),
            clear_at_cache: Arc::new(DashMap::new()),
            scan_inserted_at: Arc::new(DashMap::new()),
            ws_universes: Arc::new(DashSet::new()),
        }
    }

    /// Registers (or refreshes) a market's entry in the long-lived
    /// `clear_at_cache`. Call from the same point where the corresponding
    /// `MarketEntry` is inserted into `markets`
    /// (`discover::register_if_new`).
    pub fn remember_clear_at(
        &self,
        cid: &str,
        neg_risk: bool,
        clear_at: i64,
        yes_token_id: &str,
        no_token_id: &str,
    ) {
        self.clear_at_cache.insert(
            cid.to_string(),
            (
                neg_risk,
                clear_at,
                yes_token_id.to_string(),
                no_token_id.to_string(),
            ),
        );
    }

    /// Records a tape trade (`last_trade_price`) for `token_id`.
    /// `signed_size` already carries the sign of the TAKER side (+buy, −sell). Prunes
    /// entries older than `retention_secs` (recommended 2× the query
    /// window, so as not to lose coverage right at the window edge).
    pub fn record_trade(
        &self,
        token_id: &str,
        ts_ms: i64,
        signed_size: Decimal,
        retention_secs: u64,
    ) {
        let mut deque = self.token_flow.entry(token_id.to_string()).or_default();
        deque.push_back((ts_ms, signed_size));
        let cutoff = ts_ms - (retention_secs as i64 * 1000);
        while deque.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
            deque.pop_front();
        }
    }

    /// Net and total flow (shares) of the token in the window
    /// `[now_ms − window_secs*1000, now_ms]`.
    ///
    /// Semantics: `None` = the token never had registered trades (signal
    /// absent); `Some((0, 0))` = there was history but no trade falls in the
    /// queried window — REAL information, not absence.
    pub fn flow_window(
        &self,
        token_id: &str,
        now_ms: i64,
        window_secs: u64,
    ) -> Option<(Decimal, Decimal)> {
        let deque = self.token_flow.get(token_id)?;
        let cutoff = now_ms - (window_secs as i64 * 1000);
        let mut net = Decimal::ZERO;
        let mut total = Decimal::ZERO;
        for (ts, signed) in deque.iter() {
            if *ts >= cutoff {
                net += *signed;
                total += signed.abs();
            }
        }
        Some((net, total))
    }

    /// Mark every currently tracked WS-bound token as stale so that
    /// `price_change` events are discarded until a fresh `book` snapshot
    /// arrives for each one. Call this after every WS disconnection.
    /// Tokens served by REST poll are never marked here — a WS cut
    /// doesn't degrade them and the poller would keep emitting regardless.
    pub fn mark_all_needs_snapshot(&self) {
        for t in self.ws_token_ids() {
            self.needs_snapshot.insert(t);
        }
        // Local state is discarded on reconnect — a pending crossing from
        // before the drop no longer has a direction (a fresh snapshot will be requested).
        self.crossed_pending.clear();
    }

    /// Is the token in a WS-bound universe? Without `universe_info` (discovery/eviction
    /// race) it is treated as WS — conservative: better to over-subscribe
    /// than to lose data of a tradable market.
    fn is_ws_bound(&self, token_id: &str) -> bool {
        let Some(cid) = self.token_to_cid.get(token_id) else {
            return true;
        };
        match self.universe_info.get(cid.as_str()) {
            Some(info) => self.ws_universes.contains(&info.universe),
            None => true,
        }
    }

    /// Tokens to subscribe on the market WS: only those of universes that an
    /// enabled strategy declares (or declared live via the supervisor).
    pub fn ws_token_ids(&self) -> Vec<String> {
        self.token_to_cid
            .iter()
            .filter(|e| self.is_ws_bound(e.key()))
            .map(|e| e.key().clone())
            .collect()
    }

    /// Tokens of scan-only universes with no real-time consumer — served by
    /// the batched REST poll (`poll.rs`).
    pub fn poll_token_ids(&self) -> Vec<String> {
        self.token_to_cid
            .iter()
            .filter(|e| !self.is_ws_bound(e.key()))
            .map(|e| e.key().clone())
            .collect()
    }

    /// Tokens of all markets whose `universe_info.universe` is in
    /// `universes` — used by the supervisor to migrate a universe to the WS when
    /// enabling a strategy live.
    pub fn token_ids_for_universes(
        &self,
        universes: &std::collections::HashSet<String>,
    ) -> Vec<String> {
        self.universe_info
            .iter()
            .filter(|e| universes.contains(&e.value().universe))
            .flat_map(|e| {
                let cid = e.key().clone();
                self.markets
                    .get(&cid)
                    .map(|m| vec![m.yes_token_id.clone(), m.no_token_id.clone()])
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Evicts the `condition_id` from ALL the working-set maps (shares
    /// logic between the two passes of `evict_expired`: by expired `clear_at` and,
    /// by insertion age in scan universes with no ceiling).
    /// Returns the evicted token_ids so the caller can
    /// unsubscribe them from the WS (`TokenEvent::Unsubscribe`) — releasing them
    /// continuously keeps the subscribed total bounded (venue throttle ∝ total).
    fn remove_market(&self, cid: &str) -> Vec<String> {
        let mut evicted = Vec::new();
        if let Some((_, entry)) = self.markets.remove(cid) {
            evicted.push(entry.yes_token_id.clone());
            evicted.push(entry.no_token_id.clone());
            self.token_to_cid.remove(&entry.yes_token_id);
            self.token_to_cid.remove(&entry.no_token_id);
            self.token_books.remove(&entry.yes_token_id);
            self.token_books.remove(&entry.no_token_id);
            self.needs_snapshot.remove(&entry.yes_token_id);
            self.needs_snapshot.remove(&entry.no_token_id);
            self.crossed_pending.remove(&entry.yes_token_id);
            self.crossed_pending.remove(&entry.no_token_id);
            // Same for the escalation backoff — without this the entry
            // of the evicted token is orphaned in the DashMap forever.
            self.crossed_escalation.remove(&entry.yes_token_id);
            self.crossed_escalation.remove(&entry.no_token_id);
            self.last_top.remove(cid);
            // Without this the evicted token's tape is orphaned in
            // the DashMap forever — memory leak (new bracket every ~5 min).
            self.token_flow.remove(&entry.yes_token_id);
            self.token_flow.remove(&entry.no_token_id);
            // Without this the evicted cid's universe_info is orphaned
            // forever (same pattern as the rest of this block's maps).
            self.universe_info.remove(cid);
            // Same for the insertion instant.
            self.scan_inserted_at.remove(cid);
            tracing::debug!(cid = %cid, "market_feed: evicted bracket");
        }
        evicted
    }

    /// Remove brackets whose `clear_at` is older than `grace_secs` seconds ago.
    /// Cleans up `markets`, `token_to_cid`, and `token_books` to prevent unbounded growth.
    /// Called from the discover loop after each Gamma poll.
    ///
    /// `clear_at_cache_ttl_secs` governs a second, much more lenient pass
    /// over `clear_at_cache` — it lives far longer than the working set
    /// (`grace_secs`) because each entry is a minimal tuple that's cheap to retain.
    ///
    /// `long_lived_universes` are the `scan=true` universes WITHOUT an
    /// `end_date` ceiling (e.g. `rewards-longdated`) — their real `clear_at` may
    /// be years in the future, so the `clear_at` purge above never
    /// reaches them. A THIRD pass evicts them by INSERTION age
    /// (`scan_insertion_ttl_secs`), independent of their `clear_at`.
    /// Returns the evicted token_ids — the caller unsubscribes them from the WS.
    pub fn evict_expired(
        &self,
        grace_secs: i64,
        clear_at_cache_ttl_secs: i64,
        long_lived_universes: &std::collections::HashSet<String>,
        scan_insertion_ttl_secs: i64,
    ) -> Vec<String> {
        use chrono::Utc;
        let cutoff = Utc::now().timestamp() - grace_secs;
        let mut evicted: Vec<String> = Vec::new();

        // Collect condition IDs to evict (avoid mutating while iterating).
        let expired: Vec<String> = self
            .markets
            .iter()
            .filter(|e| {
                let entry = e.value();
                entry.clear_at > 0 && entry.clear_at < cutoff
            })
            .map(|e| e.key().clone())
            .collect();

        for cid in expired {
            evicted.extend(self.remove_market(&cid));
        }

        // Separate pass over the long-lived cache, with its
        // own threshold (much greater than grace_secs) — independent of whether the
        // cid already left the working set above.
        let cache_cutoff = Utc::now().timestamp() - clear_at_cache_ttl_secs;
        self.clear_at_cache
            .retain(|_, (_, clear_at, _, _)| *clear_at >= cache_cutoff);

        // Third pass — scan universes with no end_date ceiling,
        // purged by INSERTION age (not by clear_at, which for these
        // never expires in a reasonable time).
        if scan_insertion_ttl_secs > 0 && !long_lived_universes.is_empty() {
            let now = Utc::now();
            let stale_by_age: Vec<String> = self
                .universe_info
                .iter()
                .filter(|e| long_lived_universes.contains(&e.value().universe))
                .filter_map(|e| {
                    let cid = e.key().clone();
                    let inserted = *self.scan_inserted_at.get(&cid)?;
                    let age_secs = (now - inserted).num_seconds();
                    (age_secs >= scan_insertion_ttl_secs).then_some(cid)
                })
                .collect();
            for cid in stale_by_age {
                tracing::debug!(cid = %cid, "market_feed: evicted long-lived scan bracket by insertion age");
                evicted.extend(self.remove_market(&cid));
            }
        }
        evicted
    }
}

#[cfg(test)]
mod evict_expired_insertion_age_tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    fn entry(clear_at: i64) -> MarketEntry {
        MarketEntry {
            slug: "will-btc-hit-150k".into(), // no slug duration — non-updown, far-off clear_at typical of rewards-longdated
            yes_token_id: "yes".into(),
            no_token_id: "no".into(),
            clear_at,
            market_open_ts: 0,
            neg_risk: false,
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            rewards_min_size: None,
            rewards_max_spread: None,
            rewards_daily_rate: None,
        }
    }

    /// A market of a `scan=true` universe WITHOUT an
    /// `end_date` ceiling (e.g. `rewards-longdated`) with `clear_at` years in the future
    /// is NEVER evicted by the `clear_at` purge — but it MUST be evicted
    /// by insertion age if the universe is in `long_lived_universes`.
    #[test]
    fn purge_by_insertion_age_in_scan_universe_without_ceiling() {
        let registry = MarketRegistry::new();
        let cid = "cid-longdated";
        registry
            .markets
            .insert(cid.to_string(), entry(9_999_999_999)); // clear_at very far off
        registry.token_to_cid.insert("yes".into(), cid.to_string());
        registry.token_to_cid.insert("no".into(), cid.to_string());
        registry.universe_info.insert(
            cid.to_string(),
            UniverseInfo {
                universe: "rewards-longdated".to_string(),
                has_oracle: false,
            },
        );
        // Inserted 10 days ago — much more than the test's 1-day TTL.
        let inserted_10d_ago = Utc::now() - ChronoDuration::days(10);
        registry
            .scan_inserted_at
            .insert(cid.to_string(), inserted_10d_ago);

        let mut long_lived = std::collections::HashSet::new();
        long_lived.insert("rewards-longdated".to_string());

        // generous grace_secs/clear_at_cache_ttl_secs (they must not be the ones
        // that evict here) — the insertion TTL is 1 day (86400s).
        registry.evict_expired(60, 86_400, &long_lived, 86_400);

        assert!(!registry.markets.contains_key(cid),
            "a scan market with no ceiling must be purged by insertion age even if its clear_at is still far in the future");
        assert!(!registry.universe_info.contains_key(cid));
        assert!(!registry.scan_inserted_at.contains_key(cid));
    }

    /// A market of the SAME universe but inserted RECENTLY must
    /// not be purged yet — the purge is by age, not by belonging to the
    /// universe.
    #[test]
    fn does_not_purge_recent_market_of_same_scan_universe() {
        let registry = MarketRegistry::new();
        let cid = "cid-recent";
        registry
            .markets
            .insert(cid.to_string(), entry(9_999_999_999));
        registry.token_to_cid.insert("yes".into(), cid.to_string());
        registry.token_to_cid.insert("no".into(), cid.to_string());
        registry.universe_info.insert(
            cid.to_string(),
            UniverseInfo {
                universe: "rewards-longdated".to_string(),
                has_oracle: false,
            },
        );
        registry
            .scan_inserted_at
            .insert(cid.to_string(), Utc::now());

        let mut long_lived = std::collections::HashSet::new();
        long_lived.insert("rewards-longdated".to_string());

        registry.evict_expired(60, 86_400, &long_lived, 86_400);

        assert!(
            registry.markets.contains_key(cid),
            "a market inserted seconds ago must not be purged yet"
        );
    }

    /// A market of a universe that is NOT in `long_lived_universes`
    /// (e.g. a normal `slug` universe, with an end_date ceiling) is never purged
    /// by insertion age — only the usual `clear_at` purge applies.
    #[test]
    fn does_not_purge_on_insert_if_universe_is_not_long_lived() {
        let registry = MarketRegistry::new();
        let cid = "cid-normal";
        registry
            .markets
            .insert(cid.to_string(), entry(9_999_999_999));
        registry.token_to_cid.insert("yes".into(), cid.to_string());
        registry.token_to_cid.insert("no".into(), cid.to_string());
        registry.universe_info.insert(
            cid.to_string(),
            UniverseInfo {
                universe: "crypto-updown-long".to_string(),
                has_oracle: true,
            },
        );
        registry
            .scan_inserted_at
            .insert(cid.to_string(), Utc::now() - ChronoDuration::days(10));

        // "crypto-updown-long" is NOT in long_lived_universes.
        let long_lived = std::collections::HashSet::new();
        registry.evict_expired(60, 86_400, &long_lived, 86_400);

        assert!(registry.markets.contains_key(cid),
            "a universe outside long_lived_universes must not be purged by insertion age");
    }
}
