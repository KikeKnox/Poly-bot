use chrono::{DateTime, Utc};
use futures_util::stream::StreamExt as _;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;

use super::events::TokenEvent;
use super::registry::{MarketEntry, MarketRegistry, UniverseInfo};
use crate::config::{Config, UniverseConfig};
use crate::oracle::asset_from_slug;
use crate::research::contract::{
    EventKind, EventSource, MarketMeta, ResolutionRule, RulesWindow, SCHEMA_VERSION,
};
use crate::research::events::{stable_hash, EventInput, EventProducer};
use crate::state::StateStore;

pub(crate) const GAMMA_BASE: &str = "https://gamma-api.polymarket.com";
const CLOB_BASE: &str = "https://clob.polymarket.com";
/// Extra pre-discovery margin: time for the bot to subscribe to the bracket
/// and receive its initial book snapshot *before* it enters the trading window.
const DISCOVERY_LOOKAHEAD_SECS: i64 = 600; // 10 min
/// Throttle of the per-market `uni_scan` log (~1 log/5min).
const UNI_SCAN_LOG_INTERVAL_SECS: u64 = 300;

/// Determines which of the two token_ids is YES and which is NO using the `outcomes` field
/// of the Gamma API (JSON array in the same order as `clob_token_ids`).
///
/// The Gamma API does not guarantee index 0 = YES — for UP/DOWN neg_risk markets the DOWN
/// (NO) may come first. We look for the token whose outcome contains key words of
/// "YES"; if there is no clear match, the fallback convention index-0 = YES is used.
fn yes_no_tokens(token_ids: &[String], outcomes_json: &Option<String>) -> (String, String) {
    if let Some(json) = outcomes_json {
        if let Ok(outcomes) = serde_json::from_str::<Vec<String>>(json) {
            if outcomes.len() >= 2 {
                let is_yes = |s: &str| {
                    let lo = s.to_lowercase();
                    lo == "yes"
                        || lo.contains("up")
                        || lo.contains("above")
                        || lo.contains("higher")
                        || lo.contains("over")
                        || lo.contains("bull")
                        || lo.contains("greater")
                        || lo.ends_with('+')
                };
                if is_yes(&outcomes[0]) {
                    return (token_ids[0].clone(), token_ids[1].clone());
                } else if is_yes(&outcomes[1]) {
                    tracing::debug!(
                        outcomes = ?outcomes,
                        "discover: tokens inverted with respect to convention — token[1] is YES"
                    );
                    return (token_ids[1].clone(), token_ids[0].clone());
                } else {
                    tracing::warn!(
                        outcomes = ?outcomes,
                        "discover: no outcome recognized as YES — using index-0 convention"
                    );
                }
            }
        }
    }
    // Fallback: Polymarket convention index-0 = YES.
    (token_ids[0].clone(), token_ids[1].clone())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GammaMarket {
    pub(crate) condition_id: String,
    pub(crate) slug: Option<String>,
    clob_token_ids: Option<String>,
    /// JSON array of outcome labels in the same order as clob_token_ids.
    /// Ej: "[\"Yes\",\"No\"]", "[\"Down\",\"Up\"]", "[\"Below $X\",\"Above $X\"]".
    outcomes: Option<String>,
    closed: Option<bool>,
    active: Option<bool>,
    neg_risk: Option<bool>,
    /// Market close timestamp. May be null on recent updown markets;
    /// in that case it is extracted from the slug's epoch suffix or from `end_date`.
    #[serde(rename = "clearAt")]
    clear_at: Option<String>,
    start_date: Option<String>,
    /// Market expiration date (RFC-3339). Fallback for `clear_at`.
    end_date: Option<String>,
    /// Actual opening of the updown bracket (RFC-3339). It is the only reliable
    /// source of the cycle start — `start_date` is the market creation
    /// (~24 h earlier) and doesn't work as the opening.
    event_start_time: Option<String>,
    /// Underlying price at the bracket snapshot (several possible names in the Gamma API).
    #[serde(alias = "startValue", alias = "openPrice", alias = "referencePrice")]
    open_price: Option<f64>,
    /// Market taker fee in basis points (0 = no fee).
    /// The Gamma API returns it as `feeRateBps`. Absent → None (standard fee assumed).
    #[serde(rename = "feeRateBps", default)]
    fee_rate_bps: Option<u32>,
    /// Real rate of the market's parabolic taker fee.
    /// Confirmed with 31,493 markets that there are ≥2 coexisting regimes:
    /// `crypto_fees_v2` with `rate=0.07` (crypto updown) and
    /// `finance_prices_fees` with `rate=0.04` (stock ladders — the 10
    /// August ladders migrated from `feeSchedule=null` in July to
    /// `rate=0.04`). `None` when the market charges no taker fee.
    #[serde(rename = "feeSchedule", default)]
    fee_schedule: Option<FeeSchedule>,
    /// Minimum size (shares) to qualify for the Liquidity Rewards Program.
    /// Verified via `curl gamma-api.polymarket.com/markets`: the
    /// field is ALWAYS present and is `0` when the market has no active
    /// rewards — `register_if_new` translates that 0 to `None` (never a faked 0).
    #[serde(rename = "rewardsMinSize", default)]
    rewards_min_size: Option<f64>,
    /// Maximum spread (in CENTS, not a price fraction — confirmed in
    /// docs.polymarket.com/market-makers/liquidity-rewards: "max spread config
    /// of 3 cents") from the adjusted midpoint to qualify. Same 0 → None
    /// criterion as `rewards_min_size`.
    #[serde(rename = "rewardsMaxSpread", default)]
    rewards_max_spread: Option<f64>,
    /// Per-market rewards configuration (includes the pool's daily
    /// rate). Observed `null` on crypto updown even though
    /// `rewardsMinSize`/`rewardsMaxSpread` are populated — so the daily
    /// rate may be missing even though the rest of the rewards config exists.
    #[serde(rename = "clobRewards", default)]
    clob_rewards: Option<Vec<ClobReward>>,
}

/// An element of `clobRewards[]` — we only care about the pool's daily rate;
/// the remaining fields (id, conditionId, assetAddress, dates…) are unused.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClobReward {
    rewards_daily_rate: Option<f64>,
}

/// Gamma's `feeSchedule` — verified on the real API
/// as `{"exponent":1,"rate":0.07,"takerOnly":true,"rebateRate":0.2}`. Only
/// `rate` matters to us; `exponent`/`takerOnly`/`rebateRate` are unused today.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FeeSchedule {
    rate: Option<f64>,
}

/// Maps Gamma's raw rewards fields to `MarketEntry` — 0 (or
/// absent) in `rewardsMinSize`/`rewardsMaxSpread` is how the API
/// indicates "no rewards program" and is translated to `None` (never a faked 0).
/// Pure function separate from `register_if_new` so it can be tested without
/// building a full `Config`/`MarketRegistry`.
fn parse_rewards(gm: &GammaMarket) -> (Option<Decimal>, Option<Decimal>, Option<Decimal>) {
    let min_size = gm
        .rewards_min_size
        .and_then(|v| Decimal::try_from(v).ok())
        .filter(|&v| v > Decimal::ZERO);
    let max_spread = gm
        .rewards_max_spread
        .and_then(|v| Decimal::try_from(v).ok())
        .filter(|&v| v > Decimal::ZERO);
    let daily_rate = gm
        .clob_rewards
        .as_ref()
        .and_then(|rewards| rewards.first())
        .and_then(|r| r.rewards_daily_rate)
        .and_then(|v| Decimal::try_from(v).ok())
        .filter(|&v| v > Decimal::ZERO);
    (min_size, max_spread, daily_rate)
}

/// When there is a research tape, `EventKind::MarketMeta` is emitted
/// for each NEW bracket registered (the local instant of registration is the
/// event's `avail_ts_ms`, never backdated). The producer always exists;
/// with the hub inactive `record` discards at no cost.
pub async fn initial_discover(
    http: &reqwest::Client,
    config: &Config,
    registry: &Arc<MarketRegistry>,
    state: &Arc<StateStore>,
    recorder: &EventProducer,
) {
    let mut tag_id_cache = HashMap::new();
    let mut uni_scan_last = HashMap::new();
    let mut universe_failures = HashMap::new();
    if let Err(e) = discover_all_active_universes(
        http,
        config,
        registry,
        state,
        &mut tag_id_cache,
        &mut uni_scan_last,
        &mut universe_failures,
        recorder,
    )
    .await
    {
        tracing::warn!(error = %e, "market_feed: initial Gamma fetch failed");
    }
    tracing::info!(
        "market_feed: discovered {} brackets",
        registry.markets.len()
    );
}

pub async fn run_discover_task(
    http: reqwest::Client,
    config: Arc<Config>,
    registry: Arc<MarketRegistry>,
    sub_tx: broadcast::Sender<TokenEvent>,
    state: Arc<StateStore>,
    recorder: EventProducer,
) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
        config.market_feed.market_poll_interval_secs,
    ));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval.tick().await; // skip first tick (already discovered at startup)
    let mut consecutive_failures: u32 = 0;
    // Cache of tag_slug→tag_id and throttle of the uni_scan log,
    // both live for the whole life of the task (reset only if the bot
    // restarts the whole feed after a panic — see the restart loop in mod.rs).
    let mut tag_id_cache: HashMap<String, String> = HashMap::new();
    let mut uni_scan_last: HashMap<String, Instant> = HashMap::new();
    // Consecutive failures PER UNIVERSE (distinct from the global `consecutive_failures`
    // above, which only counts cycles where NO universe responded).
    let mut universe_failures: HashMap<String, u32> = HashMap::new();
    loop {
        interval.tick().await;
        if consecutive_failures > 0 {
            let extra_secs = (30u64 * consecutive_failures as u64).min(300);
            tracing::warn!(
                consecutive_failures,
                extra_secs,
                "market_feed: Gamma failing — waiting before retrying"
            );
            tokio::time::sleep(tokio::time::Duration::from_secs(extra_secs)).await;
        }
        if discover_and_emit(
            &http,
            &config,
            &registry,
            &sub_tx,
            &state,
            &mut tag_id_cache,
            &mut uni_scan_last,
            &mut universe_failures,
            &recorder,
        )
        .await
        {
            consecutive_failures = 0;
        } else {
            consecutive_failures += 1;
        }
    }
}

/// Returns `true` if at least one active universe could be queried without error.
#[allow(clippy::too_many_arguments)]
async fn discover_and_emit(
    http: &reqwest::Client,
    config: &Config,
    registry: &Arc<MarketRegistry>,
    sub_tx: &broadcast::Sender<TokenEvent>,
    state: &Arc<StateStore>,
    tag_id_cache: &mut HashMap<String, String>,
    uni_scan_last: &mut HashMap<String, Instant>,
    universe_failures: &mut HashMap<String, u32>,
    rec: &EventProducer,
) -> bool {
    // Evict expired brackets before discovering new ones.
    // Scan=true universes WITHOUT an end_date ceiling (rewards-longdated)
    // are never evicted by clear_at — they are also purged by
    // insertion age (see `MarketRegistry::evict_expired`).
    let long_lived_universes: HashSet<String> = config
        .universes
        .iter()
        .filter(|(_, u)| u.scan && u.end_date_max_hours == 0)
        .map(|(name, _)| name.clone())
        .collect();
    // Eviction returns the evicted token_ids → dynamic WS unsubscribe
    // (`operation:unsubscribe`, verified with a probe):
    // keeping the subscribed total bounded reduces venue throttling.
    let evicted = registry.evict_expired(
        config.market_feed.evict_grace_secs,
        config.market_feed.clear_at_cache_ttl_secs,
        &long_lived_universes,
        config.market_feed.clear_at_cache_ttl_secs,
    );
    if !evicted.is_empty() {
        let _ = sub_tx.send(TokenEvent::Unsubscribe { token_ids: evicted });
    }

    match discover_all_active_universes(
        http,
        config,
        registry,
        state,
        tag_id_cache,
        uni_scan_last,
        universe_failures,
        rec,
    )
    .await
    {
        Ok(new_token_ids) => {
            if !new_token_ids.is_empty() {
                let _ = sub_tx.send(TokenEvent::Subscribe {
                    token_ids: new_token_ids,
                });
            }
            true
        }
        Err(e) => {
            tracing::warn!(error = %e, "market_feed: Gamma fetch failed");
            false
        }
    }
}

/// Specificity priority of a universe's filter —
/// lower number = stricter filter = claims first. `"slug"` filters by
/// known slug pattern (`matches_bracket`); `"rewards"` filters by rate>0
/// but with no date ceiling; `"tag"` may filter NOTHING beyond the server-side
/// `tag_id` (and with `related_tags=true` that filter is even looser). Before, the
/// order was purely alphabetical: `crypto-multistrike` (tag, no slug filter)
/// was queried BEFORE `crypto-updown-short`/`crypto-updown-long` (slug) and
/// could steal a `condition_id` from a universe that WOULD have claimed it with
/// a correct filter — the bracket ended up in a universe that no strategy
/// declares and `strategy::run` skipped it silently (no counter or log).
fn discovery_priority(discovery: &str) -> u8 {
    match discovery {
        "slug" => 0,
        "rewards" => 1,
        "tag" => 2,
        _ => 3,
    }
}

/// Discovers markets of ALL active universes
/// (referenced by an `enabled=true` strategy, or `scan=true`) and
/// registers the new ones. Dedup by `condition_id` across universes in the same
/// pass — ordered by the SPECIFICITY of the discovery filter
/// (`discovery_priority`), not alphabetically; within the same level of
/// specificity, alphabetical so the order stays deterministic.
/// Returns the new token_ids to subscribe to the WS. Common to
/// `initial_discover` (once) and `discover_and_emit` (every poll).
#[allow(clippy::too_many_arguments)]
async fn discover_all_active_universes(
    http: &reqwest::Client,
    config: &Config,
    registry: &Arc<MarketRegistry>,
    state: &Arc<StateStore>,
    tag_id_cache: &mut HashMap<String, String>,
    uni_scan_last: &mut HashMap<String, Instant>,
    universe_failures: &mut HashMap<String, u32>,
    rec: &EventProducer,
) -> anyhow::Result<Vec<String>> {
    let mut universe_names: Vec<String> = config.active_universe_names().into_iter().collect();
    universe_names.sort_by_key(|name| {
        let prio = config
            .universes
            .get(name)
            .map(|u| discovery_priority(&u.discovery))
            .unwrap_or(u8::MAX);
        (prio, name.clone())
    });
    if universe_names.is_empty() {
        return Ok(Vec::new());
    }
    // Set of active universes — ceding a bracket only
    // applies if the slug owner can claim it (it is active this cycle).
    let universe_set: HashSet<String> = universe_names.iter().cloned().collect();

    let oracle_assets = oracle_asset_set(config);
    let mut claimed: HashSet<String> = HashSet::new();
    let mut new_token_ids: Vec<String> = Vec::new();
    let mut any_ok = false;
    let mut last_err: Option<anyhow::Error> = None;
    // `refresh_taker_fee` is one GET per bracket — it accumulates here and
    // runs in parallel (bounded concurrency) AFTER finishing the
    // discovery of all universes, instead of a sequential await inside the
    // loop (up to ~400 serial GETs in `initial_discover` with 4 universes ×
    // 100 markets).
    let mut fee_refresh_pending: Vec<(String, String)> = Vec::new();

    for uname in &universe_names {
        let Some(uni) = config.universes.get(uname) else {
            continue;
        };
        match fetch_universe_markets(http, uni, tag_id_cache, config.market_feed.evict_grace_secs)
            .await
        {
            Ok(gamma_markets) => {
                any_ok = true;
                universe_failures.remove(uname); // success resets the consecutive-failures counter
                                                 // `max_markets` is now a POST-filter cap — how many
                                                 // NEW brackets (just past the discovery filter in
                                                 // `register_if_new`) are registered for this universe in THIS cycle.
                                                 // It no longer limits how many raw markets are paginated from Gamma (that is
                                                 // governed by `fetch_gamma_page`/`MAX_PAGES`).
                let max_new_this_cycle = (uni.max_markets as usize).max(1);
                let mut registered_this_cycle: usize = 0;
                for gm in gamma_markets {
                    // Dedup: a market that matches the window of two universes is
                    // attributed to the FIRST in the specificity priority above
                    // — no longer alphabetical.
                    if !claimed.insert(gm.condition_id.clone()) {
                        continue;
                    }
                    // Cession — if this universe does NOT filter by slug
                    // (discovery != "slug") but the market matches the slug_patterns of
                    // ANOTHER active universe with discovery="slug", the bracket is CEDED:
                    // it is not registered under a universe that no strategy would operate
                    // correctly (the owner will pick it up as soon as its fetch
                    // returns it — the misclaim happened when the slug fetch paginated
                    // up to the safety ceiling and lost markets, or failed).
                    // Before, it was only logged and registered anyway.
                    if uni.discovery != "slug" {
                        if let Some(stolen_from) =
                            slug_owner_conflict(config, uname, &gm, &universe_set)
                        {
                            metrics::counter!(
                                "hrf_universe_priority_misclaim_total",
                                "universe" => uname.to_string(),
                                "slug_owner" => stolen_from.clone()
                            )
                            .increment(1);
                            tracing::warn!(
                                cid = %gm.condition_id,
                                slug = %gm.slug.as_deref().unwrap_or(""),
                                claimed_by = %uname,
                                slug_owner = %stolen_from,
                                "market_feed: bracket ceded: a universe without a slug filter does not claim markets matching another active universe's slug_patterns"
                            );
                            continue;
                        }
                    }
                    if let Some((token_ids, entry)) =
                        register_if_new(&gm, registry, uname, uni, &oracle_assets, rec)
                    {
                        tracing::info!(
                            cid      = %gm.condition_id,
                            slug     = %gm.slug.as_deref().unwrap_or(""),
                            universe = %uname,
                            "market_feed: new bracket discovered, subscribing"
                        );
                        persist_market_observed(state, &gm.condition_id, &entry).await;
                        fee_refresh_pending
                            .push((gm.condition_id.clone(), entry.yes_token_id.clone()));
                        // Only WS-bound universes (declared by an enabled
                        // strategy) subscribe to the WS — scan-only ones are
                        // picked up by `poll.rs` via REST on the
                        // next tick. Dynamic `operation:subscribe` add
                        // (~1 RTT), no epoch restart.
                        if registry.ws_universes.contains(uname.as_str()) {
                            new_token_ids.extend(token_ids);
                        }
                        registered_this_cycle += 1;
                        if registered_this_cycle >= max_new_this_cycle {
                            tracing::debug!(
                                universe = %uname, max_new_this_cycle,
                                "market_feed: max_markets cap reached for this universe — the rest will be registered in later cycles"
                            );
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                // Counter of CONSECUTIVE failures per universe — a
                // permanently down universe (pagination 422, anti-bot
                // 403, unresolvable tag_id) used to be reduced to a
                // `warn!` every 30s without ever escalating, indistinguishable from a
                // transient blip.
                let count = universe_failures.entry(uname.clone()).or_insert(0);
                *count += 1;
                if *count >= UNIVERSE_FAILURE_ERROR_THRESHOLD {
                    tracing::error!(
                        error = %e, universe = %uname, consecutive_failures = *count,
                        "market_feed: universe has {} consecutive failures — possible permanent outage", *count
                    );
                    metrics::counter!("hrf_universe_discovery_failures_total", "universe" => uname.to_string()).increment(1);
                } else {
                    tracing::warn!(error = %e, universe = %uname, consecutive_failures = *count, "market_feed: universe fetch failed");
                }
                last_err = Some(e);
            }
        }
    }

    emit_uni_scan_logs(config, registry, uni_scan_last);

    // Fee refresh in parallel (concurrency bounded by
    // `resync_max_concurrent`, reused — same order of magnitude as the REST
    // resync) instead of sequential inside the loop above.
    if !fee_refresh_pending.is_empty() {
        let concurrency = effective_concurrency(config.market_feed.resync_max_concurrent);
        futures_util::stream::iter(fee_refresh_pending)
            .for_each_concurrent(concurrency, move |(cid, token_id)| async move {
                refresh_taker_fee(http, config, registry, &cid, &token_id).await;
            })
            .await;
    }

    if any_ok {
        Ok(new_token_ids)
    } else {
        Err(last_err.unwrap_or_else(|| {
            anyhow::anyhow!("market_feed: no active universe could be queried")
        }))
    }
}

/// Number of consecutive failures of a universe after which it escalates from
/// `warn!` to `error!` + a dedicated metric.
const UNIVERSE_FAILURE_ERROR_THRESHOLD: u32 = 5;

/// Effective concurrency for the `refresh_taker_fee` batch — never
/// 0 (a misconfigured `resync_max_concurrent=0` would block `buffer_unordered`/
/// `for_each_concurrent` forever).
fn effective_concurrency(configured: usize) -> usize {
    configured.max(1)
}

/// If `gm` matches the `slug_patterns` of some ACTIVE universe with
/// `discovery="slug"` other than `exclude_uname`, returns the name of that
/// universe (the bracket's natural "owner"). `None` if none match or the
/// owner is inactive — a disabled slug universe cannot claim, so
/// a market of its own is not "stolen" if another universe registers it.
pub(crate) fn slug_owner_conflict(
    config: &Config,
    exclude_uname: &str,
    gm: &GammaMarket,
    active: &HashSet<String>,
) -> Option<String> {
    config
        .universes
        .iter()
        .find(|(name, uni)| {
            name.as_str() != exclude_uname
                && active.contains(name.as_str())
                && uni.discovery == "slug"
                && matches_bracket(gm, &uni.slug_patterns)
        })
        .map(|(name, _)| name.clone())
}

/// Same computation as `oracle::price::create_price_channels` (`asset_from_slug`
/// over `bracket_slug_patterns`), reproduced here instead of threading the real
/// `price_txs` mapping: `market_feed::run()` (out of scope for this change) does not
/// expose that mapping to `discover.rs`, and it is a pure function of `Config` — both
/// computations always agree as long as they derive from the same `bracket_slug_patterns`.
pub(crate) fn oracle_asset_set(config: &Config) -> HashSet<String> {
    config
        .market_feed
        .bracket_slug_patterns
        .iter()
        .filter_map(|p| asset_from_slug(p))
        .collect()
}

/// Structured log (target `uni_scan`) for `scan=true` universes —
/// they are discovered and observed even if no strategy operates there. Throttle per
/// market (~1 log/5min, `UNI_SCAN_LOG_INTERVAL_SECS`). The gauge
/// `hrf_universe_markets{universe=...}` is recomputed every cycle for ALL
/// universes (not just scan) and so also reflects evictions.
fn emit_uni_scan_logs(
    config: &Config,
    registry: &Arc<MarketRegistry>,
    last_logged: &mut HashMap<String, Instant>,
) {
    let mut counts: HashMap<String, u64> = HashMap::new();
    for item in registry.universe_info.iter() {
        let cid = item.key().clone();
        let info = item.value().clone();
        *counts.entry(info.universe.clone()).or_insert(0) += 1;

        let Some(uni) = config.universes.get(&info.universe) else {
            continue;
        };
        if !uni.scan {
            continue;
        }
        let due = last_logged
            .get(&cid)
            .map(|t| t.elapsed().as_secs() >= UNI_SCAN_LOG_INTERVAL_SECS)
            .unwrap_or(true);
        if !due {
            continue;
        }
        let Some(market) = registry.markets.get(&cid) else {
            continue;
        };
        // Deliberate simplification: instead of omitting the `mid`/`depth` field
        // when the book hasn't arrived yet (two tracing::info! branches), it is logged
        // as Option — `None` already conveys "no book yet" for offline analysis
        // (same criterion as `opt_dec` for the rewards fields).
        let (mid, depth) = book_mid_and_depth(registry, &market.yes_token_id);

        tracing::info!(
            target: "uni_scan",
            universe   = %info.universe,
            slug       = %market.slug,
            cid        = %cid,
            end_date   = market.clear_at,
            rate       = %opt_dec(market.rewards_daily_rate),
            min_size   = %opt_dec(market.rewards_min_size),
            max_spread = %opt_dec(market.rewards_max_spread),
            mid        = ?mid,
            depth      = ?depth,
            has_oracle = info.has_oracle,
            "G-scan: scan-only universe market"
        );
        last_logged.insert(cid, Instant::now());
    }
    for (universe, n) in counts {
        metrics::gauge!("hrf_universe_markets", "universe" => universe).set(n as f64);
    }

    // Purge from the throttle the entries of already-evicted cids — before,
    // `last_logged` grew unbounded with uptime (an entry was inserted per
    // cid but never removed). Simple and correct criterion: if the cid is no longer
    // in `registry.markets`, there is nothing to throttle.
    last_logged.retain(|cid, _| registry.markets.contains_key(cid));
}

fn opt_dec(v: Option<Decimal>) -> String {
    v.map(|d| d.to_string()).unwrap_or_default()
}

/// Mid and depth (bid+ask) of the YES token's top-of-book, if it has already arrived.
fn book_mid_and_depth(
    registry: &Arc<MarketRegistry>,
    yes_token_id: &str,
) -> (Option<Decimal>, Option<Decimal>) {
    let Some(state) = registry.token_books.get(yes_token_id) else {
        return (None, None);
    };
    let best_bid = state.bids.first();
    let best_ask = state.asks.first();
    let mid = match (best_bid, best_ask) {
        (Some(b), Some(a)) => Some((b.price + a.price) / Decimal::from(2)),
        _ => None,
    };
    let depth = match (best_bid, best_ask) {
        (Some(b), Some(a)) => Some(b.size + a.size),
        _ => None,
    };
    (mid, depth)
}

/// Persists the bracket slug so the UI can show it after
/// the registry evict. Best-effort: a DB failure does not stop discovery.
pub(crate) async fn persist_market_observed(
    state: &Arc<StateStore>,
    cid: &str,
    entry: &MarketEntry,
) {
    if let Err(e) = state
        .upsert_market(
            cid,
            &entry.slug,
            &entry.yes_token_id,
            &entry.no_token_id,
            entry.market_open_ts,
            entry.clear_at,
            entry.neg_risk,
        )
        .await
    {
        tracing::warn!(error = %e, cid, "market_feed: could not persist markets_observed");
    }
}

/// Response of the CLOB `GET /fee-rate?token_id=…` — `{"base_fee":1000}` = 1000 bps.
#[derive(Debug, Deserialize)]
struct FeeRateResp {
    base_fee: u32,
}

/// Exact taker fee from the CLOB for a token, in basis points.
///
/// The endpoint responds WITHOUT authentication
/// (`curl https://clob.polymarket.com/fee-rate?token_id=…` → HTTP 200,
/// `{"base_fee":1000}`), so a plain `reqwest::Client` suffices — the
/// authenticated `clob_sdk` client is not needed. Short timeout because it is called
/// once per newly discovered bracket and must not block the rest of the
/// discovery; any failure (network, timeout, status, parse) → `None` + warn.
async fn fetch_taker_fee_bps(http: &reqwest::Client, token_id: &str) -> Option<u32> {
    let resp = http
        .get(format!("{CLOB_BASE}/fee-rate"))
        .query(&[("token_id", token_id)])
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, token_id, "market_feed: /fee-rate unreachable");
            return None;
        }
    };
    let resp = match resp.error_for_status() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, token_id, "market_feed: /fee-rate returned HTTP error");
            return None;
        }
    };
    match resp.json::<FeeRateResp>().await {
        Ok(body) => Some(body.base_fee),
        Err(e) => {
            tracing::warn!(error = %e, token_id, "market_feed: /fee-rate unreadable response");
            None
        }
    }
}

/// Refreshes `MarketEntry.fee_rate_bps` with the real CLOB fee after registering
/// a new bracket. Fetch OUTSIDE any registry lock (done after
/// `register_if_new` already inserted and released its `Ref`); the later `get_mut` is a
/// short lock only to write the field. Gamma's `feeRateBps` (if it came in
/// discovery) stays as a secondary fallback: it is only overwritten if the fetch brings
/// a value (`Some`), a known value is never erased by a network failure.
/// Gated by `config.market_feed.fetch_taker_fee` so it can be disabled without touching code.
pub(crate) async fn refresh_taker_fee(
    http: &reqwest::Client,
    config: &Config,
    registry: &Arc<MarketRegistry>,
    cid: &str,
    token_id: &str,
) {
    if !config.market_feed.fetch_taker_fee {
        return;
    }
    if let Some(bps) = fetch_taker_fee_bps(http, token_id).await {
        if let Some(mut entry) = registry.markets.get_mut(cid) {
            entry.fee_rate_bps = Some(bps);
        }
    }
}

pub(crate) fn register_if_new(
    gm: &GammaMarket,
    registry: &Arc<MarketRegistry>,
    universe_name: &str,
    uni: &UniverseConfig,
    oracle_assets: &HashSet<String>,
    rec: &EventProducer,
) -> Option<(Vec<String>, MarketEntry)> {
    // Quick check before building the entry (may return None
    // if the slug pattern doesn't match, saving the token parse).
    if registry.markets.contains_key(&gm.condition_id) {
        return None;
    }
    // Filter by the universe's discovery kind:
    // - "slug": the usual filter, now over `uni.slug_patterns`.
    // - "rewards": no end_date ceiling — filtered here by rate>0.
    // - "tag": already filtered server-side by `tag_id` (see `fetch_universe_markets`).
    match uni.discovery.as_str() {
        "slug" => {
            if !matches_bracket(gm, &uni.slug_patterns) {
                return None;
            }
        }
        "rewards" => {
            parse_rewards(gm).2?;
        }
        _ => {}
    }
    // Note: the final check + insert uses DashMap's entry() (see below) to be atomic.

    let slug = gm.slug.clone().unwrap_or_default();
    let token_ids_json = gm.clob_token_ids.as_ref()?;
    let token_ids: Vec<String> = serde_json::from_str(token_ids_json).ok()?;
    if token_ids.len() < 2 {
        return None;
    }
    let (yes_token_id, no_token_id) = yes_no_tokens(&token_ids, &gm.outcomes);

    // Resolve the real close timestamp:
    //   1. clearAt (standard API)
    //   2. Numeric slug suffix (e.g. btc-updown-5m-1780280400)
    //   3. endDate
    // If no source gives a valid timestamp, the bracket is discarded.
    let clear_at = match resolve_clear_at(&gm.clear_at, &gm.slug, &gm.end_date) {
        Some(t) => t,
        None => {
            tracing::warn!(
                cid  = %gm.condition_id,
                slug = %slug,
                "market_feed: could not determine clear_at (clearAt/slug/endDate missing) — bracket discarded"
            );
            return None;
        }
    };

    // Real bracket opening — Gamma's eventStartTime, or clear_at −
    // slug duration. Before, start_date was used (market creation, ~24 h
    // earlier), which defeated fallback_reference and reference stamping.
    let market_open_ts = resolve_open_ts(&gm.event_start_time, clear_at, &slug, &gm.start_date);

    // fetch_gamma_markets() already filters by endDate window, so only
    // brackets closing in [now-grace, now+active_window+lookahead] arrive here.
    // in_active_window() in strategy.rs applies the exact trading gating.
    let now_ts = Utc::now().timestamp();
    let closes_in = clear_at - now_ts;
    if closes_in < 0 {
        return None; // already closed (endDate/clear_at in the past)
    }
    let is_future = market_open_ts > now_ts;

    let rewards = parse_rewards(gm);

    // has_oracle = the asset derived from the slug has an oracle
    // feed (Binance). Computed per MARKET, not copied from the universe's static
    // `has_oracle` — a universe generically "with oracle" (e.g.
    // crypto-multistrike) may still bring a market whose underlying is not
    // in `bracket_slug_patterns` (e.g. a stock under the same tag).
    let has_oracle = asset_from_slug(&slug)
        .map(|a| oracle_assets.contains(&a))
        .unwrap_or(false);

    let entry = MarketEntry {
        slug: slug.clone(),
        yes_token_id: yes_token_id.clone(),
        no_token_id: no_token_id.clone(),
        clear_at,
        market_open_ts,
        neg_risk: gm.neg_risk.unwrap_or(false),
        reference_price: gm
            .open_price
            .and_then(|p| rust_decimal::Decimal::try_from(p).ok())
            .filter(|&p| p > rust_decimal::Decimal::ZERO),
        fee_rate_bps: gm.fee_rate_bps,
        // Real rate of the market's parabolic taker fee
        // from Gamma's `feeSchedule.rate`; `None` if Gamma has no
        // `feeSchedule` (the parabolic fee with rate falls back to the constant).
        parabolic_fee_rate: gm
            .fee_schedule
            .as_ref()
            .and_then(|fs| fs.rate)
            .and_then(|r| Decimal::try_from(r).ok()),
        // Filled by the first WS tick_size_change event; Gamma
        // discovery does not report the market's tick.
        tick_size: None,
        // Rewards of the Liquidity Rewards Program (see `parse_rewards`).
        rewards_min_size: rewards.0,
        rewards_max_spread: rewards.1,
        rewards_daily_rate: rewards.2,
    };

    // Use DashMap's entry() so the check-and-insert is atomic.
    // If another thread inserted the same cid between the contains_key above and here,
    // or_insert_with won't overwrite the existing entry nor double-count the metric.
    let was_vacant = {
        use dashmap::mapref::entry::Entry;
        match registry.markets.entry(gm.condition_id.clone()) {
            Entry::Occupied(_) => false,
            Entry::Vacant(v) => {
                v.insert(entry.clone());
                true
            }
        }
    };
    if !was_vacant {
        return None; // another thread got there first
    }
    registry
        .token_to_cid
        .insert(yes_token_id.clone(), gm.condition_id.clone());
    registry
        .token_to_cid
        .insert(no_token_id.clone(), gm.condition_id.clone());
    // Same registration point as `markets` — populates the long-lived
    // cache that survives the eviction of the short working set.
    registry.remember_clear_at(
        &gm.condition_id,
        entry.neg_risk,
        entry.clear_at,
        &yes_token_id,
        &no_token_id,
    );
    // Universe + has_oracle for the dispatch of `strategy::run`.
    registry.universe_info.insert(
        gm.condition_id.clone(),
        UniverseInfo {
            universe: universe_name.to_string(),
            has_oracle,
        },
    );
    // Insertion instant — queried by `evict_expired` to
    // purge scan universes without an end_date ceiling, which never expire by clear_at.
    registry
        .scan_inserted_at
        .insert(gm.condition_id.clone(), Utc::now());

    metrics::counter!(
        "hrf_market_feed_new_brackets_total",
        "future" => if is_future { "true" } else { "false" }
    )
    .increment(1);
    tracing::info!(
        cid = %gm.condition_id,
        slug = %slug,
        future = is_future,
        closes_in_secs = closes_in,
        universe = %universe_name,
        has_oracle,
        "market_feed: registered bracket"
    );

    // The bracket metadata goes to the research tape on the
    // SAME channel as market events — not only to `markets_observed`
    // (SQLite). A single emission point: discover and prefetch converge here.
    emit_market_meta(rec, &gm.condition_id, &entry);

    Some((vec![yes_token_id, no_token_id], entry))
}

/// Builds and emits `EventKind::MarketMeta` with the metadata
/// known at registration time. `recv_ts_ms`/`avail_ts_ms` are the
/// real local instant at which the meta becomes known — NEVER `market_open_ts` nor
/// `clear_at` (those are market data and travel INSIDE the payload).
/// `outcome` stays `None`: the official result arrives later via
/// `tools::recorders::resolution` as a separate `EventKind::Resolution`.
fn emit_market_meta(prod: &EventProducer, cid: &str, entry: &MarketEntry) {
    // The producer always exists — don't build the meta if
    // the hub is inactive.
    if !prod.is_active() {
        return;
    }
    let now_ms = crate::time::now().timestamp_millis();
    // The official rule of an `*-updown-*` is "underlying spot at close
    // vs opening reference price" — the one already applied by
    // `post_close`/`LabelRule::UpDownReference`. A bracket outside the
    // updown convention declares its rule as external (not evaluable
    // locally): it is never assigned an assumed rule.
    let rule = if bracket_duration_secs(&entry.slug).is_some() {
        ResolutionRule::UpDownReference
    } else {
        ResolutionRule::Other("external:rule_not_updown".to_string())
    };
    let meta = MarketMeta {
        schema_version: SCHEMA_VERSION,
        market_cid: cid.to_string(),
        slug: entry.slug.clone(),
        yes_token_id: entry.yes_token_id.clone(),
        no_token_id: entry.no_token_id.clone(),
        asset: asset_from_slug(&entry.slug),
        neg_risk: entry.neg_risk,
        // Unix seconds — same domain as `MarketEntry`/BookSnapshot.
        market_open_ts: entry.market_open_ts,
        clear_at: entry.clear_at,
        reference_price: entry.reference_price,
        rule,
        resolution_source: EventSource::PolymarketRest,
        resolution_source_ref: Some(format!("gamma:{}", entry.slug)),
        // Single window "since opening": fee/tick known at registration.
        // `tick_size` arrives later via WS (`tick_size_change`) — here it stays
        // None instead of an invented tick.
        rules_windows: vec![RulesWindow {
            valid_from_ms: entry.market_open_ts.saturating_mul(1000),
            valid_to_ms: None,
            parabolic_fee_rate: entry.parabolic_fee_rate,
            fee_rate_bps: entry.fee_rate_bps,
            tick_size: entry.tick_size,
            min_order_size: None,
        }],
        outcome: None,
    };
    let mut e = EventInput::new(
        EventSource::PolymarketRest,
        EventKind::MarketMeta,
        serde_json::to_value(&meta).unwrap_or(serde_json::Value::Null),
        now_ms,
    );
    e.market_cid = Some(cid.to_string());
    e.asset = meta.asset.clone();
    // Dedup by stable content: re-registering the SAME bracket after an
    // eviction+rediscovery does not duplicate the meta on the tape (the
    // key's fields are immutable for a given cid).
    e.dedup_key = Some(stable_hash((
        "market_meta",
        cid,
        entry.slug.as_str(),
        entry.yes_token_id.as_str(),
        entry.no_token_id.as_str(),
        entry.clear_at,
    )));
    prod.record(e);
}

/// Discovery branch for ONE active universe. `"tag"` resolves
/// each `tag_slug` to its `tag_id` (once, cached) and queries with
/// `?tag_id=<id>&related_tags=true`; `"slug"` and `"rewards"` query without tag
/// (the slug/rewards filter is applied later, in `register_if_new`).
async fn fetch_universe_markets(
    http: &reqwest::Client,
    uni: &UniverseConfig,
    tag_id_cache: &mut HashMap<String, String>,
    evict_grace_secs: i64,
) -> anyhow::Result<Vec<GammaMarket>> {
    match uni.discovery.as_str() {
        "tag" => {
            let mut all = Vec::new();
            for tag_slug in &uni.tag_slugs {
                match resolve_tag_id(http, tag_id_cache, tag_slug).await {
                    Some(tag_id) => {
                        all.extend(fetch_gamma_page(http, uni, Some(&tag_id), evict_grace_secs).await?);
                    }
                    None => tracing::warn!(
                        tag_slug,
                        "market_feed: could not resolve tag_id — tag universe skipped this cycle"
                    ),
                }
            }
            Ok(all)
        }
        // "slug" and "rewards": same base query, without tag_id.
        _ => fetch_gamma_page(http, uni, None, evict_grace_secs).await,
    }
}

/// Resolves `tag_slug` → `tag_id` via `GET /events?tag_slug=<slug>&limit=1`.
///
/// There is no direct tag_slug→tag_id endpoint; the `tags[]` array of the first
/// event returned carries `{id, slug, label, ...}` for each tag of the event —
/// the tag whose `slug` matches the requested one is looked up there. Also confirmed that
/// `GET /markets?tag_id=102516&related_tags=true` (102516 = "multi-strikes")
/// responds with markets of that tag. Result cached in memory (the
/// mapping doesn't change during the process lifetime) to avoid repeating the call every poll.
/// (verified against `curl gamma-api.polymarket.com/events?tag_slug=multi-strikes`)
async fn resolve_tag_id(
    http: &reqwest::Client,
    cache: &mut HashMap<String, String>,
    tag_slug: &str,
) -> Option<String> {
    if let Some(id) = cache.get(tag_slug) {
        return Some(id.clone());
    }

    #[derive(Debug, Deserialize)]
    struct EventTag {
        id: String,
        slug: String,
    }
    #[derive(Debug, Deserialize)]
    struct EventWithTags {
        tags: Option<Vec<EventTag>>,
    }

    // Granular warns per failure stage (network / HTTP status / parse /
    // tag not found) — before, the chained `.ok()?` swallowed the root cause and the
    // only trace was the caller's generic warn ("could not resolve tag_id").
    let resp = match http
        .get(format!("{GAMMA_BASE}/events"))
        .query(&[("tag_slug", tag_slug), ("limit", "1")])
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(tag_slug, error = %e, "market_feed: /events unreachable while resolving tag_id");
            return None;
        }
    };
    let resp = match resp.error_for_status() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(tag_slug, error = %e, "market_feed: /events returned HTTP error while resolving tag_id");
            return None;
        }
    };
    let events: Vec<EventWithTags> = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(tag_slug, error = %e, "market_feed: /events unreadable response while resolving tag_id");
            return None;
        }
    };
    let id = events
        .first()
        .and_then(|e| e.tags.as_ref())
        .and_then(|tags| tags.iter().find(|t| t.slug == tag_slug))
        .map(|t| t.id.clone());
    let Some(id) = id else {
        tracing::warn!(
            tag_slug,
            "market_feed: tag_slug not found in the /events response"
        );
        return None;
    };

    cache.insert(tag_slug.to_string(), id.clone());
    Some(id)
}

/// `[end_date_min, end_date_max]` window of a universe, in unix timestamps.
///
/// - `end_date_min_hours > 0` → real floor (e.g. `rewards-longdated`: never
///   anything closing soon). `0` → falls back to the usual `evict_grace_secs` (don't
///   re-subscribe what just expired in the same eviction cycle).
/// - `end_date_max_hours == 0` → NO CEILING (`None`) — the only real scenario is
///   `rewards-longdated`, long-dated by definition.
fn universe_date_window(
    uni: &UniverseConfig,
    evict_grace_secs: i64,
    now: i64,
) -> (Option<i64>, Option<i64>) {
    let min = if uni.end_date_min_hours > 0 {
        Some(now + uni.end_date_min_hours as i64 * 3600)
    } else {
        Some(now - evict_grace_secs)
    };
    let max = if uni.end_date_max_hours > 0 {
        Some(now + uni.end_date_max_hours as i64 * 3600 + DISCOVERY_LOOKAHEAD_SECS)
    } else {
        None
    };
    (min, max)
}

/// Downloads Polymarket markets for ONE universe — same pagination and
/// reason for the time filter as the original discovery:
/// ordering by `startDate desc` never reaches brackets closing *now*
/// (their `startDate` is old); `end_date_min`/`end_date_max` do.
///
/// `max_markets` NO LONGER stops pagination — before, with
/// `max_markets < PAGE` (the default 50 < 100), the cutoff was ALWAYS met after
/// the first page: each universe saw EXACTLY the first 100 markets
/// by ascending `endDate`, never those closing later (`crypto-updown-long`
/// with `end_date_max_hours=200` never reached the 1h/1d brackets; the
/// `rewards-longdated` rewards filter was applied AFTER the page cutoff, so
/// its real universe was unreachable by construction). Now it paginates until
/// the date window is exhausted (short page, `n < PAGE`) or a safety page
/// ceiling (`MAX_PAGES`); `max_markets` is applied in the CALLER
/// (`discover_all_active_universes`) as a post-filter cap on how many NEW
/// brackets are registered per cycle, not as a pagination cutoff condition.
async fn fetch_gamma_page(
    http: &reqwest::Client,
    uni: &UniverseConfig,
    tag_id: Option<&str>,
    evict_grace_secs: i64,
) -> anyhow::Result<Vec<GammaMarket>> {
    const PAGE: usize = 100;
    // gamma-api.polymarket.com/markets cuts off
    // pagination with HTTP 422 around offset~2000-2100 — safety page
    // ceiling slightly below, so as
    // not to depend on hitting the API's real limit exactly.
    const MAX_PAGES: usize = 19; // 19 * 100 = 1900 max offset

    let now = Utc::now().timestamp();
    let (min_ts, max_ts) = universe_date_window(uni, evict_grace_secs, now);

    let fmt_rfc3339 = |ts: i64| -> String {
        DateTime::<Utc>::from_timestamp(ts, 0)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default()
    };
    let min_str = min_ts.map(fmt_rfc3339);
    let max_str = max_ts.map(fmt_rfc3339);

    let mut all: Vec<GammaMarket> = Vec::new();
    let mut offset: usize = 0;

    for page_num in 0..MAX_PAGES {
        let mut query: Vec<(&str, String)> = vec![
            ("active", "true".to_string()),
            ("closed", "false".to_string()),
            ("limit", PAGE.to_string()),
            ("offset", offset.to_string()),
            ("order", "endDate".to_string()),
            ("ascending", "true".to_string()),
        ];
        if let Some(ref s) = min_str {
            query.push(("end_date_min", s.clone()));
        }
        if let Some(ref s) = max_str {
            query.push(("end_date_max", s.clone()));
        }
        if let Some(tid) = tag_id {
            query.push(("tag_id", tid.to_string()));
            query.push(("related_tags", "true".to_string()));
        }

        let resp = http
            .get(format!("{GAMMA_BASE}/markets"))
            .query(&query)
            .send()
            .await?;

        // A 422 on /markets at a high offset is the REAL end of
        // pagination (documented API limit), not a structural error —
        // treated as an empty page instead of propagating the error and discarding
        // EVERYTHING already accumulated in earlier pages.
        if resp.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
            tracing::debug!(
                offset, discovery = %uni.discovery,
                "market_feed: Gamma /markets returned 422 during pagination — end of pagination (API limit)"
            );
            break;
        }
        let resp = resp.error_for_status()?;
        let page: Vec<GammaMarket> = resp.json().await?;
        let n = page.len();
        all.extend(page);

        if n < PAGE {
            break; // short page = natural end of the date window
        }
        if page_num + 1 >= MAX_PAGES {
            tracing::warn!(
                offset, discovery = %uni.discovery,
                "market_feed: Gamma /markets reached the safety page ceiling — some markets may remain undiscovered this cycle"
            );
            break;
        }
        offset += PAGE;
        // Margin between pages so as not to trigger the rate limit with the
        // pagination, now much deeper than before the fix.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }

    tracing::debug!(
        discovery = %uni.discovery,
        window_min = ?min_str,
        window_max = ?max_str,
        total = all.len(),
        "market_feed: Gamma universe fetch completed"
    );
    Ok(all)
}

/// One-off fetch of ONE market by slug — used by predictive prefetch
/// (`prefetch.rs`): `*-updown-{dur}-{epoch}` brackets have a deterministic
/// slug and exist on Gamma before opening, so the next
/// bracket of each series is discovered and subscribed in advance instead of
/// waiting for the general poll (`market_poll_interval_secs`).
pub(crate) async fn fetch_market_by_slug(
    http: &reqwest::Client,
    slug: &str,
) -> anyhow::Result<Option<GammaMarket>> {
    let resp = http
        .get(format!("{GAMMA_BASE}/markets"))
        .query(&[
            ("slug", slug.to_string()),
            ("active", "true".to_string()),
            ("closed", "false".to_string()),
        ])
        .send()
        .await?;
    if resp.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
        return Ok(None);
    }
    let mut page: Vec<GammaMarket> = resp.error_for_status()?.json().await?;
    Ok(page.pop())
}

/// Resolves the bracket's close timestamp from the available sources.
///
/// Priority order:
/// 1. `clearAt` (standard API — real close of order acceptance).
/// 2. `endDate` (market expiration date — present on updown).
/// 3. Numeric slug suffix (snapshot time for updown: e.g. `btc-updown-5m-<epoch>`).
///    The slug epoch is the BTC snapshot moment, 5 min before the real close;
///    it is used as a last resort when clearAt and endDate are not available.
fn resolve_clear_at(
    clear_at: &Option<String>,
    slug: &Option<String>,
    end_date: &Option<String>,
) -> Option<i64> {
    // Source 1: clearAt (exact platform close)
    if let Some(t) = parse_ts(clear_at).filter(|&t| t > 0) {
        return Some(t);
    }
    // Source 2: endDate (expiration date — real close on updown markets)
    if let Some(t) = parse_ts(end_date).filter(|&t| t > 0) {
        return Some(t);
    }
    // Source 3: slug epoch suffix (last resort). The epoch is the
    // OPENING of the bracket (== eventStartTime, verified against Gamma) —
    // the close is epoch + slug duration.
    if let Some(slug_str) = slug {
        if let Some(last) = slug_str.rsplit('-').next() {
            if let Ok(epoch) = last.parse::<i64>() {
                if epoch > 0 {
                    return Some(epoch + bracket_duration_secs(slug_str).unwrap_or(0));
                }
            }
        }
    }
    None
}

pub(crate) fn matches_bracket(gm: &GammaMarket, patterns: &[String]) -> bool {
    if gm.closed.unwrap_or(false) || !gm.active.unwrap_or(true) {
        return false;
    }
    let slug = gm.slug.as_deref().unwrap_or("").to_lowercase();
    if slug.is_empty() {
        return false;
    }
    patterns.iter().any(|p| slug.contains(p.as_str()))
}

/// Parse an RFC-3339 timestamp string.  Returns `None` when the string is
/// absent or malformed so callers can choose how to react (discard, warn, etc.).
fn parse_ts(s: &Option<String>) -> Option<i64> {
    s.as_deref()
        .and_then(|v| DateTime::parse_from_rfc3339(v).ok())
        .map(|dt| dt.timestamp())
}

/// Bracket duration according to the slug's segment
/// (`btc-updown-5m-…` → 300, `eth-updown-1h-…` → 3600). `None` if there is no
/// duration segment (non-updown market).
fn bracket_duration_secs(slug: &str) -> Option<i64> {
    let seg = slug.split('-').find(|s| {
        s.len() >= 2
            && (s.ends_with('m') || s.ends_with('h'))
            && s[..s.len() - 1].chars().all(|c| c.is_ascii_digit())
    })?;
    let n: i64 = seg[..seg.len() - 1].parse().ok()?;
    match seg.as_bytes()[seg.len() - 1] {
        b'm' => Some(n * 60),
        b'h' => Some(n * 3600),
        _ => None,
    }
}

/// Real bracket opening, in order of reliability:
///   1. Gamma's `eventStartTime` (exact opening)
///   2. `clear_at − slug duration` (updowns are fixed cycles)
///   3. `start_date` (non-updown markets; on updown it's the creation, ~24 h earlier)
fn resolve_open_ts(
    event_start: &Option<String>,
    clear_at: i64,
    slug: &str,
    start_date: &Option<String>,
) -> i64 {
    if let Some(t) = parse_ts(event_start).filter(|&t| t > 0) {
        return t;
    }
    if let Some(dur) = bracket_duration_secs(slug) {
        return clear_at - dur;
    }
    parse_ts(start_date).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `emit_market_meta` writes a real `market_meta` line
    /// to `<dir>/polymarket_rest/<date>.jsonl` with the contract payload
    /// (updown rule, rules window, gamma provenance) and local
    /// emission timestamps — never backdated to `market_open_ts`.
    #[tokio::test]
    async fn emit_market_meta_persists_in_tape() {
        use crate::research::events::{Hub, ResearchRecorder, ResearchRecorderConfig};
        // The test asserts `recv` against wall-clock time: if a `backtest::tests` replay
        // has the global clock armed in parallel,
        // `time::now()` returns tape time and the window fails.
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;

        let dir = tempfile::tempdir().unwrap();
        let cfg = ResearchRecorderConfig {
            dir: dir.path().join("events"),
            ..Default::default()
        };
        let hub = Hub::new();
        let recorder = ResearchRecorder::start(&cfg, &hub).unwrap();
        let prod = hub.producer();

        let entry = MarketEntry {
            slug: "btc-updown-5m-1790175300".to_string(),
            yes_token_id: "111".to_string(),
            no_token_id: "222".to_string(),
            clear_at: 1_790_175_600,
            market_open_ts: 1_790_175_300,
            neg_risk: false,
            reference_price: Some(Decimal::new(115_000, 0)),
            fee_rate_bps: Some(400),
            parabolic_fee_rate: Some(Decimal::new(4, 2)),
            tick_size: Some(Decimal::new(1, 3)),
            rewards_min_size: None,
            rewards_max_spread: None,
            rewards_daily_rate: None,
        };
        emit_market_meta(&prod, "0xabc", &entry);
        // `stop()` no longer depends on `prod` dying (detach() drops
        // the Sender directly) — it is deliberately kept alive.
        let res = recorder.stop().await;
        assert_eq!(res.written, 1);
        assert_eq!(res.deduped, 0);

        let date = Utc::now().format("%Y%m%d").to_string();
        let path = dir
            .path()
            .join("events")
            .join("polymarket_rest")
            .join(format!("{date}.jsonl"));
        let line = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["source"], "polymarket_rest");
        assert_eq!(v["kind"], "market_meta");
        assert_eq!(v["market_cid"], "0xabc");
        assert_eq!(v["asset"], "btc"); // `asset_from_slug` returns lowercase
        let p = &v["payload"];
        assert_eq!(p["market_cid"], "0xabc");
        assert_eq!(p["slug"], "btc-updown-5m-1790175300");
        assert_eq!(p["yes_token_id"], "111");
        assert_eq!(p["no_token_id"], "222");
        assert_eq!(p["neg_risk"], false);
        assert_eq!(p["clear_at"], 1_790_175_600);
        assert_eq!(p["market_open_ts"], 1_790_175_300);
        assert_eq!(p["rule"], "up_down_reference");
        assert_eq!(
            p["resolution_source_ref"],
            serde_json::json!("gamma:btc-updown-5m-1790175300")
        );
        assert!(
            p["outcome"].is_null(),
            "outcome arrives via Resolution, not here"
        );
        // avail >= recv and both ~now — never the bracket's open_ts.
        let now = crate::time::now().timestamp_millis();
        let recv = v["recv_ts_ms"].as_i64().unwrap();
        let avail = v["avail_ts_ms"].as_i64().unwrap();
        assert!(avail >= recv);
        assert!(recv >= now - 60_000 && recv <= now);
    }

    #[test]
    fn updown_slug_duration() {
        assert_eq!(bracket_duration_secs("btc-updown-5m-1783536600"), Some(300));
        assert_eq!(
            bracket_duration_secs("eth-updown-15m-1783536600"),
            Some(900)
        );
        assert_eq!(
            bracket_duration_secs("sol-updown-1h-1783536600"),
            Some(3600)
        );
        // no duration segment (non-updown) → None
        assert_eq!(bracket_duration_secs("will-btc-hit-150k"), None);
    }

    #[test]
    fn open_ts_prefers_event_start_and_falls_back_to_clear_at_minus_duration() {
        // The slug epoch IS the opening (== eventStartTime); close = epoch + 300.
        let evt = Some("2026-07-08T18:50:00Z".to_string()); // = 1783536600
        let start = Some("2026-07-07T18:57:00Z".to_string());
        let slug = "btc-updown-5m-1783536600";
        // 1. eventStartTime wins
        assert_eq!(
            resolve_open_ts(&evt, 1_783_536_900, slug, &start),
            1_783_536_600
        );
        // 2. no eventStartTime → clear_at − 300 (never the ~24h-earlier start_date)
        assert_eq!(
            resolve_open_ts(&None, 1_783_536_900, slug, &start),
            1_783_536_600
        );
        // 3. market with no duration in the slug → start_date
        let sd = resolve_open_ts(&None, 1_783_536_900, "will-btc-hit-150k", &start);
        assert!(sd > 1_783_400_000 && sd < 1_783_536_300);
    }

    #[test]
    fn slug_clear_at_adds_duration() {
        // No clearAt or endDate: the slug epoch is the opening → close = +300.
        let slug = Some("btc-updown-5m-1783536600".to_string());
        assert_eq!(resolve_clear_at(&None, &slug, &None), Some(1_783_536_900));
    }

    // ── Gamma API rewards parser ──────────────────────────────

    /// Real fixture (`curl gamma-api.polymarket.com/markets`) of a
    /// BTC 5m updown WITHOUT an active rewards program: `rewardsMinSize` and
    /// `rewardsMaxSpread` present but 0, `clobRewards` absent (null).
    const GAMMA_WITHOUT_REWARDS: &str = r#"{
        "conditionId": "0xabc",
        "slug": "btc-updown-5m-1766162100",
        "clobTokenIds": "[\"111\",\"222\"]",
        "outcomes": "[\"Up\",\"Down\"]",
        "closed": false,
        "active": true,
        "negRisk": false,
        "clearAt": "2026-07-18T12:05:00Z",
        "rewardsMinSize": 0,
        "rewardsMaxSpread": 0,
        "clobRewards": null
    }"#;

    /// Real fixture of an updown WITH an active rewards program:
    /// `rewardsMinSize`/`rewardsMaxSpread` positive. `clobRewards` is still null
    /// on the observed updowns — tested separately with a non-updown market that
    /// does bring it (see `parses_clob_rewards_daily_rate`).
    const GAMMA_WITH_REWARDS: &str = r#"{
        "conditionId": "0xdef",
        "slug": "eth-updown-5m-1766162700",
        "clobTokenIds": "[\"333\",\"444\"]",
        "outcomes": "[\"Up\",\"Down\"]",
        "closed": false,
        "active": true,
        "negRisk": false,
        "clearAt": "2026-07-18T12:05:00Z",
        "rewardsMinSize": 50,
        "rewardsMaxSpread": 0.2,
        "clobRewards": null
    }"#;

    /// Fixture (non-updown market) with `clobRewards` populated — verifies
    /// the parsing of the pool's daily rate.
    const GAMMA_WITH_CLOB_REWARDS: &str = r#"{
        "conditionId": "0x123",
        "slug": "will-israel-launch-a-major-ground-offensive-in-gaza-by-december-31-828",
        "clobTokenIds": "[\"555\",\"666\"]",
        "outcomes": "[\"Yes\",\"No\"]",
        "closed": false,
        "active": true,
        "negRisk": false,
        "clearAt": "2500-12-31T00:00:00Z",
        "rewardsMinSize": 50,
        "rewardsMaxSpread": 4.5,
        "clobRewards": [{
            "id": "106623",
            "conditionId": "0x1c39f29a392495da86a9d1882bdb7ad82bd617990a6da16674e1ee3da105e9e6",
            "assetAddress": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
            "rewardsAmount": 0,
            "rewardsDailyRate": 10,
            "startDate": "2026-03-19",
            "endDate": "2500-12-31"
        }]
    }"#;

    #[test]
    fn gamma_without_rewards_min_size_and_max_spread_at_zero_translate_to_none() {
        // rewardsMinSize/rewardsMaxSpread at 0 (never absent in the real
        // Gamma response) mean "no rewards program" — they must never
        // slip through as Some(0) (faked 0).
        let gm: GammaMarket = serde_json::from_str(GAMMA_WITHOUT_REWARDS).unwrap();
        let (min_size, max_spread, daily_rate) = parse_rewards(&gm);
        assert_eq!(min_size, None);
        assert_eq!(max_spread, None);
        assert_eq!(daily_rate, None);
    }

    #[test]
    fn gamma_with_rewards_parses_min_size_and_max_spread() {
        let gm: GammaMarket = serde_json::from_str(GAMMA_WITH_REWARDS).unwrap();
        let (min_size, max_spread, daily_rate) = parse_rewards(&gm);
        assert_eq!(min_size, Some(Decimal::from(50)));
        assert_eq!(max_spread, Some(Decimal::new(2, 1))); // 0.2
                                                          // clobRewards null in this market (observed on updown) → no daily_rate.
        assert_eq!(daily_rate, None);
    }

    #[test]
    fn parses_clob_rewards_daily_rate() {
        let gm: GammaMarket = serde_json::from_str(GAMMA_WITH_CLOB_REWARDS).unwrap();
        let (min_size, max_spread, daily_rate) = parse_rewards(&gm);
        assert_eq!(min_size, Some(Decimal::from(50)));
        assert_eq!(max_spread, Some(Decimal::new(45, 1))); // 4.5
        assert_eq!(daily_rate, Some(Decimal::from(10)));
    }

    /// Fixture (same non-updown market) with `rewardsDailyRate: 0` — same
    /// criterion as min_size/max_spread: 0 means "no rate", not a real rate.
    const GAMMA_CLOB_REWARDS_RATE_ZERO: &str = r#"{
        "conditionId": "0x456",
        "slug": "will-israel-launch-a-major-ground-offensive-in-gaza-by-december-31-829",
        "clobTokenIds": "[\"777\",\"888\"]",
        "outcomes": "[\"Yes\",\"No\"]",
        "closed": false,
        "active": true,
        "negRisk": false,
        "clearAt": "2500-12-31T00:00:00Z",
        "rewardsMinSize": 50,
        "rewardsMaxSpread": 4.5,
        "clobRewards": [{
            "id": "106624",
            "conditionId": "0x1c39f29a392495da86a9d1882bdb7ad82bd617990a6da16674e1ee3da105e9e7",
            "assetAddress": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
            "rewardsAmount": 0,
            "rewardsDailyRate": 0,
            "startDate": "2026-03-19",
            "endDate": "2500-12-31"
        }]
    }"#;

    #[test]
    fn rewards_daily_rate_at_zero_translates_to_none() {
        // rewardsDailyRate: 0 means "no confirmed rate" (pool not yet funded
        // or program ended) — same 0→None criterion as min_size/max_spread,
        // it must never slip through as Some(0) (faked 0) for the G0 gate.
        let gm: GammaMarket = serde_json::from_str(GAMMA_CLOB_REWARDS_RATE_ZERO).unwrap();
        let (min_size, max_spread, daily_rate) = parse_rewards(&gm);
        assert_eq!(min_size, Some(Decimal::from(50)));
        assert_eq!(max_spread, Some(Decimal::new(45, 1))); // 4.5
        assert_eq!(daily_rate, None);
    }

    // ── Gamma API `feeSchedule` parser ────────────────────────

    /// Real fixture of a `crypto_fees_v2` market:
    /// `feeSchedule.rate = 0.07`.
    const GAMMA_FEE_SCHEDULE_CRYPTO: &str = r#"{
        "conditionId": "0x789",
        "slug": "btc-updown-5m-1766162100",
        "clobTokenIds": "[\"111\",\"222\"]",
        "outcomes": "[\"Up\",\"Down\"]",
        "closed": false,
        "active": true,
        "negRisk": false,
        "clearAt": "2026-07-18T12:05:00Z",
        "feeSchedule": {"exponent": 1, "rate": 0.07, "takerOnly": true, "rebateRate": 0.2}
    }"#;

    /// Fixture: stock price ladders moved from `feeSchedule=null` to `finance_prices_fees` with `rate=0.04` in August.
    const GAMMA_FEE_SCHEDULE_FINANCE: &str = r#"{
        "conditionId": "0xabc123",
        "slug": "nvda-above-180-on-august-15-2026",
        "clobTokenIds": "[\"999\",\"000\"]",
        "outcomes": "[\"Yes\",\"No\"]",
        "closed": false,
        "active": true,
        "negRisk": false,
        "clearAt": "2026-08-15T20:00:00Z",
        "feeSchedule": {"exponent": 1, "rate": 0.04, "takerOnly": true, "rebateRate": 0.25}
    }"#;

    #[test]
    fn gamma_market_parses_fee_schedule_rate_crypto() {
        let gm: GammaMarket = serde_json::from_str(GAMMA_FEE_SCHEDULE_CRYPTO).unwrap();
        assert_eq!(gm.fee_schedule.as_ref().and_then(|fs| fs.rate), Some(0.07));
    }

    #[test]
    fn gamma_market_parses_fee_schedule_rate_finance() {
        let gm: GammaMarket = serde_json::from_str(GAMMA_FEE_SCHEDULE_FINANCE).unwrap();
        assert_eq!(gm.fee_schedule.as_ref().and_then(|fs| fs.rate), Some(0.04));
    }

    /// Market without `feeSchedule` (absent from the JSON, as in the updown markets seen before the field existed,
    /// or markets without a taker fee) → `None`. Discovery leaves `MarketEntry.parabolic_fee_rate = None` and the call site
    /// falls back to `TAKER_FEE_RATE_PARABOLICA`.
    /// fallback `TAKER_FEE_RATE_PARABOLICA`.
    #[test]
    fn gamma_market_without_fee_schedule_is_none() {
        let gm: GammaMarket = serde_json::from_str(GAMMA_WITHOUT_REWARDS).unwrap();
        assert!(gm.fee_schedule.is_none());
    }

    // ── Bracket ownership: handed to the owner by slug_patterns ──────────

    /// Config with two universes: `foo` (slug, pattern `btc-updown`) and
    /// `rewards-x` (rewards, no slug filter: the rewards-longdated role).
    fn cfg_two_universes() -> Config {
        use crate::config::UniverseConfig;
        let mut cfg = crate::config::base_config_with_universe(true);
        cfg.universes.insert(
            "rewards-x".into(),
            UniverseConfig {
                discovery: "rewards".into(),
                slug_patterns: Vec::new(),
                tag_slugs: Vec::new(),
                end_date_min_hours: 24,
                end_date_max_hours: 0,
                duration_filter: Vec::new(),
                has_oracle: false,
                scan: true,
                max_markets: 100,
            },
        );
        cfg
    }

    fn gm_slug(slug: &str) -> GammaMarket {
        serde_json::from_str(&format!(r#"{{"conditionId":"0xc1","slug":"{slug}"}}"#)).unwrap()
    }

    /// A `btc-updown-*` market is owned via slug_patterns (`foo`, asset) → the universe without a filter must detect it as a
    /// conflict (the call site skips it with `continue` and does not register it).
    #[test]
    fn slug_owner_conflict_detects_active_owner() {
        let cfg = cfg_two_universes();
        let active: HashSet<String> = ["foo", "rewards-x"].iter().map(|s| s.to_string()).collect();
        let gm = gm_slug("btc-updown-15m-1789819200");
        assert_eq!(
            slug_owner_conflict(&cfg, "rewards-x", &gm, &active),
            Some("foo".to_string()),
            "btc-updown-15m belongs to the slug_patterns of foo"
        );
    }

    /// If the slug owner is INACTIVE there is no conflict: a disabled universe cannot claim, so the market is not "stolen"
    /// if rewards-x registers it.
    #[test]
    fn slug_owner_conflict_none_if_owner_inactive() {
        let cfg = cfg_two_universes();
        let active: HashSet<String> = ["rewards-x"].iter().map(|s| s.to_string()).collect();
        let gm = gm_slug("btc-updown-15m-1789819200");
        assert_eq!(slug_owner_conflict(&cfg, "rewards-x", &gm, &active), None);
    }

    /// A legitimate rewards market (slug matching no asset slug_patterns) is not a conflict: rewards-longdated claims it.
    #[test]
    fn slug_owner_conflict_none_for_legitimate_rewards_market() {
        let cfg = cfg_two_universes();
        let active: HashSet<String> = ["foo", "rewards-x"].iter().map(|s| s.to_string()).collect();
        let gm = gm_slug("elon-musk-of-tweets-september-40-64");
        assert_eq!(slug_owner_conflict(&cfg, "rewards-x", &gm, &active), None);
    }

    /// The owning universe never reports itself as a conflict (exclude_uname): an extra defence on top of the
    /// `discovery != "slug"` check at the call site.
    #[test]
    fn slug_owner_conflict_excludes_own_universe() {
        let cfg = cfg_two_universes();
        let active: HashSet<String> = ["foo"].iter().map(|s| s.to_string()).collect();
        let gm = gm_slug("btc-updown-5m-1789819200");
        assert_eq!(slug_owner_conflict(&cfg, "foo", &gm, &active), None);
    }
}
