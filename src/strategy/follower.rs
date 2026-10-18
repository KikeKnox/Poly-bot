//! Wallet follower: copies entries/exits of a target wallet
//! into the paper account with its own size.
//!
//! NOT a `Strategy` (see the trait in `strategy/mod.rs`): that one reacts to
//! books we ALREADY track via `market_feed`, tied to our crypto-updown
//! universes. The target wallet trades markets that `market_feed` never
//! discovers (sports). This module is a standalone task — same pattern
//! (structure, not content) as `src/settlement/post_close.rs`/`src/settlement/orphan.rs`: periodic poll
//! against `data-api.polymarket.com/activity`, detection of `TRADE` events,
//! one-off REST fetch (CLOB book + Gamma metadata) and manual construction
//! of an `OrderIntent` sent over the shared `intent_tx` mpsc — the
//! same channel the other strategies use, so executor/risk/state
//! already work without touching them (kill switch, rate limit, daily loss stop,
//! exposure — all attributed to `StrategyId::Follower`).
//!
//! ## Known limitations of this phase (documented, not resolved here)
//!
//! - No persistence of the `/activity` cursor across restarts: the initial
//!   cursor is "right now" at startup — a restart loses the wallet's
//!   activity that happened during the downtime.
//! - No hot toggle via `StrategySupervisor` (see the guard in
//!   `strategy/supervisor.rs::spawn`) — enable/disable only via TOML + restart.
//! - Does not copy MAKER fills of the target wallet — only `TRADE` events from
//!   `/activity` (maker fills are not mechanically
//!   copyable as an "entry" until the market resolves).
//! - No backward pagination (`end=`) in the poll: if the wallet generates more
//!   than `ACTIVITY_PAGE_LIMIT` new trades between two polls, the oldest
//!   of that burst are lost (a `warn!` is logged when the page comes back
//!   full). Acceptable for this phase — cheap mechanics, not minimum latency.
//! - Sub-second latency (mempool/on-chain WS) is out of scope — separate spec
//!   if this phase confirms it is worthwhile.

use std::sync::Arc;
use std::time::Duration;

use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::{
    config::defaults,
    state::StateStore,
    types::{Level, OrderIntent, OrderSide, Position, QueuedIntent, StrategyId},
};

const DATA_API_BASE: &str = "https://data-api.polymarket.com";
const CLOB_BASE: &str = "https://clob.polymarket.com";
const GAMMA_BASE: &str = "https://gamma-api.polymarket.com";

// ── FollowerConfig ──────────────────────────────────────────────────────────

/// Wallet follower configuration.
///
/// `limits` (flatten) reuses `StrategyLimits` to hook into
/// `RiskGuard` (kill switch, hourly rate limit, daily loss stop) exactly
/// like the real strategies — the follower sends its `OrderIntent`s over
/// the same `intent_tx` as the others, so the executor already applies those
/// limits without new code (see `main.rs`, next to the other
/// strategies). `limits.universe`/`enabled` in the sense of "spawning a
/// `Strategy`" do not apply here — `enabled` does control whether `main.rs` spawns
/// the standalone task.
#[derive(Debug, Clone, Deserialize)]
pub struct FollowerConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    /// Address of the target wallet to watch.
    #[serde(default = "defaults::follower_target_wallet")]
    pub target_wallet_address: String,
    /// Polling interval against `data-api.polymarket.com/activity`.
    #[serde(default = "defaults::follower_poll_interval_secs")]
    pub poll_interval_secs: u64,
    /// Own size in USDC per copy — independent of the target
    /// wallet's real size (EV is per dollar bet, it does not depend on the amount).
    #[serde(default = "defaults::follower_size_usdc")]
    pub size_usdc: Decimal,
    /// Maximum slippage tolerated over the observed ask when copying a buy.
    #[serde(default = "defaults::follower_max_slippage_pct")]
    pub max_slippage_pct: Decimal,
    /// If `true`, also copies the target wallet's detected sells
    /// (when we have an open position attributable to `StrategyId::Follower`).
    #[serde(default = "defaults::follower_mirror_exits")]
    pub mirror_exits: bool,
}

impl Default for FollowerConfig {
    fn default() -> Self {
        Self {
            limits: crate::config::StrategyLimits::default(),
            target_wallet_address: defaults::follower_target_wallet(),
            poll_interval_secs: defaults::follower_poll_interval_secs(),
            size_usdc: defaults::follower_size_usdc(),
            max_slippage_pct: defaults::follower_max_slippage_pct(),
            mirror_exits: defaults::follower_mirror_exits(),
        }
    }
}

/// Page size of the `/activity` poll. No backward `end=` pagination
/// (see the limitation in the module doc-comment) — if the wallet generates more
/// trades than this between two polls, the oldest of the burst are lost.
const ACTIVITY_PAGE_LIMIT: usize = 50;

// ── Wallet activity (`/activity`) ───────────────────────────────────────────

/// Raw event from `data-api.polymarket.com/activity` (only the fields
/// we use — serde ignores the rest). Format verified with a real `curl`:
/// `type`, `side`, `asset` (=token_id),
/// `conditionId`, `timestamp` (unix seconds) are always present in a
/// `TRADE` event.
#[derive(Debug, Clone, Deserialize)]
struct ActivityEvent {
    #[serde(rename = "type")]
    kind: String,
    side: String,
    asset: String,
    #[serde(rename = "conditionId")]
    condition_id: String,
    timestamp: i64,
    /// On-chain hash of the trade — dedup key for the events of the
    /// boundary second (`timestamp == last_seen_ts`), which the `>=` filter
    /// brings back on every poll. `default`: the API always sends it in a
    /// TRADE, but an event without a hash must not break parsing.
    #[serde(rename = "transactionHash", default)]
    transaction_hash: String,
}

/// Trade detected in the target wallet's activity.
///
/// Does not derive `PartialEq`: `OrderSide` does not implement it (it contains variants
/// without usefully comparable data) — tests compare individual fields.
#[derive(Debug, Clone)]
struct WalletTrade {
    /// `Buy` or `Sell { size: None }` — the target wallet's size is not used,
    /// we copy with our own size (`FollowerConfig::size_usdc`).
    side: OrderSide,
    token_id: String,
    condition_id: String,
    /// Timestamp of the trade as reported by the API (unix seconds), NOT `now()`.
    ts: i64,
    /// Dedup key: `transactionHash` if the API brought it; otherwise
    /// a composite `ts|asset|side` that only collapses identical events.
    dedup_key: String,
}

/// Queries one `/activity` page for `wallet`. `end` = backward pagination
/// cursor (unix seconds, events with `timestamp <= end`) —
/// verified with a real curl, not used in the normal poll (see the module's
/// limitation), available for tests/future extensions.
async fn fetch_activity(
    http: &reqwest::Client,
    wallet: &str,
    end: Option<i64>,
) -> anyhow::Result<Vec<ActivityEvent>> {
    let mut query: Vec<(&str, String)> = vec![
        ("user", wallet.to_string()),
        ("limit", ACTIVITY_PAGE_LIMIT.to_string()),
    ];
    if let Some(e) = end {
        query.push(("end", e.to_string()));
    }
    let events: Vec<ActivityEvent> = http
        .get(format!("{DATA_API_BASE}/activity"))
        .query(&query)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(events)
}

/// `true` if the raw page (newest-first, fixed size `ACTIVITY_PAGE_LIMIT`)
/// may be truncating new trades: full AND the OLDEST event it brings
/// is still later than `since_ts`. A page full of old events
/// (`timestamp <= since_ts`) is the normal scenario of any wallet with more than
/// `ACTIVITY_PAGE_LIMIT` trades in its history — it indicates no gap, only
/// that there was no new activity since the last poll. Pure function, testable.
fn page_might_have_gaps(events: &[ActivityEvent], since_ts: i64) -> bool {
    events.len() >= ACTIVITY_PAGE_LIMIT && events.last().is_some_and(|e| e.timestamp > since_ts)
}

/// Filters `type == "TRADE"` and `timestamp >= since_ts`, and converts each event
/// into a `WalletTrade`. Pure function (no I/O) — testable with captured real JSON.
///
/// The filter used to be a strict `>` — a late-indexed trade that
/// fell in the SAME second as the cursor was dropped forever
/// (the API sorts by timestamp with second resolution; two trades of the
/// wallet in the same second are common). With `>=` the events of the
/// boundary second come back on every poll and are discarded by `dedup_key`
/// (transactionHash) in `poll_once`, not by the cursor.
fn parse_activity_events(events: Vec<ActivityEvent>, since_ts: i64) -> Vec<WalletTrade> {
    events
        .into_iter()
        .filter(|e| e.kind == "TRADE" && e.timestamp >= since_ts)
        .filter_map(|e| {
            let (side, tag) = match e.side.as_str() {
                "BUY" => (OrderSide::Buy, "BUY"),
                "SELL" => (OrderSide::Sell { size: None }, "SELL"),
                other => {
                    warn!(
                        side = other,
                        "follower: unknown side in /activity — event ignored"
                    );
                    return None;
                }
            };
            let dedup_key = if e.transaction_hash.is_empty() {
                format!("{}|{}|{}", e.timestamp, e.asset, tag)
            } else {
                e.transaction_hash
            };
            Some(WalletTrade {
                side,
                token_id: e.asset,
                condition_id: e.condition_id,
                ts: e.timestamp,
                dedup_key,
            })
        })
        .collect()
}

// ── CLOB book (minimal copy of market_feed/resync.rs::fetch_book — not
// worth exposing that private function for this) ──

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

fn parse_raw_levels(raw: &[RawBookLevel]) -> Vec<Level> {
    use std::str::FromStr;
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

async fn fetch_book(
    http: &reqwest::Client,
    token_id: &str,
) -> anyhow::Result<(Vec<Level>, Vec<Level>)> {
    let body: ClobBookResponse = http
        .get(format!("{CLOB_BASE}/book"))
        .query(&[("token_id", token_id)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let mut bids = parse_raw_levels(&body.bids);
    let mut asks = parse_raw_levels(&body.asks);
    bids.sort_unstable_by_key(|l| std::cmp::Reverse(l.price)); // desc: best bid first
    asks.sort_unstable_by_key(|a| a.price); // asc: best ask first
    Ok((bids, asks))
}

// ── Market metadata (Gamma /markets?condition_ids=) ────────────────────────

/// `condition_ids` filter verified with a real curl against
/// `gamma-api.polymarket.com/markets` — no precedent in the repo, the rest
/// of the code filters by `active`/`closed`/`tag_id`. `condition_ids` (plural,
/// a single value in our scenario) returns exactly the requested market; a
/// nonexistent `condition_id` returns `[]` (not 404, no echo of unfiltered items).
#[derive(Debug, Deserialize)]
struct GammaMarketMeta {
    #[serde(rename = "clearAt")]
    clear_at: Option<String>,
    #[serde(rename = "endDate")]
    end_date: Option<String>,
    #[serde(rename = "negRisk", default)]
    neg_risk: bool,
}

async fn fetch_market_meta(
    http: &reqwest::Client,
    condition_id: &str,
) -> anyhow::Result<Option<GammaMarketMeta>> {
    let markets: Vec<GammaMarketMeta> = http
        .get(format!("{GAMMA_BASE}/markets"))
        .query(&[("condition_ids", condition_id)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(markets.into_iter().next())
}

/// `clearAt` can be null in recent markets (also seen in
/// `market_feed::discover::GammaMarket`) — fallback to `endDate`.
fn resolve_clear_at(meta: &GammaMarketMeta) -> Option<i64> {
    let parse = |s: &str| {
        chrono::DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|dt| dt.timestamp())
    };
    meta.clear_at
        .as_deref()
        .and_then(parse)
        .or_else(|| meta.end_date.as_deref().and_then(parse))
}

// ── OrderIntent construction (pure functions — testable without I/O) ───────

/// Buy: copies the target wallet's entry with our own size
/// (`cfg.size_usdc / ask`), FOK/taker (like NC/NE/ARB) — not GTD, we are not
/// trying to be a maker here, we are copying an entry.
fn build_buy_intent(
    cfg: &FollowerConfig,
    trade: &WalletTrade,
    ask: Decimal,
    clear_at: i64,
    neg_risk: bool,
) -> OrderIntent {
    let size = cfg.size_usdc / ask;
    OrderIntent {
        market_cid: trade.condition_id.clone(),
        token_id: trade.token_id.clone(),
        side: OrderSide::Buy,
        price: ask,
        size,
        clear_at,
        signal: ask,
        neg_risk,
        limit_price: Some(ask * (Decimal::ONE + cfg.max_slippage_pct)),
        second_leg: None,
        strategy: StrategyId::Follower,
        resting: false,
        // The follower never touches `market_feed`/`MarketRegistry` (by
        // design, see the module doc) so `handle_trade` never finds
        // the market there and falls back to clear_at=0 (same mechanism as
        // a race window, but permanent — the follower
        // NEVER registers the market, not just "not yet"). Another strategy has the same
        // problem and solves it this way: `real_clear_at` is the escape hatch
        // that `handle_trade` uses when the registry lookup fails.
        real_clear_at: Some(clear_at),
    }
}

/// Sell: closes the whole position attributable to `StrategyId::Follower`
/// for this token — same pattern (FOK/taker) as the bail-sells of the other
/// strategies (see `near_expiry.rs::sell_intent`, `limit_price: None` = sell
/// at market with no ceiling, `resting: false` makes the executor treat it
/// as `OrderKind::Sell` regardless of `limit_price`).
fn build_sell_intent(pos: &Position, trade: &WalletTrade, bid: Decimal) -> OrderIntent {
    OrderIntent {
        market_cid: pos.market_cid.clone(),
        token_id: trade.token_id.clone(),
        side: OrderSide::Sell {
            size: Some(pos.size),
        },
        price: bid,
        size: pos.size,
        clear_at: pos.clear_at,
        signal: bid,
        neg_risk: pos.neg_risk,
        limit_price: None,
        second_leg: None,
        strategy: StrategyId::Follower,
        resting: false,
        // Same escape hatch as build_buy_intent — see the comment there.
        real_clear_at: Some(pos.clear_at),
    }
}

/// `true` if `pos` represents a truly OPEN position (same criterion
/// as `StateStore::list_open_positions`: not redeemed, no previous bail-sell,
/// no settlement verdict, positive size) — `get_position` does not filter
/// this, it returns the row as is (including already closed positions).
fn is_open(pos: &Position) -> bool {
    pos.size > Decimal::ZERO && !pos.redeemed && !pos.bail_sold && pos.outcome_won.is_none()
}

// ── Orchestration (I/O) ──────────────────────────────────────────────────────

async fn handle_buy(
    http: &reqwest::Client,
    intent_tx: &mpsc::Sender<QueuedIntent>,
    cfg: &FollowerConfig,
    trade: &WalletTrade,
) {
    let (_, asks) = match fetch_book(http, &trade.token_id).await {
        Ok(b) => b,
        Err(e) => {
            warn!(token_id = %trade.token_id, error = %e, "follower: book query failed — buy discarded");
            return;
        }
    };
    let Some(ask) = asks.first().map(|l| l.price) else {
        warn!(token_id = %trade.token_id, "follower: book without asks — buy discarded");
        return;
    };
    if ask <= Decimal::ZERO {
        warn!(token_id = %trade.token_id, %ask, "follower: invalid ask — buy discarded");
        return;
    }

    let meta = match fetch_market_meta(http, &trade.condition_id).await {
        Ok(Some(m)) => m,
        Ok(None) => {
            warn!(cid = %trade.condition_id, "follower: market not found in Gamma — buy discarded");
            return;
        }
        Err(e) => {
            warn!(cid = %trade.condition_id, error = %e, "follower: Gamma metadata query failed — buy discarded");
            return;
        }
    };
    let Some(clear_at) = resolve_clear_at(&meta) else {
        warn!(cid = %trade.condition_id, "follower: no clearAt/endDate in Gamma — buy discarded");
        return;
    };

    let intent = build_buy_intent(cfg, trade, ask, clear_at, meta.neg_risk);

    // Instrumentation (empirical): real lag and price paid
    // vs. the target wallet's — input to compare against the viability model.
    // (follower viability study, 2026-08-04).
    let lag_ms = crate::time::now().timestamp_millis() - trade.ts * 1000;
    info!(
        target: "follower_copy",
        token_id = %trade.token_id, condition_id = %trade.condition_id,
        lag_ms, price_paid = %ask, size_usdc = %cfg.size_usdc,
        "follower: copying target wallet BUY"
    );

    if intent_tx
        .send((intent, std::time::Instant::now()))
        .await
        .is_err()
    {
        warn!("follower: executor closed, could not send the buy order");
    }
}

async fn handle_sell(
    http: &reqwest::Client,
    state: &StateStore,
    intent_tx: &mpsc::Sender<QueuedIntent>,
    trade: &WalletTrade,
) {
    let pos = match state
        .get_position(&trade.token_id, &StrategyId::Follower.to_string())
        .await
    {
        Ok(Some(p)) => p,
        Ok(None) => {
            info!(target: "follower_copy", token_id = %trade.token_id,
                "follower: sell detected with no registered position — ignored (we do not copy the entry)");
            return;
        }
        Err(e) => {
            warn!(token_id = %trade.token_id, error = %e, "follower: position query failed — sell discarded");
            return;
        }
    };
    if !is_open(&pos) {
        info!(target: "follower_copy", token_id = %trade.token_id,
            "follower: sell detected, position already closed/redeemed — ignored");
        return;
    }

    let (bids, _) = match fetch_book(http, &trade.token_id).await {
        Ok(b) => b,
        Err(e) => {
            warn!(token_id = %trade.token_id, error = %e, "follower: book query failed — sell discarded");
            return;
        }
    };
    let Some(bid) = bids.first().map(|l| l.price) else {
        warn!(token_id = %trade.token_id, "follower: book without bids — sell discarded");
        return;
    };

    let intent = build_sell_intent(&pos, trade, bid);

    let lag_ms = crate::time::now().timestamp_millis() - trade.ts * 1000;
    info!(
        target: "follower_copy",
        token_id = %trade.token_id, lag_ms, sell_price = %bid, size = %pos.size,
        "follower: copying target wallet SELL"
    );

    if intent_tx
        .send((intent, std::time::Instant::now()))
        .await
        .is_err()
    {
        warn!("follower: executor closed, could not send the sell order");
    }
}

/// The initial cursor used to be the `timestamp` of the wallet's last
/// event — if it had not traded for a while, the first poll re-executed ALL the
/// intermediate history (orders copied on signals from hours ago at the
/// current price). The module doc fixes "initial cursor = right now": the floor
/// is `now`, and the last event can only move it forward (future event due to
/// clock skew — conservative, copying from there).
fn initial_cursor(latest_event_ts: Option<i64>, now: i64) -> i64 {
    latest_event_ts.map_or(now, |ts| ts.max(now))
}

async fn fetch_latest_ts(http: &reqwest::Client, wallet: &str) -> anyhow::Result<i64> {
    let events = fetch_activity(http, wallet, None).await?;
    Ok(initial_cursor(
        events.first().map(|e| e.timestamp),
        crate::time::now().timestamp(),
    ))
}

/// One poll cycle: queries `/activity`, processes the new trades in
/// chronological order (buys before their matching sells) and returns the
/// updated cursor (`last_seen_ts`). `seen` deduplicates the events of the
/// boundary second.
async fn poll_once(
    http: &reqwest::Client,
    state: &StateStore,
    intent_tx: &mpsc::Sender<QueuedIntent>,
    cfg: &FollowerConfig,
    last_seen_ts: i64,
    seen: &mut std::collections::HashSet<String>,
) -> i64 {
    let events = match fetch_activity(http, &cfg.target_wallet_address, None).await {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "follower: /activity query failed — poll cycle skipped");
            return last_seen_ts;
        }
    };

    if page_might_have_gaps(&events, last_seen_ts) {
        warn!(
            n = events.len(),
            "follower: /activity page full AND the oldest event is still new — \
             intermediate trades may be missing (known limitation, no end= pagination)"
        );
    }

    let mut trades = parse_activity_events(events, last_seen_ts);
    // The `>=` filter re-delivers the events of the boundary second on
    // every poll — discard them by hash BEFORE touching the network, not by the cursor.
    trades.retain(|t| seen.insert(t.dedup_key.clone()));
    trades.sort_by_key(|t| t.ts); // chronological: buy before its sell

    let mut new_last_seen = last_seen_ts;
    for trade in &trades {
        new_last_seen = new_last_seen.max(trade.ts);
        match &trade.side {
            OrderSide::Buy => handle_buy(http, intent_tx, cfg, trade).await,
            OrderSide::Sell { .. } => {
                if cfg.mirror_exits {
                    handle_sell(http, state, intent_tx, trade).await;
                }
            }
            OrderSide::Cancel => {} // /activity never reports this — defensive
        }
    }
    new_last_seen
}

/// Standalone follower task. Spawned directly in
/// `main.rs` via `spawn_named!` (NOT `spawn_strategy!`, coupled to the `Strategy`
/// trait), conditioned on `config.follower.limits.enabled`.
pub async fn run(
    state: Arc<StateStore>,
    intent_tx: mpsc::Sender<QueuedIntent>,
    cfg: FollowerConfig,
    mut shutdown: watch::Receiver<bool>,
) {
    let http = reqwest::Client::builder()
        .user_agent(crate::POLY_USER_AGENT)
        .timeout(Duration::from_secs(10))
        .build()
        .expect("follower: reqwest client");

    info!(
        wallet = %cfg.target_wallet_address,
        poll_interval_secs = cfg.poll_interval_secs,
        "follower: starting"
    );

    // Initial cursor = right now: the follower does not persist across restarts (known
    // limitation, see the module doc-comment) — if the initial fetch fails, it falls back to
    // now() anyway (same effect: do not reprocess history).
    let mut last_seen_ts = match fetch_latest_ts(&http, &cfg.target_wallet_address).await {
        Ok(ts) => ts,
        Err(e) => {
            warn!(error = %e, "follower: failed to get the initial cursor — using now()");
            crate::time::now().timestamp()
        }
    };

    // Dedup of the boundary-second events (`ts >= last_seen_ts`
    // are re-delivered every poll). Bounded by the wallet's volume; it
    // resets if it grows too large — old hashes already fall below the cursor.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut ticker = tokio::time::interval(Duration::from_secs(cfg.poll_interval_secs.max(1)));
    ticker.tick().await; // skip the first immediate tick

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => { if *shutdown.borrow() { break; } }
            _ = ticker.tick() => {
                last_seen_ts =
                    poll_once(&http, &state, &intent_tx, &cfg, last_seen_ts, &mut seen).await;
                if seen.len() > 50_000 {
                    seen.clear();
                }
            }
        }
    }
    info!("follower: stopped");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // ── /activity parsing (real JSON captured by curl) ─────────────────────

    /// Three real events from the target wallet (`curl
    /// "https://data-api.polymarket.com/activity?user=<wallet>&limit=3"`),
    /// trimmed of irrelevant fields but keeping the exact shape
    /// (includes fields the parser ignores, to verify that serde
    /// does not fail with `#[serde(deny_unknown_fields)]` absent — we never set it).
    const ACTIVITY_JSON_REAL: &str = r#"[
        {"proxyWallet":"0x00000000000000000000000000000000000000a1","timestamp":1785970653,"conditionId":"0xf55b670c4b8c00567c232a1ead0e971433ea2e1e7046f0ecf53da02ec9147155","type":"TRADE","size":42.208954,"usdcSize":28.746609,"transactionHash":"0xd81ef408ae7ef599aedef87d03b22de60df00f5cb245e2fb199866240eed32aa","price":0.6699999957,"asset":"5480434951148668608464777387792690607932611394316569163168634063975333903237","side":"BUY","outcomeIndex":0,"title":"Quilmes AC vs. Gimnasia y Esgrima de Jujuy: O/U 0.5","slug":"argpn-qui-ge-2026-08-05-total-0pt5","icon":"","eventSlug":"argpn-qui-ge-2026-08-05-more-markets","outcome":"Over","name":"example-trader","pseudonym":"Example-Trader","bio":"","profileImage":"","profileImageOptimized":""},
        {"proxyWallet":"0x00000000000000000000000000000000000000a1","timestamp":1785970652,"conditionId":"0xa7ac1c8effa5ee3fb9f327505f40b408065a7ce9970d2dd25dadd14880fa2c22","type":"TRADE","size":260.97,"usdcSize":101.7783,"transactionHash":"0x422072aff3321598890de1569e93dba1e77c7ac704ae27f7acb8bc83c48f7a0c","price":0.39,"asset":"16626392403260047672118159314461443195256986125632525413729854953336297651582","side":"SELL","outcomeIndex":1,"title":"National Bank Open: Fabian Marozsan vs Matteo Arnaldi","slug":"atp-marozsa-arnaldi-2026-08-05","icon":"","eventSlug":"atp-marozsa-arnaldi-2026-08-05","outcome":"Matteo Arnaldi","name":"example-trader","pseudonym":"Example-Trader","bio":"","profileImage":"","profileImageOptimized":""},
        {"proxyWallet":"0x00000000000000000000000000000000000000a1","timestamp":1785970649,"conditionId":"0xd62da19be9400420588aeba915c8c31e27500de60c5b6d5e356a0626c11ee9d5","type":"TRADE","size":119.96,"usdcSize":85.23158,"transactionHash":"0xe8df86a5442073b0140c185e726643270c752322f54a7a2669b960f1abc5177f","price":0.7,"asset":"75145629553703620009515306488239444065687261793577691853145683394154366222281","side":"BUY","outcomeIndex":0,"title":"National Bank Open: Ugo Humbert vs Daniel Merida Aguilar","slug":"atp-humbert-aguilar-2026-08-06","icon":"","eventSlug":"atp-humbert-aguilar-2026-08-06","outcome":"Ugo Humbert","name":"example-trader","pseudonym":"Example-Trader","bio":"","profileImage":"","profileImageOptimized":""}
    ]"#;

    #[test]
    fn parse_activity_json_real_deserializes_and_filters() {
        let events: Vec<ActivityEvent> = serde_json::from_str(ACTIVITY_JSON_REAL).unwrap();
        assert_eq!(events.len(), 3);

        // since_ts = 0 → all three pass the kind/time filter.
        let trades = parse_activity_events(events.clone(), 0);
        assert_eq!(trades.len(), 3);
        assert!(matches!(trades[0].side, OrderSide::Buy));
        assert!(matches!(trades[1].side, OrderSide::Sell { size: None }));
        assert_eq!(
            trades[0].token_id,
            "5480434951148668608464777387792690607932611394316569163168634063975333903237"
        );
        assert_eq!(
            trades[0].condition_id,
            "0xf55b670c4b8c00567c232a1ead0e971433ea2e1e7046f0ecf53da02ec9147155"
        );
        assert_eq!(trades[0].ts, 1_785_970_653);

        // since_ts = the second event's ts → `>=` includes those of the
        // boundary second; dedup by transactionHash discards them in
        // poll_once, not here.
        let only_new = parse_activity_events(events, 1_785_970_652);
        assert_eq!(only_new.len(), 2);
        assert_eq!(only_new[0].ts, 1_785_970_653);
        assert_eq!(only_new[1].ts, 1_785_970_652);
        // The dedup_key is the event's real transactionHash.
        assert_eq!(
            only_new[0].dedup_key,
            "0xd81ef408ae7ef599aedef87d03b22de60df00f5cb245e2fb199866240eed32aa"
        );
    }

    /// A trade indexed late in the same second as the cursor is
    /// no longer lost — `>=` re-delivers it and the hash dedup decides.
    #[test]
    fn boundary_second_event_survives_filter() {
        let raw = vec![
            ActivityEvent {
                kind: "TRADE".into(),
                side: "BUY".into(),
                asset: "tok".into(),
                condition_id: "cid".into(),
                timestamp: 100,
                transaction_hash: "0xaaa".into(),
            },
            ActivityEvent {
                kind: "TRADE".into(),
                side: "SELL".into(),
                asset: "tok".into(),
                condition_id: "cid".into(),
                timestamp: 100,
                transaction_hash: "0xbbb".into(),
            },
        ];
        let trades = parse_activity_events(raw, 100);
        assert_eq!(trades.len(), 2, "both share the cursor's second");
    }

    /// Event without transactionHash: the dedup_key falls back to `ts|asset|side` —
    /// two distinct events without a hash do not collapse into each other.
    #[test]
    fn dedup_key_without_hash_uses_composite() {
        let raw = vec![
            ActivityEvent {
                kind: "TRADE".into(),
                side: "BUY".into(),
                asset: "tok".into(),
                condition_id: "cid".into(),
                timestamp: 100,
                transaction_hash: String::new(),
            },
            ActivityEvent {
                kind: "TRADE".into(),
                side: "SELL".into(),
                asset: "tok".into(),
                condition_id: "cid".into(),
                timestamp: 100,
                transaction_hash: String::new(),
            },
        ];
        let trades = parse_activity_events(raw, 0);
        assert_ne!(trades[0].dedup_key, trades[1].dedup_key);
    }

    #[test]
    fn parse_activity_events_ignores_unknown_side() {
        let raw = vec![ActivityEvent {
            kind: "TRADE".into(),
            side: "MERGE".into(),
            asset: "tok".into(),
            condition_id: "cid".into(),
            timestamp: 100,
            transaction_hash: "0x1".into(),
        }];
        assert!(parse_activity_events(raw, 0).is_empty());
    }

    #[test]
    fn parse_activity_events_ignores_non_trade_type() {
        let raw = vec![ActivityEvent {
            kind: "REDEEM".into(),
            side: "BUY".into(),
            asset: "tok".into(),
            condition_id: "cid".into(),
            timestamp: 100,
            transaction_hash: "0x1".into(),
        }];
        assert!(parse_activity_events(raw, 0).is_empty());
    }

    // ── initial cursor ───────────────────────────────────────────────────────

    /// The initial cursor is `now` even if the wallet's last activity was
    /// hours ago — otherwise the first poll re-executes the whole
    /// intermediate history (stale signals at the current price).
    #[test]
    fn initial_cursor_does_not_repeat_history() {
        let now = 10_000;
        assert_eq!(initial_cursor(Some(100), now), now);
        assert_eq!(initial_cursor(None, now), now);
        // Clock skew: an event with a future ts moves the cursor forward
        // (conservative — copying from there, not backward).
        assert_eq!(initial_cursor(Some(20_000), now), 20_000);
    }

    // ── page_might_have_gaps ──────────────────────────────────────────────────

    fn page_of(n: usize, oldest_ts: i64) -> Vec<ActivityEvent> {
        // newest-first, like the real API: the last element is the oldest.
        (0..n)
            .rev()
            .map(|i| ActivityEvent {
                kind: "TRADE".into(),
                side: "BUY".into(),
                asset: "tok".into(),
                condition_id: "cid".into(),
                timestamp: oldest_ts + i as i64,
                transaction_hash: format!("0x{i}"),
            })
            .collect()
    }

    /// Real case seen in manual verification: a very
    /// active wallet always returns a page full of OLD HISTORY when there were no
    /// new trades since the last poll — it must not be flagged as a gap.
    #[test]
    fn full_page_of_old_events_is_not_gap() {
        let events = page_of(ACTIVITY_PAGE_LIMIT, 100); // oldest = ts 100
        assert!(
            !page_might_have_gaps(&events, 10_000),
            "all events are <= since_ts, no gap"
        );
    }

    /// Full page and the oldest event is STILL new (> since_ts) → there
    /// may be intermediate trades truncated outside the page.
    #[test]
    fn full_page_with_oldest_still_new_is_gap() {
        let events = page_of(ACTIVITY_PAGE_LIMIT, 100);
        assert!(
            page_might_have_gaps(&events, 50),
            "the oldest (ts=100) > since_ts (50) → possible gap"
        );
    }

    /// A non-full page (fewer than ACTIVITY_PAGE_LIMIT) is never a gap, even if everything is new.
    #[test]
    fn non_full_page_is_never_gap() {
        let events = page_of(3, 100);
        assert!(!page_might_have_gaps(&events, 0));
    }

    // ── OrderIntent construction (buy) ───────────────────────────────────────

    fn cfg_test() -> FollowerConfig {
        FollowerConfig {
            size_usdc: dec!(10),
            max_slippage_pct: dec!(0.02),
            ..FollowerConfig::default()
        }
    }

    fn buy_trade() -> WalletTrade {
        WalletTrade {
            side: OrderSide::Buy,
            token_id: "tok-1".into(),
            condition_id: "cid-1".into(),
            ts: 1_000,
            dedup_key: "0xtrade".into(),
        }
    }

    #[test]
    fn build_buy_intent_own_size_and_limit_price() {
        let cfg = cfg_test();
        let trade = buy_trade();
        let intent = build_buy_intent(&cfg, &trade, dec!(0.50), 5_000, true);

        assert_eq!(intent.market_cid, "cid-1");
        assert_eq!(intent.token_id, "tok-1");
        assert!(matches!(intent.side, OrderSide::Buy));
        assert_eq!(intent.price, dec!(0.50));
        // size_usdc(10) / ask(0.50) = 20 shares — independent of the target wallet's size.
        assert_eq!(intent.size, dec!(20));
        assert_eq!(intent.clear_at, 5_000);
        assert!(intent.neg_risk);
        assert_eq!(intent.limit_price, Some(dec!(0.51))); // 0.50 * 1.02
        assert_eq!(intent.strategy, StrategyId::Follower);
        assert!(
            !intent.resting,
            "FOK/taker, not GTD — we copy an entry, we are not a maker"
        );
        assert!(intent.second_leg.is_none());
        assert_eq!(
            intent.real_clear_at,
            Some(5_000),
            "without this, handle_trade does not find the market in the registry \
             (the follower never registers it) and falls back to clear_at=0 — orphan position forever"
        );
    }

    // ── OrderIntent construction (sell) ──────────────────────────────────────

    fn open_position() -> Position {
        Position {
            token_id: "tok-1".into(),
            market_cid: "cid-1".into(),
            slug: "some-slug".into(),
            strategy: "follower".into(),
            size: dec!(20),
            avg_price: dec!(0.50),
            cost_basis: dec!(10),
            opened_ts_ms: 1_000_000,
            clear_at: 5_000,
            neg_risk: true,
            bail_sold: false,
            outcome_won: None,
            winner_label: None,
            outcome_label: None,
            redeemed: false,
            redeem_tx: None,
        }
    }

    #[test]
    fn build_sell_intent_sells_full_position() {
        let pos = open_position();
        let trade = WalletTrade {
            side: OrderSide::Sell { size: None },
            token_id: "tok-1".into(),
            condition_id: "cid-1".into(),
            ts: 2_000,
            dedup_key: "0xsell".into(),
        };
        let intent = build_sell_intent(&pos, &trade, dec!(0.60));

        assert_eq!(intent.market_cid, "cid-1");
        assert_eq!(intent.token_id, "tok-1");
        assert!(matches!(intent.side, OrderSide::Sell { size: Some(s) } if s == dec!(20)));
        assert_eq!(intent.price, dec!(0.60));
        assert_eq!(intent.size, dec!(20));
        assert_eq!(intent.clear_at, 5_000);
        assert!(intent.neg_risk);
        assert_eq!(
            intent.limit_price, None,
            "sell at market — resting=false ignores limit_price in the executor anyway"
        );
        assert_eq!(intent.strategy, StrategyId::Follower);
        assert!(!intent.resting);
        assert_eq!(
            intent.real_clear_at,
            Some(5_000),
            "same escape hatch as the buy"
        );
    }

    #[test]
    fn is_open_rejects_closed_redeemed_or_empty_position() {
        assert!(is_open(&open_position()));

        let mut sold = open_position();
        sold.bail_sold = true;
        assert!(!is_open(&sold));

        let mut redeemed = open_position();
        redeemed.redeemed = true;
        assert!(!is_open(&redeemed));

        let mut resolved = open_position();
        resolved.outcome_won = Some(true);
        assert!(!is_open(&resolved));

        let mut empty = open_position();
        empty.size = Decimal::ZERO;
        assert!(!is_open(&empty));
    }

    // ── scenario "sell with no open position → does nothing" ──────────────────

    /// `handle_sell` must check the position BEFORE touching the network (book
    /// fetch) — without this, this test would need an HTTP mock that does not exist in
    /// this repo. No registered position → no intent over `intent_tx`.
    #[tokio::test]
    async fn handle_sell_without_position_sends_no_intent() {
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let (intent_tx, mut intent_rx) = mpsc::channel::<QueuedIntent>(4);
        let http = reqwest::Client::new();
        let trade = WalletTrade {
            side: OrderSide::Sell { size: None },
            token_id: "tok-no-position".into(),
            condition_id: "cid-x".into(),
            ts: 1_000,
            dedup_key: "0x1".into(),
        };

        handle_sell(&http, &state, &intent_tx, &trade).await;

        drop(intent_tx);
        assert!(
            intent_rx.recv().await.is_none(),
            "with no open position no OrderIntent must be sent"
        );
    }

    /// Same scenario but with an ALREADY closed position (bail_sold) attributable to
    /// Follower for that token — it must not send anything either.
    #[tokio::test]
    async fn handle_sell_with_closed_position_sends_no_intent() {
        let state = StateStore::open_memory(crate::types::Mode::Paper)
            .await
            .unwrap();
        let (intent_tx, mut intent_rx) = mpsc::channel::<QueuedIntent>(4);
        let http = reqwest::Client::new();

        // Buy fill → open position, then closed via a simulated bail-sell.
        let buy_fill = crate::types::Fill {
            market_cid: "cid-1".into(),
            token_id: "tok-1".into(),
            side: crate::types::FillSide::Buy,
            price: dec!(0.5),
            size: dec!(20),
            order_id: Some("o1".into()),
            ts: crate::time::now(),
            mode: crate::types::Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::Follower,
            clear_at: 5_000,
            outcome_label: Some("YES".into()),
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
        };
        state.apply_fill(&buy_fill).await.unwrap();
        let sell_fill = crate::types::Fill {
            side: crate::types::FillSide::Sell,
            ..buy_fill
        };
        state.apply_fill(&sell_fill).await.unwrap();

        let trade = WalletTrade {
            side: OrderSide::Sell { size: None },
            token_id: "tok-1".into(),
            condition_id: "cid-1".into(),
            ts: 2_000,
            dedup_key: "0x2".into(),
        };
        handle_sell(&http, &state, &intent_tx, &trade).await;

        drop(intent_tx);
        assert!(
            intent_rx.recv().await.is_none(),
            "an already closed position (size=0 after selling everything) must not be re-sent"
        );
    }
}
