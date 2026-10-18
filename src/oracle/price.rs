//! Multi-asset price feed via Binance WebSocket (combined stream `<sym>@trade`).
//!
//! Publishes each asset's price on `watch::Sender<AssetPrice>`.
//! Assets are derived from `market_feed.bracket_slug_patterns`
//! (prefix before `-updown`): `btc-updown` → BTCUSDT, `eth-updown` → ETHUSDT.

use crate::time::Instant;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::watch;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::config::OracleConfig;
use crate::oracle::asset_from_slug;
use crate::research::contract::{EventKind, EventSource};
use crate::research::events::{stable_hash, EventInput, EventProducer, SourceContext};
use crate::types::AssetPrice;

const MAX_BACKOFF_SECS: u64 = 30;

/// Builds the Binance combined stream URL.
fn build_url(assets: &[String], base: &str) -> String {
    let streams: Vec<String> = assets
        .iter()
        .map(|a| format!("{}usdt@trade", a.to_lowercase()))
        .collect();
    format!("{base}/stream?streams={}", streams.join("/"))
}

/// Creates the watch channels (sender, receiver) for each asset in `patterns`.
pub fn create_price_channels(
    patterns: &[String],
) -> (
    HashMap<String, watch::Sender<AssetPrice>>,
    HashMap<String, watch::Receiver<AssetPrice>>,
) {
    let assets: HashSet<String> = patterns.iter().filter_map(|p| asset_from_slug(p)).collect();

    let mut txs = HashMap::with_capacity(assets.len());
    let mut rxs = HashMap::with_capacity(assets.len());
    for asset in assets {
        let (tx, rx) = watch::channel(AssetPrice {
            asset: asset.clone(),
            ..AssetPrice::default()
        });
        txs.insert(asset.clone(), tx);
        rxs.insert(asset, rx);
    }
    (txs, rxs)
}

// ── Deserialization types ─────────────────────────────────────────────────────

#[derive(Deserialize)]
struct BinanceTrade {
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "s")]
    symbol: String,
    // Trade quantity (base asset) — for the CVD notional.
    #[serde(rename = "q")]
    qty: String,
    // true if the buyer is the maker ⇒ the aggressor (taker) SELLS.
    #[serde(rename = "m")]
    buyer_is_maker: bool,
    // Time of the TRADE in Binance's engine (ms), not the local
    // receive instant — the `@trade` stream always sends it.
    #[serde(rename = "T")]
    trade_time: i64,
    // Trade id of the venue (`t`) — the stable dedup key of the research
    // recorder (a post-reconnection resend carries the SAME id, so the
    // dedup hooks in without depending on recv_ts/local_seq). Option because the
    // field was not used before: if some payload lacks it, dedup falls back to
    // the trade's stable fields (see `event_from_binance_tick`).
    #[serde(rename = "t", default)]
    trade_id: Option<i64>,
}

/// Combined stream message: `{"stream":"btcusdt@trade","data":{...}}`
#[derive(Deserialize)]
struct CombinedMsg {
    data: BinanceTrade,
}

// ── REST bootstrap ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct BinanceTicker {
    price: String,
}

/// Binance REST fetch to publish an initial price before the WS delivers the first one.
/// This removes the startup window where `vol.last_price = 0` and the
/// NC/NE strategies cannot compute `dist_pct` or `z`.
///
/// Besides bootstrapping the price, this function ALSO decides which
/// symbols are valid for the Binance combined stream — previously `run()`
/// included ALL configured assets without checking, and a single symbol with no
/// Binance counterpart (e.g. a new stock ladder, `nvda-above`) makes
/// Binance reject the WHOLE combined stream: `run_session` falls into a
/// reconnect loop and ALL assets lose their price, not just the invalid one.
///
/// Returns the list of assets that must be included in the combined stream.
/// It distinguishes two kinds of failure:
/// - Network error/timeout: AMBIGUOUS (not evidence that the symbol does not exist)
///   → included anyway (fail-open), so as not to lose a valid asset to a
///   transient blip.
/// - Non-success HTTP status (Binance returns 400 "Invalid symbol" for pairs
///   that do not exist): a DETERMINISTIC signal that this asset would take down the
///   whole stream → excluded, with `error!` (not `warn!`: it means that asset
///   is left without an oracle until the config is fixed).
async fn bootstrap_prices(
    assets: &[String],
    txs: &HashMap<String, watch::Sender<AssetPrice>>,
) -> Vec<String> {
    let mut valid = Vec::with_capacity(assets.len());
    let Ok(http) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    else {
        tracing::warn!("oracle: could not build the bootstrap HTTP client — symbol validation skipped, all are included unverified");
        return assets.to_vec();
    };

    for asset in assets {
        let symbol = format!("{}USDT", asset.to_uppercase());
        let url = format!("https://api.binance.com/api/v3/ticker/price?symbol={symbol}");
        let resp = match http.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(asset, error = %e, "oracle: REST bootstrap unreachable — included unverified");
                valid.push(asset.clone());
                continue;
            }
        };
        if !resp.status().is_success() {
            tracing::error!(
                asset, status = %resp.status(),
                "oracle: symbol without a valid Binance counterpart — excluded from the combined stream"
            );
            continue;
        }
        // Previously `let Ok(...) else { continue }` swallowed the parse
        // failure without any warn — now it is explicit.
        let ticker = match resp.json::<BinanceTicker>().await {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(asset, error = %e, "oracle: bootstrap response unreadable — included unverified");
                valid.push(asset.clone());
                continue;
            }
        };
        let Ok(price) = ticker.price.parse::<Decimal>() else {
            tracing::warn!(asset, raw = %ticker.price, "oracle: bootstrap price unreadable — included unverified");
            valid.push(asset.clone());
            continue;
        };
        if price <= Decimal::ZERO {
            tracing::warn!(
                asset,
                "oracle: bootstrap price <= 0 — included unverified"
            );
            valid.push(asset.clone());
            continue;
        }
        if let Some(tx) = txs.get(asset) {
            let _ = tx.send(AssetPrice {
                asset: asset.clone(),
                price,
                updated_at: Utc::now(),
                instant: Some(Instant::now()),
                // The REST bootstrap carries no trades, the flow starts at zero.
                ..AssetPrice::default()
            });
            tracing::info!(asset, price = %price, "oracle: price bootstrap via REST");
        }
        valid.push(asset.clone());
    }
    valid
}

// ── Main task ─────────────────────────────────────────────────────────────────

/// Spawns the price feed and does not return (only on shutdown or fatal error).
/// Automatic reconnection with exponential backoff (cap 30 s).
pub async fn run(
    cfg: Arc<OracleConfig>,
    txs: HashMap<String, watch::Sender<AssetPrice>>,
    shutdown: tokio::sync::watch::Receiver<bool>,
) {
    run_with_recorder(cfg, txs, shutdown, EventProducer::sink()).await
}

/// Same as [`run`] but also emits each Binance trade to the research
/// recorder — the per-session `SourceContext` carries a new generation
/// on every reconnection (the venue's seq is only compared within the
/// same epoch) and the per-symbol stream `binance:<asset>`.
///
/// Wired in `main.rs`: it invokes it with `recorder.producer()`
/// when `[research_events]` is active. The producer always exists
/// (a `sink()` discards at no cost with the hub inactive).
pub async fn run_with_recorder(
    cfg: Arc<OracleConfig>,
    txs: HashMap<String, watch::Sender<AssetPrice>>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
    recorder: EventProducer,
) {
    if txs.is_empty() {
        tracing::warn!("oracle::price: no assets configured, exiting");
        return;
    }

    let assets: Vec<String> = txs.keys().cloned().collect();
    // Only the assets that bootstrap_prices confirms valid (or
    // ambiguous due to a network failure) enter the combined stream — an invalid symbol
    // can no longer take down the WHOLE feed.
    let valid_assets = bootstrap_prices(&assets, &txs).await;
    if valid_assets.is_empty() {
        tracing::error!("oracle::price: no configured asset has a valid Binance symbol — the feed does not start");
        return;
    }
    let url = build_url(&valid_assets, &cfg.ws_url);
    let stale = std::time::Duration::from_secs(cfg.stale_secs);
    let ping = std::time::Duration::from_secs(cfg.ping_interval_secs);
    let min_session = std::time::Duration::from_secs(cfg.min_session_secs_for_backoff_reset);

    let mut backoff = std::time::Duration::from_secs(1);
    let mut last_price: HashMap<String, Decimal> = HashMap::new();
    // Signed net flow (net, total) accumulated in USDT per asset.
    // Lives in the OUTER loop to survive reconnections — only reset with
    // the process (as the spec decided, so as not to lose the series over a WS blip).
    let mut cum_flow: HashMap<String, (Decimal, Decimal)> = HashMap::new();

    loop {
        tracing::info!(url, "oracle::price: connecting");
        let session_start = Instant::now();

        // A NEW generation per WS session (each reconnection) — the venue's
        // seq is only compared within the same epoch in the writer.
        let ctx = SourceContext {
            producer: recorder.clone(),
            generation: crate::research::events::new_generation(),
            stream: "binance".to_string(),
        };

        match run_session(
            &url,
            &txs,
            &cfg,
            stale,
            ping,
            &mut last_price,
            &mut cum_flow,
            &mut shutdown,
            &ctx,
        )
        .await
        {
            Ok(()) => {
                tracing::info!("oracle::price: clean shutdown");
                return;
            }
            Err(e) => {
                tracing::warn!(error = %e, backoff_secs = backoff.as_secs(), "oracle::price: session ended");
                metrics::counter!("hrf_oracle_stale_reconnect_total").increment(1);
            }
        }

        if session_start.elapsed() >= min_session {
            backoff = std::time::Duration::from_secs(1);
        }

        tokio::select! {
            _ = tokio::time::sleep(backoff) => {}
            _ = shutdown.changed() => {
                if *shutdown.borrow() { return; }
            }
        }
        backoff = (backoff * 2).min(std::time::Duration::from_secs(MAX_BACKOFF_SECS));
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_session(
    url: &str,
    txs: &HashMap<String, watch::Sender<AssetPrice>>,
    cfg: &OracleConfig,
    stale: std::time::Duration,
    ping_interval: std::time::Duration,
    last_price: &mut HashMap<String, Decimal>,
    cum_flow: &mut HashMap<String, (Decimal, Decimal)>,
    shutdown: &mut tokio::sync::watch::Receiver<bool>,
    rec: &SourceContext,
) -> anyhow::Result<()> {
    let (mut ws, _) = connect_async(url).await?;
    let mut last_msg = Instant::now();
    let mut ping_tick = tokio::time::interval(ping_interval);

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() { return Ok(()); }
            }

            _ = ping_tick.tick() => {
                if last_msg.elapsed() > stale {
                    anyhow::bail!("oracle: zombie feed (no messages for {:.0}s)", stale.as_secs_f64());
                }
                ws.send(Message::Ping(vec![])).await?;
            }

            msg = ws.next() => {
                match msg {
                    None => anyhow::bail!("oracle: WS closed by the server"),
                    Some(Err(e)) => anyhow::bail!("oracle: WS error: {e}"),
                    Some(Ok(Message::Text(txt))) => {
                        last_msg = Instant::now();
                        // Local receipt stamped when the frame arrives —
                        // it is the research event's recv_ts_ms.
                        // `crate::time::now` (injectable): wall clock in production,
                        // tape if this path ran under the replay clock.
                        let recv_ts_ms = crate::time::now().timestamp_millis();
                        if let Ok(msg) = serde_json::from_str::<CombinedMsg>(&txt) {
                            process_trade(
                                msg.data,
                                txs,
                                last_price,
                                cum_flow,
                                &cfg.max_jump_ratio,
                                rec,
                                recv_ts_ms,
                            );
                        }
                    }
                    Some(Ok(Message::Pong(_))) => { last_msg = Instant::now(); }
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

/// Pure adapter — builds the `EventInput` of a Binance trade
/// from its RAW fields (strings, not re-parsed in the payload).
///
/// - `origin_ts_ms` = `trade_time` (`T` of the venue — real trade timestamp
///   in Binance's engine).
/// - `source_seq` = `trade_id` (`t`): the venue's per-symbol seq — the
///   writer detects gaps/out-of-order with it, ONLY within the same
///   connection `generation`.
/// - `dedup_key` = hash of the venue's trade id if it exists; otherwise of the
///   trade's stable fields. It NEVER includes `recv_ts_ms` or `local_seq`
///   (a key with a temporal component never hooks the dedup).
/// - `stream` = `"<ctx.stream>:<asset>"` — the seq is per symbol.
#[allow(clippy::too_many_arguments)]
pub fn event_from_binance_tick(
    symbol: &str,
    price: &str,
    qty: &str,
    buyer_is_maker: bool,
    trade_time: i64,
    trade_id: Option<i64>,
    recv_ts_ms: i64,
    ctx: &SourceContext,
) -> EventInput {
    let asset = symbol.trim_end_matches("USDT").to_lowercase();
    let payload = serde_json::json!({
        "symbol": symbol,
        "price": price,
        "qty": qty,
        "buyer_is_maker": buyer_is_maker,
        "trade_time": trade_time,
        "trade_id": trade_id,
    });
    let dedup_key = match trade_id {
        Some(t) => stable_hash(("binance", &asset, t)),
        None => stable_hash(("binance", &asset, price, qty, trade_time, buyer_is_maker)),
    };
    let mut e = EventInput::new(
        EventSource::BinanceWs,
        EventKind::PriceTick,
        payload,
        recv_ts_ms,
    );
    e.asset = Some(asset.clone());
    e.origin_ts_ms = Some(trade_time);
    e.stream = Some(format!("{}:{}", ctx.stream, asset));
    e.generation = Some(ctx.generation);
    e.source_seq = trade_id;
    e.dedup_key = Some(dedup_key);
    e
}

#[allow(clippy::too_many_arguments)]
fn process_trade(
    trade: BinanceTrade,
    txs: &HashMap<String, watch::Sender<AssetPrice>>,
    last_price: &mut HashMap<String, Decimal>,
    cum_flow: &mut HashMap<String, (Decimal, Decimal)>,
    max_jump: &Decimal,
    rec: &SourceContext,
    recv_ts_ms: i64,
) {
    let asset = trade.symbol.trim_end_matches("USDT").to_lowercase();

    // Record the tick at the EDGE — before ANY validation
    // (unreadable/zero price included — an invalid print also arrived over
    // the socket; the payload keeps the raw strings) and before the outlier
    // filter, to keep the print as it arrived (the replay
    // re-applies max_jump and reproduces the bot's view). `avail = recv`: there
    // is no intermediate queue between frame arrival and this emission.
    // The producer always exists — `is_active()` avoids
    // building the `EventInput` (payload + dedup_key) with the hub off.
    if rec.producer.is_active() {
        rec.producer.record(event_from_binance_tick(
            &trade.symbol,
            &trade.price,
            &trade.qty,
            trade.buyer_is_maker,
            trade.trade_time,
            trade.trade_id,
            recv_ts_ms,
            rec,
        ));
    }

    let Ok(price) = trade.price.parse::<Decimal>() else {
        return;
    };
    if price <= Decimal::ZERO {
        return;
    }

    // Outlier filter: discard prints with a jump > max_jump_ratio
    if let Some(&prev) = last_price.get(&asset) {
        if prev > Decimal::ZERO {
            let jump = ((price - prev) / prev).abs();
            if jump > *max_jump {
                metrics::counter!("hrf_oracle_outlier_total", "asset" => asset.clone())
                    .increment(1);
                return;
            }
        }
    }

    last_price.insert(asset.clone(), price);

    // After the outlier filter so false prints do not accumulate
    // volume. Notional in USDT (comparable across assets); `buyer_is_maker=true`
    // ⇒ the aggressor (taker) SELLS ⇒ negative flow. Invalid qty ⇒ notional 0 (does not
    // contribute, but the price is still published: an absent signal does not block).
    let qty = trade.qty.parse::<Decimal>().unwrap_or(Decimal::ZERO);
    let notional = qty * price;
    let signed = if trade.buyer_is_maker {
        -notional
    } else {
        notional
    };
    let entry = cum_flow
        .entry(asset.clone())
        .or_insert((Decimal::ZERO, Decimal::ZERO));
    entry.0 += signed;
    entry.1 += notional;
    let (cum_flow_quote, cum_vol_quote) = *entry;

    if let Some(tx) = txs.get(&asset) {
        // Liveness: epoch of the last tick published per asset — to alert
        // on a silent Binance feed (strategies would see stale vol
        // but the gauge distinguishes "no ticks" from "stalled ticks").
        metrics::gauge!("hrf_oracle_last_tick_unix_seconds", "asset" => asset.clone())
            .set(Utc::now().timestamp() as f64);
        let _ = tx.send(AssetPrice {
            asset,
            price,
            updated_at: Utc::now(),
            instant: Some(Instant::now()),
            cum_flow_quote,
            cum_vol_quote,
            // Timestamp of the TRADE on the exchange, not of receipt.
            exchange_ts_ms: Some(trade.trade_time),
        });
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    /// Sink context for the `process_trade` tests — the tape
    /// hook discards at no cost (the producer is no longer an `Option`).
    fn ctx_sink() -> SourceContext {
        SourceContext {
            producer: EventProducer::sink(),
            generation: 0,
            stream: "binance".into(),
        }
    }

    fn make_txs(
        asset: &str,
    ) -> (
        HashMap<String, watch::Sender<AssetPrice>>,
        watch::Receiver<AssetPrice>,
    ) {
        let (tx, rx) = watch::channel(AssetPrice {
            asset: asset.to_string(),
            ..AssetPrice::default()
        });
        let mut txs = HashMap::new();
        txs.insert(asset.to_string(), tx);
        (txs, rx)
    }

    #[test]
    fn trade_deserializes_qty_and_buyer_is_maker() {
        let json = r#"{"stream":"btcusdt@trade","data":{"p":"50000.5","s":"BTCUSDT","q":"0.001","m":true,"T":1784652361123}}"#;
        let msg: CombinedMsg = serde_json::from_str(json).expect("must deserialize");
        assert_eq!(msg.data.price, "50000.5");
        assert_eq!(msg.data.symbol, "BTCUSDT");
        assert_eq!(msg.data.qty, "0.001");
        assert!(msg.data.buyer_is_maker);
    }

    /// `T` (trade time of Binance's engine) is deserialized
    /// and reaches `AssetPrice::exchange_ts_ms` intact.
    #[test]
    fn trade_deserializes_trade_time_and_reaches_exchange_ts_ms() {
        let json = r#"{"stream":"btcusdt@trade","data":{"p":"50000.5","s":"BTCUSDT","q":"0.001","m":true,"T":1784652361123}}"#;
        let msg: CombinedMsg = serde_json::from_str(json).expect("must deserialize");
        assert_eq!(msg.data.trade_time, 1784652361123);

        let (txs, mut rx) = make_txs("btc");
        let mut last_price = HashMap::new();
        let mut cum_flow = HashMap::new();
        let max_jump = dec!(1);
        process_trade(
            msg.data,
            &txs,
            &mut last_price,
            &mut cum_flow,
            &max_jump,
            &ctx_sink(),
            0,
        );
        let p = rx.borrow_and_update().clone();
        assert_eq!(p.exchange_ts_ms, Some(1784652361123));
    }

    #[test]
    fn signed_flow_aggressive_sell_subtracts_buy_adds() {
        let (txs, _rx) = make_txs("btc");
        let mut last_price = HashMap::new();
        let mut cum_flow = HashMap::new();
        let max_jump = dec!(1); // no restriction for this test

        // Trade 1: aggressive buy (buyer_is_maker=false) — adds to net.
        let buy = BinanceTrade {
            price: "100".into(),
            symbol: "BTCUSDT".into(),
            qty: "2".into(),
            buyer_is_maker: false,
            trade_time: 0,
            trade_id: None,
        };
        process_trade(
            buy,
            &txs,
            &mut last_price,
            &mut cum_flow,
            &max_jump,
            &ctx_sink(),
            0,
        );
        let (net, total) = cum_flow["btc"];
        assert_eq!(net, dec!(200)); // +2*100
        assert_eq!(total, dec!(200));

        // Trade 2: aggressive sell (buyer_is_maker=true) — subtracts from net, adds to total.
        let sell = BinanceTrade {
            price: "100".into(),
            symbol: "BTCUSDT".into(),
            qty: "1".into(),
            buyer_is_maker: true,
            trade_time: 0,
            trade_id: None,
        };
        process_trade(
            sell,
            &txs,
            &mut last_price,
            &mut cum_flow,
            &max_jump,
            &ctx_sink(),
            0,
        );
        let (net, total) = cum_flow["btc"];
        assert_eq!(net, dec!(100)); // 200 - 100
        assert_eq!(total, dec!(300)); // 200 + 100
    }

    #[test]
    fn outlier_does_not_accumulate_volume() {
        let (txs, _rx) = make_txs("btc");
        let mut last_price = HashMap::new();
        let mut cum_flow = HashMap::new();
        let max_jump = dec!(0.05); // 5%

        let first = BinanceTrade {
            price: "100".into(),
            symbol: "BTCUSDT".into(),
            qty: "1".into(),
            buyer_is_maker: false,
            trade_time: 0,
            trade_id: None,
        };
        process_trade(
            first,
            &txs,
            &mut last_price,
            &mut cum_flow,
            &max_jump,
            &ctx_sink(),
            0,
        );
        let (net_before, total_before) = cum_flow["btc"];

        // Jump from 100 -> 200 (100%) exceeds max_jump: must be discarded without accumulating.
        let outlier = BinanceTrade {
            price: "200".into(),
            symbol: "BTCUSDT".into(),
            qty: "5".into(),
            buyer_is_maker: false,
            trade_time: 0,
            trade_id: None,
        };
        process_trade(
            outlier,
            &txs,
            &mut last_price,
            &mut cum_flow,
            &max_jump,
            &ctx_sink(),
            0,
        );
        let (net_after, total_after) = cum_flow["btc"];

        assert_eq!(net_after, net_before);
        assert_eq!(total_after, total_before);
    }

    // ── invalid-symbol exclusion ──────────────────────────────────────────────

    /// `build_url` is the pure piece of the fix — `run()` now passes it
    /// only the assets that `bootstrap_prices` confirmed valid (or ambiguous due to a
    /// network failure), never those Binance explicitly rejected (non-2xx HTTP
    /// on the ticker). This test pins the contract: an asset ABSENT from the input
    /// list simply does not appear in the combined stream URL — there is no
    /// way for an excluded symbol to sneak in and take down the whole stream.
    #[test]
    fn build_url_excludes_assets_not_in_filtered_list() {
        let all_names = vec!["btc".to_string(), "eth".to_string(), "nvda".to_string()];
        // Simulates the result of bootstrap_prices after dropping "nvda" (symbol
        // without a valid Binance counterpart).
        let filtered: Vec<String> = all_names.into_iter().filter(|a| a != "nvda").collect();

        let url = build_url(&filtered, "wss://stream.binance.com:9443");

        assert!(
            url.contains("btcusdt@trade"),
            "btc must be in the stream"
        );
        assert!(
            url.contains("ethusdt@trade"),
            "eth must be in the stream"
        );
        assert!(
            !url.contains("nvdausdt@trade"),
            "excluded nvda must not appear in the URL"
        );
    }

    /// With the filtered list empty (all assets rejected),
    /// `build_url` produces a URL with no streams — `run()` returns before reaching
    /// here (`if valid_assets.is_empty() { return; }`), but `build_url` itself
    /// must not panic on an empty list.
    #[test]
    fn build_url_with_empty_list_does_not_panic() {
        let url = build_url(&[], "wss://stream.binance.com:9443");
        assert!(url.ends_with("/stream?streams="));
    }
}
