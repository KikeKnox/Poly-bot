//! Binance USDS-M futures liquidation feed via WebSocket
//! (combined stream `<sym>@forceOrder`).
//!
//! Structural clone of `oracle::price` (exponential backoff, identical ping +
//! zombie check) — same reconnection pattern, different payload.
//!
//! **The `forceOrder` stream is a SNAPSHOT, not the complete flow**: Binance limits
//! emission to at most 1 liquidation order per second per symbol. It is a
//! SAMPLE of the real liquidation tape — enough to detect cascades
//! (concentrated bursts of liquidations), but it does NOT represent the total
//! liquidated volume if there is more than 1 event/s on the symbol.
//!
//! Publishes each asset's cumulative (net, total) on `watch::Sender<LiqCum>`.
//! Assets are derived the same way as in `oracle::price`: prefix of the slug pattern
//! (`btc-updown` → BTCUSDT).

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
use crate::types::LiqCum;

const MAX_BACKOFF_SECS: u64 = 30;

/// Builds the URL of the Binance Futures (USDS-M) `forceOrder` combined stream.
fn build_url(assets: &[String], base: &str) -> String {
    let streams: Vec<String> = assets
        .iter()
        .map(|a| format!("{}usdt@forceOrder", a.to_lowercase()))
        .collect();
    format!("{base}/stream?streams={}", streams.join("/"))
}

/// Creates the watch channels (sender, receiver) for each asset in `patterns`.
/// Copied from `oracle::price::create_price_channels`.
pub fn create_liq_channels(
    patterns: &[String],
) -> (
    HashMap<String, watch::Sender<LiqCum>>,
    HashMap<String, watch::Receiver<LiqCum>>,
) {
    let assets: HashSet<String> = patterns.iter().filter_map(|p| asset_from_slug(p)).collect();

    let mut txs = HashMap::with_capacity(assets.len());
    let mut rxs = HashMap::with_capacity(assets.len());
    for asset in assets {
        let (tx, rx) = watch::channel(LiqCum {
            asset: asset.clone(),
            ..LiqCum::default()
        });
        txs.insert(asset.clone(), tx);
        rxs.insert(asset, rx);
    }
    (txs, rxs)
}

// ── Deserialization types ─────────────────────────────────────────────────────

/// Forced liquidation order (`forceOrder`). Only the fields needed for the
/// signed notional are read: symbol, side, average price and filled quantity.
#[derive(Deserialize)]
struct ForceOrder {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "S")]
    side: String,
    #[serde(rename = "ap")]
    avg_price: String,
    #[serde(rename = "z")]
    filled_qty: String,
}

#[derive(Deserialize)]
struct ForceOrderEvent {
    #[serde(rename = "o")]
    order: ForceOrder,
}

/// Combined stream message: `{"stream":"btcusdt@forceOrder","data":{...}}`
#[derive(Deserialize)]
struct CombinedMsg {
    data: ForceOrderEvent,
}

// ── Main task ─────────────────────────────────────────────────────────────────

/// Spawns the liquidation feed and does not return (only on shutdown or fatal error).
/// Automatic reconnection with exponential backoff (cap 30 s) — EXACT pattern of
/// `oracle::price::run`.
pub async fn run(
    cfg: Arc<OracleConfig>,
    txs: HashMap<String, watch::Sender<LiqCum>>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    if txs.is_empty() {
        tracing::warn!("oracle::liquidations: no assets configured, exiting");
        return;
    }

    let assets: Vec<String> = txs.keys().cloned().collect();
    let url = build_url(&assets, &cfg.futures_ws_url);
    let stale = std::time::Duration::from_secs(cfg.stale_secs);
    let ping = std::time::Duration::from_secs(cfg.ping_interval_secs);
    let min_session = std::time::Duration::from_secs(cfg.min_session_secs_for_backoff_reset);

    let mut backoff = std::time::Duration::from_secs(1);
    // Cumulative (net, total) in USDT per asset. Lives in the OUTER loop
    // to survive reconnections — same anti-coalescing as the CVD
    // (cumulative volume delta): the counter is only reset with the process.
    let mut cum_liq: HashMap<String, (Decimal, Decimal)> = HashMap::new();

    loop {
        tracing::info!(url, "oracle::liquidations: connecting");
        let session_start = Instant::now();

        match run_session(&url, &txs, stale, ping, &mut cum_liq, &mut shutdown).await {
            Ok(()) => {
                tracing::info!("oracle::liquidations: clean shutdown");
                return;
            }
            Err(e) => {
                tracing::warn!(error = %e, backoff_secs = backoff.as_secs(), "oracle::liquidations: session ended");
                metrics::counter!("hrf_oracle_liq_stale_reconnect_total").increment(1);
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

async fn run_session(
    url: &str,
    txs: &HashMap<String, watch::Sender<LiqCum>>,
    stale: std::time::Duration,
    ping_interval: std::time::Duration,
    cum_liq: &mut HashMap<String, (Decimal, Decimal)>,
    shutdown: &mut tokio::sync::watch::Receiver<bool>,
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

            // CRITICAL — zombie check: `forceOrder` can legitimately go silent for
            // minutes (without a liquidation cascade there are no events). The
            // refresh of `last_msg` CANNOT depend only on receiving a real
            // liquidation event: in a quiet market that never arrives and the feed would
            // be flagged zombie even though the socket is alive. The Pong to our pings
            // (below) is the liveness signal that survives legitimate silence —
            // EXACT pattern of `oracle::price::run_session`, copied without "improvements".
            msg = ws.next() => {
                match msg {
                    None => anyhow::bail!("oracle: WS closed by the server"),
                    Some(Err(e)) => anyhow::bail!("oracle: WS error: {e}"),
                    Some(Ok(Message::Text(txt))) => {
                        last_msg = Instant::now();
                        handle_raw_message(&txt, txs, cum_liq);
                    }
                    Some(Ok(Message::Pong(_))) => { last_msg = Instant::now(); }
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

/// Handles a raw text message from the `forceOrder` stream.
///
/// Increments `hrf_binance_forceorder_msgs_total` for EVERY message received,
/// BEFORE any filtering/deserialization/aggregation — goal: distinguish
/// "calm market" (counter rises, `liq_total_60s=0` legitimate) from "feed silently
/// dead" (counter does not rise). The 12h run gave `liq_total_60s=0` in
/// 643/643 samples and there was no way to know which of the two scenarios it was.
fn handle_raw_message(
    txt: &str,
    txs: &HashMap<String, watch::Sender<LiqCum>>,
    cum_liq: &mut HashMap<String, (Decimal, Decimal)>,
) {
    metrics::counter!("hrf_binance_forceorder_msgs_total").increment(1);
    if let Ok(msg) = serde_json::from_str::<CombinedMsg>(txt) {
        process_force_order(msg.data.order, txs, cum_liq);
    }
}

fn process_force_order(
    order: ForceOrder,
    txs: &HashMap<String, watch::Sender<LiqCum>>,
    cum_liq: &mut HashMap<String, (Decimal, Decimal)>,
) {
    let asset = order.symbol.trim_end_matches("USDT").to_lowercase();
    let Ok(avg_price) = order.avg_price.parse::<Decimal>() else {
        return;
    };
    let Ok(filled_qty) = order.filled_qty.parse::<Decimal>() else {
        return;
    };
    if avg_price <= Decimal::ZERO || filled_qty <= Decimal::ZERO {
        return;
    }

    // Executed notional of the forced liquidation.
    let notional = filled_qty * avg_price;
    // S=="SELL" ⇒ a LONG was forced to sell ⇒ bearish pressure ⇒ negative.
    // S=="BUY"  ⇒ a SHORT was forced to buy ⇒ bullish pressure ⇒ positive.
    let signed = if order.side == "SELL" {
        -notional
    } else {
        notional
    };

    let entry = cum_liq
        .entry(asset.clone())
        .or_insert((Decimal::ZERO, Decimal::ZERO));
    entry.0 += signed;
    entry.1 += notional;
    let (cum_net_notional, cum_total_notional) = *entry;

    if let Some(tx) = txs.get(&asset) {
        let _ = tx.send(LiqCum {
            asset,
            cum_net_notional,
            cum_total_notional,
            updated_at: Utc::now(),
            instant: Some(Instant::now()),
        });
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_txs(
        asset: &str,
    ) -> (
        HashMap<String, watch::Sender<LiqCum>>,
        watch::Receiver<LiqCum>,
    ) {
        let (tx, rx) = watch::channel(LiqCum {
            asset: asset.to_string(),
            ..LiqCum::default()
        });
        let mut txs = HashMap::new();
        txs.insert(asset.to_string(), tx);
        (txs, rx)
    }

    #[test]
    fn raw_message_increments_counter_and_is_processed() {
        // Exercises the same path as `run_session` — raw text
        // from the stream enters via `handle_raw_message`, which increments the counter
        // `hrf_binance_forceorder_msgs_total` (with no test recorder installed, the
        // call is a safe no-op) and AT THE SAME TIME lets the message through to
        // `process_force_order` so the state updates as usual.
        let (txs, _rx) = make_txs("btc");
        let mut cum_liq = HashMap::new();
        let json = r#"{"stream":"btcusdt@forceOrder","data":{"e":"forceOrder","o":{"s":"BTCUSDT","S":"SELL","ap":"100","z":"2"}}}"#;

        handle_raw_message(json, &txs, &mut cum_liq);

        let (net, total) = cum_liq["btc"];
        assert_eq!(net, dec!(-200));
        assert_eq!(total, dec!(200));
    }

    #[test]
    fn invalid_raw_message_does_not_break_and_does_not_update_state() {
        // The counter must rise with ANY message received (even garbage),
        // but JSON that does not deserialize must not touch the aggregated state.
        let (txs, _rx) = make_txs("btc");
        let mut cum_liq = HashMap::new();

        handle_raw_message("this is not valid forceOrder JSON", &txs, &mut cum_liq);

        assert!(
            cum_liq.is_empty(),
            "an invalid message must not generate aggregated state"
        );
    }

    #[test]
    fn force_order_deserializes_combined_stream() {
        let json = r#"{"stream":"btcusdt@forceOrder","data":{"e":"forceOrder","o":{"s":"BTCUSDT","S":"SELL","ap":"50000.5","z":"0.014"}}}"#;
        let msg: CombinedMsg = serde_json::from_str(json).expect("must deserialize");
        assert_eq!(msg.data.order.symbol, "BTCUSDT");
        assert_eq!(msg.data.order.side, "SELL");
        assert_eq!(msg.data.order.avg_price, "50000.5");
        assert_eq!(msg.data.order.filled_qty, "0.014");
    }

    #[test]
    fn sell_is_long_liquidation_with_negative_sign() {
        let (txs, _rx) = make_txs("btc");
        let mut cum_liq = HashMap::new();

        // SELL = long liquidated → bearish pressure → negative notional.
        let sell = ForceOrder {
            symbol: "BTCUSDT".into(),
            side: "SELL".into(),
            avg_price: "100".into(),
            filled_qty: "2".into(),
        };
        process_force_order(sell, &txs, &mut cum_liq);
        let (net, total) = cum_liq["btc"];
        assert_eq!(net, dec!(-200));
        assert_eq!(total, dec!(200));

        // BUY = short liquidated → bullish pressure → positive notional.
        let buy = ForceOrder {
            symbol: "BTCUSDT".into(),
            side: "BUY".into(),
            avg_price: "100".into(),
            filled_qty: "1".into(),
        };
        process_force_order(buy, &txs, &mut cum_liq);
        let (net, total) = cum_liq["btc"];
        assert_eq!(net, dec!(-100)); // -200 + 100
        assert_eq!(total, dec!(300)); // 200 + 100
    }

    #[test]
    fn url_force_order_multi_asset() {
        let assets = vec!["btc".to_string(), "eth".to_string()];
        let url = build_url(&assets, "wss://fstream.binance.com");
        assert_eq!(
            url,
            "wss://fstream.binance.com/stream?streams=btcusdt@forceOrder/ethusdt@forceOrder"
        );
    }

    #[test]
    fn accumulated_liq_survives_coalescing() {
        // Two "consecutive" liquidations of the same asset must both accumulate —
        // the accumulator lives in the OUTER loop (same anti-coalescing as the CVD),
        // none is lost even if they arrive very close together.
        let (txs, _rx) = make_txs("btc");
        let mut cum_liq = HashMap::new();

        process_force_order(
            ForceOrder {
                symbol: "BTCUSDT".into(),
                side: "SELL".into(),
                avg_price: "100".into(),
                filled_qty: "1".into(),
            },
            &txs,
            &mut cum_liq,
        );
        process_force_order(
            ForceOrder {
                symbol: "BTCUSDT".into(),
                side: "SELL".into(),
                avg_price: "100".into(),
                filled_qty: "2".into(),
            },
            &txs,
            &mut cum_liq,
        );

        let (net, total) = cum_liq["btc"];
        assert_eq!(net, dec!(-300));
        assert_eq!(total, dec!(300));
    }
}
