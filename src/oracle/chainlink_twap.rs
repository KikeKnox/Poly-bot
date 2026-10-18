//! Chainlink 60 s TWAP source via PolyBolt (ARB-15M5M).
//!
//! The 5m/15m crypto up/downs resolve on Chainlink's 60 s TWAP,
//! not on the Binance spot. PolyBolt (`wss://ws-live-v2.polymarket.com/ws`,
//! channel `price.crypto.twap`) publishes that same TWAP; `arb_15m5m` uses it to
//! know the EXACT strike (`ts = t0` of the market) of each leg.
//!
//! Protocol (copied from the recorder, `tools/grabador/polymarket.py`):
//! 1. connect and send `{"op":"auth","rid":"a1","auth":{apiKey,secret,passphrase}}`
//!    with the CLOB L2 credentials (`POLY_API_KEY/SECRET/PASSPHRASE`);
//! 2. on receiving `{"op":"authed"}`, send the subscription with a per-symbol
//!    filter (`btcusd`…) and `window_seconds = 60`;
//! 3. `{"op":"ping"}` every 20 s.
//!
//! Credentials are never printed or logged. It does not touch `reference_price`
//! nor any other strategy: it only feeds the [`TwapStore`].

use std::{
    collections::{BTreeMap, HashMap},
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};

use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex;
use rust_decimal::Decimal;
use serde_json::{json, Value};
use tokio::sync::watch;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const URL_POLYBOLT: &str = "wss://ws-live-v2.polymarket.com/ws";
const TWAP_CHANNEL: &str = "price.crypto.twap";
const WINDOW_S: i64 = 60;
/// Per-asset tick retention: more than enough for the 10 min separating the 15m
/// `t0` from the `t0` of its paired 5m, with margin for reconnections.
const RETENTION_MS: i64 = 20 * 60 * 1000;
const PING_EVERY: Duration = Duration::from_secs(20);
/// A feed with a tick every second that goes silent for a minute is dead.
const NO_MESSAGES_MAX: Duration = Duration::from_secs(60);
const MIN_SESSION_TO_RESET_BACKOFF: Duration = Duration::from_secs(60);
const BACKOFF_MAX: Duration = Duration::from_secs(30);

// ── Shared store ──────────────────────────────────────────────────────────────

#[derive(Default)]
struct Series {
    /// ts (tick epoch ms) → TWAP value.
    ticks: BTreeMap<i64, Decimal>,
    last_received: Option<Instant>,
}

/// Recent ticks of the 60 s TWAP per asset. Written by the feed task and
/// read by the strategies; the lock is short (insertion or point read).
#[derive(Default)]
pub struct TwapStore {
    series: Mutex<HashMap<String, Series>>,
}

impl TwapStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn insert(&self, asset: &str, ts_ms: i64, value: Decimal) {
        let mut series = self.series.lock();
        let ticks_series = series.entry(asset.to_owned()).or_default();
        ticks_series.ticks.insert(ts_ms, value);
        ticks_series.last_received = Some(Instant::now());
        // Retention is measured against the newest tick, not the wall
        // clock: so a late snapshot after reconnecting does not evict what is useful.
        if let Some(&last_key) = ticks_series.ticks.keys().next_back() {
            let cut = last_key - RETENTION_MS;
            ticks_series.ticks = ticks_series.ticks.split_off(&cut);
        }
    }

    /// Value of the tick with EXACT `ts_ms`. Without an exact tick there is no strike: an
    /// approximation could flip the order of the strikes and the basket pay 0.
    pub fn exact_value(&self, asset: &str, ts_ms: i64) -> Option<Decimal> {
        self.series.lock().get(asset)?.ticks.get(&ts_ms).copied()
    }

    /// Time since the asset's last tick arrived; `None` if none ever did.
    pub fn last_tick_age(&self, asset: &str) -> Option<Duration> {
        self.series
            .lock()
            .get(asset)?
            .last_received
            .map(|t| t.elapsed())
    }

    fn assets(&self) -> Vec<String> {
        self.series.lock().keys().cloned().collect()
    }
}

// ── Credentials ──────────────────────────────────────────────────────────────

/// CLOB L2 credentials. `Debug` is hand-written so they do not leak.
#[derive(Clone)]
pub struct Credentials {
    api_key: String,
    secret: String,
    passphrase: String,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Credentials(***)")
    }
}

impl Credentials {
    /// The same variables read by `clob_sdk::bootstrap`. `None` if any is missing.
    pub fn from_env() -> Option<Self> {
        let read_env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Some(Self {
            api_key: read_env("POLY_API_KEY")?,
            secret: read_env("POLY_API_SECRET")?,
            passphrase: read_env("POLY_API_PASSPHRASE")?,
        })
    }

    fn auth_message(&self) -> String {
        json!({"op": "auth", "rid": "a1", "auth": {
            "apiKey": self.api_key, "secret": self.secret, "passphrase": self.passphrase,
        }})
        .to_string()
    }
}

// ── Parsing ───────────────────────────────────────────────────────────────────

/// An already validated TWAP tick (60 s window, symbol `<asset>usd`).
#[derive(Debug, PartialEq)]
pub struct TickTwap {
    pub asset: String,
    pub ts_ms: i64,
    pub value: Decimal,
}

fn decimal_from_json(v: &Value) -> Option<Decimal> {
    let Value::Number(n) = v else { return None };
    let text = n.to_string();
    Decimal::from_str(&text)
        .or_else(|_| Decimal::from_scientific(&text))
        .ok()
}

/// Extracts the 60 s TWAP ticks from a WS text message. The normal message
/// carries the point in `payload` (`timestamp` ms, `symbol`, `value`,
/// `window_seconds`); on reconnect the server sends a snapshot with
/// `payload.data = [points]`. Anything else returns empty.
pub fn parse_ticks(txt: &str) -> Vec<TickTwap> {
    let Ok(msg) = serde_json::from_str::<Value>(txt) else {
        return vec![];
    };
    if msg.get("channel").and_then(Value::as_str) != Some(TWAP_CHANNEL) {
        return vec![];
    }
    let Some(payload) = msg.get("payload") else {
        return vec![];
    };
    let points: Vec<&Value> = match payload.get("data").and_then(Value::as_array) {
        Some(items) => items.iter().collect(),
        None => vec![payload],
    };
    points
        .into_iter()
        .filter_map(|p| {
            // The point's fields take precedence; if missing, the payload's apply.
            let field = |k: &str| p.get(k).or_else(|| payload.get(k));
            if field("window_seconds")?.as_i64()? != WINDOW_S {
                return None;
            }
            let symbol = field("symbol")?.as_str()?.to_lowercase();
            Some(TickTwap {
                asset: symbol.strip_suffix("usd")?.to_owned(),
                ts_ms: field("timestamp")?.as_i64()?,
                value: decimal_from_json(field("value")?)?,
            })
        })
        .collect()
}

fn subscription_message(assets: &[String]) -> String {
    let subs: Vec<Value> = assets
        .iter()
        .map(|a| {
            json!({"channel": TWAP_CHANNEL,
                   "filter": {"symbol": format!("{a}usd"), "window_seconds": WINDOW_S}})
        })
        .collect();
    json!({"op": "subscribe", "rid": "s1", "subscriptions": subs}).to_string()
}

// ── Task ──────────────────────────────────────────────────────────────────────

/// Keeps the [`TwapStore`] up to date. Reconnects with exponential backoff (cap
/// 30 s) and only returns on shutdown. The assets are those of the universe
/// (`btc`, `eth`…); the subscribed symbol is `<asset>usd`.
pub async fn run(
    store: Arc<TwapStore>,
    creds: Credentials,
    assets: Vec<String>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut backoff = Duration::from_secs(1);
    loop {
        tracing::info!(assets = assets.len(), "oracle::chainlink_twap: connecting to PolyBolt");
        let start = Instant::now();
        match session(&store, &creds, &assets, &mut shutdown).await {
            Ok(()) => {
                tracing::info!("oracle::chainlink_twap: clean shutdown");
                return;
            }
            Err(e) => {
                tracing::warn!(error = %e, backoff_secs = backoff.as_secs(),
                    "oracle::chainlink_twap: session ended");
                metrics::counter!("hrf_twap_reconnects_total").increment(1);
            }
        }
        if start.elapsed() >= MIN_SESSION_TO_RESET_BACKOFF {
            backoff = Duration::from_secs(1);
        }
        tokio::select! {
            _ = tokio::time::sleep(backoff) => {}
            _ = shutdown.changed() => {
                if *shutdown.borrow() { return; }
            }
        }
        backoff = (backoff * 2).min(BACKOFF_MAX);
    }
}

async fn session(
    store: &TwapStore,
    creds: &Credentials,
    assets: &[String],
    shutdown: &mut watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let (mut ws, _) = connect_async(URL_POLYBOLT).await?;
    ws.send(Message::Text(creds.auth_message())).await?;

    let mut last_message = Instant::now();
    let mut ping = tokio::time::interval(PING_EVERY);
    let mut age = tokio::time::interval(Duration::from_secs(5));

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() { return Ok(()); }
            }
            _ = ping.tick() => {
                if last_message.elapsed() > NO_MESSAGES_MAX {
                    anyhow::bail!("zombie feed (no messages for {}s)", NO_MESSAGES_MAX.as_secs());
                }
                ws.send(Message::Text(r#"{"op":"ping"}"#.to_owned())).await?;
            }
            _ = age.tick() => publish_ages(store),
            msg = ws.next() => match msg {
                None => anyhow::bail!("WS closed by the server"),
                Some(Err(e)) => anyhow::bail!("WS error: {e}"),
                Some(Ok(Message::Text(txt))) => {
                    last_message = Instant::now();
                    if !txt.contains(r#""channel""#) && txt.len() < 300 {
                        responder_op(&mut ws, &txt, assets).await?;
                        continue;
                    }
                    for t in parse_ticks(&txt) {
                        metrics::counter!("hrf_twap_ticks_total", "asset" => t.asset.clone())
                            .increment(1);
                        store.insert(&t.asset, t.ts_ms, t.value);
                    }
                }
                Some(Ok(Message::Pong(_))) => last_message = Instant::now(),
                Some(Ok(_)) => {}
            },
        }
    }
}

/// Control messages: after `authed` we must subscribe. A server `error`
/// (credentials rejected) ends the session without dumping its body.
async fn responder_op<S>(ws: &mut S, txt: &str, assets: &[String]) -> anyhow::Result<()>
where
    S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
{
    let op = serde_json::from_str::<Value>(txt)
        .ok()
        .and_then(|v| v.get("op").and_then(Value::as_str).map(str::to_owned));
    match op.as_deref() {
        Some("authed") => ws.send(Message::Text(subscription_message(assets))).await?,
        Some("error") => anyhow::bail!("PolyBolt returned op=error (credentials?)"),
        _ => {}
    }
    Ok(())
}

fn publish_ages(store: &TwapStore) {
    for asset in store.assets() {
        if let Some(age) = store.last_tick_age(&asset) {
            metrics::gauge!("hrf_twap_last_tick_age_seconds", "asset" => asset)
                .set(age.as_secs_f64());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn parses_single_tick() {
        let txt = r#"{"channel":"price.crypto.twap","seq":7,"payload":{"timestamp":1783368300000,
            "symbol":"btcusd","value":67123.45,"window_seconds":60,"source":"chainlink"}}"#;
        assert_eq!(
            parse_ticks(txt),
            vec![TickTwap { asset: "btc".into(), ts_ms: 1_783_368_300_000, value: dec!(67123.45) }]
        );
    }

    #[test]
    fn parses_snapshot_with_point_list() {
        let txt = r#"{"channel":"price.crypto.twap","payload":{"symbol":"ethusd","window_seconds":60,
            "data":[{"timestamp":1000,"value":2500.5},{"timestamp":2000,"value":2501}]}}"#;
        let ticks = parse_ticks(txt);
        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[1], TickTwap { asset: "eth".into(), ts_ms: 2000, value: dec!(2501) });
    }

    #[test]
    fn discards_other_windows_channels_and_junk() {
        let window30 = r#"{"channel":"price.crypto.twap","payload":{"timestamp":1,"symbol":"btcusd","value":1.0,"window_seconds":30}}"#;
        let spot = r#"{"channel":"price.crypto","payload":{"timestamp":1,"symbol":"btcusd","value":1.0}}"#;
        let without_usd = r#"{"channel":"price.crypto.twap","payload":{"timestamp":1,"symbol":"btceur","value":1.0,"window_seconds":60}}"#;
        for txt in [window30, spot, without_usd, r#"{"op":"authed"}"#, "not json"] {
            assert!(parse_ticks(txt).is_empty(), "should have been discarded: {txt}");
        }
    }

    #[test]
    fn store_returns_only_the_exact_value() {
        let store = TwapStore::new();
        store.insert("btc", 1_000, dec!(100));
        store.insert("btc", 2_000, dec!(101));
        assert_eq!(store.exact_value("btc", 1_000), Some(dec!(100)));
        assert_eq!(store.exact_value("btc", 1_500), None, "neighbor is not approximated");
        assert_eq!(store.exact_value("eth", 1_000), None);
    }

    #[test]
    fn store_discards_ticks_older_than_retention() {
        let store = TwapStore::new();
        store.insert("btc", 0, dec!(1));
        store.insert("btc", RETENTION_MS + 1_000, dec!(2));
        assert_eq!(store.exact_value("btc", 0), None);
        assert_eq!(store.exact_value("btc", RETENTION_MS + 1_000), Some(dec!(2)));
    }

    #[test]
    fn age_only_exists_after_first_tick() {
        let store = TwapStore::new();
        assert!(store.last_tick_age("btc").is_none());
        store.insert("btc", 1_000, dec!(1));
        assert!(store.last_tick_age("btc").unwrap() < Duration::from_secs(1));
    }

    #[test]
    fn credentials_do_not_leak_via_debug() {
        let c = Credentials { api_key: "k-secret".into(), secret: "s-secret".into(), passphrase: "p-secret".into() };
        let dbg = format!("{c:?}");
        assert!(!dbg.contains("secret") && !dbg.contains("k-s") && !dbg.contains("p-s"));
        assert!(c.auth_message().contains(r#""apiKey":"k-secret""#), "the auth message does carry them");
    }

    #[test]
    fn subscription_filters_by_symbol_and_window() {
        let msg: Value = serde_json::from_str(&subscription_message(&["btc".into()])).unwrap();
        let sub = &msg["subscriptions"][0];
        assert_eq!(sub["channel"], TWAP_CHANNEL);
        assert_eq!(sub["filter"]["symbol"], "btcusd");
        assert_eq!(sub["filter"]["window_seconds"], 60);
    }
}
