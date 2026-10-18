//! Shared vol oracle.
//!
//! One task per asset samples `watch<AssetPrice>` and publishes
//! `watch<VolSnapshot>` with σ̂ at two windows (short = 60s,
//! long = 300s), computed by realized variance over a buffer
//! subsampled to 1s (`realized_vol_per_sec`).
//!
//! Strategies consume the watch instead of keeping their own ring buffer
//! → consistent σ across strategies, a single calculation point.

use crate::time::Instant;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use rust_decimal::Decimal;
use tokio::sync::watch;

use crate::fair_value::{realized_vol_per_sec, PriceSample};
use crate::types::{AssetPrice, LiqCum, VolSnapshot};

/// Creates the VolSnapshot watch channels for each asset.
pub fn create_vol_channels(
    assets: &[String],
) -> (
    HashMap<String, watch::Sender<VolSnapshot>>,
    HashMap<String, watch::Receiver<VolSnapshot>>,
) {
    let mut txs = HashMap::with_capacity(assets.len());
    let mut rxs = HashMap::with_capacity(assets.len());
    for asset in assets {
        let snap = VolSnapshot {
            asset: asset.clone(),
            ..VolSnapshot::default()
        };
        let (tx, rx) = watch::channel(snap);
        txs.insert(asset.clone(), tx);
        rxs.insert(asset.clone(), rx);
    }
    (txs, rxs)
}

/// Launches one vol oracle task per asset.
///
/// `price_rxs`: receives `AssetPrice` from `oracle::price` for each asset.
/// `vol_txs`:   publishes `VolSnapshot` to the strategies.
/// `liq_rxs`: receives `LiqCum` (cumulative futures liquidations) for the
/// assets with a configured feed — empty map if `oracle.futures_liq_enabled=false`,
/// inert behavior in that scenario.
/// Each asset has its own sample buffer; there is no shared state.
pub fn spawn_all(
    price_rxs: HashMap<String, watch::Receiver<AssetPrice>>,
    vol_txs: HashMap<String, watch::Sender<VolSnapshot>>,
    liq_rxs: HashMap<String, watch::Receiver<LiqCum>>,
    shutdown: tokio::sync::watch::Receiver<bool>,
) {
    for (asset, price_rx) in price_rxs {
        if let Some(vol_tx) = vol_txs.get(&asset).cloned() {
            let shutdown_rx = shutdown.clone();
            let liq_rx = liq_rxs.get(&asset).cloned();
            // Supervise the per-asset task. Previously a panic here left that
            // asset's vol silently frozen → stale snapshot → all the
            // asset's strategies stopped trading without any alert.
            tokio::spawn(async move {
                let mut backoff_secs = 1u64;
                loop {
                    let handle = tokio::spawn(run_asset(
                        asset.clone(),
                        price_rx.clone(),
                        vol_tx.clone(),
                        liq_rx.clone(),
                        shutdown_rx.clone(),
                    ));
                    match handle.await {
                        Ok(()) => break, // clean exit (shutdown / channel closed)
                        Err(e) => {
                            tracing::error!(asset = %asset, error = %e,
                                "oracle::vol: task died — restarting with backoff");
                            metrics::counter!("hrf_vol_task_restarts_total", "asset" => asset.clone())
                                .increment(1);
                            tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                            backoff_secs = (backoff_secs * 2).min(30);
                        }
                    }
                }
            });
        }
    }
    // We consume the shutdown so as not to accumulate unused receivers
    drop(shutdown);
}

// ── Per-asset task ───────────────────────────────────────────────────────────

/// Buffer of raw ticks (used by CVD/last_price, NOT by σ̂ since the subsampled buffer):
/// 5 min of samples at ~1 trade/s = 300 samples. Cap BY COUNT — this
/// covers only ~3-12s real on liquid markets (BTC/ETH),
/// which is exactly why σ̂ is no longer computed over this buffer.
const MAX_SAMPLES: usize = 120;
/// Publish a new snapshot at least every this interval even if there are no trades.
const MAX_PUBLISH_INTERVAL: Duration = Duration::from_secs(5);

/// Parameters of `realized_vol_per_sec` over the buffer
/// subsampled to 1s (`sec_samples`), NOT over the tick buffer (`samples`).
const SIGMA_GRID_SECS: i64 = 1;
const SIGMA_WINDOW_SHORT: i64 = 60; // replaces sigma_5s (ALPHA_5S)
const SIGMA_WINDOW_LONG: i64 = 300; // replaces sigma_60s (ALPHA_60S)
const SIGMA_MIN_PAIRS: usize = 30;
const SIGMA_RETENTION_SECS: i64 = 360; // long window + 20% margin
const SIGMA_MAX_SEC_SAMPLES: usize = 1_000;
/// Discard the sample if the exchange timestamp deviates from the local clock
/// by more than this (unsynchronized clock / stale message after reconnection).
const MAX_TS_DEVIATION_MS: i64 = 60_000;
/// Throttle of the per-asset `vol_snapshot` log.
const VOL_LOG_THROTTLE: Duration = Duration::from_secs(30);

/// Windows of the CVD (Cumulative Volume Delta), aligned with σ5s/σ60s.
const CVD_WINDOW_SHORT_SECS: i64 = 5;
const CVD_WINDOW_LONG_SECS: i64 = 60;
/// Retention of the CVD buffer — 120s covers 2× the long window with margin.
const CVD_RETENTION_SECS: i64 = 120;
/// Hard cap on entries (a very liquid market must not grow without bound).
const CVD_MAX_SAMPLES: usize = 4096;

/// Single liquidation window in v1 (no short variant like the CVD).
const LIQ_WINDOW_SECS: i64 = 60;

async fn run_asset(
    asset: String,
    mut price_rx: watch::Receiver<AssetPrice>,
    vol_tx: watch::Sender<VolSnapshot>,
    mut liq_rx: Option<watch::Receiver<LiqCum>>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut samples: std::collections::VecDeque<PriceSample> = std::collections::VecDeque::new();
    // Buffer subsampled to 1s. The tick buffer
    // (`samples`) covers ~3s on BTC because its cap is BY COUNT (MAX_SAMPLES=120),
    // not temporal: that was the mechanism of the σ̂ bug. This one keeps one sample
    // per second with TEMPORAL retention, so 360 entries are always 360
    // seconds.
    let mut sec_samples: VecDeque<PriceSample> = VecDeque::new();
    let mut last_publish = Instant::now();
    // Throttle of the vol_snapshot log — starts "expired" so the
    // first publish logs immediately.
    let mut last_vol_log = Instant::now() - VOL_LOG_THROTTLE;
    // Instant of the last real Binance price — used as the snapshot's updated_at.
    // updated_at is not refreshed when we re-publish without new data, so the
    // strategies can detect oracle staleness (updated_at.elapsed() > threshold).
    let mut last_price_ts = Instant::now();
    // Buffer of (ts_ms, cum_net, cum_total) to derive the CVD per window.
    let mut cvd_buf: VecDeque<(i64, Decimal, Decimal)> = VecDeque::new();
    // Last known cumulative — for the deadline's synthetic sample (quiet market).
    let mut last_cum: (Decimal, Decimal) = (Decimal::ZERO, Decimal::ZERO);
    // Buffer of (ts_ms, cum_net_notional, cum_total_notional) of
    // liquidations — same anti-coalescing and same window function
    // (`cvd_window`/`push_cvd_sample` are generic over the kind of cumulative,
    // not CVD-specific) as the CVD.
    let mut liq_buf: VecDeque<(i64, Decimal, Decimal)> = VecDeque::new();
    // Last known liquidation cumulative — synthetic deadline sample.
    let mut last_liq_cum: (Decimal, Decimal) = (Decimal::ZERO, Decimal::ZERO);

    loop {
        let deadline = tokio::time::sleep(MAX_PUBLISH_INTERVAL);
        tokio::pin!(deadline);

        tokio::select! {
            _ = &mut deadline => {
                let now_ms = chrono::Utc::now().timestamp_millis();
                // Synthetic sample with the last cum — without this, in a quiet market
                // the buffer never covers the window and cvd_* stays None forever.
                push_cvd_sample(&mut cvd_buf, now_ms, last_cum.0, last_cum.1);
                // Same synthetic sample for liquidations — without rare forceOrder
                // events, the window would never be covered.
                push_cvd_sample(&mut liq_buf, now_ms, last_liq_cum.0, last_liq_cum.1);
                // Re-publish with the same last_price_ts so as not to hide staleness.
                publish(&asset, &samples, &sec_samples, &cvd_buf, &liq_buf, &vol_tx, last_price_ts, &mut last_vol_log);
                last_publish = Instant::now();
            }

            result = price_rx.changed() => {
                if result.is_err() { return; } // channel closed → shutdown
                let p = price_rx.borrow();
                if p.price <= Decimal::ZERO { continue; }

                // Timestamp of the TRADE on the exchange, not of
                // local receipt. Guard: if it deviates > MAX_TS_DEVIATION_MS from the
                // local clock (unsynchronized clock / stale message after
                // reconnection), discard the whole sample — without this, a
                // skew puts samples in past cells and breaks the
                // monotonicity that `realized_vol_per_sec` assumes.
                let now_ms = chrono::Utc::now().timestamp_millis();
                let sample_ts_ms = match p.exchange_ts_ms {
                    Some(ts) if (ts - now_ms).abs() > MAX_TS_DEVIATION_MS => {
                        metrics::counter!("hrf_oracle_ts_deviated_total", "asset" => asset.clone())
                            .increment(1);
                        continue;
                    }
                    Some(ts) => ts,
                    None => now_ms,
                };
                let sample = PriceSample {
                    ts_ms: sample_ts_ms,
                    price: p.price,
                };
                let cum_net   = p.cum_flow_quote;
                let cum_total = p.cum_vol_quote;
                drop(p);

                last_price_ts = Instant::now();
                samples.push_back(sample);
                if samples.len() > MAX_SAMPLES {
                    samples.pop_front();
                }

                // Buffer subsampled to 1s — last sample
                // of the cell wins, TEMPORAL retention + hard cap.
                push_sec_sample(&mut sec_samples, sample, SIGMA_GRID_SECS, SIGMA_RETENTION_SECS, SIGMA_MAX_SEC_SAMPLES);

                // Push on every real trade, without the 200ms throttle below —
                // otherwise resolution of the cumulative is lost between publications.
                push_cvd_sample(&mut cvd_buf, sample.ts_ms, cum_net, cum_total);
                last_cum = (cum_net, cum_total);

                // The deadline is recreated on every iteration of the select,
                // so on a liquid market (several Binance trades/s) its branch never
                // runs and liq_buf never receives the synthetic sample that keeps it
                // covered — cvd_window would return None or an already expired window.
                // We push the same synthetic sample here (repeats the last known
                // cumulative, monotonic → does not corrupt the delta) every time a
                // fresh price tick does arrive, fulfilling the invariant "every buffer receives
                // fresh samples whichever arm fires".
                push_cvd_sample(&mut liq_buf, sample.ts_ms, last_liq_cum.0, last_liq_cum.1);

                // Only publish if at least 200 ms have passed; avoids spam during trade bursts.
                if last_publish.elapsed() >= Duration::from_millis(200) {
                    publish(&asset, &samples, &sec_samples, &cvd_buf, &liq_buf, &vol_tx, last_price_ts, &mut last_vol_log);
                    last_publish = Instant::now();
                }
            }

            // No liquidation feed for the asset → inert arm
            // (pending). We do not re-publish here: the next Binance tick /
            // deadline carries the updated liq_buf.
            changed = async { match liq_rx.as_mut() { Some(rx) => rx.changed().await, None => std::future::pending().await } } => {
                match changed {
                    Ok(()) => {
                        if let Some(rx) = liq_rx.as_ref() {
                            let l = rx.borrow();
                            let ts_ms = chrono::Utc::now().timestamp_millis();
                            push_cvd_sample(&mut liq_buf, ts_ms, l.cum_net_notional, l.cum_total_notional);
                            last_liq_cum = (l.cum_net_notional, l.cum_total_notional);
                            // Same invariant — if liquidations
                            // (forceOrder) fire often while the spot is quiet,
                            // cvd_buf goes stale without this synthetic push.
                            push_cvd_sample(&mut cvd_buf, ts_ms, last_cum.0, last_cum.1);
                        }
                    }
                    Err(_) => {
                        // The liquidation feed's sender dropped
                        // (`oracle_liquidations` task dead — main does not
                        // respawn it). `changed()` on a closed channel
                        // resolves Err INSTANTLY on every turn of the select:
                        // without this guard the loop would burn a CPU per asset
                        // until shutdown. The arm is left inert (pending)
                        // and the rest of the snapshot keeps being published.
                        tracing::warn!(asset = %asset,
                            "oracle::vol: liquidations channel closed — arm inert");
                        liq_rx = None;
                    }
                }
            }

            _ = shutdown.changed() => {
                if *shutdown.borrow() { return; }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn publish(
    asset: &str,
    samples: &std::collections::VecDeque<PriceSample>,
    sec_samples: &VecDeque<PriceSample>,
    cvd_buf: &VecDeque<(i64, Decimal, Decimal)>,
    liq_buf: &VecDeque<(i64, Decimal, Decimal)>,
    vol_tx: &watch::Sender<VolSnapshot>,
    last_price_ts: Instant,
    last_vol_log: &mut Instant,
) {
    let s: Vec<PriceSample> = samples.iter().copied().collect();
    let sec: Vec<PriceSample> = sec_samples.iter().copied().collect();
    let last_price = s.last().map(|x| x.price).unwrap_or(Decimal::ZERO);

    // σ̂ by realized variance over the buffer subsampled
    // to 1s (`sec`), NOT over the raw tick buffer (`s`).
    let sigma_short =
        realized_vol_per_sec(&sec, SIGMA_GRID_SECS, SIGMA_WINDOW_SHORT, SIGMA_MIN_PAIRS);
    let sigma_long =
        realized_vol_per_sec(&sec, SIGMA_GRID_SECS, SIGMA_WINDOW_LONG, SIGMA_MIN_PAIRS);

    // σ̂ not estimable — today this scenario is invisible and the strategies
    // `return` silently. It changes no trading decision.
    if sigma_short.is_none() {
        metrics::counter!("hrf_vol_sigma_none_total", "asset" => asset.to_string(), "window" => "short").increment(1);
    }
    if sigma_long.is_none() {
        metrics::counter!("hrf_vol_sigma_none_total", "asset" => asset.to_string(), "window" => "long").increment(1);
    }

    // cvd_5s/cvd_60s = delta of the signed net flow in each window;
    // vol_quote_60s = gross volume (both directions) in the long window —
    // serves as the denominator to normalize the ratio in PM (parity_maker::pm_entry).
    let cvd_5s_window = cvd_window(cvd_buf, CVD_WINDOW_SHORT_SECS);
    let cvd_60s_window = cvd_window(cvd_buf, CVD_WINDOW_LONG_SECS);
    let cvd_5s = cvd_5s_window.map(|(net, _total)| net);
    let cvd_60s = cvd_60s_window.map(|(net, _total)| net);
    let vol_quote_60s = cvd_60s_window.map(|(_net, total)| total);

    // liq_net_60s/liq_total_60s = delta of the liquidation cumulative over
    // the 60s window. Reuses `cvd_window` (generic over the cumulative, not
    // CVD-specific) — `None` if the history does not yet cover the window.
    let liq_window = cvd_window(liq_buf, LIQ_WINDOW_SECS);
    let liq_net_60s = liq_window.map(|(net, _total)| net);
    let liq_total_60s = liq_window.map(|(_net, total)| total);

    // Turbulence index (regime signal, `oracle::turbulence`): same
    // `sec` buffer and same windows as σ̂ — return horizon = short
    // window (60s), μ/σ² history = long window (300s), same minimum of
    // pairs. `None` on warmup/hole at the anchor — an absent signal never blocks.
    let turbulence = crate::oracle::turbulence::turbulence_index(
        &sec,
        SIGMA_GRID_SECS,
        SIGMA_WINDOW_SHORT,
        SIGMA_WINDOW_LONG,
        SIGMA_MIN_PAIRS,
    );
    if let Some(t) = turbulence {
        metrics::gauge!("hrf_oracle_turbulence", "asset" => asset.to_string()).set(t);
    }

    // Log throttled to 1/30s per asset. n_cells/span_secs are the
    // direct detector of the failure mode of the 1s subsampling: if span_secs drifts from
    // window_secs, the buffer does not cover the window.
    if last_vol_log.elapsed() >= VOL_LOG_THROTTLE {
        let span_secs = match (sec.first(), sec.last()) {
            (Some(f), Some(l)) => (l.ts_ms - f.ts_ms) / 1000,
            _ => 0,
        };
        tracing::info!(target: "vol_snapshot",
            ts_ms = chrono::Utc::now().timestamp_millis(),
            asset = %asset,
            sigma_short = %opt_dec(sigma_short), sigma_long = %opt_dec(sigma_long),
            turbulence = %opt_f64(turbulence),
            n_cells = sec.len(), span_secs = span_secs, last_price = %last_price,
            ticks_in_buffer = s.len(),
            "VOL: snapshot");
        *last_vol_log = Instant::now();
    }

    let snap = VolSnapshot {
        asset: asset.to_string(),
        sigma_short,
        sigma_long,
        last_price,
        updated_at: last_price_ts,
        cvd_5s,
        cvd_60s,
        vol_quote_60s,
        liq_net_60s,
        liq_total_60s,
        turbulence,
    };
    let _ = vol_tx.send(snap);
}

/// Pushes `sample` to the buffer subsampled to `grid_secs`,
/// replacing within the same cell (last sample wins) and trimming
/// by temporal retention + hard cap — same pattern as `push_cvd_sample`.
fn push_sec_sample(
    buf: &mut VecDeque<PriceSample>,
    sample: PriceSample,
    grid_secs: i64,
    retention_secs: i64,
    max_len: usize,
) {
    let grid_ms = grid_secs * 1000;
    let same_cell = buf
        .back()
        .is_some_and(|b| b.ts_ms.div_euclid(grid_ms) == sample.ts_ms.div_euclid(grid_ms));
    if same_cell {
        if let Some(back) = buf.back_mut() {
            *back = sample;
        }
    } else {
        buf.push_back(sample);
    }
    while let Some(front) = buf.front() {
        if sample.ts_ms - front.ts_ms > retention_secs * 1000 || buf.len() > max_len {
            buf.pop_front();
        } else {
            break;
        }
    }
}

/// Empty string if there is no value —
/// distinguishes "no data" from "zero" in the log.
fn opt_dec(d: Option<Decimal>) -> String {
    d.map_or(String::new(), |v| v.to_string())
}

/// Same as `opt_dec` for turbulence (dimensionless `f64`): the offline harness
/// `p5_sigma_check` parses the `vol_snapshot` line by generic `key=value`,
/// so an extra field is compatible without touching the parser.
fn opt_f64(v: Option<f64>) -> String {
    v.map_or(String::new(), |x| x.to_string())
}

// ── CVD ───────────────────────────────────────────────────────────────────────

/// Pushes a sample (ts_ms, cum_net, cum_total) to the buffer and trims by
/// retention (120s) and hard cap (`CVD_MAX_SAMPLES`), whichever applies first.
fn push_cvd_sample(
    buf: &mut VecDeque<(i64, Decimal, Decimal)>,
    ts_ms: i64,
    cum_net: Decimal,
    cum_total: Decimal,
) {
    buf.push_back((ts_ms, cum_net, cum_total));
    while let Some(&(oldest_ts, _, _)) = buf.front() {
        if ts_ms - oldest_ts > CVD_RETENTION_SECS * 1000 || buf.len() > CVD_MAX_SAMPLES {
            buf.pop_front();
        } else {
            break;
        }
    }
}

/// Delta of the (net, total) cumulative over a window of `window_secs`.
/// Searches **from the end** (`iter().rev().find`) for the most recent sample with
/// age ≥ window — NOT the `spot_hist` pattern, which takes the oldest one that
/// qualifies and so inflates the real window up to 2×. `None` if not even the oldest sample
/// of the buffer covers the window (insufficient history: startup or reconnection).
fn cvd_window(
    buf: &VecDeque<(i64, Decimal, Decimal)>,
    window_secs: i64,
) -> Option<(Decimal, Decimal)> {
    let &(now_ts, now_net, now_total) = buf.back()?;
    let window_ms = window_secs * 1000;
    let &(_, anchor_net, anchor_total) = buf
        .iter()
        .rev()
        .find(|&&(ts, _, _)| now_ts - ts >= window_ms)?;
    Some((now_net - anchor_net, now_total - anchor_total))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod cvd_tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn cvd_window_is_delta_of_accumulated() {
        let mut buf = VecDeque::new();
        buf.push_back((0i64, dec!(0), dec!(0)));
        buf.push_back((1000, dec!(10), dec!(10)));
        buf.push_back((2000, dec!(25), dec!(35))); // "now"

        let (net, total) = cvd_window(&buf, 1).expect("1s window must be covered");
        // anchor = sample at t=1000 (exact age = 1000ms >= 1s)
        assert_eq!(net, dec!(15)); // 25 - 10
        assert_eq!(total, dec!(25)); // 35 - 10
    }

    #[test]
    fn cvd_none_without_history_covering_window() {
        let mut buf = VecDeque::new();
        buf.push_back((0i64, dec!(0), dec!(0)));
        buf.push_back((500, dec!(3), dec!(3))); // "now" — only 500ms of history

        // 5s window: not even the oldest sample (age 500ms) covers it.
        assert!(cvd_window(&buf, 5).is_none());
    }

    #[test]
    fn cvd_uses_sample_closest_to_window() {
        let mut buf = VecDeque::new();
        buf.push_back((0i64, dec!(0), dec!(0))); // age 2000ms since "now"
        buf.push_back((900, dec!(5), dec!(5))); // age 1100ms — meets the 1s window
        buf.push_back((1100, dec!(8), dec!(8))); // age 900ms — does NOT meet the 1s window
        buf.push_back((2000, dec!(20), dec!(20))); // "now"

        let (net, total) = cvd_window(&buf, 1).expect("must cover the 1s window");
        // Must anchor at t=900 (the CLOSEST that qualifies ≥1s), not at t=0 (the
        // oldest), which inflates the real window to 2s instead of 1s.
        assert_eq!(net, dec!(15)); // 20 - 5, NOT 20 - 0
        assert_eq!(total, dec!(15));
    }

    #[test]
    fn cvd_does_not_lose_volume_with_coalescing() {
        // Two trades (notional 30 and 20) arrive so close together that the tokio
        // watch channel coalesces and only the already-summed cum is observed in a single sample
        // — the total must still be recovered intact (30+20=50), not a single jump.
        let mut buf = VecDeque::new();
        buf.push_back((0i64, dec!(0), dec!(0)));
        buf.push_back((1500, dec!(50), dec!(50))); // "now" (30+20 coalesced)

        let (net, total) = cvd_window(&buf, 1).expect("1s window must be covered");
        assert_eq!(net, dec!(50));
        assert_eq!(total, dec!(50));
    }
}

#[cfg(test)]
mod pipe4_tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn liq_none_without_feed() {
        // Without a liquidation feed (`futures_liq_enabled=false` → liq_rx=None),
        // the arm stays inert (pending) and the liq_* fields are never populated —
        // absent signal, same behavior as before this pipe.
        let (price_tx, price_rx) = watch::channel(AssetPrice {
            asset: "btc".to_string(),
            price: dec!(100),
            ..AssetPrice::default()
        });
        let (vol_tx, mut vol_rx) = watch::channel(VolSnapshot::default());
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(run_asset(
            "btc".to_string(),
            price_rx,
            vol_tx,
            None,
            shutdown_rx,
        ));

        price_tx
            .send(AssetPrice {
                asset: "btc".to_string(),
                price: dec!(101),
                ..AssetPrice::default()
            })
            .expect("price_tx must send");

        vol_rx.changed().await.expect("must publish snapshot");
        let snap = vol_rx.borrow().clone();
        assert!(snap.liq_net_60s.is_none());
        assert!(snap.liq_total_60s.is_none());
    }

    #[test]
    fn liq_buf_stays_covered_with_synthetic_pushes_from_price_arm() {
        // Reproduces the bug scenario without touching the select
        // loop — simulates each price_rx tick (liquid market, several/s)
        // pushing a synthetic sample to liq_buf with the last known cum,
        // exactly as the price_rx arm now does. Before the fix, on a
        // liquid market the deadline branch never ran and liq_buf only received
        // the single real forceOrder event — with the fix, even with no more
        // real liquidations, the synthetic pushes keep the window
        // covered and the delta correctly reflects the single real event.
        let mut buf: VecDeque<(i64, Decimal, Decimal)> = VecDeque::new();

        // t=0: single real liquidation event (forceOrder).
        push_cvd_sample(&mut buf, 0, dec!(-500), dec!(500));
        let last_liq_cum = (dec!(-500), dec!(500));

        // t=1..70s: Binance price ticks every 1s (liquid market) — each
        // one pushes the synthetic sample with the cum unchanged.
        for t in 1..=70i64 {
            push_cvd_sample(&mut buf, t * 1000, last_liq_cum.0, last_liq_cum.1);
        }

        let (net, total) = cvd_window(&buf, LIQ_WINDOW_SECS)
            .expect("the synthetic pushes per price tick must cover the window");
        // With no new liquidations in the last 60s, the delta is 0 — correct,
        // NOT None (which would be the bug: stale buffer not covering the window).
        assert_eq!(net, dec!(0));
        assert_eq!(total, dec!(0));
    }

    #[test]
    fn liq_window_with_periodic_synthetic_samples() {
        // Simulates the periodic 5s deadline push on a market WITHOUT new
        // liquidations: the cum repeats unchanged in each synthetic sample. After 60s of
        // history (13 samples every 5s) the window must resolve — the delta is
        // 0 because there were no new liquidations, which is REAL information (quiet
        // market), not absence of signal (that would be None).
        let mut buf: VecDeque<(i64, Decimal, Decimal)> = VecDeque::new();
        let last_cum = (dec!(-500), dec!(500)); // single real liquidation at startup
        for i in 0..=12i64 {
            push_cvd_sample(&mut buf, i * 5000, last_cum.0, last_cum.1);
        }
        let (net, total) = cvd_window(&buf, LIQ_WINDOW_SECS)
            .expect("60s of synthetic samples must cover the window");
        assert_eq!(net, dec!(0));
        assert_eq!(total, dec!(0));
    }
}

#[cfg(test)]
mod sigma_fv_tests {
    use super::*;
    use rust_decimal_macros::dec;

    /// Mandatory integration test: 400 ticks in
    /// ~400ms, all at the same price except one. With the tick buffer
    /// (cap BY COUNT, `MAX_SAMPLES=120`) the old estimator produced a
    /// large, spurious `sigma_60s` (sub-ms dt). With the buffer subsampled to
    /// 1s, 400ms of history does not cover even a fraction of the 60s/300s
    /// window (not even 2 cells) — `sigma_long` must be `None`.
    #[tokio::test]
    async fn burst_of_400_ticks_in_400ms_produces_no_spurious_long_sigma() {
        let (price_tx, price_rx) = watch::channel(AssetPrice {
            asset: "btc".to_string(),
            price: dec!(50000),
            ..AssetPrice::default()
        });
        let (vol_tx, mut vol_rx) = watch::channel(VolSnapshot::default());
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(run_asset(
            "btc".to_string(),
            price_rx,
            vol_tx,
            None,
            shutdown_rx,
        ));

        for i in 0..400i64 {
            let price = if i == 200 { dec!(50100) } else { dec!(50000) };
            let _ = price_tx.send(AssetPrice {
                asset: "btc".to_string(),
                price,
                ..AssetPrice::default()
            });
            tokio::task::yield_now().await;
        }

        // The publish throttle (200ms) will not yet have expired after the burst
        // above (runs in microseconds of real time) — we force one more tick
        // after waiting out the interval to force the publish.
        tokio::time::sleep(Duration::from_millis(250)).await;
        let _ = price_tx.send(AssetPrice {
            asset: "btc".to_string(),
            price: dec!(50000),
            ..AssetPrice::default()
        });

        vol_rx.changed().await.expect("must publish snapshot");
        let snap = vol_rx.borrow().clone();
        assert!(snap.sigma_long.is_none(),
            "400 ticks in <1s of real history must not produce sigma_long (300s window not covered)");
    }
}
