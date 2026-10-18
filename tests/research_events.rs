//! Integration tests of the multi-source causal recorder.
//!
//! Covers the cases: trade without a book change, several trades per
//! book, out-of-order event, duplicate from resend, reconnection
//! (new generation), writer saturation and absence of a remote
//! timestamp.

use std::fs;
use std::path::PathBuf;

use poly_bot::research::contract::{EventKind, EventSource, QualityFlags};
use poly_bot::research::events::{
    EventInput, Hub, RecordedLine, ResearchRecorder, ResearchRecorderConfig,
};

fn cfg(dir: &std::path::Path, capacity: usize) -> ResearchRecorderConfig {
    ResearchRecorderConfig {
        dir: dir.to_path_buf(),
        channel_capacity: capacity,
        ..ResearchRecorderConfig::default()
    }
}

/// Reads all the JSONL lines recorded under `dir` (one file per
/// source/day) in `local_seq` order — the ledger's canonical order.
fn read_events(dir: &std::path::Path) -> Vec<RecordedLine> {
    let mut lines = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            for f in fs::read_dir(entry.path()).unwrap() {
                let f = f.unwrap();
                if f.path().extension().is_some_and(|e| e == "jsonl") {
                    files.push(f.path());
                }
            }
        }
    }
    files.sort();
    for f in files {
        for line in fs::read_to_string(f).unwrap().lines() {
            if line.trim().is_empty() {
                continue;
            }
            lines.push(serde_json::from_str::<RecordedLine>(line).unwrap());
        }
    }
    lines.sort_by_key(|l| l.event.local_seq);
    lines
}

fn entry(source: EventSource, kind: EventKind, recv_ts_ms: i64) -> EventInput {
    EventInput::new(source, kind, serde_json::json!({"x": 1}), recv_ts_ms)
}

// ── Basic recording ─────────────────────────────────────────────────────────

/// A sent event is recorded as one JSONL line with all its
/// fields, and the line keeps an increasing `local_seq`.
#[tokio::test]
async fn registered_event_is_recorded_in_jsonl() {
    let dir = tempfile::tempdir().unwrap();
    let hub = Hub::new();
    let recorder = ResearchRecorder::start(&cfg(dir.path(), 16), &hub).unwrap();
    let prod = hub.producer();

    let mut e = entry(EventSource::BinanceWs, EventKind::PriceTick, 1_000);
    e.asset = Some("btc".into());
    e.origin_ts_ms = Some(900);
    e.avail_ts_ms = 1_010;
    assert!(prod.record(e));

    drop(prod);
    let res = recorder.stop().await;
    assert_eq!(res.enqueued, 1);
    assert_eq!(res.written, 1);

    let events = read_events(dir.path());
    assert_eq!(events.len(), 1);
    let ev = &events[0].event;
    assert_eq!(ev.source, EventSource::BinanceWs);
    assert_eq!(ev.kind, EventKind::PriceTick);
    assert_eq!(ev.asset.as_deref(), Some("btc"));
    assert_eq!(ev.origin_ts_ms, Some(900));
    assert_eq!(ev.recv_ts_ms, 1_000);
    assert_eq!(ev.avail_ts_ms, 1_010);
    assert!(ev.quality.is_clean());
}

/// Absent remote timestamp: `origin_ts_ms=None` is stored as `null` on
/// disk (NEVER invented) and the event carries the MISSING_REMOTE_TS mark.
#[tokio::test]
async fn event_without_remote_timestamp_is_marked_not_invented() {
    let dir = tempfile::tempdir().unwrap();
    let hub = Hub::new();
    let recorder = ResearchRecorder::start(&cfg(dir.path(), 16), &hub).unwrap();
    let prod = hub.producer();

    let e = entry(EventSource::PolymarketWs, EventKind::Trade, 5_000);
    assert!(prod.record(e));

    drop(prod);
    let res = recorder.stop().await;
    assert_eq!(res.written, 1);

    let events = read_events(dir.path());
    let ev = &events[0].event;
    assert_eq!(ev.origin_ts_ms, None, "the missing remote ts is NOT invented");
    assert!(
        ev.quality.contains(QualityFlags::MISSING_REMOTE_TS),
        "must carry the MISSING_REMOTE_TS mark"
    );
}

/// Writer saturation: with a minimal channel and a large burst, each
/// event is recorded OR counted as discarded — the conservation law
/// `enqueued + discarded_total == sent_count` always holds, and the drops
/// are visible in `_recorder_stats.jsonl`.
#[tokio::test]
async fn saturation_counts_discards_without_blocking() {
    let dir = tempfile::tempdir().unwrap();
    let hub = Hub::new();
    let recorder = ResearchRecorder::start(&cfg(dir.path(), 1), &hub).unwrap();
    let prod = hub.producer();

    const TOTAL: u64 = 2_000;
    let mut sent_count = 0u64;
    for i in 0..TOTAL {
        let mut e = entry(EventSource::BinanceWs, EventKind::PriceTick, i as i64);
        e.dedup_key = Some(i); // each one distinct → the enqueued ones get written
        let _ = prod.record(e);
        sent_count += 1;
    }

    drop(prod);
    let res = recorder.stop().await;
    assert_eq!(
        res.enqueued + res.discarded,
        sent_count,
        "every event is recorded or counted as lost"
    );
    assert_eq!(res.written, res.enqueued);

    // The loss can be queried offline in the stats sidecar.
    let stats = dir.path().join("_recorder_stats.jsonl");
    let content = fs::read_to_string(&stats).unwrap();
    let last = content.lines().last().unwrap();
    let v: serde_json::Value = serde_json::from_str(last).unwrap();
    assert_eq!(
        v["discarded"].as_u64().unwrap(),
        res.discarded,
        "the drops must be persisted in the stats sidecar"
    );
}

// ── Dedup / order / generation ───────────────────────────────────────────────

/// The same trade represented in two messages (stable dedup_key derived
/// from event fields — never recv_ts or local_seq) is recorded ONCE:
/// volume is not duplicated and the dedup is counted.
#[tokio::test]
async fn duplicate_by_stable_content_does_not_duplicate_record() {
    let dir = tempfile::tempdir().unwrap();
    let hub = Hub::new();
    let recorder = ResearchRecorder::start(&cfg(dir.path(), 16), &hub).unwrap();
    let prod = hub.producer();

    for recv in [1_000i64, 9_999] {
        let mut e = entry(EventSource::PolymarketWs, EventKind::Trade, recv);
        e.token_id = Some("tok-1".into());
        // Key from stable trade fields: a different recv does NOT change it.
        e.dedup_key = Some(0xBEEF);
        assert!(prod.record(e));
    }

    drop(prod);
    let res = recorder.stop().await;
    assert_eq!(res.written, 1, "the resend does not write a second line");
    assert_eq!(res.deduped, 1);

    let events = read_events(dir.path());
    assert_eq!(events.len(), 1);
}

/// Out-of-order event WITHIN a generation: it is recorded (the data is
/// not lost) marked as a sequence anomaly (DUPLICATE), and the
/// canonical order remains `local_seq`.
#[tokio::test]
async fn out_of_order_event_is_recorded_marked() {
    let dir = tempfile::tempdir().unwrap();
    let hub = Hub::new();
    let recorder = ResearchRecorder::start(&cfg(dir.path(), 16), &hub).unwrap();
    let prod = hub.producer();
    for seq in [10i64, 7] {
        let mut e = entry(EventSource::BinanceWs, EventKind::PriceTick, seq);
        e.origin_ts_ms = Some(seq);
        e.stream = Some("binance:btc".into());
        e.generation = Some(1);
        e.source_seq = Some(seq);
        assert!(prod.record(e));
    }

    drop(prod);
    recorder.stop().await;

    let events = read_events(dir.path());
    assert_eq!(events.len(), 2);
    assert!(events[0].event.quality.is_clean());
    assert!(
        events[1].event.quality.contains(QualityFlags::DUPLICATE),
        "seq going backwards within the same generation → DUPLICATE mark"
    );
}

/// Seq gap within a generation → GAP mark.
#[tokio::test]
async fn seq_gap_marks_gap() {
    let dir = tempfile::tempdir().unwrap();
    let hub = Hub::new();
    let recorder = ResearchRecorder::start(&cfg(dir.path(), 16), &hub).unwrap();
    let prod = hub.producer();

    for seq in [10i64, 15] {
        let mut e = entry(EventSource::BinanceWs, EventKind::PriceTick, seq);
        e.stream = Some("binance:btc".into());
        e.generation = Some(1);
        e.source_seq = Some(seq);
        assert!(prod.record(e));
    }

    drop(prod);
    recorder.stop().await;

    let events = read_events(dir.path());
    assert!(events[1].event.quality.contains(QualityFlags::GAP));
}

/// Reconnection: a NEW generation always wins even if its seq is lower —
/// RESYNC mark, without comparing seq across epochs.
#[tokio::test]
async fn reconnection_new_generation_marks_resync() {
    let dir = tempfile::tempdir().unwrap();
    let hub = Hub::new();
    let recorder = ResearchRecorder::start(&cfg(dir.path(), 16), &hub).unwrap();
    let prod = hub.producer();

    let mut e1 = entry(EventSource::BinanceWs, EventKind::PriceTick, 1_000);
    e1.stream = Some("binance:btc".into());
    e1.generation = Some(1);
    e1.source_seq = Some(10);
    prod.record(e1);

    // New connection: Binance's trade id stream continues, but a
    // source with per-connection seq could restart — a lower seq is valid.
    let mut e2 = entry(EventSource::BinanceWs, EventKind::PriceTick, 2_000);
    e2.stream = Some("binance:btc".into());
    e2.generation = Some(2);
    e2.source_seq = Some(3);
    prod.record(e2);

    drop(prod);
    recorder.stop().await;

    let events = read_events(dir.path());
    assert_eq!(events.len(), 2);
    assert!(events[1].event.quality.contains(QualityFlags::RESYNC));
    assert!(!events[1].event.quality.contains(QualityFlags::DUPLICATE));
    // The line persists the venue's generation and seq next to the event.
    assert_eq!(events[1].generation, Some(2));
    assert_eq!(events[1].source_seq, Some(3));
    assert_eq!(events[1].stream.as_deref(), Some("binance:btc"));
}

// ── Hook adapters (pure functions) ──────────────────────────────

fn ctx(generation: u64, stream: &str) -> poly_bot::research::events::SourceContext {
    // Null producer (sink): we only care about the EventInput that
    // the adapter returns, not the recording.
    poly_bot::research::events::SourceContext {
        producer: poly_bot::research::events::EventProducer::sink(),
        generation,
        stream: stream.into(),
    }
}

/// The Binance tick adapter builds an `EventInput` with:
/// - `origin_ts_ms = T` (trade time of Binance's engine — real, not local);
/// - `source_seq`/`dedup_key` derived from the venue's trade id (stable);
/// - stream `binance:<asset>` and the connection's generation.
#[test]
fn binance_adapter_derives_origin_seq_and_trade_dedup() {
    let c = ctx(3, "binance");
    let e = poly_bot::oracle::price::event_from_binance_tick(
        "BTCUSDT",
        "50000.5",
        "0.001",
        true,
        1_784_652_361_123,
        Some(777),
        1_700_000_000_000,
        &c,
    );
    assert_eq!(e.source, EventSource::BinanceWs);
    assert_eq!(e.kind, EventKind::PriceTick);
    assert_eq!(e.asset.as_deref(), Some("btc"));
    assert_eq!(e.origin_ts_ms, Some(1_784_652_361_123));
    assert_eq!(e.source_seq, Some(777));
    assert_eq!(e.stream.as_deref(), Some("binance:btc"));
    assert_eq!(e.generation, Some(3));
    assert!(e.dedup_key.is_some());
    // The payload keeps the venue's original strings (not reparsed).
    assert_eq!(e.payload["price"], serde_json::json!("50000.5"));

    // The dedup does NOT depend on recv_ts: same trade id → same key even if
    // the local receipt changes.
    let e2 = poly_bot::oracle::price::event_from_binance_tick(
        "BTCUSDT",
        "50000.5",
        "0.001",
        true,
        1_784_652_361_123,
        Some(777),
        9_999_999_999_999,
        &c,
    );
    assert_eq!(e.dedup_key, e2.dedup_key, "the key does not carry recv_ts");
}

/// Without a venue trade id (`t` absent), the dedup falls back to stable fields of the
/// payload — never to recv_ts.
#[test]
fn binance_adapter_without_trade_id_dedups_by_content() {
    let c = ctx(3, "binance");
    let a = poly_bot::oracle::price::event_from_binance_tick(
        "ETHUSDT", "3000", "1", false, 111, None, 1_000, &c,
    );
    let b = poly_bot::oracle::price::event_from_binance_tick(
        "ETHUSDT", "3000", "1", false, 111, None, 2_000, &c,
    );
    assert_eq!(a.dedup_key, b.dedup_key);
    assert_eq!(a.source_seq, None);
}

/// The `SpotSample` adapter honestly does NOT invent a remote timestamp:
/// the sample's `source_ts_ms` is approximate (see the docs of
/// `tools::recorders::spot_stream::from_vol_snapshot`), so `origin_ts_ms` stays `None`
/// and the approximate ts travels in the labeled payload.
#[test]
fn spot_sample_adapter_does_not_invent_origin_ts() {
    use poly_bot::tools::recorders::spot_stream::{event_from_sample, SpotSample};
    use rust_decimal_macros::dec;

    let m = SpotSample {
        asset: "btc".into(),
        source: "binance".into(),
        price: dec!(63123.45),
        source_ts_ms: 1_700_000_000_123,
        recv_ts_ms: 1_700_000_000_456,
        avail_ts_ms: 1_700_000_000_789,
    };
    let e = event_from_sample(&m, 4);
    assert_eq!(e.kind, EventKind::PriceTick);
    assert_eq!(e.origin_ts_ms, None, "the approximate ts is not a real origin");
    assert_eq!(e.recv_ts_ms, 1_700_000_000_456);
    assert_eq!(e.avail_ts_ms, 1_700_000_000_789);
    assert_eq!(e.generation, Some(4));
    assert_eq!(
        e.payload["source_ts_ms_approx"],
        serde_json::json!(1_700_000_000_123i64)
    );
    // Data derived from the bot → SYNTHETIC mark.
    assert!(e.quality_extra.contains(QualityFlags::SYNTHETIC));
}
