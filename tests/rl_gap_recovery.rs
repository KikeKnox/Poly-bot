//! GAP recovery contract of the reconstructor (`Reconstructor`) that the
//! `rl_policy_env` driver uses as PROOF of recovery: a `BookObserve`
//! can only be emitted when NO half has `needs_snapshot` and both
//! books exist (`emit()`). Therefore the first book
//! observed after a GAP mark proves that a valid real snapshot arrived
//! and rebuilt the base of the invalidated half (or halves).
//!
//! Cases:
//! - GAP mark (2nd consecutive cross) → total suppression of emissions until a
//!   real snapshot; marks are placed at the causing event's `avail_ts_ms`
//!   and are NEVER deleted.
//! - recovery requires a real snapshot of EACH invalidated half — with a
//!   single re-snapshotted half there is no emission (rejection).
//! - a crossed snapshot does not clear the state (it is not a recovery).

use poly_bot::research::contract::{
    EventKind, EventSource, MarketMeta, QualityFlags, ResolutionRule, ResearchEvent,
    SCHEMA_VERSION,
};
use poly_bot::research_replay::{ResearchTape, TapeOpts};
use rust_decimal_macros::dec;

const T0: i64 = 1_700_000_000_000;
const CID: &str = "0xcond-gap";
const YES: &str = "tok-yes-gap";
const NO: &str = "tok-no-gap";
const SLUG: &str = "btc-updown-5m-1700000000";
const CLEAR_AT_S: i64 = T0 / 1000 + 60;

fn ev(
    source: EventSource,
    kind: EventKind,
    cid: Option<&str>,
    token: Option<&str>,
    payload: serde_json::Value,
    recv: i64,
    avail: i64,
    seq: u64,
) -> ResearchEvent {
    ResearchEvent::new(
        source,
        kind,
        cid.map(str::to_string),
        token.map(str::to_string),
        None,
        payload,
        Some(recv),
        recv,
        avail,
        seq,
        QualityFlags::NONE,
    )
    .expect("valid event")
}

fn meta_ev(seq: u64, avail: i64) -> ResearchEvent {
    let meta = MarketMeta {
        schema_version: SCHEMA_VERSION,
        market_cid: CID.into(),
        slug: SLUG.into(),
        yes_token_id: YES.into(),
        no_token_id: NO.into(),
        asset: Some("BTC".into()),
        neg_risk: false,
        market_open_ts: T0 / 1000,
        clear_at: CLEAR_AT_S,
        reference_price: Some(dec!(50_000)),
        rule: ResolutionRule::UpDownReference,
        resolution_source: EventSource::PolymarketRest,
        resolution_source_ref: Some("gamma:test".into()),
        rules_windows: vec![poly_bot::research_contract::RulesWindow {
            valid_from_ms: T0,
            valid_to_ms: None,
            parabolic_fee_rate: None,
            fee_rate_bps: None,
            tick_size: Some(dec!(0.01)),
            min_order_size: None,
        }],
        outcome: None,
    };
    ev(
        EventSource::PolymarketRest,
        EventKind::MarketMeta,
        Some(CID),
        None,
        serde_json::to_value(meta).unwrap(),
        avail,
        avail,
        seq,
    )
}

/// WS `book` snapshot of a token (raw wire payload).
fn book_ev(token: &str, bid: &str, ask: &str, seq: u64, avail: i64) -> ResearchEvent {
    let payload = serde_json::json!({
        "event_type": "book",
        "asset_id": token,
        "market": CID,
        "timestamp": avail.to_string(),
        "bids": [{"price": bid, "size": "1000"}],
        "asks": [{"price": ask, "size": "1000"}],
    });
    ev(
        EventSource::PolymarketWs,
        EventKind::BookSnapshot,
        Some(CID),
        Some(token),
        payload,
        avail,
        avail,
        seq,
    )
}

/// `price_change` delta (one entry per event, like the real wire).
fn delta_ev(token: &str, price: &str, size: &str, side: &str, seq: u64, avail: i64) -> ResearchEvent {
    let payload = serde_json::json!({
        "event_type": "price_change",
        "market": CID,
        "price_changes": [{
            "asset_id": token,
            "price": price,
            "size": size,
            "side": side,
        }],
    });
    ev(
        EventSource::PolymarketWs,
        EventKind::BookDelta,
        Some(CID),
        Some(token),
        payload,
        avail,
        avail,
        seq,
    )
}

/// Healthy base: meta + snapshots of both tokens (emits one book per market).
fn tape_base() -> Vec<ResearchEvent> {
    vec![
        meta_ev(1, T0),
        book_ev(YES, "0.49", "0.51", 2, T0 + 20),
        book_ev(NO, "0.48", "0.50", 3, T0 + 40),
    ]
}

fn gap_marks(rt: &ResearchTape) -> Vec<i64> {
    rt.tape
        .quality
        .iter()
        .filter(|q| q.flag.contains(QualityFlags::GAP))
        .map(|q| q.ts_ms)
        .collect()
}

/// Two consecutive deltas that leave the token crossed → GAP mark.
fn cross_twice(token: &str, seq: &mut u64, avail: &mut i64) -> Vec<ResearchEvent> {
    *avail += 10;
    *seq += 1;
    let d1 = delta_ev(token, "0.55", "10", "BUY", *seq, *avail); // bid 0.55 >= ask
    *avail += 10;
    *seq += 1;
    let d2 = delta_ev(token, "0.56", "10", "BUY", *seq, *avail); // still crossed
    vec![d1, d2]
}

#[test]
fn gap_without_snapshot_emits_no_books() {
    let mut seq = 3u64;
    let mut avail = T0 + 40;
    let mut events = tape_base();
    events.extend(cross_twice(YES, &mut seq, &mut avail));
    let mark_ts = avail;
    // Later deltas with the asset in needs_snapshot: discarded, no emission.
    for _ in 0..3 {
        avail += 10;
        seq += 1;
        events.push(delta_ev(YES, "0.50", "10", "SELL", seq, avail));
    }
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());

    let marks = gap_marks(&rt);
    assert_eq!(marks, vec![mark_ts], "one mark, at the ts of the 2nd cross");
    // No book emitted after the mark (the last one is the base's).
    assert!(
        rt.tape
            .books
            .iter()
            .all(|b| b.avail_ms.unwrap_or(0) < mark_ts),
        "there must be no BookObserve after the mark without a recovery snapshot"
    );
}

#[test]
fn gap_recovers_with_real_snapshots_of_both_halves() {
    let mut seq = 3u64;
    let mut avail = T0 + 40;
    let mut events = tape_base();
    // Mark on BOTH tokens.
    events.extend(cross_twice(YES, &mut seq, &mut avail));
    events.extend(cross_twice(NO, &mut seq, &mut avail));
    let mark_ts = avail;
    // Recovery: valid real snapshots of BOTH halves.
    avail += 10;
    seq += 1;
    events.push(book_ev(YES, "0.49", "0.51", seq, avail));
    avail += 10;
    seq += 1;
    events.push(book_ev(NO, "0.48", "0.50", seq, avail));
    // Post-recovery delta: applied and emitted (book already rebuilt).
    avail += 10;
    seq += 1;
    events.push(delta_ev(YES, "0.495", "5", "BUY", seq, avail));

    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    assert_eq!(gap_marks(&rt).len(), 2, "one mark per token");
    let recovered: Vec<i64> = rt
        .tape
        .books
        .iter()
        .filter_map(|b| b.avail_ms)
        .filter(|t| *t > mark_ts)
        .collect();
    assert!(
        !recovered.is_empty(),
        "after valid real snapshots, books are emitted again"
    );
}

#[test]
fn gap_only_one_half_does_not_recover() {
    let mut seq = 3u64;
    let mut avail = T0 + 40;
    let mut events = tape_base();
    // Mark on BOTH tokens.
    events.extend(cross_twice(YES, &mut seq, &mut avail));
    events.extend(cross_twice(NO, &mut seq, &mut avail));
    let mark_ts = avail;
    let n_books_before = {
        let rt = ResearchTape::from_events(events.clone(), vec![], 0, &TapeOpts::default());
        rt.tape.books.len()
    };
    // Only the YES snapshot arrives — NO still has needs_snapshot → no emission.
    avail += 10;
    seq += 1;
    events.push(book_ev(YES, "0.49", "0.51", seq, avail));
    avail += 10;
    seq += 1;
    events.push(delta_ev(NO, "0.47", "5", "BUY", seq, avail)); // discarded

    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    assert!(
        rt.tape
            .books
            .iter()
            .filter_map(|b| b.avail_ms)
            .all(|t| t <= mark_ts),
        "with only one half re-snapshotted there must be no emission after the mark"
    );
    assert_eq!(rt.tape.books.len(), n_books_before);
}

#[test]
fn crossed_snapshot_is_not_recovery() {
    let mut seq = 3u64;
    let mut avail = T0 + 40;
    let mut events = tape_base();
    events.extend(cross_twice(YES, &mut seq, &mut avail));
    let mark_ts = avail;
    // "Recovering" snapshot but crossed → discarded, still needs_snapshot.
    avail += 10;
    seq += 1;
    events.push(book_ev(YES, "0.60", "0.58", seq, avail)); // bid > ask
    avail += 10;
    seq += 1;
    events.push(delta_ev(YES, "0.50", "5", "SELL", seq, avail));

    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    assert!(
        gap_marks(&rt).iter().all(|t| *t <= avail),
        "the marks remain recorded"
    );
    assert!(
        rt.tape
            .books
            .iter()
            .filter_map(|b| b.avail_ms)
            .all(|t| t <= mark_ts),
        "a crossed snapshot does not recover — no BookObserve after the mark"
    );
    // The crossed snapshot's mark is also recorded (two marks:
    // the 2nd cross's and the discarded crossed snapshot's).
    assert_eq!(gap_marks(&rt).len(), 2);
}
