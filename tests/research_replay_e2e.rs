//! E2E of the causal tape `ResearchEvent` → `Tape` → `ReplayEngine`.
//!
//!
//! Properties verified:
//! 1. Strict ingestion: a malformed line aborts with `--strict`; without it
//!    it lands in `diagnostics` and loading continues (never invented).
//! 2. Canonical order `(avail_ts_ms, local_seq)` — the remote clock
//!    (`origin_ts_ms`) orders NOTHING.
//! 3. Book reconstruction: authoritative snapshot per token → market
//!    book when both halves exist; an incremental delta applies;
//!    a delta without a base → discarded + `GAP` mark (never a fabricated book).
//! 4. Official settlement: a label with `known_at_ms <= now` settles; a
//!    "future" label is invisible at the step and the market stays pending; without a
//!    label there is NEVER a silent proxy (declared pending); a post-settle
//!    correction is counted and invalidates the certification.
//! 5. Same tape + same seed ⇒ same economic digest.
//!
//! The replay clock is process-global (`crate::time::REPLAY_NOW_MS`)
//! — all runs are serialized with `REPLAY_LOCK`.

use std::collections::HashSet;

use rust_decimal_macros::dec;

use poly_bot::{
    replay::{ReplayEngine, ReplayParams, Tape},
    research_contract::{
        EventKind, EventSource, MarketMeta, MarketOutcome, OutcomeSide, QualityFlags,
        ResearchEvent, ResolutionRule, SCHEMA_VERSION,
    },
    research_events::RecordedLine,
    research_replay::{EnvAgent, ResearchTape, TapeOpts},
    state::StateStore,
    types::Mode,
};

/// Serialization of all runs: the virtual replay clock is global
/// to the process — see `tests/replay_pipeline.rs` (same pattern).
static REPLAY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const T0: i64 = 1_700_000_000_000; // ms epoch
const CID: &str = "0xcond-e2e";
const YES: &str = "tok-yes-e2e";
const NO: &str = "tok-no-e2e";
const SLUG: &str = "btc-updown-5m-1700000000";
/// The market's `clear_at` in unix SECONDS (registry domain).
const CLEAR_AT_S: i64 = T0 / 1000 + 60; // closes 60s after T0

// ── Event constructors ─────────────────────────────────────────────────

fn ev(
    source: EventSource,
    kind: EventKind,
    cid: Option<&str>,
    token: Option<&str>,
    asset: Option<&str>,
    payload: serde_json::Value,
    origin: Option<i64>,
    recv: i64,
    avail: i64,
    seq: u64,
    quality: QualityFlags,
) -> ResearchEvent {
    ResearchEvent::new(
        source,
        kind,
        cid.map(str::to_string),
        token.map(str::to_string),
        asset.map(str::to_string),
        payload,
        origin,
        recv,
        avail,
        seq,
        quality,
    )
    .expect("valid event")
}

fn meta_ev(cid: &str, seq: u64, avail: i64) -> ResearchEvent {
    let meta = MarketMeta {
        schema_version: SCHEMA_VERSION,
        market_cid: cid.into(),
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
        Some(cid),
        None,
        Some("BTC"),
        serde_json::to_value(meta).unwrap(),
        None,
        avail,
        avail,
        seq,
        QualityFlags::NONE,
    )
}

/// WS `book` snapshot of a token (raw wire payload).
fn book_ev(
    token: &str,
    bid: &str,
    ask: &str,
    size: &str,
    seq: u64,
    recv: i64,
    avail: i64,
) -> ResearchEvent {
    let payload = serde_json::json!({
        "event_type": "book",
        "asset_id": token,
        "market": CID,
        "timestamp": avail.to_string(),
        "bids": [{"price": bid, "size": size}],
        "asks": [{"price": ask, "size": size}],
    });
    ev(
        EventSource::PolymarketWs,
        EventKind::BookSnapshot,
        Some(CID),
        Some(token),
        None,
        payload,
        Some(recv), // real origin_ts_ms from the wire
        recv,
        avail,
        seq,
        QualityFlags::NONE,
    )
}

/// `price_change` delta of a token.
fn delta_ev(
    token: &str,
    price: &str,
    size: &str,
    side: &str,
    seq: u64,
    recv: i64,
    avail: i64,
) -> ResearchEvent {
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
        None,
        payload,
        Some(recv),
        recv,
        avail,
        seq,
        QualityFlags::NONE,
    )
}

fn res_ev(cid: &str, winner: OutcomeSide, known_at: i64, seq: u64, avail: i64) -> ResearchEvent {
    let outcome = MarketOutcome::new(winner, known_at, "gamma:test-resolution").unwrap();
    ev(
        EventSource::PolymarketRest,
        EventKind::Resolution,
        Some(cid),
        None,
        None,
        serde_json::to_value(outcome).unwrap(),
        None,
        avail,
        avail,
        seq,
        QualityFlags::NONE,
    )
}

fn tick_ev(asset: &str, price: &str, seq: u64, recv: i64, avail: i64) -> ResearchEvent {
    ev(
        EventSource::BinanceWs,
        EventKind::PriceTick,
        None,
        None,
        Some(asset),
        serde_json::json!({"price": price}),
        Some(recv),
        recv,
        avail,
        seq,
        QualityFlags::NONE,
    )
}

fn tape_base() -> Vec<ResearchEvent> {
    vec![
        meta_ev(CID, 1, T0),
        book_ev(YES, "0.49", "0.51", "1000", 2, T0 + 10, T0 + 20),
        book_ev(NO, "0.48", "0.50", "1000", 3, T0 + 30, T0 + 40),
        tick_ev("BTC", "50100.5", 4, T0 + 50, T0 + 50),
    ]
}

fn params() -> ReplayParams {
    ReplayParams {
        seed: 0xC0FF_EE15,
        initial_balance: dec!(1000),
        risk: poly_bot::config::RiskConfig::default(),
        obs_delay_ms: 0,
        latency_ms: 0,
        maintenance_ms: 30_000,
        timer_ms: 5_000,
        fak_strategies: HashSet::new(),
        max_book_age_ms: 300_000,
        research: None,
        settlement_proxy_diagnostic: false,
    }
}

// ── Ingestion and reconstruction ──────────────────────────────────────────────

#[test]
fn canonical_order_avail_local_seq_not_origin() {
    // Two snapshots of the SAME token with an `avail` that matches neither the
    // arrival order NOR `origin_ts_ms` — the tape must follow the
    // canonical key `(avail_ts_ms, local_seq)`, never the remote clock.
    //
    // seq 2: avail=105, origin VERY late (T0+900_000) — if the tape
    //   ordered by origin it would fall at the END and its book (0.47) would be the
    //   last YES state.
    // seq 3: avail=110, early origin — causal order: seq2 → seq3, and the
    //   last YES state is 0.49/0.51.
    let mut e2 = book_ev(YES, "0.47", "0.49", "1000", 2, T0 + 105, T0 + 105);
    e2.origin_ts_ms = Some(T0 + 900_000);
    let mut events = vec![
        meta_ev(CID, 1, T0),
        book_ev(YES, "0.49", "0.51", "1000", 3, T0 + 100, T0 + 110),
        book_ev(NO, "0.48", "0.50", "1000", 4, T0 + 120, T0 + 130),
        e2,
    ];
    // Deliberate disorder of the entry (as it would arrive from two files).
    events.swap(1, 3);
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());

    // The first market book is emitted when the second half (NO) arrives —
    // the previous YES books were left "without a half" (as in production).
    assert_eq!(rt.tape.books.len(), 1, "only one market book emitted");
    assert_eq!(
        rt.books_missing_half, 2,
        "two YES books without the opposite half"
    );
    let b = &rt.tape.books[0];
    // Final YES state = that of seq 3 (avail 110, last in causal order) —
    // NOT that of seq 2 even though its origin is the greatest.
    assert_eq!(b.snap.yes.bids.levels[0].price, dec!(0.49));
    assert_eq!(b.snap.yes.asks.levels[0].price, dec!(0.51));
    assert_eq!(b.tape_seq, Some(4));
    assert_eq!(b.avail_ms, Some(T0 + 130));
    assert_eq!(b.recv_ms, T0 + 120);
}

#[test]
fn strict_decode_rejects_malformed_line() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("tape.jsonl");
    let good_line = serde_json::to_string(&RecordedLine {
        event: meta_ev(CID, 1, T0),
        stream: None,
        generation: None,
        source_seq: None,
    })
    .unwrap();
    std::fs::write(&f, format!("{good_line}\n{{json broken!!!\n")).unwrap();

    // Strict: aborts.
    let err = ResearchTape::load(
        &f,
        &TapeOpts {
            strict: true,
            ..Default::default()
        },
    );
    assert!(
        err.is_err(),
        "a malformed line must abort in strict mode"
    );

    // Tolerant: diagnoses and continues — never fabricates the event.
    let rt = ResearchTape::load(&f, &TapeOpts::default()).unwrap();
    assert_eq!(rt.events, 1);
    assert_eq!(rt.malformed_lines, 1);
    assert_eq!(rt.diagnostics.len(), 1);
    assert!(rt.diagnostics[0].message.contains("malformed JSON"));
}

#[test]
fn snapshot_delta_rebuilds_book() {
    let mut events = tape_base();
    // Delta over YES: new bid 0.50 (BUY → bids) and ask rises to 0.52.
    events.push(delta_ev(YES, "0.50", "800", "BUY", 5, T0 + 60, T0 + 70));
    events.push(delta_ev(YES, "0.51", "0", "SELL", 6, T0 + 80, T0 + 90)); // deletes ask 0.51
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());

    // Books emitted: (a) when the NO snapshot arrives (both halves) forced;
    // (b) bid 0.50 delta (top changes); (c) ask-deleted delta (top changes).
    assert_eq!(
        rt.tape.books.len(),
        3,
        "books emitted: {:#?}",
        rt.tape.books.len()
    );
    let last = rt.tape.books.last().unwrap();
    assert_eq!(last.snap.yes.bids.levels[0].price, dec!(0.50));
    // The 0.51 ask was deleted with size=0 — asks top empty.
    assert!(last.snap.yes.asks.levels.is_empty());
    assert_eq!(last.tape_seq, Some(6));
    assert_eq!(last.avail_ms, Some(T0 + 90));
}

#[test]
fn delta_without_base_marks_gap_does_not_fabricate_book() {
    // Delta BEFORE any snapshot of the token → discarded + GAP.
    let mut events = vec![meta_ev(CID, 1, T0)];
    events.push(delta_ev(YES, "0.50", "800", "BUY", 2, T0 + 10, T0 + 10));
    events.push(book_ev(NO, "0.48", "0.50", "1000", 3, T0 + 20, T0 + 20));
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());

    assert_eq!(rt.discarded_deltas, 1);
    assert!(
        rt.diagnostics
            .iter()
            .any(|d| d.message.contains("without a previous snapshot")),
        "diagnostic of a delta without a base"
    );
    // GAP mark in the tape (positioned at the delta's avail).
    assert_eq!(rt.tape.quality.len(), 1);
    assert!(rt.tape.quality[0].flag.contains(QualityFlags::GAP));
    assert_eq!(rt.tape.quality[0].ts_ms, T0 + 10);
    // The NO book alone does NOT emit (YES half without state — as in production).
    assert!(rt.tape.books.is_empty());
    assert_eq!(rt.books_missing_half, 1);
}

#[test]
fn structural_quality_propagates_to_tape() {
    // An event with the recorder's GAP flag → TapeQuality at avail_ts_ms.
    let mut events = tape_base();
    let mut gap = book_ev(YES, "0.49", "0.51", "1000", 5, T0 + 60, T0 + 60);
    gap.quality = QualityFlags::GAP;
    events.push(gap);
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    assert!(
        rt.tape
            .quality
            .iter()
            .any(|q| q.flag.contains(QualityFlags::GAP) && q.ts_ms == T0 + 60),
        "the recorder's GAP must propagate to tape.quality"
    );
}

// ── Official settlement in the pipeline ────────────────────────────────────

async fn run_tape(tape: Tape) -> poly_bot::replay::ReplayOutcome {
    let _g = REPLAY_LOCK.lock().await;
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let mut engine = ReplayEngine::new(params(), state).await.unwrap();
    let mut agent = EnvAgent::new();
    let out = engine.run(tape, &mut agent).await.unwrap();
    engine.shutdown().await.unwrap();
    out
}

#[tokio::test(flavor = "current_thread")]
async fn official_resolution_settles_market() {
    let mut events = tape_base();
    // Official YES resolution known AFTER clear_at — settles at its
    // avail (which is later than clear_at*1000).
    events.push(res_ev(
        CID,
        OutcomeSide::Yes,
        CLEAR_AT_S * 1000 + 5_000,
        5,
        CLEAR_AT_S * 1000 + 5_000,
    ));
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    assert_eq!(rt.tape.resolutions.len(), 1);

    let out = run_tape(rt.tape).await;
    assert_eq!(out.labels_applied, 1, "official label applied");
    assert!(
        out.settlement_pending.is_empty(),
        "market settled: {:?}",
        out.settlement_pending
    );
    assert_eq!(out.settlement_corrections, 0);
    assert_eq!(out.fidelity, "certified");
}

#[tokio::test(flavor = "current_thread")]
async fn without_label_stays_pending_never_silent_proxy() {
    // A market that closes while the tape continues without a resolution — it must stay
    // declared pending; the fidelity stays that of the tape (pending
    // ones are declared, not silenced with a proxy).
    let mut events = tape_base();
    // Event later than the close so the settle_due sweep sees it.
    events.push(tick_ev(
        "BTC",
        "51000",
        5,
        CLEAR_AT_S * 1000 + 10_000,
        CLEAR_AT_S * 1000 + 10_000,
    ));
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    let out = run_tape(rt.tape).await;
    assert_eq!(out.labels_applied, 0);
    assert_eq!(
        out.settlement_pending,
        vec![CID.to_string()],
        "without an official label the market stays pending"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn label_with_future_known_at_is_invisible_at_its_avail() {
    // Double causal barrier: the resolution reaches the tape at avail=T1
    // but its `known_at_ms` is T2 > T1 — at T1 it is still invisible and the
    // market does NOT settle until T2.
    let mut events = tape_base();
    let t_arr = CLEAR_AT_S * 1000 + 5_000;
    let t_known = CLEAR_AT_S * 1000 + 20_000;
    events.push(res_ev(CID, OutcomeSide::No, t_known, 5, t_arr));
    // Event later than t_known to cross the visibility threshold.
    events.push(tick_ev("BTC", "51000", 6, t_known + 1_000, t_known + 1_000));
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    let out = run_tape(rt.tape).await;
    assert_eq!(out.labels_applied, 1);
    assert!(
        out.settlement_pending.is_empty(),
        "after t_known the label is visible and settles"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn post_settle_correction_is_declared_and_degrades_fidelity() {
    let mut events = tape_base();
    let t1 = CLEAR_AT_S * 1000 + 5_000;
    // First YES verdict — settles.
    events.push(res_ev(CID, OutcomeSide::Yes, t1, 5, t1));
    // Later official correction: the real winner was NO.
    let t2 = t1 + 10_000;
    events.push(res_ev(CID, OutcomeSide::No, t2, 6, t2));
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    let out = run_tape(rt.tape).await;
    assert_eq!(out.labels_applied, 2, "confirmed + corrected");
    assert_eq!(
        out.settlement_corrections, 1,
        "post-settle correction counted — never a second payout"
    );
    assert_eq!(
        out.fidelity, "reduced",
        "a correction invalidates the certification"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn same_tape_same_seed_same_digest() {
    let events = tape_base();
    let rt1 = ResearchTape::from_events(events.clone(), vec![], 0, &TapeOpts::default());
    let rt2 = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    let d1 = run_tape(rt1.tape).await.digest;
    let d2 = run_tape(rt2.tape).await.digest;
    assert_eq!(d1, d2, "same tape ⇒ same economic ledger");
}

// ── File ingestion (recorder layout) ─────────────────────────────────

#[test]
fn sorted_recursive_directory_load() {
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("polymarket_ws");
    std::fs::create_dir_all(&sub).unwrap();
    let line = |e: ResearchEvent| {
        serde_json::to_string(&RecordedLine {
            event: e,
            stream: Some("book".into()),
            generation: Some(0),
            source_seq: None,
        })
        .unwrap()
    };
    // Two files of the recorder's real layout; the load order is by
    // path (stable) and the causal order is imposed by (avail, local_seq).
    let mut events = tape_base();
    events.reverse(); // written in reverse in the second file
    let half = events.split_off(2);
    std::fs::write(
        sub.join("20260923.jsonl"),
        events
            .iter()
            .map(|e| line(e.clone()))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();
    std::fs::write(
        sub.join("20260924.jsonl"),
        half.iter()
            .map(|e| line(e.clone()))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();

    let rt = ResearchTape::load(dir.path(), &TapeOpts::default()).unwrap();
    assert_eq!(rt.events, 4);
    assert_eq!(rt.malformed_lines, 0);
    assert_eq!(rt.tape.books.len(), 1);
    // The oracle spot reached the tape.
    assert_eq!(rt.tape.spots.len(), 1);
    assert_eq!(rt.tape.spots[0].asset, "BTC");
    assert_eq!(rt.tape.spots[0].price, dec!(50100.5));
}

// ── Fidelity by scope + the contract's `quality` report ────────────────

#[test]
fn quality_by_token_and_cid_feeds_scope() {
    // YES delta BEFORE its snapshot: discard attributed to the CID via the
    // payload's `market` and a gapped token in scope (YES belongs to the
    // observed meta).
    let events = vec![
        meta_ev(CID, 1, T0),
        delta_ev(YES, "0.50", "800", "BUY", 2, T0 + 10, T0 + 10),
        book_ev(YES, "0.49", "0.51", "1000", 3, T0 + 20, T0 + 20),
        book_ev(NO, "0.48", "0.50", "1000", 4, T0 + 30, T0 + 40),
    ];
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    assert_eq!(rt.discarded_deltas, 1);
    let c = &rt.tape.quality_stats;
    assert_eq!(c.discarded_deltas, 1);
    assert_eq!(c.discarded_deltas_by_cid.get(CID), Some(&1));
    assert_eq!(
        c.deltas_by_cid.get(CID),
        Some(&1),
        "delta entry counted"
    );
    assert_eq!(c.discarded_deltas_by_token.get(YES), Some(&1));
    assert!(c.tokens_with_gap.contains(YES), "gapped token in scope");
    assert!(
        c.meta_cids.contains(CID),
        "the observed meta fixes the scope"
    );
    assert_eq!(c.token2cid.get(YES).map(String::as_str), Some(CID));
    // Scope summary: coverage 0% (the only in-scope delta was
    // lost) and the gapped market is CID.
    let mut gap_cids = HashSet::new();
    gap_cids.insert(CID.to_string());
    let q = c.summary(1, 0, &gap_cids);
    assert_eq!(q.discarded_deltas, 1);
    assert_eq!(q.discarded_deltas_in_scope, 1);
    assert_eq!(q.tokens_with_gap_in_scope, 1);
    assert_eq!(q.gap_markets_in_scope, vec![CID.to_string()]);
    assert_eq!(q.delta_coverage_pct_in_scope, 0.0);
}

#[tokio::test(flavor = "current_thread")]
async fn clean_tape_certifies_empty_reasons_and_external_unidentified() {
    let mut events = tape_base();
    events.push(res_ev(
        CID,
        OutcomeSide::Yes,
        CLEAR_AT_S * 1000 + 5_000,
        5,
        CLEAR_AT_S * 1000 + 5_000,
    ));
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    let out = run_tape(rt.tape).await;
    assert_eq!(out.fidelity, "certified");
    assert!(
        out.fidelity_reasons.is_empty(),
        "certified ⇒ no reasons: {:?}",
        out.fidelity_reasons
    );
    // v2: EXTERNAL execution fidelity is never claimed — constant.
    assert_eq!(out.external_fidelity, "unidentified");
    assert!(out.reconciled(), "exact reconciliation required");
    assert_eq!(out.quality.discarded_deltas_in_scope, 0);
    assert_eq!(out.quality.tokens_with_gap_in_scope, 0);
    // No deltas in the tape: trivial coverage 100 (nothing to lose).
    assert_eq!(out.quality.delta_coverage_pct_in_scope, 100.0);
}

#[tokio::test(flavor = "current_thread")]
async fn degradation_in_scope_degrades_fidelity_with_explicit_reasons() {
    // YES delta without a base (arrives before the snapshot): 1 discard and
    // 1 gapped token, both INSIDE the run's scope.
    let mut events = vec![
        meta_ev(CID, 1, T0),
        delta_ev(YES, "0.50", "800", "BUY", 7, T0 + 10, T0 + 10),
    ];
    events.extend(tape_base()[1..].iter().cloned());
    events.push(res_ev(
        CID,
        OutcomeSide::Yes,
        CLEAR_AT_S * 1000 + 5_000,
        6,
        CLEAR_AT_S * 1000 + 5_000,
    ));
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    let out = run_tape(rt.tape).await;
    assert_eq!(out.fidelity, "reduced", "in-scope degradation degrades");
    // One reason per unmet condition — never silent degradation.
    assert!(
        out.fidelity_reasons
            .iter()
            .any(|m| m == "scope_discarded_deltas=1"),
        "reasons: {:?}",
        out.fidelity_reasons
    );
    assert!(
        out.fidelity_reasons
            .iter()
            .any(|m| m == "tokens_with_gap_scope=1"),
        "reasons: {:?}",
        out.fidelity_reasons
    );
    assert_eq!(out.external_fidelity, "unidentified");
    // The `quality` report declares the global counter AND the in-scope one.
    assert_eq!(out.quality.discarded_deltas, 1);
    assert_eq!(out.quality.discarded_deltas_in_scope, 1);
    assert_eq!(out.quality.tokens_with_gap_in_scope, 1);
    assert_eq!(out.quality.gap_markets_in_scope, vec![CID.to_string()]);
    assert!(out.quality.quality_marks >= 1);
}

#[test]
fn quality_summary_serializes_all_v2_fields() {
    use poly_bot::research_contract::{EnvResponse, QualityScope, QualitySummary};
    let q = QualitySummary {
        quality_marks: 3,
        malformed: 1,
        agent: 2,
        discarded_deltas: 10,
        discarded_deltas_in_scope: 4,
        missing_half: 5,
        books_top_equal: 6,
        resyncs: 1,
        tokens_with_gap_in_scope: 2,
        gap_markets_in_scope: vec![CID.to_string()],
        delta_coverage_pct_in_scope: 60.0,
    };
    let v = serde_json::to_value(&q).unwrap();
    for k in [
        "quality_marks",
        "malformed",
        "agent",
        "discarded_deltas",
        "discarded_deltas_in_scope",
        "missing_half",
        "books_top_equal",
        "resyncs",
        "tokens_with_gap_in_scope",
        "gap_markets_in_scope",
        "delta_coverage_pct_in_scope",
    ] {
        assert!(v.get(k).is_some(), "missing key {k} in quality");
    }
    let back: QualitySummary = serde_json::from_value(v).unwrap();
    assert_eq!(back, q, "quality summary round-trip");

    // Hello v2: quality + quality_scope travel in the handshake.
    let h = EnvResponse::Hello {
        schema_version: SCHEMA_VERSION,
        fidelity: "reduced".to_string(),
        quality: q.clone(),
        quality_scope: QualityScope {
            market_cid: Some(CID.to_string()),
            markets: vec![CID.to_string()],
            window_ms: Some([T0, T0 + 60_000]),
        },
    };
    let hv = serde_json::to_value(&h).unwrap();
    assert_eq!(hv["type"], "hello");
    assert_eq!(hv["quality"]["discarded_deltas_in_scope"], 4);
    assert_eq!(hv["quality_scope"]["market_cid"], CID);
    assert_eq!(hv["quality_scope"]["markets"], serde_json::json!([CID]));
    // Read compat: a hello without quality decodes (default).
    let legacy: EnvResponse =
        serde_json::from_str(r#"{"type":"hello","schema_version":1,"fidelity":"reduced"}"#)
            .unwrap();
    assert!(matches!(legacy, EnvResponse::Hello { .. }));
}
