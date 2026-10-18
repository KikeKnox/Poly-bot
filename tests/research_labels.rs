//! Integration tests of labels and verifiable resolution rules.
//!
//! Cases:
//! - RED: Binance and the official result disagree; pending market;
//!   resolution correction (supersede); different rules per market;
//!   inaccessible response.
//! - GREEN: a market without a result stays pending/excluded — YES/NO is never
//!   invented; labels do not enter features available before the
//!   close (`as_of` filter).
//!

use poly_bot::research::contract::{
    EventSource, MarketMeta, MarketOutcome, OutcomeSide, ResolutionRule, SCHEMA_VERSION,
};
use poly_bot::research::labels::{
    self, FetchOutcome, LabelRule, PriceObservation, ReconcileOutcome, ResolutionStatus,
    RuleRegistration, PROVENANCE_PROXY_POST_CLOSE, PROVENANCE_PROXY_SPOT,
};
use poly_bot::state::StateStore;
use poly_bot::types::Mode;

// ── helpers ────────────────────────────────────────────────────────────────

fn meta_updown(cid: &str, outcome: Option<MarketOutcome>) -> MarketMeta {
    MarketMeta {
        schema_version: SCHEMA_VERSION,
        market_cid: cid.to_string(),
        slug: "btc-updown-5m-test".to_string(),
        yes_token_id: "yes-tok".to_string(),
        no_token_id: "no-tok".to_string(),
        asset: Some("BTC".to_string()),
        neg_risk: false,
        market_open_ts: 1_000,
        clear_at: 1_300,
        reference_price: Some("50000".parse().unwrap()),
        rule: ResolutionRule::UpDownReference,
        resolution_source: EventSource::PolymarketRest,
        resolution_source_ref: Some("gamma:btc-updown-5m-test".to_string()),
        rules_windows: vec![],
        outcome,
    }
}

fn official(winner: OutcomeSide, known_at_ms: i64, provenance: &str) -> MarketOutcome {
    MarketOutcome::new(winner, known_at_ms, provenance).unwrap()
}

fn obs(ts_ms: i64, price: &str) -> PriceObservation {
    PriceObservation {
        ts_ms,
        price: price.parse().unwrap(),
        source: EventSource::BinanceWs,
    }
}

// ── RED: proxy-vs-official discrepancy (Binance vs official result) ─────────────────────

/// The Binance spot verdict says YES, the official result says NO:
/// the confirmed label is the OFFICIAL one and the discrepancy is recorded in the
/// detail of the confirmed record.
#[tokio::test]
async fn proxy_vs_official_discrepancy_official_wins() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let cid = "cid-disc";

    // Proxy (Binance spot > strike) → provisional YES.
    assert!(labels::apply_provisional(
        &state,
        cid,
        EventSource::BinanceWs,
        OutcomeSide::Yes,
        PROVENANCE_PROXY_SPOT,
        1_300_100,
        Some(LabelRule::UpDownReference.tag()),
    )
    .await
    .unwrap());

    // The official result disagrees → confirmed NO.
    let rec = labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::No, 1_305_000, "gamma:resolution-cid-disc"),
        Some(LabelRule::UpDownReference.tag()),
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        rec,
        ReconcileOutcome::Confirmed { discrepancy: true },
        "the proxy vs official discrepancy must be flagged"
    );

    let label = labels::confirmed_label(&state, cid, None)
        .await
        .unwrap()
        .expect("there is a confirmed result");
    assert_eq!(
        label.winner,
        OutcomeSide::No,
        "the official result wins, not the proxy"
    );
    assert_eq!(label.provenance, "gamma:resolution-cid-disc");

    // The provisional stays in the history as proxy (provisional status,
    // recognizable provenance) — never promoted to a label.
    let recs = labels::resolutions(&state, cid).await.unwrap();
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].status, ResolutionStatus::Provisional);
    assert_eq!(recs[0].provenance, PROVENANCE_PROXY_SPOT);
    assert_eq!(recs[1].status, ResolutionStatus::Confirmed);
}

// ── RED/GREEN: pending market ─────────────────────────────────────────────

/// Without an official result there is no label — neither provisional nor anything. The provisional
/// exists for accounting but NEVER produces a `MarketLabel` (it is excluded
/// from the certified evaluation).
#[tokio::test]
async fn market_without_result_stays_pending() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let cid = "cid-pend";

    // Not even a provisional: nothing recorded → no label.
    assert!(labels::confirmed_label(&state, cid, None)
        .await
        .unwrap()
        .is_none());

    // With a provisional: still no confirmed label.
    labels::apply_provisional(
        &state,
        cid,
        EventSource::BinanceWs,
        OutcomeSide::Yes,
        PROVENANCE_PROXY_POST_CLOSE,
        1_300_100,
        None,
    )
    .await
    .unwrap();
    assert!(
        labels::confirmed_label(&state, cid, None)
            .await
            .unwrap()
            .is_none(),
        "a provisional NEVER produces a confirmed label"
    );

    // The contract also does not allow fabricating one from meta without an outcome.
    let meta = meta_updown(cid, None);
    assert!(poly_bot::research::contract::MarketLabel::from_meta(&meta).is_none());
}

// ── RED: resolution correction (supersede) ────────────────────────────────

/// The official source corrects its result: the first confirmed one becomes
/// `superseded` and the current label is the correction's. The history
/// keeps both records (audit).
#[tokio::test]
async fn resolution_correction_supersedes() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let cid = "cid-corr";

    labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::Yes, 1_400_000, "gamma:res-v1"),
        None,
        None,
    )
    .await
    .unwrap();

    let rec = labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::No, 1_500_000, "gamma:res-v2-correction"),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        rec,
        ReconcileOutcome::Corrected {
            previous: OutcomeSide::Yes,
            new: OutcomeSide::No,
        }
    );

    let recs = labels::resolutions(&state, cid).await.unwrap();
    assert_eq!(recs.len(), 2, "the history keeps both");
    assert_eq!(recs[0].status, ResolutionStatus::Superseded);
    assert_eq!(recs[1].status, ResolutionStatus::Confirmed);
    assert_eq!(
        recs[0].winner,
        OutcomeSide::Yes,
        "the corrected record keeps its original winner"
    );

    let label = labels::confirmed_label(&state, cid, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(label.winner, OutcomeSide::No);
    assert_eq!(label.known_at_ms, 1_500_000);

    // Idempotent: reapplying the same confirmed record neither duplicates nor corrects.
    let rec = labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::No, 1_500_000, "gamma:res-v2-correction"),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(rec, ReconcileOutcome::Confirmed { discrepancy: false });
    assert_eq!(
        labels::resolutions(&state, cid)
            .await
            .unwrap()
            .len(),
        2
    );
}

// ── RED: different rules per market ───────────────────────────────────────

/// There is no universal rule: each market records its own (tag + version +
/// source + reference) and provisional evaluation only proceeds where the
/// rule is locally evaluable with the available evidence.
#[tokio::test]
async fn different_rules_per_market() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let now = 1_700_000i64;

    // Market A: up/down by reference — evaluable with a spot at the close.
    let meta_a = meta_updown("cid-a", None);
    labels::register_rule(
        &state,
        &RuleRegistration::from_meta(&meta_a, LabelRule::UpDownReference),
        now,
    )
    .await
    .unwrap();

    // Market B: external rule (UMA vote) — NOT locally evaluable: neither
    // provisional nor label without official evidence.
    let mut meta_b = meta_updown("cid-b", None);
    meta_b.rule = ResolutionRule::Other("uma_vote".to_string());
    let reg_b = RuleRegistration::from_meta(&meta_b, LabelRule::from_meta(&meta_b));
    labels::register_rule(&state, &reg_b, now)
        .await
        .unwrap();

    // Market C: TWAP over a window — evaluable with observations
    // inside the window, not with a loose spot outside it.
    let meta_c = meta_updown("cid-c", None);
    let twap = LabelRule::Twap {
        window_from_ms: 2_000,
        window_to_ms: 3_000,
    };
    labels::register_rule(
        &state,
        &RuleRegistration::from_meta(&meta_c, twap.clone()),
        now,
    )
    .await
    .unwrap();

    // The three rules persist distinctly (tag + version + source).
    let ra = labels::rule_for(&state, "cid-a")
        .await
        .unwrap()
        .unwrap();
    let rb = labels::rule_for(&state, "cid-b")
        .await
        .unwrap()
        .unwrap();
    let rc = labels::rule_for(&state, "cid-c")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(ra.rule, LabelRule::UpDownReference);
    assert!(matches!(rb.rule, LabelRule::External { .. }));
    assert_eq!(rc.rule, twap);
    assert_eq!(ra.rule_version, labels::RULE_VERSION);
    assert_eq!(rc.rule_version, labels::RULE_VERSION);

    // Evaluation per rule:
    // A) up/down with spot > strike → provisional YES.
    assert_eq!(
        labels::evaluate_rule(
            &LabelRule::UpDownReference,
            meta_a.reference_price,
            &[obs(2_500, "51000")],
        ),
        Some(OutcomeSide::Yes)
    );
    // B) external rule → never locally evaluable.
    assert_eq!(
        labels::evaluate_rule(&rb.rule, meta_b.reference_price, &[obs(2_500, "51000")]),
        None,
        "external rule: no provisional — official evidence only"
    );
    // C) TWAP: only observations INSIDE the window count.
    let obs_c = [
        obs(1_000, "90000"), // outside the window (before)
        obs(2_100, "49000"), // inside
        obs(2_900, "47000"), // inside → mean 48000 < 50000 → NO
        obs(4_000, "99000"), // outside the window (after)
    ];
    assert_eq!(
        labels::evaluate_rule(&twap, meta_c.reference_price, &obs_c),
        Some(OutcomeSide::No),
        "TWAP = mean of the window, not the last spot"
    );
    // TWAP without observations in the window → not determinable (not invented).
    assert_eq!(
        labels::evaluate_rule(&twap, meta_c.reference_price, &[obs(1_000, "90000")]),
        None
    );
}

// ── RED: inaccessible response ───────────────────────────────────────────────

/// The official source does not respond or responds with something unintelligible: the market
/// stays pending — no record is written and no winner is invented.
#[tokio::test]
async fn inaccessible_response_leaves_pending() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let cid = "cid-403";

    let rec = labels::apply_fetch(
        &state,
        cid,
        EventSource::PolymarketRest,
        None,
        FetchOutcome::Inaccessible("HTTP 403 Cloudflare".to_string()),
    )
    .await
    .unwrap();
    assert_eq!(rec, ReconcileOutcome::Pending);
    assert!(labels::resolutions(&state, cid)
        .await
        .unwrap()
        .is_empty());
    assert!(labels::confirmed_label(&state, cid, None)
        .await
        .unwrap()
        .is_none());

    // A source that responds but the market is still unresolved → pending all the same.
    let rec = labels::apply_fetch(
        &state,
        cid,
        EventSource::PolymarketRest,
        None,
        FetchOutcome::Pending,
    )
    .await
    .unwrap();
    assert_eq!(rec, ReconcileOutcome::Pending);

    // When it finally responds with a result, it confirms normally.
    let rec = labels::apply_fetch(
        &state,
        cid,
        EventSource::PolymarketRest,
        None,
        FetchOutcome::Resolved(official(OutcomeSide::Yes, 1_800_000, "gamma:res-403")),
    )
    .await
    .unwrap();
    assert_eq!(rec, ReconcileOutcome::Confirmed { discrepancy: false });
}

// ── GREEN: labels outside the observable state ────────────────────────────────

/// A label cannot be derived from data with `known_at_ms` later than the
/// decision point: `confirmed_label(as_of)` filters by when it was learned
/// locally, not by when the row was persisted.
#[tokio::test]
async fn label_not_available_before_being_known() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let cid = "cid-asof";

    labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::Yes, 2_000_000, "gamma:res-asof"),
        None,
        None,
    )
    .await
    .unwrap();

    // Pre-close decision point (clear_at*1000 = 1_300_000 ms): the
    // result was not known yet → None.
    assert!(
        labels::confirmed_label(&state, cid, Some(1_300_000))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        labels::confirmed_label(&state, cid, Some(1_999_999))
            .await
            .unwrap()
            .is_none()
    );
    // From known_at_ms on the label does exist.
    assert_eq!(
        labels::confirmed_label(&state, cid, Some(2_000_000))
            .await
            .unwrap()
            .map(|l| l.winner),
        Some(OutcomeSide::Yes)
    );
}

// ── GREEN: provisional never overrides confirmed ─────────────────────────────────

#[tokio::test]
async fn provisional_does_not_overwrite_confirmed() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let cid = "cid-order";

    labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::No, 1_900_000, "gamma:res-order"),
        None,
        None,
    )
    .await
    .unwrap();

    // A late proxy with the opposite verdict inserts NOTHING.
    assert!(!labels::apply_provisional(
        &state,
        cid,
        EventSource::BinanceWs,
        OutcomeSide::Yes,
        PROVENANCE_PROXY_SPOT,
        1_950_000,
        None,
    )
    .await
    .unwrap());
    assert_eq!(
        labels::resolutions(&state, cid)
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        labels::confirmed_label(&state, cid, None)
            .await
            .unwrap()
            .map(|l| l.winner),
        Some(OutcomeSide::No)
    );

    // Idempotent provisional: same winner+provenance does not duplicate rows.
    assert!(labels::apply_provisional(
        &state,
        "cid-idem",
        EventSource::BinanceWs,
        OutcomeSide::Yes,
        PROVENANCE_PROXY_POST_CLOSE,
        1_300_100,
        None,
    )
    .await
    .unwrap());
    assert!(!labels::apply_provisional(
        &state,
        "cid-idem",
        EventSource::BinanceWs,
        OutcomeSide::Yes,
        PROVENANCE_PROXY_POST_CLOSE,
        1_300_200,
        None,
    )
    .await
    .unwrap());
}

// ── Scan-decode-everything: corruption aborts the scan ────────────────────────

/// A row with `status`/`winner` outside the vocabulary = corruption: the scan
/// decodes it and aborts with an error, never returns a partial Ok.
#[tokio::test]
async fn corrupt_row_aborts_scan() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("state.sqlite");
    let state = StateStore::open(db_path.to_str().unwrap(), Mode::Paper)
        .await
        .unwrap();
    let cid = "cid-bogus";

    labels::apply_provisional(
        &state,
        cid,
        EventSource::BinanceWs,
        OutcomeSide::Yes,
        PROVENANCE_PROXY_SPOT,
        1_300_100,
        None,
    )
    .await
    .unwrap();

    // Corruption through a raw connection: the actual file is `state_paper.sqlite`.
    let file = poly_bot::state::StateStore::resolve_path(db_path.to_str().unwrap(), Mode::Paper);
    let pool = sqlx::SqlitePool::connect(&format!("sqlite://{}", file.display()))
        .await
        .unwrap();
    sqlx::query("UPDATE market_resolutions SET status = 'bogus'")
        .execute(&pool)
        .await
        .unwrap();

    let err = labels::resolutions(&state, cid).await.unwrap_err();
    assert!(
        err.to_string().contains("bogus"),
        "the error must point at the corrupt value: {err}"
    );
    // And the label cannot come out of a corrupt scan either.
    assert!(labels::confirmed_label(&state, cid, None)
        .await
        .is_err());
}

// ── Supersede atomicity ─────────────────────────────────────

/// Insert + mark-resolution go in ONE transaction (`BEGIN IMMEDIATE`): if the
/// row to correct is not `confirmed`, EVERYTHING rolls back — neither a new row nor the mark.
/// And a post-supersede invariant: at most ONE `confirmed` row per market
/// even with chained corrections.
#[tokio::test]
async fn supersede_is_atomic_and_invariant_for_a_confirmed() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    let cid = "cid-atomic";

    // There is only a provisional: superseding a NON-confirmed row must
    // fail entirely (rollback), not leave the insert dangling.
    labels::apply_provisional(
        &state,
        cid,
        EventSource::BinanceWs,
        OutcomeSide::Yes,
        PROVENANCE_PROXY_SPOT,
        1_300_100,
        None,
    )
    .await
    .unwrap();
    let prov_id = labels::resolutions(&state, cid).await.unwrap()[0]
        .id
        .unwrap();
    let correction = labels::ResolutionRecord {
        id: None,
        market_cid: cid.to_string(),
        winner: OutcomeSide::No,
        status: ResolutionStatus::Confirmed,
        source: EventSource::PolymarketRest,
        provenance: "gamma:correction".to_string(),
        known_at_ms: 1_400_000,
        rule_tag: None,
        supersedes_id: Some(prov_id),
        detail: None,
        recorded_ts_ms: 1_400_000,
    };
    assert!(
        state
            .insert_resolution_superseding(&correction, prov_id)
            .await
            .is_err(),
        "superseding a provisional row must fail"
    );
    let recs = labels::resolutions(&state, cid).await.unwrap();
    assert_eq!(
        recs.len(),
        1,
        "rollback: neither the new row nor the mark survived"
    );
    assert_eq!(recs[0].status, ResolutionStatus::Provisional);

    // Happy path: two chained corrections → always ≤1 confirmed.
    labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::No, 1_500_000, "gamma:v1"),
        None,
        None,
    )
    .await
    .unwrap();
    labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::Yes, 1_600_000, "gamma:v2"),
        None,
        None,
    )
    .await
    .unwrap();
    labels::apply_official(
        &state,
        cid,
        EventSource::PolymarketRest,
        &official(OutcomeSide::No, 1_700_000, "gamma:v3"),
        None,
        None,
    )
    .await
    .unwrap();
    let recs = labels::resolutions(&state, cid).await.unwrap();
    let confirmed = recs
        .iter()
        .filter(|r| r.status == ResolutionStatus::Confirmed)
        .count();
    assert_eq!(
        confirmed, 1,
        "invariant: at most one current confirmed row per market"
    );
    assert_eq!(
        labels::confirmed_label(&state, cid, None)
            .await
            .unwrap()
            .map(|l| (l.winner, l.provenance)),
        Some((OutcomeSide::No, "gamma:v3".to_string()))
    );
}
