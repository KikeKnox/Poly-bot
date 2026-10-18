//! Tests of the official `resolutions_list` fetcher against sanitized real
//! Gamma fixtures.
//!
//! The fixtures in `tests/fixtures/gamma_market_*.json` derive from a
//! real response of `gamma-api.polymarket.com/markets?slug=btc-updown-5m-…`
//! (captured on 2026-09-23, ids kept — they are public market
//! identifiers, not secrets): the real shape is a JSON array with
//! `clobTokenIds`/`outcomePrices`/`outcomes` as JSON strings serialized
//! inside the JSON, singular `umaResolutionStatus` on resolved markets and
//! plural `umaResolutionStatuses` on grouped/neg-risk ones.
//!
//! Rules exercised (conservative, per the contract):
//! - Only `FetchOutcome::Resolved` writes a confirmed label.
//! - `Pending` and `Inaccessible` leave the market pending, with no event.
//! - The winner is mapped by token id, never by text label.
//! - Winning token outside the yes/no pair → Inaccessible (no `No` by elimination).
//! - `umaResolutionStatus` must be exactly "resolved" and exactly one
//!   `outcomePrices` == 1 — intermediate prices are not a resolution.
//! - `known_at_ms` = real local instant of learning, never `clear_at`.

use poly_bot::research::contract::{EventSource, OutcomeSide};
use poly_bot::research::labels::{self, FetchOutcome, ReconcileOutcome, ResolutionStatus};
use poly_bot::tools::recorders::resolution::parse_resolution_body;
use poly_bot::state::StateStore;
use poly_bot::types::Mode;

const CID: &str = "0xcc719fadd13918ec1eac3d258186caa07a1446f44e76521cdc9fb80cb14df321";
const SLUG: &str = "btc-updown-5m-1790175300";
/// First `clobTokenIds` of the real fixture (outcome "Up" → YES).
const YES_TOK: &str =
    "49355575869915607833628566107383680390001608252605296559614869970709347593818";
/// Second `clobTokenIds` (outcome "Down" → NO).
const NO_TOK: &str =
    "85090347783046098187782235350502474530416457463938134254182332167000529376123";

fn fixture(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("could not read fixture {p}: {e}"))
}

/// Resolved market with a YES winner → `Resolved` with `OutcomeSide::Yes`,
/// non-empty gamma provenance and `known_at_ms` = real local instant (≈now,
/// never the bracket's `clear_at`).
#[test]
fn fixture_resolved_yes() {
    let before_ms = poly_bot::time::now().timestamp_millis();
    let out = parse_resolution_body(
        &fixture("gamma_market_resolved_yes.json"),
        CID,
        SLUG,
        YES_TOK,
        NO_TOK,
    );
    let after_ms = poly_bot::time::now().timestamp_millis();
    match out {
        FetchOutcome::Resolved(o) => {
            assert_eq!(o.winner, OutcomeSide::Yes);
            assert!(!o.provenance.trim().is_empty());
            assert!(o.provenance.contains("gamma-api:markets/slug/"));
            assert!(
                o.known_at_ms >= before_ms && o.known_at_ms <= after_ms,
                "known_at_ms must be the real local instant, not a fixed value or clear_at"
            );
        }
        other => panic!("expected Resolved, got {other:?}"),
    }
}

/// NO winner mapped by token id (second clobTokenId), not by label.
#[test]
fn fixture_resolved_no() {
    match parse_resolution_body(
        &fixture("gamma_market_resolved_no.json"),
        CID,
        SLUG,
        YES_TOK,
        NO_TOK,
    ) {
        FetchOutcome::Resolved(o) => assert_eq!(o.winner, OutcomeSide::No),
        other => panic!("expected Resolved, got {other:?}"),
    }
}

/// Closed (trading ended) but `umaResolutionStatus != "resolved"` →
/// Pending: the market stays pending, there is no winner.
#[test]
fn unresolved_fixture_is_pending() {
    assert_eq!(
        parse_resolution_body(
            &fixture("gamma_market_unresolved.json"),
            CID,
            SLUG,
            YES_TOK,
            NO_TOK
        ),
        FetchOutcome::Pending
    );
}

/// Ambiguous/incomplete response (resolved but without clobTokenIds) →
/// Inaccessible: the winner cannot be mapped by token and is NEVER invented.
#[test]
fn ambiguous_fixture_is_inaccessible() {
    assert!(matches!(
        parse_resolution_body(
            &fixture("gamma_market_ambiguous.json"),
            CID,
            SLUG,
            YES_TOK,
            NO_TOK
        ),
        FetchOutcome::Inaccessible(_)
    ));
}

/// Winning token that does not match the observed yes/no pair (stale DB / non-binary
/// market) → Inaccessible: a `No` by elimination would be an invented label.
#[test]
fn foreign_winner_fixture_is_inaccessible() {
    assert!(matches!(
        parse_resolution_body(
            &fixture("gamma_market_foreign_winner.json"),
            CID,
            SLUG,
            YES_TOK,
            NO_TOK
        ),
        FetchOutcome::Inaccessible(_)
    ));
}

/// `umaResolutionStatus == "resolved"` but intermediate prices
/// (0.998/0.002): not a final resolution yet → Pending, retried.
#[test]
fn non_final_prices_fixture_is_pending() {
    assert_eq!(
        parse_resolution_body(
            &fixture("gamma_market_nonfinal_prices.json"),
            CID,
            SLUG,
            YES_TOK,
            NO_TOK
        ),
        FetchOutcome::Pending
    );
}

/// Gamma returns `[]` (slug purged/market absent) → Inaccessible.
#[test]
fn absent_market_fixture_is_inaccessible() {
    assert!(matches!(
        parse_resolution_body(
            &fixture("gamma_market_empty.json"),
            CID,
            SLUG,
            YES_TOK,
            NO_TOK
        ),
        FetchOutcome::Inaccessible(_)
    ));
}

/// Body that is not even JSON of the expected shape → Inaccessible, no panic.
#[test]
fn corrupt_body_is_inaccessible() {
    assert!(matches!(
        parse_resolution_body(
            b"{\"is_not\": \"un array de markets\"}",
            CID,
            SLUG,
            YES_TOK,
            NO_TOK
        ),
        FetchOutcome::Inaccessible(_)
    ));
    assert!(matches!(
        parse_resolution_body(b"no-json", CID, SLUG, YES_TOK, NO_TOK),
        FetchOutcome::Inaccessible(_)
    ));
}

/// `apply_fetch` only persists `Resolved`: Pending and Inaccessible leave the
/// market without records (GREEN end-to-end over an in-memory StateStore).
#[tokio::test]
async fn apply_fetch_only_persists_resolved() {
    let state = StateStore::open_memory(Mode::Paper).await.unwrap();
    // The cid must match the fixture's `conditionId` — without a match the
    // parse is `Inaccessible` (market not returned), not `Resolved`.
    let cid = CID;

    // Pending → does not write.
    let r = labels::apply_fetch(
        &state,
        cid,
        EventSource::PolymarketRest,
        None,
        FetchOutcome::Pending,
    )
    .await
    .unwrap();
    assert_eq!(r, ReconcileOutcome::Pending);
    assert!(labels::resolutions(&state, cid)
        .await
        .unwrap()
        .is_empty());

    // Inaccessible → does not write.
    let r = labels::apply_fetch(
        &state,
        cid,
        EventSource::PolymarketRest,
        None,
        FetchOutcome::Inaccessible("HTTP 503".into()),
    )
    .await
    .unwrap();
    assert_eq!(r, ReconcileOutcome::Pending);
    assert!(labels::resolutions(&state, cid)
        .await
        .unwrap()
        .is_empty());

    // Resolved (from the real fixture) → confirmed + label available.
    let before_ms = poly_bot::time::now().timestamp_millis();
    let fetch = parse_resolution_body(
        &fixture("gamma_market_resolved_no.json"),
        cid,
        SLUG,
        YES_TOK,
        NO_TOK,
    );
    let r = labels::apply_fetch(&state, cid, EventSource::PolymarketRest, None, fetch)
        .await
        .unwrap();
    assert_eq!(r, ReconcileOutcome::Confirmed { discrepancy: false });

    let recs = labels::resolutions(&state, cid).await.unwrap();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].status, ResolutionStatus::Confirmed);
    assert_eq!(recs[0].winner, OutcomeSide::No);
    assert!(recs[0].known_at_ms >= before_ms, "real local known_at_ms");

    let label = labels::confirmed_label(&state, cid, None)
        .await
        .unwrap()
        .expect("confirmed label after Resolved");
    assert_eq!(label.winner, OutcomeSide::No);
    assert!(label.provenance.contains("gamma-api:markets/slug/"));
}
