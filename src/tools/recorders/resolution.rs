//! Official resolutions fetcher (integration with the research tape).
//! for the research pipeline.
//!
//! `crate::research::labels` defines the provisional/confirmed/superseded vocabulary
//! and the functions `apply_official`/`apply_fetch`, but no producer called them:
//! this module is the producer: a periodic task that sweeps
//! `markets_observed` closed without a `confirmed` label, queries the Gamma
//! API by `slug` and applies the result with `apply_fetch`.
//!
//! Hard rules (inherited from the original plan and review):
//!
//! - **The winner is never invented.** Only `FetchOutcome::Resolved` writes;
//!   `Pending` and `Inaccessible` leave the market pending for the next
//!   sweep. An inaccessible fetch is NOT evidence of anything.
//! - **The winner is mapped by token, not by label.** Gamma's `clobTokenIds`
//!   are compared with `markets_observed.{yes,no}_token_id`; outcome labels
//!   ("Yes"/"Up"/"Above"…) are free text and are not parsed. A winning token that is
//!   neither yes nor no of the observed market is `Inaccessible`, never a `No` by
//!   elimination.
//! - **Conservative in the face of ambiguity**: `umaResolutionStatus ==
//!   "resolved"` and exactly ONE `outcomePrices == 1` (the rest `== 0`).
//!   A `closed` market without a `resolved` status stays `Pending`:
//!   `closed` only indicates the end of trading, not resolution.
//! - **The market's rule is registered if missing** (`UpDownReference`, the
//!   rule of the universe the bot trades) with the real window of
//!   `markets_observed`, via `insert_resolution_rule_if_absent`
//!   (`ON CONFLICT DO NOTHING`), so a rule registered by another path
//!   (post_close, with its real `reference_price`, e.g.) is NEVER overwritten, not even
//!   in a race.
//!
//! Observation task (like `post_close`/`analytics`): it does not emit an
//! `OrderIntent`, does not go through `RiskGuard`, and never touches the executor.

use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::watch;
use tracing::warn;

use crate::config::ResolutionFetchConfig;
use crate::research::contract::{EventKind, EventSource, MarketOutcome, OutcomeSide};
use crate::research::events::EventProducer;
use crate::research::labels::{self, FetchOutcome, LabelRule, ReconcileOutcome, RuleRegistration};
use crate::state::StateStore;

/// Base URL of the Gamma API: the same constant as `market_feed::discover`
/// (its `mod` is private to the package; if the URL changes, both must be updated).
const GAMMA_BASE: &str = "https://gamma-api.polymarket.com";

/// Subset of the Gamma market fields needed to resolve.
/// `clobTokenIds`/`outcomePrices` arrive as SERIALIZED JSON arrays inside a
/// string (`"[\"123\",\"456\"]"`) — same format that `discover.rs` parses.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GammaResolution {
    condition_id: Option<String>,
    slug: Option<String>,
    /// "resolved" once UMA has fixed the outcome (updown crypto resolves via
    /// UMA like the rest of the binary markets). `closed` only marks the
    /// end of trading: it is not queried, resolution requires this status.
    uma_resolution_status: Option<String>,
    outcome_prices: Option<String>,
    clob_token_ids: Option<String>,
}

/// Parses the RAW BODY of the Gamma response (JSON bytes of the endpoint
/// `/markets?slug=…`) into a `FetchOutcome`. Testable entry point with
/// sanitised real fixtures: a body that does not deserialise to the expected shape
/// is `Inaccessible`, never a panic or an invented winner.
pub fn parse_resolution_body(
    body: &[u8],
    cid: &str,
    slug: &str,
    yes_token_id: &str,
    no_token_id: &str,
) -> FetchOutcome {
    let markets: Vec<GammaResolution> = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return FetchOutcome::Inaccessible(format!("body could not be parsed: {e}")),
    };
    parse_resolution(&markets, cid, slug, yes_token_id, no_token_id)
}

/// Parses the Gamma response into a `FetchOutcome`. Pure and testable.
///
/// `Resolved` only if: the returned market matches by `condition_id` (or by
/// `slug` if the cid is absent), `umaResolutionStatus == "resolved"`, and there is
/// EXACTLY one outcome with price `1`: its `clobTokenIds[i]` is compared
/// with `yes_token_id`. Any other combination (closed but unresolved,
/// intermediate prices, corrupt arrays) is `Pending`/`Inaccessible` and
/// writes nothing.
fn parse_resolution(
    body: &[GammaResolution],
    cid: &str,
    slug: &str,
    yes_token_id: &str,
    no_token_id: &str,
) -> FetchOutcome {
    let Some(gm) = body.iter().find(|g| {
        g.condition_id.as_deref() == Some(cid)
            || (g.condition_id.is_none() && g.slug.as_deref() == Some(slug))
    }) else {
        // Successful response but without the requested market: inaccessible in practice
        // (stale slug, market purged from Gamma): it is retried and
        // never resolved falsely.
        return FetchOutcome::Inaccessible(format!(
            "gamma did not return the market (cid {cid}, slug {slug})"
        ));
    };
    if gm.uma_resolution_status.as_deref() != Some("resolved") {
        return FetchOutcome::Pending;
    }
    let (Some(token_ids_raw), Some(prices_raw)) = (&gm.clob_token_ids, &gm.outcome_prices) else {
        return FetchOutcome::Inaccessible(format!(
            "market resolved without clobTokenIds/outcomePrices (cid {cid})"
        ));
    };
    let token_ids: Vec<String> = match serde_json::from_str(token_ids_raw) {
        Ok(v) => v,
        Err(e) => {
            return FetchOutcome::Inaccessible(format!("clobTokenIds could not be parsed: {e}"));
        }
    };
    let prices: Vec<Decimal> = match serde_json::from_str::<Vec<String>>(prices_raw).map(|v| {
        v.iter()
            .map(|p| Decimal::from_str(p))
            .collect::<Result<Vec<_>, _>>()
    }) {
        Ok(Ok(v)) => v,
        _ => {
            return FetchOutcome::Inaccessible(format!("outcomePrices could not be parsed: {prices_raw}"));
        }
    };
    if token_ids.len() != prices.len() {
        return FetchOutcome::Inaccessible(format!(
            "clobTokenIds ({}) and outcomePrices ({}) have different lengths",
            token_ids.len(),
            prices.len()
        ));
    }
    // Exactly one winner with price 1 and the rest 0: a price of 0.998 is
    // not resolution yet.
    let winners: Vec<usize> = prices
        .iter()
        .enumerate()
        .filter(|(_, p)| **p == Decimal::ONE)
        .map(|(i, _)| i)
        .collect();
    if winners.len() != 1
        || prices
            .iter()
            .any(|p| *p != Decimal::ZERO && *p != Decimal::ONE)
    {
        return FetchOutcome::Pending;
    }
    let winner_token = &token_ids[winners[0]];
    let winner = if *winner_token == yes_token_id {
        OutcomeSide::Yes
    } else if *winner_token == no_token_id {
        OutcomeSide::No
    } else {
        // The winner is none of the tokens the bot observed for this
        // market (stale yes_token_id in the DB, non-binary market): confirming
        // a `No` by elimination would invent the label.
        return FetchOutcome::Inaccessible(format!(
            "winning token {winner_token} does not match the observed yes/no (cid {cid})"
        ));
    };
    let now_ms = crate::time::now().timestamp_millis();
    let provenance = format!(
        "gamma-api:markets/slug/{slug} umaStatus=resolved prices={prices_raw} fetched_ms={now_ms}"
    );
    match MarketOutcome::new(winner, now_ms, provenance) {
        Ok(o) => FetchOutcome::Resolved(o),
        Err(e) => FetchOutcome::Inaccessible(format!("invalid outcome: {e}")),
    }
}

/// GET of the Gamma market by slug → `FetchOutcome`. Never panics: any
/// network or parsing failure is `Inaccessible` (retryable, writes nothing).
async fn fetch_gamma(http: &reqwest::Client, slug: &str) -> Result<Vec<GammaResolution>, String> {
    let resp = http
        .get(format!("{GAMMA_BASE}/markets"))
        // `closed=true`: without it Gamma excludes the closed markets from the
        // listing, and the resolution pendings are closed by
        // definition (clear_at expired), so the empty query always returned
        // `[]` and the fetcher could never confirm anything.
        .query(&[("slug", slug), ("closed", "true")])
        .send()
        .await
        .map_err(|e| format!("HTTP error: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<Vec<GammaResolution>>()
        .await
        .map_err(|e| format!("payload could not be parsed: {e}"))
}

/// Registers the default resolution rule if the market has none:
/// `UpDownReference`, the rule of the updown universe the bot trades, with the
/// real window of `markets_observed`. `insert_resolution_rule_if_absent`
/// is `ON CONFLICT DO NOTHING`: if another task (post_close, with its real
/// `reference_price`, e.g.) registered the rule first, even in a race within the same
/// sweep, the existing one wins and is never overwritten.
async fn ensure_rule(state: &StateStore, pend: &crate::state::OfficialPendingMarket, now_ms: i64) {
    let reg = RuleRegistration {
        market_cid: pend.condition_id.clone(),
        slug: pend.slug.clone(),
        rule: LabelRule::UpDownReference,
        rule_version: labels::RULE_VERSION,
        resolution_source: EventSource::PolymarketRest,
        resolution_source_ref: Some(format!("gamma:{}", pend.slug)),
        // The price reference is only known to whoever captured the verdict
        // (post_close): the fetcher does not invent it.
        reference_price: None,
        market_open_ts: pend.market_open_ts,
        clear_at: Some(pend.clear_at),
    };
    if let Err(e) = state.insert_resolution_rule_if_absent(&reg, now_ms).await {
        warn!(
            cid = pend.condition_id.as_str(),
            error = %e,
            "resolution_fetcher: error registering default rule"
        );
    }
}

/// Fetcher task. One sweep every `cfg.poll_secs`; within the sweep the
/// requests are spaced `cfg.min_request_ms` apart (polite rate limit, same criterion
/// as the resync sweeper). DB/HTTP errors are logged and the cycle continues
/// — this task must never die or bring the bot down.
///
/// `main.rs` calls this wrapper; the variant with the research recorder is
/// [`run_with_recorder`]. TODO (integrator): change the `main.rs` call site
/// (`tools::recorders::resolution::run(state.clone(), config.resolution_fetcher.clone(),
/// kill_rx.clone())`) with `run_with_recorder(..., recorder.as_ref().map(|r|
/// `r.producer()))` when the research tape is active, so that the
/// official results also end up on the tape as `EventKind::Resolution`.
pub async fn run(
    state: Arc<StateStore>,
    cfg: ResolutionFetchConfig,
    shutdown: watch::Receiver<bool>,
) {
    run_with_recorder(state, cfg, shutdown, EventProducer::sink()).await
}

/// Same as [`run`] but with `recorder`: every time
/// `apply_fetch` confirms or corrects an official result, an `EventKind::Resolution`
/// event is emitted to the research tape, with `known_at_ms` = the real local instant
/// at which the result was learned (set by
/// `parse_resolution` when parsing the response; NEVER `clear_at`).
/// The producer always exists; with the hub inactive, `record` discards
/// it at no cost.
pub async fn run_with_recorder(
    state: Arc<StateStore>,
    cfg: ResolutionFetchConfig,
    mut shutdown: watch::Receiver<bool>,
    recorder: EventProducer,
) {
    let http = match reqwest::Client::builder()
        .user_agent(crate::POLY_USER_AGENT)
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "resolution_fetcher: could not build the HTTP client; task inert");
            return;
        }
    };
    let mut interval = tokio::time::interval(Duration::from_secs(cfg.poll_secs.max(1)));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    tracing::info!(
        poll_secs = cfg.poll_secs,
        batch = cfg.batch_size,
        "resolution_fetcher: starting"
    );

    loop {
        tokio::select! {
            _ = interval.tick() => {}
            change = shutdown.changed() => {
                // Err = sender dropped without send(true): this is also a shutdown;
                // returning here avoids a hot loop of sweeps if the watch
                // dies by another path.
                if change.is_err() || *shutdown.borrow() {
                    tracing::info!("resolution_fetcher: shutdown received");
                    return;
                }
            }
        }
        if *shutdown.borrow() {
            return;
        }

        // `markets_observed.clear_at` is in SECONDS (same unit as
        // `BookSnapshot::clear_at`/`upsert_market`), so the parameters are
        // passed in seconds to avoid mixing time domains. `min_clear_s` removes
        // from the set the markets older than the horizon: the ones that never
        // resolve cannot re-occupy the batch forever.
        let now_s = crate::time::now().timestamp();
        let min_clear_s = now_s.saturating_sub(cfg.pending_horizon_days as i64 * 86_400);
        let pending = match state
            .markets_pending_official(now_s, min_clear_s, cfg.batch_size as i64)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "resolution_fetcher: error listing pending — retrying on the next sweep");
                continue;
            }
        };
        if pending.is_empty() {
            continue;
        }
        tracing::debug!(pending = pending.len(), "resolution_fetcher: sweep");

        for pend in pending {
            if *shutdown.borrow() {
                return;
            }
            let cid = pend.condition_id.as_str();
            let now_ms = crate::time::now().timestamp_millis();
            ensure_rule(&state, &pend, now_ms).await;

            let fetch = match fetch_gamma(&http, &pend.slug).await {
                Ok(markets) => parse_resolution(
                    &markets,
                    cid,
                    &pend.slug,
                    &pend.yes_token_id,
                    &pend.no_token_id,
                ),
                Err(e) => FetchOutcome::Inaccessible(e),
            };
            // Keep the outcome BEFORE `apply_fetch` consumes it: if persistence
            // confirms, a `Resolution` event is emitted to the tape.
            let resolved_outcome = match &fetch {
                FetchOutcome::Resolved(o) => Some(o.clone()),
                _ => None,
            };
            match labels::apply_fetch(
                &state,
                cid,
                EventSource::PolymarketRest,
                Some(LabelRule::UpDownReference.tag()),
                fetch,
            )
            .await
            {
                Ok(ReconcileOutcome::Confirmed { discrepancy }) => {
                    tracing::info!(
                        cid,
                        slug = %pend.slug,
                        discrepancy,
                        "resolution_fetcher: official label confirmed"
                    );
                    metrics::counter!("hrf_resolution_confirmed_total").increment(1);
                    if let Some(o) = &resolved_outcome {
                        emit_resolution(&recorder, cid, &pend.slug, o, "confirmed");
                    }
                }
                Ok(ReconcileOutcome::Corrected { previous, new }) => {
                    warn!(
                        cid,
                        slug = %pend.slug,
                        previous = labels::side_tag(previous),
                        new = labels::side_tag(new),
                        "resolution_fetcher: official resolution CORRECTED"
                    );
                    metrics::counter!("hrf_resolution_corrected_total").increment(1);
                    if let Some(o) = &resolved_outcome {
                        emit_resolution(&recorder, cid, &pend.slug, o, "corrected");
                    }
                }
                Ok(ReconcileOutcome::Pending) => {}
                Err(e) => {
                    warn!(cid, error = %e, "resolution_fetcher: error applying result");
                }
            }
            tokio::time::sleep(Duration::from_millis(cfg.min_request_ms)).await;
        }
    }
}

/// Emits `EventKind::Resolution` once `apply_fetch` has persisted the result
/// (confirmed or corrected). Only called with a REAL `MarketOutcome`:
/// `Pending`/`Inaccessible` never reach here and never emit an event.
///
/// `payload` is the serialized `MarketOutcome` (winner, local `known_at_ms`,
/// `provenance`) plus a `reconcile` field that the deserializer ignores when
/// reading it back as a `MarketOutcome`. The event carries `recv_ts_ms`/
/// `avail_ts_ms` = local emission instant, consistent with `outcome.known_at_ms`,
/// which `parse_resolution` sets when parsing the response (never `clear_at`,
/// never backdated).
fn emit_resolution(
    prod: &EventProducer,
    cid: &str,
    slug: &str,
    outcome: &MarketOutcome,
    reconcile: &'static str,
) {
    // The producer always exists; without an asset writer the outcome would be
    // serialized only to be discarded.
    if !prod.is_active() {
        return;
    }
    let now_ms = crate::time::now().timestamp_millis();
    let mut payload = serde_json::to_value(outcome).unwrap_or(serde_json::Value::Null);
    if let Some(obj) = payload.as_object_mut() {
        obj.insert(
            "slug".to_string(),
            serde_json::Value::String(slug.to_string()),
        );
        obj.insert(
            "reconcile".to_string(),
            serde_json::Value::String(reconcile.to_string()),
        );
    }
    let mut e = crate::research::events::EventInput::new(
        EventSource::PolymarketRest,
        EventKind::Resolution,
        payload,
        now_ms,
    );
    e.market_cid = Some(cid.to_string());
    e.asset = crate::oracle::asset_from_slug(slug);
    // Dedup by stable content: (cid, winner). The provenance carries a variable
    // `fetched_ms`; including it would make the key unstable and the dedup would
    // never match. A CORRECTION changes the winner -> different key -> it is
    // written (correct); re-emitting the same outcome is deduplicated.
    e.dedup_key = Some(crate::research::events::stable_hash((
        "resolution",
        cid,
        labels::side_tag(outcome.winner),
    )));
    e.stream = Some("resolution_fetcher".to_string());
    prod.record(e);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gamma(cid: &str, slug: &str, status: &str, prices: &str, tokens: &str) -> GammaResolution {
        GammaResolution {
            condition_id: Some(cid.to_string()),
            slug: Some(slug.to_string()),
            uma_resolution_status: Some(status.to_string()),
            outcome_prices: Some(prices.to_string()),
            clob_token_ids: Some(tokens.to_string()),
        }
    }

    #[test]
    fn resolved_maps_winner_by_token() {
        let body = vec![gamma(
            "cid1",
            "s",
            "resolved",
            "[\"1\",\"0\"]",
            "[\"YES_TOK\",\"NO_TOK\"]",
        )];
        match parse_resolution(&body, "cid1", "s", "YES_TOK", "NO_TOK") {
            FetchOutcome::Resolved(o) => {
                assert_eq!(o.winner, OutcomeSide::Yes);
                assert!(o.provenance.contains("gamma-api:markets/slug/s"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn resolved_winner_no() {
        let body = vec![gamma(
            "cid1",
            "s",
            "resolved",
            "[\"0\",\"1\"]",
            "[\"YES_TOK\",\"NO_TOK\"]",
        )];
        match parse_resolution(&body, "cid1", "s", "YES_TOK", "NO_TOK") {
            FetchOutcome::Resolved(o) => assert_eq!(o.winner, OutcomeSide::No),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn winner_outside_yes_no_is_inaccessible() {
        // Winning token that is neither the observed market's yes nor no token
        // (stale DB / non-binary market): never a No by elimination.
        let body = vec![gamma(
            "cid1",
            "s",
            "resolved",
            "[\"1\",\"0\"]",
            "[\"OTHER_TOK\",\"NO_TOK\"]",
        )];
        assert!(matches!(
            parse_resolution(&body, "cid1", "s", "YES_TOK", "NO_TOK"),
            FetchOutcome::Inaccessible(_)
        ));
    }

    #[test]
    fn closed_without_resolved_is_pending() {
        let body = vec![gamma(
            "cid1",
            "s",
            "proposed",
            "[\"1\",\"0\"]",
            "[\"Y\",\"N\"]",
        )];
        assert_eq!(
            parse_resolution(&body, "cid1", "s", "Y", "N"),
            FetchOutcome::Pending
        );
    }

    #[test]
    fn intermediate_prices_do_not_resolve() {
        let body = vec![gamma(
            "cid1",
            "s",
            "resolved",
            "[\"0.998\",\"0.002\"]",
            "[\"Y\",\"N\"]",
        )];
        assert_eq!(
            parse_resolution(&body, "cid1", "s", "Y", "N"),
            FetchOutcome::Pending
        );
    }

    #[test]
    fn absent_market_is_inaccessible() {
        let body = vec![gamma(
            "other",
            "other-slug",
            "resolved",
            "[\"1\",\"0\"]",
            "[\"Y\",\"N\"]",
        )];
        assert!(matches!(
            parse_resolution(&body, "cid1", "s", "Y", "N"),
            FetchOutcome::Inaccessible(_)
        ));
    }

    #[test]
    fn corrupt_arrays_are_inaccessible() {
        let body = vec![gamma("cid1", "s", "resolved", "garbage", "[\"Y\",\"N\"]")];
        assert!(matches!(
            parse_resolution(&body, "cid1", "s", "Y", "N"),
            FetchOutcome::Inaccessible(_)
        ));
    }

    #[test]
    fn matches_by_slug_if_condition_id_missing() {
        let mut g = gamma("cid1", "s", "resolved", "[\"0\",\"1\"]", "[\"Y\",\"N\"]");
        g.condition_id = None;
        let body = vec![g];
        match parse_resolution(&body, "cid1", "s", "Y", "N") {
            FetchOutcome::Resolved(o) => assert_eq!(o.winner, OutcomeSide::No),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }
}
