//! Verifiable labels and resolution rules
//! (see `crate::research::contract`).
//!
//! Resolution pipeline per market, with two separate states:
//!
//! - **Provisional** (`ResolutionStatus::Provisional`): verdict derived from
//!   evidence NOT official — the last Binance spot vs. strike (the old
//!   `backtest::settle_due` proxy, [`PROVENANCE_PROXY_SPOT`]) or the
//!   deterministic verdict of `post_close` ([`PROVENANCE_PROXY_POST_CLOSE`]).
//!   Useful for provisional accounting and diagnostics, but it stays
//!   **excluded from certified evaluation**: it never produces a
//!   [`MarketLabel`].
//! - **Confirmed** (`ResolutionStatus::Confirmed`): official result with
//!   identifiable provenance (`MarketOutcome::provenance`, non-empty; checked in
//!   `MarketOutcome::new` and again in `insert_resolution`). It is the
//!   only path to a label, via [`confirmed_label`], which also filters by
//!   `known_at_ms <= as_of`, so the label never enters features
//!   available before the result is known.
//! - **Corrected** (`ResolutionStatus::Superseded`): a correction of
//!   resolution inserts a new `confirmed` with `supersedes_id`, and the
//!   earlier one becomes `superseded` and stops producing a label.
//!
//! Rules: there is NO universal rule; each market registers its own in
//! `resolution_rules` (tag + `rule_version` + source + strike + window).
//! [`LabelRule::Twap`] stores the window bounds and is evaluated over the
//! observations persisted in `resolution_observations`; a rule
//! `External` is never evaluable locally; it requires official evidence.
//!
//! Persistence in `state.rs` (`resolution_rules`, `resolution_observations`,
//! `market_resolutions`). All scans decode strictly: a
//! value outside the vocabulary (`status`/`winner`/`source` unknown,
//! empty `provenance`) aborts with an error, never a partial Ok that would make
//! the corruption invisible.

use anyhow::{bail, Context};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::str::FromStr;

use crate::research::contract::{
    EventSource, MarketLabel, MarketMeta, MarketOutcome, OutcomeSide, ResolutionRule,
    SCHEMA_VERSION,
};
use crate::state::StateStore;

/// Version of the label rules. Bumped when the evaluation semantics change;
/// it is persisted with each registered rule (`rule_version`) so that a
/// historical sample declares under which rule it was evaluated.
pub const RULE_VERSION: u32 = 1;

/// Provenance of the old backtest proxy (last Binance spot vs
/// strike). Every record with this provenance is PROVISIONAL by
/// definition; it is excluded from certified evaluation.
pub const PROVENANCE_PROXY_SPOT: &str = "proxy:binance_spot_close";

/// Provenance of the deterministic verdict of `post_close` (spot
/// frozen pre-close vs `reference_price`). Same status: proxy.
pub const PROVENANCE_PROXY_POST_CLOSE: &str = "proxy:post_close_verdict";

// ── Vocabulary ─────────────────────────────────────────────────────────────

/// Status of a resolution record in `market_resolutions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionStatus {
    /// Derived from non-official evidence (spot proxy). Excluded from the
    /// certified evaluation; it never produces a `MarketLabel`.
    Provisional,
    /// Confirmed by identifiable official evidence (non-empty provenance).
    Confirmed,
    /// Corrected by a later `confirmed` record (`supersedes_id` in
    /// the new row links to this one).
    Superseded,
}

impl ResolutionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Provisional => "provisional",
            Self::Confirmed => "confirmed",
            Self::Superseded => "superseded",
        }
    }

    /// Strict decoder: any string outside the vocabulary is
    /// corruption, not a silent default.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "provisional" => Ok(Self::Provisional),
            "confirmed" => Ok(Self::Confirmed),
            "superseded" => Ok(Self::Superseded),
            other => bail!("unknown resolution status: {other:?}"),
        }
    }
}

/// Durable record of a resolution (a row of `market_resolutions`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionRecord {
    /// `None` before persisting; `Some` on rows read back.
    pub id: Option<i64>,
    pub market_cid: String,
    pub winner: OutcomeSide,
    pub status: ResolutionStatus,
    /// Source of the evidence (oracle for proxies, venue for official results).
    pub source: EventSource,
    /// Identifiable provenance; never empty.
    pub provenance: String,
    /// Instant at which the result became known LOCALLY (ms epoch). It is the
    /// axis of the `as_of` filter of [`confirmed_label`]: a label does not exist
    /// cutoff for decisions made before this instant.
    pub known_at_ms: i64,
    /// Rule applied + version (`LabelRule::tag`), if any.
    pub rule_tag: Option<String>,
    /// For corrections: the `id` of the confirmed record that this one replaces.
    pub supersedes_id: Option<i64>,
    /// JSON detail of the evidence (discrepancy with the proxy, raw payload…).
    pub detail: Option<String>,
    /// Local persistence timestamp (ms epoch).
    pub recorded_ts_ms: i64,
}

/// Effective resolution rule for the label pipeline. Extends the
/// `ResolutionRule` of the contract (frozen) with the aggregation windows
/// that the contract does not type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum LabelRule {
    /// Up/Down by a single spot at close vs `reference_price`: the rule of
    /// the brackets the bot trades.
    UpDownReference,
    /// Mean of the observations within `[window_from_ms, window_to_ms]`
    /// vs `reference_price`. The bounds are persisted here (if the
    /// rule requires TWAP, store the observations and window bounds).
    Twap {
        window_from_ms: i64,
        window_to_ms: i64,
    },
    /// Rule not evaluable locally (UMA vote, free text…): the label
    /// ALWAYS requires official evidence — there is no provisional one.
    External { description: String },
}

impl LabelRule {
    /// Persisted tag: `"{kind}:v{RULE_VERSION}"` — each record of
    /// resolution and each registered rule declare under which rule+version
    /// were evaluated.
    pub fn tag(&self) -> String {
        let kind = match self {
            Self::UpDownReference => "up_down_reference",
            Self::Twap { .. } => "twap",
            Self::External { .. } => "external",
        };
        format!("{kind}:v{RULE_VERSION}")
    }

    /// Derives the effective rule from the contract metadata: `Other(..)` of the
    /// contract is always `External` here: not evaluable, official only.
    pub fn from_meta(meta: &MarketMeta) -> Self {
        match &meta.rule {
            ResolutionRule::UpDownReference => Self::UpDownReference,
            ResolutionRule::Other(t) => Self::External {
                description: t.clone(),
            },
        }
    }
}

/// Record of a market's resolution rule (a row of
/// `resolution_rules`): which rule applies, with which version, which source of
/// truth and which strike.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleRegistration {
    pub market_cid: String,
    pub slug: String,
    pub rule: LabelRule,
    pub rule_version: u32,
    /// Source of truth for the resolution (which feed decides the outcome).
    pub resolution_source: EventSource,
    /// Concrete reference inside the source (symbol, doc, feed).
    pub resolution_source_ref: Option<String>,
    pub reference_price: Option<Decimal>,
    pub market_open_ts: Option<i64>,
    pub clear_at: Option<i64>,
}

impl RuleRegistration {
    /// Builds the record from the contract metadata with the
    /// given effective rule (it may come from `LabelRule::from_meta` or be a TWAP
    /// declared by whoever registers it).
    pub fn from_meta(meta: &MarketMeta, rule: LabelRule) -> Self {
        Self {
            market_cid: meta.market_cid.clone(),
            slug: meta.slug.clone(),
            rule,
            rule_version: RULE_VERSION,
            resolution_source: meta.resolution_source,
            resolution_source_ref: meta.resolution_source_ref.clone(),
            reference_price: meta.reference_price,
            market_open_ts: Some(meta.market_open_ts),
            clear_at: Some(meta.clear_at),
        }
    }
}

/// Price observation for the resolution window (a row of
/// `resolution_observations`) — raw evidence, not deduplicated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriceObservation {
    /// Sample timestamp (ms epoch): whether it falls inside or outside the window is
    /// decided by the rule, not by the writer.
    pub ts_ms: i64,
    pub price: Decimal,
    pub source: EventSource,
}

/// Result of applying official evidence over the history of a market.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconcileOutcome {
    /// No applicable result; the market stays pending.
    Pending,
    /// Confirmed. `discrepancy=true` if a live provisional verdict said the
    /// opposite (recorded in the `detail` of the confirmed record).
    Confirmed { discrepancy: bool },
    /// Resolution correction: the previous confirmed record became `superseded`.
    Corrected {
        previous: OutcomeSide,
        new: OutcomeSide,
    },
}

/// Outcome of an attempt to fetch the official result. Only `Resolved`
/// writes to `market_resolutions`; `Pending` and `Inaccessible` leave the
/// markets stay pending, and they never invent a winner.
#[derive(Debug, Clone, PartialEq)]
pub enum FetchOutcome {
    /// The source returned an official winner.
    Resolved(MarketOutcome),
    /// OK response, but the market is still unresolved.
    Pending,
    /// Unreachable/invalid response (HTTP error, unparseable payload) —
    /// retried later; there is NO result.
    Inaccessible(String),
}

// ── Strict vocabulary encoding ────────────────────────────────────

/// `OutcomeSide` → tag DB ('yes'/'no').
pub fn side_tag(side: OutcomeSide) -> &'static str {
    match side {
        OutcomeSide::Yes => "yes",
        OutcomeSide::No => "no",
    }
}

/// Strict decoder for the winning side.
fn parse_side(s: &str) -> anyhow::Result<OutcomeSide> {
    match s {
        "yes" => Ok(OutcomeSide::Yes),
        "no" => Ok(OutcomeSide::No),
        other => bail!("unknown winner: {other:?}"),
    }
}

/// `EventSource` → DB tag (snake_case serde, no quotes).
pub fn source_tag(source: EventSource) -> String {
    serde_json::to_value(source)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{source:?}"))
}

/// Strict source decoder — via serde so that a new variant of the
/// contract does not require touching this table.
fn parse_source(s: &str) -> anyhow::Result<EventSource> {
    serde_json::from_value(serde_json::Value::String(s.to_string()))
        .with_context(|| format!("unknown resolution source: {s:?}"))
}

// ── Strict row decoders (called from `state.rs`) ─────────────────

/// Decodes a row of `market_resolutions`. ANY field outside the
/// vocabulary (status, winner, unknown source; empty provenance) is
/// corruption and aborts the scan: it is never filtered out or tolerated.
pub fn decode_resolution_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<ResolutionRecord> {
    let cid: String = row.try_get("market_cid")?;
    let ctx = |detail: anyhow::Error| {
        detail.context(format!("market_resolutions: corrupt row (cid {cid:?})"))
    };
    let status_raw: String = row.try_get("status")?;
    let winner_raw: String = row.try_get("winner")?;
    let source_raw: String = row.try_get("source")?;
    let provenance: String = row.try_get("provenance")?;
    if provenance.trim().is_empty() {
        bail!("market_resolutions: corrupt row (cid {cid:?}): empty provenance");
    }
    Ok(ResolutionRecord {
        id: Some(row.try_get("id")?),
        market_cid: cid.clone(),
        winner: parse_side(&winner_raw).map_err(ctx)?,
        status: ResolutionStatus::parse(&status_raw).map_err(ctx)?,
        source: parse_source(&source_raw).map_err(ctx)?,
        provenance,
        known_at_ms: row.try_get("known_at_ms")?,
        rule_tag: row.try_get("rule_tag")?,
        supersedes_id: row.try_get("supersedes_id")?,
        detail: row.try_get("detail")?,
        recorded_ts_ms: row.try_get("recorded_ts_ms")?,
    })
}

/// Decodes a row of `resolution_rules` (the `LabelRule` is parsed from the
/// `rule_detail` JSON, not from the tag; the tag is only a readable index).
pub fn decode_rule_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<RuleRegistration> {
    let cid: String = row.try_get("market_cid")?;
    let detail: String = row.try_get("rule_detail")?;
    let rule: LabelRule = serde_json::from_str(&detail)
        .with_context(|| format!("resolution_rules: rule_detail corrupt (cid {cid:?})"))?;
    let source_raw: String = row.try_get("resolution_source")?;
    let reference: Option<String> = row.try_get("reference_price")?;
    Ok(RuleRegistration {
        market_cid: cid.clone(),
        slug: row.try_get("slug")?,
        rule_version: u32::try_from(row.try_get::<i64, _>("rule_version")?).with_context(|| {
            format!("resolution_rules: rule_version out of range (cid {cid:?})")
        })?,
        rule,
        resolution_source: parse_source(&source_raw)
            .with_context(|| format!("resolution_rules: corrupt row (cid {cid:?})"))?,
        resolution_source_ref: row.try_get("resolution_source_ref")?,
        reference_price: reference
            .as_deref()
            .map(Decimal::from_str)
            .transpose()
            .with_context(|| format!("resolution_rules: reference_price corrupt (cid {cid:?})"))?,
        market_open_ts: row.try_get("market_open_ts")?,
        clear_at: row.try_get("clear_at")?,
    })
}

/// Decodes a row of `resolution_observations` (price TEXT→Decimal,
/// strict source).
pub fn decode_observation_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<PriceObservation> {
    let price_raw: String = row.try_get("price")?;
    let source_raw: String = row.try_get("source")?;
    Ok(PriceObservation {
        ts_ms: row.try_get("ts_ms")?,
        price: Decimal::from_str(&price_raw)
            .with_context(|| format!("resolution_observations: price corrupt {price_raw:?}"))?,
        source: parse_source(&source_raw)
            .with_context(|| format!("resolution_observations: corrupt row"))?,
    })
}

// ── Provisional evaluation (pure, no I/O) ──────────────────────────────────

/// Provisional up/down verdict: `spot > strike` → YES. Same criterion as
/// the old `backtest` settlement; this is the ONLY definition of the
/// proxy, so that backtest, post_close and reconciliation do not diverge.
/// `None` if the reference is missing (or ≤0) or the spot is missing; the provisional
/// nor is one invented.
pub fn provisional_updown(
    reference: Option<Decimal>,
    spot: Option<Decimal>,
) -> Option<OutcomeSide> {
    match (reference, spot) {
        (Some(r), Some(s)) if r > Decimal::ZERO => Some(if s > r {
            OutcomeSide::Yes
        } else {
            OutcomeSide::No
        }),
        _ => None,
    }
}

/// Evaluates the market's rule against the available observations.
/// Returns the provisional verdict, or `None` if the rule cannot be evaluated
/// with that evidence; it never invents a winner:
/// - `UpDownReference`: last observation by `ts_ms` vs strike.
/// - `Twap`: mean of the observations INSIDE the window vs strike;
///   empty window → `None`.
/// - `External`: always `None`; only official evidence.
pub fn evaluate_rule(
    rule: &LabelRule,
    reference_price: Option<Decimal>,
    observations: &[PriceObservation],
) -> Option<OutcomeSide> {
    match rule {
        LabelRule::UpDownReference => {
            let last = observations.iter().max_by_key(|o| o.ts_ms)?;
            provisional_updown(reference_price, Some(last.price))
        }
        LabelRule::Twap {
            window_from_ms,
            window_to_ms,
        } => {
            let mut sum = Decimal::ZERO;
            let mut n = 0u64;
            for o in observations {
                if o.ts_ms >= *window_from_ms && o.ts_ms <= *window_to_ms {
                    sum += o.price;
                    n += 1;
                }
            }
            if n == 0 {
                return None;
            }
            provisional_updown(reference_price, Some(sum / Decimal::from(n)))
        }
        LabelRule::External { .. } => None,
    }
}

// ── Orchestration over the store ─────────────────────────────────────────────

/// Registers the resolution rule of a market. Idempotent: if the rule
/// registered is identical, the row is not rewritten.
pub async fn register_rule(
    state: &StateStore,
    reg: &RuleRegistration,
    now_ms: i64,
) -> anyhow::Result<()> {
    if let Some(current) = state.resolution_rule(&reg.market_cid).await? {
        if current == *reg {
            return Ok(());
        }
    }
    state.upsert_resolution_rule(reg, now_ms).await
}

/// Rule registered for a market (strict decoder via `state`).
pub async fn rule_for(
    state: &StateStore,
    market_cid: &str,
) -> anyhow::Result<Option<RuleRegistration>> {
    state.resolution_rule(market_cid).await
}

/// Persists a price observation (evidence for window-based rules).
pub async fn record_price_observation(
    state: &StateStore,
    market_cid: &str,
    obs: &PriceObservation,
    recv_ts_ms: i64,
) -> anyhow::Result<()> {
    state
        .record_resolution_observation(market_cid, obs, recv_ts_ms)
        .await
}

/// Full history of the market's resolutions (strict decoder — see
/// `StateStore::resolutions_for`).
pub async fn resolutions(
    state: &StateStore,
    market_cid: &str,
) -> anyhow::Result<Vec<ResolutionRecord>> {
    state.resolutions_for(market_cid).await
}

/// Records a PROVISIONAL verdict (proxy). Returns `true` if it inserted.
///
/// - Never overwrites a live `confirmed` record → `false`.
/// - Idempotent: if the latest live record is provisional with the same
///   winner+provenance → `false` (an observer sampling in a loop does not
///   fill the duplicates table).
/// - empty `provenance` → error (the proxy must also declare where it comes from).
pub async fn apply_provisional(
    state: &StateStore,
    market_cid: &str,
    source: EventSource,
    winner: OutcomeSide,
    provenance: &str,
    known_at_ms: i64,
    rule_tag: Option<String>,
) -> anyhow::Result<bool> {
    if provenance.trim().is_empty() {
        bail!("apply_provisional: empty provenance (cid {market_cid})");
    }
    let recs = state.resolutions_for(market_cid).await?;
    if recs.iter().any(|r| r.status == ResolutionStatus::Confirmed) {
        return Ok(false);
    }
    let current = recs
        .iter()
        .rev()
        .find(|r| r.status != ResolutionStatus::Superseded);
    if let Some(last) = current {
        if last.status == ResolutionStatus::Provisional
            && last.winner == winner
            && last.provenance == provenance
        {
            return Ok(false);
        }
    }
    let rec = ResolutionRecord {
        id: None,
        market_cid: market_cid.to_string(),
        winner,
        status: ResolutionStatus::Provisional,
        source,
        provenance: provenance.to_string(),
        known_at_ms,
        rule_tag,
        supersedes_id: None,
        detail: None,
        recorded_ts_ms: crate::time::now().timestamp_millis(),
    };
    state.insert_resolution(&rec).await?;
    Ok(true)
}

/// Applies the market's OFFICIAL result (the only path to a label). The
/// `MarketOutcome` is already validated (non-empty provenance, by
/// `MarketOutcome::new`).
///
/// - Live confirmed record with the same winner → idempotent, no duplicate.
/// - Live confirmed record with a different winner → correction: the new row is
///   `confirmed` with `supersedes_id` → the previous one becomes `superseded`.
/// - Live provisional record that disagrees → `discrepancy=true` and the detail of
///   the divergence is recorded in the `detail` of the confirmed record.
pub async fn apply_official(
    state: &StateStore,
    market_cid: &str,
    source: EventSource,
    outcome: &MarketOutcome,
    rule_tag: Option<String>,
    detail: Option<String>,
) -> anyhow::Result<ReconcileOutcome> {
    let recs = state.resolutions_for(market_cid).await?;
    let current = recs
        .iter()
        .rev()
        .find(|r| r.status != ResolutionStatus::Superseded);

    if let Some(cur) = current.filter(|r| r.status == ResolutionStatus::Confirmed) {
        if cur.winner == outcome.winner {
            // Idempotent: re-applying the same result does not duplicate it.
            return Ok(ReconcileOutcome::Confirmed { discrepancy: false });
        }
        let prev_id = cur.id;
        let new = ResolutionRecord {
            id: None,
            market_cid: market_cid.to_string(),
            winner: outcome.winner,
            status: ResolutionStatus::Confirmed,
            source,
            provenance: outcome.provenance.clone(),
            known_at_ms: outcome.known_at_ms,
            rule_tag,
            supersedes_id: prev_id,
            detail: detail.or_else(|| {
                Some(
                    serde_json::json!({
                        "correction": {
                            "earlier": side_tag(cur.winner),
                            "previous_provenance": cur.provenance,
                        }
                    })
                    .to_string(),
                )
            }),
            recorded_ts_ms: crate::time::now().timestamp_millis(),
        };
        // Insert + mark in ONE transaction: a half-way failure would leave two
        // live `confirmed` records and break the invariant.
        state
            .insert_resolution_superseding(&new, prev_id.expect("previous row always carries an id"))
            .await?;
        return Ok(ReconcileOutcome::Corrected {
            previous: cur.winner,
            new: outcome.winner,
        });
    }

    let discrepancy = current
        .is_some_and(|r| r.status == ResolutionStatus::Provisional && r.winner != outcome.winner);
    let det = if discrepancy {
        let prov = current.expect("discrepancy implies a live provisional record");
        Some(
            serde_json::json!({
                "discrepancy": {
                    "provisional": side_tag(prov.winner),
                    "provisional_provenance": prov.provenance,
                    "official": side_tag(outcome.winner),
                }
            })
            .to_string(),
        )
    } else {
        detail
    };
    let rec = ResolutionRecord {
        id: None,
        market_cid: market_cid.to_string(),
        winner: outcome.winner,
        status: ResolutionStatus::Confirmed,
        source,
        provenance: outcome.provenance.clone(),
        known_at_ms: outcome.known_at_ms,
        rule_tag,
        supersedes_id: None,
        detail: det,
        recorded_ts_ms: crate::time::now().timestamp_millis(),
    };
    state.insert_resolution(&rec).await?;
    Ok(ReconcileOutcome::Confirmed { discrepancy })
}

/// Applies the result of a fetch of the official result. Only
/// `FetchOutcome::Resolved` writes: `Pending` and `Inaccessible` leave the
/// pending market; a winner is never fabricated from a fetch failure
/// network or an unintelligible payload.
pub async fn apply_fetch(
    state: &StateStore,
    market_cid: &str,
    source: EventSource,
    rule_tag: Option<String>,
    fetch: FetchOutcome,
) -> anyhow::Result<ReconcileOutcome> {
    match fetch {
        FetchOutcome::Resolved(outcome) => {
            apply_official(state, market_cid, source, &outcome, rule_tag, None).await
        }
        FetchOutcome::Pending | FetchOutcome::Inaccessible(_) => Ok(ReconcileOutcome::Pending),
    }
}

/// Live confirmed label of a market at instant `as_of_ms`
/// (`None` = no time filter).
///
/// `None` if there is no live `confirmed` with `known_at_ms <= as_of`; the
/// provisional NEVER produces a label, and a label cannot be derived from data
/// known AFTER the decision point (labels stay
/// outside the observable state).
pub async fn confirmed_label(
    state: &StateStore,
    market_cid: &str,
    as_of_ms: Option<i64>,
) -> anyhow::Result<Option<MarketLabel>> {
    let recs = state.resolutions_for(market_cid).await?;
    let rec = recs.iter().rev().find(|r| {
        r.status == ResolutionStatus::Confirmed && as_of_ms.map_or(true, |a| r.known_at_ms <= a)
    });
    Ok(rec.map(|r| MarketLabel {
        schema_version: SCHEMA_VERSION,
        market_cid: market_cid.to_string(),
        winner: r.winner,
        known_at_ms: r.known_at_ms,
        provenance: r.provenance.clone(),
    }))
}
