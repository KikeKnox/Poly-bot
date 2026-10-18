//! Shared research contracts (contracts A–E).
//!
//! Versioned, PURE vocabulary (no I/O, no Tokio, no SQLite) shared by the
//! research pieces built on top of it: event recording, labels, deterministic
//! replay and dataset/episodes.
//!
//! Contract rules:
//! - A: `origin_ts_ms` is never invented (`None` if the source does not provide it), and
//!   nothing is ordered by exchange clock; the canonical order is `local_seq`.
//! - B: the official result and later labels stay OUT of the observable
//!   state; they are never features of an observation.
//! - C: submission, arrival, acceptance, execution, cancellation and notification
//!   are separate events on ONE virtual clock (`sim_now_ms`), using the same
//!   terminology as `order::sim_router` (`SimClock`, `SimExecution`,
//!   `exec_id`, `local_order_id`).
//! - D: never register a future result as a feature; the reward is broken
//!   down (price, fee, risk penalty).
//! - E: the same input reproduces the economic ledger; everything is pinned by hash;
//!   research paths live apart from the active DB.

use std::collections::BTreeMap;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Schema version of the research contracts. Every serializable kind in this
/// file carries it in `schema_version` so consumers can detect data written
/// with another contract version.
pub const SCHEMA_VERSION: u32 = 1;

/// Invariant errors of the contracts (validated constructors).
#[derive(Debug, Clone, PartialEq)]
pub enum ContractError {
    /// `avail_ts_ms < recv_ts_ms`: data cannot be available to the consumer
    /// before it was received.
    AvailBeforeRecv { recv_ts_ms: i64, avail_ts_ms: i64 },
    /// Terminal step without `end_reason`, or `end_reason` on a non-terminal step.
    InconsistentEnd { terminal: bool },
    /// `RewardBreakdown.total` differs from the sum of its components.
    RewardMismatch { expected: Decimal, total: Decimal },
    /// `MarketOutcome` without provenance: an official result must say where it
    /// came from (source/doc), never an empty string.
    EmptyProvenance,
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AvailBeforeRecv {
                recv_ts_ms,
                avail_ts_ms,
            } => write!(f, "avail_ts_ms ({avail_ts_ms}) < recv_ts_ms ({recv_ts_ms})"),
            Self::InconsistentEnd { terminal } => write!(
                f,
                "inconsistent step end: terminal={terminal} vs end_reason"
            ),
            Self::RewardMismatch { expected, total } => {
                write!(
                    f,
                    "reward total ({total}) != sum of components ({expected})"
                )
            }
            Self::EmptyProvenance => {
                write!(f, "official outcome without provenance (empty string)")
            }
        }
    }
}

impl std::error::Error for ContractError {}

// ── A — Versioned research event ────────────────────────────────────

/// Source of the event. Venue and transport are kept apart because the
/// reliability of `origin_ts_ms` and the quality flags differ between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSource {
    /// Polymarket public market WS (books, trades, tick_size_change).
    PolymarketWs,
    /// Polymarket REST (Gamma API, book resync, feeSchedule).
    PolymarketRest,
    /// Polymarket user channel (own MATCHED trades) — live only.
    PolymarketUserWs,
    /// Binance WS (underlying spot price, liquidations).
    BinanceWs,
    /// Chainlink oracle (when resolution depends on its feed).
    Chainlink,
    /// Internal simulator (`order::sim_router`) — virtual clock, not wall clock.
    InternalSim,
    /// Internal bot state (decisions, ledger) with no external source.
    Internal,
}

/// Lifecycle of an order within the simulated exchange (contract C):
/// each transition is a SEPARATE event with its own `sim_now_ms`.
/// It mirrors the instants that `order::sim_router` already models: the submit of
/// the `OrderRequest`, arrival after `paper_latency_ms`, acceptance, the
/// fills (each `SimExecution` with its `exec_id`), cancels and the
/// notification to the consumer (the `UserTrade MATCHED` → fill_sink).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimEventKind {
    /// Order sent to the sim (the incoming `OrderRequest`).
    Submit,
    /// Arrival at the simulated exchange after the modelled latency.
    Arrive,
    /// Accepted by the simulated venue (resting on the book or eligible).
    Accept,
    /// Rejected (FOK without fill, insufficient funds, post_only that crosses…).
    Reject,
    /// Execution — one `SimExecution`, with its own `exec_id` per partial fill.
    Fill,
    /// Cancellation requested.
    CancelRequest,
    /// Cancellation confirmed — includes the GTD auto-expiry at clear_at.
    CancelAck,
    /// Notification to the consumer (`UserTrade MATCHED` → handle_trade).
    Notify,
}

/// Research event type.
///
/// The eight simulator transitions are grouped under `Sim(SimEventKind)`: one
/// single source of truth for the sim vocabulary (the same enum used by `SimEvent`,
/// contract C) instead of duplicating `Sim*` variants here, and
/// `matches!(kind, EventKind::Sim(_))` filters all sim traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// Full snapshot of a token/bracket book.
    BookSnapshot,
    /// Incremental book delta (price_change applied).
    BookDelta,
    /// Trade from the market tape.
    Trade,
    /// Underlying price tick (Binance oracle or other).
    PriceTick,
    /// Market resolution event (official result known).
    Resolution,
    /// Market metadata/rules (contract B) — `payload` = `MarketMeta`.
    MarketMeta,
    /// Order sent by the executor to the destination (paper or live) — the
    /// `OrderRequest`/`OrderIntent` as it left, before the result.
    OrderSubmit,
    /// Simulator lifecycle event (contract C).
    Sim(SimEventKind),
}

/// Data quality flags (bitmask). `0` = complete, healthy event.
///
/// Implemented as a serde-transparent `u32` newtype (serializes as the
/// bare integer) instead of the `bitflags` crate, to avoid adding new
/// dependencies to `Cargo.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QualityFlags(pub u32);

impl QualityFlags {
    /// No flags — complete data.
    pub const NONE: Self = Self(0);
    /// Gap in the source sequence (exchange seq gap, reconnection without
    /// resync, message loss).
    pub const GAP: Self = Self(1 << 0);
    /// The event comes from a resync (REST) after the WS went out of sync.
    pub const RESYNC: Self = Self(1 << 1);
    /// Duplicate detected by id/sequence dedup.
    pub const DUPLICATE: Self = Self(1 << 2);
    /// The source provides no remote timestamp — `origin_ts_ms` is `None`.
    /// Deliberately redundant: allows filtering without inspecting the field.
    pub const MISSING_REMOTE_TS: Self = Self(1 << 3);
    /// Reduced fidelity: history at coarser granularity than live data
    /// (e.g. 1s klines reconstructing an L2 book). The label travels WITH
    /// the data — never fill gaps with silent assumptions.
    pub const REDUCED_FIDELITY: Self = Self(1 << 4);
    /// Synthetic/derived data generated by the sim or the bot itself, not a
    /// raw venue observation.
    pub const SYNTHETIC: Self = Self(1 << 5);

    /// Empty bitmask.
    pub const fn empty() -> Self {
        Self::NONE
    }

    /// `true` if all the flags in `f` are present.
    pub const fn contains(self, f: Self) -> bool {
        self.0 & f.0 == f.0 && f.0 != 0
    }

    /// Union of flags (const, for composing constants).
    pub const fn union(self, f: Self) -> Self {
        Self(self.0 | f.0)
    }

    /// Adds a flag in place.
    pub fn insert(&mut self, f: Self) {
        self.0 |= f.0;
    }

    /// `true` if no flag is set (complete data).
    pub const fn is_clean(self) -> bool {
        self.0 == 0
    }

    /// Raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Rebuilds from raw bits (without truncating — high bits without a
    /// name are kept for forward compatibility).
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
}

/// Versioned research event (contract A).
///
/// `payload` is a `serde_json::Value` on purpose: it keeps the ORIGINAL numeric
/// payload exactly as received, without coupling the recorder schema to the
/// mutable internal types (`BookSnapshot`, `AssetPrice`…), which evolve with the
/// bot. Consumers that want typed access use [`Self::payload_as`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchEvent {
    /// Always [`SCHEMA_VERSION`] for new events.
    pub schema_version: u32,
    pub source: EventSource,
    pub kind: EventKind,
    /// Market/token/asset as applicable: a Binance tick has no `market_cid`;
    /// a Polymarket book has no `asset` of its own (the underlying is derived
    /// from the slug). Genuinely absent = `None`.
    pub market_cid: Option<String>,
    pub token_id: Option<String>,
    /// Underlying ("BTC", "BTC/USDT"…), if applicable to the event.
    pub asset: Option<String>,
    /// Original numeric payload of the event (see the struct doc).
    pub payload: Value,
    /// ORIGIN timestamp (exchange/source clock, ms epoch) when the source
    /// provides it. NEVER invented: `None` when the source does not send it or
    /// its precision is not verified. Never order decisions by this clock; the
    /// canonical ledger order is `local_seq`.
    pub origin_ts_ms: Option<i64>,
    /// Local receive time, ms epoch.
    pub recv_ts_ms: i64,
    /// Availability to the consumer, ms epoch. Invariant:
    /// `avail_ts_ms >= recv_ts_ms` — data cannot be visible before it arrives.
    /// In replay it is the virtual-clock instant at which the consumer sees it
    /// (may be > recv due to replay latency).
    pub avail_ts_ms: i64,
    /// Monotonic local sequence — the canonical order of the event ledger.
    /// Remote timestamps order nothing.
    pub local_seq: u64,
    /// Data quality flags.
    pub quality: QualityFlags,
}

impl ResearchEvent {
    /// Validated constructor. Fails if `avail_ts_ms < recv_ts_ms`.
    /// `schema_version` is set to [`SCHEMA_VERSION`].
    ///
    /// `origin_ts_ms` is accepted as given: the contract requires it to be `None`
    /// when the source does not report it; it is NOT filled with `recv_ts_ms`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: EventSource,
        kind: EventKind,
        market_cid: Option<String>,
        token_id: Option<String>,
        asset: Option<String>,
        payload: Value,
        origin_ts_ms: Option<i64>,
        recv_ts_ms: i64,
        avail_ts_ms: i64,
        local_seq: u64,
        quality: QualityFlags,
    ) -> Result<Self, ContractError> {
        let ev = Self {
            schema_version: SCHEMA_VERSION,
            source,
            kind,
            market_cid,
            token_id,
            asset,
            payload,
            origin_ts_ms,
            recv_ts_ms,
            avail_ts_ms,
            local_seq,
            quality,
        };
        ev.validate()?;
        Ok(ev)
    }

    /// Temporal invariant of the event (contract A).
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.avail_ts_ms < self.recv_ts_ms {
            return Err(ContractError::AvailBeforeRecv {
                recv_ts_ms: self.recv_ts_ms,
                avail_ts_ms: self.avail_ts_ms,
            });
        }
        Ok(())
    }

    /// Deserializes the payload into a concrete kind — a pure helper for the
    /// consumers that want typed access over the raw `Value`.
    pub fn payload_as<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(self.payload.clone())
    }
}

// ── B — Market metadata ─────────────────────────────────────────────────

/// Market resolution rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionRule {
    /// Up/Down: YES wins if the underlying price at close exceeds the
    /// opening reference price (according to the official market text).
    UpDownReference,
    /// Untyped rule — free text for markets outside the bot's
    /// up/down universe.
    Other(String),
}

/// Official winning side of a resolved market.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeSide {
    Yes,
    No,
}

/// Validity window of the market's operating parameters (contract B).
/// `valid_to_ms = None` = valid until a new warning (last window).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RulesWindow {
    pub valid_from_ms: i64,
    pub valid_to_ms: Option<i64>,
    /// Actual parabolic taker fee rate (`fee = rate × p × (1−p)`), fraction
    /// 0.0–1.0 — same domain as `MarketEntry::parabolic_fee_rate`.
    /// `None` if the source did not report it in the window.
    pub parabolic_fee_rate: Option<Decimal>,
    /// Taker fee in basis points as reported by Gamma (informational).
    pub fee_rate_bps: Option<u32>,
    /// Current minimum tick (TICK-1: changes at extreme prices).
    pub tick_size: Option<Decimal>,
    /// Minimum order size of the market in the window (shares), if the
    /// source reports it.
    pub min_order_size: Option<Decimal>,
}

/// Official market result with knowledge time and provenance
/// (contract B). Only built with a REAL winner — the absence of a result is
/// `None` in `MarketMeta::outcome`, never an invented winner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketOutcome {
    pub winner: OutcomeSide,
    /// Instant the result was known LOCALLY (ms epoch) — neither the on-chain
    /// instant nor the market's theoretical resolution time.
    pub known_at_ms: i64,
    /// Data provenance: source and reference ("gamma:…", "clob:…",
    /// URL/hash of the resolution document). Mandatory, non-empty.
    pub provenance: String,
}

impl MarketOutcome {
    /// Validated constructor: the provenance cannot be empty.
    pub fn new(
        winner: OutcomeSide,
        known_at_ms: i64,
        provenance: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let provenance = provenance.into();
        if provenance.trim().is_empty() {
            return Err(ContractError::EmptyProvenance);
        }
        Ok(Self {
            winner,
            known_at_ms,
            provenance,
        })
    }
}

/// Market metadata (contract B): resolution rules and source, validity of
/// fee/tick/minimum size, and the official result with its knowledge time and
/// provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketMeta {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    pub market_cid: String,
    pub slug: String,
    pub yes_token_id: String,
    pub no_token_id: String,
    /// Underlying derived from the slug prefix ("BTC"…), if applicable.
    pub asset: Option<String>,
    pub neg_risk: bool,
    /// Open and close of the bracket (same domain as `MarketEntry` —
    /// `clear_at`/`market_open_ts` are unix seconds in the registry).
    pub market_open_ts: i64,
    pub clear_at: i64,
    /// Opening reference price (strike), if known.
    pub reference_price: Option<Decimal>,
    /// Market resolution rule.
    pub rule: ResolutionRule,
    /// Source of truth for the resolution (which feed decides the outcome).
    pub resolution_source: EventSource,
    /// Concrete reference within the source (symbol, feed address, doc).
    pub resolution_source_ref: Option<String>,
    /// Validity windows of fee/tick/min_size, in chronological order.
    pub rules_windows: Vec<RulesWindow>,
    /// Official result if already known. OUTSIDE the observable state:
    /// later labels are never features of an observation.
    pub outcome: Option<MarketOutcome>,
}

/// Confirmed label of a market (labels consumer).
///
/// It exists as a separate kind so that it is IMPOSSIBLE to fabricate a
/// confirmed label without an official result: the only constructor is
/// [`MarketLabel::from_meta`], which returns `None` when
/// `MarketMeta::outcome` is `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketLabel {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    pub market_cid: String,
    pub winner: OutcomeSide,
    /// When the result was known locally (ms epoch).
    pub known_at_ms: i64,
    /// Provenance of the result, inherited from the `MarketOutcome`.
    pub provenance: String,
}

impl MarketLabel {
    /// `None` if the market has no official result yet — a confirmed label is
    /// never fabricated from `outcome: None` nor from an assumed
    /// winner.
    pub fn from_meta(meta: &MarketMeta) -> Option<Self> {
        meta.outcome.as_ref().map(|o| Self {
            schema_version: SCHEMA_VERSION,
            market_cid: meta.market_cid.clone(),
            winner: o.winner,
            known_at_ms: o.known_at_ms,
            provenance: o.provenance.clone(),
        })
    }
}

// ── C — Simulation events ─────────────────────────────────────────────────

/// Lifecycle event of an order in the simulator (contract C).
///
/// Terminology inherited from `order::sim_router` — not renamed:
/// - `sim_now_ms`: instant of the single virtual clock — same domain as
///   `SimClock` (injectable clock in unix ms). In replay, same seed +
///   same clock ⇒ same sequence of events.
/// - `local_order_id`: local identity of the order (`OrderRequest::
///   local_order_id` → `SimExecution::local_order_id`), reservation id
///   persisted before submit.
/// - `order_id`: order id of the simulated venue (`SimExecution::order_id`).
/// - `exec_id`: identity of the EXECUTION (`SimExecution::exec_id`), unique
///   per fill, partial fills included — only present in `SimEventKind::Fill`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimEvent {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    pub kind: SimEventKind,
    /// Single virtual clock of the sim (ms epoch of the injected domain).
    pub sim_now_ms: i64,
    /// RNG seed of the sim in this run (`with_deps::rng_seed`).
    pub seed: u64,
    /// Monotonic local sequence — the canonical order within the ledger.
    pub local_seq: u64,
    pub market_cid: Option<String>,
    pub token_id: Option<String>,
    pub local_order_id: Option<String>,
    pub order_id: Option<String>,
    /// Only in `Fill` — never equal to `order_id` (one order produces N
    /// executions; see the `SimExecution` doc).
    pub exec_id: Option<String>,
    /// Reason for the event when applicable (Reject, Notify…).
    pub reason: Option<String>,
    /// Event detail. In `Submit` the serialized `OrderRequest`; in
    /// `Fill` the `SimExecution` with its `SimExecLevel`s (negotiated price and
    /// fee per separate tranche). `Value` for the same reason as in
    /// `ResearchEvent`: the contract is not coupled to mutable internal types.
    pub detail: Value,
}

impl SimEvent {
    /// Constructor with `schema_version` set to [`SCHEMA_VERSION`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: SimEventKind,
        sim_now_ms: i64,
        seed: u64,
        local_seq: u64,
        market_cid: Option<String>,
        token_id: Option<String>,
        local_order_id: Option<String>,
        order_id: Option<String>,
        exec_id: Option<String>,
        reason: Option<String>,
        detail: Value,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            kind,
            sim_now_ms,
            seed,
            local_seq,
            market_cid,
            token_id,
            local_order_id,
            order_id,
            exec_id,
            reason,
            detail,
        }
    }
}

// ── D — Episode / StepRecord ─────────────────────────────────────────────────

/// Agent action in a step of the episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentAction {
    /// Do not trade.
    Hold,
    /// Buy `size` shares of the token at limit price `price`.
    Buy {
        token_id: String,
        price: Decimal,
        size: Decimal,
    },
    /// Sell `size` shares of the token at limit price `price`.
    Sell {
        token_id: String,
        price: Decimal,
        size: Decimal,
    },
    /// Cancel a live order by its `order_id`.
    Cancel { order_id: String },
}

/// Result of the action proposed in a step (contract D: proposed vs
/// executed/rejected).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionOutcome {
    /// Executed — `event_seqs` references the `local_seq` of the resulting events
    /// (Fill/Accept from the sim, ledger fill…).
    Executed { event_seqs: Vec<u64> },
    /// Accepted by the venue and pending result — a resting order (GTD) that
    /// stayed live, or a fill deferred to a later event.
    /// An ACK is NEVER a fill: fills arrive as `SimEventKind::Fill` / ledger fills
    /// in later steps, referenceable by seq.
    /// The reserved-cash inventory changes here (`reserved_usdc`); the
    /// positions only when the fill materializes.
    Accepted { event_seqs: Vec<u64> },
    /// Rejected by a previous gate or by the venue — `event_seqs`
    /// references the `SimEventKind::Reject`, if any.
    Rejected {
        reason: String,
        event_seqs: Vec<u64>,
    },
}

/// Open position observable in a step (all `Decimal` — money).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionSnapshot {
    pub token_id: String,
    pub market_cid: String,
    pub shares: Decimal,
    pub avg_price: Decimal,
}

/// Inventory and reserves observable after applying the step (contract D).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InventorySnapshot {
    /// Free USDC cash.
    pub cash_usdc: Decimal,
    /// USDC reserved for live buy orders (same concept as
    /// `sim_router::reserved_buy`).
    pub reserved_usdc: Decimal,
    /// Open positions per token.
    pub positions: Vec<PositionSnapshot>,
}

/// Broken-down reward (contract D): price + fee + risk penalty. `total` is the
/// exact sum of the three components (fee and penalty are typically <= 0). The
/// breakdown keeps training from confusing friction/penalty with real edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardBreakdown {
    /// Price component of the step (realized PnL + mark-to-market).
    pub price_component: Decimal,
    /// Charged friction (fees) — negative when deducted.
    pub fee_component: Decimal,
    /// Risk penalty (exposure, drawdown, kill…) — <= 0.
    pub risk_penalty: Decimal,
    /// Exact sum of the three components.
    pub total: Decimal,
}

impl RewardBreakdown {
    /// Builds with `total` computed as the exact sum of the components.
    pub fn new(price_component: Decimal, fee_component: Decimal, risk_penalty: Decimal) -> Self {
        Self {
            price_component,
            fee_component,
            risk_penalty,
            total: price_component + fee_component + risk_penalty,
        }
    }

    /// Invariant: `total == price_component + fee_component + risk_penalty`.
    pub fn validate(&self) -> Result<(), ContractError> {
        let expected = self.price_component + self.fee_component + self.risk_penalty;
        if self.total != expected {
            return Err(ContractError::RewardMismatch {
                expected,
                total: self.total,
            });
        }
        Ok(())
    }
}

/// Reason a terminal step / episode ended (contract D).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    /// The market resolved — natural end of the episode.
    MarketResolved,
    /// The time horizon was reached without resolution.
    HorizonReached,
    /// Risk stop / kill switch.
    RiskHalt,
    /// Aborted due to data quality (sustained gap/resync).
    DataQualityAbort,
    /// Manual end, or from the training harness.
    Manual,
}

/// A step of the episode (contract D): available observation, legal
/// actions, proposed vs executed/rejected action, inventory/reserves,
/// broken-down reward, terminality and quality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepRecord {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Index of the step within the episode (0-based, monotonic).
    pub step_idx: u64,
    /// `local_seq` of the `ResearchEvent`s that form the observation
    /// AVAILABLE at this step — all with `avail_ts_ms <= decision_ts_ms`.
    /// Referencing by seq (never by remote clock) is what makes replay
    /// deterministic. NEVER include events that reveal a future result
    /// (later labels/resolution — contract B/D).
    pub observation_seqs: Vec<u64>,
    /// Decision instant on the episode clock (virtual in replay).
    pub decision_ts_ms: i64,
    /// Legal actions at this step.
    pub legal_actions: Vec<AgentAction>,
    /// Action proposed by the agent (`None` = no proposal).
    pub proposed_action: Option<AgentAction>,
    /// Executed or rejected (`None` if there was no proposal).
    pub action_outcome: Option<ActionOutcome>,
    /// Inventory/reserves after applying the step.
    pub inventory: InventorySnapshot,
    /// Broken-down reward of the step.
    pub reward: RewardBreakdown,
    /// `true` if this is the last step of the episode.
    pub terminal: bool,
    /// End reason — mandatory if `terminal`, forbidden otherwise.
    pub end_reason: Option<EndReason>,
    /// Quality flags of the step.
    pub quality: QualityFlags,
}

impl StepRecord {
    /// Invariant: `terminal` and `end_reason` are consistent (both or
    /// neither).
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.terminal != self.end_reason.is_some() {
            return Err(ContractError::InconsistentEnd {
                terminal: self.terminal,
            });
        }
        Ok(())
    }
}

/// Complete episode (contract D): window metadata + steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    pub episode_id: String,
    pub market_cid: Option<String>,
    pub asset: Option<String>,
    /// Episode window on the run clock (virtual in replay).
    pub started_at_ms: i64,
    /// `None` = open episode.
    pub ended_at_ms: Option<i64>,
    /// Steps in order (increasing `step_idx`).
    pub steps: Vec<StepRecord>,
    /// End reason of the episode (`None` = open).
    pub end_reason: Option<EndReason>,
    /// Quality flags of the complete episode.
    pub quality: QualityFlags,
}

// ── E — Experiment manifest ──────────────────────────────────────────────

/// Versioned dataset file (contract E): path RELATIVE to the research
/// directory + content hash (hex, SHA-256).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFile {
    /// Path relative to `ResearchManifest::research_dir` — never absolute
    /// nor inside the bot's active DB.
    pub path: String,
    /// SHA-256 of the content, in hex.
    pub sha256: String,
}

/// Dataset split by time window (contract E).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitSpec {
    /// "train" | "validation" | "test" | any free-form name.
    pub name: String,
    pub from_ms: i64,
    pub to_ms: i64,
}

/// Experiment manifest (contract E): the same input reproduces the economic
/// ledger — code, config, model, data, splits, sim profile and seeds are
/// pinned by hash, and the data paths live in a research directory EXTERNAL to
/// the active DB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchManifest {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Manifest creation, ms epoch.
    pub created_at_ms: i64,
    /// Hash of the code tree (commit id or tree hash), hex.
    pub code_hash: String,
    /// Hash of the effective config (general + strategies + relevant env).
    pub config_hash: String,
    /// Hash of the model checkpoint used, if the episode used one.
    pub model_hash: Option<String>,
    /// Hash of the simulator profile (latency, fees, slippage, fill
    /// rates…) used to generate the `InternalSim` events.
    pub sim_profile_hash: Option<String>,
    /// Experiment seeds (sim rng, dataset sampling…).
    pub seeds: Vec<u64>,
    /// Declared dataset splits.
    pub splits: Vec<SplitSpec>,
    /// Research root directory — EXTERNAL to the bot's active DB.
    /// The `files` paths are relative to this directory.
    pub research_dir: String,
    /// Experiment data files (events, episodes, labels…).
    pub files: Vec<ManifestFile>,
}

// ── F — Episodic environment (IPC contract of the `hrf_research_env` driver) ────────
//
// One subprocess per environment, JSONL protocol over stdin/stdout (one line =
// one message). The Rust process is the ONLY owner of the economic state
// (Executor → RiskGuard → SimulatedRouter → fill_sink → SQLite :memory:,
// same pipeline as `replay.rs`); Python only proposes `AgentAction` and reads
// `EnvObservation`/`StepRecord` — it never keeps a second results ledger.
//
// Isolation of the replay clock (`time::REPLAY_NOW_MS` is global to the
// process): one `hrf_research_env` process per environment — NEVER two
// ReplayEngine in the same process assuming safety via seed.

/// JSONL request from Python to the driver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum EnvRequest {
    /// Starts an episode over the tape already loaded. `seed` + same tape ⇒
    /// same episode, byte for byte. `market_cid` restricts the episode to one
    /// market (`None` = full tape universe).
    Reset {
        episode_id: String,
        seed: u64,
        #[serde(default)]
        market_cid: Option<String>,
    },
    /// Proposes an action at the current decision point. `Hold` advances to the
    /// next decision point. Illegal action → `ActionOutcome::
    /// Rejected` recorded in the `StepRecord`, never a protocol error
    /// (the rejection is episode data, not an exception).
    Step { action: AgentAction },
    /// Legal actions at the current decision point.
    LegalActions,
    /// Closes the episode (drains the pipeline, final digest) and the process.
    Close,
}

/// Causal observation at a decision point: only events with
/// `avail_ts_ms <= decision_ts_ms` (contracts A/B — nothing from the future).
///
/// `features` are computed by the SAME extractor as the shadow
/// (`model_observer::extract_features`) over the causal state — a single
/// feature implementation for train/replay/shadow (train/serve parity).
/// `None` values are truly absent features (the scaler imputes them with the
/// train median), never invented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvObservation {
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Decision instant on the episode clock (virtual).
    pub decision_ts_ms: i64,
    pub market_cid: Option<String>,
    /// Feature map name → value (`None` = truly absent).
    pub features: BTreeMap<String, Option<f64>>,
    /// Inventory/reserves after the last applied step.
    pub inventory: InventorySnapshot,
    /// `local_seq` of the events that form this observation.
    pub observation_seqs: Vec<u64>,
    /// Quality flags of the cut (GAP/RESYNC propagated).
    pub quality: QualityFlags,
}

/// Quality summary of a replay tape/run (addendum v2).
/// Same vocabulary in the `hrf_backtest --out` report (field `quality`) and in
/// the `hello` of the episodic environment.
///
/// "Scope" = the cids/tokens of the markets whose stream was processed in
/// the run window (the `MarketMeta` observed by the reconstructor; if the run
/// filters a market, the scope is that market). The `*_in_scope` counters only
/// measure degradation attributable to that set; degradation outside the scope
/// is reported in the global counters but does not taint the episode certification.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QualitySummary {
    /// Quality marks processed in the run window (all of them, whether or
    /// not in scope).
    pub quality_marks: u64,
    /// Tape lines that did not decode (malformed/schema).
    pub malformed: u64,
    /// Historical agent events counted (never replayed).
    pub agent: u64,
    /// `price_change` deltas discarded across the whole tape (no base /
    /// `needs_snapshot`).
    pub discarded_deltas: u64,
    /// discarded deltas attributable to the run scope.
    pub discarded_deltas_in_scope: u64,
    /// Market books suppressed because the opposite half had no state.
    pub missing_half: u64,
    /// Books suppressed for unchanged top-of-book (runner dedup).
    pub books_top_equal: u64,
    /// RESYNC marks processed in the run window.
    pub resyncs: u64,
    /// Tokens in scope with a gap episode (invalidated segment or
    /// GAP mark from the recorder on the token).
    pub tokens_with_gap_in_scope: u64,
    /// Cids in scope with a gap (sorted, capped at 50 entries).
    pub gap_markets_in_scope: Vec<String>,
    /// Percentage of in-scope `price_change` entries effectively applied (100.0
    /// when the scope received no deltas — nothing to lose).
    pub delta_coverage_pct_in_scope: f64,
}

/// Quality scope announced in the environment `hello` (addendum v2):
/// the universe of markets and the window covered by the loaded tape. The
/// concrete episode filter is set in `reset.market_cid`; the abort on
/// `DataQualityAbort` only considers marks within that scope. Without a filter,
/// the whole tape is in scope.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QualityScope {
    /// Current market filter (`None` in the hello: there is no episode yet,
    /// the announced scope is the full tape).
    pub market_cid: Option<String>,
    /// Cids whose stream was observed on the tape (sorted).
    pub markets: Vec<String>,
    /// Time window of the tape `[min_ms, max_ms]` in availability clock
    /// (`None` if the tape carries no events).
    pub window_ms: Option<[i64; 2]>,
}

/// JSONL response from the driver to Python.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EnvResponse {
    /// Handshake on startup: schema version, declared fidelity of the tape
    /// (`"certified"` = causal ResearchEvent tape; `"reduced"` = historical
    /// SQL, diagnostic mode, never silenced), and, since addendum v2, the quality
    /// summary of the tape plus the scope of markets/window it covers
    /// (`quality`/`quality_scope`).
    Hello {
        schema_version: u32,
        fidelity: String,
        #[serde(default)]
        quality: QualitySummary,
        #[serde(default)]
        quality_scope: QualityScope,
    },
    /// Response to `reset`: first causal observation + legal actions.
    Reset {
        observation: EnvObservation,
        legal_actions: Vec<AgentAction>,
    },
    /// Response to `step`: the `StepRecord` of the applied step and the next
    /// observation (`None` when `step.terminal` — terminal due to
    /// resolution/risk/quality, or truncated by horizon; see
    /// `EndReason`).
    StepResult {
        step: StepRecord,
        observation: Option<EnvObservation>,
        legal_actions: Vec<AgentAction>,
    },
    /// Response to `legal_actions`.
    LegalActions { legal_actions: Vec<AgentAction> },
    /// Response to `close`: economic digest of the episode and reconciliation.
    Closed {
        episode_id: String,
        digest: String,
        reconciled: bool,
    },
    /// Protocol or engine failure. `fatal=true` invalidates the episode;
    /// `fatal=false` keeps the connection usable (e.g. malformed JSON).
    Error { message: String, fatal: bool },
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use serde_json::json;

    fn base_event() -> ResearchEvent {
        ResearchEvent::new(
            EventSource::PolymarketWs,
            EventKind::BookSnapshot,
            Some("0xcid".into()),
            Some("tok-yes".into()),
            None,
            json!({"bids": [["0.45", "100"]], "asks": [["0.47", "80"]]}),
            Some(1_700_000_000_000),
            1_700_000_000_050,
            1_700_000_000_055,
            42,
            QualityFlags::NONE,
        )
        .unwrap()
    }

    // ── A — ResearchEvent ───────────────────────────────────────────────

    #[test]
    fn research_event_roundtrip_serde_with_version() {
        let ev = base_event();
        let s = serde_json::to_string(&ev).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["schema_version"], json!(SCHEMA_VERSION));
        assert_eq!(v["source"], json!("polymarket_ws"));
        assert_eq!(v["kind"], json!("book_snapshot"));
        let back: ResearchEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn avail_ts_ms_cannot_precede_recv() {
        let res = ResearchEvent::new(
            EventSource::BinanceWs,
            EventKind::PriceTick,
            None,
            None,
            Some("BTC".into()),
            json!({"price": "67000.5"}),
            Some(100),
            200,
            150, // avail < recv — invalid
            1,
            QualityFlags::NONE,
        );
        assert_eq!(
            res.unwrap_err(),
            ContractError::AvailBeforeRecv {
                recv_ts_ms: 200,
                avail_ts_ms: 150
            }
        );
        // The boundary itself is valid.
        let ok = ResearchEvent::new(
            EventSource::BinanceWs,
            EventKind::PriceTick,
            None,
            None,
            None,
            json!(null),
            None,
            200,
            200,
            2,
            QualityFlags::NONE,
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn origin_ts_ms_can_be_none_without_being_invented() {
        // A source without a remote timestamp serializes null — never a value
        // fabricated from recv_ts_ms.
        let ev = ResearchEvent::new(
            EventSource::PolymarketRest,
            EventKind::MarketMeta,
            Some("cid".into()),
            None,
            None,
            json!({}),
            None,
            1000,
            1000,
            7,
            QualityFlags::MISSING_REMOTE_TS,
        )
        .unwrap();
        assert_eq!(ev.origin_ts_ms, None);
        let v: Value = serde_json::from_str(&serde_json::to_string(&ev).unwrap()).unwrap();
        assert_eq!(v["origin_ts_ms"], Value::Null);
        assert!(ev.quality.contains(QualityFlags::MISSING_REMOTE_TS));
    }

    #[test]
    fn quality_flags_bitmask() {
        let q = QualityFlags::GAP.union(QualityFlags::RESYNC);
        assert!(q.contains(QualityFlags::GAP));
        assert!(q.contains(QualityFlags::RESYNC));
        assert!(!q.contains(QualityFlags::DUPLICATE));
        assert!(!q.is_clean());
        let mut q2 = QualityFlags::NONE;
        assert!(q2.is_clean());
        q2.insert(QualityFlags::DUPLICATE);
        assert_eq!(q2.bits(), QualityFlags::DUPLICATE.bits());
        assert_eq!(QualityFlags::from_bits(q.bits()), q);
        // Serializes as the transparent bare integer.
        assert_eq!(serde_json::to_string(&q).unwrap(), "3");
    }

    #[test]
    fn sim_event_serializes_with_nested_kind() {
        let ev = ResearchEvent::new(
            EventSource::InternalSim,
            EventKind::Sim(SimEventKind::Fill),
            Some("cid".into()),
            Some("tok".into()),
            None,
            json!({"exec_id": "sim-exec-3"}),
            None,
            500,
            500,
            9,
            QualityFlags::SYNTHETIC,
        )
        .unwrap();
        let v: Value = serde_json::from_str(&serde_json::to_string(&ev).unwrap()).unwrap();
        assert_eq!(v["kind"], json!({"sim": "fill"}));
        let back: ResearchEvent = serde_json::from_value(v).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn payload_as_deserializes_typed() {
        let ev = base_event();
        let p: serde_json::Map<String, Value> = ev.payload_as().unwrap();
        assert!(p.contains_key("bids"));
    }

    // ── B — MarketMeta / labels ─────────────────────────────────────────

    fn meta_without_outcome() -> MarketMeta {
        MarketMeta {
            schema_version: SCHEMA_VERSION,
            market_cid: "0xcid".into(),
            slug: "btc-updown-1h".into(),
            yes_token_id: "yes".into(),
            no_token_id: "no".into(),
            asset: Some("BTC".into()),
            neg_risk: false,
            market_open_ts: 1_700_000_000,
            clear_at: 1_700_003_600,
            reference_price: Some(dec!(67000.50)),
            rule: ResolutionRule::UpDownReference,
            resolution_source: EventSource::BinanceWs,
            resolution_source_ref: Some("BTCUSDT spot".into()),
            rules_windows: vec![
                RulesWindow {
                    valid_from_ms: 0,
                    valid_to_ms: Some(1_700_000_000_000),
                    parabolic_fee_rate: Some(dec!(0.07)),
                    fee_rate_bps: Some(0),
                    tick_size: Some(dec!(0.01)),
                    min_order_size: Some(dec!(5)),
                },
                RulesWindow {
                    valid_from_ms: 1_700_000_000_000,
                    valid_to_ms: None,
                    parabolic_fee_rate: Some(dec!(0.04)),
                    fee_rate_bps: None,
                    tick_size: Some(dec!(0.001)),
                    min_order_size: None,
                },
            ],
            outcome: None,
        }
    }

    #[test]
    fn market_meta_roundtrip_serde() {
        let mut meta = meta_without_outcome();
        meta.outcome = Some(
            MarketOutcome::new(OutcomeSide::Yes, 1_700_004_000_000, "gamma:resolved").unwrap(),
        );
        let s = serde_json::to_string(&meta).unwrap();
        let back: MarketMeta = serde_json::from_str(&s).unwrap();
        assert_eq!(back, meta);
        assert_eq!(back.rules_windows.len(), 2);
        assert_eq!(back.rules_windows[1].valid_to_ms, None);
    }

    /// Contract B/D: a market without an official result CANNOT produce a
    /// confirmed label — `MarketLabel::from_meta` returns `None`.
    #[test]
    fn confirmed_label_requires_official_outcome() {
        let meta = meta_without_outcome();
        assert_eq!(MarketLabel::from_meta(&meta), None);

        let mut meta = meta_without_outcome();
        meta.outcome = Some(
            MarketOutcome::new(OutcomeSide::No, 1_700_004_000_000, "clob:resolution").unwrap(),
        );
        let label = MarketLabel::from_meta(&meta).unwrap();
        assert_eq!(label.winner, OutcomeSide::No);
        assert_eq!(label.known_at_ms, 1_700_004_000_000);
        assert_eq!(label.provenance, "clob:resolution");
        assert_eq!(label.market_cid, "0xcid");
        assert_eq!(label.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn outcome_without_provenance_is_invalid() {
        assert_eq!(
            MarketOutcome::new(OutcomeSide::Yes, 0, "  "),
            Err(ContractError::EmptyProvenance)
        );
    }

    // ── C — SimEvent ────────────────────────────────────────────────────

    #[test]
    fn sim_event_roundtrip_and_variants() {
        // The eight transitions of contract C exist and serialize.
        for (kind, expected) in [
            (SimEventKind::Submit, "submit"),
            (SimEventKind::Arrive, "arrive"),
            (SimEventKind::Accept, "accept"),
            (SimEventKind::Reject, "reject"),
            (SimEventKind::Fill, "fill"),
            (SimEventKind::CancelRequest, "cancel_request"),
            (SimEventKind::CancelAck, "cancel_ack"),
            (SimEventKind::Notify, "notify"),
        ] {
            assert_eq!(
                serde_json::to_string(&kind).unwrap(),
                format!("\"{expected}\"")
            );
        }

        let ev = SimEvent::new(
            SimEventKind::Fill,
            1_700_000_000_123, // sim_now_ms — virtual clock
            0xDEAD_BEEF,       // seed
            77,                // local_seq
            Some("cid".into()),
            Some("tok-yes".into()),
            Some("local-9".into()), // local_order_id (sim_router terminology)
            Some("sim-order-9".into()),
            Some("sim-exec-12".into()), // exec_id ≠ order_id
            None,
            json!({"shares": "10", "notional": "4.5", "fee_usdc": "0.012"}),
        );
        let s = serde_json::to_string(&ev).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["schema_version"], json!(SCHEMA_VERSION));
        assert_eq!(v["kind"], json!("fill"));
        let back: SimEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ev);
        assert_ne!(back.exec_id, back.order_id);
    }

    // ── D — StepRecord / Episode ────────────────────────────────────────

    fn inventory_state() -> InventorySnapshot {
        InventorySnapshot {
            cash_usdc: dec!(950.25),
            reserved_usdc: dec!(45.00),
            positions: vec![PositionSnapshot {
                token_id: "tok-yes".into(),
                market_cid: "0xcid".into(),
                shares: dec!(10),
                avg_price: dec!(0.45),
            }],
        }
    }

    fn step(terminal: bool, end_reason: Option<EndReason>) -> StepRecord {
        StepRecord {
            schema_version: SCHEMA_VERSION,
            step_idx: 3,
            observation_seqs: vec![40, 41, 42],
            decision_ts_ms: 1_700_000_000_055,
            legal_actions: vec![
                AgentAction::Hold,
                AgentAction::Buy {
                    token_id: "tok-yes".into(),
                    price: dec!(0.47),
                    size: dec!(10),
                },
                AgentAction::Cancel {
                    order_id: "sim-order-1".into(),
                },
            ],
            proposed_action: Some(AgentAction::Buy {
                token_id: "tok-yes".into(),
                price: dec!(0.47),
                size: dec!(10),
            }),
            action_outcome: Some(ActionOutcome::Executed {
                event_seqs: vec![43],
            }),
            inventory: inventory_state(),
            reward: RewardBreakdown::new(dec!(0.5), dec!(-0.012), dec!(-0.05)),
            terminal,
            end_reason,
            quality: QualityFlags::NONE,
        }
    }

    #[test]
    fn step_record_roundtrip_serde() {
        let step = step(true, Some(EndReason::MarketResolved));
        let s = serde_json::to_string(&step).unwrap();
        let back: StepRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(back, step);
        assert_eq!(back.reward.total, dec!(0.438));
    }

    /// Contract D: a terminal step requires `end_reason`; a non-terminal step
    /// forbids it — `validate` detects either inconsistency.
    #[test]
    fn terminal_step_requires_consistent_end_reason() {
        assert!(step(true, Some(EndReason::HorizonReached))
            .validate()
            .is_ok());
        assert!(step(false, None).validate().is_ok());
        assert_eq!(
            step(true, None).validate(),
            Err(ContractError::InconsistentEnd { terminal: true })
        );
        assert_eq!(
            step(false, Some(EndReason::Manual)).validate(),
            Err(ContractError::InconsistentEnd { terminal: false })
        );
    }

    #[test]
    fn reward_total_is_exact_sum_of_components() {
        let r = RewardBreakdown::new(dec!(1.25), dec!(-0.03), dec!(-0.10));
        assert_eq!(r.total, dec!(1.12));
        r.validate().unwrap();
        // A tampered total breaks the invariant.
        let mut r2 = r.clone();
        r2.total = dec!(9.99);
        assert_eq!(
            r2.validate(),
            Err(ContractError::RewardMismatch {
                expected: dec!(1.12),
                total: dec!(9.99)
            })
        );
    }

    #[test]
    fn episode_roundtrip_serde() {
        let ep = Episode {
            schema_version: SCHEMA_VERSION,
            episode_id: "ep-0001".into(),
            market_cid: Some("0xcid".into()),
            asset: Some("BTC".into()),
            started_at_ms: 1_700_000_000_000,
            ended_at_ms: Some(1_700_000_100_000),
            steps: vec![step(true, Some(EndReason::MarketResolved))],
            end_reason: Some(EndReason::MarketResolved),
            quality: QualityFlags::NONE,
        };
        let s = serde_json::to_string(&ep).unwrap();
        let back: Episode = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ep);
    }

    // ── E — ResearchManifest ────────────────────────────────────────────

    #[test]
    fn manifest_roundtrip_serde() {
        let m = ResearchManifest {
            schema_version: SCHEMA_VERSION,
            created_at_ms: 1_700_000_000_000,
            code_hash: "abc123".into(),
            config_hash: "def456".into(),
            model_hash: None,
            sim_profile_hash: Some("feedface".into()),
            seeds: vec![42, 0xDEAD_BEEF],
            splits: vec![
                SplitSpec {
                    name: "train".into(),
                    from_ms: 0,
                    to_ms: 1_000,
                },
                SplitSpec {
                    name: "test".into(),
                    from_ms: 1_000,
                    to_ms: 2_000,
                },
            ],
            research_dir: "research/2026-09-paper".into(),
            files: vec![ManifestFile {
                path: "events/day1.jsonl".into(),
                sha256: "00ff".into(),
            }],
        };
        let s = serde_json::to_string(&m).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["schema_version"], json!(SCHEMA_VERSION));
        assert_eq!(v["model_hash"], Value::Null);
        let back: ResearchManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
    }
}
