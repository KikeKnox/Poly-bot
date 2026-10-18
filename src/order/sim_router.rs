//! SimulatedRouter — paper router with an honest queue model.
//!
//! Models the following differences from Polymarket's real CLOB:
//!
//! - **GTD queue with position** (Nautilus `OrderMatchingEngine` model for
//!   L2 data without a trade tape): the order enters behind the displayed size at
//!   its level on the side where it rests (`queue_ahead`), which only shrinks on
//!   evidence of consumption — a decrease of the displayed size at the level between
//!   snapshots, flow crossing our price, and depth taken by our own
//!   takers. It is fill-eligible only with `queue_ahead == 0`, and the fill is
//!   capped at the flow excess over the queue (partials with a live remainder).
//! - **Latency + FOK rejection**: pauses `paper_latency_ms` before processing;
//!   if the price moved against us, the FOK is rejected. The residual rate
//!   `paper_fok_fill_rate` models insufficient liquidity (critical for ARB).
//! - **Taker fee**: FOK/Market buys and sells pay `rate × p × (1−p)` (parabolic,
//!   the contract's real formula for crypto up/down, confirmed
//!   against the Gamma API `feeSchedule` on 6/6 assets and against the official
//!   fee table at docs.polymarket.com/trading/fees); `rate` is the market's
//!   `parabolic_fee_rate` if the book carries it, otherwise it falls back to
//!   `paper_taker_fee_rate` (it used to use the triangular `min(p,1−p)`,
//!   which is the contract's `MAX_FEE_RATE_BIPS` — a ceiling, not the fee charged
//!   — and overcharged by up to 186% at p=0.50, biasing all historical paper PnL
//!   downward, more in the middle zone than at the extremes). The fee
//!   is charged QUANTIZED to 5 decimals per fill segment (the venue's official precision)
//!   and the rate can be frozen per market+validity window with
//!   `set_fee_windows` — in that case the schedule decides, not the book.
//! - **Size truncation**: GTD to whole shares; sell to 2dp — same as live.
//! - **Partial fill on sell**: limits the fill to the depth available at the bid.
//! - **Expired GTD cleanup**: replicates the CLOB's auto-cancellation at clear_at.
//! - **Own balance**: deducts buys and credits sells; rejects for insufficient funds.
//! - **Accounting**: `credit_redemption` puts the payout of a
//!   resolution into the paper cash (idempotent by `redeem_id`) and
//!   `accounting()` reconstructs cash/reserves/inventory from the log of
//!   executions + live orders — micro-level conservation is verifiable.
//!
//! **Paper/live parity:** this router never returns a synchronous `filled()`.
//! Instead it emits `UserTrade MATCHED`, which goes through the same `handle_trade`
//! as the real user WS — calibration exercises the same code as production.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashSet;
use parking_lot::Mutex;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{mpsc, watch};

use crate::{
    market_feed::{handle_trade, MarketRegistry, UserTrade},
    order::router::{
        OrderKind, OrderRequest, OrderRouter, OpenOrderSide, RouterOpenOrder, RouterOrderStatus,
        RouterResult,
        RouterTrade, TimeInForce,
    },
    types::{BookSnapshot, Fill, FillSide, Mode, TokenBook},
};

// ── State per pending GTD limit order ────────────────────────────────────────

#[derive(Debug, Clone)]
struct PendingLimit {
    req: OrderRequest,
    /// EXTERNAL queue ahead (queue position, Nautilus L2 model): displayed size
    /// at our level on the side where we rest, on acceptance.
    /// Our own earlier resting orders are NOT baked in here — they are added live
    /// each tick from their remainders (baked in they went stale after
    /// a partial and the post-trade L2 delta discounted the same
    /// consumption again → the same flow filled two of our own orders). It shrinks by (a)
    /// a decrease of the displayed size at the level between snapshots (evidence
    /// of consumption without a trade tape), (b) flow crossing our price —
    /// it first empties the external queue, then our earlier own orders, the excess
    /// executes us — and (c) depth that our own takers take from the level
    /// (`consume_depth`).
    queue_ahead: Decimal,
    /// RAW displayed size at our level expected for the NEXT
    /// snapshot: `current_level − min(current_level, cross)` — flow already
    /// absorbed by the cross is not discounted again as an L2 delta.
    /// An increase does NOT fatten the queue: whoever arrives at the level
    /// after us goes behind, never in front.
    prev_level: Decimal,
    /// Arrival order among our own resting orders — serializes our own queue at the
    /// same level: those with lower `seq` go in front (their live remainder is added
    /// live to the queue ahead of the later ones).
    seq: u64,
    /// Shares already executed for this order (partial fills with a
    /// live remainder — a partial used to remove the order from `pending`).
    filled: Decimal,
    /// Cost per share reserved on entering the book (buy GTD):
    /// `price + estimated_fee` for non-`post_only` orders (on fill they cross the
    /// spread and pay taker), plain `price` for `post_only`
    /// (genuine maker, fee 0). This way the reserve covers the REAL cost of the fill
    /// and an overdraft on fill because of the fee is no longer possible (the balance's
    /// clamp to 0 used to absorb that deficit — it "created cash").
    /// 0 for `LimitSell` (they reserve no cash: they sell inventory — with
    /// `track_inventory` they reserve SHARES via `reserved_sell`, see below).
    reserved_unit: Decimal,
    /// Instant (injectable clock) at which the order entered the book —
    /// age of the resting order for the stale/adverse-selection mark on maker
    /// fills.
    since_ms: i64,
    /// The evidence for this order became degraded — a level shrink
    /// without a cross larger than the queue ahead (ambiguous consumption: trades that
    /// reached us or cancels behind us, indistinguishable without a tape) or a post-gap
    /// tick with abstention. Later fills come out as
    /// `SimConfidence::Stressed`. Sticky per order: once the
    /// queue-position trail is lost it is not recovered.
    degraded_evidence: bool,
}

/// Fill plan for a resting order decided in the `retain` of `tick_book`
/// (emission happens outside the `pending` lock).
struct FillPlan {
    order_id: String,
    req: OrderRequest,
    size: Decimal,
    in_asks: bool,
    reserved_unit: Decimal,
    seq: u64,
    /// Age of the resting order at the tick (for the maker stale mark).
    since_ms: i64,
    /// Confidence of the plan BEFORE the maker stale check (degraded queue or
    /// post-gap tick).
    confidence: SimConfidence,
    /// Causal evidence of the fill.
    evidence: SimEvidence,
}

// ── Clock, execution records and residual depth ────────────────────────────────────────

/// Injectable clock, in unix ms. In production it is set to
/// `Utc::now().timestamp_millis()`; in tests a fixed closure makes the replay
/// deterministic: same seed + same clock ⇒ same fills and same
/// recorded executions.
pub type SimClock = Arc<dyn Fn() -> i64 + Send + Sync>;

/// The fee rates used by the sim tests are FIXTURES
/// citing the official source — 0.07 is the crypto taker rate from the official
/// fee table (<https://docs.polymarket.com/trading/fees>)
/// and from the Gamma API `feeSchedule.rate` (`crypto_fees_v2`);
/// 0.04 is the Finance/Politics rate from the same table; 0 is the documented
/// scenario for fee-free markets (geopolitics).
/// Rebates (maker 20%, taker tiers) are not confirmed per execution → the
/// conservative scenario excludes them. This constant is `true` only because that
/// source is cited here.
pub const FEE_FIXTURES_OFFICIALLY_VERIFIED: bool = true;

/// Segment of an execution against one book level: the level's NEGOTIATED
/// price (no fee) and the fee charged for that segment — internal
/// accounting separates price from friction even though the `Fill` still
/// carries the all-in average price (`cash/shares`), because all of its
/// consumers compute cost as `price × size`.
#[derive(Debug, Clone, PartialEq)]
pub struct SimExecLevel {
    pub price: Decimal,
    pub size: Decimal,
    /// Fee charged for the segment, quantized to 5 decimals (the venue's
    /// official precision — see `fair_value::fee_charged_5dp`).
    pub fee_usdc: Decimal,
    /// Spread/slippage ALREADY REALIZED in the segment — difference between the
    /// level's negotiated price and the slipped price, times `size`. Kept
    /// separate from the fee and the negotiated price (never charged twice:
    /// `cash = notional ± fee ± impact`, see `SimExecution::cost_breakdown`).
    /// 0 on maker fills (they execute at their limit, no slippage).
    pub impact_usdc: Decimal,
    /// The executed level was a PERSISTENT quote out of consensus when
    /// filled (displayed ≥ `quote_stale_ms` unchanged and deviating from the
    /// pair's cross consensus by more than `consensus_margin`) — that fill is not
    /// free liquidity, it is stale/adverse selection.
    pub stale: bool,
}

/// Economic role of the execution — who pays the fee according to the REAL CROSS
/// at the fill (not just the order's flag): marketable/taker executions
/// pay `rate × p × (1−p)`; `post_only` resting orders and maker asks are
/// genuine makers (fee 0 — "Makers are never charged fees", official docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimRole {
    Taker,
    Maker,
}

/// Cost breakdown of an execution — negotiated price, fee,
/// realized impact and penalty, kept separate (the episode's reward
/// decomposes the same way; here `penalty_usdc` is always 0 because
/// risk penalties live in the episode, not in the fill).
/// Invariant: `cash = notional + fee + impact` on buys and
/// `cash = notional − fee − impact` on sells; `residual_usdc` measures the
/// violation (must be 0).
#[derive(Debug, Clone, PartialEq)]
pub struct CostBreakdown {
    /// Negotiated price: `Σ price × size` over book levels (no fee or
    /// slippage).
    pub traded_notional: Decimal,
    /// Venue fee charged on the fill (quantized to 5dp per segment).
    pub fee_usdc: Decimal,
    /// Spread/impact ALREADY realized on the fill (slipped price vs. negotiated).
    pub impact_usdc: Decimal,
    /// Always 0 at the execution level — penalties belong to the episode's reward
    /// (`RewardBreakdown::risk_penalty`), not to the fill.
    pub penalty_usdc: Decimal,
    /// Cash moved by the execution (what enters/leaves the balance).
    pub cash: Decimal,
    /// Violation of the invariant `cash = notional ± fee ± impact` — must be
    /// 0; any non-zero value is an accounting bug.
    pub residual_usdc: Decimal,
    /// `true` if the execution was maker according to the real cross (`role`).
    pub is_maker: bool,
}

/// Conservation snapshot of the sim — live state vs. state
/// reconstructed from the execution log, live orders and
/// applied redemptions (`SimulatedRouter::accounting`).
///
/// Expected drifts are EXACT (Decimal): any difference is an
/// accounting bug — created cash, an unreleased reserve or fabricated inventory.
#[derive(Debug, Clone)]
pub struct SimAccounting {
    /// Balance the sim started with.
    pub initial_balance: Decimal,
    /// Later contributions via `add_funds`.
    pub contributions_usdc: Decimal,
    /// Resolution payouts credited via `credit_redemption`.
    pub redemptions_usdc: Decimal,
    /// Current live balance.
    pub balance: Decimal,
    /// Reconstructed balance:
    /// `initial + contributions + redemptions + Σsell_cash − Σbuy_cash`.
    pub expected_balance: Decimal,
    /// Cash reserved by live buy GTDs (current state).
    pub reserved_buy: Decimal,
    /// Expected: `Σ reserved_unit × remainder` per live order — the unit
    /// stored at submit (same cost basis the fill charges).
    pub expected_reserved_buy: Decimal,
    /// Shares reserved by live `LimitSell`s per token (current).
    pub reserved_sell: HashMap<String, Decimal>,
    /// Expected: `Σ remainder` per live `LimitSell` per token.
    pub expected_reserved_sell: HashMap<String, Decimal>,
    /// Live inventory per token (zero entries omitted).
    pub inventory_state: HashMap<String, Decimal>,
    /// Expected: `Σ (buys − sells) − redeemed` per token.
    pub expected_inventory: HashMap<String, Decimal>,
    /// Executions recorded in the log.
    pub executions: usize,
}

impl SimAccounting {
    /// Cash drift: `balance − expected_balance` (0 = conserved).
    pub fn cash_drift(&self) -> Decimal {
        self.balance - self.expected_balance
    }

    /// Buy-reserve drift (0 = every GTD reserves exactly its
    /// stored unit times the live remainder).
    pub fn buy_reserve_drift(&self) -> Decimal {
        self.reserved_buy - self.expected_reserved_buy
    }

    /// Sell-reserve drifts per token — non-zero entries only.
    pub fn sell_reserve_drift(&self) -> HashMap<String, Decimal> {
        let mut out = HashMap::new();
        for k in self
            .reserved_sell
            .keys()
            .chain(self.expected_reserved_sell.keys())
        {
            let d = self.reserved_sell.get(k).copied().unwrap_or_default()
                - self
                    .expected_reserved_sell
                    .get(k)
                    .copied()
                    .unwrap_or_default();
            if d != Decimal::ZERO {
                out.insert(k.clone(), d);
            }
        }
        out
    }

    /// Inventory drifts per token — non-zero entries only.
    pub fn inventory_drift(&self) -> HashMap<String, Decimal> {
        let mut out = HashMap::new();
        for k in self.inventory_state.keys().chain(self.expected_inventory.keys()) {
            let d = self.inventory_state.get(k).copied().unwrap_or_default()
                - self.expected_inventory.get(k).copied().unwrap_or_default();
            if d != Decimal::ZERO {
                out.insert(k.clone(), d);
            }
        }
        out
    }

    /// Micro-level conservation: reconstructed cash, reserves and inventory ==
    /// live state. Any `false` flags fabricated or lost
    /// cash/reserves/inventory.
    pub fn is_consistent(&self) -> bool {
        self.cash_drift() == Decimal::ZERO
            && self.buy_reserve_drift() == Decimal::ZERO
            && self.sell_reserve_drift().is_empty()
            && self.inventory_drift().is_empty()
    }
}

/// Causal evidence that produced an execution — which book observation
/// justified it. Together with `levels`/`notional`/`fee_usdc`/`cash` (the costs)
/// and `hypothesis` (the applied queue scenario) it gives the
/// evidence+hypothesis+costs traceability required for each simulated fill.
#[derive(Debug, Clone, PartialEq)]
pub enum SimEvidence {
    /// Taker/marketable: walk through the RESIDUAL depth visible at the
    /// instant the request reached the simulated exchange (`levels`
    /// walked, capped at the order's limit and at the profile's
    /// participation). Includes FOK/FAK/Market, `submit_pair` legs, the
    /// marketable part of a GTD at submit, and taker sells.
    DepthAtArrival { levels: usize },
    /// Resting FIFO: the tick's crossed flow (`cross` shares on the opposite
    /// side at/through our limit, already net of what our takers
    /// took from THIS snapshot) exceeded the external queue ahead
    /// (`queue_ahead`, at the start of the tick, before the level shrink) plus
    /// our earlier own resting orders (`own_ahead`); the fill is the
    /// excess capped at the remainder and at the profile's participation.
    CrossBehindQueue {
        cross: Decimal,
        queue_ahead: Decimal,
        own_ahead: Decimal,
    },
}

/// Queue hypothesis under which the execution was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimHypothesis {
    /// Taker: the residual depth on arrival is what executes; maker/taker
    /// decided by what really crossed.
    TakerDepth,
    /// Resting FIFO: crossed flow first empties the external queue, then
    /// our earlier own orders, and the excess executes us. Level shrinks
    /// without a cross advance the queue according to the profile's `cancel_attr` (κ); a
    /// shrink larger than the queue ahead is ambiguous consumption → the order's
    /// later fills become `Stressed`.
    QueueExcess,
}

/// Confidence of the execution — a FIDELITY mark, not an accounting one:
/// the recorded cost is real in both cases; `Stressed` indicates the
/// evidence that produced the fill is degraded (stale quote out of
/// consensus, tape gap or ambiguous consumption) and the execution should weigh less
/// — or be excluded — in edge/calibration metrics. Never used to
/// "improve" a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimConfidence {
    Normal,
    /// Low confidence due to degraded evidence.
    Stressed,
}

/// Record of a simulator execution.
///
/// An order can produce N executions (partial GTD/FAK): `exec_id` is
/// unique per execution and NEVER equal to `order_id`; `local_order_id` is copied
/// from the `OrderRequest` (local identity = reserve persisted pre-submit).
/// `cash` is what moves in the sim's cash: on buys
/// `notional + fee (+ slippage)`; on sells `notional − fee (− slippage)`.
#[derive(Debug, Clone, PartialEq)]
pub struct SimExecution {
    pub exec_id: String,
    pub order_id: String,
    pub local_order_id: Option<String>,
    pub token_id: String,
    pub side: FillSide,
    pub shares: Decimal,
    pub notional: Decimal,
    pub fee_usdc: Decimal,
    pub cash: Decimal,
    pub levels: Vec<SimExecLevel>,
    /// Instant of the execution (injectable clock) — used by `trades_since`
    /// to match taker Unknowns by time window, like the
    /// `match_time` of CLOB trades.
    pub ts_ms: i64,
    /// Causal evidence of the fill (which book observation produced it).
    pub evidence: SimEvidence,
    /// Queue-model hypothesis applied to this execution.
    pub hypothesis: SimHypothesis,
    /// Confidence — `Stressed` for fills under degraded evidence
    /// (stale quote out of consensus, tape gap, ambiguous consumption).
    pub confidence: SimConfidence,
    /// Economic role according to the real cross at the fill (see [`SimRole`]).
    pub role: SimRole,
}

impl SimExecution {
    /// Cost breakdown of the execution — negotiated price, fee,
    /// realized impact and penalty (0 at this level), kept separate. The
    /// `residual_usdc` is the violation of the invariant
    /// `cash = notional ± fee ± impact` measured AT LEDGER RESOLUTION
    /// (micro-USDC per level, see `cash_ledger`): the components keep
    /// full precision as evidence, but equality is checked at the
    /// granularity the ledger can settle — so the residual still
    /// closes at 0 in healthy runs and is non-zero only if there is a real
    /// accounting bug (or the defensive floor on sells with fee > notional, which the
    /// residual reveals with its sign).
    pub fn cost_breakdown(&self) -> CostBreakdown {
        let impact: Decimal = self.levels.iter().map(|l| l.impact_usdc).sum();
        // Per-level components quantized to the micro — the same expression as
        // `leg_cash` (`price*size + fee + impact == slipped*size + fee`), so
        // absent clamps the difference with `cash` is exactly 0.
        let components: Decimal = match self.side {
            FillSide::Buy => self
                .levels
                .iter()
                .map(|l| cash_ledger(l.price * l.size + l.fee_usdc + l.impact_usdc))
                .sum(),
            _ => self
                .levels
                .iter()
                .map(|l| cash_ledger(l.price * l.size - l.fee_usdc - l.impact_usdc))
                .sum(),
        };
        let residual = match self.side {
            FillSide::Buy => self.cash - components,
            _ => components - self.cash,
        };
        CostBreakdown {
            traded_notional: self.notional,
            fee_usdc: self.fee_usdc,
            impact_usdc: impact,
            penalty_usdc: Decimal::ZERO,
            cash: self.cash,
            residual_usdc: residual,
            is_maker: self.role == SimRole::Maker,
        }
    }
}

/// Residual depth per token within the CURRENT snapshot: what
/// a taker already took from this book is not available again until
/// a book with different content arrives. The side's fingerprint decides whether the
/// incoming snapshot restores depth or is the same one (a re-tick must not
/// re-consume the GTD queue nor replenish what was taken).
#[derive(Debug, Default)]
struct DepthState {
    fingerprint: Option<u64>,
    asks_taken: HashMap<Decimal, Decimal>,
    bids_taken: HashMap<Decimal, Decimal>,
}

/// Fingerprint of the content of the book side corresponding to a token
/// (bid+ask levels with price and size, including scale — a renormalized
/// book counts as different).
fn fingerprint_side(side: &TokenBook) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for l in side.bids.levels.iter().chain(side.asks.levels.iter()) {
        l.price.mantissa().hash(&mut h);
        l.price.scale().hash(&mut h);
        l.size.mantissa().hash(&mut h);
        l.size.scale().hash(&mut h);
    }
    h.finish()
}

/// Segment of a taker's walk through the depth.
struct WalkLeg {
    /// Negotiated price of the level (the book's).
    price: Decimal,
    size: Decimal,
    /// SLIPPED price of the level (`slippage` applied) — basis of the fee and
    /// of the realized impact (`|slipped − price| × size`).
    slipped: Decimal,
    /// Fee per share, UNquantized, computed over the already slipped price —
    /// quantization to 5dp is applied per segment in `leg_fee` (the venue's official
    /// precision).
    fee_share: Decimal,
    /// The level was a persistent quote out of consensus when taken
    /// (stale/adverse selection — see `leg_stale`).
    stale: bool,
}

/// (price, size) pairs of a walk — for discounting depth.
fn legs_pairs(legs: &[WalkLeg]) -> Vec<(Decimal, Decimal)> {
    legs.iter().map(|l| (l.price, l.size)).collect()
}

/// Quantizes an amount to the resolution of the durable ledger —
/// the same `to_micros`/`from_micros` pair with which `StateStore` writes
/// `transactions`/`positions` (6dp, round-half-away per `to_micros`).
/// All the cash the sim moves per wallet goes through here so that the
/// reconciliation `cash_drift == 0` is exact, not "almost".
fn cash_ledger(d: Decimal) -> Decimal {
    crate::state::from_micros(crate::state::to_micros(d))
}

// ── SimulatedRouter ───────────────────────────────────────────────────────────

pub struct SimulatedRouter {
    slippage_bps: u32,
    /// Fill probability for FOK (0.0-1.0).
    fok_fill_rate: f64,
    /// Per-strategy override of `fok_fill_rate` (ARB-15M5M: 1.0 = the book
    /// decides, no draw). Empty = the usual behavior for everyone.
    fok_fill_rate_per_strategy: Mutex<HashMap<crate::types::StrategyId, f64>>,
    /// Simulated network latency in ms.
    latency_ms: u64,
    /// Taker fee rate: `fee = rate × p × (1−p)` per share — same model
    /// as Polymarket live. It used to be a flat bps amount, which
    /// overcharged entries at high p (NC/NE) and undercharged at low p.
    taker_fee_rate: Decimal,
    /// Correlation of the liquidity rejection between the two legs of a submit_pair.
    /// On the real CLOB post_orders evaluates YES/NO at the same instant:
    /// with probability `correlation` both legs share the same draw.
    pair_fill_correlation: f64,
    order_seq: AtomicU64,
    /// Exec id counter — each execution (including partials
    /// of the same order) has its own identity, distinct from the order_id.
    exec_seq: AtomicU64,
    /// hedge_id counter PER INSTANCE — it used to be `static`, so
    /// two runs in the same process did not produce the same ids.
    hedge_seq: AtomicU64,
    /// Arrival order of resting orders into `pending` — serializes the
    /// shared queue when two own GTDs rest at the same level.
    resting_seq: AtomicU64,
    /// Accept/reject RNG DECOUPLED from `order_seq` — `draw_uniform` used to
    /// reuse the same counter as `next_order_id`, so the fill/reject
    /// sequence was a deterministic and PREDICTABLE function of the order number
    /// (not iid), biasing any analysis that crossed both
    /// (e.g. "does rejection correlate with arrival order?" — the answer was
    /// trivially yes, by construction).
    rng_state: AtomicU64,
    /// Chaos sim: probability that an ACCEPTED submit returns `unknown`
    /// instead of its real result — models the CLOB's lost
    /// ACK (the order may have executed on the venue but the
    /// response never arrived). 0.0 = deterministic (default). Hot-mutable
    /// via `set_unknown_rate` so the constructor need not be rebuilt.
    unknown_rate: Mutex<f64>,
    pending: Mutex<HashMap<String, PendingLimit>>,
    /// Last known book per token_id; updated on every tick_book.
    last_books: Mutex<HashMap<String, BookSnapshot>>,
    /// Residual depth per token within the current snapshot
    /// (see `DepthState`/`fingerprint_side`).
    depth: Mutex<HashMap<String, DepthState>>,
    /// Cash committed by live buy GTDs. New orders
    /// can only spend `balance − reserved_buy`; a fill consumes the matching part
    /// and a cancel/expiry releases ONLY that order's remainder
    /// — canceling one never releases cash reserved by another.
    reserved_buy: Mutex<Decimal>,
    /// Shares committed by live resting `LimitSell`s per token —
    /// mirror of `reserved_buy` on the inventory basis. Only consulted with
    /// `track_inventory` on (research profile); the accounting
    /// is always kept so that enabling the knob hot does not corrupt
    /// the state.
    reserved_sell: Mutex<HashMap<String, Decimal>>,
    /// Simulated inventory per token (shares). Always accounted
    /// (+buys, −sells); with `track_inventory` it is additionally REQUIRED on sells
    /// — funds/reserves/inventory are never fabricated. Without the knob
    /// sells do not consult it (parity with the operational profile).
    inventory: Mutex<HashMap<String, Decimal>>,
    /// Resting orders that left the book WITHOUT being fully filled
    /// (cancel/expiry) — `order_id → shares executed up to that moment`.
    /// On live the venue reports them CANCELED/expired, not "does not exist": without
    /// this mapping `order_status` returned `NotFound` ("the POST never arrived")
    /// for an order that did exist — a distinction that matters to the
    /// Unknown reconciler.
    gone: Mutex<HashMap<String, Decimal>>,
    /// Persistence of displayed quotes — `(token_id, in_asks, price)`
    /// → `(from_ms, size)`. A quote that reappears or changes size is a
    /// NEW quote (reset); only persistence with unchanged size counts
    /// for the stale mark (adverse selection).
    shown_since: Mutex<HashMap<(String, bool, Decimal), (i64, Decimal)>>,
    /// Last `tick_book` per token (injectable clock) — basis of the
    /// tape-gap detection (`gap_ms` of the research profile).
    last_tick_ms: Mutex<HashMap<String, i64>>,
    /// Active research profile — see
    /// [`crate::config::ResearchSimConfig`]. The default reproduces the
    /// operational behavior (`nominal`); it only changes via
    /// `apply_research_profile`, which the production bot never calls.
    research: Mutex<crate::config::ResearchSimConfig>,
    /// Log of executions, in emission order (see `executions()`).
    exec_log: Mutex<Vec<SimExecution>>,
    /// Injectable clock (unix ms) — GTD purge and hedge_ids.
    clock: SimClock,
    /// Startup nonce taken from the injected clock ONCE in `with_deps`.
    /// The sim's `order_id`/`exec_id` carry per-process counters — without
    /// this prefix, a restart reuses `sim-exec-0`, `sim-exec-1`, … which
    /// are already in `fills.exec_id` (UNIQUE index) and the durable dedup swallows
    /// every new fill as a "duplicate" without persisting or re-emitting: the sim's
    /// wallet debits the reserve but the position is never created (wallet↔DB
    /// accounting divergence observed in paper production). In replay
    /// it is deterministic (same `from_ms` ⇒ same ids); at runtime it is the
    /// startup timestamp, different on each process.
    boot_ms: i64,
    // ── paper/live parity ────────────────────────────────────────
    /// Channel → fill_sink: same destination as user_ws on live.
    fill_tx: mpsc::UnboundedSender<Fill>,
    /// Shared registry: the same one market_feed uses on live for neg_risk/clear_at.
    registry: Arc<MarketRegistry>,
    /// USDC cash available for buys.
    // No atomic multi-order reserve → theoretical overdraft under high concurrency;
    // acceptable in paper (the hard control is RiskGuard/executor exposure, not the balance).
    balance: Mutex<Decimal>,
    /// Wallet balance channel (→ UI/API); same one live uses with fetch_balance.
    wallet_bal_tx: watch::Sender<Option<Decimal>>,
    /// Empty set passed to handle_trade in paper mode.
    /// The known_order_ids filter only applies in Mode::Live, so it is never consulted.
    _paper_order_ids: DashSet<String>,
    /// Initial balance — basis of the cash reconstruction in
    /// `accounting` (`initial + contributions + redemptions + sells − buys`).
    initial_balance: Decimal,
    /// Later contributions via `add_funds` — they enter the cash reconstruction
    /// and the return reconciliation (they are not trading PnL).
    contributions: Mutex<Decimal>,
    /// Fee schedule FROZEN per market (condition_id) — if it exists and
    /// a window covers the fill instant, its rate decides instead of the
    /// book's `parabolic_fee_rate` (`set_fee_windows`). Reproducibility:
    /// the validity lives in the fixture, not in the tape.
    fee_windows: Mutex<HashMap<String, Vec<crate::research::contract::RulesWindow>>>,
    /// Applied redemptions — `(redeem_id, token_id, shares, payout)`.
    /// The full log (not just the ids) feeds the per-token reconstruction of
    /// `accounting`.
    redemptions: Mutex<Vec<(String, String, Decimal, Decimal)>>,
    /// Redemption ids already applied — dedup for `credit_redemption`
    /// (idempotent resolution: the same `redeem_id` does not credit twice).
    redeem_ids: Mutex<HashSet<String>>,
}

impl SimulatedRouter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slippage_bps: u32,
        fok_fill_rate: f64,
        latency_ms: u64,
        taker_fee_rate: Decimal,
        pair_fill_correlation: f64,
        fill_tx: mpsc::UnboundedSender<Fill>,
        registry: Arc<MarketRegistry>,
        initial_balance: Decimal,
        wallet_bal_tx: watch::Sender<Option<Decimal>>,
    ) -> Arc<Self> {
        // Own seed, independent of `order_seq` (which always starts at 0).
        // Derived from the clock so the SAME accept/reject sequence is not
        // repeated on every calibration run.
        let rng_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x2545_F491_4F6C_DD1D)
            ^ 0x9E37_79B9_7F4A_7C15;
        let clock: SimClock = Arc::new(|| chrono::Utc::now().timestamp_millis());
        Self::with_deps(
            slippage_bps,
            fok_fill_rate,
            latency_ms,
            taker_fee_rate,
            pair_fill_correlation,
            fill_tx,
            registry,
            initial_balance,
            wallet_bal_tx,
            rng_seed,
            clock,
        )
    }

    /// Constructor with injectable dependencies — RNG seed and
    /// clock — for deterministic replay: same seed + same clock ⇒
    /// same draws, same fills and same recorded executions.
    #[allow(clippy::too_many_arguments)]
    pub fn with_deps(
        slippage_bps: u32,
        fok_fill_rate: f64,
        latency_ms: u64,
        taker_fee_rate: Decimal,
        pair_fill_correlation: f64,
        fill_tx: mpsc::UnboundedSender<Fill>,
        registry: Arc<MarketRegistry>,
        initial_balance: Decimal,
        wallet_bal_tx: watch::Sender<Option<Decimal>>,
        rng_seed: u64,
        clock: SimClock,
    ) -> Arc<Self> {
        let _ = wallet_bal_tx.send(Some(initial_balance));
        let boot_ms = clock();
        Arc::new(Self {
            slippage_bps,
            fok_fill_rate,
            latency_ms,
            taker_fee_rate,
            pair_fill_correlation,
            order_seq: AtomicU64::new(0),
            exec_seq: AtomicU64::new(0),
            hedge_seq: AtomicU64::new(0),
            resting_seq: AtomicU64::new(0),
            rng_state: AtomicU64::new(rng_seed),
            boot_ms,
            unknown_rate: Mutex::new(0.0),
            pending: Mutex::new(HashMap::new()),
            last_books: Mutex::new(HashMap::new()),
            depth: Mutex::new(HashMap::new()),
            reserved_buy: Mutex::new(Decimal::ZERO),
            reserved_sell: Mutex::new(HashMap::new()),
            fok_fill_rate_per_strategy: Mutex::new(HashMap::new()),
            inventory: Mutex::new(HashMap::new()),
            gone: Mutex::new(HashMap::new()),
            shown_since: Mutex::new(HashMap::new()),
            last_tick_ms: Mutex::new(HashMap::new()),
            research: Mutex::new(crate::config::ResearchSimConfig::default()),
            exec_log: Mutex::new(Vec::new()),
            clock,
            fill_tx,
            registry,
            balance: Mutex::new(initial_balance),
            wallet_bal_tx,
            _paper_order_ids: DashSet::new(),
            initial_balance,
            contributions: Mutex::new(Decimal::ZERO),
            fee_windows: Mutex::new(HashMap::new()),
            redemptions: Mutex::new(Vec::new()),
            redeem_ids: Mutex::new(HashSet::new()),
        })
    }

    /// Recorded executions, in emission order — the sim's audit
    /// log with negotiated price, fee and cash kept separate.
    pub fn executions(&self) -> Vec<SimExecution> {
        self.exec_log.lock().clone()
    }

    /// Adds funds to the paper balance (automatic top-up when it falls below the threshold).
    /// The contribution is accounted — it enters the cash reconstruction
    /// of `accounting` and the return reconciliation (it is not PnL).
    pub fn add_funds(&self, amount: Decimal) {
        *self.contributions.lock() += amount;
        let mut bal = self.balance.lock();
        *bal += amount;
        let _ = self.wallet_bal_tx.send(Some(*bal));
    }

    /// Freezes the fee schedule of market `market_cid` — validity windows
    /// validated by [`crate::fair_value::validate_fee_windows`]
    /// (sorted, non-overlapping, rates in [0,1]; a negative rate would be an
    /// unconfirmed rebate and is rejected). While a window covers the
    /// fill instant, its rate decides instead of the book's `parabolic_fee_rate` —
    /// reproducibility: the validity lives in the fixture, not in the tape.
    /// Outside a window (or with rate `None`) the previous behavior is used.
    pub fn set_fee_windows(
        &self,
        market_cid: &str,
        windows: Vec<crate::research::contract::RulesWindow>,
    ) -> anyhow::Result<()> {
        crate::fair_value::validate_fee_windows(&windows)
            .map_err(|e| anyhow::anyhow!("invalid fee schedule for {market_cid}: {e}"))?;
        self.fee_windows
            .lock()
            .insert(market_cid.to_string(), windows);
        Ok(())
    }

    /// Credits the payout of a resolution into the paper cash and debits the
    /// shares from the inventory — the counterpart of `StateStore::apply_settlement`
    /// (which moves the durable ledger; the sim's cash belongs to the router).
    ///
    /// Idempotent by `redeem_id` — same role as `exec_id` in the fill
    /// dedup: returns `Ok(false)` if the id was already applied, moving nothing. The
    /// caller must pass a stable id per resolution (e.g.
    /// `"settle:{cid}:{token}:{strategy}"`); on live the on-chain redeem is done
    /// by the chain, here this call models it.
    pub fn credit_redemption(
        &self,
        redeem_id: &str,
        token_id: &str,
        shares: Decimal,
        payout: Decimal,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            shares >= Decimal::ZERO && payout >= Decimal::ZERO,
            "redemption with negative shares/payout ({shares}/{payout})"
        );
        if !self.redeem_ids.lock().insert(redeem_id.to_string()) {
            // The dedup is only benign if the payload matches — a retry
            // with a DIFFERENT (token, shares, payout) under the same id is a caller
            // bug, not a repeated resolution: it is ignored but
            // reported (never silent).
            let previous = self
                .redemptions
                .lock()
                .iter()
                .find(|(id, ..)| id.as_str() == redeem_id)
                .cloned();
            if let Some((_, tok, sh, pay)) = previous {
                if tok != token_id || sh != shares || pay != payout {
                    tracing::warn!(
                        redeem_id,
                        previous = %format!("{tok}/{sh}/{pay}"),
                        new = %format!("{token_id}/{shares}/{payout}"),
                        "credit_redemption: dedup with divergent payload — ignored"
                    );
                }
            }
            return Ok(false);
        }
        {
            let mut bal = self.balance.lock();
            *bal += payout;
            let _ = self.wallet_bal_tx.send(Some(*bal));
        }
        // Inventory may go negative if more is redeemed than counted —
        // it is recorded anyway: the drift stays visible in `accounting`
        // (not clamped, just as the balance may go negative).
        *self
            .inventory
            .lock()
            .entry(token_id.to_string())
            .or_default() -= shares;
        self.redemptions
            .lock()
            .push((redeem_id.to_string(), token_id.to_string(), shares, payout));
        Ok(true)
    }

    /// Conservation snapshot — reconstructs cash, reserves and
    /// inventory FROM the execution log + live orders + redemptions,
    /// and compares them with the live state. `is_consistent()` demands zero drift:
    /// `balance = initial + contributions + redemptions + Σsell_cash − Σbuy_cash`,
    /// `reserved_buy = Σ reserved_unit × remainder` of live GTDs (the
    /// unit STORED at submit, not a recomputed cost), `reserved_sell`
    /// and per-token inventory likewise reconstructed.
    pub fn accounting(&self) -> SimAccounting {
        let execs = self.exec_log.lock().clone();
        let pend = self.pending.lock();
        let mut expected_reserved_buy = Decimal::ZERO;
        let mut expected_reserved_sell: HashMap<String, Decimal> = HashMap::new();
        for lim in pend.values() {
            match &lim.req.kind {
                OrderKind::Limit { size, .. } => {
                    let rem = (*size - lim.filled).max(Decimal::ZERO);
                    expected_reserved_buy += lim.reserved_unit * rem;
                }
                OrderKind::LimitSell { size, .. } => {
                    let rem = (*size - lim.filled).max(Decimal::ZERO);
                    *expected_reserved_sell
                        .entry(lim.req.token_id.clone())
                        .or_default() += rem;
                }
                _ => {}
            }
        }
        drop(pend);

        let mut buy_cash = Decimal::ZERO;
        let mut sell_cash = Decimal::ZERO;
        let mut expected_inv: HashMap<String, Decimal> = HashMap::new();
        for e in &execs {
            match e.side {
                FillSide::Buy => {
                    buy_cash += e.cash;
                    *expected_inv.entry(e.token_id.clone()).or_default() += e.shares;
                }
                FillSide::Sell => {
                    sell_cash += e.cash;
                    *expected_inv.entry(e.token_id.clone()).or_default() -= e.shares;
                }
                _ => {}
            }
        }
        let reds = self.redemptions.lock().clone();
        let redemptions_usdc: Decimal = reds.iter().map(|r| r.3).sum();
        for (_, token, shares, _) in &reds {
            *expected_inv.entry(token.clone()).or_default() -= shares;
        }
        let expected_inventory: HashMap<String, Decimal> = expected_inv
            .into_iter()
            .filter(|(_, v)| *v != Decimal::ZERO)
            .collect();
        let inventory_state: HashMap<String, Decimal> = self
            .inventory
            .lock()
            .clone()
            .into_iter()
            .filter(|(_, v)| *v != Decimal::ZERO)
            .collect();

        let balance = *self.balance.lock();
        let expected_balance =
            self.initial_balance + *self.contributions.lock() + redemptions_usdc + sell_cash
                - buy_cash;
        SimAccounting {
            initial_balance: self.initial_balance,
            contributions_usdc: *self.contributions.lock(),
            redemptions_usdc,
            balance,
            expected_balance,
            reserved_buy: *self.reserved_buy.lock(),
            expected_reserved_buy,
            reserved_sell: self.reserved_sell.lock().clone(),
            expected_reserved_sell,
            inventory_state,
            expected_inventory,
            executions: execs.len(),
        }
    }

    /// Advances the GTD queue with the received book tick.
    /// Fills go to fill_sink via the same `handle_trade` as live.
    ///
    /// A snapshot with IDENTICAL content to the current one brings no new flow
    /// — it does not consume the GTD queue again nor replenish the depth
    /// that takers already took (see `depth`/`fingerprint_side`).
    ///
    /// It also manages the research profile knobs
    /// (`research`): κ hypothesis of attributing shrinks to cancels
    /// in front/behind, tape gaps (abstention or `Stressed` mark),
    /// quote persistence for the stale mark and the participation
    /// cap over the crossed flow.
    pub fn tick_book(&self, book: &BookSnapshot) {
        let now_ms = (self.clock)();
        let now = now_ms / 1000;
        let cfg = self.research.lock().clone();
        // Purge expired GTDs — replicates the CLOB's auto-cancellation at clear_at.
        // Paper/live parity — `clob_router.rs` sends GTDs with
        // `clear_at + 60s` and `executor.rs` purges its local mapping at `clear_at + 65`;
        // the sim used to purge bluntly at exactly `clear_at`, expiring GTDs 60s
        // EARLIER than live. Critical in the final window before close
        // (NE/NC), where those 60s are a large fraction of the window.
        {
            let mut pending = self.pending.lock();
            let expired_oids: Vec<String> = pending
                .iter()
                .filter(|(_, lim)| !(lim.req.clear_at == 0 || lim.req.clear_at + 60 > now))
                .map(|(oid, _)| oid.clone())
                .collect();
            for oid in expired_oids {
                if let Some(lim) = pending.remove(&oid) {
                    // Expiry releases the remainder's reserve, same
                    // as an explicit cancel. Its remainder stops occupying a
                    // position in the queue of later own orders by
                    // construction — the own queue is summed live.
                    // It is also recorded in `gone` — expired without filling
                    // fully ≠ "never existed" (see `order_status`).
                    self.release_reservation(&lim);
                    self.gone.lock().insert(oid.clone(), lim.filled);
                }
            }
        }

        // Tape gap per token — a clock jump between ticks > gap_ms
        // invalidates the episode (we do not know what happened in the interval).
        // Measured over EVERY arrival (even if the content is identical): the feed
        // is still alive if it re-ticks.
        let mut gap_tokens: HashSet<&str> = HashSet::new();
        {
            let mut last = self.last_tick_ms.lock();
            for tok in [book.yes.token_id.as_str(), book.no.token_id.as_str()] {
                if cfg.gap_ms > 0
                    && last
                        .get(tok)
                        .is_some_and(|prev| now_ms - *prev > cfg.gap_ms)
                {
                    gap_tokens.insert(tok);
                }
                last.insert(tok.to_string(), now_ms);
            }
        }

        // Persistence of displayed quotes — `(token, side, price)` with
        // UNCHANGED size keeps its `from_ms`; a size change or a
        // disappearance turns it into a new quote. Feeds `leg_stale`
        // (taker fills against persistent out-of-consensus quotes =
        // stale/adverse selection, not free liquidity).
        {
            let mut shown = self.shown_since.lock();
            for (tok, side) in [
                (book.yes.token_id.as_str(), &book.yes),
                (book.no.token_id.as_str(), &book.no),
            ] {
                for (in_asks, half) in [(true, &side.asks), (false, &side.bids)] {
                    let mut active: HashSet<(String, bool, Decimal)> = HashSet::new();
                    for l in &half.levels {
                        let k = (tok.to_string(), in_asks, l.price);
                        active.insert(k.clone());
                        match shown.get(&k) {
                            Some((_, s)) if *s == l.size => {}
                            _ => {
                                shown.insert(k, (now_ms, l.size));
                            }
                        }
                    }
                    shown.retain(|k, _| !(k.0 == tok && k.1 == in_asks) || active.contains(k));
                }
            }
        }

        // Content fingerprint per token — a re-tick of the same book does not
        // restore the depth taken nor add new flow to the queue.
        let mut fresh: HashSet<&str> = HashSet::new();
        {
            let mut depth = self.depth.lock();
            for (tok, side) in [
                (book.yes.token_id.as_str(), &book.yes),
                (book.no.token_id.as_str(), &book.no),
            ] {
                let fp = fingerprint_side(side);
                let e = depth.entry(tok.to_string()).or_default();
                if e.fingerprint != Some(fp) {
                    *e = DepthState {
                        fingerprint: Some(fp),
                        ..Default::default()
                    };
                    fresh.insert(tok);
                }
            }
        }
        // Update the book cache.
        {
            let mut books = self.last_books.lock();
            books.insert(book.yes.token_id.clone(), book.clone());
            books.insert(book.no.token_id.clone(), book.clone());
        }

        // RESIDUAL depth of the crossing side per fresh token —
        // precomputed BEFORE the `pending` lock (lock order:
        // depth/last_books → pending, never the reverse — `consume_depth` does
        // depth → pending). The flow that fills a resting order must be net
        // of what our own takers already took from THIS snapshot:
        // otherwise the same displayed size would count twice — once as a fill
        // of our taker and once as crossed flow for the resting order.
        let mut per_side: HashMap<String, (Vec<(Decimal, Decimal)>, Vec<(Decimal, Decimal)>)> =
            HashMap::new();
        for tok in &fresh {
            per_side.insert(
                (*tok).to_string(),
                (
                    self.residual_levels(tok, true),
                    self.residual_levels(tok, false),
                ),
            );
        }

        // Identify GTDs that fill (partially or fully) in this tick.
        // Only the queue of tokens whose snapshot is NEW (`fresh`)
        // advances — a re-tick brings no new flow or deltas.
        // Queue position (Nautilus L2 model, no trade tape):
        //   (a) a decrease of the displayed size at our level on the side where we
        //       rest is evidence of consumption of the external queue
        //       ahead, weighted by the profile's κ hypothesis
        //       (`cancel_attr`): κ=1 every shrink drains the queue (nominal,
        //       status quo), κ=0 every shrink is assumed behind us (conservative).
        //       It never produces a fill by itself — without evidence of
        //       trade-through we do not know whether the flow reached our
        //       position or it was cancels. A shrink LARGER than the queue
        //       ahead is ambiguous consumption under any hypothesis → the order
        //       becomes `degraded_evidence` (later fills `Stressed`);
        //   (b) flow crossing our price (residual, already net of what
        //       our takers took from this snapshot) first empties the
        //       EXTERNAL queue, then the remainders of our own earlier resting orders
        //       with lower seq at the same level (FIFO), and only the EXCESS
        //       executes us, capped at the remainder and at `participation × cross`
        //       — a partial leaves the order live in `pending`. If the book
        //       crosses with enough volume, the fill is complete;
        //   (c) tape gap with `gap_abstain`: the post-gap tick resynchronizes
        //       the delta basis (current level as the new expected one), degrades
        //       the order's evidence and does NOT fill or advance the queue — the
        //       episode is invalidated, not invented.
        // `in_asks` marks whether it rests on the asks (LimitSell):
        // balance accounting and the reported trade side are
        // the inverse of those of a buy GTD. `seq` orders emission —
        // `retain` over a HashMap iterates in a random order per process; without
        // sorting, the fills of one tick would come out in a different order on
        // each run and the replay would no longer be reproducible.
        let mut to_fill: Vec<FillPlan> = Vec::new();
        {
            let mut pending = self.pending.lock();
            // PRE-tick remainders of own resting orders per level —
            // the own queue in front of each order is summed live (lower seq
            // goes in front). Baked into `queue_ahead` it went stale after a
            // partial and the L2 delta of the post-trade snapshot discounted the
            // same consumption again → the same flow filled two of our own orders.
            // (token, in_asks, price, seq, remainder) — a Vec because the iteration order
            // of the HashMap is irrelevant here (it is only filtered).
            let own: Vec<(String, bool, Decimal, u64, Decimal)> = pending
                .values()
                .filter_map(|l| {
                    Self::level_side_size(&l.req.kind)
                        .map(|(p, a, _)| (l.req.token_id.clone(), a, p, l.seq, Self::remaining(l)))
                })
                .collect();
            pending.retain(|order_id, lim| {
                if !fresh.contains(lim.req.token_id.as_str()) {
                    return true;
                }
                // Compare by req.token_id (not by the key, which is the order_id).
                let side = if lim.req.token_id == book.yes.token_id {
                    &book.yes
                } else if lim.req.token_id == book.no.token_id {
                    &book.no
                } else {
                    return true;
                };
                let Some((limit_price, in_asks, size)) = Self::level_side_size(&lim.req.kind)
                else {
                    // Careful: this arm silently removes from `pending`
                    // any non-resting OrderKind.
                    return false;
                };
                // Our side (where we rest): a bid is filled by asks at
                // ≤ its price (sellers crossing up to it); an ask by
                // bids at ≥ its price.
                let own_side = if in_asks { &side.asks } else { &side.bids };
                let current_level = own_side
                    .levels
                    .iter()
                    .find(|l| l.price == limit_price)
                    .map(|l| l.size)
                    .unwrap_or_default();
                // (c) Post-gap tick with abstention — the interval is a hole
                // in the evidence: resynchronize the delta basis, degrade the
                // order and do NOT fill or advance the queue this tick.
                if gap_tokens.contains(lim.req.token_id.as_str()) && cfg.gap_abstain {
                    lim.prev_level = current_level;
                    lim.degraded_evidence = true;
                    return true;
                }
                let initial_queue = lim.queue_ahead;
                // (a) L2 delta of the level with an explicit κ hypothesis: the `cancel_attr`
                // fraction of the shrink is attributed to cancels/consumption
                // IN FRONT (drains the queue); the rest to cancels behind (does not
                // move us). It never produces a fill by itself. A shrink larger than
                // the queue ahead cannot be explained entirely by cancels in front
                // — it is ambiguous consumption (trades that reached us or cancels
                // behind, indistinguishable without a tape): we degrade the order's
                // evidence, never turn it into a fill.
                let shrink = (lim.prev_level - current_level).max(Decimal::ZERO);
                if shrink > lim.queue_ahead && shrink > Decimal::ZERO {
                    lim.degraded_evidence = true;
                }
                lim.queue_ahead = (lim.queue_ahead - shrink * cfg.cancel_attr).max(Decimal::ZERO);
                // (b) RESIDUAL crossed flow: first empties the EXTERNAL queue
                // ahead, then the pre-tick remainders of the earlier own
                // resting orders (FIFO); only the excess executes us.
                let cross_side =
                    per_side
                        .get(lim.req.token_id.as_str())
                        .map(|(a, b)| if in_asks { b } else { a });
                let cross: Decimal = cross_side
                    .map(|lv| {
                        lv.iter()
                            .filter(|(p, _)| {
                                if in_asks {
                                    *p >= limit_price
                                } else {
                                    *p <= limit_price
                                }
                            })
                            .map(|(_, s)| *s)
                            .sum()
                    })
                    .unwrap_or_default();
                let absorbed_ext = lim.queue_ahead.min(cross);
                lim.queue_ahead -= absorbed_ext;
                // The cross consumes external depth at the level: `prev_level` ends up
                // at the expected post-trade size — otherwise the L2 delta of the
                // next snapshot would discount the same flow again.
                lim.prev_level = current_level - current_level.min(cross);
                let own_in_front: Decimal = own
                    .iter()
                    .filter(|(t, a, p, s, _)| {
                        *t == lim.req.token_id && *a == in_asks && *p == limit_price && *s < lim.seq
                    })
                    .map(|(_, _, _, _, r)| *r)
                    .sum();
                // Participation limit — at most a `participation`
                // fraction of the tick's crossed flow can execute us
                // (κ=1 ⇒ `cross` ≥ excess, no effect).
                let fill = (cross - absorbed_ext - own_in_front)
                    .min(size - lim.filled)
                    .min(cross * cfg.participation)
                    .max(Decimal::ZERO);
                if fill > Decimal::ZERO {
                    lim.filled += fill;
                    let confidence = if lim.degraded_evidence
                        || gap_tokens.contains(lim.req.token_id.as_str())
                    {
                        SimConfidence::Stressed
                    } else {
                        SimConfidence::Normal
                    };
                    to_fill.push(FillPlan {
                        order_id: order_id.clone(),
                        req: lim.req.clone(),
                        size: fill,
                        in_asks,
                        reserved_unit: lim.reserved_unit,
                        seq: lim.seq,
                        since_ms: lim.since_ms,
                        confidence,
                        evidence: SimEvidence::CrossBehindQueue {
                            cross,
                            queue_ahead: initial_queue,
                            own_ahead: own_in_front,
                        },
                    });
                }
                lim.filled < size
            });
            // Emit in arrival order (seq) — deterministic and, in passing,
            // more faithful: the CLOB executes resting orders in the order in which
            // they entered the book.
            to_fill.sort_by_key(|p| p.seq);
        } // pending lock released

        for plan in to_fill {
            let order_id = plan.order_id;
            let req = plan.req;
            let size = plan.size;
            let is_ask_gtd = plan.in_asks;
            let reserved_unit = plan.reserved_unit;
            let limit_price = match &req.kind {
                OrderKind::Limit { price, .. } | OrderKind::LimitSell { price, .. } => *price,
                _ => Decimal::ZERO,
            };
            // Maker stale mark — a resting order that had spent ≥ quote_stale_ms
            // on the book and whose price ended up out of consensus was
            // probably executed by informed flow (adverse selection).
            let mut confidence = plan.confidence;
            if confidence == SimConfidence::Normal
                && self.stressed_maker_fill(&req.token_id, plan.in_asks, limit_price, plan.since_ms)
            {
                confidence = SimConfidence::Stressed;
            }
            if is_ask_gtd {
                // Maker ask (sell): on fill, USDC comes in — symmetric to the bid GTD.
                // No fee: it is genuinely maker (0% on Polymarket) — the buy-side fee
                // only affects the BUY side (see below), not this maker-ask sell.
                // Quantized to the ledger micro — the paper cash moves what
                // `transactions` settles, not a sub-micro more.
                let usdc = cash_ledger(limit_price * size);
                let mut bal = self.balance.lock();
                *bal += usdc;
                let _ = self.wallet_bal_tx.send(Some(*bal));
                // The sell consumes inventory and releases the shares reserve
                // of the executed portion (the reserve is taken on entering the book).
                {
                    let mut inv = self.inventory.lock();
                    *inv.entry(req.token_id.clone()).or_default() -= size;
                }
                {
                    let mut rs = self.reserved_sell.lock();
                    if let Some(r) = rs.get_mut(&req.token_id) {
                        *r = (*r - size).max(Decimal::ZERO);
                    }
                }
                let exec = SimExecution {
                    exec_id: self.next_exec_id(),
                    order_id: order_id.clone(),
                    local_order_id: req.local_order_id.clone(),
                    token_id: req.token_id.clone(),
                    side: FillSide::Sell,
                    shares: size,
                    notional: usdc,
                    fee_usdc: Decimal::ZERO,
                    cash: usdc,
                    levels: vec![SimExecLevel {
                        price: limit_price,
                        size,
                        fee_usdc: Decimal::ZERO,
                        // Maker: executes at its limit — no slippage.
                        impact_usdc: Decimal::ZERO,
                        // The resting order is OUR quote: the level stale mark
                        // applies to external depth taken; maker
                        // adverse selection travels in `confidence`.
                        stale: false,
                    }],
                    ts_ms: (self.clock)(),
                    evidence: plan.evidence.clone(),
                    hypothesis: SimHypothesis::QueueExcess,
                    confidence,
                    // Genuine maker ask — fee 0 by venue rule.
                    role: SimRole::Maker,
                };
                // Polymarket semantics: `side` is the TAKER's. Our GTD ask is
                // filled when a taker BUYS → side="BUY", role="maker"; handle_trade
                // inverts it to our Sell — with "SELL" here the
                // fill would enter reversed and `on_fill` would add inventory instead of subtracting it.
                self.emit_execution(
                    exec,
                    &req.market_cid,
                    limit_price,
                    "BUY",
                    "maker",
                    None,
                    req.strategy,
                    req.real_clear_at,
                );
            } else {
                // A buy GTD only fills when `ask <= limit_price`
                // (see the `retain` above) — that is, it ALWAYS crosses the spread at
                // the moment of the fill. On live that same situation executes as a
                // real TAKER (`executor.rs`: post_only=false for PM/TC),
                // paying `rate × p × (1−p)`. The sim used to ALWAYS treat it as a
                // fee-free maker — optimistic versus live. `post_only=true` (MM) is
                // genuine maker: the real CLOB rejects such orders if they would cross
                // the spread, so if they get here it is because they did not cross — no fee.
                // The fee is charged separately from the negotiated
                // price (same criterion as the taker paths) — the earlier model
                // folded it into the price and truncated it to the tick, which
                // cancelled it entirely at high prices (p=0.95: effective fee 0).
                // Quantized to 5dp like any segment — official precision.
                let notional = limit_price * size;
                let fee_usdc = if req.post_only {
                    Decimal::ZERO
                } else {
                    (self.taker_fee(&req.token_id, limit_price) * size).round_dp(5)
                };
                // To the ledger micro — same reason as `leg_cash`.
                let usdc = cash_ledger(notional + fee_usdc);
                let exec_price = usdc / size;
                {
                    let mut bal = self.balance.lock();
                    let new_bal = *bal - usdc;
                    if new_bal < Decimal::ZERO {
                        // The overdraft is NO LONGER absorbed (clamping to 0
                        // created cash — inventory paid for with an invisible deficit).
                        // The balance may go negative: the position exists and
                        // the deficit is real; it blocks new buys until
                        // sells replenish it.
                        tracing::warn!(
                            balance = %*bal, cost = %usdc, overdraft = %(-new_bal),
                            "sim_router: insufficient balance when filling GTD — balance goes negative (real deficit)"
                        );
                    }
                    *bal = new_bal;
                    let _ = self.wallet_bal_tx.send(Some(*bal));
                }
                // The fill consumes the matching part of the reserve
                // — `reserved_unit` includes the fee estimated at submit, so the
                // reserve covers the real cost (limit + fee), not just the notional.
                {
                    let mut res = self.reserved_buy.lock();
                    *res = (*res - reserved_unit * size).max(Decimal::ZERO);
                }
                // The buy adds to inventory (always accounted; the
                // REQUIREMENT on sells depends on `track_inventory`).
                {
                    let mut inv = self.inventory.lock();
                    *inv.entry(req.token_id.clone()).or_default() += size;
                }
                let exec = SimExecution {
                    exec_id: self.next_exec_id(),
                    order_id: order_id.clone(),
                    local_order_id: req.local_order_id.clone(),
                    token_id: req.token_id.clone(),
                    side: FillSide::Buy,
                    shares: size,
                    notional,
                    fee_usdc,
                    cash: usdc,
                    levels: vec![SimExecLevel {
                        price: limit_price,
                        size,
                        fee_usdc,
                        // The resting order fills at its limit — no slippage.
                        impact_usdc: Decimal::ZERO,
                        stale: false,
                    }],
                    ts_ms: (self.clock)(),
                    evidence: plan.evidence.clone(),
                    hypothesis: SimHypothesis::QueueExcess,
                    confidence,
                    // `post_only` is genuine maker (fee 0); the rest of the
                    // resting buys fill by crossing → they pay taker on live
                    // — the role follows the real cross of the fill.
                    role: if req.post_only {
                        SimRole::Maker
                    } else {
                        SimRole::Taker
                    },
                };
                // Polymarket semantics: `side` is the TAKER's. Our GTD bid is
                // filled when a taker SELLS → side="SELL", role="maker"; handle_trade
                // inverts it to our Buy (exact parity with the live user_ws).
                self.emit_execution(
                    exec,
                    &req.market_cid,
                    exec_price,
                    "SELL",
                    "maker",
                    None,
                    req.strategy,
                    req.real_clear_at,
                );
            }
        }
    }

    /// Records the execution in the sim's log and builds the `UserTrade
    /// MATCHED` that goes through the same `handle_trade` as live:
    /// `UserTrade.id` = the execution's unique exec id, `order_id` = the sim's
    /// order id, `local_order_id` = the `OrderRequest`'s.
    /// `fill_price` is the all-in average price (`cash/shares`).
    #[allow(clippy::too_many_arguments)]
    fn emit_execution(
        &self,
        exec: SimExecution,
        market_cid: &str,
        fill_price: Decimal,
        taker_side: &str,
        role: &str,
        hedge_id: Option<i64>,
        strategy: crate::types::StrategyId,
        real_clear_at: Option<i64>,
    ) {
        let t = UserTrade {
            id: exec.exec_id.clone(),
            asset_id: exec.token_id.clone(),
            market: market_cid.to_string(),
            side: taker_side.to_string(),
            size: exec.shares.to_string(),
            price: fill_price.to_string(),
            status: "MATCHED".to_string(),
            trader_side: role.to_string(),
            hedge_id,
            // Authoritative strategy from the OrderRequest — avoids the
            // order_map (market:token) collision when two strategies
            // trade the same token (fills of one attributed to the other).
            strategy: Some(strategy),
            // REAL close of the bracket, when the `OrderRequest`
            // carries it — see the doc of `UserTrade::real_clear_at`.
            real_clear_at,
            // Identity of the sim's ORDER —
            // distinct from `id` (the execution). The core stage changes
            // `handle_trade` and the dedup at once; here it already travels stamped.
            order_id: Some(exec.order_id.clone()),
            local_order_id: exec.local_order_id.clone(),
            maker_orders: Vec::new(),
        };
        self.exec_log.lock().push(exec);
        handle_trade(
            t,
            &self.registry,
            &self.fill_tx,
            &self._paper_order_ids,
            Mode::Paper,
            &None,
        );
    }

    // ── Price factors ────────────────────────────────────────

    fn slippage_buy(&self) -> Decimal {
        Decimal::ONE + Decimal::from(self.slippage_bps) / dec!(10000)
    }

    fn slippage_sell(&self) -> Decimal {
        Decimal::ONE + Decimal::from(self.slippage_bps) / dec!(10000)
    }

    /// PARABOLIC taker fee rate for the cached book of
    /// `token_id` — the `parabolic_fee_rate` brought by discovery (Gamma
    /// `feeSchedule.rate`, see `market_feed::discover`), or `self.taker_fee_rate`
    /// if the book does not carry it yet (freshly discovered market without fetch, or
    /// `fetch_taker_fee=false`).
    ///
    /// If a frozen schedule exists for the market (`set_fee_windows`), the
    /// window in force at the fill instant (injectable clock) decides —
    /// the book's rate is only a fallback outside a window or when the
    /// window reports no rate (`None`).
    fn fee_rate_for(&self, token_id: &str) -> Decimal {
        let (cid, fallback) = {
            let books = self.last_books.lock();
            match books.get(token_id) {
                Some(b) => (
                    b.market_condition_id.clone(),
                    b.parabolic_fee_rate_or(self.taker_fee_rate),
                ),
                None => return self.taker_fee_rate,
            }
        };
        let windows = self.fee_windows.lock();
        match windows.get(&cid) {
            Some(w) => crate::fair_value::current_fee_rate(w, (self.clock)(), fallback),
            None => fallback,
        }
    }

    /// Taker fee per share at the given price: `rate × p × (1−p)`
    /// (parabolic), with the real rate of the market of `token_id` if the book
    /// carries it.
    ///
    /// Shares its formula with `fair_value::fee_taker_parabolic_with_rate` — the
    /// real contract formula for crypto up/down (verified from three sources,
    /// including the Gamma API `feeSchedule`, 6/6 assets). The earlier one
    /// (`rate × min(p, 1−p)`, triangular) used the `MAX_FEE_RATE_BIPS` that
    /// the CLOB's `/fee-rate` returns — a ceiling authorized by the contract, not
    /// the fee actually charged — and overcharged by up to 186% at p=0.50.
    fn taker_fee(&self, token_id: &str, price: Decimal) -> Decimal {
        crate::fair_value::fee_taker_parabolic_with_rate(
            self.fee_rate_for(token_id),
            price,
            Decimal::ONE,
        )
    }

    /// Deterministic uniform [0,1): LCG over `rng_state`, its OWN counter
    /// decoupled from `order_seq` — both used to share the
    /// same sequence, so the accept/reject draw was an exact function
    /// of the assigned order number (not iid).
    fn draw_uniform(&self) -> f64 {
        let n = self.rng_state.fetch_add(1, Ordering::Relaxed);
        let h = n
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (h % 10_000) as f64 / 10_000.0
    }

    /// Sets the per-strategy `fok_fill_rate` overrides (from `main`).
    pub fn set_fok_fill_rate_per_strategy(&self, m: HashMap<crate::types::StrategyId, f64>) {
        *self.fok_fill_rate_per_strategy.lock() = m;
    }

    fn override_fok(&self, strategy: crate::types::StrategyId) -> Option<f64> {
        self.fok_fill_rate_per_strategy.lock().get(&strategy).copied()
    }

    /// Decides whether to reject a standalone FOK (for pairs see submit_pair: correlated draw).
    /// With an override `>= 1.0` there is no draw (the RNG is not consumed): the rejection
    /// is decided solely by the book depth, in the caller's `walk_buy`.
    fn should_reject_fok(&self, strategy: crate::types::StrategyId) -> bool {
        match self.override_fok(strategy) {
            Some(rate) if rate >= 1.0 => false,
            Some(rate) => self.draw_uniform() >= rate,
            None => self.draw_uniform() >= self.fok_fill_rate,
        }
    }

    /// Chaos sim: enables `RouterResult::unknown` injection. Deterministic
    /// under `with_deps` (same seed ⇒ same draws). With `rate = 0` NO
    /// RNG draw is consumed — the previous behavior stays intact.
    pub fn set_unknown_rate(&self, rate: f64) {
        *self.unknown_rate.lock() = rate.clamp(0.0, 1.0);
    }

    /// Applies chaos mode to an already computed result. Only converts
    /// results of orders the venue ACCEPTED (`no_fill`/`fill`): a
    /// pre-submit `rejected` cannot be a "lost ACK" because the order
    /// never reached the venue. The real fill/execution already happened via the
    /// normal path — exactly the live scenario "order executed, response
    /// lost" that the Unknown reconciler must match.
    fn maybe_unknown(&self, res: RouterResult) -> RouterResult {
        if res.rejected || res.unknown {
            return res;
        }
        let rate = *self.unknown_rate.lock();
        if rate <= 0.0 || self.draw_uniform() >= rate {
            res
        } else {
            RouterResult::unknown("sim-chaos: lost ACK — the order may have executed on the venue")
        }
    }

    // ── Residual depth ────────────────────────────────────────

    /// Levels of the cached book of `token_id` with the depth already
    /// consumed by takers in THIS snapshot discounted. `asks=true` → asks in
    /// ascending price order; `false` → bids descending.
    fn residual_levels(&self, token_id: &str, asks: bool) -> Vec<(Decimal, Decimal)> {
        let books = self.last_books.lock();
        let Some(b) = books.get(token_id) else {
            return Vec::new();
        };
        let side = if token_id == b.yes.token_id {
            &b.yes
        } else {
            &b.no
        };
        let levels = if asks {
            &side.asks.levels
        } else {
            &side.bids.levels
        };
        let depth = self.depth.lock();
        let taken = depth
            .get(token_id)
            .map(|d| if asks { &d.asks_taken } else { &d.bids_taken });
        let mut v: Vec<(Decimal, Decimal)> = levels
            .iter()
            .map(|l| {
                let t = taken
                    .and_then(|m| m.get(&l.price))
                    .copied()
                    .unwrap_or_default();
                (l.price, (l.size - t).max(Decimal::ZERO))
            })
            .filter(|&(_, s)| s > Decimal::ZERO)
            .collect();
        if asks {
            v.sort_by_key(|a| a.0);
        } else {
            v.sort_by_key(|e| std::cmp::Reverse(e.0));
        }
        v
    }

    /// EFFECTIVE best ask: first level with residual depth (levels
    /// already consumed by takers in this snapshot do not count).
    fn eff_best_ask(&self, token_id: &str) -> Option<Decimal> {
        self.residual_levels(token_id, true)
            .first()
            .map(|&(p, _)| p)
    }

    /// EFFECTIVE best bid, mirror of `eff_best_ask` (residual depth).
    fn eff_best_bid(&self, token_id: &str) -> Option<Decimal> {
        self.residual_levels(token_id, false)
            .first()
            .map(|&(p, _)| p)
    }

    /// Discounts from the current snapshot the depth a
    /// taker just took: until a book with different content arrives, those
    /// shares are not available again to the next taker — two
    /// takers in the same tick do not consume the same displayed size twice.
    ///
    /// Shared queue (Nautilus model): what one of OUR OWN takers takes from
    /// a level also consumes the queue ahead of our resting orders
    /// that rest on that same side/price — on the real venue our
    /// taker would execute first against whoever was already at the level, not
    /// against our resting order. Conservative: it only advances the queue, NEVER fills
    /// the resting order (we do not simulate self-match).
    fn consume_depth(&self, token_id: &str, asks: bool, taken: &[(Decimal, Decimal)]) {
        {
            let mut depth = self.depth.lock();
            let d = depth.entry(token_id.to_string()).or_default();
            let map = if asks {
                &mut d.asks_taken
            } else {
                &mut d.bids_taken
            };
            for &(p, s) in taken {
                *map.entry(p).or_default() += s;
            }
        }
        if taken.is_empty() {
            return;
        }
        let mut pending = self.pending.lock();
        for lim in pending.values_mut() {
            if lim.req.token_id != token_id {
                continue;
            }
            let Some((price, in_asks, _)) = Self::level_side_size(&lim.req.kind) else {
                continue;
            };
            // A resting order on the asks (LimitSell) sees its queue consumed when a
            // taker BUYS asks (asks=true); one on the bids (Limit) when one
            // SELLS against bids.
            if in_asks != asks {
                continue;
            }
            let t = taken
                .iter()
                .find(|(p, _)| *p == price)
                .map(|&(_, s)| s)
                .unwrap_or_default();
            if t > Decimal::ZERO {
                lim.queue_ahead = (lim.queue_ahead - t).max(Decimal::ZERO);
                // Our own consumption also lowers the BASE of the L2
                // delta (`prev_level`) — otherwise the next snapshot's shrink
                // would re-attribute as "queue consumption" the shares
                // that WE already took (trade + its delta counted twice:
                // the twin volume in the book after a trade of ours is not
                // new flow).
                lim.prev_level = (lim.prev_level - t).max(Decimal::ZERO);
            }
        }
    }

    /// (all-in cost per share, unquantized fee per share, slipped
    /// price) of a taker buy against the level at price `p`:
    /// multiplicative slippage over the level and parabolic fee over the already slipped price
    /// — kept separate in the internal accounting (`SimExecution`),
    /// not folded into a price that is truncated to the tick (a bug of the
    /// earlier model: the fee vanished with truncation at high prices).
    /// `slipped` travels in the `WalkLeg` to separate the realized impact
    /// from the negotiated price and the fee.
    fn taker_buy_costs(&self, token_id: &str, p: Decimal) -> (Decimal, Decimal, Decimal) {
        let slipped = p * self.slippage_buy();
        // The contractual formula p·(1−p) only makes sense in [0,1] — slippage
        // can push the level above 1 at the extreme.
        let fee = self
            .taker_fee(token_id, slipped.min(Decimal::ONE))
            .max(Decimal::ZERO);
        (slipped + fee, fee, slipped)
    }

    /// (net per share, unquantized fee per share, slipped price) of a
    /// taker sell against the bid `p`.
    fn taker_sell_nets(&self, token_id: &str, p: Decimal) -> (Decimal, Decimal, Decimal) {
        let slipped = p / self.slippage_sell();
        let fee = self
            .taker_fee(token_id, slipped.min(Decimal::ONE))
            .max(Decimal::ZERO);
        // Defensive: a malformed `parabolic_fee_rate` (fee > price) would give
        // a negative net → the sell would SUBTRACT cash and the fill would have a negative
        // price. The floor is 0: at most the sell brings in nothing.
        ((slipped - fee).max(Decimal::ZERO), fee, slipped)
    }

    /// Fee CHARGED for the segment `l` — `fee_share × size` quantized to 5
    /// decimals, the venue's official precision (see
    /// `fair_value::fee_charged_5dp`).
    fn leg_fee(l: &WalkLeg) -> Decimal {
        (l.fee_share * l.size).round_dp(5)
    }

    /// Cash of the segment with the fee already quantized and the TOTAL at the
    /// resolution of the durable ledger — `StateStore` settles amounts via
    /// `to_micros` (6dp; the current implementation ROUNDS, it does not truncate),
    /// so the wallet moves EXACTLY what the ledger records: without this
    /// rounding the sub-micro remainder of slippage/partials left a false drift
    /// in `reconcile_economic_return`. The same `to_micros`/`from_micros` pair is used
    /// so as never to diverge from the real conversion.
    /// Sell: defensive floor of 0 (fee > notional only with a malformed fee_rate)
    /// — the clamp is logged, not silenced.
    fn leg_cash(side: FillSide, l: &WalkLeg) -> Decimal {
        let raw = match side {
            FillSide::Buy => l.slipped * l.size + Self::leg_fee(l),
            _ => l.slipped * l.size - Self::leg_fee(l),
        };
        if raw < Decimal::ZERO {
            // Only possible on a sell with fee_share > slipped — a malformed
            // parameter; the floor keeps the sell from SUBTRACTING cash but the
            // accounting anomaly must be visible.
            tracing::warn!(
                price = %l.price, size = %l.size, raw = %raw,
                "sim_router: segment fee exceeds the slipped notional — sell credits 0 (floor)"
            );
        }
        cash_ledger(raw.max(Decimal::ZERO))
    }

    /// Cash moved by a complete walk — sum of `leg_cash` per
    /// segment, each at the ledger's micro resolution. Single source of truth
    /// for the charge/income: the balance debits/credits and
    /// `legs_execution` use it, so the recorded exec, the cash moved and the durable
    /// ledger entry cannot diverge.
    fn legs_cash(&self, side: FillSide, legs: &[WalkLeg]) -> Decimal {
        legs.iter().map(|l| Self::leg_cash(side, l)).sum()
    }

    /// BUY taker path: residual asks at price ≤ `limit`, up to
    /// `want` shares. It may fall short — the caller decides whether it is FOK
    /// (all-or-nothing) or FAK (partial + cancel).
    /// Each level is taken for at most a `share` fraction of its
    /// residual depth (κ=1 ⇒ no effect) and each segment is marked with
    /// `leg_stale` (persistent out-of-consensus quote = adverse selection).
    fn walk_buy(&self, token_id: &str, limit: Decimal, want: Decimal) -> Vec<WalkLeg> {
        let mut need = want;
        let mut legs = Vec::new();
        let part = self.research.lock().participation;
        for (p, avail) in self.residual_levels(token_id, true) {
            if p > limit || need <= Decimal::ZERO {
                break;
            }
            let take = avail.min(need).min(avail * part);
            if take <= Decimal::ZERO {
                continue;
            }
            let (_unit, fee_share, slipped) = self.taker_buy_costs(token_id, p);
            legs.push(WalkLeg {
                price: p,
                size: take,
                slipped,
                fee_share,
                stale: self.leg_stale(token_id, true, p),
            });
            need -= take;
        }
        legs
    }

    /// SELL taker path: residual bids in descending order, up to
    /// `want` shares (real partial: fills whatever there is).
    fn walk_sell(&self, token_id: &str, want: Decimal) -> Vec<WalkLeg> {
        let mut need = want;
        let mut legs = Vec::new();
        let part = self.research.lock().participation;
        for (p, avail) in self.residual_levels(token_id, false) {
            if need <= Decimal::ZERO {
                break;
            }
            let take = avail.min(need).min(avail * part);
            if take <= Decimal::ZERO {
                continue;
            }
            let (_unit, fee_share, slipped) = self.taker_sell_nets(token_id, p);
            legs.push(WalkLeg {
                price: p,
                size: take,
                slipped,
                fee_share,
                stale: self.leg_stale(token_id, false, p),
            });
            need -= take;
        }
        legs
    }

    /// Limit-bounded SELL taker path: only residual bids at
    /// price ≥ `limit` (a marketable limit sell does not sell below its
    /// price). Mirror of `walk_buy`.
    fn walk_sell_limit(&self, token_id: &str, limit: Decimal, want: Decimal) -> Vec<WalkLeg> {
        let mut need = want;
        let mut legs = Vec::new();
        let part = self.research.lock().participation;
        for (p, avail) in self.residual_levels(token_id, false) {
            if p < limit || need <= Decimal::ZERO {
                break;
            }
            let take = avail.min(need).min(avail * part);
            if take <= Decimal::ZERO {
                continue;
            }
            let (_unit, fee_share, slipped) = self.taker_sell_nets(token_id, p);
            legs.push(WalkLeg {
                price: p,
                size: take,
                slipped,
                fee_share,
                stale: self.leg_stale(token_id, false, p),
            });
            need -= take;
        }
        legs
    }

    /// Converts a walk into a `SimExecution` (price/fee/cash separated) and
    /// also returns the all-in average price (`cash/shares`), which is
    /// the semantics the `Fill` carries to its consumers.
    ///
    /// The fee is quantized to 5dp PER SEGMENT (the venue's official precision) and
    /// each level records the already realized impact (`|slipped − price| ×
    /// size`); `cash` is `legs_cash` — the same sum that debits/credits the
    /// balance at the call sites.
    fn legs_execution(
        &self,
        order_id: &str,
        req: &OrderRequest,
        side: FillSide,
        legs: &[WalkLeg],
    ) -> (SimExecution, Decimal) {
        let shares: Decimal = legs.iter().map(|l| l.size).sum();
        let notional: Decimal = legs.iter().map(|l| l.price * l.size).sum();
        let fee: Decimal = legs.iter().map(Self::leg_fee).sum();
        let cash = self.legs_cash(side, legs);
        let exec = SimExecution {
            exec_id: self.next_exec_id(),
            order_id: order_id.to_string(),
            local_order_id: req.local_order_id.clone(),
            token_id: req.token_id.clone(),
            side,
            shares,
            notional,
            fee_usdc: fee,
            cash,
            levels: legs
                .iter()
                .map(|l| SimExecLevel {
                    price: l.price,
                    size: l.size,
                    fee_usdc: Self::leg_fee(l),
                    // Realized impact of the segment: slippage relative to the
                    // level's negotiated price (adverse by construction).
                    impact_usdc: match side {
                        FillSide::Buy => (l.slipped - l.price).max(Decimal::ZERO) * l.size,
                        _ => (l.price - l.slipped).max(Decimal::ZERO) * l.size,
                    },
                    stale: l.stale,
                })
                .collect(),
            ts_ms: (self.clock)(),
            // Taker/marketable execution — the evidence is the residual depth
            // at arrival; the hypothesis is TakerDepth; `Stressed` if
            // any segment executed against a persistent quote out of
            // consensus (stale/adverse selection).
            evidence: SimEvidence::DepthAtArrival { levels: legs.len() },
            hypothesis: SimHypothesis::TakerDepth,
            confidence: if legs.iter().any(|l| l.stale) {
                SimConfidence::Stressed
            } else {
                SimConfidence::Normal
            },
            // A depth walk is always taker — it crossed the
            // spread against book levels.
            role: SimRole::Taker,
        };
        let fill_price = if shares > Decimal::ZERO {
            cash / shares
        } else {
            Decimal::ZERO
        };
        (exec, fill_price)
    }

    /// Generates a unique order_id for the simulator.
    fn next_order_id(&self, token_id: &str) -> String {
        let seq = self.order_seq.fetch_add(1, Ordering::Relaxed);
        format!("sim-{}-{token_id}-{seq}", self.boot_ms)
    }

    /// Unique exec id per EXECUTION — distinct from the order_id (a GTD/FAK
    /// order can produce N partial executions). `boot_ms` makes it
    /// unique ACROSS processes too: the durable dedup key is `exec_id`.
    fn next_exec_id(&self) -> String {
        let seq = self.exec_seq.fetch_add(1, Ordering::Relaxed);
        format!("sim-exec-{}-{seq}", self.boot_ms)
    }

    /// Releases the reserve of a resting order's REMAINDER: for a buy GTD,
    /// `reserved_unit × (size − filled)` of cash (floor 0 —
    /// `reserved_unit` includes the fee estimated at submit: the reserve
    /// covers the real cost of the fill, not just the notional); for a
    /// `LimitSell`, the remainder in SHARES reserved in `reserved_sell` (a
    /// mirror on the inventory basis — shares are never fabricated).
    fn release_reservation(&self, lim: &PendingLimit) {
        match &lim.req.kind {
            OrderKind::Limit { size, .. } => {
                let rem = (*size - lim.filled).max(Decimal::ZERO);
                if rem > Decimal::ZERO {
                    let mut res = self.reserved_buy.lock();
                    *res = (*res - lim.reserved_unit * rem).max(Decimal::ZERO);
                }
            }
            OrderKind::LimitSell { size, .. } => {
                let rem = (*size - lim.filled).max(Decimal::ZERO);
                if rem > Decimal::ZERO {
                    let mut rs = self.reserved_sell.lock();
                    if let Some(r) = rs.get_mut(&lim.req.token_id) {
                        *r = (*r - rem).max(Decimal::ZERO);
                    }
                }
            }
            _ => {}
        }
    }

    /// Removes a live GTD and releases the reserve of its remainder. Its remainder
    /// stops counting in front of later own orders by construction
    /// — the own queue is summed live from the remainders.
    /// The order moves to `gone` — cancelled/expired ≠ "never existed".
    fn remove_pending(&self, order_id: &str) {
        let mut pending = self.pending.lock();
        if let Some(lim) = pending.remove(order_id) {
            drop(pending);
            self.release_reservation(&lim);
            self.gone.lock().insert(order_id.to_string(), lim.filled);
        }
    }

    /// Cash available for a new order: `balance − reserved_buy`.
    fn free_cash(&self) -> Decimal {
        *self.balance.lock() - *self.reserved_buy.lock()
    }

    // ── Queue position (Nautilus L2 model) ────────────────────────────────────────

    /// (price, rests_on_asks, size) of a resting order: `Limit` rests
    /// on the bids; `LimitSell` (maker ask GTD) on the asks.
    /// `None` for any non-resting OrderKind — those should never reach
    /// `pending`.
    fn level_side_size(kind: &OrderKind) -> Option<(Decimal, bool, Decimal)> {
        match kind {
            OrderKind::Limit { price, size, .. } => Some((*price, false, *size)),
            OrderKind::LimitSell { price, size, .. } => Some((*price, true, *size)),
            _ => None,
        }
    }

    /// Live remainder of a resting order: `size − filled`, floor 0.
    fn remaining(lim: &PendingLimit) -> Decimal {
        let size = Self::level_side_size(&lim.req.kind)
            .map(|(_, _, s)| s)
            .unwrap_or_default();
        (size - lim.filled).max(Decimal::ZERO)
    }

    /// Residual size (net of what our takers already took from the current
    /// snapshot) displayed at `price` on the `asks` side of the book of `token_id`.
    fn residual_level_size(&self, token_id: &str, asks: bool, price: Decimal) -> Decimal {
        self.residual_levels(token_id, asks)
            .iter()
            .find(|(p, _)| *p == price)
            .map(|&(_, s)| s)
            .unwrap_or_default()
    }

    /// RAW size displayed at `price` on the `asks` side of the cached book of
    /// `token_id` — without discounting what our takers took. It is the basis of the
    /// L2 delta (`prev_level`): our own consumption is accounted for by
    /// `consume_depth`, not by the delta between snapshots.
    fn raw_level_size(&self, token_id: &str, asks: bool, price: Decimal) -> Decimal {
        let books = self.last_books.lock();
        let Some(b) = books.get(token_id) else {
            return Decimal::ZERO;
        };
        let side = if token_id == b.yes.token_id {
            &b.yes
        } else {
            &b.no
        };
        let levels = if asks {
            &side.asks.levels
        } else {
            &side.bids.levels
        };
        levels
            .iter()
            .find(|l| l.price == price)
            .map(|l| l.size)
            .unwrap_or_default()
    }

    // ── Research profile, consensus, stale and inventory ────────────────────────────────────────

    /// Installs the research profile (`config::ResearchSimConfig`).
    /// ONLY research/replay/tests — the operational bot never calls it and the
    /// `nominal` default reproduces the earlier behavior.
    pub fn apply_research_profile(&self, cfg: crate::config::ResearchSimConfig) {
        *self.research.lock() = cfg;
    }

    /// Simulated inventory per token (net shares). Observability only
    /// for research/tests — with `track_inventory`
    /// on, sells require it (shares are never fabricated).
    pub fn simulated_inventory(&self) -> HashMap<String, Decimal> {
        self.inventory.lock().clone()
    }

    /// Accounts for simulated inventory (`delta` >0 buy, <0 sell).
    /// ALWAYS kept (even if `track_inventory` is off) so that
    /// enabling the knob hot does not corrupt the basis.
    fn adjust_inventory(&self, token_id: &str, delta: Decimal) {
        let mut inv = self.inventory.lock();
        *inv.entry(token_id.to_string()).or_default() += delta;
    }

    /// Deviation of `price` from the pair's cross consensus —
    /// complement to 1 of the best level on the opposite side of the OTHER token of the
    /// book (`ask_yes ≈ 1 − bid_no`, `bid_yes ≈ 1 − ask_no`; whatever
    /// deviates from that is a quote that the pair's market contradicts). `None`
    /// if the other side has no levels (no consensus to measure).
    fn consensus_deviation(&self, token_id: &str, asks: bool, price: Decimal) -> Option<Decimal> {
        let books = self.last_books.lock();
        let b = books.get(token_id)?;
        let other = if token_id == b.yes.token_id {
            &b.no
        } else {
            &b.yes
        };
        let consensus = if asks {
            Decimal::ONE - other.bids.levels.iter().map(|l| l.price).max()?
        } else {
            Decimal::ONE - other.asks.levels.iter().map(|l| l.price).min()?
        };
        Some((price - consensus).abs())
    }

    /// Is the `price` quote on the `asks` side of `token_id` a
    /// PERSISTENT out-of-consensus quote? (displayed ≥ `quote_stale_ms` with unchanged
    /// size and deviation > `consensus_margin`). Mark
    /// only: it never invents or discards liquidity.
    fn leg_stale(&self, token_id: &str, asks: bool, price: Decimal) -> bool {
        let cfg = self.research.lock().clone();
        if cfg.quote_stale_ms <= 0 || cfg.consensus_margin <= Decimal::ZERO {
            return false;
        }
        let persistent = self
            .shown_since
            .lock()
            .get(&(token_id.to_string(), asks, price))
            .is_some_and(|(since, _)| (self.clock)() - *since >= cfg.quote_stale_ms);
        if !persistent {
            return false;
        }
        self.consensus_deviation(token_id, asks, price)
            .is_some_and(|d| d > cfg.consensus_margin)
    }

    /// Maker stale mark — a resting order that had spent ≥ `quote_stale_ms` on
    /// the book and whose limit ended up out of consensus was probably
    /// executed by informed flow (adverse selection), not by noise.
    fn stressed_maker_fill(
        &self,
        token_id: &str,
        in_asks: bool,
        price: Decimal,
        since_ms: i64,
    ) -> bool {
        let cfg = self.research.lock().clone();
        if cfg.quote_stale_ms <= 0 || cfg.consensus_margin <= Decimal::ZERO {
            return false;
        }
        if (self.clock)() - since_ms < cfg.quote_stale_ms {
            return false;
        }
        self.consensus_deviation(token_id, in_asks, price)
            .is_some_and(|d| d > cfg.consensus_margin)
    }
}

impl OrderRouter for SimulatedRouter {
    fn submit(&self, req: OrderRequest) -> Pin<Box<dyn Future<Output = RouterResult> + Send + '_>> {
        Box::pin(async move {
            // Network latency: during this await the market_feed may update
            // last_books with new prices, simulating the market moving.
            if self.latency_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
            }

            let res = match &req.kind {
                OrderKind::Market { usdc_amount } => {
                    // Parity with live: `build_signed_market_buy` (market
                    // arm) truncates the USDC amount to 2dp — the contract's
                    // maker_amount.
                    let usdc_amount = &usdc_amount.trunc_with_scale(2);
                    if *usdc_amount <= Decimal::ZERO {
                        return RouterResult::rejected("invalid amount");
                    }
                    let asks = self.residual_levels(&req.token_id, true);
                    if asks.is_empty() {
                        // With no book we do not invent a price from `signal`
                        // (a probability) — reject.
                        return RouterResult::rejected("no book/depth for market");
                    }
                    // FOK by amount: walk the asks until spending
                    // `usdc_amount` all-in; if the depth
                    // falls short, reject entirely (CLOB market orders are
                    // all-or-nothing on the amount).
                    let mut remaining = *usdc_amount;
                    let mut legs: Vec<WalkLeg> = Vec::new();
                    // Participation per level + stale mark (adverse
                    // selection) — see `walk_buy`.
                    let part = self.research.lock().participation;
                    for (p, avail) in asks {
                        if remaining <= Decimal::ZERO {
                            break;
                        }
                        let (unit, fee_share, slipped) = self.taker_buy_costs(&req.token_id, p);
                        if unit <= Decimal::ZERO {
                            continue;
                        }
                        let take = avail.min(remaining / unit).min(avail * part);
                        if take <= Decimal::ZERO {
                            break;
                        }
                        legs.push(WalkLeg {
                            price: p,
                            size: take,
                            slipped,
                            fee_share,
                            stale: self.leg_stale(&req.token_id, true, p),
                        });
                        remaining -= take * unit;
                    }
                    if remaining > Decimal::ZERO {
                        return RouterResult::rejected(
                            "insufficient depth for the amount (market FOK)",
                        );
                    }
                    // The cash is `legs_cash` — fee quantized to 5dp per
                    // segment, identical to what the exec records.
                    let cash = self.legs_cash(FillSide::Buy, &legs);
                    {
                        let mut bal = self.balance.lock();
                        if *bal - *self.reserved_buy.lock() < cash {
                            return RouterResult::rejected("insufficient funds");
                        }
                        *bal -= cash;
                        let _ = self.wallet_bal_tx.send(Some(*bal));
                    }
                    // The buy adds to the simulated inventory.
                    self.adjust_inventory(&req.token_id, legs.iter().map(|l| l.size).sum());
                    self.consume_depth(&req.token_id, true, &legs_pairs(&legs));
                    let order_id = self.next_order_id(&req.token_id);
                    let (exec, fill_price) =
                        self.legs_execution(&order_id, &req, FillSide::Buy, &legs);
                    self.emit_execution(
                        exec,
                        &req.market_cid,
                        fill_price,
                        "BUY",
                        "taker",
                        None,
                        req.strategy,
                        req.real_clear_at,
                    );
                    RouterResult::no_fill(Some(order_id))
                }

                OrderKind::Limit { price, size, tif } => match tif {
                    TimeInForce::Fok | TimeInForce::Fak => {
                        // Parity with live: `build_signed_market_buy` truncates the
                        // size to a WHOLE number (maker_amount ≤4dp in the contract) and
                        // aborts if it ends up 0 — the sim accepted fractions that
                        // the venue would never execute (overestimated fills).
                        let size_trunc = size.trunc_with_scale(0);
                        if size_trunc <= Decimal::ZERO {
                            return RouterResult::rejected(
                                "FOK/FAK size after truncation to whole is 0",
                            );
                        }
                        // After the latency: did the price move against us or is there
                        // no book at all? The EFFECTIVE best_ask discounts
                        // the depth already taken from this snapshot.
                        match self.eff_best_ask(&req.token_id) {
                            None => {
                                return RouterResult::rejected("no book/depth for FOK/FAK")
                            }
                            Some(ask) if ask > *price => {
                                tracing::debug!(
                                    token = %req.token_id, limit = %price, post_ask = %ask,
                                    "sim_router: FOK rejected — price moved during latency"
                                );
                                return RouterResult::rejected("price moved during latency");
                            }
                            _ => {}
                        }
                        if self.should_reject_fok(req.strategy) {
                            tracing::debug!(
                                token = %req.token_id,
                                "sim_router: FOK rejected — insufficient liquidity"
                            );
                            return RouterResult::rejected("FOK rejected: insufficient liquidity");
                        }
                        // Execution against the real depth up to the
                        // limit — FOK all-or-nothing, FAK partial (the rest is
                        // cancelled: nothing is left resting).
                        let legs = self.walk_buy(&req.token_id, *price, size_trunc);
                        let covered: Decimal = legs.iter().map(|l| l.size).sum();
                        if covered <= Decimal::ZERO {
                            return RouterResult::rejected("no depth up to the limit");
                        }
                        if *tif == TimeInForce::Fok && covered < size_trunc {
                            return RouterResult::rejected(
                                "FOK with insufficient depth up to the limit",
                            );
                        }
                        let cash = self.legs_cash(FillSide::Buy, &legs);
                        {
                            let mut bal = self.balance.lock();
                            if *bal - *self.reserved_buy.lock() < cash {
                                return RouterResult::rejected("insufficient funds");
                            }
                            *bal -= cash;
                            let _ = self.wallet_bal_tx.send(Some(*bal));
                        }
                        // The buy adds to the simulated inventory.
                        self.adjust_inventory(&req.token_id, covered);
                        self.consume_depth(&req.token_id, true, &legs_pairs(&legs));
                        let order_id = self.next_order_id(&req.token_id);
                        let (exec, fill_price) =
                            self.legs_execution(&order_id, &req, FillSide::Buy, &legs);
                        self.emit_execution(
                            exec,
                            &req.market_cid,
                            fill_price,
                            "BUY",
                            "taker",
                            None,
                            req.strategy,
                            req.real_clear_at,
                        );
                        RouterResult::no_fill(Some(order_id))
                    }
                    TimeInForce::Gtd => {
                        // Parity with live (`clob_router`): a GTD without
                        // clear_at is rejected BEFORE sending — here it used to enter
                        // as an immortal resting order (`clear_at == 0` never purges)
                        // consuming reserve forever.
                        if req.clear_at == 0 {
                            return RouterResult::rejected("GTD without clear_at");
                        }
                        let size_trunc = size.trunc_with_scale(0);
                        if size_trunc <= Decimal::ZERO {
                            return RouterResult::rejected(
                                "GTD size too small after truncation",
                            );
                        }
                        // A GTD that is already MARKETABLE at submit (best_ask ≤ limit) does not rest
                        // — on the CLOB it executes immediately as a taker walking
                        // the asks ≤ limit, and only the remainder rests
                        // at the end of the level's queue. The sim used to park it
                        // whole in `pending` and fill it on the next tick at the
                        // LIMIT PRICE: a worse price than live and a fill that the
                        // market would already have given. And a `post_only` that crosses
                        // at submit is rejected by the venue — parking it
                        // invented maker fills that live never produces.
                        let marketable = self
                            .eff_best_ask(&req.token_id)
                            .is_some_and(|a| a <= *price);
                        if req.post_only && marketable {
                            return RouterResult::rejected(
                                "post_only would cross the spread — venue rejection",
                            );
                        }
                        let legs = if marketable {
                            self.walk_buy(&req.token_id, *price, size_trunc)
                        } else {
                            Vec::new()
                        };
                        let covered: Decimal = legs.iter().map(|l| l.size).sum();
                        let remaining = size_trunc - covered;

                        // The buy GTD RESERVES cash on entering
                        // the book — `balance − reserved_buy` is all that
                        // other orders can spend; the remainder is released
                        // on fill/cancel/expiry. The reserve covers the REAL cost
                        // of the fill: non-`post_only` orders fill as taker
                        // and pay `rate × p × (1−p)` — without including it
                        // here the fill overdrew and the clamp gave it away.
                        let reserved_unit = if req.post_only {
                            *price
                        } else {
                            *price + self.taker_fee(&req.token_id, *price)
                        };
                        let cash_now = self.legs_cash(FillSide::Buy, &legs);
                        let reserve_cost = reserved_unit * remaining;
                        if self.free_cash() < cash_now + reserve_cost {
                            return RouterResult::rejected("insufficient funds for GTD");
                        }
                        if cash_now > Decimal::ZERO {
                            let mut bal = self.balance.lock();
                            *bal -= cash_now;
                            let _ = self.wallet_bal_tx.send(Some(*bal));
                        }
                        *self.reserved_buy.lock() += reserve_cost;
                        let order_id = self.next_order_id(&req.token_id);
                        // The marketable part executes now: taker fill against the
                        // real asks (the level's price, not the limit) and the
                        // depth taken is discounted from the snapshot.
                        if !legs.is_empty() {
                            self.consume_depth(&req.token_id, true, &legs_pairs(&legs));
                            // The marketable part adds to the inventory.
                            self.adjust_inventory(&req.token_id, covered);
                            let (exec, fill_price) =
                                self.legs_execution(&order_id, &req, FillSide::Buy, &legs);
                            self.emit_execution(
                                exec,
                                &req.market_cid,
                                fill_price,
                                "BUY",
                                "taker",
                                None,
                                req.strategy,
                                req.real_clear_at,
                            );
                        }
                        if remaining > Decimal::ZERO {
                            // Queue position (Nautilus L2 model): only the
                            // EXTERNAL queue (residual bids at our price).
                            // Earlier own resting orders go in front but
                            // are summed live each tick from their remainders
                            // — baked in they would go stale after partials.
                            // `prev_level` stores the RAW size of the
                            // level: basis of the per-snapshot consumption delta.
                            let residual = self.residual_level_size(&req.token_id, false, *price);
                            let prev_level = self.raw_level_size(&req.token_id, false, *price);
                            let mut pending = self.pending.lock();
                            pending.insert(
                                order_id.clone(),
                                PendingLimit {
                                    req: {
                                        let mut r = req.clone();
                                        r.kind = OrderKind::Limit {
                                            price: *price,
                                            size: remaining,
                                            tif: TimeInForce::Gtd,
                                        };
                                        r
                                    },
                                    queue_ahead: residual,
                                    prev_level,
                                    seq: self.resting_seq.fetch_add(1, Ordering::Relaxed),
                                    filled: Decimal::ZERO,
                                    reserved_unit,
                                    // Age of the resting order (maker stale mark)
                                    // and evidence intact on entry.
                                    since_ms: (self.clock)(),
                                    degraded_evidence: false,
                                },
                            );
                        }
                        RouterResult::no_fill(Some(order_id))
                    }
                },

                OrderKind::Sell { size } => {
                    let fill_size_raw = match size {
                        Some(s) if *s > Decimal::ZERO => *s,
                        // `None` used to document "sell the whole position"
                        // but this fallback used `req.signal` — a probability
                        // (0.0–1.0), NOT a position size — as if it were
                        // shares. No strategy emits `size: None` today
                        // (verified); rejecting explicitly is safer than selling
                        // an invented size if one ever did.
                        _ => {
                            tracing::warn!(
                                "sim_router: sell without explicit size — size:None not implemented, signal={}",
                                req.signal
                            );
                            return RouterResult::rejected("Sell{size: None} not implemented: no router resolves the real position");
                        }
                    };
                    let fill_size = fill_size_raw.trunc_with_scale(2);
                    if fill_size <= Decimal::ZERO {
                        return RouterResult::no_fill(None);
                    }
                    // The taker sell walks the residual bids (real partial
                    // fill); with no book/bids → rejection — it used to fall to
                    // `signal` (a probability) as if it were a price.
                    let legs = self.walk_sell(&req.token_id, fill_size);
                    if legs.is_empty() {
                        return RouterResult::rejected("no book/bids to sell");
                    }
                    // With `track_inventory` sells REQUIRE simulated
                    // inventory net of shares reserved by resting asks —
                    // funds/shares are never fabricated. Available =
                    // inventory − token's reserved_sell.
                    let sold: Decimal = legs.iter().map(|l| l.size).sum();
                    if self.research.lock().track_inventory {
                        let disp = self
                            .simulated_inventory()
                            .get(&req.token_id)
                            .copied()
                            .unwrap_or_default()
                            - self
                                .reserved_sell
                                .lock()
                                .get(&req.token_id)
                                .copied()
                                .unwrap_or_default();
                        if disp < sold {
                            return RouterResult::rejected(
                                "insufficient inventory to sell (track_inventory)",
                            );
                        }
                    }

                    let order_id = self.next_order_id(&req.token_id);
                    let (exec, fill_price) =
                        self.legs_execution(&order_id, &req, FillSide::Sell, &legs);
                    {
                        let mut bal = self.balance.lock();
                        *bal += exec.cash;
                        let _ = self.wallet_bal_tx.send(Some(*bal));
                    }
                    // The sell consumes inventory (always accounted).
                    self.adjust_inventory(&req.token_id, -sold);
                    self.consume_depth(&req.token_id, false, &legs_pairs(&legs));
                    self.emit_execution(
                        exec,
                        &req.market_cid,
                        fill_price,
                        "SELL",
                        "taker",
                        None,
                        req.strategy,
                        req.real_clear_at,
                    );
                    RouterResult::no_fill(Some(order_id))
                }

                // Maker ask GTD (resting limit sell). Models the
                // queue on the ASKS side (where it rests): ahead of us go the asks already
                // displayed at our price; the buying flow (bids ≥ our
                // price) empties that queue and the excess executes us — the same queue-position
                // mechanics as the buy GTD, mirrored side.
                OrderKind::LimitSell { price, size, tif } => match tif {
                    TimeInForce::Gtd => {
                        // Parity with live (`clob_router`): a GTD without clear_at
                        // is rejected before sending — no immortal resting order.
                        if req.clear_at == 0 {
                            return RouterResult::rejected("GTD without clear_at");
                        }
                        // Parity with live: `submit_limit_sell` truncates to a WHOLE number
                        // (limit_order builder, same decimal margin as
                        // the buy) — the sim used 2dp and subtracted fractions that
                        // the venue would truncate.
                        let size_trunc = size.trunc_with_scale(0);
                        if size_trunc <= Decimal::ZERO {
                            return RouterResult::rejected(
                                "LimitSell size too small after truncation",
                            );
                        }
                        // Mirror of the buy GTD: a limit sell that is already
                        // marketable at submit (best_bid ≥ limit) executes
                        // immediately as a taker against the bids ≥ limit and only
                        // the remainder rests; a `post_only` that crosses is
                        // rejected by the venue, not parked.
                        let marketable = self
                            .eff_best_bid(&req.token_id)
                            .is_some_and(|b| b >= *price);
                        if req.post_only && marketable {
                            return RouterResult::rejected(
                                "post_only would cross the spread — venue rejection",
                            );
                        }
                        let legs = if marketable {
                            self.walk_sell_limit(&req.token_id, *price, size_trunc)
                        } else {
                            Vec::new()
                        };
                        let covered: Decimal = legs.iter().map(|l| l.size).sum();
                        let remaining = size_trunc - covered;
                        // With `track_inventory` the sell (marketable part
                        // + resting remainder) REQUIRES available simulated
                        // inventory — shares are never fabricated. The remainder
                        // is reserved in `reserved_sell` while it rests.
                        if self.research.lock().track_inventory {
                            let disp = self
                                .simulated_inventory()
                                .get(&req.token_id)
                                .copied()
                                .unwrap_or_default()
                                - self
                                    .reserved_sell
                                    .lock()
                                    .get(&req.token_id)
                                    .copied()
                                    .unwrap_or_default();
                            if disp < size_trunc {
                                return RouterResult::rejected(
                                    "insufficient inventory for LimitSell (track_inventory)",
                                );
                            }
                        }
                        let order_id = self.next_order_id(&req.token_id);
                        if !legs.is_empty() {
                            let cash_in = self.legs_cash(FillSide::Sell, &legs);
                            {
                                let mut bal = self.balance.lock();
                                *bal += cash_in;
                                let _ = self.wallet_bal_tx.send(Some(*bal));
                            }
                            // The marketable part consumes inventory.
                            self.adjust_inventory(&req.token_id, -covered);
                            self.consume_depth(&req.token_id, false, &legs_pairs(&legs));
                            let (exec, fill_price) =
                                self.legs_execution(&order_id, &req, FillSide::Sell, &legs);
                            // Our sell is the TAKER here: side "SELL",
                            // role "taker" (no maker inversion).
                            self.emit_execution(
                                exec,
                                &req.market_cid,
                                fill_price,
                                "SELL",
                                "taker",
                                None,
                                req.strategy,
                                req.real_clear_at,
                            );
                        }
                        if remaining > Decimal::ZERO {
                            // Queue position: the ask rests on the ASKS side
                            // — `queue_ahead` is only the EXTERNAL queue
                            // (residual asks at our price); earlier own orders
                            // are summed live each tick.
                            let residual = self.residual_level_size(&req.token_id, true, *price);
                            let prev_level = self.raw_level_size(&req.token_id, true, *price);
                            let mut pending = self.pending.lock();
                            pending.insert(
                                order_id.clone(),
                                PendingLimit {
                                    req: {
                                        let mut r = req.clone();
                                        r.kind = OrderKind::LimitSell {
                                            price: *price,
                                            size: remaining,
                                            tif: TimeInForce::Gtd,
                                        };
                                        r
                                    },
                                    queue_ahead: residual,
                                    prev_level,
                                    seq: self.resting_seq.fetch_add(1, Ordering::Relaxed),
                                    filled: Decimal::ZERO,
                                    // Resting sells reserve no cash —
                                    // they sell their own inventory (they reserve
                                    // SHARES in `reserved_sell`).
                                    reserved_unit: Decimal::ZERO,
                                    since_ms: (self.clock)(),
                                    degraded_evidence: false,
                                },
                            );
                            // Share reserve of the remainder — mirror of
                            // `reserved_buy` on the inventory basis. Always kept;
                            // only REQUIRED with `track_inventory`.
                            *self
                                .reserved_sell
                                .lock()
                                .entry(req.token_id.clone())
                                .or_default() += remaining;
                        }
                        RouterResult::no_fill(Some(order_id))
                    }
                    // No caller emits LimitSell FOK/FAK today (only the PM unwind,
                    // always GTD) — an unexercised implementation is a future bug.
                    TimeInForce::Fok | TimeInForce::Fak => {
                        RouterResult::rejected("LimitSell only supports GTD")
                    }
                },

                OrderKind::Cancel { order_id } => {
                    self.remove_pending(order_id.as_str());
                    RouterResult::no_fill(None)
                }
            };
            self.maybe_unknown(res)
        })
    }

    /// Override of the YES+NO pair. The sequential default slept the latency
    /// TWICE and drew independent liquidity draws → P(pair) ≈ p² and "0 complete
    /// pairs" in calibration. On live post_orders is ONE request: a single
    /// latency, price evaluated against the same books, and a correlated draw.
    fn submit_pair(
        &self,
        req1: OrderRequest,
        req2: OrderRequest,
    ) -> Pin<Box<dyn Future<Output = (RouterResult, RouterResult)> + Send + '_>> {
        Box::pin(async move {
            if self.latency_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
            }
            let (
                OrderKind::Limit {
                    price: p1,
                    size: s1,
                    ..
                },
                OrderKind::Limit {
                    price: p2,
                    size: s2,
                    ..
                },
            ) = (&req1.kind, &req2.kind)
            else {
                let e = "sim submit_pair: both legs must be Limit";
                return (RouterResult::rejected(e), RouterResult::rejected(e));
            };

            // Price moved: both legs against the SAME post-latency books
            // (EFFECTIVE best ask — residual after this snapshot's takers).
            // Parity with live: each leg goes through `build_signed_market_buy`,
            // which truncates the size to a WHOLE number and aborts if it ends up 0 — on live the
            // batch POST never goes out, so here both are rejected.
            let s1t = s1.trunc_with_scale(0);
            let s2t = s2.trunc_with_scale(0);
            if s1t <= Decimal::ZERO || s2t <= Decimal::ZERO {
                let e = "submit_pair: size after truncation to whole is 0";
                return (RouterResult::rejected(e), RouterResult::rejected(e));
            }

            let moved1 = self.eff_best_ask(&req1.token_id).is_some_and(|a| a > *p1);
            let moved2 = self.eff_best_ask(&req2.token_id).is_some_and(|a| a > *p2);

            // Residual visible depth up to the limit — each leg
            // is FOK (all-or-nothing); with no book/depth the leg is rejected even if
            // the liquidity draw passed.
            let legs1 = self.walk_buy(&req1.token_id, *p1, s1t);
            let legs2 = self.walk_buy(&req2.token_id, *p2, s2t);
            let cov1: Decimal = legs1.iter().map(|l| l.size).sum();
            let cov2: Decimal = legs2.iter().map(|l| l.size).sum();

            // Liquidity: with probability pair_fill_correlation both legs share the draw
            // (the CLOB evaluates YES/NO of the same bracket at the same instant).
            let d1 = self.should_reject_fok(req1.strategy);
            let d2 = if self.override_fok(req2.strategy).is_some() {
                self.should_reject_fok(req2.strategy)
            } else if self.draw_uniform() < self.pair_fill_correlation {
                d1
            } else {
                self.should_reject_fok(req2.strategy)
            };
            let rej1 = moved1 || cov1 < s1t || d1;
            let rej2 = moved2 || cov2 < s2t || d2;

            // Pair balance under a single lock (accepted legs only) —
            // discounting the cash reserved by live GTDs.
            let cost1: Decimal = if rej1 {
                Decimal::ZERO
            } else {
                self.legs_cash(FillSide::Buy, &legs1)
            };
            let cost2: Decimal = if rej2 {
                Decimal::ZERO
            } else {
                self.legs_cash(FillSide::Buy, &legs2)
            };
            if cost1 + cost2 > Decimal::ZERO {
                let mut bal = self.balance.lock();
                if *bal - *self.reserved_buy.lock() < cost1 + cost2 {
                    let e = "insufficient funds for the pair";
                    return (RouterResult::rejected(e), RouterResult::rejected(e));
                }
                *bal -= cost1 + cost2;
                let _ = self.wallet_bal_tx.send(Some(*bal));
            }

            // Hedge id shared by both legs — the ARB gate
            // (complete pairs ≥90%) groups fills by this column.
            // Injectable clock + PER-INSTANCE sequence (formerly `static`:
            // two runs in the same process were not reproducible); on live
            // fills come from the user-WS without hedge_id — it stays NULL (paper-only gate).
            let hedge_id = Some(
                (self.clock)() * 1000
                    + (self.hedge_seq.fetch_add(1, Ordering::Relaxed) % 1000) as i64,
            );

            let leg = |req: &OrderRequest, legs: &[WalkLeg], rejected: bool, moved: bool| {
                if rejected {
                    RouterResult::rejected(if moved {
                        "price moved during latency"
                    } else {
                        "FOK rejected: insufficient liquidity"
                    })
                } else {
                    self.consume_depth(&req.token_id, true, &legs_pairs(legs));
                    // Each accepted leg adds to the simulated inventory.
                    self.adjust_inventory(&req.token_id, legs.iter().map(|l| l.size).sum());
                    let oid = self.next_order_id(&req.token_id);
                    let (exec, fill_price) = self.legs_execution(&oid, req, FillSide::Buy, legs);
                    self.emit_execution(
                        exec,
                        &req.market_cid,
                        fill_price,
                        "BUY",
                        "taker",
                        hedge_id,
                        req.strategy,
                        req.real_clear_at,
                    );
                    RouterResult::no_fill(Some(oid))
                }
            };
            (
                self.maybe_unknown(leg(&req1, &legs1, rej1, moved1)),
                self.maybe_unknown(leg(&req2, &legs2, rej2, moved2)),
            )
        })
    }

    fn cancel(
        &self,
        _market_cid: &str,
        order_id: &str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send + '_>> {
        let order_id = order_id.to_string();
        Box::pin(async move {
            // The cancel also TRAVELS — same operational latency as the
            // submit (`latency_ms`; on the replay path it is 0 because the
            // latency lives in the `DeferredSimRouter`'s virtual queue — the research profile's
            // `latencies.*` stages are applied by the
            // queue, never by sleeps inside the router). During the flight the
            // resting order stays on the book: a crossing book that arrives first fills
            // it and the cancel loses — exactly as on live, where a
            // late cancel arrives when the order no longer exists.
            //
            // With `latency_ms > 0` the sleep alters the nominal
            // timing profile of the router task — a DELIBERATE decision:
            // the fidelity benefit (a cancel that can lose against an
            // in-flight fill, as on live) outweighs the timing
            // alteration. It does not apply in replay (latency in the virtual queue).
            if self.latency_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
            }
            self.remove_pending(&order_id);
            Ok(())
        })
    }

    /// `pending` is exactly the set of GTDs still live
    /// in the simulator (filled or cancelled ones are removed from it immediately,
    /// see `tick_book`/`cancel`) — the executor uses it to remove from its local mapping
    /// (`active_gtd`) any entry whose order_id is no longer here.
    fn live_gtd_order_ids(
        &self,
    ) -> Pin<Box<dyn Future<Output = Option<std::collections::HashSet<String>>> + Send + '_>> {
        Box::pin(async move { Some(self.pending.lock().keys().cloned().collect()) })
    }

    /// Unknown reconciliation: the sim knows the exact state — `pending`
    /// for live orders (with `filled` accumulated from partials), `gone` for those that
    /// left the book without filling completely (cancel/expiry — on live the
    /// venue reports them CANCELED/expired, not "never existed"),
    /// `exec_log` for fully filled ones, `NotFound` for the rest.
    fn order_status(
        &self,
        order_id: &str,
    ) -> Pin<Box<dyn Future<Output = Option<RouterOrderStatus>> + Send + '_>> {
        let order_id = order_id.to_string();
        Box::pin(async move {
            if let Some(lim) = self.pending.lock().get(&order_id) {
                let size = match &lim.req.kind {
                    OrderKind::Limit { size, .. } | OrderKind::LimitSell { size, .. } => *size,
                    _ => Decimal::ZERO,
                };
                let remaining = size - lim.filled;
                return Some(RouterOrderStatus::Live { remaining });
            }
            if let Some(matched) = self.gone.lock().get(&order_id) {
                return Some(RouterOrderStatus::Gone { matched: *matched });
            }
            let matched: Decimal = self
                .exec_log
                .lock()
                .iter()
                .filter(|e| e.order_id == order_id)
                .map(|e| e.shares)
                .sum();
            if matched > Decimal::ZERO {
                Some(RouterOrderStatus::Gone { matched })
            } else {
                Some(RouterOrderStatus::NotFound)
            }
        })
    }

    /// Unknown reconciliation: live resting orders with attributes to
    /// match intents that lost their `order_id` (unanswered POST).
    fn open_gtd_orders(
        &self,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<RouterOpenOrder>>> + Send + '_>> {
        Box::pin(async move {
            let orders = self
                .pending
                .lock()
                .iter()
                .map(|(oid, lim)| {
                    // The side is propagated from the kind — `Limit` is
                    // BUY by implicit contract, `LimitSell` is SELL;
                    // any other variant stays indeterminate.
                    let (side, price, size) = match &lim.req.kind {
                        OrderKind::Limit { price, size, .. } => {
                            (OpenOrderSide::Buy, *price, *size)
                        }
                        OrderKind::LimitSell { price, size, .. } => {
                            (OpenOrderSide::Sell, *price, *size)
                        }
                        _ => (OpenOrderSide::Unknown, Decimal::ZERO, Decimal::ZERO),
                    };
                    RouterOpenOrder {
                        order_id: oid.clone(),
                        market_cid: lim.req.market_cid.clone(),
                        token_id: lim.req.token_id.clone(),
                        side,
                        price,
                        remaining: size - lim.filled,
                        clear_at: lim.req.clear_at,
                    }
                })
                .collect();
            Some(orders)
        })
    }

    /// Unknown taker reconciliation in paper: the sim's execution log
    /// plays the role of `GET /data/trades` — each execution returns its
    /// `order_id` (the identity that chaos mode's `unknown` hid).
    /// This makes the Unknown→trades→order_index path exercisable end-to-end
    /// in paper with `paper_unknown_rate > 0`.
    fn trades_since(
        &self,
        after_ts_secs: i64,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<RouterTrade>>> + Send + '_>> {
        Box::pin(async move {
            let after_ms = after_ts_secs.saturating_mul(1000);
            Some(
                self.exec_log
                    .lock()
                    .iter()
                    .filter(|e| e.ts_ms >= after_ms)
                    .map(|e| RouterTrade {
                        trade_id: e.exec_id.clone(),
                        order_id: Some(e.order_id.clone()),
                        token_id: e.token_id.clone(),
                        is_buy: e.side == FillSide::Buy,
                        price: if e.shares > Decimal::ZERO {
                            e.cash / e.shares
                        } else {
                            Decimal::ZERO
                        },
                        size: e.shares,
                        match_ts_ms: e.ts_ms,
                    })
                    .collect(),
            )
        })
    }

    fn mode(&self) -> Mode {
        Mode::Paper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StrategyId;

    fn make_sim(fok_fill_rate: f64, correlation: f64) -> Arc<SimulatedRouter> {
        let (fill_tx, _fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(None);
        SimulatedRouter::new(
            0,
            fok_fill_rate,
            0,
            Decimal::ZERO,
            correlation,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100000),
            wallet_tx,
        )
    }

    fn fok_req(token: &str) -> OrderRequest {
        OrderRequest {
            market_cid: "cid-pair".into(),
            token_id: token.into(),
            kind: OrderKind::Limit {
                price: dec!(0.5),
                size: dec!(10),
                tif: TimeInForce::Fok,
            },
            neg_risk: false,
            strategy: StrategyId::Arb,
            clear_at: 9_999_999_999,
            signal: dec!(0.5),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        }
    }

    /// Book with ample depth on both sides for the `submit_pair`
    /// tests (no book → no fill; FOK walks real asks).
    fn book_pair() -> BookSnapshot {
        use crate::types::{HalfBook, Level, TokenBook};
        let bids = || HalfBook {
            levels: vec![Level {
                price: dec!(0.49),
                size: dec!(1000),
            }],
        };
        let asks = || HalfBook {
            levels: vec![Level {
                price: dec!(0.50),
                size: dec!(1000),
            }],
        };
        BookSnapshot {
            market_condition_id: "cid-pair".into(),
            yes: TokenBook {
                token_id: "tok-yes".into(),
                bids: bids(),
                asks: asks(),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-no".into(),
                bids: bids(),
                asks: asks(),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: 9_999_999_999,
            neg_risk: false,
            slug: "test-pair".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        }
    }

    #[tokio::test]
    async fn submit_pair_total_correlation_both_or_neither() {
        // With correlation=1.0 and fill_rate=0.7, over many pairs there must NEVER be
        // a filled leg and a rejected other — that is the bug that gave "0 complete pairs".
        let sim = make_sim(0.7, 1.0);
        sim.tick_book(&book_pair());
        let mut complete = 0;
        for _ in 0..50 {
            let (r1, r2) = sim.submit_pair(fok_req("tok-yes"), fok_req("tok-no")).await;
            assert_eq!(
                r1.rejected, r2.rejected,
                "pair with a single leg: correlation broken"
            );
            if !r1.rejected {
                complete += 1;
            }
        }
        assert!(
            complete > 20,
            "with fill_rate=0.7 enough pairs must complete (got {complete})"
        );
    }

    #[tokio::test]
    async fn submit_pair_fill_rate_one_always_completes() {
        let sim = make_sim(1.0, 1.0);
        sim.tick_book(&book_pair());
        let (r1, r2) = sim.submit_pair(fok_req("tok-yes"), fok_req("tok-no")).await;
        assert!(!r1.rejected && !r2.rejected);
        assert!(r1.order_id.is_some() && r2.order_id.is_some());
    }

    /// ARB-15M5M: with override 1.0 the pair is decided only by the book (no
    /// draw); without override the same rate 0.0 always rejects it.
    #[tokio::test]
    async fn per_strategy_override_fills_by_book_without_draw() {
        let sim = make_sim(0.0, 1.0);
        sim.tick_book(&book_pair());
        let (r1, r2) = sim.submit_pair(fok_req("tok-yes"), fok_req("tok-no")).await;
        assert!(r1.rejected && r2.rejected, "without override, rate 0.0 rejects");

        sim.set_fok_fill_rate_per_strategy(HashMap::from([(StrategyId::Arb, 1.0)]));
        for _ in 0..20 {
            let (r1, r2) = sim.submit_pair(fok_req("tok-yes"), fok_req("tok-no")).await;
            assert!(!r1.rejected && !r2.rejected, "the book has depth: it fills");
        }
    }

    #[tokio::test]
    async fn submit_pair_stamps_same_hedge_id_on_both_legs() {
        // Both legs of the pair must share the hedge_id, and different
        // pairs must carry different ids.
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(None);
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            Decimal::ZERO,
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100000),
            wallet_tx,
        );
        sim.tick_book(&book_pair());

        sim.submit_pair(fok_req("tok-yes"), fok_req("tok-no")).await;
        let f1 = fill_rx.try_recv().expect("fill leg 1");
        let f2 = fill_rx.try_recv().expect("fill leg 2");
        assert!(f1.hedge_id.is_some(), "the YES leg must carry a hedge_id");
        assert_eq!(f1.hedge_id, f2.hedge_id, "both legs share the hedge_id");

        sim.submit_pair(fok_req("tok-yes"), fok_req("tok-no")).await;
        let f3 = fill_rx.try_recv().expect("fill pair 2");
        assert_ne!(
            f1.hedge_id, f3.hedge_id,
            "different pairs → different hedge_id"
        );
    }

    /// The taker fee of a GTD must NOT be folded into the price before
    /// tick truncation — at high p the tick swallowed it entirely (0.95 + 0.00475 → 0.95:
    /// effective fee 0). The fee is charged at full precision, separate from the
    /// negotiated price (same criterion as the FOK/FAK paths).
    #[tokio::test]
    async fn gtd_high_price_buy_does_not_lose_fee_to_tick() {
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(Some(dec!(1000)));
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            dec!(0.10),
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100000),
            wallet_tx,
        );
        let req = OrderRequest {
            market_cid: "cid-fee".into(),
            token_id: "tok-yes".into(),
            kind: OrderKind::Limit {
                price: dec!(0.95),
                size: dec!(3),
                tif: TimeInForce::Gtd,
            },
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 9_999_999_999,
            signal: dec!(0.95),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        };
        assert!(sim.submit(req).await.order_id.is_some());
        // ask=0.50 <= limit=0.95 → crosses the spread and fills as taker.
        sim.tick_book(&book_with_fee("tok-yes", None));
        let fill = fill_rx.try_recv().expect("must fill");
        // fee/share = 0.10 × 0.95 × 0.05 = 0.00475 → all-in 0.95475. With the
        // earlier model (fee folded into the price + truncated to the 0.01 tick)
        // it came out 0.95: the whole commission vanished in the fill.
        assert_eq!(
            fill.price,
            dec!(0.95475),
            "the fee must not vanish through tick truncation"
        );
    }

    fn book_with_fee(token_id: &str, parabolic_fee_rate: Option<Decimal>) -> BookSnapshot {
        use crate::types::{HalfBook, Level, TokenBook};
        let half = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        BookSnapshot {
            market_condition_id: "cid-fee".into(),
            yes: TokenBook {
                token_id: token_id.into(),
                bids: half("0.49"),
                asks: half("0.50"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-other".into(),
                bids: half("0.49"),
                asks: half("0.50"),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: 9_999_999_999,
            neg_risk: false,
            slug: "test-fee".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate,
            tick_size: None,
            recv_instant: None,
        }
    }

    /// If the cached book carries `parabolic_fee_rate = Some(0)`,
    /// the FOK executes with NO fee (price == signed limit) — unlike the
    /// sim default (0.10), which would have added
    /// `fee_taker_parabolic_with_rate(0.10, 0.5, 1)` to the price.
    #[tokio::test]
    async fn sim_uses_book_fee_if_present() {
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(None);
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            dec!(0.10),
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100000),
            wallet_tx,
        );
        sim.tick_book(&book_with_fee("tok-yes", Some(Decimal::ZERO)));

        let _ = sim.submit(fok_req("tok-yes")).await;
        let fill = fill_rx.try_recv().expect("fill expected");
        // No fee: executed price == signed limit price (0.5).
        assert_eq!(
            fill.price,
            dec!(0.5),
            "parabolic_fee_rate=Some(0) must not charge a fee"
        );

        // Control: another token with a book but WITHOUT `parabolic_fee_rate`
        // → falls back to the 0.10 default and the executed price DOES carry the fee.
        // (With no book at all the FOK is rejected — the control needs
        // a book without a rate, not the absence of a book.)
        sim.tick_book(&book_with_fee("tok-no-fee", None));
        let _ = sim.submit(fok_req("tok-no-fee")).await;
        let fill2 = fill_rx.try_recv().expect("fill expected (control)");
        assert!(
            fill2.price > dec!(0.5),
            "without parabolic_fee_rate it must fall back to the default and charge a fee"
        );
    }

    /// The simulator's fee must be the PARABOLIC
    /// `rate × p × (1−p)`, not the triangular `rate × min(p,1−p)` — compared
    /// against the real formula for several values of p (including the
    /// worst-error case, p=0.50, where the triangular overcharged by 186%).
    #[test]
    fn taker_fee_uses_parabolic_not_triangular_formula() {
        let sim = make_sim(1.0, 1.0);
        // taker_fee_rate = 0 in make_sim → we force the explicit rate via the
        // cached book to isolate the formula from the default rate.
        sim.tick_book(&book_with_fee("tok-yes", Some(dec!(0.07))));

        for p in [dec!(0.50), dec!(0.85), dec!(0.10)] {
            let fee = sim.taker_fee("tok-yes", p);
            let expected = dec!(0.07) * p * (Decimal::ONE - p);
            let triangular = dec!(0.07) * p.min(Decimal::ONE - p);
            assert_eq!(fee, expected, "p={p}: must use the parabolic rate×p×(1−p)");
            if p != Decimal::ONE - p {
                assert_ne!(
                    fee, triangular,
                    "p={p}: must not match the triangular"
                );
            }
        }
    }

    fn book_with_bid(token_id: &str, bid_price: &str) -> BookSnapshot {
        use crate::types::{HalfBook, Level, TokenBook};
        let half = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        BookSnapshot {
            market_condition_id: "cid-ls".into(),
            yes: TokenBook {
                token_id: token_id.into(),
                bids: half(bid_price),
                asks: half("0.99"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-other".into(),
                bids: half("0.10"),
                asks: half("0.99"),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: 9_999_999_999,
            neg_risk: false,
            slug: "test-limit-sell".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        }
    }

    /// The maker ask GTD (`LimitSell`) does not fill while the
    /// best_bid stays below the requested price, and when it reaches it, it fills EXACTLY
    /// at our price (not the taker's) as `FillSide::Sell` — the taker BOUGHT
    /// against our ask (`role="maker"`, `side="BUY"`), and `user_ws::handle_trade`
    /// inverts the side.
    #[tokio::test]
    async fn sim_limit_sell_gtd_fills_only_when_bid_reaches_price() {
        use crate::types::FillSide;

        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(Some(dec!(1000)));
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            Decimal::ZERO,
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100000),
            wallet_tx,
        );

        let req = OrderRequest {
            market_cid: "cid-ls".into(),
            token_id: "tok-yes".into(),
            kind: OrderKind::LimitSell {
                price: dec!(0.45),
                size: dec!(3),
                tif: TimeInForce::Gtd,
            },
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 9_999_999_999,
            signal: dec!(0.45),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        };
        let result = sim.submit(req).await;
        assert!(
            !result.rejected && result.order_id.is_some(),
            "the GTD ask must be accepted with no immediate fill"
        );

        // best_bid = 0.34 < 0.45 → 0 fills.
        sim.tick_book(&book_with_bid("tok-yes", "0.34"));
        assert!(
            fill_rx.try_recv().is_err(),
            "must not fill with best_bid below the requested price"
        );

        // best_bid = 0.45 → fills EXACTLY at 0.45 (our price, not the taker's).
        sim.tick_book(&book_with_bid("tok-yes", "0.45"));
        let fill = fill_rx
            .try_recv()
            .expect("must fill when the bid reaches the price");
        assert_eq!(
            fill.price,
            dec!(0.45),
            "must fill exactly at our limit price"
        );
        assert_eq!(fill.size, dec!(3));
        assert_eq!(
            fill.side,
            FillSide::Sell,
            "maker of an ask fills as Sell (inverted taker BUY)"
        );
    }

    /// Paper/live parity — `clob_router.rs` sends GTDs with
    /// `clear_at + 60s` and `executor.rs` purges its local mapping at `clear_at + 65`;
    /// the sim must NOT purge bluntly at exactly `clear_at` (60s before live).
    #[tokio::test]
    async fn tick_book_purges_gtd_with_same_margin_as_live() {
        let sim = make_sim(1.0, 1.0);
        let now = chrono::Utc::now().timestamp();

        // GTD whose clear_at passed 30s ago: on live it would still be alive (clear_at+60 > now).
        let mut req = fok_req("tok-yes");
        req.kind = OrderKind::Limit {
            price: dec!(0.5),
            size: dec!(3),
            tif: TimeInForce::Gtd,
        };
        req.clear_at = now - 30;
        let r = sim.submit(req).await;
        let oid = r.order_id.expect("GTD must be accepted");

        sim.tick_book(&book_with_fee("tok-any-other", None)); // triggers the purge
        assert!(
            sim.pending.lock().contains_key(&oid),
            "clear_at 30s ago must not be purged yet"
        );

        // GTD whose clear_at passed 65s ago: it must now be purged (> clear_at+60).
        let mut req2 = fok_req("tok-yes");
        req2.kind = OrderKind::Limit {
            price: dec!(0.5),
            size: dec!(3),
            tif: TimeInForce::Gtd,
        };
        req2.clear_at = now - 65;
        let r2 = sim.submit(req2).await;
        let oid2 = r2.order_id.expect("GTD must be accepted");

        sim.tick_book(&book_with_fee("tok-any-other", None));
        assert!(
            !sim.pending.lock().contains_key(&oid2),
            "clear_at 65s ago must be purged"
        );
    }

    /// A BUY GTD that fills because `ask <= limit_price` crossed the
    /// spread — on live that executes as a real TAKER (fee), not as a fee-free
    /// maker. The sim must charge the same fee unless `post_only=true` (MM).
    #[tokio::test]
    async fn gtd_marketable_buy_charges_taker_fee() {
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(Some(dec!(1000)));
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            dec!(0.10),
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100000),
            wallet_tx,
        );

        let req = OrderRequest {
            market_cid: "cid-fee".into(),
            token_id: "tok-yes".into(),
            kind: OrderKind::Limit {
                price: dec!(0.55),
                size: dec!(3),
                tif: TimeInForce::Gtd,
            },
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 9_999_999_999,
            signal: dec!(0.55),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        };
        let r = sim.submit(req).await;
        assert!(r.order_id.is_some());

        // book_with_fee: ask=0.50 <= limit_price=0.55 → crosses the spread, fills.
        sim.tick_book(&book_with_fee("tok-yes", None));
        let fill = fill_rx.try_recv().expect("must fill");
        assert!(
            fill.price > dec!(0.55),
            "marketable without post_only must pay the taker fee, price={}",
            fill.price
        );
    }

    /// `post_only=true` (MM) IS genuine maker — no fee.
    #[tokio::test]
    async fn gtd_post_only_buy_still_has_no_fee() {
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(Some(dec!(1000)));
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            dec!(0.10),
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100000),
            wallet_tx,
        );

        let req = OrderRequest {
            market_cid: "cid-fee".into(),
            token_id: "tok-yes".into(),
            kind: OrderKind::Limit {
                price: dec!(0.55),
                size: dec!(3),
                tif: TimeInForce::Gtd,
            },
            neg_risk: false,
            strategy: StrategyId::MarketMaking,
            clear_at: 9_999_999_999,
            signal: dec!(0.55),
            post_only: true,
            real_clear_at: None,
            local_order_id: None,
        };
        let r = sim.submit(req).await;
        assert!(r.order_id.is_some());

        sim.tick_book(&book_with_fee("tok-yes", None));
        let fill = fill_rx.try_recv().expect("must fill");
        assert_eq!(
            fill.price,
            dec!(0.55),
            "post_only (MM) stays fee-free — genuine maker"
        );
    }

    /// The reserve of a BUY GTD covers the REAL cost of the fill
    /// (limit + estimated fee). Reserving only the notional left the fill overdrawn
    /// and the clamp to 0 gave that deficit away.
    #[tokio::test]
    async fn gtd_buy_reserve_includes_fee() {
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, wallet_rx) = watch::channel(Some(dec!(5.10)));
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            dec!(0.10),
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(5.10),
            wallet_tx,
        );
        let mk = |size: Decimal| OrderRequest {
            market_cid: "cid-fee".into(),
            token_id: "tok-yes".into(),
            kind: OrderKind::Limit {
                price: dec!(0.50),
                size,
                tif: TimeInForce::Gtd,
            },
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 9_999_999_999,
            signal: dec!(0.50),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        };
        // Real cost = (0.50 + 0.10×0.50×0.50) × 10 = 5.25. The balance is 5.10:
        // it covers the notional (5.0) but NOT the fee — with the old reserve
        // (notional-only) this order got in and its fill overdrew.
        assert!(
            sim.submit(mk(dec!(10))).await.rejected,
            "the reserve includes the fee: the notional fits but the real cost does not"
        );

        // With exactly enough balance (5.25) it does get in…
        sim.add_funds(dec!(0.15));
        assert!(sim.submit(mk(dec!(10))).await.order_id.is_some());
        // …and nothing is left free: another 1-share GTD (0.525) is rejected.
        assert!(
            sim.submit(mk(dec!(1))).await.rejected,
            "the fee is also reserved"
        );

        sim.tick_book(&book_with_fee("tok-yes", None));
        let fill = fill_rx.try_recv().expect("must fill");
        assert_eq!(fill.price, dec!(0.525), "all-in = limit + fee/share");
        assert_eq!(
            *wallet_rx.borrow(),
            Some(Decimal::ZERO),
            "the fill consumes exactly what was reserved — no overdraft"
        );

        // The reserve was fully released: with another 0.525 of balance another GTD fits.
        sim.add_funds(dec!(0.525));
        assert!(
            sim.submit(mk(dec!(1))).await.order_id.is_some(),
            "the release must leave no reserve residue"
        );
    }

    /// Two GTDs that fill in the SAME tick must be emitted in arrival
    /// order (`seq`). `pending` is a HashMap — without sorting, `retain` picks them
    /// up in a random order per process and the replay stops being
    /// reproducible (exec_id/exec_log change with the same seed and clock).
    #[tokio::test]
    async fn same_tick_fills_are_emitted_in_arrival_order() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        let a = sim
            .submit(limit(dec!(0.50), dec!(5), TimeInForce::Gtd))
            .await;
        let b = sim
            .submit(limit(dec!(0.45), dec!(5), TimeInForce::Gtd))
            .await;
        let oa = a.order_id.expect("GTD A accepted");
        let ob = b.order_id.expect("GTD B accepted");
        assert_ne!(oa, ob);

        // ask=0.45 crosses both limits (0.50 and 0.45) → both fill this tick.
        sim.tick_book(&book_levels(vec![lv("0.45", "10")], vec![], None));
        let f1 = rx.try_recv().expect("first fill");
        let f2 = rx.try_recv().expect("second fill");
        assert_eq!(
            (f1.order_id.as_deref(), f2.order_id.as_deref()),
            (Some(oa.as_str()), Some(ob.as_str())),
            "the fills of a tick must come out in arrival order (deterministic replay)"
        );
    }

    /// Live parity: `build_signed_market_buy` (FOK/FAK/GTD/pairs) and
    /// `submit_limit_sell` truncate the size to a WHOLE number — the contract's maker_amount
    /// only admits ≤4dp and whole×price(≤4dp) guarantees it. The sim
    /// executed/subtracted fractions that the venue would never accept.
    #[tokio::test]
    async fn fok_and_limit_sell_truncate_size_to_integer_like_live() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(vec![lv("0.50", "10")], vec![], None));

        // FOK of 4.7 shares: live would send 4 (build_signed_market_buy).
        let r = sim
            .submit(limit(dec!(0.51), dec!(4.7), TimeInForce::Fok))
            .await;
        assert!(r.order_id.is_some());
        let fill = rx.try_recv().expect("FOK fill");
        assert_eq!(fill.size, dec!(4), "size truncated to whole as on live");

        // GTD LimitSell of 4.7: the resting remainder is 4 (live truncates the same).
        let sell = req_kind(OrderKind::LimitSell {
            price: dec!(0.60),
            size: dec!(4.7),
            tif: TimeInForce::Gtd,
        });
        let oid = sim.submit(sell).await.order_id.expect("LimitSell accepted");
        let open = sim.open_gtd_orders().await.expect("orders");
        let resting = open.iter().find(|o| o.order_id == oid).expect("live");
        assert_eq!(resting.remaining, dec!(4), "remainder truncated to whole");

        // Sub-share FOK: live aborts the build — here, rejection.
        let r2 = sim
            .submit(limit(dec!(0.51), dec!(0.7), TimeInForce::Fok))
            .await;
        assert!(r2.rejected, "size <1 after truncation → rejection as on live");
    }

    /// (side note): the fill of a BUY GTD must be clipped to the
    /// depth ACTUALLY visible on the ask at the tick, not fill the full `size`
    /// of the GTD if the book shows less liquidity than that.
    #[tokio::test]
    async fn gtd_buy_trims_to_visible_ask_depth() {
        use crate::types::{HalfBook, Level, TokenBook};
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, _) = watch::channel(Some(dec!(1000)));
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            Decimal::ZERO,
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100000),
            wallet_tx,
        );
        let req = OrderRequest {
            market_cid: "cid-depth".into(),
            token_id: "tok-yes".into(),
            kind: OrderKind::Limit {
                price: dec!(0.55),
                size: dec!(10),
                tif: TimeInForce::Gtd,
            },
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 9_999_999_999,
            signal: dec!(0.55),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        };
        let r = sim.submit(req).await;
        assert!(r.order_id.is_some());

        // Ask with only 2 shares of depth — the GTD asks for 10.
        let half = |p: &str, s: Decimal| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: s,
            }],
        };
        let book = BookSnapshot {
            market_condition_id: "cid-depth".into(),
            yes: TokenBook {
                token_id: "tok-yes".into(),
                bids: half("0.49", dec!(100)),
                asks: half("0.50", dec!(2)),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-other".into(),
                bids: half("0.49", dec!(100)),
                asks: half("0.50", dec!(100)),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: 9_999_999_999,
            neg_risk: false,
            slug: "test-depth".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        };
        sim.tick_book(&book);
        let fill = fill_rx.try_recv().expect("must fill partially");
        assert_eq!(
            fill.size,
            dec!(2),
            "must be clipped to the visible depth of the ask"
        );
    }

    /// `Sell{size: None}` is not implemented — it must be rejected
    /// explicitly instead of using `req.signal` (a probability) as if it were
    /// shares.
    #[tokio::test]
    async fn sell_without_size_is_explicitly_rejected() {
        let sim = make_sim(1.0, 1.0);
        let req = OrderRequest {
            market_cid: "cid-1".into(),
            token_id: "tok-yes".into(),
            kind: OrderKind::Sell { size: None },
            neg_risk: false,
            strategy: StrategyId::ThetaCapture,
            clear_at: 9_999_999_999,
            signal: dec!(0.5),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        };
        let result = sim.submit(req).await;
        assert!(
            result.rejected,
            "Sell{{size: None}} must be rejected explicitly"
        );
        assert!(result.order_id.is_none());
    }

    /// The accept/reject RNG must be decoupled from the
    /// `next_order_id` counter — both used to share `order_seq`.
    #[test]
    fn rng_decoupled_from_order_id_counter() {
        let sim = make_sim(0.5, 1.0);
        let before = sim.rng_state.load(Ordering::Relaxed);
        let _ = sim.next_order_id("tok-a");
        let _ = sim.next_order_id("tok-b");
        let _ = sim.next_order_id("tok-c");
        let after_ids = sim.rng_state.load(Ordering::Relaxed);
        assert_eq!(
            before, after_ids,
            "next_order_id must not advance the RNG"
        );

        sim.draw_uniform();
        let after_draw = sim.rng_state.load(Ordering::Relaxed);
        assert_eq!(
            after_draw,
            before + 1,
            "draw_uniform must advance its own counter"
        );
    }

    /// A buy GTD can pass the balance check AT
    /// SUBMIT but the available balance changes (other orders) before
    /// the queue finishes filling it — the overdraft is NO LONGER absorbed: clamping to
    /// 0 created cash (inventory paid for with an invisible deficit). The balance
    /// goes negative and blocks new buys until replenished by sells.
    #[tokio::test]
    async fn gtd_buy_fill_overdraft_stays_negative_without_panic() {
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, wallet_rx) = watch::channel(Some(dec!(5)));
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            Decimal::ZERO,
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(5),
            wallet_tx,
        );
        let req = OrderRequest {
            market_cid: "cid-buy".into(),
            token_id: "tok-yes".into(),
            kind: OrderKind::Limit {
                price: dec!(0.5),
                size: dec!(3),
                tif: TimeInForce::Gtd,
            },
            neg_risk: false,
            strategy: StrategyId::ParityMaker,
            clear_at: 9_999_999_999,
            signal: dec!(0.5),
            post_only: false,
            real_clear_at: None,
            local_order_id: None,
        };
        let result = sim.submit(req).await;
        assert!(
            result.order_id.is_some(),
            "the balance (5) covers the cost (1.5) at submit"
        );

        // Drain the balance to 0 BEFORE the GTD fills (another order took it).
        sim.add_funds(dec!(-5));

        sim.tick_book(&book_with_fee("tok-yes", None)); // ask=0.50 <= limit=0.50 → fills
        let fill = fill_rx
            .try_recv()
            .expect("the GTD must still fill (the book crossed)");
        assert_eq!(fill.size, dec!(3));
        assert_eq!(
            *wallet_rx.borrow(),
            Some(dec!(-1.5)),
            "the deficit stays accounted — the clamp to 0 used to give it away"
        );
    }

    /// `live_gtd_order_ids` must reflect exactly the GTDs live
    /// in `pending` — used, cancelled or filled ones disappear from there.
    #[tokio::test]
    async fn live_gtd_order_ids_reflects_pending() {
        let sim = make_sim(1.0, 1.0);
        let mut req = fok_req("tok-yes");
        req.kind = OrderKind::Limit {
            price: dec!(0.5),
            size: dec!(3),
            tif: TimeInForce::Gtd,
        };
        let r = sim.submit(req).await;
        let oid = r.order_id.expect("GTD must be accepted");

        let live = sim
            .live_gtd_order_ids()
            .await
            .expect("sim must support live_gtd_order_ids");
        assert!(
            live.contains(&oid),
            "the GTD just placed must show as live"
        );

        // It fills (book crosses) → must disappear from pending and hence from live_gtd_order_ids.
        sim.tick_book(&book_with_fee("tok-yes", None));
        let live_after = sim.live_gtd_order_ids().await.unwrap();
        assert!(
            !live_after.contains(&oid),
            "a filled GTD leg must no longer be reported as live"
        );
    }

    // ── Integrated close: execution against real depth ─────────────

    fn lv(p: &str, s: &str) -> crate::types::Level {
        crate::types::Level {
            price: p.parse().unwrap(),
            size: s.parse().unwrap(),
        }
    }

    /// Book for "tok-yes" with the given levels (asks ascending, bids descending) and an empty "tok-no".
    fn book_levels(
        asks: Vec<crate::types::Level>,
        bids: Vec<crate::types::Level>,
        fee: Option<Decimal>,
    ) -> BookSnapshot {
        use crate::types::{HalfBook, TokenBook};
        BookSnapshot {
            market_condition_id: "cid-sim".into(),
            yes: TokenBook {
                token_id: "tok-yes".into(),
                bids: HalfBook { levels: bids },
                asks: HalfBook { levels: asks },
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-no".into(),
                ..Default::default()
            },
            clear_at: 9_999_999_999,
            slug: "test-sim".into(),
            parabolic_fee_rate: fee,
            ..Default::default()
        }
    }

    /// Sim with no fee by default, fok_fill_rate=1, with an observable fill channel and balance.
    fn sim_obs(
        balance: Decimal,
    ) -> (
        Arc<SimulatedRouter>,
        mpsc::UnboundedReceiver<Fill>,
        watch::Receiver<Option<Decimal>>,
    ) {
        let (fill_tx, fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, wallet_rx) = watch::channel(None);
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            Decimal::ZERO,
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            balance,
            wallet_tx,
        );
        (sim, fill_rx, wallet_rx)
    }

    fn req_kind(kind: OrderKind) -> OrderRequest {
        OrderRequest {
            market_cid: "cid-sim".into(),
            token_id: "tok-yes".into(),
            kind,
            neg_risk: false,
            strategy: StrategyId::Arb,
            clear_at: 9_999_999_999,
            signal: dec!(0.5),
            post_only: true,
            real_clear_at: None,
            local_order_id: Some("loc-1".into()),
        }
    }

    fn limit(price: Decimal, size: Decimal, tif: TimeInForce) -> OrderRequest {
        req_kind(OrderKind::Limit { price, size, tif })
    }

    #[tokio::test]
    async fn fok_with_sufficient_depth_walks_levels() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(
            vec![lv("0.50", "4"), lv("0.51", "10")],
            vec![],
            None,
        ));
        let r = sim
            .submit(limit(dec!(0.51), dec!(10), TimeInForce::Fok))
            .await;
        assert!(!r.rejected);
        let f = rx.try_recv().expect("FOK with sufficient depth must fill");
        assert_eq!(f.size, dec!(10));
        // 4×0.50 + 6×0.51 = 5.06 → average price 0.506
        assert_eq!(f.price, dec!(0.506));
        assert_eq!(*wallet.borrow(), Some(dec!(94.94)));
        assert!(rx.try_recv().is_err(), "only one execution");
    }

    #[tokio::test]
    async fn fok_without_depth_up_to_limit_rejects_without_moving_cash() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(
            vec![lv("0.50", "4"), lv("0.52", "10")],
            vec![],
            None,
        ));
        let r = sim
            .submit(limit(dec!(0.51), dec!(10), TimeInForce::Fok))
            .await;
        assert!(
            r.rejected,
            "4 shares ≤ limit do not cover 10: FOK is all-or-nothing"
        );
        assert!(rx.try_recv().is_err(), "no fill");
        assert_eq!(*wallet.borrow(), Some(dec!(100)), "cash untouched");
    }

    #[tokio::test]
    async fn fak_fills_what_is_available_and_cancels_the_rest() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(
            vec![lv("0.50", "4"), lv("0.52", "10")],
            vec![],
            None,
        ));
        let r = sim
            .submit(limit(dec!(0.51), dec!(10), TimeInForce::Fak))
            .await;
        assert!(!r.rejected);
        let f = rx.try_recv().expect("partial FAK");
        assert_eq!((f.size, f.price), (dec!(4), dec!(0.50)));
        assert_eq!(*wallet.borrow(), Some(dec!(98)));
        let live = sim.live_gtd_order_ids().await.unwrap();
        assert!(
            live.is_empty(),
            "the rest of a FAK is cancelled, not left live"
        );
    }

    #[tokio::test]
    async fn market_walks_levels_by_amount() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(
            vec![lv("0.50", "4"), lv("0.60", "10")],
            vec![],
            None,
        ));
        let r = sim
            .submit(req_kind(OrderKind::Market {
                usdc_amount: dec!(5),
            }))
            .await;
        assert!(!r.rejected);
        let f = rx.try_recv().expect("market with depth must fill");
        // 4 shares at 0.50 (2 USDC) + 3 USDC / 0.60 = 5 shares → 9 shares for 5 USDC.
        assert_eq!(f.size, dec!(9));
        assert!((f.price * f.size - dec!(5)).abs() < dec!(0.000000000001));
        assert_eq!(*wallet.borrow(), Some(dec!(95)));
    }

    #[tokio::test]
    async fn market_without_book_rejects_without_fill() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        let r = sim
            .submit(req_kind(OrderKind::Market {
                usdc_amount: dec!(5),
            }))
            .await;
        assert!(
            r.rejected,
            "no book means no price: do not invent one from the signal"
        );
        assert!(rx.try_recv().is_err());
        assert_eq!(*wallet.borrow(), Some(dec!(100)));
    }

    #[tokio::test]
    async fn sell_without_book_or_without_bids_does_not_fill() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        let r = sim
            .submit(req_kind(OrderKind::Sell {
                size: Some(dec!(5)),
            }))
            .await;
        assert!(r.rejected, "sell without book must be rejected");
        assert!(rx.try_recv().is_err());

        sim.tick_book(&book_levels(vec![lv("0.50", "10")], vec![], None));
        let r = sim
            .submit(req_kind(OrderKind::Sell {
                size: Some(dec!(5)),
            }))
            .await;
        assert!(r.rejected, "sell without bids must be rejected");
        assert!(rx.try_recv().is_err(), "without depth nothing is sold");
    }

    #[tokio::test]
    async fn sell_taker_walks_bids() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(
            vec![],
            vec![lv("0.40", "3"), lv("0.30", "10")],
            None,
        ));
        let r = sim
            .submit(req_kind(OrderKind::Sell {
                size: Some(dec!(5)),
            }))
            .await;
        assert!(!r.rejected);
        let f = rx.try_recv().expect("sell with bids");
        // 3×0.40 + 2×0.30 = 1.80 → average price 0.36
        assert_eq!((f.size, f.price), (dec!(5), dec!(0.36)));
        assert_eq!(*wallet.borrow(), Some(dec!(101.80)));
    }

    #[tokio::test]
    async fn gtd_partial_leaves_remainder_and_cancel_releases_reserve() {
        let (sim, mut rx, _w) = sim_obs(dec!(10));
        let gtd = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await;
        let oid = gtd.order_id.expect("GTD accepted");

        // Reserve: 10×0.50 = 5 → 5 free; a FOK of 6 USDC does not fit.
        sim.tick_book(&book_levels(vec![lv("0.60", "10")], vec![], None));
        let r = sim
            .submit(limit(dec!(0.60), dec!(10), TimeInForce::Fok))
            .await;
        assert!(
            r.rejected,
            "cash reserved by the GTD cannot be spent"
        );
        assert!(rx.try_recv().is_err());

        // First partial fill: 4 shares.
        sim.tick_book(&book_levels(vec![lv("0.50", "4")], vec![], None));
        let f1 = rx.try_recv().expect("first partial fill");
        assert_eq!(f1.size, dec!(4));
        assert!(
            sim.live_gtd_order_ids().await.unwrap().contains(&oid),
            "remainder live"
        );

        // Second partial fill with a different book: 3 more shares.
        sim.tick_book(&book_levels(vec![lv("0.49", "3")], vec![], None));
        let f2 = rx.try_recv().expect("second partial fill");
        assert_eq!((f2.size, f2.price), (dec!(3), dec!(0.50)));
        assert!(sim.live_gtd_order_ids().await.unwrap().contains(&oid));

        // Cancel: frees 3×0.50 = 1.5 → 10 − 3.5 = 6.5 free; now the FOK of 6 fits.
        sim.cancel("cid-sim", &oid).await.unwrap();
        assert!(!sim.live_gtd_order_ids().await.unwrap().contains(&oid));
        sim.tick_book(&book_levels(vec![lv("0.60", "10")], vec![], None));
        let r = sim
            .submit(limit(dec!(0.60), dec!(10), TimeInForce::Fok))
            .await;
        assert!(
            !r.rejected,
            "after cancelling, the reserved remainder is free again"
        );
        assert_eq!(rx.try_recv().expect("fill FOK").size, dec!(10));
    }

    #[tokio::test]
    async fn gtd_same_snapshot_does_not_advance_queue_twice() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        // Previous queue: 5 bid shares at our price.
        sim.tick_book(&book_levels(
            vec![lv("0.99", "10")],
            vec![lv("0.50", "5")],
            None,
        ));
        let gtd = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await;
        assert!(gtd.order_id.is_some());

        let cross = book_levels(vec![lv("0.50", "3")], vec![lv("0.50", "5")], None);
        sim.tick_book(&cross); // the queue consumes 3 of 5
        sim.tick_book(&cross); // same snapshot: no new volume
        sim.tick_book(&cross);
        assert!(
            rx.try_recv().is_err(),
            "the same snapshot must not be consumed again for the queue"
        );

        // New book with 4: 2 finish the queue, 2 are for us.
        sim.tick_book(&book_levels(
            vec![lv("0.50", "4")],
            vec![lv("0.50", "5")],
            None,
        ));
        assert_eq!(
            rx.try_recv().expect("fill after emptying the queue").size,
            dec!(2)
        );
    }

    #[tokio::test]
    async fn two_takers_in_same_snapshot_do_not_share_depth() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        let book = book_levels(vec![lv("0.50", "6")], vec![], None);
        sim.tick_book(&book);
        assert!(
            !sim.submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
                .await
                .rejected
        );
        assert_eq!(rx.try_recv().unwrap().size, dec!(5));
        assert!(
            sim.submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
                .await
                .rejected,
            "only 1 share is left in this snapshot"
        );
        assert!(rx.try_recv().is_err());

        sim.tick_book(&book); // same content: the depth already taken stays taken
        assert!(
            sim.submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
                .await
                .rejected
        );

        sim.tick_book(&book_levels(vec![lv("0.50", "7")], vec![], None)); // new book
        assert!(
            !sim.submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
                .await
                .rejected
        );
    }

    #[tokio::test]
    async fn each_execution_carries_unique_exec_id_and_local_order_id() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        let gtd = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await;
        let oid = gtd.order_id.unwrap();
        sim.tick_book(&book_levels(vec![lv("0.50", "4")], vec![], None));
        sim.tick_book(&book_levels(vec![lv("0.50", "5")], vec![], None));
        let f1 = rx.try_recv().expect("partial 1");
        let f2 = rx.try_recv().expect("partial 2");
        assert_ne!(
            f1.exec_id, f2.exec_id,
            "each execution has its own identity"
        );
        for f in [&f1, &f2] {
            assert_ne!(
                f.exec_id.as_deref(),
                Some(oid.as_str()),
                "exec id ≠ order id"
            );
            assert_eq!(f.local_order_id.as_deref(), Some("loc-1"));
        }
        let ex = sim.executions();
        assert_eq!(ex.len(), 2);
        assert!(ex
            .iter()
            .all(|e| e.order_id == oid && e.local_order_id.as_deref() == Some("loc-1")));
        assert_eq!(f1.exec_id.as_deref(), Some(ex[0].exec_id.as_str()));
    }

    /// Regression: after a restart, `exec_seq`/`order_seq` reset to 0. Without the
    /// startup nonce (`boot_ms`), the first fill of each process reused
    /// `sim-exec-0`, and `INSERT OR IGNORE` over `idx_fills_exec_id` discarded it
    /// as a duplicate: the wallet debited the reserve but the position was never created.
    /// Two instances with different startups must produce different ids.
    #[tokio::test]
    async fn sim_ids_are_unique_across_restarts() {
        let (a, mut rx_a, _wa) = sim_deps_clock(1.0, Decimal::ZERO, 7, 1_700_000_000_000);
        let (b, mut rx_b, _wb) = sim_deps_clock(1.0, Decimal::ZERO, 7, 1_700_000_000_001);
        for sim in [&a, &b] {
            sim.tick_book(&book_levels(vec![lv("0.50", "30")], vec![], None));
            sim.submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
                .await;
        }
        let fa = rx_a.try_recv().expect("fill de A");
        let fb = rx_b.try_recv().expect("fill de B");
        assert_ne!(
            fa.exec_id, fb.exec_id,
            "exec_id must be unique across processes even if exec_seq resets"
        );
        assert_ne!(
            fa.order_id, fb.order_id,
            "order_id must be unique across processes even if order_seq resets"
        );
        assert!(
            fa.exec_id
                .as_deref()
                .unwrap()
                .starts_with("sim-exec-1700000000000-"),
            "the startup nonce is embedded in the exec_id: {:?}",
            fa.exec_id
        );
    }

    fn sim_deps(
        fok_fill_rate: f64,
        fee: Decimal,
        seed: u64,
    ) -> (
        Arc<SimulatedRouter>,
        mpsc::UnboundedReceiver<Fill>,
        watch::Receiver<Option<Decimal>>,
    ) {
        sim_deps_clock(fok_fill_rate, fee, seed, 1_700_000_000_000)
    }

    fn sim_deps_clock(
        fok_fill_rate: f64,
        fee: Decimal,
        seed: u64,
        clock_ms: i64,
    ) -> (
        Arc<SimulatedRouter>,
        mpsc::UnboundedReceiver<Fill>,
        watch::Receiver<Option<Decimal>>,
    ) {
        let (fill_tx, fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, wallet_rx) = watch::channel(None);
        let clock: SimClock = Arc::new(move || clock_ms);
        let sim = SimulatedRouter::with_deps(
            0,
            fok_fill_rate,
            0,
            fee,
            0.5,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(1000),
            wallet_tx,
            seed,
            clock,
        );
        (sim, fill_rx, wallet_rx)
    }

    /// Fill without `ts` (`handle_trade` stamps it with the real clock, outside the sim).
    fn fingerprint(f: &Fill) -> String {
        format!(
            "{:?}|{}|{}|{:?}|{:?}|{:?}|{:?}",
            f.side, f.price, f.size, f.order_id, f.exec_id, f.local_order_id, f.hedge_id
        )
    }

    #[tokio::test]
    async fn same_seed_and_clock_produce_same_fills() {
        async fn run() -> (Vec<String>, Vec<SimExecution>) {
            let (sim, mut rx, _w) = sim_deps(0.5, dec!(0.07), 42);
            let mut no = book_levels(vec![lv("0.40", "50")], vec![], None);
            no.yes.token_id = "tok-no".into();
            no.no.token_id = "tok-yes".into();
            sim.tick_book(&book_levels(
                vec![lv("0.50", "30"), lv("0.55", "30")],
                vec![lv("0.45", "20")],
                None,
            ));
            sim.tick_book(&no);
            let gtd = limit(dec!(0.52), dec!(8), TimeInForce::Gtd);
            let _ = sim.submit(gtd).await;
            for _ in 0..6 {
                let _ = sim
                    .submit(limit(dec!(0.55), dec!(3), TimeInForce::Fok))
                    .await;
                let mut no_leg = limit(dec!(0.40), dec!(2), TimeInForce::Fok);
                no_leg.token_id = "tok-no".into();
                let _ = sim
                    .submit_pair(limit(dec!(0.55), dec!(2), TimeInForce::Fok), no_leg)
                    .await;
            }
            sim.tick_book(&book_levels(
                vec![lv("0.51", "5")],
                vec![lv("0.45", "20")],
                None,
            ));
            let mut fills = Vec::new();
            while let Ok(f) = rx.try_recv() {
                fills.push(fingerprint(&f));
            }
            (fills, sim.executions())
        }
        let (a, ea) = run().await;
        let (b, eb) = run().await;
        assert!(
            !a.is_empty() && a.len() < 19,
            "there must be fills and rejections from the draw ({})",
            a.len()
        );
        assert_eq!(a, b);
        assert_eq!(ea, eb);
    }

    /// Fee fixtures: values cited from the official source (0.07 = crypto taker,
    /// official fee table + Gamma `feeSchedule`; 0.10 = historical test value;
    /// 0 = book with no fee). See FEE_FIXTURES_OFFICIALLY_VERIFIED.
    const FEE_FIXTURES: [Decimal; 3] = [dec!(0.07), dec!(0.10), dec!(0)];

    #[test]
    // The assertion on the constant IS the tripwire: it can only be true with the
    // official source cited next to the constant (docs.polymarket.com/trading/fees,
    // plus Gamma's feeSchedule).
    #[allow(clippy::assertions_on_constants)]
    fn fee_fixtures_cited_to_official_source() {
        assert!(FEE_FIXTURES_OFFICIALLY_VERIFIED);
        assert_eq!(FEE_FIXTURES[0], crate::fair_value::TAKER_FEE_RATE_PARABOLIC);
    }

    #[tokio::test]
    async fn fee_and_box_separate_and_mean_price_matches_box() {
        for rate in FEE_FIXTURES {
            let (sim, mut rx, wallet) = sim_deps(1.0, rate, 1);
            sim.tick_book(&book_levels(
                vec![lv("0.50", "4"), lv("0.51", "10")],
                vec![],
                None,
            ));
            assert!(
                !sim.submit(limit(dec!(0.51), dec!(10), TimeInForce::Fok))
                    .await
                    .rejected
            );
            let f = rx.try_recv().expect("fill");
            let ex = sim.executions().pop().expect("execution recorded");

            // Fee quantized to 5dp PER TIER (official precision).
            let fee = (rate * dec!(0.50) * dec!(0.50) * dec!(4)).round_dp(5)
                + (rate * dec!(0.51) * dec!(0.49) * dec!(6)).round_dp(5);
            assert_eq!(ex.notional, dec!(5.06), "rate={rate}");
            assert_eq!(ex.fee_usdc, fee, "rate={rate}");
            assert_eq!(ex.cash, dec!(5.06) + fee, "rate={rate}");
            assert_eq!(ex.shares, dec!(10));
            assert_eq!(ex.levels.len(), 2, "price traded per level");
            assert!(
                (f.price * f.size - ex.cash).abs() < dec!(0.000000000001),
                "rate={rate}: price×size = cash"
            );
            assert_eq!(
                *wallet.borrow(),
                Some(dec!(1000) - ex.cash),
                "rate={rate}: the cash comes out exact"
            );
        }
    }

    #[tokio::test]
    async fn fok_fill_rate_never_grants_fill_without_depth() {
        let (sim, mut rx, _w) = sim_deps(1.0, Decimal::ZERO, 1);
        sim.tick_book(&book_levels(vec![lv("0.50", "100")], vec![], None));
        // The NO leg has no book: it is rejected even though fok_fill_rate=1; the YES leg fills (no atomicity).
        let mut no_leg = limit(dec!(0.50), dec!(5), TimeInForce::Fok);
        no_leg.token_id = "tok-no".into();
        let (r1, r2) = sim
            .submit_pair(limit(dec!(0.50), dec!(5), TimeInForce::Fok), no_leg)
            .await;
        assert!(!r1.rejected && r2.rejected);
        assert_eq!(rx.try_recv().unwrap().size, dec!(5));
        assert!(rx.try_recv().is_err());

        let (sim0, mut rx0, _w0) = sim_deps(0.0, Decimal::ZERO, 1);
        sim0.tick_book(&book_levels(vec![lv("0.50", "100")], vec![], None));
        assert!(
            sim0.submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
                .await
                .rejected,
            "fok_fill_rate=0 rejects even with depth (competition/latency)"
        );
        assert!(rx0.try_recv().is_err());
    }

    // ── Sim chaos (integrated close): Unknown injection end-to-end ────────────

    /// With `paper_unknown_rate = 0` (default) the sim is deterministic: an
    /// accepted FOK returns `no_fill`, never `unknown`. Chaos mode is opt-in.
    #[tokio::test]
    async fn chaos_disabled_never_returns_unknown() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(vec![lv("0.50", "100")], vec![], None));
        let r = sim
            .submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
            .await;
        assert!(!r.unknown && !r.rejected);
        assert!(rx.try_recv().is_ok(), "the fill flows as before");
    }

    /// The chaos invariant: `unknown` does NOT mean "nothing happened". The order
    /// was executed at the venue (the fill arrives on the channel, like a user_ws
    /// trade in live) but the ACK was lost. This is exactly the scenario that the
    /// Unknown journal, mirror and reconciler must match.
    #[tokio::test]
    async fn chaos_unknown_returns_unknown_but_execution_is_real() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        sim.set_unknown_rate(1.0);
        sim.tick_book(&book_levels(vec![lv("0.50", "100")], vec![], None));
        let r = sim
            .submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
            .await;
        assert!(
            r.unknown && !r.rejected,
            "rate=1.0: every accepted submit reports unknown"
        );
        let f = rx
            .try_recv()
            .expect("the order was executed: the fill arrives as in live");
        assert_eq!(f.size, dec!(5));
        assert_eq!(f.local_order_id.as_deref(), Some("loc-1"));
        assert!(f.order_id.is_some());
    }

    /// Chaos does not turn `rejected` into `unknown`: a pre-submit rejection
    /// (no book) means the order NEVER left for the venue, so there is no ACK to lose.
    #[tokio::test]
    async fn chaos_unknown_does_not_convert_rejected() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        sim.set_unknown_rate(1.0);
        // No book: pre-venue rejection.
        let r = sim
            .submit(limit(dec!(0.50), dec!(5), TimeInForce::Fok))
            .await;
        assert!(
            r.rejected && !r.unknown,
            "a pre-submit rejection cannot be a lost ACK"
        );
        assert!(rx.try_recv().is_err());
    }

    /// In `submit_pair` chaos applies per leg: with rate=1.0, both accepted legs
    /// report `unknown` and their fills flow. This reproduces the live scenario
    /// "pair executed, batch ACK lost".
    #[tokio::test]
    async fn chaos_unknown_in_pair_both_legs_unknown_with_fill() {
        let (sim, mut rx, _w) = sim_obs(dec!(100));
        sim.set_unknown_rate(1.0);
        sim.tick_book(&book_pair());
        let mut no_leg = limit(dec!(0.50), dec!(5), TimeInForce::Fok);
        no_leg.token_id = "tok-no".into();
        let (r1, r2) = sim
            .submit_pair(limit(dec!(0.50), dec!(5), TimeInForce::Fok), no_leg)
            .await;
        assert!(r1.unknown && r2.unknown, "pair executed, ACK lost");
        assert!(!r1.rejected && !r2.rejected);
        let f1 = rx.try_recv().expect("fill leg YES");
        let f2 = rx.try_recv().expect("fill leg NO");
        assert_eq!(f1.size, dec!(5));
        assert_eq!(f2.size, dec!(5));
        assert_eq!(
            f1.hedge_id, f2.hedge_id,
            "the chaotic pair keeps the shared hedge_id"
        );
    }

    // ── Queue position (model Nautilus L2): queue_ahead + deltas ────────

    /// A resting order is NOT fill-eligible while `queue_ahead > 0`: the flow
    /// crossing our price must first empty the queue ahead of it.
    #[tokio::test]
    async fn resting_does_not_fill_while_queue_has_volume_ahead() {
        let (sim, mut rx, _w) = sim_obs(dec!(1000));
        // Queue: 5 bid shares at 0.50 were already there when ours arrived.
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.50", "5")],
            None,
        ));
        let r = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await;
        assert!(r.order_id.is_some());

        // The ask crosses our price with 3 (< queue 5): it only consumes the queue.
        sim.tick_book(&book_levels(
            vec![lv("0.50", "3")],
            vec![lv("0.50", "5")],
            None,
        ));
        assert!(
            rx.try_recv().is_err(),
            "queue_ahead>0: the cross eats the queue, it does not fill us"
        );

        // Another cross of 1: 3+1 < 5, so we are still not fill-eligible.
        sim.tick_book(&book_levels(
            vec![lv("0.50", "1")],
            vec![lv("0.50", "5")],
            None,
        ));
        assert!(rx.try_recv().is_err(), "the queue has not been emptied yet");
    }

    /// When the crossing flow exceeds the queue ahead, only the EXCESS fills us:
    /// a capped partial fill, with the remainder live in `pending`.
    #[tokio::test]
    async fn resting_fills_only_with_excess_over_queue() {
        let (sim, mut rx, _w) = sim_obs(dec!(1000));
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.50", "5")],
            None,
        ));
        let r = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await;
        let oid = r.order_id.expect("GTD accepted");

        // Cross of 8: 5 empty the queue, and the excess (3) partially fills us.
        sim.tick_book(&book_levels(
            vec![lv("0.50", "8")],
            vec![lv("0.50", "5")],
            None,
        ));
        let f = rx
            .try_recv()
            .expect("partial fill from the excess over the queue");
        assert_eq!(f.size, dec!(3), "only the excess over the queue fills");
        assert_eq!(f.price, dec!(0.50), "post_only → maker, our price");
        let Some(RouterOrderStatus::Live { remaining }) = sim.order_status(&oid).await else {
            panic!("the remainder must stay live in pending");
        };
        assert_eq!(remaining, dec!(7));
    }

    /// The displayed size at our level dropping between snapshots IS the
    /// evidence of consumption (Nautilus model for L2 data, without a
    /// trade tape). It drains the queue but never produces a fill on its own:
    /// a fill requires crossing flow. We cannot tell whether the level shrank from
    /// trades or from cancels, so without a trade-through we conservatively do not fill.
    #[tokio::test]
    async fn queue_advances_by_level_l2_delta_without_inventing_fills() {
        let (sim, mut rx, _w) = sim_obs(dec!(1000));
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.50", "5")],
            None,
        ));
        let r = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await;
        assert!(r.order_id.is_some());

        // The bids at our price shrink 5 → 2 without the ask crossing: the queue
        // drops to 2, but there is no trade-through evidence, so no fill.
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.50", "2")],
            None,
        ));
        assert!(
            rx.try_recv().is_err(),
            "the shrink consumes queue, it does not invent fills"
        );

        // The level disappears entirely (2 → 0): queue empty, order alive and still
        // unfilled. It may have been cancels, not necessarily a sweep.
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.49", "8")],
            None,
        ));
        assert!(rx.try_recv().is_err());

        // Now the ask crosses with the queue already clear → fills completely
        // (price-through keeps the existing semantics).
        sim.tick_book(&book_levels(
            vec![lv("0.50", "20")],
            vec![lv("0.49", "8")],
            None,
        ));
        let f = rx.try_recv().expect("clear queue + cross → complete fill");
        assert_eq!(f.size, dec!(10));
    }

    /// Two of our resting orders at the same level share the queue in arrival
    /// order: the second order counts the first one in its `queue_ahead`, so the
    /// same flow serializes them. The first fills first, and its consumption also
    /// counts against the queue of the one behind it.
    #[tokio::test]
    async fn two_resting_at_same_level_are_serialized() {
        let (sim, mut rx, _w) = sim_obs(dec!(1000));
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.50", "5")],
            None,
        ));
        let a = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await
            .order_id
            .expect("GTD A");
        let b = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await
            .order_id
            .expect("GTD B");
        // A: queue_ahead = 5 (the book). B: queue_ahead = 5 + 10 (A is ahead).

        // Cross of 8: A fills the excess (3); B only consumes queue (15 → 7).
        sim.tick_book(&book_levels(
            vec![lv("0.50", "8")],
            vec![lv("0.50", "5")],
            None,
        ));
        let f = rx.try_recv().expect("only A is fill-eligible");
        assert_eq!(f.order_id.as_deref(), Some(a.as_str()));
        assert_eq!(f.size, dec!(3));
        assert!(rx.try_recv().is_err(), "B is still behind A in the queue");

        // Cross of 12: A completes its 7 remaining; the excess (5) reaches B.
        sim.tick_book(&book_levels(
            vec![lv("0.50", "12")],
            vec![lv("0.50", "5")],
            None,
        ));
        let mut fills = [rx.try_recv().unwrap(), rx.try_recv().unwrap()];
        fills.sort_by_key(|f| f.order_id.clone());
        assert_eq!(fills[0].order_id.as_deref(), Some(a.as_str()));
        assert_eq!(fills[0].size, dec!(7), "A completes first");
        assert_eq!(fills[1].order_id.as_deref(), Some(b.as_str()));
        assert_eq!(fills[1].size, dec!(5), "the excess serializes to B");
    }

    /// If the first resting order at the level leaves the book unfilled (cancel),
    /// its remainder stops counting in the queue of the later ones. This is the
    /// real mechanic: whoever cancels frees their position in the queue.
    #[tokio::test]
    async fn first_cancel_speeds_up_second_queue() {
        let (sim, mut rx, _w) = sim_obs(dec!(1000));
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.50", "5")],
            None,
        ));
        let a = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await
            .order_id
            .expect("GTD A");
        let b = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await
            .order_id
            .expect("GTD B");

        sim.cancel("cid-sim", &a).await.unwrap();
        // B: queue_ahead drops from 15 (5 book + 10 from A) to 5, since A is gone.
        sim.tick_book(&book_levels(
            vec![lv("0.50", "8")],
            vec![lv("0.50", "5")],
            None,
        ));
        let f = rx.try_recv().expect("B fills with the position A freed");
        assert_eq!(f.order_id.as_deref(), Some(b.as_str()));
        assert_eq!(f.size, dec!(3), "8 crossing − 5 queue remaining");
    }

    /// Queue shared with our own takers: what one of our takers takes from a level
    /// consumes the queue ahead of the resting order there (via `consume_depth`).
    /// The same displayed size is not counted twice, but the resting order does
    /// see its position advance.
    #[tokio::test]
    async fn own_taker_also_consumes_resting_queue() {
        let (sim, mut rx, _w) = sim_obs(dec!(1000));
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.50", "5")],
            None,
        ));
        let r = sim
            .submit(limit(dec!(0.50), dec!(10), TimeInForce::Gtd))
            .await;
        assert!(r.order_id.is_some());

        // Our own sell taker takes 3 of the 5 bids at 0.50 —
        // it consumes real queue ahead of the resting order (queue_ahead 5 → 2).
        let s = sim
            .submit(req_kind(OrderKind::Sell {
                size: Some(dec!(3)),
            }))
            .await;
        assert!(!s.rejected);
        let _ = rx.try_recv().expect("fill of the sell taker");

        // Cross of 3: 2 were left in the queue → only the excess (1) fills us.
        sim.tick_book(&book_levels(
            vec![lv("0.50", "3")],
            vec![lv("0.50", "5")],
            None,
        ));
        let f = rx.try_recv().expect("fill of the GTD after our own consumption");
        assert_eq!(
            f.size,
            dec!(1),
            "our own sell had already taken 3 from the queue"
        );
    }

    /// The maker ask (LimitSell) rests on the ASKS side: its queue is the asks
    /// already displayed at its price, and the flow that fills it is bids at ≥ its
    /// price. This is the exact mirror of the buy GTD.
    #[tokio::test]
    async fn limit_sell_queue_is_the_asks_of_its_level() {
        let (sim, mut rx, _w) = sim_obs(dec!(1000));
        sim.tick_book(&book_levels(
            vec![lv("0.55", "4"), lv("0.60", "100")],
            vec![lv("0.40", "100")],
            None,
        ));
        let r = sim
            .submit(req_kind(OrderKind::LimitSell {
                price: dec!(0.55),
                size: dec!(10),
                tif: TimeInForce::Gtd,
            }))
            .await;
        assert!(r.order_id.is_some());

        // Bids at 0.55 with 3 (< ask queue of 4): consumes the queue, does not fill.
        sim.tick_book(&book_levels(
            vec![lv("0.55", "4"), lv("0.60", "100")],
            vec![lv("0.55", "3")],
            None,
        ));
        assert!(rx.try_recv().is_err(), "queue_ahead>0: only empties the queue");

        // Bids at 0.55 with 5: 1 finishes the queue, and the excess (4) partially fills.
        sim.tick_book(&book_levels(
            vec![lv("0.55", "4"), lv("0.60", "100")],
            vec![lv("0.55", "5")],
            None,
        ));
        let f = rx.try_recv().expect("fill of the ask after emptying the queue");
        assert_eq!((f.size, f.price), (dec!(4), dec!(0.55)));
        assert_eq!(f.side, FillSide::Sell, "maker ask fills as Sell");
    }

    // ── GTD marketable at submit (live parity) ───────────────────

    /// A buy GTD that is already marketable at submit (best_ask ≤ limit) executes
    /// IMMEDIATELY as a taker against the real asks, at the level's price, not at
    /// the limit. It used to be parked entirely in `pending` and filled on the next
    /// tick at the limit price: a worse price than live, and a late fill.
    #[tokio::test]
    async fn gtd_marketable_at_submit_executes_against_asks_not_at_limit() {
        let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
        let (wallet_tx, wallet_rx) = watch::channel(Some(dec!(100)));
        let sim = SimulatedRouter::new(
            0,
            1.0,
            0,
            dec!(0.10),
            1.0,
            fill_tx,
            Arc::new(MarketRegistry::new()),
            dec!(100),
            wallet_tx,
        );
        // Book already known at submit: ask 0.50 with plenty of depth, fee 0.10.
        sim.tick_book(&book_with_fee("tok-yes", None));

        let mut req = limit(dec!(0.55), dec!(3), TimeInForce::Gtd);
        req.post_only = false;
        req.token_id = "tok-yes".into();
        let r = sim.submit(req).await;
        assert!(r.order_id.is_some());

        // Immediate fill, WITHOUT another tick, at the ask 0.50 + fee/share
        // (0.10 × 0.5 × 0.5 = 0.025) → 0.525, not at the 0.55 limit.
        let f = fill_rx
            .try_recv()
            .expect("the marketable GTD fills at submit");
        assert_eq!(f.size, dec!(3));
        assert_eq!(f.price, dec!(0.525), "level price, not the limit");
        assert_eq!(*wallet_rx.borrow(), Some(dec!(98.425)));
        assert!(
            sim.live_gtd_order_ids().await.unwrap().is_empty(),
            "fully covered: nothing left resting"
        );
    }

    /// If the marketable depth does not cover the size, the executed part fills
    /// as a taker right away, and ONLY the remainder rests at the end of the level's queue.
    #[tokio::test]
    async fn gtd_marketable_partial_leaves_resting_remainder() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        // Only 4 shares at 0.50 ≤ limit 0.55; the GTD asks for 10.
        sim.tick_book(&book_levels(
            vec![lv("0.50", "4"), lv("0.60", "100")],
            vec![],
            None,
        ));
        let mut req = limit(dec!(0.55), dec!(10), TimeInForce::Gtd);
        req.post_only = false;
        let oid = sim.submit(req).await.order_id.expect("GTD accepted");

        let f = rx.try_recv().expect("the marketable part fills right away");
        assert_eq!((f.size, f.price), (dec!(4), dec!(0.50)), "taker at the ask");
        assert_eq!(*wallet.borrow(), Some(dec!(98)));
        let Some(RouterOrderStatus::Live { remaining }) = sim.order_status(&oid).await else {
            panic!("the remainder (6) must rest");
        };
        assert_eq!(remaining, dec!(6));
    }

    /// A `post_only` order that would cross the spread at submit is REJECTED by
    /// the venue. Parking it would invent maker fills that live never produces.
    #[tokio::test]
    async fn post_only_that_crosses_at_submit_is_rejected() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(vec![lv("0.50", "100")], vec![], None));

        // limit() → post_only=true; ask 0.50 ≤ limit 0.55 → crosses.
        let r = sim
            .submit(limit(dec!(0.55), dec!(3), TimeInForce::Gtd))
            .await;
        assert!(r.rejected, "post_only marketable = venue rejection");
        assert!(rx.try_recv().is_err(), "no fill");
        assert_eq!(*wallet.borrow(), Some(dec!(100)), "cash untouched");
        assert!(sim.live_gtd_order_ids().await.unwrap().is_empty());

        // Control: the SAME post_only without crossing (limit < ask) does rest.
        let ok = sim
            .submit(limit(dec!(0.45), dec!(3), TimeInForce::Gtd))
            .await;
        assert!(ok.order_id.is_some(), "post_only no marketable → resting");
    }

    /// Sell mirror: a marketable LimitSell (best_bid ≥ limit) executes immediately
    /// as a taker against the bids ≥ limit; the remainder rests.
    #[tokio::test]
    async fn limit_sell_marketable_executes_against_bids() {
        let (sim, mut rx, wallet) = sim_obs(dec!(100));
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.55", "4"), lv("0.40", "10")],
            None,
        ));
        let mut req = req_kind(OrderKind::LimitSell {
            price: dec!(0.50),
            size: dec!(6),
            tif: TimeInForce::Gtd,
        });
        req.post_only = false;
        let oid = sim.submit(req).await.order_id.expect("accepted");

        // 4 shares at 0.55 (the bids ≥ 0.50); remainder 2 resting on the asks.
        let f = rx.try_recv().expect("the marketable part sells right away");
        assert_eq!((f.size, f.price), (dec!(4), dec!(0.55)), "taker at the bid");
        assert_eq!(f.side, FillSide::Sell);
        assert_eq!(*wallet.borrow(), Some(dec!(102.20)));
        let Some(RouterOrderStatus::Live { remaining }) = sim.order_status(&oid).await else {
            panic!("the remainder (2) must rest");
        };
        assert_eq!(remaining, dec!(2));

        // post_only control: a LimitSell that crosses (bid 0.40 ≥ limit 0.35)
        // is rejected; the first sell already took the 0.55 bids.
        let r = sim
            .submit(req_kind(OrderKind::LimitSell {
                price: dec!(0.35),
                size: dec!(1),
                tif: TimeInForce::Gtd,
            }))
            .await;
        assert!(r.rejected, "LimitSell post_only marketable = rejection");
    }

    /// Flow is NOT counted twice between our own resting orders at the same level.
    /// A is ahead of B (seq). When a cross partially fills A and the next snapshot
    /// shows the level already consumed, a later small cross must go whole to A.
    /// B cannot fill with flow that, at the real venue, goes to the order ahead of it.
    /// The previous model subtracted the same external consumption twice
    /// (the absorbed crossing plus the level consumption in the post-trade snapshot),
    /// and baked A's ORIGINAL remainder into B's `queue_ahead`.
    #[tokio::test]
    async fn flow_is_not_counted_twice_between_own_resting_of_level() {
        let (sim, mut rx, _w) = sim_obs(dec!(1000));
        // bids@0.50 = 100 of external queue; asks far away.
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100")],
            vec![lv("0.50", "100")],
            None,
        ));
        let a = sim
            .submit(limit(dec!(0.50), dec!(30), TimeInForce::Gtd))
            .await
            .order_id
            .expect("GTD A");
        let b = sim
            .submit(limit(dec!(0.50), dec!(20), TimeInForce::Gtd))
            .await
            .order_id
            .expect("GTD B");

        // Tick 1 — crossed book: 120 asks at 0.50 against the level. The
        // 100 external ones absorb, A fills 20 (remainder 10); B nothing.
        sim.tick_book(&book_levels(
            vec![lv("0.60", "100"), lv("0.50", "120")],
            vec![lv("0.50", "100")],
            None,
        ));
        let f = rx.try_recv().expect("A fills 20 from the excess");
        assert_eq!(f.order_id.as_deref(), Some(a.as_str()));
        assert_eq!(f.size, dec!(20));
        assert!(rx.try_recv().is_err(), "B must not fill yet");

        // Tick 2 — post-trade: the level is consumed.
        sim.tick_book(&book_levels(vec![lv("0.60", "100")], vec![], None));
        assert!(rx.try_recv().is_err());

        // Tick 3 — cross of 5: real FIFO → all to A (rem 10 → 5). B = 0.
        sim.tick_book(&book_levels(vec![lv("0.50", "5")], vec![], None));
        let f = rx.try_recv().expect("A fills the 5");
        assert_eq!(f.order_id.as_deref(), Some(a.as_str()));
        assert_eq!(f.size, dec!(5));
        assert!(
            rx.try_recv().is_err(),
            "B cannot fill with flow that goes to A: double count (order {b})"
        );

        // Cross of 8: A finishes its 5 remaining and B fills the excess (3).
        sim.tick_book(&book_levels(vec![lv("0.50", "8")], vec![], None));
        let f1 = rx.try_recv().expect("A finishes");
        let f2 = rx.try_recv().expect("B fills the excess");
        assert_eq!(
            (f1.order_id.as_deref(), f1.size),
            (Some(a.as_str()), dec!(5))
        );
        assert_eq!(
            (f2.order_id.as_deref(), f2.size),
            (Some(b.as_str()), dec!(3))
        );
    }
}
