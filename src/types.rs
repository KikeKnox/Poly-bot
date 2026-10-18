//! Types shared across all bot modules.
//!
//! Rule: this file imports nothing from its own crate — only std types
//! and external crates — so there are no circular dependencies.

use crate::time::Instant;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// ── Operating mode ───────────────────────────────────────────────────────────

/// Bot operating mode.
///
/// Replaces the old `dry_run: bool`. The mode is injected when building the
/// `Executor` and is not consulted anywhere else: switching mode means picking
/// another `OrderRouter` implementation, not flipping a global flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Simulated: all orders go to the `SimulatedRouter`.
    Paper,
    /// Real: all orders go to the `ClobRouter`.
    Live,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Paper => write!(f, "paper"),
            Mode::Live => write!(f, "live"),
        }
    }
}

// ── Strategy identifier ──────────────────────────────────────────────────────

/// Strategy that originated an order or a fill.
///
/// Used for per-strategy kill switches, rate-limit counters and
/// PnL attribution.
///
/// `LadderArb` and `VolSurface` used to live here and were removed when they moved to
/// `crate::tools::ToolId` (observation tools, they never emit
/// OrderIntent): they never placed an order, so they never showed up in
/// `gtd_orders`/`positions`/`fills`, and removing them does not break deserialization
/// of historical data. `scan_observations` stores the string via
/// `ToolId::Display`, which keeps the same text "vol_surface"/"ladder_arb".
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyId {
    #[default]
    Arb,
    Directional,
    NearCertain,
    NearExpiry,
    MarketMaking,
    ParityMaker,
    DriftBox,
    OpenMomentum,
    ThetaCapture,
    PostCloseSweep,
    /// Scan-only phase of Polymarket Liquidity Rewards farming
    /// — never emits OrderIntent, only observes and logs (target `rf_scan`).
    RewardFarm,
    /// Wallet follower — copies entries/exits of a target wallet detected by
    /// polling `data-api.polymarket.com/activity`. It is NOT a `Strategy`
    /// (standalone task, `strategy/follower.rs`) — this variant exists only for
    /// fill attribution / kill switch / rate limit, reusing the same
    /// `RiskGuard`/`executor` infrastructure as the others. See the doc comment of
    /// `strategy/supervisor.rs::spawn` for the hot-toggle limitation.
    Follower,
    /// Paper buy markout, fixed ticket at the touch. Dedicated
    /// vehicle — see `strategy/g_rw1_maker.rs`.
    GRw1Maker,
    /// PAPER-ONLY strategy based on the numeric model
    /// (`src/tools/model/`). Consumes probabilities from the `ModelTool` tool
    /// (broadcast) and emits ONE taker FOK `OrderIntent` per market, held
    /// until settlement, no pyramiding, strict gating by horizon
    /// (BTC, exactly 5m) and freshness/edge/spread filters. NEVER live:
    /// the TOML must carry `paper_only=true`; the barrier fails CLOSED if
    /// `paper_only=false` (see `StrategyLimits::
    /// can_run` + the supervisor barrier for this specific variant).
    ModelBtc5m,
    /// ONLINE RUNTIME of RL_predictor (docs/rl_predictor_design/):
    /// linear softmax policy over 9 actions
    /// (BUY/HOLD/SELL) loaded fail-closed from a `paper_approved` artifact,
    /// with an online REINFORCE learner inside the bot (bounded worker,
    /// immutable snapshots per watch, versioned checkpoints). Consumes
    /// `ModelPrediction` as the `p_yes` feature — it is not a rename of the
    /// predictor. Structurally PAPER-ONLY: `Mode::Live` → panic in the ctor +
    /// supervisor barrier. Disabled by default until a REAL approved
    /// artifact exists — no fallback to synthetic weights.
    RlPredictor,
    /// ARB-15M5M: basket of
    /// two markets (15m + 5m of the same asset and same close) whose payout is ≥ 1
    /// depending on the order of the Chainlink TWAP-60 s strikes. Paper only until
    /// the pre-registered criterion is met.
    #[serde(rename = "arb_15m5m")]
    Arb15m5m,
}

impl StrategyId {
    pub const ALL: &'static [Self] = &[
        Self::Arb,
        Self::Directional,
        Self::NearCertain,
        Self::NearExpiry,
        Self::MarketMaking,
        Self::ParityMaker,
        Self::DriftBox,
        Self::OpenMomentum,
        Self::ThetaCapture,
        Self::PostCloseSweep,
        Self::RewardFarm,
        Self::Follower,
        Self::GRw1Maker,
        Self::ModelBtc5m,
        Self::RlPredictor,
        Self::Arb15m5m,
    ];
}

impl std::fmt::Display for StrategyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            StrategyId::Arb => "arb",
            StrategyId::Directional => "directional",
            StrategyId::NearCertain => "near_certain",
            StrategyId::NearExpiry => "near_expiry",
            StrategyId::MarketMaking => "market_making",
            StrategyId::ParityMaker => "parity_maker",
            StrategyId::DriftBox => "drift_box",
            StrategyId::OpenMomentum => "open_momentum",
            StrategyId::ThetaCapture => "theta_capture",
            StrategyId::PostCloseSweep => "post_close_sweep",
            StrategyId::RewardFarm => "reward_farm",
            StrategyId::Follower => "follower",
            StrategyId::GRw1Maker => "g_rw1_maker",
            StrategyId::ModelBtc5m => "model_btc5m",
            StrategyId::RlPredictor => "rl_predictor",
            StrategyId::Arb15m5m => "arb_15m5m",
        };
        write!(f, "{s}")
    }
}

/// `FromStr` is STRICT — an unknown name → `Err(())`.
///
/// This `match` used to never return `Err`: a wrong name (operator typo,
/// e.g. "post_close_swep") fell into a silent fallback to `StrategyId::Arb`
/// with only a `warn!`. That fallback was the SAME path used by the state controls
/// (`api/handlers.rs`, `telegram.rs`): their `let Ok(id) = ...
/// else { 400 }` branches were unreachable dead code. Result: a typo silently killed ARB,
/// the HTTP/Telegram response confirmed the name the user
/// typed (not the one actually touched), and the toggle was persisted to the TOML
/// and survived a restart.
///
/// The tolerant fallback STILL exists, but moved to [`StrategyId::from_db_str`]
/// — use it ONLY to deserialize history from the DB (fills/gtd_orders of
/// old sessions), never to interpret a name typed by a human.
impl std::str::FromStr for StrategyId {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "arb" | "arbitrage" => Ok(StrategyId::Arb),
            "directional" => Ok(StrategyId::Directional),
            "near_certain" => Ok(StrategyId::NearCertain),
            "near_expiry" => Ok(StrategyId::NearExpiry),
            "market_making" => Ok(StrategyId::MarketMaking),
            "parity_maker" => Ok(StrategyId::ParityMaker),
            "drift_box" => Ok(StrategyId::DriftBox),
            "open_momentum" => Ok(StrategyId::OpenMomentum),
            "theta_capture" => Ok(StrategyId::ThetaCapture),
            "post_close_sweep" => Ok(StrategyId::PostCloseSweep),
            "reward_farm" => Ok(StrategyId::RewardFarm),
            "follower" => Ok(StrategyId::Follower),
            "g_rw1_maker" => Ok(StrategyId::GRw1Maker),
            "model_btc5m" => Ok(StrategyId::ModelBtc5m),
            "rl_predictor" => Ok(StrategyId::RlPredictor),
            "arb_15m5m" => Ok(StrategyId::Arb15m5m),
            _ => Err(()),
        }
    }
}

impl StrategyId {
    /// TOLERANT deserialization, only for reading history already
    /// persisted in SQLite (`strategy` column of `fills`/`gtd_orders` from
    /// old sessions, including the `"recovered"`/`"unknown"`/`""` markers).
    /// Never fails: an unknown value falls back to `Arb` with a `warn!` (same
    /// behavior as the original `FromStr` before the strict-parse fix).
    /// Do NOT use to interpret a strategy name typed by
    /// a human (API/Telegram) — use `FromStr`/`.parse()` for that, which now
    /// does fail on an unknown name.
    pub fn from_db_str(s: &str) -> Self {
        use std::str::FromStr;
        match Self::from_str(s) {
            Ok(id) => id,
            Err(()) => {
                // Known history markers with no associated strategy → Arb without log noise.
                if !matches!(s, "recovered" | "unknown" | "") {
                    tracing::warn!(
                        strategy = s,
                        "unknown strategy variant in history; using Arb"
                    );
                }
                StrategyId::Arb
            }
        }
    }
}

// ── Book ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level {
    pub price: Decimal,
    pub size: Decimal,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HalfBook {
    /// Ask-side levels: sortedending (first = best ask = lowest price).
    /// Bid-side levels: sortedending (first = best bid = highest price).
    /// Ordering is guaranteed by market_feed/handle.rs and market_feed/resync.rs.
    pub levels: Vec<Level>,
}

impl HalfBook {
    /// Best ask (lowest price, first level of asks sorted).
    pub fn best_ask(&self) -> Option<Decimal> {
        self.levels.first().map(|l| l.price)
    }

    /// Best bid (highest price, first level of bids sorted).
    pub fn best_bid(&self) -> Option<Decimal> {
        self.levels.first().map(|l| l.price)
    }

    /// Size available at the best level.
    pub fn best_size(&self) -> Option<Decimal> {
        self.levels.first().map(|l| l.size)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenBook {
    pub token_id: String,
    pub bids: HalfBook,
    pub asks: HalfBook,
    /// PIPE-2: signed net flow (shares) of the tape (`last_trade_price`) in the
    /// `market_feed.tape_window_secs` window. `None` = tape disabled or the
    /// token never had recorded trades (absent signal, never blocks).
    #[serde(default)]
    pub flow_net: Option<Decimal>,
    /// PIPE-2: gross volume (shares, both directions) of the tape in the same window.
    #[serde(default)]
    pub flow_total: Option<Decimal>,
}

impl TokenBook {
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.best_ask()
    }
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.best_bid()
    }
}

/// Snapshot of a bracket: both YES and NO sides, with market metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BookSnapshot {
    pub market_condition_id: String,
    pub yes: TokenBook,
    pub no: TokenBook,
    pub market_open_ts: i64,
    pub clear_at: i64,
    pub neg_risk: bool,
    pub slug: String,
    /// Reference price (strike) of the underlying when the market opened.
    /// Populated from the Gamma API at discovery; `None` if the bracket opened before
    /// bot startup and there is no API data yet.
    pub reference_price: Option<Decimal>,
    /// WS-1: market taker fee in basis points, as reported by `MarketEntry`
    /// (Gamma `feeRateBps` or CLOB `/fee-rate` — see `market_feed::discover`).
    /// `None` if not yet known. Field currently without real readers outside
    /// discovery — the operational taker fee
    /// uses `parabolic_fee_rate`/`parabolic_fee_rate_or`.
    pub fee_rate_bps: Option<u32>,
    /// FEE-2: actual parabolic taker fee rate
    /// of the market (`MarketEntry.parabolic_fee_rate`, from `feeSchedule.rate` of
    /// Gamma). `None` if not yet known — use
    /// [`Self::parabolic_fee_rate_or`] with `fair_value::TAKER_FEE_RATE_PARABOLIC`
    /// as the fallback.
    pub parabolic_fee_rate: Option<Decimal>,
    /// TICK-1: minimum market tick (0.01 in the middle zone, 0.001 above 0.96/below 0.04);
    /// None → fallback 0.01.
    pub tick_size: Option<Decimal>,
    /// Local monotonic clock to measure book→signal latency. Not serialized.
    #[serde(skip)]
    pub recv_instant: Option<Instant>,
}

impl BookSnapshot {
    /// Seconds until the market closes, from now.
    pub fn secs_to_clear(&self) -> i64 {
        let now = crate::time::now().timestamp();
        self.clear_at.saturating_sub(now)
    }

    /// FEE-2: actual parabolic taker fee rate of the market
    /// (`parabolic_fee_rate`, already a fraction 0.0-1.0 — unlike
    /// `fee_rate_bps` it is not in basis points), or `fallback` (typically
    /// `fair_value::TAKER_FEE_RATE_PARABOLIC`) if the book does not carry it yet.
    pub fn parabolic_fee_rate_or(&self, fallback: Decimal) -> Decimal {
        self.parabolic_fee_rate.unwrap_or(fallback)
    }

    /// TICK-1: effective market tick; 0.01 if the feed has not reported it yet.
    pub fn tick(&self) -> Decimal {
        self.tick_size.unwrap_or(rust_decimal_macros::dec!(0.01))
    }

    /// Mid price of the YES token: (best_bid + best_ask) / 2. `None` if a side is missing.
    pub fn yes_mid(&self) -> Option<Decimal> {
        let bid = self.yes.best_bid()?;
        let ask = self.yes.best_ask()?;
        Some((bid + ask) / Decimal::TWO)
    }

    /// Mid price of the NO token.
    pub fn no_mid(&self) -> Option<Decimal> {
        let bid = self.no.best_bid()?;
        let ask = self.no.best_ask()?;
        Some((bid + ask) / Decimal::TWO)
    }
}

/// TICK-1: truncates (does NOT round, does NOT clamp) `price` to the scale of `tick`.
///
/// Assumes the Polymarket tick is always a power of 10 (0.1, 0.01, 0.001,
/// 0.0001…), so `tick.scale()` directly gives the number of decimals to
/// keep. No clamp: strategy callers rely on seeing out-of-range values
/// (e.g. negative or > 1) to skip.
pub fn truncate_to_tick(price: Decimal, tick: Decimal) -> Decimal {
    price.trunc_with_scale(tick.scale())
}

/// TICK-1: derives the tick from the price zone. Polymarket uses 0.001 when the price
/// is above 0.96 or below 0.04, and 0.01 in the middle zone (including the exact edges
/// 0.96 and 0.04, which are still the coarse zone).
pub fn tick_from_price(price: Decimal) -> Decimal {
    if price > rust_decimal_macros::dec!(0.96) || price < rust_decimal_macros::dec!(0.04) {
        rust_decimal_macros::dec!(0.001)
    } else {
        rust_decimal_macros::dec!(0.01)
    }
}

// ── Oracle ───────────────────────────────────────────────────────────────────

/// Price of the underlying published by `oracle::price`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetPrice {
    pub asset: String,
    pub price: Decimal,
    pub updated_at: DateTime<Utc>,
    #[serde(skip)]
    pub instant: Option<Instant>,
    /// PIPE-1: accumulated signed net flow in USDT (buy aggressor: +notional,
    /// sell aggressor: −notional). Only filled by `oracle::price`; the REST
    /// bootstrap publishes ZERO. Only reset with the process (survives
    /// Binance WS reconnections).
    #[serde(default)]
    pub cum_flow_quote: Decimal,
    /// PIPE-1: accumulated gross volume in USDT (|notional|, both directions).
    #[serde(default)]
    pub cum_vol_quote: Decimal,
    /// SPEC-SIGMA-FV §F3: timestamp of the TRADE in the Binance engine (ms) —
    /// not the local receipt instant. `None` if the source does not carry it
    /// (REST bootstrap) — `#[serde(default)]` so deserializing old snapshots
    /// does not break.
    #[serde(default)]
    pub exchange_ts_ms: Option<i64>,
}

impl AssetPrice {}

/// PIPE-4: accumulated (net, total) notional liquidated on Binance USDS-M
/// futures per asset, published by `oracle::liquidations`. Like
/// `cum_flow_quote`/`cum_vol_quote` of `AssetPrice`: a MONOTONIC counter reset only
/// with the process — anti-coalescing (delta of the accumulator) lives in
/// `oracle::vol`, not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiqCum {
    pub asset: String,
    /// Accumulated signed net notional (USDT): SELL (long liquidated) subtracts,
    /// BUY (short liquidated) adds.
    pub cum_net_notional: Decimal,
    /// Accumulated gross notional (USDT), both directions.
    pub cum_total_notional: Decimal,
    pub updated_at: DateTime<Utc>,
    /// Local monotonic clock to measure staleness. Not serialized.
    #[serde(skip)]
    pub instant: Option<Instant>,
}

impl Default for LiqCum {
    fn default() -> Self {
        Self {
            asset: String::new(),
            cum_net_notional: Decimal::ZERO,
            cum_total_notional: Decimal::ZERO,
            updated_at: DateTime::<Utc>::UNIX_EPOCH,
            instant: None,
        }
    }
}

impl Default for AssetPrice {
    fn default() -> Self {
        Self {
            asset: String::new(),
            price: Decimal::ZERO,
            updated_at: DateTime::<Utc>::UNIX_EPOCH,
            instant: None,
            cum_flow_quote: Decimal::ZERO,
            cum_vol_quote: Decimal::ZERO,
            exchange_ts_ms: None,
        }
    }
}

/// Per-asset volatility snapshot published by `oracle::vol`.
///
/// An `oracle::vol` task samples the price once per asset and computes EWMA σ;
/// strategies consume this watch instead of keeping their own ring buffer.
/// Guarantees consistent σ across strategies.
#[derive(Debug, Clone)]
pub struct VolSnapshot {
    pub asset: String,
    /// SPEC-SIGMA-FV §F2: σ̂ from realized variance (`realized_vol_per_sec`),
    /// short window (`SIGMA_WINDOW_SHORT` = 60s) — formerly `sigma_5s`: the name
    /// promised a 5s window that the per-tick EWMA estimator never really
    /// covered.
    pub sigma_short: Option<Decimal>,
    /// SPEC-SIGMA-FV §F2: same, long window (`SIGMA_WINDOW_LONG` = 300s) —
    /// formerly `sigma_60s`, with the same name defect.
    pub sigma_long: Option<Decimal>,
    /// Latest oracle price.
    pub last_price: Decimal,
    pub updated_at: Instant,
    /// PIPE-1: CVD (signed net flow delta, raw USDT) over the short window
    /// (`CVD_WINDOW_SHORT_SECS`). `None` if the asset history does not yet cover
    /// the window (startup / recent reconnection) — absent signal never blocks.
    pub cvd_5s: Option<Decimal>,
    /// PIPE-1: CVD over the long window (`CVD_WINDOW_LONG_SECS`).
    pub cvd_60s: Option<Decimal>,
    /// PIPE-1: gross volume (both directions) over the long window, USDT.
    /// Serves as the denominator to normalize `cvd_5s`/`cvd_60s` to a ratio ∈ [−1,1].
    pub vol_quote_60s: Option<Decimal>,
    /// PIPE-4: signed net notional (USDT) liquidated on futures over the
    /// 60s window — liquidation-cascade pressure. `None` if the feed is
    /// disabled (`oracle.futures_liq_enabled=false`) or the history does not yet
    /// cover the window (startup/reconnection) — absent signal never blocks.
    pub liq_net_60s: Option<Decimal>,
    /// PIPE-4: gross notional (both directions) liquidated over the same window.
    pub liq_total_60s: Option<Decimal>,
    /// Asset turbulence index (`oracle::turbulence`): univariate
    /// Mahalanobis distance of the short-window log-return against the
    /// distribution of same-horizon returns within the long window —
    /// regime signal (idea from FinRL's turbulence index). `f64` because it is
    /// a dimensionless statistic (not money). `None` during warmup (fewer than
    /// `SIGMA_MIN_PAIRS` pairs at the horizon), with a dead tick series (σ²=0) or if the
    /// anchor cell of the short-window return is missing — absent signal never blocks.
    pub turbulence: Option<f64>,
}

impl Default for VolSnapshot {
    fn default() -> Self {
        Self {
            asset: String::new(),
            sigma_short: None,
            sigma_long: None,
            last_price: Decimal::ZERO,
            updated_at: Instant::now(),
            cvd_5s: None,
            cvd_60s: None,
            vol_quote_60s: None,
            liq_net_60s: None,
            liq_total_60s: None,
            turbulence: None,
        }
    }
}

/// A3: maximum oracle freshness to take signal/bail decisions.
/// The vol oracle republishes every ≤5s keeping the `updated_at` of the last real
/// trade; 30s allowed trading on FV/z/cushion with dead prices during Binance
/// gaps. 5s = one republish cycle.
pub const VOL_SIGNAL_STALE_SECS: u64 = 5;

impl VolSnapshot {
    /// True if the oracle has not received real data for more than `secs` seconds.
    /// Strategies must call this before using last_price or sigma_*.
    pub fn is_stale(&self, secs: u64) -> bool {
        self.updated_at.elapsed().as_secs() > secs
    }

    /// A3: staleness with the standard signal threshold (`VOL_SIGNAL_STALE_SECS`).
    pub fn is_stale_for_signal(&self) -> bool {
        self.is_stale(VOL_SIGNAL_STALE_SECS)
    }
}

// ── Order ────────────────────────────────────────────────────────────────────

/// Side of an order emitted by a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    /// Buy `size` shares of the token at the best available price (taker).
    Buy,
    /// Sell `size` shares of the position. `None` = the complete position size.
    Sell { size: Option<Decimal> },
    /// Cancel the active limit order for this (market_cid, token_id).
    /// The executor resolves the order_id from its own record of live orders.
    Cancel,
}

/// Order intent emitted by a strategy.
///
/// Every bot order goes through here → to the `Executor` → to the `OrderRouter`.
/// No strategy calls `clob_sdk` directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntent {
    pub market_cid: String,
    pub token_id: String,
    pub side: OrderSide,
    /// Entry price (ask to buy at, or bid to sell at).
    pub price: Decimal,
    /// Size in shares (Buy/Sell; ignored for Cancel).
    pub size: Decimal,
    /// Unix timestamp of the market close.
    pub clear_at: i64,
    /// Sum of signals that motivated the order (for logging/diagnostics).
    pub signal: Decimal,
    /// True when the market uses the NegRisk CTF Exchange.
    #[serde(default)]
    pub neg_risk: bool,
    /// Maximum limit price (pre-trade out-of-price guard).
    /// `Some(p)` → Limit order capped at p (FOK/FAK).
    /// `None`    → Market order without a cap.
    #[serde(default)]
    pub limit_price: Option<Decimal>,
    /// Second leg of a hedge (executed only if the first fills).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_leg: Option<Box<OrderIntent>>,
    /// Strategy that originated the order.
    #[serde(default)]
    pub strategy: StrategyId,
    /// If `true`, the order is placed as GTD (passive/resting).
    /// If `false` (default), FOK is used (aggressive taker).
    #[serde(default)]
    pub resting: bool,
    /// REAL market close, when it differs from `clear_at`.
    ///
    /// `clear_at` serves two purposes that in almost all strategies
    /// coincide: (1) GTD expiry on the exchange (consumed by
    /// `clob_router.rs`/`sim_router.rs`/`executor.rs::active_gtd`) and (2) the real
    /// market close (copied to `Fill.clear_at` → `positions.clear_at` in
    /// SQLite, used by `src/settlement/orphan.rs::list_expired_unresolved` to know
    /// when a position is ready to redeem). PCS-2 operates AFTER the
    /// real close and needs a synthetic future `clear_at` so the GTD
    /// does not expire instantly — that synthetic value must NOT leak into the fill.
    ///
    /// `None` → `clear_at` IS ALREADY the real close (common scenario, all
    /// strategies except PCS-2). `Some(t)` → the executor uses `t` instead of
    /// `clear_at` when building the `Fill`.
    #[serde(default)]
    pub real_clear_at: Option<i64>,
}

/// Payload of the strategy→executor channel: the intent + the instant it was
/// enqueued. The `Instant` is wall-clock (`std::time::Instant`) on purpose: it measures
/// the real operational latency of the pipeline (`hrf_intent_queue_wait_seconds`),
/// not replay time.
pub type QueuedIntent = (OrderIntent, std::time::Instant);

impl OrderIntent {
    /// Immediate taker FOK/FAK, held to settlement (classic aggressive
    /// entry). Common case: NC, DriftBox, OpenMomentum, ARB, NE, directional.
    /// `clear_at`/`neg_risk` are taken from the `book` — the common scenario always has
    /// a real book at hand when emitting.
    #[allow(clippy::too_many_arguments)]
    pub fn taker(
        market_cid: impl Into<String>,
        token_id: impl Into<String>,
        side: OrderSide,
        price: Decimal,
        size: Decimal,
        signal: Decimal,
        strategy: StrategyId,
        book: &BookSnapshot,
    ) -> Self {
        Self {
            market_cid: market_cid.into(),
            token_id: token_id.into(),
            side,
            price,
            size,
            clear_at: book.clear_at,
            signal,
            neg_risk: book.neg_risk,
            limit_price: Some(price),
            second_leg: None,
            strategy,
            resting: false,
            real_clear_at: None,
        }
    }

    /// Resting GTD with its OWN expiry, decoupled from the real close of the
    /// market. Case: PCS-2, which operates after the market closes and
    /// needs a synthetic future `clear_at` so the GTD does not expire
    /// instantly on the exchange. Requires `gtd_expires_at` and `real_clear_at` as
    /// mandatory, separate parameters — so it is impossible to build this
    /// scenario without explicitly deciding both timestamps, which is exactly the bug
    /// it fixes (the executor copied `clear_at`, the synthetic one, into the fill).
    /// Does not take `&BookSnapshot`: PCS-2's clock-driven entry does not always
    /// have a market book at hand at the instant of the close.
    #[allow(clippy::too_many_arguments)]
    pub fn resting_custom_expiry(
        market_cid: impl Into<String>,
        token_id: impl Into<String>,
        side: OrderSide,
        price: Decimal,
        size: Decimal,
        signal: Decimal,
        strategy: StrategyId,
        neg_risk: bool,
        gtd_expires_at: i64,
        real_clear_at: i64,
    ) -> Self {
        Self {
            market_cid: market_cid.into(),
            token_id: token_id.into(),
            side,
            price,
            size,
            clear_at: gtd_expires_at,
            signal,
            neg_risk,
            limit_price: Some(price),
            second_leg: None,
            strategy,
            resting: true,
            real_clear_at: Some(real_clear_at),
        }
    }
}

// ── Fill ─────────────────────────────────────────────────────────────────────

/// Fill (execution) resulting from an order.
///
/// The fill-sink consumes this kind to update positions, PnL and notify Telegram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub market_cid: String,
    pub token_id: String,
    pub side: FillSide,
    pub price: Decimal,
    pub size: Decimal,
    pub order_id: Option<String>,
    pub ts: DateTime<Utc>,
    pub mode: Mode,
    #[serde(default)]
    pub neg_risk: bool,
    pub strategy: StrategyId,
    /// Unix timestamp of the market close (copied from the intent by the executor).
    #[serde(default)]
    pub clear_at: i64,
    /// "YES" / "NO" — derived from the registry at the point of creation.
    /// Avoids an async DB lookup in fill_sink when the market was already evicted.
    #[serde(default)]
    pub outcome_label: Option<String>,
    /// CAL-2: id shared by the two legs of an ARB pair (gate on
    /// complete pairs). Only stamped by the sim; in live it stays None.
    #[serde(default)]
    pub hedge_id: Option<i64>,
    /// Execution IDENTITY (exchange or simulator trade id).
    /// It is the fill deduplication key and is distinct from
    /// `order_id`: a GTD/FAK order can produce several partial executions.
    /// `None` = legacy origin without its own identity (deduplicated by `order_id`).
    #[serde(default)]
    pub exec_id: Option<String>,
    /// LOCAL order identity (reserve id persisted
    /// before submit). Stamped by the origin when it knows it (the simulator,
    /// from `OrderRequest::local_order_id`); if missing, the fill_sink resolves it
    /// through the `order_id → local_order_id` mapping. Never deduced from the
    /// `market:token` pair and never falls back to ARB.
    #[serde(default)]
    pub local_order_id: Option<String>,
}

/// LIVE fill whose market was in neither the asset registry nor
/// `clear_at_cache` — without reliable `neg_risk`/`clear_at`/`outcome_label` a normal
/// `Fill` cannot be built and accounted for safely, but
/// dropping it in a plain `error!` leaves potentially real money
/// with no persisted trace. `user_ws::handle_trade`
/// sends it through a dedicated channel to `StateStore::record_dead_letter_fill`
/// instead of building the `Fill` — manual reconciliation against the CLOB via
/// `order_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterFill {
    pub order_id: String,
    pub market_cid: String,
    pub token_id: String,
    pub side: FillSide,
    pub price: Decimal,
    pub size: Decimal,
    pub status: String,
    pub role: String,
    pub reason: String,
    pub ts: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FillSide {
    Buy,
    Sell,
    /// Rejected FOK — internal signal to clean up state; never persisted in the DB.
    Rejected,
    /// LIVE-9: cancellation CONFIRMED by the router — internal signal, never
    /// persisted in the DB (same path as `Rejected`).
    ///
    /// Without this, a maker that cancels and reposts has to treat its order as dead at the
    /// instant it EMITS the cancel, and that is false: the cancel travels async and the
    /// order may fill in the meantime. PM considered its `live` released, posted
    /// again, and accumulated 12 GTDs (36 shares) with a cap of 3 — each one's fill
    /// took longer to come back than the next requote took to go out.
    ///
    /// With this confirmation, `live` only drops when the order is demonstrably
    /// dead: either its fill arrived or this ack arrived. Never by assumption.
    Cancelled,
    /// The intent was dropped by a gate BEFORE existing as an
    /// order — an executor risk gate (blacklist, unresolved, kill,
    /// cooldown, rate limit, caps, pre-POST journal) or the degraded-data
    /// guard of the runner. Internal signal, never persisted in the DB.
    ///
    /// Unlike `Rejected` (terminal event of a REAL order the
    /// venue rejected), this fill carries the blocked INTENT's `size`/`price`:
    /// the strategy marked its state when emitting (`yes_live += size`,
    /// `posted_bid = Some(price)`…) and must undo ONLY that contribution.
    /// Never set `live = 0`: there may be ANOTHER real order alive on the same
    /// leg — a reset to zero would fabricate an early repost and violate
    /// the accounting cap (the LIVE-9 pattern: 12 GTDs / 36 shares with cap 3).
    GateRejected,
}

impl std::fmt::Display for FillSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FillSide::Buy => write!(f, "BUY"),
            FillSide::Sell => write!(f, "SELL"),
            FillSide::Rejected => write!(f, "REJECTED"),
            FillSide::Cancelled => write!(f, "CANCELLED"),
            FillSide::GateRejected => write!(f, "GATE_REJECTED"),
        }
    }
}

// ── Positions ────────────────────────────────────────────────────────────────

/// Open position (paper or live, depending on the StateStore mode).
#[derive(Debug, Clone)]
pub struct Position {
    pub token_id: String,
    pub market_cid: String,
    pub slug: String,
    pub strategy: String,
    pub size: Decimal,
    pub avg_price: Decimal,
    pub cost_basis: Decimal,
    pub opened_ts_ms: i64,
    pub clear_at: i64,
    pub neg_risk: bool,
    pub bail_sold: bool,
    pub outcome_won: Option<bool>,
    pub winner_label: Option<String>,
    pub outcome_label: Option<String>,
    pub redeemed: bool,
    pub redeem_tx: Option<String>,
}

// ── CLOB credentials ─────────────────────────────────────────────────────────

/// CLOB credentials derived from the private key.
#[derive(Clone)]
pub struct DerivedCreds {
    pub eoa_address: String,
    pub funder_address: Option<String>,
    pub api_key: String,
    pub api_secret: String,
    pub api_passphrase: String,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_id_roundtrip() {
        use std::str::FromStr;
        let s = StrategyId::NearCertain;
        assert_eq!(s.to_string(), "near_certain");
        assert_eq!(
            StrategyId::from_str("near_certain"),
            Ok(StrategyId::NearCertain)
        );
    }

    /// An unknown name (operator typo) must fail the strict parse
    /// instead of silently falling back to Arb — before the fix, this
    /// made the 400/"unknown strategy" branches of
    /// `api/handlers.rs` and `telegram.rs` unreachable.
    /// SPEC-FOLLOWER: same roundtrip as the other variants.
    #[test]
    fn strategy_id_roundtrip_follower() {
        use std::str::FromStr;
        assert_eq!(StrategyId::Follower.to_string(), "follower");
        assert_eq!(StrategyId::from_str("follower"), Ok(StrategyId::Follower));
    }

    #[test]
    fn from_str_rejects_unknown_name() {
        use std::str::FromStr;
        assert_eq!(StrategyId::from_str("post_close_swep"), Err(()));
        assert_eq!(StrategyId::from_str("does_not_exist"), Err(()));
        assert_eq!(StrategyId::from_str(""), Err(()));
    }

    /// `from_db_str` (history deserialization) remains
    /// tolerant — it never fails, falls back to Arb for any unknown value,
    /// including history markers with no associated strategy.
    #[test]
    fn from_db_str_never_fails_falls_back_to_arb() {
        assert_eq!(
            StrategyId::from_db_str("near_certain"),
            StrategyId::NearCertain
        );
        assert_eq!(StrategyId::from_db_str("post_close_swep"), StrategyId::Arb);
        assert_eq!(StrategyId::from_db_str("recovered"), StrategyId::Arb);
        assert_eq!(StrategyId::from_db_str("unknown"), StrategyId::Arb);
        assert_eq!(StrategyId::from_db_str(""), StrategyId::Arb);
    }

    #[test]
    fn mode_display() {
        assert_eq!(Mode::Paper.to_string(), "paper");
        assert_eq!(Mode::Live.to_string(), "live");
    }

    /// The `OrderIntent` constructors
    /// must produce the default values documented on each one.
    #[test]
    fn order_intent_constructors_default_values() {
        use rust_decimal_macros::dec;
        let book = BookSnapshot {
            market_condition_id: "cid-1".into(),
            clear_at: 1_000,
            neg_risk: true,
            ..Default::default()
        };

        let taker = OrderIntent::taker(
            "cid-1",
            "tok-1",
            OrderSide::Buy,
            dec!(0.5),
            dec!(10),
            dec!(0.5),
            StrategyId::NearCertain,
            &book,
        );
        assert_eq!(taker.clear_at, 1_000, "taker: clear_at = book.clear_at");
        assert_eq!(
            taker.real_clear_at, None,
            "taker: does not distinguish real_clear_at (clear_at is already the real one)"
        );
        assert!(taker.neg_risk);
        assert_eq!(taker.limit_price, Some(dec!(0.5)));
        assert!(!taker.resting);

        let custom = OrderIntent::resting_custom_expiry(
            "cid-1",
            "tok-1",
            OrderSide::Buy,
            dec!(0.99),
            dec!(10),
            dec!(0.99),
            StrategyId::PostCloseSweep,
            false,
            /* neg_risk */ 5_000,
            /* gtd_expires_at */ 1_000, /* real_clear_at */
        );
        assert_eq!(
            custom.clear_at, 5_000,
            "resting_custom_expiry: clear_at = gtd_expires_at (synthetic)"
        );
        assert_eq!(
            custom.real_clear_at,
            Some(1_000),
            "resting_custom_expiry: real_clear_at separate and explicit"
        );
        assert!(custom.resting);
        assert_eq!(custom.limit_price, Some(dec!(0.99)));
    }

    /// FEE-2: `parabolic_fee_rate` already comes as a fraction 0.0-1.0
    /// (not in bps); `None` falls back to the given fallback.
    #[test]
    fn parabolic_fee_rate_or_uses_real_rate_and_falls_back() {
        use rust_decimal_macros::dec;
        let mut book = BookSnapshot {
            parabolic_fee_rate: Some(dec!(0.04)),
            ..Default::default()
        };
        assert_eq!(book.parabolic_fee_rate_or(dec!(0.07)), dec!(0.04));
        book.parabolic_fee_rate = None;
        assert_eq!(book.parabolic_fee_rate_or(dec!(0.07)), dec!(0.07));
    }

    /// TICK-1: the middle zone (including edges 0.96/0.04) uses tick 0.01;
    /// above 0.96 or below 0.04 the tick refines to 0.001.
    #[test]
    fn tick_from_price_zones() {
        use rust_decimal_macros::dec;
        assert_eq!(tick_from_price(dec!(0.50)), dec!(0.01));
        assert_eq!(tick_from_price(dec!(0.98)), dec!(0.001));
        assert_eq!(tick_from_price(dec!(0.02)), dec!(0.001));
        assert_eq!(tick_from_price(dec!(0.96)), dec!(0.01));
        assert_eq!(tick_from_price(dec!(0.04)), dec!(0.01));
    }

    /// TICK-1: truncates to the tick scale without rounding or clamping.
    #[test]
    fn truncate_to_tick_scale() {
        use rust_decimal_macros::dec;
        assert_eq!(truncate_to_tick(dec!(0.9995), dec!(0.001)), dec!(0.999));
        assert_eq!(truncate_to_tick(dec!(0.567), dec!(0.01)), dec!(0.56));
        // No clamp: a value outside [0,1] is truncated all the same, so the
        // caller can detect it and skip.
        assert_eq!(truncate_to_tick(dec!(1.23), dec!(0.01)), dec!(1.23));
    }

    /// TICK-1: the extreme-zone fallback preserves 0.999 instead of mangling it to 0.99
    /// (which happened with the truncation hardcoded to 2 decimals).
    #[test]
    fn fallback_tick_preserves_extreme_zone() {
        use rust_decimal_macros::dec;
        assert_eq!(
            truncate_to_tick(dec!(0.999), tick_from_price(dec!(0.999))),
            dec!(0.999)
        );
    }
}
