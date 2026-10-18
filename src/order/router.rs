//! Trait `OrderRouter` and request/result types.
//!
//! The `Executor` delegates the actual order submission to an implementation of this trait.
//! Switching from paper to live = switching the router implementation; nothing else.
//!
//! `BoxFut` is used instead of `async fn` in the trait so that it is `dyn`-compatible
//! (Rust 1.92 without `async_trait`). Implementations do `Box::pin(async { … })`.

use std::future::Future;
use std::pin::Pin;

use rust_decimal::Decimal;

use crate::types::{Fill, Mode, StrategyId};

/// Alias to avoid repeating the long signature on every trait method.
type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

// ── Request types ────────────────────────────────────────

/// Order type.
/// `Serialize`: the research recorder persists the serialized `OrderRequest`
/// in the payload of the `OrderSubmit` event.
#[derive(Debug, Clone, serde::Serialize)]
pub enum OrderKind {
    /// Market order: buy by USDC amount (taker).
    Market { usdc_amount: Decimal },
    /// Limit order: buy up to `size` shares if the price reaches `price`.
    Limit {
        price: Decimal,
        size: Decimal,
        tif: TimeInForce,
    },
    /// Sell order (sells a position).
    Sell {
        /// `None` used to document "sell the whole position" but NO router
        /// implemented it that way — both used `size.unwrap_or(req.signal)`,
        /// and `signal` is a probability/price (0.0–1.0), not a position size.
        /// `sim_router` now explicitly rejects `None` instead of selling an invented
        /// size — see `SimulatedRouter::submit`. `clob_router` rejects it too.
        /// No strategy emits `size: None` today — if one ever needed it, resolving
        /// the real position size BEFORE building the `OrderIntent` (the `Executor`
        /// has the `StateStore` at hand) is safer than doing it in the router, which
        /// has no context about which position is "the intent's".
        size: Option<Decimal>,
    },
    /// Resting limit sell (maker ask). `size` in shares.
    ///
    /// Separate variant from `Limit` (which is a buy by implicit contract throughout
    /// the tree: `sim_router::tick_book` fills it when `best_ask <= limit_price`,
    /// `clob_router` sends it with `Side::Buy`). Adding a `side` field to `Limit`
    /// would force auditing the 6 places that `match` on `OrderKind` — a new
    /// variant makes the compiler enumerate exactly what is missing.
    LimitSell {
        price: Decimal,
        size: Decimal,
        tif: TimeInForce,
    },
    /// Cancels an active limit order by its CLOB ID.
    Cancel { order_id: String },
}

/// Time-in-Force.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TimeInForce {
    /// Fill-or-Kill: fills completely or not at all.
    Fok,
    /// Fill-and-Kill: fills what it can and cancels the rest.
    Fak,
    /// Good-Till-Day: stays on the book until the end of the day.
    Gtd,
}

/// Order request that the `Executor` passes to the router.
/// `Serialize`: payload of the recorder's `OrderSubmit` event.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OrderRequest {
    pub market_cid: String,
    pub token_id: String,
    pub kind: OrderKind,
    pub neg_risk: bool,
    pub strategy: StrategyId,
    pub clear_at: i64,
    /// Original signal for traceability/logging.
    pub signal: Decimal,
    /// If `true`, the GTD is sent with `postOnly=true`: guarantees maker without crossing the spread.
    /// Only applies to `TimeInForce::Gtd`. Ignored for FOK/FAK/Market.
    pub post_only: bool,
    /// REAL close of the bracket when it differs from `clear_at` (which
    /// may be a synthetic GTD expiration buffer, e.g. via
    /// `OrderIntent::resting_custom_expiry`). Propagated to `UserTrade` so
    /// `user_ws::handle_trade` can use it as a last resort if the market is
    /// in neither the asset registry nor `clear_at_cache` — see
    /// `OrderIntent.real_clear_at`.
    pub real_clear_at: Option<i64>,
    /// Local identity of the order (reserve id persisted pre-submit).
    /// Not sent to the exchange (there is no remote idempotency key);
    /// the simulator copies it into each execution so the fill arrives
    /// with an unambiguous owner.
    pub local_order_id: Option<String>,
}

// ── Result ────────────────────────────────────────

/// Result of a request to the router.
#[derive(Debug)]
pub struct RouterResult {
    /// Resulting fill, if the order executed (fully or partially).
    pub fill: Option<Fill>,
    /// Order ID assigned by the CLOB (relevant for GTD limit orders).
    pub order_id: Option<String>,
    /// The router rejected or killed the order (FOK without fill, insufficient cash, etc.).
    /// DEFINITIVE rejection: the exchange refused the order (validation, closed market,
    /// etc.) — the `Executor` releases the corresponding claim.
    pub rejected: bool,
    /// POST with a lost response: the exchange may have accepted the order or
    /// rejected it, but the `RouterResult` cannot tell
    /// (timeout, 5xx, network drop). The `Executor` must NOT release the claim:
    /// the order may still be live on the exchange and, if released, an identical
    /// double buy goes unaccounted.
    ///
    /// `unknown=true` is mutually exclusive with `rejected=true` by contract
    /// (the [`RouterResult::unknown`] constructor enforces it). The field is
    /// public for inspection; production call sites must use the
    /// [`RouterResult::rejected`], [`RouterResult::no_fill`] or
    /// [`RouterResult::unknown`] constructors.
    pub unknown: bool,
    /// Description of the rejection or error, for logging.
    pub reason: Option<String>,
}

impl RouterResult {
    pub fn rejected(reason: impl Into<String>) -> Self {
        Self {
            fill: None,
            order_id: None,
            rejected: true,
            unknown: false,
            reason: Some(reason.into()),
        }
    }
    pub fn no_fill(order_id: Option<String>) -> Self {
        // GTD placed on the book: no immediate fill, but not a rejection.
        Self {
            fill: None,
            order_id,
            rejected: false,
            unknown: false,
            reason: None,
        }
    }
    /// POST with a lost response: the exchange may have accepted or rejected
    /// the order. `rejected=false` by contract — it is ambiguous, not rejected.
    /// `unknown=true` and `reason` describe the cause (timeout, 5xx, etc.).
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            fill: None,
            order_id: None,
            rejected: false,
            unknown: true,
            reason: Some(reason.into()),
        }
    }
}

// ── Status query (Unknown reconciliation) ────────────────────────────────────────

/// Normalized status of an already-sent order, as reported by the
/// destination. Used by the `unknown_attempts` reconciler to decide the
/// outcome of a POST whose response was lost (`RouterResult::unknown`).
#[derive(Debug, Clone, PartialEq)]
pub enum RouterOrderStatus {
    /// Still live on the book (resting). `remaining` = unexecuted shares.
    Live { remaining: Decimal },
    /// No longer on the book. `matched` = total executed shares
    /// (0 if it was cancelled/expired/rejected without executing anything).
    Gone { matched: Decimal },
    /// The destination does not know the order (HTTP 404 or equivalent): the POST
    /// never got accepted — definitive outcome, the claim is released.
    NotFound,
}

/// Side of a live resting order at the destination.
///
/// The venue cannot always resolve an order's side — the SDK maps
/// any unrecognizable wire value to its `Side::Unknown`. Here it is kept
/// as INDETERMINATE `Unknown`: `Buy` is never inferred by default, because
/// a claim computed over an invented side can release exposure that
/// is still committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenOrderSide {
    Buy,
    Sell,
    /// Side not recognizable on the destination's wire. Every consumer that
    /// needs the side (exposure claim, matching against pendings) must
    /// treat it as indeterminate: keep the previous state or discard
    /// the snapshot — never assume BUY or SELL.
    Unknown,
}

impl From<&polymarket_client_sdk_v2::clob::types::Side> for OpenOrderSide {
    /// Reference: does not require `Copy` or `Clone` on the SDK's kind.
    fn from(s: &polymarket_client_sdk_v2::clob::types::Side) -> Self {
        use polymarket_client_sdk_v2::clob::types::Side;
        match s {
            Side::Buy => OpenOrderSide::Buy,
            Side::Sell => OpenOrderSide::Sell,
            // `Side::Unknown` from the wire (and any future unrecognized SDK
            // variant) stays indeterminate — it is not invented.
            _ => OpenOrderSide::Unknown,
        }
    }
}

/// Live resting order at the destination, with the attributes needed to
/// match it with an intent whose `order_id` was lost in an unanswered
/// POST. Matching requires compatible `market_cid` + `token_id` + `price`
/// + `side` and uniqueness — `token+price` alone used to match a
/// pending BUY against someone else's resting SELL.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterOpenOrder {
    pub order_id: String,
    /// Market cid, in the destination's own representation
    /// (live: `0x...` hex of the condition_id; sim: the `market_cid` of the
    /// original request — same domain as `unknown_attempts.cid`).
    pub market_cid: String,
    pub token_id: String,
    /// Side reported by the destination. `Unknown` = indeterminate.
    pub side: OpenOrderSide,
    pub price: Decimal,
    pub remaining: Decimal,
    pub clear_at: i64,
}

/// Single contractual rule for the exposure claim of a live
/// resting order. Centralizes the formula that used to be repeated as
/// `remaining * price` for ALL sides (periodic refresh,
/// Unknown reconciler, startup) — that formula only describes BUY.
///
/// Rule:
///
/// - `Buy`         → `remaining × price` (cash committed if filled).
///   `price` outside `(0, 1]` is invalid data: a `0` here would fabricate
///   a spurious claim reduction → `None`.
/// - `Sell` non-TC → `0` (sells own inventory already accounted in
///   `positions`; nothing more to reserve).
/// - `Sell` TC     → `remaining` (naked short: contractual downside of
///   $1/share, same interpretation as `handle_sell`).
/// - `Unknown`     → `None` — indeterminate side: without evidence there is no
///   claim rewrite.
///
/// `remaining < 0` → `None` (invalid data). `None` ALWAYS means
/// "indeterminate": the caller must KEEP the previous claim
/// (fail-closed), never reduce it with bad data.
pub fn resting_claim_usdc(
    side: OpenOrderSide,
    remaining: Decimal,
    price: Decimal,
    strategy: StrategyId,
) -> Option<Decimal> {
    if remaining < Decimal::ZERO {
        return None;
    }
    match side {
        OpenOrderSide::Buy => {
            if price <= Decimal::ZERO || price > Decimal::ONE {
                return None;
            }
            Some(remaining * price)
        }
        OpenOrderSide::Sell => {
            if strategy == StrategyId::ThetaCapture {
                Some(remaining)
            } else {
                Some(Decimal::ZERO)
            }
        }
        OpenOrderSide::Unknown => None,
    }
}

/// Unknown taker reconciliation: one of OUR OWN trades executed at the destination.
///
/// A taker intent (FOK/market) never rests — its outcome can only be proven by the
/// venue's trade tape. `order_id` is the identifier
/// of OUR order in that trade (`taker_order_id` when the bot is the taker):
/// exactly the identity that was lost with the POST response.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterTrade {
    /// Execution id (trade uuid on the CLOB; exec_id in the sim).
    pub trade_id: String,
    /// Remote order-id of our leg in the trade, if the destination exposes it.
    pub order_id: Option<String>,
    pub token_id: String,
    /// `true` = the bot BOUGHT in this trade.
    pub is_buy: bool,
    pub price: Decimal,
    pub size: Decimal,
    /// Execution instant in unix milliseconds.
    pub match_ts_ms: i64,
}

// ── Trait ────────────────────────────────────────

/// Abstraction over an order's destination.
///
/// - `SimulatedRouter`: simulates execution in paper with an honest queue model.
/// - `ClobRouter`:      sends to Polymarket's real CLOB (EIP-712 signing + retry).
///
/// Implementations do `Box::pin(async move { … })` in each method.
pub trait OrderRouter: Send + Sync {
    /// Sends an order to the destination (simulated or real) and returns the result.
    fn submit(&self, req: OrderRequest) -> BoxFut<'_, RouterResult>;

    /// Sends a YES+NO pair as a hedge.
    ///
    /// There is NO real cross-leg atomicity — neither here nor in
    /// the overrides. This default implementation is sequential (if
    /// leg 1 is rejected, leg 2 is not even sent), but `ClobRouter` overrides it with
    /// `post_orders` (a single batch POST, each order matched independently
    /// against its own book — NOT all-or-nothing) and `SimulatedRouter` models
    /// exactly that: each leg can be rejected or filled independent of the other.
    fn submit_pair(
        &self,
        req1: OrderRequest,
        req2: OrderRequest,
    ) -> BoxFut<'_, (RouterResult, RouterResult)> {
        Box::pin(async move {
            let r1 = self.submit(req1).await;
            if r1.rejected {
                let reason = r1.reason.clone().unwrap_or_default();
                return (r1, RouterResult::rejected(reason));
            }
            let r2 = self.submit(req2).await;
            (r1, r2)
        })
    }

    /// Cancels an active limit order by its CLOB ID.
    fn cancel(&self, market_cid: &str, order_id: &str) -> BoxFut<'_, anyhow::Result<()>>;

    /// Order_ids of GTDs that are still REALLY live (neither filled nor
    /// expired) at the destination, if it supports this. The executor's periodic resync
    /// (`purge_and_resync`, every 30s) uses it to remove from the local
    /// mapping (`active_gtd`) entries of GTDs that were already filled but never
    /// requoted or explicitly cancelled — without this, their exposure
    /// claim is counted twice against the position resync
    /// (once as a "live" GTD, once as a position already open in the DB).
    ///
    /// Default `None`: destination does not support it — the caller must treat it
    /// as "do not purge this way" (conservative, no change to current
    /// behavior). Implemented today only by `SimulatedRouter`; `ClobRouter`
    /// inherits this default unchanged (querying all live orders on the
    /// real CLOB periodically would be an extra HTTP call every 30s).
    fn live_gtd_order_ids(&self) -> BoxFut<'_, Option<std::collections::HashSet<String>>> {
        Box::pin(async { None })
    }

    /// Unknown reconciliation: current status of an
    /// order at the destination, queried by its `order_id`. Returns:
    /// - `Some(status)` — definitive answer from the destination.
    /// - `None` — the query did NOT resolve (network error, unmappable
    ///   status, router that does not support it). The caller keeps the
    ///   attempt pending: `None` never releases the claim.
    fn order_status(&self, order_id: &str) -> BoxFut<'_, Option<RouterOrderStatus>> {
        let _ = order_id;
        Box::pin(async { None })
    }

    /// Unknown reconciliation: live resting orders at the destination with their
    /// attributes, to match intents that never got an `order_id`
    /// (POST with a lost response). `None` = not supported → those
    /// attempts stay pending (conservative).
    fn open_gtd_orders(&self) -> BoxFut<'_, Option<Vec<RouterOpenOrder>>> {
        Box::pin(async { None })
    }

    /// Unknown taker reconciliation: own trades after
    /// `after_ts_secs` (unix seconds). A taker intent (FOK/market) cannot
    /// rest — if the POST arrived, the trade is on the tape and its
    /// `taker_order_id` returns the lost `order_id`; if there is no trade after
    /// the grace period, the order never existed and the claim can be released.
    /// `None` = not supported or query failed → attempts stay pending.
    fn trades_since(&self, after_ts_secs: i64) -> BoxFut<'_, Option<Vec<RouterTrade>>> {
        let _ = after_ts_secs;
        Box::pin(async { None })
    }

    /// Active mode of the router.
    fn mode(&self) -> Mode;
}
