//! ARB-15M5M strategy.
//!
//! Pair: the 15 m up/down `[T, T+15]` and the 5 m one `[T+10, T+15]` of the same
//! asset resolve on the SAME final Chainlink 60 s TWAP. With strikes
//! `s15 = TWAP(T)` and `s5 = TWAP(T+10 min)` (EXACT value from `chainlink_twap`):
//!   - `s15 <= s5` → Up 15m + Down 5m
//!   - `s15 >  s5` → Up 5m  + Down 15m
//! The basket pays ≥ 1 per share for any final value (2 if it lands between the strikes).
//!
//! Buys only while the marginal cost per share (L2 asks of both legs
//! + each market's parabolic fee) is `< 1 − min_margin`.
//!
//! Execution: two buy FOKs on TWO different markets, sent together by the
//! executor via `second_leg` (the routers' `submit_pair` works per
//! token and does not require the same `market_cid`). KNOWN RISK: the per-bracket
//! exposure ledger (`BracketLedger`) reserves under leg 1's `market_cid`
//! (the 15m) for both legs.
//! Both markets close at the same `t1`, so the claim is cleared at once.
//!
//! No proven edge: `enabled = false`.

use std::collections::HashMap;
use std::sync::Arc;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

use crate::{
    fair_value::{fee_taker_parabolic_with_rate, TAKER_FEE_RATE_PARABOLIC},
    oracle::chainlink_twap::TwapStore,
    state::{Arb15m5mStreak, StateStore},
    time::Instant,
    types::{
        BookSnapshot, Fill, FillSide, Level, OrderIntent, OrderSide, StrategyId, TokenBook,
        VolSnapshot,
    },
};

use super::{Strategy, StrategyConfig};

/// A cached snapshot older than this is not valid for the opposite leg
/// (the WS may have dropped). The book that triggers the evaluation is already filtered by the runner.
const MAX_PARTNER_BOOK_AGE_SECS: u64 = 5;
/// After this long without an outcome for some leg, the basket is closed with what
/// has been observed (routers answer in ms; this covers a fill that never arrives).
const BASKET_TIMEOUT_SECS: u64 = 15;
/// How long books/pairs of already closed markets are kept.
const CLOSED_RETENTION_SECS: i64 = 120;
const DURATION_15M_S: i64 = 900;
const DURATION_5M_S: i64 = 300;

// ── Config ────────────────────────────────────────────────────────────────────

/// What to do with the leg that fills when the other FOK fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrphanPolicy {
    /// Hold to resolution (the preregistered behavior).
    Hold,
    /// One sell attempt at the bid for the filled leg.
    Sell,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Arb15m5mConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
    /// Minimum margin per share after fees: buys while cost < 1 − min_margin.
    #[serde(default = "defaults::min_margin")]
    pub min_margin: Decimal,
    /// Below this many shares, no trading.
    #[serde(default = "defaults::min_size_shares")]
    pub min_size_shares: Decimal,
    /// Cap on shares per basket (0 = only liquidity and `max_order_usdc` limit it).
    #[serde(default = "defaults::max_size_shares")]
    pub max_size_shares: Decimal,
    /// Seconds before the `t1` close at which trading stops.
    #[serde(default = "defaults::close_s")]
    pub close_s: i64,
    /// Minimum wait between two submissions for the same pair.
    #[serde(default = "defaults::cooldown_ms")]
    pub cooldown_ms: u64,
    #[serde(default = "defaults::orphan_policy")]
    pub orphan_policy: OrphanPolicy,
}

mod defaults {
    use super::*;
    pub fn min_margin() -> Decimal {
        dec!(0.01)
    }
    pub fn min_size_shares() -> Decimal {
        dec!(5)
    }
    pub fn max_size_shares() -> Decimal {
        dec!(100)
    }
    pub fn close_s() -> i64 {
        5
    }
    pub fn cooldown_ms() -> u64 {
        2_000
    }
    pub fn orphan_policy() -> OrphanPolicy {
        OrphanPolicy::Hold
    }
}

impl Default for Arb15m5mConfig {
    fn default() -> Self {
        Self {
            limits: Default::default(),
            min_margin: defaults::min_margin(),
            min_size_shares: defaults::min_size_shares(),
            max_size_shares: defaults::max_size_shares(),
            close_s: defaults::close_s(),
            cooldown_ms: defaults::cooldown_ms(),
            orphan_policy: defaults::orphan_policy(),
        }
    }
}

impl StrategyConfig for Arb15m5mConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

/// External dependencies of the strategy, grouped so the wiring
/// (`BuildCtx`, supervisor) passes a single thing.
#[derive(Clone)]
pub struct Arb15m5mDeps {
    pub twap: Arc<TwapStore>,
    /// Destination for per-streak telemetry; `None` = not persisted (tests).
    pub state: Option<Arc<StateStore>>,
    /// In paper with `paper_fok_fill_rate_per_strategy.arb_15m5m = 1.0` the
    /// filled is decided by the book. Only recorded in telemetry.
    pub filled_per_book: bool,
}

impl Arb15m5mDeps {
    /// Production deps: the flag comes from the bot's paper config.
    pub fn from_config(twap: Arc<TwapStore>, state: Arc<StateStore>, cfg: &crate::config::Config) -> Self {
        let by_book = cfg.mode == crate::types::Mode::Paper
            && cfg
                .risk
                .paper_fok_fill_rate_per_strategy
                .get(&StrategyId::Arb15m5m.to_string())
                .is_some_and(|r| *r >= 1.0);
        Self { twap, state: Some(state), filled_per_book: by_book }
    }

    /// No strikes and no persistence: the strategy never trades. For tests and
    /// to build the supervisor where it does not matter.
    pub fn empty() -> Self {
        Self { twap: TwapStore::new(), state: None, filled_per_book: false }
    }
}

// ── Market and pair (slug parsing) ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Duration {
    M5,
    M15,
}

#[derive(Debug, Clone)]
struct Market {
    asset: String,
    duration: Duration,
    t1: i64,
}

/// `btc-updown-15m-1783368000` → asset, duration, `t0` (the slug's epoch) and
/// `t1 = t0 + duration` (`t0` is only used for validation). Requires `t1` to match the book's `clear_at`:
/// otherwise the slug and the market are not what we think and nothing is traded.
fn parse_market(slug: &str, clear_at: i64) -> Option<Market> {
    let mut parts = slug.split('-');
    let asset = parts.next().filter(|a| !a.is_empty())?.to_lowercase();
    if parts.next()? != "updown" {
        return None;
    }
    let (duration, seconds) = match parts.next()? {
        "5m" => (Duration::M5, DURATION_5M_S),
        "15m" => (Duration::M15, DURATION_15M_S),
        _ => return None,
    };
    let t0: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || t0 + seconds != clear_at {
        return None;
    }
    Some(Market { asset, duration, t1: clear_at })
}

// ── Rule (pure functions) ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Up 15m + Down 5m.
    Up15Down5,
    /// Up 5m + Down 15m.
    Up5Down15,
}

impl Direction {
    fn label(self) -> &'static str {
        match self {
            Direction::Up15Down5 => "up15_down5",
            Direction::Up5Down15 => "up5_down15",
        }
    }
}

/// With `s15 <= s5` the 15m Up wins if the final is ≥ s5 and the 5m Down if it is
/// < s15: always at least one. A tie falls in the first branch (pays exactly 1).
fn choose_direction(strike_15: Decimal, strike_5: Decimal) -> Direction {
    if strike_15 <= strike_5 {
        Direction::Up15Down5
    } else {
        Direction::Up5Down15
    }
}

/// Side to buy on each market given the direction: (15m leg, 5m leg).
/// `yes` is the Up (discovery normalizes it that way) and `no` the Down.
fn legs<'a>(
    direction: Direction,
    b15: &'a BookSnapshot,
    b5: &'a BookSnapshot,
) -> (&'a TokenBook, &'a TokenBook) {
    match direction {
        Direction::Up15Down5 => (&b15.yes, &b5.no),
        Direction::Up5Down15 => (&b15.no, &b5.yes),
    }
}

fn fee_rate(book: &BookSnapshot) -> Decimal {
    book.parabolic_fee_rate_or(TAKER_FEE_RATE_PARABOLIC)
}

#[derive(Debug, PartialEq)]
struct Path {
    q: Decimal,
    /// Σ price × shares per leg (without fee).
    cost_a: Decimal,
    cost_b: Decimal,
    fees: Decimal,
    /// Price of the worst level touched on each leg (the FOK limit).
    limit_a: Decimal,
    limit_b: Decimal,
}

/// Walks the asks of both legs together while the marginal cost per
/// share (`pa + pb + fee(pa) + fee(pb)`) stays `< 1 − min_margin`. Asks are
/// ascending, so the marginal cost only rises: stopping at the first failure is exact.
fn traverse(
    a: &[Level],
    rate_a: Decimal,
    b: &[Level],
    rate_b: Decimal,
    min_margin: Decimal,
    cap: Option<Decimal>,
) -> Path {
    let threshold = Decimal::ONE - min_margin;
    let mut r = Path {
        q: Decimal::ZERO,
        cost_a: Decimal::ZERO,
        cost_b: Decimal::ZERO,
        fees: Decimal::ZERO,
        limit_a: Decimal::ZERO,
        limit_b: Decimal::ZERO,
    };
    let mut it_a = a.iter().filter(|l| l.size > Decimal::ZERO);
    let mut it_b = b.iter().filter(|l| l.size > Decimal::ZERO);
    let (mut level_a, mut level_b) = (it_a.next(), it_b.next());
    let mut rest_a = level_a.map_or(Decimal::ZERO, |l| l.size);
    let mut rest_b = level_b.map_or(Decimal::ZERO, |l| l.size);

    while let (Some(la), Some(lb)) = (level_a, level_b) {
        let fee_a = fee_taker_parabolic_with_rate(rate_a, la.price, Decimal::ONE);
        let fee_b = fee_taker_parabolic_with_rate(rate_b, lb.price, Decimal::ONE);
        if la.price + lb.price + fee_a + fee_b >= threshold {
            break;
        }
        let mut step = rest_a.min(rest_b);
        if let Some(t) = cap {
            step = step.min(t - r.q);
        }
        if step <= Decimal::ZERO {
            break;
        }
        r.q += step;
        r.cost_a += la.price * step;
        r.cost_b += lb.price * step;
        r.fees += (fee_a + fee_b) * step;
        r.limit_a = la.price;
        r.limit_b = lb.price;
        rest_a -= step;
        rest_b -= step;
        if rest_a <= Decimal::ZERO {
            level_a = it_a.next();
            rest_a = level_a.map_or(Decimal::ZERO, |l| l.size);
        }
        if rest_b <= Decimal::ZERO {
            level_b = it_b.next();
            rest_b = level_b.map_or(Decimal::ZERO, |l| l.size);
        }
    }
    r
}

/// Decided basket: what is sent and what is expected to be earned.
#[derive(Debug, Clone)]
struct Plan {
    direction: Direction,
    /// Whole shares (the router truncates to whole before signing).
    q: Decimal,
    token_15: String,
    token_5: String,
    limit_15: Decimal,
    limit_5: Decimal,
    /// Average cost per share of the basket, fees included.
    medium_cost: Decimal,
    expected_margin: Decimal,
    /// Shares offered at <= the limit on each leg (for telemetry).
    depth_15: Decimal,
    depth_5: Decimal,
}

/// Σ shares of the asks with price <= `limit`.
fn depth_up_to(asks: &[Level], limit: Decimal) -> Decimal {
    asks.iter().filter(|l| l.price <= limit).map(|l| l.size).sum()
}

#[derive(Debug)]
enum Evaluation {
    /// The minimum marginal cost is already ≥ 1 − min_margin (or one side of the book is missing).
    NoMargin,
    /// There is margin but it does not reach `min_size_shares` / the $1 minimum per leg.
    Insufficient,
    Plan(Plan),
}

fn evaluate(
    cfg: &Arb15m5mConfig,
    direction: Direction,
    b15: &BookSnapshot,
    b5: &BookSnapshot,
) -> Evaluation {
    let (leg_15, leg_5) = legs(direction, b15, b5);
    let (asks_15, asks_5) = (&leg_15.asks.levels, &leg_5.asks.levels);
    let (rate_15, rate_5) = (fee_rate(b15), fee_rate(b5));

    // Share cap: the lesser of `max_size_shares` and what fits in
    // `max_order_usdc` (basket cost, at the first level's cost), if they apply.
    let first_cost = match (asks_15.first(), asks_5.first()) {
        (Some(a), Some(b)) => a.price + b.price,
        _ => return Evaluation::NoMargin,
    };
    let cap_usdc = (cfg.limits.max_order_usdc > Decimal::ZERO && first_cost > Decimal::ZERO)
        .then(|| cfg.limits.max_order_usdc / first_cost);
    let cap_shares = (cfg.max_size_shares > Decimal::ZERO).then_some(cfg.max_size_shares);
    let cap = match (cap_usdc, cap_shares) {
        (Some(u), Some(s)) => Some(u.min(s)),
        (u, s) => u.or(s),
    }
    .map(|t| t.trunc());

    let mut r = traverse(asks_15, rate_15, asks_5, rate_5, cfg.min_margin, cap);
    if r.q <= Decimal::ZERO {
        return Evaluation::NoMargin;
    }
    // Whole shares: if the last level left a fraction, the walk is repeated
    // with the truncated cap so limit prices and cost match what can be sent.
    if r.q.fract() != Decimal::ZERO {
        r = traverse(asks_15, rate_15, asks_5, rate_5, cfg.min_margin, Some(r.q.trunc()));
    }
    // Polymarket minimum: $1 per leg on marketable orders.
    if r.q < cfg.min_size_shares.max(Decimal::ONE)
        || r.cost_a < Decimal::ONE
        || r.cost_b < Decimal::ONE
    {
        return Evaluation::Insufficient;
    }
    let medium_cost = (r.cost_a + r.cost_b + r.fees) / r.q;
    Evaluation::Plan(Plan {
        direction,
        q: r.q,
        token_15: leg_15.token_id.clone(),
        token_5: leg_5.token_id.clone(),
        limit_15: r.limit_a,
        limit_5: r.limit_b,
        medium_cost,
        expected_margin: Decimal::ONE - medium_cost,
        depth_15: depth_up_to(asks_15, r.limit_a),
        depth_5: depth_up_to(asks_5, r.limit_b),
    })
}

// ── State ─────────────────────────────────────────────────────────────────────

/// One leg of a basket: sent and what is known about its outcome.
struct LiveLeg {
    cid: String,
    token: String,
    q: Decimal,
    filled_qty: Decimal,
    cost: Decimal,
    /// The venue rejected the FOK.
    rejected: bool,
    /// The executor/runner dropped it before sending.
    discarded: bool,
}

impl LiveLeg {
    fn new(cid: &str, token: &str, q: Decimal) -> Self {
        Self {
            cid: cid.to_owned(),
            token: token.to_owned(),
            q,
            filled_qty: Decimal::ZERO,
            cost: Decimal::ZERO,
            rejected: false,
            discarded: false,
        }
    }

    fn closed(&self) -> bool {
        self.rejected || self.discarded || self.filled_qty >= self.q
    }
}

struct LiveBasket {
    sent_at: Instant,
    ts_ms: i64,
    plan: Plan,
    strikes: (Decimal, Decimal),
    slugs: (String, String),
    hedge_id: Option<i64>,
    leg_15: LiveLeg,
    leg_5: LiveLeg,
}

#[derive(Default)]
struct Pair {
    cid_15: Option<String>,
    cid_5: Option<String>,
    /// Current streak: a basket was already sent and the margin has not disappeared.
    in_streak: bool,
    last_send: Option<Instant>,
    basket: Option<LiveBasket>,
}

type PairKey = (String, i64);

pub struct Arb15m5m {
    cfg: Arb15m5mConfig,
    twap: Arc<TwapStore>,
    state: Option<Arc<StateStore>>,
    filled_per_book: bool,
    books: HashMap<String, BookSnapshot>,
    markets: HashMap<String, Market>,
    pairs: HashMap<PairKey, Pair>,
    /// Intents (orphan sells) generated in `on_fill`, which cannot return them.
    pending: Vec<OrderIntent>,
}

impl Arb15m5m {
    pub fn new(cfg: Arb15m5mConfig, deps: Arb15m5mDeps) -> Self {
        Self {
            cfg,
            twap: deps.twap,
            state: deps.state,
            filled_per_book: deps.filled_per_book,
            books: HashMap::new(),
            markets: HashMap::new(),
            pairs: HashMap::new(),
            pending: Vec::new(),
        }
    }

    /// Caches the book and links its market to its pair. Returns the pair's
    /// key, or `None` if the slug is not a coherent 5m/15m up/down.
    fn register(&mut self, book: &BookSnapshot) -> Option<PairKey> {
        let cid = &book.market_condition_id;
        if !self.markets.contains_key(cid) {
            let m = parse_market(&book.slug, book.clear_at)?;
            self.markets.insert(cid.clone(), m);
        }
        self.books.insert(cid.clone(), book.clone());
        let m = &self.markets[cid];
        let key = (m.asset.clone(), m.t1);
        let pair = self.pairs.entry(key.clone()).or_default();
        match m.duration {
            Duration::M15 => pair.cid_15 = Some(cid.clone()),
            Duration::M5 => pair.cid_5 = Some(cid.clone()),
        }
        Some(key)
    }

    /// Cached book for `cid` if recent enough. Associated function (not a method)
    /// so as not to borrow all of `self` while `pairs` is mutated.
    fn fresh_book<'a>(books: &'a HashMap<String, BookSnapshot>, cid: &str) -> Option<&'a BookSnapshot> {
        let b = books.get(cid)?;
        let max = std::time::Duration::from_secs(MAX_PARTNER_BOOK_AGE_SECS);
        // Without `recv_instant` there is no age to measure (only happens in tests: the
        // runner already rejects as entry a book without it).
        let fresh = b.recv_instant.map_or(true, |r| r.elapsed() <= max);
        fresh.then_some(b)
    }

    fn evaluate_pair(&mut self, key: &PairKey) -> Vec<OrderIntent> {
        let Some(pair) = self.pairs.get(key) else {
            return vec![];
        };
        let (Some(cid_15), Some(cid_5)) = (pair.cid_15.clone(), pair.cid_5.clone()) else {
            return vec![];
        };
        // Never two live baskets on one pair.
        if pair.basket.is_some() {
            return vec![];
        }
        let (Some(b15), Some(b5)) = (
            Self::fresh_book(&self.books, &cid_15),
            Self::fresh_book(&self.books, &cid_5),
        ) else {
            return vec![];
        };
        if b15.secs_to_clear() <= self.cfg.close_s {
            return vec![];
        }
        let (asset, t1) = key;
        let strike = |t0: i64| self.twap.exact_value(asset, t0 * 1000);
        let (Some(s15), Some(s5)) = (strike(t1 - DURATION_15M_S), strike(t1 - DURATION_5M_S))
        else {
            metrics::counter!("hrf_arb15m5m_skip_without_strike_total").increment(1);
            return vec![];
        };

        let direction = choose_direction(s15, s5);
        let plan = match evaluate(&self.cfg, direction, b15, b5) {
            Evaluation::NoMargin => {
                // The margin disappeared: the streak ends.
                self.pairs.get_mut(key).expect("existing pair").in_streak = false;
                return vec![];
            }
            Evaluation::Insufficient => return vec![],
            Evaluation::Plan(p) => p,
        };

        let pair = self.pairs.get(key).expect("existing pair");
        let cooldown = std::time::Duration::from_millis(self.cfg.cooldown_ms);
        let in_cooldown = pair.last_send.is_some_and(|t| t.elapsed() < cooldown);
        if pair.in_streak || in_cooldown {
            return vec![];
        }

        let intents = build_intents(&plan, b15, b5);
        let basket = LiveBasket {
            sent_at: Instant::now(),
            ts_ms: crate::time::now().timestamp_millis(),
            strikes: (s15, s5),
            slugs: (b15.slug.clone(), b5.slug.clone()),
            hedge_id: None,
            leg_15: LiveLeg::new(&cid_15, &plan.token_15, plan.q),
            leg_5: LiveLeg::new(&cid_5, &plan.token_5, plan.q),
            plan,
        };
        tracing::info!(asset = %asset, t1, direction = basket.plan.direction.label(),
            q = %basket.plan.q, margin = %basket.plan.expected_margin,
            "ARB-15M5M: basket sent");
        metrics::counter!("hrf_arb15m5m_baskets_sent_total").increment(1);
        let pair = self.pairs.get_mut(key).expect("existing pair");
        pair.in_streak = true;
        pair.last_send = Some(Instant::now());
        pair.basket = Some(basket);
        intents
    }

    /// Closes the pair's live basket with what has been observed so far: records
    /// telemetry and applies the orphan policy.
    fn close_basket(&mut self, key: &PairKey) {
        let Some(pair) = self.pairs.get_mut(key) else {
            return;
        };
        let Some(basket) = pair.basket.take() else {
            return;
        };
        let (f15, f5) = (basket.leg_15.filled_qty, basket.leg_5.filled_qty);
        let result = match (f15 > Decimal::ZERO, f5 > Decimal::ZERO) {
            (true, true) => "both",
            (false, false) if basket.leg_15.discarded || basket.leg_5.discarded => "rejected",
            (false, false) => "none",
            _ => "orphan",
        };
        if result == "rejected" {
            // Never left: the streak is re-armed (the cooldown still prevents a loop).
            pair.in_streak = false;
        }
        tracing::info!(asset = %key.0, t1 = key.1, result, filled_15 = %f15,
            filled_5 = %f5, "ARB-15M5M: basket closed");
        metrics::counter!("hrf_arb15m5m_streaks_total", "result" => result).increment(1);

        if result == "orphan" && self.cfg.orphan_policy == OrphanPolicy::Sell {
            let filled_leg = if f15 > Decimal::ZERO { &basket.leg_15 } else { &basket.leg_5 };
            if let Some(sell) = self.orphan_sell(filled_leg) {
                self.pending.push(sell);
            }
        }
        self.persist(to_record(&basket, key, result, self.cfg.orphan_policy, self.filled_per_book));
    }

    /// A single attempt to sell the filled leg at the cached book's best bid.
    fn orphan_sell(&self, leg: &LiveLeg) -> Option<OrderIntent> {
        let book = self.books.get(&leg.cid)?;
        let token_book = if book.yes.token_id == leg.token { &book.yes } else { &book.no };
        let bid = token_book.best_bid()?;
        tracing::warn!(cid = %leg.cid, size = %leg.filled_qty, bid = %bid,
            "ARB-15M5M: orphan — selling the filled leg at the bid");
        Some(OrderIntent {
            market_cid: leg.cid.clone(),
            token_id: leg.token.clone(),
            side: OrderSide::Sell { size: Some(leg.filled_qty) },
            price: bid,
            size: leg.filled_qty,
            clear_at: book.clear_at,
            signal: Decimal::ZERO,
            neg_risk: book.neg_risk,
            limit_price: None,
            second_leg: None,
            strategy: StrategyId::Arb15m5m,
            resting: false,
            real_clear_at: None,
        })
    }

    fn persist(&self, streak: Arb15m5mStreak) {
        let Some(state) = self.state.clone() else {
            return;
        };
        let Ok(rt) = tokio::runtime::Handle::try_current() else {
            return;
        };
        rt.spawn(async move {
            if let Err(e) = state.record_arb15m5m_streak(&streak).await {
                tracing::error!(error = %e, "ARB-15M5M: could not persist the streak");
            }
        });
    }

    /// Frees memory of markets closed a while ago.
    fn prune(&mut self) {
        let cut = crate::time::now().timestamp() - CLOSED_RETENTION_SECS;
        self.pairs
            .retain(|(_, t1), p| *t1 >= cut || p.basket.is_some());
        self.markets.retain(|_, m| m.t1 >= cut);
        let live = &self.markets;
        self.books.retain(|cid, _| live.contains_key(cid));
    }
}

fn build_intents(plan: &Plan, b15: &BookSnapshot, b5: &BookSnapshot) -> Vec<OrderIntent> {
    let leg = |b: &BookSnapshot, token: &str, limit: Decimal, second_order: Option<OrderIntent>| {
        OrderIntent {
            market_cid: b.market_condition_id.clone(),
            token_id: token.to_owned(),
            side: OrderSide::Buy,
            price: limit,
            size: plan.q,
            clear_at: b.clear_at,
            signal: plan.expected_margin,
            neg_risk: b.neg_risk,
            limit_price: Some(limit),
            second_leg: second_order.map(Box::new),
            strategy: StrategyId::Arb15m5m,
            resting: false,
            real_clear_at: None,
        }
    };
    let leg_5 = leg(b5, &plan.token_5, plan.limit_5, None);
    vec![leg(b15, &plan.token_15, plan.limit_15, Some(leg_5))]
}

fn to_record(
    c: &LiveBasket,
    key: &PairKey,
    result: &str,
    policy: OrphanPolicy,
    filled_per_book: bool,
) -> Arb15m5mStreak {
    Arb15m5mStreak {
        ts_ms: c.ts_ms,
        asset: key.0.clone(),
        t1: key.1,
        slug_15m: c.slugs.0.clone(),
        slug_5m: c.slugs.1.clone(),
        cid_15m: c.leg_15.cid.clone(),
        cid_5m: c.leg_5.cid.clone(),
        token_15m: c.leg_15.token.clone(),
        token_5m: c.leg_5.token.clone(),
        strike_15m: c.strikes.0,
        strike_5m: c.strikes.1,
        direction: c.plan.direction.label().to_owned(),
        q: c.plan.q,
        limit_15m: c.plan.limit_15,
        limit_5m: c.plan.limit_5,
        medium_cost: c.plan.medium_cost,
        expected_margin: c.plan.expected_margin,
        result: result.to_owned(),
        filled_15m: c.leg_15.filled_qty,
        filled_5m: c.leg_5.filled_qty,
        real_cost: c.leg_15.cost + c.leg_5.cost,
        orphan_policy: match policy {
            OrphanPolicy::Hold => "hold",
            OrphanPolicy::Sell => "sell",
        }
        .to_owned(),
        hedge_id: c.hedge_id,
        filled_per_book,
        depth_15m: c.plan.depth_15,
        depth_5m: c.plan.depth_5,
    }
}

impl Strategy for Arb15m5m {
    fn id(&self) -> StrategyId {
        StrategyId::Arb15m5m
    }

    /// Does not use the Binance oracle: its strikes come from the Chainlink TWAP.
    fn requires_oracle(&self) -> bool {
        false
    }

    fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
        let mut intents = std::mem::take(&mut self.pending);
        if let Some(key) = self.register(book) {
            intents.extend(self.evaluate_pair(&key));
        }
        intents
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::Arb15m5m {
            return;
        }
        let Some(key) = self.pairs.iter_mut().find_map(|(key, p)| {
            let basket = p.basket.as_mut()?;
            let leg = [&mut basket.leg_15, &mut basket.leg_5]
                .into_iter()
                .find(|l| l.cid == fill.market_cid && l.token == fill.token_id)?;
            match fill.side {
                FillSide::Buy => {
                    leg.filled_qty += fill.size;
                    leg.cost += fill.price * fill.size;
                    basket.hedge_id = basket.hedge_id.or(fill.hedge_id);
                }
                FillSide::Rejected => leg.rejected = true,
                FillSide::GateRejected => leg.discarded = true,
                FillSide::Sell | FillSide::Cancelled => return None,
            }
            (basket.leg_15.closed() && basket.leg_5.closed()).then(|| key.clone())
        }) else {
            return;
        };
        self.close_basket(&key);
    }

    fn on_timer(&mut self) -> Vec<OrderIntent> {
        let expired_keys: Vec<PairKey> = self
            .pairs
            .iter()
            .filter(|(_, p)| {
                p.basket.as_ref().is_some_and(|c| {
                    c.sent_at.elapsed() >= std::time::Duration::from_secs(BASKET_TIMEOUT_SECS)
                })
            })
            .map(|(k, _)| k.clone())
            .collect();
        for key in expired_keys {
            tracing::warn!(asset = %key.0, t1 = key.1,
                "ARB-15M5M: basket without outcome after the timeout — closed with what was observed");
            self.close_basket(&key);
        }
        self.prune();
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Mode};

    const ASSET: &str = "btc";

    fn levels(v: &[(Decimal, Decimal)]) -> Vec<Level> {
        v.iter().map(|&(price, size)| Level { price, size }).collect()
    }

    fn token(id: &str, asks: &[(Decimal, Decimal)], bids: &[(Decimal, Decimal)]) -> TokenBook {
        TokenBook {
            token_id: id.into(),
            asks: HalfBook { levels: levels(asks) },
            bids: HalfBook { levels: levels(bids) },
            ..Default::default()
        }
    }

    type Asks = Vec<(Decimal, Decimal)>;

    /// Book of a market `dur` ("15m"/"5m") that closes at `t1`.
    fn book(dur: &str, t1: i64, up: Asks, down: Asks) -> BookSnapshot {
        let seg = if dur == "15m" { DURATION_15M_S } else { DURATION_5M_S };
        BookSnapshot {
            market_condition_id: format!("cid-{dur}"),
            slug: format!("{ASSET}-updown-{dur}-{}", t1 - seg),
            clear_at: t1,
            yes: token(&format!("u{dur}"), &up, &[(dec!(0.30), dec!(100))]),
            no: token(&format!("d{dur}"), &down, &[(dec!(0.20), dec!(100))]),
            ..Default::default()
        }
    }

    fn good_15m(t1: i64) -> BookSnapshot {
        book("15m", t1, vec![(dec!(0.40), dec!(50))], vec![(dec!(0.40), dec!(50))])
    }
    fn good_5m(t1: i64) -> BookSnapshot {
        book("5m", t1, vec![(dec!(0.45), dec!(50))], vec![(dec!(0.45), dec!(50))])
    }
    fn expensive_5m(t1: i64) -> BookSnapshot {
        book("5m", t1, vec![(dec!(0.60), dec!(50))], vec![(dec!(0.60), dec!(50))])
    }

    fn cfg_test() -> Arb15m5mConfig {
        Arb15m5mConfig { cooldown_ms: 0, ..Default::default() }
    }

    /// Strategy with `s15`/`s5` strikes already published for the `t1` close.
    fn strategy(cfg: Arb15m5mConfig, t1: i64, s15: Decimal, s5: Decimal) -> Arb15m5m {
        let twap = TwapStore::new();
        twap.insert(ASSET, (t1 - DURATION_15M_S) * 1000, s15);
        twap.insert(ASSET, (t1 - DURATION_5M_S) * 1000, s5);
        Arb15m5m::new(cfg, Arb15m5mDeps { twap, state: None, filled_per_book: false })
    }

    fn t1_in(seconds: i64) -> i64 {
        crate::time::now().timestamp() + seconds
    }

    fn vol() -> VolSnapshot {
        VolSnapshot::default()
    }

    fn fill(cid: &str, token: &str, side: FillSide, size: Decimal) -> Fill {
        Fill {
            market_cid: cid.into(),
            token_id: token.into(),
            side,
            price: dec!(0.40),
            size,
            order_id: None,
            ts: crate::time::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::Arb15m5m,
            clear_at: 0,
            outcome_label: None,
            hedge_id: Some(7),
            exec_id: None,
            local_order_id: None,
        }
    }

    /// Feeds both books and returns the pair's intents.
    fn send_basket(s: &mut Arb15m5m, t1: i64) -> Vec<OrderIntent> {
        assert!(s.on_book(&good_15m(t1), &vol()).is_empty(), "no pair yet");
        s.on_book(&good_5m(t1), &vol())
    }

    // ── pure rule ──

    #[test]
    fn direction_according_to_strikes_including_tie() {
        assert_eq!(choose_direction(dec!(100), dec!(101)), Direction::Up15Down5);
        assert_eq!(choose_direction(dec!(100), dec!(100)), Direction::Up15Down5);
        assert_eq!(choose_direction(dec!(101), dec!(100)), Direction::Up5Down15);
    }

    #[test]
    fn slug_is_parsed_and_must_match_clear_at() {
        let m = parse_market("eth-updown-15m-1000", 1900).unwrap();
        assert_eq!((m.asset.as_str(), m.duration, m.t1), ("eth", Duration::M15, 1900));
        assert!(parse_market("eth-updown-15m-1000", 1300).is_none(), "inconsistent t1");
        assert!(parse_market("btc-updown-1h-1000", 4600).is_none());
        assert!(parse_market("btc-updown-5m-om", 300).is_none());
    }

    #[test]
    fn greedy_walk_crosses_levels_and_respects_margin() {
        let a = levels(&[(dec!(0.40), dec!(10)), (dec!(0.45), dec!(10))]);
        let b = levels(&[(dec!(0.45), dec!(5)), (dec!(0.50), dec!(20))]);
        let rate = dec!(0.07);
        // margin 0.01 (threshold 0.99): all three segments of the walk are included.
        let r = traverse(&a, rate, &b, rate, dec!(0.01), None);
        assert_eq!(r.q, dec!(20));
        assert_eq!((r.limit_a, r.limit_b), (dec!(0.45), dec!(0.50)));
        // margin 0.02 (threshold 0.98): the third segment (0.45+0.50+fees = 0.9848) is no longer included.
        let r = traverse(&a, rate, &b, rate, dec!(0.02), None);
        assert_eq!(r.q, dec!(10));
        assert_eq!((r.limit_a, r.limit_b), (dec!(0.40), dec!(0.50)));
        assert_eq!(r.cost_a, dec!(4.00));
        assert_eq!(r.cost_b, dec!(4.75));
        let fee_unit = |p: Decimal| fee_taker_parabolic_with_rate(rate, p, Decimal::ONE);
        let expected = fee_unit(dec!(0.40)) * dec!(10)
            + fee_unit(dec!(0.45)) * dec!(5)
            + fee_unit(dec!(0.50)) * dec!(5);
        assert_eq!(r.fees, expected);
        // share cap
        assert_eq!(traverse(&a, rate, &b, rate, dec!(0.01), Some(dec!(7))).q, dec!(7));
    }

    #[test]
    fn plan_uses_whole_shares_and_respects_minimums() {
        let t1 = 10_000;
        let b15 = book("15m", t1, vec![(dec!(0.40), dec!(10.5))], vec![]);
        let b5 = book("5m", t1, vec![], vec![(dec!(0.45), dec!(10.5))]);
        let Evaluation::Plan(p) = evaluate(&cfg_test(), Direction::Up15Down5, &b15, &b5) else {
            panic!("expected a plan");
        };
        assert_eq!(p.q, dec!(10), "10.5 shares are truncated to 10");
        assert_eq!((p.token_15.as_str(), p.token_5.as_str()), ("u15m", "d5m"));
        assert_eq!((p.limit_15, p.limit_5), (dec!(0.40), dec!(0.45)));
        assert_eq!(p.expected_margin, Decimal::ONE - p.medium_cost);
        assert_eq!((p.depth_15, p.depth_5), (dec!(10.5), dec!(10.5)));

        let cfg = Arb15m5mConfig { min_size_shares: dec!(11), ..cfg_test() };
        assert!(matches!(evaluate(&cfg, Direction::Up15Down5, &b15, &b5), Evaluation::Insufficient));
        let cfg = Arb15m5mConfig { max_size_shares: dec!(6), ..cfg_test() };
        let Evaluation::Plan(p) = evaluate(&cfg, Direction::Up15Down5, &b15, &b5) else { panic!() };
        assert_eq!(p.q, dec!(6), "max_size_shares rules");
    }

    #[test]
    fn plan_respects_max_order_usdc() {
        let t1 = 10_000;
        let mut cfg = cfg_test();
        cfg.limits.max_order_usdc = dec!(17); // 17 / (0.40 + 0.45) = 20 shares
        let (b15, b5) = (good_15m(t1), good_5m(t1));
        let Evaluation::Plan(p) = evaluate(&cfg, Direction::Up15Down5, &b15, &b5) else { panic!() };
        assert_eq!(p.q, dec!(20));
    }

    #[test]
    fn without_margin_if_marginal_cost_does_not_drop_below_one() {
        let t1 = 10_000;
        let (b15, b5) = (good_15m(t1), expensive_5m(t1));
        assert!(matches!(
            evaluate(&cfg_test(), Direction::Up15Down5, &b15, &b5),
            Evaluation::NoMargin
        ));
    }

    // ── on_book / streak ──

    #[tokio::test]
    async fn emits_up15_plus_down5_with_lower_strike_15() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(200);
        let mut s = strategy(cfg_test(), t1, dec!(100), dec!(101));
        let intents = send_basket(&mut s, t1);
        assert_eq!(intents.len(), 1);
        let i = &intents[0];
        assert_eq!((i.market_cid.as_str(), i.token_id.as_str()), ("cid-15m", "u15m"));
        assert!(matches!(i.side, OrderSide::Buy));
        assert_eq!(i.strategy, StrategyId::Arb15m5m);
        assert_eq!(i.limit_price, Some(dec!(0.40)));
        let second_order = i.second_leg.as_ref().expect("second leg");
        assert_eq!((second_order.market_cid.as_str(), second_order.token_id.as_str()), ("cid-5m", "d5m"));
        assert_eq!(second_order.limit_price, Some(dec!(0.45)));
        assert_eq!(i.size, second_order.size, "same q on both legs");
    }

    #[tokio::test]
    async fn emits_up5_plus_down15_with_higher_strike_15() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(200);
        let mut s = strategy(cfg_test(), t1, dec!(101), dec!(100));
        let i = send_basket(&mut s, t1).remove(0);
        assert_eq!(i.token_id, "d15m");
        assert_eq!(i.second_leg.unwrap().token_id, "u5m");
    }

    #[tokio::test]
    async fn without_exact_strike_does_not_trade() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(200);
        let mut s = strategy(cfg_test(), t1, dec!(100), dec!(101));
        // A tick 1 s after the 5m's t0 is NOT valid as a strike.
        let twap = TwapStore::new();
        twap.insert(ASSET, (t1 - DURATION_15M_S) * 1000, dec!(100));
        twap.insert(ASSET, (t1 - DURATION_5M_S) * 1000 + 1000, dec!(101));
        s.twap = twap;
        assert!(send_basket(&mut s, t1).is_empty());
    }

    #[tokio::test]
    async fn does_not_trade_within_close_window() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(3); // close_s = 5
        let mut s = strategy(cfg_test(), t1, dec!(100), dec!(101));
        assert!(send_basket(&mut s, t1).is_empty());
    }

    #[tokio::test]
    async fn one_execution_per_streak_and_the_streak_ends_when_the_margin_goes() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(200);
        let mut s = strategy(cfg_test(), t1, dec!(100), dec!(101));
        assert_eq!(send_basket(&mut s, t1).len(), 1);
        // Both legs fill: the basket closes, but the streak continues.
        s.on_fill(&fill("cid-15m", "u15m", FillSide::Buy, dec!(50)));
        s.on_fill(&fill("cid-5m", "d5m", FillSide::Buy, dec!(50)));
        assert!(s.on_book(&good_5m(t1), &vol()).is_empty(), "same streak: no repeat");
        // The margin disappears → the streak ends → a new one can open.
        assert!(s.on_book(&expensive_5m(t1), &vol()).is_empty());
        assert_eq!(s.on_book(&good_5m(t1), &vol()).len(), 1, "new streak");
    }

    #[tokio::test]
    async fn never_two_live_baskets_in_same_pair() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(200);
        let mut s = strategy(cfg_test(), t1, dec!(100), dec!(101));
        assert_eq!(send_basket(&mut s, t1).len(), 1);
        // No fills yet: even if the streak ends and the margin reappears, the basket stays alive.
        assert!(s.on_book(&expensive_5m(t1), &vol()).is_empty());
        assert!(s.on_book(&good_5m(t1), &vol()).is_empty());
    }

    #[tokio::test]
    async fn cooldown_blocks_resend_between_streaks() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(200);
        let cfg = Arb15m5mConfig { cooldown_ms: 60_000, ..Default::default() };
        let mut s = strategy(cfg, t1, dec!(100), dec!(101));
        assert_eq!(send_basket(&mut s, t1).len(), 1);
        s.on_fill(&fill("cid-15m", "u15m", FillSide::Buy, dec!(50)));
        s.on_fill(&fill("cid-5m", "d5m", FillSide::Buy, dec!(50)));
        assert!(s.on_book(&expensive_5m(t1), &vol()).is_empty());
        assert!(s.on_book(&good_5m(t1), &vol()).is_empty(), "in cooldown");
    }

    #[tokio::test]
    async fn a_gate_discard_rearms_the_streak() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(200);
        let mut s = strategy(cfg_test(), t1, dec!(100), dec!(101));
        assert_eq!(send_basket(&mut s, t1).len(), 1);
        s.on_fill(&fill("cid-15m", "u15m", FillSide::GateRejected, dec!(50)));
        s.on_fill(&fill("cid-5m", "d5m", FillSide::GateRejected, dec!(50)));
        assert_eq!(s.on_book(&good_5m(t1), &vol()).len(), 1, "never left: retried");
    }

    // ── orphan ──

    async fn orphan(policy: OrphanPolicy) -> Vec<OrderIntent> {
        let t1 = t1_in(200);
        let cfg = Arb15m5mConfig { orphan_policy: policy, ..cfg_test() };
        let mut s = strategy(cfg, t1, dec!(100), dec!(101));
        assert_eq!(send_basket(&mut s, t1).len(), 1);
        s.on_fill(&fill("cid-15m", "u15m", FillSide::Buy, dec!(50)));
        s.on_fill(&fill("cid-5m", "d5m", FillSide::Rejected, dec!(0)));
        s.on_timer()
    }

    #[tokio::test]
    async fn hold_orphan_does_nothing() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        assert!(orphan(OrphanPolicy::Hold).await.is_empty());
    }

    #[tokio::test]
    async fn sell_orphan_sells_filled_leg_at_bid() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let sells = orphan(OrphanPolicy::Sell).await;
        assert_eq!(sells.len(), 1);
        let v = &sells[0];
        assert_eq!((v.market_cid.as_str(), v.token_id.as_str()), ("cid-15m", "u15m"));
        assert!(matches!(v.side, OrderSide::Sell { size: Some(s) } if s == dec!(50)));
        assert_eq!(v.price, dec!(0.30), "best bid of the cached book");
    }

    #[tokio::test]
    async fn timeout_closes_a_basket_without_outcome() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let t1 = t1_in(200);
        let mut s = strategy(cfg_test(), t1, dec!(100), dec!(101));
        assert_eq!(send_basket(&mut s, t1).len(), 1);
        for p in s.pairs.values_mut() {
            p.basket.as_mut().unwrap().sent_at = Instant::Real(
                std::time::Instant::now() - std::time::Duration::from_secs(BASKET_TIMEOUT_SECS + 1),
            );
        }
        s.on_timer();
        assert!(s.pairs.values().all(|p| p.basket.is_none()));
    }
}
