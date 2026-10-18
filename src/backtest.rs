//! Offline backtest: replays recorded `book_snapshots` (tools::recorders::analytics)
//! + `spot_samples` (tools::recorders::spot_stream) through a REAL strategy against
//! `SimulatedRouter`. Entry point: [`run`]; CLI: `src/bin/hrf_backtest.rs`.
//!
//! The DBs are ALWAYS opened read-only (`file:…?mode=ro`): they are evidence
//! from past runs and a WAL checkpoint would corrupt them (permanently).
//!
//! v1 fidelity and limits — read before interpreting results:
//! - The replay does NOT go through the `Executor`: no RiskGuard, exposure caps,
//!   Unknown journal or reconciliation. The report measures the raw behavior
//!   of the strategy against the simulator (the risk layer is
//!   orthogonal to the economic signal being evaluated).
//! - Intent→OrderRequest mirrors `Executor::build_request`: FOK/FAK/GTD,
//!   maker `LimitSell` for `resting+limit_price`, `post_only` for resting MM,
//!   `Market` = price×size.
//! - σ is recomputed with the SAME production function
//!   (`fair_value::realized_vol_per_sec`, same parameters as `oracle::vol`).
//! - `recv_instant`/`updated_at` are stamped with the `ts_ms` of the recorded
//!   row via `crate::time::Instant::from_replay_ms` — the staleness guards
//!   and strategy TTLs measure the REAL age of the data in
//!   replay time (a stale spot looks stale, not "just arrived").
//! - `on_timer` fires at the same cadence as the runner watchdog
//!   (5 s of replay time) — needed to handle orphan legs.
//! - Settlement: a position still open at `clear_at` is resolved with the last
//!   known asset price (Binance spot) against the bracket's
//!   `reference_price`. Positions without spot/strike stay in the report's
//!   `unresolved` — a result is never invented. This
//!   verdict is an explicit PROXY (`crate::research::labels::PROVENANCE_PROXY_SPOT`
//!   → `BacktestReport::settlement_provenance`), not the official result of the
//!   market: usable for accounting, excluded from certified evaluation.
//! - Fills carry an all-in price (cash/shares, fee included) — same
//!   as production accounting.

use crate::time::Instant;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context};
use rusqlite::{Connection, OpenFlags};
use rust_decimal::Decimal;
use serde::Serialize;
use tokio::sync::{mpsc, watch};

use crate::fair_value::{realized_vol_per_sec, PriceSample};
use crate::market_feed::MarketRegistry;
use crate::order::router::{OrderKind, OrderRequest, OrderRouter, TimeInForce};
use crate::order::sim_router::{SimClock, SimulatedRouter};
use crate::strategy::{Strategy, StrategyConfigs};
use crate::types::{
    BookSnapshot, Fill, FillSide, HalfBook, Level, OrderIntent, OrderSide, StrategyId, TokenBook,
    VolSnapshot,
};

// σ parameters identical to `oracle::vol` (private constants there —
// deliberately duplicated, with an explicit reference).
const SIGMA_GRID_SECS: i64 = 1;
const SIGMA_WINDOW_SHORT: i64 = 60;
const SIGMA_WINDOW_LONG: i64 = 300;
const SIGMA_MIN_PAIRS: usize = 30;
const SIGMA_RETENTION_SECS: i64 = 360;

/// Runner watchdog cadence (`WATCHDOG_SECS` in strategy/mod.rs).
const TIMER_MS: i64 = 5_000;

/// Strategies with FAK tif in the executor (`fak_strategies` in main.rs).
const FAK_STRATEGIES: &[StrategyId] = &[StrategyId::NearCertain, StrategyId::NearExpiry];

// ── Parameters and output ─────────────────────────────────────────────────────

pub struct BacktestParams {
    /// Analytics DB (`book_snapshots`).
    pub db_books: PathBuf,
    /// spot_stream DB (`spot_samples`); `None` = replay without oracle
    /// (strategies with `requires_oracle=false`: ARB/PM; the rest will see
    /// `VolSnapshot::default()` — same behavior as at cold start).
    pub db_spot: Option<PathBuf>,
    pub strategy: StrategyId,
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    /// Sim RNG seed (`with_deps`) — same input ⇒ same output.
    pub seed: u64,
    pub initial_balance: Decimal,
    /// Config directory to load the strategy TOMLs and the sim's
    /// paper parameters (`risk.paper_*`) + `fair_value.confidence_haircut`.
    pub config_dir: PathBuf,
}

impl Default for BacktestParams {
    fn default() -> Self {
        Self {
            db_books: PathBuf::new(),
            db_spot: None,
            strategy: StrategyId::Arb,
            from_ms: None,
            to_ms: None,
            seed: 0xC0FF_EE15,
            initial_balance: Decimal::new(5_000, 0),
            config_dir: PathBuf::from("config"),
        }
    }
}

/// Serialized report — Decimals travel as STRING (project
/// convention: `EquityPoint` serializes the same way, `parseFloat` never sees NaN).
#[derive(Debug, Clone, Serialize)]
pub struct BacktestReport {
    pub strategy: String,
    pub from_ms: i64,
    pub to_ms: i64,
    pub seed: u64,
    // ── replay counts ──
    pub books_replayed: u64,
    pub spot_samples: u64,
    pub timer_ticks: u64,
    pub intents: u64,
    pub orders_submitted: u64,
    pub intents_without_request: u64,
    pub fills: u64,
    pub rejects: u64,
    pub unknowns: u64,
    pub cancels_ok: u64,
    pub cancel_without_order: u64,
    // ── accounting ──
    pub initial_balance: String,
    pub final_cash: String,
    pub spent_buys: String,
    pub proceeds_sells: String,
    pub payouts_settlement: String,
    pub realized_pnl: String,
    pub unrealized_pnl: String,
    pub equity_total: String,
    pub max_exposure_usdc: String,
    /// `cash == initial − spent + proceeds + payouts`: accounting invariant.
    pub accounting_consistent: bool,
    // ── settlement ──
    /// Evidentiary basis of the settlements: always the spot proxy
    /// (`crate::research::labels::PROVENANCE_PROXY_SPOT`) — an explicit marker
    /// that these outcomes are NOT the official market result and are
    /// excluded from certified evaluation.
    pub settlement_provenance: String,
    pub settled_positions: u64,
    pub won_liquidations: u64,
    pub lost_liquidations: u64,
    pub live_gtds_at_end: u64,
    pub open_positions: Vec<PositionReport>,
    pub unresolved_positions: Vec<PositionReport>,
    pub by_bracket: Vec<BracketReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PositionReport {
    pub market_cid: String,
    pub token_id: String,
    pub outcome_label: String,
    pub shares: String,
    pub cost: String,
    pub mark: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BracketReport {
    pub market_cid: String,
    pub slug: String,
    pub fills: u64,
    pub pnl: String,
}

// ── Read-only loading ─────────────────────────────────────────────────────────

/// Opens a SQLite database WITHOUT write access (`mode=ro`): no WAL checkpoint and no
/// journal — a run's DB is untouchable evidence.
/// `pub(crate)`: the replay engine (`crate::replay`) loads the same tape.
pub(crate) fn open_readonly(path: &Path) -> anyhow::Result<Connection> {
    let uri = format!("file:{}?mode=ro", path.display());
    Connection::open_with_flags(
        uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .with_context(|| format!("opening {} read-only", path.display()))
}

fn dec_or(s: Option<String>) -> Option<Decimal> {
    s.and_then(|v| Decimal::from_str(&v).ok())
}

fn levels_or(s: &str, descending: bool) -> HalfBook {
    let mut levels: Vec<Level> = serde_json::from_str(s).unwrap_or_default();
    // Defensive sorting (same criterion as tools::recorders::analytics::best_bid_ask_mid:
    // never trust that the recorded JSON came sorted).
    if descending {
        levels.sort_unstable_by_key(|l| std::cmp::Reverse(l.price));
    } else {
        levels.sort_unstable_by_key(|l| l.price);
    }
    HalfBook { levels }
}

/// `pub(crate)`: `crate::replay` reuses the same tape loader.
pub(crate) struct BookRow {
    pub(crate) ts_ms: i64,
    pub(crate) snap: BookSnapshot,
}

pub(crate) fn load_books(
    conn: &Connection,
    from_ms: i64,
    to_ms: i64,
) -> anyhow::Result<Vec<BookRow>> {
    let mut stmt = conn.prepare(
        "SELECT ts_ms, market_condition_id, slug, clear_at, market_open_ts, neg_risk,
                reference_price, fee_rate_bps, parabolic_fee_rate, tick_size,
                yes_token_id, yes_bids_json, yes_asks_json,
                no_token_id, no_bids_json, no_asks_json
           FROM book_snapshots
          WHERE ts_ms >= ?1 AND ts_ms <= ?2
          ORDER BY ts_ms ASC, id ASC",
    )?;
    let rows = stmt
        .query_map([from_ms, to_ms], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, Option<String>>(6)?,
                r.get::<_, Option<u32>>(7)?,
                r.get::<_, Option<String>>(8)?,
                r.get::<_, Option<String>>(9)?,
                r.get::<_, String>(10)?,
                r.get::<_, String>(11)?,
                r.get::<_, String>(12)?,
                r.get::<_, String>(13)?,
                r.get::<_, String>(14)?,
                r.get::<_, String>(15)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = Vec::with_capacity(rows.len());
    for (
        ts_ms,
        cid,
        slug,
        clear_at,
        open_ts,
        neg_risk,
        ref_price,
        fee_bps,
        par_fee,
        tick,
        yes_tok,
        yes_bids,
        yes_asks,
        no_tok,
        no_bids,
        no_asks,
    ) in rows
    {
        out.push(BookRow {
            ts_ms,
            snap: BookSnapshot {
                market_condition_id: cid,
                slug,
                clear_at,
                market_open_ts: open_ts,
                neg_risk: neg_risk != 0,
                reference_price: dec_or(ref_price),
                fee_rate_bps: fee_bps,
                parabolic_fee_rate: dec_or(par_fee),
                tick_size: dec_or(tick),
                yes: TokenBook {
                    token_id: yes_tok,
                    bids: levels_or(&yes_bids, true),
                    asks: levels_or(&yes_asks, false),
                    ..Default::default()
                },
                no: TokenBook {
                    token_id: no_tok,
                    bids: levels_or(&no_bids, true),
                    asks: levels_or(&no_asks, false),
                    ..Default::default()
                },
                // Stamped with the ts of the ROW (not that of the event in progress):
                // whoever reads `recv_instant` in replay sees the real age of the data.
                recv_instant: Some(Instant::from_replay_ms(ts_ms)),
            },
        });
    }
    Ok(out)
}

/// `pub(crate)`: `crate::replay` reuses the same tape loader.
pub(crate) struct SpotRow {
    pub(crate) asset: String,
    pub(crate) source: String,
    pub(crate) ts_ms: i64,
    pub(crate) price: Decimal,
}

pub(crate) fn load_spots(
    conn: &Connection,
    from_ms: i64,
    to_ms: i64,
) -> anyhow::Result<Vec<SpotRow>> {
    let mut stmt = conn.prepare(
        "SELECT asset, source, price, source_ts_ms
           FROM spot_samples
          WHERE source_ts_ms >= ?1 AND source_ts_ms <= ?2
          ORDER BY source_ts_ms ASC, id ASC",
    )?;
    let rows = stmt
        .query_map([from_ms, to_ms], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows
        .into_iter()
        .filter_map(|(asset, source, price, ts_ms)| {
            Decimal::from_str(&price).ok().map(|p| SpotRow {
                asset,
                source,
                ts_ms,
                price: p,
            })
        })
        .collect())
}

// ── Oracle replay (σ with the production function) ──────────────────────────

/// Replica of the `sec_samples` buffer of `oracle::vol::run_asset`: samples
/// sorted by time with temporal retention; σ is derived on each query with
/// `realized_vol_per_sec` (same function and parameters as production).
/// Retention is the long window + 20% (`SIGMA_RETENTION_SECS`).
/// `pub(crate)`: the replay engine (`crate::replay`) feeds the
/// strategies the same `VolSnapshot` as this diagnostic.
#[derive(Default)]
pub(crate) struct VolReplay {
    samples: VecDeque<PriceSample>,
    pub(crate) last_price: Decimal,
}

impl VolReplay {
    pub(crate) fn push(&mut self, _source: &str, ts_ms: i64, price: Decimal) {
        if price <= Decimal::ZERO {
            return;
        }
        // Out of order (defensive — the query already sorts): dropped so
        // as not to break the monotonicity that `realized_vol_per_sec` debug-asserts.
        if self.samples.back().is_some_and(|s| ts_ms < s.ts_ms) {
            return;
        }
        self.samples.push_back(PriceSample { ts_ms, price });
        self.last_price = price;
        let cutoff = ts_ms - SIGMA_RETENTION_SECS * 1000;
        while self.samples.front().is_some_and(|s| s.ts_ms < cutoff) {
            self.samples.pop_front();
        }
    }

    pub(crate) fn snapshot(&self, asset: &str) -> VolSnapshot {
        let samples: Vec<PriceSample> = self.samples.iter().copied().collect();
        VolSnapshot {
            asset: asset.to_string(),
            sigma_short: realized_vol_per_sec(
                &samples,
                SIGMA_GRID_SECS,
                SIGMA_WINDOW_SHORT,
                SIGMA_MIN_PAIRS,
            ),
            sigma_long: realized_vol_per_sec(
                &samples,
                SIGMA_GRID_SECS,
                SIGMA_WINDOW_LONG,
                SIGMA_MIN_PAIRS,
            ),
            last_price: self.last_price,
            // `updated_at` is the ts of the LAST sample, not the instant of the
            // event in progress — so `elapsed()` in replay measures the real staleness
            // of the spot (a stale spot must look stale, not "just arrived").
            updated_at: Instant::from_replay_ms(self.samples.back().map(|s| s.ts_ms).unwrap_or(0)),
            ..Default::default()
        }
    }
}

// ── Config: TOMLs without env or validate (paper; secrets do not apply) ──────

#[derive(serde::Deserialize, Default)]
pub(crate) struct GeneralProbe {
    #[serde(default)]
    pub(crate) risk: crate::config::RiskConfig,
    #[serde(default)]
    pub(crate) fair_value: crate::config::FairValueConfig,
    /// `market_feed.max_book_age_ms` feeds the stale-book gate of the
    /// runner that the pipeline engine reproduces.
    #[serde(default)]
    pub(crate) market_feed: MarketFeedProbe,
    /// `[research_sim]` — the simulator's research profile
    /// is only applied when the runner asks for it (`--research-profile config`);
    /// the `nominal` default reproduces operational behavior.
    #[serde(default)]
    pub(crate) research_sim: crate::config::ResearchSimConfig,
}

/// Narrow probe of `[market_feed]` — `MarketFeedConfig` does not implement
/// `Default`; we only need `max_book_age_ms` with the same serde default.
#[derive(serde::Deserialize, Default)]
pub(crate) struct MarketFeedProbe {
    #[serde(default = "crate::config::defaults::max_book_age_ms")]
    pub(crate) max_book_age_ms: u64,
}

/// Merges `general.toml` + `config/strategies/*.toml` in the same order as
/// `Config::load`, but WITHOUT `HRF_*` env or `validate()` — the backtest is
/// offline and needs neither secrets nor mode gates.
pub(crate) fn load_configs(config_dir: &Path) -> anyhow::Result<(StrategyConfigs, GeneralProbe)> {
    use figment::providers::{Format, Toml};
    let mut fig = figment::Figment::new();
    for f in ["general.toml", "telegram.toml"] {
        let p = config_dir.join(f);
        if p.is_file() {
            fig = fig.merge(Toml::file(p));
        }
    }
    let strat_dir = config_dir.join("strategies");
    if strat_dir.is_dir() {
        let mut paths: Vec<_> = std::fs::read_dir(&strat_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("toml"))
            .collect();
        paths.sort();
        for path in paths {
            fig = fig.merge(Toml::file(path));
        }
    }
    let cfgs: StrategyConfigs = fig.extract().context("extracting StrategyConfigs")?;
    let probe: GeneralProbe = fig.extract().context("extracting risk/fair_value")?;
    Ok((cfgs, probe))
}

/// Instantiates the strategy with the same constructors as the
/// `register_strategies!` table in `strategy/registry.rs`. RewardFarm/PCS/Follower
/// cannot run in backtest (runtime dependencies or scan-only) —
/// explicit error, never a silently wrong strategy.
pub(crate) fn build_strategy(
    id: StrategyId,
    cfgs: &StrategyConfigs,
    haircut: Decimal,
) -> anyhow::Result<Box<dyn Strategy>> {
    Ok(match id {
        StrategyId::Arb => Box::new(crate::strategy::Arb::new(cfgs.arb.clone())),
        StrategyId::Directional => Box::new(crate::strategy::Directional::new(
            cfgs.directional.clone(),
            haircut,
        )),
        StrategyId::NearCertain => Box::new(crate::strategy::NearCertain::new(
            cfgs.near_certain.clone(),
            haircut,
        )),
        StrategyId::NearExpiry => {
            Box::new(crate::strategy::NearExpiry::new(cfgs.near_expiry.clone()))
        }
        StrategyId::MarketMaking => Box::new(crate::strategy::MarketMaking::new(
            cfgs.market_making.clone(),
        )),
        StrategyId::ParityMaker => {
            Box::new(crate::strategy::ParityMaker::new(cfgs.parity_maker.clone()))
        }
        StrategyId::DriftBox => Box::new(crate::strategy::DriftBox::new(
            cfgs.drift_box.clone(),
            haircut,
        )),
        StrategyId::OpenMomentum => Box::new(crate::strategy::OpenMomentum::new(
            cfgs.open_momentum.clone(),
            haircut,
        )),
        StrategyId::ThetaCapture => Box::new(crate::strategy::ThetaCapture::new(
            cfgs.theta_capture.clone(),
            haircut,
        )),
        StrategyId::GRw1Maker => {
            Box::new(crate::strategy::GRw1Maker::new(cfgs.g_rw1_maker.clone()))
        }
        StrategyId::PostCloseSweep => {
            bail!("post_close_sweep is not supported in backtest v1: it needs taker_poller_handles extracted (see registry.rs)")
        }
        StrategyId::RewardFarm => {
            bail!("reward_farm is scan-only (never emits intents): nothing to measure in backtest")
        }
        StrategyId::Follower => bail!("follower does not implement the Strategy trait"),
        // In backtest the tool emits no predictions, so the
        // strategy cannot decide anything useful. We return `bail!` to
        // force manual construction with a synthetic feed if one wants to
        // evaluate the strategy over history — outside the scope of
        // this run.
        StrategyId::ModelBtc5m => bail!(
            "model_btc5m requires an in-line prediction feed: in backtest, \
             build the strategy manually with `ModelBtc5m::new(cfg, rx, Mode::Paper)` \
             after draining synthetic predictions from a CSV file. It is not \
             in scope for the current paper run."
        ),
        // The offline evaluation of rl_predictor lives in the LAB
        // (Python runner + cohorts); backtest has no approved artifact,
        // learner or real ledger — it is never simulated with synthetic weights.
        // ARB-15M5M: needs Chainlink TWAP strikes (PolyBolt) that the
        // backtest does not reproduce; it is measured by the bot's own paper run.
        StrategyId::Arb15m5m => bail!(
            "arb_15m5m requires live Chainlink TWAP strikes: it is measured in paper, not in backtest"
        ),
        StrategyId::RlPredictor => bail!(
            "rl_predictor is an online paper-only runtime (paper_approved artifact + \
             learner + real ledger): offline evaluation lives in the LAB, \
             not in this backtest"
        ),
    })
}

// ── Simplified ledger ─────────────────────────────────────────────────────────

/// The backtest's own accounting: the Fill carries an all-in price
/// (cash/shares), so BUY deducts `price*size` from cash and SELL adds it.
/// The cost released on partial sells is proportional (avg×size) — same
/// criterion as production `positions.cost_basis`.
#[derive(Default)]
struct BtPos {
    shares: Decimal,
    cost: Decimal,
}

#[derive(Default)]
struct BracketAcc {
    fills: u64,
    pnl: Decimal,
}

struct Ledger {
    initial: Decimal,
    cash: Decimal,
    positions: HashMap<(String, String), BtPos>, // (cid, token_id)
    realized: Decimal,
    spent_buys: Decimal,
    proceeds_sells: Decimal,
    payouts: Decimal,
    max_exposure: Decimal,
    settled: u64,
    won: u64,
    lost: u64,
    per_bracket: HashMap<String, BracketAcc>,
}

impl Ledger {
    fn new(initial: Decimal) -> Self {
        Self {
            initial,
            cash: initial,
            positions: HashMap::new(),
            realized: Decimal::ZERO,
            spent_buys: Decimal::ZERO,
            proceeds_sells: Decimal::ZERO,
            payouts: Decimal::ZERO,
            max_exposure: Decimal::ZERO,
            settled: 0,
            won: 0,
            lost: 0,
            per_bracket: HashMap::new(),
        }
    }

    fn apply(&mut self, fill: &Fill) {
        let key = (fill.market_cid.clone(), fill.token_id.clone());
        let notional = fill.price * fill.size;
        match fill.side {
            FillSide::Buy => {
                self.cash -= notional;
                self.spent_buys += notional;
                let pos = self.positions.entry(key).or_default();
                pos.shares += fill.size;
                pos.cost += notional;
            }
            FillSide::Sell => {
                self.cash += notional;
                self.proceeds_sells += notional;
                let pos = self.positions.entry(key.clone()).or_default();
                // Sale with inventory 0 (oversell): the proceeds are real but
                // there is no cost to release — the delta falls entirely into PnL.
                let (released, remaining_shares) = if pos.shares > Decimal::ZERO {
                    let s = fill.size.min(pos.shares);
                    let avg = pos.cost / pos.shares;
                    (avg * s, pos.shares - s)
                } else {
                    (Decimal::ZERO, Decimal::ZERO)
                };
                let delta = notional - released;
                pos.cost -= released;
                pos.shares = remaining_shares;
                self.realized += delta;
                let acc = self.per_bracket.entry(fill.market_cid.clone()).or_default();
                acc.fills += 1;
                acc.pnl += delta;
            }
            // Rejected/Cancelled/GateRejected: internal signal, never money.
            FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected => {}
        }
        let exposure: Decimal = self.positions.values().map(|p| p.cost).sum();
        if exposure > self.max_exposure {
            self.max_exposure = exposure;
        }
    }

    /// Settlement at `clear_at`: the winning leg pays $1/share, the losing leg 0.
    /// `yes_wins: None` = no evidence to resolve (the position is left untouched).
    fn settle(&mut self, cid: &str, yes_token: &str, yes_wins: Option<bool>) -> u64 {
        let Some(yes_wins) = yes_wins else { return 0 };
        let keys: Vec<(String, String)> = self
            .positions
            .keys()
            .filter(|(c, _)| c == cid)
            .cloned()
            .collect();
        let mut n = 0;
        for key in keys {
            let Some(pos) = self.positions.get(&key) else {
                continue;
            };
            if pos.shares <= Decimal::ZERO {
                continue;
            }
            let is_yes = key.1 == yes_token;
            let wins = is_yes == yes_wins;
            let payout = if wins { pos.shares } else { Decimal::ZERO };
            let delta = payout - pos.cost;
            self.cash += payout;
            self.payouts += payout;
            self.realized += delta;
            let acc = self.per_bracket.entry(cid.to_string()).or_default();
            acc.fills += 1;
            acc.pnl += delta;
            self.settled += 1;
            if wins {
                self.won += 1;
            } else {
                self.lost += 1;
            }
            self.positions.remove(&key);
            n += 1;
        }
        n
    }

    /// Invariant: cash = initial − buys + sells + settlement payouts.
    fn consistent(&self) -> bool {
        self.cash == self.initial - self.spent_buys + self.proceeds_sells + self.payouts
    }
}

// ── Market and event metadata ────────────────────────────────────────────────

struct MarketMeta {
    yes_token: String,
    slug: String,
    clear_at: i64, // unix SECONDS (BookSnapshot::secs_to_clear convention)
    reference_price: Option<Decimal>,
}

enum Event {
    Spot(SpotRow),
    /// Box: BookSnapshot is ~10× larger than SpotRow — without indirection
    /// every spot event would pay for the ~384 B of the large variant.
    Book(Box<BookRow>),
}

impl Event {
    fn ts(&self) -> i64 {
        match self {
            Event::Spot(s) => s.ts_ms,
            Event::Book(b) => b.ts_ms,
        }
    }
}

fn merge_events(books: Vec<BookRow>, spots: Vec<SpotRow>) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::with_capacity(books.len() + spots.len());
    events.extend(spots.into_iter().map(Event::Spot));
    events.extend(books.into_iter().map(|b| Event::Book(Box::new(b))));
    // At equal ts, spot goes first: the σ/FV the strategy sees with the book
    // includes the most recent sample (same causal order as live, where
    // the vol watch already carries the latest trade when the book arrives).
    events.sort_by_key(|e| (e.ts(), matches!(e, Event::Book(_)) as u8));
    events
}

// ── Dispatch intent → router (minimal replica of Executor::build_request) ───

#[derive(Default)]
struct Rt {
    /// (cid, token) → order_id of a live GTD — the executor's `order_map`/`active_gtd`
    /// reduced to what the backtest needs for Cancel.
    active_orders: HashMap<(String, String), String>,
    intents: u64,
    orders: u64,
    without_request: u64,
    fills: u64,
    rejects: u64,
    unknowns: u64,
    cancels_ok: u64,
    cancel_without_order: u64,
    timer_ticks: u64,
}

fn build_request(intent: &OrderIntent, rt: &Rt) -> Option<OrderRequest> {
    let kind = match &intent.side {
        OrderSide::Sell { size } => match (intent.limit_price, intent.resting) {
            // SPEC-PM-UNWIND §6.2 (same rule as the executor):
            // resting+limit_price = maker GTD ask; otherwise = market sell.
            (Some(limit), true) => OrderKind::LimitSell {
                price: limit,
                size: size.unwrap_or(intent.size),
                tif: TimeInForce::Gtd,
            },
            _ => OrderKind::Sell { size: *size },
        },
        OrderSide::Cancel => {
            let oid = rt
                .active_orders
                .get(&(intent.market_cid.clone(), intent.token_id.clone()))?;
            OrderKind::Cancel {
                order_id: oid.clone(),
            }
        }
        OrderSide::Buy => match intent.limit_price {
            Some(limit) => OrderKind::Limit {
                price: limit,
                size: intent.size,
                tif: if intent.resting {
                    TimeInForce::Gtd
                } else if FAK_STRATEGIES.contains(&intent.strategy) {
                    TimeInForce::Fak
                } else {
                    TimeInForce::Fok
                },
            },
            None => OrderKind::Market {
                usdc_amount: intent.price * intent.size,
            },
        },
    };
    Some(OrderRequest {
        market_cid: intent.market_cid.clone(),
        token_id: intent.token_id.clone(),
        kind,
        neg_risk: intent.neg_risk,
        strategy: intent.strategy,
        clear_at: intent.clear_at,
        signal: intent.signal,
        post_only: intent.resting && intent.strategy == StrategyId::MarketMaking,
        real_clear_at: intent.real_clear_at,
        local_order_id: None,
    })
}

/// Local terminal fill (executor semantics on `rejected`): unblocks
/// the strategy latches. It is not money and is not persisted.
fn rejected_fill(intent: &OrderIntent) -> Fill {
    Fill {
        market_cid: intent.market_cid.clone(),
        token_id: intent.token_id.clone(),
        side: FillSide::Rejected,
        price: Decimal::ZERO,
        size: Decimal::ZERO,
        order_id: None,
        exec_id: None,
        local_order_id: None,
        ts: crate::time::now(),
        mode: crate::types::Mode::Paper,
        neg_risk: intent.neg_risk,
        strategy: intent.strategy,
        clear_at: intent.real_clear_at.unwrap_or(intent.clear_at),
        outcome_label: None,
        hedge_id: None,
    }
}

/// Observable result of a single submit — the executor's second-leg gate
/// is `has_fill || (has_order && !resting)`, computed over the
/// `RouterResult` (the synchronous fill; resting orders arrive async via fill_tx).
#[derive(Default)]
struct SubmitOut {
    has_fill: bool,
    has_order: bool,
}

async fn submit_one(
    router: &SimulatedRouter,
    intent: &OrderIntent,
    rt: &mut Rt,
    strategy: &mut dyn Strategy,
    fill_rx: &mut mpsc::UnboundedReceiver<Fill>,
    ledger: &mut Ledger,
) -> SubmitOut {
    let Some(req) = build_request(intent, rt) else {
        rt.without_request += 1;
        return SubmitOut::default();
    };
    let key = (intent.market_cid.clone(), intent.token_id.clone());
    let is_cancel = matches!(intent.side, OrderSide::Cancel);
    rt.orders += 1;
    let res = router.submit(req).await;
    let out = SubmitOut {
        has_fill: res.fill.is_some(),
        has_order: res.order_id.is_some(),
    };
    if res.rejected {
        rt.rejects += 1;
        strategy.on_fill(&rejected_fill(intent));
    } else if res.unknown {
        // Conservative like production: the order may still be live — do NOT
        // notify Rejected nor release the order_id from the mapping.
        rt.unknowns += 1;
    }
    if let Some(oid) = &res.order_id {
        rt.active_orders.insert(key, oid.clone());
    }
    if is_cancel && !res.unknown {
        rt.active_orders
            .remove(&(intent.market_cid.clone(), intent.token_id.clone()));
        rt.cancels_ok += 1;
    }
    drain_fills(fill_rx, strategy, ledger, rt);
    out
}

/// Taker YES+NO pair via `submit_pair` — the sim override evaluates both
/// legs against the same post-latency books with ONE correlated liquidity
/// draw (`pair_fill_correlation`), just like `post_orders` in
/// live. Two independent submits would give P(pair) ≈ p² and bias the
/// ARB economics downward.
async fn dispatch_pair(
    router: &SimulatedRouter,
    intent: &OrderIntent,
    second: &OrderIntent,
    rt: &mut Rt,
    strategy: &mut dyn Strategy,
    fill_rx: &mut mpsc::UnboundedReceiver<Fill>,
    ledger: &mut Ledger,
) {
    let Some(req1) = build_request(intent, rt) else {
        rt.without_request += 1;
        return;
    };
    let Some(req2) = build_request(second, rt) else {
        rt.without_request += 1;
        return;
    };
    rt.orders += 2;
    let (r1, r2) = router.submit_pair(req1, req2).await;
    for (res, leg) in [(&r1, intent), (&r2, second)] {
        if res.rejected {
            rt.rejects += 1;
            strategy.on_fill(&rejected_fill(leg));
        } else if res.unknown {
            rt.unknowns += 1;
        }
        if let Some(oid) = &res.order_id {
            rt.active_orders
                .insert((leg.market_cid.clone(), leg.token_id.clone()), oid.clone());
        }
    }
    drain_fills(fill_rx, strategy, ledger, rt);
}

/// Dispatches an intent and its `second_leg` with the executor's rule:
/// - `second_leg` Buy + `!resting` → `submit_pair` (one POST, correlated draw).
/// - Otherwise (resting, or non-Buy second leg) → sequential, only if the first
///   one filled or was accepted (`has_fill || (has_order && !resting)`): a
///   rejected first leg NEVER triggers the coverage.
async fn dispatch(
    router: &SimulatedRouter,
    intent: OrderIntent,
    rt: &mut Rt,
    strategy: &mut dyn Strategy,
    fill_rx: &mut mpsc::UnboundedReceiver<Fill>,
    ledger: &mut Ledger,
) {
    rt.intents += 1;
    if matches!(intent.side, OrderSide::Cancel)
        && !rt
            .active_orders
            .contains_key(&(intent.market_cid.clone(), intent.token_id.clone()))
    {
        rt.cancel_without_order += 1;
        rt.intents -= 1; // never reaches the router: counted separately, not as an executed intent
        return;
    }
    let second = intent.second_leg.clone();
    let resting = intent.resting;
    if let Some(l) = second.as_deref() {
        if matches!(l.side, OrderSide::Buy) && !resting {
            dispatch_pair(router, &intent, l, rt, strategy, fill_rx, ledger).await;
            return;
        }
    }
    let out = submit_one(router, &intent, rt, strategy, fill_rx, ledger).await;
    if let Some(second) = second {
        if out.has_fill || (out.has_order && !resting) {
            submit_one(router, &second, rt, strategy, fill_rx, ledger).await;
        }
    }
}

fn drain_fills(
    fill_rx: &mut mpsc::UnboundedReceiver<Fill>,
    strategy: &mut dyn Strategy,
    ledger: &mut Ledger,
    rt: &mut Rt,
) {
    while let Ok(fill) = fill_rx.try_recv() {
        rt.fills += 1;
        ledger.apply(&fill);
        strategy.on_fill(&fill);
    }
}

// ── Settlement ────────────────────────────────────────────────────────────────

/// Resolves positions in markets whose `clear_at` has already passed in replay
/// time. Returns the positions it could not resolve (no spot/strike).
fn settle_due(
    now_ms: i64,
    metas: &HashMap<String, MarketMeta>,
    vols: &HashMap<String, VolReplay>,
    ledger: &mut Ledger,
) {
    for (cid, meta) in metas {
        if now_ms <= meta.clear_at * 1000 {
            continue;
        }
        // No open position on this cid → nothing to do (avoids
        // retrying already settled cids on every later event).
        if !ledger.positions.keys().any(|(c, _)| c == cid) {
            continue;
        }
        let asset = crate::oracle::asset_from_slug(&meta.slug);
        let spot = asset
            .and_then(|a| vols.get(&a))
            .map(|v| v.last_price)
            .filter(|p| *p > Decimal::ZERO);
        // The settlement proxy lives in `crate::research::labels`
        // (`provisional_updown`, same semantics: spot > strike → YES) —
        // explicit provenance `PROVENANCE_PROXY_SPOT`, excluded from
        // certified evaluation: it is accounting, not official truth.
        let yes_wins = crate::research::labels::provisional_updown(meta.reference_price, spot)
            .map(|w| w == crate::research::contract::OutcomeSide::Yes);
        if ledger.settle(cid, &meta.yes_token, yes_wins) == 0 && yes_wins.is_none() {
            // No evidence: the position stays open until the final sweep.
        }
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

/// Runs the complete backtest: loads configs from `config_dir`, instantiates the
/// strategy by id and runs the replay. `async` because of `OrderRouter::submit`.
pub async fn run(params: &BacktestParams) -> anyhow::Result<BacktestReport> {
    let (cfgs, probe) = load_configs(&params.config_dir)?;
    let strategy = build_strategy(params.strategy, &cfgs, probe.fair_value.confidence_haircut)?;
    run_replay(params, strategy, &probe.risk).await
}

/// Replay core with the strategy already built — `run` instantiates it by
/// id; tests inject their own strategies to exercise the pipeline without
/// depending on real config thresholds.
async fn run_replay(
    params: &BacktestParams,
    mut strategy: Box<dyn Strategy>,
    risk: &crate::config::RiskConfig,
) -> anyhow::Result<BacktestReport> {
    let from_ms = params.from_ms.unwrap_or(i64::MIN + 1);
    let to_ms = params.to_ms.unwrap_or(i64::MAX - 1);

    let books_conn = open_readonly(&params.db_books)?;
    let books = load_books(&books_conn, from_ms, to_ms)?;
    let spots = match &params.db_spot {
        Some(p) => load_spots(&open_readonly(p)?, from_ms, to_ms)?,
        None => Vec::new(),
    };
    if books.is_empty() {
        bail!(
            "no book_snapshots in the window [{from_ms}, {to_ms}] for {}",
            params.db_books.display()
        );
    }

    // Injected clock: the sim purges GTDs and orders hedge_ids against IT.
    let clock_ms = Arc::new(AtomicI64::new(from_ms));
    let clock: SimClock = {
        let c = clock_ms.clone();
        Arc::new(move || c.load(Ordering::Relaxed))
    };
    let registry = Arc::new(MarketRegistry::new());
    let (fill_tx, mut fill_rx) = mpsc::unbounded_channel();
    let (wallet_tx, _wallet_rx) = watch::channel(None);
    let sim = SimulatedRouter::with_deps(
        risk.paper_slippage_bps,
        risk.paper_fok_fill_rate,
        // Latency 0: the sim does a REAL `sleep` per submit, but in replay the
        // books only advance inside the runner loop — the pause models
        // nothing (no book can move during it) and would only burn
        // wall-clock time on every order.
        0,
        risk.paper_taker_fee_rate,
        risk.paper_pair_fill_correlation,
        fill_tx,
        registry.clone(),
        params.initial_balance,
        wallet_tx,
        params.seed,
        clock,
    );

    let events = merge_events(books, spots);
    let mut vols: HashMap<String, VolReplay> = HashMap::new();
    let mut metas: HashMap<String, MarketMeta> = HashMap::new();
    let mut last_bid: HashMap<String, Decimal> = HashMap::new();
    let mut ledger = Ledger::new(params.initial_balance);
    let mut rt = Rt::default();
    let mut n_books = 0u64;
    let mut n_spots = 0u64;
    // The timer starts at the FIRST real event — from `from_ms` (which may
    // be i64::MIN+1 by default) the while loop would advance billions of ticks.
    let first_ts = events.first().map(|e| e.ts()).unwrap_or(0);
    let mut next_timer_ms = (first_ts / TIMER_MS + 1) * TIMER_MS;
    // Global replay clock: `secs_to_clear` and the strategies' `Utc::now`
    // read the event timestamp, not wall-clock — without this
    // every recorded book looks already closed and no intents appear. Released
    // on exit (ReplayClockGuard, RAII).
    let _replay_clock = crate::time::ReplayClockGuard::new(first_ts);

    for ev in events {
        let ts = ev.ts();
        clock_ms.store(ts, Ordering::Relaxed);
        crate::time::set_replay_now_ms(ts);

        // Runner watchdog: on_timer every 5 s of replay time.
        while ts >= next_timer_ms {
            rt.timer_ticks += 1;
            let intents = strategy.on_timer();
            for intent in intents {
                dispatch(
                    &sim,
                    intent,
                    &mut rt,
                    &mut *strategy,
                    &mut fill_rx,
                    &mut ledger,
                )
                .await;
            }
            next_timer_ms += TIMER_MS;
        }

        match ev {
            Event::Spot(s) => {
                n_spots += 1;
                vols.entry(s.asset.clone())
                    .or_default()
                    .push(&s.source, s.ts_ms, s.price);
            }
            Event::Book(row) => {
                n_books += 1;
                let snap = row.snap;
                // Same record that discovery would produce: clear_at_cache so
                // fills carry the real clear_at/neg_risk.
                registry.remember_clear_at(
                    &snap.market_condition_id,
                    snap.neg_risk,
                    snap.clear_at,
                    &snap.yes.token_id,
                    &snap.no.token_id,
                );
                metas
                    .entry(snap.market_condition_id.clone())
                    .or_insert_with(|| MarketMeta {
                        yes_token: snap.yes.token_id.clone(),
                        slug: snap.slug.clone(),
                        clear_at: snap.clear_at,
                        reference_price: snap.reference_price,
                    });
                if let Some(b) = snap.yes.best_bid() {
                    last_bid.insert(snap.yes.token_id.clone(), b);
                }
                if let Some(b) = snap.no.best_bid() {
                    last_bid.insert(snap.no.token_id.clone(), b);
                }

                sim.tick_book(&snap);
                drain_fills(&mut fill_rx, &mut *strategy, &mut ledger, &mut rt);

                let vol = crate::oracle::asset_from_slug(&snap.slug)
                    .and_then(|a| vols.get(&a).map(|v| v.snapshot(&a)))
                    .unwrap_or_default();
                let intents = strategy.on_book(&snap, &vol);
                for intent in intents {
                    dispatch(
                        &sim,
                        intent,
                        &mut rt,
                        &mut *strategy,
                        &mut fill_rx,
                        &mut ledger,
                    )
                    .await;
                }
            }
        }
        settle_due(ts, &metas, &vols, &mut ledger);
    }

    // Final sweep with the last OBSERVED ts (no further): a bracket whose
    // clear_at falls outside the window is NOT settled — we do not know its
    // resolution and fabricating it with the last spot would be look-ahead.
    let last_ts = clock_ms.load(Ordering::Relaxed);
    settle_due(last_ts, &metas, &vols, &mut ledger);
    drain_fills(&mut fill_rx, &mut *strategy, &mut ledger, &mut rt);

    // Marked at the last known best_bid (conservative: exit price).
    // "Unresolved" = the bracket ALREADY closed within the window but lacked
    // evidence (spot/strike) to settle it — not just any live position.
    let mut unrealized = Decimal::ZERO;
    let mut open = Vec::new();
    let mut unresolved = Vec::new();
    for ((cid, token), pos) in &ledger.positions {
        let mark = last_bid.get(token).copied();
        unrealized += mark.unwrap_or(Decimal::ZERO) * pos.shares;
        let meta = metas.get(cid);
        let outcome_label = if meta.is_some_and(|m| m.yes_token == *token) {
            "YES"
        } else {
            "NO"
        };
        let rep = PositionReport {
            market_cid: cid.clone(),
            token_id: token.clone(),
            outcome_label: outcome_label.to_string(),
            shares: pos.shares.to_string(),
            cost: pos.cost.to_string(),
            mark: mark.map(|m| m.to_string()),
        };
        let closed_unsettled = meta.is_some_and(|m| last_ts > m.clear_at * 1000);
        if closed_unsettled {
            unresolved.push(rep);
        } else {
            open.push(rep);
        }
    }
    open.sort_by(|a, b| {
        a.market_cid
            .cmp(&b.market_cid)
            .then(a.token_id.cmp(&b.token_id))
    });
    unresolved.sort_by(|a, b| {
        a.market_cid
            .cmp(&b.market_cid)
            .then(a.token_id.cmp(&b.token_id))
    });

    let mut by_bracket: Vec<BracketReport> = ledger
        .per_bracket
        .iter()
        .map(|(cid, acc)| BracketReport {
            market_cid: cid.clone(),
            slug: metas.get(cid).map(|m| m.slug.clone()).unwrap_or_default(),
            fills: acc.fills,
            pnl: acc.pnl.to_string(),
        })
        .collect();
    by_bracket.sort_by(|a, b| a.market_cid.cmp(&b.market_cid));

    let live_gtds = sim
        .open_gtd_orders()
        .await
        .map(|v| v.len() as u64)
        .unwrap_or(0);

    let equity = ledger.cash + unrealized;
    Ok(BacktestReport {
        strategy: params.strategy.to_string(),
        from_ms,
        to_ms,
        seed: params.seed,
        books_replayed: n_books,
        spot_samples: n_spots,
        timer_ticks: rt.timer_ticks,
        intents: rt.intents,
        orders_submitted: rt.orders,
        intents_without_request: rt.without_request,
        fills: rt.fills,
        rejects: rt.rejects,
        unknowns: rt.unknowns,
        cancels_ok: rt.cancels_ok,
        cancel_without_order: rt.cancel_without_order,
        initial_balance: params.initial_balance.to_string(),
        final_cash: ledger.cash.to_string(),
        spent_buys: ledger.spent_buys.to_string(),
        proceeds_sells: ledger.proceeds_sells.to_string(),
        payouts_settlement: ledger.payouts.to_string(),
        realized_pnl: ledger.realized.to_string(),
        unrealized_pnl: unrealized.to_string(),
        equity_total: equity.to_string(),
        max_exposure_usdc: ledger.max_exposure.to_string(),
        accounting_consistent: ledger.consistent(),
        settlement_provenance: crate::research::labels::PROVENANCE_PROXY_SPOT.to_string(),
        settled_positions: ledger.settled,
        won_liquidations: ledger.won,
        lost_liquidations: ledger.lost,
        live_gtds_at_end: live_gtds,
        open_positions: open,
        unresolved_positions: unresolved,
        by_bracket,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    /// DDL copied from `tools::recorders::analytics::ensure_table` and
    /// `tools::recorders::spot_stream::ensure_tables` — same schema as production.
    const DDL_BOOKS: &str = "
        CREATE TABLE book_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL, market_condition_id TEXT NOT NULL,
            slug TEXT NOT NULL, clear_at INTEGER NOT NULL, market_open_ts INTEGER NOT NULL,
            neg_risk INTEGER NOT NULL, reference_price TEXT, fee_rate_bps INTEGER,
            parabolic_fee_rate TEXT, tick_size TEXT,
            yes_token_id TEXT NOT NULL, yes_best_bid TEXT, yes_best_ask TEXT, yes_mid TEXT,
            yes_bids_json TEXT NOT NULL, yes_asks_json TEXT NOT NULL,
            no_token_id TEXT NOT NULL, no_best_bid TEXT, no_best_ask TEXT, no_mid TEXT,
            no_bids_json TEXT NOT NULL, no_asks_json TEXT NOT NULL, spot TEXT,
            UNIQUE(market_condition_id, ts_ms));
        CREATE INDEX idx_book_snapshots_market ON book_snapshots(market_condition_id);
        CREATE INDEX idx_book_snapshots_ts ON book_snapshots(ts_ms);";

    const DDL_SPOT: &str = "
        CREATE TABLE spot_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL, source TEXT NOT NULL, price TEXT NOT NULL,
            source_ts_ms INTEGER NOT NULL, recv_ts_ms INTEGER NOT NULL, avail_ts_ms INTEGER NOT NULL);
        CREATE INDEX idx_spot_samples_asset_source ON spot_samples(asset, source);
        CREATE INDEX idx_spot_samples_source_ts ON spot_samples(source_ts_ms);";

    fn insert_book(conn: &Connection, ts_ms: i64, clear_at_s: i64, yes_ask: &str) {
        conn.execute(
            "INSERT INTO book_snapshots (ts_ms, market_condition_id, slug, clear_at,
                market_open_ts, neg_risk, reference_price, tick_size,
                yes_token_id, yes_bids_json, yes_asks_json,
                no_token_id, no_bids_json, no_asks_json)
             VALUES (?1,'cid1','btc-updown-test',?2,?3,0,'50000','0.01',
                'yes-tok','[{\"price\":\"0.44\",\"size\":\"100\"}]',
                ?4,
                'no-tok','[{\"price\":\"0.54\",\"size\":\"100\"}]',
                '[{\"price\":\"0.55\",\"size\":\"100\"}]')",
            rusqlite::params![
                ts_ms,
                clear_at_s,
                clear_at_s - 3600,
                format!("[{{\"price\":\"{}\",\"size\":\"100\"}}]", yes_ask)
            ],
        )
        .unwrap();
    }

    fn insert_spot(conn: &Connection, ts_ms: i64, price: &str) {
        conn.execute(
            "INSERT INTO spot_samples (asset, source, price, source_ts_ms, recv_ts_ms, avail_ts_ms)
             VALUES ('btc','binance',?1,?2,?2,?2)",
            rusqlite::params![price, ts_ms],
        )
        .unwrap();
    }

    fn fixture(dir: &Path, clear_at_s: i64, spot_price: &str) -> (PathBuf, PathBuf) {
        let db_books = dir.join("books.sqlite");
        let db_spot = dir.join("spot.sqlite");
        let cb = Connection::open(&db_books).unwrap();
        cb.execute_batch(DDL_BOOKS).unwrap();
        let base = clear_at_s * 1000 - 120_000;
        for i in 0..10 {
            insert_book(&cb, base + i * 1000, clear_at_s, "0.45");
        }
        let cs = Connection::open(&db_spot).unwrap();
        cs.execute_batch(DDL_SPOT).unwrap();
        for i in 0..60 {
            insert_spot(&cs, base + i * 1000, spot_price);
        }
        (db_books, db_spot)
    }

    /// Self-contained test strategy: a buy on the first book.
    struct SingleBuy {
        done: bool,
    }
    impl Strategy for SingleBuy {
        fn id(&self) -> StrategyId {
            StrategyId::Arb
        }
        fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
            if self.done {
                return vec![];
            }
            self.done = true;
            vec![OrderIntent::taker(
                book.market_condition_id.clone(),
                book.yes.token_id.clone(),
                OrderSide::Buy,
                dec!(0.50),
                dec!(10),
                dec!(1),
                StrategyId::Arb,
                book,
            )]
        }
    }

    fn params(db_books: PathBuf, db_spot: PathBuf) -> BacktestParams {
        BacktestParams {
            db_books,
            db_spot: Some(db_spot),
            strategy: StrategyId::Arb,
            ..Default::default()
        }
    }

    /// Strategy that buys YES only on a specific cid (for the
    /// settlement test: buys on the bracket that goes to close).
    struct BuyAt {
        cid: &'static str,
        done: bool,
    }
    impl Strategy for BuyAt {
        fn id(&self) -> StrategyId {
            StrategyId::Arb
        }
        fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
            if self.done || book.market_condition_id != self.cid {
                return vec![];
            }
            self.done = true;
            vec![OrderIntent::taker(
                book.market_condition_id.clone(),
                book.yes.token_id.clone(),
                OrderSide::Buy,
                dec!(0.97),
                dec!(10),
                dec!(1),
                StrategyId::Arb,
                book,
            )]
        }
    }

    /// Buys 10 shares at ask 0.45 (limit 0.50) → fill; accounting invariant:
    /// cash = initial − buys + sells + payouts. The market does not close in the
    /// window → position stays open, marked at bid 0.44.
    #[tokio::test]
    async fn replay_buy_and_accounting_balance() {
        // Serializes the runs that set the global clock — see REPLAY_TEST_LOCK.
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let clear_s = (chrono::Utc::now().timestamp() / 3600 + 2) * 3600;
        let (db_books, db_spot) = fixture(dir.path(), clear_s, "51000");
        let rep = run_replay(
            &params(db_books, db_spot),
            Box::new(SingleBuy { done: false }),
            &crate::config::RiskConfig::default(),
        )
        .await
        .unwrap();
        assert!(rep.fills >= 1, "expected at least one fill: {rep:?}");
        assert!(rep.accounting_consistent);
        assert_eq!(rep.open_positions.len(), 1);
        // The test strategy buys 10 at cap 0.50 over ask 0.45.
        let spent: Decimal = Decimal::from_str(&rep.spent_buys).unwrap();
        assert!(spent > Decimal::ZERO && spent <= dec!(5));
    }

    /// Same input → same output byte for byte (fixed seed + injected clock).
    #[tokio::test]
    async fn deterministic_replay() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let clear_s = (chrono::Utc::now().timestamp() / 3600 + 2) * 3600;
        let (db_books, db_spot) = fixture(dir.path(), clear_s, "51000");
        let mk = || params(db_books.clone(), db_spot.clone());
        let r1 = run_replay(
            &mk(),
            Box::new(SingleBuy { done: false }),
            &crate::config::RiskConfig::default(),
        )
        .await
        .unwrap();
        let r2 = run_replay(
            &mk(),
            Box::new(SingleBuy { done: false }),
            &crate::config::RiskConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            serde_json::to_string(&r1).unwrap(),
            serde_json::to_string(&r2).unwrap()
        );
    }

    /// Position open when `clear_at` passes with spot > strike → YES wins,
    /// payout = shares, realized = payout − cost. E2E via `run_replay`: a
    /// second market (`cid2`) with `clear_at` already past within the window
    /// is settled when the replay crosses it.
    #[tokio::test]
    async fn settlement_yes_wins_pays_one() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let clear_s = (chrono::Utc::now().timestamp() / 3600 + 2) * 3600;
        let (db_books, db_spot) = fixture(dir.path(), clear_s, "51000");
        let past_clear = clear_s - 1800;
        let cb = Connection::open(&db_books).unwrap();
        cb.execute(
            "INSERT INTO book_snapshots (ts_ms, market_condition_id, slug, clear_at,
                market_open_ts, neg_risk, reference_price, tick_size,
                yes_token_id, yes_bids_json, yes_asks_json,
                no_token_id, no_bids_json, no_asks_json)
             VALUES (?1,'cid2','btc-updown-old',?2,?3,0,'50000','0.01',
                'yes2','[{\"price\":\"0.95\",\"size\":\"50\"}]',
                '[{\"price\":\"0.96\",\"size\":\"50\"}]',
                'no2','[{\"price\":\"0.03\",\"size\":\"50\"}]',
                '[{\"price\":\"0.04\",\"size\":\"50\"}]')",
            rusqlite::params![past_clear * 1000 - 60_000, past_clear, past_clear - 3600],
        )
        .unwrap();
        drop(cb);

        let rep = run_replay(
            &params(db_books, db_spot),
            Box::new(BuyAt {
                cid: "cid2",
                done: false,
            }),
            &crate::config::RiskConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(rep.settled_positions, 1);
        assert_eq!(rep.won_liquidations, 1);
        let realized = Decimal::from_str(&rep.realized_pnl).unwrap();
        assert!(realized > Decimal::ZERO, "expected a gain: {rep:?}");
        assert!(rep.accounting_consistent);
    }

    /// Taker YES+NO pair (`second_leg` Buy, `!resting`): the executor sends it
    /// via `submit_pair` (one correlated draw), not as two independent
    /// submits. With fill_rate=1 and ample depth both legs fill —
    /// the intent counts ONCE and positions end up on both tokens.
    #[tokio::test]
    async fn replay_taker_pair_both_legs() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        struct PairTaker {
            done: bool,
        }
        impl Strategy for PairTaker {
            fn id(&self) -> StrategyId {
                StrategyId::Arb
            }
            fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
                if self.done {
                    return vec![];
                }
                self.done = true;
                let mut i = OrderIntent::taker(
                    book.market_condition_id.clone(),
                    book.yes.token_id.clone(),
                    OrderSide::Buy,
                    dec!(0.50),
                    dec!(10),
                    dec!(1),
                    StrategyId::Arb,
                    book,
                );
                i.second_leg = Some(Box::new(OrderIntent::taker(
                    book.market_condition_id.clone(),
                    book.no.token_id.clone(),
                    OrderSide::Buy,
                    dec!(0.60),
                    dec!(10),
                    dec!(1),
                    StrategyId::Arb,
                    book,
                )));
                vec![i]
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let clear_s = (chrono::Utc::now().timestamp() / 3600 + 2) * 3600;
        let (db_books, db_spot) = fixture(dir.path(), clear_s, "51000");
        let risk = crate::config::RiskConfig {
            paper_fok_fill_rate: 1.0,
            paper_latency_ms: 0,
            ..Default::default()
        };
        let rep = run_replay(
            &params(db_books, db_spot),
            Box::new(PairTaker { done: false }),
            &risk,
        )
        .await
        .unwrap();
        assert_eq!(rep.intents, 1);
        assert_eq!(rep.orders_submitted, 2);
        assert_eq!(rep.fills, 2, "expected a fill on both legs: {rep:?}");
        assert_eq!(rep.open_positions.len(), 2);
        assert!(rep.accounting_consistent);
    }

    /// Empty books in the window → explicit error, never an empty report.
    #[tokio::test]
    async fn window_without_data_error() {
        // `run` sets up the guard even if the window comes out empty — same lock.
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let (db_books, db_spot) = fixture(dir.path(), 4_000_000_000, "51000");
        let mut p = params(db_books, db_spot);
        p.from_ms = Some(1);
        p.to_ms = Some(2);
        assert!(run(&p).await.is_err());
    }

    /// Gate 6: the strategies' `crate::time::Instant` values measure
    /// REPLAY time — an internal TTL must fire when the event clock
    /// advances. With `std::time::Instant` the replay runs compressed and
    /// `elapsed()` would be ~0 forever: this strategy would never emit.
    #[tokio::test]
    async fn strategy_ttl_advances_in_replay_time() {
        let _serial = crate::time::REPLAY_TEST_LOCK.lock().await;
        /// Emits a buy only when its internal mark is ≥3 s old in
        /// replay time (the fixture's 10 books go 1 s apart → from
        /// the 4th book on it must fire).
        struct Ttl {
            t0: Option<crate::time::Instant>,
        }
        impl Strategy for Ttl {
            fn id(&self) -> StrategyId {
                StrategyId::Arb
            }
            fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
                let Some(t0) = self.t0 else {
                    self.t0 = Some(crate::time::Instant::now());
                    return vec![];
                };
                if t0.elapsed() < std::time::Duration::from_secs(3) {
                    return vec![];
                }
                vec![OrderIntent::taker(
                    book.market_condition_id.clone(),
                    book.yes.token_id.clone(),
                    OrderSide::Buy,
                    dec!(0.50),
                    dec!(10),
                    dec!(1),
                    StrategyId::Arb,
                    book,
                )]
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let clear_s = (chrono::Utc::now().timestamp() / 3600 + 2) * 3600;
        let (db_books, db_spot) = fixture(dir.path(), clear_s, "51000");
        let rep = run_replay(
            &params(db_books, db_spot),
            Box::new(Ttl { t0: None }),
            &crate::config::RiskConfig::default(),
        )
        .await
        .unwrap();
        assert!(
            rep.orders_submitted >= 1,
            "the 3 s TTL must fire when the replay clock advances: {rep:?}"
        );
    }
}
