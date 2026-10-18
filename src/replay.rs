//! Event engine and deterministic replay of the REAL execution pipeline.
//!
//! Unlike the diagnostic backtest (`crate::backtest`, which bypasses the
//! Executor and keeps its own simplified ledger), here the recorded tape is
//! driven through the production components, with no exceptions:
//!
//! ```text
//! tape (book_snapshots/spot_samples)
//!   → virtual queue ordered by (ts, seq)
//!   → strategy (on_book/on_timer/on_fill, production)
//!   → OrderIntent → Executor (dispatch_replay: gate stale + handle)
//!   → RiskGuard + BracketLedger (reservations and real gates)
//!   → DeferredSimRouter → SimulatedRouter (deferred arrival at the venue)
//!   → accounting::fill_sink::process_fill → StateStore SQLite :memory:
//!   → broadcast → strategy
//! ```
//!
//! Properties:
//! - Zero wall-clock sleeps: every wait is an event in the virtual queue. The
//!   submit/cancel latency is modeled as a deferred arrival at the venue
//!   (`DeferredSimRouter`), not as `tokio::time::sleep`.
//! - Arrival/observation distinction: the book enters the simulated exchange at
//!   `recv_ms` (`tick_book` — resting GTDs may already fill) but the agent
//!   only sees it at `recv_ms + obs_delay_ms` (`on_book`).
//! - Determinism: same tape + same seed ⇒ same economic ledger, byte for
//!   byte. The ordering is `(ts_ms, seq)`, stable; no `HashMap` decides order.
//! - Ephemeral state: `StateStore::open_memory` — never the active DB.
//! - The `Executor` runs as a real task (it may stay parked waiting for the
//!   deferred venue response); the driver detects quiescence through
//!   counters — see `settle`.
//!
//! Concurrency: the engine requires a `current_thread` runtime (tests set it
//! with `#[tokio::test(flavor = "current_thread")]` and `hrf_backtest` with
//! `#[tokio::main(flavor = "current_thread")]`). On a single thread the order of
//! execution order between awaits is deterministic and quiescence detection by
//! counters is exact — no scheduling races between the driver and the
//! executor task.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context};
use parking_lot::Mutex;
use rust_decimal::Decimal;
use tokio::sync::{broadcast, mpsc, oneshot, watch};

use crate::backtest::{load_books, load_spots, open_readonly, VolReplay};
use crate::config::{LatencyLeg, LatencyProfile};
use crate::coordination::BracketLedger;
use crate::accounting::fill_sink;
use crate::market_feed::MarketRegistry;
use crate::order::executor::Executor;
use crate::order::router::{
    OrderRequest, OrderRouter, RouterOpenOrder, RouterOrderStatus, RouterResult, RouterTrade,
};
use crate::order::sim_router::{SimClock, SimExecution, SimulatedRouter};
use crate::risk::RiskGuard;
use crate::state::StateStore;
use crate::strategy::Strategy;
use crate::types::{
    BookSnapshot, Fill, FillSide, Mode, OrderIntent, OrderSide, StrategyId, VolSnapshot,
};

/// Iteration cap for the quiescence loop: if the pipeline does not settle
/// still after this many yields there is a bug (an await that is neither a deferred job
/// nor a job that finishes). Failing loudly is better than hanging.
const MAX_SETTLE_YIELDS: usize = 400_000;

// ── Tape ─────────────────────────────────────────────────────────────────────

/// A recorded book: `recv_ms` is the arrival instant at the simulated exchange
/// (in the tape it is the recording ts). The agent observes it in
/// `recv_ms + obs_delay_ms`, or at `avail_ms` when the tape carries the mark
/// of real availability (`ResearchEvent.avail_ts_ms` — causal tape).
#[derive(Debug, Clone)]
pub struct TapeBook {
    pub recv_ms: i64,
    /// Real availability to the consumer (`ResearchEvent` tape): if it is
    /// present sets the instant of `BookObserve` — `obs_delay_ms` and the
    /// `feed` stage of the profile are NOT added (the delay is already measured).
    pub avail_ms: Option<i64>,
    /// `local_seq` of the `ResearchEvent` that produced this entry (ordering
    /// of the causal tape). `None` for a historical SQL tape.
    pub tape_seq: Option<u64>,
    pub snap: BookSnapshot,
}

/// A spot sample from the oracle (Binance). Observation is immediate:
/// the oracle feeds a continuous watch, not a discrete book event.
#[derive(Debug, Clone)]
pub struct TapeSpot {
    pub ts_ms: i64,
    /// `local_seq` of the `ResearchEvent` origin (causal tape).
    pub tape_seq: Option<u64>,
    pub asset: String,
    pub price: Decimal,
}

/// Official resolution known during capture (`ResearchEvent` tape).
/// The event is processed at `avail_ms` — the instant the result
/// reached the bot — and `apply_official`/`confirmed_label` also apply the
/// `known_at_ms` filter of the outcome itself (double causal barrier).
#[derive(Debug, Clone)]
pub struct TapeResolution {
    pub avail_ms: i64,
    /// `local_seq` of the `ResearchEvent` origin.
    pub tape_seq: Option<u64>,
    pub cid: String,
    pub outcome: crate::research::contract::MarketOutcome,
    pub source: crate::research::contract::EventSource,
}

/// Quality mark of the tape positioned in time (segment of
/// invalidated deltas, sequence gap…). It does not alter the venue state:
/// it is only recorded for diagnostics and episode terminality.
#[derive(Debug, Clone)]
pub struct TapeQuality {
    pub ts_ms: i64,
    pub tape_seq: Option<u64>,
    pub cid: Option<String>,
    pub flag: crate::research::contract::QualityFlags,
}

/// Tape provenance — decides the `fidelity` declaration of the outcome.
/// `ResearchEvents` is `"certified"` only if `avail_ms`/`recv_ms` are
/// real (the loader guarantees it: they come from the recorder, never invented).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TapeSource {
    /// Tape reconstructed from the historical DBs (book_snapshots +
    /// spot_samples) — reduced granularity, always `"reduced"`.
    #[default]
    Sql,
    /// Causal `ResearchEvent` tape (bot recorder).
    ResearchEvents,
}

/// The complete tape of a replay.
#[derive(Debug, Default, Clone)]
pub struct Tape {
    pub books: Vec<TapeBook>,
    pub spots: Vec<TapeSpot>,
    /// Official resolutions from the tape (empty for a SQL tape — the
    /// settlement then falls back to the diagnostic proxy or stays pending).
    pub resolutions: Vec<TapeResolution>,
    /// Quality marks positioned in time (gaps, segments
    /// invalidated by deltas with no base…).
    pub quality: Vec<TapeQuality>,
    /// Addendum v2: quality metrics aggregated from the tape (count per
    /// token/cid for the `quality` report and the fidelity rule per
    /// scope). The `research_replay` reconstructor populates it. For a
    /// SQL tape it is empty (the scope counters are 0 — fidelity is already
    /// `"reduced"` by the source itself).
    pub quality_stats: TapeQualityStats,
    pub source: TapeSource,
}

/// Quality metrics aggregated from a reconstructed tape.
///
/// **"Scope"** = the cids/tokens of the markets whose stream was processed
/// in the run window: the `MarketMeta` observed by the reconstructor
/// (`cids_meta`). If the episode filters by market, `filter_tape` limits
/// `cids_meta` to that cid and the rest of the degradation falls outside the
/// scope (it is declared in the global counters but does not taint the
/// certification). The `TapeQuality` marks carry their `cid` for the same
/// filtering.
#[derive(Debug, Clone, Default)]
pub struct TapeQualityStats {
    /// Cids with an observed `MarketMeta` in the window — the scope.
    pub meta_cids: BTreeSet<String>,
    /// token → cid according to the observed market metadata (yes and no of each market).
    pub token2cid: HashMap<String, String>,
    /// `price_change` entries processed per cid resolved (applied +
    /// discarded) — denominator of the scope coverage.
    pub deltas_by_cid: HashMap<String, u64>,
    /// `price_change` entries with no attributable cid, per token (the meta that
    /// would have attributed them may arrive later in the window).
    pub deltas_no_cid_by_token: HashMap<String, u64>,
    /// discarded deltas per token (no base / `needs_snapshot`).
    pub discarded_deltas_by_token: HashMap<String, u64>,
    /// discarded deltas per cid, resolved (`market_cid` of the event or of the
    /// payload, or `token2cid` of the token).
    pub discarded_deltas_by_cid: HashMap<String, u64>,
    /// discarded deltas with no attributable cid, per token — the scope
    /// recovers them if the token turns out to belong to an observed market.
    pub discarded_deltas_no_cid_by_token: HashMap<String, u64>,
    /// Tokens with a gap episode: segment invalidated by the
    /// reconstructor or a `GAP` mark from the recorder placed over the token.
    pub tokens_with_gap: HashSet<String>,
    /// Quality marks emitted to the tape per cid (`None` is counted
    /// separately — a mark without a cid cannot be attributed to any scope).
    pub marks_by_cid: HashMap<String, u64>,
    /// Marks emitted without an attributable cid.
    pub marks_without_cid: u64,
    /// Global counters for reconstruction/loading.
    pub discarded_deltas: u64,
    pub malformed: u64,
    pub agent: u64,
    pub missing_half: u64,
    pub books_top_equal: u64,
}

impl TapeQualityStats {
    /// Bound on the cids listed in `scope_gap_markets` (bounded report).
    pub const MAX_GAP_MARKETS: usize = 50;

    /// Tokens belonging to the scope (yes/no of the observed cids).
    fn in_scope_tokens(&self) -> HashSet<&str> {
        self.token2cid
            .iter()
            .filter(|(_, c)| self.meta_cids.contains(*c))
            .map(|(t, _)| t.as_str())
            .collect()
    }

    /// `quality` summary of the contract over the tape's scope.
    ///
    /// `quality_marks`/`resyncs` are the marks ACTUALLY processed
    /// in the run window (the engine's `quality_log` in the report,
    /// or the `tape.quality` preloaded in the env's `hello`); `gap_cids`
    /// carries the cids with a processed `GAP` mark — the scope filter
    /// is applied here.
    pub fn summary(
        &self,
        quality_marks: u64,
        resyncs: u64,
        gap_cids: &HashSet<String>,
    ) -> crate::research::contract::QualitySummary {
        let in_scope_tokens = self.in_scope_tokens();
        // Deltas in scope: those attributed to an observed cid + the
        // unattributable at discard time whose token belongs to
        // a market in scope (the meta arrived after the delta).
        let discarded_deltas_in_scope: u64 = self
            .discarded_deltas_by_cid
            .iter()
            .filter(|(c, _)| self.meta_cids.contains(*c))
            .map(|(_, n)| *n)
            .sum::<u64>()
            + self
                .discarded_deltas_no_cid_by_token
                .iter()
                .filter(|(t, _)| in_scope_tokens.contains(t.as_str()))
                .map(|(_, n)| *n)
                .sum::<u64>();
        let in_scope_totals: u64 = self
            .deltas_by_cid
            .iter()
            .filter(|(c, _)| self.meta_cids.contains(*c))
            .map(|(_, n)| *n)
            .sum::<u64>()
            + self
                .deltas_no_cid_by_token
                .iter()
                .filter(|(t, _)| in_scope_tokens.contains(t.as_str()))
                .map(|(_, n)| *n)
                .sum::<u64>();
        let tokens_with_gap_in_scope = self
            .tokens_with_gap
            .iter()
            .filter(|t| in_scope_tokens.contains(t.as_str()))
            .count() as u64;
        // Markets with a gap in scope: cids of processed GAP marks +
        // cids derived from gapped tokens via `token2cid` — sorted and
        // bounded to `MAX_GAP_MARKETS`.
        let mut gap_markets: BTreeSet<String> = gap_cids
            .iter()
            .filter(|c| self.meta_cids.contains(*c))
            .cloned()
            .collect();
        for t in &self.tokens_with_gap {
            if let Some(c) = self.token2cid.get(t) {
                if self.meta_cids.contains(c) {
                    gap_markets.insert(c.clone());
                }
            }
        }
        let coverage = if in_scope_totals == 0 {
            100.0 // no deltas in scope, so there is no loss to declare
        } else {
            (100.0 * (in_scope_totals - discarded_deltas_in_scope) as f64 / in_scope_totals as f64)
                .max(0.0)
        };
        crate::research::contract::QualitySummary {
            quality_marks,
            malformed: self.malformed,
            agent: self.agent,
            discarded_deltas: self.discarded_deltas,
            discarded_deltas_in_scope,
            missing_half: self.missing_half,
            books_top_equal: self.books_top_equal,
            resyncs,
            tokens_with_gap_in_scope,
            gap_markets_in_scope: gap_markets
                .into_iter()
                .take(Self::MAX_GAP_MARKETS)
                .collect(),
            delta_coverage_pct_in_scope: coverage,
        }
    }
}

/// Loads the tape from the recorded DBs (read-only, same convention as the
/// diagnostic backtest — the evidence is not touched).
pub fn load_tape(
    db_books: &std::path::Path,
    db_spot: Option<&std::path::Path>,
    from_ms: i64,
    to_ms: i64,
) -> anyhow::Result<Tape> {
    let conn = open_readonly(db_books)?;
    let books = load_books(&conn, from_ms, to_ms)?
        .into_iter()
        .map(|r| TapeBook {
            recv_ms: r.ts_ms,
            avail_ms: None,
            tape_seq: None,
            snap: r.snap,
        })
        .collect();
    let spots = match db_spot {
        Some(p) => load_spots(&open_readonly(p)?, from_ms, to_ms)?
            .into_iter()
            .map(|s| TapeSpot {
                ts_ms: s.ts_ms,
                tape_seq: None,
                asset: s.asset,
                price: s.price,
            })
            .collect(),
        None => Vec::new(),
    };
    Ok(Tape {
        books,
        spots,
        ..Default::default()
    })
}

// ── Parameters ────────────────────────────────────────────────────────────────

/// Parameters of the replay through the real pipeline.
#[derive(Debug, Clone)]
pub struct ReplayParams {
    /// Seed of the sim RNG (`with_deps`) — same input ⇒ same output.
    pub seed: u64,
    pub initial_balance: Decimal,
    /// REAL risk config (caps, cooldowns, kill) — the same one that produces
    /// `RiskGuard` in production.
    pub risk: crate::config::RiskConfig,
    /// Agent observation delay after arrival at the exchange (ms).
    /// `0` = the agent sees the book at the same instant it enters the venue.
    pub obs_delay_ms: i64,
    /// agent→venue latency applied to submits/cancels/query (ms): the
    /// `OrderRequest` is evaluated against the sim book at `now + latency`.
    pub latency_ms: i64,
    /// Cadence of the executor maintenance (`purge_and_resync`) —
    /// 30_000 ms in production.
    pub maintenance_ms: i64,
    /// Strategy watchdog cadence (`on_timer`) — 5_000 ms.
    pub timer_ms: i64,
    /// FAK strategies (the `fak_strategies` table of `main.rs`).
    pub fak_strategies: HashSet<StrategyId>,
    /// Runner's stale-book gate (`strategy::run` → `entry_block`):
    /// `market_feed.max_book_age_ms` in production (500 ms by default).
    /// The age is measured on the virtual clock: `observe_ts − recv_ms` (= obs_delay).
    pub max_book_age_ms: i64,
    /// Research profile of the sim (`κ`, participation, stale,
    /// gaps, inventory). `None` = nominal (the sim default reproduces the
    /// operational behavior); `Some` is applied via
    /// `apply_research_profile` when building the `SimulatedRouter`.
    pub research: Option<crate::config::ResearchSimConfig>,
    /// Settlement proxy from the last spot (`--settlement-proxy-
    /// diagnostic`). `false` (default): only markets with a label are settled
    /// official `confirmed` and `known_at_ms <= now`; without a label they stay
    /// pending (truncated, never an invented payout). `true` = explicit
    /// diagnostic that forces `fidelity="reduced"` in the outcome.
    pub settlement_proxy_diagnostic: bool,
}

impl Default for ReplayParams {
    fn default() -> Self {
        Self {
            seed: 0xC0FF_EE15,
            initial_balance: Decimal::new(5_000, 0),
            risk: crate::config::RiskConfig::default(),
            obs_delay_ms: 0,
            latency_ms: 0,
            maintenance_ms: 30_000,
            timer_ms: 5_000,
            fak_strategies: HashSet::new(),
            max_book_age_ms: crate::config::defaults::max_book_age_ms() as i64,
            research: None,
            settlement_proxy_diagnostic: false,
        }
    }
}

// ── Virtual event queue ───────────────────────────────────────────────────

/// Deferred work toward the venue: each `Executor` call to the router
/// reaches the simulated exchange `latency_ms` after being emitted. The result
/// is returned through a `oneshot` when the driver processes the job at its ts.
enum RouterJob {
    Submit {
        req: OrderRequest,
        tx: oneshot::Sender<RouterResult>,
    },
    SubmitPair {
        req1: OrderRequest,
        req2: OrderRequest,
        tx: oneshot::Sender<(RouterResult, RouterResult)>,
    },
    Cancel {
        cid: String,
        order_id: String,
        tx: oneshot::Sender<anyhow::Result<()>>,
    },
    LiveGtd {
        tx: oneshot::Sender<Option<HashSet<String>>>,
    },
    OrderStatus {
        order_id: String,
        tx: oneshot::Sender<Option<RouterOrderStatus>>,
    },
    OpenGtd {
        tx: oneshot::Sender<Option<Vec<RouterOpenOrder>>>,
    },
    TradesSince {
        after_ts_secs: i64,
        tx: oneshot::Sender<Option<Vec<RouterTrade>>>,
    },
}

/// Venue response already resolved by the `SimulatedRouter`, pending
/// DELIVERY to the executor at its `ts_ms` — the `ack` stage of the profile
/// latencies (`[research_sim.latencies]`): the venue processes the submit/cancel
/// on job arrival, and the return of the response to the agent is another hop.
enum RouterResp {
    Submit(oneshot::Sender<RouterResult>, RouterResult),
    SubmitPair(
        oneshot::Sender<(RouterResult, RouterResult)>,
        (RouterResult, RouterResult),
    ),
    Cancel(oneshot::Sender<anyhow::Result<()>>, anyhow::Result<()>),
    LiveGtd(
        oneshot::Sender<Option<HashSet<String>>>,
        Option<HashSet<String>>,
    ),
    OrderStatus(
        oneshot::Sender<Option<RouterOrderStatus>>,
        Option<RouterOrderStatus>,
    ),
    OpenGtd(
        oneshot::Sender<Option<Vec<RouterOpenOrder>>>,
        Option<Vec<RouterOpenOrder>>,
    ),
    TradesSince(
        oneshot::Sender<Option<Vec<RouterTrade>>>,
        Option<Vec<RouterTrade>>,
    ),
}

impl RouterResp {
    /// Delivers the response to the parked executor (the `send` never blocks).
    fn deliver(self) {
        match self {
            RouterResp::Submit(tx, r) => {
                let _ = tx.send(r);
            }
            RouterResp::SubmitPair(tx, r) => {
                let _ = tx.send(r);
            }
            RouterResp::Cancel(tx, r) => {
                let _ = tx.send(r);
            }
            RouterResp::LiveGtd(tx, r) => {
                let _ = tx.send(r);
            }
            RouterResp::OrderStatus(tx, r) => {
                let _ = tx.send(r);
            }
            RouterResp::OpenGtd(tx, r) => {
                let _ = tx.send(r);
            }
            RouterResp::TradesSince(tx, r) => {
                let _ = tx.send(r);
            }
        }
    }
}

/// Event variants of the virtual queue. Ordering at equal `ts_ms` is given by the
/// insertion `seq` — fully deterministic.
enum EvKind {
    /// Spot sample for the oracle.
    Spot(TapeSpot),
    /// The book ARRIVES at the simulated exchange: `tick_book` can fill resting GTDs
    /// and sets the state against which future submits will arrive.
    BookArrive(Box<TapeBook>),
    /// The agent OBSERVES the book (delay `obs_delay_ms`): `on_book` → intents.
    BookObserve(Box<TapeBook>),
    /// Strategy watchdog (`on_timer`), same cadence as the runner.
    Timer,
    /// Executor maintenance (`purge_and_resync`), every 30s.
    Maintenance,
    /// Global kill check (`global_kill_rx` arm of `run()`).
    KillCheck,
    /// Strategy intent entering the executor queue.
    Dispatch {
        intent: Box<OrderIntent>,
        enqueued_ms: i64,
    },
    /// Deferred venue call that arrives at its `ts_ms` (Box: the job carries
    /// oneshot+request — large variant, keep `Ev` compact). `ack_ms`
    /// is how long the RESPONSE takes to return to the agent after it is processed —
    /// `ack` stage of the latency profile (0 in nominal).
    Router { job: Box<RouterJob>, ack_ms: i64 },
    /// Deferred delivery of the venue response to the executor (`ack` stage
    /// — ver `RouterResp`).
    RouterResp(Box<RouterResp>),
    /// Official result of a market known at this instant (tape
    /// `ResearchEvent` → `crate::research::labels::apply_official` over the
    /// ephemeral `StateStore`; the label becomes available to `settle_due`
    /// when `known_at_ms <= now`).
    Resolution(Box<TapeResolution>),
    /// Quality mark from the tape (invalidated segment, gap) — it is only
    /// recorded; it does not touch economic state.
    Quality(crate::research::contract::QualityFlags, Option<String>),
}

struct Ev {
    ts_ms: i64,
    seq: u64,
    /// `local_seq` of the `ResearchEvent` that originated this event (tape
    /// causal). `None` for internal events (Dispatch, Timer, Router…) and in
    /// historical SQL tape.
    tape_seq: Option<u64>,
    kind: EvKind,
}

/// Min-heap by `(ts_ms, seq)` — `BinaryHeap` is a max-heap, so we invert.
impl PartialEq for Ev {
    fn eq(&self, o: &Self) -> bool {
        self.ts_ms == o.ts_ms && self.seq == o.seq
    }
}
impl Eq for Ev {}
impl PartialOrd for Ev {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ev {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        Reverse((self.ts_ms, self.seq)).cmp(&Reverse((o.ts_ms, o.seq)))
    }
}

/// Queue shared between the driver and the `DeferredSimRouter`: the router uses it
/// to schedule arrivals from the executor task.
struct SharedQueue {
    heap: Mutex<BinaryHeap<Ev>>,
    seq: AtomicU64,
    /// Virtual clock the driver sets on each event (ms epoch). It is the
    /// same value that `crate::time::now()` returns at that instant — it is
    /// duplicated as an atomic because the deferred router reads it from the task
    /// of the executor without going through the driver.
    now_ms: Arc<AtomicI64>,
    /// Router jobs scheduled but not yet processed by the driver — the
    /// quiescence counter (`settle`): an executor message that only
    /// waits on one of these jobs cannot advance without moving the clock.
    parked: AtomicUsize,
}

impl SharedQueue {
    fn new(now_ms: Arc<AtomicI64>) -> Self {
        Self {
            heap: Mutex::new(BinaryHeap::new()),
            seq: AtomicU64::new(0),
            now_ms,
            parked: AtomicUsize::new(0),
        }
    }
    fn push(&self, ts_ms: i64, kind: EvKind) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.heap.lock().push(Ev {
            ts_ms,
            seq,
            tape_seq: None,
            kind,
        });
    }
    /// Push referencing the `local_seq` of the originating `ResearchEvent` —
    /// keeps the event↔tape traceability for the `observation_seqs` of the
    /// episodic environment.
    fn push_tape(&self, ts_ms: i64, kind: EvKind, tape_seq: u64) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.heap.lock().push(Ev {
            ts_ms,
            seq,
            tape_seq: Some(tape_seq),
            kind,
        });
    }
    fn push_router(&self, ts_ms: i64, job: RouterJob, ack_ms: i64) {
        self.parked.fetch_add(1, Ordering::Relaxed);
        self.push(
            ts_ms,
            EvKind::Router {
                job: Box::new(job),
                ack_ms,
            },
        );
    }
    /// Venue response deferred by the `ack` stage: it counts the same in the
    /// quiescence counter — the executor waiting for it stays parked
    /// until the event matures.
    fn push_resp(&self, ts_ms: i64, resp: RouterResp) {
        self.parked.fetch_add(1, Ordering::Relaxed);
        self.push(ts_ms, EvKind::RouterResp(Box::new(resp)));
    }
    /// Is there a `BookObserve` for the same `cid` already due (`ts <= up_to`)?
    /// Replica of the runner's `drain_latest`: between two wakeups the
    /// channel may accumulate several books of the same market and only the LAST
    /// reaches `on_book` — the intermediate ones are coalesced (dropped as Lagged).
    fn has_pending_observe(&self, cid: &str, up_to_ts: i64) -> bool {
        self.heap.lock().iter().any(|e| {
            e.ts_ms <= up_to_ts
                && matches!(&e.kind, EvKind::BookObserve(b) if b.snap.market_condition_id == cid)
        })
    }
    fn pop(&self) -> Option<Ev> {
        let ev = self.heap.lock().pop();
        if let Some(ev) = &ev {
            if matches!(ev.kind, EvKind::Router { .. } | EvKind::RouterResp(_)) {
                self.parked.fetch_sub(1, Ordering::Relaxed);
            }
        }
        ev
    }
    /// Extracts ALL pending `Router` jobs (with their `ack_ms`), out of
    /// time order. Pending `RouterResp` responses are DELIVERED
    /// right here (they are a `send`, with nothing to process at the venue).
    /// Used by `restart_executor`: an executor parked on a `oneshot` of
    /// deferred job cannot finish if nobody moves the queue — its jobs must be
    /// resolved before waiting for its task. Decrements `parked`
    /// the same as `pop`.
    fn drain_router_jobs(&self) -> Vec<(RouterJob, i64)> {
        let mut heap = self.heap.lock();
        let mut jobs = Vec::new();
        let mut rest = Vec::with_capacity(heap.len());
        let mut drained = 0usize;
        while let Some(ev) = heap.pop() {
            match ev.kind {
                EvKind::Router { job, ack_ms } => {
                    drained += 1;
                    jobs.push((*job, ack_ms));
                }
                EvKind::RouterResp(resp) => {
                    drained += 1;
                    resp.deliver();
                }
                _ => rest.push(ev),
            }
        }
        for ev in rest {
            heap.push(ev);
        }
        if drained > 0 {
            self.parked.fetch_sub(drained, Ordering::Relaxed);
        }
        jobs
    }
    fn peek_ts(&self) -> Option<i64> {
        self.heap.lock().peek().map(|e| e.ts_ms)
    }
}

// ── Router with deferred arrival ───────────────────────────────────────────────

/// `OrderRouter` wrapping the `SimulatedRouter` (with `latency_ms=0`
/// internal) moving the latency to the virtual queue: each call arrives at
/// venue at `now + latency_ms` — during that gap the market can move
/// (other `tick_book`s), exactly like the sim's wall-clock `sleep` in
/// realtime, but without sleeping at all.
///
/// If the driver dies without processing the job (`rx` closed), the submit returns
/// `RouterResult::unknown` — a lost response is the conservative outcome
/// production uses for a POST with no response.
struct DeferredSimRouter {
    queue: Arc<SharedQueue>,
    /// Additional flat latency (`--latency-ms` CLI flag) — it is added to the
    /// profile stage if one is installed.
    latency_ms: i64,
    /// Per-stage profile `[research_sim.latencies]` — `Some` only when
    /// the runner installed a research profile (`--research-profile`);
    /// `None` = nominal, all the latency is a flat `latency_ms`.
    latencies: Option<LatencyProfile>,
    /// Replay seed: the stage jitter is derived from
    /// `(seed, draw_seq)` — same input, same draws.
    seed: u64,
    /// Router draw counter. The executor's calls are
    /// sequential (a single task), so the sampling order is
    /// deterministic. It is reset when each `run()` starts so that an engine
    /// reused reproduces the same run.
    draw_seq: AtomicU64,
}

/// Deterministic uniform draw in `[0,1)` — splitmix64 over `(seed, n)`.
/// Pure function shared by the deferred router and the driver (stages
/// `feed`/`decision`): the same `(seed, n)` always produces the same `u`.
fn draw_u(seed: u64, n: u64) -> f64 {
    let mut z = seed.wrapping_add(n).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

impl DeferredSimRouter {
    /// Jitter of a profile stage (0 if no profile is installed). The
    /// `load` is the number of router jobs in flight — an honest proxy for
    /// "events in the same batch" with the information available.
    fn sample_leg(&self, leg: LatencyLeg) -> i64 {
        let Some(l) = &self.latencies else { return 0 };
        let load = self.queue.parked.load(Ordering::Relaxed) as f64;
        l.sample(leg, load, self.draw_u())
    }
    fn draw_u(&self) -> f64 {
        let n = self.draw_seq.fetch_add(1, Ordering::Relaxed);
        draw_u(self.seed, n)
    }
    /// Virtual ts of arrival at the venue for this stage: `now + latency_ms`
    /// flat + sample of the profile stage (submit/cancel/query).
    fn arrival(&self, leg: LatencyLeg) -> i64 {
        self.queue.now_ms.load(Ordering::Relaxed) + self.latency_ms + self.sample_leg(leg)
    }
}

impl OrderRouter for DeferredSimRouter {
    fn submit(
        &self,
        req: OrderRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RouterResult> + Send + '_>> {
        let (tx, rx) = oneshot::channel();
        self.queue.push_router(
            self.arrival(LatencyLeg::Submit),
            RouterJob::Submit { req, tx },
            self.sample_leg(LatencyLeg::Ack),
        );
        Box::pin(async move {
            rx.await.unwrap_or_else(|_| {
                RouterResult::unknown("replay: venue response lost (driver closed)")
            })
        })
    }

    fn submit_pair(
        &self,
        req1: OrderRequest,
        req2: OrderRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = (RouterResult, RouterResult)> + Send + '_>,
    > {
        let (tx, rx) = oneshot::channel();
        self.queue.push_router(
            self.arrival(LatencyLeg::Submit),
            RouterJob::SubmitPair { req1, req2, tx },
            self.sample_leg(LatencyLeg::Ack),
        );
        Box::pin(async move {
            rx.await.unwrap_or_else(|_| {
                let u = || {
                    RouterResult::unknown("replay: venue response lost (driver closed)")
                };
                (u(), u())
            })
        })
    }

    fn cancel(
        &self,
        market_cid: &str,
        order_id: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + '_>> {
        let (tx, rx) = oneshot::channel();
        self.queue.push_router(
            self.arrival(LatencyLeg::Cancel),
            RouterJob::Cancel {
                cid: market_cid.to_string(),
                order_id: order_id.to_string(),
                tx,
            },
            self.sample_leg(LatencyLeg::Ack),
        );
        Box::pin(async move {
            rx.await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("replay: driver closed")))
        })
    }

    fn live_gtd_order_ids(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<HashSet<String>>> + Send + '_>>
    {
        let (tx, rx) = oneshot::channel();
        // Queries travel the same HTTP path as a submit.
        self.queue.push_router(
            self.arrival(LatencyLeg::Submit),
            RouterJob::LiveGtd { tx },
            self.sample_leg(LatencyLeg::Ack),
        );
        Box::pin(async move { rx.await.ok().flatten() })
    }

    fn order_status(
        &self,
        order_id: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<RouterOrderStatus>> + Send + '_>>
    {
        let (tx, rx) = oneshot::channel();
        self.queue.push_router(
            self.arrival(LatencyLeg::Submit),
            RouterJob::OrderStatus {
                order_id: order_id.to_string(),
                tx,
            },
            self.sample_leg(LatencyLeg::Ack),
        );
        Box::pin(async move { rx.await.ok().flatten() })
    }

    fn open_gtd_orders(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Option<Vec<RouterOpenOrder>>> + Send + '_>,
    > {
        let (tx, rx) = oneshot::channel();
        self.queue.push_router(
            self.arrival(LatencyLeg::Submit),
            RouterJob::OpenGtd { tx },
            self.sample_leg(LatencyLeg::Ack),
        );
        Box::pin(async move { rx.await.ok().flatten() })
    }

    fn trades_since(
        &self,
        after_ts_secs: i64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<Vec<RouterTrade>>> + Send + '_>>
    {
        let (tx, rx) = oneshot::channel();
        self.queue.push_router(
            self.arrival(LatencyLeg::Submit),
            RouterJob::TradesSince { after_ts_secs, tx },
            self.sample_leg(LatencyLeg::Ack),
        );
        Box::pin(async move { rx.await.ok().flatten() })
    }

    fn mode(&self) -> Mode {
        Mode::Paper
    }
}

// ── Executor task ─────────────────────────────────────────────────────────

/// Messages from the driver to the executor task (a channel separate from the
/// production `intent_rx`, which is unused in replay: intents arrive
/// as `ExecMsg::Intent` with the virtual emission instant; the queue wait is
/// measured inside the task when it picks up the message).
enum ExecMsg {
    Intent {
        intent: OrderIntent,
        /// Virtual instant at which the strategy emitted the intent — the wait
        /// in the queue is measured when the message is PICKED UP in the task (`now - enqueued`),
        /// so the time the executor spends parked on a deferred submit
        /// counts as real waiting and the `queue_stale` gate is
        /// reachable.
        enqueued_ms: i64,
    },
    Maintenance,
    KillCheck,
}

/// Executor task loop in replay: same batching as `run()`
/// (drain the queue on a Buy and prioritize Sell/Cancel, `sort_by_key`
/// stable), same staleness gate via `dispatch_replay` and the
/// same `handle` within the task. No wall-clock timers: Maintenance and KillCheck
/// are messages scheduled by the virtual queue.
async fn executor_task(
    mut exec: Executor,
    mut rx: mpsc::UnboundedReceiver<ExecMsg>,
    done: Arc<AtomicUsize>,
    clock_ms: Arc<AtomicI64>,
) {
    // Queue wait measured when PICKED UP (virtual clock − emission instant):
    // the time the executor spends parked on a deferred submit counts —
    // so an intent queued behind a slow submit can yield `queue_stale`
    // just as in production.
    let wait_of = |enqueued_ms: i64| {
        std::time::Duration::from_millis(
            clock_ms
                .load(Ordering::Relaxed)
                .saturating_sub(enqueued_ms)
                .max(0) as u64,
        )
    };
    let mut held: Option<ExecMsg> = None;
    loop {
        let msg = match held.take() {
            Some(m) => m,
            None => match rx.recv().await {
                Some(m) => m,
                None => break,
            },
        };
        match msg {
            ExecMsg::Intent {
                intent,
                enqueued_ms,
            } if matches!(intent.side, OrderSide::Buy) => {
                // Same drained as run(): on a Buy, pull the intents
                // contiguous queued intents and advance Sell/Cancel.
                let mut batch = vec![(intent, enqueued_ms)];
                while let Ok(m) = rx.try_recv() {
                    match m {
                        ExecMsg::Intent {
                            intent: i,
                            enqueued_ms: e,
                        } => batch.push((i, e)),
                        other => {
                            held = Some(other);
                            break;
                        }
                    }
                }
                batch.sort_by_key(|(i, _)| match i.side {
                    OrderSide::Sell { .. } | OrderSide::Cancel => 0u8,
                    OrderSide::Buy => 1u8,
                });
                for (p, e) in batch {
                    exec.dispatch_replay(p, wait_of(e)).await;
                    done.fetch_add(1, Ordering::Relaxed);
                }
            }
            ExecMsg::Intent {
                intent,
                enqueued_ms,
            } => {
                exec.dispatch_replay(intent, wait_of(enqueued_ms)).await;
                done.fetch_add(1, Ordering::Relaxed);
            }
            ExecMsg::Maintenance => {
                exec.purge_and_resync().await;
                done.fetch_add(1, Ordering::Relaxed);
            }
            ExecMsg::KillCheck => {
                exec.global_kill_check_replay().await;
                done.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

// ── Market metadata for settlement ──────────────────────────────────────

struct MarketMeta {
    yes_token: String,
    slug: String,
    clear_at: i64, // unix SECONDS (BookSnapshot convention)
    neg_risk: bool,
    reference_price: Option<Decimal>,
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Result of an engine run: pipeline counters + economic digest.
pub struct ReplayOutcome {
    /// Intents dispatched to the executor.
    pub intents: u64,
    /// Venue calls that reached execution in the sim.
    pub router_calls: u64,
    /// Buy/Sell fills that the broadcast delivered to the strategy (persisted).
    pub market_fills: u64,
    /// `GateRejected` signals delivered (executor/risk gates).
    pub gate_rejected: u64,
    /// `Rejected` signals delivered (router rejection via the executor).
    pub rejected_router: u64,
    /// `Cancelled` signals delivered (cancel acks).
    pub cancel_acks: u64,
    /// Positions settled by `apply_settlement` when crossing `clear_at`.
    pub settlements: u64,
    /// Pending Unknown entries left unresolved in the journal at the end.
    pub unresolved_unknowns: u64,
    /// GTDs live at the venue at the end of the run.
    pub live_gtds: u64,
    /// Fills in the exact order in which the broadcast delivered them to the strategy.
    pub fills_seen: Vec<Fill>,
    /// Executions recorded by the sim (exec ids deterministic per seed).
    pub executions: Vec<SimExecution>,
    pub wallet_final: Option<Decimal>,
    /// Canonical digest of the economic ledger (see `build_digest`).
    pub digest: String,
    /// Return↔ledger reconciliation computed at the end of the
    /// run — the pipeline's economic acceptance (`reconciled()` requires
    /// `deriva_caja == 0` and `deriva_pnl == 0`; it also goes into the digest).
    pub reconciliation: fill_sink::ReturnReconciliation,
    /// Official labels applied to the ephemeral StateStore from the
    /// tape (`apply_official` → `Confirmed`/`Corrected`).
    pub labels_applied: u64,
    /// Resolution corrections received AFTER settling the
    /// market — the episode is explicitly invalidated (never a
    /// second payout). `corrections > 0` ⇒ result not certifiable.
    pub settlement_corrections: u64,
    /// Cids that crossed `clear_at` without a confirmed label at the end of the
    /// run — pending, never settled by a silent proxy.
    pub settlement_pending: Vec<String>,
    /// `true` if the settlement used the diagnostic spot proxy
    /// (`settlement_proxy_diagnostic`) in any market — forces
    /// `fidelity = "reduced"` even though the tape is a `ResearchEvent`.
    pub settlement_diagnostic: bool,
    /// CAUSAL fidelity declaration of the run (integrity of the
    /// tape + settlement + reconciliation): `"certified"` only if the
    /// tape is `ResearchEvent` (recv/avail real), no proxy was used for
    /// settlement nor post-settle corrections, reconciliation closed
    /// exactly and the scope suffered no degradation; `"reduced"` in
    /// any other scenario, with the reasons in `fidelity_reasons`.
    pub fidelity: &'static str,
    /// Explicit reasons why the run ended as
    /// `"reduced"` — one per unmet condition (empty if
    /// `"certified"`). It is never certified silently.
    pub fidelity_reasons: Vec<String>,
    /// EXTERNAL execution fidelity (vs the real venue) —
    /// constant `"unidentified"`: execution fidelity is not
    /// identified and nothing else is ever asserted (it is not merged with the
    /// causal integrity of `fidelity`).
    pub external_fidelity: &'static str,
    /// Quality summary of the run with the metrics of
    /// scope (see `TapeQuality` — scope = market metadata observed in the
    /// window of the run).
    pub quality: crate::research::contract::QualitySummary,
    realized_pnl: Decimal,
    ledger: Arc<BracketLedger>,
}

impl ReplayOutcome {
    /// Directional net of the bracket in the ledger (source of the bracket cap).
    pub fn net_exposure(&self, cid: &str) -> Decimal {
        self.ledger.net_exposure(cid)
    }
    /// GTDs live at the end — same data as the venue's `live_gtd_order_ids`.
    pub fn final_live_gtds(&self) -> u64 {
        self.live_gtds
    }
    /// Persisted realized PnL (`state.realized_pnl_total`).
    pub fn realized_pnl(&self) -> Decimal {
        self.realized_pnl
    }
    /// Economic return reconcilable with the durable ledger — the
    /// real acceptance gate (not just the algorithm's reward).
    pub fn reconciled(&self) -> bool {
        self.reconciliation.reconciled()
    }
}

/// Mark of the virtual step processed by [`ReplayEngine::step_once`] —
/// the episodic environment distinguishes decision points (a `BookObserve`
/// that reached `on_book`) from the pipeline's internal progress.
#[derive(Debug, Clone)]
pub enum StepMark {
    /// Decision point: the agent observed a book of market `cid`.
    Decision {
        ts_ms: i64,
        /// Internal seq of the event in the virtual queue.
        seq: u64,
        /// `local_seq` of the originating ResearchEvent if the tape is causal.
        tape_seq: Option<u64>,
        cid: String,
    },
    /// Observable tape event that was not a decision point (spot,
    /// resolution, quality). `observable` distinguishes material the agent
    /// could have seen (spot) from label/quality infrastructure.
    Tape {
        ts_ms: i64,
        seq: u64,
        tape_seq: Option<u64>,
        observable: bool,
    },
    /// Internal advance: timer, maintenance, dispatch, router, kill-check,
    /// arrival at the venue. It is never an agent observation.
    Advance {
        ts_ms: i64,
        seq: u64,
        tape_seq: Option<u64>,
    },
}

// ── Engine ─────────────────────────────────────────────────────────────────────

/// Replay engine over the real pipeline. Reusable across runs over the SAME
/// logical tape (the clock and queue survive — useful for restart tests);
/// each `run` processes until the scheduled event queue is empty.
pub struct ReplayEngine {
    params: ReplayParams,
    state: Arc<StateStore>,
    sim: Arc<SimulatedRouter>,
    router: Arc<DeferredSimRouter>,
    risk: Arc<RiskGuard>,
    ledger: Arc<BracketLedger>,
    order_map: Arc<Mutex<HashMap<String, StrategyId>>>,
    registry: Arc<MarketRegistry>,
    queue: Arc<SharedQueue>,
    clock_ms: Arc<AtomicI64>,
    // Shared fills channel (executor + sim → fill_sink→broadcast).
    fill_rx: mpsc::UnboundedReceiver<Fill>,
    fill_persist_tx: mpsc::UnboundedSender<Fill>,
    bcast_tx: broadcast::Sender<Fill>,
    bcast_rx: broadcast::Receiver<Fill>,
    wallet_rx: watch::Receiver<Option<Decimal>>,
    // Executor task + quiescence counters.
    exec_tx: mpsc::UnboundedSender<ExecMsg>,
    exec_done: Arc<AtomicUsize>,
    exec_dispatched: usize,
    exec_task: Option<tokio::task::JoinHandle<()>>,
    known_order_ids: Arc<dashmap::DashSet<String>>,
    // Driver state that survives between runs.
    metas: BTreeMap<String, MarketMeta>,
    vols: HashMap<String, VolReplay>,
    /// Cids with COMPLETE settlement (both halves: `apply_settlement` in
    /// SQLite + `credit_redemption` in the sim).
    /// An Err in either half leaves the market pending for retry.
    settled: HashSet<String>,
    /// Positions whose SQLite half was already settled in THIS run —
    /// key (cid, token, strategy). Persisting per key avoids a double REDEEM
    /// if `settle_due` re-enters after a partial failure.
    settle_sqlite_ok: HashSet<(String, String, String)>,
    /// Positions with a settled SQLite half whose credit in the sim
    /// is still pending (or failed): key (cid, token, strategy) →
    /// (redeem_id, shares, payout). The payout only reaches `settled`
    /// when this half also confirms.
    settle_pending_sim: HashMap<(String, String, String), (String, Decimal, Decimal)>,
    /// Settlement counter per (cid, token,
    /// strategy) — the `redeem_id` carries the ordinal so that a
    /// legitimate re-settlement (post-settle reopening) is not swallowed by
    /// the sim's dedup, and a retry does not duplicate the payout.
    settle_seq: HashMap<(String, String, String), u64>,
    /// Winner settled per cid — a later correction compares against it
    /// to explicitly invalidate the episode.
    settled_winner: HashMap<String, bool>,
    /// Resolution corrections received after the settle.
    stats_corrections: u64,
    /// Labels applied via `apply_official` (Confirmed + Corrected).
    stats_labels: u64,
    /// Cids in `expired_cids` without a confirmed label (pending diagnostics).
    pending_settlement: HashSet<String>,
    /// `true` if any settle used the diagnostic spot proxy.
    proxy_used: bool,
    /// Source of the tape in progress — sets the outcome's `fidelity`.
    tape_source: TapeSource,
    /// Quality marks positioned in time (consumed in
    /// `step_once` for the episode's `DataQualityAbort` terminality).
    /// It is PER RUN: `seed_tape` empties it when seeding a new tape.
    quality_log: Vec<(i64, Option<String>, crate::research::contract::QualityFlags)>,
    /// Quality metrics of the seeded tape (scope = market metadata
    /// observed in the window); empty for a SQL tape.
    tape_stats: TapeQualityStats,
    /// Guard on the replay clock while the episode runs —
    /// in incremental mode (`step_once`) it cannot die when leaving a
    /// function scope: it lives in the engine between steps.
    replay_guard: Option<crate::time::ReplayClockGuard>,
    kill_signaled: bool,
    next_timer_ms: Option<i64>,
    next_maint_ms: Option<i64>,
    tape_end_ms: i64,
    // Accumulated outcome counters.
    stats_intents: u64,
    stats_router_calls: u64,
    stats_fills: u64,
    stats_gate: u64,
    stats_rejected: u64,
    stats_cancel_acks: u64,
    stats_settlements: u64,
    fills_seen: Vec<Fill>,
    /// Driver draw counter (`feed`/`decision` stages of the profile of
    /// latencies). It is reset when each `run()` starts — same tape + same
    /// seed ⇒ same draws, also in a reused engine.
    draw_seq: u64,
}

impl ReplayEngine {
    /// Builds the complete pipeline over `state` (MUST be ephemeral —
    /// `StateStore::open_memory`; the engine never opens the active DB).
    pub async fn new(params: ReplayParams, state: Arc<StateStore>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            params.obs_delay_ms >= 0 && params.latency_ms >= 0,
            "replay delays cannot be negative"
        );
        let clock_ms = Arc::new(AtomicI64::new(0));
        let queue = Arc::new(SharedQueue::new(clock_ms.clone()));
        let (fill_persist_tx, fill_rx) = mpsc::unbounded_channel::<Fill>();
        let (bcast_tx, bcast_rx) = broadcast::channel::<Fill>(4096);
        let (wallet_tx, wallet_rx) = watch::channel::<Option<Decimal>>(None);
        let registry = Arc::new(MarketRegistry::new());
        let clock: SimClock = {
            let c = clock_ms.clone();
            Arc::new(move || c.load(Ordering::Relaxed))
        };
        let sim = SimulatedRouter::with_deps(
            params.risk.paper_slippage_bps,
            params.risk.paper_fok_fill_rate,
            // The latency does NOT go through the sim (its sleep would be wall-clock): the
            // DeferredSimRouter moves it to the virtual queue.
            0,
            params.risk.paper_taker_fee_rate,
            params.risk.paper_pair_fill_correlation,
            fill_persist_tx.clone(),
            registry.clone(),
            params.initial_balance,
            wallet_tx.clone(),
            params.seed,
            clock,
        );
        // The research profile is only installed if the runner
        // explicitly requests it (`--research-profile`); in production the sim
        // stays nominal always.
        if let Some(r) = &params.research {
            sim.apply_research_profile(r.clone());
        }
        let router = Arc::new(DeferredSimRouter {
            queue: queue.clone(),
            latency_ms: params.latency_ms,
            // The `[research_sim.latencies]` stages only exist if the runner
            // installed a profile; in nominal mode everything is a flat `latency_ms`.
            latencies: params.research.as_ref().map(|r| r.latencies.clone()),
            seed: params.seed,
            draw_seq: AtomicU64::new(0),
        });
        let ledger = BracketLedger::new();
        let order_map = Arc::new(Mutex::new(HashMap::new()));
        let known_order_ids = Arc::new(dashmap::DashSet::new());
        let risk = RiskGuard::new(params.risk.clone(), Mode::Paper);
        risk.attach_persistence(state.clone());
        if let Ok(rows) = state.load_risk_state().await {
            risk.restore_persisted(rows);
        }
        let (exec_tx, exec_rx) = mpsc::unbounded_channel();
        let exec_done = Arc::new(AtomicUsize::new(0));
        let exec = Self::build_executor(
            &params,
            &state,
            &risk,
            &ledger,
            &order_map,
            &known_order_ids,
            router.clone(),
            fill_persist_tx.clone(),
            wallet_rx.clone(),
        )
        .await;
        let exec_task = tokio::spawn(executor_task(
            exec,
            exec_rx,
            exec_done.clone(),
            clock_ms.clone(),
        ));

        Ok(Self {
            params,
            state,
            sim,
            router,
            risk,
            ledger,
            order_map,
            registry,
            queue,
            clock_ms,
            fill_rx,
            fill_persist_tx,
            bcast_tx,
            bcast_rx,
            wallet_rx,
            exec_tx,
            exec_done,
            exec_dispatched: 0,
            exec_task: Some(exec_task),
            known_order_ids,
            metas: BTreeMap::new(),
            vols: HashMap::new(),
            settled: HashSet::new(),
            settle_sqlite_ok: HashSet::new(),
            settle_pending_sim: HashMap::new(),
            settle_seq: HashMap::new(),
            settled_winner: HashMap::new(),
            stats_corrections: 0,
            stats_labels: 0,
            pending_settlement: HashSet::new(),
            proxy_used: false,
            tape_source: TapeSource::Sql,
            quality_log: Vec::new(),
            tape_stats: TapeQualityStats::default(),
            replay_guard: None,
            kill_signaled: false,
            next_timer_ms: None,
            next_maint_ms: None,
            tape_end_ms: i64::MIN,
            stats_intents: 0,
            stats_router_calls: 0,
            stats_fills: 0,
            stats_gate: 0,
            stats_rejected: 0,
            stats_cancel_acks: 0,
            stats_settlements: 0,
            fills_seen: Vec::new(),
            draw_seq: 0,
        })
    }

    /// Instantiates the production `Executor` over the given state (without
    /// spawning the task — the caller launches it with `executor_task`, or does
    /// prior restores such as the GTD in `restart_executor`). The `intent_rx`
    /// production one is left without a sender (it is never polled in replay).
    #[allow(clippy::too_many_arguments)]
    async fn build_executor(
        params: &ReplayParams,
        state: &Arc<StateStore>,
        risk: &Arc<RiskGuard>,
        ledger: &Arc<BracketLedger>,
        order_map: &Arc<Mutex<HashMap<String, StrategyId>>>,
        known_order_ids: &Arc<dashmap::DashSet<String>>,
        router: Arc<DeferredSimRouter>,
        fill_persist_tx: mpsc::UnboundedSender<Fill>,
        wallet_rx: watch::Receiver<Option<Decimal>>,
    ) -> Executor {
        let (dead_tx, intent_rx) = mpsc::channel(1);
        drop(dead_tx);
        let mut exec = Executor::new(
            intent_rx,
            fill_persist_tx,
            router,
            risk.clone(),
            state.clone(),
            order_map.clone(),
            wallet_rx,
            known_order_ids.clone(),
            params.fak_strategies.clone(),
            ledger.clone(),
        );
        // As in main after building the Executor: seed the journal
        // durable of pending Unknowns (also models a clean restart).
        exec.seed_unknown_attempts_from_journal_pub().await;
        exec
    }

    /// Access to the simulator — to seed controlled chaos in tests
    /// (`set_unknown_rate`, `add_funds`, `executions`).
    pub fn sim(&self) -> &Arc<SimulatedRouter> {
        &self.sim
    }

    pub fn risk(&self) -> &Arc<RiskGuard> {
        &self.risk
    }

    pub fn state(&self) -> &Arc<StateStore> {
        &self.state
    }

    /// Replaces the parameters (e.g. another RiskConfig before a restart —
    /// the durable journal decides the blocker, not the process config).
    pub fn set_params(&mut self, params: ReplayParams) {
        self.params = params;
    }

    /// Restarts the executor over the SAME StateStore: closes the current task
    /// and builds a new `Executor` with a new `RiskGuard` (the config
    /// in effect in `self.params` — it may differ from the one that produced the
    /// durable data) seeding the Unknown journal.
    ///
    /// Closing semantics: dropping `exec_tx` makes the channel return the backlog
    /// before `None` — messages already queued are PROCESSED (ordered drain,
    /// not lost; what a real crash would lose is what was emitted
    /// afterwards). If the executor was left parked on a deferred venue job,
    /// the pending `Router` jobs run first at the current ts to
    /// release its `oneshot` — without this, `t.await` would hang because nobody else
    /// moves the queue. The task may park again while it drains the
    /// backlog (one msg may chain another router call), so the
    /// drain repeats until the task finishes (defensive cap:
    /// `MAX_SETTLE_YIELDS`).
    ///
    /// Precondition: call with the tape queue empty (between `run()`s) —
    /// pending market events are NOT processed here.
    pub async fn restart_executor(&mut self) -> anyhow::Result<()> {
        let (exec_tx, exec_rx) = mpsc::unbounded_channel();
        let old_tx = std::mem::replace(&mut self.exec_tx, exec_tx);
        drop(old_tx);
        if let Some(t) = self.exec_task.take() {
            let mut yields = 0usize;
            loop {
                for (job, ack_ms) in self.queue.drain_router_jobs() {
                    let now = self.queue.now_ms.load(Ordering::Relaxed);
                    self.run_router_job(now, job, ack_ms).await;
                }
                if t.is_finished() {
                    break;
                }
                tokio::task::yield_now().await;
                yields += 1;
                anyhow::ensure!(
                    yields < MAX_SETTLE_YIELDS,
                    "replay restart: the executor did not finish within {MAX_SETTLE_YIELDS} yields"
                );
            }
            let _ = t.await;
        }
        self.exec_done = Arc::new(AtomicUsize::new(0));
        self.exec_dispatched = 0;

        // New RiskGuard with the CURRENT config + persisted state (the
        // the durable state decides, not the config).
        let risk = RiskGuard::new(self.params.risk.clone(), Mode::Paper);
        risk.attach_persistence(self.state.clone());
        if let Ok(rows) = self.state.load_risk_state().await {
            risk.restore_persisted(rows);
        }
        self.risk = risk;

        // new known_order_ids (session) seeded from the durable index —
        // as in production startup (`order_ids_since`).
        let known = Arc::new(dashmap::DashSet::new());
        if let Ok(ids) = self.state.order_ids_since(0).await {
            for id in ids {
                known.insert(id);
            }
        }
        self.known_order_ids = known;

        let mut exec = Self::build_executor(
            &self.params,
            &self.state,
            &self.risk,
            &self.ledger,
            &self.order_map,
            &self.known_order_ids,
            self.router.clone(),
            self.fill_persist_tx.clone(),
            self.wallet_rx.clone(),
        )
        .await;

        // Restore live GTDs by querying the venue — like the startup reconciliation
        // in production: `open_gtd_orders` is the source of truth
        // for the destination; `active_gtd` is repopulated with claim = price×remaining
        // (same criterion as `refresh_gtd_claims`). `RouterOpenOrder` does not
        // carry `market_cid` or strategy: the cid is resolved by
        // `registry.token_to_cid` (populated in BookArrive) and the strategy by
        // durable `order_index`/`gtd_orders` — without a persisted owner, it uses
        // Arb (conservative: the claim stays in the bracket either way).
        if let Some(orders) = self.sim.open_gtd_orders().await {
            for o in orders {
                let cid = self
                    .registry
                    .token_to_cid
                    .get(&o.token_id)
                    .map(|r| r.clone());
                let Some(cid) = cid else {
                    tracing::warn!(
                        order_id = %o.order_id, token = %o.token_id,
                        "replay restart: live GTD without a cid in the registry — not restored"
                    );
                    continue;
                };
                let strat = match self.state.order_index_lookup(&o.order_id).await {
                    Ok(Some((s, _))) => s,
                    _ => self
                        .state
                        .get_gtd_strategy(&o.order_id)
                        .await
                        .ok()
                        .flatten()
                        .unwrap_or(StrategyId::Arb),
                };
                exec.restore_gtd(
                    &cid,
                    &o.token_id,
                    o.order_id,
                    o.price * o.remaining,
                    o.clear_at,
                    strat,
                )
                .await;
            }
        }

        let done = self.exec_done.clone();
        let task = tokio::spawn(executor_task(exec, exec_rx, done, self.clock_ms.clone()));
        self.exec_task = Some(task);
        Ok(())
    }

    // ── Quiescence ───────────────────────────────────────────────────────────

    /// Yields the runtime until the executor task is "idle": either all
    /// dispatched messages have finished, or the executor is parked
    /// waiting on a deferred venue job (`parked > 0` — only it records
    /// it, and they only advance once the driver moves the clock). Messages
    /// queued behind a parked one cannot run before it, so waiting for them here
    /// would be a deadlock. On a `current_thread` runtime this condition
    /// is exact.
    async fn settle(&mut self) -> anyhow::Result<()> {
        for _ in 0..MAX_SETTLE_YIELDS {
            let done = self.exec_done.load(Ordering::Relaxed);
            let parked = self.queue.parked.load(Ordering::Relaxed);
            if self.exec_dispatched == done || parked > 0 {
                return Ok(());
            }
            tokio::task::yield_now().await;
        }
        bail!(
            "replay: pipeline not quiescent after {} yields (dispatched={} done={} parked={})",
            MAX_SETTLE_YIELDS,
            self.exec_dispatched,
            self.exec_done.load(Ordering::Relaxed),
            self.queue.parked.load(Ordering::Relaxed)
        )
    }

    /// Drains the fill channel in emission order and processes them through the SAME
    /// production function (`accounting::fill_sink::process_fill` — resolution of
    /// owner, retries and broadcast included). `fill.ts` is re-stamped
    /// to the virtual clock: `user_ws::handle_trade` (out of scope) stamps it
    /// with wall-clock `Utc::now()`, and the economic ledger must depend on
    /// the tape, not on the machine's clock — it is the same invariant as
    /// `recv_instant`/`Instant::from_replay_ms` for books.
    async fn drain_fills(&mut self, strategy: &mut dyn Strategy) {
        while let Ok(mut fill) = self.fill_rx.try_recv() {
            fill.ts = crate::time::now();
            fill_sink::process_fill(
                &self.state,
                &self.risk,
                &self.ledger,
                &self.bcast_tx,
                &self.order_map,
                fill,
            )
            .await;
        }
        while let Ok(f) = self.bcast_rx.try_recv() {
            match f.side {
                FillSide::Buy | FillSide::Sell => self.stats_fills += 1,
                FillSide::GateRejected => self.stats_gate += 1,
                FillSide::Rejected => self.stats_rejected += 1,
                FillSide::Cancelled => self.stats_cancel_acks += 1,
            }
            self.fills_seen.push(f.clone());
            strategy.on_fill(&f);
        }
    }

    /// Settlement at `clear_at` (equivalent to the job of `src/settlement/orphan.rs` in
    /// production, applied by `state.apply_settlement`).
    ///
    /// Market verdict:
    /// - Certified path: official `confirmed` label queried via
    ///   `crate::research::labels::confirmed_label(state, cid, now_ms)` — un label
    ///   with `known_at_ms > now_ms` is INVISIBLE (the resolution arrived but
    /// is not yet known) and the market stays pending.
    /// - No label ⇒ pending, NEVER a silent proxy. The spot proxy
    ///   (`reference_price` vs last spot) only runs with
    ///   `params.settlement_proxy_diagnostic` and mark `proxy_used` →
    ///   `fidelity="reduced"` in the outcome.
    /// - Without evidence, the position stays open — it is never
    ///   invents the result.
    ///
    /// Two-phase settlement with a progress marker per
    /// `(cid, token, strategy)`:
    /// the SQLite half (`apply_settlement_detalle`, transactional and
    /// idempotent per position) and the sim half (`credit_redemption`,
    /// idempotent per deterministic `redeem_id` with ordinal). A cid only
    /// enters `settled` when BOTH halves of ALL its positions
    /// have been confirmed; an `Err` leaves the market pending for the next
    /// sweep without a double payout.
    async fn settle_due(&mut self, now_ms: i64) {
        // 1) Winner per expired cid: official label first; proxy only
        //    under a diagnostic flag.
        let mut expired_cids: Vec<String> = Vec::new();
        for (cid, meta) in &self.metas {
            if self.settled.contains(cid) || now_ms <= meta.clear_at * 1000 {
                continue;
            }
            expired_cids.push(cid.clone());
        }
        for cid in expired_cids {
            let meta = match self.metas.get(&cid) {
                Some(m) => m,
                None => continue,
            };
            let meta_slug = meta.slug.clone();
            let meta_ref = meta.reference_price;
            let label = match crate::research::labels::confirmed_label(
                &self.state,
                &cid,
                Some(now_ms),
            )
            .await
            {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!(cid, "replay: confirmed_label failed: {e}");
                    self.pending_settlement.insert(cid.clone());
                    continue;
                }
            };
            let (yes_wins, winner_label, via_proxy) = match label {
                Some(l) => (
                    l.winner == crate::research::contract::OutcomeSide::Yes,
                    crate::research::labels::side_tag(l.winner).to_string(),
                    false,
                ),
                None if self.params.settlement_proxy_diagnostic => {
                    let asset = crate::oracle::asset_from_slug(&meta_slug);
                    let spot = asset
                        .and_then(|a| self.vols.get(&a))
                        .map(|v| v.last_price)
                        .filter(|p| *p > Decimal::ZERO);
                    match (spot, meta_ref) {
                        (Some(s), Some(strike)) if strike > Decimal::ZERO => {
                            self.proxy_used = true;
                            tracing::warn!(
                                cid,
                                "replay: settlement by spot PROXY (diagnostic, \
                                 fidelity=reduced) — no official label"
                            );
                            (
                                s > strike,
                                if s > strike { "YES" } else { "NO" }.to_string(),
                                true,
                            )
                        }
                        _ => {
                            self.pending_settlement.insert(cid.clone());
                            continue;
                        }
                    }
                }
                None => {
                    // No confirmed label: pending — the clock
                    // settlement advances but the result is not invented.
                    self.pending_settlement.insert(cid.clone());
                    continue;
                }
            };
            let _ = via_proxy; // already marked in proxy_used

            // 2) SQLite half per open position of the cid (idempotent).
            let open = match self.state.list_open_positions().await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("replay: list_open_positions failed in settle: {e}");
                    continue;
                }
            };
            let mut all_ok = true;
            for pos in open.into_iter().filter(|p| p.market_cid == cid) {
                let key = (cid.clone(), pos.token_id.clone(), pos.strategy.clone());
                if self.settle_sqlite_ok.contains(&key) {
                    continue;
                }
                let won = (pos.token_id == self.metas[&cid].yes_token) == yes_wins;
                let strat = StrategyId::from_db_str(&pos.strategy);
                match self
                    .state
                    .apply_settlement_detail(
                        &pos,
                        won,
                        &winner_label,
                        Decimal::ZERO,
                        Decimal::ZERO,
                        strat,
                        None,
                    )
                    .await
                {
                    Ok(Some(ap)) => {
                        self.stats_settlements += 1;
                        self.settle_sqlite_ok.insert(key.clone());
                        // The ledger already settled the REDEEM; the
                        // CASH and INVENTORY of the sim belong to the router — without
                        // this credit, `wallet_final` ends up short by
                        // Σ payouts, and inventory kept shares already
                        // settled. The values are the FRESH ones that
                        // apply_settlement read within its tx.
                        //
                        // The per-key ordinal distinguishes a
                        // real re-settlement (post-settle reopening) from the
                        // retry of the same — dedup without swallowing payouts.
                        let n = self.settle_seq.entry(key.clone()).or_insert(0);
                        let redeem_id =
                            format!("settle:{cid}:{}:{}:{n}", pos.token_id, pos.strategy);
                        *n += 1;
                        self.settle_pending_sim
                            .insert(key, (redeem_id, ap.size, ap.payout));
                    }
                    Ok(None) => {
                        // Already settled in an earlier pass (internal race
                        // of apply_settlement): the sim half, if it was left
                        // pending, still in `settle_pending_sim` and is
                        // retried below — the dedup by redeem_id is the
                        // second barrier.
                        self.settle_sqlite_ok.insert(key);
                    }
                    Err(e) => {
                        tracing::warn!(cid, "replay: apply_settlement failed: {e}");
                        all_ok = false;
                    }
                }
            }
            // 3) Sim half: retry the pending credits of this cid.
            let pend_keys: Vec<(String, String, String)> = self
                .settle_pending_sim
                .keys()
                .filter(|k| k.0 == cid)
                .cloned()
                .collect();
            for key in pend_keys {
                let (redeem_id, size, payout) = match self.settle_pending_sim.get(&key) {
                    Some(v) => v.clone(),
                    None => continue,
                };
                match self.sim.credit_redemption(&redeem_id, &key.1, size, payout) {
                    Ok(_) => {
                        self.settle_pending_sim.remove(&key);
                    }
                    Err(e) => {
                        tracing::warn!(cid, redeem_id, "replay: credit_redemption failed: {e}");
                        all_ok = false;
                    }
                }
            }
            // 4) The cid is only marked as settled once both halves are closed:
            //    no pending sim credits and no open positions (remaining).
            let sim_remaining = self.settle_pending_sim.keys().any(|k| k.0 == cid);
            if all_ok && !sim_remaining {
                self.settled.insert(cid.clone());
                self.settled_winner.insert(cid.clone(), yes_wins);
                self.pending_settlement.remove(&cid);
            } else {
                self.pending_settlement.insert(cid);
            }
        }
    }

    /// Applies an official resolution from the tape to the `StateStore`
    /// ephemeral via `crate::research::labels::apply_official`. A correction
    /// (`Corrected`) over a market ALREADY settled invalidates the episode
    /// explicitly — `stats_corrections` — and NEVER produces a second payout
    /// (the position is already at 0 and `settled` blocks re-settlement).
    async fn apply_resolution(&mut self, res: &TapeResolution) {
        match crate::research::labels::apply_official(
            &self.state,
            &res.cid,
            res.source,
            &res.outcome,
            None,
            None,
        )
        .await
        {
            Ok(crate::research::labels::ReconcileOutcome::Confirmed { .. }) => {
                self.stats_labels += 1;
            }
            Ok(crate::research::labels::ReconcileOutcome::Corrected { previous, new }) => {
                self.stats_labels += 1;
                let post_settle =
                    self.settled.contains(&res.cid) || self.settled_winner.contains_key(&res.cid);
                if post_settle {
                    self.stats_corrections += 1;
                    tracing::warn!(
                        cid = %res.cid,
                        previous = ?previous,
                        new = ?new,
                        "replay: resolution correction AFTER settlement — episode invalidated (no second payout)"
                    );
                }
            }
            Ok(crate::research::labels::ReconcileOutcome::Pending) => {}
            Err(e) => {
                tracing::warn!(cid = %res.cid, "replay: apply_official failed: {e}");
            }
        }
    }

    /// Dispatches the intents of a strategy decision to the queue of the
    /// executor as `Dispatch` events at this same `ts_ms` (the mpsc queue of
    /// production is ~immediate; the measured wait will be `now - enqueued`).
    fn emit_intents(&mut self, ts_ms: i64, intents: Vec<OrderIntent>) {
        for intent in intents {
            self.queue.push(
                ts_ms,
                EvKind::Dispatch {
                    intent: Box::new(intent),
                    enqueued_ms: ts_ms,
                },
            );
        }
    }

    /// Deterministic sample of a latency profile stage from the
    /// driver (`feed`/`decision` stages — the router ones are sampled by the
    /// `DeferredSimRouter`). `load` = events in the same batch; 0 if there is no
    /// research profile installed.
    fn sample_leg(&mut self, leg: LatencyLeg, load: f64) -> i64 {
        let Some(r) = &self.params.research else {
            return 0;
        };
        self.draw_seq += 1;
        r.latencies
            .sample(leg, load, draw_u(self.params.seed, self.draw_seq))
    }

    /// Delivers the venue response to the executor: with `ack_ms > 0` it goes through
    /// the virtual queue (`ack` stage — the executor stays parked until
    /// the event matures); if it is already due, it is delivered inline.
    fn deliver_resp(&mut self, resp_ts: i64, resp: RouterResp) {
        if resp_ts <= self.queue.now_ms.load(Ordering::Relaxed) {
            resp.deliver();
        } else {
            self.queue.push_resp(resp_ts, resp);
        }
    }

    /// Processes a router job arriving at its `ts_ms`: runs the call
    /// REAL against the sim (which already sees all books with `recv <= ts`) and
    /// schedules the delivery of the response to the executor at `ts_ms + ack_ms`
    /// (immediate delivery if `ack_ms <= 0`, as in nominal).
    async fn run_router_job(&mut self, ts_ms: i64, job: RouterJob, ack_ms: i64) {
        self.stats_router_calls += 1;
        let resp_ts = ts_ms + ack_ms;
        match job {
            RouterJob::Submit { req, tx } => {
                let r = self.sim.submit(req).await;
                self.deliver_resp(resp_ts, RouterResp::Submit(tx, r));
            }
            RouterJob::SubmitPair { req1, req2, tx } => {
                let r = self.sim.submit_pair(req1, req2).await;
                self.deliver_resp(resp_ts, RouterResp::SubmitPair(tx, r));
            }
            RouterJob::Cancel { cid, order_id, tx } => {
                let r = self.sim.cancel(&cid, &order_id).await;
                self.deliver_resp(resp_ts, RouterResp::Cancel(tx, r));
            }
            RouterJob::LiveGtd { tx } => {
                let r = self.sim.live_gtd_order_ids().await;
                self.deliver_resp(resp_ts, RouterResp::LiveGtd(tx, r));
            }
            RouterJob::OrderStatus { order_id, tx } => {
                let r = self.sim.order_status(&order_id).await;
                self.deliver_resp(resp_ts, RouterResp::OrderStatus(tx, r));
            }
            RouterJob::OpenGtd { tx } => {
                let r = self.sim.open_gtd_orders().await;
                self.deliver_resp(resp_ts, RouterResp::OpenGtd(tx, r));
            }
            RouterJob::TradesSince { after_ts_secs, tx } => {
                let r = self.sim.trades_since(after_ts_secs).await;
                self.deliver_resp(resp_ts, RouterResp::TradesSince(tx, r));
            }
        }
    }

    /// Seeds the tape into the virtual queue and arms the replay clock at the
    /// first ts. Returns control to the caller: progress is driven with
    /// [`Self::step_once`] (incremental mode, episodic environment) or
    /// [`Self::run`] (full run — same loop reused).
    ///
    /// Seeding semantics:
    /// - Venue arrival (`BookArrive`) at `recv_ms`; agent observation
    ///   (`BookObserve`) at `avail_ms` if the tape carries it (causal
    ///   `ResearchEvent`) or at `recv_ms + obs_delay_ms + stage feed` if not.
    /// - The OBSERVABLE stream (BookObserve/Spot/Resolution/Quality) is
    ///   seeded in canonical order `(avail_ts_ms, local_seq)` — at equal
    ///   `ts_ms` the order is decided by the tape's `local_seq`, never by the
    ///   `origin_ts_ms` (remote, SQL tape: arrival order of the vectors,
    ///   books before spots, as always).
    /// - Per-run determinism: the jitter draws are reset on each
    ///   seed — a reused engine over the same tape reproduces the ledger.
    pub fn seed_tape(&mut self, mut tape: Tape) -> anyhow::Result<()> {
        self.draw_seq = 0;
        self.router.draw_seq.store(0, Ordering::Relaxed);
        self.tape_source = tape.source;
        // Quality metrics of the tape + per-run mark log
        // (the run outcome only declares what this tape processed).
        self.tape_stats = std::mem::take(&mut tape.quality_stats);
        self.quality_log.clear();
        // `feed` stage load: how many books arrive in the same ms
        // (same WS batch — with a 1s tape this is the honest approximation).
        // On a causal tape (real `avail_ms`) the stage does NOT apply — the delay
        // is already measured in the event.
        let mut feed_batch: HashMap<i64, u32> = HashMap::new();
        if self.params.research.is_some() {
            for b in &tape.books {
                if b.avail_ms.is_none() {
                    *feed_batch.entry(b.recv_ms).or_default() += 1;
                }
            }
        }
        let mut tape_end = self.tape_end_ms;
        // Venue side: arrivals sorted by (recv_ms, tape local_seq).
        let mut arrives: Vec<(i64, u64, TapeBook)> = Vec::with_capacity(tape.books.len());
        for (i, b) in tape.books.into_iter().enumerate() {
            arrives.push((b.recv_ms, b.tape_seq.unwrap_or(i as u64), b));
        }
        arrives.sort_by_key(|(ts, seq, _)| (*ts, *seq));
        // Agent side: the observable stream ordered by (avail/observe,
        // local_seq) — the canonical key of the causal tape. The
        // events without `tape_seq` (SQL) keep their vector order after
        // those with seq at the same instant (rank = idx + u64::MAX/2 — never
        // collides with a real local_seq of the same mixed tape).
        struct PendObs {
            ts: i64,
            rank: u64,
            kind: EvKind,
            tape_seq: Option<u64>,
        }
        let mut obs_events: Vec<PendObs> = Vec::new();
        let mut synth_rank = u64::MAX / 2;
        for (_, _, b) in arrives.iter() {
            let observe_ts = match b.avail_ms {
                Some(avail) => avail,
                None => {
                    // `feed` leg of the profile: venue→agent propagation — it is
                    // ADDED to `obs_delay_ms` (the age the stale-book gate sees
                    // includes it).
                    let feed = self.sample_leg(
                        LatencyLeg::Feed,
                        feed_batch.get(&b.recv_ms).copied().unwrap_or(0) as f64,
                    );
                    b.recv_ms + self.params.obs_delay_ms + feed
                }
            };
            tape_end = tape_end.max(observe_ts);
            let rank = b.tape_seq.unwrap_or_else(|| {
                synth_rank += 1;
                synth_rank
            });
            obs_events.push(PendObs {
                ts: observe_ts,
                rank,
                kind: EvKind::BookObserve(Box::new(b.clone())),
                tape_seq: b.tape_seq,
            });
        }
        for s in tape.spots {
            tape_end = tape_end.max(s.ts_ms);
            let rank = s.tape_seq.unwrap_or_else(|| {
                synth_rank += 1;
                synth_rank
            });
            obs_events.push(PendObs {
                ts: s.ts_ms,
                rank,
                kind: EvKind::Spot(s.clone()),
                tape_seq: s.tape_seq,
            });
        }
        for r in tape.resolutions {
            tape_end = tape_end.max(r.avail_ms);
            let tape_seq = r.tape_seq;
            let rank = tape_seq.unwrap_or_else(|| {
                synth_rank += 1;
                synth_rank
            });
            obs_events.push(PendObs {
                ts: r.avail_ms,
                rank,
                kind: EvKind::Resolution(Box::new(r)),
                tape_seq,
            });
        }
        for q in tape.quality {
            tape_end = tape_end.max(q.ts_ms);
            let rank = q.tape_seq.unwrap_or_else(|| {
                synth_rank += 1;
                synth_rank
            });
            obs_events.push(PendObs {
                ts: q.ts_ms,
                rank,
                kind: EvKind::Quality(q.flag, q.cid),
                tape_seq: q.tape_seq,
            });
        }
        // Canonical order of the observable stream: (ts, rank) with rank =
        // tape local_seq when it exists.
        obs_events.sort_by_key(|p| (p.ts, p.rank));
        for (ts, _, b) in arrives {
            match b.tape_seq {
                Some(s) => self.queue.push_tape(ts, EvKind::BookArrive(Box::new(b)), s),
                None => self.queue.push(ts, EvKind::BookArrive(Box::new(b))),
            }
        }
        for p in obs_events {
            match p.tape_seq {
                Some(s) => self.queue.push_tape(p.ts, p.kind, s),
                None => self.queue.push(p.ts, p.kind),
            }
        }
        self.tape_end_ms = tape_end;

        // Periodic cadences: the first production maintenance tick
        // is immediate (tokio::interval); the timer one follows the cadence of the
        // runner. They reschedule themselves until tape_end.
        let first_ts = self
            .queue
            .peek_ts()
            .context("empty tape: nothing to replay")?;
        // The clock guard lives in the engine — in incremental mode the
        // episode spans several `step_once` calls and the clock cannot be disarmed
        // between steps. It is re-armed if the tape changed base.
        self.replay_guard = Some(crate::time::ReplayClockGuard::new(first_ts));
        if self.next_timer_ms.is_none() {
            self.next_timer_ms = Some((first_ts / self.params.timer_ms + 1) * self.params.timer_ms);
        }
        if self.next_maint_ms.is_none() {
            self.next_maint_ms = Some(first_ts);
        }
        if let Some(t) = self.next_timer_ms {
            if t <= tape_end {
                self.queue.push(t, EvKind::Timer);
            }
        }
        if let Some(m) = self.next_maint_ms {
            if m <= tape_end {
                self.queue.push(m, EvKind::Maintenance);
            }
        }
        Ok(())
    }

    /// Advances ONE event of the virtual queue. Returns `Some(StepMark)` with the
    /// step mark — `Decision` when a `BookObserve` reached
    /// `on_book` (decision point of the episodic environment) — or `None`
    /// when the queue is empty.
    ///
    /// After each event the run's usual close-out runs: quiescence of the
    /// executor, `drain_fills` through `fill_sink`, kill-check and `settle_due`.
    /// It is the SAME loop as `run()` — the incremental environment reuses the
    /// real pipeline event by event, without rebuilding state or re-running.
    pub async fn step_once(
        &mut self,
        strategy: &mut dyn Strategy,
    ) -> anyhow::Result<Option<StepMark>> {
        let Some(ev) = self.queue.pop() else {
            return Ok(None);
        };
        let ts = ev.ts_ms;
        let tape_seq = ev.tape_seq;
        self.clock_ms.store(ts, Ordering::Relaxed);
        crate::time::set_replay_now_ms(ts);

        let mut mark = StepMark::Advance {
            ts_ms: ts,
            seq: ev.seq,
            tape_seq,
        };
        match ev.kind {
            EvKind::Spot(s) => {
                // Same key as `oracle::asset_from_slug` (lowercase) —
                // the tape may carry "BTC" or "btc".
                self.vols
                    .entry(s.asset.to_lowercase())
                    .or_default()
                    .push("tape", s.ts_ms, s.price);
                mark = StepMark::Tape {
                    ts_ms: ts,
                    seq: ev.seq,
                    tape_seq,
                    observable: true,
                };
            }
            EvKind::BookArrive(b) => {
                let snap = &b.snap;
                self.registry.remember_clear_at(
                    &snap.market_condition_id,
                    snap.neg_risk,
                    snap.clear_at,
                    &snap.yes.token_id,
                    &snap.no.token_id,
                );
                // The registry records token→cid just like the feed does when
                // receiving the book (used by the GTD restore and the
                // metadata lookups).
                self.registry
                    .token_to_cid
                    .insert(snap.yes.token_id.clone(), snap.market_condition_id.clone());
                self.registry
                    .token_to_cid
                    .insert(snap.no.token_id.clone(), snap.market_condition_id.clone());
                self.metas
                    .entry(snap.market_condition_id.clone())
                    .or_insert_with(|| MarketMeta {
                        yes_token: snap.yes.token_id.clone(),
                        slug: snap.slug.clone(),
                        clear_at: snap.clear_at,
                        neg_risk: snap.neg_risk,
                        reference_price: snap.reference_price,
                    });
                self.sim.tick_book(snap);
            }
            EvKind::BookObserve(b) => {
                // Replica of the runner's gates (`strategy::run`), in the
                // same order and with the same semantics — see the header doc.
                //
                // NOT replicated (infrastructure absent from the tape):
                // `universe_info`/`has_oracle` filter of the registry — the
                // tape does not record universe_info; the recorded books
                // reached the runner in production by definition.
                let mut snap = b.snap.clone();
                snap.recv_instant = Some(crate::time::Instant::from_replay_ms(b.recv_ms));

                // (a) Coalescing latest-per-cid: if another observation of the
                //     same market is already due in the queue, this book
                //     is dropped — it is the runner's `drain_latest`
                //     only the last book of the batch reaches
                //     `on_book`.
                if !self
                    .queue
                    .has_pending_observe(&snap.market_condition_id, ts)
                {
                    // (b) Kill-wait: strategy killed does not process books
                    //     (production waits for the reset; the intermediate books
                    //     are coalesced by drain_latest → same effect).
                    if !(self.risk.is_strategy_killed(strategy.id())
                        && !self.risk.paper_disable_kill_switch_active())
                    {
                        // (c) entry_block due to data quality — identical to the
                        //     runner: recv missing/future/age > max_book_age.
                        let entry_block = match snap.recv_instant {
                            None => Some("missing_receive_time"),
                            Some(recv) => {
                                match crate::time::Instant::now().checked_duration_since(recv) {
                                    None => Some("future_receive_time"),
                                    Some(age)
                                        if age
                                            > std::time::Duration::from_millis(
                                                self.params.max_book_age_ms.max(0) as u64,
                                            ) =>
                                    {
                                        Some("stale_book")
                                    }
                                    _ => None,
                                }
                            }
                        };
                        // (d) Runner vol gate: slug without asset → skip;
                        //     empty mapping → VolSnapshot::default; populated mapping
                        //     without the asset → skip (no cross-asset fallback).
                        if let Some(asset) = crate::oracle::asset_from_slug(&snap.slug) {
                            let vol = match self.vols.get(&asset) {
                                Some(v) => Some(v.snapshot(&asset)),
                                None if self.vols.is_empty() => Some(VolSnapshot::default()),
                                None => None,
                            };
                            if let Some(vol) = vol {
                                let intents = strategy.on_book(&snap, &vol);
                                mark = StepMark::Decision {
                                    ts_ms: ts,
                                    seq: ev.seq,
                                    tape_seq,
                                    cid: snap.market_condition_id.clone(),
                                };
                                // (e) Buy intents against a degraded book are
                                //     rejected LOCALLY with GateRejected to the
                                //     strategy — `reject_degraded_entry` of the
                                //     runner, without persisting.
                                let n_intents = intents.len() as f64;
                                for intent in intents {
                                    if let Some(reason) = entry_block {
                                        if crate::strategy::intent_contains_buy(&intent) {
                                            crate::strategy::reject_degraded_entry(
                                                strategy, &intent, reason,
                                            );
                                            continue;
                                        }
                                    }
                                    // `decision` leg: compute time of the
                                    // strategy between observing the book and the
                                    // intent entering the executor queue (0 in
                                    // nominal).
                                    let decision = self.sample_leg(LatencyLeg::Decision, n_intents);
                                    self.queue.push(
                                        ts + decision,
                                        EvKind::Dispatch {
                                            intent: Box::new(intent),
                                            enqueued_ms: ts + decision,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
            EvKind::Timer => {
                let next = ts + self.params.timer_ms;
                self.next_timer_ms = Some(next);
                if next <= self.tape_end_ms {
                    self.queue.push(next, EvKind::Timer);
                }
                let intents = strategy.on_timer();
                self.emit_intents(ts, intents);
            }
            EvKind::Maintenance => {
                let next = ts + self.params.maintenance_ms;
                self.next_maint_ms = Some(next);
                if next <= self.tape_end_ms {
                    self.queue.push(next, EvKind::Maintenance);
                }
                self.exec_tx
                    .send(ExecMsg::Maintenance)
                    .context("executor task dead")?;
                self.exec_dispatched += 1;
            }
            EvKind::KillCheck => {
                self.exec_tx
                    .send(ExecMsg::KillCheck)
                    .context("executor task dead")?;
                self.exec_dispatched += 1;
            }
            EvKind::Dispatch {
                intent,
                enqueued_ms,
            } => {
                self.stats_intents += 1;
                self.exec_tx
                    .send(ExecMsg::Intent {
                        intent: *intent,
                        enqueued_ms,
                    })
                    .context("executor task dead")?;
                self.exec_dispatched += 1;
            }
            EvKind::Router { job, ack_ms } => {
                self.run_router_job(ts, *job, ack_ms).await;
            }
            EvKind::RouterResp(resp) => resp.deliver(),
            EvKind::Resolution(res) => {
                self.apply_resolution(&res).await;
                mark = StepMark::Tape {
                    ts_ms: ts,
                    seq: ev.seq,
                    tape_seq,
                    observable: false, // the label is never a feature
                };
            }
            EvKind::Quality(flag, cid) => {
                self.quality_log.push((ts, cid, flag));
            }
        }

        // Quiescence: the executor either finishes its real job or stays
        // parked on deferred jobs the clock has not yet reached.
        self.settle().await?;
        // Fills emitted during this event (sim or executor) → fill_sink
        // production → broadcast → strategy.
        self.drain_fills(strategy).await;
        // Global kill: if something triggered it (e.g. a persistence failure or
        // a test via `risk()`), schedules the check — same effect
        // as the `run()` watch.
        if self.risk.is_killed() && !self.kill_signaled {
            self.kill_signaled = true;
            self.queue.push(ts, EvKind::KillCheck);
        } else if !self.risk.is_killed() {
            self.kill_signaled = false;
        }
        // Settlement of markets closed with evidence.
        self.settle_due(ts).await;
        Ok(Some(mark))
    }

    /// Run close-out after emptying the queue: final settlement sweep at the
    /// last observed ts, residual drain and outcome with the digest
    /// economic. Used by `run()` and by the episodic environment's close-out.
    pub async fn finalize(&mut self, strategy: &mut dyn Strategy) -> anyhow::Result<ReplayOutcome> {
        // Final sweep to the last observed ts (without looking further).
        let last_ts = self.clock_ms.load(Ordering::Relaxed);
        self.settle_due(last_ts).await;
        self.settle().await?;
        self.drain_fills(strategy).await;
        self.replay_guard = None; // disarms the virtual replay clock
        Ok(self.outcome().await)
    }

    /// Runs the tape until the virtual queue is empty. Returns the outcome
    /// with the pipeline counters and the economic ledger digest.
    /// It is the full-run wrapper over `seed_tape`+`step_once`+
    /// `finalize` — the same loop that drives the incremental environment.
    pub async fn run(
        &mut self,
        tape: Tape,
        strategy: &mut dyn Strategy,
    ) -> anyhow::Result<ReplayOutcome> {
        self.seed_tape(tape)?;
        while self.step_once(strategy).await?.is_some() {}
        self.finalize(strategy).await
    }

    /// Orderly engine shutdown: drops the executor task's channel
    /// (drains the queued backlog, like `restart_executor`) and waits
    /// for it to finish. Essential between environment episodes: the task
    /// would stay alive forever waiting on `rx.recv()`.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        let (exec_tx, _rx) = mpsc::unbounded_channel();
        let old_tx = std::mem::replace(&mut self.exec_tx, exec_tx);
        drop(old_tx);
        if let Some(t) = self.exec_task.take() {
            let mut yields = 0usize;
            loop {
                for (job, ack_ms) in self.queue.drain_router_jobs() {
                    let now = self.queue.now_ms.load(Ordering::Relaxed);
                    self.run_router_job(now, job, ack_ms).await;
                }
                if t.is_finished() {
                    break;
                }
                tokio::task::yield_now().await;
                yields += 1;
                anyhow::ensure!(
                    yields < MAX_SETTLE_YIELDS,
                    "replay shutdown: the executor did not finish after {MAX_SETTLE_YIELDS} yields"
                );
            }
            let _ = t.await;
        }
        self.replay_guard = None;
        Ok(())
    }

    /// Current virtual clock (last processed ts; the tape's `first_ts`
    /// if no event has been advanced yet).
    pub fn clock_now_ms(&self) -> i64 {
        self.clock_ms.load(Ordering::Relaxed)
    }

    /// Time end of the seeded tape (maximum ts of scheduled events).
    pub fn tape_end_ms(&self) -> i64 {
        self.tape_end_ms
    }

    /// `true` if the virtual queue is empty (no event left to process).
    pub fn queue_is_empty(&self) -> bool {
        self.queue.peek_ts().is_none()
    }

    /// `true` if the market completed settlement (both halves).
    pub fn market_settled(&self, cid: &str) -> bool {
        self.settled.contains(cid)
    }

    /// Post-settle resolution corrections seen in the run.
    pub fn settlement_corrections(&self) -> u64 {
        self.stats_corrections
    }

    /// Cids in expired without a confirmed label (pending settlement).
    pub fn settlement_pending(&self) -> Vec<String> {
        let mut v: Vec<String> = self.pending_settlement.iter().cloned().collect();
        v.sort();
        v
    }

    /// Quality marks registered until now — incremental consumption:
    /// returns the entries with `ts > after_ms` (for the step cut).
    pub fn quality_since(
        &self,
        after_ms: i64,
    ) -> Vec<(i64, Option<String>, crate::research::contract::QualityFlags)> {
        self.quality_log
            .iter()
            .filter(|(ts, ..)| *ts > after_ms)
            .cloned()
            .collect()
    }

    /// Complete quality log for drain by index (the environment keeps
    /// `quality_seen` and consumes `log[seen..]` — two marks with the same
    /// `ts_ms` are never lost, an edge scenario that `quality_since` by
    /// timestamp cannot distinguish).
    pub fn quality_log(&self) -> &[(i64, Option<String>, crate::research::contract::QualityFlags)] {
        &self.quality_log
    }

    /// `condition_id` of a token's market, if the registry knows it
    /// (populated by `BookArrive`/tape meta — never invented).
    pub fn token_cid(&self, token_id: &str) -> Option<String> {
        self.registry.token_to_cid.get(token_id).map(|c| c.clone())
    }

    /// Market metadata known from the tape — to build
    /// agent intents with the REAL market data, not assumed data.
    /// Returns `(slug, clear_at, neg_risk, yes_token)`.
    pub fn market_meta(&self, cid: &str) -> Option<(String, i64, bool, String)> {
        self.metas
            .get(cid)
            .map(|m| (m.slug.clone(), m.clear_at, m.neg_risk, m.yes_token.clone()))
    }

    /// Enqueues an agent `OrderIntent` at the virtual instant
    /// (+ `decision` stage of the profile, 0 in nominal). Entry point
    /// of the episodic environment: the action responds to the last
    /// causal observation and enters the real pipeline (Dispatch → Executor →
    /// RiskGuard → router) with the same modeled latency as an intent of
    /// strategy — never a bypass of the sim.
    pub fn push_agent_intent(&mut self, intent: OrderIntent) {
        let now = self.clock_ms.load(Ordering::Relaxed);
        let decision = self.sample_leg(LatencyLeg::Decision, 1.0);
        let ts = now + decision;
        self.queue.push(
            ts,
            EvKind::Dispatch {
                intent: Box::new(intent),
                enqueued_ms: ts,
            },
        );
    }

    /// Builds the run outcome: counters + canonical economic digest.
    async fn outcome(&mut self) -> ReplayOutcome {
        let unresolved = self
            .state
            .count_unresolved_unknowns()
            .await
            .unwrap_or(-1)
            .max(0) as u64;
        let live_gtds = self
            .sim
            .live_gtd_order_ids()
            .await
            .map(|s| s.len() as u64)
            .unwrap_or(0);
        let realized = self.state.realized_pnl_total().await.unwrap_or_default();
        // Return↔ledger reconciliation at the end of the run — the
        // final sim cash (live balance, more reliable than the watch if it was never
        // published) against the BUY/SELL/REDEEM flows of the durable ledger.
        let accounting = self.sim.accounting();
        let flows = self.state.economic_flows().await.unwrap_or_default();
        let reconciliation = fill_sink::reconcile_economic_return(
            &flows,
            accounting.initial_balance,
            accounting.contributions_usdc,
            accounting.balance,
        );
        let digest = self.build_digest(&reconciliation).await;
        // `quality` summary of the contract over the run's scope —
        // the marks ACTUALLY processed (this run's quality_log)
        // + the reconstructor's counters per token/cid.
        let mut resyncs = 0u64;
        let mut gap_cids: HashSet<String> = HashSet::new();
        for (_, cid, flag) in &self.quality_log {
            if flag.contains(crate::research::contract::QualityFlags::RESYNC) {
                resyncs += 1;
            }
            if flag.contains(crate::research::contract::QualityFlags::GAP) {
                if let Some(c) = cid {
                    gap_cids.insert(c.clone());
                }
            }
        }
        let quality = self
            .tape_stats
            .summary(self.quality_log.len() as u64, resyncs, &gap_cids);
        // Declared CAUSAL fidelity — "certified" requires ALL the
        // conditions (causal ResearchEvent tape, no diagnostic proxy
        // for settlement, no post-settle corrections, reconciliation
        // exact and zero degradation in the run's scope).
        // Each violation leaves an explicit reason — a certification is never
        // announced that does not exist. EXTERNAL execution fidelity vs
        // the real venue is not identified: `external_fidelity` declares it
        // as a constant and is never merged with the causal one.
        let reconciled = reconciliation.reconciled();
        let mut fidelity_reasons: Vec<String> = Vec::new();
        if self.tape_source != TapeSource::ResearchEvents {
            fidelity_reasons.push("tape_without_research_events".to_string());
        }
        if self.proxy_used {
            fidelity_reasons.push("settlement_proxy_diagnostic".to_string());
        }
        if self.stats_corrections > 0 {
            fidelity_reasons.push(format!("settlement_corrections={}", self.stats_corrections));
        }
        if !reconciled {
            fidelity_reasons.push("failed_reconciliation".to_string());
        }
        if quality.discarded_deltas_in_scope > 0 {
            fidelity_reasons.push(format!(
                "scope_discarded_deltas={}",
                quality.discarded_deltas_in_scope
            ));
        }
        if quality.tokens_with_gap_in_scope > 0 {
            fidelity_reasons.push(format!(
                "tokens_with_gap_scope={}",
                quality.tokens_with_gap_in_scope
            ));
        }
        let fidelity = if fidelity_reasons.is_empty() {
            "certified"
        } else {
            "reduced"
        };
        let mut settlement_pending: Vec<String> = self.pending_settlement.iter().cloned().collect();
        settlement_pending.sort();
        // Counters are PER RUN (the engine may be reused, e.g.
        // after `restart_executor`) — accumulated values from an earlier run do not
        // contaminate the next run's outcome.
        ReplayOutcome {
            intents: std::mem::take(&mut self.stats_intents),
            router_calls: std::mem::take(&mut self.stats_router_calls),
            market_fills: std::mem::take(&mut self.stats_fills),
            gate_rejected: std::mem::take(&mut self.stats_gate),
            rejected_router: std::mem::take(&mut self.stats_rejected),
            cancel_acks: std::mem::take(&mut self.stats_cancel_acks),
            settlements: std::mem::take(&mut self.stats_settlements),
            unresolved_unknowns: unresolved,
            live_gtds,
            fills_seen: std::mem::take(&mut self.fills_seen),
            executions: self.sim.executions(),
            wallet_final: *self.wallet_rx.borrow(),
            digest,
            reconciliation,
            labels_applied: std::mem::take(&mut self.stats_labels),
            settlement_corrections: std::mem::take(&mut self.stats_corrections),
            settlement_pending,
            settlement_diagnostic: std::mem::take(&mut self.proxy_used),
            fidelity,
            fidelity_reasons,
            external_fidelity: "unidentified",
            quality,
            realized_pnl: realized,
            ledger: self.ledger.clone(),
        }
    }

    /// Canonical economic ledger digest for byte-by-byte comparison:
    /// positions, transactions, Unknown journal, sim executions,
    /// balance and the return↔ledger reconciliation — everything that decides the
    /// economics of the run.
    ///
    /// All `ts_ms` values are deterministic: `state.rs` stamps with
    /// `crate::time::now()` (virtual in replay — REDEEM of
    /// `apply_settlement`, `record_unknown_attempt_pending`,
    /// `mark_unknown_resolved`, `order_index`, `gtd_orders`, quarantine) and
    /// the `fill.ts` values are re-stamped to the virtual clock in `drain_fills`.
    /// No normalization: what is compared is what was persisted.
    async fn build_digest(&self, recon: &fill_sink::ReturnReconciliation) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        // Open positions (canonical order by token+strategy).
        let mut open = self.state.list_open_positions().await.unwrap_or_default();
        open.sort_by(|a, b| {
            (a.token_id.clone(), a.strategy.clone()).cmp(&(b.token_id.clone(), b.strategy.clone()))
        });
        for p in &open {
            let _ = writeln!(
                s,
                "POS|{}|{}|{}|{}|{}|{}|{}",
                p.token_id,
                p.strategy,
                p.size,
                p.avg_price,
                p.cost_basis,
                p.opened_ts_ms,
                p.clear_at
            );
        }
        // Transactions (ascending order by id — recent_transactions comes DESC).
        let mut txs = self
            .state
            .recent_transactions(u32::MAX, None)
            .await
            .unwrap_or_default();
        txs.reverse();
        for t in txs {
            let _ = writeln!(s, "TX|{}", serde_json::to_string(&t).unwrap_or_default());
        }
        // Pending Unknown journal — `ts_ms` included: `state.rs` stamps it
        // with `crate::time::now()` (deterministic in replay).
        let mut unk = self
            .state
            .list_unresolved_unknowns()
            .await
            .unwrap_or_default();
        unk.sort_by(|a, b| a.attempt_id.cmp(&b.attempt_id));
        for u in unk {
            let _ = writeln!(
                s,
                "UNK|{}|{}|{}|{}|{}|{}|{}",
                u.attempt_id, u.cid, u.token_id, u.strategy, u.cost_um, u.resting, u.ts_ms
            );
        }
        // Sim executions (deterministic ids per seed+clock).
        for e in self.sim.executions() {
            let _ = writeln!(
                s,
                "EXEC|{}|{}|{}|{}|{}|{}|{}",
                e.exec_id, e.order_id, e.token_id, e.side, e.shares, e.cash, e.ts_ms
            );
        }
        let _ = writeln!(
            s,
            "WALLET|{}",
            self.wallet_rx
                .borrow()
                .map(|d| d.to_string())
                .unwrap_or_default()
        );
        // The return↔ledger reconciliation goes INTO the digest — two runs
        // with the same tape must end with the same economic verdict.
        let _ = writeln!(
            s,
            "RECON|{}|{}|{}|{}|{}|{}",
            recon.reconciled(),
            recon.cash_return,
            recon.ledger_net_flow,
            recon.pnl_ledger,
            recon.cash_drift,
            recon.pnl_drift
        );
        s
    }
}

// ── CLI entry (`hrf_backtest --engine pipeline`) ────────────────────────────

/// Selector of the simulator's research profile for a replay
/// (`--research-profile` of `hrf_backtest`).
///
/// - `Nominal`: no profile is applied — the sim reproduces the
///   operational behavior (the default).
/// - `Conservative`: `ResearchSimConfig::conservative()` — worst scenario
///   identifiable per hypothesis (κ=0, 25% participation, stale, gaps,
///   inventory). Hardcoded, not from the TOML.
/// - `Config`: the `[research_sim]` section of the loaded `general.toml` —
///   to certify calibrated profiles without recompiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResearchProfile {
    Nominal,
    Conservative,
    Config,
}

/// Replays the recorded window through the REAL PIPELINE and returns the report
/// JSON (pipeline counters + deterministic economic digest). Reuses
/// `BacktestParams` of the diagnostic engine: same DBs, same TOML config and
/// same seed — only the path the tape travels changes.
///
/// `fak_strategies` is derived from the config as in `main.rs`
/// (`near_certain.use_fak` / `near_expiry.use_fak`) — the table is not duplicated.
pub async fn run_pipeline(
    params: &crate::backtest::BacktestParams,
    latency_ms: i64,
    obs_delay_ms: i64,
    research: ResearchProfile,
    settlement_proxy_diagnostic: bool,
) -> anyhow::Result<serde_json::Value> {
    let from_ms = params.from_ms.unwrap_or(i64::MIN + 1);
    let to_ms = params.to_ms.unwrap_or(i64::MAX - 1);
    let tape = load_tape(&params.db_books, params.db_spot.as_deref(), from_ms, to_ms)?;
    anyhow::ensure!(
        !tape.books.is_empty() || !tape.spots.is_empty(),
        "the window [{from_ms}, {to_ms}] has no recorded events"
    );
    run_pipeline_tape(
        params,
        tape,
        "sql",
        latency_ms,
        obs_delay_ms,
        research,
        settlement_proxy_diagnostic,
    )
    .await
}

/// Core of the real-pipeline runner over any `Tape` —
/// historical SQL (`tape_kind="sql"`, declared reduced fidelity) or tape
/// causal `ResearchEvent` (`tape_kind="research_events"`, `--research-
/// events` of the binary). `settlement_proxy_diagnostic` enables the proxy of
/// spot ONLY as an explicit diagnostic (forces `fidelity="reduced"`).
#[allow(clippy::too_many_arguments)]
pub async fn run_pipeline_tape(
    params: &crate::backtest::BacktestParams,
    tape: Tape,
    tape_kind: &str,
    latency_ms: i64,
    obs_delay_ms: i64,
    research: ResearchProfile,
    settlement_proxy_diagnostic: bool,
) -> anyhow::Result<serde_json::Value> {
    let (cfgs, probe) = crate::backtest::load_configs(&params.config_dir)?;
    let mut strategy = crate::backtest::build_strategy(
        params.strategy,
        &cfgs,
        probe.fair_value.confidence_haircut,
    )?;

    let mut fak_strategies = HashSet::new();
    if cfgs.near_certain.use_fak {
        fak_strategies.insert(StrategyId::NearCertain);
    }
    if cfgs.near_expiry.use_fak {
        fak_strategies.insert(StrategyId::NearExpiry);
    }

    // Profile resolution — `Config` reads `[research_sim]` from the TOML
    // loaded; `Nominal` leaves the sim without a profile (operational).
    let research_cfg = match research {
        ResearchProfile::Nominal => None,
        ResearchProfile::Conservative => Some(crate::config::ResearchSimConfig::conservative()),
        ResearchProfile::Config => Some(probe.research_sim.clone()),
    };
    let research_profile = research_cfg
        .as_ref()
        .map(|r| r.profile.clone())
        .unwrap_or_else(|| "nominal".to_string());

    let engine_params = ReplayParams {
        seed: params.seed,
        initial_balance: params.initial_balance,
        risk: probe.risk,
        obs_delay_ms,
        latency_ms,
        fak_strategies,
        // Runner's stale-book gate: `market_feed.max_book_age_ms` from the
        // loaded config — same origin as `main.rs`.
        max_book_age_ms: probe.market_feed.max_book_age_ms as i64,
        research: research_cfg,
        settlement_proxy_diagnostic,
        ..Default::default()
    };
    let state = StateStore::open_memory(Mode::Paper)
        .await
        .context("ephemeral StateStore for the replay")?;
    let mut engine = ReplayEngine::new(engine_params, state).await?;
    let out = engine.run(tape, &mut *strategy).await?;

    Ok(serde_json::json!({
        "engine": "pipeline",
        "tape_kind": tape_kind,
        "strategy": params.strategy.to_string(),
        "seed": params.seed,
        "latency_ms": latency_ms,
        "obs_delay_ms": obs_delay_ms,
        "research_profile": research_profile,
        "intents": out.intents,
        "router_calls": out.router_calls,
        "market_fills": out.market_fills,
        "gate_rejected": out.gate_rejected,
        "rejected_router": out.rejected_router,
        "cancel_acks": out.cancel_acks,
        "settlements": out.settlements,
        "unresolved_unknowns": out.unresolved_unknowns,
        "live_gtds": out.live_gtds,
        "realized_pnl": out.realized_pnl().to_string(),
        "wallet_final": out.wallet_final.map(|d| d.to_string()),
        // Economic acceptance gate — return reconcilable with the
        // ledger (drifts at 0), visible in the runner's JSON report.
        "reconciled": out.reconciled(),
        "recon_derives_cash": out.reconciliation.cash_drift.to_string(),
        "recon_derives_pnl": out.reconciliation.pnl_drift.to_string(),
        // Fidelity declaration and official settlement — the
        // report never presents a proxy-based run as certified.
        // `fidelity` measures ONLY causal integrity (with its reasons if it is
        // "reduced"); `external_fidelity` remains "unidentified" always.
        "fidelity": out.fidelity,
        "fidelity_reasons": out.fidelity_reasons,
        "external_fidelity": out.external_fidelity,
        "quality": out.quality,
        "labels_applied": out.labels_applied,
        "settlement_corrections": out.settlement_corrections,
        "settlement_pending": out.settlement_pending,
        "settlement_diagnostic": out.settlement_diagnostic,
        "digest": out.digest,
    }))
}
