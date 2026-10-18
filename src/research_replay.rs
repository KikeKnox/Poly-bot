//! Causal adapter from `ResearchEvent` (recorder JSONL) to the `Tape` of the
//! `ReplayEngine`, plus the incremental episodic environment (tape loading and
//! episode stepping for the numeric environment).
//!
//! ## Tape loading
//!
//! `ResearchTape::load` reads a `.jsonl` file or a directory (recursive,
//! `*.jsonl` paths sorted), using the layout written by `ResearchRecorder`:
//! `dir/<source>/<yyyymmdd>.jsonl`. It decodes each line as a
//! [`LineaGrabada`] with STRICT decoding (a malformed line stays in
//! `diagnostics` — with `strict = true` it aborts — and the event is never fabricated)
//! and sorts the stream by the canonical key `(avail_ts_ms, local_seq)` —
//! NEVER by `origin_ts_ms` (remote clock, which orders nothing).
//!
//! ## Book reconstruction
//!
//! Recorder events are PER TOKEN (a `WsMessage`/`BookSummary` records a single
//! `asset_id`); the engine's `TapeBook` is PER MARKET (YES+NO legs). The
//! reconstructor replicates the logic of `market_feed::handle` over the per-token state:
//!
//! - `BookSnapshot` (WS `book`/`initial_book` and REST `BookSummary`):
//!   authoritative — replaces the token's bids/asks and emits the market book
//!   (forced) if the cid's meta is known and NEITHER half is `needs_snapshot`
//!   (mixing a fresh half with a frozen one fabricates phantom edges).
//! - `BookDelta` (`price_changes`): applies entries to the token's state;
//!   a delta WITHOUT a prior base (token without snapshot) is discarded and the
//!   segment is marked `QualityFlags::GAP` in `tape.quality` — a book is never
//!   fabricated. A crossed book after a delta → `crossed_pending`; if the next
//!   delta of the SAME token is still crossed → `needs_snapshot` + `GAP` mark
//!   (documented simplification: without the time backoff of `crossed_escalation`,
//!   which in production only throttles WARN spam — the economic effect,
//!   suppression of emissions and discarding of deltas, is kept).
//! - `tick_size_change` (arrives as a `BookDelta` with `new_tick_size`): updates
//!   the meta's tick and forces re-emission.
//! - `Trade` (`last_trade_price`): only feeds the per-token flow aggregate
//!   (window `tape_window_secs`); it does NOT emit a book — in production the
//!   twin `price_change` is the one that emits, so it is not counted twice.
//! - `PriceTick` → `TapeSpot` at `avail_ts_ms` (the oracle observation is immediate: continuous watch).
//! - `MarketMeta` → registers cid→(yes,no,slug,clear_at,…); if the payload
//!   already carries `outcome`, it also produces a `TapeResolution`.
//! - `Resolution` → `TapeResolution` at `avail_ts_ms` (the label is applied to the
//!   ephemeral `StateStore` via `apply_official` in the engine; the outcome's own
//!   `known_at_ms` also filters its availability — a double causal barrier).
//! - `OrderSubmit`/`Sim(_)` → do NOT enter the tape: they are the product of the
//!   historical agent (a new policy must not inherit fills for free); they are counted in `agent_events`.
//! - Structural quality (`GAP`/`RESYNC`/`DUPLICATE`/`REDUCED_FIDELITY`) of ANY event →
//!   `TapeQuality` at `avail_ts_ms` for the episode's `DataQualityAbort` terminal state.
//!   `MISSING_REMOTE_TS` and `SYNTHETIC` are informational and do not degrade the segment.
//!
//! ## Episodic environment
//!
//! [`ResearchEnv`] drives the SAME `ReplayEngine` event by event via
//! `seed_tape`/`step_once`/`finalize` — without rebuilding state or re-running the
//! tape on each step. The internal agent [`EnvAgent`] is a passive `Strategy`: it
//! emits no intents of its own, it only captures what the pipeline delivers to it
//! (`on_book` → last real observation; `on_fill` → fills/rejects in the window) under
//! `StrategyId::Follower` — the enum variant dedicated to external entities that reuse
//! the attribution infrastructure (see the variant's doc in `types.rs`).
//!
//! `AgentAction`s are translated into REAL `OrderIntent`s and enter the pipeline via
//! `ReplayEngine::push_agent_intent` (Dispatch → Executor → RiskGuard → DeferredSimRouter →
//! SimulatedRouter → fill_sink): same ledger, same modeled latencies, zero parallel accounting.
//! `Buy`/`Sell` are sent as limit GTD (`resting: true`): crossing the book produces an
//! immediate fill (`Executed`); resting on the book produces `Accepted` and is cancelable
//! via `Cancel{order_id}`.
//!
//! ## Reward (economic identity)
//!
//! `equity_t = balance_total_sim + Σ_token exit_value(token)` where `exit_value` walks the
//! bids of the LAST OBSERVED BOOK (causal) up to the available depth; shares without visible
//! depth are worth 0 (purely conservative — never overvalues). With `fees_t = Σ fee_usdc`
//! from the sim's execution log:
//!
//! ```text
//! price_component = Δ(equity + fees)   // gross of friction
//! fee_component   = −Δfees             // ≤ 0
//! risk_penalty    = 0                  // the env applies no shaping;
//!                                      // the breakdown exists so the
//!                                      // trainer can recompose it
//! total           = price + fee + risk = exact Δequity
//! ```
//!
//! ⇒ `Σ total = equity_T − equity_0` and, when everything is liquidated (inventory 0),
//! `equity_T = balance` — the identity against the ledger.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use rust_decimal::Decimal;
use serde_json::Value;

use crate::market_feed::handle::{RawLevel, RawPriceChange, WsMessage};
use crate::market_feed::registry::{book_is_crossed, effective_tick};
use crate::order::router::OrderRouter;
use crate::replay::{
    ReplayEngine, ReplayOutcome, ReplayParams, StepMark, Tape, TapeBook, TapeQuality,
    TapeQualityStats, TapeResolution, TapeSource, TapeSpot,
};
use crate::research_contract::{
    ActionOutcome, AgentAction, EndReason, EnvObservation, EventKind, EventSource,
    InventorySnapshot, MarketOutcome, PositionSnapshot, QualityFlags, QualityScope, QualitySummary,
    ResearchEvent, RewardBreakdown, StepRecord, SCHEMA_VERSION,
};
use crate::research_events::RecordedLine;
use crate::types::{
    BookSnapshot, Fill, FillSide, HalfBook, Level, Mode, OrderIntent, OrderSide, StrategyId,
    TokenBook, VolSnapshot,
};

// ════════════════════════════════════════════════════════════════════════════
// Part 1 — Loading and reconstruction of the causal tape
// ════════════════════════════════════════════════════════════════════════════

/// Options for loading/reconstructing the `ResearchEvent` tape.
#[derive(Debug, Clone)]
pub struct TapeOpts {
    /// Window of the per-token flow aggregate (seconds) — same semantics as
    /// `market_feed.tape_window_secs` in production. `0` = flow disabled
    /// (`Trade` events do not accumulate `flow_net`/`flow_total` in the books).
    pub tape_window_secs: u64,
    /// `true`: a malformed line aborts the load with an error (acceptance
    /// mode). `false` (default): the line stays in `diagnostics` and the load
    /// continues — the degraded segment is marked, not invented.
    pub strict: bool,
    /// Safety cap on lines read (`None` = no cap).
    pub max_lines: Option<u64>,
}

impl Default for TapeOpts {
    fn default() -> Self {
        Self {
            tape_window_secs: 0,
            strict: false,
            max_lines: None,
        }
    }
}

/// Diagnostic of the load/reconstruction — a malformed line, a delta without
/// a base, a book without meta… Each one carries its provenance (file,
/// line, `local_seq` if known) for auditing.
#[derive(Debug, Clone, PartialEq)]
pub struct TapeDiagnostic {
    /// Path of the file or origin label (`"<stream>"` in memory).
    pub origin: String,
    /// Line number in the file (0 if not applicable).
    pub line: u64,
    /// `local_seq` of the affected event, if it could be decoded.
    pub local_seq: Option<u64>,
    /// Actionable description of the problem.
    pub message: String,
}

/// Result of loading a `ResearchEvent` tape: the `Tape` for the
/// `ReplayEngine` (source=ResearchEvents) plus the diagnostics.
#[derive(Debug)]
pub struct ResearchTape {
    pub tape: Tape,
    pub diagnostics: Vec<TapeDiagnostic>,
    /// Events decoded successfully.
    pub events: u64,
    /// Lines that did not decode (malformed / version / invariant).
    pub malformed_lines: u64,
    /// `OrderSubmit`/`Sim(_)` events of the historical agent — counted,
    /// NEVER replayed on the tape.
    pub agent_events: u64,
    /// Market books suppressed because the opposite half still had no
    /// state (production doesn't emit either — `try_emit_snapshot` returns).
    pub books_missing_half: u64,
    /// Books suppressed for an unchanged top-of-book (the runner's emission
    /// dedup — same effect as `hrf_market_feed_emit_skipped_total`).
    pub books_top_equal: u64,
    /// Deltas discarded for lack of a base / `needs_snapshot`.
    pub discarded_deltas: u64,
}

impl ResearchTape {
    /// Loads from a `.jsonl` file or a directory (recursive). Paths are sorted
    /// lexicographically before being read — identical input yields an identical tape.
    pub fn load(path: &Path, opts: &TapeOpts) -> anyhow::Result<Self> {
        let mut files: Vec<PathBuf> = Vec::new();
        if path.is_dir() {
            collect_jsonl(path, &mut files)
                .with_context(|| format!("walking {}", path.display()))?;
            files.sort();
            anyhow::ensure!(
                !files.is_empty(),
                "directory has no .jsonl files: {}",
                path.display()
            );
        } else {
            anyhow::ensure!(path.is_file(), "not a file or directory: {}", path.display());
            files.push(path.to_path_buf());
        }

        let mut events: Vec<ResearchEvent> = Vec::new();
        let mut diagnostics: Vec<TapeDiagnostic> = Vec::new();
        let mut malformed: u64 = 0;
        let mut read: u64 = 0;
        for f in &files {
            let fh = std::fs::File::open(f).with_context(|| format!("opening {}", f.display()))?;
            let origin = f.display().to_string();
            for (n, line) in std::io::BufReader::new(fh).lines().enumerate() {
                read += 1;
                if let Some(max) = opts.max_lines {
                    if read > max {
                        bail!("line cap exceeded ({max}) while loading {}", f.display());
                    }
                }
                let line = line.with_context(|| format!("reading {}:{}", origin, n + 1))?;
                let t = line.trim();
                if t.is_empty() {
                    continue;
                }
                let line_number = (n + 1) as u64;
                match serde_json::from_str::<RecordedLine>(t) {
                    Ok(l) => {
                        let ev = l.event;
                        if let Err(e) = ev.validate() {
                            malformed += 1;
                            let d = TapeDiagnostic {
                                origin: origin.clone(),
                                line: line_number,
                                local_seq: Some(ev.local_seq),
                                message: format!("temporal invariant violated: {e}"),
                            };
                            if opts.strict {
                                bail!("{}:{}: {}", origin, line_number, d.message);
                            }
                            diagnostics.push(d);
                            continue;
                        }
                        if ev.schema_version != SCHEMA_VERSION {
                            malformed += 1;
                            let d = TapeDiagnostic {
                                origin: origin.clone(),
                                line: line_number,
                                local_seq: Some(ev.local_seq),
                                message: format!(
                                    "schema_version {} != {} of the contract",
                                    ev.schema_version, SCHEMA_VERSION
                                ),
                            };
                            if opts.strict {
                                bail!("{}:{}: {}", origin, line_number, d.message);
                            }
                            diagnostics.push(d);
                            continue;
                        }
                        events.push(ev);
                    }
                    Err(e) => {
                        malformed += 1;
                        let d = TapeDiagnostic {
                            origin: origin.clone(),
                            line: line_number,
                            local_seq: None,
                            message: format!("malformed JSON: {e}"),
                        };
                        if opts.strict {
                            bail!("{}:{}: {}", origin, line_number, d.message);
                        }
                        diagnostics.push(d);
                    }
                }
            }
        }
        Ok(Self::from_events(events, diagnostics, malformed, opts))
    }

    /// Builds the tape from already-decoded events (tests and composition).
    /// The canonical order is imposed here:
    /// `(avail_ts_ms, local_seq)`.
    pub fn from_events(
        mut events: Vec<ResearchEvent>,
        diagnostics: Vec<TapeDiagnostic>,
        malformed: u64,
        opts: &TapeOpts,
    ) -> Self {
        // Canonical key: availability first, then local causal order.
        // `local_seq` is the recorder's global monotonic counter — the tie-break is
        // total (two events never share a local_seq within one capture;
        // if captures are mixed, the seq ties and the stable sort_by_key order over the
        // load index applies — no remote clock).
        events.sort_by_key(|e| (e.avail_ts_ms, e.local_seq));
        let mut rec = Reconstructor::new(opts);
        let n = events.len() as u64;
        for ev in &events {
            rec.process(ev);
        }
        let mut diagnostics = diagnostics;
        diagnostics.append(&mut rec.diagnostics);
        let (agent_events, books_missing_half, books_top_equal, discarded_deltas) = (
            rec.agent_events,
            rec.books_missing_half,
            rec.books_top_equal,
            rec.discarded_deltas,
        );
        let mut tape = rec.into_tape();
        tape.quality_stats.malformed = malformed;
        Self {
            tape,
            diagnostics,
            events: n,
            malformed_lines: malformed,
            agent_events,
            books_missing_half,
            books_top_equal,
            discarded_deltas,
        }
    }
}

/// Collects `*.jsonl` files recursively (stable order by path).
fn collect_jsonl(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let p = e.path();
        if p.is_dir() {
            collect_jsonl(&p, out)?;
        } else if p.extension().is_some_and(|x| x == "jsonl")
            && !p.file_name().is_some_and(|n| n == "_recorder_stats.jsonl")
        {
            out.push(p);
        }
    }
    Ok(())
}

/// Preloads official labels from a separate research DB
/// (`--labels-db`): reads `market_resolutions` (rows `confirmed` +
/// `superseded`, sorted by id per cid — the sequence reproduces the
/// corrections as they happened) and turns them into `TapeResolution`s
/// positioned at `known_at_ms` — the REAL instant the label became known,
/// never backdated to `clear_at` nor to the end of the tape.
///
/// `provisional` is always excluded: it is not a certifiable label.
pub fn load_labels_db(path: &Path) -> anyhow::Result<Vec<TapeResolution>> {
    // Read-only, like `backtest::open_readonly` — the labels DB is untouchable
    // evidence (no WAL checkpoint nor journal).
    let conn = crate::backtest::open_readonly(path)?;
    let mut stmt = conn.prepare(
        "SELECT market_cid, winner, status, source, provenance, known_at_ms
           FROM market_resolutions
          WHERE status IN ('confirmed','superseded')
          ORDER BY market_cid, id",
    )?;
    let mut out = Vec::new();
    let mut rows = stmt.query([])?;
    while let Some(r) = rows.next()? {
        let cid: String = r.get(0)?;
        let winner_raw: String = r.get(1)?;
        let source_raw: String = r.get(3)?;
        let provenance: String = r.get(4)?;
        let known_at_ms: i64 = r.get(5)?;
        let winner = match winner_raw.as_str() {
            "yes" => crate::research_contract::OutcomeSide::Yes,
            "no" => crate::research_contract::OutcomeSide::No,
            other => bail!("labels_db {path:?}: unknown winner {other:?}"),
        };
        let source: EventSource = serde_json::from_value(Value::String(source_raw.clone()))
            .with_context(|| format!("labels_db {path:?}: unknown source {source_raw:?}"))?;
        anyhow::ensure!(
            !provenance.trim().is_empty(),
            "labels_db {path:?}: empty provenance in {cid}"
        );
        out.push(TapeResolution {
            avail_ms: known_at_ms,
            tape_seq: None,
            cid,
            outcome: MarketOutcome {
                winner,
                known_at_ms,
                provenance,
            },
            source,
        });
    }
    Ok(out)
}

// ── Internal state of the reconstructor ────────────────────────────────────────

/// Market meta rebuilt from `EventKind::MarketMeta` — the REAL data recorded by
/// discovery (never assumed).
struct MetaRec {
    slug: String,
    yes: String,
    no: String,
    market_open_ts: i64,
    clear_at: i64,
    neg_risk: bool,
    reference_price: Option<Decimal>,
    fee_rate_bps: Option<u32>,
    parabolic_fee_rate: Option<Decimal>,
    /// Current tick: the last of `rules_windows`, updated by `tick_size_change`
    /// (like `MarketEntry::tick_size` in the registry).
    tick_size: Option<Decimal>,
}

/// Per-token state, mirror of `registry.token_books` (`TokenState`).
#[derive(Default)]
struct TokenBookLevels {
    bids: Vec<Level>, // desc (best bid first)
    asks: Vec<Level>, // asc (best ask first)
}

/// Per-token flow aggregate: (ts_ms, signed net, gross) per trade.
struct Flow {
    trades: VecDeque<(i64, Decimal, Decimal)>,
}

impl Flow {
    /// `flow_window` of the registry: (net, total) in `[now − secs*1000, now]`.
    /// `None` if the token never had trades (absent signal, not zero).
    fn window(&mut self, now_ms: i64, secs: u64) -> Option<(Decimal, Decimal)> {
        if self.trades.is_empty() {
            return None;
        }
        let since = now_ms.saturating_sub(secs.saturating_mul(1000) as i64);
        // Retention = 2× window (same margin as `record_trade`).
        let retention = since.saturating_sub(secs.saturating_mul(1000) as i64);
        while self.trades.front().is_some_and(|(t, ..)| *t < retention) {
            self.trades.pop_front();
        }
        let mut net = Decimal::ZERO;
        let mut tot = Decimal::ZERO;
        for (t, n, a) in &self.trades {
            if *t >= since && *t <= now_ms {
                net += *n;
                tot += *a;
            }
        }
        Some((net, tot))
    }
}

struct Reconstructor<'a> {
    opts: &'a TapeOpts,
    metas: HashMap<String, MetaRec>,
    token2cid: HashMap<String, String>,
    books: HashMap<String, TokenBookLevels>,
    needs_snapshot: HashSet<String>,
    crossed_pending: HashSet<String>,
    /// `last_top` of the registry: [yes_bid, yes_ask, no_bid, no_ask].
    last_top: HashMap<String, [Option<Decimal>; 4]>,
    flow: HashMap<String, Flow>,
    /// Tokens with a currently invalidated segment (delta without base / persistent
    /// crossing): the `GAP` mark is emitted once per invalidation episode, not for
    /// every delta (noise). The flag is lifted by the token's next valid snapshot.
    invalidated: HashSet<String>,
    out_books: Vec<TapeBook>,
    out_spots: Vec<TapeSpot>,
    out_res: Vec<TapeResolution>,
    out_quality: Vec<TapeQuality>,
    diagnostics: Vec<TapeDiagnostic>,
    agent_events: u64,
    books_missing_half: u64,
    books_top_equal: u64,
    discarded_deltas: u64,
    // Counts per token/cid for the replay report scope
    // (see `TapeQuality` in `replay.rs`).
    /// `price_change` entries processed per cid (resolved).
    deltas_by_cid: HashMap<String, u64>,
    /// `price_change` entries with no attributable cid, per token.
    deltas_no_cid_by_token: HashMap<String, u64>,
    /// Discarded deltas per token.
    discarded_by_token: HashMap<String, u64>,
    /// Discarded deltas per cid (resolved).
    discarded_by_cid: HashMap<String, u64>,
    /// Discarded deltas with no attributable cid, per token.
    discarded_no_cid_by_token: HashMap<String, u64>,
    /// Tokens with a gap episode (invalidation or a GAP mark from the recorder).
    tokens_with_gap: HashSet<String>,
}

impl<'a> Reconstructor<'a> {
    fn new(opts: &'a TapeOpts) -> Self {
        Self {
            opts,
            metas: HashMap::new(),
            token2cid: HashMap::new(),
            books: HashMap::new(),
            needs_snapshot: HashSet::new(),
            crossed_pending: HashSet::new(),
            last_top: HashMap::new(),
            flow: HashMap::new(),
            invalidated: HashSet::new(),
            out_books: Vec::new(),
            out_spots: Vec::new(),
            out_res: Vec::new(),
            out_quality: Vec::new(),
            diagnostics: Vec::new(),
            agent_events: 0,
            books_missing_half: 0,
            books_top_equal: 0,
            discarded_deltas: 0,
            deltas_by_cid: HashMap::new(),
            deltas_no_cid_by_token: HashMap::new(),
            discarded_by_token: HashMap::new(),
            discarded_by_cid: HashMap::new(),
            discarded_no_cid_by_token: HashMap::new(),
            tokens_with_gap: HashSet::new(),
        }
    }

    fn diag(&mut self, ev: &ResearchEvent, message: impl Into<String>) {
        self.diagnostics.push(TapeDiagnostic {
            origin: "<stream>".into(),
            line: 0,
            local_seq: Some(ev.local_seq),
            message: message.into(),
        });
    }

    /// Propagates the event's structural marks to `tape.quality`
    /// (positioned at `avail_ts_ms` — the causal instant at which the
    /// consumer would have seen them).
    fn propagate_quality(&mut self, ev: &ResearchEvent) {
        let structural = ev.quality.0
            & (QualityFlags::GAP.0
                | QualityFlags::RESYNC.0
                | QualityFlags::DUPLICATE.0
                | QualityFlags::REDUCED_FIDELITY.0);
        if structural != 0 {
            // A recorder GAP posted on a token counts toward `tokens_with_gap`
            // (the scope filters it through token2cid).
            if structural & QualityFlags::GAP.0 != 0 {
                if let Some(t) = &ev.token_id {
                    self.tokens_with_gap.insert(t.clone());
                }
            }
            self.out_quality.push(TapeQuality {
                ts_ms: ev.avail_ts_ms,
                tape_seq: Some(ev.local_seq),
                cid: ev.market_cid.clone(),
                flag: QualityFlags(structural),
            });
        }
    }

    /// Marks a degraded segment of the token (once per invalidation episode —
    /// the flag is lifted by the token's next valid snapshot).
    /// The token always enters `tokens_with_gap` (the set deduplicates;
    /// emitting the mark is still once per episode).
    fn mark_segment(&mut self, ev: &ResearchEvent, token: &str, cid: Option<String>) {
        self.tokens_with_gap.insert(token.to_string());
        if self.invalidated.insert(token.to_string()) {
            self.out_quality.push(TapeQuality {
                ts_ms: ev.avail_ts_ms,
                tape_seq: Some(ev.local_seq),
                cid,
                flag: QualityFlags::GAP,
            });
        }
    }

    fn process(&mut self, ev: &ResearchEvent) {
        self.propagate_quality(ev);
        match ev.kind {
            EventKind::MarketMeta => self.on_meta(ev),
            EventKind::BookSnapshot => self.on_book_snapshot(ev),
            EventKind::BookDelta => self.on_book_delta(ev),
            EventKind::Trade => self.on_trade(ev),
            EventKind::PriceTick => self.on_price_tick(ev),
            EventKind::Resolution => self.on_resolution(ev),
            // Actions of the historical agent — never enter the tape of a new policy.
            EventKind::OrderSubmit | EventKind::Sim(_) => {
                self.agent_events += 1;
            }
        }
    }

    /// cid of the event: the `market_cid` stamped on it by the hook from the live
    /// registry comes first, then `payload.market` (the wire field).
    fn cid_of(&self, ev: &ResearchEvent, payload_market: Option<&str>) -> Option<String> {
        ev.market_cid
            .clone()
            .or_else(|| payload_market.map(str::to_string))
    }

    fn on_meta(&mut self, ev: &ResearchEvent) {
        let meta: crate::research_contract::MarketMeta = match ev.payload_as() {
            Ok(m) => m,
            Err(e) => {
                self.diag(ev, format!("unreadable MarketMeta payload: {e}"));
                return;
            }
        };
        self.token2cid
            .insert(meta.yes_token_id.clone(), meta.market_cid.clone());
        self.token2cid
            .insert(meta.no_token_id.clone(), meta.market_cid.clone());
        let tick_size = meta.rules_windows.last().and_then(|w| w.tick_size);
        if let Some(outcome) = &meta.outcome {
            // The meta already carried the official result — causal resolution.
            self.out_res.push(TapeResolution {
                avail_ms: ev.avail_ts_ms,
                tape_seq: Some(ev.local_seq),
                cid: meta.market_cid.clone(),
                outcome: outcome.clone(),
                source: ev.source,
            });
        }
        self.metas.insert(
            meta.market_cid.clone(),
            MetaRec {
                slug: meta.slug,
                yes: meta.yes_token_id,
                no: meta.no_token_id,
                market_open_ts: meta.market_open_ts,
                clear_at: meta.clear_at,
                neg_risk: meta.neg_risk,
                reference_price: meta.reference_price,
                fee_rate_bps: meta.rules_windows.last().and_then(|w| w.fee_rate_bps),
                parabolic_fee_rate: meta.rules_windows.last().and_then(|w| w.parabolic_fee_rate),
                tick_size,
            },
        );
    }

    /// Parses `{price,size}` string levels into `Level` — same rule as
    /// `handle::parse_levels` (size<=0 discarded, unreadable ones dropped).
    fn parse_levels(raw: &[RawLevel]) -> Vec<Level> {
        crate::market_feed::handle::parse_levels(raw)
    }

    /// Authoritative snapshot (WS `book`/`initial_book` or REST resync —
    /// both payloads carry `asset_id` + `bids`/`asks`).
    fn on_book_snapshot(&mut self, ev: &ResearchEvent) {
        let msg: WsMessage = match ev.payload_as() {
            Ok(m) => m,
            Err(e) => {
                self.diag(ev, format!("unreadable BookSnapshot payload: {e}"));
                return;
            }
        };
        // `BookSummary` (REST) serializes `token_id` when the field is not
        // asset_id — the wire alias; WsMessage does not declare it.
        let token = msg.asset_id.clone().filter(|s| !s.is_empty()).or_else(|| {
            ev.payload
                .get("token_id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        });
        let Some(token) = token else {
            self.diag(ev, "BookSnapshot without asset_id/token_id".to_string());
            return;
        };
        let market_field = msg
            .extra
            .get("market")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let cid = self.cid_of(ev, market_field.as_deref());

        // Token state: authoritative (dedup by price keeping the last occurrence +
        // order — same sequence as `handle.rs`).
        let mut bids = dedup_levels_by_price(Self::parse_levels(&msg.bids));
        let mut asks = dedup_levels_by_price(Self::parse_levels(&msg.asks));
        bids.sort_unstable_by_key(|l| std::cmp::Reverse(l.price));
        asks.sort_unstable_by_key(|l| l.price);

        // Authoritative snapshot clears the crossing state and the segment's invalidation mark.
        self.crossed_pending.remove(&token);
        self.needs_snapshot.remove(&token);
        self.invalidated.remove(&token);

        if book_is_crossed(&bids, &asks) {
            // Production: discards and requests a resync (needs_snapshot). The
            // book does NOT enter the state — the last healthy base is kept.
            self.needs_snapshot.insert(token.clone());
            let c = cid.clone().or_else(|| self.token2cid.get(&token).cloned());
            self.mark_segment(ev, &token, c);
            self.diag(ev, format!("crossed snapshot discarded (token {token})"));
            return;
        }
        self.books
            .insert(token.clone(), TokenBookLevels { bids, asks });
        self.emit(ev, &token, cid.as_deref(), true);
    }

    /// Incremental delta (`price_changes`) or `tick_size_change`.
    fn on_book_delta(&mut self, ev: &ResearchEvent) {
        let msg: WsMessage = match ev.payload_as() {
            Ok(m) => m,
            Err(e) => {
                self.diag(ev, format!("unreadable BookDelta payload: {e}"));
                return;
            }
        };
        let market_field = msg
            .extra
            .get("market")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let cid = self.cid_of(ev, market_field.as_deref());

        // `tick_size_change`: updates the meta's tick and forces re-emission
        // (same semantics as `handle.rs`).
        if let Some(tick_str) = msg.new_tick_size.as_deref() {
            let token = msg.asset_id.as_deref().unwrap_or("").to_string();
            match Decimal::from_str_exact(tick_str) {
                Ok(t) => {
                    if let Some(c) = cid.as_ref().or_else(|| self.token2cid.get(&token)) {
                        if let Some(m) = self.metas.get_mut(c) {
                            m.tick_size = Some(t);
                        }
                    }
                    self.emit(ev, &token, cid.as_deref(), true);
                }
                Err(_) => self.diag(ev, format!("unreadable tick_size: {tick_str}")),
            }
            return;
        }

        if msg.price_changes.is_empty() {
            return;
        }
        let mut updated: Vec<String> = Vec::new();
        for entry in &msg.price_changes {
            let token = entry.asset_id.as_str();
            if token.is_empty() {
                continue;
            }
            // Scope count per delta ENTRY (applied or discarded) — denominator of
            // `scope_deltas_pct_coverage`.
            let c_entry = cid.clone().or_else(|| self.token2cid.get(token).cloned());
            match &c_entry {
                Some(c) => *self.deltas_by_cid.entry(c.clone()).or_insert(0) += 1,
                None => {
                    *self
                        .deltas_no_cid_by_token
                        .entry(token.to_string())
                        .or_insert(0) += 1
                }
            }
            if self.needs_snapshot.contains(token) {
                self.discard(token, &c_entry);
                continue;
            }
            match self.books.get_mut(token) {
                Some(st) => {
                    apply_price_change_entry(st, entry);
                    if !updated.iter().any(|t| t == token) {
                        updated.push(token.to_string());
                    }
                }
                None => {
                    // Delta without base: the segment stays invalidated until the next
                    // snapshot — a book is never fabricated.
                    self.discard(token, &c_entry);
                    self.mark_segment(ev, token, c_entry);
                    self.diag(
                        ev,
                        format!("price_change without a previous snapshot (token {token})"),
                    );
                }
            }
        }

        for token in updated {
            // Crossed check at the END of the message (never per entry).
            let crossed = self
                .books
                .get(&token)
                .map(|st| book_is_crossed(&st.bids, &st.asks))
                .unwrap_or(false);
            if crossed {
                if self.crossed_pending.contains(&token) {
                    // Second consecutive crossing → genuine desync:
                    // needs_snapshot + segment invalidation.
                    // (Without the time backoff of `crossed_escalation` —
                    // offline reconstruction has no wall-clock Instant;
                    // the economic effect is kept: suppression of
                    // emissions and discarding of subsequent deltas.)
                    self.crossed_pending.remove(&token);
                    self.needs_snapshot.insert(token.clone());
                    let c = cid.clone().or_else(|| self.token2cid.get(&token).cloned());
                    self.mark_segment(ev, &token, c);
                    self.diag(
                        ev,
                        format!("persistently crossed book (token {token}) — resync"),
                    );
                } else {
                    self.crossed_pending.insert(token.clone());
                }
                continue; // no emission (production also continues here)
            }
            self.crossed_pending.remove(&token);
            self.emit(ev, &token, cid.as_deref(), false);
        }
    }

    /// Emits the market book — mirror of `try_emit_snapshot`
    /// (registry gates included). Pushes the book to the tape if it emits.
    fn emit(&mut self, ev: &ResearchEvent, token: &str, cid_hint: Option<&str>, force: bool) {
        let Some(cid) = cid_hint
            .map(str::to_string)
            .or_else(|| self.token2cid.get(token).cloned())
        else {
            self.diag(
                ev,
                format!("book for {token} with no known cid (no meta) — discarded"),
            );
            return;
        };
        let Some(meta) = self.metas.get(&cid) else {
            // Without meta there is no yes/no/slug/clear_at — fabricating the book would be
            // inventing data. Discarded + segment marked.
            let c = Some(cid.clone());
            self.mark_segment(ev, token, c);
            self.diag(ev, format!("book for {cid} without MarketMeta — discarded"));
            return;
        };
        if self.needs_snapshot.contains(&meta.yes) || self.needs_snapshot.contains(&meta.no) {
            // Frozen half — do not emit.
            return;
        }
        let (Some(yb), Some(nb)) = (self.books.get(&meta.yes), self.books.get(&meta.no)) else {
            // Production also returns here: an emit requires BOTH halves.
            self.books_missing_half += 1;
            return;
        };
        let top = [
            yb.bids.first().map(|l| l.price),
            yb.asks.first().map(|l| l.price),
            nb.bids.first().map(|l| l.price),
            nb.asks.first().map(|l| l.price),
        ];
        if !force && self.last_top.get(&cid).is_some_and(|t| *t == top) {
            self.books_top_equal += 1;
            return;
        }
        self.last_top.insert(cid.clone(), top);
        let tick = effective_tick(&top, meta.tick_size);
        let (yf, nf) = if self.opts.tape_window_secs > 0 {
            let w = self.opts.tape_window_secs;
            let now = ev.avail_ts_ms;
            (
                self.flow
                    .entry(meta.yes.clone())
                    .or_insert_with(|| Flow {
                        trades: VecDeque::new(),
                    })
                    .window(now, w),
                self.flow
                    .entry(meta.no.clone())
                    .or_insert_with(|| Flow {
                        trades: VecDeque::new(),
                    })
                    .window(now, w),
            )
        } else {
            (None, None)
        };
        let snap = BookSnapshot {
            market_condition_id: cid,
            yes: TokenBook {
                token_id: meta.yes.clone(),
                bids: HalfBook {
                    levels: yb.bids.clone(),
                },
                asks: HalfBook {
                    levels: yb.asks.clone(),
                },
                flow_net: yf.map(|(n, _)| n),
                flow_total: yf.map(|(_, t)| t),
            },
            no: TokenBook {
                token_id: meta.no.clone(),
                bids: HalfBook {
                    levels: nb.bids.clone(),
                },
                asks: HalfBook {
                    levels: nb.asks.clone(),
                },
                flow_net: nf.map(|(n, _)| n),
                flow_total: nf.map(|(_, t)| t),
            },
            market_open_ts: meta.market_open_ts,
            clear_at: meta.clear_at,
            neg_risk: meta.neg_risk,
            slug: meta.slug.clone(),
            reference_price: meta.reference_price,
            fee_rate_bps: meta.fee_rate_bps,
            parabolic_fee_rate: meta.parabolic_fee_rate,
            tick_size: Some(tick),
            recv_instant: None, // set by the engine to `recv_ms` in replay
        };
        self.out_books.push(TapeBook {
            recv_ms: ev.recv_ts_ms,
            avail_ms: Some(ev.avail_ts_ms),
            tape_seq: Some(ev.local_seq),
            snap,
        });
    }

    /// Discard of ONE delta entry: global counter (kept: the report always
    /// declares it) + attribution per token/cid for the scope counters.
    fn discard(&mut self, token: &str, cid: &Option<String>) {
        self.discarded_deltas += 1;
        *self
            .discarded_by_token
            .entry(token.to_string())
            .or_insert(0) += 1;
        match cid {
            Some(c) => *self.discarded_by_cid.entry(c.clone()).or_insert(0) += 1,
            None => {
                *self
                    .discarded_no_cid_by_token
                    .entry(token.to_string())
                    .or_insert(0) += 1
            }
        }
    }

    /// `last_trade_price` — flow only (never emits a book: the twin `price_change`
    /// already does it — do not count twice).
    fn on_trade(&mut self, ev: &ResearchEvent) {
        if self.opts.tape_window_secs == 0 {
            return;
        }
        let msg: WsMessage = match ev.payload_as() {
            Ok(m) => m,
            Err(e) => {
                self.diag(ev, format!("unreadable Trade payload: {e}"));
                return;
            }
        };
        let Some(token) = msg.asset_id.as_deref().filter(|s| !s.is_empty()) else {
            return;
        };
        let (Some(size_s), Some(side_s)) = (msg.size.as_deref(), msg.side.as_deref()) else {
            return;
        };
        let Ok(size) = Decimal::from_str_exact(size_s) else {
            return;
        };
        let signed = if side_s.eq_ignore_ascii_case("BUY") {
            size
        } else if side_s.eq_ignore_ascii_case("SELL") {
            -size
        } else {
            return;
        };
        self.flow
            .entry(token.to_string())
            .or_insert_with(|| Flow {
                trades: VecDeque::new(),
            })
            .trades
            .push_back((ev.avail_ts_ms, signed, size));
    }

    /// Tick of the underlying → `TapeSpot` at `avail_ts_ms`.
    fn on_price_tick(&mut self, ev: &ResearchEvent) {
        let price = ev.payload.get("price").and_then(|v| {
            v.as_str()
                .and_then(|s| Decimal::from_str_exact(s).ok())
                .or_else(|| v.as_f64().and_then(Decimal::from_f64_retain))
        });
        let (Some(asset), Some(p)) = (ev.asset.clone(), price) else {
            self.diag(ev, "PriceTick without asset or unreadable price".to_string());
            return;
        };
        self.out_spots.push(TapeSpot {
            ts_ms: ev.avail_ts_ms,
            tape_seq: Some(ev.local_seq),
            asset,
            price: p,
        });
    }

    /// Official resolution — the payload is `MarketOutcome` + extras.
    fn on_resolution(&mut self, ev: &ResearchEvent) {
        let outcome: MarketOutcome = match ev.payload_as() {
            Ok(o) => o,
            Err(e) => {
                self.diag(ev, format!("unreadable Resolution payload: {e}"));
                return;
            }
        };
        let Some(cid) = ev.market_cid.clone() else {
            self.diag(ev, "Resolution without market_cid".to_string());
            return;
        };
        self.out_res.push(TapeResolution {
            avail_ms: ev.avail_ts_ms,
            tape_seq: Some(ev.local_seq),
            cid,
            outcome,
            source: ev.source,
        });
    }

    fn into_tape(self) -> Tape {
        // Quality aggregate per token/cid — the report scope is the cids with an
        // observed meta; `marks_by_cid` keeps the attribution of each emitted mark
        // for the env's filtering.
        let mut marks_by_cid: HashMap<String, u64> = HashMap::new();
        let mut marks_without_cid = 0u64;
        for q in &self.out_quality {
            match &q.cid {
                Some(c) => *marks_by_cid.entry(c.clone()).or_insert(0) += 1,
                None => marks_without_cid += 1,
            }
        }
        Tape {
            books: self.out_books,
            spots: self.out_spots,
            resolutions: self.out_res,
            quality: self.out_quality,
            quality_stats: TapeQualityStats {
                meta_cids: self.metas.keys().cloned().collect(),
                token2cid: self.token2cid,
                deltas_by_cid: self.deltas_by_cid,
                deltas_no_cid_by_token: self.deltas_no_cid_by_token,
                discarded_deltas_by_token: self.discarded_by_token,
                discarded_deltas_by_cid: self.discarded_by_cid,
                discarded_deltas_no_cid_by_token: self.discarded_no_cid_by_token,
                tokens_with_gap: self.tokens_with_gap,
                marks_by_cid,
                marks_without_cid,
                discarded_deltas: self.discarded_deltas,
                malformed: 0, // set by `ResearchTape::from_events`
                agent: self.agent_events,
                missing_half: self.books_missing_half,
                books_top_equal: self.books_top_equal,
            },
            source: TapeSource::ResearchEvents,
        }
    }
}

/// Same as `dedup_levels_by_price` in `handle.rs` (keeps the last occurrence of
/// the price) — a local replica so as not to depend on a private helper.
fn dedup_levels_by_price(levels: Vec<Level>) -> Vec<Level> {
    let mut by_price: HashMap<Decimal, Decimal> = HashMap::with_capacity(levels.len());
    let mut order: Vec<Decimal> = Vec::with_capacity(levels.len());
    for l in levels {
        if !by_price.contains_key(&l.price) {
            order.push(l.price);
        }
        by_price.insert(l.price, l.size);
    }
    order
        .into_iter()
        .map(|price| Level {
            price,
            size: by_price[&price],
        })
        .collect()
}

/// `apply_price_change_entry` from `handle.rs` — same rule (BUY→bids,
/// SELL→asks, size=0 removes the level, ordered insertion).
fn apply_price_change_entry(state: &mut TokenBookLevels, entry: &RawPriceChange) {
    let Ok(price) = Decimal::from_str_exact(&entry.price) else {
        return;
    };
    let Ok(size) = Decimal::from_str_exact(&entry.size) else {
        return;
    };
    let is_bid = entry.side.eq_ignore_ascii_case("BUY");
    let levels = if is_bid {
        &mut state.bids
    } else if entry.side.eq_ignore_ascii_case("SELL") {
        &mut state.asks
    } else {
        return;
    };
    let cmp = |l: &Level| {
        if is_bid {
            l.price.cmp(&price).reverse()
        } else {
            l.price.cmp(&price)
        }
    };
    if size.is_zero() {
        if let Ok(pos) = levels.binary_search_by(cmp) {
            levels.remove(pos);
        }
    } else if let Ok(pos) = levels.binary_search_by(cmp) {
        levels[pos].size = size;
    } else {
        let insert_pos = levels.partition_point(|l| {
            if is_bid {
                l.price > price
            } else {
                l.price < price
            }
        });
        levels.insert(insert_pos, Level { price, size });
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Part 2 — Incremental episodic environment
// ════════════════════════════════════════════════════════════════════════════

/// Config of the episodic environment.
#[derive(Debug, Clone)]
pub struct EnvConfig {
    /// Parameters of the `ReplayEngine` (seed, balance, latencies, risk…).
    pub params: ReplayParams,
    /// Quality-flag mask that aborts the episode with
    /// `EndReason::DataQualityAbort`. Default: `GAP` (a data gap makes the
    /// observation unreliable; `RESYNC` alone does not abort — the data was
    /// recovered by the snapshot that follows it).
    pub abort_on_quality: QualityFlags,
    /// Maximum number of steps of the episode (truncation horizon).
    /// `None` = until the end of the tape.
    pub max_steps: Option<u64>,
    /// Cap on `size` per Buy/Sell action — actions above it are rejected as
    /// illegal WITHOUT touching the pipeline (the mask does not replace the
    /// RiskGuard: it is a rule of the environment).
    pub max_action_size: Decimal,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            params: ReplayParams::default(),
            abort_on_quality: QualityFlags::GAP,
            max_steps: None,
            max_action_size: Decimal::new(10_000, 0),
        }
    }
}

/// Internal agent of the environment: a PASSIVE `Strategy` that emits no intents
/// of its own. It captures exactly what the pipeline delivers to it — the last
/// observed book and its vol (the agent's causal observation) and the fills/
/// rejects/acks of the window — under `StrategyId::Follower` (the variant meant
/// for external entities that reuse the attribution infrastructure, not a real strategy).
pub struct EnvAgent {
    /// Last book that reached `on_book` per market (the causal observation —
    /// never the omniscient state of the venue).
    pub last_book: HashMap<String, BookSnapshot>,
    /// Last vol seen per asset.
    pub last_vol: HashMap<String, VolSnapshot>,
    /// Fills/rejects/acks received since the env's last `drain`.
    pub window_fills: Vec<Fill>,
    /// Total books observed (diagnostics).
    pub books_seen: u64,
}

impl EnvAgent {
    pub fn new() -> Self {
        Self {
            last_book: HashMap::new(),
            last_vol: HashMap::new(),
            window_fills: Vec::new(),
            books_seen: 0,
        }
    }
}

impl Default for EnvAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::strategy::Strategy for EnvAgent {
    fn id(&self) -> StrategyId {
        StrategyId::Follower
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
        self.last_book
            .insert(book.market_condition_id.clone(), book.clone());
        self.last_vol.insert(vol.asset.clone(), vol.clone());
        self.books_seen += 1;
        Vec::new() // the env injects the actions via push_agent_intent
    }

    fn on_fill(&mut self, fill: &Fill) {
        self.window_fills.push(fill.clone());
    }

    fn requires_oracle(&self) -> bool {
        false
    }
}

/// State of an episode in progress.
struct Episode {
    id: String,
    step_idx: u64,
    /// If `Some`, the episode is restricted to that `market_cid` (decision points
    /// of other markets advance the clock but are not the agent's observation)
    /// and its resolution ends the episode.
    market_filter: Option<String>,
    /// Equity/fees accumulated at the end of the previous step (reward deltas).
    equity_prev: Decimal,
    fees_prev: Decimal,
    /// `local_seq` of the events that form the observation of the CURRENT decision
    /// point (the ones the next `step`'s `StepRecord` declares).
    obs_seqs: Vec<u64>,
    /// Structural quality of the current observation (window before the decision point).
    obs_quality: QualityFlags,
    /// Seq buffer accumulated while advancing to the NEXT decision point
    /// (observable events of the current window).
    window_seq: Vec<u64>,
    /// Accumulated quality of the current window.
    window_quality: QualityFlags,
    /// Quality of the window restricted to the episode's SCOPE — with a
    /// `market_filter` only the marks whose `cid` is the episode's; without a
    /// filter, all items (the whole tape is the scope). It is the ONLY one that
    /// can trigger `DataQualityAbort`: a foreign mark stays recorded in
    /// `window_quality`/`obs_quality` but never aborts.
    window_quality_in_scope: QualityFlags,
    /// In-scope part of `obs_quality` (frozen window before the decision point).
    obs_quality_in_scope: QualityFlags,
    /// Entries of `engine.quality_log()` already drained into `window_quality`
    /// (drain by index — two marks with the same ts are never lost).
    quality_seen: usize,
    /// End of the episode (`Some` ⇒ rejects later `step`s).
    end: Option<EndReason>,
}

/// Incremental episodic environment over the real `ReplayEngine`.
///
/// One `ResearchEnv` = one tape; each `reset` builds a NEW engine
/// (fresh `:memory:` StateStore — never the operational DB) and seeds the
/// filtered tape; `step` drives `step_once` until the next decision point. The
/// replay clock is process-global — one env per process (the `hrf_research_env`
/// binary guarantees it by construction).
pub struct ResearchEnv {
    tape: Tape,
    cfg: EnvConfig,
    engine: Option<ReplayEngine>,
    agent: EnvAgent,
    ep: Option<Episode>,
    /// Per-asset spot series for `extract_features` (same history the shadow keeps —
    /// returns/vol of the causal observation).
    series: HashMap<String, crate::model_observer::SpotSeries>,
    /// Declared fidelity of the tape (for the protocol's Hello).
    fidelity: &'static str,
}

/// Result of `step`: the `StepRecord` of the applied step and, if the episode
/// continues, the next causal observation + legal actions.
pub struct EnvStep {
    pub step: StepRecord,
    pub observation: Option<EnvObservation>,
    pub legal_actions: Vec<AgentAction>,
}

impl ResearchEnv {
    /// Builds the environment over a loaded `ResearchTape`.
    pub fn new(tape: ResearchTape, cfg: EnvConfig) -> Self {
        // Declared fidelity: "certified" only with a clean reconstruction (zero
        // diagnostics, zero discarded deltas) and with no settlement proxy enabled.
        // Any degradation — however small — is declared "reduced", never silenced.
        let degraded = cfg.params.settlement_proxy_diagnostic
            || tape.discarded_deltas > 0
            || !tape.diagnostics.is_empty();
        Self {
            tape: tape.tape,
            cfg,
            engine: None,
            agent: EnvAgent::new(),
            ep: None,
            series: HashMap::new(),
            fidelity: if degraded { "reduced" } else { "certified" },
        }
    }

    /// Declared fidelity of the environment (protocol's Hello).
    pub fn fidelity(&self) -> &'static str {
        self.fidelity
    }

    /// Summary `quality` of the whole tape for the protocol's Hello — the declared
    /// scope is ALL observed metas (the episode's concrete filter is set by
    /// `reset.market_cid`; there is no episode yet here). The counted marks are those
    /// of the loaded `tape.quality` (not yet processed by the engine).
    pub fn quality_summary(&self) -> QualitySummary {
        let mut resyncs = 0u64;
        let mut gap_cids: HashSet<String> = HashSet::new();
        for q in &self.tape.quality {
            if q.flag.contains(QualityFlags::RESYNC) {
                resyncs += 1;
            }
            if q.flag.contains(QualityFlags::GAP) {
                if let Some(c) = &q.cid {
                    gap_cids.insert(c.clone());
                }
            }
        }
        self.tape
            .quality_stats
            .summary(self.tape.quality.len() as u64, resyncs, &gap_cids)
    }

    /// Quality scope announced in the Hello — markets whose stream the tape observed
    /// and their time window. `market_cid` reflects the current episode's filter
    /// (`None` before the reset).
    pub fn quality_scope(&self) -> QualityScope {
        let mut min_ms: Option<i64> = None;
        let mut max_ms: Option<i64> = None;
        let mut acc = |ts: i64| {
            min_ms = Some(min_ms.map_or(ts, |m: i64| m.min(ts)));
            max_ms = Some(max_ms.map_or(ts, |m: i64| m.max(ts)));
        };
        for b in &self.tape.books {
            acc(b.avail_ms.unwrap_or(b.recv_ms));
        }
        for s in &self.tape.spots {
            acc(s.ts_ms);
        }
        for r in &self.tape.resolutions {
            acc(r.avail_ms);
        }
        for q in &self.tape.quality {
            acc(q.ts_ms);
        }
        QualityScope {
            market_cid: self.ep.as_ref().and_then(|e| e.market_filter.clone()),
            markets: self.tape.quality_stats.meta_cids.iter().cloned().collect(),
            window_ms: min_ms.zip(max_ms).map(|(a, b)| [a, b]),
        }
    }

    /// `true` if there is an open episode.
    pub fn in_episode(&self) -> bool {
        self.ep.is_some()
    }

    /// `episode_id` of the open episode (for the protocol's `Closed`).
    pub fn episode_id(&self) -> Option<&str> {
        self.ep.as_ref().map(|e| e.id.as_str())
    }

    /// Starts an episode: fresh engine + seed of the tape (filtered by `market_cid`
    /// if requested) and advances to the first decision point.
    /// Returns the first causal observation + legal actions.
    pub async fn reset(
        &mut self,
        episode_id: &str,
        seed: u64,
        market_cid: Option<&str>,
    ) -> anyhow::Result<(EnvObservation, Vec<AgentAction>)> {
        // Close the previous episode if it was left open (drains the engine).
        if let Some(mut e) = self.engine.take() {
            let _ = e.shutdown().await;
        }
        self.agent = EnvAgent::new();
        self.series.clear();
        self.ep = None;

        // Filter by market: the tape keeps the market's metas/resolutions and the
        // books/spots whose cid (or asset of the market) applies.
        let tape = filter_tape(&self.tape, market_cid);

        let mut params = self.cfg.params.clone();
        params.seed = seed;
        let state = crate::state::StateStore::open_memory(Mode::Paper)
            .await
            .context("env: ephemeral StateStore")?;
        let mut engine = ReplayEngine::new(params, state)
            .await
            .context("env: ReplayEngine::new")?;
        engine.seed_tape(tape).context("env: seed_tape")?;
        self.engine = Some(engine);

        self.ep = Some(Episode {
            id: episode_id.to_string(),
            step_idx: 0,
            market_filter: market_cid.map(str::to_string),
            equity_prev: Decimal::ZERO,
            fees_prev: Decimal::ZERO,
            obs_seqs: Vec::new(),
            obs_quality: QualityFlags::NONE,
            window_seq: Vec::new(),
            window_quality: QualityFlags::NONE,
            window_quality_in_scope: QualityFlags::NONE,
            obs_quality_in_scope: QualityFlags::NONE,
            quality_seen: 0,
            end: None,
        });

        // Advance to the first decision point (or end of tape).
        match self.advance_to_decision().await? {
            Some(o) => {
                // Base equity/fees for the first reward.
                let (eq, fees) = self.equity_and_fees().await;
                let ep = self.ep.as_mut().expect("episode");
                ep.equity_prev = eq;
                ep.fees_prev = fees;
                let legal = self.legal_actions().await;
                Ok((o, legal))
            }
            None => {
                bail!("env: the tape produces no decision point (no observable books)")
            }
        }
    }

    /// Applies the agent's action at the current decision point and advances to the
    /// next. `Hold` only advances. Returns the `StepRecord` and the next observation
    /// (`None` if the step was terminal).
    ///
    /// Step semantics: the action responds to the observation of the CURRENT decision
    /// point (`decision_ts_ms` = ts of the last observed book). The intent enters the
    /// pipeline at that instant (+ the profile's `decision` stage) and the clock advances
    /// to the next point — the fills/acks/rejects the action produces may materialize in
    /// THIS step or in later ones, as in production (ack ≠ fill).
    pub async fn step(&mut self, action: AgentAction) -> anyhow::Result<EnvStep> {
        if self.ep.as_ref().is_none_or(|e| e.end.is_some()) {
            bail!("env: step after the episode ended or without reset");
        }
        let (step_idx, obs_seqs, obs_quality, market_filter) = {
            let ep = self.ep.as_ref().expect("episode");
            (
                ep.step_idx,
                ep.obs_seqs.clone(),
                ep.obs_quality,
                ep.market_filter.clone(),
            )
        };
        let decision_ts = self.engine.as_ref().expect("engine").clock_now_ms();
        // The signal window of the step starts clean: the fills/acks that arrived
        // during the PREVIOUS advance were already classified in their `StepRecord`
        // (a signal never counts twice).
        self.agent.window_fills.clear();
        // Legal actions AT the decision point — those the StepRecord declares
        // (computed BEFORE applying the action).
        let legal_prev = self.legal_actions().await;

        // 1) Environment legality + translation to a real `OrderIntent`. An illegal
        //    action is recorded as `Rejected` — never a protocol exception nor a
        //    silent bypass.
        let mut action_outcome: Option<ActionOutcome> = None;
        // Live orders BEFORE the dispatch: a fill whose `order_id` was not live before
        // belongs to THIS action (fills of earlier resting orders that the tick wakes
        // are NOT attributed to the new intent — distinguish own from foreign by
        // `order_id`, not by `price`: the fill carries the executed fee-inclusive price).
        let mut orders_before = std::collections::HashSet::new();
        match self.intent_for(&action).await {
            Ok(Some(intent)) => {
                if let Some(live) = self
                    .engine
                    .as_ref()
                    .expect("engine")
                    .sim()
                    .open_gtd_orders()
                    .await
                {
                    orders_before = live.into_iter().map(|o| o.order_id).collect();
                }
                self.engine
                    .as_mut()
                    .expect("engine")
                    .push_agent_intent(intent);
            }
            Ok(None) => {} // Hold / nothing to dispatch
            Err(e) => {
                action_outcome = Some(ActionOutcome::Rejected {
                    reason: e.to_string(),
                    event_seqs: vec![],
                });
            }
        }

        // 2) Advance the tape to the next decision point (the fills/acks of the
        //    action land in `agent.window_fills`).
        let next = self.advance_to_decision().await?;

        // 3) Action outcome: if the env already rejected it, it stays; otherwise
        //    it is classified by what the pipeline materialized in the window.
        if action_outcome.is_none() && !matches!(action, AgentAction::Hold) {
            action_outcome = Some(self.classify_outcome(&action, &orders_before).await);
        }

        // 4) Reward: Δequity identity with the fee separated.
        let (equity, fees) = self.equity_and_fees().await;
        let (price_component, fee_component) = {
            let ep = self.ep.as_ref().expect("episode");
            (
                (equity + fees) - (ep.equity_prev + ep.fees_prev),
                -(fees - ep.fees_prev),
            )
        };
        let reward = RewardBreakdown::new(price_component, fee_component, Decimal::ZERO);
        {
            let ep = self.ep.as_mut().expect("episode");
            ep.equity_prev = equity;
            ep.fees_prev = fees;
        }

        // 5) Terminality: quality of the step's window, risk kill,
        //    resolution of the episode's market, end of tape or step horizon —
        //    `terminal` and `truncated` are never confused.
        let (step_quality, step_quality_in_scope) = {
            let ep = self.ep.as_ref().expect("episode");
            (
                obs_quality.union(ep.window_quality),
                ep.obs_quality_in_scope.union(ep.window_quality_in_scope),
            )
        };
        let mut end_reason: Option<EndReason> = None;
        {
            let engine = self.engine.as_ref().expect("engine");
            // Quality abort is SCOPED — with market_filter only the marks of the
            // episode's market abort; foreign ones stay declared in `step.quality`
            // (recorded, not hidden) but do not stop the episode.
            if step_quality_in_scope.0 & self.cfg.abort_on_quality.0 != 0
                && !self.cfg.abort_on_quality.is_clean()
            {
                end_reason = Some(EndReason::DataQualityAbort);
            } else if engine.risk().is_killed() {
                end_reason = Some(EndReason::RiskHalt);
            } else if let Some(cid) = &market_filter {
                if engine.market_settled(cid) {
                    end_reason = Some(EndReason::MarketResolved);
                }
            }
            if end_reason.is_none() && next.is_none() {
                // Without more decision points: `MarketResolved` only if the episode's
                // market actually settled (or, without a filter, nothing pending is left).
                // A market still open when the tape runs out is a horizon — never MarketResolved.
                end_reason = Some(match &market_filter {
                    Some(cid) if engine.market_settled(cid) => EndReason::MarketResolved,
                    Some(_) => EndReason::HorizonReached,
                    None if engine.settlement_pending().is_empty() => EndReason::MarketResolved,
                    None => EndReason::HorizonReached,
                });
            }
        }
        if end_reason.is_none() && self.cfg.max_steps.is_some_and(|m| step_idx + 1 >= m) {
            end_reason = Some(EndReason::HorizonReached);
        }
        let terminal = end_reason.is_some();

        // 6) StepRecord of the step.
        let inventory = self.inventory().await;
        let step = StepRecord {
            schema_version: SCHEMA_VERSION,
            step_idx,
            observation_seqs: obs_seqs,
            decision_ts_ms: decision_ts,
            legal_actions: legal_prev,
            proposed_action: Some(action.clone()),
            action_outcome,
            inventory,
            reward,
            terminal,
            end_reason,
            quality: step_quality,
        };
        debug_assert!(step.validate().is_ok());
        {
            let ep = self.ep.as_mut().expect("episode");
            ep.step_idx += 1;
            ep.end = end_reason.or(ep.end);
            // The observation window is consumed by the StepRecord; `window_seq`/`window_quality`
            // already accumulate toward the obs of the NEXT point (populated by `advance_to_decision`).
        }

        let legal_next = if terminal {
            vec![]
        } else {
            self.legal_actions().await
        };
        Ok(EnvStep {
            step,
            observation: if terminal { None } else { next },
            legal_actions: legal_next,
        })
    }

    /// Legal actions at the current decision point (discrete, bounded mask):
    ///
    /// - `Hold` always.
    /// - `Buy` per token of the LAST observed book of each market of the
    ///   episode, at best ask (aggressive limit that crosses) with
    ///   `size = min(max_action_size, ask depth)` — buying without cash is
    ///   rejected by the real RiskGuard, not by the mask.
    /// - `Sell` only over tokens with an OPEN POSITION in the ledger
    ///   (`list_open_positions`), at best bid with `size` capped by the
    ///   position — the env does not offer short sells that the sim would reject.
    /// - `Cancel` for each live GTD order of the agent at the venue.
    ///
    /// The mask is indicative and is fixed BEFORE dispatch — the real
    /// RiskGuard/executor remains the final arbiter (a legal Buy can come
    /// back `Rejected` for exposure, cooldown or kill).
    pub async fn legal_actions(&self) -> Vec<AgentAction> {
        let mut v = vec![AgentAction::Hold];
        let Some(engine) = &self.engine else {
            return v;
        };
        let Some(ep) = &self.ep else {
            return v;
        };
        if ep.end.is_some() {
            return v;
        }
        let filter = ep.market_filter.clone();
        // Sell only against really open positions (same regime as `inventory`:
        // the ledger, not the venue's internal state).
        let positions = engine
            .state()
            .list_open_positions()
            .await
            .unwrap_or_default();
        let mut shares_per_token: HashMap<String, Decimal> = HashMap::new();
        for p in &positions {
            if filter.as_ref().is_none_or(|f| &p.market_cid == f) {
                *shares_per_token.entry(p.token_id.clone()).or_default() += p.size;
            }
        }
        let max_size = self.cfg.max_action_size;
        for (cid, book) in &self.agent.last_book {
            if filter.as_ref().is_some_and(|f| cid != f) {
                continue;
            }
            for tb in [&book.yes, &book.no] {
                if let Some(ask) = tb.best_ask() {
                    // Cap at the visible depth of the level (buying more than the ask is
                    // no more informative, and the excess would be rejected by the venue).
                    let depth = tb.asks.levels.first().map(|l| l.size).unwrap_or(max_size);
                    v.push(AgentAction::Buy {
                        token_id: tb.token_id.clone(),
                        price: ask,
                        size: max_size.min(depth),
                    });
                }
                let held = shares_per_token
                    .get(&tb.token_id)
                    .copied()
                    .unwrap_or_default();
                if held > Decimal::ZERO {
                    if let Some(bid) = tb.best_bid() {
                        v.push(AgentAction::Sell {
                            token_id: tb.token_id.clone(),
                            price: bid,
                            size: held.min(max_size),
                        });
                    }
                }
            }
        }
        // Cancels of the agent's live orders (GTD resting in the sim).
        if let Some(orders) = engine.sim().open_gtd_orders().await {
            for o in orders {
                v.push(AgentAction::Cancel {
                    order_id: o.order_id,
                });
            }
        }
        v
    }

    /// Closes the episode: finalizes the engine (settlement sweep, drain) and
    /// shuts down the executor task. Returns the run's outcome (digest +
    /// reconciliation) for the protocol's `Closed`.
    pub async fn close(&mut self) -> anyhow::Result<ReplayOutcome> {
        let mut engine = self.engine.take().context("env: no reset")?;
        let out = engine.finalize(&mut self.agent).await?;
        engine.shutdown().await?;
        self.ep = None;
        Ok(out)
    }

    // ── internals ─────────────────────────────────────────────────────────

    /// Advances the virtual queue until the next `StepMark::Decision` of the
    /// episode's market (or of any market if there is no filter), accumulating in
    /// `window_seq`/`window_quality` the observable events of the window. On reaching
    /// the point: freezes the window into `obs_seqs`/`obs_quality` (what the next
    /// `StepRecord` declares) and builds the causal observation — never the omniscient
    /// state of the venue. `None` if the tape ran out without further decisions.
    async fn advance_to_decision(&mut self) -> anyhow::Result<Option<EnvObservation>> {
        loop {
            let mark = {
                let engine = self.engine.as_mut().expect("engine");
                engine.step_once(&mut self.agent).await?
            };
            // Drain by index of the quality log (marks of the just-processed event
            // and of any intermediate event).
            {
                let engine = self.engine.as_ref().expect("engine");
                let ep = self.ep.as_mut().expect("episode");
                let log = engine.quality_log();
                for (_, cid, f) in &log[ep.quality_seen..] {
                    ep.window_quality = ep.window_quality.union(*f);
                    // Scope: without a filter every mark belongs to the episode;
                    // with a filter only those of the `market_cid` (a mark without
                    // cid is not attributable to the filtered market → it does not abort,
                    // but it stays recorded in `window_quality`).
                    if ep
                        .market_filter
                        .as_ref()
                        .is_none_or(|mf| cid.as_deref() == Some(mf.as_str()))
                    {
                        ep.window_quality_in_scope = ep.window_quality_in_scope.union(*f);
                    }
                }
                ep.quality_seen = log.len();
            }
            let Some(mark) = mark else {
                return Ok(None);
            };
            match mark {
                StepMark::Decision {
                    ts_ms,
                    tape_seq,
                    cid,
                    ..
                } => {
                    {
                        let ep = self.ep.as_mut().expect("episode");
                        if let Some(s) = tape_seq {
                            ep.window_seq.push(s);
                        }
                        if ep.market_filter.as_ref().is_some_and(|f| &cid != f) {
                            continue; // decision point of ANOTHER market
                        }
                        // Freezes the window as the observation of the point.
                        ep.obs_seqs = std::mem::take(&mut ep.window_seq);
                        ep.obs_quality = ep.window_quality;
                        ep.window_quality = QualityFlags::NONE;
                        ep.obs_quality_in_scope = ep.window_quality_in_scope;
                        ep.window_quality_in_scope = QualityFlags::NONE;
                    }
                    // Causal observation: what the agent has just seen —
                    // features via the SAME extractor as the shadow.
                    let book = self.agent.last_book.get(&cid).cloned();
                    let obs = self.observation(ts_ms, &cid, book).await;
                    return Ok(Some(obs));
                }
                StepMark::Tape {
                    tape_seq,
                    observable,
                    ..
                } => {
                    if observable {
                        if let Some(s) = tape_seq {
                            self.ep.as_mut().expect("episode").window_seq.push(s);
                        }
                    }
                }
                StepMark::Advance { .. } => {}
            }
        }
    }

    /// Builds the causal observation at the decision point: features via
    /// `model_observer::extract_features` (same extractor as the shadow) over the LAST
    /// observed book — never labels nor future data.
    async fn observation(
        &mut self,
        ts_ms: i64,
        cid: &str,
        book: Option<BookSnapshot>,
    ) -> EnvObservation {
        let mut features: BTreeMap<String, Option<f64>> = BTreeMap::new();
        if let Some(b) = &book {
            let asset = crate::oracle::asset_from_slug(&b.slug).unwrap_or_default();
            let vol = self.agent.last_vol.get(&asset).cloned().unwrap_or_default();
            let series = self
                .series
                .entry(asset.clone())
                .or_insert_with(crate::model_observer::SpotSeries::new);
            let fm = crate::model_observer::extract_features(b, &vol, series, ts_ms);
            features = fm.into_iter().collect();
            // The `own_*` fields of the extractor are 0 for the shadow; in the env they
            // reflect the agent's real state (live orders at the venue +
            // the ledger's net shares).
            let engine = self.engine.as_ref().expect("engine");
            let open_orders = engine
                .sim()
                .open_gtd_orders()
                .await
                .map(|v| v.len() as f64)
                .unwrap_or(0.0);
            features.insert("own_open_orders".into(), Some(open_orders));
        }
        let ep = self.ep.as_ref().expect("episode");
        EnvObservation {
            schema_version: SCHEMA_VERSION,
            decision_ts_ms: ts_ms,
            market_cid: Some(cid.to_string()),
            features,
            inventory: self.inventory().await,
            observation_seqs: ep.obs_seqs.clone(),
            quality: ep.obs_quality,
        }
    }

    /// Observable inventory: free cash + reserved + ledger positions
    /// (`list_open_positions` — same availability regime as the rest of the
    /// pipeline, not the venue's internal state).
    async fn inventory(&self) -> InventorySnapshot {
        let Some(engine) = &self.engine else {
            return InventorySnapshot {
                cash_usdc: Decimal::ZERO,
                reserved_usdc: Decimal::ZERO,
                positions: vec![],
            };
        };
        let accounting = engine.sim().accounting();
        let positions = engine
            .state()
            .list_open_positions()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|p| PositionSnapshot {
                token_id: p.token_id,
                market_cid: p.market_cid,
                shares: p.size,
                avg_price: p.avg_price,
            })
            .collect();
        InventorySnapshot {
            // `balance` includes the reserved amount — free cash is the difference.
            cash_usdc: accounting.balance - accounting.reserved_buy,
            reserved_usdc: accounting.reserved_buy,
            positions,
        }
    }

    /// Conservative `equity` + accumulated fees (for the reward delta).
    async fn equity_and_fees(&self) -> (Decimal, Decimal) {
        let Some(engine) = &self.engine else {
            return (Decimal::ZERO, Decimal::ZERO);
        };
        let accounting = engine.sim().accounting();
        let mut equity = accounting.balance;
        // Conservative mark-to-market per ledger position: walks the bids of the last
        // OBSERVED book of the token with the available depth; shares without visible depth are worth 0.
        if let Ok(positions) = engine.state().list_open_positions().await {
            for p in positions {
                let book = self.agent.last_book.get(&p.market_cid);
                equity += exit_value(&p.token_id, p.size, book);
            }
        }
        let fees: Decimal = engine.sim().executions().iter().map(|e| e.fee_usdc).sum();
        (equity, fees)
    }

    /// Translates the agent's action into a real pipeline `OrderIntent`.
    /// Returns `Ok(None)` for `Hold`; `Err(reason)` for an illegal action
    /// (the caller records it as `Rejected` — it never touches money).
    async fn intent_for(&self, action: &AgentAction) -> anyhow::Result<Option<OrderIntent>> {
        let engine = self.engine.as_ref().expect("engine");
        match action {
            AgentAction::Hold => Ok(None),
            AgentAction::Buy {
                token_id,
                price,
                size,
            }
            | AgentAction::Sell {
                token_id,
                price,
                size,
            } => {
                if *price <= Decimal::ZERO || *price >= Decimal::ONE {
                    bail!("price out of range (0,1): {price}");
                }
                if *size <= Decimal::ZERO {
                    bail!("size not positive: {size}");
                }
                if *size > self.cfg.max_action_size {
                    bail!(
                        "size {size} exceeds max_action_size {}",
                        self.cfg.max_action_size
                    );
                }
                let Some(cid) = engine.token_cid(token_id) else {
                    bail!("unknown token_id in the tape: {token_id}");
                };
                if let Some(f) = &self.ep.as_ref().expect("episode").market_filter {
                    if &cid != f {
                        bail!("action on a market outside the episode ({cid} != {f})");
                    }
                }
                let Some((_slug, clear_at, neg_risk, _yes)) = engine.market_meta(&cid) else {
                    bail!("market {cid} has no meta in the tape");
                };
                let buy = matches!(action, AgentAction::Buy { .. });
                Ok(Some(OrderIntent {
                    market_cid: cid,
                    token_id: token_id.clone(),
                    side: if buy {
                        OrderSide::Buy
                    } else {
                        OrderSide::Sell { size: Some(*size) }
                    },
                    price: *price,
                    size: *size,
                    clear_at,
                    signal: Decimal::ZERO,
                    neg_risk,
                    // GTD limit resting: crossing → immediate fill (Executed);
                    // stays on the book → Accepted and cancelable.
                    limit_price: Some(*price),
                    second_leg: None,
                    strategy: StrategyId::Follower,
                    resting: true,
                    real_clear_at: None,
                }))
            }
            AgentAction::Cancel { order_id } => {
                // Resolve order_id → token via the sim's live orders — a cancel
                // against an unknown order is never invented.
                let orders = engine.sim().open_gtd_orders().await.unwrap_or_default();
                let Some(o) = orders.iter().find(|o| &o.order_id == order_id) else {
                    bail!("unknown or non-live order_id: {order_id}");
                };
                let Some(cid) = engine.token_cid(&o.token_id) else {
                    bail!("token of order {order_id} has no cid");
                };
                let Some((_slug, clear_at, neg_risk, _yes)) = engine.market_meta(&cid) else {
                    bail!("market {cid} has no meta");
                };
                Ok(Some(OrderIntent {
                    market_cid: cid,
                    token_id: o.token_id.clone(),
                    side: OrderSide::Cancel,
                    // The cancel travels through the executor; the real `order_id` is
                    // resolved there against the journal/venue — the intent carries
                    // `Cancel` as side and the order's price.
                    price: o.price,
                    size: o.remaining,
                    clear_at,
                    signal: Decimal::ZERO,
                    neg_risk,
                    limit_price: None,
                    second_leg: None,
                    strategy: StrategyId::Follower,
                    resting: false,
                    real_clear_at: None,
                }))
            }
        }
    }

    /// Classifies the outcome of the dispatched action in the step's window, with
    /// the REAL evidence of the pipeline:
    ///
    /// - `Rejected` if a `GateRejected`/`Rejected` keyed to the intent arrived
    ///   (the intent's own token+price).
    /// - `Executed` if a Buy/Sell fill materialized (or the `Cancelled` ack of a
    ///   cancel — an executed cancel is a terminal outcome).
    /// - `Accepted` if the order stayed live at the venue (GTD resting) or the
    ///   cancel is still in flight without ack — an ACK is never a fill.
    ///
    /// `event_seqs` is left empty: the contract's `local_seq`s belong to the RECORDER
    /// of external events; the internal pipeline's effects are referenced by
    /// `order_id`/`exec_id` in the outcome's `fills_seen`.
    ///
    /// `orders_before` = `order_id`s live before the dispatch: attributes fills by a
    /// new `order_id` (a fill's `f.price` is fee-inclusive — it does not match the
    /// intent's `price`).
    async fn classify_outcome(
        &self,
        action: &AgentAction,
        orders_before: &std::collections::HashSet<String>,
    ) -> ActionOutcome {
        let fills = &self.agent.window_fills;
        if std::env::var_os("HRF_ENV_DEBUG").is_some() {
            eprintln!("[env-debug] action={action:?} window_fills={fills:?}");
            let oo = self
                .engine
                .as_ref()
                .expect("engine")
                .sim()
                .open_gtd_orders()
                .await
                .unwrap_or_default();
            eprintln!("[env-debug] open_gtd={oo:?}");
        }
        match action {
            AgentAction::Cancel { order_id } => {
                if fills.iter().any(|f| {
                    matches!(f.side, FillSide::Cancelled) && f.order_id.as_deref() == Some(order_id)
                }) {
                    ActionOutcome::Executed { event_seqs: vec![] }
                } else if fills.iter().any(|f| {
                    matches!(f.side, FillSide::GateRejected | FillSide::Rejected)
                        && f.order_id.as_deref() == Some(order_id)
                }) {
                    ActionOutcome::Rejected {
                        reason: "cancel_rejected".into(),
                        event_seqs: vec![],
                    }
                } else {
                    // Cancel in flight (or the executor has not emitted an ack yet) —
                    // pending, never executed by assumption.
                    ActionOutcome::Accepted { event_seqs: vec![] }
                }
            }
            AgentAction::Buy {
                token_id, price, ..
            }
            | AgentAction::Sell {
                token_id, price, ..
            } => {
                // Own = fill of THIS action's order: a NEW `order_id` (not live before the
                // dispatch). Without an `order_id` (gate rejects keyed to the intent) the
                // echo is the intent's token+price+size — which is comparable.
                let is_intent_fill = |f: &Fill| {
                    &f.token_id == token_id
                        && match &f.order_id {
                            Some(id) => !orders_before.contains(id),
                            None => f.price == *price,
                        }
                };
                if let Some(f) = fills.iter().find(|f| {
                    is_intent_fill(f) && matches!(f.side, FillSide::GateRejected | FillSide::Rejected)
                }) {
                    return ActionOutcome::Rejected {
                        reason: format!("{:?}", f.side).to_lowercase(),
                        event_seqs: vec![],
                    };
                }
                if fills
                    .iter()
                    .any(|f| is_intent_fill(f) && matches!(f.side, FillSide::Buy | FillSide::Sell))
                {
                    return ActionOutcome::Executed { event_seqs: vec![] };
                }
                // Is there a live order with that token+price? → Accepted.
                let orders = self
                    .engine
                    .as_ref()
                    .expect("engine")
                    .sim()
                    .open_gtd_orders()
                    .await
                    .unwrap_or_default();
                if orders
                    .iter()
                    .any(|o| &o.token_id == token_id && o.price == *price)
                {
                    return ActionOutcome::Accepted { event_seqs: vec![] };
                }
                // No signal in the window: the submit may still be in flight (modeled
                // latency) or have died without notice. The honest outcome is
                // Accepted-pending; the next step will resolve it with the evidence that arrives.
                ActionOutcome::Accepted { event_seqs: vec![] }
            }
            AgentAction::Hold => ActionOutcome::Accepted { event_seqs: vec![] },
        }
    }
}

/// Filters the tape to one market: keeps the books/resolutions of the `market_cid`,
/// the spots of its asset (the underlying's oracle is part of the episode) and the
/// metas (the metas record the token→cid mapping — without them the registry does
/// not know the market).
///
/// Episode scope:
/// - `quality` keeps ALL the marks: those foreign to the filtered market are
///   recorded in the episode's observations, but they CANNOT abort — the
///   `abort_on_quality` gate filters by `cid` (see `ResearchEnv::advance_to_decision`).
/// - `quality_stats.meta_cids` is bounded to the filtered cid: the degradation of
///   other markets stays declared in the global counters but falls outside the
///   episode's scope.
fn filter_tape(tape: &Tape, market_cid: Option<&str>) -> Tape {
    let Some(cid) = market_cid else {
        return tape.clone();
    };
    let asset = tape
        .books
        .iter()
        .find(|b| b.snap.market_condition_id == cid)
        .and_then(|b| crate::oracle::asset_from_slug(&b.snap.slug));
    let mut quality_stats = tape.quality_stats.clone();
    quality_stats.meta_cids.retain(|c| c == cid);
    Tape {
        books: tape
            .books
            .iter()
            .filter(|b| b.snap.market_condition_id == cid)
            .cloned()
            .collect(),
        spots: tape
            .spots
            .iter()
            // The engine keys `vols` by `asset.to_lowercase()` — the filter compares
            // the same way (the source may write "BTC").
            .filter(|s| asset.as_ref().is_none_or(|a| s.asset.to_lowercase() == *a))
            .cloned()
            .collect(),
        resolutions: tape
            .resolutions
            .iter()
            .filter(|r| &r.cid == cid)
            .cloned()
            .collect(),
        // All the marks — the scope is decided by the abort gate, not by the tape filter.
        quality: tape.quality.clone(),
        quality_stats,
        source: tape.source,
    }
}

/// Conservative exit value of `shares` of the token, walking the bids of the last
/// observed book with its depth: shares exceeding the visible depth are worth 0
/// (never overvalues). No book → 0.
fn exit_value(token_id: &str, shares: Decimal, book: Option<&BookSnapshot>) -> Decimal {
    let Some(b) = book else {
        return Decimal::ZERO;
    };
    let tb = if b.yes.token_id == token_id {
        &b.yes
    } else if b.no.token_id == token_id {
        &b.no
    } else {
        return Decimal::ZERO;
    };
    let mut rest = shares;
    let mut value = Decimal::ZERO;
    for lvl in &tb.bids.levels {
        if rest <= Decimal::ZERO {
            break;
        }
        let take = rest.min(lvl.size);
        value += take * lvl.price;
        rest -= take;
    }
    // Remainder without visible depth → 0 (conservative).
    value
}
