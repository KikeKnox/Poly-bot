//! Causal recorder of multi-source events for research.
//!
//! Records EVERY relevant market event exactly as it reached the bot (Polymarket books and deltas, the trade tape,
//! Binance ticks…) as a [`ResearchEvent`] of the frozen contract (src/research/contract.rs), in append-only JSONL under
//! its own directory (NEVER the active DB of `state.rs`, nor the analytics).
//!
//! ## Design
//!
//! Same pattern as `src/tools/recorders/spot_stream.rs`: the producers are clonable
//! and NON-blocking handles (`try_send` over a BOUNDED `mpsc`: a full channel drops the event and counts it, and never
//! slows the trading hot path), and the writer runs on a dedicated OS thread doing `write`/`flush` per batch.
//!
//! Format: one JSON line per event in `dir/<source>/<yyyymmdd>.jsonl` (UTC date of `recv_ts_ms`). JSONL rather than SQLite
//! because consumption is sequential and offline (replay/labelling), appending is the cheapest and most robust option for the
//! hot path, the contract's `payload` is already JSON, and there is no online query that would justify indexes or a second
//! writer competing for the WAL write lock.
//!
//! ## Quality rules
//!
//! - **Dedup by stable content** (`dedup_key`): the key is computed by the hook from fields of the event ITSELF (the venue's
//!   trade id, or a hash of `token+price+size+side+timestamp` when there is no id). NEVER from `recv_ts` nor `local_seq`: a key
//!   with a time component re-derives a different id on every resend and the dedup never matches. Bounded FIFO window
//!   (`dedup_capacity`); a duplicate found is counted in `deduped` and not written twice.
//!   Caveat: the window lives ONLY in the writer thread's memory. After a process restart, a replay of an earlier session may
//!   be recorded a second time (bounded by the gap in `local_seq` at the close and by the `RESYNC` mark of the new connection generation).
//! - **Generation/epoch** (`generation` + `stream` + `source_seq`): the monotonicity of the venue's seq is evaluated ONLY
//!   within the same connection generation. A new generation marks the stream's first event with `RESYNC` and resets the
//!   observed seq (its seq may be lower and still wins: it is a different epoch). Within the same generation, `seq <= last`
//!   is marked `DUPLICATE` (sequence anomaly/reordering, though it is still written) and `seq > last+1` is marked `GAP`.
//! - **Loss by saturation**: `local_seq` is assigned BEFORE the `try_send`, so a dropped event leaves a visible hole in the
//!   ledger's seq, and the next event enqueued after a drop carries the `GAP` mark in addition to the `discarded_total`
//!   counter (also persisted in `_recorder_stats.jsonl`).
//! - **Three clocks, none replaced**: `origin_ts_ms` only if the source provides it (otherwise `None` + `MISSING_REMOTE_TS`
//!   mark: never invented), `recv_ts_ms` = local reception, `avail_ts_ms` = delivery to the consumer (invariant `avail >= recv`).
//!
//! ## Wiring
//!
//! 1. `main.rs`: `ResearchRecorder::start(&cfg)` once at startup if `[research_events].enabled`. If creation fails the
//!    recorder is absent and the bot keeps running (research never aborts the bot).
//! 2. `market_feed/ws.rs`: `SourceContext` per session of `ws_loop` (`generation: new_generation()`,
//!    `stream: "poly-ws:shard-N"`) and `handle::dispatch_with_recorder` in the dispatcher. The trade tape is only requested
//!    if `research_events.record_tape` (it forces `custom_feature_enabled` even when `tape_window_secs == 0`; the flag
//!    multiplies the traffic per connection).
//! 3. `main.rs`: `oracle::price::run_with_recorder` with `recorder.producer()`.
//! 4. `tools::recorders::spot_stream` (task per asset in `main.rs`): for each `SpotSample`,
//!    `producer.register(spot_stream::event_from_sample(&m, gen))`.
//! 5. `market_feed/mod.rs`: `resync::run_resync_task_with_recorder` with `recorder.producer()`. The REST books from the
//!    sweeper enter marked `RESYNC`.
//! 6. `order/executor.rs`: `Executor::record_order_submit` emits `EventKind::OrderSubmit` with the serialized `OrderRequest`
//!    just before each `audit::submit*`: the intent as it left, before the result. `record_router_result` emits the observed
//!    outcome (`Sim(Accept|Reject|Fill|Notify)`), and `record_cancel_request`/`record_cancel_result` cover
//!    `CancelRequest`/`CancelAck`.
//! 7. `market_feed/discover.rs`: `register_if_new` emits `EventKind::MarketMeta` (contract B) when each new bracket is
//!    registered. It arrives via `initial_discover`, `run_discover_task` and `run_prefetch_task`, all of them with the
//!    `recorder` propagated since `mod.rs::run`.
//! 8. `tools::recorders::resolution::run_with_recorder`: emits `EventKind::Resolution` when `apply_fetch` confirms or corrects
//!    an official outcome. The call site in `main.rs` is still pending (it calls `run` today; see the TODO in the doc of
//!    `tools::recorders::resolution::run`).
//! 9. Shutdown: `main` calls `recorder.stop().await` after `cancel_all`.
//!    `stop()` does `detach()` on the shared [`Hub`]: it removes the `Sender` under the write lock and closes the channel
//!    directly, WITHOUT waiting for the producer's tasks/clones to die (this removes the need to drop every clone by hand).
//!    `start` runs `preflight` (write+flush+fsync of `_preflight.json`): without an `ok:true` marker the capture is not valid.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::research::contract::{EventKind, EventSource, QualityFlags, ResearchEvent};

/// Cap on the envelopes the writer drains at once before writing and flushing; avoids unbounded batches if the producer bursts.
const BATCH_MAX: usize = 500;

/// Sidecar file with counter snapshots (one per batch plus one final).
const STATS_FILE: &str = "_recorder_stats.jsonl";

/// Preflight marker: written, flushed and fsynced at startup. `ok:true` certifies that the directory is creatable, writable
/// and durable in THIS process. A capture process without an `ok:true` `_preflight.json` is not a valid capture.
/// On failure it tries to leave `ok:false` with the error.
const PREFLIGHT_FILE: &str = "_preflight.json";

/// Root sidecar with one marker per recording session (`attach`). It lives
/// next to `_recorder_stats.jsonl`/`_preflight.json`, NEVER inside
/// `<dir>/<source>/` — the frozen replay parser only reads those
/// subdirectories and must not see lines that are not `ResearchEvent`.
const SESSIONS_FILE: &str = "_sessions.jsonl";

// ── Config ────────────────────────────────────────────────────────────

/// Recorder config. Built from the `[research_events]` section of `config.rs` and instantiated in `main.rs`.
#[derive(Debug, Clone)]
pub struct ResearchRecorderConfig {
    /// Base directory of the JSONL files (`<dir>/<source>/<yyyymmdd>.jsonl`).
    /// Must be a research path SEPARATE from the active DB.
    pub dir: PathBuf,
    /// Size of the bounded producer→writer channel. When saturated, the producer drops and counts the event (it never blocks the hot path).
    pub channel_capacity: usize,
    /// FIFO window of dedup keys remembered by the writer.
    pub dedup_capacity: usize,
}

impl Default for ResearchRecorderConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("data/research_events"),
            channel_capacity: 8_192,
            dedup_capacity: 262_144,
        }
    }
}

// ── Event input (what the hooks build) ─────────────────────────

/// Event data BEFORE `local_seq` is assigned: what the emission hooks build (e.g. `dispatch_with_recorder` and the Binance tick hook).
///
/// `local_seq` and the `MISSING_REMOTE_TS` mark are set by [`EventProducer`];
/// `GAP`/`RESYNC`/`DUPLICATE` are evaluated by the writer, since they need cross-event state. The remaining fields describe the event as it arrived.
#[derive(Debug, Clone)]
pub struct EventInput {
    pub source: EventSource,
    pub kind: EventKind,
    pub market_cid: Option<String>,
    pub token_id: Option<String>,
    pub asset: Option<String>,
    /// Original numeric payload as it arrived (see the doc of `ResearchEvent`).
    pub payload: Value,
    /// Remote timestamp ONLY if the source provides it; `None` otherwise.
    pub origin_ts_ms: Option<i64>,
    /// Local reception (ms epoch).
    pub recv_ts_ms: i64,
    /// Availability to the consumer (ms epoch). If passed `< recv_ts_ms` it is
    /// clamped to `recv` (the contract's invariant wins).
    pub avail_ts_ms: i64,
    /// Stream identity for generation/seq tracking (e.g. `"poly-ws:shard-0"`, `"binance:btc"`). `None` = no tracking.
    pub stream: Option<String>,
    /// Epoch of the WS connection that delivered the event; incremented on each
    /// reconnection (see [`new_generation`]). Persisted in the line.
    pub generation: Option<u64>,
    /// Venue sequence WITHIN the generation (Binance trade id…), if the source provides it. Persisted in the line.
    pub source_seq: Option<i64>,
    /// Stable-content dedup key (see the module doc). `None` = do not deduplicate this event.
    pub dedup_key: Option<u64>,
    /// Extra quality marks the hook provides (e.g. `SYNTHETIC` for data derived by the bot, `REDUCED_FIDELITY` for lower-granularity
    /// history). The producer adds the ones it derives (`MISSING_REMOTE_TS`) and the writer the cross-event ones (GAP/RESYNC/…).
    pub quality_extra: QualityFlags,
}

impl EventInput {
    /// Minimal constructor: `avail_ts_ms` starts equal to `recv_ts_ms` (valid when there is no intermediate step between arrival
    /// and consumption) and the remaining optionals are `None`.
    pub fn new(source: EventSource, kind: EventKind, payload: Value, recv_ts_ms: i64) -> Self {
        Self {
            source,
            kind,
            market_cid: None,
            token_id: None,
            asset: None,
            payload,
            origin_ts_ms: None,
            recv_ts_ms,
            avail_ts_ms: recv_ts_ms,
            stream: None,
            generation: None,
            source_seq: None,
            dedup_key: None,
            quality_extra: QualityFlags::NONE,
        }
    }
}

/// Stable hash for dedup keys. Hash ONLY fields of the event itself that do not change between representations or resends
/// (the venue's trade id, or `token+price+size+side+timestamp`). NEVER `recv_ts`/`local_seq`.
pub fn stable_hash<T: Hash>(v: T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// Global counter of connection generations: one per WS session, shared across sources so the ids are unique in the ledger.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// New connection generation/epoch. The wiring calls it once per WS session (every reconnection included) and sets it in the `SourceContext`.
pub fn new_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

/// Context of a concrete source for an emission hook: the producer, the generation of ITS current connection, and the stream name.
#[derive(Clone)]
pub struct SourceContext {
    pub producer: EventProducer,
    /// Epoch of the current connection: [`new_generation`] per session.
    pub generation: u64,
    /// Logical stream name ("poly-ws:shard-0", "binance"…). Hooks can suffix it per key ("binance:btc") if the venue's seq is per symbol.
    pub stream: String,
}

// ── Persisted line ────────────────────────────────────────────────────────

/// What is written to disk: the flattened `ResearchEvent` plus the stream-tracking fields (`stream`, `generation`, `source_seq`)
/// that the contract does not have. They sit next to the event, not inside `payload`, so the payload stays intact for `payload_as`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedLine {
    #[serde(flatten)]
    pub event: ResearchEvent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_seq: Option<i64>,
}

// ── Hub (stable identity, survives start/stop) ─────────────────────────

/// Counters shared among the cloned `EventProducer`s.
#[derive(Debug, Default)]
struct Counters {
    enqueued: AtomicU64,
    discarded: AtomicU64,
    /// Monotonic local recorder sequence: the canonical order of the ledger. It is assigned BEFORE the `try_send`, so a dropped
    /// event leaves a hole in the recorded seq, which IS the loss mark in the dataset itself.
    /// Does NOT advance while the hub is inactive (see `EventProducer::record`).
    local_seq: AtomicU64,
    /// There was at least one drop since the last enqueued event: the next enqueued event is marked `GAP` by the writer.
    had_discard: AtomicBool,
}

/// Stable identity of the tape producer, which lives longer than any individual `ResearchRecorder`. A single `Sender` protected
/// by an `RwLock` is the only way an `EventProducer` reaches the writer: no producer clone keeps its own `Sender`, so
/// stopping the recording (`detach`) never depends on those clones dying (this removes the bug class by construction: before,
/// shutdown hung if a cloned `Sender` was still alive in some producer).
///
/// Switching recording on and off at runtime is then: create the `Hub` ONCE at startup, inject `hub.producer()` wherever the
/// recorder's producer used to be injected (the wiring of executor, oracle, market_feed… is not touched again), and start/stop
/// a `ResearchRecorder` over THAT hub as many times as the tool is toggled.
pub struct Hub {
    /// `false` = `record()` discards without enqueuing and without touching `counters`
    /// (see `EventProducer::record`). It is read on the hot path, so it is an `AtomicBool` with `Ordering::Relaxed`: there is
    /// no invariant beyond "is there a live writer right now?".
    active: AtomicBool,
    /// The only real `Sender`. `attach` publishes it, `detach` removes it under the same write lock: a `record()` in progress
    /// sees `None` and discards cleanly, never a channel closed halfway.
    tx: RwLock<Option<mpsc::Sender<Envelope>>>,
    counters: Counters,
    /// Number of times this hub was `attach`ed in the process. 0 = never; >1 = re-attached after a previous `detach`.
    /// The writer of that session forces `RESYNC` on the first event of each stream (the gap between sessions must not be silent).
    session: AtomicU64,
}

impl Hub {
    /// Newly created hub: inactive, without a writer. `record()` through a producer of this hub discards until the first `attach`.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            active: AtomicBool::new(false),
            tx: RwLock::new(None),
            counters: Counters::default(),
            session: AtomicU64::new(0),
        })
    }

    /// Clonable, non-blocking handle to this hub.
    pub fn producer(self: &Arc<Self>) -> EventProducer {
        EventProducer { hub: self.clone() }
    }

    /// `true` if a writer with a published `Sender` is live right now. The hot-path hooks use it to avoid building the `EventInput`
    /// (payload included) when recording is off.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Publishes the writer's `Sender` and activates recording. Returns the resulting session number (1 = first startup of the process).
    fn attach(&self, tx: mpsc::Sender<Envelope>) -> u64 {
        *self.tx.write() = Some(tx);
        self.active.store(true, Ordering::Relaxed);
        self.session.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Deactivates recording and removes the `Sender` under the write lock. The channel closes as soon as the returned value
    /// is dropped, without waiting for any cloned `EventProducer` to die.
    fn detach(&self) -> Option<mpsc::Sender<Envelope>> {
        self.active.store(false, Ordering::Relaxed);
        self.tx.write().take()
    }
}

/// Envelope that travels through the channel: the already-sealed event plus the stream/dedup metadata that the writer
/// evaluates (they need cross-event state).
struct Envelope {
    ev: ResearchEvent,
    stream: Option<String>,
    generation: Option<u64>,
    source_seq: Option<i64>,
    dedup_key: Option<u64>,
    /// There were drops since the last enqueued event → the writer marks `GAP`.
    prior_loss: bool,
}

/// Clonable, NON-blocking handle to record events into the hub. One clone per task/producer; cloning is cheap (an `Arc` to the `Hub`).
/// NO clone keeps its own `Sender`: all of them read the same one under the hub's `RwLock` on each `record()`, so a `detach()`
/// turns them all off at once without depending on any of them being dropped.
#[derive(Clone)]
pub struct EventProducer {
    hub: Arc<Hub>,
}

impl EventProducer {
    /// Seals the entry (assigns `local_seq`, marks `MISSING_REMOTE_TS` if there is no `origin_ts_ms`, clamps `avail >= recv`) and
    /// enqueues it with `try_send`. It NEVER blocks. Returns `true` if it was enqueued; `false` if the hub is inactive (recording
    /// off: it does NOT touch `local_seq` or the counters, and it is not a loss), or if it was discarded because of saturation or a
    /// dead writer (those two DO count in `discarded_total` and build the next `GAP`).
    pub fn record(&self, entry: EventInput) -> bool {
        if !self.hub.is_active() {
            return false;
        }
        let counters = &self.hub.counters;
        let local_seq = counters.local_seq.fetch_add(1, Ordering::Relaxed);
        let mut quality = entry.quality_extra;
        if entry.origin_ts_ms.is_none() {
            quality.insert(QualityFlags::MISSING_REMOTE_TS);
        }
        // The contract's invariant wins: data cannot be available before it arrives. If the hook violated it, it is clamped.
        let avail_ts_ms = entry.avail_ts_ms.max(entry.recv_ts_ms);
        let ev = ResearchEvent {
            schema_version: crate::research::contract::SCHEMA_VERSION,
            source: entry.source,
            kind: entry.kind,
            market_cid: entry.market_cid,
            token_id: entry.token_id,
            asset: entry.asset,
            payload: entry.payload,
            origin_ts_ms: entry.origin_ts_ms,
            recv_ts_ms: entry.recv_ts_ms,
            avail_ts_ms,
            local_seq,
            quality,
        };
        debug_assert!(ev.validate().is_ok());
        let source_label = source_name(ev.source);
        let envelope = Envelope {
            ev,
            stream: entry.stream,
            generation: entry.generation,
            source_seq: entry.source_seq,
            dedup_key: entry.dedup_key,
            prior_loss: counters.had_discard.swap(false, Ordering::Relaxed),
        };
        // (perf) `try_send` directly under the read lock, without cloning the `Sender` first: cloning touched its atomic
        // refcount on EVERY event of the hot path. The guard is released as soon as `try_send` returns (it never blocks, so the
        // guard is not held longer than before, when it was already held to read the `Option<Sender>`).
        let send_result = match self.hub.tx.read().as_ref() {
            Some(tx) => tx.try_send(envelope),
            None => return false, // detach() won the race right after the is_active() check
        };
        match send_result {
            Ok(()) => {
                counters.enqueued.fetch_add(1, Ordering::Relaxed);
                metrics::counter!(
                    "hrf_research_events_queued_total",
                    "source" => source_label
                )
                .increment(1);
                true
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                counters.discarded.fetch_add(1, Ordering::Relaxed);
                counters.had_discard.store(true, Ordering::Relaxed);
                metrics::counter!(
                    "hrf_research_events_discarded_total",
                    "source" => source_label,
                    "reason" => "saturation"
                )
                .increment(1);
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                counters.discarded.fetch_add(1, Ordering::Relaxed);
                counters.had_discard.store(true, Ordering::Relaxed);
                metrics::counter!(
                    "hrf_research_events_discarded_total",
                    "source" => source_label,
                    "reason" => "dead_writer"
                )
                .increment(1);
                false
            }
        }
    }

    /// `true` if there is a live writer behind this producer. The hot-path hooks check it BEFORE building the `EventInput`
    /// (payload included), so that this cost is not paid while recording is off.
    pub fn is_active(&self) -> bool {
        self.hub.is_active()
    }

    /// Sink producer: a private hub that is never `attach`ed, so `record()` always returns `false` without touching any counter.
    /// For pure adapter tests and optional wiring (recorder absent but hook present).
    pub fn sink() -> Self {
        Self { hub: Hub::new() }
    }
}

// ── Recorder (writer on a dedicated OS thread) ───────────────────────────────

/// Totals returned by [`ResearchRecorder::stop`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecorderTotals {
    pub enqueued: u64,
    pub written: u64,
    pub deduped: u64,
    pub discarded: u64,
}

/// Research event recorder: the "writer" half of a recording session over a [`Hub`]. `start` opens it (creates the directory,
/// starts the writer thread and publishes its `Sender` to the hub); `stop` closes it (removes the `Sender` and waits for the
/// thread). The [`Hub`] outlives each `ResearchRecorder`: it is the piece that the toggleable tool adapter switches on and off
/// repeatedly without rewiring the producers.
pub struct ResearchRecorder {
    hub: Arc<Hub>,
    writer: Option<JoinHandle<WriterStats>>,
    /// Snapshot of `hub.counters.enqueued/discarded` taken in `start()`, BEFORE `attach()`. With the hub inactive (or just created)
    /// no producer can touch them yet, so it is a consistent baseline to report THIS session's delta. The `Hub`'s `enqueued`/`discarded`
    /// are accumulated over the whole process (they survive each session); the writer's `written`/`deduped` are per session
    /// (`WriterState` starts empty in each `run_writer`). Without this baseline, `stop()` and `_recorder_stats.jsonl` mixed a
    /// process-wide total with a session counter after a re-attach. `local_seq` is NOT touched by this fix: it stays continuous
    /// and global, since it is the key of the ledger's canonical ordering.
    enqueued_baseline: u64,
    discarded_baseline: u64,
}

impl ResearchRecorder {
    /// Write preflight: creates `cfg.dir` if needed, writes `_preflight.json` with `write+flush+sync_data` (fsync: a buffered write
    /// that never reaches disk does not count as valid) and checks that the file is not empty. Actionable errors: the `io::Error`
    /// names the directory and the failing operation.
    ///
    /// Returns the marker's path on success. If `create_dir_all` fails there is not even a place to leave the marker, so the
    /// propagated error is the evidence (the capture harness requires a `_preflight.json` with `ok:true` in the dir to consider
    /// the capture valid).
    ///
    /// `start` runs it BEFORE creating the channel/writer: a process whose recorder cannot write must NOT be able to declare
    /// itself a valid capture. The bot's behavior does not change: `main.rs` already degrades to `warn` if `start` returns an
    /// error; this only makes the error real and visible on disk.
    pub fn preflight(cfg: &ResearchRecorderConfig) -> std::io::Result<PathBuf> {
        fs::create_dir_all(&cfg.dir).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "research_events preflight: could not create the directory {}: {e}",
                    cfg.dir.display()
                ),
            )
        })?;
        let path = cfg.dir.join(PREFLIGHT_FILE);
        let write = |ok: bool, error: Option<&str>| -> std::io::Result<()> {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)?;
            let line = serde_json::json!({
                "ts_ms": chrono::Utc::now().timestamp_millis(),
                "ok": ok,
                "dir": cfg.dir.to_string_lossy(),
                "channel_capacity": cfg.channel_capacity,
                "dedup_capacity": cfg.dedup_capacity,
                "error": error,
            });
            f.write_all(line.to_string().as_bytes())?;
            f.write_all(b"\n")?;
            f.flush()?;
            f.sync_data()?; // real fsync: the write must be durable
            drop(f);
            if fs::metadata(&path)?.len() == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "preflight: the marker was left empty after flush+sync",
                ));
            }
            Ok(())
        };
        match write(true, None) {
            Ok(()) => Ok(path),
            Err(e) => {
                // Best effort: record the failure in the directory itself.
                let _ = write(false, Some(&e.to_string()));
                Err(std::io::Error::new(
                    e.kind(),
                    format!(
                        "research_events preflight: dir {} is not writable/durable: {e}",
                        cfg.dir.display()
                    ),
                ))
            }
        }
    }

    /// Creates `cfg.dir` if needed, runs the real-write [`Self::preflight`]
    /// and starts the writer thread over `hub`. If the dir does not accept
    /// durable writes, or the thread does not start, `start` fails WITHOUT
    /// touching the hub (it stays as it was: on first use, inactive).
    ///
    /// The `Sender` is only published to the hub (`hub.attach`) AFTER the
    /// writer thread has started successfully — never before, so the "asset"
    /// hub is not left pointing at a writer that never existed.
    pub fn start(cfg: &ResearchRecorderConfig, hub: &Arc<Hub>) -> std::io::Result<Self> {
        Self::preflight(cfg)?;
        // Baseline of THIS session (see the field's doc-comment): taken BEFORE
        // `attach()`, while the hub is still inactive, so no producer can
        // touch these counters yet.
        let enqueued_baseline = hub.counters.enqueued.load(Ordering::Relaxed);
        let discarded_baseline = hub.counters.discarded.load(Ordering::Relaxed);
        let (tx, rx) = mpsc::channel(cfg.channel_capacity.max(1));
        // A session > 0 already lived in this hub means this startup is a RESUMPTION (re-attach after a previous stop). The writer
        // must force RESYNC on the first event of each stream it sees, even if the origin connection (WS) did not reconnect.
        let is_resumed = hub.session.load(Ordering::Relaxed) > 0;
        let writer_cfg = cfg.clone();
        let writer_hub = hub.clone();
        let writer = std::thread::Builder::new()
            .name("research_events_writer".to_string())
            .spawn(move || {
                run_writer(
                    writer_cfg,
                    rx,
                    writer_hub,
                    is_resumed,
                    enqueued_baseline,
                    discarded_baseline,
                )
            })
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("research_events: could not spawn the writer thread: {e}"),
                )
            })?;
        let session = hub.attach(tx);
        record_session(&cfg.dir, session);
        Ok(Self {
            hub: hub.clone(),
            writer: Some(writer),
            enqueued_baseline,
            discarded_baseline,
        })
    }

    /// Orderly close of THIS session: removes the hub's `Sender` (`detach`) and waits for the writer thread (it drains what is
    /// queued and flushes). It does NOT depend on any cloned `EventProducer` dying: `detach` closes the channel directly under
    /// the write lock. The hub stays alive after `stop`: a later `ResearchRecorder::start` over the same hub resumes recording
    /// without rewiring the producers. A writer panic is downgraded to an error log plus partial counters: research never brings
    /// the shutdown down.
    pub async fn stop(mut self) -> RecorderTotals {
        let writer = self.writer.take().expect("stop() was already called");
        drop(self.hub.detach()); // closes the channel right now, without waiting for anyone
        let hub = self.hub.clone();
        let stats = match tokio::task::spawn_blocking(move || writer.join()).await {
            Ok(Ok(s)) => s,
            Ok(Err(_)) | Err(_) => {
                tracing::error!(
                    "research_events: the writer thread panicked, closing metrics are partial"
                );
                WriterStats::default()
            }
        };
        RecorderTotals {
            enqueued: hub
                .counters
                .enqueued
                .load(Ordering::Relaxed)
                .saturating_sub(self.enqueued_baseline),
            discarded: hub
                .counters
                .discarded
                .load(Ordering::Relaxed)
                .saturating_sub(self.discarded_baseline),
            written: stats.written,
            deduped: stats.deduped,
        }
    }
}

/// Session-marker sidecar, ONLY here, never as lines inside the source JSONL files (the frozen replay parser must not see them).
/// One line per `attach`: the gap between the previous `stop` and this `start` is what the first RESYNC event of each stream
/// (see `WriterState`) makes visible in the dataset itself. Best effort: a write failure here does not abort `start` (same rule
/// as the rest of the module: research never brings the bot down).
fn record_session(dir: &Path, session: u64) {
    let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(SESSIONS_FILE))
    else {
        return;
    };
    let line = serde_json::json!({
        "session": session,
        "ts_ms": chrono::Utc::now().timestamp_millis(),
    });
    let _ = writeln!(f, "{line}");
}

// ── Writer ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
struct WriterStats {
    written: u64,
    deduped: u64,
}

/// Per-stream state for the freshness check, scoped to the generation: the seq is only compared within the same epoch.
#[derive(Debug, Clone, Copy)]
struct StreamState {
    generation: u64,
    last_seq: Option<i64>,
}

/// Mutable writer state: dedup and stream tracking. Kept apart from the I/O so that `process_envelope` can be tested without file paths.
struct WriterState {
    /// Seen dedup keys (bounded FIFO by `dedup_capacity`).
    dedup_seen: HashSet<u64>,
    dedup_fifo: VecDeque<u64>,
    dedup_capacity: usize,
    /// Last (generation, seq) seen per stream.
    streams: HashMap<String, StreamState>,
    /// Pending loss signal, to apply to the next WRITTEN event. If the envelope that carried it is deduplicated (not written),
    /// the GAP is not lost: it propagates to the next event that does enter the ledger.
    pending_gap: bool,
    /// `true` when this writer started over a hub ALREADY used before (re-attach after a `stop`). The `streams` map starts empty
    /// in each new writer, so without this flag the first time a stream is seen after the resumption would fall into the `None`
    /// branch (no mark) and the gap of the earlier session would stay silent.
    force_resync_unseen: bool,
    stats: WriterStats,
}

/// Result of processing an envelope.
enum Decision {
    /// Write this line.
    Write(RecordedLine),
    /// Duplicate by stable content: not written (volume is not duplicated), only counted.
    Duplicate,
}

impl WriterState {
    fn new(dedup_capacity: usize, force_resync_unseen: bool) -> Self {
        Self {
            dedup_seen: HashSet::new(),
            dedup_fifo: VecDeque::new(),
            dedup_capacity,
            streams: HashMap::new(),
            pending_gap: false,
            force_resync_unseen,
            stats: WriterStats::default(),
        }
    }

    /// Applies dedup and quality marks (GAP/RESYNC/DUPLICATE) and decides whether the envelope
    /// is written. Pure with respect to I/O: it only mutates the state.
    fn process_envelope(&mut self, over: Envelope) -> Decision {
        // This envelope observes the loss signal; if it is deduplicated and not written, the signal is kept in `pending_gap`
        // and the next written event inherits it: the GAP mark never dies with a duplicate.
        self.pending_gap |= over.prior_loss;
        if let Some(k) = over.dedup_key {
            if !self.dedup_seen.insert(k) {
                self.stats.deduped += 1;
                metrics::counter!(
                    "hrf_research_events_dedup_total",
                    "source" => source_name(over.ev.source)
                )
                .increment(1);
                return Decision::Duplicate;
            }
            self.dedup_fifo.push_back(k);
            if self.dedup_fifo.len() > self.dedup_capacity {
                if let Some(oldest) = self.dedup_fifo.pop_front() {
                    self.dedup_seen.remove(&oldest);
                }
            }
        }

        let mut ev = over.ev;
        // Pending loss mark (its own, or inherited from a deduplicated envelope): it is consumed here, on the event that is written.
        if self.pending_gap {
            ev.quality.insert(QualityFlags::GAP);
            self.pending_gap = false;
        }

        // Freshness scoped to the generation: the venue's seq is only compared within the same epoch. A NEW generation always wins
        // (its seq may be lower) and is marked RESYNC; an older one never reopens the window.
        if let (Some(stream), Some(gen)) = (over.stream.clone(), over.generation) {
            match self.streams.get_mut(&stream) {
                None => {
                    // First event of this stream for THIS writer. If the hub came from an earlier session (resumption),
                    // this "first" event actually follows a silent gap, so it is marked RESYNC even if the origin
                    // connection did not change generation.
                    if self.force_resync_unseen {
                        ev.quality.insert(QualityFlags::RESYNC);
                    }
                    self.streams.insert(
                        stream,
                        StreamState {
                            generation: gen,
                            last_seq: over.source_seq,
                        },
                    );
                }
                Some(st) if st.generation == gen => {
                    if let Some(seq) = over.source_seq {
                        match st.last_seq {
                            Some(last) if seq <= last => {
                                // Repeated or out of order within the same generation: it is written anyway (the data is not lost)
                                // but marked, because the source's ordering is not reliable here.
                                ev.quality.insert(QualityFlags::DUPLICATE);
                            }
                            Some(last) if seq > last + 1 => {
                                ev.quality.insert(QualityFlags::GAP);
                                st.last_seq = Some(seq);
                            }
                            _ => {
                                st.last_seq = Some(seq);
                            }
                        }
                    }
                }
                Some(st) => {
                    // Generation change = reconnection/resync: the observed seq is reset and the event is marked RESYNC.
                    ev.quality.insert(QualityFlags::RESYNC);
                    *st = StreamState {
                        generation: gen,
                        last_seq: over.source_seq,
                    };
                }
            }
        }

        self.stats.written += 1;
        Decision::Write(RecordedLine {
            event: ev,
            stream: over.stream,
            generation: over.generation,
            source_seq: over.source_seq,
        })
    }
}

fn source_name(source: EventSource) -> &'static str {
    match source {
        EventSource::PolymarketWs => "polymarket_ws",
        EventSource::PolymarketRest => "polymarket_rest",
        EventSource::PolymarketUserWs => "polymarket_user_ws",
        EventSource::BinanceWs => "binance_ws",
        EventSource::Chainlink => "chainlink",
        EventSource::InternalSim => "internal_sim",
        EventSource::Internal => "internal",
    }
}

/// Directory and file path of an event:
/// `<dir>/<source>/<yyyymmdd-UTC-of-recv_ts_ms>.jsonl`. An invalid `recv_ts_ms` (out of range) falls into
/// `without_date.jsonl`: a date is never invented.
fn event_path(dir: &Path, ev: &ResearchEvent) -> PathBuf {
    let date = chrono::DateTime::from_timestamp_millis(ev.recv_ts_ms)
        .map(|d| d.format("%Y%m%d").to_string())
        .unwrap_or_else(|| "without_date".to_string());
    dir.join(source_name(ev.source))
        .join(format!("{date}.jsonl"))
}

fn run_writer(
    cfg: ResearchRecorderConfig,
    mut rx: mpsc::Receiver<Envelope>,
    hub: Arc<Hub>,
    force_resync_unseen: bool,
    enqueued_baseline: u64,
    discarded_baseline: u64,
) -> WriterStats {
    let mut state = WriterState::new(cfg.dedup_capacity, force_resync_unseen);
    let mut writers: HashMap<PathBuf, BufWriter<File>> = HashMap::new();
    let mut stats_file = open_stats(&cfg.dir);

    while let Some(first) = rx.blocking_recv() {
        let mut batch = vec![first];
        while batch.len() < BATCH_MAX {
            match rx.try_recv() {
                Ok(s) => batch.push(s),
                Err(_) => break,
            }
        }

        let mut touched: HashSet<PathBuf> = HashSet::new();
        for over in batch {
            if let Decision::Write(line) = state.process_envelope(over) {
                let path = event_path(&cfg.dir, &line.event);
                touched.insert(path.clone());
                let w = writers.entry(path.clone()).or_insert_with(|| {
                    open_jsonl(&path).unwrap_or_else(|e| {
                        panic!("research_events: could not open {}: {e}", path.display())
                    })
                });
                match serde_json::to_string(&line) {
                    Ok(s) => {
                        if let Err(e) = w.write_all(s.as_bytes()).and_then(|_| w.write_all(b"\n")) {
                            tracing::error!(error = %e, "research_events: error writing event");
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "research_events: error serializing event");
                    }
                }
            }
        }
        // Flush per batch: bounds the loss on crash to the batch in progress.
        for path in &touched {
            if let Some(w) = writers.get_mut(path) {
                if let Err(e) = w.flush() {
                    tracing::error!(error = %e, "research_events: error during flush");
                }
            }
        }
        persist_stats(
            &mut stats_file,
            &hub.counters,
            &state.stats,
            enqueued_baseline,
            discarded_baseline,
        );
    }

    for (_, mut w) in writers {
        let _ = w.flush();
    }
    persist_stats(
        &mut stats_file,
        &hub.counters,
        &state.stats,
        enqueued_baseline,
        discarded_baseline,
    );
    state.stats
}

fn open_jsonl(path: &Path) -> std::io::Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let f = OpenOptions::new().create(true).append(true).open(path)?;
    Ok(BufWriter::new(f))
}

fn open_stats(dir: &Path) -> Option<BufWriter<File>> {
    open_jsonl(&dir.join(STATS_FILE)).ok()
}

/// Snapshot of THIS SESSION's counters in `_recorder_stats.jsonl` after
/// each batch and on close — saturation/loss stays readable on disk without
/// relying on the process still being alive to expose metrics.
/// `enqueued`/`discarded` subtract the baseline taken at `attach()` (see the
/// doc-comment of `ResearchRecorder::enqueued_baseline`): without this, after a
/// re-attach this file mixed a total for the WHOLE process with
/// `written`/`deduped`, which are per-session.
fn persist_stats(
    stats_file: &mut Option<BufWriter<File>>,
    counters: &Counters,
    stats: &WriterStats,
    enqueued_baseline: u64,
    discarded_baseline: u64,
) {
    let Some(w) = stats_file else { return };
    let line = serde_json::json!({
        "ts_ms": chrono::Utc::now().timestamp_millis(),
        "enqueued": counters.enqueued.load(Ordering::Relaxed).saturating_sub(enqueued_baseline),
        "discarded": counters.discarded.load(Ordering::Relaxed).saturating_sub(discarded_baseline),
        "written": stats.written,
        "deduped": stats.deduped,
    });
    if serde_json::to_writer(&mut *w, &line).is_ok() {
        let _ = w.write_all(b"\n");
        let _ = w.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering as Ord;

    fn envelope(
        source_seq: Option<i64>,
        generation: Option<u64>,
        dedup_key: Option<u64>,
    ) -> Envelope {
        Envelope {
            ev: ResearchEvent {
                schema_version: crate::research::contract::SCHEMA_VERSION,
                source: EventSource::BinanceWs,
                kind: EventKind::PriceTick,
                market_cid: None,
                token_id: None,
                asset: Some("btc".into()),
                payload: Value::Null,
                origin_ts_ms: Some(1),
                recv_ts_ms: 1,
                avail_ts_ms: 1,
                local_seq: 0,
                quality: QualityFlags::NONE,
            },
            stream: Some("binance:btc".into()),
            generation,
            source_seq,
            dedup_key,
            prior_loss: false,
        }
    }

    fn recorded(d: Decision) -> RecordedLine {
        match d {
            Decision::Write(l) => l,
            Decision::Duplicate => panic!("expected Write"),
        }
    }

    #[test]
    fn dedup_by_stable_content_discards_resend() {
        let mut st = WriterState::new(100, false);
        let _ = recorded(st.process_envelope(envelope(Some(1), Some(1), Some(42))));
        // Same dedup_key (identical stable fields) with a different seq —
        // recv_ts/local_seq are NEVER part of the key.
        assert!(matches!(
            st.process_envelope(envelope(Some(2), Some(1), Some(42))),
            Decision::Duplicate
        ));
        assert_eq!(st.stats.deduped, 1);
        assert_eq!(st.stats.written, 1);
    }

    #[test]
    fn out_of_order_seq_is_marked_duplicate_but_written() {
        let mut st = WriterState::new(100, false);
        let _ = recorded(st.process_envelope(envelope(Some(10), Some(1), None)));
        let l = recorded(st.process_envelope(envelope(Some(7), Some(1), None)));
        assert!(l.event.quality.contains(QualityFlags::DUPLICATE));
        assert_eq!(st.stats.written, 2, "out-of-order events are written, not lost");
    }

    #[test]
    fn seq_gap_in_same_generation_marks_gap() {
        let mut st = WriterState::new(100, false);
        let _ = recorded(st.process_envelope(envelope(Some(10), Some(1), None)));
        let l = recorded(st.process_envelope(envelope(Some(12), Some(1), None)));
        assert!(l.event.quality.contains(QualityFlags::GAP));
    }

    #[test]
    fn new_generation_marks_resync_and_resets_seq() {
        let mut st = WriterState::new(100, false);
        let _ = recorded(st.process_envelope(envelope(Some(10), Some(1), None)));
        // Reconnection: generation 2 with a LOWER seq — always wins, without
        // comparing seq across epochs.
        let l = recorded(st.process_envelope(envelope(Some(3), Some(2), None)));
        assert!(l.event.quality.contains(QualityFlags::RESYNC));
        assert!(!l.event.quality.contains(QualityFlags::DUPLICATE));
        // And the seq of the new epoch is tracked from zero: a backwards step
        // within gen 2 is marked.
        let l2 = recorded(st.process_envelope(envelope(Some(2), Some(2), None)));
        assert!(l2.event.quality.contains(QualityFlags::DUPLICATE));
    }

    #[test]
    fn previous_loss_marks_gap_on_next_event() {
        let mut st = WriterState::new(100, false);
        let mut s = envelope(None, None, None);
        s.prior_loss = true;
        let l = recorded(st.process_envelope(s));
        assert!(l.event.quality.contains(QualityFlags::GAP));
    }

    /// The loss signal of a DEDUPLICATED event does not die with it: it is
    /// kept and inherited by the next event that is actually written.
    #[test]
    fn gap_over_deduplicated_propagates_to_next_written() {
        let mut st = WriterState::new(100, false);
        let _ = recorded(st.process_envelope(envelope(Some(1), Some(1), Some(7))));
        // Resend with prior_loss → dedup wins, but the GAP survives.
        let mut dup = envelope(Some(2), Some(1), Some(7));
        dup.prior_loss = true;
        assert!(matches!(st.process_envelope(dup), Decision::Duplicate));
        let l = recorded(st.process_envelope(envelope(Some(3), Some(1), None)));
        assert!(
            l.event.quality.contains(QualityFlags::GAP),
            "the loss of the deduplicated event must not disappear"
        );
    }

    /// The preflight writes a real marker with `ok:true` —
    /// dir creation + write + flush + fsync verified.
    #[test]
    fn preflight_writes_ok_marker() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = ResearchRecorderConfig {
            dir: dir.path().join("events"),
            ..Default::default()
        };
        let path = ResearchRecorder::preflight(&cfg).unwrap();
        assert!(path.ends_with("_preflight.json"));
        let content = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["ok"], serde_json::json!(true));
        assert!(v["error"].is_null());
    }

    /// A dir that cannot be created (a FILE occupies the path) makes the
    /// preflight fail with an actionable error — the process cannot
    /// declare a valid capture.
    #[test]
    fn preflight_fails_in_non_writable_dir() {
        let dir = tempfile::tempdir().unwrap();
        let obstacle = dir.path().join("occupied");
        std::fs::write(&obstacle, b"I am a file, not a dir").unwrap();
        let cfg = ResearchRecorderConfig {
            dir: obstacle.clone(),
            ..Default::default()
        };
        let e = ResearchRecorder::preflight(&cfg).unwrap_err();
        assert!(
            e.to_string().contains("research_events preflight"),
            "actionable error with context: {e}"
        );
        // And `start` propagates the same failure — it does not start the writer.
        let hub = Hub::new();
        assert!(ResearchRecorder::start(&cfg, &hub).is_err());
    }

    /// Deterministic producer saturation: channel of capacity 1 with no
    /// reader — the first send fills the channel, the rest are discarded and
    /// counted, and `had_discard` is armed to mark the next enqueued event
    /// with GAP.
    #[test]
    fn channel_saturation_counts_discards_and_arms_mark() {
        let hub = Hub::new();
        let (tx, _rx) = mpsc::channel(1);
        hub.attach(tx);
        let prod = hub.producer();
        let e = EventInput::new(EventSource::Internal, EventKind::Trade, Value::Null, 1);

        assert!(prod.record(e.clone()));
        assert!(!prod.record(e.clone()));
        assert!(!prod.record(e));

        assert_eq!(hub.counters.enqueued.load(Ord::Relaxed), 1);
        assert_eq!(hub.counters.discarded.load(Ord::Relaxed), 2);
        assert!(hub.counters.had_discard.load(Ord::Relaxed));
        // The seq was also consumed by the discards → visible gap.
        assert_eq!(hub.counters.local_seq.load(Ord::Relaxed), 3);
    }

    // ── Hub: record while inactive, detach without hanging, resume ────────

    fn simple_entry() -> EventInput {
        EventInput::new(EventSource::BinanceWs, EventKind::PriceTick, Value::Null, 1)
    }

    /// With the hub inactive, `record` returns `false` WITHOUT consuming
    /// `local_seq` — the next event, once the asset writer is attached, stays
    /// at a contiguous seq (it does not inherit a gap from the attempts that
    /// failed while it was off).
    #[test]
    fn record_with_inactive_hub_does_not_consume_seq_and_next_is_contiguous() {
        let hub = Hub::new();
        let prod = hub.producer();

        assert!(!prod.is_active());
        assert!(!prod.record(simple_entry()));
        assert!(
            !prod.record(simple_entry()),
            "several cold attempts do not consume seq either"
        );
        assert_eq!(hub.counters.local_seq.load(Ord::Relaxed), 0);

        let (tx, mut rx) = mpsc::channel(8);
        hub.attach(tx);
        assert!(prod.is_active());
        assert!(prod.record(simple_entry()));
        let over = rx.try_recv().expect("the asset event was enqueued");
        assert_eq!(
            over.ev.local_seq, 0,
            "the first event ACTUALLY recorded stays at seq 0 (contiguous)"
        );
    }

    /// Regression test: `stop()` does not depend on cloned `EventProducer`s
    /// dying — with 3 live clones, `detach` removes the `Sender` from the hub
    /// and the writer join completes. The `timeout` is the assertion: if the
    /// design reintroduced that dependency, this test would hang instead of
    /// failing.
    #[tokio::test]
    async fn detach_with_live_clones_does_not_hang() {
        let dir = tempfile::tempdir().unwrap();
        let hub = Hub::new();
        let cfg = ResearchRecorderConfig {
            dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let recorder = ResearchRecorder::start(&cfg, &hub).unwrap();
        let p1 = hub.producer();
        let p2 = hub.producer();
        let p3 = hub.producer();
        assert!(p1.record(simple_entry()));

        let res = tokio::time::timeout(std::time::Duration::from_secs(5), recorder.stop())
            .await
            .expect("stop() must not hang with live cloned producers");
        assert_eq!(res.written, 1);

        // The clones are still live and now inactive — record() does not revive
        // the closed channel.
        assert!(!p2.record(simple_entry()));
        drop(p3);
    }

    /// Resume within the same process: the first event of each stream carries
    /// RESYNC even if the origin connection generation did not change — the gap
    /// between sessions must not be silent. The root sidecar `_sessions.jsonl`
    /// records both sessions.
    #[tokio::test]
    async fn reattach_marks_resync_per_stream_and_records_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let hub = Hub::new();
        let cfg = ResearchRecorderConfig {
            dir: dir.path().to_path_buf(),
            ..Default::default()
        };

        let recorder1 = ResearchRecorder::start(&cfg, &hub).unwrap();
        let prod = hub.producer();
        let mut e1 = simple_entry();
        e1.stream = Some("binance:btc".into());
        e1.generation = Some(1);
        e1.source_seq = Some(10);
        assert!(prod.record(e1));
        recorder1.stop().await;

        // Resume over the SAME hub — same WS generation (1): the
        // the gap is opened by the recorder's detach/attach, not by a reconnection.
        let recorder2 = ResearchRecorder::start(&cfg, &hub).unwrap();
        let mut e2 = simple_entry();
        e2.stream = Some("binance:btc".into());
        e2.generation = Some(1);
        e2.source_seq = Some(11);
        assert!(prod.record(e2));
        recorder2.stop().await;

        let mut lines = Vec::new();
        for f in std::fs::read_dir(dir.path().join("binance_ws")).unwrap() {
            let f = f.unwrap();
            for l in std::fs::read_to_string(f.path()).unwrap().lines() {
                if !l.trim().is_empty() {
                    lines.push(serde_json::from_str::<RecordedLine>(l).unwrap());
                }
            }
        }
        lines.sort_by_key(|l| l.event.local_seq);
        assert_eq!(lines.len(), 2);
        assert!(
            !lines[0].event.quality.contains(QualityFlags::RESYNC),
            "first session of the process: no previous session to resume"
        );
        assert!(
            lines[1].event.quality.contains(QualityFlags::RESYNC),
            "first event of the stream after the re-attach → RESYNC"
        );

        let sessions = std::fs::read_to_string(dir.path().join("_sessions.jsonl")).unwrap();
        assert_eq!(
            sessions.lines().filter(|l| !l.trim().is_empty()).count(),
            2,
            "root sessions sidecar: one line per attach"
        );
    }

    /// `sink()` keeps returning `false` from `record` (the contract already
    /// used by the hook adapters in tests/research_events.rs).
    #[test]
    fn sink_keeps_returning_false() {
        let prod = EventProducer::sink();
        assert!(!prod.record(simple_entry()));
    }
}
