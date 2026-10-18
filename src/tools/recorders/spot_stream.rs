//! Compact spot/oracle series for research.
//!
//! Independent of the book dedup in `src/tools/recorders/analytics.rs`. That
//! recorder deliberately leaves `spot` out of `content_hash`, so a stable book
//! hides the real spot movement. This series records the spot when it CHANGES,
//! whether or not the book changed, and never duplicates depth (it does not
//! store books at all).
//!
//! Same pattern as `src/tools/recorders/analytics.rs`: a rusqlite writer on a
//! dedicated OS thread (blocking), fed by a BOUNDED `mpsc` channel from the
//! async side via `try_send` (it never blocks the producer), with a WAL
//! journal. Unlike `analytics.rs`, this DB is for RESEARCH, not operations: it
//! has no automatic purge or retention. If `file_path` already exists it is
//! only opened and migrated with `CREATE TABLE IF NOT EXISTS`; an existing file
//! is never moved, renamed or purged. It does not touch the holdout.
//!
//! ## Wiring (`main.rs`, right after `analytics_recorder`)
//!
//! The module is wired in `main` as follows:
//!
//! 1. `let recorder = SpotStreamRecorder::start(&db_path, channel_capacity)?;` once
//!    at startup, with the path of the research DB (a config parameter,
//!    separate from `hrf_market_analytics*.sqlite`).
//! 2. For each asset in the `vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>`
//!    mapping that `analytics_recorder`/strategies already use, spawn a small
//!    async task with its own cloned `recorder.producer()`:
//!    ```ignore
//!    let producer = recorder.producer();
//!    let mut rx = vol_rx.clone();
//!    tokio::spawn(async move {
//!        loop {
//!            tokio::select! {
//!                _ = rx.changed() => {
//!                    let snap = rx.borrow_and_update().clone();
//!                    let now_ms = chrono::Utc::now().timestamp_millis();
//!                    for sample in from_vol_snapshot(&snap, Instant::now(), now_ms) {
//!                        producer.send(sample);
//!                    }
//!                }
//!                _ = shutdown.changed() => { if *shutdown.borrow() { break; } }
//!            }
//!        }
//!    });
//!    ```
//! 3. On shutdown: let those tasks finish (they drop their cloned `Producer`),
//!    then call `recorder.close().await`, which returns a `CloseResult` for
//!    `main` to log or await.
//!
//! ## Known limitation: `source_ts_ms` from `VolSnapshot` is approximate
//!
//! `VolSnapshot` (see `src/types.rs`) does NOT expose an origin timestamp in
//! epoch-ms, only `updated_at`, a `crate::time::Instant` (monotonic, with no
//! relation to the wall clock). `oracle::price::AssetPrice` DOES carry a real
//! `exchange_ts_ms` (the Binance trade timestamp), but `oracle::vol::publish`
//! does not re-export it in `VolSnapshot` today. This module's
//! `from_vol_snapshot` approximates `source_ts_ms` by subtracting the monotonic
//! time elapsed since the last real tick from a wall-clock reference taken by
//! the caller at the same instant; see its doc-comment. If the exact trade
//! timestamp is needed, the right path is to extend `VolSnapshot` with
//! `exchange_ts_ms` (out of scope for this module).
//!
//! ## SQL schema
//!
//! - `spot_samples(id, asset, source, price TEXT, source_ts_ms, recv_ts_ms, avail_ts_ms)`
//!   — one row per written sample (after dedup). `price` is TEXT (exact
//!   Decimal, never floats), like the monetary columns elsewhere in the
//!   project.
//! - `spot_stream_stats(id, ts_ms, enqueued, written_count, deduplicated, discarded_count)`
//!   — a cumulative snapshot per processed batch and a final one at close. It
//!   lets saturation be detected by querying the table, with no extra
//!   instrumentation.

use crate::time::Instant;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use rusqlite::{params, Connection};
use rust_decimal::Decimal;
use tokio::sync::{mpsc, watch};

use crate::research::events::EventProducer;
use crate::types::VolSnapshot;

const TABLE_SAMPLES: &str = "spot_samples";
const TABLE_STATS: &str = "spot_stream_stats";

/// Maximum number of samples the writer drains at once before opening and
/// committing a transaction. Avoids unbounded transactions when the producer bursts.
const BATCH_MAX: usize = 500;

// ── Public types ───────────────────────────────────────────────────────────

/// A single spot/oracle sample for an asset, from a specific source
/// (`"binance"`, …).
#[derive(Debug, Clone, PartialEq)]
pub struct SpotSample {
    pub asset: String,
    pub source: String,
    /// Asset price. Never floats: exact `Decimal`, persisted as TEXT.
    pub price: Decimal,
    /// ORIGIN timestamp (exchange/oracle), epoch-ms.
    pub source_ts_ms: i64,
    /// LOCAL reception time, epoch-ms.
    pub recv_ts_ms: i64,
    /// Time at which the sample became available to the strategies, epoch-ms.
    /// Today it always equals `recv_ts_ms`: the producer consumes the same
    /// `watch<VolSnapshot>` as the strategies, with no queue or intermediate
    /// batching step between "received" and "available". It is stored as its
    /// own explicit field (not derived from `recv_ts_ms` in the schema) so the
    /// table does not need a migration if a real intermediate step is ever added.
    pub avail_ts_ms: i64,
}

/// Totals returned by the recorder on close (`SpotStreamRecorder::close`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CloseResult {
    pub enqueued: u64,
    pub written: u64,
    pub deduped: u64,
    pub discarded: u64,
}

// ── Producer (non-blocking side) ──────────────────────────────────────────

/// Counters shared by all the cloned `Producer`s of the same
/// `SpotStreamRecorder`. Atomic, because they can live in different async
/// tasks (one per asset; see the module doc).
#[derive(Debug, Default)]
struct ProducerCounters {
    enqueued: AtomicU64,
    discarded: AtomicU64,
}

/// Cloneable, non-blocking handle for queueing samples to the writer.
#[derive(Clone)]
pub struct Producer {
    tx: mpsc::Sender<SpotSample>,
    counters: Arc<ProducerCounters>,
}

impl Producer {
    /// Queues `sample` without blocking (`try_send`). `true` = queued,
    /// `false` = discarded (channel full due to saturation, or writer down).
    /// Both cases count as "discarded" and differ only by the metric's
    /// `reason` label. The caller does not need the detail to decide anything:
    /// losing a research sample never blocks or affects trading.
    pub fn send(&self, sample: SpotSample) -> bool {
        let asset = sample.asset.clone();
        let source = sample.source.clone();
        match self.tx.try_send(sample) {
            Ok(()) => {
                self.counters.enqueued.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("hrf_spot_stream_queued_total", "asset" => asset, "source" => source)
                    .increment(1);
                true
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.counters.discarded.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("hrf_spot_stream_discarded_total", "asset" => asset, "source" => source, "reason" => "saturation")
                    .increment(1);
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.counters.discarded.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("hrf_spot_stream_discarded_total", "asset" => asset, "source" => source, "reason" => "dead_writer")
                    .increment(1);
                false
            }
        }
    }
}

// ── Recorder (SQLite writer on a dedicated thread) ───────────────────────────

#[derive(Debug, Clone, Copy, Default)]
struct WriterStats {
    written: u64,
    deduped: u64,
}

/// Spot/oracle recorder: opens and migrates the research DB and starts the
/// writer thread. See the module doc for the intended wiring.
pub struct SpotStreamRecorder {
    tx: mpsc::Sender<SpotSample>,
    counters: Arc<ProducerCounters>,
    writer: Option<JoinHandle<WriterStats>>,
}

impl SpotStreamRecorder {
    /// Opens (or migrates, if it already exists) the DB at `db_path` and starts
    /// the writer thread. `channel_capacity` is the size of the bounded channel
    /// between producers and writer. It is a parameter so tests can force
    /// saturation; in production a reasonable constant is enough.
    pub fn start(db_path: &Path, channel_capacity: usize) -> rusqlite::Result<Self> {
        let conn = open_and_migrate(db_path)?;
        let (tx, rx) = mpsc::channel(channel_capacity);
        let counters = Arc::new(ProducerCounters::default());
        let writer_counters = counters.clone();
        let writer = std::thread::Builder::new()
            .name("spot_stream_writer".to_string())
            .spawn(move || run_writer(conn, rx, writer_counters))
            .expect("spot_stream: failed to spawn the writer thread");
        Ok(Self {
            tx,
            counters,
            writer: Some(writer),
        })
    }

    /// A new producer handle, cloned from the internal channel: one `Producer`
    /// per task consuming a `watch<VolSnapshot>` (see the module doc).
    pub fn producer(&self) -> Producer {
        Producer {
            tx: self.tx.clone(),
            counters: self.counters.clone(),
        }
    }

    /// Orderly close: drops the internal `Sender` and waits for the writer
    /// thread to drain what was left in the channel. The `mpsc::Receiver` keeps
    /// returning queued items until the channel is empty, even after it closes,
    /// so no manual `try_recv` drain is needed. It then writes the final stats
    /// and returns the totals so `main` can log or await them.
    ///
    /// If cloned `Producer`s are still alive in other tasks, the channel does
    /// NOT close until they are dropped. The wiring in `main` must stop the
    /// producers (or let their tasks finish) before calling this.
    pub async fn close(mut self) -> CloseResult {
        let writer = self.writer.take().expect("close() was already called");
        let counters = self.counters.clone();
        drop(self); // drops self.tx; the channel closes once no Producer is left alive
        let stats = tokio::task::spawn_blocking(move || writer.join())
            .await
            .expect("spot_stream: panicked waiting for the writer thread to join")
            .expect("spot_stream: the writer thread panicked");
        CloseResult {
            enqueued: counters.enqueued.load(Ordering::Relaxed),
            discarded: counters.discarded.load(Ordering::Relaxed),
            written: stats.written,
            deduped: stats.deduped,
        }
    }
}

// ── Toggleable tool adapter ─────────────────────────────────────

/// Full lifecycle as `ToolId::SpotStream`: starts the recorder, one task per
/// asset that feeds its `Producer` from the `watch<VolSnapshot>` (this used to
/// live inline in `main.rs`; it is here so `ToolSupervisor` can stop it by
/// signal), and the orderly close on `stop`.
///
/// Every `Producer` cloned into the per-asset tasks must be dropped before
/// `close()`. Otherwise the channel never closes and `close()` hangs. That is
/// why `stop` waits on a `JoinSet` (not on individual `watch`es per asset): as
/// each subtask ends it drops its cloned `Producer`, and only then is `close()`
/// called.
pub async fn run(
    db_path: PathBuf,
    channel_capacity: usize,
    vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    research_prod: EventProducer,
    mut stop: watch::Receiver<bool>,
) {
    let recorder = match SpotStreamRecorder::start(&db_path, channel_capacity) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                error = %e,
                db_path = %db_path.display(),
                "spot_stream: failed to start the recorder; the spot time series is disabled"
            );
            return;
        }
    };

    let mut per_asset = tokio::task::JoinSet::new();
    for (asset, vol_rx) in vol_rxs {
        let producer = recorder.producer();
        let mut rx = vol_rx;
        let research_prod_task = research_prod.clone();
        // Besides the spot DB, each sample also enters the causal events ledger
        // (SYNTHETIC: a view derived from the oracle via `VolSnapshot`).
        let research_gen = crate::research::events::new_generation();
        per_asset.spawn(async move {
            while rx.changed().await.is_ok() {
                let snap = rx.borrow_and_update().clone();
                let now = crate::time::Instant::now();
                let now_ms = crate::time::now().timestamp_millis();
                for sample in from_vol_snapshot(&snap, now, now_ms) {
                    producer.send(sample.clone());
                    // Do not build the event if the hub is inactive.
                    if research_prod_task.is_active() {
                        research_prod_task.record(event_from_sample(&sample, research_gen));
                    }
                }
            }
            tracing::debug!(%asset, "spot_stream: oracle channel closed, subtask exiting");
        });
    }

    while stop.changed().await.is_ok() {
        if *stop.borrow() {
            break;
        }
    }
    // The per-asset subtasks hold no state of their own to flush. What matters
    // is that they drop their cloned `Producer` before `close()`, so aborting
    // them is safe.
    per_asset.abort_all();
    while per_asset.join_next().await.is_some() {}

    let totals = recorder.close().await;
    tracing::info!(
        enqueued = totals.enqueued,
        written = totals.written,
        deduped = totals.deduped,
        discarded = totals.discarded,
        "spot_stream: closed"
    );
}

// ── Migration / opening ─────────────────────────────────────────────────────

/// Opens `db_path` (creating it if missing) and ensures the schema with
/// `CREATE TABLE IF NOT EXISTS`. It never moves, renames or purges an existing
/// file. For a file that already holds data, this is a schema no-op that keeps
/// the rows already recorded.
fn open_and_migrate(db_path: &Path) -> rusqlite::Result<Connection> {
    if let Some(parent) = db_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let conn = Connection::open(db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    ensure_tables(&conn)?;
    Ok(conn)
}

fn ensure_tables(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(&format!(
        "
        CREATE TABLE IF NOT EXISTS {TABLE_SAMPLES} (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            asset        TEXT    NOT NULL,
            source       TEXT    NOT NULL,
            price        TEXT    NOT NULL,
            source_ts_ms INTEGER NOT NULL,
            recv_ts_ms   INTEGER NOT NULL,
            avail_ts_ms  INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_{TABLE_SAMPLES}_asset_source ON {TABLE_SAMPLES}(asset, source);
        CREATE INDEX IF NOT EXISTS idx_{TABLE_SAMPLES}_source_ts ON {TABLE_SAMPLES}(source_ts_ms);

        CREATE TABLE IF NOT EXISTS {TABLE_STATS} (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms            INTEGER NOT NULL,
            enqueued         INTEGER NOT NULL,
            written_count    INTEGER NOT NULL,
            deduplicated     INTEGER NOT NULL,
            discarded_count  INTEGER NOT NULL
        );
        "
    ))
}

// ── Writer (dedicated OS thread, same pattern as analytics_recorder.rs) ──────────────

/// Entry point of the writer thread. Blocks until the channel is closed AND
/// empty (`mpsc::Receiver::blocking_recv` keeps returning queued items after
/// the close, before it returns `None`). That is why `close()` needs no
/// separate manual drain.
fn run_writer(
    mut conn: Connection,
    mut rx: mpsc::Receiver<SpotSample>,
    counters: Arc<ProducerCounters>,
) -> WriterStats {
    // Last sample WRITTEN per (asset, source). Content dedup: a sample is not
    // rewritten if (price, source_ts_ms) is identical to the last written pair.
    let mut last_written: std::collections::HashMap<(String, String), (Decimal, i64)> =
        std::collections::HashMap::new();
    let mut written: u64 = 0;
    let mut deduped: u64 = 0;

    while let Some(first) = rx.blocking_recv() {
        let mut batch = vec![first];
        while batch.len() < BATCH_MAX {
            match rx.try_recv() {
                Ok(m) => batch.push(m),
                Err(_) => break,
            }
        }

        let tx_sql = match conn.transaction() {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(error = %e, "spot_stream: could not open transaction; batch discarded");
                continue;
            }
        };
        for sample in &batch {
            // Content dedup: if (price, source_ts_ms) matches the last sample
            // written for that (asset, source), the sample adds nothing. It is
            // counted and discarded. recv/avail do not take part: a re-delivered
            // identical observation is still the same observation.
            let key = (sample.asset.clone(), sample.source.clone());
            if last_written.get(&key) == Some(&(sample.price, sample.source_ts_ms)) {
                deduped += 1;
                continue;
            }
            match insert_sample(&tx_sql, sample) {
                Ok(()) => {
                    last_written.insert(key, (sample.price, sample.source_ts_ms));
                    written += 1;
                    metrics::counter!("hrf_spot_stream_written_total", "asset" => sample.asset.clone(), "source" => sample.source.clone())
                        .increment(1);
                }
                Err(e) => {
                    tracing::error!(error = %e, asset = %sample.asset, source = %sample.source, "spot_stream: error inserting sample");
                }
            }
        }
        if let Err(e) = tx_sql.commit() {
            tracing::error!(error = %e, "spot_stream: error committing transaction");
        }
        persist_stats(&conn, &counters, written, deduped);
    }

    persist_stats(&conn, &counters, written, deduped);
    WriterStats { written, deduped }
}

fn insert_sample(conn: &Connection, sample: &SpotSample) -> rusqlite::Result<()> {
    conn.execute(
        &format!(
            "INSERT INTO {TABLE_SAMPLES} (asset, source, price, source_ts_ms, recv_ts_ms, avail_ts_ms)
             VALUES (?1,?2,?3,?4,?5,?6)"
        ),
        params![
            sample.asset,
            sample.source,
            sample.price.to_string(),
            sample.source_ts_ms,
            sample.recv_ts_ms,
            sample.avail_ts_ms,
        ],
    )?;
    Ok(())
}

/// Dumps a cumulative snapshot of the four counters to `spot_stream_stats`.
/// Called after each processed batch and once more at close (see `run_writer`),
/// so saturation can be detected by querying the table without depending on
/// the process being alive to expose metrics.
fn persist_stats(conn: &Connection, counters: &ProducerCounters, written: u64, deduped: u64) {
    let ts_ms = chrono::Utc::now().timestamp_millis();
    let enqueued = counters.enqueued.load(Ordering::Relaxed);
    let discarded = counters.discarded.load(Ordering::Relaxed);
    if let Err(e) = conn.execute(
        &format!(
            "INSERT INTO {TABLE_STATS} (ts_ms, enqueued, written_count, deduplicated, discarded_count)
             VALUES (?1,?2,?3,?4,?5)"
        ),
        params![
            ts_ms,
            enqueued as i64,
            written as i64,
            deduped as i64,
            discarded as i64
        ],
    ) {
        tracing::error!(error = %e, "spot_stream: error persisting statistics");
    }
}

// ── Pure adapter: VolSnapshot → SpotSample ────────────────────────────────

/// Derives the binance `SpotSample` from a `VolSnapshot`. PURE function: no
/// I/O, testable without channels or a DB.
///
/// `VolSnapshot` has no origin timestamp in epoch-ms (see the limitation
/// documented in the module doc). `now_instant`/`now_ms` must be a pair
/// (monotonic Instant, epoch-ms) taken by the CALLER at the same moment.
/// `source_ts_ms` is approximated by subtracting from `now_ms` the monotonic
/// time elapsed since the last real tick (`updated_at`).
///
/// `last_price <= 0` (cold start: a `VolSnapshot::default()` not yet seeded)
/// produces no sample: the signal is absent, not a real zero.
pub fn from_vol_snapshot(
    snapshot: &VolSnapshot,
    now_instant: Instant,
    now_ms: i64,
) -> Vec<SpotSample> {
    // An absent signal is not a real zero: on cold start (last_price=0 in the
    // not-yet-seeded VolSnapshot) no sample is recorded.
    let mut samples = Vec::with_capacity(1);
    if snapshot.last_price > Decimal::ZERO {
        let dt = now_instant.saturating_duration_since(snapshot.updated_at);
        samples.push(SpotSample {
            asset: snapshot.asset.clone(),
            source: "binance".to_string(),
            price: snapshot.last_price,
            source_ts_ms: now_ms - dt.as_millis() as i64,
            recv_ts_ms: now_ms,
            avail_ts_ms: now_ms,
        });
    }
    samples
}

/// Pure adapter `SpotSample` → `EventInput` for the research recorder
/// (`crate::research::events`). The `spot_stream:<asset>` task in `main.rs`
/// registers it for each sample.
///
/// IMPORTANT (same limitation as `from_vol_snapshot`): the sample's
/// `source_ts_ms` is APPROXIMATE. `VolSnapshot` does not expose the exchange's
/// real trade timestamp. So `origin_ts_ms` stays `None` (never invented; the
/// recorder sets the NO_REMOTE_TS mark), and the approximate ts travels in the
/// payload labeled `source_ts_ms_approx`. The event also carries the
/// `SYNTHETIC` mark (via `quality_extra`): it is a view derived by the bot (an
/// aggregation of the oracle via `VolSnapshot`), not a raw venue observation.
///
/// `dedup_key` = hash of the same stable fields this module's dedup uses
/// (asset+source+price+source_ts_ms), so a re-delivered identical sample is not
/// duplicated. For raw Binance ticks with a real `trade_id`/`trade_time`, use
/// `oracle::price::event_from_binance_tick`.
pub fn event_from_sample(
    m: &SpotSample,
    generation: u64,
) -> crate::research::events::EventInput {
    use crate::research::contract::{EventKind, EventSource, QualityFlags};
    use crate::research::events::{stable_hash, EventInput};

    let source = if m.source == "binance" {
        EventSource::BinanceWs
    } else {
        // Source unknown to the contract vocabulary. The original string stays in
        // the payload; the typed source is Internal.
        EventSource::Internal
    };
    let mut e = EventInput::new(
        source,
        EventKind::PriceTick,
        serde_json::json!({
            "price": m.price.to_string(),
            "source": m.source,
            "source_ts_ms_approx": m.source_ts_ms,
        }),
        m.recv_ts_ms,
    );
    e.asset = Some(m.asset.clone());
    e.avail_ts_ms = m.avail_ts_ms;
    e.origin_ts_ms = None; // approximate, not real: never invented
    e.stream = Some(format!("spot_stream:{}", m.asset));
    e.generation = Some(generation);
    e.dedup_key = Some(stable_hash((
        "spot_sample",
        &m.asset,
        &m.source,
        m.price,
        m.source_ts_ms,
    )));
    e.quality_extra = QualityFlags::SYNTHETIC;
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use std::time::Duration;

    fn sample(asset: &str, source: &str, price: Decimal, source_ts_ms: i64) -> SpotSample {
        SpotSample {
            asset: asset.to_string(),
            source: source.to_string(),
            price,
            source_ts_ms,
            recv_ts_ms: source_ts_ms + 5,
            avail_ts_ms: source_ts_ms + 5,
        }
    }

    // ── Producer / saturation ──────────────────────────────────────────────

    /// Capacity-1 channel with no reader: the first send fills the channel, and
    /// the following ones are discarded by saturation. They are counted and,
    /// once the snapshot is persisted, queryable in `spot_stream_stats`.
    #[test]
    fn channel_saturation_counts_and_persists_discards() {
        let (tx, _rx) = mpsc::channel(1);
        let counters = Arc::new(ProducerCounters::default());
        let producer = Producer {
            tx,
            counters: counters.clone(),
        };

        assert!(producer.send(sample("btc", "binance", dec!(100), 1)));
        assert!(!producer.send(sample("btc", "binance", dec!(101), 2)));
        assert!(!producer.send(sample("btc", "binance", dec!(102), 3)));

        assert_eq!(counters.enqueued.load(Ordering::Relaxed), 1);
        assert_eq!(counters.discarded.load(Ordering::Relaxed), 2);

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("stats.sqlite");
        let conn = open_and_migrate(&db_path).unwrap();
        persist_stats(&conn, &counters, 0, 0);

        let discarded: i64 = conn
            .query_row(
                &format!("SELECT discarded_count FROM {TABLE_STATS} ORDER BY id DESC LIMIT 1"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            discarded, 2,
            "saturation must be detectable by querying spot_stream_stats"
        );
    }

    // ── Writer: dedup, round trip, close ───────────────────────────────

    #[tokio::test]
    async fn spot_change_without_book_change_generates_new_rows() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("spot.sqlite");
        let recorder = SpotStreamRecorder::start(&db_path, 16).unwrap();
        let producer = recorder.producer();

        // Two samples from the SAME (asset, source) with different prices. The
        // book plays no part at all: this series does not consult it.
        assert!(producer.send(sample("btc", "binance", dec!(100000), 1000)));
        assert!(producer.send(sample("btc", "binance", dec!(100050), 2000)));

        // close() requires that no Producer is left alive. Otherwise the channel
        // does not close and the writer join hangs.
        drop(producer);
        let result = recorder.close().await;
        assert_eq!(result.enqueued, 2);
        assert_eq!(result.written, 2);
        assert_eq!(result.deduped, 0);
        assert_eq!(result.discarded, 0);

        let conn = Connection::open(&db_path).unwrap();
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE_SAMPLES}"), [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn identical_sample_is_deduplicated_and_counted() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("spot.sqlite");
        let recorder = SpotStreamRecorder::start(&db_path, 16).unwrap();
        let producer = recorder.producer();

        producer.send(sample("eth", "binance", dec!(3500.5), 1000));
        // Same (asset, source, price, source_ts_ms): different recv/avail must not
        // matter for the dedup; it is ONLY by (asset, source, price, ts).
        let mut repeated = sample("eth", "binance", dec!(3500.5), 1000);
        repeated.recv_ts_ms = 9999;
        repeated.avail_ts_ms = 9999;
        producer.send(repeated);

        drop(producer); // close() hangs if a Producer is still alive
        let result = recorder.close().await;
        assert_eq!(result.written, 1);
        assert_eq!(result.deduped, 1);

        let conn = Connection::open(&db_path).unwrap();
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE_SAMPLES}"), [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn times_and_price_are_preserved_exactly_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("spot.sqlite");
        let recorder = SpotStreamRecorder::start(&db_path, 16).unwrap();
        let producer = recorder.producer();

        let m = SpotSample {
            asset: "btc".to_string(),
            source: "binance".to_string(),
            price: dec!(63123.456789),
            source_ts_ms: 1_700_000_000_123,
            recv_ts_ms: 1_700_000_000_456,
            avail_ts_ms: 1_700_000_000_789,
        };
        producer.send(m.clone());
        drop(producer); // close() hangs if a Producer is still alive
        let result = recorder.close().await;
        assert_eq!(result.written, 1);

        // Reopen the DB (a new `Connection`, not the writer's internal `conn`).
        let conn = Connection::open(&db_path).unwrap();
        let (asset, source, price_txt, source_ts, recv_ts, avail_ts): (String, String, String, i64, i64, i64) =
            conn.query_row(
                &format!("SELECT asset, source, price, source_ts_ms, recv_ts_ms, avail_ts_ms FROM {TABLE_SAMPLES}"),
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
            )
            .unwrap();

        assert_eq!(asset, m.asset);
        assert_eq!(source, m.source);
        assert_eq!(
            price_txt.parse::<Decimal>().unwrap(),
            m.price,
            "price Decimal exact round trip"
        );
        assert_eq!(source_ts, m.source_ts_ms);
        assert_eq!(recv_ts, m.recv_ts_ms);
        assert_eq!(avail_ts, m.avail_ts_ms);
    }

    #[tokio::test]
    async fn idempotent_migration_keeps_previous_data() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("spot.sqlite");

        // First open: writes one row and closes.
        {
            let recorder = SpotStreamRecorder::start(&db_path, 16).unwrap();
            let producer = recorder.producer();
            producer.send(sample("btc", "binance", dec!(100000), 1000));
            drop(producer);
            let result = recorder.close().await;
            assert_eq!(result.written, 1);
        }

        // Second open on the SAME file (idempotent migration): the previous
        // data must still be there, and the new row is added.
        {
            let recorder = SpotStreamRecorder::start(&db_path, 16).unwrap();
            let producer = recorder.producer();
            producer.send(sample("eth", "binance", dec!(3500), 2000));
            drop(producer);
            let result = recorder.close().await;
            assert_eq!(result.written, 1);
        }

        let conn = Connection::open(&db_path).unwrap();
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE_SAMPLES}"), [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(
            count, 2,
            "the idempotent migration must not lose previous rows"
        );
    }

    // ── Pure adapter: from_vol_snapshot ──────────────────────────────────

    #[test]
    fn from_vol_snapshot_cold_start_produces_no_samples() {
        let snap = VolSnapshot::default(); // last_price=0 (cold start)
        let samples = from_vol_snapshot(&snap, Instant::now(), 1_700_000_000_000);
        assert!(
            samples.is_empty(),
            "cold start (VolSnapshot::default()) must not produce samples"
        );
    }

    #[test]
    fn from_vol_snapshot_derives_binance_with_approximate_ts() {
        let t0 = Instant::now();
        let updated_at = t0; // binance tick at t=0
        let now_instant = t0.checked_add(Duration::from_millis(500)).unwrap(); // "now" at t=500ms
        let now_ms = 1_700_000_000_000i64;

        let snap = VolSnapshot {
            asset: "btc".to_string(),
            last_price: dec!(101234),
            updated_at,
            ..VolSnapshot::default()
        };

        let samples = from_vol_snapshot(&snap, now_instant, now_ms);
        assert_eq!(samples.len(), 1);

        let binance = &samples[0];
        assert_eq!(binance.source, "binance");
        assert_eq!(binance.asset, "btc");
        assert_eq!(binance.price, dec!(101234));
        assert_eq!(binance.source_ts_ms, now_ms - 500);
        assert_eq!(binance.recv_ts_ms, now_ms);
        assert_eq!(binance.avail_ts_ms, now_ms);
    }
}
