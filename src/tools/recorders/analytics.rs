//! Market analytics recorder — READ-ONLY, isolated from the trading logic.
//!
//! On every second, records the COMPLETE book state (bids and asks of both
//! YES/NO sides, with their full depth — not just a reference price) of each
//! crypto up/down market the bot is already following, into a SQLite database
//! SEPARATE from `StateStore` (`hrf_state*.sqlite`). It never emits
//! `OrderIntent` and never touches `RiskGuard`, the executor or any strategy. It is
//! one more consumer of the book broadcast, like `accounting::markout::run` or
//! `settlement::post_close::run`.
//!
//! It subscribes once more to the `broadcast::Sender<Arc<BookSnapshot>>` that already
//! feeds all strategies (`book_tx` in `main.rs`), with no network I/O of its own:
//! the Polymarket WS connection already exists and already delivers this data through
//! `market_feed/`.
//!
//! Pattern: an in-memory cache updated by broadcast events, plus a
//! `tokio::time::interval` of 1s that records the LAST known state of each market
//! (one row per second per market). The WS arrives at an irregular rate, so
//! "one row per event" would produce an erratic cadence.
//! The SQLite writer runs on its OWN OS thread (rusqlite is blocking),
//! receiving rows through a BOUNDED `mpsc::Sender` (`DB_CHANNEL_CAPACITY`).
//! When the channel is full, the newest row is dropped instead of blocking the tick. The
//! analytics are backtesting input, not a money path (see the
//! doc-comment of `try_send_row`). It uses WAL journaling and never shares a connection
//! or table with `state.rs`.
//!
//! ## Failure isolation
//!
//! Like ANY other task in the `JoinSet` of `main.rs`, this task should only
//! end on the `shutdown` signal. `main.rs` treats the return of ANY task (normal
//! or by panic, without distinguishing the reason) as a reason to shut down the WHOLE bot. For that reason, the processing of
//! each book and each tick is wrapped in `catch_unwind`: an internal panic (corrupt row,
//! whatever it is) is logged and the main loop stays alive. It never propagates out of
//! `run`. The SQLite writer lives on a separate OS thread (not a tokio task), so it
//! cannot take down the runtime even if that thread dies: `run` simply stops being
//! able to persist (it detects the closed channel) and stays alive without
//! retrying, waiting for shutdown.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use rusqlite::{params, Connection};
use rust_decimal::Decimal;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use crate::market_feed::registry::MarketRegistry;
use crate::state::StateStore;
use crate::types::{BookSnapshot, Level, Mode, VolSnapshot};

/// One market per row per second (column `market_condition_id`). Unlike the
/// analyzer's `db.rs`, which partitions by (coin, timeframe) into separate tables,
/// one table is enough here: the number of crypto up/down markets active at once is
/// small, and a single table with indexes is sufficient.
const TABLE: &str = "book_snapshots";

// ── Persisted row ───────────────────────────────────────────────────────────

/// Everything `BookSnapshot` carries for a market at the instant of a tick,
/// plus the best bid/ask/mid derived from each side (defensive, see
/// `best_bid_ask_mid`) and the full depth serialized as JSON.
#[derive(Debug, Clone)]
struct Row {
    ts_ms: i64,
    market_condition_id: String,
    slug: String,
    clear_at: i64,
    market_open_ts: i64,
    neg_risk: bool,
    reference_price: Option<Decimal>,
    fee_rate_bps: Option<u32>,
    parabolic_fee_rate: Option<Decimal>,
    tick_size: Option<Decimal>,
    yes_token_id: String,
    yes_best_bid: Option<Decimal>,
    yes_best_ask: Option<Decimal>,
    yes_mid: Option<Decimal>,
    yes_bids_json: String,
    yes_asks_json: String,
    no_token_id: String,
    no_best_bid: Option<Decimal>,
    no_best_ask: Option<Decimal>,
    no_mid: Option<Decimal>,
    no_bids_json: String,
    no_asks_json: String,
    /// Spot price of the market's asset from the Binance oracle at the instant of the
    /// row (short-term predictor for the adverse-selection filter
    /// of market making). `None` if the asset does not have a real `VolSnapshot` yet
    /// (cold start or reconnection: the `watch` channel is pre-seeded with
    /// `VolSnapshot::default()`, whose `last_price` is `Decimal::ZERO`, so it is
    /// filtered explicitly instead of relying on the mapping lookup failing).
    /// Deliberately OUTSIDE `content_hash`; see its doc-comment.
    spot: Option<Decimal>,
}

/// Best bid/ask/mid of one side of the book, sorting the levels DEFENSIVELY
/// before taking the first one.
///
/// `HalfBook::levels` of `BookSnapshot` is ALREADY sorted by
/// `market_feed/handle.rs` and `resync.rs` on the normal path (see the doc-comment
/// of `HalfBook`), but this function does not rely on that guarantee: Polymarket's
/// real API does NOT guarantee order. This was already learned and fixed once in this
/// project (`src/accounting/mark_price.rs::mid_from_raw`, a real bug: without
/// sorting, `.first()` returned a garbage price from the wrong extreme and inflated
/// `positions_value` by ~1.8× over the real cost). Same criterion here: a pure,
/// defensive function, testable with unsorted data.
fn best_bid_ask_mid(
    bids: &[Level],
    asks: &[Level],
) -> (Option<Decimal>, Option<Decimal>, Option<Decimal>) {
    let mut bids = bids.to_vec();
    let mut asks = asks.to_vec();
    bids.sort_unstable_by_key(|l| std::cmp::Reverse(l.price)); // descending: best bid (highest price) first
    asks.sort_unstable_by_key(|a| a.price); // ascending: best ask (lowest price) first
    let best_bid = bids.first().map(|l| l.price);
    let best_ask = asks.first().map(|l| l.price);
    let mid = match (best_bid, best_ask) {
        (Some(b), Some(a)) => Some((b + a) / Decimal::TWO),
        (Some(b), None) => Some(b),
        (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    (best_bid, best_ask, mid)
}

/// Operational bugfix: serializing the COMPLETE depth with no limit (up to ~99 levels
/// per side in busy markets, times 4 lists per market, times 1 row per second)
/// grew `hrf_market_analytics.sqlite` to 42 GB in about 5 hours of real operation. At
/// that rate (~8.4 GB/h) the disk would have filled in a little over a day. Levels far
/// from the top are rarely touched in practice (liquidity is concentrated near the best
/// price) and add nothing material to modeling, so each side is capped to its best
/// `MAX_DEPTH_LEVELS` levels, sorted defensively (same criterion as
/// `best_bid_ask_mid`), before serializing.
const MAX_DEPTH_LEVELS: usize = 15;

fn top_levels_json(levels: &[Level], descending: bool) -> String {
    let mut levels = levels.to_vec();
    if descending {
        levels.sort_unstable_by_key(|l| std::cmp::Reverse(l.price));
    } else {
        levels.sort_unstable_by_key(|a| a.price);
    }
    levels.truncate(MAX_DEPTH_LEVELS);
    serde_json::to_string(&levels).unwrap_or_else(|_| "[]".to_string())
}

impl Row {
    /// Hash of all the content of the row except `ts_ms`. Used to deduplicate
    /// consecutive identical snapshots of the same market.
    ///
    /// Operational bugfix: measured in production, 96.4% of the recorded rows were the
    /// book byte-for-byte identical to the previous row of the same market (the 1s tick
    /// dumps the LAST known state even when no new WS event has arrived). That was
    /// ~65 GB/day, filling the disk in about 4 days. Since they add no new observation,
    /// they are discarded before reaching the writer: a row is persisted only when its
    /// content changes relative to the last recorded row of that market.
    ///
    /// `spot` is deliberately NOT part of this hash: it changes almost every second
    /// with the Binance oracle, and if it were included, practically no row would ever be
    /// deduplicated, reproducing the same disk incident (~65 GB/day) that the
    /// deduplication above already solved.
    fn content_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.clear_at.hash(&mut h);
        self.market_open_ts.hash(&mut h);
        self.neg_risk.hash(&mut h);
        self.reference_price.hash(&mut h);
        self.fee_rate_bps.hash(&mut h);
        self.parabolic_fee_rate.hash(&mut h);
        self.tick_size.hash(&mut h);
        self.yes_best_bid.hash(&mut h);
        self.yes_best_ask.hash(&mut h);
        self.yes_mid.hash(&mut h);
        self.yes_bids_json.hash(&mut h);
        self.yes_asks_json.hash(&mut h);
        self.no_best_bid.hash(&mut h);
        self.no_best_ask.hash(&mut h);
        self.no_mid.hash(&mut h);
        self.no_bids_json.hash(&mut h);
        self.no_asks_json.hash(&mut h);
        h.finish()
    }
}

/// Builds the row to persist for a market at the instant `ts_ms`.
/// Serializes the best `MAX_DEPTH_LEVELS` levels of each side as JSON
/// (not just the best price), in addition to the separate indexed columns for
/// best_bid/best_ask/mid.
///
/// `vol_rxs` is the same asset→`watch::Receiver<VolSnapshot>` mapping that
/// `settlement::post_close::run` and `reference::run` already consume. The asset is derived from the slug
/// with `oracle::asset_from_slug` (same criterion across the whole project).
fn build_row(
    book: &BookSnapshot,
    ts_ms: i64,
    vol_rxs: &HashMap<String, watch::Receiver<VolSnapshot>>,
) -> Row {
    let (yes_bb, yes_ba, yes_mid) = best_bid_ask_mid(&book.yes.bids.levels, &book.yes.asks.levels);
    let (no_bb, no_ba, no_mid) = best_bid_ask_mid(&book.no.bids.levels, &book.no.asks.levels);
    let spot = crate::oracle::asset_from_slug(&book.slug)
        .and_then(|asset| vol_rxs.get(&asset))
        .map(|rx| rx.borrow().last_price)
        .filter(|p| *p > Decimal::ZERO);
    Row {
        ts_ms,
        market_condition_id: book.market_condition_id.clone(),
        slug: book.slug.clone(),
        clear_at: book.clear_at,
        market_open_ts: book.market_open_ts,
        neg_risk: book.neg_risk,
        reference_price: book.reference_price,
        fee_rate_bps: book.fee_rate_bps,
        parabolic_fee_rate: book.parabolic_fee_rate,
        tick_size: book.tick_size,
        yes_token_id: book.yes.token_id.clone(),
        yes_best_bid: yes_bb,
        yes_best_ask: yes_ba,
        yes_mid,
        yes_bids_json: top_levels_json(&book.yes.bids.levels, true),
        yes_asks_json: top_levels_json(&book.yes.asks.levels, false),
        no_token_id: book.no.token_id.clone(),
        no_best_bid: no_bb,
        no_best_ask: no_ba,
        no_mid,
        no_bids_json: top_levels_json(&book.no.bids.levels, true),
        no_asks_json: top_levels_json(&book.no.asks.levels, false),
        spot,
    }
}

// ── SQLite writer (dedicated OS thread) ─────────────────────────────────────

/// `busy_timeout` shared by the writer (`run_writer`) and the purger
/// (`run_purger`). WAL allows only one writer at a time on the same
/// file, and both threads write to it.
///
/// Operational bugfix: before this fix the writer used 5s and the purger 30s. A purge
/// batch (thousands of rows) under heavy disk contention can take more than 5s to
/// release the lock. The writer, the side that cannot afford to fail (each row is a
/// non-repeatable observation of the 1s tick), was the one that timed out first and
/// lost rows (`SQLITE_BUSY`, logged in `run_writer`). The asymmetry was backwards:
/// purging is background maintenance where waiting longer costs nothing, so both
/// share the more generous value instead of the critical side having the shorter
/// margin (same criterion already applied in `src/settlement/orphan.rs`:
/// insufficient margin, not a broken configuration).
const BUSY_TIMEOUT_SECS: u64 = 30;

fn ensure_table(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(&format!(
        "
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms                INTEGER NOT NULL,
            market_condition_id  TEXT    NOT NULL,
            slug                 TEXT    NOT NULL,
            clear_at             INTEGER NOT NULL,
            market_open_ts       INTEGER NOT NULL,
            neg_risk             INTEGER NOT NULL,
            reference_price      TEXT,
            fee_rate_bps         INTEGER,
            parabolic_fee_rate   TEXT,
            tick_size            TEXT,
            yes_token_id         TEXT    NOT NULL,
            yes_best_bid         TEXT,
            yes_best_ask         TEXT,
            yes_mid              TEXT,
            yes_bids_json        TEXT    NOT NULL,
            yes_asks_json        TEXT    NOT NULL,
            no_token_id          TEXT    NOT NULL,
            no_best_bid          TEXT,
            no_best_ask          TEXT,
            no_mid               TEXT,
            no_bids_json         TEXT    NOT NULL,
            no_asks_json         TEXT    NOT NULL,
            spot                 TEXT,
            UNIQUE(market_condition_id, ts_ms)
        );
        CREATE INDEX IF NOT EXISTS idx_{TABLE}_market ON {TABLE}(market_condition_id);
        CREATE INDEX IF NOT EXISTS idx_{TABLE}_ts     ON {TABLE}(ts_ms);
        "
    ))
}

fn insert_row(conn: &Connection, row: &Row) -> rusqlite::Result<()> {
    conn.execute(
        &format!(
            "INSERT OR IGNORE INTO {TABLE} (
                ts_ms, market_condition_id, slug, clear_at, market_open_ts, neg_risk,
                reference_price, fee_rate_bps, parabolic_fee_rate, tick_size,
                yes_token_id, yes_best_bid, yes_best_ask, yes_mid, yes_bids_json, yes_asks_json,
                no_token_id, no_best_bid, no_best_ask, no_mid, no_bids_json, no_asks_json, spot
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23)"
        ),
        params![
            row.ts_ms,
            row.market_condition_id,
            row.slug,
            row.clear_at,
            row.market_open_ts,
            row.neg_risk as i64,
            row.reference_price.map(|d| d.to_string()),
            row.fee_rate_bps,
            row.parabolic_fee_rate.map(|d| d.to_string()),
            row.tick_size.map(|d| d.to_string()),
            row.yes_token_id,
            row.yes_best_bid.map(|d| d.to_string()),
            row.yes_best_ask.map(|d| d.to_string()),
            row.yes_mid.map(|d| d.to_string()),
            row.yes_bids_json,
            row.yes_asks_json,
            row.no_token_id,
            row.no_best_bid.map(|d| d.to_string()),
            row.no_best_ask.map(|d| d.to_string()),
            row.no_mid.map(|d| d.to_string()),
            row.no_bids_json,
            row.no_asks_json,
            row.spot.map(|d| d.to_string()),
        ],
    )?;
    Ok(())
}

/// Writer entry point. Blocks the calling OS thread until the channel closes.
/// It is NEVER called from the tokio runtime, because rusqlite is blocking.
fn run_writer(db_path: &str, mut rx: mpsc::Receiver<Row>) -> anyhow::Result<()> {
    if let Some(parent) = Path::new(db_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let conn = Connection::open(db_path)?;
    // WAL: reads (offline analysis) do not block writes; NORMAL: fsync only at
    // checkpoints. `auto_vacuum=INCREMENTAL` only takes effect if the DB is being
    // created (empty file, no tables yet). SQLite ignores the mode change on an
    // existing DB with data, which stays in `NONE` unless migrated separately (see the
    // doc-comment of `run_purger`). With this PRAGMA set since the first open,
    // `PRAGMA incremental_vacuum` in the purge thread can return free pages to the
    // filesystem on new DBs.
    conn.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA auto_vacuum=INCREMENTAL;",
    )?;
    // Two OS threads now write to the same file (this writer and
    // `run_purger`). WAL allows only one writer at a time. Without a
    // busy_timeout, the writer would fail with SQLITE_BUSY instead of
    // waiting its turn. It uses the same value as the purger (`BUSY_TIMEOUT_SECS`, see
    // its doc-comment), because this is the side that cannot afford to fail.
    conn.busy_timeout(Duration::from_secs(BUSY_TIMEOUT_SECS))?;
    ensure_table(&conn)?;
    info!(
        db_path,
        "analytics_recorder: SQLite writer started (WAL)"
    );

    while let Some(row) = rx.blocking_recv() {
        if let Err(e) = insert_row(&conn, &row) {
            error!(error = %e, market = %row.market_condition_id, "analytics_recorder: error inserting row");
        }
    }
    info!("analytics_recorder: channel closed, SQLite writer finishing");
    Ok(())
}

// ── Retention purge (dedicated OS thread, independent of the writer) ────────

/// `ts_ms` threshold below which a row is a candidate for purging.
/// `retention_hours=0` is the documented signal for "no retention": it returns
/// `None`, and the caller must not delete anything (see the `[analytics]`
/// comment in `config/general.toml`).
fn retention_threshold_ms(retention_hours: u64, now_ms: i64) -> Option<i64> {
    if retention_hours == 0 {
        return None;
    }
    Some(now_ms - retention_hours as i64 * 3_600_000)
}

/// Deletes, in bounded batches (`DELETE ... WHERE id IN (SELECT ... LIMIT n)`,
/// a portable equivalent of `DELETE ... LIMIT n`), the rows with `ts_ms` below
/// `threshold_ms`, repeating until none are left. Never a single transaction over the
/// whole oldest part of the table: that would keep the WAL open too long and block the
/// writer. Returns the total number of deleted rows.
fn purge_older_than(
    conn: &Connection,
    threshold_ms: i64,
    batch_size: u32,
) -> rusqlite::Result<u64> {
    if batch_size == 0 {
        return Ok(0);
    }
    let mut total = 0u64;
    loop {
        let deleted = conn.execute(
            &format!(
                "DELETE FROM {TABLE} WHERE id IN (SELECT id FROM {TABLE} WHERE ts_ms < ?1 LIMIT ?2)"
            ),
            params![threshold_ms, batch_size],
        )? as u64;
        total += deleted;
        if deleted < batch_size as u64 {
            break; // last batch was partial (or empty): nothing else is below the threshold
        }
    }
    Ok(total)
}

/// One complete purge cycle: computes the current threshold from the present time
/// and deletes in batches. `retention_hours=0` is a no-op. Kept separate from
/// `run_purger` so the retention logic can be tested without threads or sleeps.
fn purge_cycle(conn: &Connection, retention_hours: u64, batch_size: u32) -> rusqlite::Result<u64> {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let Some(threshold_ms) = retention_threshold_ms(retention_hours, now_ms) else {
        return Ok(0);
    };
    purge_older_than(conn, threshold_ms, batch_size)
}

/// Dedicated OS thread for retention purging. It has its OWN cadence
/// (`purge_interval_secs`), fully separate from the writer thread
/// (`run_writer`) and from the 1s tick that records snapshots (`run`). It must never
/// compete with the recorder's 1-row-per-market-per-second writes through a massive `DELETE`.
/// It opens its OWN connection to the same file (WAL supports one writer at a
/// time; `busy_timeout` avoids `SQLITE_BUSY` when it coincides with an INSERT).
///
/// Space reclamation: a `DELETE` in SQLite leaves free pages inside the file and does
/// not return them to the filesystem. The chosen approach is `PRAGMA auto_vacuum=INCREMENTAL`
/// (set in `run_writer` when the DB is created) plus `PRAGMA incremental_vacuum` here after
/// each cycle. It returns free pages incrementally, without the long pause or the
/// temporary space (~the size of the DB) of a full `VACUUM`. Its limit: it ONLY applies to
/// a DB that was CREATED with `auto_vacuum=INCREMENTAL`. In an existing DB with data
/// (such as the 9.1 GiB one under `runtime/`), the mode PRAGMA is silently ignored and
/// `incremental_vacuum` is a no-op. Migrating that existing DB would require a separate
/// migration (`VACUUM INTO` a new file with the mode already set, which needs
/// ~9.1 GiB of temporary space). It is out of scope for this task and is not executed.
fn run_purger(
    db_path: String,
    retention_hours: u64,
    purge_interval_secs: u64,
    batch_size: u32,
    shutdown: watch::Receiver<bool>,
) {
    if retention_hours == 0 {
        info!(
            "analytics_recorder: retention_hours=0, retention purge disabled, nothing is deleted"
        );
        return;
    }
    let conn = match Connection::open(&db_path) {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "analytics_recorder: purger could not open the DB, thread exiting");
            return;
        }
    };
    if let Err(e) = conn.busy_timeout(Duration::from_secs(BUSY_TIMEOUT_SECS)) {
        warn!(error = %e, "analytics_recorder: could not set busy_timeout on the purge thread");
    }
    info!(
        retention_hours,
        purge_interval_secs, batch_size, "analytics_recorder: retention purge started"
    );

    let interval_secs = purge_interval_secs.max(1);
    loop {
        // Sleeps in 1s steps while checking shutdown, instead of one long sleep,
        // so the thread responds to shutdown without waiting for the whole
        // interval (which can be several minutes).
        for _ in 0..interval_secs {
            if *shutdown.borrow() {
                info!("analytics_recorder: retention purge stopped (shutdown)");
                return;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        match purge_cycle(&conn, retention_hours, batch_size) {
            Ok(0) => {}
            Ok(n) => info!(
                deleted_rows = n,
                retention_hours, "analytics_recorder: retention purge completed"
            ),
            Err(e) => error!(error = %e, "analytics_recorder: error during retention purge"),
        }
        // No-op if the DB was not created with auto_vacuum=INCREMENTAL (see the doc-comment above).
        if let Err(e) = conn.execute_batch("PRAGMA incremental_vacuum;") {
            warn!(error = %e, "analytics_recorder: incremental_vacuum failed");
        }
    }
}

// ── In-memory cache eviction (closed markets) ───────────────────────────────

/// Grace period after close (`clear_at`, in epoch seconds) before evicting the
/// cache entry of a market. Found during a code review: the recorder kept
/// `cache`/`last_hash` entries per market, but never purged them when a market closed.
/// but never purged them when a market closed. A closed market never receives books
/// again, so its entry stayed alive forever and memory grew without limit over the days.
/// 10 minutes of margin: a recently closed market can still receive a final
/// liquidation/settlement book before it stops emitting entirely.
const EVICTION_GRACE_SECS: i64 = 600;

/// Pure evictor over the two cache maps. Same criterion as
/// `purge_older_than`/`retention_threshold_ms`: the retention logic is kept in a
/// function that can be tested without starting the full async task.
fn evict_closed_markets(
    cache: &DashMap<String, Arc<BookSnapshot>>,
    last_hash: &DashMap<String, u64>,
    now_secs: i64,
) {
    cache.retain(|_, book| now_secs <= book.clear_at + EVICTION_GRACE_SECS);
    last_hash.retain(|market_condition_id, _| cache.contains_key(market_condition_id));
}

// ── Backpressure toward the writer (bounded channel) ────────────────────────

/// Size of the channel between the async tick and the SQLite writer (operational
/// bugfix). It used to be an `mpsc::unbounded_channel`: if the writer got stuck under
/// sustained contention (the same scenario behind `BUSY_TIMEOUT_SECS`), rows piled up in
/// memory WITHOUT A LIMIT. This is the same underlying pattern that already caused two
/// unbounded-growth incidents in this project (uncapped depth, duplicate rows without
/// dedup; see the doc-comments of `MAX_DEPTH_LEVELS` and `Row::content_hash`), this time
/// in RAM instead of on disk. 10,000 rows bound memory to a few tens of MB (each row is at
/// most `MAX_DEPTH_LEVELS` levels × 4 JSON lists, on the order of 1-2 KB). That is enough to
/// absorb a writer stall of several minutes without the channel filling up during normal
/// operation.
const DB_CHANNEL_CAPACITY: usize = 10_000;

/// Maximum frequency of the "channel full, rows dropped" warning. Under sustained
/// contention the channel can stay full on every tick (1/s). Without a limit, that
/// `warn!` would flood the log at the same rate. The dropped-row count is accumulated
/// between warnings and emitted in a single `warn!` per interval.
const DROPPED_LOG_INTERVAL: Duration = Duration::from_secs(30);

/// Queues `row` for the writer without blocking the tick. Returns `true` if the
/// caller should keep considering the writer alive, and `false` if the channel
/// is closed (writer down) and persisting should stop.
///
/// A full channel (`TrySendError::Full`) is NOT treated as a writer failure:
/// the row is dropped and processing continues. The alternative (waiting for room with
/// `send().await`) would block this same async loop, which also consumes the book
/// broadcast. Losing one analytics snapshot is acceptable (it is backtesting input, not a
/// money path); losing the bot's tick cadence is not. Extracted from `run` so the
/// backpressure policy can be tested without starting the full async task (same criterion
/// as `purge_cycle`/`evict_closed_markets`).
/// Outcome of trying to queue a row to the SQLite writer: it distinguishes
/// "actually sent" (`Sent`) from "dropped, but the writer is still alive" (`Dropped`).
/// The caller NEEDS that distinction to know whether it may record the row as written in
/// `last_hash` (see `send_tick_rows`).
enum SendOutcome {
    Sent,
    Dropped,
    WriterDown,
}

fn try_send_row(
    tx: &mpsc::Sender<Row>,
    row: Row,
    dropped_rows: &mut u64,
    last_dropped_log: &mut Instant,
) -> SendOutcome {
    match tx.try_send(row) {
        Ok(()) => SendOutcome::Sent,
        Err(mpsc::error::TrySendError::Full(_)) => {
            *dropped_rows += 1;
            if last_dropped_log.elapsed() >= DROPPED_LOG_INTERVAL {
                warn!(
                    dropped_rows = *dropped_rows,
                    "analytics_recorder: channel to the SQLite writer is full, analytics rows dropped (backpressure)"
                );
                *dropped_rows = 0;
                *last_dropped_log = Instant::now();
            }
            SendOutcome::Dropped
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            warn!("analytics_recorder: SQLite writer is down, persisting stopped (task stays alive)");
            SendOutcome::WriterDown
        }
    }
}

/// Deduplicates by hash and sends the rows of ONE tick to the SQLite writer.
///
/// Bugfix (backpressure of the bounded channel): `last_hash` is only updated after a
/// SUCCESSFUL send (`SendOutcome::Sent`), never before attempting the `send` nor on
/// `Dropped`. Setting it earlier broke the exact scenario that motivated hash dedup: a
/// market with a STABLE book. If the channel filled up for an instant and the row was
/// dropped, but `last_hash` already held its hash, the dedup `continue` would skip that row
/// on EVERY following tick while the book did not change (the hash is the same, so nothing
/// would retrigger it). The whole market would be lost until the next book change, not just
/// one snapshot. With this version, a dropped row is retried on the next tick with the same
/// hash until a send succeeds.
///
/// Returns `false` if the writer died (channel closed). The caller must stop trying
/// to persist in this cycle.
fn send_tick_rows(
    rows: Vec<Row>,
    last_hash: &DashMap<String, u64>,
    db_tx: &mpsc::Sender<Row>,
    dropped_rows: &mut u64,
    last_dropped_log: &mut Instant,
) -> bool {
    for row in rows {
        let hash = row.content_hash();
        if last_hash
            .get(&row.market_condition_id)
            .is_some_and(|h| *h == hash)
        {
            continue; // book identical to the last recorded one: nothing new to persist
        }
        let market_condition_id = row.market_condition_id.clone();
        match try_send_row(db_tx, row, dropped_rows, last_dropped_log) {
            SendOutcome::Sent => {
                last_hash.insert(market_condition_id, hash);
            }
            SendOutcome::Dropped => {} // retried on the next tick, same hash
            SendOutcome::WriterDown => return false,
        }
    }
    true
}

// ── Async task (consumer of the book broadcast) ──────────────────────────

/// Analytics recorder: subscribes to the existing book broadcast, keeps an in-memory cache
/// of the last book of each market, and every 1s dumps that complete cache to the analytics
/// DB through the SQLite writer.
///
/// `book_rx` must be one more `book_tx.subscribe()`, taken BEFORE `main.rs`
/// drops its own `book_tx` handle (`drop(book_tx)`).
#[allow(clippy::too_many_arguments)]
pub async fn run(
    mut book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    db_path: String,
    registry: Arc<MarketRegistry>,
    vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    mode: Mode,
    retention_hours: u64,
    purge_interval_secs: u64,
    purge_batch_size: u32,
    mut shutdown: watch::Receiver<bool>,
) {
    // `db_path` used to be taken as-is, without a mode suffix, so paper and live would
    // write to the SAME analytics file with the same config. This reuses
    // `StateStore::resolve_path` (the same logic that separates
    // `hrf_state_paper.sqlite`/`hrf_state_live.sqlite`) instead of duplicating the suffix logic.
    let db_path = StateStore::resolve_path(&db_path, mode)
        .to_str()
        .expect("market analytics path is not valid UTF-8")
        .to_string();

    info!(db_path = %db_path, "analytics_recorder: starting");

    let (db_tx, db_rx) = mpsc::channel::<Row>(DB_CHANNEL_CAPACITY);
    // The `JoinHandle` is kept (it used to be discarded) so that this function can join
    // it at the end. Without that, `run` returned as soon as this `select!` exited, leaving
    // the writer thread still writing/flushing. The `ToolSupervisor` treats the return of
    // this task as "recorder stopped" and could toggle on→off→on with two writers competing
    // for the same `db_path`.
    let writer_handle = {
        let writer_db_path = db_path.clone();
        match std::thread::Builder::new()
            .name("analytics_recorder_writer".to_string())
            .spawn(move || {
                if let Err(e) = run_writer(&writer_db_path, db_rx) {
                    error!(error = %e, "analytics_recorder: fatal error in the SQLite writer");
                }
            }) {
            Ok(h) => h,
            Err(e) => {
                // The thread could not even be spawned. Log it and end the task CLEANLY
                // (normal return, never panic): it must never take down the main process.
                error!(error = %e, "analytics_recorder: could not spawn the writer thread, task exiting");
                return;
            }
        }
    };

    // Retention purge: independent OS thread, its own connection, its own
    // cadence (see the doc-comment of `run_purger`). It never shares the writer loop
    // or the 1s tick below. The `JoinHandle` is kept for the same reason as above:
    // to join it before returning.
    let purger_handle = {
        let purge_db_path = db_path.clone();
        let purge_shutdown = shutdown.clone();
        match std::thread::Builder::new()
            .name("analytics_recorder_purger".to_string())
            .spawn(move || {
                run_purger(
                    purge_db_path,
                    retention_hours,
                    purge_interval_secs,
                    purge_batch_size,
                    purge_shutdown,
                )
            }) {
            Ok(h) => Some(h),
            Err(e) => {
                error!(error = %e, "analytics_recorder: could not spawn the retention purge thread");
                None
            }
        }
    };

    let cache: DashMap<String, Arc<BookSnapshot>> = DashMap::new();
    let last_hash: DashMap<String, u64> = DashMap::new();
    // Universes already reported, so the same `warn!` is not repeated on every tick. A market
    // whose universe does not match the `crypto-updown*` filter is still discarded as before
    // (what gets recorded does NOT change; that is a product decision), but now the FIRST time
    // such a universe appears is logged instead of disappearing silently. Before this, a
    // a "g-rw1" universe discarded 312 brackets with no error, making its tertiary
    // metric impossible to compute.
    let mut warned_universes: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    // Once the writer is confirmed dead (channel closed), stop trying to persist,
    // but the async task stays alive until shutdown.
    let mut writer_alive = true;
    // Backpressure of the bounded channel (see `try_send_row`): count of dropped
    // rows and the timestamp of the last warning, to limit the log frequency to
    // `DROPPED_LOG_INTERVAL`.
    let mut dropped_rows: u64 = 0;
    let mut last_dropped_log = Instant::now();

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("analytics_recorder: shutdown received");
                    break;
                }
            }
            recv = book_rx.recv() => match recv {
                Ok(book) => {
                    // Operational bugfix: `book_tx` carries ALL the markets that `market_feed`
                    // discovers (crypto-updown, but also crypto-multistrike, rewards-longdated, etc.).
                    // Without filtering, this task recorded universes outside the scope
                    // documented above ("every crypto up/down market"), inflating the number
                    // of tracked markets (1,298 distinct ones measured in ~5h) far beyond what the
                    // bot actually trades. A market without `UniverseInfo` (a discovery/eviction race)
                    // is conservatively discarded, as `strategy::mod::run` already does.
                    let universe = registry.universe_info
                        .get(&book.market_condition_id)
                        .map(|info| info.universe.clone());
                    let is_crypto_updown = universe.as_deref().is_some_and(|u| u.starts_with("crypto-updown"));
                    if !is_crypto_updown {
                        // Warns the first time THIS universe is seen (dedup by HashSet).
                        // It never changes what gets recorded, it only makes the drop visible.
                        if let Some(u) = universe {
                            if warned_universes.insert(u.clone()) {
                                warn!(
                                    universe = %u,
                                    market_condition_id = %book.market_condition_id,
                                    slug = %book.slug,
                                    "analytics_recorder: universe not matched by the crypto-updown* filter, market discarded (check whether it should be in scope; the criterion does not change by itself)"
                                );
                            }
                        }
                        continue;
                    }
                    // A panic here must NEVER propagate out of the task. It is isolated
                    // like any other internal failure.
                    if std::panic::catch_unwind(AssertUnwindSafe(|| {
                        cache.insert(book.market_condition_id.clone(), book);
                    })).is_err() {
                        error!("analytics_recorder: panic isolated while updating the cache, book discarded, task stays alive");
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    // RECOVERABLE (same criterion as sim_pump/strategy::run): the consumer fell
                    // behind the broadcast buffer. It is not a channel close, so it is logged and processing continues.
                    warn!(n, "analytics_recorder: books lost due to broadcast lag, continuing");
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("analytics_recorder: book channel closed, finishing");
                    break;
                }
            },
            _ = ticker.tick() => {
                let ts_ms = chrono::Utc::now().timestamp_millis();
                // Evicts long-closed markets. It runs on EVERY tick, even if the SQLite
                // writer has already died, so the cache does not grow without bound either way.
                evict_closed_markets(&cache, &last_hash, ts_ms / 1000);
                if !writer_alive {
                    continue;
                }
                let built = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    let mut rows = Vec::with_capacity(cache.len());
                    for entry in cache.iter() {
                        rows.push(build_row(entry.value().as_ref(), ts_ms, &vol_rxs));
                    }
                    rows
                }));
                match built {
                    Ok(rows) => {
                        if !send_tick_rows(rows, &last_hash, &db_tx, &mut dropped_rows, &mut last_dropped_log) {
                            writer_alive = false;
                        }
                    }
                    Err(_) => {
                        error!("analytics_recorder: panic isolated while building the tick's rows, cycle discarded, task stays alive");
                    }
                }
            }
        }
    }

    // A burst of drops that does not reach `DROPPED_LOG_INTERVAL` before shutdown would
    // be lost silently: it was neither logged on drain nor on close. A short burst (<30s)
    // stayed invisible, or resurfaced hours later added to the counter of a later episode,
    // suggesting a false recency. The residual is logged here once, at close.
    if dropped_rows > 0 {
        warn!(
            dropped_rows,
            "analytics_recorder: rows dropped by backpressure still pending a log at close"
        );
    }
    drop(db_tx); // closes the channel, so the writer ends its loop and returns

    // Join the writer/purger threads before returning. See the comment at the
    // `writer_handle` spawn above.
    match tokio::task::spawn_blocking(move || writer_handle.join()).await {
        Ok(Ok(())) => {}
        Ok(Err(_)) => error!("analytics_recorder: the writer thread panicked"),
        Err(e) => error!(error = %e, "analytics_recorder: joining the writer thread failed"),
    }
    if let Some(h) = purger_handle {
        match tokio::task::spawn_blocking(move || h.join()).await {
            Ok(Ok(())) => {}
            Ok(Err(_)) => error!("analytics_recorder: the purge thread panicked"),
            Err(e) => error!(error = %e, "analytics_recorder: joining the purge thread failed"),
        }
    }
    info!("analytics_recorder: stopped");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_feed::registry::UniverseInfo;
    use crate::types::{HalfBook, TokenBook};
    use rust_decimal_macros::dec;

    // ── best_bid_ask_mid ─────────────────────────────────────────────────────

    fn lvl(price: Decimal, size: Decimal) -> Level {
        Level { price, size }
    }

    /// Regression of the same kind as `src/accounting/mark_price.rs::mid_from_raw_sorts_...`:
    /// the levels arrive UNSORTED (worst first), and the function must still find
    /// the best bid/ask by sorting defensively before taking `.first()`.
    #[test]
    fn best_bid_ask_mid_defensively_sorts_unordered_levels() {
        let bids = vec![
            lvl(dec!(0.10), dec!(5)),
            lvl(dec!(0.55), dec!(3)), // best bid, out of order
            lvl(dec!(0.30), dec!(1)),
        ];
        let asks = vec![
            lvl(dec!(0.90), dec!(2)),
            lvl(dec!(0.60), dec!(4)), // best ask, out of order
        ];
        let (bb, ba, mid) = best_bid_ask_mid(&bids, &asks);
        assert_eq!(bb, Some(dec!(0.55)));
        assert_eq!(ba, Some(dec!(0.60)));
        assert_eq!(mid, Some(dec!(0.575)));
    }

    #[test]
    fn best_bid_ask_mid_single_side_uses_that_side() {
        let bids = vec![lvl(dec!(0.40), dec!(1))];
        let (bb, ba, mid) = best_bid_ask_mid(&bids, &[]);
        assert_eq!(bb, Some(dec!(0.40)));
        assert_eq!(ba, None);
        assert_eq!(mid, Some(dec!(0.40)));

        let asks = vec![lvl(dec!(0.44), dec!(1))];
        let (bb2, ba2, mid2) = best_bid_ask_mid(&[], &asks);
        assert_eq!(bb2, None);
        assert_eq!(ba2, Some(dec!(0.44)));
        assert_eq!(mid2, Some(dec!(0.44)));
    }

    #[test]
    fn best_bid_ask_mid_both_empty_is_none() {
        assert_eq!(best_bid_ask_mid(&[], &[]), (None, None, None));
    }

    // ── build_row: spot cold-start ──────────────────────────────────────────

    /// Bugfix: `vol_rxs` always has a pre-seeded entry per asset
    /// (`VolSnapshot::default()`, `last_price=0`), even before the oracle's first
    /// real tick. `build_row` must tell that cold start apart from a real price,
    /// instead of persisting `Some(Decimal::ZERO)` (see the doc-comment of `Row::spot`).
    #[test]
    fn build_row_spot_is_none_on_cold_start_but_some_with_real_price() {
        let book = make_book("cid-spot", 300);

        let (_tx_cold, rx_cold) = watch::channel(VolSnapshot::default());
        let mut vol_rxs_cold = HashMap::new();
        vol_rxs_cold.insert("btc".to_string(), rx_cold);
        let row_cold = build_row(&book, 0, &vol_rxs_cold);
        assert_eq!(
            row_cold.spot, None,
            "cold start (VolSnapshot::default(), last_price=0) must give spot=None, not Some(0)"
        );

        let (_tx_live, rx_live) = watch::channel(VolSnapshot {
            last_price: dec!(101_234),
            ..VolSnapshot::default()
        });
        let mut vol_rxs_live = HashMap::new();
        vol_rxs_live.insert("btc".to_string(), rx_live);
        let row_live = build_row(&book, 0, &vol_rxs_live);
        assert_eq!(row_live.spot, Some(dec!(101_234)));
    }

    // ── content_hash ─────────────────────────────────────────────────────────

    fn make_row(spot: Option<Decimal>) -> Row {
        Row {
            ts_ms: 0,
            market_condition_id: "cid".to_string(),
            slug: "btc-updown-5m-cid".to_string(),
            clear_at: 0,
            market_open_ts: 0,
            neg_risk: false,
            reference_price: Some(dec!(100000)),
            fee_rate_bps: Some(200),
            parabolic_fee_rate: Some(dec!(0.05)),
            tick_size: Some(dec!(0.01)),
            yes_token_id: "cid-yes".to_string(),
            yes_best_bid: Some(dec!(0.60)),
            yes_best_ask: Some(dec!(0.62)),
            yes_mid: Some(dec!(0.61)),
            yes_bids_json: "[]".to_string(),
            yes_asks_json: "[]".to_string(),
            no_token_id: "cid-no".to_string(),
            no_best_bid: Some(dec!(0.38)),
            no_best_ask: Some(dec!(0.40)),
            no_mid: Some(dec!(0.39)),
            no_bids_json: "[]".to_string(),
            no_asks_json: "[]".to_string(),
            spot,
        }
    }

    /// `spot` is deliberately OUTSIDE `content_hash` (see its doc-comment):
    /// two rows identical except for `spot` must produce the SAME hash, or
    /// hash deduplication would almost never work.
    #[test]
    fn content_hash_ignores_spot() {
        let a = make_row(Some(dec!(100050)));
        let b = make_row(Some(dec!(99980)));
        let c = make_row(None);
        assert_eq!(a.content_hash(), b.content_hash());
        assert_eq!(a.content_hash(), c.content_hash());
    }

    // ── run (integration tests) ─────────────────────────────────────────────────

    fn make_book(cid: &str, clear_delta_secs: i64) -> Arc<BookSnapshot> {
        let side = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        let now = chrono::Utc::now().timestamp();
        Arc::new(BookSnapshot {
            market_condition_id: cid.to_string(),
            yes: TokenBook {
                token_id: format!("{cid}-yes"),
                bids: side("0.60"),
                asks: side("0.62"),
                ..Default::default()
            },
            no: TokenBook {
                token_id: format!("{cid}-no"),
                bids: side("0.38"),
                asks: side("0.40"),
                ..Default::default()
            },
            market_open_ts: now - 300,
            clear_at: now + clear_delta_secs,
            neg_risk: false,
            slug: format!("btc-updown-5m-{cid}"),
            reference_price: Some(dec!(100000)),
            fee_rate_bps: Some(200),
            parabolic_fee_rate: Some(dec!(0.05)),
            tick_size: Some(dec!(0.01)),
            recv_instant: None,
        })
    }

    /// `Lagged` is RECOVERABLE, like `sim_pump`/`strategy::run`: the task
    /// must log and stay alive, never terminate. Verified end-to-end: a lag is
    /// forced with a minimal-capacity channel, and then it is checked
    /// that the task keeps recording to the DB after the lag.
    #[tokio::test]
    async fn lagged_does_not_end_the_task_and_keeps_recording() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("analytics_test.sqlite");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let (book_tx, book_rx) = broadcast::channel::<Arc<BookSnapshot>>(2);
        let (_kill_tx, kill_rx) = watch::channel(false);

        let registry = Arc::new(MarketRegistry::new());
        for i in 0..20 {
            registry.universe_info.insert(
                format!("cid-{i}"),
                UniverseInfo {
                    universe: "crypto-updown-long".into(),
                    has_oracle: true,
                },
            );
        }
        registry.universe_info.insert(
            "cid-final".into(),
            UniverseInfo {
                universe: "crypto-updown-long".into(),
                has_oracle: true,
            },
        );

        tokio::spawn(run(
            book_rx,
            db_path_str.clone(),
            registry,
            HashMap::new(),
            Mode::Paper,
            0,
            600,
            5000,
            kill_rx,
        ));

        // Capacity-2 channel: sending many more without giving the
        // receiver time to drain it forces `RecvError::Lagged` on the next recv.
        for i in 0..20 {
            let _ = book_tx.send(make_book(&format!("cid-{i}"), 120));
        }

        // After the lag, a valid book must still be cached and persisted
        // on the next 1s tick.
        let _ = book_tx.send(make_book("cid-final", 120));
        tokio::time::sleep(Duration::from_millis(1300)).await;

        // `run` applies the mode suffix: opening the
        // resolved file, not the raw path passed to `run`.
        let resolved = StateStore::resolve_path(&db_path_str, Mode::Paper);
        let conn = Connection::open(&resolved).unwrap();
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |r| r.get(0))
            .unwrap();
        assert!(
            count > 0,
            "the task must keep recording to the DB after a Lagged error"
        );
    }

    #[tokio::test]
    async fn persists_full_book_with_derived_columns() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("analytics_test2.sqlite");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let (book_tx, book_rx) = broadcast::channel::<Arc<BookSnapshot>>(8);
        let (_kill_tx, kill_rx) = watch::channel(false);

        let registry = Arc::new(MarketRegistry::new());
        registry.universe_info.insert(
            "cid-x".into(),
            UniverseInfo {
                universe: "crypto-updown-long".into(),
                has_oracle: true,
            },
        );

        tokio::spawn(run(
            book_rx,
            db_path_str.clone(),
            registry,
            HashMap::new(),
            Mode::Paper,
            0,
            600,
            5000,
            kill_rx,
        ));

        book_tx.send(make_book("cid-x", 300)).unwrap();
        tokio::time::sleep(Duration::from_millis(1300)).await;

        let resolved = StateStore::resolve_path(&db_path_str, Mode::Paper);
        let conn = Connection::open(&resolved).unwrap();
        let (yes_bb, yes_ba, slug): (String, String, String) = conn
            .query_row(
                &format!("SELECT yes_best_bid, yes_best_ask, slug FROM {TABLE} WHERE market_condition_id = 'cid-x'"),
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(yes_bb, "0.60");
        assert_eq!(yes_ba, "0.62");
        assert_eq!(slug, "btc-updown-5m-cid-x");
    }

    /// Operational bugfix: without deduplication, each 1s tick
    /// dumps the cached book even if it has not changed: ~65 GB/day measured
    /// in production. A book with no changes between two ticks must not generate a
    /// row for the unchanged book.
    #[tokio::test]
    async fn does_not_duplicate_rows_when_book_does_not_change_between_ticks() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("analytics_test3.sqlite");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let (book_tx, book_rx) = broadcast::channel::<Arc<BookSnapshot>>(8);
        let (_kill_tx, kill_rx) = watch::channel(false);

        let registry = Arc::new(MarketRegistry::new());
        registry.universe_info.insert(
            "cid-dup".into(),
            UniverseInfo {
                universe: "crypto-updown-long".into(),
                has_oracle: true,
            },
        );

        tokio::spawn(run(
            book_rx,
            db_path_str.clone(),
            registry,
            HashMap::new(),
            Mode::Paper,
            0,
            600,
            5000,
            kill_rx,
        ));

        book_tx.send(make_book("cid-dup", 300)).unwrap();
        // Two 1s ticks without any new book: the cache still holds the
        // same book, so only the first row should be persisted.
        tokio::time::sleep(Duration::from_millis(2300)).await;

        let resolved = StateStore::resolve_path(&db_path_str, Mode::Paper);
        let conn = Connection::open(&resolved).unwrap();
        let count: i64 = conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {TABLE} WHERE market_condition_id = 'cid-dup'"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            count, 1,
            "an unchanged book must not generate a new row per tick"
        );
    }

    /// Path isolation: before this fix, `run` used
    /// `db_path` as-is, so paper and live with the SAME config
    /// would write to the same file. With the fix, each mode resolves to a
    /// different file (same logic as `StateStore::resolve_path`) and the
    /// raw file without a suffix is never touched.
    #[tokio::test]
    async fn paper_and_live_write_to_different_analytics_files() {
        let dir = tempfile::tempdir().unwrap();
        let raw_path = dir.path().join("hrf_market_analytics.sqlite");
        let raw_str = raw_path.to_str().unwrap().to_string();

        let (book_tx_paper, book_rx_paper) = broadcast::channel::<Arc<BookSnapshot>>(4);
        let (_kill_tx_paper, kill_rx_paper) = watch::channel(false);
        let registry_paper = Arc::new(MarketRegistry::new());
        registry_paper.universe_info.insert(
            "cid-paper".into(),
            UniverseInfo {
                universe: "crypto-updown-long".into(),
                has_oracle: true,
            },
        );
        tokio::spawn(run(
            book_rx_paper,
            raw_str.clone(),
            registry_paper,
            HashMap::new(),
            Mode::Paper,
            0,
            600,
            5000,
            kill_rx_paper,
        ));
        book_tx_paper.send(make_book("cid-paper", 300)).unwrap();

        let (book_tx_live, book_rx_live) = broadcast::channel::<Arc<BookSnapshot>>(4);
        let (_kill_tx_live, kill_rx_live) = watch::channel(false);
        let registry_live = Arc::new(MarketRegistry::new());
        registry_live.universe_info.insert(
            "cid-live".into(),
            UniverseInfo {
                universe: "crypto-updown-long".into(),
                has_oracle: true,
            },
        );
        tokio::spawn(run(
            book_rx_live,
            raw_str.clone(),
            registry_live,
            HashMap::new(),
            Mode::Live,
            0,
            600,
            5000,
            kill_rx_live,
        ));
        book_tx_live.send(make_book("cid-live", 300)).unwrap();

        tokio::time::sleep(Duration::from_millis(1300)).await;

        let paper_path = StateStore::resolve_path(&raw_str, Mode::Paper);
        let live_path = StateStore::resolve_path(&raw_str, Mode::Live);
        assert_ne!(
            paper_path, live_path,
            "paper and live must resolve to distinct files"
        );
        assert!(
            paper_path.exists(),
            "missing paper analytics file"
        );
        assert!(live_path.exists(), "missing live analytics file");
        assert!(
            !raw_path.exists(),
            "must not write to the raw path without a suffix, a sign that paper/live share a file"
        );
    }

    // ── retention: retention_threshold_ms / purge_cycle / purge_older_than ──────

    #[test]
    fn retention_threshold_ms_zero_means_no_retention() {
        assert_eq!(retention_threshold_ms(0, 1_000_000), None);
    }

    #[test]
    fn retention_threshold_ms_computes_threshold_in_hours() {
        let now_ms = 10 * 3_600_000; // hour 10
                                     // 3h retention: the threshold lands at hour 7
        assert_eq!(retention_threshold_ms(3, now_ms), Some(7 * 3_600_000));
    }

    /// Same `market_condition_id` for all rows: only `ts_ms` changes, which is
    /// the only thing the purge cares about.
    fn row_with_ts(ts_ms: i64) -> Row {
        let mut r = make_row(None);
        r.ts_ms = ts_ms;
        r
    }

    /// TDD required: "with retention configured, rows older
    /// than the threshold are deleted and recent ones are not".
    #[test]
    fn purge_older_than_deletes_old_rows_and_keeps_recent() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("purge_test.sqlite")).unwrap();
        ensure_table(&conn).unwrap();

        insert_row(&conn, &row_with_ts(1_000)).unwrap(); // oldest, deleted
        insert_row(&conn, &row_with_ts(2_000)).unwrap(); // oldest, deleted
        insert_row(&conn, &row_with_ts(10_000)).unwrap(); // recent, kept

        let deleted = purge_older_than(&conn, 5_000, 100).unwrap();
        assert_eq!(deleted, 2);

        let remaining: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, 1);
        let remaining_ts: i64 = conn
            .query_row(&format!("SELECT ts_ms FROM {TABLE}"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            remaining_ts, 10_000,
            "only the recent row should survive"
        );
    }

    /// TDD required: "batched purging terminates and leaves no rows
    /// above the threshold after several passes": 25 old rows with batches of
    /// 5 forces several inner iterations of `purge_older_than`.
    #[test]
    fn purge_older_than_in_batches_leaves_no_rows_above_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("purge_batches.sqlite")).unwrap();
        ensure_table(&conn).unwrap();

        for i in 0..25 {
            insert_row(&conn, &row_with_ts(i)).unwrap();
        }
        for i in 0..3 {
            insert_row(&conn, &row_with_ts(100_000 + i)).unwrap();
        }

        let deleted = purge_older_than(&conn, 1_000, 5).unwrap();
        assert_eq!(
            deleted, 25,
            "must delete the 25 old rows in several passes of 5"
        );

        let below_threshold: i64 = conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {TABLE} WHERE ts_ms < 1000"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            below_threshold, 0,
            "no old row should remain below the threshold after several passes"
        );

        let total: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(total, 3, "the recent rows must survive intact");
    }

    /// `batch_size=0` guard: without it, the inner loop of
    /// `purge_older_than` would never end (0 rows deleted < batch_size=0
    /// is always false).
    #[test]
    fn purge_older_than_with_zero_batch_size_deletes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("purge_batch0.sqlite")).unwrap();
        ensure_table(&conn).unwrap();
        insert_row(&conn, &row_with_ts(1)).unwrap();

        let deleted = purge_older_than(&conn, 1_000_000, 0).unwrap();
        assert_eq!(deleted, 0);
    }

    /// TDD required: "with retention set to 0, nothing is deleted", verified at the
    /// level of `purge_cycle`, the function that `run_purger` actually calls
    /// on each cycle (not just `retention_threshold_ms` in isolation).
    #[test]
    fn purge_cycle_with_zero_retention_deletes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("purge_cycle_zero.sqlite")).unwrap();
        ensure_table(&conn).unwrap();
        insert_row(&conn, &row_with_ts(1)).unwrap(); // ts_ms=1, as old as possible

        let deleted = purge_cycle(&conn, 0, 100).unwrap();
        assert_eq!(deleted, 0);
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            count, 1,
            "retention_hours=0 must not delete anything, regardless of the age of the row"
        );
    }

    /// TDD required: "the purge does not run within the write path,
    /// verifiable by structure" — very old rows are inserted using
    /// ONLY `insert_row` (`purge_older_than`/`purge_cycle` is never called
    /// in this test) and they must remain there: the write path has no
    /// code that could trigger a delete.
    #[test]
    fn insert_row_never_triggers_a_purge() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("no_purge_on_write.sqlite")).unwrap();
        ensure_table(&conn).unwrap();

        for i in 0..5 {
            insert_row(&conn, &row_with_ts(i)).unwrap(); // ts_ms 0..4: very old
        }

        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            count, 5,
            "insert_row must not delete anything by itself, regardless of how old the row is"
        );
    }

    // ── backpressure: try_send_row (bounded channel) ──────────────────────

    /// TDD required: "full channel: the row is dropped and the caller
    /// stays alive" — it must not block or be treated as a dead writer.
    #[test]
    fn try_send_row_full_channel_drops_row_and_stays_alive() {
        let (tx, _rx) = mpsc::channel::<Row>(1);
        tx.try_send(make_row(None)).unwrap(); // fills the channel's only slot

        let mut dropped = 0u64;
        let mut last_log = Instant::now();
        let outcome = try_send_row(&tx, make_row(None), &mut dropped, &mut last_log);

        assert!(
            matches!(outcome, SendOutcome::Dropped),
            "full channel must not mark the writer as down"
        );
        assert_eq!(dropped, 1, "the discarded row must be counted");
    }

    /// TDD required: "closed channel (writer down): the caller must
    /// find out so it stops trying to persist" — same criterion as before
    /// bounding the channel, now through `TrySendError::Closed`.
    #[test]
    fn try_send_row_closed_channel_indicates_dead_writer() {
        let (tx, rx) = mpsc::channel::<Row>(1);
        drop(rx);

        let mut dropped = 0u64;
        let mut last_log = Instant::now();
        let outcome = try_send_row(&tx, make_row(None), &mut dropped, &mut last_log);

        assert!(
            matches!(outcome, SendOutcome::WriterDown),
            "closed channel must indicate that the writer is no longer alive"
        );
    }

    /// TDD required: the full-channel warning is rate-limited. After logging
    /// (interval elapsed), the drop counter is reset instead of growing
    /// without bound during a long contention.
    #[test]
    fn try_send_row_resets_counter_after_log_interval() {
        let (tx, _rx) = mpsc::channel::<Row>(1);
        tx.try_send(make_row(None)).unwrap(); // channel full for the whole test

        let mut dropped = 5u64; // simulates drops accumulated in a previous cycle
        let mut last_log = Instant::now().checked_sub(Duration::from_secs(31)).unwrap();
        let outcome = try_send_row(&tx, make_row(None), &mut dropped, &mut last_log);

        assert!(matches!(outcome, SendOutcome::Dropped));
        assert_eq!(
            dropped, 0,
            "after logging (interval elapsed), the counter must reset to 0"
        );
    }

    /// Complements the previous test: within the interval, the counter must
    /// keep accumulating instead of resetting on every drop.
    #[test]
    fn try_send_row_accumulates_counter_within_interval() {
        let (tx, _rx) = mpsc::channel::<Row>(1);
        tx.try_send(make_row(None)).unwrap();

        let mut dropped = 0u64;
        let mut last_log = Instant::now(); // just logged, within the interval
        try_send_row(&tx, make_row(None), &mut dropped, &mut last_log);
        try_send_row(&tx, make_row(None), &mut dropped, &mut last_log);

        assert_eq!(
            dropped, 2,
            "within the interval, the counter must keep accumulating"
        );
    }

    // ── send_tick_rows: the backpressure bug + hash dedup ────────────────────

    /// TDD required (new backpressure regression): marking `last_hash`
    /// BEFORE the send succeeds loses the ENTIRE market, not just one
    /// row. Reproduces the exact scenario: a quiet market (stable book, same
    /// same hash on two consecutive ticks) whose first attempt hits a full channel.
    /// Without the fix, `last_hash` would already have the hash set after tick 1, and
    /// tick 2 (same hash, channel already free) would skip it as a "duplicate",
    /// the row would never reach the writer even though the channel had been drained.
    #[test]
    fn send_tick_rows_retries_after_full_channel_instead_of_losing_the_market() {
        let (tx, mut rx) = mpsc::channel::<Row>(1);
        tx.try_send(make_row(None)).unwrap(); // fills the only slot: the next attempt sees "Full"

        let last_hash: DashMap<String, u64> = DashMap::new();
        let mut dropped = 0u64;
        let mut last_log = Instant::now();
        let mut row = make_row(None);
        row.market_condition_id = "cid-quiet".to_string();

        // Tick 1: channel full, so the row is dropped. It must NOT be marked as recorded.
        let alive = send_tick_rows(
            vec![row.clone()],
            &last_hash,
            &tx,
            &mut dropped,
            &mut last_log,
        );
        assert!(alive, "full channel must not indicate a dead writer");
        assert!(
            last_hash.get("cid-quiet").is_none(),
            "a row dropped due to a full channel must NOT be marked as already recorded"
        );

        rx.try_recv().unwrap(); // the writer drains the row that occupied the channel

        // Tick 2: SAME book (same hash). The market must not be orphaned
        // just because the previous tick hit the full channel.
        let alive2 = send_tick_rows(
            vec![row.clone()],
            &last_hash,
            &tx,
            &mut dropped,
            &mut last_log,
        );
        assert!(alive2);
        assert_eq!(
            last_hash.get("cid-quiet").map(|h| *h),
            Some(row.content_hash()),
            "after a truly successful send it must be marked as recorded"
        );
        let received = rx
            .try_recv()
            .expect("the row must have reached the writer on the retry");
        assert_eq!(received.market_condition_id, "cid-quiet");
    }

    /// Complements the previous test: if the send succeeds on the first attempt, the
    /// hash is marked immediately (normal behavior, no backpressure).
    #[test]
    fn send_tick_rows_marks_hash_after_successful_send_without_backpressure() {
        let (tx, mut rx) = mpsc::channel::<Row>(4);
        let last_hash: DashMap<String, u64> = DashMap::new();
        let mut dropped = 0u64;
        let mut last_log = Instant::now();
        let row = make_row(None);

        let alive = send_tick_rows(
            vec![row.clone()],
            &last_hash,
            &tx,
            &mut dropped,
            &mut last_log,
        );
        assert!(alive);
        assert_eq!(
            last_hash.get(&row.market_condition_id).map(|h| *h),
            Some(row.content_hash())
        );
        rx.try_recv().expect("the row must have been sent");
    }

    /// Closed channel (writer down): `send_tick_rows` must return `false`
    /// so the caller stops trying to persist in this cycle.
    #[test]
    fn send_tick_rows_reports_dead_writer_on_closed_channel() {
        let (tx, rx) = mpsc::channel::<Row>(1);
        drop(rx);
        let last_hash: DashMap<String, u64> = DashMap::new();
        let mut dropped = 0u64;
        let mut last_log = Instant::now();

        let alive = send_tick_rows(
            vec![make_row(None)],
            &last_hash,
            &tx,
            &mut dropped,
            &mut last_log,
        );
        assert!(!alive);
    }

    // ── in-memory cache eviction ─────────────────────────────────────────────────

    /// TDD required: "an entry from a closed market disappears from the
    /// cache".
    #[test]
    fn evict_closed_markets_removes_only_long_closed_market() {
        let cache: DashMap<String, Arc<BookSnapshot>> = DashMap::new();
        let last_hash: DashMap<String, u64> = DashMap::new();

        let closed = make_book("cid-closed", -3600); // closed 1h ago, beyond the margin
        let open = make_book("cid-open", 300); // closes in 5 min, still open

        cache.insert("cid-closed".to_string(), closed);
        cache.insert("cid-open".to_string(), open);
        last_hash.insert("cid-closed".to_string(), 111);
        last_hash.insert("cid-open".to_string(), 222);

        let now_secs = chrono::Utc::now().timestamp();
        evict_closed_markets(&cache, &last_hash, now_secs);

        assert!(
            !cache.contains_key("cid-closed"),
            "a market closed for longer than the margin must be evicted"
        );
        assert!(
            cache.contains_key("cid-open"),
            "a market still open must not be evicted"
        );
        assert!(
            !last_hash.contains_key("cid-closed"),
            "last_hash must be evicted together with the cache"
        );
        assert!(last_hash.contains_key("cid-open"));
    }

    // ── join of writer/purger before returning ────────────────────────

    /// Regression: previously, `run` returned as soon as the `select!` above
    /// exited, without joining the purge thread (nor the writer). The
    /// `ToolSupervisor` could toggle on→off→on with the purge thread still alive
    /// and writing to the same DB. `purge_interval_secs` is set high
    /// (5), and the shutdown signal arrives almost immediately (50ms): the purge
    /// thread starts, checks `shutdown.borrow()` (still `false`), and enters
    /// its first `sleep(1s)`. With the correct join, `run` CANNOT
    /// return before that sleep completes. Lower bound (not upper) on purpose:
    /// it is robust to a slow CI (wall-clock sleeps make tests flaky under load).
    /// Here only the minimum elapsed time matters,
    /// never that it finished quickly.
    #[tokio::test]
    async fn run_does_not_return_before_joining_purge_thread() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("analytics_join.sqlite");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let (_book_tx, book_rx) = broadcast::channel::<Arc<BookSnapshot>>(2);
        let (kill_tx, kill_rx) = watch::channel(false);
        let registry = Arc::new(MarketRegistry::new());

        let start = std::time::Instant::now();
        let handle = tokio::spawn(run(
            book_rx,
            db_path_str,
            registry,
            HashMap::new(),
            Mode::Paper,
            1, // retention_hours > 0 starts the purge thread
            5, // purge_interval_secs: the first check sleep is 1s anyway
            100,
            kill_rx,
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        kill_tx.send(true).unwrap();

        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("run() did not finish in time")
            .unwrap();

        assert!(
            start.elapsed() >= Duration::from_millis(900),
            "run() returned before joining the purge thread (elapsed={:?}) — \
             the join is not waiting for its shutdown-check sleep(1s)",
            start.elapsed()
        );
    }
}
