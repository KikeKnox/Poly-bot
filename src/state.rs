//! SQLite persistence (WAL) for the bot.
//!
//! ## Design notes
//!
//! - **Mode fixed at construction**: `StateStore::open(path, mode)` points all
//!   its queries at the tables of the chosen mode. No caller decides which table
//!   to use; switching mode means building another store.
//!
//! - **Money as INTEGER micro-USDC (×1_000_000)**: no TEXT→Decimal parsing on
//!   every read. The `Decimal ↔ i64` conversion happens in `to_micros`/`from_micros`.
//!
//! - **One schema per mode**: the tables have the same names in paper and live because
//!   they live in separate databases (`hrf_paper.sqlite` / `hrf_live.sqlite`), or
//!   in the same one with a mode suffix if `db_path` points to a single file.
//!
//! Convention: if `db_path` ends in `.sqlite`, the files
//! `{stem}_paper.sqlite` and `{stem}_live.sqlite` are created. This lets both modes be
//! opened at the same time without collisions, and removes the duplicated tables of the
//! earlier schema (`paper_positions` + `live_positions`).

use anyhow::Context;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, SqlitePool};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use crate::types::{DeadLetterFill, Fill, FillSide, Mode, Position, StrategyId};

// ── micro-USDC conversion helper ──────────────────────────────────────────

/// Converts `Decimal` to micro-USDC (i64), rounding to the nearest micro-unit.
/// Note used by `main.rs::paper_topup` when it records an automatic top-up. It is
/// shared so that `topups_total()` filters on the same literal without duplicating it
/// (and without the risk of the two drifting apart over time).
pub const TOPUP_NOTE: &str = "automatic paper top-up";

/// A point on the equity curve for the UI. Besides the gross equity it carries
/// the deposits accumulated up to that instant and the derived `pnl`. Without
/// this, each consumer had to subtract the top-ups itself, and the chart ended up
/// showing the +$1000 `paper_topup` steps
/// as if they were profit.
#[derive(Debug, Clone, PartialEq)]
pub struct EquityPoint {
    pub ts_ms: i64,
    pub cash: Decimal,
    pub positions_value: Decimal,
    pub equity: Decimal,
    /// Accumulated deposits (initial capital + manual + top-ups) at `ts_ms`.
    pub deposits: Decimal,
    /// `equity − deposits` = realized + unrealized PnL. Starts at 0.
    pub pnl: Decimal,
}

/// Edge and expectancy statistics of a strategy — output of
/// `StateStore::strategy_edge_stats` (see its docstring for the exact definition
/// of a "closed trade" and for the approximations it makes over the actual
/// `transactions` schema).
///
/// The `Option<Decimal>` fields are `None` when the data does not exist: `win_rate` and
/// `expectancy_per_trade` without closed trades, `profit_factor` without losses,
/// `expectancy_pct` with zero deployed cost. The handler serializes them as
/// `null`, never as a 0 that a reader would mistake for a real measurement (same
/// rule as `sigma_short`/`sigma_long` in `ui_market_health`).
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyEdgeStats {
    /// Closed round-trips per inventory (definition in `strategy_edge_stats`).
    pub closed_trades: i64,
    /// Round-trips still open at the end of the history: their PnL is not
    /// realized, so they are excluded from win_rate and expectancy.
    pub open_trades: i64,
    /// Closed trades with pnl > 0.
    pub wins: i64,
    /// Closed trades with pnl < 0.
    pub losses: i64,
    /// Closed trades with pnl == 0: they count in `closed_trades` and in the
    /// expectancy, but not in `wins` or `losses`.
    pub flat_trades: i64,
    /// `wins / closed_trades`.
    pub win_rate: Option<Decimal>,
    /// Mean pnl of the winning trades.
    pub avg_win: Option<Decimal>,
    /// Mean pnl of the losing trades (negative).
    pub avg_loss: Option<Decimal>,
    /// `gross_win / |gross_loss|`. `None` if there were no losses: infinity is not
    /// fabricated; the "all winners" scenario is read from `wins`/`closed_trades`.
    pub profit_factor: Option<Decimal>,
    /// `total_pnl / closed_trades`: expectancy per trade in USDC.
    pub expectancy_per_trade: Option<Decimal>,
    /// `total_pnl / total_cost_deployed`: return over the capital the
    /// strategy had to deploy in the closed trades.
    pub expectancy_pct: Option<Decimal>,
    /// Total realized PnL of the closed trades.
    pub total_pnl: Decimal,
    /// USDC bought within the closed trades (sum of the BUYs of each
    /// round-trip; aligns the numerator and denominator of `expectancy_pct`).
    pub total_cost_deployed: Decimal,
    /// Peak live cost basis observed by the strategy, rebuilt in one
    /// chronological pass over `transactions` (BUY sum `usdc_um`,
    /// SELL/REDEEM releases `usdc_um − pnl_um`, the cost allocated that the
    /// accounting actually removed: `AppliedFill::Sell::cost_released`).
    pub max_exposure: Decimal,
}

/// Prices and sizes outside the representable range are converted to 0 (no panic),
/// but the case is logged: this is the money write path, and a silent 0 can
/// hide a real loss of funds.
#[inline]
pub fn to_micros(d: Decimal) -> i64 {
    use rust_decimal::prelude::ToPrimitive;
    let scaled = (d * dec!(1_000_000)).round();
    scaled.to_i64().unwrap_or_else(|| {
        tracing::error!(value = %d, "to_micros: value not representable as i64, returning 0 (possible loss of amount)");
        0
    })
}

/// Converts micro-USDC (i64) to `Decimal`.
#[inline]
pub fn from_micros(i: i64) -> Decimal {
    Decimal::from(i) / dec!(1_000_000)
}

/// Outcome of an arb_15m5m basket (streak) for the `arb15m5m_streaks` table.
#[derive(Debug, Clone)]
pub struct Arb15m5mStreak {
    pub ts_ms: i64,
    pub asset: String,
    pub t1: i64,
    pub slug_15m: String,
    pub slug_5m: String,
    pub cid_15m: String,
    pub cid_5m: String,
    pub token_15m: String,
    pub token_5m: String,
    pub strike_15m: Decimal,
    pub strike_5m: Decimal,
    pub direction: String,
    pub q: Decimal,
    pub limit_15m: Decimal,
    pub limit_5m: Decimal,
    pub medium_cost: Decimal,
    pub expected_margin: Decimal,
    pub result: String,
    pub filled_15m: Decimal,
    pub filled_5m: Decimal,
    pub real_cost: Decimal,
    pub orphan_policy: String,
    pub hedge_id: Option<i64>,
    /// The paper filled was decided by the book (override 1.0), not by a draw.
    pub filled_per_book: bool,
    /// Shares offered at <= the limit price in each leg, in the book that
    /// was used for the decision (not the one at the simulation instant after latency).
    pub depth_15m: Decimal,
    pub depth_5m: Decimal,
}

/// RAII timer: records the query duration in a histogram when dropped.
struct MetricTimer {
    name: &'static str,
    op: &'static str,
    t0: std::time::Instant,
}

impl MetricTimer {
    fn new(name: &'static str, op: &'static str) -> Self {
        Self {
            name,
            op,
            t0: std::time::Instant::now(),
        }
    }
}

impl Drop for MetricTimer {
    fn drop(&mut self) {
        metrics::histogram!(self.name, "op" => self.op).record(self.t0.elapsed().as_secs_f64());
    }
}

/// Result of `apply_fill`: which mutation the transaction applied.
#[derive(Debug, PartialEq)]
pub enum AppliedFill {
    /// The fill already exists in `fills` (dedup by order_id): no mutations.
    Duplicate,
    /// Buy applied (position opened or averaged).
    Buy,
    /// Sell applied. `pnl` is the realized PnL of this sell and `cost_released` the
    /// cost that accounting ACTUALLY removed from the position: the same
    /// number the ledger must release. Passing `avg_price` and `size` instead made the
    /// caller recompute `avg_price × size`, which differs from the pro-rata split at
    /// micro precision and over-released exposure.
    Sell {
        pnl: Decimal,
        cost_released: Decimal,
    },
    /// Sell without a live position in the DB: fill recorded, position untouched.
    SellWithoutPosition,
}

/// Persisted row of `risk_state` (kill switch + session PnL of a scope:
/// `"global"` or a strategy's name). See
/// `StateStore::save_risk_state`/`load_risk_state` and `risk::RiskGuard::restore_persisted`.
#[derive(Debug, Clone)]
pub struct RiskStateRow {
    pub scope: String,
    pub killed: bool,
    pub reason: Option<String>,
    pub session_pnl: Decimal,
    pub loss_stop_baseline: Decimal,
}

/// Row of the durable journal of send attempts whose result is unknown. It is
/// persisted in SQLite BEFORE the POST (`status='pending'`), so that a crash or
/// a restart does not lose the record. A blind resend of the same attempt is
/// therefore never considered while the bracket has an `unresolved` entry, and the
/// reconciler can still see the attempt.
///
/// `status='resolved'` marks the attempt as closed after an explicit decision
/// (`mark_unknown_resolved`) or after a definitive acceptance or rejection by the
/// exchange.
///
/// `cost_um` is stored in micro-USDC (scale 1e6), the same domain as
/// `cost_basis_um` in `positions`. `attempt_id` is the PK and is generated in the
/// `Executor` before submit (see `Executor::next_attempt_id`).
#[derive(Debug, Clone)]
pub struct UnknownAttemptRow {
    pub attempt_id: String,
    pub cid: String,
    pub token_id: String,
    pub strategy: String,
    pub cost_um: i64,
    pub resting: bool,
    pub reason: String,
    pub ts_ms: i64,
    pub resolved: bool,
    pub resolved_ts_ms: Option<i64>,
    pub resolution: Option<String>,
    /// Declared price and size of the intent, used by the Unknown reconciler.
    /// NULL in old rows and in the quarantine rows of fills.
    pub price_um: Option<i64>,
    pub size_um: Option<i64>,
    /// Close of the intent's bracket (used for the claim horizon in the
    /// ledger); NULL in old and quarantine rows.
    pub clear_at: Option<i64>,
    /// Side declared by the intent (`"buy"`/`"sell"`), persisted in the same insert
    /// as the pending. NULL means unknown (legacy rows, `qfill-*` quarantine, cancel
    /// pendings); the reconciler must not match or compute claims from an inferred
    /// side.
    /// Unknown side is never inferred.
    pub side: Option<String>,
}

/// Raw row of `order_index` used as identity evidence.
/// `strategy` is returned WITHOUT parsing: a row whose value cannot be recognized
/// is NOT "absent" but indeterminate (the caller decides, fail-closed). Unlike
/// `order_index_lookup`, it exposes `cid`/`token_id` to validate the attribution
/// against the pending before use.
pub struct OrderIndexEvidence {
    pub order_id: String,
    pub local_order_id: Option<String>,
    /// Strategy as persisted; it may not parse.
    pub strategy: String,
    pub cid: String,
    pub token_id: String,
}

/// Raw row of `gtd_orders` (legacy owner).
/// Same rule: `strategy` is raw; "present but invalid" is not "absent".
pub struct GtdOrderRow {
    pub market_cid: String,
    /// Strategy as persisted; it may not parse.
    pub strategy: String,
}

/// Row of `markets_observed` awaiting an official label; used by the job of
/// `tools::recorders::resolution`. `market_open_ts`/`clear_at` are in SECONDS
/// (`BookSnapshot` domain) and are returned too, so the fetcher records the
/// resolution rule with the real market window instead of `None`; a `None` window
/// would leave the rule unbound.
#[derive(Debug, Clone)]
pub struct OfficialPendingMarket {
    pub condition_id: String,
    pub slug: String,
    pub yes_token_id: String,
    pub no_token_id: String,
    pub market_open_ts: Option<i64>,
    pub clear_at: i64,
}

/// Economic flow aggregates of the durable ledger (`transactions` +
/// `positions`), all in USDC at Decimal precision from micros.
///
/// This is the surface of the economic reconciliation: the wallet's cash (sim or real)
/// must move EXACTLY by `net_cash_flow()`, and the ledger's realized PnL must equal
/// `returns + open_cost`; see
/// `accounting::fill_sink::reconcile_economic_return`. `open_cost_drift()`
/// verifies that the accounting cost is conserved to the micro-unit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EconomicFlows {
    /// `Σ usdc` of the BUY transactions (cash that went out; includes fee and
    /// impact, because the ledger's `usdc` is `fill.price × size`, with the price
    /// all-inclusive).
    pub buys_usdc: Decimal,
    /// `Σ usdc` of the SELLs (proceeds received).
    pub sells_usdc: Decimal,
    /// `Σ pnl` realized on sells.
    pub sells_pnl: Decimal,
    /// `Σ usdc` of the REDEEMs (resolution payouts).
    pub redeem_usdc: Decimal,
    /// `Σ pnl` realized on redemptions.
    pub redeem_pnl: Decimal,
    /// `Σ usdc` of kinds other than BUY/SELL/REDEEM: visible, not merged.
    pub other_usdc: Decimal,
    pub n_buys: i64,
    pub n_sells: i64,
    pub n_redeem: i64,
    pub n_other: i64,
    /// `cost_basis` of the positions with `size > 0`: cost not yet realized.
    pub open_cost: Decimal,
}

impl EconomicFlows {
    /// Net cash flow of the ledger: `sells + redeem − buys`, what the
    /// wallet must have moved net (without external contributions or withdrawals).
    pub fn net_cash_flow(&self) -> Decimal {
        self.sells_usdc + self.redeem_usdc - self.buys_usdc
    }

    /// Total realized PnL of the ledger (sells + redemptions).
    pub fn realized_pnl(&self) -> Decimal {
        self.sells_pnl + self.redeem_pnl
    }

    /// Cost removed from the inventory by sells and redemptions:
    /// `Σ(usdc − pnl)` per kind. In a sell it is the pro-rated cost that left
    /// the position; in a redeem, the settled cost. In post-settlement sells
    /// (proceeds whose cost was already attributed by settlement) the removed amount is 0:
    /// `usdc == pnl`, which is correct.
    pub fn released_cost(&self) -> Decimal {
        (self.sells_usdc - self.sells_pnl) + (self.redeem_usdc - self.redeem_pnl)
    }

    /// Cost conservation to the micro-unit: `bought − removed − open`.
    /// 0 means the bought cost is fully accounted for as (removed when realized) +
    /// (still open); any deviation is cash or cost created or lost.
    pub fn open_cost_drift(&self) -> Decimal {
        self.buys_usdc - self.released_cost() - self.open_cost
    }
}

/// Values that `apply_settlement_detalle` settled with a fresh read
/// inside the transaction: the surface the caller uses to credit paper
/// cash (`SimulatedRouter::credit_redemption`)
/// with exactly what the ledger recorded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppliedSettlement {
    /// Shares settled (fresh size; the position is left at 0).
    pub size: Decimal,
    /// Cost removed from the position.
    pub cost_basis: Decimal,
    /// Settled payout: `size` if won, 0 if lost.
    pub payout: Decimal,
    /// Settled PnL: `payout − cost_basis`.
    pub pnl: Decimal,
}

// ── StateStore ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct StateStore {
    pool: SqlitePool,
    pub mode: Mode,
    /// In-memory cache of the blacklist: the executor used to query SQLite for
    /// every buy on its single task. It is loaded when the DB is opened and kept
    /// up to date in add_to_blacklist (the blacklist only grows; there is no delete endpoint).
    blacklist_cache: dashmap::DashSet<String>,
}

impl StateStore {
    /// Opens (or creates) the database for the given mode.
    ///
    /// The file is named `{stem}_{mode}.sqlite` so that paper and live do not
    /// share files even when `db_path` has the same prefix.
    pub async fn open(db_path: &str, mode: Mode) -> anyhow::Result<Arc<Self>> {
        let path = Self::resolve_path(db_path, mode);

        // `settlement::orphan::settle()` (UPDATE+INSERT+UPDATE transaction) failed with
        // "database is locked" under real MM load (115 times in one calibration run).
        // Ruled out by a direct test: busy_timeout was set on every pool connection
        // (sqlx applies the same `SqliteConnectOptions` to all 4; `PRAGMA busy_timeout`
        // returns 5000 on 4 simultaneous connections). The real cause is simpler:
        // SQLite allows only ONE writer at a time, even in WAL. With MM's write volume
        // (thousands per hour), the settle transaction sometimes waits more than 5s for
        // its turn, and busy_timeout really expires.
        // The position is not lost: outcome_won stays NULL and the settle is retried on the
        // next 60s poll. A larger timeout avoids that retry in the normal case;
        // orphan settlement is not latency-sensitive.
        // hence the 20s timeout set below.
        let pool = if path.to_str() == Some(":memory:") {
            let options = SqliteConnectOptions::from_str("sqlite::memory:")?
                .journal_mode(SqliteJournalMode::Wal)
                .synchronous(SqliteSynchronous::Normal)
                .busy_timeout(Duration::from_secs(20))
                .foreign_keys(true);
            SqlitePoolOptions::new()
                .max_connections(1)
                .connect_with(options)
                .await?
        } else {
            // Create the file if it does not exist
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            let url = format!("sqlite://{}?mode=rwc", path.display());
            let options = SqliteConnectOptions::from_str(&url)?
                .journal_mode(SqliteJournalMode::Wal)
                .synchronous(SqliteSynchronous::Normal)
                .busy_timeout(Duration::from_secs(20))
                .foreign_keys(true);
            SqlitePoolOptions::new()
                .max_connections(4)
                .connect_with(options)
                .await
                .with_context(|| format!("opening DB {}", path.display()))?
        };

        let store = Arc::new(Self {
            pool,
            mode,
            blacklist_cache: dashmap::DashSet::new(),
        });
        store.migrate().await?;

        // Populate the blacklist cache from the DB.
        for cid in store.blacklist().await? {
            store.blacklist_cache.insert(cid);
        }

        // Log whether the DB already holds data from previous runs (it is not a new DB)
        if path.to_str() != Some(":memory:") {
            let fills: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fills")
                .fetch_one(&store.pool)
                .await
                .unwrap_or(0);
            let positions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM positions")
                .fetch_one(&store.pool)
                .await
                .unwrap_or(0);
            tracing::info!(
                fills,
                positions,
                "state: DB opened, previous fills and positions loaded"
            );
        }

        Ok(store)
    }

    /// Opens an in-memory database (for tests).
    pub async fn open_memory(mode: Mode) -> anyhow::Result<Arc<Self>> {
        Self::open(":memory:", mode).await
    }

    /// Applies the mode suffix (`_paper`/`_live`) to a DB path.
    ///
    /// `pub` (not `pub(crate)`) because it is used both by
    /// `src/tools/recorders/analytics.rs` (inside the lib) and by `main.rs`
    /// (a binary crate, where `pub(crate)` is not enough) for the DB of
    /// `tools::recorders::spot_stream`: the three auxiliary DBs carry the
    /// same mode suffix so that paper and live never share a file.
    /// Keeping them apart prevents paper and live data from mixing.
    pub fn resolve_path(db_path: &str, mode: Mode) -> PathBuf {
        if db_path == ":memory:" {
            return PathBuf::from(":memory:");
        }
        let p = Path::new(db_path);
        let raw_stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("hrf_state");
        // Remove any existing mode suffix to avoid a double suffix
        // if db_path was already "hrf_state_paper.sqlite".
        let stem = raw_stem
            .strip_suffix("_paper")
            .or_else(|| raw_stem.strip_suffix("_live"))
            .unwrap_or(raw_stem);
        let dir = p.parent().unwrap_or(Path::new("."));
        dir.join(format!("{}_{}.sqlite", stem, mode))
    }

    // ── Migration ─────────────────────────────────────────────────────────────

    async fn migrate(&self) -> anyhow::Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS positions (
                token_id        TEXT NOT NULL,
                strategy        TEXT NOT NULL DEFAULT '',
                market_cid      TEXT NOT NULL,
                size_um         INTEGER NOT NULL DEFAULT 0,   -- micro-USDC ×1e6
                avg_price_um    INTEGER NOT NULL DEFAULT 0,
                cost_basis_um   INTEGER NOT NULL DEFAULT 0,
                opened_ts_ms    INTEGER NOT NULL DEFAULT 0,
                clear_at        INTEGER NOT NULL DEFAULT 0,
                neg_risk        INTEGER NOT NULL DEFAULT 0,   -- bool
                bail_sold       INTEGER NOT NULL DEFAULT 0,
                outcome_won     INTEGER,                      -- NULL=pending, 0=lost, 1=won_flag
                winner_label    TEXT,
                outcome_label   TEXT,
                redeemed        INTEGER NOT NULL DEFAULT 0,
                redeem_tx       TEXT,
                PRIMARY KEY (token_id, strategy)
            )",
        )
        .execute(&self.pool)
        .await?;

        // Migration: if the table existed with a simple PK (token_id only), recreate it with a composite PK.
        // The pragma read used to fall back with `.unwrap_or(2)`, which silently treated ANY
        // read failure as "already migrated". A real failure (permissions, one-off corruption)
        // would then skip the migration without a trace. The error is now logged, and only then
        // is "already migrated" assumed.
        let pk_cols: i64 = match sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('positions') WHERE pk > 0",
        )
        .fetch_one(&self.pool)
        .await
        {
            Ok(n) => n,
            Err(e) => {
                tracing::error!(error = %e, "state: failed reading pragma_table_info('positions'); \
                    assuming the schema is already migrated; check manually whether `positions` has a simple PK");
                2
            }
        };
        if pk_cols < 2 {
            tracing::info!("state: migrating positions to PRIMARY KEY (token_id, strategy)");
            sqlx::query("ALTER TABLE positions RENAME TO _positions_v1")
                .execute(&self.pool)
                .await?;
            sqlx::query(
                "CREATE TABLE positions (
                    token_id        TEXT NOT NULL,
                    strategy        TEXT NOT NULL DEFAULT '',
                    market_cid      TEXT NOT NULL,
                    size_um         INTEGER NOT NULL DEFAULT 0,
                    avg_price_um    INTEGER NOT NULL DEFAULT 0,
                    cost_basis_um   INTEGER NOT NULL DEFAULT 0,
                    opened_ts_ms    INTEGER NOT NULL DEFAULT 0,
                    clear_at        INTEGER NOT NULL DEFAULT 0,
                    neg_risk        INTEGER NOT NULL DEFAULT 0,
                    bail_sold       INTEGER NOT NULL DEFAULT 0,
                    outcome_won     INTEGER,
                    winner_label    TEXT,
                    outcome_label   TEXT,
                    redeemed        INTEGER NOT NULL DEFAULT 0,
                    redeem_tx       TEXT,
                    PRIMARY KEY (token_id, strategy)
                )",
            )
            .execute(&self.pool)
            .await?;
            // Explicit columns, by NAME on both sides (instead of `SELECT *`), so the
            // migration does not depend on the physical column order of `_positions_v1`
            // (the renamed old table), which may differ from this definition.
            // Explicit names keep the copy correct either way.
            sqlx::query(
                "INSERT OR IGNORE INTO positions
                 (token_id, strategy, market_cid, size_um, avg_price_um, cost_basis_um,
                  opened_ts_ms, clear_at, neg_risk, bail_sold, outcome_won, winner_label,
                  outcome_label, redeemed, redeem_tx)
                 SELECT token_id, strategy, market_cid, size_um, avg_price_um, cost_basis_um,
                        opened_ts_ms, clear_at, neg_risk, bail_sold, outcome_won, winner_label,
                        outcome_label, redeemed, redeem_tx
                 FROM _positions_v1",
            )
            .execute(&self.pool)
            .await?;
            sqlx::query("DROP TABLE _positions_v1")
                .execute(&self.pool)
                .await?;
            tracing::info!("state: migration of positions completed");
        }

        // Index for the market_cid filter of `has_open_position_in_market`, which runs for
        // each GTD during startup reconciliation (main.rs).
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_positions_market_cid ON positions(market_cid)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS fills (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms           INTEGER NOT NULL,
                market_cid      TEXT NOT NULL,
                token_id        TEXT NOT NULL,
                side            TEXT NOT NULL,   -- 'BUY' | 'SELL'
                price_um        INTEGER NOT NULL,
                size_um         INTEGER NOT NULL,
                order_id        TEXT,
                exec_id         TEXT,
                local_order_id  TEXT,
                hedge_id        INTEGER,
                neg_risk        INTEGER NOT NULL DEFAULT 0,
                strategy        TEXT NOT NULL DEFAULT 'arb'
            )",
        )
        .execute(&self.pool)
        .await?;

        // Execution identity: databases created BEFORE this change keep the old fills table,
        // in which the CREATE above is a no-op;
        // so the columns are added with ALTER, tolerating "duplicate column"
        // (same pattern as post_close_observations below).
        for col_def in ["exec_id TEXT", "local_order_id TEXT"] {
            let sql = format!("ALTER TABLE fills ADD COLUMN {col_def}");
            if let Err(e) = sqlx::query(&sql).execute(&self.pool).await {
                if !e.to_string().to_lowercase().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts_ms)")
            .execute(&self.pool)
            .await?;
        // Dedup is NO LONGER by `order_id`: a GTD/FAK order can produce SEVERAL partial
        // executions that share an order_id, and with the earlier UNIQUE index the second
        // partial fill was silently dropped. The dedup key is now `exec_id` (the trade id).
        // `order_id` stays a NON-unique index for the legacy fallback: old rows without
        // exec_id stored their id in order_id.
        // Their dedup follows the fallback in `record_fill_conn`, which is the same
        // for rows that have no exec_id.
        sqlx::query("DROP INDEX IF EXISTS idx_fills_order_id")
            .execute(&self.pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id)")
            .execute(&self.pool)
            .await?;
        // Avoids duplicates when the fill arrives both through the HTTP response and
        // through the user WS (or through backfill + WS). Partial: rows without exec_id (legacy
        // or internal fills) do not participate; their dedup follows the fallback in
        // `record_fill_conn`.
        sqlx::query(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_fills_exec_id \
             ON fills(exec_id) WHERE exec_id IS NOT NULL",
        )
        .execute(&self.pool)
        .await?;

        // Durable mapping `order_id → (local_order_id,
        // strategy, cid, token)` for EVERY order accepted (taker and resting).
        // `gtd_orders` only covers GTDs; `order_map` only lives in memory. Without
        // this table, a live fill after a restart has no resolvable owner and would fall
        // to the Arb catch-all. The executor writes it when it learns the `order_id` of the
        // submit; fill_sink reads it to attribute the strategy.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS order_index (
                order_id       TEXT PRIMARY KEY,
                local_order_id TEXT,
                strategy       TEXT NOT NULL,
                cid            TEXT NOT NULL,
                token_id       TEXT NOT NULL,
                ts_ms          INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_order_index_local ON order_index(local_order_id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS transactions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms           INTEGER NOT NULL,
                kind            TEXT NOT NULL,   -- 'BUY' | 'SELL' | 'REDEEM'
                market_cid      TEXT NOT NULL,
                token_id        TEXT NOT NULL,
                outcome_label   TEXT,
                winner_label    TEXT,
                shares_um       INTEGER NOT NULL,
                price_um        INTEGER NOT NULL,
                usdc_um         INTEGER NOT NULL,
                pnl_um          INTEGER,
                strategy        TEXT NOT NULL DEFAULT 'arb'
            )",
        )
        .execute(&self.pool)
        .await?;

        // Supports the ORDER BY ts_ms of pnl_series() over the whole history
        // (pagination of /ui/history uses the rowid `id` and does not need this index).
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_transactions_ts ON transactions(ts_ms)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS deposits (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                amount_um   INTEGER NOT NULL,
                note        TEXT NOT NULL DEFAULT '',
                ts_ms       INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS markets_observed (
                condition_id    TEXT PRIMARY KEY,
                slug            TEXT NOT NULL,
                yes_token_id    TEXT NOT NULL,
                no_token_id     TEXT NOT NULL,
                market_open_ts  INTEGER NOT NULL DEFAULT 0,
                clear_at        INTEGER NOT NULL,
                neg_risk        INTEGER NOT NULL DEFAULT 0,
                first_seen_ts   INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS blacklist (
                condition_id TEXT PRIMARY KEY
            )",
        )
        .execute(&self.pool)
        .await?;

        // Post-fill markout: book mid at +5s/+30s after the fill. This is the data the
        // MM/PM/ThetaCapture gates require (markout@5s ≥ 0; the gates were retired).
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS markouts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms           INTEGER NOT NULL,   -- fill timestamp
                market_cid      TEXT NOT NULL,
                token_id        TEXT NOT NULL,
                strategy        TEXT NOT NULL,
                side            TEXT NOT NULL,      -- 'BUY' | 'SELL'
                fill_price_um   INTEGER NOT NULL,
                mid_5s_um       INTEGER,            -- NULL if there was no book in time
                mid_30s_um      INTEGER
            )",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_markouts_strategy ON markouts(strategy)")
            .execute(&self.pool)
            .await?;

        // Scan observations: LadderArb/VolSurface are scan-only, and their gate
        // requires ≥1 week of data, which cannot be validated without persistence.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS scan_observations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms       INTEGER NOT NULL,
                strategy    TEXT NOT NULL,
                market_key  TEXT NOT NULL,   -- slug or pair of slugs of the opportunity
                kind        TEXT NOT NULL,   -- 'ladder_opp' | 'vol_divergence' | ...
                edge_um     INTEGER NOT NULL,-- edge (ladder) or ratio (VS) in micros
                payload     TEXT NOT NULL    -- JSON with the details
            )",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_scan_strategy_ts ON scan_observations(strategy, ts_ms)",
        )
        .execute(&self.pool)
        .await?;

        // Passive post-close observer: samples of the book after clear_at
        // with a deterministic verdict (oracle spot vs. reference_price), to
        // measure how much of the winning side can be traded before the market
        // corrects (input for a future "post-close sweep" strategy).
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS post_close_observations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms               INTEGER NOT NULL,
                secs_since_close    INTEGER NOT NULL,
                market_cid          TEXT NOT NULL,
                slug                TEXT NOT NULL,
                clear_at            INTEGER NOT NULL,
                reference_price_um  INTEGER,
                spot_um             INTEGER,
                spot_stale          INTEGER NOT NULL,
                rel_move_um         INTEGER,
                verdict             TEXT NOT NULL,   -- 'yes' | 'no' | 'indeterminate'
                yes_bid_um          INTEGER,
                yes_ask_um          INTEGER,
                yes_bid_sz_um       INTEGER,
                yes_ask_sz_um       INTEGER,
                no_bid_um           INTEGER,
                no_ask_um           INTEGER,
                no_bid_sz_um        INTEGER,
                no_ask_sz_um        INTEGER,
                winner_ask_um       INTEGER,         -- ask of the side of the verdict (NULL if indeterminate)
                winner_ask_sz_um    INTEGER,
                cl_spot_um          INTEGER,         -- Retired. Historical columns, always NULL in new rows
                cl_verdict          TEXT,            -- retired
                cl_rel_move_um      INTEGER          -- retired
            )",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_pcs_market_ts ON post_close_observations(market_cid, ts_ms)")
            .execute(&self.pool)
            .await?;

        // Retired columns (external feed removed) are added after the CREATE TABLE IF
        // NOT EXISTS above. In an existing database that CREATE is a no-op, so the 3
        // columns are added here with ALTER TABLE, tolerating the "duplicate column"
        // error in databases created AFTER this change (where CREATE TABLE already
        // includes them). They are kept for compatibility with the historical rows,
        // which were written before; new rows always write them as NULL.
        // Nothing in the current write path sets them.
        for col_def in [
            "cl_spot_um INTEGER",
            "cl_verdict TEXT",
            "cl_rel_move_um INTEGER",
        ] {
            let sql = format!("ALTER TABLE post_close_observations ADD COLUMN {col_def}");
            if let Err(e) = sqlx::query(&sql).execute(&self.pool).await {
                if !e.to_string().to_lowercase().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }

        // GTD strategy attribution, which survives restarts so that
        // startup reconciliation does not attribute all GTDs to Arb.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS gtd_orders (
                order_id   TEXT PRIMARY KEY,
                market_cid TEXT NOT NULL,
                strategy   TEXT NOT NULL,
                ts_ms      INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Kill switches and session PnL are persisted; without this,
        // `RiskGuard::new` always started clean (fail-open) on every restart,
        // "forgetting" a kill switch or a daily-loss stop per asset.
        // `scope` is "global" or the name (`StrategyId::to_string()`) of a
        // strategy: one row per scope, overwritten on each mutation.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS risk_state (
                scope               TEXT PRIMARY KEY,
                killed              INTEGER NOT NULL,
                reason              TEXT,
                session_pnl_um      INTEGER NOT NULL,
                loss_stop_baseline_um INTEGER NOT NULL,
                updated_ms          INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Orphan LIVE fills (market found neither in the asset registry nor in
        // `clear_at_cache`). Previously this was only an `error!` in the log, so real money
        // could be left with no persisted trace. `resolved` is set by hand after a future
        // manual reconciliation (there is no automatic reinjection yet).
        // Orphan fills are recorded here so that they stay visible.
        // Equity time series (cash + mark-to-market value of open positions)
        // — unlike `pnl_series` (realized PnL only), this captures the total return
        // including open positions marked to market, which is what answers
        // "am I winning or losing right now?".
        // Written by `accounting::mark_price::run` (src/accounting/mark_price.rs) every 5 min.
        // There is no way to rebuild the history retroactively: it starts accumulating
        // from the moment this table is deployed.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS equity_snapshots (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms               INTEGER NOT NULL,
                cash_um             INTEGER NOT NULL,
                positions_value_um  INTEGER NOT NULL,
                equity_um           INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_equity_snapshots_ts ON equity_snapshots(ts_ms)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS dead_letter_fills (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms       INTEGER NOT NULL,
                order_id    TEXT NOT NULL,
                market_cid  TEXT NOT NULL,
                token_id    TEXT NOT NULL,
                side        TEXT NOT NULL,
                price_um    INTEGER NOT NULL,
                size_um     INTEGER NOT NULL,
                status      TEXT NOT NULL,
                role        TEXT NOT NULL,
                reason      TEXT NOT NULL,
                resolved    INTEGER NOT NULL DEFAULT 0
            )",
        )
        .execute(&self.pool)
        .await?;
        // Quarantine rows (live fills without a resolvable owner) must store EVERYTHING needed
        // to rebuild the Fill and apply it late, once the owner is resolved (for example the race
        // between the WS fill and the POST response).
        // Rows for an unknown market (orphan fills) leave these columns NULL.
        // (see the ALTER loop below)
        for col_def in [
            "exec_id TEXT",
            "local_order_id TEXT",
            "strategy TEXT",
            "clear_at INTEGER",
            "neg_risk INTEGER",
            "attempt_id TEXT",
        ] {
            let sql = format!("ALTER TABLE dead_letter_fills ADD COLUMN {col_def}");
            if let Err(e) = sqlx::query(&sql).execute(&self.pool).await {
                if !e.to_string().to_lowercase().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }

        // Unknown-outcome fix (blocking): durable journal of send attempts whose
        // result is unknown. The row is inserted with `resolved=0` BEFORE calling
        // `router.submit`; a definitive rejection or a confirmed fill marks it
        // `resolved=1`. Without this, a crash or a restart would lose the mark, and
        // a blind resend of the same attempt would be possible.
        // (see `Executor::handle_buy`).
        //
        // Nothing closes it automatically (no clock or clear_at): the `unresolved` state
        // requires explicit reconciliation (a future dedicated worker that queries the exchange's
        // `/orders`). While `resolved=0`, `Executor::handle_buy` blocks submits for the same
        // bracket (fail-closed, conservative).
        // The row is never closed by a timer.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS unknown_attempts (
                attempt_id     TEXT PRIMARY KEY,
                cid            TEXT NOT NULL,
                token_id       TEXT NOT NULL,
                strategy       TEXT NOT NULL,
                cost_um        INTEGER NOT NULL,
                resting        INTEGER NOT NULL,
                reason         TEXT NOT NULL,
                ts_ms          INTEGER NOT NULL,
                resolved       INTEGER NOT NULL DEFAULT 0,
                resolved_ts_ms INTEGER,
                resolution     TEXT
            )",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_unknown_attempts_unresolved_cid_token \
             ON unknown_attempts(cid, token_id) WHERE resolved = 0",
        )
        .execute(&self.pool)
        .await?;
        // Unknown reconciler: `price_um`/`size_um` let the order be matched against the live
        // orders on the exchange when the `order_id` was lost with the POST response.
        // `clear_at` sets the expiry horizon of the claim kept in the ledger.
        // `side` stores the side declared by the intent ("buy"/"sell") in the SAME insert
        // as the pending. It is NULL in legacy rows, in quarantine rows (`qfill-*`) and in
        // cancel rows, where the side is unknown.
        // It is never inferred, so the reconciler does not match claims
        // from a guessed side.
        for col_def in [
            "price_um INTEGER",
            "size_um INTEGER",
            "clear_at INTEGER",
            "side TEXT",
        ] {
            let sql = format!("ALTER TABLE unknown_attempts ADD COLUMN {col_def}");
            if let Err(e) = sqlx::query(&sql).execute(&self.pool).await {
                if !e.to_string().to_lowercase().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }

        // ── Verifiable labels and resolution rules ──────────────────
        // Three new tables, added without touching the existing ones.
        // They hold the labels and rules that verify each resolution.
        //
        // `resolution_rules`: the resolution rule registered per market. There is NO universal
        // rule. Each market stores its tag and version, the source of truth
        // (`resolution_source`/`_ref`), the strike and the validity window.
        // `rule_detail` is the JSON of the `LabelRule` (TWAP windows, description of
        // external rules).
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS resolution_rules (
                market_cid              TEXT PRIMARY KEY,
                slug                    TEXT NOT NULL,
                rule_tag                TEXT NOT NULL,
                rule_detail             TEXT NOT NULL,
                rule_version            INTEGER NOT NULL,
                resolution_source       TEXT NOT NULL,
                resolution_source_ref   TEXT,
                reference_price         TEXT,
                market_open_ts          INTEGER,
                clear_at                INTEGER,
                updated_ts_ms           INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // `resolution_observations`: raw price observations that feed rules with a
        // window (TWAP) or serve as evidence for the provisional result. The price is
        // stored as TEXT and parsed to Decimal (monetary convention).
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS resolution_observations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market_cid  TEXT NOT NULL,
                ts_ms       INTEGER NOT NULL,
                price       TEXT NOT NULL,
                source      TEXT NOT NULL,
                recv_ts_ms  INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_res_obs_market_ts ON resolution_observations(market_cid, ts_ms)",
        )
        .execute(&self.pool)
        .await?;

        // `market_resolutions`: history of resolutions per market.
        // `status` = 'provisional' (a proxy, excluded from certified evaluation)
        // | 'confirmed' (identifiable official evidence) | 'superseded'
        // (corrected by a later record; `supersedes_id` links them). The
        // `provenance` is never empty: the write path validates it.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS market_resolutions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                market_cid      TEXT NOT NULL,
                winner          TEXT NOT NULL,
                status          TEXT NOT NULL,
                source          TEXT NOT NULL,
                provenance      TEXT NOT NULL,
                known_at_ms     INTEGER NOT NULL,
                rule_tag        TEXT,
                supersedes_id   INTEGER,
                detail          TEXT,
                recorded_ts_ms  INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_market_resolutions_cid ON market_resolutions(market_cid, id)",
        )
        .execute(&self.pool)
        .await?;

        // One row per arb_15m5m basket (streak), written when its outcome is known.
        // The PnL is NOT stored here: it is obtained at settlement by joining
        // `market_cid` with `transactions`.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS arb15m5m_streaks (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms             INTEGER NOT NULL,   -- time of the send
                asset            TEXT NOT NULL,
                t1                INTEGER NOT NULL,   -- common close of the pair (epoch s)
                slug_15m          TEXT NOT NULL,
                slug_5m           TEXT NOT NULL,
                cid_15m           TEXT NOT NULL,
                cid_5m            TEXT NOT NULL,
                token_15m         TEXT NOT NULL,      -- token bought in each market
                token_5m          TEXT NOT NULL,
                strike_15m        TEXT NOT NULL,      -- exact TWAP-60 s at t0, as Decimal
                strike_5m         TEXT NOT NULL,
                direction           TEXT NOT NULL,      -- 'up15_down5' | 'up5_down15'
                q_um              INTEGER NOT NULL,   -- shares sent ×1e6
                limit_15m_um     INTEGER NOT NULL,   -- limit prices of the FOK orders
                limit_5m_um      INTEGER NOT NULL,
                mean_cost_um    INTEGER NOT NULL,   -- per share, fees included
                expected_margin_um INTEGER NOT NULL,  -- 1 − medium_cost
                result         TEXT NOT NULL,      -- 'both' | 'orphan' | 'none' | 'rejected'
                filled_15m_um    INTEGER NOT NULL,
                filled_5m_um     INTEGER NOT NULL,
                real_cost_um     INTEGER NOT NULL,   -- Σ price × shares of the observed fills
                orphan_policy TEXT NOT NULL,
                hedge_id          INTEGER,            -- from the sim; NULL in live
                filled_per_book INTEGER NOT NULL DEFAULT 0, -- 1 = the sim decided from the book, without a draw
                depth_15m_um INTEGER NOT NULL DEFAULT 0, -- shares at <= limit on the 15m leg at decision time
                depth_5m_um  INTEGER NOT NULL DEFAULT 0
            )",
        )
        .execute(&self.pool)
        .await?;
        // Databases created before the last three columns were added.
        for col_def in [
            "filled_per_book INTEGER NOT NULL DEFAULT 0",
            "depth_15m_um INTEGER NOT NULL DEFAULT 0",
            "depth_5m_um INTEGER NOT NULL DEFAULT 0",
        ] {
            let sql = format!("ALTER TABLE arb15m5m_streaks ADD COLUMN {col_def}");
            if let Err(e) = sqlx::query(&sql).execute(&self.pool).await {
                if !e.to_string().to_lowercase().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_arb15m5m_streaks_t1 ON arb15m5m_streaks(t1)")
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Persists the outcome of an arb_15m5m streak.
    pub async fn record_arb15m5m_streak(&self, r: &Arb15m5mStreak) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO arb15m5m_streaks
                (ts_ms, asset, t1, slug_15m, slug_5m, cid_15m, cid_5m, token_15m, token_5m,
                 strike_15m, strike_5m, direction, q_um, limit_15m_um, limit_5m_um,
                 mean_cost_um, expected_margin_um, result, filled_15m_um, filled_5m_um,
                 real_cost_um, orphan_policy, hedge_id,
                 filled_per_book, depth_15m_um, depth_5m_um)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(r.ts_ms)
        .bind(&r.asset)
        .bind(r.t1)
        .bind(&r.slug_15m)
        .bind(&r.slug_5m)
        .bind(&r.cid_15m)
        .bind(&r.cid_5m)
        .bind(&r.token_15m)
        .bind(&r.token_5m)
        .bind(r.strike_15m.to_string())
        .bind(r.strike_5m.to_string())
        .bind(&r.direction)
        .bind(to_micros(r.q))
        .bind(to_micros(r.limit_15m))
        .bind(to_micros(r.limit_5m))
        .bind(to_micros(r.medium_cost))
        .bind(to_micros(r.expected_margin))
        .bind(&r.result)
        .bind(to_micros(r.filled_15m))
        .bind(to_micros(r.filled_5m))
        .bind(to_micros(r.real_cost))
        .bind(&r.orphan_policy)
        .bind(r.hedge_id)
        .bind(r.filled_per_book)
        .bind(to_micros(r.depth_15m))
        .bind(to_micros(r.depth_5m))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── Dead-letter fills ─────────────────────────────────────────────────────

    /// Persists a LIVE fill that could not be accounted for safely
    /// (market unknown to the registry and `clear_at_cache` at that
    /// instant). Self-contained: it does not touch any other table.
    pub async fn record_dead_letter_fill(&self, dl: &DeadLetterFill) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO dead_letter_fills
                (ts_ms, order_id, market_cid, token_id, side, price_um, size_um, status, role, reason)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(dl.ts.timestamp_millis())
        .bind(&dl.order_id)
        .bind(&dl.market_cid)
        .bind(&dl.token_id)
        .bind(dl.side.to_string())
        .bind(to_micros(dl.price))
        .bind(to_micros(dl.size))
        .bind(&dl.status)
        .bind(&dl.role)
        .bind(&dl.reason)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Counts the unresolved rows, to expose in `/api/status` and in the
    /// Telegram `/state` without fetching the whole content.
    pub async fn count_unresolved_dead_letters(&self) -> anyhow::Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) AS n FROM dead_letter_fills WHERE resolved = 0")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.try_get("n")?)
    }

    // ── Unknown attempts journal (blocking fix) ───────────────────────────────
    //
    // The executor records a `pending` BEFORE calling `router.submit`. If the POST
    // response is lost (`unknown`), the `pending` stays `resolved=0` and blocks new
    // submits for the same `cid:token_id` until it is explicitly reconciled via
    // `mark_unknown_resolved`. A definitive rejection or a confirmed fill closes it
    // with `resolved=1`, so that it no longer blocks.
    // The pending is therefore never silently dropped.
    //
    // Nothing resolves it automatically: the `unresolved` state needs a future worker
    // that queries the exchange's `/orders` and decides whether the order is still
    // live. Without that piece, this fix only documents the conservative blocker.

    /// Inserts a pending record before the POST. Returns `Err` if the write fails (in
    /// which case the `Executor` aborts the submit: fail-closed).
    ///
    /// `price_um`/`size_um`/`clear_at` feed the reconciler: match the order against the
    /// live orders on the exchange when the `order_id` was lost, and set the horizon of the
    /// claim kept in the ledger.
    ///
    /// Legacy API: equivalent to calling
    /// [`Self::record_unknown_attempt_pending_meta`] with `side = None`:
    /// unknown side. New submits must go through the `_meta` variant so that the side is
    /// stored durably in the SAME insert.
    #[allow(clippy::too_many_arguments)]
    pub async fn record_unknown_attempt_pending(
        &self,
        attempt_id: &str,
        cid: &str,
        token_id: &str,
        strategy: &str,
        cost_um: i64,
        resting: bool,
        reason: &str,
        price_um: i64,
        size_um: i64,
        clear_at: i64,
    ) -> anyhow::Result<()> {
        self.record_unknown_attempt_pending_meta(
            attempt_id,
            cid,
            token_id,
            strategy,
            cost_um,
            resting,
            reason,
            Some(price_um),
            Some(size_um),
            Some(clear_at),
            None,
        )
        .await
    }

    /// Pre-POST pending with complete metadata in ONE single
    /// insert. It is not inserted first and enriched later (a crash between
    /// both writes would leave the pending without its evidence).
    ///
    /// - `price_um`/`size_um`/`clear_at`: `None` when the real value is not known (cancel
    ///   pendings: the `price=0` of the `Cancel` intent is a protocol marker, NOT the
    ///   price of the order; stamping it here
    ///   would make the reconciler rebuild `remaining × 0 = 0` and
    ///   drop the claim through the `accepted-live` path).
    /// - `side`: `Some("buy")`/`Some("sell")` per the side declared by the
    ///   intent; `None` = unknown (cancel, quarantine, legacy). It is never
    ///   inferred.
    #[allow(clippy::too_many_arguments)]
    pub async fn record_unknown_attempt_pending_meta(
        &self,
        attempt_id: &str,
        cid: &str,
        token_id: &str,
        strategy: &str,
        cost_um: i64,
        resting: bool,
        reason: &str,
        price_um: Option<i64>,
        size_um: Option<i64>,
        clear_at: Option<i64>,
        side: Option<&str>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO unknown_attempts
                (attempt_id, cid, token_id, strategy, cost_um, resting, reason, ts_ms,
                 resolved, price_um, size_um, clear_at, side)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
        )
        .bind(attempt_id)
        .bind(cid)
        .bind(token_id)
        .bind(strategy)
        .bind(cost_um)
        .bind(resting as i64)
        .bind(reason)
        // `crate::time`: the Unknown reconciler compares `ts_ms` against
        // `crate::time::now()` (grace periods, trade windows, expiry). In
        // replay it must be the virtual clock, not the wall clock.
        .bind(crate::time::now().timestamp_millis())
        .bind(price_um)
        .bind(size_um)
        .bind(clear_at)
        .bind(side)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Tests only: backdates `ts_ms` of a pending in the journal. The
    /// reconciliation of taker Unknowns uses the attempt's age to decide
    /// whether it can already query the trade tape.
    #[cfg(test)]
    pub async fn test_set_unknown_ts_ms(&self, attempt_id: &str, ts_ms: i64) -> anyhow::Result<()> {
        sqlx::query("UPDATE unknown_attempts SET ts_ms = ? WHERE attempt_id = ?")
            .bind(ts_ms)
            .bind(attempt_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Marks a `pending` as `resolved`. The reason (`resolution`) is mandatory and
    /// describes the observed outcome. It is the ONLY valid mechanism to unblock an
    /// Unknown: the caller (a future reconciliation worker or a manual operator)
    /// confirms what happened to the order by querying the exchange's `/orders`.
    /// Integration with that worker is a future job. Today the method serves the tests
    /// and a human operator, and it is the place where a late exchange response
    /// will be handled.
    /// It is not called from the executor yet.
    /// Until then, the block stays in place.
    pub async fn mark_unknown_resolved(
        &self,
        attempt_id: &str,
        resolution: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE unknown_attempts
             SET resolved = 1,
                 resolved_ts_ms = ?,
                 resolution = ?
             WHERE attempt_id = ?",
        )
        // `crate::time`: same reason as the `ts_ms` of the insert: the journal
        // lives on the clock of the domain that produces it.
        .bind(crate::time::now().timestamp_millis())
        .bind(resolution)
        .bind(attempt_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Lists ALL the `unresolved` rows of the journal, used by the `Executor`
    /// at startup to repopulate its in-memory cache with the Unknowns that
    /// survived the last crash (the journal is the durable truth; the HashMap cache is
    /// only a mirror). If the DB fails, the `Executor` is fail-closed: it blocks new
    /// submits until the DB comes back.
    pub async fn list_unresolved_unknowns(&self) -> anyhow::Result<Vec<UnknownAttemptRow>> {
        let rows = sqlx::query(
            "SELECT attempt_id, cid, token_id, strategy, cost_um, resting,
                    reason, ts_ms, resolved, resolved_ts_ms, resolution,
                    price_um, size_um, clear_at, side
             FROM unknown_attempts
             WHERE resolved = 0
             ORDER BY ts_ms ASC",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(UnknownAttemptRow {
                attempt_id: r.try_get("attempt_id")?,
                cid: r.try_get("cid")?,
                token_id: r.try_get("token_id")?,
                strategy: r.try_get("strategy")?,
                cost_um: r.try_get("cost_um")?,
                resting: r.try_get::<i64, _>("resting")? != 0,
                reason: r.try_get("reason")?,
                ts_ms: r.try_get("ts_ms")?,
                resolved: r.try_get::<i64, _>("resolved")? != 0,
                resolved_ts_ms: r.try_get("resolved_ts_ms")?,
                resolution: r.try_get("resolution")?,
                price_um: r.try_get("price_um")?,
                size_um: r.try_get("size_um")?,
                clear_at: r.try_get("clear_at")?,
                side: r.try_get("side")?,
            });
        }
        Ok(out)
    }

    /// Returns true if there is AT LEAST one `unresolved` entry for the bracket
    /// (`cid` only, without token_id). Used by `Executor::handle_buy` to
    /// BLOCK new submits on any token of the same bracket while an earlier
    /// Unknown has not been reconciled.
    ///
    /// The block is at the **bracket** level, not the individual token: if
    /// the YES leg of bracket cid-1 ended Unknown, the NO leg of the same
    /// bracket is also blocked, which is exactly the window of double buying without
    /// accounting coverage that this fix targets. The test
    /// `second_attempt_other_token_same_bracket_also_blocked`
    /// checks this invariant.
    ///
    /// Documented decision: the scope is per bracket, NOT global: an
    /// unresolved entry for cid-A does not block submits for cid-B (test
    /// `rejection_does_not_block_other_bracket_that_has_its_own_unresolved`).
    /// If the decision is reversed to global in the future, change the
    /// `WHERE cid = ?` to a constant or remove the filter.
    pub async fn has_unresolved_unknown(&self, cid: &str) -> anyhow::Result<bool> {
        let n: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unknown_attempts
             WHERE resolved = 0 AND cid = ?",
        )
        .bind(cid)
        .fetch_one(&self.pool)
        .await?;
        Ok(n > 0)
    }

    /// Counts ALL the `unresolved` entries; useful for `/api/status`.
    pub async fn count_unresolved_unknowns(&self) -> anyhow::Result<i64> {
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM unknown_attempts WHERE resolved = 0")
            .fetch_one(&self.pool)
            .await?;
        Ok(n)
    }

    /// Strategy that owns an `attempt_id` of the Unknown journal. The executor uses the
    /// pre-POST `attempt_id` as the order's `local_order_id`, so this query resolves the
    /// owner of a live fill that carries a `local_order_id`. It survives restarts and does
    /// not depend on the in-memory `order_map`. `None` if the row does not exist or the
    /// stored strategy does not parse (quarantine rows carry the strategy declared by
    /// the fill, which may not be a valid `StrategyId`).
    /// Owner lookup for the fill path (see `fill_sink`).
    pub async fn get_unknown_attempt_strategy(
        &self,
        attempt_id: &str,
    ) -> anyhow::Result<Option<StrategyId>> {
        let row: Option<String> =
            sqlx::query_scalar("SELECT strategy FROM unknown_attempts WHERE attempt_id = ?")
                .bind(attempt_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.and_then(|s| StrategyId::from_str(&s).ok()))
    }

    // ── Order index ──────────────────────────────────────────────────────────

    /// Records `order_id → (local_order_id, strategy, cid, token)` when the
    /// executor learns the `order_id` of the submit. `INSERT OR REPLACE`: a
    /// re-record with the same key (e.g. startup reconciliation) only updates the
    /// row. Best-effort from the executor: if it fails, the matching live fill falls
    /// through to the next link of the resolution chain
    /// (`order_map`/`gtd_orders`) or to quarantine.
    pub async fn record_order_index(
        &self,
        order_id: &str,
        local_order_id: Option<&str>,
        strategy: StrategyId,
        cid: &str,
        token_id: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO order_index
                (order_id, local_order_id, strategy, cid, token_id, ts_ms)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(order_id)
        .bind(local_order_id)
        .bind(strategy.to_string())
        .bind(cid)
        .bind(token_id)
        // `crate::time`: `order_ids_since` is queried with a cutoff from
        // `crate::time::now()` (pruning of `known_order_ids`), so the stamp
        // must be in the same clock domain.
        .bind(crate::time::now().timestamp_millis())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Returns the `order_id`s recorded in `order_index` with `ts_ms >=
    /// cutoff_ms`. The periodic pruning of `known_order_ids` keeps the recent ids
    /// (they can still produce late trades on the user WS)
    /// in addition to the live ones in `active_gtd`.
    pub async fn order_ids_since(
        &self,
        cutoff_ms: i64,
    ) -> anyhow::Result<std::collections::HashSet<String>> {
        let rows = sqlx::query("SELECT order_id FROM order_index WHERE ts_ms >= ?")
            .bind(cutoff_ms)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r.try_get::<String, _>("order_id").ok())
            .collect())
    }

    /// Is a fill with this `exec_id` already durable? The Unknown reconciler uses it to
    /// resolve the journal ONLY once the synthesized fill is already settled: queuing it
    /// to the channel and resolving the journal first would leave a crash window in which
    /// the pending is closed without a position.
    pub async fn fill_exists_exec(&self, exec_id: &str) -> anyhow::Result<bool> {
        let exists: Option<i64> =
            sqlx::query_scalar("SELECT id FROM fills WHERE exec_id = ? LIMIT 1")
                .bind(exec_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(exists.is_some())
    }

    /// Tests only: backdates the `ts_ms` of a row of `order_index`.
    #[cfg(test)]
    pub async fn test_set_order_index_ts_ms(
        &self,
        order_id: &str,
        ts_ms: i64,
    ) -> anyhow::Result<()> {
        sqlx::query("UPDATE order_index SET ts_ms = ? WHERE order_id = ?")
            .bind(ts_ms)
            .bind(order_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Owner of an order by its exchange `order_id`. Returns
    /// `(strategy, local_order_id)`. `local_order_id` may be `None` in rows restored from
    /// sessions prior to that field. `None` if the order is not indexed or the stored
    /// strategy does not parse.
    pub async fn order_index_lookup(
        &self,
        order_id: &str,
    ) -> anyhow::Result<Option<(StrategyId, Option<String>)>> {
        let row =
            sqlx::query("SELECT strategy, local_order_id FROM order_index WHERE order_id = ?")
                .bind(order_id)
                .fetch_optional(&self.pool)
                .await?;
        match row {
            Some(r) => {
                let strategy: String = r.try_get("strategy")?;
                Ok(StrategyId::from_str(&strategy)
                    .ok()
                    .map(|s| (s, r.try_get("local_order_id").ok().flatten())))
            }
            None => Ok(None),
        }
    }

    /// Owner by `local_order_id`. Covers live fills that carry the local identity but
    /// whose `order_id` is not indexed (e.g. an accepted order whose
    /// `record_order_index` call failed).
    pub async fn order_index_lookup_local(
        &self,
        local_order_id: &str,
    ) -> anyhow::Result<Option<(StrategyId, Option<String>)>> {
        let row =
            sqlx::query("SELECT strategy, order_id FROM order_index WHERE local_order_id = ?")
                .bind(local_order_id)
                .fetch_optional(&self.pool)
                .await?;
        match row {
            Some(r) => {
                let strategy: String = r.try_get("strategy")?;
                Ok(StrategyId::from_str(&strategy)
                    .ok()
                    .map(|s| (s, r.try_get("order_id").ok().flatten())))
            }
            None => Ok(None),
        }
    }

    /// Full row of `order_index` by `order_id`, with the strategy as stored (raw). Unlike
    /// `order_index_lookup`, a row present with invalid data is NOT confused with
    /// "absent": the value is returned raw, so the caller parses and decides
    /// (fail-closed on conflict).
    pub async fn order_index_row(
        &self,
        order_id: &str,
    ) -> anyhow::Result<Option<OrderIndexEvidence>> {
        let row = sqlx::query(
            "SELECT order_id, local_order_id, strategy, cid, token_id
             FROM order_index WHERE order_id = ?",
        )
        .bind(order_id)
        .fetch_optional(&self.pool)
        .await?;
        match row {
            Some(r) => Ok(Some(OrderIndexEvidence {
                order_id: r.try_get("order_id")?,
                local_order_id: r.try_get("local_order_id")?,
                strategy: r.try_get("strategy")?,
                cid: r.try_get("cid")?,
                token_id: r.try_get("token_id")?,
            })),
            None => Ok(None),
        }
    }

    /// Same as [`Self::order_index_row`], but by `local_order_id`
    /// (the identity of the pending in the Unknown journal).
    pub async fn order_index_row_by_local(
        &self,
        local_order_id: &str,
    ) -> anyhow::Result<Option<OrderIndexEvidence>> {
        let row = sqlx::query(
            "SELECT order_id, local_order_id, strategy, cid, token_id
             FROM order_index WHERE local_order_id = ?",
        )
        .bind(local_order_id)
        .fetch_optional(&self.pool)
        .await?;
        match row {
            Some(r) => Ok(Some(OrderIndexEvidence {
                order_id: r.try_get("order_id")?,
                local_order_id: r.try_get("local_order_id")?,
                strategy: r.try_get("strategy")?,
                cid: r.try_get("cid")?,
                token_id: r.try_get("token_id")?,
            })),
            None => Ok(None),
        }
    }

    /// Raw row of `gtd_orders` as evidence, with the same rule as
    /// [`Self::order_index_row`]: a row present with an unrecognized strategy is NOT
    /// "absent" but indeterminate.
    pub async fn gtd_order_row(
        &self,
        order_id: &str,
    ) -> anyhow::Result<Option<GtdOrderRow>> {
        let row = sqlx::query("SELECT market_cid, strategy FROM gtd_orders WHERE order_id = ?")
            .bind(order_id)
            .fetch_optional(&self.pool)
            .await?;
        match row {
            Some(r) => Ok(Some(GtdOrderRow {
                market_cid: r.try_get("market_cid")?,
                strategy: r.try_get("strategy")?,
            })),
            None => Ok(None),
        }
    }

    /// Durable quarantine of a fill whose owner is NOT
    /// resolvable (no link in the chain: neither `local_order_id` in the journal, nor
    /// `order_id` in `order_index`/`gtd_orders`, nor `order_map` in memory).
    /// Previously the strategy of the Arb catch-all was attributed to it and it was
    /// accounted as its position: real money moved to the wrong owner.
    ///
    /// Effects, in ONE transaction:
    /// 1. Row in `dead_letter_fills` (reason = the one passed as a parameter) for
    ///    manual reconciliation: the fill is NOT applied to positions.
    /// 2. `pending` row in `unknown_attempts` with an `attempt_id` derived from the
    ///    fill's identity → `has_unresolved_unknown(cid)` becomes true and
    ///    `Executor::handle_buy` BLOCKS new buys for the WHOLE bracket
    ///    until an operator reconciles it with `mark_unknown_resolved`.
    ///
    /// Idempotent for the same fill identity: a second delivery of the same exec/order
    /// id collides with the attempt's PRIMARY KEY and `INSERT OR IGNORE` absorbs it (the
    /// quarantine was already in place). It returns `()` regardless: the intended effect
    /// (blocker + trace) is already achieved.
    /// BEGIN IMMEDIATE instead of the default deferred one: a deferred transaction that
    /// reads before writing can fail with `SQLITE_BUSY_SNAPSHOT` (517) on its first write or
    /// at COMMIT if another writer committed in between, and that error does NOT respect
    /// `busy_timeout` (the `database is locked` errors that fill_sink retried despite the
    /// 20s timeout). `BEGIN IMMEDIATE` acquires the RESERVED lock at the start: there the
    /// busy_timeout does apply, and no other writer can interleave while the
    /// transaction is live.
    async fn begin_immediate(&self) -> anyhow::Result<sqlx::Transaction<'static, sqlx::Sqlite>> {
        Ok(self.pool.begin_with("BEGIN IMMEDIATE").await?)
    }

    pub async fn quarantine_unowned_fill(&self, fill: &Fill, reason: &str) -> anyhow::Result<()> {
        let identity = fill
            .exec_id
            .as_deref()
            .or(fill.order_id.as_deref())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                // No identity at all: a synthetic id from the timestamp, so that
                // the row exists and the bracket is blocked all the same.
                // `crate::time`: the id enters the `unknown_attempts` journal
                // (via `qfill-*`) and the replay digest compares it byte by byte,
                // so it must be deterministic.
                format!(
                    "no-id-{}",
                    crate::time::now().timestamp_nanos_opt().unwrap_or(0)
                )
            });
        let attempt_id = format!("qfill-{identity}");
        let order_id = fill
            .order_id
            .clone()
            .or_else(|| fill.exec_id.clone())
            .unwrap_or_else(|| identity.clone());
        let mut tx = self.begin_immediate().await?;
        // Stores the COMPLETE row of the fill (identities, declared strategy,
        // clear_at, neg_risk, attempt_id) so that the reconciler can rebuild it and
        // apply it later if the owner appears.
        sqlx::query(
            "INSERT INTO dead_letter_fills
                (ts_ms, order_id, market_cid, token_id, side, price_um, size_um,
                 status, role, reason,
                 exec_id, local_order_id, strategy, clear_at, neg_risk, attempt_id)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(fill.ts.timestamp_millis())
        .bind(&order_id)
        .bind(&fill.market_cid)
        .bind(&fill.token_id)
        .bind(fill.side.to_string())
        .bind(to_micros(fill.price))
        .bind(to_micros(fill.size))
        .bind("MATCHED")
        .bind("unknown")
        .bind(reason)
        .bind(&fill.exec_id)
        .bind(&fill.local_order_id)
        .bind(fill.strategy.to_string())
        .bind(fill.clear_at)
        .bind(fill.neg_risk as i64)
        .bind(&attempt_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "INSERT OR IGNORE INTO unknown_attempts
                (attempt_id, cid, token_id, strategy, cost_um, resting, reason, ts_ms,
                 resolved, price_um, size_um, clear_at)
             VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0, ?, ?, ?)",
        )
        .bind(&attempt_id)
        .bind(&fill.market_cid)
        .bind(&fill.token_id)
        // The strategy DECLARED by the fill (may be the Arb catch-all from
        // handle_trade); the `reason` makes clear that it is not a verified attribution.
        .bind(fill.strategy.to_string())
        .bind(to_micros(fill.price * fill.size))
        .bind(reason)
        // `crate::time`: Unknown journal row; its `ts_ms` is compared
        // against the virtual clock in the reconciler.
        .bind(crate::time::now().timestamp_millis())
        .bind(to_micros(fill.price))
        .bind(to_micros(fill.size))
        .bind(fill.clear_at)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Lists the fills in quarantine that are not yet resolved. They are told apart
    /// from the dead-letters by the `attempt_id` column (only quarantine writes the
    /// companion `qfill-*` row in `unknown_attempts`). Each row carries the rebuilt
    /// Fill exactly as it arrived,
    /// ready for a retry of `resolve_owner` + `apply_fill`.
    pub async fn list_quarantined_fills(&self) -> anyhow::Result<Vec<(i64, Fill, String)>> {
        let rows = sqlx::query(
            "SELECT id, ts_ms, order_id, market_cid, token_id, side, price_um,
                    size_um, exec_id, local_order_id, strategy, clear_at,
                    neg_risk, attempt_id
             FROM dead_letter_fills
             WHERE resolved = 0 AND attempt_id IS NOT NULL
             ORDER BY id ASC",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let side_str: String = r.try_get("side")?;
            let Some(side) = (match side_str.as_str() {
                "BUY" => Some(FillSide::Buy),
                "SELL" => Some(FillSide::Sell),
                _ => None,
            }) else {
                tracing::warn!(
                    id = r.try_get::<i64, _>("id").unwrap_or_default(),
                    side = %side_str,
                    "state: quarantined dead-letter with an unusable side, skipping"
                );
                continue;
            };
            let strat_str: Option<String> = r.try_get("strategy")?;
            out.push((
                r.try_get("id")?,
                Fill {
                    market_cid: r.try_get("market_cid")?,
                    token_id: r.try_get("token_id")?,
                    side,
                    price: from_micros(r.try_get("price_um")?),
                    size: from_micros(r.try_get("size_um")?),
                    order_id: Some(r.try_get::<String, _>("order_id")?),
                    exec_id: r.try_get("exec_id")?,
                    local_order_id: r.try_get("local_order_id")?,
                    strategy: strat_str
                        .as_deref()
                        .map(StrategyId::from_db_str)
                        .unwrap_or(StrategyId::Arb),
                    clear_at: r.try_get::<Option<i64>, _>("clear_at")?.unwrap_or(0),
                    neg_risk: r.try_get::<Option<i64>, _>("neg_risk")?.unwrap_or(0) != 0,
                    ts: chrono::DateTime::from_timestamp_millis(r.try_get("ts_ms")?)
                        .unwrap_or_else(chrono::Utc::now),
                    mode: Mode::Live,
                    outcome_label: None,
                    hedge_id: None,
                },
                r.try_get::<String, _>("attempt_id")?,
            ));
        }
        Ok(out)
    }

    /// Closes a quarantine ALREADY applied to positions: marks the dead-letter as
    /// resolved and the companion `unknown_attempts` row as reconciled (this
    /// unblocks the bracket in `has_unresolved_unknown`). A single transaction: if the
    /// second write fails, the dead-letter stays pending and the reconciler will retry it
    /// (idempotent: the fill is already in `fills`, and `record_fill_conn` dedups it by
    /// `exec_id`).
    pub async fn resolve_quarantined_fill(
        &self,
        dead_letter_id: i64,
        attempt_id: &str,
        resolution: &str,
    ) -> anyhow::Result<()> {
        let mut tx = self.begin_immediate().await?;
        sqlx::query("UPDATE dead_letter_fills SET resolved = 1 WHERE id = ?")
            .bind(dead_letter_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "UPDATE unknown_attempts
             SET resolved = 1, resolved_ts_ms = ?, resolution = ?
             WHERE attempt_id = ?",
        )
        // `crate::time`: column of the Unknown journal, on the virtual clock.
        .bind(crate::time::now().timestamp_millis())
        .bind(resolution)
        .bind(attempt_id)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    // ── Risk state ────────────────────────────────────────────────────────────

    /// Persists (or updates) the risk state of a scope (`"global"` or the
    /// name of a strategy). Self-contained: it does not touch any other table.
    pub async fn save_risk_state(
        &self,
        scope: &str,
        killed: bool,
        reason: Option<&str>,
        session_pnl: Decimal,
        loss_stop_baseline: Decimal,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO risk_state
                (scope, killed, reason, session_pnl_um, loss_stop_baseline_um, updated_ms)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(scope)
        .bind(killed as i64)
        .bind(reason)
        .bind(to_micros(session_pnl))
        .bind(to_micros(loss_stop_baseline))
        // `crate::time`: `updated_ms` is audit data, but it follows the clock
        // domain in use (virtual during replay).
        .bind(crate::time::now().timestamp_millis())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Reads all the persisted risk state (all the scopes). It is called
    /// once at startup, in `main.rs`, to restore `RiskGuard`.
    pub async fn load_risk_state(&self) -> anyhow::Result<Vec<RiskStateRow>> {
        let rows = sqlx::query(
            "SELECT scope, killed, reason, session_pnl_um, loss_stop_baseline_um FROM risk_state",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(RiskStateRow {
                scope: r.try_get("scope")?,
                killed: r.try_get::<i64, _>("killed")? != 0,
                reason: r.try_get("reason").ok().flatten(),
                session_pnl: from_micros(r.try_get("session_pnl_um")?),
                loss_stop_baseline: from_micros(r.try_get("loss_stop_baseline_um")?),
            });
        }
        Ok(out)
    }

    // ── GTD orders ────────────────────────────────────────────────────────────

    /// Records (or updates) the strategy that owns a GTD that is live on the CLOB.
    pub async fn record_gtd_order(
        &self,
        order_id: &str,
        market_cid: &str,
        strategy: StrategyId,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO gtd_orders (order_id, market_cid, strategy, ts_ms) VALUES (?, ?, ?, ?)",
        )
        .bind(order_id)
        .bind(market_cid)
        .bind(strategy.to_string())
        // `crate::time`: `purge_stale_gtd_orders` compares `ts_ms` against a
        // cutoff from `crate::time::now()`, so it must be in the same clock domain.
        .bind(crate::time::now().timestamp_millis())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Removes the record of a GTD (cancelled or expired).
    pub async fn delete_gtd_order(&self, order_id: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM gtd_orders WHERE order_id = ?")
            .bind(order_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// `gtd_orders` was only purged on explicit cancellation: a GTD whose cancellation
    /// record was lost (crash, bug in the caller) stayed there forever. This exposes a purge
    /// by age, so that the caller (the executor, which owns the GTD lifecycle and is outside
    /// this package's scope) can invoke it periodically if it deems it necessary.
    /// This module does not schedule the purge by itself, and it is not run
    /// automatically anywhere in this file.
    /// Returns the number of rows purged.
    pub async fn purge_stale_gtd_orders(&self, older_than_secs: i64) -> anyhow::Result<u64> {
        // `crate::time`: the cutoff decides which rows are purged; in replay it is
        // tape time, not wall-clock time.
        let cutoff = crate::time::now().timestamp_millis() - older_than_secs * 1000;
        let r = sqlx::query("DELETE FROM gtd_orders WHERE ts_ms < ?")
            .bind(cutoff)
            .execute(&self.pool)
            .await?;
        Ok(r.rows_affected())
    }

    /// `pool` is private to the module: this helper only exists so that the tests of
    /// other modules (e.g. `order::executor`) can insert a `gtd_orders` row with an arbitrary
    /// `ts_ms` and thus exercise the age-based purge end to end. It is never compiled
    /// outside `cfg(test)`.
    #[cfg(test)]
    pub(crate) async fn insert_gtd_order_row_for_test(
        &self,
        order_id: &str,
        market_cid: &str,
        strategy: &str,
        ts_ms: i64,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO gtd_orders (order_id, market_cid, strategy, ts_ms) VALUES (?, ?, ?, ?)",
        )
        .bind(order_id)
        .bind(market_cid)
        .bind(strategy)
        .bind(ts_ms)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Timestamp (ms) of the most recent fill in the DB: the starting point of the
    /// backfill of trades lost during downtime.
    pub async fn last_fill_ts_ms(&self) -> anyhow::Result<Option<i64>> {
        let row = sqlx::query("SELECT MAX(ts_ms) AS ts FROM fills")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.try_get::<Option<i64>, _>("ts")?)
    }

    /// Strategy that owns a GTD from an earlier session (None if none is recorded).
    pub async fn get_gtd_strategy(&self, order_id: &str) -> anyhow::Result<Option<StrategyId>> {
        let row = sqlx::query("SELECT strategy FROM gtd_orders WHERE order_id = ?")
            .bind(order_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row
            .and_then(|r| r.try_get::<String, _>("strategy").ok())
            .and_then(|s| s.parse().ok()))
    }

    /// Inserts a scan observation.
    pub async fn record_scan(
        &self,
        ts_ms: i64,
        strategy: &str,
        market_key: &str,
        kind: &str,
        edge: Decimal,
        payload: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO scan_observations (ts_ms, strategy, market_key, kind, edge_um, payload)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(ts_ms)
        .bind(strategy)
        .bind(market_key)
        .bind(kind)
        .bind(to_micros(edge))
        .bind(payload)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Number of scan observations per strategy (verification/tests).
    pub async fn scan_observations_count(&self, strategy: &str) -> anyhow::Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) AS n FROM scan_observations WHERE strategy = ?")
            .bind(strategy)
            .fetch_one(&self.pool)
            .await?;
        Ok(row.try_get("n")?)
    }

    // ── Markouts ──────────────────────────────────────────────────────────────

    /// Inserts the markout of a fill (a single write per fill, off the hot path).
    #[allow(clippy::too_many_arguments)]
    pub async fn record_markout(
        &self,
        ts_ms: i64,
        market_cid: &str,
        token_id: &str,
        strategy: &str,
        side: &str,
        fill_price: Decimal,
        mid_5s: Option<Decimal>,
        mid_30s: Option<Decimal>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO markouts (ts_ms, market_cid, token_id, strategy, side, fill_price_um, mid_5s_um, mid_30s_um)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(ts_ms)
        .bind(market_cid)
        .bind(token_id)
        .bind(strategy)
        .bind(side)
        .bind(to_micros(fill_price))
        .bind(mid_5s.map(to_micros))
        .bind(mid_30s.map(to_micros))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Markout rows of a strategy: (token_id, fill_price, mid_5s, mid_30s).
    pub async fn markouts_for_strategy(
        &self,
        strategy: &str,
    ) -> anyhow::Result<Vec<(String, Decimal, Option<Decimal>, Option<Decimal>)>> {
        let rows = sqlx::query(
            "SELECT token_id, fill_price_um, mid_5s_um, mid_30s_um FROM markouts \
             WHERE strategy = ? ORDER BY ts_ms",
        )
        .bind(strategy)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|r| {
                Ok((
                    r.try_get::<String, _>("token_id")?,
                    from_micros(r.try_get("fill_price_um")?),
                    r.try_get::<Option<i64>, _>("mid_5s_um")?.map(from_micros),
                    r.try_get::<Option<i64>, _>("mid_30s_um")?.map(from_micros),
                ))
            })
            .collect()
    }

    // ── Passive post-close observer ──────────────────────────────────────────

    /// Inserts a post-close observation (one row per sample, modelled on `record_scan`).
    /// The `cl_*` columns stay in the schema for the historical rows; the external feed
    /// that fed them was retired, and new rows leave them NULL
    /// (they are not part of the INSERT).
    #[allow(clippy::too_many_arguments)]
    pub async fn record_post_close(
        &self,
        ts_ms: i64,
        secs_since_close: i64,
        market_cid: &str,
        slug: &str,
        clear_at: i64,
        reference_price: Option<Decimal>,
        spot: Option<Decimal>,
        spot_stale: bool,
        rel_move: Option<Decimal>,
        verdict: &str,
        yes_bid: Option<Decimal>,
        yes_ask: Option<Decimal>,
        yes_bid_sz: Option<Decimal>,
        yes_ask_sz: Option<Decimal>,
        no_bid: Option<Decimal>,
        no_ask: Option<Decimal>,
        no_bid_sz: Option<Decimal>,
        no_ask_sz: Option<Decimal>,
        winner_ask: Option<Decimal>,
        winner_ask_sz: Option<Decimal>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO post_close_observations (
                ts_ms, secs_since_close, market_cid, slug, clear_at,
                reference_price_um, spot_um, spot_stale, rel_move_um, verdict,
                yes_bid_um, yes_ask_um, yes_bid_sz_um, yes_ask_sz_um,
                no_bid_um, no_ask_um, no_bid_sz_um, no_ask_sz_um,
                winner_ask_um, winner_ask_sz_um
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(ts_ms)
        .bind(secs_since_close)
        .bind(market_cid)
        .bind(slug)
        .bind(clear_at)
        .bind(reference_price.map(to_micros))
        .bind(spot.map(to_micros))
        .bind(spot_stale as i64)
        .bind(rel_move.map(to_micros))
        .bind(verdict)
        .bind(yes_bid.map(to_micros))
        .bind(yes_ask.map(to_micros))
        .bind(yes_bid_sz.map(to_micros))
        .bind(yes_ask_sz.map(to_micros))
        .bind(no_bid.map(to_micros))
        .bind(no_ask.map(to_micros))
        .bind(no_bid_sz.map(to_micros))
        .bind(no_ask_sz.map(to_micros))
        .bind(winner_ask.map(to_micros))
        .bind(winner_ask_sz.map(to_micros))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Number of post-close observations persisted (verification/tests).
    pub async fn post_close_observations_count(&self) -> anyhow::Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) AS n FROM post_close_observations")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.try_get("n")?)
    }

    // ── Verifiable labels and resolution rules ───────────────────────────────
    //
    // Persistence for `crate::research::labels`. The reconciliation (provisional →
    // confirmed → superseded) lives there; here we only read and write. Scans use a
    // strict decoder that rejects unknown values: the SQL fetches
    // ALL rows of the market WITHOUT filtering by `status`/`winner`. An out-of-vocabulary
    // value is corruption: it aborts the whole scan with an error, never a partial Ok
    // that would hide it.
    // Those rows are then never silently skipped.

    /// Upsert of the resolution rule registered for a market (PK by cid).
    pub async fn upsert_resolution_rule(
        &self,
        reg: &crate::research::labels::RuleRegistration,
        updated_ts_ms: i64,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO resolution_rules
                (market_cid, slug, rule_tag, rule_detail, rule_version,
                 resolution_source, resolution_source_ref, reference_price,
                 market_open_ts, clear_at, updated_ts_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(market_cid) DO UPDATE SET
                slug = excluded.slug,
                rule_tag = excluded.rule_tag,
                rule_detail = excluded.rule_detail,
                rule_version = excluded.rule_version,
                resolution_source = excluded.resolution_source,
                resolution_source_ref = excluded.resolution_source_ref,
                reference_price = excluded.reference_price,
                market_open_ts = excluded.market_open_ts,
                clear_at = excluded.clear_at,
                updated_ts_ms = excluded.updated_ts_ms",
        )
        .bind(&reg.market_cid)
        .bind(&reg.slug)
        .bind(reg.rule.tag())
        .bind(serde_json::to_string(&reg.rule).context("serializing LabelRule")?)
        .bind(reg.rule_version as i64)
        .bind(crate::research::labels::source_tag(reg.resolution_source))
        .bind(&reg.resolution_source_ref)
        .bind(reg.reference_price.map(|d| d.to_string()))
        .bind(reg.market_open_ts)
        .bind(reg.clear_at)
        .bind(updated_ts_ms)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Reads the rule registered for a market. `None` = not registered; a
    /// row with an unknown vocabulary = error (strict decoder).
    pub async fn resolution_rule(
        &self,
        market_cid: &str,
    ) -> anyhow::Result<Option<crate::research::labels::RuleRegistration>> {
        let row = sqlx::query(
            "SELECT market_cid, slug, rule_tag, rule_detail, rule_version,
                    resolution_source, resolution_source_ref, reference_price,
                    market_open_ts, clear_at
               FROM resolution_rules WHERE market_cid = ?",
        )
        .bind(market_cid)
        .fetch_optional(&self.pool)
        .await?;
        match row {
            Some(r) => crate::research::labels::decode_rule_row(&r).map(Some),
            None => Ok(None),
        }
    }

    /// Persists a price observation for the resolution window (raw
    /// evidence: rules with a window, such as TWAP, are evaluated over these rows;
    /// they are not deduplicated, since they are observations).
    pub async fn record_resolution_observation(
        &self,
        market_cid: &str,
        obs: &crate::research::labels::PriceObservation,
        recv_ts_ms: i64,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO resolution_observations
                (market_cid, ts_ms, price, source, recv_ts_ms)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(market_cid)
        .bind(obs.ts_ms)
        .bind(obs.price.to_string())
        .bind(crate::research::labels::source_tag(obs.source))
        .bind(recv_ts_ms)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// All the price observations of a market, in arrival order.
    /// Strict decoder: an unparseable price or an unknown source is an error.
    pub async fn resolution_observations(
        &self,
        market_cid: &str,
    ) -> anyhow::Result<Vec<crate::research::labels::PriceObservation>> {
        let rows = sqlx::query(
            "SELECT ts_ms, price, source FROM resolution_observations
              WHERE market_cid = ? ORDER BY id ASC",
        )
        .bind(market_cid)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(crate::research::labels::decode_observation_row)
            .collect()
    }

    /// Inserts a resolution record (provisional or confirmed).
    /// Returns the assigned `id`. An empty `provenance` is rejected here too
    /// (defense in depth; `MarketOutcome::new` already validates it).
    pub async fn insert_resolution(
        &self,
        rec: &crate::research::labels::ResolutionRecord,
    ) -> anyhow::Result<i64> {
        if rec.provenance.trim().is_empty() {
            anyhow::bail!(
                "insert_resolution: provenance is empty (cid {})",
                rec.market_cid
            );
        }
        let res = sqlx::query(
            "INSERT INTO market_resolutions
                (market_cid, winner, status, source, provenance, known_at_ms,
                 rule_tag, supersedes_id, detail, recorded_ts_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&rec.market_cid)
        .bind(crate::research::labels::side_tag(rec.winner))
        .bind(rec.status.as_str())
        .bind(crate::research::labels::source_tag(rec.source))
        .bind(&rec.provenance)
        .bind(rec.known_at_ms)
        .bind(&rec.rule_tag)
        .bind(rec.supersedes_id)
        .bind(&rec.detail)
        .bind(rec.recorded_ts_ms)
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid())
    }

    /// Complete history of the resolution rows of a market, in insertion order.
    /// NO `WHERE` on `status`/`winner`: the strict decoder
    /// (`decode_resolution_row`) aborts on any value outside the vocabulary; filtering
    /// in SQL would make the corruption invisible.
    pub async fn resolutions_for(
        &self,
        market_cid: &str,
    ) -> anyhow::Result<Vec<crate::research::labels::ResolutionRecord>> {
        let rows = sqlx::query(
            "SELECT id, market_cid, winner, status, source, provenance,
                    known_at_ms, rule_tag, supersedes_id, detail, recorded_ts_ms
               FROM market_resolutions
              WHERE market_cid = ? ORDER BY id ASC",
        )
        .bind(market_cid)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(crate::research::labels::decode_resolution_row)
            .collect()
    }

    /// Marks a confirmed record as superseded (corrected). The link
    /// `supersedes_id` is carried by the NEW row; here only the status changes.
    /// Applies only to `confirmed` rows: a provisional one is never corrected (it is
    /// a proxy, kept as historical evidence of what the oracle said).
    ///
    /// Prefer [`Self::insert_resolution_superseding`] for the correction path:
    /// insert and mark must be atomic; a standalone UPDATE after a separate insert
    /// would leave TWO valid `confirmed` rows if it failed.
    pub async fn mark_resolution_superseded(&self, superseded_id: i64) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE market_resolutions SET status = 'superseded'
              WHERE id = ? AND status = 'confirmed'",
        )
        .bind(superseded_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// ATOMIC resolution correction: inserts the new row and marks the earlier
    /// confirmed one as `superseded` in ONE `BEGIN IMMEDIATE` transaction (the write
    /// lock goes through `busy_timeout`, see `begin_immediate`). Without atomicity, a
    /// failure between both writes would leave two valid `confirmed` rows, and a later
    /// correction would only touch the last one.
    /// Invariant: at most one confirmed row per market.
    ///
    /// Returns the `id` of the new row. If the row to correct does not exist or
    /// is not `confirmed`, the transaction ROLLS BACK: neither insert nor mark.
    pub async fn insert_resolution_superseding(
        &self,
        rec: &crate::research::labels::ResolutionRecord,
        superseded_id: i64,
    ) -> anyhow::Result<i64> {
        if rec.provenance.trim().is_empty() {
            anyhow::bail!(
                "insert_resolution_superseding: provenance is empty (cid {})",
                rec.market_cid
            );
        }
        let mut tx = self.begin_immediate().await?;
        let res = sqlx::query(
            "INSERT INTO market_resolutions
                (market_cid, winner, status, source, provenance, known_at_ms,
                 rule_tag, supersedes_id, detail, recorded_ts_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&rec.market_cid)
        .bind(crate::research::labels::side_tag(rec.winner))
        .bind(rec.status.as_str())
        .bind(crate::research::labels::source_tag(rec.source))
        .bind(&rec.provenance)
        .bind(rec.known_at_ms)
        .bind(&rec.rule_tag)
        .bind(rec.supersedes_id)
        .bind(&rec.detail)
        .bind(rec.recorded_ts_ms)
        .execute(&mut *tx)
        .await?;
        let new_id = res.last_insert_rowid();
        let upd = sqlx::query(
            "UPDATE market_resolutions SET status = 'superseded'
              WHERE id = ? AND status = 'confirmed'",
        )
        .bind(superseded_id)
        .execute(&mut *tx)
        .await?;
        if upd.rows_affected() != 1 {
            // Rollback on drop of `tx`: neither the new row nor the supersede.
            anyhow::bail!(
                "insert_resolution_superseding: row {superseded_id} does not exist or is not confirmed"
            );
        }
        tx.commit().await?;
        Ok(new_id)
    }

    /// Observed markets whose close has passed and that have no current `confirmed`
    /// resolution: the job set of the official fetcher (`src/tools/recorders/resolution.rs`).
    /// `clear_at` is in SECONDS (same unit as `BookSnapshot::clear_at` and
    /// `upsert_market`); `min_clear_s` bounds the horizon from below.
    ///
    /// A superseded `confirmed` row does NOT exclude the market: a correction may arrive
    /// again from the source. The `market_resolutions` rows are not decoded here (the
    /// `status='confirmed'` filter only bounds the job set; the strict decoder still applies
    /// in `resolutions_for`).
    ///
    /// Ordered by `clear_at DESC` (most recently closed first) on purpose: markets that
    /// NEVER reach `resolved` (disputed, purged from Gamma, or with a corrupt `clear_at`,
    /// which the `> 0` filter lets through) must not hold the head of the batch forever
    /// and starve the real pending markets.
    /// This ordering was chosen after a review finding.
    pub async fn markets_pending_official(
        &self,
        now_s: i64,
        min_clear_s: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<OfficialPendingMarket>> {
        let rows = sqlx::query(
            "SELECT m.condition_id, m.slug, m.yes_token_id, m.no_token_id,
                    m.market_open_ts, m.clear_at
               FROM markets_observed m
              WHERE m.clear_at > 0
                AND m.clear_at <= ?
                AND m.clear_at > ?
                AND NOT EXISTS (SELECT 1 FROM market_resolutions r
                                WHERE r.market_cid = m.condition_id
                                  AND r.status = 'confirmed')
              ORDER BY m.clear_at ASC LIMIT ?",
        )
        .bind(now_s)
        .bind(min_clear_s)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| OfficialPendingMarket {
                condition_id: r.get::<String, _>("condition_id"),
                slug: r.get::<String, _>("slug"),
                yes_token_id: r.get::<String, _>("yes_token_id"),
                no_token_id: r.get::<String, _>("no_token_id"),
                market_open_ts: r.get::<Option<i64>, _>("market_open_ts"),
                clear_at: r.get::<i64, _>("clear_at"),
            })
            .collect())
    }

    /// Inserts the resolution rule only if the market does not already have one
    /// (`ON CONFLICT DO NOTHING`). Returns `true` if it inserted.
    ///
    /// Unlike `upsert_resolution_rule` (which rewrites the row), this is the safe variant
    /// for the fetcher. `ensure_rule` did a check-then-act (`rule_for` → `register_rule`),
    /// and `register_rule` did an upsert. If post_close registered the rich rule (with the
    /// real `reference_price`/`market_open_ts`/`clear_at`) between the fetcher's read and
    /// write, the fetcher's bare rule OVERWROTE it, and the provisional pipeline
    /// silently died for that market.
    /// Here SQLite provides the atomicity (PK + DO NOTHING): whoever arrives first wins,
    /// and post_close's upsert can still enrich the row afterwards.
    /// The fetcher's bare rule therefore never replaces a rich one.
    /// (`ON CONFLICT` keeps the existing row.)
    pub async fn insert_resolution_rule_if_absent(
        &self,
        reg: &crate::research::labels::RuleRegistration,
        updated_ts_ms: i64,
    ) -> anyhow::Result<bool> {
        let res = sqlx::query(
            "INSERT INTO resolution_rules
                (market_cid, slug, rule_tag, rule_detail, rule_version,
                 resolution_source, resolution_source_ref, reference_price,
                 market_open_ts, clear_at, updated_ts_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(market_cid) DO NOTHING",
        )
        .bind(&reg.market_cid)
        .bind(&reg.slug)
        .bind(reg.rule.tag())
        .bind(serde_json::to_string(&reg.rule).context("serializing LabelRule")?)
        .bind(reg.rule_version as i64)
        .bind(crate::research::labels::source_tag(reg.resolution_source))
        .bind(&reg.resolution_source_ref)
        .bind(reg.reference_price.map(|d| d.to_string()))
        .bind(reg.market_open_ts)
        .bind(reg.clear_at)
        .bind(updated_ts_ms)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() == 1)
    }

    // ── Positions ─────────────────────────────────────────────────────────────

    async fn open_position_conn(
        conn: &mut sqlx::SqliteConnection,
        pos: &Position,
    ) -> anyhow::Result<()> {
        // Full list of fields so that INSERT OR REPLACE does not reset to DEFAULT
        // the omitted columns (bail_sold, outcome_won, winner_label, redeemed, redeem_tx).
        sqlx::query(
            "INSERT OR REPLACE INTO positions
             (token_id, market_cid, strategy, size_um, avg_price_um, cost_basis_um,
              opened_ts_ms, clear_at, neg_risk, outcome_label,
              bail_sold, outcome_won, winner_label, redeemed, redeem_tx)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&pos.token_id)
        .bind(&pos.market_cid)
        .bind(&pos.strategy)
        .bind(to_micros(pos.size))
        .bind(to_micros(pos.avg_price))
        .bind(to_micros(pos.cost_basis))
        .bind(pos.opened_ts_ms)
        .bind(pos.clear_at)
        .bind(pos.neg_risk as i64)
        .bind(&pos.outcome_label)
        .bind(pos.bail_sold as i64)
        .bind(pos.outcome_won.map(|b| b as i64))
        .bind(&pos.winner_label)
        .bind(pos.redeemed as i64)
        .bind(&pos.redeem_tx)
        .execute(conn)
        .await?;
        Ok(())
    }

    pub async fn list_open_positions(&self) -> anyhow::Result<Vec<Position>> {
        let rows = sqlx::query(
            "SELECT p.token_id, p.market_cid, COALESCE(m.slug, '') AS slug,
                    p.strategy, p.size_um, p.avg_price_um, p.cost_basis_um,
                    p.opened_ts_ms, p.clear_at, p.neg_risk, p.bail_sold, p.outcome_won,
                    p.winner_label, p.outcome_label, p.redeemed, p.redeem_tx
             FROM positions p
             LEFT JOIN markets_observed m ON m.condition_id = p.market_cid
             WHERE p.redeemed = 0 AND p.bail_sold = 0 AND p.outcome_won IS NULL",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(Self::row_to_position).collect()
    }

    pub async fn get_position(
        &self,
        token_id: &str,
        strategy: &str,
    ) -> anyhow::Result<Option<Position>> {
        let mut conn = self.pool.acquire().await?;
        Self::get_position_conn(&mut conn, token_id, strategy).await
    }

    async fn get_position_conn(
        conn: &mut sqlx::SqliteConnection,
        token_id: &str,
        strategy: &str,
    ) -> anyhow::Result<Option<Position>> {
        let row = sqlx::query(
            "SELECT token_id, market_cid, strategy, size_um, avg_price_um, cost_basis_um,
                    opened_ts_ms, clear_at, neg_risk, bail_sold, outcome_won, winner_label,
                    outcome_label, redeemed, redeem_tx
             FROM positions WHERE token_id = ? AND strategy = ?",
        )
        .bind(token_id)
        .bind(strategy)
        .fetch_optional(conn)
        .await?;
        row.as_ref().map(Self::row_to_position).transpose()
    }

    async fn reduce_position_conn(
        conn: &mut sqlx::SqliteConnection,
        token_id: &str,
        strategy: &str,
        sold_size: Decimal,
    ) -> anyhow::Result<Decimal> {
        // Reduces the size and returns the cost allocated for the SELL's PnL.
        // Sets bail_sold=1 only if the sell covers the whole position (full close).
        // outcome_won is left NULL so that the orphan checker resolves it after settlement.
        // A partial sell (partial FAK fill) leaves the position visible and active
        // with the remaining size, which the orphan checker settles at expiry.
        // Do not divide before multiplying: at a price below 1 that would give 0.
        // Two positive i64 values multiplied fit in i128. We truncate the cost
        // sold to the micro and keep ALL the remainder in the position; a full sell
        // consumes the exact remaining cost, never avg_price.
        // The caller holds the read, the reduction and the posting within its tx.
        let row = sqlx::query(
            "SELECT size_um, cost_basis_um, avg_price_um FROM positions WHERE token_id = ? AND strategy = ?",
        ).bind(token_id).bind(strategy).fetch_one(&mut *conn).await?;
        let size_um: i64 = row.try_get("size_um")?;
        let cost_um: i64 = row.try_get("cost_basis_um")?;
        let avg_um: i64 = row.try_get("avg_price_um")?;
        let sold_um = to_micros(sold_size);
        // A corrupt row inherits the "transient failure" handling (fill_sink retries 3
        // times and ends in kill_global), so the message must be enough to diagnose the row
        // without opening the database.
        anyhow::ensure!(
            size_um > 0 && cost_um >= 0 && sold_um >= 0 && sold_um <= size_um,
            "invalid position cost allocation: token={token_id} strategy={strategy} \
             size_um={size_um} cost_um={cost_um} sold_um={sold_um}"
        );
        let allocated_um =
            i64::try_from(i128::from(cost_um) * i128::from(sold_um) / i128::from(size_um))?;
        // `avg_price_um` must stay cost/size after the partial sell.
        // It is the fallback mark price (src/accounting/mark_price.rs, api/handlers.rs): leaving it
        // stale would value the residual position at a price that no longer exists. It is
        // computed in Rust and bound, to avoid overflow risk inside
        // SQLite. Without a remaining size there is no possible division: the current
        // value is kept.
        //
        // `avg_price` is a DERIVED ratio, not a conserved quantity like the cost (which is
        // truncated on purpose to keep the remainder in the position). It is rounded to the
        // nearest micro with the same `to_micros` as the rest of the module: truncating it
        // biased it downwards and undervalued the residual position.
        // (see the rounding in `to_micros`).
        let remaining_um = size_um - sold_um;
        let remaining_cost_um = cost_um - allocated_um;
        let new_avg_um = if remaining_um > 0 {
            to_micros(from_micros(remaining_cost_um) / from_micros(remaining_um))
        } else {
            avg_um
        };
        // A share of an outcome is never worth more than 1 USDC. Without this cap, a
        // legacy row with an inflated cost (the kind the migration deals with)
        // comes out of the recalculation with an avg_price > 1 that leaks into
        // `positions_value` → equity through the mark-price fallback.
        let new_avg_um = new_avg_um.min(1_000_000);
        let res = sqlx::query(
            "UPDATE positions SET
                 size_um       = size_um - ?,
                 cost_basis_um = cost_basis_um - ?,
                 avg_price_um  = ?,
                 bail_sold     = CASE WHEN size_um <= ? THEN 1 ELSE bail_sold END
             WHERE token_id = ? AND strategy = ? AND size_um > 0",
        )
        .bind(sold_um)
        .bind(allocated_um)
        .bind(new_avg_um)
        .bind(sold_um)
        .bind(token_id)
        .bind(strategy)
        .execute(conn)
        .await?;
        // Without this, an UPDATE that touched no row would still return `allocated_um`,
        // and a PnL would be posted against a reduction that never happened.
        // (rows_affected must be exactly one.)
        anyhow::ensure!(
            res.rows_affected() == 1,
            "the position reduction did not affect any row: token={token_id} \
             strategy={strategy} size_um={size_um} sold_um={sold_um}"
        );
        Ok(from_micros(allocated_um))
    }

    /// Settles a resolved position in ONE transaction (same pattern as `apply_fill`):
    /// sets `outcome_won`/`winner_label`, records the REDEEM entry with its PnL and, if
    /// `redeem_tx` is given, marks `redeemed=1`. Previously these were 3 separate writes:
    /// a failure after the first left the position outside `list_expired_unresolved`
    /// (outcome_won was no longer NULL) and the REDEEM with its PnL was lost for good.
    /// (Hence the single transaction.)
    ///
    /// `settlement::orphan::poll` reads the position, makes an HTTP call to the CLOB, and
    /// only afterwards computes `payout`/`pnl` from that snapshot. If a late fill
    /// (e.g. the post-close sweep GTD) arrives in that window, the `pos`/`payout`/
    /// `pnl` this function receives may be stale. So the row is re-read INSIDE the
    /// transaction and payout/pnl are recalculated from the fresh size/
    /// cost_basis. The `payout`/`pnl` parameters remain only as a reference and to keep the
    /// signature compatible. The call site in src/settlement/orphan.rs:169 does not
    /// need to change. `AND outcome_won IS NULL` is also added to the UPDATE: if the
    /// position was already settled by another race (or revived by a later fill), this call
    /// is an idempotent no-op instead of rewriting or duplicating the REDEEM.
    /// Returns `true` if the settlement was applied, `false` if the row was already
    /// settled (idempotent no-op: the caller must NOT account the pnl in risk/metrics again).
    /// Settlement is therefore safe to retry.
    /// Only the first call writes the REDEEM.
    #[allow(clippy::too_many_arguments)]
    pub async fn apply_settlement(
        &self,
        pos: &Position,
        won: bool,
        winner_label: &str,
        payout: Decimal,
        pnl: Decimal,
        strategy_id: StrategyId,
        redeem_tx: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(self
            .apply_settlement_detail(pos, won, winner_label, payout, pnl, strategy_id, redeem_tx)
            .await?
            .is_some())
    }

    /// Same as `apply_settlement`, but returns the FRESH values that the
    /// transaction settled (re-read inside it)
    /// `None` if the position was already settled (idempotency). Whoever
    /// credits paper cash (`SimulatedRouter::credit_redemption`) uses
    /// these values to move EXACTLY what the ledger recorded, not
    /// those of the `pos` snapshot, which may be stale.
    pub async fn apply_settlement_detail(
        &self,
        pos: &Position,
        won: bool,
        winner_label: &str,
        payout: Decimal,
        pnl: Decimal,
        strategy_id: StrategyId,
        redeem_tx: Option<&str>,
    ) -> anyhow::Result<Option<AppliedSettlement>> {
        let _ = (payout, pnl); // recomputed below from a fresh read
        let mut tx = self.begin_immediate().await?;

        // Fresh re-read inside the transaction: uses the most recent size/cost_basis
        // instead of trusting `pos` (which may be stale). If the row no longer exists
        // (extreme case), it falls back to the snapshot passed by the caller.
        let fresh = Self::get_position_conn(&mut tx, &pos.token_id, &pos.strategy).await?;
        let (size, cost_basis) = match &fresh {
            Some(p) => (p.size, p.cost_basis),
            None => (pos.size, pos.cost_basis),
        };
        let payout = if won { size } else { Decimal::ZERO };
        let pnl = payout - cost_basis;

        // Settlement sets size/cost_basis to zero: the row remains as history (outcome,
        // labels, redeem_tx) but no longer represents inventory. The REDEEM above already
        // captured the fresh size/cost. A late SELL falls into the zero-cost branch
        // of `apply_fill` (proceeds with zero cost) instead of spending cost already
        // attributed.
        let updated = sqlx::query(
            "UPDATE positions SET outcome_won = ?, winner_label = ?,
                    size_um = 0, cost_basis_um = 0
             WHERE token_id = ? AND strategy = ? AND outcome_won IS NULL",
        )
        .bind(won as i64)
        .bind(winner_label)
        .bind(&pos.token_id)
        .bind(&pos.strategy)
        .execute(&mut *tx)
        .await?;

        if updated.rows_affected() == 0 {
            // Already settled by another race (or by another duplicate call to settlement::orphan::poll),
            // so the REDEEM and the pnl are not duplicated.
            tracing::warn!(token = %pos.token_id, strategy = %pos.strategy,
                "apply_settlement: position already settled, REDEEM not repeated (idempotent)");
            tx.commit().await?;
            return Ok(None);
        }

        Self::record_transaction_conn(
            &mut tx,
            // `crate::time`: the REDEEM feeds the economic ledger; in replay
            // it must be the timestamp of the closing event, not the wall clock.
            crate::time::now().timestamp_millis(),
            "REDEEM",
            &pos.market_cid,
            &pos.token_id,
            Some(winner_label),
            Some(winner_label),
            size,
            if won { Decimal::ONE } else { Decimal::ZERO },
            payout,
            Some(pnl),
            strategy_id,
        )
        .await?;
        if let Some(rtx) = redeem_tx {
            sqlx::query("UPDATE positions SET redeemed = 1, redeem_tx = ? WHERE token_id = ? AND strategy = ?")
                .bind(rtx)
                .bind(&pos.token_id)
                .bind(&pos.strategy)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        Ok(Some(AppliedSettlement {
            size,
            cost_basis,
            payout,
            pnl,
        }))
    }

    /// Expired unresolved positions (clear_at has passed, outcome_won still NULL).
    pub async fn list_expired_unresolved(&self) -> anyhow::Result<Vec<Position>> {
        // `crate::time`: the `clear_at < now` filter is a reconciliation decision;
        // in replay it uses the tape clock.
        let now = crate::time::now().timestamp();
        let rows = sqlx::query(
            "SELECT token_id, market_cid, strategy, size_um, avg_price_um, cost_basis_um,
                    opened_ts_ms, clear_at, neg_risk, bail_sold, outcome_won, winner_label,
                    outcome_label, redeemed, redeem_tx
             FROM positions
             WHERE bail_sold = 0 AND redeemed = 0 AND outcome_won IS NULL AND clear_at > 0 AND clear_at < ?",
        )
        .bind(now)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|r| Self::row_to_position(&r))
            .collect()
    }

    fn row_to_position(r: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Position> {
        use sqlx::Row;
        Ok(Position {
            token_id: r.try_get("token_id")?,
            market_cid: r.try_get("market_cid")?,
            slug: r.try_get("slug").unwrap_or_default(),
            strategy: r.try_get("strategy")?,
            size: from_micros(r.try_get("size_um")?),
            avg_price: from_micros(r.try_get("avg_price_um")?),
            cost_basis: from_micros(r.try_get("cost_basis_um")?),
            opened_ts_ms: r.try_get("opened_ts_ms")?,
            clear_at: r.try_get("clear_at")?,
            neg_risk: r.try_get::<i64, _>("neg_risk")? != 0,
            bail_sold: r.try_get::<i64, _>("bail_sold")? != 0,
            outcome_won: r.try_get::<Option<i64>, _>("outcome_won")?.map(|v| v != 0),
            winner_label: r.try_get("winner_label")?,
            outcome_label: r.try_get("outcome_label").ok().flatten(),
            redeemed: r.try_get::<i64, _>("redeemed")? != 0,
            redeem_tx: r.try_get("redeem_tx").ok().flatten(),
        })
    }

    /// Sum of cost_basis of open positions (not bail_sold, not redeemed, outcome unknown).
    /// USDC exposure of open positions in markets still active (clear_at > now_ts).
    /// Closed markets do not count: those positions can no longer be sold or modified.
    /// Returns true if there is any open position in the bracket `market_cid`.
    /// Used in startup reconciliation to decide whether a GTD order on the CLOB
    /// has an associated position or is stale (from an earlier session).
    pub async fn has_open_position_in_market(&self, market_cid: &str) -> anyhow::Result<bool> {
        let row = sqlx::query(
            "SELECT COUNT(*) AS n FROM positions
             WHERE market_cid = ? AND redeemed = 0 AND bail_sold = 0 AND outcome_won IS NULL",
        )
        .bind(market_cid)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.try_get::<i64, _>("n")? > 0)
    }

    /// Sum of `cost_basis` of OPEN positions WITH INVENTORY:
    /// `outcome_won IS NULL` + `size_um > 0` (after `apply_settlement` the row
    /// keeps size/cost at 0, but the explicit condition aligns the query with the doc:
    /// "cost still open" requires live shares).
    pub async fn total_open_cost(&self) -> anyhow::Result<Decimal> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(cost_basis_um), 0) AS total
             FROM positions WHERE redeemed = 0 AND bail_sold = 0 AND outcome_won IS NULL AND size_um > 0",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(from_micros(row.try_get::<i64, _>("total")?))
    }

    // ── Fills ─────────────────────────────────────────────────────────────────

    /// Inserts the raw fill record within the transaction of
    /// `apply_fill`. Returns `false` if it is a duplicate (inserts nothing).
    ///
    /// Dedup key = `exec_id` (the id of the EXECUTION: trade UUID in live,
    /// `sim-exec-N` in paper). A GTD/FAK order can produce several partial executions that
    /// share `order_id` but carry different `exec_id`s: each one must be accounted for.
    /// (So the dedup key is per execution, not per order.)
    ///
    /// Legacy fallback: rows written before this change stored the execution identity in
    /// `order_id` (exec_id = NULL). A fill that reappears after the migration with
    /// `exec_id = X` must match the row with `order_id = X`, so the query covers both cases:
    /// (the existing row is the older one).
    ///
    /// ```sql
    /// exec_id = :k OR (exec_id IS NULL AND order_id = :k)
    /// ```
    ///
    /// with `k = exec_id.or(order_id)` of the incoming fill. Fills that have neither of
    /// the two identities (e.g. synthetic test fills) are not deduplicated.
    /// The `INSERT OR IGNORE` is kept as a safety net against a race with the unique index
    /// `idx_fills_exec_id`.
    async fn record_fill_conn(
        conn: &mut sqlx::SqliteConnection,
        fill: &Fill,
    ) -> anyhow::Result<bool> {
        let side = match fill.side {
            FillSide::Buy => "BUY",
            FillSide::Sell => "SELL",
            FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected => {
                unreachable!("Rejected/Cancelled/GateRejected are never persisted")
            }
        };
        let dedup_key = fill.exec_id.as_deref().or(fill.order_id.as_deref());
        if let Some(key) = dedup_key {
            let exists: Option<i64> = sqlx::query_scalar(
                "SELECT id FROM fills
                 WHERE exec_id = ? OR (exec_id IS NULL AND order_id = ?)
                 LIMIT 1",
            )
            .bind(key)
            .bind(key)
            .fetch_optional(&mut *conn)
            .await?;
            if exists.is_some() {
                return Ok(false);
            }
            // Own maker legs of the same trade carry the composite exec_id `trade_id:order_id`
            // (they used to share the bare trade_id, so every leg after the first was dropped
            // on collision). For rows written before this fix: if a row exists with
            // `exec_id = trade_id` (the old format) and the SAME order_id, the leg was already
            // recorded
            // and is deduplicated.
            if let Some((base_tid, _)) = key.split_once(':') {
                if let Some(oid) = fill.order_id.as_deref() {
                    let legacy: Option<i64> = sqlx::query_scalar(
                        "SELECT id FROM fills WHERE exec_id = ? AND order_id = ? LIMIT 1",
                    )
                    .bind(base_tid)
                    .bind(oid)
                    .fetch_optional(&mut *conn)
                    .await?;
                    if legacy.is_some() {
                        return Ok(false);
                    }
                }
            }
        }
        let r = sqlx::query(
            "INSERT OR IGNORE INTO fills
             (ts_ms, market_cid, token_id, side, price_um, size_um, order_id,
              exec_id, local_order_id, hedge_id, neg_risk, strategy)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(fill.ts.timestamp_millis())
        .bind(&fill.market_cid)
        .bind(&fill.token_id)
        .bind(side)
        .bind(to_micros(fill.price))
        .bind(to_micros(fill.size))
        .bind(&fill.order_id)
        .bind(&fill.exec_id)
        .bind(&fill.local_order_id)
        .bind(fill.hedge_id)
        .bind(fill.neg_risk as i64)
        .bind(fill.strategy.to_string())
        .execute(conn)
        .await?;
        Ok(r.rows_affected() > 0)
    }

    /// Applies a complete fill in ONE transaction: record in `fills`
    /// (dedup by `exec_id`, with the legacy `order_id` fallback; see
    /// `record_fill_conn`), opening/averaging/reduction of the position, and the
    /// entry in `transactions`. Everything is persisted or nothing is: a crash halfway
    /// no longer leaves a fill without its position.
    ///
    /// The PnL and avg_price of the Sell are returned so that the caller can apply
    /// the NON-transactional effects (ledger release, risk.record_pnl).
    pub async fn apply_fill(&self, fill: &Fill) -> anyhow::Result<AppliedFill> {
        let _t = MetricTimer::new("hrf_sqlite_query_seconds", "apply_fill");
        let mut tx = self.begin_immediate().await?;

        if !Self::record_fill_conn(&mut tx, fill).await? {
            tx.commit().await?;
            return Ok(AppliedFill::Duplicate);
        }

        let usdc = fill.price * fill.size;
        let strategy_str = fill.strategy.to_string();

        let applied = match fill.side {
            FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected => {
                unreachable!("Rejected/Cancelled/GateRejected are filtered out before apply_fill")
            }
            FillSide::Buy => {
                let outcome_label = match Self::get_position_conn(
                    &mut tx,
                    &fill.token_id,
                    &strategy_str,
                )
                .await?
                {
                    // Critical: `get_position_conn` looks up by token_id+strategy
                    // WITHOUT filtering bail_sold/redeemed/outcome_won. If the row found
                    // is already CLOSED (full sell after a bail, or settled/redeemed), it is
                    // NOT "the same position" to average into: it is a legitimate re-entry
                    // (NearExpiry sells everything after a bail and allows re-entering the same
                    // token, see near_expiry.rs; MarketMaking/ThetaCapture have the same
                    // buy → full sell → rebuy cycle). Averaging into the dead row would revive
                    // bail_sold=1/redeemed=1/outcome_won inherited via `..existing`,
                    // leaving the position invisible to exposure/redemption/UI while it
                    // could still be sold.
                    Some(existing)
                        if existing.bail_sold
                            || existing.redeemed
                            || existing.outcome_won.is_some() =>
                    {
                        tracing::info!(
                            token = %fill.token_id, strategy = %strategy_str,
                            bail_sold = existing.bail_sold, redeemed = existing.redeemed,
                            outcome_won = ?existing.outcome_won,
                            "apply_fill: re-entry over a previously closed position, treated as NEW"
                        );
                        let label = match &fill.outcome_label {
                            Some(l) => Some(l.clone()),
                            None => existing.outcome_label.clone().or(Self::outcome_label_conn(
                                &mut tx,
                                &fill.token_id,
                                &fill.market_cid,
                            )
                            .await),
                        };
                        Self::open_position_conn(
                            &mut tx,
                            &Position {
                                token_id: fill.token_id.clone(),
                                market_cid: fill.market_cid.clone(),
                                slug: existing.slug,
                                strategy: strategy_str.clone(),
                                size: fill.size,
                                avg_price: fill.price,
                                cost_basis: usdc,
                                opened_ts_ms: fill.ts.timestamp_millis(),
                                clear_at: fill.clear_at,
                                neg_risk: fill.neg_risk,
                                bail_sold: false,
                                outcome_won: None,
                                winner_label: None,
                                outcome_label: label.clone(),
                                redeemed: false,
                                redeem_tx: None,
                            },
                        )
                        .await?;
                        label
                    }
                    Some(existing) => {
                        // Repairs clear_at/neg_risk/outcome_label if the row was left
                        // broken by an earlier fill (e.g. the fallback to clear_at=0 in paper)
                        // and the new fill carries a valid value.
                        // Historical records are not migrated: it only repairs going forward.
                        let label = existing
                            .outcome_label
                            .clone()
                            .or_else(|| fill.outcome_label.clone());
                        let clear_at = if existing.clear_at == 0 && fill.clear_at != 0 {
                            fill.clear_at
                        } else {
                            existing.clear_at
                        };
                        let neg_risk = existing.neg_risk || fill.neg_risk;
                        let new_size = existing.size + fill.size;
                        let new_cost = existing.cost_basis + usdc;
                        let new_avg = if new_size > Decimal::ZERO {
                            new_cost / new_size
                        } else {
                            fill.price
                        };
                        Self::open_position_conn(
                            &mut tx,
                            &Position {
                                size: new_size,
                                avg_price: new_avg,
                                cost_basis: new_cost,
                                clear_at,
                                neg_risk,
                                outcome_label: label.clone(),
                                ..existing
                            },
                        )
                        .await?;
                        label
                    }
                    None => {
                        // Primary: the label from the registry embedded in the fill by user_ws/sim_router.
                        // DB fallback: covers the scenario where the market is still in markets_observed.
                        let label = match &fill.outcome_label {
                            Some(l) => Some(l.clone()),
                            None => {
                                Self::outcome_label_conn(&mut tx, &fill.token_id, &fill.market_cid)
                                    .await
                            }
                        };
                        Self::open_position_conn(
                            &mut tx,
                            &Position {
                                token_id: fill.token_id.clone(),
                                market_cid: fill.market_cid.clone(),
                                slug: String::new(),
                                strategy: strategy_str.clone(),
                                size: fill.size,
                                avg_price: fill.price,
                                cost_basis: usdc,
                                opened_ts_ms: fill.ts.timestamp_millis(),
                                clear_at: fill.clear_at,
                                neg_risk: fill.neg_risk,
                                bail_sold: false,
                                outcome_won: None,
                                winner_label: None,
                                outcome_label: label.clone(),
                                redeemed: false,
                                redeem_tx: None,
                            },
                        )
                        .await?;
                        label
                    }
                };
                Self::record_transaction_conn(
                    &mut tx,
                    fill.ts.timestamp_millis(),
                    "BUY",
                    &fill.market_cid,
                    &fill.token_id,
                    outcome_label.as_deref(),
                    None,
                    fill.size,
                    fill.price,
                    usdc,
                    None,
                    fill.strategy,
                )
                .await?;
                AppliedFill::Buy
            }
            FillSide::Sell => {
                match Self::get_position_conn(&mut tx, &fill.token_id, &strategy_str).await? {
                    None => AppliedFill::SellWithoutPosition,
                    // Mirror arm of the BUY branch's re-entry handling. A SELL
                    // can arrive AFTER settlement (a late post-close sweep GTD, or the orphan checker
                    // getting ahead of an in-flight fill). `apply_settlement` already sets
                    // size_um/cost_basis_um to zero (root fix), but without this arm the late
                    // sell would fall into `SellWithoutPosition` and the proceeds (real money
                    // received) would be lost. They are booked as a SELL with ZERO allocated
                    // cost, so the PnL of this sell is the full proceeds and `Σ pnl == Σ flows`
                    // still holds.
                    // Legacy rows settled with residual cost were migrated separately.
                    // Those are handled by that migration, not by this arm.
                    //
                    // This arm goes BEFORE the `size == 0` one: with settlement already setting
                    // the row to zero, the opposite order would leave it dead and the proceeds would
                    // be lost again through
                    // `SellWithoutPosition`.
                    Some(pos) if pos.outcome_won.is_some() || pos.redeemed => {
                        // Same oversell clamp as the sibling arm;
                        // otherwise the proceeds and PnL would inflate the daily-loss-stop
                        // and the entry would sell more shares than the row held. The
                        // row only clamps if it still has a size: an already emptied position
                        // cannot contradict the fill's size.
                        let sold = if pos.size > Decimal::ZERO {
                            fill.size.min(pos.size)
                        } else {
                            fill.size
                        };
                        if sold < fill.size {
                            tracing::warn!(token = %fill.token_id, fill_size = %fill.size,
                                pos_size = %pos.size, "apply_fill: sell larger than the position, clamped");
                        }
                        let proceeds = from_micros(to_micros(fill.price * sold));
                        tracing::warn!(
                            token = %fill.token_id, strategy = %strategy_str,
                            redeemed = pos.redeemed, outcome_won = ?pos.outcome_won,
                            proceeds = %proceeds,
                            "apply_fill: sell over an already settled position; cost was already booked to the REDEEM, only the proceeds are recorded"
                        );
                        Self::record_transaction_conn(
                            &mut tx,
                            fill.ts.timestamp_millis(),
                            "SELL",
                            &fill.market_cid,
                            &fill.token_id,
                            None,
                            None,
                            sold,
                            fill.price,
                            proceeds,
                            Some(proceeds),
                            fill.strategy,
                        )
                        .await?;
                        AppliedFill::Sell {
                            pnl: proceeds,
                            cost_released: Decimal::ZERO,
                        }
                    }
                    Some(pos) if pos.size == Decimal::ZERO => AppliedFill::SellWithoutPosition,
                    Some(pos) => {
                        // Oversell (fill.size > position size in DB = drift): clamp to the
                        // known size; otherwise the PnL would inflate the daily-loss-stop and the
                        // ledger would release more exposure than it claimed.
                        let sold = fill.size.min(pos.size);
                        if sold < fill.size {
                            tracing::warn!(token = %fill.token_id, fill_size = %fill.size,
                                pos_size = %pos.size, "apply_fill: sell larger than the position, clamped");
                        }
                        let allocated_cost = Self::reduce_position_conn(
                            &mut tx,
                            &fill.token_id,
                            &strategy_str,
                            sold,
                        )
                        .await?;
                        // Reconcile with the amount actually persisted, not with
                        // avg_price (truncated) nor the sub-micro fractions of the fill.
                        let proceeds = from_micros(to_micros(fill.price * sold));
                        let pnl = proceeds - allocated_cost;
                        Self::record_transaction_conn(
                            &mut tx,
                            fill.ts.timestamp_millis(),
                            "SELL",
                            &fill.market_cid,
                            &fill.token_id,
                            None,
                            None,
                            sold,
                            fill.price,
                            proceeds,
                            Some(pnl),
                            fill.strategy,
                        )
                        .await?;
                        AppliedFill::Sell {
                            pnl,
                            cost_released: allocated_cost,
                        }
                    }
                }
            }
        };

        tx.commit().await?;
        Ok(applied)
    }

    // ── Transactions ─────────────────────────────────────────────────────────

    /// Server-side cap per page of /ui/history (avoids huge payloads
    /// if someone asks for an arbitrary limit).
    const MAX_HISTORY_PAGE: u32 = 2000;

    /// `ts_ms` is an explicit parameter (it used to be stamped unconditionally with
    /// `Utc::now()`). Fill callers pass `fill.ts`, as `record_fill_conn` does, so that a
    /// `backfill_missed_trades` after downtime does not leave an artificial step in
    /// `pnl_series()`, with the transactions stamped at the startup instant. The REDEEM of
    /// `apply_settlement` uses `crate::time::now()` (there is no associated fill): in
    /// production that is wall time; in replay it is the virtual timestamp of the closing
    /// event, since the engine's digest compares it byte by byte.
    /// (The digest is therefore deterministic in replay.)
    #[allow(clippy::too_many_arguments)]
    async fn record_transaction_conn(
        conn: &mut sqlx::SqliteConnection,
        ts_ms: i64,
        kind: &str,
        market_cid: &str,
        token_id: &str,
        outcome_label: Option<&str>,
        winner_label: Option<&str>,
        shares: Decimal,
        price: Decimal,
        usdc: Decimal,
        pnl: Option<Decimal>,
        strategy: StrategyId,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO transactions
             (ts_ms, kind, market_cid, token_id, outcome_label, winner_label,
              shares_um, price_um, usdc_um, pnl_um, strategy)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(ts_ms)
        .bind(kind)
        .bind(market_cid)
        .bind(token_id)
        .bind(outcome_label)
        .bind(winner_label)
        .bind(to_micros(shares))
        .bind(to_micros(price))
        .bind(to_micros(usdc))
        .bind(pnl.map(to_micros))
        .bind(strategy.to_string())
        .execute(conn)
        .await?;
        Ok(())
    }

    /// Accumulated realized PnL per strategy (only transactions of kind SELL/REDEEM with pnl_um).
    pub async fn realized_pnl_by_strategy(
        &self,
    ) -> anyhow::Result<std::collections::HashMap<String, Decimal>> {
        let rows = sqlx::query(
            "SELECT strategy, COALESCE(SUM(pnl_um), 0) AS total
             FROM transactions WHERE kind IN ('SELL', 'REDEEM') AND pnl_um IS NOT NULL
             GROUP BY strategy",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut map = std::collections::HashMap::new();
        for r in rows {
            let strat: String = r.try_get("strategy")?;
            let total: i64 = r.try_get("total")?;
            map.insert(strat, from_micros(total));
        }
        Ok(map)
    }

    /// Economic flow aggregates of the durable ledger: the basis of the
    /// `returns ↔ ledger` reconciliation (the sim/wallet cash must move
    /// EXACTLY by `sells + redeem − buys`, and the realized PnL must equal
    /// `returns + open_cost`).
    ///
    /// `open_cost` is the `cost_basis` of the positions with size > 0:
    /// the cost not yet realized. `open_cost_drift()` measures the
    /// cost conservation to the micro: `buys − removed − open == 0`.
    /// Additive: a new read, without touching existing queries or signatures.
    pub async fn economic_flows(&self) -> anyhow::Result<EconomicFlows> {
        let rows = sqlx::query(
            "SELECT kind,
                    COALESCE(SUM(usdc_um), 0) AS usdc,
                    COALESCE(SUM(pnl_um), 0)  AS pnl,
                    COUNT(*)                  AS n
             FROM transactions
             GROUP BY kind",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut f = EconomicFlows {
            open_cost: self.total_open_cost().await?,
            ..Default::default()
        };
        for r in rows {
            let kind: String = r.try_get("kind")?;
            let usdc = from_micros(r.try_get::<i64, _>("usdc")?);
            let pnl = from_micros(r.try_get::<i64, _>("pnl")?);
            let n: i64 = r.try_get("n")?;
            match kind.as_str() {
                "BUY" => {
                    f.buys_usdc = usdc;
                    f.n_buys = n;
                }
                "SELL" => {
                    f.sells_usdc = usdc;
                    f.sells_pnl = pnl;
                    f.n_sells = n;
                }
                "REDEEM" => {
                    f.redeem_usdc = usdc;
                    f.redeem_pnl = pnl;
                    f.n_redeem = n;
                }
                // Future/unknown kinds stay VISIBLE in `other`, not
                // merged into the trading figures.
                _ => {
                    f.other_usdc += usdc;
                    f.n_other += n;
                }
            }
        }
        Ok(f)
    }

    /// Per-strategy statistics over the WHOLE history (not only the
    /// last 200 transactions of `recent_transactions`). The Metrics tab used to compute them
    /// in the browser from /ui/history, and with more than 200 trades its numbers diverged from
    /// the Dashboard (which aggregates in SQL).
    /// Returns (strategy, n, pnl, wins, won, lost, best, worst).
    #[allow(clippy::type_complexity)]
    pub async fn strategy_stats(
        &self,
    ) -> anyhow::Result<
        Vec<(
            String,
            i64,
            Decimal,
            i64,
            Decimal,
            Decimal,
            Decimal,
            Decimal,
        )>,
    > {
        let rows = sqlx::query(
            "SELECT strategy,
                    COUNT(*)                                            AS n,
                    COALESCE(SUM(pnl_um), 0)                            AS pnl,
                    COALESCE(SUM(pnl_um > 0), 0)                        AS wins,
                    COALESCE(SUM(CASE WHEN pnl_um > 0 THEN pnl_um END), 0) AS won,
                    COALESCE(SUM(CASE WHEN pnl_um < 0 THEN pnl_um END), 0) AS lost,
                    COALESCE(MAX(pnl_um), 0)                            AS best,
                    COALESCE(MIN(pnl_um), 0)                            AS worst
             FROM transactions
             WHERE kind IN ('SELL', 'REDEEM') AND pnl_um IS NOT NULL
             GROUP BY strategy",
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                Ok((
                    r.try_get::<String, _>("strategy")?,
                    r.try_get::<i64, _>("n")?,
                    from_micros(r.try_get::<i64, _>("pnl")?),
                    r.try_get::<i64, _>("wins")?,
                    from_micros(r.try_get::<i64, _>("won")?),
                    from_micros(r.try_get::<i64, _>("lost")?),
                    from_micros(r.try_get::<i64, _>("best")?),
                    from_micros(r.try_get::<i64, _>("worst")?),
                ))
            })
            .collect()
    }

    /// Expectancy/edge statistics per strategy over the WHOLE history of
    /// `transactions`: the economic evaluation piece needed to decide
    /// whether to trade live (idea from Freqtrade's Edge module: win
    /// rate, profit factor, expectancy per trade and return over capital).
    ///
    /// ## Definition of a "closed trade"
    ///
    /// The DB does not store a position id per transaction: the natural key
    /// of a position is `(token_id, strategy)` (the PK of `positions`). Trades are rebuilt
    /// by INVENTORY, event by event, never by a GROUP BY over persistent flags.
    /// For each group `(strategy, token_id)` the transactions are walked in order
    /// `(ts_ms, id)`. A BUY opens or extends the trade, and it closes when the inventory
    /// returns to 0: either the SELLs consumed the whole size, or a REDEEM arrived (which
    /// always settles the whole position:
    /// `apply_settlement` sets `size_um`/`cost_basis_um` to 0).
    /// (See the approximations below.)
    ///
    /// Approximations, stated openly:
    /// - Partial sells are aggregated into the SAME trade (their pnl summed), and a
    ///   repurchase after a full close opens a NEW trade, even though the
    ///   `positions` row is reused (the buy → full sell → rebuy cycle of
    ///   MM/ThetaCapture/NearExpiry).
    /// - A SELL/REDEEM with the inventory already at 0 (the late sell
    ///   after settlement, from the zero-cost branch of `apply_fill`, or a fill
    ///   whose BUY was never persisted) counts as a closed trade with cost
    ///   0: its PnL is truly realized and belongs to the strategy.
    /// - A trade still open at the end of the history does NOT count as
    ///   closed (it goes to `open_trades`): its PnL is not realized.
    /// - `total_cost_deployed` sums only the BUYs of the closed trades,
    ///   so that `expectancy_pct` compares PnL with the capital that
    ///   generated it; `max_exposure` covers the peak including what is still open.
    ///
    /// Cost: two full scans of `transactions` (the same table that
    /// `pnl_series`/`strategy_stats` already scan in full). It is an on-demand
    /// diagnostic endpoint, not a hot path.
    pub async fn strategy_edge_stats(
        &self,
    ) -> anyhow::Result<std::collections::HashMap<String, StrategyEdgeStats>> {
        let _t = MetricTimer::new("hrf_sqlite_query_seconds", "edge_stats");

        // Pass 1: rebuilds round-trips per inventory, grouped
        // by (strategy, token_id), the real identity of the position. `id`
        // breaks ties between rows with the same ts_ms, in insertion order.
        let rows = sqlx::query(
            "SELECT strategy, token_id, kind, shares_um, usdc_um, pnl_um
             FROM transactions
             ORDER BY strategy, token_id, ts_ms, id",
        )
        .fetch_all(&self.pool)
        .await?;

        #[derive(Default)]
        struct Acc {
            closed_trades: i64,
            open_trades: i64,
            wins: i64,
            losses: i64,
            flat_trades: i64,
            gross_win_um: i64,
            gross_loss_um: i64,
            total_pnl_um: i64,
            total_cost_um: i64,
        }

        let mut accs: std::collections::HashMap<String, Acc> = std::collections::HashMap::new();

        // State of the round-trip in progress within the current group.
        let mut cur_key: Option<(String, String)> = None;
        let mut open_shares_um: i64 = 0;
        let mut trip_pnl_um: i64 = 0;
        let mut trip_cost_um: i64 = 0;
        let mut trip_has_tx = false;

        for r in &rows {
            let strategy: String = r.try_get("strategy")?;
            let token_id: String = r.try_get("token_id")?;
            let kind: String = r.try_get("kind")?;
            let shares_um: i64 = r.try_get("shares_um")?;
            let usdc_um: i64 = r.try_get("usdc_um")?;
            let pnl_um: Option<i64> = r.try_get("pnl_um")?;

            let key = (strategy.clone(), token_id);
            if cur_key.as_ref() != Some(&key) {
                // Group change: the trade that was left open counts as open.
                if let Some((s, _)) = cur_key.take() {
                    if open_shares_um > 0 || trip_has_tx {
                        accs.entry(s).or_default().open_trades += 1;
                    }
                }
                open_shares_um = 0;
                trip_pnl_um = 0;
                trip_cost_um = 0;
                trip_has_tx = false;
                cur_key = Some(key);
            }

            match kind.as_str() {
                "BUY" => {
                    open_shares_um = open_shares_um.saturating_add(shares_um);
                    trip_cost_um = trip_cost_um.saturating_add(usdc_um);
                    trip_has_tx = true;
                }
                "SELL" | "REDEEM" => {
                    trip_pnl_um = trip_pnl_um.saturating_add(pnl_um.unwrap_or(0));
                    trip_has_tx = true;
                    if kind == "REDEEM" {
                        // `apply_settlement` always settles the whole position.
                        open_shares_um = 0;
                    } else {
                        // saturating: an oversell already clamped by apply_fill
                        // (or an orphan fill) cannot drive the inventory negative:
                        // it stays at 0 and closes the trade.
                        open_shares_um = open_shares_um.saturating_sub(shares_um);
                    }
                    if open_shares_um == 0 {
                        // Trade closed. With a previous inventory of 0 this is the
                        // standalone zero-cost trade documented above.
                        let a = accs.entry(strategy).or_default();
                        a.closed_trades += 1;
                        a.total_pnl_um += trip_pnl_um;
                        a.total_cost_um += trip_cost_um;
                        if trip_pnl_um > 0 {
                            a.wins += 1;
                            a.gross_win_um += trip_pnl_um;
                        } else if trip_pnl_um < 0 {
                            a.losses += 1;
                            a.gross_loss_um += trip_pnl_um;
                        } else {
                            a.flat_trades += 1;
                        }
                        trip_pnl_um = 0;
                        trip_cost_um = 0;
                        trip_has_tx = false;
                    }
                }
                _ => {}
            }
        }
        // The last group of the iterator may also leave a trade open.
        if let Some((s, _)) = cur_key {
            if open_shares_um > 0 || trip_has_tx {
                accs.entry(s).or_default().open_trades += 1;
            }
        }

        // Pass 2: peak live cost per strategy, in global chronological order
        // (pass 1 is ordered by token, not by time).
        // BUY sum `usdc_um`; SELL/REDEEM releases `usdc_um − pnl_um`, which is
        // exactly the persisted allocated cost: for a normal SELL it is
        // `allocated_cost`, for a losing REDEEM it is the whole cost
        // (usdc=0, pnl=−cost), and for a late sell after settlement it is 0
        // (pnl=proceeds; the REDEEM already booked the cost). `pnl_um` NULL on a close is
        // treated as "cost already booked" (releases 0): conservative for a maximum.
        // The running total is clamped to ≥0 so that a corrupt row does not drag the
        // running value negative.
        let exp_rows = sqlx::query(
            "SELECT strategy, kind, usdc_um, pnl_um
             FROM transactions ORDER BY strategy, ts_ms, id",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut running: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut peak: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        for r in &exp_rows {
            let strategy: String = r.try_get("strategy")?;
            let kind: String = r.try_get("kind")?;
            let usdc_um: i64 = r.try_get("usdc_um")?;
            let pnl_um: Option<i64> = r.try_get("pnl_um")?;

            let run = running.entry(strategy.clone()).or_insert(0);
            match kind.as_str() {
                "BUY" => *run = run.saturating_add(usdc_um),
                "SELL" | "REDEEM" => {
                    let released = usdc_um.saturating_sub(pnl_um.unwrap_or(usdc_um));
                    *run = (*run - released.max(0)).max(0);
                }
                _ => {}
            }
            let p = peak.entry(strategy).or_insert(0);
            if *run > *p {
                *p = *run;
            }
        }

        let mut out = std::collections::HashMap::new();
        for (strategy, a) in accs {
            let total_pnl = from_micros(a.total_pnl_um);
            let total_cost = from_micros(a.total_cost_um);
            let gross_win = from_micros(a.gross_win_um);
            let gross_loss = from_micros(a.gross_loss_um); // negative
            let closed = a.closed_trades;
            let max_exposure = from_micros(*peak.get(&strategy).unwrap_or(&0));
            out.insert(
                strategy.clone(),
                StrategyEdgeStats {
                    closed_trades: closed,
                    open_trades: a.open_trades,
                    wins: a.wins,
                    losses: a.losses,
                    flat_trades: a.flat_trades,
                    win_rate: (closed > 0).then(|| Decimal::from(a.wins) / Decimal::from(closed)),
                    avg_win: (a.wins > 0).then(|| gross_win / Decimal::from(a.wins)),
                    avg_loss: (a.losses > 0).then(|| gross_loss / Decimal::from(a.losses)),
                    profit_factor: (gross_loss < Decimal::ZERO)
                        .then(|| gross_win / gross_loss.abs()),
                    expectancy_per_trade: (closed > 0).then(|| total_pnl / Decimal::from(closed)),
                    expectancy_pct: (total_cost > Decimal::ZERO).then(|| total_pnl / total_cost),
                    total_pnl,
                    total_cost_deployed: total_cost,
                    max_exposure,
                },
            );
        }
        Ok(out)
    }

    /// Total USDC volume traded (all transactions, BUY included).
    pub async fn total_volume_usdc(&self) -> anyhow::Result<Decimal> {
        let row = sqlx::query("SELECT COALESCE(SUM(usdc_um), 0) AS v FROM transactions")
            .fetch_one(&self.pool)
            .await?;
        Ok(from_micros(row.try_get::<i64, _>("v")?))
    }

    /// Full series (the whole history) of closed PnL for the Metrics curve,
    /// instead of being limited to the last 200 trades of `recent_transactions`.
    /// Vec ordered by ts_ms ascending; the handler serializes it in columnar form
    /// to keep the payload small.
    pub async fn pnl_series(&self) -> anyhow::Result<Vec<(i64, Decimal)>> {
        let rows = sqlx::query(
            "SELECT ts_ms, pnl_um FROM transactions
             WHERE kind IN ('SELL', 'REDEEM') AND pnl_um IS NOT NULL
             ORDER BY ts_ms ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                Ok((
                    r.try_get::<i64, _>("ts_ms")?,
                    from_micros(r.try_get::<i64, _>("pnl_um")?),
                ))
            })
            .collect()
    }

    /// Closed PnL distribution by UTC hour (0-23), summed over the whole
    /// history: same table and filter as `pnl_series()`. Only includes the
    /// hours with at least one closed transaction (does not fill in all 24 rows).
    pub async fn pnl_by_hour_of_day(&self) -> anyhow::Result<Vec<(u32, Decimal)>> {
        let rows = sqlx::query(
            "SELECT CAST(strftime('%H', ts_ms / 1000, 'unixepoch') AS INTEGER) AS hour, \
                    SUM(pnl_um) AS pnl \
             FROM transactions \
             WHERE kind IN ('SELL', 'REDEEM') AND pnl_um IS NOT NULL \
             GROUP BY hour ORDER BY hour",
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                Ok((
                    r.try_get::<i64, _>("hour")? as u32,
                    from_micros(r.try_get::<i64, _>("pnl")?),
                ))
            })
            .collect()
    }

    /// Stores a point of the equity series (cash + mark-to-market value of
    /// open positions); see the table's doc-comment in `migrate()`.
    pub async fn record_equity_snapshot(
        &self,
        ts_ms: i64,
        cash: Decimal,
        positions_value: Decimal,
        equity: Decimal,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO equity_snapshots (ts_ms, cash_um, positions_value_um, equity_um)
             VALUES (?, ?, ?, ?)",
        )
        .bind(ts_ms)
        .bind(to_micros(cash))
        .bind(to_micros(positions_value))
        .bind(to_micros(equity))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Full equity series for the Money tab's chart, in chronological order.
    /// Defensive cap at 5000 points (at 1 snapshot per 5 min that is ~17
    /// days, more than enough for any real calibration run; it prevents a forgotten
    /// instance running for months from inflating the payload without limit).
    pub async fn equity_series(&self) -> anyhow::Result<Vec<EquityPoint>> {
        // Each point carries the deposits accumulated up to its own
        // `ts_ms`, so that the consumer can plot `pnl` instead of
        // `equity`. A correlated subquery (not a JOIN with a window) because it also
        // corrects the history already recorded: no migration or re-collection is needed.
        // `deposits` has few rows (5 over 6 days of real running), so the cost over 5000
        // points is negligible.
        let rows = sqlx::query(
            "SELECT e.ts_ms, e.cash_um, e.positions_value_um, e.equity_um,
                    (SELECT COALESCE(SUM(d.amount_um), 0)
                       FROM deposits d WHERE d.ts_ms <= e.ts_ms) AS deposits_um
             FROM equity_snapshots e ORDER BY e.ts_ms ASC LIMIT 5000",
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                let equity = from_micros(r.try_get::<i64, _>("equity_um")?);
                let deposits = from_micros(r.try_get::<i64, _>("deposits_um")?);
                Ok(EquityPoint {
                    ts_ms: r.try_get::<i64, _>("ts_ms")?,
                    cash: from_micros(r.try_get::<i64, _>("cash_um")?),
                    positions_value: from_micros(r.try_get::<i64, _>("positions_value_um")?),
                    equity,
                    deposits,
                    // Identity: equity = deposits − open_cost + realized +
                    // positions_value, so equity − deposits = realized +
                    // unrealized. This is the true PnL, and it is immune to top-ups by
                    // construction (it starts at 0 and does not step).
                    pnl: equity - deposits,
                })
            })
            .collect()
    }

    /// Largest drawdown since a historical cumulative maximum, over the same series
    /// of closed PnL as `pnl_series()` (chronological order).
    pub async fn max_drawdown(&self) -> anyhow::Result<Decimal> {
        let points = self.pnl_series().await?;
        let mut cum = Decimal::ZERO;
        let mut peak = Decimal::ZERO;
        let mut max_dd = Decimal::ZERO;
        for (_, pnl) in points {
            cum += pnl;
            if cum > peak {
                peak = cum;
            }
            let dd = peak - cum;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        Ok(max_dd)
    }

    /// Total realized PnL.
    pub async fn realized_pnl_total(&self) -> anyhow::Result<Decimal> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(pnl_um), 0) AS total
             FROM transactions WHERE kind IN ('SELL', 'REDEEM') AND pnl_um IS NOT NULL",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(from_micros(row.try_get::<i64, _>("total")?))
    }

    /// Recent transactions for the UI, paginated by cursor.
    /// `before_id` (the id of the last row of the previous page) allows loading backwards
    /// without repeating or skipping rows; `limit` is clamped to
    /// `MAX_HISTORY_PAGE`.
    pub async fn recent_transactions(
        &self,
        limit: u32,
        before_id: Option<i64>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let limit = limit.clamp(1, Self::MAX_HISTORY_PAGE) as i64;
        let rows = sqlx::query(
            "SELECT t.id, t.ts_ms, t.kind, t.market_cid, t.token_id, t.outcome_label, t.winner_label,
                    t.shares_um, t.price_um, t.usdc_um, t.pnl_um, t.strategy,
                    COALESCE(m.slug, '') AS slug
             FROM transactions t
             LEFT JOIN markets_observed m ON m.condition_id = t.market_cid
             WHERE (?1 IS NULL OR t.id < ?1)
             ORDER BY t.id DESC LIMIT ?2",
        )
        .bind(before_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| {
                let pnl_um: Option<i64> = r.try_get("pnl_um").ok().flatten();
                serde_json::json!({
                    "id":            r.try_get::<i64, _>("id").unwrap_or(0),
                    "ts_ms":         r.try_get::<i64, _>("ts_ms").unwrap_or(0),
                    "kind":          r.try_get::<String, _>("kind").unwrap_or_default(),
                    "market_cid":    r.try_get::<String, _>("market_cid").unwrap_or_default(),
                    "token_id":      r.try_get::<String, _>("token_id").unwrap_or_default(),
                    "slug":          r.try_get::<String, _>("slug").unwrap_or_default(),
                    "outcome_label": r.try_get::<Option<String>, _>("outcome_label").ok().flatten(),
                    "winner_label":  r.try_get::<Option<String>, _>("winner_label").ok().flatten(),
                    "shares":        from_micros(r.try_get("shares_um").unwrap_or(0)).to_string(),
                    "price":         from_micros(r.try_get("price_um").unwrap_or(0)).to_string(),
                    "usdc":          from_micros(r.try_get("usdc_um").unwrap_or(0)).to_string(),
                    "pnl":           pnl_um.map(|v| from_micros(v).to_string()),
                    "strategy":      r.try_get::<String, _>("strategy").unwrap_or_default(),
                })
            })
            .collect())
    }

    // ── Deposits ──────────────────────────────────────────────────────────────

    pub async fn add_deposit(&self, amount: Decimal, note: &str) -> anyhow::Result<()> {
        sqlx::query("INSERT INTO deposits (amount_um, note, ts_ms) VALUES (?, ?, ?)")
            .bind(to_micros(amount))
            .bind(note)
            .bind(chrono::Utc::now().timestamp_millis())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn deposits_total(&self) -> anyhow::Result<Decimal> {
        let row = sqlx::query("SELECT COALESCE(SUM(amount_um), 0) AS total FROM deposits")
            .fetch_one(&self.pool)
            .await?;
        Ok(from_micros(row.try_get::<i64, _>("total")?))
    }

    /// Automatic top-ups (`main.rs::paper_topup`, balance < $100 → +$1000) vs.
    /// capital actually contributed (initial config + manual deposits). Without
    /// this distinction, `deposits_total()` grows mechanically every time the
    /// follower burns cash quickly, not because the user contributed more,
    /// and it inflates the denominator of any "% return", making it less and less
    /// representative of "how $1000 of real money performs". Returns
    /// (number of top-ups, total amount topped up).
    pub async fn topups_total(&self) -> anyhow::Result<(i64, Decimal)> {
        let row = sqlx::query(
            "SELECT COUNT(*) AS n, COALESCE(SUM(amount_um), 0) AS total FROM deposits WHERE note = ?",
        )
        .bind(TOPUP_NOTE)
        .fetch_one(&self.pool)
        .await?;
        Ok((
            row.try_get::<i64, _>("n")?,
            from_micros(row.try_get::<i64, _>("total")?),
        ))
    }

    // ── Observed markets ──────────────────────────────────────────────────────

    async fn outcome_label_conn(
        conn: &mut sqlx::SqliteConnection,
        token_id: &str,
        market_cid: &str,
    ) -> Option<String> {
        let yes_id: Option<String> =
            sqlx::query_scalar("SELECT yes_token_id FROM markets_observed WHERE condition_id = ?")
                .bind(market_cid)
                .fetch_optional(conn)
                .await
                .ok()
                .flatten();
        yes_id.map(|y| if token_id == y { "YES" } else { "NO" }.to_string())
    }

    pub async fn get_market_slug(&self, condition_id: &str) -> Option<String> {
        sqlx::query_scalar("SELECT slug FROM markets_observed WHERE condition_id = ?")
            .bind(condition_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten()
    }

    /// Real UPSERT (it used to be `INSERT OR IGNORE`: a `condition_id` already
    /// seen never refreshed `slug`/`yes_token_id`/`clear_at`/`neg_risk`).
    /// `outcome_label_conn` depends on a fresh `yes_token_id` and returns "NO" for
    /// any token that does not match, including one from another market if the
    /// stale record was left behind. `market_open_ts`/`first_seen_ts` are preserved from the
    /// first INSERT (there is no reason to refresh them on every re-observation).
    #[allow(clippy::too_many_arguments)]
    pub async fn upsert_market(
        &self,
        condition_id: &str,
        slug: &str,
        yes_token_id: &str,
        no_token_id: &str,
        market_open_ts: i64,
        clear_at: i64,
        neg_risk: bool,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO markets_observed
             (condition_id, slug, yes_token_id, no_token_id, market_open_ts, clear_at,
              neg_risk, first_seen_ts)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(condition_id) DO UPDATE SET
                 slug         = excluded.slug,
                 yes_token_id = excluded.yes_token_id,
                 no_token_id  = excluded.no_token_id,
                 clear_at     = excluded.clear_at,
                 neg_risk     = excluded.neg_risk",
        )
        .bind(condition_id)
        .bind(slug)
        .bind(yes_token_id)
        .bind(no_token_id)
        .bind(market_open_ts)
        .bind(clear_at)
        .bind(neg_risk as i64)
        .bind(chrono::Utc::now().timestamp())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── Blacklist ─────────────────────────────────────────────────────────────

    pub async fn add_to_blacklist(&self, condition_id: &str) -> anyhow::Result<()> {
        sqlx::query("INSERT OR IGNORE INTO blacklist (condition_id) VALUES (?)")
            .bind(condition_id)
            .execute(&self.pool)
            .await?;
        self.blacklist_cache.insert(condition_id.to_string());
        Ok(())
    }

    /// In-memory cache lookup: no SQLite query on the executor's hot path.
    pub async fn is_blacklisted(&self, condition_id: &str) -> anyhow::Result<bool> {
        Ok(self.blacklist_cache.contains(condition_id))
    }

    pub async fn blacklist(&self) -> anyhow::Result<Vec<String>> {
        let rows = sqlx::query("SELECT condition_id FROM blacklist")
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter()
            .map(|r| Ok(r.try_get("condition_id")?))
            .collect()
    }

    // ── History for the UI ──────────────────────────────────────────────────
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn to_from_micros_roundtrip() {
        let cases = [
            dec!(0),
            dec!(1),
            dec!(0.000001),
            dec!(99999.999999),
            dec!(-5.5),
        ];
        for c in cases {
            let rt = from_micros(to_micros(c));
            // Difference < 1 micro-USDC
            assert!((rt - c).abs() < dec!(0.000001), "roundtrip {c} → {rt}");
        }
    }

    #[tokio::test]
    async fn deposits_total_accumulates() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        store.add_deposit(dec!(100), "initial").await.unwrap();
        store.add_deposit(dec!(50), "extra").await.unwrap();
        let total = store.deposits_total().await.unwrap();
        assert_eq!(total, dec!(150));
    }

    /// Automatic top-ups from paper_topup must be distinguishable from the
    /// real capital (initial config + manual deposits). If the filter of
    /// `topups_total()` stops matching the literal note used by
    /// `main.rs::paper_topup` (they share the `TOPUP_NOTE` constant), this
    /// test catches it.
    #[tokio::test]
    async fn topups_total_distinguishes_reloads_from_real_capital() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        store
            .add_deposit(dec!(1000), "initial paper capital (config)")
            .await
            .unwrap();
        store.add_deposit(dec!(1000), TOPUP_NOTE).await.unwrap();
        store.add_deposit(dec!(1000), TOPUP_NOTE).await.unwrap();
        store
            .add_deposit(dec!(50), "manual deposit via API")
            .await
            .unwrap();

        let (count, sum) = store.topups_total().await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(sum, dec!(2000));
        assert_eq!(store.deposits_total().await.unwrap(), dec!(3050));
    }

    /// The equity chart showed the automatic paper top-ups (+$1000 each) as a
    /// vertical step, so a LOSING run was read as an ascending curve. Measured on the real run
    /// of 27-Aug → 3-Sep-2026: equity 1000 → 3795.92 with five deposits of 1000
    /// = −1,204.08 real, while the chart drew +280%.
    /// The top-ups were counted as profit.
    ///
    /// The series must expose `pnl = equity − cumulative deposits AT THAT
    /// INSTANT` (not the final total, which would retroactively shift the points
    /// before the first top-up).
    #[tokio::test]
    async fn equity_series_discounts_topups_and_does_not_stair_step() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();

        // `add_deposit` stamps with `Utc::now()` and the whole test fits in a single
        // millisecond, so the snapshots are anchored to the REAL ts of the last
        // deposit instead of `now()`; otherwise both points see the two deposits and the
        // test measures the clock, not the bug.
        store
            .add_deposit(dec!(1000), "initial paper capital (config)")
            .await
            .unwrap();
        let t1 = sqlx::query_scalar::<_, i64>("SELECT MAX(ts_ms) FROM deposits")
            .fetch_one(&store.pool)
            .await
            .unwrap()
            + 1;
        // Lost 100: equity 900 over 1000 contributed.
        store
            .record_equity_snapshot(t1, dec!(900), dec!(0), dec!(900))
            .await
            .unwrap();

        store.add_deposit(dec!(1000), TOPUP_NOTE).await.unwrap();
        // The top-up is stamped in the same ms as t1; we push it so that the 2nd
        // snapshot is unambiguously AFTER both deposits.
        sqlx::query("UPDATE deposits SET ts_ms = ? WHERE note = ?")
            .bind(t1 + 10)
            .bind(TOPUP_NOTE)
            .execute(&store.pool)
            .await
            .unwrap();
        let t2 = t1 + 11;
        // After the top-up, equity JUMPS to 1850, but it has only lost 50 more.
        store
            .record_equity_snapshot(t2, dec!(1850), dec!(0), dec!(1850))
            .await
            .unwrap();

        let series = store.equity_series().await.unwrap();
        assert_eq!(series.len(), 2);

        assert_eq!(series[0].deposits, dec!(1000));
        assert_eq!(series[0].pnl, dec!(-100));
        assert_eq!(
            series[1].deposits,
            dec!(2000),
            "the 2nd point sees BOTH deposits"
        );
        assert_eq!(series[1].pnl, dec!(-150));

        // The invariant that matters: gross equity rises +950 between the two
        // points (the top-up step), while the real PnL FALLS. A chart over
        // `equity` would say "winning"; over `pnl` it tells the truth.
        assert!(
            series[1].equity > series[0].equity,
            "gross equity steps upward"
        );
        assert!(
            series[1].pnl < series[0].pnl,
            "real PnL must keep falling despite the top-up"
        );
    }

    /// The PnL of the series must match `realized_pnl_total()` when there are no
    /// open positions: the same figure by two independent paths (the aggregate of
    /// `transactions` vs. equity − deposits). If someone touches one of the two
    /// without the other, this catches it.
    #[tokio::test]
    async fn series_pnl_matches_realized_pnl_total() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        store
            .add_deposit(dec!(5000), "initial paper capital (config)")
            .await
            .unwrap();

        // Final equity observed in the real run, with no open positions.
        let t = chrono::Utc::now().timestamp_millis();
        store
            .record_equity_snapshot(t, dec!(3795.92), dec!(0), dec!(3795.92))
            .await
            .unwrap();

        let series = store.equity_series().await.unwrap();
        assert_eq!(series.last().unwrap().pnl, dec!(-1204.08));
    }

    #[tokio::test]
    async fn mode_is_preserved() {
        let paper = StateStore::open_memory(Mode::Paper).await.unwrap();
        let live = StateStore::open_memory(Mode::Live).await.unwrap();
        assert_eq!(paper.mode, Mode::Paper);
        assert_eq!(live.mode, Mode::Live);
    }

    #[tokio::test]
    async fn recent_transactions_pages_by_cursor() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let mut conn = store.pool.acquire().await.unwrap();
        for i in 0..3 {
            StateStore::record_transaction_conn(
                &mut conn,
                1000,
                "BUY",
                "cid",
                &format!("tok-{i}"),
                None,
                None,
                dec!(10),
                dec!(0.5),
                dec!(5),
                None,
                StrategyId::Arb,
            )
            .await
            .unwrap();
        }
        // :memory: uses max_connections(1): release before recent_transactions
        // requests its own connection from the pool, or it blocks waiting for a free one.
        drop(conn);

        let page1 = store.recent_transactions(2, None).await.unwrap();
        assert_eq!(page1.len(), 2, "first page must return 2 rows");
        let cursor = page1[1]["id"].as_i64().expect("id present in the row");

        let page2 = store.recent_transactions(2, Some(cursor)).await.unwrap();
        assert_eq!(page2.len(), 1, "second page must return the remaining row");

        let ids1: Vec<i64> = page1.iter().map(|r| r["id"].as_i64().unwrap()).collect();
        let ids2: Vec<i64> = page2.iter().map(|r| r["id"].as_i64().unwrap()).collect();
        assert!(
            ids1.iter().all(|id| !ids2.contains(id)),
            "there must be no duplicate ids across pages"
        );
    }

    #[tokio::test]
    async fn recent_transactions_limit_is_clamped() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let mut conn = store.pool.acquire().await.unwrap();
        StateStore::record_transaction_conn(
            &mut conn,
            1000,
            "BUY",
            "cid",
            "tok",
            None,
            None,
            dec!(10),
            dec!(0.5),
            dec!(5),
            None,
            StrategyId::Arb,
        )
        .await
        .unwrap();
        // :memory: uses max_connections(1): release before recent_transactions
        // requests its own connection from the pool, or it blocks waiting for a free one.
        drop(conn);

        // An absurdly large limit must not break the query (it is clamped to MAX_HISTORY_PAGE).
        let rows = store.recent_transactions(999_999, None).await.unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[tokio::test]
    async fn pnl_series_returns_only_closed_sorted() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        // BUY without pnl: must not appear in the series.
        sqlx::query(
            "INSERT INTO transactions (ts_ms, kind, market_cid, token_id, shares_um, price_um, usdc_um, pnl_um, strategy)
             VALUES (1000, 'BUY', 'cid', 'tok', 10000000, 500000, 5000000, NULL, 'arb')",
        ).execute(&store.pool).await.unwrap();
        // Winning REDEEM, later ts.
        sqlx::query(
            "INSERT INTO transactions (ts_ms, kind, market_cid, token_id, shares_um, price_um, usdc_um, pnl_um, strategy)
             VALUES (3000, 'REDEEM', 'cid', 'tok', 10000000, 1000000, 10000000, 5000000, 'arb')",
        ).execute(&store.pool).await.unwrap();
        // Losing SELL, intermediate ts.
        sqlx::query(
            "INSERT INTO transactions (ts_ms, kind, market_cid, token_id, shares_um, price_um, usdc_um, pnl_um, strategy)
             VALUES (2000, 'SELL', 'cid', 'tok', 10000000, 300000, 3000000, -2000000, 'arb')",
        ).execute(&store.pool).await.unwrap();

        let series = store.pnl_series().await.unwrap();
        assert_eq!(series.len(), 2, "the BUY without pnl must be excluded");
        assert_eq!(series[0].0, 2000, "ascending order: SELL (ts=2000) first");
        assert_eq!(series[1].0, 3000, "ascending order: REDEEM (ts=3000) after");
        assert!(series[0].1 < Decimal::ZERO, "losing SELL: negative pnl");
        assert!(series[1].1 > Decimal::ZERO, "winning REDEEM: positive pnl");
    }

    // ── Test helpers for Fill ────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn test_fill(
        token_id: &str,
        strategy: StrategyId,
        side: FillSide,
        price: Decimal,
        size: Decimal,
        clear_at: i64,
        neg_risk: bool,
        outcome_label: Option<&str>,
        order_id: &str,
    ) -> Fill {
        Fill {
            market_cid: "cid-1".into(),
            token_id: token_id.into(),
            side,
            price,
            size,
            order_id: Some(order_id.into()),
            ts: chrono::Utc::now(),
            mode: Mode::Paper,
            neg_risk,
            strategy,
            clear_at,
            outcome_label: outcome_label.map(|s| s.to_string()),
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
        }
    }

    #[tokio::test]
    async fn accounting_buy_partial_sell_redeem_reconciles_cashflows() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let buy = test_fill(
            "accounting",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.44),
            dec!(10.989011),
            1000,
            false,
            None,
            "buy",
        );
        store.apply_fill(&buy).await.unwrap();
        let stale = store
            .get_position("accounting", "arb")
            .await
            .unwrap()
            .unwrap();
        let sell = test_fill(
            "accounting",
            StrategyId::Arb,
            FillSide::Sell,
            dec!(0.47),
            dec!(10.98),
            1000,
            false,
            None,
            "sell",
        );
        store.apply_fill(&sell).await.unwrap();
        store
            .apply_settlement(
                &stale,
                true,
                "YES",
                dec!(999),
                dec!(999),
                StrategyId::Arb,
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            store.realized_pnl_total().await.unwrap(),
            dec!(0.334446),
            "SELL + REDEEM - BUY, amounts persisted to the micro"
        );
        let remaining = store
            .get_position("accounting", "arb")
            .await
            .unwrap()
            .unwrap();
        // Settlement empties the row: the residual (0.009011 shares / 0.003965
        // of cost) was already paid by the REDEEM, so keeping it would double count it.
        // The row is kept as history without inventory state.
        assert_eq!(remaining.size, Decimal::ZERO);
        assert_eq!(remaining.cost_basis, Decimal::ZERO);
        assert_eq!(remaining.outcome_won, Some(true));
    }

    /// Conservation matrix: each SELL must consume exactly the cost that its PnL
    /// deducts. It covers micro-unit remainders, truncated avg_price, intermediate values
    /// larger than i64, and closing and reopening the same token.
    #[tokio::test]
    async fn accounting_repeated_sales_conserve_every_micro() {
        for (price, size, parts) in [
            (
                dec!(0.333333),
                dec!(0.000011),
                vec![dec!(0.000001), dec!(0.000003)],
            ),
            (
                dec!(0.444444),
                dec!(10.000003),
                vec![dec!(1.000001), dec!(2.000001)],
            ),
            (
                dec!(0.44),
                dec!(50000),
                vec![dec!(30000), dec!(19999.999999)],
            ),
            (
                dec!(1),
                from_micros(i64::MAX),
                vec![from_micros(i64::MAX / 2)],
            ),
        ] {
            for close_by_sale in [false, true] {
                let store = StateStore::open_memory(Mode::Paper).await.unwrap();
                let buy = test_fill(
                    "micro",
                    StrategyId::Arb,
                    FillSide::Buy,
                    price,
                    size,
                    1000,
                    false,
                    None,
                    "buy",
                );
                store.apply_fill(&buy).await.unwrap();
                let original = store.get_position("micro", "arb").await.unwrap().unwrap();
                let mut allocated_total = Decimal::ZERO;
                let mut parts = parts.clone();
                if close_by_sale {
                    // Deliberate oversell: keeps the existing clamping policy.
                    parts.push(size);
                }
                for (i, part) in parts.into_iter().enumerate() {
                    let before = store.get_position("micro", "arb").await.unwrap().unwrap();
                    let fill = test_fill(
                        "micro",
                        StrategyId::Arb,
                        FillSide::Sell,
                        dec!(0.470001),
                        part,
                        1000,
                        false,
                        None,
                        &format!("sell-{i}"),
                    );
                    let applied = store.apply_fill(&fill).await.unwrap();
                    let after = store.get_position("micro", "arb").await.unwrap().unwrap();
                    let scenario = format!("price={price} size={size} sell #{i}");
                    let AppliedFill::Sell { pnl, cost_released } = applied else {
                        panic!("expected SELL in {scenario}")
                    };
                    let row = sqlx::query("SELECT usdc_um, pnl_um FROM transactions WHERE kind = 'SELL' ORDER BY id DESC LIMIT 1")
                        .fetch_one(&store.pool).await.unwrap();
                    let proceeds: Decimal = from_micros(row.get("usdc_um"));
                    // The cost reported to the ledger must be the SAME one
                    // that the settled PnL deducts.
                    assert_eq!(
                        proceeds - pnl,
                        cost_released,
                        "PnL cost vs. released cost in {scenario}"
                    );
                    allocated_total += cost_released;
                    assert_eq!(
                        before.cost_basis,
                        cost_released + after.cost_basis,
                        "cost removed from the position in {scenario}"
                    );
                    assert_eq!(
                        original.cost_basis,
                        allocated_total + after.cost_basis,
                        "cumulative conservation in {scenario}"
                    );
                    assert_eq!(
                        from_micros(row.get("pnl_um")),
                        pnl,
                        "pnl persisted in {scenario}"
                    );
                    assert!(
                        matches!(
                            store.apply_fill(&fill).await.unwrap(),
                            AppliedFill::Duplicate
                        ),
                        "dedup in {scenario}"
                    );
                    assert_eq!(
                        store
                            .get_position("micro", "arb")
                            .await
                            .unwrap()
                            .unwrap()
                            .cost_basis,
                        after.cost_basis,
                        "the duplicate must not move the cost in {scenario}"
                    );
                }
                let last = store.get_position("micro", "arb").await.unwrap().unwrap();
                if close_by_sale {
                    assert_eq!(last.size, Decimal::ZERO);
                    assert_eq!(last.cost_basis, Decimal::ZERO);
                    assert!(last.bail_sold);
                } else {
                    store
                        .apply_settlement(
                            &original,
                            false,
                            "NO",
                            dec!(999),
                            dec!(999),
                            StrategyId::Arb,
                            Some("test-redeem"),
                        )
                        .await
                        .unwrap();
                    store
                        .apply_settlement(
                            &original,
                            false,
                            "NO",
                            dec!(999),
                            dec!(999),
                            StrategyId::Arb,
                            Some("test-redeem"),
                        )
                        .await
                        .unwrap();
                }
                // Reads monetary flows independently of the cost and the average price.
                let flows: i64 = sqlx::query_scalar("SELECT SUM(CASE WHEN kind = 'BUY' THEN -usdc_um ELSE usdc_um END) FROM transactions")
                    .fetch_one(&store.pool).await.unwrap();
                assert_eq!(
                    store.realized_pnl_total().await.unwrap(),
                    from_micros(flows)
                );
                let reopen = test_fill(
                    "micro",
                    StrategyId::Arb,
                    FillSide::Buy,
                    dec!(0.123457),
                    dec!(3.000001),
                    1000,
                    false,
                    None,
                    "reopen",
                );
                store.apply_fill(&reopen).await.unwrap();
                let reopened = store.get_position("micro", "arb").await.unwrap().unwrap();
                assert_eq!(
                    reopened.cost_basis,
                    from_micros(to_micros(reopen.price * reopen.size))
                );
                assert!(
                    !reopened.bail_sold && !reopened.redeemed && reopened.outcome_won.is_none()
                );
                store
                    .apply_settlement(
                        &reopened,
                        true,
                        "YES",
                        dec!(0),
                        dec!(0),
                        StrategyId::Arb,
                        None,
                    )
                    .await
                    .unwrap();
                let flows: i64 = sqlx::query_scalar("SELECT SUM(CASE WHEN kind = 'BUY' THEN -usdc_um ELSE usdc_um END) FROM transactions")
                    .fetch_one(&store.pool).await.unwrap();
                assert_eq!(
                    store.realized_pnl_total().await.unwrap(),
                    from_micros(flows)
                );
            }
        }
    }

    #[tokio::test]
    async fn accounting_buy_after_partial_sale_preserves_remaining_cost() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        for (side, price, size, id) in [
            (FillSide::Buy, dec!(0.44), dec!(10.989011), "buy"),
            (FillSide::Sell, dec!(0.47), dec!(10.98), "partial"),
            (FillSide::Buy, dec!(0.123457), dec!(3.000001), "expansion"),
        ] {
            store
                .apply_fill(&test_fill(
                    "rebuy",
                    StrategyId::Arb,
                    side,
                    price,
                    size,
                    1000,
                    false,
                    None,
                    id,
                ))
                .await
                .unwrap();
        }
        let pos = store
            .get_position("rebuy", "arb")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            pos.cost_basis,
            dec!(0.003965) + from_micros(to_micros(dec!(0.123457) * dec!(3.000001)))
        );
        store
            .apply_settlement(
                &pos,
                true,
                "YES",
                Decimal::ZERO,
                Decimal::ZERO,
                StrategyId::Arb,
                None,
            )
            .await
            .unwrap();
        let flows: i64 = sqlx::query_scalar(
            "SELECT SUM(CASE WHEN kind = 'BUY' THEN -usdc_um ELSE usdc_um END) FROM transactions",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(
            store.realized_pnl_total().await.unwrap(),
            from_micros(flows)
        );
    }

    /// A SELL arriving AFTER settlement (a late post-close sweep GTD, or the orphan checker
    /// getting ahead of an in-flight fill) used to spend again a cost already booked to
    /// the REDEEM: `apply_settlement` sets `outcome_won`/`redeemed` but does NOT zero
    /// `size_um`/`cost_basis_um`, and the SELL branch only checked `None`/`size == 0`.
    /// Real case: a BUY of 20 at 0.40 (cost 8), then a losing REDEEM (pnl −8), then a SELL
    /// of 20 at 0.35.
    /// The realized PnL must be −1 (paid 8, collected 7), not −9.
    #[tokio::test]
    async fn accounting_sell_after_redeem_does_not_spend_cost_twice() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let buy = test_fill(
            "settled",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.40),
            dec!(20),
            1_000_000,
            false,
            Some("YES"),
            "buy",
        );
        store.apply_fill(&buy).await.unwrap();
        let pos = store
            .get_position("settled", "arb")
            .await
            .unwrap()
            .unwrap();

        // Losing REDEEM: the whole cost (8 USDC) is booked here.
        store
            .apply_settlement(
                &pos,
                false,
                "NO",
                Decimal::ZERO,
                Decimal::ZERO,
                StrategyId::Arb,
                Some("tx-redeem"),
            )
            .await
            .unwrap();

        // The late sale collects 7 real USDC over a position whose cost was already
        // consumed: its PnL is the full proceeds, not proceeds − cost.
        let sell = test_fill(
            "settled",
            StrategyId::Arb,
            FillSide::Sell,
            dec!(0.35),
            dec!(20),
            1_000_000,
            false,
            None,
            "late-sell",
        );
        store.apply_fill(&sell).await.unwrap();

        let flows: i64 = sqlx::query_scalar(
            "SELECT SUM(CASE WHEN kind = 'BUY' THEN -usdc_um ELSE usdc_um END) FROM transactions",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(
            store.realized_pnl_total().await.unwrap(),
            from_micros(flows),
            "Σ pnl must match Σ flows"
        );
        assert_eq!(
            store.realized_pnl_total().await.unwrap(),
            dec!(-1),
            "paid 8 and collected 7: −1, not the phantom loss of −8"
        );
    }

    /// The post-settlement sell branch used to clamp the oversell against the row's `size`.
    /// Since the root fix, settlement sets `size_um` to zero, so there is no longer a bound, and
    /// there must not be one: the fill is the venue's evidence (if it executed 25 shares, then
    /// 25 × price was really received). Clamping to a stale residual would lose real proceeds and
    /// break `Σ pnl == Σ flows`. The clamp still applies in the live-position branch, where the
    /// row is the source of truth.
    /// (Only the live branch applies the clamp.)
    #[tokio::test]
    async fn sell_after_redeem_settles_oversell_without_trim() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let buy = test_fill(
            "oversell",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.40),
            dec!(20),
            1_000_000,
            false,
            Some("YES"),
            "buy",
        );
        store.apply_fill(&buy).await.unwrap();
        let pos = store
            .get_position("oversell", "arb")
            .await
            .unwrap()
            .unwrap();
        store
            .apply_settlement(
                &pos,
                false,
                "NO",
                Decimal::ZERO,
                Decimal::ZERO,
                StrategyId::Arb,
                Some("tx-redeem"),
            )
            .await
            .unwrap();

        // Drift: the fill carries 25 shares over a position of 20.
        let sell = test_fill(
            "oversell",
            StrategyId::Arb,
            FillSide::Sell,
            dec!(0.35),
            dec!(25),
            1_000_000,
            false,
            None,
            "late-sell",
        );
        let applied = store.apply_fill(&sell).await.unwrap();

        let row = sqlx::query("SELECT shares_um, usdc_um FROM transactions WHERE kind = 'SELL'")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(
            from_micros(row.get("shares_um")),
            dec!(25),
            "the settled row (size=0) cannot bound it: the entry reflects the real fill"
        );
        assert_eq!(
            from_micros(row.get("usdc_um")),
            dec!(8.75),
            "full proceeds: 25 × 0.35"
        );
        let AppliedFill::Sell { pnl, .. } = applied else {
            panic!("expected SELL")
        };
        assert_eq!(
            pnl,
            dec!(8.75),
            "the PnL that feeds the daily-loss-stop is the fill's full proceeds"
        );
        assert_eq!(
            store.realized_pnl_total().await.unwrap(),
            dec!(0.75),
            "paid 8, the REDEEM booked −8 and the late sell brought in 8.75: +0.75"
        );
    }

    /// `avg_price` is the mark-price fallback and feeds `positions_value` → equity, so it
    /// can never exceed 1 USDC per share (an outcome share is never worth more than $1).
    /// On a legacy row with an inflated cost (the kind the migration deals with), the
    /// recalculation produced 3.300000 and multiplied that line's value by 10.
    /// (The 1 USDC cap prevents this.)
    #[tokio::test]
    async fn recomputed_avg_price_does_not_exceed_one_usdc() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let buy = test_fill(
            "legacy",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.33),
            dec!(1),
            1000,
            false,
            None,
            "buy",
        );
        store.apply_fill(&buy).await.unwrap();
        // Simulates the legacy row: cost inflated ×10 by old SQL that truncated the cost
        // removed by the partial sells to 0 (not migrated).
        sqlx::query("UPDATE positions SET cost_basis_um = 3300000 WHERE token_id = 'legacy'")
            .execute(&store.pool)
            .await
            .unwrap();

        let partial = test_fill(
            "legacy",
            StrategyId::Arb,
            FillSide::Sell,
            dec!(0.35),
            dec!(0.5),
            1000,
            false,
            None,
            "partial",
        );
        store.apply_fill(&partial).await.unwrap();

        let pos = store.get_position("legacy", "arb").await.unwrap().unwrap();
        assert!(
            pos.avg_price <= dec!(1),
            "an outcome share is never worth more than 1 USDC, but avg_price = {}",
            pos.avg_price
        );
    }

    /// `reduce_position_conn` did not recalculate `avg_price_um`, so after a partial sell
    /// the row kept a stale average price that was no longer `cost_basis/size`. That field
    /// is the mark-price fallback (`src/accounting/mark_price.rs`, `api/handlers.rs`), so
    /// the residual position was valued at an obsolete price.
    /// (Found by review.)
    /// An extreme case: size 11 µ / cost 4 µ, sell of 10 µ.
    #[tokio::test]
    async fn partial_sell_recomputes_avg_price() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let buy = test_fill(
            "avg",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.333333),
            dec!(0.000011),
            1000,
            false,
            None,
            "buy",
        );
        store.apply_fill(&buy).await.unwrap();

        let partial = test_fill(
            "avg",
            StrategyId::Arb,
            FillSide::Sell,
            dec!(0.470001),
            dec!(0.000010),
            1000,
            false,
            None,
            "partial",
        );
        store.apply_fill(&partial).await.unwrap();

        let pos = store.get_position("avg", "arb").await.unwrap().unwrap();
        assert_eq!(pos.size, dec!(0.000001));
        assert_eq!(pos.cost_basis, dec!(0.000001));
        // The invariant is "avg_price == cost/size to the micro", not a concrete value:
        // `avg_price_um` is an integer, so the ratio is compared rounded to the micro,
        // which keeps the test valid when the remainders do not divide exactly
        // (2 µ / 3 µ and similar).
        assert_eq!(
            pos.avg_price,
            from_micros(to_micros(pos.cost_basis / pos.size)),
            "the average price must still be cost/size after the partial sell"
        );

        // Full close: with no remaining size there is no possible division, so there must
        // be no panic from division by zero, and the position is left at zero.
        let close = test_fill(
            "avg",
            StrategyId::Arb,
            FillSide::Sell,
            dec!(0.470001),
            dec!(0.000001),
            1000,
            false,
            None,
            "close",
        );
        store.apply_fill(&close).await.unwrap();
        let closed = store.get_position("avg", "arb").await.unwrap().unwrap();
        assert_eq!(closed.size, Decimal::ZERO);
        assert_eq!(closed.cost_basis, Decimal::ZERO);
    }

    // ── Critical: apply_fill must not revive closed positions ────────────────

    #[tokio::test]
    async fn apply_fill_reopens_position_closed_by_bail_sold() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();

        // 1) Initial buy: opens the position.
        let buy1 = test_fill(
            "tok-1",
            StrategyId::NearExpiry,
            FillSide::Buy,
            dec!(0.40),
            dec!(20),
            1_000_000,
            false,
            Some("YES"),
            "o1",
        );
        store.apply_fill(&buy1).await.unwrap();

        // 2) Bail: sells the WHOLE size → bail_sold=1 (real scenario from
        // near_expiry.rs, already confirmed in review).
        let sell = test_fill(
            "tok-1",
            StrategyId::NearExpiry,
            FillSide::Sell,
            dec!(0.35),
            dec!(20),
            1_000_000,
            false,
            None,
            "o2",
        );
        store.apply_fill(&sell).await.unwrap();

        // Bug repro: the closed position stays invisible to exposure.
        assert!(!store.has_open_position_in_market("cid-1").await.unwrap());
        assert_eq!(store.total_open_cost().await.unwrap(), Decimal::ZERO);
        assert!(store.list_open_positions().await.unwrap().is_empty());

        // 3) Re-entry: NE buys the same token again.
        let buy2 = test_fill(
            "tok-1",
            StrategyId::NearExpiry,
            FillSide::Buy,
            dec!(0.50),
            dec!(15),
            1_000_000,
            false,
            Some("YES"),
            "o3",
        );
        store.apply_fill(&buy2).await.unwrap();

        let pos = store
            .get_position("tok-1", "near_expiry")
            .await
            .unwrap()
            .expect("position reopened");
        assert!(
            !pos.bail_sold,
            "re-entry must not carry over bail_sold=1"
        );
        assert!(
            pos.outcome_won.is_none(),
            "must not carry over outcome_won"
        );
        assert!(!pos.redeemed, "must not carry over redeemed");
        assert_eq!(
            pos.size,
            dec!(15),
            "must not average with the earlier closed position"
        );
        assert_eq!(pos.avg_price, dec!(0.50));
        assert_eq!(pos.cost_basis, dec!(7.5));

        // And now it is visible again for exposure/redemption (the bug hid it).
        assert!(store.has_open_position_in_market("cid-1").await.unwrap());
        assert_eq!(store.total_open_cost().await.unwrap(), dec!(7.5));
        let open = store.list_open_positions().await.unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].token_id, "tok-1");
    }

    #[tokio::test]
    async fn apply_fill_reopens_liquidated_and_redeemed_position() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();

        let buy1 = test_fill(
            "tok-2",
            StrategyId::MarketMaking,
            FillSide::Buy,
            dec!(0.30),
            dec!(10),
            1_000_000,
            false,
            Some("YES"),
            "o1",
        );
        store.apply_fill(&buy1).await.unwrap();
        let pos = store
            .get_position("tok-2", "market_making")
            .await
            .unwrap()
            .unwrap();
        let payout = pos.size;
        let pnl = payout - pos.cost_basis;
        store
            .apply_settlement(
                &pos,
                true,
                "YES",
                payout,
                pnl,
                StrategyId::MarketMaking,
                Some("0xredeemtx"),
            )
            .await
            .unwrap();

        let closed = store
            .get_position("tok-2", "market_making")
            .await
            .unwrap()
            .unwrap();
        assert!(closed.redeemed);
        assert_eq!(closed.outcome_won, Some(true));

        // Re-entry after redemption: MM reopens the same token in a new cycle.
        let buy2 = test_fill(
            "tok-2",
            StrategyId::MarketMaking,
            FillSide::Buy,
            dec!(0.45),
            dec!(5),
            1_000_000,
            false,
            Some("YES"),
            "o2",
        );
        store.apply_fill(&buy2).await.unwrap();

        let reopened = store
            .get_position("tok-2", "market_making")
            .await
            .unwrap()
            .unwrap();
        assert!(!reopened.redeemed, "must not carry over redeemed=1");
        assert!(
            reopened.outcome_won.is_none(),
            "must not carry over outcome_won"
        );
        assert_eq!(
            reopened.redeem_tx, None,
            "must not carry over the stale redeem_tx"
        );
        assert_eq!(reopened.size, dec!(5));
    }

    // ── apply_settlement must be idempotent ──────────────────────────────────

    #[tokio::test]
    async fn apply_settlement_is_idempotent() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let buy = test_fill(
            "tok-1",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.60),
            dec!(10),
            1000,
            false,
            Some("YES"),
            "o1",
        );
        store.apply_fill(&buy).await.unwrap();
        let pos = store.get_position("tok-1", "arb").await.unwrap().unwrap();

        let payout = pos.size;
        let pnl = payout - pos.cost_basis;
        store
            .apply_settlement(&pos, true, "YES", payout, pnl, StrategyId::Arb, None)
            .await
            .unwrap();
        // Second call (race with settlement::orphan::poll): must not duplicate the REDEEM.
        store
            .apply_settlement(&pos, true, "YES", payout, pnl, StrategyId::Arb, None)
            .await
            .unwrap();

        let total_pnl = store.realized_pnl_total().await.unwrap();
        assert_eq!(
            total_pnl, pnl,
            "a duplicate settlement must not add the pnl twice"
        );

        let txs = store.recent_transactions(10, None).await.unwrap();
        let redeem_count = txs.iter().filter(|t| t["kind"] == "REDEEM").count();
        assert_eq!(redeem_count, 1, "there must be exactly ONE REDEEM transaction");
    }

    #[tokio::test]
    async fn apply_settlement_uses_fresh_size_not_the_passed_snapshot() {
        // Simulates the race: settlement::orphan::poll reads `pos` with size=10, but between that
        // read and the settlement a late fill raises the position to
        // size=15 (a late post-close sweep GTD). apply_settlement must settle
        // over the FRESH size (15), not the stale snapshot (10).
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let buy1 = test_fill(
            "tok-1",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.50),
            dec!(10),
            1000,
            false,
            Some("YES"),
            "o1",
        );
        store.apply_fill(&buy1).await.unwrap();
        let stale_pos = store.get_position("tok-1", "arb").await.unwrap().unwrap(); // size=10, cost=5

        // Late fill arrives AFTER the settlement::orphan::poll read but BEFORE the settlement.
        let buy2 = test_fill(
            "tok-1",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.50),
            dec!(5),
            1000,
            false,
            Some("YES"),
            "o2",
        );
        store.apply_fill(&buy2).await.unwrap(); // size=15, cost=7.5

        // src/settlement/orphan.rs computes payout/pnl over the OBSOLETE snapshot (size=10).
        let stale_payout = stale_pos.size;
        let stale_pnl = stale_payout - stale_pos.cost_basis;
        store
            .apply_settlement(
                &stale_pos,
                true,
                "YES",
                stale_payout,
                stale_pnl,
                StrategyId::Arb,
                None,
            )
            .await
            .unwrap();

        // The persisted REDEEM must reflect the FRESH size (15), not the stale one (10).
        let txs = store.recent_transactions(10, None).await.unwrap();
        let redeem = txs
            .iter()
            .find(|t| t["kind"] == "REDEEM")
            .expect("a REDEEM must exist");
        let shares: Decimal = redeem["shares"].as_str().unwrap().parse().unwrap();
        assert_eq!(
            shares,
            dec!(15),
            "the REDEEM must use the fresh size, not the stale snapshot"
        );
        let pnl: Decimal = redeem["pnl"].as_str().unwrap().parse().unwrap();
        assert_eq!(
            pnl,
            dec!(7.5),
            "the pnl must be computed over the fresh size/cost_basis (15 - 7.5)"
        );
    }

    #[tokio::test]
    async fn apply_settlement_zeroes_size_and_cost_and_late_sell_settles_proceeds() {
        // Root fix: the settled row stays as history but without inventory state,
        // i.e. size_um = cost_basis_um = 0. A SELL that arrives later cannot spend cost
        // already booked to the REDEEM: it is settled with zero cost and its full proceeds as PnL.
        // (The zero-cost branch of `apply_fill` handles it.)
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let buy = test_fill(
            "tok-1",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.60),
            dec!(10),
            1000,
            false,
            Some("YES"),
            "o1",
        );
        store.apply_fill(&buy).await.unwrap();
        let pos = store.get_position("tok-1", "arb").await.unwrap().unwrap();
        let pnl = pos.size - pos.cost_basis;
        store
            .apply_settlement(&pos, true, "YES", pos.size, pnl, StrategyId::Arb, None)
            .await
            .unwrap();

        let settled = store.get_position("tok-1", "arb").await.unwrap().unwrap();
        assert_eq!(
            settled.size,
            Decimal::ZERO,
            "settled: no residual size"
        );
        assert_eq!(
            settled.cost_basis,
            Decimal::ZERO,
            "settled: no residual cost"
        );
        assert_eq!(settled.outcome_won, Some(true));

        // Late sell (fill in flight at settlement), zero-cost branch: full proceeds,
        // without trimming by size (the emptied row cannot contradict it).
        let late_sell = test_fill(
            "tok-1",
            StrategyId::Arb,
            FillSide::Sell,
            dec!(0.55),
            dec!(4),
            2000,
            false,
            Some("YES"),
            "o2",
        );
        let applied = store.apply_fill(&late_sell).await.unwrap();
        match applied {
            AppliedFill::Sell { pnl, cost_released } => {
                assert_eq!(pnl, dec!(2.2), "0.55 × 4: full proceeds");
                assert_eq!(cost_released, Decimal::ZERO);
            }
            other => panic!("a late sell over a settled position must be booked, not {other:?}"),
        }
    }

    // ── apply_fill must repair clear_at/neg_risk/outcome_label ───────────────

    #[tokio::test]
    async fn apply_fill_repairs_clear_at_neg_risk_outcome_label() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        // Initial "broken" fill: clear_at=0 (the `real_clear_at.unwrap_or(0)` fallback),
        // neg_risk=false, no outcome_label.
        let buy1 = test_fill(
            "tok-1",
            StrategyId::MarketMaking,
            FillSide::Buy,
            dec!(0.40),
            dec!(10),
            0,
            false,
            None,
            "o1",
        );
        store.apply_fill(&buy1).await.unwrap();
        let broken = store
            .get_position("tok-1", "market_making")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(broken.clear_at, 0);
        assert_eq!(broken.outcome_label, None);

        // Later fill with correct data.
        let buy2 = test_fill(
            "tok-1",
            StrategyId::MarketMaking,
            FillSide::Buy,
            dec!(0.42),
            dec!(5),
            1_234_567,
            true,
            Some("YES"),
            "o2",
        );
        store.apply_fill(&buy2).await.unwrap();

        let repaired = store
            .get_position("tok-1", "market_making")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            repaired.clear_at, 1_234_567,
            "a broken clear_at must be repaired"
        );
        assert!(repaired.neg_risk, "neg_risk must be repaired to true");
        assert_eq!(
            repaired.outcome_label.as_deref(),
            Some("YES"),
            "an empty outcome_label must be repaired"
        );
        // Normal averaging still works.
        assert_eq!(repaired.size, dec!(15));
    }

    // ── record_transaction_conn must stamp the fill's ts ─────────────────────

    #[tokio::test]
    async fn apply_fill_seals_transactions_with_fill_ts_not_now() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let old_ts = chrono::DateTime::from_timestamp_millis(1_600_000_000_000).unwrap();
        let mut fill = test_fill(
            "tok-1",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(0.5),
            dec!(10),
            1000,
            false,
            None,
            "o1",
        );
        fill.ts = old_ts;
        store.apply_fill(&fill).await.unwrap();

        let txs = store.recent_transactions(1, None).await.unwrap();
        let ts_ms = txs[0]["ts_ms"].as_i64().unwrap();
        assert_eq!(
            ts_ms, 1_600_000_000_000,
            "the transaction must stamp the fill's ts, not Utc::now()"
        );
    }

    // ── cost_basis proration without overflow for large positions ───────────

    #[tokio::test]
    async fn reduce_position_does_not_overflow_with_large_positions() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        // 50,000 shares at price 1.0 → cost_basis_um=5e10, size_um=5e10. The
        // ORIGINAL formula (cost_basis_um * sold_um) with sold_um=3e10 would give
        // 1.5e21 > i64::MAX (9.2e18): overflow. The reordered formula avoids that
        // intermediate value.
        let buy = test_fill(
            "tok-1",
            StrategyId::Arb,
            FillSide::Buy,
            dec!(1.0),
            dec!(50000),
            1000,
            false,
            None,
            "o1",
        );
        store.apply_fill(&buy).await.unwrap();

        let sell = test_fill(
            "tok-1",
            StrategyId::Arb,
            FillSide::Sell,
            dec!(1.0),
            dec!(30000),
            1000,
            false,
            None,
            "o2",
        );
        store.apply_fill(&sell).await.unwrap();

        let pos = store.get_position("tok-1", "arb").await.unwrap().unwrap();
        assert_eq!(pos.size, dec!(20000));
        assert_eq!(
            pos.cost_basis,
            dec!(20000),
            "cost_basis must remain exact without overflow"
        );
    }

    // ── to_micros logs (and still returns 0) for unrepresentable values ──────

    #[test]
    fn to_micros_unrepresentable_value_returns_zero() {
        // A Decimal whose value ×1e6 does not fit in i64.
        let huge: Decimal = "99999999999999999999".parse().unwrap();
        assert_eq!(to_micros(huge), 0);
    }

    // ── positions migration robust to column reordering ──────────────────────

    #[tokio::test]
    async fn migrates_positions_from_simple_pk_to_composite_pk_by_column_name() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("state.sqlite");
        // resolve_path adds the mode suffix: state.sqlite (Paper) → state_paper.sqlite.
        let paper_path = dir.path().join("state_paper.sqlite");

        // Manually creates the OLD schema (simple PK on token_id) with one row:
        // simulates a DB from a session earlier than the composite-PK migration.
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&format!("sqlite://{}?mode=rwc", paper_path.display()))
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE positions (
                token_id        TEXT PRIMARY KEY,
                strategy        TEXT NOT NULL DEFAULT '',
                market_cid      TEXT NOT NULL,
                size_um         INTEGER NOT NULL DEFAULT 0,
                avg_price_um    INTEGER NOT NULL DEFAULT 0,
                cost_basis_um   INTEGER NOT NULL DEFAULT 0,
                opened_ts_ms    INTEGER NOT NULL DEFAULT 0,
                clear_at        INTEGER NOT NULL DEFAULT 0,
                neg_risk        INTEGER NOT NULL DEFAULT 0,
                bail_sold       INTEGER NOT NULL DEFAULT 0,
                outcome_won     INTEGER,
                winner_label    TEXT,
                outcome_label   TEXT,
                redeemed        INTEGER NOT NULL DEFAULT 0,
                redeem_tx       TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO positions
             (token_id, strategy, market_cid, size_um, avg_price_um, cost_basis_um,
              opened_ts_ms, clear_at, neg_risk, bail_sold, outcome_won, winner_label,
              outcome_label, redeemed, redeem_tx)
             VALUES ('tok-1', 'arb', 'cid-1', 10000000, 500000, 5000000, 1000, 2000, 0, 0, NULL, NULL, 'YES', 0, NULL)",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool.close().await;

        let store = StateStore::open(db_path.to_str().unwrap(), Mode::Paper)
            .await
            .unwrap();

        let pos = store
            .get_position("tok-1", "arb")
            .await
            .unwrap()
            .expect("migrated position must exist");
        assert_eq!(pos.size, dec!(10));
        assert_eq!(pos.avg_price, dec!(0.5));
        assert_eq!(pos.outcome_label.as_deref(), Some("YES"));

        let pk_cols: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM pragma_table_info('positions') WHERE pk > 0")
                .fetch_one(&store.pool)
                .await
                .unwrap();
        assert_eq!(
            pk_cols, 2,
            "the migration must leave the composite PK (token_id, strategy)"
        );
    }

    // ── Concurrent writes under load must not fail with "database is locked" ──

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_writes_do_not_fail_with_database_is_locked() {
        // open_memory uses max_connections(1) (no contention possible), so the real file
        // path (max_connections(4)) is needed to exercise the pool under real concurrency
        // of OS threads (requires the multi_thread runtime above).
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("state.sqlite");
        let store = StateStore::open(db_path.to_str().unwrap(), Mode::Paper)
            .await
            .unwrap();

        let mut handles = Vec::new();
        for i in 0..300 {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                store
                    .add_deposit(dec!(1), &format!("concurrent-{i}"))
                    .await
            }));
        }
        for h in handles {
            h.await
                .unwrap()
                .expect("must not fail with 'database is locked'");
        }

        assert_eq!(store.deposits_total().await.unwrap(), dec!(300));
    }

    // ── upsert_market must refresh conflicting fields ────────────────────────

    #[tokio::test]
    async fn upsert_market_updates_fields_on_conflict() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        store
            .upsert_market("cid-1", "slug-v1", "yes-1", "no-1", 100, 200, false)
            .await
            .unwrap();
        store
            .upsert_market("cid-1", "slug-v2", "yes-2", "no-2", 999, 300, true)
            .await
            .unwrap();

        let slug = store.get_market_slug("cid-1").await;
        assert_eq!(
            slug.as_deref(),
            Some("slug-v2"),
            "upsert_market must refresh the slug on conflict"
        );

        let row = sqlx::query(
            "SELECT yes_token_id, clear_at, neg_risk, market_open_ts FROM markets_observed WHERE condition_id = 'cid-1'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let yes_token_id: String = row.try_get("yes_token_id").unwrap();
        let clear_at: i64 = row.try_get("clear_at").unwrap();
        let neg_risk: i64 = row.try_get("neg_risk").unwrap();
        let market_open_ts: i64 = row.try_get("market_open_ts").unwrap();
        assert_eq!(
            yes_token_id, "yes-2",
            "yes_token_id must be refreshed"
        );
        assert_eq!(clear_at, 300, "clear_at must be refreshed");
        assert_eq!(neg_risk, 1, "neg_risk must be refreshed");
        assert_eq!(
            market_open_ts, 100,
            "market_open_ts is preserved from the first insert"
        );
    }

    // ── Index on positions(market_cid) + purge of old gtd_orders ────────────

    #[tokio::test]
    async fn positions_market_cid_index_exists() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let idx: Option<String> = sqlx::query_scalar(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_positions_market_cid'",
        )
        .fetch_optional(&store.pool)
        .await
        .unwrap();
        assert!(idx.is_some(), "the idx_positions_market_cid index is missing");
    }

    #[tokio::test]
    async fn purge_stale_gtd_orders_deletes_only_old_ones() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        let old_ts = chrono::Utc::now().timestamp_millis() - 999_999_000; // very old
        sqlx::query(
            "INSERT INTO gtd_orders (order_id, market_cid, strategy, ts_ms) VALUES (?, ?, ?, ?)",
        )
        .bind("old-1")
        .bind("cid")
        .bind("arb")
        .bind(old_ts)
        .execute(&store.pool)
        .await
        .unwrap();
        store
            .record_gtd_order("fresh-1", "cid", StrategyId::Arb)
            .await
            .unwrap();

        let purged = store.purge_stale_gtd_orders(3600).await.unwrap();
        assert_eq!(purged, 1);
        assert!(store.get_gtd_strategy("old-1").await.unwrap().is_none());
        assert!(store.get_gtd_strategy("fresh-1").await.unwrap().is_some());
    }

    // ── Dead-letter fills ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn record_dead_letter_fill_persists_and_counts_as_unresolved() {
        use crate::types::{DeadLetterFill, FillSide};
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        assert_eq!(store.count_unresolved_dead_letters().await.unwrap(), 0);

        let dl = DeadLetterFill {
            order_id: "trade-1".to_string(),
            market_cid: "cid-orphan_fill".to_string(),
            token_id: "tok-a".to_string(),
            side: FillSide::Buy,
            price: rust_decimal_macros::dec!(0.5),
            size: rust_decimal_macros::dec!(10),
            status: "MATCHED".to_string(),
            role: "maker".to_string(),
            reason: "market not found in registry nor clear_at_cache".to_string(),
            ts: chrono::Utc::now(),
        };
        store.record_dead_letter_fill(&dl).await.unwrap();
        assert_eq!(
            store.count_unresolved_dead_letters().await.unwrap(),
            1,
            "a newly inserted dead-letter must count as unresolved"
        );

        let row = sqlx::query("SELECT order_id, market_cid, side, price_um FROM dead_letter_fills")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        let order_id: String = row.try_get("order_id").unwrap();
        let side: String = row.try_get("side").unwrap();
        assert_eq!(order_id, "trade-1");
        assert_eq!(side, "BUY");
    }

    /// Inserts a closed SELL with the given PnL, using the same raw-SQL pattern as
    /// `pnl_series_returns_only_closed_sorted` (there is no other reusable helper in
    /// this file for inserting closed test fills).
    async fn insert_test_fill_with_pnl(store: &StateStore, idx: i64, pnl: Decimal) {
        sqlx::query(
            "INSERT INTO transactions (ts_ms, kind, market_cid, token_id, shares_um, price_um, usdc_um, pnl_um, strategy)
             VALUES (?1, 'SELL', 'cid', 'tok', 10000000, 500000, 5000000, ?2, 'arb')",
        )
        .bind(1000 + idx * 1000)
        .bind(to_micros(pnl))
        .execute(&store.pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn max_drawdown_computes_largest_drop_from_peak() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        // Closed PnL series: +10, +5 (peak=15), -20 (dd=20), +3 (dd stays at 20, below the peak)
        for (i, pnl) in [dec!(10), dec!(5), dec!(-20), dec!(3)]
            .into_iter()
            .enumerate()
        {
            insert_test_fill_with_pnl(&store, i as i64, pnl).await;
        }
        let dd = store.max_drawdown().await.unwrap();
        assert_eq!(dd, dec!(20));
    }

    /// Inserts a closed SELL with `ts_ms` set to fall exactly in the UTC hour
    /// `hour` (0-23), using `hour` hours since epoch (1970-01-01 UTC) so as not to depend
    /// on the system clock. Variant of `insert_test_fill_with_pnl` that controls the hour
    /// instead of the order.
    async fn insert_test_fill_at_hour(store: &StateStore, hour: u32, pnl: Decimal) {
        let ts_ms = i64::from(hour) * 3_600_000;
        sqlx::query(
            "INSERT INTO transactions (ts_ms, kind, market_cid, token_id, shares_um, price_um, usdc_um, pnl_um, strategy)
             VALUES (?1, 'SELL', 'cid', 'tok', 10000000, 500000, 5000000, ?2, 'arb')",
        )
        .bind(ts_ms)
        .bind(to_micros(pnl))
        .execute(&store.pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn pnl_by_hour_of_day_groups_by_utc_hour() {
        let store = StateStore::open_memory(Mode::Paper).await.unwrap();
        // Two fills in hour 10 UTC, one in hour 15 UTC.
        insert_test_fill_at_hour(&store, 10, dec!(5)).await;
        insert_test_fill_at_hour(&store, 10, dec!(3)).await;
        insert_test_fill_at_hour(&store, 15, dec!(-2)).await;

        let by_hour = store.pnl_by_hour_of_day().await.unwrap();
        let hour_10 = by_hour.iter().find(|(h, _)| *h == 10).unwrap().1;
        let hour_15 = by_hour.iter().find(|(h, _)| *h == 15).unwrap().1;
        assert_eq!(hour_10, dec!(8));
        assert_eq!(hour_15, dec!(-2));
    }

    /// `BEGIN IMMEDIATE`: with deferred transactions (read, then write), a second writer
    /// that took its snapshot before the other commit failed with `SQLITE_BUSY_SNAPSHOT` (517),
    /// an error that does NOT respect `busy_timeout` (the `database is locked` errors that
    /// fill_sink retried). With IMMEDIATE, BEGIN requests the RESERVED lock at once: the second
    /// writer WAITS for `busy_timeout` instead of failing. Here it is seen directly: while tx1
    /// lives, the second BEGIN cannot complete (a deferred BEGIN would have finished at once,
    /// since it does not request the lock until the first write).
    /// (That is the behavior this test pins down.)
    #[tokio::test]
    async fn begin_immediate_waits_for_write_lock() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("lock_test.sqlite");
        let state = StateStore::open(db_path.to_str().unwrap(), Mode::Paper)
            .await
            .unwrap();

        let tx1 = state.begin_immediate().await.unwrap(); // RESERVED lock taken
        let st2 = state.clone();
        let waiter = tokio::spawn(async move { st2.begin_immediate().await.map(|_| ()) });
        // 150ms << busy_timeout (20s): if the second tx is still waiting, it means
        // the BEGIN requested the lock (IMMEDIATE); a deferred one would already have finished.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        assert!(
            !waiter.is_finished(),
            "BEGIN IMMEDIATE did not wait for the lock: it behaved as deferred"
        );
        tx1.rollback().await.unwrap();
        waiter.await.unwrap().unwrap();
    }
}
