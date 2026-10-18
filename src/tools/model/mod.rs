//! `crate::tools::model` — version of the model observer organized as
//! **supervisor tool**.
//!
//! ## Design (separate from the historical `model_observer.rs`)
//!
//! The numeric model observer lived in `crate::model_observer` as a
//! standalone task wired by hand in `main.rs`. To integrate it into the architecture
//! (`src/tools/` = in-process observers that never emit orders) this
//! module:
//!
//! 1. Re-exports ALL public items of `model_observer.rs` (`ShadowModel`,
//!    `Predictor`, `FeatureMap`, `extract_features`, `SpotSeries`,
//!    `ShadowWriter`, `ShadowProducer`, `ShadowTotals`, `ShadowLine`,
//!    `stats_line`, `SHADOW_SCHEMA_VERSION`, `SUPPORTED_FEATURES`,
//!    `ArtifactError`, `PredictionError`, `sha256_hex`). The binary and
//!    tests that imported `crate::model_observer::*` keep compiling without
//!    changes (`pub use` compatibility).
//!
//! 2. Adds [`ModelTool`] and [`ModelProducer`] / [`ModelWriter`] — the
//!    **tool** of the supervisor. It produces ONLY probabilities (P(YES)) per
//!    decision; the strategy `ModelBtc5m` (`crate::strategy::model_btc5m`) is
//!    the only one authorized to translate a probability into an `OrderIntent`. Structural
//!    guarantee: `ModelToolCtx` carries no `intent_tx`, `risk` or `state`, so
//!    the tool **cannot emit orders or reserve cash by construction**
//!    (same principle as `ShadowCtx`, now exposed to the tool supervisor
//!    so that `ToolId::ModelTool` takes part in the hot toggle and orderly stop
//!    like the rest of the tools).
//!
//! ## Why a separate producer (not reusing `ShadowProducer`)
//!
//! `ShadowProducer` is coupled to the `ShadowLine` format and to the record
//! path of the historical shadow. Here the tool:
//! - **Does not require an artifact to start**: if the model does not load, the tool
//!   keeps emitting recorded abstentions (`model_unavailable`) — useful
//!   to diagnose the environment without bringing the bot down.
//! - **Dedicated probability channel** (`ModelPrediction`) consumable by the
//!   strategy via `broadcast::Receiver` — `run_model_tool` emits ONE line
//!   per evaluated book, and the strategy decides what to do with it.
//! - **Same `tokio` runtime + writer on an OS thread** as `ShadowWriter`, with
//!   non-blocking `try_send`.
//!
//! The VIRTUAL (shadow) decisions keep being emitted through the original path
//! so as not to break the existing research instrument: this module is
//! ADDITIVE, it replaces nothing.
pub mod artifacts;

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, watch};

use crate::types::{BookSnapshot, VolSnapshot};

// ── Compatibility re-exports for `crate::model_observer::*` ─────────────────
//
// The historical module (`model_observer.rs`) still owns the
// implementation. This `mod.rs` wraps it and adds the supervisor tool.
// `main.rs` and the existing tests can keep importing from
// `crate::model_observer` (via `lib.rs::model_observer`) without changes.
pub use crate::model_observer::{
    sha256_hex, ArtifactError, PredictionError, extract_features, FeatureMap,
    SUPPORTED_FEATURES, Predictor, ShadowModel, ShadowProducer, ShadowTotals,
    ShadowWriter, SHADOW_SCHEMA_VERSION, SpotSeries,
};

/// Sentinel predictor: always returns `PredictionError::ModelUnavailable`.
/// Useful for tests (pure `model_unavailable`) and as a fallback when the
/// artifact does not load but we want the tool to stay up emitting
/// abstentions, instead of spawning an empty task that does nothing.
#[derive(Debug, Default)]
pub struct NullPredictor;

impl Predictor for NullPredictor {
    fn predict(&self, _feats: &FeatureMap) -> Result<f64, PredictionError> {
        Err(PredictionError::ModelUnavailable("null predictor".into()))
    }
    fn model_hash(&self) -> &str {
        "null"
    }
}

// ── Configuration specific to the model tool ─────────────────────────────────

/// Configuration of the `ModelTool` tool. Same pattern as `ResearchEvents`:
/// `[model_tool]` section in `config/strategies/<tool>.toml`. The supervisor
/// enables it gated by `enabled` (default false) and exposes it to the hot
/// toggle of the API/Telegram.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelToolConfig {
    #[serde(default)]
    pub enabled: bool,
        /// Directory of the versioned model artifact (manifest.json +
        /// model.json). If it points to a dir without manifest/model the tool starts
        /// anyway and emits `None` predictions (recorded as
        /// `model_unavailable`) — not fail-closed: the purpose is to be able to
        /// calibrate/diagnose without a manually placed artifact.
    #[serde(default = "defaults::model_tool_artifact_dir")]
    pub artifact_dir: String,
        /// Directory of the JSONL record of probabilities. Same convention as
        /// `model_shadow.decisions_dir` — research path, not an active DB.
    #[serde(default = "defaults::model_tool_predictions_dir")]
    pub predictions_dir: String,
        /// Minimum cadence between evaluations of the same market (ms).
    #[serde(default = "defaults::model_tool_min_interval_ms")]
    pub min_decision_interval_ms: u64,
        /// Cadence of the `stats` lines (s).
    #[serde(default = "defaults::model_tool_stats_interval_secs")]
    pub stats_interval_secs: u64,
        /// Capacity of the producer→writer channel.
    #[serde(default = "defaults::model_tool_channel_capacity")]
    pub channel_capacity: usize,
        /// Capacity of the strategy↔tool broadcast. Small (16) is enough: the
        /// strategy consumes every book (it does not drain history).
    #[serde(default = "defaults::model_tool_broadcast_capacity")]
    pub broadcast_capacity: usize,
        /// Effective latency budget (ms). `0` = derived from the horizon of
        /// signal / 10. Exceeding it = `latency_exceeded` abstention.
    #[serde(default)]
    pub latency_budget_ms: u64,
        /// Signal horizon in ms (only to derive the budget if
    /// `latency_budget_ms == 0`).
    #[serde(default = "defaults::model_tool_signal_horizon_ms")]
    pub signal_horizon_ms: u64,
}

impl Default for ModelToolConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            artifact_dir: defaults::model_tool_artifact_dir(),
            predictions_dir: defaults::model_tool_predictions_dir(),
            min_decision_interval_ms: defaults::model_tool_min_interval_ms(),
            stats_interval_secs: defaults::model_tool_stats_interval_secs(),
            channel_capacity: defaults::model_tool_channel_capacity(),
            broadcast_capacity: defaults::model_tool_broadcast_capacity(),
            latency_budget_ms: 0,
            signal_horizon_ms: defaults::model_tool_signal_horizon_ms(),
        }
    }
}

pub mod defaults {
    use std::path::PathBuf;

        /// Default artifact = the minimal copy committed alongside the code.
    pub fn model_tool_artifact_dir() -> String {
                // `<workspace>/src/tools/model/artifacts/btc5m` — resolve it relative to the
                // binary's manifest (CARGO_MANIFEST_DIR) at runtime; the default
                // TOML points to the absolute repo path, so the `.env` override is
                // unnecessary in the current paper setup. See `ModelTool::artifact_path`.
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/tools/model/artifacts/btc5m")
            .to_string_lossy()
            .into_owned()
    }
    pub fn model_tool_predictions_dir() -> String {
        "data/model_tool".to_string()
    }
    pub fn model_tool_min_interval_ms() -> u64 {
        1_000
    }
    pub fn model_tool_stats_interval_secs() -> u64 {
        60
    }
    pub fn model_tool_channel_capacity() -> usize {
        4_096
    }
    pub fn model_tool_broadcast_capacity() -> usize {
        16
    }
    pub fn model_tool_signal_horizon_ms() -> u64 {
        30_000
    }
}

impl ModelToolConfig {
        /// Effective latency budget (ms). Same criterion as
        /// `ModelShadowConfig::effective_budget_ms`: explicit if declared, otherwise
        /// derived from the signal horizon.
    pub fn effective_budget_ms(&self) -> u64 {
        if self.latency_budget_ms > 0 {
            self.latency_budget_ms
        } else {
            (self.signal_horizon_ms / 10).max(1)
        }
    }
}

// ── Prediction / broadcast ─────────────────────────────────────────────────────

/// Point prediction emitted by `ModelTool` to the strategy↔tool broadcast.
///
/// **Not** an order and contains nothing executable: it is the point probability
/// P(YES) ∈ [0,1] (or `None` if the model was not available) together with the
/// features relevant for the strategy to decide whether to translate it into an
/// `OrderIntent`. The strategy (`ModelBtc5m`) is the ONLY one that reads this
/// channel; nothing in the executor or the ledger consumes it directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub schema_version: u32,
    pub ts_ms: i64,
    pub market_cid: String,
    pub slug: String,
    pub token_id_yes: String,
    pub token_id_no: String,
    pub side: String,
        /// P(YES) ∈ [0,1] from the model (same math as `ShadowModel::predict`).
        /// `None` = prediction not available (artifact missing or error).
    pub p_model: Option<f64>,
        /// market_p (yes_mid) recorded at the time of the decision — the
        /// strategy does NOT use it to infer token labels (identity comes from
    /// `market_meta`/`token_id_yes`/`token_id_no`).
    pub p_market: Option<f64>,
        /// Book age at evaluation (ms).
    pub book_age_ms: u64,
        /// Spot age at evaluation (ms).
    pub spot_age_ms: u64,
        /// Latency recv→evaluation (ms).
    pub latency_ms: u64,
    /// Effective budget applied (ms).
    pub budget_ms: u64,
        /// Reason for abstention if NO probability was emitted (`book_stale`,
    /// `spot_stale`, `latency_exceeded`, `model_unavailable`, `post_close`).
    pub abstention: Option<String>,
        /// Hash of the model (sha256 of model.json). `None` if there was no model.
    pub model_hash: Option<String>,
}

pub const MODEL_TOOL_SCHEMA_VERSION: u32 = 1;

// ── Producer / writer ───────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct ModelCounters {
    enqueued: AtomicU64,
    discarded: AtomicU64,
}

/// Non-blocking clonable handle to the JSONL writer. Pure `try_send`: a
/// record that does not fit is discarded and counted — the hot path never stops.
#[derive(Clone)]
pub struct ModelProducer {
    tx: mpsc::Sender<ModelPrediction>,
    counters: Arc<ModelCounters>,
}

impl ModelProducer {
        /// Enqueues a prediction. `false` if it was discarded (channel full or writer
        /// down) — counted in `discarded_total`.
    pub fn record(&self, pred: ModelPrediction) -> bool {
        match self.tx.try_send(pred) {
            Ok(()) => {
                self.counters.enqueued.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => {
                self.counters.discarded.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("hrf_model_tool_discarded_total").increment(1);
                false
            }
        }
    }

        /// Emits a prediction to the strategy↔tool broadcast. `None` if there are no
        /// active subscribers (the strategy is off) — NOT an error, only
        /// an honest skip.
    pub fn broadcast(&self, tx: &broadcast::Sender<ModelPrediction>, pred: ModelPrediction) {
                // broadcast::send fails only if there are no receivers — the strategy
                // being off is the normal scenario in paper; we do not count it as an error.
        let _ = tx.send(pred);
    }

    /// Sink producer (tests / absent writer): everything is discarded.
    pub fn sink() -> Self {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        Self {
            tx,
            counters: Arc::new(ModelCounters::default()),
        }
    }
}

/// JSONL writer of the prediction record: dedicated OS thread,
/// append-only, one file per UTC day. Same pattern as `ShadowWriter`.
pub struct ModelWriter {
    tx: mpsc::Sender<ModelPrediction>,
    counters: Arc<ModelCounters>,
    writer: Option<JoinHandle<u64>>,
}

impl ModelWriter {
    pub fn start(dir: &Path, capacity: usize) -> std::io::Result<Self> {
        fs::create_dir_all(dir)?;
        let (tx, rx) = mpsc::channel(capacity.max(1));
        let counters = Arc::new(ModelCounters::default());
        let dir = dir.to_path_buf();
        let writer_counters = counters.clone();
        let writer = std::thread::Builder::new()
            .name("model_tool_writer".to_string())
            .spawn(move || run_writer_model(dir, rx, writer_counters))?;
        Ok(Self {
            tx,
            counters,
            writer: Some(writer),
        })
    }

    pub fn producer(&self) -> ModelProducer {
        ModelProducer {
            tx: self.tx.clone(),
            counters: self.counters.clone(),
        }
    }

        /// Orderly close: drops the sender and waits for the thread. Same criterion as
    /// `ShadowWriter::close`.
    pub async fn close(mut self) -> (u64, u64) {
        let writer = self.writer.take().expect("close() was already called");
        let counters = self.counters.clone();
        drop(self);
        let written = match tokio::task::spawn_blocking(move || writer.join()).await {
            Ok(Ok(n)) => n,
            _ => {
                tracing::error!("model_tool: panic in the writer thread — partial close");
                0
            }
        };
        (
            written,
            counters.discarded.load(Ordering::Relaxed),
        )
    }
}

/// Heartbeat of the dedup: an unchanged row is rewritten anyway after this
/// time since the last write for the same market, so the record keeps
/// showing that the market was being evaluated (cadence reconstructible at 5 min).
const DEDUP_HEARTBEAT_MS: i64 = 300_000;

/// Dedup of the JSONL record by content hash and market.
/// Measured in paper: ~96 rows/s, 89 % identical to the previous row of the same cid
/// apart from `ts_ms` and ages (abstentions
/// `book_stale` on markets with no activity) → ~4.8 GB/day. It only affects the
/// file path: the broadcast to the strategy still receives every prediction.
#[derive(Default)]
struct DedupRecord {
        /// cid → (content hash, ts_ms of the last row written).
    last_key: HashMap<String, (u64, i64)>,
}

impl DedupRecord {
        /// `true` if the row adds something: first for the cid, content different from
        /// the last one written, or heartbeat expired. It excludes from the hash what changes in
        /// each tick without being new information: `ts_ms`, `book_age_ms`,
    /// `spot_age_ms`, `latency_ms`.
    fn must_write(&mut self, pred: &ModelPrediction) -> bool {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        pred.schema_version.hash(&mut h);
        pred.slug.hash(&mut h);
        pred.token_id_yes.hash(&mut h);
        pred.token_id_no.hash(&mut h);
        pred.side.hash(&mut h);
        pred.p_model.map(f64::to_bits).hash(&mut h);
        pred.p_market.map(f64::to_bits).hash(&mut h);
        pred.budget_ms.hash(&mut h);
        pred.abstention.hash(&mut h);
        pred.model_hash.hash(&mut h);
        let hash = h.finish();

        match self.last_key.get_mut(&pred.market_cid) {
            Some((h_prev, ts_prev))
                if *h_prev == hash && pred.ts_ms - *ts_prev < DEDUP_HEARTBEAT_MS =>
            {
                false
            }
            Some(entry) => {
                *entry = (hash, pred.ts_ms);
                true
            }
            None => {
                self.last_key.insert(pred.market_cid.clone(), (hash, pred.ts_ms));
                true
            }
        }
    }

        /// Forgets the state: each daily file path starts with the full row of
        /// each market (self-contained) and the mapping does not grow beyond one day.
    fn restart(&mut self) {
        self.last_key.clear();
    }
}

fn run_writer_model(
    dir: PathBuf,
    rx: mpsc::Receiver<ModelPrediction>,
    counters: Arc<ModelCounters>,
) -> u64 {
    let mut rx = rx;
    let mut written = 0u64;
    let mut file: Option<(String, BufWriter<File>)> = None;
    let mut dedup = DedupRecord::default();
    loop {
        let first = match rx.blocking_recv() {
            Some(p) => p,
            None => break,
        };
        let mut batch = vec![first];
        while batch.len() < 500 {
            match rx.try_recv() {
                Ok(p) => batch.push(p),
                Err(_) => break,
            }
        }
        let day = crate::time::now().format("%Y%m%d").to_string();
        let path = dir.join(format!("model-{day}.jsonl"));
        if file.as_ref().map(|(d, _)| d != &day).unwrap_or(true) {
            dedup.restart();
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f) => file = Some((day, BufWriter::new(f))),
                Err(e) => {
                    tracing::error!(error = %e, path = %path.display(),
                        "model_tool: could not open the record file path");
                    file = None;
                }
            }
        }
        if let Some((_, w)) = file.as_mut() {
            for pred in &batch {
                if !dedup.must_write(pred) {
                    metrics::counter!("hrf_model_tool_dedup_total").increment(1);
                    continue;
                }
                match serde_json::to_writer(&mut *w, pred) {
                    Ok(()) => {
                        let _ = w.write_all(b"\n");
                        written += 1;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "model_tool: prediction not serializable");
                    }
                }
            }
            let _ = w.flush();
        } else {
            let lost = batch.len() as u64;
            counters.discarded.fetch_add(lost, Ordering::Relaxed);
            metrics::counter!("hrf_model_tool_discarded_total").increment(lost);
        }
    }
    if let Some((_, mut w)) = file {
        let _ = w.flush();
    }
    written
}

// ── Context + runner of the tool ─────────────────────────────────────────────

/// Context of the `ModelTool` tool. Same principle as `ShadowCtx`: no
/// `intent_tx`, no `risk` or `state` — the tool **cannot emit orders nor
/// reserve cash by construction**.
pub struct ModelToolCtx {
    pub book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    pub vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    pub shutdown: watch::Receiver<bool>,
    pub max_book_age: Duration,
        /// Broadcast channel toward the consuming strategy. The tool emits here
        /// each `ModelPrediction` (probability or abstention); the strategy is
        /// the one that decides whether to translate it into an `OrderIntent`.
    pub prediction_tx: broadcast::Sender<ModelPrediction>,
}

/// Tool runner. Consumes the book broadcast, keeps the spot series
/// per slug and a per-cid throttle, evaluates the model (`Predictor::predict`) and
/// emits a `ModelPrediction` per book that passes the staleness filters.
/// **Never** writes to `state.rs`, **never** sends via `intent_tx`.
pub async fn run_model_tool(
    predictor: Arc<dyn Predictor>,
    cfg: ModelToolConfig,
    mut ctx: ModelToolCtx,
    producer: ModelProducer,
) {
    let model_hash = predictor.model_hash().to_string();
    tracing::info!(
        model_hash = %model_hash,
        budget_ms = cfg.effective_budget_ms(),
        "model_tool: starting (probabilities ONLY — NEVER emits orders)"
    );

    let budget_ms = cfg.effective_budget_ms();
    let mut series: HashMap<String, SpotSeries> = HashMap::new();
    let mut last_eval: HashMap<String, i64> = HashMap::new();
    let mut stats_tick = tokio::time::interval(Duration::from_secs(cfg.stats_interval_secs.max(5)));
    stats_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    stats_tick.tick().await; // discard the immediate tick

    let mut n_predictions = 0u64;
    let mut n_abstentions = 0u64;

    loop {
        let books = tokio::select! {
            books = crate::strategy::drain_latest(&mut ctx.book_rx) => match books {
                Some(v) => v,
                None => {
                    tracing::warn!("model_tool: books channel closed, exiting");
                    break;
                }
            },
            _ = stats_tick.tick() => {
                tracing::debug!(
                    n_predictions,
                    n_abstentions,
                    discarded_total = producer.counters.discarded.load(Ordering::Relaxed),
                    "model_tool: stats"
                );
                continue;
            }
            _ = ctx.shutdown.changed() => {
                tracing::info!("model_tool: shutdown received");
                break;
            }
        };

        for book in books {
            if *ctx.shutdown.borrow() {
                break;
            }
            let now_ms = crate::time::now().timestamp_millis();

            // Per-market throttle
            if cfg.min_decision_interval_ms > 0 {
                if let Some(&ult) = last_eval.get(&book.market_condition_id) {
                    if now_ms - ult < cfg.min_decision_interval_ms as i64 {
                        continue;
                    }
                }
                last_eval.insert(book.market_condition_id.clone(), now_ms);
            }

            // Book staleness (same criterion as `ShadowCtx`)
            let book_age_ms = book
                .recv_instant
                .and_then(|r| crate::time::Instant::now().checked_duration_since(r))
                .map(|d| d.as_millis() as u64)
                .unwrap_or(u64::MAX);

            // Oracle staleness
            let asset = crate::oracle::asset_from_slug(&book.slug);
            let (vol_age_ms, spot_fresh) = match asset.as_ref() {
                Some(a) => match ctx.vol_rxs.get(a.as_str()) {
                    Some(rx) => {
                        let v = rx.borrow().clone();
                        let age = v.updated_at.elapsed().as_millis() as u64;
                        (age, !v.is_stale_for_signal())
                    }
                    None if ctx.vol_rxs.is_empty() => (0u64, true),
                    None => (u64::MAX, false),
                },
                None => (u64::MAX, false),
            };

            // Series spot
            let series_entry = series
                .entry(book.slug.clone())
                .or_insert_with(SpotSeries::new);
            let vol_for_features = match asset.as_ref() {
                Some(a) => match ctx.vol_rxs.get(a.as_str()) {
                    Some(rx) => rx.borrow().clone(),
                    None if ctx.vol_rxs.is_empty() => crate::types::VolSnapshot::default(),
                    None => crate::types::VolSnapshot::default(),
                },
                None => crate::types::VolSnapshot::default(),
            };
            let feats = extract_features(
                &book,
                &vol_for_features,
                series_entry,
                now_ms,
            );

            // Book staleness (gate)
            if book_age_ms == u64::MAX {
                let pred = ModelPrediction {
                    schema_version: MODEL_TOOL_SCHEMA_VERSION,
                    ts_ms: now_ms,
                    market_cid: book.market_condition_id.clone(),
                    slug: book.slug.clone(),
                    token_id_yes: book.yes.token_id.clone(),
                    token_id_no: book.no.token_id.clone(),
                    side: "buy_yes".into(), // no effective action — the abstention decides
                    p_model: None,
                    p_market: book.yes_mid().and_then(|d| rust_decimal::Decimal::try_from(d).ok())
                        .and_then(|d| f64::try_from(d).ok()),
                    book_age_ms,
                    spot_age_ms: vol_age_ms,
                    latency_ms: 0,
                    budget_ms,
                    abstention: Some("book_stale_no_recv_ts".into()),
                    model_hash: Some(model_hash.clone()),
                };
                n_abstentions += 1;
                metrics::counter!("hrf_model_tool_abstentions_total", "reason" => "book_stale_no_recv_ts")
                    .increment(1);
                producer.record(pred.clone());
                producer.broadcast(&ctx.prediction_tx, pred);
                continue;
            }
            if book_age_ms > ctx.max_book_age.as_millis() as u64 {
                let pred = ModelPrediction {
                    schema_version: MODEL_TOOL_SCHEMA_VERSION,
                    ts_ms: now_ms,
                    market_cid: book.market_condition_id.clone(),
                    slug: book.slug.clone(),
                    token_id_yes: book.yes.token_id.clone(),
                    token_id_no: book.no.token_id.clone(),
                    side: "buy_yes".into(),
                    p_model: None,
                    p_market: None,
                    book_age_ms,
                    spot_age_ms: vol_age_ms,
                    latency_ms: 0,
                    budget_ms,
                    abstention: Some("book_stale".into()),
                    model_hash: Some(model_hash.clone()),
                };
                n_abstentions += 1;
                metrics::counter!("hrf_model_tool_abstentions_total", "reason" => "book_stale")
                    .increment(1);
                producer.record(pred.clone());
                producer.broadcast(&ctx.prediction_tx, pred);
                continue;
            }
            if !spot_fresh {
                let pred = ModelPrediction {
                    schema_version: MODEL_TOOL_SCHEMA_VERSION,
                    ts_ms: now_ms,
                    market_cid: book.market_condition_id.clone(),
                    slug: book.slug.clone(),
                    token_id_yes: book.yes.token_id.clone(),
                    token_id_no: book.no.token_id.clone(),
                    side: "buy_yes".into(),
                    p_model: None,
                    p_market: None,
                    book_age_ms,
                    spot_age_ms: vol_age_ms,
                    latency_ms: 0,
                    budget_ms,
                    abstention: Some("spot_stale".into()),
                    model_hash: Some(model_hash.clone()),
                };
                n_abstentions += 1;
                metrics::counter!("hrf_model_tool_abstentions_total", "reason" => "spot_stale")
                    .increment(1);
                producer.record(pred.clone());
                producer.broadcast(&ctx.prediction_tx, pred);
                continue;
            }

            // Post-close
            if crate::time::now().timestamp() >= book.clear_at {
                let pred = ModelPrediction {
                    schema_version: MODEL_TOOL_SCHEMA_VERSION,
                    ts_ms: now_ms,
                    market_cid: book.market_condition_id.clone(),
                    slug: book.slug.clone(),
                    token_id_yes: book.yes.token_id.clone(),
                    token_id_no: book.no.token_id.clone(),
                    side: "buy_yes".into(),
                    p_model: None,
                    p_market: None,
                    book_age_ms,
                    spot_age_ms: vol_age_ms,
                    latency_ms: 0,
                    budget_ms,
                    abstention: Some("post_close".into()),
                    model_hash: Some(model_hash.clone()),
                };
                n_abstentions += 1;
                metrics::counter!("hrf_model_tool_abstentions_total", "reason" => "post_close")
                    .increment(1);
                producer.record(pred.clone());
                producer.broadcast(&ctx.prediction_tx, pred);
                continue;
            }

            // Prediction
            let p_model = match predictor.predict(&feats) {
                Ok(p) => Some(p),
                Err(e) => {
                    let pred = ModelPrediction {
                        schema_version: MODEL_TOOL_SCHEMA_VERSION,
                        ts_ms: now_ms,
                        market_cid: book.market_condition_id.clone(),
                        slug: book.slug.clone(),
                        token_id_yes: book.yes.token_id.clone(),
                        token_id_no: book.no.token_id.clone(),
                        side: "buy_yes".into(),
                        p_model: None,
                        p_market: book.yes_mid().and_then(|d| {
                            rust_decimal::Decimal::try_from(d).ok()
                        }).and_then(|d| f64::try_from(d).ok()),
                        book_age_ms,
                        spot_age_ms: vol_age_ms,
                        latency_ms: 0,
                        budget_ms,
                        abstention: Some(format!("model_unavailable:{e}")),
                        model_hash: Some(model_hash.clone()),
                    };
                    n_abstentions += 1;
                    metrics::counter!("hrf_model_tool_abstentions_total", "reason" => "model_unavailable")
                        .increment(1);
                    producer.record(pred.clone());
                    producer.broadcast(&ctx.prediction_tx, pred);
                    continue;
                }
            };

            // Latency recv→full evaluation
            let latency_ms = book_age_ms; // book_age is already recv→now

            if latency_ms > budget_ms {
                let pred = ModelPrediction {
                    schema_version: MODEL_TOOL_SCHEMA_VERSION,
                    ts_ms: now_ms,
                    market_cid: book.market_condition_id.clone(),
                    slug: book.slug.clone(),
                    token_id_yes: book.yes.token_id.clone(),
                    token_id_no: book.no.token_id.clone(),
                    side: "buy_yes".into(),
                    p_model,
                    p_market: book.yes_mid().and_then(|d| {
                        rust_decimal::Decimal::try_from(d).ok()
                    }).and_then(|d| f64::try_from(d).ok()),
                    book_age_ms,
                    spot_age_ms: vol_age_ms,
                    latency_ms,
                    budget_ms,
                    abstention: Some("latency_exceeded".into()),
                    model_hash: Some(model_hash.clone()),
                };
                n_abstentions += 1;
                metrics::counter!("hrf_model_tool_abstentions_total", "reason" => "latency_exceeded")
                    .increment(1);
                producer.record(pred.clone());
                producer.broadcast(&ctx.prediction_tx, pred);
                continue;
            }

            // Prediction OK
            n_predictions += 1;
            metrics::counter!("hrf_model_tool_predictions_total").increment(1);
            let pred = ModelPrediction {
                schema_version: MODEL_TOOL_SCHEMA_VERSION,
                ts_ms: now_ms,
                market_cid: book.market_condition_id.clone(),
                slug: book.slug.clone(),
                token_id_yes: book.yes.token_id.clone(),
                token_id_no: book.no.token_id.clone(),
                side: "buy_yes".into(), // the tool does not decide the side — the strategy does
                p_model,
                p_market: book.yes_mid().and_then(|d| {
                    rust_decimal::Decimal::try_from(d).ok()
                }).and_then(|d| f64::try_from(d).ok()),
                book_age_ms,
                spot_age_ms: vol_age_ms,
                latency_ms,
                budget_ms,
                abstention: None,
                model_hash: Some(model_hash.clone()),
            };
            producer.record(pred.clone());
            producer.broadcast(&ctx.prediction_tx, pred);
            metrics::histogram!("hrf_model_tool_latency_seconds")
                .record(latency_ms as f64 / 1_000.0);
            tokio::task::yield_now().await;
        }
    }

    tracing::info!(
        n_predictions,
        n_abstentions,
        "model_tool: runner finished"
    );
}

/// Resolves the absolute path of the artifact from the config. If the path is
/// relative it is interpreted relative to the process CWD (not to the
/// binary's manifest): this way the operator can override with an absolute path for the run without
/// recompiling.
pub fn artifact_path(cfg_artifact_dir: &str) -> PathBuf {
    let p = PathBuf::from(cfg_artifact_dir);
    if p.is_absolute() {
        p
    } else {
        // Resolve relative to the CWD; `cargo run` invokes it from the repo root
        // through the `fail-fast` of `main.rs`, which requires `config/` to be present.
        std::env::current_dir()
            .map(|cwd| cwd.join(&p))
            .unwrap_or(p)
    }
}

/// Loads the predictor trying the real artifact. If it fails, returns a
/// `NullPredictor` with `warn` — a `null` model is recorded as
/// `model_unavailable` in each abstention and the tool keeps emitting (no
/// we abort startup because of a missing artifact; the paper run can
/// continue with honest abstentions while the operator diagnoses it).
///
/// See `Predictor` for the contract; `ShadowModel::load` for the exact
/// validation criteria.
pub fn load_predictor(cfg_artifact_dir: &str) -> Arc<dyn Predictor> {
    let dir = artifact_path(cfg_artifact_dir);
    match ShadowModel::load(&dir) {
        Ok(m) => {
            tracing::info!(
                artifact = %dir.display(),
                hash = m.model_hash(),
                "model_tool: predictor loaded from artifact"
            );
            Arc::new(m)
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                artifact = %dir.display(),
                "model_tool: could not load the artifact — NULL predictor \
                 (all predictions will be `model_unavailable` abstentions)"
            );
            Arc::new(NullPredictor)
        }
    }
}

/// Tool handle: the tokio task of the runner + the JSONL writer.
/// `stop()` does a cooperative shutdown: the supervisor sends `true` to the
/// `stop_tx` of the runner (forwards the global kill, see
/// `ToolSupervisor::spawn_model_tool`), waits for the task with `STOP_TIMEOUT`,
/// and then `join_and_drain` consumes the `ToolHandle` (if the `runner`
/// inside still exists it waits for it; it always flushes the writer). The
/// `Take`/`Join` is done in two places: the real `JoinHandle` lives in
/// the supervisor's `running` slot (`is_finished` queries it), and the
/// writer stays inside the `ToolHandle` to drain it in order.
/// Orderly failure: a panic in the runner does not prevent draining the writer.
pub struct ToolHandle {
    runner: Option<tokio::task::JoinHandle<()>>,
    writer: Option<ModelWriter>,
}

impl ToolHandle {
        /// `JoinHandle` of the runner — so the supervisor can plug it into
        /// its `running` map and query `is_finished` in `is_running`.
        /// Returns a reference; the `JoinHandle` stays alive while the
        /// `ToolHandle` is still alive.
    pub fn runner_handle(&self) -> Option<&tokio::task::JoinHandle<()>> {
        self.runner.as_ref()
    }

        /// Consuming variant: extracts the runner's `JoinHandle` so that the
        /// supervisor inserts it into its `running` map while retaining the
        /// `ToolHandle` to drain the writer.
    pub fn take_runner(&mut self) -> Option<tokio::task::JoinHandle<()>> {
        self.runner.take()
    }

        /// Async consuming variant: waits for the runner (if it is still inside the
        /// `ToolHandle` — the supervisor usually extracted it via
        /// `take_runner`) and then drains the writer. Call it ONCE after the
        /// supervisor sends the stop signal to `stop_tx`.
    pub async fn join_and_drain(mut self) -> (u64, u64) {
        if let Some(mut runner) = self.runner.take() {
            let _ = (&mut runner).await;
        }
        if let Some(writer) = self.writer.take() {
            writer.close().await
        } else {
            (0, 0)
        }
    }

        /// Starts `run_model_tool` with its own `ModelWriter`, connected to the
        /// shared `prediction_tx` with the strategies. `stop_rx` is the
        /// signal from the supervisor (global tool-stop). The spawn fails only if the
        /// writer cannot start (I/O of the predictions directory);
        /// a missing predictor degrades to `NullPredictor` without aborting.
    pub fn spawn(
        cfg: ModelToolConfig,
        book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
        vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
        prediction_tx: broadcast::Sender<ModelPrediction>,
        stop_rx: watch::Receiver<bool>,
        max_book_age: Duration,
    ) -> Result<Self, String> {
        let writer = ModelWriter::start(
            &PathBuf::from(&cfg.predictions_dir),
            cfg.channel_capacity,
        )
        .map_err(|e| {
            format!(
                "model_tool: could not start the JSONL writer at `{}`: {e}",
                cfg.predictions_dir
            )
        })?;
        let producer = writer.producer();
        let predictor = load_predictor(&cfg.artifact_dir);
        let ctx = ModelToolCtx {
            book_rx,
            vol_rxs,
            shutdown: stop_rx,
            max_book_age,
            prediction_tx,
        };
        let runner = tokio::spawn(async move {
            run_model_tool(predictor, cfg, ctx, producer).await;
            // The writer's drain is done by `join_and_drain` after the
            // supervisor — the `ToolHandle` only exposes the final wait.
        });
        Ok(Self {
            runner: Some(runner),
            writer: Some(writer),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_budget_ms_explicit_vs_derived() {
        let mut cfg = ModelToolConfig::default();
        cfg.latency_budget_ms = 250;
        cfg.signal_horizon_ms = 5_000;
        assert_eq!(cfg.effective_budget_ms(), 250);

        cfg.latency_budget_ms = 0;
        cfg.signal_horizon_ms = 5_000;
        assert_eq!(cfg.effective_budget_ms(), 500);

        cfg.signal_horizon_ms = 5;
        assert_eq!(cfg.effective_budget_ms(), 1, "minimum 1 ms");
    }

    #[test]
    fn default_config_starts_off() {
        let cfg = ModelToolConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.signal_horizon_ms, 30_000);
        assert!(cfg.artifact_dir.contains("artifacts/btc5m"));
    }

    #[test]
    fn producer_sink_discards_everything() {
        let p = ModelProducer::sink();
        let ok = p.record(ModelPrediction {
            schema_version: MODEL_TOOL_SCHEMA_VERSION,
            ts_ms: 0,
            market_cid: "cid".into(),
            slug: "s".into(),
            token_id_yes: "y".into(),
            token_id_no: "n".into(),
            side: "buy_yes".into(),
            p_model: Some(0.5),
            p_market: Some(0.5),
            book_age_ms: 0,
            spot_age_ms: 0,
            latency_ms: 0,
            budget_ms: 0,
            abstention: None,
            model_hash: None,
        });
        assert!(!ok, "sink must discard (channel closed)");
    }

    fn make_abstention(cid: &str, ts_ms: i64, book_age_ms: u64) -> ModelPrediction {
        ModelPrediction {
            schema_version: MODEL_TOOL_SCHEMA_VERSION,
            ts_ms,
            market_cid: cid.into(),
            slug: "highest-temperature-in-nyc".into(),
            token_id_yes: "y".into(),
            token_id_no: "n".into(),
            side: "buy_yes".into(),
            p_model: None,
            p_market: None,
            book_age_ms,
            spot_age_ms: 10,
            latency_ms: 0,
            budget_ms: 500,
            abstention: Some("book_stale".into()),
            model_hash: Some("h".into()),
        }
    }

    #[test]
    fn dedup_discards_repeat_that_only_changes_ts_and_ages() {
        let mut d = DedupRecord::default();
        assert!(d.must_write(&make_abstention("a", 1_000, 5_000)), "first row for the cid");
        let mut rep = make_abstention("a", 2_000, 6_000);
        rep.spot_age_ms = 999;
        rep.latency_ms = 7;
        assert!(!d.must_write(&rep), "same content except ts/ages → duplicate");
    }

    #[test]
    fn dedup_writes_if_content_changes() {
        let mut d = DedupRecord::default();
        assert!(d.must_write(&make_abstention("a", 1_000, 0)));
        let mut p = make_abstention("a", 2_000, 0);
        p.abstention = None;
        p.p_model = Some(0.61);
        assert!(d.must_write(&p), "abstention change → written");
        let mut p2 = p.clone();
        p2.ts_ms = 3_000;
        p2.p_model = Some(0.62);
        assert!(d.must_write(&p2), "p_model change → written");
        let mut p3 = p2.clone();
        p3.ts_ms = 4_000;
        assert!(!d.must_write(&p3), "same p_model → duplicate");
    }

    #[test]
    fn dedup_is_per_market() {
        let mut d = DedupRecord::default();
        assert!(d.must_write(&make_abstention("a", 1_000, 0)));
        assert!(d.must_write(&make_abstention("b", 1_000, 0)), "another cid is independent");
        assert!(!d.must_write(&make_abstention("a", 1_500, 0)));
        assert!(!d.must_write(&make_abstention("b", 1_500, 0)));
    }

    #[test]
    fn dedup_rewrites_when_heartbeat_expires() {
        let mut d = DedupRecord::default();
        assert!(d.must_write(&make_abstention("a", 0, 0)));
        assert!(!d.must_write(&make_abstention("a", DEDUP_HEARTBEAT_MS - 1, 0)));
        assert!(
            d.must_write(&make_abstention("a", DEDUP_HEARTBEAT_MS, 0)),
            "heartbeat: the market records itself even if nothing changes"
        );
        assert!(!d.must_write(&make_abstention("a", DEDUP_HEARTBEAT_MS + 1, 0)), "the heartbeat is re-armed");
    }

    #[test]
    fn dedup_restart_forgets_state() {
        let mut d = DedupRecord::default();
        assert!(d.must_write(&make_abstention("a", 0, 0)));
        d.restart();
        assert!(d.must_write(&make_abstention("a", 1, 0)), "a new file starts complete");
    }

    #[tokio::test]
    async fn writer_does_not_write_duplicate_rows() {
        let dir = tempfile::tempdir().unwrap();
        let w = ModelWriter::start(dir.path(), 64).unwrap();
        let p = w.producer();
        for i in 0..10 {
            assert!(p.record(make_abstention("a", 1_000 + i, i as u64)));
        }
        let mut change = make_abstention("a", 2_000, 0);
        change.abstention = Some("spot_stale".into());
        assert!(p.record(change));
        drop(p);
        let (written_count, discarded_count) = w.close().await;
        assert_eq!((written_count, discarded_count), (2, 0));
        let file_path = fs::read_dir(dir.path()).unwrap().next().unwrap().unwrap().path();
        let lines = fs::read_to_string(file_path).unwrap().lines().count();
        assert_eq!(lines, 2, "1st row + abstention change");
    }
}

// Anchor imports that only appear in doc code / tests.
#[allow(dead_code)]
fn _unused_imports(_: DateTime<Utc>, _: Decimal) {}