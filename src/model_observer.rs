//! Shadow observer for the trained numeric model.
//!
//! Pure observation task: it consumes the `broadcast<BookSnapshot>` from
//! `market_feed` and the oracle's `watch<VolSnapshot>`, loads the versioned
//! trained-model artifact (`model.json` + `manifest.json`, see
//! `research_contract.rs`), produces a prediction and a **VIRTUAL decision**
//! that is recorded in JSONL — it is never sent to any order channel. The
//! [`ShadowCtx`] context does not even carry `intent_tx`, `risk` or `state`:
//! "the shadow generates no orders, reserves no cash and touches no
//! ledger/executor/RiskGuard" is a STRUCTURAL property of the kind, not a
//! behavior that depends on discipline in the code.
//!
//! ## Design
//!
//! - **Does not block the hot path**: the task is independent; the only channel
//!   it touches is its own towards the writer (bounded `try_send` — a
//!   record that does not fit is dropped and counted). Same pattern as
//!   `research_events.rs` / `spot_stream.rs`.
//! - **Research failures are isolated from the bot**: prediction errors
//!   are recorded as abstentions and the loop continues; the writer may
//!   fail without stopping anything. A panic in the task itself follows the
//!   common JoinSet policy (`main.rs`: the first task that ends triggers
//!   the bot's orderly shutdown) — same rule as
//!   `analytics`/`spot_stream`, not an exception.
//! - **Own JSONL record** (not the causal ledger of `research_events`):
//!   the contract's `EventKind` has no variant for trained-model predictions,
//!   and extending it would touch `research_contract.rs`, which is out of
//!   scope here. Shadow lines carry
//!   `features+prediction+thresholds+model_hash` — enough for the
//!   deterministic replay of the same record (same entry ⇒ same
//!   decision, weights frozen per run).
//! - **Latency budget derived from the signal horizon**:
//!   `effective_budget = latency_budget_ms` if declared, otherwise
//!   `signal_horizon_ms / 10`. The measured latency is the full recv→decision
//!   time (broadcast queue + features + prediction); exceeding the
//!   budget produces a recorded abstention, never a decision on
//!   dead prices.
//! - **No gate enables live**: the observer has no path to `OrderIntent` —
//!   no flag turns it operational. A gate to go live would be a new
//!   strategy built on the evidence of the shadow records, a separate manual decision.
//!
//! ## Features
//!
//! `extract_features` reproduces `dataset.py::_observation_from_state` from the
//! training pipeline (same formulas, same names). Honest divergences of the
//! live path, documented:
//! - `flow_buy_vol`/`flow_sell_vol`/`flow_imbalance` are derived from
//!   `TokenBook.flow_net`/`flow_total` of the YES token (window
//!   `market_feed.tape_window_secs` = 60s, like `FLOW_WINDOW_MS` in the
//!   dataset). Declared scope divergence: the dataset aggregated the
//!   tape by `market_cid` (YES+NO); the shadow only sees YES flow.
//!   `trades_count` does not exist live → `None` (imputed by the scaler).
//! - `spot_ret_*`/`spot_vol` are computed over a spot ticks series sampled
//!   at the decision cadence (the live path does not expose the raw tick
//!   tape; `spot_stream` records but does not feed live). Identical
//!   windows: `RET_WINDOW=5`, `VOL_WINDOW=10`.
//! - `origin_delay_ms` does not exist live (WS books carry no remote ts)
//!   → `None`; `quality_bits` = 0 (live books carry no flags).
//! - `own_open_orders`/`own_net_shares` = 0: the shadow never trades — its
//!   real "own state" is zero, not a simulated inventory.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, watch};

use crate::config::ModelShadowConfig;
use crate::types::{BookSnapshot, VolSnapshot};

// ── Feature pipeline constants (mirror of dataset.py) ────────────────

/// Aggregated depth levels for `*_depth_*` (`DEPTH_LEVELS`).
const DEPTH_LEVELS: usize = 5;
/// Long return window (`RET_WINDOW`).
const RET_WINDOW: usize = 5;
/// Realized volatility window (`VOL_WINDOW`): pstdev of the last
/// ≤10 log-returns of the spot ticks series.
const VOL_WINDOW: usize = 10;
/// Cap of the spot ticks series per asset (only the last
/// VOL_WINDOW+1 points are read — the rest is unused history).
const SPOT_SERIES_CAP: usize = 64;
/// Cap of the latency ring used for the p50/p95/p99 percentiles.
const LATENCIES_CAP: usize = 4_096;
/// Maximum number of lines the writer drains per batch before flushing.
const BATCH_MAX: usize = 500;
/// Maximum age of the per-market state (`ultima_eval` per cid,
/// `series` per slug) before it is evicted — bounds the maps to the live
/// markets even when the bot runs for days with bracket rotation (up/down
/// brackets live minutes/hours; 1h of silence = dead market).
const STATE_EVICT_MS: i64 = 3_600_000;

/// Schema version of the shadow lines (independent of the `research_contract`
/// contract — this observer's own format).
pub const SHADOW_SCHEMA_VERSION: u32 = 1;

/// Features the live extractor can produce. The loader requires that ALL
/// `feature_names` of the artifact be listed here: an artifact trained on
/// features this binary cannot compute is rejected (strict versioning — a
/// silent None imputed by the median would hide schema drift).
pub const SUPPORTED_FEATURES: &[&str] = &[
    "yes_bid",
    "yes_ask",
    "yes_mid",
    "yes_spread",
    "yes_depth_bid",
    "yes_depth_ask",
    "yes_imbalance",
    "yes_microprice",
    "no_bid",
    "no_ask",
    "no_mid",
    "no_spread",
    "no_imbalance",
    "parity_ask_sum",
    "flow_buy_vol",
    "flow_sell_vol",
    "flow_imbalance",
    "trades_count",
    "spot",
    "spot_ret_1",
    "spot_ret_n",
    "spot_vol",
    "moneyness",
    "time_left_ms",
    "time_left_frac",
    "book_age_ms",
    "spot_age_ms",
    "origin_delay_ms",
    "quality_bits",
    "own_open_orders",
    "own_net_shares",
];

/// Feature map of one observation (name → value; `None` = missing,
/// imputed by the scaler's median in `predict`, as in
/// `FeatureScaler.transform` of the Python pipeline).
pub type FeatureMap = HashMap<String, Option<f64>>;

// ── SHA-256 (pure std — manifest verification) ───────────────────────────────
//
// `sha2` cannot be added to Cargo.toml without breaking `--locked` (the lockfile
// records the workspace deps), and no direct dependency exposes a general-purpose
// SHA-256 hasher (rustls only fixes the TLS provider; `hex` only encodes).
// Own implementation, verified against the RFC 4634 test vectors in
// `tests/model_shadow.rs`.

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// SHA-256 in hex — used to verify the artifact's `ManifestFile.sha256`.
pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(sha256_digest(data))
}

// ── Loading the versioned artifact ────────────────────────────────────────────

/// Errors loading/validating the artifact. Any of them prevents
/// prediction: a wrong schema or broken integrity is NEVER turned into
/// silent abstentions through imputation — the trained model does not load.
#[derive(Debug)]
pub enum ArtifactError {
    /// I/O reading `manifest.json`/`model.json`.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// Invalid JSON or wrong manifest/model schema.
    Schema { detail: String },
    /// The actual sha256 of `model.json` does not match the one declared in the
    /// manifest — modified artifact or corruption.
    HashMismatch { expected: String, real: String },
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Schema { detail } => write!(f, "invalid artifact: {detail}"),
            Self::HashMismatch { expected, real } => write!(
                f,
                "sha256 of model.json does not match the manifest (expected {expected}, real {real})"
            ),
        }
    }
}

impl std::error::Error for ArtifactError {}

/// Hot-path prediction failure. The observer turns it into a recorded
/// `model_error` abstention — never into a decision.
#[derive(Debug)]
pub enum PredictionError {
    /// The predictor could not produce output (worker down, invalid input…).
    ModelUnavailable(String),
}

impl std::fmt::Display for PredictionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelUnavailable(m) => write!(f, "trained model unavailable: {m}"),
        }
    }
}

impl std::error::Error for PredictionError {}

/// Shadow predictor — minimal abstraction so tests can inject slow or failing
/// models without touching the real artifact.
pub trait Predictor: Send + Sync {
    /// P(YES) ∈ [0,1] over the map of raw features (unscaled — the
    /// predictor applies its internal scaler with median imputation).
    fn predict(&self, feats: &FeatureMap) -> Result<f64, PredictionError>;
    /// Artifact hash (sha256 of `model.json` verified against the
    /// manifest) — freezes the weights per run and travels in every record.
    fn model_hash(&self) -> &str;
}

/// Logistic model (`kind: "logistic_stdlib_baseline"`): z-score with the
/// scaler's means/stds + median imputation, sigmoid of
/// `bias + w·z`. Weights frozen since loading — never re-fitted.
pub struct ShadowModel {
    feature_names: Vec<String>,
    weights: Vec<f64>,
    bias: f64,
    means: HashMap<String, f64>,
    stds: HashMap<String, f64>,
    medians: HashMap<String, f64>,
    model_hash: String,
}

/// Artifact sub-documents (strict deserialization to `Value` first
/// so the versioning error messages can be controlled).
#[derive(Deserialize)]
struct ManifestDoc {
    schema_version: u32,
    #[serde(default)]
    files: Vec<ManifestFileDoc>,
}

#[derive(Deserialize)]
struct ManifestFileDoc {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct ModelDoc {
    schema_version: u32,
    kind: String,
    feature_names: Vec<String>,
    model: ModelParamsDoc,
    scaler: ScalerDoc,
}

#[derive(Deserialize)]
struct ModelParamsDoc {
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Deserialize)]
struct ScalerDoc {
    means: HashMap<String, f64>,
    stds: HashMap<String, f64>,
    medians: HashMap<String, f64>,
}

impl ShadowModel {
    /// Loads and validates the versioned artifact in `dir`
    /// (`manifest.json` + `model.json`). Rejects:
    /// - `schema_version` of the manifest or of the model ≠ 1;
    /// - manifest without a `model.json` entry, or non-matching sha256;
    /// - `kind` other than `logistic_stdlib_baseline`;
    /// - `weights.len() != feature_names.len()`;
    /// - features without an entry in `means`/`stds`/`medians`;
    /// - feature_names not supported by `extract_features`.
    pub fn load(dir: &Path) -> Result<Self, ArtifactError> {
        let read = |name: &str| -> Result<Vec<u8>, ArtifactError> {
            let path = dir.join(name);
            fs::read(&path).map_err(|source| ArtifactError::Io { path, source })
        };
        let manifest_raw = read("manifest.json")?;
        let manifest: ManifestDoc =
            serde_json::from_slice(&manifest_raw).map_err(|e| ArtifactError::Schema {
                detail: format!("manifest.json does not parse: {e}"),
            })?;
        if manifest.schema_version != crate::research_contract::SCHEMA_VERSION {
            return Err(ArtifactError::Schema {
                detail: format!(
                    "manifest schema_version {} != {}",
                    manifest.schema_version,
                    crate::research_contract::SCHEMA_VERSION
                ),
            });
        }
        let entry = manifest
            .files
            .iter()
            .find(|f| f.path == "model.json")
            .ok_or_else(|| ArtifactError::Schema {
                detail: "manifest has no entry for model.json".into(),
            })?;
        let model_raw = read("model.json")?;
        let real = sha256_hex(&model_raw);
        if real != entry.sha256 {
            return Err(ArtifactError::HashMismatch {
                expected: entry.sha256.clone(),
                real,
            });
        }
        let doc: ModelDoc =
            serde_json::from_slice(&model_raw).map_err(|e| ArtifactError::Schema {
                detail: format!("model.json does not parse: {e}"),
            })?;
        if doc.schema_version != crate::research_contract::SCHEMA_VERSION {
            return Err(ArtifactError::Schema {
                detail: format!("model schema_version {} != 1", doc.schema_version),
            });
        }
        if doc.kind != "logistic_stdlib_baseline" {
            return Err(ArtifactError::Schema {
                detail: format!("unknown model kind: {}", doc.kind),
            });
        }
        if doc.feature_names.is_empty() || doc.model.weights.len() != doc.feature_names.len() {
            return Err(ArtifactError::Schema {
                detail: format!(
                    "weights ({}) != feature_names ({})",
                    doc.model.weights.len(),
                    doc.feature_names.len()
                ),
            });
        }
        for name in &doc.feature_names {
            if !SUPPORTED_FEATURES.contains(&name.as_str()) {
                return Err(ArtifactError::Schema {
                    detail: format!("feature not supported by the live extractor: {name}"),
                });
            }
            for (field, map) in [
                ("means", &doc.scaler.means),
                ("stds", &doc.scaler.stds),
                ("medians", &doc.scaler.medians),
            ] {
                if !map.contains_key(name) {
                    return Err(ArtifactError::Schema {
                        detail: format!("scaler.{field} has no entry for {name}"),
                    });
                }
            }
        }
        Ok(Self {
            feature_names: doc.feature_names,
            weights: doc.model.weights,
            bias: doc.model.bias,
            means: doc.scaler.means,
            stds: doc.scaler.stds,
            medians: doc.scaler.medians,
            model_hash: real,
        })
    }

    /// Number of features in the artifact.
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }
}

impl Predictor for ShadowModel {
    /// Same math as `train.py::logistic_predict_proba` +
    /// `FeatureScaler.transform`: median imputation, z-score with
    /// `std==0 → 1`, sigmoid.
    fn predict(&self, feats: &FeatureMap) -> Result<f64, PredictionError> {
        let mut logit = self.bias;
        for (i, name) in self.feature_names.iter().enumerate() {
            let v = feats
                .get(name)
                .copied()
                .flatten()
                .unwrap_or_else(|| self.medians[name]);
            let std = match self.stds[name] {
                s if s != 0.0 => s,
                _ => 1.0,
            };
            logit += self.weights[i] * (v - self.means[name]) / std;
        }
        Ok(1.0 / (1.0 + (-logit).exp()))
    }

    fn model_hash(&self) -> &str {
        &self.model_hash
    }
}

// ── Feature extraction (mirror of dataset.py) ────────────────────────────

/// Causal spot series per asset (estimated ts_ms, price). Fed by the
/// observer at the decision cadence from `VolSnapshot.last_price`.
pub struct SpotSeries {
    points: VecDeque<(i64, f64)>,
}

impl SpotSeries {
    pub fn new() -> Self {
        Self {
            points: VecDeque::with_capacity(SPOT_SERIES_CAP),
        }
    }

    /// Records a sample (estimated tick ts, price).
    pub fn record(&mut self, ts_ms: i64, price: f64) {
        self.points.push_back((ts_ms, price));
        while self.points.len() > SPOT_SERIES_CAP {
            self.points.pop_front();
        }
    }

    /// ts_ms of the last sample (used to evict dead series).
    fn last_ts_ms(&self) -> Option<i64> {
        self.points.back().map(|(t, _)| *t)
    }
}

impl Default for SpotSeries {
    fn default() -> Self {
        Self::new()
    }
}

fn to_f64(d: Decimal) -> Option<f64> {
    d.try_into().ok()
}

/// Features of one book side — mirror of `dataset.py::_book_features`.
/// `None` for each metric whose input is missing (empty book, a single side…).
fn book_features(bids: &crate::types::HalfBook, asks: &crate::types::HalfBook) -> [Option<f64>; 8] {
    let bid = bids.best_bid().and_then(to_f64);
    let ask = asks.best_ask().and_then(to_f64);
    let depth_bid = if bids.levels.is_empty() {
        None
    } else {
        bids.levels
            .iter()
            .take(DEPTH_LEVELS)
            .map(|l| to_f64(l.size))
            .collect::<Option<Vec<_>>>()
            .map(|v| v.iter().sum())
    };
    let depth_ask = if asks.levels.is_empty() {
        None
    } else {
        asks.levels
            .iter()
            .take(DEPTH_LEVELS)
            .map(|l| to_f64(l.size))
            .collect::<Option<Vec<_>>>()
            .map(|v| v.iter().sum())
    };
    let mid = bid.zip(ask).map(|(b, a)| (b + a) / 2.0);
    let spread = bid.zip(ask).map(|(b, a)| a - b);
    let imbalance = match (depth_bid, depth_ask) {
        (Some(db), Some(gives)) if db + gives > 0.0 => Some((db - gives) / (db + gives)),
        _ => None,
    };
    let microprice = match (bid, ask, bids.best_size(), asks.best_size()) {
        (Some(b), Some(a), Some(bs), Some(as_)) => {
            let bs = to_f64(bs).unwrap_or(0.0);
            let as_ = to_f64(as_).unwrap_or(0.0);
            if bs + as_ > 0.0 {
                Some((a * bs + b * as_) / (bs + as_))
            } else {
                mid
            }
        }
        _ => None,
    };
    [
        bid, ask, mid, spread, depth_bid, depth_ask, imbalance, microprice,
    ]
}

/// Underlying features from the spot ticks series — mirror of
/// `dataset.py::_spot_features` (only the last VOL_WINDOW+1 points are read).
fn spot_features(series: &SpotSeries, now_ms: i64) -> [Option<f64>; 5] {
    if series.points.is_empty() {
        return [None; 5];
    }
    let n = series.points.len();
    let ini = n.saturating_sub(VOL_WINDOW + 1);
    let prices: Vec<f64> = series.points.iter().skip(ini).map(|(_, p)| *p).collect();
    let spot = *prices.last().unwrap();
    let age_ms = (now_ms - series.points.back().unwrap().0) as f64;
    let ret_1 =
        (prices.len() >= 2).then(|| (prices[prices.len() - 1] / prices[prices.len() - 2]).ln());
    // `spot_ret_n` = return over EXACTLY RET_WINDOW ticks, as in
    // `dataset.py::_spot_features` (`prices[-1] / prices[-1-RET_WINDOW]`).
    // NOT `prices[0]`: with len > RET_WINDOW+1 the oldest (stalest) point in the buffer
    // would inflate the return and the resulting z-score would bias the prediction
    // (a silent divergence from training — a bug found in review).
    let ret_n = (prices.len() > RET_WINDOW)
        .then(|| (prices[prices.len() - 1] / prices[prices.len() - 1 - RET_WINDOW]).ln());
    let rets: Vec<f64> = (1..prices.len())
        .filter(|&i| prices[i - 1] > 0.0)
        .map(|i| (prices[i] / prices[i - 1]).ln())
        .collect();
    let vol = if rets.len() >= 2 {
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
        Some(var.sqrt())
    } else {
        None
    };
    [Some(spot), ret_1, ret_n, vol, Some(age_ms)]
}

/// Builds the feature map of a shadow observation — mirror of
/// `dataset.py::_observation_from_state` over the live `BookSnapshot`.
/// `now_ms` is the decision instant (equivalent to `avail_ts_ms` of the
/// offline dataset). `series` holds the asset's spot history.
///
/// Fields that do not exist on the live path stay `None` (imputed by
/// the scaler's median in `predict`, never invented): `trades_count`,
/// `origin_delay_ms`. `quality_bits` = 0 (live books carry no contract
/// flags) and `own_*` = 0 (the shadow does not trade — its real own state
/// is zero).
pub fn extract_features(
    book: &BookSnapshot,
    vol: &VolSnapshot,
    series: &mut SpotSeries,
    now_ms: i64,
) -> FeatureMap {
    let [yes_bid, yes_ask, yes_mid, yes_spread, yes_depth_bid, yes_depth_ask, yes_imbalance, micro_y] =
        book_features(&book.yes.bids, &book.yes.asks);
    let [no_bid, no_ask, no_mid, no_spread, _db_no, _gives_no, no_imbalance, _micro_no] =
        book_features(&book.no.bids, &book.no.asks);
    let parity = yes_ask.zip(no_ask).map(|(y, n)| y + n);

    // Flow: dataset.py splits buy/sell from the tape over FLOW_WINDOW_MS and
    // aggregates it by market_cid — BOTH tokens (YES+NO). Live, only the
    // aggregate of the YES token (`book.yes.flow_*`) is available: declared
    // scope divergence — in a binary market the sell of YES is usually the
    // buy of NO, so the net YES is a reasonable approximation of the
    // net of the market, but it is NOT identical to the training feature.
    // From the aggregate: buy = (total+net)/2, sell = (total−net)/2 (same
    // algebraic identity as dataset.py).
    let (buy_vol, sell_vol, flow_imb) = match (book.yes.flow_total, book.yes.flow_net) {
        (Some(tot), Some(net)) => {
            let t = to_f64(tot).unwrap_or(0.0);
            let n = to_f64(net).unwrap_or(0.0);
            let imb = (t > 0.0).then_some(n / t);
            (Some((t + n) / 2.0), Some((t - n) / 2.0), imb)
        }
        _ => (Some(0.0), Some(0.0), None),
    };

    // Spot: push the sample into the ticks series BEFORE computing returns, with the
    // estimated ts of the oracle's last tick (now − VolSnapshot age).
    let spot_age_ms = vol.updated_at.elapsed().as_millis() as i64;
    let spot_price = to_f64(vol.last_price).filter(|p| *p > 0.0);
    if let Some(p) = spot_price {
        series.record(now_ms - spot_age_ms, p);
    }
    let [spot, ret_1, ret_n, spot_vol, _age] = spot_features(series, now_ms);

    let time_left_ms = book.clear_at.saturating_mul(1_000) - now_ms;
    let lifetime_ms = (book.clear_at - book.market_open_ts).saturating_mul(1_000);
    let time_left_frac =
        (lifetime_ms > 0).then(|| (time_left_ms as f64 / lifetime_ms as f64).clamp(0.0, 1.0));
    let moneyness = match (spot, book.reference_price.and_then(to_f64)) {
        (Some(s), Some(r)) if r > 0.0 => Some((s / r).ln()),
        _ => None,
    };
    let book_age_ms = book.recv_instant.map(|r| r.elapsed().as_millis() as f64);

    let mut f = FeatureMap::with_capacity(SUPPORTED_FEATURES.len());
    f.insert("yes_bid".into(), yes_bid);
    f.insert("yes_ask".into(), yes_ask);
    f.insert("yes_mid".into(), yes_mid);
    f.insert("yes_spread".into(), yes_spread);
    f.insert("yes_depth_bid".into(), yes_depth_bid);
    f.insert("yes_depth_ask".into(), yes_depth_ask);
    f.insert("yes_imbalance".into(), yes_imbalance);
    f.insert("yes_microprice".into(), micro_y);
    f.insert("no_bid".into(), no_bid);
    f.insert("no_ask".into(), no_ask);
    f.insert("no_mid".into(), no_mid);
    f.insert("no_spread".into(), no_spread);
    f.insert("no_imbalance".into(), no_imbalance);
    f.insert("parity_ask_sum".into(), parity);
    f.insert("flow_buy_vol".into(), buy_vol);
    f.insert("flow_sell_vol".into(), sell_vol);
    f.insert("flow_imbalance".into(), flow_imb);
    f.insert("trades_count".into(), None); // no tape on the live path
    f.insert("spot".into(), spot);
    f.insert("spot_ret_1".into(), ret_1);
    f.insert("spot_ret_n".into(), ret_n);
    f.insert("spot_vol".into(), spot_vol);
    f.insert("moneyness".into(), moneyness);
    f.insert("time_left_ms".into(), Some(time_left_ms as f64));
    f.insert("time_left_frac".into(), time_left_frac);
    f.insert("book_age_ms".into(), book_age_ms);
    f.insert("spot_age_ms".into(), Some(spot_age_ms as f64));
    f.insert("origin_delay_ms".into(), None); // no remote ts live
    f.insert("quality_bits".into(), Some(0.0));
    f.insert("own_open_orders".into(), Some(0.0)); // the shadow never trades
    f.insert("own_net_shares".into(), Some(0.0));
    f
}

// ── Records (JSONL lines) ─────────────────────────────────────────────────

/// Shadow record line. Three kinds:
/// - `decision`: virtual decision (buy YES/NO at ask, taker) with
///   features + prediction + thresholds — comparable against a replay of the
///   same record (weights frozen via `model_hash`).
/// - `make_abstention`: evaluation without action — explicit reason, never
///   silent (`book_stale`, `spot_stale`, `latency_exceeded`,
///   `model_error`, `no_edge`, `post_close`, `absent_vol`).
/// - `stats`: periodic snapshot of p50/p95/p99 latencies and counters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ShadowLine {
    Decision {
        schema_version: u32,
        ts_ms: i64,
        market_cid: String,
        slug: String,
        token_id: String,
        /// "buy_yes" | "buy_no" — the VIRTUAL action (never reaches any channel).
        side: String,
        ask_price: Decimal,
        shares: Decimal,
        /// Estimated parabolic taker fee over the virtual buy
        /// (the market's actual rate if the book provides it).
        fee_est: Decimal,
        /// Breakeven price: ask + fee/share (convention of
        /// `fair_value::breakeven_price_parabolic`).
        breakeven: Decimal,
        edge: Decimal,
        edge_min: Decimal,
        p_model: f64,
        /// Market baseline (yes_mid) recorded for comparison.
        p_market: Option<f64>,
        features: FeatureMap,
        /// Measured recv→decision latency (ms) and budget applied.
        latency_ms: u64,
        budget_ms: u64,
        signal_horizon_ms: u64,
        /// Estimated book reception time in ms epoch (now − age).
        book_recv_ms: Option<i64>,
        model_hash: String,
        post_close: bool,
    },
    Abstention {
        schema_version: u32,
        ts_ms: i64,
        market_cid: String,
        slug: String,
        reason: String,
        /// Prediction, if one was produced (None if the blocker came earlier).
        p_model: Option<f64>,
        p_market: Option<f64>,
        features: Option<FeatureMap>,
        /// Meaning depends on `reason` (mixed on purpose — the reason
        /// disambiguates it in the replay): for `latency_exceeded`/`no_edge` it is
        /// the full recv→evaluation latency; for `book_stale` it is the
        /// AGE of the book (the latency that triggered the guard); in blocks
        /// before reception (`without_recv_ts`, `post_close`,
        /// `absent_vol`, `spot_stale`, `model_error`) it is 0.
        latency_ms: u64,
        budget_ms: u64,
        model_hash: String,
    },
    Stats {
        schema_version: u32,
        ts_ms: i64,
        n_decisions: u64,
        n_abstentions: u64,
        p50_ms: u64,
        p95_ms: u64,
        p99_ms: u64,
        /// Records lost to channel saturation or a failed writer.
        discarded: u64,
    },
}

/// Stats line for the producer (also used in tests).
pub fn stats_line(
    n_decisions: u64,
    n_abstentions: u64,
    p50_ms: u64,
    p95_ms: u64,
    p99_ms: u64,
    discarded: u64,
) -> ShadowLine {
    ShadowLine::Stats {
        schema_version: SHADOW_SCHEMA_VERSION,
        ts_ms: crate::time::now().timestamp_millis(),
        n_decisions,
        n_abstentions,
        p50_ms,
        p95_ms,
        p99_ms,
        discarded,
    }
}

// ── Writer (dedicated OS thread, research_events pattern) ──────────────────────

#[derive(Debug, Default)]
struct ShadowCounters {
    enqueued: AtomicU64,
    discarded: AtomicU64,
}

/// Non-blocking, clonable handle to the writer. Pure `try_send`: a
/// record that does not fit is dropped and counted — research never slows
/// the observer or, by extension, the bot.
#[derive(Clone)]
pub struct ShadowProducer {
    tx: mpsc::Sender<ShadowLine>,
    counters: Arc<ShadowCounters>,
}

impl ShadowProducer {
    /// Enqueues a line. `false` if it was dropped (channel full or writer
    /// down) — counted in `discarded_total`.
    pub fn record(&self, line: ShadowLine) -> bool {
        match self.tx.try_send(line) {
            Ok(()) => {
                self.counters.enqueued.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => {
                self.counters.discarded.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("hrf_model_shadow_discarded_total").increment(1);
                false
            }
        }
    }

    /// Sink producer (tests / no writer): everything is dropped.
    pub fn sink() -> Self {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        Self {
            tx,
            counters: Arc::new(ShadowCounters::default()),
        }
    }
}

/// Totals returned by [`ShadowWriter::close`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShadowTotals {
    pub enqueued: u64,
    pub written: u64,
    pub discarded: u64,
}

/// JSONL writer for shadow records: dedicated OS thread, append-only, one
/// file per UTC day (`<dir>/shadow-<yyyymmdd>.jsonl`). Kept outside the active
/// DB and analytics — same rule as `research_events`.
pub struct ShadowWriter {
    tx: mpsc::Sender<ShadowLine>,
    counters: Arc<ShadowCounters>,
    writer: Option<JoinHandle<u64>>,
}

impl ShadowWriter {
    /// Creates `dir` if needed and starts the writer thread.
    pub fn start(dir: &Path, capacity: usize) -> std::io::Result<Self> {
        fs::create_dir_all(dir)?;
        let (tx, rx) = mpsc::channel(capacity.max(1));
        let counters = Arc::new(ShadowCounters::default());
        let dir = dir.to_path_buf();
        let writer_counters = counters.clone();
        let writer = std::thread::Builder::new()
            .name("model_shadow_writer".to_string())
            .spawn(move || run_writer_shadow(dir, rx, writer_counters))?;
        Ok(Self {
            tx,
            counters,
            writer: Some(writer),
        })
    }

    pub fn producer(&self) -> ShadowProducer {
        ShadowProducer {
            tx: self.tx.clone(),
            counters: self.counters.clone(),
        }
    }

    /// Orderly close: drops the sender and waits for the thread (drains what
    /// is queued). A writer panic degrades to an error + partial counters —
    /// it never breaks the shutdown.
    pub async fn close(mut self) -> ShadowTotals {
        let writer = self.writer.take().expect("close() was already called");
        let counters = self.counters.clone();
        drop(self);
        let written = match tokio::task::spawn_blocking(move || writer.join()).await {
            Ok(Ok(n)) => n,
            _ => {
                tracing::error!("model_shadow: panic in the writer thread — partial close");
                0
            }
        };
        ShadowTotals {
            enqueued: counters.enqueued.load(Ordering::Relaxed),
            written,
            discarded: counters.discarded.load(Ordering::Relaxed),
        }
    }
}

fn run_writer_shadow(
    dir: PathBuf,
    rx: mpsc::Receiver<ShadowLine>,
    counters: Arc<ShadowCounters>,
) -> u64 {
    let mut rx = rx;
    let mut written = 0u64;
    let mut file: Option<(String, BufWriter<File>)> = None;
    loop {
        // First element (blocking) + bounded drain of the batch.
        let first = match rx.blocking_recv() {
            Some(l) => l,
            None => break,
        };
        let mut batch = vec![first];
        while batch.len() < BATCH_MAX {
            match rx.try_recv() {
                Ok(l) => batch.push(l),
                Err(_) => break,
            }
        }
        let day = crate::time::now().format("%Y%m%d").to_string();
        let path = dir.join(format!("shadow-{day}.jsonl"));
        if file.as_ref().map(|(d, _)| d != &day).unwrap_or(true) {
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f) => file = Some((day, BufWriter::new(f))),
                Err(e) => {
                    tracing::error!(error = %e, path = %path.display(),
                        "model_shadow: could not open the record file");
                    file = None;
                }
            }
        }
        if let Some((_, w)) = file.as_mut() {
            for line in &batch {
                match serde_json::to_writer(&mut *w, line) {
                    Ok(()) => {
                        let _ = w.write_all(b"\n");
                        written += 1;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "model_shadow: line not serializable");
                    }
                }
            }
            let _ = w.flush();
        } else {
            // No file open (the journal failed to open): the batch is lost —
            // count it in `discarded_total` so the closing record is honest
            // (it used to be invisible).
            let lost = batch.len() as u64;
            counters.discarded.fetch_add(lost, Ordering::Relaxed);
            metrics::counter!("hrf_model_shadow_discarded_total").increment(lost);
        }
    }
    if let Some((_, mut w)) = file {
        let _ = w.flush();
    }
    written
}

// ── Latency statistics ────────────────────────────────────────────────

/// Ring of latencies (ms) for p50/p95/p99 — bounded, last N measurements.
struct LatencyStats {
    window: VecDeque<u64>,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            window: VecDeque::with_capacity(LATENCIES_CAP),
        }
    }
    fn record(&mut self, ms: u64) {
        self.window.push_back(ms);
        while self.window.len() > LATENCIES_CAP {
            self.window.pop_front();
        }
    }
    /// Rank-based percentile over a sorted copy (p99 of 4096 is cheap).
    fn percentile(&self, p: f64) -> u64 {
        if self.window.is_empty() {
            return 0;
        }
        let mut v: Vec<u64> = self.window.iter().copied().collect();
        v.sort_unstable();
        let idx = ((v.len() - 1) as f64 * p / 100.0).round() as usize;
        v[idx.min(v.len() - 1)]
    }
}

// ── Virtual decision ─────────────────────────────────────────────────────────

/// Side of the virtual buy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VirtualSide {
    BuyYes,
    BuyNo,
}

impl VirtualSide {
    fn as_str(&self) -> &'static str {
        match self {
            Self::BuyYes => "buy_yes",
            Self::BuyNo => "buy_no",
        }
    }
}

/// Shadow decision rule (conservative, taker): virtual buy of the side whose
/// probability exceeds its ask by more than `edge_min`. When both sides
/// exceed the threshold (inconsistent books) the one with the larger edge is taken.
/// Returns `None` = `no_edge` abstention.
fn decide(
    p_model: f64,
    book: &BookSnapshot,
    edge_min: Decimal,
) -> Option<(VirtualSide, Decimal, Decimal)> {
    let p_dec = Decimal::from_f64_retain(p_model).unwrap_or(Decimal::ZERO);
    let candidates = [
        book.yes
            .best_ask()
            .map(|ask| (VirtualSide::BuyYes, ask, p_dec - ask)),
        book.no
            .best_ask()
            .map(|ask| (VirtualSide::BuyNo, ask, Decimal::ONE - p_dec - ask)),
    ];
    // Exact edge tie: `max_by` would return the LAST maximum, but Python's
    // `max()` (offline evaluation, `shadow_eval_t9.py`) returns the FIRST — the
    // first is kept for strict shadow↔replay comparability.
    let mut best: Option<(VirtualSide, Decimal, Decimal)> = None;
    for (side, ask, edge) in candidates.into_iter().flatten() {
        if edge > edge_min && best.as_ref().is_none_or(|(_, _, e)| edge > *e) {
            best = Some((side, ask, edge));
        }
    }
    best
}

// ── Runner ──────────────────────────────────────────────────────────────────

/// Observer context. Deliberately SMALLER than `ToolCtx`: no
/// `intent_tx` (it can never emit orders — a structural guarantee), no
/// `risk`/`state`/`ledger` (it reserves no cash and consults no gates), no
/// `scan_tx` (its record is its own).
pub struct ShadowCtx {
    pub book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    /// Vol per asset (slug prefix), same as `StrategyCtx`.
    pub vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    pub shutdown: watch::Receiver<bool>,
    /// Guard: older books are not evaluated (`book_stale` abstention).
    pub max_book_age: Duration,
}

/// Shadow observer loop. Spawn a single task; it exits on shutdown or when the
/// books channel closes. It never blocks the hot path: it drains the broadcast
/// with the same intra-cid coalescing as `strategy::run`, and its only output
/// channel is `try_send` to the writer.
pub async fn run_observer(
    predictor: Arc<dyn Predictor>,
    cfg: ModelShadowConfig,
    mut ctx: ShadowCtx,
    producer: ShadowProducer,
) {
    tracing::info!(
        model_hash = %predictor.model_hash(),
        budget_ms = cfg.effective_budget_ms(),
        "model_shadow: starting (pure observation — NEVER emits orders)"
    );

    let budget_ms = cfg.effective_budget_ms();
    let mut series: HashMap<String, SpotSeries> = HashMap::new();
    let mut last_eval: HashMap<String, i64> = HashMap::new();
    let mut latencies = LatencyStats::new();
    let mut n_decisions = 0u64;
    let mut n_abstentions = 0u64;

    let mut stats_tick = tokio::time::interval(Duration::from_secs(cfg.stats_interval_secs.max(5)));
    stats_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    stats_tick.tick().await; // the first `interval` tick is immediate — consume it

    macro_rules! abstain {
        ($book:expr, $reason:expr, $p:expr, $feats:expr, $lat:expr) => {{
            n_abstentions += 1;
            metrics::counter!("hrf_model_shadow_abstentions_total", "reason" => $reason)
                .increment(1);
            producer.record(ShadowLine::Abstention {
                schema_version: SHADOW_SCHEMA_VERSION,
                ts_ms: crate::time::now().timestamp_millis(),
                market_cid: $book.market_condition_id.clone(),
                slug: $book.slug.clone(),
                reason: $reason.to_string(),
                p_model: $p,
                p_market: $book.yes_mid().and_then(to_f64),
                features: $feats,
                latency_ms: $lat,
                budget_ms,
                model_hash: predictor.model_hash().to_string(),
            });
        }};
    }

    loop {
        let books = tokio::select! {
            books = crate::strategy::drain_latest(&mut ctx.book_rx) => match books {
                Some(v) => v,
                None => {
                    tracing::warn!("model_shadow: books channel closed, exiting");
                    break;
                }
            },
            _ = stats_tick.tick() => {
                // Evict per-market state: entries idle for longer than
                // STATE_EVICT_MS are dropped — the maps stay bounded to the
                // live markets despite bracket rotation.
                let now_evict = crate::time::now().timestamp_millis();
                last_eval.retain(|_, t| now_evict - *t <= STATE_EVICT_MS);
                series.retain(|_, s| {
                    s.last_ts_ms()
                        .is_some_and(|t| now_evict - t <= STATE_EVICT_MS)
                });
                producer.record(stats_line(
                    n_decisions,
                    n_abstentions,
                    latencies.percentile(50.0),
                    latencies.percentile(95.0),
                    latencies.percentile(99.0),
                    producer.counters.discarded.load(Ordering::Relaxed),
                ));
                continue;
            }
            _ = ctx.shutdown.changed() => {
                tracing::info!("model_shadow: shutdown received");
                break;
            }
        };

        for book in books {
            if *ctx.shutdown.borrow() {
                break;
            }
            let now_ms = crate::time::now().timestamp_millis();

            // Scope: declared markets/assets (empty = all). Out of scope = plain
            // skip (neither decision nor abstention — metric only).
            if !cfg.markets.is_empty()
                && !cfg.markets.iter().any(|m| m == &book.market_condition_id)
            {
                metrics::counter!("hrf_model_shadow_out_of_scope_total").increment(1);
                continue;
            }
            let asset = crate::oracle::asset_from_slug(&book.slug);
            if !cfg.assets.is_empty()
                && !asset
                    .as_ref()
                    .is_some_and(|a| cfg.assets.iter().any(|x| x == a))
            {
                metrics::counter!("hrf_model_shadow_out_of_scope_total").increment(1);
                continue;
            }

            // Per-market throttle: the decision cadence is bounded
            // (same rule as the 1s sampling of the analytics recorder).
            if cfg.min_decision_interval_ms > 0 {
                if let Some(&ult) = last_eval.get(&book.market_condition_id) {
                    if now_ms - ult < cfg.min_decision_interval_ms as i64 {
                        continue;
                    }
                }
                last_eval.insert(book.market_condition_id.clone(), now_ms);
            }

            // Book staleness guard — recorded abstention:
            // deciding on a stale book means deciding on dead prices.
            let book_age = book
                .recv_instant
                .and_then(|r| crate::time::Instant::now().checked_duration_since(r));
            match book_age {
                None => {
                    abstain!(&*book, "without_recv_ts", None, None, 0);
                    continue;
                }
                Some(age) if age > ctx.max_book_age => {
                    abstain!(&*book, "book_stale", None, None, age.as_millis() as u64);
                    continue;
                }
                _ => {}
            }

            // Post-close: the market has already resolved — the virtual decision has
            // no horizon (dataset marks `post_close`; here we abstain).
            if crate::time::now().timestamp() >= book.clear_at {
                abstain!(&*book, "post_close", None, None, 0);
                continue;
            }

            // Asset vol — no cross-asset fallback. Empty map
            // (tests / no oracle) → default; missing entry → abstention.
            let vol = match &asset {
                Some(a) => match ctx.vol_rxs.get(a.as_str()) {
                    Some(rx) => rx.borrow().clone(),
                    None if ctx.vol_rxs.is_empty() => VolSnapshot::default(),
                    None => {
                        abstain!(&*book, "absent_vol", None, None, 0);
                        continue;
                    }
                },
                None => {
                    abstain!(&*book, "absent_vol", None, None, 0);
                    continue;
                }
            };
            // Silent oracle → dead spot → abstention.
            if vol.is_stale_for_signal() {
                abstain!(&*book, "spot_stale", None, None, 0);
                continue;
            }

            let series = series
                .entry(book.slug.clone())
                .or_insert_with(SpotSeries::new);
            let feats = extract_features(&book, &vol, series, now_ms);

            let p_model = match predictor.predict(&feats) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, cid = %book.market_condition_id,
                        "model_shadow: prediction failed — abstention");
                    abstain!(&*book, "model_error", None, Some(feats), 0);
                    continue;
                }
            };

            // Latency budget: FULL recv→decision time (broadcast queue +
            // features + prediction). Derived from the signal horizon —
            // exceeding it abstains, it never decides on dead data.
            let latency_ms = book
                .recv_instant
                .and_then(|r| crate::time::Instant::now().checked_duration_since(r))
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            latencies.record(latency_ms);
            metrics::histogram!("hrf_model_shadow_latency_seconds")
                .record(latency_ms as f64 / 1_000.0);
            if latency_ms > budget_ms {
                abstain!(
                    &*book,
                    "latency_exceeded",
                    Some(p_model),
                    Some(feats),
                    latency_ms
                );
                continue;
            }

            match decide(p_model, book.as_ref(), cfg.edge_min) {
                Some((side, ask, edge)) => {
                    n_decisions += 1;
                    let rate =
                        book.parabolic_fee_rate_or(crate::fair_value::TAKER_FEE_RATE_PARABOLIC);
                    let fee_est = crate::fair_value::fee_taker_parabolic_with_rate(
                        rate,
                        ask,
                        cfg.shares_virtual,
                    );
                    let breakeven =
                        crate::fair_value::breakeven_price_parabolic(book.as_ref(), ask);
                    let token_id = match side {
                        VirtualSide::BuyYes => book.yes.token_id.clone(),
                        VirtualSide::BuyNo => book.no.token_id.clone(),
                    };
                    metrics::counter!("hrf_model_shadow_decisiones_total",
                        "side" => side.as_str())
                    .increment(1);
                    producer.record(ShadowLine::Decision {
                        schema_version: SHADOW_SCHEMA_VERSION,
                        ts_ms: now_ms,
                        market_cid: book.market_condition_id.clone(),
                        slug: book.slug.clone(),
                        token_id,
                        side: side.as_str().to_string(),
                        ask_price: ask,
                        shares: cfg.shares_virtual,
                        fee_est,
                        breakeven,
                        edge,
                        edge_min: cfg.edge_min,
                        p_model,
                        p_market: book.yes_mid().and_then(to_f64),
                        features: feats,
                        latency_ms,
                        budget_ms,
                        signal_horizon_ms: cfg.signal_horizon_ms,
                        book_recv_ms: book.recv_instant.map(|_| now_ms - latency_ms as i64),
                        model_hash: predictor.model_hash().to_string(),
                        post_close: false,
                    });
                }
                None => {
                    abstain!(&*book, "no_edge", Some(p_model), Some(feats), latency_ms);
                }
            }
            tokio::task::yield_now().await;
        }
    }

    // Final stats on exit — an honest closing record.
    producer.record(stats_line(
        n_decisions,
        n_abstentions,
        latencies.percentile(50.0),
        latencies.percentile(95.0),
        latencies.percentile(99.0),
        producer.counters.discarded.load(Ordering::Relaxed),
    ));
}
