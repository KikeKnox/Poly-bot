//! `RlPredictor` strategy — ONLINE RUNTIME, PAPER-ONLY.
//!
//! Operational runtime for the RL predictor contract. It consumes the `p_model`
//! probabilities of the frozen predictor (`ModelTool`) as an observation FEATURE —
//! never as the policy — and chooses actions from the 9-action contract space
//! through a linear softmax policy loaded from a `paper_approved` artifact
//! (`crate::rl::artifact`).
//!
//! ## Hot path vs. learning
//!
//! `on_book`/`on_fill`/`on_timer` NEVER compute gradients, do disk I/O or touch the
//! DB: inference reads an immutable `Arc<PolicySnapshot>` through a `watch`, episode
//! accounting is an in-memory virtual ledger (micros), and experience leaves through
//! a BOUNDED `mpsc` channel towards the learner task (`crate::rl::learner`). Under
//! pressure, whatever does not fit is dropped and counted — the bot's accounting
//! ledger does not depend on that channel.
//!
//! ## Accounting truth
//!
//! Own positions are rebuilt from `StateStore` at startup (recovery) and kept in an
//! `EpisodeLedger` per market, fed ONLY by confirmed `Fill`s from the broadcast
//! (`fill_sink` has already persisted them to SQLite — the strategy does not modify
//! inventory by emitting an intent). `SELL` requires `shares > 0` in the
//! virtual ledger; if ownership cannot be proven (pending or failed recovery, or an
//! uncertain state), EVERY action other than HOLD is masked and the condition is
//! recorded. Settlement is observed through the reconciliation task, which polls
//! `positions.outcome_won`/`redeemed` (authority: the bot's SQLite) — the resolution
//! is never invented.
//!
//! ## Fail-closed
//!
//! - `Mode::Live` → `panic!` in the constructor (same structural barrier as
//!   `ModelBtc5m`; `StrategyLimits::can_run`/`paper_only` already filter, this is the
//!   final safety net). Live learning is FORBIDDEN, as the whole strategy is, so no
//!   separate gate is needed.
//! - Without a valid `paper_approved` artifact the strategy is built in a disabled
//!   state: it emits no intents, does not learn, and publishes the reason in
//!   `rl_strategy_status()` for the API. There is never a fallback to synthetic,
//!   zero or random weights, or to "renaming the predictor".
//! - Each `ModelPrediction.model_hash` must match the artifact's `predictor_sha256`:
//!   a policy trained for another predictor is rejected by observation, not by trust.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
#[cfg(test)]
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::{broadcast, mpsc, watch};

use crate::config::{Config, StrategyLimits};
#[cfg(test)]
use crate::fair_value::fee_charged_5dp;
use crate::rl::artifact::{self, ApprovedArtifact};
use crate::rl::contract::*;
use crate::rl::episode::{BookLevelsUm, EpisodeLedger, TokenPosition};
use crate::rl::learner::{
    self, EpisodeEndMsg, LearnerHandle, LearnerMsg, ReinforceConfig, TransitionRec,
};
use crate::rl::obs;
use crate::rl::policy::PolicySnapshot;
use crate::rl::rng::SplitMix64;
use crate::state::{to_micros, StateStore};
use crate::strategy::model_btc5m::is_target_market;
use crate::tools::model::{ModelPrediction, MODEL_TOOL_SCHEMA_VERSION};
use crate::types::{
    BookSnapshot, Fill, FillSide, Mode, OrderIntent, OrderSide, StrategyId, VolSnapshot,
};

use super::Strategy;

/// Label persisted in `positions.strategy` for this strategy — the key
/// of the ledger attribution (recovery and settle-watch filter by it).
const STRATEGY_LABEL: &str = "rl_predictor";

// ── Retention bounds ──────────────────────────────────────────────────────────

/// `closed_eps` is only for observability in tests and replay: it keeps at most the
/// last N `closed_ids` episodes (~1 day of 5-minute brackets: ~288 closes/day; 256
/// covers ~21 h).
const CLOSED_EPS_CAP: usize = 256;

/// Slack after `clear_at` before a cid is pruned from `terminal`. Brackets last
/// minutes and settle almost immediately, but a late fill from the durable ledger must
/// still be recognized as terminal while a delayed book or fill can plausibly arrive.
/// One hour is very conservative for 5-minute markets.
const TERMINAL_RETENTION_SLACK_SECS: i64 = 3_600;

/// Size threshold that triggers the lazy sweep of `terminal`. The sweep runs in
/// `on_timer` only when the map exceeds this bound, so it does not iterate on every
/// tick. Well above the number of simultaneously live markets (dozens), and below the
/// daily growth (~288).
const TERMINAL_PRUNE_THRESHOLD: usize = 512;

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct RlPredictorConfig {
    #[serde(flatten)]
    pub limits: StrategyLimits,
    /// Directory of the APPROVED policy artifact (`policy.json` + `manifest.json` with
    /// `status="paper_approved"` and `approval.scope="paper"`). Empty or invalid → the
    /// strategy is disabled. It NEVER points to the predictor's artifact (`.../btc5m`).
    #[serde(default)]
    pub artifact_dir: String,
    /// Checkpoint directory for online learning (learner). Created if it does not exist.
    #[serde(default = "defaults::checkpoint_dir")]
    pub checkpoint_dir: String,
    /// Expected sha256 of the predictor (`model.json`). Empty = verify at runtime against
    /// each prediction's `ModelPrediction.model_hash` (predictions that differ from the
    /// artifact's `predictor_sha256` are rejected).
    #[serde(default)]
    pub expected_predictor_sha256: String,
    /// Fixed virtual budget per episode (market), so the paper sim is not infinite and
    /// learned ruin is measurable. Default 10.
    #[serde(default = "defaults::episode_budget")]
    pub initial_episode_budget: Decimal,
    /// `true` = adaptive paper: actions are SAMPLED over the legal mask and cohorts
    /// update the weights. `false` = frozen evaluation: deterministic argmax and NO
    /// trainable experience.
    #[serde(default)]
    pub learning_enabled: bool,
    /// Maximum age of the prediction used as a feature (ms).
    #[serde(default = "defaults::max_prediction_age_ms")]
    pub max_prediction_age_ms: i64,
    /// Entry window (remaining seconds). SELL never uses this window.
    #[serde(default = "defaults::min_remaining_secs")]
    pub min_remaining_secs: i64,
    #[serde(default = "defaults::max_remaining_secs")]
    pub max_remaining_secs: i64,
    /// Maximum tolerable spread on the BOUGHT leg (0 = no filter). It does not block sells.
    #[serde(default = "defaults::max_spread")]
    pub max_spread: Decimal,
    /// Slippage margin of the protective limit (USDC/share).
    #[serde(default = "defaults::slippage_margin")]
    pub slippage_margin: Decimal,
    /// Fee rate fallback, used when the book does not carry `parabolic_fee_rate`.
    #[serde(default = "defaults::fee_rate_fallback")]
    pub fee_rate_fallback: Decimal,
    /// Minimum executable size (shares). Below it the action is illegal.
    #[serde(default = "defaults::min_order_shares")]
    pub min_order_shares: Decimal,
    /// If an intent goes longer than this without a confirmed fill or reject, the market
    /// enters an uncertain state (blocks new orders; it does NOT invent a rejection).
    #[serde(default = "defaults::pending_timeout_ms")]
    pub pending_timeout_ms: i64,
    /// Polling cadence of settlement in the reconciliation task.
    #[serde(default = "defaults::settle_poll_secs")]
    pub settle_poll_secs: u64,
    /// Maximum time to wait for the official outcome after `clear_at` before closing the
    /// episode as `settle_timeout` (excluded from cohorts).
    #[serde(default = "defaults::settle_timeout_secs")]
    pub settle_timeout_secs: i64,
    /// Inventory-state penalty in the reward (default 0 in v1).
    #[serde(default)]
    pub lambda_inventory: f64,
    /// Incremental drawdown penalty in the reward (default 0 in v1).
    #[serde(default)]
    pub lambda_drawdown: f64,
    /// Base seed of the sampling PRNG (decision auditability).
    #[serde(default)]
    pub rng_seed: u64,

    // ── Learner (episodic REINFORCE) ───────────────────────────────────
    /// Capacity of the strategy→learner channel (bounded; drops are counted).
    #[serde(default = "defaults::learner_capacity")]
    pub learner_capacity: usize,
    #[serde(default = "defaults::learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "defaults::max_grad_norm")]
    pub max_grad_norm: f64,
    /// Hard gate on the update magnitude (`‖Δw‖₂`).
    #[serde(default = "defaults::max_update_l2")]
    pub max_update_l2: f64,
    /// KL gates over the observations recorded for the cohort.
    #[serde(default = "defaults::max_mean_kl")]
    pub max_mean_kl: f64,
    #[serde(default = "defaults::max_step_kl")]
    pub max_step_kl: f64,
    /// Minimum number of complete episodes that close a cohort.
    #[serde(default = "defaults::min_episodes_per_update")]
    pub min_episodes_per_update: usize,
    /// Cap on accumulated episodes in a cohort (protection under freeze).
    #[serde(default = "defaults::cohort_cap")]
    pub cohort_cap: usize,
    /// Maximum wait at the cohort boundary before open episodes are invalidated.
    #[serde(default = "defaults::boundary_max_wait_ms")]
    pub boundary_max_wait_ms: i64,
}

impl Default for RlPredictorConfig {
    fn default() -> Self {
        Self {
            limits: StrategyLimits {
                enabled: false, // disabled until a real approved artifact exists
                paper_only: true,
                max_orders_per_hour: 120,
                max_order_usdc: Decimal::new(5, 0),
                daily_loss_usdc_stop: Decimal::new(10, 0),
                universe: vec!["crypto-updown-short".into(), "g-rw1".into()],
            },
            artifact_dir: String::new(),
            checkpoint_dir: defaults::checkpoint_dir(),
            expected_predictor_sha256: String::new(),
            initial_episode_budget: defaults::episode_budget(),
            learning_enabled: false,
            max_prediction_age_ms: defaults::max_prediction_age_ms(),
            min_remaining_secs: defaults::min_remaining_secs(),
            max_remaining_secs: defaults::max_remaining_secs(),
            max_spread: defaults::max_spread(),
            slippage_margin: defaults::slippage_margin(),
            fee_rate_fallback: defaults::fee_rate_fallback(),
            min_order_shares: defaults::min_order_shares(),
            pending_timeout_ms: defaults::pending_timeout_ms(),
            settle_poll_secs: defaults::settle_poll_secs(),
            settle_timeout_secs: defaults::settle_timeout_secs(),
            lambda_inventory: 0.0,
            lambda_drawdown: 0.0,
            rng_seed: 0,
            learner_capacity: defaults::learner_capacity(),
            learning_rate: defaults::learning_rate(),
            max_grad_norm: defaults::max_grad_norm(),
            max_update_l2: defaults::max_update_l2(),
            max_mean_kl: defaults::max_mean_kl(),
            max_step_kl: defaults::max_step_kl(),
            min_episodes_per_update: defaults::min_episodes_per_update(),
            cohort_cap: defaults::cohort_cap(),
            boundary_max_wait_ms: defaults::boundary_max_wait_ms(),
        }
    }
}

impl crate::strategy::StrategyConfig for RlPredictorConfig {
    fn limits(&self) -> &StrategyLimits {
        &self.limits
    }
}

impl RlPredictorConfig {
    /// Operational subset that shares the math of the contract
    /// (`rl::obs`) — same source for runtime and the offline driver.
    pub fn tunables(&self) -> obs::PolicyTunables {
        obs::PolicyTunables {
            slippage_margin: self.slippage_margin,
            min_order_shares: self.min_order_shares,
            max_order_usdc: self.limits.max_order_usdc,
            min_remaining_secs: self.min_remaining_secs,
            max_remaining_secs: self.max_remaining_secs,
            max_spread: self.max_spread,
        }
    }

    fn reinforce_config(&self) -> ReinforceConfig {
        ReinforceConfig {
            learning_rate: self.learning_rate,
            max_grad_norm: self.max_grad_norm,
            max_update_l2: self.max_update_l2,
            max_mean_kl: self.max_mean_kl,
            max_step_kl: self.max_step_kl,
            min_episodes_per_update: self.min_episodes_per_update,
            cohort_cap: self.cohort_cap,
            boundary_max_wait_ms: self.boundary_max_wait_ms,
            ..ReinforceConfig::default()
        }
    }
}

pub mod defaults {
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    pub fn checkpoint_dir() -> String {
        "data/rl_predictor".into()
    }
    pub fn episode_budget() -> Decimal {
        dec!(10)
    }
    pub fn max_prediction_age_ms() -> i64 {
        1500
    }
    pub fn min_remaining_secs() -> i64 {
        30
    }
    pub fn max_remaining_secs() -> i64 {
        240
    }
    pub fn max_spread() -> Decimal {
        dec!(0.05)
    }
    pub fn slippage_margin() -> Decimal {
        dec!(0.01)
    }
    pub fn fee_rate_fallback() -> Decimal {
        dec!(0.07)
    }
    pub fn min_order_shares() -> Decimal {
        dec!(1)
    }
    pub fn pending_timeout_ms() -> i64 {
        3_000
    }
    pub fn settle_poll_secs() -> u64 {
        15
    }
    pub fn settle_timeout_secs() -> i64 {
        600
    }
    pub fn learner_capacity() -> usize {
        2_048
    }
    pub fn learning_rate() -> f64 {
        0.05
    }
    pub fn max_grad_norm() -> f64 {
        1.0
    }
    pub fn max_update_l2() -> f64 {
        0.5
    }
    pub fn max_mean_kl() -> f64 {
        0.05
    }
    pub fn max_step_kl() -> f64 {
        0.2
    }
    pub fn min_episodes_per_update() -> usize {
        4
    }
    pub fn cohort_cap() -> usize {
        64
    }
    pub fn boundary_max_wait_ms() -> i64 {
        360_000
    }
}

// ── Recovery / reconciliation against the real ledger ─────────────────────────

/// Recovery state after a restart ("restart recovers
/// positions/orders from the ledger before enabling").
#[derive(Debug, Clone)]
pub enum RecoveryState {
    /// The `positions` table could not be queried yet — everything blocked except HOLD.
    Pending,
    /// Own positions by `market_cid` → `token_id` → (size, cost).
    Ready {
        positions: HashMap<String, HashMap<String, TokenPosition>>,
    },
    /// The ledger query failed — ownership cannot be proven: block
    /// ALL orders (including SELL) and keep the alarm visible.
    Failed(String),
}

/// Settlement polling record sent from the strategy to the
/// reconciliation when a market passes `clear_at` with an open position.
#[derive(Debug)]
struct SettleWatch {
    cid: String,
    yes_token: String,
    no_token: String,
    /// Wait deadline (ms epoch); once expired → `EpisodeEnd` truncated.
    deadline_ms: i64,
}

/// Official resolution event observed in `positions` (real SQLite).
/// `won` is `None` when the position is `redeemed` but `outcome_won`
/// is still NULL — the episode is invalidated (`settle_outcome_unknown`) instead
/// of assuming a fabricated defeat.
#[derive(Debug)]
pub struct SettleEvent {
    pub cid: String,
    pub token_id: String,
    pub won: Option<bool>,
}

/// Return channels of the reconciliation task.
struct Reconcile {
    recovery_rx: watch::Receiver<RecoveryState>,
    watch_tx: Option<mpsc::UnboundedSender<SettleWatch>>,
    settle_rx: mpsc::UnboundedReceiver<SettleEvent>,
}

/// Reconciliation task (the only one that touches SQLite for this strategy):
/// 1) recovery at startup, 2) periodic polling of `outcome_won`/`redeemed`
/// for the registered tokens. It lives inside the tokio runtime; the hot path
/// of the strategy never waits on it.
async fn reconcile_run(
    db_path: String,
    mode: Mode,
    recovery_tx: watch::Sender<RecoveryState>,
    mut watch_rx: mpsc::UnboundedReceiver<SettleWatch>,
    settle_tx: mpsc::UnboundedSender<SettleEvent>,
    poll: std::time::Duration,
) {
    let state = match StateStore::open(&db_path, mode).await {
        Ok(s) => s,
        Err(e) => {
            let _ = recovery_tx.send(RecoveryState::Failed(format!(
                "StateStore::open({db_path}): {e}"
            )));
            tracing::error!(
                "rl_predictor: no ledger access — recovery FAILED, all orders blocked"
            );
            return;
        }
    };
    let rec = match state.list_open_positions().await {
        Ok(positions) => {
            let mut by_cid: HashMap<String, HashMap<String, TokenPosition>> = HashMap::new();
            for p in positions {
                if p.strategy != STRATEGY_LABEL || p.size <= Decimal::ZERO {
                    continue;
                }
                by_cid.entry(p.market_cid.clone()).or_default().insert(
                    p.token_id.clone(),
                    TokenPosition {
                        size_um: to_micros(p.size),
                        cost_um: to_micros(p.cost_basis),
                    },
                );
            }
            RecoveryState::Ready { positions: by_cid }
        }
        Err(e) => RecoveryState::Failed(format!("list_open_positions: {e}")),
    };
    let _ = recovery_tx.send(rec);

    // Settlement polling: token → (cid, deadline). Unbounded, because
    // settle events are ≤2 per market and are NEVER lost (unlike the
    // learner's experience channel, which may drop messages).
    let mut watched: HashMap<String, (String, i64)> = HashMap::new();
    let mut tick = tokio::time::interval(poll.max(std::time::Duration::from_secs(1)));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            w = watch_rx.recv() => match w {
                Some(w) => {
                    watched.insert(w.yes_token.clone(), (w.cid.clone(), w.deadline_ms));
                    watched.insert(w.no_token.clone(), (w.cid, w.deadline_ms));
                }
                None => {
                    if watched.is_empty() { break; }
                    // The strategy closed its channel but markets remain
                    // to resolve — keep going until the deadline runs out or shutdown.
                }
            },
            _ = tick.tick() => {}
        }
        if watched.is_empty() {
            continue;
        }
        let mut done = Vec::new();
        for (token, (cid, deadline)) in &watched {
            match state.get_position(token, STRATEGY_LABEL).await {
                Ok(Some(pos)) if pos.outcome_won.is_some() || pos.redeemed => {
                    // `redeemed` without `outcome_won` propagates `won=None`
                    // — the strategy invalidates the episode, it never assumes
                    // a defeat with a fabricated result.
                    let _ = settle_tx.send(SettleEvent {
                        cid: cid.clone(),
                        token_id: token.clone(),
                        won: pos.outcome_won,
                    });
                    done.push(token.clone());
                }
                Ok(_) => {
                    if crate::time::now().timestamp_millis() > *deadline {
                        // Timeout without outcome — the strategy closes it as
                        // `settle_timeout`; polling stops.
                        done.push(token.clone());
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        token = %token, cid = %cid,
                        "rl_predictor: settlement polling failed ({e}) — retrying on next tick"
                    );
                }
            }
        }
        for t in done {
            watched.remove(&t);
        }
    }
}

// ── Per-market state ───────────────────────────────────────────────────────

/// Deduplication key of a fill: `exec_id` (identity of the
/// execution — unique even across partial fills of the same order) or,
/// for legacy sources without exec_id, the full event content
/// PLUS `local_order_id` (pre-submit rejections carry the attempt_id
/// — two distinct rejections of the same order are not merged).
fn fill_dedup_key(fill: &Fill) -> String {
    match &fill.exec_id {
        Some(e) => format!("exec:{e}"),
        None => format!(
            "legacy:{}:{}:{:?}:{}:{}:{}",
            fill.order_id.as_deref().unwrap_or("-"),
            fill.local_order_id.as_deref().unwrap_or("-"),
            fill.side,
            fill.price,
            fill.size,
            fill.ts.timestamp_millis()
        ),
    }
}

/// In-flight order: at most ONE per market. Its resolution arrives
/// via a confirmed `Fill` — never considered resolved by `resolved_path` when emitted.
#[derive(Debug)]
struct PendingOrder {
    token_id: String,
    /// Accounting reserve set on emit (micros; BUY only).
    reserved_um: i64,
    emitted_ts_ms: i64,
    action: usize,
}

#[derive(Debug)]
struct MarketState {
    episode: EpisodeLedger,
    yes_token: String,
    no_token: String,
    clear_at: i64,
    /// Market fee rate (from the first book that created it — 5m brackets
    /// do not change regime).
    fee_rate: Decimal,
    /// In-flight intent (at most one per market).
    pending: Option<PendingOrder>,
    /// Uncertain state: pending expired without resolution — blocks ALL
    /// orders until a fill/reject reconciles it or the market closes.
    uncertain: bool,
    /// Last decision whose reward is not yet assigned (computed in the
    /// next accounting point — observable consequences).
    pending_transition: Option<TransitionRec>,
    /// Last prediction received for this cid.
    last_prediction: Option<ModelPrediction>,
    /// Settlement polling already registered in the reconciliation task.
    settle_watched: bool,
    /// Sequence of decisions in the episode (for the sampling seed).
    decision_seq: u64,
    /// `exec_id` (or legacy key) of fills ALREADY applied to this episode —
    /// deduplicates broadcast re-emissions. Intents are
    /// `resting=false` (FOK/FAK) → a single execution per order; an exec_id
    /// repeated can only be a re-emission and is ignored, not counted.
    seen_execs: HashSet<String>,
    /// Decisions with a REAL choice (`sampled=true` on the transition:
    /// mask with >1 legal action and no forced HOLD). An episode with
    /// `sampled_steps == 0` — only warmup/forced — closes excluded from
    /// cohorts: there is no gradient to learn (parity with the driver
    /// offline `trainable_step = mask_bits != 1`).
    sampled_steps: u32,
}

impl MarketState {
    /// Read-only view for the contractual math of `rl::obs`
    /// (same structure the offline driver uses — parity by design).
    fn view(&self) -> obs::MarketView<'_> {
        obs::MarketView {
            episode: &self.episode,
            yes_token: &self.yes_token,
            no_token: &self.no_token,
            fee_rate: self.fee_rate,
            pending: self.pending.is_some(),
            uncertain: self.uncertain,
        }
    }
}

// ── Published state (API/diagnostics) ───────────────────────────────────────

/// Strategy state for `/ui/rl_learner` and logs: built with
/// a valid artifact, or disabled with an explicit reason.
#[derive(Debug, Clone, Default)]
pub struct RlStrategyStatus {
    /// `true` if a policy is loaded and verified.
    pub active: bool,
    /// Reason if `active == false` (artifact missing, not approved, hash
    /// broken, prohibited mode, different predictor…).
    pub disabled_reason: Option<String>,
    pub policy_sha256: String,
    pub artifact_dir: String,
    pub learning_enabled: bool,
}

/// State of the LAST constructed strategy (the hot-toggle replaces it).
static RL_STRATEGY_STATUS: LazyLock<Mutex<RlStrategyStatus>> =
    LazyLock::new(|| Mutex::new(RlStrategyStatus::default()));

/// Read for the API — never exposes weights, only state.
pub fn rl_strategy_status() -> RlStrategyStatus {
    RL_STRATEGY_STATUS.lock().unwrap().clone()
}

fn publish_status(s: RlStrategyStatus) {
    *RL_STRATEGY_STATUS.lock().unwrap() = s;
}

// ── Pure decision helpers (no &self — testable and borrow-free) ─────────

/// Is the prediction usable as a feature? Same gate chain as
/// `model_btc5m`: schema, abstention, p_model Some and finite in [0,1],
/// `model_hash` present, tokens/slug = book, freshness. In addition, the
/// `model_hash` must match the `predictor_sha256` of the policy
/// artifact: the policy was trained against THAT frozen predictor.
fn pred_usable(
    cfg: &RlPredictorConfig,
    book: &BookSnapshot,
    pred: Option<&ModelPrediction>,
    predictor_sha256: &str,
) -> Option<f64> {
    let p = pred?;
    if p.schema_version != MODEL_TOOL_SCHEMA_VERSION
        || p.abstention.is_some()
        || p.token_id_yes != book.yes.token_id
        || p.token_id_no != book.no.token_id
        || p.slug != book.slug
    {
        return None;
    }
    let pm = p.p_model?;
    if !pm.is_finite() || !(0.0..=1.0).contains(&pm) {
        return None;
    }
    match p.model_hash.as_deref() {
        Some(h) if !h.is_empty() => {
            if !predictor_sha256.is_empty() && h != predictor_sha256 {
                metrics::counter!("hrf_rl_pred_hash_mismatch_total").increment(1);
                return None;
            }
        }
        _ => return None, // without a hash there is no predictor↔policy traceability
    }
    let now_ms = crate::time::now().timestamp_millis();
    if p.ts_ms > now_ms || now_ms - p.ts_ms > cfg.max_prediction_age_ms {
        return None;
    }
    Some(pm)
}

/// Thin wrappers over `rl::obs` — the contractual math (features,
/// mask, sizes/limits) has ONE single implementation shared with
/// the offline driver (`rl_policy_env`); here only `MarketState` is adapted
/// to a `MarketView`. Train/serve parity holds by construction.
fn mask_legal(
    cfg: &RlPredictorConfig,
    m: &MarketState,
    book: &BookSnapshot,
    p_usable: bool,
    boundary: bool,
) -> ActionMask {
    obs::mask_legal(&cfg.tunables(), &m.view(), book, p_usable, boundary)
}

/// Observation of 22 features (contractual order `FEATURE_NAMES`).
/// `None` if any feature is not finite — the observation is invalidated
/// as a whole; nothing is imputed.
fn build_obs(
    cfg: &RlPredictorConfig,
    m: &MarketState,
    book: &BookSnapshot,
    p_yes: Option<f64>,
    book_age_secs: f64,
) -> Option<[f64; NUM_FEATURES]> {
    obs::build_obs(&cfg.tunables(), &m.view(), book, p_yes, book_age_secs)
}

// ── Strategy state ───────────────────────────────────────────────────────

pub struct RlPredictor {
    cfg: RlPredictorConfig,
    prediction_rx: broadcast::Receiver<ModelPrediction>,
    mode: Mode,
    /// `Err(reason)` = no valid policy → full abstention.
    policy: Result<PolicySnapshot, String>,
    /// sha256 of the predictor the loaded policy was trained against.
    predictor_sha256: String,
    /// Channel to the learner (None if `learning_enabled=false` or no policy).
    learner: Option<LearnerHandle>,
    reconcile: Option<Reconcile>,
    markets: HashMap<String, MarketState>,
    /// Predictions received before the first book of their cid.
    pred_stash: HashMap<String, ModelPrediction>,
    /// Sampling PRNG (adaptive paper mode).
    rng: SplitMix64,
    /// Instrumentation for the replay/test harness: PnL and reason per
    /// closed episode (an empty `Vec` in production — zero cost).
    /// Bounded to the last `CLOSED_EPS_CAP` episodes.
    closed_eps: Vec<ClosedEpisode>,
    /// Cids whose episode has ALREADY closed (`close_episode`) → `clear_at` of the
    /// market (used by the lazy `prune_terminal` pruning). A
    /// terminal market does NOT reopen experience: late books/fills do not
    /// recreate `MarketState` nor feed the learner's cohort with
    /// artificial flat episodes (the late economic consequence
    /// remains in the durable ledger — `fill_sink` — even if the episode
    /// does not exist).
    terminal: HashMap<String, i64>,
    /// Last REAL observation with a free choice (≥2 legal actions)
    /// captured inside `on_book`, together with the probs computed with the
    /// SAME snapshot of that decision. Used by the `online_proof` harness
    /// for the retrospective post-update probe (version tied to the
    /// inference, not read before/after the call).
    #[cfg(test)]
    probe_obs: Option<ProbeCapture>,
    /// Record of ALL messages emitted to the learner by this
    /// instance (`cfg(test)` harness): the real trajectory sent,
    /// reusable to feed another worker instance in tests
    /// of concurrency/recovery without reconstructing data.
    #[cfg(test)]
    sent_log: Mutex<Vec<LearnerMsg>>,
}

/// Internal capture of a decision with a real choice (`cfg(test)` harness).
#[cfg(test)]
#[derive(Debug, Clone)]
struct ProbeCapture {
    obs: [f64; NUM_FEATURES],
    mask: ActionMask,
    /// Version and probs of the SAME `Arc<PolicySnapshot>` that decided.
    version: u64,
    policy_sha256: String,
    probs: [f64; NUM_ACTIONS],
}

/// Episode-close record for accounting verification in tests.
/// The fields are only read under `cfg(test)` (`online_proof` harness).
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct ClosedEpisode {
    pub cid: String,
    pub reason: String,
    /// Ledger invalidation reason, if any (`pending_timeout`,
    /// `mixed_version`…) — different from the close reason.
    #[allow(dead_code)]
    pub invalid_reason: Option<String>,
    pub trainable: bool,
    /// `EpisodeLedger.realized_pnl_um` at close — accounting identity
    /// against the `wallet_delta` of the replay engine.
    pub pnl_um: i64,
    pub cash_um: i64,
    pub budget_um: i64,
}

impl RlPredictor {
    /// Builds the strategy. **PAPER-ONLY structural**: `mode=Live` →
    /// immediate panic (same as `ModelBtc5m::new` — live learning
    /// is prohibited by design, not by a toggle).
    ///
    /// The policy artifact is loaded NOW (fail-closed): any
    /// loading problem leaves the strategy disabled with a reason —
    /// "a strategy installed without a valid policy stays
    /// disabled and explains why".
    ///
    /// `prediction_rx=None` (predictor tool not started): the strategy
    /// it is built anyway — without `p_yes` the BUYs stay masked
    /// by `p_missing` but SELL/HOLD keep managing positions
    /// recovered from the ledger (the absence of a predictor never freezes
    /// own inventory).
    pub fn new(
        cfg: RlPredictorConfig,
        prediction_rx: Option<broadcast::Receiver<ModelPrediction>>,
        mode: Mode,
        config: &Config,
    ) -> Self {
        if mode == Mode::Live {
            panic!(
                "RlPredictor: the online runtime NEVER operates in mode Live — \
                   tried to build with mode=Live (paper_only={}, \
                   enabled={}). Learning and the \
                   live operation of this strategy are forbidden by contract.",
                cfg.limits.paper_only, cfg.limits.enabled,
            );
        }

        let expected = (!cfg.expected_predictor_sha256.is_empty())
            .then_some(cfg.expected_predictor_sha256.clone());
        let policy = artifact::load_approved(
            std::path::Path::new(&cfg.artifact_dir),
            expected.as_deref(),
        );

        // Missing prediction channel → dead receiver (`Closed` on
        // drain): `p_missing=1` permanently. This is not a silent failure —
        // it is logged and observable via the mask/features.
        let prediction_rx = prediction_rx.unwrap_or_else(|| {
            tracing::warn!(
                "rl_predictor: no ModelPrediction channel — BUYs masked \
                  permanently (p_missing=1); SELL/HOLD of inventory keep managing positions"
            );
            broadcast::channel(2).1
        });

        let mut this = Self {
            policy: Err(String::new()), // replaced below
            predictor_sha256: String::new(),
            learner: None,
            reconcile: None,
            markets: HashMap::new(),
            pred_stash: HashMap::new(),
            rng: SplitMix64::new(cfg.rng_seed),
            prediction_rx,
            cfg: cfg.clone(),
            mode,
            closed_eps: Vec::new(),
            terminal: HashMap::new(),
            #[cfg(test)]
            probe_obs: None,
            #[cfg(test)]
            sent_log: Mutex::new(Vec::new()),
        };

        match policy {
            Ok(art) => this.install_artifact(art, &config.db_path),
            Err(e) => {
                let reason = e.to_string();
                this.policy = Err(reason.clone());
                publish_status(RlStrategyStatus {
                    active: false,
                    disabled_reason: Some(reason.clone()),
                    policy_sha256: String::new(),
                    artifact_dir: cfg.artifact_dir.clone(),
                    learning_enabled: cfg.learning_enabled,
                });
                tracing::warn!(
                    "rl_predictor: DISABLED — {reason} \
                       (without a paper_approved artifact there is no policy; there is no fallback)"
                );
            }
        }
        this
    }

    /// Wires artifact + learner + reconciliation after a successful load.
    /// `db_path` is the real durable SQLite (`config.db_path`) — the task
    /// reconciliation opens its own connection to the SAME file.
    fn install_artifact(&mut self, art: ApprovedArtifact, db_path: &str) {
        let artifact_dir = art.artifact_dir.clone();
        let policy_sha = art.policy_sha256.clone();
        self.predictor_sha256 = art.predictor_sha256.clone();

        // Online learner (only if learning_enabled) — versioned snapshots.
        self.learner = if self.cfg.learning_enabled {
            learner::spawn_learner(
                &art,
                PathBuf::from(&self.cfg.checkpoint_dir),
                self.cfg.reinforce_config(),
                self.cfg.learner_capacity,
            )
        } else {
            None
        };
        if self.cfg.learning_enabled && self.learner.is_none() {
            tracing::warn!(
                "rl_predictor: no tokio runtime — learner not started; \
                   running with the base v0 policy without learning"
            );
        }
        self.policy = Ok(self
            .learner
            .as_ref()
            .map(|l| l.snapshot().as_ref().clone())
            .unwrap_or_else(|| PolicySnapshot {
                version: 0,
                update_index: 0,
                policy: Arc::new(art.policy.clone()),
                policy_sha256: policy_sha.clone(),
                parent_sha256: String::new(),
                base_sha256: policy_sha.clone(),
                published_ts_ms: crate::time::now().timestamp_millis(),
            }));

        // Reconciliation with the real ledger (recovery + settle-watch).
        if tokio::runtime::Handle::try_current().is_ok() {
            let (recovery_tx, recovery_rx) = watch::channel(RecoveryState::Pending);
            let (watch_tx, watch_rx) = mpsc::unbounded_channel();
            let (settle_tx, settle_rx) = mpsc::unbounded_channel();
            tokio::spawn(reconcile_run(
                db_path.to_string(),
                self.mode,
                recovery_tx,
                watch_rx,
                settle_tx,
                std::time::Duration::from_secs(self.cfg.settle_poll_secs),
            ));
            self.reconcile = Some(Reconcile {
                recovery_rx,
                watch_tx: Some(watch_tx),
                settle_rx,
            });
        }

        publish_status(RlStrategyStatus {
            active: true,
            disabled_reason: None,
            policy_sha256: policy_sha.clone(),
            artifact_dir: artifact_dir.to_string_lossy().into_owned(),
            learning_enabled: self.cfg.learning_enabled,
        });
        tracing::info!(
            policy_sha256 = %policy_sha,
            artifact = %artifact_dir.display(),
            learning = self.cfg.learning_enabled,
            "rl_predictor: paper_approved policy loaded and active"
        );
    }

    /// Constructor for tests: EXPLICIT synthetic policy (fixture, never
    /// enters production — the real path is `new` + `load_approved`),
    /// injected recovery and no learner or DB.
    #[cfg(test)]
    fn new_test(
        mut cfg: RlPredictorConfig,
        prediction_rx: Option<broadcast::Receiver<ModelPrediction>>,
        policy: crate::rl::policy::LinearSoftmaxPolicy,
        recovery: RecoveryState,
        learner: Option<LearnerHandle>,
    ) -> Self {
        cfg.limits.enabled = true;
        let (recovery_tx, recovery_rx) = watch::channel(recovery);
        drop(recovery_tx); // channel closed, but the last value stays readable
        let (watch_tx, _watch_rx) = mpsc::unbounded_channel::<SettleWatch>();
        let (settle_tx, settle_rx) = mpsc::unbounded_channel::<SettleEvent>();
        drop(settle_tx);
        Self {
            policy: Ok(PolicySnapshot {
                version: 0,
                update_index: 0,
                policy: Arc::new(policy),
                policy_sha256: "test".into(),
                parent_sha256: String::new(),
                base_sha256: "test".into(),
                published_ts_ms: 0,
            }),
            predictor_sha256: String::new(), // tests inject the hash via pred
            learner,
            reconcile: Some(Reconcile {
                recovery_rx,
                watch_tx: Some(watch_tx),
                settle_rx,
            }),
            markets: HashMap::new(),
            pred_stash: HashMap::new(),
            rng: SplitMix64::new(cfg.rng_seed),
            prediction_rx: prediction_rx.unwrap_or_else(|| broadcast::channel(2).1),
            cfg,
            mode: Mode::Paper,
            closed_eps: Vec::new(),
            terminal: HashMap::new(),
            probe_obs: None,
            #[cfg(test)]
            sent_log: Mutex::new(Vec::new()),
        }
    }

    /// Constructor for the online replay harness (tests ONLY): accepts a
    /// `ApprovedArtifact` ALREADY built by `artifact::load_candidate_for_test`
    /// — injecting the `candidate` is EXPLICIT here; `new` (production)
    /// still requires `load_approved` (paper_approved + manifest). The rest
    /// of the wiring is the REAL one: `spawn_learner` + `reconcile_run` over the
    /// `db_path` shared with the replay engine.
    #[cfg(test)]
    fn new_test_wired(
        mut cfg: RlPredictorConfig,
        prediction_rx: Option<broadcast::Receiver<ModelPrediction>>,
        art: ApprovedArtifact,
        db_path: &str,
    ) -> Self {
        cfg.limits.enabled = true;
        let mut this = Self {
            policy: Err(String::new()),
            predictor_sha256: String::new(),
            learner: None,
            reconcile: None,
            markets: HashMap::new(),
            pred_stash: HashMap::new(),
            rng: SplitMix64::new(cfg.rng_seed),
            prediction_rx: prediction_rx.unwrap_or_else(|| broadcast::channel(2).1),
            cfg,
            mode: Mode::Paper,
            closed_eps: Vec::new(),
            terminal: HashMap::new(),
            probe_obs: None,
            #[cfg(test)]
            sent_log: Mutex::new(Vec::new()),
        };
        this.install_artifact(art, db_path);
        this
    }

    /// Current policy snapshot (the learner publishes new versions via
    /// `watch`; without a learner, the base v0 of the approved artifact).
    fn current_policy(&self) -> Option<PolicySnapshot> {
        match &self.learner {
            Some(l) => Some(l.snapshot().as_ref().clone()),
            None => self.policy.clone().ok(),
        }
    }

    fn boundary_active(&self) -> bool {
        self.learner
            .as_ref()
            .map(|l| l.control.boundary_active())
            .unwrap_or(false)
    }

    /// Retrospective probe of the online harness: re-runs the INFERENCE
    /// real (`LinearSoftmaxPolicy::action_probs`, the same function that
    /// serves the decision) over the REAL observation/mask captured
    /// at the first free-choice decision of the episode — reading
    /// the snapshot via `current_policy()` INSIDE this call.
    /// Returns `(version, sha256, probs)` BOUND to the `Arc` that produced
    /// the probs — it is not a version read before/after, it is the one
    /// snapshot that computed them. `None` if there has been no free decision yet.
    /// It is retrospective plumbing evidence, not a new operable decision —
    /// probes NEVER enter the learner.
    #[cfg(test)]
    fn probe_infer(&self) -> Option<(u64, String, [f64; NUM_ACTIONS], usize)> {
        let cap = self.probe_obs.as_ref()?;
        let snap = self.current_policy()?;
        let probs = snap.policy.action_probs(&cap.obs, &cap.mask)?;
        let n_legal = cap.mask.iter().filter(|b| **b).count();
        Some((snap.version, snap.policy_sha256.clone(), probs, n_legal))
    }

    /// Probabilities produced by the snapshot of the captured decision
    /// (baseline of the probe — measured at decision time).
    #[cfg(test)]
    fn probe_base(&self) -> Option<ProbeCapture> {
        self.probe_obs.clone()
    }

    /// Drains the prediction queue, keeping the last one of each cid
    /// (latest-per-key — the broadcast may carry several brackets).
    fn drain_predictions(&mut self) {
        const MAX_DRAIN: usize = 32;
        for _ in 0..MAX_DRAIN {
            match self.prediction_rx.try_recv() {
                Ok(p) => {
                    if let Some(m) = self.markets.get_mut(&p.market_cid) {
                        m.last_prediction = Some(p);
                    } else if self.pred_stash.len() < 256 {
                        // Market not seen yet: keep the last one per cid.
                        self.pred_stash.insert(p.market_cid.clone(), p);
                    }
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(k)) => {
                    metrics::counter!("hrf_rl_pred_lag_total").increment(k);
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
            }
        }
    }

    /// Applies a settlement event to the episode of its market. Returns
    /// `true` if a NEW settlement was accounted for (closes the episode).
    fn apply_settle_event(&mut self, ev: &SettleEvent) -> bool {
        let Some(m) = self.markets.get_mut(&ev.cid) else {
            return false;
        };
        match ev.won {
            Some(won) => {
                if m.episode.apply_settlement(&ev.token_id, won) {
                    metrics::counter!("hrf_rl_settlements_total").increment(1);
                    tracing::info!(
                        cid = %ev.cid, token = %ev.token_id, won,
                        "rl_predictor: official settlement booked"
                    );
                    return true;
                }
                false
            }
            None => {
                // Redeemed position with outcome_won NULL — the result
                // unknown. Do not fabricate a defeat: the episode stays
                // is excluded from cohorts and closed by the normal timeout.
                m.episode.invalidate("settle_outcome_unknown");
                metrics::counter!("hrf_rl_settle_outcome_unknown_total").increment(1);
                tracing::warn!(
                    cid = %ev.cid, token = %ev.token_id,
                    "rl_predictor: settlement redeemed without outcome_won — episode excluded (unknown result)"
                );
                false
            }
        }
    }

    /// Drains the pending settlement events of the reconciliation task.
    /// Returns the cids with a NEW settlement applied.
    fn drain_settles(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        if let Some(r) = &mut self.reconcile {
            while let Ok(ev) = r.settle_rx.try_recv() {
                events.push(ev);
            }
        }
        let mut closed_ids = Vec::new();
        for ev in events {
            if self.apply_settle_event(&ev) {
                closed_ids.push(ev.cid);
            }
        }
        closed_ids
    }

    /// Accounting point: reward of the previous `pending_transition` with the
    /// equity computed over the fresh book (observed consequences) and
    /// sent to the learner. Returns the exit value used.
    fn account(&mut self, cid: &str, book: &BookSnapshot, now_ms: i64) -> i64 {
        let Some(m) = self.markets.get_mut(cid) else { return 0 };
        let mut bids: HashMap<String, BookLevelsUm> = HashMap::new();
        for (token, leg) in [(&m.yes_token, &book.yes), (&m.no_token, &book.no)] {
            if token.is_empty() {
                continue;
            }
            bids.insert(
                token.clone(),
                leg.bids
                    .levels
                    .iter()
                    .map(|l| (to_micros(l.price), to_micros(l.size)))
                    .collect(),
            );
        }
        let (exit_um, complete) = m.episode.exit_value_um(&bids, m.fee_rate);
        if !complete {
            // Position without a bids book: the remainder is worth 0 and the episode
            // is marked (quality) — "mark without liquidity".
            m.episode.invalidate("mark_without_liquidity");
        }
        let rb = m.episode.account(
            exit_um,
            now_ms,
            self.cfg.lambda_inventory,
            self.cfg.lambda_drawdown,
        );
        if let Some(mut tr) = m.pending_transition.take() {
            tr.reward_norm = rb.reward_norm;
            tr.net_pnl_delta_um = rb.net_pnl_delta_um;
            if let Some(l) = &self.learner {
                let msg = LearnerMsg::Transition(Box::new(tr));
                // cfg(test) tap: only what try_send ACCEPTED reached the
                // worker — what a full queue rejected is not experience.
                if l.send(msg.clone()) {
                    #[cfg(test)]
                    self.sent_log.lock().unwrap().push(msg);
                } else {
                    metrics::counter!("hrf_rl_learner_drops_total").increment(1);
                }
            }
        }
        exit_um
    }

    /// Episode close: terminal reward + `EpisodeEnd` + purge.
    /// `exit_um` is the residual exit value (0 after settle or without a book).
    fn close_episode(&mut self, cid: &str, trainable: bool, reason: &str, now_ms: i64, exit_um: i64) {
        let Some(mut m) = self.markets.remove(cid) else { return };
        // Terminal market: from here on neither late books nor fills
        // reopen experience for this cid (idempotency of the cycle
        // lifetime — without artificial flat episodes in the cohort).
        self.terminal.insert(cid.to_string(), m.clear_at);
        if let Some(mut tr) = m.pending_transition.take() {
            let rb = m.episode.account(
                exit_um,
                now_ms,
                self.cfg.lambda_inventory,
                self.cfg.lambda_drawdown,
            );
            tr.reward_norm = rb.reward_norm;
            tr.net_pnl_delta_um = rb.net_pnl_delta_um;
            if let Some(l) = &self.learner {
                let msg = LearnerMsg::Transition(Box::new(tr));
                if l.send(msg.clone()) {
                    #[cfg(test)]
                    self.sent_log.lock().unwrap().push(msg);
                }
            }
        }
        // `m.sampled_steps == 0` = episode without a single decision with
        // a real choice (warmup/forced or flat post-clear): there is no
        // gradient — it is closed EXCLUDED, not added to the cohort.
        let trainable = trainable && m.episode.trainable && m.sampled_steps > 0;
        let invalid_reason = m.episode.invalid_reason.clone();
        if let Some(l) = &self.learner {
            let msg = LearnerMsg::EpisodeEnd(EpisodeEndMsg {
                episode_id: cid.to_string(),
                trainable,
                reason: Some(reason.to_string()),
                invalid_reason: invalid_reason.clone(),
            });
            if l.send(msg.clone()) {
                #[cfg(test)]
                self.sent_log.lock().unwrap().push(msg);
            }
        }
        tracing::info!(
            cid = %cid, trainable, reason,
            invalid_reason = invalid_reason.as_deref().unwrap_or("-"),
            sampled_steps = m.sampled_steps,
            episode_pnl_um = m.episode.realized_pnl_um,
            "rl_predictor: episode closed"
        );
        self.push_closed_ep(ClosedEpisode {
            cid: cid.to_string(),
            reason: reason.to_string(),
            invalid_reason,
            trainable,
            pnl_um: m.episode.realized_pnl_um,
            cash_um: m.episode.cash_um,
            budget_um: m.episode.budget_um,
        });
    }

    /// `closed_eps` is observability/test history — capped to the
    /// last `CLOSED_EPS_CAP` episodes so that a long-lived process does not
    /// grow without limit (~288 closes/day on 5-min brackets).
    fn push_closed_ep(&mut self, ep: ClosedEpisode) {
        self.closed_eps.push(ep);
        if self.closed_eps.len() > CLOSED_EPS_CAP {
            let excess = self.closed_eps.len() - CLOSED_EPS_CAP;
            self.closed_eps.drain(..excess);
        }
    }

    /// Lazy sweep of `terminal` — the set grew without bound in a
    /// long-lived process (~288 cids/day). A cid is pruned once it has passed
    /// `clear_at + settle_timeout_secs + TERMINAL_RETENTION_SLACK_SECS`: a
    /// from there it is not plausible that a late book/fill arrives,
    /// legitimate, and the guard `secs_to_clear() <= 0 && !markets.contains_key`
    /// of `on_book` already prevents recreating the state anyway. The durable ledger
    /// (`fill_sink`) is immune to this pruning. The threshold
    /// `TERMINAL_PRUNE_THRESHOLD` avoids iterating the map on every tick while
    /// it is small.
    fn prune_terminal(&mut self, now_secs: i64) {
        if self.terminal.len() <= TERMINAL_PRUNE_THRESHOLD {
            return;
        }
        let horizon =
            now_secs - self.cfg.settle_timeout_secs - TERMINAL_RETENTION_SLACK_SECS;
        let antes = self.terminal.len();
        self.terminal.retain(|_, clear_at| *clear_at > horizon);
        let pruned = antes - self.terminal.len();
        if pruned > 0 {
            metrics::counter!("hrf_rl_terminal_pruned_total").increment(pruned as u64);
            tracing::debug!(
                pruned,
                remaining_len = self.terminal.len(),
                "rl_predictor: lazy pruning of `terminal`"
            );
        }
    }

    /// Applies a confirmed `Fill` to the episode's virtual ledger —
    /// economic consequence (the fill_sink already persisted it to SQLite).
    /// Rejected/GateRejected/Cancelled do not touch inventory (they resolve
    /// pending above). Returns the reserve that the pending had set.
    fn apply_fill(m: &mut MarketState, fill: &Fill) {
        match fill.side {
            FillSide::Buy => {
                let reserved = m.pending.as_ref().map(|p| p.reserved_um).unwrap_or(0);
                m.episode
                    .apply_buy_fill(&fill.token_id, fill.price, fill.size, reserved, m.fee_rate);
                metrics::counter!("hrf_rl_fills_total", "side" => "buy").increment(1);
            }
            FillSide::Sell => {
                if !m.episode
                    .apply_sell_fill(&fill.token_id, fill.price, fill.size, m.fee_rate)
                {
                    metrics::counter!("hrf_rl_ownership_ambiguous_total").increment(1);
                    tracing::warn!(
                        cid = %fill.market_cid, token = %fill.token_id,
                        "rl_predictor: SELL without enough own inventory — episode excluded from cohorts"
                    );
                }
                metrics::counter!("hrf_rl_fills_total", "side" => "sell").increment(1);
            }
            _ => {}
        }
    }

    /// Emits the `OrderIntent` of the chosen action and sets `pending`/reserve.
    /// `None` for HOLD or if the action is no longer executable.
    /// The sizes/limits are computed by `obs::action_params` — the SAME
    /// translation the offline driver uses (train/serve parity).
    fn emit_action(
        cfg: &RlPredictorConfig,
        cid: &str,
        m: &mut MarketState,
        book: &BookSnapshot,
        action: usize,
    ) -> Option<OrderIntent> {
        let now_ms = crate::time::now().timestamp_millis();
        let spec = obs::action_params(&cfg.tunables(), &m.view(), book, action)?;
        if spec.buy {
            // All-in reserve (notional + fee estimated at the limit).
            if !m.episode.reserve_buy(spec.reserve_um) {
                metrics::counter!("hrf_rl_skip_no_cash_total").increment(1);
                return None;
            }
        }
        m.pending = Some(PendingOrder {
            token_id: spec.token_id.clone(),
            reserved_um: spec.reserve_um,
            emitted_ts_ms: now_ms,
            action: spec.action,
        });
        metrics::counter!(
            "hrf_rl_intents_total",
            "side" => if spec.buy { "buy" } else { "sell" }
        )
        .increment(1);
        tracing::info!(
            target: "decision_audit",
            cid = %cid, token = %spec.token_id, action = ACTION_NAMES[spec.action],
            price = %spec.price, limit = %spec.limit, size = %spec.size,
            side = if spec.buy { "BUY" } else { "SELL" },
            "rl_predictor: intent"
        );
        Some(OrderIntent {
            market_cid: cid.to_string(),
            token_id: spec.token_id,
            side: if spec.buy {
                OrderSide::Buy
            } else {
                OrderSide::Sell { size: Some(spec.size) }
            },
            price: spec.price,
            size: spec.size,
            clear_at: book.clear_at,
            signal: Decimal::ZERO,
            neg_risk: book.neg_risk,
            limit_price: Some(spec.limit),
            second_leg: None,
            strategy: StrategyId::RlPredictor,
            resting: false,
            real_clear_at: None,
        })
    }
}

// ── Trait Strategy ───────────────────────────────────────────────────────────

impl Strategy for RlPredictor {
    fn id(&self) -> StrategyId {
        StrategyId::RlPredictor
    }

    fn requires_oracle(&self) -> bool {
        true
    }

    fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
        let cid = book.market_condition_id.clone();
        let now_ms = crate::time::now().timestamp_millis();

        // 0) Policy loaded — without an approved artifact the strategy is
        //    disabled (fail-closed; reason in `rl_strategy_status`).
        let Some(snapshot) = self.current_policy() else {
            return vec![];
        };
        if self.learner.as_ref().is_some_and(|l| l.learner_dead()) {
            // Learner down: operate with the last snapshot; position
            // management (SELL) is NOT turned off — the alarm stays in metrics.
            metrics::counter!("hrf_rl_learner_dead_ticks_total").increment(1);
        }

        // 1) Scope: exactly BTC 5m (same predicate as `model_btc5m`).
        if !is_target_market(book) {
            return vec![];
        }

        // 2) Recovery: without proven ownership there are NO decisions.
        match self.reconcile.as_ref().map(|r| r.recovery_rx.borrow().clone()) {
            None | Some(RecoveryState::Pending) => {
                metrics::counter!("hrf_rl_blocked_recovery_pending_total").increment(1);
                return vec![];
            }
            Some(RecoveryState::Failed(reason)) => {
                metrics::counter!("hrf_rl_blocked_recovery_failed_total").increment(1);
                tracing::warn!(cid = %cid, reason = %reason,
                    "rl_predictor: ownership unprovable — blocked");
                return vec![];
            }
            Some(RecoveryState::Ready { positions }) => {
                // 3) Market state (create the episode at the first book in scope).
                //    Terminal market (episode already closed): a post-clear book
                //    does NOT reopen experience — recreating it would generate
                //    artificial flat episodes that pollute the learner's cohort.
                //    Same for a cid seen FOR THE FIRST TIME already closed
                //    (`secs_to_clear() <= 0`): there was never an operable window.
                if self.terminal.contains_key(&cid)
                    || (book.secs_to_clear() <= 0 && !self.markets.contains_key(&cid))
                {
                    return vec![];
                }
                if !self.markets.contains_key(&cid) {
                    let episode = match positions.get(&cid) {
                        Some(pos) if !pos.is_empty() => EpisodeLedger::recovered(
                            cid.clone(),
                            self.cfg.initial_episode_budget,
                            pos.clone(),
                            now_ms,
                        ),
                        _ => EpisodeLedger::new(
                            cid.clone(),
                            self.cfg.initial_episode_budget,
                            now_ms,
                        ),
                    };
                    self.markets.insert(
                        cid.clone(),
                        MarketState {
                            episode,
                            yes_token: book.yes.token_id.clone(),
                            no_token: book.no.token_id.clone(),
                            clear_at: book.clear_at,
                            fee_rate: book
                                .parabolic_fee_rate_or(self.cfg.fee_rate_fallback),
                            pending: None,
                            uncertain: false,
                            pending_transition: None,
                            last_prediction: self.pred_stash.remove(&cid),
                            settle_watched: false,
                            decision_seq: 0,
                            seen_execs: HashSet::new(),
                            sampled_steps: 0,
                        },
                    );
                    if let Some(l) = &self.learner {
                        let msg = LearnerMsg::EpisodeStart {
                            episode_id: cid.clone(),
                            policy_version: snapshot.version,
                            ts_ms: now_ms,
                        };
                        if l.send(msg.clone()) {
                            #[cfg(test)]
                            self.sent_log.lock().unwrap().push(msg);
                        }
                    }
                }
            }
        }

        // Fill in token identity if the state was created by a fill
        // (fill→first book race): the contractual ids come from the book.
        if let Some(m) = self.markets.get_mut(&cid) {
            if m.yes_token.is_empty() {
                m.yes_token = book.yes.token_id.clone();
            }
            if m.no_token.is_empty() {
                m.no_token = book.no.token_id.clone();
            }
        }

        // 4) Drain accumulated predictions and settlements (non-blocking).
        self.drain_predictions();
        self.drain_settles();

        // 5) Accounting point: closes the previous transition with the
        //    observed consequences up to NOW (fresh book + fills already
        //    applied by `on_fill`, which runs earlier in the runner).
        self.account(&cid, book, now_ms);

        // 6) Decision over the fresh book.
        let (mask, obs, m_fields) = {
            let m = self.markets.get(&cid).unwrap();
            let p_yes = pred_usable(
                &self.cfg,
                book,
                m.last_prediction.as_ref(),
                &self.predictor_sha256,
            );
            let book_age_secs = book
                .recv_instant
                .and_then(|r| crate::time::Instant::now().checked_duration_since(r))
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let obs = build_obs(&self.cfg, m, book, p_yes, book_age_secs);
            let mask = mask_legal(
                &self.cfg,
                m,
                book,
                p_yes.is_some(),
                self.boundary_active(),
            );
            (mask, obs, (m.uncertain, m.pending.is_some()))
        };
        let (uncertain, has_pending) = m_fields;

        // 6a) Invalid observation (NaN/inf) → structural HOLD; no
        //     transition is recorded (NaN invalidates the observation).
        let Some(obs) = obs else {
            metrics::counter!("hrf_rl_obs_invalid_total").increment(1);
            return vec![];
        };

        // 6b) Forced-HOLD masks (uncertain state or intent in flight) and
        //     single-bit masks (warmup/pre-open, post-clear,
        //     boundary…): the action is not a policy choice → the
        //     transition is RECORDED with `sampled=false` (reward still in
        //     the trajectory for the return telescope, but it stays
        //     outside the gradient and the denominator — parity with the
        //     driver offline `trainable_step = mask_bits != 1`).
        let forced_hold = uncertain || has_pending;
        let n_legal = mask.iter().filter(|b| **b).count();
        let free_choice = !forced_hold && n_legal > 1;

        // 7) Inference: sampling (adaptive paper) or argmax (evaluation).
        let sampled = self.cfg.learning_enabled && self.learner.is_some() && !forced_hold;
        let (action, behavior_prob, probs) = if sampled {
            match snapshot.policy.sample_legal(&obs, &mask, &mut self.rng) {
                Some(v) => v,
                None => (HOLD, 1.0, [0.0; NUM_ACTIONS]),
            }
        } else {
            match snapshot.policy.argmax_legal(&obs, &mask) {
                Some((a, p)) => (a, p, {
                    let mut v = [0.0; NUM_ACTIONS];
                    // full probs for the record/KL-gate.
                    if let Some(pp) = snapshot.policy.action_probs(&obs, &mask) {
                        v = pp;
                    }
                    v
                }),
                None => (HOLD, 1.0, [0.0; NUM_ACTIONS]),
            }
        };

        // 8) Record the decision (full audit + cohort if applicable).
        let decision_seq = {
            let m = self.markets.get_mut(&cid).unwrap();
            m.decision_seq += 1;
            m.decision_seq
        };
        tracing::info!(
            target: "decision_audit",
            cid = %cid,
            seq = decision_seq,
            policy_version = snapshot.version,
            policy_sha256 = %snapshot.policy_sha256,
            action = ACTION_NAMES[action],
            behavior_prob,
            sampled,
            forced_hold,
            "rl_predictor: decision"
        );
        if self.cfg.learning_enabled && self.learner.is_some() {
            let m = self.markets.get_mut(&cid).unwrap();
            if free_choice {
                m.sampled_steps += 1;
                // Probe capture for the harness: the REAL obs/mask of the
                // first free-choice decision + version and probs
                // tied to the SAME snapshot that just decided.
                #[cfg(test)]
                if self.probe_obs.is_none() {
                    self.probe_obs = Some(ProbeCapture {
                        obs,
                        mask,
                        version: snapshot.version,
                        policy_sha256: snapshot.policy_sha256.clone(),
                        probs,
                    });
                }
            }
            m.pending_transition = Some(TransitionRec {
                episode_id: cid.clone(),
                policy_version: snapshot.version,
                obs,
                mask,
                action,
                behavior_prob,
                probs,
                sampled: free_choice,
                reward_norm: 0.0, // assigned at the next accounting point
                net_pnl_delta_um: 0,
                ts_ms: now_ms,
            });
        }

        // 9) Emit intent (HOLD → None).
        let intent = {
            let m = self.markets.get_mut(&cid).unwrap();
            Self::emit_action(&self.cfg, &cid, m, book, action)
        };
        intent.into_iter().collect()
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::RlPredictor {
            return;
        }
        let cid = fill.market_cid.clone();
        let dedup_key = fill_dedup_key(fill);
        if self.terminal.contains_key(&cid) && !self.markets.contains_key(&cid) {
            // Late fill of an episode ALREADY closed: the real consequence
            // was settled in the durable ledger by `fill_sink`; the training view
            // is not reopened (no experience is fabricated
            // post-terminal).
            metrics::counter!("hrf_rl_fill_post_terminal_total").increment(1);
            tracing::warn!(
                cid = %cid, exec = ?fill.exec_id, side = ?fill.side,
                "rl_predictor: late fill of terminal episode — durable ledger already settled it, episode not reopened"
            );
            return;
        }
        if !self.markets.contains_key(&cid) {
            // Fill of a market without state yet (fill→first book race):
            // recovered episode (not trainable) — real economic consequence,
            // never lost.
            let mut seen = HashSet::new();
            seen.insert(dedup_key);
            let mut m = MarketState {
                episode: EpisodeLedger::recovered(
                    cid.clone(),
                    self.cfg.initial_episode_budget,
                    HashMap::new(),
                    crate::time::now().timestamp_millis(),
                ),
                yes_token: String::new(),
                no_token: String::new(),
                clear_at: fill.clear_at,
                fee_rate: self.cfg.fee_rate_fallback,
                pending: None,
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: false,
                decision_seq: 0,
                seen_execs: seen,
                sampled_steps: 0,
            };
            Self::apply_fill(&mut m, fill);
            // Register the recovered episode in the learner with the
            // current version — its EpisodeEnd will count as excluded with
            // reason (`recovered_restart` → trainable=false), not as a drop
            // anonymous, and subsequent transitions will not be orphans.
            if let Some(l) = &self.learner {
                let msg = LearnerMsg::EpisodeStart {
                    episode_id: cid.clone(),
                    policy_version: l.current_version(),
                    ts_ms: crate::time::now().timestamp_millis(),
                };
                if l.send(msg.clone()) {
                    #[cfg(test)]
                    self.sent_log.lock().unwrap().push(msg);
                }
            }
            self.markets.insert(cid, m);
            return;
        }
        let m = self.markets.get_mut(&cid).unwrap();
        // Dedup by exec_id — a broadcast re-emission cannot be
        // applied twice as a real economic consequence.
        if !m.seen_execs.insert(dedup_key) {
            metrics::counter!("hrf_rl_fill_dup_total").increment(1);
            tracing::warn!(
                cid = %cid, exec = ?fill.exec_id, side = ?fill.side,
                "rl_predictor: duplicate fill ignored (exec_id already applied)"
            );
            return;
        }
        Self::apply_fill(m, fill);
        // Resolve pending: only the event of the pending's token releases it
        // (a fill of another token does not touch it). Rejected/GateRejected/
        // Cancelled also release it — the intent no longer exists.
        if let Some(p) = &m.pending {
            if p.token_id == fill.token_id {
                match fill.side {
                    FillSide::Buy | FillSide::Sell => {
                        m.pending = None;
                        m.uncertain = false;
                    }
                    FillSide::Rejected | FillSide::GateRejected | FillSide::Cancelled => {
                        m.episode.release_reservation(p.reserved_um);
                        m.pending = None;
                        m.uncertain = false;
                        metrics::counter!("hrf_rl_rejects_total").increment(1);
                    }
                }
            }
        }
    }

    fn on_timer(&mut self) -> Vec<OrderIntent> {
        let now_ms = crate::time::now().timestamp_millis();
        let now_secs = crate::time::now().timestamp();

        // 1) Settlements observed by the reconciliation task.
        self.drain_settles();

        // 2) Pending expired → uncertain state (does NOT invent a rejection).
        //    closed markets: register polling / close episode.
        let mut close: Vec<(String, bool, &'static str)> = Vec::new();
        for (cid, m) in self.markets.iter_mut() {
            if let Some(p) = &m.pending {
                if now_ms - p.emitted_ts_ms > self.cfg.pending_timeout_ms {
                    m.uncertain = true;
                    m.episode.invalidate("pending_timeout");
                    metrics::counter!("hrf_rl_uncertain_total").increment(1);
                    tracing::warn!(
                        cid = %cid, token = %p.token_id,
                        action = ACTION_NAMES[p.action],
                        "rl_predictor: intent without resolution — market in uncertain state"
                    );
                }
            }
            if now_secs <= m.clear_at {
                continue;
            }
            let with_position = m.episode.positions.values().any(|p| p.size_um > 0);
            if with_position {
                if !m.settle_watched {
                    if let Some(wtx) =
                        self.reconcile.as_ref().and_then(|r| r.watch_tx.as_ref())
                    {
                        let _ = wtx.send(SettleWatch {
                            cid: cid.clone(),
                            yes_token: m.yes_token.clone(),
                            no_token: m.no_token.clone(),
                            deadline_ms: now_ms + self.cfg.settle_timeout_secs * 1000,
                        });
                        m.settle_watched = true;
                    }
                }
                if now_secs > m.clear_at + self.cfg.settle_timeout_secs {
                    close.push((cid.clone(), false, "settle_timeout"));
                }
            } else if m.episode.positions.values().all(|p| p.size_um <= 0)
                && !m.episode.settled_tokens.is_empty()
            {
                // Position that has already been officially resolved.
                close.push((cid.clone(), true, "settled"));
            } else {
                // Episode without a position (all HOLD) — clean close after
                // clear_at; valid trajectory, HOLD also learns.
                close.push((cid.clone(), true, "closed_flat"));
            }
        }
        for (cid, trainable, reason) in close {
            self.close_episode(&cid, trainable, reason, now_ms, 0);
        }

        // 3) Lazy pruning of terminal cids already outside the window.
        self.prune_terminal(now_secs);
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::policy::{LinearSoftmaxPolicy, PolicyScaler};
    use crate::types::{HalfBook, Level, TokenBook};
    use rust_decimal_macros::dec;

    fn uniform_policy() -> LinearSoftmaxPolicy {
        LinearSoftmaxPolicy {
            scaler: PolicyScaler {
                mean: vec![0.0; NUM_FEATURES],
                scale: vec![1.0; NUM_FEATURES],
            },
            weights: vec![vec![0.0; NUM_FEATURES]; NUM_ACTIONS],
            bias: vec![0.0; NUM_ACTIONS],
        }
    }

    /// Policy that forces `action` with a strong bias — for tests of
    /// deterministic behavior (argmax).
    fn biased_policy(action: usize) -> LinearSoftmaxPolicy {
        let mut p = uniform_policy();
        p.bias[action] = 10.0;
        p
    }

    fn make_book(yes_ask: &str, no_ask: &str, secs_left: i64) -> BookSnapshot {
        let lvls = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        let pa = yes_ask.parse::<Decimal>().unwrap();
        let pna = no_ask.parse::<Decimal>().unwrap();
        let now = crate::time::now().timestamp();
        BookSnapshot {
            market_condition_id: "cid-rl".into(),
            yes: TokenBook {
                token_id: "tok-yes".into(),
                bids: lvls(&(pa - dec!(0.01)).to_string()),
                asks: lvls(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-no".into(),
                bids: lvls(&(pna - dec!(0.01)).to_string()),
                asks: lvls(no_ask),
                ..Default::default()
            },
            market_open_ts: now - (300 - secs_left),
            clear_at: now + secs_left,
            neg_risk: false,
            slug: format!("btc-updown-5m-{}", now - (300 - secs_left)),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: Some(dec!(0.07)),
            tick_size: None,
            recv_instant: Some(crate::time::Instant::now()),
        }
    }

    fn pred_for(book: &BookSnapshot, p_model: f64, hash: &str) -> ModelPrediction {
        ModelPrediction {
            schema_version: MODEL_TOOL_SCHEMA_VERSION,
            ts_ms: crate::time::now().timestamp_millis(),
            market_cid: book.market_condition_id.clone(),
            slug: book.slug.clone(),
            token_id_yes: book.yes.token_id.clone(),
            token_id_no: book.no.token_id.clone(),
            side: "x".into(),
            p_model: Some(p_model),
            p_market: Some(0.5),
            book_age_ms: 0,
            spot_age_ms: 0,
            latency_ms: 0,
            budget_ms: 0,
            abstention: None,
            model_hash: Some(hash.into()),
        }
    }

    fn cfg_test() -> RlPredictorConfig {
        RlPredictorConfig::default()
    }

    fn strat_with_policy(policy: LinearSoftmaxPolicy) -> RlPredictor {
        let (_tx, rx) = broadcast::channel(16);
        RlPredictor::new_test(
            cfg_test(),
            Some(rx),
            policy,
            RecoveryState::Ready {
                positions: HashMap::new(),
            },
            None,
        )
    }

    fn vol() -> VolSnapshot {
        VolSnapshot::default()
    }

    /// Without a usable prediction, the policy can only choose HOLD or SELL —
    /// BUY stays masked (`p_missing` allows SELL without a loaded model).
    #[test]
    fn without_predictor_buy_masked() {
        let mut s = strat_with_policy(biased_policy(BUY_YES_FULL));
        let book = make_book("0.50", "0.50", 120);
        let intents = s.on_book(&book, &vol());
        assert!(
            intents.is_empty(),
            "without a usable prediction it cannot buy — argmax falls back to HOLD"
        );
        let m = s.markets.get("cid-rl").unwrap();
        let mask = mask_legal(&s.cfg, m, &book, false, false);
        assert!(!mask[BUY_YES_FULL] && !mask[BUY_YES_SMALL]);
        assert!(mask[HOLD]);
    }

    /// With a fresh prediction and a policy biased to BUY_YES_FULL → Buy intent.
    #[test]
    fn buy_with_predictor_and_policy() {
        let mut s = strat_with_policy(biased_policy(BUY_YES_FULL));
        s.predictor_sha256 = "hh".into();
        let book = make_book("0.50", "0.50", 120);
        let mut m0 = MarketState {
            episode: EpisodeLedger::new("cid-rl", dec!(10), 0),
            yes_token: "tok-yes".into(),
            no_token: "tok-no".into(),
            clear_at: book.clear_at,
            fee_rate: dec!(0.07),
            pending: None,
            uncertain: false,
            pending_transition: None,
            last_prediction: Some(pred_for(&book, 0.8, "hh")),
            settle_watched: false,
            decision_seq: 0,
            seen_execs: HashSet::new(),
            sampled_steps: 0,
        };
        let mask = mask_legal(&s.cfg, &m0, &book, true, false);
        assert!(mask[BUY_YES_FULL], "BUY_YES_FULL must be legal here");
        let intent = RlPredictor::emit_action(&s.cfg, "cid-rl", &mut m0, &book, BUY_YES_FULL)
            .expect("must emit a BUY intent");
        assert!(matches!(intent.side, OrderSide::Buy));
        assert_eq!(intent.token_id, "tok-yes");
        assert!(intent.size > Decimal::ZERO);
        assert!(m0.pending.is_some(), "pending stays latched after emitting");
        assert!(m0.episode.reserved() > Decimal::ZERO);
    }

    /// A BUY fill confirms the position and resolves the pending.
    #[test]
    fn fill_buy_updates_ledger_and_pending() {
        let mut s = strat_with_policy(biased_policy(BUY_YES_FULL));
        s.predictor_sha256 = "hh".into();
        let book = make_book("0.50", "0.50", 120);
        {
            let m = s.markets.entry("cid-rl".into()).or_insert_with(|| MarketState {
                episode: EpisodeLedger::new("cid-rl", dec!(10), 0),
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: book.clear_at,
                fee_rate: dec!(0.07),
                pending: Some(PendingOrder {
                    token_id: "tok-yes".into(),
                    reserved_um: to_micros(dec!(5.20)),
                    emitted_ts_ms: 0,
                    action: BUY_YES_FULL,
                }),
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: false,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            });
            m.episode.reserve_buy(to_micros(dec!(5.20)));
        }
        s.on_fill(&Fill {
            market_cid: "cid-rl".into(),
            token_id: "tok-yes".into(),
            side: FillSide::Buy,
            price: dec!(0.5),
            size: dec!(10),
            order_id: Some("o1".into()),
            ts: crate::time::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RlPredictor,
            clear_at: book.clear_at,
            outcome_label: None,
            hedge_id: None,
            exec_id: Some("e1".into()),
            local_order_id: None,
        });
        let m = s.markets.get("cid-rl").unwrap();
        assert!(m.pending.is_none(), "fill resolves the pending");
        assert_eq!(m.episode.shares_of("tok-yes"), dec!(10));
        assert!(m.episode.cost_of("tok-yes") > Decimal::ZERO);
    }

    /// GateRejected/Rejected releases the pending and returns the reserve.
    #[test]
    fn reject_releases_pending_and_reserve() {
        let mut s = strat_with_policy(biased_policy(BUY_YES_FULL));
        let book = make_book("0.50", "0.50", 120);
        {
            let m = s.markets.entry("cid-rl".into()).or_insert_with(|| MarketState {
                episode: EpisodeLedger::new("cid-rl", dec!(10), 0),
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: book.clear_at,
                fee_rate: dec!(0.07),
                pending: Some(PendingOrder {
                    token_id: "tok-yes".into(),
                    reserved_um: to_micros(dec!(5.20)),
                    emitted_ts_ms: 0,
                    action: BUY_YES_FULL,
                }),
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: false,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            });
            m.episode.reserve_buy(to_micros(dec!(5.20)));
        }
        s.on_fill(&Fill {
            market_cid: "cid-rl".into(),
            token_id: "tok-yes".into(),
            side: FillSide::GateRejected,
            price: dec!(0.5),
            size: dec!(10),
            order_id: None,
            ts: crate::time::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RlPredictor,
            clear_at: book.clear_at,
            outcome_label: None,
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
        });
        let m = s.markets.get("cid-rl").unwrap();
        assert!(m.pending.is_none());
        assert_eq!(m.episode.reserved(), Decimal::ZERO);
        assert_eq!(m.episode.shares_of("tok-yes"), Decimal::ZERO);
    }

    /// Regression: two identical `GateRejected` events (same price,
    /// size, ts_ms) are TWO distinct events — the executor stamps them with a
    /// unique `exec_id` (`gate-<ns>-<n>`). The second result must NOT be swallowed by
    /// the dedup: each one releases the pending of its intent. If they were merged,
    /// the pending would hang until `pending_timeout` and the episode
    /// would be invalidated without touching inventory.
    #[test]
    fn two_identical_gate_rejected_release_each_pending() {
        let mut s = strat_with_policy(biased_policy(BUY_YES_FULL));
        let book = make_book("0.50", "0.50", 120);
        let ts = crate::time::now();
        let mk_gate = |exec_id: &str| Fill {
            market_cid: "cid-rl".into(),
            token_id: "tok-yes".into(),
            side: FillSide::GateRejected,
            price: dec!(0.41),
            size: dec!(5.07),
            order_id: None,
            ts,
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RlPredictor,
            clear_at: book.clear_at,
            outcome_label: None,
            hedge_id: None,
            exec_id: Some(exec_id.into()),
            local_order_id: None,
        };
        s.markets.insert(
            "cid-rl".into(),
            MarketState {
                episode: EpisodeLedger::new("cid-rl", dec!(10), 0),
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: book.clear_at,
                fee_rate: dec!(0.07),
                pending: None,
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: false,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            },
        );
        // Two consecutive intents rejected by the same gate in the same
        // ms — each one marks its own pending.
        for exec_id in ["gate-1", "gate-2"] {
            {
                let m = s.markets.get_mut("cid-rl").unwrap();
                m.episode.reserve_buy(to_micros(dec!(5.20)));
                m.pending = Some(PendingOrder {
                    token_id: "tok-yes".into(),
                    reserved_um: to_micros(dec!(5.20)),
                    emitted_ts_ms: 0,
                    action: BUY_YES_FULL,
                });
            }
            s.on_fill(&mk_gate(exec_id));
            let m = s.markets.get("cid-rl").unwrap();
            assert!(
                m.pending.is_none(),
                "each GateRejected releases its pending ({exec_id})"
            );
        }
        let m = s.markets.get("cid-rl").unwrap();
        assert_eq!(m.episode.reserved(), Decimal::ZERO);
        assert!(
            m.episode.invalid_reason.is_none(),
            "correct rejections do not invalidate the episode"
        );
        assert!(!m.uncertain, "without pending_timeout there is no forced HOLD");
    }

    /// Regression of the legacy dedup: two pre-submit `Rejected` events for the SAME
    /// order in the same ms carry a different `local_order_id` (attempt_id) —
    /// the legacy key tells them apart and both resolve their pending.
    #[test]
    fn dedup_legacy_distinguishes_rejected_by_local_order_id() {
        let mut s = strat_with_policy(biased_policy(BUY_YES_FULL));
        let book = make_book("0.50", "0.50", 120);
        let ts = crate::time::now();
        let mk_rej = |local_id: &str| Fill {
            market_cid: "cid-rl".into(),
            token_id: "tok-yes".into(),
            side: FillSide::Rejected,
            price: dec!(0.5),
            size: dec!(5.07),
            order_id: None,
            ts,
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RlPredictor,
            clear_at: book.clear_at,
            outcome_label: None,
            hedge_id: None,
            exec_id: None,
            local_order_id: Some(local_id.into()),
        };
        s.markets.insert(
            "cid-rl".into(),
            MarketState {
                episode: EpisodeLedger::new("cid-rl", dec!(10), 0),
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: book.clear_at,
                fee_rate: dec!(0.07),
                pending: None,
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: false,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            },
        );
        for local_id in ["loc-1", "loc-2"] {
            {
                let m = s.markets.get_mut("cid-rl").unwrap();
                m.episode.reserve_buy(to_micros(dec!(5.20)));
                m.pending = Some(PendingOrder {
                    token_id: "tok-yes".into(),
                    reserved_um: to_micros(dec!(5.20)),
                    emitted_ts_ms: 0,
                    action: BUY_YES_FULL,
                });
            }
            s.on_fill(&mk_rej(local_id));
            assert!(
                s.markets["cid-rl"].pending.is_none(),
                "Rejected with local_order_id {local_id} releases its pending"
            );
        }
        // The exact same event re-sent DOES deduplicate (idempotent replay).
        s.on_fill(&mk_rej("loc-2"));
        let m = s.markets.get("cid-rl").unwrap();
        assert_eq!(
            m.seen_execs.len(),
            2,
            "the exact re-send adds no new key — dedup intact"
        );
        assert_eq!(m.episode.reserved(), Decimal::ZERO);
    }

    /// With an own position and no predictor, SELL_YES_ALL stays legal and is
    /// emitted — the exit never depends on the loaded model.
    #[test]
    fn sell_legal_without_predictor() {
        let mut s = strat_with_policy(biased_policy(SELL_YES_ALL));
        let book = make_book("0.50", "0.50", 120);
        {
            let mut ep = EpisodeLedger::new("cid-rl", dec!(10), 0);
            ep.apply_buy_fill("tok-yes", dec!(0.5), dec!(10), 0, dec!(0.07));
            s.markets.insert(
                "cid-rl".into(),
                MarketState {
                    episode: ep,
                    yes_token: "tok-yes".into(),
                    no_token: "tok-no".into(),
                    clear_at: book.clear_at,
                    fee_rate: dec!(0.07),
                    pending: None,
                    uncertain: false,
                    pending_transition: None,
                    last_prediction: None,
                    settle_watched: false,
                    decision_seq: 0,
                    seen_execs: HashSet::new(),
                sampled_steps: 0,
                },
            );
        }
        // Without any prediction: the argmax falls over SELL_YES_ALL (bias).
        let intents = s.on_book(&book, &vol());
        assert_eq!(intents.len(), 1);
        assert!(matches!(intents[0].side, OrderSide::Sell { .. }));
        assert_eq!(intents[0].token_id, "tok-yes");
        assert_eq!(intents[0].size, dec!(10));
    }

    /// Pending recovery: ALL actions stay blocked (not even
    /// the market state is created).
    #[test]
    fn pending_recovery_blocks() {
        let (_tx, rx) = broadcast::channel(16);
        let mut s = RlPredictor::new_test(
            cfg_test(),
            Some(rx),
            biased_policy(BUY_YES_FULL),
            RecoveryState::Pending,
            None,
        );
        let book = make_book("0.50", "0.50", 120);
        assert!(s.on_book(&book, &vol()).is_empty());
        assert!(s.markets.is_empty(), "no state is created without recovery");
    }

    /// At most one pending intent per market: after emitting a BUY, the next
    /// book with a pending gets a HOLD-only mask (forced, no transition).
    #[test]
    fn one_pending_per_market() {
        let mut s = strat_with_policy(biased_policy(BUY_YES_FULL));
        s.predictor_sha256 = "hh".into();
        let book = make_book("0.50", "0.50", 120);
        {
            let mut ep = EpisodeLedger::new("cid-rl", dec!(10), 0);
            ep.reserve_buy(to_micros(dec!(5.2)));
            s.markets.insert(
                "cid-rl".into(),
                MarketState {
                    episode: ep,
                    yes_token: "tok-yes".into(),
                    no_token: "tok-no".into(),
                    clear_at: book.clear_at,
                    fee_rate: dec!(0.07),
                    pending: Some(PendingOrder {
                        token_id: "tok-yes".into(),
                        reserved_um: to_micros(dec!(5.2)),
                        emitted_ts_ms: crate::time::now().timestamp_millis(),
                        action: BUY_YES_FULL,
                    }),
                    uncertain: false,
                    pending_transition: None,
                    last_prediction: Some(pred_for(&book, 0.9, "hh")),
                    settle_watched: false,
                    decision_seq: 0,
                    seen_execs: HashSet::new(),
                sampled_steps: 0,
                },
            );
        }
        let intents = s.on_book(&book, &vol());
        assert!(intents.is_empty(), "no other intent is emitted while pending");
    }

    /// Uncertain state (expired pending): blocks even SELL of inventory.
    #[test]
    fn uncertain_blocks_all() {
        let mut s = strat_with_policy(biased_policy(SELL_YES_ALL));
        let book = make_book("0.50", "0.50", 120);
        let mut ep = EpisodeLedger::new("cid-rl", dec!(10), 0);
        ep.apply_buy_fill("tok-yes", dec!(0.5), dec!(10), 0, dec!(0.07));
        s.markets.insert(
            "cid-rl".into(),
            MarketState {
                episode: ep,
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: book.clear_at,
                fee_rate: dec!(0.07),
                pending: None,
                uncertain: true,
                pending_transition: None,
                last_prediction: None,
                settle_watched: false,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            },
        );
        let intents = s.on_book(&book, &vol());
        assert!(intents.is_empty(), "uncertain state blocks ALL orders");
    }

    /// The observation has exactly the 22 contractual features with
    /// `p_missing=1` when there is no usable prediction.
    #[test]
    fn obs_contractual_with_p_missing() {
        let book = make_book("0.50", "0.55", 120);
        let ep = EpisodeLedger::new("cid-rl", dec!(10), 0);
        let m = MarketState {
            episode: ep,
            yes_token: "tok-yes".into(),
            no_token: "tok-no".into(),
            clear_at: book.clear_at,
            fee_rate: dec!(0.07),
            pending: None,
            uncertain: false,
            pending_transition: None,
            last_prediction: None,
            settle_watched: false,
            decision_seq: 0,
            seen_execs: HashSet::new(),
            sampled_steps: 0,
        };
        let cfg = cfg_test();
        let obs = build_obs(&cfg, &m, &book, None, 0.5).unwrap();
        assert_eq!(obs.len(), NUM_FEATURES);
        assert_eq!(obs[F_P_YES], 0.5, "p_yes=0.5 if missing");
        assert_eq!(obs[F_P_MISSING], 1.0);
        assert_eq!(obs[F_YES_ASK], 0.5);
        assert_eq!(obs[F_NO_ASK], 0.55);
        assert_eq!(obs[F_BUY_FEE_RATE], 0.07);
        assert!(obs[F_TIME_REMAINING] > 0.0 && obs[F_TIME_REMAINING] <= 1.0);
    }

    /// Prediction from a different predictor (`model_hash` ≠ `predictor_sha256`
    /// of the artifact) is rejected: the policy was trained against another predictor.
    #[test]
    fn different_pred_hash_rejects() {
        let cfg = cfg_test();
        let book = make_book("0.50", "0.55", 120);
        let pred = pred_for(&book, 0.9, "other-hash");
        assert!(
            pred_usable(&cfg, &book, Some(&pred), "hash-of-artifact").is_none(),
            "different predictor → feature missing (BUYs masked)"
        );
    }

    /// End-to-end idempotent settlement in the strategy: two events of the
    /// same token do not duplicate the payment or break the episode.
    #[test]
    fn settle_idempotent_strategy() {
        let mut s = strat_with_policy(biased_policy(HOLD));
        let mut ep = EpisodeLedger::new("cid-rl", dec!(10), 0);
        ep.apply_buy_fill("tok-yes", dec!(0.5), dec!(10), 0, dec!(0.07));
        s.markets.insert(
            "cid-rl".into(),
            MarketState {
                episode: ep,
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: 0,
                fee_rate: dec!(0.07),
                pending: None,
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: true,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            },
        );
        let cash_before = s.markets["cid-rl"].episode.cash_um;
        let ev = SettleEvent {
            cid: "cid-rl".into(),
            token_id: "tok-yes".into(),
            won: Some(true),
        };
        if let Some(m) = s.markets.get_mut("cid-rl") {
            let won = ev.won.unwrap();
            assert!(m.episode.apply_settlement(&ev.token_id, won));
            assert!(!m.episode.apply_settlement(&ev.token_id, won));
        }
        assert_eq!(
            s.markets["cid-rl"].episode.cash_um,
            cash_before + to_micros(dec!(10)),
            "paid only once even if the event arrives duplicated"
        );
    }

    fn mk_fill(
        cid: &str,
        token: &str,
        side: FillSide,
        order_id: Option<&str>,
        exec_id: Option<&str>,
        price: Decimal,
        size: Decimal,
    ) -> Fill {
        Fill {
            market_cid: cid.into(),
            token_id: token.into(),
            side,
            price,
            size,
            order_id: order_id.map(str::to_string),
            ts: crate::time::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::RlPredictor,
            clear_at: 0,
            outcome_label: None,
            hedge_id: None,
            exec_id: exec_id.map(str::to_string),
            local_order_id: None,
        }
    }

    /// A fill of a market without state (fill→first book race, restart
    /// mid-flight) creates a RECOVERED episode: the economic consequence is applied,
    /// and an explicit `EpisodeStart` is sent to the learner with the
    /// current version and the episode closes as excluded — the inherited
    /// trajectory is never attributed to the new policy.
    #[test]
    fn fill_without_state_creates_recorded_recovered_episode() {
        let (handle, mut rx) = crate::rl::learner::handle_for_test(uniform_policy());
        let (_ptx, prx) = broadcast::channel(16);
        let mut s = RlPredictor::new_test(
            cfg_test(),
            Some(prx),
            uniform_policy(),
            RecoveryState::Ready {
                positions: HashMap::new(),
            },
            Some(handle),
        );
        s.on_fill(&mk_fill(
            "cid-rl",
            "tok-x",
            FillSide::Buy,
            Some("o1"),
            Some("e1"),
            dec!(0.5),
            dec!(4),
        ));
        match rx.try_recv() {
            Ok(LearnerMsg::EpisodeStart { episode_id, .. }) => {
                assert_eq!(episode_id, "cid-rl", "recovered episode registered explicitly")
            }
            other => panic!("expected EpisodeStart, got {other:?}"),
        }
        let m = s.markets.get("cid-rl").unwrap();
        assert!(!m.episode.trainable, "recovered episode does not train");
        assert_eq!(
            m.episode.invalid_reason.as_deref(),
            Some("recovered_restart")
        );
        assert_eq!(
            m.episode.shares_of("tok-x"),
            dec!(4),
            "the economic consequence of the fill is kept"
        );
        // The close notifies EpisodeEnd with trainable=false (excluded with
        // reason — the learner's caller counts `episodes_excluded`).
        s.close_episode(
            "cid-rl",
            true,
            "closed_flat",
            crate::time::now().timestamp_millis(),
            0,
        );
        match rx.try_recv() {
            Ok(LearnerMsg::EpisodeEnd(e)) => {
                assert_eq!(e.episode_id, "cid-rl");
                assert!(!e.trainable, "recovered must be closed as excluded");
                assert_eq!(e.reason.as_deref(), Some("closed_flat"));
            }
            other => panic!("expected EpisodeEnd, got {other:?}"),
        }
    }

    /// Dedup: a re-emission of the same `exec_id` (duplicated broadcast) is
    /// ignored — the economic consequence is applied only once.
    #[test]
    fn duplicate_exec_id_does_not_duplicate_position() {
        let mut s = strat_with_policy(biased_policy(HOLD));
        let book = make_book("0.50", "0.50", 120);
        s.markets.insert(
            "cid-rl".into(),
            MarketState {
                episode: EpisodeLedger::new("cid-rl", dec!(10), 0),
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: book.clear_at,
                fee_rate: dec!(0.07),
                pending: None,
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: false,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            },
        );
        let f = mk_fill(
            "cid-rl",
            "tok-yes",
            FillSide::Buy,
            Some("o1"),
            Some("e1"),
            dec!(0.5),
            dec!(4),
        );
        s.on_fill(&f);
        s.on_fill(&f); // identical broadcast re-emission
        let ep = &s.markets["cid-rl"].episode;
        assert_eq!(ep.shares_of("tok-yes"), dec!(4), "the duplicate is not counted");
        // Cash charged once: `fill.price` is ALL-INCLUSIVE
        // (sim cash/shares — StateStore convention): the debit is
        // usdc = 0.5·4 = 2.0 exactly, with no extra fee (already included).
        assert_eq!(ep.cash_um, to_micros(dec!(10) - dec!(2)));
        // The estimated fee is accounted for ONCE as an audit entry, never
        // as an extra charge over the all-in price.
        let fee_est = fee_charged_5dp(dec!(0.07), dec!(0.5), dec!(4));
        assert_eq!(ep.fees_paid_um, to_micros(fee_est));
    }

    /// Two DISTINCT executions of the same `order_id` (FAK partial fills)
    /// are both applied — the dedup is by `exec_id`, never by `order_id`.
    #[test]
    fn partials_distinct_exec_id_both_apply() {
        let mut s = strat_with_policy(biased_policy(HOLD));
        let book = make_book("0.50", "0.50", 120);
        s.markets.insert(
            "cid-rl".into(),
            MarketState {
                episode: EpisodeLedger::new("cid-rl", dec!(10), 0),
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: book.clear_at,
                fee_rate: dec!(0.07),
                pending: None,
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: false,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            },
        );
        s.on_fill(&mk_fill(
            "cid-rl",
            "tok-yes",
            FillSide::Buy,
            Some("o1"),
            Some("e1"),
            dec!(0.5),
            dec!(4),
        ));
        s.on_fill(&mk_fill(
            "cid-rl",
            "tok-yes",
            FillSide::Buy,
            Some("o1"), // same order_id
            Some("e2"), // distinct exec_id → legitimate execution
            dec!(0.5),
            dec!(6),
        ));
        assert_eq!(
            s.markets["cid-rl"].episode.shares_of("tok-yes"),
            dec!(10),
            "both partial executions are accounted for"
        );
    }

    /// `redeemed` with `outcome_won=NULL` (`won=None`) is NOT a silent
    /// defeat: the episode is invalidated for cohorts, the cash is not
    /// touched and the event does not count as a resolved settlement.
    #[test]
    fn redeemed_without_outcome_excludes_without_touching_cash() {
        let mut s = strat_with_policy(biased_policy(HOLD));
        let mut ep = EpisodeLedger::new("cid-rl", dec!(10), 0);
        ep.apply_buy_fill("tok-yes", dec!(0.5), dec!(10), 0, dec!(0.07));
        s.markets.insert(
            "cid-rl".into(),
            MarketState {
                episode: ep,
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: 0,
                fee_rate: dec!(0.07),
                pending: None,
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: true,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            },
        );
        let cash_before = s.markets["cid-rl"].episode.cash_um;
        let was_resolved = s.apply_settle_event(&SettleEvent {
            cid: "cid-rl".into(),
            token_id: "tok-yes".into(),
            won: None,
        });
        assert!(!was_resolved, "unknown outcome does not resolve the episode");
        let ep = &s.markets["cid-rl"].episode;
        assert!(!ep.trainable, "episode excluded from cohorts");
        assert_eq!(
            ep.invalid_reason.as_deref(),
            Some("settle_outcome_unknown")
        );
        assert_eq!(ep.cash_um, cash_before, "no defeat or prize is fabricated");
        // The token is NOT marked settled — a later real outcome
        // can still reconcile (idempotent settlement via apply_settlement).
        assert!(!ep.settled_tokens.contains_key("tok-yes"));
    }

    /// Complement: `Some(false)` IS a known defeat — it resolves
    /// the episode (position at 0, cash intact) and it stays trainable.
    #[test]
    fn losing_outcome_resolves_episode() {
        let mut s = strat_with_policy(biased_policy(HOLD));
        let mut ep = EpisodeLedger::new("cid-rl", dec!(10), 0);
        ep.apply_buy_fill("tok-yes", dec!(0.5), dec!(10), 0, dec!(0.07));
        s.markets.insert(
            "cid-rl".into(),
            MarketState {
                episode: ep,
                yes_token: "tok-yes".into(),
                no_token: "tok-no".into(),
                clear_at: 0,
                fee_rate: dec!(0.07),
                pending: None,
                uncertain: false,
                pending_transition: None,
                last_prediction: None,
                settle_watched: true,
                decision_seq: 0,
                seen_execs: HashSet::new(),
                sampled_steps: 0,
            },
        );
        let cash_before = s.markets["cid-rl"].episode.cash_um;
        let was_resolved = s.apply_settle_event(&SettleEvent {
            cid: "cid-rl".into(),
            token_id: "tok-yes".into(),
            won: Some(false),
        });
        assert!(was_resolved, "known defeat resolves the episode");
        let ep = &s.markets["cid-rl"].episode;
        assert_eq!(ep.cash_um, cash_before, "defeat neither credits nor re-charges");
        assert_eq!(ep.shares_of("tok-yes"), Decimal::ZERO);
        assert!(ep.trainable, "episode resolved DOES train");
    }

    /// Idempotent post-clear lifecycle: once the episode is closed
    /// of a cid, later books and timers do NOT recreate `MarketState` nor
    /// `EpisodeStart`/`EpisodeEnd` — and a late fill does not reopen the training
    /// view (the real consequence already lives in the ledger
    /// durable). Without this the learner fills up with artificial flat
    /// episodes that produce updates with zero gradient.
    #[test]
    fn postclear_book_timer_is_idempotent() {
        let (handle, mut rx) = crate::rl::learner::handle_for_test(uniform_policy());
        let (_ptx, prx) = broadcast::channel(16);
        let mut cfg = cfg_test();
        cfg.learning_enabled = true;
        let mut s = RlPredictor::new_test(
            cfg,
            Some(prx),
            uniform_policy(),
            RecoveryState::Ready {
                positions: HashMap::new(),
            },
            Some(handle),
        );
        let v = vol();

        // a) A cid seen FOR THE FIRST TIME already closed does not open an episode.
        let closed_book = make_book("0.50", "0.50", -5);
        assert!(s.on_book(&closed_book, &v).is_empty());
        assert!(
            s.markets.is_empty(),
            "first contact post-clear must not create state"
        );
        assert!(
            rx.try_recv().is_err(),
            "EpisodeStart must not be emitted for a terminal cid"
        );

        // b) Real episode: pre-clear book creates the market.
        let open_book = make_book("0.50", "0.50", 120);
        let _ = s.on_book(&open_book, &v);
        match rx.try_recv() {
            Ok(LearnerMsg::EpisodeStart { episode_id, .. }) => {
                assert_eq!(episode_id, "cid-rl")
            }
            other => panic!("expected EpisodeStart, got {other:?}"),
        }

        // c) Episode close (as on_timer does after clear_at).
        //    `close` first flushes the pending transition and then emits
        //    EpisodeEnd — drain until it is found.
        s.close_episode(
            "cid-rl",
            true,
            "closed_flat",
            crate::time::now().timestamp_millis(),
            0,
        );
        let mut end = None;
        while let Ok(m) = rx.try_recv() {
            if let LearnerMsg::EpisodeEnd(e) = m {
                end = Some(e);
            }
        }
        let Some(e) = end else {
            panic!("expected EpisodeEnd, never arrived")
        };
        assert_eq!(e.episode_id, "cid-rl");
        assert!(s.markets.is_empty());

        // d) Repeat book + post-clear timer: NOTHING must be recreated.
        for _ in 0..4 {
            let _ = s.on_book(&closed_book, &v);
            let _ = s.on_timer();
        }
        assert!(s.markets.is_empty(), "terminal market is not recreated");
        assert!(
            rx.try_recv().is_err(),
            "post-clear must not emit extra EpisodeStart/Transition/EpisodeEnd"
        );
        assert_eq!(s.closed_eps.len(), 1, "only one closed episode");

        // e) Late fill of the terminal cid: does not reopen the view (warn).
        s.on_fill(&mk_fill(
            "cid-rl",
            "tok-yes",
            FillSide::Buy,
            Some("o-late"),
            Some("e-late"),
            dec!(0.5),
            dec!(4),
        ));
        assert!(s.markets.is_empty(), "late fill does not recreate the episode");
        assert!(rx.try_recv().is_err(), "late fill does not emit EpisodeStart");
    }

    /// Warmup gate: decisions with a SINGLE-bit mask (HOLD-only,
    /// pre-open `secs_left > max_remaining_secs`) are recorded with
    /// `sampled=false` and an episode without ANY free choice closes
    /// `trainable=false` — it never pollutes the cohort with a zero gradient.
    #[test]
    fn warmup_episode_without_choice_closes_excluded() {
        let (handle, mut rx) = crate::rl::learner::handle_for_test(uniform_policy());
        let (_ptx, prx) = broadcast::channel(16);
        let mut cfg = cfg_test();
        cfg.learning_enabled = true;
        let mut s = RlPredictor::new_test(
            cfg,
            Some(prx),
            biased_policy(BUY_YES_FULL), // even though the policy wants to buy
            RecoveryState::Ready {
                positions: HashMap::new(),
            },
            Some(handle),
        );
        let v = vol();
        // Pre-open: 300 s remaining > max_remaining_secs (240) → HOLD-only.
        let book_warmup = make_book("0.50", "0.50", 300);
        let _ = s.on_book(&book_warmup, &v);
        match rx.try_recv() {
            Ok(LearnerMsg::EpisodeStart { .. }) => {}
            other => panic!("expected EpisodeStart, got {other:?}"),
        }
        let m = s.markets.get("cid-rl").unwrap();
        assert_eq!(m.sampled_steps, 0, "warmup does not count as a free choice");
        assert!(
            m.pending_transition
                .as_ref()
                .is_some_and(|t| !t.sampled),
            "warmup transition recorded with sampled=false"
        );
        s.close_episode(
            "cid-rl",
            true,
            "closed_flat",
            crate::time::now().timestamp_millis(),
            0,
        );
        // `close` flushes the pending transition before EpisodeEnd.
        let mut end = None;
        while let Ok(m) = rx.try_recv() {
            if let LearnerMsg::EpisodeEnd(e) = m {
                end = Some(e);
            }
        }
        let Some(e) = end else {
            panic!("expected EpisodeEnd, never arrived")
        };
        assert!(!e.trainable, "episode without a free choice does not train");
    }

    /// Counterpart of the warmup gate: a decision with ≥2 legal actions
    /// (usable predictor + entry window) IS recorded as `sampled=true`.
    #[test]
    fn decision_with_free_choice_is_sampled() {
        let (handle, mut rx) = crate::rl::learner::handle_for_test(uniform_policy());
        let (ptx, prx) = broadcast::channel(16);
        let mut cfg = cfg_test();
        cfg.learning_enabled = true;
        let mut s = RlPredictor::new_test(
            cfg,
            Some(prx),
            uniform_policy(),
            RecoveryState::Ready {
                positions: HashMap::new(),
            },
            Some(handle),
        );
        s.predictor_sha256 = "hh".into();
        let book = make_book("0.50", "0.50", 120);
        let _ = ptx.send(pred_for(&book, 0.8, "hh"));
        let _ = s.on_book(&book, &vol());
        match rx.try_recv() {
            Ok(LearnerMsg::EpisodeStart { .. }) => {}
            other => panic!("expected EpisodeStart, got {other:?}"),
        }
        let m = s.markets.get("cid-rl").unwrap();
        let tr = m.pending_transition.as_ref().expect("transition recorded");
        let n_legal = tr.mask.iter().filter(|b| **b).count();
        assert!(n_legal >= 2, "the test mask must have ≥2 legal actions");
        assert!(tr.sampled, "a free choice must be marked sampled=true");
        assert_eq!(m.sampled_steps, 1);
        assert!(s.probe_obs.is_some(), "the probe captured the real obs");
    }

    // ── Bounded retention of `terminal` and `closed_eps` ──────────────

    /// The lazy sweep of `terminal` removes the entries whose
    /// `clear_at + settle_timeout + slack` has passed — a long-lived process
    /// no longer accumulates them without limit (~288 cids/day).
    #[test]
    fn terminal_prune_respects_horizon() {
        let mut s = strat_with_policy(biased_policy(HOLD));
        let now_secs: i64 = 1_700_000_000;
        let horizon = now_secs - s.cfg.settle_timeout_secs - TERMINAL_RETENTION_SLACK_SECS;
        // Populate above the threshold: half already expired, half recent.
        let total = TERMINAL_PRUNE_THRESHOLD + 20;
        let recent = total / 2;
        for i in 0..total {
            let clear_at = if i % 2 == 0 {
                horizon - 60 // expired: outside the late-fill window
            } else {
                now_secs - 10 // recent: a late book/fill is still plausible
            };
            s.terminal.insert(format!("cid-{i:05}"), clear_at);
        }
        // Legitimate cid with a LONG name and a recent clear_at: pruning is by
        // time, never by the shape of the string.
        let long_cid = format!("0x{}", "ab".repeat(256));
        s.terminal.insert(long_cid.clone(), now_secs - 10);
        let pre = s.terminal.len();

        s.prune_terminal(now_secs);

        assert_eq!(s.terminal.len(), recent + 1);
        assert_eq!(pre - s.terminal.len(), total / 2);
        assert!(
            s.terminal.contains_key(&long_cid),
            "a recent legitimate cid (even with a long string) is not pruned"
        );
        assert!(s.terminal.contains_key("cid-00001")); // odd → recent
        assert!(!s.terminal.contains_key("cid-00000")); // even → expired
    }

    /// The sweep is lazy — below the threshold it neither iterates nor
    /// removes nothing (an expired cid not yet swept is still
    /// recognized as terminal by `on_book`, which is what matters).
    #[test]
    fn terminal_prune_is_lazy() {
        let mut s = strat_with_policy(biased_policy(HOLD));
        let now_secs: i64 = 1_700_000_000;
        let horizon = now_secs - s.cfg.settle_timeout_secs - TERMINAL_RETENTION_SLACK_SECS;
        for i in 0..10 {
            s.terminal.insert(format!("stale-ts-{i}"), horizon - 1_000);
        }
        s.prune_terminal(now_secs);
        assert_eq!(
            s.terminal.len(),
            10,
            "below the threshold the sweep does not run (it could remove 10)"
        );
    }

    /// `closed_eps` keeps at most the last `CLOSED_EPS_CAP`
    /// episodes — the Vec no longer grows without bound in a long-lived process.
    /// Exercises `push_closed_ep` (the same path as `close_episode`):
    /// the most recent ones — those read by tests/observability — are
    /// kept.
    #[test]
    fn closed_eps_caps_to_last() {
        let mut s = strat_with_policy(biased_policy(HOLD));
        for i in 0..(CLOSED_EPS_CAP + 10) {
            s.push_closed_ep(ClosedEpisode {
                cid: format!("ep-{i}"),
                reason: "closed_flat".into(),
                invalid_reason: None,
                trainable: true,
                pnl_um: i as i64,
                cash_um: 0,
                budget_um: 0,
            });
        }
        assert_eq!(s.closed_eps.len(), CLOSED_EPS_CAP);
        assert_eq!(s.closed_eps[0].cid, "ep-10", "the oldest ones are discarded");
        assert_eq!(
            s.closed_eps.last().unwrap().cid,
            format!("ep-{}", CLOSED_EPS_CAP + 9),
            "the most recent ones are kept"
        );
        // A LONG legitimate cid is still indexable after pruning.
        let long_cid = format!("0x{}", "cd".repeat(128));
        s.push_closed_ep(ClosedEpisode {
            cid: long_cid.clone(),
            reason: "closed_flat".into(),
            invalid_reason: None,
            trainable: true,
            pnl_um: 0,
            cash_um: 0,
            budget_um: 0,
        });
        assert_eq!(s.closed_eps.len(), CLOSED_EPS_CAP);
        assert!(s.closed_eps.iter().any(|e| e.cid == long_cid));
    }
}

// ── E2E online learning test (replay harness, tests ONLY) ────────
//
// `cargo test --release online_proof -- --ignored --nocapture` with env vars:
//   RL_ONLINE_TAPE      slice dir (ResearchEvent jsonl)
//   RL_ONLINE_LABELS    slice labels_v1.json
//   RL_ONLINE_ARTIFACT  `candidate` policy artifact dir (policy.json)
//   RL_ONLINE_PREDICTOR accepted predictor dir (manifest+model.json)
//   RL_ONLINE_CID       market_cid to filter on (default: first book of the tape)
//   RL_ONLINE_OUT       evidence dir (default: target/rl_online_proof)
//
// Proven chain: captured events → REAL RlPredictor (spawn_learner +
// reconcile_run + real EpisodeLedger over shared SQLite) →
// executor/sim/fill_sink real → EpisodeEnd → worker Rust → checkpoint +
// new-version snapshot → later inference on the SAME instance
// stamped with the new version (no restart and no faked second episode).
//
// The base policy is the `candidate` injected by `load_candidate_for_test`
// (explicit, cfg(test)) — production `load_approved` still
// rejects it; this test relaxes no real guard. The observation
// post-update reuses the last real book captured on a market already
// closed: it is a retrospective repetition (the label already exists), plumbing evidence
// only — never post-label causality nor OOS performance.
#[cfg(test)]
mod online_proof {
    use super::*;
    use crate::model_observer::{self, Predictor, ShadowModel, SpotSeries};
    use crate::replay::{ReplayEngine, ReplayParams, Tape, TapeResolution};
    use crate::research::contract::{EventSource, MarketOutcome, OutcomeSide};
    use crate::research_replay::{ResearchTape, TapeOpts};
    use crate::state::StateStore;
    use anyhow::Context;
    use serde_json::{json, Value};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant as Wall};

    /// `Strategy` shim wrapping a REAL `RlPredictor`: it emulates the
    /// `ModelTool` broadcast (same feature extractor and same
    /// frozen predictor — the prob comes from the accepted artifact, never
    /// invented), keeps the last real book/vol for the probes and
    /// records the policy version stamped on each transition.
    ///
    /// The per-asset spot series is fed EXACTLY as in the driver
    /// `rl_policy_env`: `extract_features` records the causal sample
    /// (`now − age of the VolSnapshot`) into `SpotSeries` before evaluating the
    /// loaded model — never an empty series nor fabricated features.
    struct Shim {
        inner: RlPredictor,
        pred_tx: broadcast::Sender<ModelPrediction>,
        model: Option<Arc<ShadowModel>>,
        series: HashMap<String, SpotSeries>,
        /// (ts_replay_ms, stamped version, action, sampled) per
        /// BookObserve with a recorded transition — the version that served
        /// the decision and whether the step was a real choice (warmup gate),
        /// no later reads.
        decision_versions: Vec<(i64, u64, usize, bool)>,
        n_books: u64,
        n_intents: u64,
        /// Wall-clock latency probes per piece of the real path.
        feat_ns: Vec<u128>,
        mask_ns: Vec<u128>,
        probs_ns: Vec<u128>,
        on_book_ns: Vec<u128>,
    }

    impl Shim {
        /// Per-piece latencies over the real state of a live market —
        /// the SAME functions that `on_book` runs (features, mask,
        /// distribution). Used during the tape (live learner in another
        /// thread consuming transitions) and in the update window.
        fn probe(&mut self, book: &BookSnapshot, label: &'static str) {
            let Some(m) = self.inner.markets.get(&book.market_condition_id) else {
                return;
            };
            let p_yes = pred_usable(
                &self.inner.cfg,
                book,
                m.last_prediction.as_ref(),
                &self.inner.predictor_sha256,
            );
            let age_secs = book
                .recv_instant
                .and_then(|r| crate::time::Instant::now().checked_duration_since(r))
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let t = Wall::now();
            let obs = build_obs(&self.inner.cfg, m, book, p_yes, age_secs);
            self.feat_ns.push(t.elapsed().as_nanos());
            let t = Wall::now();
            let mask = mask_legal(
                &self.inner.cfg,
                m,
                book,
                p_yes.is_some(),
                self.inner.boundary_active(),
            );
            self.mask_ns.push(t.elapsed().as_nanos());
            if let Some(obs) = obs {
                let t = Wall::now();
                if let Some(snap) = self.inner.current_policy() {
                    let _ = snap.policy.action_probs(&obs, &mask);
                }
                self.probs_ns.push(t.elapsed().as_nanos());
            }
            let _ = label; // phase label — kept for the report
        }
    }

    impl Strategy for Shim {
        fn id(&self) -> StrategyId {
            self.inner.id()
        }

        fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) -> Vec<OrderIntent> {
            let now_ms = crate::time::now().timestamp_millis();
            // 1) Real causal prediction (p_yes of the accepted predictor over
            //    the causal features of the book) → broadcast, same as the
            //    production ModelTool. REAL measured ages (never a made-up 0
            //    value); without a predictor → declared abstention.
            let book_age_ms = book
                .recv_instant
                .and_then(|r| crate::time::Instant::now().checked_duration_since(r))
                .map(|d| d.as_millis() as u64)
                .unwrap_or(u64::MAX);
            let spot_age_ms = vol.updated_at.elapsed().as_millis() as u64;
            let (p_model, abstention, model_hash) = match &self.model {
                Some(model) => {
                    let asset =
                        crate::oracle::asset_from_slug(&book.slug).unwrap_or_default();
                    let series = self.series.entry(asset).or_insert_with(SpotSeries::new);
                    let feats = model_observer::extract_features(book, vol, series, now_ms);
                    match model.predict(&feats) {
                        Ok(p) => (Some(p), None, Some(model.model_hash().to_string())),
                        Err(_) => (
                            None,
                            Some("predict_error".to_string()),
                            Some(model.model_hash().to_string()),
                        ),
                    }
                }
                None => (None, Some("model_unavailable".to_string()), None),
            };
            let _ = self.pred_tx.send(ModelPrediction {
                schema_version: MODEL_TOOL_SCHEMA_VERSION,
                ts_ms: now_ms,
                market_cid: book.market_condition_id.clone(),
                slug: book.slug.clone(),
                token_id_yes: book.yes.token_id.clone(),
                token_id_no: book.no.token_id.clone(),
                side: "buy_yes".into(), // the tool does not decide the side
                p_model,
                p_market: book.yes_mid().and_then(|d| f64::try_from(d).ok()),
                book_age_ms,
                spot_age_ms,
                latency_ms: 0,
                budget_ms: 0,
                abstention,
                model_hash,
            });

            // 2) Decision of the real runtime.
            let t = Wall::now();
            let intents = self.inner.on_book(book, vol);
            self.on_book_ns.push(t.elapsed().as_nanos());
            self.n_books += 1;
            self.n_intents += intents.len() as u64;
            // 3) REAL version stamped on the transition recorded by the
            //    decision (policy_version set inside on_book, not a
            //    later read of the snapshot).
            if let Some(m) = self.inner.markets.get(&book.market_condition_id) {
                if let Some(tr) = &m.pending_transition {
                    self.decision_versions
                        .push((now_ms, tr.policy_version, tr.action, tr.sampled));
                }
            }
            // 4) Per-piece probes with the live learner.
            self.probe(book, "tape");
            intents
        }

        fn on_fill(&mut self, fill: &Fill) {
            self.inner.on_fill(fill)
        }

        fn on_timer(&mut self) -> Vec<OrderIntent> {
            self.inner.on_timer()
        }
    }

    /// Replica of the driver's `labels_to_resolutions` (private in the bin
    /// frozen): labels_v1.json → `TapeResolution` with `avail_ms =
    /// real known_at_ms — the label is never moved earlier.
    fn labels_to_resolutions(labels: &Value) -> anyhow::Result<Vec<TapeResolution>> {
        let mut out = Vec::new();
        let items: Vec<&Value> = match labels {
            Value::Array(a) => a.iter().collect(),
            Value::Object(_) => vec![labels],
            _ => anyhow::bail!("labels: JSON must be an object or array"),
        };
        for l in items {
            let cid = l
                .get("market_cid")
                .and_then(Value::as_str)
                .context("labels: missing market_cid")?;
            let winner = match l.get("winner").and_then(Value::as_str) {
                Some("yes") => OutcomeSide::Yes,
                Some("no") => OutcomeSide::No,
                other => anyhow::bail!("labels: unknown winner {other:?}"),
            };
            let known_at_ms = l
                .get("known_at_ms")
                .and_then(Value::as_i64)
                .context("labels: missing known_at_ms (it is never invented)")?;
            let provenance = l
                .get("provenance")
                .and_then(Value::as_str)
                .context("labels: missing provenance")?;
            anyhow::ensure!(!provenance.trim().is_empty(), "labels: provenance is empty");
            out.push(TapeResolution {
                avail_ms: known_at_ms,
                tape_seq: None,
                cid: cid.to_string(),
                outcome: MarketOutcome {
                    winner,
                    known_at_ms,
                    provenance: provenance.to_string(),
                },
                source: EventSource::PolymarketRest,
            });
        }
        Ok(out)
    }

    /// Replica of `filter_tape` (private in research_replay, file
    /// frozen): the cid's books, the slug's asset spots, resolutions
    /// of the cid and ALL quality markers (the scope is decided by the
    /// gate, not the filter).
    fn tape_for_cid(tape: &Tape, cid: &str) -> Tape {
        let asset = tape
            .books
            .iter()
            .find(|b| b.snap.market_condition_id == cid)
            .and_then(|b| crate::oracle::asset_from_slug(&b.snap.slug));
        let mut quality_stats = tape.quality_stats.clone();
        quality_stats.meta_cids.retain(|c| c.as_str() == cid);
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
                .filter(|s| asset.as_ref().is_none_or(|a| s.asset.to_lowercase() == *a))
                .cloned()
                .collect(),
            resolutions: tape
                .resolutions
                .iter()
                .filter(|r| r.cid == cid)
                .cloned()
                .collect(),
            quality: tape.quality.clone(),
            quality_stats,
            source: tape.source.clone(),
        }
    }

    /// Concurrency probe statistics (separate OS thread). The
    /// `n_in`/`n_out` counts are EXACT (`update_in_flight` flag, which the
    /// worker sets only during the real cut); the INTERVALS
    /// `(t0_ns,t1_ns)` go into a bounded uniform reservoir — the
    /// percentiles come from a declared sample, the counters do not.
    struct ProbeStats {
        in_flight: Vec<(i64, i64)>,
        out_flight: Vec<(i64, i64)>,
        n_in: u64,
        n_out: u64,
        versions: std::collections::BTreeSet<u64>,
    }

    /// Uniform reservoir sampling (algorithm R): each sample has
    /// probability `cap/seen` of being kept — no hidden temporal bias.
    fn reservoir_push<T>(v: &mut Vec<T>, x: T, seen: u64, cap: usize, rng: &mut SplitMix64) {
        if v.len() < cap {
            v.push(x);
        } else {
            let j = rng.next_u64() % seen;
            if (j as usize) < cap {
                v[j as usize] = x;
            }
        }
    }

    /// RAII guard of the probe thread: stops and joins the std::thread
    /// also when the test exits by error/panic — never an orphan thread
    /// left behind after a failed assert.
    struct ProbeGuard {
        stop: Arc<std::sync::atomic::AtomicBool>,
        handle: Option<std::thread::JoinHandle<ProbeStats>>,
    }

    impl ProbeGuard {
        /// Launches the probe: `watch.borrow + action_probs` (the same
        /// hot-path inference — the scaler lives inside the policy)
        /// over the real obs/mask captured, on its own OS thread.
        fn spawn(
            rx: watch::Receiver<Arc<PolicySnapshot>>,
            ctl: Arc<learner::LearnerControl>,
            obs: [f64; NUM_FEATURES],
            mask: ActionMask,
        ) -> Self {
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stop_c = stop.clone();
            let handle = std::thread::spawn(move || {
                const CAP: usize = 100_000;
                let mut rng = SplitMix64::new(0xC0FFEE);
                let mut st = ProbeStats {
                    in_flight: Vec::new(),
                    out_flight: Vec::new(),
                    n_in: 0,
                    n_out: 0,
                    versions: Default::default(),
                };
                while !stop_c.load(Ordering::Relaxed) {
                    // t0 BEFORE the borrow: the measurement includes the read
                    // of the current snapshot + scaler + action_probs — the same
                    // cost a real decision pays. `black_box` prevents
                    // the optimizer from eliminating the entry or result.
                    let t0 = learner::monotonic_ns();
                    let snap = rx.borrow().clone();
                    let p = snap.policy.action_probs(
                        std::hint::black_box(&obs),
                        std::hint::black_box(&mask),
                    );
                    std::hint::black_box(&p);
                    let t1 = learner::monotonic_ns();
                    st.versions.insert(snap.version);
                    if ctl.update_in_flight.load(Ordering::Relaxed) {
                        st.n_in += 1;
                        reservoir_push(&mut st.in_flight, (t0, t1), st.n_in, CAP, &mut rng);
                    } else {
                        st.n_out += 1;
                        reservoir_push(&mut st.out_flight, (t0, t1), st.n_out, CAP, &mut rng);
                    }
                }
                st
            });
            Self {
                stop,
                handle: Some(handle),
            }
        }

        /// Stops the probe and returns its statistics.
        fn finish(mut self) -> ProbeStats {
            self.stop.store(true, Ordering::Relaxed);
            self.handle
                .take()
                .and_then(|h| h.join().ok())
                .unwrap_or(ProbeStats {
                    in_flight: Vec::new(),
                    out_flight: Vec::new(),
                    n_in: 0,
                    n_out: 0,
                    versions: Default::default(),
                })
        }
    }

    impl Drop for ProbeGuard {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    /// Recursive copy — the restart/rollback works over a COPY of
    /// the real chain: the original evidence is never mutated.
    fn copy_dir_rec(src: &Path, dst: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(dst)?;
        for e in std::fs::read_dir(src)? {
            let e = e?;
            let to = dst.join(e.file_name());
            if e.file_type()?.is_dir() {
                copy_dir_rec(&e.path(), &to)?;
            } else {
                std::fs::copy(e.path(), &to)?;
            }
        }
        Ok(())
    }

    /// Percentiles in µs — measured in this run, never a promise.
    fn pct(ns: &[u128]) -> Value {
        if ns.is_empty() {
            return Value::Null;
        }
        let mut v = ns.to_vec();
        v.sort_unstable();
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize] as f64 / 1000.0;
        json!({
            "n": v.len(),
            "p50_us": q(0.50), "p95_us": q(0.95), "p99_us": q(0.99),
            "max_us": *v.last().unwrap() as f64 / 1000.0,
        })
    }

    async fn wait_for(mut f: impl FnMut() -> bool, tries: usize, step_ms: u64) -> bool {
        for _ in 0..tries {
            if f() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(step_ms)).await;
        }
        f()
    }

    // `current_thread`: ReplayEngine contract (the same one the driver
    // `rl_policy_env` uses — multi-thread makes job quiescence in the
    // executor very rare). The learner lives on the same thread cooperatively:
    // concurrency measurement is real interleaving, not parallelism.
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "replay harness — requires a real tape + artifacts via RL_ONLINE_*"]
    async fn online_learning_real_episode() -> anyhow::Result<()> {
        let env = |k: &str| -> anyhow::Result<PathBuf> {
            std::env::var_os(k)
                .map(PathBuf::from)
                .with_context(|| format!("missing env {k} — the harness requires real data"))
        };
        let tape_dir = env("RL_ONLINE_TAPE")?;
        let labels_path = env("RL_ONLINE_LABELS")?;
        let art_dir = env("RL_ONLINE_ARTIFACT")?;
        let pred_dir = env("RL_ONLINE_PREDICTOR")?;
        let out_root = std::env::var_os("RL_ONLINE_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/rl_online_proof"));

        // Unique evidence dir per run — the earlier evidence
        // is never overwritten.
        let out_dir = out_root.join(format!(
            "run_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        std::fs::create_dir_all(&out_dir)?;
        let ckpt_dir = out_dir.join("checkpoints");
        let db_path = out_dir.join("proof.sqlite");
        let db_str = db_path.to_string_lossy().into_owned();
        let proof_path = out_dir.join("proof.json");

        let mut proof = json!({
            "kind": "rl_online_proof_increment_1",
            "tape_dir": tape_dir.display().to_string(),
            "labels": labels_path.display().to_string(),
            "artifact_dir": art_dir.display().to_string(),
            "predictor_dir": pred_dir.display().to_string(),
            "checkpoint_dir": ckpt_dir.display().to_string(),
            "db": db_str,
            "notes": [
                "base artifact = candidate injected by the harness (cfg(test)); \
                 production load_approved still requires paper_approved",
                "post-update observation over an already closed market = \
                 retrospective repetition — no causal claims nor OOS",
            ],
        });
        // Early evidence: the file exists even if an assert fails.
        let flush = |proof: &Value| {
            let _ = std::fs::write(
                &proof_path,
                serde_json::to_string_pretty(proof).unwrap() + "\n",
            );
        };

        // ── Real tape filtered to the cid + official label at real known_at ──
        let rt = ResearchTape::load(&tape_dir, &TapeOpts::default())
            .with_context(|| format!("loading tape {}", tape_dir.display()))?;
        let cid = std::env::var("RL_ONLINE_CID")
            .ok()
            .or_else(|| rt.tape.books.first().map(|b| b.snap.market_condition_id.clone()))
            .context("no cid: neither RL_ONLINE_CID nor books in the tape")?;
        let mut tape = tape_for_cid(&rt.tape, &cid);
        let labels: Value = serde_json::from_str(
            &std::fs::read_to_string(&labels_path)
                .with_context(|| format!("reading {}", labels_path.display()))?,
        )?;
        let resolutions_list: Vec<TapeResolution> = labels_to_resolutions(&labels)?
            .into_iter()
            .filter(|r| r.cid == cid)
            .collect();
        let n_labels = resolutions_list.len();
        tape.resolutions.extend(resolutions_list);
        proof["tape"] = json!({
            "cid": cid,
            "events": rt.events,
            "diagnostics": rt.diagnostics.len(),
            "books_cid": tape.books.len(),
            "spots": tape.spots.len(),
            "resolutions_cid": tape.resolutions.len(),
            "injected_labels": n_labels,
            "quality_marks": tape.quality.len(),
        });
        flush(&proof);

        // ── Shared real SQLite: engine writes, reconcile reads ──
        let state = StateStore::open(&db_str, Mode::Paper).await?;

        // ── Real frozen predictor (hash = artifact gate) ──
        let model = Arc::new(
            ShadowModel::load(&pred_dir)
                .map_err(|e| anyhow::anyhow!("invalid predictor: {e}"))?,
        );

        // ── `candidate` artifact explicitly injected by the harness ──
        let art = artifact::load_candidate_for_test(&art_dir)
            .map_err(|e| anyhow::anyhow!("invalid candidate: {e}"))?;
        anyhow::ensure!(
            art.predictor_sha256 == model.model_hash(),
            "candidate predictor_sha256 != sha256(model.json)"
        );
        let base_sha = art.policy_sha256.clone();
        let base_weights = art.policy.weights.clone();
        let base_bias = art.policy.bias.clone();

        let mut cfg = RlPredictorConfig::default();
        cfg.learning_enabled = true;
        cfg.checkpoint_dir = ckpt_dir.to_string_lossy().into_owned();
        cfg.min_episodes_per_update = 1; // vertical test: cohort of 1
        cfg.settle_poll_secs = 1;
        // The clear→known_at gap of the capture is ~67e6 ms of clock
        // virtual time: the settle timeout must NEVER fire before the
        // official label (400_000 s ≫ the real gap).
        cfg.settle_timeout_secs = 400_000;
        cfg.boundary_max_wait_ms = 300_000_000;
        cfg.rng_seed = 7;
        cfg.learner_capacity = 8192;

        let params = ReplayParams {
            seed: 0xB007,
            ..Default::default()
        };
        let initial_balance = params.initial_balance;

        let (pred_tx, pred_rx) = broadcast::channel(1024);
        let mut shim = Shim {
            inner: RlPredictor::new_test_wired(cfg.clone(), Some(pred_rx), art, &db_str),
            pred_tx,
            model: Some(model),
            series: HashMap::new(),
            decision_versions: Vec::new(),
            n_books: 0,
            n_intents: 0,
            feat_ns: Vec::new(),
            mask_ns: Vec::new(),
            probs_ns: Vec::new(),
            on_book_ns: Vec::new(),
        };
        let control = shim
            .inner
            .learner
            .as_ref()
            .context("learner did not start in the harness")?
            .control
            .clone();

        // Recovery Ready BEFORE starting the tape — same as the production
        // runner: without proven ownership there are no decisions.
        let ready = wait_for(
            || {
                matches!(
                    shim.inner
                        .reconcile
                        .as_ref()
                        .map(|r| r.recovery_rx.borrow().clone()),
                    Some(RecoveryState::Ready { .. })
                )
            },
            30,
            400,
        )
        .await;
        anyhow::ensure!(ready, "recovery never reached Ready within 12 s");

        // ══ SINGLE PASS: real episode → EpisodeEnd → Rust update ══════
        // freeze-learning BEFORE the episode: EpisodeEnd reaches the
        // worker but must not publish a version while frozen.
        control.set_frozen(true);
        let mut engine = ReplayEngine::new(params.clone(), state.clone()).await?;
        engine.seed_tape(tape)?;
        let mut steps = 0u64;
        while engine.step_once(&mut shim).await?.is_some() {
            steps += 1;
        }
        // The resolution was already applied in SQLite at its real known_at. The task
        // of reconcile polls the shared db; manual `on_timer` drains the
        // SettleEvent until it closes the episode — the replay clock keeps
        // armed at tape_end (without simulating extra dead hours).
        for _ in 0..30 {
            if shim.inner.closed_eps.iter().any(|e| e.cid == cid) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1_100)).await;
            let _ = shim.on_timer();
        }

        let Some(closed) = shim.inner.closed_eps.iter().find(|e| e.cid == cid) else {
            anyhow::bail!("the cid's episode never closed");
        };
        let ep_pnl_um = closed.cash_um - closed.budget_um;
        let ep_pnl_realized_um = closed.pnl_um;
        let ep_reason = closed.reason.clone();
        let ep_trainable = closed.trainable;

        anyhow::ensure!(
            wait_for(
                || control.episodes_completed.load(Ordering::Relaxed) >= 1,
                40,
                250
            )
            .await,
            "the worker did not account for EpisodeEnd (episodes_completed=0)"
        );
        // Real freeze: experience incorporated, version intact.
        let published_frozen = wait_for(
            || control.version.load(Ordering::Relaxed) > 0,
            5,
            400,
        )
        .await;
        proof["freeze"] = json!({
            "episodes_completed_under_freeze":
                control.episodes_completed.load(Ordering::Relaxed),
            "version_while_frozen":
                control.version.load(Ordering::Relaxed),
            "updates_applied_while_frozen":
                control.updates_applied.load(Ordering::Relaxed),
            "published_under_freeze": published_frozen,
        });

        // ── Lifecycle accounting: EXACTLY one episode ──
        // With the idempotent terminal market, no episodes can exist
        // artificial flat post-clear episodes in the cohort).
        let n_closed = shim.inner.closed_eps.len();
        let eps_completed = control.episodes_completed.load(Ordering::Relaxed);
        let eps_excluded = control.episodes_excluded.load(Ordering::Relaxed);
        anyhow::ensure!(
            n_closed == 1 && eps_completed == 1,
            "spurious episodes: closed_eps={n_closed} completed={eps_completed} excluded={eps_excluded}"
        );

        // ── PRE-update probe: real obs saved + probs tied to v0 ──
        // The capture was made INSIDE the real decision (same Arc) — the
        // obs has ≥2 legal actions by construction.
        let cap = shim
            .inner
            .probe_base()
            .context("no free-choice decision captured during the tape")?;
        let cap_n_legal = cap.mask.iter().filter(|b| **b).count();
        anyhow::ensure!(
            cap_n_legal >= 2,
            "the probe obs has a 1-action mask — it does not prove a distribution change"
        );
        let (v0_probe, sha0_probe, probs0_probe, _) = shim
            .inner
            .probe_infer()
            .context("probe_infer without a pre-update snapshot")?;
        anyhow::ensure!(
            v0_probe == 0 && sha0_probe == base_sha,
            "pre-update probe did not serve the base: v={v0_probe} sha={sha0_probe}"
        );

        // ══ UNFREEZE → real worker update ═══════════════════════════
        // on_book is not called post-clear (terminal market — idempotent).
        // The cut is made by the learner tick with an empty `cohort_open`.
        //
        // REAL CONCURRENCY: the inference runs on a dedicated OS `std::thread`
        // running `watch.borrow + action_probs` — the SAME
        // production hot-path functions — while the worker performs
        // the update in `spawn_blocking` (blocking thread pool). They are
        // distinct OS threads: the measured overlap is true parallelism,
        // not a cooperative interleave of the current_thread runtime. The
        // classification during/outside the update uses the atomic
        // `update_in_flight` flag of the worker — exact, with no interval intersection
        // nor doubtful clock resolution.
        let probe_rx = shim
            .inner
            .learner
            .as_ref()
            .unwrap()
            .policy_rx
            .clone();
        let guard = ProbeGuard::spawn(probe_rx, control.clone(), cap.obs, cap.mask);

        control.set_frozen(false);
        let mut probe_ns: Vec<u128> = Vec::new();
        let t_wait = Wall::now();
        let mut probe_post: Option<(u64, String, [f64; NUM_ACTIONS], usize)> = None;
        while t_wait.elapsed() < Duration::from_secs(15) {
            let t = Wall::now();
            let pr = shim.inner.probe_infer();
            probe_ns.push(t.elapsed().as_nanos());
            if let Some(p) = &pr {
                if p.0 >= 1 {
                    probe_post = pr;
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        // `finish` stops and joins the probe; if the bail fires earlier, the
        // Drop of the guard does it anyway (no orphan threads).
        let stats = guard.finish();
        let Some((v_tip_probe, sha_tip_probe, probs_tip_probe, n_legal_tip)) = probe_post
        else {
            anyhow::bail!(
                "the Rust worker never published v1 after unfreeze (rejected={}, completed={}, excluded={})",
                control.updates_rejected.load(Ordering::Relaxed),
                control.episodes_completed.load(Ordering::Relaxed),
                control.episodes_excluded.load(Ordering::Relaxed)
            );
        };

        // Monotonic span (ns) of the real CUT — measured inside the
        // closure around `close_cohort` (no pool wait nor
        // ticks without a cut). Post-hoc classification of the retained
        // intervals: how many INTERSECT the span and how many stay
        // fully CONTAINED inside — cross-check of the flag.
        let u_start = control.update_span_start_ns.load(Ordering::Relaxed);
        let u_end = control.update_span_end_ns.load(Ordering::Relaxed);
        let dispatch_ns = control.update_dispatch_ns.load(Ordering::Relaxed);
        let in_dur_ns: Vec<u128> = stats
            .in_flight
            .iter()
            .map(|(a, b)| (b - a) as u128)
            .collect();
        let out_dur_ns: Vec<u128> = stats
            .out_flight
            .iter()
            .map(|(a, b)| (b - a) as u128)
            .collect();
        let n_intersect = stats
            .in_flight
            .iter()
            .filter(|(t0, t1)| *t0 <= u_end && *t1 >= u_start)
            .count();
        let n_contained = stats
            .in_flight
            .iter()
            .filter(|(t0, t1)| *t0 >= u_start && *t1 <= u_end)
            .count();
        // Real overlap: EXACT counter per flag + post-hoc verification
        // over retained intervals (both must be >0 if there was computation
        // measurable concurrency).
        let overlap = stats.n_in as usize;

        let snap_tip = shim
            .inner
            .learner
            .as_ref()
            .unwrap()
            .snapshot()
            .clone();
        let (meta_tip, pol_tip, sha_tip) =
            artifact::load_checkpoint_chain(&ckpt_dir, &base_sha)
                .map_err(|e| anyhow::anyhow!("chain broken after update: {e}"))?;

        // ── Real trajectory emitted by the runtime (cfg(test) tap): ──
        // reusable JSONL dump — only messages ACCEPTED by try_send.
        {
            let msgs = shim.inner.sent_log.lock().unwrap();
            let mut jl = String::new();
            for m in msgs.iter() {
                jl.push_str(&serde_json::to_string(m).unwrap());
                jl.push('\n');
            }
            std::fs::write(out_dir.join("transitions.jsonl"), &jl)?;
            proof["trajectory"] = json!({
                "file": out_dir.join("transitions.jsonl").display().to_string(),
                "n_msgs": msgs.len(),
            });
            flush(&proof);
        }

        // ══ RESTART + ROLLBACK over the REAL chain just written ══════
        // Copy of the checkpoints dir (the original evidence is not mutated):
        // a NEW worker instance resumes from the tip of the
        // chain and an operator rollback publishes the next version
        // with the base weights — real files on disk, no mocks.
        let restart_dir = out_dir.join("restart_checkpoints");
        copy_dir_rec(&ckpt_dir, &restart_dir)?;
        let art_restart = artifact::load_candidate_for_test(&art_dir)
            .map_err(|e| anyhow::anyhow!("invalid candidate for restart: {e}"))?;
        let h_restart = learner::spawn_learner(
            &art_restart,
            restart_dir.clone(),
            cfg.reinforce_config(),
            cfg.learner_capacity,
        )
        .context("restart: spawn_learner returned None (no runtime?)")?;
        let snap_rs = h_restart.snapshot().clone();
        let restart_restores = snap_rs.version == meta_tip.version
            && snap_rs.update_index == meta_tip.update_index
            && snap_rs.policy_sha256 == sha_tip
            && snap_rs.parent_sha256 == meta_tip.parent_sha256
            && snap_rs.policy.weights == pol_tip.weights
            && snap_rs.policy.bias == pol_tip.bias
            && snap_rs.policy.scaler == pol_tip.scaler;

        h_restart.control.request_rollback();
        let rollback_published = wait_for(
            || {
                h_restart.control.version.load(Ordering::Relaxed)
                    >= meta_tip.version + 1
            },
            40,
            250,
        )
        .await;
        let snap_rb = h_restart.snapshot().clone();
        let rb_meta_path = restart_dir
            .join("checkpoints")
            .join(format!("v{:06}", snap_rb.version))
            .join("meta.json");
        let rb_meta: Option<artifact::OnlineMetaDoc> = std::fs::read(&rb_meta_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        let rb_episodes = rb_meta
            .as_ref()
            .and_then(|m| m.counters.get("completed_episodes"))
            .and_then(Value::as_u64);
        proof["restart_rollback"] = json!({
            "restart_dir": restart_dir.display().to_string(),
            "resume_version": snap_rs.version,
            "resume_update_index": snap_rs.update_index,
            "resume_sha256": snap_rs.policy_sha256,
            "resume_parent_sha256": snap_rs.parent_sha256,
            "restores_weights_scaler_bias_version": restart_restores,
            "rollback_published": rollback_published,
            "rollback_version": snap_rb.version,
            "rollback_sha256": snap_rb.policy_sha256,
            "rollback_weights_are_base": snap_rb.policy.weights == base_weights
                && snap_rb.policy.bias == base_bias
                && snap_rb.policy.scaler == art_restart.policy.scaler,
            "meta_kind": rb_meta.as_ref().map(|m| m.kind.clone()),
            "meta_parent_sha256": rb_meta.as_ref().map(|m| m.parent_sha256.clone()),
            "meta_parent_version": rb_meta.as_ref().map(|m| m.parent_version),
            "meta_base_sha256": rb_meta.as_ref().map(|m| m.base_sha256.clone()),
            "meta_completed_episodes": rb_episodes,
        });
        flush(&proof);

        let outcome = engine.finalize(&mut shim).await?;
        engine.shutdown().await?;

        // ── Accounting identity: Σ(cash−budget) over episodes == wallet_delta ──
        let ep_pnl_total_um: i64 = shim
            .inner
            .closed_eps
            .iter()
            .map(|e| e.cash_um - e.budget_um)
            .sum();
        let wallet_delta_um = outcome
            .wallet_final
            .map(|w| to_micros(w - initial_balance));

        // ── Distribution before/after over the SAME real obs ──
        // probs_base = those produced by the real decision (v0, same Arc);
        // probs_tip = probe_infer post-update (v1, same Arc read inside
        // the call). The stored mask has ≥2 legal actions.
        let delta_probs = cap
            .probs
            .iter()
            .zip(probs_tip_probe.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let probs_evidence = json!({
            "obs_version_decision": cap.version,
            "obs_policy_sha256_decision": cap.policy_sha256,
            "n_legal_actions": cap_n_legal,
            "probe": "retrospective — REAL obs saved during the episode; \
                      inference uses the shared production function; \
                      probes NEVER feed the learner",
            "probs_base_v0": cap.probs.to_vec(),
            "probs_probe_pre": probs0_probe.to_vec(),
            "probs_tip": probs_tip_probe.to_vec(),
            "bound_version_tip": v_tip_probe,
            "bound_sha_tip": sha_tip_probe,
            "max_abs_delta": delta_probs,
        });
        let w_diff: f64 = pol_tip
            .weights
            .iter()
            .zip(base_weights.iter())
            .flat_map(|(r1, r0)| r1.iter().zip(r0.iter()).map(|(a, b)| (a - b).abs()))
            .fold(0.0, f64::max);
        let b_diff: f64 = pol_tip
            .bias
            .iter()
            .zip(base_bias.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        proof["run"] = json!({
            "steps": steps,
            "delivered_books": shim.n_books,
            "emitted_intents": shim.n_intents,
            "seen_fills": outcome.fills_seen.len(),
            "settlements": outcome.settlements,
            "reconciled": outcome.reconciled(),
            "digest": outcome.digest,
            "fidelity": outcome.fidelity,
            "fidelity_reasons": outcome.fidelity_reasons,
            "settlement_pending": outcome.settlement_pending,
        });
        proof["episode"] = json!({
            "cid": cid,
            "reason": ep_reason,
            "trainable": ep_trainable,
            "pnl_um": ep_pnl_um,
            "pnl_realized_um": ep_pnl_realized_um,
            "closed_eps_total": shim.inner.closed_eps.len(),
        });
        proof["learner"] = control.status();
        proof["update"] = json!({
            "version_tip_snapshot": snap_tip.version,
            "policy_sha256_tip": snap_tip.policy_sha256,
            "base_sha256": base_sha,
            "checkpoint_tip": {
                "version": meta_tip.version,
                "kind": meta_tip.kind,
                "policy_sha256": meta_tip.policy_sha256,
                "parent_sha256": meta_tip.parent_sha256,
                "sha256_recomputed_chain": sha_tip,
                "dir": ckpt_dir.join("checkpoints")
                    .join(format!("v{:06}", meta_tip.version))
                    .display().to_string(),
            },
            "max_abs_delta_weights": w_diff,
            "max_abs_delta_bias": b_diff,
            "probs_same_obs": probs_evidence,
        });
        proof["accounting"] = json!({
            "episodes_pnl_um": ep_pnl_total_um,
            "wallet_delta_um": wallet_delta_um,
            "drift_um": wallet_delta_um.map(|w| ep_pnl_total_um - w),
        });
        proof["latency_us"] = json!({
            "during_tape": {
                "features": pct(&shim.feat_ns),
                "mask": pct(&shim.mask_ns),
                "action_probs": pct(&shim.probs_ns),
                "on_book_total": pct(&shim.on_book_ns),
            },
            "probes_post_unfreeze": pct(&probe_ns),
            "update_span_ns": {
                "start_ns": u_start,
                "end_ns": u_end,
                "duration_ns": if u_end > u_start { u_end - u_start } else { 0 },
                "dispatch_total_ns": dispatch_ns,
                "clock": "Monotonic process Instant, ns — span measured \
                                                    INSIDE the closure around close_cohort \
                          (real computation); dispatch_total includes waiting for \
                                                    pool + join, reported separately",
            },
            "concurrency": {
                "probe_thread": "dedicated OS std::thread — real inference \
                               (watch.borrow + action_probs, black_box on \
                               entry and exit) over the captured obs/mask; \
                                                              NEVER enters the learner",
                                "update_thread": "spawn_blocking (blocking pool) — CPU of the \
                                gradient outside the async worker; the I/O of the \
                                checkpoint goes in its own spawn_blocking; \
                                                                publish after the join",
                "classifier": "update_in_flight flag set ONLY during \
                                 close_cohort (never waiting on a pool nor \
                                 ticks without a cut) — exact counters; \
                                 retained intervals verified post-hoc \
                                                                  against the published span",
                "reservoir_cap_per_class": 100_000,
                "sample_coverage": "percentiles over a uniform reservoir \
                                 (algorithm R); exact n_* counters",
                "inference_during_update": pct(&in_dur_ns),
                "inference_outside_update": pct(&out_dur_ns),
                "n_during_update_exact": stats.n_in,
                "n_outside_update_exact": stats.n_out,
                "retained_intervals_in": stats.in_flight.len(),
                "intervals_in_intersecting_span": n_intersect,
                "intervals_in_contained_in_span": n_contained,
                "versions_seen": stats.versions,
                                "unit": "intervals measured in ns, percentiles in µs",
            },
            "note": "current_thread runtime (ReplayEngine contract), but \
                     the probe runs on a separate OS thread and the update on the \
                     blocking pool: the reported overlap is real parallelism \
                     measured with a shared monotonic clock. Measured in this \
                                          run, not a promise. Predictor/executor out of scope.",
        });
        let n_sampled = shim
            .decision_versions
            .iter()
            .filter(|(_, _, _, s)| *s)
            .count();
        proof["decisiones"] = json!({
            "n_stamped": shim.decision_versions.len(),
            "n_sampled_free_choice": n_sampled,
            "n_gate_excluded": shim.decision_versions.len() - n_sampled,
            "observed_versions": shim
                .decision_versions
                .iter()
                .map(|(_, v, _, _)| *v)
                .collect::<std::collections::BTreeSet<_>>(),
            "probe_new_version": {
                "bound_version": v_tip_probe,
                "bound_sha": sha_tip_probe,
                "n_legal": n_legal_tip,
                "different_probs": delta_probs > 0.0,
            },
        });
        flush(&proof);

        // ══ Gate ASSERTS ═══════════════════════════════════════════
        anyhow::ensure!(!published_frozen, "freeze did not hold the version");
        anyhow::ensure!(meta_tip.version >= 1, "no checkpoint published");
        anyhow::ensure!(
            sha_tip == snap_tip.policy_sha256,
            "the published snapshot does not match the chain tip on disk"
        );
        anyhow::ensure!(
            meta_tip.parent_sha256 == base_sha || meta_tip.parent_version >= 1,
            "the chain does not descend from the base"
        );
        anyhow::ensure!(sha_tip != base_sha, "the policy did not change any bytes");
        anyhow::ensure!(
            w_diff > 0.0 || b_diff > 0.0,
            "the checkpoint weights are identical to the base — there was no real update"
        );
        anyhow::ensure!(
            v_tip_probe >= 1 && sha_tip_probe == snap_tip.policy_sha256,
            "probe_infer did not bind the new version (v={v_tip_probe} sha={sha_tip_probe})"
        );
        anyhow::ensure!(
            delta_probs > 0.0,
            "the probs over the real obs with ≥2 legal actions did not change after the update"
        );
        anyhow::ensure!(
            wallet_delta_um == Some(ep_pnl_total_um),
            "accounting identity broken: Σ(cash−budget)={ep_pnl_total_um} != wallet_delta={wallet_delta_um:?}"
        );
        anyhow::ensure!(
            outcome.reconciled(),
            "the engine's return↔ledger reconciliation did not close"
        );
        // Restart over the real chain: version, sha, weights, scaler and bias
        // of the tip restored intact in a new worker instance.
        anyhow::ensure!(
            restart_restores,
            "restart did not restore the tip v{} intact (sha {}, weights/scaler/bias/version)",
            meta_tip.version, &sha_tip[..12]
        );
        // Operator rollback: new version with the base weights and
        // parent→v_tip chain intact in the persisted meta.
        anyhow::ensure!(rollback_published, "the rollback never published a version");
        let rb_meta = rb_meta.context("rollback checkpoint meta.json unreadable")?;
        anyhow::ensure!(
            rb_meta.kind == "rollback"
                && rb_meta.parent_sha256 == sha_tip
                && rb_meta.parent_version == meta_tip.version
                && rb_meta.base_sha256 == base_sha
                && rb_meta.policy_sha256 == snap_rb.policy_sha256,
            "rollback checkpoint chain is inconsistent"
        );
        anyhow::ensure!(
            snap_rb.policy.weights == base_weights && snap_rb.policy.bias == base_bias,
            "the rollback did not restore the base weights/bias"
        );
        anyhow::ensure!(
            rb_episodes.unwrap_or(0) >= 1,
            "counters not restored after restart (completed_episodes<1)"
        );
        // Real concurrency measured: exact counter (flag) AND verification
        // post-hoc check of retained intervals against the published span.
        anyhow::ensure!(
            overlap > 0 && n_intersect > 0,
            "no inference overlapping with the real cut — flag={} \
             intersect={} (span {}..{} ns, probes total {})",
            stats.n_in,
            n_intersect,
            u_start,
            u_end,
            stats.n_in + stats.n_out
        );
        // Clean shutdown of the restart learner (do not leave a stray task).
        drop(h_restart);

        eprintln!(
            "online_proof OK — v0→v{}, sha {}→{}, drift_um={:?}, evidence {}",
            meta_tip.version,
            &base_sha[..12],
            &sha_tip[..12],
            wallet_delta_um.map(|w| ep_pnl_total_um - w),
            proof_path.display()
        );
        Ok(())
    }
}
