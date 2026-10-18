//! Dedicated online-learning worker.
//!
//! ## Concurrency
//!
//! - **bounded** `mpsc` channel strategy→learner. Back-pressure: `try_send`; what
//!   does not fit is dropped and counted (`transitions_dropped`/`episodes_dropped`).
//!   The trading hot path NEVER blocks on the learner, and the accounting
//!   ledger does not depend on this channel.
//! - Policy publication via `watch::Sender<Arc<PolicySnapshot>>`: immutable,
//!   versioned snapshots; the strategy reads the latest one with `borrow`
//!   without holding locks. A partial policy is never published.
//! - Checkpoints are written off the hot path (by this task).
//! - Learner down → the strategy keeps the last snapshot `Arc` and keeps
//!   managing positions (control `alive=false` + warn).
//!
//! ## Algorithm
//!
//! Episodic REINFORCE with reward-to-go, `gamma=1` (v1), baseline = running mean
//! of the returns of episodes from PREVIOUS batches (independent of the current
//! action), gradient clipping and configurable learning rate.
//! One update per cohort: COMPLETE episodes, fully executed under the frozen
//! version `v`. Episodes that span a version change are excluded (experience
//! from different weights is never mixed, as in on-policy). HOLD and SELL learn
//! the same way as BUY — every decision emitted by the policy is a cohort
//! transition.
//!
//! During cohort closing (`boundary`) the learner raises a shared gate that the
//! strategy checks to avoid OPENING new positions; SELLs of existing inventory
//! are never blocked.
//!
//! ## Publication gates
//!
//! Finiteness, `‖Δw‖₂ ≤ max_update_l2`, mean/per-step KL over the cohort's
//! recorded observations (`≤ max_mean_kl`/`≤ max_step_kl`). If any gate fails →
//! the update is aborted, the previous policy stays intact, the `updates_rejected`
//! counter is incremented and the reason is recorded. These gates do NOT prove
//! profitability — they are numerical safety checks.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use tokio::sync::{mpsc, watch};

use crate::rl::artifact::{
    self, ApprovedArtifact, OnlineMetaDoc, PolicyArtifactError, PolicyTrainingDoc,
};
use crate::rl::contract::{ActionMask, NUM_ACTIONS, NUM_FEATURES};
use crate::rl::policy::{LinearSoftmaxPolicy, PolicySnapshot};

/// Monotonic process clock (ns) for measurement spans — NOT the replay clock
/// nor wall-clock: it only serves to compare overlaps between the learner's
/// update and the inferences inside the same process.
/// Nanoseconds: real inference is sub-µs, so µs would have no resolution.
pub(crate) fn monotonic_ns() -> i64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as i64
}

/// RAII guard: resets the flag to `false` when leaving the scope — even while
/// unwinding from a panic the flag is never left stuck at `true`.
struct FlagOff<'a>(&'a AtomicBool);

impl Drop for FlagOff<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

// ── Messages strategy → learner ────────────────────────────────────────────

/// Complete transition with its reward already assigned (the strategy computes
/// it at the next accounting point — an action is only incorporated once its
/// consequences are observable).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransitionRec {
    /// Episode = market (unique cid).
    pub episode_id: String,
    /// Policy version that produced the decision (behavior version).
    pub policy_version: u64,
    /// RAW observation (not standardized — the scaler travels with the policy).
    pub obs: [f64; NUM_FEATURES],
    pub mask: ActionMask,
    pub action: usize,
    /// π_behavior(action) recorded at decision time.
    pub behavior_prob: f64,
    /// Full probability vector over legal actions (0 for illegal ones).
    pub probs: [f64; NUM_ACTIONS],
    /// `true` = sampled action (learns); `false` = argmax evaluation
    /// (does NOT feed the gradient — REINFORCE is not applied on-policy over argmax).
    pub sampled: bool,
    /// Normalized reward assigned to this transition.
    pub reward_norm: f64,
    /// Raw equity delta in micros (audit).
    pub net_pnl_delta_um: i64,
    pub ts_ms: i64,
}

/// End of episode: the trajectory is closed (or excluded).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpisodeEndMsg {
    pub episode_id: String,
    /// `false` + `reason` → accounted for but excluded from cohorts
    /// (truncated, stale, ambiguous ownership, recovered, mixed-version…).
    pub trainable: bool,
    pub reason: Option<String>,
    /// Ledger invalidation reason (`pending_timeout`, `mixed_version`…)
    /// when it differs from the closing reason (`closed_flat`, `settled`…).
    #[serde(default)]
    pub invalid_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LearnerMsg {
    EpisodeStart {
        episode_id: String,
        policy_version: u64,
        ts_ms: i64,
    },
    Transition(Box<TransitionRec>),
    EpisodeEnd(EpisodeEndMsg),
}

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ReinforceConfig {
    pub learning_rate: f64,
    /// L2 clip of the mean gradient (per step), BEFORE multiplying by
    /// `learning_rate` — exact parity with `parity.py clip_gradients`.
    /// `<= 0` disables the clip (valid; the `max_update_l2` gate still applies).
    pub max_grad_norm: f64,
    /// Hard gate: if the post-update `‖Δw‖₂` exceeds this, the update is rejected.
    pub max_update_l2: f64,
    /// KL gates: mean and maximum per transition over the cohort's observations.
    pub max_mean_kl: f64,
    pub max_step_kl: f64,
    /// Minimum number of complete episodes to close a cohort.
    pub min_episodes_per_update: usize,
    /// Cap on episodes accumulated in a cohort (memory while frozen).
    pub cohort_cap: usize,
    /// Maximum wait with the boundary gate raised before invalidating the open
    /// episodes (remaining) and cutting the cohort.
    pub boundary_max_wait_ms: i64,
    /// Internal learner tick interval (boundary timeout, rollback).
    pub tick_ms: u64,
}

impl Default for ReinforceConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            max_grad_norm: 1.0,
            max_update_l2: 0.5,
            max_mean_kl: 0.05,
            max_step_kl: 0.2,
            min_episodes_per_update: 4,
            cohort_cap: 64,
            boundary_max_wait_ms: 360_000,
            tick_ms: 500,
        }
    }
}

// ── Shared control (API / operator) ──────────────────────────────────────

/// Observable state and controls of the learner. Shared between the learner
/// task, the strategy and the API. Registered in [`learner_control`] when the
/// worker starts so that `/ui/rl_learner` and `/control/rl_learner` can find it.
pub struct LearnerControl {
    /// Manual freeze-learning: inference keeps running, updates do not.
    frozen: AtomicBool,
    /// Rollback request to the approved base (served on the next tick).
    rollback_requested: AtomicBool,
    /// Cohort boundary gate — the strategy reads it to avoid OPENING new
    /// positions while the cohort is being cut (SELL is never blocked).
    pub boundary: AtomicBool,
    alive: AtomicBool,

    pub version: AtomicU64,
    pub update_index: AtomicU64,
    pub transitions_rx: AtomicU64,
    pub episodes_completed: AtomicU64,
    pub episodes_excluded: AtomicU64,
    pub transitions_dropped: AtomicU64,
    pub episodes_dropped: AtomicU64,
    /// Transitions whose episode has no known `EpisodeStart` (start lost to a
    /// full channel, or an episode predating a restart).
    pub transitions_orphan: AtomicU64,
    /// Complete episodes discarded by the cohort cap during a long wait
    /// (the silent `pop_front` is no longer invisible).
    pub episodes_evicted: AtomicU64,
    /// In-flight episodes evicted by the defensive cap on `pending`
    /// (safety net for an `EpisodeEnd` lost to a full channel — the orphan
    /// entry must not grow without bound).
    pub pending_evicted: AtomicU64,
    pub updates_applied: AtomicU64,
    pub updates_rejected: AtomicU64,
    pub checkpoints_written: AtomicU64,
    /// `true` ONLY while the real cut (`close_cohort`) runs inside
    /// `spawn_blocking` — never while waiting for the pool nor on ticks without
    /// a cut. Exact classifier of overlap with the hot path.
    pub update_in_flight: AtomicBool,
    /// Nanoseconds of the process monotonic clock (`monotonic_ns`) for the
    /// start/end of the LAST cohort cut actually executed (`close_cohort` —
    /// publish or rejection). 0 = no cut yet. Same scale as the inference
    /// probes — the comparison requires a common clock and unit.
    pub update_span_start_ns: AtomicI64,
    pub update_span_end_ns: AtomicI64,
    /// Total duration (ns) of the last cut's `spawn_blocking` round-trip
    /// (dispatch + computation + join) — kept apart from the pure computation
    /// span so scheduling is not confused with gradient work.
    pub update_dispatch_ns: AtomicI64,
    policy_sha: Mutex<String>,
    last_event: Mutex<String>,
}

impl Default for LearnerControl {
    fn default() -> Self {
        Self::new()
    }
}

impl LearnerControl {
    pub fn new() -> Self {
        Self {
            frozen: AtomicBool::new(false),
            rollback_requested: AtomicBool::new(false),
            boundary: AtomicBool::new(false),
            alive: AtomicBool::new(false),
            version: AtomicU64::new(0),
            update_index: AtomicU64::new(0),
            transitions_rx: AtomicU64::new(0),
            episodes_completed: AtomicU64::new(0),
            episodes_excluded: AtomicU64::new(0),
            transitions_dropped: AtomicU64::new(0),
            episodes_dropped: AtomicU64::new(0),
            transitions_orphan: AtomicU64::new(0),
            episodes_evicted: AtomicU64::new(0),
            pending_evicted: AtomicU64::new(0),
            updates_applied: AtomicU64::new(0),
            updates_rejected: AtomicU64::new(0),
            checkpoints_written: AtomicU64::new(0),
            update_in_flight: AtomicBool::new(false),
            update_span_start_ns: AtomicI64::new(0),
            update_span_end_ns: AtomicI64::new(0),
            update_dispatch_ns: AtomicI64::new(0),
            policy_sha: Mutex::new(String::new()),
            last_event: Mutex::new(String::new()),
        }
    }

    pub fn frozen(&self) -> bool {
        self.frozen.load(Ordering::Relaxed)
    }
    pub fn set_frozen(&self, v: bool) {
        self.frozen.store(v, Ordering::Relaxed);
    }
    pub fn request_rollback(&self) {
        self.rollback_requested.store(true, Ordering::Relaxed);
    }
    fn take_rollback(&self) -> bool {
        self.rollback_requested.swap(false, Ordering::Relaxed)
    }
    pub fn boundary_active(&self) -> bool {
        self.boundary.load(Ordering::Relaxed)
    }
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    fn note(&self, msg: impl Into<String>) {
        *self.last_event.lock().unwrap() = msg.into();
    }

    /// Serializable snapshot for `/ui/rl_learner`.
    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({
            "alive": self.is_alive(),
            "frozen": self.frozen(),
            "boundary": self.boundary_active(),
            "policy_version": self.version.load(Ordering::Relaxed),
            "update_index": self.update_index.load(Ordering::Relaxed),
            "policy_sha256": self.policy_sha.lock().unwrap().clone(),
            "transitions_rx": self.transitions_rx.load(Ordering::Relaxed),
            "episodes_completed": self.episodes_completed.load(Ordering::Relaxed),
            "episodes_excluded": self.episodes_excluded.load(Ordering::Relaxed),
            "transitions_dropped": self.transitions_dropped.load(Ordering::Relaxed),
            "episodes_dropped": self.episodes_dropped.load(Ordering::Relaxed),
            "transitions_orphan": self.transitions_orphan.load(Ordering::Relaxed),
            "episodes_evicted": self.episodes_evicted.load(Ordering::Relaxed),
            "pending_evicted": self.pending_evicted.load(Ordering::Relaxed),
            "updates_applied": self.updates_applied.load(Ordering::Relaxed),
            "updates_rejected": self.updates_rejected.load(Ordering::Relaxed),
            "checkpoints_written": self.checkpoints_written.load(Ordering::Relaxed),
            "update_in_flight": self.update_in_flight.load(Ordering::Relaxed),
            "update_span_start_ns": self.update_span_start_ns.load(Ordering::Relaxed),
            "update_span_end_ns": self.update_span_end_ns.load(Ordering::Relaxed),
            "update_dispatch_ns": self.update_dispatch_ns.load(Ordering::Relaxed),
            "last_event": self.last_event.lock().unwrap().clone(),
        })
    }
}

/// Registry of the learner for the API — the `rl_predictor` strategy is unique,
/// so a singleton is enough; respawning the strategy (hot toggle) replaces the handle.
static RL_LEARNER: LazyLock<Mutex<Option<Arc<LearnerControl>>>> =
    LazyLock::new(|| Mutex::new(None));

/// Public handle for the API (`None` if the strategy never started).
pub fn learner_control() -> Option<Arc<LearnerControl>> {
    RL_LEARNER.lock().unwrap().clone()
}

fn record_control(c: Arc<LearnerControl>) {
    *RL_LEARNER.lock().unwrap() = Some(c);
}

// ── REINFORCE update (pure core, testable) ───────────────────────────────

/// Result of an applied and validated update.
pub struct UpdateResult {
    pub new_policy: LinearSoftmaxPolicy,
    pub episodes: usize,
    pub transitions: usize,
    pub mean_return: f64,
    pub update_l2: f64,
    pub mean_kl: f64,
    pub max_kl: f64,
    /// Total gradient before clipping (diagnostic).
    pub grad_l2_pre_clip: f64,
}

/// Reason an update was rejected (every case → previous policy left intact).
#[derive(Debug)]
pub enum UpdateReject {
    NotFinite,
    UpdateL2 { l2: f64, max: f64 },
    MeanKl { kl: f64, max: f64 },
    StepKl { kl: f64, max: f64 },
    NoTransitions,
    /// The chosen action has π=0 under the cohort's policy — corrupt data or
    /// uncorrected off-policy data (parity.py aborts the same way).
    OffPolicy,
    /// The whole cohort produced zero advantage — there is no signal to learn;
    /// publishing an identical version would only pollute the chain.
    DeltaZero,
}

impl std::fmt::Display for UpdateReject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFinite => write!(f, "resulting weights are not finite"),
            Self::UpdateL2 { l2, max } => write!(f, "‖Δw‖₂={l2:.4} > {max}"),
            Self::MeanKl { kl, max } => write!(f, "mean KL={kl:.4} > {max}"),
            Self::StepKl { kl, max } => write!(f, "max KL per step={kl:.4} > {max}"),
            Self::NoTransitions => write!(f, "cohort has no sampled transitions"),
            Self::OffPolicy => write!(
                f,
                "chosen action with π=0 under the cohort's policy (corrupt or off-policy data)"
            ),
            Self::DeltaZero => write!(f, "total gradient is zero (null advantage across the whole cohort)"),
        }
    }
}

/// Running mean of episode returns — the REINFORCE baseline
/// ("previous batches", independent of the current action).
#[derive(Debug, Default, Clone)]
pub struct ReturnBaseline {
    pub sum: f64,
    pub n: u64,
}

impl ReturnBaseline {
    pub fn value(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }
    pub fn observe(&mut self, ret: f64) {
        if ret.is_finite() {
            self.sum += ret;
            self.n += 1;
        }
    }
}

/// Episodic REINFORCE update over ONE cohort of complete trajectories.
///
/// Canonical formula (exact parity with the offline `parity.py`):
///
/// 1. Reward-to-go `G_t = Σ_{k≥t} r_k` (γ=1) over normalized rewards.
/// 2. `b` = mean of the returns of PREVIOUS batches (the caller folds this
///    cohort's returns in AFTER the update — they never contaminate it).
/// 3. `A_t = G_t − b`, the same for every action of the step.
/// 4. Per step and action `j`: `g_logit[j] = A_t·(δ(j==a_t) − π_j)` with `π`
///    RECOMPUTED under the cohort's frozen policy (not the recorded probs —
///    re-evaluation is the off-policy guard: if `π(a_t)<=0` the data is
///    corrupt and the update aborts). Illegal actions (`π_j<=0`) do not
///    participate.
/// 5. `dW[a][f] += g_logit[a]·x_std[f]`, `db[a] += g_logit[a]`.
/// 6. Mean over the TOTAL steps of the cohort (`n_steps`).
/// 7. Global L2 norm clip to `max_grad_norm` (never per element),
///    BEFORE the learning rate.
/// 8. `param += lr · clipped_grad` (gradient ascent).
///
/// Numerical aborts (cohort intact, current policy untouched):
/// non-finite reward/x_std, π(a_t)≤0, non-finite or zero gradient norm.
///
/// - `episodes`: complete trajectories (only `sampled` ones contribute gradient;
///   their rewards still enter the `G_t` of earlier steps of the episode).
///
/// Read-only; returns the NEW policy without mutating the entry one.
pub fn reinforce_update(
    policy: &LinearSoftmaxPolicy,
    episodes: &[Vec<TransitionRec>],
    baseline: &ReturnBaseline,
    cfg: &ReinforceConfig,
) -> Result<UpdateResult, UpdateReject> {
    let b = baseline.value();
    let mut gw = vec![vec![0.0f64; NUM_FEATURES]; NUM_ACTIONS];
    let mut gb = [0.0f64; NUM_ACTIONS];
    let mut n_steps = 0usize;
    let mut returns = Vec::with_capacity(episodes.len());

    for ep in episodes {
        // Reward-to-go (γ=1): G_t = Σ_{k≥t} r_k over already normalized rewards.
        let mut g_suffix = vec![0.0f64; ep.len()];
        let mut acc = 0.0;
        for t in (0..ep.len()).rev() {
            if !ep[t].reward_norm.is_finite() {
                return Err(UpdateReject::NotFinite);
            }
            acc += ep[t].reward_norm;
            g_suffix[t] = acc;
        }
        if let Some(&r) = g_suffix.first() {
            returns.push(r);
        }
        for (t, tr) in ep.iter().enumerate() {
            if !tr.sampled {
                continue; // argmax/evaluation never contributes gradient
            }
            // π recomputed under the cohort's frozen policy — parity with
            // `policy.probs(x_std, mask)` and a guard against off-policy/corrupt
            // data (recorded probs are not trusted).
            let (Some(x), Some(probs)) = (
                policy.standardize(&tr.obs),
                policy.action_probs(&tr.obs, &tr.mask),
            ) else {
                // Invalid obs/mask in a transition that passed the entry
                // filter — corruption, fail-closed abort.
                return Err(UpdateReject::NotFinite);
            };
            if probs[tr.action] <= 0.0 {
                return Err(UpdateReject::OffPolicy);
            }
            let adv = g_suffix[t] - b;
            n_steps += 1;
            if adv == 0.0 {
                continue;
            }
            for a in 0..NUM_ACTIONS {
                if probs[a] <= 0.0 {
                    continue; // illegal action does not participate
                }
                // ∇log π(a|s) w.r.t. w[a] = x·(δ_{a==a_t} − π(a)).
                let gl = adv * ((a == tr.action) as u8 as f64 - probs[a]);
                gb[a] += gl;
                for j in 0..NUM_FEATURES {
                    gw[a][j] += gl * x[j];
                }
            }
        }
    }
    if n_steps == 0 {
        return Err(UpdateReject::NoTransitions);
    }
    // Mean per STEP over the whole cohort (parity: `dW *= 1/n_steps`).
    let inv = 1.0 / n_steps as f64;
    let mut grad_l2 = 0.0;
    for a in 0..NUM_ACTIONS {
        gb[a] *= inv;
        for j in 0..NUM_FEATURES {
            gw[a][j] *= inv;
            grad_l2 += gw[a][j] * gw[a][j];
        }
        grad_l2 += gb[a] * gb[a];
    }
    let grad_l2_pre = grad_l2.sqrt();
    if !grad_l2_pre.is_finite() {
        return Err(UpdateReject::NotFinite);
    }
    if grad_l2_pre == 0.0 {
        return Err(UpdateReject::DeltaZero);
    }
    // Global L2 clip of the mean gradient, BEFORE the learning rate.
    if cfg.max_grad_norm > 0.0 && grad_l2_pre > cfg.max_grad_norm {
        let f = cfg.max_grad_norm / grad_l2_pre;
        for a in 0..NUM_ACTIONS {
            gb[a] *= f;
            for w in gw[a].iter_mut() {
                *w *= f;
            }
        }
    }

    // New policy: `param += lr · grad` (ascent). A copy — the entry policy
    // remains the snapshot v until the checkpoint is published.
    let mut new_policy = policy.clone();
    for a in 0..NUM_ACTIONS {
        new_policy.bias[a] += cfg.learning_rate * gb[a];
        for j in 0..NUM_FEATURES {
            new_policy.weights[a][j] += cfg.learning_rate * gw[a][j];
        }
    }
    if !new_policy.is_valid() {
        return Err(UpdateReject::NotFinite);
    }

    // Magnitude gate ‖Δw‖₂.
    let mut d2 = 0.0;
    for a in 0..NUM_ACTIONS {
        let db = new_policy.bias[a] - policy.bias[a];
        d2 += db * db;
        for j in 0..NUM_FEATURES {
            let dw = new_policy.weights[a][j] - policy.weights[a][j];
            d2 += dw * dw;
        }
    }
    let update_l2 = d2.sqrt();
    if update_l2 > cfg.max_update_l2 {
        return Err(UpdateReject::UpdateL2 {
            l2: update_l2,
            max: cfg.max_update_l2,
        });
    }

    // KL gate over the cohort's recorded observations.
    let mut kl_sum = 0.0;
    let mut kl_max = 0.0f64;
    let mut kl_n = 0usize;
    for ep in episodes {
        for tr in ep.iter() {
            if !tr.sampled {
                continue;
            }
            let Some(p_new) = new_policy.action_probs(&tr.obs, &tr.mask) else {
                continue;
            };
            let mut kl = 0.0;
            for a in 0..NUM_ACTIONS {
                let po = tr.probs[a];
                if po <= 0.0 {
                    continue;
                }
                let pn = p_new[a].max(1e-12);
                kl += po * (po / pn).ln();
            }
            kl_sum += kl.max(0.0);
            kl_max = kl_max.max(kl);
            kl_n += 1;
        }
    }
    let mean_kl = if kl_n > 0 { kl_sum / kl_n as f64 } else { 0.0 };
    if mean_kl > cfg.max_mean_kl {
        return Err(UpdateReject::MeanKl {
            kl: mean_kl,
            max: cfg.max_mean_kl,
        });
    }
    if kl_max > cfg.max_step_kl {
        return Err(UpdateReject::StepKl {
            kl: kl_max,
            max: cfg.max_step_kl,
        });
    }

    Ok(UpdateResult {
        new_policy,
        episodes: episodes.len(),
        transitions: n_steps,
        mean_return: if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f64>() / returns.len() as f64
        },
        update_l2,
        mean_kl,
        max_kl: kl_max,
        grad_l2_pre_clip: grad_l2_pre,
    })
}

// ── LearnerCore: cohort state machine (pure, no tokio) ───────────────────────

/// Defensive cap on `pending`: an `EpisodeEnd` dropped by a full channel would
/// leave its entry orphaned forever. Pruning on the boundary timeout covers the
/// common case; this cap is the safety net against future leaks — once exceeded,
/// `evict_pending_if_exceeds` evicts the stalest episode. ~288 episodes/day on
/// 5-minute brackets: 4k entries are only reached after ~14 days of orphans.
const PENDING_CAP: usize = 4_096;

struct EpAccum {
    /// Version the episode started under.
    version: u64,
    /// `EpisodeStart` timestamp (ms, caller clock) — age order used for
    /// the eviction by the defensive cap on `pending`.
    ts_ms: i64,
    transitions: Vec<TransitionRec>,
    /// Policy version changed in the middle of the episode → not trainable.
    mixed: bool,
}

/// Learner state. `pub(crate)` for tests — the cohort and update logic is
/// synchronous and deterministic; `run_learner` only wraps it in
/// channels/timer/IO.
pub struct LearnerCore {
    /// Private copy of the current policy (version `version`).
    pub policy: LinearSoftmaxPolicy,
    pub version: u64,
    pub update_index: u64,
    /// sha256 of the `policy.json` of the currently published snapshot
    /// (v0 = hash of the base artifact; v>0 = hash of the written checkpoint).
    pub policy_sha256: String,
    /// sha256 of the PARENT snapshot (for the next `meta.parent_sha256`).
    pub parent_sha256: String,
    /// sha256 of the approved base artifact (root of the chain).
    pub base_sha256: String,
    pub predictor_sha256: String,
    pub cfg: ReinforceConfig,
    /// Baseline = mean of the returns of batches ALREADY closed. The returns of
    /// the current cohort are folded in AFTER its update — they never feed the
    /// baseline they are compared against.
    pub baseline: ReturnBaseline,
    /// Accumulated trained episodes (sum over published cohorts — honest
    /// provenance for `training.completed_episodes`).
    pub episodes_trained: u64,
    /// `training` of the approved base artifact — real provenance of the base
    /// weights (reused by the rollback checkpoint).
    pub base_training: PolicyTrainingDoc,

    /// Open episodes of the current cohort.
    cohort_open: HashSet<String>,
    /// In-flight transitions per episode.
    pending: HashMap<String, EpAccum>,
    /// Complete, trainable trajectories of the current cohort, with their
    /// episodic return (to fold the baseline in AFTER the update).
    /// `VecDeque` capped at `cohort_cap`: during a long freeze completed episodes
    /// would otherwise accumulate without limit — the oldest are discarded (the
    /// caller counts them in `episodes_evicted`) so the cohort that is finally
    /// trained contains the most recent data.
    cohort_done: std::collections::VecDeque<(Vec<TransitionRec>, f64)>,
    /// Instant (ms, caller clock) at which the boundary was raised.
    pub boundary_since_ms: Option<i64>,
}

/// What `tick`/`on_msg` can return — the async task executes it (IO,
/// publish) and the core stays testable without tokio.
pub enum CoreEvent {
    /// Raise the boundary gate (the strategy stops opening positions).
    BoundaryOn,
    /// Update ready: publish `snapshot` and write `meta` + policy_bytes.
    Publish {
        snapshot: PolicySnapshot,
        meta: OnlineMetaDoc,
        policy_bytes: Vec<u8>,
        stats: UpdateResult,
    },
    /// Update rejected by a gate — previous policy left intact.
    UpdateRejected { reason: String },
}

impl LearnerCore {
    pub fn new(
        policy: LinearSoftmaxPolicy,
        policy_sha256: String,
        base_sha256: String,
        predictor_sha256: String,
        cfg: ReinforceConfig,
        base_training: PolicyTrainingDoc,
    ) -> Self {
        Self {
            policy,
            version: 0,
            update_index: 0,
            parent_sha256: policy_sha256.clone(),
            policy_sha256,
            base_sha256,
            predictor_sha256,
            cfg,
            baseline: ReturnBaseline::default(),
            episodes_trained: 0,
            base_training,
            cohort_open: HashSet::new(),
            pending: HashMap::new(),
            cohort_done: std::collections::VecDeque::new(),
            boundary_since_ms: None,
        }
    }

    /// Restores from a loaded checkpoint (restart): version/update/
    /// baseline of the last valid link.
    pub fn resume_from(&mut self, meta: &OnlineMetaDoc, policy: LinearSoftmaxPolicy, sha: String) {
        self.version = meta.version;
        self.update_index = meta.update_index;
        self.policy_sha256 = sha.clone();
        self.parent_sha256 = meta.parent_sha256.clone();
        self.policy = policy;
        if let Some(b) = meta
            .counters
            .get("baseline_sum")
            .and_then(|v| v.as_f64())
        {
            self.baseline.sum = b;
        }
        if let Some(n) = meta.counters.get("baseline_n").and_then(|v| v.as_u64()) {
            self.baseline.n = n;
        }
        // Accumulated count of trained episodes (provenance).
        if let Some(e) = meta
            .counters
            .get("completed_episodes")
            .and_then(|v| v.as_u64())
        {
            self.episodes_trained = e;
        }
    }

    /// Records the opening of an episode. Returns the evicted `episode_id`
    /// if the defensive cap on `pending` evicted the stalest one — the caller
    /// counts it in `pending_evicted`.
    pub fn on_episode_start(
        &mut self,
        episode_id: String,
        policy_version: u64,
        ts_ms: i64,
    ) -> Option<String> {
        // Only episodes opened under the CURRENT version form the cohort (a
        // stale start may arrive if the boundary relaxes); episodes of another
        // version (`mixed`) keep receiving data but are not trainable in this cohort.
        let mixed = policy_version != self.version;
        if !mixed {
            self.cohort_open.insert(episode_id.clone());
        }
        self.pending.insert(
            episode_id,
            EpAccum {
                version: policy_version,
                ts_ms,
                transitions: Vec::new(),
                mixed,
            },
        );
        self.evict_pending_if_exceeds()
    }

    /// Evicts the stalest in-flight episode when `pending` exceeds
    /// `PENDING_CAP`. The stalest by its start `ts_ms` is the one that has been
    /// waiting longest for an `EpisodeEnd` that was probably lost — if it is still
    /// legitimate, its late messages are counted as orphan/dropped, never silenced.
    fn evict_pending_if_exceeds(&mut self) -> Option<String> {
        if self.pending.len() <= PENDING_CAP {
            return None;
        }
        let oldest = self
            .pending
            .iter()
            .min_by_key(|(_, e)| e.ts_ms)
            .map(|(k, _)| k.clone())?;
        self.cohort_open.remove(&oldest);
        self.pending.remove(&oldest);
        Some(oldest)
    }

    /// Adds a transition. `false` = unknown episode (its start was lost to a full
    /// channel, or it predates a restart — the caller counts it in
    /// `transitions_orphan`); the transition is discarded because data from an
    /// episode whose opening was never observed cannot be mixed in.
    pub fn on_transition(&mut self, tr: TransitionRec) -> bool {
        match self.pending.get_mut(&tr.episode_id) {
            Some(ep) => {
                if tr.policy_version != ep.version {
                    ep.mixed = true; // crossed a version boundary
                }
                ep.transitions.push(tr);
                true
            }
            None => false,
        }
    }

    /// `trainable=false` ⇒ the episode is accounted for and EXCLUDED from the
    /// cohort (the reason goes in `reason` for the caller's log).
    /// Returns `(accepted, evicted_by_cap, transitions)`.
    /// The episodic return is stored with the trajectory: the baseline is folded
    /// in ONLY after a successfully applied update (parity with `trainer.py`,
    /// which folds `ep_returns` after `apply_gradients`).
    pub fn on_episode_end(
        &mut self,
        episode_id: &str,
        trainable: bool,
    ) -> Option<(bool, usize, Vec<TransitionRec>)> {
        self.cohort_open.remove(episode_id);
        let ep = self.pending.remove(episode_id)?;
        let ok = trainable && !ep.mixed && ep.version == self.version;
        let mut evicted = 0;
        if ok {
            let ret: f64 = ep.transitions.iter().map(|t| t.reward_norm).sum();
            if self.cohort_done.len() >= self.cfg.cohort_cap {
                self.cohort_done.pop_front(); // cap: drops the oldest one
                evicted += 1;
            }
            self.cohort_done.push_back((ep.transitions.clone(), ret));
        }
        Some((ok, evicted, ep.transitions))
    }

    /// Does the cohort have enough complete episodes to cut?
    /// The boundary is raised once (new entries are blocked) and the cut runs
    /// when the open episodes drain or `boundary_max_wait_ms` expires.
    pub fn maybe_boundary(&mut self, now_ms: i64) -> Option<CoreEvent> {
        if self.boundary_since_ms.is_none()
            && self.cohort_done.len() >= self.cfg.min_episodes_per_update
        {
            self.boundary_since_ms = Some(now_ms);
            return Some(CoreEvent::BoundaryOn);
        }
        None
    }

    /// Runs the cut if appropriate (call on every tick while the boundary is raised).
    /// Returns `Publish`/`UpdateRejected`, or `None` if it must still wait.
    pub fn maybe_update(&mut self, now_ms: i64) -> Option<CoreEvent> {
        self.maybe_update_timed(now_ms, None).0
    }

    /// Instrumented `maybe_update`: besides the event it returns the monotonic
    /// span (ns) of the REAL cut (`close_cohort`: drain + baseline +
    /// `reinforce_update` + snapshot construction). `None` if the tick did not
    /// reach a cut — no boundary, or the cohort still open and waiting — because
    /// those ticks are not learning work. `flag` is set `true` ONLY during the
    /// real cut: never on early returns nor during the wait/dispatch of
    /// `spawn_blocking` — an exact classifier of "update in flight" for the
    /// inference probes.
    pub fn maybe_update_timed(
        &mut self,
        now_ms: i64,
        flag: Option<&AtomicBool>,
    ) -> (Option<CoreEvent>, Option<(i64, i64)>) {
        let Some(since) = self.boundary_since_ms else {
            return (None, None);
        };
        if !self.cohort_open.is_empty()
            && now_ms - since < self.cfg.boundary_max_wait_ms
        {
            return (None, None); // waiting for the open episodes to close
        }
        // Episodes still open are invalidated for THIS cohort. They are REMOVED
        // from `pending`, not merely marked `mixed` — marked entries kept their
        // `Vec<TransitionRec>` forever if their `EpisodeEnd` was lost to a full
        // channel (TTL 0). Now their late messages (transitions and EpisodeEnd)
        // are counted as orphan/dropped by the caller — visible, not accumulated.
        // Entries of `pending` outside `cohort_open` (episodes of other versions)
        // are left alone: they keep waiting for their real `EpisodeEnd`, with the
        // defensive `PENDING_CAP` as the safety net.
        let wait_expired = now_ms - since >= self.cfg.boundary_max_wait_ms;
        if wait_expired && !self.cohort_open.is_empty() {
            for ep in self.cohort_open.drain() {
                self.pending.remove(&ep);
            }
        }
        let s0 = monotonic_ns();
        // RAII guard: the flag returns to `false` even if the cut panics — a
        // stuck flag would report "update in flight" forever on a dead learner.
        let _off = flag.map(|f| {
            f.store(true, Ordering::Relaxed);
            FlagOff(f)
        });
        let ev = self.close_cohort(now_ms);
        drop(_off);
        (ev, Some((s0, monotonic_ns())))
    }

    /// `update_config` in the online manifest format
    /// (`lr`/`grad_clip`/`seed`/`cohort_episodes` required +
    /// its own gates as auditable extras).
    fn update_config_doc(&self) -> serde_json::Value {
        serde_json::json!({
            "lr": self.cfg.learning_rate,
            "grad_clip": self.cfg.max_grad_norm,
            "seed": 0, // the update is deterministic (no RNG)
            "cohort_episodes": self.cfg.min_episodes_per_update,
            "max_update_l2": self.cfg.max_update_l2,
            "max_mean_kl": self.cfg.max_mean_kl,
            "max_step_kl": self.cfg.max_step_kl,
        })
    }

    /// Cuts the current cohort, runs the update and produces the new snapshot.
    /// `kind="online_candidate"`: an online update NEVER publishes
    /// `paper_approved` nor gets relabeled as offline-approved.
    fn close_cohort(&mut self, now_ms: i64) -> Option<CoreEvent> {
        // `unzip` moves the trajectories (without cloning) and splits off the returns.
        let (episodes, rets): (Vec<Vec<TransitionRec>>, Vec<f64>) =
            self.cohort_done.drain(..).unzip();
        self.boundary_since_ms = None;
        match reinforce_update(&self.policy, &episodes, &self.baseline, &self.cfg) {
            Ok(res) => {
                self.update_index += 1;
                self.episodes_trained += res.episodes as u64;
                // The returns of THIS cohort feed the baseline of the next
                // batch — never its own (parity with trainer.py: folded after
                // apply_gradients; an aborted update folds nothing).
                for ret in &rets {
                    self.baseline.observe(*ret);
                }
                let new_version = self.version + 1;
                let training = PolicyTrainingDoc {
                    run_id: format!("online-{}", new_version),
                    // Online cohort without an offline manifest: honest sentinel
                    // = sha256 of empty content (valid hex64; it is not a
                    // fabricated data hash nor a non-hex literal).
                    data_manifest_sha256: artifact::EMPTY_MANIFEST_SHA256.into(),
                    split_manifest_sha256: artifact::EMPTY_MANIFEST_SHA256.into(),
                    seed: 0,
                    completed_episodes: self.episodes_trained,
                    updates: self.update_index,
                };
                // status="online_candidate": online promotion is anchored to the
                // parent+hash chain of the online_manifest — never to a forged
                // `paper_approved`.
                let policy_bytes = match artifact::serialize_policy(
                    &res.new_policy,
                    &self.predictor_sha256,
                    &training,
                    "online_candidate",
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        return Some(CoreEvent::UpdateRejected {
                            reason: format!("serializing checkpoint: {e}"),
                        })
                    }
                };
                let sha = crate::model_observer::sha256_hex(&policy_bytes);
                let snapshot = PolicySnapshot {
                    version: new_version,
                    update_index: self.update_index,
                    policy: Arc::new(res.new_policy.clone()),
                    policy_sha256: sha.clone(),
                    parent_sha256: self.policy_sha256.clone(),
                    base_sha256: self.base_sha256.clone(),
                    published_ts_ms: now_ms,
                };
                let meta = OnlineMetaDoc {
                    schema_version: artifact::POLICY_SCHEMA_VERSION,
                    kind: "online_candidate".into(),
                    version: new_version,
                    update_index: self.update_index,
                    policy_sha256: sha,
                    parent_sha256: self.policy_sha256.clone(),
                    parent_version: self.version,
                    base_sha256: self.base_sha256.clone(),
                    base_artifact_dir: String::new(), // filled in by the task
                    created_ts_ms: now_ms,
                    update_config: self.update_config_doc(),
                    counters: serde_json::json!({
                        "episodes": res.episodes,
                        "transitions": res.transitions,
                        "mean_return": res.mean_return,
                        "update_l2": res.update_l2,
                        "mean_kl": res.mean_kl,
                        "max_kl": res.max_kl,
                        "completed_episodes": self.episodes_trained,
                        "updates": self.update_index,
                        "baseline_sum": self.baseline.sum,
                        "baseline_n": self.baseline.n,
                    }),
                };
                Some(CoreEvent::Publish {
                    snapshot,
                    meta,
                    policy_bytes,
                    stats: res,
                })
            }
            Err(e) => Some(CoreEvent::UpdateRejected {
                reason: e.to_string(),
            }),
        }
    }

    /// Manual rollback: the policy returns to the approved base as a NEW version
    /// (the chain moves forward; `kind="rollback"` documents it).
    /// `base_policy` is the approved artifact's policy, which the caller keeps in
    /// memory (the base artifact's bytes are not re-serialized).
    /// Does not mutate state — `confirm_publish` does it after writing the checkpoint.
    pub fn rollback(&mut self, base_policy: &LinearSoftmaxPolicy, now_ms: i64) -> Option<CoreEvent> {
        let new_version = self.version + 1;
        // Honest provenance: the rollback weights ARE the base artifact's — its
        // `training` documents exactly the run that produced them (the real
        // `completed_episodes`/`updates`/hashes of the approved artifact, never
        // zeros nor non-hex "online" literals).
        let mut training = self.base_training.clone();
        training.run_id = format!("{}+rollback-v{}", training.run_id, new_version);
        let policy_bytes = artifact::serialize_policy(
            base_policy,
            &self.predictor_sha256,
            &training,
            "online_candidate",
        )
        .ok()?;
        let sha = crate::model_observer::sha256_hex(&policy_bytes);
        let snapshot = PolicySnapshot {
            version: new_version,
            update_index: self.update_index,
            policy: Arc::new(base_policy.clone()),
            policy_sha256: sha.clone(),
            parent_sha256: self.policy_sha256.clone(),
            base_sha256: self.base_sha256.clone(),
            published_ts_ms: now_ms,
        };
        let meta = OnlineMetaDoc {
            schema_version: artifact::POLICY_SCHEMA_VERSION,
            kind: "rollback".into(),
            version: new_version,
            update_index: self.update_index,
            policy_sha256: sha,
            parent_sha256: self.policy_sha256.clone(),
            parent_version: self.version,
            base_sha256: self.base_sha256.clone(),
            base_artifact_dir: String::new(),
            created_ts_ms: now_ms,
            // Same format — a rollback link is also a promoted
            // online_candidate (the manifest sets promoted_as).
            update_config: self.update_config_doc(),
            counters: serde_json::json!({
                "completed_episodes": self.episodes_trained,
                "updates": self.update_index,
                "baseline_sum": self.baseline.sum,
                "baseline_n": self.baseline.n,
            }),
        };
        self.cohort_done.clear();
        self.cohort_open.clear();
        self.boundary_since_ms = None;
        Some(CoreEvent::Publish {
            snapshot,
            meta,
            policy_bytes,
            stats: UpdateResult {
                new_policy: base_policy.clone(),
                episodes: 0,
                transitions: 0,
                mean_return: 0.0,
                update_l2: 0.0,
                mean_kl: 0.0,
                max_kl: 0.0,
                grad_l2_pre_clip: 0.0,
            },
        })
    }

    /// Applies a publication emitted by `close_cohort`/`rollback` — the task calls
    /// this after WRITING the checkpoint and sending the snapshot.
    pub fn confirm_publish(&mut self, snapshot: &PolicySnapshot) {
        self.policy = snapshot.policy.as_ref().clone();
        self.version = snapshot.version;
        self.policy_sha256 = snapshot.policy_sha256.clone();
        self.parent_sha256 = snapshot.parent_sha256.clone();
    }
}

// ── Strategy-side handle + spawn ───────────────────────────────────────────

/// What the strategy needs from the learner: a bounded experience channel,
/// a watch channel of snapshots and shared control.
pub struct LearnerHandle {
    tx: mpsc::Sender<LearnerMsg>,
    pub policy_rx: watch::Receiver<Arc<PolicySnapshot>>,
    pub control: Arc<LearnerControl>,
}

impl LearnerHandle {
    /// `false` if the channel is full or the learner went down — the strategy
    /// counts the drop and keeps operating (it never blocks).
    pub fn send(&self, msg: LearnerMsg) -> bool {
        match self.tx.try_send(msg) {
            Ok(()) => true,
            Err(_) => {
                self.control.transitions_dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Current snapshot (Arc clone — never holds locks).
    pub fn snapshot(&self) -> Arc<PolicySnapshot> {
        self.policy_rx.borrow().clone()
    }

    /// Current version/policy, used to stamp transitions.
    pub fn current_version(&self) -> u64 {
        self.policy_rx.borrow().version
    }

    /// `true` if the learner died (channel closed) — the strategy keeps using
    /// the last snapshot and leaves the alarm visible.
    pub fn learner_dead(&self) -> bool {
        self.tx.is_closed()
    }
}

/// Handle with an inspectable queue for strategy tests (verifies that a
/// recovered episode sends explicit `EpisodeStart`/`EpisodeEnd` messages).
#[cfg(test)]
pub(crate) fn handle_for_test(
    policy: LinearSoftmaxPolicy,
) -> (LearnerHandle, mpsc::Receiver<LearnerMsg>) {
    let (tx, rx) = mpsc::channel(64);
    let (_ptx, prx) = watch::channel(Arc::new(PolicySnapshot {
        version: 0,
        update_index: 0,
        policy: Arc::new(policy),
        policy_sha256: "test".into(),
        parent_sha256: String::new(),
        base_sha256: "test".into(),
        published_ts_ms: 0,
    }));
    (
        LearnerHandle {
            tx,
            policy_rx: prx,
            control: Arc::new(LearnerControl::new()),
        },
        rx,
    )
}

/// Spawns the learning worker. `base` is the APPROVED artifact, already loaded
/// and verified by the strategy — the learner never fabricates a policy. On
/// restart it recovers the checkpoint chain if it is intact up to the base; if
/// it is broken, it starts from the base and logs it.
///
/// Returns `None` if no tokio runtime is available (test context without an
/// executor) — the strategy keeps the base v0 policy without a learner.
pub fn spawn_learner(
    base: &ApprovedArtifact,
    checkpoint_dir: PathBuf,
    cfg: ReinforceConfig,
    capacity: usize,
) -> Option<LearnerHandle> {
    let (msg_tx, msg_rx) = mpsc::channel(capacity.max(1));
    let control = Arc::new(LearnerControl::new());

    // Recovery: intact checkpoint chain → last link; broken → the base
    // (explicit rejection, never invented weights).
    let mut core = LearnerCore::new(
        base.policy.clone(),
        base.policy_sha256.clone(),
        base.policy_sha256.clone(),
        base.predictor_sha256.clone(),
        cfg.clone(),
        base.training.clone(),
    );
    match artifact::load_checkpoint_chain(&checkpoint_dir, &base.policy_sha256) {
        Ok((meta, policy, sha)) => {
            tracing::info!(
                version = meta.version,
                update_index = meta.update_index,
                "rl_learner: checkpoint chain recovered — resuming from v{}",
                meta.version
            );
            core.resume_from(&meta, policy, sha);
        }
        Err(e) => {
            if checkpoint_dir.join("active.json").exists()
                || checkpoint_dir.join("checkpoints").exists()
            {
                tracing::warn!(
                    "rl_learner: checkpoint chain unreadable/broken ({e}) — \
                     resuming from the approved base artifact"
                );
            }
        }
    }

    let snapshot0 = Arc::new(PolicySnapshot {
        version: core.version,
        update_index: core.update_index,
        policy: Arc::new(core.policy.clone()),
        policy_sha256: core.policy_sha256.clone(),
        parent_sha256: core.parent_sha256.clone(),
        base_sha256: core.base_sha256.clone(),
        published_ts_ms: crate::time::now().timestamp_millis(),
    });
    let (policy_tx, policy_rx) = watch::channel(snapshot0);
    control
        .version
        .store(core.version, Ordering::Relaxed);
    control
        .update_index
        .store(core.update_index, Ordering::Relaxed);
    *control.policy_sha.lock().unwrap() = core.policy_sha256.clone();
    record_control(control.clone());

    let base_policy = base.policy.clone();
    let base_dir = base.artifact_dir.clone();
    let ctl = control.clone();
    let handle = tokio::runtime::Handle::try_current().ok()?;
    handle.spawn(run_learner(
        core, msg_rx, policy_tx, checkpoint_dir, ctl, base_policy, base_dir,
    ));
    Some(LearnerHandle {
        tx: msg_tx,
        policy_rx,
        control,
    })
}

/// Learner task: consumes experience, manages cohorts and publishes
/// snapshots. Any update failure leaves the previous policy intact.
async fn run_learner(
    mut core: LearnerCore,
    mut rx: mpsc::Receiver<LearnerMsg>,
    policy_tx: watch::Sender<Arc<PolicySnapshot>>,
    checkpoint_dir: PathBuf,
    control: Arc<LearnerControl>,
    base_policy: LinearSoftmaxPolicy,
    base_dir: PathBuf,
) {
    control.alive.store(true, Ordering::Relaxed);
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(
        core.cfg.tick_ms.max(50),
    ));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            msg = rx.recv() => match msg {
                Some(m) => process_msg(&mut core, &control, m),
                None => break, // channels closed — strategy stopped
            },
            _ = tick.tick() => {}
        }
        let now_ms = crate::time::now().timestamp_millis();
        // A manual rollback takes priority (operator).
        if control.take_rollback() {
            if let Some(ev) = core.rollback(&base_policy, now_ms) {
                execute_event(&mut core, &policy_tx, &checkpoint_dir, &control, &base_dir, ev)
                    .await;
            }
        }
        // Under freeze the boundary is neither raised nor cut — before, the
        // arming was erased and re-armed on every tick (gate oscillation).
        if !control.frozen() {
            if let Some(ev) = core.maybe_boundary(now_ms) {
                execute_event(&mut core, &policy_tx, &checkpoint_dir, &control, &base_dir, ev)
                    .await;
            }
            // The cut (cohort + gradient) is blocking CPU work: it goes to the
            // spawn_blocking pool so it does not occupy the runtime thread (an
            // async task does NOT guarantee decoupling — inline computation would
            // block the worker running it). The invariant holds: `execute_event`
            // writes the checkpoint BEFORE publishing the snapshot, after the join.
            // The span and the flag are measured INSIDE the closure around the real
            // cut — pool waiting, join rescheduling and ticks without a cut do not
            // count as "update in flight". The dispatch round-trip is measured
            // separately (`update_dispatch_ns`).
            let ctl = control.clone();
            let d0 = monotonic_ns();
            let joined = tokio::task::spawn_blocking(move || {
                let out =
                    core.maybe_update_timed(now_ms, Some(&ctl.update_in_flight));
                (core, out)
            })
            .await;
            let d1 = monotonic_ns();
            match joined {
                Ok((c, (ev, span))) => {
                    core = c;
                    if let Some((s0, s1)) = span {
                        control.update_span_start_ns.store(s0, Ordering::Relaxed);
                        control.update_span_end_ns.store(s1, Ordering::Relaxed);
                        control.update_dispatch_ns.store(d1 - d0, Ordering::Relaxed);
                    }
                    if let Some(ev) = ev {
                        execute_event(
                            &mut core,
                            &policy_tx,
                            &checkpoint_dir,
                            &control,
                            &base_dir,
                            ev,
                        )
                        .await;
                    }
                }
                Err(e) => {
                    // Panic during the update: the core dies with the join —
                    // fail-closed. The task ends WITHOUT draining (there is no
                    // valid state left to process more experience): the last
                    // published snapshot stays intact, `tx` closes →
                    // `learner_dead()` reports the death and the strategy keeps
                    // serving the current policy.
                    control.alive.store(false, Ordering::Relaxed);
                    control.note(format!("update worker died: {e}"));
                    tracing::error!(
                        "rl_learner: spawn_blocking for the update failed ({e}), learner stopped"
                    );
                    return;
                }
            }
        } else if core.boundary_since_ms.is_some() {
            // Freeze-learning: we keep collecting experience but do not
            // publish — an already raised boundary is cancelled ONCE so it does
            // not block entries; while still frozen it is not re-raised.
            core.boundary_since_ms = None;
            control.boundary.store(false, Ordering::Relaxed);
        }
    }
    // Drain what is left (clean shutdown: no queue is left unaccounted for).
    while let Ok(m) = rx.try_recv() {
        process_msg(&mut core, &control, m);
    }
    control.alive.store(false, Ordering::Relaxed);
    control.note("learner stopped");
    tracing::info!("rl_learner: task stopped — last policy v{} kept", core.version);
}

fn process_msg(core: &mut LearnerCore, control: &LearnerControl, msg: LearnerMsg) {
    match msg {
        LearnerMsg::EpisodeStart {
            episode_id,
            policy_version,
            ts_ms,
        } => {
            if let Some(evicted) = core.on_episode_start(episode_id, policy_version, ts_ms) {
                // The `pending` cap evicted the stalest episode — almost certainly
                // an orphan whose `EpisodeEnd` was lost.
                control.pending_evicted.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("hrf_rl_pending_evicted_total").increment(1);
                tracing::warn!(
                    ep = %evicted,
                    "rl_learner: `pending` cap evicted the stalest in-flight episode (possibly a lost EpisodeEnd)"
                );
            }
        }
        LearnerMsg::Transition(tr) => {
            control.transitions_rx.fetch_add(1, Ordering::Relaxed);
            if !core.on_transition(*tr) {
                // Episode with no known EpisodeStart (start lost to a full channel,
                // or a trajectory predating a restart) — dedicated counter, distinct
                // from drops due to a full channel.
                control.transitions_orphan.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("hrf_rl_transitions_orphan_total").increment(1);
            }
        }
        LearnerMsg::EpisodeEnd(end) => {
            match core.on_episode_end(&end.episode_id, end.trainable) {
                Some((true, evicted, trs)) => {
                    control.episodes_completed.fetch_add(1, Ordering::Relaxed);
                    if evicted > 0 {
                        control
                            .episodes_evicted
                            .fetch_add(evicted as u64, Ordering::Relaxed);
                        metrics::counter!("hrf_rl_episodes_evicted_total")
                            .increment(evicted as u64);
                        tracing::warn!(
                            evicted,
                            "rl_learner: cohort cap discarded complete episodes (oldest first)"
                        );
                    }
                    metrics::counter!("hrf_rl_episodes_completed_total").increment(1);
                    tracing::debug!(
                        ep = %end.episode_id,
                        transitions = trs.len(),
                        "rl_learner: complete episode added to the cohort"
                    );
                }
                Some((false, ..)) => {
                    control.episodes_excluded.fetch_add(1, Ordering::Relaxed);
                    metrics::counter!("hrf_rl_episodes_excluded_total").increment(1);
                    let invalid = end.invalid_reason.unwrap_or_default();
                    tracing::info!(
                        ep = %end.episode_id,
                        reason = %end.reason.unwrap_or_else(|| "?".into()),
                        invalid_reason = %invalid,
                        "rl_learner: episode excluded from cohort"
                    );
                    if !invalid.is_empty() {
                        control.note(format!("episode excluded: {invalid}"));
                    }
                }
                None => {
                    control.episodes_dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Executes a `CoreEvent`: writes the checkpoint BEFORE publishing the snapshot
/// (a crash between the two must leave the disk consistent: first the complete
/// artifact, then the pointer, and finally the publication).
/// The write is synchronous I/O — it runs in `spawn_blocking` so it does not
/// occupy the runtime thread; the publish still waits for the join.
async fn execute_event(
    core: &mut LearnerCore,
    policy_tx: &watch::Sender<Arc<PolicySnapshot>>,
    checkpoint_dir: &PathBuf,
    control: &Arc<LearnerControl>,
    base_dir: &PathBuf,
    ev: CoreEvent,
) {
    match ev {
        CoreEvent::BoundaryOn => {
            control.boundary.store(true, Ordering::Relaxed);
            control.note("cohort at boundary — entries blocked");
            metrics::counter!("hrf_rl_cohort_boundary_total").increment(1);
        }
        CoreEvent::UpdateRejected { reason } => {
            control.updates_rejected.fetch_add(1, Ordering::Relaxed);
            control.boundary.store(false, Ordering::Relaxed);
            control.note(format!("update rejected: {reason}"));
            metrics::counter!("hrf_rl_updates_rejected_total").increment(1);
            tracing::warn!("rl_learner: update REJECTED ({reason}) — previous policy left intact");
        }
        CoreEvent::Publish {
            snapshot,
            mut meta,
            policy_bytes,
            stats,
        } => {
            meta.base_artifact_dir = base_dir.to_string_lossy().into_owned();
            let dir_ck = checkpoint_dir.clone();
            let policy_ck = snapshot.policy.clone();
            let meta_ck = meta.clone();
            let wr = tokio::task::spawn_blocking(move || {
                artifact::write_checkpoint(&dir_ck, &policy_ck, &meta_ck, &policy_bytes)
            })
            .await;
            // This arm only translates the `JoinHandle` into an outcome; applying
            // the invariants (`control.boundary` released on EVERY outcome,
            // version/state intact on failure) lives in `apply_outcome_publish` —
            // synchronous and testable without a runtime or filesystem.
            let outcome = match wr {
                Ok(Ok(d)) => CheckpointOutcome::Written(d),
                Ok(Err(PolicyArtifactError::Schema { detail }))
                    if detail.contains("already exists") =>
                {
                    // Duplicate version — counter bug; do not publish.
                    CheckpointOutcome::Duplicate(detail)
                }
                Ok(Err(e)) => {
                    tracing::error!("rl_learner: could not write checkpoint: {e}");
                    CheckpointOutcome::Failed(format!("checkpoint failed: {e}"))
                }
                Err(e) => {
                    tracing::error!("rl_learner: checkpoint spawn_blocking failed: {e}");
                    CheckpointOutcome::Failed(format!("checkpoint join failed: {e}"))
                }
            };
            apply_outcome_publish(core, policy_tx, control, snapshot, stats, outcome);
        }
    }
}

/// Outcome of the checkpoint write attempt — the async task translates the
/// `JoinHandle` into this shape so that applying the invariants is a pure,
/// testable function (a real `JoinError` cannot be injected without sabotaging
/// the runtime).
enum CheckpointOutcome {
    /// `checkpoints/vNNNNNN` directory written.
    Written(PathBuf),
    /// The version already existed on disk (`Schema "already exists"`) — learner
    /// counter bug; do not publish.
    Duplicate(String),
    /// Io/Schema/join failure of `spawn_blocking` — same outcome: update
    /// rejected, current policy intact.
    Failed(String),
}

/// Applies the publish result: success → `confirm_publish` + send of the
/// snapshot + counters; failure → `updates_rejected` + note, WITHOUT advancing
/// `core.version` nor `control.version`/`update_index`/`policy_sha`, nor
/// publishing through `policy_tx` — the current policy stays intact and the
/// next cohort retries the publish.
///
/// `control.boundary` is released on ALL outcomes via the RAII guard — a
/// checkpoint failure (disk full, permissions, duplicate version, join) must
/// not block purchases forever: the flag feeds `mask_legal` through
/// `boundary_active`.
fn apply_outcome_publish(
    core: &mut LearnerCore,
    policy_tx: &watch::Sender<Arc<PolicySnapshot>>,
    control: &LearnerControl,
    snapshot: PolicySnapshot,
    stats: UpdateResult,
    outcome: CheckpointOutcome,
) {
    // RAII guard: covers early returns and any panic in this section —
    // the boundary is never left stuck after a failed publish.
    let _off = FlagOff(&control.boundary);
    let dir = match outcome {
        CheckpointOutcome::Written(d) => d,
        CheckpointOutcome::Duplicate(detail) => {
            control.updates_rejected.fetch_add(1, Ordering::Relaxed);
            control.note(format!("duplicate checkpoint: {detail}"));
            return;
        }
        CheckpointOutcome::Failed(reason) => {
            control.updates_rejected.fetch_add(1, Ordering::Relaxed);
            control.note(reason);
            return;
        }
    };
    control.checkpoints_written.fetch_add(1, Ordering::Relaxed);
    core.confirm_publish(&snapshot);
    let _ = policy_tx.send(Arc::new(snapshot.clone_light()));
    control
        .version
        .store(core.version, Ordering::Relaxed);
    control
        .update_index
        .store(core.update_index, Ordering::Relaxed);
    *control.policy_sha.lock().unwrap() = core.policy_sha256.clone();
    control.updates_applied.fetch_add(1, Ordering::Relaxed);
    control.note(format!(
        "published v{} ({} episodes, mean return {:.4}, ‖Δw‖₂={:.4}, mean KL={:.4})",
        core.version, stats.episodes, stats.mean_return, stats.update_l2, stats.mean_kl
    ));
    metrics::counter!("hrf_rl_updates_applied_total").increment(1);
    tracing::info!(
        version = core.version,
        update_index = core.update_index,
        episodes = stats.episodes,
        transitions = stats.transitions,
        mean_return = stats.mean_return,
        update_l2 = stats.update_l2,
        mean_kl = stats.mean_kl,
        checkpoint = %dir.display(),
        "rl_learner: new policy version published"
    );
}

impl PolicySnapshot {
    /// Cheap clone of the snapshot sharing the policy (the internal policy is
    /// already an Arc — the snapshot does not duplicate weights).
    fn clone_light(&self) -> PolicySnapshot {
        PolicySnapshot {
            version: self.version,
            update_index: self.update_index,
            policy: self.policy.clone(),
            policy_sha256: self.policy_sha256.clone(),
            parent_sha256: self.parent_sha256.clone(),
            base_sha256: self.base_sha256.clone(),
            published_ts_ms: self.published_ts_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::contract::*;

    fn policy_zeros() -> LinearSoftmaxPolicy {
        LinearSoftmaxPolicy {
            scaler: crate::rl::policy::PolicyScaler {
                mean: vec![0.0; NUM_FEATURES],
                scale: vec![1.0; NUM_FEATURES],
            },
            weights: vec![vec![0.0; NUM_FEATURES]; NUM_ACTIONS],
            bias: vec![0.0; NUM_ACTIONS],
        }
    }

    fn cfg_test() -> ReinforceConfig {
        ReinforceConfig {
            learning_rate: 0.5,
            max_grad_norm: 100.0,
            max_update_l2: 100.0,
            max_mean_kl: 10.0,
            max_step_kl: 10.0,
            min_episodes_per_update: 2,
            ..Default::default()
        }
    }

    fn training_doc() -> PolicyTrainingDoc {
        PolicyTrainingDoc {
            run_id: "test-base".into(),
            data_manifest_sha256: artifact::EMPTY_MANIFEST_SHA256.into(),
            split_manifest_sha256: artifact::EMPTY_MANIFEST_SHA256.into(),
            seed: 7,
            completed_episodes: 100,
            updates: 5,
        }
    }

    fn core_test() -> LearnerCore {
        LearnerCore::new(
            policy_zeros(),
            "base".into(),
            "base".into(),
            "pred".into(),
            cfg_test(),
            training_doc(),
        )
    }

    fn trans(ep: &str, action: usize, obs: [f64; NUM_FEATURES], reward: f64) -> TransitionRec {
        let policy = policy_zeros();
        let probs = policy.action_probs(&obs, &[true; NUM_ACTIONS]).unwrap();
        TransitionRec {
            episode_id: ep.into(),
            policy_version: 0,
            obs,
            mask: [true; NUM_ACTIONS],
            action,
            behavior_prob: probs[action],
            probs,
            sampled: true,
            reward_norm: reward,
            net_pnl_delta_um: 0,
            ts_ms: 0,
        }
    }

    /// Regression test for the measurement clock scale: `monotonic_ns` returns
    /// NANOSECONDS of the process monotonic clock — if the unit changed (µs/ms)
    /// this test would catch it, because comparisons with the inference probes
    /// require the same scale.
    #[test]
    fn monotonic_ns_is_nanoseconds_and_advances() {
        let t0 = monotonic_ns();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let t1 = monotonic_ns();
        let d = t1 - t0;
        assert!(d > 0, "the monotonic clock does not advance");
        assert!(d >= 1_000_000, "sleep 2 ms must give ≥1e6 ns, got {d}: wrong scale");
        assert!(d < 1_000_000_000, "sleep 2 ms must give <1e9 ns, got {d}: not ns");
    }

    /// Zero reward -> null advantage across the whole cohort -> update rejected
    /// as `DeltaZero` (the current policy is kept without versioning).
    #[test]
    fn reward_zero_does_not_change_weights() {
        let p = policy_zeros();
        let eps = vec![vec![
            trans("e1", BUY_YES_FULL, {
                let mut o = [0.0; NUM_FEATURES];
                o[F_P_YES] = 0.8;
                o
            }, 0.0),
        ]];
        let res = reinforce_update(&p, &eps, &ReturnBaseline::default(), &cfg_test());
        assert!(matches!(res, Err(UpdateReject::DeltaZero)));
    }

    /// Positive reward -> the logit of the chosen action RISES on that obs.
    /// The gradient also depends on the STATE (the obs x): the same action with
    /// a different obs produces a different Δw. This is the key test for the
    /// shared-features cancellation case.
    #[test]
    fn positive_reward_raises_logit_and_depends_on_state() {
        let p = policy_zeros();
        let obs_a = {
            let mut o = [0.0; NUM_FEATURES];
            o[F_P_YES] = 1.0;
            o
        };
        let eps = vec![vec![trans("e1", BUY_YES_FULL, obs_a, 1.0)]];
        let res = reinforce_update(&p, &eps, &ReturnBaseline::default(), &cfg_test()).unwrap();
        let logit_new = res
            .new_policy
            .logits_masked(&obs_a, &[true; NUM_ACTIONS])
            .unwrap()[BUY_YES_FULL];
        let logit_old = p.logits_masked(&obs_a, &[true; NUM_ACTIONS]).unwrap()[BUY_YES_FULL];
        assert!(logit_new > logit_old, "the rewarded action must raise its logit");

        // State dependence: Δw·x must be proportional to x; with a different obs
        // the logit change differs (the gradient is x·coef).
        let obs_b = {
            let mut o = [0.0; NUM_FEATURES];
            o[F_P_YES] = -1.0;
            o
        };
        let dl_a = res.new_policy.weights[BUY_YES_FULL][F_P_YES] * obs_a[F_P_YES];
        let dl_b = res.new_policy.weights[BUY_YES_FULL][F_P_YES] * obs_b[F_P_YES];
        assert_ne!(dl_a, dl_b, "the logit change depends on x (state)");
    }

    /// Two different observations produce different gradients in the
    /// direction of their features: genuine state dependence, not a
    /// uniform shared gradient.
    #[test]
    fn gradient_differs_per_state() {
        let p = policy_zeros();
        let obs_a = {
            let mut o = [0.0; NUM_FEATURES];
            o[F_P_YES] = 1.0;
            o
        };
        let obs_b = {
            let mut o = [0.0; NUM_FEATURES];
            o[F_FREE_BUDGET] = 1.0;
            o
        };
        let res_a = reinforce_update(
            &p,
            &[vec![trans("e", BUY_YES_FULL, obs_a, 1.0)]],
            &ReturnBaseline::default(),
            &cfg_test(),
        )
        .unwrap();
        let res_b = reinforce_update(
            &p,
            &[vec![trans("e", BUY_YES_FULL, obs_b, 1.0)]],
            &ReturnBaseline::default(),
            &cfg_test(),
        )
        .unwrap();
        assert!(res_a.new_policy.weights[BUY_YES_FULL][F_P_YES] != 0.0);
        assert_eq!(res_a.new_policy.weights[BUY_YES_FULL][F_FREE_BUDGET], 0.0);
        assert!(res_b.new_policy.weights[BUY_YES_FULL][F_FREE_BUDGET] != 0.0);
        assert_eq!(res_b.new_policy.weights[BUY_YES_FULL][F_P_YES], 0.0);
    }

    /// Masked actions receive neither probability nor gradient.
    #[test]
    fn masked_action_without_gradient() {
        let p = policy_zeros();
        let mut t = trans("e", BUY_YES_FULL, {
            let mut o = [0.0; NUM_FEATURES];
            o[F_P_YES] = 1.0;
            o
        }, 1.0);
        // Only HOLD and BUY_YES_FULL are legal.
        t.mask = MASK_HOLD_ONLY;
        t.mask[BUY_YES_FULL] = true;
        t.probs = p.action_probs(&t.obs, &t.mask).unwrap();
        t.behavior_prob = t.probs[BUY_YES_FULL];
        let res = reinforce_update(
            &p,
            &[vec![t]],
            &ReturnBaseline::default(),
            &cfg_test(),
        )
        .unwrap();
        // Illegal actions do not move (prob 0 -> coef 0).
        for a in 0..NUM_ACTIONS {
            if a != HOLD && a != BUY_YES_FULL {
                assert!(res.new_policy.weights[a].iter().all(|w| *w == 0.0));
            }
        }
    }

    /// Argmax transitions (sampled=false) do NOT contribute gradient.
    #[test]
    fn argmax_does_not_learn() {
        let p = policy_zeros();
        let mut t = trans("e", BUY_YES_FULL, {
            let mut o = [0.0; NUM_FEATURES];
            o[F_P_YES] = 1.0;
            o
        }, 5.0);
        t.sampled = false;
        let res = reinforce_update(&p, &[vec![t]], &ReturnBaseline::default(), &cfg_test());
        assert!(matches!(res, Err(UpdateReject::NoTransitions)));
    }

    /// Cohort with gated KL: a huge LR trips the gate and the update
    /// is rejected, leaving the policy intact.
    #[test]
    fn gate_l2_rejects_huge_update() {
        let p = policy_zeros();
        let mut cfg = cfg_test();
        cfg.learning_rate = 1000.0;
        cfg.max_grad_norm = 0.0; // no clip, so the L2 gate bites
        cfg.max_update_l2 = 0.01;
        let eps = vec![vec![trans("e", BUY_YES_FULL, {
            let mut o = [0.0; NUM_FEATURES];
            o[F_P_YES] = 1.0;
            o
        }, 10.0)]];
        match reinforce_update(&p, &eps, &ReturnBaseline::default(), &cfg) {
            Err(UpdateReject::UpdateL2 { .. }) => {}
            other => panic!("expected UpdateL2, got {:?}", other.map(|_| ())),
        }
    }

    /// Core cohort: a whole episode under v0 is trained; one that mixes
    /// versions is excluded.
    #[test]
    fn core_excludes_mixed_version_episodes() {
        let mut core = core_test();
        let _ = core.on_episode_start("e1".into(), 0, 0);
        core.on_transition(trans("e1", HOLD, [0.0; NUM_FEATURES], 0.1));
        // e1 is still open when the version "changes": simulate it by marking
        // the next transition with another version.
        core.on_transition(TransitionRec {
            policy_version: 99,
            ..trans("e1", HOLD, [0.0; NUM_FEATURES], 0.1)
        });
        let (ok, ..) = core.on_episode_end("e1", true).unwrap();
        assert!(!ok, "episode with mixed versions must be excluded");
    }

    /// A clean, complete episode enters the cohort; its return ONLY feeds the
    /// baseline after a published update (the cohort is never compared against
    /// itself, `trainer.py` parity).
    #[test]
    fn core_complete_cohort_does_not_contaminate_baseline() {
        let mut core = core_test();
        let _ = core.on_episode_start("e1".into(), 0, 0);
        core.on_transition(trans("e1", HOLD, [0.0; NUM_FEATURES], 0.5));
        let (ok, _evicted, trs) = core.on_episode_end("e1", true).unwrap();
        assert!(ok);
        assert_eq!(trs.len(), 1);
        assert_eq!(core.cohort_done.len(), 1);
        // The return stays pending: baseline intact until the update.
        assert_eq!(core.baseline.n, 0);
        assert_eq!(core.baseline.value(), 0.0);
    }

    /// With the cohort closed, the update uses the baseline of the PREVIOUS
    /// batches (0 in the first one); it is folded in only after publishing.
    #[test]
    fn baseline_only_previous_batches() {
        let mut core = core_test();
        // Two episodes, return +1 each (min_episodes_per_update=2).
        for name in ["e1", "e2"] {
            let _ = core.on_episode_start(name.into(), 0, 0);
            let mut obs = [0.0; NUM_FEATURES];
            obs[F_P_YES] = 0.3;
            core.on_transition(trans(name, BUY_YES_FULL, obs, 1.0));
            let (ok, ..) = core.on_episode_end(name, true).unwrap();
            assert!(ok);
        }
        assert_eq!(core.baseline.n, 0, "the current cohort does not contaminate its baseline");
        // boundary + update: the cohort is cut with baseline=0.
        assert!(matches!(core.maybe_boundary(1_000), Some(CoreEvent::BoundaryOn)));
        let ev = core.maybe_update(1_000).expect("complete cohort -> update");
        let CoreEvent::Publish { stats, meta, .. } = ev else {
            panic!("expected Publish, got another event");
        };
        assert_eq!(stats.episodes, 2);
        // After the update succeeds, the cohort's returns are folded in;
        // the next batch will see baseline = 1.0.
        assert_eq!(core.baseline.n, 2);
        assert!((core.baseline.value() - 1.0).abs() < 1e-9);
        // The meta counters reflect the baseline after the fold.
        assert_eq!(meta.counters["baseline_n"], serde_json::json!(2));
    }

    /// A REJECTED update does not fold returns into the baseline (`trainer.py`
    /// parity: `raise UpdateAborted` before the fold).
    #[test]
    fn rejected_update_does_not_fold_baseline() {
        let mut core = core_test();
        // Huge lr -> max_update_l2 gate rejects -> baseline intact.
        core.cfg.max_update_l2 = 0.0001;
        for name in ["e1", "e2"] {
            let _ = core.on_episode_start(name.into(), 0, 0);
            let mut obs = [0.0; NUM_FEATURES];
            obs[F_P_YES] = 0.3;
            core.on_transition(trans(name, BUY_YES_FULL, obs, 1.0));
            core.on_episode_end(name, true);
        }
        assert!(matches!(core.maybe_boundary(1_000), Some(CoreEvent::BoundaryOn)));
        match core.maybe_update(1_000) {
            Some(CoreEvent::UpdateRejected { .. }) => {}
            other => panic!("expected UpdateRejected, got {:?}", other.map(|_| ())),
        }
        assert_eq!(core.baseline.n, 0, "aborted update does not feed the baseline");
    }

    /// Transition for an unknown episode -> false (dedicated counter
    /// `transitions_orphan` in the caller), never silent.
    #[test]
    fn orphan_transition_rejected() {
        let mut core = core_test();
        assert!(!core.on_transition(trans("phantom", HOLD, [0.0; NUM_FEATURES], 0.1)));
    }

    /// The cohort cap evicts the oldest episode and returns
    /// it counted (the caller adds it to `episodes_evicted`).
    #[test]
    fn cohort_cap_counts_evictions() {
        let mut core = core_test();
        core.cfg.cohort_cap = 2;
        for name in ["e1", "e2", "e3"] {
            let _ = core.on_episode_start(name.into(), 0, 0);
            core.on_transition(trans(name, HOLD, [0.0; NUM_FEATURES], 0.1));
        }
        let (_, e1, _) = core.on_episode_end("e1", true).unwrap();
        assert_eq!(e1, 0);
        let (_, e2, _) = core.on_episode_end("e2", true).unwrap();
        assert_eq!(e2, 0);
        let (_, e3, _) = core.on_episode_end("e3", true).unwrap();
        assert_eq!(e3, 1, "the cap evicts the oldest episode and reports it");
        assert_eq!(core.cohort_done.len(), 2);
    }

    /// Mean per STEP (not per episode), L2 clip of the gradient BEFORE the
    /// learning rate, probs recomputed (not the recorded ones).
    /// Numerical check against the canonical formula in `parity.py`.
    #[test]
    fn canonical_update_mean_per_step_and_clip_pre_lr() {
        // Policy with bias -> non-uniform probs; two steps with different masks
        // and features -> heterogeneous gradient that can be checked by hand.
        let mut p = policy_zeros();
        p.bias[BUY_YES_FULL] = 0.5;
        p.bias[HOLD] = -0.2;
        let scaler_id = crate::rl::policy::PolicyScaler {
            mean: vec![0.0; NUM_FEATURES],
            scale: vec![1.0; NUM_FEATURES],
        };
        p.scaler = scaler_id.clone();

        // ep1: 2 steps (BUY with reward 1.0, HOLD with reward 0.0);
        // ep2: 1 step (SELL with reward 2.0). n_steps=3.
        let obs1 = {
            let mut o = [0.0; NUM_FEATURES];
            o[F_P_YES] = 0.7;
            o[F_FREE_BUDGET] = 0.4;
            o
        };
        let obs2 = {
            let mut o = [0.0; NUM_FEATURES];
            o[F_P_YES] = 0.2;
            o
        };
        let mk = |ep: &str, obs, action, reward| {
            let mut t = trans(ep, action, obs, reward);
            t.probs = p.action_probs(&obs, &t.mask).unwrap();
            t.behavior_prob = t.probs[action];
            t
        };
        let eps = vec![
            vec![mk("e1", obs1, BUY_YES_FULL, 1.0), mk("e1", obs2, HOLD, 0.0)],
            vec![mk("e2", obs2, SELL_YES_ALL, 2.0)],
        ];
        let baseline = ReturnBaseline { sum: 0.4, n: 2 }; // b = 0.2
        let cfg = ReinforceConfig {
            learning_rate: 0.1,
            max_grad_norm: 1e9, // harmless clip: formula without scaling
            max_update_l2: 1e9,
            max_mean_kl: 1e9,
            max_step_kl: 1e9,
            min_episodes_per_update: 2,
            ..Default::default()
        };
        let res = reinforce_update(&p, &eps, &baseline, &cfg).unwrap();
        assert_eq!(res.transitions, 3, "n_steps counts ALL steps of the cohort");

        // Canonical reference computed by hand (parity.py):
        //   G: ep1=[1.0,0.0] ep2=[2.0]; b=0.2 → adv=[0.8,-0.2,1.8]
        //   gl_a = adv·(δ−π_a); dW[a][f] += gl_a·x[f]; db[a] += gl_a
        //   mean /3; Δw = lr·grad (harmless clip)
        let probs1 = p.action_probs(&obs1, &[true; NUM_ACTIONS]).unwrap();
        let probs2 = p.action_probs(&obs2, &[true; NUM_ACTIONS]).unwrap();
        let steps = [
            (0.8f64, probs1, BUY_YES_FULL, obs1),
            (-0.2, probs2, HOLD, obs2),
            (1.8, probs2, SELL_YES_ALL, obs2),
        ];
        let mut gw = [[0.0f64; NUM_FEATURES]; NUM_ACTIONS];
        let mut gb = [0.0f64; NUM_ACTIONS];
        for (adv, probs, a_t, x) in steps {
            for a in 0..NUM_ACTIONS {
                let gl = adv * ((a == a_t) as u8 as f64 - probs[a]);
                gb[a] += gl;
                for f in 0..NUM_FEATURES {
                    gw[a][f] += gl * x[f];
                }
            }
        }
        let inv = 1.0 / 3.0;
        for a in 0..NUM_ACTIONS {
            let exp_b = gb[a] * inv * 0.1;
            assert!(
                (res.new_policy.bias[a] - (p.bias[a] + exp_b)).abs() < 1e-12,
                "bias[{a}] diverges from the canonical formula"
            );
            for f in 0..NUM_FEATURES {
                let exp_w = gw[a][f] * inv * 0.1;
                assert!(
                    (res.new_policy.weights[a][f] - (p.weights[a][f] + exp_w)).abs() < 1e-12,
                    "weights[{a}][{f}] diverges from the canonical formula"
                );
            }
        }
    }

    /// The L2 clip is applied to the GRADIENT (pre-lr). With grad_clip=0.05
    /// and lr=2: the policy moves ‖Δw‖ = 2·0.05 (pre-lr clip), not 0.05.
    #[test]
    fn clip_l2_before_lr() {
        let p = policy_zeros();
        let mut obs = [0.0; NUM_FEATURES];
        obs[F_P_YES] = 1.0;
        let mut t = trans("e", BUY_YES_FULL, obs, 1.0);
        t.probs = p.action_probs(&obs, &t.mask).unwrap();
        let cfg = ReinforceConfig {
            learning_rate: 2.0,
            max_grad_norm: 0.05, // clip of the mean gradient
            max_update_l2: 1e9,
            max_mean_kl: 1e9,
            max_step_kl: 1e9,
            min_episodes_per_update: 1,
            ..Default::default()
        };
        let res = reinforce_update(&p, &[vec![t]], &ReturnBaseline::default(), &cfg).unwrap();
        // ‖Δw‖ = lr · min(‖grad‖, clip) = 2 · 0.05 = 0.1
        assert!(
            (res.update_l2 - 0.1).abs() < 1e-9,
            "‖Δw‖={} must be lr·clip=0.1 (pre-lr clip)",
            res.update_l2
        );
        assert!(res.grad_l2_pre_clip > 0.05, "unclipped gradient exceeds the clip");
    }

    /// Chosen action with recomputed π = 0 (mask without that action in the
    /// recorded probs but flagged sampled) aborts as off-policy.
    /// Equivalently: manipulated recorded probs are not trusted.
    #[test]
    fn recomputed_probs_abort_offpolicy() {
        let p = policy_zeros();
        let mut obs = [0.0; NUM_FEATURES];
        obs[F_P_YES] = 1.0;
        // Transition with a mask that does NOT include the chosen action:
        // recomputed π(a_t)=0 -> abort (corrupt data).
        let mut t = trans("e", BUY_YES_FULL, obs, 1.0);
        t.mask = MASK_HOLD_ONLY;
        t.probs = p.action_probs(&obs, &t.mask).unwrap();
        match reinforce_update(&p, &[vec![t]], &ReturnBaseline::default(), &cfg_test()) {
            Err(UpdateReject::OffPolicy) => {}
            other => panic!("expected OffPolicy, got {:?}", other.map(|_| ())),
        }
    }

    /// Under freeze the boundary is disarmed ONCE and not re-armed (no gate
    /// oscillation); on unfreeze the cohort is still complete and the boundary
    /// re-arms: the strategy is never blocked indefinitely.
    /// Exercises the real `run_learner` loop.
    #[tokio::test]
    async fn freeze_disarms_boundary_without_oscillating() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let cfg = ReinforceConfig {
            min_episodes_per_update: 2,
            cohort_cap: 8,
            boundary_max_wait_ms: 60_000,
            tick_ms: 20,
            ..cfg_test()
        };
        let mut core = LearnerCore::new(
            policy_zeros(),
            "base".into(),
            "base".into(),
            "pred".into(),
            cfg,
            training_doc(),
        );
        // Complete cohort (2 closed ids) + one episode that stays open and
        // holds the boundary's cut.
        let _ = core.on_episode_start("a".into(), 0, 0);
        let _ = core.on_episode_start("b".into(), 0, 0);
        let _ = core.on_episode_start("open".into(), 0, 0);
        core.on_episode_end("a", true);
        core.on_episode_end("b", true);

        let control = Arc::new(LearnerControl::new());
        let (tx, rx) = mpsc::channel(64);
        let (ptx, _prx) = watch::channel(Arc::new(PolicySnapshot {
            version: 0,
            update_index: 0,
            policy: Arc::new(policy_zeros()),
            policy_sha256: "base".into(),
            parent_sha256: String::new(),
            base_sha256: "base".into(),
            published_ts_ms: 0,
        }));
        let ckpt = dir.path().to_path_buf();
        let task = tokio::spawn(run_learner(
            core,
            rx,
            ptx,
            ckpt,
            control.clone(),
            policy_zeros(),
            dir.path().to_path_buf(),
        ));

        // Full cohort -> the first tick arms the boundary (the open episode
        // holds the cut).
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            control.boundary_active(),
            "a complete cohort must arm the boundary"
        );

        // Freeze: disarms once and does NOT re-arm: no gate flapping.
        control.set_frozen(true);
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            !control.boundary_active(),
            "freeze must disarm the boundary (once)"
        );
        // Second long stretch: if it oscillated, the flag would become true again.
        assert!(
            !control.boundary_active(),
            "boundary must not re-arm under freeze"
        );

        // Unfreeze: the cohort is still complete -> re-arms (no permanent blocker).
        control.set_frozen(false);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            control.boundary_active(),
            "on unfreeze the boundary re-arms (cohort still complete)"
        );

        drop(tx); // closes the channel -> the worker finishes cleanly
        task.await.unwrap();
        assert!(!control.is_alive());
    }

    /// CROSS parity check: the fixture `tests/fixtures/parity_reinforce_v1.json`
    /// is generated by `rl_predictor.integration.gen_parity_fixture` with the
    /// canonical math of `parity.py` (`full_update`). The test feeds the real
    /// INPUT through `reinforce_update` and compares each parameter with the
    /// Python output, including the pre-clip norm and the pre-lr clip.
    /// The per-step `probs` of the fixture (Python softmax) are cross-checked
    /// against those recomputed by the Rust policy (softmax+mask parity).
    #[test]
    fn parity_fixture_python() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/parity_reinforce_v1.json"
        );
        let fx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path).expect("parity fixture: regenerate with gen_parity_fixture.py"),
        )
        .unwrap();
        assert_eq!(
            fx["kind"].as_str().unwrap(),
            "rl_predictor.parity.reinforce_update.v1"
        );

        // Entry policy (identity scaler: the fixture's x is already x_std).
        let mut p = policy_zeros();
        for (a, row) in fx["policy"]["weights"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            for (f, v) in row.as_array().unwrap().iter().enumerate() {
                p.weights[a][f] = v.as_f64().unwrap();
            }
        }
        for (a, v) in fx["policy"]["bias"].as_array().unwrap().iter().enumerate() {
            p.bias[a] = v.as_f64().unwrap();
        }

        let mut episodes: Vec<Vec<TransitionRec>> = Vec::new();
        for (i, ep) in fx["cohort"].as_array().unwrap().iter().enumerate() {
            let mut trs = Vec::new();
            for s in ep.as_array().unwrap() {
                let mut obs = [0.0; NUM_FEATURES];
                for (f, v) in s["x_std"].as_array().unwrap().iter().enumerate() {
                    obs[f] = v.as_f64().unwrap();
                }
                let mut mask = [false; NUM_ACTIONS];
                let bits = s["mask"].as_u64().unwrap();
                for a in 0..NUM_ACTIONS {
                    mask[a] = bits & (1 << a) != 0;
                }
                let action = s["action"].as_u64().unwrap() as usize;
                let probs = p.action_probs(&obs, &mask).unwrap();
                // Softmax+mask parity: the probs recorded by Python must match
                // those recomputed in Rust.
                for (a, pv) in s["probs"].as_array().unwrap().iter().enumerate() {
                    assert!(
                        (probs[a] - pv.as_f64().unwrap()).abs() < 1e-12,
                        "probs[{a}] diverges from the Python softmax in ep{i}"
                    );
                }
                trs.push(TransitionRec {
                    episode_id: format!("fx-ep{i}"),
                    policy_version: 0,
                    obs,
                    mask,
                    action,
                    behavior_prob: probs[action],
                    probs,
                    // `sampled` flag of the fixture (warmup/eval contributes no
                    // gradient); absent = true for compatibility.
                    sampled: s["sampled"].as_bool().unwrap_or(true),
                    reward_norm: s["reward"].as_f64().unwrap(),
                    net_pnl_delta_um: 0,
                    ts_ms: 0,
                });
            }
            episodes.push(trs);
        }

        let baseline = ReturnBaseline {
            sum: fx["baseline"].as_f64().unwrap(),
            n: 1,
        };
        let cfg = ReinforceConfig {
            learning_rate: fx["lr"].as_f64().unwrap(),
            max_grad_norm: fx["grad_clip"].as_f64().unwrap(),
            max_update_l2: 1e9,
            max_mean_kl: 1e9,
            max_step_kl: 1e9,
            ..cfg_test()
        };
        let res = reinforce_update(&p, &episodes, &baseline, &cfg).unwrap();
        let exp = &fx["expected"];
        assert_eq!(res.transitions, exp["n_steps"].as_u64().unwrap() as usize);
        assert!(
            (res.grad_l2_pre_clip - exp["grad_norm_preclip"].as_f64().unwrap()).abs() < 1e-9,
            "pre-clip gradient norm diverges from parity.py"
        );
        // The fixture's clip MUST engage (scale < 1): if ‖Δw‖ ≈ lr·clip, it
        // confirms the clip happens BEFORE the lr (not after).
        assert!(exp["clip_scale"].as_f64().unwrap() < 1.0);
        for (a, row) in exp["weights"].as_array().unwrap().iter().enumerate() {
            for (f, v) in row.as_array().unwrap().iter().enumerate() {
                assert!(
                    (res.new_policy.weights[a][f] - v.as_f64().unwrap()).abs() < 1e-9,
                    "weights[{a}][{f}] diverge from the canonical Python update"
                );
            }
        }
        for (a, v) in exp["bias"].as_array().unwrap().iter().enumerate() {
            assert!(
                (res.new_policy.bias[a] - v.as_f64().unwrap()).abs() < 1e-9,
                "bias[{a}] diverges from the canonical Python update"
            );
        }
    }

    // ── publish failure releases `boundary` ───────────────────────────

    fn snapshot_v(version: u64, sha: &str) -> PolicySnapshot {
        PolicySnapshot {
            version,
            update_index: version,
            policy: Arc::new(policy_zeros()),
            policy_sha256: sha.into(),
            parent_sha256: "sha-v0".into(),
            base_sha256: "sha-v0".into(),
            published_ts_ms: 1_000,
        }
    }

    fn stats_test() -> UpdateResult {
        UpdateResult {
            new_policy: policy_zeros(),
            episodes: 2,
            transitions: 4,
            mean_return: 0.1,
            update_l2: 0.01,
            mean_kl: 0.001,
            max_kl: 0.002,
            grad_l2_pre_clip: 0.02,
        }
    }

    /// ALL failure outcomes of the publish release the `boundary` gate. Without
    /// the fix it stayed engaged, so `mask_legal` blocked all BUYs
    /// forever. The outcome is injected into the synchronous helper
    /// `apply_outcome_publish`. A real `JoinError` from `spawn_blocking` cannot be
    /// injected without sabotaging the runtime, so the "join" variant is tested at
    /// the outcome level (the `Publish` arm match that translates the `JoinHandle`
    /// is covered by the async `execute_event` tests below).
    #[test]
    fn failed_publish_releases_boundary() {
        for (name_str, outcome) in [
            (
                "io",
                CheckpointOutcome::Failed("checkpoint failed: simulated io".into()),
            ),
            (
                "duplicate",
                CheckpointOutcome::Duplicate(
                    "checkpoint v000001 already exists — duplicate version".into(),
                ),
            ),
            (
                "join",
                CheckpointOutcome::Failed("checkpoint join failed: task panicked".into()),
            ),
        ] {
            let mut core = core_test();
            let control = LearnerControl::new();
            let (ptx, prx) = watch::channel(Arc::new(snapshot_v(0, "sha-v0")));
            // Observable state as after BoundaryOn: gate armed and sha v0.
            control.boundary.store(true, Ordering::Relaxed);
            *control.policy_sha.lock().unwrap() = "sha-v0".into();

            apply_outcome_publish(
                &mut core,
                &ptx,
                &control,
                snapshot_v(1, "sha-v1"),
                stats_test(),
                outcome,
            );

            assert!(
                !control.boundary.load(Ordering::Relaxed),
                "{name_str}: boundary must be released after a failed publish"
            );
            assert_eq!(core.version, 0, "{name_str}: the current policy does not advance");
            assert_eq!(
                control.version.load(Ordering::Relaxed),
                0,
                "{name_str}: control.version does not advance"
            );
            assert_eq!(
                control.updates_rejected.load(Ordering::Relaxed),
                1,
                "{name_str}: the failure counts as a rejected update"
            );
            assert_eq!(
                control.checkpoints_written.load(Ordering::Relaxed),
                0,
                "{name_str}: there is no checkpoint"
            );
            assert_eq!(
                *control.policy_sha.lock().unwrap(),
                "sha-v0",
                "{name_str}: policy_sha intact"
            );
            assert_eq!(
                prx.borrow().version,
                0,
                "{name_str}: nothing is published via policy_tx"
            );
        }
    }

    /// Sanity check: the success outcome still publishes the version, snapshot
    /// and counters, and the guard also releases the boundary.
    #[test]
    fn successful_publish_publishes_and_releases_boundary() {
        let mut core = core_test();
        let control = LearnerControl::new();
        let (ptx, prx) = watch::channel(Arc::new(snapshot_v(0, "sha-v0")));
        control.boundary.store(true, Ordering::Relaxed);

        apply_outcome_publish(
            &mut core,
            &ptx,
            &control,
            snapshot_v(1, "sha-v1"),
            stats_test(),
            CheckpointOutcome::Written(PathBuf::from("checkpoints/v000001")),
        );

        assert!(!control.boundary.load(Ordering::Relaxed));
        assert_eq!(core.version, 1);
        assert_eq!(control.version.load(Ordering::Relaxed), 1);
        assert_eq!(control.updates_applied.load(Ordering::Relaxed), 1);
        assert_eq!(control.checkpoints_written.load(Ordering::Relaxed), 1);
        assert_eq!(*control.policy_sha.lock().unwrap(), "sha-v1");
        assert_eq!(prx.borrow().version, 1);
        assert_eq!(prx.borrow().policy_sha256, "sha-v1");
    }

    /// Through the REAL async path: `execute_event` with an impossible checkpoint
    /// directory (its parent is a regular file -> `create_dir_all` gives Io).
    /// The cohort is really cut with `close_cohort` and `spawn_blocking` runs on
    /// the test runtime.
    #[tokio::test]
    async fn execute_event_io_failure_releases_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let blocker = tmp.path().join("not-a-dir");
        std::fs::write(&blocker, b"x").unwrap();
        let ckpt_dir = blocker.join("ckpt"); // create_dir_all → ENOTDIR → Io

        let mut core = core_test();
        for name in ["e1", "e2"] {
            let _ = core.on_episode_start(name.into(), 0, 0);
            core.on_transition(trans(name, HOLD, {
                let mut o = [0.0; NUM_FEATURES];
                o[F_P_YES] = 0.3;
                o
            }, 1.0));
            core.on_episode_end(name, true);
        }
        assert!(matches!(
            core.maybe_boundary(1_000),
            Some(CoreEvent::BoundaryOn)
        ));
        let ev = core.maybe_update(1_000).expect("complete cohort -> Publish");
        assert!(matches!(ev, CoreEvent::Publish { .. }));

        let control = Arc::new(LearnerControl::new());
        control.boundary.store(true, Ordering::Relaxed);
        let (ptx, prx) = watch::channel(Arc::new(snapshot_v(0, "sha-v0")));
        let base_dir = tmp.path().to_path_buf();
        execute_event(&mut core, &ptx, &ckpt_dir, &control, &base_dir, ev).await;

        assert!(
            !control.boundary.load(Ordering::Relaxed),
            "a checkpoint Io error must not leave the boundary engaged"
        );
        assert_eq!(core.version, 0, "the current policy does not advance");
        assert_eq!(control.updates_rejected.load(Ordering::Relaxed), 1);
        assert_eq!(control.checkpoints_written.load(Ordering::Relaxed), 0);
        assert_eq!(prx.borrow().version, 0);
        assert!(
            control.status()["last_event"]
                .as_str()
                .unwrap()
                .contains("checkpoint failed"),
            "the note records the reason"
        );
    }

    /// Through the REAL path: `checkpoints/v000001/` already exists ->
    /// `write_checkpoint` returns `Schema "already exists"` (duplicate version) ->
    /// rejected without publishing, and the boundary released.
    #[tokio::test]
    async fn execute_event_duplicate_version_releases_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let ckpt_dir = tmp.path().join("ckpt");
        std::fs::create_dir_all(ckpt_dir.join("checkpoints/v000001")).unwrap();

        let mut core = core_test();
        for name in ["e1", "e2"] {
            let _ = core.on_episode_start(name.into(), 0, 0);
            core.on_transition(trans(name, HOLD, {
                let mut o = [0.0; NUM_FEATURES];
                o[F_P_YES] = 0.3;
                o
            }, 1.0));
            core.on_episode_end(name, true);
        }
        assert!(matches!(
            core.maybe_boundary(1_000),
            Some(CoreEvent::BoundaryOn)
        ));
        let ev = core.maybe_update(1_000).expect("complete cohort -> Publish");
        assert!(matches!(ev, CoreEvent::Publish { .. }));

        let control = Arc::new(LearnerControl::new());
        control.boundary.store(true, Ordering::Relaxed);
        let (ptx, prx) = watch::channel(Arc::new(snapshot_v(0, "sha-v0")));
        let base_dir = tmp.path().to_path_buf();
        execute_event(&mut core, &ptx, &ckpt_dir, &control, &base_dir, ev).await;

        assert!(
            !control.boundary.load(Ordering::Relaxed),
            "a duplicate version must not leave the boundary engaged"
        );
        assert_eq!(core.version, 0);
        assert_eq!(control.updates_rejected.load(Ordering::Relaxed), 1);
        assert_eq!(control.checkpoints_written.load(Ordering::Relaxed), 0);
        assert_eq!(prx.borrow().version, 0);
        assert!(
            control.status()["last_event"]
                .as_str()
                .unwrap()
                .contains("duplicate"),
            "the note records the duplicate reason"
        );
    }

    /// After a failed publish the next cohort retries: the released `boundary`
    /// lets `maybe_boundary` re-arm, and the new `Publish` (same pending version)
    /// writes and publishes successfully.
    #[tokio::test]
    async fn retry_after_failure_publishes() {
        let tmp = tempfile::tempdir().unwrap();
        let ckpt_dir = tmp.path().join("ckpt");

        let mut core = core_test();
        let control = Arc::new(LearnerControl::new());
        let (ptx, prx) = watch::channel(Arc::new(snapshot_v(0, "sha-v0")));
        let base_dir = tmp.path().to_path_buf();

        // First cohort -> publish with a valid empty dir... simulate the failure
        // by pre-creating the v1 dir (duplicate) and releasing afterwards.
        for name in ["e1", "e2"] {
            let _ = core.on_episode_start(name.into(), 0, 0);
            core.on_transition(trans(name, HOLD, {
                let mut o = [0.0; NUM_FEATURES];
                o[F_P_YES] = 0.3;
                o
            }, 1.0));
            core.on_episode_end(name, true);
        }
        assert!(matches!(
            core.maybe_boundary(1_000),
            Some(CoreEvent::BoundaryOn)
        ));
        let ev = core.maybe_update(1_000).unwrap();
        // Inject the failure as an outcome (path already covered by the async
        // test above): the boundary must be free and v0 intact.
        if let CoreEvent::Publish {
            snapshot, stats, ..
        } = ev
        {
            apply_outcome_publish(
                &mut core,
                &ptx,
                &control,
                snapshot,
                stats,
                CheckpointOutcome::Failed("transient io error".into()),
            );
        } else {
            panic!("expected Publish");
        }
        assert!(!control.boundary.load(Ordering::Relaxed));
        assert_eq!(core.version, 0);

        // Second cohort: new experience -> boundary re-arms and the publish of v1
        // (real, to disk) succeeds. Reward 2.0 != folded baseline (1.0); with a
        // null advantage the update would be DeltaZero and never reach publish.
        for name in ["e3", "e4"] {
            let _ = core.on_episode_start(name.into(), 0, 0);
            core.on_transition(trans(name, HOLD, {
                let mut o = [0.0; NUM_FEATURES];
                o[F_P_YES] = 0.3;
                o
            }, 2.0));
            core.on_episode_end(name, true);
        }
        assert!(
            matches!(core.maybe_boundary(2_000), Some(CoreEvent::BoundaryOn)),
            "the boundary must be able to re-arm after the failed publish"
        );
        let ev = core.maybe_update(2_000).unwrap();
        execute_event(&mut core, &ptx, &ckpt_dir, &control, &base_dir, ev).await;
        assert_eq!(core.version, 1, "the next cohort publishes v1");
        assert_eq!(prx.borrow().version, 1);
        assert_eq!(control.checkpoints_written.load(Ordering::Relaxed), 1);
        assert!(ckpt_dir.join("checkpoints/v000001/policy.json").exists());
    }

    // ── `pending` does not retain orphan episodes ───────────────────

    /// The boundary-timeout drain REMOVES the entries from `pending` (before, it
    /// only marked them `mixed`: if their `EpisodeEnd` was lost to a full channel,
    /// the entry with its `Vec<TransitionRec>` was orphaned forever). Entries
    /// outside `cohort_open` (episodes of other versions) are NOT touched.
    #[test]
    fn boundary_timeout_removes_pending() {
        let mut core = core_test();
        for name in ["e1", "e2"] {
            let _ = core.on_episode_start(name.into(), 0, 0);
            core.on_transition(trans(name, HOLD, [0.0; NUM_FEATURES], 0.1));
            core.on_episode_end(name, true);
        }
        // Open episode of the current cohort that does NOT close in time...
        let _ = core.on_episode_start("open".into(), 0, 0);
        core.on_transition(trans("open", HOLD, [0.0; NUM_FEATURES], 0.1));
        // ...and an episode of ANOTHER version (outside cohort_open): legitimate.
        let _ = core.on_episode_start("other-version".into(), 99, 0);

        assert!(matches!(
            core.maybe_boundary(1_000),
            Some(CoreEvent::BoundaryOn)
        ));
        let wait = core.cfg.boundary_max_wait_ms;
        let ev = core.maybe_update(1_000 + wait);
        assert!(
            ev.is_some(),
            "the boundary timeout still cuts the cohort"
        );
        assert!(
            !core.pending.contains_key("open"),
            "pending must be free of the drained episode"
        );
        assert!(core.cohort_open.is_empty());
        assert!(
            core.pending.contains_key("other-version"),
            "an episode of another version still waits for its real EpisodeEnd"
        );
        // Late messages of the drained episode: counted, not accumulated.
        assert!(!core.on_transition(trans("open", HOLD, [0.0; NUM_FEATURES], 0.1)));
        assert!(core.on_episode_end("open", true).is_none());
    }

    /// Defensive cap: `pending` never exceeds `PENDING_CAP`. The episode with the
    /// oldest `ts_ms` is evicted (and removed from `cohort_open` so the boundary
    /// does not wait for a ghost).
    #[test]
    fn cap_pending_evicts_oldest() {
        let mut core = core_test();
        for i in 0..PENDING_CAP {
            let _ = core.on_episode_start(format!("ep-{i}"), 0, i as i64);
        }
        assert_eq!(core.pending.len(), PENDING_CAP);
        // One more start -> exceeds the cap -> evicts the one with the minimum ts_ms (ep-0).
        let evicted = core.on_episode_start("ep-new".into(), 0, 1_000_000);
        assert_eq!(evicted.as_deref(), Some("ep-0"));
        assert_eq!(core.pending.len(), PENDING_CAP);
        assert!(!core.pending.contains_key("ep-0"));
        assert!(
            !core.cohort_open.contains("ep-0"),
            "the evicted episode does not hold the boundary either"
        );
    }
}
