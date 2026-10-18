//! RL predictor — online runtime of the bot.
//!
//! Implementation of the contract (authority: online amendment > spec > execution notes):
//!
//! - [`contract`]: fixed action space (9, contractual order), observation
//!   vector `rl_predictor.features.v1` (22 features) and explicit mask.
//! - [`policy`]: linear softmax policy with INDEPENDENT weights per action
//!   (`logits[a] = bias[a] + w[a]·x_std`), deterministic legal argmax and
//!   sampling with a recorded seed for adaptive PAPER mode.
//! - [`artifact`]: fail-closed loader for the approved (`paper_approved`)
//!   `policy.json`/`manifest.json` and for the online checkpoint manifest.
//! - [`episode`]: virtual per-episode ledger in micros and the accounting
//!   reward from the spec (equity delta net of costs; deposits/topups never count).
//! - [`learner`]: dedicated learning worker (episodic REINFORCE,
//!   frozen-policy cohorts, versioned snapshots, atomic
//!   checkpoints, rollback, freeze).
//!
//! The operable strategy lives in `crate::strategy::rl_predictor`.

pub mod artifact;
pub mod contract;
pub mod episode;
pub mod learner;
/// Observation/mask/action translation for `policy_v1`, shared by
/// the runtime strategy and the offline driver (`bin/rl_policy_env`).
pub mod obs;
pub mod policy;
pub mod rng;
