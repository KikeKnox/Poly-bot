//! Linear per-action softmax policy and its inference on the hot path: no
//! locks, no I/O, no allocation per decision.
//!
//! `logits[a] = bias[a] + w[a]·x_std`, where `x_std` is the standardized
//! observation (`clamp((x-mean)/scale, -10, 10)`). The mask is applied
//! BEFORE the softmax (illegal actions get no probability) and the
//! maximum legal logit is subtracted for numerical stability.
//!
//! INDEPENDENT weights per action — `weights[action][feature]`. A state block
//! shared across actions cancels under the softmax and yields zero gradient
//! on that state (a classic softmax-policy mistake). Here each action has its
//! own weight vector over the state.

use std::sync::Arc;

use crate::rl::contract::{ActionMask, NUM_ACTIONS, NUM_FEATURES, HOLD};
use crate::rl::rng::SplitMix64;

/// Standardization bound: `clamp(x_std, -10, 10)`.
pub const STD_CLAMP: f64 = 10.0;

/// Frozen scaler of the policy (`mean`/`scale` per feature, `scale>0`).
/// Normalization is NOT updated online in v1.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PolicyScaler {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

/// Policy weights: `weights[action][feature]` + `bias[action]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSoftmaxPolicy {
    pub scaler: PolicyScaler,
    /// `[action][feature]` — exact dimension NUM_ACTIONS × NUM_FEATURES.
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl LinearSoftmaxPolicy {
    /// Validates dimensions and finiteness (the artifact load does it earlier;
    /// here it is an extra defense for snapshots built by the learner).
    pub fn is_valid(&self) -> bool {
        self.weights.len() == NUM_ACTIONS
            && self.weights.iter().all(|w| w.len() == NUM_FEATURES)
            && self.bias.len() == NUM_ACTIONS
            && self.scaler.mean.len() == NUM_FEATURES
            && self.scaler.scale.len() == NUM_FEATURES
            && self
                .weights
                .iter()
                .flatten()
                .chain(self.bias.iter())
                .all(|v| v.is_finite())
    }

    /// Standardizes the raw observation. `None` if any feature is
    /// NaN/±inf — the whole observation is invalidated, it is not imputed.
    pub fn standardize(&self, obs: &[f64; NUM_FEATURES]) -> Option<[f64; NUM_FEATURES]> {
        let mut out = [0.0f64; NUM_FEATURES];
        for i in 0..NUM_FEATURES {
            let x = obs[i];
            if !x.is_finite() {
                return None;
            }
            out[i] = ((x - self.scaler.mean[i]) / self.scaler.scale[i])
                .clamp(-STD_CLAMP, STD_CLAMP);
        }
        Some(out)
    }

    /// Masked logits: `-inf` for illegal actions. `None` if the
    /// observation is invalid.
    pub fn logits_masked(
        &self,
        obs: &[f64; NUM_FEATURES],
        mask: &ActionMask,
    ) -> Option<[f64; NUM_ACTIONS]> {
        let x = self.standardize(obs)?;
        let mut logits = [f64::NEG_INFINITY; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            if mask[a] {
                logits[a] = self.bias[a]
                    + self.weights[a]
                        .iter()
                        .zip(x.iter())
                        .map(|(w, f)| w * f)
                        .sum::<f64>();
            }
        }
        Some(logits)
    }

    /// Softmax probabilities over legal actions (0.0 for illegal ones).
    /// `None` if the observation is invalid or no action is legal.
    pub fn action_probs(
        &self,
        obs: &[f64; NUM_FEATURES],
        mask: &ActionMask,
    ) -> Option<[f64; NUM_ACTIONS]> {
        let logits = self.logits_masked(obs, mask)?;
        let max = logits
            .iter()
            .copied()
            .filter(|l| l.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            return None; // empty mask — should not happen (HOLD is always legal)
        }
        let mut probs = [0.0f64; NUM_ACTIONS];
        let mut sum = 0.0;
        for a in 0..NUM_ACTIONS {
            if mask[a] && logits[a].is_finite() {
                probs[a] = (logits[a] - max).exp();
                sum += probs[a];
            }
        }
        if sum <= 0.0 || !sum.is_finite() {
            return None;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        Some(probs)
    }

    /// Deterministic legal argmax — evaluation mode. Ties are resolved by
    /// ascending index (HOLD, index 0, wins any tie).
    /// `None` if the observation is invalid.
    pub fn argmax_legal(
        &self,
        obs: &[f64; NUM_FEATURES],
        mask: &ActionMask,
    ) -> Option<(usize, f64)> {
        let probs = self.action_probs(obs, mask)?;
        let mut best = HOLD;
        let mut best_p = -1.0f64;
        for a in 0..NUM_ACTIONS {
            if probs[a] > best_p {
                best = a;
                best_p = probs[a];
            }
        }
        Some((best, best_p))
    }

    /// Samples a legal action with `rng`. Returns (action, prob of the
    /// action under the behavior policy, vector of probs) — the recorded prob is
    /// the `behavior_prob` that travels to the cohort.
    /// `None` if the observation is invalid.
    pub fn sample_legal(
        &self,
        obs: &[f64; NUM_FEATURES],
        mask: &ActionMask,
        rng: &mut SplitMix64,
    ) -> Option<(usize, f64, [f64; NUM_ACTIONS])> {
        let probs = self.action_probs(obs, mask)?;
        let x = rng.next_f64();
        let mut acc = 0.0;
        let mut chosen = NUM_ACTIONS - 1;
        for a in 0..NUM_ACTIONS {
            acc += probs[a];
            if x < acc {
                chosen = a;
                break;
            }
        }
        // Numerical edge: acc may end up < x due to rounding and falls to the last
        // index; if that one were illegal (p=0), fall back to the last legal one.
        if probs[chosen] <= 0.0 {
            for a in (0..NUM_ACTIONS).rev() {
                if probs[a] > 0.0 {
                    chosen = a;
                    break;
                }
            }
        }
        Some((chosen, probs[chosen], probs))
    }
}

/// Immutable policy snapshot published by the learner. The hot path reads it
/// through `watch`/`Arc`. It is NEVER mutated after publication (policy, hash and
/// meta travel together; a new version is another `Arc`).
#[derive(Debug, Clone)]
pub struct PolicySnapshot {
    /// Monotonic published version (0 = approved base artifact).
    pub version: u64,
    /// Number of the online update that produced it (0 = base).
    pub update_index: u64,
    pub policy: Arc<LinearSoftmaxPolicy>,
    /// sha256 of the `policy.json` bytes that produced these weights.
    pub policy_sha256: String,
    /// sha256 of the parent policy (online chain → approved base).
    pub parent_sha256: String,
    /// sha256 of the approved base artifact (root of the chain).
    pub base_sha256: String,
    /// ts_ms of publication.
    pub published_ts_ms: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::contract::*;

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

    #[test]
    fn probs_sum_to_one_only_over_legal() {
        let p = uniform_policy();
        let obs = [0.0f64; NUM_FEATURES];
        let mut mask = MASK_HOLD_ONLY;
        mask[BUY_YES_FULL] = true;
        mask[SELL_NO_ALL] = true;
        let probs = p.action_probs(&obs, &mask).unwrap();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // Illegal ones are strictly 0, including the masked buys.
        assert_eq!(probs[BUY_YES_SMALL], 0.0);
        assert_eq!(probs[BUY_NO_FULL], 0.0);
        assert_eq!(probs[SELL_YES_HALF], 0.0);
        assert!((probs[HOLD] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn argmax_deterministic_and_tie_goes_to_hold() {
        let p = uniform_policy();
        let obs = [0.0f64; NUM_FEATURES];
        let mut mask = MASK_HOLD_ONLY;
        mask[BUY_YES_FULL] = true;
        let (a, _) = p.argmax_legal(&obs, &mask).unwrap();
        assert_eq!(a, HOLD, "a full tie must resolve to HOLD (index 0)");
    }

    /// With independent weights per action, two distinct states MUST give
    /// different probabilities if any action weights that feature.
    #[test]
    fn different_states_change_probabilities() {
        let mut p = uniform_policy();
        p.weights[BUY_YES_FULL][F_P_YES] = 2.0;
        let mut mask = MASK_HOLD_ONLY;
        mask[BUY_YES_FULL] = true;
        let obs_a = {
            let mut o = [0.0f64; NUM_FEATURES];
            o[F_P_YES] = 0.1;
            o
        };
        let obs_b = {
            let mut o = [0.0f64; NUM_FEATURES];
            o[F_P_YES] = 0.9;
            o
        };
        let pa = p.action_probs(&obs_a, &mask).unwrap()[BUY_YES_FULL];
        let pb = p.action_probs(&obs_b, &mask).unwrap()[BUY_YES_FULL];
        assert!(pb > pa, "P(BUY) must grow with p_yes: {pa} -> {pb}");
    }

    /// An action's identity does NOT depend on which other actions are legal: the
    /// logit of BUY_YES_FULL is the same with mask {HOLD,BUY} as with mask
    /// {HOLD,BUY,SELL_NO_ALL} (same obs and weights).
    #[test]
    fn mask_subset_does_not_change_identity() {
        let mut p = uniform_policy();
        p.weights[BUY_YES_FULL][F_P_YES] = 1.5;
        p.weights[SELL_NO_ALL][F_NO_BID] = 0.7;
        let obs = {
            let mut o = [0.0f64; NUM_FEATURES];
            o[F_P_YES] = 0.8;
            o[F_NO_BID] = 0.4;
            o
        };
        let mut m1 = MASK_HOLD_ONLY;
        m1[BUY_YES_FULL] = true;
        let mut m2 = m1;
        m2[SELL_NO_ALL] = true;
        let l1 = p.logits_masked(&obs, &m1).unwrap()[BUY_YES_FULL];
        let l2 = p.logits_masked(&obs, &m2).unwrap()[BUY_YES_FULL];
        assert_eq!(l1, l2);
        assert_eq!(
            p.logits_masked(&obs, &m2).unwrap()[SELL_NO_ALL],
            0.7 * 0.4
        );
    }

    #[test]
    fn obs_nan_invalidates() {
        let p = uniform_policy();
        let mut obs = [0.0f64; NUM_FEATURES];
        obs[F_P_YES] = f64::NAN;
        assert!(p.action_probs(&obs, &MASK_HOLD_ONLY).is_none());
        assert!(p.argmax_legal(&obs, &MASK_HOLD_ONLY).is_none());
    }

    #[test]
    fn sample_respects_mask_and_is_deterministic() {
        let mut p = uniform_policy();
        p.weights[BUY_YES_FULL][F_P_YES] = 5.0;
        let obs = {
            let mut o = [0.0f64; NUM_FEATURES];
            o[F_P_YES] = 0.9;
            o
        };
        let mut mask = MASK_HOLD_ONLY;
        mask[BUY_YES_FULL] = true;
        let mut r1 = SplitMix64::new(1234);
        let mut r2 = SplitMix64::new(1234);
        let mut violations = 0;
        for _ in 0..200 {
            let (a, prob, _) = p.sample_legal(&obs, &mask, &mut r1).unwrap();
            assert!(mask[a]);
            assert!(prob > 0.0);
            let (b, _, _) = p.sample_legal(&obs, &mask, &mut r2).unwrap();
            assert_eq!(a, b, "same seed must give the same action");
            let _ = &mut violations;
        }
    }
}
