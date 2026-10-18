"""Episodic REINFORCE trainer.

- Frozen cohorts: N episodes are collected with ONE frozen policy;
  ONE update is applied when the cohort closes; the next cohort is
  collected with the NEW policy (true on-policy — never optimizing
  REINFORCE over actions from another version as if they were on-policy).
- Reward-to-go gamma=1; baseline = mean of the returns of earlier batches;
  clip and lr are configurable and recorded.
- Master seed: each episode derives a deterministic rng from
  (seed, update_index, episode_index) — byte-for-byte reproducible.
- Invalid episodes (truncated/stale/gap/ambiguous cash) are EXCLUDED from
  the cohort with a counted reason — they never block nor get "repaired".
- Finiteness: non-finite reward/x_std/gradient abort the update
  (numeric update abort) and the policy is left intact.
- Scaler: fitted ONLY on train observations, once, before the
  first batch; never re-fitted on the fly (consistency addendum).

The delivered policy has status 'candidate' — paper_approved or
online_candidate is never fabricated here.
"""

import hashlib
import json
import random

from . import parity
from .artifact import atomic_write_json, policy_to_dict, sha256_bytes
from .contract import (
    NUM_ACTIONS,
    STATUS_CANDIDATE,
    check_mask,
    check_observation_vector,
)
from .policy import LinearPolicy


def policy_fingerprint(policy):
    """sha256 of the serialized weights+bias — version identity for the
    on-policy check and for the parent->child chain."""
    blob = json.dumps({"w": policy.weights, "b": policy.bias},
                      sort_keys=True, separators=(",", ":"))
    return sha256_bytes(blob.encode("utf-8"))


class OnPolicyViolation(RuntimeError):
    """Cohort mixes versions or does not match the live policy."""


class UpdateAborted(RuntimeError):
    """Update aborted by numerics (non-finite rewards/gradients)."""


class Episode:
    """An episode collected under ONE policy version."""

    __slots__ = ("episode_id", "policy_fingerprint", "steps", "trainable",
                 "invalid_reasons", "seed")

    def __init__(self, episode_id, fingerprint, seed):
        self.episode_id = episode_id
        self.policy_fingerprint = fingerprint
        self.seed = seed
        self.steps = []  # dicts {x_std, mask, action, behavior_prob, reward}
        self.trainable = True
        self.invalid_reasons = []

    @property
    def episode_return(self):
        return sum(s["reward"] for s in self.steps)


class ReinforceTrainer:
    """On-policy cohorts + REINFORCE update (see parity.py for the
    frozen mathematical definition)."""

    def __init__(self, policy, scaler, *, lr, grad_clip, seed,
                 cohort_episodes, gamma=1.0):
        if lr <= 0 or grad_clip <= 0:
            raise ValueError("lr and grad_clip must be > 0")
        if cohort_episodes <= 0:
            raise ValueError("cohort_episodes must be > 0")
        self.policy = policy
        self.scaler = scaler
        self.lr = float(lr)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)
        self.cohort_episodes = int(cohort_episodes)
        self.gamma = float(gamma)
        self.updates = 0
        self.completed_episodes = 0
        self.excluded_episodes = 0
        self.aborted_updates = 0
        # Baseline: returns of episodes from batches already closed.
        self._ret_sum = 0.0
        self._ret_n = 0
        self.history = []

    @property
    def baseline(self):
        """Mean return of episodes from earlier batches."""
        return self._ret_sum / self._ret_n if self._ret_n else 0.0

    def _episode_seed(self, update_index, episode_index):
        h = hashlib.sha256(
            f"{self.seed}:{update_index}:{episode_index}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    def collect_episode(self, env, episode_id, update_index, episode_index):
        """Collects a COMPLETE episode with the current policy (it must
        stay frozen for the whole cohort — the fingerprint is
        checked in update()).

        env (duck-typed):
          reset(episode_id, seed) -> {"obs_raw": [22], "mask": int|list}
          step(action_id) -> {"obs_raw": [22]|None, "mask": ...,
              "reward": float, "terminal": bool, "truncated": bool,
              "invalid_reasons": [str]}
        """
        fp = policy_fingerprint(self.policy)
        ep_seed = self._episode_seed(update_index, episode_index)
        rng = random.Random(ep_seed)
        ep = Episode(episode_id, fp, ep_seed)
        st = env.reset(episode_id, seed=ep_seed)
        obs_raw = check_observation_vector(st["obs_raw"])
        while True:
            mask = check_mask(st["mask"])
            x_std = self.scaler.transform(obs_raw)
            action, prob = self.policy.sample(x_std, mask, rng)
            rec = env.step(action)
            reward = float(rec.get("reward", 0.0))
            ep.steps.append({
                "x_std": x_std,
                # Raw observation under which the action was chosen —
                # needed by the train-only scaler fit (x_std under
                # an identity scaler loses the real range after the clamp).
                "obs_raw": list(obs_raw),
                "mask": mask,
                "action": action,
                "behavior_prob": prob,
                "reward": reward,
                # `sampled` = whether the mask under which the action was chosen
                # offered a real decision (the driver's causal flag).
                # Warmup / forced-HOLD steps keep sampled=False: their
                # reward still enters G_t but they add no gradient
                # nor count in n_steps (parity with the Rust learner).
                "sampled": bool(st.get("trainable_step", True)),
                "warmup": bool(st.get("warmup", False)),
            })
            for r in rec.get("invalid_reasons") or ():
                ep.invalid_reasons.append(r)
            if rec.get("truncated"):
                ep.trainable = False
                ep.invalid_reasons.append("truncated")
            if rec.get("invalid_reasons"):
                ep.trainable = False
            if rec.get("terminal"):
                break
            obs_raw = check_observation_vector(rec["obs_raw"])
            st = rec
        self.completed_episodes += 1
        return ep

    def update(self, cohort):
        """ONE update over the closed cohort.

        - Checks the uniform fingerprint == live policy (on-policy).
        - Excludes non-trainable episodes (counted reason).
        - baseline fixed to earlier batches before computing.
        - Numeric abort: policy left intact, counter recorded."""
        if not cohort:
            raise ValueError("update without cohort")
        fp_now = policy_fingerprint(self.policy)
        fps = {ep.policy_fingerprint for ep in cohort}
        if fps != {fp_now}:
            raise OnPolicyViolation(
                f"cohort with fingerprints {sorted(fps)} != live policy "
                f"{fp_now}: off-policy without correction")
        trainable = []
        for ep in cohort:
            if ep.trainable and not ep.invalid_reasons:
                trainable.append([dict(s) for s in ep.steps])
            else:
                self.excluded_episodes += 1
        b = self.baseline  # BEFORE folding in this cohort
        report = {
            "update_index": self.updates + 1,
            "baseline": b,
            "cohort_episodes": len(cohort),
            "trainable_episodes": len(trainable),
            "excluded": [e.episode_id for e in cohort
                         if not (e.trainable and not e.invalid_reasons)],
        }
        if not trainable:
            self.aborted_updates += 1
            report["aborted"] = "whole cohort not trainable"
            self.history.append(report)
            raise UpdateAborted(report["aborted"])
        try:
            dW, db, n_steps, ep_returns = parity.cohort_gradients(
                trainable, self.policy, b, self.gamma)
            dW, db, norm, scale = parity.clip_gradients(
                dW, db, self.grad_clip)
        except ValueError as e:
            self.aborted_updates += 1
            report["aborted"] = str(e)
            self.history.append(report)
            raise UpdateAborted(str(e)) from e
        parity.apply_gradients(self.policy, dW, db, self.lr)
        self.updates += 1
        # The cohort now feeds the baseline of the NEXT batch.
        for r in ep_returns:
            self._ret_sum += r
            self._ret_n += 1
        report.update({
            "n_steps": n_steps,
            "episode_returns": ep_returns,
            "grad_norm_preclip": norm,
            "clip_scale": scale,
            "lr": self.lr,
            "updates_total": self.updates,
        })
        self.history.append(report)
        return report

    def run_updates(self, env_factory, n_updates, episode_prefix="ep"):
        """True on-policy loop: fresh cohort per update.

        env_factory() -> new env per episode (one process per environment,
        never a shared subprocess)."""
        reports = []
        for u in range(n_updates):
            cohort = []
            for i in range(self.cohort_episodes):
                env = env_factory()
                try:
                    cohort.append(self.collect_episode(
                        env, f"{episode_prefix}-u{u}-{i}", u, i))
                finally:
                    close = getattr(env, "close", None)
                    if close:
                        close()
            reports.append(self.update(cohort))
        return reports

    # ── checkpoints / artifact ─────────────────────────────────────

    def save_checkpoint(self, path, *, predictor_sha256, run_id,
                        data_manifest_sha256, split_manifest_sha256,
                        fixture=False):
        """Atomic versioned checkpoint: policy.json candidate (or
        explicit fixture). Returns (dict, file_sha256)."""
        training = {
            "run_id": run_id,
            "data_manifest_sha256": data_manifest_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "seed": self.seed,
            "completed_episodes": self.completed_episodes,
            "updates": self.updates,
        }
        if fixture:
            training["fixture"] = True
        d = policy_to_dict(
            self.policy, self.scaler,
            predictor_sha256=predictor_sha256,
            training=training, status=STATUS_CANDIDATE)
        atomic_write_json(path, d)
        from .artifact import sha256_file
        return d, sha256_file(path)


def make_online_link(parent_sha256, child_sha256, update_index, *,
                     lr, grad_clip, seed, cohort_episodes,
                     completed_episodes, updates, ts_ms):
    """Online chain link: promoted_as
    online_candidate — never fakes paper_approved."""
    return {
        "update_index": int(update_index),
        "parent_sha256": parent_sha256,
        "child_sha256": child_sha256,
        "update_config": {
            "lr": float(lr), "grad_clip": float(grad_clip),
            "seed": int(seed), "cohort_episodes": int(cohort_episodes),
        },
        "counters": {"completed_episodes": int(completed_episodes),
                     "updates": int(updates)},
        "promoted_as": "online_candidate",
        "ts_ms": int(ts_ms),
    }
