"""REINFORCE: frozen cohorts, on-policy, baseline from previous batches,
return-to-go gamma=1, clip, seeds, numeric aborts."""

import random
import unittest

from rl_predictor import parity
from rl_predictor.contract import NUM_ACTIONS, NUM_FEATURES
from rl_predictor.policy import LinearPolicy
from rl_predictor.scaler import FeatureScaler
from rl_predictor.trainer import (
    Episode,
    OnPolicyViolation,
    ReinforceTrainer,
    UpdateAborted,
    policy_fingerprint,
)

MASK_ALL = (1 << NUM_ACTIONS) - 1


def flat_obs(v):
    return [v] * NUM_FEATURES


class SyntheticEnv:
    """TEST-only env (synthetic fixture; never trains the delivery
    policy). Emits a fixed obs_raw and a reward per table."""

    instances = 0

    def __init__(self, rewards=None, steps=2, obs=0.3, mask=MASK_ALL,
                 truncated=False, invalid=None):
        self.rewards = rewards or {0: 0.01}
        self.max_steps = steps
        self.obs = obs
        self.mask = mask
        self.truncated = truncated
        self.invalid = list(invalid or [])
        self.t = 0
        self.closed = False
        SyntheticEnv.instances += 1

    def reset(self, episode_id, seed):
        self.t = 0
        return {"obs_raw": flat_obs(self.obs), "mask": self.mask}

    def step(self, action):
        self.t += 1
        terminal = self.t >= self.max_steps
        last = terminal and self.truncated
        return {
            "obs_raw": None if terminal else flat_obs(self.obs),
            "mask": self.mask,
            "reward": self.rewards.get(action, self.rewards.get("*", 0.0)),
            "terminal": terminal,
            "truncated": last,
            "invalid_reasons": self.invalid if self.t == 1 else [],
        }

    def close(self):
        self.closed = True


def make_trainer(seed=5, lr=0.1, clip=10.0, cohort=2):
    sc = FeatureScaler([0.0] * NUM_FEATURES, [1.0] * NUM_FEATURES)
    return ReinforceTrainer(LinearPolicy(), sc, lr=lr, grad_clip=clip,
                            seed=seed, cohort_episodes=cohort)


def cohort_from_steps(steps_list, fp):
    out = []
    for i, steps in enumerate(steps_list):
        e = Episode(f"ep{i}", fp, seed=0)
        e.steps = steps
        out.append(e)
    return out


def steps_of(actions, rewards, obs=0.5):
    return [{"x_std": flat_obs(obs), "mask": MASK_ALL, "action": a,
             "behavior_prob": 1.0 / NUM_ACTIONS, "reward": r}
            for a, r in zip(actions, rewards)]


class TestReturnsToGo(unittest.TestCase):
    def test_gamma1(self):
        self.assertEqual(parity.returns_to_go([1.0, 2.0, 3.0]),
                         [6.0, 5.0, 3.0])

    def test_gamma_other(self):
        g = parity.returns_to_go([1.0, 1.0, 1.0], gamma=0.5)
        self.assertEqual(g, [1.75, 1.5, 1.0])


class TestGradientMath(unittest.TestCase):
    def test_single_step_exact(self):
        """One step, action 2, advantage A: dW[j][f] = A(δj2 − πj)x_f."""
        pol = LinearPolicy()  # uniform: π=1/9
        cohort = [[{"x_std": flat_obs(1.0), "mask": MASK_ALL,
                    "action": 2, "behavior_prob": 1.0 / 9,
                    "reward": 0.9}]]
        dW, db, n, rets = parity.cohort_gradients(cohort, pol, 0.0)
        self.assertEqual(n, 1)
        self.assertEqual(rets, [0.9])
        for j in range(NUM_ACTIONS):
            expect = 0.9 * ((1.0 if j == 2 else 0.0) - 1.0 / 9)
            self.assertAlmostEqual(db[j], expect)
            self.assertAlmostEqual(dW[j][0], expect)

    def test_illegal_actions_no_gradient(self):
        mask = 0b000000011  # only 0,1
        cohort = [[{"x_std": flat_obs(1.0), "mask": mask, "action": 1,
                    "behavior_prob": 0.5, "reward": 1.0}]]
        dW, db, _, _ = parity.cohort_gradients(
            cohort, LinearPolicy(), 0.0)
        for j in range(2, NUM_ACTIONS):
            self.assertEqual(db[j], 0.0)
            self.assertTrue(all(v == 0.0 for v in dW[j]))

    def test_action_prob_zero_rejected(self):
        cohort = [[{"x_std": flat_obs(0.1), "mask": MASK_ALL,
                    "action": 7, "behavior_prob": 1e-9,
                    "reward": 1.0}]]
        # The mask forbids 7 -> π=0 -> off-policy/corrupt data.
        mask = MASK_ALL & ~(1 << 7)
        cohort[0][0]["mask"] = mask
        with self.assertRaises(ValueError):
            parity.cohort_gradients(cohort, LinearPolicy(), 0.0)

    def test_clip_global_norm(self):
        cohort = [[{"x_std": flat_obs(100.0), "mask": MASK_ALL,
                    "action": 1, "behavior_prob": 1 / 9,
                    "reward": 100.0}]]
        dW, db, _, _ = parity.cohort_gradients(
            cohort, LinearPolicy(), 0.0)
        dW, db, norm, scale = parity.clip_gradients(dW, db, 1.0)
        self.assertGreater(norm, 1.0)
        self.assertLess(scale, 1.0)
        import math
        post = math.sqrt(sum(v * v for r in dW for v in r)
                         + sum(v * v for v in db))
        self.assertAlmostEqual(post, 1.0)


class TestTrainer(unittest.TestCase):
    def test_weights_change_after_update(self):
        tr = make_trainer()
        w0 = [list(r) for r in tr.policy.weights]
        envs = lambda: SyntheticEnv(rewards={0: 0.5}, steps=3)
        tr.run_updates(envs, 1)
        self.assertNotEqual(tr.policy.weights, w0)
        self.assertEqual(tr.updates, 1)

    def test_on_policy_fingerprint_guard(self):
        tr = make_trainer()
        fp = policy_fingerprint(tr.policy)
        cohort = cohort_from_steps(
            [steps_of([0, 1], [0.1, 0.2])], "other_version")
        with self.assertRaises(OnPolicyViolation):
            tr.update(cohort)
        cohort = cohort_from_steps(
            [steps_of([0, 1], [0.1, 0.2])], fp)
        tr.update(cohort)  # coincide: ok

    def test_baseline_previous_batches_not_current(self):
        """If the baseline were the mean of the current batch, a cohort with
        uniform returns would give advantage 0 -> no gradient. With the
        baseline taken from PREVIOUS batches (0 for the first one), there is one."""
        tr = make_trainer()
        fp = policy_fingerprint(tr.policy)
        cohort1 = cohort_from_steps(
            [steps_of([0], [1.0]), steps_of([0], [1.0])], fp)
        rep1 = tr.update(cohort1)
        self.assertEqual(rep1["baseline"], 0.0)
        self.assertNotEqual(tr.policy.weights,
                            [[0.0] * NUM_FEATURES] * NUM_ACTIONS)
        # Second batch: baseline = mean of the FIRST batch's returns.
        fp2 = policy_fingerprint(tr.policy)
        cohort2 = cohort_from_steps(
            [steps_of([0], [1.0]), steps_of([0], [1.0])], fp2)
        rep2 = tr.update(cohort2)
        self.assertEqual(rep2["baseline"], 1.0)  # prev batch mean
        # Advantage = 1.0 - 1.0 = 0 -> no weight change in this update.
        self.assertEqual(rep2["episode_returns"], [1.0, 1.0])

    def test_fresh_episodes_each_update(self):
        SyntheticEnv.instances = 0
        tr = make_trainer(cohort=3)
        tr.run_updates(lambda: SyntheticEnv(steps=2), 2)
        # 2 updates × 3 episodes, each with a NEW env.
        self.assertEqual(SyntheticEnv.instances, 6)
        self.assertEqual(tr.completed_episodes, 6)

    def test_determinism_same_seed(self):
        def run(seed):
            tr = make_trainer(seed=seed)
            tr.run_updates(lambda: SyntheticEnv(
                rewards={0: 0.3, 1: 0.1, "*": -0.02}, steps=4), 2)
            return tr.policy.weights, tr.policy.bias
        self.assertEqual(run(11), run(11))

    def test_truncated_excluded_from_learning(self):
        tr = make_trainer(cohort=2)
        calls = []
        def factory():
            e = SyntheticEnv(steps=2, truncated=True)
            return e
        with self.assertRaises(UpdateAborted):
            tr.run_updates(factory, 1)
        self.assertEqual(tr.excluded_episodes, 2)
        self.assertEqual(tr.aborted_updates, 1)
        # Policy unchanged after the abort.
        self.assertEqual(tr.policy.weights,
                         [[0.0] * NUM_FEATURES] * NUM_ACTIONS)

    def test_invalid_episode_partial_cohort(self):
        tr = make_trainer(cohort=2)
        envs = iter([SyntheticEnv(steps=2, invalid=["book_stale"]),
                     SyntheticEnv(steps=2, rewards={0: 0.4})])
        tr.run_updates(lambda: next(envs), 1)
        self.assertEqual(tr.excluded_episodes, 1)
        rep = tr.history[-1]
        self.assertEqual(rep["trainable_episodes"], 1)

    def test_nonfinite_reward_aborts(self):
        tr = make_trainer()
        fp = policy_fingerprint(tr.policy)
        bad = steps_of([0], [float("nan")])
        cohort = cohort_from_steps([bad, steps_of([0], [0.1])], fp)
        with self.assertRaises(UpdateAborted):
            tr.update(cohort)
        self.assertEqual(tr.updates, 0)
        self.assertEqual(tr.aborted_updates, 1)

    def test_checkpoint_candidate_atomic(self):
        import os
        import tempfile
        tr = make_trainer()
        tr.run_updates(lambda: SyntheticEnv(steps=2), 1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "policy.json")
            doc, sha = tr.save_checkpoint(
                path, predictor_sha256="a" * 64, run_id="t",
                data_manifest_sha256="b" * 64,
                split_manifest_sha256="c" * 64)
            from rl_predictor.artifact import load_policy, sha256_file
            self.assertEqual(sha, sha256_file(path))
            p, _, _ = load_policy(path)
            self.assertEqual(p["status"], "candidate")
            self.assertEqual(p["training"]["updates"], 1)


class TestCollectEpisode(unittest.TestCase):
    def test_records_behavior_prob_and_mask(self):
        tr = make_trainer()
        ep = tr.collect_episode(
            SyntheticEnv(steps=3, rewards={0: 0.1}), "e1", 0, 0)
        self.assertEqual(len(ep.steps), 3)
        for s in ep.steps:
            self.assertAlmostEqual(s["behavior_prob"], 1.0 / NUM_ACTIONS)
            self.assertEqual(s["mask"], MASK_ALL)
            self.assertTrue(0 <= s["action"] < NUM_ACTIONS)


if __name__ == "__main__":
    unittest.main()
