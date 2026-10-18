"""Train-only scaler and masked linear softmax policy."""

import math
import random
import unittest

from rl_predictor.contract import NUM_ACTIONS, NUM_FEATURES
from rl_predictor.policy import LinearPolicy, PolicyError
from rl_predictor.scaler import CLAMP_STD, FeatureScaler, ScalerError


def obs(v=0.5):
    return [v] * NUM_FEATURES


class TestScaler(unittest.TestCase):
    def test_fit_transform(self):
        sc = FeatureScaler.fit([obs(0.0), obs(2.0)])
        self.assertEqual(sc.mean, [1.0] * NUM_FEATURES)
        self.assertEqual(sc.transform(obs(1.0)), [0.0] * NUM_FEATURES)
        self.assertEqual(sc.transform(obs(2.0)), [1.0] * NUM_FEATURES)

    def test_constant_feature_scale_positive(self):
        sc = FeatureScaler.fit([obs(3.0)] * 4)
        self.assertTrue(all(s > 0 for s in sc.scale))
        self.assertEqual(sc.transform(obs(3.0)), [0.0] * NUM_FEATURES)

    def test_clamp(self):
        sc = FeatureScaler.fit([obs(0.0), obs(2.0)])
        self.assertEqual(sc.transform(obs(1000.0)), [CLAMP_STD] *
                         NUM_FEATURES)
        self.assertEqual(sc.transform(obs(-1000.0)), [-CLAMP_STD] *
                         NUM_FEATURES)

    def test_transform_does_not_mutate(self):
        sc = FeatureScaler.fit([obs(0.0), obs(4.0)])
        m0, s0 = list(sc.mean), list(sc.scale)
        sc.transform(obs(9.9))
        self.assertEqual((sc.mean, sc.scale), (m0, s0))

    def test_rejects(self):
        with self.assertRaises(ScalerError):
            FeatureScaler.fit([])
        with self.assertRaises(ScalerError):
            FeatureScaler([0.0] * NUM_FEATURES, [0.0] * NUM_FEATURES)
        with self.assertRaises(ScalerError):
            FeatureScaler([0.0] * NUM_FEATURES, [-1.0] * NUM_FEATURES)
        sc = FeatureScaler.fit([obs(0.0), obs(1.0)])
        with self.assertRaises(Exception):
            sc.transform([float("nan")] * NUM_FEATURES)

    def test_roundtrip(self):
        sc = FeatureScaler.fit([obs(1.0), obs(3.0)])
        sc2 = FeatureScaler.from_dict(sc.to_dict())
        self.assertEqual(sc.mean, sc2.mean)
        self.assertEqual(sc.scale, sc2.scale)


class TestPolicy(unittest.TestCase):
    def test_zero_weights_uniform_over_legal(self):
        p = LinearPolicy()
        mask = 0b000000111  # HOLD + 2 BUY
        pr = p.probs(obs(), mask)
        self.assertAlmostEqual(sum(pr), 1.0)
        for a in range(3):
            self.assertAlmostEqual(pr[a], 1.0 / 3.0)
        for a in range(3, NUM_ACTIONS):
            self.assertEqual(pr[a], 0.0)  # illegal exactly 0

    def test_argmax_tie_prefers_hold(self):
        p = LinearPolicy()
        self.assertEqual(p.argmax(obs(), (1 << NUM_ACTIONS) - 1), 0)

    def test_argmax_legal_only(self):
        # Force SELL_NO_ALL as the winner but illegal -> falls to the next legal action.
        w = [[0.0] * NUM_FEATURES for _ in range(NUM_ACTIONS)]
        w[8][0] = 10.0
        w[1][0] = 5.0
        p = LinearPolicy(w, [0.0] * NUM_ACTIONS)
        mask = 0b000000011  # only HOLD and BUY_YES_SMALL
        self.assertEqual(p.argmax(obs(1.0), mask), 1)

    def test_sample_reproducible_and_only_legal(self):
        p = LinearPolicy()
        mask = 0b000000101
        acts1 = [p.sample(obs(), mask, random.Random(42))[0]
                 for _ in range(50)]
        acts2 = [p.sample(obs(), mask, random.Random(42))[0]
                 for _ in range(50)]
        self.assertEqual(acts1, acts2)
        self.assertTrue(all(a in (0, 2) for a in acts1))
        # Recorded probability consistent with probs.
        _, prob = p.sample(obs(), mask, random.Random(1))
        self.assertTrue(0.0 < prob <= 1.0)

    def test_log_prob_rejects_illegal(self):
        p = LinearPolicy()
        with self.assertRaises(PolicyError):
            p.log_prob(5, obs(), 0b000000001)
        lp = p.log_prob(0, obs(), 0b000000001)
        self.assertAlmostEqual(lp, 0.0)

    def test_numerical_stability_extreme_logits(self):
        w = [[0.0] * NUM_FEATURES for _ in range(NUM_ACTIONS)]
        w[1][0] = 1e6  # huge logits: without subtracting the max -> overflow
        p = LinearPolicy(w, [0.0] * NUM_ACTIONS)
        pr = p.probs(obs(1.0), 0b000000011)
        self.assertTrue(all(math.isfinite(x) for x in pr))
        self.assertAlmostEqual(sum(pr), 1.0)
        self.assertEqual(p.argmax(obs(1.0), 0b000000011), 1)

    def test_apply_update_and_finitude_guard(self):
        p = LinearPolicy()
        dW = [[0.01] * NUM_FEATURES for _ in range(NUM_ACTIONS)]
        db = [0.01] * NUM_ACTIONS
        p.apply_update(dW, db, 0.5)
        self.assertAlmostEqual(p.weights[3][7], 0.005)
        dW_bad = [[float("nan")] * NUM_FEATURES for _ in range(NUM_ACTIONS)]
        with self.assertRaises(PolicyError):
            p.apply_update(dW_bad, db, 0.5)

    def test_params_validation(self):
        with self.assertRaises(PolicyError):
            LinearPolicy([[0.0] * 5] * NUM_ACTIONS, [0.0] * NUM_ACTIONS)
        with self.assertRaises(PolicyError):
            LinearPolicy(None, [float("inf")] * NUM_ACTIONS)


if __name__ == "__main__":
    unittest.main()
