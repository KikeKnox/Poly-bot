"""Pure tests of the class-block softmax policy of
`run_env_episodes.py` (fix for the original defect, "RL policy").

No IPC, no subprocess and no `hrf_research_env` binary: all the
episodes/observations in this suite are SYNTHETIC test FIXTURES. They are
NOT evidence about the market or about learning; they only verify the
policy mechanics (sensitivity to state, YES/NO identity and non-zero
gradient).

Covers:
- State sensitivity: same set of legal actions, two different
  observations → different probabilities (with non-zero weights). In the
  earlier version the state cancelled out in the softmax and the
  probabilities were identical.
- YES/NO identity: two Buys with the same price/size but token_ids that
  map to different sides (explicit metadata mapping) → different kinds and
  logits.
- Non-zero gradient: synthetic episode with return ≠ 0 → Δw ≠ 0 in the
  state blocks.
- Side derivation from `market_meta` metadata of the tape
  (explicit yes_token_id/no_token_id) — NEVER from the order of
  `legal_actions`, which may omit a token — and class `*_unknown`
  when the token has no meta.
"""

import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_env_episodes as pol  # noqa: E402


def _obs(yes_mid=0.5, **extra):
    """Minimal synthetic observation (test fixture, not market data)."""
    feats = {"yes_mid": yes_mid,
             "yes_spread": 0.02,
             "yes_imbalance": 0.1,
             "time_left_frac": 0.5,
             "spot_ret_1": 0.001,
             "flow_imbalance": 0.0,
             "own_open_orders": 0.0}
    feats.update(extra)
    return {"schema_version": 1, "market_cid": "cid-sintetico",
            "features": feats, "inventory": {}}


def _legal():
    """Synthetic legal actions in the order the driver emits them:
    Hold, then per market [Buy YES, Sell YES?, Buy NO, Sell NO?],
    then Cancels."""
    return [
        "hold",
        {"buy": {"token_id": "tok-yes", "price": "0.55", "size": "10"}},
        {"buy": {"token_id": "tok-no", "price": "0.45", "size": "10"}},
        {"sell": {"token_id": "tok-yes", "price": "0.54", "size": "5"}},
        {"sell": {"token_id": "tok-no", "price": "0.44", "size": "5"}},
        {"cancel": {"order_id": "ord-1"}},
    ]


# Explicit token→side map of the fixture (equivalent to read market_meta).
SIDES = {"tok-yes": "yes", "tok-no": "no"}


def _meta(cid, yes_tok, no_tok):
    """Synthetic market_meta event (test fixture)."""
    return {"schema_version": 1, "source": "polymarket_rest",
            "kind": "market_meta", "market_cid": cid, "token_id": None,
            "asset": "eth",
            "payload": {"market_cid": cid, "yes_token_id": yes_tok,
                        "no_token_id": no_tok}}


def _tape(events):
    """Temporary tape with the given events; the caller does
    `addCleanup(d.cleanup)` to keep it alive for the test."""
    d = tempfile.TemporaryDirectory()
    p = Path(d.name) / "tape.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events))
    return d, str(p)


def _probs(w, obs, legal, sides=None):
    return pol.softmax(pol.logits(w, obs, legal, sides or SIDES))


class MetaSidesMapTest(unittest.TestCase):
    """YES/NO derivation from `market_meta` metadata — fix for the
    original defect (inference from the order of legal_actions)."""

    def test_map_from_market_meta(self):
        d, tape = _tape([_meta("m1", "tok-yes", "tok-no")])
        self.addCleanup(d.cleanup)
        sides, conflicts = pol.sides_map_from_meta(tape)
        self.assertEqual(sides, {"tok-yes": "yes", "tok-no": "no"})
        self.assertEqual(conflicts, [])

    def test_only_no_legal_is_not_buy_yes(self):
        # Regression: YES without an ask and without inventory → only NO has
        # a legal action. Inference from order used to classify it as buy_yes.
        legal = ["hold",
                 {"buy": {"token_id": "tok-no", "price": "0.5",
                          "size": "1"}}]
        self.assertEqual(
            pol.action_kind(legal[1], SIDES), "buy_no")
        lg = pol.logits(pol.initial_weights(), _obs(), legal, SIDES)
        self.assertEqual(len(lg), 2)

    def test_only_yes_legal_classifies_buy_yes(self):
        legal = ["hold",
                 {"buy": {"token_id": "tok-yes", "price": "0.5",
                          "size": "1"}}]
        self.assertEqual(pol.action_kind(legal[1], SIDES), "buy_yes")

    def test_reorder_does_not_change_identity(self):
        # The order of legal_actions carries no meaning: NO first.
        legal = [{"buy": {"token_id": "tok-no", "price": "0.5",
                          "size": "1"}},
                 {"buy": {"token_id": "tok-yes", "price": "0.5",
                          "size": "1"}}]
        self.assertEqual(
            [pol.action_kind(a, SIDES) for a in legal],
            ["buy_no", "buy_yes"])

    def test_multi_market(self):
        d, tape = _tape([_meta("m1", "m1y", "m1n"),
                         _meta("m2", "m2y", "m2n")])
        self.addCleanup(d.cleanup)
        sides, _ = pol.sides_map_from_meta(tape)
        self.assertEqual(
            [sides[t] for t in ("m1y", "m1n", "m2y", "m2n")],
            ["yes", "no", "yes", "no"])
        legal = [{"buy": {"token_id": "m2n", "price": "0.5",
                          "size": "1"}},
                 {"buy": {"token_id": "m1y", "price": "0.5",
                          "size": "1"}}]
        self.assertEqual(
            [pol.action_kind(a, sides) for a in legal],
            ["buy_no", "buy_yes"])

    def test_persistent_identity_across_steps(self):
        # The same mapping is used for all the steps: if step t only
        # has an action over NO, it is not reclassified.
        legal_t1 = ["hold",
                    {"buy": {"token_id": "tok-yes", "price": "0.5",
                             "size": "1"}},
                    {"buy": {"token_id": "tok-no", "price": "0.5",
                             "size": "1"}}]
        legal_t2 = ["hold",
                    {"sell": {"token_id": "tok-no", "price": "0.5",
                              "size": "1"}}]
        self.assertEqual(pol.action_kind(legal_t1[2], SIDES), "buy_no")
        self.assertEqual(pol.action_kind(legal_t2[1], SIDES), "sell_no")

    def test_contradictory_meta_falls_to_unknown(self):
        # Two metas assign the same token to different sides → no
        # reliable identity, it is left out of the mapping.
        d, tape = _tape([_meta("m1", "tok-x", "tok-y"),
                         _meta("m2", "tok-y", "tok-x")])
        self.addCleanup(d.cleanup)
        sides, conflicts = pol.sides_map_from_meta(tape)
        self.assertNotIn("tok-x", sides)
        self.assertNotIn("tok-y", sides)
        self.assertEqual(sorted(conflicts), ["tok-x", "tok-y"])
        a = {"buy": {"token_id": "tok-x", "price": "0.5", "size": "1"}}
        self.assertEqual(pol.action_kind(a, sides), "buy_unknown")

    def test_unknown_class_without_meta(self):
        a = {"buy": {"token_id": "phantom", "price": "0.5", "size": "1"}}
        self.assertEqual(pol.action_kind(a, SIDES), "buy_unknown")
        s = {"sell": {"token_id": "phantom", "price": "0.5", "size": "1"}}
        self.assertEqual(pol.action_kind(s, SIDES), "sell_unknown")

    def test_basic_classes(self):
        self.assertEqual(pol.action_kind("hold", SIDES), "hold")
        self.assertEqual(
            pol.action_kind({"cancel": {"order_id": "x"}}, SIDES),
            "cancel")
        self.assertEqual(pol.action_kind(_legal()[1], SIDES), "buy_yes")
        self.assertEqual(pol.action_kind(_legal()[2], SIDES), "buy_no")


class StateSensitivityTest(unittest.TestCase):
    """The original defect: the state must NOT cancel out in the softmax."""

    def test_identical_probs_with_zero_weights(self):
        # Starting point: zero weights → uniform (documents that the initial
        # sampling is exploratory, not a defect).
        w = pol.initial_weights()
        legal = _legal()
        self.assertEqual(_probs(w, _obs(yes_mid=0.1), legal),
                         _probs(w, _obs(yes_mid=0.9), legal))

    def test_probs_differ_after_update(self):
        # After a real update with a return, the same set of
        # actions yields different probabilities in two states.
        w = pol.initial_weights()
        legal = _legal()
        rng = random.Random(3)
        obs = _obs(yes_mid=0.5)
        i, _a, probs, cache = pol.choose(w, obs, legal, rng, SIDES)
        g_state, g_u = pol.grad_log_policy(cache, probs, i)
        pol.apply_gradient(w, g_state, g_u, adv=1.0)
        p_lo = _probs(w, _obs(yes_mid=0.0), legal)
        p_hi = _probs(w, _obs(yes_mid=10.0), legal)
        self.assertNotEqual(p_lo, p_hi)


class YesNoIdentityTest(unittest.TestCase):
    """Buy YES vs Buy NO with the same price/size must be distinguishable."""

    def test_same_price_size_different_logits(self):
        legal = [
            "hold",
            {"buy": {"token_id": "tok-yes", "price": "0.50",
                     "size": "10"}},
            {"buy": {"token_id": "tok-no", "price": "0.50",
                     "size": "10"}},
        ]
        obs = _obs()
        w = pol.initial_weights()
        # State bias only in the buy_yes block: the two buys
        # share ψ (same price/size) — only the YES/NO identity separates them,
        # via their class block.
        w["state"]["buy_yes"][0] = 1.0
        sides = SIDES
        lg = pol.logits(w, obs, legal, sides)
        self.assertNotEqual(lg[1], lg[2])
        self.assertAlmostEqual(lg[1], 1.0)   # buy_yes receives the bias
        self.assertAlmostEqual(lg[2], 0.0)   # buy_no no

    def test_exact_gradient_per_block(self):
        # ∇log π_i in the block of class c = φ·(1[c==cls_i] − mass(c)).
        legal = _legal()
        w = pol.initial_weights()
        rng = random.Random(11)
        i, _a, probs, cache = pol.choose(w, _obs(), legal, rng, SIDES)
        phi_s, kinds, _psis = cache
        g_state, _g_u = pol.grad_log_policy(cache, probs, i)
        # Classes without legal actions → mass 0 and never chosen → zero grad.
        for c in ("buy_unknown", "sell_unknown"):
            self.assertTrue(all(v == 0.0 for v in g_state[c]),
                            f"block {c} should have zero gradient")
        # Chosen class: coefficient (1 − mass).
        cls_i = kinds[i]
        mass_i = sum(p for p, c in zip(probs, kinds) if c == cls_i)
        self.assertEqual(g_state[cls_i],
                         [(1.0 - mass_i) * f for f in phi_s])
        # Another class with mass > 0: coefficient −mass.
        other = next(c for c in set(kinds) if c != cls_i)
        mass_o = sum(p for p, c in zip(probs, kinds) if c == other)
        self.assertEqual(g_state[other], [-mass_o * f for f in phi_s])


class NonNullGradientTest(unittest.TestCase):
    """Controlled SYNTHETIC episode (test fixture — NOT evidence about
    the market): non-zero return ⇒ Δw ≠ 0 in state blocks."""

    def test_synthetic_episode_moves_weights(self):
        w = pol.initial_weights()
        w0 = copy.deepcopy(w)
        legal = _legal()
        rng = random.Random(42)
        grads = []
        obs = _obs(yes_mid=0.5)
        # Synthetic 3-step trajectory with total return 1.0.
        for _ in range(3):
            i, _a, probs, cache = pol.choose(w, obs, legal, rng, SIDES)
            grads.append((cache, probs, i))
        ret, baseline = 1.0, 0.0
        adv = ret - baseline
        for cache, probs, i in grads:
            g_state, g_u = pol.grad_log_policy(cache, probs, i)
            pol.apply_gradient(w, g_state, g_u, adv)
        delta = pol.delta_w_norm(w, w0)
        self.assertGreater(delta, 0.0)
        # The state block of the chosen class moved (non-zero gradient
        # over state — the original defect left it at zero).
        cls_i = grads[0][0][1][grads[0][2]]
        self.assertTrue(
            any(abs(v) > 0 for v in w["state"][cls_i]),
            "the state block of the chosen class did not move")

    def test_zero_return_does_not_move_weights(self):
        # adv = 0 ⇒ null update (the weights must not move without
        # a return signal).
        w = pol.initial_weights()
        w0 = copy.deepcopy(w)
        legal = _legal()
        rng = random.Random(5)
        i, _a, probs, cache = pol.choose(w, _obs(), legal, rng, SIDES)
        g_state, g_u = pol.grad_log_policy(cache, probs, i)
        pol.apply_gradient(w, g_state, g_u, adv=0.0)
        self.assertEqual(pol.delta_w_norm(w, w0), 0.0)

    def test_reproducible_with_seed(self):
        # Same seed ⇒ same sequence of sampled actions.
        w = pol.initial_weights()
        legal = _legal()
        obs = _obs()
        picks = []
        for seed in (99, 99):
            rng = random.Random(seed)
            i, _a, _p, _c = pol.choose(w, obs, legal, rng, SIDES)
            picks.append(i)
        self.assertEqual(picks[0], picks[1])


if __name__ == "__main__":
    unittest.main()
