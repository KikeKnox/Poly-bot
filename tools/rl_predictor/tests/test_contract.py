"""Literal contract: 22 features and 9 actions in exact order; mask;
token identity from metadata (not from position in the legal list)."""

import unittest

from rl_predictor import contract as C


class TestNames(unittest.TestCase):
    def test_action_order_literal(self):
        self.assertEqual(C.ACTION_NAMES, (
            "HOLD", "BUY_YES_SMALL", "BUY_YES_FULL",
            "BUY_NO_SMALL", "BUY_NO_FULL",
            "SELL_YES_HALF", "SELL_YES_ALL",
            "SELL_NO_HALF", "SELL_NO_ALL"))
        self.assertEqual(C.NUM_ACTIONS, 9)
        self.assertEqual(C.ACTION_INDEX["HOLD"], 0)
        self.assertEqual(C.ACTION_INDEX["SELL_NO_ALL"], 8)

    def test_feature_order_literal(self):
        self.assertEqual(C.NUM_FEATURES, 22)
        self.assertEqual(C.FEATURE_NAMES[0], "p_yes")
        self.assertEqual(C.FEATURE_NAMES[1], "p_missing")
        self.assertEqual(C.FEATURE_NAMES[2:6],
                         ("yes_bid", "yes_ask", "no_bid", "no_ask"))
        self.assertEqual(C.FEATURE_NAMES[10], "time_remaining_frac")
        self.assertEqual(C.FEATURE_NAMES[15], "free_budget_usdc")
        self.assertEqual(C.FEATURE_NAMES[17], "pending_order")
        self.assertEqual(C.FEATURE_NAMES[21], "buy_fee_rate")
        self.assertEqual(len(set(C.FEATURE_NAMES)), 22)

    def test_schemas(self):
        self.assertEqual(C.FEATURE_SCHEMA, "rl_predictor.features.v1")
        self.assertEqual(C.POLICY_ALGORITHM, "reinforce_linear_v1")
        self.assertEqual(C.POLICY_SCHEMA_VERSION, 1)


class TestMask(unittest.TestCase):
    def test_hold_always_required(self):
        with self.assertRaises(C.ContractError):
            C.check_mask(0)  # no HOLD
        with self.assertRaises(C.ContractError):
            C.check_mask(0b101111110)  # everything except HOLD

    def test_int_and_list_forms(self):
        self.assertEqual(C.check_mask(0b111111111), 511)
        self.assertEqual(C.check_mask([1] * 9), 511)
        self.assertEqual(C.check_mask([True] + [False] * 8), 1)

    def test_out_of_range(self):
        with self.assertRaises(C.ContractError):
            C.check_mask(1 << 9)
        with self.assertRaises(C.ContractError):
            C.check_mask(-1)
        with self.assertRaises(C.ContractError):
            C.check_mask([1] * 8)

    def test_legal_ids_ordered(self):
        self.assertEqual(C.legal_action_ids(0b100000001), [0, 8])


class TestTokenSide(unittest.TestCase):
    """Identity from metadata. Positional inference is a known failure:
    a mask that omits the first side does NOT rename the other one."""

    def test_metadata_not_position(self):
        yes, no = "tok_y", "tok_n"
        # Even when only the NO action remains legal, the token is still no.
        self.assertEqual(C.token_side(no, yes, no), "no")
        self.assertEqual(C.token_side(yes, yes, no), "yes")
        self.assertIsNone(C.token_side("other", yes, no))
        self.assertIsNone(C.token_side(yes, None, no))

    def test_contradictory_metadata_unknown(self):
        self.assertIsNone(C.token_side("t", "t", "t"))


class TestObsVector(unittest.TestCase):
    def test_len_and_finite(self):
        good = [0.0] * 22
        self.assertEqual(C.check_observation_vector(good), good)
        with self.assertRaises(C.ContractError):
            C.check_observation_vector([0.0] * 21)
        with self.assertRaises(C.ContractError):
            C.check_observation_vector([0.0] * 23)

    def test_nan_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            v = [0.0] * 22
            v[7] = bad
            with self.assertRaises(C.ContractError):
                C.check_observation_vector(v)
        v = [0.0] * 22
        v[0] = "abc"
        with self.assertRaises(C.ContractError):
            C.check_observation_vector(v)


if __name__ == "__main__":
    unittest.main()
