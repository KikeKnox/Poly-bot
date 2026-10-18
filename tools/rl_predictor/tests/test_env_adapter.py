"""JSONL adapter: fail-closed gate on the v1 contract against the real driver wire
and the full loop with an in-memory fake v1 driver."""

import unittest

from rl_predictor.contract import FEATURE_SCHEMA, NUM_FEATURES
from rl_predictor.env_adapter import (
    AdapterError,
    ContractUnsupportedError,
    V1EnvAdapter,
    inspect_reset_response,
    parse_policy_obs,
)
from rl_predictor.trainer import ReinforceTrainer
from rl_predictor.policy import LinearPolicy
from rl_predictor.scaler import FeatureScaler


def legacy_reset_response():
    """Real shape of the inspected contract F: EnvObservation with the
    predictor features (mapping name->value) and legal_actions as
    AgentAction serde."""
    return {
        "type": "reset",
        "observation": {
            "schema_version": 1,
            "decision_ts_ms": 1790356000000,
            "market_cid": "m1",
            "features": {"yes_mid": 0.51, "spread": 0.02,
                         "moneyness": 0.001, "time_left_ms": 120000},
            "inventory": {"cash_usdc": "10", "reserved_usdc": "0",
                          "positions": []},
        },
        "legal_actions": ["hold", {"buy": {"token_id": "t_yes",
                                           "price": "0.55",
                                           "size": "5"}}],
    }


def v1_reset_response(account=None, mask=0b111111111):
    return {
        "type": "reset",
        "observation": {
            "policy_v1": {
                "feature_schema": FEATURE_SCHEMA,
                "features": [0.5] * NUM_FEATURES,
                "feature_names": None,
                "action_mask": mask,
                "meta": {"yes_token_id": "tY", "no_token_id": "tN",
                         "market_cid": "m1"},
                "account": account or {
                    "equity_um": 10_000_000, "deposits_um": 0,
                    "withdrawals_um": 0, "exposure_um": 0},
                "invalid_reasons": [],
            },
        },
        "legal_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }


class TestInspect(unittest.TestCase):
    def test_legacy_contract_rejected(self):
        ok, missing, notes = inspect_reset_response(legacy_reset_response())
        self.assertFalse(ok)
        self.assertIn("observation.policy_v1", missing)
        self.assertTrue(any("PREDICTOR" in n for n in notes))

    def test_v1_accepted(self):
        ok, missing, _ = inspect_reset_response(v1_reset_response())
        self.assertTrue(ok, missing)

    def test_v1_missing_fields_listed(self):
        r = v1_reset_response()
        del r["observation"]["policy_v1"]["account"]
        del r["observation"]["policy_v1"]["meta"]["yes_token_id"]
        ok, missing, _ = inspect_reset_response(r)
        self.assertFalse(ok)
        self.assertIn("policy_v1.account", missing)
        self.assertIn("policy_v1.meta.yes_token_id", missing)

    def test_mask_without_hold_rejected(self):
        ok, missing, _ = inspect_reset_response(v1_reset_response(
            mask=0b111111110))
        self.assertFalse(ok)
        self.assertTrue(any("action_mask" in m for m in missing))

    def test_nonfinite_features_rejected(self):
        r = v1_reset_response()
        r["observation"]["policy_v1"]["features"][3] = float("nan")
        ok, missing, _ = inspect_reset_response(r)
        self.assertFalse(ok)

    def test_parse_raises_with_handoff(self):
        with self.assertRaises(ContractUnsupportedError) as cm:
            parse_policy_obs(legacy_reset_response())
        e = cm.exception
        self.assertIn("policy_v1", str(e))
        self.assertEqual(e.handoff["driver"],
                         "hrf_research_env (contract F)")
        self.assertIn("policy_v1.action_mask",
                      e.handoff["required"])


class FakeTransport:
    """In-memory v1 driver: reset -> v1 obs; step -> equity rises
    with BUY(1) and falls with other actions; terminal after max_steps."""

    def __init__(self, max_steps=4, equity_start=10_000_000):
        self.max_steps = max_steps
        self.equity = equity_start
        self.sent = []
        self.t = 0
        self.closed = False

    def send(self, obj):
        self.sent.append(obj)

    def recv(self):
        req = self.sent[-1]
        cmd = req["cmd"]
        if cmd == "reset":
            self.t = 0
            return v1_reset_response(
                account={"equity_um": self.equity, "deposits_um": 0,
                         "withdrawals_um": 0, "exposure_um": 0})
        if cmd == "step":
            self.t += 1
            a = req["action"]["policy_v1"]
            self.equity += 50_000 if a == 1 else -10_000
            terminal = self.t >= self.max_steps
            obs = None
            if not terminal:
                obs = {
                    "policy_v1": {
                        "feature_schema": FEATURE_SCHEMA,
                        "features": [0.5] * NUM_FEATURES,
                        "action_mask": 0b111111111,
                        "meta": {"yes_token_id": "tY",
                                 "no_token_id": "tN",
                                 "market_cid": "m1"},
                        "account": {
                            "equity_um": self.equity,
                            "deposits_um": 0, "withdrawals_um": 0,
                            "exposure_um": 0},
                        "invalid_reasons": [],
                    },
                }
            else:
                obs = {"policy_v1": {
                    "feature_schema": FEATURE_SCHEMA,
                    "features": [0.5] * NUM_FEATURES,
                    "action_mask": 0b111111111,
                    "meta": {"yes_token_id": "tY", "no_token_id": "tN",
                             "market_cid": "m1"},
                    "account": {"equity_um": self.equity,
                                "deposits_um": 0, "withdrawals_um": 0,
                                "exposure_um": 0},
                    "invalid_reasons": []}}
            return {"type": "step_result",
                    "step": {"terminal": terminal, "truncated": False,
                             "invalid_reasons": []},
                    "observation": obs,
                    "legal_actions": list(range(9))}
        if cmd == "close":
            return {"type": "closed", "episode_id": "e",
                    "digest": "d", "reconciled": True}
        raise AdapterError(f"unexpected cmd {cmd}")

    def close(self):
        self.closed = True


class FakeLegacyTransport(FakeTransport):
    def recv(self):
        req = self.sent[-1]
        if req["cmd"] == "reset":
            return legacy_reset_response()
        if req["cmd"] == "close":
            return {"type": "closed"}
        raise AdapterError("stop")


class TestAdapterFailClosed(unittest.TestCase):
    def test_legacy_driver_fails_closed(self):
        ad = V1EnvAdapter(FakeLegacyTransport(), budget_um=10_000_000)
        with self.assertRaises(ContractUnsupportedError) as cm:
            ad.reset("ep1", seed=0)
        self.assertIn("policy_v1", str(cm.exception))
        self.assertTrue(cm.exception.missing)


class TestAdapterV1Loop(unittest.TestCase):
    def test_episode_collect_and_rewards(self):
        ad = V1EnvAdapter(FakeTransport(max_steps=4),
                          budget_um=10_000_000)
        tr = ReinforceTrainer(
            LinearPolicy(),
            FeatureScaler([0.0] * NUM_FEATURES, [1.0] * NUM_FEATURES),
            lr=0.01, grad_clip=5.0, seed=3, cohort_episodes=1)
        ep = tr.collect_episode(ad, "ep1", 0, 0)
        ad.close()
        self.assertEqual(len(ep.steps), 4)
        # Rewards come from the equity delta reported by the driver,
        # normalized by budget: ±0.005 per step depending on the action.
        for s in ep.steps:
            self.assertIn(s["reward"], (0.005, -0.001))
        self.assertEqual(ep.policy_fingerprint,
                         __import__("rl_predictor.trainer",
                                    fromlist=["x"]).policy_fingerprint(
                                        tr.policy))

    def test_mask_and_meta_flow(self):
        ad = V1EnvAdapter(FakeTransport(max_steps=1), budget_um=1)
        st = ad.reset("ep", 0)
        self.assertEqual(st["mask"], 0b111111111)
        self.assertEqual(st["meta"]["yes_token_id"], "tY")
        self.assertEqual(st["obs_raw"], [0.5] * NUM_FEATURES)


if __name__ == "__main__":
    unittest.main()
