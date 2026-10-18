"""CLI: contract/validate/check-driver/make-fixture/train/snapshot."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PKG_PARENT = str(Path(__file__).resolve().parents[2])


def run_cli(*argv):
    cmd = [sys.executable, "-m", "rl_predictor.cli", *argv]
    env = dict(os.environ, PYTHONPATH=PKG_PARENT)
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


class TestContractCmd(unittest.TestCase):
    def test_contract(self):
        r = run_cli("contract")
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual(len(d["feature_names"]), 22)
        self.assertEqual(len(d["action_names"]), 9)
        self.assertEqual(d["feature_schema"], "rl_predictor.features.v1")


FAKE_DRIVER = r'''
import json, sys
print(json.dumps({"type":"hello","schema_version":1,"fidelity":"certified"}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req["cmd"] == "reset":
        print(json.dumps({"type":"reset","observation":{
            "schema_version":1,"decision_ts_ms":1,"market_cid":"m",
            "features":{"yes_mid":0.5,"spread":0.02},
            "inventory":{"cash_usdc":"10","reserved_usdc":"0",
                         "positions":[]}},
            "legal_actions":["hold"]}), flush=True)
    elif req["cmd"] == "close":
        print(json.dumps({"type":"closed","episode_id":"x",
                          "digest":"d","reconciled":True}), flush=True)
        break
'''


class TestCheckDriver(unittest.TestCase):
    def test_legacy_driver_exit3_with_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            drv = os.path.join(d, "fake_driver.py")
            with open(drv, "w") as f:
                f.write(FAKE_DRIVER)
            # --bin requires an executable: a sh wrapper that invokes the script.
            sh = os.path.join(d, "drv.sh")
            with open(sh, "w") as f:
                f.write(f"#!/bin/sh\nexec {sys.executable} {drv} \"$@\"\n")
            os.chmod(sh, 0o755)
            r = run_cli("check-driver", "--bin", sh, "--tape", "/dev/null",
                        "--timeout", "10")
            self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
            d2 = json.loads(r.stdout)
            self.assertFalse(d2["supported"])
            self.assertIn("observation.policy_v1", d2["missing"])
            self.assertIn("handoff", d2)
            self.assertIn("action_mask",
                          json.dumps(d2["handoff"]["required"]))

    def test_missing_binary_exit4(self):
        r = run_cli("check-driver", "--bin", "/nonexistent/x",
                    "--tape", "/dev/null")
        self.assertEqual(r.returncode, 4)
        self.assertFalse(json.loads(r.stdout)["supported"])


class TestValidateCmd(unittest.TestCase):
    def _write_policy(self, d, status="candidate", fixture=False):
        from rl_predictor.artifact import atomic_write_json, policy_to_dict
        from rl_predictor.policy import LinearPolicy
        from rl_predictor.scaler import FeatureScaler
        tr = {"run_id": "r", "data_manifest_sha256": "1" * 64,
              "split_manifest_sha256": "2" * 64, "seed": 1,
              "completed_episodes": 2, "updates": 1}
        if fixture:
            tr["fixture"] = True
        p = policy_to_dict(LinearPolicy(),
                           FeatureScaler([0.0] * 22, [1.0] * 22),
                           predictor_sha256="0" * 64,
                           training=tr, status=status)
        path = os.path.join(d, "policy.json")
        atomic_write_json(path, p)
        return path

    def test_candidate_not_activatable(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write_policy(d)
            r = run_cli("validate", "--policy", p)
            self.assertEqual(r.returncode, 3, r.stdout)
            self.assertEqual(json.loads(r.stdout)["verdict"],
                             "not_activatable")

    def test_invalid_artifact_exit2(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                json.dump({"schema_version": 99}, f)
            r = run_cli("validate", "--policy", path)
            self.assertEqual(r.returncode, 2)

    def test_approved_with_manifest_activates(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write_policy(d, status="paper_approved")
            env = dict(os.environ, PYTHONPATH=PKG_PARENT)
            gen = subprocess.run(
                [sys.executable, "-c",
                 "import json,sys;from rl_predictor.artifact import "
                 "manifest_for_policy;"
                 f"m=manifest_for_policy('{p}',predictor_sha256='{'0'*64}',"
                 "report_sha256='" + "3" * 64 + "');"
                 "print(json.dumps(m))"],
                capture_output=True, text=True, env=env)
            self.assertEqual(gen.returncode, 0, gen.stderr)
            mp = os.path.join(d, "manifest.json")
            with open(mp, "w") as f:
                f.write(gen.stdout)
            r = run_cli("validate", "--policy", p, "--manifest", mp)
            self.assertEqual(r.returncode, 0, r.stdout)
            self.assertEqual(json.loads(r.stdout)["verdict"],
                             "activatable")


class TestMakeFixture(unittest.TestCase):
    def test_fixture_written_and_not_promotable(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_cli("make-fixture", "--out", d, "--seed", "9")
            self.assertEqual(r.returncode, 0, r.stderr)
            pj = os.path.join(d, "fixture_policy_init.json")
            self.assertTrue(os.path.exists(pj))
            self.assertTrue(os.path.exists(
                os.path.join(d, "fixture_cohort.jsonl")))
            self.assertTrue(os.path.exists(
                os.path.join(d, "fixture_expected_update.json")))
            with open(pj) as f:
                pol = json.load(f)
            self.assertTrue(pol["training"]["fixture"])
            self.assertEqual(pol["status"], "candidate")
            r2 = run_cli("validate", "--policy", pj)
            self.assertEqual(r2.returncode, 3)  # never activatable
            # Determinism: same seed -> same expected.
            with tempfile.TemporaryDirectory() as d2:
                r3 = run_cli("make-fixture", "--out", d2, "--seed", "9")
                self.assertEqual(r3.returncode, 0)
                self.assertEqual(
                    Path(d, "fixture_expected_update.json").read_text(),
                    Path(d2, "fixture_expected_update.json").read_text())


class TestTrainCmd(unittest.TestCase):
    def _transitions(self, path, source="recorded_real"):
        with open(path, "w") as f:
            f.write(json.dumps({
                "schema": "rl_predictor.transitions.v1",
                "source": source}) + "\n")
            for e in range(2):
                steps = [{"obs_raw": [0.4 + 0.01 * i] * 22,
                          "mask": 0b111111111,
                          "action": i % 9,
                          "behavior_prob": 1.0 / 9,
                          "reward": 0.01 * (e + 1)}
                         for i in range(3)]
                f.write(json.dumps({"episode_id": f"ep{e}",
                                    "trainable": True,
                                    "steps": steps}) + "\n")

    def test_train_writes_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            tp = os.path.join(d, "t.jsonl")
            self._transitions(tp)
            out = os.path.join(d, "out")
            r = run_cli("train", "--transitions", tp, "--out", out,
                        "--predictor-sha256", "a" * 64,
                        "--data-manifest-sha256", "b" * 64,
                        "--split-manifest-sha256", "c" * 64,
                        "--seed", "4")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            d2 = json.loads(r.stdout)
            self.assertEqual(d2["verdict"], "trained_candidate")
            pj = json.loads(Path(out, "policy.json").read_text())
            self.assertEqual(pj["status"], "candidate")
            self.assertEqual(pj["training"]["updates"], 1)
            self.assertEqual(pj["action_names"][8], "SELL_NO_ALL")

    def test_train_rejects_fixture_source(self):
        with tempfile.TemporaryDirectory() as d:
            tp = os.path.join(d, "t.jsonl")
            self._transitions(tp, source="fixture")
            r = run_cli("train", "--transitions", tp,
                        "--out", os.path.join(d, "o"),
                        "--predictor-sha256", "a" * 64,
                        "--data-manifest-sha256", "b" * 64,
                        "--split-manifest-sha256", "c" * 64)
            self.assertEqual(r.returncode, 2)
            self.assertIn("fixture", json.loads(r.stdout)["error"])

    def test_train_rejects_bad_schema(self):
        with tempfile.TemporaryDirectory() as d:
            tp = os.path.join(d, "t.jsonl")
            with open(tp, "w") as f:
                f.write(json.dumps({"schema": "other.thing"}) + "\n")
            r = run_cli("train", "--transitions", tp,
                        "--out", os.path.join(d, "o"),
                        "--predictor-sha256", "a" * 64,
                        "--data-manifest-sha256", "b" * 64,
                        "--split-manifest-sha256", "c" * 64)
            self.assertEqual(r.returncode, 2)


class TestSnapshotCmd(unittest.TestCase):
    def test_snapshot_copy_verify_and_secret_exclusion(self):
        with tempfile.TemporaryDirectory() as src, \
                tempfile.TemporaryDirectory() as dst:
            nm = os.path.join(src, "research", "numeric_model")
            tt = os.path.join(nm, "tests")
            os.makedirs(tt)
            with open(os.path.join(nm, "m.py"), "w") as f:
                f.write("x = 1\n")
            with open(os.path.join(tt, "t.py"), "w") as f:
                f.write("y = 2\n")
            os.makedirs(os.path.join(nm, "__pycache__"))
            with open(os.path.join(nm, "__pycache__", "m.pyc"),
                      "w") as f:
                f.write("cache")
            secret = os.path.join(nm, ".env")
            with open(secret, "w") as f:
                f.write("POLY_PRIVATE_KEY=abcdef0123456789\n")
            os.makedirs(os.path.join(src, "tools"))
            with open(os.path.join(src, "tools", "u.py"), "w") as f:
                f.write("z = 3\n")
            r = run_cli("snapshot", "--src", src, "--dest", dst)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            d2 = json.loads(r.stdout)
            self.assertEqual(d2["verdict"], "verified")
            self.assertTrue(os.path.exists(
                os.path.join(dst, "research", "numeric_model", "m.py")))
            self.assertFalse(os.path.exists(
                os.path.join(dst, "research", "numeric_model", ".env")))
            self.assertFalse(os.path.exists(
                os.path.join(dst, "research", "numeric_model",
                             "__pycache__", "m.pyc")))
            man = json.loads(Path(dst, "MANIFEST.json").read_text())
            self.assertTrue(all(
                len(e["sha256"]) == 64 for e in man["entries"]))
            # Corrupting a destination -> verify-only detects it.
            with open(os.path.join(dst, "tools", "u.py"), "a") as f:
                f.write("corruption\n")
            r2 = run_cli("snapshot", "--src", src, "--dest", dst,
                         "--verify-only")
            self.assertEqual(r2.returncode, 3)
            self.assertIn("u.py", json.dumps(json.loads(r2.stdout)))


if __name__ == "__main__":
    unittest.main()
