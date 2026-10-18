"""Artifacts: policy.json, manifest, online manifest, hashes and the
activation verdict. An explicit fixture is never promotable."""

import json
import os
import tempfile
import unittest

from rl_predictor import artifact as A
from rl_predictor.contract import (
    ACTION_NAMES,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    STATUS_CANDIDATE,
    STATUS_ONLINE_CANDIDATE,
    STATUS_PAPER_APPROVED,
)
from rl_predictor.policy import LinearPolicy
from rl_predictor.scaler import FeatureScaler
from rl_predictor.trainer import make_online_link

ZERO64 = "0" * 64
ONE64 = "1" * 64
TWO64 = "2" * 64
THREE64 = "3" * 64


def make_policy_dict(status=STATUS_CANDIDATE, fixture=False,
                     predictor=ZERO64):
    sc = FeatureScaler([0.0] * 22, [1.0] * 22)
    tr = {
        "run_id": "r1", "data_manifest_sha256": ONE64,
        "split_manifest_sha256": TWO64, "seed": 7,
        "completed_episodes": 4, "updates": 1,
    }
    if fixture:
        tr["fixture"] = True
    return A.policy_to_dict(
        LinearPolicy(), sc, predictor_sha256=predictor,
        training=tr, status=status)


def sha_of_dict(d):
    return A.sha256_bytes(json.dumps(d, sort_keys=True).encode())


class TestPolicyValidation(unittest.TestCase):
    def test_valid_roundtrip(self):
        p = make_policy_dict()
        self.assertEqual(A.validate_policy_dict(p), p)

    def test_feature_names_exact_literal(self):
        p = make_policy_dict()
        p["feature_names"] = list(reversed(FEATURE_NAMES))
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)
        p = make_policy_dict()
        p["feature_names"] = list(FEATURE_NAMES) + ["extra"]
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)

    def test_status_enum(self):
        p = make_policy_dict()
        p["status"] = "mystery"
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)

    def test_fixture_not_promotable(self):
        p = make_policy_dict(status=STATUS_PAPER_APPROVED)
        p["training"]["fixture"] = True
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)

    def test_dims_and_finite(self):
        p = make_policy_dict()
        p["weights"][0] = [0.0] * 21
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)
        p = make_policy_dict()
        p["weights"][4][9] = float("nan")
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)
        p = make_policy_dict()
        p["bias"] = [0.0] * 8
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)

    def test_training_counters_required_positive(self):
        p = make_policy_dict()
        p["training"]["completed_episodes"] = 0
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)
        p = make_policy_dict()
        del p["training"]["seed"]
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)

    def test_predictor_hash_checked(self):
        p = make_policy_dict()
        p["predictor_sha256"] = "nothex"
        with self.assertRaises(A.ArtifactError):
            A.validate_policy_dict(p)


class TestManifest(unittest.TestCase):
    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "policy.json")
            p = make_policy_dict(status=STATUS_PAPER_APPROVED)
            A.atomic_write_json(path, p)
            m = A.manifest_for_policy(path, predictor_sha256=ZERO64,
                                      report_sha256=THREE64)
            A.validate_manifest_dict(m)
            self.assertEqual(m["policy_sha256"], A.sha256_file(path))
            # Load and check the real activation from file.
            p2, pol, sc = A.load_policy(path)
            ok, _ = A.check_activation(
                p2, manifest_dict=m, policy_sha256=m["policy_sha256"])
            self.assertTrue(ok)

    def test_manifest_hash_binding(self):
        m = {"schema_version": 1, "policy_sha256": ONE64,
             "predictor_sha256": ZERO64, "feature_schema": FEATURE_SCHEMA,
             "approval": {"scope": "paper", "report_sha256": None}}
        p = make_policy_dict(status=STATUS_PAPER_APPROVED)
        ok, reason = A.check_activation(p, manifest_dict=m,
                                        policy_sha256=TWO64)
        self.assertFalse(ok)
        self.assertIn("policy_sha256", reason)


class TestActivation(unittest.TestCase):
    def test_candidate_never_activates(self):
        p = make_policy_dict(status=STATUS_CANDIDATE)
        ok, reason = A.check_activation(p, policy_sha256=sha_of_dict(p))
        self.assertFalse(ok)
        self.assertIn("candidate", reason)

    def test_approved_needs_manifest(self):
        p = make_policy_dict(status=STATUS_PAPER_APPROVED)
        ok, _ = A.check_activation(p, policy_sha256=sha_of_dict(p))
        self.assertFalse(ok)

    def test_online_candidate_chain(self):
        # paper_approved root
        root = make_policy_dict(status=STATUS_PAPER_APPROVED)
        root_sha = sha_of_dict(root)
        child = make_policy_dict(status=STATUS_ONLINE_CANDIDATE)
        child_sha = sha_of_dict(child)
        om = {
            "schema_version": 1,
            "kind": A.ONLINE_MANIFEST_KIND,
            "root_policy_sha256": root_sha,
            "links": [make_online_link(
                root_sha, child_sha, 1, lr=0.05, grad_clip=1.0, seed=7,
                cohort_episodes=4, completed_episodes=4, updates=1,
                ts_ms=123)],
        }
        A.validate_online_manifest_dict(om)
        ok, reason = A.check_activation(
            child, online_manifest_dict=om, policy_sha256=child_sha)
        self.assertTrue(ok, reason)

    def test_online_chain_broken_rejected(self):
        om = {
            "schema_version": 1, "kind": A.ONLINE_MANIFEST_KIND,
            "root_policy_sha256": ZERO64,
            "links": [
                make_online_link(ZERO64, ONE64, 1, lr=0.1, grad_clip=1,
                                 seed=1, cohort_episodes=1,
                                 completed_episodes=1, updates=1,
                                 ts_ms=1),
                make_online_link(TWO64, THREE64, 2, lr=0.1, grad_clip=1,
                                 seed=1, cohort_episodes=1,
                                 completed_episodes=2, updates=2,
                                 ts_ms=2),
            ],
        }
        with self.assertRaises(A.ArtifactError):
            A.validate_online_manifest_dict(om)

    def test_online_promoted_as_paper_rejected(self):
        om = {
            "schema_version": 1, "kind": A.ONLINE_MANIFEST_KIND,
            "root_policy_sha256": ZERO64,
            "links": [dict(
                make_online_link(ZERO64, ONE64, 1, lr=0.1, grad_clip=1,
                                 seed=1, cohort_episodes=1,
                                 completed_episodes=1, updates=1,
                                 ts_ms=1),
                promoted_as="paper_approved")],
        }
        with self.assertRaises(A.ArtifactError):
            A.validate_online_manifest_dict(om)


if __name__ == "__main__":
    unittest.main()
