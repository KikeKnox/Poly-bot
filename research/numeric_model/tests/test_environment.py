"""Tests of the `environment.py` client against a FAKE mini-driver in
Python (`fake_env_driver.py`) that speaks the JSONL protocol of contract
F — the Rust `hrf_research_env` binary is not needed for this suite.

Covers: hello handshake (fidelity), reset/step/legal_actions/close, echo
of the `AgentAction` in exact serde wire format, terminal vs truncated, fatal
vs non-fatal error, guaranteed process close by the context
manager and rejection of a step after terminal.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import environment  # noqa: E402

_HERE = Path(__file__).resolve().parent
DRIVER = _HERE / "fake_env_driver.py"


class ResearchEnvTest(unittest.TestCase):
    """Happy path: hello → reset → steps → terminal → close."""

    def _make_env(self, env_extra=None, timeout_s=10):
        # ResearchEnv does not expose the Popen `env`: we use a shell wrapper
        # that sets the hook variables and delegates to the driver.
        with tempfile.NamedTemporaryFile(
                "w", suffix=".sh", delete=False) as w:
            exports = "\n".join(
                f"export {k}={v!r}" for k, v in (env_extra or {}).items())
            w.write(f"#!/bin/sh\n{exports}\n"
                    f"exec {sys.executable} {DRIVER} \"$@\"\n")
            wpath = w.name
        os.chmod(wpath, 0o755)
        self.addCleanup(os.unlink, wpath)
        Path("/tmp/fake_events.jsonl").touch()
        return environment.ResearchEnv(
            "/tmp/fake_events.jsonl", bin_path=wpath,
            timeout_s=timeout_s)

    def test_hello_and_fidelity(self):
        with self._make_env() as env:
            self.assertEqual(env.schema_version, 1)
            self.assertEqual(env.fidelity, "certified")
            self.assertFalse(env.closed)
        self.assertTrue(env.closed)

    def test_fidelity_reduced_is_declared(self):
        with self._make_env(
                {"FAKE_ENV_FIDELITY": "reduced"}) as env:
            self.assertEqual(env.fidelity, "reduced")

    def test_reset_returns_observation_and_legal_actions(self):
        with self._make_env() as env:
            r = env.reset("ep-1", seed=42, market_cid="0xcid")
            obs = r["observation"]
            self.assertEqual(obs["schema_version"], 1)
            self.assertEqual(obs["market_cid"], "0xcid")
            self.assertIn("features", obs)
            self.assertIn("inventory", obs)
            self.assertIn("observation_seqs", obs)
            self.assertTrue(r["legal_actions"])
            # Hold is among the legal actions as an externally tagged serde string.
            self.assertIn("hold", r["legal_actions"])

    def test_step_echoes_wire_of_actions(self):
        """The StepRecord returns `proposed_action` with the action exactly
        as the client serialized it — verifies the exact externally
        tagged serde format."""
        with self._make_env() as env:
            env.reset("ep-1", seed=0)
            # Hold → string "hold" (external serde unit).
            r = env.step(environment.hold())
            self.assertEqual(r["step"]["proposed_action"], "hold")
            self.assertFalse(r["step"]["terminal"])
            self.assertIsNotNone(r["observation"])
        # Second episode: buy with Decimal as a string.
        with self._make_env({"FAKE_ENV_STEPS": "5"}) as env:
            env.reset("ep-2", seed=1)
            r = env.step(environment.buy("tok-yes", "0.47", 10))
            self.assertEqual(r["step"]["proposed_action"], {
                "buy": {"token_id": "tok-yes",
                        "price": "0.47", "size": "10"}})
            r = env.step({"cancel": {"order_id": "sim-order-1"}})
            self.assertEqual(r["step"]["proposed_action"],
                             {"cancel": {"order_id": "sim-order-1"}})

    def test_terminal_and_step_after_terminal_rejected(self):
        with self._make_env({"FAKE_ENV_STEPS": "2"}) as env:
            env.reset("ep-t", seed=0)
            r = env.step("hold")
            self.assertFalse(r["step"]["terminal"])
            r = env.step("hold")
            self.assertTrue(r["step"]["terminal"])
            self.assertEqual(r["step"]["end_reason"], "horizon_reached")
            self.assertIsNone(r["observation"])
            # Step after terminal: the driver answers with a fatal error.
            with self.assertRaises(environment.EnvError) as ctx:
                env.step("hold")
            self.assertTrue(ctx.exception.fatal)

    def test_non_fatal_error_leaves_connection_usable(self):
        with self._make_env(
                {"FAKE_ENV_NONFATAL_FIRST_STEP": "1"}) as env:
            env.reset("ep-nf", seed=0)
            with self.assertRaises(environment.EnvError) as ctx:
                env.step("hold")
            self.assertFalse(ctx.exception.fatal)
            self.assertFalse(env.closed)
            # The connection stays usable.
            r = env.step("hold")
            self.assertFalse(r["step"]["terminal"])

    def test_fatal_error_invalidates_episode(self):
        with self._make_env(
                {"FAKE_ENV_FATAL_ON_RESET": "1"}) as env:
            with self.assertRaises(environment.EnvError) as ctx:
                env.reset("ep-f", seed=0)
            self.assertTrue(ctx.exception.fatal)
            self.assertTrue(env.closed)

    def test_legal_actions(self):
        with self._make_env() as env:
            env.reset("ep-la", seed=0)
            actions = env.legal_actions()
            self.assertIn("hold", actions)
            self.assertTrue(
                any("buy" in a for a in actions if isinstance(a, dict)))

    def test_close_returns_digest_and_kills_process(self):
        env = self._make_env()
        env.reset("ep-c", seed=0)
        res = env.close()
        self.assertTrue(res["reconciled"])
        self.assertEqual(res["digest"], "sha256:fake-driver-digest")
        self.assertTrue(env.closed)
        self.assertIsNotNone(env._proc.poll(),
                             "the process is still alive after close")
        # close is idempotent.
        self.assertIsNone(env.close())

    def test_context_manager_closes_process_on_exception(self):
        env = self._make_env()
        proc = env._proc
        with self.assertRaises(ValueError):
            with env:
                env.reset("ep-x", seed=0)
                raise ValueError("boom from the caller")
        self.assertIsNotNone(proc.poll(),
                             "the process was left alive after the exception")

    def test_broken_handshake_is_fatal_error(self):
        with self.assertRaises(environment.EnvError) as ctx:
            self._make_env({"FAKE_ENV_BAD_HELLO": "1"})
        self.assertTrue(ctx.exception.fatal)

    def test_normalize_action(self):
        self.assertEqual(environment.normalize_action("hold"), "hold")
        self.assertEqual(
            environment.normalize_action({"hold": {}}), "hold")
        self.assertEqual(
            environment.normalize_action(
                {"buy": {"token_id": "t", "price": 0.5, "size": 10}}),
            {"buy": {"token_id": "t", "price": "0.5", "size": "10"}})
        self.assertEqual(
            environment.normalize_action(
                {"cancel": {"order_id": "o1"}}),
            {"cancel": {"order_id": "o1"}})
        with self.assertRaises(ValueError):
            environment.normalize_action({"teleport": {}})


if __name__ == "__main__":
    unittest.main()
