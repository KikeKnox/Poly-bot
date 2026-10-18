"""Adapter between rl_predictor and the JSONL driver of the episodic environment.

Inspected real wire (research/numeric_model/environment.py,
src/research/contract.rs): one subprocess
`hrf_research_env --tape <f>` per environment, JSONL stdin/stdout,
cmd reset/step/legal_actions/close; AgentAction hold|buy|sell|cancel;
EnvObservation with `features` (BTreeMap name->value from the extractor of
the PREDICTOR) and `inventory`.

The current contract does NOT produce the v1 policy observation
(feature_schema 'rl_predictor.features.v1', 22 features sorted,
action_mask of 9 bits, account_state for accounting reward, metadata
yes/no token). Therefore this adapter is FAIL-CLOSED: if the driver does not
emit the required `policy_v1` block, it raises ContractUnsupportedError with
a `handoff` payload that enumerates EXACTLY what the driver must add —
the bot is not patched, no answers are invented, and the mask is not derived
from the legal items by position.

Required v1 interface of the driver (handoff to the integrator):
- reset/step_result: observation["policy_v1"] = {
    "feature_schema": "rl_predictor.features.v1",
    "features": [22 floats order contractual],
    "action_mask": int (bit i = action i valid; HOLD always),
    "meta": {"yes_token_id": str, "no_token_id": str,
             "market_cid": str},
    "account": {"equity_um": int, "deposits_um": int,
                "withdrawals_um": int, "exposure_um": int},
    "invalid_reasons": [str]  # stale/gap/truncated/ambiguous
  }
- step accepts {"cmd":"step","action":{"policy_v1": action_id}} and the
  driver translates id->intent with ITS size/limit semantics; the
  Rust ledger remains the authority for fills/equity.
- reward: the adapter keeps NO ledger; it computes net_pnl_delta between
  `account` snapshots reported by the driver (reward.py), gamma=1.
"""

import json
import math
import os
import select
import subprocess

from .contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    NUM_ACTIONS,
    NUM_FEATURES,
    ContractError,
    check_mask,
)


class AdapterError(RuntimeError):
    pass


class ContractUnsupportedError(AdapterError):
    """The driver does not produce the v1 contract. `handoff` describes the
    exact patch scope for the integrator — an explicit error, not a promise."""

    def __init__(self, message, missing=None, handoff=None):
        super().__init__(message)
        self.missing = list(missing or [])
        self.handoff = handoff or build_handoff(self.missing)


REQUIRED_POLICY_V1_KEYS = (
    "feature_schema",
    "features",
    "action_mask",
    "meta",
    "account",
)
REQUIRED_ACCOUNT_KEYS = (
    "equity_um",
    "deposits_um",
    "withdrawals_um",
    "exposure_um",
)
REQUIRED_META_KEYS = ("yes_token_id", "no_token_id")


def build_handoff(missing):
    """Patch scope for the driver, for the integrator (this adapter does not touch the bot)."""
    return {
        "driver": "hrf_research_env (contract F)",
        "block_bytes": "observation.policy_v1 / step_result.policy_v1",
        "missing_in_inspected_response": list(missing),
        "required": {
            "policy_v1.feature_schema": FEATURE_SCHEMA,
            "policy_v1.features": f"[{NUM_FEATURES} floats, literal order "
                                  "fixed by the contract]",
            "policy_v1.action_mask": "int bits 0..8, HOLD always valid",
            "policy_v1.meta": list(REQUIRED_META_KEYS) + ["market_cid"],
            "policy_v1.account": list(REQUIRED_ACCOUNT_KEYS),
            "policy_v1.invalid_reasons": "exclusion reason items",
            "step.action": {"policy_v1": "action_id 0..8"},
        },
        "notes": [
            "The Rust driver is the authority for ledger/fills/equity; Python "
            "only diffs snapshots and normalizes reward (reward.py).",
            "The mask must come from the state, not be inferred from the order "
            "of a legal-items list that omits sides without an action.",
            "features/policy_v1 belong to the POLICY vector; the extractor "
            "of the frozen predictor (current features map) is not touched.",
        ],
    }


def inspect_reset_response(resp):
    """Pure verdict over a reset/observation response.

    Returns (supported: bool, missing: [str], notes: [str]). It does not raise:
    the caller decides (the CLI probe prints the verdict; the adapter raises)."""
    missing, notes = [], []
    if not isinstance(resp, dict):
        return False, ["response is not an object"], []
    obs = resp.get("observation")
    if not isinstance(obs, dict):
        return False, ["observation missing or not an object"], []
    pv1 = obs.get("policy_v1")
    if not isinstance(pv1, dict):
        missing.append("observation.policy_v1")
        if isinstance(obs.get("features"), dict):
            notes.append(
                "observation.features is the PREDICTOR mapping "
                f"({len(obs['features'])} keys, e.g. "
                f"{sorted(obs['features'])[:3]}), not the policy v1 vector")
        return False, missing, notes
    fs = pv1.get("feature_schema")
    if fs != FEATURE_SCHEMA:
        missing.append(f"policy_v1.feature_schema={fs!r} != {FEATURE_SCHEMA!r}")
    feats = pv1.get("features")
    if not isinstance(feats, list) or len(feats) != NUM_FEATURES:
        missing.append(
            f"policy_v1.features with {0 if feats is None else len(feats)} "
            f"items != {NUM_FEATURES}")
    elif any(not isinstance(x, (int, float)) or isinstance(x, bool)
             or not math.isfinite(x) for x in feats):
        missing.append("policy_v1.features with non-finite values")
    else:
        names = pv1.get("feature_names")
        if names is not None and names != list(FEATURE_NAMES):
            missing.append("policy_v1.feature_names != contract order")
    try:
        check_mask(pv1.get("action_mask"))
    except ContractError as e:
        missing.append(f"policy_v1.action_mask: {e}")
    meta = pv1.get("meta")
    if not isinstance(meta, dict):
        missing.append("policy_v1.meta")
    else:
        for k in REQUIRED_META_KEYS:
            if meta.get(k) is None:
                missing.append(f"policy_v1.meta.{k}")
    acc = pv1.get("account")
    if not isinstance(acc, dict):
        missing.append("policy_v1.account")
    else:
        for k in REQUIRED_ACCOUNT_KEYS:
            v = acc.get(k)
            if not isinstance(v, int) or isinstance(v, bool):
                missing.append(f"policy_v1.account.{k} (int required)")
    return not missing, missing, notes


def parse_policy_obs(resp):
    """Extracts the validated v1 observation or raises ContractUnsupportedError."""
    ok, missing, notes = inspect_reset_response(resp)
    if not ok:
        raise ContractUnsupportedError(
            "driver does not produce the policy v1 contract: "
            + "; ".join(missing + notes),
            missing=missing)
    pv1 = resp["observation"]["policy_v1"]
    return {
        "obs_raw": [float(x) for x in pv1["features"]],
        "mask": check_mask(pv1["action_mask"]),
        "meta": dict(pv1["meta"]),
        "account": dict(pv1["account"]),
        "invalid_reasons": list(pv1.get("invalid_reasons") or ()),
        # Causal flags from the driver: `warmup` (pre-open/pending GAP) and
        # `trainable_step` (the mask offered a real decision, not a forced
        # HOLD). The trainer uses them to exclude the step from the
        # gradient/denominator — masking to HOLD is not enough.
        "warmup": bool(pv1.get("warmup", False)),
        "trainable_step": bool(pv1.get("trainable_step", True)),
    }


def parse_step_response(resp):
    """step_result v1 -> dict compatible with trainer.collect_episode."""
    if not isinstance(resp, dict) or resp.get("type") != "step_result":
        raise AdapterError(f"unexpected step response: {resp!r}")
    step = resp.get("step") or {}
    out = {
        "terminal": bool(step.get("terminal")),
        "truncated": bool(step.get("truncated")),
        "invalid_reasons": list(step.get("invalid_reasons") or ()),
    }
    obs = resp.get("observation")
    if obs is not None:
        parsed = parse_policy_obs({"observation": obs})
        out["obs_raw"] = parsed["obs_raw"]
        out["mask"] = parsed["mask"]
        out["invalid_reasons"] += parsed["invalid_reasons"]
        out["account"] = parsed["account"]
        out["warmup"] = parsed["warmup"]
        out["trainable_step"] = parsed["trainable_step"]
    return out


class JsonlSubprocess:
    """Minimal JSONL stdin/stdout transport over subprocess (POSIX).

    One process per environment — never shared between independent
    episodes (isolation of the replay clock)."""

    def __init__(self, cmd, cwd=None, timeout_s=None, env=None,
                 stderr_path=None):
        self.timeout_s = timeout_s
        # stderr goes to stderr_path (never an undrained PIPE — deadlock) or DEVNULL.
        self._stderr_fh = open(stderr_path, "w") if stderr_path \
            else subprocess.DEVNULL
        self._proc = subprocess.Popen(
            [str(c) for c in cmd], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._stderr_fh,
            text=True, bufsize=1, cwd=cwd, env=env)
        self._closed = False

    def send(self, obj):
        if self._proc.poll() is not None:
            raise AdapterError("driver terminated")
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def recv(self):
        fd = self._proc.stdout.fileno()
        if self.timeout_s is not None:
            r, _, _ = select.select([fd], [], [], self.timeout_s)
            if not r:
                raise AdapterError("timeout waiting for the driver response")
        line = self._proc.stdout.readline()
        if not line:
            raise AdapterError(
                f"driver closed stdout (exit={self._proc.poll()})")
        try:
            resp = json.loads(line)
        except json.JSONDecodeError as e:
            raise AdapterError(f"non-JSON driver response: {line[:200]!r}")
        if resp.get("type") == "error":
            raise AdapterError(
                f"driver error fatal={resp.get('fatal')}: "
                f"{resp.get('message')}")
        return resp

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            if self._stderr_fh is not subprocess.DEVNULL:
                try:
                    self._stderr_fh.close()
                except Exception:
                    pass


class V1EnvAdapter:
    """v1 environment for the trainer over a JSONL transport.

    With the current driver (current contract) the first reset raises
    ContractUnsupportedError — fail-closed. `transport` is injectable for
    tests (in-memory fake v1 driver)."""

    def __init__(self, transport, *, budget_um,
                 lambda_inventory=0.0, lambda_drawdown=0.0):
        self._t = transport
        self.budget_um = int(budget_um)
        self.lambda_inventory = float(lambda_inventory)
        self.lambda_drawdown = float(lambda_drawdown)
        self._prev_account = None
        self._peak_adj = None
        self._dd_prev = 0
        self._closed = False

    @classmethod
    def spawn(cls, bin_path, tape, *, budget_um, extra_args=(), cwd=None,
              timeout_s=30, stderr_path=None, **kw):
        t = JsonlSubprocess([bin_path, "--tape", tape, *extra_args],
                            cwd=cwd, timeout_s=timeout_s,
                            stderr_path=stderr_path)
        hello = t.recv()
        if hello.get("type") != "hello":
            t.close()
            raise AdapterError(f"unexpected handshake: {hello}")
        self = cls(t, budget_um=budget_um, **kw)
        self.hello = hello
        return self

    def _request(self, req, expect):
        self._t.send(req)
        resp = self._t.recv()
        if resp.get("type") != expect:
            raise AdapterError(
                f"response {resp.get('type')!r} for {req['cmd']!r}")
        return resp

    def _reward_from_account(self, account):
        """Step reward from the delta of snapshots reported by the
        driver (no ledger of its own). Returns the normalized reward."""
        from .reward import normalized_reward
        snap = type("S", (), account)  # *_um attributes
        if self._prev_account is None:
            self._prev_account = snap
            return 0.0
        prev, self._prev_account = self._prev_account, snap
        delta = (snap.equity_um - prev.equity_um) \
            - (snap.deposits_um - prev.deposits_um) \
            + (snap.withdrawals_um - prev.withdrawals_um)
        adj = snap.equity_um - snap.deposits_um + snap.withdrawals_um
        if self._peak_adj is None or adj > self._peak_adj:
            self._peak_adj = adj
        dd = max(0, self._peak_adj - adj)
        dd_delta, self._dd_prev = dd - self._dd_prev, dd
        return normalized_reward(
            delta, budget_um=self.budget_um,
            exposure_um=snap.exposure_um, dd_delta_um=dd_delta,
            elapsed_ms=account.get("elapsed_ms", 0),
            lambda_inventory=self.lambda_inventory,
            lambda_drawdown=self.lambda_drawdown)

    # ── env interface for ReinforceTrainer ──────────────────────────

    def reset(self, episode_id, seed, market_cid=None):
        resp = self._request(
            {"cmd": "reset", "episode_id": str(episode_id),
             "seed": int(seed), "market_cid": market_cid},
            expect="reset")
        parsed = parse_policy_obs(resp)  # fail-closed here
        self._prev_account = None
        self._peak_adj = None
        self._dd_prev = 0
        self._meta = parsed["meta"]
        self._reward_from_account(parsed["account"])
        return parsed

    def step(self, action_id):
        resp = self._request(
            {"cmd": "step", "action": {"policy_v1": int(action_id)}},
            expect="step_result")
        rec = parse_step_response(resp)
        rec["reward"] = (
            self._reward_from_account(rec["account"])
            if "account" in rec else 0.0)
        return rec

    def close(self):
        if self._closed:
            return None
        self._closed = True
        try:
            self._t.send({"cmd": "close"})
            resp = self._t.recv()
            return resp if resp.get("type") == "closed" else None
        except AdapterError:
            return None
        finally:
            self._t.close()
