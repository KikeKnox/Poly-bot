#!/usr/bin/env python3
"""FAKE mini-driver that speaks the JSONL protocol of `hrf_research_env`
(contract F of research_contract.rs) for the tests of `environment.py`
— it does not need the real Rust binary.

Emits `hello` on startup and answers `reset`/`step`/`legal_actions`/
`close`. Supports hooks via environment variables to exercise the
error paths:

- `FAKE_ENV_STEPS` (default 3): number of steps until `terminal`.
- `FAKE_ENV_NONFATAL_FIRST_STEP=1`: the first `step` answers
  `{"type":"error","fatal":false}` and the connection stays usable.
- `FAKE_ENV_FATAL_ON_RESET=1`: `reset` answers with a fatal error.
- `FAKE_ENV_BAD_HELLO=1`: the handshake emits a non-hello JSON.
- `FAKE_ENV_FIDELITY`: value of `fidelity` in the hello (default
  "certified").

The `step` returns a synthetic `StepRecord` with `proposed_action` equal
to the received action (echo — the tests verify the exact wire format).
"""

import json
import os
import sys

SCHEMA_VERSION = 1


def _observation(step_idx, market_cid):
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_ts_ms": 1_700_000_000_000 + step_idx * 1_000,
        "market_cid": market_cid,
        "features": {"yes_mid": 0.5 + step_idx * 0.01, "spot": None},
        "inventory": {"cash_usdc": "1000.00", "reserved_usdc": "0",
                      "positions": []},
        "observation_seqs": list(range(step_idx + 1)),
        "quality": 0,
    }


def _legal_actions():
    return [
        "hold",
        {"buy": {"token_id": "tok-yes", "price": "0.47", "size": "10"}},
        {"sell": {"token_id": "tok-yes", "price": "0.50", "size": "5"}},
        {"cancel": {"order_id": "sim-order-1"}},
    ]


def _emit(resp):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def _error(message, fatal):
    _emit({"type": "error", "message": message, "fatal": fatal})


def main():
    max_steps = int(os.environ.get("FAKE_ENV_STEPS", "3"))
    nonfatal_first = os.environ.get("FAKE_ENV_NONFATAL_FIRST_STEP") == "1"
    fatal_reset = os.environ.get("FAKE_ENV_FATAL_ON_RESET") == "1"
    fidelity = os.environ.get("FAKE_ENV_FIDELITY", "certified")

    if os.environ.get("FAKE_ENV_BAD_HELLO") == "1":
        _emit({"type": "error", "message": "no hello", "fatal": True})
    else:
        _emit({"type": "hello", "schema_version": SCHEMA_VERSION,
               "fidelity": fidelity})

    state = {"episode": None, "step_idx": 0, "terminal": False,
             "market_cid": None, "steps_seen": 0}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            # Malformed JSON: NON-fatal error — the connection stays usable.
            _error(f"malformed json: {e}", fatal=False)
            continue
        cmd = req.get("cmd")
        if cmd == "reset":
            if fatal_reset:
                _error("reset aborted (test hook)", fatal=True)
                continue
            state["episode"] = req["episode_id"]
            state["market_cid"] = req.get("market_cid")
            state["step_idx"] = 0
            state["terminal"] = False
            state["steps_seen"] = 0
            _emit({"type": "reset",
                   "observation": _observation(0, state["market_cid"]),
                   "legal_actions": _legal_actions()})
        elif cmd == "step":
            if state["episode"] is None:
                _error("step without reset", fatal=True)
                continue
            if state["terminal"]:
                _error("step after terminal", fatal=True)
                continue
            if nonfatal_first and state["steps_seen"] == 0:
                state["steps_seen"] += 1
                _error("injected non-fatal error (test hook)",
                       fatal=False)
                continue
            idx = state["step_idx"]
            state["step_idx"] += 1
            state["steps_seen"] += 1
            terminal = state["step_idx"] >= max_steps
            state["terminal"] = terminal
            action = req.get("action")
            step = {
                "schema_version": SCHEMA_VERSION,
                "step_idx": idx,
                "observation_seqs": list(range(idx + 1)),
                "decision_ts_ms": 1_700_000_000_000 + idx * 1_000,
                "legal_actions": _legal_actions(),
                "proposed_action": action,
                "action_outcome": (
                    {"executed": {"event_seqs": [idx + 100]}}
                    if action != "hold" else
                    {"executed": {"event_seqs": []}}
                ),
                "inventory": {"cash_usdc": "1000.00",
                              "reserved_usdc": "0", "positions": []},
                "reward": {"price_component": "0.01",
                           "fee_component": "-0.001",
                           "risk_penalty": "0",
                           "total": "0.009"},
                "terminal": terminal,
                "end_reason": "horizon_reached" if terminal else None,
                "quality": 0,
            }
            _emit({"type": "step_result", "step": step,
                   "observation": (
                       None if terminal
                       else _observation(idx + 1, state["market_cid"])),
                   "legal_actions": _legal_actions()})
        elif cmd == "legal_actions":
            _emit({"type": "legal_actions",
                   "legal_actions": _legal_actions()})
        elif cmd == "close":
            _emit({"type": "closed",
                   "episode_id": state["episode"] or "",
                   "digest": "sha256:fake-driver-digest",
                   "reconciled": True})
            return 0
        else:
            _error(f"unknown cmd: {cmd!r}", fatal=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
