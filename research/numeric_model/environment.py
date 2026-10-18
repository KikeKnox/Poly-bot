"""Python client of the episodic environment `hrf_research_env` (contract F of
`src/research_contract.rs`, frozen by the integrator).

One subprocess per environment, JSONL protocol over stdin/stdout (one line
= one message). The Rust process is the ONLY owner of the economic state
(Executor → RiskGuard → SimulatedRouter → fill_sink → SQLite :memory:);
Python only proposes `AgentAction` and reads `EnvObservation`/`StepRecord` —
it never keeps a second ledger.

Replay clock isolation (`REPLAY_NOW_MS` is process-global in
Rust): one `hrf_research_env` process per environment — a
subprocess is NEVER shared between episodes that must be independent.

Wire (serde, as serialized by the Rust contract):

- `EnvRequest` (internal tag `cmd`, snake_case):
    {"cmd": "reset", "episode_id": "...", "seed": 0, "market_cid": null}
    {"cmd": "step", "action": <AgentAction>}
    {"cmd": "legal_actions"}
    {"cmd": "close"}
- `AgentAction` (externally tagged enum, snake_case; `Decimal` travels
  as a string — rust_decimal serde default):
    "hold"
    {"buy":  {"token_id": "...", "price": "0.45", "size": "10"}}
    {"sell": {"token_id": "...", "price": "0.50", "size": "5"}}
    {"cancel": {"order_id": "..."}}
- `EnvResponse` (internal tag `type`, snake_case):
    {"type": "hello", "schema_version": 1, "fidelity": "certified"|"reduced"}
    {"type": "reset", "observation": {...}, "legal_actions": [...]}
    {"type": "step_result", "step": {StepRecord}, "observation": {...}|null,
     "legal_actions": [...]}
    {"type": "legal_actions", "legal_actions": [...]}
    {"type": "closed", "episode_id": "...", "digest": "...", "reconciled": true}
    {"type": "error", "message": "...", "fatal": true|false}

`fatal=true` invalidates the episode (the environment becomes unusable);
`fatal=false` (e.g. a malformed JSON line) leaves the connection usable.

Stdlib only. `select` (POSIX) is used for the optional read timeout.
"""

import json
import os
import select
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# Default binary relative to the repo root (this file lives in
# research/numeric_model/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = _REPO_ROOT / "target" / "debug" / "hrf_research_env"


class EnvError(RuntimeError):
    """Error reported by the driver (`EnvResponse::Error`).

    `fatal=True` invalidates the episode and the connection; `fatal=False` leaves
    the connection usable (the driver keeps answering)."""

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


class EnvClosed(EnvError):
    """Operation on an already closed environment."""


def hold():
    """Hold action (serde unit → plain string)."""
    return "hold"


def buy(token_id, price, size):
    return {"buy": {"token_id": str(token_id),
                    "price": str(price), "size": str(size)}}


def sell(token_id, price, size):
    return {"sell": {"token_id": str(token_id),
                     "price": str(price), "size": str(size)}}


def cancel(order_id):
    return {"cancel": {"order_id": str(order_id)}}


def normalize_action(action):
    """Normalizes a caller's action to the contract's wire format.

    Accepts: "hold" | {"hold": ...} | {"buy"/"sell"/"cancel": {...}}.
    The `price`/`size` fields are serialized as strings (Decimal)."""
    if isinstance(action, str):
        if action == "hold":
            return "hold"
        raise ValueError(f"unknown action: {action!r}")
    if not isinstance(action, dict) or len(action) != 1:
        raise ValueError(f"malformed action: {action!r}")
    (tag, body), = action.items()
    if tag == "hold":
        return "hold"
    if tag in ("buy", "sell"):
        return {tag: {
            "token_id": str(body["token_id"]),
            "price": str(body["price"]),
            "size": str(body["size"]),
        }}
    if tag == "cancel":
        return {"cancel": {"order_id": str(body["order_id"])}}
    raise ValueError(f"unknown action: {tag!r}")


class ResearchEnv:
    """Episodic environment over an `hrf_research_env` subprocess.

    Typical usage:

        with ResearchEnv(events="tape.jsonl") as env:
            env.reset("ep-1", seed=42)
            while True:
                r = env.step(hold())
                if r["step"]["terminal"]:
                    break

    Parameters:
    - `events`: path to the ResearchEvent JSONL tape (passed as
      `--tape <path>` to the binary — a `<source>/<date>.jsonl` dir or file).
    - `bin_path`: driver executable; by default
      `<repo>/target/debug/hrf_research_env`.
    - `extra_args`: additional driver arguments (e.g.
      `["--labels-db", "..."]`).
    - `timeout_s`: read timeout per response (None = blocking).
      POSIX (uses `select`).
    """

    def __init__(self, events, bin_path=None, extra_args=(), cwd=None,
                 timeout_s=None):
        self._bin = Path(bin_path) if bin_path else DEFAULT_BIN
        self.timeout_s = timeout_s
        self._closed = False
        self._episode = None
        cmd = [str(self._bin), "--tape", str(events)]
        cmd += [str(a) for a in extra_args]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            cwd=str(cwd or _REPO_ROOT),
        )
        try:
            hello = self._recv()
        except Exception:
            self._kill()
            raise
        if hello.get("type") != "hello":
            self._kill()
            raise EnvError(
                f"unexpected handshake (expected 'hello'): {hello}",
                fatal=True)
        self.schema_version = hello.get("schema_version")
        self.fidelity = hello.get("fidelity")

    # ── protocol ─────────────────────────────────────────────────────

    def _send(self, req):
        # `_closed` is checked by the public methods — `close()` sends
        # its own cmd after setting the flag; here all that matters is that the
        # process is still alive.
        if self._proc.poll() is not None:
            raise EnvClosed("the driver process has already exited",
                            fatal=True)
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()

    def _recv(self):
        fd = self._proc.stdout.fileno()
        if self.timeout_s is not None:
            r, _, _ = select.select([fd], [], [], self.timeout_s)
            if not r:
                raise EnvError(
                    f"timeout of {self.timeout_s}s waiting for a response "
                    "from the driver", fatal=True)
        line = self._proc.stdout.readline()
        if not line:
            rc = self._proc.poll()
            raise EnvError(
                "the driver closed stdout without answering "
                f"(exit={rc})", fatal=True)
        try:
            resp = json.loads(line)
        except json.JSONDecodeError as e:
            raise EnvError(
                f"non-JSON response from the driver: {line[:200]!r} ({e})",
                fatal=True)
        if resp.get("type") == "error":
            err = EnvError(resp.get("message", "driver error"),
                           fatal=bool(resp.get("fatal")))
            if err.fatal:
                self._closed = True
            raise err
        return resp

    def _request(self, req, expect):
        if self._closed:
            raise EnvClosed("environment closed", fatal=True)
        self._send(req)
        resp = self._recv()
        t = resp.get("type")
        if t != expect:
            raise EnvError(
                f"unexpected response {t!r} for cmd "
                f"{req.get('cmd')!r} (expected {expect!r})", fatal=True)
        return resp

    # ── environment API ──────────────────────────────────────────────────

    def reset(self, episode_id, seed, market_cid=None):
        """Starts an episode: returns {"observation", "legal_actions"}.

        `seed` + same tape ⇒ same episode byte for byte. `market_cid`
        restricts the episode to one market (None = full universe)."""
        resp = self._request(
            {"cmd": "reset", "episode_id": str(episode_id),
             "seed": int(seed), "market_cid": market_cid},
            expect="reset")
        self._episode = episode_id
        return {"observation": resp["observation"],
                "legal_actions": resp["legal_actions"]}

    def step(self, action):
        """Proposes an action. Returns {"step", "observation",
        "legal_actions"}; `observation` is None when `step.terminal`."""
        resp = self._request(
            {"cmd": "step", "action": normalize_action(action)},
            expect="step_result")
        return {"step": resp["step"],
                "observation": resp.get("observation"),
                "legal_actions": resp.get("legal_actions", [])}

    def legal_actions(self):
        """Legal actions at the current decision point."""
        resp = self._request({"cmd": "legal_actions"},
                             expect="legal_actions")
        return resp["legal_actions"]

    def close(self):
        """Closes the episode (drains the pipeline) and the process.

        Returns {"episode_id", "digest", "reconciled"} or None if it was already
        closed. Idempotent: always leaves the process dead."""
        if self._closed:
            self._kill()
            return None
        self._closed = True
        try:
            if self._proc.poll() is None:
                self._send({"cmd": "close"})
                resp = self._recv()
                if resp.get("type") != "closed":
                    raise EnvError(
                        f"unexpected response to close: {resp}",
                        fatal=True)
                return {"episode_id": resp.get("episode_id"),
                        "digest": resp.get("digest"),
                        "reconciled": resp.get("reconciled")}
            return None
        except EnvError:
            raise
        finally:
            self._kill()

    # ── process lifecycle ────────────────────────────────────────────

    def _kill(self):
        self._closed = True
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:
            pass
        for f in (proc.stdin, proc.stdout):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

    @property
    def closed(self):
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always closes the process, even on an exception from the caller.
        try:
            self.close()
        except EnvError:
            if exc_type is None:
                raise
        return False

    def __del__(self):
        self._kill()
