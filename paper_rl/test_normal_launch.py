#!/usr/bin/env python3
"""TEST of the normal startup channel: `./launch.sh` + hot toggle of
`rl_predictor` through the existing API (`/control/strategy`).

NOT an operational launcher: it is a bounded test showing that the bot
started through the standard entry point (launch.sh copied byte for byte) activates
the RL strategy hot with the real `paper_approved` policy, the
`model_tool` prediction channel already wired in main.rs and the online learner
alive — with no restart and no dedicated scripts.

Isolation (the REAL bot is never touched):
- The sandbox is a temporary dir OUTSIDE the repo: dotenvy finds no `.env`
  in any ancestor and the inherited env is reduced to a whitelist
  (PATH/HOME/CARGO_BUILD_JOBS) — no credentials and no HRF_* overrides.
- `config/` is copied whole EXCEPT `telegram.toml`, which is GENERATED harmless
  (the real file is neither read nor copied).
- `Cargo.toml`, `Cargo.lock`, `src/`, `target/` are SYMLINKS to the repo:
  `cargo build --locked` inside launch.sh resolves the real workspace and
  reuses the already compiled binary (incremental no-op), keeping the
  same entry path and the same binary.
- New DB (`hrf_state_paper.sqlite` in the sandbox), learner checkpoints
  in the sandbox `data/`, API port from the normal config (18766→8766 per TOML).

Sequence: spawn `bash launch.sh` (own group) → poll API → toggle
rl_predictor ON → check /ui/rl_learner (policy active, hash
paper_bootstrap_v1, learner alive) → observe transitions ~window →
toggle OFF → SIGTERM to the group → wait/fallback KILL → summary JSON in
paper_rl/data/. The bot is ALWAYS left stopped on exit (try/finally).

Usage: python3 paper_rl/test_normal_launch.py [--seconds N]   (hard cap 300)
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE = os.path.join(REPO, "paper_rl")
OUT_DIR = os.path.join(PROFILE, "data")
MAX_SECONDS = 300  # hard cap of the smoke run (≤5 min)
POLICY_SHA = "70b1dffc0d5b281d1a6b05799607f56859f59d234ab795684fc9f8c8a0aa376a"
PREDICTOR_SHA = "ef4e4b0860ef9601fb21462539f4d43d2c8af378060eb2dcb63d153502e4c396"


def api_base(config_dir):
    """Reads ONLY bind/port from the copied general.toml (no external deps)."""
    bind, port = "127.0.0.1", 8766
    in_api = False
    with open(os.path.join(config_dir, "general.toml"), encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("["):
                in_api = s == "[api]"
            elif in_api and s.startswith("bind"):
                bind = s.split("=", 1)[1].strip().strip('"')
            elif in_api and s.startswith("port"):
                port = int(s.split("=", 1)[1].strip())
    return f"http://{bind}:{port}"


def http_json(url, method="GET", body=None, timeout=5):
    req = urllib.request.Request(url, method=method)
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("content-type", "application/json")
    else:
        data = None
    with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def free_port(base):
    try:
        with urllib.request.urlopen(base + "/ui/tools", timeout=2) as r:
            return False, f"port busy: another process answered {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"port busy: HTTP {e.code}"
    except Exception:
        return True, "ok"


def build_sandbox(root):
    """Sandbox with the SAME entry path: real launch.sh + real config
    (minus telegram.toml, generated harmless) + symlinks to the source tree."""
    shutil.copy(os.path.join(REPO, "launch.sh"), root)
    for name in ("Cargo.toml", "Cargo.lock", "src", "target"):
        os.symlink(os.path.join(REPO, name), os.path.join(root, name))
    cfg_src = os.path.join(REPO, "config")
    cfg_dst = os.path.join(root, "config")
    # Copy only non-sensitive configuration; never copy secrets
    # in order to delete them afterwards, because the read would already have happened.
    os.makedirs(os.path.join(cfg_dst, "strategies"))
    shutil.copy2(os.path.join(cfg_src, "general.toml"), cfg_dst)
    for name in os.listdir(os.path.join(cfg_src, "strategies")):
        if name.endswith(".toml"):
            shutil.copy2(os.path.join(cfg_src, "strategies", name),
                         os.path.join(cfg_dst, "strategies", name))
    # The real telegram.toml is NEVER copied or read: the loader always
    # merges it, so the sandbox carries a harmless one generated from defaults.
    real_tg = os.path.join(cfg_dst, "telegram.toml")
    if os.path.exists(real_tg):
        os.remove(real_tg)
    with open(real_tg, "w", encoding="utf-8") as f:
        f.write(
            "# HARMLESS telegram.toml of the test sandbox (not a copy of the real one)\n"
            "[telegram]\nfill_alerts = false\nperiodic_summary_hours = 0\n"
        )
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    return cfg_dst


def main():
    seconds = MAX_SECONDS
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--seconds" and i + 1 < len(args):
            seconds = min(int(args[i + 1]), MAX_SECONDS)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = int(time.time())
    summary_path = os.path.join(OUT_DIR, f"normal_launch_{stamp}.json")
    log_path = os.path.join(OUT_DIR, f"normal_launch_{stamp}.log")
    summary = {
        "kind": "normal_launch_toggle",
        "stamp": stamp,
        "seconds": seconds,
        "policy_sha256": POLICY_SHA,
        "predictor_sha256": PREDICTOR_SHA,
    }
    proc = None
    pgid = None
    try:
        sandbox = tempfile.mkdtemp(prefix="hrf_rl_normal_")
        summary["sandbox"] = sandbox
        cfg_dir = build_sandbox(sandbox)
        base = api_base(cfg_dir)
        summary["api"] = base
        ok, why = free_port(base)
        if not ok:
            raise SystemExit(f"ABORT: {why} — not starting over someone else's bot")
        # env whitelist: PATH/HOME for the toolchain; HRF_MODE/CARGO_TARGET_DIR
        # are exported by launch.sh itself. No secrets and no inherited HRF_*.
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "CARGO_BUILD_JOBS": "2",
            "HRF_NO_DOTENV": "1",
        }
        log = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            ["bash", "launch.sh"],
            cwd=sandbox,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)
        summary["pgid"] = pgid

        # 1) Wait for the API (the build is an incremental no-op over the already compiled target).
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit(f"bot died at startup (rc={proc.returncode}); see {log_path}")
            try:
                rows = http_json(base + "/ui/strategies").get("strategies", [])
                if rows:
                    break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("API did not answer within 120s")

        rl = next((r for r in rows if r.get("strategy") == "rl_predictor"), None)
        summary["rl_row_boot"] = rl
        if rl is None:
            raise SystemExit("rl_predictor does not appear in /ui/strategies")
        if rl.get("enabled") or rl.get("running"):
            raise SystemExit("rl_predictor should have started disabled (enabled=false)")

        # 2) Toggle ON through the existing channel.
        resp = http_json(base + "/control/strategy", method="POST",
                         body={"strategy": "rl_predictor", "enabled": True})
        summary["toggle_on"] = resp
        if not resp.get("enabled"):
            raise SystemExit(f"toggle ON rejected: {resp}")

        # 3) /ui/rl_learner: active policy with approved hash + live learner.
        deadline = time.time() + 30
        rl_status = {}
        while time.time() < deadline:
            rl_status = http_json(base + "/ui/rl_learner")
            strat = rl_status.get("strategy", {})
            learner = rl_status.get("learner") or {}
            if strat.get("active") and learner.get("alive"):
                break
            time.sleep(0.5)
        summary["rl_learner_on"] = rl_status
        strat = rl_status.get("strategy", {})
        learner = rl_status.get("learner") or {}
        assert strat.get("active"), f"policy not active: {rl_status}"
        assert strat.get("policy_sha256") == POLICY_SHA, \
            f"unexpected policy hash: {strat.get('policy_sha256')}"
        assert learner.get("alive"), "learner not alive"
        assert learner.get("learning_enabled") or strat.get("learning_enabled"), \
            "learning_enabled must be true"

        # 4) Observation window: model_tool predictions + books →
        #    transitions to the learner. "Learner alive" is NOT enough: the
        #    transitions_rx counter is the evidence that books reach
        #    on_book (real bug: crypto-updown-long universe not declared →
        #    all books skipped by the runner → transitions_rx=0).
        t_end = min(time.time() + seconds - 20, time.time() + seconds)
        max_transitions = 0
        while time.time() < t_end:
            time.sleep(5)
            summary["rl_learner_mid"] = http_json(base + "/ui/rl_learner")
            tools = http_json(base + "/ui/tools").get("tools", [])
            summary["model_tool_row"] = next(
                (t for t in tools if t.get("tool") == "model_tool"), None)
            learner = (summary["rl_learner_mid"].get("learner") or {})
            max_transitions = max(max_transitions,
                                  int(learner.get("transitions_rx") or 0))
        summary["transitions_rx_max"] = max_transitions

        # 5) Toggle OFF through the same channel. Real semantics: OFF activates
        #    the kill switch → the task is PAUSED (running may stay
        #    true) and the learner stays alive — a 200 does NOT mean "stopped".
        resp_off = http_json(base + "/control/strategy", method="POST",
                             body={"strategy": "rl_predictor", "enabled": False})
        summary["toggle_off"] = resp_off
        time.sleep(1)
        rows_off = http_json(base + "/ui/strategies").get("strategies", [])
        summary["rl_row_off"] = next(
            (r for r in rows_off if r.get("strategy") == "rl_predictor"), None)
        summary["rl_learner_final"] = http_json(base + "/ui/rl_learner")
        summary["outcome"] = "ok_toggle_cycle"
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=10)
        summary["bot_rc"] = None if proc is None else proc.returncode
        alive = False
        if pgid is not None:
            try:
                os.killpg(pgid, 0)
                alive = True
            except (ProcessLookupError, PermissionError):
                alive = False
        summary["process_group_alive"] = alive
        summary["stopped"] = not alive
        # Market coverage seen by THIS process (after shutdown, from its
        # own log): brackets registered by discovery. If the provider does not
        # publish brackets in the window, transitions_rx=0 is an honest
        # EXTERNAL blocker, not a defect — it is reported as such.
        try:
            with open(log_path, encoding="utf-8", errors="replace") as lf:
                log_txt = lf.read()
            summary["coverage"] = {
                "brackets_registered": log_txt.count("registered bracket"),
                "model_predictions": log_txt.count("n_predictions"),
            }
        except OSError:
            summary["coverage"] = None
        # Verdict: the normal channel integration is demonstrated ONLY if
        # there was coverage AND transitions; otherwise defect vs lack of data
        # is told apart explicitly.
        if summary.get("outcome") == "ok_toggle_cycle":
            cov = summary.get("coverage") or {}
            if summary.get("transitions_rx_max", 0) > 0:
                summary["outcome"] = "ok_rl_experience_flow"
            elif cov.get("brackets_registered", 0) == 0:
                summary["outcome"] = "blocked_no_market_coverage"
            else:
                summary["outcome"] = "fail_no_transitions_with_coverage"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"summary: {summary_path}\nlog: {log_path}")
    print(json.dumps({k: summary.get(k) for k in (
        "outcome", "bot_rc", "stopped", "transitions_rx_max",
        "rl_row_off")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
