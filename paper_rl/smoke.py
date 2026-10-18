#!/usr/bin/env python3
"""Supervisor of the PAPER smoke run of the online RL runtime (paper_rl profile).

Launches `../launch_rl_paper.sh` as a child process in its own process group (pgid),
waits for readiness via the loopback API + log markers, runs at most
`--seconds` seconds (hard cap 600 = 10 min) and then shuts down in a
controlled way: SIGTERM to the group → wait with timeout → SIGKILL only if the child
does not die. At the end it verifies that no process of its own is left alive.

Output: `data/smoke_<ts>.log` (bot stdout/stderr) and
`data/smoke_<ts>.json` (verifiable summary: what was observed and the result).

Usage:
    python3 smoke.py [--seconds 300]

NOT a service: the bot is STOPPED when it finishes.
"""
import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request

PROFILE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(PROFILE, "..", "launch_rl_paper.sh")
API = "http://127.0.0.1:18766"
# StateStore::resolve_path appends the mode suffix to the configured path.
DB_PATH = os.path.join(PROFILE, "data", "state_paper.sqlite")
HARD_CAP_S = 600  # 10 min maximum — never persistent

def get(path, timeout=3):
    try:
        with urllib.request.urlopen(API + path, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def read_log(log_path):
    try:
        with open(log_path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def db_stats():
    """Executions and realized PnL from the PROFILE's DB, in read-only mode.

    Called AFTER the bot has shut down — never against the repo's historical DB.
    Returns None if the DB does not exist yet.
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        con = sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        cur = con.cursor()
        fills = cur.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        tx = cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(pnl_um), 0) FROM transactions"
        ).fetchone()
        return {
            "fills": fills,
            "transactions": tx[0],
            "realized_pnl_um": tx[1],
        }
    except sqlite3.Error:
        return None
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=300,
                    help="observation duration (max %d)" % HARD_CAP_S)
    ap.add_argument("--build", action="store_true",
                    help="allow the launcher to build before starting")
    args = ap.parse_args()
    run_s = min(max(args.seconds, 30), HARD_CAP_S)

    ts = int(time.time())
    os.makedirs(os.path.join(PROFILE, "data"), exist_ok=True)
    log_path = os.path.join(PROFILE, "data", "smoke_%d.log" % ts)
    json_path = os.path.join(PROFILE, "data", "smoke_%d.json" % ts)

    print("[smoke] profile=%s" % PROFILE)
    print("[smoke] log=%s" % log_path)

    # Do not clobber someone else's bot: the profile's port must be free.
    if get("/healthz") is not None:
        print("[smoke] ABORT: a service already answers on %s — not starting." % API)
        sys.exit(2)

    print("[smoke] launching launcher (own group, max %ds)..." % run_s)

    logf = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", LAUNCHER],
        cwd=PROFILE,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group → kill only that group
    )
    pgid = os.getpgid(proc.pid)

    summary = {
        "ts": ts, "run_seconds_max": run_s, "log": log_path,
        "checks": {}, "markers": {}, "learner_final": None,
        "open_positions": None, "db": None,
        "update": None, "clean_shutdown": False,
        "child_exited": False, "exit_code": None,
    }

    def mark(name, found):
        summary["markers"][name] = bool(found)

    try:
        t0 = time.time()
        ready = False
        while time.time() - t0 < run_s:
            if proc.poll() is not None:
                print("[smoke] the bot exited on its own (exit=%s)" % proc.returncode)
                break
            st = get("/ui/rl_learner")
            if st:
                strat = st.get("strategy") or {}
                learn = st.get("learner") or {}
                summary["checks"]["strategy_active"] = strat.get("active")
                summary["checks"]["policy_sha256"] = strat.get("policy_sha256")
                summary["checks"]["learning_enabled"] = strat.get("learning_enabled")
                summary["checks"]["learner_alive"] = learn.get("alive")
                summary["checks"]["transitions_rx"] = learn.get("transitions_rx")
                summary["checks"]["episodes_completed"] = learn.get("episodes_completed")
                summary["checks"]["updates_applied"] = learn.get("updates_applied")
                summary["checks"]["checkpoints_written"] = learn.get("checkpoints_written")
                if strat.get("active") and learn.get("alive"):
                    ready = True
            tools = get("/ui/tools")
            if tools is not None:
                summary["checks"]["tools_endpoint_ok"] = True
                # model_tool running → predictor delivering
                txt = json.dumps(tools)
                summary["checks"]["model_tool_running"] = \
                    '"model_tool"' in txt and ("Running" in txt or '"running"' in txt.lower())
            # /paper/positions returns OPEN positions — not a count of
            # executions. Real executions are read from the DB after shutdown.
            positions = get("/paper/positions")
            if positions is not None:
                summary["checks"]["paper_positions_endpoint_ok"] = True
            if ready:
                time.sleep(2)
            else:
                time.sleep(1)
        summary["checks"]["ready"] = ready
        summary["learner_final"] = get("/ui/rl_learner")
        summary["open_positions"] = get("/paper/positions")

        log = read_log(log_path)
        mark("bot_starting", "poly-bot starting" in log)
        mark("model_tool_predictor_loaded",
             "model_tool: predictor loaded" in log)
        mark("model_tool_running", "model_tool: starting" in log)
        mark("artifact_approved_loaded",
             "paper_approved policy loaded" in log)
        mark("strategy_runner_started", "strategy runner: starting" in log)
        mark("market_feed_started", "market_feed" in log)
        mark("episode_closed", "rl_predictor: episode closed" in log)
        mark("policy_update", "new policy version published" in log)
        mark("learner_checkpoint_resumed",
             "checkpoint chain recovered" in log)
        mark("gate_warns", "rl_predictor" in log and "rejected" in log.lower())
        mark("intent", "rl_predictor: intent" in log)
        summary["update"] = summary["markers"]["policy_update"]
    finally:
        # Controlled shutdown: SIGTERM to the launcher's GROUP, wait, KILL only if
        # the child does not die. try/finally guarantees the attempt even on errors.
        print("[smoke] shutting down: SIGTERM to group %d..." % pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
            summary["child_exited"] = True
            summary["exit_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            print("[smoke] SIGTERM was not enough after 60s: SIGKILL to group %d" % pgid)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
            summary["child_exited"] = proc.returncode is not None
            summary["exit_code"] = proc.returncode
        # Verification: no process of our own is left in the group
        try:
            os.killpg(pgid, 0)
            summary["clean_shutdown"] = False
            print("[smoke] WARNING: group %d still exists" % pgid)
        except ProcessLookupError:
            summary["clean_shutdown"] = True
            print("[smoke] group %d finished — no processes of our own left" % pgid)
        logf.close()

    log = read_log(log_path)
    mark("shutdown_complete", "shutdown complete" in log)
    mark("learner_stopped", "rl_learner: task stopped" in log)
    # Real executions/transactions/PnL — profile DB, read-only,
    # only after the shutdown is complete.
    summary["db"] = db_stats()

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[smoke] summary=%s" % json_path)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "learner_final"}, indent=2))
    print("[smoke] STOPPED — the bot is not left running.")


if __name__ == "__main__":
    sys.exit(main())
