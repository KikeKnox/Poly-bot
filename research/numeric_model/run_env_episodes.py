#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Episodic run + learning over `hrf_research_env` (G5 of the numeric
environment closure; policy repaired per addendum v2 — "RL policy").

Policy: PER-ACTION softmax with weight blocks per class
(episodic REINFORCE with a moving-average return baseline). It is NOT a
logistic regression (there are no labels or supervised fitting) nor a random
policy: the weights are updated between episodes with the
`reward.total` signal of the driver's reconciled ledger.

    logit(a) = w_cls[class(a)]·φ_state(obs) + u·ψ_action(a)

- `class(a)`: hold | buy_yes | buy_no | sell_yes | sell_no | cancel
  (| buy_unknown | sell_unknown if the side is not derivable).
- `φ_state` = [1.0] + observation features (FEATS, None → 0.0).
- `ψ_action` = [price, size/SIZE_REF] for buy/sell; zeros for
  hold/cancel.
- `u` is shared across kinds; `w_cls` is one block per class, so the
  state contribution is NOT cancelled across actions (defect:
  the earlier version concatenated the same state features to all
  actions and the softmax cancelled them → zero gradient over state,
  and buy YES vs buy NO with equal price/size were indistinguishable).

YES/NO side identity (fix): EXPLICIT mapping
`token_id → "yes"|"no"` built from the tape's `market_meta` events
(`payload.yes_token_id`/`no_token_id`), read once at start and
reused across all steps and episodes. NEVER inferred from the order
of `legal_actions`: the driver may omit a token (no ask → no Buy,
no inventory/bid → no Sell) and the relative order carries no semantics —
with only the NO token legal, a parity inference would classify it
as `buy_yes`. A token with no known meta or with contradictory metas →
class `*_unknown` (honest, never a fabricated side).

By contract F design, one `hrf_research_env` process can chain
sequential episodes with `reset` (fresh engine per reset, same process
— `REPLAY_NOW_MS` is global, so there are never two simultaneous engines).

The JSON report separates three independent claims (addendum v2):
- `smoke_ipc`: the protocol/handshake/steps worked.
- `weights_update`: technical evidence that Δw ≠ 0 when there is a
  non-zero return (L2 norm of the weight change).
- `oos_evidence`: ALWAYS "insufficient" in this runner — a short
  in-sample run does not allow declaring useful learning. The policy
  may learn to abstain; transactions are not forced.

Artifacts:
- `--episodes-out`: JSONL, one line per episode with its full `StepRecord`s
  (contract D) + the driver's `closed` digest.
- `--report-out`: JSON with the weight trajectory, returns per
  episode, the `hello` fidelity declaration and the three
  evidence blocks above.

Stdlib only.
"""

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import ResearchEnv, EnvError  # noqa: E402

# Observation features used by the policy (fixed, documented
# subset; None → 0.0 — the real scaler imputes, here we keep the
# policy minimal and explicit). `own_open_orders` is emitted by
# `model_observer::extract_features` (0.0 if the episode has no
# live own orders).
FEATS = [
    "yes_mid",
    "yes_spread",
    "yes_imbalance",
    "time_left_frac",
    "spot_ret_1",
    "flow_imbalance",
    "own_open_orders",
]

# Action classes: one block of state weights per class. `*_unknown`
# covers buy/sell actions whose token_id cannot be mapped to YES/NO.
KINDS = [
    "hold",
    "buy_yes",
    "buy_no",
    "sell_yes",
    "sell_no",
    "cancel",
    "buy_unknown",
    "sell_unknown",
]

N_STATE = 1 + len(FEATS)  # dimension of φ_state (bias + features)
N_ACT = 2                 # dimension of ψ_action: [price, normalized_size]

# Reference scale to normalize `size` in ψ_action (prices already
# travel in [0,1]; the size in shares is bounded to a comparable
# magnitude so that u is not dominated by a single component).
SIZE_REF = 100.0

ALPHA = 0.05  # policy gradient learning rate


def _f(x):
    """Numeric normalization: float(x) with None/non-numeric → 0.0."""
    try:
        return float(x) if x is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _tag_body(action):
    """(tag, body) of a wire action: "hold" | {"buy"/"sell"/"cancel":
    {...}} | {"hold": ...}. Returns (None, None) if unrecognizable."""
    if isinstance(action, str):
        return action, None
    if isinstance(action, dict) and len(action) == 1:
        (tag, body), = action.items()
        return tag, body
    return None, None


def phi_state(obs):
    """φ_state = [1.0] + observation features (None → 0.0)."""
    feats = obs.get("features") or {}
    return [1.0] + [_f(feats.get(k)) for k in FEATS]


def sides_map_from_meta(tape_path):
    """token_id → "yes"|"no" map from the tape's `market_meta`.

    Reads the `market_meta` events (`payload.yes_token_id` /
    `payload.no_token_id`) — explicit metadata of the capture
    protocol, never the order of `legal_actions`. Returns
    `(sides, conflicts)`; a token with contradictory metas is left out
    of the mapping and in `conflicts` → its actions fall into `*_unknown`.
    """
    p = Path(tape_path)
    paths = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    sides, conflicts = {}, set()
    for fp in paths:
        with open(fp, "rb") as f:
            for raw in f:
                if b'"market_meta"' not in raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if ev.get("kind") != "market_meta":
                    continue
                pl = ev.get("payload") or {}
                for tok, side in ((pl.get("yes_token_id"), "yes"),
                                  (pl.get("no_token_id"), "no")):
                    if tok is None:
                        continue
                    prev = sides.get(tok)
                    if prev is None and tok not in conflicts:
                        sides[tok] = side
                    elif prev is not None and prev != side:
                        # Contradictory metas: no reliable identity.
                        del sides[tok]
                        conflicts.add(tok)
    return sides, sorted(conflicts)


def action_kind(action, sides):
    """Action class: hold | cancel | {buy,sell}_{yes,no,unknown}."""
    tag, body = _tag_body(action)
    if tag in ("buy", "sell"):
        tid = body.get("token_id") if isinstance(body, dict) else None
        side = sides.get(tid)
        return f"{tag}_{side}" if side else f"{tag}_unknown"
    if tag in ("hold", "cancel"):
        return tag
    return "hold"  # defensive: unrecognizable action = do not trade


def psi_action(action):
    """ψ_action = [price, size/SIZE_REF] for buy/sell; zeros otherwise."""
    tag, body = _tag_body(action)
    if tag in ("buy", "sell") and isinstance(body, dict):
        return [_f(body.get("price")), _f(body.get("size")) / SIZE_REF]
    return [0.0] * N_ACT


def initial_weights():
    """Zero weights: serializable flat dict
    {"state": {class: [N_STATE]}, "action": [N_ACT]}."""
    return {
        "state": {c: [0.0] * N_STATE for c in KINDS},
        "action": [0.0] * N_ACT,
    }


def logits(w, obs, legal, sides):
    """logit(a) = w_cls[class(a)]·φ_state(obs) + u·ψ_action(a)."""
    phi_s = phi_state(obs)
    ws, u = w["state"], w["action"]
    out = []
    for a in legal:
        wc = ws[action_kind(a, sides)]
        ps = psi_action(a)
        out.append(sum(wi * fi for wi, fi in zip(wc, phi_s))
                   + sum(ui * si for ui, si in zip(u, ps)))
    return out


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    tot = sum(exps)
    return [e / tot for e in exps]


def choose(w, obs, legal, rng, sides):
    """Samples the softmax over the legal actions.

    `sides` is the tape's fixed token_id → "yes"|"no" mapping
    (`sides_map_from_meta`) — identical across all steps, so the
    identity of each token does not depend on which actions are legal at
    a given instant.

    Returns (index, wire_action, probs, cache) where cache =
    (φ_state, kinds, ψs) — what is needed for the REINFORCE gradient without
    recomputing features."""
    phi_s = phi_state(obs)
    kinds = [action_kind(a, sides) for a in legal]
    psis = [psi_action(a) for a in legal]
    u = w["action"]
    scores = [
        sum(wi * fi for wi, fi in zip(w["state"][c], phi_s))
        + sum(ui * si for ui, si in zip(u, ps))
        for c, ps in zip(kinds, psis)
    ]
    probs = softmax(scores)
    x = rng.random()
    acc, i = 0.0, len(probs) - 1
    for j, p in enumerate(probs):
        acc += p
        if x < acc:
            i = j
            break
    return i, legal[i], probs, (phi_s, kinds, psis)


def grad_log_policy(cache, probs, i):
    """∇ log π(a_i) for the block policy.

    - State block of class c: φ_state·(1[c == class(a_i)] −
      Σ_{a: class(a)=c} π(a)) — only the chosen class's block moves
      (minus the class-probability-weighted mean).
    - Action block u: ψ(a_i) − Σ_a π(a)·ψ(a).

    Returns (g_state: {class: [N_STATE]}, g_u: [N_ACT]).
    """
    phi_s, kinds, psis = cache
    mass = {}
    for p, c in zip(probs, kinds):
        mass[c] = mass.get(c, 0.0) + p
    cls_i = kinds[i]
    g_state = {}
    for c in KINDS:
        coef = (1.0 if c == cls_i else 0.0) - mass.get(c, 0.0)
        g_state[c] = [coef * f for f in phi_s]
    mean_psi = [sum(p * ps[j] for p, ps in zip(probs, psis))
                for j in range(N_ACT)]
    g_u = [psis[i][j] - mean_psi[j] for j in range(N_ACT)]
    return g_state, g_u


def apply_gradient(w, g_state, g_u, adv, alpha=ALPHA):
    """w += alpha · adv · ∇log π (REINFORCE with advantage)."""
    f = alpha * adv
    if f == 0.0:
        return
    for c, g in g_state.items():
        wc = w["state"][c]
        for j in range(len(wc)):
            wc[j] += f * g[j]
    for j in range(N_ACT):
        w["action"][j] += f * g_u[j]


def delta_w_norm(w, w0):
    """L2 norm of w − w0 over all blocks."""
    s = 0.0
    for c in KINDS:
        s += sum((a - b) ** 2
                 for a, b in zip(w["state"][c], w0["state"][c]))
    s += sum((a - b) ** 2 for a, b in zip(w["action"], w0["action"]))
    return math.sqrt(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", required=True)
    ap.add_argument("--market-cid", default=None)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--labels-db", default=None)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--episodes-out", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--bin", default=None)
    args = ap.parse_args()

    extra = ["--max-steps", str(args.max_steps)]
    if args.labels_db:
        extra += ["--labels-db", args.labels_db]

    # token_id → side map from the tape's market_meta — fixed for
    # the whole run (persistent identity across steps and episodes).
    sides, side_conflicts = sides_map_from_meta(args.tape)

    w = initial_weights()
    w0 = copy.deepcopy(w)
    baseline = 0.0
    smoke = {
        "ok": False,
        "handshake": False,
        "completed_episodes": 0,
        "total_steps": 0,
        "errors": [],
        "closed_ok": False,
        "reconciled": None,
    }
    report = {
        "method": "episodic REINFORCE, per-action softmax with blocks "
                  "per class (w_cls·φ_state + u·ψ_action; moving-average "
                  "return baseline; alpha=%.3f)" % ALPHA,
        "features": FEATS,
        "kinds": KINDS,
        "side_derivation": "market_meta from the tape: explicit yes_token_id/"
                           "no_token_id (no inference from the "
                           "order of legal_actions; token without meta or "
                           "contradictory → *_unknown)",
        "episodes": [],
        "meta_sides": {"mapped_tokens": len(sides),
                       "conflicts": side_conflicts},
    }

    with ResearchEnv(args.tape, bin_path=args.bin, extra_args=extra,
                     timeout_s=600) as env:
        smoke["handshake"] = True
        report["hello"] = {"schema_version": env.schema_version,
                           "fidelity": env.fidelity}
        ep_logs = []
        for ep in range(args.episodes):
            ep_id = f"ep-{ep:02d}-seed{args.seed}"
            rng = random.Random(args.seed * 1000 + ep)
            try:
                r0 = env.reset(ep_id, seed=args.seed,
                               market_cid=args.market_cid)
            except EnvError as e:
                smoke["errors"].append(
                    {"episode_id": ep_id, "phase": "reset",
                     "error": str(e), "fatal": e.fatal})
                if e.fatal:
                    break
                continue
            obs, legal = r0["observation"], r0["legal_actions"]
            steps, grads, ret = [], [], 0.0
            ok_episode = True
            while True:
                if not legal:
                    # No mask (should not happen: Hold is always
                    # legal) — forces hold with a degenerate softmax.
                    legal = ["hold"]
                i, action_wire, probs, cache = choose(
                    w, obs, legal, rng, sides)
                try:
                    res = env.step(action_wire)
                except EnvError as e:
                    steps.append({"protocol_error": str(e),
                                  "fatal": e.fatal})
                    smoke["errors"].append(
                        {"episode_id": ep_id, "phase": "step",
                         "error": str(e), "fatal": e.fatal})
                    ok_episode = not e.fatal
                    break
                st = res["step"]
                steps.append(st)
                rt = _f(st.get("reward", {}).get("total"))
                ret += rt
                grads.append((cache, probs, i))
                if st.get("terminal"):
                    break
                obs, legal = res["observation"], res["legal_actions"]

            # REINFORCE update when the episode closes.
            adv = ret - baseline
            for cache, probs, i in grads:
                g_state, g_u = grad_log_policy(cache, probs, i)
                apply_gradient(w, g_state, g_u, adv)
            baseline += (ret - baseline) / (ep + 1)

            smoke["total_steps"] += len(steps)
            if ok_episode and steps and not steps[-1].get(
                    "protocol_error"):
                smoke["completed_episodes"] += 1
            ep_logs.append({"episode_id": ep_id, "seed": args.seed,
                            "market_cid": args.market_cid,
                            "n_steps": len(steps), "return": ret,
                            "end_reason": (steps[-1] or {}).get("end_reason")
                            if steps else None,
                            "steps": steps})
            report["episodes"].append({
                "episode_id": ep_id, "n_steps": len(steps),
                "return": ret, "advantage": adv,
                "end_reason": (steps[-1] or {}).get("end_reason")
                if steps else None})
            if not ok_episode:
                break

        try:
            close = env.close()
        except EnvError as e:
            close = None
            smoke["errors"].append({"phase": "close", "error": str(e),
                                     "fatal": e.fatal})
        smoke["closed_ok"] = close is not None
        smoke["reconciled"] = (close or {}).get("reconciled")
        smoke["ok"] = (smoke["handshake"]
                       and smoke["completed_episodes"] > 0
                       and smoke["closed_ok"]
                       and not any(e.get("fatal") for e in
                                   smoke["errors"]))
        report["closed"] = close
        report["weights_final"] = w

    # Separate evidence blocks (addendum v2): smoke_ipc only
    # certifies the protocol; weights_update certifies that Δw moves
    # with a non-zero return; oos_evidence is never declared here.
    report["smoke_ipc"] = smoke
    report["weights_update"] = {
        "delta_w_norm": delta_w_norm(w, w0),
        "episodes_with_return": sum(
            1 for e in report["episodes"] if e["return"] != 0.0),
        "episodes_total": len(report["episodes"]),
    }
    report["oos_evidence"] = {
        "verdict": "insufficient",
        "reason": f"{len(report['episodes'])} episodes over a single "
        "in-sample tape: without a temporal holdout or statistical "
        "significance, useful OOS learning is not declared",
    }

    with open(args.episodes_out, "w") as f:
        for e in ep_logs:
            f.write(json.dumps(e) + "\n")
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({
        "episodes": len(ep_logs),
        "returns": [e["return"] for e in report["episodes"]],
        "fidelity": report["hello"]["fidelity"],
        "reconciled": (close or {}).get("reconciled"),
        "smoke_ipc_ok": smoke["ok"],
        "delta_w_norm": report["weights_update"]["delta_w_norm"],
    }))


if __name__ == "__main__":
    main()
