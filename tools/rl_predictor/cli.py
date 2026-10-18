"""CLI for rl_predictor (stdlib only, no external dependencies).

Subcomandos:
  contract           print the contract v1 (features/actions)
  validate           validate policy.json (+ manifest/online manifest) and
                     emit the activation verdict
  check-driver       probe the JSONL driver: does it produce contract v1?
  make-fixture       deterministic parity fixture (NOT deployable)
  train              train from recorded transitions v1 (schema
                     rl_predictor.transitions.v1); one update per
                     offline cohort; a fixture-marked file is rejected
  snapshot           build/verify migration_stage

Exit codes: 0 ok / 1 usage error / 2 invalid artifact or data /
3 unsupported contract or negative verdict / 4 driver/IO error.
"""

import argparse
import hashlib
import json
import sys

from . import contract as C
from .artifact import (
    ArtifactError,
    check_activation,
    load_policy,
    sha256_file,
    validate_manifest_dict,
    validate_online_manifest_dict,
)
from .env_adapter import (
    AdapterError,
    ContractUnsupportedError,
    JsonlSubprocess,
    V1EnvAdapter,
    build_handoff,
    inspect_reset_response,
)
from .scaler import FeatureScaler
from .trainer import ReinforceTrainer

TRANSITIONS_SCHEMA = "rl_predictor.transitions.v1"


def _out(obj):
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False,
              sort_keys=True)
    sys.stdout.write("\n")


def cmd_contract(_a):
    _out({
        "feature_schema": C.FEATURE_SCHEMA,
        "algorithm": C.POLICY_ALGORITHM,
        "feature_names": list(C.FEATURE_NAMES),
        "action_names": list(C.ACTION_NAMES),
    })
    return 0


def cmd_validate(a):
    try:
        policy, _p, _s = load_policy(a.policy)
    except (ArtifactError, OSError) as e:
        _out({"verdict": "invalid", "errors": [str(e)]})
        return 2
    manifest = online = None
    if a.manifest:
        try:
            with open(a.manifest, encoding="utf-8") as f:
                manifest = validate_manifest_dict(json.load(f))
        except (ArtifactError, OSError, json.JSONDecodeError) as e:
            _out({"verdict": "invalid", "errors": [f"manifest: {e}"]})
            return 2
    if a.online_manifest:
        try:
            with open(a.online_manifest, encoding="utf-8") as f:
                online = validate_online_manifest_dict(json.load(f))
        except (ArtifactError, OSError, json.JSONDecodeError) as e:
            _out({"verdict": "invalid",
                  "errors": [f"online manifest: {e}"]})
            return 2
    psha = sha256_file(a.policy)
    ok, reason = check_activation(
        policy, manifest_dict=manifest, online_manifest_dict=online,
        policy_sha256=psha)
    _out({"verdict": "activatable" if ok else "not_activatable",
          "status": policy["status"], "reason": reason,
          "policy_sha256": psha})
    return 0 if ok else 3


def cmd_check_driver(a):
    try:
        t = JsonlSubprocess([a.bin, "--tape", a.tape],
                            timeout_s=a.timeout)
    except OSError as e:
        _out({"supported": False, "stage": "spawn", "error": str(e),
              "handoff": build_handoff(["spawn"])})
        return 4
    try:
        hello = t.recv()
        if hello.get("type") != "hello":
            _out({"supported": False, "stage": "handshake",
                  "error": f"expected hello: {hello}",
                  "handoff": build_handoff(["hello"])})
            return 4
        t.send({"cmd": "reset", "episode_id": "probe-v1",
                "seed": 0, "market_cid": None})
        resp = t.recv()
        ok, missing, notes = inspect_reset_response(resp)
        verdict = {
            "supported": ok, "stage": "reset",
            "fidelity": hello.get("fidelity"),
            "schema_version": hello.get("schema_version"),
            "missing": missing, "notes": notes,
        }
        if not ok:
            verdict["handoff"] = build_handoff(missing)
        _out(verdict)
        return 0 if ok else 3
    except AdapterError as e:
        _out({"supported": False, "stage": "io", "error": str(e),
              "handoff": build_handoff(["io"])})
        return 4
    finally:
        try:
            t.send({"cmd": "close"})
            t.recv()
        except Exception:
            pass
        t.close()


def cmd_make_fixture(a):
    """Python/Rust parity fixture: deterministic cohort + update
    expected. Artifact marked fixture=true -> never promotable."""
    from .parity import full_update
    from .policy import LinearPolicy
    from .artifact import atomic_write_json, validate_policy_dict
    import os
    import random

    rng = random.Random(a.seed)
    # Small deterministic initial weights (not zero, so the gradient is real).
    w0 = [[rng.uniform(-0.05, 0.05) for _ in range(C.NUM_FEATURES)]
          for _ in range(C.NUM_ACTIONS)]
    b0 = [rng.uniform(-0.05, 0.05) for _ in range(C.NUM_ACTIONS)]
    pol = LinearPolicy(w0, b0)
    mask_full = (1 << C.NUM_ACTIONS) - 1
    # Partial mask: some steps without SELL to exercise masking.
    mask_no_sell = mask_full & ~(
        (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8))
    cohort = []
    for ep_i in range(a.episodes):
        ep = []
        for t in range(a.steps):
            x = [rng.uniform(-2, 2) for _ in range(C.NUM_FEATURES)]
            mask = mask_no_sell if t % 3 == 2 else mask_full
            probs = pol.probs(x, mask)
            erng = random.Random(f"{a.seed}:{ep_i}:{t}")
            u = erng.random()
            acc, act = 0.0, 0
            for j, p in enumerate(probs):
                if p > 0:
                    acc += p
                    if u < acc:
                        act = j
                        break
            ep.append({
                "x_std": x, "mask": mask, "action": act,
                "behavior_prob": probs[act],
                "reward": rng.uniform(-0.02, 0.03),
            })
        cohort.append(ep)
    new_pol, rep = full_update(cohort, pol, baseline=0.0, lr=a.lr,
                               grad_clip=a.grad_clip)
    os.makedirs(a.out, exist_ok=True)
    fixture_policy = {
        "schema_version": C.POLICY_SCHEMA_VERSION,
        "algorithm": C.POLICY_ALGORITHM,
        "feature_schema": C.FEATURE_SCHEMA,
        "feature_names": list(C.FEATURE_NAMES),
        "action_names": list(C.ACTION_NAMES),
        "predictor_sha256": "0" * 64,
        "scaler": {"mean": [0.0] * C.NUM_FEATURES,
                   "scale": [1.0] * C.NUM_FEATURES},
        "weights": w0, "bias": b0,
        "training": {
            "run_id": f"parity-fixture-seed{a.seed}",
            "data_manifest_sha256": "0" * 64,
            "split_manifest_sha256": "0" * 64,
            "seed": a.seed, "completed_episodes": a.episodes,
            "updates": 1, "fixture": True,
        },
        "status": "candidate",
    }
    validate_policy_dict(fixture_policy)
    atomic_write_json(os.path.join(a.out, "fixture_policy_init.json"),
                      fixture_policy)
    with open(os.path.join(a.out, "fixture_cohort.jsonl"), "w",
              encoding="utf-8") as f:
        for ep in cohort:
            f.write(json.dumps({"steps": ep}) + "\n")
    expected = {
        "formula": "parity.py::full_update — see module docstring",
        "baseline": rep["baseline"], "grad_norm_preclip":
        rep["grad_norm_preclip"], "clip_scale": rep["clip_scale"],
        "lr": rep["lr"], "grad_clip": rep["grad_clip"],
        "n_steps": rep["n_steps"],
        "episode_returns": rep["episode_returns"],
        "updated_weights": rep["weights"], "updated_bias": rep["bias"],
    }
    atomic_write_json(os.path.join(a.out, "fixture_expected_update.json"),
                      expected)
    _out({"written": ["fixture_policy_init.json", "fixture_cohort.jsonl",
                      "fixture_expected_update.json"],
          "n_steps": rep["n_steps"], "fixture": True,
          "note": "explicit fixture: NOT promotable or deployable"})
    return 0


def _load_transitions(path):
    """Load rl_predictor.transitions.v1: header + episodes.

    Each episode line: {"episode_id","steps":[{obs_raw[22],mask,
    action,behavior_prob,reward}...], "trainable":bool?}.
    Header: {"schema","source"}; source 'fixture' -> hard rejection."""
    episodes = []
    header = None
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ArtifactError(f"{path}:{ln} invalid JSON: {e}")
            if header is None:
                if obj.get("schema") != TRANSITIONS_SCHEMA:
                    raise ArtifactError(
                        f"{path}: schema {obj.get('schema')!r} != "
                        f"{TRANSITIONS_SCHEMA!r}")
                header = obj
                if header.get("source") == "fixture":
                    raise ArtifactError(
                        "transitions source=fixture: fixtures do not "
                        "train a deliverable policy")
                continue
            steps = []
            if not obj.get("trainable", True):
                continue
            for i, s in enumerate(obj.get("steps", [])):
                obs = C.check_observation_vector(s.get("obs_raw"))
                mask = C.check_mask(s.get("mask"))
                act = s.get("action")
                if not isinstance(act, int) or not (0 <= act <
                                                  C.NUM_ACTIONS):
                    raise ArtifactError(f"{path}:{ln} step {i} action "
                                        f"{act!r} out of 0..8")
                reward = s.get("reward")
                if not isinstance(reward, (int, float)) \
                        or isinstance(reward, bool):
                    raise ArtifactError(f"{path}:{ln} step {i} reward "
                                        "not numeric")
                steps.append({"obs_raw": obs, "mask": mask,
                              "action": act,
                              "behavior_prob": float(
                                  s.get("behavior_prob", 1.0)),
                              "reward": float(reward)})
            if steps:
                episodes.append({"episode_id": obj.get("episode_id",
                                                       f"line{ln}"),
                                 "steps": steps})
    if not episodes:
        raise ArtifactError(f"{path}: no trainable episodes")
    return header, episodes


def cmd_train(a):
    """One offline update over a recorded cohort (the multi-update
    on-policy loop needs a driver/env; offline the data is already frozen)."""
    try:
        header, episodes = _load_transitions(a.transitions)
    except (ArtifactError, OSError) as e:
        _out({"verdict": "error", "error": str(e)})
        return 2
    # Train-only scaler over the observations of THIS training set.
    obs = [s["obs_raw"] for ep in episodes for s in ep["steps"]]
    try:
        scaler = FeatureScaler.fit(obs)
    except Exception as e:
        _out({"verdict": "error", "error": f"scaler fit: {e}"})
        return 2
    # Initial weights: zero (transparent baseline) or from a previous policy.
    if a.init_policy:
        try:
            init, policy, _sc = load_policy(a.init_policy)
        except (ArtifactError, OSError) as e:
            _out({"verdict": "error", "error": f"init policy: {e}"})
            return 2
        if init.get("training", {}).get("fixture") is True:
            _out({"verdict": "error",
                  "error": "init policy fixture does not train or deliver"})
            return 2
    else:
        from .policy import LinearPolicy
        policy = LinearPolicy()
    trainer = ReinforceTrainer(
        policy, scaler, lr=a.lr, grad_clip=a.grad_clip, seed=a.seed,
        cohort_episodes=len(episodes))
    from .trainer import Episode, policy_fingerprint
    cohort = []
    for ep in episodes:
        e = Episode(ep["episode_id"], policy_fingerprint(trainer.policy),
                    seed=0)
        e.steps = [{
            "x_std": scaler.transform(s["obs_raw"]),
            "mask": s["mask"], "action": s["action"],
            "behavior_prob": s["behavior_prob"],
            "reward": s["reward"],
        } for s in ep["steps"]]
        cohort.append(e)
    try:
        report = trainer.update(cohort)
    except Exception as e:
        _out({"verdict": "error", "error": f"update: {e}"})
        return 2
    # Episodes actually consumed by the offline cohort (honest counter for
    # the artifact training block).
    trainer.completed_episodes = len(episodes)
    import os
    os.makedirs(a.out, exist_ok=True)
    d, psha = trainer.save_checkpoint(
        os.path.join(a.out, "policy.json"),
        predictor_sha256=a.predictor_sha256,
        run_id=a.run_id,
        data_manifest_sha256=a.data_manifest_sha256,
        split_manifest_sha256=a.split_manifest_sha256)
    _out({"verdict": "trained_candidate",
          "policy_sha256": psha, "status": d["status"],
          "report": report,
          "note": "offline: ONE update per recorded cohort; multi-update "
                  "on-policy requires a v1 driver (see check-driver)"})
    return 0


def cmd_snapshot(a):
    from .snapshot import build_snapshot, verify_snapshot
    if a.verify_only:
        ok, problems = verify_snapshot(a.dest)
        _out({"verdict": "verified" if ok else "mismatch",
              "problems": problems})
        return 0 if ok else 3
    try:
        manifest, warnings = build_snapshot(a.src, a.dest)
    except OSError as e:
        _out({"verdict": "error", "error": str(e)})
        return 4
    ok, problems = verify_snapshot(a.dest)
    _out({"verdict": "verified" if ok else "mismatch",
          "files": len(manifest["entries"]),
          "excluded_secret_like": warnings,
          "problems": problems})
    return 0 if ok else 3


def build_parser():
    p = argparse.ArgumentParser(prog="rl_predictor",
                                description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("contract").set_defaults(fn=cmd_contract)

    v = sub.add_parser("validate")
    v.add_argument("--policy", required=True)
    v.add_argument("--manifest")
    v.add_argument("--online-manifest")
    v.set_defaults(fn=cmd_validate)

    c = sub.add_parser("check-driver")
    c.add_argument("--bin", required=True)
    c.add_argument("--tape", required=True)
    c.add_argument("--timeout", type=float, default=15)
    c.set_defaults(fn=cmd_check_driver)

    m = sub.add_parser("make-fixture")
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--episodes", type=int, default=4)
    m.add_argument("--steps", type=int, default=6)
    m.add_argument("--lr", type=float, default=0.05)
    m.add_argument("--grad-clip", type=float, default=1.0)
    m.set_defaults(fn=cmd_make_fixture)

    t = sub.add_parser("train")
    t.add_argument("--transitions", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--predictor-sha256", required=True)
    t.add_argument("--data-manifest-sha256", required=True)
    t.add_argument("--split-manifest-sha256", required=True)
    t.add_argument("--init-policy")
    t.add_argument("--run-id", default="offline-run")
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("snapshot")
    s.add_argument("--src", required=True, help="root of the BOT (read-only)")
    s.add_argument("--dest", required=True)
    s.add_argument("--verify-only", action="store_true")
    s.set_defaults(fn=cmd_snapshot)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
