"""Policy artifacts (policy file, paper manifest and online manifest).

- `policy.json`: schema_version=1, algorithm='reinforce_linear_v1',
  feature_schema/feature_names/action_names EXACTOS, predictor_sha256,
  scaler{mean,scale}, weights[9][22], bias[9], training{...},
  status ∈ {candidate, paper_approved, online_candidate}.
- `manifest.json`: schema_version=1, policy_sha256 (bytes of the file),
  predictor_sha256, feature_schema, approval{scope:'paper',
  report_sha256}.
- `online_manifest.json`: parent+hash+update chain.
  root = the startup paper_approved policy; each link ties
  parent_sha256 -> child_sha256 with a monotonic update_index, update
  config and counters. Online descendants do NOT pose as
  paper_approved: promotion is recorded as online_candidate.

Fail-closed validation: dimensions, finiteness, names/order, hashes and
predictor compatibility are checked BEFORE admitting. candidate is only
for test/evaluation; activation requires paper_approved + manifest or
online_candidate + a valid chain. An artifact marked fixture is never
promotable — status != candidate is a direct rejection.
"""

import hashlib
import json
import math
import os
import tempfile

from .contract import (
    ACTION_NAMES,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    NUM_ACTIONS,
    NUM_FEATURES,
    ONLINE_MANIFEST_SCHEMA_VERSION,
    POLICY_ALGORITHM,
    POLICY_SCHEMA_VERSION,
    STATUS_CANDIDATE,
    STATUS_ONLINE_CANDIDATE,
    STATUS_PAPER_APPROVED,
    VALID_STATUSES,
    ContractError,
)
from .policy import LinearPolicy
from .scaler import FeatureScaler

ONLINE_MANIFEST_KIND = "rl_predictor.online_manifest.v1"
TRAINING_KEYS = (
    "run_id",
    "data_manifest_sha256",
    "split_manifest_sha256",
    "seed",
    "completed_episodes",
    "updates",
)


class ArtifactError(ContractError):
    pass


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path, obj):
    """Atomic tmp+rename write: a checkpoint is never published
    half-written."""
    path = str(path)
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _sha256_str(v, field):
    if not isinstance(v, str) or len(v) != 64 or any(
            c not in "0123456789abcdef" for c in v):
        raise ArtifactError(f"{field}: invalid sha256 hex: {v!r}")
    return v


def _finite(x, field):
    if not isinstance(x, (int, float)) or isinstance(x, bool) \
            or not math.isfinite(x):
        raise ArtifactError(f"{field}={x!r} not finite")
    return float(x)


def validate_policy_dict(p):
    """Validate a policy.json dict. Returns the normalized dict.

    Errors are aggregated: raises ArtifactError listing ALL the problems
    found, not just the first one."""
    errs = []

    def need(cond, msg):
        if not cond:
            errs.append(msg)

    need(isinstance(p, dict), "policy is not a JSON object")
    if not isinstance(p, dict):
        raise ArtifactError("; ".join(errs))
    need(p.get("schema_version") == POLICY_SCHEMA_VERSION,
         f"schema_version={p.get('schema_version')!r} != "
         f"{POLICY_SCHEMA_VERSION}")
    need(p.get("algorithm") == POLICY_ALGORITHM,
         f"algorithm={p.get('algorithm')!r} != {POLICY_ALGORITHM!r}")
    need(p.get("feature_schema") == FEATURE_SCHEMA,
         f"feature_schema={p.get('feature_schema')!r} != {FEATURE_SCHEMA!r}")
    need(p.get("feature_names") == list(FEATURE_NAMES),
         "feature_names does not exactly match the contract "
         "(literal order, 22)")
    need(p.get("action_names") == list(ACTION_NAMES),
         "action_names does not exactly match the contract "
         "(literal order, 9)")
    st = p.get("status")
    need(st in VALID_STATUSES, f"status={st!r} not accepted: {VALID_STATUSES}")
    try:
        _sha256_str(p.get("predictor_sha256"), "predictor_sha256")
    except ArtifactError as e:
        errs.append(str(e))
    tr = p.get("training")
    need(isinstance(tr, dict), "training missing or not an object")
    if isinstance(tr, dict):
        for k in TRAINING_KEYS:
            need(k in tr, f"training.{k} missing")
        ep, up = tr.get("completed_episodes"), tr.get("updates")
        need(isinstance(ep, int) and ep > 0,
             f"training.completed_episodes={ep!r} must be an integer > 0")
        need(isinstance(up, int) and up > 0,
             f"training.updates={up!r} must be an integer > 0")
        need(isinstance(tr.get("seed"), int), "training.seed must be an integer")
        for k in ("data_manifest_sha256", "split_manifest_sha256"):
            try:
                _sha256_str(tr.get(k), f"training.{k}")
            except ArtifactError as e:
                errs.append(str(e))
        if tr.get("fixture") is True and st != STATUS_CANDIDATE:
            errs.append(
                "explicit fixture artifact is not promotable: status must "
                "be 'candidate'")
    w, b = p.get("weights"), p.get("bias")
    need(isinstance(w, list) and len(w) == NUM_ACTIONS,
         f"weights must be [{NUM_ACTIONS}][{NUM_FEATURES}]")
    need(isinstance(b, list) and len(b) == NUM_ACTIONS,
         f"bias must be [{NUM_ACTIONS}]")
    if isinstance(w, list) and len(w) == NUM_ACTIONS:
        for a in range(NUM_ACTIONS):
            row = w[a]
            if not (isinstance(row, list) and len(row) == NUM_FEATURES):
                errs.append(f"weights[{a}] does not have {NUM_FEATURES} columns")
                break
            for f_, x in enumerate(row):
                try:
                    _finite(x, f"weights[{a}][{f_}]")
                except ArtifactError as e:
                    errs.append(str(e))
                    break
    if isinstance(b, list) and len(b) == NUM_ACTIONS:
        for a, x in enumerate(b):
            try:
                _finite(x, f"bias[{a}]")
            except ArtifactError as e:
                errs.append(str(e))
                break
    try:
        FeatureScaler.from_dict(p.get("scaler"))
    except Exception as e:
        errs.append(f"invalid scaler: {e}")
    if errs:
        raise ArtifactError("; ".join(errs))
    return p


def policy_to_dict(policy, scaler, *, predictor_sha256, training, status,
                 run_sha256=None):
    """Build a validated policy.json dict.

    Never fabricates paper_approved: the caller may only pass statuses from
    VALID_STATUSES and a fixture is forced to candidate by validation."""
    p = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "algorithm": POLICY_ALGORITHM,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "predictor_sha256": _sha256_str(predictor_sha256,
                                        "predictor_sha256"),
        "scaler": scaler.to_dict(),
        "weights": [list(r) for r in policy.weights],
        "bias": list(policy.bias),
        "training": dict(training),
        "status": status,
    }
    if run_sha256 is not None:
        p["training"]["run_sha256"] = run_sha256
    return validate_policy_dict(p)


def load_policy(path):
    """Load and validate policy.json. Returns (dict, LinearPolicy, FeatureScaler)."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        p = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ArtifactError(f"policy.json is not valid JSON: {e}")
    validate_policy_dict(p)
    return p, LinearPolicy(p["weights"], p["bias"]), \
        FeatureScaler.from_dict(p["scaler"])


def validate_manifest_dict(m):
    errs = []
    if not isinstance(m, dict):
        raise ArtifactError("manifest is not a JSON object")
    if m.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errs.append(f"manifest schema_version={m.get('schema_version')!r}")
    for k in ("policy_sha256", "predictor_sha256"):
        try:
            _sha256_str(m.get(k), f"manifest.{k}")
        except ArtifactError as e:
            errs.append(str(e))
    if m.get("feature_schema") != FEATURE_SCHEMA:
        errs.append(f"manifest feature_schema={m.get('feature_schema')!r}")
    ap = m.get("approval")
    if not isinstance(ap, dict) or ap.get("scope") != "paper":
        errs.append("manifest approval.scope != 'paper'")
    elif ap.get("report_sha256") is not None:
        try:
            _sha256_str(ap["report_sha256"], "manifest.approval.report_sha256")
        except ArtifactError as e:
            errs.append(str(e))
    if errs:
        raise ArtifactError("; ".join(errs))
    return m


def manifest_for_policy(policy_path, *, predictor_sha256, report_sha256=None):
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "policy_sha256": sha256_file(policy_path),
        "predictor_sha256": _sha256_str(predictor_sha256,
                                        "predictor_sha256"),
        "feature_schema": FEATURE_SCHEMA,
        "approval": {"scope": "paper",
                     "report_sha256": report_sha256},
    }


def validate_online_manifest_dict(m):
    """Online chain: paper_approved root + parent->child links
    with a monotonic update_index, config and counters."""
    errs = []
    if not isinstance(m, dict):
        raise ArtifactError("online manifest is not a JSON object")
    if m.get("schema_version") != ONLINE_MANIFEST_SCHEMA_VERSION:
        errs.append(f"online schema_version={m.get('schema_version')!r}")
    if m.get("kind") != ONLINE_MANIFEST_KIND:
        errs.append(f"online kind={m.get('kind')!r}")
    try:
        _sha256_str(m.get("root_policy_sha256"), "root_policy_sha256")
    except ArtifactError as e:
        errs.append(str(e))
    links = m.get("links")
    if not isinstance(links, list) or not links:
        errs.append("online manifest has no links")
    else:
        prev_child = m.get("root_policy_sha256")
        prev_idx = 0
        for i, lk in enumerate(links):
            where = f"links[{i}]"
            if not isinstance(lk, dict):
                errs.append(f"{where} is not an object")
                break
            try:
                _sha256_str(lk.get("parent_sha256"), f"{where}.parent_sha256")
                _sha256_str(lk.get("child_sha256"), f"{where}.child_sha256")
            except ArtifactError as e:
                errs.append(str(e))
                break
            if lk.get("parent_sha256") != prev_child:
                errs.append(
                    f"{where}: parent_sha256 does not chain to the previous "
                    "child (broken chain)")
            if lk.get("child_sha256") == lk.get("parent_sha256"):
                errs.append(f"{where}: child==parent (empty update)")
            idx = lk.get("update_index")
            if not isinstance(idx, int) or idx <= prev_idx:
                errs.append(
                    f"{where}: update_index={idx!r} not monotonic")
            cfg = lk.get("update_config")
            if not isinstance(cfg, dict) or "lr" not in cfg \
                    or "grad_clip" not in cfg or "seed" not in cfg \
                    or "cohort_episodes" not in cfg:
                errs.append(
                    f"{where}: update_config incomplete "
                    "(lr/grad_clip/seed/cohort_episodes)")
            ctr = lk.get("counters")
            if not isinstance(ctr, dict) \
                    or not isinstance(ctr.get("completed_episodes"), int) \
                    or not isinstance(ctr.get("updates"), int):
                errs.append(f"{where}: counters incomplete")
            if lk.get("promoted_as") != STATUS_ONLINE_CANDIDATE:
                errs.append(
                    f"{where}: promoted_as={lk.get('promoted_as')!r} != "
                    f"{STATUS_ONLINE_CANDIDATE!r} (must not pose as "
                    "paper_approved)")
            prev_child = lk.get("child_sha256")
            prev_idx = idx if isinstance(idx, int) else prev_idx
    if errs:
        raise ArtifactError("; ".join(errs))
    return m


def check_activation(policy_dict, *, manifest_dict=None,
                     online_manifest_dict=None, policy_sha256=None):
    """Activation verdict for the strategy.

    Returns (activatable: bool, reason: str). Rules:
    - fixture -> never.
    - paper_approved -> valid paper manifest whose policy_sha256
      matches the file.
    - online_candidate -> valid online chain whose last child is this
      policy and whose root is the startup paper_approved policy.
    - candidate -> test/evaluation only."""
    if policy_sha256 is None:
        policy_sha256 = sha256_bytes(
            json.dumps(policy_dict, sort_keys=True).encode("utf-8"))
    if policy_dict.get("training", {}).get("fixture") is True:
        return False, "fixture artifact: not promotable"
    st = policy_dict.get("status")
    if st == STATUS_CANDIDATE:
        return False, "candidate: test/evaluation only"
    if st == STATUS_PAPER_APPROVED:
        if manifest_dict is None:
            return False, "paper_approved without manifest.json"
        try:
            validate_manifest_dict(manifest_dict)
        except ArtifactError as e:
            return False, f"invalid manifest: {e}"
        if manifest_dict["policy_sha256"] != policy_sha256:
            return False, "manifest policy_sha256 does not match the file"
        if manifest_dict["predictor_sha256"] != \
                policy_dict["predictor_sha256"]:
            return False, "manifest predictor_sha256 != policy"
        return True, "paper_approved + valid manifest"
    if st == STATUS_ONLINE_CANDIDATE:
        if online_manifest_dict is None:
            return False, "online_candidate without online manifest"
        try:
            validate_online_manifest_dict(online_manifest_dict)
        except ArtifactError as e:
            return False, f"invalid online manifest: {e}"
        tip = online_manifest_dict["links"][-1]["child_sha256"]
        if tip != policy_sha256:
            return False, "online manifest tip != this policy"
        return True, "online_candidate + valid parent/hash chain"
    return False, f"status {st!r} not activatable"
