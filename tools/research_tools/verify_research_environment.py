#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance verifier for the numeric environment.

A SINGLE command that checks gates G1–G9 over an EXTERNAL run directory
(``--run-dir``, e.g. ``research_runs/example-run``) and the repo. Each gate
returns ``pass`` | ``fail`` | ``pending`` with mechanical evidence (artifact
path, hash, count) — never a hidden exception:

- ``fail``: verified engineering defect (split overlap, tape
  with non-monotonic local_seq, missing fidelity, live config, corrupt
  evidence…).
- ``pending``: insufficient coverage or missing artifact — explicit state,
  neither pass nor fail.
- ``pass``: the mechanical check ran and found no violations.

Gates:

- G1 splits: reads ``model.json`` (or the sibling ``manifest.json``) and verifies that
  the declared splits do not overlap in time and that
  ``split_markets`` does not share markets between partitions.
- G2 capture: counts events per ``source``/``kind`` in
  ``<run>/events/**/*.jsonl``, verifies non-decreasing ``local_seq`` per
  file and ``avail_ts_ms >= recv_ts_ms``, reads ``_recorder_stats.jsonl``
  (declared discarded/deduplicated) and verifies the closed-segment
  manifests (``<run>/manifests/*.json``): declared schema
  (``closed-segment/v1`` or equivalent), key fields (``files``,
  ``utc_window``/``window_ms``, ``totals``/``recorder_stats_final``) and
  ``files[].sha256`` verified against disk when the file exists.
  Missing schema or missing key fields → the manifest is
  ``incomplete`` (pending); a sha256 that does not match → ``corrupt`` (fail).
  Heterogeneous schemas between manifests (one declares a schema and another does not)
  are declared explicitly — they never pass silently.
- G3 ingestion/replay (report contract v2): ``fidelity`` ∈
  {certified, reduced}; ``digest`` non-empty string WITHOUT ``RECON|false``;
  ``reconciled == true`` (absent → fail if the report uses v2 fields,
  pending if it is a pre-contract report); ``external_fidelity`` if present
  must be ``"unidentified"`` (anything else → fail; absent → fail on
  certified, pending on reduced); ``fidelity_reasons`` a list of strings
  (certified requires an empty list; reduced without reasons → pending);
  ``fidelity == "certified"`` requires ``quality`` with clean scope
  (``scope_discarded_deltas == 0`` and ``tokens_with_gap_scope == 0``)
  — certified without quality → fail «certified without quality breakdown»,
  reduced without quality → pending.
- G4 settlement: reads ``settlement_diagnostic`` (actual contract
  name; the legacy ``diagnostic_settlement`` also counts as a
  defect if it is true). ``certified`` + diagnostic/proxy → fail;
  ``certified`` + ``settlement_corrections > 0`` → fail;
  ``reconciled == false`` → fail; ``settlements > 0`` (or records with
  ``state == "applied"``) → pass; explicit pending
  (non-empty ``settlement_pending`` or ``pending`` records) → pending;
  ``settlements == 0`` without declared pending → pending (insufficient).
- G5 environment: ``--env-log`` with steps that carry ``reward`` whose itemized
  sum adds up and consistent ``terminal``/``end_reason``.
- G6 calibration: ``certification_label`` that contains
  ``unidentified``/``unidentified`` → pending (honest
  insufficiency, NEVER a disguised pass); pass requires a clean label AND at least
  one comparison with ``comparability.state == "comparable"``;
  clean label without a valid comparison → fail (contradiction);
  no label → fail.
- G7 trained model: ``metrics_eval.json``/``metrics_train.json`` present with
  n>0 and consistent with each other; ``sha256(model.json)`` == entry in
  ``files[]`` of the sibling ``manifest.json``; the manifest declares
  dataset ``sources`` (path + sha256 + window) and label stats
  (``n`` + real ``known_at`` + ``excluded``). A model without tape
  provenance → pending; a hash that does not match → fail.
- G8 shadow: ``--shadow-dir`` record (``*.jsonl``) read by streaming
  (O(1) memory): ALL ``decision``/``abstention`` rows carry
  ``model_hash == sha256(model.json)`` verified — any row with
  another hash or no hash → fail with a count; ``stats`` lines with
  p50/p95/p99, a window declared within the sources window and a
  declaration of no order emission; ZERO intents by construction.
- G9 integrity: the run-dir lives outside the repo; ``mode == "paper"`` in
  ``general.toml``; the evidence originals
  (``research/data/*/dataset_manifest.json``) match their sha256; git
  reports no modified tracked evidence; the run-dir contains no
  files inside the repo nor copies of ``.env``.

Output: machine JSON on stdout (or ``--out acceptance.json``) + a readable
summary on stderr. Exit status: 0 if there is no ``fail``; 1 if any
engineering gate fails; ``--strict`` turns ``pending`` into an acceptance
failure (exit 1). By default ``pending`` does NOT block the exit code but
is counted and visible — a run can report «engineering ok,
coverage pending» without hiding it.

Usage:
    python3 tools/research_tools/verify_research_environment.py \
        --run-dir research_runs/example-run \
        [--repo PATH] [--model model.json] [--replay-report r.json] \
        [--env-log episodes.jsonl] [--calibration-report metrics.json] \
        [--metrics-dir DIR] [--shadow-dir DIR] [--out acceptance.json] \
        [--strict]
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

VERSION = 2

PASS = "pass"
FAIL = "fail"
PENDING = "pending"

# Sentinel to distinguish «absent field» from «field present as null» —
# contract v2 treats both differently in some fields.
_ABSENT = object()

# Fields that reveal the replay report already follows contract v2.
# If any appears, the other mandatory fields
# that are absent are a defect (fail), not mere insufficiency.
V2_REPLAY_FIELDS = (
    "reconciled", "fidelity_reasons", "external_fidelity", "quality",
)

# IN-SCOPE quality metrics that a `certified` replay must declare
# as zero (v2: zero in-scope degradation).
G3_QUALITY_SCOPE = ("scope_discarded_deltas", "tokens_with_gap_scope")

# Line types of the shadow record (model_observer::ShadowLine) — the
# shadow is structurally incapable of emitting orders; any line that
# looks like an order intent invalidates the gate.
VALID_SHADOW_LINES = {"decision", "make_abstention", "stats"}
FORBIDDEN_SHADOW_LINES = {"order_intent", "intent", "order", "submit"}


def _res(gate, state, evidence=None, details=None):
    return {
        "gate": gate,
        "state": state,
        "evidence": evidence or {},
        "details": list(details or []),
    }


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block_bytes in iter(lambda: f.read(chunk), b""):
            h.update(block_bytes)
    return h.hexdigest()


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"__error__": str(e)}


def _iter_jsonl(path):
    """JSON lines of a .jsonl: yield (obj, None) per valid line and
    (None, reason) for each corrupt line or read error — the
    caller counts/reports them; a bad line never aborts the gate."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), None
                except json.JSONDecodeError:
                    yield None, "non-JSON line"
    except OSError as e:
        yield None, str(e)


def _a_ms(v):
    """Converts a window endpoint to epoch ms: int/float as is,
    ISO-8601 string (with or without «Z») parsed. None if not parseable."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(
                datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
                * 1000
            )
        except ValueError:
            return None
    return None


def _window_ms(val):
    """Normalizes a declared window to a sorted ``(t0_ms, t1_ms)`` —
    accepts a pair of ms or a pair of ISO strings. None if not a valid pair."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        a, b = _a_ms(val[0]), _a_ms(val[1])
        if a is not None and b is not None:
            return (min(a, b), max(a, b))
    return None


def _is_sha256(s):
    return (
        isinstance(s, str) and len(s) == 64
        and all(c in "0123456789abcdefABCDEF" for c in s)
    )


# ── G1 — model splits ──────────────────────────────────────────────────


def check_g1(model_path):
    """Splits without temporal overlap and without shared markets.

    Reads `splits` ([{name, from_ms, to_ms}]) and `split_markets` ({name:
    [cid…]}) from model.json; if the model does not carry them, it looks at the
    sibling `manifest.json`. Overlap or a shared market → fail;
    declared splits without a markets list → partial pending."""
    g = "G1_splits"
    if model_path is None or not Path(model_path).is_file():
        return _res(g, PENDING, {"model": model_path},
                    ["model.json missing — pass --model or place it at "
                     "<run-dir>/model/model.json"])
    data = _load_json(model_path)
    if "__error__" in data:
        return _res(g, FAIL, {"model": str(model_path)},
                    [f"model.json unreadable: {data['__error__']}"])
    manifest = {}
    man_path = Path(model_path).parent / "manifest.json"
    if man_path.is_file():
        manifest = _load_json(man_path)
    splits = data.get("splits") or manifest.get("splits")
    if not splits:
        return _res(g, FAIL, {"model": str(model_path)},
                    ["trained model without declared splits — no time "
                     "boundaries to verify"])
    details = []
    sorted_splits = sorted(splits, key=lambda s: s.get("from_ms", 0))
    overlaps = []
    for i in range(1, len(sorted_splits)):
        prev, cur = sorted_splits[i - 1], sorted_splits[i]
        if cur.get("from_ms", 0) < prev.get("to_ms", 0):
            overlaps.append(
                {
                    "a": prev.get("name"), "b": cur.get("name"),
                    "overlap_ms": prev["to_ms"] - cur["from_ms"],
                }
            )
    sm = data.get("split_markets") or manifest.get("split_markets")
    shared = {}
    if sm:
        names = sorted(sm)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = sorted(set(sm[a]) & set(sm[b]))
                if inter:
                    shared[f"{a}∩{b}"] = inter[:10]
    else:
        details.append(
            "split_markets missing — markets per partition not verifiable"
        )
    evidence = {
        "model": str(model_path),
        "splits": sorted_splits,
        "n_splits": len(sorted_splits),
        "overlaps": overlaps,
        "shared_markets": shared,
        "model_sha256": _sha256(model_path),
    }
    if overlaps or shared:
        return _res(g, FAIL, evidence, details + [
            "temporal overlap between partitions" if overlaps else "",
            "markets shared between partitions" if shared else "",
        ])
    if not sm:
        return _res(g, PENDING, evidence, details + [
            "clean time boundaries; market↔partition membership "
            "not declared in the artifact",
        ])
    return _res(g, PASS, evidence, details)


# ── G2 — event capture ─────────────────────────────────────────────────────


# Tolerances of the physical order in the file (G2): the canonical order of
# contract A is the `local_seq` VALUE (intact and unique per event — the
# loader reorders by it), not the position on disk. The recorder assigns
# seq at production time and enqueues asynchronously, so concurrent
# writes can invert nearby pairs. A small, bounded inversion is
# zero loss — it is reported as known losses; only a
# large permutation (real unrecoverable disorder) fails the gate.
G2_MAX_SHIFT = 256
G2_MAX_INVERSIONS_FRACTION = 0.02

# Keys that declare the schema of a closed-segment manifest
# (`closed-segment/v1` or a declared equivalent).
G2_SCHEMA_KEYS = ("schema", "schema_version")
# Keys that declare the time window of the segment.
G2_WINDOW_KEYS = ("utc_window", "window_ms", "window")


def _manifest_window(m):
    """Window (t0_ms, t1_ms) of a manifest: explicit window_* or
    totals.recv_ts_ms_min/max. None if not declared."""
    for k in G2_WINDOW_KEYS:
        v = _window_ms(m.get(k))
        if v is not None:
            return v
    totals = m.get("totals")
    if isinstance(totals, dict):
        vmin, vmax = totals.get("recv_ts_ms_min"), totals.get("recv_ts_ms_max")
        if (
            isinstance(vmin, (int, float)) and not isinstance(vmin, bool)
            and isinstance(vmax, (int, float)) and not isinstance(vmax, bool)
        ):
            return (int(min(vmin, vmax)), int(max(vmin, vmax)))
    return None


def _check_manifest_segment(mp, run_dir):
    """Verifies a capture/closed-segment manifest.

    Verdict per manifest:
    - ``corrupt``: declared sha256 that does not match the file on
      disk, or unreadable JSON — defect (fail).
    - ``incomplete``: missing schema, missing key fields (files[],
      window, totals/recorder_stats_final) or declared files that do not
      exist on disk / entries without sha256 — insufficient (pending).
    - ``ok``: declared schema + key fields + all files verified by hash.
    """
    entry = {"manifest": str(mp)}
    m = _load_json(mp)
    if "__error__" in m:
        entry.update(verdict="corrupt",
                       failures=[f"unreadable manifest: {m['__error__']}"])
        return entry
    schema = None
    for k in G2_SCHEMA_KEYS:
        if m.get(k) is not None:
            schema = m[k]
            break
    files = m.get("files")
    missing_items = []
    if schema is None:
        missing_items.append("schema")
    if not isinstance(files, list) or not files:
        missing_items.append("files")
        files = []
    window = _manifest_window(m)
    if window is None:
        missing_items.append("utc_window/totals.recv_ts_ms")
    if not isinstance(m.get("totals"), dict) \
            and m.get("recorder_stats_final") is None:
        missing_items.append("totals/recorder_stats_final")
    verified, absent, failed, without_sha = 0, [], [], []
    base = Path(m.get("research_dir") or run_dir or Path(mp).parent)
    for fe in files:
        rel = fe.get("path") if isinstance(fe, dict) else None
        sha = fe.get("sha256") if isinstance(fe, dict) else None
        if not rel or not _is_sha256(sha):
            without_sha.append(str(rel))
            continue
        fp = Path(rel) if Path(rel).is_absolute() else base / rel
        if not fp.is_file():
            absent.append(rel)
            continue
        if _sha256(fp) != sha:
            failed.append(rel)
            continue
        verified += 1
    entry.update(
        schema=schema,
        window_ms=list(window) if window else None,
        n_files=len(files),
        verified=verified,
        absent=absent,
        without_sha256=without_sha,
        failed=failed,
        missing_fields=missing_items,
    )
    if failed:
        entry["verdict"] = "corrupt"
    elif missing_items or absent or without_sha:
        entry["verdict"] = "incomplete"
    else:
        entry["verdict"] = "ok"
    return entry


def check_g2(run_dir):
    """Event tape: counts per source/kind, canonical `local_seq` order
    (duplicates and displacement of the inversions), avail>=recv
    invariant, recorder stats and closed-segment manifest
    if it exists."""
    g = "G2_capture"
    ev_dir = Path(run_dir) / "events" if run_dir else None
    if ev_dir is None or not ev_dir.is_dir():
        return _res(g, PENDING, {"events_dir": str(ev_dir)},
                    ["no events directory in the run-dir"])
    file_paths = sorted(
        p for p in ev_dir.rglob("*.jsonl") if p.name != "_recorder_stats.jsonl"
    )
    stats_path = ev_dir / "_recorder_stats.jsonl"
    total = 0
    by_source = {}
    by_kind = {}
    n_inversions = 0
    max_shift = 0
    viol_avail = []
    global_seqs = {}
    details = []
    for fp in file_paths:
        prev_seq = None
        n_file = 0
        n_bad = 0
        for ev, err in _iter_jsonl(fp):
            if ev is None:
                n_bad += 1
                details.append(f"{fp.name}: {err}")
                continue
            n_file += 1
            src = str(ev.get("source"))
            kind = str(ev.get("kind"))
            by_source[src] = by_source.get(src, 0) + 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
            seq = ev.get("local_seq")
            if isinstance(seq, int):
                if prev_seq is not None and seq < prev_seq:
                    n_inversions += 1
                    max_shift = max(
                        max_shift, prev_seq - seq
                    )
                prev_seq = max(prev_seq, seq) if prev_seq is not None else seq
                global_seqs[seq] = global_seqs.get(seq, 0) + 1
            recv, avail = ev.get("recv_ts_ms"), ev.get("avail_ts_ms")
            if (
                isinstance(recv, int) and isinstance(avail, int)
                and avail < recv
            ):
                viol_avail.append(
                    f"{fp.name}: avail<{recv - avail}ms (seq={seq})"
                )
        if n_bad:
            details.append(f"{fp.name}: {n_bad} non-JSON lines")
        total += n_file
    duplicates = sum(1 for n in global_seqs.values() if n > 1)
    stats = None
    if stats_path.is_file():
        last_line = None
        for line, _err in _iter_jsonl(stats_path):
            if line:
                last_line = line
        stats = last_line
    else:
        details.append("_recorder_stats.jsonl missing — recorder losses "
                        "not verifiable on disk")
    manifests = sorted((Path(run_dir) / "manifests").glob("*.json")) \
        if (Path(run_dir) / "manifests").is_dir() else []
    man_check = [_check_manifest_segment(mp, run_dir) for mp in manifests]
    if not manifests:
        details.append("no closed-segment manifest in "
                        "<run>/manifests — the active segment is not "
                        "hash-certified (correct while it is still open)")
    else:
        # Heterogeneous schemas (e.g. c1 declares `closed-segment/v1` and
        # post3 declares nothing) are made explicit — never silent.
        with_schema = [Path(m["manifest"]).name
                      for m in man_check if m.get("schema") is not None]
        without_schema = [Path(m["manifest"]).name
                      for m in man_check if m.get("schema") is None]
        if with_schema and without_schema:
            details.append(
                "heterogeneous schemas between segment manifests: "
                f"{with_schema} declare a schema and {without_schema} do not — "
                "inconsistency declared explicitly"
            )
    evidence = {
        "events_dir": str(ev_dir),
        "n_files": len(file_paths),
        "events_total": total,
        "by_source": by_source,
        "by_kind": by_kind,
        "duplicate_local_seq": duplicates,
        "inversions_file": n_inversions,
        "max_seq_shift": max_shift,
        "recorder_stats_last_line": stats,
        "segment_manifests": man_check,
    }
    corrupt_man = [m for m in man_check if m["verdict"] == "corrupt"]
    incomplete_man = [
        m for m in man_check if m["verdict"] == "incomplete"
    ]
    frac_inv = n_inversions / total if total else 0.0
    real_disorder = (
        max_shift > G2_MAX_SHIFT
        or frac_inv > G2_MAX_INVERSIONS_FRACTION
    )
    if duplicates or viol_avail or corrupt_man or real_disorder:
        details += [
            f"duplicate local_seq across files: {duplicates}",
            f"avail<recv: {viol_avail[:5]}",
            f"seq inversions in file: {n_inversions} "
            f"(fraction {frac_inv:.4%}, max displacement "
            f"{max_shift} — tolerances "
            f"{G2_MAX_SHIFT}/{G2_MAX_INVERSIONS_FRACTION:.0%})",
            f"corrupt manifests (sha256 mismatch): "
            f"{[Path(m['manifest']).name for m in corrupt_man]}",
        ]
        return _res(g, FAIL, evidence, [d for d in details if d])
    if n_inversions:
        # Zero loss but visible: bounded inversion from concurrent
        # writes — the canonical order remains in the local_seq value.
        details.append(
            f"bounded seq inversions in file: {n_inversions} "
            f"(fraction {frac_inv:.4%}, max displacement "
            f"{max_shift}) — concurrent recorder writes; "
            "canonical order recoverable by local_seq value, no loss"
        )
    if incomplete_man:
        details.append(
            "incomplete segment manifests (schema/key fields "
            "absent or files not verifiable on disk): "
            + ", ".join(
                f"{Path(m['manifest']).name}"
                f"[missing={m.get('missing_fields')}, "
                f"absent={len(m.get('absent') or [])}, "
                f"without_sha256={len(m.get('without_sha256') or [])}]"
                for m in incomplete_man
            )
        )
    if total == 0:
        return _res(g, PENDING, evidence,
                    details + ["0 events recorded — capture without evidence"])
    if incomplete_man:
        return _res(g, PENDING, evidence,
                    details + ["incomplete manifests — the segment integrity "
                                "is not certified by hash"])
    return _res(g, PASS, evidence, details)


# ── G3/G4 — replay report ───────────────────────────────────────────────────


def check_g3(replay_report):
    """Replay report under contract v2 (addendum): valid `fidelity`,
    `digest` string without ``RECON|false``, `reconciled == true`,
    `external_fidelity` never asserted and, for ``certified``, ``quality``
    with clean scope. Missing mandatory field → fail when the
    report already uses v2 fields; a pre-contract report is pending."""
    g = "G3_replay_ingestion"
    if replay_report is None or not Path(replay_report).is_file():
        return _res(g, PENDING, {"replay_report": replay_report},
                    ["no replay report — pass --replay-report"])
    r = _load_json(replay_report)
    if "__error__" in r:
        return _res(g, FAIL, {"replay_report": str(replay_report)},
                    [f"unreadable report: {r['__error__']}"])
    fidelity = r.get("fidelity")
    digest = r.get("digest")
    if not fidelity:
        return _res(g, FAIL, {"replay_report": str(replay_report)},
                    ["no `fidelity` field — the replay does not declare its "
                     "fidelity level (certified/reduced)"])
    failures, pending = [], []
    if fidelity not in ("certified", "reduced"):
        failures.append(f"fidelity={fidelity!r} — value outside "
                      "{certified, reduced}")
    if not digest:
        failures.append("digest missing/empty — without a digest there is no "
                      "reproducible ledger")
    elif not isinstance(digest, str):
        failures.append("digest is not a string — outside the contract format")
    elif "RECON|false" in digest:
        failures.append("digest contains «RECON|false» — negative economic "
                      "reconciliation inside the digest itself")
    is_v2 = any(k in r for k in V2_REPLAY_FIELDS)
    reconciled = r.get("reconciled", _ABSENT)
    if reconciled is _ABSENT or reconciled is None:
        if is_v2:
            failures.append("reconciled missing/null in a report with "
                          "v2 fields — the contract requires it")
        else:
            pending.append("pre-contract report: no `reconciled` "
                              "— regenerate with the updated replay")
    elif not isinstance(reconciled, bool):
        failures.append(f"reconciled={reconciled!r} is not a boolean")
    elif not reconciled:
        failures.append("reconciled=false — the economic ledger does not "
                      "add up; certified would be a contradiction")
    fext = r.get("external_fidelity", _ABSENT)
    if fext is _ABSENT or fext is None:
        destination = failures if fidelity == "certified" else pending
        destination.append("external_fidelity missing — contract v2 requires "
                       "declaring it (always «unidentified»)")
    elif fext != "unidentified":
        failures.append(f"external_fidelity={fext!r} — nobody identified the "
                      "external fidelity; asserting it is a defect")
    reasons = r.get("fidelity_reasons", _ABSENT)
    if reasons is not _ABSENT and reasons is not None:
        if not (isinstance(reasons, list)
                and all(isinstance(x, str) for x in reasons)):
            failures.append("fidelity_reasons is not a list of strings")
        elif fidelity == "certified" and reasons:
            failures.append("fidelity=certified with non-empty "
                          "fidelity_reasons — internal contradiction of the emitter")
        elif fidelity == "reduced" and not reasons:
            pending.append("fidelity=reduced with empty "
                              "fidelity_reasons — degradation without a declared reason")
    elif fidelity == "certified":
        failures.append("fidelity_reasons missing — contract v2 requires "
                      "declaring it (a list, empty if certified)")
    elif fidelity == "reduced":
        pending.append("fidelity=reduced without fidelity_reasons — the "
                          "reason for the degradation is not declared")
    quality = r.get("quality", _ABSENT)
    if fidelity == "certified":
        if quality is _ABSENT or quality is None:
            failures.append("certified without quality breakdown — quality "
                          "missing; no way to verify «zero "
                          "in-scope degradation»")
        elif not isinstance(quality, dict):
            failures.append("quality is not an object — outside the contract")
        else:
            missing = [k for k in G3_QUALITY_SCOPE if k not in quality]
            if missing:
                failures.append(f"quality without scope metrics "
                              f"{missing} — scope not verifiable")
            else:
                if quality["scope_discarded_deltas"] != 0:
                    failures.append(
                        "certified with scope_discarded_deltas="
                        f"{quality['scope_discarded_deltas']} ≠ 0 — "
                        "degradation inside the scope")
                if quality["tokens_with_gap_scope"] != 0:
                    failures.append(
                        "certified with tokens_with_gap_scope="
                        f"{quality['tokens_with_gap_scope']} ≠ 0 — "
                        "gaps inside the scope")
    elif quality is _ABSENT or quality is None:
        pending.append("fidelity=reduced without quality — the quality "
                          "breakdown cannot be audited")
    evidence = {
        "replay_report": str(replay_report),
        "fidelity": fidelity,
        "digest_len": len(digest) if isinstance(digest, str) else None,
        "digest_recon_false": isinstance(digest, str)
        and "RECON|false" in digest,
        "reconciled": None if reconciled is _ABSENT else reconciled,
        "external_fidelity": None if fext is _ABSENT else fext,
        "fidelity_reasons": None if reasons is _ABSENT else reasons,
        "quality": quality if isinstance(quality, dict) else None,
        "v2_fields_present": [k for k in V2_REPLAY_FIELDS if k in r],
    }
    if failures:
        return _res(g, FAIL, evidence, failures + pending)
    if pending:
        return _res(g, PENDING, evidence, pending)
    return _res(g, PASS, evidence)


def check_g4(replay_report):
    """Settlement (contract v2): reads ``settlement_diagnostic`` — the real
    field name; the legacy ``diagnostic_settlement`` also counts if
    true. certified + diagnostic/proxy → fail; certified +
    settlement_corrections>0 → fail; reconciled=false → fail;
    settlements>0 → pass; explicit pending → pending; 0 without
    pending → pending."""
    g = "G4_settlement"
    if replay_report is None or not Path(replay_report).is_file():
        return _res(g, PENDING, {"replay_report": replay_report},
                    ["no replay report — G4 depends on G3"])
    r = _load_json(replay_report)
    if "__error__" in r:
        return _res(g, FAIL, {"replay_report": str(replay_report)},
                    [f"unreadable report: {r['__error__']}"])
    fidelity = r.get("fidelity")
    diag = r.get("settlement_diagnostic")
    diag_legacy = r.get("diagnostic_settlement")
    corrections = r.get("settlement_corrections")
    reconciled = r.get("reconciled")
    settlements = r.get("settlements")
    settlement_pending = r.get("settlement_pending")
    evidence = {
        "replay_report": str(replay_report),
        "fidelity": fidelity,
        "settlement_diagnostic": diag,
        "diagnostic_settlement_legacy": diag_legacy,
        "settlement_corrections": corrections,
        "reconciled": reconciled,
        "settlements": settlements,
        "settlement_pending": settlement_pending,
    }
    failures, pending = [], []
    active_diag = bool(diag) or bool(diag_legacy)
    if active_diag:
        if fidelity == "certified":
            failures.append(
                "fidelity=certified with settlement_diagnostic active — "
                "the settlement proxy contaminates the certification")
        else:
            pending.append(
                "settlement_diagnostic active — reduced fidelity "
                "is coherent, but the certification is degraded")
    if (
        isinstance(corrections, (int, float))
        and not isinstance(corrections, bool)
        and corrections > 0
    ):
        if fidelity == "certified":
            failures.append(
                f"certified with settlement_corrections={corrections} — "
                "a post-settle resolution correction invalidates the "
                "episode, it is never certifiable")
        else:
            pending.append(
                f"settlement_corrections={corrections} — episodes "
                "invalidated by resolution correction")
    elif fidelity == "certified" and corrections is None:
        pending.append(
            "certified without declared settlement_corrections — the "
            "contract requires an explicit ==0")
    if reconciled is False:
        failures.append("reconciled=false — the economic ledger does not "
                      "add up; nothing to certify in settlement")
    # Count of applied / pending settlements.
    n_applied = n_pending = 0
    anomalous_states = []
    if isinstance(settlements, bool):
        pass
    elif isinstance(settlements, int):
        n_applied = settlements
    else:
        records = (
            settlements.get("records", [])
            if isinstance(settlements, dict)
            else settlements if isinstance(settlements, list) else []
        )
        for s in records:
            st = s.get("state") if isinstance(s, dict) else None
            if st == "applied":
                n_applied += 1
            elif st == "pending":
                n_pending += 1
            else:
                anomalous_states.append(st)
    if isinstance(settlement_pending, list):
        n_pending += len(settlement_pending)
    if anomalous_states:
        pending.append(f"settlement records with anomalous "
                          f"states: {sorted(set(map(str, anomalous_states)))}")
    if failures:
        return _res(g, FAIL, evidence, failures + pending)
    if n_applied > 0:
        return _res(g, PASS, evidence,
                    pending + [f"{n_applied} settlements applied in "
                                  "the window"])
    if n_pending > 0:
        return _res(g, PENDING, evidence,
                    pending + [f"{n_pending} settlements pending "
                                  "declared — no fictitious payout"])
    return _res(g, PENDING, evidence,
                pending + ["settlements==0 or absent — no official "
                              "settlement observed in the window "
                              "(insufficient, no fictitious payout)"])


# ── G5 — episodic environment ──────────────────────────────────────────────────


def _env_log_steps(path):
    """Extracts StepRecords from an environment artifact: Episode JSON (with
    `steps`), JSON list of steps, or JSONL with one step/episode per line."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError as e:
        return None, str(e)
    if not text:
        return [], None
    if text[0] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(data, list):
                return data, None
            if isinstance(data, dict) and "steps" in data:
                return list(data["steps"]), None
            if isinstance(data, dict):
                return [data], None
    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None, f"unreadable JSONL line: {line[:80]}"
        if isinstance(obj, dict) and "steps" in obj:
            steps.extend(obj["steps"])
        else:
            steps.append(obj)
    return steps, None


def _reward_matches(reward):
    """`total` == price_component + fee_component + risk_penalty (Decimal)."""
    try:
        total = Decimal(str(reward["total"]))
        sum_value = (
            Decimal(str(reward["price_component"]))
            + Decimal(str(reward["fee_component"]))
            + Decimal(str(reward["risk_penalty"]))
        )
    except (KeyError, InvalidOperation):
        return False
    return total == sum_value


def check_g5(env_log):
    """Episodic environment: steps with a consistent itemized reward and
    coherent terminal/end_reason."""
    g = "G5_environment"
    if env_log is None or not Path(env_log).is_file():
        return _res(g, PENDING, {"env_log": env_log},
                    ["no episodes artifact — pass --env-log"])
    steps, err = _env_log_steps(env_log)
    if err:
        return _res(g, FAIL, {"env_log": str(env_log)}, [err])
    with_reward = [s for s in steps if isinstance(s, dict) and "reward" in s]
    bad_reward = [s.get("step_idx") for s in with_reward
                  if not _reward_matches(s["reward"])]
    bad_terminal = [
        s.get("step_idx")
        for s in with_reward
        if bool(s.get("terminal")) != (s.get("end_reason") is not None)
    ]
    n_terminal = sum(1 for s in with_reward if s.get("terminal"))
    evidence = {
        "env_log": str(env_log),
        "n_steps": len(steps),
        "n_with_reward": len(with_reward),
        "n_terminal": n_terminal,
        "inconsistent_reward": bad_reward[:10],
        "inconsistent_terminal": bad_terminal[:10],
    }
    if bad_reward or bad_terminal:
        return _res(g, FAIL, evidence,
                    ["reward.total does not add up or terminal/end_reason "
                     "inconsistent"])
    if not with_reward:
        return _res(g, PENDING, evidence,
                    ["artifact without steps carrying reward — environment not "
                     "executed or format without StepRecord"])
    return _res(g, PASS, evidence,
                [f"{len(with_reward)} valid steps, {n_terminal} terminal"])


# ── G6 — calibration report with comparability ─────────────────────────────


def check_g6(calibration_report):
    """Calibration report (v2 addendum): the label
    ``certification_label`` with ``unidentified``/``unidentified``
    and the ``no_comparable``/``unidentified`` profiles are
    honest INSUFFICIENCY → pending, never a disguised pass. Pass requires a
    clean label AND at least one comparison with
    ``comparability.state == "comparable"``. Clean label without a
    valid comparison → fail (contradiction)."""
    g = "G6_calibration"
    if calibration_report is None or not Path(calibration_report).is_file():
        return _res(g, PENDING, {"calibration_report": calibration_report},
                    ["no calibration report — pass "
                     "--calibration-report"])
    m = _load_json(calibration_report)
    if "__error__" in m:
        return _res(g, FAIL, {"calibration_report": str(calibration_report)},
                    [f"unreadable report: {m['__error__']}"])
    label = m.get("certification_label")
    # Comparability per comparison within the replay block.
    declared_comp = []
    for profile, ent in (m.get("replay") or {}).get("profiles", {}).items():
        for strat, e in (ent.get("strategies") or {}).items():
            ce = (e.get("comparison") or {}).get("comparability")
            if ce:
                declared_comp.append(
                    {"profile": profile, "strategy": strat,
                     "state": ce.get("state")})
    states = [c["state"] for c in declared_comp]
    n_comparable = sum(1 for e in states if e == "comparable")
    evidence = {
        "calibration_report": str(calibration_report),
        "certification_label": label,
        "declared_comparability": declared_comp,
        "n_comparable": n_comparable,
        "tool_version": m.get("version"),
    }
    if not label:
        return _res(g, FAIL, evidence,
                    ["report without a comparability label — an "
                     "«out_of_tolerance» inherited from an invalid "
                     "comparison is not evidence"])
    if "unidentified" in str(label):
        return _res(g, PENDING, evidence,
                    [f"label {label!r} — external fidelity not "
                     "identified: honest insufficiency, not a disguised "
                     "pass"])
    if n_comparable == 0:
        if declared_comp:
            return _res(g, FAIL, evidence,
                        [f"label {label!r} claims certification but "
                         f"the comparisons are {sorted(set(map(str, states)))} "
                         "— contradiction"])
        return _res(g, FAIL, evidence,
                    [f"label {label!r} claims certification without any "
                     "replay↔real comparison executed"])
    return _res(g, PASS, evidence,
                [f"{n_comparable} «comparable» comparison(s) and "
                 "label without «unidentified»"])


# ── G7 — model metrics and provenance ──────────────────────────────────

# Keys accepted in the sibling manifest of the model to declare the
# tape dataset sources (list of {path, sha256, window}).
G7_SOURCES_KEYS = ("sources", "dataset_sources")
# Keys accepted for the label statistics (n + real known_at
# + excluded).
G7_LABELS_KEYS = ("labels", "labels_stats", "stats_labels")
# Window keys inside each source.
G7_SOURCE_WINDOW_KEYS = ("window_ms", "utc_window", "window")


def _dataset_levels(man):
    """Levels where the dataset manifest can be nested:
    ``man``, ``man.dataset`` and ``man.dataset.manifest`` (train.py
    embeds the full dataset_manifest under that key)."""
    levels = [man]
    ds = man.get("dataset")
    if isinstance(ds, dict):
        levels.append(ds)
        if isinstance(ds.get("manifest"), dict):
            levels.append(ds["manifest"])
    return levels


def _manifest_sources(man):
    """List of dataset sources declared in a model manifest
    (``sources``/aliases, or nested under ``dataset``). [] if none."""
    for level in _dataset_levels(man):
        for k in G7_SOURCES_KEYS:
            f = level.get(k)
            if isinstance(f, list):
                return f
    return []


def _manifest_labels(man):
    """Label stats block of the manifest (dict or None)."""
    for level in _dataset_levels(man):
        for k in G7_LABELS_KEYS:
            lab = level.get(k)
            if isinstance(lab, dict):
                return lab
    return None


def check_g7(metrics_dir, model_path=None):
    """model metrics + provenance of the tape dataset.

    - ``metrics_eval.json``/``metrics_train.json`` present with n>0 and
      consistent with each other (``test.n`` of train == ``n_test`` of eval
      when both exist).
    - ``sha256(model.json)`` == entry of the sibling ``manifest.json``
      of the model.
    - The manifest declares the dataset ``sources`` (path + sha256 +
      window per source), verified against disk when the file
      exists, and label stats (``n`` + real ``known_at`` +
      ``excluded``).

    Missing or unverifiable provenance → pending (insufficient, honest);
    a declared hash that does not match, or inconsistent metrics → fail."""
    g = "G7_model_metrics"
    if metrics_dir is None or not Path(metrics_dir).is_dir():
        return _res(g, PENDING, {"metrics_dir": metrics_dir},
                    ["no metrics directory — pass --metrics-dir "
                     "o --model"])
    details, failures, pending = [], [], []
    evidence = {"metrics_dir": str(metrics_dir)}
    n_eval = n_test_train = None
    for name_str, n_key in (
        ("metrics_eval.json", "n_test"),
        ("metrics_train.json", "n_train"),
    ):
        fp = Path(metrics_dir) / name_str
        if not fp.is_file():
            pending.append(f"{name_str} missing")
            evidence[name_str] = "missing"
            continue
        m = _load_json(fp)
        if "__error__" in m:
            failures.append(f"{name_str} unreadable: {m['__error__']}")
            continue
        n = m.get(n_key)
        if n is None:
            n = m.get("n") or m.get("n_obs") or m.get("n_samples")
        evidence[name_str] = {"n": n, "sha256": _sha256(fp)}
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            # n<=0 with an explicitly declared OOS insufficiency is
            # PENDING (insufficient empirical sample, honest); n<=0
            # without a declaration it is fail (broken/incomplete artifact).
            insuf = (
                m.get("oos_evaluation") == "insufficient"
                or m.get("test_empty") is True
                or (m.get("sufficiency") or {}).get("oos_evidence")
                == "insufficient")
            if insuf:
                pending.append(
                    f"{name_str}: n={n} with a declared OOS insufficiency "
                    "— without an evaluable partition there are no metrics to "
                    "approve")
            else:
                failures.append(f"{name_str}: n={n}")
        if name_str == "metrics_eval.json":
            n_eval = n
        else:
            test_b = m.get("test")
            if isinstance(test_b, dict) and isinstance(
                    test_b.get("n"), int):
                n_test_train = test_b["n"]
    if (
        n_eval is not None and n_test_train is not None
        and n_eval != n_test_train
    ):
        failures.append(
            f"inconsistent metrics: metrics_eval.n_test={n_eval} ≠ "
            f"metrics_train.test.n={n_test_train}")

    # ── provenance: model.json ↔ sibling manifest ↔ tape sources ──
    man = None
    model_sha = None
    if model_path and Path(model_path).is_file():
        model_sha = _sha256(model_path)
        evidence["model"] = str(model_path)
        evidence["model_sha256"] = model_sha
        man_path = Path(model_path).parent / "manifest.json"
        if not man_path.is_file():
            pending.append(
                "no sibling manifest.json for the model — dataset provenance "
                "not verifiable")
        else:
            man = _load_json(man_path)
            evidence["manifest_model"] = str(man_path)
            if "__error__" in man:
                failures.append(
                    f"model manifest unreadable: {man['__error__']}")
                man = None
    else:
        pending.append("no model.json — dataset provenance not "
                          "verifiable (pass --model)")
    if man is not None:
        base = Path(man.get("research_dir") or Path(model_path).parent)
        entries = man.get("files") or []
        ent_model = [
            e for e in entries
            if isinstance(e, dict)
            and Path(str(e.get("path", ""))).name == Path(model_path).name
        ]
        if not ent_model:
            pending.append(
                "manifest has no files[] entry for model.json — the "
                "artifact is not bound by hash")
        elif ent_model[0].get("sha256") != model_sha:
            failures.append(
                f"sha256(model.json)={model_sha} ≠ manifest.files[] "
                f"({ent_model[0].get('sha256')}) — the verified model "
                "is not the declared one")
        sources = _manifest_sources(man)
        if not sources:
            pending.append(
                "manifest has no dataset sources (path+sha256+window) "
                "— the model does not declare tape provenance")
        else:
            sources_summary = []
            for i, f in enumerate(sources):
                if not isinstance(f, dict):
                    pending.append(f"source[{i}] is not an object")
                    continue
                rel = f.get("path") or f.get("file") or f.get("path")
                sha = f.get("sha256")
                vent = None
                for k in G7_SOURCE_WINDOW_KEYS:
                    vent = _window_ms(f.get(k))
                    if vent is not None:
                        break
                missing = []
                if not rel:
                    missing.append("path")
                if not _is_sha256(sha):
                    missing.append("sha256")
                if vent is None:
                    missing.append("window")
                sources_summary.append(
                    {"path": rel, "window_ms": list(vent) if vent else None,
                     "missing": missing})
                if missing:
                    pending.append(
                        f"source[{i}] incomplete — missing {missing}")
                    continue
                fp = Path(rel) if Path(rel).is_absolute() else base / rel
                if not fp.is_file():
                    pending.append(
                        f"source[{i}] {rel}: does not exist on disk — sha256 "
                        "not verifiable")
                elif _sha256(fp) != sha:
                    failures.append(
                        f"source[{i}] {rel}: sha256 does not match the "
                        "manifest")
            evidence["dataset_sources"] = sources_summary
        labels = _manifest_labels(man)
        if labels is None:
            pending.append(
                "manifest has no stats for labels (n, real known_at, "
                "excluded) — resolutions without declared traceability")
        else:
            n_lab = labels.get("n") or labels.get("total") \
                or labels.get("n_labels")
            known = [k for k in labels
                         if str(k).startswith("known_at")]
            evidence["labels_stats"] = labels
            if not isinstance(n_lab, int) or isinstance(n_lab, bool) \
                    or n_lab <= 0:
                pending.append("labels without a declared n>0")
            if not known:
                pending.append(
                    "labels without known_at_* — without knowledge timestamps "
                    "real, back-dated labels are not discarded")
    if failures:
        return _res(g, FAIL, evidence, details + failures + pending)
    if pending:
        return _res(g, PENDING, evidence, details + pending)
    return _res(g, PASS, evidence, details)


# ── G8 — shadow log ────────────────────────────────────────────────────


# Window keys declared in the shadow stats lines.
G8_WINDOW_KEYS = ("window_ms", "utc_window", "window")


def _declaration_no_emission(line):
    """No-emission declaration in a stats line:
    True = declares NOT emitting, False = declares emitting, None = no declaration."""
    if "emits_orders" in line:
        return not bool(line["emits_orders"])
    if "emitted_orders" in line:
        return line["emitted_orders"] == 0
    if "emits_no_orders" in line:
        return bool(line["emits_no_orders"])
    if "order_emission" in line:
        return line["order_emission"] in (False, 0, "none", "no")
    return None


def _line_window(line):
    for k in G8_WINDOW_KEYS:
        v = _window_ms(line.get(k))
        if v is not None:
            return v
    return None


def check_g8(shadow_dir, model_sha256=None, source_windows=None):
    """Shadow record verified in streaming (O(1) memory; real tapes
    exceed 200k lines).

    - ALL ``decision``/``abstention`` rows carry
      ``model_hash == sha256(model.json)`` verified by G7; a single
      row with another hash or without a hash → fail with a count.
    - Zero lines of kind order/intent (the shadow does not emit by
      construction) and an explicit no-emission declaration.
    - ``stats`` with p50/p95/p99 latencies, a declared window contained
      in the sources' window and an explicit no-emission declaration
      (``emits_orders:false``/``emitted_orders:0``/
      ``emits_no_orders:true``/``order_emission:"none"``).
    """
    g = "G8_shadow"
    if shadow_dir is None or not Path(shadow_dir).is_dir():
        return _res(g, PENDING, {"shadow_dir": shadow_dir},
                    ["no shadow log — pass --shadow-dir"])
    file_paths = sorted(Path(shadow_dir).glob("*.jsonl"))
    count = {t: 0 for t in VALID_SHADOW_LINES}
    forbidden_count = 0
    stats_without_latency = 0
    n_stats = 0
    hash_ok = hash_mismatch = absent_hash = 0
    shadow_window = None
    decl_no_emission = None
    for fp in file_paths:
        for line, _err in _iter_jsonl(fp):
            if not isinstance(line, dict):
                continue
            kind = line.get("kind")
            if kind in FORBIDDEN_SHADOW_LINES:
                forbidden_count += 1
            elif kind in ("decision", "make_abstention"):
                count[kind] += 1
                if model_sha256 is not None:
                    h = line.get("model_hash")
                    if not isinstance(h, str) or not h:
                        absent_hash += 1
                    elif h != model_sha256:
                        hash_mismatch += 1
                    else:
                        hash_ok += 1
            elif kind == "stats":
                count["stats"] += 1
                n_stats += 1
                if not all(
                    k in line for k in ("p50_ms", "p95_ms", "p99_ms")
                ):
                    stats_without_latency += 1
                v = _line_window(line)
                if v is not None:
                    shadow_window = v
                nd = _declaration_no_emission(line)
                if nd is not None:
                    decl_no_emission = nd
    evidence = {
        "shadow_dir": str(shadow_dir),
        "file_paths": [f.name for f in file_paths],
        "lines": count,
        "forbidden_lines": forbidden_count,
        "n_stats": n_stats,
        "expected_model_sha256": model_sha256,
        "model_hash_ok": hash_ok,
        "model_hash_mismatch": hash_mismatch,
        "absent_model_hash": absent_hash,
        "shadow_window_ms": list(shadow_window) if shadow_window else None,
        "declaration_no_emission": decl_no_emission,
    }
    failures, pending = [], []
    if forbidden_count:
        failures.append(f"{forbidden_count} lines with order/intent kind — "
                      "the shadow cannot emit orders by "
                      "construction")
    if decl_no_emission is False:
        failures.append("the shadow log declares it EMITS orders — "
                      "which contradicts its construction")
    if stats_without_latency:
        failures.append(f"{stats_without_latency} stats lines without "
                      "p50/p95/p99")
    if model_sha256 is not None and (hash_mismatch or absent_hash):
        failures.append(
            f"rows decision/abstention with model_hash ≠ "
            f"sha256(model.json): {hash_mismatch} different + "
            f"{absent_hash} missing — the evaluated shadow is not the "
            "verified model")
    if shadow_window is not None and source_windows:
        within = any(
            t0 <= shadow_window[0] and shadow_window[1] <= t1
            for t0, t1 in source_windows
        )
        if not within:
            failures.append(
                f"shadow window {shadow_window} outside the "
                f"sources windows {source_windows}")
    if failures:
        return _res(g, FAIL, evidence, failures)
    if count["decision"] + count["make_abstention"] == 0:
        pending.append("empty shadow log or without "
                          "decision/make_abstention")
    if n_stats == 0:
        pending.append("no stats lines — latencies under load not "
                          "measured (short shadow)")
    if model_sha256 is None:
        pending.append("no verified model — the binding "
                          "shadow↔model by model_hash cannot be checked")
    if shadow_window is None:
        pending.append("stats without a declared window — the "
                          "temporal coverage of the shadow is not auditable")
    elif not source_windows:
        pending.append("shadow window declared but without "
                          "sources windows to compare it with")
    if decl_no_emission is None:
        pending.append("no order no-emission declaration in "
                          "the stats lines")
    if pending:
        return _res(g, PENDING, evidence, pending)
    return _res(g, PASS, evidence,
                [f"{hash_ok} rows bound to the verified model"])


# ── G9 — integrity ─────────────────────────────────────────────────────────


def check_g9(repo, run_dir):
    """Integrity: run-dir outside the repo, live flags unchanged,
    original evidence intact by hash/manifest, git with no changes
    in tracked evidence, no .env copied into the run-dir."""
    g = "G9_integrity"
    details = []
    failures = []
    repo = Path(repo).resolve() if repo else None
    run = Path(run_dir).resolve() if run_dir else None
    if repo is None or not repo.is_dir():
        return _res(g, PENDING, {"repo": str(repo)}, ["repo missing"])
    evidence = {"repo": str(repo), "run_dir": str(run)}

    if run is not None:
        if run == repo or repo in run.parents:
            failures.append(f"run-dir inside the repo: {run}")
        else:
            details.append("run-dir outside the repo")
        # No file of the run-dir may live inside the repo, nor
        # carry secrets (copied .env).
        for env in run.rglob(".env") if run.is_dir() else []:
            failures.append(f".env copied to the run-dir: {env}")
        for env in run.rglob("*.env") if run.is_dir() else []:
            failures.append(f"env file in the run-dir: {env}")

    # Research runs must not be configured for live: mode=paper in general.toml.
    gtoml = repo / "config" / "general.toml"
    if gtoml.is_file():
        try:
            cfg = tomllib.loads(gtoml.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            failures.append(f"general.toml unreadable: {e}")
            cfg = {}
        mode = cfg.get("mode", "paper")
        evidence["mode"] = mode
        if mode != "paper":
            failures.append(f"config/general.toml mode={mode!r} ≠ paper")
    else:
        details.append("config/general.toml missing")
    # Original evidence: dataset manifests with sha256 per file.
    verified, bad_hash = [], []
    for mf in sorted(repo.glob("research/data/*/dataset_manifest.json")):
        m = _load_json(mf)
        base = mf.parent
        for fe in m.get("files", []) or []:
            fp = base / fe.get("path", "")
            if not fp.is_file():
                bad_hash.append(f"{fp}: missing")
                continue
            if _sha256(fp) != fe.get("sha256"):
                bad_hash.append(f"{fp}: sha256 ≠ manifest")
            else:
                verified.append(str(fp))
    evidence["verified_evidence"] = verified
    evidence["corrupt_evidence"] = bad_hash
    if bad_hash:
        failures.append(f"original evidence altered: {bad_hash}")

    # git: tracked evidence must not appear modified or deleted
    # (the ?? untracked files are informative; the hash above is
    # the real verification).
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--",
             "research/"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            modified = [
                ln for ln in proc.stdout.splitlines()
                if ln[:2].strip() and not ln.startswith("??")
            ]
            evidence["git_modified_research"] = modified
            if modified:
                failures.append(
                    f"git: tracked evidence modified: {modified}"
                )
        else:
            details.append(f"git status rc={proc.returncode}: "
                            f"{proc.stderr.strip()[:200]}")
    except (OSError, subprocess.TimeoutExpired) as e:
        details.append(f"git unavailable: {e}")

    if failures:
        return _res(g, FAIL, evidence, details + failures)
    return _res(g, PASS, evidence, details)


# ── Default artifact location ──────────────────────────────────


def _search(run_dir, patterns):
    """First existing file under run_dir that matches any of the
    glob patterns (in order)."""
    if run_dir is None:
        return None
    base = Path(run_dir)
    for pat in patterns:
        hits = sorted(base.glob(pat))
        if hits:
            return hits[0]
    return None


def resolve_artifacts(args):
    """Reasonable defaults inside the run-dir when the flag is not passed."""
    run_dir = Path(args.run_dir) if args.run_dir else None
    model = args.model or _search(
        run_dir, ["model/model.json", "**/model.json"]
    )
    replay = args.replay_report or _search(
        run_dir,
        ["replay_report.json", "replay/report.json",
         "**/replay_report*.json", "**/*replay*.json"],
    )
    env_log = args.env_log or _search(
        run_dir,
        ["env/episodes.jsonl", "**/*episode*.json", "**/*episode*.jsonl",
         "**/*env_log*.jsonl", "**/env*.jsonl"],
    )
    calib = args.calibration_report or _search(
        run_dir,
        ["calibration/*.json", "**/*calibra*.json", "**/metrics-T*.json"],
    )
    metrics_dir = args.metrics_dir
    if metrics_dir is None and model is not None:
        metrics_dir = str(Path(model).parent)
    shadow = args.shadow_dir or _search(
        run_dir, ["shadow", "**/shadow"]
    )
    return {
        "model": model,
        "replay_report": replay,
        "env_log": env_log,
        "calibration_report": calib,
        "metrics_dir": metrics_dir,
        "shadow_dir": shadow,
    }


def _source_windows(run_dir, model_path):
    """Windows (t0_ms, t1_ms) of the available data sources:
    capture manifests (``<run>/manifests/*.json``: totals.recv_ts_ms_min/max)
    and dataset sources declared in the sibling manifest of the model.
    None if there is no parseable one."""
    windows = []
    if run_dir:
        mdir = Path(run_dir) / "manifests"
        if mdir.is_dir():
            for mp in sorted(mdir.glob("*.json")):
                m = _load_json(mp)
                if "__error__" in m:
                    continue
                v = _manifest_window(m)
                if v is not None:
                    windows.append(v)
    if model_path and Path(model_path).is_file():
        man_path = Path(model_path).parent / "manifest.json"
        if man_path.is_file():
            man = _load_json(man_path)
            if "__error__" not in man:
                for f in _manifest_sources(man):
                    if not isinstance(f, dict):
                        continue
                    for k in G7_SOURCE_WINDOW_KEYS:
                        v = _window_ms(f.get(k))
                        if v is not None:
                            windows.append(v)
                            break
    return windows or None


def readable_summary(result):
    lines = [
        f"verdict: {result['verdict']} · "
        f"pass={result['summary']['pass']} "
        f"fail={result['summary']['fail']} "
        f"pending={result['summary']['pending']}",
    ]
    for g in result["gates"]:
        mark = {PASS: "PASS", FAIL: "FAIL", PENDING: "PEND"}[g["state"]]
        lines.append(f"  [{mark}] {g['gate']}")
        for d in g["details"][:3]:
            lines.append(f"        {d}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="G1–G9 acceptance verifier for the close-out of the "
        "numeric environment."
    )
    ap.add_argument("--run-dir", default=None,
                    help="EXTERNAL run directory (e.g. "
                    "research_runs/close-2026-09-23)")
    ap.add_argument("--repo", default=None,
                    help="Repo root (default: parent of tools/)")
    ap.add_argument("--model", default=None,
                    help="model.json of the training run to verify (G1/G7)")
    ap.add_argument("--replay-report", default=None,
                    help="Replay report JSON (G3/G4)")
    ap.add_argument("--env-log", default=None,
                    help="Environment episodes/StepRecords (G5)")
    ap.add_argument("--calibration-report", default=None,
                    help="calibrate_paper metrics JSON (G6)")
    ap.add_argument("--metrics-dir", default=None,
                    help="Dir with metrics_eval/train.json (default: dir "
                    "of --model)")
    ap.add_argument("--shadow-dir", default=None,
                    help="Dir with the shadow log *.jsonl (G8)")
    ap.add_argument("--out", default=None,
                    help="Write the acceptance JSON to this path "
                    "(in addition to the readable summary on stderr)")
    ap.add_argument("--strict", action="store_true",
                    help="pending → acceptance failure (exit 1). By "
                    "default, pending is reported explicitly with exit 0.")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve() if args.repo else \
        Path(__file__).resolve().parents[2]
    art = resolve_artifacts(args)
    # Hash of the model verified ONCE, shared by G7 (provenance
    # against the sibling manifest) and G8 (shadow↔model binding).
    model_path = art["model"]
    model_sha256 = (
        _sha256(model_path)
        if model_path and Path(model_path).is_file()
        else None
    )
    windows = _source_windows(args.run_dir, model_path)

    gates = [
        check_g1(model_path),
        check_g2(args.run_dir),
        check_g3(art["replay_report"]),
        check_g4(art["replay_report"]),
        check_g5(art["env_log"]),
        check_g6(art["calibration_report"]),
        check_g7(art["metrics_dir"], model_path),
        check_g8(art["shadow_dir"], model_sha256, windows),
        check_g9(repo, args.run_dir),
    ]
    n = {PASS: 0, FAIL: 0, PENDING: 0}
    for g in gates:
        n[g["state"]] += 1
    verdict = (
        "fail" if n[FAIL]
        else ("pending" if n[PENDING] else "pass")
    )
    if args.strict and n[PENDING]:
        verdict = "fail" if n[FAIL] else "fail_by_pending"
    result = {
        "tool": "verify_research_environment",
        "version": VERSION,
        "contract": "close-2026-09-23:addendum-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "repo": str(repo),
        "resolved_artifacts": {k: str(v) if v else None
                               for k, v in art.items()},
        "gates": gates,
        "summary": n,
        "verdict": verdict,
        "semantics": {
            "fail": "verified engineering defect",
            "pending": "insufficient coverage / missing artifact — "
            "explicit state, not approved",
            "strict": "with --strict, pending also forces exit 1",
        },
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    print(readable_summary(result), file=sys.stderr)
    if n[FAIL] or (args.strict and n[PENDING]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
