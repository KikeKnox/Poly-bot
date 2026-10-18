#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset builder from a REAL `ResearchEvent` tape recorded by
the bot's recorder (closure 2026-09-24, addendum v2).

Unlike `export_sqlite.py` (reconstruction from historical DBs),
here the input IS the contract's causal tape: a directory of sources
`<source>/<date>.jsonl` (recorder layout) or an explicit list of
file paths. The whole pipeline is STREAMING line by line — the source
files weigh GBs (the post3 capture is ~4.8 GB / 6.7 M events; the
full Rust loader already went OOM on this tape) and NOTHING is
materialized whole in memory.

Pipeline:
1. K-way merge by `(avail_ts_ms, local_seq)` — the contract's canonical order.
   The bounded LOCAL inversions produced by the recorder's write race
   are resolved with a window reorder
   (`REORDER_WINDOW`) accounted in the manifest (inversions and
   maximum shift); a real shift >= window is an explicit
   failure, never a silent reordering.
   `local_seq` is REASSIGNED in that canonical order (as
   `export_sqlite.py` does), so the resulting tape meets the invariant
   "avail monotonic in seq" that `build_observations_incremental`
   needs for streaming. It is declared in the manifest.
2. Each file is hashed (sha256) WHILE it is read — real provenance without
   double I/O; if `--manifest` is passed (capture manifest with
   `files[].sha256`) each hash is VERIFIED against it (mismatch = failure).
3. Labels: ONLY from official `resolution`/`market_meta.outcome` events
   on the tape, with a real `known_at` (`payload.known_at_ms`; fallback to the
   event's own `avail_ts_ms` — see `dataset.build_labels`).
   Availability at the fit cutoff and the number of late/excluded are reported.
4. Observations with the incremental builder of `dataset.py` (time
   mesh `--decision-min-interval-ms`, default 1000 ms/market —
   without a mesh a high-frequency tape would emit one observation per
   delta) + the ALREADY FIXED purged/embargoed split of `dataset.py`
   (`temporal_split` + `filter_labels_available`, reused — not
   reimplemented).

Output in `--out`: `events.jsonl` (merged canonical tape — the dataset) +
`dataset_manifest.json` (sources with path+sha256, capture window,
counts per kind, quality marks/exclusions, n rows and n markets
per train/validation/test partition, labels report).

Input decision (documented): the complete capture DIRECTORY is used
and not the `tape_post3_g4.jsonl` slice because (a) the slice is
concatenated by source (non-monotonic local_seq → unusable for the
incremental builder without reordering in memory) and (b) the capture
manifest gives sha256 per source file — verifiable provenance.

Usage:
    python3 research/numeric_model/dataset_tape.py \
        --events-dir research_runs/closure-2026-09-23/events_post3 \
        --manifest research_runs/closure-2026-09-23/manifests/post3_capture.json \
        --out research_runs/closure-2026-09-24/dataset_tape
"""

import argparse
import hashlib
import heapq
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402

SCHEMA_VERSION = 1

# Default time mesh for decision points (ms per market).
DECISION_GRID_DEFAULT_MS = 1000


class CorruptTape(Exception):
    """Explicit integrity failure of the source tape: malformed JSON,
    event without minimum fields, source not monotonic in (avail, seq) or
    sha256 that does not match the declared capture manifest."""


def _iso(ts_ms):
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError, TypeError):
        return None


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_sources(events_dir):
    """Recorder layout: `<dir>/<source>/<date>.jsonl`. Returns the
    sorted list of event files (one level of depth —
    the root-level auxiliaries of kind `_recorder_stats.jsonl` are NOT events
    of the contract and are explicitly ignored)."""
    base = Path(events_dir)
    files = sorted(p for p in base.glob("*/*.jsonl") if p.is_file())
    if not files:
        raise CorruptTape(
            f"--events-dir {base} contains no <source>/<date>.jsonl")
    return files


def _iter_events(path, stats):
    """Per-file generator: parses line by line, accumulates the sha256 of the
    bytes read and counts inversions of `(avail_ts_ms, local_seq)` —
    the recorder's write race produces bounded LOCAL
    inversions (measured: ~0.23% of events, max shift 69 in c1)
    that `_sorted_iter` resolves with a window reorder. Malformed
    JSON or an event without minimum fields = `CorruptTape` (explicit
    failure, contract: never tolerant decoding)."""
    h = hashlib.sha256()
    max_key = None
    n = 0
    f_min = f_max = None
    with open(path, "rb") as f:
        for raw in f:
            h.update(raw)
            line = raw.strip()
            if not line:
                stats["empty_lines"] += 1
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                raise CorruptTape(
                    f"{path}:{n + 1} malformed JSON: {e}") from e
            if not isinstance(ev, dict) or "avail_ts_ms" not in ev \
                    or "local_seq" not in ev:
                raise CorruptTape(
                    f"{path}:{n + 1} event without avail_ts_ms/local_seq")
            key = (ev["avail_ts_ms"], ev["local_seq"])
            if max_key is not None and key < max_key:
                stats["inversions_source"] += 1
            else:
                max_key = key
            n += 1
            stats["n_events"] += 1
            k = ev.get("kind")
            kn = ("sim:" + next(iter(k.values()))
                  if isinstance(k, dict) else k)
            stats["counts_by_kind"][kn] = \
                stats["counts_by_kind"].get(kn, 0) + 1
            q = ev.get("quality", 0) or 0
            stats["quality_histogram"][q] = \
                stats["quality_histogram"].get(q, 0) + 1
            a = ev["avail_ts_ms"]
            if stats["min_avail"] is None or a < stats["min_avail"]:
                stats["min_avail"] = a
            if stats["max_avail"] is None or a > stats["max_avail"]:
                stats["max_avail"] = a
            if f_min is None or a < f_min:
                f_min = a
            if f_max is None or a > f_max:
                f_max = a
            yield ev
    stats["by_file"][str(path)] = {
        "sha256": h.hexdigest(), "n_events": n,
        "bytes": Path(path).stat().st_size,
        "window_ms": [f_min, f_max],
    }


# Per-file reorder window: the recorder's inversions are
# local and bounded (max shift measured 69 over 12.6 M events
# in c1); 8192 gives a ×100 margin with no appreciable memory. A real
# shift >= window would indicate a structurally broken tape → failure.
REORDER_WINDOW = 8192


def _sorted_iter(path, stats):
    """Emits the source in canonical order `(avail_ts_ms, local_seq)` with
    a heap of at most `REORDER_WINDOW` events (bounded sorting
    in streaming). Measures the real shift per event; if it exceeds
    the window the source was not reorderable → `CorruptTape`."""
    buf = []
    idx_in = idx_out = 0
    for ev in _iter_events(path, stats):
        heapq.heappush(
            buf, (ev["avail_ts_ms"], ev["local_seq"], idx_in, ev))
        idx_in += 1
        if len(buf) > REORDER_WINDOW:
            _, _, i_orig, e = heapq.heappop(buf)
            shift = abs(i_orig - idx_out)
            if shift > stats["max_shift"]:
                stats["max_shift"] = shift
            idx_out += 1
            yield e
    while buf:
        _, _, i_orig, e = heapq.heappop(buf)
        shift = abs(i_orig - idx_out)
        if shift > stats["max_shift"]:
            stats["max_shift"] = shift
        idx_out += 1
        yield e
    if stats["max_shift"] >= REORDER_WINDOW:
        raise CorruptTape(
            f"{path}: shift {stats['max_shift']} >= "
            f"window {REORDER_WINDOW} — source not reorderable")


def _stats_base():
    return {
        "n_events": 0, "empty_lines": 0,
        "counts_by_kind": {}, "quality_histogram": {},
        "min_avail": None, "max_avail": None, "by_file": {},
        "inversions_source": 0, "max_shift": 0,
    }


def iter_canonical_tape(sources, stats=None):
    """K-way merge of the sources by `(avail_ts_ms, local_seq)` —
    the contract's canonical order — with `local_seq` REASSIGNED to the merge
    index (documented in the manifest). Pure generator: O(number of
    sources) memory."""
    stats = stats if stats is not None else _stats_base()
    streams = [
        ((ev["avail_ts_ms"], ev["local_seq"], i, ev)
         for ev in _sorted_iter(p, stats))
        for i, p in enumerate(sources)
    ]
    seq = 0
    for _, _, _, ev in heapq.merge(*streams, key=lambda t: (t[0], t[1])):
        ev["local_seq"] = seq
        seq += 1
        yield ev
    stats["reassigned_local_seq"] = True


def merge_to_file(sources, out_path, stats=None):
    """`iter_canonical_tape` written to `out_path` (compact jsonl)."""
    stats = stats if stats is not None else _stats_base()
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in iter_canonical_tape(sources, stats):
            f.write(json.dumps(ev, ensure_ascii=False,
                               separators=(",", ":")) + "\n")
    return stats


def _load_capture_manifest(manifest_path):
    """Capture manifest (format `captura_*.json`): files[] with a path
    relative to research_dir + declared sha256. Returns
    {resolved_path: declared_sha256} and the base dir."""
    doc = json.loads(Path(manifest_path).read_text())
    base = Path(doc.get("research_dir") or Path(manifest_path).parent)
    expected = {}
    for f in doc.get("files") or []:
        p = Path(f["path"])
        if not p.is_absolute():
            p = base / p
        expected[str(p.resolve())] = f.get("sha256")
    return expected, doc


def labels_report(labels):
    """Summary of label availability for the manifest."""
    gan = {}
    prov = set()
    with_payload = with_avail = without_known = 0
    knowns = []
    for lab in labels.values():
        gan[lab["winner"]] = gan.get(lab["winner"], 0) + 1
        prov.add(lab["provenance"])
        if lab.get("known_at_ms") is None:
            without_known += 1
        else:
            knowns.append(lab["known_at_ms"])
            if lab.get("known_at_source") == "avail_ts_ms":
                with_avail += 1
            else:
                with_payload += 1
    return {
        "n_labels": len(labels),
        "winners": gan,
        "provenances": sorted(prov),
        "with_known_at_payload": with_payload,
        "with_known_at_avail_fallback": with_avail,
        "without_known_at": without_known,
        "known_at_n": len(knowns),
        "known_at_min_ms": min(knowns) if knowns else None,
        "known_at_max_ms": max(knowns) if knowns else None,
    }


def run(sources, out_dir, manifest_capture=None, write_tape=True,
        cutoff_fit_ms=None,
        decision_min_interval_ms=DECISION_GRID_DEFAULT_MS,
        train_frac=0.6, val_frac=0.0, train_end_ms=None, val_end_ms=None,
        embargo_ms=None, do_split=True):
    """Full pipeline: canonical merge → labels → observations → certified
    split → `events.jsonl` + `dataset_manifest.json` in `out_dir`.
    Returns the manifest dict."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = [str(Path(p).resolve()) for p in sources]

    stats = _stats_base()
    pending_stats = True   # the first consumption of the tape fills `stats`
    if write_tape:
        tape = out / "events.jsonl"
        merge_to_file(sources, tape, stats)
        pending_stats = False
        iter_tape = lambda: dataset.iter_events_jsonl(tape)  # noqa: E731
    else:
        tape = None

        # No canonical file: each pass re-merges from the sources
        # (re-reads the GBs — only for diagnostic runs). The FIRST
        # pass fills `stats` (provenance/counts/window).
        def iter_tape():
            nonlocal pending_stats
            if pending_stats:
                yield from iter_canonical_tape(sources, stats)
                pending_stats = False
            else:
                yield from iter_canonical_tape(sources, _stats_base())

    # Official labels from the tape (dedicated streaming pass — with
    # --no-tape it is the pass that also fills the source stats).
    labels = dataset.build_labels(iter_tape(), assume_sorted=True)

    # Verifiable provenance: sha256 recomputed during the read vs
    # the declared capture manifest (if supplied). Mismatch = explicit
    # failure — tampered evidence is not analyzed.
    verification = None
    if manifest_capture:
        expected, doc_cap = _load_capture_manifest(manifest_capture)
        verification = {"manifest": str(manifest_capture), "files": []}
        for p, st in stats["by_file"].items():
            dec = expected.get(p)
            ok = (dec == st["sha256"])
            verification["files"].append({
                "path": p, "sha256": st["sha256"],
                "sha256_manifest": dec, "verified": ok,
            })
            if not ok:
                raise CorruptTape(
                    f"sha256 does not match the capture manifest: "
                    f"{p} (got {st['sha256']}, declared {dec})")
        missing = sorted(set(expected) - set(stats["by_file"]))
        verification["declared_unused"] = missing

    rep_labels = labels_report(labels)

    split_meta = None
    obs_stats = None
    parts = None
    if do_split:
        # Causal observations on a time mesh (same builder and same
        # purged/embargoed split as train.py — reused).
        obs = dataset.build_observations_incremental(
            iter_tape(), assume_sorted=True,
            decision_min_interval_ms=decision_min_interval_ms)
        obs = dataset.attach_labels(obs, labels)
        n_post_close = sum(1 for o in obs if o["post_close"])
        obs_sup = [o for o in obs if o["label"] is not None
                   and not o["post_close"]]

        if train_end_ms is None and obs_sup:
            train_end_ms, val_end_der = dataset.boundaries_from_fractions(
                obs_sup, train_frac=train_frac, val_frac=val_frac)
            if val_end_ms is None:
                val_end_ms = val_end_der
        elif val_end_ms is None and val_frac > 0:
            # Same rule as train.py: explicit train boundary +
            # validation fraction → `val_frac` quantile of the temporal mass
            # AFTER the embargo.
            embargo_eff = (dataset.EMBARGO_DEFAULT_MS
                           if embargo_ms is None else embargo_ms)
            post = [o for o in obs_sup
                    if o["decision_ts_ms"] > train_end_ms + embargo_eff]
            if post:
                val_end_ms = dataset.boundary_from_fraction(
                    post, val_frac)
                if val_end_ms <= train_end_ms:
                    val_end_ms = None
        if train_end_ms is not None:
            parts, split_meta = dataset.temporal_split(
                obs_sup, train_end_ms=train_end_ms,
                val_end_ms=val_end_ms, embargo_ms=embargo_ms)
            # Availability cutoff = fit instant (post-capture):
            # all the tape's labels are already known when building
            # the dataset. A cutoff at the train boundary would simulate
            # training AT the boundary — not the real scenario. An explicit
            # `cutoff_fit_ms` can be passed to exercise/audit
            # the exclusion of labels not yet available.
            knowns = [l["known_at_ms"] for l in labels.values()
                      if l.get("known_at_ms") is not None]
            if cutoff_fit_ms is None:
                cutoff_fit_ms = max(knowns) if knowns else train_end_ms
            train_obs, excl_train = dataset.filter_labels_available(
                parts["train"], cutoff_ms=cutoff_fit_ms)
            val_obs, excl_val = dataset.filter_labels_available(
                parts["validation"],
                cutoff_ms=cutoff_fit_ms) if val_end_ms is not None else (
                    [], [])
            rep_labels["cutoff_fit_ms"] = cutoff_fit_ms
            obs_stats = {
                "n_obs_total": len(obs),
                "n_obs_post_close": n_post_close,
                "n_supervised_obs": len(obs_sup),
                "n_obs_train_usables": len(train_obs),
                "n_late_label_or_no_availability_train":
                    len(excl_train),
                "n_late_label_or_no_availability_validation":
                    len(excl_val),
                "n_obs": {n: len(parts[n]) for n in parts},
                "n_markets": {
                    n: len({o["market_cid"] for o in parts[n]})
                    for n in parts},
            }
            rep_labels["available_at_cutoff_fit"] = len({
                o["market_cid"] for o in train_obs})
            rep_labels["excluded_at_cutoff_fit"] = len({
                o["market_cid"] for o in excl_train})

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - t0, 2),
        "kind": "dataset_researchevent_tape",
        "sources": [
            {"path": p, **stats["by_file"][p]} for p in sources
        ],
        "capture_manifest_verification": verification,
        "capture_window_ms": [stats["min_avail"], stats["max_avail"]],
        "capture_window_iso": [_iso(stats["min_avail"]),
                              _iso(stats["max_avail"])],
        "n_events": stats["n_events"],
        "counts_by_kind": stats["counts_by_kind"],
        "quality": {
            "quality_histogram": {
                str(k): v for k, v in
                sorted(stats["quality_histogram"].items())},
            "empty_lines": stats["empty_lines"],
            "malformed": 0,
            "inversions_source": stats["inversions_source"],
            "max_shift": stats["max_shift"],
            "reorder_window": REORDER_WINDOW,
        },
        "order": (
            "local_seq reassigned by (avail_ts_ms, local_seq_orig): "
            "canonical order of the contract; bounded local inversions "
            "of the recorder reordered by window and accounted in "
            "quality.inversions_source/max_shift"),
        "reassigned_local_seq": stats.get("reassigned_local_seq", False),
        "decision_grid_ms": decision_min_interval_ms,
        "labels": rep_labels,
        "split": {
            "boundaries": (split_meta or {}).get("boundaries"),
            "splits": (split_meta or {}).get("splits"),
            "purged_markets": (split_meta or {}).get("purged_markets"),
            "embargo_markets": (split_meta or {}).get("embargo_markets"),
            "stats": obs_stats,
        },
        "notes": [
            "Real ResearchEvent tape from the recorder — original capture avail_ts_ms and "
            "local_seq (not reconstructed from "
            "SQL). Labels only from official resolution/market_meta.outcome events "
            "with a real known_at (payload; fallback avail_ts_ms "
            "of the event itself — never clear_at nor backdating).",
            "decision_grid_ms limits the emission of observations per "
            "market; events suppressed as a decision point "
            "are still applied to the causal state.",
            "The historical SQL data (t8b_confirmed/t8b_provisional) "
            "is marked NO_APTO_LEAKAGE — this dataset is the only "
            "evidence aligned with the new captures.",
        ],
        "files": [],
    }
    if tape is not None:
        manifest["files"].append({
            "path": tape.name, "sha256": _sha256_file(tape),
            "bytes": tape.stat().st_size})
    mpath = out / "dataset_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"manifest": manifest, "tape": str(tape) if tape else None,
            "labels": labels,
            # In-memory partitions (they do not go to the manifest — they are per
            # observation, not per file): for tests and consumers
            # who want to write the already partitioned dataset.
            "partitions": parts if do_split and split_meta else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--events-dir",
                     help="capture directory <source>/<date>.jsonl")
    src.add_argument("--events", nargs="+",
                     help="explicit list of event jsonl file paths")
    ap.add_argument("--manifest", default=None,
                    help="capture manifest (files[].sha256) to "
                         "verify provenance — mismatch = failure")
    ap.add_argument("--out", required=True,
                    help="output directory of the dataset")
    ap.add_argument("--no-tape", action="store_true",
                    help="do not write the canonical events.jsonl (stats only; "
                         "passes re-merge from the sources)")
    ap.add_argument("--skip-obs", action="store_true",
                    help="only merge + labels, no observations/split "
                         "pass (quick diagnostic)")
    ap.add_argument("--decision-min-interval-ms", type=int,
                    default=DECISION_GRID_DEFAULT_MS,
                    help="decision time mesh per market "
                         f"(default {DECISION_GRID_DEFAULT_MS})")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--train-end-ms", type=int, default=None)
    ap.add_argument("--val-end-ms", type=int, default=None)
    ap.add_argument("--embargo-ms", type=int, default=None)
    ap.add_argument("--cutoff-fit-ms", type=int, default=None,
                    help="fit instant for label availability "
                         "(default: last known_at_ms of the "
                         "tape — the real fit happens post-capture)")
    args = ap.parse_args(argv)

    sources = (discover_sources(args.events_dir) if args.events_dir
               else [str(Path(p)) for p in args.events])
    res = run(
        sources, args.out, manifest_capture=args.manifest,
        write_tape=not args.no_tape,
        decision_min_interval_ms=args.decision_min_interval_ms,
        train_frac=args.train_frac, val_frac=args.val_frac,
        train_end_ms=args.train_end_ms, val_end_ms=args.val_end_ms,
        embargo_ms=args.embargo_ms, do_split=not args.skip_obs,
        cutoff_fit_ms=args.cutoff_fit_ms)
    m = res["manifest"]
    print(json.dumps({
        "n_events": m["n_events"],
        "iso_window": m["capture_window_iso"],
        "counts_by_kind": m["counts_by_kind"],
        "labels": m["labels"],
        "split_stats": (m["split"] or {}).get("stats"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
