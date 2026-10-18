"""Tests for `dataset_tape.py` — SYNTHETIC fixtures
test-only: every event/tape of this file is generated inside the test
with made-up timestamps (base T0); NO data comes from real captures or
DBs.

Verify the real tape dataset pipeline:
1. K-way merge by (avail_ts_ms, local_seq) with `local_seq` reassigned
   to the canonical order of the contract.
2. Explicit failure on a non-monotonic source / malformed JSON / sha256
   that does not match the capture manifest — never tolerated.
3. Labels ONLY from official resolution/market_meta.outcome events:
   real known_at from the payload; fallback to the event's own avail_ts_ms
   (never clear_at nor backdating).
4. Purged/embargoed split reused from dataset.py + report of label
   availability at the train cutoff.
5. Decision grid `decision_min_interval_ms` (one obs every N ms per
   market; suppressed events are still applied to the state).

Run with:
    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s \
        research/numeric_model/tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# The package intentionally has no __init__.py: the tests import the
# flat modules by adding the parent directory to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
import dataset_tape  # noqa: E402

T0 = 1_800_000_000_000  # epoch ms base — made-up fixture timestamps, test-only


def _ev(seq, avail, kind, cid=None, source="polymarket_ws", payload=None,
        quality=0, asset=None, token=None):
    """Minimal ResearchEvent — fixtures ONLY."""
    return {
        "schema_version": 1, "source": source, "kind": kind,
        "market_cid": cid, "token_id": token, "asset": asset,
        "payload": payload or {},
        "origin_ts_ms": None, "recv_ts_ms": avail, "avail_ts_ms": avail,
        "local_seq": seq, "quality": quality,
    }


def _book(cid, tok, bid, ask, avail, seq):
    """Per-token book_snapshot in the REAL Polymarket WS format
    (levels as {"price","size"} dicts) — test-only fixture."""
    return _ev(seq, avail, "book_snapshot", cid, token=tok, payload={
        "bids": [{"price": str(bid), "size": "10"}],
        "asks": [{"price": str(ask), "size": "10"}]})


def _delta(cid_payload, changes, avail, seq):
    """book_delta in the REAL WS format: top-level market_cid null,
    market in payload.market and each price_change with its asset_id —
    test-only fixture."""
    return _ev(seq, avail, "book_delta", None, payload={
        "market": cid_payload, "price_changes": changes})


def _meta_payload(cid, winner=None, known_ms=None):
    """Synthetic MarketMeta (test-only)."""
    p = {
        "schema_version": 1, "market_cid": cid,
        "slug": f"btc-updown-{cid}", "yes_token_id": f"{cid}-yes",
        "no_token_id": f"{cid}-no", "asset": "btc", "neg_risk": False,
        "market_open_ts": T0 // 1000, "clear_at": T0 // 1000 + 3600,
        "reference_price": "100.0", "rule": "up_down_reference",
        "resolution_source": "binance_ws",
        "resolution_source_ref": "BTCUSDT spot",
        "rules_windows": [], "outcome": None,
    }
    if winner is not None:
        p["outcome"] = {"winner": winner, "known_at_ms": known_ms,
                        "provenance": "gamma:fixture-test"}
    return p


def _resolution(cid, winner, known_ms=None, avail=None):
    """Synthetic official resolution event (test-only) — like the ones in
    post3 tape: gamma provenance + known_at = real fetch ts."""
    av = T0 + (avail if avail is not None else 0)
    return _ev(0, av, "resolution", cid, source="polymarket_rest",
               payload={"winner": winner, "known_at_ms": known_ms,
                        "provenance": "gamma-api:fixture-test",
                        "reconcile": "confirmed"}, quality=8)


def _writes_source(dirpath, source, events):
    """Writes a source `<dir>/<source>/20260924.jsonl` with the given events
    (one ResearchEvent per line — test-only fixture)."""
    d = Path(dirpath) / source
    d.mkdir(parents=True, exist_ok=True)
    p = d / "20260924.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return p


class CanonicalMergeTest(unittest.TestCase):
    """The merge orders by (avail, seq) and reassigns local_seq."""

    def test_merge_two_interleaved_sources(self):
        with tempfile.TemporaryDirectory() as d:
            # Source A: even seqs; source B: odd seqs — the canonical order
            # interleaves by avail, not by file.
            evs_a = [_ev(0, T0 + 100, "price_tick", asset="btc",
                         payload={"price": "100"}, source="binance_ws"),
                     _ev(2, T0 + 300, "price_tick", asset="btc",
                         payload={"price": "101"}, source="binance_ws")]
            evs_b = [_book("m1", "m1-yes", "0.4", "0.5", T0 + 200, 1),
                     _book("m1", "m1-yes", "0.41", "0.51", T0 + 400, 3)]
            pa = _writes_source(d, "binance_ws", evs_a)
            pb = _writes_source(d, "polymarket_ws", evs_b)

            stats = dataset_tape._stats_base()
            merged = list(dataset_tape.iter_canonical_tape(
                [str(pa), str(pb)], stats))

            self.assertEqual(
                [e["avail_ts_ms"] for e in merged],
                [T0 + 100, T0 + 200, T0 + 300, T0 + 400])
            # local_seq reassigned to 0..n-1 in the canonical order.
            self.assertEqual(
                [e["local_seq"] for e in merged], [0, 1, 2, 3])
            self.assertEqual(stats["n_events"], 4)
            self.assertEqual(stats["counts_by_kind"],
                             {"price_tick": 2, "book_snapshot": 2})
            self.assertEqual(stats["min_avail"], T0 + 100)
            self.assertEqual(stats["max_avail"], T0 + 400)
            # sha256 per file present and verifiable.
            for p in (str(pa), str(pb)):
                self.assertEqual(
                    stats["by_file"][p]["sha256"],
                    dataset_tape._sha256_file(p))

    def test_bounded_inversion_is_reordered_and_counted(self):
        """Bounded local inversion (recorder write race):
        the merge emits the canonical order and counts the inversion and the
        shift — never a silent reordering."""
        with tempfile.TemporaryDirectory() as d:
            evs = [_ev(0, T0 + 300, "price_tick", asset="btc",
                       payload={"price": "1"}, source="binance_ws"),
                   _ev(1, T0 + 100, "price_tick", asset="btc",
                       payload={"price": "2"}, source="binance_ws")]
            pa = _writes_source(d, "binance_ws", evs)
            stats = dataset_tape._stats_base()
            out = list(dataset_tape.iter_canonical_tape(
                [str(pa)], stats))
            self.assertEqual(
                [e["avail_ts_ms"] for e in out], [T0 + 100, T0 + 300])
            self.assertEqual(stats["inversions_source"], 1)
            self.assertEqual(stats["max_shift"], 1)

    def test_shift_greater_than_window_fails(self):
        """Real shift >= REORDER_WINDOW: the source is not
        reorderable within the bounded window → explicit failure."""
        with tempfile.TemporaryDirectory() as d:
            n = dataset_tape.REORDER_WINDOW + 16
            evs = [_ev(i, T0 + i, "price_tick", asset="btc",
                       payload={"price": str(i)}, source="binance_ws")
                   for i in range(n)]
            # The first event has a LARGER avail than all the others:
            # it must shift n-1 positions → > window → failure.
            evs[0]["avail_ts_ms"] = T0 + n + 100
            pa = _writes_source(d, "binance_ws", evs)
            with self.assertRaises(dataset_tape.CorruptTape):
                list(dataset_tape.iter_canonical_tape([str(pa)]))

    def test_malformed_json_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d) / "polymarket_ws"
            dd.mkdir()
            p = dd / "20260924.jsonl"
            p.write_text('{"ok":1}\n{this is not json}\n')
            with self.assertRaises(dataset_tape.CorruptTape):
                list(dataset_tape.iter_canonical_tape([str(p)]))

    def test_discover_sources_ignores_root_helpers(self):
        """Root `_*.jsonl` files (recorder stats, preflight) are not
        contract events and are ignored — only `<source>/*.jsonl` counts."""
        with tempfile.TemporaryDirectory() as d:
            Path(d, "_recorder_stats.jsonl").write_text("{}\n")
            _writes_source(d, "internal", [
                _ev(0, T0, "order_submit", "m1")])
            files = dataset_tape.discover_sources(d)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].parent.name, "internal")


class LabelsTapeTest(unittest.TestCase):
    """Labels only from official resolution/meta.outcome, real known_at."""

    def test_resolution_known_at_from_payload(self):
        evs = [_resolution("m1", "yes", known_ms=T0 + 5000,
                           avail=5000)]
        labels = dataset.build_labels(evs)
        self.assertEqual(labels["m1"]["winner"], "yes")
        self.assertEqual(labels["m1"]["known_at_ms"], T0 + 5000)
        self.assertEqual(labels["m1"]["known_at_source"], "payload")

    def test_resolution_without_known_at_uses_event_avail(self):
        """Without known_at in the payload, known_at is the REAL avail_ts_ms
        of the event (when it reached the tape) — never clear_at nor 0."""
        ev = _resolution("m1", "no", known_ms=None, avail=7777)
        labels = dataset.build_labels([ev])
        self.assertEqual(labels["m1"]["known_at_ms"], T0 + 7777)
        self.assertEqual(labels["m1"]["known_at_source"], "avail_ts_ms")

    def test_market_without_result_produces_no_label(self):
        evs = [_ev(0, T0 + 100, "market_meta", "m1",
                   payload=_meta_payload("m1"))]
        self.assertEqual(dataset.build_labels(evs), {})


class RunPipelineTest(unittest.TestCase):
    """run(): merge + labels + certified split + manifest."""

    def _tape(self, d):
        """Two sources with 4 markets: m1/m2 labelled early, m3
        late; m4 crosses the boundary (purged). Test-only fixture."""
        books, rest = [], []
        seq = 0
        metas = [
            ("m1", "yes", 4_000), ("m2", "no", 150_000),
            ("m3", "yes", 240_000), ("m4", "no", 250_000),
        ]
        for i, (cid, win, k) in enumerate(metas):
            rest.append(_ev(seq, T0 + 50 + i, "market_meta", cid,
                            source="polymarket_rest",
                            payload=_meta_payload(cid)))
            seq += 1
            rest.append(_resolution(cid, win, known_ms=T0 + k,
                                    avail=k))
            seq += 1
        # Books (YES + NO per market, real dict format): m1/m2 in
        # the early window; m3/m4 in the late one; m4 with obs on both sides of
        # the boundary → purged.
        for cid, t0 in (("m1", 1_000), ("m2", 1_500), ("m3", 200_000),
                        ("m4", 3_000)):
            for j in range(3):
                t = T0 + t0 + j * 2_000
                books.append(_book(cid, f"{cid}-yes", "0.4", "0.5",
                                   t, seq))
                seq += 1
                books.append(_book(cid, f"{cid}-no", "0.5", "0.6",
                                   t, seq))
                seq += 1
        # m4 also post-boundary (train_end will be ~quantile of the mass).
        books.append(_book("m4", "m4-yes", "0.4", "0.5",
                           T0 + 220_000, seq))
        seq += 1
        # Each source must come ordered by (avail, seq) — invariant
        # of the tape contract that the merge verifies.
        rest.sort(key=lambda e: (e["avail_ts_ms"], e["local_seq"]))
        books.sort(key=lambda e: (e["avail_ts_ms"], e["local_seq"]))
        _writes_source(d, "polymarket_rest", rest)
        _writes_source(d, "polymarket_ws", books)
        return d

    def test_full_run_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            self._tape(d)
            out = Path(d) / "out"
            res = dataset_tape.run(
                dataset_tape.discover_sources(d), out,
                train_end_ms=T0 + 100_000)
            m = res["manifest"]

            # Dataset written + real hash in files.
            tape = out / "events.jsonl"
            self.assertTrue(tape.exists())
            self.assertEqual(
                m["files"][0]["sha256"],
                dataset_tape._sha256_file(tape))

            # Provenance: each source with path+sha256.
            self.assertEqual(len(m["sources"]), 2)
            for f in m["sources"]:
                self.assertTrue(f["sha256"])
                self.assertTrue(f["n_events"] > 0)

            # Real capture window.
            self.assertEqual(
                m["capture_window_ms"][0], T0 + 50)

            # Counts per kind (2 books — YES+NO — per market/step).
            ck = m["counts_by_kind"]
            self.assertEqual(ck["market_meta"], 4)
            self.assertEqual(ck["resolution"], 4)
            self.assertEqual(ck["book_snapshot"], 25)

            # Labels: 4 official with real known_at.
            self.assertEqual(m["labels"]["n_labels"], 4)
            self.assertEqual(
                m["labels"]["with_known_at_payload"], 4)

            # Split: m4 crosses the boundary → purged whole; m3 whole
            # in test; never the same market in two partitions.
            st = m["split"]["stats"]
            test_mk = {o["market_cid"]
                       for o in res["partitions"]["test"]}
            train_mk = {o["market_cid"]
                        for o in res["partitions"]["train"]}
            self.assertIn("m3", test_mk)
            self.assertFalse(train_mk & test_mk)
            purged = {p["market_cid"]
                    for p in m["split"]["purged_markets"]}
            self.assertIn("m4", purged)
            self.assertFalse(purged & (train_mk | test_mk))
            self.assertGreaterEqual(st["n_obs_train_usables"], 1)

    def test_late_labels_reported_at_cutoff(self):
        """Labels with known_at later than the fit cutoff count as
        excluded — the manifest declares the real availability. The
        default cutoff is the last known_at (post-capture fit);
        here an explicit one is passed to exercise the exclusion."""
        with tempfile.TemporaryDirectory() as d:
            self._tape(d)
            out = Path(d) / "out"
            # very early cutoff: only m1/m2 labels available in time
            # (known_at 4000/4500 < 100_000); m3/m4 late.
            res = dataset_tape.run(
                dataset_tape.discover_sources(d), out,
                train_end_ms=T0 + 100_000,
                cutoff_fit_ms=T0 + 100_000)
            lab = res["manifest"]["labels"]
            st = res["manifest"]["split"]["stats"]
            # m2 has obs in train but its label (known_at=150_000) did not
            # exist at the cutoff → excluded from the fit and counted.
            self.assertEqual(lab["excluded_at_cutoff_fit"], 1)
            self.assertEqual(
                st["n_late_label_or_no_availability_train"], 3)
            # m1 stays usable; m3 stays in test (retrospective eval OK
            # with a late label — the rule only vetoes the train fit).
            self.assertEqual(lab["available_at_cutoff_fit"], 1)
            self.assertEqual(lab["cutoff_fit_ms"], T0 + 100_000)
            self.assertEqual(
                {o["market_cid"] for o in res["partitions"]["test"]},
                {"m3"})

    def test_capture_manifest_verification(self):
        """Source sha256 checked against the capture manifest:
        match → ok; mismatch → CorruptTape (altered evidence)."""
        with tempfile.TemporaryDirectory() as d:
            self._tape(d)
            sources = dataset_tape.discover_sources(d)
            cap = {"research_dir": d, "files": [
                {"path": str(Path(p).relative_to(d)),
                 "sha256": dataset_tape._sha256_file(p)}
                for p in sources]}
            mpath = Path(d) / "captura.json"
            mpath.write_text(json.dumps(cap))
            res = dataset_tape.run(sources, Path(d) / "ok",
                                   manifest_capture=str(mpath))
            ver = res["manifest"]["capture_manifest_verification"]
            self.assertTrue(all(f["verified"] for f in ver["files"]))

            # Mismatch → explicit failure.
            cap["files"][0]["sha256"] = "0" * 64
            mpath.write_text(json.dumps(cap))
            with self.assertRaises(dataset_tape.CorruptTape):
                dataset_tape.run(sources, Path(d) / "bad",
                                 manifest_capture=str(mpath))


class GridDecisionTest(unittest.TestCase):
    """decision_min_interval_ms: at most 1 obs every N ms per market."""

    def test_grid_limits_observations_per_market(self):
        evs = [_ev(0, T0 + 50, "market_meta", "m1",
                   payload=_meta_payload("m1"))]
        for i in range(10):
            evs.append(_book("m1", "m1-yes", "0.4", "0.5",
                             T0 + 100 + i * 100, i + 1))
        all_items = dataset.build_observations_incremental(evs)
        self.assertEqual(len(all_items), 10)
        gridded = dataset.build_observations_incremental(
            evs, decision_min_interval_ms=300)
        ts = [o["decision_ts_ms"] for o in gridded]
        # With a 300 ms grid over 100 ms steps: ~1 in 3.
        self.assertEqual(ts, [T0 + 100, T0 + 400, T0 + 700, T0 + 1000])
        # Each emitted obs still sees the updated book (the suppressed deltas
        # were applied to the causal state anyway).
        self.assertEqual(gridded[-1]["features"]["yes_bid"], 0.4)

    def test_delta_real_ws_format_applies_to_state(self):
        """book_delta from the real WS: top-level market_cid null, market in
        payload.market and each price_change with its own asset_id. Without
        this path the delta was not applied and the book stayed frozen at the
        snapshot — a regression in the real tape format."""
        evs = [
            _ev(0, T0 + 50, "market_meta", "m1",
                payload=_meta_payload("m1")),
            _book("m1", "m1-yes", "0.40", "0.50", T0 + 100, 1),
            _delta("m1", [{"asset_id": "m1-yes", "price": "0.45",
                           "side": "BUY", "size": "7"},
                          {"asset_id": "m1-yes", "price": "0.50",
                           "side": "SELL", "size": "0"}],
                   T0 + 200, 2),
            # A trade emitted AFTER the delta is the decision point
            # that drains the pending deltas: it sees the already updated book.
            _ev(3, T0 + 3000, "trade", "m1",
                payload={"price": "0.45", "side": "BUY", "size": "7"}),
            _book("m1", "m1-yes", "0.41", "0.51", T0 + 5000, 4),
        ]
        obs = dataset.build_observations_incremental(
            evs, decision_min_interval_ms=1000)
        # The delta (+200, within the 1 s grid) is suppressed as a
        # decision point BUT is applied to the state: the trade's obs
        # (+3000) sees bid 0.45 and the ask 0.50 removed by size=0.
        self.assertEqual(len(obs), 3)
        self.assertEqual(obs[0]["market_cid"], "m1")
        self.assertEqual(obs[1]["features"]["yes_bid"], 0.45)
        self.assertIsNone(obs[1]["features"]["yes_ask"])
        # The next snapshot's obs sees the replaced book.
        self.assertEqual(obs[2]["features"]["yes_bid"], 0.41)


if __name__ == "__main__":
    unittest.main()
