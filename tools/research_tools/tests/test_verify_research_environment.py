#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for `tools/research_tools/verify_research_environment.py` — synthetic run-dir and repo
in tmpdir (the real run and the repo are never touched).

Covers gates G1–G9: each one with a pass, a fail and a pending scenario
where applicable, plus the exit-code semantics (--strict).

Stdlib only (`unittest`). Run:
    python3 -m unittest discover -s tools/research_tools/tests -p "test_verify_research_environment.py" -v
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import verify_research_environment as vre  # noqa: E402


def _ev(seq, recv=1000, avail=None, source="binance_ws", kind="price_tick"):
    return {
        "schema_version": 1,
        "source": source,
        "kind": kind,
        "market_cid": None,
        "token_id": None,
        "asset": "btc",
        "payload": {},
        "origin_ts_ms": None,
        "recv_ts_ms": recv,
        "avail_ts_ms": avail if avail is not None else recv,
        "local_seq": seq,
        "quality": 0,
    }


def _write_jsonl(path, objs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(o) + "\n" for o in objs), encoding="utf-8"
    )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _v2_report(**over):
    """Replay report conforming to contract v2 — SYNTHETIC test fixture,
    not market evidence."""
    r = {
        "fidelity": "certified",
        "fidelity_reasons": [],
        "external_fidelity": "unidentified",
        "reconciled": True,
        "digest": "POS|abc|1\nRECON|true|0.00000",
        "settlement_diagnostic": False,
        "settlement_corrections": 0,
        "settlements": 2,
        "settlement_pending": [],
        "quality": {
            "quality_marks": 0, "malformed": 0, "agent": 0,
            "discarded_deltas": 3, "scope_discarded_deltas": 0,
            "without_half": 0, "books_top_equal": 0, "resyncs": 0,
            "tokens_with_gap_scope": 0, "scope_gap_markets": [],
            "scope_deltas_pct_coverage": 99.9,
        },
    }
    r.update(over)
    return r


def _v2_model(root, window=(0, 1000)):
    """model.json + sibling manifest.json with complete v2 provenance
    (sources with sha256+window, label stats) — synthetic fixture.
    Returns (dir, model_path, model_sha256)."""
    d = Path(root) / "model"
    d.mkdir(parents=True, exist_ok=True)
    model = d / "model.json"
    model.write_text(json.dumps({
        "splits": [{"name": "train", "from_ms": 0, "to_ms": 400},
                   {"name": "test", "from_ms": 400, "to_ms": 1000}],
        "split_markets": {"train": ["a"], "test": ["b"]},
    }))
    source = d / "source_tape.jsonl"
    _write_jsonl(source, [_ev(1), _ev(2)])
    (d / "metrics_eval.json").write_text(json.dumps({"n_test": 10}))
    (d / "metrics_train.json").write_text(json.dumps({
        "n_train": 50, "test": {"n": 10}}))
    (d / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "research_dir": str(d),
        "files": [{"path": "model.json", "sha256": _sha(model)}],
        "sources": [{
            "path": "source_tape.jsonl",
            "sha256": _sha(source),
            "window_ms": list(window),
        }],
        "labels": {"n": 5, "known_at_ms_min": 100,
                   "known_at_ms_max": 900, "excluded": 1},
    }))
    return d, str(model), _sha(model)


def _comparable_calibration():
    """Calibration report with a clean label and one
    «comparable» comparison — synthetic fixture."""
    return {
        "certification_label": "external_fidelity_evaluable",
        "replay": {"profiles": {"nominal": {"strategies": {"arb": {
            "comparison": {"comparability": {"state": "comparable"}},
        }}}}},
    }


def _repo_fake(root):
    """Minimal repo: paper config + one strategy and a
    dataset manifest with a correct hash."""
    repo = Path(root) / "repo"
    (repo / "config" / "strategies").mkdir(parents=True)
    (repo / "config" / "general.toml").write_text(
        'mode = "paper"\n', encoding="utf-8"
    )
    (repo / "config" / "strategies" / "arb.toml").write_text(
        "enabled = true\n", encoding="utf-8"
    )
    data = repo / "research" / "data" / "t8b_confirmed"
    data.mkdir(parents=True)
    ev = data / "events.jsonl"
    ev.write_text("{}\n", encoding="utf-8")
    sha = hashlib.sha256(ev.read_bytes()).hexdigest()
    (data / "dataset_manifest.json").write_text(
        json.dumps({"files": [{"path": "events.jsonl",
                                  "sha256": sha}]}),
        encoding="utf-8",
    )
    return repo


def _run_dir(root):
    return Path(root) / "run"


class TestG1Splits(unittest.TestCase):
    def test_overlapping_splits_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "model.json"
            m.write_text(json.dumps({
                "splits": [
                    {"name": "train", "from_ms": 0, "to_ms": 100},
                    {"name": "test", "from_ms": 50, "to_ms": 200},
                ],
                "split_markets": {"train": ["a"], "test": ["b"]},
            }))
            r = vre.check_g1(str(m))
            self.assertEqual(r["state"], "fail")
            self.assertEqual(r["evidence"]["overlaps"][0]["overlap_ms"], 50)

    def test_shared_markets_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "model.json"
            m.write_text(json.dumps({
                "splits": [
                    {"name": "train", "from_ms": 0, "to_ms": 100},
                    {"name": "test", "from_ms": 200, "to_ms": 300},
                ],
                "split_markets": {"train": ["a", "x"], "test": ["x"]},
            }))
            r = vre.check_g1(str(m))
            self.assertEqual(r["state"], "fail")
            comp = r["evidence"]["shared_markets"]
            self.assertTrue(comp)
            self.assertIn("x", next(iter(comp.values())))

    def test_clean_splits_with_markets_is_pass(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "model.json"
            m.write_text(json.dumps({
                "splits": [
                    {"name": "train", "from_ms": 0, "to_ms": 100},
                    {"name": "validation", "from_ms": 100, "to_ms": 200},
                    {"name": "test", "from_ms": 200, "to_ms": 300},
                ],
                "split_markets": {"train": ["a"], "validation": ["b"],
                                  "test": ["c"]},
            }))
            r = vre.check_g1(str(m))
            self.assertEqual(r["state"], "pass")

    def test_splits_without_markets_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "model.json"
            m.write_text(json.dumps({
                "splits": [{"name": "train", "from_ms": 0, "to_ms": 100},
                           {"name": "test", "from_ms": 100, "to_ms": 200}],
            }))
            r = vre.check_g1(str(m))
            self.assertEqual(r["state"], "pending")

    def test_absent_model_is_pending(self):
        r = vre.check_g1(None)
        self.assertEqual(r["state"], "pending")

    def test_splits_from_sibling_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "model.json"
            m.write_text(json.dumps({"kind": "logistic"}))
            (Path(td) / "manifest.json").write_text(json.dumps({
                "splits": [{"name": "train", "from_ms": 0, "to_ms": 100},
                           {"name": "test", "from_ms": 100, "to_ms": 200}],
                "split_markets": {"train": ["a"], "test": ["b"]},
            }))
            r = vre.check_g1(str(m))
            self.assertEqual(r["state"], "pass")


class TestG2Capture(unittest.TestCase):
    def test_monotonic_events_pass(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            _write_jsonl(run / "events" / "binance_ws" / "20260923.jsonl",
                         [_ev(1), _ev(2), _ev(3)])
            _write_jsonl(run / "events" / "polymarket_ws" / "20260923.jsonl",
                         [_ev(4, source="polymarket_ws", kind="book_delta")])
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "pass")
            self.assertEqual(r["evidence"]["events_total"], 4)
            self.assertEqual(r["evidence"]["by_source"]["binance_ws"], 3)

    def test_non_monotonic_local_seq_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            _write_jsonl(run / "events" / "binance_ws" / "20260923.jsonl",
                         [_ev(3), _ev(1)])
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "fail")

    def test_bounded_inversion_passes_with_detail(self):
        # Concurrent write by the recorder: 1 inversion of distance 1
        # among 100 events does not destroy the canonical order (seq value).
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            evs = [_ev(i) for i in range(1, 51)] \
                + [_ev(52), _ev(51)] \
                + [_ev(i) for i in range(53, 101)]
            _write_jsonl(run / "events" / "binance_ws" / "20260923.jsonl",
                         evs)
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "pass")
            self.assertEqual(r["evidence"]["inversions_file"], 1)
            self.assertEqual(r["evidence"]["max_seq_shift"], 1)
            self.assertTrue(
                any("bounded seq inversions" in d for d in r["details"])
            )

    def test_large_shift_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            limit = vre.G2_MAX_SHIFT
            evs = [_ev(limit + 100)] + [_ev(i) for i in range(1, 60)]
            _write_jsonl(run / "events" / "binance_ws" / "20260923.jsonl",
                         evs)
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "fail")

    def test_avail_before_recv_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            _write_jsonl(run / "events" / "x" / "20260923.jsonl",
                         [_ev(1, recv=1000, avail=999)])
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "fail")

    def test_without_events_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            (run / "events").mkdir(parents=True)
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "pending")

    def _manifest_v1(self, run, name_str, files):
        """Segment manifest conforming to closed-segment/v1."""
        (run / "manifests").mkdir(parents=True, exist_ok=True)
        (run / "manifests" / name_str).write_text(json.dumps({
            "schema": "closed-segment/v1",
            "research_dir": str(run),
            "files": files,
            "totals": {"events": 1, "recv_ts_ms_min": 0,
                       "recv_ts_ms_max": 1000,
                       "recorder_stats_final": {"written": 1}},
            "utc_window": ["2026-09-23T15:16:11Z",
                            "2026-09-23T16:00:10Z"],
        }))

    def test_manifest_segment_with_correct_hash(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            fp = run / "events" / "s" / "20260923.jsonl"
            _write_jsonl(fp, [_ev(1)])
            _write_jsonl(run / "events" / "_recorder_stats.jsonl",
                         [{"written": 1, "discarded_total": 0}])
            self._manifest_v1(run, "seg1.json", [{
                "path": "events/s/20260923.jsonl", "sha256": _sha(fp)}])
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "pass", r["details"])
            m = r["evidence"]["segment_manifests"][0]
            self.assertEqual(m["verified"], 1)
            self.assertEqual(m["verdict"], "ok")

    def test_manifest_segment_bad_hash_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            fp = run / "events" / "s" / "20260923.jsonl"
            _write_jsonl(fp, [_ev(1)])
            self._manifest_v1(run, "seg1.json", [{
                "path": "events/s/20260923.jsonl",
                "sha256": "0" * 64}])
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "fail")
            self.assertEqual(
                r["evidence"]["segment_manifests"][0]["verdict"],
                "corrupt")

    def test_manifest_without_schema_is_pending(self):
        # Real post3_capture manifest kind: files+sha256 but no schema,
        # no window or totals → incomplete (insufficient), not fail.
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            fp = run / "events" / "s" / "20260923.jsonl"
            _write_jsonl(fp, [_ev(1)])
            (run / "manifests").mkdir(parents=True)
            (run / "manifests" / "post3.json").write_text(json.dumps({
                "research_dir": str(run),
                "files": [{"path": "events/s/20260923.jsonl",
                           "sha256": _sha(fp)}],
            }))
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "pending")
            m = r["evidence"]["segment_manifests"][0]
            self.assertEqual(m["verdict"], "incomplete")
            self.assertIn("schema", m["missing_fields"])

    def test_manifests_declared_heterogeneous_schema(self):
        # Replica of the real c1 vs post3 inconsistency: one declares a
        # schema and the other does not → pending + explicit detail.
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            fp = run / "events" / "s" / "20260923.jsonl"
            _write_jsonl(fp, [_ev(1)])
            self._manifest_v1(run, "c1_capture.json", [{
                "path": "events/s/20260923.jsonl", "sha256": _sha(fp)}])
            (run / "manifests" / "post3_capture.json").write_text(
                json.dumps({
                    "research_dir": str(run),
                    "files": [{"path": "events/s/20260923.jsonl",
                               "sha256": _sha(fp)}],
                }))
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "pending")
            self.assertTrue(any(
                "heterogeneous schemas" in d for d in r["details"]),
                r["details"])

    def test_manifest_absent_file_is_pending(self):
        # file declared but deleted from disk → sha256 not
        # verifiable → incomplete (pending), not fail.
        with tempfile.TemporaryDirectory() as td:
            run = _run_dir(td)
            fp = run / "events" / "s" / "20260923.jsonl"
            _write_jsonl(fp, [_ev(1)])
            self._manifest_v1(run, "seg1.json", [
                {"path": "events/s/20260923.jsonl", "sha256": _sha(fp)},
                {"path": "events/s/deleted.jsonl", "sha256": "a" * 64}])
            r = vre.check_g2(str(run))
            self.assertEqual(r["state"], "pending")
            m = r["evidence"]["segment_manifests"][0]
            self.assertEqual(m["absent"], ["events/s/deleted.jsonl"])


class TestG3G4Replay(unittest.TestCase):
    def _report(self, td, data):
        fp = Path(td) / "replay_report.json"
        fp.write_text(json.dumps(data))
        return str(fp)

    # ── negative regression: synthetic probe ──
    # Contradictory synthetic JSON that verifier v1 ACCEPTED.
    # Only tests the rejection — it is NEVER market evidence.
    PROBE_HERMES = {
        "fidelity": "certified",
        "digest": "RECON|false",
        "reconciled": False,
        "settlement_diagnostic": True,
        "settlements": 1,
    }

    def test_contradictory_synthetic_probe_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, dict(self.PROBE_HERMES))
            g3 = vre.check_g3(rp)
            g4 = vre.check_g4(rp)
            self.assertEqual(g3["state"], "fail", g3)
            self.assertEqual(g4["state"], "fail", g4)

    def test_complete_v2_report_passes(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report())
            self.assertEqual(vre.check_g3(rp)["state"], "pass")
            self.assertEqual(vre.check_g4(rp)["state"], "pass")

    def test_digest_with_recon_false_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                digest="POS|x\nRECON|false|9.9"))
            r = vre.check_g3(rp)
            self.assertEqual(r["state"], "fail")
            self.assertTrue(r["evidence"]["digest_recon_false"])

    def test_reconciled_false_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(reconciled=False))
            self.assertEqual(vre.check_g3(rp)["state"], "fail")
            self.assertEqual(vre.check_g4(rp)["state"], "fail")

    def test_reconciled_absent_in_v2_report_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            data = _v2_report()
            del data["reconciled"]
            rp = self._report(td, data)
            self.assertEqual(vre.check_g3(rp)["state"], "fail")

    def test_settlement_diagnostic_with_certified_is_fail(self):
        # Real name of the field (the bug read `diagnostic_settlement`).
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(settlement_diagnostic=True))
            self.assertEqual(vre.check_g4(rp)["state"], "fail")

    def test_legacy_diagnostic_true_with_certified_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(diagnostic_settlement=True))
            self.assertEqual(vre.check_g4(rp)["state"], "fail")

    def test_settlement_corrections_with_certified_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(settlement_corrections=1))
            self.assertEqual(vre.check_g4(rp)["state"], "fail")

    def test_certified_without_quality_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            data = _v2_report()
            del data["quality"]
            rp = self._report(td, data)
            r = vre.check_g3(rp)
            self.assertEqual(r["state"], "fail")
            self.assertTrue(any("certified without quality breakdown" in d
                                or "without quality breakdown" in d
                                for d in r["details"]), r["details"])

    def test_certified_quality_dirty_scope_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            data = _v2_report()
            data["quality"]["scope_discarded_deltas"] = 4
            rp = self._report(td, data)
            self.assertEqual(vre.check_g3(rp)["state"], "fail")

    def test_certified_quality_without_scope_metrics_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                quality={"quality_marks": 0}))
            self.assertEqual(vre.check_g3(rp)["state"], "fail")

    def test_asserted_external_fidelity_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                external_fidelity="verified_against_venue"))
            self.assertEqual(vre.check_g3(rp)["state"], "fail")

    def test_certified_non_empty_reasons_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                fidelity_reasons=["but_is_certified"]))
            self.assertEqual(vre.check_g3(rp)["state"], "fail")

    def test_reduced_without_v2_fields_is_pending(self):
        # Reduced pre-contract report: insufficient, not a defect.
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, {
                "fidelity": "reduced", "digest": "POS|x"})
            self.assertEqual(vre.check_g3(rp)["state"], "pending")

    def test_reduced_with_reasons_and_quality_passes(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                fidelity="reduced",
                fidelity_reasons=["scope_discarded_deltas=4"],
                reconciled=True,
                quality={"scope_discarded_deltas": 4,
                         "tokens_with_gap_scope": 1},
                settlement_diagnostic=True))
            self.assertEqual(vre.check_g3(rp)["state"], "pass")
            # G4: diagnostic in reduced is reported but settlements>0
            # is still a pass according to the contract.
            g4 = vre.check_g4(rp)
            self.assertEqual(g4["state"], "pass")
            self.assertTrue(any("settlement_diagnostic" in d
                                for d in g4["details"]))

    def test_without_fidelity_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            r = vre.check_g3(self._report(td, {"digest": "x"}))
            self.assertEqual(r["state"], "fail")

    def test_absent_report_is_pending(self):
        self.assertEqual(vre.check_g3(None)["state"], "pending")

    def test_positive_settlements_pass(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(settlements=3))
            self.assertEqual(vre.check_g4(rp)["state"], "pass")

    def test_zero_settlements_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                fidelity="reduced",
                fidelity_reasons=["window without closures"],
                settlements=0))
            self.assertEqual(vre.check_g4(rp)["state"], "pending")

    def test_explicit_settlement_pending_is_pending(self):
        # Explicitly declared pending → pending (not pass).
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                settlements=0,
                settlement_pending=["0xabc", "0xdef"]))
            r = vre.check_g4(rp)
            self.assertEqual(r["state"], "pending")
            self.assertTrue(any("pending" in d
                                for d in r["details"]))

    def test_applied_records_pass(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                settlements=[
                    {"market": "m1", "state": "applied"},
                    {"market": "m2", "state": "pending"},
                ]))
            self.assertEqual(vre.check_g4(rp)["state"], "pass")

    def test_pending_only_records_are_pending(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                settlements=[{"market": "m1", "state": "pending"}]))
            self.assertEqual(vre.check_g4(rp)["state"], "pending")

    def test_records_with_anomalous_state_are_pending(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._report(td, _v2_report(
                settlements=[{"market": "m1", "state": "reverted"}]))
            self.assertEqual(vre.check_g4(rp)["state"], "pending")


class TestG5Environment(unittest.TestCase):
    def _step(self, idx, terminal=False, end=None, reward=None):
        return {
            "schema_version": 1, "step_idx": idx,
            "reward": reward or {
                "price_component": "1.5", "fee_component": "-0.5",
                "risk_penalty": "0", "total": "1.0"},
            "terminal": terminal, "end_reason": end,
        }

    def test_valid_episode_passes(self):
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "ep.json"
            fp.write_text(json.dumps({
                "steps": [self._step(0),
                          self._step(1, terminal=True,
                                     end="market_resolved")],
            }))
            r = vre.check_g5(str(fp))
            self.assertEqual(r["state"], "pass")
            self.assertEqual(r["evidence"]["n_terminal"], 1)

    def test_reward_that_does_not_add_up_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "ep.jsonl"
            _write_jsonl(fp, [self._step(0, reward={
                "price_component": "1", "fee_component": "0",
                "risk_penalty": "0", "total": "2"})])
            self.assertEqual(vre.check_g5(str(fp))["state"], "fail")

    def test_terminal_without_end_reason_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "ep.jsonl"
            _write_jsonl(fp, [self._step(0, terminal=True, end=None)])
            self.assertEqual(vre.check_g5(str(fp))["state"], "fail")

    def test_without_steps_is_pending(self):
        self.assertEqual(vre.check_g5(None)["state"], "pending")


class TestG6Calibration(unittest.TestCase):
    def _report(self, td, data):
        fp = Path(td) / "metrics.json"
        fp.write_text(json.dumps(data))
        return str(fp)

    def test_clean_label_with_comparable_comparison_passes(self):
        with tempfile.TemporaryDirectory() as td:
            fp = self._report(td, _comparable_calibration())
            r = vre.check_g6(fp)
            self.assertEqual(r["state"], "pass", r["details"])
            self.assertEqual(r["evidence"]["n_comparable"], 1)

    def test_unidentified_label_is_pending(self):
        # The honest label of calibrate_paper v2 (external fidelity
        # unidentified) must NEVER produce a hidden pass.
        with tempfile.TemporaryDirectory() as td:
            fp = self._report(td, {
                "certification_label":
                    "verified_internal_consistency; "
                    "external_fidelity_unidentified"})
            self.assertEqual(vre.check_g6(fp)["state"], "pending")

    def test_not_comparable_comparison_is_pending(self):
        # Profile declared no_comparable/unidentified → insufficient,
        # honest, not a pass.
        with tempfile.TemporaryDirectory() as td:
            fp = self._report(td, {
                "certification_label":
                    "verified_internal_consistency; "
                    "external_fidelity_unidentified",
                "replay": {"profiles": {"nominal": {"strategies": {
                    "arb": {"comparison": {"comparability": {
                        "state": "no_comparable"}}}}}}},
            })
            self.assertEqual(vre.check_g6(fp)["state"], "pending")

    def test_clean_label_without_comparisons_is_fail(self):
        # Contradiction: claims certification without comparison evidence.
        with tempfile.TemporaryDirectory() as td:
            fp = self._report(td, {
                "certification_label": "external_fidelity_evaluable"})
            self.assertEqual(vre.check_g6(fp)["state"], "fail")

    def test_clean_label_with_all_not_comparable_is_fail(self):
        # Contradiction: the label claims it and every comparison says
        # no_comparable.
        with tempfile.TemporaryDirectory() as td:
            fp = self._report(td, {
                "certification_label": "external_fidelity_evaluable",
                "replay": {"profiles": {"nominal": {"strategies": {
                    "arb": {"comparison": {"comparability": {
                        "state": "no_comparable"}}}}}}},
            })
            self.assertEqual(vre.check_g6(fp)["state"], "fail")

    def test_v1_report_without_comparability_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            fp = self._report(td, {
                "version": 1,
                "profile_certification": {
                    "conservative": {"state": "out_of_tolerance"}}})
            self.assertEqual(vre.check_g6(fp)["state"], "fail")

    def test_absent_is_pending(self):
        self.assertEqual(vre.check_g6(None)["state"], "pending")


class TestG7Metrics(unittest.TestCase):
    def test_model_with_complete_provenance_passes(self):
        with tempfile.TemporaryDirectory() as td:
            d, model, _sha_m = _v2_model(td)
            r = vre.check_g7(str(d), model)
            self.assertEqual(r["state"], "pass", r["details"])
            self.assertEqual(r["evidence"]["model_sha256"], _sha_m)

    def test_ok_metrics_without_model_is_pending(self):
        # Metrics present but no model.json → provenance not
        # verifiable → pending, not pass.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "metrics_eval.json").write_text(json.dumps({"n_test": 100}))
            (d / "metrics_train.json").write_text(
                json.dumps({"n_train": 500}))
            self.assertEqual(vre.check_g7(str(d))["state"], "pending")

    def test_model_without_manifest_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "model.json").write_text("{}")
            (d / "metrics_eval.json").write_text(json.dumps({"n_test": 10}))
            (d / "metrics_train.json").write_text(
                json.dumps({"n_train": 50}))
            r = vre.check_g7(str(d), str(d / "model.json"))
            self.assertEqual(r["state"], "pending")
            self.assertTrue(any("manifest" in x for x in r["details"]))

    def test_manifest_without_sources_is_pending(self):
        # Model without dataset provenance from the tape → pending.
        with tempfile.TemporaryDirectory() as td:
            d, model, _ = _v2_model(td)
            man = json.loads((d / "manifest.json").read_text())
            del man["sources"]
            del man["labels"]
            (d / "manifest.json").write_text(json.dumps(man))
            r = vre.check_g7(str(d), model)
            self.assertEqual(r["state"], "pending")
            self.assertTrue(any("sources" in x for x in r["details"]))
            self.assertTrue(any("labels" in x for x in r["details"]))

    def test_model_sha_differs_from_manifest_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            d, model, _ = _v2_model(td)
            man = json.loads((d / "manifest.json").read_text())
            man["files"][0]["sha256"] = "0" * 64
            (d / "manifest.json").write_text(json.dumps(man))
            self.assertEqual(
                vre.check_g7(str(d), model)["state"], "fail")

    def test_bad_sha_source_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            d, model, _ = _v2_model(td)
            man = json.loads((d / "manifest.json").read_text())
            man["sources"][0]["sha256"] = "f" * 64
            (d / "manifest.json").write_text(json.dumps(man))
            self.assertEqual(
                vre.check_g7(str(d), model)["state"], "fail")

    def test_source_without_window_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            d, model, _ = _v2_model(td)
            man = json.loads((d / "manifest.json").read_text())
            del man["sources"][0]["window_ms"]
            (d / "manifest.json").write_text(json.dumps(man))
            r = vre.check_g7(str(d), model)
            self.assertEqual(r["state"], "pending")

    def test_inconsistent_train_eval_metrics_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            d, model, _ = _v2_model(td)
            (d / "metrics_train.json").write_text(json.dumps({
                "n_train": 50, "test": {"n": 999}}))
            self.assertEqual(
                vre.check_g7(str(d), model)["state"], "fail")

    def test_n_zero_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            d, model, _ = _v2_model(td)
            (d / "metrics_eval.json").write_text(json.dumps({"n_test": 0}))
            self.assertEqual(
                vre.check_g7(str(d), model)["state"], "fail")

    def test_absent_dir_is_pending(self):
        self.assertEqual(vre.check_g7(None)["state"], "pending")

    def test_one_absent_metrics_file_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            d, model, _ = _v2_model(td)
            (d / "metrics_train.json").unlink()
            self.assertEqual(
                vre.check_g7(str(d), model)["state"], "pending")


class TestG8Shadow(unittest.TestCase):
    MODEL_SHA = "a" * 64
    SOURCES = [(0, 1000)]

    def _line(self, kind, **kw):
        return {"kind": kind, "schema_version": 1, "ts_ms": 1, **kw}

    def _stats_ok(self):
        return self._line(
            "stats", n_decisiones=1, n_abstentions=1,
            p50_ms=1, p95_ms=3, p99_ms=5, discarded_total=0,
            window_ms=[10, 900], emits_orders=False)

    def test_valid_record_tied_to_model_passes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_jsonl(d / "shadow-20260923.jsonl", [
                self._line("decision", latency_ms=3,
                            model_hash=self.MODEL_SHA),
                self._line("make_abstention", reason="no_edge",
                            model_hash=self.MODEL_SHA),
                self._stats_ok(),
            ])
            r = vre.check_g8(str(d), self.MODEL_SHA, self.SOURCES)
            self.assertEqual(r["state"], "pass", r["details"])
            self.assertEqual(r["evidence"]["model_hash_ok"], 2)

    def test_row_with_different_hash_in_the_middle_is_fail(self):
        # A single decision row with a different hash in the middle of the file
        # → fail with an exact count.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            rows = [self._line("decision", model_hash=self.MODEL_SHA)
                     for _ in range(50)]
            rows.insert(25, self._line(
                "decision", model_hash="b" * 64))
            rows.append(self._stats_ok())
            _write_jsonl(d / "shadow.jsonl", rows)
            r = vre.check_g8(str(d), self.MODEL_SHA, self.SOURCES)
            self.assertEqual(r["state"], "fail")
            self.assertEqual(r["evidence"]["model_hash_mismatch"], 1)
            self.assertEqual(r["evidence"]["model_hash_ok"], 50)

    def test_row_without_hash_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_jsonl(d / "shadow.jsonl", [
                self._line("decision", model_hash=self.MODEL_SHA),
                self._line("decision"),  # no model_hash
                self._stats_ok(),
            ])
            r = vre.check_g8(str(d), self.MODEL_SHA, self.SOURCES)
            self.assertEqual(r["state"], "fail")
            self.assertEqual(r["evidence"]["absent_model_hash"], 1)

    def test_without_verified_model_is_pending(self):
        # Without the sha256 of model the binding cannot be checked.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_jsonl(d / "shadow.jsonl", [
                self._line("decision"), self._stats_ok()])
            self.assertEqual(vre.check_g8(str(d))["state"], "pending")

    def test_intent_line_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            _write_jsonl(Path(td) / "shadow.jsonl", [
                self._line("decision", model_hash=self.MODEL_SHA),
                self._line("order_intent"),
            ])
            self.assertEqual(vre.check_g8(
                str(td), self.MODEL_SHA, self.SOURCES)["state"], "fail")

    def test_stats_without_latencies_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            _write_jsonl(Path(td) / "s.jsonl", [
                self._line("decision", model_hash=self.MODEL_SHA),
                self._line("stats", n_decisiones=1),
            ])
            self.assertEqual(vre.check_g8(
                str(td), self.MODEL_SHA, self.SOURCES)["state"], "fail")

    def test_window_outside_sources_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            stats = self._stats_ok()
            stats["window_ms"] = [5000, 9000]  # outside (0,1000)
            _write_jsonl(d / "s.jsonl", [
                self._line("decision", model_hash=self.MODEL_SHA),
                stats])
            self.assertEqual(vre.check_g8(
                str(d), self.MODEL_SHA, self.SOURCES)["state"], "fail")

    def test_emission_declaration_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            stats = self._stats_ok()
            stats["emits_orders"] = True
            _write_jsonl(d / "s.jsonl", [
                self._line("decision", model_hash=self.MODEL_SHA),
                stats])
            self.assertEqual(vre.check_g8(
                str(d), self.MODEL_SHA, self.SOURCES)["state"], "fail")

    def test_without_no_emission_declaration_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            stats = self._stats_ok()
            del stats["emits_orders"]
            _write_jsonl(d / "s.jsonl", [
                self._line("decision", model_hash=self.MODEL_SHA),
                stats])
            self.assertEqual(vre.check_g8(
                str(d), self.MODEL_SHA, self.SOURCES)["state"], "pending")

    def test_without_declared_window_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            stats = self._stats_ok()
            del stats["window_ms"]
            _write_jsonl(d / "s.jsonl", [
                self._line("decision", model_hash=self.MODEL_SHA),
                stats])
            self.assertEqual(vre.check_g8(
                str(d), self.MODEL_SHA, self.SOURCES)["state"], "pending")

    def test_without_stats_is_pending(self):
        with tempfile.TemporaryDirectory() as td:
            _write_jsonl(Path(td) / "s.jsonl", [
                self._line("decision", model_hash=self.MODEL_SHA)])
            self.assertEqual(vre.check_g8(
                str(td), self.MODEL_SHA, self.SOURCES)["state"], "pending")

    def test_absent_dir_is_pending(self):
        self.assertEqual(vre.check_g8(None)["state"], "pending")


class TestG9Integrity(unittest.TestCase):
    def test_healthy_repo_and_run_outside_passes(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            run = _run_dir(td)
            run.mkdir()
            r = vre.check_g9(repo, str(run))
            self.assertEqual(r["state"], "pass", r["details"])

    def test_run_dir_inside_repo_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            r = vre.check_g9(repo, str(repo / "run"))
            self.assertEqual(r["state"], "fail")

    def test_mode_live_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            (repo / "config" / "general.toml").write_text(
                'mode = "live"\n', encoding="utf-8"
            )
            self.assertEqual(vre.check_g9(repo, None)["state"], "fail")

    def test_corrupt_evidence_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            ev = repo / "research" / "data" / "t8b_confirmed" / "events.jsonl"
            ev.write_text("{}\n{}\n", encoding="utf-8")  # changes the hash
            r = vre.check_g9(repo, str(_run_dir(td)))
            self.assertEqual(r["state"], "fail")

    def test_env_in_run_dir_is_fail(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            run = _run_dir(td)
            run.mkdir()
            (run / ".env").write_text("SECRET=x\n", encoding="utf-8")
            r = vre.check_g9(repo, str(run))
            self.assertEqual(r["state"], "fail")


class TestMainEndToEnd(unittest.TestCase):
    def _full_run(self, td):
        """Synthetic run-dir with every artifact conforming to the
        contract v2 (test fixtures, not market evidence)."""
        run = _run_dir(td)
        ev_fp = run / "events" / "s" / "20260923.jsonl"
        _write_jsonl(ev_fp, [_ev(1), _ev(2)])
        _write_jsonl(run / "events" / "_recorder_stats.jsonl",
                     [{"written": 2, "discarded_total": 0}])
        (run / "manifests").mkdir(parents=True)
        (run / "manifests" / "captura.json").write_text(json.dumps({
            "schema": "closed-segment/v1",
            "research_dir": str(run),
            "files": [{"path": "events/s/20260923.jsonl",
                       "sha256": _sha(ev_fp)}],
            "totals": {"events": 2, "recv_ts_ms_min": 0,
                       "recv_ts_ms_max": 1000,
                       "recorder_stats_final": {"written": 2}},
            "utc_window": ["2026-09-23T15:16:11Z",
                            "2026-09-23T16:00:10Z"],
        }))
        _d, model, model_sha = _v2_model(run, window=(0, 1000))
        (run / "replay_report.json").write_text(
            json.dumps(_v2_report()))
        (run / "episodes.json").write_text(json.dumps({"steps": [{
            "step_idx": 0, "reward": {"price_component": "0",
                                      "fee_component": "0",
                                      "risk_penalty": "0", "total": "0"},
            "terminal": True, "end_reason": "horizon_reached"}]}))
        (run / "calib_metrics.json").write_text(
            json.dumps(_comparable_calibration()))
        _write_jsonl(run / "shadow" / "s.jsonl", [
            {"kind": "decision", "latency_ms": 1,
             "model_hash": model_sha},
            {"kind": "make_abstention", "reason": "no_edge",
             "model_hash": model_sha},
            {"kind": "stats", "p50_ms": 1, "p95_ms": 2, "p99_ms": 3,
             "window_ms": [10, 900], "emits_orders": False},
        ])
        return run

    def test_full_run_passes_all_gates(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            run = self._full_run(td)
            rc = vre.main([
                "--run-dir", str(run), "--repo", str(repo),
                "--model", str(run / "model" / "model.json"),
                "--replay-report", str(run / "replay_report.json"),
                "--env-log", str(run / "episodes.json"),
                "--calibration-report", str(run / "calib_metrics.json"),
                "--shadow-dir", str(run / "shadow"),
                "--out", str(Path(td) / "acceptance.json"),
            ])
            self.assertEqual(rc, 0)
            res = json.loads((Path(td) / "acceptance.json").read_text())
            self.assertEqual(res["verdict"], "pass")
            self.assertEqual(res["summary"]["pass"], 9)

    def test_pending_does_not_break_exit_without_strict(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            run = _run_dir(td)
            run.mkdir()  # empty run-dir → everything pending except G9
            rc = vre.main(["--run-dir", str(run), "--repo", str(repo)])
            self.assertEqual(rc, 0)

    def test_strict_converts_pending_into_failure(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            run = _run_dir(td)
            run.mkdir()
            rc = vre.main(
                ["--run-dir", str(run), "--repo", str(repo), "--strict"])
            self.assertEqual(rc, 1)

    def test_gate_failure_gives_exit_1(self):
        with tempfile.TemporaryDirectory() as td:
            repo = _repo_fake(td)
            (repo / "config" / "general.toml").write_text(
                'mode = "live"\n', encoding="utf-8")
            run = _run_dir(td)
            run.mkdir()
            rc = vre.main(["--run-dir", str(run), "--repo", str(repo)])
            self.assertEqual(rc, 1)


class TestRealArtifacts(unittest.TestCase):
    """Verification against the REAL artifacts in the run-dir
    $HRF_REAL_RUN_DIR (default research_runs/close-2026-09-23) — only where they DO comply
    or where non-compliance is already expected (documented in the test).
    Skipped if the run-dir does not exist."""

    RUN = Path(os.environ.get("HRF_REAL_RUN_DIR", "research_runs/close-2026-09-23"))
    # Byte cap to verify the real sha256 without paying the cost of
    # the files of several GB in a unit test.
    MAX_BYTES_TEST = 100 * 1024 * 1024

    def setUp(self):
        if not self.RUN.is_dir():
            self.skipTest("real run-dir not available on this machine")

    def _manifest_entries(self, name_str):
        m = json.loads((self.RUN / "manifests" / name_str).read_text())
        return m

    def test_real_sha256_verifiable_post3(self):
        # post3_capture: verifies the files < 100 MB against the
        # sha256 declared in the real manifest (real disk hash).
        m = self._manifest_entries("post3_capture.json")
        base = Path(m["research_dir"])
        verified, large = [], []
        for fe in m["files"]:
            fp = base / fe["path"]
            self.assertTrue(fp.is_file(), f"{fp} missing on disk")
            if fp.stat().st_size > self.MAX_BYTES_TEST:
                large.append(fe["path"])
                continue
            self.assertEqual(
                hashlib.sha256(fp.read_bytes()).hexdigest(),
                fe["sha256"], f"real sha256 ≠ manifest in {fe['path']}")
            verified.append(fe["path"])
        self.assertTrue(verified)

    def test_real_sha256_verifiable_c1(self):
        # c1_capture: small files of the real segment under
        # runtime/research_captures verified by hash.
        m = self._manifest_entries("c1_capture.json")
        base = Path(m["research_dir"])
        if not base.is_dir():
            self.skipTest("c1 research_dir not present")
        verified = 0
        for fe in m["files"]:
            fp = base / fe["path"]
            if not fp.is_file() \
                    or fp.stat().st_size > self.MAX_BYTES_TEST:
                continue
            self.assertEqual(
                hashlib.sha256(fp.read_bytes()).hexdigest(),
                fe["sha256"], f"real sha256 ≠ manifest in {fe['path']}")
            verified += 1
        self.assertGreater(verified, 0)

    def test_real_schema_inconsistency_c1_vs_post3(self):
        # The real inconsistency: c1 declares `closed-segment/v1` and
        # post3 declares no schema — the verifier makes it explicit.
        c1 = self._manifest_entries("c1_capture.json")
        post3 = self._manifest_entries("post3_capture.json")
        self.assertEqual(c1.get("schema"), "closed-segment/v1")
        self.assertIsNone(post3.get("schema"))
        self.assertIsNone(post3.get("utc_window"))
        self.assertIsNone(post3.get("totals"))

    def test_old_replay_report_fails_g3(self):
        # The old replay_report does NOT meet contract v2 (certified
        # without quality or external_fidelity) — expected: fail.
        rp = self.RUN / "replay_report.json"
        if not rp.is_file():
            self.skipTest("old replay_report not present")
        r = vre.check_g3(str(rp))
        self.assertEqual(r["state"], "fail")

    def test_shadow_real_hash_not_tied_to_model(self):
        # The recorded shadow uses a model_hash
        # different from the sha256 of the certified model.json → real fail.
        model = self.RUN / "artifacts" / "t8b_retrain" / "model.json"
        shadow = self.RUN / "shadow"
        if not model.is_file() or not shadow.is_dir():
            self.skipTest("real model/shadow not present")
        model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
        r = vre.check_g8(str(shadow), model_sha)
        self.assertEqual(r["state"], "fail")
        self.assertGreater(
            r["evidence"]["model_hash_mismatch"]
            + r["evidence"]["absent_model_hash"], 0)

    def test_real_T7_metrics_are_pending_not_pass(self):
        # Real metrics-T7.json: honest label with
        # external_fidelity_unidentified → pending (was pass in v1).
        fp = self.RUN / "artifacts" / "calibration" / "metrics-T7.json"
        if not fp.is_file():
            self.skipTest("real metrics-T7 not present")
        self.assertEqual(vre.check_g6(str(fp))["state"], "pending")

    def test_manifest_real_model_without_provenance_is_pending(self):
        # The real model manifest declares files[] (hash ok) but
        # WITHOUT dataset sources or label stats → pending.
        d = self.RUN / "artifacts" / "t8b_retrain"
        if not (d / "manifest.json").is_file():
            self.skipTest("real model manifest not present")
        r = vre.check_g7(str(d), str(d / "model.json"))
        self.assertEqual(r["state"], "pending")
        self.assertTrue(any("sources" in x for x in r["details"]),
                        r["details"])


if __name__ == "__main__":
    unittest.main()
