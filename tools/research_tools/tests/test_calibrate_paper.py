#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for `tools/research_tools/calibrate_paper.py` — controlled fixtures, no DBs.

Covers these detections:
- impossible fill prediction against the recorded book,
- badly covered interval (fill with no book evidence within the window),
- time bias in the fill (simulated trace systematically shifted),
- book overconsumption (size larger than the displayed depth),
- "unidentified" state when observations are missing.

Stdlib only (`unittest`). Run:
    python3 -m unittest discover -s tools/research_tools/tests -p "test_calibrate_paper.py" -v
"""

import json
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import calibrate_paper as cp  # noqa: E402


def book(bids=(), asks=()):
    """Book of ONE token: (price_um, size_um) lists, already sorted."""
    return {"bids": sorted(bids, reverse=True), "asks": sorted(asks)}


def fill(side, price_um, size_um, ts_ms=1_000_000):
    return {"side": side, "price_um": price_um, "size_um": size_um, "ts_ms": ts_ms}


# ── Impossible fill ──────────────────────────────────────────────────────────


class TestImpossibleFill(unittest.TestCase):
    def test_buy_above_highest_ask_is_impossible(self):
        # A BUY all-in at 0.80 when the highest displayed ask is 0.60:
        # not even walking the book can pay 0.80 — the sim fabricated the price.
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 100_000_000),
                                                      (600_000, 50_000_000)])
        v = cp.fill_verdict(fill("BUY", 800_000, 1_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_IMPOSSIBLE)

    def test_buy_below_bid_is_impossible(self):
        # A BUY at 0.20 with the displayed bid at 0.50: the selling flow would have
        # executed at 0.50 first; the fill contradicts the recorded tape.
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 100_000_000)])
        v = cp.fill_verdict(fill("BUY", 200_000, 1_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_IMPOSSIBLE)

    def test_sell_above_ask_is_impossible(self):
        # Mirror for SELL: selling above the displayed ask is impossible.
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 100_000_000)])
        v = cp.fill_verdict(fill("SELL", 700_000, 1_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_IMPOSSIBLE)

    def test_buy_without_liquidity_is_impossible(self):
        b = book(bids=[(500_000, 100_000_000)], asks=[])
        v = cp.fill_verdict(fill("BUY", 550_000, 1_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_IMPOSSIBLE)

    def test_fill_without_book_is_uncovered_not_impossible(self):
        # The absence of evidence is NOT a violation: it is neither certified nor
        # condemned — it stays out of the price metrics.
        v = cp.fill_verdict(fill("BUY", 550_000, 1_000_000), None, fee_um=0)
        self.assertEqual(v, cp.VERDICT_NO_COVERAGE)

    def test_all_in_price_deducts_fee_before_comparing(self):
        # BUY all-in fill 0.5675 = level 0.55 + parabolic fee 0.0175:
        # comparing the raw all-in would mark a healthy fill as "impossible".
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 100_000_000)])
        v = cp.fill_verdict(fill("BUY", 567_500, 1_000_000), b, fee_um=17_500)
        self.assertEqual(v, cp.VERDICT_OK)


# ── Interval coverage (over the production function, sqlite in
#    memory) ───────────────────────────────────────────────────────────────


def _conn_books(rows):
    """In-memory book_snapshots with the columns used by book_evidence.

    rows = (cid, ts_ms, yes_token, no_token, yes_bids_json, yes_asks_json,
            no_bids_json, no_asks_json, fee_rate, tick_size, clear_at)
    """
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE book_snapshots (market_condition_id TEXT, ts_ms INT,"
        " yes_token_id TEXT, no_token_id TEXT, yes_bids_json TEXT,"
        " yes_asks_json TEXT, no_bids_json TEXT, no_asks_json TEXT,"
        " parabolic_fee_rate TEXT, tick_size TEXT, clear_at INT)"
    )
    conn.executemany(
        "INSERT INTO book_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    return conn


_BOOK_JSON = '[{"price":"0.50","size":"100"}]'


class TestCoverage(unittest.TestCase):
    def test_returns_previous_and_next_in_window(self):
        conn = _conn_books([
            ("c1", 100_000, "t1", "t2", _BOOK_JSON, _BOOK_JSON,
             _BOOK_JSON, _BOOK_JSON, "0", "0.01", 0),
            ("c1", 200_000, "t1", "t2", _BOOK_JSON, _BOOK_JSON,
             _BOOK_JSON, _BOOK_JSON, "0", "0.01", 0),
        ])
        evs, problem = cp.book_evidence(conn, "c1", "t1", 150_000)
        self.assertIsNone(problem)
        self.assertEqual(sorted(d for _, _, d, _ in evs),
                         ["earlier", "later"])

    def test_outside_window_in_both_directions_without_evidence(self):
        conn = _conn_books([
            ("c1", 100_000, "t1", "t2", _BOOK_JSON, _BOOK_JSON,
             _BOOK_JSON, _BOOK_JSON, "0", "0.01", 0),
            ("c1", 900_000, "t1", "t2", _BOOK_JSON, _BOOK_JSON,
             _BOOK_JSON, _BOOK_JSON, "0", "0.01", 0),
        ])
        evs, problem = cp.book_evidence(conn, "c1", "t1", 500_000)
        self.assertEqual(evs, [])
        self.assertIsNone(problem)

    def test_absent_token_in_row_is_unknown_token(self):
        conn = _conn_books([
            ("c1", 100_000, "t1", "t2", _BOOK_JSON, _BOOK_JSON,
             _BOOK_JSON, _BOOK_JSON, "0", "0.01", 0),
        ])
        evs, problem = cp.book_evidence(conn, "c1", "other", 100_500)
        self.assertEqual(evs, [])
        self.assertEqual(problem, "unknown_token")

    def test_row_tick_size_reaches_book(self):
        conn = _conn_books([
            ("c1", 100_000, "t1", "t2", _BOOK_JSON, _BOOK_JSON,
             _BOOK_JSON, _BOOK_JSON, "0", "0.001", 0),
        ])
        evs, _ = cp.book_evidence(conn, "c1", "t1", 100_500)
        self.assertEqual(evs[0][0]["tick_um"], 1_000)


class TestEvidenceUnion(unittest.TestCase):
    """The tape is a ~1s sampling with dedup: a fill between two recorded states
    is discarded only if it contradicts BOTH. And it is a hard violation only
    when the bracket was stable (ant≈post)."""

    def test_impossible_under_previous_but_ok_under_next_is_not_discarded(self):
        fl = fill("BUY", 325_000, 1_000_000)
        ant = book(bids=[(120_000, 5_000_000)], asks=[(130_000, 40_000_000)])
        # post: the market rose — ask 0.30–0.34 covers the fill at 0.325.
        post = book(bids=[(290_000, 5_000_000)],
                    asks=[(300_000, 5_000_000), (340_000, 5_000_000)])
        v, stable, _ = cp.multi_verdict(
            fl, [(ant, 0, "earlier"), (post, 0, "later")]
        )
        self.assertEqual(v, cp.VERDICT_OK)
        self.assertFalse(stable)  # the book moved: transition evidence

    def test_impossible_under_both_evidences_is_discarded(self):
        fl = fill("BUY", 608_000, 1_000_000)
        ant = book(bids=[(870_000, 40_000_000)], asks=[(880_000, 5_000_000)])
        post = book(bids=[(700_000, 5_000_000)], asks=[(710_000, 5_000_000)])
        v, stable, _ = cp.multi_verdict(
            fl, [(ant, 0, "earlier"), (post, 0, "later")]
        )
        self.assertEqual(v, cp.VERDICT_IMPOSSIBLE)
        self.assertFalse(stable)

    def test_impossible_with_stable_bracket_is_hard_violation(self):
        fl = fill("BUY", 400_000, 1_000_000)
        ant = book(bids=[(500_000, 40_000_000)], asks=[(550_000, 5_000_000)])
        post = book(bids=[(500_000, 30_000_000)], asks=[(550_000, 8_000_000)])
        v, stable, _ = cp.multi_verdict(
            fl, [(ant, 0, "earlier"), (post, 0, "later")]
        )
        self.assertEqual(v, cp.VERDICT_IMPOSSIBLE)
        self.assertTrue(stable)

    def test_without_evidence_is_uncovered(self):
        v, stable, ambiguous = cp.multi_verdict(
            fill("BUY", 550_000, 1_000_000), []
        )
        self.assertEqual(v, cp.VERDICT_NO_COVERAGE)
        self.assertIsNone(stable)
        self.assertFalse(ambiguous)


# ── Unknown maker/taker role (conservative when condemning) ───────────


class TestAmbiguousRole(unittest.TestCase):
    def test_explainable_under_one_reading_is_NOT_impossible(self):
        # BUY all-in 0.57 with max ask 0.55: as maker (fee=0) the
        # negotiated price would be 0.57 → impossible; as taker the fee lowers it
        # to 0.56 → within band → ok. The role is UNKNOWN:
        # do not condemn if any admissible reading explains it.
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 100_000_000)])
        v, ambiguous = cp.conservative_verdict(
            fill("BUY", 570_000, 1_000_000), b, fee_um=10_000
        )
        self.assertEqual(v, cp.VERDICT_OK)
        self.assertTrue(ambiguous)

    def test_impossible_requires_both_readings(self):
        # All-in 0.80: neither maker (0.80) nor taker (0.79) explains it with max ask
        # 0.60 → impossible, with no ambiguity in the verdict (both
        # readings agree to condemn).
        b = book(bids=[(500_000, 100_000_000)],
                 asks=[(550_000, 100_000_000), (600_000, 50_000_000)])
        v, ambiguous = cp.conservative_verdict(
            fill("BUY", 800_000, 1_000_000), b, fee_um=10_000
        )
        self.assertEqual(v, cp.VERDICT_IMPOSSIBLE)
        self.assertFalse(ambiguous)

    def test_ambiguity_propagates_to_multi_verdict(self):
        fl = fill("BUY", 570_000, 1_000_000)
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 100_000_000)])
        v, _, ambiguous = cp.multi_verdict(fl, [(b, 10_000, "earlier")])
        self.assertEqual(v, cp.VERDICT_OK)
        self.assertTrue(ambiguous)


# ── Fee over the negotiated price and market tick ──────────────


class TestTradedFee(unittest.TestCase):
    def test_fee_iterates_over_traded_not_over_all_in(self):
        # rate=0.07: f(0.5675)≈0.01717 → p1≈0.5503 → fee=f(p1)≈0.01733.
        # The test fee is computed by hand the same way (one iteration).
        allin = 567_500
        f0 = cp.parabolic_fee_um(allin, "0.07")
        expected = cp.parabolic_fee_um(allin - f0, "0.07")
        self.assertEqual(
            cp.traded_fee_um(allin, "0.07", "BUY"), expected
        )
        # And it differs from the raw fee over the all-in (the bias this fixes).
        self.assertNotEqual(expected, f0)

    def test_sell_adds_fee_in_iteration(self):
        allin = 400_000
        f0 = cp.parabolic_fee_um(allin, "0.07")
        expected = cp.parabolic_fee_um(allin + f0, "0.07")
        self.assertEqual(
            cp.traded_fee_um(allin, "0.07", "SELL"), expected
        )


class TestMarketTick(unittest.TestCase):
    def test_tick_0001_tightens_tolerance(self):
        # max ask 0.55, fill at 0.555: with tick 0.01 (tol=10000µ) it is
        # taker zone → ok; with tick 0.001 (tol=1000µ) it is 5 ticks
        # above the ask → impossible.
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 100_000_000)])
        b["tick_um"] = 1_000
        v = cp.fill_verdict(fill("BUY", 555_000, 1_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_IMPOSSIBLE)
        del b["tick_um"]  # no tick in the snapshot → default TOL_PRICE_UM
        v = cp.fill_verdict(fill("BUY", 555_000, 1_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_OK)

    def test_stable_bracket_uses_book_tick(self):
        # Same bid, ask differs by 0.005: with tick 0.001 it is not stable
        # (5 ticks of difference); with tick 0.01 it would be.
        ant = book(bids=[(500_000, 10_000)], asks=[(550_000, 10_000)])
        post = book(bids=[(500_000, 10_000)], asks=[(555_000, 10_000)])
        for b in (ant, post):
            b["tick_um"] = 1_000
        self.assertFalse(cp.stable_bracket(ant, post))
        for b in (ant, post):
            b["tick_um"] = 10_000
        self.assertTrue(cp.stable_bracket(ant, post))


# ── Book overconsumption ────────────────────────────────────────────────────


class TestOverconsumption(unittest.TestCase):
    def test_size_greater_than_depth_at_price_is_overconsumption(self):
        # BUY taker at 0.55 with only 2 shares shown at ≤0.55:
        # filling 10 consumes liquidity that the tape does not show.
        b = book(bids=[(500_000, 100_000_000)],
                 asks=[(550_000, 2_000_000), (560_000, 3_000_000)])
        v = cp.fill_verdict(fill("BUY", 550_000, 10_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_OVERCONSUMPTION)

    def test_size_within_depth_is_ok(self):
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 20_000_000)])
        v = cp.fill_verdict(fill("BUY", 550_000, 10_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_OK)

    def test_fill_within_spread_is_unmeasurable(self):
        # Price between bid and ask: resting maker that can only fill by
        # crossing — without a trades tape the cross is not observable.
        b = book(bids=[(500_000, 100_000_000)], asks=[(550_000, 100_000_000)])
        v = cp.fill_verdict(fill("BUY", 520_000, 1_000_000), b, fee_um=0)
        self.assertEqual(v, cp.VERDICT_NOT_MEASURABLE)


# ── Bias in fill time ───────────────────────────────────────────────────────


class TestTimeBias(unittest.TestCase):
    def test_positive_systematic_bias_is_detected(self):
        # Sim fills ~3s AFTER the real trace, systematically.
        deltas = [3_000 + i * 10 for i in range(40)]
        r = cp.time_bias_ms(deltas)
        self.assertEqual(r["state"], "out_of_tolerance")
        self.assertEqual(r["median_ms"], 3_195)  # median of the series

    def test_centered_deltas_pass(self):
        deltas = [(-200 if i % 2 else 300) for i in range(60)]
        r = cp.time_bias_ms(deltas)
        self.assertEqual(r["state"], "ok")
        self.assertLessEqual(abs(r["median_ms"]), cp.TOL_TIME_BIAS_MS)

    def test_few_pairs_unidentified(self):
        r = cp.time_bias_ms([100, -100, 50])
        self.assertEqual(r["state"], "unidentified")


# ── Comparison of counts and state by observations ───────────────────────


class TestAggregatedComparison(unittest.TestCase):
    def test_count_within_band(self):
        r = cp.count_in_band(100, 150)
        self.assertEqual(r["state"], "ok")
        self.assertAlmostEqual(r["ratio"], 1.5)

    def test_count_outside_band(self):
        r = cp.count_in_band(100, 400)
        self.assertEqual(r["state"], "out_of_tolerance")

    def test_count_without_reference_is_unidentified(self):
        r = cp.count_in_band(0, 10)
        self.assertEqual(r["state"], "unidentified")

    def test_metric_with_few_observations_is_not_certified(self):
        e = cp.evaluate_metric(5, value=0.9, within=lambda v: v < 1.0)
        self.assertEqual(e["state"], "unidentified")
        e = cp.evaluate_metric(50, value=0.9, within=lambda v: v < 1.0)
        self.assertEqual(e["state"], "ok")


# ── Statistics and markout ───────────────────────────────────────────────────


class TestStatistics(unittest.TestCase):
    def test_quantile_and_median(self):
        self.assertEqual(cp.median([1, 2, 3]), 2)
        self.assertEqual(cp.median([1, 2, 3, 4]), 2.5)
        self.assertEqual(cp.quantile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5), 5.5)

    def test_adverse_markout_with_sign(self):
        # BUY at 0.50 with mid at 30s at 0.55 → markout +0.05 (favorable).
        self.assertEqual(cp.markout_signed_um("BUY", 500_000, 550_000), 50_000)
        # SELL at 0.50 with mid 0.55 → the mid rose after selling: adverse.
        self.assertEqual(cp.markout_signed_um("SELL", 500_000, 550_000), -50_000)

    def test_fit_share_with_few_obs_is_unidentified(self):
        r = cp.fit_share([0.1, 0.2])
        self.assertEqual(r["state"], "unidentified")
        r = cp.fit_share([0.1 * (i % 10) for i in range(100)])
        self.assertEqual(r["state"], "ok")
        self.assertGreater(r["value"], 0)
        self.assertLessEqual(r["value"], 1.0)


# ── Replay profiles (--research-profile) ─────────────────────────────────────


class TestReplayProfileState(unittest.TestCase):
    def test_all_in_band_and_reconciled_is_ok(self):
        strategies = {
            "arb": {"reconciled": True,
                    "comparison": {"count": {"state": "ok"}}},
            "near_expiry": {"reconciled": True,
                            "comparison": {"count": {"state": "ok"}}},
        }
        self.assertEqual(cp.replay_profile_state(strategies), "ok")

    def test_any_strategy_outside_band_is_outside(self):
        strategies = {
            "arb": {"reconciled": True,
                    "comparison": {"count": {"state": "out_of_tolerance"}}},
            "near_expiry": {"reconciled": True,
                            "comparison": {"count": {"state": "unidentified"}}},
        }
        self.assertEqual(
            cp.replay_profile_state(strategies), "out_of_tolerance"
        )

    def test_unidentified_count_does_not_certify(self):
        strategies = {
            "arb": {"reconciled": True,
                    "comparison": {"count": {"state": "unidentified"}}},
        }
        self.assertEqual(cp.replay_profile_state(strategies), "unidentified")

    def test_without_comparisons_is_unidentified(self):
        self.assertEqual(cp.replay_profile_state({}), "unidentified")
        strategies = {"arb": {"state": "unidentified", "reason_text": "failed"}}
        self.assertEqual(cp.replay_profile_state(strategies), "unidentified")


class TestShadowConfig(unittest.TestCase):
    """`--research-profile config` reads `[research_sim]` from --config-dir: the
    shadow copies config/ and installs the calibrated profile without touching the origin."""

    def test_copies_tree_and_adds_research_sim(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "config"
            (src / "strategies").mkdir(parents=True)
            (src / "general.toml").write_text(
                "[risk]\npaper_latency_ms = 25\n", encoding="utf-8"
            )
            (src / "strategies" / "arb.toml").write_text(
                "enabled = true\n", encoding="utf-8"
            )
            dest = Path(td) / "shadow"
            cp.generate_shadow_config(src, dest, share=1.0)
            parsed = cp.ri.load_toml(dest / "general.toml")
            rs = parsed["research_sim"]
            self.assertEqual(rs["profile"], "calibrated-T7")
            self.assertEqual(rs["share"], 1.0)
            self.assertEqual(rs["cancel_attr"], 0)
            self.assertTrue(rs["track_inventory"])
            self.assertEqual(rs["latencies"]["submit_ms"], 250.0)
            # The rest of the config is preserved and the tree is copied whole.
            self.assertEqual(parsed["risk"]["paper_latency_ms"], 25)
            self.assertTrue((dest / "strategies" / "arb.toml").is_file())
            # The origin is NOT modified.
            self.assertNotIn(
                "[research_sim]", (src / "general.toml").read_text(encoding="utf-8")
            )

    def test_regenerate_over_existing_destination(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "config"
            src.mkdir()
            (src / "general.toml").write_text("x = 1\n", encoding="utf-8")
            dest = Path(td) / "shadow"
            cp.generate_shadow_config(src, dest, share=0.5)
            cp.generate_shadow_config(src, dest, share=0.9)
            parsed = cp.ri.load_toml(dest / "general.toml")
            self.assertEqual(parsed["research_sim"]["share"], 0.9)


class TestExecuteReplayCmd(unittest.TestCase):
    def test_passes_research_profile_and_parses_digest(self):
        digest = "EXEC|e1|o1|tok1|Buy|1.5|0.9|1000\n"
        report = {
            "digest": digest,
            "reconciled": True,
            "research_profile": "conservative",
        }
        proc = mock.Mock(
            returncode=0, stdout=json.dumps(report), stderr=""
        )
        with mock.patch.object(cp.subprocess, "run", return_value=proc) as mrun:
            res = cp.execute_replay(
                "bin", "arb", "books.db", None, 0, 1,
                latency_ms=0, obs_delay_ms=0,
                config_dir="cfg", profile="conservative",
            )
        cmd = mrun.call_args[0][0]
        i = cmd.index("--research-profile")
        self.assertEqual(cmd[i + 1], "conservative")
        self.assertEqual(res["fills_sim"][0]["token_id"], "tok1")
        # price_um = cash/shares = 0.9/1.5 = 0.6 USDC
        self.assertEqual(res["fills_sim"][0]["price_um"], 600_000)
        self.assertEqual(res["report"]["research_profile"], "conservative")

    def test_subprocess_error_is_reported(self):
        proc = mock.Mock(returncode=2, stdout="", stderr="boom")
        with mock.patch.object(cp.subprocess, "run", return_value=proc):
            res = cp.execute_replay(
                "bin", "arb", "books.db", None, 0, 1,
                latency_ms=0, obs_delay_ms=0, profile="nominal",
            )
        self.assertIn("error", res)

    def test_timeout_does_not_kill_the_run(self):
        # A replay that exceeds the timeout returns an error instead of
        # propagating the exception and tearing down the whole run.
        with mock.patch.object(
            cp.subprocess, "run",
            side_effect=cp.subprocess.TimeoutExpired(cmd="x", timeout=1),
        ):
            res = cp.execute_replay(
                "bin", "arb", "books.db", None, 0, 1,
                latency_ms=0, obs_delay_ms=0, profile="nominal",
            )
        self.assertIn("timeout", res["error"])


# ── Comparability classifier ────────────────────────────────────────────────


def _cond_real(**kw):
    """Conditions of an ONLINE multi-strategy paper trace — what the DB
    can prove: capital/inventory/initial orders and seed are not
    recoverable afterwards (None = not declared)."""
    base = {
        "mode": "online_paper",
        "inputs": {"books_db": "analytics.sqlite", "spot_db": "spot.sqlite",
                   "window": [1000, 2000]},
        "strategies": ["arb", "near_expiry"],
        "config": "config@unknown",
        "version": None,
        "initial_capital": None,
        "initial_inventory": None,
        "initial_active_orders": None,
        "seed": None,
        "warmup": "continuous",
    }
    base.update(kw)
    return base


def _cond_sim(**kw):
    base = {
        "mode": "cold_replay",
        "inputs": {"books_db": "analytics.sqlite", "spot_db": "spot.sqlite",
                   "window": [1000, 2000]},
        "strategies": ["arb"],
        "config": "config@actual",
        "version": "wtree",
        "initial_capital": 5000.0,
        "initial_inventory": {},
        "initial_active_orders": [],
        "seed": 0xC0FFEE15,
        "warmup": "cold",
    }
    base.update(kw)
    return base


class TestComparability(unittest.TestCase):
    def test_hot_multi_strategy_vs_cold_single_strategy_not_comparable(self):
        # The real scenario: online multi-strategy paper trace
        # against a single-strategy cold replay → NOT comparable, never a
        # fidelity verdict (neither ok nor out_of_tolerance).
        c = cp.classify_comparability(_cond_real(), _cond_sim())
        self.assertEqual(c["state"], cp.STATE_NOT_COMPARABLE)
        self.assertNotIn(c["state"], (cp.STATE_OK, cp.STATE_OUTSIDE))
        reasons = " ".join(c["reasons"])
        self.assertIn("strategies", reasons)
        self.assertIn("warmup", reasons)
        self.assertIn("initial_capital", reasons)

    def test_identical_conditions_are_comparable(self):
        a = _cond_sim()
        b = _cond_sim(other_profile=True)  # extra field ignored
        c = cp.classify_comparability(a, b)
        self.assertEqual(c["state"], "comparable")
        self.assertEqual(c["reasons"], [])

    def test_undeclared_condition_is_not_comparable(self):
        # A single undeclared condition (seed of one side) already invalidates the
        # controlled comparison — it is not reconstructed artificially.
        b = _cond_sim()
        b["seed"] = None
        c = cp.classify_comparability(_cond_sim(), b)
        self.assertEqual(c["state"], cp.STATE_NOT_COMPARABLE)
        self.assertTrue(any("seed" in m for m in c["reasons"]))

    def test_different_inputs_not_comparable(self):
        b = _cond_sim()
        b["inputs"] = dict(b["inputs"], window=[1000, 3000])
        c = cp.classify_comparability(_cond_sim(), b)
        self.assertEqual(c["state"], cp.STATE_NOT_COMPARABLE)
        self.assertTrue(any("inputs" in m for m in c["reasons"]))

    def test_not_comparable_profile_state_is_never_fidelity_failure(self):
        # Comparison with different conditions and count out of band:
        # the certificate is "not_comparable", NOT "out_of_tolerance".
        strategies = {
            "arb": {
                "reconciled": True,
                "comparison": {
                    "count": {"state": "out_of_tolerance"},
                    "comparability": {"state": "no_comparable",
                                       "reasons": ["strategies differ"]},
                },
            },
        }
        self.assertEqual(
            cp.profile_certification_state(strategies), "no_comparable"
        )

    def test_profile_state_without_declared_comparability_is_not_comparable(self):
        # Without declared conditions there is no controlled comparison: honest default
        # = not_comparable (never a fidelity pass or fail).
        strategies = {
            "arb": {"reconciled": True,
                    "comparison": {"count": {"state": "out_of_tolerance"}}},
        }
        self.assertEqual(
            cp.profile_certification_state(strategies), "no_comparable"
        )

    def test_comparable_profile_state_delegates_to_metrics(self):
        # With comparable conditions the verdict can be a fidelity one.
        strategies = {
            "arb": {
                "reconciled": True,
                "comparison": {
                    "count": {"state": "ok"},
                    "comparability": {"state": "comparable", "reasons": []},
                },
            },
        }
        self.assertEqual(cp.profile_certification_state(strategies), "ok")


# ── Metrics by quantities (a split fill does not worsen the
#    certificate) ────────────────────────────────────────────────────────────


class TestExecutionAggregates(unittest.TestCase):
    def _real(self, order_id, price_um, size_um):
        return {"order_id": order_id, "side": "BUY", "price_um": price_um,
                "size_um": size_um, "ts_ms": 1}

    def test_splitting_a_fill_does_not_change_quantities(self):
        whole = [self._real("o1", 600_000, 10_000_000)]
        split_parts = [
            self._real("o1", 600_000, 4_000_000),
            self._real("o1", 600_000, 6_000_000),
        ]
        a = cp.execution_aggregates(whole)
        b = cp.execution_aggregates(split_parts)
        self.assertEqual(a["shares_um"], b["shares_um"])
        self.assertEqual(a["notional_um"], b["notional_um"])
        self.assertEqual(a["n_orders"], b["n_orders"])
        self.assertNotEqual(a["n_fills"], b["n_fills"])

    def test_real_notional_is_price_times_size(self):
        a = cp.execution_aggregates([self._real("o1", 500_000, 10_000_000)])
        self.assertEqual(a["shares_um"], 10_000_000)
        self.assertEqual(a["notional_um"], 5_000_000)
        self.assertEqual(a["mean_price_um"], 500_000)

    def test_sim_format_uses_cash_and_shares(self):
        # EXEC of the digest: cash_um is already the total notional of the fill.
        a = cp.execution_aggregates(
            [{"order_id": "o1", "side": "BUY", "shares_um": 2_000_000,
              "cash_um": 1_000_000, "ts_ms": 1}]
        )
        self.assertEqual(a["notional_um"], 1_000_000)
        self.assertEqual(a["mean_price_um"], 500_000)


class TestCompareTraceQuantities(unittest.TestCase):
    def _fills(self, prefix, n, price_um, size_um, order_prefix="o"):
        return [
            {"order_id": f"{order_prefix}{i}", "exec_id": f"{prefix}{i}",
             "side": "BUY", "price_um": price_um, "size_um": size_um,
             "token_id": "t1", "ts_ms": 1000 + i, "strategy": "s"}
            for i in range(n)
        ]

    def test_partitioning_into_partials_does_not_worsen_certificate(self):
        # Same execution split into partials: the rows count can
        # leave the band, but shares/notional (the certificate) hold.
        real = self._fills("r", 30, 500_000, 1_000_000)
        sims = []
        for f in real:
            half = f["size_um"] // 2
            s1 = dict(f, size_um=half, exec_id=f["exec_id"] + "a")
            s2 = dict(f, size_um=f["size_um"] - half, exec_id=f["exec_id"] + "b")
            sims += [s1, s2]
        c = cp.compare_traces(real, sims)
        self.assertEqual(c["aggregates"]["real"]["shares_um"],
                         c["aggregates"]["sim"]["shares_um"])
        self.assertEqual(c["shares"]["state"], "ok")
        self.assertEqual(c["notional"]["state"], "ok")
        self.assertEqual(c["count"]["ratio"], 2.0)  # diagnostic, not certificate

    def test_metamorphic_destroyed_notional_leaves_band(self):
        # If the conservative profile destroyed notional (impossible fills),
        # the quantity band detects it even if the count matches.
        real = self._fills("r", 30, 500_000, 1_000_000)
        sims = self._fills("s", 30, 500_000, 200_000)  # 5x fewer shares
        c = cp.compare_traces(real, sims)
        self.assertEqual(c["count"]["state"], "ok")  # count is misleading
        self.assertEqual(c["shares"]["state"], "out_of_tolerance")
        self.assertEqual(c["notional"]["state"], "out_of_tolerance")

    def test_comparability_attached_when_passed(self):
        real = self._fills("r", 30, 500_000, 1_000_000)
        sims = self._fills("s", 30, 500_000, 1_000_000)
        comp = {"state": "no_comparable", "reasons": ["x"]}
        c = cp.compare_traces(real, sims, comparability=comp)
        self.assertEqual(c["comparability"]["state"], "no_comparable")


class TestApproximateInventory(unittest.TestCase):
    def test_net_exposure_from_fills(self):
        fills = [
            {"side": "BUY", "size_um": 10_000_000, "token_id": "t1", "ts_ms": 1},
            {"side": "BUY", "size_um": 5_000_000, "token_id": "t2", "ts_ms": 2},
            {"side": "SELL", "size_um": 10_000_000, "token_id": "t1", "ts_ms": 3},
        ]
        inv = cp.approximate_inventory(fills)
        self.assertEqual(inv["max_exposure_um"], 15_000_000)  # after the 2nd fill
        self.assertEqual(inv["final_exposure_um"], 5_000_000)
        self.assertEqual(inv["n_tokens"], 2)


# ── Global label and latency stages ─────────────────────────────────────────


class TestRunEndToEnd(unittest.TestCase):
    """Smoke test of the `run` pipeline over synthetic DBs in tmp: exercises the
    new report sections (label, stages, comparability) without
    touching the real DBs or `--replay-bin`."""

    def _repo_fake(self, td):
        repo = Path(td) / "repo"
        (repo / "config").mkdir(parents=True)
        (repo / "config" / "general.toml").write_text(
            'mode = "paper"\ndb_path = "state.sqlite"\n'
            '[analytics]\ndb_path = "an.sqlite"\n'
            '[spot_stream]\ndb_path = "sp.sqlite"\n'
            "[risk]\npaper_latency_ms = 25\n",
            encoding="utf-8",
        )
        # state: minimal fills + markouts
        st = sqlite3.connect(repo / "state_paper.sqlite")
        st.execute("CREATE TABLE fills (id INTEGER PRIMARY KEY, ts_ms INT,"
                   " market_cid TEXT, token_id TEXT, side TEXT,"
                   " price_um INT, size_um INT, strategy TEXT,"
                   " order_id TEXT, exec_id TEXT)")
        st.execute("CREATE TABLE markouts (ts_ms INT, strategy TEXT,"
                   " side TEXT, fill_price_um INT, mid_5s_um INT,"
                   " mid_30s_um INT)")
        for i in range(60):
            st.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (i, 1_000_000 + i * 1000, "c1", "t1", "BUY",
                        550_000, 1_000_000, "arb", f"o{i}", f"e{i}"))
        st.commit(); st.close()
        an = sqlite3.connect(repo / "an_paper.sqlite")
        an.execute("CREATE TABLE book_snapshots (market_condition_id TEXT,"
                   " ts_ms INT, slug TEXT, yes_token_id TEXT,"
                   " no_token_id TEXT, yes_bids_json TEXT,"
                   " yes_asks_json TEXT, no_bids_json TEXT,"
                   " no_asks_json TEXT, parabolic_fee_rate TEXT,"
                   " tick_size TEXT, clear_at INT)")
        bk = '[{"price":"0.50","size":"100"}]'
        ak = '[{"price":"0.55","size":"100"}]'
        for i in range(200):
            an.execute("INSERT INTO book_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       ("c1", 999_000 + i * 1000, "s", "t1", "t2",
                        bk, ak, bk, ak, "0", "0.001", 0))
        an.commit(); an.close()
        sp = sqlite3.connect(repo / "sp_paper.sqlite")
        sp.execute("CREATE TABLE spot_samples (asset TEXT, source TEXT,"
                   " source_ts_ms INT, recv_ts_ms INT, avail_ts_ms INT)")
        for i in range(100):
            t = 999_000 + i * 1000
            sp.execute("INSERT INTO spot_samples VALUES (?,?,?,?,?)",
                       ("btc", "binance", t, t, t))
        sp.commit(); sp.close()
        return repo

    def test_run_produces_metrics_and_v2_report(self):
        import argparse
        with tempfile.TemporaryDirectory() as td:
            repo = self._repo_fake(td)
            out = Path(td) / "out"
            args = argparse.Namespace(
                repo_root=str(repo), from_ms=None, until_ms=None,
                split_ms=None, replay_bin=None, replay_hours=6.0,
                replay_strategies="arb", replay_profiles="nominal",
                out_dir=str(out),
            )
            rc = cp.run(args)
            self.assertEqual(rc, 0)
            m = json.loads((out / "metrics-T7.json").read_text())
            self.assertEqual(
                m["certification_label"],
                "verified_internal_consistency; "
                "external_fidelity_unidentified",
            )
            self.assertIn("inference", m["leg_latencies"])
            self.assertIn("signal_horizon_ms", m["a_priori_tolerances"])
            # The report renders and declares the label.
            inf = (out / "report-T7.md").read_text()
            self.assertIn("verified_internal_consistency", inf)


class TestCertificationLabel(unittest.TestCase):
    def test_without_real_executions_is_internal_consistency(self):
        e = cp.certification_label(real_exchange_fills=False)
        self.assertEqual(
            e, "verified_internal_consistency; external_fidelity_unidentified"
        )

    def test_latency_legs_distinguishes_inference(self):
        lat = {
            "transport_ms": {"p50": 0, "p95": 0, "p99": 0, "max": 5,
                              "n": 100, "state": "ok"},
            "internal_ms": {"p50": 0, "p95": 1, "max": 3, "n": 100,
                           "state": "ok"},
            "identified_transport": False,
        }
        p = cp.latency_legs(lat)
        self.assertIn("inference", p)
        self.assertEqual(p["feed_book"]["state"], "unidentified")
        self.assertEqual(p["spot_transport"]["state"], "unidentified")
        self.assertEqual(p["internal_queue"]["state"], "measured_descriptive")
        for stage in ("submit", "ack", "cancel", "notify", "fill"):
            self.assertEqual(p[stage]["state"], "unmeasurable")


if __name__ == "__main__":
    unittest.main()
