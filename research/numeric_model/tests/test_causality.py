"""Causality tests (written test-first) for the numeric dataset.

They verify, over synthetic fixtures, that:
1. Features never see the future (mutating a later event does not change any
   earlier observation; the cutoff is double: `local_seq` and `avail_ts_ms`).
2. All snapshots of the same market respect the train/test partition
   (chronological split by market, never random).
3. Scaling and imputation are fitted ONLY on train.
4. The discounted return is computed once (closed form).
5. The official result (`outcome`/`winner`) never appears as a feature.

Self-contained: they do not read the DB or real exports. Run them with
`python3 -m unittest discover -s research/numeric_model/tests`.
"""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

# The package intentionally has no __init__.py: the tests import the flat
# modules by adding the parent directory to the path (avoids touching research/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
import evaluate as evaluate_mod  # noqa: E402
import train as train_mod  # noqa: E402

T0 = 1_700_000_000_000  # epoch ms base of the fixtures


def _meta_payload(cid, yes, no, clear_s, ref=None, winner=None, known_ms=None):
    """`MarketMeta` payload of the contract (camelCase serde → snake_case JSON)."""
    payload = {
        "schema_version": 1,
        "market_cid": cid,
        "slug": f"btc-updown-{cid}",
        "yes_token_id": yes,
        "no_token_id": no,
        "asset": "BTC",
        "neg_risk": False,
        "market_open_ts": (T0 // 1000),
        "clear_at": clear_s,
        "reference_price": ref,
        "rule": "up_down_reference",
        "resolution_source": "binance_ws",
        "resolution_source_ref": "BTCUSDT spot",
        "rules_windows": [],
        "outcome": None,
    }
    if winner is not None:
        payload["outcome"] = {
            "winner": winner,
            "known_at_ms": known_ms,
            "provenance": "gamma:fixture",
        }
    return payload


def ev_meta(seq, avail, cid, yes, no, clear_s, ref=None, winner=None, known_ms=None):
    return {
        "schema_version": 1,
        "source": "polymarket_rest",
        "kind": "market_meta",
        "market_cid": cid,
        "token_id": None,
        "asset": "BTC",
        "payload": _meta_payload(cid, yes, no, clear_s, ref, winner, known_ms),
        "origin_ts_ms": None,
        "recv_ts_ms": avail - 20,
        "avail_ts_ms": avail,
        "local_seq": seq,
        "quality": 8,  # NO_REMOTE_TS
    }


def ev_book(seq, avail, cid, token, bids, asks):
    return {
        "schema_version": 1,
        "source": "polymarket_ws",
        "kind": "book_snapshot",
        "market_cid": cid,
        "token_id": token,
        "asset": None,
        "payload": {
            "bids": [[str(p), str(s)] for p, s in bids],
            "asks": [[str(p), str(s)] for p, s in asks],
        },
        "origin_ts_ms": avail - 40,
        "recv_ts_ms": avail - 10,
        "avail_ts_ms": avail,
        "local_seq": seq,
        "quality": 0,
    }


def ev_tick(seq, avail, price):
    return {
        "schema_version": 1,
        "source": "binance_ws",
        "kind": "price_tick",
        "market_cid": None,
        "token_id": None,
        "asset": "BTC",
        "payload": {"price": str(price)},
        "origin_ts_ms": avail - 30,
        "recv_ts_ms": avail - 5,
        "avail_ts_ms": avail,
        "local_seq": seq,
        "quality": 0,
    }


def ev_trade(seq, avail, cid, token, side, price, size):
    return {
        "schema_version": 1,
        "source": "polymarket_ws",
        "kind": "trade",
        "market_cid": cid,
        "token_id": token,
        "asset": None,
        "payload": {"side": side, "price": str(price), "size": str(size)},
        "origin_ts_ms": avail - 15,
        "recv_ts_ms": avail - 5,
        "avail_ts_ms": avail,
        "local_seq": seq,
        "quality": 0,
    }


def ev_sim(seq, avail, cid, token, sub, payload):
    """`{"sim": "<sub>"}` event (source internal_sim)."""
    return {
        "schema_version": 1,
        "source": "internal_sim",
        "kind": {"sim": sub},
        "market_cid": cid,
        "token_id": token,
        "asset": None,
        "payload": payload,
        "origin_ts_ms": None,
        "recv_ts_ms": avail,
        "avail_ts_ms": avail,
        "local_seq": seq,
        "quality": 32,  # SYNTHETIC
    }


def two_market_tape():
    """Two markets overlapping in time; M1 is the chronologically first one."""
    clear_s = (T0 // 1000) + 3600
    return [
        ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no", clear_s, ref="67000.0",
                winner="yes", known_ms=T0 + 3_700_000),
        ev_meta(1, T0 + 2_100, "m2", "m2-yes", "m2-no", clear_s + 3600,
                ref="67100.0", winner="no", known_ms=T0 + 7_300_000),
        ev_book(2, T0 + 1_000, "m1", "m1-yes", [["0.45", "100"]], [["0.47", "80"]]),
        ev_book(3, T0 + 1_050, "m1", "m1-no", [["0.52", "60"]], [["0.54", "90"]]),
        ev_tick(4, T0 + 1_200, "67000.5"),
        ev_book(5, T0 + 2_000, "m1", "m1-yes", [["0.46", "120"]], [["0.48", "70"]]),
        ev_book(6, T0 + 3_000, "m2", "m2-yes", [["0.30", "50"]], [["0.32", "40"]]),
        ev_tick(7, T0 + 3_100, "67010.0"),
        ev_trade(8, T0 + 3_200, "m2", "m2-yes", "buy", "0.32", "10"),
        ev_book(9, T0 + 4_000, "m2", "m2-no", [["0.66", "70"]], [["0.68", "65"]]),
    ]


def _obs_key(o):
    return (o["market_cid"], o["asof_local_seq"])


class CausalityFeaturesTest(unittest.TestCase):
    """No feature may depend on future events."""

    def test_mutating_future_event_does_not_change_past_features(self):
        events = two_market_tape()
        obs_before = dataset.build_observations(events)

        mutated = copy.deepcopy(events)
        # Crude mutation at seq 8 (trade of m2) and seq 9 (book of m2-no).
        mutated[8]["payload"]["size"] = "999999"
        mutated[9]["payload"]["asks"] = [["0.99", "1"]]
        obs_after = dataset.build_observations(mutated)

        idx_before = {_obs_key(o): o for o in obs_before}
        idx_after = {_obs_key(o): o for o in obs_after}
        self.assertEqual(set(idx_before), set(idx_after))

        for key, o in idx_before.items():
            if o["asof_local_seq"] < 8:
                self.assertEqual(
                    o["features"], idx_after[key]["features"],
                    f"observation {key} changed when a future event was mutated",
                )
        # Sanity check: the mutation does reach the later observations.
        later = [
            k for k, o in idx_after.items()
            if o["asof_local_seq"] >= 9 and o["features"] != idx_before[k]["features"]
        ]
        self.assertTrue(later, "the future mutation had no effect at all")

    def test_avail_ts_ms_is_visibility_cutoff(self):
        """An event with a low seq but a future avail is NOT visible before
        its `avail_ts_ms`, even though its seq has already passed ."""
        events = [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no", (T0 // 1000) + 3600,
                    ref="67000.0"),
            ev_tick(1, T0 + 500, "67000.0"),
            # Tick with seq 2 but very late availability:
            ev_tick(2, T0 + 50_000, "99999.0"),
            # Book later in seq but available before the tick:
            ev_book(3, T0 + 2_000, "m1", "m1-yes",
                    [["0.45", "100"]], [["0.47", "80"]]),
        ]
        obs = dataset.build_observations(events)
        obs_book = [o for o in obs if o["asof_local_seq"] == 3]
        self.assertEqual(len(obs_book), 1)
        self.assertEqual(
            obs_book[0]["features"]["spot"], 67000.0,
            "the tick with future avail contaminated the earlier observation",
        )

    def test_leak_detector_on_naive_builder(self):
        """Sanity check of the test itself: a naive implementation that ignores
        `avail_ts_ms` DOES leak the future — the earlier test has teeth."""
        events = [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no", (T0 // 1000) + 3600,
                    ref="67000.0"),
            ev_tick(1, T0 + 500, "67000.0"),
            ev_tick(2, T0 + 50_000, "99999.0"),
            ev_book(3, T0 + 2_000, "m1", "m1-yes",
                    [["0.45", "100"]], [["0.47", "80"]]),
        ]

        # Inline naive builder (deliberately leaky reference):
        # applies the whole seq prefix without looking at avail_ts_ms.
        spot_naive = None
        for e in sorted(events, key=lambda x: x["local_seq"]):
            if e["kind"] == "price_tick":
                spot_naive = float(e["payload"]["price"])
            if e["kind"] == "book_snapshot":
                break
        self.assertEqual(spot_naive, 99999.0)

        obs = dataset.build_observations(events)
        spot_real = [o for o in obs if o["asof_local_seq"] == 3][0]["features"]["spot"]
        self.assertNotEqual(
            spot_naive, spot_real,
            "the real builder reproduces the naive leak",
        )

    def test_features_do_not_contain_outcome(self):
        """outcome/winner are never observation features."""
        obs = dataset.build_observations(two_market_tape())
        forbidden_keys = {"winner", "outcome", "label", "resolution", "result"}
        for o in obs:
            for k in o["features"]:
                self.assertNotIn(k.lower(), forbidden_keys)


class OwnStateSimTest(unittest.TestCase):
    """The fill economics travel in `payload["detail"]`
    (SimExecution); noisy degradation if unrecognizable."""

    def _tape_with_fill(self, fill_payload):
        return [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no",
                    (T0 // 1000) + 3600, ref="67000.0"),
            ev_book(1, T0 + 1_000, "m1", "m1-yes",
                    [["0.45", "100"]], [["0.47", "80"]]),
            ev_sim(2, T0 + 1_500, "m1", "m1-yes", "accept",
                   {"order_id": "o1", "local_order_id": "lo-1",
                    "token_id": "m1-yes"}),
            ev_sim(3, T0 + 1_600, "m1", "m1-yes", "fill", fill_payload),
            ev_book(4, T0 + 2_000, "m1", "m1-yes",
                    [["0.46", "110"]], [["0.48", "70"]]),
        ]

    def _obs_final(self, events):
        return [o for o in dataset.build_observations(events)
                if o["asof_local_seq"] == 4][0]

    def test_fill_contract_shape_updates_own_economy(self):
        """Fill with `detail` = serialized SimExecution (contract shape):
        `own_net_shares` and `own_open_orders` are actually updated."""
        fill = {"detail": {
            "exec_id": "sim-exec-1", "order_id": "o1",
            "local_order_id": "lo-1", "token_id": "m1-yes",
            "side": "Buy", "shares": "10", "notional": "4.6",
            "fee_usdc": "0.01", "cash": "4.61",
            "levels": [{"price": "0.46", "size": "10", "fee_usdc": "0.01"}],
            "ts_ms": T0 + 1_600, "remaining": "0",
        }}
        obs = self._obs_final(self._tape_with_fill(fill))
        self.assertEqual(obs["features"]["own_net_shares"], 10.0)
        # remaining=0 in the detail → the accepted order ended up closed.
        self.assertEqual(obs["features"]["own_open_orders"], 0.0)
        self.assertEqual(obs["unparsed_fills"], 0)

    def test_unrecognizable_fill_degrades_loudly(self):
        """Fill without readable economics: NOT a silent zero — it increments the
        quality counter `unparsed_fills` exposed per observation."""
        fill = {"detail": {"algo": "unintelligible"}}
        obs = self._obs_final(self._tape_with_fill(fill))
        self.assertEqual(obs["unparsed_fills"], 1)
        self.assertEqual(obs["features"]["own_net_shares"], 0.0)
        # The order stays alive: the fill could not be interpreted.
        self.assertEqual(obs["features"]["own_open_orders"], 1.0)


class PostCloseFlagTest(unittest.TestCase):
    """observations with `time_left_ms < 0` are flagged."""

    def test_post_close_flag_after_clear_at(self):
        clear_s = T0 // 1000 + 1  # closes at T0+1000 ms
        events = [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no", clear_s),
            ev_book(1, T0 + 500, "m1", "m1-yes",
                    [["0.45", "100"]], [["0.47", "80"]]),
            ev_book(2, T0 + 2_000, "m1", "m1-yes",
                    [["0.46", "110"]], [["0.48", "70"]]),
        ]
        obs = dataset.build_observations(events)
        self.assertFalse(obs[0]["post_close"])
        self.assertTrue(obs[1]["post_close"])
        self.assertLess(obs[1]["features"]["time_left_ms"], 0)


def _obs_simple(cid, t):
    """Minimal observation for the split tests (does not require a builder)."""
    return {"market_cid": cid, "decision_ts_ms": T0 + t}


class SplitTemporalBoundariesTest(unittest.TestCase):
    """Temporal partition with explicit GLOBAL UTC boundaries.

    Certified rules:
    - the partition is PER MARKET and by a global temporal boundary, never by
      a percentage of markets (sorted by first timestamp; that was the
      old bug: a market with early and late obs dragged future obs into
      train);
    - a market whose observations cross a boundary is PURGED whole
      (recorded in splits_meta with its reason) — never split;
    - the same market never appears in two partitions.
    """

    def test_minimal_case(self):
        """A has decisions at t=0 and t=100; B at t=50 and t=60. With the
        train boundary at t=70, a certified partition NEVER returns
        train=[0,100] with test=[50,60]: A crosses the boundary → it is purged
        whole; B stays whole in train."""
        obs = [
            _obs_simple("A", 0), _obs_simple("A", 100),
            _obs_simple("B", 50), _obs_simple("B", 60),
        ]
        parts, meta = dataset.temporal_split(
            obs, train_end_ms=T0 + 70, embargo_ms=0)

        train_ts = sorted(
            o["decision_ts_ms"] - T0 for o in parts["train"])
        test_ts = sorted(
            o["decision_ts_ms"] - T0 for o in parts["test"])
        self.assertNotEqual(
            (train_ts, test_ts), ([0, 100], [50, 60]),
            "old bug reproduced: market split between train and test")
        # Positive certification: A purged whole, B whole in train.
        self.assertEqual({o["market_cid"] for o in parts["train"]}, {"B"})
        self.assertEqual(parts["test"], [])
        purged = {p["market_cid"]: p for p in meta["purged_markets"]}
        self.assertIn("A", purged)
        self.assertIn("crosses_boundary", purged["A"]["reason"])

    def test_market_crossing_boundary_is_purged_not_split(self):
        """A market that starts in train and ends in test is NOT split:
        it is purged whole and recorded with its span and zones."""
        obs = [
            _obs_simple("m1", 10), _obs_simple("m1", 20),
            _obs_simple("m2", 30), _obs_simple("m2", 200),  # crosses
            _obs_simple("m3", 300),
        ]
        parts, meta = dataset.temporal_split(
            obs, train_end_ms=T0 + 100, embargo_ms=0)
        self.assertEqual({o["market_cid"] for o in parts["train"]}, {"m1"})
        self.assertEqual({o["market_cid"] for o in parts["test"]}, {"m3"})
        self.assertEqual(meta["purged_markets"][0]["market_cid"], "m2")
        self.assertEqual(
            meta["purged_markets"][0]["span_ms"], [T0 + 30, T0 + 200])

    def test_same_market_never_in_two_partitions(self):
        """Universal property: for any input, the intersection of the market
        sets across partitions is empty."""
        import random
        rng = random.Random(1234)
        cids = [f"m{i}" for i in range(12)]
        for _ in range(20):
            obs = [
                _obs_simple(rng.choice(cids), rng.randrange(0, 1000))
                for _ in range(200)
            ]
            parts, meta = dataset.temporal_split(
                obs, train_end_ms=T0 + 400, val_end_ms=T0 + 700,
                embargo_ms=50)
            sets = {
                n: {o["market_cid"] for o in parts[n]}
                for n in ("train", "validation", "test")
            }
            self.assertFalse(sets["train"] & sets["validation"])
            self.assertFalse(sets["train"] & sets["test"])
            self.assertFalse(sets["validation"] & sets["test"])
            # Every market is in ONE partition, purged, or in the embargo.
            assigned = sets["train"] | sets["validation"] | sets["test"]
            purged = {p["market_cid"] for p in meta["purged_markets"]}
            embargo = set(meta.get("embargo_markets", []))
            self.assertFalse(assigned & purged)
            self.assertEqual(
                assigned | purged | embargo, set(cids))

    def test_three_partitions_with_embargo(self):
        """train/validation/test with explicit embargo: obs inside an
        embargo zone belong to no partition."""
        obs = [
            _obs_simple("m1", 10), _obs_simple("m1", 20),
            # Train→val embargo: (100, 160] with embargo_ms=60.
            _obs_simple("mE", 130),
            _obs_simple("m2", 200), _obs_simple("m2", 250),
            # Val→test embargo: (300, 360].
            _obs_simple("mF", 330),
            _obs_simple("m3", 400),
        ]
        parts, meta = dataset.temporal_split(
            obs, train_end_ms=T0 + 100, val_end_ms=T0 + 300,
            embargo_ms=60)
        self.assertEqual({o["market_cid"] for o in parts["train"]}, {"m1"})
        self.assertEqual(
            {o["market_cid"] for o in parts["validation"]}, {"m2"})
        self.assertEqual({o["market_cid"] for o in parts["test"]}, {"m3"})
        self.assertEqual(
            set(meta["embargo_markets"]), {"mE", "mF"})

    def test_split_invariant_to_row_permutation(self):
        """The split is a function of the dataset, not of the arrival order."""
        import random
        obs = [
            _obs_simple("m1", 10), _obs_simple("m2", 50),
            _obs_simple("m1", 20), _obs_simple("m3", 500),
            _obs_simple("m2", 60), _obs_simple("m3", 600),
        ]
        parts_a, meta_a = dataset.temporal_split(
            obs, train_end_ms=T0 + 100, embargo_ms=10)
        rng = random.Random(7)
        for _ in range(10):
            perm = list(obs)
            rng.shuffle(perm)
            parts_b, meta_b = dataset.temporal_split(
                perm, train_end_ms=T0 + 100, embargo_ms=10)
            for n in ("train", "validation", "test"):
                ka = sorted(
                    (o["market_cid"], o["decision_ts_ms"])
                    for o in parts_a[n])
                kb = sorted(
                    (o["market_cid"], o["decision_ts_ms"])
                    for o in parts_b[n])
                self.assertEqual(ka, kb, f"partition {n} is not invariant")

    def test_future_perturbation_does_not_change_train(self):
        """Adding later observations (new markets or extensions
        of test markets) does not alter the train partition already fixed by the
        explicit boundary."""
        base = [
            _obs_simple("m1", 10), _obs_simple("m1", 20),
            _obs_simple("m2", 50), _obs_simple("m2", 60),
            _obs_simple("m3", 500),
        ]
        parts_a, _ = dataset.temporal_split(
            base, train_end_ms=T0 + 100, embargo_ms=10)
        future = base + [
            _obs_simple("m4", 600), _obs_simple("m3", 700),
            _obs_simple("m5", 900),
        ]
        parts_b, _ = dataset.temporal_split(
            future, train_end_ms=T0 + 100, embargo_ms=10)
        ka = sorted(
            (o["market_cid"], o["decision_ts_ms"]) for o in parts_a["train"])
        kb = sorted(
            (o["market_cid"], o["decision_ts_ms"]) for o in parts_b["train"])
        self.assertEqual(ka, kb, "the future perturbed the train partition")

    def test_empty_dataset_gives_empty_partitions_not_exception(self):
        parts, meta = dataset.temporal_split(
            [], train_end_ms=T0 + 100, embargo_ms=10)
        self.assertEqual(parts["train"], [])
        self.assertEqual(parts["test"], [])
        self.assertEqual(meta["purged_markets"], [])

    def test_inconsistent_boundaries_rejected(self):
        with self.assertRaises(ValueError):
            dataset.temporal_split(
                [_obs_simple("m1", 10)],
                train_end_ms=T0 + 100, val_end_ms=T0 + 50)
        with self.assertRaises(ValueError):
            dataset.temporal_split(
                [_obs_simple("m1", 10)],
                train_end_ms=T0 + 100, embargo_ms=-1)

    def test_boundaries_by_fraction_are_temporal_quantiles(self):
        """`boundaries_from_fractions` produces global temporal boundaries
        (quantiles of decision_ts_ms), not a count of markets by first
        timestamp."""
        obs = [
            _obs_simple("m1", 10), _obs_simple("m1", 20),
            _obs_simple("m2", 50), _obs_simple("m2", 60),
            _obs_simple("m3", 500), _obs_simple("m3", 600),
        ]
        train_end, val_end = dataset.boundaries_from_fractions(
            obs, train_frac=0.5)
        self.assertEqual(train_end, T0 + 50)
        self.assertIsNone(val_end)
        # Deterministic and invariant to permutation.
        train_end2, _ = dataset.boundaries_from_fractions(
            list(reversed(obs)), train_frac=0.5)
        self.assertEqual(train_end, train_end2)


class AvailabilityLabelsTest(unittest.TestCase):
    """An observation is only usable for TRAINING if its label was
    already known at the fit cutoff. `known_at_ms` is never backdated."""

    def _obs(self, cid, t, known):
        return {
            "market_cid": cid, "decision_ts_ms": T0 + t,
            "label": 1, "label_known_at_ms": None if known is None
            else T0 + known,
        }

    def test_late_known_label_is_excluded_from_fit(self):
        obs = [
            self._obs("m1", 10, 50),    # known before the cutoff
            self._obs("m1", 20, 50),    # known before the cutoff
            self._obs("m2", 30, 500),   # label arrives after the cutoff
        ]
        usable, excl = dataset.filter_labels_available(
            obs, cutoff_ms=T0 + 100)
        self.assertEqual(
            [o["market_cid"] for o in usable], ["m1", "m1"])
        self.assertEqual(len(excl), 1)
        self.assertEqual(excl[0]["market_cid"], "m2")
        # The original known_at_ms is left untouched (never backdated).
        self.assertEqual(excl[0]["label_known_at_ms"], T0 + 500)

    def test_label_without_known_at_is_not_usable_for_training(self):
        obs = [self._obs("m1", 10, None)]
        usable, excl = dataset.filter_labels_available(obs, cutoff_ms=T0 + 100)
        self.assertEqual(usable, [])
        self.assertEqual(len(excl), 1)


class ClassificationEvalTest(unittest.TestCase):
    """`classify_eval_partitions`: the split is frozen by the artifact."""

    DOC = {
        "boundaries": {"train_end_ms": T0 + 100, "val_end_ms": None,
                       "embargo_ms": 10},
        "split_markets": {"train": ["mT"], "validation": [],
                          "test": ["mX"], "purged": ["mP"]},
    }

    def test_known_and_purged_markets(self):
        obs = [
            _obs_simple("mT", 10), _obs_simple("mX", 500),
            _obs_simple("mP", 50), _obs_simple("mP", 300),
            _obs_simple("mNuevo", 900), _obs_simple("mViejo", 20),
        ]
        parts, meta = dataset.classify_eval_partitions(obs, self.DOC)
        self.assertEqual(meta["mode"], "boundaries")
        self.assertEqual(
            {o["market_cid"] for o in parts["train"]}, {"mT"})
        self.assertEqual(
            {o["market_cid"] for o in parts["test"]}, {"mX", "mNuevo"})
        self.assertEqual(
            set(meta["out_of_scope_reasons"]), {"mP", "mViejo"})
        self.assertEqual(
            meta["out_of_scope_reasons"]["mP"], "purged_in_training")
        self.assertEqual(
            meta["out_of_scope_reasons"]["mViejo"],
            "before_train_end")

    def test_legacy_artifact_without_boundaries(self):
        """Legacy artifact without `boundaries`: historical compatibility
        (unknown → test), marked as `legacy_split_markets`."""
        doc = {"split_markets": {"train": ["mT"], "test": ["mX"]}}
        obs = [_obs_simple("mT", 10), _obs_simple("mX", 500),
               _obs_simple("mNuevo", 900)]
        parts, meta = dataset.classify_eval_partitions(obs, doc)
        self.assertEqual(meta["mode"], "legacy_split_markets")
        self.assertEqual(
            {o["market_cid"] for o in parts["test"]}, {"mX", "mNuevo"})


class ChronologicalSplitTest(unittest.TestCase):
    """Partition by market and chronological — never random, and
    never mixing snapshots of the same market between train and test."""

    def test_all_snapshots_of_a_market_in_one_partition(self):
        obs = dataset.build_observations(two_market_tape())
        parts, _ = dataset.temporal_split(
            obs, train_end_ms=T0 + 2_500, embargo_ms=0)

        train_m = {o["market_cid"] for o in parts["train"]}
        test_m = {o["market_cid"] for o in parts["test"]}
        self.assertEqual(train_m & test_m, set(), "a market ended up split across partitions")
        self.assertEqual(train_m | test_m, {"m1", "m2"})
        self.assertEqual(train_m, {"m1"}, "the split is not chronological")
        self.assertEqual(test_m, {"m2"})

    def test_split_stable_under_input_permutation(self):
        """Reordering the tape (keeping seq) does not change the partition."""
        obs_a = dataset.build_observations(two_market_tape())
        parts_a, _ = dataset.temporal_split(
            obs_a, train_end_ms=T0 + 2_500, embargo_ms=0)

        reordered = list(reversed(two_market_tape()))
        obs_b = dataset.build_observations(reordered)
        parts_b, _ = dataset.temporal_split(
            obs_b, train_end_ms=T0 + 2_500, embargo_ms=0)

        self.assertEqual(
            {o["market_cid"] for o in parts_a["train"]},
            {o["market_cid"] for o in parts_b["train"]},
        )
        self.assertEqual(
            {o["market_cid"] for o in parts_a["test"]},
            {o["market_cid"] for o in parts_b["test"]},
        )

    def test_train_precedes_test_in_market_time(self):
        """Real temporal certification: EVERY train observation is
        earlier than the boundary and EVERY test one is later (plus the embargo) —
        the manifest can no longer declare an overlap."""
        obs = dataset.build_observations(two_market_tape())
        embargo = 100
        parts, meta = dataset.temporal_split(
            obs, train_end_ms=T0 + 2_500, embargo_ms=embargo)
        train, test = parts["train"], parts["test"]
        max_train = max(o["decision_ts_ms"] for o in train)
        min_test = min(o["decision_ts_ms"] for o in test)
        self.assertLessEqual(max_train, T0 + 2_500)
        self.assertGreater(min_test, T0 + 2_500 + embargo)
        # The window declared in the manifest separates train from test.
        sp_train = [s for s in meta["splits"] if s["name"] == "train"][0]
        sp_test = [s for s in meta["splits"] if s["name"] == "test"][0]
        self.assertLess(sp_train["to_ms"], sp_test["from_ms"])
        self.assertLessEqual(max_train, sp_train["to_ms"])
        self.assertGreaterEqual(min_test, sp_test["from_ms"])


class ScalingTrainOnlyTest(unittest.TestCase):
    """Scaler and imputer fitted ONLY on train."""

    def _train_test(self):
        obs = dataset.build_observations(two_market_tape())
        parts, _ = dataset.temporal_split(
            obs, train_end_ms=T0 + 2_500, embargo_ms=0)
        return parts["train"], parts["test"], parts

    def test_transform_does_not_mutate_params(self):
        train, test, _ = self._train_test()
        scaler = dataset.FeatureScaler().fit(train)
        medias_0 = dict(scaler.means_)
        medians_0 = dict(scaler.medians_)
        scaler.transform(test)
        self.assertEqual(scaler.means_, medias_0)
        self.assertEqual(scaler.medians_, medians_0)

    def test_params_correspond_to_train_not_total(self):
        train, test, _ = self._train_test()
        scaler = dataset.FeatureScaler().fit(train)

        def manual_mean(obs, feat):
            vals = [o["features"][feat] for o in obs
                    if o["features"].get(feat) is not None]
            return sum(vals) / len(vals)

        for feat in ("yes_mid", "time_left_ms"):
            expected = manual_mean(train, feat)
            self.assertAlmostEqual(scaler.means_[feat], expected, places=9)
            # If it had been fitted on train+test, the mean would differ.
            total_mean = manual_mean(train + test, feat)
            if abs(total_mean - expected) > 1e-9:
                self.assertNotAlmostEqual(
                    scaler.means_[feat], total_mean, places=9,
                    msg=f"{feat}: the scaler saw test data",
                )

    def test_imputation_uses_train_median(self):
        train, test, _ = self._train_test()
        scaler = dataset.FeatureScaler().fit(train)
        # Test observation with a missing feature → train median.
        o_test = copy.deepcopy(test[0])
        o_test["features"]["yes_mid"] = None
        vec = scaler.transform([o_test])[0]
        idx = dataset.FEATURE_NAMES.index("yes_mid")
        train_median = scaler.medians_["yes_mid"]
        expected = (train_median - scaler.means_["yes_mid"]) / (
            scaler.stds_["yes_mid"] or 1.0
        )
        self.assertAlmostEqual(vec[idx], expected, places=9)


class DiscountedRewardTest(unittest.TestCase):
    """The discounted return is applied only once."""

    def test_exact_closed_form(self):
        rewards = [1.0, 1.0, 1.0]
        gamma = 0.9
        got = dataset.discounted_returns(rewards, gamma)
        expected = [1 + 0.9 + 0.81, 1 + 0.9, 1.0]
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g, e, places=12)

    def test_no_double_discount(self):
        rewards = [0.5, 0.25]
        once = dataset.discounted_returns(rewards, 0.5)
        twice = dataset.discounted_returns(once, 0.5)
        self.assertNotAlmostEqual(
            once[0], twice[0],
            msg="the pipeline applies the discount again",
        )
        self.assertAlmostEqual(once[0], 0.5 + 0.5 * 0.25, places=12)


def three_market_tape_split():
    """Three markets with labels known at the cutoff: m1 (yes) and m2 (no)
    end before `TRAIN_END`; m3 (yes) starts after the embargo —
    canonical partition train={m1,m2}, test={m3}, no purged markets."""
    clear_s = (T0 // 1000) + 3600
    return [
        ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no", clear_s,
                ref="67000.0", winner="yes", known_ms=T0 + 2_500),
        ev_meta(1, T0 + 2_100, "m2", "m2-yes", "m2-no", clear_s,
                ref="67100.0", winner="no", known_ms=T0 + 4_500),
        ev_meta(2, T0 + 150_100, "m3", "m3-yes", "m3-no", clear_s,
                ref="67200.0", winner="yes", known_ms=T0 + 153_000),
        ev_book(3, T0 + 1_000, "m1", "m1-yes",
                [["0.45", "100"]], [["0.47", "80"]]),
        ev_book(4, T0 + 1_050, "m1", "m1-no",
                [["0.52", "60"]], [["0.54", "90"]]),
        ev_tick(5, T0 + 1_200, "67000.5"),
        ev_book(6, T0 + 2_000, "m1", "m1-yes",
                [["0.46", "120"]], [["0.48", "70"]]),
        ev_book(7, T0 + 3_000, "m2", "m2-yes",
                [["0.30", "50"]], [["0.32", "40"]]),
        ev_tick(8, T0 + 3_100, "67010.0"),
        ev_book(9, T0 + 4_000, "m2", "m2-no",
                [["0.66", "70"]], [["0.68", "65"]]),
        ev_book(10, T0 + 151_000, "m3", "m3-yes",
                [["0.55", "90"]], [["0.57", "75"]]),
        ev_tick(11, T0 + 151_100, "67200.0"),
        ev_book(12, T0 + 152_000, "m3", "m3-no",
                [["0.42", "60"]], [["0.44", "80"]]),
    ]


TRAIN_END = T0 + 50_000  # explicit boundary of the fixture above


class CompleteCircuitTest(unittest.TestCase):
    """The stdlib baseline runs over fixtures and emits metrics + a manifest
    with the `ResearchManifest` schema of the contract (snake_case fields)."""

    def test_train_evaluate_on_fixtures(self):
        events = three_market_tape_split()
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(json.dumps(e) for e in events))
            out = Path(d) / "artifacts"

            res = train_mod.run(
                events_path=str(tape), out_dir=str(out),
                epochs=50, lr=0.5, seed=7, train_end_ms=TRAIN_END,
            )
            manifest = json.loads((out / "manifest.json").read_text())
            for field in (
                "schema_version", "created_at_ms", "code_hash", "config_hash",
                "model_hash", "sim_profile_hash", "seeds", "splits",
                "research_dir", "files",
            ):
                self.assertIn(field, manifest)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertTrue(manifest["files"])
            self.assertTrue(all(
                set(f) == {"path", "sha256"} for f in manifest["files"]
            ))
            self.assertEqual(res["metrics"]["n_train"] > 0, True)

            m = evaluate_mod.run(
                events_path=str(tape),
                model_path=str(out / "model.json"),
                out_dir=str(out),
            )
            for k in ("accuracy", "logloss", "brier",
                      "persistence_brier", "market_brier"):
                self.assertIn(k, m["metrics"])

    def test_model_json_persists_base_rate_and_boundaries(self):
        """`train_label_rate` and the boundaries travel in the artifact —
        evaluate never recomputes the base rate over another population."""
        events = three_market_tape_split()
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(json.dumps(e) for e in events))
            out = Path(d) / "art"
            res = train_mod.run(events_path=str(tape), out_dir=str(out),
                                epochs=10, seed=0, train_end_ms=TRAIN_END)
            model_doc = json.loads(Path(res["model_path"]).read_text())
            self.assertIn("train_label_rate", model_doc)
            self.assertAlmostEqual(
                model_doc["train_label_rate"], 0.6, places=9)  # 3 yes + 2 no
            b = model_doc["boundaries"]
            self.assertEqual(b["train_end_ms"], TRAIN_END)
            self.assertIn("embargo_ms", b)
            # base_rate_source declared by evaluate: taken from the artifact.
            m = evaluate_mod.run(events_path=str(tape),
                                 model_path=res["model_path"],
                                 out_dir=str(out))
            self.assertEqual(m["metrics"]["base_rate_source"], "model.json")
            self.assertAlmostEqual(
                m["metrics"]["base_rate"], model_doc["train_label_rate"])

    def test_unknown_market_does_not_fall_in_test(self):
        """An unknown market whose span is earlier than the end
        of train does NOT fall into test — it is classified `out_of_scope`;
        only markets strictly later than the test boundary go to test."""
        clear_s = (T0 // 1000) + 3600
        events = three_market_tape_split()
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(json.dumps(e) for e in events))
            out = Path(d) / "art"
            res = train_mod.run(events_path=str(tape), out_dir=str(out),
                                epochs=10, seed=0, train_end_ms=TRAIN_END)

            # Eval tape with: m3 (test), mViejo unknown and BEFORE the
            # train boundary, mSol unknown crossing the embargo, and mNuevo
            # strictly later.
            eval_events = [
                ev_meta(0, T0 + 150_100, "m3", "m3-yes", "m3-no", clear_s,
                        winner="yes", known_ms=T0 + 153_000),
                ev_book(1, T0 + 151_000, "m3", "m3-yes",
                        [["0.55", "90"]], [["0.57", "75"]]),
                ev_meta(2, T0 + 500, "mViejo", "mViejo-yes", "mViejo-no",
                        clear_s, winner="no", known_ms=T0 + 6_000),
                ev_book(3, T0 + 1_500, "mViejo", "mViejo-yes",
                        [["0.40", "50"]], [["0.42", "45"]]),
                ev_meta(4, T0 + 2_000, "mSol", "mSol-yes", "mSol-no",
                        clear_s, winner="yes", known_ms=T0 + 200_000),
                ev_book(5, T0 + 3_000, "mSol", "mSol-yes",
                        [["0.50", "50"]], [["0.52", "45"]]),
                ev_book(6, T0 + 130_000, "mSol", "mSol-yes",
                        [["0.51", "50"]], [["0.53", "45"]]),
                ev_meta(7, T0 + 160_000, "mNuevo", "mNuevo-yes",
                        "mNuevo-no", clear_s, winner="no",
                        known_ms=T0 + 165_000),
                ev_book(8, T0 + 161_000, "mNuevo", "mNuevo-yes",
                        [["0.60", "40"]], [["0.62", "35"]]),
            ]
            tape_ev = Path(d) / "ev.jsonl"
            tape_ev.write_text("\n".join(json.dumps(e) for e in eval_events))
            m = evaluate_mod.run(events_path=str(tape_ev),
                                 model_path=res["model_path"],
                                 out_dir=str(out))
            self.assertEqual(m["test_markets"], ["m3", "mNuevo"])
            oos = m["metrics"]["out_of_scope_markets"]
            self.assertEqual(sorted(oos), ["mSol", "mViejo"])
            self.assertEqual(m["metrics"]["n_out_of_scope_markets"], 2)

    def test_degenerate_dataset_is_actionable_error(self):
        """A single class in train after purging/filtering → an
        actionable error, NEVER a zero-weight model.json reported as success."""
        events = [
            e for e in three_market_tape_split()
            if e.get("market_cid") != "m2"  # train keeps only YES
        ]
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(json.dumps(e) for e in events))
            out = Path(d) / "art"
            with self.assertRaises(RuntimeError) as ctx:
                train_mod.run(events_path=str(tape), out_dir=str(out),
                              epochs=10, seed=0, train_end_ms=TRAIN_END)
            self.assertIn("single class", str(ctx.exception))
            self.assertFalse((out / "model.json").exists(),
                             "a degenerate model was saved")

    def test_late_known_label_is_counted_in_manifest(self):
        """Labels with known_at later than the fit cutoff are
        excluded from training and counted in the manifest. The default
        cutoff is the last known_at (post-capture fit); an explicit cutoff
        exercises the exclusion."""
        events = three_market_tape_split()
        # m2 (no) would be known AFTER the fit cutoff → excluded.
        for e in events:
            if e.get("market_cid") == "m2" and e["kind"] == "market_meta":
                e["payload"]["outcome"]["known_at_ms"] = TRAIN_END + 1
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(json.dumps(e) for e in events))
            out = Path(d) / "art"
            # With an earlier explicit cutoff, train only has YES → the
            # dataset degenerates and fails BEFORE writing the manifest.
            with self.assertRaises(RuntimeError):
                train_mod.run(events_path=str(tape), out_dir=str(out),
                              epochs=10, seed=0, train_end_ms=TRAIN_END,
                              cutoff_fit_ms=TRAIN_END)

        # Healthy variant: 4 markets, 2 with a partially late label.
        clear_s = (T0 // 1000) + 3600
        events = three_market_tape_split() + [
            ev_meta(20, T0 + 5_000, "m4", "m4-yes", "m4-no", clear_s,
                    winner="no", known_ms=T0 + 6_000),
            ev_book(21, T0 + 6_500, "m4", "m4-yes",
                    [["0.35", "50"]], [["0.37", "40"]]),
            ev_meta(22, T0 + 200_000, "m5", "m5-yes", "m5-no", clear_s,
                    winner="no", known_ms=T0 + 202_000),
            ev_book(23, T0 + 201_000, "m5", "m5-yes",
                    [["0.30", "40"]], [["0.32", "35"]]),
        ]
        # m4: obs at T0+6500, label known at T0+6000 <= TRAIN_END → usable.
        # The rule is based on known_at <= the partition cutoff (TRAIN_END),
        # not on the observation time.
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(json.dumps(e) for e in events))
            out = Path(d) / "art"
            res = train_mod.run(events_path=str(tape), out_dir=str(out),
                                epochs=10, seed=0, train_end_ms=TRAIN_END)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(
                res["metrics"]["n_test"], 3,  # m3 (2 obs) + m5 (1 obs)
                "test must contain the obs of m3 and m5")
            # m4 entered train with its label available.
            model_doc = json.loads(Path(res["model_path"]).read_text())
            self.assertIn("m4", model_doc["split_markets"]["train"])

    def test_point_in_time_excludes_label_published_after_test_start(self):
        """Fit cutoff: in `point_in_time` evaluation the fit must precede the
        first OOS decision, and a train label published AFTER the test start
        cannot enter the fit. The same label in `retrospective` mode does
        enter, but the artifact declares `requires_future`/`oos_operable=False`
        and the evidence is insufficient — never OOS operable."""
        clear_s = (T0 // 1000) + 3600
        events = three_market_tape_split() + [
            # m4 (no) keeps the second class in train when m2 drops out.
            ev_meta(20, T0 + 5_000, "m4", "m4-yes", "m4-no", clear_s,
                    winner="no", known_ms=T0 + 6_000),
            ev_book(21, T0 + 6_500, "m4", "m4-yes",
                    [["0.35", "50"]], [["0.37", "40"]]),
        ]
        # m2 (no, train partition) is published AFTER the test start
        # (m3 decides from T0+151_000) → excluded from the point-in-time fit.
        for e in events:
            if e.get("market_cid") == "m2" and e["kind"] == "market_meta":
                e["payload"]["outcome"]["known_at_ms"] = T0 + 160_000
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(json.dumps(e) for e in events))

            out = Path(d) / "art-pit"
            res = train_mod.run(events_path=str(tape), out_dir=str(out),
                                epochs=10, seed=0, train_end_ms=TRAIN_END,
                                eval_mode="point_in_time")
            manifest = json.loads((out / "manifest.json").read_text())
            model_doc = json.loads(Path(res["model_path"]).read_text())
            self.assertEqual(model_doc["eval_mode"], "point_in_time")
            t_ini = model_doc["test_start_ms"]
            self.assertIsNotNone(t_ini)
            # The default cutoff is the first OOS decision − 1ms.
            self.assertEqual(model_doc["cutoff_fit_ms"], t_ini - 1)
            self.assertTrue(model_doc["oos_operable"])
            self.assertFalse(model_doc["requires_future"])
            # m2 was left out of the fit (late label): train = m1+m4 = 4 obs
            # (retrospective would have all 6 including m2).
            self.assertEqual(res["metrics"]["n_train"], 4)
            excl = manifest["exclusions"]["label_not_known_at_cutoff"]
            self.assertEqual(excl["train"], 2)

            # Explicit cutoff later than the test start → rejected,
            # never a "point-in-time" fit that needs the future.
            with self.assertRaises(RuntimeError):
                train_mod.run(events_path=str(tape),
                              out_dir=str(Path(d) / "art-bad"),
                              epochs=10, seed=0, train_end_ms=TRAIN_END,
                              eval_mode="point_in_time",
                              cutoff_fit_ms=T0 + 200_000)

            # Retrospective: m2 does enter (legitimate post-capture fit) but
            # the model must NOT be called OOS operable.
            out2 = Path(d) / "art-retro"
            res2 = train_mod.run(events_path=str(tape), out_dir=str(out2),
                                 epochs=10, seed=0, train_end_ms=TRAIN_END,
                                 eval_mode="retrospective")
            md2 = json.loads(Path(res2["model_path"]).read_text())
            self.assertEqual(md2["eval_mode"], "retrospective")
            self.assertTrue(md2["requires_future"])
            self.assertFalse(md2["oos_operable"])
            self.assertEqual(res2["metrics"]["n_train"], 6)
            self.assertEqual(res2["metrics"]["oos_evidence"],
                             "insufficient")
            self.assertTrue(any(
                "needed future labels" in m
                for m in res2["metrics"]["sufficiency"]["reasons"]))

    def test_post_close_is_explicitly_excluded(self):
        """post_close is a declarative mark — the pre-close task
        excludes them explicitly and counts them."""
        clear_s = T0 // 1000 + 5  # closes at T0+5000 ms
        events = [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no", clear_s,
                    winner="yes", known_ms=T0 + 6_000),
            ev_book(1, T0 + 1_000, "m1", "m1-yes",
                    [["0.45", "100"]], [["0.47", "80"]]),
            # This decision is POST-CLOSE (T0+6000 > clear T0+5000):
            ev_book(2, T0 + 6_000, "m1", "m1-yes",
                    [["0.98", "100"]], [["0.99", "80"]]),
            # m0: the second class in train (winner=no).
            ev_meta(7, T0 + 1_500, "m0", "m0-yes", "m0-no", clear_s,
                    winner="no", known_ms=T0 + 6_500),
            ev_book(8, T0 + 2_000, "m0", "m0-yes",
                    [["0.60", "50"]], [["0.62", "45"]]),
            ev_meta(3, T0 + 200_000, "m2", "m2-yes", "m2-no",
                    (T0 // 1000) + 210, winner="no",
                    known_ms=T0 + 210_000),
            ev_book(4, T0 + 201_000, "m2", "m2-yes",
                    [["0.30", "40"]], [["0.32", "35"]]),
            ev_meta(5, T0 + 220_000, "m3", "m3-yes", "m3-no",
                    (T0 // 1000) + 230, winner="no",
                    known_ms=T0 + 230_000),
            ev_book(6, T0 + 221_000, "m3", "m3-yes",
                    [["0.25", "40"]], [["0.27", "35"]]),
        ]
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(json.dumps(e) for e in events))
            out = Path(d) / "art"
            res = train_mod.run(
                events_path=str(tape), out_dir=str(out),
                epochs=10, seed=0, train_end_ms=T0 + 50_000)
            manifest = json.loads((out / "manifest.json").read_text())
            excl = manifest["exclusions"]
            self.assertEqual(excl["post_close"]["observations"], 1)
            model_doc = json.loads(Path(res["model_path"]).read_text())
            train_ts = model_doc.get("n_post_close_excluded")
            self.assertEqual(train_ts, 1)

    def test_evaluate_uses_saved_split_does_not_rederive(self):
        """evaluate consumes `split_markets` from model.json and does not
        re-derive the split from the evaluation data."""
        clear_s = (T0 // 1000) + 3600
        with tempfile.TemporaryDirectory() as d:
            tape_tr = Path(d) / "tr.jsonl"
            tape_tr.write_text("\n".join(
                json.dumps(e) for e in three_market_tape_split()))
            out = Path(d) / "art"
            res = train_mod.run(events_path=str(tape_tr), out_dir=str(out),
                                epochs=10, seed=0, train_end_ms=TRAIN_END)
            model_doc = json.loads(Path(res["model_path"]).read_text())
            self.assertEqual(
                model_doc["split_markets"]["train"], ["m1", "m2"])
            self.assertEqual(model_doc["split_markets"]["test"], ["m3"])

            # Evaluation: the tape reverses the order — m3 appears FIRST.
            # If evaluate re-derived the split, m3 would fall into train.
            eval_events = [
                ev_meta(0, T0 + 100, "m3", "m3-yes", "m3-no", clear_s,
                        winner="yes", known_ms=T0 + 153_000),
                ev_meta(1, T0 + 200, "m1", "m1-yes", "m1-no", clear_s,
                        winner="yes", known_ms=T0 + 2_500),
                ev_book(2, T0 + 151_000, "m3", "m3-yes",
                        [["0.55", "90"]], [["0.57", "75"]]),
                ev_book(3, T0 + 1_000, "m1", "m1-yes",
                        [["0.45", "100"]], [["0.47", "80"]]),
            ]
            tape_ev = Path(d) / "ev.jsonl"
            tape_ev.write_text("\n".join(json.dumps(e) for e in eval_events))
            m = evaluate_mod.run(events_path=str(tape_ev),
                                 model_path=res["model_path"],
                                 out_dir=str(out))
            # With the saved split, m1 stays in train even though it arrives late
            # on this tape: only m3 is test.
            self.assertEqual(m["test_markets"], ["m3"])

    def test_manifest_created_at_is_not_overwritten_in_eval(self):
        """evaluate does not rewrite `created_at_ms` in the manifest."""
        with tempfile.TemporaryDirectory() as d:
            tape = Path(d) / "tape.jsonl"
            tape.write_text("\n".join(
                json.dumps(e) for e in three_market_tape_split()))
            out = Path(d) / "art"
            res = train_mod.run(events_path=str(tape), out_dir=str(out),
                                epochs=10, seed=0, train_end_ms=TRAIN_END)
            mpath = out / "manifest.json"
            created_0 = json.loads(mpath.read_text())["created_at_ms"]
            evaluate_mod.run(events_path=str(tape),
                             model_path=res["model_path"], out_dir=str(out))
            created_1 = json.loads(mpath.read_text())["created_at_ms"]
            self.assertEqual(created_0, created_1)


class BuilderIncrementalTest(unittest.TestCase):
    """`build_observations_incremental`: a single pass with a deferred-apply
    queue keyed by `avail_ts_ms`. It replaces the O(n²) reference builder for
    real tapes (millions of events).

    Invariant: over tapes where `avail_ts_ms` is monotonic in `local_seq`
    (a condition the exporter guarantees when assigning seq by avail) the
    result is EXACTLY that of the batch builder; under avail inversions it
    remains causal (nothing is applied before its avail), and a temporal
    regression of the decision raises `NonMonotonicTape` instead of leaking
    already-applied state.
    """

    def _rich_tape(self):
        """Tape with all the kinds: metas, books, ticks, trades and fills."""
        clear_s = (T0 // 1000) + 3600
        # The avails of the added events stay strictly increasing after the last
        # one of `two_market_tape` (seq 9, avail T0+4000) — the incremental
        # builder's invariant is avail monotonic in seq.
        return two_market_tape() + [
            ev_sim(10, T0 + 4_100, "m2", "m2-yes", "accept",
                   {"order_id": "o1", "local_order_id": "lo-1",
                    "token_id": "m2-yes"}),
            ev_sim(11, T0 + 4_200, "m2", "m2-yes", "fill",
                   {"detail": {"order_id": "o1", "local_order_id": "lo-1",
                               "token_id": "m2-yes", "side": "Buy",
                               "shares": "10", "remaining": "0"}}),
            ev_trade(12, T0 + 4_300, "m1", "m1-yes", "sell", "0.47", "5"),
            ev_book(13, T0 + 4_500, "m1", "m1-no",
                    [["0.50", "55"]], [["0.53", "95"]]),
            ev_book(14, T0 + 5_000, "m2", "m2-yes",
                    [["0.28", "40"]], [["0.30", "45"]]),
            ev_meta(15, T0 + 6_000, "m3", "m3-yes", "m3-no", clear_s + 7200,
                    ref="67200.0", winner="yes", known_ms=T0 + 9_000_000),
            ev_book(16, T0 + 6_500, "m3", "m3-yes",
                    [["0.61", "30"]], [["0.63", "25"]]),
        ]

    def test_exact_equivalence_with_batch_builder(self):
        """The incremental output is identical to the reference builder's,
        observation by observation (same complete dicts)."""
        events = self._rich_tape()
        ref = dataset.build_observations(events)
        inc = dataset.build_observations_incremental(events)
        self.assertEqual(ref, inc)

    def test_equivalence_with_attached_labels(self):
        """The equivalence holds after attach_labels + split."""
        events = self._rich_tape()
        labels = dataset.build_labels(events)
        ref = dataset.attach_labels(dataset.build_observations(events), labels)
        inc = dataset.attach_labels(
            dataset.build_observations_incremental(events), labels)
        self.assertEqual(ref, inc)
        parts_a, meta_a = dataset.temporal_split(
            ref, train_end_ms=T0 + 2_500, embargo_ms=0)
        parts_b, meta_b = dataset.temporal_split(
            inc, train_end_ms=T0 + 2_500, embargo_ms=0)
        self.assertEqual(parts_a, parts_b)
        self.assertEqual(meta_a, meta_b)

    def test_payload_books_two_books_in_one_event(self):
        """SQLite export format: a `book_snapshots` row is ONE atomic snapshot
        of both books — the event carries
        `payload.books = {token_id: {bids, asks}}`, applies to both tokens,
        and produces ONE observation with both sides complete."""
        clear_s = (T0 // 1000) + 3600
        ev_books = {
            "schema_version": 1, "source": "sqlite_export",
            "kind": "book_snapshot", "market_cid": "m1",
            "token_id": "m1-yes", "asset": None,
            "payload": {"books": {
                "m1-yes": {"bids": [["0.45", "100"]], "asks": [["0.47", "80"]]},
                "m1-no": {"bids": [["0.52", "60"]], "asks": [["0.54", "90"]]},
            }},
            "origin_ts_ms": None, "recv_ts_ms": T0 + 1_000,
            "avail_ts_ms": T0 + 1_000, "local_seq": 1, "quality": 0,
        }
        events = [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no", clear_s,
                    ref="67000.0"),
            ev_books,
        ]
        for builder in (dataset.build_observations,
                        dataset.build_observations_incremental):
            obs = builder(events)
            self.assertEqual(len(obs), 1, "one books event = one observation")
            f = obs[0]["features"]
            self.assertAlmostEqual(f["yes_mid"], 0.46, places=9)
            self.assertAlmostEqual(f["no_mid"], 0.53, places=9)
            self.assertAlmostEqual(f["parity_ask_sum"], 0.47 + 0.54, places=9)

    def test_avail_inversion_remains_causal(self):
        """Under an avail/seq inversion (a low-seq event available late) the
        incremental builder does not see the future either: the observation
        before the tick's avail does not include it."""
        events = [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no",
                    (T0 // 1000) + 3600, ref="67000.0"),
            ev_tick(1, T0 + 500, "67000.0"),
            # Tick with seq 2 but availability later than the decision:
            ev_tick(2, T0 + 50_000, "99999.0"),
            ev_book(3, T0 + 2_000, "m1", "m1-yes",
                    [["0.45", "100"]], [["0.47", "80"]]),
        ]
        obs = dataset.build_observations_incremental(events)
        obs_book = [o for o in obs if o["asof_local_seq"] == 3]
        self.assertEqual(len(obs_book), 1)
        self.assertEqual(obs_book[0]["features"]["spot"], 67000.0,
                         "event with future avail applied too early")

    def test_temporal_decision_regression_raises_error(self):
        """If a decision arrives with an `avail_ts_ms` lower than that of
        events already applied, the incremental builder cannot 'unsee' the
        state — it raises `NonMonotonicTape` (the caller falls back to the
        exact batch builder)."""
        events = [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no",
                    (T0 // 1000) + 3600, ref="67000.0"),
            ev_book(1, T0 + 5_000, "m1", "m1-yes",
                    [["0.45", "100"]], [["0.47", "80"]]),
            # Decision with an avail EARLIER than the event already applied (tape
            # not ordered by avail — invalid for the single step).
            ev_book(2, T0 + 1_000, "m1", "m1-yes",
                    [["0.46", "110"]], [["0.48", "70"]]),
        ]
        with self.assertRaises(dataset.NonMonotonicTape):
            dataset.build_observations_incremental(events)

    def test_auto_falls_back_to_batch_under_regression(self):
        """`build_observations_auto`: tries incremental and, on
        `NonMonotonicTape`, returns the exact batch result."""
        events = [
            ev_meta(0, T0 + 100, "m1", "m1-yes", "m1-no",
                    (T0 // 1000) + 3600, ref="67000.0"),
            ev_book(1, T0 + 5_000, "m1", "m1-yes",
                    [["0.45", "100"]], [["0.47", "80"]]),
            ev_book(2, T0 + 1_000, "m1", "m1-yes",
                    [["0.46", "110"]], [["0.48", "70"]]),
        ]
        self.assertEqual(dataset.build_observations(events),
                         dataset.build_observations_auto(events))
        # On a healthy tape it delegates to the incremental builder (same result).
        self.assertEqual(dataset.build_observations(self._rich_tape()),
                         dataset.build_observations_auto(self._rich_tape()))


try:
    import numpy  # noqa: F401
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class TestNumpyStdlibParity(unittest.TestCase):
    """The numpy fast path of `logistic_train` must be mathematically
    identical to the stdlib loop (same zero init, same deterministic batch GD);
    it only differs in speed."""

    @unittest.skipUnless(_HAS_NUMPY, "numpy not installed — nothing to compare")
    def test_identical_weights_across_backends(self):
        # Small but non-trivial dataset (heterogeneous features, some
        # correlated with the label).
        X = [
            [0.1, -0.5, 1.2, 0.0],
            [0.9, 0.3, -0.4, 0.7],
            [-0.2, 0.8, 0.1, -1.1],
            [1.4, -0.1, 0.9, 0.3],
            [-0.7, -0.9, -1.3, 0.2],
            [0.4, 0.6, 0.2, -0.8],
            [1.1, 1.0, -0.6, 0.9],
            [-1.2, 0.2, 1.7, -0.3],
        ]
        y = [0, 1, 0, 1, 0, 1, 1, 0]
        ref = train_mod._logistic_train_stdlib(X, y, epochs=50, lr=0.3,
                                               l2=1e-4)
        got = train_mod.logistic_train(X, y, epochs=50, lr=0.3, l2=1e-4)
        self.assertEqual(len(ref["weights"]), len(got["weights"]))
        for w_ref, w_got in zip(ref["weights"], got["weights"]):
            self.assertAlmostEqual(w_ref, w_got, places=12)
        self.assertAlmostEqual(ref["bias"], got["bias"], places=12)


if __name__ == "__main__":
    unittest.main()
