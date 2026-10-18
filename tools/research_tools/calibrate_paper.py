#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper simulator calibrator and bounded certification.

Measures the fidelity of the simulator (`SimulatedRouter` / replay engine)
against the evidence recorded in the bot's DBs, all read-only:

- **Real reference trace** (`hrf_state_*.sqlite::fills`): the fills the
  simulator produced while trading in paper — they are the authoritative
  reference, with the declared caveat that they were produced by the online
  version of the sim (nominal profile) and under simultaneous multi-strategy
  operation (shared exposure/balance), which a single-strategy replay does
  not reproduce 1:1.
- **Recorded book** (`hrf_market_analytics_*.sqlite::book_snapshots`):
  public evidence against which the CONSISTENCY of each fill is validated
  (not the exact counterfactual — without a trade tape the queue priority
  is not observable).
- **Feed latencies** (`hrf_spot_stream_*.sqlite::spot_samples`): triple
  timestamp (source/recv/avail) — the only real latency measurement available.
  Books have no `origin_ts` (recorder off): book latency is declared NOT
  identified, not interpolated.
- **Post-fill markout** (`markouts`, already recorded online at +5s/+30s).

Two disjoint periods: candidate tuning is done ONLY with the calibration
period (first temporal half); certification is evaluated over the
evaluation period (second half). The tolerances are constants frozen
a priori — see the TOLERANCES block and its rationale.

Evaluated profiles:
- `nominal`: reproduces the current online behavior (κ=1, participation=1,
  no stale/gap marks). It is the profile that produced the recorded trace.
- `conservative`: `ResearchSimConfig::conservative()` (κ=0, participation=0.25,
  stale/gap marks, generous estimated latencies).
- `calibrated-T7`: what this tool proposes from the measurement —
  installed as a `[research_sim]` section in a SHADOW config-dir (copy of
  `config/` + adjusted section) and run with `--research-profile config`;
  the operational TOML is not touched.

`hrf_backtest --engine pipeline` exposes `--research-profile
nominal|conservative|config` (wired by the coordinator): `config` reads the
`[research_sim]` section of the `--config-dir` TOML, so a calibrated
profile is certified without touching `src/` or the operational config.

Comparability: the replay↔trace comparison first goes through a
conditions classifier (`classify_comparability`): same inputs
(tape/DB/window), same strategy+config+version, same capital and initial
inventory state, same running orders at start, same seed and declared
warm-up. Any difference or undeclared condition yields
`no_comparable` with the reasons — NEVER a fidelity verdict (neither «ok»
nor «out_of_tolerance»). The online multi-strategy paper trace compared
with a cold single-strategy replay is structurally not comparable: the
metrics are still computed as diagnostics, but do not certify.

The replay this tool runs is a RE-DECIDING POLICY replay
(the strategy is rebuilt and re-decides over the tape) — mode (b). The
replay of FIXED actions of the historical agent (mode a), the only one that
supports controlled order/fill matching, is not available today: this is declared.

Global label: without comparable real-exchange executions, the report
says literally `verified_internal_consistency;
external_fidelity_unidentified` — it neither demands live nor fakes
external certification.

Usage:
    python3 tools/research_tools/calibrate_paper.py run [--since-ms N --until-ms N] \
        [--replay-bin target/release/hrf_backtest] [--replay-hours H] \
        [--replay-profiles nominal,conservative,config] \
        [--out-dir research/calibration]
"""

import argparse
import bisect
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_inventory as ri  # noqa: E402  (reuses resolution + readonly)

VERSION = 2
UM = 1_000_000  # micro-units (µUSDC / µshares) — all money in integers

# ── Tolerances frozen A PRIORI ──────────────────────────────────────────────
#
# Defined BEFORE the measurement, by tolerable economic error and signal
# lifetime — never tuned after the fact over the results:
#
# - TOL_PRICE_UM = 1 standard tick (0.01 USDC): it is the smallest economic
#   error distinguishable by the venue and ≈1% of the maximum payout per
#   share. A fill that contradicts the tape by more than one tick is not
#   rounding noise.
TOL_PRICE_UM = 10_000
# - COVERAGE_WINDOW_MS: the recorder deduplicates by content — the absence
#   of a row means the book is UNCHANGED, so the last snapshot is still
#   valid evidence. 120 s covers static books; beyond that a feed loss is
#   suspected and the fill is «without_coverage» (neither certifies nor condemns).
COVERAGE_WINDOW_MS = 120_000
# - TOL_TIME_BIAS_MS: the signal lifetime of these brackets is seconds
#   (book cadence ~sub-second on crypto markets; strategy watchdog 5 s).
#   A median shift >2 s between sim and real fills indicates the sim
#   decides against another market state.
TOL_TIME_BIAS_MS = 2_000
# - COUNT_BAND: acceptable n_sim/n_real ratio. The online multi-strategy run
#   is not reproducible by a single-strategy replay — the band is wide on
#   purpose; outside [0.5, 2] the divergence is structural.
COUNT_BAND = (0.5, 2.0)
# - MATCH_FILL_MS: window to match a sim fill with the closest real fill of
#   the same token+side in the trace comparison.
MATCH_FILL_MS = 120_000
# - MIN_OBS: minimum observations for a metric to leave «unidentified»
#   (one fill or a dozen certify nothing).
MIN_OBS = 30
# - FEED_GAP_MS: discontinuity between consecutive snapshots of the same
#   market counted as a feed/service gap (1 min, same threshold as the
#   inventory tool).
FEED_GAP_MS = 60_000
# - MARKOUT_HORIZONS_MS: the ones the recorder already writes online.
MARKOUT_HORIZONS_MS = (5_000, 30_000)
# - OVERCONSUMPTION_FRACTION_TOL: fraction of taker fills with overconsumption
#   tolerated before declaring the profile out of range. 5% = recorder
#   deduplication/latency tail, not systematic bias.
OVERCONSUMPTION_FRACTION_TOL = 0.05
# - IMPOSSIBLE_FRACTION_TOL: fills impossible under ALL readings (maker and
#   taker). 0.5% = margin for fills at the edge of the coverage window.
IMPOSSIBLE_FRACTION_TOL = 0.005

# ── Verdicts of a fill against the book ─────────────────────────────────────

VERDICT_OK = "ok"
VERDICT_IMPOSSIBLE = "impossible"
VERDICT_NO_COVERAGE = "without_coverage"
VERDICT_OVERCONSUMPTION = "overconsumption"
VERDICT_NOT_MEASURABLE = "unmeasurable"

# Severity for the conservative reading (worst maker/taker interpretation):
# a fill is only certified «ok» if it is ok under ALL readings.
_SEVERITY = {
    VERDICT_OK: 0,
    VERDICT_NOT_MEASURABLE: 1,
    VERDICT_OVERCONSUMPTION: 2,
    VERDICT_IMPOSSIBLE: 3,
    VERDICT_NO_COVERAGE: -1,
}

STATE_OK = "ok"
STATE_OUTSIDE = "out_of_tolerance"
STATE_NOT_IDENTIFIED = "unidentified"
# Comparability: comparing runs is NOT a fidelity verdict when
# the conditions differ or are missing — it is a separate structural state.
STATE_NOT_COMPARABLE = "no_comparable"
COMPARABLE_STATE = "comparable"

# Signal horizon: the quality/latency thresholds are set by the
# LIFETIME OF THE SIGNAL the strategy exploits, not by the bracket
# duration (5 min). The strategy watchdog runs at ~5 s and the book
# cadence on crypto markets is sub-second → signal horizon ≈ 5 s.
# Decision latency budget = horizon/10, same convention as
# `model_observer` (effective_budget = signal_horizon_ms / 10).
SIGNAL_HORIZON_MS = 5_000
LATENCY_BUDGET_MS = SIGNAL_HORIZON_MS // 10

# Replay mode this tool runs: the engine rebuilds the strategy and
# RE-DECIDES over the tape. This is mode (b) — a replay of FIXED actions
# of the historical agent (a), the only one that supports controlled
# order/fill matching, does not exist today in the pipeline.
REPLAY_MODE = "policy_redecides"
REPLAY_MODE_NOTE = (
    "policy replay that re-decides over the tape (mode b): the "
    "strategy is rebuilt with the given config and re-decides — it does not "
    "reproduce the fixed actions of the historical agent (mode a), so "
    "controlled order/fill matching is not supported"
)

# Global report label: the reference is an INTERNAL paper trace,
# never a real exchange execution → external fidelity is not identifiable.
INTERNAL_LABEL = (
    "verified_internal_consistency; external_fidelity_unidentified"
)

# Fields that define a controlled comparison: all of them must be
# EQUAL across both runs; a field not declared on either side invalidates
# the comparison (it is not reconstructed artificially).
CONDITION_FIELDS = (
    "inputs",                  # same tape/DB/window
    "strategies",              # same strategy (set)
    "config",                  # same effective config
    "version",                 # same code version
    "initial_capital",         # same initial capital
    "initial_inventory",       # same inventory state at start
    "initial_active_orders",   # same live orders at start
    "seed",                    # same seed
    "warmup",                  # declared, identical warm-up
)

# Initial conditions of the replay this tool runs: passed EXPLICITLY to the
# binary (--seed/--balance) and declared in `replay_conditions` — same
# source, no possible divergence.
SEED_REPLAY = 0xC0FFEE15
BALANCE_REPLAY_USDC = "5000"


# ── Statistics (stdlib) ─────────────────────────────────────────────────────


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    m = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return m


def quantile(vals, q):
    """Quantile with linear interpolation (same convention as numpy 'linear')."""
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    if n == 1:
        return v[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return v[lo]
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def summary(vals, min_obs=MIN_OBS):
    """n + median + p90 + max; state «unidentified» if n < min_obs."""
    vals = list(vals)
    n = len(vals)
    out = {
        "n": n,
        "state": STATE_OK if n >= min_obs else STATE_NOT_IDENTIFIED,
        "median": median(vals),
        "p90": quantile(vals, 0.9),
        "max": max(vals) if vals else None,
    }
    return out


# ── Fill vs book: verdicts ──────────────────────────────────────────────────


def a_um(text):
    """'0.55' → 550000 (µUSDC). Decimal, never float."""
    return int(Decimal(str(text)) * UM)


def parabolic_fee_um(price_um, rate):
    """Fee of the up/down contract per share: `rate × p × (1−p)` in µUSDC.
    `rate` is a Decimal (e.g. 0.07 crypto taker); None/0 → 0."""
    if rate is None:
        return 0
    fee = Decimal(rate) * Decimal(price_um) * Decimal(UM - price_um) / Decimal(UM)
    return int(fee)


def traded_fee_um(allin_um, rate, side):
    """Fee on the TRADED price, not on the all-in price.

    The persisted all-in already includes the fee (BUY: p+fee; SELL: p−fee),
    so f(allin) biases the fee. One fixed-point iteration is enough (the fee
    is ≤ ~7% of the price and the function is contractive here):
    BUY  p1 = allin − f(allin); fee = f(p1)   (allin = p + fee(p))
    SELL p1 = allin + f(allin); fee = f(p1)   (allin = p − fee(p))
    """
    f0 = parabolic_fee_um(allin_um, rate)
    p1 = allin_um - f0 if side == "BUY" else allin_um + f0
    p1 = min(max(p1, 0), UM)
    return parabolic_fee_um(p1, rate)


def depth_up_to(levels, price_um, is_asks):
    """Shares (µ) shown on levels reachable at `price_um`:
    asks with price ≤ p (a buy sweeps them) / bids with price ≥ p."""
    if is_asks:
        return sum(s for p, s in levels if p <= price_um)
    return sum(s for p, s in levels if p >= price_um)


def fill_verdict(fill, book, tol_price_um=None, fee_um=0):
    """Consistency of ONE fill with the current book.

    `fill`: {side: "BUY"|"SELL", price_um, size_um}. `price_um` is the
    persisted ALL-IN price (cash/shares: a buy includes the fee, a sell
    deducts it); `fee_um` is removed to recover the traded price.

    Verdicts:
    - "impossible": the traded price contradicts the tape under
      this reading (buy above the highest ask shown, buy below the best bid,
      symmetric sells, or against an empty side).
    - "overconsumption" (overconsumption): price in the taker zone but `size`
      larger than the depth shown reachable at that price — the sim consumed
      liquidity the tape does not show.
    - "unmeasurable": price inside the spread — a resting maker only fills
      by a cross and without a trade tape the cross is not observable.
    - "ok": taker-like with enough depth.
    - "without_coverage": book None (outside the evidence window).

    `tol_price_um` defaults to the market tick (`book["tick_um"]`, the
    snapshot's `tick_size` column — Polymarket has 0.001 ticks); it falls
    back to TOL_PRICE_UM only if the snapshot carries no tick.
    """
    if book is None:
        return VERDICT_NO_COVERAGE
    if tol_price_um is None:
        tol_price_um = book.get("tick_um") or TOL_PRICE_UM
    side = fill["side"]
    size = fill["size_um"]
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if side == "BUY":
        p = fill["price_um"] - fee_um  # traded = all-in − fee
        if not asks:
            return VERDICT_IMPOSSIBLE
        best_ask = asks[0][0]
        top_ask = asks[-1][0]
        best_bid = bids[0][0] if bids else None
        if p > top_ask + tol_price_um:
            return VERDICT_IMPOSSIBLE
        if best_bid is not None and p < best_bid - tol_price_um:
            return VERDICT_IMPOSSIBLE
        if p >= best_ask - tol_price_um:
            if size > depth_up_to(asks, p + tol_price_um, is_asks=True):
                return VERDICT_OVERCONSUMPTION
            return VERDICT_OK
        return VERDICT_NOT_MEASURABLE
    # SELL: all-in = notional − fee → traded = all-in + fee
    p = fill["price_um"] + fee_um
    if not bids:
        return VERDICT_IMPOSSIBLE
    best_bid = bids[0][0]
    low_bid = bids[-1][0]
    best_ask = asks[0][0] if asks else None
    if p < low_bid - tol_price_um:
        return VERDICT_IMPOSSIBLE
    if best_ask is not None and p > best_ask + tol_price_um:
        return VERDICT_IMPOSSIBLE
    if p <= best_bid + tol_price_um:
        if size > depth_up_to(bids, p - tol_price_um, is_asks=False):
            return VERDICT_OVERCONSUMPTION
        return VERDICT_OK
    return VERDICT_NOT_MEASURABLE


def conservative_verdict(fill, book, fee_um):
    """Maker/taker reading of the all-in price when the ROLE is unknown.

    Conservative about CONDEMNING: the fill is «impossible» only if it is
    so under BOTH readings; if any admissible reading explains it, it gets the
    LEAST severe verdict (min) — symmetric to the union of earlier∨later
    evidence in `multi_verdict`.

    Returns (verdict, ambiguous_role): `ambiguous_role`=True when the two
    readings disagree — reported for observability, not silenced."""
    v_maker = fill_verdict(fill, book, fee_um=0)
    v_taker = fill_verdict(fill, book, fee_um=fee_um)
    return (
        min((v_maker, v_taker), key=lambda v: _SEVERITY[v]),
        v_maker != v_taker,
    )


def multi_verdict(fill, evidence_items):
    """Verdict against ALL the in-window evidence (production
    function — used by `evaluate_fills_vs_books`).

    `evidence_items` = [(book, fee_um, direction)] with direction
    "earlier"|"later" and fee already estimated on the traded price of
    THAT evidence (its `parabolic_fee_rate`). The tape is a ~1s SAMPLING
    with content dedup: a fill between two recorded states can only be
    ruled out if it contradicts BOTH — the final verdict is the LEAST severe
    among the evidence (the tape has to rule out the fill, not one
    lucky sample).

    Returns (verdict, stable, ambiguous_role): `stable` is True only with
    earlier AND later evidence whose bid/ask bands agree within
    one tick — an «impossible» with a stable bracket is a hard violation; with
    the book in transition it is ambiguous (the tape does not bound the real state).
    `ambiguous_role` = some evidence disagrees between the maker and taker reading.
    """
    if not evidence_items:
        return VERDICT_NO_COVERAGE, None, False
    res = [conservative_verdict(fill, b, fee) for b, fee, _ in evidence_items]
    v = min((r[0] for r in res), key=lambda x: _SEVERITY[x])
    ambiguous = any(r[1] for r in res)
    b_ant = next((b for b, _, d in evidence_items if d == "earlier"), None)
    b_post = next((b for b, _, d in evidence_items if d == "later"), None)
    return v, stable_bracket(b_ant, b_post), ambiguous


def stable_bracket(b_ant, b_post, tol_price_um=None):
    """True only with evidence on BOTH sides whose bands (best bid/ask)
    agree within one tick: the fill happened with the market quiet and the
    tape bounds the real state tightly. The tick is taken from the snapshot
    itself (`tick_um`); if they differ between earlier and later the
    LARGER one is used (more conservative when declaring stability)."""
    if b_ant is None or b_post is None:
        return None
    if tol_price_um is None:
        tol_price_um = max(
            b.get("tick_um") or TOL_PRICE_UM for b in (b_ant, b_post)
        )

    def bands(b):
        return (
            b["bids"][0][0] if b["bids"] else None,
            b["asks"][0][0] if b["asks"] else None,
        )

    bid_a, ask_a = bands(b_ant)
    bid_p, ask_p = bands(b_post)
    if None in (bid_a, bid_p, ask_a, ask_p):
        return False
    return (
        abs(bid_a - bid_p) <= tol_price_um
        and abs(ask_a - ask_p) <= tol_price_um
    )


# ── Trace comparison ────────────────────────────────────────────────────────


def time_bias_ms(deltas, tol_ms=TOL_TIME_BIAS_MS, min_obs=MIN_OBS):
    """Fill time bias: median of (ts_sim − ts_real) per pair.
    Signed median = bias; p90 of |Δ| = dispersion."""
    deltas = list(deltas)
    n = len(deltas)
    if n < min_obs:
        return {"state": STATE_NOT_IDENTIFIED, "n": n, "median_ms": median(deltas)}
    med = median(deltas)
    return {
        "state": STATE_OK if abs(med) <= tol_ms else STATE_OUTSIDE,
        "n": n,
        "median_ms": med,
        "p90_abs_ms": quantile([abs(d) for d in deltas], 0.9),
        "tol_ms": tol_ms,
    }


def count_in_band(n_real, n_sim, band=COUNT_BAND, min_obs=MIN_OBS):
    """Ratio n_sim/n_real against the a priori band."""
    if n_real < min_obs:
        return {"state": STATE_NOT_IDENTIFIED, "n_real": n_real, "n_sim": n_sim}
    ratio = n_sim / n_real
    return {
        "state": STATE_OK if band[0] <= ratio <= band[1] else STATE_OUTSIDE,
        "n_real": n_real,
        "n_sim": n_sim,
        "ratio": ratio,
        "band": list(band),
    }


def evaluate_metric(n_obs, value, within, min_obs=MIN_OBS):
    """State of a point metric: «unidentified» if observations are
    missing — NEVER an automatic pass."""
    if n_obs < min_obs:
        return {"state": STATE_NOT_IDENTIFIED, "n": n_obs, "value": value}
    return {
        "state": STATE_OK if within(value) else STATE_OUTSIDE,
        "n": n_obs,
        "value": value,
    }


def markout_signed_um(side, fill_price_um, mid_um):
    """Signed markout: positive = favorable move after the fill.
    BUY: mid−price; SELL: price−mid. None if the mid does not exist."""
    if mid_um is None:
        return None
    return (mid_um - fill_price_um) if side == "BUY" else (fill_price_um - mid_um)


def fit_share(ratios, min_obs=MIN_OBS):
    """Bound for `share` from the observed size_fill/depth_at_touch ratio
    of real taker fills: empirical p95 clamped to (0,1].
    With few obs → «unidentified» (the conservative default 0.25 rules)."""
    ratios = [r for r in ratios if r is not None]
    n = len(ratios)
    if n < min_obs:
        return {"state": STATE_NOT_IDENTIFIED, "n": n}
    p95 = quantile(ratios, 0.95)
    return {
        "state": STATE_OK,
        "n": n,
        "p50": quantile(ratios, 0.5),
        "p95": p95,
        "value": max(min(p95, 1.0), 1e-6),
        "max": max(ratios),
    }


# ── Read-only access to the DBs ─────────────────────────────────────────────


def load_fills(conn_state, from_ms, until_ms):
    """Real paper fills (authoritative reference) in the window."""
    rows = conn_state.execute(
        "SELECT ts_ms, market_cid, token_id, side, price_um, size_um,"
        "       strategy, order_id, exec_id"
        "  FROM fills WHERE ts_ms BETWEEN ? AND ? ORDER BY ts_ms, id",
        (from_ms, until_ms),
    ).fetchall()
    return [
        {
            "ts_ms": r[0],
            "market_cid": r[1],
            "token_id": r[2],
            "side": r[3],
            "price_um": r[4],
            "size_um": r[5],
            "strategy": r[6],
            "order_id": r[7],
            "exec_id": r[8],
        }
        for r in rows
    ]


def _parse_levels(json_str):
    """'[{"price":"0.03","size":"19"},…]' → [(price_um, size_um)] sorted."""
    import json as _json

    levels = []
    for l in _json.loads(json_str or "[]"):
        levels.append((a_um(l["price"]), a_um(l["size"])))
    return levels


_COLS_BOOK = (
    "ts_ms, yes_token_id, no_token_id, yes_bids_json, yes_asks_json,"
    " no_bids_json, no_asks_json, parabolic_fee_rate, tick_size"
)


def _book_row(r, token_id):
    """From a book_snapshots row extracts the half-book of the fill's token
    plus the fee rate and the market tick. None if the token is not in the
    row. `b["tick_um"]` = tick_size in µ (0.001 exists on Polymarket) —
    scales the price tolerance; None/unreadable → the default is used."""
    if token_id == r[1]:
        bids, asks = r[3], r[4]
    elif token_id == r[2]:
        bids, asks = r[5], r[6]
    else:
        return None
    b = {
        "bids": sorted(_parse_levels(bids), reverse=True),
        "asks": sorted(_parse_levels(asks)),
    }
    if r[8]:
        try:
            b["tick_um"] = a_um(r[8])
        except Exception:
            b["tick_um"] = None
    rate = None
    if r[7]:
        try:
            rate = Decimal(r[7])
        except Exception:
            rate = None
    return b, rate


def book_evidence(conn_an, cid, token_id, ts_ms, window_ms=COVERAGE_WINDOW_MS):
    """In-window evidence for the fill: the PREVIOUS snapshot and the
    immediately FOLLOWING one (both if within the window — the tape samples
    at ~1s with dedup and a fill can fall between two recorded states).

    Returns (evidence, problem): evidence = [(book, rate, direction,
    delta_ms)]; problem = None | "unknown_token" (the row does not contain the
    fill's token)."""
    evs = []
    problem = None
    for op, order, dirn in (("<=", "DESC", "earlier"), (">", "ASC", "later")):
        r = conn_an.execute(
            f"SELECT {_COLS_BOOK} FROM book_snapshots"
            f" WHERE market_condition_id = ? AND ts_ms {op} ?"
            f" ORDER BY ts_ms {order} LIMIT 1",
            (cid, ts_ms),
        ).fetchone()
        if r is None or abs(r[0] - ts_ms) > window_ms:
            continue
        out = _book_row(r, token_id)
        if out is None:
            problem = "unknown_token"
            continue
        evs.append((out[0], out[1], dirn, abs(r[0] - ts_ms)))
    return evs, problem


def evaluate_fills_vs_books(conn_an, fills):
    """Verdict of each fill against ALL the evidence in the window (union
    earlier∨later: the tape must rule out the fill, not one sample) +
    consumption ratios of the taker-like fills (for the participation fit).

    The verdict core is `multi_verdict` — the SAME function the tests
    exercise (no duplicated logic)."""
    count = {}
    n_stable_impossible = 0
    n_ambiguous_role = 0
    share_ratios = []
    detail = []
    for f in fills:
        evs, problem = book_evidence(
            conn_an, f["market_cid"], f["token_id"], f["ts_ms"]
        )
        if problem:
            v, stable, ambiguous = "unknown_token", None, False
        elif not evs:
            v, stable, ambiguous = VERDICT_NO_COVERAGE, None, False
        else:
            # Fee of each evidence estimated on the TRADED price
            # with the fee rate of THAT snapshot.
            evs_fee = [
                (b, traded_fee_um(f["price_um"], rate, f["side"]), d)
                for b, rate, d, _ in evs
            ]
            v, stable, ambiguous = multi_verdict(f, evs_fee)
            if v == VERDICT_IMPOSSIBLE and stable:
                n_stable_impossible += 1
            if ambiguous:
                n_ambiguous_role += 1
            # Consumption ratio for taker-like fills (crossing zone) using
            # the EARLIER evidence (the state the taker saw on arrival);
            # without earlier, the available one is used.
            b_ant = next((b for b, _, d, _ in evs if d == "earlier"), None)
            b_post = next((b for b, _, d, _ in evs if d == "later"), None)
            b_ref = b_ant if b_ant is not None else b_post
            rate_ref = next(r for b, r, _, _ in evs if b is b_ref)
            fee = traded_fee_um(f["price_um"], rate_ref, f["side"])
            v_taker = fill_verdict(f, b_ref, fee_um=fee)
            if v_taker in (VERDICT_OK, VERDICT_OVERCONSUMPTION):
                p = f["price_um"] - fee if f["side"] == "BUY" else f["price_um"] + fee
                levels = b_ref["asks"] if f["side"] == "BUY" else b_ref["bids"]
                tol = b_ref.get("tick_um") or TOL_PRICE_UM
                depth = depth_up_to(
                    levels,
                    p + tol if f["side"] == "BUY" else p - tol,
                    is_asks=f["side"] == "BUY",
                )
                if depth > 0:
                    share_ratios.append(f["size_um"] / depth)
        count[v] = count.get(v, 0) + 1
        detail.append(
            {**f, "verdict": v, "stable": stable,
             "ambiguous_role": ambiguous, "n_evidence": len(evs)}
        )
    return {
        "count": count,
        "n_stable_impossible": n_stable_impossible,
        "n_ambiguous_role": n_ambiguous_role,
        "share_ratios": share_ratios,
        "detail": detail,
    }


def measure_spot_latencies(conn_spot, from_ms, until_ms):
    """Measured feed latency: source→recv (venue→agent transport) and
    recv→avail (internal queue). Books have NO origin_ts — this measurement
    is the only one with a triple timestamp and only bounds the `feed` stage."""
    rows = conn_spot.execute(
        "SELECT recv_ts_ms - source_ts_ms, avail_ts_ms - recv_ts_ms"
        "  FROM spot_samples WHERE source_ts_ms BETWEEN ? AND ?",
        (from_ms, until_ms),
    ).fetchall()
    transport = [r[0] for r in rows]
    internal = [r[1] for r in rows]
    neg = sum(1 for d in transport if d < 0)
    out = {
        "n": len(rows),
        "negative": neg,
        "transport_ms": summary(transport, min_obs=1),
        "internal_ms": summary(internal, min_obs=1),
    }
    out["transport_ms"]["p50"] = quantile(transport, 0.5)
    out["transport_ms"]["p95"] = quantile(transport, 0.95)
    out["transport_ms"]["p99"] = quantile(transport, 0.99)
    # NOTHING in the data distinguishes a venue stamp from a local
    # receive stamp — `source_ts_ms` may have been assigned by the recorder
    # itself (in fact p50=0 indicates that). Without evidence of a real
    # remote timestamp the stage is «unidentified» BY DEFAULT; the
    # statistics are reported as descriptive, never as calibration.
    out["identified_transport"] = False
    out["transport_note"] = (
        "source_ts_ms cannot be shown to be a venue stamp (p50=0 suggests a "
        "local receive stamp) — transport «unidentified» by "
        "default; statistics are descriptive only"
    )
    return out


def measure_feed_gaps(conn_an, from_ms, until_ms):
    """Feed losses: gaps > FEED_GAP_MS between snapshots of the same
    market, segmented by time remaining to close. Dedup means a gap may
    be a static book — it is reported as an upper bound."""
    rows = conn_an.execute(
        "SELECT market_condition_id, ts_ms, clear_at FROM book_snapshots"
        " WHERE ts_ms BETWEEN ? AND ? ORDER BY market_condition_id, ts_ms",
        (from_ms, until_ms),
    )
    markets = 0
    gaps = 0
    gap_total_ms = 0
    gap_max_ms = 0
    by_bucket = {"<60s": 0, "60-300s": 0, ">300s": 0, "unknown": 0}
    prev_cid = prev_ts = prev_clear = None
    for cid, ts, clear in rows:
        if cid != prev_cid:
            markets += 1
            prev_cid, prev_ts, prev_clear = cid, ts, clear
            continue
        d = ts - prev_ts
        if d > FEED_GAP_MS:
            gaps += 1
            gap_total_ms += d
            gap_max_ms = max(gap_max_ms, d)
            # clear_at falsy (0/None) = market with no recorded close date:
            # the gap is not assigned to an invented bucket — it goes to
            # "unknown".
            remaining = (prev_clear - prev_ts // 1000) if prev_clear else None
            if remaining is None:
                bucket = "unknown"
            elif remaining < 60:
                bucket = "<60s"
            elif remaining < 300:
                bucket = "60-300s"
            else:
                bucket = ">300s"
            by_bucket[bucket] += 1
        prev_ts, prev_clear = ts, clear
    return {
        "markets": markets,
        "gaps": gaps,
        "gap_total_ms": gap_total_ms,
        "gap_max_ms": gap_max_ms,
        "gaps_by_time_remaining": by_bucket,
        "threshold_ms": FEED_GAP_MS,
        "note": "dedup: gap = static book OR real loss — upper bound",
    }


def load_markouts(conn_state, from_ms, until_ms):
    rows = conn_state.execute(
        "SELECT strategy, side, fill_price_um, mid_5s_um, mid_30s_um"
        "  FROM markouts WHERE ts_ms BETWEEN ? AND ?",
        (from_ms, until_ms),
    ).fetchall()
    out = {}
    for strat, side, fp, m5, m30 in rows:
        e = out.setdefault(strat, {"n": 0, "m5": [], "m30": []})
        e["n"] += 1
        v5 = markout_signed_um(side, fp, m5)
        v30 = markout_signed_um(side, fp, m30)
        if v5 is not None:
            e["m5"].append(v5)
        if v30 is not None:
            e["m30"].append(v30)
    return {
        s: {
            "n": e["n"],
            "markout_5s_um": summary(e["m5"], min_obs=1),
            "markout_30s_um": summary(e["m30"], min_obs=1),
            "adverse_fraction_30s": (
                sum(1 for v in e["m30"] if v < 0) / len(e["m30"]) if e["m30"] else None
            ),
        }
        for s, e in out.items()
    }


# ── Comparative replay (hrf_backtest --engine pipeline) ────────────────────


def execute_replay(
    bin_path, strategy, db_books, db_spot, from_ms, until_ms,
    latency_ms, obs_delay_ms, seed=SEED_REPLAY, config_dir="config",
    profile="nominal", balance=BALANCE_REPLAY_USDC, timeout_s=1800,
):
    """Runs the pipeline engine over the recorded tape and returns the JSON
    report + the simulated fills parsed from the digest (EXEC lines).

    `profile`, `seed` and `balance` are ALWAYS passed explicitly (nominal/5000
    are the binary's defaults, but making them explicit leaves the cmd
    self-consistent with the conditions declared in the report)."""
    cmd = [
        str(bin_path),
        "--engine", "pipeline",
        "--strategy", strategy,
        "--db-books", str(db_books),
        "--from-ms", str(from_ms),
        "--to-ms", str(until_ms),
        "--latency-ms", str(latency_ms),
        "--obs-delay-ms", str(obs_delay_ms),
        "--seed", str(seed),
        "--balance", str(balance),
        "--config-dir", str(config_dir),
        "--research-profile", profile,
    ]
    if db_spot:
        cmd += ["--db-spot", str(db_spot)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        # A replay that exceeds the timeout must not kill the whole run:
        # the strategy is left «unidentified» with the reason.
        return {
            "error": f"timeout after {timeout_s}s — window too "
            "long or binary hung"
        }
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-500:] or f"rc={proc.returncode}"}
    report = json.loads(proc.stdout)
    sims = []
    for line in report.get("digest", "").splitlines():
        if not line.startswith("EXEC|"):
            continue
        # EXEC|exec_id|order_id|token_id|side|shares|cash|ts_ms
        p = line.split("|")
        try:
            sims.append(
                {
                    "exec_id": p[1],
                    "order_id": p[2],
                    "token_id": p[3],
                    "side": "BUY" if p[4] == "Buy" else p[4].upper(),
                    "shares_um": a_um(p[5]),
                    "cash_um": a_um(p[6]),
                    "ts_ms": int(p[7]),
                }
            )
        except (IndexError, ValueError):
            continue
    for s in sims:
        s["price_um"] = int(s["cash_um"] * UM / s["shares_um"]) if s["shares_um"] else 0
    return {"report": report, "fills_sim": sims, "cmd": cmd}


def _shares_of(f):
    """Shares (µ) of the fill in either format: the real DB
    stores `size_um`; the replay's EXEC digest stores `shares_um`."""
    return f.get("shares_um") if f.get("shares_um") is not None else f.get("size_um", 0)


def _notional_of(f):
    """Total notional of the fill in µUSDC (absolute value): the real DB stores
    price per share (`price_um` × `size_um`); the EXEC digest stores the total
    cash moved (`cash_um`, signed by side → abs)."""
    if f.get("cash_um") is not None:
        return abs(f["cash_um"])
    return int(f["price_um"] * _shares_of(f) / UM)


def execution_aggregates(fills):
    """Aggregates of a fill trace by QUANTITIES, not by rows.

    The certificate compares executed shares and notional: splitting a fill
    into partials (same `order_id`) changes `n_fills` but not the quantities —
    a fill split must never worsen the comparison.
    `mean_price_um` is the aggregate VWAP (notional/shares)."""
    fills = list(fills)
    shares = sum(_shares_of(f) for f in fills)
    notional = sum(_notional_of(f) for f in fills)
    orders = {f.get("order_id") for f in fills if f.get("order_id") is not None}
    return {
        "n_fills": len(fills),
        "n_orders": len(orders),
        "shares_um": shares,
        "notional_um": notional,
        "mean_price_um": int(notional * UM / shares) if shares else None,
    }


def aggregate_in_band(real_um, sim_um, n_real, band=COUNT_BAND, min_obs=MIN_OBS):
    """Sim/real ratio over a QUANTITY (shares or notional) against the a
    priori band. `n_real` = real fills backing the quantity: without
    enough observations the ratio identifies nothing."""
    if n_real < min_obs or not real_um:
        return {
            "state": STATE_NOT_IDENTIFIED,
            "real_um": real_um,
            "sim_um": sim_um,
            "n_real": n_real,
        }
    ratio = sim_um / real_um
    return {
        "state": STATE_OK if band[0] <= ratio <= band[1] else STATE_OUTSIDE,
        "real_um": real_um,
        "sim_um": sim_um,
        "ratio": ratio,
        "band": list(band),
    }


def approximate_inventory(fills):
    """Gross exposure reconstructed ONLY from fills: signed sum of shares
    per token (BUY +size / SELL −size) and total exposure = Σ|pos| evaluated
    after each fill (mean per event, not per time — declared as such).

    It is an informative bound without valuation or reserves: partial fills
    do not inflate it and the result is comparable across traces with the
    same convention."""
    pos = {}
    exposures = []
    n_tokens = set()
    for f in sorted(fills, key=lambda x: x["ts_ms"]):
        tok = f.get("token_id")
        s = _shares_of(f)
        delta = s if f["side"] == "BUY" else -s
        pos[tok] = pos.get(tok, 0) + delta
        n_tokens.add(tok)
        exposures.append(sum(abs(v) for v in pos.values()))
    return {
        "n_tokens": len(n_tokens),
        "max_exposure_um": max(exposures) if exposures else 0,
        "mean_exposure_um": (
            int(sum(exposures) / len(exposures)) if exposures else 0
        ),
        "final_exposure_um": exposures[-1] if exposures else 0,
        "note": "inventory reconstructed from fills, mean per event — "
        "no mark-to-market valuation or reserves",
    }


def compare_traces(real_fills, fills_sim, comparability=None):
    """Aggregate real↔sim comparison: row count (diagnostic), bands
    over shares/notional (quantity certificate), time bias and
    price deviation on pairs matched by (token_id, side, closest
    ts ≤ MATCH).

    `comparability` (optional) is the verdict of
    `classify_comparability`: it is attached to the result so the
    profile certificate never derives a fidelity verdict from a
    comparison with differing conditions."""
    cmp_ = {
        "count": count_in_band(len(real_fills), len(fills_sim)),
        "aggregates": {
            "real": execution_aggregates(real_fills),
            "sim": execution_aggregates(fills_sim),
        },
    }
    ag_r, ag_s = cmp_["aggregates"]["real"], cmp_["aggregates"]["sim"]
    cmp_["shares"] = aggregate_in_band(
        ag_r["shares_um"], ag_s["shares_um"], len(real_fills)
    )
    cmp_["notional"] = aggregate_in_band(
        ag_r["notional_um"], ag_s["notional_um"], len(real_fills)
    )
    if ag_r["mean_price_um"] is not None and ag_s["mean_price_um"] is not None:
        cmp_["mean_price_deviation_um"] = evaluate_metric(
            len(real_fills),
            abs(ag_s["mean_price_um"] - ag_r["mean_price_um"]),
            within=lambda v: v <= TOL_PRICE_UM,
        )
    else:
        cmp_["mean_price_deviation_um"] = {
            "state": STATE_NOT_IDENTIFIED,
            "n": len(real_fills),
        }
    if comparability is not None:
        cmp_["comparability"] = comparability
    pairs = []
    used = set()
    real_ord = sorted(real_fills, key=lambda f: f["ts_ms"])
    for s in sorted(fills_sim, key=lambda f: f["ts_ms"]):
        best = None
        best_d = None
        for i, r in enumerate(real_ord):
            if i in used or r["token_id"] != s["token_id"] or r["side"] != s["side"]:
                continue
            d = abs(r["ts_ms"] - s["ts_ms"])
            if d > MATCH_FILL_MS:
                continue
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None:
            used.add(best)
            r = real_ord[best]
            pairs.append(
                {
                    "dt_ms": s["ts_ms"] - r["ts_ms"],
                    "dprecio_um": s["price_um"] - r["price_um"],
                    "token_id": s["token_id"],
                }
            )
    cmp_["pairs"] = len(pairs)
    cmp_["time_bias"] = time_bias_ms([p["dt_ms"] for p in pairs])
    cmp_["price_deviation_um"] = evaluate_metric(
        len(pairs),
        median([abs(p["dprecio_um"]) for p in pairs]) or 0,
        within=lambda v: v <= TOL_PRICE_UM,
    )
    return cmp_


# ── Comparability classifier ─────────────────────
#
# A comparison only produces fidelity verdicts if both runs
# share ALL the conditions in `CONDITION_FIELDS`. An ONLINE multi-strategy
# paper trace is never comparable with a cold single-strategy replay:
# neither the initial capital/inventory state nor the live orders nor the
# seed are recoverable from the DB — reasons are declared, they are not
# reconstructed.


def _norm_cond(v):
    """Normalizes a conditions field for comparison: strategies as a sorted
    set, orders/inventory/lists sorted, dicts as is."""
    if isinstance(v, (list, tuple, set, frozenset)):
        try:
            return sorted(v)
        except TypeError:
            return [repr(x) for x in v]
    return v


def classify_comparability(cond_a, cond_b):
    """Comparability verdict between two runs.

    `cond_a`/`cond_b` are dicts with the keys of `CONDITION_FIELDS` (plus an
    informative `mode`). Returns:

    - ``state``: "comparable" only if ALL fields are declared on
      both sides and equal; "no_comparable" in any other scenario.
    - ``reasons``: readable list of differences/absences — the evidence of
      why, never a fidelity failure.
    - ``fields``: per field, ``equal`` | ``differs`` | ``undeclared``.
    """
    fields = {}
    reasons = []
    for field in CONDITION_FIELDS:
        val_a = _norm_cond(cond_a.get(field))
        vb = _norm_cond(cond_b.get(field))
        # Only `None` = not declared: an EMPTY list/dict is a legitimate,
        # declared initial condition (e.g. cold start = 0 live orders,
        # inventory state {}).
        if val_a is None or vb is None:
            side = []
            if val_a is None:
                side.append(cond_a.get("mode", "A"))
            if vb is None:
                side.append(cond_b.get("mode", "B"))
            fields[field] = "undeclared"
            reasons.append(
                f"{field} not declared in {', '.join(side)} — "
                "initial condition missing (not reconstructed)"
            )
        elif val_a != vb:
            fields[field] = "differs"
            reasons.append(f"{field} differs: {val_a!r} ≠ {vb!r}")
        else:
            fields[field] = "equal"
    return {
        "state": COMPARABLE_STATE if not reasons else STATE_NOT_COMPARABLE,
        "reasons": reasons,
        "fields": fields,
        "conditions": {"a": cond_a, "b": cond_b},
    }


def real_trace_conditions(strategies, inputs, n_fills):
    """Declarable conditions of the ONLINE paper trace (reference).

    The DB records fills but NOT the initial state of the run: capital,
    inventory state, live orders (GTDs issued before the window) and seed
    stay ``None`` — undeclared, never artificially reconstructed. The
    online warm-up is «continuous» (the bot was already trading)."""
    return {
        "mode": "online_paper_multi_strategy",
        "inputs": inputs,
        "strategies": sorted(strategies),
        "config": "operational_at_the_moment (not versioned in the DB)",
        "version": None,  # the DB does not record the code version
        "initial_capital": None,
        "initial_inventory": None,
        "initial_active_orders": None,
        "seed": None,
        "warmup": "continuous",
        "n_window_fills": n_fills,
    }


def replay_conditions(strategy, profile, config_dir, seed, inputs, balance):
    """Conditions of the single-strategy replay: re-deciding policy
    over the same tape, cold start (no orders or inventory state)."""
    return {
        "mode": "cold_single_strategy_replay",
        "inputs": inputs,
        "strategies": [strategy],
        "config": f"{config_dir} (profile {profile})",
        "version": "working-tree",
        "initial_capital": balance,
        "initial_inventory": {},
        "initial_active_orders": [],
        "seed": seed,
        "warmup": "cold",
    }


def profile_certification_state(strategies):
    """Certification of a profile: comparability FIRST, then the
    metrics. A comparison with differing or undeclared conditions is
    «no_comparable» — never «ok» nor «out_of_tolerance» (it is not a
    fidelity verdict, it is a structural invalidity of the comparison).

    When all the comparisons are declared comparable it delegates to
    `replay_profile_state` (count/quantities within band +
    reconciliation)."""
    comps = [e for e in strategies.values() if "comparison" in e]
    if not comps:
        return STATE_NOT_IDENTIFIED
    if any(
        e["comparison"].get("comparability", {}).get("state")
        != COMPARABLE_STATE
        for e in comps
    ):
        return STATE_NOT_COMPARABLE
    return replay_profile_state(strategies)


def latency_legs(lat_spot):
    """Per-stage latency map: distinguishes INFERENCE latency from
    the feed→decision→submit→ack→fill chain. With the current data only
    the internal recv→avail queue of spot is measured (descriptive); the
    rest is declared unidentified/unmeasurable — never calibrated to 0.

    Reference thresholds by signal horizon (not by bracket
    duration): `budget_ms` = SIGNAL_HORIZON_MS/10, same convention as
    the shadow observer."""
    out = {
        "feed_book": {
            "state": STATE_NOT_IDENTIFIED,
            "note": "book_snapshots does not record origin_ts — market "
            "feed transport not identified",
            "budget_ms": LATENCY_BUDGET_MS,
        },
        "spot_transport": {
            "state": (
                "measured" if lat_spot.get("identified_transport")
                else STATE_NOT_IDENTIFIED
            ),
            "descriptive": lat_spot.get("transport_ms"),
            "note": lat_spot.get(
                "transport_note",
                "source_ts_ms cannot be shown to be a venue stamp",
            ),
            "budget_ms": LATENCY_BUDGET_MS,
        },
        "internal_queue": {
            "state": "measured_descriptive",
            "descriptive": lat_spot.get("internal_ms"),
            "note": "recv→avail of spot_samples; spot source only",
            "budget_ms": LATENCY_BUDGET_MS,
        },
        "inference": {
            "state": "unmeasured",
            "note": "no decision timestamps in the trace — inference/strategy "
            "latency is distinct from feed→decision→fill and "
            "is not recorded today",
            "budget_ms": LATENCY_BUDGET_MS,
        },
        "decision": {
            "state": "unmeasured",
            "note": "feed→decision cannot be decomposed without decision marks",
            "budget_ms": LATENCY_BUDGET_MS,
        },
    }
    for stage in ("submit", "ack", "cancel", "notify", "fill"):
        out[stage] = {
            "state": "unmeasurable",
            "note": "paper: there is no real venue — the stage is produced by "
            "the sim with the profile's synthetic latencies, not measurable "
            "against the exchange",
            "budget_ms": LATENCY_BUDGET_MS,
        }
    return out


def certification_label(real_exchange_fills):
    """Global report label. Without comparable real-exchange executions
    the only possible certification is internal — literal, without
    faking external fidelity or demanding live."""
    if real_exchange_fills:
        return "external_fidelity_evaluable"
    return INTERNAL_LABEL


def mid_posterior(conn_an, cid, token_id, ts_ms, window_ms=COVERAGE_WINDOW_MS):
    """First snapshot ≥ ts_ms within the coverage window → token mid
    ((best_bid+best_ask)/2). None without evidence."""
    r = conn_an.execute(
        f"SELECT {_COLS_BOOK} FROM book_snapshots"
        " WHERE market_condition_id = ? AND ts_ms >= ?"
        " ORDER BY ts_ms ASC LIMIT 1",
        (cid, ts_ms),
    ).fetchone()
    if r is None or abs(r[0] - ts_ms) > window_ms:
        return None
    out = _book_row(r, token_id)
    if out is None:
        return None
    book, _rate = out
    if not book["bids"] or not book["asks"]:
        return None
    return (book["bids"][0][0] + book["asks"][0][0]) // 2


def markouts_sim(conn_an, fills_sim, mapping, horizons=MARKOUT_HORIZONS_MS):
    """Adverse selection of the SIM fills: mid markout at +5s/+30s after
    each fill, using the recorded book (same convention as the online
    `markouts`). Approximation: the mid of the first snapshot after the
    horizon within the coverage window."""
    out = {}
    for f in fills_sim:
        cid = f.get("market_cid")
        if cid is None:
            continue
        strat = f.get("strategy") or "?"
        e = out.setdefault(strat, {h: [] for h in horizons} | {"n": 0})
        e["n"] += 1
        for h in horizons:
            mid = mid_posterior(conn_an, cid, f["token_id"], f["ts_ms"] + h)
            v = markout_signed_um(f["side"], f["price_um"], mid)
            if v is not None:
                e[h].append(v)
    return {
        s: {
            "n": e["n"],
            **{
                f"markout_{h // 1000}s_um": summary(e[h], min_obs=1)
                for h in horizons
            },
        }
        for s, e in out.items()
    }


def token_to_cid_map(conn_an, token_ids):
    """token_id → market_condition_id resolved from book_snapshots (the
    replay's EXEC digest carries no cid; the token identifies the market)."""
    mapping = {}
    for t in sorted(set(token_ids)):
        r = conn_an.execute(
            "SELECT market_condition_id FROM book_snapshots"
            " WHERE yes_token_id = ? OR no_token_id = ? LIMIT 1",
            (t, t),
        ).fetchone()
        mapping[t] = r[0] if r else None
    return mapping


def generate_shadow_config(config_dir_src, dest_dir, share):
    """Shadow config-dir = copy of the operational config + a
    `[research_sim]` section with the calibrated-T7 profile: participation = p95 of the
    consumption ratio observed in calibration (bounded to (0,1]); the rest of
    the hypotheses take the CONSERVATIVE values (κ=0, stale, gap, inventory) —
    same criterion as `ResearchSimConfig::conservative()`.

    The operational TOML is NOT modified: `--research-profile config` reads the
    section from this alternate `--config-dir`."""
    dest = Path(dest_dir)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(config_dir_src, dest)
    # Decimals are written as TOML NUMBERS (rust_decimal with the `serde`
    # feature deserializes i64/u64/f64; strings would require serde-str).
    section = (
        "\n# ── Calibrated profile installed by tools/calibrate_paper.py ──\n"
        "# SHADOW config-dir for `hrf_backtest --research-profile config` —\n"
        "# the operational general.toml is left intact.\n"
        "[research_sim]\n"
        'profile = "calibrated-T7"\n'
        "cancel_attr = 0\n"
        f"share = {share}\n"
        "quote_stale_ms = 300000\n"
        "consensus_margin = 0.02\n"
        "gap_ms = 60000\n"
        "gap_abstain = true\n"
        "track_inventory = true\n"
        "\n"
        "[research_sim.latencies]\n"
        "feed_ms = 100.0\n"
        "decision_ms = 50.0\n"
        "submit_ms = 250.0\n"
        "ack_ms = 100.0\n"
        "cancel_ms = 250.0\n"
        "notify_ms = 150.0\n"
        "jitter_ms = 50.0\n"
        "load_mult = 0.5\n"
    )
    gtoml = dest / "general.toml"
    gtoml.write_text(gtoml.read_text(encoding="utf-8") + section)
    return dest


def replay_profile_state(strategies):
    """Certification of a profile from its replay↔real comparisons.

    - `ok`: all executed strategies have the n_sim/n_real count
      within band and the replay reconciled.
    - `out_of_tolerance`: some strategy with a count outside the band.
    - `unidentified`: no comparisons executed or all insufficient.

    The verdict is over AGGREGATE CONSISTENCY against the nominal trace —
    an «out_of_tolerance» does not prove the profile is wrong (the
    structural replay↔online divergence is declared), only that it does not
    reproduce the recorded trace within the a priori band."""
    comps = [e for e in strategies.values() if "comparison" in e]
    if not comps:
        return STATE_NOT_IDENTIFIED
    if all(e.get("reconciled") for e in comps) and all(
        e["comparison"]["count"]["state"] == STATE_OK for e in comps
    ):
        return STATE_OK
    if any(e["comparison"]["count"]["state"] == STATE_OUTSIDE for e in comps):
        return STATE_OUTSIDE
    return STATE_NOT_IDENTIFIED


# ── Effective config of the operational profile (for the "nominal" profile) ───


def operational_paper_params(config_dir):
    """Reads `risk.paper_*` from general.toml — the parameters the recorded
    trace ran with (nominal profile). Read-only."""
    cfg = ri.load_toml(Path(config_dir) / "general.toml")
    risk = cfg.get("risk", {}) if isinstance(cfg, dict) else {}
    return {
        "paper_latency_ms": risk.get("paper_latency_ms"),
        "paper_fok_fill_rate": risk.get("paper_fok_fill_rate"),
        "paper_slippage_bps": risk.get("paper_slippage_bps"),
        "paper_taker_fee_rate": risk.get("paper_taker_fee_rate"),
        "paper_pair_fill_correlation": risk.get("paper_pair_fill_correlation"),
    }


# ── Main pipeline ───────────────────────────────────────────────────────────


def _iso(ms):
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _verdict_fractions(count):
    n_measurable = sum(
        n for v, n in count.items() if v not in (VERDICT_NO_COVERAGE, "unknown_token")
    )
    return {
        "n_measurable": n_measurable,
        "impossible_fraction": count.get(VERDICT_IMPOSSIBLE, 0) / n_measurable
        if n_measurable
        else None,
        "overconsumption_fraction": count.get(VERDICT_OVERCONSUMPTION, 0) / n_measurable
        if n_measurable
        else None,
        "unmeasurable_fraction": count.get(VERDICT_NOT_MEASURABLE, 0) / n_measurable
        if n_measurable
        else None,
    }


def run(args):
    repo = Path(args.repo_root).resolve()
    sources = ri.effective_sources(
        repo, repo / "config/general.toml", repo / ".env", {}
    )
    conns = {}
    for name_str in ("state", "analytics", "spot_stream"):
        p = Path(sources[name_str]["resolved_path"])
        if not p.is_file():
            print(f"error: source {name_str} missing: {p}", file=sys.stderr)
            return 2
        conns[name_str] = ri.open_readonly(p)

    # Effective window = overlap fills ∩ books (the evidence rules).
    bmin, bmax = conns["analytics"].execute(
        "SELECT MIN(ts_ms), MAX(ts_ms) FROM book_snapshots"
    ).fetchone()
    if bmin is None or bmax is None:
        print(
            "error: book_snapshots empty — without book evidence there is "
            "nothing to calibrate or certify",
            file=sys.stderr,
        )
        return 2
    since = args.from_ms if args.from_ms is not None else bmin
    until = args.until_ms if args.until_ms is not None else bmax
    since, until = max(since, bmin), min(until, bmax)
    if since >= until:
        print(
            f"error: effective window empty after the overlap "
            f"([{_iso(since)}, {_iso(until)}])",
            file=sys.stderr,
        )
        return 2
    # Temporal split: calibration = first half, evaluation = second half.
    split_ms = args.split_ms if args.split_ms is not None else (since + until) // 2
    if not (since <= split_ms <= until):
        print(
            f"error: --split-ms {split_ms} outside the window "
            f"[{since}, {until}]",
            file=sys.stderr,
        )
        return 2
    if args.replay_bin and args.replay_hours <= 0:
        print("error: --replay-hours must be > 0", file=sys.stderr)
        return 2

    fills_all = load_fills(conns["state"], since, until)
    fills_calib = [f for f in fills_all if f["ts_ms"] < split_ms]
    fills_eval = [f for f in fills_all if f["ts_ms"] >= split_ms]
    # Fills outside the books window (no evidence — counted, not
    # certified): the whole fill history for an honest count.
    n_fills_total = conns["state"].execute("SELECT COUNT(*) FROM fills").fetchone()[0]

    # ── CALIBRATION measurement (only this period fits candidates) ──
    lat_calib = measure_spot_latencies(conns["spot_stream"], since, split_ms)
    ev_calib = evaluate_fills_vs_books(conns["analytics"], fills_calib)
    part_adjustment = fit_share(ev_calib["share_ratios"])

    # ── EVALUATION measurement (never fits; certifies) ──
    lat_eval = measure_spot_latencies(conns["spot_stream"], split_ms, until)
    ev_eval = evaluate_fills_vs_books(conns["analytics"], fills_eval)
    eval_gaps = measure_feed_gaps(conns["analytics"], split_ms, until)
    markouts_eval = load_markouts(conns["state"], split_ms, until)

    frac = _verdict_fractions(ev_eval["count"])
    # Hard violation = «impossible» with a STABLE bracket (the tape bounds the
    # real state tightly). The total includes sampling transitions.
    det_eval = ev_eval["detail"]
    n_stables = sum(1 for d in det_eval if d.get("stable"))
    frac_imp_est = (
        ev_eval["n_stable_impossible"] / n_stables if n_stables else None
    )
    cert_consistency = {
        "stable_impossible": evaluate_metric(
            n_stables,
            frac_imp_est if frac_imp_est is not None else 0.0,
            within=lambda v: v <= IMPOSSIBLE_FRACTION_TOL,
        ),
        "impossible_total": evaluate_metric(
            frac["n_measurable"],
            frac["impossible_fraction"] or 0.0,
            within=lambda v: v <= IMPOSSIBLE_FRACTION_TOL,
        ),
        "overconsumption": evaluate_metric(
            frac["n_measurable"],
            frac["overconsumption_fraction"] or 0.0,
            within=lambda v: v <= OVERCONSUMPTION_FRACTION_TOL,
        ),
    }

    # ── Comparative replay (optional, bounded) ──
    replay_out = {"state": STATE_NOT_IDENTIFIED, "reason_text": "not requested (--replay-bin)"}
    if args.replay_bin:
        VALID_PROFILES = ("nominal", "conservative", "config")
        profiles_req = [
            s.strip() for s in args.replay_profiles.split(",") if s.strip()
        ]
        unknown = [p for p in profiles_req if p not in VALID_PROFILES]
        if unknown:
            print(
                f"error: --replay-profiles unknown {unknown} "
                f"(valid: {VALID_PROFILES})",
                file=sys.stderr,
            )
            return 2
        hours = args.replay_hours
        dur = int(hours * 3_600_000)
        # Replay window = the evaluation sub-window with the MOST real
        # fills of the strategy (maximizes matchable pairs); it never leaves
        # the evaluation period (calibration stays intact). The
        # same window is used for ALL profiles → comparable.
        def best_window(strat):
            ts = sorted(
                f["ts_ms"] for f in fills_eval if f["strategy"] == strat
            )
            if not ts:
                return int(max(split_ms, until - dur)), int(until)
            i_best, n_best = 0, 0
            for i, t in enumerate(ts):
                n = bisect.bisect_right(ts, t + dur) - i
                if n > n_best:
                    i_best, n_best = i, n
            d = max(ts[i_best], split_ms)
            h = min(d + dur, until)
            if h - d < dur:  # at the end of the window: move the start back
                d = max(split_ms, h - dur)
            return int(d), int(h)

        lat_op = operational_paper_params(repo / "config")
        # obs_delay only if the transport stage is IDENTIFIED (without a
        # demonstrable remote stamp the default is unidentified → 0).
        obs_delay = (
            int(lat_calib["transport_ms"]["p95"])
            if lat_calib.get("identified_transport")
            and lat_calib["transport_ms"].get("p95") is not None
            else 0
        )

        # `--research-profile config` reads `[research_sim]` from --config-dir:
        # a SHADOW config-dir is generated with the calibrated-T7 profile (the
        # participation fitted in calibration + rest conservative). If the
        # participation is not identified no profile is fabricated — the
        # `config` profile stays unidentified.
        config_dirs = {
            p: str(repo / "config") for p in profiles_req if p != "config"
        }
        shadow_info = None
        if "config" in profiles_req:
            if part_adjustment["state"] == STATE_OK:
                shadow = generate_shadow_config(
                    repo / "config",
                    Path(args.out_dir) / "config-shadow-T7",
                    part_adjustment["value"],
                )
                config_dirs["config"] = str(shadow)
                shadow_info = {
                    "config_dir": str(shadow),
                    "installed_share": part_adjustment["value"],
                    "note": "copy of config/ + calibrated-T7 [research_sim]; "
                    "the operational TOML is not modified",
                }
            else:
                shadow_info = {
                    "state": STATE_NOT_IDENTIFIED,
                    "reason_text": "participation not identified in calibration — "
                    "no calibrated [research_sim] to install",
                }

        tasks = []
        for profile in profiles_req:
            if profile == "config" and config_dirs.get("config") is None:
                continue
            for strat in args.replay_strategies.split(","):
                r_from, r_until = best_window(strat)
                real = [
                    f for f in fills_eval
                    if f["strategy"] == strat and r_from <= f["ts_ms"] <= r_until
                ]
                tasks.append((profile, strat, r_from, r_until, real))

        def job(t):
            profile, strat, r_from, r_until, real = t
            res = execute_replay(
                args.replay_bin, strat,
                sources["analytics"]["resolved_path"],
                sources["spot_stream"]["resolved_path"],
                r_from, r_until,
                latency_ms=lat_op["paper_latency_ms"] or 0,
                obs_delay_ms=obs_delay,
                config_dir=config_dirs[profile],
                profile=profile,
            )
            return profile, strat, r_from, r_until, real, res

        replay_out = {
            "state": "executed",
            "profiles": {},
            # The replay is a re-deciding policy — controlled
            # order/fill matching would require a replay of fixed
            # actions, not available today.
            "replay_mode": REPLAY_MODE,
            "replay_mode_note": REPLAY_MODE_NOTE,
        }
        # Strategies that produced the real trace in the window (the
        # «strategies» condition of the online side is MULTI-strategy even
        # though the specific fill is filtered by one).
        trace_strategies = sorted({f["strategy"] for f in fills_eval})
        results = []
        with ThreadPoolExecutor(max_workers=min(len(tasks) or 1, 6)) as ex:
            results = list(ex.map(job, tasks))
        for profile, strat, r_from, r_until, real, res in results:
            ent = replay_out["profiles"].setdefault(profile, {"strategies": {}})
            if profile == "config" and shadow_info:
                ent["shadow_config"] = shadow_info
            if "error" in res:
                ent["strategies"][strat] = {
                    "state": STATE_NOT_IDENTIFIED,
                    "reason_text": f"replay failed: {res['error']}",
                }
                continue
            sims = res["fills_sim"]
            # The SIM fills are also checked against the recorded book:
            # consistency of the profile with the public evidence, not just
            # similarity to the nominal trace.
            mapping = token_to_cid_map(conns["analytics"], [s["token_id"] for s in sims])
            for s in sims:
                s["market_cid"] = mapping.get(s["token_id"])
                s["size_um"] = s["shares_um"]
                s["strategy"] = strat
            ev_sim = evaluate_fills_vs_books(
                conns["analytics"], [s for s in sims if s.get("market_cid")]
            )
            # Conditions of both runs + classifier:
            # the aggregate comparison is still computed as a diagnostic,
            # but the certificate is «no_comparable» if the conditions
            # differ — never «out_of_tolerance».
            inputs = {
                "books_db": Path(sources["analytics"]["resolved_path"]).name,
                "spot_db": Path(sources["spot_stream"]["resolved_path"]).name,
                "window": [r_from, r_until],
            }
            comp = classify_comparability(
                real_trace_conditions(trace_strategies, inputs, len(real)),
                replay_conditions(
                    strat, profile, config_dirs[profile], SEED_REPLAY,
                    inputs, balance=BALANCE_REPLAY_USDC,
                ),
            )
            cmp_ = compare_traces(real, sims, comparability=comp)
            intents = res["report"].get("intents")
            router_calls = res["report"].get("router_calls")
            ent["strategies"][strat] = {
                "window": {"from_ms": r_from, "until_ms": r_until},
                "params": {
                    "latency_ms": lat_op["paper_latency_ms"] or 0,
                    "obs_delay_ms": obs_delay,
                    "config_dir": config_dirs[profile],
                },
                "reported_profile": res["report"].get("research_profile"),
                "reconciled": res["report"].get("reconciled"),
                "comparison": cmp_,
                "fill_ratio_sim": {
                    "fills_per_intent": (
                        len(sims) / intents if intents else None
                    ),
                    "fills_per_router_call": (
                        len(sims) / router_calls if router_calls else None
                    ),
                    "intents": intents,
                    "router_calls": router_calls,
                    "note": "ratio conditioned on replay intents; the real "
                    "side does not record intents — not comparable per fill",
                },
                "exposure": {
                    "real_approx": approximate_inventory(real),
                    "sim_approx": approximate_inventory(sims),
                },
                "markout_sim": markouts_sim(conns["analytics"], sims, mapping),
                "sim_vs_book_consistency": {
                    "n_sim": len(sims),
                    "count": ev_sim["count"],
                    "n_stable": sum(
                        1 for d in ev_sim["detail"] if d.get("stable")
                    ),
                    "n_stable_impossible": ev_sim["n_stable_impossible"],
                    "n_ambiguous_role": ev_sim["n_ambiguous_role"],
                },
                "warning": (
                    "real trace = ONLINE multi-strategy sim; replay = single-strategy "
                    "with current config + cold start (GTD orders issued before "
                    "the window are not reproduced — expected under-execution) — "
                    "aggregate consistency, not counterfactual"
                ),
            }
        if "config" in profiles_req and "config" not in replay_out["profiles"]:
            replay_out["profiles"]["config"] = {
                "state": STATE_NOT_IDENTIFIED,
                "reason_text": shadow_info["reason_text"] if shadow_info else "no shadow config",
                "strategies": {},
            }

    # ── Profiles ──
    lat_op = operational_paper_params(repo / "config")
    profiles = {
        "nominal": {
            "description": "current online behavior (κ=1, participation=1, "
            "no stale/gap marks) — it is the profile that produced the recorded trace",
            "reachable_via": "--research-profile nominal (default)",
            "reachable_params": lat_op,
            "params_sim": {
                "cancel_attr": 1.0, "share": 1.0,
                "quote_stale_ms": 0, "gap_ms": 0, "track_inventory": False,
            },
        },
        "conservative": {
            "description": "ResearchSimConfig::conservative() — worst scenario "
            "identifiable by hypothesis",
            "reachable_via": "--research-profile conservative",
            "params_sim": {
                "cancel_attr": 0.0, "share": 0.25,
                "quote_stale_ms": 300_000, "consensus_margin": 0.02,
                "gap_ms": 60_000, "gap_abstain": True, "track_inventory": True,
            },
        },
        "calibrated-T7": {
            "description": "Proposal: participation bounded to the p95 of the "
            "consumption observed in calibration; obs_delay = p95 spot "
            "transport; rest = conservative",
            "reachable_via": "--research-profile config + config-dir shadow "
            "(research/calibration/config-shadow-T7)",
            "adjusted_share": part_adjustment,
            "obs_delay_ms_propuesto": (
                int(lat_calib["transport_ms"]["p95"])
                if lat_calib.get("identified_transport")
                and lat_calib["transport_ms"].get("p95") is not None
                else None
            ),
            "obs_delay_note": (
                "p95 spot transport measured"
                if lat_calib.get("identified_transport")
                else "spot source_ts_ms cannot be shown to be a venue "
                "stamp (p50=0 suggests a local stamp): feed stage not "
                "identified — obs_delay is not proposed"
            ),
        },
    }

    # Certification per profile: nominal is judged by the consistency of the
    # real trace (it is ITS output); conservative/calibrated-T7 by their replay
    # against that trace and against the recorded book.
    profile_certification = {
        "nominal": {
            "state": cert_consistency["stable_impossible"]["state"],
            "evidence": "real online trace (its own output): fraction of "
            "impossible fills with a stable bracket",
            "value": cert_consistency["stable_impossible"].get("value"),
        },
    }
    rp = replay_out.get("profiles", {})
    for profile, name_str in (("conservative", "conservative"), ("config", "calibrated-T7")):
        ent = rp.get(profile)
        if ent is None or ent.get("state") == STATE_NOT_IDENTIFIED:
            profile_certification[name_str] = {
                "state": STATE_NOT_IDENTIFIED,
                "evidence": (ent or {}).get("reason_text")
                or "profile not run (--replay-profiles)",
            }
            continue
        profile_certification[name_str] = {
            "state": profile_certification_state(ent["strategies"]),
            "evidence": "comparability classifier per comparison "
            "(conditions in 'comparison.comparability'); if "
            "comparable: bands over shares/notional + replay "
            "reconciliation; consistency of the sim fills against the book in "
            "'sim_vs_book_consistency'",
        }

    # Comparison BETWEEN profiles within the same replay (same window,
    # seed and config): that one is comparable by construction (same
    # conditions, only the sim profile changes). It is reported over
    # QUANTITIES (shares/notional): the count of EXEC rows is NOT a bound
    # — `share`<1 splits executions into partials and the row-based ordering
    # is falsified without the profile being any less conservative
    # (monotonicity of count or PnL is not required).
    between_profiles = {}
    strategies_to_replay = sorted(
        {s for ent in rp.values() for s in ent.get("strategies", {})}
    )
    for strat in strategies_to_replay:
        by_profile = {}
        for profile, ent in rp.items():
            e = ent.get("strategies", {}).get(strat, {})
            ag = e.get("comparison", {}).get("aggregates", {}).get("sim", {})
            if ag.get("shares_um") is not None:
                by_profile[profile] = {
                    "n_fills": ag["n_fills"],
                    "n_orders": ag["n_orders"],
                    "shares_um": ag["shares_um"],
                    "notional_um": ag["notional_um"],
                }
        present = [
            by_profile[p]
            for p in ("nominal", "config", "conservative")
            if p in by_profile
        ]
        order_shares = all(
            present[i]["shares_um"] >= present[i + 1]["shares_um"]
            for i in range(len(present) - 1)
        )
        between_profiles[strat] = {
            "by_profile": by_profile,
            "order_nom_cal_cons_shares": (
                order_shares if len(present) >= 2 else None
            ),
            "note": "internal diagnostic — comparable comparison (same "
            "window/seed/config, only the profile changes); the count of "
            "EXEC rows is not a bound (partials), shares/notional are "
            "aggregate quantities",
        }

    metrics = {
        "tool": "calibrate_paper",
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # Global label — the reference is an INTERNAL paper trace,
        # never a real exchange execution (there are no real-venue fills in
        # these DBs: the «fills» were produced by the online sim).
        "certification_label": certification_label(
            real_exchange_fills=False
        ),
        "replay_mode": REPLAY_MODE,
        "replay_mode_note": REPLAY_MODE_NOTE,
        "window": {
            "from_ms": since, "until_ms": until, "split_ms": split_ms,
            "from_iso": _iso(since), "until_iso": _iso(until),
            "split_iso": _iso(split_ms),
        },
        "a_priori_tolerances": {
            "tol_price_um": TOL_PRICE_UM,
            "coverage_window_ms": COVERAGE_WINDOW_MS,
            "tol_time_bias_ms": TOL_TIME_BIAS_MS,
            "count_band": list(COUNT_BAND),
            "match_fill_ms": MATCH_FILL_MS,
            "min_obs": MIN_OBS,
            "feed_gap_ms": FEED_GAP_MS,
            "tol_overconsumption_fraction": OVERCONSUMPTION_FRACTION_TOL,
            "tol_impossible_fraction": IMPOSSIBLE_FRACTION_TOL,
            # Thresholds by SIGNAL HORIZON, not by the bracket
            # duration (5 min): latency budget = signal/10.
            "signal_horizon_ms": SIGNAL_HORIZON_MS,
            "latency_budget_ms": LATENCY_BUDGET_MS,
        },
        "adjustment_vs_evaluation": {
            "rule": "candidate parameters are fitted ONLY with the "
            "calibration period (fills < split_ms); the evaluation does not "
            "fit anything and the tolerances are frozen a priori",
            "split_ms": split_ms,
            "params_not_identifiable": "declared unidentified — "
            "never forced to 0 nor learned to favor the model",
        },
        "fills": {
            "total_db": n_fills_total,
            "in_window_books": len(fills_all),
            "calibration": len(fills_calib),
            "evaluation": len(fills_eval),
            "by_strategy_eval": {},
        },
        "spot_latencies": {"calibration": lat_calib, "evaluation": lat_eval},
        # Latency per LEG — distinguishes inference from
        # feed→decision→submit→ack→fill; what is not measurable is declared.
        "leg_latencies": latency_legs(lat_eval),
        "fill_verdicts": {
            "calibration": ev_calib["count"],
            "evaluation": ev_eval["count"],
            "eval_fractions": frac,
            "n_stable_eval": n_stables,
            "n_stable_impossible_eval": ev_eval["n_stable_impossible"],
            "n_ambiguous_role_eval": ev_eval["n_ambiguous_role"],
        },
        "consistency_certification": cert_consistency,
        "profile_certification": profile_certification,
        "comparison_between_profiles": between_profiles,
        "calib_adjusted_share": part_adjustment,
        "eval_feed_gaps": eval_gaps,
        "markouts_eval": markouts_eval,
        "replay": replay_out,
        "profiles": profiles,
        "limitations": [
            "the real fills are output of the ONLINE sim (nominal profile) — "
            "they certify the recorded trace, not the counterfactual of another profile",
            "the replay↔online trace comparison is structurally "
            "no_comparable: multi-strategy vs single-strategy, cold start, "
            "and initial capital/inventory/orders/seed not "
            "recoverable — the cross metrics are diagnostics, never a "
            "fidelity verdict",
            "the replay is a re-deciding policy (mode b); controlled "
            "order/fill matching would require a replay of fixed "
            "actions of the historical agent (mode a), not available",
            "κ/share/stale/gap/track_inventory are reachable via "
            "--research-profile (nominal|conservative|config), but their "
            "calibration against the real trace stays limited by the "
            "structural divergence of the replay (above)",
            "books without origin_ts AND local spot source_ts (p50=0): no "
            "latency stage is measurable with the current data "
            "(inference/submit/ack/fill declared unmeasurable)",
            "online multi-strategy vs cold single-strategy replay: the "
            "trace comparison is aggregate, never exact fill-by-fill — "
            "the online GTDs issued before the window have no origin "
            "in the replay (structural under-execution)",
            "a fill split into partials can evade the per-ROW overconsumption "
            "verdict — that is why the certificate uses quantities "
            "aggregated per order (shares/notional), not EXEC rows",
            "without a trade tape: queue priority and maker fills within the "
            "spread are 'unmeasurable' — priority calibration is not claimed",
            "book gaps may be dedup (static book), not real "
            "loss — they are reported as an upper bound",
        ],
    }
    for f in fills_eval:
        s = metrics["fills"]["by_strategy_eval"].setdefault(f["strategy"], 0)
        metrics["fills"]["by_strategy_eval"][f["strategy"]] = s + 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics-T7.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    report_path = out_dir / "report-T7.md"
    report_path.write_text(render_report(metrics))
    print(f"metrics: {metrics_path}")
    print(f"report:  {report_path}")
    for c in conns.values():
        c.close()
    return 0


def _table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_report(m):
    """Reproducible report generated from the metrics — same input ⇒ same
    report (except the generation date). Tolerant of v1 metrics (without
    comparability fields): what is absent is marked «not declared»."""
    v = m["window"]
    L = []
    L.append("# Bounded certification of the paper simulator\n")
    L.append(f"Generated: {m['generated_at']} · tool v{m['version']}\n")
    L.append(
        f"Effective window (overlap fills ∩ books): `{v['from_iso']}` → "
        f"`{v['until_iso']}`. Calibration < `{v['split_iso']}` ≤ evaluation.\n"
    )
    # The global label goes on top — it distinguishes an internal paper trace
    # from a real exchange execution and never fakes external certification.
    L.append(
        f"**Global label**: `{m.get('certification_label', INTERNAL_LABEL)}`"
        " — the reference is an INTERNAL PAPER trace (the fills were produced by the "
        "online sim), not a real exchange execution.\n"
    )
    if m.get("replay_mode"):
        L.append(
            f"**Replay mode**: `{m['replay_mode']}` — "
            f"{m.get('replay_mode_note', REPLAY_MODE_NOTE)}\n"
        )
    f = m["fills"]
    L.append("## 1. Coverage of the real reference trace\n")
    L.append(
        f"Fills in DB: **{f['total_db']}** · in books window: "
        f"**{f['in_window_books']}** · calibration: {f['calibration']} · "
        f"evaluation: {f['evaluation']}\n"
    )
    L.append("Fills outside the books window are «without coverage» "
             "— they neither certify nor condemn.\n")
    L.append(_table(
        [(k, n) for k, n in sorted(f["by_strategy_eval"].items())],
        ["strategy (eval)", "fills"],
    ) + "\n")

    L.append("## 2. Fills ↔ recorded book consistency (evaluation)\n")
    L.append(
        "The tape is a ~1s sampling with dedup (analytics_recorder): a "
        "fill between two recorded states is only ruled out if it contradicts the "
        "earlier AND the later snapshot (union of evidence). Also the "
        "maker/taker role of the fill is unknown: «impossible» requires it to "
        "hold under BOTH readings of the all-in price (conservative when condemning) "
        "and the discrepancy is counted as «ambiguous_role». «stable» = bid/ask "
        "bands equal within ±1 tick (the market's real tick) between the "
        "earlier and the later — hard violation; with the book in "
        "transition the «impossible» is ambiguous (the tape does not bound the state).\n"
    )
    L.append(_table(
        [(k, n) for k, n in sorted(m["fill_verdicts"]["evaluation"].items())],
        ["verdict", "fills (eval)"],
    ) + "\n")
    fr = m["fill_verdicts"]["eval_fractions"]
    vf = m["fill_verdicts"]
    L.append(
        f"Measurable: {fr['n_measurable']} · impossible total: "
        f"{_pct(fr['impossible_fraction'])} · **impossible with stable "
        f"bracket: {vf['n_stable_impossible_eval']}/{vf['n_stable_eval']}** "
        f"(tol ≤ {_pct(IMPOSSIBLE_FRACTION_TOL)}) · overconsumption: "
        f"{_pct(fr['overconsumption_fraction'])} "
        f"(tol ≤ {_pct(OVERCONSUMPTION_FRACTION_TOL)}) · unmeasurable: "
        f"{_pct(fr['unmeasurable_fraction'])} · maker/taker role ambiguous: "
        f"{vf.get('n_ambiguous_role_eval', 0)}\n"
    )
    cc = m["consistency_certification"]
    L.append(_table(
        [
            ("impossible (stable bracket)", cc["stable_impossible"]["state"],
             cc["stable_impossible"]["value"]),
            ("impossible (total, diagnostic)", cc["impossible_total"]["state"],
             cc["impossible_total"]["value"]),
            ("overconsumption", cc["overconsumption"]["state"], cc["overconsumption"]["value"]),
        ],
        ["metric", "state", "value"],
    ) + "\n")

    L.append("## 3. Measured latencies (spot — the only source with a triple timestamp)\n")
    for per in ("calibration", "evaluation"):
        t = m["spot_latencies"][per]["transport_ms"]
        ident = m["spot_latencies"][per].get("identified_transport")
        L.append(
            f"- {per}: n={m['spot_latencies'][per]['n']} · "
            f"transport source→recv p50={_ms(t.get('p50'))} "
            f"p95={_ms(t.get('p95'))} p99={_ms(t.get('p99'))} "
            f"max={_ms(t.get('max'))} · negative={m['spot_latencies'][per]['negative']}"
            + ("" if ident else
               " · **source_ts not shown to be a venue stamp → "
               "transport NOT identified**")
            + "\n"
        )
    L.append("The `feed` stage of BOOKS has no origin_ts → unidentified; "
             "and without evidence of a real remote timestamp in spot, there is "
             "no transport measurement either — the latency stays «unidentified», "
             "not calibrated to 0 (the p50/p95 above are descriptive).\n")
    stages = m.get("leg_latencies")
    if stages:
        L.append(
            "Latency per stage (thresholds by signal horizon "
            f"{m['a_priori_tolerances'].get('signal_horizon_ms', '—')} ms, "
            "not by bracket duration):\n"
        )
        L.append(_table(
            [
                (p, e.get("state", "—"), e.get("note", "—"))
                for p, e in stages.items()
            ],
            ["stage", "state", "note"],
        ) + "\n")

    L.append("## 4. Participation / overconsumption (calibration fit)\n")
    p = m["calib_adjusted_share"]
    if p["state"] == STATE_OK:
        L.append(
            f"Size/depth ratio on taker fills (calib, n={p['n']}): "
            f"p50={p['p50']:.3f} p95={p['p95']:.3f} max={p['max']:.3f} → "
            f"proposed participation ≤ {p['value']:.3f}\n"
        )
    else:
        L.append(f"Participation: **unidentified** (n={p['n']} < "
                 f"{MIN_OBS}) — the conservative default 0.25 rules.\n")

    L.append("## 5. Feed losses (evaluation)\n")
    h = m["eval_feed_gaps"]
    L.append(
        f"Markets: {h['markets']} · gaps >{h['threshold_ms']//1000}s: "
        f"{h['gaps']} · total gap {h['gap_total_ms']//1000}s · max "
        f"{h['gap_max_ms']//1000}s · by time remaining: "
        f"{h['gaps_by_time_remaining']}\n"
    )
    L.append(f"_{h['note']}_\n")

    L.append("## 6. Recorded post-fill markout (evaluation)\n")
    rows = []
    for s, e in sorted(m["markouts_eval"].items()):
        rows.append(
            (s, e["n"], _ms(e["markout_5s_um"]["median"], "um"),
             _ms(e["markout_30s_um"]["median"], "um"),
             _pct(e["adverse_fraction_30s"]))
        )
    L.append(_table(rows, ["strategy", "n", "median 5s (µ)", "median 30s (µ)",
                           "adverse fraction 30s"]) + "\n")

    L.append("## 7. Replay vs real trace comparison\n")
    r = m["replay"]
    if r.get("state") != "executed":
        L.append(f"**Unidentified**: {r.get('reason_text', 'no data')}.\n")
    else:
        L.append(
            "The certificate compares QUANTITIES (executed shares/notional), "
            "not the number of EXEC rows: a fill split into partials does not "
            "worsen the certificate. The `comparability` column is the "
            "verdict of the conditions classifier — `no_comparable` "
            "means the comparison does not support a fidelity "
            "verdict (counts/ratios stay as diagnostics).\n"
        )
        rows = []
        warning = None
        for profile, ent in sorted(r["profiles"].items()):
            if ent.get("state") == STATE_NOT_IDENTIFIED:
                L.append(f"- `{profile}`: unidentified — {ent['reason_text']}\n")
                continue
            for s, e in sorted(ent["strategies"].items()):
                if e.get("state") == STATE_NOT_IDENTIFIED:
                    rows.append((profile, s, "—", "—", "—", "—", "—", "—",
                                 "unidentified", e["reason_text"], "—", "—"))
                    continue
                c = e["comparison"]
                cb = e.get("sim_vs_book_consistency", {})
                comp_state = c.get("comparability", {}).get(
                    "state", "undeclared"
                )
                rows.append(
                    (
                        profile, s,
                        f"{_iso(e['window']['from_ms'])}→{_iso(e['window']['until_ms'])}",
                        e.get("reported_profile", "—"),
                        comp_state,
                        _ratio_txt(c.get("shares")),
                        _ratio_txt(c.get("notional")),
                        f"{c['count'].get('n_sim')}/{c['count'].get('n_real')}",
                        c["pairs"],
                        f"{cb.get('n_stable_impossible')}/{cb.get('n_stable')}",
                        _ms(c["time_bias"].get("median_ms")),
                        e["reconciled"],
                    )
                )
                warning = warning or e.get("warning")
                reasons = c.get("comparability", {}).get("reasons") or []
                if reasons:
                    L.append(
                        f"- `{profile}/{s}` not comparable: "
                        + "; ".join(reasons[:4])
                        + (f" (+{len(reasons) - 4} more)" if len(reasons) > 4 else "")
                        + "\n"
                    )
        L.append(_table(
            rows,
            ["profile", "strategy", "window", "reported_profile",
             "comparability", "shares", "notional", "n_sim/n_real",
             "pairs", "sim_impossible (stable)", "time_bias med",
             "reconciled"],
        ) + "\n")
        if warning:
            L.append(f"_{warning}_\n")
        ep = m.get("comparison_between_profiles", {})
        if ep:
            L.append("Sim quantities per profile within the same replay "
                     "(comparable internal comparison: same window/seed/"
                     "config, only the sim profile changes):\n")
            L.append(_table(
                [
                    (
                        s,
                        (
                            {
                                p: {
                                    "shares_um": v["shares_um"],
                                    "notional_um": v["notional_um"],
                                    "fills": v["n_fills"],
                                    "orders": v["n_orders"],
                                }
                                for p, v in v["by_profile"].items()
                            }
                            if "by_profile" in v
                            # v1 metrics compat (old content):
                            else v.get("n_sim_per_profile", "—")
                        ),
                        v.get(
                            "order_nom_cal_cons_shares",
                            v.get("order_nom_cal_cons"),
                        ),
                    )
                    for s, v in sorted(ep.items())
                ],
                ["strategy", "per profile (shares/notional/fills/orders)",
                 "order shares nom≥cal≥cons"],
            ) + "\n")
            L.append(
                "_The count of EXEC rows is not a bound between profiles: "
                "`share`<1 splits executions into partials — the "
                "ordering is evaluated over shares, and neither count nor PnL "
                "is required to be monotonic._\n"
            )

    L.append("## 8. Certification per profile\n")
    cp = m.get("profile_certification", {})
    L.append(_table(
        [
            (name_str,
             cp.get(name_str, {}).get("state", STATE_NOT_IDENTIFIED),
             cp.get(name_str, {}).get("evidence", "—"))
            for name_str in ("nominal", "conservative", "calibrated-T7")
        ],
        ["profile", "state", "evidence"],
    ) + "\n")
    L.append(
        "`no_comparable` = the comparison conditions differ or are not "
        "declared (online multi-strategy vs cold single-strategy, "
        "initial capital/inventory/orders and seed not recoverable): "
        "it is NOT a fidelity verdict on the profile — neither a fail nor a "
        "pass. An «out_of_tolerance» is only produced when the "
        "comparison IS comparable. The internal evidence that remains "
        "valid is the consistency of the sim fills against the book "
        "(`sim_impossible`) and the comparison between profiles of the same replay "
        "(quantities).\n"
    )

    L.append("## 9. Supported / unsupported ranges\n")
    L.append("""**Supported (with evidence in this run):**
- Consistency of taker fills against the recorded book within
  ±120 s of coverage and 1 tick of price tolerance (real trace and
  the fills of each replay profile).
- Maximum participation observed in taker fills (p95 bound of the
  size/depth ratio in calibration), installable as `[research_sim]` in a
  shadow config-dir via `--research-profile config`.
- Sim parameters (κ, participation, stale/gap, `track_inventory`)
  EXECUTABLE in replay via `--research-profile` — their comparison between
  profiles within the same replay (shares/notional quantities) is the
  only comparison that is comparable by construction.
- Post-fill markout at 5s/30s (recorded online; for sim fills the later
  mid of the recorded book is computed with the same convention).
- Replay↔real aggregates (shares/notional, VWAP, reconstructed exposure)
  as diagnostics — never as a fidelity verdict while the
  comparison is «no_comparable».

**Unsupported / unidentified (declared, not passed):**
- Fidelity verdict replay↔online trace: the comparison is
  structurally `no_comparable` (multi-strategy, cold start,
  initial conditions and seed not recoverable from the DB).
- Attribution of replay↔real divergences to ONE parameter: the real trace
  mixes multi-strategy + online sim version + orders issued before
  the window (cold start) — the comparison is aggregate.
- Latency of ALL stages: books have no `origin_ts` and spot's
  `source_ts_ms` is a LOCAL receive stamp (p50 = 0 ms) —
  neither feed nor submit/ack/cancel/notify are measurable with the current
  data.
- Queue priority and maker fills within the spread: without a trade tape
  they are «unmeasurable» — priority calibration is NOT claimed.
- Any conclusion about live: out of scope.
""")

    L.append("## 10. Declared limitations\n")
    for l in m["limitations"]:
        L.append(f"- {l}\n")
    L.append("\n## A priori tolerances (frozen)\n")
    L.append("```json\n" + json.dumps(m["a_priori_tolerances"], indent=2) + "\n```\n")
    return "".join(L)


def _ratio_txt(res_band):
    """'sim/real (state)' for the quantity bands; '—' if missing."""
    if not res_band:
        return "—"
    sim = res_band.get("sim_um")
    real = res_band.get("real_um")
    if sim is None or real is None:
        return res_band.get("state", "—")
    return f"{sim}/{real} ({res_band.get('state', '—')})"


def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def _ms(x, unit="ms"):
    return "—" if x is None else f"{x}{unit}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="measure + calibrate + certify + report")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--since-ms", dest="from_ms", type=int, default=None)
    p.add_argument("--until-ms", type=int, default=None)
    p.add_argument("--split-ms", type=int, default=None,
                   help="end of the calibration period (default: temporal half)")
    p.add_argument("--replay-bin", default=None,
                   help="path to hrf_backtest for the replay↔real comparison")
    p.add_argument("--replay-hours", type=float, default=6.0,
                   help="hours of the eval window covered by the replay (bounded)")
    p.add_argument("--replay-strategies", default="arb,near_expiry")
    p.add_argument(
        "--replay-profiles", default="nominal,conservative,config",
        help="--research-profile profiles to run per strategy "
        "(nominal|conservative|config; `config` uses the shadow config-dir "
        "with [research_sim] = calibrated-T7 if the participation is "
        "identified)",
    )
    p.add_argument("--out-dir", default="research/calibration")
    p2 = sub.add_parser(
        "report",
        help="re-renders report-T7.md from an existing metrics-T7.json",
    )
    p2.add_argument("--metrics", required=True)
    p2.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "report":
        m = json.loads(Path(args.metrics).read_text())
        out = Path(args.out) if args.out else Path(args.metrics).parent / "report-T7.md"
        out.write_text(render_report(m))
        print(f"report: {out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
