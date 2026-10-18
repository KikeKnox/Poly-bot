#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T9 offline evaluation: shadow of the T8b loaded model vs baselines over the
TEST partition (data after training, split frozen
in `model.json`).

Rebuilds the observations with the SAME causal pipeline of
`dataset.py` (streaming, two passes — it does not materialize the tape), filters to
the test markets fixed by training and to pre-close
observations (`post_close` excluded — live, the observer abstains), and
reports:

- classification metrics of the loaded model and of two baselines
  (`persistence` = train base rate; `market` = `yes_mid` as the
  implied probability);
- VIRTUAL PnL under the live shadow's decision rule
  (`p − ask > edge_min` → virtual buy of 1 share of the side with the highest
  edge): payoff = label−price (YES) / (1−label)−price (NO), without
  modeling fills or fees — a documented optimistic bound;
- coverage (fraction of observations with a virtual decision) and
  breakdown by market and by UTC day, with uncertainty (standard
  deviation of the PnL per decided observation and of the brier per group).

Method caveats (same as the T9 report):
- The observations of a tape are not independent samples — the
  breakdowns are descriptive, not hypothesis tests.
- The split is by market and chronological with respect to train; there is no
  train/test overlap by construction (frozen `split_markets`).
- This does NOT reproduce the live shadow: it is the offline upper bound —
  no latency, no staleness, no abstentions due to budget.

With `--decisions-out <f.jsonl>` it also writes the record of shadow decisions
with the live shadow's schema (`kind:"decision"` —
model_hash = sha256 of the model.json FILE in EVERY row, chain of
custody loaded model↔decision) and periodic+final `kind:"stats"`
rows that declare `emits_orders:false` (the offline path
emits no orders by construction).

Usage:
    python3 research/numeric_model/shadow_eval_t9.py \
        --events research/data/t8b_confirmed/events.jsonl \
        --model research/data/t8b_confirmed/model/model.json \
        --out research/shadow/eval_offline.json \
        --decisions-out research/shadow/shadow-tape.jsonl
"""

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import train as train_mod  # noqa: E402

EDGE_MIN = 0.02  # default of `model_shadow.edge_min` in config.rs

# Estimated taker fee of the live shadow (Polymarket model: fee =
# p·(1−p)·0.07 on the price of the side bought — verified against the
# real rows of the live shadow jsonl: ask 0.5→0.0175, 0.39→0.016653,
# 0.59→0.016933).
FEE_TAKER = 0.07

# Cadence of `stats` rows in the jsonl (like the live shadow ~5 min).
STATS_INTERVAL_MS = 300_000


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _metrics(y, p):
    if not y:
        return {"n": 0}
    m = train_mod.classification_metrics(y, p)
    m["n"] = len(y)
    return m


def _pnl_stats(pnls):
    if not pnls:
        return {"n": 0, "pnl_total": 0.0, "medium_pnl": 0.0, "std": 0.0}
    n = len(pnls)
    tot = sum(pnls)
    mean = tot / n
    var = sum((x - mean) ** 2 for x in pnls) / n
    return {
        "n": n,
        "pnl_total": tot,
        "medium_pnl": mean,
        "std": math.sqrt(var),
        # Approximate CI of the mean (descriptive — obs not independent).
        "mean_ic95": 1.96 * math.sqrt(var / n) if n else None,
    }


def _decision_virtual(p, o, edge_min):
    """Rule identical to `decide()` in src/model_observer.rs: buys the
    side with the highest edge over the ask if it exceeds `edge_min`. Returns
    (side, price, edge) or None."""
    f = o["features"]
    candidates = []
    if f["yes_ask"] is not None:
        candidates.append(("yes", f["yes_ask"], p - f["yes_ask"]))
    if f["no_ask"] is not None:
        candidates.append(("no", f["no_ask"], (1.0 - p) - f["no_ask"]))
    candidates = [c for c in candidates if c[2] > edge_min]
    if not candidates:
        return None
    side, price, edge = max(candidates, key=lambda c: c[2])
    return side, price, edge


def _payoff(side, price, label):
    if side == "yes":
        return label - price
    return (1 - label) - price


def _accumulates(g, y, p_mod, p_base, p_mkt, pnl_mod, pnl_base, pnl_mkt):
    g["y"].append(y)
    g["p_mod"].append(p_mod)
    g["p_base"].append(p_base)
    g["p_mkt"].append(p_mkt)
    if pnl_mod is not None:
        g["pnl_mod"].append(pnl_mod)
    if pnl_base is not None:
        g["pnl_base"].append(pnl_base)
    if pnl_mkt is not None:
        g["pnl_mkt"].append(pnl_mkt)


def _new_group():
    return {
        "y": [], "p_mod": [], "p_base": [], "p_mkt": [],
        "pnl_mod": [], "pnl_base": [], "pnl_mkt": [],
    }


def _group_summary(g, n_total):
    n = len(g["y"])
    cov = len(g["pnl_mod"]) / n if n else 0.0
    return {
        "n_obs": n,
        "decision_coverage": cov,
        "loaded_model": _metrics(g["y"], g["p_mod"]),
        "persistence": _metrics(g["y"], g["p_base"]),
        "market": _metrics(g["y"], g["p_mkt"]),
        "virtual_model_pnl": _pnl_stats(g["pnl_mod"]),
        "virtual_persistence_pnl": _pnl_stats(g["pnl_base"]),
        "virtual_market_pnl": _pnl_stats(g["pnl_mkt"]),
    }


def _decision_row(o, pm, dec, edge_min, metas, model_hash):
    """`decision` row of the shadow with the live shadow's schema:
    the offline path does NOT emit orders — it is a recorded virtual
    decision, same rule as `decide()`."""
    side, price, edge = dec
    meta = metas.get(o["market_cid"]) or {}
    tok = (meta.get("yes_token_id") if side == "yes"
           else meta.get("no_token_id"))
    fee = round(price * (1.0 - price) * FEE_TAKER, 6)
    return {
        "kind": "decision", "schema_version": 1,
        "ts_ms": o["decision_ts_ms"],
        "market_cid": o["market_cid"],
        "slug": meta.get("slug"),
        "token_id": tok,
        "side": f"buy_{side}",
        "ask_price": str(price),
        "shares": "1",
        "fee_est": str(fee),
        "breakeven": str(round(price + fee, 6)),
        "edge": repr(edge),
        "edge_min": str(edge_min),
        "p_model": pm,
        "market_p": (o["features"]["yes_mid"]
                      if o["features"]["yes_mid"] is not None else 0.5),
        "features": o["features"],
        "latency_ms": 0,
        "budget_ms": None,
        "signal_horizon_ms": None,
        "book_recv_ms": o["decision_ts_ms"],
        "model_hash": model_hash,
        "post_close": bool(o["post_close"]),
    }


def _stats_row(ts_ms, n_dec, n_abs, n_without_label, model_hash,
                window_ms=None):
    """`stats` row of the jsonl — declares that the offline path does NOT emit
    orders. `window_ms` is the window [t0,t1] of
    decisions covered by the run (auditable against sources)."""
    return {
        "kind": "stats", "schema_version": 1, "ts_ms": ts_ms,
        "n_decisions": n_dec, "n_abstentions": n_abs,
        "n_decisions_without_label": n_without_label,
        "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "discarded_total": 0,
        "emits_orders": False,
        "via": "offline_tape_without_orders",
        "model_hash": model_hash,
        "window_ms": list(window_ms) if window_ms else None,
    }


def run(events_path, model_path, out_path, edge_min=EDGE_MIN,
        decisions_out=None, decision_min_interval_ms=None):
    model_doc = json.loads(Path(model_path).read_text())
    scaler = dataset.FeatureScaler.from_dict(model_doc["scaler"])
    split_mk = model_doc.get("split_markets")
    if not split_mk:
        raise RuntimeError(
            "model.json without `split_markets`: the artifact does not freeze the "
            "partition — OOS per market cannot be guaranteed.")
    test_set = set(split_mk["test"])
    # Every decision row of the shadow carries the hash of the evaluated
    # model.json FILE — it is the loaded model↔decision chain of custody.
    model_hash = _sha256_file(model_path)
    if decision_min_interval_ms is None:
        decision_min_interval_ms = model_doc.get(
            "decision_min_interval_ms")

    # Train base rate (persistence): from the context if present, otherwise from the
    # artifact itself (train_label_rate saved by train.py).
    ctx_path = Path(model_path).parent / "metrics_context.json"
    if ctx_path.exists():
        base_rate = json.loads(ctx_path.read_text())["base_rate_train"]
    else:
        base_rate = model_doc.get("train_label_rate", 0.5)

    # Streaming passes (labels → metas → observations), pipeline
    # identical to evaluate.py's — the scaler is NOT refitted here.
    labels = dataset.build_labels(
        dataset.iter_events_jsonl(events_path), assume_sorted=True)
    metas = dataset.build_meta_index(
        dataset.iter_events_jsonl(events_path), assume_sorted=True)
    obs = dataset.build_observations_incremental(
        dataset.iter_events_jsonl(events_path), assume_sorted=True,
        decision_min_interval_ms=decision_min_interval_ms)
    obs = dataset.attach_labels(obs, labels)

    # Classification by the artifact's frozen split (B1 closure): unknown
    # or purged markets do NOT fall into test — only those of the frozen
    # set or strictly after the boundary. `post_close`
    # excluded (live, the observer abstains — compare on equal terms).
    parts, cls_meta = dataset.classify_eval_partitions(obs, model_doc)
    # Decisions over the WHOLE pre-close test partition (as live —
    # decide needs no label); metrics only over the labeled ones.
    obs_dec = [o for o in parts["test"] if not o["post_close"]]
    obs_dec.sort(key=lambda o: o["decision_ts_ms"])
    n_post_close = sum(1 for o in parts["test"] if o["post_close"])
    n_out_of_scope = len(cls_meta["out_of_scope_reasons"])

    X = scaler.transform(obs_dec)
    p_model_dec = train_mod.logistic_predict_proba(model_doc["model"], X)

    global_g = _new_group()
    by_market = defaultdict(_new_group)
    by_day = defaultdict(_new_group)
    n_days = set()

    fdec = open(decisions_out, "w", encoding="utf-8") \
        if decisions_out else None
    n_decisions = n_abstentions = n_without_label = 0
    next_stats_ts = None
    ts_min = obs_dec[0]["decision_ts_ms"] if obs_dec else None
    try:
        for o, pm in zip(obs_dec, p_model_dec):
            ts = o["decision_ts_ms"]
            if next_stats_ts is None:
                next_stats_ts = (
                    ts - (ts % STATS_INTERVAL_MS) + STATS_INTERVAL_MS)
            dec_mod = _decision_virtual(pm, o, edge_min)
            if dec_mod is not None:
                n_decisions += 1
                if o["label"] is None:
                    n_without_label += 1
                if fdec is not None:
                    fdec.write(json.dumps(
                        _decision_row(o, pm, dec_mod, edge_min, metas,
                                       model_hash),
                        ensure_ascii=False, separators=(",", ":")) + "\n")
            else:
                n_abstentions += 1
            if fdec is not None:
                while ts >= next_stats_ts:
                    fdec.write(json.dumps(
                        _stats_row(next_stats_ts, n_decisions,
                                    n_abstentions, n_without_label,
                                    model_hash,
                                    window_ms=(ts_min, ts)),
                        ensure_ascii=False, separators=(",", ":")) + "\n")
                    next_stats_ts += STATS_INTERVAL_MS

            if o["label"] is None:
                continue
            y = o["label"]
            f = o["features"]
            p_mkt = f["yes_mid"] if f["yes_mid"] is not None else 0.5

            dec_base = _decision_virtual(base_rate, o, edge_min)
            dec_mkt = _decision_virtual(p_mkt, o, edge_min)
            pnl_mod = (_payoff(dec_mod[0], dec_mod[1], y)
                       if dec_mod else None)
            pnl_base = (_payoff(dec_base[0], dec_base[1], y)
                        if dec_base else None)
            pnl_mkt = (_payoff(dec_mkt[0], dec_mkt[1], y)
                       if dec_mkt else None)

            day = _dt.datetime.fromtimestamp(
                ts / 1000, _dt.timezone.utc
            ).strftime("%Y-%m-%d")
            n_days.add(day)

            _accumulates(global_g, y, pm, base_rate, p_mkt,
                     pnl_mod, pnl_base, pnl_mkt)
            _accumulates(by_market[o["market_cid"]], y, pm, base_rate,
                     p_mkt, pnl_mod, pnl_base, pnl_mkt)
            _accumulates(by_day[day], y, pm, base_rate, p_mkt,
                     pnl_mod, pnl_base, pnl_mkt)
    finally:
        if fdec is not None:
            # Final stats: end of the run — declares that the offline
            # path emits no orders and the window covered.
            last_ts = obs_dec[-1]["decision_ts_ms"] if obs_dec else None
            fdec.write(json.dumps(
                _stats_row(int(_dt.datetime.now(
                    _dt.timezone.utc).timestamp() * 1000),
                    n_decisions, n_abstentions, n_without_label,
                    model_hash,
                    window_ms=(ts_min, last_ts)),
                ensure_ascii=False, separators=(",", ":")) + "\n")
            fdec.close()

    obs_eval = [o for o in obs_dec if o["label"] is not None]
    n_total = len(obs_eval)
    result = {
        "schema_version": 1,
        "edge_min": edge_min,
        "model_hash": model_hash,
        "decisions_out": str(decisions_out) if decisions_out else None,
        "emits_orders": False,
        "n_test_obs_pre_close": n_total,
        "n_test_decision_points": len(obs_dec),
        "n_post_close_excluded": n_post_close,
        "n_decisions": n_decisions,
        "n_abstentions": n_abstentions,
        "n_decisions_without_label": n_without_label,
        "n_test_markets": len(
            test_set & {o["market_cid"] for o in obs_dec}),
        "n_out_of_scope_markets": n_out_of_scope,
        "split_mode": cls_meta["mode"],
        "n_days": len(n_days),
        "global": _group_summary(global_g, n_total),
        "by_market": {
            cid: _group_summary(g, n_total)
            for cid, g in sorted(by_market.items())
        },
        "by_day": {
            d: _group_summary(g, n_total)
            for d, g in sorted(by_day.items())
        },
        "method_warning": (
            "Tape observations are not independent; virtual PnL without "
            "fills or fees (optimistic bound); split by market "
            "frozen in model.json; the decisions jsonl is a "
            "virtual record — it emits NO orders."
        ),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge-min", type=float, default=EDGE_MIN)
    ap.add_argument("--decisions-out", default=None,
                    help="jsonl of shadow decisions (decision rows with "
                         "model_hash=sha256(model.json) + stats that "
                         "declare emits_orders=false)")
    ap.add_argument("--decision-min-interval-ms", type=int, default=None,
                    help="decision time mesh (default: the one in the "
                         "model.json artifact)")
    args = ap.parse_args(argv)
    res = run(args.events, args.model, args.out, args.edge_min,
              decisions_out=args.decisions_out,
              decision_min_interval_ms=args.decision_min_interval_ms)
    g = res["global"]
    print(json.dumps({
        "model_hash": res["model_hash"],
        "n_obs": g["n_obs"],
        "n_decisions": res["n_decisions"],
        "n_abstentions": res["n_abstentions"],
        "coverage": g["decision_coverage"],
        "model_brier": g["loaded_model"].get("brier"),
        "persistence_brier": g["persistence"].get("brier"),
        "market_brier": g["market"].get("brier"),
        "model_pnl": g["virtual_model_pnl"],
        "market_pnl": g["virtual_market_pnl"],
    }, indent=2))


if __name__ == "__main__":
    main()
