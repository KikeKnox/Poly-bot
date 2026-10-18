#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T8b context metrics: per-partition breakdown (post_close, balance
of kinds, feature coverage), metrics over the
pre-close subset and proxy↔official agreement over the dataset's markets.

Writes `metrics_context.json` next to the loaded model. Usage:

    python3 research/numeric_model/analyze_t8b.py \
        --events research/data/t8b_confirmed/events.jsonl \
        --model research/data/t8b_confirmed/model/model.json \
        --state-db /path/hrf_state_paper.sqlite   # optional
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import train as train_mod  # noqa: E402


def _metrics(sub, scaler, model, base):
    X = scaler.transform(sub)
    y = [o["label"] for o in sub]
    pm = train_mod.logistic_predict_proba(model, X)
    pmkt = [
        o["features"]["yes_mid"]
        if o["features"]["yes_mid"] is not None else 0.5
        for o in sub
    ]
    return {
        "n": len(sub),
        "loaded_model": train_mod.classification_metrics(y, pm),
        "persistence": train_mod.classification_metrics(y, [base] * len(y)),
        "market": train_mod.classification_metrics(y, pmkt),
    }


def run(events_path, model_path, state_db=None, streaming=True):
    md = json.loads(Path(model_path).read_text())
    scaler = dataset.FeatureScaler.from_dict(md["scaler"])

    if streaming:
        labels = dataset.build_labels(
            dataset.iter_events_jsonl(events_path), assume_sorted=True)
        try:
            obs = dataset.build_observations_incremental(
                dataset.iter_events_jsonl(events_path),
                assume_sorted=True)
        except dataset.NonMonotonicTape as e:
            # No fallback possible over an already consumed generator.
            raise RuntimeError(
                "Non-monotonic tape in avail_ts_ms: streaming cannot "
                "go back nor fall back to the batch builder. Re-export "
                "sorted by avail or run with --no-streaming. "
                f"Detail: {e}"
            ) from e
        obs = dataset.attach_labels(obs, labels)
    else:
        events = dataset.load_events_jsonl(events_path)
        obs = dataset.attach_labels(
            dataset.build_observations_auto(events),
            dataset.build_labels(events))
    obs = [o for o in obs if o["label"] is not None]

    # Classification by the artifact's frozen split (B1 closure): unknown
    # markets do not automatically fall into test — they stay
    # out_of_scope unless strictly after the boundary.
    parts, cls_meta = dataset.classify_eval_partitions(obs, md)
    train = parts["train"]
    test = parts["test"]
    # Persistence: base rate persisted in the artifact (never recomputed
    # over the eval population); legacy fallback only if the field is missing.
    if "train_label_rate" in md:
        base = md["train_label_rate"]
        base_src = "model.json"
    else:
        base = (sum(o["label"] for o in train) / len(train)) if train else 0.5
        base_src = "recomputed_legacy"

    res = {
        "test_all": _metrics(test, scaler, md["model"], base),
        "test_only_pre_close": _metrics(
            [o for o in test if not o["post_close"]], scaler,
            md["model"], base),
        "base_rate_train": base,
        "base_rate_source": base_src,
        "split_mode": cls_meta["mode"],
        "out_of_scope": {
            "markets": sorted(cls_meta["out_of_scope_reasons"]),
            "reasons": cls_meta["out_of_scope_reasons"],
            "n_obs": len(parts["out_of_scope"]),
        },
        "n_markets": {
            "train": len({o["market_cid"] for o in train}),
            "test": len({o["market_cid"] for o in test}),
        },
        "partition": {},
    }
    for name_str, sub in (("train", train), ("test", test)):
        n = len(sub)
        res["partition"][name_str] = {
            "n": n,
            "post_close": sum(1 for o in sub if o["post_close"]),
            # None (not NaN) when the partition is empty.
            "label_yes": (
                sum(o["label"] for o in sub) / n if n else None),
            "cov_yes_mid": (sum(
                1 for o in sub
                if o["features"]["yes_mid"] is not None) / n
                if n else None),
            "cov_spot": (sum(
                1 for o in sub
                if o["features"]["spot"] is not None) / n
                if n else None),
            "cov_moneyness": (sum(
                1 for o in sub
                if o["features"]["moneyness"] is not None) / n
                if n else None),
        }

    # Proxy ↔ official label agreement over the dataset's markets.
    if state_db:
        s = sqlite3.connect(
            Path(state_db).resolve().as_uri() + "?mode=ro", uri=True)
        s.execute("PRAGMA query_only=ON")
        prov = {}
        for cid, v in s.execute(
            "SELECT market_cid, verdict FROM post_close_observations "
            "WHERE verdict IN ('yes','no') ORDER BY ts_ms, rowid"
        ):
            prov[cid] = v
        lab_mkt = {}
        for o in obs:
            lab_mkt[o["market_cid"]] = "yes" if o["label"] == 1 else "no"
        agree = dis = 0
        for cid, w in lab_mkt.items():
            if cid in prov:
                if prov[cid] == w:
                    agree += 1
                else:
                    dis += 1
        res["proxy_vs_official_in_dataset"] = {
            "agree": agree, "diverge": dis}

    out = Path(model_path).parent / "metrics_context.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--state-db", default=None)
    ap.add_argument("--no-streaming", action="store_true")
    args = ap.parse_args(argv)
    res = run(args.events, args.model, state_db=args.state_db,
              streaming=not args.no_streaming)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
