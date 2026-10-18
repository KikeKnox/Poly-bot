"""Evaluation of the trained baseline on the TEST partition.

Rebuilds the observations with the SAME causal pipeline of
`dataset.py`, applies the scaler saved in `model.json` (fitted only on
train — never refitted here) and reports the loaded model's metrics against two
reference baselines:

- `persistence`: predict the base YES rate learned on train.
- `market`: use `yes_mid` unscaled as the probability — it compares the
  loaded model with the market's implied price, never trains on it as
  truth (T8 brief step 3).

Writes `metrics_eval.json` and updates the directory's manifest.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import train as train_mod  # noqa: E402

SCHEMA_VERSION = 1


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(events_path, model_path, out_dir=None, streaming=False,
        decision_min_interval_ms=None):
    model_doc = json.loads(Path(model_path).read_text())
    scaler = dataset.FeatureScaler.from_dict(model_doc["scaler"])

    # Artifact decision mesh: if the loaded model was trained with
    # `decision_min_interval_ms`, the evaluation uses the SAME mesh to
    # evaluate the same decision points (dataset_tape/train parity).
    if decision_min_interval_ms is None:
        decision_min_interval_ms = model_doc.get(
            "decision_min_interval_ms")
    if streaming:
        # T8b: two streaming passes (labels → observations) without
        # materializing the tape — for GB-sized datasets.
        labels = dataset.build_labels(
            dataset.iter_events_jsonl(events_path), assume_sorted=True)
        try:
            obs = dataset.build_observations_incremental(
                dataset.iter_events_jsonl(events_path),
                assume_sorted=True,
                decision_min_interval_ms=decision_min_interval_ms)
        except dataset.NonMonotonicTape as e:
            # No fallback possible over an already consumed generator —
            # actionable message, not a raw stack trace.
            raise RuntimeError(
                "Non-monotonic tape in avail_ts_ms: --streaming cannot "
                "go back nor fall back to the batch builder. Re-export "
                "sorted by avail or run without --streaming (exact "
                f"O(n^2) fallback). Detail: {e}"
            ) from e
        obs = dataset.attach_labels(obs, labels)
    else:
        events = dataset.load_events_jsonl(events_path)
        if decision_min_interval_ms is None:
            obs = dataset.attach_labels(
                dataset.build_observations_auto(events),
                dataset.build_labels(events))
        else:
            obs = dataset.attach_labels(
                dataset.build_observations_incremental(
                    events,
                    decision_min_interval_ms=decision_min_interval_ms),
                dataset.build_labels(events))
    obs = [o for o in obs if o["label"] is not None]

    # The artifact's task rules: if the loaded model was trained for the
    # pre-close regime, the evaluation excludes the same observations.
    include_post_close = bool(model_doc.get("include_post_close", True))
    n_post_close = sum(1 for o in obs if o["post_close"])
    if not include_post_close:
        obs = [o for o in obs if not o["post_close"]]

    # The split is fixed by training (model.json), not re-derived over
    # the evaluation data. With declared `boundaries`, an unknown
    # market does NOT automatically fall into test: only those strictly
    # after the test boundary; the rest → out_of_scope.
    parts, cls_meta = dataset.classify_eval_partitions(obs, model_doc)
    train_obs = parts["train"]
    test_obs = parts["test"]
    oos_obs = parts["out_of_scope"]
    splits = model_doc.get("splits", [])

    if not test_obs:
        # The frozen split assigned no market of this tape to
        # test (purge of markets that cross the boundary): OOS
        # evaluation is impossible and is DECLARED insufficient — metrics are never fabricated
        # over an empty partition nor is the circuit aborted.
        metrics = {
            "n_test": 0,
            "n_post_close_excluded": (
                0 if include_post_close else n_post_close),
            "n_out_of_scope": len(oos_obs),
            "n_out_of_scope_markets": len(cls_meta["out_of_scope_reasons"]),
            "out_of_scope_markets": sorted(cls_meta["out_of_scope_reasons"]),
            "out_of_scope_reasons": cls_meta["out_of_scope_reasons"],
            "split_mode": cls_meta["mode"],
            "oos_evaluation": "insufficient",
            "reason": (
                "empty test partition after purge/embargo of the frozen "
                "split — the capture contributes no labeled markets "
                "after the boundary; the circuit is technically demonstrated "
                "without statistical OOS evidence"),
            # Fit regime declared by the artifact (retrospective |
            # point_in_time) — never relabeled here.
            "eval_mode": model_doc.get("eval_mode"),
            "oos_operable": model_doc.get("oos_operable", False),
            "requires_future": model_doc.get("requires_future"),
            "accuracy": None, "logloss": None, "brier": None,
            "persistence_brier": None, "market_brier": None,
        }
        out = Path(out_dir or Path(model_path).parent)
        out.mkdir(parents=True, exist_ok=True)
        met_path = out / "metrics_eval.json"
        met_path.write_text(json.dumps(metrics, indent=2,
                                       ensure_ascii=False))
        mpath = out / "manifest.json"
        if mpath.exists():
            manifest = json.loads(mpath.read_text())
            entry = {"path": met_path.name,
                     "sha256": _sha256_file(met_path)}
            manifest["files"] = [
                f for f in manifest.get("files", [])
                if f["path"] != entry["path"]
            ] + [entry]
            mpath.write_text(json.dumps(manifest, indent=2,
                                        ensure_ascii=False))
        return {"metrics": metrics, "splits": splits,
                "test_markets": []}

    Xte = scaler.transform(test_obs)
    yte = [o["label"] for o in test_obs]
    p_model = train_mod.logistic_predict_proba(model_doc["model"], Xte)

    # Persistence baseline: train base rate PERSISTED in the artifact
    # (`train_label_rate`) — never recomputed over the evaluation
    # population. Old artifacts without the field fall back to computing over
    # the declared train markets, marked as a legacy fallback.
    if "train_label_rate" in model_doc:
        base_rate = model_doc["train_label_rate"]
        base_src = "model.json"
    else:
        base_rate = (
            sum(o["label"] for o in train_obs) / len(train_obs)
            if train_obs else 0.5
        )
        base_src = "recomputed_legacy"
    p_persist = [base_rate] * len(yte)

    # Market baseline: raw yes_mid as the implied probability.
    p_mkt = [
        o["features"]["yes_mid"] if o["features"]["yes_mid"] is not None else 0.5
        for o in test_obs
    ]

    metrics = {
        "n_test": len(test_obs),
        "n_post_close_excluded": (
            0 if include_post_close else n_post_close),
        "n_out_of_scope": len(oos_obs),
        "n_out_of_scope_markets": len(cls_meta["out_of_scope_reasons"]),
        "out_of_scope_markets": sorted(cls_meta["out_of_scope_reasons"]),
        "out_of_scope_reasons": cls_meta["out_of_scope_reasons"],
        "split_mode": cls_meta["mode"],
        "eval_mode": model_doc.get("eval_mode"),
        "oos_operable": model_doc.get("oos_operable", False),
        "requires_future": model_doc.get("requires_future"),
        "base_rate": base_rate,
        "base_rate_source": base_src,
        "accuracy": train_mod.classification_metrics(yte, p_model)["accuracy"],
        "logloss": train_mod.classification_metrics(yte, p_model)["logloss"],
        "brier": train_mod.classification_metrics(yte, p_model)["brier"],
        "persistence_brier": train_mod.classification_metrics(
            yte, p_persist)["brier"],
        "market_brier": train_mod.classification_metrics(
            yte, p_mkt)["brier"],
    }

    out = Path(out_dir or Path(model_path).parent)
    out.mkdir(parents=True, exist_ok=True)
    met_path = out / "metrics_eval.json"
    met_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Updates the directory's manifest if it exists (adds the file path of the
    # evaluation metrics with its hash — contract E).
    mpath = out / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        entry = {"path": met_path.name, "sha256": _sha256_file(met_path)}
        manifest["files"] = [
            f for f in manifest.get("files", []) if f["path"] != entry["path"]
        ] + [entry]
        # `created_at_ms` is NOT touched: the manifest keeps the experiment's
        # creation instant, not that of the last evaluation.
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return {"metrics": metrics, "splits": splits,
            "test_markets": sorted({o["market_cid"] for o in test_obs})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True,
                    help="ResearchEvent JSONL tape")
    ap.add_argument("--model", required=True, help="model.json from train.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--decision-min-interval-ms", type=int, default=None,
                    help="decision time mesh (default: the one in the "
                         "model.json artifact)")
    args = ap.parse_args(argv)
    res = run(args.events, args.model, args.out, streaming=args.streaming,
              decision_min_interval_ms=args.decision_min_interval_ms)
    print(json.dumps(res["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
