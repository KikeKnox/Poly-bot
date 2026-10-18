"""Stdlib baseline (batch gradient-descent logistic regression)
trained over a `ResearchEvent` JSONL tape or over a synthetic
demonstration tape (`--demo`).

Full pipeline:
  events → causal observations → labels → chronological split by
  market → scaler fitted ONLY on train → deterministic logistic GD →
  metrics + `model.json` + `manifest.json` artifact, with the
  `ResearchManifest` schema from the contract (everything pinned by hash).

On fixtures the numeric result is not the target: what matters is that the
pipeline runs and produces reproducible artifacts.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402

SCHEMA_VERSION = 1
_EPS = 1e-15


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def code_hash():
    """Hash of the package code tree (its own .py files, sorted)."""
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for p in sorted(here.glob("*.py")) + sorted((here / "tests").glob("*.py")):
        h.update(p.name.encode())
        h.update(_sha256_file(p).encode())
    return h.hexdigest()


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logistic_train_stdlib(X, y, epochs, lr, l2):
    """Batch logistic regression, pure stdlib, deterministic (init 0)."""
    n = len(X)
    d = len(X[0]) if n else 0
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            err = _sigmoid(z) - yi
            for j in range(d):
                gw[j] += err * xi[j]
            gb += err
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return {"weights": w, "bias": b}


def logistic_train(X, y, epochs=200, lr=0.1, l2=1e-4):
    """Batch logistic GD. Uses vectorized numpy when it is installed
    (same math, init 0, deterministic); the stdlib fallback keeps the
    zero-dependency scaffold — on real datasets of ~10⁶ observations the
    Python loop is prohibitive (~min/epoch)."""
    try:
        import numpy as np
    except ImportError:
        return _logistic_train_stdlib(X, y, epochs, lr, l2)
    if not X:
        return {"weights": [0.0] * len(dataset.FEATURE_NAMES), "bias": 0.0}
    A = np.asarray(X, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    w = np.zeros(A.shape[1])
    b = 0.0
    n = len(A)
    for _ in range(epochs):
        z = np.clip(A @ w + b, -500, 500)
        err = (1.0 / (1.0 + np.exp(-z))) - t
        w -= lr * (A.T @ err / n + l2 * w)
        b -= lr * float(err.mean())
    return {"weights": w.tolist(), "bias": float(b)}


def logistic_predict_proba(model, X):
    w, b = model["weights"], model["bias"]
    return [_sigmoid(b + sum(wj * xj for wj, xj in zip(w, xi))) for xi in X]


def classification_metrics(y_true, p_hat):
    """Accuracy, logloss and Brier — honest metrics over probabilities."""
    n = len(y_true)
    if n == 0:
        return {"accuracy": None, "logloss": None, "brier": None, "n": 0}
    acc = sum(1 for yt, p in zip(y_true, p_hat) if (p >= 0.5) == bool(yt)) / n
    ll = -sum(
        yt * math.log(min(max(p, _EPS), 1 - _EPS))
        + (1 - yt) * math.log(min(max(1 - p, _EPS), 1 - _EPS))
        for yt, p in zip(y_true, p_hat)
    ) / n
    brier = sum((p - yt) ** 2 for yt, p in zip(y_true, p_hat)) / n
    return {"accuracy": acc, "logloss": ll, "brier": brier, "n": n}


def _build_supervised(events, decision_min_interval_ms=None):
    # Incremental O(n) builder with an exact batch fallback when the tape
    # is not monotonic in avail — required for large real tapes.
    if decision_min_interval_ms is None:
        obs = dataset.build_observations_auto(events)
    else:
        # A decision grid only applies to the incremental builder (the
        # reference batch builder does not implement a grid — it is used without one).
        obs = dataset.build_observations_incremental(
            events, decision_min_interval_ms=decision_min_interval_ms)
    labels = dataset.build_labels(events)
    obs = dataset.attach_labels(obs, labels)
    return [o for o in obs if o["label"] is not None]


def _build_supervised_streaming(events_path, decision_min_interval_ms=None):
    """Two streaming passes over the tape (labels, then observations)
    without materializing the events — for tapes of several GBs.
    The tape must be sorted by local_seq (the exporter guarantees it)."""
    labels = dataset.build_labels(
        dataset.iter_events_jsonl(events_path), assume_sorted=True)
    try:
        obs = dataset.build_observations_incremental(
            dataset.iter_events_jsonl(events_path), assume_sorted=True,
            decision_min_interval_ms=decision_min_interval_ms)
    except dataset.NonMonotonicTape as e:
        # No fallback is possible: the batch builder would need to re-traverse a tape
        # already consumed by the generator. Actionable message, not a raw stack trace.
        raise RuntimeError(
            "Tape not monotonic in avail_ts_ms: --streaming cannot "
            "rewind or fall back to the batch builder (the tape is already "
            "consumed). Re-export it sorted by avail or run without "
            f"--streaming (exact O(n^2) fallback). Detail: {e}"
        ) from e
    obs = dataset.attach_labels(obs, labels)
    return [o for o in obs if o["label"] is not None]


def _demo_events():
    """Self-contained synthetic tape for `--demo` (6 markets).

    Markets last 60 s and start every 120 s: the gap exceeds the default
    embargo (FLOW_WINDOW_MS = 60 s) so the certified temporal split yields
    a real train and test set with both outcomes (markets that cross the
    boundary are purged — that is the intended semantics).
    Labels become known ~10 s after each market starts, before the train
    boundary, to respect the availability rule."""
    t0 = 1_700_000_000_000
    events = []
    seq = 0

    def meta(cid, off_ms, ref, winner):
        nonlocal seq
        open_s = (t0 + off_ms) // 1000
        events.append({
            "schema_version": 1, "source": "polymarket_rest",
            "kind": "market_meta", "market_cid": cid, "token_id": None,
            "asset": "BTC",
            "payload": {
                "schema_version": 1, "market_cid": cid,
                "slug": f"btc-updown-{cid}", "yes_token_id": f"{cid}-yes",
                "no_token_id": f"{cid}-no", "asset": "BTC", "neg_risk": False,
                "market_open_ts": open_s, "clear_at": open_s + 60,
                "reference_price": ref, "rule": "up_down_reference",
                "resolution_source": "binance_ws",
                "resolution_source_ref": "BTCUSDT spot",
                "rules_windows": [],
                "outcome": {"winner": winner,
                            "known_at_ms": t0 + off_ms + 10_000,
                            "provenance": "gamma:demo"},
            },
            "origin_ts_ms": None, "recv_ts_ms": t0 + off_ms - 20,
            "avail_ts_ms": t0 + off_ms, "local_seq": seq, "quality": 8,
        })
        seq += 1

    def book(cid, tok, off_ms, bid, ask):
        nonlocal seq
        events.append({
            "schema_version": 1, "source": "polymarket_ws",
            "kind": "book_snapshot", "market_cid": cid, "token_id": tok,
            "asset": None,
            "payload": {"bids": [[str(bid), "100"]],
                        "asks": [[str(ask), "80"]]},
            "origin_ts_ms": t0 + off_ms - 40, "recv_ts_ms": t0 + off_ms - 10,
            "avail_ts_ms": t0 + off_ms, "local_seq": seq, "quality": 0,
        })
        seq += 1

    def tick(off_ms, price):
        nonlocal seq
        events.append({
            "schema_version": 1, "source": "binance_ws", "kind": "price_tick",
            "market_cid": None, "token_id": None, "asset": "BTC",
            "payload": {"price": str(price)},
            "origin_ts_ms": t0 + off_ms - 30, "recv_ts_ms": t0 + off_ms - 5,
            "avail_ts_ms": t0 + off_ms, "local_seq": seq, "quality": 0,
        })
        seq += 1

    # Alternating winners so that train and test both contain both outcomes.
    for i, (cid, base, winner) in enumerate(
        [("demo-m1", 0.45, "yes"), ("demo-m2", 0.55, "no"),
         ("demo-m3", 0.35, "yes"), ("demo-m4", 0.60, "no"),
         ("demo-m5", 0.42, "yes"), ("demo-m6", 0.58, "no")]
    ):
        off = i * 120_000
        meta(cid, off + 100, "67000.0", winner)
        for step in range(4):
            book(cid, f"{cid}-yes", off + 1_000 + step * 1_000,
                 base + step * 0.01, base + 0.02 + step * 0.01)
            book(cid, f"{cid}-no", off + 1_050 + step * 1_000,
                 1 - base - 0.03 - step * 0.01, 1 - base - 0.01 - step * 0.01)
            tick(off + 1_100 + step * 1_000, 67000.0 + step * 5 + i * 3)
    return events


def _write_manifest(out_dir, research_dir, splits, seeds, extra_files,
                    config=None):
    files = []
    for p in extra_files:
        p = Path(p)
        files.append({
            "path": str(p.relative_to(research_dir))
            if str(p).startswith(str(research_dir)) else p.name,
            "sha256": _sha256_file(p),
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_ms": int(time.time() * 1000),
        "code_hash": code_hash(),
        # Hash of the ACTUAL hyperparameter dict — never a hardcoded literal:
        # the manifest is bound to the concrete run.
        "config_hash": _sha256_bytes(json.dumps(
            config or {}, sort_keys=True).encode()),
        "config": config,
        "model_hash": None,
        "sim_profile_hash": None,
        "seeds": list(seeds),
        "splits": splits,
        "research_dir": str(research_dir),
        "files": files,
    }
    mpath = Path(out_dir) / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def _coverage_fields(observations):
    """Coverage of the minimum fields for the pre-close task: fraction of
    observations with a determinable `post_close` (metadata with `clear_at` —
    `time_left_ms` not None) and with `decision_ts_ms`/`market_cid` present."""
    n = len(observations)
    if not n:
        return {"n": 0, "post_close_determinable": 0.0,
                "with_market_cid": 0.0, "with_decision_ts": 0.0}
    det = sum(1 for o in observations
              if o["features"].get("time_left_ms") is not None)
    cid = sum(1 for o in observations if o.get("market_cid"))
    dts = sum(1 for o in observations
              if o.get("decision_ts_ms") is not None)
    return {"n": n, "post_close_determinable": det / n,
            "with_market_cid": cid / n, "with_decision_ts": dts / n}


# Target thresholds for OOS sufficiency: the evaluation is declared
# `insufficient` unless ALL of them are met AND the operator explicitly
# asserts it (`--declare-sufficient`). A ~25 min capture window never
# reaches them — the pipeline is demonstrated without faking statistical
# evidence.
SUFFICIENCY_MIN_TEST_OBS = 10_000
SUFFICIENCY_MIN_TEST_MARKETS = 30
SUFFICIENCY_MIN_SPAN_MS = 3_600_000  # minimum 1 h of OOS test span


def _evaluate_sufficiency(test_obs, declare_sufficient=False):
    """Honest sufficiency verdict of the OOS evaluation.

    `target_verdict` comes ONLY from the thresholds; `oos_evidence` is
    `sufficient` only if the target passes AND the operator declares it
    with `--declare-sufficient` — never a hidden pass."""
    n_obs = len(test_obs)
    n_mk = len({o["market_cid"] for o in test_obs})
    ts = [o["decision_ts_ms"] for o in test_obs]
    span = (max(ts) - min(ts)) if ts else 0
    reasons = []
    if n_obs < SUFFICIENCY_MIN_TEST_OBS:
        reasons.append(
            f"n_test_obs {n_obs} < {SUFFICIENCY_MIN_TEST_OBS}")
    if n_mk < SUFFICIENCY_MIN_TEST_MARKETS:
        reasons.append(
            f"n_test_markets {n_mk} < {SUFFICIENCY_MIN_TEST_MARKETS}")
    if span < SUFFICIENCY_MIN_SPAN_MS:
        reasons.append(
            f"span_test_ms {span} < {SUFFICIENCY_MIN_SPAN_MS}")
    target = "sufficient" if not reasons else "insufficient"
    evidence = ("sufficient"
                 if target == "sufficient" and declare_sufficient
                 else "insufficient")
    if target == "sufficient" and not declare_sufficient:
        reasons.append(
            "thresholds met but without --declare-sufficient: "
            "insufficient is kept (no hidden pass)")
    return {
        "oos_evidence": evidence,
        "target_verdict": target,
        "declared_by_operator": bool(declare_sufficient),
        "reasons": reasons,
        "n_test_obs": n_obs, "n_test_markets": n_mk,
        "span_test_ms": span,
        "thresholds": {
            "min_test_obs": SUFFICIENCY_MIN_TEST_OBS,
            "min_test_markets": SUFFICIENCY_MIN_TEST_MARKETS,
            "min_span_ms": SUFFICIENCY_MIN_SPAN_MS,
        },
    }


def run(events_path=None, out_dir=None, epochs=200, lr=0.1, seed=0,
        train_frac=0.6, val_frac=0.0, train_end_ms=None, val_end_ms=None,
        embargo_ms=None, include_post_close=False,
        min_field_coverage=0.5, demo=False, streaming=False,
        decision_min_interval_ms=None, dataset_manifest_path=None,
        declare_sufficient=False, cutoff_fit_ms=None,
        eval_mode="retrospective"):
    """Trains the baseline and writes the artifacts. Returns a dict with metrics.

    Certified temporal split: explicit global UTC boundaries
    (`--train-end-ms`/`--val-end-ms`) or derived as quantiles of
    `decision_ts_ms` via `train_frac`/`val_frac`; markets crossing a
    boundary are purged whole; embargo between partitions
    (`EMBARGO_DEFAULT_MS = FLOW_WINDOW_MS`); labels are trainable only if
    `known_at_ms <= cutoff` of their partition. The scaler, imputation and
    `train_label_rate` are fitted ONLY on usable train data.
    """
    out = Path(out_dir or (Path(__file__).parent / "artifacts"))
    out.mkdir(parents=True, exist_ok=True)

    if demo:
        obs = _build_supervised(_demo_events())
    elif streaming:
        obs = _build_supervised_streaming(
            events_path, decision_min_interval_ms=decision_min_interval_ms)
    else:
        obs = _build_supervised(
            dataset.load_events_jsonl(events_path),
            decision_min_interval_ms=decision_min_interval_ms)
    if not obs:
        raise RuntimeError(
            "empty dataset: no observation with a confirmed label — "
            "check the tape (resolutions/outcomes) before training")

    # Minimum field coverage: without `clear_at` it is impossible to know whether a
    # decision is post-close — requiring it prevents silent leakage.
    coverage = _coverage_fields(obs)
    if coverage["post_close_determinable"] < min_field_coverage:
        raise RuntimeError(
            "insufficient coverage for the pre-close task: only "
            f"{coverage['post_close_determinable']:.1%} of observations "
            f"have a determinable `post_close` (minimum "
            f"{min_field_coverage:.0%}) — missing metadata with clear_at")

    # `post_close` is a declarative mark: for the pre-close task these are
    # filtered out EXPLICITLY here and counted in the manifest.
    n_post_close = sum(1 for o in obs if o["post_close"])
    if not include_post_close:
        obs = [o for o in obs if not o["post_close"]]
    if not obs:
        raise RuntimeError(
            "empty dataset after excluding post-close observations — "
            "the tape has no labeled pre-close decisions")

    # Boundaries: explicit, or temporal quantiles of decision_ts_ms.
    if train_end_ms is None:
        train_end_ms, val_end_der = dataset.boundaries_from_fractions(
            obs, train_frac=train_frac, val_frac=val_frac)
        if val_end_ms is None:
            val_end_ms = val_end_der
    elif val_end_ms is None and val_frac > 0:
        # Explicit train boundary + validation fraction: val_end is the
        # `val_frac` quantile of the time mass AFTER the embargo.
        embargo_eff = (dataset.EMBARGO_DEFAULT_MS if embargo_ms is None
                       else embargo_ms)
        post = [o for o in obs
                if o["decision_ts_ms"] > train_end_ms + embargo_eff]
        if post:
            val_end_ms = dataset.boundary_from_fraction(post, val_frac)
            if val_end_ms <= train_end_ms:
                val_end_ms = None

    partitions, split_meta = dataset.temporal_split(
        obs, train_end_ms=train_end_ms, val_end_ms=val_end_ms,
        embargo_ms=embargo_ms)
    embargo_ms_eff = split_meta["boundaries"]["embargo_ms"]

    # Label availability at the FIT instant and the evaluation regime
    # (review fix): `max(known_at)` over the whole dataset does NOT guarantee
    # a fit earlier than the test decisions.
    #
    # - "retrospective" (default): retrospective holdout after capture.
    #   The fit happens now with all labels already known; the default cutoff
    #   is the last `known_at` of the fitting partitions
    #   (train/validation — the test one does not configure the fit). If the cutoff
    #   falls at or after the test start the model needed future data →
    #   `oos_operable=False` is declared: valid as a retrospective, never
    #   as an operable OOS.
    # - "point_in_time": simulates a real fit BEFORE the OOS window. Without an
    #   explicit cutoff it is set to `first test decision − 1ms`; with an
    #   explicit cutoff it must be strictly < test start. Train and
    #   validation are filtered by `label_known_at_ms <= cutoff`: a train
    #   label published after the test start is excluded.
    test_ts = [o["decision_ts_ms"] for o in partitions["test"]
               if o.get("decision_ts_ms") is not None]
    test_start_ms = min(test_ts) if test_ts else None
    if eval_mode not in ("retrospective", "point_in_time"):
        raise RuntimeError(f"unknown eval_mode: {eval_mode!r}")
    if cutoff_fit_ms is None:
        if eval_mode == "point_in_time" and test_start_ms is not None:
            cutoff_fit_ms = test_start_ms - 1
        else:
            knowns_fit = [o["label_known_at_ms"]
                          for o in (partitions["train"]
                                    + partitions["validation"])
                          if o.get("label_known_at_ms") is not None]
            cutoff_fit_ms = (max(knowns_fit) if knowns_fit
                             else train_end_ms)
    elif (eval_mode == "point_in_time" and test_start_ms is not None
          and cutoff_fit_ms >= test_start_ms):
        raise RuntimeError(
            "cutoff_fit_ms is not earlier than the test start "
            f"({cutoff_fit_ms} >= {test_start_ms}): in point-in-time "
            "evaluation the fit must precede the OOS decisions — "
            "use an earlier --cutoff-fit-ms or --eval-mode retrospective")
    train_obs, excl_label_train = dataset.filter_labels_available(
        partitions["train"], cutoff_ms=cutoff_fit_ms)
    val_obs, excl_label_val = dataset.filter_labels_available(
        partitions["validation"], cutoff_ms=cutoff_fit_ms) \
        if val_end_ms is not None else ([], [])
    test_obs = partitions["test"]  # retrospective eval: late labels are OK

    # Honest declaration of the regime: only a point-in-time fit strictly
    # before the first OOS decision may be called operable — and even then
    # sufficiency is decided by `_evaluate_sufficiency` (thresholds + operator
    # declaration).
    requires_future = (test_start_ms is not None
                       and cutoff_fit_ms >= test_start_ms)
    oos_operable = (eval_mode == "point_in_time"
                    and test_start_ms is not None
                    and not requires_future)

    if not train_obs:
        raise RuntimeError(
            "empty train set after purging markets that cross the boundary and "
            "excluding labels without availability — widen the train "
            "window (--train-end-ms/--train-frac) or add more data")
    kinds = {o["label"] for o in train_obs}
    if len(kinds) < 2:
        raise RuntimeError(
            "degenerate dataset: train has a single class "
            f"({sorted(kinds)}) after purging/filtering — a model with "
            "zero weights or base-rate only is NOT saved as a success")
    # Empty test set (purge/embargo): does NOT abort — the model is produced as
    # technical evidence of the pipeline and the OOS evaluation is declared
    # insufficient via `_evaluate_sufficiency` (n_test=0 < thresholds).
    # Test metrics are never fabricated over an empty partition.
    test_empty = not test_obs

    scaler = dataset.FeatureScaler().fit(train_obs)
    Xtr = scaler.transform(train_obs)
    ytr = [o["label"] for o in train_obs]
    base_rate = sum(ytr) / len(ytr)
    Xte = scaler.transform(test_obs) if test_obs else []
    yte = [o["label"] for o in test_obs]
    Xva = scaler.transform(val_obs)
    yva = [o["label"] for o in val_obs]

    model = logistic_train(Xtr, ytr, epochs=epochs, lr=lr)

    metrics = {
        "n_train": len(train_obs), "n_validation": len(val_obs),
        "n_test": len(test_obs),
        "n_post_close_excluded": n_post_close,
        "n_late_label_excluded": (
            len(excl_label_train) + len(excl_label_val)),
        "cutoff_labels_ms": cutoff_fit_ms,
        "train": classification_metrics(
            ytr, logistic_predict_proba(model, Xtr)),
        "validation": classification_metrics(
            yva, logistic_predict_proba(model, Xva)),
        "test": (None if test_empty else classification_metrics(
            yte, logistic_predict_proba(model, Xte))),
        "test_empty": test_empty,
        "eval_mode": eval_mode,
        "test_start_ms": test_start_ms,
        "oos_operable": oos_operable,
        "requires_future": requires_future,
    }

    # OOS sufficiency verdict: computed over the ACTUAL test partition of the
    # split — it is never inflated to show green.
    sufficiency = _evaluate_sufficiency(
        test_obs, declare_sufficient=declare_sufficient)
    if requires_future:
        # The fit used labels published after the test start: the
        # evaluation is retrospective, NOT a point-in-time operable OOS.
        sufficiency["reasons"].append(
            f"fit with cutoff {cutoff_fit_ms} >= test start "
            f"{test_start_ms}: the model needed future labels — "
            "retrospective evaluation, not an operable OOS")
        sufficiency["oos_evidence"] = "insufficient"
    metrics["oos_evidence"] = sufficiency["oos_evidence"]
    metrics["sufficiency"] = sufficiency

    purged_cids = [p["market_cid"] for p in split_meta["purged_markets"]]
    exclusions = {
        "post_close": {"observations": n_post_close,
                       "reason": "pre_close_task"},
        "label_not_known_at_cutoff": {
            "train": len(excl_label_train),
            "validation": len(excl_label_val),
        },
        "embargo": {
            "markets": split_meta["embargo_markets"],
            "observations": split_meta["n_embargo_obs"],
        },
        "purged_markets": split_meta["purged_markets"],
        "coverage": coverage,
    }

    model_doc = {
        "schema_version": SCHEMA_VERSION,
        "kind": "logistic_stdlib_baseline",
        "feature_names": dataset.FEATURE_NAMES,
        "model": model,
        "scaler": scaler.to_dict(),
        "train_frac": train_frac, "val_frac": val_frac,
        "epochs": epochs, "lr": lr, "seed": seed,
        # Actual temporal boundaries of the fit — evaluate.py uses them to
        # classify unknown markets (out_of_scope); it never re-derives them.
        "boundaries": split_meta["boundaries"],
        "splits": split_meta["splits"],
        # Persisted train base rate: evaluate/shadow read it from the
        # artifact — they never recompute it over another population.
        "train_label_rate": base_rate,
        "include_post_close": include_post_close,
        # Decision grid used in the fit — evaluate/shadow inherit it from the
        # artifact so they evaluate the same decision points.
        "decision_min_interval_ms": decision_min_interval_ms,
        "n_post_close_excluded": n_post_close,
        # Fit regime (review fix): retrospective |
        # point_in_time, actual cutoff applied and a declaration of whether the
        # model may be called an operable OOS (never if it needed future data).
        "eval_mode": eval_mode,
        "cutoff_fit_ms": cutoff_fit_ms,
        "test_start_ms": test_start_ms,
        "oos_operable": oos_operable,
        "requires_future": requires_future,
        # Market assignment fixed at training time: evaluate.py uses it as-is —
        # it never re-derives the split over the evaluation data.
        "split_markets": {
            "train": sorted({o["market_cid"] for o in partitions["train"]}),
            "validation": sorted(
                {o["market_cid"] for o in partitions["validation"]}),
            "test": sorted({o["market_cid"] for o in test_obs}),
            "purged": purged_cids,
        },
        "exclusions": exclusions,
    }
    mpath = out / "model.json"
    mpath.write_text(json.dumps(model_doc, indent=2, ensure_ascii=False))

    met_path = out / "metrics_train.json"
    met_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    config = {
        "train_frac": train_frac, "val_frac": val_frac,
        "train_end_ms": train_end_ms, "val_end_ms": val_end_ms,
        "embargo_ms": embargo_ms_eff,
        "include_post_close": include_post_close,
        "min_field_coverage": min_field_coverage,
        "epochs": epochs, "lr": lr,
        "seed": seed, "streaming": streaming, "demo": demo,
        "decision_min_interval_ms": decision_min_interval_ms,
        "dataset_manifest": (
            str(dataset_manifest_path) if dataset_manifest_path else None),
        "eval_mode": eval_mode,
        "cutoff_fit_ms": cutoff_fit_ms,
    }
    manifest = _write_manifest(
        out, research_dir=out.resolve(), splits=split_meta["splits"],
        seeds=[seed], extra_files=[mpath, met_path], config=config)
    manifest["boundaries"] = split_meta["boundaries"]
    manifest["exclusions"] = exclusions

    # Dataset provenance: the manifest declares the sources with hash/window and
    # the label stats of the dataset_manifest emitted by dataset_tape.py/export_sqlite.py,
    # and VERIFIES that the events file used matches the declared sha256 —
    # altered evidence is an explicit failure.
    if dataset_manifest_path:
        dm_path = Path(dataset_manifest_path)
        dm = json.loads(dm_path.read_text())
        prov = {"manifest_path": str(dm_path), "manifest": dm}
        if events_path:
            h = _sha256_file(events_path)
            dec = None
            for f in dm.get("files") or []:
                if Path(f.get("path", "")).name == "events.jsonl":
                    dec = f.get("sha256")
            prov["events_sha256"] = h
            prov["declared_events_sha256"] = dec
            prov["verified_events_sha256"] = (
                (dec == h) if dec is not None else None)
            if dec is not None and dec != h:
                raise RuntimeError(
                    "events.jsonl does not match the sha256 declared "
                    f"in {dm_path} ({h} != {dec}) — altered evidence, "
                    "training on it is refused")
        manifest["dataset"] = prov

    manifest["eval_mode"] = eval_mode
    manifest["cutoff_fit_ms"] = cutoff_fit_ms
    manifest["test_start_ms"] = test_start_ms
    manifest["oos_operable"] = oos_operable
    manifest["requires_future"] = requires_future
    manifest["oos_evidence"] = sufficiency["oos_evidence"]
    manifest["sufficiency"] = sufficiency
    manifest["model_hash"] = _sha256_file(mpath)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))

    return {"metrics": metrics, "manifest": manifest,
            "model_path": str(mpath)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--events", help="ResearchEvent JSONL tape")
    src.add_argument("--demo", action="store_true",
                     help="self-contained synthetic tape")
    ap.add_argument("--out", default=None, help="artifacts directory")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--train-end-ms", type=int, default=None,
                    help="explicit UTC boundary train|rest (ms epoch)")
    ap.add_argument("--val-end-ms", type=int, default=None,
                    help="explicit UTC boundary validation|test (ms)")
    ap.add_argument("--embargo-ms", type=int, default=None,
                    help="embargo between partitions (default "
                         "FLOW_WINDOW_MS=60000)")
    ap.add_argument("--include-post-close", action="store_true",
                    help="do not exclude post-close observations")
    ap.add_argument("--min-field-coverage", type=float, default=0.5,
                    help="minimum coverage of determinable post_close")
    ap.add_argument("--streaming", action="store_true",
                    help="two streaming passes over the tape (multi-GB tapes)")
    ap.add_argument("--decision-min-interval-ms", type=int, default=None,
                    help="per-market temporal decision grid (real "
                         "high-frequency tapes; use the same value as "
                         "in dataset_tape.py so partitions are identical)")
    ap.add_argument("--dataset-manifest", default=None,
                    help="dataset_manifest.json of the dataset (provenance: "
                         "sources+hashes+window+label stats are embedded in "
                         "the manifest and the events.jsonl sha256 is verified)")
    ap.add_argument("--declare-sufficient", action="store_true",
                    help="declare OOS evidence sufficient ONLY if the "
                         "target thresholds already pass (without it, the "
                         "verdict stays insufficient)")
    ap.add_argument("--cutoff-fit-ms", type=int, default=None,
                    help="fit instant for label availability (default: "
                         "retrospective → last known_at of train/val; "
                         "point_in_time → first test decision − 1ms)")
    ap.add_argument("--eval-mode", choices=("retrospective",
                                            "point_in_time"),
                    default="retrospective",
                    help="retrospective: post-capture holdout (may "
                         "use labels after the test start; NOT an "
                         "operable OOS) | point_in_time: requires a fit "
                         "strictly earlier than the test start")
    args = ap.parse_args(argv)
    res = run(events_path=args.events, out_dir=args.out, epochs=args.epochs,
              lr=args.lr, seed=args.seed, train_frac=args.train_frac,
              val_frac=args.val_frac, train_end_ms=args.train_end_ms,
              val_end_ms=args.val_end_ms, embargo_ms=args.embargo_ms,
              include_post_close=args.include_post_close,
              min_field_coverage=args.min_field_coverage,
              demo=args.demo, streaming=args.streaming,
              decision_min_interval_ms=args.decision_min_interval_ms,
              dataset_manifest_path=args.dataset_manifest,
              declare_sufficient=args.declare_sufficient,
              cutoff_fit_ms=args.cutoff_fit_ms,
              eval_mode=args.eval_mode)
    print(json.dumps(res["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
