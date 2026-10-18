#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite exporter → JSONL tape of `ResearchEvent` (T8b).

Rebuilds a causal tape of the contract (`src/research_contract.rs`)
from the bot's three historical DBs, ALWAYS read-only (`mode=ro` +
`PRAGMA query_only=ON` — global restriction: never write nor
checkpoint the originals):

- `hrf_state_*.sqlite` — `markets_observed` (meta), `positions` /
  `transactions` (official labels via the `winner_label` of the
  orphan checker's settle — the `winner` field comes from Polymarket's CLOB API,
  provenance `clob_api:orphan_settle`), `fills` (own state, sim),
  `post_close_observations` (PROVISIONAL PCS-1 labels — only with
  `--labels provisional`; never presented as certified).
- `hrf_market_analytics_*.sqlite` — `book_snapshots` (one atomic snapshot
  of BOTH books per row → one `book_snapshot` event with
  `payload.books = {token: {bids, asks}}`).
- `hrf_spot_stream_*.sqlite` — `spot_samples` (ticks with real LOCAL
  `recv_ts_ms` / `avail_ts_ms`; `source_ts_ms` is APPROXIMATE —
  `spot_stream.rs` derives it as `now − dt` from `Instant`, the
  `VolSnapshot` carries no remote trade ts → `price_tick` with
  `origin_ts_ms=None`, `payload.source_ts_ms_approx` and quality
  `SYNTHETIC|NO_REMOTE_TS`, same as `event_from_sample`).

`local_seq` is REASSIGNED by `avail_ts_ms` (ordered merge of all the
sources): the DBs do not keep the recorder's seq (which was
disabled, `[research_events] enabled=false` — `data/research_events/`
does not exist). It is an honest reconstruction: `avail_ts_ms`
is always a real local timestamp of the row; the per-kind priority on
ties is fixed (meta < spot < book < fill < resolution). With seq
monotonic in avail, `build_observations_incremental` reproduces the
batch builder exactly.

Labels: `confirmed` = `winner_label` from positions/transactions (official
venue evidence collected by the settle; T3's `market_resolutions` table
does not exist in this DB — the fetcher never ran on it).
`provisional` = last determined verdict of `post_close_observations`
(spot vs strike proxy — NOT certified; over the 322 markets with both
sources, 314 agree, 8 diverge).

Usage:
    python3 research/numeric_model/export_sqlite.py \
        --out research/data/t8b_confirmed --labels confirmed
"""

import argparse
import hashlib
import heapq
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import research_inventory  # noqa: E402  — effective path resolution (T1)

SCHEMA_VERSION = 1

# Contract QualityFlags (research_contract.rs).
Q_NO_REMOTE_TS = 1 << 3
Q_REDUCED_FIDELITY = 1 << 4
Q_SYNTHETIC = 1 << 5

# Kind priority on avail ties (seq is assigned by (avail, prio)).
PRIO = {"market_meta": 0, "price_tick": 1, "book_snapshot": 2,
        "sim_fill": 3, "resolution": 4}

# Official labels: venue outcome_label → contract side.
MAP_WINNER = {"Up": "yes", "Down": "no", "Yes": "yes", "No": "no"}

PROVENANCE_SETTLE = "clob_api:orphan_settle"
PROVENANCE_PROXY = "proxy:post_close_verdict"


def open_ro(path):
    """Strict read-only open — same barrier as research_inventory."""
    return research_inventory.open_readonly(Path(path))


def _iso(ts_ms):
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError, TypeError):
        return None


def _levels(col_json):
    """`yes_bids_json` ([{"price","size"}…]) → string pairs [[price, size]…]
    (the contract's `bids`/`asks` format)."""
    if not col_json:
        return []
    out = []
    for lv in json.loads(col_json):
        out.append([str(lv.get("price")), str(lv.get("size"))])
    return out


# ── Labels ────────────────────────────────────────────────────────────────────


def _norm_outcome(label):
    """Normalizes a venue outcome_label to 'yes'/'no' (both
    vocabularies: Up/Down and Yes/No)."""
    m = {"up": "yes", "yes": "yes", "down": "no", "no": "no"}
    return m.get(str(label).lower())


def confirmed_labels(state):
    """Official winner per market from `positions.winner_label` +
    `transactions.winner_label` (orphan checker's settle against the CLOB
    API). Decodes ALL rows and aborts on an out-of-vocabulary winner or a
    conflict between sources (decode everything —
    never filter corruption silently).

    `orphan.rs` fallback hazard: when the winning token has
    `winner=Some(true)` but `outcome=None`, the settle writes
    "Yes"/"No" depending on whether the bot's POSITION won — not the real side. In an
    updown market (outcomes "Up"/"Down") a winner_label "Yes"/"No" is the
    signature of that fallback → potentially INVERTED label: it is excluded from the
    confirmed set and counted in `suspicious`. Extra cross-check: the
    winner implied by `positions.outcome_label`+`outcome_won` must
    match — a contradiction → suspicious too."""
    lab = {}
    raw_labels = {}
    query = (
        "SELECT DISTINCT market_cid, winner_label FROM positions "
        "WHERE winner_label IS NOT NULL",
        "SELECT DISTINCT market_cid, winner_label FROM transactions "
        "WHERE winner_label IS NOT NULL",
    )
    for sql in query:
        for cid, w in state.execute(sql):
            wn = MAP_WINNER.get(w)
            if wn is None:
                raise ValueError(
                    f"winner_label out of vocabulary: {w!r} (cid {cid})")
            if cid in lab and lab[cid] != wn:
                raise ValueError(
                    f"conflicting winner_label for {cid}: "
                    f"{lab[cid]!r} vs {wn!r}")
            lab[cid] = wn
            raw_labels[cid] = w

    # Fallback signature: updown slug + a Yes/No vocabulary label.
    suspicious = set()
    slugs = {
        cid: s for cid, s in state.execute(
            "SELECT condition_id, slug FROM markets_observed")
    }
    for cid, raw in raw_labels.items():
        slug = slugs.get(cid) or ""
        if "-updown-" in slug and str(raw) in ("Yes", "No"):
            suspicious.add(cid)

    # Cross-check: winner implied by positions.outcome_label+outcome_won
    # (won=1 → winner = the token's outcome; won=0 → the opposite).
    contradictions = set()
    for cid, olab, won in state.execute(
        "SELECT market_cid, outcome_label, outcome_won FROM positions "
        "WHERE outcome_won IS NOT NULL AND outcome_label IS NOT NULL"
    ):
        side = _norm_outcome(olab)
        if side is None:
            continue
        implied = side if won else ("no" if side == "yes" else "yes")
        if cid in lab and lab[cid] != implied:
            contradictions.add(cid)
    suspicious |= contradictions

    # known_at = local instant at which the bot settled the market (first
    # REDEEM of that cid). No fallback to clear_at*1000 (addendum v2):
    # a market without a REDEEM entry keeps known_at_ms=None —
    # `label_without_availability`, excluded from the causal train.
    known = {}
    for cid, ts in state.execute(
        "SELECT market_cid, MIN(ts_ms) FROM transactions "
        "WHERE kind='REDEEM' GROUP BY market_cid"
    ):
        known[cid] = ts
    out = {}
    for cid, wn in lab.items():
        if cid in suspicious:
            continue
        out[cid] = {
            "winner": wn,
            "known_at_ms": known.get(cid),
            "provenance": PROVENANCE_SETTLE,
        }
    return out, {
        "raw_labels": len(lab),
        "suspicious_fallback_labels": len(suspicious),
        "outcome_contradictions": len(contradictions),
        "suspicious_cids": sorted(suspicious),
    }


def provisional_labels(state):
    """Last determined verdict of `post_close_observations` per market
    (PCS-1: frozen spot vs reference_price). NOT certified — the
    artifact that uses it must declare it. ORDER BY (ts_ms, rowid): on
    ts ties the later-inserted wins — deterministic."""
    prov = {}
    for cid, v, ts in state.execute(
        "SELECT market_cid, verdict, ts_ms FROM post_close_observations "
        "WHERE verdict IN ('yes','no') ORDER BY ts_ms, rowid"
    ):
        prov[cid] = (v, ts)
    return {
        cid: {"winner": v, "known_at_ms": ts, "provenance": PROVENANCE_PROXY}
        for cid, (v, ts) in prov.items()
    }, {"raw_labels": len(prov)}


# ── Event generators (yield (avail_ms, prio, event)) ────────────────────────


def _ev(source, kind, cid, token, asset, payload, origin, recv, avail,
        quality):
    return {
        "schema_version": SCHEMA_VERSION, "source": source, "kind": kind,
        "market_cid": cid, "token_id": token, "asset": asset,
        "payload": payload, "origin_ts_ms": origin, "recv_ts_ms": recv,
        "avail_ts_ms": avail, "local_seq": None, "quality": quality,
    }


def _meta_payload(cid, slug, yes_tok, no_tok, open_ts, clear_at, neg_risk):
    asset = (slug or "").split("-updown")[0].lower() or None
    return {
        "schema_version": SCHEMA_VERSION, "market_cid": cid,
        "slug": slug, "yes_token_id": yes_tok, "no_token_id": no_tok,
        "asset": asset, "neg_risk": bool(neg_risk),
        "market_open_ts": open_ts, "clear_at": clear_at,
        "reference_price": None,  # filled with the first row
        "rule": "up_down_reference",
        "resolution_source": "binance_ws",
        "resolution_source_ref": (
            f"{asset.upper()}USDT spot" if asset else None),
        "rules_windows": [], "outcome": None,
    }


def gen_metas(state, analytics, cids, min_snap_ts):
    """market_meta per labeled market. `avail` = min(first_seen_ts,
    first local snapshot of the market) — the earliest local instant at
    which the market existed for the bot (not a remote timestamp). The
    stream is reordered by avail before emitting: heapq.merge requires each
    source to be ordered so that seq is monotonic in avail.

    If the cid is not in `markets_observed`, the meta is SYNTHESIZED from the
    first row of the cid's `book_snapshots` (which carries slug, tokens,
    clear_at and market_open_ts) — without a meta the builder ignores the market and
    the row would remain a silent imputation. It carries
    `provenance='sqlite:book_snapshots:sintetizado'` and
    REDUCED_FIDELITY; `run()` counts how many."""
    items = []
    synthesized = 0
    for cid in sorted(cids):
        row = state.execute(
            "SELECT slug, yes_token_id, no_token_id, market_open_ts, "
            "clear_at, neg_risk, first_seen_ts FROM markets_observed "
            "WHERE condition_id = ?", (cid,)
        ).fetchone()
        if row is None:
            continue
        slug, yes_tok, no_tok, open_ts, clear_at, neg_risk, first_seen = row
        avail = (first_seen or 0) * 1000
        if cid in min_snap_ts and min_snap_ts[cid] < avail:
            avail = min_snap_ts[cid]
        payload = _meta_payload(cid, slug, yes_tok, no_tok, open_ts,
                                clear_at, neg_risk)
        items.append((avail, PRIO["market_meta"], _ev(
            "polymarket_rest", "market_meta", cid, None,
            payload["asset"], payload, None, avail, avail,
            Q_NO_REMOTE_TS)))

    # Honest fallback: meta synthesized from the first row of the book.
    with_meta = {ev["market_cid"] for _, _, ev in items}
    for cid in sorted(set(cids) - with_meta):
        row_entry = analytics.execute(
            "SELECT ts_ms, slug, yes_token_id, no_token_id, "
            "market_open_ts, clear_at FROM book_snapshots "
            "WHERE market_condition_id = ? ORDER BY ts_ms LIMIT 1",
            (cid,)).fetchone()
        if row_entry is None:
            continue
        ts, slug, yes_tok, no_tok, open_ts, clear_at = row_entry
        payload = _meta_payload(cid, slug, yes_tok, no_tok, open_ts,
                                clear_at, 0)
        payload["provenance"] = "sqlite:book_snapshots:sintetizado"
        # avail = ts of the first row (local epoch-ms). Do NOT use
        # market_open_ts: it comes in venue epoch-S, not comparable.
        avail = ts
        items.append((avail, PRIO["market_meta"], _ev(
            "polymarket_ws", "market_meta", cid, None,
            payload["asset"], payload, None, avail, avail,
            Q_NO_REMOTE_TS | Q_REDUCED_FIDELITY)))
        synthesized += 1

    items.sort(key=lambda t: t[0])
    yield from items
    gen_metas.synthesized = synthesized


def gen_books(analytics, cids):
    """One `book_snapshot` per row with `payload.books` = both books —
    the row IS an atomic snapshot of the two tokens. Merge of per-cid streams
    (index `idx_book_snapshots_market`) already sorted by ts."""
    streams = []
    for cid in sorted(cids):
        cur = analytics.execute(
            "SELECT ts_ms, slug, yes_token_id, yes_bids_json, "
            "yes_asks_json, no_token_id, no_bids_json, no_asks_json "
            "FROM book_snapshots WHERE market_condition_id = ? "
            "ORDER BY ts_ms", (cid,))

        def _stream(cur=cur, cid=cid):
            for (ts, slug, yes_tok, yb, ya, no_tok, nb, na) in cur:
                ev = _ev(
                    "polymarket_ws", "book_snapshot", cid, yes_tok, None,
                    {"books": {
                        yes_tok: {"bids": _levels(yb), "asks": _levels(ya)},
                        no_tok: {"bids": _levels(nb), "asks": _levels(na)},
                    }},
                    None, ts, ts,
                    Q_NO_REMOTE_TS | Q_REDUCED_FIDELITY)
                yield ts, PRIO["book_snapshot"], ev

        streams.append(_stream())
    yield from heapq.merge(*streams, key=lambda t: (t[0], t[1]))


def gen_spot(spotdb, assets, lo_ms, hi_ms):
    """Spot ticks from `spot_samples` (only `source='binance'` — the
    oracle feed; other sources, if they appeared, are reported in the manifest).

    `source_ts_ms` of the row is APPROXIMATE (`spot_stream.rs`: the
    `VolSnapshot` carries no remote trade timestamp; it is derived as
    `now_ms − dt` from `Instant`). Same as `event_from_sample`:
    `origin_ts_ms=None` (never invented), the approximation travels in
    `payload.source_ts_ms_approx` and quality carries `SYNTHETIC|NO_REMOTE_TS`
    (a derived view of the bot, not a raw venue observation)."""
    if not assets or lo_ms is None:
        return
    ph = ",".join("?" for _ in assets)
    cur = spotdb.execute(
        f"SELECT asset, source, price, source_ts_ms, recv_ts_ms, "
        f"avail_ts_ms FROM spot_samples "
        f"WHERE avail_ts_ms BETWEEN ? AND ? AND source = 'binance' "
        f"AND asset IN ({ph}) ORDER BY avail_ts_ms",
        (lo_ms, hi_ms, *assets))
    for asset, source, price, src_approx, recv, avail in cur:
        yield avail, PRIO["price_tick"], _ev(
            "binance_ws", "price_tick", None, None, asset,
            {"price": str(price), "source": source,
             "source_ts_ms_approx": src_approx},
            None, recv, avail,
            Q_SYNTHETIC | Q_NO_REMOTE_TS)


def gen_fills(state, cids):
    """Own fills → `{"sim":"fill"}` events (contract C). In paper they are
    simulator executions: quality SYNTHETIC + NO_REMOTE_TS."""
    if not cids:
        return
    ph = ",".join("?" for _ in cids)
    cur = state.execute(
        f"SELECT ts_ms, market_cid, token_id, side, price_um, size_um, "
        f"order_id, local_order_id, exec_id, strategy FROM fills "
        f"WHERE market_cid IN ({ph}) ORDER BY ts_ms", tuple(cids))
    for (ts, cid, tok, side, price_um, size_um, oid, loid, eid, strat) in cur:
        side_s = "Buy" if side == "BUY" else "Sell" if side == "SELL" else side
        detail = {
            "exec_id": eid or f"fill-{cid}-{ts}",
            "order_id": oid, "local_order_id": loid, "token_id": tok,
            "side": side_s,
            "shares": str(size_um / 1e6) if size_um is not None else None,
            "price": str(price_um / 1e6) if price_um is not None else None,
            "ts_ms": ts, "strategy": strat,
        }
        yield ts, PRIO["sim_fill"], _ev(
            "internal_sim", {"sim": "fill"}, cid, tok, None,
            {"detail": detail}, None, ts, ts,
            Q_SYNTHETIC | Q_NO_REMOTE_TS)


def gen_resolutions(labels, avail_bound_ms=None):
    """One `resolution` event per labeled market (winner + provenance +
    local known_at_ms). Reordered by avail before emitting (see
    `gen_metas`).

    Labels without a real `known_at_ms` (no known availability): the
    event is emitted with `avail_ts_ms = avail_bound_ms` — the END of the
    exported window, a conservative upper bound that never fakes an
    early availability. The payload declares
    `avail_is_upper_bound: true` and `known_at_ms: None`: the label is
    unusable for causal train (`filter_labels_available`) but
    available for retrospective eval. `avail_bound_ms=None` (no
    snapshot coverage) degrades to 0 — no data to contaminate."""
    items = []
    for cid in sorted(labels):
        lab = labels[cid]
        known = lab.get("known_at_ms")
        bound = known is None
        avail = known if known is not None else (avail_bound_ms or 0)
        items.append((avail, PRIO["resolution"], _ev(
            "polymarket_rest", "resolution", cid, None, None,
            {"winner": lab["winner"], "known_at_ms": known,
             "provenance": lab["provenance"],
             "avail_is_upper_bound": bound},
            None, avail, avail, Q_NO_REMOTE_TS)))
    items.sort(key=lambda t: t[0])
    yield from items


# ── Orchestration ─────────────────────────────────────────────────────────────


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat_source(path):
    p = Path(path)
    if not p.is_file():
        return {"resolved_path": str(p), "exists": False}
    st = p.stat()
    return {"resolved_path": str(p), "exists": True, "bytes": st.st_size,
            "mtime_iso": datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc).isoformat()}


def _validate_out_dir(out_dir, sources):
    """Local barrier (adapted from `research_inventory._validate_out_dir`,
    without its veto on the repo tree — this exporter writes to
    `research/data/` by T8b contract): forbids `out` from being the
    DIRECTORY of a source DB or an ancestor of it — the artifacts
    would land next to the live DB or on the file itself. An `out`
    descending from that directory (e.g. `research/data/` when the
    spot DB lives at the repo root) is legitimate and allowed."""
    out = Path(out_dir).resolve()
    for name_str, f in sources.items():
        if name_str == "mode":
            continue
        src = Path(f["resolved_path"]).resolve()
        if out == src or out in src.parents:
            raise ValueError(
                f"--out {out} writes over the source '{name_str}' or next "
                f"to it ({src}) — choose another directory")
    return out


def run(out_dir, labels_kind, repo_root=REPO_ROOT, config_path=None,
        env_path=None, mode="paper"):
    sources = research_inventory.effective_sources(
        Path(repo_root),
        Path(config_path or Path(repo_root) / "config" / "general.toml"),
        Path(env_path or Path(repo_root) / ".env"),
        cli_overrides={"mode": mode})
    out = _validate_out_dir(out_dir, sources)
    out.mkdir(parents=True, exist_ok=True)

    state = open_ro(sources["state"]["resolved_path"])
    analytics = open_ro(sources["analytics"]["resolved_path"])
    spotdb = open_ro(sources["spot_stream"]["resolved_path"])

    labels, stats_labels = (
        confirmed_labels(state) if labels_kind == "confirmed"
        else provisional_labels(state))
    cids = sorted(labels)

    # Labeled markets WITH snapshot coverage + first ts per cid
    # (queries indexed by market_condition_id — never a full scan).
    min_snap_ts = {}
    with_snap = []
    for cid in cids:
        row = analytics.execute(
            "SELECT MIN(ts_ms), MAX(ts_ms), COUNT(*) FROM book_snapshots "
            "WHERE market_condition_id = ?", (cid,)).fetchone()
        if row and row[2]:
            min_snap_ts[cid] = row[0]
            with_snap.append(cid)
    # reference_price and clear_at per market from the first row of the
    # snapshot (markets_observed does not store reference_price).
    ref_price = {}
    for cid in with_snap:
        row = analytics.execute(
            "SELECT reference_price FROM book_snapshots "
            "WHERE market_condition_id = ? AND reference_price IS NOT NULL "
            "ORDER BY ts_ms LIMIT 1", (cid,)).fetchone()
        if row:
            ref_price[cid] = row[0]

    assets = set()
    meta_rows = {}
    for cid in cids:
        row = state.execute(
            "SELECT slug, clear_at FROM markets_observed "
            "WHERE condition_id = ?", (cid,)).fetchone()
        if row:
            meta_rows[cid] = row
        else:
            # cid without a row in markets_observed: asset from the slug of the
            # first row of book_snapshots (same as gen_metas).
            row_entry = analytics.execute(
                "SELECT slug, clear_at FROM book_snapshots "
                "WHERE market_condition_id = ? ORDER BY ts_ms LIMIT 1",
                (cid,)).fetchone()
            if row_entry:
                meta_rows[cid] = row_entry
        if cid in meta_rows:
            a = (meta_rows[cid][0] or "").split("-updown")[0].lower()
            if a:
                assets.add(a)

    # No backdating (addendum v2): a label without a known real
    # availability (no REDEEM entry) keeps `known_at_ms=None` — it is
    # excluded from causal train via `filter_labels_available` and is
    # counted here as `label_without_availability`. NEVER a silent
    # `clear_at*1000`: that fabricated an availability that
    # breaks label causality (verified: the T8b retrain used
    # labels available "before" the venue resolved).
    without_availability = sorted(
        cid for cid, lab in labels.items() if lab["known_at_ms"] is None)
    stats_labels["label_without_availability"] = len(without_availability)
    stats_labels["cids_without_availability"] = without_availability

    # reference_price patch on the already generated metas: since the metas
    # are emitted by a generator, it is injected into the payload inside the stream.
    def _metas_with_ref():
        for avail, prio, ev in gen_metas(state, analytics, cids,
                                         min_snap_ts):
            cid = ev["market_cid"]
            if cid in ref_price:
                ev["payload"]["reference_price"] = ref_price[cid]
            yield avail, prio, ev

    lo_ms = min(min_snap_ts.values()) if min_snap_ts else None
    hi_ms = None
    if min_snap_ts:
        hi_ms = max(
            analytics.execute(
                "SELECT MAX(ts_ms) FROM book_snapshots "
                "WHERE market_condition_id = ?", (cid,)
            ).fetchone()[0] for cid in with_snap)

    streams = [
        _metas_with_ref(),
        gen_books(analytics, with_snap),
        gen_spot(spotdb, sorted(assets), lo_ms, hi_ms),
        gen_fills(state, with_snap),
        gen_resolutions(labels, avail_bound_ms=hi_ms),
    ]

    tape = out / "events.jsonl"
    counts = {}
    n = 0
    prev_avail = None
    inversions = 0
    t0 = time.time()
    with open(tape, "w", encoding="utf-8") as f:
        for avail, prio, ev in heapq.merge(
                *streams, key=lambda t: (t[0], t[1])):
            if prev_avail is not None and avail < prev_avail:
                inversions += 1
            prev_avail = avail
            ev["local_seq"] = n
            k = ev["kind"]
            kn = k["sim"] if isinstance(k, dict) else k
            counts[kn] = counts.get(kn, 0) + 1
            f.write(json.dumps(ev, ensure_ascii=False,
                               separators=(",", ":")) + "\n")
            n += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - t0, 2),
        "labels_kind": labels_kind,
        "mode": mode,
        "labels": stats_labels,
        "metas_synthesized_from_books": gen_metas.synthesized,
        "provenance": sorted({l["provenance"] for l in labels.values()}),
        "sources": {k: _stat_source(sources[k]["resolved_path"])
                    for k in ("state", "analytics", "spot_stream")},
        "coverage": {
            "label_markets": len(labels),
            "markets_with_snapshots": len(with_snap),
            "assets_spot": sorted(assets),
            "snapshots_range_ms": [lo_ms, hi_ms],
            "snapshots_iso_range": [_iso(lo_ms), _iso(hi_ms)],
            "n_events": n,
            "counts_by_kind": counts,
            "inversions_avail_in_seq": inversions,
        },
        "notes": [
            "local_seq reassigned by (avail_ts_ms, prio): the DBs do not "
            "keep the recorder's seq (disabled during the capture).",
            "avail_ts_ms is always a real LOCAL timestamp of the exported "
            "row; meta.avail = min(first_seen, first snapshot).",
            "Confirmed labels = winner_label from the orphan settle against the "
            "CLOB API; market_resolutions (T3) does not exist in this DB. "
            "Anti-fallback cross-check of orphan.rs in "
            "labels.suspicious_fallback_labels.",
            "No backdating of known_at_ms (addendum v2): labels without "
            "a REDEEM entry keep known_at_ms=None "
            "(labels.label_without_availability) and their resolution event is "
            "emitted at the end of the tape as an upper bound "
            "(payload.avail_is_upper_bound) — never clear_at*1000.",
            "quality: books NO_REMOTE_TS|REDUCED_FIDELITY (deduplicated "
            "snapshots, no remote ts); metas synthesized from "
            "book_snapshots additionally carry REDUCED_FIDELITY; fills "
            "SYNTHETIC|NO_REMOTE_TS (paper simulator); ticks "
            "SYNTHETIC|NO_REMOTE_TS — origin_ts_ms=None because "
            "source_ts_ms of spot_samples is APPROXIMATE (now-dt from "
            "Instant, spot_stream.rs) and travels in "
            "payload.source_ts_ms_approx.",
            "Spot ticks filtered to source='binance' (oracle feed).",
            "No tape flow (market trades): the T2 recorder was "
            "disabled during the capture — flow_* features absent.",
        ],
        "files": [
            {"path": tape.name, "sha256": _sha256_file(tape),
             "bytes": tape.stat().st_size},
        ],
    }
    mpath = out / "dataset_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"manifest": manifest, "events": str(tape)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True,
                    help="output research directory")
    ap.add_argument("--labels", choices=["confirmed", "provisional"],
                    default="confirmed")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--config", default=None)
    ap.add_argument("--env", default=None)
    ap.add_argument("--mode", choices=["paper", "live"], default="paper",
                    help="mode of the source DBs (default paper; explicit "
                         "so that an HRF_MODE=live in the environment does not silently export "
                         "the live DB)")
    args = ap.parse_args(argv)
    res = run(args.out, args.labels, repo_root=args.repo_root,
              config_path=args.config, env_path=args.env, mode=args.mode)
    print(json.dumps(res["manifest"]["coverage"], indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
