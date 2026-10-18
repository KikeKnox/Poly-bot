"""Builder of numeric observations from a `ResearchEvent` tape
(JSON serialization of the `src/research_contract.rs` contract).

Strict causality (contracts A/B/D):
- The canonical order is `local_seq`; remote timestamps order nothing.
- An observation emitted at the event with seq S and decision at T only sees
  events with `local_seq <= S` AND `avail_ts_ms <= T`. The double cutoff is
  what keeps an event with a low seq but future availability
  (e.g. a late resync) from contaminating past features.
- The official result (`MarketMeta.outcome`, `MarketLabel`) NEVER enters
  the features: it is only used via `build_labels` as a supervised target.

Cost: `build_observations` replays the visible prefix per observation —
O(n²) in the worst case. Acceptable for the phase A scaffold; phase B
(real data) will have to move to incremental state with an application queue
deferred by `avail_ts_ms`.

Stdlib only (json/math/statistics): numpy/pandas are installed on the
system but the phase A contract does not need them.
"""

import heapq
import json
import math
import statistics

SCHEMA_VERSION = 1


class NonMonotonicTape(Exception):
    """The tape cannot be processed in a single causal pass: a decision
    arrived with an `avail_ts_ms` lower than that of events already applied to the state
    — the incremental builder cannot "unsee" that state. The caller must
    fall back to the O(n²) batch builder (`build_observations`), which does process it
    exactly (it rebuilds the visible prefix per decision)."""

# Contract kinds that trigger a decision point by default.
DECISION_KINDS_DEFAULT = ("book_snapshot", "book_delta", "trade")

# Aggregated depth level and flow window for the features.
DEPTH_LEVELS = 5
FLOW_WINDOW_MS = 60_000
RET_WINDOW = 5
VOL_WINDOW = 10

FEATURE_NAMES = [
    # YES book
    "yes_bid", "yes_ask", "yes_mid", "yes_spread",
    "yes_depth_bid", "yes_depth_ask", "yes_imbalance", "yes_microprice",
    # NO book
    "no_bid", "no_ask", "no_mid", "no_spread", "no_imbalance",
    # YES+NO parity (up/down bracket arbitrage)
    "parity_ask_sum",
    # Tape flow in the FLOW_WINDOW_MS window
    "flow_buy_vol", "flow_sell_vol", "flow_imbalance", "trades_count",
    # Binance spot and derived values
    "spot", "spot_ret_1", "spot_ret_n", "spot_vol", "moneyness",
    # Time remaining (ms and fraction of the bracket's life)
    "time_left_ms", "time_left_frac",
    # Data quality and age
    "book_age_ms", "spot_age_ms", "origin_delay_ms", "quality_bits",
    # Own state (observable orders/inventory, contract C)
    "own_open_orders", "own_net_shares",
]


def load_events_jsonl(path):
    """Reads a JSONL tape of serialized `ResearchEvent`s."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def iter_events_jsonl(path):
    """Streaming version of `load_events_jsonl` (T8b): generator that parses
    line by line without materializing the tape — needed for real tapes
    of ~10⁶ events (the list of dicts exhausts RAM)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _kind(ev):
    """`kind` can be a string or the nested object `{"sim": "<kind>"}`."""
    k = ev.get("kind")
    if isinstance(k, dict):
        return ("sim", next(iter(k.values())))
    return (k, None)


def _num(v):
    """Parses a Decimal serialized as a string (or number) into a float."""
    if v is None:
        return None
    return float(v)


def _parse_levels(levels):
    """Parses book levels into (price, size). Supports both contract
    formats: pairs `[price, size]` (T8b SQLite exporter) and dicts
    `{"price": ..., "size": ...}` (real Polymarket WS tape)."""
    out = []
    for lv in levels or []:
        if isinstance(lv, dict):
            out.append((_num(lv.get("price")), _num(lv.get("size"))))
        elif isinstance(lv, (list, tuple)) and len(lv) >= 2:
            out.append((_num(lv[0]), _num(lv[1])))
    return [(p, s) for p, s in out if p is not None and s is not None]


def _market_cid(ev, kind=None):
    """Effective `market_cid` of the event: top-level, falling back to
    `payload.market` for the real Polymarket WS book_delta (where the
    top-level travels null and the market is only in the payload)."""
    c = ev.get("market_cid")
    if c is None and (kind or _kind(ev)[0]) in ("book_snapshot",
                                              "book_delta"):
        c = (ev.get("payload") or {}).get("market")
    return c


class _MarketState:
    """Causal state of a market: metas, YES/NO books, flow and own state."""

    def __init__(self):
        self.meta = None
        self.books = {}          # token_id -> {"bids": {p: s}, "asks": {p: s}}
        self.trades = []         # (avail_ts_ms, side, price, size)
        self.last_book_ts = None # avail_ts_ms of the last book event
        self.last_origin_delay = None
        self.last_quality = 0
        self.open_orders = {}    # order_id -> alive (accepted, not closed)
        self.net_shares = {}     # token_id -> net shares (own fills)
        self._order_token = {}   # order_id/local_order_id -> token_id
        # Quality counter: fills whose payload produced no shares — the
        # degradation is loud (visible per observation), never silent
        # zeros in `own_net_shares`.
        self.unparsed_fills = 0

    def apply_meta(self, ev):
        self.meta = ev.get("payload") or {}

    def apply_book(self, ev):
        payload = ev.get("payload") or {}
        kind, _ = _kind(ev)
        # SQLite exporter format (T8b): `payload.books` is a dict
        # {token_id: {"bids": [[p,s]..], "asks": ..}} — ONE row of
        # `book_snapshots` is an atomic snapshot of BOTH books of the
        # market, so the event applies to all the declared tokens
        # (complete-replacement semantics, like any snapshot).
        books_multi = payload.get("books")
        if isinstance(books_multi, dict):
            for tok_b, lv in books_multi.items():
                if not isinstance(lv, dict):
                    continue
                self.books[tok_b] = {
                    "bids": {p: s for p, s in _parse_levels(lv.get("bids"))},
                    "asks": {p: s for p, s in _parse_levels(lv.get("asks"))},
                }
        else:
            tok = ev.get("token_id")
            if kind == "book_snapshot":
                if not tok:
                    return
                book = self.books.setdefault(tok, {"bids": {}, "asks": {}})
                book["bids"] = {
                    p: s for p, s in _parse_levels(payload.get("bids"))
                }
                book["asks"] = {
                    p: s for p, s in _parse_levels(payload.get("asks"))
                }
            else:  # book_delta: incremental update, size 0 = delete level
                # per-level bids/asks (exporter format): apply to the
                # event's token if it exists.
                if tok:
                    book = self.books.setdefault(
                        tok, {"bids": {}, "asks": {}})
                    for side_key in ("bids", "asks"):
                        for p, s in _parse_levels(payload.get(side_key)):
                            if s == 0:
                                book[side_key].pop(p, None)
                            else:
                                book[side_key][p] = s
                # price_changes: two formats —
                #  [side, price, size] over the event's token
                #  (exporter), or dict {asset_id, price, side, size}
                #  from the real Polymarket WS, where EACH change carries its
                #  own token (asset_id) and the event's may be
                #  null — without this case the real delta was never applied.
                for ch in payload.get("price_changes") or []:
                    if isinstance(ch, dict):
                        ctok = ch.get("asset_id") or tok
                        p = _num(ch.get("price"))
                        s = _num(ch.get("size"))
                        if not ctok or p is None or s is None:
                            continue
                        side = ch.get("side")
                    elif len(ch) >= 3 and tok:
                        ctok = tok
                        side, p, s = ch[0], _num(ch[1]), _num(ch[2])
                    else:
                        continue
                    if p is None or s is None:
                        continue
                    key = ("bids"
                           if str(side).lower() in ("buy", "bid")
                           else "asks")
                    cbook = self.books.setdefault(
                        ctok, {"bids": {}, "asks": {}})
                    if s == 0:
                        cbook[key].pop(p, None)
                    else:
                        cbook[key][p] = s
        self.last_book_ts = ev["avail_ts_ms"]
        self.last_quality = ev.get("quality", 0) or 0
        if ev.get("origin_ts_ms") is not None:
            self.last_origin_delay = ev["recv_ts_ms"] - ev["origin_ts_ms"]

    def apply_trade(self, ev):
        payload = ev.get("payload") or {}
        self.trades.append((
            ev["avail_ts_ms"],
            str(payload.get("side", "")).lower(),
            _num(payload.get("price")),
            _num(payload.get("size")),
        ))

    def apply_sim(self, ev, sub):
        """Own causal accounting: live orders and net shares from fills.

        Contract C: in a faithful tape, a `{"sim":"fill"}` carries the
        serialized `SimExecution` in `payload["detail"]` (with `side`
        "Buy"/"Sell", `shares`, `order_id`, `local_order_id`, `token_id`);
        the ids may also live at the payload top level (fields
        of the `SimEvent`). `detail` is read first with a fallback to the
        top level to tolerate tapes with flat economics.

        A fill whose payload yields no shares does NOT count as a silent
        zero: it increments `unparsed_fills`, a quality counter that
        is exposed in every observation.
        """
        payload = ev.get("payload") or {}
        detail = payload.get("detail")
        econ = detail if isinstance(detail, dict) else payload
        oid = (
            econ.get("order_id") or payload.get("order_id")
            or econ.get("local_order_id") or payload.get("local_order_id")
        )
        tok = econ.get("token_id") or payload.get("token_id") or ev.get("token_id")
        if sub == "accept" and oid:
            self.open_orders[oid] = True
            if tok:
                self._order_token[oid] = tok
        elif sub in ("cancel_ack", "reject") and oid:
            self.open_orders.pop(oid, None)
        elif sub == "fill":
            shares = _num(econ.get("shares"))
            side = str(econ.get("side") or "").lower()
            if shares is None or side not in ("buy", "sell"):
                # Unrecognizable payload — loud degradation.
                self.unparsed_fills += 1
                return
            signed = shares if side == "buy" else -shares
            tok_f = tok or self._order_token.get(oid)
            if tok_f:
                self.net_shares[tok_f] = self.net_shares.get(tok_f, 0.0) + signed
            # `remaining` does not exist in `SimExecution`; a tape that
            # reports it (detail or top level) can mark a full close.
            remaining = _num(econ.get("remaining"))
            if remaining is None:
                remaining = _num(payload.get("remaining"))
            if oid and remaining == 0:
                self.open_orders.pop(oid, None)


class _SpotState:
    """Causal series of spot ticks per underlying."""

    def __init__(self):
        self.series = {}  # asset -> [(avail_ts_ms, price)]

    def apply(self, ev):
        asset = ev.get("asset")
        price = _num((ev.get("payload") or {}).get("price"))
        if asset and price is not None:
            self.series.setdefault(asset, []).append((ev["avail_ts_ms"], price))


def _book_features(book):
    """Features of one side of the book; `None` if the side is empty."""
    bids = sorted(book["bids"].items(), key=lambda kv: -kv[0])
    asks = sorted(book["asks"].items(), key=lambda kv: kv[0])
    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    depth_bid = sum(s for _, s in bids[:DEPTH_LEVELS]) if bids else None
    depth_ask = sum(s for _, s in asks[:DEPTH_LEVELS]) if asks else None
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None
    spread = (ask - bid) if bid is not None and ask is not None else None
    if depth_bid is not None and depth_ask is not None and (depth_bid + depth_ask) > 0:
        imb = (depth_bid - depth_ask) / (depth_bid + depth_ask)
    else:
        imb = None
    if bid is not None and ask is not None and bids and asks:
        bs, as_ = bids[0][1], asks[0][1]
        micro = (ask * bs + bid * as_) / (bs + as_) if (bs + as_) > 0 else mid
    else:
        micro = None
    return {
        "bid": bid, "ask": ask, "mid": mid, "spread": spread,
        "depth_bid": depth_bid, "depth_ask": depth_ask,
        "imbalance": imb, "microprice": micro,
    }


def _spot_features(series, now_ms):
    """Spot, log returns, realized volatility and age of the last tick.

    Only the last VOL_WINDOW+1 ticks are consulted (RET_WINDOW <
    VOL_WINDOW), so we work over the tail of the series — on real tapes
    the series grows to ~10⁶ ticks and copying it whole per observation
    was the builder's hidden O(n) bottleneck (T8b)."""
    if not series:
        return {"spot": None, "spot_ret_1": None, "spot_ret_n": None,
                "spot_vol": None, "spot_age_ms": None}
    prices = [p for _, p in series[-(VOL_WINDOW + 1):]]
    spot = prices[-1]
    age = now_ms - series[-1][0]
    ret_1 = math.log(prices[-1] / prices[-2]) if len(prices) >= 2 else None
    ret_n = (
        math.log(prices[-1] / prices[-1 - RET_WINDOW])
        if len(prices) > RET_WINDOW else None
    )
    rets = [
        math.log(prices[i] / prices[i - 1])
        for i in range(max(1, len(prices) - VOL_WINDOW), len(prices))
        if prices[i - 1] > 0
    ]
    vol = statistics.pstdev(rets) if len(rets) >= 2 else None
    return {"spot": spot, "spot_ret_1": ret_1, "spot_ret_n": ret_n,
            "spot_vol": vol, "spot_age_ms": age}


def _apply_event(states, spot, e):
    """Applies ONE event to the causal state (markets + spot). It is the only
    mutation point: the reference batch builder and the incremental one
    share it, so they cannot diverge in application semantics."""
    k, sub = _kind(e)
    if k == "price_tick":
        spot.apply(e)
        return
    c = _market_cid(e, k)
    if c is None:
        return
    st = states.setdefault(c, _MarketState())
    if k == "market_meta":
        st.apply_meta(e)
    elif k in ("book_snapshot", "book_delta"):
        st.apply_book(e)
    elif k == "trade":
        st.apply_trade(e)
    elif k == "order_submit":
        # Own sent order: counts as a live intent until
        # accept/reject — it is accounted via sim events; here we only
        # record the local_order_id for traceability.
        pass
    elif k == "sim":
        st.apply_sim(e, sub)


def _observation_from_state(st, spot, ev):
    """Builds the observation dict (features + causal metadata) for the
    decision event `ev` with the state already materialized. Shared by
    both builders — a single feature definition."""
    decision_ts = ev["avail_ts_ms"]
    seq_cut = ev["local_seq"]
    cid = _market_cid(ev)
    meta = st.meta or {}
    yes_tok, no_tok = meta.get("yes_token_id"), meta.get("no_token_id")
    asset = meta.get("asset") or ev.get("asset")

    yes = _book_features(st.books.get(yes_tok, {"bids": {}, "asks": {}}))
    no = _book_features(st.books.get(no_tok, {"bids": {}, "asks": {}}))
    sp = _spot_features(spot.series.get(asset, []), decision_ts)

    # Tape flow in the backward window from the decision.
    cut = decision_ts - FLOW_WINDOW_MS
    win = [t for t in st.trades if t[0] > cut]
    buy_vol = sum(s for _, side, _, s in win if side == "buy" and s)
    sell_vol = sum(s for _, side, _, s in win if side == "sell" and s)
    if buy_vol + sell_vol > 0:
        flow_imb = (buy_vol - sell_vol) / (buy_vol + sell_vol)
    else:
        flow_imb = None

    # Time remaining: the meta's clear_at is in unix seconds.
    clear_ms = meta.get("clear_at") * 1000 if meta.get("clear_at") else None
    open_ms = (
        meta.get("market_open_ts") * 1000
        if meta.get("market_open_ts") else None
    )
    time_left = (clear_ms - decision_ts) if clear_ms else None
    if time_left is not None and open_ms and clear_ms and clear_ms > open_ms:
        life = clear_ms - open_ms
        time_left_frac = max(0.0, min(1.0, time_left / life))
    else:
        time_left_frac = None

    # Moneyness vs the opening reference (strike) price.
    ref = _num(meta.get("reference_price"))
    moneyness = (
        math.log(sp["spot"] / ref)
        if sp["spot"] is not None and ref and ref > 0 else None
    )

    parity = None
    if yes["ask"] is not None and no["ask"] is not None:
        parity = yes["ask"] + no["ask"]

    net_shares = sum(st.net_shares.values()) if st.net_shares else 0.0

    features = {
        "yes_bid": yes["bid"], "yes_ask": yes["ask"],
        "yes_mid": yes["mid"], "yes_spread": yes["spread"],
        "yes_depth_bid": yes["depth_bid"], "yes_depth_ask": yes["depth_ask"],
        "yes_imbalance": yes["imbalance"], "yes_microprice": yes["microprice"],
        "no_bid": no["bid"], "no_ask": no["ask"],
        "no_mid": no["mid"], "no_spread": no["spread"],
        "no_imbalance": no["imbalance"],
        "parity_ask_sum": parity,
        "flow_buy_vol": buy_vol if win else 0.0,
        "flow_sell_vol": sell_vol if win else 0.0,
        "flow_imbalance": flow_imb,
        "trades_count": float(len(win)),
        "spot": sp["spot"], "spot_ret_1": sp["spot_ret_1"],
        "spot_ret_n": sp["spot_ret_n"], "spot_vol": sp["spot_vol"],
        "moneyness": moneyness,
        "time_left_ms": time_left, "time_left_frac": time_left_frac,
        "book_age_ms": (
            decision_ts - st.last_book_ts if st.last_book_ts else None
        ),
        "spot_age_ms": sp["spot_age_ms"],
        "origin_delay_ms": st.last_origin_delay,
        "quality_bits": float(st.last_quality),
        "own_open_orders": float(len(st.open_orders)),
        "own_net_shares": net_shares,
    }
    return {
        "market_cid": cid,
        "decision_ts_ms": decision_ts,
        "asof_local_seq": seq_cut,
        "asof_avail_ts_ms": decision_ts,
        "features": features,
        "quality_bits": st.last_quality,
        # Post-close flag: the observation arrives AFTER clear_at
        # (`time_left_ms < 0`) — the caller decides whether to exclude it; it is not
        # filtered here so as not to lose data silently.
        "post_close": bool(time_left is not None and time_left < 0),
        # Quality counter: own fills with an unrecognizable payload
        # seen up to this point (loud degradation, contract C).
        "unparsed_fills": st.unparsed_fills,
    }


def build_observations(events, decision_kinds=None):
    """Builds causal observations from a `ResearchEvent` tape.

    Returns a list of dicts:
      market_cid, decision_ts_ms, asof_local_seq, asof_avail_ts_ms,
      features (dict FEATURE_NAMES -> float|None), quality_bits.

    Each observation is emitted at the `avail_ts_ms` of a decision event
    (`decision_kinds`) that has a `market_cid`, and only sees events with
    `local_seq <= asof_local_seq` AND `avail_ts_ms <= decision_ts_ms`.
    """
    decision_kinds = decision_kinds or DECISION_KINDS_DEFAULT
    ordered = sorted(events, key=lambda e: e["local_seq"])
    observations = []

    for i, ev in enumerate(ordered):
        kind, _ = _kind(ev)
        cid = _market_cid(ev, kind)
        if kind not in decision_kinds or cid is None:
            continue
        decision_ts = ev["avail_ts_ms"]
        seq_cut = ev["local_seq"]
        # Visible prefix: double cutoff seq + avail (contract A).
        visible = [
            e for e in ordered[: i + 1]
            if e["avail_ts_ms"] <= decision_ts and e["local_seq"] <= seq_cut
        ]

        states = {}
        spot = _SpotState()
        for e in visible:
            _apply_event(states, spot, e)

        st = states.get(cid, _MarketState())
        observations.append(_observation_from_state(st, spot, ev))
    return observations


def build_observations_incremental(events, decision_kinds=None,
                                   assume_sorted=False,
                                   decision_min_interval_ms=None):
    """Single-pass version of `build_observations` (T8b) — O(n) in
    applications instead of O(n²) — for real tapes with millions of
    events.

    `assume_sorted=True` trusts that `events` already arrives sorted by
    `local_seq` (e.g. `iter_events_jsonl` over a tape exported with
    seq by position) and allows passing a GENERATOR without materializing the
    list — the key to processing GB-sized tapes without OOM. It checks seq
    monotonicity as it consumes and raises `ValueError` if the tape
    lies.

    Keeps the causal state incrementally: events are walked
    in `local_seq` order and applied only when their `avail_ts_ms` is
    visible to the current decision (min-heap of pending by avail).
    At each decision the eligible pending events are drained and applied in
    seq order — identical to the batch builder's visible prefix when
    `avail_ts_ms` is monotonic in `local_seq` (a condition the
    exporter guarantees when assigning seq). Under avail/seq inversions it is still
    causal (nothing is applied before its avail) although the replay order
    may differ from the batch — the canonical contract is the double cutoff, not
    a specific application order.

    `decision_min_interval_ms` (real-tape closure): if given,
    observations are emitted at most every N ms PER MARKET — the suppressed
    decision event is still applied to the state when eligible
    (only the emission is skipped, never the application), so each
    emitted observation remains causally exact. Needed on
    high-frequency real tapes (~10⁶ deltas): without a time mesh the
    builder would materialize one observation per delta.

    Temporal regression: if a decision arrives with an `avail_ts_ms` lower than
    that of events already applied, the state contains information the
    decision should not see and it cannot be reverted — raises
    `NonMonotonicTape`. `build_observations_auto` encapsulates the fallback.
    """
    decision_kinds = decision_kinds or DECISION_KINDS_DEFAULT
    if assume_sorted:
        ordered = events
    else:
        ordered = sorted(events, key=lambda e: e["local_seq"])
    states = {}
    spot = _SpotState()
    pending = []          # heap (avail_ts_ms, local_seq, idx, event)
    max_avail_applied = None
    prev_seq = None
    last_emit = {}        # cid -> decision_ts of the last emitted obs
    observations = []

    for i, ev in enumerate(ordered):
        if assume_sorted:
            if prev_seq is not None and ev["local_seq"] < prev_seq:
                raise ValueError(
                    "assume_sorted: non-monotonic local_seq "
                    f"({prev_seq} -> {ev['local_seq']})")
            prev_seq = ev["local_seq"]
        heapq.heappush(
            pending, (ev["avail_ts_ms"], ev["local_seq"], i, ev))
        kind, _ = _kind(ev)
        cid = _market_cid(ev, kind)
        if kind not in decision_kinds or cid is None:
            continue
        decision_ts = ev["avail_ts_ms"]
        if (decision_min_interval_ms is not None
                and decision_ts - last_emit.get(cid, -2**63)
                < decision_min_interval_ms):
            # Time mesh: the event stays in `pending` and will be applied
            # when draining the next emitted decision — never lost.
            continue
        last_emit[cid] = decision_ts
        if max_avail_applied is not None and decision_ts < max_avail_applied:
            raise NonMonotonicTape(
                f"decision at seq {ev['local_seq']} with avail_ts_ms "
                f"{decision_ts} < max avail already applied {max_avail_applied} "
                "— use build_observations (batch) for this tape"
            )
        # Drain eligible pending events (avail <= T), applying them in
        # seq order — all those already pushed have local_seq <= that of `ev`.
        eligibles = []
        while pending and pending[0][0] <= decision_ts:
            eligibles.append(heapq.heappop(pending))
        eligibles.sort(key=lambda t: t[1])
        for _, _, _, e in eligibles:
            _apply_event(states, spot, e)
            a = e["avail_ts_ms"]
            if max_avail_applied is None or a > max_avail_applied:
                max_avail_applied = a

        st = states.get(cid, _MarketState())
        observations.append(_observation_from_state(st, spot, ev))
    return observations


def build_observations_auto(events, decision_kinds=None):
    """Incremental first; on `NonMonotonicTape` it falls back to the O(n²)
    batch builder, which processes any tape exactly. It is the entry point
    for tapes MATERIALIZED in memory: fast on well-ordered tapes
    and correct on those that are not. With `--streaming` (generators) there is no
    possible fallback — the batch needs to traverse the tape again and it is already
    consumed: train/evaluate wrap `NonMonotonicTape` with an
    actionable message in that case."""
    try:
        return build_observations_incremental(
            events, decision_kinds=decision_kinds)
    except NonMonotonicTape:
        return build_observations(events, decision_kinds=decision_kinds)


def build_labels(events, assume_sorted=False):
    """Confirmed labels per market (contract B).

    Only from `MarketMeta.outcome` (or a `resolution` event with winner +
    provenance). A market without an official result produces NO label —
    absence = pending/excluded, never invented. `assume_sorted=True`
    accepts a generator in seq order without materializing it (T8b).
    """
    labels = {}
    if not assume_sorted:
        events = sorted(events, key=lambda e: e["local_seq"])
    for ev in events:
        kind, _ = _kind(ev)
        if kind == "market_meta":
            payload = ev.get("payload") or {}
            oc = payload.get("outcome")
            if oc and oc.get("winner") and oc.get("provenance"):
                labels[payload.get("market_cid") or ev.get("market_cid")] = {
                    "winner": oc["winner"],
                    "known_at_ms": oc.get("known_at_ms"),
                    "known_at_source": "payload",
                    "provenance": oc["provenance"],
                }
        elif kind == "resolution":
            payload = ev.get("payload") or {}
            if payload.get("winner") and payload.get("provenance"):
                known = payload.get("known_at_ms")
                source = "payload"
                if known is None:
                    # The event ITSELF is the availability record:
                    # its `avail_ts_ms` is a real tape timestamp
                    # (the instant the resolution reached the recorder) —
                    # an honest fallback, never a fabricated timestamp
                    # (e.g. clear_at*1000, forbidden).
                    known = ev.get("avail_ts_ms")
                    source = "avail_ts_ms"
                labels[ev.get("market_cid")] = {
                    "winner": payload["winner"],
                    "known_at_ms": known,
                    "known_at_source": source,
                    "provenance": payload["provenance"],
                }
    return {k: v for k, v in labels.items() if k}


def attach_labels(observations, labels):
    """Adds `label` (1 = YES wins, 0 = NO) to every observation with a
    confirmed label. Observations of markets without an outcome stay without a
    label (excludable from supervised training)."""
    out = []
    for o in observations:
        lab = labels.get(o["market_cid"])
        o = dict(o)
        o["label"] = (1 if lab["winner"] == "yes" else 0) if lab else None
        o["label_known_at_ms"] = lab["known_at_ms"] if lab else None
        o["label_known_at_source"] = (
            lab.get("known_at_source") if lab else None)
        o["label_provenance"] = lab["provenance"] if lab else None
        out.append(o)
    return out


def build_meta_index(events, assume_sorted=False):
    """Index `market_cid → meta` from the tape's `market_meta` events.

    Returns per cid: slug, yes_token_id, no_token_id, asset, clear_at,
    market_open_ts and reference_price — what is needed to annotate shadow
    decision rows (slug/token of the chosen side) without re-deriving the
    state. `assume_sorted=True` accepts a generator in seq order."""
    if not assume_sorted:
        events = sorted(events, key=lambda e: e["local_seq"])
    metas = {}
    for ev in events:
        kind, _ = _kind(ev)
        if kind != "market_meta":
            continue
        payload = ev.get("payload") or {}
        cid = payload.get("market_cid") or ev.get("market_cid")
        if not cid:
            continue
        metas[cid] = {
            "slug": payload.get("slug") or ev.get("slug"),
            "yes_token_id": payload.get("yes_token_id"),
            "no_token_id": payload.get("no_token_id"),
            "asset": payload.get("asset") or ev.get("asset"),
            "clear_at": payload.get("clear_at"),
            "market_open_ts": payload.get("market_open_ts"),
            "reference_price": payload.get("reference_price"),
        }
    return metas


# ── Certified temporal split (closure B1) ─────────────────────────────────────
#
# Replaces the old `chronological_market_split`, which ordered markets
# by first timestamp and assigned the first `train_frac` to train: a
# market with early AND late observations dragged into train
# observations LATER than those of test (real overlap measured in
# T8b: 141.43 min). The new semantics:
#
# - Explicit GLOBAL UTC boundaries (`train_end_ms`, `val_end_ms`) over
#   `decision_ts_ms` — never a percentage of markets.
# - Embargo between partitions `EMBARGO_DEFAULT_MS = FLOW_WINDOW_MS`: it is
#   longest backward-looking window with a fixed horizon that the
#   features use (trade flow of the last 60 s). The
#   TARGET horizon (until the market closes) is not covered by an embargo but by
#   two strict rules: (a) a market whose observations cross a
#   boundary is PURGED whole — never split; (b) an observation only
#   trains if its `label_known_at_ms` already existed at the cutoff of its
#   partition (`filter_labels_available`).
# - Temporal zones: train = ts <= train_end; embargo₁ = (train_end,
#   train_end + embargo]; validation = (embargo₁, val_end]; embargo₂ =
#   (val_end, val_end + embargo]; test = ts > last boundary + embargo.
#   A market with observations in >1 zone is purged; one entirely in an
#   embargo zone is excluded without entering any partition.
# - The same market never in two partitions (invariant by construction:
#   the zone is decided per market, not per observation).

EMBARGO_DEFAULT_MS = FLOW_WINDOW_MS

# Canonical zone order (for deterministic purge reasons).
_ZONE_ORDER = ("train", "embargo_train", "validation", "embargo_val", "test")


def boundary_from_fraction(observations, frac):
    """`frac` quantile of `decision_ts_ms` over the given observations —
    deterministic and invariant to row permutation."""
    ts = sorted(o["decision_ts_ms"] for o in observations)
    if not ts:
        raise ValueError(
            "empty dataset: temporal boundaries cannot be derived")
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac outside (0,1]: {frac}")
    return ts[max(0, min(len(ts) - 1, math.ceil(len(ts) * frac) - 1))]


def boundaries_from_fractions(observations, train_frac=0.6, val_frac=0.0):
    """Derives GLOBAL temporal boundaries as quantiles of
    `decision_ts_ms` over ALL observations (not of markets by
    first timestamp — that was the bug). Deterministic and invariant to
    row permutation. Returns `(train_end_ms, val_end_ms|None)`.

    The fractions are applied over the temporal MASS of observations:
    `train_end` is the `train_frac` quantile of decision time and
    `val_end` the `train_frac + val_frac` quantile."""
    train_end = boundary_from_fraction(observations, train_frac)
    val_end = (
        boundary_from_fraction(
            observations, min(1.0, train_frac + val_frac))
        if val_frac > 0 else None
    )
    if val_end is not None and val_end <= train_end:
        # The validation fraction adds no new temporal mass: without a
        # real window between boundaries there is no validation — degrading to
        # two partitions is preferable to an empty window.
        val_end = None
    return train_end, val_end


def temporal_split(observations, train_end_ms, val_end_ms=None,
                   embargo_ms=None):
    """Certified temporal split per market with global UTC boundaries.

    Returns `(partitions, meta)`:
    - `partitions`: {"train": [...], "validation": [...], "test": [...]}
      (validation empty if `val_end_ms` is None).
    - `meta`: boundaries, splits (declared windows of kind SplitSpec),
      `purged_markets` (markets that cross a boundary, with reason/zones/
      span), `embargo_markets` (markets entirely in an embargo zone)
      and observation counts per partition.

    Guaranteed invariants:
    - every train obs has decision_ts_ms <= train_end_ms; every
      test obs is > the corresponding boundary + embargo;
    - a market with obs in more than one zone is purged WHOLE from all the
      affected partitions (recorded in `meta["purged_markets"]`);
    - no market appears in two partitions.
    """
    embargo_ms = EMBARGO_DEFAULT_MS if embargo_ms is None else embargo_ms
    if embargo_ms < 0:
        raise ValueError(f"negative embargo_ms: {embargo_ms}")
    if val_end_ms is not None and val_end_ms <= train_end_ms:
        raise ValueError(
            f"val_end_ms ({val_end_ms}) <= train_end_ms ({train_end_ms})")

    embargo1_end = train_end_ms + embargo_ms
    embargo2_end = (
        val_end_ms + embargo_ms if val_end_ms is not None else None
    )

    def _zone(ts):
        if ts <= train_end_ms:
            return "train"
        if ts <= embargo1_end:
            return "embargo_train"
        if val_end_ms is None:
            return "test"
        if ts <= val_end_ms:
            return "validation"
        if ts <= embargo2_end:
            return "embargo_val"
        return "test"

    by_market = {}
    for o in observations:
        by_market.setdefault(o["market_cid"], []).append(o)

    partitions = {"train": [], "validation": [], "test": []}
    purged = []
    embargo_markets = []
    for cid in sorted(by_market):
        obs_m = by_market[cid]
        ts_m = [o["decision_ts_ms"] for o in obs_m]
        zones = {_zone(t) for t in ts_m}
        if len(zones) > 1:
            sorted_zones = sorted(
                zones, key=lambda z: _ZONE_ORDER.index(z))
            purged.append({
                "market_cid": cid,
                "reason": "crosses_boundary:" + "+".join(sorted_zones),
                "zones": sorted_zones,
                "span_ms": [min(ts_m), max(ts_m)],
                "n_obs": len(obs_m),
            })
            continue
        z = next(iter(zones))
        if z.startswith("embargo"):
            embargo_markets.append(cid)
            continue
        partitions[z].extend(obs_m)

    def _span(obs):
        if not obs:
            return None
        ts = [o["decision_ts_ms"] for o in obs]
        return [min(ts), max(ts)]

    # Declared windows (SplitSpec): zone edges, not the min/max of what
    # survived — the manifest declares the real boundary.
    test_start = (val_end_ms if val_end_ms is not None
                  else train_end_ms) + embargo_ms
    splits = [{"name": "train",
               "from_ms": (_span(partitions["train"]) or [None])[0],
               "to_ms": train_end_ms}]
    if val_end_ms is not None:
        splits.append({"name": "validation",
                       "from_ms": embargo1_end,
                       "to_ms": val_end_ms})
    splits.append({"name": "test",
                   "from_ms": test_start,
                   "to_ms": (_span(partitions["test"]) or [None, None])[1]})

    meta = {
        "boundaries": {
            "train_end_ms": train_end_ms,
            "val_end_ms": val_end_ms,
            "embargo_ms": embargo_ms,
        },
        "splits": splits,
        "observed_span_ms": {
            n: _span(partitions[n]) for n in partitions
        },
        "purged_markets": purged,
        "embargo_markets": embargo_markets,
        "n_obs": {n: len(partitions[n]) for n in partitions},
        "n_embargo_obs": sum(
            len(by_market[c]) for c in embargo_markets),
    }
    # Self-certification: no market in two partitions; boundaries
    # respected by construction.
    mk = {n: {o["market_cid"] for o in partitions[n]} for n in partitions}
    assert not (mk["train"] & mk["validation"])
    assert not (mk["train"] & mk["test"])
    assert not (mk["validation"] & mk["test"])
    return partitions, meta


def filter_labels_available(observations, cutoff_ms):
    """Filters observations whose label was NOT known at `cutoff_ms`.

    An observation is only usable for TRAINING if `label_known_at_ms`
    exists and is <= the fit cutoff (the end of its partition's
    window). Returns `(usable, excluded)`; the excluded keep their
    original `label_known_at_ms` — never backdated (contract C).
    """
    usable, excluded = [], []
    for o in observations:
        known = o.get("label_known_at_ms")
        if known is not None and known <= cutoff_ms:
            usable.append(o)
        else:
            excluded.append(o)
    return usable, excluded


def classify_eval_partitions(observations, model_doc):
    """Classifies observations of an EVALUATION tape according to the split
    frozen in `model_doc` (artifact of train.py).

    An UNKNOWN market does not automatically fall into test:
    - if the artifact declares `boundaries`, only a market whose temporal
      span is STRICTLY later than the test boundary
      (end of validation/train + embargo) enters test; one earlier than the
      end of train, overlapping train/validation/embargo or that would have been
      purged, is classified `out_of_scope`;
    - old artifacts without `boundaries`: legacy fallback (known in
      `split_markets.train` → train; the rest → test), marked
      `mode="legacy_split_markets"` in the returned meta.

    Returns `(partitions, meta)` with partitions including the `out_of_scope`
    bucket and meta with `market_partition` (cid → partition) and
    `out_of_scope_reasons` (cid → reason).
    """
    split_mk = model_doc.get("split_markets") or {}
    boundaries = model_doc.get("boundaries")
    train_set = set(split_mk.get("train", []))
    val_set = set(split_mk.get("validation", []))
    test_set = set(split_mk.get("test", []))
    purged_set = set(split_mk.get("purged", []))

    by_market = {}
    for o in observations:
        by_market.setdefault(o["market_cid"], []).append(o)

    partitions = {"train": [], "validation": [], "test": [],
                  "out_of_scope": []}
    market_partition = {}
    oos_reasons = {}
    embargo = boundaries.get("embargo_ms", 0) if boundaries else 0
    train_end = boundaries["train_end_ms"] if boundaries else None
    test_start = (
        (boundaries["val_end_ms"] or boundaries["train_end_ms"]) + embargo
        if boundaries else None
    )

    for cid in sorted(by_market):
        obs_m = by_market[cid]
        if cid in train_set:
            dest = "train"
        elif cid in val_set:
            dest = "validation"
        elif cid in test_set:
            dest = "test"
        elif cid in purged_set:
            dest = "out_of_scope"
            oos_reasons[cid] = "purged_in_training"
        elif test_start is None:
            # Legacy: without declared boundaries the scope cannot be certified —
            # historical behavior (unknown → test).
            dest = "test"
        else:
            ts_m = [o["decision_ts_ms"] for o in obs_m]
            a, b = min(ts_m), max(ts_m)
            if a > test_start:
                dest = "test"  # strictly later than the boundary
            elif b <= train_end:
                dest = "out_of_scope"
                oos_reasons[cid] = "before_train_end"
            else:
                dest = "out_of_scope"
                oos_reasons[cid] = "overlaps_boundary_or_embargo"
        market_partition[cid] = dest
        partitions[dest].extend(obs_m)

    meta = {
        "mode": ("boundaries" if boundaries
                 else "legacy_split_markets"),
        "market_partition": market_partition,
        "out_of_scope_reasons": oos_reasons,
        "n_obs": {n: len(partitions[n]) for n in partitions},
    }
    return partitions, meta


class FeatureScaler:
    """Z-score scaling + median imputation fitted ONLY on train.

    `fit` fixes means/deviations/medians; `transform` mutates nothing —
    test parameters never touch the scaler (causality RED).
    Features absent in ALL of train: mean/median 0, std 1 (identity
    after imputing 0) — the absence stays explicit in `missing_in_train`.
    """

    def __init__(self):
        self.means_ = {}
        self.stds_ = {}
        self.medians_ = {}
        self.missing_in_train = set()
        self.fitted = False

    def fit(self, observations):
        for feat in FEATURE_NAMES:
            vals = [
                o["features"][feat] for o in observations
                if o["features"].get(feat) is not None
            ]
            if vals:
                self.means_[feat] = sum(vals) / len(vals)
                self.stds_[feat] = (
                    statistics.pstdev(vals) if len(vals) >= 2 else 0.0
                )
                self.medians_[feat] = statistics.median(vals)
            else:
                self.missing_in_train.add(feat)
                self.means_[feat] = 0.0
                self.stds_[feat] = 1.0
                self.medians_[feat] = 0.0
        self.fitted = True
        return self

    def transform(self, observations):
        if not self.fitted:
            raise RuntimeError("FeatureScaler.transform before fit")
        rows = []
        for o in observations:
            row = []
            for feat in FEATURE_NAMES:
                v = o["features"].get(feat)
                if v is None:
                    v = self.medians_[feat]  # imputation: TRAIN median
                std = self.stds_[feat] or 1.0
                row.append((v - self.means_[feat]) / std)
            rows.append(row)
        return rows

    def to_dict(self):
        return {
            "means": self.means_, "stds": self.stds_,
            "medians": self.medians_,
            "missing_in_train": sorted(self.missing_in_train),
        }

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.means_ = dict(d["means"])
        s.stds_ = dict(d["stds"])
        s.medians_ = dict(d["medians"])
        s.missing_in_train = set(d.get("missing_in_train", []))
        s.fitted = True
        return s


def discounted_returns(rewards, gamma):
    """Backward discounted return, exactly ONCE:
    G_t = r_t + gamma·G_{t+1}. Closed form — the caller must not discount
    the result again (T8 RED: reward discounted only once)."""
    out = [0.0] * len(rewards)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out
