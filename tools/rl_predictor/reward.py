"""rl_predictor accounting reward.

net_pnl_delta = (equity_next − equity_prev)
              − (deposits_next − deposits_prev)
              + (withdrawals_next − withdrawals_prev)

equity = cash + CONSERVATIVE exit value of the positions: walks the bids and
their depth, deducts the estimated exit cost; the unliquidable remainder is
worth ZERO with a quality flag (never the original cost nor the optimistic
mid). Cash already includes the executed fees, so they are not subtracted twice.

No bonus for a profitable sell and no special penalty for a realized loss: the
reward is ONLY the flow-adjusted equity delta, so realized /
unrealized / costs are stored separately for auditing and are NEVER added
back into the reward.

Money in integer micro-USDC (the i128 equivalent of the Rust ledger) and
Decimal in entries; float only in the final normalization of the model.
"""

from decimal import ROUND_DOWN, Decimal

from .contract import EPISODE_WINDOW_MS

MICROS = Decimal(1_000_000)
SHARE_QUANTUM = Decimal("0.000001")  # quantum real de shares


def to_um(x):
    """USDC (Decimal/str/int) -> micro-USDC int, truncating like to_micros."""
    return int((Decimal(str(x)) * MICROS).quantize(
        Decimal(1), rounding=ROUND_DOWN))


def taker_fee_um(price, shares, fee_rate):
    """Parabolic taker fee of the bot: rate·p·(1−p) per share.

    price, shares, fee_rate: Decimal. Devuelve micro-USDC (truncate).
    The fee is kept separate from the negotiated price at full precision; it is
    NEVER folded into the price and truncated (that would erase the friction
    near 0/1)."""
    p = Decimal(str(price))
    s = Decimal(str(shares))
    r = Decimal(str(fee_rate))
    fee = r * p * (Decimal(1) - p) * s
    return to_um(fee)


class Position:
    __slots__ = ("shares", "cost_um")

    def __init__(self, shares=Decimal(0), cost_um=0):
        self.shares = Decimal(shares)
        self.cost_um = int(cost_um)

    def copy(self):
        return Position(self.shares, self.cost_um)


def exit_value_um(shares, bids, fee_rate=Decimal(0), min_size=Decimal(0)):
    """Conservative exit value of `shares` against the bid side.

    bids: iterable of (price Decimal, depth_shares Decimal), sorted best
    first. Returns (value_um, flags:set[str]).
    - walks the real depth; remainder without liquidity -> 0 + flag
    - deducts the estimated exit fee per executed slice
    - shares < min_size -> residual not admitted -> 0 + flag"""
    flags = set()
    remaining = Decimal(shares)
    if remaining <= 0:
        return 0, flags
    if min_size and remaining < Decimal(min_size):
        flags.add("residual_below_min_size")
        return 0, flags
    value_um = 0
    for price, depth in bids:
        if remaining <= 0:
            break
        p, d = Decimal(price), Decimal(depth)
        if d <= 0 or p <= 0:
            continue
        take = remaining if remaining <= d else d
        gross_um = to_um(take * p)
        fee_um = taker_fee_um(p, take, fee_rate)
        value_um += max(0, gross_um - fee_um)
        remaining -= take
    if remaining > 0:
        flags.add("residual_no_liquidity")
    return value_um, flags


def net_pnl_delta_um(prev, nxt):
    """Exact contractual formula over accounting snapshots."""
    return (nxt.equity_um - prev.equity_um) \
        - (nxt.deposits_um - prev.deposits_um) \
        + (nxt.withdrawals_um - prev.withdrawals_um)


def adjusted_equity_um(snap):
    """Flow-adjusted equity (base of the drawdown)."""
    return snap.equity_um - snap.deposits_um + snap.withdrawals_um


def normalized_reward(delta_um, *, budget_um, exposure_um=0,
                      dd_delta_um=0, elapsed_ms=0,
                      lambda_inventory=0.0, lambda_drawdown=0.0):
    """Normalized reward of the step (float; the only place with float).

    Lambdas >= 0, default 0 for the first accounting baseline; tuning is
    train/val only. elapsed_ms is clamped to the episode window."""
    if budget_um <= 0:
        raise ValueError("initial_episode_budget must be > 0")
    if lambda_inventory < 0 or lambda_drawdown < 0:
        raise ValueError("lambdas must be non-negative")
    elapsed_frac = min(max(float(elapsed_ms) / EPISODE_WINDOW_MS, 0.0), 1.0)
    base = delta_um / budget_um
    inv = lambda_inventory * (exposure_um / budget_um) * elapsed_frac
    dd = lambda_drawdown * max(0, dd_delta_um) / budget_um
    return base - inv - dd


class AccountSnapshot:
    """Accounting state at a decision instant."""

    __slots__ = ("equity_um", "deposits_um", "withdrawals_um",
                 "exposure_um", "cash_um", "positions_value_um",
                 "realized_um", "unrealized_um", "costs_um",
                 "quality_flags", "invalid_reasons", "dd_delta_um")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class EpisodeAccountant:
    """Episode ledger with a fixed virtual budget (infinite paper,
    separate from learned ruin; no auto-topup).

    It does not replace the Rust ledger at runtime; it reproduces ITS arithmetic
    for offline training and parity. Every transition feeds the reward through
    the equity delta; a top-up reward is never learned."""

    def __init__(self, initial_budget_um):
        if initial_budget_um <= 0:
            raise ValueError("initial_episode_budget must be > 0")
        self.budget_um = int(initial_budget_um)
        self.cash_um = int(initial_budget_um)
        self.positions = {}  # token_id -> Position
        self.deposits_um = 0
        self.withdrawals_um = 0
        self.realized_um = 0
        self.fees_paid_um = 0
        self.quality_flags = set()
        self.invalid_reasons = []
        self._settled = set()
        self._peak_adj_um = None
        self.drawdown_um = 0

    # ── flows (NOT reward) ──────────────────────────────────────

    def deposit(self, amount_um):
        self.cash_um += int(amount_um)
        self.deposits_um += int(amount_um)

    def withdraw(self, amount_um):
        self.cash_um -= int(amount_um)
        self.withdrawals_um += int(amount_um)

    # ── confirmed executions (never by emitting an intent) ───────────

    def apply_buy(self, token_id, shares, price, fee_rate):
        shares = Decimal(shares)
        price = Decimal(price)
        if shares <= 0 or price <= 0 or price >= 1:
            raise ValueError("invalid buy: shares/price out of range")
        notional_um = to_um(shares * price)
        fee_um = taker_fee_um(price, shares, fee_rate)
        cost = notional_um + fee_um
        # No balance clamp: if it is overdrawn it stays negative and visible.
        self.cash_um -= cost
        self.fees_paid_um += fee_um
        pos = self.positions.setdefault(str(token_id), Position())
        pos.shares += shares
        pos.cost_um += cost
        return cost

    def apply_sell(self, token_id, shares, price, fee_rate):
        shares = Decimal(shares)
        price = Decimal(price)
        pos = self.positions.get(str(token_id))
        if pos is None or pos.shares <= 0:
            raise ValueError("sell without an available own position")
        if shares <= 0 or shares > pos.shares:
            raise ValueError("invalid sell shares")
        gross_um = to_um(shares * price)
        fee_um = taker_fee_um(price, shares, fee_rate)
        proceeds = gross_um - fee_um
        self.cash_um += proceeds
        self.fees_paid_um += fee_um
        # cost removed: ONE single number, whole (floor) pro-rating; the
        # remainder keeps the rest, never a recomputed avg_price.
        size_um = int((pos.shares * MICROS).to_integral_value(ROUND_DOWN))
        sold_um = int((shares * MICROS).to_integral_value(ROUND_DOWN))
        released = (pos.cost_um * sold_um) // size_um
        pos.shares -= shares
        pos.cost_um -= released
        if pos.shares == 0:
            pos.cost_um = 0
        self.realized_um += proceeds - released
        return proceeds, released

    def apply_settlement(self, token_id, won):
        """Official payout, booked ONCE per token/episode."""
        token_id = str(token_id)
        if token_id in self._settled:
            raise ValueError("duplicate settlement (paid once)")
        self._settled.add(token_id)
        pos = self.positions.get(token_id)
        if pos is None or pos.shares <= 0:
            return 0
        payout = to_um(pos.shares) if won else 0
        self.cash_um += payout
        self.realized_um += payout - pos.cost_um
        pos.shares = Decimal(0)
        pos.cost_um = 0
        return payout

    # ── snapshots y reward ──────────────────────────────────────────

    def position_tokens(self):
        return [t for t, p in self.positions.items() if p.shares > 0]

    def exposure_um(self):
        """Exposure = live residual cost (capital at risk)."""
        return sum(p.cost_um for p in self.positions.values()
                   if p.shares > 0)

    def snapshot(self, bids_by_token=None, fee_rate=Decimal(0),
                 min_size=Decimal(0)):
        """Marks the state: conservative equity + audit breakdown.

        bids_by_token: {token_id: [(price, depth_shares), ...]}. A token
        with a position and no book is marked residual_no_liquidity -> 0."""
        bids_by_token = bids_by_token or {}
        pv_um = 0
        live_cost_um = 0
        for token, pos in self.positions.items():
            if pos.shares <= 0:
                continue
            live_cost_um += pos.cost_um
            v, flags = exit_value_um(pos.shares, bids_by_token.get(token, []),
                                     fee_rate, min_size)
            pv_um += v
            self.quality_flags |= {f"{token}:{f}" for f in flags}
        snap = AccountSnapshot(
            equity_um=self.cash_um + pv_um,
            deposits_um=self.deposits_um,
            withdrawals_um=self.withdrawals_um,
            exposure_um=self.exposure_um(),
            cash_um=self.cash_um,
            positions_value_um=pv_um,
            realized_um=self.realized_um,
            unrealized_um=pv_um - live_cost_um,
            costs_um=self.fees_paid_um,
            quality_flags=set(self.quality_flags),
            invalid_reasons=list(self.invalid_reasons),
        )
        adj = adjusted_equity_um(snap)
        if self._peak_adj_um is None or adj > self._peak_adj_um:
            self._peak_adj_um = adj
        prev_dd = self.drawdown_um
        self.drawdown_um = max(0, self._peak_adj_um - adj)
        snap.dd_delta_um = self.drawdown_um - prev_dd
        return snap

    def mark_invalid(self, reason):
        """Episode excluded from learning with a reason (stale book, gap,
        cash ownership ambiguous, truncated). Kept for diagnostics."""
        self.invalid_reasons.append(str(reason))

    @property
    def trainable(self):
        return not self.invalid_reasons
