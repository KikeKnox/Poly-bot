"""Accounting reward: portfolio delta adjusted for flows, conservative exit,
no double counting, drawdown over flow-adjusted equity."""

import unittest
from decimal import Decimal

from rl_predictor.reward import (
    EpisodeAccountant,
    adjusted_equity_um,
    exit_value_um,
    net_pnl_delta_um,
    normalized_reward,
    taker_fee_um,
    to_um,
)

D = Decimal


class TestExitValue(unittest.TestCase):
    def test_walks_depth(self):
        # 10 shares against bids of 4@0.60 and 4@0.50: sells 8, remainder 0.
        v, flags = exit_value_um(D(10), [(D("0.60"), D(4)),
                                         (D("0.50"), D(4))])
        self.assertEqual(v, to_um(D("2.4") + D("2.0")))
        self.assertIn("residual_no_liquidity", flags)

    def test_no_liquidity_zero_flagged(self):
        v, flags = exit_value_um(D(5), [])
        self.assertEqual(v, 0)
        self.assertIn("residual_no_liquidity", flags)

    def test_below_min_size_zero(self):
        v, flags = exit_value_um(D("0.5"), [(D("0.9"), D(10))],
                                 min_size=D(1))
        self.assertEqual(v, 0)
        self.assertIn("residual_below_min_size", flags)

    def test_fee_subtracted_not_folded(self):
        # p=0.95 rate=0.02: fee = .02*.95*.05 = 0.00095/share — the tick
        # CANNOT eat it because the fee is charged separately.
        v, _ = exit_value_um(D(1), [(D("0.95"), D(10))],
                             fee_rate=D("0.02"))
        self.assertEqual(v, to_um(D("0.95")) -
                         taker_fee_um(D("0.95"), D(1), D("0.02")))
        self.assertEqual(taker_fee_um(D("0.95"), D(1), D("0.02")), 950)


class TestAccounting(unittest.TestCase):
    def test_buy_sell_reward_equals_equity(self):
        """Sum of rewards (lambdas=0) equals the flow-adjusted equity change.
        Buy 4@0.5 and sell 4@0.6."""
        acc = EpisodeAccountant(to_um(D(10)))
        s0 = acc.snapshot()
        acc.apply_buy("tokY", D(4), D("0.5"), D("0.02"))
        bids = {"tokY": [(D("0.6"), D(100))]}
        s1 = acc.snapshot(bids)
        r1 = net_pnl_delta_um(s0, s1)
        acc.apply_sell("tokY", D(4), D("0.6"), D("0.02"))
        s2 = acc.snapshot()
        r2 = net_pnl_delta_um(s1, s2)
        # Final equity = cash (position closed).
        self.assertEqual(s2.equity_um, acc.cash_um)
        # The sum of the deltas reproduces the total change since s0.
        self.assertEqual(r1 + r2, s2.equity_um - s0.equity_um)
        # And there was a net profit (4×0.1 minus parabolic fees).
        self.assertGreater(r1 + r2, 0)
        # Separate audit: realized >0, not re-added to the reward.
        self.assertGreater(acc.realized_um, 0)

    def test_deposit_not_reward(self):
        acc = EpisodeAccountant(to_um(D(10)))
        s0 = acc.snapshot()
        acc.deposit(to_um(D(1000)))  # paper top-up
        s1 = acc.snapshot()
        self.assertEqual(net_pnl_delta_um(s0, s1), 0)

    def test_withdrawal_not_loss(self):
        acc = EpisodeAccountant(to_um(D(10)))
        s0 = acc.snapshot()
        acc.withdraw(to_um(D(4)))
        s1 = acc.snapshot()
        self.assertEqual(net_pnl_delta_um(s0, s1), 0)

    def test_sell_no_double_count(self):
        """Selling at a profit gives NO bonus on top of the value: the reward of the
        sell step can even be ~0 if the mark already reflected the value."""
        acc = EpisodeAccountant(to_um(D(10)))
        acc.apply_buy("tokY", D(4), D("0.5"), D(0))
        s1 = acc.snapshot({"tokY": [(D("0.6"), D(10))]})
        acc.apply_sell("tokY", D(4), D("0.6"), D(0))
        s2 = acc.snapshot()
        # The profit was already in s1's equity (0.6 bid available);
        # executing the sell only realizes what was marked: delta ≈ 0.
        self.assertAlmostEqual(net_pnl_delta_um(s1, s2), 0, delta=2)

    def test_partial_sell_cost_prorated(self):
        acc = EpisodeAccountant(to_um(D(10)))
        acc.apply_buy("tokY", D(4), D("0.5"), D(0))
        cost0 = acc.positions["tokY"].cost_um
        acc.apply_sell("tokY", D(2), D("0.6"), D(0))
        pos = acc.positions["tokY"]
        self.assertEqual(pos.shares, D(2))
        self.assertEqual(pos.cost_um, cost0 // 2)  # pro-rated split

    def test_settlement_once(self):
        acc = EpisodeAccountant(to_um(D(10)))
        acc.apply_buy("tokY", D(4), D("0.5"), D(0))
        payout = acc.apply_settlement("tokY", won=True)
        self.assertEqual(payout, to_um(D(4)))
        with self.assertRaises(ValueError):
            acc.apply_settlement("tokY", won=True)

    def test_drawdown_on_flow_adjusted_equity(self):
        """A deposit does NOT cure drawdown: it is adjusted for flows."""
        acc = EpisodeAccountant(to_um(D(10)))
        s0 = acc.snapshot()  # adj = 10
        acc.apply_buy("tokY", D(5), D("0.5"), D(0))
        s1 = acc.snapshot({"tokY": [(D("0.2"), D(10))]})  # price falls
        adj1 = adjusted_equity_um(s1)
        self.assertGreater(acc.drawdown_um, 0)
        dd_before = acc.drawdown_um
        acc.deposit(to_um(D(100)))  # equity rises, adj_equity too...
        s2 = acc.snapshot({"tokY": [(D("0.2"), D(10))]})
        # adj(equity) = equity - deposits: the deposit is neutralized.
        self.assertEqual(acc.drawdown_um, dd_before)


class TestNormalizedReward(unittest.TestCase):
    def test_base_and_lambdas(self):
        b = to_um(D(10))
        r = normalized_reward(to_um(D(1)), budget_um=b)
        self.assertAlmostEqual(r, 0.1)
        r_inv = normalized_reward(
            0, budget_um=b, exposure_um=to_um(D(5)), elapsed_ms=150_000,
            lambda_inventory=0.1)
        self.assertAlmostEqual(r_inv, -0.1 * 0.5 * 0.5)
        r_dd = normalized_reward(0, budget_um=b,
                                 dd_delta_um=to_um(D(2)),
                                 lambda_drawdown=0.5)
        self.assertAlmostEqual(r_dd, -0.5 * 0.2)

    def test_guards(self):
        with self.assertRaises(ValueError):
            normalized_reward(0, budget_um=0)
        with self.assertRaises(ValueError):
            normalized_reward(0, budget_um=1, lambda_inventory=-0.1)

    def test_invalid_marks(self):
        acc = EpisodeAccountant(to_um(D(5)))
        acc.mark_invalid("book_stale")
        self.assertFalse(acc.trainable)
        snap = acc.snapshot()
        self.assertIn("book_stale", snap.invalid_reasons)


if __name__ == "__main__":
    unittest.main()
