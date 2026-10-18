//! Net directional exposure ledger per bracket.
//!
//! Several strategies may trade the same bracket by design:
//! this ledger does NOT cap total capital per bracket, it caps **same-side
//! stacking**: the net directional exposure `|YES − NO|` in USDC.
//! A covered pair (ARB) adds to both sides and nets to ≈ 0, and a leg that
//! covers the other side REDUCES the net — hedges are never blocked.
//!
//! `max_usdc == 0` disables the cap, but accounting stays live: it feeds
//! the concentration metric (`hrf_bracket_net_exposure_usdc`) and is the
//! basis for the executor's exposure cache.
//!
//! The core primitive is `try_add_exposure`: atomic under a single lock,
//! it avoids the TOCTOU a separate "query + insert" sequence would have.

use parking_lot::Mutex;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;

use crate::types::StrategyId;

struct BracketEntry {
    clear_at: i64,
    /// token_id → (owner → committed USDC). The net directional exposure and the
    /// caps are computed over the per-token TOTAL (sum of owners) — shared
    /// by design. What is NOT shared is the claim: a `release` only
    /// deducts from the owner that contributed it (previously the bucket
    /// was fungible and a release by A could eat B's exposure, with the
    /// clamp to 0 erasing the trace).
    per_token: HashMap<String, HashMap<StrategyId, Decimal>>,
}

impl BracketEntry {
    /// Net directional exposure: larger side minus the rest. With the two tokens of an
    /// up/down bracket it equals `|YES − NO|`; with a single side, that side.
    fn net(&self) -> Decimal {
        let total: Decimal = self.token_totals().sum();
        let max = self.token_totals().max().unwrap_or_default();
        max - (total - max)
    }

    fn token_totals(&self) -> impl Iterator<Item = Decimal> + '_ {
        self.per_token
            .values()
            .map(|m| m.values().copied().sum::<Decimal>())
    }
}

/// Directional exposure accumulated per bracket, shared across strategies.
pub struct BracketLedger {
    inner: Mutex<HashMap<String, BracketEntry>>,
}

impl BracketLedger {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(HashMap::new()),
        })
    }

    /// Tries to commit `amount` USDC on `token_id` within the bracket,
    /// attributed to `owner`.
    ///
    /// - Returns `Some(resulting_net)` if accepted (and records it).
    /// - Returns `None` if `max_usdc > 0` and the resulting net would exceed it.
    /// - `max_usdc == 0` = cap disabled: always accepts, but still accounts.
    pub fn try_add_exposure(
        &self,
        cid: &str,
        clear_at: i64,
        token_id: &str,
        amount: Decimal,
        max_usdc: Decimal,
        owner: StrategyId,
    ) -> Option<Decimal> {
        self.try_add_many(cid, clear_at, &[(token_id, amount, owner)], max_usdc)
    }

    /// Atomic variant for YES+NO pairs (ARB): checks the net with BOTH
    /// legs applied — avoids the first leg alone rejecting a covered pair.
    /// Each leg carries its own owner (in practice the same strategy,
    /// but the claim is attributed separately).
    pub fn try_add_pair_exposure(
        &self,
        cid: &str,
        clear_at: i64,
        leg1: (&str, Decimal, StrategyId),
        leg2: (&str, Decimal, StrategyId),
        max_usdc: Decimal,
    ) -> Option<Decimal> {
        self.try_add_many(cid, clear_at, &[leg1, leg2], max_usdc)
    }

    /// Like `try_add_pair_exposure`, but each leg carries its own bracket
    /// (`cid`, `clear_at`): ARB-15M5M buys in TWO markets. With the same `cid`
    /// it delegates to the usual path. With different cids the cap is evaluated per
    /// bracket and the pair is accepted or rejected as a whole under a single lock; no
    /// entry is created if it is rejected. Returns the largest resulting net.
    pub fn try_add_pair_exposure_per_market(
        &self,
        leg1: (&str, i64, &str, Decimal, StrategyId),
        leg2: (&str, i64, &str, Decimal, StrategyId),
        max_usdc: Decimal,
    ) -> Option<Decimal> {
        if leg1.0 == leg2.0 {
            return self.try_add_pair_exposure(
                leg1.0,
                leg1.1,
                (leg1.2, leg1.3, leg1.4),
                (leg2.2, leg2.3, leg2.4),
                max_usdc,
            );
        }
        let mut map = self.inner.lock();
        let mut net_max = Decimal::ZERO;
        for (cid, _, token, amount, _) in [leg1, leg2] {
            let mut totals: HashMap<String, Decimal> = map
                .get(cid)
                .map(|e| {
                    e.per_token
                        .iter()
                        .map(|(t, m)| (t.clone(), m.values().copied().sum()))
                        .collect()
                })
                .unwrap_or_default();
            *totals.entry(token.to_owned()).or_default() += amount;
            let total: Decimal = totals.values().copied().sum();
            let max = totals.values().copied().max().unwrap_or_default();
            let net = max - (total - max);
            if max_usdc > Decimal::ZERO && net > max_usdc {
                return None;
            }
            net_max = net_max.max(net);
        }
        for (cid, clear_at, token, amount, owner) in [leg1, leg2] {
            let entry = map.entry(cid.to_owned()).or_insert_with(|| BracketEntry {
                clear_at,
                per_token: HashMap::new(),
            });
            *entry
                .per_token
                .entry(token.to_owned())
                .or_default()
                .entry(owner)
                .or_default() += amount;
        }
        Some(net_max)
    }

    fn try_add_many(
        &self,
        cid: &str,
        clear_at: i64,
        legs: &[(&str, Decimal, StrategyId)],
        max_usdc: Decimal,
    ) -> Option<Decimal> {
        let mut map = self.inner.lock();
        // If the bracket did NOT exist, `or_insert_with` creates it with
        // THIS caller's `clear_at` even if the attempt ends up rejected —
        // leaving an empty phantom entry whose `clear_at` governs the
        // bracket for any later VALID caller (e.g. the synthetic one
        // instead of the real close). `existed` tells both cases apart
        // so the creation can be undone on rejection.
        let existed = map.contains_key(cid);
        let entry = map.entry(cid.to_owned()).or_insert_with(|| BracketEntry {
            clear_at,
            per_token: HashMap::new(),
        });
        // Prospective net with all the legs applied, without mutating yet.
        // The prospect is computed over per-token TOTALS (the directional cap
        // is shared); per-owner attribution is only materialized on
        // acceptance.
        let mut prospect: HashMap<String, Decimal> = entry
            .per_token
            .iter()
            .map(|(t, m)| (t.clone(), m.values().copied().sum()))
            .collect();
        for (token, amount, _) in legs {
            *prospect.entry((*token).to_owned()).or_default() += *amount;
        }
        let total: Decimal = prospect.values().copied().sum();
        let max = prospect.values().copied().max().unwrap_or_default();
        let net = max - (total - max);
        if max_usdc > Decimal::ZERO && net > max_usdc {
            // If the entry was created NOW (new bracket) and is
            // rejected, remove it — don't leave a phantom entry whose
            // `clear_at` from a rejected caller overrides that of a later
            // valid caller. If it already existed, leave it untouched (no mutation).
            if !existed {
                map.remove(cid);
            }
            return None;
        }
        for (token, amount, owner) in legs {
            *entry
                .per_token
                .entry((*token).to_owned())
                .or_default()
                .entry(*owner)
                .or_default() += *amount;
        }
        Some(net)
    }

    /// Releases a token's exposure (sell, GTD cancel, router rejection)
    /// deducting ONLY from `owner`'s bucket.
    ///
    /// Returns the NON-released remainder: if the owner has less than `amount`
    /// recorded, their share is released and the excess is NOT deducted from other
    /// strategies' claims (previously the fungible bucket + clamp to 0 hid it).
    /// The remainder is logged with `warn` to make the drift visible.
    pub fn release_exposure(
        &self,
        cid: &str,
        token_id: &str,
        amount: Decimal,
        owner: StrategyId,
    ) -> Decimal {
        let mut map = self.inner.lock();
        let Some(entry) = map.get_mut(cid) else {
            return amount;
        };
        let Some(owners) = entry.per_token.get_mut(token_id) else {
            return amount;
        };
        let Some(exp) = owners.get_mut(&owner) else {
            tracing::warn!(
                cid,
                token_id,
                owner = %owner,
                amount = %amount,
                "ledger: release without owner claim — other owners' exposure untouched"
            );
            return amount;
        };
        let released = amount.min(*exp);
        *exp -= released;
        if *exp == Decimal::ZERO {
            owners.remove(&owner);
        }
        let shortfall = amount - released;
        if shortfall > Decimal::ZERO {
            tracing::warn!(
                cid,
                token_id,
                owner = %owner,
                amount = %amount,
                released = %released,
                "ledger: release exceeds owner claim — remainder not released"
            );
        }
        shortfall
    }

    /// Adds exposure without checking the cap, attributed to `owner` — only for
    /// rebuilding from the DB at startup / periodic resync.
    pub fn force_add_exposure(
        &self,
        cid: &str,
        clear_at: i64,
        token_id: &str,
        amount: Decimal,
        owner: StrategyId,
    ) {
        let mut map = self.inner.lock();
        let entry = map.entry(cid.to_owned()).or_insert_with(|| BracketEntry {
            clear_at,
            per_token: HashMap::new(),
        });
        *entry
            .per_token
            .entry(token_id.to_owned())
            .or_default()
            .entry(owner)
            .or_default() += amount;
    }

    /// Purges brackets whose `clear_at` has passed. Also releases residual claims
    /// (FOK without fill, partial FAKs): once the bracket resolves they no longer matter.
    ///
    /// Design note (audit finding, not a code bug):
    /// this removes already-closed brackets from the ledger EVEN with open
    /// unredeemed positions — that capital stops counting against
    /// `max_total_exposure_usdc` without appearing in `wallet_bal_rx` either. With
    /// `auto_redeem=false` this accumulates (the global cap becomes looser
    /// than it should be while there are unredeemed positions). Not fixed
    /// here: resizing the ledger to include closed positions
    /// pending redeem is a larger redesign, out of scope for a point
    /// fix — the limitation and its trigger condition are documented.
    pub fn purge_expired(&self, now_ts: i64) {
        let mut map = self.inner.lock();
        if map.is_empty() {
            return;
        }
        map.retain(|_, v| now_ts < v.clear_at);
    }

    /// Current net directional exposure of the bracket (0 if it doesn't exist).
    pub fn net_exposure(&self, cid: &str) -> Decimal {
        self.inner
            .lock()
            .get(cid)
            .map(|e| e.net())
            .unwrap_or_default()
    }

    /// GROSS exposure of a bracket: sum of all the cid's `per_token` entries,
    /// WITHOUT netting (unlike `net_exposure`). RF-2 uses it for
    /// bracket exclusivity against PM/MM: a balanced
    /// box (YES=NO) gives `net_exposure == 0` but ANOTHER strategy IS
    /// trading that bracket — `net_exposure` alone would let RF post
    /// on top of an already-covered ParityMaker box.
    pub fn gross_exposure(&self, cid: &str) -> Decimal {
        self.inner
            .lock()
            .get(cid)
            .map(|e| e.token_totals().sum())
            .unwrap_or_default()
    }

    /// Total gross exposure (sum over all tokens of all brackets).
    /// It is the source for the executor's global exposure check: equivalent to
    /// in-flight GTDs + open positions without paying for a SQLite query per order.
    pub fn total_exposure(&self) -> Decimal {
        self.inner
            .lock()
            .values()
            .flat_map(|e| e.token_totals())
            .sum()
    }

    /// Claim attributed to a specific `owner` on a token — inspection surface
    /// for tests and for diagnosing the over-release warning.
    #[cfg(test)]
    pub fn owner_exposure(&self, cid: &str, token_id: &str, owner: StrategyId) -> Decimal {
        self.inner
            .lock()
            .get(cid)
            .and_then(|e| e.per_token.get(token_id))
            .and_then(|m| m.get(&owner))
            .copied()
            .unwrap_or_default()
    }

    /// Clears the ledger — only for the periodic resync from the DB.
    pub fn clear(&self) {
        self.inner.lock().clear();
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn two_market_pair_is_claimed_and_released_by_cid() {
        let l = BracketLedger::new();
        let s = StrategyId::Arb15m5m;
        let net = l
            .try_add_pair_exposure_per_market(
                ("c15", 100, "up15", dec!(4), s),
                ("c5", 100, "down5", dec!(4), s),
                dec!(10),
            )
            .expect("fits");
        assert_eq!(net, dec!(4));
        assert_eq!(l.owner_exposure("c15", "up15", s), dec!(4));
        assert_eq!(l.owner_exposure("c5", "down5", s), dec!(4));
        l.release_exposure("c15", "up15", dec!(4), s);
        l.release_exposure("c5", "down5", dec!(4), s);
        assert_eq!(l.total_exposure(), dec!(0), "no drift after releasing both");
    }

    #[test]
    fn two_market_pair_exceeds_single_bracket_cap_and_leaves_no_trace() {
        let l = BracketLedger::new();
        let s = StrategyId::Arb15m5m;
        assert!(l
            .try_add_pair_exposure_per_market(
                ("c15", 100, "up15", dec!(4), s),
                ("c5", 100, "down5", dec!(12), s),
                dec!(10),
            )
            .is_none());
        assert_eq!(l.total_exposure(), dec!(0));
    }

    #[test]
    fn same_market_pair_behaves_as_before() {
        let l = BracketLedger::new();
        let s = StrategyId::Arb;
        // Covered YES+NO: net 0 even though each leg exceeds the cap on its own.
        let net = l.try_add_pair_exposure_per_market(
            ("c", 100, "yes", dec!(12), s),
            ("c", 100, "no", dec!(12), s),
            dec!(10),
        );
        assert_eq!(net, Some(dec!(0)));
    }

    use super::*;
    use rust_decimal_macros::dec;

    const MAX: Decimal = dec!(10);

    #[test]
    fn first_entry_passes_if_within_cap() {
        let l = BracketLedger::new();
        assert_eq!(
            l.try_add_exposure("cid", 9_999_999, "yes", dec!(8), MAX, StrategyId::Arb),
            Some(dec!(8))
        );
        assert_eq!(l.net_exposure("cid"), dec!(8));
    }

    #[test]
    fn same_side_blocked_when_exceeding_cap() {
        let l = BracketLedger::new();
        assert!(l
            .try_add_exposure("cid", 9_999_999, "yes", dec!(8), MAX, StrategyId::Arb)
            .is_some());
        // 8 + 5 = 13 > 10 → same-side stacking rejected.
        assert!(l
            .try_add_exposure("cid", 9_999_999, "yes", dec!(5), MAX, StrategyId::Arb)
            .is_none());
        assert_eq!(l.net_exposure("cid"), dec!(8)); // unchanged
    }

    #[test]
    fn opposite_leg_is_never_blocked() {
        let l = BracketLedger::new();
        assert!(l
            .try_add_exposure("cid", 9_999_999, "yes", dec!(10), MAX, StrategyId::Arb)
            .is_some());
        // Cap full, but the leg covers the YES: net drops to 0 → passes.
        assert_eq!(
            l.try_add_exposure("cid", 9_999_999, "no", dec!(10), MAX, StrategyId::Arb),
            Some(dec!(0))
        );
        // And after the coverage there is room again for the same side.
        assert_eq!(
            l.try_add_exposure("cid", 9_999_999, "yes", dec!(6), MAX, StrategyId::Arb),
            Some(dec!(6))
        );
    }

    /// A REJECTED attempt on a bracket that did NOT exist yet must not
    /// leave a phantom entry — its `clear_at` (possibly synthetic,
    /// e.g. a GTD expiry buffer) must not fix the bracket's `clear_at`
    /// for a later valid caller with the real `clear_at`.
    #[test]
    fn rejection_on_new_bracket_leaves_no_ghost_entry() {
        let l = BracketLedger::new();
        // Rejected: 15 > MAX(10) on a bracket that doesn't exist yet, with a
        // "synthetic" clear_at (100).
        assert!(l
            .try_add_exposure("cid", 100, "yes", dec!(15), MAX, StrategyId::Arb)
            .is_none());
        assert_eq!(
            l.len(),
            0,
            "no phantom entry should remain after the rejection"
        );

        // A later valid caller, with the REAL clear_at (9_999_999), must be
        // able to set the bracket from scratch.
        assert_eq!(
            l.try_add_exposure("cid", 9_999_999, "yes", dec!(5), MAX, StrategyId::Arb),
            Some(dec!(5))
        );
        assert_eq!(l.net_exposure("cid"), dec!(5));

        // Confirm that the governing clear_at is the valid caller's: the
        // bracket survives a purge at t=500 (> 100 of the rejected synthetic,
        // < 9_999_999 of the real one).
        l.purge_expired(500);
        assert_eq!(
            l.net_exposure("cid"),
            dec!(5),
            "the real clear_at (not the rejected synthetic one) must govern the bracket"
        );
    }

    /// If the bracket ALREADY existed (with real exposure inside), a
    /// rejected attempt must NOT delete the existing entry.
    #[test]
    fn rejection_on_existing_bracket_does_not_delete_it() {
        let l = BracketLedger::new();
        assert!(l
            .try_add_exposure("cid", 9_999_999, "yes", dec!(8), MAX, StrategyId::Arb)
            .is_some());
        // Rejected (8+5=13 > MAX=10), but the bracket ALREADY existed.
        assert!(l
            .try_add_exposure("cid", 9_999_999, "yes", dec!(5), MAX, StrategyId::Arb)
            .is_none());
        assert_eq!(
            l.len(),
            1,
            "the existing entry must not be deleted by a later rejection"
        );
        assert_eq!(
            l.net_exposure("cid"),
            dec!(8),
            "unchanged — the rejection must not mutate anything"
        );
    }

    #[test]
    fn covered_pair_net_zero() {
        let l = BracketLedger::new();
        let net = l.try_add_pair_exposure(
            "cid",
            9_999_999,
            ("yes", dec!(50), StrategyId::Arb),
            ("no", dec!(50), StrategyId::Arb),
            MAX,
        );
        assert_eq!(net, Some(dec!(0))); // 50+50 but net 0 → passes even with MAX=10
    }

    #[test]
    fn uneven_pair_evaluated_by_its_net() {
        let l = BracketLedger::new();
        // Pair net = |15 − 3| = 12 > 10 → rejected atomically, without mutating.
        assert!(l
            .try_add_pair_exposure(
                "cid",
                9_999_999,
                ("yes", dec!(15), StrategyId::Arb),
                ("no", dec!(3), StrategyId::Arb),
                MAX
            )
            .is_none());
        assert_eq!(l.net_exposure("cid"), Decimal::ZERO);
    }

    #[test]
    fn cap_zero_is_unlimited_but_counts() {
        let l = BracketLedger::new();
        assert!(l
            .try_add_exposure(
                "cid",
                9_999_999,
                "yes",
                dec!(999),
                Decimal::ZERO,
                StrategyId::Arb
            )
            .is_some());
        assert_eq!(
            l.try_add_exposure(
                "cid",
                9_999_999,
                "yes",
                dec!(1),
                Decimal::ZERO,
                StrategyId::Arb
            ),
            Some(dec!(1000))
        );
        assert_eq!(l.net_exposure("cid"), dec!(1000));
    }

    #[test]
    fn release_reduces_and_reopens_margin() {
        let l = BracketLedger::new();
        l.try_add_exposure("cid", 9_999_999, "yes", dec!(10), MAX, StrategyId::Arb);
        l.release_exposure("cid", "yes", dec!(4), StrategyId::Arb);
        assert_eq!(l.net_exposure("cid"), dec!(6));
        assert!(l
            .try_add_exposure("cid", 9_999_999, "yes", dec!(4), MAX, StrategyId::Arb)
            .is_some());
        assert!(l
            .try_add_exposure("cid", 9_999_999, "yes", dec!(1), MAX, StrategyId::Arb)
            .is_none()); // full
    }

    #[test]
    fn release_does_not_go_below_zero() {
        let l = BracketLedger::new();
        l.try_add_exposure("cid", 9_999_999, "yes", dec!(5), MAX, StrategyId::Arb);
        l.release_exposure("cid", "yes", dec!(999), StrategyId::Arb);
        assert_eq!(l.net_exposure("cid"), Decimal::ZERO);
    }

    #[test]
    fn force_add_ignores_cap() {
        let l = BracketLedger::new();
        l.force_add_exposure("cid", 9_999_999, "yes", dec!(200), StrategyId::Arb);
        assert_eq!(l.net_exposure("cid"), dec!(200));
    }

    #[test]
    fn purge_expired_removes_expired() {
        let l = BracketLedger::new();
        l.force_add_exposure("old", 100, "yes", dec!(10), StrategyId::Arb);
        l.force_add_exposure("future", 99_999_999, "yes", dec!(10), StrategyId::Arb);
        l.purge_expired(1_000);
        assert_eq!(l.len(), 1);
        assert_eq!(l.net_exposure("old"), Decimal::ZERO);
        assert_eq!(l.net_exposure("future"), dec!(10));
    }

    #[test]
    fn total_exposure_gross_sum_of_all_brackets() {
        let l = BracketLedger::new();
        l.try_add_exposure(
            "cid-1",
            9_999_999,
            "yes",
            dec!(4),
            Decimal::ZERO,
            StrategyId::Arb,
        );
        l.try_add_exposure(
            "cid-1",
            9_999_999,
            "no",
            dec!(4),
            Decimal::ZERO,
            StrategyId::Arb,
        );
        l.try_add_exposure(
            "cid-2",
            9_999_999,
            "yes",
            dec!(2),
            Decimal::ZERO,
            StrategyId::Arb,
        );
        // Gross (not net): cid-1's covered pair counts 8, not 0.
        assert_eq!(l.total_exposure(), dec!(10));
        l.clear();
        assert_eq!(l.total_exposure(), Decimal::ZERO);
    }

    #[test]
    fn gross_exposure_does_not_net_balanced_box() {
        let l = BracketLedger::new();
        // Balanced box YES=23/NO=23 (e.g. ParityMaker): net = 0, but the
        // gross exposure is 46 — RF-2 must see it so as not to post on top.
        l.try_add_pair_exposure(
            "cid",
            9_999_999,
            ("yes", dec!(23), StrategyId::ParityMaker),
            ("no", dec!(23), StrategyId::ParityMaker),
            Decimal::ZERO,
        );
        assert_eq!(
            l.net_exposure("cid"),
            Decimal::ZERO,
            "net directional exposure of a balanced box is 0"
        );
        assert_eq!(
            l.gross_exposure("cid"),
            dec!(46),
            "gross adds both sides without netting"
        );
    }

    #[test]
    fn gross_exposure_nonexistent_bracket_is_zero() {
        let l = BracketLedger::new();
        assert_eq!(l.gross_exposure("cid-missing"), Decimal::ZERO);
    }

    /// The claim is per owner. A `release` by
    /// strategy A cannot deduct exposure that B committed on the
    /// same token, and the excess is returned as an unreleased remainder instead
    /// of silently eating it (earlier behavior: fungible bucket with
    /// clamp to 0).
    #[test]
    fn release_only_deducts_from_owner() {
        let l = BracketLedger::new();
        l.try_add_exposure("cid", 9_999_999, "yes", dec!(6), MAX, StrategyId::Arb);
        l.try_add_exposure(
            "cid",
            9_999_999,
            "yes",
            dec!(4),
            MAX,
            StrategyId::NearCertain,
        );

        // B sells/releases 4 — A's claim stays intact.
        let rem = l.release_exposure("cid", "yes", dec!(4), StrategyId::NearCertain);
        assert_eq!(rem, Decimal::ZERO);
        assert_eq!(l.owner_exposure("cid", "yes", StrategyId::Arb), dec!(6));
        assert_eq!(
            l.owner_exposure("cid", "yes", StrategyId::NearCertain),
            Decimal::ZERO
        );

        // A tries to release MORE than its own — the excess touches nobody else
        // (in fact B no longer has a claim; the remainder is reported).
        let rem = l.release_exposure("cid", "yes", dec!(10), StrategyId::Arb);
        assert_eq!(rem, dec!(4), "the excess over the own claim is reported");
        assert_eq!(l.net_exposure("cid"), Decimal::ZERO);

        // Release by an owner that never claimed: total no-op.
        let rem = l.release_exposure("cid", "yes", dec!(5), StrategyId::MarketMaking);
        assert_eq!(rem, dec!(5));
    }

    #[test]
    fn independent_brackets_do_not_interfere() {
        let l = BracketLedger::new();
        l.try_add_exposure("cid-1", 9_999_999, "yes", dec!(10), MAX, StrategyId::Arb);
        assert!(l
            .try_add_exposure("cid-2", 9_999_999, "yes", dec!(10), MAX, StrategyId::Arb)
            .is_some());
        assert!(l
            .try_add_exposure("cid-1", 9_999_999, "yes", dec!(1), MAX, StrategyId::Arb)
            .is_none());
    }
}
