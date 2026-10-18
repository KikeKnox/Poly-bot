//! LadderArb tool — **scan-only**, migrated from `strategy::ladder_arb`.
//!
//! Prior gate (phase 4): at least one week of scanning real BTC/ETH slugs,
//! confirming there is volume and edge > fees in pairs of adjacent brackets by strike.
//!
//! It is a pure observer: it never places orders, it only accumulates data per slug
//! (mid price, timestamp) and counts opportunities via Prometheus metrics and
//! `scan_observations`. If trading is ever enabled, this stops being a `Tool`
//! and becomes a `Strategy` again.
//!
//! Strike parser: slug `btc-updown-95000-2025-01-31` → strike 95000.
//! Grouping: (asset_prefix, date_suffix) as the "expiry"; strikes as the
//! rungs of the ladder.
//!
use std::{collections::HashMap, time::Instant};

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::{
    config::LadderArbConfig,
    tools::scan::sink::ScanObs,
    types::{BookSnapshot, VolSnapshot},
};

use crate::tools::{Tool, ToolId};

struct BracketObs {
    slug: String,
    mid: Decimal,
    strike: u64,
    expiry: String,
    clear_at: i64,
}

pub struct LadderArb {
    cfg: LadderArbConfig,
    obs: HashMap<String, BracketObs>,
    /// Cooldown per pair (key = "lo_slug|hi_slug") so the same pair is not logged on every tick.
    alerted: HashMap<String, Instant>,
    /// Opportunity persistence; the gate requires at least one week of data.
    scan_tx: tokio::sync::mpsc::UnboundedSender<ScanObs>,
}

impl LadderArb {
    pub fn new(cfg: LadderArbConfig, scan_tx: tokio::sync::mpsc::UnboundedSender<ScanObs>) -> Self {
        Self {
            cfg,
            obs: HashMap::new(),
            alerted: HashMap::new(),
            scan_tx,
        }
    }
}

impl Tool for LadderArb {
    fn id(&self) -> ToolId {
        ToolId::LadderArb
    }

    fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) {
        // Accumulate the observation and purge expired.
        let now = chrono::Utc::now().timestamp();
        self.obs.retain(|_, o| o.clear_at > now);
        if let Some(mid) = book_mid(book) {
            if let Some((strike, expiry)) = parse_slug(&book.slug) {
                self.obs.insert(
                    book.slug.clone(),
                    BracketObs {
                        slug: book.slug.clone(),
                        mid,
                        strike,
                        expiry,
                        clear_at: book.clear_at,
                    },
                );
            }
        }

        // Look for adjacent pairs with edge
        self.scan_pairs();
    }
}

impl LadderArb {
    fn scan_pairs(&mut self) {
        let fee = self.cfg.fee_threshold;
        let min = self.cfg.min_edge_after_fee;
        const ALERT_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(60);

        // Purge expired entries so the map does not grow indefinitely.
        let now_instant = Instant::now();
        self.alerted
            .retain(|_, t| now_instant.duration_since(*t) < ALERT_COOLDOWN);

        let mut by_expiry: HashMap<&str, Vec<&BracketObs>> = HashMap::new();
        for obs in self.obs.values() {
            by_expiry.entry(&obs.expiry).or_default().push(obs);
        }

        for (expiry, mut brackets) in by_expiry {
            if brackets.len() < 2 {
                continue;
            }
            brackets.sort_by_key(|b| b.strike);

            for pair in brackets.windows(2) {
                let lo = pair[0];
                let hi = pair[1];
                let edge = dec!(1) - lo.mid - (dec!(1) - hi.mid) - fee;
                metrics::counter!("hrf_ladder_arb_pairs_evaluated_total").increment(1);
                if edge >= min {
                    metrics::counter!("hrf_ladder_arb_opps_total").increment(1);

                    let pair_key = format!("{}|{}", lo.slug, hi.slug);
                    let now = Instant::now();
                    let last = self.alerted.get(&pair_key).copied();
                    if last.is_some_and(|t| now.duration_since(t) < ALERT_COOLDOWN) {
                        continue; // already alerted recently
                    }
                    self.alerted.insert(pair_key, now);

                    tracing::info!(
                        expiry, lo_slug = %lo.slug, hi_slug = %hi.slug,
                        lo_mid = %lo.mid, hi_mid = %hi.mid, edge = %edge,
                        "LADDER-SCAN: opportunity detected"
                    );
                    // Persist: the gate aggregates over scan_observations.
                    let _ = self.scan_tx.send(ScanObs {
                        strategy:   ToolId::LadderArb,
                        market_key: format!("{}|{}", lo.slug, hi.slug),
                        kind:       "ladder_opp",
                        edge,
                        payload:    format!(
                            r#"{{"expiry":"{expiry}","lo_mid":"{}","hi_mid":"{}","lo_strike":{},"hi_strike":{}}}"#,
                            lo.mid, hi.mid, lo.strike, hi.strike,
                        ),
                    });
                }
            }
        }
    }
}

/// Parses `btc-updown-95000-2025-01-31` → (95000, "btc-updown--2025-01-31")
/// Returns None if the slug does not have a recognizable format.
///
/// Real up/down slugs look like `btc-updown-5m-1783368300`: the number is the
/// closing EPOCH, not a strike (the strike of an up/down is the opening price
/// and does not appear in the slug). This parser used to take it as a strike, so
/// LadderArb/VolSurface persisted noise (VolSurface computed ln(spot/epoch)).
/// Both tools stay inert with these slugs until markets with a real strike in
/// the slug exist.
pub(crate) fn parse_slug(slug: &str) -> Option<(u64, String)> {
    let parts: Vec<&str> = slug.split('-').collect();
    // The first large number used to be located with `.position()`; if it was
    // an epoch (>= 1e9), the WHOLE search was abandoned with `return None`, so a
    // slug with an epoch BEFORE the strike lost a valid strike later in `parts`.
    // Now the scan continues (`continue`) after a discarded candidate (epoch or
    // non-numeric) instead of stopping at the first large number. This is
    // strictly more permissive: it does not change the result for any slug that
    // already parsed correctly.
    for (idx, part) in parts.iter().enumerate() {
        let Ok(n) = part.parse::<u64>() else { continue };
        if n <= 1000 {
            continue;
        } // not a strike candidate
        if n >= 1_000_000_000 {
            continue;
        } // closing epoch, not a strike: keep searching
          // The "expiry" is the slug without the strike (prefix + date suffix)
        let expiry = format!("{}-{}", parts[..idx].join("-"), parts[idx + 1..].join("-"));
        return Some((n, expiry));
    }
    None
}

/// Book mid price as (best_bid_yes + best_ask_yes) / 2.
fn book_mid(book: &BookSnapshot) -> Option<Decimal> {
    let bid = book.yes.best_bid()?;
    let ask = book.yes.best_ask()?;
    Some((bid + ask) / dec!(2))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_slug_extracts_strike() {
        let (strike, expiry) = parse_slug("btc-updown-95000-2025-01-31").unwrap();
        assert_eq!(strike, 95000);
        assert!(expiry.contains("btc-updown"));
        assert!(expiry.contains("2025-01-31"));
    }

    #[test]
    fn parse_slug_rejects_malformed() {
        assert!(parse_slug("btc-updown").is_none());
        assert!(parse_slug("abc-def-ghi").is_none());
    }

    #[test]
    fn parse_slug_rejects_epoch_updown() {
        // In real up/down slugs the number is the closing epoch, not a strike.
        assert!(parse_slug("btc-updown-5m-1783368300").is_none());
        assert!(parse_slug("doge-updown-15m-1783368000").is_none());
    }

    /// A slug with an epoch (>=1e9) BEFORE a valid strike must not abandon the
    /// search at the first large number; it must keep scanning until it finds
    /// the real strike further on. Synthetic format (the real format of
    /// `crypto-multistrike` could not be confirmed without network access); the
    /// fix is strictly more permissive and does not break any of the cases
    /// covered above.
    #[test]
    fn parse_slug_keeps_searching_after_previous_epoch() {
        let (strike, expiry) = parse_slug("btc-multistrike-1783368300-95000-2025-01-31")
            .expect("should find the strike even when an epoch appears earlier in the slug");
        assert_eq!(strike, 95000);
        assert!(expiry.contains("1783368300"));
        assert!(expiry.contains("2025-01-31"));
    }
}
