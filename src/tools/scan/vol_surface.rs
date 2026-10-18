//! VolSurface (VS) tool: cross-expiry implied volatility scan, migrated from
//! `strategy::vol_surface`.
//!
//! For the same asset and the same strike, brackets with different durations imply
//! the same σ if the market is efficient:
//!
//!   yes_price = Φ(ln(spot/strike) / (σ × √T))
//!   → σ_implied = ln(spot/strike) / (Φ⁻¹(yes_mid) × √T)
//!
//! If σ_short >> σ_long, the short bracket overestimates vol → maker edge in it.
//! If σ_short << σ_long, the long bracket overestimates vol.
//!
//! It is a pure observer: it never places orders. It emits logs and metrics when
//! it detects a divergence ≥ vol_ratio_threshold, and persists each divergence in
//! scan_observations for the gate. Promotion to trading requires at least one week
//! of scans confirming a real edge; if enabled, it stops being a `Tool` and becomes
//! a `Strategy` again.

use std::{collections::HashMap, time::Instant};

use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;

use crate::{
    config::VolSurfaceConfig,
    fair_value::phi_inv,
    tools::scan::ladder_arb::parse_slug,
    tools::scan::sink::ScanObs,
    types::{BookSnapshot, VolSnapshot},
};

use crate::tools::{Tool, ToolId};

// ── Cache de observations ────────────────────────────────────────────────────

struct BracketEntry {
    asset: String, // slug prefix: "btc", "eth"
    strike: u64,   // bracket level (large number)
    clear_at: i64,
    yes_mid: f64, // (best_bid_yes + best_ask_yes) / 2
    slug: String,
}

// ── VolSurface ────────────────────────────────────────────────────────────────

pub struct VolSurface {
    cfg: VolSurfaceConfig,
    cache: HashMap<String, BracketEntry>, // CID → entry
    alerted: HashMap<String, Instant>,    // "cid_a|cid_b" → last alert
    /// Persistence of divergences — the gate requires at least one week of data.
    scan_tx: tokio::sync::mpsc::UnboundedSender<ScanObs>,
}

impl VolSurface {
    pub fn new(
        cfg: VolSurfaceConfig,
        scan_tx: tokio::sync::mpsc::UnboundedSender<ScanObs>,
    ) -> Self {
        Self {
            cfg,
            cache: HashMap::new(),
            alerted: HashMap::new(),
            scan_tx,
        }
    }
}

impl Tool for VolSurface {
    fn id(&self) -> ToolId {
        ToolId::VolSurface
    }

    fn on_book(&mut self, book: &BookSnapshot, vol: &VolSnapshot) {
        let now = chrono::Utc::now().timestamp();

        // Purge expired entries and old alerts (60s cooldown).
        self.cache.retain(|_, e| e.clear_at > now);
        let cooldown = std::time::Duration::from_secs(60);
        self.alerted.retain(|_, t| t.elapsed() < cooldown);

        // Update the cache with the current book.
        let yes_mid = match (book.yes.best_bid(), book.yes.best_ask()) {
            (Some(bid), Some(ask)) => {
                let m = (bid + ask) / dec!(2);
                match m.to_f64() {
                    Some(v) if v > 0.0 && v < 1.0 => v,
                    _ => return,
                }
            }
            _ => return,
        };

        let asset = book.slug.split('-').next().unwrap_or("unknown").to_string();
        let strike = match parse_slug(&book.slug) {
            Some((s, _)) => s,
            None => return,
        };

        let remaining = book.secs_to_clear();

        self.cache.insert(
            book.market_condition_id.clone(),
            BracketEntry {
                asset: asset.clone(),
                strike,
                clear_at: book.clear_at,
                yes_mid,
                slug: book.slug.clone(),
            },
        );

        // Oracle: ln(spot / strike) needed for σ_implied.
        let spot = vol.last_price;
        if spot <= dec!(0) || vol.is_stale_for_signal() {
            return;
        }
        let spot_f64 = match spot.to_f64() {
            Some(v) if v > 0.0 => v,
            _ => return,
        };
        let strike_f64 = strike as f64;
        let ln_ratio = (spot_f64 / strike_f64).ln();
        if ln_ratio.abs() < 1e-6 {
            // spot ≈ strike → z undefined
            return;
        }

        // Look for the longest bracket with the same (asset, strike).
        // We want: current (short) vs one with remaining >> current.
        let min_short = self.cfg.min_remaining_short as i64;
        let max_short = self.cfg.max_remaining_short as i64;

        // The current bracket must be the "short" one (within the analysis window).
        if remaining < min_short || remaining > max_short {
            return;
        }

        // Look for a "long" bracket with the same (asset, strike) and current remaining > max_short.
        // Use e.clear_at - now to avoid the stale remaining value in the cache.
        let long_entry = self
            .cache
            .iter()
            .filter(|(k, e)| {
                k.as_str() != book.market_condition_id.as_str()
                    && e.asset == asset
                    && e.strike == strike
                    && (e.clear_at - now) > max_short
            })
            .map(|(_, e)| e)
            .next();

        let long_entry = match long_entry {
            Some(e) => (e.clear_at - now, e.yes_mid, e.slug.clone()),
            None => return,
        };
        let (long_remaining, long_yes_mid, long_slug) = long_entry;

        // Compute σ_implied for both brackets.
        let sigma_short = match sigma_implied(yes_mid, remaining, ln_ratio) {
            Some(s) => s,
            None => return,
        };
        let sigma_long = match sigma_implied(long_yes_mid, long_remaining, ln_ratio) {
            Some(s) => s,
            None => return,
        };

        if sigma_long <= 0.0 {
            return;
        }
        let ratio = sigma_short / sigma_long;

        metrics::counter!("hrf_vs_pairs_evaluated_total").increment(1);
        if ratio >= self.cfg.vol_ratio_threshold || ratio <= 1.0 / self.cfg.vol_ratio_threshold {
            let pair_key = format!("{}|{}", book.slug, long_slug);
            if let std::collections::hash_map::Entry::Vacant(e) = self.alerted.entry(pair_key) {
                e.insert(Instant::now());
                metrics::counter!("hrf_vs_divergence_total").increment(1);
                tracing::info!(
                    short_slug = %book.slug,
                    long_slug  = %long_slug,
                    sigma_short,
                    sigma_long,
                    ratio,
                    yes_mid_short = yes_mid,
                    yes_mid_long  = long_yes_mid,
                    "VS-SCAN: implied volatility divergence detected"
                );
                // Persist — the gate aggregates over scan_observations.
                let _ = self.scan_tx.send(ScanObs {
                    strategy:   ToolId::VolSurface,
                    market_key: format!("{}|{}", book.slug, long_slug),
                    kind:       "vol_divergence",
                    edge:       rust_decimal::Decimal::from_f64_retain(ratio).unwrap_or_default(),
                    payload:    format!(
                        r#"{{"sigma_short":{sigma_short},"sigma_long":{sigma_long},"yes_mid_short":{yes_mid},"yes_mid_long":{long_yes_mid}}}"#,
                    ),
                });
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Implied volatility given a market price, remaining time and ln(spot/strike).
///
/// If z ≈ 0 or the inputs are invalid, returns None.
fn sigma_implied(yes_mid: f64, remaining_secs: i64, ln_ratio: f64) -> Option<f64> {
    if remaining_secs <= 0 {
        return None;
    }
    let z = phi_inv(yes_mid)?;
    if z.abs() < 0.1 {
        return None;
    } // z too small → unstable σ
    let sqrt_t = (remaining_secs as f64).sqrt();
    let sigma = ln_ratio / (z * sqrt_t);
    if sigma <= 0.0 {
        return None;
    }
    Some(sigma)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_implied_consistent_with_fair_value() {
        // If yes_mid = Φ(d) and d = ln_ratio / (σ × √T), then
        // sigma_implied(yes_mid, T, ln_ratio) must return σ.
        use crate::fair_value::standard_normal_cdf;

        let sigma = 0.001_f64; // known σ
        let t = 300_i64; // 5 min in seconds
        let ln_ratio = 0.05_f64; // spot 5% above the strike
        let d = ln_ratio / (sigma * (t as f64).sqrt());
        let yes_mid = standard_normal_cdf(d);

        let sigma_calc = sigma_implied(yes_mid, t, ln_ratio).unwrap();
        assert!(
            (sigma_calc - sigma).abs() < 1e-6,
            "sigma_calc={sigma_calc}, expected={sigma}"
        );
    }

    #[test]
    fn sigma_implied_rejects_zero_z() {
        // yes_mid = 0.5 → phi_inv returns None → sigma_implied returns None
        assert!(sigma_implied(0.5, 300, 0.01).is_none());
    }

    #[test]
    fn sigma_implied_rejects_negative_sigma() {
        // Negative ln_ratio with yes_mid > 0.5 → negative sigma → None
        // yes_mid = 0.8 → z > 0; ln_ratio < 0 → sigma < 0
        assert!(sigma_implied(0.8, 300, -0.05).is_none());
    }

    #[test]
    fn sigma_implied_rejects_zero_remaining() {
        assert!(sigma_implied(0.7, 0, 0.05).is_none());
    }
}
