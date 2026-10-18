//! Stamping of `reference_price` for updown brackets.
//!
//! The Gamma API does NOT return `openPrice`/`startValue`/`referencePrice` for
//! updown markets, so the "strike"
//! (underlying price at bracket open) is captured here: when
//! `now` crosses `market_open_ts`, the latest Binance oracle price is copied
//! into the registry entry. Later BookSnapshots carry it and the
//! FV-driven strategies (NC/NE/MM/DIR/TC/OM/DriftBox) stop being inert
//! — in an 18h calibration run none of them emitted a single signal because of this.

use std::collections::HashMap;
use std::sync::Arc;

use rust_decimal::Decimal;
use tokio::sync::watch;

use super::registry::MarketRegistry;
use crate::oracle::asset_from_slug;
use crate::types::VolSnapshot;

/// Window after the open in which the oracle's current price still counts as
/// the reference.
///
/// Previously 30s, the same as `market_poll_interval_secs` (the discovery
/// cycle) — a bracket registered late (cold startup, feed restart,
/// or an unlucky discovery cycle that finds it already near the
/// edge of the window) lost the whole 30s window and was left with
/// `reference_price=None` FOREVER (this task is idempotent: it only writes
/// if still `None`, so missing it once means missing it for the whole bracket).
/// 120s gives real margin over the discovery poll without getting so close to
/// `strategy::REFERENCE_FALLBACK_MAX_AGE_SECS` (30s, fallback window in the
/// strategies themselves) as to confuse the two mechanisms — they are
/// deliberately independent: this one stamps ONCE in the shared
/// registry, that one is a PER-STRATEGY fallback over the current price.
const STAMP_MAX_AGE_SECS: i64 = 120;

/// Periodic task (1 s): stamps `reference_price` on registry entries
/// whose bracket has just opened. Idempotent: only writes if still `None`.
pub async fn run(
    registry: Arc<MarketRegistry>,
    vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    mut shutdown: watch::Receiver<bool>,
) {
    tracing::info!("reference: stamping task starting");
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() { break; }
                continue;
            }
            _ = tick.tick() => {}
        }

        let now = chrono::Utc::now().timestamp();
        for mut e in registry.markets.iter_mut() {
            let m = e.value_mut();
            if !eligible_for_stamp(m.reference_price, m.market_open_ts, now) {
                continue;
            }
            let age = now - m.market_open_ts;
            let Some(asset) = asset_from_slug(&m.slug) else {
                continue;
            };
            let Some(rx) = vol_rxs.get(&asset) else {
                continue;
            };
            let (price, stale) = {
                let v = rx.borrow();
                (v.last_price, v.is_stale_for_signal())
            };
            if price <= Decimal::ZERO || stale {
                continue;
            }
            m.reference_price = Some(price);
            metrics::counter!("hrf_reference_stamped_total", "asset" => asset).increment(1);
            tracing::debug!(
                slug = %m.slug, %price, age,
                "reference: opening price stamped from the oracle"
            );
        }
    }
    tracing::info!("reference: task finished (shutdown)");
}

/// Eligibility condition for stamping, extracted as a pure
/// function so the widened window can be tested without running the real task
/// (1s timer + watch channels).
fn eligible_for_stamp(reference_price: Option<Decimal>, market_open_ts: i64, now: i64) -> bool {
    if reference_price.is_some() || market_open_ts <= 0 {
        return false;
    }
    let age = now - market_open_ts;
    (0..=STAMP_MAX_AGE_SECS).contains(&age)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    /// A bracket registered 90s after its open (cold startup
    /// / feed restart) now DOES fall inside the window — before widening
    /// `STAMP_MAX_AGE_SECS` from 30 to 120 it was lost forever.
    #[test]
    fn extended_window_covers_90s_cold_start() {
        let market_open_ts = 1_000_000i64;
        let now = market_open_ts + 90;
        assert!(
            eligible_for_stamp(None, market_open_ts, now),
            "a 90s delay must remain eligible with the widened window (120s)"
        );
    }

    #[test]
    fn outside_extended_window_is_not_eligible() {
        let market_open_ts = 1_000_000i64;
        let now = market_open_ts + 121;
        assert!(!eligible_for_stamp(None, market_open_ts, now));
    }

    #[test]
    fn already_has_reference_price_not_eligible() {
        let market_open_ts = 1_000_000i64;
        assert!(!eligible_for_stamp(
            Some(dec!(100000)),
            market_open_ts,
            market_open_ts + 1
        ));
    }

    #[test]
    fn bracket_not_yet_open_is_not_eligible() {
        let market_open_ts = 1_000_000i64;
        assert!(!eligible_for_stamp(
            None,
            market_open_ts,
            market_open_ts - 5
        ));
    }
}
