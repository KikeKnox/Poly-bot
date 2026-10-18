//! `/ui/pnl_analytics` combines `max_drawdown`, `pnl_by_hour_of_day`
//! and `markouts_for_strategy` (one call per `StrategyId::ALL`). This test
//! confirms the combination does not panic with minimal data (one test
//! markout, no closed fills yet).
//!
//! This repo has no harness that builds a complete `ApiState`
//! (it requires `RiskGuard`/`ToolSupervisor`/`PrometheusHandle`/etc. — no
//! existing `/ui/*` test invokes the real axum handler, they all verify
//! the underlying `StateStore`/component layer). `markout_average_ignores_rows_without_mid5s`
//! replicates the handler's corrected formula over
//! raw `markouts_for_strategy` data to pin the correct behavior in a test:
//! the divisor and the reported `n` must count only
//! rows with `mid_5s = Some(...)`, never an unfiltered `rows.len()`.

use poly_bot::{state::StateStore, types::Mode};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

#[tokio::test]
async fn pnl_analytics_does_not_panic_with_minimal_data() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();
    store
        .record_markout(
            1_700_000_000_000,
            "cid1",
            "tok1",
            "near_certain",
            "buy",
            dec!(0.90),
            Some(dec!(0.89)),
            None,
        )
        .await
        .unwrap();

    let dd = store.max_drawdown().await.unwrap();
    assert_eq!(dd, dec!(0)); // no closed fills yet → no drawdown

    let by_hour = store.pnl_by_hour_of_day().await.unwrap();
    assert!(by_hour.is_empty());

    let markouts = store.markouts_for_strategy("near_certain").await.unwrap();
    assert_eq!(markouts.len(), 1);
}

/// Regression: with 2 recorded fills where only
/// 1 has `mid_5s` available, the average and `n` must be computed over
/// THAT subset (n=1), not over the 2 total rows — otherwise the
/// average is biased toward 0 and, in the extreme case of 0 rows with `mid_5s`,
/// would report `"0.0"` as if it were a real measurement.
#[tokio::test]
async fn markout_average_ignores_rows_without_mid5s() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();
    // Fill with mid_5s available.
    store
        .record_markout(
            1_700_000_000_000,
            "cid1",
            "tok1",
            "near_certain",
            "buy",
            dec!(0.90),
            Some(dec!(0.89)),
            None,
        )
        .await
        .unwrap();
    // Fill in an illiquid bracket: no book snapshot at +5s.
    store
        .record_markout(
            1_700_000_001_000,
            "cid1",
            "tok2",
            "near_certain",
            "buy",
            dec!(0.50),
            None,
            None,
        )
        .await
        .unwrap();

    let rows = store.markouts_for_strategy("near_certain").await.unwrap();
    assert_eq!(rows.len(), 2); // the raw source does return both rows

    // Exact replica of the handler's corrected formula (src/api/handlers.rs).
    let with_mid5s: Vec<Decimal> = rows
        .iter()
        .filter_map(|(_, fill, mid_5s, _)| mid_5s.map(|m| (m - fill) / fill * dec!(10_000)))
        .collect();
    assert_eq!(with_mid5s.len(), 1); // only the row with mid_5s = Some(...) counts

    let n = with_mid5s.len();
    let avg_bps = with_mid5s.iter().sum::<Decimal>() / Decimal::from(n as i64);
    // (0.89 - 0.90) / 0.90 * 10_000 ≈ -111.1 bps — and NOT (-111.1 + 0) / 2 ≈ -55.6,
    // which is what the total-divisor bug would have produced.
    assert_eq!(avg_bps.round_dp(1), dec!(-111.1));
}

/// Extreme case: if no row has `mid_5s`, the strategy
/// must not produce a row in `markout_by_strategy` (nor fake an average
/// of "0.0" bps that a reader would interpret as a measured neutral markout).
#[tokio::test]
async fn markout_with_no_row_with_mid5s_produces_no_average() {
    let store = StateStore::open_memory(Mode::Paper).await.unwrap();
    store
        .record_markout(
            1_700_000_000_000,
            "cid1",
            "tok1",
            "near_certain",
            "buy",
            dec!(0.90),
            None,
            None,
        )
        .await
        .unwrap();

    let rows = store.markouts_for_strategy("near_certain").await.unwrap();
    assert_eq!(rows.len(), 1);

    let with_mid5s: Vec<Decimal> = rows
        .iter()
        .filter_map(|(_, fill, mid_5s, _)| mid_5s.map(|m| (m - fill) / fill * dec!(10_000)))
        .collect();
    assert!(with_mid5s.is_empty()); // the handler must not push a row for this strategy
}
