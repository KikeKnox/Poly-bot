//! Per-asset turbulence index — a regime signal inspired by FinRL's
//! *turbulence index* (univariate Mahalanobis distance over spot
//! log-returns):
//!
//! ```text
//!   turb = (r_short − μ_long)² / σ²_long
//! ```
//!
//! - `r_short`: spot log-return at the horizon of the **short window**
//!   (`window_short_secs`, in production `SIGMA_WINDOW_SHORT` = 60 s):
//!   `ln(p_t / p_{t−W})` between the last cell and the cell exactly
//!   `horizon_cells` back.
//! - `μ_long`/`σ²_long`: mean and variance (population, ÷n) of the log-returns
//!   **at the same horizon** computed over all the cell pairs within
//!   the **long window** (`window_long_secs`, in production
//!   `SIGMA_WINDOW_LONG` = 300 s), excluding the `r_short` pair — the
//!   history does not contain the current observation, as in FinRL.
//!
//! That μ/σ² are estimated at the SAME horizon as `r_short` is not a detail: it is
//! what makes the index a dimensionless distance (~χ²(1) in a calm regime,
//! median ≈ 0.45) instead of a ratio of variances across different scales.
//!
//! Conventions inherited from `oracle::vol`: the input is the
//! SAME `sec_samples` buffer subsampled to 1 s by `push_sec_sample`
//! (last sample of the cell wins, time-based retention), and returns are
//! formed only between cell pairs separated EXACTLY by the horizon —
//! intermediate gaps do not matter (the fixed-horizon return only uses the
//! two endpoints), but if an endpoint is missing the pair is skipped, not bridged.
//! This makes it more robust to gaps than the short σ̂: an empty cell only removes
//! the pairs that use it as an anchor, not all the consecutive ones.
//!
//! Warmup / fail-closed: returns `None` (absent signal, never a false 0) if
//! data is missing for `r_short` (last cell without an exact anchor at `horizon_cells`),
//! if there are fewer than `min_pairs` historical pairs, if σ²_long ≤ 0 (dead
//! series → undefined 0/0 index) or if the parameters are invalid
//! (`window_short_secs >= window_long_secs`, etc.). A one-off `None` (e.g.
//! a hole right at the anchor cell) recovers by itself on the next publish —
//! there is no state to resynchronize.
//!
//! Pure functions only, no I/O or channels (same style as `fair_value.rs`):
//! the wiring lives in `oracle::vol::publish`, which populates
//! `VolSnapshot::turbulence` and emits the gauge `hrf_oracle_turbulence{asset}`.

use std::collections::HashMap;

use rust_decimal::prelude::ToPrimitive;

use crate::fair_value::PriceSample;

/// Univariate turbulence index of the asset (see the module doc comment).
///
/// `samples`: price samples sorted by ascending `ts_ms` (in
/// production, the `sec_samples` buffer of `oracle::vol` already subsampled to
/// `grid_secs`; the function re-subsamples anyway in case raw ticks arrive).
/// `window_short_secs` must be a multiple of `grid_secs` (otherwise the effective
/// horizon is truncated to `horizon_cells · grid_secs`).
///
/// Returns `Some(turb)` with `turb ≥ 0` — values ~O(1) in a calm regime,
/// spikes of tens/hundreds on shocks. `None` on warmup or insufficient data.
pub fn turbulence_index(
    samples: &[PriceSample],
    grid_secs: i64,
    window_short_secs: i64,
    window_long_secs: i64,
    min_pairs: usize,
) -> Option<f64> {
    if grid_secs <= 0
        || window_short_secs <= 0
        || window_long_secs <= 0
        || window_short_secs >= window_long_secs
        || min_pairs < 2
    {
        return None;
    }
    let horizon_cells = window_short_secs.div_euclid(grid_secs);
    if horizon_cells <= 0 || samples.len() < 2 {
        return None;
    }
    debug_assert!(
        samples.windows(2).all(|w| w[0].ts_ms <= w[1].ts_ms),
        "turbulence_index: samples must arrive sorted by ascending ts_ms"
    );

    let grid_ms = grid_secs * 1000;
    let last_ts = samples.last().unwrap().ts_ms;
    let window_start_ts = last_ts - window_long_secs * 1000;

    // Subsampling to `grid_secs` cells (same convention as
    // `realized_vol_per_sec`): last sample of the cell wins, only cells
    // within the long window. `cells` ends up sorted by cell_id.
    let mut cells: Vec<(i64, f64)> = Vec::new();
    for s in samples {
        if s.ts_ms < window_start_ts || s.price <= rust_decimal::Decimal::ZERO {
            continue;
        }
        let Some(price) = s.price.to_f64() else {
            continue;
        };
        let cell = s.ts_ms.div_euclid(grid_ms);
        match cells.last_mut() {
            Some((last_cell, last_price)) if *last_cell == cell => *last_price = price,
            _ => cells.push((cell, price)),
        }
    }
    if cells.len() < 2 {
        return None;
    }

    // Cell→price map to resolve exact anchors at `horizon_cells`.
    let by_cell: HashMap<i64, f64> = cells.iter().copied().collect();

    // r_short: horizon return of the LAST cell. If its exact anchor does not
    // exist (hole right at t−W), there is no signal this cycle — fail-closed.
    let &(last_cell, last_price) = cells.last().unwrap();
    let anchor_price = *by_cell.get(&(last_cell - horizon_cells))?;
    let r_short = (last_price / anchor_price).ln();

    // μ_long/σ²_long: all the horizon returns within the window,
    // EXCLUDING the r_short pair (history without the current observation).
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n_pairs = 0usize;
    for &(cell, price) in &cells {
        if cell == last_cell {
            continue; // the current observation does not enter the history
        }
        if let Some(&p0) = by_cell.get(&(cell - horizon_cells)) {
            let r = (price / p0).ln();
            sum += r;
            sum_sq += r * r;
            n_pairs += 1;
        }
    }
    if n_pairs < min_pairs {
        return None;
    }

    let n = n_pairs as f64;
    let mu = sum / n;
    // Population variance (÷n): the index is a distance, not an unbiased
    // estimator; with n≥30 the ÷(n−1) difference is negligible.
    let var = sum_sq / n - mu * mu;
    if var <= 0.0 || !var.is_finite() || !r_short.is_finite() {
        // var ≤ 0 (dead series / float rounding) or non-finite r → undefined.
        return None;
    }
    Some((r_short - mu).powi(2) / var)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::prelude::FromPrimitive;
    use rust_decimal::Decimal;

    const GRID: i64 = 1;
    const W_SHORT: i64 = 60; // = production SIGMA_WINDOW_SHORT
    const W_LONG: i64 = 300; // = production SIGMA_WINDOW_LONG
    const MIN_PAIRS: usize = 30; // = production SIGMA_MIN_PAIRS

    /// Deterministic synthetic series: 1 sample/second from t=0, price
    /// oscillating ±`amp` relative around 100 via an LCG (no dependencies).
    fn series(n: usize, amp: f64, seed: u64) -> Vec<PriceSample> {
        let mut x = seed;
        (0..n)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((x >> 33) & 0x00FF_FFFF) as f64 / 0x00FF_FFFF as f64; // [0,1)
                let p = 100.0 * (1.0 + amp * (2.0 * u - 1.0));
                PriceSample {
                    ts_ms: i as i64 * 1000,
                    price: Decimal::from_f64(p).unwrap(),
                }
            })
            .collect()
    }

    fn turb(samples: &[PriceSample]) -> Option<f64> {
        turbulence_index(samples, GRID, W_SHORT, W_LONG, MIN_PAIRS)
    }

    #[test]
    fn warmup_does_not_emit() {
        // < horizon+1 cells: there is no anchor for r_short.
        assert_eq!(turb(&series(60, 0.0002, 7)), None);
        // 90 cells → only 29 historical pairs (< MIN_PAIRS=30).
        assert_eq!(turb(&series(90, 0.0002, 7)), None);
        // Empty / a single sample.
        assert_eq!(turb(&[]), None);
        assert_eq!(turb(&series(1, 0.0002, 7)), None);
    }

    #[test]
    fn quiet_series_low_index() {
        // 360s of ±0.02% noise: r_short within the historical distribution →
        // index of the order of χ²(1), far from a shock spike.
        let t = turb(&series(360, 0.0002, 42)).expect("must emit with history");
        assert!(
            (0.0..50.0).contains(&t),
            "calm index out of range: {t}"
        );
    }

    #[test]
    fn shock_high_peak() {
        let mut s = series(360, 0.0002, 42);
        // Instantaneous +2% shock in the last cell: the 60s return jumps
        // tens of σ above the calm regime.
        let last = s.last_mut().unwrap();
        last.price *= Decimal::from_f64(1.02).unwrap();
        let t_shock = turb(&s).expect("must emit after the shock");
        let t_calm = turb(&series(360, 0.0002, 42)).unwrap();
        assert!(
            t_shock > 100.0 && t_shock > 10.0 * t_calm,
            "shock={t_shock} vs calm={t_calm}"
        );
    }

    #[test]
    fn intermediate_gap_does_not_turn_signal_off() {
        // Removing a cell in the middle of the history only eliminates the pairs that
        // use it as anchor/endpoint — the rest remain and the index emits.
        let mut s = series(360, 0.0002, 42);
        s.remove(150);
        assert!(turb(&s).is_some());
    }

    #[test]
    fn gap_at_anchor_turns_signal_off() {
        // Without the exact cell 60s before the last, r_short cannot be formed:
        // fail-closed → None (the pair is not bridged with another older one).
        let mut s = series(360, 0.0002, 42);
        s.remove(299); // cell t=299s, anchor of the last (t=359s)
        assert_eq!(turb(&s), None);
    }

    #[test]
    fn dead_series_zero_variance_does_not_emit() {
        // Constant price: all returns = 0 → σ²=0 → undefined 0/0.
        let s: Vec<PriceSample> = (0..360i64)
            .map(|i| PriceSample {
                ts_ms: i * 1000,
                price: Decimal::from(100),
            })
            .collect();
        assert_eq!(turb(&s), None);
    }

    #[test]
    fn invalid_params_do_not_emit() {
        let s = series(360, 0.0002, 42);
        assert_eq!(turbulence_index(&s, 0, W_SHORT, W_LONG, MIN_PAIRS), None);
        assert_eq!(turbulence_index(&s, GRID, 0, W_LONG, MIN_PAIRS), None);
        assert_eq!(turbulence_index(&s, GRID, W_SHORT, 0, MIN_PAIRS), None);
        // Short window >= long: the history cannot contain the signal.
        assert_eq!(turbulence_index(&s, GRID, W_LONG, W_SHORT, MIN_PAIRS), None);
        assert_eq!(turbulence_index(&s, GRID, W_SHORT, W_LONG, 1), None);
    }
}
