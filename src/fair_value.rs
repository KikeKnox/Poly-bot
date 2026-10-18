/// Probabilistic fair-value model for Polymarket up/down brackets.
///
/// Formula: P(YES) ≈ Φ( ln(spot/strike) / (σ·√t_remaining) )
/// where Φ is the standard normal CDF (Abramowitz & Stegun 26.2.17).
///
/// The volatility σ is computed from realized variance over a fixed
/// time grid (see `realized_vol_per_sec`).
///
/// This module exposes only pure functions — no state, no I/O — to
/// make offline backtesting easier.
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal::{Decimal, MathematicalOps};

use crate::types::BookSnapshot;

// ── Standard normal CDF ──────────────────────────────────────────────────────

/// Approximation of the standard normal CDF Φ(x).
///
/// Uses the rational approximation of Abramowitz & Stegun §26.2.17, which gives a
/// maximum error < 7.5×10⁻⁸ over the whole real range.
pub(crate) fn standard_normal_cdf(x: f64) -> f64 {
    let neg = x < 0.0;
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.2316419 * x);
    // A&S 26.2.17 coefficients
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi = 1.0 - ((-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()) * poly;
    if neg {
        1.0 - phi
    } else {
        phi
    }
}

// ── Inverse normal CDF ───────────────────────────────────────────────────────

/// Inverse of the standard normal CDF Φ⁻¹(p).
///
/// Initial estimate: rational approximation of A&S §26.2.22 (Beasley-Springer).
/// Refined with 5 Newton-Raphson iterations. Error < 1e-6 for p ∈ (0.001, 0.999).
///
/// Returns `None` if p ≤ 0, p ≥ 1, or |p − 0.5| < 1e-6 (z ≈ 0 → σ undefined).
pub(crate) fn phi_inv(p: f64) -> Option<f64> {
    if p <= 0.0 || p >= 1.0 {
        return None;
    }
    if (p - 0.5).abs() < 1e-6 {
        return None;
    }

    let sign = if p < 0.5 { -1.0_f64 } else { 1.0_f64 };
    let q = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * q.ln()).sqrt();
    // Initial estimate (A&S 26.2.22)
    let num = 2.515517 + 0.802853 * t + 0.010328 * t * t;
    let den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t;
    let mut x = sign * (t - num / den);

    // Newton-Raphson: x_{n+1} = x_n − (Φ(x_n) − p) / φ(x_n)
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    for _ in 0..5 {
        let fx = standard_normal_cdf(x) - p;
        let phi_x = (-0.5 * x * x).exp() / sqrt_2pi;
        if phi_x < 1e-15 {
            break;
        }
        x -= fx / phi_x;
    }
    Some(x)
}

// ── CLOB fees ────────────────────────────────────────────────────────────────

/// Base rate of the Polymarket taker fee (`baseRate` in `baseRate × min(p, 1−p)`).
///
/// FEE-1: verified against the API and the contract. `GET /fee-rate?token_id=…`
/// returns `{"base_fee":1000}` (= 1000 bps = 10%) for the BTC/ETH up/down brackets,
/// which is also the `MAX_FEE_RATE_BIPS` of the V1 contract. Before, 0.07 was used (the
/// coefficient published in the docs), which underestimated the real fee.
///
/// Maker (resting GTD) = 0% according to the official docs ("Makers are never charged fees").
/// TO BE CONFIRMED: the SDK signs this same `feeRateBps` on maker AND taker orders,
/// and the contract charges according to the signed `feeRateBps`. The hypothesis (unverified) is that
/// the signed value is an authorized MAXIMUM and the operator applies 0 to the maker. If it turned out
/// that the maker pays, MM/PM/TC would be unviable at the root. Resolve with a real
/// $5 maker order counting the shares received.
pub const TAKER_FEE_RATE: Decimal = rust_decimal_macros::dec!(0.10);

/// A4 / FEE-1: exact CLOB taker fee for a leg at price `p`.
///
/// The contract (`CalculatorHelper.sol`) charges `baseRate × min(p, 1−p) × shares`, NOT
/// `baseRate × p × (1−p)` as was previously modeled: the real curve is triangular, not
/// parabolic. They roughly coincide at the extremes but diverge ~2x at p≈0.5,
/// which is exactly where ARB operates.
///
/// Useful corollary: buying at p > 0.5 the upside is `1 − p`, which is exactly
/// `min(p, 1−p)` → the taker fee ALWAYS takes the same fraction (`TAKER_FEE_RATE`)
/// of the potential profit, whether buying at 0.99 or 0.999. Buying dearer does not escape it.
///
/// FEE-2: base rate of the parabolic taker fee, verified by
/// triple source on the public API: (1) Gamma's per-market `feeSchedule`
/// returns `{"exponent":1,"rate":0.07,"takerOnly":true,"rebateRate":0.2}` for
/// the live BTC/ETH up/down; (2) the official Polymarket fee table
/// matches EXACTLY `0.07 × p × (1−p)` per share (at p=0.10 → $0.63/100sh,
/// at p=0.50 → $1.75/100sh — the FEE-1 triangular would give $0.70 and $5.00
/// respectively); (3) `GET /fee-rate` (FEE-1 source, `base_fee: 1000`) is
/// the contract's `MAX_FEE_RATE_BIPS`, not the fee charged — deprecated as a
/// source.
///
/// Maker = 0% (unchanged with respect to FEE-1).
///
/// This constant is the conservative FALLBACK
/// when the market does not carry a `feeSchedule` in the Gamma API —
/// verified with 31,493 markets that at least 2 `feeSchedule.rate` regimes
/// coexist: `crypto_fees_v2` with `rate=0.07` (crypto up/down, where ARB
/// mostly operates) and `finance_prices_fees` with `rate=0.04` (stock
/// ladders). Using 0.07 for a market with a real rate of 0.04 overestimates the
/// fee by ~75% — conservative, but filters out real +EV opportunities. Prefer
/// [`fee_taker_parabolic_with_rate`] with the market's real rate
/// (`BookSnapshot::parabolic_fee_rate_or`) when available.
pub const TAKER_FEE_RATE_PARABOLIC: Decimal = rust_decimal_macros::dec!(0.07);

/// FEE-2: exact (parabolic) taker fee for `notional_shares` at price `price`,
/// with an explicit `rate`.
///
/// `fee = rate × price × (1 − price) × notional_shares`. Maximum at
/// `price = 0.5` (where ARB operates in the middle zone) and tends to 0 at the
/// extremes (`price → 0` or `price → 1`) — unlike the triangular
/// FEE-1 model, which does not decay at the extremes as a fraction of the upside.
pub fn fee_taker_parabolic_with_rate(
    rate: Decimal,
    price: Decimal,
    notional_shares: Decimal,
) -> Decimal {
    rate * price * (Decimal::ONE - price) * notional_shares
}

/// Breakeven of a taker BUY entry at
/// `ask`, with the parabolic fee (FEE-2) discounted — `ask + fee(ask)`. Uses the
/// market's real rate (`book.parabolic_fee_rate`) if discovery supplied it,
/// otherwise `TAKER_FEE_RATE_PARABOLIC`. Previously copied by hand in 5 strategies (NC,
/// drift_box, open_momentum, directional×2) before this helper.
pub fn breakeven_price_parabolic(book: &BookSnapshot, ask: Decimal) -> Decimal {
    let rate = book.parabolic_fee_rate_or(TAKER_FEE_RATE_PARABOLIC);
    ask + fee_taker_parabolic_with_rate(rate, ask, Decimal::ONE)
}

/// Sell mirror of
/// [`breakeven_price_parabolic`] — net EV of closing (selling) at `bid`, with the
/// parabolic fee discounted: `bid − fee(bid)`. Used by `theta_capture` to
/// decide the bail. It is not merged with the buy function: the opposite sign
/// (add vs. subtract) makes a single function with a "direction" less readable
/// than two named functions.
pub fn net_of_fee_sell_parabolic(book: &BookSnapshot, bid: Decimal) -> Decimal {
    let rate = book.parabolic_fee_rate_or(TAKER_FEE_RATE_PARABOLIC);
    bid - fee_taker_parabolic_with_rate(rate, bid, Decimal::ONE)
}

/// Expected relative move of the underlying in the remaining time: `σ·√t`.
///
/// It is the scale of `fair_value_yes` itself (there it is the denominator of `d`), and serves
/// to answer *"is there any room left in this bracket?"*: when `σ·√t` becomes small
/// relative to the distance to the strike, the outcome is already decided — the price will not
/// move any more, the losing token can only converge to 0, and no flow is left to cross.
///
/// Preferable to a threshold in seconds: it adapts to the volatility and to the duration of the
/// bracket, whereas a "less than N seconds left" only holds for the σ of the day it
/// was measured. Returns `None` with invalid parameters.
pub fn expected_move(sigma_per_sec: Decimal, t_remaining: i64) -> Option<Decimal> {
    if sigma_per_sec <= Decimal::ZERO || t_remaining <= 0 {
        return None;
    }
    let sqrt_t = Decimal::from(t_remaining).sqrt()?;
    Some(sigma_per_sec * sqrt_t)
}

// ── Main fair-value ──────────────────────────────────────────────────────────

/// Risk-neutral probability that the asset closes **above** the strike.
///
/// Parameters:
/// - `spot`:         current price of the underlying (e.g. BTC in USD)
/// - `strike`:       bracket level (e.g. 100_000 USD)
/// - `sigma_per_sec`: realized volatility in price/√s units (output
///   of `realized_vol_per_sec`)
/// - `t_remaining`:  time until close in seconds (> 0)
///
/// Returns `None` if the parameters are not valid for the computation.
pub fn fair_value_yes(
    spot: Decimal,
    strike: Decimal,
    sigma_per_sec: Decimal,
    t_remaining: i64,
) -> Option<Decimal> {
    if spot <= Decimal::ZERO
        || strike <= Decimal::ZERO
        || sigma_per_sec <= Decimal::ZERO
        || t_remaining <= 0
    {
        return None;
    }
    let sqrt_t = Decimal::from(t_remaining).sqrt().unwrap_or_default();
    let denom = sigma_per_sec * sqrt_t;
    if denom.is_zero() {
        return None;
    }
    // d = ln(spot/strike) / (σ·√t)  — in f64 arithmetic for ln().
    // The Decimal→f64 precision loss (~16 significant digits) is
    // irrelevant here: the resulting FV is truncated to 6 decimals and the noise of the
    // σ estimator dominates by several orders of magnitude over the f64 error.
    // Not a bug; do not use pure Decimal (it has no ln()).
    let ratio: f64 = spot.to_f64()? / strike.to_f64()?;
    let d = ratio.ln() / denom.to_f64()?;
    let prob = standard_normal_cdf(d);
    // Convert to Decimal with 6 decimals of precision
    Decimal::from_str_exact(&format!("{:.6}", prob)).ok()
}

// ── Volatility EWMA ──────────────────────────────────────────────────────────

/// Price sample with timestamp (reuses BtcSample from directional, but
/// accepts any type with the fields `ts_ms: i64` and `price: Decimal`).
#[derive(Clone, Copy)]
pub struct PriceSample {
    pub ts_ms: i64,
    pub price: Decimal,
}

/// σ̂ in price/√s from realized variance over a FIXED
/// time grid.
///
/// Estimator by fixed time cells (not per tick), so it is not dominated
/// by sub-second microstructure noise from the Binance WS
/// (σ̂/σ_real ratio of a per-tick estimator: p10 = 0.03, p90 = 2.64 on BTC — it is not
/// a bias, it is noise).
///
/// Algorithm:
///  1. Subsample `samples` into cells of `grid_secs` (last price of each cell),
///     covering backwards `window_secs` from the most recent sample.
///  2. Log returns between CONSECUTIVE cells. A gap (empty cell) breaks the pair:
///     it is skipped, not bridged — bridging would attribute the move of N cells to one.
///  3. σ² = Σ ret² / (n_pairs · grid_secs);  σ = √σ².
///
/// Returns `None` if there are fewer than `min_pairs` valid pairs (below ~30 the
/// estimator has a standard error > 13 % and is not usable as a denominator).
///
/// Production parameters (`oracle::vol`): grid_secs = 1, window_secs = 300,
/// min_pairs = 30. Measured empirical coverage: 69.16 % against 68.27 % theoretical
/// (n = 64,177 obs / 12,042 brackets, Binance 1 s klines).
///
/// Invariants:
/// - `samples` must arrive sorted by `ts_ms` ascending. It is not sorted here
///   (it would allocate memory on the hot path and hide a bug in the caller);
///   it is checked with a monotonicity `debug_assert!`.
/// - A sample's cell is `ts_ms / (grid_secs * 1000)`. Within a cell the LAST
///   sample (the most recent) wins, not the first.
/// - `ln()` in `f64` (same as `fair_value_yes`); accumulation and result in `Decimal`.
/// - `grid_secs <= 0 || window_secs <= 0 || min_pairs < 2` → `None`.
pub fn realized_vol_per_sec(
    samples: &[PriceSample],
    grid_secs: i64,
    window_secs: i64,
    min_pairs: usize,
) -> Option<Decimal> {
    if grid_secs <= 0 || window_secs <= 0 || min_pairs < 2 {
        return None;
    }
    if samples.is_empty() {
        return None;
    }
    debug_assert!(
        samples.windows(2).all(|w| w[0].ts_ms <= w[1].ts_ms),
        "realized_vol_per_sec: samples must arrive sorted by ts_ms ascending"
    );

    let grid_ms = grid_secs * 1000;
    let last_ts = samples.last().unwrap().ts_ms;
    let window_start_ts = last_ts - window_secs * 1000;

    // Subsampling: walks samples (already sorted by ts_ms) and keeps the
    // last price of each grid_secs cell. `cells` ends up ordered by
    // cell_id naturally (ts_ms does not decrease ⇒ cell_id does not decrease).
    let mut cells: Vec<(i64, Decimal)> = Vec::new();
    for s in samples {
        if s.ts_ms < window_start_ts || s.price <= Decimal::ZERO {
            continue;
        }
        let cell = s.ts_ms.div_euclid(grid_ms);
        match cells.last_mut() {
            Some((last_cell, last_price)) if *last_cell == cell => {
                *last_price = s.price; // last sample of the cell wins
            }
            _ => cells.push((cell, s.price)),
        }
    }

    if cells.len() < 2 {
        return None;
    }

    let mut sum_sq: Decimal = Decimal::ZERO;
    let mut n_pairs: usize = 0;
    for w in cells.windows(2) {
        let (c0, p0) = w[0];
        let (c1, p1) = w[1];
        if c1 - c0 != 1 {
            continue; // gap: skipped, not bridged
        }
        let p0f: f64 = p0.to_f64()?;
        let p1f: f64 = p1.to_f64()?;
        let log_ret = (p1f / p0f).ln();
        let ret_sq = Decimal::from_f64(log_ret * log_ret).unwrap_or(Decimal::ZERO);
        sum_sq += ret_sq;
        n_pairs += 1;
    }

    if n_pairs < min_pairs {
        return None;
    }

    let var = sum_sq / (Decimal::from(n_pairs) * Decimal::from(grid_secs));
    if var <= Decimal::ZERO {
        return None;
    }
    var.sqrt()
}

// ── Charged fee quantized and fee-window schedule by validity ───────────────

/// Taker fee CHARGED for a fill — the official parabolic
/// `rate × p × (1−p) × shares` quantized to 5 decimals, the precision
/// published by the venue ("Fees are rounded to 5 decimal places. The smallest fee
/// charged is 0.00001 USDC", docs.polymarket.com/trading/fees).
/// 22-sep-2026).
///
/// The venue does not publish the rounding mode at the midpoint; [`Decimal::round_dp`]
/// (MidpointNearestEven) is used. The maximum effect of the
/// tie-break is ±0.000005 USDC per fill — irrelevant for the
/// conservative scenario, which also excludes unconfirmed rebates.
pub fn fee_charged_5dp(rate: Decimal, price: Decimal, shares: Decimal) -> Decimal {
    fee_taker_parabolic_with_rate(rate, price, shares).round_dp(5)
}

/// Validation error of a schedule of rules windows
/// (`RulesWindow`) frozen for a market.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeeWindowsError {
    /// `valid_to_ms` ≤ `valid_from_ms` — empty or inverted window.
    InvertedWindow { idx: usize },
    /// The windows are not sorted by `valid_from_ms` ascending.
    Unordered { idx: usize },
    /// Window `idx` covers instants already covered by the previous one —
    /// the validity would be ambiguous.
    Overlapped { idx: usize },
    /// Rate outside [0, 1]: a negative fee would be a rebate (not confirmed
    /// by the venue — excluded from the conservative scenario) and one > 1 would exceed the
    /// traded notional.
    InvalidRate { idx: usize },
}

impl std::fmt::Display for FeeWindowsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvertedWindow { idx } => {
                write!(f, "window {idx}: valid_to_ms <= valid_from_ms")
            }
            Self::Unordered { idx } => {
                write!(f, "window {idx}: valid_from_ms not ascending")
            }
            Self::Overlapped { idx } => {
                write!(f, "window {idx}: overlaps the previous window")
            }
            Self::InvalidRate { idx } => {
                write!(f, "window {idx}: parabolic_fee_rate outside [0,1]")
            }
        }
    }
}

impl std::error::Error for FeeWindowsError {}

/// Validates a frozen fee-window schedule — sorted by
/// `valid_from_ms`, non-empty/non-inverted, without overlaps and with rates in [0,1].
/// Pure function: the schedule lives in fixtures/manifests, the
/// API is not queried here.
pub fn validate_fee_windows(
    windows: &[crate::research::contract::RulesWindow],
) -> Result<(), FeeWindowsError> {
    let mut prev_to: Option<i64> = None;
    for (idx, w) in windows.iter().enumerate() {
        if let Some(to) = w.valid_to_ms {
            if to <= w.valid_from_ms {
                return Err(FeeWindowsError::InvertedWindow { idx });
            }
        }
        if let Some(prev) = prev_to {
            if w.valid_from_ms < prev {
                // Covers both strict disorder and overlap: if
                // `from` falls inside the previous window, it is an overlap; if
                // it falls before it starts, it is disorder.
                let prev_from = windows[idx - 1].valid_from_ms;
                return Err(if w.valid_from_ms < prev_from {
                    FeeWindowsError::Unordered { idx }
                } else {
                    FeeWindowsError::Overlapped { idx }
                });
            }
        }
        if let Some(rate) = w.parabolic_fee_rate {
            if rate < Decimal::ZERO || rate > Decimal::ONE {
                return Err(FeeWindowsError::InvalidRate { idx });
            }
        }
        prev_to = Some(w.valid_to_ms.unwrap_or(i64::MAX));
    }
    Ok(())
}

/// Rate in force at `now_ms` according to the frozen schedule. Returns the rate
/// of the window that covers `now` (`valid_from ≤ now < valid_to`, `None` =
/// open-ended); if the window in force reports no rate (`None`) or none covers
/// `now`, returns `fallback` — a rate is never invented.
///
/// `windows` must come validated by [`validate_fee_windows`]; with an invalid
/// schedule the result is that of the first window that covers (traversal
/// order), with no additional guarantees.
pub fn current_fee_rate(
    windows: &[crate::research::contract::RulesWindow],
    now_ms: i64,
    fallback: Decimal,
) -> Decimal {
    for w in windows {
        let covers = w.valid_from_ms <= now_ms && w.valid_to_ms.is_none_or(|to| now_ms < to);
        if covers {
            return w.parabolic_fee_rate.unwrap_or(fallback);
        }
    }
    fallback
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // ── fee_taker_parabolic_with_rate ──

    /// Explicit rate different from the fallback: `finance_prices_fees` market
    /// (rate=0.04) gives a ~43% lower fee than the crypto fallback (0.07) at the
    /// same price and notional.
    #[test]
    fn taker_fee_parabolic_with_rate_uses_explicit_rate() {
        assert_eq!(
            fee_taker_parabolic_with_rate(dec!(0.04), dec!(0.5), Decimal::ONE),
            dec!(0.01)
        );
        // p=0.5 is the point of maximum fee: 0.07 × 0.5 × 0.5 × 1 = 0.0175/share,
        // matches the official table ($1.75 per 100 shares at p=0.50).
        assert_eq!(
            fee_taker_parabolic_with_rate(TAKER_FEE_RATE_PARABOLIC, dec!(0.5), Decimal::ONE),
            dec!(0.0175),
        );
    }

    // ── breakeven_price_parabolic / net_of_fee_sell_parabolic ──

    /// The helper must match exactly the inline formula
    /// (`ask + fee_taker_parabolic_with_rate(rate, ask, 1)`) that was
    /// copied by hand in 5 places, both with the market's real rate and
    /// with the fallback.
    #[test]
    fn parabolic_breakeven_price_matches_inline_formula() {
        for (ask, parabolic_fee_rate) in [
            (dec!(0.10), Some(dec!(0.02))),
            (dec!(0.50), Some(dec!(0.07))),
            (dec!(0.80), None),
            (dec!(0.97), Some(dec!(0.10))),
        ] {
            let book = BookSnapshot {
                parabolic_fee_rate,
                ..Default::default()
            };
            let rate = parabolic_fee_rate.unwrap_or(TAKER_FEE_RATE_PARABOLIC);
            let inline = ask + fee_taker_parabolic_with_rate(rate, ask, Decimal::ONE);
            assert_eq!(breakeven_price_parabolic(&book, ask), inline);
        }
    }

    /// Sell mirror: must match `bid - fee_taker_parabolic_with_rate(rate, bid, 1)`,
    /// the formula that was inline in `theta_capture.rs`.
    #[test]
    fn parabolic_net_of_fee_sell_matches_inline_formula() {
        for (bid, parabolic_fee_rate) in [
            (dec!(0.10), Some(dec!(0.02))),
            (dec!(0.50), Some(dec!(0.07))),
            (dec!(0.80), None),
            (dec!(0.97), Some(dec!(0.10))),
        ] {
            let book = BookSnapshot {
                parabolic_fee_rate,
                ..Default::default()
            };
            let rate = parabolic_fee_rate.unwrap_or(TAKER_FEE_RATE_PARABOLIC);
            let inline = bid - fee_taker_parabolic_with_rate(rate, bid, Decimal::ONE);
            assert_eq!(net_of_fee_sell_parabolic(&book, bid), inline);
        }
    }

    // ── standard_normal_cdf ──

    #[test]
    fn cdf_zero_is_half() {
        let p = standard_normal_cdf(0.0);
        assert!((p - 0.5).abs() < 1e-6, "Φ(0) must be 0.5, was {p}");
    }

    #[test]
    fn cdf_large_positive_near_one() {
        let p = standard_normal_cdf(4.0);
        assert!(p > 0.9999, "Φ(4) must be > 0.9999, was {p}");
    }

    #[test]
    fn cdf_large_negative_near_zero() {
        let p = standard_normal_cdf(-4.0);
        assert!(p < 0.0001, "Φ(-4) must be < 0.0001, was {p}");
    }

    #[test]
    fn cdf_one_sigma() {
        // Φ(1) ≈ 0.8413
        let p = standard_normal_cdf(1.0);
        assert!((p - 0.8413).abs() < 0.0001, "Φ(1) ≈ 0.8413, was {p}");
    }

    // ── fair_value_yes ──

    #[test]
    fn fv_spot_equals_strike_is_half() {
        // spot=strike → ln(1)=0 → Φ(0)=0.5
        let fv = fair_value_yes(dec!(100_000), dec!(100_000), dec!(0.001), 60).unwrap();
        // 0.500000 ± 1e-4
        assert!((fv - dec!(0.5)).abs() < dec!(0.001), "fv={fv}");
    }

    #[test]
    fn fv_spot_well_above_strike_is_high() {
        // spot >> strike → large d → Φ(d) → 1
        let fv = fair_value_yes(dec!(110_000), dec!(100_000), dec!(0.001), 60).unwrap();
        assert!(fv > dec!(0.9), "fv={fv}");
    }

    #[test]
    fn fv_spot_well_below_strike_is_low() {
        // spot << strike → very negative d → Φ(d) → 0
        let fv = fair_value_yes(dec!(90_000), dec!(100_000), dec!(0.001), 60).unwrap();
        assert!(fv < dec!(0.1), "fv={fv}");
    }

    // ── phi_inv ──

    #[test]
    fn phi_inv_roundtrip() {
        // phi_inv(Φ(x)) ≈ x for x ∈ {-2, -1, 1, 2} (x=0 returns None by design)
        for &x in &[-2.0_f64, -1.0, 1.0, 2.0] {
            let p = standard_normal_cdf(x);
            let x2 = phi_inv(p).expect("phi_inv must return Some");
            assert!(
                (x2 - x).abs() < 1e-5,
                "phi_inv(Φ({x})) = {x2}, error = {}",
                (x2 - x).abs()
            );
        }
    }

    #[test]
    fn phi_inv_rejects_edges() {
        assert!(phi_inv(0.0).is_none());
        assert!(phi_inv(1.0).is_none());
        assert!(phi_inv(0.5).is_none()); // z ≈ 0 → undefined
        assert!(phi_inv(-0.1).is_none());
    }

    #[test]
    fn phi_inv_symmetry() {
        // Φ⁻¹(1-p) = -Φ⁻¹(p)
        let p = 0.8;
        let x1 = phi_inv(p).unwrap();
        let x2 = phi_inv(1.0 - p).unwrap();
        assert!((x1 + x2).abs() < 1e-5, "asymmetry: {x1} vs {x2}");
    }

    #[test]
    fn fv_invalid_params_return_none() {
        assert!(fair_value_yes(dec!(0), dec!(100_000), dec!(0.001), 60).is_none());
        assert!(fair_value_yes(dec!(100_000), dec!(0), dec!(0.001), 60).is_none());
        assert!(fair_value_yes(dec!(100_000), dec!(100_000), dec!(0), 60).is_none());
        assert!(fair_value_yes(dec!(100_000), dec!(100_000), dec!(0.001), 0).is_none());
    }

    // ── realized_vol_per_sec ─────────────────────────────────────────────────

    /// Generates the T1 price sequence: 301 points (300 steps), alternating
    /// log return ±`step` per 1s cell, starting at `base`.
    fn grid_step_prices(n_cells: usize, base: f64, step: f64) -> Vec<f64> {
        let mut price = base;
        let mut out = Vec::with_capacity(n_cells + 1);
        out.push(price);
        for i in 0..n_cells {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            price *= (sign * step).exp();
            out.push(price);
        }
        out
    }

    fn grid_samples() -> Vec<PriceSample> {
        let prices = grid_step_prices(300, 100_000.0, 0.0001);
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| PriceSample {
                ts_ms: (i as i64) * 1000,
                price: Decimal::from_f64(p).unwrap(),
            })
            .collect()
    }

    #[test]
    fn grid_1s_recovers_known_step() {
        let samples = grid_samples();
        let sigma = realized_vol_per_sec(&samples, 1, 300, 30).expect("must estimate σ̂");
        // target: 0.0001, tolerance 5%
        let target = dec!(0.0001);
        let tol = target * dec!(0.05);
        assert!(
            (sigma - target).abs() <= tol,
            "sigma={sigma}, target={target}"
        );
    }

    /// T2 — the test that catches the bug: 50 extra ticks per second (synthetic
    /// bid-ask bounce, small amplitude, dt~1ms) interleaved BEFORE the "anchor" point
    /// of each cell (which remains the last one of the cell and therefore the one that
    /// wins the subsampling). The result of `realized_vol_per_sec` must not
    /// change by more than 5% with respect to T1.
    #[test]
    fn insensitive_to_sub_second_bid_ask_bounce() {
        let prices = grid_step_prices(300, 100_000.0, 0.0001);
        let mut samples: Vec<PriceSample> = Vec::new();
        let bounce_amp = 0.00004; // smaller than the real step (0.0001); the dt~1ms amplifies it all the same
        for i in 0..300usize {
            let cell_ts = (i as i64) * 1000;
            let anchor_price = prices[i]; // STARTING price of cell i
                                          // 50 bounce ticks, dt≈1ms, BEFORE the end-of-cell anchor.
            for k in 0..50i64 {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                let bounce_price = anchor_price * (1.0 + sign * bounce_amp);
                samples.push(PriceSample {
                    ts_ms: cell_ts + 1 + k,
                    price: Decimal::from_f64(bounce_price).unwrap(),
                });
            }
            // Real anchor of the cell: the "true" price of T1 at i+1,
            // placed at the END of the cell (ts=cell_ts+999) so that the
            // subsampling (last sample of the cell) recovers it exactly.
            samples.push(PriceSample {
                ts_ms: cell_ts + 999,
                price: Decimal::from_f64(prices[i + 1]).unwrap(),
            });
        }

        let sigma_new = realized_vol_per_sec(&samples, 1, 300, 30)
            .expect("must estimate σ̂ despite the sub-second noise");
        let target = dec!(0.0001);
        let tol = target * dec!(0.05);
        assert!((sigma_new - target).abs() <= tol,
            "realized_vol_per_sec must be insensitive to the bounce: sigma={sigma_new}, target={target}");
    }

    #[test]
    fn gap_60s_contributes_no_giant_return() {
        // 150 contiguous cells (0..150s), a 60s gap (150..210s), then 150
        // more cells (210..360s) — same step pattern as T1.
        let prices = grid_step_prices(300, 100_000.0, 0.0001);
        let mut samples: Vec<PriceSample> = Vec::new();
        for (i, price) in prices.iter().enumerate().take(151) {
            samples.push(PriceSample {
                ts_ms: (i as i64) * 1000,
                price: Decimal::from_f64(*price).unwrap(),
            });
        }
        // Gap: jump from 150s to 210s (60s without samples).
        for (i, price) in prices.iter().enumerate().skip(150).take(151) {
            let ts = (i as i64) * 1000 + 60_000;
            samples.push(PriceSample {
                ts_ms: ts,
                price: Decimal::from_f64(*price).unwrap(),
            });
        }

        let sigma =
            realized_vol_per_sec(&samples, 1, 360, 30).expect("must return Some despite the gap");
        // The gap breaks ONE pair (150s→210s are not consecutive cells): the rest
        // is still the same 0.0001 step → the result must not blow up.
        let target = dec!(0.0001);
        assert!(
            sigma < target * dec!(1.5),
            "the gap must not contribute a giant return: sigma={sigma}"
        );
    }

    #[test]
    fn fewer_than_min_pairs_gives_none() {
        // 20 samples at 1s ⇒ 19 pairs < min_pairs=30.
        let samples: Vec<PriceSample> = (0..20i64)
            .map(|i| PriceSample {
                ts_ms: i * 1000,
                price: dec!(100_000) + Decimal::from(i),
            })
            .collect();
        assert!(realized_vol_per_sec(&samples, 1, 300, 30).is_none());
    }

    #[test]
    fn constant_price_gives_none() {
        let samples: Vec<PriceSample> = (0..100i64)
            .map(|i| PriceSample {
                ts_ms: i * 1000,
                price: dec!(100_000),
            })
            .collect();
        assert!(realized_vol_per_sec(&samples, 1, 300, 30).is_none());
    }

    #[test]
    fn invalid_params_give_none() {
        let samples = grid_samples();
        assert!(
            realized_vol_per_sec(&samples, 0, 300, 30).is_none(),
            "grid_secs=0"
        );
        assert!(
            realized_vol_per_sec(&samples, 1, 0, 30).is_none(),
            "window_secs=0"
        );
        assert!(
            realized_vol_per_sec(&samples, 1, 300, 1).is_none(),
            "min_pairs=1 < 2"
        );
        assert!(
            realized_vol_per_sec(&[], 1, 300, 30).is_none(),
            "empty slice"
        );
    }

    #[test]
    fn scaling_doubles_sigma_exactly() {
        let samples_1x = grid_samples();
        let prices_2x = grid_step_prices(300, 100_000.0, 0.0002); // doubled step
        let samples_2x: Vec<PriceSample> = prices_2x
            .iter()
            .enumerate()
            .map(|(i, &p)| PriceSample {
                ts_ms: (i as i64) * 1000,
                price: Decimal::from_f64(p).unwrap(),
            })
            .collect();

        let sigma_1x = realized_vol_per_sec(&samples_1x, 1, 300, 30).unwrap();
        let sigma_2x = realized_vol_per_sec(&samples_2x, 1, 300, 30).unwrap();

        let ratio = sigma_2x / sigma_1x;
        assert!(
            (ratio - dec!(2)).abs() < dec!(0.01),
            "ratio={ratio}, expected ≈2"
        );
    }

    #[test]
    fn same_cell_last_sample_wins() {
        // cell 0: two samples (100 then 200) — 200 wins.
        // cell 1: one sample (300).
        // cell 2: two samples (500 then 600) — 600 wins.
        // cell 3: one sample (900).
        // Compare against a "clean" fixture with a single sample per cell
        // (already with the winning price): they must give EXACTLY the same σ̂, which
        // confirms that the duplicate is resolved to the last sample.
        let with_dupes = vec![
            PriceSample {
                ts_ms: 0,
                price: dec!(100),
            },
            PriceSample {
                ts_ms: 500,
                price: dec!(200),
            }, // cell 0, wins
            PriceSample {
                ts_ms: 1000,
                price: dec!(300),
            }, // cell 1
            PriceSample {
                ts_ms: 2000,
                price: dec!(500),
            },
            PriceSample {
                ts_ms: 2500,
                price: dec!(600),
            }, // cell 2, wins
            PriceSample {
                ts_ms: 3000,
                price: dec!(900),
            }, // cell 3
        ];
        let clean = vec![
            PriceSample {
                ts_ms: 0,
                price: dec!(200),
            }, // cell 0 (winner)
            PriceSample {
                ts_ms: 1000,
                price: dec!(300),
            }, // cell 1
            PriceSample {
                ts_ms: 2000,
                price: dec!(600),
            }, // cell 2 (winner)
            PriceSample {
                ts_ms: 3000,
                price: dec!(900),
            }, // cell 3
        ];
        let sigma_dupes = realized_vol_per_sec(&with_dupes, 1, 300, 2).expect("3 valid pairs");
        let sigma_clean = realized_vol_per_sec(&clean, 1, 300, 2).expect("3 valid pairs");
        assert_eq!(
            sigma_dupes, sigma_clean,
            "the cell must resolve to the last sample, not the first"
        );
    }
}
