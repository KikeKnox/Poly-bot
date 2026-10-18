//! Price and volatility oracle.
//!
//! - `oracle::price` — multi-asset price feed via Binance WS.
//!   Publishes a `watch<AssetPrice>` per asset.
//! - `oracle::vol`   — shared vol oracle.
//!   Samples `price` and publishes a `watch<VolSnapshot>` per asset with EWMA σ.
//!   Strategies use this watch instead of keeping their own ring buffer.
//! - `oracle::turbulence` — per-asset turbulence index (univariate
//!   Mahalanobis over log-returns, from FinRL's turbulence index).
//!   Pure function that `oracle::vol` evaluates over its `sec_samples` buffer and
//!   publishes as `VolSnapshot::turbulence` — a regime signal for the
//!   strategies.

pub mod chainlink_twap;
pub mod liquidations;
pub mod price;
pub mod turbulence;
pub mod vol;

/// Extracts the asset (lowercase) from the first segment of a slug or pattern.
///
/// `"btc-updown"` → `Some("btc")`, `"eth-updown-5m-…"` → `Some("eth")`.
/// `None` if the slug contains no '-' or the first segment is empty.
pub fn asset_from_slug(slug: &str) -> Option<String> {
    let seg = slug.split('-').next()?;
    if seg.is_empty() {
        None
    } else {
        Some(seg.to_lowercase())
    }
}
