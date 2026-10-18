//! Sink for scan observations.
//!
//! LadderArb and VolSurface (`crate::tools`) are scan-only tools: they detect
//! opportunities but do not trade. Their gates require ≥1 week of real data;
//! this sink persists them in `scan_observations` (fill_sink pattern: mpsc channel
//! → SQLite, off the hot path).

use std::sync::Arc;

use rust_decimal::Decimal;
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::{state::StateStore, tools::ToolId};

/// Observation of an opportunity detected by a scan-only tool.
pub struct ScanObs {
    pub strategy: ToolId,
    /// Slug or pair of slugs that identifies the opportunity.
    pub market_key: String,
    pub kind: &'static str,
    /// Edge (ladder) or vol ratio (VS) — what the gate will aggregate.
    pub edge: Decimal,
    /// JSON with the complete detail.
    pub payload: String,
}

pub async fn run(
    state: Arc<StateStore>,
    mut rx: mpsc::UnboundedReceiver<ScanObs>,
    mut shutdown: watch::Receiver<bool>,
) {
    info!("scan_sink: starting");
    loop {
        tokio::select! {
            obs = rx.recv() => match obs {
                Some(o) => {
                    if let Err(e) = state.record_scan(
                        chrono::Utc::now().timestamp_millis(),
                        &o.strategy.to_string(),
                        &o.market_key,
                        o.kind,
                        o.edge,
                        &o.payload,
                    ).await {
                        warn!("scan_sink: error persisting observation: {e}");
                    }
                }
                None => {
                    info!("scan_sink: channel closed, exiting");
                    return;
                }
            },
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("scan_sink: shutdown received");
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Mode;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn scan_sink_persists_observations() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (tx, rx) = mpsc::unbounded_channel();
        let (_kill_tx, kill_rx) = watch::channel(false);
        tokio::spawn(run(state.clone(), rx, kill_rx));

        tx.send(ScanObs {
            strategy: ToolId::LadderArb,
            market_key: "btc-updown-95000|btc-updown-96000".into(),
            kind: "ladder_opp",
            edge: dec!(0.021),
            payload: r#"{"lo_mid":"0.45","hi_mid":"0.52"}"#.into(),
        })
        .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let n = state.scan_observations_count("ladder_arb").await.unwrap();
        assert_eq!(n, 1);
    }
}
