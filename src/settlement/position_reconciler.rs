//! Position reconciler (live only) — periodically compares our open
//! positions against Polymarket's truth (Data API, which indexes the real
//! on-chain state) and alerts on drift.
//!
//! It corrects nothing automatically: if the Data API returns stale
//! state or lags in indexing, "correcting" with that data would
//! replace a good state with a bad one. For now it is detection +
//! metrics only — an auto-correction step (ledger or DB) is a future extension
//! once the signal is trusted.

use std::{collections::HashMap, sync::Arc};

use polymarket_client_sdk_v2::data::{types::request::PositionsRequest, Client};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::watch;
use tracing::{info, warn};

use crate::state::StateStore;
use crate::types::Position;

const POLL_SECS: u64 = 90;
// The Data API indexer takes a few seconds to reflect recent fills.
// Reuse the poll interval itself as a margin: a position newer than
// that may legitimately not be indexed yet — it is not real drift.
const MIN_POSITION_AGE_SECS: i64 = POLL_SECS as i64;
// Tolerance for rounding differences between our accounting and the indexer.
const SIZE_DRIFT_TOLERANCE: Decimal = dec!(0.01);

#[derive(Debug, PartialEq, Eq)]
enum MismatchKind {
    /// In our DB, with no backing on Polymarket.
    Zombie,
    /// On Polymarket, with no record in our DB — a fill that was never recorded.
    Ghost,
    /// Same token on both sides, different size beyond the tolerance.
    SizeDrift,
}

#[derive(Debug, PartialEq, Eq)]
struct Mismatch {
    kind: MismatchKind,
    cid: String,
    token: String,
}

/// Compares local positions against Polymarket's truth. Pure function —
/// no I/O — so it can be tested with fixed data (pattern from `fair_value.rs`).
fn diff_positions(
    local: &[Position],
    mut remote: HashMap<(String, String), Decimal>,
    now_ms: i64,
) -> Vec<Mismatch> {
    let mut mismatches = Vec::new();

    // Local rows are per (token, strategy) and the Data API aggregates by
    // (cid, token): without aggregating here, two strategies on the same token
    // produced a SizeDrift (each partial row compared against the remote
    // total) and a Zombie (the second row no longer finds the key
    // consumed by the first) — both false positives. The aggregated `opened_ts_ms`
    // = the most recent: it is the piece that may not be indexed yet.
    let mut local_agg: HashMap<(String, String), (Decimal, i64)> = HashMap::new();
    for pos in local {
        let e = local_agg
            .entry((pos.market_cid.clone(), pos.token_id.clone()))
            .or_default();
        e.0 += pos.size;
        e.1 = e.1.max(pos.opened_ts_ms);
    }

    for (key, (size, opened_ts_ms)) in local_agg {
        if now_ms - opened_ts_ms < MIN_POSITION_AGE_SECS * 1000 {
            remote.remove(&key); // too recent to require it to be indexed already
            continue;
        }
        match remote.remove(&key) {
            None => mismatches.push(Mismatch {
                kind: MismatchKind::Zombie,
                cid: key.0,
                token: key.1,
            }),
            Some(remote_size) => {
                if (remote_size - size).abs() > SIZE_DRIFT_TOLERANCE {
                    mismatches.push(Mismatch {
                        kind: MismatchKind::SizeDrift,
                        cid: key.0,
                        token: key.1,
                    });
                }
            }
        }
    }

    for (cid, token) in remote.into_keys() {
        mismatches.push(Mismatch {
            kind: MismatchKind::Ghost,
            cid,
            token,
        });
    }

    mismatches
}

pub async fn run(own_address: String, state: Arc<StateStore>, mut shutdown: watch::Receiver<bool>) {
    let address: polymarket_client_sdk_v2::types::Address = match own_address.parse() {
        Ok(a) => a,
        Err(e) => {
            warn!(own_address, error = %e, "position_reconciler: invalid address — disabled");
            return;
        }
    };
    let client = Client::default();
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(POLL_SECS));
    ticker.tick().await; // skip the first, immediate tick

    info!(
        own_address,
        poll_secs = POLL_SECS,
        "position_reconciler: started"
    );

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => { if *shutdown.borrow() { break; } }
            _ = ticker.tick() => { poll(&client, address, &state).await; }
        }
    }
    info!("position_reconciler: stopped");
}

async fn poll(
    client: &Client,
    address: polymarket_client_sdk_v2::types::Address,
    state: &StateStore,
) {
    let local = match state.list_open_positions().await {
        Ok(v) => v,
        Err(e) => {
            warn!("position_reconciler: error reading local positions: {e}");
            return;
        }
    };

    let req = PositionsRequest::builder().user(address).build();
    let remote = match client.positions(&req).await {
        Ok(v) => v,
        Err(e) => {
            warn!("position_reconciler: error querying the Data API: {e}");
            return;
        }
    };

    let remote_by_key: HashMap<(String, String), Decimal> = remote
        .into_iter()
        .map(|p| ((p.condition_id.to_string(), p.asset.to_string()), p.size))
        .collect();

    let now_ms = chrono::Utc::now().timestamp_millis();
    for m in diff_positions(&local, remote_by_key, now_ms) {
        let kind = match m.kind {
            MismatchKind::Zombie => "zombie",
            MismatchKind::Ghost => "ghost",
            MismatchKind::SizeDrift => "size_drift",
        };
        warn!(cid = %m.cid, token = %m.token, kind, "position_reconciler: discrepancy with Polymarket");
        metrics::counter!("hrf_position_reconcile_mismatch_total", "kind" => kind).increment(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn pos(cid: &str, token: &str, size: Decimal, opened_ts_ms: i64) -> Position {
        Position {
            token_id: token.into(),
            market_cid: cid.into(),
            slug: String::new(),
            strategy: "arb".into(),
            size,
            avg_price: dec!(0.5),
            cost_basis: dec!(1),
            opened_ts_ms,
            clear_at: 0,
            neg_risk: false,
            bail_sold: false,
            outcome_won: None,
            winner_label: None,
            outcome_label: None,
            redeemed: false,
            redeem_tx: None,
        }
    }

    #[test]
    fn matches_without_discrepancies() {
        let local = vec![pos("cid-1", "tok-1", dec!(10), 0)];
        let remote = HashMap::from([(("cid-1".to_string(), "tok-1".to_string()), dec!(10))]);
        assert!(diff_positions(&local, remote, 1_000_000_000).is_empty());
    }

    #[test]
    fn local_zombie_without_remote_backup() {
        let local = vec![pos("cid-1", "tok-1", dec!(10), 0)];
        let remote = HashMap::new();
        let out = diff_positions(&local, remote, 1_000_000_000);
        assert_eq!(
            out,
            vec![Mismatch {
                kind: MismatchKind::Zombie,
                cid: "cid-1".into(),
                token: "tok-1".into()
            }]
        );
    }

    #[test]
    fn remote_ghost_without_local_record() {
        let local = vec![];
        let remote = HashMap::from([(("cid-1".to_string(), "tok-1".to_string()), dec!(5))]);
        let out = diff_positions(&local, remote, 1_000_000_000);
        assert_eq!(
            out,
            vec![Mismatch {
                kind: MismatchKind::Ghost,
                cid: "cid-1".into(),
                token: "tok-1".into()
            }]
        );
    }

    #[test]
    fn size_drift_beyond_tolerance() {
        let local = vec![pos("cid-1", "tok-1", dec!(10), 0)];
        let remote = HashMap::from([(("cid-1".to_string(), "tok-1".to_string()), dec!(8))]);
        let out = diff_positions(&local, remote, 1_000_000_000);
        assert_eq!(
            out,
            vec![Mismatch {
                kind: MismatchKind::SizeDrift,
                cid: "cid-1".into(),
                token: "tok-1".into()
            }]
        );
    }

    #[test]
    fn drift_within_tolerance_does_not_report() {
        let local = vec![pos("cid-1", "tok-1", dec!(10), 0)];
        let remote = HashMap::from([(("cid-1".to_string(), "tok-1".to_string()), dec!(10.005))]);
        assert!(diff_positions(&local, remote, 1_000_000_000).is_empty());
    }

    /// Two strategies on the same token — local rows are
    /// aggregated before comparing against the remote total (per token). Without
    /// aggregation it produced a SizeDrift (partial row vs total) and a Zombie
    /// (remote key consumed by the first row).
    #[test]
    fn multi_strategy_same_token_aggregates_before_comparing() {
        let mut p1 = pos("cid-1", "tok-1", dec!(10), 0);
        p1.strategy = "arb".into();
        let mut p2 = pos("cid-1", "tok-1", dec!(5), 0);
        p2.strategy = "nc".into();
        let remote = HashMap::from([(("cid-1".to_string(), "tok-1".to_string()), dec!(15))]);
        assert!(
            diff_positions(&[p1, p2], remote, 1_000_000_000).is_empty(),
            "10 + 5 = 15 must match the aggregated remote"
        );
    }

    /// And if the SUM differs from the remote, there is real drift.
    #[test]
    fn multi_strategy_same_token_different_sum_is_drift() {
        let mut p1 = pos("cid-1", "tok-1", dec!(10), 0);
        p1.strategy = "arb".into();
        let mut p2 = pos("cid-1", "tok-1", dec!(5), 0);
        p2.strategy = "nc".into();
        let remote = HashMap::from([(("cid-1".to_string(), "tok-1".to_string()), dec!(12))]);
        let out = diff_positions(&[p1, p2], remote, 1_000_000_000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, MismatchKind::SizeDrift);
    }

    /// Local position too recent: even if it has no remote backing
    /// yet (the indexer lags), it must not be reported as a zombie.
    #[test]
    fn recent_position_does_not_generate_false_positive() {
        let now = 1_000_000_000i64;
        let local = vec![pos("cid-1", "tok-1", dec!(10), now - 1000)]; // 1s old
        let remote = HashMap::new();
        assert!(diff_positions(&local, remote, now).is_empty());
    }
}
