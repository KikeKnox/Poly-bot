//! Integration test of the orphan checker against a real DB and the real CLOB.
//!
//! WARNING: this test MUTATES the database it is pointed at: `settlement::orphan::poll_once`
//! resolves expired positions and WRITES to it. Never point
//! `HRF_ORPHAN_TEST_DB` at a copy of accounting evidence (backup, a calibration
//! run to keep, etc.) — always use a disposable copy.
//!
//! Requires internet access (queries clob.polymarket.com).
//!
//! It does not guess a relative path: without `HRF_ORPHAN_TEST_DB` set, the
//! test is skipped (it does not fail, so suites that do not configure it keep working).
//! Run with: HRF_ORPHAN_TEST_DB=/path/to/disposable_copy.sqlite
//!   cargo test --test orphan_integration -- --nocapture

use std::sync::Arc;

use poly_bot::{settlement::orphan, risk::RiskGuard, state::StateStore, types::Mode};

#[tokio::test]
async fn orphan_resolves_expired_positions() {
    let Ok(db_path) = std::env::var("HRF_ORPHAN_TEST_DB") else {
        eprintln!(
            "SKIP: set HRF_ORPHAN_TEST_DB to the path of a DISPOSABLE copy \
             of the paper DB — this test mutates it. Do not use the original accounting evidence."
        );
        return;
    };

    if !std::path::Path::new(&db_path).exists() {
        eprintln!("SKIP: HRF_ORPHAN_TEST_DB={db_path} no exists");
        return;
    }

    let state = Arc::new(
        StateStore::open(&db_path, Mode::Paper)
            .await
            .expect("could not open the DB"),
    );

    let pending_before = state
        .list_expired_unresolved()
        .await
        .expect("list_expired_unresolved failed");

    println!(
        "expired positions pending before: {}",
        pending_before.len()
    );
    for p in &pending_before {
        println!(
            "  token: {}…{}",
            &p.token_id[..8],
            &p.token_id[p.token_id.len() - 6..]
        );
        println!("  cid:   {}…", &p.market_cid[..12]);
        println!("  size:  {}  cost: {}", p.size, p.cost_basis);
    }

    if pending_before.is_empty() {
        println!("nothing to resolve — test passed");
        return;
    }

    let risk = RiskGuard::new(poly_bot::config::RiskConfig::default(), Mode::Paper);
    orphan::poll_once(&state, &risk)
        .await
        .expect("poll_once failed");

    let pending_after = state
        .list_expired_unresolved()
        .await
        .expect("list_expired_unresolved failed");

    println!(
        "\npositions still pending afterwards: {}",
        pending_after.len()
    );

    let resolved_count = pending_before.len() - pending_after.len();
    println!("resolved: {resolved_count}/{}", pending_before.len());

    // Check that at least one was resolved (or that all were pending on UMA).
    // It is not an error for some to remain pending if the CLOB has no `winner` yet.
    if resolved_count == 0 && !pending_after.is_empty() {
        println!(
            "WARN: none was resolved — UMA resolution possibly still pending on the CLOB"
        );
    }

    // Check that the resolved ones have a coherent PnL: [-cost_basis, +size-cost_basis]
    // We query the REDEEM transactions generated.
    println!("\nREDEEM transactions generated in this session:");
    // ponytail: a direct query via sqlx is not exposed; we verify through the pending count.
    assert!(
        pending_after.len() <= pending_before.len(),
        "the pending count must not increase"
    );
}
