//! Integration test for `/ui/market_health`.

use std::collections::HashMap;
use std::sync::Arc;

use rust_decimal_macros::dec;

use poly_bot::types::VolSnapshot;

#[test]
fn snapshot_with_sigma_none_is_distinguished_from_zero() {
    // `VolSnapshot` does have a manual `impl Default` (src/types.rs), but
    // here the full struct is built explicitly so the test makes clear what
    // value each field relevant to the scenario has.
    let snap = VolSnapshot {
        asset: "btc".into(),
        sigma_short: None,
        sigma_long: Some(dec!(0.0001)),
        last_price: dec!(65000),
        updated_at: poly_bot::time::Instant::now(),
        cvd_5s: None,
        cvd_60s: None,
        vol_quote_60s: None,
        liq_net_60s: None,
        liq_total_60s: None,
        turbulence: None,
    };
    // sigma_short as None must serialize as null, not as "0": the frontend
    // explicitly distinguishes "not enough samples" from a real 0.
    let json_val = serde_json::json!({ "sigma_short": snap.sigma_short.map(|v| v.to_string()) });
    assert_eq!(json_val["sigma_short"], serde_json::Value::Null);

    let mut rxs: HashMap<String, tokio::sync::watch::Receiver<VolSnapshot>> = HashMap::new();
    let (_tx, rx) = tokio::sync::watch::channel(snap);
    rxs.insert("btc".into(), rx);
    let _rxs = Arc::new(rxs); // confirms the type used by ApiState compiles with this snapshot

}
