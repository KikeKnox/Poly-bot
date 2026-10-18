//! Integration test for the kill + limits combination that feeds the
//! `/ui/risk` endpoint. The handler is a direct serialization of these
//! `RiskGuard` getters, each already covered individually; this test
//! confirms that the combination used by the Risk page works.

use rust_decimal_macros::dec;

#[tokio::test]
async fn ui_risk_reflects_kill_and_strategy_limits() {
    let risk = poly_bot::risk::RiskGuard::new(
        poly_bot::config::RiskConfig::default(),
        poly_bot::types::Mode::Paper,
    );
    let id = poly_bot::types::StrategyId::NearCertain;
    risk.set_strategy_limits(id, 20, dec!(50), dec!(10));
    risk.kill_strategy(id, "risk test");

    assert!(risk.is_strategy_killed(id));
    assert_eq!(
        risk.strategy_kill_reason(id),
        Some("risk test".to_string())
    );
    let (rate, usdc) = risk.get_strategy_limits(id);
    assert_eq!(rate, 20);
    assert_eq!(usdc, dec!(50));
}
