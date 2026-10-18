#[tokio::test]
async fn kill_global_is_recorded_in_ui_alerts() {
    let risk = poly_bot::risk::RiskGuard::new(
        poly_bot::config::RiskConfig::default(),
        poly_bot::types::Mode::Paper,
    );
    let alert_log = poly_bot::api::alert_log::AlertLog::new(50);
    risk.attach_alert_log(alert_log.clone());

    risk.kill_global("integration test");

    let recent = alert_log.recent(5);
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].severity, "critical");
    assert!(recent[0].message.contains("integration test"));
}
