#[test]
fn funnel_reads_real_counters_from_prometheus_registry() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        metrics::counter!("hrf_nc_skip_z_total").increment(3);
        metrics::counter!("hrf_nc_skip_imbalance_total").increment(7);
    });

    let text = handle.render();
    let metrics = poly_bot::api::prom_parse::parse_prometheus_text(&text);
    let funnel =
        poly_bot::api::strategy_funnel::funnel_for_strategy(&metrics, "near_certain").unwrap();

    assert!(funnel
        .iter()
        .any(|c| c.name == "hrf_nc_skip_z_total" && c.value == 3.0));
    assert!(funnel
        .iter()
        .any(|c| c.name == "hrf_nc_skip_imbalance_total" && c.value == 7.0));
}
