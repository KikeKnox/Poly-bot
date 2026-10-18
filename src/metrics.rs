use metrics_exporter_prometheus::PrometheusHandle;

/// Installs the global Prometheus recorder and returns the handle used to serve /metrics.
///
/// Call it in main.rs before spawning tasks so that metrics emitted by
/// market_feed and other tasks are not lost in the noop recorder.
pub fn install_prometheus_recorder() -> anyhow::Result<PrometheusHandle> {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::set_global_recorder(recorder)
        .map_err(|e| anyhow::anyhow!("failed to install Prometheus recorder: {e}"))?;
    Ok(handle)
}
