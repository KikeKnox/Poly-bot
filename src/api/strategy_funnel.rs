//! Groups the Prometheus skip counters by strategy using the metric-name
//! prefix. This avoids listing each counter by hand;
//! a new counter under an already-mapped prefix shows up on its own. A counter of
//! a NEW strategy does need a new entry in `PREFIX_BY_STRATEGY`
//! (ver test `unknown_prefix_breaks_nothing`).

use super::prom_parse::PromMetric;

/// Metric-name prefix per strategy. Keep in sync with the
/// `metrics::counter!("hrf_<prefix>_skip_..._total")` of each `strategy/*.rs`.
const PREFIX_BY_STRATEGY: &[(&str, &str)] = &[
    ("arb", "hrf_arb_"),
    ("directional", "hrf_dir_"),
    ("near_certain", "hrf_nc_"),
    ("near_expiry", "hrf_ne_"),
    ("market_making", "hrf_mm_"),
    ("parity_maker", "hrf_pm_"),
    ("drift_box", "hrf_driftbox_"),
    ("open_momentum", "hrf_om_"),
    ("theta_capture", "hrf_tc_"),
    ("post_close_sweep", "hrf_pcs2_"),
    ("reward_farm", "hrf_rf_"),
];

#[derive(Debug, Clone, serde::Serialize)]
pub struct FunnelCounter {
    pub name: String,
    pub value: f64,
}

/// Display name for a metric: if it has labels (e.g.
/// `reason="late"`), they are added to the name so that variants of the same
/// counter (`hrf_pcs2_skip_total{reason="late"}` vs. `{reason="no_ref"}`)
/// do not collapse into the same funnel row. Deterministic label order
/// (important for tests and so the frontend does not make the order "flicker").
fn display_name(m: &PromMetric) -> String {
    if m.labels.is_empty() {
        m.name.clone()
    } else {
        let mut pairs: Vec<String> = m.labels.iter().map(|(k, v)| format!("{k}={v}")).collect();
        pairs.sort();
        format!("{}{{{}}}", m.name, pairs.join(","))
    }
}

/// `None` if the strategy has no mapped prefix (scan-only tool,
/// or strategy with no skip counters yet); the caller responds empty,
/// it is not an error.
pub fn funnel_for_strategy(metrics: &[PromMetric], strategy: &str) -> Option<Vec<FunnelCounter>> {
    let prefix = PREFIX_BY_STRATEGY.iter().find(|(s, _)| *s == strategy)?.1;
    Some(
        metrics
            .iter()
            .filter(|m| m.name.starts_with(prefix) && m.name.contains("skip"))
            .map(|m| FunnelCounter {
                name: display_name(m),
                value: m.value,
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn metric(name: &str, value: f64) -> PromMetric {
        PromMetric {
            name: name.into(),
            labels: HashMap::new(),
            value,
        }
    }

    #[test]
    fn filters_only_counters_with_prefix_and_skip_in_name() {
        let metrics = vec![
            metric("hrf_nc_skip_z_total", 10.0),
            metric("hrf_nc_skip_imbalance_total", 20.0),
            metric("hrf_om_skip_age_total", 5.0),
            metric("hrf_nc_buy_total", 3.0), // same prefix, not a skip, so excluded
        ];
        let funnel = funnel_for_strategy(&metrics, "near_certain").unwrap();
        assert_eq!(funnel.len(), 2);
        assert!(funnel
            .iter()
            .any(|c| c.name == "hrf_nc_skip_z_total" && c.value == 10.0));
        assert!(funnel
            .iter()
            .any(|c| c.name == "hrf_nc_skip_imbalance_total" && c.value == 20.0));
    }

    #[test]
    fn unknown_prefix_breaks_nothing() {
        let metrics = vec![metric("hrf_nc_skip_z_total", 10.0)];
        assert!(funnel_for_strategy(&metrics, "follower").is_none());
    }

    fn metric_with_labels(name: &str, labels: &[(&str, &str)], value: f64) -> PromMetric {
        PromMetric {
            name: name.into(),
            labels: labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            value,
        }
    }

    /// Two counters with the same `name` but different `reason` labels must
    /// produce different `FunnelCounter.name` values so they do not collapse into the
    /// same funnel row (bug: 6 variants of `hrf_pcs2_skip_total`
    /// indistinguishable).
    #[test]
    fn different_labels_produce_different_names() {
        let metrics = vec![
            metric_with_labels("hrf_pcs2_skip_total", &[("reason", "late")], 3.0),
            metric_with_labels("hrf_pcs2_skip_total", &[("reason", "no_ref")], 5.0),
        ];
        let funnel = funnel_for_strategy(&metrics, "post_close_sweep").unwrap();
        assert_eq!(funnel.len(), 2);
        let names: Vec<&str> = funnel.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"hrf_pcs2_skip_total{reason=late}"));
        assert!(names.contains(&"hrf_pcs2_skip_total{reason=no_ref}"));
        assert_ne!(names[0], names[1]);
    }
}
