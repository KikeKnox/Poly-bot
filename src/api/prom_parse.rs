//! Minimal parser for Prometheus exposition-format text, used to
//! group skip counters by strategy without having to expose each
//! counter by hand in a new endpoint every time one is added.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct PromMetric {
    pub name: String,
    pub labels: HashMap<String, String>,
    pub value: f64,
}

/// Parses the full text of `PrometheusHandle::render()`. Ignores comment
/// lines (`# HELP`, `# TYPE`) and blank lines. A line that does not match
/// the expected format is silently discarded (better to show fewer
/// metrics than to panic over output we do not control).
pub fn parse_prometheus_text(text: &str) -> Vec<PromMetric> {
    text.lines().filter_map(parse_line).collect()
}

fn parse_line(line: &str) -> Option<PromMetric> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let (name_and_labels, value_str) = line.rsplit_once(' ')?;
    let value: f64 = value_str.parse().ok()?;

    if let Some(brace_start) = name_and_labels.find('{') {
        let name = name_and_labels[..brace_start].to_string();
        let labels_str = name_and_labels[brace_start + 1..].strip_suffix('}')?;
        let labels = parse_labels(labels_str);
        Some(PromMetric {
            name,
            labels,
            value,
        })
    } else {
        Some(PromMetric {
            name: name_and_labels.to_string(),
            labels: HashMap::new(),
            value,
        })
    }
}

fn parse_labels(labels_str: &str) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    if labels_str.is_empty() {
        return labels;
    }
    // Naive split on `,`: enough because the label values in this
    // repo are asset/strategy names with no commas or inner quotes.
    for pair in labels_str.split(',') {
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        let v = v.trim().trim_matches('"');
        labels.insert(k.trim().to_string(), v.to_string());
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_counter_without_labels() {
        let text = "hrf_arb_skip_edge_total 42\n";
        let metrics = parse_prometheus_text(text);
        assert_eq!(
            metrics,
            vec![PromMetric {
                name: "hrf_arb_skip_edge_total".into(),
                labels: HashMap::new(),
                value: 42.0
            }]
        );
    }

    #[test]
    fn parses_counter_with_labels() {
        let text = r#"hrf_vol_sigma_none_total{asset="btc",window="short"} 7"#;
        let metrics = parse_prometheus_text(text);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "hrf_vol_sigma_none_total");
        assert_eq!(metrics[0].labels.get("asset"), Some(&"btc".to_string()));
        assert_eq!(metrics[0].labels.get("window"), Some(&"short".to_string()));
        assert_eq!(metrics[0].value, 7.0);
    }

    #[test]
    fn ignores_help_and_type_comments_and_empty_lines() {
        let text = "# HELP hrf_x algo\n# TYPE hrf_x counter\n\nhrf_x 1\n";
        let metrics = parse_prometheus_text(text);
        assert_eq!(
            metrics,
            vec![PromMetric {
                name: "hrf_x".into(),
                labels: HashMap::new(),
                value: 1.0
            }]
        );
    }

    #[test]
    fn malformed_line_is_discarded_without_panic() {
        let text = "this is not a valid metric\nhrf_ok 5\n";
        let metrics = parse_prometheus_text(text);
        assert_eq!(
            metrics,
            vec![PromMetric {
                name: "hrf_ok".into(),
                labels: HashMap::new(),
                value: 5.0
            }]
        );
    }
}
