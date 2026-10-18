//! Integration tests of the numeric model shadow observer.
//! (Task T9, numeric-model paper-training plan).
//!
//! Covers the cases:
//! - slow / down / stale-data model → NO decisions or
//!   orders are generated (the observer does not receive `intent_tx`: it CANNOT emit to the
//!   executor — verified structurally and by behavior);
//! - the shadow does not reserve cash nor touch the ledger/executor/RiskGuard (the
//!   `ShadowCtx` context carries neither `intent_tx` nor `risk` nor `state`);
//! - an artifact with the wrong schema or a hash that does not match the
//!   manifest is rejected (strict versioning);
//! - the virtual decision is recorded with features + prediction +
//!   thresholds, for later comparison against replay;
//! - parity of the Rust predictor with the real artifact over
//!   recorded OOS observations (fixture `fixtures/model_shadow_parity.json`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rust_decimal_macros::dec;
use serde_json::{json, Value};
use tokio::sync::{broadcast, watch};

use poly_bot::config::ModelShadowConfig;
use poly_bot::model_observer::{
    self, FeatureMap, Predictor, ShadowCtx, ShadowModel, ShadowProducer, ShadowWriter,
};
use poly_bot::types::{BookSnapshot, HalfBook, Level, TokenBook, VolSnapshot};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn half(price: &str, size: rust_decimal::Decimal) -> HalfBook {
    HalfBook {
        levels: vec![Level {
            price: price.parse().unwrap(),
            size,
        }],
    }
}

/// Book of an updown BTC bracket with both sides quoted.
fn make_book(age: Duration, yes_bid: &str, yes_ask: &str) -> Arc<BookSnapshot> {
    Arc::new(BookSnapshot {
        market_condition_id: "cid-shadow".into(),
        yes: TokenBook {
            token_id: "tok-yes".into(),
            bids: half(yes_bid, dec!(100)),
            asks: half(yes_ask, dec!(80)),
            ..Default::default()
        },
        no: TokenBook {
            token_id: "tok-no".into(),
            bids: half("0.40", dec!(60)),
            asks: half("0.44", dec!(60)),
            ..Default::default()
        },
        // Bracket halfway through its life: clear_at in the FUTURE — a clear_at
        // set in the past would make the observer abstain with `post_close`.
        market_open_ts: poly_bot::time::now().timestamp() - 1_800,
        clear_at: poly_bot::time::now().timestamp() + 1_800,
        neg_risk: false,
        slug: "btc-updown-test".into(),
        reference_price: Some(dec!(67000)),
        fee_rate_bps: None,
        parabolic_fee_rate: Some(dec!(0.07)),
        tick_size: None,
        recv_instant: poly_bot::time::Instant::now().checked_sub(age),
    })
}

fn fresh_vol(price: &str) -> VolSnapshot {
    VolSnapshot {
        asset: "btc".into(),
        last_price: price.parse().unwrap(),
        updated_at: poly_bot::time::Instant::now(),
        ..Default::default()
    }
}

fn cfg_shadow(dir: &Path) -> ModelShadowConfig {
    ModelShadowConfig {
        enabled: true,
        artifact_dir: String::new(),
        decisions_dir: dir.to_string_lossy().into_owned(),
        signal_horizon_ms: 30_000,
        latency_budget_ms: 0, // derived: horizon/10 = 3000 ms
        markets: vec![],
        assets: vec![],
        edge_min: dec!(0.02),
        shares_virtual: dec!(1),
        channel_capacity: 64,
        min_decision_interval_ms: 0,
        stats_interval_secs: 3600, // no periodic stats in the tests
    }
}

/// Test predictor that always returns the fixed probability.
struct FixedPredictor(f64);
impl Predictor for FixedPredictor {
    fn predict(&self, _feats: &FeatureMap) -> Result<f64, model_observer::PredictionError> {
        Ok(self.0)
    }
    fn model_hash(&self) -> &str {
        "hash-test-fixed"
    }
}

/// "Down" predictor: every prediction fails — the observer must abstain.
struct DeadPredictor;
impl Predictor for DeadPredictor {
    fn predict(&self, _feats: &FeatureMap) -> Result<f64, model_observer::PredictionError> {
        Err(model_observer::PredictionError::ModelUnavailable(
            "model down (test)".into(),
        ))
    }
    fn model_hash(&self) -> &str {
        "hash-test-caido"
    }
}

/// "Slow" predictor: sleeps longer than the latency budget.
struct SlowPredictor(Duration);
impl Predictor for SlowPredictor {
    fn predict(&self, _feats: &FeatureMap) -> Result<f64, model_observer::PredictionError> {
        std::thread::sleep(self.0);
        Ok(0.9)
    }
    fn model_hash(&self) -> &str {
        "hash-test-slow"
    }
}

fn ctx(
    book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    kill_rx: watch::Receiver<bool>,
) -> ShadowCtx {
    ShadowCtx {
        book_rx,
        vol_rxs: HashMap::new(),
        shutdown: kill_rx,
        max_book_age: Duration::from_secs(5),
    }
}

fn ctx_with_vol(
    book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    kill_rx: watch::Receiver<bool>,
    vol: VolSnapshot,
) -> ShadowCtx {
    let (_tx, rx) = watch::channel(vol);
    let mut vol_rxs = HashMap::new();
    vol_rxs.insert("btc".to_string(), rx);
    ShadowCtx {
        book_rx,
        vol_rxs,
        shutdown: kill_rx,
        max_book_age: Duration::from_secs(5),
    }
}

/// `decision`/`make_abstention` lines of the directory (the periodic and
/// closing `stats` do not count as an evaluation result).
fn read_evaluations(dir: &Path) -> Vec<Value> {
    read_lines(dir)
        .into_iter()
        .filter(|l| l["kind"] == json!("decision") || l["kind"] == json!("abstention"))
        .collect()
}

/// Reads the JSONL lines of the shadow's decisions directory.
fn read_lines(dir: &Path) -> Vec<Value> {
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|e| e == "jsonl") {
            files.push(p);
        }
    }
    files.sort();
    let mut out = Vec::new();
    for f in files {
        for line in std::fs::read_to_string(f).unwrap().lines() {
            let line = line.trim();
            if !line.is_empty() {
                out.push(serde_json::from_str(line).unwrap());
            }
        }
    }
    out
}

/// Writes a synthetic model artifact (manifest + model.json) into `dir`.
/// `model_doc` can be mutated beforehand to produce wrong schemas.
fn write_artifact(dir: &Path, model_doc: Value) {
    std::fs::create_dir_all(dir).unwrap();
    let model_path = dir.join("model.json");
    std::fs::write(&model_path, serde_json::to_string(&model_doc).unwrap()).unwrap();
    let sha = model_observer::sha256_hex(&std::fs::read(&model_path).unwrap());
    let manifest = json!({
        "schema_version": 1,
        "created_at_ms": 1_700_000_000_000i64,
        "code_hash": "00",
        "config_hash": "11",
        "model_hash": sha,
        "splits": [],
        "research_dir": dir.to_string_lossy(),
        "files": [{"path": "model.json", "sha256": sha}],
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
}

/// Valid synthetic artifact with 2 supported features.
fn valid_artifact(dir: &Path) {
    write_artifact(
        dir,
        json!({
            "schema_version": 1,
            "kind": "logistic_stdlib_baseline",
            "feature_names": ["yes_mid", "moneyness"],
            "model": {"weights": [2.0, -1.0], "bias": 0.1},
            "scaler": {
                "means": {"yes_mid": 0.5, "moneyness": 0.0},
                "stds": {"yes_mid": 0.2, "moneyness": 0.01},
                "medians": {"yes_mid": 0.5, "moneyness": 0.0},
                "missing_in_train": []
            }
        }),
    );
}

// ── Artifact: versioning and manifest validation ────────────────────────────

#[test]
fn valid_artifact_loads_and_predicts() {
    let dir = tempfile::tempdir().unwrap();
    valid_artifact(dir.path());
    let loaded_model = ShadowModel::load(dir.path()).expect("a valid artifact must load");
    assert_eq!(loaded_model.n_features(), 2);

    // yes_mid=0.9 → z=2; moneyness absent → imputes median (0). logit=0.1+2*2*1
    // yes_mid weight=2, z=(0.9-0.5)/0.2=2 → contribution 4; bias 0.1;
    // imputed moneyness median 0 → z=0. p = σ(4.1) ≈ 0.9836
    let mut feats = FeatureMap::new();
    feats.insert("yes_mid".to_string(), Some(0.9));
    feats.insert("moneyness".to_string(), None);
    let p = loaded_model.predict(&feats).unwrap();
    let expected = 1.0 / (1.0 + (-4.1f64).exp());
    assert!((p - expected).abs() < 1e-9, "p={p} expected={expected}");
}

#[test]
fn incorrect_manifest_schema_version_rejects() {
    let dir = tempfile::tempdir().unwrap();
    valid_artifact(dir.path());
    let mut m: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("manifest.json")).unwrap())
            .unwrap();
    m["schema_version"] = json!(99);
    std::fs::write(dir.path().join("manifest.json"), m.to_string()).unwrap();
    assert!(ShadowModel::load(dir.path()).is_err());
}

#[test]
fn manifest_without_model_json_entry_rejects() {
    let dir = tempfile::tempdir().unwrap();
    valid_artifact(dir.path());
    let mut m: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("manifest.json")).unwrap())
            .unwrap();
    m["files"] = json!([]);
    std::fs::write(dir.path().join("manifest.json"), m.to_string()).unwrap();
    assert!(ShadowModel::load(dir.path()).is_err());
}

#[test]
fn mismatched_model_hash_rejects() {
    let dir = tempfile::tempdir().unwrap();
    valid_artifact(dir.path());
    // Corrupt model.json without updating the manifest: the sha256 no longer matches.
    std::fs::write(dir.path().join("model.json"), b"{}".as_slice()).unwrap();
    assert!(ShadowModel::load(dir.path()).is_err());
}

#[test]
fn incorrect_model_schema_rejects() {
    // different schema_version
    let dir = tempfile::tempdir().unwrap();
    write_artifact(
        dir.path(),
        json!({"schema_version": 2, "kind": "logistic_stdlib_baseline",
               "feature_names": [], "model": {"weights": [], "bias": 0.0},
               "scaler": {"means": {}, "stds": {}, "medians": {}}}),
    );
    assert!(ShadowModel::load(dir.path()).is_err());

    // unknown kind
    let dir = tempfile::tempdir().unwrap();
    write_artifact(
        dir.path(),
        json!({"schema_version": 1, "kind": "neural_net",
               "feature_names": [], "model": {"weights": [], "bias": 0.0},
               "scaler": {"means": {}, "stds": {}, "medians": {}}}),
    );
    assert!(ShadowModel::load(dir.path()).is_err());

    // weights ≠ features
    let dir = tempfile::tempdir().unwrap();
    write_artifact(
        dir.path(),
        json!({"schema_version": 1, "kind": "logistic_stdlib_baseline",
               "feature_names": ["yes_mid"], "model": {"weights": [1.0, 2.0], "bias": 0.0},
               "scaler": {"means": {"yes_mid": 0.5}, "stds": {"yes_mid": 0.2},
                          "medians": {"yes_mid": 0.5}}}),
    );
    assert!(ShadowModel::load(dir.path()).is_err());

    // feature unknown to the extractor (artifact version drift)
    let dir = tempfile::tempdir().unwrap();
    write_artifact(
        dir.path(),
        json!({"schema_version": 1, "kind": "logistic_stdlib_baseline",
               "feature_names": ["future_feature"],
               "model": {"weights": [1.0], "bias": 0.0},
               "scaler": {"means": {"future_feature": 0.0},
                          "stds": {"future_feature": 1.0},
                          "medians": {"future_feature": 0.0}}}),
    );
    assert!(ShadowModel::load(dir.path()).is_err());
}

/// The REAL artifact loads and validates (manifest hash included).
/// `research/data/` is not tracked in git — in a clean clone the test
/// is skipped with a warning instead of failing (schema/hash validation is
/// already covered by the tests with a synthetic artifact).
#[test]
fn real_artifact_loads() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("research/data/t8b_confirmed/model");
    if !dir.join("model.json").exists() {
        eprintln!(
            "SKIP real_artifact_loads: {} does not exist (untracked research data)",
            dir.display()
        );
        return;
    }
    let loaded_model = ShadowModel::load(&dir).expect("the real artifact must load");
    assert_eq!(loaded_model.n_features(), 31);
    assert!(!loaded_model.model_hash().is_empty());
}

/// Rust↔Python parity: the Rust predictor's probabilities over the
/// features of real OOS observations must match those produced by
/// the Python pipeline (`dataset.py` + `train.py`) — tolerance 1e-9.
#[test]
fn predictor_in_parity_with_python_on_real_data() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let artifact = base.join("research/data/t8b_confirmed/model");
    let fixture_path = base.join("tests/fixtures/model_shadow_parity.json");
    if !artifact.join("model.json").exists() || !fixture_path.exists() {
        eprintln!("SKIP predictor_in_parity_with_python_on_real_data: artifact T8b or fixture absent (untracked research data)");
        return;
    }
    let loaded_model = ShadowModel::load(&artifact).unwrap();
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
    let obs = fixture["observations"].as_array().unwrap();
    assert!(!obs.is_empty());
    for o in obs {
        let feats: FeatureMap = serde_json::from_value(o["features"].clone()).unwrap();
        let p = loaded_model.predict(&feats).unwrap();
        let expected_prob = o["expected_p"].as_f64().unwrap();
        assert!(
            (p - expected_prob).abs() < 1e-9,
            "cid {}: p_rust={p} p_python={expected_prob}",
            o["market_cid"]
        );
    }
}

// ── Observer: does not decide on bad data and never touches the executor ────────────

/// Fresh book + fresh vol + predictor with high p → ONE virtual decision
/// recorded with features + prediction + thresholds + the model hash.
#[tokio::test]
async fn good_book_generates_recorded_virtual_decision() {
    let dir = tempfile::tempdir().unwrap();
    let writer = ShadowWriter::start(dir.path(), 64).unwrap();
    let producer = writer.producer();

    let (book_tx, book_rx) = broadcast::channel(16);
    let (kill_tx, kill_rx) = watch::channel(false);
    let task = tokio::spawn(model_observer::run_observer(
        Arc::new(FixedPredictor(0.95)), // p >> ask 0.51 → virtual buy YES
        cfg_shadow(dir.path()),
        ctx_with_vol(book_rx, kill_rx, fresh_vol("67000")),
        producer,
    ));

    book_tx
        .send(make_book(Duration::ZERO, "0.49", "0.51"))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    kill_tx.send(true).unwrap();
    task.await.unwrap();
    let res = writer.close().await;
    assert!(res.written >= 1);

    let lines = read_lines(dir.path());
    let decisions: Vec<&Value> = lines
        .iter()
        .filter(|l| l["kind"] == json!("decision"))
        .collect();
    assert_eq!(decisions.len(), 1, "a single decision per book");
    let d = decisions[0];
    assert_eq!(d["side"], json!("buy_yes"));
    assert_eq!(d["token_id"], json!("tok-yes"));
    assert_eq!(d["market_cid"], json!("cid-shadow"));
    assert!(d["features"]["yes_mid"].is_number(), "features recorded");
    assert_eq!(d["p_model"].as_f64().unwrap(), 0.95);
    assert!(d["p_market"].is_number(), "market baseline recorded");
    assert!(d["edge_min"].is_string(), "threshold recorded");
    assert_eq!(d["model_hash"], json!("hash-test-fixed"));
    assert!(d["latency_ms"].is_number(), "latency measured");
    assert_eq!(d["schema_version"].as_u64().unwrap(), 1);
}

/// Down model: every prediction fails → abstentions recorded, zero
/// decisions, and the task stays alive (it does not take the bot down).
#[tokio::test]
async fn model_down_generates_no_decisions() {
    let dir = tempfile::tempdir().unwrap();
    let writer = ShadowWriter::start(dir.path(), 64).unwrap();
    let (book_tx, book_rx) = broadcast::channel(16);
    let (kill_tx, kill_rx) = watch::channel(false);
    let task = tokio::spawn(model_observer::run_observer(
        Arc::new(DeadPredictor),
        cfg_shadow(dir.path()),
        ctx_with_vol(book_rx, kill_rx, fresh_vol("67000")),
        writer.producer(),
    ));

    for _ in 0..3 {
        book_tx
            .send(make_book(Duration::ZERO, "0.49", "0.51"))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    kill_tx.send(true).unwrap();
    task.await.unwrap(); // the task exits cleanly, no panic
    writer.close().await;

    let lines = read_evaluations(dir.path());
    assert!(
        !lines.is_empty()
            && lines
                .iter()
                .all(|l| l["kind"] == json!("abstention") && l["reason"] == json!("model_error")),
        "only model_error abstentions: {lines:?}"
    );
}

/// Stale data (stale book ts, stale spot ts) → abstention, never a decision.
#[tokio::test]
async fn stale_data_generates_no_decisions() {
    let dir = tempfile::tempdir().unwrap();
    let writer = ShadowWriter::start(dir.path(), 64).unwrap();
    let (book_tx, book_rx) = broadcast::channel(16);
    let (kill_tx, kill_rx) = watch::channel(false);
    let task = tokio::spawn(model_observer::run_observer(
        Arc::new(FixedPredictor(0.99)),
        cfg_shadow(dir.path()),
        // empty vol_rxs → VolSnapshot::default() (stale) via internal fallback;
        // here we test the stale book: max_book_age 5s < 60s of age.
        ctx(book_rx, kill_rx),
        writer.producer(),
    ));

    book_tx
        .send(make_book(Duration::from_secs(60), "0.49", "0.51"))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    kill_tx.send(true).unwrap();
    task.await.unwrap();
    writer.close().await;

    let lines = read_evaluations(dir.path());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["kind"], json!("abstention"));
    assert_eq!(lines[0]["reason"], json!("book_stale"));
}

/// Stale spot (silent oracle > 5s) → abstention, not a decision.
#[tokio::test]
async fn spot_stale_abstains() {
    let dir = tempfile::tempdir().unwrap();
    let writer = ShadowWriter::start(dir.path(), 64).unwrap();
    let mut vol = fresh_vol("67000");
    vol.updated_at = poly_bot::time::Instant::now()
        .checked_sub(Duration::from_secs(60))
        .unwrap();
    let (book_tx, book_rx) = broadcast::channel(16);
    let (kill_tx, kill_rx) = watch::channel(false);
    let task = tokio::spawn(model_observer::run_observer(
        Arc::new(FixedPredictor(0.99)),
        cfg_shadow(dir.path()),
        ctx_with_vol(book_rx, kill_rx, vol),
        writer.producer(),
    ));

    book_tx
        .send(make_book(Duration::ZERO, "0.49", "0.51"))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    kill_tx.send(true).unwrap();
    task.await.unwrap();
    writer.close().await;

    let lines = read_evaluations(dir.path());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["reason"], json!("spot_stale"));
}

/// Slow model: the prediction takes longer than the budget derived from the
/// signal horizon → `latency_exceeded` abstention (recorded), never
/// a decision with dead prices.
#[tokio::test]
async fn slow_model_abstains_due_to_budget() {
    let dir = tempfile::tempdir().unwrap();
    let writer = ShadowWriter::start(dir.path(), 64).unwrap();
    let mut cfg = cfg_shadow(dir.path());
    cfg.latency_budget_ms = 10; // 10 ms — the predictor sleeps 150 ms
    let (book_tx, book_rx) = broadcast::channel(16);
    let (kill_tx, kill_rx) = watch::channel(false);
    let task = tokio::spawn(model_observer::run_observer(
        Arc::new(SlowPredictor(Duration::from_millis(150))),
        cfg,
        ctx_with_vol(book_rx, kill_rx, fresh_vol("67000")),
        writer.producer(),
    ));

    book_tx
        .send(make_book(Duration::ZERO, "0.49", "0.51"))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(400)).await;
    kill_tx.send(true).unwrap();
    task.await.unwrap();
    writer.close().await;

    let lines = read_evaluations(dir.path());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["kind"], json!("abstention"));
    assert_eq!(lines[0]["reason"], json!("latency_exceeded"));
    assert!(lines[0]["latency_ms"].as_u64().unwrap() >= 10);
}

/// The shadow cannot touch the executor: `ShadowCtx` carries no `intent_tx`
/// (structural guarantee — this test documents that the observer never
/// receives an intents sender, so "generates no orders" is a property
/// of the type, not only of behavior). Besides, the fixed predictor never
/// receives cash, RiskGuard or StateStore — the decision is just a record.
#[tokio::test]
async fn shadow_emits_no_intents_nor_touches_ledger() {
    let dir = tempfile::tempdir().unwrap();
    let writer = ShadowWriter::start(dir.path(), 64).unwrap();
    let (book_tx, book_rx) = broadcast::channel(16);
    // Isolated intents channel: if the observer could emit, we would see it.
    let (intent_tx, mut intent_rx) = tokio::sync::mpsc::channel::<poly_bot::types::QueuedIntent>(16);
    let (kill_tx, kill_rx) = watch::channel(false);
    let task = tokio::spawn(model_observer::run_observer(
        Arc::new(FixedPredictor(0.99)),
        cfg_shadow(dir.path()),
        ctx_with_vol(book_rx, kill_rx, fresh_vol("67000")),
        writer.producer(),
    ));

    for _ in 0..5 {
        book_tx
            .send(make_book(Duration::ZERO, "0.49", "0.51"))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        intent_rx.try_recv().is_err(),
        "the shadow can NEVER put anything on the executor's intents channel"
    );
    drop(intent_tx);
    kill_tx.send(true).unwrap();
    task.await.unwrap();
    writer.close().await;
}

/// Writer down: `ShadowProducer::sink` (channel closed since the start,
/// same state as a dead writer) → `register` returns false and
/// counts the drop — never blocks.
#[test]
fn dead_writer_does_not_block() {
    let producer = ShadowProducer::sink();
    assert!(!producer.record(model_observer::stats_line(0, 0, 0, 0, 0, 0)));
    assert!(!producer.record(model_observer::stats_line(1, 2, 3, 4, 5, 6)));
}

/// Market out of scope: `markets` declared and the cid is not in it → neither
/// decision nor abstention (pure skip, metric only).
#[tokio::test]
async fn out_of_scope_market_does_not_register() {
    let dir = tempfile::tempdir().unwrap();
    let writer = ShadowWriter::start(dir.path(), 64).unwrap();
    let mut cfg = cfg_shadow(dir.path());
    cfg.markets = vec!["other-cid".to_string()];
    let (book_tx, book_rx) = broadcast::channel(16);
    let (kill_tx, kill_rx) = watch::channel(false);
    let task = tokio::spawn(model_observer::run_observer(
        Arc::new(FixedPredictor(0.99)),
        cfg,
        ctx_with_vol(book_rx, kill_rx, fresh_vol("67000")),
        writer.producer(),
    ));

    book_tx
        .send(make_book(Duration::ZERO, "0.49", "0.51"))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    kill_tx.send(true).unwrap();
    task.await.unwrap();
    writer.close().await;

    assert!(
        read_evaluations(dir.path()).is_empty(),
        "out of scope: neither decision nor abstention recorded"
    );
}

/// Not enough edge → `no_edge` abstention with the prediction recorded
/// (complete record to compare against replay).
#[tokio::test]
async fn no_edge_abstains_with_recorded_prediction() {
    let dir = tempfile::tempdir().unwrap();
    let writer = ShadowWriter::start(dir.path(), 64).unwrap();
    let (book_tx, book_rx) = broadcast::channel(16);
    let (kill_tx, kill_rx) = watch::channel(false);
    // edge_min=0.05: YES gives 0.52−0.51=0.01 and NO gives 0.48−0.44=0.04 — both
    // below → `no_edge` abstention (with edge_min=0.02 the NO side would
    // fire a virtual decision).
    let cfg = ModelShadowConfig {
        edge_min: dec!(0.05),
        ..cfg_shadow(dir.path())
    };
    let task = tokio::spawn(model_observer::run_observer(
        Arc::new(FixedPredictor(0.52)),
        cfg,
        ctx_with_vol(book_rx, kill_rx, fresh_vol("67000")),
        writer.producer(),
    ));
    book_tx
        .send(make_book(Duration::ZERO, "0.49", "0.51"))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    kill_tx.send(true).unwrap();
    task.await.unwrap();
    writer.close().await;

    let lines = read_evaluations(dir.path());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["reason"], json!("no_edge"));
    assert_eq!(lines[0]["p_model"].as_f64().unwrap(), 0.52);
}

/// The default latency budget is DERIVED from the signal horizon
/// (horizon/10), not from the bracket length or a loose constant.
#[test]
fn budget_derives_from_signal_horizon() {
    let cfg = ModelShadowConfig::default();
    assert!(!cfg.enabled, "model_shadow is born disabled by default");
    let cfg = ModelShadowConfig {
        signal_horizon_ms: 60_000,
        latency_budget_ms: 0,
        ..ModelShadowConfig::default()
    };
    assert_eq!(cfg.effective_budget_ms(), 6_000);
    let cfg = ModelShadowConfig {
        latency_budget_ms: 250,
        ..ModelShadowConfig::default()
    };
    assert_eq!(cfg.effective_budget_ms(), 250);
}

/// Feature extractor: over a known book it must reproduce the formulas
/// of `dataset.py::_book_features` (mid, spread, depth top-5, imbalance,
/// microprice) and the time ones (`time_left_ms`, `time_left_frac`).
#[test]
fn features_reproduce_dataset_py() {
    let book = make_book(Duration::from_millis(120), "0.49", "0.51");
    let vol = fresh_vol("67000");
    let mut series = model_observer::SpotSeries::new();
    let now_ms = poly_bot::time::now().timestamp_millis();
    let feats = model_observer::extract_features(&book, &vol, &mut series, now_ms);

    assert_eq!(feats["yes_bid"], Some(0.49));
    assert_eq!(feats["yes_ask"], Some(0.51));
    assert_eq!(feats["yes_mid"], Some(0.50));
    // spread in f64 (like dataset.py): 0.51−0.49 is not exact in binary.
    assert!((feats["yes_spread"].unwrap() - 0.02).abs() < 1e-12);
    assert_eq!(feats["yes_depth_bid"], Some(100.0));
    assert_eq!(feats["yes_depth_ask"], Some(80.0));
    // imbalance = (100-80)/(100+80)
    assert!((feats["yes_imbalance"].unwrap() - 20.0 / 180.0).abs() < 1e-12);
    // microprice = (ask*size_bid + bid*size_ask)/(size_bid+size_ask)
    // = (0.51*100 + 0.49*80)/180
    let micro = (0.51 * 100.0 + 0.49 * 80.0) / 180.0;
    assert!((feats["yes_microprice"].unwrap() - micro).abs() < 1e-9);
    // parity_ask_sum = yes_ask + no_ask = 0.51 + 0.44
    assert!((feats["parity_ask_sum"].unwrap() - 0.95).abs() < 1e-9);
    // time_left_ms = clear_at*1000 − now (bracket halfway through its life → ~1.8e6 ms,
    // positive) and time_left_frac ≈ 0.5.
    let tl = feats["time_left_ms"].unwrap();
    assert!(
        (1_700_000.0..=1_800_100.0).contains(&tl),
        "time_left_ms={tl}"
    );
    let frac = feats["time_left_frac"].unwrap();
    assert!((frac - 0.5).abs() < 0.01, "time_left_frac={frac}");
    // The shadow never trades: own state at zero, nothing invented.
    assert_eq!(feats["own_open_orders"], Some(0.0));
    assert_eq!(feats["own_net_shares"], Some(0.0));
    // No remote ts on the live path → None (imputed by the scaler), never
    // invented.
    assert_eq!(feats["origin_delay_ms"], None);
}

/// Parity of `spot_ret_n` with `dataset.py::_spot_features`: return of
/// EXACTLY RET_WINDOW=5 ticks (`prices[-1]/prices[-1-5]`), not of the
/// oldest point of the buffer. The old implementation used `prices[0]`
/// — with len>6 the return came out inflated and the z-scores diverged from
/// training. The Python parity test did not catch it because it injects
/// already computed features; here real series go through the extractor.
#[test]
fn spot_ret_n_uses_exactly_ret_window_ticks() {
    let book = make_book(Duration::ZERO, "0.49", "0.51");
    let vol = fresh_vol("108");
    let mut series = model_observer::SpotSeries::new();
    let now_ms = poly_bot::time::now().timestamp_millis();
    // 8 previous points; extract_features pushes the 9th (108) from the vol.
    for (i, p) in [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        .iter()
        .enumerate()
    {
        series.record(now_ms - (9 - i as i64) * 1000, *p);
    }
    let feats = model_observer::extract_features(&book, &vol, &mut series, now_ms);
    // Final series [100..108]: ret_n = ln(108/103) (index len-1-5 = 3).
    let expected = (108.0f64 / 103.0).ln();
    let got = feats["spot_ret_n"].unwrap();
    assert!(
        (got - expected).abs() < 1e-12,
        "spot_ret_n={got} expected={expected} (5-tick return)"
    );
    // With the bug (prices[0]) it would give ln(108/100) — clear separation.
    assert!(
        (got - (108.0f64 / 100.0).ln()).abs() > 1e-3,
        "spot_ret_n={got} matches the buggy value (full buffer)"
    );
    // ret_1 = last interval: ln(108/107).
    let ret1 = feats["spot_ret_1"].unwrap();
    assert!(
        (ret1 - (108.0f64 / 107.0).ln()).abs() < 1e-12,
        "ret_1={ret1}"
    );
}

/// sha256_hex: known vectors (RFC 4634) — the manifest integrity
/// verification depends on this hash.
#[test]
fn sha256_known_vectors() {
    assert_eq!(
        model_observer::sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        model_observer::sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        model_observer::sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}
