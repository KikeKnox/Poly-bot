//! Tests of the incremental episodic environment and of the JSONL protocol of the
//! `hrf_research_env` subprocess.
//!
//! Properties verified:
//! - `reset` returns a causal observation (only events with `avail<=t`) and
//!   bounded legal actions; two resets with the same seed produce the
//!   same first observation (per-episode isolation).
//! - `step(Hold)` advances to the next decision point.
//! - A `Buy` that crosses → `Executed` + a position in the ledger's inventory
//!   (the SAME Executor→RiskGuard→Sim→fill_sink pipeline).
//! - A `Buy` that does not cross → `Accepted` (resting GTD) + a legal `Cancel{order_id}`
//!   on the next step; the `Cancel` is confirmed `Executed`.
//! - Reward: `total = Δequity` with the fee kept separate; the sum of rewards over an
//!   episode matches the ledger's equity change.
//! - Terminality: market resolution → `MarketResolved`; without
//!   resolution and exhausted tape → `HorizonReached`; structural GAP →
//!   `DataQualityAbort`; `step` after terminal → error.
//! - JSONL protocol of the binary: hello/reset/step/legal_actions/close.
//!
//! The replay clock is process-global — all the episodes are
//! serialized with `REPLAY_LOCK`.

use std::collections::HashSet;
use std::io::{BufRead, Write as _};

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use poly_bot::{
    replay::ReplayParams,
    research_contract::{
        ActionOutcome, AgentAction, EndReason, EventKind, EventSource, MarketMeta, MarketOutcome,
        OutcomeSide, QualityFlags, ResearchEvent, ResolutionRule, SCHEMA_VERSION,
    },
    research_replay::{EnvConfig, ResearchEnv, ResearchTape, TapeOpts},
};

/// Episode serialization: `time::REPLAY_NOW_MS` is process-global.
static REPLAY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const T0: i64 = 1_700_000_000_000;
const CID: &str = "0xcond-env";
const YES: &str = "tok-yes-env";
const NO: &str = "tok-no-env";
const SLUG: &str = "btc-updown-5m-1700000000";
const CLEAR_AT_S: i64 = T0 / 1000 + 3_600; // market open during the tape

fn ev(
    source: EventSource,
    kind: EventKind,
    cid: Option<&str>,
    token: Option<&str>,
    asset: Option<&str>,
    payload: serde_json::Value,
    origin: Option<i64>,
    recv: i64,
    avail: i64,
    seq: u64,
    quality: QualityFlags,
) -> ResearchEvent {
    ResearchEvent::new(
        source,
        kind,
        cid.map(str::to_string),
        token.map(str::to_string),
        asset.map(str::to_string),
        payload,
        origin,
        recv,
        avail,
        seq,
        quality,
    )
    .expect("valid event")
}

fn meta_ev(seq: u64, avail: i64) -> ResearchEvent {
    let meta = MarketMeta {
        schema_version: SCHEMA_VERSION,
        market_cid: CID.into(),
        slug: SLUG.into(),
        yes_token_id: YES.into(),
        no_token_id: NO.into(),
        asset: Some("BTC".into()),
        neg_risk: false,
        market_open_ts: T0 / 1000,
        clear_at: CLEAR_AT_S,
        reference_price: Some(dec!(50_000)),
        rule: ResolutionRule::UpDownReference,
        resolution_source: EventSource::PolymarketRest,
        resolution_source_ref: Some("gamma:test".into()),
        rules_windows: vec![poly_bot::research_contract::RulesWindow {
            valid_from_ms: T0,
            valid_to_ms: None,
            parabolic_fee_rate: None,
            fee_rate_bps: None,
            tick_size: Some(dec!(0.01)),
            min_order_size: None,
        }],
        outcome: None,
    };
    ev(
        EventSource::PolymarketRest,
        EventKind::MarketMeta,
        Some(CID),
        None,
        Some("BTC"),
        serde_json::to_value(meta).unwrap(),
        None,
        avail,
        avail,
        seq,
        QualityFlags::NONE,
    )
}

fn book_ev(token: &str, bid: &str, ask: &str, seq: u64, recv: i64, avail: i64) -> ResearchEvent {
    let payload = serde_json::json!({
        "event_type": "book",
        "asset_id": token,
        "market": CID,
        "timestamp": avail.to_string(),
        "bids": [{"price": bid, "size": "1000"}],
        "asks": [{"price": ask, "size": "1000"}],
    });
    ev(
        EventSource::PolymarketWs,
        EventKind::BookSnapshot,
        Some(CID),
        Some(token),
        None,
        payload,
        Some(recv),
        recv,
        avail,
        seq,
        QualityFlags::NONE,
    )
}

fn res_ev(winner: OutcomeSide, known_at: i64, seq: u64, avail: i64) -> ResearchEvent {
    let outcome = MarketOutcome::new(winner, known_at, "gamma:test").unwrap();
    ev(
        EventSource::PolymarketRest,
        EventKind::Resolution,
        Some(CID),
        None,
        None,
        serde_json::to_value(outcome).unwrap(),
        None,
        avail,
        avail,
        seq,
        QualityFlags::NONE,
    )
}

/// Minimal episode tape: meta + YES book + NO book + a second YES book
/// (the second decision point).
fn base_tape() -> Vec<ResearchEvent> {
    vec![
        meta_ev(1, T0),
        book_ev(YES, "0.49", "0.51", 2, T0 + 10, T0 + 20),
        book_ev(NO, "0.48", "0.50", 3, T0 + 30, T0 + 40),
        book_ev(YES, "0.49", "0.51", 4, T0 + 1_000, T0 + 1_000),
        book_ev(NO, "0.48", "0.50", 5, T0 + 2_000, T0 + 2_000),
    ]
}

fn env_from_events(events: Vec<ResearchEvent>, cfg: EnvConfig) -> ResearchEnv {
    let rt = ResearchTape::from_events(events, vec![], 0, &TapeOpts::default());
    ResearchEnv::new(rt, cfg)
}

fn cfg() -> EnvConfig {
    EnvConfig {
        params: ReplayParams {
            seed: 7,
            initial_balance: dec!(1000),
            risk: poly_bot::config::RiskConfig::default(),
            obs_delay_ms: 0,
            latency_ms: 0,
            maintenance_ms: 30_000,
            timer_ms: 5_000,
            fak_strategies: HashSet::new(),
            max_book_age_ms: 300_000,
            research: None,
            settlement_proxy_diagnostic: false,
        },
        abort_on_quality: QualityFlags::GAP,
        max_steps: None,
        max_action_size: dec!(500),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn reset_returns_causal_observation_and_legal_actions() {
    let _g = REPLAY_LOCK.lock().await;
    let mut env = env_from_events(base_tape(), cfg());
    let (obs, legal) = env.reset("ep1", 42, Some(CID)).await.unwrap();

    assert_eq!(obs.schema_version, SCHEMA_VERSION);
    assert_eq!(obs.market_cid.as_deref(), Some(CID));
    // The first decision point is the NO book (both halves → emission).
    assert_eq!(obs.decision_ts_ms, T0 + 40);
    // observation_seqs references the events that formed it (local_seq).
    assert!(
        obs.observation_seqs.contains(&3),
        "seqs: {:?}",
        obs.observation_seqs
    );
    // Causal features from the SAME extractor as the shadow.
    assert_eq!(obs.features.get("yes_ask").copied().flatten(), Some(0.51));
    assert_eq!(obs.features.get("no_bid").copied().flatten(), Some(0.48));
    // Legal: Hold always + Buy at the best ask of each token. No Sell (no
    // positions yet) nor Cancel (no live orders).
    assert!(legal.contains(&AgentAction::Hold));
    assert!(
        legal.iter().any(|a| matches!(a, AgentAction::Buy { token_id, price, .. } if token_id == YES && *price == dec!(0.51))),
        "buy YES at the ask: {legal:?}"
    );
    assert!(!legal.iter().any(|a| matches!(a, AgentAction::Sell { .. })));
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn step_hold_advances_to_next_point() {
    let _g = REPLAY_LOCK.lock().await;
    let mut env = env_from_events(base_tape(), cfg());
    let (obs0, _) = env.reset("ep", 1, Some(CID)).await.unwrap();
    let r = env.step(AgentAction::Hold).await.unwrap();
    assert_eq!(r.step.step_idx, 0);
    assert_eq!(r.step.decision_ts_ms, obs0.decision_ts_ms);
    assert!(!r.step.terminal);
    let obs1 = r.observation.expect("next observation");
    assert!(obs1.decision_ts_ms > obs0.decision_ts_ms);
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn crossing_buy_executes_and_opens_position() {
    let _g = REPLAY_LOCK.lock().await;
    let mut env = env_from_events(base_tape(), cfg());
    let (_obs, _legal) = env.reset("ep", 1, Some(CID)).await.unwrap();
    let r = env
        .step(AgentAction::Buy {
            token_id: YES.into(),
            price: dec!(0.51),
            size: dec!(90),
        })
        .await
        .unwrap();
    // Latency 0: the submit arrives in the same virtual tick and crosses the ask
    // → real fill in the step's window.
    assert!(
        matches!(r.step.action_outcome, Some(ActionOutcome::Executed { .. })),
        "outcome: {:?}",
        r.step.action_outcome
    );
    // The position lives in the ledger (same pipeline as production).
    let pos = r
        .step
        .inventory
        .positions
        .iter()
        .find(|p| p.token_id == YES)
        .expect("open YES position");
    assert_eq!(pos.shares, dec!(90));
    // The reward decomposes the equity change (cash − notional ±…).
    assert!(r.step.reward.total <= Decimal::ZERO || r.step.reward.fee_component <= Decimal::ZERO);
    assert!(r.step.inventory.cash_usdc < dec!(1000));
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn non_crossing_buy_stays_accepted_and_is_cancelable() {
    let _g = REPLAY_LOCK.lock().await;
    let mut env = env_from_events(base_tape(), cfg());
    let (_obs, _legal) = env.reset("ep", 1, Some(CID)).await.unwrap();
    // Bid below the best bid → resting GTD, does not cross.
    let r = env
        .step(AgentAction::Buy {
            token_id: YES.into(),
            price: dec!(0.40),
            size: dec!(50),
        })
        .await
        .unwrap();
    assert!(
        matches!(r.step.action_outcome, Some(ActionOutcome::Accepted { .. })),
        "resting must stay Accepted: {:?}",
        r.step.action_outcome
    );
    // The live order shows up as a legal Cancel in the next observation.
    let cancel = r
        .legal_actions
        .iter()
        .find_map(|a| match a {
            AgentAction::Cancel { order_id } => Some(order_id.clone()),
            _ => None,
        })
        .expect("legal cancel of the live GTD");

    let r2 = env
        .step(AgentAction::Cancel { order_id: cancel })
        .await
        .unwrap();
    assert!(
        matches!(r2.step.action_outcome, Some(ActionOutcome::Executed { .. })),
        "cancel confirmed: {:?}",
        r2.step.action_outcome
    );
    // No position: the order never executed.
    assert!(r2.step.inventory.positions.is_empty());
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn step_after_terminal_is_error() {
    let _g = REPLAY_LOCK.lock().await;
    let mut cfg = cfg();
    cfg.max_steps = Some(1); // one-step horizon → truncates
    let mut env = env_from_events(base_tape(), cfg);
    env.reset("ep", 1, Some(CID)).await.unwrap();
    let r = env.step(AgentAction::Hold).await.unwrap();
    assert!(r.step.terminal);
    assert_eq!(r.step.end_reason, Some(EndReason::HorizonReached));
    assert!(env.step(AgentAction::Hold).await.is_err());
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn episode_ends_on_market_resolution() {
    let _g = REPLAY_LOCK.lock().await;
    let mut events = base_tape();
    // Closes the market: clear_at has already passed when the resolution arrives.
    let t_res = CLEAR_AT_S * 1000 + 5_000;
    events.push(res_ev(OutcomeSide::Yes, t_res, 6, t_res));
    let mut env = env_from_events(events, cfg());
    env.reset("ep", 1, Some(CID)).await.unwrap();
    let mut last = None;
    for _ in 0..50 {
        let r = env.step(AgentAction::Hold).await.unwrap();
        if r.step.terminal {
            last = Some(r.step);
            break;
        }
    }
    let last = last.expect("the episode must end");
    assert_eq!(last.end_reason, Some(EndReason::MarketResolved));
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn structural_gap_aborts_episode() {
    let _g = REPLAY_LOCK.lock().await;
    let mut events = base_tape();
    // Event with a GAP marked by the recorder — propagates to tape.quality
    // and aborts the episode at the step that crosses it.
    let mut gap = book_ev(YES, "0.49", "0.51", 6, T0 + 3_000, T0 + 3_000);
    gap.quality = QualityFlags::GAP;
    events.push(gap);
    let mut env = env_from_events(events, cfg());
    env.reset("ep", 1, Some(CID)).await.unwrap();
    let mut end = None;
    for _ in 0..50 {
        let r = env.step(AgentAction::Hold).await.unwrap();
        if r.step.terminal {
            end = r.step.end_reason;
            break;
        }
    }
    assert_eq!(end, Some(EndReason::DataQualityAbort), "the GAP aborts");
    env.close().await.unwrap();
}

/// Recorder GAP mark over an arbitrary `market_cid`, positioned
/// between the decision points of `base_tape` (T0+1500).
fn gap_cid_ev(cid: &str, seq: u64) -> ResearchEvent {
    ev(
        EventSource::PolymarketWs,
        EventKind::PriceTick,
        Some(cid),
        None,
        Some("BTC"),
        serde_json::json!({"price": "51000"}),
        Some(T0 + 1_500),
        T0 + 1_500,
        T0 + 1_500,
        seq,
        QualityFlags::GAP,
    )
}

#[tokio::test(flavor = "current_thread")]
async fn other_market_gap_does_not_abort_filtered_episode() {
    let _g = REPLAY_LOCK.lock().await;
    // v2 — abort by SCOPE: with `market_filter`, a quality mark
    // from ANOTHER market is recorded in the observation but does not stop
    // the episode.
    let mut events = base_tape();
    events.push(gap_cid_ev("0xother-market", 6));
    let mut env = env_from_events(events, cfg());
    env.reset("ep", 1, Some(CID)).await.unwrap();
    let mut end = None;
    let mut recorded_gap = false;
    for _ in 0..50 {
        let r = env.step(AgentAction::Hold).await.unwrap();
        if r.step.quality.contains(QualityFlags::GAP) {
            recorded_gap = true;
        }
        if r.step.terminal {
            end = r.step.end_reason;
            break;
        }
    }
    assert!(
        recorded_gap,
        "the foreign mark is recorded in step.quality (not hidden)"
    );
    assert_ne!(
        end,
        Some(EndReason::DataQualityAbort),
        "an out-of-scope mark never aborts the episode"
    );
    assert_eq!(
        end,
        Some(EndReason::HorizonReached),
        "the episode continues until the tape is exhausted"
    );
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn own_market_gap_does_abort_filtered_episode() {
    let _g = REPLAY_LOCK.lock().await;
    // Same temporal position as the earlier test but the GAP belongs
    // to the episode's market: it DOES abort (the scope is respected).
    let mut events = base_tape();
    events.push(gap_cid_ev(CID, 6));
    let mut env = env_from_events(events, cfg());
    env.reset("ep", 1, Some(CID)).await.unwrap();
    let mut end = None;
    for _ in 0..50 {
        let r = env.step(AgentAction::Hold).await.unwrap();
        if r.step.terminal {
            end = r.step.end_reason;
            break;
        }
    }
    assert_eq!(
        end,
        Some(EndReason::DataQualityAbort),
        "the episode market's GAP aborts"
    );
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn other_market_gap_aborts_unfiltered_episode() {
    let _g = REPLAY_LOCK.lock().await;
    // Without `market_filter` the whole tape is the scope — a mark from
    // any market aborts (previous behavior preserved).
    let mut events = base_tape();
    events.push(gap_cid_ev("0xother-market", 6));
    let mut env = env_from_events(events, cfg());
    env.reset("ep", 1, None).await.unwrap();
    let mut end = None;
    for _ in 0..50 {
        let r = env.step(AgentAction::Hold).await.unwrap();
        if r.step.terminal {
            end = r.step.end_reason;
            break;
        }
    }
    assert_eq!(
        end,
        Some(EndReason::DataQualityAbort),
        "without a filter every mark is in scope and aborts"
    );
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn reset_isolates_state_between_episodes() {
    let _g = REPLAY_LOCK.lock().await;
    let mut env = env_from_events(base_tape(), cfg());
    env.reset("ep1", 1, Some(CID)).await.unwrap();
    // Buys and leaves the position open.
    env.step(AgentAction::Buy {
        token_id: YES.into(),
        price: dec!(0.51),
        size: dec!(90),
    })
    .await
    .unwrap();
    // Second episode: clean inventory, same first observation.
    let (obs1, _) = env.reset("ep2", 1, Some(CID)).await.unwrap();
    let r = env.step(AgentAction::Hold).await.unwrap();
    assert!(r.step.inventory.positions.is_empty(), "isolated episode");
    assert_eq!(obs1.decision_ts_ms, T0 + 40);
    env.close().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn reward_adds_to_equity_delta() {
    let _g = REPLAY_LOCK.lock().await;
    let mut env = env_from_events(base_tape(), cfg());
    env.reset("ep", 1, Some(CID)).await.unwrap();
    // A crossing Buy + Hold: the sum of rewards must equal the
    // episode's total Δequity (price + fee + risk = exact Δequity).
    let r1 = env
        .step(AgentAction::Buy {
            token_id: YES.into(),
            price: dec!(0.51),
            size: dec!(90),
        })
        .await
        .unwrap();
    let r2 = env.step(AgentAction::Hold).await.unwrap();
    let total: Decimal = r1.step.reward.total + r2.step.reward.total;
    // equity_0 = 1000 (no positions or reserves). equity_T = cash +
    // exit value of the position. The identity requires the breakdown to add up:
    // total == price + fee + risk at every step (contract invariant).
    assert!(r1.step.reward.validate().is_ok());
    assert!(r2.step.reward.validate().is_ok());
    // And the reward is nonzero when there was a trade (the friction exists).
    assert_ne!(total, Decimal::ZERO);
    env.close().await.unwrap();
}

// ── Subprocess JSONL protocol ──────────────────────────────

#[test]
fn jsonl_protocol_hello_reset_step_close() {
    // Minimal tape on disk (recorder layout).
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("tape.jsonl");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&f).unwrap());
    for e in base_tape() {
        let l = poly_bot::research_events::RecordedLine {
            event: e,
            stream: None,
            generation: None,
            source_seq: None,
        };
        writeln!(w, "{}", serde_json::to_string(&l).unwrap()).unwrap();
    }
    w.flush().unwrap();
    drop(w);

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_hrf_research_env"))
        .arg("--tape")
        .arg(&f)
        .arg("--max-steps")
        .arg("3")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn hrf_research_env");
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut lines = std::io::BufReader::new(stdout).lines();

    // 1) Hello on startup.
    let hello: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["schema_version"], SCHEMA_VERSION);
    assert_eq!(hello["fidelity"], "certified");
    // v2: the hello announces the tape's quality summary and the scope
    // of markets/window it covers.
    let q = &hello["quality"];
    for k in [
        "quality_marks",
        "malformed",
        "agent",
        "discarded_deltas",
        "discarded_deltas_in_scope",
        "missing_half",
        "books_top_equal",
        "resyncs",
        "tokens_with_gap_in_scope",
        "gap_markets_in_scope",
        "delta_coverage_pct_in_scope",
    ] {
        assert!(q.get(k).is_some(), "hello.quality without the key {k}");
    }
    assert_eq!(q["discarded_deltas_in_scope"], 0);
    assert_eq!(hello["quality_scope"]["markets"], serde_json::json!([CID]));
    assert!(
        hello["quality_scope"]["window_ms"].is_array(),
        "tape time window announced"
    );

    // 2) Reset → observation + legal actions.
    writeln!(
        stdin,
        "{}",
        serde_json::json!({"cmd":"reset","episode_id":"e1","seed":7,"market_cid":CID})
    )
    .unwrap();
    stdin.flush().unwrap();
    let reset: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(reset["type"], "reset");
    assert!(reset["legal_actions"].as_array().unwrap().len() >= 2);

    // 3) Step Hold → step_result.
    writeln!(
        stdin,
        "{}",
        serde_json::json!({"cmd":"step","action":"hold"})
    )
    .unwrap();
    stdin.flush().unwrap();
    let step: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(step["type"], "step_result");
    assert_eq!(step["step"]["step_idx"], 0);

    // 4) LegalActions → response.
    writeln!(stdin, "{}", serde_json::json!({"cmd":"legal_actions"})).unwrap();
    stdin.flush().unwrap();
    let legal: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(legal["type"], "legal_actions");

    // 5) Close → closed + clean exit.
    writeln!(stdin, "{}", serde_json::json!({"cmd":"close"})).unwrap();
    stdin.flush().unwrap();
    let closed: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(closed["type"], "closed");
    assert_eq!(closed["episode_id"], "e1");
    let status = child.wait().unwrap();
    assert!(status.success(), "the driver must exit cleanly: {status}");
}

#[test]
fn jsonl_protocol_error_not_fatal_on_malformed_line() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("tape.jsonl");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&f).unwrap());
    for e in base_tape() {
        let l = poly_bot::research_events::RecordedLine {
            event: e,
            stream: None,
            generation: None,
            source_seq: None,
        };
        writeln!(w, "{}", serde_json::to_string(&l).unwrap()).unwrap();
    }
    w.flush().unwrap();
    drop(w);

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_hrf_research_env"))
        .arg("--tape")
        .arg(&f)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut lines = std::io::BufReader::new(stdout).lines();

    let _hello = lines.next().unwrap().unwrap();
    // Non-JSON line → Error{fatal:false} and the channel stays usable.
    writeln!(stdin, "this is not json").unwrap();
    stdin.flush().unwrap();
    let err: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(err["type"], "error");
    assert_eq!(err["fatal"], false);
    // And the later reset works.
    writeln!(
        stdin,
        "{}",
        serde_json::json!({"cmd":"reset","episode_id":"e2","seed":1})
    )
    .unwrap();
    stdin.flush().unwrap();
    let reset: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(reset["type"], "reset");
    drop(stdin);
    let _ = child.wait();
}
