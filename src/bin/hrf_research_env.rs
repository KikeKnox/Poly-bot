//! `hrf_research_env` — episodic driver of the research environment
//! (numeric environment, contract F of
//! `research_contract`).
//!
//! One process = one environment; JSONL protocol over stdin/stdout (one line
//! = one message):
//!
//! ```text
//! Python ──► {"cmd":"reset","episode_id":"e1","seed":7,"market_cid":null}
//! Python ◄── {"type":"reset","observation":{…},"legal_actions":[…]}
//! Python ──► {"cmd":"step","action":{"buy":{…}}}
//! Python ◄── {"type":"step_result","step":{…},"observation":{…},…}
//! Python ──► {"cmd":"legal_actions"}   ◄── {"type":"legal_actions",…}
//! Python ──► {"cmd":"close"}           ◄── {"type":"closed",…}  (and exits)
//! ```
//!
//! The Rust process is the ONLY owner of the economic state (Executor →
//! RiskGuard → SimulatedRouter → fill_sink → SQLite `:memory:`, same
//! pipeline as `replay.rs`); the client proposes `AgentAction` and reads
//! `EnvObservation`/`StepRecord` — it never keeps a second ledger.
//!
//! Replay clock isolation (`time::REPLAY_NOW_MS` is process-global):
//! ONE `hrf_research_env` process per environment — never two
//! `ReplayEngine`s in the same process.
//!
//! Usage:
//!   hrf_research_env --tape <dir|file.jsonl> [--labels-db <path>]
//!       [--tape-window-secs N] [--max-steps N] [--balance USDC]
//!       [--latency-ms N] [--obs-delay-ms N] [--strict]
//!       [--settlement-proxy-diagnostic]

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{bail, Context};
use poly_bot::research_contract::{EnvRequest, EnvResponse, SCHEMA_VERSION};
use poly_bot::research_replay::{load_labels_db, EnvConfig, ResearchEnv, ResearchTape, TapeOpts};
use rust_decimal::Decimal;
use std::str::FromStr;

fn usage() -> ! {
    eprintln!(
        "usage: hrf_research_env --tape <dir|file.jsonl> [--labels-db <path>] \\
         [--tape-window-secs N] [--max-steps N] [--balance USDC] \\
         [--latency-ms N] [--obs-delay-ms N] [--strict] \\
         [--settlement-proxy-diagnostic]"
    );
    std::process::exit(2);
}

/// `current_thread`: the `ReplayEngine` requires a single-thread runtime
/// (quiescence by counters — see the doc of `replay.rs`).
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut tape_path: Option<PathBuf> = None;
    let mut labels_db: Option<PathBuf> = None;
    let mut opts = TapeOpts::default();
    let mut cfg = EnvConfig::default();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tape" => tape_path = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--labels-db" => {
                labels_db = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--tape-window-secs" => {
                opts.tape_window_secs = args.next().unwrap_or_else(|| usage()).parse()?
            }
            "--max-steps" => cfg.max_steps = Some(args.next().unwrap_or_else(|| usage()).parse()?),
            "--balance" => {
                cfg.params.initial_balance =
                    Decimal::from_str(&args.next().unwrap_or_else(|| usage()))?
            }
            "--latency-ms" => {
                cfg.params.latency_ms = args.next().unwrap_or_else(|| usage()).parse()?
            }
            "--obs-delay-ms" => {
                cfg.params.obs_delay_ms = args.next().unwrap_or_else(|| usage()).parse()?
            }
            "--strict" => opts.strict = true,
            "--settlement-proxy-diagnostic" => cfg.params.settlement_proxy_diagnostic = true,
            "--help" | "-h" => usage(),
            _ => usage(),
        }
    }
    let tape_path = tape_path.unwrap_or_else(|| usage());

    // Load the causal tape + external official labels (if requested —
    // positioned at `known_at_ms`, never backdated).
    let mut rt = ResearchTape::load(&tape_path, &opts)
        .with_context(|| format!("loading tape {}", tape_path.display()))?;
    if let Some(db) = &labels_db {
        let res =
            load_labels_db(db).with_context(|| format!("loading labels from {}", db.display()))?;
        rt.tape.resolutions.extend(res);
        // Reorder by availability after the merge (the tape's resolutions
        // were seeded in order; those from labels-db enter by their
        // `avail_ms` — `seed_tape` sorts by ts, so extending is enough).
    }
    let env = ResearchEnv::new(rt, cfg);

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Startup hello — contract version + declared fidelity +
    // (v2) quality summary and scope of the loaded tape.
    emit(
        &mut out,
        &EnvResponse::Hello {
            schema_version: SCHEMA_VERSION,
            fidelity: env.fidelity().to_string(),
            quality: env.quality_summary(),
            quality_scope: env.quality_scope(),
        },
    )?;

    run_loop(env, stdin.lock(), &mut out).await
}

/// Serializes a JSONL response (one line) and flushes it — the client
/// blocks reading one line per response.
fn emit(out: &mut impl Write, resp: &EnvResponse) -> anyhow::Result<()> {
    let line = serde_json::to_string(resp).context("serializing EnvResponse")?;
    writeln!(out, "{line}")?;
    out.flush()?;
    Ok(())
}

/// Protocol loop: one stdin line = one `EnvRequest`. Parse
/// errors → `Error{fatal:false}` (connection stays usable); engine errors →
/// `Error{fatal:true}` and exit with code 1 (the episode is
/// invalidated — success is never answered after a ledger failure).
async fn run_loop(
    mut env: ResearchEnv,
    stdin: impl BufRead,
    out: &mut impl Write,
) -> anyhow::Result<()> {
    for line in stdin.lines() {
        let line = line.context("reading stdin")?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let req: EnvRequest = match serde_json::from_str(t) {
            Ok(r) => r,
            Err(e) => {
                emit(
                    out,
                    &EnvResponse::Error {
                        message: format!("malformed JSON: {e}"),
                        fatal: false,
                    },
                )?;
                continue;
            }
        };
        let res = match req {
            EnvRequest::Reset {
                episode_id,
                seed,
                market_cid,
            } => env
                .reset(&episode_id, seed, market_cid.as_deref())
                .await
                .map(|(observation, legal_actions)| EnvResponse::Reset {
                    observation,
                    legal_actions,
                }),
            EnvRequest::Step { action } => {
                env.step(action).await.map(|r| EnvResponse::StepResult {
                    step: r.step,
                    observation: r.observation,
                    legal_actions: r.legal_actions,
                })
            }
            EnvRequest::LegalActions => {
                let legal_actions = env.legal_actions().await;
                Ok(EnvResponse::LegalActions { legal_actions })
            }
            EnvRequest::Close => {
                // Episode close: economic digest + ledger
                // reconciliation. If there is no open episode, `close` fails —
                // a fatal Error is returned (a Closed is not fabricated).
                let episode_id = env.episode_id().unwrap_or("").to_string();
                match env.close().await {
                    Ok(o) => {
                        emit(
                            out,
                            &EnvResponse::Closed {
                                episode_id,
                                digest: o.digest,
                                reconciled: o.reconciliation.reconciled(),
                            },
                        )?;
                        return Ok(());
                    }
                    Err(e) => Err(e),
                }
            }
        };
        match res {
            Ok(resp) => emit(out, &resp)?,
            Err(e) => {
                emit(
                    out,
                    &EnvResponse::Error {
                        message: format!("{e:#}"),
                        fatal: true,
                    },
                )?;
                bail!("hrf_research_env: fatal engine error: {e:#}");
            }
        }
    }
    // EOF on stdin: the client closed without `close` — the open episode is
    // discarded (nothing to report; the process dies with its state).
    Ok(())
}
