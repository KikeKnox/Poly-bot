//! `hrf_backtest` — deterministic offline replay of book_snapshots +
//! spot_samples recorded by a real strategy against `SimulatedRouter`.
//!
//! Example:
//!   hrf_backtest --strategy arb \
//!       --db-books data/market_analytics.sqlite --db-spot data/spot_stream.sqlite \
//!       --from-ms 1735000000000 --to-ms 1735070000000 --out report.json
//!
//! The DBs are opened in `ro` mode — run evidence is never touched.
//! Prints the JSON report to stdout if `--out` is not given.

use std::path::PathBuf;

use anyhow::{bail, Context};
use poly_bot::backtest::{run, BacktestParams};
use poly_bot::types::StrategyId;
use rust_decimal::Decimal;
use std::str::FromStr;

fn usage() -> ! {
    eprintln!(
        "usage: hrf_backtest --strategy <id> --db-books <path> [--db-spot <path>] \\
         [--from-ms N] [--to-ms N] [--seed N] [--balance USDC] [--config-dir DIR] [--out FILE] \\
         [--engine diagnostic|pipeline] [--latency-ms N] [--obs-delay-ms N] \\
         [--research-profile nominal|conservative|config] \\
         [--research-events <dir|file.jsonl> [--labels-db <path>] [--tape-window-secs N]] \\
         [--settlement-proxy-diagnostic]"
    );
    std::process::exit(2);
}

/// `current_thread`: the `--engine pipeline` engine requires a single-thread runtime
/// (quiescence by counters, see the doc of `replay.rs`); the diagnostic engine
/// is agnostic and runs the same on one thread.
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut p = BacktestParams::default();
    let mut out: Option<PathBuf> = None;
    let mut have_books = false;
    let mut engine = "diagnostic".to_string();
    let mut latency_ms: i64 = 0;
    let mut obs_delay_ms: i64 = 0;
    let mut research = poly_bot::replay::ResearchProfile::Nominal;
    // Causal ResearchEvent tape + official labels + settlement proxy
    // for DIAGNOSTICS only.
    let mut research_events: Option<PathBuf> = None;
    let mut labels_db: Option<PathBuf> = None;
    let mut tape_window_secs: u64 = 0;
    let mut settlement_proxy = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--strategy" => {
                let v = args.next().unwrap_or_else(|| usage());
                p.strategy = StrategyId::from_str(&v)
                    .map_err(|_| anyhow::anyhow!("unknown strategy: {v}"))?;
            }
            "--db-books" => {
                p.db_books = PathBuf::from(args.next().unwrap_or_else(|| usage()));
                have_books = true;
            }
            "--db-spot" => p.db_spot = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--from-ms" => p.from_ms = Some(args.next().unwrap_or_else(|| usage()).parse()?),
            "--to-ms" => p.to_ms = Some(args.next().unwrap_or_else(|| usage()).parse()?),
            "--seed" => p.seed = args.next().unwrap_or_else(|| usage()).parse()?,
            "--balance" => {
                p.initial_balance = Decimal::from_str(&args.next().unwrap_or_else(|| usage()))
                    .map_err(|e| anyhow::anyhow!("--balance is not a valid decimal: {e}"))?
            }
            "--config-dir" => p.config_dir = PathBuf::from(args.next().unwrap_or_else(|| usage())),
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            // Event engine of the real pipeline (Executor→RiskGuard→Sim→
            // fill_sink→SQLite :memory:), with deferred arrival at the venue and
            // delayed observation by the agent — all on a virtual clock.
            "--engine" => engine = args.next().unwrap_or_else(|| usage()),
            "--latency-ms" => latency_ms = args.next().unwrap_or_else(|| usage()).parse()?,
            "--obs-delay-ms" => obs_delay_ms = args.next().unwrap_or_else(|| usage()).parse()?,
            // Research profile of the sim — nominal (default) =
            // operating behavior; conservative = worst identifiable
            // scenario; config = [research_sim] section of the TOML.
            "--research-profile" => {
                let v = args.next().unwrap_or_else(|| usage());
                research = match v.as_str() {
                    "nominal" => poly_bot::replay::ResearchProfile::Nominal,
                    "conservative" => poly_bot::replay::ResearchProfile::Conservative,
                    "config" => poly_bot::replay::ResearchProfile::Config,
                    _ => bail!("unknown --research-profile: '{v}' (valid: nominal|conservative|config)"),
                };
            }
            // Causal tape from the recorder (mutually exclusive with the
            // SQL DBs — the replay's source of truth is a single one).
            "--research-events" => {
                research_events = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            // Official labels from a separate research DB
            // (read-only; positioned at the real `known_at_ms`).
            "--labels-db" => {
                labels_db = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            // Window of the aggregated flow.
            "--tape-window-secs" => {
                tape_window_secs = args.next().unwrap_or_else(|| usage()).parse()?
            }
            // Settlement proxy by last spot — explicit DIAGNOSTIC
            // only; forces `fidelity="reduced"` in the report.
            "--settlement-proxy-diagnostic" => settlement_proxy = true,
            _ => usage(),
        }
    }
    // Tape source exclusion rules: either historical SQL
    // (--db-books [+ --db-spot]) or the causal tape
    // (--research-events) — never both, and the SQL DBs do not accept
    // labels-db (SQL labels already live in the DB itself if they exist).
    if research_events.is_some() {
        anyhow::ensure!(
            !have_books && p.db_spot.is_none(),
            "--research-events is mutually exclusive with --db-books/--db-spot \
             (the causal tape carries its own books and ticks)"
        );
        anyhow::ensure!(
            engine == "pipeline",
            "--research-events requires --engine pipeline (the diagnostic engine \
             does not consume the causal tape)"
        );
    } else if !have_books {
        usage();
    }
    if labels_db.is_some() {
        anyhow::ensure!(
            research_events.is_some(),
            "--labels-db only applies with --research-events (SQL labels \
             live in the tape's own DB)"
        );
    }
    if p.db_spot.is_none() && research_events.is_none() {
        eprintln!(
            "warning: without --db-spot, strategies with an oracle will see an empty VolSnapshot \
             (only useful for ARB/PM, which do not need it)"
        );
    }

    if engine == "pipeline" {
        let report = if let Some(tape_path) = &research_events {
            // Causal tape: strict decode (a malformed line
            // aborts; a half-broken tape is never replayed in acceptance
            // mode), canonical order (avail_ts_ms, local_seq).
            let mut rt = poly_bot::research_replay::ResearchTape::load(
                tape_path,
                &poly_bot::research_replay::TapeOpts {
                    tape_window_secs,
                    strict: true,
                    max_lines: None,
                },
            )
            .with_context(|| format!("loading tape {}", tape_path.display()))?;
            for d in &rt.diagnostics {
                eprintln!(
                    "tape diagnostic [{}:{} seq={:?}]: {}",
                    d.origin, d.line, d.local_seq, d.message
                );
            }
            if let Some(db) = &labels_db {
                let res = poly_bot::research_replay::load_labels_db(db)
                    .with_context(|| format!("loading labels from {}", db.display()))?;
                eprintln!("labels-db: {} resolutions preloaded", res.len());
                rt.tape.resolutions.extend(res);
            }
            eprintln!(
                "causal tape: {} events → {} books, {} spots, {} resolutions, \
                 {} quality marks (malformed={}, agent={}, without_half={}, \
                 discarded_deltas={})",
                rt.events,
                rt.tape.books.len(),
                rt.tape.spots.len(),
                rt.tape.resolutions.len(),
                rt.tape.quality.len(),
                rt.malformed_lines,
                rt.agent_events,
                rt.books_missing_half,
                rt.discarded_deltas,
            );
            poly_bot::replay::run_pipeline_tape(
                &p,
                rt.tape,
                "research_events",
                latency_ms,
                obs_delay_ms,
                research,
                settlement_proxy,
            )
            .await?
        } else {
            poly_bot::replay::run_pipeline(&p, latency_ms, obs_delay_ms, research, settlement_proxy)
                .await?
        };
        let json = serde_json::to_string_pretty(&report)?;
        match out {
            Some(path) => {
                std::fs::write(&path, &json)
                    .with_context(|| format!("writing {}", path.display()))?;
                eprintln!("report at {}", path.display());
            }
            None => println!("{json}"),
        }
        return Ok(());
    }
    anyhow::ensure!(
        engine == "diagnostic",
        "unknown --engine: '{engine}' (valid: diagnostic|pipeline)"
    );

    let report = run(&p).await?;
    let json = serde_json::to_string_pretty(&report)?;
    match out {
        Some(path) => {
            std::fs::write(&path, &json)
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!("report at {}", path.display());
        }
        None => println!("{json}"),
    }
    if !report.accounting_consistent {
        bail!("BROKEN ACCOUNTING INVARIANT: cash != initial - buys + sells + payouts");
    }
    Ok(())
}
