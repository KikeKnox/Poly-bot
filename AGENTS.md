# AGENTS.md

Technical guide for AI coding agents working on this repository. Humans: see [README.md](README.md).

poly-bot is a Rust/Tokio bot for Polymarket up/down markets. Paper mode is the default; live mode is selected explicitly. The project is closed: no further development is planned. Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Configuration: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Commands

```bash
cargo build --release --locked                 # binary: target/release/poly-bot
cargo test                                     # unit tests in src/ + integration tests in tests/
cargo test <name>                              # one test by substring
cargo test --test pipeline                     # one integration file (tests/pipeline.rs)
cargo check --locked --offline --all-targets   # the check that catches broken tests after a signature change
./launch.sh --check                            # verify requirements without building
python3 -m unittest discover -s tools/research_tools/tests   # Python tools (also research/numeric_model/tests)
```

Do not start long-running processes (the bot, recorders) unless the user asks.

## Conventions

- Money is `rust_decimal::Decimal`, never floats. SQLite stores integer micro-USDC.
- `src/order/executor.rs` is the only path that submits orders. Strategies emit `OrderIntent`; never call `clob_sdk` from a strategy.
- Tools (`src/tools/`) and research code observe only and must never place orders.
- `src/fair_value.rs` holds pure functions with no I/O so the offline backtest can reuse them.
- Live trading must stay fail-closed: do not weaken `paper_only`, the daily-loss-stop check or the mode checks in `src/config.rs`.
- Every recorder that stores per-tick snapshots must deduplicate by content hash from the first commit.
- Tests build their own wiring (dummy channels, in-memory SQLite) and need no network or credentials; keep it that way.
- Follow the neighboring code's naming, error handling and comment density. Comments explain why, not what.
- Code, comments, logs and docs are in English. Do not commit `.env`, databases or recorded data.
- Keep changes scoped: report adjacent problems instead of fixing them in the same diff.
