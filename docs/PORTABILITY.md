# Moving the project to another Linux machine

Copy or clone the whole repository folder. There is no need to run `clean.sh` first. The numeric model used by the `model_tool` lives in `src/tools/model/artifacts/btc5m` and its configured paths are relative.

On the destination, from the repository root:

    bash launch.sh --check
    bash launch.sh

Requirements: Linux, Bash, a Rust/Cargo version compatible with `Cargo.lock`, a C compiler, `cmake`, `pkg-config`, CA certificates and Internet access. The first build can take a while. The web UI is not committed (`src/api/static/dist/` is git-ignored), so Node.js and `pnpm` are needed once to build it; `launch.sh` does that automatically when the build is missing or `web/` changed. Only Linux has been tested; Windows and macOS are not supported.

Launching without flags forces paper mode even if the environment or `.env` declares live, and requires no secrets. Rust still loads the other optional overrides from `.env`, so any custom override must be valid on the destination. `--live` is an explicit request and does not remove any of Rust's live barriers (see [CONFIGURATION.md](CONFIGURATION.md)); paper-only strategies stay paper-only.

Paper mode has no time limit. By default it keeps `paper_disable_kill_switch = true`, so simulated losses do not switch strategies off automatically. Ctrl-C stops the bot in an orderly way.

## Cleaning

`clean.sh` with no options removes the local `target/`, Python caches (`__pycache__`, `*.pyc`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`; not `.venv`) and operational logs, asking before each group. `--build`, `--cache` and `--logs` clean them separately. `--data` deletes local SQLite files, including paper and live history. `--all` adds data and calibration folders. It never deletes configuration or the model, does not follow external links, and refuses to run while a bot is running in the folder. Run `./clean.sh --help` for the full list.

## Verification checks

The repository has these checks for the scripts and the build:

- `cargo check --all-targets --locked --offline`
- `cargo test --lib --locked --offline`
- `python3 tests/test_portable_scripts.py` (launch and clean scripts, isolated fixtures)
- `bash -n launch.sh` and `./launch.sh --check`

These checks reuse the toolchain of the same computer; they do not prove that a machine lacking the requirements can start the bot.
