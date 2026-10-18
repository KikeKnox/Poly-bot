#!/usr/bin/env bash
# Portable launcher for Linux. .env is loaded by Rust; it is never executed as shell.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
check=false
mode=paper
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            printf '%s\n' 'Usage: ./launch.sh [--check] [--live]' \
                'No flags: continuous paper mode. Requires neither .env nor keys.' \
                '--check: checks requirements without building or starting.' \
                '--live: explicit live request; Rust validates credentials and permissions.' \
                'Requires Linux, Bash, Rust/Cargo, cc, cmake and pkg-config.' \
                'Frontend: pnpm/Node if no built UI exists. Ctrl-C to stop.'
            exit 0 ;;
        --check) check=true ;;
        --live) mode=live ;;
        *) die "Unknown argument: $arg" ;;
    esac
done
[[ $(uname -s) == Linux ]] || die 'This launcher is verified on Linux only.'
[[ -f Cargo.toml && -f Cargo.lock && -f config/general.toml ]] || die 'Incomplete project folder.'
for tool in cargo rustc cc cmake pkg-config; do
    command -v "$tool" >/dev/null || die "Missing $tool; install the project requirements."
done
[[ -f src/tools/model/artifacts/btc5m/model.json && -f src/tools/model/artifacts/btc5m/manifest.json ]] || die 'Missing the BTC5m model.'
# Copying a live .env never changes the default startup mode.
export HRF_MODE="$mode"
export CARGO_TARGET_DIR="$PWD/target"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
DIST=src/api/static/dist/index.html
if [[ ! -f "$DIST" ]]; then
    command -v pnpm >/dev/null || die 'Missing built UI and pnpm. Install Node/pnpm to build it.'
    command -v node >/dev/null || die 'Missing Node.js to build the UI.'
fi
printf '[poly-bot] Requested mode: %s; configuration: config/\n' "$mode"
if $check; then
    printf '%s\n' '[poly-bot] Requirements present. This check does not replace Rust configuration validation.'
    exit 0
fi
# Cargo validates sources, dependencies and platform, even after copying target/.
cargo build --release --locked --bin poly-bot
need_ui=false
[[ -f "$DIST" ]] || need_ui=true
if [[ -f "$DIST" && -d web ]]; then
    newer=$(find web -path web/node_modules -prune -o -path web/dist -prune -o -type f -newer "$DIST" -print -quit)
    [[ -z "$newer" ]] || need_ui=true
fi
if $need_ui; then
    if command -v pnpm >/dev/null && command -v node >/dev/null; then
        if ! (pnpm --dir web install --frozen-lockfile && pnpm --dir web build); then
            [[ -f "$DIST" ]] || die 'Frontend build failed and there is no previous UI.'
            printf '%s\n' '[WARNING] Frontend build failed: serving the previous built UI.'
        fi
    elif [[ -f "$DIST" ]]; then
        printf '%s\n' '[WARNING] No pnpm/Node: serving the previous built UI.'
    else
        die 'Cannot build the frontend.'
    fi
fi
printf '%s\n' '[poly-bot] UI at /ui/ and metrics at /metrics, same API port (default 8766).' \
    '[poly-bot] No time limit. Ctrl-C for an orderly shutdown.'
export RUST_BACKTRACE=full
exec ./target/release/poly-bot