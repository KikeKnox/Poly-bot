#!/usr/bin/env bash
# Local cleanup + external DBs declared in .env. Does not follow links or delete configs/models.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
build=false data=false cal=false cache=false logs=false yes=false
for arg in "$@"; do
    case "$arg" in
        --build) build=true ;;
        --data) data=true ;;
        --db) build=true; data=true ;;
        --cal) cal=true ;;
        --cache) cache=true ;;
        --logs) logs=true ;;
        --all) build=true; data=true; cal=true; cache=true; logs=true ;;
        --yes|-y) yes=true ;;
        --pack) die 'Packaging disabled: copy the folder directly.' ;;
        --help|-h)
            printf '%s\n' 'Usage: ./clean.sh [--build|--data|--db|--cal|--cache|--logs|--all] [--yes]' \
                'No options: cleans target/, Python caches and local logs.' \
                '--data: deletes local SQLite files in the root, data/, db/ and runtime/, plus the external DBs' \
                'from HRF_DB_PATH (state/capital) and, with a separate confirmation,' \
                'HRF_ANALYTICS__DB_PATH / HRF_SPOT_STREAM__DB_PATH (environment or .env).' \
                '--cal: deletes local calibration_* folders.' \
                '--cache: deletes __pycache__, *.pyc and pytest/mypy/ruff caches (not .venv).' \
                '--logs: deletes *.log and rotated (*.log.N[.gz]) files in the root and runtime/.' \
                '--all: everything above. Does not delete config, the loaded model, secrets or' \
                'evidence under paper_rl/ or research/.'
            exit 0 ;;
        *) die "Unknown argument: $arg" ;;
    esac
done
if ! $build && ! $data && ! $cal && ! $cache && ! $logs; then build=true; cache=true; logs=true; fi
[[ $(uname -s) == Linux && -d /proc ]] || die 'The process guard requires Linux /proc.'
# Checked before ANY deletion, without relying on pgrep or stale PID files.
for proc in /proc/[0-9]*; do
    [[ -r "$proc/comm" ]] || continue
    IFS= read -r name < "$proc/comm" || continue
    [[ "$name" == poly-bot ]] || continue
    cwd=$(readlink "$proc/cwd" 2>/dev/null || true)
    [[ -n "$cwd" ]] || die 'Cannot verify the directory of a running bot.'
    [[ "$cwd" != "$ROOT" ]] || die 'A bot is running in this folder; stop it before cleaning.'
done
confirm() {
    if $yes; then return 0; fi
    local answer
    read -r -p "$1 [y/N] " answer || return 1
    [[ "$answer" == y || "$answer" == Y ]]
}
if $build && [[ -e target || -L target ]]; then
    [[ ! -L target ]] || die 'target/ is a link: an external destination will not be deleted.'
    if confirm 'Delete local target/?'; then rm -rf -- "$ROOT/target"; fi
fi
# Value of an HRF_* key: the environment first (dotenvy does not override exported vars),
# then the last assignment in .env. It is read as text, never executed.
env_value() {
    local key=$1 v=${!1:-}
    if [[ -z "$v" && -f .env ]]; then
        v=$(sed -n "s/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}$key=//p" .env | tail -n 1)
        v=${v%%[[:space:]]#*}; v=${v%\"}; v=${v#\"}; v=${v%\'}; v=${v#\'}
    fi
    printf '%s' "$v"
}
# Adds to $1 (array name) the DB of key $2 with the _paper/_live suffixes
# that StateStore::resolve_path adds, plus its -wal/-shm. No links.
add_env_db() {
    local -n out=$1
    local p base dir stem f
    p=$(env_value "$2")
    [[ -n "$p" ]] || return 0
    [[ "$p" == /* ]] || p="$ROOT/$p"
    dir=$(dirname -- "$p"); base=$(basename -- "$p"); stem=${base%.*}
    stem=${stem%_paper}; stem=${stem%_live}
    for base in "$p" "$dir/${stem}_paper.sqlite" "$dir/${stem}_live.sqlite"; do
        for f in "$base" "$base-wal" "$base-shm" "$base-journal"; do
            if [[ -f "$f" && ! -L "$f" && -z "${seen[$f]:-}" ]]; then seen[$f]=1; out+=("$f"); fi
        done
    done
}
if $data; then
    declare -A seen=()
    files=()
    while IFS= read -r -d '' f; do seen[$ROOT/${f#./}]=1; files+=("$f"); done < <(
        find . -maxdepth 1 -type f -name '*.sqlite*' -print0
        for dir in data db runtime; do
            if [[ -d "$dir" && ! -L "$dir" ]]; then
                find "$dir" -type f -name '*.sqlite*' -print0
            fi
        done
    )
    # The state DB (positions, fills, paper capital/deposits_total) may live
    # outside the repo via HRF_DB_PATH in .env; without this the bot does NOT start from zero.
    add_env_db files HRF_DB_PATH
    if (( ${#files[@]} )); then
        printf 'SQLite files to delete: %s\n' "${files[@]}"
        if confirm 'Delete these SQLite files (including paper/live history)?'; then rm -f -- "${files[@]}"; fi
    else
        printf '%s\n' 'There are no state SQLite files to delete.'
    fi
    # External recorders (market data, not money): separate confirmation.
    recorders=()
    add_env_db recorders HRF_ANALYTICS__DB_PATH
    add_env_db recorders HRF_SPOT_STREAM__DB_PATH
    if (( ${#recorders[@]} )); then
        printf 'External recorder SQLite files: %s\n' "${recorders[@]}"
        if confirm 'Also delete these market-data DBs?'; then rm -f -- "${recorders[@]}"; fi
    fi
fi
if $cal; then
    for dir in calibration_*; do
        [[ -d "$dir" && ! -L "$dir" ]] || continue
        if confirm "Delete $dir?"; then rm -rf -- "$ROOT/$dir"; fi
    done
fi
if $cache; then
    caches=()
    while IFS= read -r -d '' f; do caches+=("$f"); done < <(
        find . -type d \( -name .git -o -name .worktrees -o -name target -o -name node_modules -o -name .venv -o -name venv \) -prune -o \
            -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -print0 -o \
            -type f \( -name '*.pyc' -o -name '*.pyo' \) -print0
    )
    if (( ${#caches[@]} )); then
        printf 'Python caches to delete: %s\n' "${caches[@]}"
        if confirm 'Delete these local Python caches?'; then rm -rf -- "${caches[@]}"; fi
    fi
fi
if $logs; then
    # Operational logs only: root and runtime/. The .log files under paper_rl/ or research/ are evidence.
    logfiles=()
    while IFS= read -r -d '' f; do logfiles+=("$f"); done < <(
        find . -maxdepth 1 -type f \( -name '*.log' -o -name '*.log.[0-9]*' \) -print0
        if [[ -d runtime && ! -L runtime ]]; then
            find runtime -type f \( -name '*.log' -o -name '*.log.[0-9]*' \) -print0
        fi
    )
    if (( ${#logfiles[@]} )); then
        printf 'Logs to delete: %s\n' "${logfiles[@]}"
        if confirm 'Delete these local logs?'; then rm -f -- "${logfiles[@]}"; fi
    fi
fi
printf '%s\n' 'Cleanup finished. Configuration and loaded_model kept.'