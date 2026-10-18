#!/usr/bin/env bash
# Paper-mode calibration run with all the configured strategies enabled.
#
# Saves the mode (paper/live) and each strategy's ON/OFF state before
# starting, forces paper and enables them, and on exit restores the
# original state in both .env and config/strategies/*.toml.
#
# Collects for N hours (default 48):
#   - paper fills and positions of all strategies (SQLite)
#   - rejected NC z-scores via debug log (field z in "NC: skip insufficient z" lines)
#   - post-fill maker markouts (markouts table)
#   - Prometheus hrf_* metric snapshots every 60 s
#
# Usage: ./calibrate.sh [hours]
# Which strategies get enabled: edit the CALIB_<STRATEGY> variables further
# down ("Strategies to enable in this calibration" block) before launching.
# Results go to calibration_<hours>/ — re-running with the same hours RESETS that folder.
# To stop early: Ctrl-C (the state is restored anyway).
#
# Persistent history across runs: on finish (normal or Ctrl-C) one row per
# strategy is appended to calibration_per_strategy/<strategy>.csv
# (start date, scheduled/actual duration, TOML parameters used in that
# run, and results: trades, WR, PnL, markout). It is created if missing;
# it is never overwritten, only appended to — this allows comparing runs and
# parameters over time instead of losing each calibration.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_DIR"

# Canonical forms (symlinks resolved) used as reference for the safety
# checks of safe_rm_rf_rundir — compare against the `readlink -f` of the
# path to delete, never against the raw variables.
PROJ_DIR_REAL="$(readlink -f -- "$PROJ_DIR" 2>/dev/null || echo "$PROJ_DIR")"
HOME_REAL="$(readlink -f -- "${HOME:-/nonexistent-home}" 2>/dev/null || echo "${HOME:-/nonexistent-home}")"

HORAS="${1:-48}"

die() { echo "[ERROR] $*" >&2; exit 1; }

# Applies the same mode suffix as `StateStore::resolve_path` (src/state.rs):
# strips an existing _paper/_live suffix from the stem and always appends _paper —
# calibrate.sh only runs in paper (see the HRF_MODE forcing below). Avoids
# duplicating that logic in Rust and in bash without keeping them in sync.
resolve_paper_db_path() {
    local raw="$1" dir base stem
    dir="$(dirname -- "$raw")"
    base="$(basename -- "$raw")"
    stem="${base%.sqlite}"
    stem="${stem%_paper}"
    stem="${stem%_live}"
    echo "${dir}/${stem}_paper.sqlite"
}

# An unvalidated `rm -rf "$RUN_DIR"` is dangerous: an empty HRF_CALIB_DIR,
# pointing to $HOME, to an ancestor of the repo, or to a "sacred" subdirectory
# that happens to start with $PROJ_DIR (.git, src, config, target) are all real
# cases in which "is it inside the project?" or "is the variable non-empty?"
# are NOT enough as a safety check. Three layers, all mandatory:
#   1. The final name must start with "calibration_" — inside or outside the
#      repo. An rm -rf never acts on anything that does not look like that.
#   2. Ownership marker: only a directory that ALREADY contains
#      ".hrf_calib_marker" (i.e. one this script created earlier) is deleted.
#      If it does not exist yet, there is nothing to delete (first run). If it
#      exists WITHOUT the marker it is assumed foreign and we abort — never guess.
#   3. Explicit, unconditional deny list (neither the $PROJ_DIR prefix nor the
#      "calibration_*" pattern bypasses it): /, $HOME, $PROJ_DIR itself, and
#      its .git/src/config/target subdirectories.
safe_rm_rf_rundir() {
    local dir="${1:-}" resolved base

    [[ -n "$dir" ]] || die "RUN_DIR is empty — aborting for safety, nothing is deleted"

    resolved="$(readlink -f -- "$dir" 2>/dev/null || true)"
    [[ -n "$resolved" ]] || die "could not resolve RUN_DIR ('$dir') — aborting for safety, nothing is deleted"

    # Layer 3: explicit deny list — evaluated FIRST and with no exceptions.
    case "$resolved" in
        "/"|"$HOME_REAL"|"$PROJ_DIR_REAL")
            die "RUN_DIR ('$resolved') is a protected directory — aborting for safety, nothing is deleted" ;;
        "$PROJ_DIR_REAL/.git"|"$PROJ_DIR_REAL/src"|"$PROJ_DIR_REAL/config"|"$PROJ_DIR_REAL/target")
            die "RUN_DIR ('$resolved') is a sacred repo subdirectory — aborting for safety, nothing is deleted" ;;
    esac

    # Layer 1: the name must unambiguously be a calibration directory.
    base="$(basename -- "$resolved")"
    [[ "$base" == calibration_* ]] \
        || die "RUN_DIR ('$resolved') does not have a 'calibration_*' name — aborting for safety, nothing is deleted"

    # Outside the repo only if the user explicitly asked for it via HRF_CALIB_DIR
    # (to keep artifacts out of the tree, as was already done with the DBs).
    case "$resolved" in
        "$PROJ_DIR_REAL"/*) ;;
        *)
            [[ -n "${HRF_CALIB_DIR:-}" ]] \
                || die "RUN_DIR ('$resolved') is outside the project and HRF_CALIB_DIR is not set — aborting"
            ;;
    esac

    # Layer 2: ownership marker. `[[ -f X ]]` in bash FOLLOWS symlinks — a
    # foreign directory with ".hrf_calib_marker -> /any/real/file" would pass
    # the check without this script ever having created it (finding from
    # review round 2). That is why `! -L` is also required: the marker
    # must be a truly REGULAR file, never a link.
    if [[ -e "$resolved" ]]; then
        local marker="$resolved/.hrf_calib_marker"
        [[ -f "$marker" && ! -L "$marker" ]] \
            || die "RUN_DIR ('$resolved') already exists but has no valid .hrf_calib_marker (regular file, not a symlink) — calibrate.sh did not create it, delete it by hand if it is safe to continue"
        rm -rf -- "$resolved"
    fi
    # If it does not exist, this is the first run with this RUN_DIR: nothing to delete.
}

# All known strategies — used to save/restore the original state of each
# TOML and to build CALIB_ON from the booleans.
STRATS=(arb directional near_certain near_expiry market_making parity_maker
        ladder_arb drift_box open_momentum theta_capture vol_surface post_close_sweep
        reward_farm)

# ── Strategies to enable in this calibration — edit here ──────────────────
# true = enabled during the run; false = stays OFF. The rest of the
# script needs no changes.
CALIB_ARB=${CALIB_ARB:-true}
CALIB_DIRECTIONAL=${CALIB_DIRECTIONAL:-false}
CALIB_NEAR_CERTAIN=${CALIB_NEAR_CERTAIN:-true}
CALIB_NEAR_EXPIRY=${CALIB_NEAR_EXPIRY:-true}
# MM is dead (markout −1966 bps@5s in all buckets, n=429, no +EV regime) — do not re-enable without a new thesis
CALIB_MARKET_MAKING=${CALIB_MARKET_MAKING:-false}
# PM is part of the calibration: it is the maker route to parity (fee 0, no
# race against the book) after discarding taker ARB in live. Requoting
# is already implemented; the README's −611 bps markout predates that code.
# Gate: markout ≥ 0 at 5 s + completed boxes.
CALIB_PARITY_MAKER=${CALIB_PARITY_MAKER:-true}
# ladder_arb and vol_surface OFF — parse_slug rejects the epoch of updown
# slugs as a strike, and without a strike they have no pairs to scan
# (0 pairs evaluated in the 4h run). Re-enable only if the universe
# again includes markets with a strike in the slug.
CALIB_LADDER_ARB=${CALIB_LADDER_ARB:-false}
# Frozen: taker in the mean zone, fee 10% of upside
CALIB_DRIFT_BOX=${CALIB_DRIFT_BOX:-false}
# Frozen: taker in the mean zone, fee 10% of upside
CALIB_OPEN_MOMENTUM=${CALIB_OPEN_MOMENTUM:-false}
CALIB_THETA_CAPTURE=${CALIB_THETA_CAPTURE:-true}
CALIB_VOL_SURFACE=${CALIB_VOL_SURFACE:-false}
# Off by default in calibration — asymmetric 1:100 risk; first validate the
# deterministic verdict with real data before leaving it active in long
# unsupervised runs. Can be enabled via env var for supervised runs:
# CALIB_POST_CLOSE_SWEEP=true ./calibrate.sh <hours>
CALIB_POST_CLOSE_SWEEP=${CALIB_POST_CLOSE_SWEEP:-false}
# reward_farm in scan-only mode (measure pool/competition; real posting has
# its own switch in the TOML and stays OFF unless explicitly decided).
CALIB_REWARD_FARM=${CALIB_REWARD_FARM:-true}

CALIB_ON=()
for s in "${STRATS[@]}"; do
    varname="CALIB_${s^^}"
    [[ "${!varname}" == true ]] && CALIB_ON+=("$s")
done

[[ -f .env ]]   || die ".env not found"
[[ -d config ]] || die "config/ directory not found"

# Load .env to read the current state (before any modification)
# shellcheck disable=SC1091
source .env 2>/dev/null || true

pgrep -f "target/release/poly-bot" >/dev/null \
    && die "an poly-bot instance is already running — stop it before launching the calibration"

# Metrics are served on the API port (axum), not on :9090.
PROM_PORT="${HRF_PROMETHEUS_PORT:-${HRF_API__PORT:-9090}}"

# Paper DB: the active database lives outside the repo (HRF_DB_PATH in .env)
# since the .sqlite files were externalized. `hrf_state_paper.sqlite` at the
# repo root is NO LONGER the active database — it is the accounting-evidence
# backup, and summary()/save_history() run it through `sqlite3`, which would
# checkpoint and merge its WAL if used by mistake. That is why there is NO
# fallback to that path: without HRF_DB_PATH we abort rather than risk touching the backup.
[[ -n "${HRF_DB_PATH:-}" ]] \
    || die "HRF_DB_PATH is not set in .env — the active paper database lives outside the repo; set it before calibrating (there is no fallback to the root hrf_state_paper.sqlite: it is the accounting-evidence backup, not the active database)"
DB="$(resolve_paper_db_path "$HRF_DB_PATH")"
[[ -f "$DB" ]] || die "paper DB not found at '$DB' — start the bot at least once in paper mode before calibrating"

RUN_DIR="${HRF_CALIB_DIR:-calibration_${HORAS}}"
safe_rm_rf_rundir "$RUN_DIR"
mkdir -p "$RUN_DIR"
# Ownership marker (layer 2 of safe_rm_rf_rundir): without it, the next run
# could not tell "I created this directory" from "someone put something
# else in RUN_DIR" and would always abort.
touch "$RUN_DIR/.hrf_calib_marker"
START_MS=$(( $(date +%s) * 1000 ))
echo "$START_MS" > "$RUN_DIR/start_ms"
echo "[calib] ${HORAS}h run — artifacts in $RUN_DIR/"

# ── Save original state ─────────────────────────────────────────────────────

# Mode: the HRF_MODE env var (from .env) takes precedence over config/general.toml
if [[ -n "${HRF_MODE:-}" ]]; then
    original_mode="$HRF_MODE"
else
    original_mode=$(grep -m1 '^mode[[:space:]]*=' config/general.toml \
        | sed -E 's/mode[[:space:]]*=[[:space:]]*//' | tr -d '"' | xargs 2>/dev/null \
        || echo "paper")
    original_mode="${original_mode:-paper}"
fi

# Total exposure and kill switch (config/general.toml → [risk])
# Only saved as a RECORD of what was there — the TOML is no longer overwritten
# (see "Computed max exposure" below: the env var
# HRF_RISK__MAX_TOTAL_EXPOSURE_USDC is used, so the file is never touched).
original_exposure=$(grep -m1 '^max_total_exposure_usdc' config/general.toml \
    | sed -E 's/max_total_exposure_usdc[[:space:]]*=[[:space:]]*//' | tr -d '"' | xargs 2>/dev/null \
    || echo "35")
original_kill_switch=$(grep -m1 '^paper_disable_kill_switch' config/general.toml \
    | sed -E 's/paper_disable_kill_switch[[:space:]]*=[[:space:]]*//' | xargs 2>/dev/null \
    || echo "true")

# Enabled state of each strategy (read from the TOMLs before modifying them)
declare -A original_enabled
for s in "${STRATS[@]}"; do
    toml="config/strategies/${s}.toml"
    val=$(grep -m1 '^enabled' "$toml" \
        | sed -E 's/enabled[[:space:]]*=[[:space:]]*//' | xargs 2>/dev/null \
        || echo "false")
    original_enabled[$s]="$val"
done

# Show saved state
echo "[calib] saved state:"
echo "  mode                    → $original_mode"
echo "  max_total_exposure_usdc → $original_exposure"
echo "  paper_disable_kill_switch → $original_kill_switch"
for s in "${STRATS[@]}"; do
    echo "  $s → ${original_enabled[$s]}"
done

# Save a readable snapshot (for post-run diagnostics)
{
    echo "mode=$original_mode"
    echo "max_total_exposure_usdc=$original_exposure"
    echo "paper_disable_kill_switch=$original_kill_switch"
    for s in "${STRATS[@]}"; do
        echo "${s}=${original_enabled[$s]}"
    done
} > "$RUN_DIR/original_state.txt"

# ── Config restore (defined BEFORE mutating anything) ──────────────────────

# Defined here, and not with the other helper functions, because the trap that
# invokes it is armed right after mutating the TOMLs: if the definition came
# later, an intermediate failure would fire the trap on a nonexistent function
# and the production config would be left with the calibration flags set.
RESTORED_CONFIG=0

restore_state() {
    # Idempotent: the EXIT trap, the ERR trap and the explicit call from
    # finalize() may fire it several times in the same run.
    if (( RESTORED_CONFIG )); then
        echo "[calib] state already restored — nothing to do"
        return 0
    fi
    echo "[calib] restoring original state..."
    # .env: restore from backup (includes the original mode)
    if [[ -f .env.calib_backup ]]; then
        cp .env.calib_backup .env
        rm -f .env.calib_backup
        echo "[calib]   mode → $original_mode (restored in .env)"
    fi
    # general.toml: kill switch only — max_total_exposure_usdc was never touched
    # (it was passed via the HRF_RISK__MAX_TOTAL_EXPOSURE_USDC env var), so the TOML
    # still holds the original value ($original_exposure, saved as a record).
    sed -i -E "s/^(paper_disable_kill_switch)[[:space:]]*=.*/\1 = ${original_kill_switch}/" config/general.toml
    echo "[calib]   max_total_exposure_usdc (TOML untouched) = $original_exposure"
    echo "[calib]   paper_disable_kill_switch → $original_kill_switch"
    # Strategies: restore enabled in each TOML
    for s in "${STRATS[@]}"; do
        toml="config/strategies/${s}.toml"
        val="${original_enabled[$s]}"
        sed -i -E "s/^(enabled)[[:space:]]*=.*/\1 = $val/" "$toml"
        echo "[calib]   $s → enabled = $val"
    done
    # Set the flag only HERE, after all steps have succeeded — it used to be the
    # first line of the function, so a mid-way failure (cp/sed on a full or
    # read-only FS) left the config PARTIALLY restored with the flag already at 1:
    # re-entry through the EXIT trap answered "already restored" and retried
    # nothing. Every step above is an absolute (not incremental) assignment, so
    # repeating the whole function on re-entry is safe (idempotent) —
    # *at-least-once* semantics instead of *at-most-once*.
    RESTORED_CONFIG=1
    echo "[calib] state restored"
}

# ── Force paper + enable the strategies ──────────────────────────────────────

# From here on the PRODUCTION config is mutated on disk. The trap covers ERR and
# EXIT as well as INT/TERM: without them, a failure of any command under
# `set -e` between this line and finalize() left the TOMLs with the calibration
# `enabled` values and `paper_disable_kill_switch = true` persisted. Further down it is
# replaced by the version that also stops the bot and produces the summary.
trap 'restore_state' EXIT ERR
trap 'echo "[calib] interrupted"; restore_state; exit 130' INT TERM

# launch.sh re-sources .env, so forcing paper requires modifying it.
# We back it up and restore it in finalize().
cp .env .env.calib_backup

if grep -q '^HRF_MODE=' .env; then
    sed -i 's/^HRF_MODE=.*/HRF_MODE=paper/' .env
else
    echo "" >> .env
    echo "# calibrate.sh — temporary override" >> .env
    echo "HRF_MODE=paper" >> .env
fi

if [[ "$original_mode" != "paper" ]]; then
    echo "[calib] mode '$original_mode' → paper (temporary for this run)"
fi

# .env overrides the exported RUST_LOG (launch.sh re-sources it): force it there
# so the NC (z-score) and new-strategy debug logs reach the log.
CALIB_RUST_LOG="info,poly_bot::strategy::near_certain=debug,poly_bot::strategy::theta_capture=debug,poly_bot::strategy::vol_surface=debug,poly_bot::strategy::post_close_sweep=debug,poly_bot::strategy::parity_maker=debug"
if grep -q '^RUST_LOG=' .env; then
    sed -i "s|^RUST_LOG=.*|RUST_LOG=${CALIB_RUST_LOG}|" .env
else
    echo "RUST_LOG=${CALIB_RUST_LOG}" >> .env
fi
echo "[calib] RUST_LOG forced in .env for the run (backup already saved)"

# CALIB_ON was built above from the CALIB_<STRATEGY> booleans.
for s in "${STRATS[@]}"; do
    toml="config/strategies/${s}.toml"
    val=false
    for on in "${CALIB_ON[@]}"; do [[ "$s" == "$on" ]] && val=true; done
    sed -i -E "s/^(enabled)[[:space:]]*=.*/\1 = $val/" "$toml"
done
echo "[calib] strategies enabled: ${CALIB_ON[*]} (rest OFF)"

sed -i -E 's/^(paper_disable_kill_switch)[[:space:]]*=.*/\1 = true/' config/general.toml
echo "[calib] paper_disable_kill_switch → true"

# ── Computed max exposure (2/3 of the paper wallet balance) ──────────────────
# Paper balance = paper_initial_deposit_usdc (config/general.toml → [risk]),
# read from the TOML, NOT hardcoded. This run's max exposure is
# 2/3 of that balance, leaving a 1/3 cushion against slippage/fee spikes.
paper_balance=$(grep -m1 '^paper_initial_deposit_usdc' config/general.toml \
    | sed -E 's/paper_initial_deposit_usdc[[:space:]]*=[[:space:]]*//' | tr -d '"' | xargs 2>/dev/null \
    || echo "1000")
calib_exposure=$(awk -v b="$paper_balance" 'BEGIN { printf "%.2f", b * 2 / 3 }')
echo "[calib] paper balance (paper_initial_deposit_usdc) = $paper_balance"
echo "[calib] computed max exposure (2/3 of balance) = $calib_exposure"

# Preferred method: env var HRF_RISK__MAX_TOTAL_EXPOSURE_USDC (verified: figment
# maps it to risk.max_total_exposure_usdc with "__" as the nesting separator).
# This way config/general.toml is NEVER touched for this key — nothing to restore.
export HRF_RISK__MAX_TOTAL_EXPOSURE_USDC="$calib_exposure"
echo "[calib] max_total_exposure_usdc → ${calib_exposure} (via HRF_RISK__MAX_TOTAL_EXPOSURE_USDC; config/general.toml unmodified)"

# Binance futures liquidations (@forceOrder) are OFF by default in the TOML
# (oracle.futures_liq_enabled = false) — calibration runs do need to capture
# liq_net_60s/liq_total_60s (pm_entry features) in order to evaluate them.
# figment maps HRF_ORACLE__FUTURES_LIQ_ENABLED to oracle.futures_liq_enabled
# ("__" as the nesting separator, same mechanism as
# HRF_RISK__MAX_TOTAL_EXPOSURE_USDC above). It only applies to this run via
# env var — the TOML is never touched, nothing to restore. Verified: the var
# survives launch.sh's re-source of .env because .env does not define that key.
export HRF_ORACLE__FUTURES_LIQ_ENABLED=true
echo "[calib] oracle.futures_liq_enabled → true (via HRF_ORACLE__FUTURES_LIQ_ENABLED; config/general.toml unmodified)"

# ── Bot ───────────────────────────────────────────────────────────────────────
# LOG_FORMAT=json: required so nc_zscores_summary can parse the log.
# Near-certain at debug to capture rejected z-scores; new strategies at debug
# for full calibration data.
export LOG_FORMAT=json
export RUST_LOG="info,\
poly_bot::strategy::near_certain=debug,\
poly_bot::strategy::theta_capture=debug,\
poly_bot::strategy::vol_surface=debug,\
poly_bot::strategy::post_close_sweep=debug,poly_bot::strategy::parity_maker=debug"

./launch.sh > "$RUN_DIR/bot.log" 2>&1 &
BOT_PID=$!
echo "$BOT_PID" > "$RUN_DIR/bot.pid"
echo "[calib] bot started (pid $BOT_PID)"
echo "[calib] logs: tail -f $RUN_DIR/bot.log"
echo "[calib] logs: tail -f /proc/$BOT_PID/fd/1"


# ── Helper functions ───────────────────────────────────────────────────────

snapshot() {
    {
        echo "# snapshot_ts $(date +%s)"
        curl -sf --max-time 10 "http://127.0.0.1:${PROM_PORT}/metrics" | grep '^hrf_' || true
    } >> "$RUN_DIR/metrics.prom"
}


nc_zscores_summary() {
    local log="$RUN_DIR/bot.log"
    [[ -f "$log" ]] || return 0
    python3 - "$log" <<'PYEOF'
import sys, json
zs = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        msg = obj.get("fields", {}).get("message", "")
        if msg == "NC: skip insufficient z":
            z = obj["fields"].get("z")
            if z is not None:
                zs.append(float(z))
    except Exception:
        pass
if not zs:
    print("  (no NC z-score skips in the log — is LOG_FORMAT=json set?)")
    sys.exit(0)
zs.sort()
n = len(zs)
buckets = [(0,0.5),(0.5,1.0),(1.0,1.5),(1.5,2.0),(2.0,999)]
labels  = ["<0.5","0.5-1.0","1.0-1.5","1.5-2.0",">=2.0"]
print(f"  n={n}  min={zs[0]:.3f}  p25={zs[n//4]:.3f}  p50={zs[n//2]:.3f}  p75={zs[3*n//4]:.3f}  max={zs[-1]:.3f}")
print("  distribution by bucket:")
for (lo, hi), lbl in zip(buckets, labels):
    cnt = sum(1 for z in zs if lo <= z < hi)
    pct = 100*cnt//n if n else 0
    bar = "#"*(pct//5)
    print(f"    {lbl:>7}  {cnt:>5}  {pct:>3}%  {bar}")
PYEOF
}

metrics_summary() {
    local prom="$RUN_DIR/metrics.prom"
    [[ -f "$prom" ]] || { echo "  (no metrics.prom file)"; return 0; }
    python3 - "$prom" "$HORAS" <<'PYEOF'
import sys, re

fname, hours = sys.argv[1], float(sys.argv[2])
snapshots = []
buf = []

for line in open(fname):
    if line.startswith('# snapshot_ts') and buf:
        snap = {}
        for l in buf:
            m = re.match(r'(hrf_\S+)\s+([\d.e+\-]+)', l.strip())
            if m:
                snap[m.group(1)] = float(m.group(2))
        if snap:
            snapshots.append(snap)
        buf = []
    buf.append(line)

if buf:
    snap = {}
    for l in buf:
        m = re.match(r'(hrf_\S+)\s+([\d.e+\-]+)', l.strip())
        if m:
            snap[m.group(1)] = float(m.group(2))
    if snap:
        snapshots.append(snap)

if len(snapshots) < 2:
    print("  (fewer than 2 snapshots — run too short or bot did not start)")
    sys.exit(0)

first, last = snapshots[0], snapshots[-1]

def d(k):
    return max(0, last.get(k, 0) - first.get(k, 0))

def pct(num, den):
    return f'{100*num/den:.1f}%' if den > 0 else '—'

def rph(n):
    return f'{n/hours:.1f}/h'

print(f"\n{'─'*60}")
print("PROMETHEUS METRICS — full-run deltas")
print(f"{'─'*60}")

arb_sig   = d('hrf_arb_signals_total')
arb_edge  = d('hrf_arb_skip_edge_total')
arb_spr   = d('hrf_arb_skip_spread_total')
arb_amt   = d('hrf_arb_skip_min_amount_total')
arb_oy    = d('hrf_arb_orphan_total{leg="yes"}')
arb_on    = d('hrf_arb_orphan_total{leg="no"}')
arb_tot   = arb_sig + arb_edge + arb_spr + arb_amt
print(f"\nARB  signals={arb_sig:.0f} ({rph(arb_sig)})  skip_edge={arb_edge:.0f} ({pct(arb_edge,arb_tot)})  skip_spread={arb_spr:.0f} ({pct(arb_spr,arb_tot)})  skip_amt={arb_amt:.0f}")
if arb_oy + arb_on > 0:
    print(f"ARB  ⚠ orphans: yes={arb_oy:.0f}  no={arb_on:.0f}  — see SQL §4b for per-leg P&L (if bail- > bail+: reduce no_leg_slippage)")
else:
    print(f"ARB  orphans: 0  ✓")

nc_sig  = d('hrf_nc_signals_total')
nc_bail = d('hrf_nc_bail_total')
nc_skz  = d('hrf_nc_skip_z_total')
nc_ski  = d('hrf_nc_skip_imbalance_total')
nc_rej  = d('hrf_nc_fok_rejected_total')
nc_tot  = nc_sig + nc_skz + nc_ski
print(f"\nNC  signals={nc_sig:.0f} ({rph(nc_sig)})  bail={pct(nc_bail,nc_sig)}")
print(f"NC  skip_z={nc_skz:.0f} ({pct(nc_skz,nc_tot)})  skip_imbalance={nc_ski:.0f} ({pct(nc_ski,nc_tot)})  fok_rejected={nc_rej:.0f}  [calibrate z_threshold]")

ne_sig  = d('hrf_ne_signals_total')
ne_bail = d('hrf_ne_bail_total')
ne_skd  = d('hrf_ne_skip_dist_total')
ne_rej  = d('hrf_ne_fok_rejected_total')
ne_tot  = ne_sig + ne_skd
print(f"\nNE  signals={ne_sig:.0f} ({rph(ne_sig)})  bail={pct(ne_bail,ne_sig)}")
print(f"NE  skip_dist={ne_skd:.0f} ({pct(ne_skd,ne_tot)})  fok_rejected={ne_rej:.0f}  [calibrate min_dist_pct]")

mm_q      = d('hrf_mm_quotes_total')
mm_vol    = d('hrf_mm_paused_total{reason="vol_spike"}')
mm_spread = d('hrf_mm_paused_total{reason="tight_spread"}')
mm_fv     = d('hrf_mm_skip_fv_extreme_total')
mm_yes    = d('hrf_mm_fill_total{side="yes"}')
mm_no     = d('hrf_mm_fill_total{side="no"}')
mm_fills  = mm_yes + mm_no
print(f"\nMM  quotes={mm_q:.0f} ({rph(mm_q)})  fills={mm_fills:.0f} (yes={mm_yes:.0f}/no={mm_no:.0f})  fill_rate={pct(mm_fills,mm_q)}")
print(f"MM  vol_pause={mm_vol:.0f}  spread_pause={mm_spread:.0f}  cancel_fv_extreme={mm_fv:.0f}")
print(f"MM  [GATE markout@5s → markouts table not implemented]")

db_sig   = d('hrf_driftbox_signals_total')
db_fill  = d('hrf_driftbox_fill_total')
db_bail  = d('hrf_driftbox_bail_total')
db_att   = d('hrf_driftbox_box_attempts_total')
db_comp  = d('hrf_driftbox_box_completed_total')
db_bfok  = d('hrf_driftbox_box_fok_rejected_total')
db_skc   = d('hrf_driftbox_skip_confidence_total')
db_skp   = d('hrf_driftbox_skip_price_total')
db_conv  = pct(db_fill, db_sig)
flag = ' ✓ [GATE ok]' if db_att > 0 and db_comp/db_att >= 0.5 else ' ⚠ [GATE low]' if db_att > 0 else ''
print(f"\nDriftBox  signals={db_sig:.0f}  fills={db_fill:.0f} ({db_conv})  bail={pct(db_bail,db_fill)}")
print(f"DriftBox  skip_conf={db_skc:.0f}  skip_price={db_skp:.0f}  [calibrate min_confidence / max_entry_price]")
print(f"DriftBox  box_attempts={db_att:.0f}  box_completed={db_comp:.0f} ({pct(db_comp,db_att)}){flag}  box_fok_rej={db_bfok:.0f}")

om_sig  = d('hrf_om_signals_total')
om_fill = d('hrf_om_fill_total')
om_bail = d('hrf_om_bail_total')
om_age  = d('hrf_om_skip_age_total')
om_conf = d('hrf_om_skip_confidence_total')
om_pr   = d('hrf_om_skip_price_total')
print(f"\nOpenMomentum  signals={om_sig:.0f} ({rph(om_sig)})  fills={om_fill:.0f} ({pct(om_fill,om_sig)})  bail={pct(om_bail,om_fill)}")
print(f"OpenMomentum  skip_age={om_age:.0f}  skip_conf={om_conf:.0f}  skip_price={om_pr:.0f}  [calibrate max_open_secs / min_confidence / max_entry_price]")

tc_placed = d('hrf_tc_bid_placed_total')
tc_fills  = d('hrf_tc_bid_fills_total')
tc_ow     = d('hrf_tc_cancel_total{reason="out_of_window"}')
tc_rev    = d('hrf_tc_cancel_total{reason="oracle_reversal"}')
tc_bail   = d('hrf_tc_bail_total')
tc_lp     = d('hrf_tc_loser_placed_total')
tc_ls     = d('hrf_tc_loser_sold_total')
fr        = pct(tc_fills, tc_placed)
flag = ' ✓ [GATE ok]' if tc_placed > 0 and tc_fills/tc_placed >= 0.30 else ' ⚠ [GATE <30%]' if tc_placed > 0 else ''
print(f"\nThetaCapture  placed={tc_placed:.0f}  filled={tc_fills:.0f} fill_rate={fr}{flag}")
print(f"ThetaCapture  cancel_ow={tc_ow:.0f}  cancel_rev={tc_rev:.0f}  bail={tc_bail:.0f}  loser={tc_lp:.0f}/sold={tc_ls:.0f}")
print(f"ThetaCapture  [GATE markout@5s → markouts table not implemented]")

vs_div  = d('hrf_vs_divergence_total')
vs_eval = d('hrf_vs_pairs_evaluated_total')
flag = ' ✓' if hours > 0 and vs_div/hours >= 1 else ' ⚠ low frequency'
print(f"\nVolSurface  evaluated_pairs={vs_eval:.0f}  divergences={vs_div:.0f} ({rph(vs_div)})  div_rate={pct(vs_div,vs_eval)}{flag}  [GATE ≥1/day]")

la_opp  = d('hrf_ladder_arb_opps_total')
la_eval = d('hrf_ladder_arb_pairs_evaluated_total')
flag = ' ✓' if hours > 0 and la_opp/hours >= 1 else ' ⚠ low frequency'
print(f"LadderArb  evaluated_pairs={la_eval:.0f}  opportunities={la_opp:.0f} ({rph(la_opp)})  hit_rate={pct(la_opp,la_eval)}{flag}  [GATE ≥1/day]")

dir_sig = d('hrf_dir_signals_total')
dir_fill = d('hrf_dir_fill_total')
dir_ske = d('hrf_dir_skip_no_entry_total')
dir_rej = d('hrf_dir_fok_rejected_total')
print(f"\nDIR  signals={dir_sig:.0f}  fills={dir_fill:.0f} ({pct(dir_fill,dir_sig)})  fok_rej={dir_rej:.0f}  skip_no_entry={dir_ske:.0f}")

pm_yes = d('hrf_pm_fill_total{side="yes"}')
pm_no  = d('hrf_pm_fill_total{side="no"}')
pm_q   = d('hrf_pm_quotes_total')
print(f"\nPM  quotes={pm_q:.0f} ({rph(pm_q)})  fills=yes:{pm_yes:.0f}/no:{pm_no:.0f} ({pct(pm_yes+pm_no,pm_q)})")

# reward_farm in scan-only mode — the gate is "there is a pool with a real rate
# and RF is not contaminated by fills" (real posting has its own switch in the
# TOML, OFF unless explicitly decided). Names/labels taken verbatim from reward_farm.rs.
rf_scan     = d('hrf_rf_scan_total')
rf_fill     = d('hrf_rf_fill_total')
rf_skip_mm  = d('hrf_rf_skip_max_markets_total')
rf_skip_bo  = d('hrf_rf_skip_busy_bracket_total')
rf_aband    = d('hrf_rf_abandon_extreme_total')
rf_pause    = d('hrf_rf_pause_vol_spike_total')
rf_ret_yes  = d('hrf_rf_toxic_withdrawal_total{side="yes"}')
rf_ret_no   = d('hrf_rf_toxic_withdrawal_total{side="no"}')
rf_skip_cap = d('hrf_rf_skip_cap_total_total')
rf_unw_hold = d('hrf_rf_unwind_total{reason="hold_fv"}')
rf_unw_dfv  = d('hrf_rf_unwind_total{reason="dump_fv"}')
rf_unw_dnfv = d('hrf_rf_unwind_total{reason="dump_no_fv"}')
rf_unw_rep  = d('hrf_rf_unwind_total{reason="repost_maker"}')
rf_proxy    = d('hrf_rf_reward_proxy_pusd_accrued')
flag = ' ✓ [GATE G0 ok — 0 fills]' if rf_fill == 0 else ' ⚠ [GATE G0: there were fills, check toxicity]'
print(f"\nRF  scans={rf_scan:.0f} ({rph(rf_scan)})  fills={rf_fill:.0f}{flag}")
print(f"RF  skip_max_markets={rf_skip_mm:.0f}  skip_busy_bracket={rf_skip_bo:.0f}  abandon_extreme={rf_aband:.0f}  pause_vol_spike={rf_pause:.0f}  skip_cap_total={rf_skip_cap:.0f}")
print(f"RF  toxic_withdrawal: yes={rf_ret_yes:.0f} no={rf_ret_no:.0f}")
print(f"RF  unwind: hold_fv={rf_unw_hold:.0f} dump_fv={rf_unw_dfv:.0f} dump_no_fv={rf_unw_dnfv:.0f} repost_maker={rf_unw_rep:.0f}")
print(f"RF  reward_proxy_pusd_accrued (delta)={rf_proxy:.4f}")
print(f"RF  [pool with vs without rate per market: not exposed as an aggregate Prometheus metric — see the daily_rate field in the structured log, target=rf_scan]")
PYEOF
}

save_history() {
    local out_dir="calibration_per_strategy"
    mkdir -p "$out_dir"
    local end_ms=$(( $(date +%s) * 1000 ))
    echo "[calib] saving per-strategy history in $out_dir/"
    # Runs BEFORE restore_state(): it needs to read the TOMLs exactly as they
    # were during the run (including hot toggles via the API), not the original
    # values restore_state() is about to write.
    python3 - "$DB" "$START_MS" "$end_ms" "$HORAS" "$out_dir" <<'PYEOF'
import sys, csv, os, sqlite3, tomllib, datetime

db_path, start_ms, end_ms, hours_sched, out_dir = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
)
real_hours = round((end_ms - start_ms) / 3_600_000, 3)
run_started_at = datetime.datetime.fromtimestamp(
    start_ms / 1000, tz=datetime.timezone.utc
).strftime('%Y-%m-%dT%H:%M:%SZ')

STRATS = ["arb", "directional", "near_certain", "near_expiry", "market_making",
          "parity_maker", "ladder_arb", "drift_box", "open_momentum",
          "theta_capture", "vol_surface", "post_close_sweep", "reward_farm"]

FIELDS = [
    "run_started_at_utc", "scheduled_hours", "real_hours", "enabled", "params",
    "n_positions", "won", "lost", "bails", "pending", "win_rate_pct",
    "n_buy_fills", "n_sell_fills", "volume_usdc",
    "n_closes", "pnl_total_usdc", "pnl_avg_usdc",
    "markout_buy_mk5_bps", "markout_sell_mk5_bps",
    "complete_pairs_pct",
]

con = sqlite3.connect(db_path)


def params_of(strategy):
    """All parameters of the strategy's TOML (except 'enabled', which is
    stored separately), as they actually stood during this run."""
    path = f"config/strategies/{strategy}.toml"
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return "", False
    section = data.get(strategy, {})
    enabled = bool(section.get("enabled", False))
    parts = [f"{k}={v}" for k, v in sorted(section.items()) if k != "enabled"]
    return ";".join(parts), enabled


def results_of(strategy):
    n, won, lost, bails, pending = con.execute("""
        SELECT count(*),
               sum(CASE WHEN outcome_won=1 THEN 1 ELSE 0 END),
               sum(CASE WHEN outcome_won=0 THEN 1 ELSE 0 END),
               sum(CASE WHEN bail_sold=1 THEN 1 ELSE 0 END),
               sum(CASE WHEN outcome_won IS NULL AND bail_sold=0 THEN 1 ELSE 0 END)
        FROM positions WHERE strategy=? AND opened_ts_ms>=?
    """, (strategy, start_ms)).fetchone()
    resolved_count = (won or 0) + (lost or 0)
    wr = round(100.0 * won / resolved_count, 1) if resolved_count else ""

    n_closes, pnl_total, pnl_avg = con.execute("""
        SELECT count(*), sum(pnl_um)/1e6, avg(pnl_um)/1e6
        FROM transactions
        WHERE strategy=? AND ts_ms>=? AND pnl_um IS NOT NULL AND kind IN ('SELL','REDEEM')
    """, (strategy, start_ms)).fetchone()

    n_buy, n_sell, volume = con.execute("""
        SELECT
            sum(CASE WHEN side='BUY'  THEN 1 ELSE 0 END),
            sum(CASE WHEN side='SELL' THEN 1 ELSE 0 END),
            sum(CASE WHEN side='BUY'  THEN size_um*price_um ELSE 0 END)/1e12
        FROM fills WHERE strategy=? AND ts_ms>=?
    """, (strategy, start_ms)).fetchone()

    mk_buy, mk_sell = con.execute("""
        SELECT
            AVG(CASE WHEN side='BUY'  THEN (mid_5s_um - fill_price_um)*10000.0/fill_price_um END),
            AVG(CASE WHEN side='SELL' THEN (fill_price_um - mid_5s_um)*10000.0/fill_price_um END)
        FROM markouts WHERE strategy=? AND ts_ms>=?
    """, (strategy, start_ms)).fetchone()

    pairs_pct = ""
    if strategy == "arb":
        total, complete = con.execute("""
            SELECT count(*), sum(CASE WHEN legs=2 THEN 1 ELSE 0 END)
            FROM (
                SELECT hedge_id, count(*) legs FROM fills
                WHERE ts_ms>=? AND strategy='arb' AND side='BUY' AND hedge_id IS NOT NULL
                GROUP BY hedge_id
            )
        """, (start_ms,)).fetchone()
        if total:
            pairs_pct = round(100.0 * complete / total, 1)

    return {
        "n_positions": n or 0, "won": won or 0, "lost": lost or 0,
        "bails": bails or 0, "pending": pending or 0, "win_rate_pct": wr,
        "n_buy_fills": n_buy or 0, "n_sell_fills": n_sell or 0,
        "volume_usdc": round(volume, 4) if volume else 0,
        "n_closes": n_closes or 0,
        "pnl_total_usdc": round(pnl_total, 4) if pnl_total is not None else 0,
        "pnl_avg_usdc": round(pnl_avg, 4) if pnl_avg is not None else 0,
        "markout_buy_mk5_bps": round(mk_buy, 1) if mk_buy is not None else "",
        "markout_sell_mk5_bps": round(mk_sell, 1) if mk_sell is not None else "",
        "complete_pairs_pct": pairs_pct,
    }


os.makedirs(out_dir, exist_ok=True)
for strategy in STRATS:
    params, enabled = params_of(strategy)
    row = {
        "run_started_at_utc": run_started_at,
        "scheduled_hours": hours_sched,
        "real_hours": real_hours,
        "enabled": enabled,
        "params": params,
        **results_of(strategy),
    }
    csv_path = os.path.join(out_dir, f"{strategy}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"  {strategy}.csv <- n={row['n_positions']} pnl=${row['pnl_total_usdc']}")

con.close()
PYEOF
}

summary() {
    echo "[calib] generating summary in $RUN_DIR/summary.txt"
    {
        echo "Calibration run — start ts_ms=$START_MS, end $(date -u +%FT%TZ)"
        echo "Original mode: $original_mode"
        echo "Filter tables by ts_ms/opened_ts_ms >= $START_MS to exclude earlier data."
        echo ""
        { echo ".param set :start_ms $START_MS"; cat tools/research_tools/calibration_analysis.sql; } \
            | sqlite3 "$DB" 2>/dev/null || echo "(SQL: error or empty DB)"
        echo ""
        echo "== NC: z-score distribution of skips (calibrate z_threshold) =="
        nc_zscores_summary
        echo ""
        echo "== PROMETHEUS METRICS =="
        metrics_summary
    } > "$RUN_DIR/summary.txt" 2>&1 || echo "[calib] warning: incomplete summary"
}

stop_processes() {
    if kill -0 "$BOT_PID" 2>/dev/null; then
        echo "[calib] stopping bot (SIGINT to $BOT_PID)..."
        kill -INT "$BOT_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$BOT_PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$BOT_PID" 2>/dev/null && kill -KILL "$BOT_PID" 2>/dev/null || true
    fi
}

FINALIZED=0

finalize() {
    # Idempotent like restore_state(): the EXIT trap and the explicit call at
    # the end of the script both invoke it.
    if (( FINALIZED )); then
        return 0
    fi
    FINALIZED=1
    # Disarming all four traps here used to leave the restore unprotected. With
    # `set -e`, a save_history failure (python3 missing, sqlite without the
    # expected-probability table) aborted the function BEFORE restore_state and no trap
    # was left to call it: the TOMLs kept the calibration `enabled` values
    # and `paper_disable_kill_switch = true`. Now only INT/TERM/ERR (which would
    # re-enter here) are disarmed and the EXIT trap is repointed to restore_state
    # until it has really run.
    trap - INT TERM ERR
    trap 'restore_state' EXIT
    snapshot || true
    stop_processes || true
    # Before restore_state(): it needs this run's TOMLs. Its failure must not
    # cost us the restore: the history is lost, the config is not.
    save_history || echo "[calib] ⚠ save_history failed — this run's history was not saved" >&2
    restore_state
    trap - EXIT   # config already restored; what remains is cosmetic
    summary
    gzip -f "$RUN_DIR/bot.log" || true
    echo "[calib] run finished — see $RUN_DIR/summary.txt"
}
# Replaces the early traps (the ones right after mutating the config):
# from here on the launched bot and the summary functions exist, so the full
# exit path is the one to run. ERR and EXIT included so that a failure under
# `set -e` does not skip the restore.
trap 'echo "[calib] interrupted"; finalize; exit 130' INT TERM
trap 'echo "[calib] ⚠ unexpected failure — restoring config"; finalize' ERR
trap 'finalize' EXIT

# ── Main loop ─────────────────────────────────────────────────────────────────
FIN=$(( $(date +%s) + HORAS * 3600 ))
while (( $(date +%s) < FIN )); do
    if ! kill -0 "$BOT_PID" 2>/dev/null; then
        echo "[calib] ⚠ the bot exited early — check $RUN_DIR/bot.log" >&2
        break
    fi
snapshot || true
    sleep 60
done

finalize
