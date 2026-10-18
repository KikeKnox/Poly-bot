#!/usr/bin/env bash
# ── Launcher isolation test under a contaminated environment ──────────────────
# Sets HRF_*/POLY_*/TELEGRAM_* variables to junk values (NONE real)
# —including RL_PAPER_CLEAN=1, the flag that used to give a bypass in an earlier version—
# and requires `launch_rl_paper.sh --check` to pass all of its internal asserts.
# --check only prints NAMES and verdicts, never values.
# Exit 0 = isolated; exit 1 = contamination reached the process.
set -euo pipefail
cd "$(dirname "$0")/.."

# Test contamination — all junk values, none real.
export RL_PAPER_CLEAN=1
export HRF_MODE=live
export HRF_API__BIND=0.0.0.0
export HRF_API__PORT=9999
export HRF_RISK__MAX_DRAWDOWN_PCT=999
export HRF_RL_PREDICTOR__PAPER_ONLY=false
export HRF_ARB__ENABLED=true
export HRF_MODEL_TOOL__ENABLED=false
export POLY_PRIVATE_KEY=test_junk_not_real
export POLY_API_KEY=test_junk_not_real
export POLY_API_SECRET=test_junk_not_real
export TELEGRAM_BOT_TOKEN=test_junk_not_real
export TELEGRAM_ALLOWED_IDS=test_junk_not_real
export HRF_DRY_RUN=false
export SECRET_EXTRA_ANYTHING=test_junk_not_real

printf '%s\n' '[test] Contaminated environment (junk, no real secrets). Running --check…'
if ./launch_rl_paper.sh --check; then
    printf '%s\n' '[test] ISOLATION: OK — no contaminated value survives the re-exec.'
else
    printf '%s\n' '[test] ISOLATION: FAILED — see FAILs above.' >&2
    exit 1
fi
