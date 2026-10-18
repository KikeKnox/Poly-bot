#!/usr/bin/env bash
# Installs poly-bot as a systemd service with automatic restart.
# Usage: sudo ./deploy/install-service.sh
set -euo pipefail

UNIT_NAME="poly-bot.service"
UNIT_DEST="/etc/systemd/system/$UNIT_NAME"

die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[install] $*"; }

[[ $EUID -eq 0 ]] || die "must be run with sudo (it writes to /etc/systemd/system)."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HRF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# The repo owner is who should run the bot: the .sqlite files, .env and config/
# are already theirs, and starting as root would leave files it then cannot touch.
HRF_USER="$(stat -c '%U' "$HRF_DIR")"
HRF_GROUP="$(stat -c '%G' "$HRF_DIR")"
[[ "$HRF_USER" != "root" ]] || die "the repo belongs to root — clone it as a normal user before installing the service."

[[ -x "$HRF_DIR/launch.sh" ]] || die "cannot find $HRF_DIR/launch.sh (or it is not executable)."
[[ -f "$HRF_DIR/.env" ]]      || die "cannot find $HRF_DIR/.env — the bot will not start without it."

info "repo     : $HRF_DIR"
info "user     : $HRF_USER:$HRF_GROUP"

sed -e "s|__HRF_DIR__|$HRF_DIR|g" \
    -e "s|__HRF_USER__|$HRF_USER|g" \
    -e "s|__HRF_GROUP__|$HRF_GROUP|g" \
    "$SCRIPT_DIR/$UNIT_NAME" > "$UNIT_DEST"
chmod 644 "$UNIT_DEST"

systemctl daemon-reload
systemctl enable "$UNIT_NAME"
info "unit installed and enabled at boot."

# `enable` does not start it: if the bot was already running by hand, the user
# should stop it before systemd launches a second process on the same DBs.
if pgrep -u "$HRF_USER" -f 'target/release/poly-bot' >/dev/null 2>&1; then
    info "NOTE: an poly-bot is already running by hand. Stop it and then:"
    info "  sudo systemctl start $UNIT_NAME"
else
    systemctl start "$UNIT_NAME"
    info "service started."
fi

echo
info "Check    : systemctl status $UNIT_NAME"
info "Logs      : journalctl -u $UNIT_NAME -f"
info "Restarts : systemctl show $UNIT_NAME -p NRestarts"
