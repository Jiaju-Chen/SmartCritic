#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SEQUENCE_ROOT=${SEQUENCE_ROOT:-/root/vepfs-data/chenjiaju/runs/SmartCritic/luna_webshop_adaptive_rms_20260903}
STATUS_FILE=${STATUS_FILE:-$SEQUENCE_ROOT/sequence.status}
LOG=${LOG:-$SEQUENCE_ROOT/sequence.log}
mkdir -p "$SEQUENCE_ROOT"

run_phase() {
  local phase=$1
  shift
  printf 'phase=%s\nstatus=running\nupdated_at=%s\n' \
    "$phase" "$(date -Is)" > "$STATUS_FILE"
  echo "[$(date -Is)] START $phase" | tee -a "$LOG"
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  phase_status=${PIPESTATUS[0]}
  set -e
  if (( phase_status != 0 )); then
    printf 'phase=%s\nstatus=failed\nupdated_at=%s\n' \
      "$phase" "$(date -Is)" > "$STATUS_FILE"
    exit "$phase_status"
  fi
  echo "[$(date -Is)] FINISH $phase" | tee -a "$LOG"
}

run_phase train "$SCRIPT_DIR/run_train.sh"
run_phase eval500 "$SCRIPT_DIR/run_eval500.sh"

printf 'phase=complete\nstatus=finished\nupdated_at=%s\n' \
  "$(date -Is)" > "$STATUS_FILE"
