#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SEQUENCE_ROOT=${SEQUENCE_ROOT:-/root/vepfs-data/chenjiaju/runs/SmartCritic/luna_webshop_gamma_alpha_20260902}
STATUS_FILE=${STATUS_FILE:-$SEQUENCE_ROOT/sequence.status}
LOG=${LOG:-$SEQUENCE_ROOT/sequence.log}
mkdir -p "$SEQUENCE_ROOT"

record_status() {
  local phase=$1
  local status=$2
  {
    echo "phase=$phase"
    echo "status=$status"
    echo "updated_at=$(date -Is)"
  } > "$STATUS_FILE"
}

run_phase() {
  local phase=$1
  shift
  record_status "$phase" running
  echo "[$(date -Is)] START $phase" | tee -a "$LOG"
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  phase_status=${PIPESTATUS[0]}
  set -e
  if (( phase_status != 0 )); then
    record_status "$phase" failed
    echo "[$(date -Is)] FAILED $phase exit=$phase_status" | tee -a "$LOG"
    exit "$phase_status"
  fi
  echo "[$(date -Is)] FINISH $phase" | tee -a "$LOG"
}

run_phase g0_train \
  env EXPERIMENT_ID=g0_alpha1 ALPHA=1.0 WANDB_RUN_ID=lzwsg0a1_0902 \
  "$SCRIPT_DIR/run_train.sh"

run_phase g0_eval500 \
  env EXPERIMENT_ID=g0_alpha1 ALPHA=1.0 WANDB_RUN_ID=lzwsg0a1e500_0902 \
  "$SCRIPT_DIR/run_eval500.sh"

run_phase g1_train \
  env EXPERIMENT_ID=g1_alpha3 ALPHA=3.0 WANDB_RUN_ID=lzwsg1a3_0902 \
  "$SCRIPT_DIR/run_train.sh"

run_phase g1_eval500 \
  env EXPERIMENT_ID=g1_alpha3 ALPHA=3.0 WANDB_RUN_ID=lzwsg1a3e500_0902 \
  "$SCRIPT_DIR/run_eval500.sh"

record_status complete finished
echo "[$(date -Is)] COMPLETE G0/G1 training and 500-case evaluations" | tee -a "$LOG"
