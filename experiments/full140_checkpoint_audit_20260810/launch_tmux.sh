#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-full140_checkpoint_audit_20260810}
AUDIT_ROOT=${AUDIT_ROOT:-/home/dataset-local/cjj/RL/runs/full140_checkpoint_audit_20260810}
MASTER_LOG=${AUDIT_ROOT}/audit.master.log

mkdir -p "${AUDIT_ROOT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "cd /home/dataset-local/cjj/RL/GiGPO_PVF && AUDIT_ROOT='${AUDIT_ROOT}' bash experiments/full140_checkpoint_audit_20260810/run_all.sh 2>&1 | tee -a '${MASTER_LOG}'"

echo "session=${SESSION}"
echo "master_log=${MASTER_LOG}"
echo "results_csv=${AUDIT_ROOT}/results.csv"
echo "results_json=${AUDIT_ROOT}/results.json"

