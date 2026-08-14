#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF}
AUDIT_ROOT=${AUDIT_ROOT:-/home/dataset-local/cjj/RL/runs/dual_critic_unseen_audit_20260814}
SESSION=${SESSION:-dual_critic_unseen_audit_20260814}
SCRIPT_DIR=${PROJECT_ROOT}/experiments/dual_critic_unseen_audit_20260814

mkdir -p "${AUDIT_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "bash '${SCRIPT_DIR}/tmux_entry.sh' 2>&1 | tee -a '${AUDIT_ROOT}/audit.master.log'"

echo "started tmux session: ${SESSION}"
echo "master log: ${AUDIT_ROOT}/audit.master.log"
