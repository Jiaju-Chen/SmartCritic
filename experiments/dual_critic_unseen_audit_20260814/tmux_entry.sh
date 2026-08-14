#!/usr/bin/env bash
set -o pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF}
AUDIT_ROOT=${AUDIT_ROOT:-/home/dataset-local/cjj/RL/runs/dual_critic_unseen_audit_20260814}
SCRIPT_DIR=${PROJECT_ROOT}/experiments/dual_critic_unseen_audit_20260814

mkdir -p "${AUDIT_ROOT}"
echo "[$(date)] start dual-critic unseen audit"
bash "${SCRIPT_DIR}/run_all.sh"
status=$?
echo "[$(date)] finished dual-critic unseen audit status=${status}"
printf 'status=%s\n' "${status}" > "${AUDIT_ROOT}/queue_status.txt"
exit "${status}"
