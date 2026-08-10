#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/cjj/RL/GiGPO_PVF

SCRIPT_DIR=experiments/full140_checkpoint_audit_20260810
AUDIT_ROOT=${AUDIT_ROOT:-/home/dataset-local/cjj/RL/runs/full140_checkpoint_audit_20260810}
METHODS=(critic2l sao_skipobs)
STEPS=(25 50 75 100 125 150)

mkdir -p "${AUDIT_ROOT}"

for method in "${METHODS[@]}"; do
  for step in "${STEPS[@]}"; do
    printf '[%s] audit begin method=%s step=%s\n' "$(date)" "${method}" "${step}"
    AUDIT_ROOT="${AUDIT_ROOT}" bash "${SCRIPT_DIR}/run_one.sh" "${method}" "${step}"
    /home/dataset-local/cjj/RL/envs/gigpo-baselines/bin/python \
      "${SCRIPT_DIR}/collect_results.py"
    printf '[%s] audit complete method=%s step=%s\n' "$(date)" "${method}" "${step}"
    sleep 15
  done
done

printf '[%s] all full-140 checkpoint audits complete\n' "$(date)"

