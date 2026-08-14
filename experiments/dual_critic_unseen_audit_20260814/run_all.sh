#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF}
SCRIPT_DIR=${PROJECT_ROOT}/experiments/dual_critic_unseen_audit_20260814
ENV_PYTHON=${ENV_PYTHON:-/home/dataset-local/cjj/RL/envs/gigpo-baselines/bin/python}

cd "${PROJECT_ROOT}"

RUNS=(
  "dual2 best"
  "dual2 latest"
  "dual28 best"
  "dual28 latest"
)

for spec in "${RUNS[@]}"; do
  read -r method slot <<< "${spec}"
  printf '[%s] audit begin method=%s slot=%s\n' "$(date)" "${method}" "${slot}"
  bash "${SCRIPT_DIR}/run_one.sh" "${method}" "${slot}"
  "${ENV_PYTHON}" "${SCRIPT_DIR}/collect_results.py"
  printf '[%s] audit complete method=%s slot=%s\n' "$(date)" "${method}" "${slot}"
  sleep 15
done

printf '[%s] all dual-critic unseen audits complete\n' "$(date)"
