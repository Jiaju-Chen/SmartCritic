#!/usr/bin/env bash
set -o pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_hygae_unified_critic2l_t128_v140_vb20_8gpu_seed0_pilot5_20260815}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/hygae_unified_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}

mkdir -p "$RUN_DIR"/logs
echo "[$(date)] start $RUN_NAME"
/home/dataset-local/cjj/RL/GiGPO_PVF_HYGAE/experiments/hygae_unified_alfworld_20260815/run_train.sh
status=$?
echo "[$(date)] finished $RUN_NAME status=$status"
exit $status

