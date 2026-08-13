#!/usr/bin/env bash
set -o pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF_WebShop}
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_dualcritic2l_webshop_t128_v128_8gpu_seed0_20260813}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/dual_critic_webshop}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}

mkdir -p "$RUN_DIR/logs"
echo "[$(date)] start $RUN_NAME"
"$PROJECT_ROOT/experiments/dual_critic_webshop_20260813/run_train.sh"
status=$?
echo "[$(date)] finished $RUN_NAME status=$status"
exit "$status"
