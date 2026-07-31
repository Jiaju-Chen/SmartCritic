#!/usr/bin/env bash
set -o pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_critic2l_t128_v32_8gpu_seed0_20260731}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/truncated_qwen_critic_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}

mkdir -p "$RUN_DIR"/logs
echo "[$(date)] start $RUN_NAME"
/home/dataset-local/cjj/RL/GiGPO_PVF/experiments/truncated_qwen_critic_alfworld_20260731/run_train.sh
status=$?
echo "[$(date)] finished $RUN_NAME status=$status"
exit $status
