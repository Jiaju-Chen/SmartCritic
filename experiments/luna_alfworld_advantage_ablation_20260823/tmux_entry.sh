#!/usr/bin/env bash
set -uo pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_directmix2h_alfworld_t128_v140_vb20_8gpu_seed0_20260823}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_alfworld_advantage_ablation}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
mkdir -p "$RUN_DIR/logs"
LOG_FILE=${LOG_FILE:-$RUN_DIR/logs/train.tmux.log}

echo "[$(date)] starting $RUN_NAME" | tee -a "$LOG_FILE"
bash experiments/luna_alfworld_advantage_ablation_20260823/run_train.sh 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
echo "[$(date)] finished $RUN_NAME status=$status" | tee -a "$LOG_FILE"
exit "$status"
