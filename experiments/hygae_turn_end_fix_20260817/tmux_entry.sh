#!/usr/bin/env bash
set -o pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_hygae_turnend_critic2l_diag10_20260817}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/hygae_turn_end_fix_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}

mkdir -p "$RUN_DIR"/logs
echo "[$(date)] start $RUN_NAME"
/home/dataset-local/cjj/RL/GiGPO_PVF_HYGAE_TURNEND/experiments/hygae_turn_end_fix_20260817/run_train.sh
status=$?
echo "[$(date)] finished $RUN_NAME status=$status"
exit $status
