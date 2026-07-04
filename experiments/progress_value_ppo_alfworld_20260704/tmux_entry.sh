#!/usr/bin/env bash
set -o pipefail

RUN_NAME=${RUN_NAME:-pvf_ppo_qwen25_15b_seed0_t16_g8_total128_val140_8gpu_envcpu02_ray96_20260704}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/progress_value_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}

mkdir -p "$RUN_DIR"/logs
echo "[$(date)] start $RUN_NAME"
/home/dataset-local/cjj/RL/GiGPO_PVF/experiments/progress_value_ppo_alfworld_20260704/run_train.sh
status=$?
echo "[$(date)] finished $RUN_NAME status=$status"
exit $status
