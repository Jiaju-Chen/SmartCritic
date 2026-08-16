#!/usr/bin/env bash
set -o pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2h_t128_v140_vb20_8gpu_seed0_20260816}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}

mkdir -p "$RUN_DIR"/logs
echo "[$(date)] start $RUN_NAME"
/home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_UNIFIED/experiments/luna_unified_critic_alfworld_20260816/run_train.sh
status=$?
echo "[$(date)] finished $RUN_NAME status=$status"
exit $status
