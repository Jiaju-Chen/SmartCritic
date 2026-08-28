#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa}
ORCHESTRATOR_LOG=${ORCHESTRATOR_LOG:-$RUN_ROOT/stable_rerun_orchestrator_20260828.log}
mkdir -p "$(dirname "$ORCHESTRATOR_LOG")"

exec > >(tee -a "$ORCHESTRATOR_LOG") 2>&1
echo "[$(date)] starting stable SearchQA Luna reruns"
echo "[$(date)] policy: 2-layer critic first, then 28-layer critic; each run uses 200 steps"

cd "$PROJECT_ROOT"
RUN_NAME=ppo_qwen25_15b_luna_unified2l_residual_searchr1_stable_8gpu_seed0_200it_20260828 \
WANDB_RUN_ID=lunasearchr1_stable2l_0828 \
LOCAL_PORT=18000 \
bash experiments/luna_unified_critic_searchqa_stable_20260828/run_2l.sh

echo "[$(date)] 2-layer critic completed successfully; starting 28-layer critic"

RUN_NAME=ppo_qwen25_15b_luna_unified28l_residual_searchr1_stable_8gpu_seed0_200it_20260828 \
WANDB_RUN_ID=lunasearchr1_stable28l_0828 \
LOCAL_PORT=18001 \
bash experiments/luna_unified_critic_searchqa_stable_20260828/run_28l.sh

echo "[$(date)] both stable SearchQA Luna reruns completed successfully"
