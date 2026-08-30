#!/usr/bin/env bash

EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$EXPERIMENT_DIR/../.." && pwd)}
export ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
export RUN_NAME=${RUN_NAME:-luna_unified_searchqa_2l_residual_whiten_8gpu_t256_v512_200step_20260830}
export RUN_ROOT=/home/dataset-local/cjj/RL/runs/luna_unified_searchqa
export RUN_DIR="$RUN_ROOT/$RUN_NAME"
export CKPT_DIR="/home/dataset-local/cjj/RL/checkpoints/luna_unified_searchqa/$RUN_NAME"
export WANDB_RUN_ID=${WANDB_RUN_ID:-lunauni8white0830}
export WANDB_ENTITY=cjj01-ustc
export WANDB_MODE=online
export WANDB_RESUME=never
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOTAL_TRAINING_STEPS=200
export TEST_FREQ=50
export SAVE_FREQ=50
export SEARCH_URL=http://127.0.0.1:18003/retrieve
export RAY_TMP_ROOT=/dev/shm/scq-luna8-white-ray
export FAST_TMP_ROOT=/dev/shm/scq-luna8-white-tmp
export VAL_FILE="$RUN_ROOT/val_subsets/searchqa_test_stratified512_seed0.parquet"
export SESSION_NAME=${SESSION_NAME:-searchqa_luna2l_whiten_20260830}
