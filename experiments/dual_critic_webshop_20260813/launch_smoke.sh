#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_dualcritic2l_webshop_smoke_20260813}
export SESSION=${SESSION:-ppo_dualcritic2l_webshop_smoke_20260813}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_RUN_ID=${WANDB_RUN_ID:-dcw2lsmoke0813}
export CRITIC_NUM_LAYERS=2
export TURN_CRITIC_NUM_LAYERS=2
export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE=8
export VAL_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export TOTAL_EPOCHS=1
export TEST_FREQ=1
export SAVE_FREQ=-1
export VAL_BEFORE_TRAIN=False
export NUM_CPUS_PER_ENV_WORKER=0.25
export RAY_NUM_CPUS=32

exec bash "$(dirname "$0")/launch_tmux.sh"
