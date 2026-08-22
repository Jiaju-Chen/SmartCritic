#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=${RUN_NAME:-ppo_luna_directmix2h_webshop_smoke1_20260822}
export SESSION=${SESSION:-ppo_luna_webshop_directmix_smoke1_20260822}
export WANDB_RUN_ID=${WANDB_RUN_ID:-lwadirectsmoke_0822}
export WANDB_MODE=${WANDB_MODE:-offline}
export ADVANTAGE_COMPOSITION=direct
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=16
export VAL_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=16
export TOTAL_EPOCHS=1
export TEST_FREQ=1
export SAVE_FREQ=1
export VAL_BEFORE_TRAIN=False

exec bash "$(dirname "$0")/launch_tmux.sh"
