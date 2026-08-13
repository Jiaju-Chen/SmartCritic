#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_dualcritic2l_webshop_t128_v128_8gpu_seed0_v4_20260813}
export SESSION=${SESSION:-ppo_dualcritic2l_webshop_v4_20260813}
export WANDB_RUN_ID=${WANDB_RUN_ID:-dcw2lv4_0813}
export CRITIC_NUM_LAYERS=2
export TURN_CRITIC_NUM_LAYERS=2

exec bash "$(dirname "$0")/launch_tmux.sh"
